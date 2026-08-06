from __future__ import annotations

import hashlib
import math
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from src.stage2_encoder_only.five_class import (
    CLASS_BOUNDS,
    CLASS_NAMES,
    REPRESENTATIVES,
    FiveClassPedalEncoderModel,
    average_five_class_logits,
    classify_pedal_values,
    decode_five_classes,
    five_class_collate_fn,
)
from src.stage2_encoder_only.training import build_optimizer, set_deterministic_seed, train_step


class TinyEncoder(nn.Module):
    def __init__(self, hidden_size: int = 12) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size, dropout_rate=0.0)
        self.embedding = nn.Embedding(6000, hidden_size)
        self.projection = nn.Linear(hidden_size, hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        del attention_mask
        batch, length = input_ids.shape
        hidden = self.embedding(input_ids).view(batch, length // 8, 8, -1).mean(2)
        return SimpleNamespace(last_hidden_state=self.projection(hidden))


def model_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().numpy().tobytes())
    return digest.hexdigest()


class FiveClassCoreTests(unittest.TestCase):
    def setUp(self) -> None:
        set_deterministic_seed(20260710)

    @staticmethod
    def model() -> FiveClassPedalEncoderModel:
        return FiveClassPedalEncoderModel(TinyEncoder(), dropout=0.0)

    @staticmethod
    def batch() -> dict[str, torch.Tensor]:
        ids = torch.randint(0, 5000, (2, 3 * 8))
        return {
            "input_ids": ids,
            "token_attention_mask": torch.ones_like(ids),
            "pedal_targets": torch.randint(0, 5, (2, 3, 4)),
            "note_mask": torch.ones((2, 3), dtype=torch.bool),
        }

    def test_boundary_mapping(self) -> None:
        values = np.asarray([[0, 1, 63, 64], [95, 96, 126, 127]], dtype=np.int64)
        np.testing.assert_array_equal(
            classify_pedal_values(values),
            [[0, 1, 1, 2], [2, 3, 3, 4]],
        )
        self.assertEqual(CLASS_NAMES, ("ZERO", "LOW", "MID", "HIGH", "FULL"))
        self.assertEqual(CLASS_BOUNDS, ((0, 0), (1, 63), (64, 95), (96, 126), (127, 127)))

    def test_representative_decoding(self) -> None:
        classes = np.asarray([[0, 1, 2, 3], [4, 0, 1, 2]], dtype=np.int64)
        np.testing.assert_array_equal(
            decode_five_classes(classes),
            [[0, 32, 80, 111], [127, 0, 32, 80]],
        )
        np.testing.assert_array_equal(REPRESENTATIVES, [0, 32, 80, 111, 127])

    def test_invalid_target_range_dtype_and_shape(self) -> None:
        for values in (
            np.asarray([[-1, 0, 1, 2]], dtype=np.int64),
            np.asarray([[0, 1, 2, 128]], dtype=np.int64),
        ):
            with self.subTest(values=values.tolist()), self.assertRaises(ValueError):
                classify_pedal_values(values)
        with self.assertRaises(TypeError):
            classify_pedal_values(np.zeros((2, 4), dtype=np.float32))
        with self.assertRaises(ValueError):
            classify_pedal_values(np.zeros((2, 3), dtype=np.int64))

    def test_output_shape_and_independent_heads(self) -> None:
        model = self.model()
        batch = self.batch()
        output = model(batch["input_ids"], batch["token_attention_mask"])
        self.assertEqual(tuple(output.logits.shape), (2, 3, 4, 5))
        self.assertEqual(len({id(head) for head in model.classification_heads}), 4)
        self.assertEqual(len({head.weight.data_ptr() for head in model.classification_heads}), 4)
        self.assertTrue(all(head.out_features == 5 for head in model.classification_heads))

    def test_padding_is_excluded_from_loss(self) -> None:
        logits = torch.randn(1, 2, 4, 5)
        targets = torch.randint(0, 5, (1, 2, 4))
        targets[:, 1] = -100
        changed = logits.clone()
        changed[:, 1] = torch.randn_like(changed[:, 1]) * 10000
        first = FiveClassPedalEncoderModel.compute_loss(logits, targets)
        second = FiveClassPedalEncoderModel.compute_loss(changed, targets)
        self.assertTrue(torch.equal(first, second))

    def test_all_heads_and_encoder_receive_finite_nonzero_gradients(self) -> None:
        model = self.model()
        output = model(**{
            "input_ids": self.batch()["input_ids"],
            "token_attention_mask": self.batch()["token_attention_mask"],
        })
        del output
        batch = self.batch()
        loss = model(**batch).loss
        assert loss is not None
        loss.backward()
        self.assertTrue(any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in model.encoder.parameters()))
        for head in model.classification_heads:
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()))
            self.assertTrue(any(float(p.grad.abs().sum()) > 0 for p in head.parameters()))

    def test_training_step_updates_all_heads_and_active_encoder(self) -> None:
        model = self.model()
        optimizer = build_optimizer(model, encoder_lr=1e-2, head_lr=1e-2, weight_decay=0.0)
        before_encoder = model.encoder.projection.weight.detach().clone()
        before_heads = [head.weight.detach().clone() for head in model.classification_heads]
        metrics = train_step(model, self.batch(), optimizer, amp_enabled=False)
        self.assertTrue(math.isfinite(float(metrics["loss"])))
        self.assertFalse(torch.equal(before_encoder, model.encoder.projection.weight))
        self.assertTrue(all(not torch.equal(before, head.weight) for before, head in zip(before_heads, model.classification_heads)))

    def test_overlap_averages_logits_before_argmax_and_covers_edges(self) -> None:
        first = np.zeros((3, 4, 5), dtype=np.float32)
        second = np.zeros((3, 4, 5), dtype=np.float32)
        first[..., 0] = 10
        first[..., 2] = 9
        second[..., 1] = 10
        second[..., 2] = 9
        averaged, counts = average_five_class_logits(4, [(0, first), (1, second)])
        np.testing.assert_array_equal(counts, [1, 2, 2, 1])
        self.assertTrue(np.all(averaged[1:3].argmax(-1) == 2))
        self.assertTrue(np.all(first.argmax(-1) == 0))
        self.assertTrue(np.all(second.argmax(-1) == 1))

    def test_overlap_rejects_duplicate_or_uncovered_windows(self) -> None:
        logits = np.zeros((2, 4, 5), dtype=np.float32)
        with self.assertRaises(ValueError):
            average_five_class_logits(3, [(0, logits), (0, logits)])
        with self.assertRaises(RuntimeError):
            average_five_class_logits(4, [(1, logits)])

    def test_deterministic_model_initialization_hash(self) -> None:
        set_deterministic_seed(20260710)
        first = self.model()
        first_hash = model_hash(first)
        set_deterministic_seed(20260710)
        second = self.model()
        self.assertEqual(first_hash, model_hash(second))

    def test_collate_preserves_non_pedal_tokens_bit_for_bit(self) -> None:
        original = torch.tensor(
            [[5, 261, 133, 262, 5261, 5324, 5325, 5388],
             [6, 262, 134, 263, 5262, 5263, 5357, 5387]],
            dtype=torch.long,
        )
        sample = {
            "input_ids": original.reshape(-1).clone(),
            "pedal_targets": original[:, 4:] - 5261,
            "note_mask": torch.ones(2, dtype=torch.bool),
            "metadata": {"split": "train"},
        }
        batch = five_class_collate_fn([sample])
        np.testing.assert_array_equal(
            batch["input_ids"][0].reshape(2, 8)[:, :4].numpy(),
            original[:, :4].numpy(),
        )
        np.testing.assert_array_equal(batch["pedal_targets"].numpy(), [[[0, 1, 2, 4], [1, 1, 3, 3]]])

    def test_representatives_are_fixed_constants_not_statistics(self) -> None:
        source = np.asarray([[1, 63, 64, 95], [96, 126, 0, 127]], dtype=np.int64)
        classify_pedal_values(source)
        np.testing.assert_array_equal(REPRESENTATIVES, [0, 32, 80, 111, 127])


if __name__ == "__main__":
    unittest.main()
