from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from src.stage2_binary.dataset import (
    binary_stage2_collate_fn,
    binarize_raw_pedals,
    decode_joint_targets,
    encode_joint_targets,
    make_masked_binary_sample,
)
from src.stage2_binary.model import (
    IndependentBinaryPedalModel,
    JointBinaryPedalModel,
)
from src.stage2_binary.training import build_binary_optimizer


class StubEncoder(nn.Module):
    def __init__(self, hidden_size: int = 12) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size, dropout_rate=0.0)
        self.embedding = nn.Embedding(6000, hidden_size)
        self.projection = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> SimpleNamespace:
        del attention_mask
        batch_size, flat_length = input_ids.shape
        embedded = self.embedding(input_ids).view(
            batch_size, flat_length // 8, 8, self.config.hidden_size
        )
        return SimpleNamespace(
            last_hidden_state=self.projection(embedded.mean(dim=2))
        )


class BinaryStage2Tests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)

    @staticmethod
    def inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ids = torch.randint(0, 5000, (2, 3 * 8))
        attention_mask = torch.ones_like(input_ids)
        note_mask = torch.ones((2, 3), dtype=torch.bool)
        return input_ids, attention_mask, note_mask

    def test_threshold_boundary_63_off_64_on(self) -> None:
        raw = torch.tensor([[0, 63, 64, 127]])
        self.assertTrue(
            torch.equal(binarize_raw_pedals(raw), torch.tensor([[0, 0, 1, 1]]))
        )

    def test_exhaustive_joint_round_trip(self) -> None:
        joint = torch.arange(16)
        bits = decode_joint_targets(joint)
        self.assertEqual(tuple(bits.shape), (16, 4))
        self.assertTrue(torch.all((bits == 0) | (bits == 1)))
        self.assertTrue(torch.equal(encode_joint_targets(bits), joint))

    def test_masked_sample_preserves_only_non_pedal_inputs(self) -> None:
        tokens = np.array(
            [
                [10, 20, 30, 40, 5261, 5324, 5325, 5388],
                [11, 21, 31, 41, 5388, 5261, 5325, 5324],
            ],
            dtype=np.int64,
        )
        sample = make_masked_binary_sample(tokens, 0, 2)
        masked = sample["input_ids"].view(2, 8)
        self.assertTrue(torch.equal(masked[:, :4], torch.from_numpy(tokens[:, :4])))
        self.assertTrue(
            torch.equal(masked[:, 4:], torch.ones((2, 4), dtype=torch.long))
        )
        batch = binary_stage2_collate_fn([sample])
        self.assertNotIn("pedal_targets", batch)
        self.assertEqual(tuple(batch["binary_targets"].shape), (1, 2, 4))
        self.assertEqual(tuple(batch["joint_targets"].shape), (1, 2))
        self.assertTrue(
            torch.all(
                (batch["binary_targets"] == 0)
                | (batch["binary_targets"] == 1)
            )
        )
        self.assertTrue(
            torch.all(
                (batch["joint_targets"] >= 0)
                & (batch["joint_targets"] <= 15)
            )
        )

    def test_independent_shape_values_and_heads(self) -> None:
        model = IndependentBinaryPedalModel(StubEncoder(), dropout=0.0)
        input_ids, attention_mask, note_mask = self.inputs()
        targets = torch.randint(0, 2, (2, 3, 4))
        output = model(input_ids, attention_mask, targets, note_mask)
        self.assertEqual(tuple(output.logits.shape), (2, 3, 4, 2))
        self.assertTrue(torch.all((targets == 0) | (targets == 1)))
        self.assertEqual(len(model.classification_heads), 4)
        self.assertEqual(len({id(head) for head in model.classification_heads}), 4)

    def test_joint_shape_and_values(self) -> None:
        model = JointBinaryPedalModel(StubEncoder(), dropout=0.0)
        input_ids, attention_mask, note_mask = self.inputs()
        bits = torch.randint(0, 2, (2, 3, 4))
        targets = encode_joint_targets(bits)
        output = model(input_ids, attention_mask, targets, note_mask)
        self.assertEqual(tuple(output.logits.shape), (2, 3, 16))
        self.assertTrue(torch.all((targets >= 0) & (targets <= 15)))

    def test_padding_is_ignored_by_both_losses(self) -> None:
        independent_logits = torch.randn(1, 2, 4, 2)
        independent_targets = torch.randint(0, 2, (1, 2, 4))
        independent_targets[:, 1] = -100
        independent_changed = independent_logits.clone()
        independent_changed[:, 1] = (
            torch.randn_like(independent_changed[:, 1]) * 1e5
        )
        self.assertTrue(
            torch.allclose(
                IndependentBinaryPedalModel.compute_loss(
                    independent_logits, independent_targets
                ),
                IndependentBinaryPedalModel.compute_loss(
                    independent_changed, independent_targets
                ),
            )
        )

        joint_logits = torch.randn(1, 2, 16)
        joint_targets = torch.randint(0, 16, (1, 2))
        joint_targets[:, 1] = -100
        joint_changed = joint_logits.clone()
        joint_changed[:, 1] = torch.randn_like(joint_changed[:, 1]) * 1e5
        self.assertTrue(
            torch.allclose(
                JointBinaryPedalModel.compute_loss(joint_logits, joint_targets),
                JointBinaryPedalModel.compute_loss(joint_changed, joint_targets),
            )
        )

    def test_both_models_have_finite_loss_gradient_and_trainable_encoder(self) -> None:
        input_ids, attention_mask, note_mask = self.inputs()
        bits = torch.randint(0, 2, (2, 3, 4))
        cases = (
            (IndependentBinaryPedalModel(StubEncoder(), dropout=0.0), bits),
            (
                JointBinaryPedalModel(StubEncoder(), dropout=0.0),
                encode_joint_targets(bits),
            ),
        )
        for model, targets in cases:
            with self.subTest(model=model.__class__.__name__):
                output = model(
                    input_ids,
                    attention_mask,
                    note_mask=note_mask,
                    **{model.target_name: targets},
                )
                self.assertIsNotNone(output.loss)
                self.assertTrue(math.isfinite(float(output.loss.item())))
                output.loss.backward()
                self.assertTrue(
                    all(p.requires_grad for p in model.encoder.parameters())
                )
                encoder_gradients = [p.grad for p in model.encoder.parameters()]
                self.assertTrue(any(g is not None for g in encoder_gradients))
                self.assertTrue(
                    all(
                        g is None or bool(torch.isfinite(g).all())
                        for g in encoder_gradients
                    )
                )
                head_gradients = [
                    p.grad for p in model.prediction_head_parameters()
                ]
                self.assertTrue(all(g is not None for g in head_gradients))
                self.assertTrue(
                    all(bool(torch.isfinite(g).all()) for g in head_gradients)
                )

    def test_forward_does_not_mutate_input_and_outputs_only_pedal_targets(self) -> None:
        input_ids, attention_mask, _ = self.inputs()
        original = input_ids.clone()
        outputs = (
            IndependentBinaryPedalModel(StubEncoder(), dropout=0.0)(
                input_ids, attention_mask
            ),
            JointBinaryPedalModel(StubEncoder(), dropout=0.0)(
                input_ids, attention_mask
            ),
        )
        self.assertTrue(torch.equal(input_ids, original))
        self.assertEqual(outputs[0].logits.shape[-2:], (4, 2))
        self.assertEqual(outputs[1].logits.shape[-1], 16)
        self.assertNotEqual(outputs[0].logits.shape[2], 8)

    def test_binary_collator_padding_matches_both_target_representations(self) -> None:
        base = np.array(
            [[10, 20, 30, 40, 5261, 5324, 5325, 5388]] * 2,
            dtype=np.int64,
        )
        batch = binary_stage2_collate_fn(
            [
                make_masked_binary_sample(base, 0, 2),
                make_masked_binary_sample(base, 0, 1),
            ]
        )
        self.assertEqual(tuple(batch["binary_targets"].shape), (2, 2, 4))
        self.assertEqual(tuple(batch["joint_targets"].shape), (2, 2))
        self.assertTrue(torch.all(batch["binary_targets"][1, 1] == -100))
        self.assertEqual(int(batch["joint_targets"][1, 1]), -100)
        self.assertFalse(bool(batch["note_mask"][1, 1]))

    def test_optimizer_groups_cover_all_trainable_parameters(self) -> None:
        for model in (
            IndependentBinaryPedalModel(StubEncoder()),
            JointBinaryPedalModel(StubEncoder()),
        ):
            optimizer = build_binary_optimizer(model)
            grouped = [
                id(parameter)
                for group in optimizer.param_groups
                for parameter in group["params"]
            ]
            trainable = [id(p) for p in model.parameters() if p.requires_grad]
            self.assertEqual(len(grouped), len(set(grouped)))
            self.assertEqual(set(grouped), set(trainable))


if __name__ == "__main__":
    unittest.main()
