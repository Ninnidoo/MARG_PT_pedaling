"""Tests for the squared-CDF ordinal Stage 2 objective."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from src.stage2_encoder_only.model import Stage2PedalEncoderModel
from src.stage2_encoder_only.ordinal_loss import (
    ordinal_target_cdf,
    squared_cdf_ordinal_loss,
)
from src.stage2_encoder_only.train import atomic_torch_save, build_best_checkpoint
from src.stage2_encoder_only.train_ordinal import (
    LOSS_CONFIGURATION,
    ordinal_train_step,
)
from src.stage2_encoder_only.training import build_optimizer, create_grad_scaler


class TinyEncoder(nn.Module):
    def __init__(self, hidden_size: int = 8) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size, dropout_rate=0.0)
        self.embedding = nn.Embedding(256, hidden_size)
        self.projection = nn.Linear(hidden_size, hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        del attention_mask
        batch, flat = input_ids.shape
        hidden = self.embedding(input_ids).view(batch, flat // 8, 8, -1).mean(2)
        return SimpleNamespace(last_hidden_state=self.projection(hidden))


class Stage2OrdinalTests(unittest.TestCase):
    def test_exact_target_cdf_construction(self) -> None:
        cdf = ordinal_target_cdf(torch.tensor([0, 64, 127]))
        self.assertEqual(tuple(cdf.shape), (3, 127))
        self.assertTrue(bool(cdf[0].all()))
        self.assertFalse(bool(cdf[2].any()))
        self.assertFalse(bool(cdf[1, :64].any()))
        self.assertTrue(bool(cdf[1, 64:].all()))

    def test_perfect_prediction_approaches_zero(self) -> None:
        logits = torch.full((1, 128), -40.0)
        logits[0, 73] = 40.0
        loss = squared_cdf_ordinal_loss(logits, torch.tensor([73]), 1.0)
        self.assertLess(float(loss.ordinal), 1e-12)

    def test_distance_sensitivity(self) -> None:
        target = torch.tensor([64])
        near = torch.full((1, 128), -30.0)
        far = near.clone()
        near[0, 60] = 30.0
        far[0, 5] = 30.0
        near_loss = squared_cdf_ordinal_loss(near, target, 1.0).ordinal
        far_loss = squared_cdf_ordinal_loss(far, target, 1.0).ordinal
        self.assertLess(float(near_loss), float(far_loss))

    def test_endpoint_mixture_worse_than_local_distribution(self) -> None:
        target = torch.tensor([64])
        endpoint = torch.full((1, 128), -80.0)
        endpoint[0, 0] = math.log(63.0 / 127.0)
        endpoint[0, 127] = math.log(64.0 / 127.0)
        local = torch.full((1, 128), -80.0)
        local[0, 63] = math.log(0.5)
        local[0, 65] = math.log(0.5)
        endpoint_loss = squared_cdf_ordinal_loss(endpoint, target, 1.0).ordinal
        local_loss = squared_cdf_ordinal_loss(local, target, 1.0).ordinal
        self.assertLess(float(local_loss), float(endpoint_loss))

    def test_ignore_index_contributes_to_neither_loss(self) -> None:
        valid_logits = torch.randn(2, 128)
        ignored_logits = torch.randn(1, 128) * 1000
        combined = torch.cat((valid_logits, ignored_logits))
        targets = torch.tensor([3, 90, -100])
        actual = squared_cdf_ordinal_loss(combined, targets, 0.5)
        expected = squared_cdf_ordinal_loss(valid_logits, targets[:2], 0.5)
        self.assertTrue(torch.allclose(actual.ce, expected.ce))
        self.assertTrue(torch.allclose(actual.ordinal, expected.ordinal))

    def test_gradient_flow_is_finite_and_nonzero(self) -> None:
        logits = torch.randn(2, 3, 4, 128, requires_grad=True)
        targets = torch.randint(0, 128, (2, 3, 4))
        loss = squared_cdf_ordinal_loss(logits, targets, 1.0)
        loss.total.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)

    def test_stage2_shape_handling(self) -> None:
        logits = torch.randn(2, 3, 4, 128)
        targets = torch.randint(0, 128, (2, 3, 4))
        losses = squared_cdf_ordinal_loss(logits, targets, 0.1)
        self.assertEqual(losses.total.ndim, 0)
        with self.assertRaisesRegex(ValueError, "target shape"):
            squared_cdf_ordinal_loss(logits, targets[:, :, :3], 0.1)

    def test_lambda_zero_matches_existing_unweighted_ce(self) -> None:
        logits = torch.randn(2, 3, 4, 128)
        targets = torch.randint(0, 128, (2, 3, 4))
        targets[0, 2] = -100
        actual = squared_cdf_ordinal_loss(logits, targets, 0.0).total
        expected = F.cross_entropy(logits.reshape(-1, 128), targets.reshape(-1), ignore_index=-100)
        self.assertTrue(torch.allclose(actual, expected, rtol=1e-6, atol=1e-7))

    def test_amp_dtype_inputs_keep_ordinal_float32_and_finite(self) -> None:
        targets = torch.tensor([0, 31, 64, 127])
        for dtype in (torch.float16, torch.bfloat16):
            logits = torch.randn(4, 128).to(dtype)
            losses = squared_cdf_ordinal_loss(logits, targets, 0.5)
            self.assertEqual(losses.ordinal.dtype, torch.float32)
            self.assertTrue(bool(torch.isfinite(losses.total)))

    def test_training_step_logs_separate_losses_and_updates_model(self) -> None:
        model = Stage2PedalEncoderModel(TinyEncoder(), dropout=0.0)
        optimizer = build_optimizer(model, encoder_lr=1e-2, head_lr=1e-2, weight_decay=0.0)
        input_ids = torch.randint(0, 256, (2, 24))
        batch = {
            "input_ids": input_ids,
            "token_attention_mask": torch.ones_like(input_ids),
            "pedal_targets": torch.randint(0, 128, (2, 3, 4)),
            "note_mask": torch.ones(2, 3, dtype=torch.bool),
        }
        before = [parameter.detach().clone() for parameter in model.parameters()]
        metrics = ordinal_train_step(
            model, batch, optimizer, 0.5,
            create_grad_scaler(False, "cpu"), False, 1.0,
        )
        for name in ("ce_loss", "ordinal_loss", "total_loss"):
            self.assertTrue(math.isfinite(float(metrics[name])))
        self.assertAlmostEqual(
            float(metrics["total_loss"]),
            float(metrics["ce_loss"]) + 0.5 * float(metrics["ordinal_loss"]),
            places=5,
        )
        self.assertGreater(float(metrics["encoder_gradient_norm"]), 0.0)
        self.assertGreater(float(metrics["head_gradient_norm"]), 0.0)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, model.parameters())))

    def test_checkpoint_roundtrip_preserves_lambda_and_loss_configuration(self) -> None:
        model = Stage2PedalEncoderModel(TinyEncoder(), dropout=0.0)
        configuration = {"lambda_ordinal": 0.5, "loss_configuration": dict(LOSS_CONFIGURATION)}
        checkpoint = build_best_checkpoint(model, configuration, 2, 1.25)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "best.pt"
            atomic_torch_save(checkpoint, path)
            loaded = torch.load(path, weights_only=False)
        self.assertEqual(loaded["configuration"]["lambda_ordinal"], 0.5)
        self.assertEqual(loaded["configuration"]["loss_configuration"], LOSS_CONFIGURATION)
        clone = Stage2PedalEncoderModel(TinyEncoder(), dropout=0.0)
        clone.load_state_dict(loaded["model_state"])

    def test_invalid_all_ignored_and_negative_lambda_are_rejected(self) -> None:
        logits = torch.randn(2, 128)
        with self.assertRaisesRegex(ValueError, "valid"):
            squared_cdf_ordinal_loss(logits, torch.full((2,), -100), 1.0)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            squared_cdf_ordinal_loss(logits, torch.tensor([1, 2]), -0.1)


if __name__ == "__main__":
    unittest.main()
