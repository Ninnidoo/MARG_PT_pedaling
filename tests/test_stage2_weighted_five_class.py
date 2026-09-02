"""Synthetic contract tests for train-global inverse-sqrt weighted five-class CE."""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.stage2_encoder_only.five_class import FiveClassPedalEncoderModel
from src.stage2_encoder_only.train_five_class import inverse_sqrt_class_weights
from src.stage2_encoder_only.training import build_optimizer, train_step


class TinyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8, dropout_rate=0.0)
        self.embedding = nn.Embedding(64, 8)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        del attention_mask
        batch, length = input_ids.shape
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids).view(batch, length // 8, 8, -1).mean(2))


class WeightedFiveClassTests(unittest.TestCase):
    def test_inverse_sqrt_weights_have_requested_normalization(self) -> None:
        counts = np.asarray([100, 400, 900, 1600, 2500], dtype=np.int64)
        weights = inverse_sqrt_class_weights(counts)
        np.testing.assert_allclose(weights / weights[0], np.sqrt(counts[0] / counts))
        probabilities = counts / counts.sum()
        self.assertAlmostEqual(float(np.dot(probabilities, weights)), 1.0, places=14)
        self.assertTrue(np.isfinite(weights).all())

    def test_amp_logit_dtype_preserves_all_five_normalized_weights(self) -> None:
        counts = [2975658, 1758841, 2014383, 869828, 4335746]
        weights = torch.tensor(inverse_sqrt_class_weights(counts), dtype=torch.float32)
        expected = torch.tensor([0.9258, 1.2042, 1.1252, 1.7124, 0.7669])
        self.assertTrue(torch.allclose(weights, expected, atol=5e-4, rtol=0.0))
        logits = torch.randn(1, 1, 4, 5, dtype=torch.float16)
        targets = torch.tensor([[[0, 1, 2, 3]]])
        loss_dtype_weights = weights.to(dtype=logits.dtype)
        self.assertEqual(torch.unique(loss_dtype_weights).numel(), 5)
        self.assertTrue(torch.isfinite(loss_dtype_weights).all())
        loss = FiveClassPedalEncoderModel.compute_loss(logits, targets, weights)
        self.assertTrue(torch.isfinite(loss))

    def test_weighted_ce_matches_torch_and_ignores_padding(self) -> None:
        torch.manual_seed(7)
        logits = torch.randn(1, 2, 4, 5, requires_grad=True)
        targets = torch.tensor([[[0, 1, 2, 3], [4, -100, 1, 0]]])
        weights = torch.tensor(inverse_sqrt_class_weights([10, 20, 30, 40, 50]), dtype=torch.float32)
        actual = FiveClassPedalEncoderModel.compute_loss(logits, targets, weights)
        expected = F.cross_entropy(logits.reshape(-1, 5), targets.reshape(-1), weight=weights, ignore_index=-100)
        self.assertTrue(torch.allclose(actual, expected))
        actual.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_weighted_loss_and_gradients_are_finite_for_all_heads(self) -> None:
        torch.manual_seed(11)
        model = FiveClassPedalEncoderModel(TinyEncoder(), dropout=0.0)
        model.set_class_weights(inverse_sqrt_class_weights([9, 17, 33, 55, 101]))
        ids = torch.randint(0, 63, (2, 3 * 8))
        batch = {
            "input_ids": ids,
            "token_attention_mask": torch.ones_like(ids),
            "pedal_targets": torch.randint(0, 5, (2, 3, 4)),
            "note_mask": torch.ones((2, 3), dtype=torch.bool),
        }
        optimizer = build_optimizer(model, encoder_lr=1e-2, head_lr=1e-2, weight_decay=0.0)
        metrics = train_step(model, batch, optimizer, amp_enabled=False)
        self.assertTrue(math.isfinite(float(metrics["loss"])))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))


if __name__ == "__main__":
    unittest.main()
