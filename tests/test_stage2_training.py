from __future__ import annotations

import math
import random
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from src.stage2_encoder_only.model import Stage2PedalEncoderModel
from src.stage2_encoder_only.training import (
    build_optimizer,
    calculate_pedal_metrics,
    create_grad_scaler,
    evaluation_step,
    set_deterministic_seed,
    train_step,
)


class TinyEncoder(nn.Module):
    def __init__(self, hidden_size: int = 12) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size, dropout_rate=0.0)
        self.embedding = nn.Embedding(256, hidden_size)
        self.projection = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> SimpleNamespace:
        del attention_mask
        batch_size, flat_length = input_ids.shape
        hidden = self.embedding(input_ids).view(
            batch_size,
            flat_length // 8,
            8,
            self.config.hidden_size,
        )
        return SimpleNamespace(
            last_hidden_state=self.projection(hidden.mean(dim=2))
        )


class Stage2TrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        set_deterministic_seed()

    @staticmethod
    def model() -> Stage2PedalEncoderModel:
        return Stage2PedalEncoderModel(TinyEncoder(), dropout=0.0)

    @staticmethod
    def batch() -> dict[str, torch.Tensor]:
        input_ids = torch.randint(0, 256, (2, 24))
        return {
            "input_ids": input_ids,
            "token_attention_mask": torch.ones_like(input_ids),
            "pedal_targets": torch.randint(0, 128, (2, 3, 4)),
            "note_mask": torch.ones((2, 3), dtype=torch.bool),
        }

    def test_metric_correctness_and_padding_exclusion(self) -> None:
        targets = torch.tensor(
            [[[0, 1, 2, 3], [4, 5, 6, 7], [-100, -100, -100, -100]]]
        )
        predictions = torch.tensor([[[0, 9, 2, 8], [4, 5, 0, 7], [1, 2, 3, 4]]])
        logits = torch.full((1, 3, 4, 128), -10.0)
        logits.scatter_(-1, predictions.unsqueeze(-1), 10.0)
        metrics = calculate_pedal_metrics(logits, targets, loss=1.25)

        self.assertEqual(metrics["loss"], 1.25)
        self.assertEqual(metrics["valid_target_count"], 8)
        self.assertEqual(metrics["valid_note_count"], 2)
        self.assertAlmostEqual(metrics["pedal_token_accuracy"], 0.625)
        self.assertAlmostEqual(metrics["exact_note_accuracy"], 0.0)
        self.assertAlmostEqual(metrics["pedal1_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["pedal2_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["pedal3_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["pedal4_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["pedal_value_mae"], 2.375)

        changed = logits.clone()
        changed[:, 2] = torch.randn_like(changed[:, 2]) * 1000
        changed_metrics = calculate_pedal_metrics(changed, targets, loss=1.25)
        self.assertEqual(metrics, changed_metrics)

    def test_optimizer_groups_are_complete_disjoint_and_use_requested_lrs(self) -> None:
        model = self.model()
        optimizer = build_optimizer(
            model,
            encoder_lr=2e-5,
            head_lr=3e-4,
            weight_decay=0.02,
        )
        by_name = {group["group_name"]: group for group in optimizer.param_groups}
        self.assertEqual(set(by_name), {"encoder", "heads"})
        self.assertEqual(by_name["encoder"]["lr"], 2e-5)
        self.assertEqual(by_name["heads"]["lr"], 3e-4)
        grouped_ids = [
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        trainable_ids = [
            id(parameter) for parameter in model.parameters() if parameter.requires_grad
        ]
        self.assertEqual(len(grouped_ids), len(set(grouped_ids)))
        self.assertEqual(set(grouped_ids), set(trainable_ids))

    def test_grad_scaler_exposes_experiment_initial_scale(self) -> None:
        scaler = create_grad_scaler(enabled=True, device="cpu")
        self.assertTrue(scaler.is_enabled())
        self.assertEqual(scaler.get_scale(), 1024.0)
        with self.assertRaisesRegex(ValueError, "amp_init_scale"):
            create_grad_scaler(enabled=True, device="cpu", amp_init_scale=0.0)

    def test_training_step_cpu_changes_parameters_and_clips_all_gradients(self) -> None:
        model = self.model()
        optimizer = build_optimizer(
            model,
            encoder_lr=1e-2,
            head_lr=1e-2,
            weight_decay=0.0,
        )
        scaler = create_grad_scaler(enabled=False, device="cpu")
        before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        metrics = train_step(
            model,
            self.batch(),
            optimizer,
            scaler=scaler,
            amp_enabled=False,
            max_grad_norm=0.1,
        )
        self.assertTrue(math.isfinite(metrics["loss"]))
        self.assertTrue(math.isfinite(metrics["encoder_gradient_norm"]))
        self.assertTrue(math.isfinite(metrics["head_gradient_norm"]))
        self.assertTrue(math.isfinite(metrics["total_gradient_norm"]))
        self.assertLessEqual(metrics["clipped_gradient_norm"], 0.10001)
        self.assertTrue(
            any(
                not torch.equal(before[name], parameter)
                for name, parameter in model.named_parameters()
            )
        )
        self.assertTrue(
            all(
                parameter.grad is not None
                and bool(torch.isfinite(parameter.grad).all())
                for parameter in model.parameters()
            )
        )
        self.assertTrue(
            all(
                any(parameter.grad is not None for parameter in head.parameters())
                for head in model.classification_heads
            )
        )

    def test_evaluation_changes_no_parameters_leaves_no_gradients_and_uses_eval(self) -> None:
        model = self.model()
        optimizer = build_optimizer(model)
        train_step(model, self.batch(), optimizer, amp_enabled=False)
        before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        metrics = evaluation_step(model, self.batch(), amp_enabled=False)

        self.assertTrue(math.isfinite(metrics["loss"]))
        self.assertFalse(model.training)
        self.assertTrue(
            all(
                torch.equal(before[name], parameter)
                for name, parameter in model.named_parameters()
            )
        )
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_seed_setup_is_deterministic(self) -> None:
        set_deterministic_seed()
        first = (random.random(), np.random.rand(), torch.rand(1))
        set_deterministic_seed()
        second = (random.random(), np.random.rand(), torch.rand(1))
        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])
        self.assertTrue(torch.equal(first[2], second[2]))


if __name__ == "__main__":
    unittest.main()
