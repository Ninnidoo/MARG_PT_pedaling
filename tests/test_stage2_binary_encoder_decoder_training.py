"""Focused tests for encoder-decoder gradient accumulation."""

from __future__ import annotations

import unittest

import torch

from src.stage2_binary.full_training import make_binary_loader
from src.stage2_binary.training import build_binary_optimizer
from src.stage2_binary_encoder_decoder.training import run_accumulated_training_epoch
from tests.test_stage2_binary_encoder_decoder import make_model


def sample(notes: int, offset: int):
    input_ids = (torch.arange(notes * 8) + offset).remainder(200).long()
    raw = torch.tensor(
        [[0, 127, 0, 127], [127, 127, 0, 0], [127, 0, 127, 0]],
        dtype=torch.long,
    )
    repeats = (notes + len(raw) - 1) // len(raw)
    return {
        "input_ids": input_ids,
        "pedal_targets": raw.repeat(repeats, 1)[:notes],
        "note_mask": torch.ones(notes, dtype=torch.bool),
        "metadata": {"offset": offset},
    }


class AccumulatedTrainingTest(unittest.TestCase):
    def test_partial_final_group_steps_and_reaches_both_parameter_groups(self) -> None:
        model = make_model()
        optimizer = build_binary_optimizer(
            model, encoder_lr=1e-5, head_lr=1e-4, weight_decay=0.01
        )
        loader = make_binary_loader(
            [sample(3 + index % 2, index) for index in range(5)],
            batch_size=1,
            pin_memory=False,
        )
        metrics, checks = run_accumulated_training_epoch(
            model,
            loader,
            device=torch.device("cpu"),
            optimizer=optimizer,
            scaler=torch.amp.GradScaler("cpu", enabled=False),
            amp_enabled=False,
            accumulation_steps=2,
            max_grad_norm=1.0,
            max_consecutive_amp_skips=8,
        )
        self.assertEqual(checks["micro_batches"], 5)
        self.assertEqual(checks["attempted_steps"], 3)
        self.assertEqual(checks["optimizer_steps"], 3)
        self.assertEqual(checks["amp_skipped_steps"], 0)
        self.assertTrue(checks["encoder_gradient_present"])
        self.assertTrue(checks["head_gradient_present"])
        self.assertTrue(0.0 <= metrics["binary_accuracy"] <= 1.0)
        self.assertTrue(0.0 <= metrics["exact_pattern_accuracy"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
