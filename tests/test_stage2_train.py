"""Focused tests for the reusable Stage 2 full-training driver."""

from __future__ import annotations

import signal
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from src.stage2_encoder_only.train import (
    EarlyStopping,
    EpochMetricAccumulator,
    GracefulStop,
    atomic_torch_save,
    build_best_checkpoint,
    build_last_checkpoint,
    deterministic_train_order,
    restore_training_state,
)
from src.stage2_encoder_only.training import build_optimizer, create_grad_scaler


class TinyTrainModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(3, 4)
        self.classification_heads = nn.ModuleList(
            [nn.Linear(4, 2) for _ in range(4)]
        )


def batch_metrics(
    *,
    targets: int,
    loss: float,
    token_accuracy: float,
    exact_accuracy: float,
    mae: float,
) -> dict[str, float | int]:
    notes = targets // 4
    return {
        "loss": loss,
        "pedal_token_accuracy": token_accuracy,
        "exact_note_accuracy": exact_accuracy,
        "pedal1_accuracy": token_accuracy,
        "pedal2_accuracy": token_accuracy,
        "pedal3_accuracy": token_accuracy,
        "pedal4_accuracy": token_accuracy,
        "pedal_value_mae": mae,
        "valid_target_count": targets,
        "valid_note_count": notes,
    }


class Stage2TrainTests(unittest.TestCase):
    def test_epoch_metrics_are_count_weighted(self) -> None:
        accumulator = EpochMetricAccumulator()
        accumulator.update(
            batch_metrics(
                targets=8,
                loss=2.0,
                token_accuracy=0.25,
                exact_accuracy=0.0,
                mae=4.0,
            )
        )
        accumulator.update(
            batch_metrics(
                targets=24,
                loss=1.0,
                token_accuracy=0.75,
                exact_accuracy=1.0,
                mae=2.0,
            )
        )

        metrics = accumulator.compute()
        self.assertAlmostEqual(metrics["loss"], 1.25)
        self.assertAlmostEqual(metrics["pedal_token_accuracy"], 0.625)
        self.assertAlmostEqual(metrics["exact_note_accuracy"], 0.75)
        self.assertAlmostEqual(metrics["pedal1_accuracy"], 0.625)
        self.assertAlmostEqual(metrics["pedal_value_mae"], 2.5)
        self.assertEqual(metrics["valid_target_count"], 32)
        self.assertEqual(metrics["valid_note_count"], 8)

    def test_best_checkpoint_selection_and_contents(self) -> None:
        model = TinyTrainModel()
        stopping = EarlyStopping(patience=2, min_delta=0.1)
        improved, should_stop = stopping.update(1.0, 1)
        self.assertTrue(improved)
        self.assertFalse(should_stop)
        improved, _ = stopping.update(0.95, 2)
        self.assertFalse(improved)
        improved, _ = stopping.update(0.9, 3)
        self.assertTrue(improved)

        checkpoint = build_best_checkpoint(model, {"seed": 7}, 3, 0.9)
        self.assertEqual(checkpoint["best_epoch"], 3)
        self.assertEqual(checkpoint["best_validation_loss"], 0.9)
        self.assertEqual(checkpoint["configuration"], {"seed": 7})
        self.assertIn("model_state", checkpoint)
        self.assertNotIn("optimizer_state", checkpoint)

    def test_last_checkpoint_contains_required_state(self) -> None:
        model = TinyTrainModel()
        optimizer = build_optimizer(model)
        scaler = create_grad_scaler(enabled=False, device="cpu")
        stopping = EarlyStopping(
            patience=4, min_delta=1e-4, best_loss=0.7, counter=2, best_epoch=3
        )
        checkpoint = build_last_checkpoint(
            model, optimizer, scaler, 4, 123, stopping, {"batch_size": 8}
        )
        required = {
            "model_state",
            "optimizer_state",
            "grad_scaler_state",
            "completed_epoch",
            "global_optimizer_step",
            "best_validation_loss",
            "early_stopping_counter",
            "configuration",
            "python_rng_state",
            "numpy_rng_state",
            "torch_cpu_rng_state",
            "torch_cuda_rng_state",
        }
        self.assertTrue(required.issubset(checkpoint))
        self.assertEqual(checkpoint["completed_epoch"], 4)
        self.assertEqual(checkpoint["global_optimizer_step"], 123)

    def test_atomic_checkpoint_replaces_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "last.pt"
            atomic_torch_save({"version": 1}, destination)
            atomic_torch_save({"version": 2}, destination)
            loaded = torch.load(destination, weights_only=False)
            self.assertEqual(loaded, {"version": 2})
            self.assertEqual(
                [path.name for path in destination.parent.iterdir()], ["last.pt"]
            )

    def test_resume_starts_after_completed_epoch(self) -> None:
        source_model = TinyTrainModel()
        source_optimizer = build_optimizer(source_model)
        source_scaler = create_grad_scaler(enabled=False, device="cpu")
        stopping = EarlyStopping(
            patience=4, min_delta=1e-4, best_loss=0.5, counter=1, best_epoch=2
        )
        checkpoint = build_last_checkpoint(
            source_model,
            source_optimizer,
            source_scaler,
            6,
            321,
            stopping,
            {"seed": 20260710},
        )

        target_model = TinyTrainModel()
        target_optimizer = build_optimizer(target_model)
        target_scaler = create_grad_scaler(enabled=False, device="cpu")
        target_stopping = EarlyStopping(patience=4, min_delta=1e-4)
        start_epoch, global_step = restore_training_state(
            checkpoint,
            target_model,
            target_optimizer,
            target_scaler,
            target_stopping,
        )
        self.assertEqual(start_epoch, 7)
        self.assertEqual(global_step, 321)
        self.assertEqual(target_stopping.best_loss, 0.5)
        self.assertEqual(target_stopping.counter, 1)
        for source, target in zip(
            source_model.parameters(), target_model.parameters()
        ):
            self.assertTrue(torch.equal(source, target))

    def test_early_stopping_logic(self) -> None:
        stopping = EarlyStopping(patience=2, min_delta=0.01)
        self.assertEqual(stopping.update(1.0, 1), (True, False))
        self.assertEqual(stopping.update(0.995, 2), (False, False))
        self.assertEqual(stopping.update(0.994, 3), (False, True))
        self.assertEqual(stopping.update(0.98, 4), (True, False))
        self.assertEqual(stopping.best_epoch, 4)

    def test_non_finite_loss_is_rejected(self) -> None:
        metrics = batch_metrics(
            targets=8,
            loss=float("nan"),
            token_accuracy=0.5,
            exact_accuracy=0.5,
            mae=1.0,
        )
        with self.assertRaises(FloatingPointError):
            EpochMetricAccumulator().update(metrics)

    def test_graceful_stop_records_signal(self) -> None:
        stop = GracefulStop()
        stop.handler(signal.SIGTERM, None)
        self.assertTrue(stop.requested)
        self.assertEqual(stop.signal_number, signal.SIGTERM)

    def test_train_order_is_deterministic_and_epoch_specific(self) -> None:
        first = deterministic_train_order(20, 20260710, 1)
        second = deterministic_train_order(20, 20260710, 1)
        later = deterministic_train_order(20, 20260710, 2)
        self.assertEqual(first, second)
        self.assertNotEqual(first, later)
        self.assertEqual(sorted(first), list(range(20)))

    def test_optimizer_omits_no_trainable_parameter(self) -> None:
        model = TinyTrainModel()
        optimizer = build_optimizer(model, encoder_lr=1e-5, head_lr=1e-4)
        optimizer_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        trainable_ids = {
            id(parameter) for parameter in model.parameters() if parameter.requires_grad
        }
        self.assertEqual(optimizer_ids, trainable_ids)
        self.assertEqual(
            [group["group_name"] for group in optimizer.param_groups],
            ["encoder", "heads"],
        )


if __name__ == "__main__":
    unittest.main()
