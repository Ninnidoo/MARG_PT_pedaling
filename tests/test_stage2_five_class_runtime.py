"""Synthetic-only tests for the five-class training run-time contract.

Nothing in this module constructs ``Stage2PedalDataset`` or opens a split CSV.
The tests use tiny in-memory modules and temporary directories exclusively.
"""

from __future__ import annotations

import csv
import json
import random
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch import nn

from src.stage2_encoder_only.train import (
    EarlyStopping,
    build_last_checkpoint,
    restore_training_state,
)
from src.stage2_encoder_only.train_five_class import (
    _early_stopping_already_satisfied,
    _validate_resume_checkpoint,
    _validate_resume_metrics,
    default_configuration,
)
from src.stage2_encoder_only.training import build_optimizer, create_grad_scaler
from src.stage2_encoder_only.run_five_class import (
    AtomicRunStatus,
    DuplicateRunError,
    acquire_run_lock,
    atomic_json,
    collect_git_metadata,
    initial_run_status,
    release_run_lock,
)


class TinyRuntimeModel(nn.Module):
    """Minimal model with the same optimizer grouping surface as Stage 2."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(3, 4)
        self.classification_heads = nn.ModuleList(
            [nn.Linear(4, 5) for _ in range(4)]
        )

    def loss(self) -> torch.Tensor:
        # Touch every parameter so AdamW has a populated state for every group.
        return sum(parameter.square().sum() for parameter in self.parameters())


def assert_nested_equal(
    case: unittest.TestCase, expected: object, actual: object
) -> None:
    if isinstance(expected, torch.Tensor):
        case.assertIsInstance(actual, torch.Tensor)
        case.assertTrue(torch.equal(expected, actual))
    elif isinstance(expected, dict):
        case.assertIsInstance(actual, dict)
        case.assertEqual(set(expected), set(actual))
        for key in expected:
            assert_nested_equal(case, expected[key], actual[key])
    elif isinstance(expected, (list, tuple)):
        case.assertIsInstance(actual, type(expected))
        case.assertEqual(len(expected), len(actual))
        for left, right in zip(expected, actual):
            assert_nested_equal(case, left, right)
    else:
        case.assertEqual(expected, actual)


class FiveClassCheckpointRuntimeTests(unittest.TestCase):
    def test_exact_epoch_boundary_restores_every_training_state(self) -> None:
        random.seed(177)
        np.random.seed(177)
        torch.manual_seed(177)
        source_model = TinyRuntimeModel()
        source_optimizer = build_optimizer(
            source_model, encoder_lr=1e-3, head_lr=2e-3, weight_decay=0.01
        )
        source_scaler = create_grad_scaler(False, "cpu", 1024.0)
        source_model.loss().backward()
        source_optimizer.step()
        source_optimizer.zero_grad(set_to_none=True)
        stopping = EarlyStopping(
            patience=4,
            min_delta=1e-4,
            best_loss=0.375,
            counter=3,
            best_epoch=5,
        )
        configuration = default_configuration("/tmp/synthetic-five-class")
        checkpoint = build_last_checkpoint(
            source_model,
            source_optimizer,
            source_scaler,
            completed_epoch=6,
            global_step=741,
            early_stopping=stopping,
            configuration=configuration,
        )
        _validate_resume_checkpoint(checkpoint, configuration)
        self.assertNotIn("scheduler_state", checkpoint)

        expected_python = random.random()
        expected_numpy = np.random.random(4)
        expected_torch = torch.rand(4)

        target_model = TinyRuntimeModel()
        target_optimizer = build_optimizer(
            target_model, encoder_lr=1e-3, head_lr=2e-3, weight_decay=0.01
        )
        target_scaler = create_grad_scaler(False, "cpu", 1024.0)
        target_stopping = EarlyStopping(patience=4, min_delta=1e-4)
        random.seed(999)
        np.random.seed(999)
        torch.manual_seed(999)

        start_epoch, global_step = restore_training_state(
            checkpoint,
            target_model,
            target_optimizer,
            target_scaler,
            target_stopping,
        )

        self.assertEqual(start_epoch, 7)
        self.assertEqual(global_step, 741)
        self.assertEqual(target_stopping.best_loss, 0.375)
        self.assertEqual(target_stopping.counter, 3)
        self.assertEqual(target_stopping.best_epoch, 5)
        assert_nested_equal(self, checkpoint["model_state"], target_model.state_dict())
        assert_nested_equal(
            self, checkpoint["optimizer_state"], target_optimizer.state_dict()
        )
        assert_nested_equal(
            self, checkpoint["grad_scaler_state"], target_scaler.state_dict()
        )
        self.assertEqual(random.random(), expected_python)
        np.testing.assert_array_equal(np.random.random(4), expected_numpy)
        self.assertTrue(torch.equal(torch.rand(4), expected_torch))

    def test_resume_validation_rejects_non_boundary_checkpoint(self) -> None:
        configuration = default_configuration("/tmp/synthetic-five-class")
        complete = {
            "model_state": {},
            "optimizer_state": {},
            "grad_scaler_state": {},
            "completed_epoch": 2,
            "global_optimizer_step": 9,
            "best_validation_loss": 1.0,
            "early_stopping_counter": 0,
            "python_rng_state": (),
            "numpy_rng_state": (),
            "torch_cpu_rng_state": torch.empty(0, dtype=torch.uint8),
            "torch_cuda_rng_state": [],
            "configuration": configuration,
        }
        incomplete = dict(complete)
        del incomplete["optimizer_state"]
        with self.assertRaisesRegex(ValueError, "exact epoch boundary"):
            _validate_resume_checkpoint(incomplete, configuration)

    def test_resume_validation_rejects_scheduler_state(self) -> None:
        model = TinyRuntimeModel()
        optimizer = build_optimizer(model)
        scaler = create_grad_scaler(False, "cpu")
        configuration = default_configuration("/tmp/synthetic-five-class")
        checkpoint = build_last_checkpoint(
            model,
            optimizer,
            scaler,
            completed_epoch=1,
            global_step=1,
            early_stopping=EarlyStopping(4, 1e-4),
            configuration=configuration,
        )
        checkpoint["scheduler_state"] = {"last_epoch": 1}
        with self.assertRaisesRegex(ValueError, "unexpected scheduler"):
            _validate_resume_checkpoint(checkpoint, configuration)

    def test_resume_with_exhausted_patience_must_not_enter_another_epoch(self) -> None:
        exhausted = EarlyStopping(
            patience=4, min_delta=1e-4, best_loss=0.5,
            counter=4, best_epoch=3,
        )
        remaining = EarlyStopping(
            patience=4, min_delta=1e-4, best_loss=0.5,
            counter=3, best_epoch=3,
        )
        self.assertTrue(_early_stopping_already_satisfied(exhausted))
        self.assertFalse(_early_stopping_already_satisfied(remaining))

    def test_resume_metrics_must_end_at_completed_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "metrics.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["epoch", "loss"])
                writer.writeheader()
                writer.writerow({"epoch": 1, "loss": 2.0})
                writer.writerow({"epoch": 2, "loss": 1.0})
            _validate_resume_metrics(path, 2)
            with self.assertRaisesRegex(RuntimeError, "not aligned"):
                _validate_resume_metrics(path, 3)


class FiveClassOrchestrationRuntimeTests(unittest.TestCase):
    def test_active_duplicate_lock_is_blocked_and_records_required_pids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "output"
            lock = acquire_run_lock(
                root, "synthetic-run-a", host_pid=424242,
                container_pid=os.getpid(),
            )
            payload = json.loads(lock.path.read_text(encoding="utf-8"))
            self.assertEqual(payload["run_id"], "synthetic-run-a")
            self.assertEqual(payload["host_pid"], 424242)
            self.assertEqual(payload["container_pid"], os.getpid())
            self.assertEqual(payload["container_process"]["pid"], os.getpid())
            with self.assertRaises(DuplicateRunError):
                acquire_run_lock(
                    root, "synthetic-run-b", host_pid=424243,
                    container_pid=os.getpid(),
                )
            archive = release_run_lock(lock, "test_complete")
            self.assertIsNotNone(archive)
            assert archive is not None
            self.assertTrue(archive.is_file())
            self.assertFalse(lock.path.exists())
            self.assertEqual(list(root.glob(".run.lock*")), [])

    def test_proven_stale_lock_is_archived_before_new_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "output"
            impossible_pid = 2_000_000_000
            stale = acquire_run_lock(
                root, "stale-run", host_pid=impossible_pid,
                container_pid=impossible_pid,
            )
            self.assertTrue(stale.path.is_file())
            current = acquire_run_lock(
                root, "replacement-run", host_pid=os.getpid(),
                container_pid=os.getpid(),
            )
            self.assertEqual(current.payload["run_id"], "replacement-run")
            self.assertEqual(len(current.stale_archives), 1)
            stale_archive = Path(current.stale_archives[0])
            self.assertTrue(stale_archive.is_file())
            archived_payload = json.loads(
                stale_archive.read_text(encoding="utf-8")
            )
            self.assertEqual(archived_payload["run_id"], "stale-run")
            self.assertEqual(
                json.loads(current.path.read_text(encoding="utf-8"))["run_id"],
                "replacement-run",
            )
            release_run_lock(current, "test_complete")

    def test_atomic_status_updates_always_leave_complete_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "run_status.json"
            initial = initial_run_status(
                "synthetic-status", host_pid=77, container_pid=88,
                tmux_session="synthetic-tmux", gpu_uuid="synthetic-gpu",
                start_time_utc="2026-08-04T00:00:00+00:00",
                start_time_kst="2026-08-04T09:00:00+09:00",
            )
            status = AtomicRunStatus(path, initial)
            first = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(first["stage"], "initializing")
            status.update(
                stage="training", current_epoch=3, current_step=125,
                best_epoch=2, best_validation_loss=0.75,
            )
            second = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(second["stage"], "training")
            self.assertEqual(second["current_epoch"], 3)
            self.assertEqual(second["current_step"], 125)
            self.assertFalse(second["completed"])
            self.assertFalse(second["failed"])
            self.assertEqual(list(path.parent.glob(".*.tmp")), [])

    def test_failed_atomic_replace_preserves_previous_valid_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "run_status.json"
            atomic_json(path, {"stage": "testing", "generation": 1})
            with mock.patch(
                "src.stage2_encoder_only.run_five_class.os.replace",
                side_effect=OSError("synthetic interruption before rename"),
            ):
                with self.assertRaises(OSError):
                    atomic_json(path, {"stage": "training", "generation": 2})
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"stage": "testing", "generation": 1},
            )
            self.assertEqual(list(path.parent.glob(".*.tmp")), [])

    def test_git_executable_absence_is_nonfatal_and_yields_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            with mock.patch(
                "src.stage2_encoder_only.run_five_class.subprocess.run",
                side_effect=FileNotFoundError("synthetic git executable missing"),
            ):
                metadata = collect_git_metadata(temporary_directory, environ={})
            self.assertEqual(metadata["commit"], "unknown")
            self.assertEqual(metadata["commit_source"], "unknown")
            self.assertEqual(metadata["dirty_status"], "unknown")
            self.assertIsNone(metadata["dirty"])
            self.assertEqual(metadata["dirty_status_source"], "unknown")
            self.assertEqual(len(metadata["warnings"]), 2)
            self.assertTrue(
                all("FileNotFoundError" in item for item in metadata["warnings"])
            )

    def test_host_git_environment_has_priority_over_subprocess(self) -> None:
        environment = {
            "STAGE2_PROJECT_GIT_COMMIT": "abc123",
            "STAGE2_PROJECT_GIT_STATUS": " M synthetic.py",
        }
        with mock.patch(
            "src.stage2_encoder_only.run_five_class.subprocess.run"
        ) as subprocess_run:
            metadata = collect_git_metadata("/synthetic/project", environment)
        subprocess_run.assert_not_called()
        self.assertEqual(metadata["commit"], "abc123")
        self.assertEqual(metadata["commit_source"], "STAGE2_PROJECT_GIT_COMMIT")
        self.assertEqual(metadata["dirty_status"], " M synthetic.py")
        self.assertTrue(metadata["dirty"])
        self.assertEqual(metadata["warnings"], [])


if __name__ == "__main__":
    unittest.main()
