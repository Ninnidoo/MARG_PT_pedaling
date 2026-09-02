from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from src.stage2_event_model.status_logging import (
    NonFiniteStatusMetricError,
    status_to_json_safe,
    write_failure_status,
    write_run_status,
)


def status() -> dict:
    return {
        "status": "running", "stage": "training", "epoch": 1,
        "global_optimizer_attempt": 0, "global_optimizer_step": 0,
        "best_epoch": None, "best_val_total": None,
        "latest_train_total": None, "latest_val_total": None,
        "latest_validation_components": None,
        "amp_overflow_count": 0, "max_consecutive_amp_overflow": 0,
        "failure_reason": None,
    }


class StatusLoggingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.path = self.root / "run_status.json"

    def tearDown(self):
        self.temporary.cleanup()

    def test_prevalidation_none_is_strict_json(self):
        write_run_status(self.path, status())
        value = json.loads(self.path.read_text())
        self.assertIsNone(value["best_val_total"])
        self.assertIsNone(value["latest_val_total"])

    def test_finite_values_are_exact(self):
        value = status()
        value.update(best_epoch=1, best_val_total=0.25, latest_train_total=0.5, latest_val_total=0.25)
        write_run_status(self.path, value)
        loaded = json.loads(self.path.read_text())
        self.assertEqual(loaded["latest_train_total"], 0.5)
        self.assertEqual(loaded["best_val_total"], 0.25)

    def test_legacy_benign_best_inf_becomes_null(self):
        value = status()
        value["best_val_total"] = math.inf
        safe = status_to_json_safe(value)
        self.assertIsNone(safe["best_val_total"])
        self.assertIn("best_val_total", safe["undefined_status_fields"])

    def test_unexpected_loss_inf_and_nan_fail(self):
        for nonfinite in (math.inf, math.nan):
            value = status()
            value["latest_train_total"] = nonfinite
            with self.assertRaises(NonFiniteStatusMetricError):
                write_run_status(self.path, value)

    def test_amp_overflow_norm_is_explicit_strict_json(self):
        value = status()
        value.update(amp_overflow_count=1, max_consecutive_amp_overflow=1)
        value["last_overflow"] = {
            "epoch": 1, "global_optimizer_attempt": 2001,
            "gradient_norm": math.inf, "scale_before": 1024.0,
            "scale_after": 512.0, "affected_groups": ["encoder"],
        }
        write_run_status(self.path, value)
        loaded = json.loads(self.path.read_text())
        self.assertIsNone(loaded["last_overflow"]["gradient_norm"])
        self.assertEqual(
            loaded["last_overflow"]["gradient_norm_nonfinite_kind"],
            "positive_infinity",
        )

    def test_attempt_2000_like_status_serializes(self):
        value = status()
        value.update(
            epoch=1, global_optimizer_attempt=2000, global_optimizer_step=2000,
            latest_train_total=0.9036892784626746,
        )
        write_run_status(self.path, value)
        loaded = json.loads(self.path.read_text())
        self.assertEqual(loaded["global_optimizer_step"], 2000)
        self.assertIsNone(loaded["best_val_total"])

    def test_failure_handler_preserves_primary_and_records_nonfinite(self):
        value = status()
        value["latest_train_total"] = math.inf
        error = write_failure_status(
            self.path, value, failure_reason="FloatingPointError: primary",
            primary_traceback="PRIMARY TRACEBACK", updated_at="now",
            emergency_log_path=self.root / "emergency.log",
        )
        self.assertIsNone(error)
        loaded = json.loads(self.path.read_text())
        self.assertEqual(loaded["status"], "failed")
        self.assertEqual(loaded["traceback"], "PRIMARY TRACEBACK")
        self.assertEqual(loaded["nonfinite_status_fields"][0]["path"], "latest_train_total")

    def test_failure_status_write_failure_keeps_emergency_primary_trace(self):
        emergency = self.root / "emergency.log"
        error = write_failure_status(
            self.root, status(), failure_reason="RuntimeError: primary",
            primary_traceback="ORIGINAL PRIMARY TRACE", updated_at="now",
            emergency_log_path=emergency,
        )
        self.assertIsNotNone(error)
        self.assertIn("ORIGINAL PRIMARY TRACE", emergency.read_text())

    def test_atomicity_on_serialization_failure(self):
        write_run_status(self.path, status())
        before = self.path.read_bytes()
        broken = status()
        broken["latest_train_total"] = math.inf
        with self.assertRaises(NonFiniteStatusMetricError):
            write_run_status(self.path, broken)
        self.assertEqual(self.path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
