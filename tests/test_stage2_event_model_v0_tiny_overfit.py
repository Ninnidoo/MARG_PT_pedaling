"""Preflight invariants for the deterministic tiny memorization subset."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.stage2_event_model.dataset import CANONICAL_CACHE_ID


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "analysis/custom_event_model_v0_tiny_overfit/tiny_subset_manifest.json"


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


class TinySubsetPreflightTests(unittest.TestCase):
    def test_tiny_subset_is_train_only_deterministic_and_bounded(self) -> None:
        payload = load_manifest()
        self.assertEqual(payload["cache_id"], CANONICAL_CACHE_ID)
        self.assertEqual(payload["split"], "train")
        self.assertEqual(payload["seed"], 42)
        self.assertEqual(payload["validation_access_count"], 0)
        self.assertEqual(payload["asap_test_access_count"], 0)
        self.assertLessEqual(len(payload["selected"]), 16)
        self.assertEqual(len(payload["selection_sha256"]), 64)

    def test_tiny_subset_covers_every_required_branch(self) -> None:
        payload = load_manifest()
        self.assertLessEqual(set(payload["required_coverage"]), set(payload["covered"]))
        self.assertFalse(payload["missing_desired"])
        summary = payload["summary"]
        self.assertGreaterEqual(summary["initial_target_count"], 1)
        self.assertGreaterEqual(summary["terminal_supervision_window_count"], 1)
        for slot in range(1, 7):
            row = summary["main_slots"][f"Slot{slot}"]
            self.assertGreaterEqual(row["non_none_count"], 1)
            self.assertEqual(row["timing_valid_count"], row["non_none_count"])
        for slot in range(1, 5):
            row = summary["terminal_slots"][f"Slot{slot}"]
            self.assertGreaterEqual(row["non_none_count"], 1)
            self.assertEqual(row["timing_valid_count"], row["non_none_count"])

    def test_each_selected_window_adds_new_coverage(self) -> None:
        payload = load_manifest()
        covered: set[str] = set()
        for row in payload["selected"]:
            new = set(row["newly_covered_requirements"])
            self.assertTrue(new)
            self.assertTrue(new.isdisjoint(covered))
            covered.update(row["all_window_coverage"])


if __name__ == "__main__":
    unittest.main()
