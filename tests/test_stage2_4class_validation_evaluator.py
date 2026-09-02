"""Targeted frozen-metric tests for four-model validation evaluation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage2_four_class.validation_evaluator import (
    PATTERN_COUNT,
    TRANSITION_TOLERANCE,
    classification_metrics,
    confusion_from_pairs,
    pattern_ids,
    pattern_metrics,
    pooled_transition_counts,
)


class FourModelMetricTests(unittest.TestCase):
    def test_four_class_micro_and_macro(self) -> None:
        target = np.asarray([[0, 1, 2, 3], [0, 1, 2, 3]])
        predicted = target.copy()
        metrics = classification_metrics(confusion_from_pairs(predicted, target))
        self.assertEqual(metrics["token_accuracy"], 1.0)
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(metrics["pedal_samples"], 8)

    def test_256_pattern_distribution(self) -> None:
        classes = np.asarray([[0, 1, 2, 3], [3, 2, 1, 0]])
        identifiers = pattern_ids(classes)
        histogram = np.bincount(identifiers, minlength=PATTERN_COUNT)
        metrics = pattern_metrics(histogram, histogram.copy())
        self.assertEqual(metrics["bins"], 256)
        self.assertAlmostEqual(metrics["js_divergence_base2"], 0.0)
        self.assertAlmostEqual(metrics["intersection"], 1.0)
        self.assertAlmostEqual(sum(metrics["candidate_distribution"]), 1.0)
        self.assertAlmostEqual(sum(metrics["target_distribution"]), 1.0)

    def test_direction_aware_one_to_one_tolerance_one(self) -> None:
        self.assertEqual(TRANSITION_TOLERANCE, 1)
        candidate = {"transitions": [
            {"direction": "DOWN", "score_position": 10, "onset_index": 10},
            {"direction": "DOWN", "score_position": 11, "onset_index": 11},
            {"direction": "UP", "score_position": 20, "onset_index": 20},
        ]}
        target = {"transitions": [
            {"direction": "DOWN", "score_position": 10, "onset_index": 10},
            {"direction": "UP", "score_position": 21, "onset_index": 21},
        ]}
        metrics = pooled_transition_counts(candidate, target)
        self.assertEqual(metrics["pooled"]["tp"], 2)
        self.assertEqual(metrics["pooled"]["fp"], 1)
        self.assertEqual(metrics["pooled"]["fn"], 0)
        self.assertEqual(metrics["pooled"]["tolerance_distinct_onsets"], 1)


if __name__ == "__main__":
    unittest.main()
