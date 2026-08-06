from __future__ import annotations

import unittest

import numpy as np

from src.stage2_encoder_only.audit_six_class import (
    CLASS_NAMES,
    classify,
    count_extreme_repedal_patterns,
    count_repedal_patterns,
    histogram_class_counts,
    weighted_summary,
)


class SixClassAuditHelpersTest(unittest.TestCase):
    def test_fixed_boundaries_cover_all_values(self) -> None:
        values = np.arange(128)
        classes = classify(values)
        expected_counts = [1, 31, 32, 32, 31, 1]
        self.assertEqual(len(CLASS_NAMES), 6)
        self.assertEqual(np.bincount(classes, minlength=6).tolist(), expected_counts)
        self.assertEqual(classes[[0, 1, 31, 32, 63, 64, 95, 96, 126, 127]].tolist(),
                         [0, 1, 1, 2, 2, 3, 3, 4, 4, 5])

    def test_histogram_class_counts(self) -> None:
        histogram = np.arange(1, 129, dtype=np.int64)
        counts = histogram_class_counts(histogram)
        self.assertEqual(int(counts.sum()), int(histogram.sum()))
        self.assertEqual(int(counts[0]), 1)
        self.assertEqual(int(counts[-1]), 128)

    def test_weighted_summary_uses_linear_quantile(self) -> None:
        summary = weighted_summary(np.array([0.0, 10.0]), np.array([1, 1]))
        self.assertEqual(summary["mean"], 5.0)
        self.assertEqual(summary["median"], 5.0)
        self.assertEqual(summary["min"], 0.0)
        self.assertEqual(summary["max"], 10.0)

    def test_repedal_pattern_counters(self) -> None:
        binary = np.array([1, 1, 0, 0, 1, 1, 0, 0, 0, 1], dtype=np.uint8)
        self.assertEqual(count_repedal_patterns(binary, 2), 1)
        self.assertEqual(count_repedal_patterns(binary, 3), 2)
        classes = np.array([4, 5, 1, 0, 4, 3, 1, 4], dtype=np.uint8)
        self.assertEqual(count_extreme_repedal_patterns(classes, 2), 1)

    def test_classification_rejects_out_of_range(self) -> None:
        with self.assertRaises(ValueError):
            classify(np.array([-1, 0]))
        with self.assertRaises(ValueError):
            classify(np.array([127, 128]))


if __name__ == "__main__":
    unittest.main()
