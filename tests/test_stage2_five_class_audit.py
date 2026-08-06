from __future__ import annotations

import unittest

import numpy as np

from src.stage2_encoder_only.audit_five_class import CLASS_NAMES, class_counts, classify_five


class FiveClassAuditHelpersTest(unittest.TestCase):
    def test_fixed_boundaries(self) -> None:
        values = np.arange(128)
        classes = classify_five(values)
        self.assertEqual(len(CLASS_NAMES), 5)
        self.assertEqual(np.bincount(classes, minlength=5).tolist(), [1, 63, 32, 31, 1])
        self.assertEqual(
            classes[[0, 1, 63, 64, 95, 96, 126, 127]].tolist(),
            [0, 1, 1, 2, 2, 3, 3, 4],
        )

    def test_histogram_accounting(self) -> None:
        histogram = np.arange(1, 129, dtype=np.int64)
        counts = class_counts(histogram)
        self.assertEqual(int(counts.sum()), int(histogram.sum()))
        self.assertEqual(int(counts[0]), 1)
        self.assertEqual(int(counts[-1]), 128)

    def test_out_of_range_rejected(self) -> None:
        with self.assertRaises(ValueError):
            classify_five(np.array([-1, 0]))
        with self.assertRaises(ValueError):
            classify_five(np.array([127, 128]))


if __name__ == "__main__":
    unittest.main()
