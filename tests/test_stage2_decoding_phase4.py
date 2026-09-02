import unittest

import numpy as np

from src.stage2_four_class.decoding_strategies import (
    cc64_integer_to_classes,
    decode_argmax,
    decode_auxiliary_median,
    decode_expectation,
    decode_ordered_median,
    decode_side_constrained,
    decode_top2,
    softmax_final_logits,
    stable_top2_indices,
)


class DecodingPhase4Tests(unittest.TestCase):
    def test_median_examples_and_ge_half_boundary(self):
        posterior = np.asarray(
            [
                [0.10, 0.20, 0.45, 0.25],
                [0.49, 0.02, 0.20, 0.29],
                [0.50, 0.10, 0.10, 0.30],
            ]
        )
        np.testing.assert_array_equal(decode_ordered_median(posterior), [2, 1, 0])

    def test_expectation_and_half_up_without_representative_snapping(self):
        posterior = np.asarray([[0.10, 0.20, 0.45, 0.25]])
        continuous, integer = decode_expectation(posterior)
        expected = 0 * 0.10 + 51 * 0.20 + 79 * 0.45 + 127 * 0.25
        self.assertAlmostEqual(float(continuous[0]), expected, places=12)
        self.assertEqual(int(integer[0]), int(np.floor(expected + 0.5)))
        self.assertNotIn(int(integer[0]), [0, 51, 79, 127])

    def test_top2_support_tie_rule_and_seed42_reproducibility(self):
        posterior = np.asarray(
            [
                [0.10, 0.20, 0.45, 0.25],
                [0.40, 0.40, 0.10, 0.10],
                [0.25, 0.25, 0.25, 0.25],
            ]
        )
        top2 = stable_top2_indices(posterior)
        np.testing.assert_array_equal(top2, [[2, 3], [0, 1], [0, 1]])
        first, _ = decode_top2(posterior, piece_id="piece_test", seed=42)
        second, _ = decode_top2(posterior, piece_id="piece_test", seed=42)
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.all((first == top2[:, 0]) | (first == top2[:, 1])))

    def test_same_posterior_is_consumed_without_mutation(self):
        logits = np.asarray([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
        posterior = softmax_final_logits(logits)
        before = posterior.copy()
        decode_argmax(posterior)
        decode_ordered_median(posterior)
        decode_expectation(posterior)
        decode_top2(posterior, piece_id="same_posterior", seed=42)
        np.testing.assert_array_equal(posterior, before)

    def test_canonical_boundaries(self):
        np.testing.assert_array_equal(
            cc64_integer_to_classes(np.asarray([25, 26, 63, 64, 103, 104])),
            [0, 1, 1, 2, 2, 3],
        )

    def test_phase4_standard_ce_cached_median_exact_regression(self):
        import json
        from pathlib import Path

        root = Path("analysis/stage2_encoder_only_4class_decoding_phase4_v0")
        manifest = root / "posterior_cache/manifest.json"
        if not manifest.is_file():
            self.skipTest("completed Phase 4 posterior cache is unavailable")
        for item in json.loads(manifest.read_text())["pieces"]:
            piece = item["piece_id"]
            logits = np.load(root / "posterior_cache" / piece / "final_logits_float32.npy", allow_pickle=False)
            expected = np.load(root / "predictions" / "median" / piece / "classes_int64.npy", allow_pickle=False)
            np.testing.assert_array_equal(decode_ordered_median(softmax_final_logits(logits)), expected)

    def test_auxiliary_median_is_regression_independent(self):
        posterior = np.asarray([[0.10, 0.20, 0.45, 0.25]])
        regression_a = np.asarray([0])
        regression_b = np.asarray([127])
        self.assertFalse(np.array_equal(regression_a, regression_b))
        np.testing.assert_array_equal(decode_auxiliary_median(posterior), decode_auxiliary_median(posterior))

    def test_side_constrained_off_and_on(self):
        posterior = np.asarray([[0.01, 0.02, 0.47, 0.50], [0.50, 0.47, 0.02, 0.01]])
        classes = decode_side_constrained(np.asarray([40, 90]), posterior)
        self.assertIn(int(classes[0]), (0, 1))
        self.assertIn(int(classes[1]), (2, 3))
        np.testing.assert_array_equal(classes, [1, 2])

    def test_side_constrained_transition_invariance_and_no_cross_side_change(self):
        raw = np.asarray([10, 40, 63, 64, 90, 127, 30, 80])
        posterior = np.asarray([
            [0.1, 0.2, 0.3, 0.4], [0.9, 0.1, 0.0, 0.0],
            [0.1, 0.9, 0.0, 0.0], [0.9, 0.0, 0.1, 0.0],
            [0.0, 0.0, 0.1, 0.9], [0.0, 0.0, 0.9, 0.1],
            [0.2, 0.1, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1],
        ])
        fused = decode_side_constrained(raw, posterior)
        np.testing.assert_array_equal(raw >= 64, fused >= 2)
        regression_classes = cc64_integer_to_classes(raw)
        self.assertEqual(int(np.sum((regression_classes >= 2) != (fused >= 2))), 0)


if __name__ == "__main__":
    unittest.main()
