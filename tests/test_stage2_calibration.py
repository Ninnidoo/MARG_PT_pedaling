"""Synthetic tests for hierarchical Stage 2 decoder calibration."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from src.stage2_encoder_only.calibrate_decoder import (
    aggregate_candidate_metrics,
    apply_region_calibration,
    calibration_config_roundtrip,
    conditional_depth_decode,
    fit_region_calibrator,
    hierarchical_decode,
    make_piece_folds,
    performance_candidate_metrics,
    prior_corrected_logits,
    region_logits,
    select_conditional_candidate,
    select_final_candidate,
    smoothed_intermediate_priors,
    success_criteria,
)
from src.stage2_encoder_only.evaluate_oracle import average_overlapping_logits


class Stage2CalibrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.counts = np.arange(1, 127, dtype=np.int64)
        self.priors, self.log_priors = smoothed_intermediate_priors(self.counts)

    def test_region_logits_use_endpoint_and_intermediate_logsumexp(self) -> None:
        logits = np.arange(128, dtype=np.float32)[None, :]
        groups = region_logits(logits)
        self.assertEqual(float(groups[0, 0]), 0.0)
        self.assertEqual(float(groups[0, 2]), 127.0)
        expected = torch.logsumexp(torch.from_numpy(logits[:, 1:127]), dim=-1)
        np.testing.assert_allclose(groups[:, 1], expected.numpy(), rtol=1e-6)

    def test_raw_hierarchy_selects_zero_full_and_conditional_map(self) -> None:
        logits = np.full((3, 128), -20.0, dtype=np.float32)
        logits[0, 0] = 8.0
        logits[1, 127] = 8.0
        logits[2, 42] = 8.0
        prediction = hierarchical_decode(logits, self.log_priors, "map")
        np.testing.assert_array_equal(prediction, [0, 127, 42])

    def test_conditional_mean_and_median_rules(self) -> None:
        uniform = np.zeros((1, 126), dtype=np.float32)
        mean = conditional_depth_decode(uniform, 0.0, "mean", np.zeros(126))
        median = conditional_depth_decode(uniform, 0.0, "median", np.zeros(126))
        np.testing.assert_array_equal(mean, [64])
        np.testing.assert_array_equal(median, [63])

    def test_temperature_and_bias_calibration_is_finite_and_normalized(self) -> None:
        groups = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
        calibrated = apply_region_calibration(groups, 2.0, 0.5, -0.25)
        probabilities = torch.softmax(torch.from_numpy(calibrated), dim=-1).numpy()
        self.assertTrue(np.isfinite(probabilities).all())
        np.testing.assert_allclose(probabilities.sum(axis=-1), 1.0)
        with self.assertRaises(ValueError):
            apply_region_calibration(groups, 0.0)

    def test_region_calibration_reduces_nll_deterministically(self) -> None:
        targets = np.tile(np.arange(3), 30)
        groups = np.zeros((90, 3), dtype=np.float32)
        groups[np.arange(90), targets] = 2.0
        groups[::3] *= -1.5
        first = fit_region_calibrator(groups, targets, "bias_temperature")
        second = fit_region_calibrator(groups, targets, "bias_temperature")
        self.assertLess(first["final_nll"], first["initial_nll"])
        self.assertGreater(first["temperature"], 1e-4)
        self.assertEqual(first["bias_zero"], 0.0)
        self.assertAlmostEqual(first["temperature"], second["temperature"], places=7)
        self.assertAlmostEqual(first["bias_intermediate"], second["bias_intermediate"], places=7)

    def test_prior_smoothing_and_correction(self) -> None:
        counts = np.zeros(126, dtype=np.int64)
        counts[0] = 99
        priors, log_priors = smoothed_intermediate_priors(counts)
        self.assertAlmostEqual(float(priors.sum()), 1.0, places=6)
        logits = np.zeros((1, 126), dtype=np.float32)
        np.testing.assert_array_equal(
            prior_corrected_logits(logits, 0.0, log_priors), logits
        )
        corrected = prior_corrected_logits(logits, 1.0, log_priors)
        self.assertGreater(float(corrected[0, 1]), float(corrected[0, 0]))

    def test_piece_folds_prevent_leakage_and_cover_each_performance_once(self) -> None:
        records = [
            {"piece_id": "a", "performance_path": "a1"},
            {"piece_id": "a", "performance_path": "a2"},
            {"piece_id": "b", "performance_path": "b1"},
        ]
        folds = make_piece_folds(records)
        self.assertEqual(len(folds), 2)
        held_out = [index for fold in folds for index in fold["held_out_indices"]]
        self.assertEqual(sorted(held_out), [0, 1, 2])
        for fold in folds:
            self.assertFalse(
                set(fold["held_out_indices"]) & set(fold["calibration_indices"])
            )

    def test_conditional_selection_obeys_ties_and_final_order(self) -> None:
        rows = [
            {"tau": 0.5, "decoder": "mean", "intermediate_mae": 10.00, "overall_mae": 9.0, "intermediate_tolerance_10": 0.5},
            {"tau": 0.25, "decoder": "median", "intermediate_mae": 10.04, "overall_mae": 8.0, "intermediate_tolerance_10": 0.4},
            {"tau": 0.0, "decoder": "map", "intermediate_mae": 10.20, "overall_mae": 1.0, "intermediate_tolerance_10": 1.0},
        ]
        selected = select_conditional_candidate(rows)
        self.assertEqual((selected["tau"], selected["decoder"]), (0.25, "median"))
        exact_ties = [
            {"tau": tau, "decoder": decoder, "intermediate_mae": 1.0, "overall_mae": 1.0, "intermediate_tolerance_10": 1.0}
            for tau, decoder in ((0.0, "mean"), (0.0, "median"), (0.0, "map"))
        ]
        self.assertEqual(select_conditional_candidate(exact_ties)["decoder"], "map")

    def test_out_of_fold_aggregation_micro_and_macro(self) -> None:
        targets = np.asarray([[0, 10, 127, 20], [127, 20, 0, 30]])
        records = [
            performance_candidate_metrics(targets, targets, "a"),
            performance_candidate_metrics(targets, targets, "b"),
        ]
        aggregate = aggregate_candidate_metrics(records)
        self.assertEqual(aggregate["performance_paths"], ["a", "b"])
        self.assertEqual(aggregate["micro_exact_accuracy"], 1.0)
        self.assertEqual(aggregate["macro_exact_accuracy"], 1.0)
        self.assertEqual(aggregate["micro_mae"], 0.0)

    def test_success_criteria_pass_and_fail(self) -> None:
        passing = {
            "micro_mae": 27.0,
            "micro_intermediate_mae": 34.0,
            "micro_predicted_intermediate_ratio": 0.3,
            "micro_intermediate_endpoint_collapse_ratio": 0.2,
            "micro_group_macro_f1": 0.57,
            "non_degenerate": True,
            "all_folds_finite": True,
        }
        self.assertTrue(success_criteria(passing)["success_criteria_met"])
        failing = dict(passing, micro_predicted_intermediate_ratio=0.9)
        self.assertFalse(success_criteria(failing)["success_criteria_met"])

    def test_final_selection_uses_success_then_mae_and_simplicity(self) -> None:
        base = {
            "success_criteria_met": True,
            "micro_mae": 27.0,
            "micro_intermediate_mae": 33.0,
            "micro_group_macro_f1": 0.58,
        }
        rows = [
            dict(base, candidate_name="temperature_hierarchy"),
            dict(base, candidate_name="raw_hierarchy_map"),
            dict(base, candidate_name="posterior_median", success_criteria_met=False, micro_mae=20.0),
        ]
        ranked = select_final_candidate(rows)
        self.assertEqual(ranked[0]["candidate_name"], "raw_hierarchy_map")
        self.assertEqual([row["final_rank"] for row in ranked], [1, 2, 3])

    def test_json_roundtrip_preserves_decoder_outputs(self) -> None:
        logits = np.full((3, 128), -20.0, dtype=np.float32)
        logits[0, 0] = 8.0
        logits[1, 127] = 8.0
        logits[2, 50] = 8.0
        config = {
            "log_intermediate_priors": self.log_priors.tolist(),
            "conditional_decoder": "median",
            "selected_tau": 0.25,
            "fitted_temperature": 1.2,
            "fitted_group_biases": {"zero": 0.0, "intermediate": 0.1, "full": -0.2},
        }
        expected = hierarchical_decode(
            logits, self.log_priors, "median", 0.25, 1.2, 0.1, -0.2
        )
        np.testing.assert_array_equal(calibration_config_roundtrip(config, logits), expected)

    def test_overlap_reconstruction_is_order_invariant(self) -> None:
        first = np.zeros((3, 4, 128), dtype=np.float32)
        second = np.ones((2, 4, 128), dtype=np.float32)
        windows = [(0, first), (1, second)]
        normal, counts = average_overlapping_logits(3, windows)
        reverse, reverse_counts = average_overlapping_logits(3, list(reversed(windows)))
        np.testing.assert_array_equal(normal, reverse)
        np.testing.assert_array_equal(counts, reverse_counts)


if __name__ == "__main__":
    unittest.main()
