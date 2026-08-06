"""Synthetic tests for overlap-aware Stage 2 Oracle evaluation."""

from __future__ import annotations

import unittest

import numpy as np

from src.stage2_encoder_only.dataset import generate_window_starts
from src.stage2_encoder_only.evaluate_oracle import (
    aggregate_standard,
    aggregate_transition,
    average_overlapping_logits,
    compute_train_majorities,
    configuration_histogram,
    distribution_similarity,
    intermediate_metrics,
    macro_standard,
    make_baselines,
    pedal_configuration_ids,
    standard_metrics,
    transition_metrics,
)


class Stage2EvaluateTests(unittest.TestCase):
    def test_overlap_logits_are_averaged_once_and_order_invariant(self) -> None:
        first = np.zeros((4, 4, 128), dtype=np.float32)
        second = np.zeros((3, 4, 128), dtype=np.float32)
        first[..., 3] = 2.0
        second[..., 7] = 4.0
        windows = [(0, first), (2, second)]

        averaged, counts = average_overlapping_logits(5, windows)
        reversed_average, reversed_counts = average_overlapping_logits(
            5, list(reversed(windows))
        )

        np.testing.assert_array_equal(averaged, reversed_average)
        np.testing.assert_array_equal(counts, [1, 1, 2, 2, 1])
        np.testing.assert_array_equal(counts, reversed_counts)
        np.testing.assert_array_equal(averaged[0], first[0])
        np.testing.assert_array_equal(averaged[4], second[2])
        np.testing.assert_array_equal(averaged[2], (first[2] + second[0]) / 2)

        predictions = averaged.argmax(axis=-1)
        targets = np.zeros((5, 4), dtype=np.int64)
        metrics = standard_metrics(predictions, targets)
        self.assertEqual(metrics["valid_note_count"], 5)
        self.assertEqual(metrics["valid_target_count"], 20)

    def test_tail_coverage_for_short_exact_and_tail_aligned_sequences(self) -> None:
        for notes, expected_starts in (
            (100, [0]),
            (512, [0]),
            (1025, [0, 256, 512, 513]),
        ):
            starts = generate_window_starts(notes, 512, 256)
            self.assertEqual(starts, expected_starts)
            windows = [
                (
                    start,
                    np.zeros(
                        (min(512, notes - start), 4, 128), dtype=np.float32
                    ),
                )
                for start in starts
            ]
            _, counts = average_overlapping_logits(notes, windows)
            self.assertGreaterEqual(int(counts.min()), 1)
            self.assertGreaterEqual(int(counts[0]), 1)
            self.assertGreaterEqual(int(counts[-1]), 1)

    def test_intermediate_mask_endpoint_collapse_and_padding(self) -> None:
        targets = np.asarray(
            [[0, 1, 64, 126], [127, -100, 5, 100]], dtype=np.int64
        )
        predictions = np.asarray(
            [[0, 1, 0, 127], [127, 127, 127, 100]], dtype=np.int64
        )
        metrics = intermediate_metrics(predictions, targets)
        self.assertEqual(metrics["intermediate_target_count"], 5)
        self.assertAlmostEqual(metrics["intermediate_exact_accuracy"], 0.4)
        self.assertAlmostEqual(metrics["intermediate_predicted_zero_ratio"], 0.2)
        self.assertAlmostEqual(metrics["intermediate_predicted_full_ratio"], 0.4)
        self.assertAlmostEqual(metrics["endpoint_collapse_ratio"], 0.6)

    def test_transition_detection_subsets_and_direction(self) -> None:
        targets = np.asarray([[0, 0, 10, 10]], dtype=np.int64)
        predictions = np.asarray([[0, 5, 10, 10]], dtype=np.int64)
        metrics = transition_metrics(predictions, targets)
        self.assertEqual(metrics["transition_position_count"], 1)
        self.assertEqual(metrics["steady_position_count"], 2)
        self.assertAlmostEqual(metrics["transition_position_exact_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["transition_position_mae"], 0.0)
        self.assertAlmostEqual(metrics["steady_position_exact_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["steady_position_mae"], 2.5)
        self.assertAlmostEqual(metrics["transition_detection_precision"], 0.5)
        self.assertAlmostEqual(metrics["transition_detection_recall"], 1.0)
        self.assertAlmostEqual(metrics["transition_detection_f1"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["transition_direction_accuracy"], 1.0)

    def test_transition_aggregation_never_crosses_performances(self) -> None:
        first = np.zeros((1, 4), dtype=np.int64)
        second = np.full((1, 4), 127, dtype=np.int64)
        first_metrics = transition_metrics(first, first)
        second_metrics = transition_metrics(second, second)
        combined = aggregate_transition([first_metrics, second_metrics])
        self.assertEqual(combined["transition_position_count"], 0)
        self.assertEqual(combined["steady_position_count"], 6)
        self.assertAlmostEqual(combined["steady_position_exact_accuracy"], 1.0)

    def test_train_majorities_and_all_baselines(self) -> None:
        train_targets = [
            np.asarray(
                [[1, 2, 3, 4], [1, 2, 3, 4], [5, 2, 3, 4]],
                dtype=np.int64,
            )
        ]
        global_majority, slots, _, _ = compute_train_majorities(train_targets)
        self.assertEqual(global_majority, 2)
        np.testing.assert_array_equal(slots, [1, 2, 3, 4])

        test_targets = np.asarray(
            [[9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]],
            dtype=np.int64,
        )
        baselines = make_baselines(test_targets, global_majority, slots)
        np.testing.assert_array_equal(baselines["all_zero"], 0)
        np.testing.assert_array_equal(baselines["all_full"], 127)
        np.testing.assert_array_equal(baselines["global_majority"], 2)
        np.testing.assert_array_equal(
            baselines["slot_majority"],
            np.asarray([[1, 2, 3, 4]] * 3),
        )
        np.testing.assert_array_equal(baselines["persistence"][0], slots)
        np.testing.assert_array_equal(
            baselines["persistence"][1:], test_targets[:-1]
        )

    def test_all_sixteen_configurations_map_uniquely(self) -> None:
        values = np.asarray(
            [
                [127 if pattern & weight else 0 for weight in (8, 4, 2, 1)]
                for pattern in range(16)
            ],
            dtype=np.int64,
        )
        identifiers = pedal_configuration_ids(values)
        np.testing.assert_array_equal(identifiers, np.arange(16))
        np.testing.assert_array_equal(configuration_histogram(values), np.ones(16))

    def test_identical_distribution_has_zero_js_and_unit_intersection(self) -> None:
        histogram = np.arange(1, 17, dtype=np.int64)
        metrics = distribution_similarity(histogram, histogram)
        self.assertAlmostEqual(metrics["js_divergence_base2"], 0.0)
        self.assertAlmostEqual(metrics["js_distance_base2"], 0.0)
        self.assertAlmostEqual(metrics["histogram_intersection"], 1.0)

    def test_micro_and_macro_differ_for_unequal_lengths(self) -> None:
        short_targets = np.zeros((1, 4), dtype=np.int64)
        short_predictions = np.zeros_like(short_targets)
        long_targets = np.zeros((3, 4), dtype=np.int64)
        long_predictions = np.ones_like(long_targets)
        per_performance = [
            standard_metrics(short_predictions, short_targets),
            standard_metrics(long_predictions, long_targets),
        ]
        micro = aggregate_standard(per_performance)
        macro = macro_standard(per_performance)
        self.assertAlmostEqual(micro["pedal_token_accuracy"], 0.25)
        self.assertAlmostEqual(macro["pedal_token_accuracy"], 0.5)
        self.assertAlmostEqual(micro["exact_note_accuracy"], 0.25)
        self.assertAlmostEqual(macro["exact_note_accuracy"], 0.5)

    def test_duplicate_window_start_is_rejected(self) -> None:
        logits = np.zeros((2, 4, 128), dtype=np.float32)
        with self.assertRaises(ValueError):
            average_overlapping_logits(2, [(0, logits), (0, logits)])


if __name__ == "__main__":
    unittest.main()
