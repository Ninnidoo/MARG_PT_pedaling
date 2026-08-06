"""Synthetic tests for the validation posterior diagnostic."""

from __future__ import annotations

import math
import unittest

import numpy as np

from src.stage2_encoder_only.diagnose_posterior import (
    aggregate_decoder_metrics,
    aggregate_endpoint_mixture,
    aggregate_slot_posterior,
    aggregate_transition_posterior,
    coarse_confusion,
    decoder_metrics,
    endpoint_mixture_diagnostic,
    finalize_class_statistics,
    initialize_class_statistics,
    posterior_group_masses,
    posterior_mean_decode,
    posterior_median_decode,
    slot_posterior_diagnostic,
    target_group_ids,
    transition_posterior_diagnostic,
    true_class_ranks,
    update_class_statistics,
)
from src.stage2_encoder_only.evaluate_oracle import average_overlapping_logits


def one_hot(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    posterior = np.zeros((*values.shape, 128), dtype=np.float32)
    np.put_along_axis(posterior, values[..., None], 1.0, axis=-1)
    return posterior


class Stage2PosteriorTests(unittest.TestCase):
    def test_group_masses_partition_all_classes(self) -> None:
        posterior = np.zeros((1, 128), dtype=np.float32)
        posterior[0, [0, 1, 126, 127]] = [0.2, 0.1, 0.2, 0.5]
        masses = posterior_group_masses(posterior)
        self.assertAlmostEqual(float(masses["p_zero"][0]), 0.2)
        self.assertAlmostEqual(float(masses["p_intermediate"][0]), 0.3)
        self.assertAlmostEqual(float(masses["p_full"][0]), 0.5)
        self.assertAlmostEqual(
            float(
                masses["p_zero"][0]
                + masses["p_intermediate"][0]
                + masses["p_full"][0]
            ),
            1.0,
        )
        self.assertEqual(int(masses["best_intermediate_class"][0]), 126)

    def test_posterior_mean_uses_float32_expectation_and_ties_to_even(self) -> None:
        posterior = np.zeros((3, 128), dtype=np.float32)
        posterior[0, 10] = 1.0
        posterior[1, [0, 1]] = 0.5
        posterior[2, [126, 127]] = [0.4, 0.6]
        prediction, expectation = posterior_mean_decode(posterior)
        np.testing.assert_array_equal(prediction, [10, 0, 127])
        np.testing.assert_allclose(expectation, [10.0, 0.5, 126.6])
        self.assertEqual(expectation.dtype, np.float32)

    def test_posterior_median_is_smallest_class_reaching_half(self) -> None:
        posterior = np.zeros((2, 128), dtype=np.float32)
        posterior[0, [2, 3, 4]] = [0.49, 0.01, 0.5]
        posterior[1, [0, 127]] = [0.5, 0.5]
        np.testing.assert_array_equal(posterior_median_decode(posterior), [3, 0])

    def test_true_class_rank_has_deterministic_lower_class_tie_break(self) -> None:
        posterior = np.zeros((3, 128), dtype=np.float32)
        posterior[0, [1, 2, 3]] = [0.4, 0.4, 0.2]
        posterior[1, [4, 5, 6]] = [0.5, 0.25, 0.25]
        posterior[2] = 1.0 / 128.0
        ranks = true_class_ranks(posterior, np.asarray([2, 6, 10]))
        np.testing.assert_array_equal(ranks, [2, 3, 11])
        self.assertAlmostEqual(float((ranks <= 5).mean()), 2.0 / 3.0)

    def test_coarse_prediction_uses_summed_group_probability(self) -> None:
        posterior = np.zeros((1, 128), dtype=np.float32)
        posterior[0, [0, 10, 20, 127]] = [0.30, 0.20, 0.20, 0.30]
        masses = posterior_group_masses(posterior)
        group_prediction = np.stack(
            [masses["p_zero"], masses["p_intermediate"], masses["p_full"]],
            axis=-1,
        ).argmax(axis=-1)
        self.assertEqual(int(posterior.argmax(axis=-1)[0]), 0)
        self.assertEqual(int(group_prediction[0]), 1)
        result = coarse_confusion(np.asarray([1]), group_prediction)
        self.assertEqual(result["confusion_matrix"][1, 1], 1)

    def test_endpoint_mixture_categories_are_exhaustive(self) -> None:
        posterior = np.zeros((3, 128), dtype=np.float32)
        posterior[0, 64] = 1.0
        posterior[1, [0, 127]] = 0.5
        posterior[2] = 1.0 / 128.0
        prediction = np.asarray([64, 64, 64])
        targets = np.asarray([64, 64, 64])
        result = endpoint_mixture_diagnostic(posterior, prediction, targets)
        aggregate = aggregate_endpoint_mixture([result])
        self.assertEqual(aggregate["all"]["intermediate_prediction_count"], 3)
        self.assertEqual(aggregate["all"]["locally_supported_count"], 1)
        self.assertEqual(aggregate["all"]["endpoint_mixture_dominated_count"], 1)
        self.assertEqual(aggregate["all"]["diffuse_other_count"], 1)

    def test_transition_stale_state_is_local_to_each_performance(self) -> None:
        targets = np.asarray([[0, 0, 0, 0], [127, 64, 64, 64]])
        posterior = one_hot(targets).astype(np.float32)
        posterior[1, 0] = 0.0
        posterior[1, 0, [0, 127]] = [0.8, 0.2]
        posterior[1, 1] = 0.0
        posterior[1, 1, [127, 64]] = [0.75, 0.25]
        argmax = posterior.argmax(axis=-1)
        mean, _ = posterior_mean_decode(posterior)
        median = posterior_median_decode(posterior)
        ranks = true_class_ranks(posterior, targets)
        result = transition_posterior_diagnostic(
            posterior, targets, ranks, argmax, mean, median
        )
        aggregate = aggregate_transition_posterior([result])
        self.assertEqual(aggregate["stale_groups"]["all"]["count"], 2)
        self.assertEqual(aggregate["stale_groups"]["upward"]["count"], 1)
        self.assertEqual(aggregate["stale_groups"]["downward"]["count"], 1)
        self.assertEqual(
            aggregate["stale_groups"]["endpoint_to_endpoint"]["count"], 1
        )
        self.assertEqual(
            aggregate["stale_groups"]["involving_intermediate"]["count"], 1
        )
        self.assertEqual(
            aggregate["stale_groups"]["all"][
                "previous_probability_greater_fraction"
            ],
            1.0,
        )

        constant_a = np.zeros((1, 4), dtype=np.int64)
        constant_b = np.full((1, 4), 127, dtype=np.int64)
        per_performance = []
        for constant in (constant_a, constant_b):
            probabilities = one_hot(constant)
            per_performance.append(
                transition_posterior_diagnostic(
                    probabilities,
                    constant,
                    true_class_ranks(probabilities, constant),
                    constant,
                    constant,
                    constant,
                )
            )
        no_boundary = aggregate_transition_posterior(per_performance)
        self.assertEqual(no_boundary["stale_groups"]["all"]["count"], 0)

    def test_delta_mae_never_crosses_performance_boundaries(self) -> None:
        first_prediction = np.asarray([[0, 2, 2, 4]])
        first_target = np.asarray([[0, 1, 3, 3]])
        second_prediction = np.asarray([[127, 127, 127, 127]])
        second_target = second_prediction.copy()
        aggregate = aggregate_decoder_metrics(
            [
                decoder_metrics(first_prediction, first_target),
                decoder_metrics(second_prediction, second_target),
            ]
        )
        self.assertAlmostEqual(aggregate["delta_mae"], 5.0 / 6.0)

    def test_slot_statistics_keep_slots_separate(self) -> None:
        targets = np.asarray([[0, 5, 0, 127], [127, 5, 0, 127]])
        posterior = one_hot(targets)
        predictions = targets.copy()
        ranks = true_class_ranks(posterior, targets)
        slots = aggregate_slot_posterior(
            [
                slot_posterior_diagnostic(
                    posterior, targets, ranks, predictions, predictions
                )
            ]
        )
        self.assertEqual(len(slots), 4)
        self.assertEqual(slots[0]["transition_count"], 1)
        self.assertEqual(slots[1]["transition_count"], 0)
        self.assertEqual(slots[1]["target_intermediate_ratio"], 1.0)
        self.assertEqual(slots[3]["target_full_ratio"], 1.0)

    def test_class_support_keeps_train_and_validation_separate(self) -> None:
        train_support = np.zeros(128, dtype=np.int64)
        train_support[[5, 7]] = [10, 20]
        statistics = initialize_class_statistics(train_support)
        targets = np.asarray([[5, 5, 5, 5]])
        posterior = one_hot(targets)
        update_class_statistics(
            statistics,
            posterior,
            targets,
            true_class_ranks(posterior, targets),
            targets,
            targets,
            targets,
        )
        rows, _ = finalize_class_statistics(statistics)
        self.assertEqual(rows[5]["train_support"], 10)
        self.assertEqual(rows[5]["validation_support"], 4)
        self.assertEqual(rows[7]["train_support"], 20)
        self.assertEqual(rows[7]["validation_support"], 0)
        self.assertTrue(math.isnan(rows[7]["argmax_recall"]))
        self.assertEqual(rows[5]["argmax_prediction_count"], 4)

    def test_target_groups_cover_zero_intermediate_and_full(self) -> None:
        np.testing.assert_array_equal(
            target_group_ids(np.asarray([0, 1, 126, 127])), [0, 1, 1, 2]
        )
        with self.assertRaises(ValueError):
            target_group_ids(np.asarray([-1]))

    def test_overlap_average_is_order_invariant_before_softmax(self) -> None:
        first = np.zeros((3, 4, 128), dtype=np.float32)
        second = np.zeros((2, 4, 128), dtype=np.float32)
        first[..., 3] = 2.0
        second[..., 7] = 4.0
        windows = [(0, first), (1, second)]
        average, counts = average_overlapping_logits(3, windows)
        reverse, reverse_counts = average_overlapping_logits(
            3, list(reversed(windows))
        )
        np.testing.assert_array_equal(average, reverse)
        np.testing.assert_array_equal(counts, reverse_counts)
        np.testing.assert_array_equal(counts, [1, 2, 2])


if __name__ == "__main__":
    unittest.main()
