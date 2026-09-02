from __future__ import annotations

import unittest

import numpy as np
import torch

from src.stage2_binary.validation_evaluator import (
    binary_logits_to_raw_pedals,
    joint_ids_from_raw_pedals,
    normalized_joint16,
    official_pt_pedal_similarity,
    replace_pedal_tokens,
    select_validation_scores,
)


class BinaryValidationEvaluatorTests(unittest.TestCase):
    def test_threshold_boundary_and_exhaustive_joint_encoding(self) -> None:
        boundary = np.asarray([[63, 64, 63, 64]], dtype=np.int64)
        self.assertTrue(np.array_equal(joint_ids_from_raw_pedals(boundary), [5]))
        raw = np.asarray(
            [
                [127 if (joint_id >> shift) & 1 else 0 for shift in (3, 2, 1, 0)]
                for joint_id in range(16)
            ],
            dtype=np.int64,
        )
        self.assertTrue(
            np.array_equal(joint_ids_from_raw_pedals(raw), np.arange(16))
        )

    def test_distribution_normalization_and_identical_metrics(self) -> None:
        histogram = np.arange(1, 17, dtype=np.int64)
        probability = normalized_joint16(histogram)
        self.assertAlmostEqual(float(probability.sum()), 1.0, places=12)
        metrics = official_pt_pedal_similarity(histogram, histogram.copy())
        self.assertAlmostEqual(metrics["js_distance_base2"], 0.0, places=12)
        self.assertAlmostEqual(metrics["js_divergence_base2"], 0.0, places=12)
        self.assertAlmostEqual(metrics["histogram_intersection"], 1.0, places=12)

    def test_pedal_replacement_preserves_all_non_pedal_tokens_exactly(self) -> None:
        tokens = np.asarray(
            [
                [10, 20, 30, 40, 5261, 5262, 5263, 5264],
                [11, 21, 31, 41, 5388, 5387, 5386, 5385],
            ],
            dtype=np.int64,
        )
        raw = np.asarray([[127, 0, 64, 63], [1, 2, 3, 4]], dtype=np.int64)
        replaced = replace_pedal_tokens(tokens, raw).reshape(-1, 8)
        self.assertTrue(np.array_equal(replaced[:, :4], tokens[:, :4]))
        self.assertTrue(np.array_equal(replaced[:, 4:] - 5261, raw))

    def test_both_binary_logit_interfaces_decode_to_raw_zero_or_127(self) -> None:
        independent = torch.full((16, 4, 2), -1.0)
        for joint_id in range(16):
            for slot, shift in enumerate((3, 2, 1, 0)):
                independent[joint_id, slot, (joint_id >> shift) & 1] = 1.0
        independent_raw = binary_logits_to_raw_pedals(
            independent, "independent_4x2"
        )
        self.assertTrue(
            np.array_equal(
                joint_ids_from_raw_pedals(independent_raw), np.arange(16)
            )
        )
        joint = torch.eye(16) * 2.0 - 1.0
        joint_raw = binary_logits_to_raw_pedals(joint, "joint_16")
        self.assertTrue(
            np.array_equal(joint_ids_from_raw_pedals(joint_raw), np.arange(16))
        )

    def test_score_selection_uses_direct_support_then_lexical_tie_break(self) -> None:
        metadata = [
            {"midi_performance": "p/a.mid", "midi_score": "score/z.mid"},
            {"midi_performance": "p/b.mid", "midi_score": "score/a.mid"},
            {"midi_performance": "p/c.mid", "midi_score": "score/z.mid"},
            {"midi_performance": "p/d.mid", "midi_score": "score/b.mid"},
            {"midi_performance": "p/e.mid", "midi_score": "score/a.mid"},
        ]
        split = [
            {
                "metadata_index": str(index),
                "composer": "Composer",
                "title": "Title",
                "piece_id": piece,
                "performance_path": row["midi_performance"],
                "split": "validation",
            }
            for index, (piece, row) in enumerate(
                zip(("piece1", "piece1", "piece1", "piece2", "piece2"), metadata)
            )
        ]
        selected, validation = select_validation_scores(split, metadata)
        self.assertEqual(len(validation), 5)
        by_piece = {row["piece_id"]: row for row in selected}
        self.assertEqual(by_piece["piece1"]["selected_score_path"], "score/z.mid")
        self.assertEqual(by_piece["piece1"]["selected_score_support"], 2)
        self.assertEqual(by_piece["piece2"]["selected_score_path"], "score/a.mid")


if __name__ == "__main__":
    unittest.main()
