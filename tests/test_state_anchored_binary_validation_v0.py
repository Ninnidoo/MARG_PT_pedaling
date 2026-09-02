"""Focused tests for the Run B canonical binary validation adapter."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_state_anchored_binary_validation_v0 import (
    binary_confusion,
    classification_metrics,
    pattern_ids,
    pattern_metrics,
    represented_horizon_note_mask,
    transition_mapping_from_events,
)


def test_binary_accuracy_and_macro_f1() -> None:
    target = np.asarray([[0, 0, 1, 1], [0, 1, 0, 1]])
    predicted = np.asarray([[0, 1, 1, 1], [0, 1, 0, 0]])
    mask = np.ones_like(target, dtype=bool)
    metrics = classification_metrics(binary_confusion(predicted, target, mask))
    assert metrics["pedal_samples"] == 8
    assert metrics["accuracy"] == 0.75
    assert 0.0 < metrics["macro_f1"] < 1.0


def test_binary_pattern_all_16_bins_and_base2_identity() -> None:
    bits = np.asarray([[int(bit) for bit in f"{index:04b}"] for index in range(16)])
    identifiers = pattern_ids(bits)
    assert identifiers.tolist() == list(range(16))
    histogram = np.bincount(identifiers, minlength=16)
    metrics = pattern_metrics(histogram, histogram.copy())
    assert metrics["bins"] == 16
    assert metrics["js_divergence_base2"] == 0.0
    assert abs(metrics["intersection"] - 1.0) < 1e-12


def test_represented_horizon_excludes_complete_final_onset_group() -> None:
    tokens = np.zeros((6, 8), dtype=np.int64)
    tokens[:, 1] = 261 + np.asarray([10, 0, 20, 0, 30, 0])
    mask = represented_horizon_note_mask(tokens)
    assert mask.tolist() == [True, True, True, True, False, False]


def test_binary_mask_excludes_invalid_slots() -> None:
    target = np.asarray([[0, 1, 0, 1], [1, 1, 1, 1]])
    predicted = np.asarray([[0, 0, 0, 1], [0, 0, 0, 0]])
    mask = np.asarray([[True, True, True, True], [False, False, False, False]])
    confusion = binary_confusion(predicted, target, mask)
    assert int(confusion.sum()) == 4


def test_transition_mapping_is_direction_and_tick_keyed() -> None:
    mapping = transition_mapping_from_events([
        {"direction": "UP", "tick": 10, "score_position": 3, "onset_index": 4},
        {"direction": "DOWN", "tick": 20, "score_position": 5, "onset_index": 6},
    ])
    assert mapping[("UP", 10)] == {"score_position": 3, "onset_index": 4}
    assert mapping[("DOWN", 20)] == {"score_position": 5, "onset_index": 6}


def run_tests() -> None:
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            value()


if __name__ == "__main__":
    run_tests()
    print("PASS: 5 focused binary validation tests")
