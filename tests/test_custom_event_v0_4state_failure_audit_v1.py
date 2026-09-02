import copy

import numpy as np

from src.stage2_event_model.failure_audit_v1 import (
    assert_aligned_universe,
    collapse_binary_patterns,
    deterministic_digest,
    effective_changes,
    error_decomposition,
    mismatch_episodes,
    oracle_initial_replacement,
    residence_episodes,
    slot_funnel_counts,
)


def test_same_side_error():
    result = error_decomposition([0], [1])
    assert result["same_side_depth"] == 1 and result["cross_threshold"] == 0


def test_cross_threshold_error():
    result = error_decomposition([2], [1])
    assert result["same_side_depth"] == 0 and result["cross_threshold"] == 1


def test_effective_change_taxonomy():
    rows = effective_changes([0, 1, 1, 3, 2, 0])
    assert [row["type"] for row in rows] == ["TYPE_A", "TYPE_C", "TYPE_B", "TYPE_C"]


def test_binary_16_pattern_collapse():
    states = np.array([[0, 1, 2, 3], [1, 0, 3, 2]])
    assert collapse_binary_patterns(states).tolist() == [3, 3]


def test_mismatch_episode_extraction():
    rows = mismatch_episodes([0, 1, 1, 0], [0, 0, 0, 0], [0, .25, .5, .75], [0, 0, 0, 0])
    assert len(rows) == 1 and rows[0]["sample_count"] == 2
    assert rows[0]["start_cause"] == "A_CANDIDATE_CHANGED_REFERENCE_DID_NOT"


def test_residence_episode_extraction():
    rows = residence_episodes([0, 0, 1, 1, 2], [0, .2, .4, .6, .8], [0, 0, 1, 1, 2])
    assert [row["sample_count"] for row in rows] == [2, 2, 1]


def test_slot_funnel_counting():
    classes = np.array([[1, 2, 0, 4, 0, 0], [1, 2, 3, 0, 0, 0]])
    timing = np.array([[.5, .5, 0, 0, 0, 0], [.7, .2, .4, 0, 0, 0]])
    rows = slot_funnel_counts(classes, timing)
    assert [row["raw_non_none"] for row in rows] == [2, 2, 1, 1, 0, 0]
    assert sum(row["prefix_active"] for row in rows) == 5
    assert sum(row["same_time_surviving"] for row in rows) == 4


def test_oracle_initial_only_replaces_initial():
    original = [{"initial_logits": np.arange(4), "main": np.arange(3), "terminal": np.arange(2)}]
    before = copy.deepcopy(original)
    oracle = oracle_initial_replacement(original, [2])
    assert np.argmax(oracle[0]["initial_logits"]) == 2
    assert np.array_equal(oracle[0]["main"], before[0]["main"])
    assert np.array_equal(oracle[0]["terminal"], before[0]["terminal"])
    assert np.array_equal(original[0]["initial_logits"], before[0]["initial_logits"])


def test_same_aligned_universe_preservation():
    arrays = [np.zeros((7, 4), dtype=np.int8) for _ in range(3)]
    assert_aligned_universe(*arrays, notes=7)


def test_deterministic_repeat():
    value = np.arange(24, dtype=np.int16).reshape(3, 8)
    assert deterministic_digest(value) == deterministic_digest(value.copy())
