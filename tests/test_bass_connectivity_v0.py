from __future__ import annotations

import math

from src.bass_connectivity_v0 import NoteAttack, connectivity_score, earliest_anchor_groups


def score(**overrides):
    values = dict(
        keyoff_seconds=1.0,
        next_onset_seconds=2.0,
        current_pitch=40,
        next_pitch=43,
        pedal_on_at_keyoff=True,
        first_pedal_off_seconds=2.0,
    )
    values.update(overrides)
    return connectivity_score(**values)


def test_keyoff_at_pedal_off_is_zero():
    assert score(pedal_on_at_keyoff=False).score == 0.0


def test_different_bass_off_exactly_at_next_onset_is_one():
    assert score().score == 1.0


def test_different_bass_left_half_gap_is_exp_minus_half():
    assert math.isclose(score(first_pedal_off_seconds=1.5).score, math.exp(-0.5), rel_tol=0.0, abs_tol=1e-12)


def test_different_bass_right_quarter_gap_is_exp_minus_half():
    assert math.isclose(score(first_pedal_off_seconds=2.25).score, math.exp(-0.5), rel_tol=0.0, abs_tol=1e-12)


def test_same_bass_connected_to_next_is_one():
    assert score(next_pitch=40, first_pedal_off_seconds=2.5).score == 1.0


def test_no_subsequent_off_diff_zero_same_one():
    assert score(first_pedal_off_seconds=None).score == 0.0
    assert score(next_pitch=40, first_pedal_off_seconds=None).score == 1.0


def test_keyoff_at_or_after_next_is_excluded():
    result = score(keyoff_seconds=2.0)
    assert result.score is None
    assert result.status == "excluded_keyoff_at_or_after_next_bass"


def test_earliest_anchor_grouping_is_not_single_linkage():
    attacks = [
        NoteAttack(i, 0, 60 + i, 64, i, time, i)
        for i, time in enumerate((0.000, 0.015, 0.030), 1)
    ]
    groups = earliest_anchor_groups(attacks)
    assert [[round(1000 * note.onset_seconds) for note in group] for group in groups] == [
        [0, 15],
        [30],
    ]

