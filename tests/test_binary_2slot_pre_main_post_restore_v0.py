#!/usr/bin/env python3
"""Focused executable regression tests for restored Condition D timeline."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage2_binary_2slot.dataset import (
    HumanIntervalPrimitive,
    PerformanceTimeline,
    _interval_transitions,
)
from src.stage2_binary_2slot.rollout import PerformanceStateStore
from src.stage2_binary_2slot.static_targets import (
    build_static_human_target,
    build_static_pre_target,
    build_static_timeline_targets,
)
from src.stage2_event_tokenizer.binary_2slot import (
    DOWN,
    OFF,
    ON,
    UP,
    BinaryTransition,
    interval_region,
    state_after,
)


def primitive(region, index, left, right, state, events=(), main=None):
    return HumanIntervalPrimitive(region, index, main, left, right, state, tuple(events))


def test_a_pre_left_off_without_transition_stays_off():
    target = build_static_pre_target(primitive("PRE", 0, 9.0, 10.0, OFF))
    assert target.count == 0
    assert not target.static_initial_correction_inserted
    assert state_after(OFF, target.retained_transitions) == OFF


def test_b_pre_left_on_inserts_static_down_at_zero():
    target = build_static_pre_target(primitive("PRE", 0, 9.0, 10.0, ON))
    assert target.count == 1
    assert target.static_initial_correction_inserted
    assert target.static_initial_correction_retained
    assert target.retained_transitions == (BinaryTransition(DOWN, 0.0, synthetic=True),)
    assert state_after(OFF, target.retained_transitions) == ON


def test_c_pre_correction_precedes_real_crossings_and_preserves_parity():
    human = (BinaryTransition(UP, 0.0), BinaryTransition(DOWN, 0.75))
    interval = primitive("PRE", 0, 9.0, 10.0, ON, human)
    target = build_static_pre_target(interval)
    # DOWN(sync), UP(actual), DOWN(actual) -> odd last-one retention.
    assert target.count == 1
    assert target.static_initial_correction_inserted
    assert not target.static_initial_correction_retained
    assert target.retained_transitions == (human[-1],)
    assert state_after(OFF, target.retained_transitions) == interval.human_end_state == ON


def test_d_event_exactly_at_t1_is_first_main_tau_zero():
    crossings = ((10, DOWN, 10.0),)
    assert _interval_transitions(crossings, 9.0, 10.0, include_right=False) == ()
    main = _interval_transitions(crossings, 10.0, 11.0, include_right=False)
    assert len(main) == 1 and main[0].tau == 0.0
    assert interval_region(10.0, 10.0, 20.0) == "MAIN"


def test_e_event_exactly_at_note_end_is_post():
    crossings = ((20, UP, 20.0),)
    assert _interval_transitions(crossings, 19.0, 20.0, include_right=False) == ()
    post = _interval_transitions(crossings, 20.0, 21.0, include_right=True)
    assert len(post) == 1 and post[0].tau == 0.0


def test_f_event_exactly_at_post_right_is_tau_one():
    post = _interval_transitions(((21, UP, 21.0),), 20.0, 21.0, include_right=True)
    assert len(post) == 1 and post[0].tau == 1.0
    assert interval_region(21.0, 10.0, 20.0) == "POST"


def test_g_event_beyond_post_horizon_is_truncated():
    assert _interval_transitions(((22, UP, 21.000001),), 20.0, 21.0, include_right=True) == ()
    assert interval_region(21.000001, 10.0, 20.0) == "OUTSIDE"


def test_h_static_target_does_not_accept_or_depend_on_prediction():
    interval = primitive("MAIN", 1, 10.0, 11.0, OFF, (BinaryTransition(DOWN, 0.5),), 0)
    first = build_static_human_target(interval)
    for arbitrary_prediction in (0, 1, 2):
        assert build_static_human_target(interval) == first
        assert arbitrary_prediction in {0, 1, 2}


def test_i_state_is_continuous_pre_main_post():
    pre = primitive("PRE", 0, 9.0, 10.0, ON)
    main0 = primitive("MAIN", 1, 10.0, 11.0, ON, (BinaryTransition(UP, 0.5),), 0)
    final_main = primitive("MAIN", 2, 11.0, 12.0, OFF, (), 1)
    post = primitive("POST", 3, 12.0, 13.0, OFF, (BinaryTransition(DOWN, 1.0),))
    timeline = PerformanceTimeline("p", "p.mid", pre, (main0, final_main), post)
    targets = build_static_timeline_targets(timeline)
    store = PerformanceStateStore()
    assert store.begin("p") == OFF
    assert store.record_interval("p", "PRE", targets[0].count) == ON
    assert store.record_interval("p", "MAIN", targets[1].count, main_onset_index=0) == OFF
    assert store.record_interval("p", "MAIN", targets[2].count, main_onset_index=1) == OFF
    assert store.record_interval("p", "POST", targets[3].count) == ON
    assert store.finish("p", expected_main_count=2) == ON


def run_directly():
    tests = {name: value for name, value in globals().items() if name.startswith("test_")}
    result = {}
    for name, function in sorted(tests.items()):
        function()
        result[name] = "passed"
    return result


if __name__ == "__main__":
    print(run_directly())
