#!/usr/bin/env python3
"""Focused executable regression tests for Run A MAIN-only semantics."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage2_binary_2slot.dataset import HumanIntervalPrimitive, PerformanceTimeline
from src.stage2_binary_2slot.main_only import (
    MainOnlyStateStore,
    build_static_main_target,
    modeled_main_intervals,
    recovery_within,
)
from src.stage2_event_tokenizer.binary_2slot import BinaryTransition, DOWN, OFF, ON, UP


def interval(index, left, right, state, events=()):
    return HumanIntervalPrimitive("MAIN", index + 1, index, left, right, state, tuple(events))


def test_static_target_never_adds_mismatch_correction():
    value = interval(0, 0.0, 1.0, ON, ())
    target = build_static_main_target(value)
    assert target.count == 0 and target.timing_mask == (False, False)


def test_static_compression_and_timing_are_frozen():
    events = (
        BinaryTransition(DOWN, 0.1), BinaryTransition(UP, 0.2),
        BinaryTransition(DOWN, 0.3), BinaryTransition(UP, 0.4),
    )
    target = build_static_main_target(interval(0, 0.0, 1.0, OFF, events))
    assert target.count == 2 and target.timing_targets == (0.3, 0.4)


def test_modeled_horizon_excludes_final_tail():
    main = (
        interval(0, 0.0, 1.0, OFF),
        interval(1, 1.0, 2.0, OFF),
        interval(2, 2.0, 3.0, OFF),
    )
    timeline = PerformanceTimeline("x", "x.mid", None, main, None)  # type: ignore[arg-type]
    modeled = modeled_main_intervals(timeline)
    assert len(modeled) == 2 and modeled[-1].right_seconds == 2.0


def test_state_store_starts_off_and_survives_windows():
    store = MainOnlyStateStore()
    assert store.begin("p") == OFF
    assert store.record("p", 0, 1) == ON
    assert store.record("p", 1, 2) == ON
    assert store.record("p", 2, 1) == OFF
    assert store.finish("p", 3) == OFF


def test_recovery_uses_full_chronological_trajectory():
    result = recovery_within([False, False, True, False], [False, True, True, False])
    assert result[1] == 1 / 3 and result[2] == 2 / 3 and result[4] == 2 / 3


def run_directly():
    tests = {name: value for name, value in globals().items() if name.startswith("test_")}
    result = {}
    for name, function in sorted(tests.items()):
        function()
        result[name] = "passed"
    return result


if __name__ == "__main__":
    print(run_directly())
