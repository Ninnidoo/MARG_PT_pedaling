#!/usr/bin/env python3
"""Focused regressions for direct same-human Run B metric mapping."""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_state_anchored_transition_v1_full import (  # noqa: E402
    direct_same_performance_trajectory,
    interval_event_seconds,
)
from src.stage2_event_tokenizer.binary_2slot import BinaryTransition, DOWN  # noqa: E402
from src.stage2_state_anchored.targets import StateAnchoredIntervalTarget  # noqa: E402


def _interval(event: BinaryTransition) -> StateAnchoredIntervalTarget:
    return StateAnchoredIntervalTarget(
        onset_index=0,
        left_seconds=2.0,
        right_seconds=2.5,
        start_state=0,
        end_state=1,
        mode="CHANGE",
        timing_targets=(float(event.tau), 0.0),
        timing_mask=(True, False),
        raw_transitions=(event,),
        retained_transitions=(event,),
        endpoint_state_preserved=True,
    )


def test_schema_has_boundaries_but_no_legacy_duration_property() -> None:
    names = {field.name for field in dataclasses.fields(StateAnchoredIntervalTarget)}
    assert {"left_seconds", "right_seconds", "timing_targets", "mode", "start_state", "end_state"} <= names
    assert "duration_seconds" not in names
    assert not hasattr(_interval(BinaryTransition(DOWN, 0.25)), "duration_seconds")


def test_duration_is_derived_from_canonical_onset_boundaries() -> None:
    event = BinaryTransition(DOWN, 0.4)
    assert abs(interval_event_seconds(_interval(event), event) - 2.2) < 1e-12


def test_exact_tau_zero_absolute_timing_is_preserved() -> None:
    event = BinaryTransition(DOWN, 0.0)
    seconds = interval_event_seconds(_interval(event), event)
    assert seconds == 2.0
    mapped = direct_same_performance_trajectory(
        ({"direction": DOWN, "seconds": seconds},), (2.0, 2.5),
    )
    assert mapped["transitions"][0]["onset_index"] == 0
    assert mapped["transitions"][0]["score_position"] == 0


def test_direct_mapping_uses_same_performance_indices_and_nearest_onset() -> None:
    mapped = direct_same_performance_trajectory(
        (
            {"direction": DOWN, "seconds": 0.0},
            {"direction": "UP", "seconds": 0.8},
            {"direction": DOWN, "seconds": 1.5},
        ),
        (0.0, 1.0, 2.0),
    )
    assert [item["score_position"] for item in mapped["transitions"]] == [0, 1, 1]
    assert all(item["score_position"] == item["onset_index"] for item in mapped["transitions"])


def run_directly() -> dict[str, str]:
    tests = {name: value for name, value in globals().items() if name.startswith("test_")}
    result = {}
    for name, function in sorted(tests.items()):
        function()
        result[name] = "passed"
    return result


if __name__ == "__main__":
    print(run_directly())
