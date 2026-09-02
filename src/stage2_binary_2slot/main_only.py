"""Opt-in MAIN-only static-target primitives for Condition D Run A."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from src.stage2_event_tokenizer.binary_2slot import (
    OFF,
    BinaryTransition,
    compress_binary_transitions,
    state_after,
)

from .dataset import HumanIntervalPrimitive, PerformanceTimeline
from .rollout import next_binary_state


@dataclass(frozen=True)
class StaticMainTarget:
    count: int
    timing_targets: tuple[float, float]
    timing_mask: tuple[bool, bool]
    retained_transitions: tuple[BinaryTransition, ...]
    raw_human_transitions: int
    human_end_state: int


def build_static_main_target(interval: HumanIntervalPrimitive) -> StaticMainTarget:
    """Compress human crossings without model-dependent reconciliation."""

    if interval.region != "MAIN":
        raise ValueError("MAIN-only target requires a MAIN interval")
    retained = compress_binary_transitions(interval.human_transitions)
    if state_after(interval.human_start_state, retained) != interval.human_end_state:
        raise AssertionError("static max-2 target changed the human interval-end state")
    count = len(retained)
    taus = [float(event.tau) for event in retained]
    padded = (taus + [0.0, 0.0])[:2]
    return StaticMainTarget(
        count=count,
        timing_targets=(padded[0], padded[1]),
        timing_mask=(count >= 1, count >= 2),
        retained_transitions=retained,
        raw_human_transitions=len(interval.human_transitions),
        human_end_state=interval.human_end_state,
    )


def modeled_main_intervals(timeline: PerformanceTimeline) -> tuple[HumanIntervalPrimitive, ...]:
    """Return exactly [t_i,t_(i+1)) for i=1..M-1, excluding final tail."""

    if len(timeline.main) < 2:
        raise ValueError("MAIN-only Run A requires at least two distinct onsets")
    result = tuple(timeline.main[:-1])
    if [interval.main_onset_index for interval in result] != list(range(len(result))):
        raise AssertionError("MAIN-only intervals are not strictly chronological")
    if result[-1].right_seconds != timeline.main[-1].left_seconds:
        raise AssertionError("MAIN-only horizon must end exactly at t_M")
    return result


class MainOnlyStateStore:
    """Trainer-owned hard state; no PRE/POST or human-state reset exists."""

    def __init__(self) -> None:
        self._states: dict[str, int] = {}
        self._next_onset: dict[str, int] = {}

    def begin(self, performance_id: str) -> int:
        if performance_id in self._states:
            raise RuntimeError("performance has already begun")
        self._states[performance_id] = OFF
        self._next_onset[performance_id] = 0
        return OFF

    def current(self, performance_id: str) -> int:
        if performance_id not in self._states:
            raise KeyError("performance has not begun")
        return self._states[performance_id]

    def record(self, performance_id: str, onset_index: int, predicted_count: int) -> int:
        expected = self._next_onset.get(performance_id)
        if expected is None:
            raise KeyError("performance has not begun")
        if int(onset_index) != expected:
            raise RuntimeError(f"modeled onset must be chronological; expected {expected}")
        self._states[performance_id] = next_binary_state(
            self._states[performance_id], int(predicted_count)
        )
        self._next_onset[performance_id] = expected + 1
        return self._states[performance_id]

    def finish(self, performance_id: str, expected_intervals: int) -> int:
        state = self.current(performance_id)
        if self._next_onset[performance_id] != int(expected_intervals):
            raise RuntimeError("performance has missing or duplicate MAIN-only supervision")
        del self._states[performance_id]
        del self._next_onset[performance_id]
        return state


def recovery_within(
    start_agreement: Sequence[bool],
    end_agreement: Sequence[bool],
    horizons: Sequence[int] = (1, 2, 4, 8),
) -> dict[int, float]:
    """Chronological recovery fraction for mismatch-start intervals."""

    if len(start_agreement) != len(end_agreement):
        raise ValueError("start/end agreement lengths differ")
    indices = [index for index, agreement in enumerate(start_agreement) if not agreement]
    return {
        int(horizon): (
            sum(
                any(end_agreement[index : min(len(end_agreement), index + int(horizon))])
                for index in indices
            ) / len(indices)
            if indices else 1.0
        )
        for horizon in horizons
    }
