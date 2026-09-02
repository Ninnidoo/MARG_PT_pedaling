"""Reusable MAIN-only targets for State-Anchored Transition v1.

MIDI parsing, distinct-onset extraction, threshold-64 crossings, and
right-open interval ownership remain owned by the existing Binary 2-Slot data
adapter.  This module only converts those canonical human primitives into
State, Mode, and normalized timing targets.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.stage2_binary_2slot.dataset import HumanIntervalPrimitive, PerformanceTimeline
from src.stage2_event_tokenizer.binary_2slot import (
    OFF,
    ON,
    BinaryTransition,
    compress_binary_transitions,
    state_after,
)


HOLD = "HOLD"
CHANGE = "CHANGE"
RETURN = "RETURN"
MODE_NAMES = (HOLD, CHANGE, RETURN)


@dataclass(frozen=True)
class StateAnchoredIntervalTarget:
    """Static target for one MAIN interval ``[t_i, t_(i+1))``."""

    onset_index: int
    left_seconds: float
    right_seconds: float
    start_state: int
    end_state: int
    mode: str
    timing_targets: tuple[float, float]
    timing_mask: tuple[bool, bool]
    raw_transitions: tuple[BinaryTransition, ...]
    retained_transitions: tuple[BinaryTransition, ...]
    endpoint_state_preserved: bool

    @property
    def raw_transition_count(self) -> int:
        return len(self.raw_transitions)

    @property
    def retained_transition_count(self) -> int:
        return len(self.retained_transitions)


@dataclass(frozen=True)
class StateAnchoredPerformanceTargets:
    """All onset States and the ``M-1`` modeled interval targets."""

    performance_id: str
    onset_states: tuple[int, ...]
    intervals: tuple[StateAnchoredIntervalTarget, ...]


def _validate_chronological(transitions: tuple[BinaryTransition, ...]) -> None:
    previous = -1.0
    for event in transitions:
        tau = float(event.tau)
        if not 0.0 <= tau < 1.0:
            raise AssertionError(f"MAIN timing must satisfy 0 <= tau < 1, got {tau}")
        if tau < previous:
            raise AssertionError("MAIN transitions are not chronological")
        previous = tau


def build_interval_target(
    interval: HumanIntervalPrimitive,
    *,
    next_onset_state: int,
) -> StateAnchoredIntervalTarget:
    """Build one immutable State/Mode/Timing target without reconciliation."""

    if interval.region != "MAIN" or interval.main_onset_index is None:
        raise ValueError("State-Anchored v1 requires a MAIN onset interval")
    if next_onset_state not in {OFF, ON}:
        raise ValueError("next onset state must be OFF or ON")

    raw = tuple(interval.human_transitions)
    _validate_chronological(raw)
    raw_end_state = state_after(interval.human_start_state, raw)
    if raw_end_state != next_onset_state:
        raise AssertionError("raw trajectory endpoint differs from S_(i+1)")

    retained = compress_binary_transitions(raw)
    _validate_chronological(retained)
    retained_end_state = state_after(interval.human_start_state, retained)
    endpoint_preserved = retained_end_state == raw_end_state
    if not endpoint_preserved:
        raise AssertionError("compression changed the raw trajectory endpoint state")

    count = len(retained)
    mode = MODE_NAMES[count]
    state_flips = interval.human_start_state != next_onset_state
    if (mode == CHANGE) != state_flips:
        raise AssertionError("CHANGE iff state flip invariant failed")
    if (mode in {HOLD, RETURN}) != (not state_flips):
        raise AssertionError("HOLD/RETURN iff same state invariant failed")

    values = tuple(float(event.tau) for event in retained)
    padded = (values + (0.0, 0.0))[:2]
    return StateAnchoredIntervalTarget(
        onset_index=int(interval.main_onset_index),
        left_seconds=float(interval.left_seconds),
        right_seconds=float(interval.right_seconds),
        start_state=int(interval.human_start_state),
        end_state=int(next_onset_state),
        mode=mode,
        timing_targets=(padded[0], padded[1]),
        timing_mask=(count >= 1, count >= 2),
        raw_transitions=raw,
        retained_transitions=retained,
        endpoint_state_preserved=endpoint_preserved,
    )


def build_performance_targets(
    timeline: PerformanceTimeline,
) -> StateAnchoredPerformanceTargets:
    """Build States at all ``M`` onsets and targets for only ``M-1`` intervals."""

    onset_states = tuple(int(interval.human_start_state) for interval in timeline.main)
    if any(state not in {OFF, ON} for state in onset_states):
        raise AssertionError("onset state is not binary")
    intervals = tuple(
        build_interval_target(interval, next_onset_state=onset_states[index + 1])
        for index, interval in enumerate(timeline.main[:-1])
    )
    if len(intervals) != max(0, len(onset_states) - 1):
        raise AssertionError("State/interval cardinality mismatch")
    if [target.onset_index for target in intervals] != list(range(len(intervals))):
        raise AssertionError("modeled MAIN intervals are not chronological")
    return StateAnchoredPerformanceTargets(
        performance_id=timeline.performance_id,
        onset_states=onset_states,
        intervals=intervals,
    )
