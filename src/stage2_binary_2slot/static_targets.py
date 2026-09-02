"""Model-independent PRE/MAIN/POST targets for Condition D.

The only synthetic event in this module is the frozen, performance-derived
PRE initial synchronization event.  No function accepts model predictions or
a free-running model state, so these targets cannot mutate during training.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.stage2_event_tokenizer.binary_2slot import (
    OFF,
    ON,
    BinaryTransition,
    compress_binary_transitions,
    correction_transition,
    state_after,
)

from .dataset import HumanIntervalPrimitive, PerformanceTimeline
from .rollout import next_binary_state


@dataclass(frozen=True)
class StaticIntervalTarget:
    count: int
    timing_targets: tuple[float, float]
    timing_mask: tuple[bool, bool]
    retained_transitions: tuple[BinaryTransition, ...]
    static_initial_correction_inserted: bool
    static_initial_correction_retained: bool
    retained_real_human_transitions: int
    raw_human_transitions: int
    human_end_state: int


def _target(
    interval: HumanIntervalPrimitive,
    sequence: tuple[BinaryTransition, ...],
    *,
    application_start_state: int,
    correction_inserted: bool,
) -> StaticIntervalTarget:
    retained = compress_binary_transitions(sequence)
    if state_after(application_start_state, retained) != interval.human_end_state:
        raise AssertionError("static max-2 target changed the human interval-end state")
    count = len(retained)
    taus = [float(event.tau) for event in retained]
    padded = (taus + [0.0, 0.0])[:2]
    return StaticIntervalTarget(
        count=count,
        timing_targets=(padded[0], padded[1]),
        timing_mask=(count >= 1, count >= 2),
        retained_transitions=retained,
        static_initial_correction_inserted=correction_inserted,
        static_initial_correction_retained=any(event.synthetic for event in retained),
        retained_real_human_transitions=sum(not event.synthetic for event in retained),
        raw_human_transitions=len(interval.human_transitions),
        human_end_state=interval.human_end_state,
    )


def build_static_pre_target(interval: HumanIntervalPrimitive) -> StaticIntervalTarget:
    """Build PRE from fixed OFF plus optional MIDI-derived tau-zero DOWN.

    The correction precedes actual events at the same timestamp and receives no
    reserved slot before the frozen parity-preserving compression is applied.
    """

    if interval.region != "PRE":
        raise ValueError("static PRE target requires a PRE interval")
    correction_inserted = interval.human_start_state == ON
    sequence = tuple(interval.human_transitions)
    if correction_inserted:
        sequence = (correction_transition(OFF),) + sequence
    return _target(
        interval,
        sequence,
        application_start_state=OFF,
        correction_inserted=correction_inserted,
    )


def build_static_human_target(interval: HumanIntervalPrimitive) -> StaticIntervalTarget:
    """Compress a MAIN or POST human sequence without any correction."""

    if interval.region not in {"MAIN", "POST"}:
        raise ValueError("static human target requires a MAIN or POST interval")
    return _target(
        interval,
        tuple(interval.human_transitions),
        application_start_state=interval.human_start_state,
        correction_inserted=False,
    )


def build_static_timeline_targets(
    timeline: PerformanceTimeline,
) -> tuple[StaticIntervalTarget, ...]:
    """Return immutable PRE, every MAIN (including final tail), and POST targets."""

    intervals = timeline.represented_intervals
    targets = (
        build_static_pre_target(timeline.pre),
        *(build_static_human_target(interval) for interval in timeline.main),
        build_static_human_target(timeline.post),
    )
    if len(targets) != len(intervals):
        raise AssertionError("static target/interval cardinality mismatch")

    state = OFF
    for interval, target in zip(intervals, targets):
        if interval.region != "PRE" and state != interval.human_start_state:
            raise AssertionError("static target chain did not enter human interval state")
        state = next_binary_state(state, target.count)
        if state != interval.human_end_state:
            raise AssertionError("static target chain failed the final-state invariant")
    return targets

