"""Frozen grammar-restoring prediction decoder for Custom Event Model v0.

No metric-tuned post-processing lives here. Window aggregation and MIDI writing
are intentionally outside this pure performance-level decoder.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from src.stage2_event_tokenizer.tokenizer import REPRESENTATIVES


DECODER_VERSION = "1.0.0"
DECODER_CONFIG: dict[str, Any] = {
    "decoder_version": DECODER_VERSION,
    "initial": "finite deterministic argmax; lowest class index wins ties",
    "main_prefix": "first NONE terminates six left-packed slots",
    "main_timing": "finite raw tau clipped to [0,1]",
    "main_order": "stable (tau, original_slot_index) pair sort",
    "main_nonterminal_boundary": "right-open after MIDI tick conversion",
    "main_final_boundary": "latest-note-off inclusive",
    "same_time": "last destination SET wins",
    "same_state": "absolute SET no-op suppressed",
    "state_machine": "one continuous Initial->Main->Terminal state",
    "terminal_prefix": "first NONE terminates four left-packed slots",
    "terminal_timing": "finite z clamped >=0; gap=expm1(z)",
    "terminal_order": "slot order; cumulative inter-event gaps; never sorted",
    "terminal_noop": "clock advances before same-state suppression",
    "terminal_boundary": "discrete active slots strictly after latest note-off and increasing",
    "forbidden": [
        "confidence_threshold", "calibration", "temperature", "sampling",
        "beam_search", "smoothing", "minimum_duration", "metric_tuned_postprocessing",
    ],
}


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


DECODER_ID = canonical_json_sha256(DECODER_CONFIG)


class PredictionDecodeError(ValueError):
    """Prediction corruption or a grammar-incompatible input."""


@dataclass(frozen=True)
class MainIntervalBoundary:
    index: int
    left_time: float
    right_time: float
    is_final_main_interval: bool


@dataclass(frozen=True)
class PerformanceTimeline:
    first_onset_time: float
    latest_note_off_time: float
    main_intervals: tuple[MainIntervalBoundary, ...]


@dataclass(frozen=True)
class DecodedPedalEvent:
    time: float
    destination_state: int
    cc64_value: int
    source: str
    main_interval_index: int | None
    original_slot_index: int | None
    tau: float | None
    terminal_gap_log: float | None
    terminal_gap_seconds: float | None
    ordering_key: tuple[float, int, int, int]


@dataclass(frozen=True)
class DecodedPedalTimeline:
    initial_state: int
    initial_event: DecodedPedalEvent
    active_slot_events: tuple[DecodedPedalEvent, ...]
    emitted_events: tuple[DecodedPedalEvent, ...]
    final_pedal_state: int
    diagnostics: Mapping[str, int]

    @property
    def events(self) -> tuple[DecodedPedalEvent, ...]:
        return (self.initial_event, *self.emitted_events)

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_state": self.initial_state,
            "initial_event": asdict(self.initial_event),
            "active_slot_events": [asdict(event) for event in self.active_slot_events],
            "emitted_events": [asdict(event) for event in self.emitted_events],
            "final_pedal_state": self.final_pedal_state,
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class MainIntervalTicks:
    index: int
    left_tick: int
    right_tick: int
    is_final_main_interval: bool


@dataclass(frozen=True)
class QuantizedPedalEvent:
    tick: int
    destination_state: int
    cc64_value: int
    source: str
    main_interval_index: int | None
    original_slot_index: int | None
    ordering_key: tuple[int, int, int, int]


@dataclass(frozen=True)
class QuantizedPedalTimeline:
    events: tuple[QuantizedPedalEvent, ...]
    final_pedal_state: int
    same_tick_main_collapsed: int
    same_state_suppressed: int


def _array(value: Any, *, name: str, shape: tuple[int, ...]) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    result = np.asarray(value)
    if result.shape != shape:
        raise PredictionDecodeError(f"{name} shape {result.shape}, expected {shape}")
    if not np.issubdtype(result.dtype, np.number):
        raise PredictionDecodeError(f"{name} is not numeric")
    if not bool(np.isfinite(result).all()):
        bad = np.argwhere(~np.isfinite(result))[0].tolist()
        raise PredictionDecodeError(f"{name} contains NaN/Inf at index {bad}")
    return result


def _argmax(values: np.ndarray) -> int:
    # np.argmax deterministically returns the lowest class index on a tie.
    return int(np.argmax(values))


def _event(
    *, time: float, state: int, source: str, interval: int | None,
    slot: int | None, tau: float | None = None, gap_log: float | None = None,
    gap_seconds: float | None = None,
) -> DecodedPedalEvent:
    priority = {"INITIAL": 0, "MAIN": 1, "TERMINAL": 2}[source]
    return DecodedPedalEvent(
        float(time), int(state), int(REPRESENTATIVES[state]), source, interval, slot,
        tau, gap_log, gap_seconds,
        (float(time), priority, -1 if interval is None else int(interval),
         -1 if slot is None else int(slot)),
    )


def _validate_timeline(timeline: PerformanceTimeline, count: int) -> None:
    if len(timeline.main_intervals) != count or not count:
        raise PredictionDecodeError("main prediction/timeline interval count mismatch")
    values = [timeline.first_onset_time, timeline.latest_note_off_time]
    for expected, interval in enumerate(timeline.main_intervals):
        values.extend((interval.left_time, interval.right_time))
        if interval.index != expected:
            raise PredictionDecodeError("main interval indices are not consecutive")
        if not interval.left_time < interval.right_time:
            raise PredictionDecodeError(f"non-positive main interval {interval.index}")
        if interval.is_final_main_interval != (expected == count - 1):
            raise PredictionDecodeError("only final main interval may be inclusive")
    if not all(math.isfinite(float(value)) for value in values):
        raise PredictionDecodeError("performance timeline contains NaN/Inf")
    if timeline.main_intervals[0].left_time != timeline.first_onset_time:
        raise PredictionDecodeError("first interval does not start at first onset")
    if timeline.main_intervals[-1].right_time != timeline.latest_note_off_time:
        raise PredictionDecodeError("final interval does not end at latest note-off")


def decode_predictions_v1(
    *, initial_logits: Any, main_event_logits: Any, main_timing_predictions: Any,
    terminal_event_logits: Any, terminal_timing_predictions: Any,
    timeline: PerformanceTimeline,
) -> DecodedPedalTimeline:
    """Decode one performance-level prediction with the frozen grammar."""

    count = len(timeline.main_intervals)
    initial = _array(initial_logits, name="initial_logits", shape=(4,))
    main_logits = _array(main_event_logits, name="main_event_logits", shape=(count, 6, 5))
    main_timing = _array(main_timing_predictions, name="main_timing_predictions", shape=(count, 6))
    terminal_logits = _array(terminal_event_logits, name="terminal_event_logits", shape=(4, 5))
    terminal_timing = _array(terminal_timing_predictions, name="terminal_timing_predictions", shape=(4,))
    _validate_timeline(timeline, count)

    initial_state = _argmax(initial)
    initial_event = _event(time=timeline.first_onset_time, state=initial_state,
                           source="INITIAL", interval=None, slot=None)
    d = {key: 0 for key in (
        "main_active_slots", "main_ignored_after_first_none", "main_tau_clamp_low",
        "main_tau_clamp_high", "main_raw_order_violations", "main_same_time_collapsed",
        "terminal_active_slots", "terminal_ignored_after_first_none",
        "terminal_negative_z_clamp", "same_state_suppressed_main",
        "same_state_suppressed_terminal",
    )}
    active: list[DecodedPedalEvent] = []
    main_by_interval: list[list[DecodedPedalEvent]] = []
    for interval_index, interval in enumerate(timeline.main_intervals):
        classes = [_argmax(main_logits[interval_index, slot]) for slot in range(6)]
        active_count = classes.index(0) if 0 in classes else 6
        if active_count < 6:
            d["main_ignored_after_first_none"] += sum(c != 0 for c in classes[active_count + 1:])
        raw_taus = [float(main_timing[interval_index, slot]) for slot in range(active_count)]
        d["main_raw_order_violations"] += sum(
            raw_taus[i] < raw_taus[i - 1] for i in range(1, len(raw_taus))
        )
        pairs: list[DecodedPedalEvent] = []
        for slot in range(active_count):
            raw_tau = raw_taus[slot]
            d["main_tau_clamp_low"] += int(raw_tau < 0.0)
            d["main_tau_clamp_high"] += int(raw_tau > 1.0)
            tau = min(1.0, max(0.0, raw_tau))
            time = interval.left_time + tau * (interval.right_time - interval.left_time)
            pairs.append(_event(time=time, state=classes[slot] - 1, source="MAIN",
                                interval=interval.index, slot=slot, tau=tau))
        pairs.sort(key=lambda item: (item.tau, item.original_slot_index))
        d["main_active_slots"] += len(pairs)
        active.extend(pairs)
        main_by_interval.append(pairs)

    emitted: list[DecodedPedalEvent] = []
    current = initial_state
    for pairs in main_by_interval:
        index = 0
        while index < len(pairs):
            end = index + 1
            while end < len(pairs) and pairs[end].time == pairs[index].time:
                end += 1
            effective = pairs[end - 1]
            d["main_same_time_collapsed"] += end - index - 1
            if effective.destination_state == current:
                d["same_state_suppressed_main"] += 1
            else:
                emitted.append(effective)
                current = effective.destination_state
            index = end

    classes = [_argmax(terminal_logits[slot]) for slot in range(4)]
    active_count = classes.index(0) if 0 in classes else 4
    if active_count < 4:
        d["terminal_ignored_after_first_none"] = sum(c != 0 for c in classes[active_count + 1:])
    clock = float(timeline.latest_note_off_time)
    for slot in range(active_count):
        raw_z = float(terminal_timing[slot])
        d["terminal_negative_z_clamp"] += int(raw_z < 0.0)
        z = max(0.0, raw_z)
        gap = math.expm1(z)
        clock += gap  # No-op SETs still advance the terminal clock.
        item = _event(time=clock, state=classes[slot] - 1, source="TERMINAL",
                      interval=None, slot=slot, gap_log=z, gap_seconds=gap)
        d["terminal_active_slots"] += 1
        active.append(item)
        if item.destination_state == current:
            d["same_state_suppressed_terminal"] += 1
        else:
            emitted.append(item)
            current = item.destination_state

    return DecodedPedalTimeline(initial_state, initial_event, tuple(active),
                                tuple(emitted), current, d)


def enforce_main_tick_boundary(
    candidate_tick: int, *, left_tick: int, right_tick: int,
    is_final_main_interval: bool,
) -> int:
    """Enforce [left,right) for non-final and [left,right] for final main."""
    left_tick, right_tick = int(left_tick), int(right_tick)
    if right_tick <= left_tick:
        raise PredictionDecodeError("non-positive MIDI main interval")
    upper = right_tick if is_final_main_interval else right_tick - 1
    return min(upper, max(left_tick, int(candidate_tick)))


def strict_terminal_tick(candidate_tick: int, previous_active_tick: int) -> int:
    """One-tick grammar guard; continuous decoded gaps remain unchanged."""
    return max(int(candidate_tick), int(previous_active_tick) + 1)


def quantize_decoded_timeline_v1(
    decoded: DecodedPedalTimeline, *, seconds_to_tick: Callable[[float], int],
    first_onset_tick: int, latest_note_off_tick: int,
    main_intervals: Sequence[MainIntervalTicks],
) -> QuantizedPedalTimeline:
    """Apply MIDI boundaries and global same-tick restoration without writing."""
    boundaries = {interval.index: interval for interval in main_intervals}
    main: list[tuple[int, DecodedPedalEvent]] = []
    terminal: list[tuple[int, DecodedPedalEvent]] = []
    previous_terminal_tick = int(latest_note_off_tick)
    for item in decoded.active_slot_events:
        candidate = int(seconds_to_tick(item.time))
        if item.source == "MAIN":
            if item.main_interval_index not in boundaries:
                raise PredictionDecodeError("missing MIDI main interval boundary")
            boundary = boundaries[int(item.main_interval_index)]
            tick = enforce_main_tick_boundary(
                candidate, left_tick=boundary.left_tick, right_tick=boundary.right_tick,
                is_final_main_interval=boundary.is_final_main_interval,
            )
            main.append((tick, item))
        else:
            tick = strict_terminal_tick(candidate, previous_terminal_tick)
            previous_terminal_tick = tick
            terminal.append((tick, item))
    main.sort(key=lambda row: (row[0], row[1].time,
                               int(row[1].main_interval_index or 0),
                               int(row[1].original_slot_index or 0)))
    output = [QuantizedPedalEvent(
        int(first_onset_tick), decoded.initial_state, int(REPRESENTATIVES[decoded.initial_state]),
        "INITIAL", None, None, (int(first_onset_tick), 0, -1, -1),
    )]
    current, collapsed, suppressed, index = decoded.initial_state, 0, 0, 0
    while index < len(main):
        end = index + 1
        while end < len(main) and main[end][0] == main[index][0]:
            end += 1
        tick, effective = main[end - 1]
        collapsed += end - index - 1
        if effective.destination_state == current:
            suppressed += 1
        else:
            output.append(QuantizedPedalEvent(
                tick, effective.destination_state, effective.cc64_value, "MAIN",
                effective.main_interval_index, effective.original_slot_index,
                (tick, 1, int(effective.main_interval_index or 0),
                 int(effective.original_slot_index or 0)),
            ))
            current = effective.destination_state
        index = end
    for tick, item in terminal:
        if item.destination_state == current:
            suppressed += 1
            continue
        output.append(QuantizedPedalEvent(
            tick, item.destination_state, item.cc64_value, "TERMINAL", None,
            item.original_slot_index, (tick, 2, -1, int(item.original_slot_index or 0)),
        ))
        current = item.destination_state
    return QuantizedPedalTimeline(tuple(output), current, collapsed, suppressed)


__all__ = [
    "DECODER_CONFIG", "DECODER_ID", "DECODER_VERSION", "DecodedPedalEvent",
    "DecodedPedalTimeline", "MainIntervalBoundary", "MainIntervalTicks",
    "PerformanceTimeline", "PredictionDecodeError", "QuantizedPedalEvent",
    "QuantizedPedalTimeline", "decode_predictions_v1",
    "enforce_main_tick_boundary", "quantize_decoded_timeline_v1",
    "strict_terminal_tick",
]
