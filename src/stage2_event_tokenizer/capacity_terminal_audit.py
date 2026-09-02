"""Isolated capacity/terminal audit helpers for the immutable v0 tokenizer.

This module does not alter v0 encoding semantics.  It only projects an already
encoded interval's ``all_events`` onto diagnostic fixed-capacity views.
"""

from __future__ import annotations

import bisect
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import mido

from .decoder import DecodedControl
from .tokenizer import (
    NONE_TOKEN,
    REPRESENTATIVES,
    EncodedPerformance,
    EventToken,
    apply_tokens,
    parse_raw_midi,
)

CAPACITIES = (1, 2, 3, 4, 6, 8)
VARIANT_ORDER = (
    "k1_last1",
    "k2_v0_odd_even",
    "k2_last2",
    "k3_last3",
    "k4_last4",
    "k6_last6",
    "k8_last8",
    "unlimited",
)
VARIANT_CAPACITY = {
    "k1_last1": 1,
    "k2_v0_odd_even": 2,
    "k2_last2": 2,
    "k3_last3": 3,
    "k4_last4": 4,
    "k6_last6": 6,
    "k8_last8": 8,
    "unlimited": None,
}
TAIL_SECONDS = (0.05, 0.10, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 5.00)


def retain_last_k(events: Sequence[EventToken], capacity: int) -> tuple[EventToken, ...]:
    """Keep all events when possible, otherwise the chronological final K."""

    if capacity <= 0:
        raise ValueError("capacity must be positive")
    values = tuple(events)
    return values if len(values) <= capacity else values[-capacity:]


def retained_events(interval, variant: str) -> tuple[EventToken, ...]:
    if variant == "unlimited":
        return tuple(interval.all_events)
    if variant == "k2_v0_odd_even":
        return tuple(slot for slot in interval.slots if slot.event_class)
    capacity = VARIANT_CAPACITY.get(variant)
    if capacity is None:
        raise KeyError(f"unknown capacity variant: {variant}")
    return retain_last_k(interval.all_events, capacity)


def padded_slots(interval, variant: str) -> tuple[EventToken, ...]:
    capacity = VARIANT_CAPACITY.get(variant)
    if capacity is None:
        raise ValueError("Unlimited has no fixed slots")
    if variant == "k2_v0_odd_even":
        return tuple(interval.slots)
    values = retained_events(interval, variant)
    return values + (NONE_TOKEN,) * (capacity - len(values))


def assert_final_state_preserved(interval, variant: str) -> None:
    expected = apply_tokens(interval.start_state, interval.all_events)
    actual = apply_tokens(interval.start_state, retained_events(interval, variant))
    if expected != actual:
        raise AssertionError(f"{variant} changed interval final state")


def event_count_bin(value: int, *, maximum_exact: int, tail_label: str) -> str:
    return str(value) if value <= maximum_exact else tail_label


def nearest_rank_quantiles(counts: Mapping[int, int], levels: Iterable[float]) -> dict[str, int]:
    """Return deterministic nearest-rank quantiles from an integer histogram."""

    ordered = sorted((int(value), int(count)) for value, count in counts.items() if count > 0)
    total = sum(count for _, count in ordered)
    if not total:
        return {quantile_name(level): 0 for level in levels}
    cumulative_values: list[int] = []
    cumulative = 0
    for _, count in ordered:
        cumulative += count
        cumulative_values.append(cumulative)
    result: dict[str, int] = {}
    for level in levels:
        rank = max(1, int(__import__("math").ceil(float(level) * total)))
        index = bisect.bisect_left(cumulative_values, rank)
        result[quantile_name(level)] = ordered[index][0]
    return result


def quantile_name(level: float) -> str:
    value = float(level) * 100
    return f"p{int(value)}" if value.is_integer() else f"p{str(value).replace('.', 'p')}"


def terminal_coverage(delays_by_performance: Sequence[Sequence[float]], tails: Sequence[float] = TAIL_SECONDS) -> list[dict[str, float | int]]:
    affected = [tuple(float(value) for value in delays) for delays in delays_by_performance if delays]
    if any(value < 0 for delays in affected for value in delays):
        raise AssertionError("post-note-off delay is negative")
    event_total = sum(len(delays) for delays in affected)
    rows: list[dict[str, float | int]] = []
    previous_events = previous_performances = -1.0
    for tail in tails:
        captured = sum(value <= tail + 1e-12 for delays in affected for value in delays)
        fully = sum(all(value <= tail + 1e-12 for value in delays) for delays in affected)
        event_fraction = captured / event_total if event_total else 1.0
        performance_fraction = fully / len(affected) if affected else 1.0
        if event_fraction < previous_events or performance_fraction < previous_performances:
            raise AssertionError("terminal coverage curve is not monotonic")
        previous_events, previous_performances = event_fraction, performance_fraction
        rows.append(
            {
                "tail_seconds": float(tail),
                "post_noteoff_events": event_total,
                "captured_events": captured,
                "event_coverage": event_fraction,
                "affected_performances": len(affected),
                "fully_covered_affected_performances": fully,
                "affected_performance_full_coverage": performance_fraction,
            }
        )
    return rows


def tick_to_seconds(path: str | Path):
    """Build a deterministic SMF tempo-map converter for arbitrary raw ticks."""

    midi = mido.MidiFile(str(path), clip=False)
    tempo_messages: list[tuple[int, int, int, int]] = []
    for track_index, track in enumerate(midi.tracks):
        tick = 0
        for message_index, message in enumerate(track):
            tick += int(message.time)
            if message.type == "set_tempo":
                tempo_messages.append((tick, track_index, message_index, int(message.tempo)))
    grouped: dict[int, int] = {0: 500000}
    for tick, _, _, tempo in sorted(tempo_messages):
        grouped[tick] = tempo
    ticks = sorted(grouped)
    tempos = [grouped[tick] for tick in ticks]
    cumulative = [0.0]
    for index in range(1, len(ticks)):
        cumulative.append(
            cumulative[-1]
            + mido.tick2second(ticks[index] - ticks[index - 1], midi.ticks_per_beat, tempos[index - 1])
        )

    def convert(tick: int) -> float:
        index = bisect.bisect_right(ticks, int(tick)) - 1
        return cumulative[index] + mido.tick2second(
            int(tick) - ticks[index], midi.ticks_per_beat, tempos[index]
        )

    return convert


@dataclass(frozen=True)
class PostNoteoffEvent:
    delay_seconds: float
    destination_state: int
    transition_type: str


def post_noteoff_events(encoded: EncodedPerformance) -> tuple[PostNoteoffEvent, ...]:
    convert = tick_to_seconds(encoded.source.path)
    noteoff_seconds = convert(encoded.source.latest_note_off)
    result: list[PostNoteoffEvent] = []
    for event in encoded.effective_events:
        if event.tick <= encoded.source.latest_note_off:
            continue
        delay = convert(event.tick) - noteoff_seconds
        if delay < -1e-12:
            raise AssertionError("post-note-off delay is negative")
        before_on = event.previous_state >= 2
        after_on = event.state >= 2
        if not before_on and after_on:
            transition_type = "OFF_TO_ON"
        elif before_on and not after_on:
            transition_type = "ON_TO_OFF"
        elif before_on:
            transition_type = "ON_TO_ON_DEPTH"
        else:
            transition_type = "OFF_TO_OFF_DEPTH"
        result.append(PostNoteoffEvent(max(0.0, delay), event.state, transition_type))
    return tuple(result)


def decoded_controls(encoded: EncodedPerformance, variant: str) -> tuple[DecodedControl, ...]:
    first = encoded.source.distinct_onsets[0]
    controls = [DecodedControl(max(0, first - 1), REPRESENTATIVES[encoded.initial_state], "initial", None)]
    for interval in encoded.intervals:
        for event in retained_events(interval, variant):
            if event.state is None or event.tau is None:
                continue
            tick = int(round(interval.start_tick + event.tau * (interval.end_tick - interval.start_tick)))
            if event.source_tick is not None and tick != event.source_tick:
                raise AssertionError("exact tau failed to reconstruct source event tick")
            controls.append(DecodedControl(tick, REPRESENTATIVES[event.state], "event", interval.index))
    return tuple(controls)


def write_variant_midi(encoded: EncodedPerformance, output_path: str | Path, variant: str) -> Path:
    """Serialize one diagnostic capacity view without changing core decoder code."""

    source_path = Path(encoded.source.path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    midi = mido.MidiFile(str(source_path), clip=False)
    generated = decoded_controls(encoded, variant)
    rebuilt_tracks: list[mido.MidiTrack] = []
    for track_index, track in enumerate(midi.tracks):
        absolute = 0
        rows = []
        for order, message in enumerate(track):
            absolute += int(message.time)
            source_cc64 = (
                message.type == "control_change"
                and int(message.channel) != 9
                and int(message.control) == 64
            )
            if not source_cc64:
                rows.append((absolute, 0, order, message.copy(time=0)))
        if track_index == encoded.source.preferred_cc_track:
            for order, control in enumerate(generated):
                priority = -2 if control.kind == "initial" else -1
                rows.append(
                    (
                        control.tick,
                        priority,
                        order,
                        mido.Message(
                            "control_change",
                            channel=encoded.source.preferred_cc_channel,
                            control=64,
                            value=control.value,
                            time=0,
                        ),
                    )
                )
        rows.sort(key=lambda item: (item[0], item[1], item[2]))
        rebuilt = mido.MidiTrack()
        previous = 0
        for tick, _, _, message in rows:
            rebuilt.append(message.copy(time=tick - previous))
            previous = tick
        rebuilt_tracks.append(rebuilt)
    midi.tracks.clear()
    midi.tracks.extend(rebuilt_tracks)
    midi.save(output)
    if parse_raw_midi(output).note_signature != encoded.source.note_signature:
        raise AssertionError("capacity oracle changed non-pedal note identity")
    return output
