from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mido
import numpy as np


RANDOM_SEED = 20260728
DEFAULT_INPUT = Path("third_party/PianistTransformer/data/midis/testset/human")
DEFAULT_OUTPUT = Path("analysis/stage2_pedal_target_audit")
IOI_BIN_ORDER = ["<100ms", "100-250ms", "250-500ms", "500-1000ms", ">=1000ms"]
COUNT_BUCKET_ORDER = ["0", "1", "2", "3", "4+"]
CLASS_ORDER = ["NONE", "UP", "DOWN"]
DEPTH_CLASS_ORDER = ["LIGHT", "MEDIUM", "DEEP"]
RAPID_WINDOWS_MS = [25, 50, 100, 200]
LONG_IOI_SECONDS = 5.0
NEAR_THRESHOLD_LOW = 56
NEAR_THRESHOLD_HIGH = 72


@dataclass(frozen=True)
class TempoSegment:
    tick: int
    time_sec: float
    tempo_us_per_beat: int


@dataclass(frozen=True)
class CCEvent:
    """One raw sustain-controller event after tempo-aware time conversion."""

    tick: int
    time_sec: float
    track_index: int
    message_index: int
    channel: int
    value: int


@dataclass(frozen=True)
class Transition:
    """A state-machine pedal transition caused by one raw CC64 event."""

    state_machine: str
    event_type: str
    tick: int
    time_sec: float
    previous_cc64: int
    new_cc64: int
    source_cc_index: int


@dataclass
class ParsedMidi:
    path: Path
    ticks_per_beat: int
    midi_type: int
    note_count: int
    note_onset_ticks: list[int]
    note_onset_times_sec: list[float]
    note_channel_counts: dict[int, int]
    cc_events: list[CCEvent]
    cc_channel_counts: dict[int, int]
    cc_track_counts: dict[int, int]
    tempo_change_count: int
    channel_conflict_count: int
    track_conflict_count: int
    drum_note_count: int


@dataclass(frozen=True)
class StateMachine:
    name: str
    down_threshold: int
    up_threshold: int
    up_inclusive: bool


STATE_MACHINES = [
    StateMachine("baseline_64", 64, 64, False),
    StateMachine("narrow_68_60", 68, 60, True),
    StateMachine("wide_72_56", 72, 56, True),
]


def safe_ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator else None


def finite_or_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): finite_or_none(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_or_none(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def numeric_filename_key(path: Path) -> tuple[int, int, str]:
    stem = path.stem
    parts = stem.split("-", maxsplit=1)
    if len(parts) == 2 and all(part.isdigit() for part in parts):
        return int(parts[0]), int(parts[1]), path.name
    return sys.maxsize, sys.maxsize, path.name


def discover_midi_files(input_dir: Path) -> list[Path]:
    paths = [*input_dir.glob("*.mid"), *input_dir.glob("*.midi")]
    return sorted(paths, key=numeric_filename_key)


def build_tempo_segments(
    ticks_per_beat: int,
    tempo_events: Sequence[tuple[int, int, int, int]],
) -> list[TempoSegment]:
    """Build a piecewise-constant tempo map from absolute-tick tempo events."""

    grouped: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for tick, track_index, message_index, tempo in tempo_events:
        grouped[int(tick)].append((track_index, message_index, int(tempo)))

    segments = [TempoSegment(0, 0.0, 500_000)]
    current_tick = 0
    current_time = 0.0
    current_tempo = 500_000
    for tick in sorted(grouped):
        if tick < current_tick:
            raise ValueError("Tempo events are not sorted by absolute tick")
        current_time += (
            (tick - current_tick) * current_tempo / ticks_per_beat / 1_000_000.0
        )
        current_tick = tick
        ordered = sorted(grouped[tick], key=lambda item: (item[0], item[1]))
        current_tempo = ordered[-1][2]
        segment = TempoSegment(current_tick, current_time, current_tempo)
        if segments[-1].tick == current_tick:
            segments[-1] = segment
        else:
            segments.append(segment)
    return segments


def tick_to_seconds(
    tick: int,
    ticks_per_beat: int,
    segments: Sequence[TempoSegment],
) -> float:
    segment_ticks = [segment.tick for segment in segments]
    index = bisect.bisect_right(segment_ticks, tick) - 1
    segment = segments[max(index, 0)]
    return segment.time_sec + (
        (tick - segment.tick)
        * segment.tempo_us_per_beat
        / ticks_per_beat
        / 1_000_000.0
    )


def parse_midi(path: Path) -> ParsedMidi:
    """Parse notes, CC64, channels, tracks, and tempo into absolute seconds."""

    midi = mido.MidiFile(path)
    tempo_events: list[tuple[int, int, int, int]] = []
    raw_notes: list[tuple[int, int, int, int]] = []
    raw_cc: list[tuple[int, int, int, int, int]] = []
    note_channels: Counter[int] = Counter()
    cc_channels: Counter[int] = Counter()
    cc_tracks: Counter[int] = Counter()
    drum_note_count = 0

    for track_index, track in enumerate(midi.tracks):
        absolute_tick = 0
        for message_index, message in enumerate(track):
            absolute_tick += int(message.time)
            if message.type == "set_tempo":
                tempo_events.append(
                    (absolute_tick, track_index, message_index, int(message.tempo))
                )
            elif message.type == "note_on" and int(message.velocity) > 0:
                channel = int(message.channel)
                raw_notes.append((absolute_tick, track_index, message_index, channel))
                note_channels[channel] += 1
                if channel == 9:
                    drum_note_count += 1
            elif message.type == "control_change" and int(message.control) == 64:
                channel = int(message.channel)
                value = int(message.value)
                raw_cc.append(
                    (absolute_tick, track_index, message_index, channel, value)
                )
                cc_channels[channel] += 1
                cc_tracks[track_index] += 1

    segments = build_tempo_segments(midi.ticks_per_beat, tempo_events)
    note_ticks = sorted({item[0] for item in raw_notes})
    note_times = [
        tick_to_seconds(tick, midi.ticks_per_beat, segments) for tick in note_ticks
    ]

    raw_cc.sort(key=lambda item: (item[0], item[1], item[2]))
    cc_events = [
        CCEvent(
            tick=tick,
            time_sec=tick_to_seconds(tick, midi.ticks_per_beat, segments),
            track_index=track_index,
            message_index=message_index,
            channel=channel,
            value=value,
        )
        for tick, track_index, message_index, channel, value in raw_cc
    ]

    channel_conflicts = 0
    track_conflicts = 0
    by_tick: dict[int, list[CCEvent]] = defaultdict(list)
    for event in cc_events:
        by_tick[event.tick].append(event)
    for events in by_tick.values():
        if len(events) < 2 or len({event.value for event in events}) < 2:
            continue
        if len({event.channel for event in events}) > 1:
            channel_conflicts += 1
        if len({event.track_index for event in events}) > 1:
            track_conflicts += 1

    if any(event.value < 0 or event.value > 127 for event in cc_events):
        raise ValueError("CC64 value outside MIDI range 0..127")
    if note_ticks != sorted(note_ticks):
        raise ValueError("Unique note onsets are not sorted")

    return ParsedMidi(
        path=path,
        ticks_per_beat=int(midi.ticks_per_beat),
        midi_type=int(midi.type),
        note_count=len(raw_notes),
        note_onset_ticks=note_ticks,
        note_onset_times_sec=note_times,
        note_channel_counts=dict(sorted(note_channels.items())),
        cc_events=cc_events,
        cc_channel_counts=dict(sorted(cc_channels.items())),
        cc_track_counts=dict(sorted(cc_tracks.items())),
        tempo_change_count=len(tempo_events),
        channel_conflict_count=channel_conflicts,
        track_conflict_count=track_conflicts,
        drum_note_count=drum_note_count,
    )


def detect_transitions(
    cc_events: Sequence[CCEvent],
    machine: StateMachine,
) -> list[Transition]:
    """Run a threshold or hysteresis state machine over raw CC64 events."""

    state_on = False
    previous_value = 0
    transitions: list[Transition] = []
    for source_index, event in enumerate(cc_events):
        event_type: str | None = None
        if not state_on and event.value >= machine.down_threshold:
            state_on = True
            event_type = "DOWN"
        elif state_on:
            release = (
                event.value <= machine.up_threshold
                if machine.up_inclusive
                else event.value < machine.up_threshold
            )
            if release:
                state_on = False
                event_type = "UP"
        if event_type is not None:
            transitions.append(
                Transition(
                    state_machine=machine.name,
                    event_type=event_type,
                    tick=event.tick,
                    time_sec=event.time_sec,
                    previous_cc64=previous_value,
                    new_cc64=event.value,
                    source_cc_index=source_index,
                )
            )
        previous_value = event.value
    return transitions


def assign_transition_intervals(
    transitions: Sequence[Transition],
    onset_ticks: Sequence[int],
) -> tuple[list[list[int]], list[int], list[int]]:
    """Assign transitions to left-closed/right-open IOIs by absolute tick."""

    interval_count = max(len(onset_ticks) - 1, 0)
    assignments: list[list[int]] = [[] for _ in range(interval_count)]
    before: list[int] = []
    after: list[int] = []
    if not onset_ticks:
        return assignments, list(range(len(transitions))), []

    first_tick = onset_ticks[0]
    last_tick = onset_ticks[-1]
    for transition_index, transition in enumerate(transitions):
        if transition.tick < first_tick:
            before.append(transition_index)
        elif transition.tick >= last_tick:
            after.append(transition_index)
        else:
            interval_index = bisect.bisect_right(onset_ticks, transition.tick) - 1
            if interval_index < 0 or interval_index >= interval_count:
                raise AssertionError("Transition interval assignment failed")
            assignments[interval_index].append(transition_index)
    return assignments, before, after


def ioi_duration_bin(duration_sec: float) -> str:
    duration_ms = duration_sec * 1000.0
    if duration_ms < 100:
        return "<100ms"
    if duration_ms < 250:
        return "100-250ms"
    if duration_ms < 500:
        return "250-500ms"
    if duration_ms < 1000:
        return "500-1000ms"
    return ">=1000ms"


def transition_count_bucket(count: int) -> str:
    return str(count) if count <= 3 else "4+"


def count_distribution(
    assignments: Sequence[Sequence[int]],
) -> dict[str, int]:
    counts = Counter(transition_count_bucket(len(items)) for items in assignments)
    return {bucket: int(counts.get(bucket, 0)) for bucket in COUNT_BUCKET_ORDER}


def weighted_quantile(
    values: Sequence[float],
    weights: Sequence[float],
    quantile: float,
) -> float | None:
    if not values:
        return None
    value_array = np.asarray(values, dtype=float)
    weight_array = np.asarray(weights, dtype=float)
    positive = weight_array > 0
    if not np.any(positive):
        return None
    value_array = value_array[positive]
    weight_array = weight_array[positive]
    order = np.argsort(value_array, kind="stable")
    value_array = value_array[order]
    weight_array = weight_array[order]
    cumulative = np.cumsum(weight_array)
    cutoff = float(quantile) * float(cumulative[-1])
    index = int(np.searchsorted(cumulative, cutoff, side="left"))
    return float(value_array[min(index, len(value_array) - 1)])


def numeric_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "median": None,
            "p5": None,
            "p10": None,
            "p25": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    array = np.asarray(values, dtype=float)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "median": float(np.median(array)),
        "p5": float(np.quantile(array, 0.05)),
        "p10": float(np.quantile(array, 0.10)),
        "p25": float(np.quantile(array, 0.25)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def tau_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    summary = numeric_summary(values)
    count = len(values)
    summary.update(
        {
            "proportion_tau_lt_0_05": safe_ratio(
                sum(value < 0.05 for value in values), count
            ),
            "proportion_tau_lt_0_10": safe_ratio(
                sum(value < 0.10 for value in values), count
            ),
            "proportion_tau_gt_0_90": safe_ratio(
                sum(value > 0.90 for value in values), count
            ),
            "proportion_tau_gt_0_95": safe_ratio(
                sum(value > 0.95 for value in values), count
            ),
        }
    )
    return summary


def integrate_interval_depth(
    start_tick: int,
    end_tick: int,
    start_sec: float,
    end_sec: float,
    cc_events: Sequence[CCEvent],
    cc_ticks: Sequence[int],
) -> tuple[dict[str, Any], np.ndarray]:
    """Integrate the zero-order-held ON-depth trajectory over one IOI."""

    duration = end_sec - start_sec
    if duration <= 0:
        raise AssertionError("IOI duration must be positive")

    previous_index = bisect.bisect_right(cc_ticks, start_tick) - 1
    current_value = cc_events[previous_index].value if previous_index >= 0 else 0
    event_index = bisect.bisect_right(cc_ticks, start_tick)
    current_time = start_sec
    weighted_depth = np.zeros(128, dtype=np.float64)

    while event_index < len(cc_events) and cc_events[event_index].tick < end_tick:
        event = cc_events[event_index]
        segment_duration = max(0.0, event.time_sec - current_time)
        if current_value >= 64 and segment_duration > 0:
            weighted_depth[current_value] += segment_duration
        current_value = event.value
        current_time = event.time_sec
        event_index += 1

    final_duration = max(0.0, end_sec - current_time)
    if current_value >= 64 and final_duration > 0:
        weighted_depth[current_value] += final_duration

    on_duration = float(weighted_depth.sum())
    if on_duration <= 0:
        metrics = {
            "on_duration_sec": 0.0,
            "on_duration_ratio": 0.0,
            "on_depth_mean": None,
            "on_depth_median": None,
            "on_depth_min": None,
            "on_depth_max": None,
            "pedal_state_coverage": "CONTINUOUSLY_OFF",
        }
        return metrics, weighted_depth

    values = np.arange(128, dtype=float)
    mean_depth = float(np.dot(values, weighted_depth) / on_duration)
    active_values = np.flatnonzero(weighted_depth > 0)
    median_depth = weighted_quantile(
        active_values.tolist(),
        weighted_depth[active_values].tolist(),
        0.5,
    )
    coverage = (
        "CONTINUOUSLY_ON"
        if math.isclose(on_duration, duration, rel_tol=0.0, abs_tol=1e-9)
        else "PARTIALLY_ON"
    )
    metrics = {
        "on_duration_sec": on_duration,
        "on_duration_ratio": on_duration / duration,
        "on_depth_mean": mean_depth,
        "on_depth_median": median_depth,
        "on_depth_min": int(active_values[0]),
        "on_depth_max": int(active_values[-1]),
        "pedal_state_coverage": coverage,
    }
    return metrics, weighted_depth


def rapid_reversal_rows(
    filename: str,
    machine_name: str,
    transitions: Sequence[Transition],
    cc_events: Sequence[CCEvent],
    onset_ticks: Sequence[int],
) -> list[dict[str, Any]]:
    """Return adjacent opposite transitions no more than 200 ms apart."""

    rows: list[dict[str, Any]] = []
    for pair_index, (first, second) in enumerate(
        zip(transitions, transitions[1:], strict=False)
    ):
        if first.event_type == second.event_type:
            continue
        gap_ms = (second.time_sec - first.time_sec) * 1000.0
        if gap_ms < -1e-9 or gap_ms > 200.0:
            continue
        start_index = min(first.source_cc_index, second.source_cc_index)
        end_index = max(first.source_cc_index, second.source_cc_index)
        between_values = [
            event.value for event in cc_events[start_index : end_index + 1]
        ]
        near_count = sum(
            NEAR_THRESHOLD_LOW <= value <= NEAR_THRESHOLD_HIGH
            for value in between_values
        )
        near_fraction = safe_ratio(near_count, len(between_values))
        near_candidate = bool(
            between_values
            and all(
                NEAR_THRESHOLD_LOW <= value <= NEAR_THRESHOLD_HIGH
                for value in between_values
            )
        )
        first_interval = (
            bisect.bisect_right(onset_ticks, first.tick) - 1
            if onset_ticks and onset_ticks[0] <= first.tick < onset_ticks[-1]
            else None
        )
        second_interval = (
            bisect.bisect_right(onset_ticks, second.tick) - 1
            if onset_ticks and onset_ticks[0] <= second.tick < onset_ticks[-1]
            else None
        )
        rows.append(
            {
                "file": filename,
                "state_machine": machine_name,
                "pair_index": pair_index,
                "pair_type": f"{first.event_type}->{second.event_type}",
                "first_type": first.event_type,
                "second_type": second.event_type,
                "first_time_sec": first.time_sec,
                "second_time_sec": second.time_sec,
                "gap_ms": gap_ms,
                "first_cc64": first.new_cc64,
                "second_cc64": second.new_cc64,
                "between_cc64_values": "|".join(map(str, between_values)),
                "near_threshold_fraction": near_fraction,
                "near_threshold_candidate": near_candidate,
                "within_25ms": gap_ms <= 25.0,
                "within_50ms": gap_ms <= 50.0,
                "within_100ms": gap_ms <= 100.0,
                "within_200ms": True,
                "first_interval_index": first_interval,
                "second_interval_index": second_interval,
                "same_ioi": first_interval is not None
                and first_interval == second_interval,
            }
        )
    return rows


def ordered_timing_matches(
    baseline: Sequence[Transition],
    comparison: Sequence[Transition],
) -> dict[str, Any]:
    """Match each hysteresis event to the latest baseline crossing in its gesture."""

    all_differences: list[float] = []
    matched_baseline_indices: set[int] = set()
    previous_comparison_source_index = -1
    unmatched_comparison = 0
    for comparison_event in comparison:
        candidates = [
            (baseline_index, baseline_event)
            for baseline_index, baseline_event in enumerate(baseline)
            if (
                previous_comparison_source_index
                < baseline_event.source_cc_index
                <= comparison_event.source_cc_index
                and baseline_event.event_type == comparison_event.event_type
            )
        ]
        if not candidates:
            unmatched_comparison += 1
        else:
            baseline_index, baseline_event = candidates[-1]
            matched_baseline_indices.add(baseline_index)
            all_differences.append(
                comparison_event.time_sec - baseline_event.time_sec
            )
        previous_comparison_source_index = comparison_event.source_cc_index

    matched = len(all_differences)
    absolute_ms = [abs(value) * 1000.0 for value in all_differences]
    signed_ms = [value * 1000.0 for value in all_differences]
    return {
        "match_strategy": (
            "latest_same_direction_baseline_crossing_since_previous_"
            "hysteresis_event"
        ),
        "matched_count": matched,
        "unmatched_baseline_count": len(baseline)
        - len(matched_baseline_indices),
        "unmatched_comparison_count": unmatched_comparison,
        "absolute_timing_error_ms": numeric_summary(absolute_ms),
        "signed_timing_difference_ms": numeric_summary(signed_ms),
        "_absolute_ms": absolute_ms,
        "_signed_ms": signed_ms,
    }


def class_distribution(labels: Iterable[str]) -> dict[str, int]:
    counter = Counter(labels)
    return {label: int(counter.get(label, 0)) for label in CLASS_ORDER}


def inverse_frequency_candidates(counts: dict[str, int]) -> dict[str, float | None]:
    total = sum(counts.values())
    class_count = len(counts)
    return {
        label: (total / (class_count * count) if count else None)
        for label, count in counts.items()
    }


def depth_class(
    value: float | None,
    method: str,
    tertile_boundaries: tuple[int, int],
) -> str:
    if value is None:
        return "NONE"
    if method == "equal_width":
        if value < 85:
            return "LIGHT"
        if value < 106:
            return "MEDIUM"
        return "DEEP"
    first, second = tertile_boundaries
    if value <= first:
        return "LIGHT"
    if value <= second:
        return "MEDIUM"
    return "DEEP"


def analyze_file(parsed: ParsedMidi, repo_root: Path) -> dict[str, Any]:
    filename = parsed.path.name
    relative_path = parsed.path.resolve().relative_to(repo_root.resolve()).as_posix()
    base_row: dict[str, Any] = {
        "file": filename,
        "relative_path": relative_path,
        "status": "analyzed",
        "error": "",
        "midi_type": parsed.midi_type,
        "ticks_per_beat": parsed.ticks_per_beat,
        "tempo_change_count": parsed.tempo_change_count,
        "note_count": parsed.note_count,
        "unique_onset_count": len(parsed.note_onset_ticks),
        "ioi_count": max(len(parsed.note_onset_ticks) - 1, 0),
        "cc64_event_count": len(parsed.cc_events),
        "cc64_channel_distribution": json.dumps(parsed.cc_channel_counts),
        "cc64_track_distribution": json.dumps(parsed.cc_track_counts),
        "note_channel_distribution": json.dumps(parsed.note_channel_counts),
        "channel_conflict_count": parsed.channel_conflict_count,
        "track_conflict_count": parsed.track_conflict_count,
        "drum_note_count": parsed.drum_note_count,
    }
    if not parsed.cc_events:
        base_row["status"] = "excluded_no_cc64"
        return {
            "per_file": base_row,
            "interval_rows": [],
            "transition_rows": [],
            "overflow_rows": [],
            "crossing_rows": [],
            "depth_hist": np.zeros(128, dtype=float),
            "machine_summaries": {},
            "timing_matches": {},
            "warnings": [],
        }
    if len(parsed.note_onset_ticks) < 2:
        base_row["status"] = "excluded_insufficient_unique_onsets"
        return {
            "per_file": base_row,
            "interval_rows": [],
            "transition_rows": [],
            "overflow_rows": [],
            "crossing_rows": [],
            "depth_hist": np.zeros(128, dtype=float),
            "machine_summaries": {},
            "timing_matches": {},
            "warnings": ["fewer_than_two_unique_note_onsets"],
        }

    onset_ticks = parsed.note_onset_ticks
    onset_times = parsed.note_onset_times_sec
    interval_count = len(onset_ticks) - 1
    durations = [
        onset_times[index + 1] - onset_times[index]
        for index in range(interval_count)
    ]
    if any(duration <= 0 for duration in durations):
        raise AssertionError("Found zero-length or negative IOI")

    cc_before = sum(event.tick < onset_ticks[0] for event in parsed.cc_events)
    cc_after = sum(event.tick >= onset_ticks[-1] for event in parsed.cc_events)
    base_row["cc64_events_before_first_onset"] = cc_before
    base_row["cc64_events_at_or_after_last_onset"] = cc_after
    base_row["long_ioi_count_gt_5s"] = sum(
        duration > LONG_IOI_SECONDS for duration in durations
    )
    base_row["max_ioi_duration_sec"] = max(durations)

    transitions_by_machine: dict[str, list[Transition]] = {}
    assignments_by_machine: dict[str, list[list[int]]] = {}
    machine_summaries: dict[str, Any] = {}
    crossing_rows: list[dict[str, Any]] = []
    for machine in STATE_MACHINES:
        transitions = detect_transitions(parsed.cc_events, machine)
        assignments, before_transitions, after_transitions = (
            assign_transition_intervals(transitions, onset_ticks)
        )
        distribution = count_distribution(assignments)
        rapid_rows = rapid_reversal_rows(
            filename,
            machine.name,
            transitions,
            parsed.cc_events,
            onset_ticks,
        )
        crossing_rows.extend(rapid_rows)
        main_transition_indices = {
            transition_index
            for interval in assignments
            for transition_index in interval
        }
        main_transitions = [
            transition
            for transition_index, transition in enumerate(transitions)
            if transition_index in main_transition_indices
        ]
        machine_summary = {
            "up": sum(event.event_type == "UP" for event in transitions),
            "down": sum(event.event_type == "DOWN" for event in transitions),
            "total": len(transitions),
            "main_ioi_up": sum(
                event.event_type == "UP" for event in main_transitions
            ),
            "main_ioi_down": sum(
                event.event_type == "DOWN" for event in main_transitions
            ),
            "main_ioi_total": len(main_transitions),
            "transitions_before_first_onset": len(before_transitions),
            "transitions_at_or_after_last_onset": len(after_transitions),
            "ioi_transition_count": distribution,
            "two_slot_coverage": safe_ratio(
                distribution["0"] + distribution["1"] + distribution["2"],
                interval_count,
            ),
            "rapid_reversal": {
                str(window): sum(
                    float(row["gap_ms"]) <= window for row in rapid_rows
                )
                for window in RAPID_WINDOWS_MS
            },
            "near_threshold_rapid_reversal": {
                str(window): sum(
                    float(row["gap_ms"]) <= window
                    and bool(row["near_threshold_candidate"])
                    for row in rapid_rows
                )
                for window in RAPID_WINDOWS_MS
            },
            "rapid_reversal_with_both_events_in_main_iois": {
                str(window): sum(
                    float(row["gap_ms"]) <= window
                    and row["first_interval_index"] is not None
                    and row["second_interval_index"] is not None
                    for row in rapid_rows
                )
                for window in RAPID_WINDOWS_MS
            },
            "near_threshold_rapid_reversal_with_both_events_in_main_iois": {
                str(window): sum(
                    float(row["gap_ms"]) <= window
                    and bool(row["near_threshold_candidate"])
                    and row["first_interval_index"] is not None
                    and row["second_interval_index"] is not None
                    for row in rapid_rows
                )
                for window in RAPID_WINDOWS_MS
            },
        }
        transitions_by_machine[machine.name] = transitions
        assignments_by_machine[machine.name] = assignments
        machine_summaries[machine.name] = machine_summary
        prefix = machine.name
        base_row[f"{prefix}_up_count"] = machine_summary["up"]
        base_row[f"{prefix}_down_count"] = machine_summary["down"]
        base_row[f"{prefix}_transition_count"] = machine_summary["total"]
        base_row[f"{prefix}_main_ioi_up_count"] = machine_summary[
            "main_ioi_up"
        ]
        base_row[f"{prefix}_main_ioi_down_count"] = machine_summary[
            "main_ioi_down"
        ]
        base_row[f"{prefix}_main_ioi_transition_count"] = machine_summary[
            "main_ioi_total"
        ]
        base_row[f"{prefix}_transitions_before_first_onset"] = machine_summary[
            "transitions_before_first_onset"
        ]
        base_row[f"{prefix}_transitions_at_or_after_last_onset"] = (
            machine_summary["transitions_at_or_after_last_onset"]
        )
        base_row[f"{prefix}_two_slot_coverage"] = machine_summary[
            "two_slot_coverage"
        ]
        for window in RAPID_WINDOWS_MS:
            base_row[f"{prefix}_rapid_reversal_{window}ms"] = machine_summary[
                "rapid_reversal"
            ][str(window)]
            base_row[f"{prefix}_near_threshold_rapid_{window}ms"] = (
                machine_summary["near_threshold_rapid_reversal"][str(window)]
            )

    baseline = transitions_by_machine["baseline_64"]
    baseline_assignments = assignments_by_machine["baseline_64"]
    baseline_distribution = machine_summaries["baseline_64"][
        "ioi_transition_count"
    ]
    base_row["transition_count_0_iois"] = baseline_distribution["0"]
    base_row["transition_count_1_iois"] = baseline_distribution["1"]
    base_row["transition_count_2_iois"] = baseline_distribution["2"]
    base_row["transition_count_3_iois"] = baseline_distribution["3"]
    base_row["transition_count_4plus_iois"] = baseline_distribution["4+"]
    base_row["overflow_ioi_count"] = (
        baseline_distribution["3"] + baseline_distribution["4+"]
    )
    base_row["overflow_ioi_ratio"] = safe_ratio(
        base_row["overflow_ioi_count"], interval_count
    )

    same_timestamp_opposite = 0
    by_transition_tick: dict[int, set[str]] = defaultdict(set)
    for transition in baseline:
        by_transition_tick[transition.tick].add(transition.event_type)
    same_timestamp_opposite = sum(
        event_types == {"UP", "DOWN"}
        for event_types in by_transition_tick.values()
    )
    base_row["same_timestamp_opposite_transition_count"] = (
        same_timestamp_opposite
    )

    cc_ticks = [event.tick for event in parsed.cc_events]
    interval_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    overflow_rows: list[dict[str, Any]] = []
    depth_hist = np.zeros(128, dtype=np.float64)
    status_counts: Counter[str] = Counter()

    for interval_index in range(interval_count):
        start_tick = onset_ticks[interval_index]
        end_tick = onset_ticks[interval_index + 1]
        start_sec = onset_times[interval_index]
        end_sec = onset_times[interval_index + 1]
        duration_sec = end_sec - start_sec
        assigned_indices = baseline_assignments[interval_index]
        assigned = [baseline[index] for index in assigned_indices]
        depth_metrics, interval_depth = integrate_interval_depth(
            start_tick,
            end_tick,
            start_sec,
            end_sec,
            parsed.cc_events,
            cc_ticks,
        )
        depth_hist += interval_depth
        status_counts[depth_metrics["pedal_state_coverage"]] += 1

        slot1 = assigned[0] if len(assigned) >= 1 else None
        slot2 = assigned[1] if len(assigned) >= 2 else None

        def tau_for(event: Transition | None) -> float | None:
            if event is None:
                return None
            tau = (event.time_sec - start_sec) / duration_sec
            if not (0.0 <= tau < 1.0):
                raise AssertionError(
                    f"tau outside [0,1): {filename} interval={interval_index} tau={tau}"
                )
            return tau

        slot1_tau = tau_for(slot1)
        slot2_tau = tau_for(slot2)
        interval_row = {
            "file": filename,
            "interval_index": interval_index,
            "onset_start_tick": start_tick,
            "onset_end_tick": end_tick,
            "onset_start_sec": start_sec,
            "onset_end_sec": end_sec,
            "ioi_duration_sec": duration_sec,
            "ioi_duration_ms": duration_sec * 1000.0,
            "ioi_duration_bin": ioi_duration_bin(duration_sec),
            "transition_count": len(assigned),
            "slot1_type": slot1.event_type if slot1 else "NONE",
            "slot1_time_sec": slot1.time_sec if slot1 else None,
            "slot1_tau": slot1_tau,
            "slot2_type": slot2.event_type if slot2 else "NONE",
            "slot2_time_sec": slot2.time_sec if slot2 else None,
            "slot2_tau": slot2_tau,
            "overflow_event_count": max(0, len(assigned) - 2),
            **depth_metrics,
            "equal_width_depth_class": "",
            "quantile_depth_class": "",
        }
        interval_rows.append(interval_row)

        event_types: list[str] = []
        event_times: list[str] = []
        event_taus: list[str] = []
        for sequence_index, transition in enumerate(assigned):
            tau = tau_for(transition)
            event_types.append(transition.event_type)
            event_times.append(f"{transition.time_sec:.9f}")
            event_taus.append(f"{tau:.9f}")
            transition_rows.append(
                {
                    "file": filename,
                    "interval_index": interval_index,
                    "event_sequence_index": sequence_index,
                    "slot_assignment": (
                        sequence_index + 1 if sequence_index < 2 else "OVERFLOW"
                    ),
                    "event_type": transition.event_type,
                    "absolute_tick": transition.tick,
                    "absolute_time_sec": transition.time_sec,
                    "tau": tau,
                    "previous_cc64": transition.previous_cc64,
                    "new_cc64": transition.new_cc64,
                    "source_cc_event_index": transition.source_cc_index,
                    "ioi_duration_bin": ioi_duration_bin(duration_sec),
                }
            )
        if len(assigned) >= 3:
            overflow_rows.append(
                {
                    "file": filename,
                    "interval_index": interval_index,
                    "onset_start_sec": start_sec,
                    "onset_end_sec": end_sec,
                    "ioi_duration_sec": duration_sec,
                    "ioi_duration_ms": duration_sec * 1000.0,
                    "transition_count": len(assigned),
                    "event_types": "|".join(event_types),
                    "event_absolute_times_sec": "|".join(event_times),
                    "event_relative_times_tau": "|".join(event_taus),
                }
            )

    base_row["continuously_off_ioi_count"] = status_counts["CONTINUOUSLY_OFF"]
    base_row["continuously_on_ioi_count"] = status_counts["CONTINUOUSLY_ON"]
    base_row["partially_on_ioi_count"] = status_counts["PARTIALLY_ON"]
    base_row["ever_on_ioi_count"] = (
        status_counts["CONTINUOUSLY_ON"] + status_counts["PARTIALLY_ON"]
    )
    base_row["baseline_rapid_rate_per_1000_iois"] = (
        1000.0
        * machine_summaries["baseline_64"]["rapid_reversal"]["200"]
        / interval_count
    )
    base_row["baseline_near_threshold_rapid_rate_per_1000_iois"] = (
        1000.0
        * machine_summaries["baseline_64"]["near_threshold_rapid_reversal"]["200"]
        / interval_count
    )
    base_row["baseline_transition_rate_per_1000_iois"] = (
        1000.0 * len(baseline) / interval_count
    )

    timing_matches: dict[str, Any] = {}
    for comparison_name in ("narrow_68_60", "wide_72_56"):
        timing_matches[comparison_name] = ordered_timing_matches(
            baseline,
            transitions_by_machine[comparison_name],
        )

    warnings: list[str] = []
    if parsed.channel_conflict_count:
        warnings.append("multi_channel_cc64_conflict")
    if parsed.track_conflict_count:
        warnings.append("multi_track_same_tick_cc64_conflict")
    if same_timestamp_opposite:
        warnings.append("opposite_transitions_at_same_timestamp")
    if cc_before:
        warnings.append("cc64_before_first_onset")
    if cc_after:
        warnings.append("cc64_at_or_after_last_onset")
    if base_row["long_ioi_count_gt_5s"]:
        warnings.append("ioi_longer_than_5_seconds")
    if parsed.drum_note_count:
        warnings.append("channel_10_note_events_present")
    base_row["warnings"] = "|".join(warnings)

    return {
        "per_file": base_row,
        "interval_rows": interval_rows,
        "transition_rows": transition_rows,
        "overflow_rows": overflow_rows,
        "crossing_rows": crossing_rows,
        "depth_hist": depth_hist,
        "machine_summaries": machine_summaries,
        "timing_matches": timing_matches,
        "warnings": warnings,
    }


def aggregate_machine_summaries(
    results: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for machine in STATE_MACHINES:
        summaries = [
            result["machine_summaries"].get(machine.name)
            for result in results
            if result["machine_summaries"].get(machine.name)
        ]
        distribution = {
            bucket: sum(
                int(summary["ioi_transition_count"][bucket])
                for summary in summaries
            )
            for bucket in COUNT_BUCKET_ORDER
        }
        total_iois = sum(distribution.values())
        aggregate[machine.name] = {
            "down_threshold": machine.down_threshold,
            "up_threshold": machine.up_threshold,
            "up_comparison": "<=" if machine.up_inclusive else "<",
            "up": sum(int(summary["up"]) for summary in summaries),
            "down": sum(int(summary["down"]) for summary in summaries),
            "total": sum(int(summary["total"]) for summary in summaries),
            "main_ioi_up": sum(
                int(summary["main_ioi_up"]) for summary in summaries
            ),
            "main_ioi_down": sum(
                int(summary["main_ioi_down"]) for summary in summaries
            ),
            "main_ioi_total": sum(
                int(summary["main_ioi_total"]) for summary in summaries
            ),
            "transitions_before_first_onset": sum(
                int(summary["transitions_before_first_onset"])
                for summary in summaries
            ),
            "transitions_at_or_after_last_onset": sum(
                int(summary["transitions_at_or_after_last_onset"])
                for summary in summaries
            ),
            "ioi_transition_count": distribution,
            "two_slot_coverage": safe_ratio(
                distribution["0"] + distribution["1"] + distribution["2"],
                total_iois,
            ),
            "rapid_reversal": {
                str(window): sum(
                    int(summary["rapid_reversal"][str(window)])
                    for summary in summaries
                )
                for window in RAPID_WINDOWS_MS
            },
            "near_threshold_rapid_reversal": {
                str(window): sum(
                    int(summary["near_threshold_rapid_reversal"][str(window)])
                    for summary in summaries
                )
                for window in RAPID_WINDOWS_MS
            },
            "rapid_reversal_with_both_events_in_main_iois": {
                str(window): sum(
                    int(
                        summary[
                            "rapid_reversal_with_both_events_in_main_iois"
                        ][str(window)]
                    )
                    for summary in summaries
                )
                for window in RAPID_WINDOWS_MS
            },
            "near_threshold_rapid_reversal_with_both_events_in_main_iois": {
                str(window): sum(
                    int(
                        summary[
                            "near_threshold_rapid_reversal_with_both_events_in_main_iois"
                        ][str(window)]
                    )
                    for summary in summaries
                )
                for window in RAPID_WINDOWS_MS
            },
        }
    baseline_total = aggregate["baseline_64"]["total"]
    for comparison_name in ("narrow_68_60", "wide_72_56"):
        comparison = aggregate[comparison_name]
        comparison["net_events_removed_vs_baseline"] = (
            baseline_total - comparison["total"]
        )
        comparison["net_events_removed_ratio_vs_baseline"] = safe_ratio(
            baseline_total - comparison["total"], baseline_total
        )
        comparison["net_main_ioi_events_removed_vs_baseline"] = (
            aggregate["baseline_64"]["main_ioi_total"]
            - comparison["main_ioi_total"]
        )
        comparison["net_main_ioi_events_removed_ratio_vs_baseline"] = safe_ratio(
            aggregate["baseline_64"]["main_ioi_total"]
            - comparison["main_ioi_total"],
            aggregate["baseline_64"]["main_ioi_total"],
        )
    return aggregate


def aggregate_timing_matches(
    results: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for comparison_name in ("narrow_68_60", "wide_72_56"):
        matches = [
            result["timing_matches"].get(comparison_name)
            for result in results
            if result["timing_matches"].get(comparison_name)
        ]
        absolute = [
            value
            for match in matches
            for value in match.get("_absolute_ms", [])
        ]
        signed = [
            value for match in matches for value in match.get("_signed_ms", [])
        ]
        aggregate[comparison_name] = {
            "match_strategy": (
                "latest_same_direction_baseline_crossing_since_previous_"
                "hysteresis_event"
            ),
            "matched_count": sum(int(match["matched_count"]) for match in matches),
            "unmatched_baseline_count": sum(
                int(match["unmatched_baseline_count"]) for match in matches
            ),
            "unmatched_comparison_count": sum(
                int(match["unmatched_comparison_count"]) for match in matches
            ),
            "absolute_timing_error_ms": numeric_summary(absolute),
            "signed_timing_difference_ms": numeric_summary(signed),
        }
    return aggregate


def slot_statistics(
    interval_rows: Sequence[dict[str, Any]],
    include_overflow: bool,
) -> dict[str, Any]:
    selected = [
        row
        for row in interval_rows
        if include_overflow or int(row["transition_count"]) < 3
    ]
    slot1 = class_distribution(row["slot1_type"] for row in selected)
    slot2 = class_distribution(row["slot2_type"] for row in selected)
    combined = class_distribution(
        label
        for row in selected
        for label in (row["slot1_type"], row["slot2_type"])
    )
    return {
        "interval_count": len(selected),
        "slot1": {
            "counts": slot1,
            "frequencies": {
                label: safe_ratio(count, sum(slot1.values()))
                for label, count in slot1.items()
            },
            "inverse_frequency_weight_candidates": inverse_frequency_candidates(
                slot1
            ),
        },
        "slot2": {
            "counts": slot2,
            "frequencies": {
                label: safe_ratio(count, sum(slot2.values()))
                for label, count in slot2.items()
            },
            "inverse_frequency_weight_candidates": inverse_frequency_candidates(
                slot2
            ),
        },
        "combined": {
            "counts": combined,
            "frequencies": {
                label: safe_ratio(count, sum(combined.values()))
                for label, count in combined.items()
            },
            "inverse_frequency_weight_candidates": inverse_frequency_candidates(
                combined
            ),
        },
    }


def tau_statistics(
    transition_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    groups: dict[str, list[float]] = {
        "all": [],
        "UP": [],
        "DOWN": [],
        "slot1": [],
        "slot2": [],
        "UP_in_slot1": [],
        "DOWN_in_slot1": [],
        "UP_in_slot2": [],
        "DOWN_in_slot2": [],
    }
    for row in transition_rows:
        tau = float(row["tau"])
        event_type = str(row["event_type"])
        groups["all"].append(tau)
        groups[event_type].append(tau)
        slot = row["slot_assignment"]
        if slot in (1, 2):
            slot_name = f"slot{slot}"
            groups[slot_name].append(tau)
            groups[f"{event_type}_in_{slot_name}"].append(tau)
    return {name: tau_summary(values) for name, values in groups.items()}


def transition_aggregate_by_duration(
    interval_rows: Sequence[dict[str, Any]],
    transition_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    interval_counts = Counter(row["ioi_duration_bin"] for row in interval_rows)
    count_buckets: dict[str, Counter[str]] = {
        name: Counter() for name in IOI_BIN_ORDER
    }
    for row in interval_rows:
        count_buckets[row["ioi_duration_bin"]][
            transition_count_bucket(int(row["transition_count"]))
        ] += 1
    direction_counts: dict[str, Counter[str]] = {
        name: Counter() for name in IOI_BIN_ORDER
    }
    for row in transition_rows:
        direction_counts[row["ioi_duration_bin"]][row["event_type"]] += 1
    output: dict[str, Any] = {}
    for name in IOI_BIN_ORDER:
        bucket_counts = {
            bucket: int(count_buckets[name].get(bucket, 0))
            for bucket in COUNT_BUCKET_ORDER
        }
        output[name] = {
            "ioi_count": int(interval_counts.get(name, 0)),
            "transition_count_distribution": bucket_counts,
            "two_slot_coverage": safe_ratio(
                bucket_counts["0"] + bucket_counts["1"] + bucket_counts["2"],
                sum(bucket_counts.values()),
            ),
            "up": int(direction_counts[name].get("UP", 0)),
            "down": int(direction_counts[name].get("DOWN", 0)),
        }
    return output


def conditional_transition_distribution(
    interval_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    counts = Counter(
        transition_count_bucket(int(row["transition_count"]))
        for row in interval_rows
        if int(row["transition_count"]) > 0
    )
    total = sum(counts.values())
    return {
        bucket: {
            "count": int(counts.get(bucket, 0)),
            "proportion": safe_ratio(counts.get(bucket, 0), total),
        }
        for bucket in ("1", "2", "3", "4+")
    }


def depth_aggregate(
    interval_rows: Sequence[dict[str, Any]],
    depth_hist: np.ndarray,
) -> tuple[dict[str, Any], tuple[int, int]]:
    active_values = np.flatnonzero(depth_hist > 0)
    q1_value = weighted_quantile(
        active_values.tolist(), depth_hist[active_values].tolist(), 1.0 / 3.0
    )
    q2_value = weighted_quantile(
        active_values.tolist(), depth_hist[active_values].tolist(), 2.0 / 3.0
    )
    q1 = int(q1_value if q1_value is not None else 85)
    q2 = int(q2_value if q2_value is not None else 106)
    boundaries = (q1, q2)

    mean_depths = [
        float(row["on_depth_mean"])
        for row in interval_rows
        if row["on_depth_mean"] is not None
    ]
    median_depths = [
        float(row["on_depth_median"])
        for row in interval_rows
        if row["on_depth_median"] is not None
    ]
    status_counts = Counter(row["pedal_state_coverage"] for row in interval_rows)
    total_iois = len(interval_rows)
    total_on_duration = float(depth_hist.sum())

    time_weighted_quantiles = {
        name: weighted_quantile(
            active_values.tolist(),
            depth_hist[active_values].tolist(),
            quantile,
        )
        for name, quantile in (
            ("p5", 0.05),
            ("p10", 0.10),
            ("p25", 0.25),
            ("median", 0.50),
            ("p75", 0.75),
            ("p90", 0.90),
            ("p95", 0.95),
        )
    }
    time_weighted_mean = (
        float(np.dot(np.arange(128), depth_hist) / total_on_duration)
        if total_on_duration
        else None
    )
    output = {
        "total_on_state_duration_sec": total_on_duration,
        "time_weighted_cc64_distribution_sec": {
            str(value): float(depth_hist[value]) for value in range(64, 128)
        },
        "time_weighted_mean": time_weighted_mean,
        "time_weighted_quantiles": time_weighted_quantiles,
        "per_ioi_weighted_mean_depth": numeric_summary(mean_depths),
        "per_ioi_weighted_median_depth": numeric_summary(median_depths),
        "ioi_state_coverage": {
            "ever_on": {
                "count": int(
                    status_counts["CONTINUOUSLY_ON"]
                    + status_counts["PARTIALLY_ON"]
                ),
                "proportion": safe_ratio(
                    status_counts["CONTINUOUSLY_ON"]
                    + status_counts["PARTIALLY_ON"],
                    total_iois,
                ),
            },
            "continuously_off": {
                "count": int(status_counts["CONTINUOUSLY_OFF"]),
                "proportion": safe_ratio(
                    status_counts["CONTINUOUSLY_OFF"], total_iois
                ),
            },
            "continuously_on": {
                "count": int(status_counts["CONTINUOUSLY_ON"]),
                "proportion": safe_ratio(
                    status_counts["CONTINUOUSLY_ON"], total_iois
                ),
            },
            "partially_on": {
                "count": int(status_counts["PARTIALLY_ON"]),
                "proportion": safe_ratio(
                    status_counts["PARTIALLY_ON"], total_iois
                ),
            },
        },
        "tertile_boundaries": {
            "q1_cc64": q1,
            "q2_cc64": q2,
            "interval_definition": f"[64,{q1}], ({q1},{q2}], ({q2},127]",
            "q2_at_midi_maximum": q2 >= 127,
            "cc64_127_on_time_proportion": safe_ratio(
                float(depth_hist[127]), total_on_duration
            ),
            "tie_warning": (
                "The second quantile equals 127, so value-threshold classes "
                "cannot split the tied mass at 127 without an arbitrary rule."
                if q2 >= 127
                else ""
            ),
        },
    }
    return output, boundaries


def build_depth_candidate_rows(
    interval_rows: Sequence[dict[str, Any]],
    per_file_rows: Sequence[dict[str, Any]],
    depth_hist_by_file: dict[str, np.ndarray],
    aggregate_depth_hist: np.ndarray,
    tertile_boundaries: tuple[int, int],
) -> list[dict[str, Any]]:
    for row in interval_rows:
        row["equal_width_depth_class"] = depth_class(
            row["on_depth_median"], "equal_width", tertile_boundaries
        )
        row["quantile_depth_class"] = depth_class(
            row["on_depth_median"], "quantile", tertile_boundaries
        )

    rows: list[dict[str, Any]] = []
    intervals_by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in interval_rows:
        intervals_by_file[row["file"]].append(row)

    method_specs = {
        "equal_width": {
            "boundaries": "64<=LIGHT<85;85<=MEDIUM<106;106<=DEEP<128"
        },
        "time_weighted_tertile": {
            "boundaries": (
                f"64<=LIGHT<={tertile_boundaries[0]};"
                f"{tertile_boundaries[0]}<MEDIUM<={tertile_boundaries[1]};"
                f"{tertile_boundaries[1]}<DEEP<=127"
            )
        },
    }

    for method_name, spec in method_specs.items():
        class_column = (
            "equal_width_depth_class"
            if method_name == "equal_width"
            else "quantile_depth_class"
        )
        for scope, filename, selected_intervals, hist in [
            ("aggregate", "ALL", list(interval_rows), aggregate_depth_hist),
            *[
                (
                    "per_file",
                    per_file["file"],
                    intervals_by_file.get(per_file["file"], []),
                    depth_hist_by_file.get(
                        per_file["file"], np.zeros(128, dtype=float)
                    ),
                )
                for per_file in per_file_rows
                if per_file.get("status") == "analyzed"
            ],
        ]:
            on_intervals = [
                row for row in selected_intervals if row[class_column] != "NONE"
            ]
            interval_counts = Counter(row[class_column] for row in on_intervals)
            total_interval_count = len(on_intervals)

            def value_class(value: int) -> str:
                return depth_class(
                    float(value),
                    (
                        "equal_width"
                        if method_name == "equal_width"
                        else "quantile"
                    ),
                    tertile_boundaries,
                )

            duration_by_class = Counter()
            values_by_class: dict[str, list[int]] = defaultdict(list)
            weights_by_class: dict[str, list[float]] = defaultdict(list)
            for value in range(64, 128):
                duration = float(hist[value])
                if duration <= 0:
                    continue
                label = value_class(value)
                duration_by_class[label] += duration
                values_by_class[label].append(value)
                weights_by_class[label].append(duration)
            total_duration = sum(duration_by_class.values())
            for label in DEPTH_CLASS_ORDER:
                rows.append(
                    {
                        "scope": scope,
                        "file": filename,
                        "method": method_name,
                        "boundaries": spec["boundaries"],
                        "class": label,
                        "ioi_count": int(interval_counts.get(label, 0)),
                        "ioi_proportion_among_on_iois": safe_ratio(
                            interval_counts.get(label, 0), total_interval_count
                        ),
                        "on_duration_sec": float(duration_by_class.get(label, 0)),
                        "on_duration_proportion": safe_ratio(
                            duration_by_class.get(label, 0), total_duration
                        ),
                        "representative_time_weighted_median_cc64": (
                            weighted_quantile(
                                values_by_class.get(label, []),
                                weights_by_class.get(label, []),
                                0.5,
                            )
                        ),
                    }
                )
    return rows


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not fieldnames:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(finite_or_none(row))


def save_figures(
    output_dir: Path,
    interval_rows: Sequence[dict[str, Any]],
    transition_rows: Sequence[dict[str, Any]],
    crossing_rows: Sequence[dict[str, Any]],
    depth_hist: np.ndarray,
    depth_candidate_rows: Sequence[dict[str, Any]],
    machine_summary: dict[str, Any],
    slot_summary: dict[str, Any],
    tertile_boundaries: tuple[int, int],
) -> list[Path]:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    plt.style.use("seaborn-v0_8-whitegrid")

    counts = Counter(
        transition_count_bucket(int(row["transition_count"]))
        for row in interval_rows
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = COUNT_BUCKET_ORDER
    values = [counts[label] for label in labels]
    bars = ax.bar(
        labels,
        values,
        color=["#4C78A8", "#59A14F", "#F28E2B", "#E15759", "#B07AA1"],
    )
    ax.set_yscale("log")
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.12,
            f"{value:,}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_title("Baseline transition count per IOI (log scale)")
    ax.set_xlabel("Transitions in IOI")
    ax.set_ylabel("IOI count")
    fig.tight_layout()
    path = figures_dir / "transition_count_per_ioi.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved.append(path)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(CLASS_ORDER))
    width = 0.34
    slot1 = slot_summary["first_two_with_overflow_included"]["slot1"]["counts"]
    slot2 = slot_summary["first_two_with_overflow_included"]["slot2"]["counts"]
    ax.bar(
        x - width / 2,
        [slot1[label] for label in CLASS_ORDER],
        width,
        label="Slot 1",
        color="#4C78A8",
    )
    ax.bar(
        x + width / 2,
        [slot2[label] for label in CLASS_ORDER],
        width,
        label="Slot 2",
        color="#F28E2B",
    )
    ax.set_xticks(x, CLASS_ORDER)
    ax.set_title("Two-slot class distribution (first two, overflow retained)")
    ax.set_ylabel("Target count")
    ax.legend()
    fig.tight_layout()
    path = figures_dir / "slot_class_distribution.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved.append(path)

    all_tau = [float(row["tau"]) for row in transition_rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(all_tau, bins=50, range=(0, 1), color="#4C78A8", alpha=0.85)
    ax.set_title("All baseline transition relative times")
    ax.set_xlabel("tau")
    ax.set_ylabel("Event count")
    fig.tight_layout()
    path = figures_dir / "tau_all.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved.append(path)

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, color in (("UP", "#E15759"), ("DOWN", "#59A14F")):
        values = [
            float(row["tau"])
            for row in transition_rows
            if row["event_type"] == label
        ]
        ax.hist(
            values,
            bins=50,
            range=(0, 1),
            density=True,
            histtype="step",
            linewidth=1.8,
            label=label,
            color=color,
        )
    ax.set_title("UP versus DOWN relative time")
    ax.set_xlabel("tau")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    path = figures_dir / "tau_up_down.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved.append(path)

    fig, ax = plt.subplots(figsize=(8, 5))
    for slot, color in ((1, "#4C78A8"), (2, "#F28E2B")):
        values = [
            float(row["tau"])
            for row in transition_rows
            if row["slot_assignment"] == slot
        ]
        ax.hist(
            values,
            bins=50,
            range=(0, 1),
            density=True,
            histtype="step",
            linewidth=1.8,
            label=f"Slot {slot}",
            color=color,
        )
    ax.set_title("Slot 1 versus Slot 2 relative time")
    ax.set_xlabel("tau")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    path = figures_dir / "tau_slot1_slot2.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved.append(path)

    fig, ax = plt.subplots(figsize=(9, 5))
    for label, color in zip(
        IOI_BIN_ORDER,
        ["#4C78A8", "#59A14F", "#F28E2B", "#E15759", "#B07AA1"],
        strict=True,
    ):
        values = [
            float(row["tau"])
            for row in transition_rows
            if row["ioi_duration_bin"] == label
        ]
        if values:
            hist, edges = np.histogram(values, bins=30, range=(0, 1), density=True)
            centers = (edges[:-1] + edges[1:]) / 2
            ax.plot(centers, hist, label=label, color=color, linewidth=1.5)
    ax.set_title("Relative time by IOI duration")
    ax.set_xlabel("tau")
    ax.set_ylabel("Density")
    ax.legend(ncol=2)
    fig.tight_layout()
    path = figures_dir / "tau_by_ioi_duration.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved.append(path)

    fig, ax = plt.subplots(figsize=(10, 5))
    values = np.arange(64, 128)
    minutes = depth_hist[64:128] / 60.0
    ax.bar(values, minutes, width=0.9, color="#59A14F")
    ax.axvline(
        85,
        color="#444444",
        linestyle="--",
        linewidth=1,
        label="Equal-width: 85, 106",
    )
    ax.axvline(106, color="#444444", linestyle="--", linewidth=1)
    ax.axvline(
        tertile_boundaries[0],
        color="#E15759",
        linestyle=":",
        linewidth=1.5,
        label=(
            "Time-weighted quantiles: "
            f"{tertile_boundaries[0]}, {tertile_boundaries[1]}"
        ),
    )
    ax.axvline(
        tertile_boundaries[1],
        color="#E15759",
        linestyle=":",
        linewidth=1.5,
    )
    ax.set_title("Time-weighted pedal ON-depth")
    ax.set_xlabel("CC64 value")
    ax.set_ylabel("Held duration (minutes)")
    ax.legend()
    fig.tight_layout()
    path = figures_dir / "on_depth_distribution.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved.append(path)

    aggregate_depth_rows = [
        row for row in depth_candidate_rows if row["scope"] == "aggregate"
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(DEPTH_CLASS_ORDER))
    width = 0.34
    for offset, method, color in (
        (-width / 2, "equal_width", "#4C78A8"),
        (width / 2, "time_weighted_tertile", "#F28E2B"),
    ):
        method_rows = {
            row["class"]: row
            for row in aggregate_depth_rows
            if row["method"] == method
        }
        ax.bar(
            x + offset,
            [
                method_rows[label]["ioi_proportion_among_on_iois"]
                for label in DEPTH_CLASS_ORDER
            ],
            width,
            label=method,
            color=color,
        )
    ax.set_xticks(x, DEPTH_CLASS_ORDER)
    ax.set_ylim(0, 1)
    ax.set_title("Depth class balance among pedal-ON IOIs")
    ax.set_ylabel("IOI proportion")
    ax.legend()
    fig.tight_layout()
    path = figures_dir / "depth_class_balance.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved.append(path)

    baseline_pairs = [
        row for row in crossing_rows if row["state_machine"] == "baseline_64"
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    for pair_type, color in (("UP->DOWN", "#E15759"), ("DOWN->UP", "#4C78A8")):
        gaps = [
            float(row["gap_ms"])
            for row in baseline_pairs
            if row["pair_type"] == pair_type
        ]
        if gaps:
            ax.hist(
                gaps,
                bins=40,
                range=(0, 200),
                alpha=0.55,
                label=pair_type,
                color=color,
            )
    ax.set_title("Baseline rapid reversal interval distribution")
    ax.set_xlabel("Adjacent opposite-transition gap (ms)")
    ax.set_ylabel("Pair count")
    ax.legend()
    fig.tight_layout()
    path = figures_dir / "rapid_reversal_interval_distribution.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved.append(path)

    fig, ax = plt.subplots(figsize=(10, 5))
    names = [machine.name for machine in STATE_MACHINES]
    x = np.arange(len(names))
    width = 0.26
    totals = [machine_summary[name]["total"] for name in names]
    rapid = [machine_summary[name]["rapid_reversal"]["200"] for name in names]
    near = [
        machine_summary[name]["near_threshold_rapid_reversal"]["200"]
        for name in names
    ]
    ax.bar(x - width, totals, width, label="All transitions", color="#4C78A8")
    ax.bar(x, rapid, width, label="Rapid <=200 ms", color="#F28E2B")
    ax.bar(
        x + width,
        near,
        width,
        label="Near-threshold rapid",
        color="#E15759",
    )
    ax.set_xticks(x, names)
    ax.set_title("Threshold and hysteresis state-machine comparison")
    ax.set_ylabel("Event or pair count")
    ax.legend()
    fig.tight_layout()
    path = figures_dir / "hysteresis_comparison.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved.append(path)
    return saved


def build_summary_markdown(
    aggregate: dict[str, Any],
    depth_candidate_rows: Sequence[dict[str, Any]],
    warning_files: dict[str, list[str]],
) -> str:
    def format_optional(value: Any, format_spec: str) -> str:
        return format(value, format_spec) if value is not None else "n/a"

    inventory = aggregate["inventory"]
    baseline = aggregate["state_machines"]["baseline_64"]
    slots = aggregate["slot_class_distribution"]
    tau = aggregate["tau_distribution"]
    depth = aggregate["depth"]
    hysteresis = aggregate["state_machines"]
    counts = baseline["ioi_transition_count"]
    depth_rows = [
        row for row in depth_candidate_rows if row["scope"] == "aggregate"
    ]

    lines = [
        "# Stage 2 Pedal Target Audit",
        "",
        "## Scope and definitions",
        "",
        "- Source: raw repository-provided human performance MIDI.",
        "- No Pianist Transformer pedal tokens, model predictions, training, or GPU code were used.",
        "- Note onsets at the same tick are grouped; IOIs are `[t_i, t_(i+1))` in tempo-aware seconds.",
        "- Baseline OFF is CC64 `<64`; ON is CC64 `>=64`; DOWN is OFF-to-ON and UP is ON-to-OFF.",
        "- CC64 is interpreted as a zero-order-held signal with initial value 0.",
        "- Near-threshold rapid reversal means every raw CC64 value from the first crossing event through the opposite crossing event is in `[56,72]`.",
        f"- A long IOI warning uses a transparent audit threshold of `{LONG_IOI_SECONDS:g}` seconds.",
        "",
        "## Inventory",
        "",
        f"- MIDI files discovered: {inventory['total_midi_files']:,}",
        f"- CC64-containing files analyzed: {inventory['analyzed_files']:,}",
        f"- Files without CC64: {', '.join(inventory['files_without_cc64']) or 'none'}",
        f"- Parsing failures: {inventory['failed_files']:,}",
        f"- Raw CC64 channel distribution: {inventory['cc64_channel_distribution']}",
        f"- Multi-channel conflict files: {inventory['multi_channel_conflict_files']:,}",
        "",
        "## Baseline transition and two-slot coverage",
        "",
        f"- Total IOIs: {aggregate['total_iois']:,}",
        f"- IOI counts for 0/1/2/3/4+ transitions: {counts['0']:,} / {counts['1']:,} / {counts['2']:,} / {counts['3']:,} / {counts['4+']:,}",
        f"- Two-slot lossless coverage `P(count<=2)`: {baseline['two_slot_coverage']:.6f}",
        f"- Overflow IOIs: {counts['3'] + counts['4+']:,} ({(1.0 - baseline['two_slot_coverage']):.6%})",
        f"- Baseline UP / DOWN inside valid IOIs: {baseline['main_ioi_up']:,} / {baseline['main_ioi_down']:,}.",
        f"- Baseline UP / DOWN on the full MIDI timeline: {baseline['up']:,} / {baseline['down']:,}; {baseline['transitions_before_first_onset'] + baseline['transitions_at_or_after_last_onset']:,} transitions are outside main IOIs.",
        "",
        "## Slot class balance",
        "",
        "The first table includes the first two events from overflow IOIs; the second excludes overflow IOIs entirely. No training treatment is selected here.",
        "",
    ]
    for label, key in (
        ("First two with overflow included", "first_two_with_overflow_included"),
        ("Overflow IOIs excluded", "overflow_excluded"),
    ):
        combined = slots[key]["combined"]
        lines.extend(
            [
                f"### {label}",
                "",
                f"- Combined NONE / UP / DOWN: {combined['counts']['NONE']:,} / {combined['counts']['UP']:,} / {combined['counts']['DOWN']:,}",
                f"- Candidate normalized inverse-frequency weights: {combined['inverse_frequency_weight_candidates']}",
                "",
            ]
        )

    lines.extend(
        [
            "## Relative transition time",
            "",
            f"- All events: median tau {tau['all']['median']:.4f}, p10-p90 {tau['all']['p10']:.4f}-{tau['all']['p90']:.4f}.",
            f"- UP median tau: {tau['UP']['median']:.4f}; DOWN median tau: {tau['DOWN']['median']:.4f}.",
            f"- Near onset: tau <0.05 {tau['all']['proportion_tau_lt_0_05']:.3%}; tau <0.10 {tau['all']['proportion_tau_lt_0_10']:.3%}.",
            f"- Near next onset: tau >0.90 {tau['all']['proportion_tau_gt_0_90']:.3%}; tau >0.95 {tau['all']['proportion_tau_gt_0_95']:.3%}.",
            "",
            "## Pedal ON-depth",
            "",
            f"- IOIs ever ON: {depth['ioi_state_coverage']['ever_on']['count']:,} ({depth['ioi_state_coverage']['ever_on']['proportion']:.3%}).",
            f"- Continuously OFF / continuously ON / partially ON: {depth['ioi_state_coverage']['continuously_off']['proportion']:.3%} / {depth['ioi_state_coverage']['continuously_on']['proportion']:.3%} / {depth['ioi_state_coverage']['partially_on']['proportion']:.3%}.",
            f"- Whole-ON-time weighted mean CC64: {depth['time_weighted_mean']:.3f}; median: {depth['time_weighted_quantiles']['median']:.1f}.",
            f"- Time-weighted tertile boundary values: {depth['tertile_boundaries']['q1_cc64']}, {depth['tertile_boundaries']['q2_cc64']}.",
            f"- CC64=127 accounts for {depth['tertile_boundaries']['cc64_127_on_time_proportion']:.3%} of ON time. Because q2 equals 127, strict value-threshold tertiles collapse to two occupied classes; splitting the tied 127 mass would require an additional arbitrary rule.",
            "",
            "| Method | Class | ON-IOI count | ON-IOI proportion | ON-time proportion | Representative CC64 |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in depth_rows:
        lines.append(
            f"| {row['method']} | {row['class']} | {row['ioi_count']:,} | "
            f"{format_optional(row['ioi_proportion_among_on_iois'], '.3%')} | "
            f"{format_optional(row['on_duration_proportion'], '.3%')} | "
            f"{format_optional(row['representative_time_weighted_median_cc64'], '.1f')} |"
        )

    lines.extend(
        [
            "",
        "Equal-width boundaries are directly interpretable fixed ranges. Quantile boundaries target ON-time balance only when ties can be separated; here the dominant CC64=127 tie prevents three occupied value-threshold classes. IOI counts also need not follow ON-time proportions because one IOI target uses its time-weighted median depth.",
            "",
            "## Crossing sensitivity",
            "",
            "| State machine | Full-timeline UP | Full-timeline DOWN | Main-IOI total | Two-slot coverage | Rapid <=200 ms | Near-threshold rapid <=200 ms | Net main-IOI removed vs baseline |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name in ("baseline_64", "narrow_68_60", "wide_72_56"):
        item = hysteresis[name]
        lines.append(
            f"| {name} | {item['up']:,} | {item['down']:,} | {item['main_ioi_total']:,} | "
            f"{item['two_slot_coverage']:.6f} | "
            f"{item['rapid_reversal']['200']:,} | "
            f"{item['near_threshold_rapid_reversal']['200']:,} | "
            f"{item.get('net_main_ioi_events_removed_vs_baseline', 0):,} |"
        )

    lines.extend(
        [
            "",
            "Rapid reversal is an audit category, not a noise label: short UP-to-DOWN pairs can be genuine repedaling. Each hysteresis event is timing-matched to the latest same-direction baseline crossing since the previous hysteresis event; unmatched baseline events and net count reduction are both retained in the JSON.",
            "",
            "## Data-quality observations",
            "",
            f"- CC64 before first onset: {aggregate['sanity_checks']['cc64_events_before_first_onset']:,} events across {aggregate['sanity_checks']['files_with_cc64_before_first_onset']:,} files.",
            f"- CC64 at or after last onset: {aggregate['sanity_checks']['cc64_events_at_or_after_last_onset']:,} events across {aggregate['sanity_checks']['files_with_cc64_at_or_after_last_onset']:,} files.",
            f"- Opposite transitions at the same timestamp: {aggregate['sanity_checks']['same_timestamp_opposite_transitions']:,}.",
            f"- IOIs longer than {LONG_IOI_SECONDS:g} s: {aggregate['sanity_checks']['long_iois']:,}.",
            f"- High near-threshold crossing-rate files (>= empirical p95): {', '.join(warning_files['near_threshold_crossing_rate']) or 'none'}.",
            f"- High transition-rate files (>= empirical p95): {', '.join(warning_files['transition_rate']) or 'none'}.",
            "",
            "## Objective architecture/loss observations",
            "",
            "- The measured two-slot coverage quantifies representational fit; overflow examples remain fully enumerated in `overflow_intervals.csv`.",
            "- NONE, UP, and DOWN frequencies plus candidate inverse-frequency weights are reported, but no class weighting is selected.",
            "- Tau boundary concentration is reported separately from class labels, so event-time regression choices can be evaluated without changing transition definitions.",
            "- Equal-width and time-weighted-tertile depth labels expose different balance/interpretability tradeoffs; neither is selected as final.",
            "- Baseline and hysteresis outputs are sensitivity analyses. They do not establish which rapid reversals are performance intent versus controller jitter.",
            "",
            "## Reproduction",
            "",
            "```powershell",
            "& 'C:\\Users\\eagle\\AppData\\Local\\Programs\\Python\\Python311\\python.exe' 'scripts\\analyze_stage2_pedal_targets.py'",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def run_analysis(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    input_dir = input_dir if input_dir.is_absolute() else repo_root / input_dir
    output_dir = output_dir if output_dir.is_absolute() else repo_root / output_dir
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = discover_midi_files(input_dir)
    per_file_rows: list[dict[str, Any]] = []
    interval_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    overflow_rows: list[dict[str, Any]] = []
    crossing_rows: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    aggregate_depth_hist = np.zeros(128, dtype=np.float64)
    depth_hist_by_file: dict[str, np.ndarray] = {}
    inventory_channels: Counter[int] = Counter()

    for index, path in enumerate(paths, start=1):
        try:
            parsed = parse_midi(path)
            inventory_channels.update(parsed.cc_channel_counts)
            result = analyze_file(parsed, repo_root)
            results.append(result)
            per_file_rows.append(result["per_file"])
            interval_rows.extend(result["interval_rows"])
            transition_rows.extend(result["transition_rows"])
            overflow_rows.extend(result["overflow_rows"])
            crossing_rows.extend(result["crossing_rows"])
            aggregate_depth_hist += result["depth_hist"]
            depth_hist_by_file[path.name] = result["depth_hist"]
        except Exception as exc:
            row = {
                "file": path.name,
                "relative_path": path.resolve().relative_to(
                    repo_root.resolve()
                ).as_posix(),
                "status": "parse_or_analysis_failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            per_file_rows.append(row)
            errors.append(row.copy())
        if index % 25 == 0 or index == len(paths):
            print(f"Scanned {index}/{len(paths)} MIDI files", flush=True)

    analyzed_rows = [row for row in per_file_rows if row["status"] == "analyzed"]
    no_cc64_rows = [
        row for row in per_file_rows if row["status"] == "excluded_no_cc64"
    ]
    failed_rows = [
        row for row in per_file_rows if row["status"] == "parse_or_analysis_failed"
    ]

    machine_summary = aggregate_machine_summaries(results)
    timing_matches = aggregate_timing_matches(results)
    slots = {
        "first_two_with_overflow_included": slot_statistics(
            interval_rows, include_overflow=True
        ),
        "overflow_excluded": slot_statistics(
            interval_rows, include_overflow=False
        ),
    }
    tau = tau_statistics(transition_rows)
    by_duration = transition_aggregate_by_duration(
        interval_rows, transition_rows
    )
    conditional = conditional_transition_distribution(interval_rows)
    depth, tertile_boundaries = depth_aggregate(
        interval_rows, aggregate_depth_hist
    )
    depth_candidate_rows = build_depth_candidate_rows(
        interval_rows,
        per_file_rows,
        depth_hist_by_file,
        aggregate_depth_hist,
        tertile_boundaries,
    )

    near_rates = [
        float(row["baseline_near_threshold_rapid_rate_per_1000_iois"])
        for row in analyzed_rows
    ]
    transition_rates = [
        float(row["baseline_transition_rate_per_1000_iois"])
        for row in analyzed_rows
    ]
    near_p95 = float(np.quantile(near_rates, 0.95)) if near_rates else 0.0
    transition_p95 = (
        float(np.quantile(transition_rates, 0.95)) if transition_rates else 0.0
    )
    warning_files = {
        "near_threshold_crossing_rate": [
            row["file"]
            for row in analyzed_rows
            if float(row["baseline_near_threshold_rapid_rate_per_1000_iois"])
            >= near_p95
            and float(row["baseline_near_threshold_rapid_rate_per_1000_iois"])
            > 0
        ],
        "transition_rate": [
            row["file"]
            for row in analyzed_rows
            if float(row["baseline_transition_rate_per_1000_iois"])
            >= transition_p95
        ],
    }
    for row in analyzed_rows:
        row["near_threshold_rate_empirical_p95"] = near_p95
        row["transition_rate_empirical_p95"] = transition_p95
        row["high_near_threshold_crossing_rate"] = (
            float(row["baseline_near_threshold_rapid_rate_per_1000_iois"])
            >= near_p95
            and float(row["baseline_near_threshold_rapid_rate_per_1000_iois"])
            > 0
        )
        row["high_transition_rate"] = (
            float(row["baseline_transition_rate_per_1000_iois"])
            >= transition_p95
        )

    aggregate = {
        "analysis_name": "stage2_event_based_pedal_target_audit",
        "random_seed": RANDOM_SEED,
        "source": "raw_human_performance_midi",
        "tokenizer_used": False,
        "model_prediction_used": False,
        "input_directory": input_dir.as_posix(),
        "output_directory": output_dir.as_posix(),
        "environment": {
            "python_executable": sys.executable,
            "python_version": sys.version,
            "mido_version": metadata.version("mido"),
            "numpy_version": np.__version__,
            "matplotlib_version": matplotlib.__version__,
            "new_packages_installed_for_this_audit": [],
        },
        "inventory": {
            "total_midi_files": len(paths),
            "analyzed_files": len(analyzed_rows),
            "files_with_cc64": len(analyzed_rows),
            "files_without_cc64": [row["file"] for row in no_cc64_rows],
            "failed_files": len(failed_rows),
            "failed_file_details": errors,
            "cc64_channel_distribution": dict(
                sorted(inventory_channels.items())
            ),
            "multi_channel_cc64_files": sum(
                len(json.loads(row["cc64_channel_distribution"])) > 1
                for row in analyzed_rows
            ),
            "multi_channel_conflict_files": sum(
                int(row["channel_conflict_count"]) > 0 for row in analyzed_rows
            ),
            "multi_track_conflict_files": sum(
                int(row["track_conflict_count"]) > 0 for row in analyzed_rows
            ),
        },
        "total_notes": sum(int(row["note_count"]) for row in analyzed_rows),
        "total_unique_onsets": sum(
            int(row["unique_onset_count"]) for row in analyzed_rows
        ),
        "total_iois": len(interval_rows),
        "total_raw_cc64_events": sum(
            int(row["cc64_event_count"]) for row in analyzed_rows
        ),
        "state_machines": machine_summary,
        "hysteresis_timing_matches": timing_matches,
        "transition_distribution_conditional_on_nonempty_ioi": conditional,
        "transition_and_direction_by_ioi_duration": by_duration,
        "slot_class_distribution": slots,
        "tau_distribution": tau,
        "depth": depth,
        "rapid_reversal_pair_gap_ms": {
            machine.name: {
                pair_type: numeric_summary(
                    [
                        float(row["gap_ms"])
                        for row in crossing_rows
                        if row["state_machine"] == machine.name
                        and row["pair_type"] == pair_type
                    ]
                )
                for pair_type in ("UP->DOWN", "DOWN->UP")
            }
            for machine in STATE_MACHINES
        },
        "sanity_checks": {
            "tau_outside_zero_one": sum(
                not (0 <= float(row["tau"]) < 1) for row in transition_rows
            ),
            "cc64_outside_zero_127": 0,
            "nonpositive_iois": sum(
                float(row["ioi_duration_sec"]) <= 0 for row in interval_rows
            ),
            "same_timestamp_opposite_transitions": sum(
                int(row.get("same_timestamp_opposite_transition_count", 0))
                for row in analyzed_rows
            ),
            "cc64_events_before_first_onset": sum(
                int(row.get("cc64_events_before_first_onset", 0))
                for row in analyzed_rows
            ),
            "files_with_cc64_before_first_onset": sum(
                int(row.get("cc64_events_before_first_onset", 0)) > 0
                for row in analyzed_rows
            ),
            "cc64_events_at_or_after_last_onset": sum(
                int(row.get("cc64_events_at_or_after_last_onset", 0))
                for row in analyzed_rows
            ),
            "files_with_cc64_at_or_after_last_onset": sum(
                int(row.get("cc64_events_at_or_after_last_onset", 0)) > 0
                for row in analyzed_rows
            ),
            "long_ioi_threshold_sec": LONG_IOI_SECONDS,
            "long_iois": sum(
                int(row.get("long_ioi_count_gt_5s", 0))
                for row in analyzed_rows
            ),
            "near_threshold_rate_empirical_p95_per_1000_iois": near_p95,
            "transition_rate_empirical_p95_per_1000_iois": transition_p95,
            "high_near_threshold_crossing_rate_files": warning_files[
                "near_threshold_crossing_rate"
            ],
            "high_transition_rate_files": warning_files["transition_rate"],
        },
    }

    write_csv(output_dir / "per_file_summary.csv", per_file_rows)
    write_csv(output_dir / "interval_summary.csv", interval_rows)
    write_csv(output_dir / "transition_events.csv", transition_rows)
    write_csv(output_dir / "overflow_intervals.csv", overflow_rows)
    write_csv(output_dir / "crossing_noise_candidates.csv", crossing_rows)
    write_csv(
        output_dir / "depth_threshold_candidates.csv", depth_candidate_rows
    )
    write_csv(output_dir / "errors.csv", errors)
    with (output_dir / "aggregate_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            finite_or_none(aggregate),
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")

    figure_paths = save_figures(
        output_dir,
        interval_rows,
        transition_rows,
        crossing_rows,
        aggregate_depth_hist,
        depth_candidate_rows,
        machine_summary,
        slots,
        tertile_boundaries,
    )
    summary_text = build_summary_markdown(
        aggregate, depth_candidate_rows, warning_files
    )
    (output_dir / "summary.md").write_text(summary_text, encoding="utf-8")

    baseline = machine_summary["baseline_64"]
    counts = baseline["ioi_transition_count"]
    combined_slots = slots["first_two_with_overflow_included"]["combined"][
        "counts"
    ]
    print("")
    print(f"Analyzed files: {len(analyzed_rows)}")
    print(f"Failed files: {len(failed_rows)}")
    print(f"Total IOIs: {len(interval_rows)}")
    print(
        "Transition count IOIs 0/1/2/3+: "
        f"{counts['0']}/{counts['1']}/{counts['2']}/"
        f"{counts['3'] + counts['4+']}"
    )
    print(f"Two-slot coverage: {baseline['two_slot_coverage']:.6%}")
    print(
        "Total UP/DOWN inside valid IOIs: "
        f"{baseline['main_ioi_up']}/{baseline['main_ioi_down']}"
    )
    print(
        "Combined slot NONE/UP/DOWN: "
        f"{combined_slots['NONE']}/{combined_slots['UP']}/"
        f"{combined_slots['DOWN']}"
    )
    print(
        "Depth tertile boundaries: "
        f"{tertile_boundaries[0]}, {tertile_boundaries[1]}"
    )
    print(
        "Rapid reversals 25/50/100/200 ms: "
        + "/".join(
            str(baseline["rapid_reversal"][str(window)])
            for window in RAPID_WINDOWS_MS
        )
    )
    print(
        "Hysteresis transition totals baseline/narrow/wide: "
        f"{machine_summary['baseline_64']['total']}/"
        f"{machine_summary['narrow_68_60']['total']}/"
        f"{machine_summary['wide_72_56']['total']}"
    )
    print(f"Output: {output_dir}")
    print(f"Figures: {len(figure_paths)}")
    return aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit raw human MIDI for a two-slot event-based pedal target."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Human MIDI directory (default: {DEFAULT_INPUT.as_posix()})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Audit output directory (default: {DEFAULT_OUTPUT.as_posix()})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_analysis(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
