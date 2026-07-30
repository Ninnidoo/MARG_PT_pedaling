from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


TARGET_TICKS_PER_BEAT = 500
TARGET_TEMPO = 120
TARGET_TICKS_PER_SECOND = TARGET_TICKS_PER_BEAT * (TARGET_TEMPO / 60.0)


@dataclass(frozen=True)
class TokenizerOnlyConfig:
    """Small config object with the token ranges used by the official tokenizer.

    This mirrors PianoT5GemmaConfig constants in
    third_party/PianistTransformer/src/model/pianoformer.py lines 46-66.
    It is used only when importing the full model config would require missing
    model dependencies. The actual tokenizer functions are still imported from
    third_party/PianistTransformer/src/utils/midi.py.
    """

    mask_token_id: int = 1
    bos_token_id: int = 2
    play_token_id: int = 4
    pitch_start: int = 5
    velocity_start: int = 5 + 128
    timing_start: int = 5 + 128 + 128
    pedal_start: int = 5 + 128 + 128 + 5000

    @property
    def valid_id_range(self) -> list[tuple[int, int]]:
        return [
            (5, 133),
            (261, 5252),
            (133, 261),
            (261, 5261),
            (5261, 5389),
            (5261, 5389),
            (5261, 5389),
            (5261, 5389),
        ]


@dataclass(frozen=True)
class PedalEvent:
    time_sec: float
    cc64_value: int
    tick: int | None = None
    instrument_index: int | None = None


@dataclass(frozen=True)
class RepedalEvent:
    release_time: float
    redepress_time: float
    duration_ms: float
    minimum_cc64: int


def ticks_to_seconds(ticks: float) -> float:
    return float(ticks) / TARGET_TICKS_PER_SECOND


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return math.nan
    return numerator / denominator


def normalize_distribution(counts: np.ndarray) -> np.ndarray:
    total = float(np.sum(counts))
    if total == 0:
        return np.zeros_like(counts, dtype=float)
    return counts.astype(float) / total


def js_divergence(p_counts: np.ndarray, q_counts: np.ndarray) -> float:
    p = normalize_distribution(p_counts)
    q = normalize_distribution(q_counts)
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = (a > 0) & (b > 0)
        if not np.any(mask):
            return 0.0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def wasserstein_distance_1d(p_counts: np.ndarray, q_counts: np.ndarray) -> float:
    """Discrete 1D earth mover distance over CC64 values 0..127."""

    p = normalize_distribution(p_counts)
    q = normalize_distribution(q_counts)
    return float(np.sum(np.abs(np.cumsum(p) - np.cumsum(q))))


def histogram_128(values: np.ndarray) -> np.ndarray:
    rounded = np.rint(values).astype(int)
    clipped = np.clip(rounded, 0, 127)
    return np.bincount(clipped, minlength=128)[:128]


def extract_original_cc64_events(midi_obj) -> list[PedalEvent]:
    """Extract CC64 events from every non-drum instrument in original seconds."""

    tick_to_time = midi_obj.get_tick_to_time_mapping()
    events: list[PedalEvent] = []
    for instrument_index, instrument in enumerate(midi_obj.instruments):
        if instrument.is_drum:
            continue
        for cc in instrument.control_changes:
            if cc.number != 64:
                continue
            if cc.time < len(tick_to_time):
                time_sec = float(tick_to_time[cc.time])
            else:
                time_sec = float(tick_to_time[-1])
            events.append(
                PedalEvent(
                    time_sec=time_sec,
                    cc64_value=int(cc.value),
                    tick=int(cc.time),
                    instrument_index=instrument_index,
                )
            )
    events.sort(key=lambda e: (e.time_sec, e.instrument_index or 0, e.tick or 0))
    return events


def extract_normalized_cc64_events(midi_obj) -> list[PedalEvent]:
    """Extract CC64 events from a 120 BPM / 500 TPB normalized MIDI object."""

    events: list[PedalEvent] = []
    for instrument_index, instrument in enumerate(midi_obj.instruments):
        if instrument.is_drum:
            continue
        for cc in instrument.control_changes:
            if cc.number != 64:
                continue
            events.append(
                PedalEvent(
                    time_sec=ticks_to_seconds(cc.time),
                    cc64_value=int(cc.value),
                    tick=int(cc.time),
                    instrument_index=instrument_index,
                )
            )
    events.sort(key=lambda e: (e.time_sec, e.instrument_index or 0, e.tick or 0))
    return events


def extract_normalized_notes(midi_obj) -> list[dict[str, float | int]]:
    notes: list[dict[str, float | int]] = []
    for instrument_index, instrument in enumerate(midi_obj.instruments):
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            notes.append(
                {
                    "instrument_index": instrument_index,
                    "pitch": int(note.pitch),
                    "velocity": int(note.velocity),
                    "start_tick": int(note.start),
                    "end_tick": int(note.end),
                    "start_sec": ticks_to_seconds(note.start),
                    "end_sec": ticks_to_seconds(note.end),
                    "duration_sec": ticks_to_seconds(note.end - note.start),
                }
            )
    notes.sort(key=lambda n: (n["start_tick"], n["pitch"], n["end_tick"]))
    return notes


def midi_duration_seconds(midi_obj) -> float:
    max_tick = int(getattr(midi_obj, "max_tick", 0) or 0)
    for instrument in midi_obj.instruments:
        for note in instrument.notes:
            max_tick = max(max_tick, int(note.end))
        for cc in instrument.control_changes:
            max_tick = max(max_tick, int(cc.time))
    return ticks_to_seconds(max_tick)


def sample_zero_order_hold(
    events: Sequence[PedalEvent],
    grid_sec: np.ndarray,
    default_value: int = 0,
) -> np.ndarray:
    if len(grid_sec) == 0:
        return np.array([], dtype=float)
    if not events:
        return np.full_like(grid_sec, float(default_value), dtype=float)

    times = np.asarray([event.time_sec for event in events], dtype=float)
    values = np.asarray([event.cc64_value for event in events], dtype=float)
    indices = np.searchsorted(times, grid_sec, side="right") - 1
    out = np.full_like(grid_sec, float(default_value), dtype=float)
    valid = indices >= 0
    out[valid] = values[indices[valid]]
    return out


def value_at_time(events: Sequence[PedalEvent], time_sec: float, default_value: int = 0) -> int:
    if not events:
        return default_value
    times = [event.time_sec for event in events]
    index = bisect.bisect_right(times, time_sec) - 1
    if index < 0:
        return default_value
    return events[index].cc64_value


def token_rows_from_ids(config, ids: Sequence[int], normalized_midi_obj) -> list[dict[str, float | int]]:
    notes = normalized_midi_obj.instruments[0].notes if normalized_midi_obj.instruments else []
    intervals: list[int] = []
    last_time = 0
    for note in notes:
        intervals.append(int(note.start - last_time))
        last_time = int(note.start)
    intervals.append(4990)

    rows: list[dict[str, float | int]] = []
    last_time = 0
    note_count = min(len(notes), len(ids) // 8)
    for note_index in range(note_count):
        base = note_index * 8
        last_time += intervals[note_index]
        next_interval = intervals[note_index + 1]
        sample_ticks = [
            last_time,
            last_time + next_interval * 1 / 4,
            last_time + next_interval * 2 / 4,
            last_time + next_interval * 3 / 4,
        ]
        pedal_values = [int(ids[base + j] - config.pedal_start) for j in range(4, 8)]
        row: dict[str, float | int] = {
            "note_index": note_index,
            "pitch": int(ids[base] - config.pitch_start),
            "ioi_from_prev_ticks": int(ids[base + 1] - config.timing_start),
            "ioi_from_prev_sec": ticks_to_seconds(ids[base + 1] - config.timing_start),
            "velocity": int(ids[base + 2] - config.velocity_start),
            "duration_ticks": int(ids[base + 3] - config.timing_start),
            "duration_sec": ticks_to_seconds(ids[base + 3] - config.timing_start),
            "onset_tick": int(last_time),
            "onset_sec": ticks_to_seconds(last_time),
            "next_ioi_ticks": int(next_interval),
            "next_ioi_sec": ticks_to_seconds(next_interval),
        }
        for pedal_index, value in enumerate(pedal_values, start=1):
            row[f"pedal{pedal_index}"] = value
            row[f"pedal{pedal_index}_sample_tick"] = float(sample_ticks[pedal_index - 1])
            row[f"pedal{pedal_index}_sample_time_sec"] = ticks_to_seconds(sample_ticks[pedal_index - 1])
        rows.append(row)
    return rows


def occupancy_ratios(values: np.ndarray) -> dict[str, float]:
    total = len(values)
    return {
        "released_exact_ratio": safe_ratio(float(np.sum(values == 0)), total),
        "full_exact_ratio": safe_ratio(float(np.sum(values == 127)), total),
        "intermediate_time_ratio": safe_ratio(float(np.sum((values > 0) & (values < 127))), total),
        "released_0_15_ratio": safe_ratio(float(np.sum(values <= 15)), total),
        "partial_16_111_ratio": safe_ratio(float(np.sum((values >= 16) & (values <= 111))), total),
        "deep_112_127_ratio": safe_ratio(float(np.sum(values >= 112)), total),
    }


def detect_repedals(
    events: Sequence[PedalEvent],
    high_threshold: int = 96,
    low_threshold: int = 31,
    max_duration_ms: int = 300,
) -> list[RepedalEvent]:
    max_duration_sec = max_duration_ms / 1000.0
    current_value = 0
    high_seen = False
    candidate_release_time: float | None = None
    candidate_minimum = 127
    repedals: list[RepedalEvent] = []

    for event in sorted(events, key=lambda e: e.time_sec):
        previous_value = current_value
        current_value = event.cc64_value

        if candidate_release_time is None:
            if current_value >= high_threshold:
                high_seen = True
            if high_seen and previous_value > low_threshold and current_value <= low_threshold:
                candidate_release_time = event.time_sec
                candidate_minimum = current_value
            continue

        candidate_minimum = min(candidate_minimum, current_value)
        elapsed = event.time_sec - candidate_release_time
        if current_value >= high_threshold:
            if elapsed <= max_duration_sec:
                repedals.append(
                    RepedalEvent(
                        release_time=candidate_release_time,
                        redepress_time=event.time_sec,
                        duration_ms=elapsed * 1000.0,
                        minimum_cc64=int(candidate_minimum),
                    )
                )
            candidate_release_time = None
            candidate_minimum = 127
            high_seen = True
        elif elapsed > max_duration_sec:
            candidate_release_time = None
            candidate_minimum = 127
            high_seen = current_value >= high_threshold

    return repedals


def match_repedals(
    raw_repedals: Sequence[RepedalEvent],
    token_repedals: Sequence[RepedalEvent],
    tolerance_ms: int = 150,
) -> tuple[list[dict[str, float | int | bool]], int]:
    tolerance_sec = tolerance_ms / 1000.0
    used_token_indices: set[int] = set()
    rows: list[dict[str, float | int | bool]] = []

    for raw in raw_repedals:
        best_index = None
        best_score = math.inf
        for token_index, token in enumerate(token_repedals):
            if token_index in used_token_indices:
                continue
            release_error = abs(token.release_time - raw.release_time)
            redepress_error = abs(token.redepress_time - raw.redepress_time)
            if release_error <= tolerance_sec and redepress_error <= tolerance_sec:
                score = release_error + redepress_error
                if score < best_score:
                    best_score = score
                    best_index = token_index

        if best_index is None:
            rows.append(
                {
                    "raw_release_time": raw.release_time,
                    "raw_redepress_time": raw.redepress_time,
                    "duration_ms": raw.duration_ms,
                    "minimum_cc64": raw.minimum_cc64,
                    "preserved": False,
                    "token_release_time": math.nan,
                    "token_redepress_time": math.nan,
                    "release_timing_error_ms": math.nan,
                    "redepress_timing_error_ms": math.nan,
                }
            )
            continue

        token = token_repedals[best_index]
        used_token_indices.add(best_index)
        rows.append(
            {
                "raw_release_time": raw.release_time,
                "raw_redepress_time": raw.redepress_time,
                "duration_ms": raw.duration_ms,
                "minimum_cc64": raw.minimum_cc64,
                "preserved": True,
                "token_release_time": token.release_time,
                "token_redepress_time": token.redepress_time,
                "release_timing_error_ms": abs(token.release_time - raw.release_time) * 1000.0,
                "redepress_timing_error_ms": abs(token.redepress_time - raw.redepress_time) * 1000.0,
            }
        )

    return rows, len(used_token_indices)


def detect_threshold_transitions(
    events: Sequence[PedalEvent],
    threshold: int = 64,
) -> list[dict[str, float | int | str]]:
    current_value = 0
    transitions: list[dict[str, float | int | str]] = []
    for event in sorted(events, key=lambda e: e.time_sec):
        previous_value = current_value
        current_value = event.cc64_value
        if previous_value < threshold <= current_value:
            transitions.append(
                {
                    "kind": "down",
                    "time_sec": event.time_sec,
                    "from_value": previous_value,
                    "to_value": current_value,
                }
            )
        elif previous_value >= threshold > current_value:
            transitions.append(
                {
                    "kind": "up",
                    "time_sec": event.time_sec,
                    "from_value": previous_value,
                    "to_value": current_value,
                }
            )
    return transitions


def match_transitions(
    raw_transitions: Sequence[dict[str, float | int | str]],
    token_transitions: Sequence[dict[str, float | int | str]],
    tolerance_ms: int = 500,
) -> list[dict[str, float | str]]:
    tolerance_sec = tolerance_ms / 1000.0
    used: set[int] = set()
    matches: list[dict[str, float | str]] = []
    for raw in raw_transitions:
        raw_time = float(raw["time_sec"])
        raw_kind = str(raw["kind"])
        best_index = None
        best_error = math.inf
        for token_index, token in enumerate(token_transitions):
            if token_index in used or str(token["kind"]) != raw_kind:
                continue
            error = abs(float(token["time_sec"]) - raw_time)
            if error <= tolerance_sec and error < best_error:
                best_error = error
                best_index = token_index
        if best_index is None:
            continue
        used.add(best_index)
        token = token_transitions[best_index]
        matches.append(
            {
                "kind": raw_kind,
                "raw_time_sec": raw_time,
                "token_time_sec": float(token["time_sec"]),
                "abs_error_ms": best_error * 1000.0,
            }
        )
    return matches


def summarize_errors_ms(errors_ms: Sequence[float]) -> dict[str, float]:
    values = np.asarray([v for v in errors_ms if not math.isnan(float(v))], dtype=float)
    if len(values) == 0:
        return {
            "mean_transition_error_ms": math.nan,
            "median_transition_error_ms": math.nan,
            "std_transition_error_ms": math.nan,
            "p90_transition_error_ms": math.nan,
            "p95_transition_error_ms": math.nan,
        }
    return {
        "mean_transition_error_ms": float(np.mean(values)),
        "median_transition_error_ms": float(np.median(values)),
        "std_transition_error_ms": float(np.std(values)),
        "p90_transition_error_ms": float(np.percentile(values, 90)),
        "p95_transition_error_ms": float(np.percentile(values, 95)),
    }


def note_density_at_times(note_onsets: np.ndarray, grid_sec: np.ndarray, window_sec: float) -> np.ndarray:
    if len(grid_sec) == 0:
        return np.array([], dtype=float)
    if len(note_onsets) == 0:
        return np.zeros_like(grid_sec, dtype=float)
    half = window_sec / 2.0
    left = np.searchsorted(note_onsets, grid_sec - half, side="left")
    right = np.searchsorted(note_onsets, grid_sec + half, side="right")
    return (right - left).astype(float) / window_sec


def bin_label_for_value(value: float, bins: Sequence[tuple[str, float, float]]) -> str:
    for label, start, end in bins:
        if start <= value < end:
            return label
    return bins[-1][0]


def ioi_bins() -> list[tuple[str, float, float]]:
    return [
        ("<100ms", 0.0, 0.100),
        ("100-250ms", 0.100, 0.250),
        ("250-500ms", 0.250, 0.500),
        ("500-1000ms", 0.500, 1.000),
        (">=1000ms", 1.000, math.inf),
    ]


def rows_to_events(rows: Iterable[dict[str, float | int]]) -> list[PedalEvent]:
    events: list[PedalEvent] = []
    for row in rows:
        events.append(PedalEvent(time_sec=float(row["time_sec"]), cc64_value=int(row["cc64_value"])))
    events.sort(key=lambda e: e.time_sec)
    return events
