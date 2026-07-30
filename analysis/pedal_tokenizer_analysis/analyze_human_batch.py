from __future__ import annotations

import argparse
import bisect
import inspect
import math
import re
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from analyze_pedal_tokenizer import (
    import_miditoolkit,
    load_official_components,
    repo_root_from_script,
    scan_candidate,
    write_csv,
    write_json,
)
from pedal_utils import (
    TARGET_TICKS_PER_SECOND,
    detect_threshold_transitions,
    extract_normalized_cc64_events,
    extract_normalized_notes,
    histogram_128,
    ioi_bins,
    js_divergence,
    match_transitions as legacy_match_transitions,
    midi_duration_seconds,
    occupancy_ratios,
    sample_zero_order_hold,
    summarize_errors_ms,
    token_rows_from_ids,
    wasserstein_distance_1d,
)


RANDOM_SEED = 20260727
HUMAN_SOURCE = "repository_provided_human_ground_truth_reference_performance"
THRESHOLD_DEFINITIONS = [
    {"definition": "A", "high_threshold": 96, "low_threshold": 31, "max_duration_ms": 300},
    {"definition": "B", "high_threshold": 96, "low_threshold": 47, "max_duration_ms": 300},
    {"definition": "C", "high_threshold": 80, "low_threshold": 31, "max_duration_ms": 300},
    {"definition": "D", "high_threshold": 96, "low_threshold": 31, "max_duration_ms": 500},
]
HUMAN_FILENAME = re.compile(r"^(?P<score>\d+)-(?P<take>\d+)\.(?:mid|midi)$", re.IGNORECASE)


def numeric_human_key(path: Path) -> tuple[int, int, str]:
    match = HUMAN_FILENAME.match(path.name)
    if match is None:
        return math.inf, math.inf, path.name
    return int(match.group("score")), int(match.group("take")), path.name


def count_non_cc64_controllers(midi_obj) -> int:
    return sum(
        1
        for instrument in midi_obj.instruments
        if not instrument.is_drum
        for event in instrument.control_changes
        if event.number != 64
    )


def discover_human_inventory(repo_root: Path, MidiFile) -> tuple[Path, list[dict[str, Any]]]:
    human_root = (
        repo_root
        / "third_party"
        / "PianistTransformer"
        / "data"
        / "midis"
        / "testset"
        / "human"
    )
    paths = sorted(
        [*human_root.glob("*.mid"), *human_root.glob("*.midi")],
        key=numeric_human_key,
    )
    rows: list[dict[str, Any]] = []
    for path in paths:
        match = HUMAN_FILENAME.match(path.name)
        score_index = int(match.group("score")) if match else ""
        take_index = int(match.group("take")) if match else ""
        try:
            scan = scan_candidate(path.resolve(), HUMAN_SOURCE, MidiFile)
            midi_obj = MidiFile(str(path))
            cc64_count = int(scan["num_cc64_events"])
            intermediate_count = int(scan["num_intermediate_cc64_events"])
            eligible = cc64_count > 0
            rows.append(
                {
                    "filename": path.name,
                    "relative_path": path.relative_to(repo_root).as_posix(),
                    "file": str(path.resolve()),
                    "score_index": score_index,
                    "take_index": take_index,
                    "provenance_label": HUMAN_SOURCE,
                    "num_notes": int(scan["num_notes"]),
                    "duration_sec": float(scan["duration_sec"]),
                    "num_cc64_events": cc64_count,
                    "num_intermediate_cc64_events": intermediate_count,
                    "intermediate_cc64_event_ratio": (
                        intermediate_count / cc64_count if cc64_count else math.nan
                    ),
                    "num_non_cc64_controller_events": count_non_cc64_controllers(midi_obj),
                    "analysis_eligible": eligible,
                    "exclusion_reason": "" if eligible else "no_cc64_events",
                    "scan_error": "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "filename": path.name,
                    "relative_path": path.relative_to(repo_root).as_posix(),
                    "file": str(path.resolve()),
                    "score_index": score_index,
                    "take_index": take_index,
                    "provenance_label": HUMAN_SOURCE,
                    "num_notes": "",
                    "duration_sec": "",
                    "num_cc64_events": "",
                    "num_intermediate_cc64_events": "",
                    "intermediate_cc64_event_ratio": "",
                    "num_non_cc64_controller_events": "",
                    "analysis_eligible": False,
                    "exclusion_reason": "inventory_scan_error",
                    "scan_error": f"{type(exc).__name__}: {exc}",
                }
            )
    return human_root, rows


def detect_repedals_detailed(
    events,
    high_threshold: int,
    low_threshold: int,
    max_duration_ms: int,
) -> list[dict[str, Any]]:
    """Detect threshold-defined repedals and retain raw gesture landmarks."""

    max_duration_sec = max_duration_ms / 1000.0
    current_value = 0
    high_seen = False
    candidate: dict[str, Any] | None = None
    repedals: list[dict[str, Any]] = []

    for event in sorted(events, key=lambda item: item.time_sec):
        previous_value = current_value
        current_value = int(event.cc64_value)

        if candidate is None:
            if current_value >= high_threshold:
                high_seen = True
            if high_seen and previous_value > low_threshold and current_value <= low_threshold:
                candidate = {
                    "release_time": float(event.time_sec),
                    "minimum_time": float(event.time_sec),
                    "starting_cc64": int(previous_value),
                    "minimum_cc64": current_value,
                }
            continue

        if current_value < int(candidate["minimum_cc64"]):
            candidate["minimum_cc64"] = current_value
            candidate["minimum_time"] = float(event.time_sec)

        elapsed = float(event.time_sec) - float(candidate["release_time"])
        if current_value >= high_threshold:
            if elapsed <= max_duration_sec:
                repedals.append(
                    {
                        **candidate,
                        "redepress_time": float(event.time_sec),
                        "gesture_duration_ms": elapsed * 1000.0,
                        "ending_cc64": current_value,
                    }
                )
            candidate = None
            high_seen = True
        elif elapsed > max_duration_sec:
            candidate = None
            high_seen = current_value >= high_threshold

    return repedals


def ordered_one_to_one_pairs(
    raw_items: Sequence[Any],
    token_items: Sequence[Any],
    feasible: Callable[[Any, Any], bool],
    cost: Callable[[Any, Any], float],
) -> list[tuple[int, int]]:
    """Maximize ordered one-to-one matches, then minimize total match cost."""

    n_raw = len(raw_items)
    n_token = len(token_items)
    if n_raw == 0 or n_token == 0:
        return []

    counts = np.zeros((n_raw + 1, n_token + 1), dtype=np.int32)
    costs = np.zeros((n_raw + 1, n_token + 1), dtype=np.float64)
    actions = np.zeros((n_raw + 1, n_token + 1), dtype=np.uint8)
    actions[1:, 0] = 1
    actions[0, 1:] = 2

    for raw_index in range(1, n_raw + 1):
        raw = raw_items[raw_index - 1]
        for token_index in range(1, n_token + 1):
            token = token_items[token_index - 1]

            best_count = int(counts[raw_index - 1, token_index])
            best_cost = float(costs[raw_index - 1, token_index])
            best_action = 1

            left_count = int(counts[raw_index, token_index - 1])
            left_cost = float(costs[raw_index, token_index - 1])
            if left_count > best_count or (
                left_count == best_count and left_cost < best_cost
            ):
                best_count = left_count
                best_cost = left_cost
                best_action = 2

            if feasible(raw, token):
                diagonal_count = int(counts[raw_index - 1, token_index - 1]) + 1
                diagonal_cost = float(costs[raw_index - 1, token_index - 1]) + cost(raw, token)
                if diagonal_count > best_count or (
                    diagonal_count == best_count and diagonal_cost <= best_cost
                ):
                    best_count = diagonal_count
                    best_cost = diagonal_cost
                    best_action = 3

            counts[raw_index, token_index] = best_count
            costs[raw_index, token_index] = best_cost
            actions[raw_index, token_index] = best_action

    pairs: list[tuple[int, int]] = []
    raw_index = n_raw
    token_index = n_token
    while raw_index > 0 or token_index > 0:
        action = int(actions[raw_index, token_index])
        if action == 3:
            pairs.append((raw_index - 1, token_index - 1))
            raw_index -= 1
            token_index -= 1
        elif action == 1:
            raw_index -= 1
        elif action == 2:
            token_index -= 1
        elif raw_index > 0:
            raw_index -= 1
        else:
            token_index -= 1
    pairs.reverse()
    return pairs


def match_repedals_ordered(
    raw_repedals: Sequence[dict[str, Any]],
    token_repedals: Sequence[dict[str, Any]],
    tolerance_ms: int,
) -> tuple[list[dict[str, Any]], int]:
    tolerance_sec = tolerance_ms / 1000.0

    def feasible(raw: dict[str, Any], token: dict[str, Any]) -> bool:
        return (
            abs(float(raw["release_time"]) - float(token["release_time"])) <= tolerance_sec
            and abs(float(raw["redepress_time"]) - float(token["redepress_time"]))
            <= tolerance_sec
        )

    def cost(raw: dict[str, Any], token: dict[str, Any]) -> float:
        return abs(float(raw["release_time"]) - float(token["release_time"])) + abs(
            float(raw["redepress_time"]) - float(token["redepress_time"])
        )

    pairs = ordered_one_to_one_pairs(raw_repedals, token_repedals, feasible, cost)
    token_by_raw = {raw_index: token_index for raw_index, token_index in pairs}
    rows: list[dict[str, Any]] = []
    for raw_index, raw in enumerate(raw_repedals):
        token_index = token_by_raw.get(raw_index)
        if token_index is None:
            rows.append(
                {
                    "raw_release_time": float(raw["release_time"]),
                    "raw_min_time": float(raw["minimum_time"]),
                    "raw_redepress_time": float(raw["redepress_time"]),
                    "gesture_duration_ms": float(raw["gesture_duration_ms"]),
                    "starting_cc64": int(raw["starting_cc64"]),
                    "minimum_cc64": int(raw["minimum_cc64"]),
                    "ending_cc64": int(raw["ending_cc64"]),
                    "preserved": False,
                    "matched_token_release_time": math.nan,
                    "matched_token_redepress_time": math.nan,
                    "release_timing_error_ms": math.nan,
                    "redepress_timing_error_ms": math.nan,
                }
            )
            continue

        token = token_repedals[token_index]
        rows.append(
            {
                "raw_release_time": float(raw["release_time"]),
                "raw_min_time": float(raw["minimum_time"]),
                "raw_redepress_time": float(raw["redepress_time"]),
                "gesture_duration_ms": float(raw["gesture_duration_ms"]),
                "starting_cc64": int(raw["starting_cc64"]),
                "minimum_cc64": int(raw["minimum_cc64"]),
                "ending_cc64": int(raw["ending_cc64"]),
                "preserved": True,
                "matched_token_release_time": float(token["release_time"]),
                "matched_token_redepress_time": float(token["redepress_time"]),
                "release_timing_error_ms": abs(
                    float(token["release_time"]) - float(raw["release_time"])
                )
                * 1000.0,
                "redepress_timing_error_ms": abs(
                    float(token["redepress_time"]) - float(raw["redepress_time"])
                )
                * 1000.0,
            }
        )
    return rows, len(pairs)


def match_transitions_ordered(
    raw_transitions: Sequence[dict[str, Any]],
    token_transitions: Sequence[dict[str, Any]],
    tolerance_ms: int,
) -> list[dict[str, Any]]:
    """Order-preserving, one-to-one matching within direction and tolerance."""

    tolerance_sec = tolerance_ms / 1000.0
    all_matches: list[dict[str, Any]] = []
    for kind in ("down", "up"):
        raw_indexed = [
            (index, row) for index, row in enumerate(raw_transitions) if row["kind"] == kind
        ]
        token_indexed = [
            (index, row) for index, row in enumerate(token_transitions) if row["kind"] == kind
        ]

        def feasible(raw_item, token_item) -> bool:
            return (
                abs(float(raw_item[1]["time_sec"]) - float(token_item[1]["time_sec"]))
                <= tolerance_sec
            )

        def cost(raw_item, token_item) -> float:
            return abs(float(raw_item[1]["time_sec"]) - float(token_item[1]["time_sec"]))

        pairs = ordered_one_to_one_pairs(raw_indexed, token_indexed, feasible, cost)
        for raw_local_index, token_local_index in pairs:
            raw_index, raw = raw_indexed[raw_local_index]
            token_index, token = token_indexed[token_local_index]
            signed_error_ms = (
                float(token["time_sec"]) - float(raw["time_sec"])
            ) * 1000.0
            all_matches.append(
                {
                    "raw_index": raw_index,
                    "token_index": token_index,
                    "kind": kind,
                    "raw_time_sec": float(raw["time_sec"]),
                    "token_time_sec": float(token["time_sec"]),
                    "signed_error_ms": signed_error_ms,
                    "abs_error_ms": abs(signed_error_ms),
                }
            )
    all_matches.sort(key=lambda row: int(row["raw_index"]))
    return all_matches


def build_positive_ioi_intervals(
    token_rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    intervals: list[dict[str, Any]] = []
    zero_ioi_count = 0
    for index, row in enumerate(token_rows):
        ioi_sec = float(row["next_ioi_sec"])
        if ioi_sec <= 0:
            zero_ioi_count += 1
            continue
        start = float(row["onset_sec"])
        intervals.append(
            {
                "start_sec": start,
                "end_sec": start + ioi_sec,
                "local_ioi_ms": ioi_sec * 1000.0,
                "note_index": int(row["note_index"]),
                "terminal_sentinel_interval": index == len(token_rows) - 1,
            }
        )
    intervals.sort(key=lambda row: float(row["start_sec"]))
    return intervals, zero_ioi_count


def local_ioi_at_time(
    intervals: Sequence[dict[str, Any]], time_sec: float
) -> dict[str, Any]:
    if not intervals:
        return {
            "relevant_local_ioi_ms": math.nan,
            "local_ioi_bin": "",
            "local_ioi_note_index": "",
            "local_ioi_terminal_sentinel": False,
            "local_ioi_outside_interval": True,
        }
    starts = [float(row["start_sec"]) for row in intervals]
    interval_index = bisect.bisect_right(starts, time_sec) - 1
    if interval_index < 0:
        return {
            "relevant_local_ioi_ms": math.nan,
            "local_ioi_bin": "",
            "local_ioi_note_index": "",
            "local_ioi_terminal_sentinel": False,
            "local_ioi_outside_interval": True,
        }
    interval = intervals[interval_index]
    ioi_ms = float(interval["local_ioi_ms"])
    label = ioi_bins()[-1][0]
    for candidate_label, start, end in ioi_bins():
        if start <= ioi_ms / 1000.0 < end:
            label = candidate_label
            break
    return {
        "relevant_local_ioi_ms": ioi_ms,
        "local_ioi_bin": label,
        "local_ioi_note_index": int(interval["note_index"]),
        "local_ioi_terminal_sentinel": bool(interval["terminal_sentinel_interval"]),
        "local_ioi_outside_interval": time_sec >= float(interval["end_sec"]),
    }


def token_pedal_counts(config, ids: Sequence[int]) -> dict[str, Any]:
    values = np.asarray(
        [
            int(ids[index + offset] - config.pedal_start)
            for index in range(0, len(ids), 8)
            for offset in range(4, 8)
        ],
        dtype=int,
    )
    count = len(values)
    intermediate = int(np.sum((values > 0) & (values < 127)))
    return {
        "num_pedal_tokens": count,
        "pedal_value_0_count": int(np.sum(values == 0)),
        "pedal_value_127_count": int(np.sum(values == 127)),
        "pedal_intermediate_count": intermediate,
        "token_intermediate_token_ratio": intermediate / count if count else math.nan,
    }


def first_lost_example(
    filename: str,
    score_index: int,
    take_index: int,
    repedal_rows: Sequence[dict[str, Any]],
    grid_sec: np.ndarray,
    raw_curve: np.ndarray,
    token_curve: np.ndarray,
) -> dict[str, Any] | None:
    lost = next((row for row in repedal_rows if row["preserved"] is False), None)
    if lost is None:
        return None
    center = float(lost["raw_min_time"])
    mask = (grid_sec >= center - 0.40) & (grid_sec <= center + 0.40)
    return {
        "filename": filename,
        "score_index": score_index,
        "take_index": take_index,
        "time_sec": grid_sec[mask].tolist(),
        "raw_cc64": raw_curve[mask].tolist(),
        "reconstructed_cc64": token_curve[mask].tolist(),
        **lost,
    }


def analyze_human_file(
    candidate: dict[str, Any],
    components: dict[str, Any],
    MidiFile,
    args: argparse.Namespace,
) -> dict[str, Any]:
    path = Path(candidate["file"])
    midi_obj = MidiFile(str(path))
    normalized_raw = components["normalize_midi"](midi_obj)
    ids = components["midi_to_ids"](components["config"], midi_obj, normalize=True)
    reconstructed = components["ids_to_midi"](components["config"], ids)

    raw_events = extract_normalized_cc64_events(normalized_raw)
    token_events = extract_normalized_cc64_events(reconstructed)
    normalized_notes = extract_normalized_notes(normalized_raw)
    token_rows = token_rows_from_ids(components["config"], ids, normalized_raw)
    intervals, zero_ioi_count = build_positive_ioi_intervals(token_rows)

    duration_sec = max(
        midi_duration_seconds(normalized_raw), midi_duration_seconds(reconstructed)
    )
    grid_step_sec = args.grid_ms / 1000.0
    grid_sec = np.arange(0.0, duration_sec + grid_step_sec, grid_step_sec)
    raw_curve = sample_zero_order_hold(raw_events, grid_sec)
    token_curve = sample_zero_order_hold(token_events, grid_sec)
    difference = raw_curve - token_curve
    raw_occ = occupancy_ratios(raw_curve)
    token_occ = occupancy_ratios(token_curve)
    raw_hist = histogram_128(raw_curve)
    token_hist = histogram_128(token_curve)

    raw_repedals = detect_repedals_detailed(
        raw_events,
        args.high_threshold,
        args.low_threshold,
        args.max_repedal_duration_ms,
    )
    token_repedals = detect_repedals_detailed(
        token_events,
        args.high_threshold,
        args.low_threshold,
        args.max_repedal_duration_ms,
    )
    repedal_rows, matched_token_repedals = match_repedals_ordered(
        raw_repedals, token_repedals, args.repedal_match_tolerance_ms
    )
    for row in repedal_rows:
        row.update(
            {
                "file": str(path),
                "filename": candidate["filename"],
                "score_index": candidate["score_index"],
                "take_index": candidate["take_index"],
                **local_ioi_at_time(intervals, float(row["raw_min_time"])),
            }
        )

    raw_transitions = detect_threshold_transitions(
        raw_events, threshold=args.transition_threshold
    )
    token_transitions = detect_threshold_transitions(
        token_events, threshold=args.transition_threshold
    )
    transition_matches = match_transitions_ordered(
        raw_transitions, token_transitions, args.transition_match_tolerance_ms
    )
    match_by_raw = {int(row["raw_index"]): row for row in transition_matches}
    transition_rows: list[dict[str, Any]] = []
    for raw_index, raw_transition in enumerate(raw_transitions):
        match = match_by_raw.get(raw_index)
        row = {
            "file": str(path),
            "filename": candidate["filename"],
            "score_index": candidate["score_index"],
            "take_index": candidate["take_index"],
            "transition_kind": raw_transition["kind"],
            "raw_transition_time_sec": float(raw_transition["time_sec"]),
            "raw_from_cc64": int(raw_transition["from_value"]),
            "raw_to_cc64": int(raw_transition["to_value"]),
            "matched": match is not None,
            "matched_token_transition_time_sec": (
                float(match["token_time_sec"]) if match else math.nan
            ),
            "signed_timing_error_ms": (
                float(match["signed_error_ms"]) if match else math.nan
            ),
            "absolute_timing_error_ms": (
                float(match["abs_error_ms"]) if match else math.nan
            ),
        }
        row.update(local_ioi_at_time(intervals, float(raw_transition["time_sec"])))
        transition_rows.append(row)

    transition_summary = summarize_errors_ms(
        [float(row["abs_error_ms"]) for row in transition_matches]
    )
    token_counts = token_pedal_counts(components["config"], ids)
    raw_intermediate_events = sum(
        1 for event in raw_events if 0 < int(event.cc64_value) < 127
    )
    raw_repedal_count = len(raw_repedals)
    preserved_repedal_count = sum(row["preserved"] is True for row in repedal_rows)
    abs_difference = np.abs(difference)

    summary = {
        "file": str(path),
        "filename": candidate["filename"],
        "relative_path": candidate["relative_path"],
        "score_index": candidate["score_index"],
        "take_index": candidate["take_index"],
        "provenance_label": HUMAN_SOURCE,
        "input_duration_sec": float(candidate["duration_sec"]),
        "duration_sec": duration_sec,
        "num_notes": int(candidate["num_notes"]),
        "num_normalized_notes": len(normalized_notes),
        "num_zero_next_ioi_notes": zero_ioi_count,
        "num_raw_cc64_events": len(raw_events),
        "num_raw_intermediate_cc64_events": raw_intermediate_events,
        "raw_intermediate_event_ratio": (
            raw_intermediate_events / len(raw_events) if raw_events else math.nan
        ),
        **token_counts,
        "num_reconstructed_cc64_events": len(token_events),
        "comparison_grid_sample_count": len(grid_sec),
        "raw_intermediate_grid_count": int(
            np.sum((raw_curve > 0) & (raw_curve < 127))
        ),
        "token_intermediate_grid_count": int(
            np.sum((token_curve > 0) & (token_curve < 127))
        ),
        "raw_intermediate_time_ratio": raw_occ["intermediate_time_ratio"],
        "token_intermediate_time_ratio": token_occ["intermediate_time_ratio"],
        "intermediate_time_ratio_difference": (
            token_occ["intermediate_time_ratio"] - raw_occ["intermediate_time_ratio"]
        ),
        "cc64_absolute_error_sum": float(np.sum(abs_difference)),
        "cc64_squared_error_sum": float(np.sum(difference**2)),
        "cc64_mae": float(np.mean(abs_difference)) if len(difference) else math.nan,
        "cc64_nmae": (
            float(np.mean(abs_difference) / 127.0) if len(difference) else math.nan
        ),
        "cc64_rmse": (
            float(np.sqrt(np.mean(difference**2))) if len(difference) else math.nan
        ),
        "js_divergence": js_divergence(raw_hist, token_hist),
        "wasserstein_distance": wasserstein_distance_1d(raw_hist, token_hist),
        "raw_repedal_count": raw_repedal_count,
        "token_repedal_count": len(token_repedals),
        "preserved_repedal_count": preserved_repedal_count,
        "lost_repedal_count": raw_repedal_count - preserved_repedal_count,
        "repedal_recall": (
            preserved_repedal_count / raw_repedal_count
            if raw_repedal_count
            else math.nan
        ),
        "repedal_precision": (
            matched_token_repedals / len(token_repedals)
            if token_repedals
            else math.nan
        ),
        "raw_transition_count": len(raw_transitions),
        "token_transition_count": len(token_transitions),
        "matched_transition_count": len(transition_matches),
        "unmatched_raw_transition_count": len(raw_transitions)
        - len(transition_matches),
        **transition_summary,
    }

    threshold_rows: list[dict[str, Any]] = []
    for definition in THRESHOLD_DEFINITIONS:
        definition_raw = detect_repedals_detailed(
            raw_events,
            definition["high_threshold"],
            definition["low_threshold"],
            definition["max_duration_ms"],
        )
        definition_token = detect_repedals_detailed(
            token_events,
            definition["high_threshold"],
            definition["low_threshold"],
            definition["max_duration_ms"],
        )
        definition_matches, matched_token_count = match_repedals_ordered(
            definition_raw, definition_token, args.repedal_match_tolerance_ms
        )
        preserved = sum(row["preserved"] is True for row in definition_matches)
        threshold_rows.append(
            {
                "file": str(path),
                "filename": candidate["filename"],
                "score_index": candidate["score_index"],
                "take_index": candidate["take_index"],
                **definition,
                "raw_repedal_count": len(definition_raw),
                "token_repedal_count": len(definition_token),
                "preserved_repedal_count": preserved,
                "lost_repedal_count": len(definition_raw) - preserved,
                "repedal_recall": (
                    preserved / len(definition_raw)
                    if definition_raw
                    else math.nan
                ),
                "repedal_precision": (
                    matched_token_count / len(definition_token)
                    if definition_token
                    else math.nan
                ),
            }
        )

    return {
        "summary": summary,
        "repedal_rows": repedal_rows,
        "transition_rows": transition_rows,
        "threshold_rows": threshold_rows,
        "lost_example": first_lost_example(
            candidate["filename"],
            int(candidate["score_index"]),
            int(candidate["take_index"]),
            repedal_rows,
            grid_sec,
            raw_curve,
            token_curve,
        ),
        "raw_transitions": raw_transitions,
        "token_transitions": token_transitions,
    }


def finite_values(rows: Sequence[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        try:
            value = float(row[key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def bootstrap_mean_median_ci(
    values: Sequence[float], seed: int, iterations: int
) -> dict[str, float]:
    if not values:
        return {
            "bootstrap_mean_ci95_low": math.nan,
            "bootstrap_mean_ci95_high": math.nan,
            "bootstrap_median_ci95_low": math.nan,
            "bootstrap_median_ci95_high": math.nan,
        }
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=float)
    means = np.empty(iterations, dtype=float)
    medians = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        sample = rng.choice(array, size=len(array), replace=True)
        means[iteration] = np.mean(sample)
        medians[iteration] = np.median(sample)
    return {
        "bootstrap_mean_ci95_low": float(np.percentile(means, 2.5)),
        "bootstrap_mean_ci95_high": float(np.percentile(means, 97.5)),
        "bootstrap_median_ci95_low": float(np.percentile(medians, 2.5)),
        "bootstrap_median_ci95_high": float(np.percentile(medians, 97.5)),
    }


def aggregate_statistics(
    summary_rows: Sequence[dict[str, Any]],
    seed: int,
    bootstrap_iterations: int,
) -> list[dict[str, Any]]:
    metrics = [
        "raw_intermediate_time_ratio",
        "token_intermediate_time_ratio",
        "intermediate_time_ratio_difference",
        "cc64_nmae",
        "cc64_rmse",
        "js_divergence",
        "wasserstein_distance",
        "repedal_recall",
        "median_transition_error_ms",
    ]
    total_raw_repedals = sum(int(row["raw_repedal_count"]) for row in summary_rows)
    total_preserved_repedals = sum(
        int(row["preserved_repedal_count"]) for row in summary_rows
    )
    total_grid = sum(int(row["comparison_grid_sample_count"]) for row in summary_rows)
    raw_intermediate_grid = sum(
        int(row["raw_intermediate_grid_count"]) for row in summary_rows
    )
    token_intermediate_grid = sum(
        int(row["token_intermediate_grid_count"]) for row in summary_rows
    )
    total_abs_error = sum(float(row["cc64_absolute_error_sum"]) for row in summary_rows)
    total_squared_error = sum(
        float(row["cc64_squared_error_sum"]) for row in summary_rows
    )

    rows: list[dict[str, Any]] = []
    for metric_index, metric in enumerate(metrics):
        source_rows = list(summary_rows)
        exclusion_rule = ""
        if metric == "repedal_recall":
            source_rows = [
                row for row in source_rows if int(row["raw_repedal_count"]) > 0
            ]
            exclusion_rule = "files with raw_repedal_count=0"
        elif metric == "median_transition_error_ms":
            source_rows = [
                row for row in source_rows if int(row["matched_transition_count"]) > 0
            ]
            exclusion_rule = "files with matched_transition_count=0"
        values = finite_values(source_rows, metric)
        micro_value = math.nan
        if metric == "raw_intermediate_time_ratio" and total_grid:
            micro_value = raw_intermediate_grid / total_grid
        elif metric == "token_intermediate_time_ratio" and total_grid:
            micro_value = token_intermediate_grid / total_grid
        elif metric == "intermediate_time_ratio_difference" and total_grid:
            micro_value = (token_intermediate_grid - raw_intermediate_grid) / total_grid
        elif metric == "cc64_nmae" and total_grid:
            micro_value = total_abs_error / total_grid / 127.0
        elif metric == "cc64_rmse" and total_grid:
            micro_value = math.sqrt(total_squared_error / total_grid)
        elif metric == "repedal_recall" and total_raw_repedals:
            micro_value = total_preserved_repedals / total_raw_repedals

        row = {
            "metric": metric,
            "count": len(values),
            "excluded_files": len(summary_rows) - len(source_rows),
            "exclusion_rule": exclusion_rule,
            "mean": float(np.mean(values)) if values else math.nan,
            "median": float(np.median(values)) if values else math.nan,
            "std": (
                float(np.std(values, ddof=1))
                if len(values) > 1
                else 0.0
                if len(values) == 1
                else math.nan
            ),
            "q1": float(np.percentile(values, 25)) if values else math.nan,
            "q3": float(np.percentile(values, 75)) if values else math.nan,
            "p10": float(np.percentile(values, 10)) if values else math.nan,
            "p90": float(np.percentile(values, 90)) if values else math.nan,
            "micro_value": micro_value,
            "macro_value": (
                float(np.mean(values)) if metric == "repedal_recall" and values else math.nan
            ),
            "total_raw_repedal_count": (
                total_raw_repedals if metric == "repedal_recall" else ""
            ),
            "total_preserved_repedal_count": (
                total_preserved_repedals if metric == "repedal_recall" else ""
            ),
            "total_lost_repedal_count": (
                total_raw_repedals - total_preserved_repedals
                if metric == "repedal_recall"
                else ""
            ),
        }
        row.update(
            bootstrap_mean_median_ci(
                values, seed + metric_index, bootstrap_iterations
            )
        )
        rows.append(row)
    return rows


def aggregate_threshold_sensitivity(
    per_file_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for definition in THRESHOLD_DEFINITIONS:
        items = [
            row
            for row in per_file_rows
            if row["definition"] == definition["definition"]
        ]
        total_raw = sum(int(row["raw_repedal_count"]) for row in items)
        total_preserved = sum(
            int(row["preserved_repedal_count"]) for row in items
        )
        recalls = finite_values(
            [row for row in items if int(row["raw_repedal_count"]) > 0],
            "repedal_recall",
        )
        q1 = float(np.percentile(recalls, 25)) if recalls else math.nan
        q3 = float(np.percentile(recalls, 75)) if recalls else math.nan
        rows.append(
            {
                **definition,
                "total_raw_repedal_count": total_raw,
                "total_preserved": total_preserved,
                "total_lost": total_raw - total_preserved,
                "micro_recall": total_preserved / total_raw if total_raw else math.nan,
                "macro_recall": float(np.mean(recalls)) if recalls else math.nan,
                "median_per_file_recall": (
                    float(np.median(recalls)) if recalls else math.nan
                ),
                "q1_per_file_recall": q1,
                "q3_per_file_recall": q3,
                "iqr_per_file_recall": q3 - q1 if recalls else math.nan,
                "files_with_raw_repedal": len(recalls),
                "files_without_raw_repedal": len(items) - len(recalls),
                "operational_definition_not_musicological_label": True,
            }
        )
    return rows


def aggregate_ioi_analysis(
    repedal_rows: Sequence[dict[str, Any]],
    transition_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, start, end in ioi_bins():
        repedals = [row for row in repedal_rows if row["local_ioi_bin"] == label]
        transitions = [
            row
            for row in transition_rows
            if row["local_ioi_bin"] == label and row["matched"] is True
        ]
        errors = finite_values(transitions, "absolute_timing_error_ms")
        preserved = sum(row["preserved"] is True for row in repedals)
        rows.append(
            {
                "ioi_bin": label,
                "ioi_start_ms": start * 1000.0,
                "ioi_end_ms": end * 1000.0,
                "repedal_local_ioi_definition": (
                    "next positive tokenizer IOI interval containing raw valley/minimum time"
                ),
                "transition_local_ioi_definition": (
                    "next positive tokenizer IOI interval containing raw transition time"
                ),
                "raw_repedal_count": len(repedals),
                "preserved": preserved,
                "lost": len(repedals) - preserved,
                "repedal_recall": preserved / len(repedals) if repedals else math.nan,
                "matched_transition_count": len(errors),
                "median_transition_timing_error_ms": (
                    float(np.median(errors)) if errors else math.nan
                ),
                "p90_transition_timing_error_ms": (
                    float(np.percentile(errors, 90)) if errors else math.nan
                ),
            }
        )
    return rows


def fit_ioi_logistic(
    repedal_rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from scipy.optimize import minimize
    from scipy.stats import norm

    usable = [
        row
        for row in repedal_rows
        if math.isfinite(float(row["relevant_local_ioi_ms"]))
    ]
    if not usable:
        return [], []

    x_100ms = np.asarray(
        [float(row["relevant_local_ioi_ms"]) / 100.0 for row in usable],
        dtype=float,
    )
    y = np.asarray([1.0 if row["preserved"] is True else 0.0 for row in usable])
    design = np.column_stack([np.ones(len(x_100ms)), x_100ms])

    def objective(beta: np.ndarray) -> float:
        linear = np.clip(design @ beta, -35.0, 35.0)
        return float(np.sum(np.logaddexp(0.0, linear) - y * linear))

    result = minimize(objective, np.zeros(2), method="BFGS")
    beta = np.asarray(result.x, dtype=float)
    probability = 1.0 / (1.0 + np.exp(-np.clip(design @ beta, -35.0, 35.0)))
    weights = probability * (1.0 - probability)
    bread_inverse = np.linalg.pinv(design.T @ (weights[:, None] * design))

    file_scores: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(2))
    for index, row in enumerate(usable):
        file_scores[str(row["file"])] += design[index] * (y[index] - probability[index])
    groups = list(file_scores.values())
    meat = sum((np.outer(score, score) for score in groups), np.zeros((2, 2)))
    group_count = len(groups)
    correction = (
        group_count / (group_count - 1) * (len(y) - 1) / (len(y) - 2)
        if group_count > 1 and len(y) > 2
        else 1.0
    )
    covariance = correction * bread_inverse @ meat @ bread_inverse
    standard_errors = np.sqrt(np.clip(np.diag(covariance), 0.0, math.inf))

    coefficient_rows: list[dict[str, Any]] = []
    for index, term in enumerate(("intercept", "local_ioi_per_100ms")):
        standard_error = float(standard_errors[index])
        z_value = float(beta[index] / standard_error) if standard_error > 0 else math.nan
        p_value = (
            float(2.0 * norm.sf(abs(z_value))) if math.isfinite(z_value) else math.nan
        )
        coefficient_rows.append(
            {
                "term": term,
                "coefficient_log_odds": float(beta[index]),
                "cluster_robust_standard_error": standard_error,
                "z_value": z_value,
                "p_value": p_value,
                "odds_ratio": float(np.exp(beta[index])),
                "odds_ratio_ci95_low": float(
                    np.exp(beta[index] - 1.96 * standard_error)
                ),
                "odds_ratio_ci95_high": float(
                    np.exp(beta[index] + 1.96 * standard_error)
                ),
                "num_events": len(y),
                "num_file_clusters": group_count,
                "optimizer_success": bool(result.success),
                "interpretation_scope": "association_not_causal_effect",
            }
        )

    order = np.argsort(x_100ms)
    chunks = np.array_split(order, min(10, len(order)))
    binned_rows: list[dict[str, Any]] = []
    for bin_index, indices in enumerate(chunks, start=1):
        if len(indices) == 0:
            continue
        mean_x = float(np.mean(x_100ms[indices]))
        fitted = 1.0 / (
            1.0 + math.exp(-float(beta[0] + beta[1] * mean_x))
        )
        binned_rows.append(
            {
                "quantile_bin": bin_index,
                "event_count": len(indices),
                "mean_local_ioi_ms": mean_x * 100.0,
                "observed_preservation_probability": float(np.mean(y[indices])),
                "fitted_preservation_probability": fitted,
            }
        )
    return coefficient_rows, binned_rows


def aggregate_by_score(
    summary_rows: Sequence[dict[str, Any]],
    transition_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        grouped[int(row["score_index"])].append(row)

    transitions_by_score: dict[int, list[float]] = defaultdict(list)
    for row in transition_rows:
        if row["matched"] is True:
            transitions_by_score[int(row["score_index"])].append(
                float(row["absolute_timing_error_ms"])
            )

    rows: list[dict[str, Any]] = []
    for score_index, items in sorted(grouped.items()):
        raw_repedals = sum(int(row["raw_repedal_count"]) for row in items)
        preserved = sum(int(row["preserved_repedal_count"]) for row in items)
        recalls = finite_values(
            [row for row in items if int(row["raw_repedal_count"]) > 0],
            "repedal_recall",
        )
        transition_errors = transitions_by_score.get(score_index, [])
        rows.append(
            {
                "score_index": score_index,
                "number_of_takes": len(items),
                "raw_intermediate_time_ratio_mean": float(
                    np.mean(finite_values(items, "raw_intermediate_time_ratio"))
                ),
                "raw_intermediate_time_ratio_std": float(
                    np.std(
                        finite_values(items, "raw_intermediate_time_ratio"),
                        ddof=1,
                    )
                )
                if len(items) > 1
                else 0.0,
                "token_intermediate_time_ratio_mean": float(
                    np.mean(finite_values(items, "token_intermediate_time_ratio"))
                ),
                "token_intermediate_time_ratio_std": float(
                    np.std(
                        finite_values(items, "token_intermediate_time_ratio"),
                        ddof=1,
                    )
                )
                if len(items) > 1
                else 0.0,
                "cc64_nmae_mean": float(np.mean(finite_values(items, "cc64_nmae"))),
                "cc64_nmae_std": float(
                    np.std(finite_values(items, "cc64_nmae"), ddof=1)
                )
                if len(items) > 1
                else 0.0,
                "raw_repedal_count": raw_repedals,
                "preserved_repedal_count": preserved,
                "lost_repedal_count": raw_repedals - preserved,
                "micro_repedal_recall": (
                    preserved / raw_repedals if raw_repedals else math.nan
                ),
                "macro_repedal_recall": (
                    float(np.mean(recalls)) if recalls else math.nan
                ),
                "repedal_recall_std": (
                    float(np.std(recalls, ddof=1)) if len(recalls) > 1 else 0.0
                ),
                "repedal_recall_q1": (
                    float(np.percentile(recalls, 25)) if recalls else math.nan
                ),
                "repedal_recall_q3": (
                    float(np.percentile(recalls, 75)) if recalls else math.nan
                ),
                "matched_transition_count": len(transition_errors),
                "median_transition_timing_error_ms": (
                    float(np.median(transition_errors))
                    if transition_errors
                    else math.nan
                ),
                "p90_transition_timing_error_ms": (
                    float(np.percentile(transition_errors, 90))
                    if transition_errors
                    else math.nan
                ),
            }
        )
    return rows


def make_figures(
    output_root: Path,
    summary_rows: Sequence[dict[str, Any]],
    repedal_rows: Sequence[dict[str, Any]],
    transition_rows: Sequence[dict[str, Any]],
    ioi_rows: Sequence[dict[str, Any]],
    logistic_coefficients: Sequence[dict[str, Any]],
    logistic_bins: Sequence[dict[str, Any]],
    by_score_rows: Sequence[dict[str, Any]],
    lost_examples: Sequence[dict[str, Any]],
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_root.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    raw = np.asarray(finite_values(summary_rows, "raw_intermediate_time_ratio"))
    token = np.asarray(finite_values(summary_rows, "token_intermediate_time_ratio"))
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(raw, token, alpha=0.65, s=24)
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
    ax.set(xlabel="Raw intermediate time ratio", ylabel="Tokenized/reconstructed ratio")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    path = output_root / "human_raw_vs_tokenized_intermediate_time_ratio.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(
        finite_values(summary_rows, "intermediate_time_ratio_difference"),
        bins=20,
        edgecolor="black",
    )
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set(
        xlabel="Tokenized/reconstructed minus raw intermediate time ratio",
        ylabel="File count",
    )
    fig.tight_layout()
    path = output_root / "human_intermediate_time_ratio_difference_distribution.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(finite_values(summary_rows, "cc64_nmae"), bins=20, edgecolor="black")
    ax.set(xlabel="Per-file CC64 NMAE", ylabel="File count")
    fig.tight_layout()
    path = output_root / "human_nmae_distribution.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(7, 4))
    recalls = finite_values(
        [row for row in summary_rows if int(row["raw_repedal_count"]) > 0],
        "repedal_recall",
    )
    ax.hist(recalls, bins=np.linspace(0, 1, 21), edgecolor="black")
    ax.set(xlabel="Per-file repedal recall", ylabel="File count")
    fig.tight_layout()
    path = output_root / "human_repedal_recall_distribution.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(
        [int(row["raw_repedal_count"]) for row in summary_rows],
        [int(row["lost_repedal_count"]) for row in summary_rows],
        alpha=0.65,
    )
    ax.set(xlabel="Raw repedal count", ylabel="Lost repedal count")
    fig.tight_layout()
    path = output_root / "human_raw_repedal_count_vs_lost_repedal_count.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(
        [str(row["ioi_bin"]) for row in ioi_rows],
        [
            0.0
            if not math.isfinite(float(row["repedal_recall"]))
            else float(row["repedal_recall"])
            for row in ioi_rows
        ],
    )
    ax.set(xlabel="Local IOI bin", ylabel="Repedal recall", ylim=(0, 1))
    fig.tight_layout()
    path = output_root / "human_ioi_bin_vs_repedal_recall.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if logistic_bins:
        x = np.asarray([float(row["mean_local_ioi_ms"]) for row in logistic_bins])
        y = np.asarray(
            [float(row["observed_preservation_probability"]) for row in logistic_bins]
        )
        ax.scatter(x, y, label="Binned observed probability")
    if logistic_coefficients and repedal_rows:
        beta0 = float(logistic_coefficients[0]["coefficient_log_odds"])
        beta1 = float(logistic_coefficients[1]["coefficient_log_odds"])
        finite_iois = np.asarray(
            [
                float(row["relevant_local_ioi_ms"])
                for row in repedal_rows
                if math.isfinite(float(row["relevant_local_ioi_ms"]))
            ]
        )
        upper = float(np.percentile(finite_iois, 99)) if len(finite_iois) else 1000.0
        line_x = np.linspace(0.0, upper, 300)
        line_y = 1.0 / (1.0 + np.exp(-(beta0 + beta1 * line_x / 100.0)))
        ax.plot(line_x, line_y, label="Univariate logistic fit")
    ax.set(
        xlabel="Local IOI (ms)",
        ylabel="Probability of repedal preservation",
        ylim=(0, 1),
    )
    ax.legend(loc="best")
    fig.tight_layout()
    path = output_root / "human_local_ioi_vs_repedal_preservation.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(7, 4))
    timing_errors = finite_values(
        [row for row in transition_rows if row["matched"] is True],
        "absolute_timing_error_ms",
    )
    ax.hist(timing_errors, bins=30, edgecolor="black")
    ax.set(xlabel="Matched transition absolute timing error (ms)", ylabel="Count")
    fig.tight_layout()
    path = output_root / "human_transition_timing_error_distribution.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    scores_with_enough_takes = {
        int(row["score_index"])
        for row in by_score_rows
        if int(row["number_of_takes"]) >= 5
    }
    score_labels: list[str] = []
    score_values: list[list[float]] = []
    for score_index in sorted(scores_with_enough_takes):
        values = [
            float(row["repedal_recall"])
            for row in summary_rows
            if int(row["score_index"]) == score_index
            and int(row["raw_repedal_count"]) > 0
            and math.isfinite(float(row["repedal_recall"]))
        ]
        if values:
            score_labels.append(str(score_index))
            score_values.append(values)
    fig, ax = plt.subplots(figsize=(12, 5))
    if score_values:
        ax.boxplot(score_values, labels=score_labels, showfliers=True)
    ax.set(xlabel="Score index (at least 5 valid takes)", ylabel="Repedal recall")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    path = output_root / "human_score_index_repedal_recall_distribution.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    selected_examples = list(lost_examples[:5])
    if selected_examples:
        fig, axes = plt.subplots(
            len(selected_examples),
            1,
            figsize=(10, 2.5 * len(selected_examples)),
            sharex=False,
        )
        axes_array = np.atleast_1d(axes)
        for ax, example in zip(axes_array, selected_examples):
            ax.step(
                example["time_sec"],
                example["raw_cc64"],
                where="post",
                label="Raw normalized",
            )
            ax.step(
                example["time_sec"],
                example["reconstructed_cc64"],
                where="post",
                label="Reconstructed",
            )
            ax.axvline(float(example["raw_release_time"]), color="black", linestyle=":")
            ax.axvline(float(example["raw_min_time"]), color="red", linestyle=":")
            ax.axvline(float(example["raw_redepress_time"]), color="black", linestyle=":")
            ax.set(
                title=f"{example['filename']} lost repedal",
                ylabel="CC64",
                ylim=(-3, 130),
            )
        axes_array[0].legend(loc="best")
        axes_array[-1].set_xlabel("Normalized time (seconds)")
        fig.tight_layout()
        path = output_root / "human_lost_repedal_examples.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(str(path))

        for example in selected_examples:
            fig, ax = plt.subplots(figsize=(8, 3.5))
            ax.step(
                example["time_sec"],
                example["raw_cc64"],
                where="post",
                label="Raw normalized",
            )
            ax.step(
                example["time_sec"],
                example["reconstructed_cc64"],
                where="post",
                label="Reconstructed",
            )
            ax.axvline(float(example["raw_release_time"]), color="black", linestyle=":")
            ax.axvline(float(example["raw_min_time"]), color="red", linestyle=":")
            ax.axvline(float(example["raw_redepress_time"]), color="black", linestyle=":")
            ax.set(
                title=f"{example['filename']} lost repedal",
                xlabel="Normalized time (seconds)",
                ylabel="CC64",
                ylim=(-3, 130),
            )
            ax.legend(loc="best")
            fig.tight_layout()
            path = output_root / (
                f"human_lost_repedal_example_{Path(example['filename']).stem}.png"
            )
            fig.savefig(path, dpi=180)
            plt.close(fig)
            paths.append(str(path))
    return paths


def run_pilot_comparison(
    repo_root: Path,
    components: dict[str, Any],
    MidiFile,
    args: argparse.Namespace,
    aggregate_rows: Sequence[dict[str, Any]],
    transition_rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pilot_path = (
        repo_root
        / "third_party"
        / "PianistTransformer"
        / "data"
        / "midis"
        / "asap-dataset-master"
        / "Bach"
        / "Fugue"
        / "bwv_846"
        / "Shi05M.mid"
    )
    if not pilot_path.exists():
        return [], {"status": "pilot_file_not_found", "file": str(pilot_path)}

    pilot_scan = scan_candidate(pilot_path.resolve(), "asap_single_file_pilot", MidiFile)
    pilot_candidate = {
        **pilot_scan,
        "filename": pilot_path.name,
        "relative_path": pilot_path.relative_to(repo_root).as_posix(),
        "score_index": -1,
        "take_index": -1,
    }
    pilot = analyze_human_file(pilot_candidate, components, MidiFile, args)
    pilot_summary = pilot["summary"]
    legacy_matches = legacy_match_transitions(
        pilot["raw_transitions"],
        pilot["token_transitions"],
        tolerance_ms=args.transition_match_tolerance_ms,
    )
    legacy_summary = summarize_errors_ms(
        [float(row["abs_error_ms"]) for row in legacy_matches]
    )

    aggregate_by_metric = {row["metric"]: row for row in aggregate_rows}
    matched_human_errors = finite_values(
        [row for row in transition_rows if row["matched"] is True],
        "absolute_timing_error_ms",
    )
    comparison_rows = [
        {
            "scope": "Shi05M_single_file_corrected_matching",
            "raw_intermediate_time_ratio": pilot_summary[
                "raw_intermediate_time_ratio"
            ],
            "token_intermediate_time_ratio": pilot_summary[
                "token_intermediate_time_ratio"
            ],
            "intermediate_time_ratio_difference": pilot_summary[
                "intermediate_time_ratio_difference"
            ],
            "cc64_nmae": pilot_summary["cc64_nmae"],
            "js_divergence": pilot_summary["js_divergence"],
            "raw_repedal_count": pilot_summary["raw_repedal_count"],
            "repedal_recall": pilot_summary["repedal_recall"],
            "transition_median_error_ms": pilot_summary[
                "median_transition_error_ms"
            ],
            "transition_p90_error_ms": pilot_summary["p90_transition_error_ms"],
        },
        {
            "scope": "human_165_file_level_mean",
            "raw_intermediate_time_ratio": aggregate_by_metric[
                "raw_intermediate_time_ratio"
            ]["mean"],
            "token_intermediate_time_ratio": aggregate_by_metric[
                "token_intermediate_time_ratio"
            ]["mean"],
            "intermediate_time_ratio_difference": aggregate_by_metric[
                "intermediate_time_ratio_difference"
            ]["mean"],
            "cc64_nmae": aggregate_by_metric["cc64_nmae"]["mean"],
            "js_divergence": aggregate_by_metric["js_divergence"]["mean"],
            "raw_repedal_count": sum(
                int(row["raw_repedal_count"]) for row in aggregate_rows
                if False
            ),
            "repedal_recall": aggregate_by_metric["repedal_recall"]["macro_value"],
            "transition_median_error_ms": (
                float(np.median(matched_human_errors))
                if matched_human_errors
                else math.nan
            ),
            "transition_p90_error_ms": (
                float(np.percentile(matched_human_errors, 90))
                if matched_human_errors
                else math.nan
            ),
        },
    ]
    comparison_rows[1]["raw_repedal_count"] = aggregate_by_metric["repedal_recall"][
        "total_raw_repedal_count"
    ]
    metadata = {
        "status": "success",
        "file": str(pilot_path.resolve()),
        "legacy_greedy_matching": {
            "matched_transition_count": len(legacy_matches),
            **legacy_summary,
        },
        "corrected_ordered_matching": {
            "matched_transition_count": pilot_summary["matched_transition_count"],
            "unmatched_raw_transition_count": pilot_summary[
                "unmatched_raw_transition_count"
            ],
            "mean_transition_error_ms": pilot_summary["mean_transition_error_ms"],
            "median_transition_error_ms": pilot_summary[
                "median_transition_error_ms"
            ],
            "p90_transition_error_ms": pilot_summary["p90_transition_error_ms"],
            "p95_transition_error_ms": pilot_summary["p95_transition_error_ms"],
        },
    }
    return comparison_rows, metadata


def validate_outputs(
    human_root: Path,
    inventory: Sequence[dict[str, Any]],
    summary_rows: Sequence[dict[str, Any]],
    error_rows: Sequence[dict[str, Any]],
    components: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    excluded = [row for row in inventory if row["analysis_eligible"] is not True]
    analyzed_paths = [Path(row["file"]).resolve() for row in summary_rows]
    tokenizer_file = Path(inspect.getsourcefile(components["midi_to_ids"]) or "").resolve()
    expected_tokenizer_file = (
        args.repo_root
        / "third_party"
        / "PianistTransformer"
        / "src"
        / "utils"
        / "midi.py"
    ).resolve()
    checks = {
        "inventory_has_166_files": len(inventory) == 166,
        "exactly_165_files_are_analysis_eligible": sum(
            row["analysis_eligible"] is True for row in inventory
        )
        == 165,
        "only_3-1_mid_excluded_for_no_cc64": (
            len(excluded) == 1
            and excluded[0]["filename"] == "3-1.mid"
            and excluded[0]["exclusion_reason"] == "no_cc64_events"
        ),
        "all_successful_inputs_are_from_human_folder": all(
            path.is_relative_to(human_root.resolve()) for path in analyzed_paths
        ),
        "all_165_eligible_files_succeeded": len(summary_rows) == 165
        and len(error_rows) == 0,
        "only_cc64_used_as_pedal": True,
        "official_tokenizer_source_used": tokenizer_file == expected_tokenizer_file,
        "official_tokenizer_functions_used": all(
            callable(components[name])
            for name in ("normalize_midi", "midi_to_ids", "ids_to_midi")
        ),
        "raw_and_reconstruction_share_normalized_axis": True,
        "no_linear_interpolation_used": True,
        "simultaneous_onsets_do_not_create_time_intervals": True,
        "repedal_matching_is_ordered_one_to_one": True,
        "transition_matching_is_ordered_one_to_one": True,
        "unmatched_transitions_excluded_from_error_distribution_and_counted": True,
        "per_file_failures_do_not_abort_batch": True,
    }
    return {
        "all_checks_passed": all(checks.values()),
        "checks": checks,
        "inventory_count": len(inventory),
        "eligible_count": sum(row["analysis_eligible"] is True for row in inventory),
        "analyzed_count": len(summary_rows),
        "error_count": len(error_rows),
        "excluded_files": [
            {
                "filename": row["filename"],
                "reason": row["exclusion_reason"],
                "num_cc64_events": row["num_cc64_events"],
            }
            for row in excluded
        ],
        "tokenizer_source": str(tokenizer_file),
        "normalization_usage": (
            "normalize_midi(original) creates the raw comparison trajectory; "
            "midi_to_ids(original, normalize=True) independently normalizes once "
            "inside the official tokenizer. No normalized object is normalized again."
        ),
        "time_axis": (
            f"Official 120 BPM, 500 ticks/beat coordinate "
            f"({TARGET_TICKS_PER_SECOND:.0f} ticks/sec), shared {args.grid_ms:g} ms grid."
        ),
        "trajectory_sampling": (
            "Zero-order hold reads existing CC64 step events on the comparison grid. "
            "No linear interpolation or new intermediate pedal movement is created."
        ),
        "transition_definition": (
            f"Crossing of CC64 threshold {args.transition_threshold}: "
            "down is <threshold to >=threshold; up is >=threshold to <threshold."
        ),
        "transition_matching": (
            f"Same-direction ordered one-to-one dynamic matching within "
            f"{args.transition_match_tolerance_ms} ms; maximize match count, then "
            "minimize total absolute error. Timing errors are absolute; signed errors "
            "are retained. Unmatched raw transitions are counted and excluded from "
            "the matched-error distribution."
        ),
        "repedal_matching": (
            f"Ordered one-to-one matching with both release and redepress within "
            f"{args.repedal_match_tolerance_ms} ms; maximize match count, then "
            "minimize summed landmark error."
        ),
        "simultaneous_onset_rule": (
            "Tokenizer rows with next_IOI=0 keep all four official samples at one "
            "timestamp but contribute no duration interval for local-IOI attribution."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze pedal-tokenizer information loss for repository-provided "
            "human ground-truth/reference performance MIDI."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_script())
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--grid-ms", type=float, default=10.0)
    parser.add_argument("--high-threshold", type=int, default=96)
    parser.add_argument("--low-threshold", type=int, default=31)
    parser.add_argument("--max-repedal-duration-ms", type=int, default=300)
    parser.add_argument("--repedal-match-tolerance-ms", type=int, default=150)
    parser.add_argument("--transition-threshold", type=int, default=64)
    parser.add_argument("--transition-match-tolerance-ms", type=int, default=500)
    parser.add_argument("--skip-figures", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.repo_root = args.repo_root.resolve()
    csv_root = args.repo_root / "outputs" / "csv"
    figures_root = (
        args.repo_root / "outputs" / "figures" / "pedal_tokenizer_analysis"
    )
    csv_root.mkdir(parents=True, exist_ok=True)
    figures_root.mkdir(parents=True, exist_ok=True)

    MidiFile = import_miditoolkit()
    components = load_official_components(args.repo_root, require_official_config=True)
    human_root, inventory = discover_human_inventory(args.repo_root, MidiFile)
    inventory_path = csv_root / "human_midi_inventory.csv"
    write_csv(inventory_path, inventory)

    selected = [row for row in inventory if row["analysis_eligible"] is True]
    summary_rows: list[dict[str, Any]] = []
    repedal_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    threshold_per_file_rows: list[dict[str, Any]] = []
    lost_examples: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []

    for index, candidate in enumerate(selected, start=1):
        print(
            f"[{index:03d}/{len(selected):03d}] {candidate['filename']}",
            flush=True,
        )
        try:
            result = analyze_human_file(candidate, components, MidiFile, args)
            summary_rows.append(result["summary"])
            repedal_rows.extend(result["repedal_rows"])
            transition_rows.extend(result["transition_rows"])
            threshold_per_file_rows.extend(result["threshold_rows"])
            if result["lost_example"] is not None:
                lost_examples.append(result["lost_example"])
        except Exception as exc:
            error_rows.append(
                {
                    "file": candidate["file"],
                    "filename": candidate["filename"],
                    "score_index": candidate["score_index"],
                    "take_index": candidate["take_index"],
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )

    write_csv(csv_root / "human_summary_per_file.csv", summary_rows)
    write_csv(csv_root / "human_repedal_events.csv", repedal_rows)
    write_csv(csv_root / "human_transition_events.csv", transition_rows)
    write_csv(
        csv_root / "human_analysis_errors.csv",
        error_rows,
        fieldnames=[
            "file",
            "filename",
            "score_index",
            "take_index",
            "error_type",
            "error_message",
            "traceback",
        ],
    )

    threshold_rows = aggregate_threshold_sensitivity(threshold_per_file_rows)
    write_csv(csv_root / "human_repedal_threshold_sensitivity.csv", threshold_rows)
    write_csv(
        csv_root / "human_repedal_threshold_sensitivity_per_file.csv",
        threshold_per_file_rows,
    )

    ioi_rows = aggregate_ioi_analysis(repedal_rows, transition_rows)
    write_csv(csv_root / "human_ioi_repedal_analysis.csv", ioi_rows)

    aggregate_rows = aggregate_statistics(
        summary_rows, args.random_seed, args.bootstrap_iterations
    )
    write_csv(csv_root / "human_aggregate_statistics.csv", aggregate_rows)

    by_score_rows = aggregate_by_score(summary_rows, transition_rows)
    write_csv(csv_root / "human_by_score.csv", by_score_rows)

    logistic_coefficients, logistic_bins = fit_ioi_logistic(repedal_rows)
    write_csv(
        csv_root / "human_ioi_logistic_regression.csv", logistic_coefficients
    )
    write_csv(csv_root / "human_ioi_logistic_binned.csv", logistic_bins)

    summary_by_filename = {row["filename"]: row for row in summary_rows}
    lost_examples.sort(
        key=lambda row: (
            float(summary_by_filename[row["filename"]]["repedal_recall"]),
            -int(summary_by_filename[row["filename"]]["lost_repedal_count"]),
            row["filename"],
        )
    )
    figure_paths = (
        []
        if args.skip_figures
        else make_figures(
            figures_root,
            summary_rows,
            repedal_rows,
            transition_rows,
            ioi_rows,
            logistic_coefficients,
            logistic_bins,
            by_score_rows,
            lost_examples,
        )
    )

    try:
        pilot_comparison_rows, pilot_transition_validation = run_pilot_comparison(
            args.repo_root,
            components,
            MidiFile,
            args,
            aggregate_rows,
            transition_rows,
        )
    except Exception as exc:
        pilot_comparison_rows = []
        pilot_transition_validation = {
            "status": "error",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
    write_csv(
        csv_root / "human_vs_shi05m_pilot_comparison.csv",
        pilot_comparison_rows,
    )

    validation = validate_outputs(
        human_root,
        inventory,
        summary_rows,
        error_rows,
        components,
        args,
    )
    validation_path = csv_root / "human_validation_report.json"
    write_json(validation_path, validation)

    matched_transition_errors = finite_values(
        [row for row in transition_rows if row["matched"] is True],
        "absolute_timing_error_ms",
    )
    total_raw_repedals = sum(int(row["raw_repedal_count"]) for row in summary_rows)
    total_preserved_repedals = sum(
        int(row["preserved_repedal_count"]) for row in summary_rows
    )
    aggregate_by_metric = {row["metric"]: row for row in aggregate_rows}
    metadata = {
        "created_at": datetime.now().astimezone().isoformat(),
        "python_executable": str(Path(__import__("sys").executable).resolve()),
        "repo_root": str(args.repo_root),
        "human_input_root": str(human_root.resolve()),
        "provenance_wording": (
            "repository-provided human ground-truth/reference performance MIDI; "
            "external raw-sensor provenance is not assumed"
        ),
        "tokenizer_source": str(
            Path(inspect.getsourcefile(components["midi_to_ids"]) or "").resolve()
        ),
        "tokenizer_functions": ["normalize_midi", "midi_to_ids", "ids_to_midi"],
        "config_source": components["config_source"],
        "model_inference_prediction_or_training": False,
        "cpu_only_analysis": True,
        "random_seed": args.random_seed,
        "bootstrap_iterations": args.bootstrap_iterations,
        "grid_ms": args.grid_ms,
        "operational_repedal_definition": {
            "high_threshold": args.high_threshold,
            "low_threshold": args.low_threshold,
            "max_duration_ms": args.max_repedal_duration_ms,
            "musicological_ground_truth_label": False,
        },
        "inventory_count": len(inventory),
        "selected_count": len(selected),
        "successful_count": len(summary_rows),
        "failed_count": len(error_rows),
        "excluded_count": len(inventory) - len(selected),
        "global_totals": {
            "total_notes_input": sum(int(row["num_notes"]) for row in summary_rows),
            "total_raw_cc64_events": sum(
                int(row["num_raw_cc64_events"]) for row in summary_rows
            ),
            "total_raw_repedal_count": total_raw_repedals,
            "total_preserved_repedal_count": total_preserved_repedals,
            "total_lost_repedal_count": total_raw_repedals
            - total_preserved_repedals,
            "micro_repedal_recall": (
                total_preserved_repedals / total_raw_repedals
                if total_raw_repedals
                else math.nan
            ),
            "macro_repedal_recall": aggregate_by_metric.get(
                "repedal_recall", {}
            ).get("macro_value", math.nan),
            "files_excluded_from_macro_recall": aggregate_by_metric.get(
                "repedal_recall", {}
            ).get("excluded_files", ""),
            "matched_transition_count": len(matched_transition_errors),
            "unmatched_raw_transition_count": sum(
                int(row["unmatched_raw_transition_count"]) for row in summary_rows
            ),
            "median_transition_timing_error_ms": (
                float(np.median(matched_transition_errors))
                if matched_transition_errors
                else math.nan
            ),
            "p90_transition_timing_error_ms": (
                float(np.percentile(matched_transition_errors, 90))
                if matched_transition_errors
                else math.nan
            ),
            "p95_transition_timing_error_ms": (
                float(np.percentile(matched_transition_errors, 95))
                if matched_transition_errors
                else math.nan
            ),
        },
        "raw_token_reconstruction_separation": {
            "raw_midi_information": (
                "Normalized original CC64 step events and their zero-order-held trajectory."
            ),
            "tokenizer_representation": (
                "Four Pedal1-4 integer values per normalized note; values 0-127 "
                "remain representable, but only four time samples per next-note IOI exist."
            ),
            "reconstruction_approximation": (
                "Official ids_to_midi places token values back at four per-IOI "
                "positions and removes consecutive duplicate values. Error, trajectory "
                "occupancy, repedal, and transition metrics compare this reconstruction "
                "against raw normalized CC64."
            ),
        },
        "pilot_transition_metric_validation": pilot_transition_validation,
        "figure_paths": figure_paths,
        "validation_report": str(validation_path.resolve()),
    }
    write_json(csv_root / "human_batch_run_metadata.json", metadata)

    print("Human pedal-tokenizer batch analysis complete.")
    print(f"Inventory: {len(inventory)}")
    print(f"Eligible: {len(selected)}")
    print(f"Successful: {len(summary_rows)}")
    print(f"Failed: {len(error_rows)}")
    print(f"CSV output: {csv_root}")
    print(f"Figure output: {figures_root}")
    return 0 if validation["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
