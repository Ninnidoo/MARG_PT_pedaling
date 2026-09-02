"""Pure diagnostics for the frozen Custom Event v0 four-state failure audit.

This module deliberately contains no training, checkpoint selection, or metric
tuning.  All arrays use the canonical state order ZERO, LOW, HALF, FULL.
"""

from __future__ import annotations

import copy
import hashlib
from typing import Any, Mapping, Sequence

import numpy as np


STATE_NAMES = ("ZERO", "LOW", "HALF", "FULL")
EVENT_NAMES = ("NONE", "SET_ZERO", "SET_LOW", "SET_HALF", "SET_FULL")
STATE_REPRESENTATIVES = (0, 51, 79, 127)


def canonical_states(raw_cc64: Any) -> np.ndarray:
    values = np.asarray(raw_cc64)
    if np.any((values < 0) | (values > 127)):
        raise ValueError("CC64 values must lie in [0,127]")
    return np.select(
        [values <= 25, values <= 63, values <= 103], [0, 1, 2], default=3
    ).astype(np.int8)


def binary_states(states: Any) -> np.ndarray:
    values = np.asarray(states, dtype=np.int64)
    if np.any((values < 0) | (values > 3)):
        raise ValueError("four-state values must lie in [0,3]")
    return (values >= 2).astype(np.int8)


def confusion(candidate: Any, target: Any, classes: int) -> np.ndarray:
    predicted = np.asarray(candidate, dtype=np.int64)
    reference = np.asarray(target, dtype=np.int64)
    if predicted.shape != reference.shape or predicted.size == 0:
        raise ValueError("candidate and target must have the same non-empty shape")
    if np.any((predicted < 0) | (predicted >= classes)) or np.any(
        (reference < 0) | (reference >= classes)
    ):
        raise ValueError("class value out of range")
    return np.bincount(
        reference.reshape(-1) * classes + predicted.reshape(-1),
        minlength=classes * classes,
    ).reshape(classes, classes)


def confusion_metrics(matrix: Any) -> dict[str, Any]:
    value = np.asarray(matrix, dtype=np.int64)
    if value.ndim != 2 or value.shape[0] != value.shape[1] or value.sum() <= 0:
        raise ValueError("confusion matrix must be non-empty and square")
    rows = []
    for index in range(len(value)):
        tp = int(value[index, index])
        predicted = int(value[:, index].sum())
        support = int(value[index].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {"class_id": index, "precision": precision, "recall": recall,
             "f1": f1, "support": support, "predicted": predicted}
        )
    return {
        "accuracy": float(np.trace(value) / value.sum()),
        "macro_f1": float(np.mean([row["f1"] for row in rows])),
        "class_metrics": rows,
        "confusion_matrix": value.tolist(),
        "row_normalized": np.divide(
            value, value.sum(1, keepdims=True),
            out=np.zeros_like(value, dtype=float), where=value.sum(1, keepdims=True) != 0,
        ).tolist(),
    }


def error_decomposition(candidate: Any, target: Any) -> dict[str, Any]:
    predicted = np.asarray(candidate, dtype=np.int64).reshape(-1)
    reference = np.asarray(target, dtype=np.int64).reshape(-1)
    if predicted.shape != reference.shape:
        raise ValueError("candidate/target shape mismatch")
    wrong = predicted != reference
    same = wrong & (binary_states(predicted) == binary_states(reference))
    cross = wrong & ~same
    total = int(wrong.sum())
    return {
        "total_incorrect": total,
        "same_side_depth": int(same.sum()),
        "cross_threshold": int(cross.sum()),
        "same_side_percent_of_errors": 100.0 * float(same.sum()) / total if total else 0.0,
        "cross_threshold_percent_of_errors": 100.0 * float(cross.sum()) / total if total else 0.0,
    }


def effective_changes(states: Any) -> list[dict[str, Any]]:
    values = np.asarray(states, dtype=np.int64).reshape(-1)
    result = []
    for index in range(1, len(values)):
        source, destination = int(values[index - 1]), int(values[index])
        if source == destination:
            continue
        if {source, destination} == {0, 1}:
            kind = "TYPE_A"
        elif {source, destination} == {2, 3}:
            kind = "TYPE_B"
        else:
            kind = "TYPE_C"
        result.append(
            {"index": index, "source": source, "destination": destination,
             "pair": f"{STATE_NAMES[source]}->{STATE_NAMES[destination]}", "type": kind}
        )
    return result


def pattern_ids(states: Any, base: int) -> np.ndarray:
    values = np.asarray(states, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("patterns require shape [N,4]")
    if np.any((values < 0) | (values >= base)):
        raise ValueError("pattern class outside base")
    return values @ np.asarray([base ** 3, base ** 2, base, 1], dtype=np.int64)


def collapse_binary_patterns(states: Any) -> np.ndarray:
    return pattern_ids(binary_states(states), 2)


def mismatch_episodes(
    candidate: Any, target: Any, sample_times: Any, onset_positions: Any
) -> list[dict[str, Any]]:
    predicted = np.asarray(candidate, dtype=np.int64).reshape(-1)
    reference = np.asarray(target, dtype=np.int64).reshape(-1)
    times = np.asarray(sample_times, dtype=float).reshape(-1)
    onsets = np.asarray(onset_positions, dtype=np.int64).reshape(-1)
    if not (len(predicted) == len(reference) == len(times) == len(onsets)):
        raise ValueError("mismatch episode inputs must share length")
    result, index = [], 0
    while index < len(predicted):
        if predicted[index] == reference[index]:
            index += 1
            continue
        end = index + 1
        while end < len(predicted) and predicted[end] != reference[end]:
            end += 1
        candidate_changed = index > 0 and predicted[index] != predicted[index - 1]
        reference_changed = index > 0 and reference[index] != reference[index - 1]
        if index == 0:
            cause = "D_BOUNDARY"
        elif candidate_changed and not reference_changed:
            cause = "A_CANDIDATE_CHANGED_REFERENCE_DID_NOT"
        elif reference_changed and not candidate_changed:
            cause = "B_REFERENCE_CHANGED_CANDIDATE_DID_NOT"
        elif candidate_changed and reference_changed:
            cause = "C_BOTH_CHANGED_DESTINATION_DIFFERED"
        else:
            cause = "D_AMBIGUOUS_SAMPLING_BOUNDARY"
        result.append({
            "start_index": index, "end_index_exclusive": end,
            "sample_count": end - index,
            "distinct_onset_span": int(onsets[end - 1] - onsets[index]),
            "elapsed_seconds": float(times[end - 1] - times[index]),
            "start_target_state": int(reference[index]),
            "start_candidate_state": int(predicted[index]),
            "start_mismatch_kind": (
                "same_side" if binary_states([reference[index]])[0]
                == binary_states([predicted[index]])[0] else "cross_threshold"
            ),
            "start_cause": cause,
        })
        index = end
    return result


def residence_episodes(
    states: Any, sample_times: Any, onset_positions: Any
) -> list[dict[str, Any]]:
    values = np.asarray(states, dtype=np.int64).reshape(-1)
    times = np.asarray(sample_times, dtype=float).reshape(-1)
    onsets = np.asarray(onset_positions, dtype=np.int64).reshape(-1)
    if not (len(values) == len(times) == len(onsets)):
        raise ValueError("residence episode inputs must share length")
    result, index = [], 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[end] == values[index]:
            end += 1
        result.append({
            "state": int(values[index]), "start_index": index,
            "end_index_exclusive": end, "sample_count": end - index,
            "duration_seconds": float(times[end - 1] - times[index]),
            "distinct_onset_span": int(onsets[end - 1] - onsets[index]),
        })
        index = end
    return result


def slot_funnel_counts(classes: Any, timing: Any) -> list[dict[str, int]]:
    """Count raw/prefix/sorted/continuous-same-time stages for one performance."""
    event = np.asarray(classes, dtype=np.int64)
    tau = np.asarray(timing, dtype=float)
    if event.ndim != 2 or event.shape[1] != 6 or tau.shape != event.shape:
        raise ValueError("slot funnel requires [onset,6] class/timing arrays")
    rows = [{"raw_non_none": 0, "prefix_active": 0, "timing_sort_surviving": 0,
             "same_time_surviving": 0} for _ in range(6)]
    for onset in range(len(event)):
        for slot in range(6):
            rows[slot]["raw_non_none"] += int(event[onset, slot] != 0)
        active = next((slot for slot in range(6) if event[onset, slot] == 0), 6)
        pairs = sorted([(float(np.clip(tau[onset, slot], 0, 1)), slot)
                        for slot in range(active)])
        for _, slot in pairs:
            rows[slot]["prefix_active"] += 1
            rows[slot]["timing_sort_surviving"] += 1
        cursor = 0
        while cursor < len(pairs):
            end = cursor + 1
            while end < len(pairs) and pairs[end][0] == pairs[cursor][0]:
                end += 1
            rows[pairs[end - 1][1]]["same_time_surviving"] += 1
            cursor = end
    return rows


def oracle_initial_replacement(
    records: Sequence[Mapping[str, Any]], target_initial: Sequence[int]
) -> list[dict[str, Any]]:
    if len(records) != len(target_initial):
        raise ValueError("one oracle initial is required per record")
    result = copy.deepcopy(list(records))
    for record, state in zip(result, target_initial, strict=True):
        logits = np.full(4, -1.0e9, dtype=np.float32)
        logits[int(state)] = 0.0
        record["initial_logits"] = logits
    return result


def assert_aligned_universe(*arrays: Any, notes: int) -> None:
    expected = (notes, 4)
    if any(np.asarray(array).shape != expected for array in arrays):
        raise AssertionError(f"aligned universe must be {expected}")


def deterministic_digest(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.dtype.str.encode() + str(array.shape).encode() + array.tobytes()).hexdigest()

