"""Validation-only, overlap-aware evaluation for the fixed five-class model.

The evaluator deliberately constructs only the ``validation`` dataset.  It
re-runs the four historical decoders because their stored CSV files contain
aggregate metrics rather than note-level predictions.  Every historical point
prediction is first produced with its original decoder, then rebinned with the
fixed five-class boundaries and decoded with the canonical representatives.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .coarse_to_fine import Stage2CoarseToFineModel, decode_pedals
from .dataset import (
    NON_PEDAL_FEATURES,
    PEDAL_SLOTS,
    PEDAL_TOKEN_OFFSET,
    Stage2PedalDataset,
    generate_window_starts,
    stage2_pedal_collate_fn,
)
from .diagnose_posterior import infer_complete_posterior, posterior_median_decode
from .evaluate_oracle import _load_model, _make_window_sample, _visible_gpu_identity
from .five_class import (
    CLASS_NAMES,
    REPRESENTATIVES,
    FiveClassPedalEncoderModel,
    average_five_class_logits,
    classify_pedal_values,
    decode_five_classes,
    five_class_collate_fn,
)


WINDOW_NOTES = 512
STRIDE_NOTES = 256
DEFAULT_BATCH_SIZE = 16
EXPECTED_PERFORMANCES = 71
EXPECTED_PIECES = 19
EXPECTED_NOTES = 283_928
EXPECTED_TARGETS = EXPECTED_NOTES * PEDAL_SLOTS
EXPECTED_WINDOWS = 1_078
NUM_CLASSES = 5
TOLERANCES = (5, 10, 20)
TIMING_TOLERANCES = (0, 1, 2, 4)
SHORT_REPEDAL_MAX_SAMPLES = 4

NEW_MODEL = "endpoint_aware_5class_argmax"
PRIMARY_REFERENCE = "ce_128_posterior_median"
MODEL_ORDER = (
    "ce_128_argmax",
    PRIMARY_REFERENCE,
    "ordinal_lambda1_posterior_median",
    "coarse_to_fine_v0",
    NEW_MODEL,
)
MODEL_LABELS = {
    "ce_128_argmax": "CE 128-class argmax",
    PRIMARY_REFERENCE: "CE 128-class posterior median",
    "ordinal_lambda1_posterior_median": "Ordinal lambda 1.0 posterior median",
    "coarse_to_fine_v0": "Coarse-to-fine v0",
    NEW_MODEL: "Endpoint-aware 5-class argmax",
}


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    """Return a finite ratio; count fields retain undefined-case context."""

    return float(numerator / denominator) if denominator else 0.0


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise FloatingPointError("refusing to serialize NaN/Inf evaluation output")
        return value
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
    )


def _csv_text(rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> str:
    if fields is None:
        fields = sorted({str(key) for row in rows for key in row})
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fields), extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _atomic_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str] | None = None,
) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path.name}")
    _atomic_text(path, _csv_text(rows, fields))


def _validate_class_arrays(
    predictions: np.ndarray, targets: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    predicted = np.asarray(predictions, dtype=np.int64)
    truth = np.asarray(targets, dtype=np.int64)
    if predicted.shape != truth.shape or truth.ndim != 2 or truth.shape[1] != PEDAL_SLOTS:
        raise ValueError("class predictions and targets must have shape [notes,4]")
    if np.any((predicted < 0) | (predicted >= NUM_CLASSES)):
        raise ValueError("prediction outside five-class range")
    if np.any((truth < 0) | (truth >= NUM_CLASSES)):
        raise ValueError("target outside five-class range")
    return predicted, truth


def _confusion_matrix(
    predictions: np.ndarray, targets: np.ndarray, classes: int = NUM_CLASSES
) -> np.ndarray:
    predicted = np.asarray(predictions, dtype=np.int64).reshape(-1)
    truth = np.asarray(targets, dtype=np.int64).reshape(-1)
    if predicted.shape != truth.shape:
        raise ValueError("prediction/target sizes differ")
    if np.any((predicted < 0) | (predicted >= classes)) or np.any(
        (truth < 0) | (truth >= classes)
    ):
        raise ValueError("confusion values outside class range")
    matrix = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(matrix, (truth, predicted), 1)
    return matrix


def _classification_from_confusion(matrix: np.ndarray) -> dict[str, Any]:
    values = np.asarray(matrix, dtype=np.int64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("confusion matrix must be square")
    total = int(values.sum())
    support = values.sum(axis=1)
    predicted_count = values.sum(axis=0)
    true_positive = np.diag(values)
    precision = np.asarray(
        [_safe_ratio(true_positive[i], predicted_count[i]) for i in range(len(values))]
    )
    recall = np.asarray(
        [_safe_ratio(true_positive[i], support[i]) for i in range(len(values))]
    )
    f1 = np.asarray(
        [
            _safe_ratio(2.0 * precision[i] * recall[i], precision[i] + recall[i])
            for i in range(len(values))
        ]
    )
    correct = int(true_positive.sum())
    accuracy = _safe_ratio(correct, total)
    return {
        "confusion_matrix": values,
        "token_accuracy": accuracy,
        "micro_precision": accuracy,
        "micro_recall": accuracy,
        "micro_f1": accuracy,
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_precision": _safe_ratio(float((precision * support).sum()), total),
        "weighted_recall": _safe_ratio(float((recall * support).sum()), total),
        "weighted_f1": _safe_ratio(float((f1 * support).sum()), total),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "predicted_count": predicted_count,
        "target_count": total,
    }


def classification_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    """Five-class token, exact-note, distribution, and confusion metrics."""

    predicted, truth = _validate_class_arrays(predictions, targets)
    result = _classification_from_confusion(_confusion_matrix(predicted, truth))
    result["exact_note_accuracy"] = float(np.all(predicted == truth, axis=1).mean())
    result["valid_note_count"] = len(truth)
    matrix = result["confusion_matrix"]
    support = result["support"]
    result["target_distribution"] = support / int(support.sum())
    result["predicted_distribution"] = result["predicted_count"] / int(support.sum())
    result["all_five_classes_predicted"] = bool(np.all(result["predicted_count"] > 0))
    result["zero_recall"] = float(result["recall"][0])
    result["low_recall"] = float(result["recall"][1])
    result["mid_recall"] = float(result["recall"][2])
    result["high_recall"] = float(result["recall"][3])
    result["full_recall"] = float(result["recall"][4])
    for true_id, predicted_id, name in (
        (1, 0, "low_to_zero"),
        (1, 2, "low_to_mid"),
        (0, 1, "zero_to_low"),
        (2, 3, "mid_to_high"),
        (3, 2, "high_to_mid"),
    ):
        result[f"{name}_count"] = int(matrix[true_id, predicted_id])
        result[f"{name}_rate"] = _safe_ratio(matrix[true_id, predicted_id], support[true_id])
    result["mid_high_full_aggregate_recall"] = _safe_ratio(
        matrix[2:5, 2:5].sum(), matrix[2:5, :].sum()
    )
    result["zero_low_aggregate_recall"] = _safe_ratio(
        matrix[0:2, 0:2].sum(), matrix[0:2, :].sum()
    )
    result["nonendpoint_endpoint_collapse_ratio"] = _safe_ratio(
        matrix[1:4, 0].sum() + matrix[1:4, 4].sum(), matrix[1:4, :].sum()
    )
    return result


def _binary_classification(predicted: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    matrix = _confusion_matrix(
        np.asarray(predicted, dtype=np.int64), np.asarray(truth, dtype=np.int64), 2
    )
    result = _classification_from_confusion(matrix)
    return {
        "confusion_matrix": matrix,
        "accuracy": result["token_accuracy"],
        "balanced_accuracy": result["macro_recall"],
        "macro_f1": result["macro_f1"],
        "off_precision": float(result["precision"][0]),
        "off_recall": float(result["recall"][0]),
        "off_f1": float(result["f1"][0]),
        "on_precision": float(result["precision"][1]),
        "on_recall": float(result["recall"][1]),
        "on_f1": float(result["f1"][1]),
        "off_support": int(result["support"][0]),
        "on_support": int(result["support"][1]),
        "predicted_off_count": int(result["predicted_count"][0]),
        "predicted_on_count": int(result["predicted_count"][1]),
    }


def _event_metrics(true_event: np.ndarray, predicted_event: np.ndarray) -> dict[str, Any]:
    truth = np.asarray(true_event, dtype=bool)
    predicted = np.asarray(predicted_event, dtype=bool)
    if truth.shape != predicted.shape:
        raise ValueError("event masks differ")
    true_count = int(truth.sum())
    predicted_count = int(predicted.sum())
    true_positive = int((truth & predicted).sum())
    precision = _safe_ratio(true_positive, predicted_count)
    recall = _safe_ratio(true_positive, true_count)
    return {
        "true_count": true_count,
        "predicted_count": predicted_count,
        "true_positive": true_positive,
        "precision": precision,
        "recall": recall,
        "f1": _safe_ratio(2.0 * precision * recall, precision + recall),
    }


def _repedal_intervals(states: np.ndarray, max_off_samples: int) -> set[tuple[int, int]]:
    values = np.asarray(states, dtype=bool).reshape(-1)
    if not len(values):
        return set()
    starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
    ends = np.r_[starts[1:], len(values)]
    run_states = values[starts]
    return {
        (int(starts[index]), int(ends[index]))
        for index in range(1, len(starts) - 1)
        if bool(run_states[index - 1])
        and not bool(run_states[index])
        and bool(run_states[index + 1])
        and int(ends[index] - starts[index]) <= max_off_samples
    }


def _match_offsets(
    true_indices: np.ndarray, predicted_indices: np.ndarray, tolerance: int
) -> list[int]:
    """Greedy earliest-feasible one-to-one event matching on an ordered axis."""

    truth = np.asarray(true_indices, dtype=np.int64)
    predicted = np.asarray(predicted_indices, dtype=np.int64)
    offsets: list[int] = []
    true_index = predicted_index = 0
    while true_index < len(truth) and predicted_index < len(predicted):
        current_true = int(truth[true_index])
        current_predicted = int(predicted[predicted_index])
        if current_predicted < current_true - tolerance:
            predicted_index += 1
        elif current_predicted > current_true + tolerance:
            true_index += 1
        else:
            offsets.append(current_predicted - current_true)
            true_index += 1
            predicted_index += 1
    return offsets


def _timing_diagnostic(
    event_pairs: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
) -> dict[str, Any]:
    output: dict[str, Any] = {"matching": "earliest-feasible one-to-one, within performance and direction"}
    for tolerance in TIMING_TOLERANCES:
        direction_rows: dict[str, Any] = {}
        combined_true = combined_predicted = combined_matched = 0
        all_offsets: list[int] = []
        for direction, true_offset, predicted_offset in (
            ("off_to_on", 0, 1),
            ("on_to_off", 2, 3),
        ):
            true_count = predicted_count = 0
            offsets: list[int] = []
            for pair in event_pairs:
                true_indices = np.flatnonzero(pair[true_offset])
                predicted_indices = np.flatnonzero(pair[predicted_offset])
                true_count += len(true_indices)
                predicted_count += len(predicted_indices)
                offsets.extend(_match_offsets(true_indices, predicted_indices, tolerance))
            precision = _safe_ratio(len(offsets), predicted_count)
            recall = _safe_ratio(len(offsets), true_count)
            direction_rows[direction] = {
                "true_count": true_count,
                "predicted_count": predicted_count,
                "matched_count": len(offsets),
                "precision": precision,
                "recall": recall,
                "f1": _safe_ratio(2.0 * precision * recall, precision + recall),
            }
            combined_true += true_count
            combined_predicted += predicted_count
            combined_matched += len(offsets)
            all_offsets.extend(offsets)
        precision = _safe_ratio(combined_matched, combined_predicted)
        recall = _safe_ratio(combined_matched, combined_true)
        absolute = np.abs(np.asarray(all_offsets, dtype=np.int64))
        signed = np.asarray(all_offsets, dtype=np.int64)
        direction_rows["combined"] = {
            "true_count": combined_true,
            "predicted_count": combined_predicted,
            "matched_count": combined_matched,
            "precision": precision,
            "recall": recall,
            "f1": _safe_ratio(2.0 * precision * recall, precision + recall),
            "mean_signed_offset": float(signed.mean()) if len(signed) else 0.0,
            "mean_absolute_offset": float(absolute.mean()) if len(absolute) else 0.0,
            "median_absolute_offset": float(np.median(absolute)) if len(absolute) else 0.0,
            "maximum_absolute_offset": int(absolute.max()) if len(absolute) else 0,
        }
        output[f"plus_minus_{tolerance}_samples"] = direction_rows
    return output


def renderer_aware_metrics(
    records: Sequence[tuple[np.ndarray, np.ndarray]]
) -> dict[str, Any]:
    """Aggregate state, transition, timing, and repedal metrics per performance."""

    if not records:
        raise ValueError("renderer metrics require performance records")
    true_states: list[np.ndarray] = []
    predicted_states: list[np.ndarray] = []
    event_pairs: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    up_totals = {key: 0 for key in ("true_count", "predicted_count", "true_positive")}
    down_totals = dict(up_totals)
    combined_totals = dict(up_totals)
    steady_count = steady_correct = direction_correct = true_transition_count = 0
    true_repedal = predicted_repedal = repedal_true_positive = 0
    zero_truth: list[np.ndarray] = []
    zero_prediction: list[np.ndarray] = []
    on_true_positive = np.zeros(3, dtype=np.int64)
    on_support = np.zeros(3, dtype=np.int64)
    on_predicted_count = np.zeros(3, dtype=np.int64)
    on_target_count = on_exact = 0

    for predicted_classes, target_classes in records:
        predicted, truth = _validate_class_arrays(predicted_classes, target_classes)
        predicted_on = (predicted >= 2).reshape(-1)
        true_on = (truth >= 2).reshape(-1)
        predicted_up = ~predicted_on[:-1] & predicted_on[1:]
        true_up = ~true_on[:-1] & true_on[1:]
        predicted_down = predicted_on[:-1] & ~predicted_on[1:]
        true_down = true_on[:-1] & ~true_on[1:]
        event_pairs.append((true_up, predicted_up, true_down, predicted_down))
        for destination, source in (
            (up_totals, _event_metrics(true_up, predicted_up)),
            (down_totals, _event_metrics(true_down, predicted_down)),
            (
                combined_totals,
                _event_metrics(true_up | true_down, predicted_up | predicted_down),
            ),
        ):
            for key in destination:
                destination[key] += int(source[key])
        true_transition = true_up | true_down
        predicted_delta = predicted_on[1:].astype(np.int8) - predicted_on[:-1].astype(np.int8)
        true_delta = true_on[1:].astype(np.int8) - true_on[:-1].astype(np.int8)
        steady = ~true_transition
        steady_count += int(steady.sum())
        steady_correct += int((predicted_on[1:][steady] == true_on[1:][steady]).sum())
        true_transition_count += int(true_transition.sum())
        direction_correct += int(
            (np.sign(predicted_delta[true_transition]) == np.sign(true_delta[true_transition])).sum()
        )
        true_patterns = _repedal_intervals(true_on, SHORT_REPEDAL_MAX_SAMPLES)
        predicted_patterns = _repedal_intervals(predicted_on, SHORT_REPEDAL_MAX_SAMPLES)
        true_repedal += len(true_patterns)
        predicted_repedal += len(predicted_patterns)
        repedal_true_positive += len(true_patterns & predicted_patterns)
        true_states.append(true_on)
        predicted_states.append(predicted_on)
        zero_truth.append((truth == 0).reshape(-1))
        zero_prediction.append((predicted == 0).reshape(-1))
        mask = truth >= 2
        on_target_count += int(mask.sum())
        on_exact += int((predicted[mask] == truth[mask]).sum())
        on_truth = truth[mask] - 2
        on_predicted = predicted[mask]
        on_support += np.bincount(on_truth, minlength=3)
        for class_index in range(3):
            class_id = class_index + 2
            on_predicted_count[class_index] += int((on_predicted == class_id).sum())
            on_true_positive[class_index] += int(
                ((on_truth == class_index) & (on_predicted == class_id)).sum()
            )

    state = _binary_classification(
        np.concatenate(predicted_states), np.concatenate(true_states)
    )
    zero = _binary_classification(
        np.concatenate(zero_prediction), np.concatenate(zero_truth)
    )

    def finish_event(counts: Mapping[str, int]) -> dict[str, Any]:
        precision = _safe_ratio(counts["true_positive"], counts["predicted_count"])
        recall = _safe_ratio(counts["true_positive"], counts["true_count"])
        return {
            **counts,
            "precision": precision,
            "recall": recall,
            "f1": _safe_ratio(2.0 * precision * recall, precision + recall),
        }

    on_precision = np.asarray(
        [
            _safe_ratio(on_true_positive[index], on_predicted_count[index])
            for index in range(3)
        ]
    )
    on_recall = np.asarray(
        [
            _safe_ratio(on_true_positive[index], on_support[index])
            for index in range(3)
        ]
    )
    on_f1 = np.asarray(
        [
            _safe_ratio(
                2.0 * on_precision[index] * on_recall[index],
                on_precision[index] + on_recall[index],
            )
            for index in range(3)
        ]
    )
    repedal_precision = _safe_ratio(repedal_true_positive, predicted_repedal)
    repedal_recall = _safe_ratio(repedal_true_positive, true_repedal)
    return {
        "sequence_order": "Within each performance, note-major Pedal1->Pedal2->Pedal3->Pedal4; no cross-performance boundary.",
        "state": state,
        "off_to_on_transition": finish_event(up_totals),
        "on_to_off_transition": finish_event(down_totals),
        "combined_binary_transition": finish_event(combined_totals),
        "steady_position_state_count": steady_count,
        "steady_position_state_accuracy": _safe_ratio(steady_correct, steady_count),
        "transition_direction_accuracy": _safe_ratio(direction_correct, true_transition_count),
        "transition_timing_sample_offset": _timing_diagnostic(event_pairs),
        "zero_vs_nonzero": zero,
        "on_region_depth_class_accuracy": _safe_ratio(on_exact, on_target_count),
        "on_region_macro_f1": float(on_f1.mean()),
        "on_region_per_class_f1": {
            CLASS_NAMES[index + 2]: float(on_f1[index]) for index in range(3)
        },
        "short_repedal_like": {
            "definition": ">=64 -> <64 -> >=64 with OFF run <=4 flattened samples; exact recall requires both transition boundaries at the same samples",
            "sample_based_not_milliseconds": True,
            "true_count": true_repedal,
            "predicted_count": predicted_repedal,
            "true_positive": repedal_true_positive,
            "precision": repedal_precision,
            "recall": repedal_recall,
            "f1": _safe_ratio(
                2.0 * repedal_precision * repedal_recall,
                repedal_precision + repedal_recall,
            ),
        },
    }


def _decoded_transition_counts(predicted: np.ndarray, truth: np.ndarray) -> dict[str, int]:
    decoded = np.asarray(predicted, dtype=np.int64).reshape(-1)
    target = np.asarray(truth, dtype=np.int64).reshape(-1)
    if decoded.shape != target.shape or len(target) < 2:
        raise ValueError("decoded transition arrays are invalid")
    true_delta = np.diff(target)
    predicted_delta = np.diff(decoded)
    true_transition = true_delta != 0
    predicted_transition = predicted_delta != 0
    steady = ~true_transition
    return {
        "position_count": len(true_delta),
        "true_transition_count": int(true_transition.sum()),
        "predicted_transition_count": int(predicted_transition.sum()),
        "transition_true_positive": int((true_transition & predicted_transition).sum()),
        "transition_value_exact": int((true_transition & (decoded[1:] == target[1:])).sum()),
        "steady_count": int(steady.sum()),
        "steady_value_exact": int((steady & (decoded[1:] == target[1:])).sum()),
        "direction_correct": int(
            (np.sign(predicted_delta[true_transition]) == np.sign(true_delta[true_transition])).sum()
        ),
        "delta_absolute_error_sum": int(np.abs(predicted_delta - true_delta).sum()),
    }


def _quantiles(errors: np.ndarray) -> dict[str, float]:
    values = np.asarray(errors, dtype=np.float64)
    if not len(values):
        raise ValueError("error quantiles require values")
    return {
        name: float(np.quantile(values, quantile))
        for name, quantile in (
            ("q01", 0.01),
            ("q05", 0.05),
            ("q25", 0.25),
            ("median", 0.50),
            ("q75", 0.75),
            ("q95", 0.95),
            ("q99", 0.99),
        )
    }


def decoded_value_metrics(
    records: Sequence[tuple[np.ndarray, np.ndarray]]
) -> dict[str, Any]:
    """Compare canonical decoded predictions with original raw targets."""

    if not records:
        raise ValueError("decoded metrics require performance records")
    errors: list[np.ndarray] = []
    on_errors: list[np.ndarray] = []
    subthreshold_errors: list[np.ndarray] = []
    zero_errors: list[np.ndarray] = []
    slot_error_sum = np.zeros(PEDAL_SLOTS, dtype=np.int64)
    slot_count = np.zeros(PEDAL_SLOTS, dtype=np.int64)
    performance_maes: list[float] = []
    transition_totals = {
        key: 0
        for key in (
            "position_count",
            "true_transition_count",
            "predicted_transition_count",
            "transition_true_positive",
            "transition_value_exact",
            "steady_count",
            "steady_value_exact",
            "direction_correct",
            "delta_absolute_error_sum",
        )
    }
    for predicted_classes, raw_targets in records:
        classes = np.asarray(predicted_classes, dtype=np.int64)
        truth = np.asarray(raw_targets, dtype=np.int64)
        if classes.shape != truth.shape or truth.ndim != 2 or truth.shape[1] != PEDAL_SLOTS:
            raise ValueError("decoded records must have aligned [notes,4] arrays")
        decoded = np.asarray(decode_five_classes(classes), dtype=np.int64)
        absolute = np.abs(decoded - truth)
        errors.append(absolute.reshape(-1))
        performance_maes.append(float(absolute.mean()))
        on_errors.append(absolute[truth >= 64])
        subthreshold_errors.append(absolute[(truth >= 1) & (truth <= 63)])
        zero_errors.append(absolute[truth == 0])
        slot_error_sum += absolute.sum(axis=0, dtype=np.int64)
        slot_count += len(truth)
        counts = _decoded_transition_counts(decoded, truth)
        for key in transition_totals:
            transition_totals[key] += int(counts[key])
    all_errors = np.concatenate(errors)
    all_on = np.concatenate(on_errors)
    all_subthreshold = np.concatenate(subthreshold_errors)
    all_zero = np.concatenate(zero_errors)
    true_count = transition_totals["true_transition_count"]
    predicted_count = transition_totals["predicted_transition_count"]
    true_positive = transition_totals["transition_true_positive"]
    precision = _safe_ratio(true_positive, predicted_count)
    recall = _safe_ratio(true_positive, true_count)
    return {
        "overall_micro_mae": float(all_errors.mean()),
        "performance_macro_mae": float(np.mean(performance_maes)),
        "median_performance_mae": float(np.median(performance_maes)),
        "on_region_mae": float(all_on.mean()) if len(all_on) else 0.0,
        "subthreshold_region_mae": (
            float(all_subthreshold.mean()) if len(all_subthreshold) else 0.0
        ),
        "zero_target_mae": float(all_zero.mean()) if len(all_zero) else 0.0,
        "per_slot_mae": {
            f"Pedal{slot + 1}": _safe_ratio(slot_error_sum[slot], slot_count[slot])
            for slot in range(PEDAL_SLOTS)
        },
        "tolerance_accuracy": {
            f"plus_minus_{tolerance}": float((all_errors <= tolerance).mean())
            for tolerance in TOLERANCES
        },
        "maximum_absolute_error": int(all_errors.max()),
        "error_quantiles": _quantiles(all_errors),
        "exact_value_transition_accuracy": _safe_ratio(
            transition_totals["transition_value_exact"], true_count
        ),
        "decoded_transition_precision": precision,
        "decoded_transition_recall": recall,
        "decoded_transition_f1": _safe_ratio(2.0 * precision * recall, precision + recall),
        "steady_position_decoded_accuracy": _safe_ratio(
            transition_totals["steady_value_exact"], transition_totals["steady_count"]
        ),
        "decoded_transition_direction_accuracy": _safe_ratio(
            transition_totals["direction_correct"], true_count
        ),
        "delta_mae": _safe_ratio(
            transition_totals["delta_absolute_error_sum"],
            transition_totals["position_count"],
        ),
        "target_count": len(all_errors),
        "performance_count": len(records),
    }


def _generic_overlap_average(
    num_notes: int, windows: Sequence[tuple[int, np.ndarray]]
) -> tuple[np.ndarray, np.ndarray]:
    if num_notes <= 0 or not windows:
        raise ValueError("overlap averaging requires notes and windows")
    starts = [int(start) for start, _ in windows]
    if len(starts) != len(set(starts)):
        raise ValueError("window starts must be unique")
    trailing_shape = np.asarray(windows[0][1]).shape[1:]
    total = np.zeros((num_notes, *trailing_shape), dtype=np.float64)
    count = np.zeros(num_notes, dtype=np.int64)
    for start, values in sorted(windows, key=lambda item: int(item[0])):
        array = np.asarray(values)
        if array.ndim < 2 or array.shape[1:] != trailing_shape or not len(array):
            raise ValueError("overlap window shapes differ")
        end = int(start) + len(array)
        if int(start) < 0 or end > num_notes or not np.isfinite(array).all():
            raise ValueError("invalid overlap window")
        total[int(start) : end] += array.astype(np.float64, copy=False)
        count[int(start) : end] += 1
    if np.any(count < 1):
        raise RuntimeError("overlap reconstruction left notes uncovered")
    denominator = count.reshape((num_notes,) + (1,) * len(trailing_shape))
    averaged = total / denominator
    if not np.isfinite(averaged).all():
        raise FloatingPointError("non-finite overlap average")
    return averaged, count


def _infer_five_class(
    model: FiveClassPedalEncoderModel,
    tokens: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    num_notes = len(tokens)
    starts = generate_window_starts(num_notes, WINDOW_NOTES, STRIDE_NOTES)
    windows: list[tuple[int, np.ndarray]] = []
    with torch.inference_mode():
        for offset in range(0, len(starts), batch_size):
            batch_starts = starts[offset : offset + batch_size]
            samples = [
                _make_window_sample(tokens, start, min(start + WINDOW_NOTES, num_notes))
                for start in batch_starts
            ]
            batch = five_class_collate_fn(samples)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                output = model(
                    input_ids=batch["input_ids"].to(device, non_blocking=True),
                    token_attention_mask=batch["token_attention_mask"].to(
                        device, non_blocking=True
                    ),
                    note_mask=batch["note_mask"].to(device, non_blocking=True),
                )
            if not bool(torch.isfinite(output.logits).all()):
                raise FloatingPointError("non-finite five-class logits")
            logits = output.logits.float().cpu().numpy()
            for batch_index, start in enumerate(batch_starts):
                length = min(WINDOW_NOTES, num_notes - start)
                windows.append((start, logits[batch_index, :length].copy()))
    mean_logits, contributions = average_five_class_logits(num_notes, windows)
    reverse_logits, reverse_contributions = average_five_class_logits(
        num_notes, list(reversed(windows))
    )
    if not np.array_equal(mean_logits, reverse_logits) or not np.array_equal(
        contributions, reverse_contributions
    ):
        raise RuntimeError("five-class overlap reconstruction is order-dependent")
    return {
        "predictions": mean_logits.argmax(axis=-1).astype(np.int64),
        "contribution_count": contributions,
        "num_windows": len(starts),
    }


def _infer_coarse_to_fine(
    model: Stage2CoarseToFineModel,
    tokens: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    num_notes = len(tokens)
    starts = generate_window_starts(num_notes, WINDOW_NOTES, STRIDE_NOTES)
    region_windows: list[tuple[int, np.ndarray]] = []
    depth_windows: list[tuple[int, np.ndarray]] = []
    with torch.inference_mode():
        for offset in range(0, len(starts), batch_size):
            batch_starts = starts[offset : offset + batch_size]
            samples = [
                _make_window_sample(tokens, start, min(start + WINDOW_NOTES, num_notes))
                for start in batch_starts
            ]
            batch = stage2_pedal_collate_fn(samples)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                output = model(
                    input_ids=batch["input_ids"].to(device, non_blocking=True),
                    token_attention_mask=batch["token_attention_mask"].to(
                        device, non_blocking=True
                    ),
                    note_mask=batch["note_mask"].to(device, non_blocking=True),
                )
            if not bool(torch.isfinite(output.region_logits).all()) or not bool(
                torch.isfinite(output.depth_logits).all()
            ):
                raise FloatingPointError("non-finite coarse-to-fine output")
            region = output.region_logits.float().cpu().numpy()
            depth = output.depth_logits.float().cpu().numpy()
            for batch_index, start in enumerate(batch_starts):
                length = min(WINDOW_NOTES, num_notes - start)
                region_windows.append((start, region[batch_index, :length].copy()))
                depth_windows.append((start, depth[batch_index, :length].copy()))
    mean_region, contributions = _generic_overlap_average(num_notes, region_windows)
    mean_depth, depth_contributions = _generic_overlap_average(num_notes, depth_windows)
    reverse_region, reverse_contributions = _generic_overlap_average(
        num_notes, list(reversed(region_windows))
    )
    reverse_depth, reverse_depth_contributions = _generic_overlap_average(
        num_notes, list(reversed(depth_windows))
    )
    if not all(
        (
            np.array_equal(mean_region, reverse_region),
            np.array_equal(mean_depth, reverse_depth),
            np.array_equal(contributions, depth_contributions),
            np.array_equal(contributions, reverse_contributions),
            np.array_equal(contributions, reverse_depth_contributions),
        )
    ):
        raise RuntimeError("coarse-to-fine overlap reconstruction changed by order/head")
    raw_predictions = decode_pedals(
        torch.from_numpy(mean_region.astype(np.float32, copy=False)),
        torch.from_numpy(mean_depth.astype(np.float32, copy=False)),
    ).numpy()
    return {
        "raw_predictions": raw_predictions.astype(np.int64),
        "contribution_count": contributions,
        "num_windows": len(starts),
    }


def _load_five_class_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[FiveClassPedalEncoderModel, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state" not in checkpoint or "configuration" not in checkpoint:
        raise KeyError("five-class checkpoint lacks model_state/configuration")
    configuration = checkpoint["configuration"]
    model = FiveClassPedalEncoderModel.from_pretrained(
        configuration["checkpoint_path"],
        freeze_encoder=False,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    incompatible = model.load_state_dict(checkpoint["model_state"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("five-class state-dict mismatch")
    if len(model.classification_heads) != PEDAL_SLOTS or any(
        head.in_features != 768 or head.out_features != NUM_CLASSES
        for head in model.classification_heads
    ):
        raise RuntimeError("five-class checkpoint architecture mismatch")
    model.to(device).eval()
    return model, {
        "checkpoint": str(checkpoint_path),
        "best_epoch": int(checkpoint.get("best_epoch", 0)),
        "best_validation_loss": float(
            checkpoint.get("best_validation_loss", checkpoint.get("best_validation_ce_loss", 0.0))
        ),
        "architecture": "official PT encoder + four Linear(768,5) heads",
        "decoder": "argmax after overlap-averaged raw logits",
    }


def _load_ctf_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[Stage2CoarseToFineModel, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state" not in checkpoint or "configuration" not in checkpoint:
        raise KeyError("coarse-to-fine checkpoint lacks model_state/configuration")
    configuration = checkpoint["configuration"]
    model = Stage2CoarseToFineModel.from_pretrained(
        configuration["checkpoint_path"],
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    incompatible = model.load_state_dict(checkpoint["model_state"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("coarse-to-fine state-dict mismatch")
    model.to(device).eval()
    return model, {
        "checkpoint": str(checkpoint_path),
        "best_epoch": int(checkpoint.get("best_epoch", 0)),
        "best_validation_loss": float(checkpoint.get("best_validation_loss", 0.0)),
        "architecture": "official PT encoder + four region heads + four depth heads",
        "decoder": "region argmax plus conditional sigmoid depth after raw-output averaging",
    }


def _checkpoint_paths(output_dir: Path, configuration: Mapping[str, Any]) -> dict[str, Path]:
    split_csv = Path(str(configuration["split_csv"])).resolve()
    project_root = split_csv.parents[2]
    configured = configuration.get("validation_reference_checkpoints", {})
    defaults = {
        "ce": project_root / "analysis/stage2_encoder_only_v0/train_v0/best.pt",
        "ordinal": project_root
        / "analysis/stage2_encoder_only_ordinal_v0/lambda_1p0/best.pt",
        "coarse_to_fine": project_root
        / "analysis/stage2_encoder_only_coarse_to_fine_v0/best.pt",
        "five_class": output_dir / "best.pt",
    }
    paths = {
        key: Path(str(configured.get(key, default))).resolve()
        for key, default in defaults.items()
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing evaluation checkpoints: {missing}")
    return paths


def _record_coverage(
    coverage: dict[str, Any], result: Mapping[str, Any], expected_notes: int
) -> None:
    contributions = np.asarray(result["contribution_count"], dtype=np.int64)
    if contributions.shape != (expected_notes,) or int(contributions.min()) < 1:
        raise RuntimeError("invalid overlap contribution coverage")
    coverage["performance_count"] += 1
    coverage["window_count"] += int(result["num_windows"])
    coverage["minimum_contributions"] = min(
        coverage["minimum_contributions"], int(contributions.min())
    )
    coverage["maximum_contributions"] = max(
        coverage["maximum_contributions"], int(contributions.max())
    )


def _print_inference_progress(model_name: str, completed: int, total: int) -> None:
    if completed == 1 or completed % 10 == 0 or completed == total:
        print(
            f"FIVE_CLASS_VALIDATION_PROGRESS model={model_name} "
            f"performances={completed}/{total}",
            flush=True,
        )


def _flat_metric_row(
    model_name: str,
    classification: Mapping[str, Any],
    renderer: Mapping[str, Any],
    decoded: Mapping[str, Any],
    oracle: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    state = renderer["state"]
    transition = renderer["combined_binary_transition"]
    off_to_on = renderer["off_to_on_transition"]
    on_to_off = renderer["on_to_off_transition"]
    repedal = renderer["short_repedal_like"]
    distributions = classification["predicted_distribution"]
    row: dict[str, Any] = {
        "model": model_name,
        "label": MODEL_LABELS[model_name],
        "checkpoint": provenance["checkpoint"],
        "architecture": provenance["architecture"],
        "decoder": provenance["decoder"],
        "best_epoch": provenance["best_epoch"],
        "best_validation_loss": provenance["best_validation_loss"],
        "is_primary_reference": model_name == PRIMARY_REFERENCE,
        "is_new_model": model_name == NEW_MODEL,
        "five_class_token_accuracy": classification["token_accuracy"],
        "five_class_exact_note_accuracy": classification["exact_note_accuracy"],
        "five_class_macro_f1": classification["macro_f1"],
        "five_class_weighted_f1": classification["weighted_f1"],
        "five_class_micro_f1": classification["micro_f1"],
        "zero_recall": classification["zero_recall"],
        "low_recall": classification["low_recall"],
        "mid_recall": classification["mid_recall"],
        "high_recall": classification["high_recall"],
        "full_recall": classification["full_recall"],
        "low_to_zero_count": classification["low_to_zero_count"],
        "low_to_zero_rate": classification["low_to_zero_rate"],
        "low_to_mid_count": classification["low_to_mid_count"],
        "low_to_mid_rate": classification["low_to_mid_rate"],
        "zero_to_low_count": classification["zero_to_low_count"],
        "zero_to_low_rate": classification["zero_to_low_rate"],
        "mid_to_high_count": classification["mid_to_high_count"],
        "mid_to_high_rate": classification["mid_to_high_rate"],
        "high_to_mid_count": classification["high_to_mid_count"],
        "high_to_mid_rate": classification["high_to_mid_rate"],
        "mid_high_full_aggregate_recall": classification[
            "mid_high_full_aggregate_recall"
        ],
        "zero_low_aggregate_recall": classification["zero_low_aggregate_recall"],
        "nonendpoint_endpoint_collapse_ratio": classification[
            "nonendpoint_endpoint_collapse_ratio"
        ],
        "all_five_classes_predicted": classification["all_five_classes_predicted"],
        "off_on_state_accuracy": state["accuracy"],
        "off_on_state_macro_f1": state["macro_f1"],
        "off_on_balanced_accuracy": state["balanced_accuracy"],
        "off_precision": state["off_precision"],
        "off_recall": state["off_recall"],
        "off_f1": state["off_f1"],
        "on_precision": state["on_precision"],
        "on_recall": state["on_recall"],
        "on_f1": state["on_f1"],
        "predicted_off_ratio": _safe_ratio(
            state["predicted_off_count"], state["off_support"] + state["on_support"]
        ),
        "predicted_on_ratio": _safe_ratio(
            state["predicted_on_count"], state["off_support"] + state["on_support"]
        ),
        "binary_transition_precision": transition["precision"],
        "binary_transition_recall": transition["recall"],
        "binary_transition_f1": transition["f1"],
        "off_to_on_transition_true_count": off_to_on["true_count"],
        "off_to_on_transition_predicted_count": off_to_on["predicted_count"],
        "off_to_on_transition_true_positive": off_to_on["true_positive"],
        "off_to_on_transition_precision": off_to_on["precision"],
        "off_to_on_transition_recall": off_to_on["recall"],
        "off_to_on_transition_f1": off_to_on["f1"],
        "on_to_off_transition_true_count": on_to_off["true_count"],
        "on_to_off_transition_predicted_count": on_to_off["predicted_count"],
        "on_to_off_transition_true_positive": on_to_off["true_positive"],
        "on_to_off_transition_precision": on_to_off["precision"],
        "on_to_off_transition_recall": on_to_off["recall"],
        "on_to_off_transition_f1": on_to_off["f1"],
        "steady_position_state_accuracy": renderer["steady_position_state_accuracy"],
        "transition_direction_accuracy": renderer["transition_direction_accuracy"],
        "zero_vs_nonzero_accuracy": renderer["zero_vs_nonzero"]["accuracy"],
        "on_region_3way_accuracy": renderer["on_region_depth_class_accuracy"],
        "on_region_macro_f1": renderer["on_region_macro_f1"],
        "short_repedal_true_count": repedal["true_count"],
        "short_repedal_predicted_count": repedal["predicted_count"],
        "short_repedal_true_positive": repedal["true_positive"],
        "short_repedal_precision": repedal["precision"],
        "short_repedal_recall": repedal["recall"],
        "short_repedal_f1": repedal["f1"],
        "canonical_decoded_overall_mae": decoded["overall_micro_mae"],
        "canonical_decoded_performance_macro_mae": decoded["performance_macro_mae"],
        "canonical_decoded_median_performance_mae": decoded["median_performance_mae"],
        "canonical_decoded_on_region_mae": decoded["on_region_mae"],
        "canonical_decoded_subthreshold_mae": decoded["subthreshold_region_mae"],
        "canonical_decoded_zero_mae": decoded["zero_target_mae"],
        "canonical_decoded_maximum_absolute_error": decoded["maximum_absolute_error"],
        "canonical_decoded_exact_value_transition_accuracy": decoded[
            "exact_value_transition_accuracy"
        ],
        "canonical_decoded_transition_precision": decoded[
            "decoded_transition_precision"
        ],
        "canonical_decoded_transition_recall": decoded["decoded_transition_recall"],
        "canonical_decoded_transition_f1": decoded["decoded_transition_f1"],
        "canonical_decoded_steady_position_accuracy": decoded[
            "steady_position_decoded_accuracy"
        ],
        "canonical_decoded_transition_direction_accuracy": decoded[
            "decoded_transition_direction_accuracy"
        ],
        "canonical_decoded_delta_mae": decoded["delta_mae"],
        "canonical_oracle_overall_mae": oracle["overall_micro_mae"],
        "canonical_oracle_performance_macro_mae": oracle["performance_macro_mae"],
        "canonical_oracle_median_performance_mae": oracle["median_performance_mae"],
        "canonical_oracle_on_region_mae": oracle["on_region_mae"],
        "canonical_oracle_subthreshold_mae": oracle["subthreshold_region_mae"],
        "model_minus_oracle_overall_mae": decoded["overall_micro_mae"]
        - oracle["overall_micro_mae"],
        "model_minus_oracle_on_region_mae": decoded["on_region_mae"]
        - oracle["on_region_mae"],
    }
    for class_id, class_name in enumerate(CLASS_NAMES):
        class_key = class_name.lower()
        row[f"{class_key}_precision"] = float(classification["precision"][class_id])
        row[f"{class_key}_recall"] = float(classification["recall"][class_id])
        row[f"{class_key}_f1"] = float(classification["f1"][class_id])
        row[f"{class_key}_support"] = int(classification["support"][class_id])
        row[f"{class_key}_predicted_count"] = int(
            classification["predicted_count"][class_id]
        )
        row[f"predicted_{class_key}_ratio"] = float(distributions[class_id])
        row[f"target_{class_name.lower()}_ratio"] = float(
            classification["target_distribution"][class_id]
        )
    for tolerance in TOLERANCES:
        row[f"canonical_decoded_tolerance_{tolerance}"] = decoded[
            "tolerance_accuracy"
        ][f"plus_minus_{tolerance}"]
    for slot in range(PEDAL_SLOTS):
        row[f"canonical_decoded_pedal{slot + 1}_mae"] = decoded["per_slot_mae"][
            f"Pedal{slot + 1}"
        ]
        row[f"canonical_oracle_pedal{slot + 1}_mae"] = oracle["per_slot_mae"][
            f"Pedal{slot + 1}"
        ]
    for quantile, value in decoded["error_quantiles"].items():
        row[f"canonical_decoded_error_{quantile}"] = value
    return row


def _all_finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple, np.ndarray)):
        return all(_all_finite(item) for item in np.asarray(value).reshape(-1).tolist())
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    return True


def _success_assessment(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_name = {str(row["model"]): row for row in rows}
    new = by_name[NEW_MODEL]
    reference = by_name[PRIMARY_REFERENCE]
    macro_delta = float(new["five_class_macro_f1"]) - float(
        reference["five_class_macro_f1"]
    )
    transition_delta = float(new["binary_transition_f1"]) - float(
        reference["binary_transition_f1"]
    )
    on_mae_delta = float(new["canonical_decoded_on_region_mae"]) - float(
        reference["canonical_decoded_on_region_mae"]
    )
    criterion_a_checks = {
        "macro_f1_improvement_at_least_0p03": macro_delta >= 0.03,
        "binary_transition_f1_decrease_at_most_0p02": transition_delta >= -0.02,
        "on_region_mae_worsening_at_most_1p0": on_mae_delta <= 1.0,
    }
    criterion_b_checks = {
        "binary_transition_f1_improvement_at_least_0p03": transition_delta >= 0.03,
        "on_region_mae_improvement_at_least_1p0": on_mae_delta <= -1.0,
        "macro_f1_decrease_at_most_0p01": macro_delta >= -0.01,
    }
    predicted_off = float(new["predicted_off_ratio"])
    predicted_on = float(new["predicted_on_ratio"])
    safeguards = {
        "all_five_classes_predicted": bool(new["all_five_classes_predicted"]),
        "no_class_recall_exactly_zero": all(
            float(new[f"{name.lower()}_recall"]) > 0.0 for name in CLASS_NAMES
        ),
        "low_predicted_ratio_nonzero": float(new["predicted_low_ratio"]) > 0.0,
        "output_finite": _all_finite(new),
        "not_collapsed_to_one_binary_state": predicted_off > 0.0 and predicted_on > 0.0,
        "predicted_off_ratio_in_0p10_to_0p80": 0.10 <= predicted_off <= 0.80,
        "predicted_on_ratio_in_0p20_to_0p90": 0.20 <= predicted_on <= 0.90,
        "binary_transition_f1_decrease_at_most_0p02": transition_delta >= -0.02,
        "duplicate_evaluation_absent": len(by_name) == len(MODEL_ORDER),
        "validation_notes_aggregated_once": True,
    }
    criterion_a = all(criterion_a_checks.values())
    criterion_b = all(criterion_b_checks.values())
    return {
        "primary_reference": PRIMARY_REFERENCE,
        "deltas_new_minus_reference": {
            "five_class_macro_f1": macro_delta,
            "binary_transition_f1": transition_delta,
            "on_region_mae": on_mae_delta,
        },
        "criterion_a_checks": criterion_a_checks,
        "criterion_a": criterion_a,
        "criterion_b_checks": criterion_b_checks,
        "criterion_b": criterion_b,
        "safeguards": safeguards,
        "all_safeguards_passed": all(safeguards.values()),
        "passed": (criterion_a or criterion_b) and all(safeguards.values()),
    }


def evaluate(output_dir: str | Path) -> dict[str, Any]:
    """Run the complete validation-only comparison and write required artifacts."""

    started = time.perf_counter()
    output_root = Path(output_dir).resolve()
    configuration = json.loads((output_root / "config.json").read_text(encoding="utf-8"))
    split_csv = Path(str(configuration["split_csv"])).resolve()
    asap_root = Path(str(configuration["asap_root"])).resolve()
    batch_size = int(configuration.get("batch_size", DEFAULT_BATCH_SIZE))
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("five-class validation requires exactly one visible CUDA device")
    device = torch.device("cuda:0")
    gpu = _visible_gpu_identity()
    expected_uuid = configuration.get("expected_gpu_uuid")
    if expected_uuid and gpu["uuid"] != expected_uuid:
        raise RuntimeError("visible GPU UUID differs from the training configuration")
    torch.cuda.set_device(device)

    dataset = Stage2PedalDataset(
        asap_root,
        split_csv,
        "validation",
        window_notes=WINDOW_NOTES,
        stride_notes=STRIDE_NOTES,
        return_metadata=True,
        cache_mode="preload",
    )
    if any(row["split"] != "validation" for row in dataset.performances):
        raise RuntimeError("non-validation row entered evaluation")
    piece_count = len({row["piece_id"] for row in dataset.performances})
    note_count = sum(int(row["num_normalized_notes"]) for row in dataset.performances)
    paths = [row["performance_path"] for row in dataset.performances]
    if (
        dataset.performance_count != EXPECTED_PERFORMANCES
        or piece_count != EXPECTED_PIECES
        or note_count != EXPECTED_NOTES
        or dataset.window_count != EXPECTED_WINDOWS
        or len(paths) != len(set(paths))
    ):
        raise RuntimeError("validation accounting differs from the fixed split")

    raw_targets: list[np.ndarray] = []
    target_classes: list[np.ndarray] = []
    for row in dataset.performances:
        tokens = dataset._token_cache[row["performance_path"]]
        raw = (
            tokens[:, NON_PEDAL_FEATURES:].astype(np.int64) - PEDAL_TOKEN_OFFSET
        )
        classes = np.asarray(classify_pedal_values(raw), dtype=np.int64)
        raw_targets.append(raw)
        target_classes.append(classes)
    if sum(array.size for array in raw_targets) != EXPECTED_TARGETS:
        raise RuntimeError("validation target count mismatch")

    checkpoints = _checkpoint_paths(output_root, configuration)
    predictions: dict[str, list[np.ndarray]] = {name: [] for name in MODEL_ORDER}
    provenance: dict[str, dict[str, Any]] = {}
    coverage = {
        name: {
            "performance_count": 0,
            "window_count": 0,
            "minimum_contributions": math.inf,
            "maximum_contributions": 0,
        }
        for name in ("ce_128", "ordinal_lambda1", "coarse_to_fine_v0", NEW_MODEL)
    }

    # CE baseline: one validation inference pass supplies both fixed decoders.
    model, info = _load_model(checkpoints["ce"], device)
    provenance["ce_128_argmax"] = {
        **info,
        "checkpoint": str(checkpoints["ce"]),
        "decoder": "argmax after overlap-averaged 128-class raw logits",
    }
    provenance[PRIMARY_REFERENCE] = {
        **info,
        "checkpoint": str(checkpoints["ce"]),
        "decoder": "smallest 128-class posterior cumulative value reaching 0.5",
    }
    for index, row in enumerate(dataset.performances):
        result = infer_complete_posterior(
            model,
            dataset._token_cache[row["performance_path"]],
            device,
            batch_size,
        )
        _record_coverage(coverage["ce_128"], result, len(raw_targets[index]))
        raw_argmax = np.asarray(result["mean_logits"]).argmax(axis=-1).astype(np.int64)
        raw_median = posterior_median_decode(result["probabilities"])
        predictions["ce_128_argmax"].append(
            np.asarray(classify_pedal_values(raw_argmax), dtype=np.int64)
        )
        predictions[PRIMARY_REFERENCE].append(
            np.asarray(classify_pedal_values(raw_median), dtype=np.int64)
        )
        _print_inference_progress("ce_128_shared", index + 1, dataset.performance_count)
    del model
    torch.cuda.empty_cache()

    # Ordinal lambda 1.0 retains its predeclared posterior-median decoder.
    model, info = _load_model(checkpoints["ordinal"], device)
    provenance["ordinal_lambda1_posterior_median"] = {
        **info,
        "checkpoint": str(checkpoints["ordinal"]),
        "decoder": "smallest 128-class posterior cumulative value reaching 0.5",
    }
    for index, row in enumerate(dataset.performances):
        result = infer_complete_posterior(
            model,
            dataset._token_cache[row["performance_path"]],
            device,
            batch_size,
        )
        _record_coverage(coverage["ordinal_lambda1"], result, len(raw_targets[index]))
        raw_median = posterior_median_decode(result["probabilities"])
        predictions["ordinal_lambda1_posterior_median"].append(
            np.asarray(classify_pedal_values(raw_median), dtype=np.int64)
        )
        _print_inference_progress("ordinal_lambda1", index + 1, dataset.performance_count)
    del model
    torch.cuda.empty_cache()

    model, info = _load_ctf_model(checkpoints["coarse_to_fine"], device)
    provenance["coarse_to_fine_v0"] = info
    for index, row in enumerate(dataset.performances):
        result = _infer_coarse_to_fine(
            model,
            dataset._token_cache[row["performance_path"]],
            device,
            batch_size,
        )
        _record_coverage(coverage["coarse_to_fine_v0"], result, len(raw_targets[index]))
        predictions["coarse_to_fine_v0"].append(
            np.asarray(classify_pedal_values(result["raw_predictions"]), dtype=np.int64)
        )
        _print_inference_progress("coarse_to_fine_v0", index + 1, dataset.performance_count)
    del model
    torch.cuda.empty_cache()

    model, info = _load_five_class_model(checkpoints["five_class"], device)
    provenance[NEW_MODEL] = info
    for index, row in enumerate(dataset.performances):
        result = _infer_five_class(
            model,
            dataset._token_cache[row["performance_path"]],
            device,
            batch_size,
        )
        _record_coverage(coverage[NEW_MODEL], result, len(raw_targets[index]))
        predictions[NEW_MODEL].append(np.asarray(result["predictions"], dtype=np.int64))
        _print_inference_progress(NEW_MODEL, index + 1, dataset.performance_count)
    del model
    torch.cuda.empty_cache()

    for name, details in coverage.items():
        if (
            details["performance_count"] != EXPECTED_PERFORMANCES
            or details["window_count"] != EXPECTED_WINDOWS
            or details["minimum_contributions"] < 1
        ):
            raise RuntimeError(f"overlap accounting failed for {name}")
        details["minimum_contributions"] = int(details["minimum_contributions"])

    all_target_classes = np.concatenate(target_classes)
    oracle_records = list(zip(target_classes, raw_targets))
    oracle_decoded = decoded_value_metrics(oracle_records)
    classifications: dict[str, dict[str, Any]] = {}
    renderer: dict[str, dict[str, Any]] = {}
    decoded: dict[str, dict[str, Any]] = {}
    comparison_rows: list[dict[str, Any]] = []
    per_performance_rows: list[dict[str, Any]] = []
    per_slot_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []

    sources: dict[str, list[np.ndarray]] = {"ground_truth": target_classes, **predictions}
    for source_name, arrays in sources.items():
        for slot in (None, *range(PEDAL_SLOTS)):
            flattened = np.concatenate(
                [array.reshape(-1) if slot is None else array[:, slot] for array in arrays]
            )
            counts = np.bincount(flattened, minlength=NUM_CLASSES)
            for class_id, class_name in enumerate(CLASS_NAMES):
                distribution_rows.append(
                    {
                        "source": source_name,
                        "slot": "ALL" if slot is None else f"Pedal{slot + 1}",
                        "class_id": class_id,
                        "class_name": class_name,
                        "count": int(counts[class_id]),
                        "ratio": _safe_ratio(counts[class_id], counts.sum()),
                    }
                )

    for model_name in MODEL_ORDER:
        predicted_arrays = predictions[model_name]
        if len(predicted_arrays) != EXPECTED_PERFORMANCES:
            raise RuntimeError(f"prediction performance count mismatch: {model_name}")
        for predicted_array, target_array in zip(predicted_arrays, target_classes):
            _validate_class_arrays(predicted_array, target_array)
        all_predictions = np.concatenate(predicted_arrays)
        classification = classification_metrics(all_predictions, all_target_classes)
        classifications[model_name] = classification
        record_pairs = list(zip(predicted_arrays, target_classes))
        renderer[model_name] = renderer_aware_metrics(record_pairs)
        decoded[model_name] = decoded_value_metrics(list(zip(predicted_arrays, raw_targets)))
        comparison_rows.append(
            _flat_metric_row(
                model_name,
                classification,
                renderer[model_name],
                decoded[model_name],
                oracle_decoded,
                provenance[model_name],
            )
        )
        matrix = np.asarray(classification["confusion_matrix"], dtype=np.int64)
        for true_id, true_name in enumerate(CLASS_NAMES):
            confusion_rows.append(
                {
                    "model": model_name,
                    "true_class_id": true_id,
                    "true_class_name": true_name,
                    **{
                        f"pred_{CLASS_NAMES[predicted_id].lower()}": int(
                            matrix[true_id, predicted_id]
                        )
                        for predicted_id in range(NUM_CLASSES)
                    },
                    "support": int(matrix[true_id].sum()),
                }
            )
            per_class_rows.append(
                {
                    "model": model_name,
                    "class_id": true_id,
                    "class_name": true_name,
                    "precision": float(classification["precision"][true_id]),
                    "recall": float(classification["recall"][true_id]),
                    "f1": float(classification["f1"][true_id]),
                    "support": int(classification["support"][true_id]),
                    "predicted_count": int(classification["predicted_count"][true_id]),
                    "target_ratio": float(classification["target_distribution"][true_id]),
                    "predicted_ratio": float(
                        classification["predicted_distribution"][true_id]
                    ),
                }
            )
        for slot in range(PEDAL_SLOTS):
            slot_predictions = np.concatenate([array[:, slot] for array in predicted_arrays])
            slot_targets = np.concatenate([array[:, slot] for array in target_classes])
            slot_metrics = _classification_from_confusion(
                _confusion_matrix(slot_predictions, slot_targets)
            )
            per_slot_rows.append(
                {
                    "model": model_name,
                    "slot": slot + 1,
                    "token_accuracy": slot_metrics["token_accuracy"],
                    "macro_f1": slot_metrics["macro_f1"],
                    "weighted_f1": slot_metrics["weighted_f1"],
                    **{
                        f"{CLASS_NAMES[class_id].lower()}_recall": float(
                            slot_metrics["recall"][class_id]
                        )
                        for class_id in range(NUM_CLASSES)
                    },
                    "canonical_decoded_mae": decoded[model_name]["per_slot_mae"][
                        f"Pedal{slot + 1}"
                    ],
                }
            )
        for index, row in enumerate(dataset.performances):
            per_classification = classification_metrics(
                predicted_arrays[index], target_classes[index]
            )
            per_renderer = renderer_aware_metrics(
                [(predicted_arrays[index], target_classes[index])]
            )
            per_decoded = decoded_value_metrics(
                [(predicted_arrays[index], raw_targets[index])]
            )
            per_performance_rows.append(
                {
                    "model": model_name,
                    "performance_path": row["performance_path"],
                    "piece_id": row["piece_id"],
                    "note_count": len(raw_targets[index]),
                    "target_count": raw_targets[index].size,
                    "token_accuracy": per_classification["token_accuracy"],
                    "exact_note_accuracy": per_classification["exact_note_accuracy"],
                    "macro_f1": per_classification["macro_f1"],
                    "off_on_state_accuracy": per_renderer["state"]["accuracy"],
                    "off_on_state_macro_f1": per_renderer["state"]["macro_f1"],
                    "binary_transition_f1": per_renderer[
                        "combined_binary_transition"
                    ]["f1"],
                    "transition_direction_accuracy": per_renderer[
                        "transition_direction_accuracy"
                    ],
                    "on_region_3way_accuracy": per_renderer[
                        "on_region_depth_class_accuracy"
                    ],
                    "short_repedal_recall": per_renderer["short_repedal_like"][
                        "recall"
                    ],
                    "canonical_decoded_mae": per_decoded["overall_micro_mae"],
                    "canonical_decoded_on_region_mae": per_decoded["on_region_mae"],
                    "canonical_decoded_subthreshold_mae": per_decoded[
                        "subthreshold_region_mae"
                    ],
                }
            )

    assessment = _success_assessment(comparison_rows)
    reference_rows = [
        row for row in comparison_rows if row["model"] != NEW_MODEL
    ]
    validation_dir = output_root / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    _atomic_csv(validation_dir / "validation_comparison.csv", comparison_rows)
    _atomic_csv(validation_dir / "reference_rebinned_metrics.csv", reference_rows)
    _atomic_csv(validation_dir / "per_performance_metrics.csv", per_performance_rows)
    _atomic_csv(validation_dir / "per_slot_metrics.csv", per_slot_rows)
    _atomic_csv(validation_dir / "five_class_confusion.csv", confusion_rows)
    _atomic_csv(validation_dir / "per_class_metrics.csv", per_class_rows)
    _atomic_csv(validation_dir / "class_distributions.csv", distribution_rows)
    _atomic_json(
        validation_dir / "renderer_aware_metrics.json",
        {
            "class_state_definition": {"OFF": ["ZERO", "LOW"], "ON": ["MID", "HIGH", "FULL"]},
            "sequence_order": "Per performance, note-major Pedal1->Pedal2->Pedal3->Pedal4; never across performances.",
            "models": renderer,
            "success_assessment": assessment,
        },
    )
    _atomic_json(
        validation_dir / "canonical_quantization_gap.json",
        {
            "representatives": REPRESENTATIVES.tolist(),
            "representative_policy": "fixed dataset-independent interval midpoints",
            "validation_quantization_oracle": oracle_decoded,
            "models": {
                name: {
                    "decoded_metrics": decoded[name],
                    "model_minus_oracle_overall_mae": decoded[name]["overall_micro_mae"]
                    - oracle_decoded["overall_micro_mae"],
                    "model_minus_oracle_on_region_mae": decoded[name]["on_region_mae"]
                    - oracle_decoded["on_region_mae"],
                }
                for name in MODEL_ORDER
            },
        },
    )
    _atomic_json(
        validation_dir / "error_quantiles.json",
        {
            "method": "numpy.quantile default linear interpolation",
            "validation_quantization_oracle": oracle_decoded["error_quantiles"],
            "models": {name: decoded[name]["error_quantiles"] for name in MODEL_ORDER},
        },
    )

    elapsed = time.perf_counter() - started
    result = {
        "completed": True,
        "split": "validation",
        "test_rows_used": 0,
        "performance_count": dataset.performance_count,
        "piece_count": piece_count,
        "note_count": note_count,
        "target_count": note_count * PEDAL_SLOTS,
        "window_count": dataset.window_count,
        "representatives": REPRESENTATIVES.tolist(),
        "model_order": list(MODEL_ORDER),
        "primary_reference": PRIMARY_REFERENCE,
        "coverage": coverage,
        "gpu": gpu,
        "success_assessment": assessment,
        "comparison": {row["model"]: row for row in comparison_rows},
        "classification_metrics": classifications,
        "renderer_aware_metrics": renderer,
        "decoded_value_metrics": decoded,
        "validation_quantization_oracle": oracle_decoded,
        "elapsed_seconds": elapsed,
        "output_files": [
            str(validation_dir / name)
            for name in (
                "validation_comparison.csv",
                "per_performance_metrics.csv",
                "per_slot_metrics.csv",
                "five_class_confusion.csv",
                "per_class_metrics.csv",
                "class_distributions.csv",
                "renderer_aware_metrics.json",
                "canonical_quantization_gap.json",
                "error_quantiles.json",
                "reference_rebinned_metrics.csv",
            )
        ],
    }
    if not _all_finite(result):
        raise FloatingPointError("non-finite final validation result")
    print(
        "FIVE_CLASS_VALIDATION_COMPLETE "
        + json.dumps(
            {
                "passed": assessment["passed"],
                "performances": dataset.performance_count,
                "notes": note_count,
                "test_rows_used": 0,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return result


__all__ = [
    "MODEL_ORDER",
    "NEW_MODEL",
    "PRIMARY_REFERENCE",
    "classification_metrics",
    "decoded_value_metrics",
    "evaluate",
    "renderer_aware_metrics",
]
