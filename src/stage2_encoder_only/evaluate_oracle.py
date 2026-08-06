"""Overlap-aware Oracle evaluation for the Stage 2 pedal classifier."""

from __future__ import annotations

import argparse
import csv
import io
import math
import subprocess
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .dataset import (
    MASK_ID,
    NON_PEDAL_FEATURES,
    PEDAL_NUM_CLASSES,
    PEDAL_SLOTS,
    PEDAL_TOKEN_OFFSET,
    Stage2PedalDataset,
    generate_window_starts,
    stage2_pedal_collate_fn,
)
from .model import Stage2PedalEncoderModel
from .training import set_deterministic_seed


WINDOW_NOTES = 512
STRIDE_NOTES = 256
DEFAULT_BATCH_SIZE = 16
DEFAULT_SEED = 20260710
EXPECTED_TEST_PERFORMANCES = 104
EXPECTED_TEST_PIECES = 23
PT_PATTERN_THRESHOLD = 64
PT_PATTERN_EPSILON = 1e-10
PT_PATTERN_SOURCE = "third_party/PianistTransformer/src/evaluate/evaluate.py"
PINNED_PT_COMMIT = "747df2d12291e37f6638b39f1b71517e579ad48c"
TRAINING_PROJECT_COMMIT = "1d64469760e963492897723c92e61ddad7c76a54"

MODEL_ORDER = (
    "stage2",
    "all_zero",
    "all_full",
    "global_majority",
    "slot_majority",
    "persistence",
)
MODEL_LABELS = {
    "stage2": "Stage 2",
    "all_zero": "All zero",
    "all_full": "All full",
    "global_majority": "Global majority",
    "slot_majority": "Slot majority",
    "persistence": "Previous-note persistence*",
}

CSV_FIELDS = [
    "metadata_index",
    "composer",
    "title",
    "piece_id",
    "performance_path",
    "num_notes",
    "num_pedal_tokens",
    "num_windows",
    "max_window_contributions_per_note",
    "stage2_loss",
    "stage2_pedal_token_accuracy",
    "stage2_exact_note_accuracy",
    "stage2_pedal_value_mae",
    "stage2_intermediate_accuracy",
    "stage2_intermediate_mae",
    "stage2_transition_accuracy",
    "stage2_transition_mae",
    "stage2_transition_f1",
    "all_zero_accuracy",
    "all_full_accuracy",
    "global_majority_accuracy",
    "slot_majority_accuracy",
    "persistence_accuracy",
]


def _safe_ratio(numerator: float | int, denominator: float | int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _finite_mean(values: Iterable[float | int]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(finite.mean()) if finite.size else float("nan")


def average_overlapping_logits(
    num_notes: int,
    windows: Sequence[tuple[int, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Average note logits after canonical start-order accumulation."""

    if num_notes <= 0:
        raise ValueError("num_notes must be positive")
    if not windows:
        raise ValueError("at least one window is required")
    starts = [int(start) for start, _ in windows]
    if len(starts) != len(set(starts)):
        raise ValueError("window starts must be unique")

    logit_sum = np.zeros(
        (num_notes, PEDAL_SLOTS, PEDAL_NUM_CLASSES), dtype=np.float64
    )
    contribution_count = np.zeros(num_notes, dtype=np.int64)
    for start, logits in sorted(windows, key=lambda item: int(item[0])):
        values = np.asarray(logits)
        if values.ndim != 3 or values.shape[1:] != (
            PEDAL_SLOTS,
            PEDAL_NUM_CLASSES,
        ):
            raise ValueError("window logits must have shape [notes, 4, 128]")
        end = int(start) + len(values)
        if int(start) < 0 or end > num_notes or not len(values):
            raise ValueError("window lies outside the performance")
        if not np.isfinite(values).all():
            raise FloatingPointError("non-finite window logits")
        logit_sum[int(start) : end] += values.astype(np.float64, copy=False)
        contribution_count[int(start) : end] += 1

    if np.any(contribution_count < 1):
        missing = np.flatnonzero(contribution_count < 1)
        raise RuntimeError(f"uncovered notes: {missing[:10].tolist()}")
    if contribution_count[0] < 1 or contribution_count[-1] < 1:
        raise RuntimeError("first or last note is uncovered")
    mean_logits = logit_sum / contribution_count[:, None, None]
    if not np.isfinite(mean_logits).all():
        raise FloatingPointError("non-finite averaged logits")
    return mean_logits, contribution_count


def standard_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    loss: float = float("nan"),
) -> dict[str, float | int]:
    """Calculate the existing Stage 2 metric definitions with count details."""

    predicted = np.asarray(predictions, dtype=np.int64)
    target = np.asarray(targets, dtype=np.int64)
    if predicted.shape != target.shape or target.ndim != 2 or target.shape[1] != 4:
        raise ValueError("predictions and targets must have shape [notes, 4]")
    valid = target != -100
    if np.any(valid & ((target < 0) | (target >= PEDAL_NUM_CLASSES))):
        raise ValueError("targets must be in [0, 127] or -100")
    valid_target_count = int(valid.sum())
    valid_notes = valid.any(axis=1)
    valid_note_count = int(valid_notes.sum())
    if not valid_target_count or not valid_note_count:
        raise ValueError("metrics require valid targets")

    correct = (predicted == target) & valid
    note_correct = ((predicted == target) | ~valid).all(axis=1) & valid_notes
    absolute_error = np.abs(predicted[valid] - target[valid])
    slot_correct = []
    slot_counts = []
    for slot in range(PEDAL_SLOTS):
        slot_valid = valid[:, slot]
        slot_counts.append(int(slot_valid.sum()))
        slot_correct.append(int(correct[:, slot].sum()))

    loss_value = float(loss)
    if not math.isnan(loss_value) and not math.isfinite(loss_value):
        raise FloatingPointError("non-finite loss")
    return {
        "loss": loss_value,
        "pedal_token_accuracy": _safe_ratio(int(correct.sum()), valid_target_count),
        "exact_note_accuracy": _safe_ratio(int(note_correct.sum()), valid_note_count),
        "pedal1_accuracy": _safe_ratio(slot_correct[0], slot_counts[0]),
        "pedal2_accuracy": _safe_ratio(slot_correct[1], slot_counts[1]),
        "pedal3_accuracy": _safe_ratio(slot_correct[2], slot_counts[2]),
        "pedal4_accuracy": _safe_ratio(slot_correct[3], slot_counts[3]),
        "pedal_value_mae": float(absolute_error.mean()),
        "valid_target_count": valid_target_count,
        "valid_note_count": valid_note_count,
        "_loss_sum": loss_value * valid_target_count,
        "_token_correct": int(correct.sum()),
        "_exact_notes": int(note_correct.sum()),
        "_slot_correct": slot_correct,
        "_slot_counts": slot_counts,
        "_absolute_error_sum": int(absolute_error.sum()),
    }


def aggregate_standard(
    per_performance: Sequence[dict[str, float | int]],
) -> dict[str, float | int]:
    if not per_performance:
        raise ValueError("cannot aggregate an empty metric sequence")
    target_count = sum(int(item["valid_target_count"]) for item in per_performance)
    note_count = sum(int(item["valid_note_count"]) for item in per_performance)
    token_correct = sum(int(item["_token_correct"]) for item in per_performance)
    exact_notes = sum(int(item["_exact_notes"]) for item in per_performance)
    slot_correct = [
        sum(int(item["_slot_correct"][slot]) for item in per_performance)
        for slot in range(PEDAL_SLOTS)
    ]
    slot_counts = [
        sum(int(item["_slot_counts"][slot]) for item in per_performance)
        for slot in range(PEDAL_SLOTS)
    ]
    absolute_error = sum(
        int(item["_absolute_error_sum"]) for item in per_performance
    )
    finite_loss_items = [
        item for item in per_performance if math.isfinite(float(item["loss"]))
    ]
    loss = (
        sum(float(item["_loss_sum"]) for item in finite_loss_items)
        / sum(int(item["valid_target_count"]) for item in finite_loss_items)
        if finite_loss_items
        else float("nan")
    )
    return {
        "loss": loss,
        "pedal_token_accuracy": _safe_ratio(token_correct, target_count),
        "exact_note_accuracy": _safe_ratio(exact_notes, note_count),
        "pedal1_accuracy": _safe_ratio(slot_correct[0], slot_counts[0]),
        "pedal2_accuracy": _safe_ratio(slot_correct[1], slot_counts[1]),
        "pedal3_accuracy": _safe_ratio(slot_correct[2], slot_counts[2]),
        "pedal4_accuracy": _safe_ratio(slot_correct[3], slot_counts[3]),
        "pedal_value_mae": _safe_ratio(absolute_error, target_count),
        "valid_target_count": target_count,
        "valid_note_count": note_count,
    }


def macro_standard(
    per_performance: Sequence[dict[str, float | int]],
) -> dict[str, float]:
    names = (
        "loss",
        "pedal_token_accuracy",
        "exact_note_accuracy",
        "pedal1_accuracy",
        "pedal2_accuracy",
        "pedal3_accuracy",
        "pedal4_accuracy",
        "pedal_value_mae",
        "valid_target_count",
        "valid_note_count",
    )
    return {
        name: _finite_mean(float(item[name]) for item in per_performance)
        for name in names
    }


def intermediate_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> dict[str, float | int]:
    predicted = np.asarray(predictions, dtype=np.int64)
    target = np.asarray(targets, dtype=np.int64)
    if predicted.shape != target.shape:
        raise ValueError("predictions and targets must have the same shape")
    mask = (target >= 1) & (target <= 126)
    count = int(mask.sum())
    values = predicted[mask]
    truth = target[mask]
    correct = int((values == truth).sum())
    absolute_error = int(np.abs(values - truth).sum())
    predicted_zero = int((values == 0).sum())
    predicted_full = int((values == 127).sum())
    return {
        "intermediate_target_count": count,
        "intermediate_exact_accuracy": _safe_ratio(correct, count),
        "intermediate_mae": _safe_ratio(absolute_error, count),
        "intermediate_predicted_zero_ratio": _safe_ratio(predicted_zero, count),
        "intermediate_predicted_full_ratio": _safe_ratio(predicted_full, count),
        "endpoint_collapse_ratio": _safe_ratio(
            predicted_zero + predicted_full, count
        ),
        "_correct": correct,
        "_absolute_error_sum": absolute_error,
        "_predicted_zero": predicted_zero,
        "_predicted_full": predicted_full,
    }


def aggregate_intermediate(
    per_performance: Sequence[dict[str, float | int]],
) -> dict[str, float | int]:
    count = sum(int(item["intermediate_target_count"]) for item in per_performance)
    correct = sum(int(item["_correct"]) for item in per_performance)
    absolute_error = sum(int(item["_absolute_error_sum"]) for item in per_performance)
    zero = sum(int(item["_predicted_zero"]) for item in per_performance)
    full = sum(int(item["_predicted_full"]) for item in per_performance)
    return {
        "intermediate_target_count": count,
        "intermediate_exact_accuracy": _safe_ratio(correct, count),
        "intermediate_mae": _safe_ratio(absolute_error, count),
        "intermediate_predicted_zero_ratio": _safe_ratio(zero, count),
        "intermediate_predicted_full_ratio": _safe_ratio(full, count),
        "endpoint_collapse_ratio": _safe_ratio(zero + full, count),
    }


def macro_intermediate(
    per_performance: Sequence[dict[str, float | int]],
) -> dict[str, float]:
    names = (
        "intermediate_target_count",
        "intermediate_exact_accuracy",
        "intermediate_mae",
        "intermediate_predicted_zero_ratio",
        "intermediate_predicted_full_ratio",
        "endpoint_collapse_ratio",
    )
    return {
        name: _finite_mean(float(item[name]) for item in per_performance)
        for name in names
    }


def distribution_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> dict[str, float | int]:
    predicted = np.asarray(predictions, dtype=np.int64)
    target = np.asarray(targets, dtype=np.int64)
    if predicted.shape != target.shape:
        raise ValueError("predictions and targets must have the same shape")
    valid = target != -100
    values = predicted[valid]
    truth = target[valid]
    count = int(valid.sum())
    predicted_zero = int((values == 0).sum())
    predicted_full = int((values == 127).sum())
    predicted_intermediate = count - predicted_zero - predicted_full
    target_zero = int((truth == 0).sum())
    target_full = int((truth == 127).sum())
    target_intermediate = count - target_zero - target_full
    return {
        "valid_target_count": count,
        "predicted_zero_ratio": _safe_ratio(predicted_zero, count),
        "predicted_full_ratio": _safe_ratio(predicted_full, count),
        "predicted_intermediate_ratio": _safe_ratio(predicted_intermediate, count),
        "target_zero_ratio": _safe_ratio(target_zero, count),
        "target_full_ratio": _safe_ratio(target_full, count),
        "target_intermediate_ratio": _safe_ratio(target_intermediate, count),
        "_predicted_zero": predicted_zero,
        "_predicted_full": predicted_full,
        "_predicted_intermediate": predicted_intermediate,
        "_target_zero": target_zero,
        "_target_full": target_full,
        "_target_intermediate": target_intermediate,
    }


def aggregate_distribution(
    per_performance: Sequence[dict[str, float | int]],
) -> dict[str, float | int]:
    count = sum(int(item["valid_target_count"]) for item in per_performance)
    totals = {
        name: sum(int(item[f"_{name}"]) for item in per_performance)
        for name in (
            "predicted_zero",
            "predicted_full",
            "predicted_intermediate",
            "target_zero",
            "target_full",
            "target_intermediate",
        )
    }
    return {
        "valid_target_count": count,
        **{f"{name}_ratio": _safe_ratio(value, count) for name, value in totals.items()},
    }


def transition_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> dict[str, float | int]:
    """Evaluate transitions within one performance only."""

    predicted = np.asarray(predictions, dtype=np.int64)
    target = np.asarray(targets, dtype=np.int64)
    if predicted.shape != target.shape:
        raise ValueError("predictions and targets must have the same shape")
    valid = target.reshape(-1) != -100
    predicted_flat = predicted.reshape(-1)[valid]
    target_flat = target.reshape(-1)[valid]
    if len(target_flat) < 2:
        raise ValueError("transition metrics require at least two valid targets")

    true_delta = np.diff(target_flat)
    predicted_delta = np.diff(predicted_flat)
    true_transition = true_delta != 0
    predicted_transition = predicted_delta != 0
    steady = ~true_transition
    exact = predicted_flat[1:] == target_flat[1:]
    absolute_error = np.abs(predicted_flat[1:] - target_flat[1:])
    true_count = int(true_transition.sum())
    steady_count = int(steady.sum())
    predicted_count = int(predicted_transition.sum())
    true_positive = int((true_transition & predicted_transition).sum())
    transition_exact = int((exact & true_transition).sum())
    steady_exact = int((exact & steady).sum())
    transition_error = int(absolute_error[true_transition].sum())
    steady_error = int(absolute_error[steady].sum())
    direction_correct = int(
        (
            np.sign(predicted_delta[true_transition])
            == np.sign(true_delta[true_transition])
        ).sum()
    )
    precision = _safe_ratio(true_positive, predicted_count)
    recall = _safe_ratio(true_positive, true_count)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if math.isfinite(precision)
        and math.isfinite(recall)
        and precision + recall
        else float("nan")
    )
    positions = len(target_flat) - 1
    return {
        "transition_position_count": true_count,
        "transition_position_ratio": _safe_ratio(true_count, positions),
        "transition_position_exact_accuracy": _safe_ratio(
            transition_exact, true_count
        ),
        "transition_position_mae": _safe_ratio(transition_error, true_count),
        "steady_position_count": steady_count,
        "steady_position_exact_accuracy": _safe_ratio(steady_exact, steady_count),
        "steady_position_mae": _safe_ratio(steady_error, steady_count),
        "transition_detection_precision": precision,
        "transition_detection_recall": recall,
        "transition_detection_f1": f1,
        "transition_direction_accuracy": _safe_ratio(direction_correct, true_count),
        "_position_count": positions,
        "_predicted_transition_count": predicted_count,
        "_true_positive": true_positive,
        "_transition_exact": transition_exact,
        "_transition_error_sum": transition_error,
        "_steady_exact": steady_exact,
        "_steady_error_sum": steady_error,
        "_direction_correct": direction_correct,
    }


def aggregate_transition(
    per_performance: Sequence[dict[str, float | int]],
) -> dict[str, float | int]:
    if not per_performance:
        raise ValueError("cannot aggregate empty transition metrics")
    summed = {
        name: sum(int(item[name]) for item in per_performance)
        for name in (
            "transition_position_count",
            "steady_position_count",
            "_position_count",
            "_predicted_transition_count",
            "_true_positive",
            "_transition_exact",
            "_transition_error_sum",
            "_steady_exact",
            "_steady_error_sum",
            "_direction_correct",
        )
    }
    precision = _safe_ratio(
        summed["_true_positive"], summed["_predicted_transition_count"]
    )
    recall = _safe_ratio(
        summed["_true_positive"], summed["transition_position_count"]
    )
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if math.isfinite(precision)
        and math.isfinite(recall)
        and precision + recall
        else float("nan")
    )
    return {
        "transition_position_count": summed["transition_position_count"],
        "transition_position_ratio": _safe_ratio(
            summed["transition_position_count"], summed["_position_count"]
        ),
        "transition_position_exact_accuracy": _safe_ratio(
            summed["_transition_exact"], summed["transition_position_count"]
        ),
        "transition_position_mae": _safe_ratio(
            summed["_transition_error_sum"], summed["transition_position_count"]
        ),
        "steady_position_count": summed["steady_position_count"],
        "steady_position_exact_accuracy": _safe_ratio(
            summed["_steady_exact"], summed["steady_position_count"]
        ),
        "steady_position_mae": _safe_ratio(
            summed["_steady_error_sum"], summed["steady_position_count"]
        ),
        "transition_detection_precision": precision,
        "transition_detection_recall": recall,
        "transition_detection_f1": f1,
        "transition_direction_accuracy": _safe_ratio(
            summed["_direction_correct"], summed["transition_position_count"]
        ),
    }


def macro_transition(
    per_performance: Sequence[dict[str, float | int]],
) -> dict[str, float]:
    names = (
        "transition_position_count",
        "transition_position_ratio",
        "transition_position_exact_accuracy",
        "transition_position_mae",
        "steady_position_count",
        "steady_position_exact_accuracy",
        "steady_position_mae",
        "transition_detection_precision",
        "transition_detection_recall",
        "transition_detection_f1",
        "transition_direction_accuracy",
    )
    return {
        name: _finite_mean(float(item[name]) for item in per_performance)
        for name in names
    }


def compute_train_majorities(
    train_targets: Iterable[np.ndarray],
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    global_counts = np.zeros(PEDAL_NUM_CLASSES, dtype=np.int64)
    slot_counts = np.zeros((PEDAL_SLOTS, PEDAL_NUM_CLASSES), dtype=np.int64)
    performance_count = 0
    for targets in train_targets:
        values = np.asarray(targets, dtype=np.int64)
        if values.ndim != 2 or values.shape[1] != PEDAL_SLOTS:
            raise ValueError("train targets must have shape [notes, 4]")
        if np.any((values < 0) | (values >= PEDAL_NUM_CLASSES)):
            raise ValueError("train targets must be in [0, 127]")
        global_counts += np.bincount(
            values.reshape(-1), minlength=PEDAL_NUM_CLASSES
        )
        for slot in range(PEDAL_SLOTS):
            slot_counts[slot] += np.bincount(
                values[:, slot], minlength=PEDAL_NUM_CLASSES
            )
        performance_count += 1
    if not performance_count:
        raise ValueError("train majority calculation requires performances")
    return (
        int(global_counts.argmax()),
        slot_counts.argmax(axis=1).astype(np.int64),
        global_counts,
        slot_counts,
    )


def make_baselines(
    targets: np.ndarray,
    global_majority: int,
    slot_majorities: np.ndarray,
) -> dict[str, np.ndarray]:
    target = np.asarray(targets, dtype=np.int64)
    if target.ndim != 2 or target.shape[1] != PEDAL_SLOTS or not len(target):
        raise ValueError("targets must be a non-empty [notes, 4] array")
    slots = np.asarray(slot_majorities, dtype=np.int64)
    if slots.shape != (PEDAL_SLOTS,):
        raise ValueError("slot_majorities must contain four classes")
    persistence = np.empty_like(target)
    persistence[0] = slots
    if len(target) > 1:
        persistence[1:] = target[:-1]
    return {
        "all_zero": np.zeros_like(target),
        "all_full": np.full_like(target, 127),
        "global_majority": np.full_like(target, int(global_majority)),
        "slot_majority": np.broadcast_to(slots, target.shape).copy(),
        "persistence": persistence,
    }


def pedal_configuration_ids(
    values: np.ndarray,
    threshold: int = PT_PATTERN_THRESHOLD,
) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[1] != PEDAL_SLOTS:
        raise ValueError("pedal values must have shape [notes, 4]")
    binary = array >= threshold
    weights = np.asarray([8, 4, 2, 1], dtype=np.int64)
    return (binary.astype(np.int64) * weights).sum(axis=1)


def configuration_histogram(values: np.ndarray) -> np.ndarray:
    return np.bincount(
        pedal_configuration_ids(values), minlength=16
    ).astype(np.int64)


def distribution_similarity(
    target_histogram: np.ndarray,
    prediction_histogram: np.ndarray,
    epsilon: float = PT_PATTERN_EPSILON,
) -> dict[str, float]:
    """Return normalized base-2 JS divergence/distance and intersection."""

    target_hist = np.asarray(target_histogram, dtype=np.float64)
    prediction_hist = np.asarray(prediction_histogram, dtype=np.float64)
    if target_hist.shape != (16,) or prediction_hist.shape != (16,):
        raise ValueError("configuration histograms must have 16 bins")
    if target_hist.sum() <= 0 or prediction_hist.sum() <= 0:
        raise ValueError("configuration histograms must be non-empty")
    target_probability = target_hist / target_hist.sum() + epsilon
    prediction_probability = prediction_hist / prediction_hist.sum() + epsilon
    target_probability /= target_probability.sum()
    prediction_probability /= prediction_probability.sum()
    midpoint = 0.5 * (target_probability + prediction_probability)
    divergence = 0.5 * float(
        np.sum(target_probability * np.log2(target_probability / midpoint))
        + np.sum(
            prediction_probability
            * np.log2(prediction_probability / midpoint)
        )
    )
    divergence = max(divergence, 0.0)
    return {
        "js_divergence_base2": divergence,
        "js_distance_base2": math.sqrt(divergence),
        "histogram_intersection": float(
            np.minimum(target_probability, prediction_probability).sum()
        ),
    }


def _make_window_sample(
    full_tokens: np.ndarray,
    start: int,
    end: int,
) -> dict[str, Any]:
    original = full_tokens[start:end]
    input_tokens = original.copy()
    input_tokens[:, NON_PEDAL_FEATURES:] = MASK_ID
    targets = original[:, NON_PEDAL_FEATURES:] - PEDAL_TOKEN_OFFSET
    return {
        "input_ids": torch.from_numpy(input_tokens.reshape(-1)).long(),
        "pedal_targets": torch.from_numpy(targets.copy()).long(),
        "note_mask": torch.ones(len(original), dtype=torch.bool),
        "metadata": {"window_start_note": start, "window_end_note": end},
    }


def infer_complete_performance(
    model: Stage2PedalEncoderModel,
    full_tokens: np.ndarray,
    device: torch.device,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    """Infer all windows, average overlap logits, and predict each note once."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    num_notes = len(full_tokens)
    starts = generate_window_starts(num_notes, WINDOW_NOTES, STRIDE_NOTES)
    window_outputs: list[tuple[int, np.ndarray]] = []
    with torch.inference_mode():
        for offset in range(0, len(starts), batch_size):
            batch_starts = starts[offset : offset + batch_size]
            samples = [
                _make_window_sample(
                    full_tokens,
                    start,
                    min(start + WINDOW_NOTES, num_notes),
                )
                for start in batch_starts
            ]
            cpu_batch = stage2_pedal_collate_fn(samples)
            input_ids = cpu_batch["input_ids"].to(device, non_blocking=True)
            attention = cpu_batch["token_attention_mask"].to(
                device, non_blocking=True
            )
            note_mask = cpu_batch["note_mask"].to(device, non_blocking=True)
            with torch.amp.autocast(
                device_type="cuda", dtype=torch.float16, enabled=True
            ):
                output = model(
                    input_ids=input_ids,
                    token_attention_mask=attention,
                    note_mask=note_mask,
                )
            if not bool(torch.isfinite(output.logits).all()):
                raise FloatingPointError("non-finite inference logits")
            logits = output.logits.float().cpu().numpy()
            for batch_index, start in enumerate(batch_starts):
                length = min(WINDOW_NOTES, num_notes - start)
                window_outputs.append(
                    (start, logits[batch_index, :length].copy())
                )

    mean_logits, contribution_count = average_overlapping_logits(
        num_notes, window_outputs
    )
    reversed_logits, reversed_count = average_overlapping_logits(
        num_notes, list(reversed(window_outputs))
    )
    if not np.array_equal(mean_logits, reversed_logits) or not np.array_equal(
        contribution_count, reversed_count
    ):
        raise RuntimeError("window order changed overlap reconstruction")
    predictions = mean_logits.argmax(axis=-1).astype(np.int64)
    targets = (
        full_tokens[:, NON_PEDAL_FEATURES:].astype(np.int64)
        - PEDAL_TOKEN_OFFSET
    )
    logits_tensor = torch.from_numpy(mean_logits.astype(np.float32, copy=False))
    targets_tensor = torch.from_numpy(targets)
    loss = float(
        F.cross_entropy(
            logits_tensor.reshape(-1, PEDAL_NUM_CLASSES),
            targets_tensor.reshape(-1),
        ).item()
    )
    if not math.isfinite(loss):
        raise FloatingPointError("non-finite complete-performance loss")
    return {
        "predictions": predictions,
        "targets": targets,
        "loss": loss,
        "num_windows": len(starts),
        "contribution_count": contribution_count,
        "max_contributions": int(contribution_count.max()),
        "window_starts": starts,
    }


def _load_split_rows(split_csv: Path) -> list[dict[str, str]]:
    with split_csv.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _visible_gpu_identity() -> dict[str, str]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid,name",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [row.strip() for row in result.stdout.splitlines() if row.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"expected one visible GPU, found {len(rows)}")
    uuid, name = [item.strip() for item in rows[0].split(",", 1)]
    return {"uuid": uuid, "name": name}


def _load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[Stage2PedalEncoderModel, dict[str, Any]]:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    required = {
        "model_state",
        "configuration",
        "best_epoch",
        "best_validation_loss",
    }
    missing_checkpoint = required - set(checkpoint)
    if missing_checkpoint:
        raise KeyError(f"best checkpoint missing {sorted(missing_checkpoint)}")
    configuration = checkpoint["configuration"]
    if bool(configuration["freeze_encoder"]):
        raise RuntimeError("checkpoint configuration unexpectedly freezes encoder")
    model = Stage2PedalEncoderModel.from_pretrained(
        configuration["checkpoint_path"],
        freeze_encoder=False,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    incompatible = model.load_state_dict(checkpoint["model_state"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"state-dict mismatch missing={incompatible.missing_keys} "
            f"unexpected={incompatible.unexpected_keys}"
        )
    if len(model.classification_heads) != 4 or any(
        head.in_features != 768 or head.out_features != 128
        for head in model.classification_heads
    ):
        raise RuntimeError("prediction heads do not match four Linear(768, 128)")
    if model.freeze_encoder or not all(
        parameter.requires_grad for parameter in model.encoder.parameters()
    ):
        raise RuntimeError("encoder was accidentally frozen")
    model = model.to(device)
    model.eval()
    return model, {
        "best_epoch": int(checkpoint["best_epoch"]),
        "best_validation_loss": float(checkpoint["best_validation_loss"]),
        "configuration": configuration,
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "architecture": "official PT encoder + four independent Linear(768, 128) heads",
    }


def _targets_from_dataset(dataset: Stage2PedalDataset) -> Iterable[np.ndarray]:
    for row in dataset.performances:
        tokens = dataset._token_cache[row["performance_path"]]
        yield (
            tokens[:, NON_PEDAL_FEATURES:].astype(np.int64)
            - PEDAL_TOKEN_OFFSET
        )


def _percentile(values: Sequence[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    def clean(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    output = [
        "| " + " | ".join(clean(value) for value in headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    output.extend(
        "| " + " | ".join(clean(value) for value in row) + " |"
        for row in rows
    )
    return "\n".join(output)


def _format_metric(value: Any, digits: int = 6) -> str:
    number = float(value)
    return "NA" if not math.isfinite(number) else f"{number:.{digits}f}"


def _aggregate_group(
    records: Sequence[dict[str, Any]], key: str
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record[key])].append(record)
    output = []
    for group_name in sorted(groups):
        members = groups[group_name]
        standard = aggregate_standard(
            [member["stage2_standard"] for member in members]
        )
        output.append(
            {
                "name": group_name,
                "performances": len(members),
                "notes": sum(int(member["num_notes"]) for member in members),
                **standard,
            }
        )
    return output


def _render_report(results: dict[str, Any]) -> str:
    checkpoint = results["checkpoint"]
    verification = results["verification"]
    timing = results["timing"]
    models = results["models"]
    stage2 = models["stage2"]
    records = results["performance_records"]
    target_distribution = results["target_distribution"]

    lines = [
        "# Experiment 1 v0 Oracle Test Evaluation",
        "",
        "## 1. Evaluation scope",
        "",
        "This is an Oracle Stage 2 evaluation: each human ASAP performance's "
        "Pitch/IOI/Velocity/Duration tokens are retained, Pedal1–4 are masked, "
        "and the trained Stage 2 encoder predicts pedal tokens that are compared "
        "with that same human performance's original pedal targets. It is not an "
        "end-to-end score-to-performance evaluation, and original Pianist "
        "Transformer generation was not run.",
        "",
        "## 2. Checkpoint and environment",
        "",
        f"- Best checkpoint: `{results['checkpoint_path']}`",
        f"- Best epoch: {checkpoint['best_epoch']}",
        f"- Recorded best validation loss: {checkpoint['best_validation_loss']:.10f}",
        f"- Exact state-dict load: yes; missing keys={checkpoint['missing_keys']}, "
        f"unexpected keys={checkpoint['unexpected_keys']}",
        f"- Architecture: {checkpoint['architecture']}",
        "- Encoder inference state: `freeze_encoder=False`; encoder parameters "
        "retained `requires_grad=True`, no gradients were created, and no "
        "parameter version changed during inference.",
        f"- Project commit: `{results['project_commit']}`",
        f"- Pinned PT commit: `{results['pt_commit']}`",
        f"- GPU mapping: host GPU 1 UUID `{results['gpu']['uuid']}` -> "
        f"container GPU 0 -> PyTorch `cuda:0` ({results['gpu']['name']})",
        f"- Inference batch size: {results['batch_size']}; FP16 autocast, "
        "`model.eval()`, and `torch.inference_mode()`",
        f"- Dataset preload/majority time: {timing['preparation_seconds']:.3f} s; "
        f"checkpoint/model load: {timing['model_load_seconds']:.3f} s; "
        f"test inference and metrics: {timing['inference_seconds']:.3f} s; "
        f"total: {timing['total_seconds']:.3f} s",
        f"- Peak allocated GPU memory: {timing['peak_gpu_memory_bytes']} bytes "
        f"({timing['peak_gpu_memory_bytes'] / 1024**3:.3f} GiB)",
        "",
        "Checkpoint configuration:",
        "",
        _markdown_table(
            ["Key", "Value"],
            [
                [key, checkpoint["configuration"][key]]
                for key in sorted(checkpoint["configuration"])
            ],
        ),
        "",
        "## 3. Test-set verification",
        "",
        f"- Performances: {verification['performance_count']} (expected 104)",
        f"- Unique pieces: {verification['piece_count']} (expected 23)",
        f"- Complete notes: {verification['note_count']}",
        f"- Complete pedal targets: {verification['pedal_target_count']}",
        f"- Deterministic windows: {verification['window_count']}",
        f"- Unique performance paths: {verification['unique_path_count']}; "
        "all MIDI files existed and tokenized successfully.",
        "- Only `split=test` rows were evaluated; metadata order from the split "
        "CSV was preserved.",
        "- Every original note, including the first and last, had at least one "
        "window contribution. Averaged logits were invariant to reversed window "
        "input order.",
        f"- Contribution counts ranged from {verification['min_contributions']} "
        f"to {verification['max_contributions']} per note. Final aggregation "
        f"counted each of the {verification['note_count']} original notes once, "
        "not once per overlapping window.",
        "",
        "## 4. Overall Stage 2 results",
        "",
        _markdown_table(
            ["Aggregation", "Loss", "Token acc.", "Exact-note acc.", "P1", "P2", "P3", "P4", "MAE", "Targets", "Notes"],
            [
                [
                    label,
                    _format_metric(metrics["loss"]),
                    _format_metric(metrics["pedal_token_accuracy"]),
                    _format_metric(metrics["exact_note_accuracy"]),
                    _format_metric(metrics["pedal1_accuracy"]),
                    _format_metric(metrics["pedal2_accuracy"]),
                    _format_metric(metrics["pedal3_accuracy"]),
                    _format_metric(metrics["pedal4_accuracy"]),
                    _format_metric(metrics["pedal_value_mae"]),
                    _format_metric(metrics["valid_target_count"], 1),
                    _format_metric(metrics["valid_note_count"], 1),
                ]
                for label, metrics in (
                    ("Micro", stage2["standard_micro"]),
                    ("Macro mean", stage2["standard_macro"]),
                )
            ],
        ),
        "",
        "Micro metrics pool correct/error counts across all complete test "
        "performances. Macro metrics first compute a complete-performance metric "
        "and then average across 104 performances.",
        "",
        "## 5. Baseline comparison",
        "",
        _markdown_table(
            ["Method", "Loss", "Token acc.", "Exact note", "P1", "P2", "P3", "P4", "MAE", "Intermediate acc.", "Transition acc.", "Transition F1", "JS distance", "Intersection"],
            [
                [
                    MODEL_LABELS[name],
                    _format_metric(models[name]["standard_micro"]["loss"]),
                    _format_metric(models[name]["standard_micro"]["pedal_token_accuracy"]),
                    _format_metric(models[name]["standard_micro"]["exact_note_accuracy"]),
                    _format_metric(models[name]["standard_micro"]["pedal1_accuracy"]),
                    _format_metric(models[name]["standard_micro"]["pedal2_accuracy"]),
                    _format_metric(models[name]["standard_micro"]["pedal3_accuracy"]),
                    _format_metric(models[name]["standard_micro"]["pedal4_accuracy"]),
                    _format_metric(models[name]["standard_micro"]["pedal_value_mae"]),
                    _format_metric(models[name]["intermediate_micro"]["intermediate_exact_accuracy"]),
                    _format_metric(models[name]["transition_micro"]["transition_position_exact_accuracy"]),
                    _format_metric(models[name]["transition_micro"]["transition_detection_f1"]),
                    _format_metric(models[name]["configuration"]["js_distance_base2"]),
                    _format_metric(models[name]["configuration"]["histogram_intersection"]),
                ]
                for name in MODEL_ORDER
            ],
        ),
        "",
        "`*` Previous-note persistence is teacher-forced and not deployable: the "
        "first note uses train slot-majority classes, and later notes copy the "
        "previous note's complete ground-truth pedal vector. Deterministic "
        "baselines have no cross-entropy entry. Global and slot-majority classes "
        f"were computed from train only: global={results['global_majority']}, "
        f"slots={results['slot_majorities']}.",
        "",
        "## 6. Intermediate-pedal results",
        "",
        f"Human target distribution: zero={target_distribution['target_zero_ratio']:.6f}, "
        f"full-127={target_distribution['target_full_ratio']:.6f}, "
        f"intermediate={target_distribution['target_intermediate_ratio']:.6f}.",
        "",
        _markdown_table(
            ["Method", "Intermediate N", "Intermediate acc.", "Intermediate MAE", "Int. -> 0", "Int. -> 127", "Endpoint collapse", "Pred. zero", "Pred. 127", "Pred. intermediate"],
            [
                [
                    MODEL_LABELS[name],
                    models[name]["intermediate_micro"]["intermediate_target_count"],
                    _format_metric(models[name]["intermediate_micro"]["intermediate_exact_accuracy"]),
                    _format_metric(models[name]["intermediate_micro"]["intermediate_mae"]),
                    _format_metric(models[name]["intermediate_micro"]["intermediate_predicted_zero_ratio"]),
                    _format_metric(models[name]["intermediate_micro"]["intermediate_predicted_full_ratio"]),
                    _format_metric(models[name]["intermediate_micro"]["endpoint_collapse_ratio"]),
                    _format_metric(models[name]["distribution_micro"]["predicted_zero_ratio"]),
                    _format_metric(models[name]["distribution_micro"]["predicted_full_ratio"]),
                    _format_metric(models[name]["distribution_micro"]["predicted_intermediate_ratio"]),
                ]
                for name in MODEL_ORDER
            ],
        ),
        "",
        "## 7. Transition results",
        "",
        "This analysis flattens the original four-sample PT pedal-token sequence "
        "within each performance. It is token-grid analysis, not continuous "
        "raw-CC64 event-timing evaluation, and performance boundaries are never "
        "joined.",
        "Per-performance subset metrics are `nan` when their denominator is "
        "zero (for example, no intermediate targets, true transitions, or "
        "predicted transitions). Macro means exclude only those undefined "
        "performance-level values.",
        "",
        _markdown_table(
            ["Aggregation", "True transitions", "True ratio", "Transition acc.", "Transition MAE", "Steady acc.", "Steady MAE", "Detection precision", "Detection recall", "Detection F1", "Direction acc."],
            [
                [
                    label,
                    _format_metric(metrics["transition_position_count"], 1),
                    _format_metric(metrics["transition_position_ratio"]),
                    _format_metric(metrics["transition_position_exact_accuracy"]),
                    _format_metric(metrics["transition_position_mae"]),
                    _format_metric(metrics["steady_position_exact_accuracy"]),
                    _format_metric(metrics["steady_position_mae"]),
                    _format_metric(metrics["transition_detection_precision"]),
                    _format_metric(metrics["transition_detection_recall"]),
                    _format_metric(metrics["transition_detection_f1"]),
                    _format_metric(metrics["transition_direction_accuracy"]),
                ]
                for label, metrics in (
                    ("Micro", stage2["transition_micro"]),
                    ("Macro mean", stage2["transition_macro"]),
                )
            ],
        ),
        "",
        "## 8. PT-style 16-configuration distribution",
        "",
        f"The exact repository implementation was found at `{PT_PATTERN_SOURCE}` "
        "in `plot_pedal_pattern_distribution` (lines 243–338 at the pinned "
        "commit). It binarizes each slot as 1 for value >= 64, maps Pedal1–4 "
        "with weights 8/4/2/1, adds epsilon 1e-10 to histogram probabilities, "
        "calls SciPy `jensenshannon(..., base=2)`, and defines histogram "
        "intersection as `sum(min(p_i, q_i))`. SciPy returns the square-root "
        "Jensen–Shannon distance even though the upstream variable/report calls "
        "it divergence. The table reports both the true base-2 divergence and "
        "the upstream-comparable base-2 distance; probabilities are renormalized "
        "after epsilon so identical intersections equal one.",
        "",
        _markdown_table(
            ["Method", "Base-2 JS divergence", "PT-code JS distance", "Histogram intersection"],
            [
                [
                    MODEL_LABELS[name],
                    _format_metric(models[name]["configuration"]["js_divergence_base2"]),
                    _format_metric(models[name]["configuration"]["js_distance_base2"]),
                    _format_metric(models[name]["configuration"]["histogram_intersection"]),
                ]
                for name in MODEL_ORDER
            ],
        ),
        "",
        "These values are measured on this leakage-safe ASAP test split and "
        "Oracle pipeline; no paper number is reused as if directly comparable.",
        "",
        "## 9. Performance-level variation",
        "",
    ]

    variation_names = (
        ("Token accuracy", "pedal_token_accuracy"),
        ("Exact-note accuracy", "exact_note_accuracy"),
        ("Pedal-value MAE", "pedal_value_mae"),
    )
    lines.extend(
        [
            _markdown_table(
                ["Metric", "10th percentile", "Median", "90th percentile"],
                [
                    [
                        label,
                        _format_metric(
                            _percentile(
                                [
                                    float(record["stage2_standard"][key])
                                    for record in records
                                ],
                                10,
                            )
                        ),
                        _format_metric(
                            _percentile(
                                [
                                    float(record["stage2_standard"][key])
                                    for record in records
                                ],
                                50,
                            )
                        ),
                        _format_metric(
                            _percentile(
                                [
                                    float(record["stage2_standard"][key])
                                    for record in records
                                ],
                                90,
                            )
                        ),
                    ]
                    for label, key in variation_names
                ],
            ),
            "",
        ]
    )
    ordered = sorted(
        records,
        key=lambda item: (
            float(item["stage2_standard"]["pedal_token_accuracy"]),
            str(item["performance_path"]),
        ),
    )
    for heading, selected in (
        ("10 worst performances by Stage 2 token accuracy", ordered[:10]),
        ("10 best performances by Stage 2 token accuracy", list(reversed(ordered[-10:]))),
    ):
        lines.extend(
            [
                f"### {heading}",
                "",
                _markdown_table(
                    ["Performance", "Composer", "Notes", "Token acc.", "Exact note", "MAE"],
                    [
                        [
                            item["performance_path"],
                            item["composer"],
                            item["num_notes"],
                            _format_metric(item["stage2_standard"]["pedal_token_accuracy"]),
                            _format_metric(item["stage2_standard"]["exact_note_accuracy"]),
                            _format_metric(item["stage2_standard"]["pedal_value_mae"]),
                        ]
                        for item in selected
                    ],
                ),
                "",
            ]
        )

    composer_groups = _aggregate_group(records, "composer")
    piece_groups = _aggregate_group(records, "piece_id")
    lines.extend(
        [
            "### Composer-level aggregate results",
            "",
            _markdown_table(
                ["Composer", "Performances", "Notes", "Token acc.", "Exact note", "MAE"],
                [
                    [
                        item["name"],
                        item["performances"],
                        item["notes"],
                        _format_metric(item["pedal_token_accuracy"]),
                        _format_metric(item["exact_note_accuracy"]),
                        _format_metric(item["pedal_value_mae"]),
                    ]
                    for item in composer_groups
                ],
            ),
            "",
            "These performance, composer, and piece differences are descriptive; "
            "no statistical significance is claimed.",
            "",
            "## 10. Interpretation",
            "",
        ]
    )

    stage2_accuracy = float(stage2["standard_micro"]["pedal_token_accuracy"])
    trivial_names = ("all_zero", "all_full", "global_majority", "slot_majority")
    trivial_best_name = max(
        trivial_names,
        key=lambda name: float(models[name]["standard_micro"]["pedal_token_accuracy"]),
    )
    trivial_best_accuracy = float(
        models[trivial_best_name]["standard_micro"]["pedal_token_accuracy"]
    )
    persistence_accuracy = float(
        models["persistence"]["standard_micro"]["pedal_token_accuracy"]
    )
    transition_accuracy = float(
        stage2["transition_micro"]["transition_position_exact_accuracy"]
    )
    steady_accuracy = float(
        stage2["transition_micro"]["steady_position_exact_accuracy"]
    )
    endpoint_collapse = float(
        stage2["intermediate_micro"]["endpoint_collapse_ratio"]
    )
    intermediate_prediction = float(
        stage2["distribution_micro"]["predicted_intermediate_ratio"]
    )
    worst_piece = min(piece_groups, key=lambda item: float(item["pedal_token_accuracy"]))
    best_piece = max(piece_groups, key=lambda item: float(item["pedal_token_accuracy"]))
    worst_composer = min(
        composer_groups, key=lambda item: float(item["pedal_token_accuracy"])
    )
    best_composer = max(
        composer_groups, key=lambda item: float(item["pedal_token_accuracy"])
    )
    lengths = np.asarray([record["num_notes"] for record in records], dtype=np.float64)
    accuracies = np.asarray(
        [record["stage2_standard"]["pedal_token_accuracy"] for record in records],
        dtype=np.float64,
    )
    length_correlation = float(np.corrcoef(lengths, accuracies)[0, 1])

    lines.extend(
        [
            f"- Trivial/majority baselines: Stage 2 token accuracy "
            f"({_format_metric(stage2_accuracy)}) {'exceeds' if stage2_accuracy > trivial_best_accuracy else 'does not exceed'} "
            f"the strongest trivial/majority baseline, {MODEL_LABELS[trivial_best_name]} "
            f"({_format_metric(trivial_best_accuracy)}).",
            f"- Persistence: Stage 2 {'exceeds' if stage2_accuracy > persistence_accuracy else 'does not exceed'} "
            f"the teacher-forced persistence baseline "
            f"({_format_metric(persistence_accuracy)}).",
            f"- Intermediate behavior: {_format_metric(intermediate_prediction)} of "
            f"all predictions are intermediate; endpoint collapse among "
            f"intermediate targets is {_format_metric(endpoint_collapse)}.",
            f"- Temporal difficulty: transition exact accuracy is "
            f"{_format_metric(transition_accuracy)} versus "
            f"{_format_metric(steady_accuracy)} at steady positions.",
            f"- Concentration: piece-level token accuracy spans "
            f"{_format_metric(worst_piece['pedal_token_accuracy'])} "
            f"({worst_piece['name']}) to "
            f"{_format_metric(best_piece['pedal_token_accuracy'])} "
            f"({best_piece['name']}); composer-level accuracy spans "
            f"{_format_metric(worst_composer['pedal_token_accuracy'])} "
            f"({worst_composer['name']}) to "
            f"{_format_metric(best_composer['pedal_token_accuracy'])} "
            f"({best_composer['name']}). The descriptive Pearson correlation "
            f"between sequence length and performance token accuracy is "
            f"{length_correlation:.4f}.",
            "",
            "## 11. Limitations",
            "",
            "- Inputs are Oracle human-performance non-pedal tokens.",
            "- Stage 1 score-to-performance distribution shift is not evaluated.",
            "- Pedal targets remain the original four-point PT tokenizer representation.",
            "- Transition metrics operate on the token grid, not continuous time.",
            "- No listening test was performed.",
            "- No raw-CC64 event-timing reconstruction was performed.",
            "- Original PT predictions were not compared in this task.",
            "",
            "## 12. Recommended next task",
            "",
        ]
    )
    if stage2_accuracy <= persistence_accuracy:
        recommendation = (
            "Run another targeted diagnostic before end-to-end comparison: quantify "
            "how much the current objective learns beyond teacher-forced state "
            "persistence, then evaluate temporal/delta-aware loss or conditioning "
            "changes. The current model should not be retrained from this test result "
            "without first defining the change on train/validation data."
        )
    elif endpoint_collapse > 0.7:
        recommendation = (
            "Investigate intermediate-depth loss/target balancing on train and "
            "validation data before an end-to-end run, because endpoint collapse "
            "remains dominant despite beating persistence."
        )
    else:
        recommendation = (
            "Proceed to the pre-specified Original PT versus two-stage end-to-end "
            "comparison, while retaining this Oracle result as the Stage 2 upper-bound "
            "diagnostic."
        )
    lines.extend([recommendation, ""])
    return "\n".join(lines)


def _render_csv(rows: Sequence[dict[str, Any]]) -> str:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue()


def evaluate(configuration: dict[str, Any]) -> dict[str, Any]:
    total_start = time.perf_counter()
    asap_root = Path(configuration["asap_root"]).resolve()
    split_csv = Path(configuration["split_csv"]).resolve()
    best_checkpoint = Path(configuration["best_checkpoint"]).resolve()
    project_root = Path(configuration["project_root"]).resolve()
    output_dir = Path(configuration["output_dir"]).resolve()
    smoke_test = bool(configuration["smoke_test"])
    limit = int(configuration["limit_performances"]) if smoke_test else None
    if not smoke_test and output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    if smoke_test and limit != 2:
        raise ValueError("smoke test must evaluate exactly two performances")
    if not best_checkpoint.is_file():
        raise FileNotFoundError(best_checkpoint)
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"expected exactly one visible CUDA device, got {torch.cuda.device_count()}"
        )
    gpu = _visible_gpu_identity()
    if gpu["uuid"] != configuration["expected_gpu_uuid"]:
        raise RuntimeError(
            f"GPU UUID mismatch: expected {configuration['expected_gpu_uuid']}, "
            f"got {gpu['uuid']}"
        )
    device = torch.device("cuda:0")
    set_deterministic_seed(int(configuration["seed"]))
    torch.cuda.set_device(0)
    torch.cuda.synchronize(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)

    all_rows = _load_split_rows(split_csv)
    test_rows = [row for row in all_rows if row["split"] == "test"]
    train_rows = [row for row in all_rows if row["split"] == "train"]
    if len(test_rows) != EXPECTED_TEST_PERFORMANCES:
        raise RuntimeError(f"expected 104 test performances, found {len(test_rows)}")
    if len({row["piece_id"] for row in test_rows}) != EXPECTED_TEST_PIECES:
        raise RuntimeError("test split does not contain 23 unique pieces")
    if len({row["performance_path"] for row in test_rows}) != len(test_rows):
        raise RuntimeError("duplicate test performance path")
    if any(row["split"] != "test" for row in test_rows):
        raise RuntimeError("non-test row selected")
    if not train_rows or any(row["split"] != "train" for row in train_rows):
        raise RuntimeError("invalid train rows for majority calculation")

    preparation_start = time.perf_counter()
    train_dataset = Stage2PedalDataset(
        asap_root,
        split_csv,
        "train",
        window_notes=WINDOW_NOTES,
        stride_notes=STRIDE_NOTES,
        cache_mode="preload",
    )
    global_majority, slot_majorities, _, _ = compute_train_majorities(
        _targets_from_dataset(train_dataset)
    )
    del train_dataset
    test_dataset = Stage2PedalDataset(
        asap_root,
        split_csv,
        "test",
        window_notes=WINDOW_NOTES,
        stride_notes=STRIDE_NOTES,
        cache_mode="preload",
    )
    if test_dataset.performance_count != EXPECTED_TEST_PERFORMANCES:
        raise RuntimeError("Dataset test performance count mismatch")
    if [row["performance_path"] for row in test_dataset.performances] != [
        row["performance_path"] for row in test_rows
    ]:
        raise RuntimeError("Dataset changed test metadata order")
    preparation_seconds = time.perf_counter() - preparation_start

    model_start = time.perf_counter()
    model, checkpoint = _load_model(best_checkpoint, device)
    parameter_versions = {
        name: parameter._version for name, parameter in model.named_parameters()
    }
    model_load_seconds = time.perf_counter() - model_start

    selected_rows = test_dataset.performances[:limit]
    per_model_standard: dict[str, list[dict[str, Any]]] = {
        name: [] for name in MODEL_ORDER
    }
    per_model_intermediate: dict[str, list[dict[str, Any]]] = {
        name: [] for name in MODEL_ORDER
    }
    per_model_transition: dict[str, list[dict[str, Any]]] = {
        name: [] for name in MODEL_ORDER
    }
    per_model_distribution: dict[str, list[dict[str, Any]]] = {
        name: [] for name in MODEL_ORDER
    }
    pattern_histograms = {
        name: np.zeros(16, dtype=np.int64) for name in MODEL_ORDER
    }
    target_pattern_histogram = np.zeros(16, dtype=np.int64)
    csv_rows: list[dict[str, Any]] = []
    performance_records: list[dict[str, Any]] = []
    total_windows = 0
    min_contributions = math.inf
    max_contributions = 0

    inference_start = time.perf_counter()
    torch.cuda.synchronize(device)
    for performance_number, row in enumerate(selected_rows, start=1):
        tokens = test_dataset._token_cache[row["performance_path"]]
        inference = infer_complete_performance(
            model,
            tokens,
            device,
            batch_size=int(configuration["batch_size"]),
        )
        targets = inference["targets"]
        predictions_by_model = {
            "stage2": inference["predictions"],
            **make_baselines(targets, global_majority, slot_majorities),
        }
        target_pattern_histogram += configuration_histogram(targets)
        metrics_for_performance: dict[str, dict[str, Any]] = {}
        for name in MODEL_ORDER:
            predictions = predictions_by_model[name]
            loss = inference["loss"] if name == "stage2" else float("nan")
            standard = standard_metrics(predictions, targets, loss=loss)
            intermediate = intermediate_metrics(predictions, targets)
            transitions = transition_metrics(predictions, targets)
            distribution = distribution_metrics(predictions, targets)
            per_model_standard[name].append(standard)
            per_model_intermediate[name].append(intermediate)
            per_model_transition[name].append(transitions)
            per_model_distribution[name].append(distribution)
            pattern_histograms[name] += configuration_histogram(predictions)
            metrics_for_performance[name] = {
                "standard": standard,
                "intermediate": intermediate,
                "transition": transitions,
            }

        counts = inference["contribution_count"]
        if len(counts) != len(tokens) or int(counts.min()) < 1:
            raise RuntimeError("complete-note coverage validation failed")
        if int(metrics_for_performance["stage2"]["standard"]["valid_note_count"]) != len(tokens):
            raise RuntimeError("duplicate or missing final note accounting")
        total_windows += int(inference["num_windows"])
        min_contributions = min(min_contributions, int(counts.min()))
        max_contributions = max(max_contributions, int(counts.max()))
        stage2_standard = metrics_for_performance["stage2"]["standard"]
        stage2_intermediate = metrics_for_performance["stage2"]["intermediate"]
        stage2_transition = metrics_for_performance["stage2"]["transition"]
        csv_rows.append(
            {
                "metadata_index": row["metadata_index"],
                "composer": row["composer"],
                "title": row["title"],
                "piece_id": row["piece_id"],
                "performance_path": row["performance_path"],
                "num_notes": len(tokens),
                "num_pedal_tokens": len(tokens) * PEDAL_SLOTS,
                "num_windows": inference["num_windows"],
                "max_window_contributions_per_note": inference["max_contributions"],
                "stage2_loss": stage2_standard["loss"],
                "stage2_pedal_token_accuracy": stage2_standard["pedal_token_accuracy"],
                "stage2_exact_note_accuracy": stage2_standard["exact_note_accuracy"],
                "stage2_pedal_value_mae": stage2_standard["pedal_value_mae"],
                "stage2_intermediate_accuracy": stage2_intermediate["intermediate_exact_accuracy"],
                "stage2_intermediate_mae": stage2_intermediate["intermediate_mae"],
                "stage2_transition_accuracy": stage2_transition["transition_position_exact_accuracy"],
                "stage2_transition_mae": stage2_transition["transition_position_mae"],
                "stage2_transition_f1": stage2_transition["transition_detection_f1"],
                "all_zero_accuracy": metrics_for_performance["all_zero"]["standard"]["pedal_token_accuracy"],
                "all_full_accuracy": metrics_for_performance["all_full"]["standard"]["pedal_token_accuracy"],
                "global_majority_accuracy": metrics_for_performance["global_majority"]["standard"]["pedal_token_accuracy"],
                "slot_majority_accuracy": metrics_for_performance["slot_majority"]["standard"]["pedal_token_accuracy"],
                "persistence_accuracy": metrics_for_performance["persistence"]["standard"]["pedal_token_accuracy"],
            }
        )
        performance_records.append(
            {
                **row,
                "num_notes": len(tokens),
                "stage2_standard": stage2_standard,
            }
        )
        print(
            f"evaluated performance {performance_number}/{len(selected_rows)} "
            f"path={row['performance_path']} notes={len(tokens)} "
            f"windows={inference['num_windows']} "
            f"accuracy={stage2_standard['pedal_token_accuracy']:.6f}",
            flush=True,
        )
        del inference, predictions_by_model

    torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - inference_start
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("evaluation unexpectedly created parameter gradients")
    changed_versions = [
        name
        for name, parameter in model.named_parameters()
        if parameter._version != parameter_versions[name]
    ]
    if changed_versions:
        raise RuntimeError(f"model parameters changed during inference: {changed_versions}")

    models: dict[str, dict[str, Any]] = {}
    for name in MODEL_ORDER:
        models[name] = {
            "standard_micro": aggregate_standard(per_model_standard[name]),
            "standard_macro": macro_standard(per_model_standard[name]),
            "intermediate_micro": aggregate_intermediate(
                per_model_intermediate[name]
            ),
            "intermediate_macro": macro_intermediate(
                per_model_intermediate[name]
            ),
            "transition_micro": aggregate_transition(per_model_transition[name]),
            "transition_macro": macro_transition(per_model_transition[name]),
            "distribution_micro": aggregate_distribution(
                per_model_distribution[name]
            ),
            "configuration": distribution_similarity(
                target_pattern_histogram, pattern_histograms[name]
            ),
        }

    note_count = sum(len(test_dataset._token_cache[row["performance_path"]]) for row in selected_rows)
    verification = {
        "performance_count": len(selected_rows),
        "piece_count": len({row["piece_id"] for row in selected_rows}),
        "unique_path_count": len({row["performance_path"] for row in selected_rows}),
        "note_count": note_count,
        "pedal_target_count": note_count * PEDAL_SLOTS,
        "window_count": total_windows,
        "min_contributions": int(min_contributions),
        "max_contributions": int(max_contributions),
        "all_test_performance_count": len(test_rows),
        "all_test_piece_count": len({row["piece_id"] for row in test_rows}),
    }
    results = {
        "checkpoint_path": str(best_checkpoint),
        "checkpoint": checkpoint,
        "project_commit": str(configuration["project_commit"]),
        "pt_commit": str(configuration["pt_commit"]),
        "gpu": gpu,
        "batch_size": int(configuration["batch_size"]),
        "global_majority": global_majority,
        "slot_majorities": slot_majorities.tolist(),
        "verification": verification,
        "models": models,
        "target_distribution": models["stage2"]["distribution_micro"],
        "csv_rows": csv_rows,
        "performance_records": performance_records,
        "timing": {
            "preparation_seconds": preparation_seconds,
            "model_load_seconds": model_load_seconds,
            "inference_seconds": inference_seconds,
            "total_seconds": time.perf_counter() - total_start,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        },
    }
    if smoke_test:
        if len(csv_rows) != 2 or list(csv_rows[0]) != CSV_FIELDS:
            raise RuntimeError("smoke output schema validation failed")
        print(
            "SMOKE_SUCCESS "
            f"performances=2 notes={note_count} windows={total_windows} "
            f"coverage={min_contributions}-{max_contributions} "
            f"loss={models['stage2']['standard_micro']['loss']:.6f} "
            f"accuracy={models['stage2']['standard_micro']['pedal_token_accuracy']:.6f} "
            f"missing_keys={checkpoint['missing_keys']} "
            f"unexpected_keys={checkpoint['unexpected_keys']}",
            flush=True,
        )
        return results

    if len(selected_rows) != EXPECTED_TEST_PERFORMANCES:
        raise RuntimeError("full evaluation did not process all test performances")
    report_text = _render_report(results)
    csv_text = _render_csv(csv_rows)
    output_dir.mkdir(parents=False)
    (output_dir / "per_performance_metrics.csv").write_text(
        csv_text, encoding="utf-8"
    )
    (output_dir / "ORACLE_TEST_EVALUATION.md").write_text(
        report_text, encoding="utf-8"
    )
    print(
        "FULL_EVALUATION_SUCCESS "
        f"performances={len(selected_rows)} notes={note_count} "
        f"windows={total_windows} output={output_dir}",
        flush=True,
    )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root", default="/workspace/project"
    )
    parser.add_argument(
        "--asap-root", default="/workspace/public/ASAP/asap-dataset-v1.1"
    )
    parser.add_argument(
        "--split-csv",
        default="/workspace/project/analysis/stage2_encoder_only_v0/asap_split.csv",
    )
    parser.add_argument(
        "--best-checkpoint",
        default="/workspace/project/analysis/stage2_encoder_only_v0/train_v0/best.pt",
    )
    parser.add_argument(
        "--output-dir",
        default="/workspace/project/analysis/stage2_encoder_only_v0/oracle_test_v0",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--project-commit", default=TRAINING_PROJECT_COMMIT)
    parser.add_argument("--pt-commit", default=PINNED_PT_COMMIT)
    parser.add_argument(
        "--expected-gpu-uuid",
        default="GPU-6982dbee-fbaf-f359-d7ef-a22d0e83400b",
    )
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--limit-performances", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    configuration = vars(parse_args())
    if configuration["batch_size"] <= 0:
        raise ValueError("batch_size must be positive")
    evaluate(configuration)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
