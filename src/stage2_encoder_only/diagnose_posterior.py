"""Validation-only posterior diagnostic for Stage 2 pedal prediction."""

from __future__ import annotations

import argparse
import csv
import io
import math
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .dataset import (
    NON_PEDAL_FEATURES,
    PEDAL_NUM_CLASSES,
    PEDAL_SLOTS,
    PEDAL_TOKEN_OFFSET,
    Stage2PedalDataset,
    generate_window_starts,
    stage2_pedal_collate_fn,
)
from .evaluate_oracle import (
    PINNED_PT_COMMIT,
    STRIDE_NOTES,
    TRAINING_PROJECT_COMMIT,
    WINDOW_NOTES,
    _load_model,
    _make_window_sample,
    _visible_gpu_identity,
    aggregate_distribution,
    aggregate_intermediate,
    aggregate_standard,
    aggregate_transition,
    average_overlapping_logits,
    distribution_metrics,
    intermediate_metrics,
    standard_metrics,
    transition_metrics,
)
from .model import Stage2PedalEncoderModel
from .training import set_deterministic_seed


DEFAULT_BATCH_SIZE = 16
DEFAULT_SEED = 20260710
EXPECTED_VALIDATION_PERFORMANCES = 71
EXPECTED_VALIDATION_PIECES = 19
EXPECTED_VALIDATION_WINDOWS = 1078
GROUP_NAMES = ("zero", "intermediate", "full")
DECODER_NAMES = ("argmax", "posterior_mean", "posterior_median")
DECODER_LABELS = {
    "argmax": "Argmax",
    "posterior_mean": "Posterior mean",
    "posterior_median": "Posterior median",
}
TOLERANCES = (5, 10, 20)

PERFORMANCE_FIELDS = [
    "metadata_index",
    "composer",
    "title",
    "piece_id",
    "performance_path",
    "num_notes",
    "num_pedal_tokens",
    "num_windows",
    "target_zero_ratio",
    "target_full_ratio",
    "target_intermediate_ratio",
    "argmax_accuracy",
    "argmax_mae",
    "argmax_intermediate_accuracy",
    "argmax_transition_f1",
    "posterior_mean_accuracy",
    "posterior_mean_mae",
    "posterior_mean_intermediate_mae",
    "posterior_mean_predicted_intermediate_ratio",
    "posterior_mean_transition_f1",
    "posterior_median_accuracy",
    "posterior_median_mae",
    "posterior_median_intermediate_mae",
    "posterior_median_predicted_intermediate_ratio",
    "posterior_median_transition_f1",
    "mean_intermediate_probability_on_intermediate_targets",
    "median_intermediate_probability_on_intermediate_targets",
    "intermediate_true_class_mean_rank",
    "transition_stale_state_fraction",
    "transition_mean_stale_log_probability_advantage",
]

CLASS_FIELDS = [
    "class",
    "train_support",
    "validation_support",
    "validation_ratio",
    "argmax_prediction_count",
    "argmax_recall",
    "mean_probability_when_true",
    "median_probability_when_true",
    "mean_true_class_rank",
    "top5_recall",
    "top10_recall",
    "posterior_mean_absolute_error_when_true",
    "best_intermediate_recall_when_applicable",
]


def safe_ratio(numerator: float | int, denominator: float | int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def finite_mean(values: Sequence[float | int]) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(finite.mean()) if finite.size else float("nan")


def descriptive_correlation(x: np.ndarray, y: np.ndarray) -> float:
    first = np.asarray(x, dtype=np.float64)
    second = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(first) & np.isfinite(second)
    first = first[finite]
    second = second[finite]
    if (
        first.size < 2
        or float(np.ptp(first)) == 0.0
        or float(np.ptp(second)) == 0.0
    ):
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def target_group_ids(targets: np.ndarray) -> np.ndarray:
    values = np.asarray(targets, dtype=np.int64)
    groups = np.ones(values.shape, dtype=np.int8)
    groups[values == 0] = 0
    groups[values == 127] = 2
    if np.any((values < 0) | (values > 127)):
        raise ValueError("targets must be in [0, 127]")
    return groups


def posterior_group_masses(probabilities: np.ndarray) -> dict[str, np.ndarray]:
    posterior = np.asarray(probabilities, dtype=np.float32)
    if posterior.shape[-1] != PEDAL_NUM_CLASSES:
        raise ValueError("probabilities must end with 128 classes")
    if not np.isfinite(posterior).all():
        raise FloatingPointError("non-finite posterior")
    sums = posterior.sum(axis=-1, dtype=np.float32)
    if not np.allclose(sums, 1.0, atol=2e-5, rtol=2e-5):
        raise ValueError("posterior probabilities do not sum to one")
    zero = posterior[..., 0]
    intermediate = posterior[..., 1:127].sum(axis=-1, dtype=np.float32)
    full = posterior[..., 127]
    best_offset = posterior[..., 1:127].argmax(axis=-1)
    best_class = best_offset.astype(np.int64) + 1
    best_probability = np.take_along_axis(
        posterior, best_class[..., None], axis=-1
    )[..., 0]
    endpoint = zero + full
    return {
        "p_zero": zero,
        "p_intermediate": intermediate,
        "p_full": full,
        "best_intermediate_probability": best_probability,
        "best_intermediate_class": best_class,
        "endpoint_probability": endpoint,
        "endpoint_margin": np.maximum(zero, full) - best_probability,
        "best_intermediate_beats_endpoints": best_probability
        > np.maximum(zero, full),
    }


def softmax_float32(mean_logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(mean_logits, dtype=np.float32)
    if logits.shape[-1] != PEDAL_NUM_CLASSES or not np.isfinite(logits).all():
        raise ValueError("mean logits must be finite with 128 classes")
    posterior = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    if not np.isfinite(posterior).all():
        raise FloatingPointError("non-finite softmax posterior")
    if not np.allclose(posterior.sum(axis=-1), 1.0, atol=2e-5, rtol=2e-5):
        raise RuntimeError("softmax posterior does not sum to one")
    return posterior.astype(np.float32, copy=False)


def posterior_mean_decode(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    posterior = np.asarray(probabilities, dtype=np.float32)
    classes = np.arange(PEDAL_NUM_CLASSES, dtype=np.float32)
    expectation = np.sum(posterior * classes, axis=-1, dtype=np.float32)
    prediction = np.clip(np.rint(expectation), 0, 127).astype(np.int64)
    return prediction, expectation


def posterior_median_decode(probabilities: np.ndarray) -> np.ndarray:
    posterior = np.asarray(probabilities, dtype=np.float32)
    cumulative = np.cumsum(posterior, axis=-1, dtype=np.float32)
    return (cumulative < 0.5).sum(axis=-1).clip(0, 127).astype(np.int64)


def true_class_ranks(probabilities: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Return stable ranks, breaking probability ties by lower class index."""

    posterior = np.asarray(probabilities, dtype=np.float32)
    truth = np.asarray(targets, dtype=np.int64)
    if posterior.shape[:-1] != truth.shape:
        raise ValueError("posterior and target shapes do not match")
    true_probability = np.take_along_axis(
        posterior, truth[..., None], axis=-1
    )[..., 0]
    classes = np.arange(PEDAL_NUM_CLASSES, dtype=np.int64)
    greater = (posterior > true_probability[..., None]).sum(axis=-1)
    tied_lower = (
        (posterior == true_probability[..., None])
        & (classes < truth[..., None])
    ).sum(axis=-1)
    return (1 + greater + tied_lower).astype(np.int16)


def intermediate_true_ranks(
    probabilities: np.ndarray, targets: np.ndarray
) -> np.ndarray:
    posterior = np.asarray(probabilities, dtype=np.float32)[..., 1:127]
    truth = np.asarray(targets, dtype=np.int64)
    if np.any((truth < 1) | (truth > 126)):
        raise ValueError("intermediate ranks require targets in [1, 126]")
    true_index = truth - 1
    true_probability = np.take_along_axis(
        posterior, true_index[..., None], axis=-1
    )[..., 0]
    classes = np.arange(126, dtype=np.int64)
    greater = (posterior > true_probability[..., None]).sum(axis=-1)
    tied_lower = (
        (posterior == true_probability[..., None])
        & (classes < true_index[..., None])
    ).sum(axis=-1)
    return (1 + greater + tied_lower).astype(np.int16)


def posterior_entropy(probabilities: np.ndarray) -> np.ndarray:
    posterior = np.asarray(probabilities, dtype=np.float32)
    positive = np.clip(posterior, np.finfo(np.float32).tiny, 1.0)
    return -np.sum(posterior * np.log(positive), axis=-1, dtype=np.float32)


def coarse_confusion(
    target_groups: np.ndarray, predicted_groups: np.ndarray
) -> dict[str, Any]:
    truth = np.asarray(target_groups, dtype=np.int64).reshape(-1)
    prediction = np.asarray(predicted_groups, dtype=np.int64).reshape(-1)
    if truth.shape != prediction.shape:
        raise ValueError("coarse target and prediction shapes differ")
    if np.any((truth < 0) | (truth > 2) | (prediction < 0) | (prediction > 2)):
        raise ValueError("coarse groups must be in [0, 2]")
    matrix = np.zeros((3, 3), dtype=np.int64)
    np.add.at(matrix, (truth, prediction), 1)
    return confusion_from_matrix(matrix)


def confusion_from_matrix(matrix: np.ndarray) -> dict[str, Any]:
    values = np.asarray(matrix, dtype=np.int64)
    if values.shape != (3, 3):
        raise ValueError("confusion matrix must be 3x3")
    total = int(values.sum())
    accuracy = safe_ratio(int(np.trace(values)), total)
    precision = []
    recall = []
    f1 = []
    for group in range(3):
        p = safe_ratio(int(values[group, group]), int(values[:, group].sum()))
        r = safe_ratio(int(values[group, group]), int(values[group].sum()))
        score = (
            2 * p * r / (p + r)
            if math.isfinite(p) and math.isfinite(r) and p + r
            else float("nan")
        )
        precision.append(p)
        recall.append(r)
        f1.append(score)
    return {
        "confusion_matrix": values,
        "group_accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": finite_mean(f1),
        "balanced_accuracy": finite_mean(recall),
    }


def decoder_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    predicted = np.asarray(predictions, dtype=np.int64)
    truth = np.asarray(targets, dtype=np.int64)
    standard = standard_metrics(predicted, truth)
    intermediate = intermediate_metrics(predicted, truth)
    transitions = transition_metrics(predicted, truth)
    distribution = distribution_metrics(predicted, truth)
    error = predicted.reshape(-1) - truth.reshape(-1)
    intermediate_mask = (truth.reshape(-1) >= 1) & (truth.reshape(-1) <= 126)
    intermediate_error = error[intermediate_mask]
    flat_prediction = predicted.reshape(-1)
    flat_truth = truth.reshape(-1)
    delta_error = np.diff(flat_prediction) - np.diff(flat_truth)
    return {
        "standard": standard,
        "intermediate": intermediate,
        "transition": transitions,
        "distribution": distribution,
        "_squared_error_sum": int(np.square(error, dtype=np.int64).sum()),
        "_target_count": len(error),
        "_tolerance_correct": {
            tolerance: int((np.abs(error) <= tolerance).sum())
            for tolerance in TOLERANCES
        },
        "_intermediate_tolerance_correct": {
            tolerance: int((np.abs(intermediate_error) <= tolerance).sum())
            for tolerance in TOLERANCES
        },
        "_delta_absolute_error_sum": int(np.abs(delta_error).sum()),
        "_delta_count": len(delta_error),
    }


def aggregate_decoder_metrics(
    per_performance: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    standard = aggregate_standard([item["standard"] for item in per_performance])
    intermediate = aggregate_intermediate(
        [item["intermediate"] for item in per_performance]
    )
    transitions = aggregate_transition(
        [item["transition"] for item in per_performance]
    )
    distribution = aggregate_distribution(
        [item["distribution"] for item in per_performance]
    )
    target_count = sum(int(item["_target_count"]) for item in per_performance)
    intermediate_count = int(intermediate["intermediate_target_count"])
    squared_error = sum(
        int(item["_squared_error_sum"]) for item in per_performance
    )
    delta_error = sum(
        int(item["_delta_absolute_error_sum"]) for item in per_performance
    )
    delta_count = sum(int(item["_delta_count"]) for item in per_performance)
    return {
        **standard,
        "rmse": math.sqrt(safe_ratio(squared_error, target_count)),
        **{
            f"tolerance_accuracy_{tolerance}": safe_ratio(
                sum(
                    int(item["_tolerance_correct"][tolerance])
                    for item in per_performance
                ),
                target_count,
            )
            for tolerance in TOLERANCES
        },
        "intermediate_exact_accuracy": intermediate["intermediate_exact_accuracy"],
        "intermediate_mae": intermediate["intermediate_mae"],
        **{
            f"intermediate_tolerance_accuracy_{tolerance}": safe_ratio(
                sum(
                    int(item["_intermediate_tolerance_correct"][tolerance])
                    for item in per_performance
                ),
                intermediate_count,
            )
            for tolerance in TOLERANCES
        },
        "predicted_zero_ratio": distribution["predicted_zero_ratio"],
        "predicted_full_ratio": distribution["predicted_full_ratio"],
        "predicted_intermediate_ratio": distribution[
            "predicted_intermediate_ratio"
        ],
        "endpoint_collapse_ratio": intermediate["endpoint_collapse_ratio"],
        "transition_position_exact_accuracy": transitions[
            "transition_position_exact_accuracy"
        ],
        "transition_position_mae": transitions["transition_position_mae"],
        "steady_position_exact_accuracy": transitions[
            "steady_position_exact_accuracy"
        ],
        "steady_position_mae": transitions["steady_position_mae"],
        "transition_detection_precision": transitions[
            "transition_detection_precision"
        ],
        "transition_detection_recall": transitions[
            "transition_detection_recall"
        ],
        "transition_detection_f1": transitions["transition_detection_f1"],
        "delta_mae": safe_ratio(delta_error, delta_count),
        "valid_target_count": target_count,
        "valid_note_count": standard["valid_note_count"],
    }


def endpoint_mixture_diagnostic(
    probabilities: np.ndarray,
    posterior_mean_predictions: np.ndarray,
    targets: np.ndarray,
) -> dict[str, dict[str, Any]]:
    posterior = np.asarray(probabilities, dtype=np.float32).reshape(
        -1, PEDAL_NUM_CLASSES
    )
    prediction = np.asarray(posterior_mean_predictions, dtype=np.int64).reshape(-1)
    truth = np.asarray(targets, dtype=np.int64).reshape(-1)
    mean_intermediate = (prediction >= 1) & (prediction <= 126)
    true_intermediate = (truth >= 1) & (truth <= 126)
    classes = np.arange(PEDAL_NUM_CLASSES, dtype=np.int64)
    local_mass_5 = np.sum(
        posterior
        * (np.abs(classes[None, :] - prediction[:, None]) <= 5),
        axis=1,
        dtype=np.float32,
    )
    local_mass_10 = np.sum(
        posterior
        * (np.abs(classes[None, :] - prediction[:, None]) <= 10),
        axis=1,
        dtype=np.float32,
    )
    endpoint_mass = posterior[:, 0] + posterior[:, 127]
    locally_supported = local_mass_10 >= 0.25
    endpoint_mixture = (endpoint_mass >= 0.75) & (local_mass_10 < 0.25)
    diffuse = ~(locally_supported | endpoint_mixture)
    category_masks = {
        "locally_supported": locally_supported,
        "endpoint_mixture_dominated": endpoint_mixture,
        "diffuse_other": diffuse,
    }
    output: dict[str, dict[str, Any]] = {}
    for scope, scope_mask, denominator in (
        ("all", mean_intermediate, len(truth)),
        (
            "true_intermediate",
            mean_intermediate & true_intermediate,
            int(true_intermediate.sum()),
        ),
    ):
        count = int(scope_mask.sum())
        entry: dict[str, Any] = {
            "denominator": denominator,
            "intermediate_prediction_count": count,
            "intermediate_prediction_ratio": safe_ratio(count, denominator),
            "mean_local_mass_5": float(local_mass_5[scope_mask].mean())
            if count
            else float("nan"),
            "mean_local_mass_10": float(local_mass_10[scope_mask].mean())
            if count
            else float("nan"),
        }
        for category, category_mask in category_masks.items():
            selected = scope_mask & category_mask
            category_count = int(selected.sum())
            entry[f"{category}_count"] = category_count
            entry[f"{category}_ratio"] = safe_ratio(category_count, count)
            entry[f"{category}_absolute_error_sum"] = int(
                np.abs(prediction[selected] - truth[selected]).sum()
            )
        output[scope] = entry
    return output


def aggregate_endpoint_mixture(
    per_performance: Sequence[dict[str, dict[str, Any]]]
) -> dict[str, dict[str, float | int]]:
    output: dict[str, dict[str, float | int]] = {}
    for scope in ("all", "true_intermediate"):
        denominator = sum(int(item[scope]["denominator"]) for item in per_performance)
        count = sum(
            int(item[scope]["intermediate_prediction_count"])
            for item in per_performance
        )
        entry: dict[str, float | int] = {
            "denominator": denominator,
            "intermediate_prediction_count": count,
            "intermediate_prediction_ratio": safe_ratio(count, denominator),
        }
        for category in (
            "locally_supported",
            "endpoint_mixture_dominated",
            "diffuse_other",
        ):
            category_count = sum(
                int(item[scope][f"{category}_count"])
                for item in per_performance
            )
            error_sum = sum(
                int(item[scope][f"{category}_absolute_error_sum"])
                for item in per_performance
            )
            entry[f"{category}_count"] = category_count
            entry[f"{category}_ratio"] = safe_ratio(category_count, count)
            entry[f"{category}_mae"] = safe_ratio(error_sum, category_count)
        output[scope] = entry
    return output


def transition_posterior_diagnostic(
    probabilities: np.ndarray,
    targets: np.ndarray,
    ranks: np.ndarray,
    argmax_predictions: np.ndarray,
    mean_predictions: np.ndarray,
    median_predictions: np.ndarray,
) -> dict[str, Any]:
    posterior = np.asarray(probabilities, dtype=np.float32).reshape(
        -1, PEDAL_NUM_CLASSES
    )
    truth = np.asarray(targets, dtype=np.int64).reshape(-1)
    rank = np.asarray(ranks).reshape(-1)
    argmax = np.asarray(argmax_predictions, dtype=np.int64).reshape(-1)
    mean = np.asarray(mean_predictions, dtype=np.int64).reshape(-1)
    median = np.asarray(median_predictions, dtype=np.int64).reshape(-1)
    if len(truth) < 2:
        raise ValueError("transition diagnostic requires two targets")
    groups = posterior_group_masses(posterior)
    entropy = posterior_entropy(posterior)
    true_probability = np.take_along_axis(
        posterior, truth[:, None], axis=-1
    )[:, 0]
    delta = np.diff(truth)
    transition = delta != 0
    current_indices = np.arange(1, len(truth))
    subset_output: dict[str, dict[str, float | int]] = {}
    for name, subset in (("transition", transition), ("steady", ~transition)):
        indices = current_indices[subset]
        count = len(indices)
        subset_output[name] = {
            "count": count,
            "ratio": safe_ratio(count, len(current_indices)),
            "p_zero_sum": float(groups["p_zero"][indices].sum()),
            "p_full_sum": float(groups["p_full"][indices].sum()),
            "p_intermediate_sum": float(
                groups["p_intermediate"][indices].sum()
            ),
            "true_probability_sum": float(true_probability[indices].sum()),
            "rank_sum": int(rank[indices].sum()),
            "entropy_sum": float(entropy[indices].sum()),
            "endpoint_margin_sum": float(groups["endpoint_margin"][indices].sum()),
            "argmax_correct": int((argmax[indices] == truth[indices]).sum()),
            "mean_absolute_error_sum": int(
                np.abs(mean[indices] - truth[indices]).sum()
            ),
            "median_absolute_error_sum": int(
                np.abs(median[indices] - truth[indices]).sum()
            ),
        }

    tiny = np.finfo(np.float32).tiny
    transition_indices = current_indices[transition]
    previous_targets = truth[transition_indices - 1]
    current_targets = truth[transition_indices]
    previous_probability = posterior[transition_indices, previous_targets]
    current_probability = posterior[transition_indices, current_targets]
    advantage = np.log(np.maximum(previous_probability, tiny)) - np.log(
        np.maximum(current_probability, tiny)
    )
    transition_delta = current_targets - previous_targets
    endpoint_to_endpoint = np.isin(previous_targets, (0, 127)) & np.isin(
        current_targets, (0, 127)
    )
    involving_intermediate = (
        ((previous_targets >= 1) & (previous_targets <= 126))
        | ((current_targets >= 1) & (current_targets <= 126))
    )
    stale_groups: dict[str, dict[str, Any]] = {}
    for name, mask in (
        ("all", np.ones(len(transition_indices), dtype=bool)),
        ("upward", transition_delta > 0),
        ("downward", transition_delta < 0),
        ("endpoint_to_endpoint", endpoint_to_endpoint),
        ("involving_intermediate", involving_intermediate),
    ):
        count = int(mask.sum())
        stale_groups[name] = {
            "count": count,
            "advantages": advantage[mask].astype(np.float32, copy=True),
            "previous_probability_greater": int(
                (previous_probability[mask] > current_probability[mask]).sum()
            ),
            "argmax_previous": int(
                (argmax[transition_indices[mask]] == previous_targets[mask]).sum()
            ),
            "argmax_current": int(
                (argmax[transition_indices[mask]] == current_targets[mask]).sum()
            ),
        }
    return {"subsets": subset_output, "stale_groups": stale_groups}


def aggregate_transition_posterior(
    per_performance: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    subsets: dict[str, dict[str, float | int]] = {}
    total_positions = sum(
        int(item["subsets"]["transition"]["count"])
        + int(item["subsets"]["steady"]["count"])
        for item in per_performance
    )
    for name in ("transition", "steady"):
        count = sum(int(item["subsets"][name]["count"]) for item in per_performance)
        sums = {
            key: sum(float(item["subsets"][name][key]) for item in per_performance)
            for key in (
                "p_zero_sum",
                "p_full_sum",
                "p_intermediate_sum",
                "true_probability_sum",
                "rank_sum",
                "entropy_sum",
                "endpoint_margin_sum",
                "argmax_correct",
                "mean_absolute_error_sum",
                "median_absolute_error_sum",
            )
        }
        subsets[name] = {
            "count": count,
            "ratio": safe_ratio(count, total_positions),
            "mean_p_zero": safe_ratio(sums["p_zero_sum"], count),
            "mean_p_full": safe_ratio(sums["p_full_sum"], count),
            "mean_p_intermediate": safe_ratio(
                sums["p_intermediate_sum"], count
            ),
            "mean_true_probability": safe_ratio(
                sums["true_probability_sum"], count
            ),
            "mean_true_class_rank": safe_ratio(sums["rank_sum"], count),
            "mean_entropy": safe_ratio(sums["entropy_sum"], count),
            "mean_endpoint_margin": safe_ratio(
                sums["endpoint_margin_sum"], count
            ),
            "argmax_accuracy": safe_ratio(sums["argmax_correct"], count),
            "posterior_mean_mae": safe_ratio(
                sums["mean_absolute_error_sum"], count
            ),
            "posterior_median_mae": safe_ratio(
                sums["median_absolute_error_sum"], count
            ),
        }
    stale: dict[str, dict[str, float | int]] = {}
    for group in (
        "all",
        "upward",
        "downward",
        "endpoint_to_endpoint",
        "involving_intermediate",
    ):
        entries = [item["stale_groups"][group] for item in per_performance]
        count = sum(int(entry["count"]) for entry in entries)
        arrays = [entry["advantages"] for entry in entries if len(entry["advantages"])]
        advantage = np.concatenate(arrays) if arrays else np.asarray([], dtype=np.float32)
        stale[group] = {
            "count": count,
            "mean_stale_log_probability_advantage": float(advantage.mean())
            if count
            else float("nan"),
            "median_stale_log_probability_advantage": float(np.median(advantage))
            if count
            else float("nan"),
            "previous_probability_greater_fraction": safe_ratio(
                sum(int(entry["previous_probability_greater"]) for entry in entries),
                count,
            ),
            "argmax_previous_fraction": safe_ratio(
                sum(int(entry["argmax_previous"]) for entry in entries), count
            ),
            "argmax_current_fraction": safe_ratio(
                sum(int(entry["argmax_current"]) for entry in entries), count
            ),
        }
    return {"subsets": subsets, "stale_groups": stale}


def slot_posterior_diagnostic(
    probabilities: np.ndarray,
    targets: np.ndarray,
    ranks: np.ndarray,
    argmax_predictions: np.ndarray,
    mean_predictions: np.ndarray,
) -> list[dict[str, Any]]:
    posterior = np.asarray(probabilities, dtype=np.float32)
    truth = np.asarray(targets, dtype=np.int64)
    rank = np.asarray(ranks)
    argmax = np.asarray(argmax_predictions, dtype=np.int64)
    mean = np.asarray(mean_predictions, dtype=np.int64)
    group_mass = posterior_group_masses(posterior)
    outputs: list[dict[str, Any]] = []
    tiny = np.finfo(np.float32).tiny
    for slot in range(PEDAL_SLOTS):
        slot_truth = truth[:, slot]
        slot_argmax = argmax[:, slot]
        target_groups = target_group_ids(slot_truth)
        prediction_groups = target_group_ids(slot_argmax)
        intermediate_mask = target_groups == 1
        delta = np.diff(slot_truth)
        transition = delta != 0
        indices = np.flatnonzero(transition) + 1
        previous = slot_truth[indices - 1]
        current = slot_truth[indices]
        previous_probability = posterior[indices, slot, previous]
        current_probability = posterior[indices, slot, current]
        advantage = np.log(np.maximum(previous_probability, tiny)) - np.log(
            np.maximum(current_probability, tiny)
        )
        outputs.append(
            {
                "slot": slot + 1,
                "count": len(slot_truth),
                "target_group_counts": np.bincount(target_groups, minlength=3),
                "prediction_group_counts": np.bincount(
                    prediction_groups, minlength=3
                ),
                "p_intermediate_sum": float(
                    group_mass["p_intermediate"][:, slot].sum()
                ),
                "intermediate_rank_sum": int(rank[:, slot][intermediate_mask].sum()),
                "intermediate_count": int(intermediate_mask.sum()),
                "argmax_correct": int((slot_argmax == slot_truth).sum()),
                "mean_absolute_error_sum": int(
                    np.abs(mean[:, slot] - slot_truth).sum()
                ),
                "transition_count": len(indices),
                "transition_position_count": max(len(slot_truth) - 1, 0),
                "stale_advantages": advantage.astype(np.float32, copy=True),
                "stale_previous_greater": int(
                    (previous_probability > current_probability).sum()
                ),
                "stale_argmax_previous": int(
                    (slot_argmax[indices] == previous).sum()
                ),
                "stale_argmax_current": int(
                    (slot_argmax[indices] == current).sum()
                ),
            }
        )
    return outputs


def aggregate_slot_posterior(
    per_performance: Sequence[list[dict[str, Any]]]
) -> list[dict[str, float | int | list[float]]]:
    output = []
    for slot in range(PEDAL_SLOTS):
        entries = [item[slot] for item in per_performance]
        count = sum(int(entry["count"]) for entry in entries)
        target_counts = sum(
            (entry["target_group_counts"] for entry in entries),
            np.zeros(3, dtype=np.int64),
        )
        prediction_counts = sum(
            (entry["prediction_group_counts"] for entry in entries),
            np.zeros(3, dtype=np.int64),
        )
        intermediate_count = sum(int(entry["intermediate_count"]) for entry in entries)
        transition_count = sum(int(entry["transition_count"]) for entry in entries)
        position_count = sum(
            int(entry["transition_position_count"]) for entry in entries
        )
        advantages = [
            entry["stale_advantages"]
            for entry in entries
            if len(entry["stale_advantages"])
        ]
        advantage = np.concatenate(advantages) if advantages else np.asarray([])
        output.append(
            {
                "slot": slot + 1,
                "target_zero_ratio": safe_ratio(target_counts[0], count),
                "target_intermediate_ratio": safe_ratio(target_counts[1], count),
                "target_full_ratio": safe_ratio(target_counts[2], count),
                "argmax_zero_ratio": safe_ratio(prediction_counts[0], count),
                "argmax_intermediate_ratio": safe_ratio(prediction_counts[1], count),
                "argmax_full_ratio": safe_ratio(prediction_counts[2], count),
                "mean_p_intermediate": safe_ratio(
                    sum(float(entry["p_intermediate_sum"]) for entry in entries),
                    count,
                ),
                "intermediate_true_class_mean_rank": safe_ratio(
                    sum(int(entry["intermediate_rank_sum"]) for entry in entries),
                    intermediate_count,
                ),
                "argmax_accuracy": safe_ratio(
                    sum(int(entry["argmax_correct"]) for entry in entries), count
                ),
                "posterior_mean_mae": safe_ratio(
                    sum(int(entry["mean_absolute_error_sum"]) for entry in entries),
                    count,
                ),
                "transition_ratio": safe_ratio(transition_count, position_count),
                "transition_count": transition_count,
                "mean_stale_log_probability_advantage": float(advantage.mean())
                if transition_count
                else float("nan"),
                "stale_previous_probability_greater_fraction": safe_ratio(
                    sum(int(entry["stale_previous_greater"]) for entry in entries),
                    transition_count,
                ),
                "stale_argmax_previous_fraction": safe_ratio(
                    sum(int(entry["stale_argmax_previous"]) for entry in entries),
                    transition_count,
                ),
                "stale_argmax_current_fraction": safe_ratio(
                    sum(int(entry["stale_argmax_current"]) for entry in entries),
                    transition_count,
                ),
            }
        )
    return output


def initialize_class_statistics(train_support: np.ndarray) -> dict[str, Any]:
    support = np.asarray(train_support, dtype=np.int64)
    if support.shape != (PEDAL_NUM_CLASSES,):
        raise ValueError("train support must have 128 classes")
    return {
        "train_support": support.copy(),
        "validation_support": np.zeros(128, dtype=np.int64),
        "argmax_prediction_count": np.zeros(128, dtype=np.int64),
        "argmax_correct": np.zeros(128, dtype=np.int64),
        "probability_sum": np.zeros(128, dtype=np.float64),
        "probability_values": [[] for _ in range(128)],
        "rank_sum": np.zeros(128, dtype=np.int64),
        "top5": np.zeros(128, dtype=np.int64),
        "top10": np.zeros(128, dtype=np.int64),
        "mean_absolute_error_sum": np.zeros(128, dtype=np.int64),
        "best_intermediate_correct": np.zeros(128, dtype=np.int64),
    }


def update_class_statistics(
    statistics: dict[str, Any],
    probabilities: np.ndarray,
    targets: np.ndarray,
    ranks: np.ndarray,
    argmax_predictions: np.ndarray,
    mean_predictions: np.ndarray,
    best_intermediate_classes: np.ndarray,
) -> None:
    posterior = np.asarray(probabilities, dtype=np.float32).reshape(-1, 128)
    truth = np.asarray(targets, dtype=np.int64).reshape(-1)
    rank = np.asarray(ranks, dtype=np.int64).reshape(-1)
    argmax = np.asarray(argmax_predictions, dtype=np.int64).reshape(-1)
    mean = np.asarray(mean_predictions, dtype=np.int64).reshape(-1)
    best = np.asarray(best_intermediate_classes, dtype=np.int64).reshape(-1)
    statistics["validation_support"] += np.bincount(truth, minlength=128)
    statistics["argmax_prediction_count"] += np.bincount(argmax, minlength=128)
    for class_value in np.unique(truth):
        mask = truth == class_value
        class_int = int(class_value)
        probabilities_when_true = posterior[mask, class_int]
        statistics["argmax_correct"][class_int] += int(
            (argmax[mask] == class_int).sum()
        )
        statistics["probability_sum"][class_int] += float(
            probabilities_when_true.sum()
        )
        statistics["probability_values"][class_int].append(
            probabilities_when_true.copy()
        )
        statistics["rank_sum"][class_int] += int(rank[mask].sum())
        statistics["top5"][class_int] += int((rank[mask] <= 5).sum())
        statistics["top10"][class_int] += int((rank[mask] <= 10).sum())
        statistics["mean_absolute_error_sum"][class_int] += int(
            np.abs(mean[mask] - class_int).sum()
        )
        if 1 <= class_int <= 126:
            statistics["best_intermediate_correct"][class_int] += int(
                (best[mask] == class_int).sum()
            )


def finalize_class_statistics(
    statistics: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    total_validation = int(statistics["validation_support"].sum())
    for class_value in range(128):
        validation_support = int(statistics["validation_support"][class_value])
        arrays = statistics["probability_values"][class_value]
        values = np.concatenate(arrays) if arrays else np.asarray([], dtype=np.float32)
        rows.append(
            {
                "class": class_value,
                "train_support": int(statistics["train_support"][class_value]),
                "validation_support": validation_support,
                "validation_ratio": safe_ratio(validation_support, total_validation),
                "argmax_prediction_count": int(
                    statistics["argmax_prediction_count"][class_value]
                ),
                "argmax_recall": safe_ratio(
                    statistics["argmax_correct"][class_value], validation_support
                ),
                "mean_probability_when_true": safe_ratio(
                    statistics["probability_sum"][class_value], validation_support
                ),
                "median_probability_when_true": float(np.median(values))
                if validation_support
                else float("nan"),
                "mean_true_class_rank": safe_ratio(
                    statistics["rank_sum"][class_value], validation_support
                ),
                "top5_recall": safe_ratio(
                    statistics["top5"][class_value], validation_support
                ),
                "top10_recall": safe_ratio(
                    statistics["top10"][class_value], validation_support
                ),
                "posterior_mean_absolute_error_when_true": safe_ratio(
                    statistics["mean_absolute_error_sum"][class_value],
                    validation_support,
                ),
                "best_intermediate_recall_when_applicable": safe_ratio(
                    statistics["best_intermediate_correct"][class_value],
                    validation_support,
                )
                if 1 <= class_value <= 126
                else float("nan"),
            }
        )
    supported = [
        row
        for row in rows
        if row["train_support"] > 0 and row["validation_support"] > 0
    ]
    log_support = np.log(
        np.asarray([row["train_support"] for row in supported], dtype=np.float64)
    )
    probability = np.asarray(
        [row["mean_probability_when_true"] for row in supported], dtype=np.float64
    )
    recall = np.asarray([row["argmax_recall"] for row in supported], dtype=np.float64)
    summary = {
        "log_train_support_probability_correlation": descriptive_correlation(
            log_support, probability
        ),
        "log_train_support_argmax_recall_correlation": descriptive_correlation(
            log_support, recall
        ),
    }
    return rows, summary


def infer_complete_posterior(
    model: Stage2PedalEncoderModel,
    full_tokens: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    num_notes = len(full_tokens)
    starts = generate_window_starts(num_notes, WINDOW_NOTES, STRIDE_NOTES)
    window_logits: list[tuple[int, np.ndarray]] = []
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
            batch = stage2_pedal_collate_fn(samples)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention = batch["token_attention_mask"].to(device, non_blocking=True)
            note_mask = batch["note_mask"].to(device, non_blocking=True)
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
                window_logits.append(
                    (start, logits[batch_index, :length].copy())
                )
    mean_logits, contribution_count = average_overlapping_logits(
        num_notes, window_logits
    )
    reversed_logits, reversed_count = average_overlapping_logits(
        num_notes, list(reversed(window_logits))
    )
    if not np.array_equal(mean_logits, reversed_logits) or not np.array_equal(
        contribution_count, reversed_count
    ):
        raise RuntimeError("window order changed posterior reconstruction")
    posterior = softmax_float32(mean_logits)
    targets = (
        full_tokens[:, NON_PEDAL_FEATURES:].astype(np.int64)
        - PEDAL_TOKEN_OFFSET
    )
    loss = float(
        F.cross_entropy(
            torch.from_numpy(mean_logits.astype(np.float32)).reshape(-1, 128),
            torch.from_numpy(targets).reshape(-1),
        ).item()
    )
    if not math.isfinite(loss):
        raise FloatingPointError("non-finite reconstructed cross-entropy")
    return {
        "mean_logits": mean_logits,
        "probabilities": posterior,
        "targets": targets,
        "loss": loss,
        "contribution_count": contribution_count,
        "num_windows": len(starts),
    }


def summarize_posterior_group(
    values: dict[str, list[np.ndarray]], group_name: str
) -> dict[str, Any]:
    arrays = {
        name: np.concatenate(chunks) if chunks else np.asarray([], dtype=np.float32)
        for name, chunks in values.items()
    }
    count = len(arrays["p_intermediate"])
    if not count:
        raise ValueError(f"empty posterior target group {group_name}")
    p_intermediate = arrays["p_intermediate"]
    rank = arrays["true_rank"]
    true_probability = arrays["true_probability"]
    output = {
        "group": group_name,
        "target_count": count,
        "mean_p_zero": float(arrays["p_zero"].mean()),
        "median_p_zero": float(np.median(arrays["p_zero"])),
        "mean_p_full": float(arrays["p_full"].mean()),
        "median_p_full": float(np.median(arrays["p_full"])),
        "mean_p_intermediate": float(p_intermediate.mean()),
        "median_p_intermediate": float(np.median(p_intermediate)),
        "p_intermediate_p10": float(np.percentile(p_intermediate, 10)),
        "p_intermediate_p50": float(np.percentile(p_intermediate, 50)),
        "p_intermediate_p90": float(np.percentile(p_intermediate, 90)),
        "mean_best_intermediate_probability": float(
            arrays["best_intermediate_probability"].mean()
        ),
        "mean_endpoint_probability": float(arrays["endpoint_probability"].mean()),
        "mean_endpoint_margin": float(arrays["endpoint_margin"].mean()),
        "mean_intermediate_vs_endpoint_logit_margin": float(
            arrays["logit_margin"].mean()
        ),
        "fraction_p_intermediate_gt_0_10": float((p_intermediate > 0.10).mean()),
        "fraction_p_intermediate_gt_0_25": float((p_intermediate > 0.25).mean()),
        "fraction_p_intermediate_gt_0_50": float((p_intermediate > 0.50).mean()),
        "fraction_best_intermediate_beats_endpoints": float(
            arrays["best_intermediate_beats"].mean()
        ),
        "mean_posterior_entropy_nats": float(arrays["entropy"].mean()),
        "mean_true_class_rank": float(rank.mean()),
        "median_true_class_rank": float(np.median(rank)),
        "top1_recall": float((rank <= 1).mean()),
        "top2_recall": float((rank <= 2).mean()),
        "top5_recall": float((rank <= 5).mean()),
        "top10_recall": float((rank <= 10).mean()),
        "top20_recall": float((rank <= 20).mean()),
        "mean_true_class_probability": float(true_probability.mean()),
        "mean_negative_log_true_probability": float(
            (-np.log(np.maximum(true_probability, np.finfo(np.float32).tiny))).mean()
        ),
    }
    if group_name == "intermediate":
        best_class = arrays["best_intermediate_class"].astype(np.int64)
        target = arrays["target"].astype(np.int64)
        absolute_error = np.abs(best_class - target)
        intermediate_rank = arrays["intermediate_rank"]
        output.update(
            {
                "correct_class_is_best_intermediate_fraction": float(
                    (best_class == target).mean()
                ),
                "best_intermediate_class_mae": float(absolute_error.mean()),
                "best_intermediate_mae_true_top5": float(
                    absolute_error[rank <= 5].mean()
                )
                if np.any(rank <= 5)
                else float("nan"),
                "best_intermediate_mae_true_top10": float(
                    absolute_error[rank <= 10].mean()
                )
                if np.any(rank <= 10)
                else float("nan"),
                "mean_true_intermediate_rank": float(intermediate_rank.mean()),
                "median_true_intermediate_rank": float(
                    np.median(intermediate_rank)
                ),
                "true_intermediate_top5_recall": float(
                    (intermediate_rank <= 5).mean()
                ),
                "true_intermediate_top10_recall": float(
                    (intermediate_rank <= 10).mean()
                ),
            }
        )
    return output


def summarize_best_intermediate(
    per_performance: Sequence[dict[str, float | int]],
    intermediate_summary: dict[str, Any],
) -> dict[str, float | int]:
    count = sum(int(item["count"]) for item in per_performance)
    return {
        "count": count,
        "exact_accuracy": safe_ratio(
            sum(int(item["correct"]) for item in per_performance), count
        ),
        "mae": safe_ratio(
            sum(int(item["absolute_error_sum"]) for item in per_performance),
            count,
        ),
        **{
            f"tolerance_accuracy_{tolerance}": safe_ratio(
                sum(
                    int(item[f"tolerance_correct_{tolerance}"])
                    for item in per_performance
                ),
                count,
            )
            for tolerance in TOLERANCES
        },
        "mean_true_intermediate_rank": intermediate_summary[
            "mean_true_intermediate_rank"
        ],
        "median_true_intermediate_rank": intermediate_summary[
            "median_true_intermediate_rank"
        ],
        "true_intermediate_top5_recall": intermediate_summary[
            "true_intermediate_top5_recall"
        ],
        "true_intermediate_top10_recall": intermediate_summary[
            "true_intermediate_top10_recall"
        ],
    }


def markdown_table(headers: Sequence[Any], rows: Sequence[Sequence[Any]]) -> str:
    def clean(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    return "\n".join(
        [
            "| " + " | ".join(clean(value) for value in headers) + " |",
            "|" + "|".join("---" for _ in headers) + "|",
            *[
                "| " + " | ".join(clean(value) for value in row) + " |"
                for row in rows
            ],
        ]
    )


def format_metric(value: Any, digits: int = 6) -> str:
    number = float(value)
    return "NA" if not math.isfinite(number) else f"{number:.{digits}f}"


def render_csv(rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> str:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue()


def select_diagnosis(results: dict[str, Any]) -> dict[str, str]:
    intermediate = results["posterior_groups"]["intermediate"]
    decoders = results["decoders"]
    transition = results["transition_posterior"]
    intermediate_mass = float(intermediate["mean_p_intermediate"])
    intermediate_rank = float(intermediate["mean_true_class_rank"])
    mean_intermediate_improvement = float(
        decoders["argmax"]["intermediate_mae"]
        - decoders["posterior_mean"]["intermediate_mae"]
    )
    median_intermediate_improvement = float(
        decoders["argmax"]["intermediate_mae"]
        - decoders["posterior_median"]["intermediate_mae"]
    )
    best_decoder_improvement = max(
        mean_intermediate_improvement, median_intermediate_improvement
    )
    mean_overall_change = float(
        decoders["posterior_mean"]["pedal_value_mae"]
        - decoders["argmax"]["pedal_value_mae"]
    )
    median_overall_change = float(
        decoders["posterior_median"]["pedal_value_mae"]
        - decoders["argmax"]["pedal_value_mae"]
    )
    stale_fraction = float(
        transition["stale_groups"]["all"][
            "previous_probability_greater_fraction"
        ]
    )
    max_transition_recall = max(
        float(decoders[name]["transition_detection_recall"])
        for name in DECODER_NAMES
    )

    meaningful_mass = intermediate_mass >= 0.10
    decoder_substantial = best_decoder_improvement >= 5.0
    decoder_acceptable = min(mean_overall_change, median_overall_change) <= 2.0
    poor_rank = intermediate_rank > 20.0
    strong_stale = stale_fraction >= 0.60
    poor_transition_recall = max_transition_recall < 0.20

    if meaningful_mass and decoder_substantial and decoder_acceptable:
        code = "A"
        label = "DECODING/CALIBRATION BOTTLENECK"
        experiment = "decoding/calibration study without retraining"
    elif (not meaningful_mass) and poor_rank and not decoder_substantial:
        code = "B"
        label = "OBJECTIVE/CLASS-COLLAPSE BOTTLENECK"
        experiment = "encoder-only CE + distance-aware auxiliary loss"
    elif meaningful_mass and strong_stale and poor_transition_recall:
        code = "C"
        label = "TEMPORAL-CONDITIONING BOTTLENECK"
        experiment = "encoder plus lightweight temporal decoder"
    else:
        code = "D"
        label = "MIXED BOTTLENECK"
        experiment = "tightly controlled combination"
    evidence = (
        f"mean intermediate mass={intermediate_mass:.6f}, mean true-class "
        f"rank={intermediate_rank:.3f}, best deterministic intermediate-MAE "
        f"improvement={best_decoder_improvement:.3f}, posterior-mean overall-MAE "
        f"change={mean_overall_change:+.3f}, posterior-median overall-MAE "
        f"change={median_overall_change:+.3f}, stale-state fraction="
        f"{stale_fraction:.6f}, and best decoder transition recall="
        f"{max_transition_recall:.6f}. For this diagnostic, a substantial "
        "decoder recovery is at least 5 pedal-value MAE units with no more than "
        "2 units of overall-MAE degradation; these are practical descriptive "
        "thresholds, not significance tests."
    )
    return {
        "code": code,
        "label": label,
        "experiment": experiment,
        "evidence": evidence,
    }


def render_report(results: dict[str, Any]) -> str:
    checkpoint = results["checkpoint"]
    verification = results["verification"]
    timing = results["timing"]
    groups = results["posterior_groups"]
    ranks = results["posterior_groups"]
    decoders = results["decoders"]
    mixture = results["endpoint_mixture"]
    transition = results["transition_posterior"]
    slots = results["slots"]
    class_rows = results["class_rows"]
    class_summary = results["class_summary"]
    coarse = results["coarse"]
    diagnosis = select_diagnosis(results)
    results["diagnosis"] = diagnosis

    lines = [
        "# Experiment 1 v0 Validation Posterior Diagnostic",
        "",
        "## 1. Scope",
        "",
        "This is a validation-only posterior diagnostic. It uses Oracle human "
        "non-pedal performance inputs, does not read test targets, does not "
        "retrain or modify the model, and is not final model evaluation or "
        "end-to-end Stage 1 inference.",
        "",
        "## 2. Checkpoint and environment",
        "",
        f"- Best checkpoint: `{results['checkpoint_path']}`",
        f"- Best epoch: {checkpoint['best_epoch']}",
        f"- Recorded best validation loss: {checkpoint['best_validation_loss']:.10f}",
        f"- Exact state load: missing keys={checkpoint['missing_keys']}, "
        f"unexpected keys={checkpoint['unexpected_keys']}",
        f"- Architecture: {checkpoint['architecture']}",
        "- Model state: `eval()`, FP16 autocast forward, `inference_mode()`, "
        "no gradients, no optimizer, and no parameter-version changes.",
        f"- Project commit: `{results['project_commit']}`",
        f"- PT commit: `{results['pt_commit']}`",
        f"- GPU: host GPU 1 UUID `{results['gpu']['uuid']}` -> container GPU 0 "
        f"-> PyTorch `cuda:0` ({results['gpu']['name']})",
        f"- Batch size: {results['batch_size']}",
        f"- Preparation: {timing['preparation_seconds']:.3f} s; model load: "
        f"{timing['model_load_seconds']:.3f} s; inference/statistics: "
        f"{timing['inference_seconds']:.3f} s; total: "
        f"{timing['total_seconds']:.3f} s",
        f"- Peak allocated GPU memory: {timing['peak_gpu_memory_bytes']} bytes "
        f"({timing['peak_gpu_memory_bytes'] / 1024**3:.3f} GiB)",
        "",
        "Checkpoint configuration:",
        "",
        markdown_table(
            ["Key", "Value"],
            [
                [key, checkpoint["configuration"][key]]
                for key in sorted(checkpoint["configuration"])
            ],
        ),
        "",
        "## 3. Validation-set verification",
        "",
        f"- Validation performances: {verification['performance_count']} (expected 71)",
        f"- Unique pieces: {verification['piece_count']} (expected 19)",
        f"- Complete notes: {verification['note_count']}",
        f"- Complete pedal targets: {verification['target_count']}",
        f"- Deterministic windows: {verification['window_count']} (expected 1,078)",
        f"- Unique paths: {verification['unique_path_count']}; all MIDI files "
        "existed and tokenized successfully.",
        "- Only validation rows were selected. Train tokens were read only to "
        "compute per-class train support; no test MIDI or test target was read.",
        f"- Contribution range: {verification['min_contributions']}–"
        f"{verification['max_contributions']}; first and last notes were covered, "
        "and reversing window order produced identical mean logits.",
        f"- Final statistics count each of the {verification['note_count']} original "
        "notes exactly once.",
        "",
        "## 4. Argmax reproduction",
        "",
        markdown_table(
            ["Loss", "Token accuracy", "Exact-note accuracy", "P1", "P2", "P3", "P4", "MAE"],
            [[
                format_metric(decoders["argmax"]["loss"]),
                format_metric(decoders["argmax"]["pedal_token_accuracy"]),
                format_metric(decoders["argmax"]["exact_note_accuracy"]),
                format_metric(decoders["argmax"]["pedal1_accuracy"]),
                format_metric(decoders["argmax"]["pedal2_accuracy"]),
                format_metric(decoders["argmax"]["pedal3_accuracy"]),
                format_metric(decoders["argmax"]["pedal4_accuracy"]),
                format_metric(decoders["argmax"]["pedal_value_mae"]),
            ]],
        ),
        "",
        "These finite full-performance metrics may differ slightly from the "
        "training-time validation metrics because training counted overlapping "
        "windows independently, whereas this diagnostic averages logits and "
        "counts each complete note once.",
        "",
        "## 5. Endpoint versus intermediate probability mass",
        "",
        "Softmax is applied in float32 only after mean-logit reconstruction. "
        "Entropy is natural-log entropy in nats.",
        "",
        markdown_table(
            ["Target", "N", "Mean p0", "Median p0", "Mean p-int", "Median p-int", "p-int P10", "P50", "P90", "Mean p127", "Best-int prob.", "Endpoint prob.", "Endpoint margin", "p-int>0.25", "Best int beats endpoints", "Entropy"],
            [
                [
                    name.upper(),
                    groups[name]["target_count"],
                    format_metric(groups[name]["mean_p_zero"]),
                    format_metric(groups[name]["median_p_zero"]),
                    format_metric(groups[name]["mean_p_intermediate"]),
                    format_metric(groups[name]["median_p_intermediate"]),
                    format_metric(groups[name]["p_intermediate_p10"]),
                    format_metric(groups[name]["p_intermediate_p50"]),
                    format_metric(groups[name]["p_intermediate_p90"]),
                    format_metric(groups[name]["mean_p_full"]),
                    format_metric(groups[name]["mean_best_intermediate_probability"]),
                    format_metric(groups[name]["mean_endpoint_probability"]),
                    format_metric(groups[name]["mean_endpoint_margin"]),
                    format_metric(groups[name]["fraction_p_intermediate_gt_0_25"]),
                    format_metric(groups[name]["fraction_best_intermediate_beats_endpoints"]),
                    format_metric(groups[name]["mean_posterior_entropy_nats"]),
                ]
                for name in GROUP_NAMES
            ],
        ),
        "",
        "Intermediate-mass thresholds by target group:",
        "",
        markdown_table(
            ["Target", "p-int > 0.10", "p-int > 0.25", "p-int > 0.50", "Mean int-vs-endpoint logit margin"],
            [
                [
                    name.upper(),
                    format_metric(groups[name]["fraction_p_intermediate_gt_0_10"]),
                    format_metric(groups[name]["fraction_p_intermediate_gt_0_25"]),
                    format_metric(groups[name]["fraction_p_intermediate_gt_0_50"]),
                    format_metric(groups[name]["mean_intermediate_vs_endpoint_logit_margin"]),
                ]
                for name in GROUP_NAMES
            ],
        ),
        "",
        f"On intermediate targets, mean p-intermediate is "
        f"{groups['intermediate']['mean_p_intermediate']:.6f}; under the "
        "predeclared descriptive 0.10 threshold used in Section 13, this is "
        f"{'meaningful/nontrivial' if groups['intermediate']['mean_p_intermediate'] >= 0.10 else 'low/effectively absent'}. "
        f"The best intermediate class beats both endpoint classes for "
        f"{groups['intermediate']['fraction_best_intermediate_beats_endpoints']:.6f} "
        "of intermediate targets; the mass and logit-margin tables show whether "
        "remaining support sits below endpoint modes.",
        "",
        "## 6. True-class rank and coarse three-way classification",
        "",
        markdown_table(
            ["Target", "Mean rank", "Median rank", "Top1", "Top2", "Top5", "Top10", "Top20", "Mean true p", "Mean NLL"],
            [
                [
                    name.upper(),
                    format_metric(ranks[name]["mean_true_class_rank"]),
                    format_metric(ranks[name]["median_true_class_rank"]),
                    format_metric(ranks[name]["top1_recall"]),
                    format_metric(ranks[name]["top2_recall"]),
                    format_metric(ranks[name]["top5_recall"]),
                    format_metric(ranks[name]["top10_recall"]),
                    format_metric(ranks[name]["top20_recall"]),
                    format_metric(ranks[name]["mean_true_class_probability"]),
                    format_metric(ranks[name]["mean_negative_log_true_probability"]),
                ]
                for name in ("overall", *GROUP_NAMES)
            ],
        ),
        "",
        "Intermediate-only ordering:",
        "",
        markdown_table(
            ["Correct is best intermediate", "Best-intermediate MAE", "MAE if true top5", "MAE if true top10", "Mean rank among intermediates", "Intermediate top5", "Intermediate top10"],
            [[
                format_metric(groups["intermediate"]["correct_class_is_best_intermediate_fraction"]),
                format_metric(groups["intermediate"]["best_intermediate_class_mae"]),
                format_metric(groups["intermediate"]["best_intermediate_mae_true_top5"]),
                format_metric(groups["intermediate"]["best_intermediate_mae_true_top10"]),
                format_metric(groups["intermediate"]["mean_true_intermediate_rank"]),
                format_metric(groups["intermediate"]["true_intermediate_top5_recall"]),
                format_metric(groups["intermediate"]["true_intermediate_top10_recall"]),
            ]],
        ),
        "",
        "Posterior-mass group prediction confusion (rows=true; columns=ZERO, INTERMEDIATE, FULL):",
        "",
        markdown_table(
            ["True group", "Pred ZERO", "Pred INTERMEDIATE", "Pred FULL"],
            [
                [GROUP_NAMES[index].upper(), *coarse["posterior_mass"]["confusion_matrix"][index].tolist()]
                for index in range(3)
            ],
        ),
        "",
        "Collapsed 128-class argmax confusion (rows=true; columns=ZERO, INTERMEDIATE, FULL):",
        "",
        markdown_table(
            ["True group", "Pred ZERO", "Pred INTERMEDIATE", "Pred FULL"],
            [
                [GROUP_NAMES[index].upper(), *coarse["collapsed_argmax"]["confusion_matrix"][index].tolist()]
                for index in range(3)
            ],
        ),
        "",
        markdown_table(
            ["Grouping method", "Accuracy", "Macro F1", "Balanced accuracy", "ZERO F1", "INTERMEDIATE F1", "FULL F1"],
            [
                [
                    label,
                    format_metric(metrics["group_accuracy"]),
                    format_metric(metrics["macro_f1"]),
                    format_metric(metrics["balanced_accuracy"]),
                    *[format_metric(value) for value in metrics["f1"]],
                ]
                for label, metrics in (
                    ("Posterior summed mass", coarse["posterior_mass"]),
                    ("Collapsed 128-class argmax", coarse["collapsed_argmax"]),
                )
            ],
        ),
        "",
        markdown_table(
            ["Grouping method", "Group", "Precision", "Recall", "F1"],
            [
                [
                    label,
                    GROUP_NAMES[group_index].upper(),
                    format_metric(metrics["precision"][group_index]),
                    format_metric(metrics["recall"][group_index]),
                    format_metric(metrics["f1"][group_index]),
                ]
                for label, metrics in (
                    ("Posterior summed mass", coarse["posterior_mass"]),
                    ("Collapsed 128-class argmax", coarse["collapsed_argmax"]),
                )
                for group_index in range(3)
            ],
        ),
        "",
        "The three-way result is a broad-region diagnostic, not a replacement "
        "for the 128-class prediction task.",
        "",
        "## 7. Alternative decoding comparison",
        "",
        "Posterior mean uses float32 expectation followed by NumPy `rint` "
        "(round-to-nearest, ties-to-even) and clipping to [0,127]. Posterior "
        "median is the smallest class whose cumulative mass is at least 0.5. A "
        "posterior mean near 64 can reflect a 0/127 mixture rather than learned "
        "half-pedal support.",
        "",
        markdown_table(
            ["Decoder", "Exact acc.", "Exact note", "MAE", "RMSE", "Tol±5", "Tol±10", "Tol±20", "Int. exact", "Int. MAE", "Int. tol±10", "Pred int.", "Int. endpoint collapse", "Transition acc.", "Transition recall", "Transition F1", "Delta MAE"],
            [
                [
                    DECODER_LABELS[name],
                    format_metric(decoders[name]["pedal_token_accuracy"]),
                    format_metric(decoders[name]["exact_note_accuracy"]),
                    format_metric(decoders[name]["pedal_value_mae"]),
                    format_metric(decoders[name]["rmse"]),
                    format_metric(decoders[name]["tolerance_accuracy_5"]),
                    format_metric(decoders[name]["tolerance_accuracy_10"]),
                    format_metric(decoders[name]["tolerance_accuracy_20"]),
                    format_metric(decoders[name]["intermediate_exact_accuracy"]),
                    format_metric(decoders[name]["intermediate_mae"]),
                    format_metric(decoders[name]["intermediate_tolerance_accuracy_10"]),
                    format_metric(decoders[name]["predicted_intermediate_ratio"]),
                    format_metric(decoders[name]["endpoint_collapse_ratio"]),
                    format_metric(decoders[name]["transition_position_exact_accuracy"]),
                    format_metric(decoders[name]["transition_detection_recall"]),
                    format_metric(decoders[name]["transition_detection_f1"]),
                    format_metric(decoders[name]["delta_mae"]),
                ]
                for name in DECODER_NAMES
            ],
        ),
        "",
        "Additional decoder details:",
        "",
        markdown_table(
            ["Decoder", "P1", "P2", "P3", "P4", "Int tol±5", "Int tol±20", "Pred zero", "Pred full", "Steady acc.", "Steady MAE", "Transition MAE", "Detection precision"],
            [
                [
                    DECODER_LABELS[name],
                    format_metric(decoders[name]["pedal1_accuracy"]),
                    format_metric(decoders[name]["pedal2_accuracy"]),
                    format_metric(decoders[name]["pedal3_accuracy"]),
                    format_metric(decoders[name]["pedal4_accuracy"]),
                    format_metric(decoders[name]["intermediate_tolerance_accuracy_5"]),
                    format_metric(decoders[name]["intermediate_tolerance_accuracy_20"]),
                    format_metric(decoders[name]["predicted_zero_ratio"]),
                    format_metric(decoders[name]["predicted_full_ratio"]),
                    format_metric(decoders[name]["steady_position_exact_accuracy"]),
                    format_metric(decoders[name]["steady_position_mae"]),
                    format_metric(decoders[name]["transition_position_mae"]),
                    format_metric(decoders[name]["transition_detection_precision"]),
                ]
                for name in DECODER_NAMES
            ],
        ),
        "",
        "Best-intermediate constrained diagnostic (not deployable because every "
        "prediction is forced into 1–126):",
        "",
        markdown_table(
            ["Intermediate exact", "Intermediate MAE", "Tol±5", "Tol±10", "Tol±20", "Mean true intermediate rank", "Top5", "Top10"],
            [[
                format_metric(results["best_intermediate"]["exact_accuracy"]),
                format_metric(results["best_intermediate"]["mae"]),
                format_metric(results["best_intermediate"]["tolerance_accuracy_5"]),
                format_metric(results["best_intermediate"]["tolerance_accuracy_10"]),
                format_metric(results["best_intermediate"]["tolerance_accuracy_20"]),
                format_metric(results["best_intermediate"]["mean_true_intermediate_rank"]),
                format_metric(results["best_intermediate"]["true_intermediate_top5_recall"]),
                format_metric(results["best_intermediate"]["true_intermediate_top10_recall"]),
            ]],
        ),
        "",
        "## 8. Genuine intermediate support versus endpoint mixture",
        "",
        "Fixed categories are: locally supported if local mass ±10 is at least "
        "0.25; endpoint-mixture dominated if endpoint mass is at least 0.75 and "
        "local mass ±10 is below 0.25; diffuse/other otherwise.",
        "",
        markdown_table(
            ["Scope", "Mean-int predictions", "Ratio", "Locally supported N", "Locally supported ratio", "Local MAE", "Endpoint-mixture N", "Endpoint-mixture ratio", "Mixture MAE", "Diffuse N", "Diffuse ratio", "Diffuse MAE"],
            [
                [
                    label,
                    mixture[scope]["intermediate_prediction_count"],
                    format_metric(mixture[scope]["intermediate_prediction_ratio"]),
                    mixture[scope]["locally_supported_count"],
                    format_metric(mixture[scope]["locally_supported_ratio"]),
                    format_metric(mixture[scope]["locally_supported_mae"]),
                    mixture[scope]["endpoint_mixture_dominated_count"],
                    format_metric(mixture[scope]["endpoint_mixture_dominated_ratio"]),
                    format_metric(mixture[scope]["endpoint_mixture_dominated_mae"]),
                    mixture[scope]["diffuse_other_count"],
                    format_metric(mixture[scope]["diffuse_other_ratio"]),
                    format_metric(mixture[scope]["diffuse_other_mae"]),
                ]
                for scope, label in (
                    ("all", "All targets"),
                    ("true_intermediate", "True intermediate targets"),
                )
            ],
        ),
        "",
        "These fixed categories diagnose posterior shape; they are not tuned "
        "decoder hyperparameters.",
        "",
        "## 9. Transition-conditioned analysis",
        "",
        "Sequences are flattened within each performance only; boundaries are "
        "never joined.",
        "",
        markdown_table(
            ["Subset", "N", "Ratio", "p0", "p-int", "p127", "True p", "True rank", "Entropy", "Endpoint margin", "Argmax acc.", "Mean MAE", "Median MAE"],
            [
                [
                    name.title(),
                    transition["subsets"][name]["count"],
                    format_metric(transition["subsets"][name]["ratio"]),
                    format_metric(transition["subsets"][name]["mean_p_zero"]),
                    format_metric(transition["subsets"][name]["mean_p_intermediate"]),
                    format_metric(transition["subsets"][name]["mean_p_full"]),
                    format_metric(transition["subsets"][name]["mean_true_probability"]),
                    format_metric(transition["subsets"][name]["mean_true_class_rank"]),
                    format_metric(transition["subsets"][name]["mean_entropy"]),
                    format_metric(transition["subsets"][name]["mean_endpoint_margin"]),
                    format_metric(transition["subsets"][name]["argmax_accuracy"]),
                    format_metric(transition["subsets"][name]["posterior_mean_mae"]),
                    format_metric(transition["subsets"][name]["posterior_median_mae"]),
                ]
                for name in ("transition", "steady")
            ],
        ),
        "",
        "Stale-state preference at true transitions:",
        "",
        markdown_table(
            ["Transition group", "N", "Mean log p(prev)-log p(cur)", "Median", "p(prev)>p(cur)", "Argmax=previous", "Argmax=current"],
            [
                [
                    name.replace("_", " ").title(),
                    transition["stale_groups"][name]["count"],
                    format_metric(transition["stale_groups"][name]["mean_stale_log_probability_advantage"]),
                    format_metric(transition["stale_groups"][name]["median_stale_log_probability_advantage"]),
                    format_metric(transition["stale_groups"][name]["previous_probability_greater_fraction"]),
                    format_metric(transition["stale_groups"][name]["argmax_previous_fraction"]),
                    format_metric(transition["stale_groups"][name]["argmax_current_fraction"]),
                ]
                for name in (
                    "all",
                    "upward",
                    "downward",
                    "endpoint_to_endpoint",
                    "involving_intermediate",
                )
            ],
        ),
        "",
        "## 10. Slot-conditioned analysis",
        "",
        "Slot transitions compare each Pedal slot across consecutive notes.",
        "",
        markdown_table(
            ["Slot", "Target 0", "Target int.", "Target 127", "Argmax 0", "Argmax int.", "Argmax 127", "Mean p-int", "Int true rank", "Argmax acc.", "Mean MAE", "Transition ratio", "Stale p(prev)>p(cur)", "Stale advantage", "Argmax=previous", "Argmax=current"],
            [
                [
                    f"Pedal{item['slot']}",
                    format_metric(item["target_zero_ratio"]),
                    format_metric(item["target_intermediate_ratio"]),
                    format_metric(item["target_full_ratio"]),
                    format_metric(item["argmax_zero_ratio"]),
                    format_metric(item["argmax_intermediate_ratio"]),
                    format_metric(item["argmax_full_ratio"]),
                    format_metric(item["mean_p_intermediate"]),
                    format_metric(item["intermediate_true_class_mean_rank"]),
                    format_metric(item["argmax_accuracy"]),
                    format_metric(item["posterior_mean_mae"]),
                    format_metric(item["transition_ratio"]),
                    format_metric(item["stale_previous_probability_greater_fraction"]),
                    format_metric(item["mean_stale_log_probability_advantage"]),
                    format_metric(item["stale_argmax_previous_fraction"]),
                    format_metric(item["stale_argmax_current_fraction"]),
                ]
                for item in slots
            ],
        ),
        "",
        "## 11. Class-frequency analysis",
        "",
    ]

    highest_validation = sorted(
        class_rows, key=lambda row: (-row["validation_support"], row["class"])
    )[:10]
    intermediate_rows = [row for row in class_rows if 1 <= row["class"] <= 126]
    highest_train_intermediate = sorted(
        intermediate_rows, key=lambda row: (-row["train_support"], row["class"])
    )[:10]
    supported_intermediate = [
        row for row in intermediate_rows if row["validation_support"] > 0
    ]
    highest_recall_intermediate = sorted(
        supported_intermediate,
        key=lambda row: (-row["argmax_recall"], row["class"]),
    )[:10]
    never_argmax = [
        row["class"]
        for row in intermediate_rows
        if row["argmax_prediction_count"] == 0
    ]
    lines.extend(
        [
            markdown_table(
                ["Summary", "Classes (class:support or class:recall)"],
                [
                    [
                        "Highest validation support",
                        ", ".join(
                            f"{row['class']}:{row['validation_support']}"
                            for row in highest_validation
                        ),
                    ],
                    [
                        "Highest train support among intermediate classes",
                        ", ".join(
                            f"{row['class']}:{row['train_support']}"
                            for row in highest_train_intermediate
                        ),
                    ],
                    [
                        "Highest intermediate argmax recall",
                        ", ".join(
                            f"{row['class']}:{format_metric(row['argmax_recall'], 4)}"
                            for row in highest_recall_intermediate
                        ),
                    ],
                    [
                        "Intermediate classes never selected by argmax",
                        f"{len(never_argmax)} classes: "
                        + ", ".join(str(value) for value in never_argmax),
                    ],
                ],
            ),
            "",
            f"Natural-log train support versus mean true-class probability "
            f"correlation: {class_summary['log_train_support_probability_correlation']:.6f}. "
            f"Natural-log train support versus argmax recall correlation: "
            f"{class_summary['log_train_support_argmax_recall_correlation']:.6f}. "
            "These correlations are descriptive, use classes supported in both "
            "train and validation, and do not establish causality or significance.",
            "`nan` appears in the per-class CSV only when validation support is "
            "zero or best-intermediate recall is inapplicable to endpoint classes.",
            "",
            "## 12. Performance-level variation",
            "",
        ]
    )
    performance_rows = results["performance_rows"]
    variation_metrics = (
        ("Argmax accuracy", "argmax_accuracy"),
        ("Argmax MAE", "argmax_mae"),
        ("Posterior-mean MAE", "posterior_mean_mae"),
        ("Posterior-median MAE", "posterior_median_mae"),
        (
            "Intermediate p-mass on intermediate targets",
            "mean_intermediate_probability_on_intermediate_targets",
        ),
        ("Intermediate true-class rank", "intermediate_true_class_mean_rank"),
        ("Transition stale-state fraction", "transition_stale_state_fraction"),
    )
    lines.extend(
        [
            markdown_table(
                ["Metric", "10th percentile", "Median", "90th percentile"],
                [
                    [
                        label,
                        format_metric(
                            np.nanpercentile(
                                [float(row[key]) for row in performance_rows], 10
                            )
                        ),
                        format_metric(
                            np.nanpercentile(
                                [float(row[key]) for row in performance_rows], 50
                            )
                        ),
                        format_metric(
                            np.nanpercentile(
                                [float(row[key]) for row in performance_rows], 90
                            )
                        ),
                    ]
                    for label, key in variation_metrics
                ],
            ),
            "",
            "These are descriptive validation-performance differences; no "
            "statistical significance is inferred.",
            "",
            "## 13. Interpretation and decision",
            "",
            f"Primary diagnosis: **{diagnosis['code']}. {diagnosis['label']}**.",
            "",
            diagnosis["evidence"],
            "",
            "## 14. Recommended next experiment",
            "",
            f"Run exactly one next primary experiment: **{diagnosis['experiment']}**. "
            "Define its architecture/loss and success criteria using train and "
            "validation only; do not tune it against the fixed test result. This "
            "task does not launch that experiment.",
            "",
            "## 15. Limitations",
            "",
            "- This is a validation-only model-selection diagnostic.",
            "- Posterior mean can reflect endpoint uncertainty rather than learned intermediate depth.",
            "- Targets use the four-point PT pedal tokenizer.",
            "- No raw-CC64 timing reconstruction was performed.",
            "- No end-to-end Stage 1 input was evaluated.",
            "- No listening test was performed.",
            "- No model retraining or calibration was performed.",
            "",
        ]
    )
    return "\n".join(lines)


def evaluate(configuration: dict[str, Any]) -> dict[str, Any]:
    total_start = time.perf_counter()
    asap_root = Path(configuration["asap_root"]).resolve()
    split_csv = Path(configuration["split_csv"]).resolve()
    checkpoint_path = Path(configuration["best_checkpoint"]).resolve()
    output_dir = Path(configuration["output_dir"]).resolve()
    smoke_test = bool(configuration["smoke_test"])
    limit = int(configuration["limit_performances"]) if smoke_test else None
    if not smoke_test and output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    if smoke_test and limit != 2:
        raise ValueError("smoke test must evaluate exactly two performances")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one CUDA device must be visible")
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

    with split_csv.open(newline="", encoding="utf-8") as handle:
        validation_rows = [
            row for row in csv.DictReader(handle) if row["split"] == "validation"
        ]
    if len(validation_rows) != EXPECTED_VALIDATION_PERFORMANCES:
        raise RuntimeError("validation split must contain 71 performances")
    if len({row["piece_id"] for row in validation_rows}) != EXPECTED_VALIDATION_PIECES:
        raise RuntimeError("validation split must contain 19 unique pieces")
    if len({row["performance_path"] for row in validation_rows}) != len(
        validation_rows
    ):
        raise RuntimeError("duplicate validation performance path")
    if any(row["split"] != "validation" for row in validation_rows):
        raise RuntimeError("non-validation row selected")

    preparation_start = time.perf_counter()
    train_dataset = Stage2PedalDataset(
        asap_root,
        split_csv,
        "train",
        window_notes=WINDOW_NOTES,
        stride_notes=STRIDE_NOTES,
        cache_mode="preload",
    )
    train_support = np.zeros(128, dtype=np.int64)
    for row in train_dataset.performances:
        tokens = train_dataset._token_cache[row["performance_path"]]
        targets = (
            tokens[:, NON_PEDAL_FEATURES:].astype(np.int64)
            - PEDAL_TOKEN_OFFSET
        )
        train_support += np.bincount(targets.reshape(-1), minlength=128)
    del train_dataset
    validation_dataset = Stage2PedalDataset(
        asap_root,
        split_csv,
        "validation",
        window_notes=WINDOW_NOTES,
        stride_notes=STRIDE_NOTES,
        cache_mode="preload",
    )
    if validation_dataset.performance_count != EXPECTED_VALIDATION_PERFORMANCES:
        raise RuntimeError("Dataset validation performance count mismatch")
    if validation_dataset.window_count != EXPECTED_VALIDATION_WINDOWS:
        raise RuntimeError("Dataset validation window count mismatch")
    if [row["performance_path"] for row in validation_dataset.performances] != [
        row["performance_path"] for row in validation_rows
    ]:
        raise RuntimeError("validation metadata order changed")
    preparation_seconds = time.perf_counter() - preparation_start

    model_start = time.perf_counter()
    model, checkpoint = _load_model(checkpoint_path, device)
    parameter_versions = {
        name: parameter._version for name, parameter in model.named_parameters()
    }
    model_load_seconds = time.perf_counter() - model_start

    selected_rows = validation_dataset.performances[:limit]
    posterior_group_chunks: dict[str, dict[str, list[np.ndarray]]] = {
        name: defaultdict(list) for name in ("overall", *GROUP_NAMES)
    }
    decoder_performance: dict[str, list[dict[str, Any]]] = {
        name: [] for name in DECODER_NAMES
    }
    mixture_performance: list[dict[str, dict[str, Any]]] = []
    transition_performance: list[dict[str, Any]] = []
    slot_performance: list[list[dict[str, Any]]] = []
    best_intermediate_performance: list[dict[str, int]] = []
    posterior_confusion = np.zeros((3, 3), dtype=np.int64)
    collapsed_confusion = np.zeros((3, 3), dtype=np.int64)
    class_statistics = initialize_class_statistics(train_support)
    performance_rows: list[dict[str, Any]] = []
    total_windows = 0
    min_contributions = math.inf
    max_contributions = 0

    inference_start = time.perf_counter()
    torch.cuda.synchronize(device)
    for performance_number, row in enumerate(selected_rows, start=1):
        tokens = validation_dataset._token_cache[row["performance_path"]]
        inference = infer_complete_posterior(
            model,
            tokens,
            device,
            batch_size=int(configuration["batch_size"]),
        )
        probabilities = inference["probabilities"]
        mean_logits = inference["mean_logits"]
        targets = inference["targets"]
        counts = inference["contribution_count"]
        if len(counts) != len(tokens) or int(counts.min()) < 1:
            raise RuntimeError("complete note coverage failed")
        if not np.isfinite(probabilities).all():
            raise FloatingPointError("non-finite posterior values")
        if not np.allclose(probabilities.sum(axis=-1), 1.0, atol=2e-5):
            raise RuntimeError("posterior normalization failed")

        group_mass = posterior_group_masses(probabilities)
        argmax_prediction = probabilities.argmax(axis=-1).astype(np.int64)
        mean_prediction, _ = posterior_mean_decode(probabilities)
        median_prediction = posterior_median_decode(probabilities)
        best_intermediate = group_mass["best_intermediate_class"]
        ranks = true_class_ranks(probabilities, targets)
        entropy = posterior_entropy(probabilities)
        true_probability = np.take_along_axis(
            probabilities, targets[..., None], axis=-1
        )[..., 0]
        logit_margin = mean_logits[..., 1:127].max(axis=-1) - np.maximum(
            mean_logits[..., 0], mean_logits[..., 127]
        )
        target_groups = target_group_ids(targets)
        posterior_group_prediction = np.stack(
            (
                group_mass["p_zero"],
                group_mass["p_intermediate"],
                group_mass["p_full"],
            ),
            axis=-1,
        ).argmax(axis=-1)
        collapsed_argmax = target_group_ids(argmax_prediction)
        posterior_confusion += coarse_confusion(
            target_groups, posterior_group_prediction
        )["confusion_matrix"]
        collapsed_confusion += coarse_confusion(
            target_groups, collapsed_argmax
        )["confusion_matrix"]

        common_values = {
            "p_zero": group_mass["p_zero"],
            "p_full": group_mass["p_full"],
            "p_intermediate": group_mass["p_intermediate"],
            "best_intermediate_probability": group_mass[
                "best_intermediate_probability"
            ],
            "best_intermediate_class": best_intermediate,
            "endpoint_probability": group_mass["endpoint_probability"],
            "endpoint_margin": group_mass["endpoint_margin"],
            "best_intermediate_beats": group_mass[
                "best_intermediate_beats_endpoints"
            ],
            "logit_margin": logit_margin,
            "entropy": entropy,
            "true_rank": ranks,
            "true_probability": true_probability,
            "target": targets,
        }
        for group_name, group_id in (("overall", None), ("zero", 0), ("intermediate", 1), ("full", 2)):
            mask = np.ones(targets.shape, dtype=bool) if group_id is None else target_groups == group_id
            for metric_name, metric_values in common_values.items():
                posterior_group_chunks[group_name][metric_name].append(
                    np.asarray(metric_values[mask]).copy()
                )
            if group_name == "intermediate":
                posterior_group_chunks[group_name]["intermediate_rank"].append(
                    intermediate_true_ranks(
                        probabilities[mask], targets[mask]
                    )
                )

        decoder_predictions = {
            "argmax": argmax_prediction,
            "posterior_mean": mean_prediction,
            "posterior_median": median_prediction,
        }
        performance_decoder_metrics: dict[str, dict[str, Any]] = {}
        for decoder_name, prediction in decoder_predictions.items():
            metrics = decoder_metrics(prediction, targets)
            if decoder_name == "argmax":
                metrics["standard"]["loss"] = inference["loss"]
                metrics["standard"]["_loss_sum"] = (
                    inference["loss"]
                    * int(metrics["standard"]["valid_target_count"])
                )
            decoder_performance[decoder_name].append(metrics)
            performance_decoder_metrics[decoder_name] = metrics

        intermediate_mask = target_groups == 1
        best_error = np.abs(best_intermediate[intermediate_mask] - targets[intermediate_mask])
        best_intermediate_performance.append(
            {
                "count": len(best_error),
                "correct": int((best_error == 0).sum()),
                "absolute_error_sum": int(best_error.sum()),
                **{
                    f"tolerance_correct_{tolerance}": int(
                        (best_error <= tolerance).sum()
                    )
                    for tolerance in TOLERANCES
                },
            }
        )
        mixture_performance.append(
            endpoint_mixture_diagnostic(probabilities, mean_prediction, targets)
        )
        transition_detail = transition_posterior_diagnostic(
            probabilities,
            targets,
            ranks,
            argmax_prediction,
            mean_prediction,
            median_prediction,
        )
        transition_performance.append(transition_detail)
        slot_performance.append(
            slot_posterior_diagnostic(
                probabilities,
                targets,
                ranks,
                argmax_prediction,
                mean_prediction,
            )
        )
        update_class_statistics(
            class_statistics,
            probabilities,
            targets,
            ranks,
            argmax_prediction,
            mean_prediction,
            best_intermediate,
        )

        target_zero_ratio = float((targets == 0).mean())
        target_full_ratio = float((targets == 127).mean())
        target_intermediate_ratio = 1.0 - target_zero_ratio - target_full_ratio
        intermediate_probabilities = group_mass["p_intermediate"][intermediate_mask]
        stale_all = transition_detail["stale_groups"]["all"]
        stale_advantages = stale_all["advantages"]
        performance_rows.append(
            {
                "metadata_index": row["metadata_index"],
                "composer": row["composer"],
                "title": row["title"],
                "piece_id": row["piece_id"],
                "performance_path": row["performance_path"],
                "num_notes": len(tokens),
                "num_pedal_tokens": len(tokens) * PEDAL_SLOTS,
                "num_windows": inference["num_windows"],
                "target_zero_ratio": target_zero_ratio,
                "target_full_ratio": target_full_ratio,
                "target_intermediate_ratio": target_intermediate_ratio,
                "argmax_accuracy": performance_decoder_metrics["argmax"]["standard"]["pedal_token_accuracy"],
                "argmax_mae": performance_decoder_metrics["argmax"]["standard"]["pedal_value_mae"],
                "argmax_intermediate_accuracy": performance_decoder_metrics["argmax"]["intermediate"]["intermediate_exact_accuracy"],
                "argmax_transition_f1": performance_decoder_metrics["argmax"]["transition"]["transition_detection_f1"],
                "posterior_mean_accuracy": performance_decoder_metrics["posterior_mean"]["standard"]["pedal_token_accuracy"],
                "posterior_mean_mae": performance_decoder_metrics["posterior_mean"]["standard"]["pedal_value_mae"],
                "posterior_mean_intermediate_mae": performance_decoder_metrics["posterior_mean"]["intermediate"]["intermediate_mae"],
                "posterior_mean_predicted_intermediate_ratio": performance_decoder_metrics["posterior_mean"]["distribution"]["predicted_intermediate_ratio"],
                "posterior_mean_transition_f1": performance_decoder_metrics["posterior_mean"]["transition"]["transition_detection_f1"],
                "posterior_median_accuracy": performance_decoder_metrics["posterior_median"]["standard"]["pedal_token_accuracy"],
                "posterior_median_mae": performance_decoder_metrics["posterior_median"]["standard"]["pedal_value_mae"],
                "posterior_median_intermediate_mae": performance_decoder_metrics["posterior_median"]["intermediate"]["intermediate_mae"],
                "posterior_median_predicted_intermediate_ratio": performance_decoder_metrics["posterior_median"]["distribution"]["predicted_intermediate_ratio"],
                "posterior_median_transition_f1": performance_decoder_metrics["posterior_median"]["transition"]["transition_detection_f1"],
                "mean_intermediate_probability_on_intermediate_targets": float(intermediate_probabilities.mean()) if len(intermediate_probabilities) else float("nan"),
                "median_intermediate_probability_on_intermediate_targets": float(np.median(intermediate_probabilities)) if len(intermediate_probabilities) else float("nan"),
                "intermediate_true_class_mean_rank": float(ranks[intermediate_mask].mean()) if np.any(intermediate_mask) else float("nan"),
                "transition_stale_state_fraction": safe_ratio(
                    stale_all["previous_probability_greater"], stale_all["count"]
                ),
                "transition_mean_stale_log_probability_advantage": float(stale_advantages.mean()) if len(stale_advantages) else float("nan"),
            }
        )

        total_windows += int(inference["num_windows"])
        min_contributions = min(min_contributions, int(counts.min()))
        max_contributions = max(max_contributions, int(counts.max()))
        print(
            f"diagnosed performance {performance_number}/{len(selected_rows)} "
            f"path={row['performance_path']} notes={len(tokens)} "
            f"windows={inference['num_windows']} "
            f"argmax_accuracy={performance_rows[-1]['argmax_accuracy']:.6f} "
            f"mean_mae={performance_rows[-1]['posterior_mean_mae']:.6f}",
            flush=True,
        )
        del inference, probabilities, mean_logits

    torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - inference_start
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("posterior inference created gradients")
    changed = [
        name
        for name, parameter in model.named_parameters()
        if parameter._version != parameter_versions[name]
    ]
    if changed:
        raise RuntimeError(f"model parameters changed during diagnostic: {changed}")

    posterior_summaries = {
        name: summarize_posterior_group(posterior_group_chunks[name], name)
        for name in ("overall", *GROUP_NAMES)
    }
    decoder_summaries = {
        name: aggregate_decoder_metrics(decoder_performance[name])
        for name in DECODER_NAMES
    }
    endpoint_mixture = aggregate_endpoint_mixture(mixture_performance)
    transition_posterior = aggregate_transition_posterior(
        transition_performance
    )
    slots = aggregate_slot_posterior(slot_performance)
    class_rows, class_summary = finalize_class_statistics(class_statistics)
    best_intermediate_summary = summarize_best_intermediate(
        best_intermediate_performance, posterior_summaries["intermediate"]
    )
    note_count = sum(int(row["num_notes"]) for row in performance_rows)
    verification = {
        "performance_count": len(selected_rows),
        "piece_count": len({row["piece_id"] for row in selected_rows}),
        "unique_path_count": len({row["performance_path"] for row in selected_rows}),
        "note_count": note_count,
        "target_count": note_count * PEDAL_SLOTS,
        "window_count": total_windows,
        "min_contributions": int(min_contributions),
        "max_contributions": int(max_contributions),
        "full_validation_performance_count": len(validation_rows),
        "full_validation_piece_count": len({row["piece_id"] for row in validation_rows}),
    }
    results = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint": checkpoint,
        "project_commit": str(configuration["project_commit"]),
        "pt_commit": str(configuration["pt_commit"]),
        "gpu": gpu,
        "batch_size": int(configuration["batch_size"]),
        "verification": verification,
        "posterior_groups": posterior_summaries,
        "decoders": decoder_summaries,
        "best_intermediate": best_intermediate_summary,
        "coarse": {
            "posterior_mass": confusion_from_matrix(posterior_confusion),
            "collapsed_argmax": confusion_from_matrix(collapsed_confusion),
        },
        "endpoint_mixture": endpoint_mixture,
        "transition_posterior": transition_posterior,
        "slots": slots,
        "class_rows": class_rows,
        "class_summary": class_summary,
        "performance_rows": performance_rows,
        "timing": {
            "preparation_seconds": preparation_seconds,
            "model_load_seconds": model_load_seconds,
            "inference_seconds": inference_seconds,
            "total_seconds": time.perf_counter() - total_start,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        },
    }
    if smoke_test:
        if list(performance_rows[0]) != PERFORMANCE_FIELDS or len(class_rows) != 128:
            raise RuntimeError("smoke output schema failed")
        if verification["performance_count"] != 2:
            raise RuntimeError("smoke did not process two performances")
        print(
            "SMOKE_SUCCESS "
            f"performances=2 notes={note_count} windows={total_windows} "
            f"coverage={min_contributions}-{max_contributions} "
            f"posterior_sum_valid=True missing_keys={checkpoint['missing_keys']} "
            f"unexpected_keys={checkpoint['unexpected_keys']} test_rows_used=0",
            flush=True,
        )
        return results

    if len(selected_rows) != EXPECTED_VALIDATION_PERFORMANCES:
        raise RuntimeError("full diagnostic did not process 71 performances")
    report = render_report(results)
    performance_csv = render_csv(performance_rows, PERFORMANCE_FIELDS)
    class_csv = render_csv(class_rows, CLASS_FIELDS)
    output_dir.mkdir(parents=False)
    (output_dir / "POSTERIOR_DIAGNOSTIC.md").write_text(
        report, encoding="utf-8"
    )
    (output_dir / "per_performance_posterior.csv").write_text(
        performance_csv, encoding="utf-8"
    )
    (output_dir / "per_class_posterior.csv").write_text(
        class_csv, encoding="utf-8"
    )
    print(
        "FULL_DIAGNOSTIC_SUCCESS "
        f"performances={len(selected_rows)} notes={note_count} "
        f"windows={total_windows} diagnosis={results['diagnosis']['code']} "
        f"output={output_dir}",
        flush=True,
    )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asap-root", default="/workspace/public/ASAP/asap-dataset-v1.1")
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
        default="/workspace/project/analysis/stage2_encoder_only_v0/validation_posterior_diagnostic_v0",
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
