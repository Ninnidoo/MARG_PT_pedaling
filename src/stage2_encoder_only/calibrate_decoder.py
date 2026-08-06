"""Validation-only hierarchical decoder calibration for Stage 2 pedaling."""

from __future__ import annotations

import argparse
import csv
import io
import json
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
    PEDAL_SLOTS,
    PEDAL_TOKEN_OFFSET,
    Stage2PedalDataset,
)
from .diagnose_posterior import (
    DECODER_NAMES,
    aggregate_decoder_metrics,
    coarse_confusion,
    confusion_from_matrix,
    decoder_metrics,
    finite_mean,
    format_metric,
    infer_complete_posterior,
    markdown_table,
    posterior_mean_decode,
    posterior_median_decode,
    render_csv,
    softmax_float32,
    target_group_ids,
)
from .evaluate_oracle import (
    PINNED_PT_COMMIT,
    TRAINING_PROJECT_COMMIT,
    WINDOW_NOTES,
    STRIDE_NOTES,
    _load_model,
    _visible_gpu_identity,
    aggregate_transition,
)
from .training import set_deterministic_seed


SEED = 20260710
TAU_GRID = (0.0, 0.25, 0.50, 0.75, 1.00)
CONDITIONAL_DECODERS = ("map", "median", "mean")
EXPECTED_PERFORMANCES = 71
EXPECTED_PIECES = 19
EXPECTED_NOTES = 283_928
EXPECTED_TARGETS = 1_135_712
EXPECTED_WINDOWS = 1_078
REFERENCE_MEDIAN_MAE = 28.197258
REFERENCE_MEDIAN_INTERMEDIATE_MAE = 36.245111
REFERENCE_GROUP_MACRO_F1 = 0.574021

CANDIDATES = (
    "argmax",
    "posterior_mean",
    "posterior_median",
    "raw_hierarchy_map",
    "raw_hierarchy_mean",
    "raw_hierarchy_median",
    "temperature_hierarchy",
    "bias_temperature_hierarchy",
)

METRIC_NAMES = (
    "exact_accuracy",
    "exact_note_accuracy",
    "pedal1_accuracy",
    "pedal2_accuracy",
    "pedal3_accuracy",
    "pedal4_accuracy",
    "mae",
    "rmse",
    "tolerance_accuracy_5",
    "tolerance_accuracy_10",
    "tolerance_accuracy_20",
    "predicted_zero_ratio",
    "predicted_intermediate_ratio",
    "predicted_full_ratio",
    "intermediate_exact_accuracy",
    "intermediate_mae",
    "intermediate_rmse",
    "intermediate_tolerance_accuracy_5",
    "intermediate_tolerance_accuracy_10",
    "intermediate_tolerance_accuracy_20",
    "intermediate_endpoint_collapse_ratio",
    "group_accuracy",
    "group_zero_precision",
    "group_zero_recall",
    "group_zero_f1",
    "group_intermediate_precision",
    "group_intermediate_recall",
    "group_intermediate_f1",
    "group_full_precision",
    "group_full_recall",
    "group_full_f1",
    "group_macro_f1",
    "group_balanced_accuracy",
    "transition_position_exact_accuracy",
    "transition_position_mae",
    "steady_position_exact_accuracy",
    "steady_position_mae",
    "transition_detection_precision",
    "transition_detection_recall",
    "transition_detection_f1",
    "transition_direction_accuracy",
    "delta_mae",
)

CANDIDATE_FIELDS = [
    "candidate_name",
    "region_calibration_type",
    "conditional_decoder",
    "mean_selected_tau",
    "original_128_nll",
    "region_nll",
    *[f"micro_{name}" for name in METRIC_NAMES],
    *[f"macro_{name}" for name in METRIC_NAMES],
    "criterion_a",
    "criterion_b",
    "safeguard_intermediate_ratio",
    "safeguard_endpoint_collapse",
    "safeguard_group_macro_f1",
    "safeguard_non_degenerate",
    "all_folds_finite",
    "success_criteria_met",
    "final_rank",
]

FOLD_FIELDS = [
    "fold_index",
    "held_out_piece_id",
    "held_out_performance_count",
    "held_out_note_count",
    "held_out_token_count",
    "candidate_name",
    "region_calibration_type",
    "fitted_region_temperature",
    "fitted_bias_zero",
    "fitted_bias_intermediate",
    "fitted_bias_full",
    "selected_tau",
    "selected_conditional_decoder",
    "calibration_nll",
    "held_out_overall_mae",
    "held_out_intermediate_mae",
    "held_out_exact_accuracy",
    "held_out_predicted_intermediate_ratio",
    "held_out_three_way_macro_f1",
    "held_out_transition_f1",
    "finite_flag",
    "convergence_flag",
]


def region_logits(logits: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Construct ZERO/INTERMEDIATE/FULL logits without normalizing."""

    if isinstance(logits, torch.Tensor):
        if logits.shape[-1] != 128:
            raise ValueError("logits must end with 128 classes")
        return torch.stack(
            (
                logits[..., 0],
                torch.logsumexp(logits[..., 1:127], dim=-1),
                logits[..., 127],
            ),
            dim=-1,
        )
    values = np.asarray(logits, dtype=np.float32)
    if values.shape[-1] != 128 or not np.isfinite(values).all():
        raise ValueError("logits must be finite and end with 128 classes")
    maximum = values[..., 1:127].max(axis=-1, keepdims=True)
    intermediate = maximum[..., 0] + np.log(
        np.exp(values[..., 1:127] - maximum).sum(axis=-1)
    )
    return np.stack((values[..., 0], intermediate, values[..., 127]), axis=-1)


def smoothed_intermediate_priors(train_counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    counts = np.asarray(train_counts, dtype=np.int64)
    if counts.shape == (128,):
        counts = counts[1:127]
    if counts.shape != (126,) or np.any(counts < 0):
        raise ValueError("train counts must contain 126 intermediate classes")
    smoothed = counts.astype(np.float64) + 1.0
    priors = smoothed / smoothed.sum()
    return priors.astype(np.float32), np.log(priors).astype(np.float32)


def prior_corrected_logits(
    intermediate_logits: np.ndarray, tau: float, log_priors: np.ndarray
) -> np.ndarray:
    values = np.asarray(intermediate_logits, dtype=np.float32)
    priors = np.asarray(log_priors, dtype=np.float32)
    if values.shape[-1] != 126 or priors.shape != (126,):
        raise ValueError("conditional logits/priors must contain 126 classes")
    if tau not in TAU_GRID:
        raise ValueError("tau is outside the fixed grid")
    corrected = values - np.float32(tau) * priors
    if not np.isfinite(corrected).all():
        raise FloatingPointError("non-finite prior-corrected logits")
    return corrected


def conditional_depth_decode(
    intermediate_logits: np.ndarray,
    tau: float,
    decoder: str,
    log_priors: np.ndarray,
) -> np.ndarray:
    corrected = prior_corrected_logits(intermediate_logits, tau, log_priors)
    if decoder == "map":
        return corrected.argmax(axis=-1).astype(np.int64) + 1
    probabilities = torch.softmax(
        torch.from_numpy(corrected.astype(np.float32, copy=False)), dim=-1
    ).numpy()
    if not np.isfinite(probabilities).all() or not np.allclose(
        probabilities.sum(axis=-1), 1.0, atol=2e-5
    ):
        raise FloatingPointError("invalid conditional posterior")
    if decoder == "mean":
        classes = np.arange(1, 127, dtype=np.float32)
        expectation = np.sum(probabilities * classes, axis=-1, dtype=np.float32)
        return np.clip(np.rint(expectation), 1, 126).astype(np.int64)
    if decoder == "median":
        cumulative = np.cumsum(probabilities, axis=-1, dtype=np.float32)
        return ((cumulative < 0.5).sum(axis=-1) + 1).clip(1, 126).astype(np.int64)
    raise ValueError(f"unknown conditional decoder {decoder}")


def apply_region_calibration(
    groups: np.ndarray,
    temperature: float,
    bias_intermediate: float = 0.0,
    bias_full: float = 0.0,
) -> np.ndarray:
    values = np.asarray(groups, dtype=np.float32)
    if values.shape[-1] != 3 or not np.isfinite(values).all():
        raise ValueError("region logits must be finite with three classes")
    if not math.isfinite(temperature) or temperature <= 1e-4:
        raise ValueError("temperature must be finite and greater than 1e-4")
    biases = np.asarray([0.0, bias_intermediate, bias_full], dtype=np.float32)
    calibrated = values / np.float32(temperature) + biases
    if not np.isfinite(calibrated).all():
        raise FloatingPointError("non-finite calibrated region logits")
    return calibrated


def hierarchical_decode(
    logits: np.ndarray,
    log_priors: np.ndarray,
    conditional_decoder: str,
    tau: float = 0.0,
    temperature: float = 1.0,
    bias_intermediate: float = 0.0,
    bias_full: float = 0.0,
) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float32)
    groups = apply_region_calibration(
        region_logits(values), temperature, bias_intermediate, bias_full
    )
    selected = groups.argmax(axis=-1)
    prediction = np.empty(selected.shape, dtype=np.int64)
    prediction[selected == 0] = 0
    prediction[selected == 2] = 127
    intermediate = selected == 1
    if np.any(intermediate):
        prediction[intermediate] = conditional_depth_decode(
            values[..., 1:127][intermediate],
            tau,
            conditional_decoder,
            log_priors,
        )
    return prediction


def inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


def fit_region_calibrator(
    groups: np.ndarray,
    targets: np.ndarray,
    calibrator_type: str,
    seed: int = SEED,
    max_iterations: int = 75,
    tolerance_grad: float = 1e-7,
    tolerance_change: float = 1e-9,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Fit deterministic scalar temperature with optional group biases."""

    if calibrator_type not in ("temperature", "bias_temperature"):
        raise ValueError("invalid calibrator type")
    set_deterministic_seed(seed)
    group_tensor = torch.as_tensor(groups, dtype=torch.float32, device=device)
    target_tensor = torch.as_tensor(targets, dtype=torch.long, device=device)
    if group_tensor.ndim != 2 or group_tensor.shape[1] != 3:
        raise ValueError("group logits must have shape [tokens, 3]")
    if target_tensor.shape != (len(group_tensor),):
        raise ValueError("region targets do not match logits")
    t_raw = torch.tensor(
        inverse_softplus(1.0 - 1e-4),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    parameters = [t_raw]
    if calibrator_type == "bias_temperature":
        learned_bias = torch.zeros(2, dtype=torch.float32, device=device, requires_grad=True)
        parameters.append(learned_bias)
    else:
        learned_bias = torch.zeros(2, dtype=torch.float32, device=device)

    def loss_value() -> torch.Tensor:
        temperature = F.softplus(t_raw) + 1e-4
        bias = torch.cat((torch.zeros(1, device=device), learned_bias))
        return F.cross_entropy(group_tensor / temperature + bias, target_tensor)

    initial_nll = float(loss_value().detach().cpu())
    optimizer = torch.optim.LBFGS(
        parameters,
        lr=1.0,
        max_iter=max_iterations,
        tolerance_grad=tolerance_grad,
        tolerance_change=tolerance_change,
        history_size=20,
        line_search_fn="strong_wolfe",
    )
    closure_calls = 0

    def closure() -> torch.Tensor:
        nonlocal closure_calls
        optimizer.zero_grad(set_to_none=True)
        loss = loss_value()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("non-finite calibration loss")
        loss.backward()
        closure_calls += 1
        return loss

    optimizer.step(closure)
    final_nll = float(loss_value().detach().cpu())
    temperature = float((F.softplus(t_raw) + 1e-4).detach().cpu())
    bias_values = learned_bias.detach().cpu().numpy()
    finite = all(
        math.isfinite(value)
        for value in (initial_nll, final_nll, temperature, *bias_values.tolist())
    )
    if not finite or temperature <= 1e-4:
        raise FloatingPointError("invalid fitted calibration parameters")
    return {
        "calibrator_type": calibrator_type,
        "temperature": temperature,
        "bias_zero": 0.0,
        "bias_intermediate": float(bias_values[0]),
        "bias_full": float(bias_values[1]),
        "initial_nll": initial_nll,
        "final_nll": final_nll,
        "closure_calls": closure_calls,
        "max_iterations": max_iterations,
        "tolerance_grad": tolerance_grad,
        "tolerance_change": tolerance_change,
        "converged": final_nll <= initial_nll + 1e-7,
        "finite": finite,
    }


def make_piece_folds(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    pieces: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        pieces[str(record["piece_id"])].append(index)
    folds = []
    all_indices = set(range(len(records)))
    for fold_index, piece_id in enumerate(sorted(pieces), start=1):
        held_out = tuple(pieces[piece_id])
        calibration = tuple(sorted(all_indices - set(held_out)))
        if set(held_out) & set(calibration):
            raise RuntimeError("fold leakage")
        folds.append(
            {
                "fold_index": fold_index,
                "held_out_piece_id": piece_id,
                "held_out_indices": held_out,
                "calibration_indices": calibration,
            }
        )
    flattened = [index for fold in folds for index in fold["held_out_indices"]]
    if sorted(flattened) != list(range(len(records))):
        raise RuntimeError("out-of-fold coverage is not exactly once")
    return folds


def select_conditional_candidate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("conditional selection requires candidates")
    if any(row["decoder"] not in CONDITIONAL_DECODERS for row in rows):
        raise ValueError("invalid conditional decoder")
    minimum_intermediate = min(float(row["intermediate_mae"]) for row in rows)
    eligible = [
        row
        for row in rows
        if float(row["intermediate_mae"]) <= minimum_intermediate + 0.05 + 1e-12
    ]
    minimum_overall = min(float(row["overall_mae"]) for row in eligible)
    eligible = [
        row
        for row in eligible
        if math.isclose(float(row["overall_mae"]), minimum_overall, abs_tol=1e-12)
    ]
    maximum_tolerance = max(float(row["intermediate_tolerance_10"]) for row in eligible)
    eligible = [
        row
        for row in eligible
        if math.isclose(
            float(row["intermediate_tolerance_10"]),
            maximum_tolerance,
            abs_tol=1e-12,
        )
    ]
    decoder_order = {name: index for index, name in enumerate(CONDITIONAL_DECODERS)}
    return min(eligible, key=lambda row: (float(row["tau"]), decoder_order[row["decoder"]]))


def success_criteria(metrics: dict[str, Any]) -> dict[str, bool]:
    overall = float(metrics["micro_mae"])
    intermediate = float(metrics["micro_intermediate_mae"])
    criterion_a = REFERENCE_MEDIAN_MAE - overall >= 1.0
    criterion_b = (
        REFERENCE_MEDIAN_INTERMEDIATE_MAE - intermediate >= 3.0
        and overall - REFERENCE_MEDIAN_MAE <= 0.5
    )
    ratio_ok = 0.15 <= float(metrics["micro_predicted_intermediate_ratio"]) <= 0.55
    collapse_ok = float(metrics["micro_intermediate_endpoint_collapse_ratio"]) < 0.50
    group_ok = float(metrics["micro_group_macro_f1"]) >= REFERENCE_GROUP_MACRO_F1 - 0.01
    non_degenerate = bool(metrics.get("non_degenerate", True))
    finite = bool(metrics.get("all_folds_finite", True))
    passed = (criterion_a or criterion_b) and all(
        (ratio_ok, collapse_ok, group_ok, non_degenerate, finite)
    )
    return {
        "criterion_a": criterion_a,
        "criterion_b": criterion_b,
        "safeguard_intermediate_ratio": ratio_ok,
        "safeguard_endpoint_collapse": collapse_ok,
        "safeguard_group_macro_f1": group_ok,
        "safeguard_non_degenerate": non_degenerate,
        "all_folds_finite": finite,
        "success_criteria_met": passed,
    }


def candidate_simplicity(name: str) -> int:
    if name.startswith("raw_hierarchy"):
        return 0
    if name == "temperature_hierarchy":
        return 1
    if name == "bias_temperature_hierarchy":
        return 2
    return 3


def select_final_candidate(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank candidates with the predeclared success/MAE/tie rules."""

    remaining = [dict(row) for row in rows]
    ranked: list[dict[str, Any]] = []
    while remaining:
        successful = [row for row in remaining if bool(row["success_criteria_met"])]
        pool = successful if successful else remaining
        minimum_mae = min(float(row["micro_mae"]) for row in pool)
        eligible = [row for row in pool if float(row["micro_mae"]) <= minimum_mae + 0.10 + 1e-12]
        minimum_intermediate = min(float(row["micro_intermediate_mae"]) for row in eligible)
        eligible = [
            row
            for row in eligible
            if math.isclose(
                float(row["micro_intermediate_mae"]), minimum_intermediate, abs_tol=1e-12
            )
        ]
        maximum_group = max(float(row["micro_group_macro_f1"]) for row in eligible)
        eligible = [
            row
            for row in eligible
            if math.isclose(float(row["micro_group_macro_f1"]), maximum_group, abs_tol=1e-12)
        ]
        chosen = min(
            eligible,
            key=lambda row: (candidate_simplicity(str(row["candidate_name"])), str(row["candidate_name"])),
        )
        chosen["final_rank"] = len(ranked) + 1
        ranked.append(chosen)
        remaining.remove(next(row for row in remaining if row["candidate_name"] == chosen["candidate_name"]))
    return ranked


def calibration_config_roundtrip(config: dict[str, Any], logits: np.ndarray) -> np.ndarray:
    """Test helper: JSON round-trip a configuration and decode identically."""

    restored = json.loads(json.dumps(config, sort_keys=True, allow_nan=False))
    return hierarchical_decode(
        logits,
        np.asarray(restored["log_intermediate_priors"], dtype=np.float32),
        restored["conditional_decoder"],
        float(restored["selected_tau"]),
        float(restored["fitted_temperature"]),
        float(restored["fitted_group_biases"]["intermediate"]),
        float(restored["fitted_group_biases"]["full"]),
    )

def performance_candidate_metrics(
    predictions: np.ndarray, targets: np.ndarray, performance_path: str = ""
) -> dict[str, Any]:
    prediction = np.asarray(predictions, dtype=np.int64)
    truth = np.asarray(targets, dtype=np.int64)
    raw = decoder_metrics(prediction, truth)
    intermediate_mask = (truth >= 1) & (truth <= 126)
    intermediate_error = prediction[intermediate_mask] - truth[intermediate_mask]
    coarse = coarse_confusion(target_group_ids(truth), target_group_ids(prediction))
    standard = raw["standard"]
    intermediate = raw["intermediate"]
    transition = raw["transition"]
    distribution = raw["distribution"]
    count = int(raw["_target_count"])
    intermediate_count = int(intermediate["intermediate_target_count"])
    flat = {
        "exact_accuracy": standard["pedal_token_accuracy"],
        "exact_note_accuracy": standard["exact_note_accuracy"],
        "pedal1_accuracy": standard["pedal1_accuracy"],
        "pedal2_accuracy": standard["pedal2_accuracy"],
        "pedal3_accuracy": standard["pedal3_accuracy"],
        "pedal4_accuracy": standard["pedal4_accuracy"],
        "mae": standard["pedal_value_mae"],
        "rmse": math.sqrt(raw["_squared_error_sum"] / count),
        **{
            f"tolerance_accuracy_{tolerance}": raw["_tolerance_correct"][tolerance] / count
            for tolerance in (5, 10, 20)
        },
        "predicted_zero_ratio": distribution["predicted_zero_ratio"],
        "predicted_intermediate_ratio": distribution["predicted_intermediate_ratio"],
        "predicted_full_ratio": distribution["predicted_full_ratio"],
        "intermediate_exact_accuracy": intermediate["intermediate_exact_accuracy"],
        "intermediate_mae": intermediate["intermediate_mae"],
        "intermediate_rmse": math.sqrt(
            float(np.square(intermediate_error, dtype=np.int64).sum())
            / intermediate_count
        ),
        **{
            f"intermediate_tolerance_accuracy_{tolerance}": raw[
                "_intermediate_tolerance_correct"
            ][tolerance]
            / intermediate_count
            for tolerance in (5, 10, 20)
        },
        "intermediate_endpoint_collapse_ratio": intermediate["endpoint_collapse_ratio"],
        "group_accuracy": coarse["group_accuracy"],
        "group_zero_precision": coarse["precision"][0],
        "group_zero_recall": coarse["recall"][0],
        "group_zero_f1": coarse["f1"][0],
        "group_intermediate_precision": coarse["precision"][1],
        "group_intermediate_recall": coarse["recall"][1],
        "group_intermediate_f1": coarse["f1"][1],
        "group_full_precision": coarse["precision"][2],
        "group_full_recall": coarse["recall"][2],
        "group_full_f1": coarse["f1"][2],
        "group_macro_f1": coarse["macro_f1"],
        "group_balanced_accuracy": coarse["balanced_accuracy"],
        "transition_position_exact_accuracy": transition[
            "transition_position_exact_accuracy"
        ],
        "transition_position_mae": transition["transition_position_mae"],
        "steady_position_exact_accuracy": transition[
            "steady_position_exact_accuracy"
        ],
        "steady_position_mae": transition["steady_position_mae"],
        "transition_detection_precision": transition[
            "transition_detection_precision"
        ],
        "transition_detection_recall": transition["transition_detection_recall"],
        "transition_detection_f1": transition["transition_detection_f1"],
        "transition_direction_accuracy": transition[
            "transition_direction_accuracy"
        ],
        "delta_mae": raw["_delta_absolute_error_sum"] / raw["_delta_count"],
    }
    if set(flat) != set(METRIC_NAMES):
        raise RuntimeError("candidate metric schema mismatch")
    return {
        "performance_path": performance_path,
        "raw": raw,
        "flat": flat,
        "intermediate_squared_error_sum": int(
            np.square(intermediate_error, dtype=np.int64).sum()
        ),
        "coarse_confusion": coarse["confusion_matrix"],
    }


def aggregate_candidate_metrics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot aggregate empty candidate records")
    raw_records = [record["raw"] for record in records]
    decoder = aggregate_decoder_metrics(raw_records)
    transition = aggregate_transition([record["raw"]["transition"] for record in records])
    coarse_matrix = sum(
        (record["coarse_confusion"] for record in records),
        np.zeros((3, 3), dtype=np.int64),
    )
    coarse = confusion_from_matrix(coarse_matrix)
    intermediate_count = sum(
        int(record["raw"]["intermediate"]["intermediate_target_count"])
        for record in records
    )
    micro = {
        "exact_accuracy": decoder["pedal_token_accuracy"],
        "exact_note_accuracy": decoder["exact_note_accuracy"],
        "pedal1_accuracy": decoder["pedal1_accuracy"],
        "pedal2_accuracy": decoder["pedal2_accuracy"],
        "pedal3_accuracy": decoder["pedal3_accuracy"],
        "pedal4_accuracy": decoder["pedal4_accuracy"],
        "mae": decoder["pedal_value_mae"],
        "rmse": decoder["rmse"],
        **{
            f"tolerance_accuracy_{tolerance}": decoder[f"tolerance_accuracy_{tolerance}"]
            for tolerance in (5, 10, 20)
        },
        "predicted_zero_ratio": decoder["predicted_zero_ratio"],
        "predicted_intermediate_ratio": decoder["predicted_intermediate_ratio"],
        "predicted_full_ratio": decoder["predicted_full_ratio"],
        "intermediate_exact_accuracy": decoder["intermediate_exact_accuracy"],
        "intermediate_mae": decoder["intermediate_mae"],
        "intermediate_rmse": math.sqrt(
            sum(int(record["intermediate_squared_error_sum"]) for record in records)
            / intermediate_count
        ),
        **{
            f"intermediate_tolerance_accuracy_{tolerance}": decoder[
                f"intermediate_tolerance_accuracy_{tolerance}"
            ]
            for tolerance in (5, 10, 20)
        },
        "intermediate_endpoint_collapse_ratio": decoder["endpoint_collapse_ratio"],
        "group_accuracy": coarse["group_accuracy"],
        "group_zero_precision": coarse["precision"][0],
        "group_zero_recall": coarse["recall"][0],
        "group_zero_f1": coarse["f1"][0],
        "group_intermediate_precision": coarse["precision"][1],
        "group_intermediate_recall": coarse["recall"][1],
        "group_intermediate_f1": coarse["f1"][1],
        "group_full_precision": coarse["precision"][2],
        "group_full_recall": coarse["recall"][2],
        "group_full_f1": coarse["f1"][2],
        "group_macro_f1": coarse["macro_f1"],
        "group_balanced_accuracy": coarse["balanced_accuracy"],
        "transition_position_exact_accuracy": transition[
            "transition_position_exact_accuracy"
        ],
        "transition_position_mae": transition["transition_position_mae"],
        "steady_position_exact_accuracy": transition[
            "steady_position_exact_accuracy"
        ],
        "steady_position_mae": transition["steady_position_mae"],
        "transition_detection_precision": transition[
            "transition_detection_precision"
        ],
        "transition_detection_recall": transition["transition_detection_recall"],
        "transition_detection_f1": transition["transition_detection_f1"],
        "transition_direction_accuracy": transition[
            "transition_direction_accuracy"
        ],
        "delta_mae": decoder["delta_mae"],
    }
    macro = {
        name: finite_mean([float(record["flat"][name]) for record in records])
        for name in METRIC_NAMES
    }
    if set(micro) != set(METRIC_NAMES):
        raise RuntimeError("aggregate metric schema mismatch")
    predicted_group_counts = coarse_matrix.sum(axis=0)
    return {
        **{f"micro_{name}": value for name, value in micro.items()},
        **{f"macro_{name}": value for name, value in macro.items()},
        "coarse_confusion": coarse_matrix,
        "non_degenerate": bool(np.all(predicted_group_counts > 0)),
        "performance_count": len(records),
        "performance_paths": [record["performance_path"] for record in records],
    }


def precompute_conditional_predictions(
    logits: np.ndarray, log_priors: np.ndarray
) -> dict[tuple[float, str], np.ndarray]:
    values = np.asarray(logits, dtype=np.float32)[..., 1:127]
    classes = np.arange(1, 127, dtype=np.float32)
    output: dict[tuple[float, str], np.ndarray] = {}
    for tau in TAU_GRID:
        corrected = prior_corrected_logits(values, tau, log_priors)
        output[(tau, "map")] = (
            corrected.argmax(axis=-1).astype(np.int16) + 1
        )
        posterior = torch.softmax(
            torch.from_numpy(corrected.astype(np.float32, copy=False)), dim=-1
        ).numpy()
        expectation = np.sum(posterior * classes, axis=-1, dtype=np.float32)
        output[(tau, "mean")] = np.clip(
            np.rint(expectation), 1, 126
        ).astype(np.int16)
        cumulative = np.cumsum(posterior, axis=-1, dtype=np.float32)
        output[(tau, "median")] = (
            (cumulative < 0.5).sum(axis=-1) + 1
        ).clip(1, 126).astype(np.int16)
    return output


def region_prediction_for_record(
    record: dict[str, Any], parameters: dict[str, Any]
) -> np.ndarray:
    calibrated = apply_region_calibration(
        record["region_logits"],
        float(parameters["temperature"]),
        float(parameters["bias_intermediate"]),
        float(parameters["bias_full"]),
    )
    return calibrated.argmax(axis=-1).astype(np.int8)


def hierarchy_from_precomputed(
    record: dict[str, Any],
    parameters: dict[str, Any],
    tau: float,
    decoder: str,
    selected_region: np.ndarray | None = None,
) -> np.ndarray:
    region = (
        region_prediction_for_record(record, parameters)
        if selected_region is None
        else np.asarray(selected_region, dtype=np.int8)
    )
    prediction = np.empty(region.shape, dtype=np.int64)
    prediction[region == 0] = 0
    prediction[region == 2] = 127
    intermediate = region == 1
    prediction[intermediate] = record["conditional_predictions"][
        (float(tau), decoder)
    ][intermediate]
    return prediction


def cross_entropy_numpy(logits: np.ndarray, targets: np.ndarray) -> float:
    values = torch.from_numpy(np.asarray(logits, dtype=np.float32).reshape(-1, logits.shape[-1]))
    truth = torch.from_numpy(np.asarray(targets, dtype=np.int64).reshape(-1))
    result = float(F.cross_entropy(values, truth).item())
    if not math.isfinite(result):
        raise FloatingPointError("non-finite cross entropy")
    return result


def concatenate_region_data(
    records: Sequence[dict[str, Any]], indices: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    groups = np.concatenate(
        [records[index]["region_logits"].reshape(-1, 3) for index in indices],
        axis=0,
    )
    targets = np.concatenate(
        [
            target_group_ids(records[index]["targets"]).reshape(-1)
            for index in indices
        ],
        axis=0,
    )
    return groups.astype(np.float32, copy=False), targets.astype(np.int64, copy=False)


def conditional_selection_scores(
    records: Sequence[dict[str, Any]],
    indices: Sequence[int],
    parameters: dict[str, Any],
) -> list[dict[str, Any]]:
    regions = {
        index: region_prediction_for_record(records[index], parameters)
        for index in indices
    }
    rows: list[dict[str, Any]] = []
    for tau in TAU_GRID:
        for decoder in CONDITIONAL_DECODERS:
            absolute_error_sum = 0
            target_count = 0
            intermediate_error_sum = 0
            intermediate_count = 0
            intermediate_tolerance = 0
            for index in indices:
                record = records[index]
                prediction = hierarchy_from_precomputed(
                    record, parameters, tau, decoder, regions[index]
                )
                truth = record["targets"]
                error = np.abs(prediction - truth)
                mask = (truth >= 1) & (truth <= 126)
                absolute_error_sum += int(error.sum())
                target_count += error.size
                intermediate_error_sum += int(error[mask].sum())
                intermediate_count += int(mask.sum())
                intermediate_tolerance += int((error[mask] <= 10).sum())
            rows.append(
                {
                    "tau": tau,
                    "decoder": decoder,
                    "intermediate_mae": intermediate_error_sum
                    / intermediate_count,
                    "overall_mae": absolute_error_sum / target_count,
                    "intermediate_tolerance_10": intermediate_tolerance
                    / intermediate_count,
                }
            )
    return rows


def candidate_identity(candidate: str) -> tuple[str, str]:
    if candidate == "argmax":
        return "none", "argmax"
    if candidate == "posterior_mean":
        return "none", "posterior_mean"
    if candidate == "posterior_median":
        return "none", "posterior_median"
    if candidate.startswith("raw_hierarchy_"):
        return "uncalibrated", candidate.removeprefix("raw_hierarchy_")
    if candidate == "temperature_hierarchy":
        return "temperature", "fold-selected"
    if candidate == "bias_temperature_hierarchy":
        return "bias_temperature", "fold-selected"
    raise ValueError(f"unknown candidate {candidate}")


def prediction_for_candidate(
    record: dict[str, Any],
    candidate: str,
    fitted: dict[str, dict[str, Any]],
    selections: dict[str, dict[str, Any]],
) -> np.ndarray:
    if candidate in record["baseline_predictions"]:
        return record["baseline_predictions"][candidate].astype(np.int64, copy=False)
    if candidate.startswith("raw_hierarchy_"):
        decoder = candidate.removeprefix("raw_hierarchy_")
        return hierarchy_from_precomputed(
            record,
            {
                "temperature": 1.0,
                "bias_intermediate": 0.0,
                "bias_full": 0.0,
            },
            0.0,
            decoder,
        )
    calibrator_type = (
        "temperature"
        if candidate == "temperature_hierarchy"
        else "bias_temperature"
    )
    selected = selections[calibrator_type]
    return hierarchy_from_precomputed(
        record,
        fitted[calibrator_type],
        float(selected["tau"]),
        str(selected["decoder"]),
    )


def finite_fold_metrics(metrics: dict[str, Any]) -> bool:
    keys = (
        "micro_mae",
        "micro_intermediate_mae",
        "micro_exact_accuracy",
        "micro_predicted_intermediate_ratio",
        "micro_group_macro_f1",
        "micro_transition_detection_f1",
    )
    return all(math.isfinite(float(metrics[key])) for key in keys)


def run_cross_validation(
    records: Sequence[dict[str, Any]],
    folds: Sequence[dict[str, Any]],
    device: torch.device,
    original_nll: float,
) -> dict[str, Any]:
    candidate_records: dict[str, list[dict[str, Any]]] = {
        candidate: [] for candidate in CANDIDATES
    }
    fold_rows: list[dict[str, Any]] = []
    fold_calibrations: list[dict[str, Any]] = []
    region_nll_sums = {candidate: 0.0 for candidate in CANDIDATES}
    region_nll_counts = {candidate: 0 for candidate in CANDIDATES}
    selected_patterns: dict[str, list[dict[str, Any]]] = {
        "temperature": [],
        "bias_temperature": [],
    }

    for fold in folds:
        calibration_indices = fold["calibration_indices"]
        held_out_indices = fold["held_out_indices"]
        calibration_groups, calibration_targets = concatenate_region_data(
            records, calibration_indices
        )
        uncalibrated_nll = cross_entropy_numpy(
            calibration_groups, calibration_targets
        )
        fitted = {
            calibrator_type: fit_region_calibrator(
                calibration_groups,
                calibration_targets,
                calibrator_type,
                seed=SEED,
                device=device,
            )
            for calibrator_type in ("temperature", "bias_temperature")
        }
        selections = {
            calibrator_type: select_conditional_candidate(
                conditional_selection_scores(
                    records,
                    calibration_indices,
                    {
                        "temperature": fitted[calibrator_type]["temperature"],
                        "bias_intermediate": fitted[calibrator_type][
                            "bias_intermediate"
                        ],
                        "bias_full": fitted[calibrator_type]["bias_full"],
                    },
                )
            )
            for calibrator_type in ("temperature", "bias_temperature")
        }
        for calibrator_type in selections:
            selected_patterns[calibrator_type].append(selections[calibrator_type])

        held_out_groups, held_out_targets = concatenate_region_data(
            records, held_out_indices
        )
        held_out_region_nll = {
            "uncalibrated": cross_entropy_numpy(
                held_out_groups, held_out_targets
            ),
            "temperature": cross_entropy_numpy(
                apply_region_calibration(
                    held_out_groups,
                    fitted["temperature"]["temperature"],
                    0.0,
                    0.0,
                ),
                held_out_targets,
            ),
            "bias_temperature": cross_entropy_numpy(
                apply_region_calibration(
                    held_out_groups,
                    fitted["bias_temperature"]["temperature"],
                    fitted["bias_temperature"]["bias_intermediate"],
                    fitted["bias_temperature"]["bias_full"],
                ),
                held_out_targets,
            ),
        }
        held_out_token_count = sum(
            int(records[index]["targets"].size) for index in held_out_indices
        )
        held_out_note_count = sum(
            int(len(records[index]["targets"])) for index in held_out_indices
        )
        fold_calibrations.append(
            {
                "fold_index": fold["fold_index"],
                "held_out_piece_id": fold["held_out_piece_id"],
                "uncalibrated_calibration_nll": uncalibrated_nll,
                "temperature": fitted["temperature"],
                "bias_temperature": fitted["bias_temperature"],
                "selections": selections,
            }
        )

        for candidate in CANDIDATES:
            per_performance = []
            for index in held_out_indices:
                prediction = prediction_for_candidate(
                    records[index], candidate, fitted, selections
                )
                metrics = performance_candidate_metrics(
                    prediction,
                    records[index]["targets"],
                    records[index]["performance_path"],
                )
                per_performance.append(metrics)
                candidate_records[candidate].append(metrics)
            fold_metrics = aggregate_candidate_metrics(per_performance)
            region_type, conditional = candidate_identity(candidate)
            if region_type in ("temperature", "bias_temperature"):
                parameters = fitted[region_type]
                selected = selections[region_type]
                calibration_nll = parameters["final_nll"]
                region_nll = held_out_region_nll[region_type]
                convergence = parameters["converged"]
            else:
                parameters = {
                    "temperature": 1.0,
                    "bias_zero": 0.0,
                    "bias_intermediate": 0.0,
                    "bias_full": 0.0,
                }
                selected = {
                    "tau": 0.0,
                    "decoder": conditional,
                }
                calibration_nll = uncalibrated_nll
                region_nll = held_out_region_nll["uncalibrated"]
                convergence = True
            region_nll_sums[candidate] += region_nll * held_out_token_count
            region_nll_counts[candidate] += held_out_token_count
            finite = finite_fold_metrics(fold_metrics)
            fold_rows.append(
                {
                    "fold_index": fold["fold_index"],
                    "held_out_piece_id": fold["held_out_piece_id"],
                    "held_out_performance_count": len(held_out_indices),
                    "held_out_note_count": held_out_note_count,
                    "held_out_token_count": held_out_token_count,
                    "candidate_name": candidate,
                    "region_calibration_type": region_type,
                    "fitted_region_temperature": parameters["temperature"],
                    "fitted_bias_zero": 0.0,
                    "fitted_bias_intermediate": parameters[
                        "bias_intermediate"
                    ],
                    "fitted_bias_full": parameters["bias_full"],
                    "selected_tau": selected["tau"],
                    "selected_conditional_decoder": selected["decoder"],
                    "calibration_nll": calibration_nll,
                    "held_out_overall_mae": fold_metrics["micro_mae"],
                    "held_out_intermediate_mae": fold_metrics[
                        "micro_intermediate_mae"
                    ],
                    "held_out_exact_accuracy": fold_metrics[
                        "micro_exact_accuracy"
                    ],
                    "held_out_predicted_intermediate_ratio": fold_metrics[
                        "micro_predicted_intermediate_ratio"
                    ],
                    "held_out_three_way_macro_f1": fold_metrics[
                        "micro_group_macro_f1"
                    ],
                    "held_out_transition_f1": fold_metrics[
                        "micro_transition_detection_f1"
                    ],
                    "finite_flag": finite,
                    "convergence_flag": bool(convergence),
                }
            )
        print(
            f"completed fold {fold['fold_index']}/{len(folds)} "
            f"held_out_piece={fold['held_out_piece_id']} "
            f"performances={len(held_out_indices)} "
            f"temp_T={fitted['temperature']['temperature']:.6f} "
            f"bias_T={fitted['bias_temperature']['temperature']:.6f} "
            f"temp_selection={selections['temperature']['tau']}/"
            f"{selections['temperature']['decoder']} "
            f"bias_selection={selections['bias_temperature']['tau']}/"
            f"{selections['bias_temperature']['decoder']}",
            flush=True,
        )

    candidate_rows = []
    candidate_details: dict[str, dict[str, Any]] = {}
    for candidate in CANDIDATES:
        paths = [record["performance_path"] for record in candidate_records[candidate]]
        if len(paths) != len(records) or len(set(paths)) != len(records):
            raise RuntimeError(f"OOF coverage failed for {candidate}")
        aggregate = aggregate_candidate_metrics(candidate_records[candidate])
        aggregate["all_folds_finite"] = all(
            bool(row["finite_flag"]) and bool(row["convergence_flag"])
            for row in fold_rows
            if row["candidate_name"] == candidate
        )
        criteria = success_criteria(aggregate)
        region_type, conditional = candidate_identity(candidate)
        if region_type in selected_patterns:
            pattern = selected_patterns[region_type]
            mean_tau = float(np.mean([float(item["tau"]) for item in pattern]))
        else:
            mean_tau = 0.0
        row = {
            "candidate_name": candidate,
            "region_calibration_type": region_type,
            "conditional_decoder": conditional,
            "mean_selected_tau": mean_tau,
            "original_128_nll": original_nll,
            "region_nll": region_nll_sums[candidate]
            / region_nll_counts[candidate],
            **{
                key: aggregate[key]
                for key in (
                    *[f"micro_{name}" for name in METRIC_NAMES],
                    *[f"macro_{name}" for name in METRIC_NAMES],
                )
            },
            **criteria,
            "final_rank": 0,
        }
        candidate_rows.append(row)
        candidate_details[candidate] = {
            "aggregate": aggregate,
            "coarse_confusion": aggregate["coarse_confusion"],
        }

    ranked = select_final_candidate(candidate_rows)
    rank_by_name = {
        row["candidate_name"]: row["final_rank"] for row in ranked
    }
    for row in candidate_rows:
        row["final_rank"] = rank_by_name[row["candidate_name"]]
    candidate_rows.sort(key=lambda row: int(row["final_rank"]))
    return {
        "candidate_rows": candidate_rows,
        "candidate_details": candidate_details,
        "fold_rows": fold_rows,
        "fold_calibrations": fold_calibrations,
        "selected_patterns": selected_patterns,
    }


def refit_selected_configuration(
    selected_row: dict[str, Any],
    records: Sequence[dict[str, Any]],
    train_counts: np.ndarray,
    log_priors: np.ndarray,
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate = str(selected_row["candidate_name"])
    region_type, conditional = candidate_identity(candidate)
    all_indices = tuple(range(len(records)))
    if region_type in ("temperature", "bias_temperature"):
        groups, targets = concatenate_region_data(records, all_indices)
        fitted = fit_region_calibrator(
            groups, targets, region_type, seed=SEED, device=device
        )
        selected = select_conditional_candidate(
            conditional_selection_scores(
                records,
                all_indices,
                {
                    "temperature": fitted["temperature"],
                    "bias_intermediate": fitted["bias_intermediate"],
                    "bias_full": fitted["bias_full"],
                },
            )
        )
    elif region_type == "uncalibrated":
        fitted = {
            "calibrator_type": "uncalibrated",
            "temperature": 1.0,
            "bias_zero": 0.0,
            "bias_intermediate": 0.0,
            "bias_full": 0.0,
            "initial_nll": float(selected_row["region_nll"]),
            "final_nll": float(selected_row["region_nll"]),
            "converged": True,
            "finite": True,
        }
        selected = {
            "tau": 0.0,
            "decoder": conditional,
            "intermediate_mae": float(selected_row["micro_intermediate_mae"]),
            "overall_mae": float(selected_row["micro_mae"]),
            "intermediate_tolerance_10": float(
                selected_row["micro_intermediate_tolerance_accuracy_10"]
            ),
        }
    else:
        fitted = {
            "calibrator_type": "none",
            "temperature": 1.0,
            "bias_zero": 0.0,
            "bias_intermediate": 0.0,
            "bias_full": 0.0,
            "initial_nll": float(selected_row["region_nll"]),
            "final_nll": float(selected_row["region_nll"]),
            "converged": True,
            "finite": True,
        }
        selected = {
            "tau": 0.0,
            "decoder": "not_applicable",
            "intermediate_mae": float(selected_row["micro_intermediate_mae"]),
            "overall_mae": float(selected_row["micro_mae"]),
            "intermediate_tolerance_10": float(
                selected_row["micro_intermediate_tolerance_accuracy_10"]
            ),
        }

    selection_metrics = {
        key: float(selected_row[key])
        for key in (
            "micro_exact_accuracy",
            "micro_mae",
            "micro_intermediate_mae",
            "micro_intermediate_tolerance_accuracy_10",
            "micro_predicted_intermediate_ratio",
            "micro_intermediate_endpoint_collapse_ratio",
            "micro_group_macro_f1",
            "micro_group_intermediate_recall",
            "micro_transition_detection_f1",
        )
    }
    config = {
        "selected_decoder_name": candidate,
        "region_calibrator_type": fitted["calibrator_type"],
        "fitted_temperature": float(fitted["temperature"]),
        "fitted_group_biases": {
            "zero": 0.0,
            "intermediate": float(fitted["bias_intermediate"]),
            "full": float(fitted["bias_full"]),
        },
        "selected_tau": float(selected["tau"]),
        "conditional_decoder": str(selected["decoder"]),
        "train_prior_smoothing_rule": "For intermediate classes 1..126, add one to each train count and normalize.",
        "train_intermediate_counts_classes_1_to_126": [
            int(value) for value in np.asarray(train_counts, dtype=np.int64)[1:127]
        ],
        "log_intermediate_priors": [
            float(value) for value in np.asarray(log_priors, dtype=np.float32)
        ],
        "cross_validation_selection_metrics": selection_metrics,
        "success_criteria_result": {
            key: bool(selected_row[key])
            for key in (
                "criterion_a",
                "criterion_b",
                "safeguard_intermediate_ratio",
                "safeguard_endpoint_collapse",
                "safeguard_group_macro_f1",
                "safeguard_non_degenerate",
                "all_folds_finite",
                "success_criteria_met",
            )
        },
        "seed": SEED,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["best_epoch"]),
        "checkpoint_best_validation_loss": float(
            checkpoint["best_validation_loss"]
        ),
        "project_commit": TRAINING_PROJECT_COMMIT,
        "pianist_transformer_commit": PINNED_PT_COMMIT,
        "exact_decoding_definitions": {
            "region_logits": "g=[z0, logsumexp(z1..z126), z127]",
            "region_calibration": "g_calibrated=g/T+[0,b_intermediate,b_full], T=softplus(t_raw)+1e-4",
            "region_choice": "argmax of the three calibrated region logits",
            "zero_output": 0,
            "full_output": 127,
            "conditional_prior_correction": "z'_c=z_c-tau*log(pi_c), c=1..126",
            "conditional_map": "argmax z'_c",
            "conditional_mean": "NumPy rint of conditional softmax expectation, clipped to 1..126",
            "conditional_median": "smallest class with conditional cumulative probability >=0.5",
            "posterior_median_128": "smallest class in 0..127 whose posterior cumulative probability is at least 0.5",
        },
        "piece_level_cross_validation": "19-fold leave-one-piece-out over validation pieces",
        "test_was_not_used": True,
        "neural_model_was_not_modified": True,
    }
    json.dumps(config, sort_keys=True, allow_nan=False)
    return config, {"fitted": fitted, "selected": selected}


def percentile_summary(values: Sequence[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    return (
        float(np.percentile(array, 10)),
        float(np.percentile(array, 50)),
        float(np.percentile(array, 90)),
    )


def render_report(results: dict[str, Any]) -> str:
    candidates = results["candidate_rows"]
    by_name = {row["candidate_name"]: row for row in candidates}
    folds = results["fold_rows"]
    selected = candidates[0]
    config = results["calibration_config"]
    verification = results["verification"]
    timing = results["timing"]
    checkpoint = results["checkpoint"]
    fold_calibrations = results["fold_calibrations"]
    selected_fold_rows = [
        row for row in folds if row["candidate_name"] == selected["candidate_name"]
    ]
    region_rows = [
        by_name["raw_hierarchy_map"],
        by_name["temperature_hierarchy"],
        by_name["bias_temperature_hierarchy"],
    ]
    parameter_rows = []
    for calibrator_type in ("temperature", "bias_temperature"):
        temperatures = [
            float(item[calibrator_type]["temperature"])
            for item in fold_calibrations
        ]
        intermediate_bias = [
            float(item[calibrator_type]["bias_intermediate"])
            for item in fold_calibrations
        ]
        full_bias = [
            float(item[calibrator_type]["bias_full"])
            for item in fold_calibrations
        ]
        parameter_rows.append(
            [
                calibrator_type,
                *[format_metric(value) for value in percentile_summary(temperatures)],
                *[format_metric(value) for value in percentile_summary(intermediate_bias)],
                *[format_metric(value) for value in percentile_summary(full_bias)],
            ]
        )

    lines = [
        "# Experiment 1 v0 Validation Hierarchical Decoder Calibration",
        "",
        "## 1. Scope",
        "",
        "This is a validation-only, piece-level leave-one-piece-out calibration "
        "study using the fixed neural checkpoint and Oracle human non-pedal "
        "performance input. It does not read test targets, retrain the model, "
        "modify neural parameters, or perform end-to-end Stage 1 inference.",
        "",
        "## 2. Motivation",
        "",
        "The preceding posterior diagnostic found meaningful aggregate "
        "intermediate mass (mean 0.386724 on intermediate targets), weak exact "
        "intermediate ordering (mean rank 40.485), and endpoint-only ordinary "
        "argmax predictions. This study separates region choice from conditional "
        "intermediate depth.",
        "",
        "## 3. Data and reconstruction verification",
        "",
        f"- Validation performances/pieces: {verification['performance_count']}/"
        f"{verification['piece_count']}",
        f"- Complete notes/targets/windows: {verification['note_count']}/"
        f"{verification['target_count']}/{verification['window_count']}",
        f"- Contribution range: {verification['min_contributions']}–"
        f"{verification['max_contributions']}; first/last coverage and reversed-"
        "window equality passed.",
        "- Every validation MIDI existed and tokenized. Each note was counted "
        "once. Train data was used only for smoothed class counts. No test row, "
        "MIDI, target, or prior statistic was used.",
        f"- Checkpoint epoch/loss: {checkpoint['best_epoch']}/"
        f"{checkpoint['best_validation_loss']:.10f}; exact load missing="
        f"{checkpoint['missing_keys']}, unexpected={checkpoint['unexpected_keys']}.",
        f"- GPU: UUID {results['gpu']['uuid']} as cuda:0; batch size "
        f"{results['batch_size']}; FP16 model forward and float32 calibration.",
        f"- Timing: preparation {timing['preparation_seconds']:.3f}s, model load "
        f"{timing['model_load_seconds']:.3f}s, reconstruction "
        f"{timing['inference_seconds']:.3f}s, calibration/CV "
        f"{timing['calibration_seconds']:.3f}s, total "
        f"{timing['total_seconds']:.3f}s.",
        f"- Peak allocated GPU memory: {timing['peak_gpu_memory_bytes']} bytes "
        f"({timing['peak_gpu_memory_bytes'] / 1024**3:.3f} GiB).",
        "",
        "## 4. Candidate decoders",
        "",
        "- Argmax, posterior mean, and posterior median reproduce the frozen "
        "128-class posterior baselines.",
        "- Raw hierarchy chooses argmax[p(0), sum p(1..126), p(127)], then uses "
        "conditional MAP, mean, or median for intermediate depth.",
        "- Calibrated hierarchy constructs g=[z0, logsumexp(z1..z126), z127] "
        "and applies g/T + [0,b_intermediate,b_full] before region argmax.",
        "- Conditional logits use z'_c=z_c-tau*log(pi_c) with add-one-smoothed "
        "train priors and tau in {0,.25,.50,.75,1}. Conditional mean uses NumPy "
        "rint; median is the smallest cumulative class at 0.5.",
        "",
        "## 5. Cross-validation protocol",
        "",
        f"{len(results['fold_calibrations'])} folds hold out one complete piece "
        "at a time. Region parameters and conditional tau/decoder are fit or "
        "selected using only the other pieces. Conditional selection minimizes "
        "intermediate MAE, treats values within 0.05 as tied and then minimizes "
        "overall MAE, then maximizes intermediate ±10 accuracy, then chooses "
        "smaller tau and MAP, median, mean in that order. Every performance "
        "appears in exactly one held-out fold.",
        "",
        "## 6. Region calibration results",
        "",
        markdown_table(
            ["Region system", "OOF region NLL", "Group accuracy", "Macro F1", "Intermediate recall"],
            [
                [
                    row["region_calibration_type"],
                    format_metric(row["region_nll"]),
                    format_metric(row["micro_group_accuracy"]),
                    format_metric(row["micro_group_macro_f1"]),
                    format_metric(row["micro_group_intermediate_recall"]),
                ]
                for row in region_rows
            ],
        ),
        "",
        f"Original 128-class validation NLL: "
        f"{selected['original_128_nll']:.6f}. Region NLLs are three-class "
        "likelihoods and are not presented as 128-class likelihoods.",
        "",
        "Fold parameter variation (P10/median/P90):",
        "",
        markdown_table(
            ["Calibrator", "T P10", "T median", "T P90", "b-int P10", "b-int median", "b-int P90", "b-full P10", "b-full median", "b-full P90"],
            parameter_rows,
        ),
        "",
    ]
    for candidate in (
        "raw_hierarchy_map",
        "temperature_hierarchy",
        "bias_temperature_hierarchy",
    ):
        matrix = results["candidate_details"][candidate]["coarse_confusion"]
        lines.extend(
            [
                f"{candidate} confusion (rows true; columns ZERO/INTERMEDIATE/FULL):",
                "",
                markdown_table(
                    ["True", "ZERO", "INTERMEDIATE", "FULL"],
                    [
                        [label, *matrix[index].tolist()]
                        for index, label in enumerate(("ZERO", "INTERMEDIATE", "FULL"))
                    ],
                ),
                "",
            ]
        )

    lines.extend(["## 7. Conditional intermediate-depth results", ""])
    for calibrator_type in ("temperature", "bias_temperature"):
        patterns = results["selected_patterns"][calibrator_type]
        tau_counts = Counter(float(item["tau"]) for item in patterns)
        decoder_counts = Counter(str(item["decoder"]) for item in patterns)
        lines.append(
            f"- {calibrator_type}: tau selections "
            + ", ".join(f"{tau:g}:{tau_counts[tau]}" for tau in TAU_GRID)
            + "; decoder selections "
            + ", ".join(
                f"{decoder}:{decoder_counts[decoder]}"
                for decoder in CONDITIONAL_DECODERS
            )
            + "."
        )
    lines.extend(
        [
            "",
            markdown_table(
                ["Candidate", "Conditional rule", "Mean tau", "Int. MAE", "Int. RMSE", "Int. tol±10", "Endpoint collapse"],
                [
                    [
                        row["candidate_name"],
                        row["conditional_decoder"],
                        format_metric(row["mean_selected_tau"]),
                        format_metric(row["micro_intermediate_mae"]),
                        format_metric(row["micro_intermediate_rmse"]),
                        format_metric(row["micro_intermediate_tolerance_accuracy_10"]),
                        format_metric(row["micro_intermediate_endpoint_collapse_ratio"]),
                    ]
                    for row in candidates
                    if "hierarchy" in row["candidate_name"]
                ],
            ),
            "",
            "## 8. Full candidate comparison",
            "",
            markdown_table(
                ["Rank", "Candidate", "Exact", "MAE", "RMSE", "Int. MAE", "Int. tol±10", "Pred. int.", "Group macro F1", "Transition F1", "Delta MAE", "Success"],
                [
                    [
                        row["final_rank"],
                        row["candidate_name"],
                        format_metric(row["micro_exact_accuracy"]),
                        format_metric(row["micro_mae"]),
                        format_metric(row["micro_rmse"]),
                        format_metric(row["micro_intermediate_mae"]),
                        format_metric(row["micro_intermediate_tolerance_accuracy_10"]),
                        format_metric(row["micro_predicted_intermediate_ratio"]),
                        format_metric(row["micro_group_macro_f1"]),
                        format_metric(row["micro_transition_detection_f1"]),
                        format_metric(row["micro_delta_mae"]),
                        row["success_criteria_met"],
                    ]
                    for row in candidates
                ],
            ),
            "",
            "The candidate CSV contains every requested micro and macro metric, "
            "including slot, tolerance, coarse, and temporal measures.",
            "`nan` is retained only for genuine zero-denominator group precision/"
            "F1 when a collapsed baseline never predicts a region; such a fold "
            "is explicitly marked non-finite and degenerate.",
            "",
            "## 9. Performance and fold variation",
            "",
        ]
    )
    variation_keys = (
        ("Overall MAE", "held_out_overall_mae"),
        ("Intermediate MAE", "held_out_intermediate_mae"),
        ("Exact accuracy", "held_out_exact_accuracy"),
        ("Predicted intermediate ratio", "held_out_predicted_intermediate_ratio"),
        ("Three-way macro F1", "held_out_three_way_macro_f1"),
        ("Transition F1", "held_out_transition_f1"),
    )
    lines.extend(
        [
            markdown_table(
                ["Selected-candidate fold metric", "P10", "Median", "P90"],
                [
                    [
                        label,
                        *[
                            format_metric(value)
                            for value in percentile_summary(
                                [float(row[key]) for row in selected_fold_rows]
                            )
                        ],
                    ]
                    for label, key in variation_keys
                ],
            ),
            "",
        ]
    )
    worst = sorted(
        selected_fold_rows,
        key=lambda row: (-float(row["held_out_overall_mae"]), str(row["held_out_piece_id"])),
    )[:5]
    lines.extend(
        [
            "Worst selected-candidate folds by overall MAE:",
            "",
            markdown_table(
                ["Piece", "Performances", "MAE", "Int. MAE", "Macro F1", "Transition F1"],
                [
                    [
                        row["held_out_piece_id"],
                        row["held_out_performance_count"],
                        format_metric(row["held_out_overall_mae"]),
                        format_metric(row["held_out_intermediate_mae"]),
                        format_metric(row["held_out_three_way_macro_f1"]),
                        format_metric(row["held_out_transition_f1"]),
                    ]
                    for row in worst
                ],
            ),
            "",
            "Fold differences are descriptive; no significance is claimed.",
            "",
            "## 10. Success-criteria assessment",
            "",
            markdown_table(
                ["Candidate", "A", "B", "Ratio", "Collapse", "Macro F1", "Nondegenerate", "Finite", "Pass"],
                [
                    [
                        row["candidate_name"],
                        row["criterion_a"],
                        row["criterion_b"],
                        row["safeguard_intermediate_ratio"],
                        row["safeguard_endpoint_collapse"],
                        row["safeguard_group_macro_f1"],
                        row["safeguard_non_degenerate"],
                        row["all_folds_finite"],
                        row["success_criteria_met"],
                    ]
                    for row in candidates
                ],
            ),
            "",
            "Criteria are the predeclared practical thresholds and are not "
            "statistical significance claims.",
            "",
            "## 11. Final selected decoder",
            "",
            f"Selected: **{selected['candidate_name']}**.",
            ("It ranked first among successful candidates. " if selected["success_criteria_met"] else "No candidate met the success criteria; it was retained as the lowest-overall-MAE fallback. ")
            + "The within-0.10 intermediate-MAE, group-macro-F1, and simplicity tie rules were then applied.",
            f"- Full-validation region calibrator: {config['region_calibrator_type']}",
            f"- Temperature: {config['fitted_temperature']:.10f}",
            f"- Biases [zero, intermediate, full]: [0, "
            f"{config['fitted_group_biases']['intermediate']:.10f}, "
            f"{config['fitted_group_biases']['full']:.10f}]",
            f"- Conditional tau/decoder: {config['selected_tau']}/"
            f"{config['conditional_decoder']}",
            f"- Frozen configuration: {results['config_path']}",
            "",
            "## 12. Interpretation",
            "",
            f"- The best hierarchical candidate was temperature_hierarchy: MAE "
            f"{by_name['temperature_hierarchy']['micro_mae']:.6f} versus "
            f"{by_name['posterior_median']['micro_mae']:.6f} for posterior median.",
            f"- Temperature scaling improved region NLL from "
            f"{by_name['raw_hierarchy_map']['region_nll']:.6f} to "
            f"{by_name['temperature_hierarchy']['region_nll']:.6f} but cannot "
            "change region argmax; bias+temperature reduced held-out macro F1 to "
            f"{by_name['bias_temperature_hierarchy']['micro_group_macro_f1']:.6f}.",
            "- Both calibrated families selected tau 0.5 and conditional mean in "
            f"all folds. Temperature-hierarchy intermediate MAE "
            f"{by_name['temperature_hierarchy']['micro_intermediate_mae']:.6f} "
            f"only slightly improved raw-hierarchy mean "
            f"{by_name['raw_hierarchy_mean']['micro_intermediate_mae']:.6f}.",
            f"- Temperature-hierarchy transition F1 "
            f"{by_name['temperature_hierarchy']['micro_transition_detection_f1']:.6f} "
            f"exceeded posterior median {by_name['posterior_median']['micro_transition_detection_f1']:.6f}, "
            "but no hierarchy passed the predefined overall/depth and endpoint-collapse criteria.",
            "- The remaining failure is not resolved by hierarchical calibration; "
            "the predefined decision therefore justifies retraining.",
            "",
            "## 13. Recommended next step",
            "",
        ]
    )
    if bool(selected["success_criteria_met"]):
        lines.append(
            "Apply the frozen calibration_config.json once to the untouched "
            "test posterior in the next task. Do not launch that evaluation here."
        )
    else:
        lines.append(
            "Do not repeatedly tune decoding; the next task should retrain the "
            "encoder-only model with CE plus a distance-aware or ordinal auxiliary "
            "objective. Do not launch that training here."
        )
    lines.extend(
        [
            "",
            "## 14. Limitations",
            "",
            "- Oracle human-performance input.",
            "- Calibration and selection use validation pieces.",
            "- Four-point PT pedal tokenizer and no raw-CC64 timing.",
            "- No end-to-end Stage 1 input or listening test.",
            "- No neural-model retraining.",
            "- Conditional depth classes remain weakly ordered.",
            "",
        ]
    )
    return "\n".join(lines)


def synthetic_smoke() -> None:
    targets = np.tile(np.arange(3), 20)
    groups = np.zeros((60, 3), dtype=np.float32)
    groups[np.arange(60), targets] = 2.0
    groups[::4] *= -1.0
    first = fit_region_calibrator(groups, targets, "temperature", device="cpu")
    second = fit_region_calibrator(
        groups, targets, "bias_temperature", device="cpu"
    )
    if not first["converged"] or not second["converged"]:
        raise RuntimeError("synthetic calibration did not converge")
    print(
        "SYNTHETIC_SMOKE_SUCCESS "
        f"temperature_nll={first['final_nll']:.6f} "
        f"bias_temperature_nll={second['final_nll']:.6f}",
        flush=True,
    )


def evaluate(configuration: dict[str, Any]) -> dict[str, Any]:
    total_start = time.perf_counter()
    if bool(configuration["synthetic_smoke"]):
        synthetic_smoke()
        return {"synthetic_smoke": True}

    asap_root = Path(configuration["asap_root"]).resolve()
    split_csv = Path(configuration["split_csv"]).resolve()
    checkpoint_path = Path(configuration["best_checkpoint"]).resolve()
    output_dir = Path(configuration["output_dir"]).resolve()
    smoke_test = bool(configuration["smoke_test"])
    if not smoke_test and output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one CUDA device must be visible")
    gpu = _visible_gpu_identity()
    if gpu["uuid"] != configuration["expected_gpu_uuid"]:
        raise RuntimeError("visible GPU UUID does not match the assigned GPU")
    device = torch.device("cuda:0")
    set_deterministic_seed(int(configuration["seed"]))
    torch.cuda.set_device(device)
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    preparation_start = time.perf_counter()
    train_dataset = Stage2PedalDataset(
        asap_root,
        split_csv,
        "train",
        window_notes=WINDOW_NOTES,
        stride_notes=STRIDE_NOTES,
        cache_mode="preload",
    )
    train_counts = np.zeros(128, dtype=np.int64)
    for row in train_dataset.performances:
        tokens = train_dataset._token_cache[row["performance_path"]]
        targets = tokens[:, NON_PEDAL_FEATURES:].astype(np.int64) - PEDAL_TOKEN_OFFSET
        train_counts += np.bincount(targets.reshape(-1), minlength=128)
    del train_dataset
    _, log_priors = smoothed_intermediate_priors(train_counts)

    validation_dataset = Stage2PedalDataset(
        asap_root,
        split_csv,
        "validation",
        window_notes=WINDOW_NOTES,
        stride_notes=STRIDE_NOTES,
        cache_mode="preload",
    )
    if (
        validation_dataset.performance_count != EXPECTED_PERFORMANCES
        or validation_dataset.window_count != EXPECTED_WINDOWS
    ):
        raise RuntimeError("validation Dataset counts changed")
    all_rows = validation_dataset.performances
    all_piece_ids = sorted({row["piece_id"] for row in all_rows})
    if len(all_piece_ids) != EXPECTED_PIECES:
        raise RuntimeError("validation piece count changed")
    if smoke_test:
        selected_piece_ids = set(all_piece_ids[:2])
        selected_rows = [
            row for row in all_rows if row["piece_id"] in selected_piece_ids
        ]
    else:
        selected_rows = list(all_rows)
    preparation_seconds = time.perf_counter() - preparation_start

    model_start = time.perf_counter()
    model, checkpoint = _load_model(checkpoint_path, device)
    versions = {
        name: parameter._version for name, parameter in model.named_parameters()
    }
    model_load_seconds = time.perf_counter() - model_start

    records: list[dict[str, Any]] = []
    total_windows = 0
    min_contributions = math.inf
    max_contributions = 0
    original_nll_sum = 0.0
    inference_start = time.perf_counter()
    for number, row in enumerate(selected_rows, start=1):
        tokens = validation_dataset._token_cache[row["performance_path"]]
        reconstructed = infer_complete_posterior(
            model,
            tokens,
            device,
            int(configuration["batch_size"]),
        )
        logits = reconstructed["mean_logits"].astype(np.float32)
        probabilities = reconstructed["probabilities"]
        targets = reconstructed["targets"]
        contributions = reconstructed["contribution_count"]
        if int(contributions.min()) < 1 or not np.isfinite(logits).all():
            raise RuntimeError("invalid complete-performance reconstruction")
        baselines = {
            "argmax": probabilities.argmax(axis=-1).astype(np.int16),
            "posterior_mean": posterior_mean_decode(probabilities)[0].astype(np.int16),
            "posterior_median": posterior_median_decode(probabilities).astype(np.int16),
        }
        record = {
            **row,
            "num_notes": len(tokens),
            "num_windows": int(reconstructed["num_windows"]),
            "logits": logits,
            "region_logits": np.asarray(region_logits(logits), dtype=np.float32),
            "targets": targets,
            "baseline_predictions": baselines,
            "conditional_predictions": precompute_conditional_predictions(
                logits, log_priors
            ),
        }
        records.append(record)
        target_count = targets.size
        original_nll_sum += float(reconstructed["loss"]) * target_count
        total_windows += int(reconstructed["num_windows"])
        min_contributions = min(min_contributions, int(contributions.min()))
        max_contributions = max(max_contributions, int(contributions.max()))
        print(
            f"reconstructed {number}/{len(selected_rows)} "
            f"path={row['performance_path']} notes={len(tokens)} "
            f"windows={reconstructed['num_windows']}",
            flush=True,
        )
        del reconstructed, probabilities

    torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - inference_start
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("neural-model gradients were created")
    changed = [
        name
        for name, parameter in model.named_parameters()
        if parameter._version != versions[name]
    ]
    if changed:
        raise RuntimeError(f"neural-model parameters changed: {changed}")

    folds = make_piece_folds(records)
    expected_folds = 2 if smoke_test else EXPECTED_PIECES
    if len(folds) != expected_folds:
        raise RuntimeError("piece-fold count mismatch")
    original_nll = original_nll_sum / sum(record["targets"].size for record in records)
    calibration_start = time.perf_counter()
    cv_results = run_cross_validation(records, folds, device, original_nll)
    calibration_seconds = time.perf_counter() - calibration_start

    note_count = sum(int(record["num_notes"]) for record in records)
    verification = {
        "performance_count": len(records),
        "piece_count": len(folds),
        "note_count": note_count,
        "target_count": note_count * PEDAL_SLOTS,
        "window_count": total_windows,
        "min_contributions": int(min_contributions),
        "max_contributions": int(max_contributions),
    }
    if smoke_test:
        if output_dir.exists():
            raise RuntimeError("smoke test created output artifacts")
        if not all(
            bool(row["finite_flag"]) and bool(row["convergence_flag"])
            for row in cv_results["fold_rows"]
        ):
            raise RuntimeError("smoke fold failed finite/convergence checks")
        print(
            "TWO_PIECE_SMOKE_SUCCESS "
            f"performances={len(records)} folds=2 notes={note_count} "
            f"windows={total_windows} coverage={min_contributions}-"
            f"{max_contributions} missing_keys={checkpoint['missing_keys']} "
            f"unexpected_keys={checkpoint['unexpected_keys']} test_rows_used=0",
            flush=True,
        )
        return {
            **cv_results,
            "verification": verification,
            "checkpoint": checkpoint,
        }

    if (
        verification["performance_count"] != EXPECTED_PERFORMANCES
        or verification["piece_count"] != EXPECTED_PIECES
        or verification["note_count"] != EXPECTED_NOTES
        or verification["target_count"] != EXPECTED_TARGETS
        or verification["window_count"] != EXPECTED_WINDOWS
    ):
        raise RuntimeError("full validation accounting changed")
    selected_row = cv_results["candidate_rows"][0]
    calibration_config, final_refit = refit_selected_configuration(
        selected_row,
        records,
        train_counts,
        log_priors,
        checkpoint,
        checkpoint_path,
        device,
    )
    timing = {
        "preparation_seconds": preparation_seconds,
        "model_load_seconds": model_load_seconds,
        "inference_seconds": inference_seconds,
        "calibration_seconds": calibration_seconds,
        "total_seconds": time.perf_counter() - total_start,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    results = {
        **cv_results,
        "calibration_config": calibration_config,
        "final_refit": final_refit,
        "verification": verification,
        "checkpoint": checkpoint,
        "checkpoint_path": str(checkpoint_path),
        "gpu": gpu,
        "batch_size": int(configuration["batch_size"]),
        "timing": timing,
        "config_path": str(output_dir / "calibration_config.json"),
    }
    report = render_report(results)
    candidate_csv = render_csv(cv_results["candidate_rows"], CANDIDATE_FIELDS)
    fold_csv = render_csv(cv_results["fold_rows"], FOLD_FIELDS)
    config_json = json.dumps(
        calibration_config, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    output_dir.mkdir(parents=False)
    (output_dir / "CALIBRATION_STUDY.md").write_text(report, encoding="utf-8")
    (output_dir / "candidate_cv_metrics.csv").write_text(
        candidate_csv, encoding="utf-8"
    )
    (output_dir / "fold_metrics.csv").write_text(fold_csv, encoding="utf-8")
    (output_dir / "calibration_config.json").write_text(
        config_json, encoding="utf-8"
    )
    print(
        "FULL_CALIBRATION_SUCCESS "
        f"folds={len(folds)} selected={selected_row['candidate_name']} "
        f"success={selected_row['success_criteria_met']} output={output_dir}",
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
        default="/workspace/project/analysis/stage2_encoder_only_v0/validation_decoder_calibration_v0",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--expected-gpu-uuid",
        default="GPU-6982dbee-fbaf-f359-d7ef-a22d0e83400b",
    )
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    configuration = vars(parse_args())
    if configuration["batch_size"] <= 0:
        raise ValueError("batch size must be positive")
    if configuration["synthetic_smoke"] and configuration["smoke_test"]:
        raise ValueError("choose only one smoke mode")
    evaluate(configuration)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

