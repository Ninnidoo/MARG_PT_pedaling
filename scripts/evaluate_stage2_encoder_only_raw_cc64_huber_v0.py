#!/usr/bin/env python3
"""Frozen canonical validation evaluation for the raw-CC64 Huber candidate."""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_original_pt_early_metrics import PinnedTokenizerConfig
from scripts.evaluate_stage2_encoder_only_loss_phase3_v0 import (
    audit_validation_inputs,
    infer_huber,
)
from scripts.run_stage2_4class_validation_eval_v0 import (
    ASAP_ROOT,
    PATTERN_COUNT,
    align_and_cache,
    atomic_json,
    atomic_npy,
    finalize_transition,
    ids_sha256,
    prediction_paths,
    render_stage2_candidate,
    sum_transition,
    top_patterns,
)
from scripts.run_stage2_encoder_only_raw_cc64_huber_v0 import (
    canonical_config,
    make_raw_cc64_loader,
    read_json,
    update_status,
)
from src.stage2_binary.canonical_stage1 import sha256_file
from src.stage2_binary.full_training import SharedBinaryWindowDataset
from src.stage2_four_class.raw_cc64_huber import RawCC64HuberEncoderModel
from src.stage2_four_class.validation_evaluator import (
    canonical_classes_from_tokens,
    classification_metrics,
    confusion_from_pairs,
    pattern_ids,
    pattern_metrics,
    pooled_transition_counts,
)


OBJECTIVE = "raw_cc64_huber"


def load_best_model(
    config: Mapping[str, Any], device: torch.device
) -> tuple[RawCC64HuberEncoderModel, dict[str, Any]]:
    checkpoint = Path(config["output_root"]) / "best.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    base = canonical_config(config)
    model = RawCC64HuberEncoderModel.from_pretrained(
        base["checkpoint_path"],
        delta=float(config["huber_delta_normalized"]),
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    incompatible = model.load_state_dict(payload["model_state"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"raw Huber checkpoint mismatch: {incompatible}")
    if not all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
        raise FloatingPointError("raw Huber checkpoint contains non-finite parameters")
    metadata = {
        "path": str(checkpoint),
        "sha256": sha256_file(checkpoint),
        "epoch": int(payload["epoch"]),
        "validation_objective": float(payload["validation_objective"]),
        "validation_raw_cc64_mae": float(payload["validation_raw_cc64_mae"]),
        "initialization_source": base["checkpoint_path"],
        "representative_checkpoint_used_for_initialization": False,
    }
    return model.to(device).eval(), metadata


def regression_diagnostics(
    config: Mapping[str, Any], model: RawCC64HuberEncoderModel, device: torch.device
) -> dict[str, Any]:
    base = canonical_config(config)
    dataset = SharedBinaryWindowDataset(base["cache_root"], "validation")
    loader = make_raw_cc64_loader(
        dataset,
        batch_size=int(base["micro_batch_size"]["encoder_only"]),
        pin_memory=bool(base["pin_memory"]),
    )
    predictions: list[np.ndarray] = []
    predictions_unclipped: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.inference_mode():
        for cpu_batch in loader:
            batch = {
                key: value.to(device, non_blocking=True)
                if isinstance(value, torch.Tensor)
                else value
                for key, value in cpu_batch.items()
            }
            with torch.amp.autocast(
                device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"
            ):
                output = model(
                    input_ids=batch["input_ids"],
                    token_attention_mask=batch["token_attention_mask"],
                    note_mask=batch["note_mask"],
                )
            valid = batch["pedal_targets"] != -100
            raw_unclipped = output.predictions[valid].float() * 127.0
            raw_prediction = raw_unclipped.clamp(0.0, 127.0)
            raw_target = batch["pedal_targets"][valid].float()
            if not bool(torch.isfinite(raw_unclipped).all()):
                raise FloatingPointError("non-finite raw validation prediction")
            predictions.append(raw_prediction.cpu().numpy())
            predictions_unclipped.append(raw_unclipped.cpu().numpy())
            targets.append(raw_target.cpu().numpy())
    predicted = np.concatenate(predictions).astype(np.float64)
    unclipped = np.concatenate(predictions_unclipped).astype(np.float64)
    target = np.concatenate(targets).astype(np.float64)
    error = predicted - target
    absolute = np.abs(error)
    quantile_names = ("p01", "p05", "p25", "p50", "p75", "p95", "p99")
    quantile_values = np.quantile(predicted, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    ranges = ((0, 25, "ZERO"), (26, 63, "LOW"), (64, 103, "HALF"), (104, 127, "FULL"))
    by_range: dict[str, Any] = {}
    for lower, upper, name in ranges:
        mask = (target >= lower) & (target <= upper)
        by_range[name] = {
            "range": [lower, upper],
            "count": int(mask.sum()),
            "mae": float(absolute[mask].mean()),
            "rmse": float(np.sqrt(np.mean(error[mask] ** 2))),
        }
    return {
        "universe": "canonical ASAP validation cache windows; overlapping-window weighted",
        "target_pipeline": "cached raw integer CC64 / 127; no class/representative quantization",
        "samples": int(len(target)),
        "raw_cc64_mae": float(absolute.mean()),
        "raw_cc64_rmse": float(np.sqrt(np.mean(error ** 2))),
        "prediction_mean": float(predicted.mean()),
        "prediction_std": float(predicted.std()),
        "target_mean": float(target.mean()),
        "target_std": float(target.std()),
        "prediction_quantiles": dict(zip(quantile_names, quantile_values.tolist())),
        "raw_prediction_minimum_before_clipping": float(unclipped.min()),
        "raw_prediction_maximum_before_clipping": float(unclipped.max()),
        "clip_at_0_proportion": float(np.mean(unclipped <= 0.0)),
        "clip_at_127_proportion": float(np.mean(unclipped >= 127.0)),
        "absolute_error_by_target_range": by_range,
        "asap_test_access_count": 0,
    }


def confusion_error_diagnostics(confusion: Any) -> dict[str, Any]:
    matrix = np.asarray(confusion, dtype=np.int64)
    directional = {
        "ZERO_to_LOW": int(matrix[0, 1]),
        "LOW_to_ZERO": int(matrix[1, 0]),
        "LOW_to_HALF": int(matrix[1, 2]),
        "HALF_to_LOW": int(matrix[2, 1]),
        "HALF_to_FULL": int(matrix[2, 3]),
        "FULL_to_HALF": int(matrix[3, 2]),
    }
    crossing_low_to_high = int(matrix[:2, 2:].sum())
    crossing_high_to_low = int(matrix[2:, :2].sum())
    cross_pairs = {
        f"{source}_to_{target}": int(matrix[source_index, target_index])
        for source_index, source in enumerate(("ZERO", "LOW", "HALF", "FULL"))
        for target_index, target in enumerate(("ZERO", "LOW", "HALF", "FULL"))
        if (source_index < 2) != (target_index < 2)
    }
    return {
        "adjacent_directional_counts": directional,
        "adjacent_undirected_counts": {
            "ZERO_LOW": directional["ZERO_to_LOW"] + directional["LOW_to_ZERO"],
            "LOW_HALF": directional["LOW_to_HALF"] + directional["HALF_to_LOW"],
            "HALF_FULL": directional["HALF_to_FULL"] + directional["FULL_to_HALF"],
        },
        "same_on_off_side_error": {
            "ZERO_LOW": directional["ZERO_to_LOW"] + directional["LOW_to_ZERO"],
            "HALF_FULL": directional["HALF_to_FULL"] + directional["FULL_to_HALF"],
            "total": int(matrix[0, 1] + matrix[1, 0] + matrix[2, 3] + matrix[3, 2]),
        },
        "boundary_64_crossing_error": {
            "low_side_to_high_side": crossing_low_to_high,
            "high_side_to_low_side": crossing_high_to_low,
            "total": crossing_low_to_high + crossing_high_to_low,
            "directional_pair_counts": cross_pairs,
        },
    }


def evaluate_aligned(
    config: Mapping[str, Any], stage1: list[dict[str, str]],
    validation: list[dict[str, str]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    output = Path(config["output_root"]) / "validation_eval"
    shared_cache = Path(config["representative_validation_alignment_cache"])
    baseline = read_json(config["representative_huber_evaluation"])
    excluded_humans = {
        item["human"] for item in baseline["alignment_failures"] if "human" in item
    }
    humans: dict[str, dict[str, Any]] = {}
    candidates: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    validation_by_piece: dict[str, list[Mapping[str, str]]] = {}
    for row in validation:
        validation_by_piece.setdefault(row["piece_id"], []).append(row)
    for row in stage1:
        piece = row["piece_id"]
        score = Path(row["selected_score_absolute_path"])
        for human in validation_by_piece[piece]:
            identifier = human["metadata_index"]
            if identifier in excluded_humans:
                continue
            try:
                humans[identifier] = align_and_cache(
                    shared_cache, kind="human", identifier=identifier, score_path=score,
                    performance_path=ASAP_ROOT / human["performance_path"],
                )
            except Exception as error:
                failures.append({"piece": piece, "human": identifier, "reason": f"{type(error).__name__}: {error}"})
        metadata = read_json(prediction_paths(output, OBJECTIVE, piece)["metadata"])
        try:
            candidates[piece] = align_and_cache(
                output, kind=OBJECTIVE, identifier=piece, score_path=score,
                performance_path=Path(metadata["candidate_midi"]),
            )
        except Exception as error:
            failures.append({"piece": piece, "candidate": OBJECTIVE, "reason": f"{type(error).__name__}: {error}"})
    pairs = [
        (row["piece_id"], row["metadata_index"])
        for row in validation
        if row["metadata_index"] not in excluded_humans
        and row["metadata_index"] in humans and row["piece_id"] in candidates
    ]
    if len(pairs) != int(baseline["successful_pairs"]):
        raise RuntimeError("raw and Representative Huber evaluation pair universes differ")
    confusion = np.zeros((4, 4), dtype=np.int64)
    candidate_histogram = np.zeros(PATTERN_COUNT, dtype=np.int64)
    target_histogram = np.zeros(PATTERN_COUNT, dtype=np.int64)
    transition = {
        scope: {key: 0 for key in ("candidate", "reference", "tp", "fp", "fn")}
        for scope in ("pooled", "up", "down")
    }
    aligned_notes = 0
    for piece, identifier in pairs:
        candidate = candidates[piece]
        target = humans[identifier]
        candidate_classes = canonical_classes_from_tokens(candidate["tokens"])
        target_classes = canonical_classes_from_tokens(target["tokens"])
        if candidate_classes.shape != target_classes.shape:
            raise RuntimeError("aligned candidate/reference shapes differ")
        confusion += confusion_from_pairs(candidate_classes, target_classes)
        candidate_histogram += np.bincount(pattern_ids(candidate_classes), minlength=PATTERN_COUNT)
        target_histogram += np.bincount(pattern_ids(target_classes), minlength=PATTERN_COUNT)
        sum_transition(transition, pooled_transition_counts(candidate["transitions"], target["transitions"]))
        aligned_notes += len(candidate_classes)
    if aligned_notes != int(baseline["aligned_notes"]):
        raise RuntimeError("raw and Representative Huber aligned-note universes differ")
    classification = classification_metrics(confusion)
    classification["error_diagnostics"] = confusion_error_diagnostics(confusion)
    patterns = pattern_metrics(candidate_histogram, target_histogram)
    return {
        "objective": OBJECTIVE,
        "candidate_pieces": len(candidates),
        "human_performances": 71,
        "successful_pairs": len(pairs),
        "excluded_baseline_alignment_ids": sorted(excluded_humans),
        "aligned_notes": aligned_notes,
        "classification": classification,
        "transition": finalize_transition(transition),
        "patterns": {
            **patterns,
            "top_candidate_patterns": top_patterns(candidate_histogram),
            "top_target_patterns": top_patterns(target_histogram),
        },
        "asap_test_access_count": 0,
        "repedal_metric_execution_count": 0,
    }, failures


def run(config: Mapping[str, Any]) -> int:
    output = Path(config["output_root"]) / "validation_eval"
    output.mkdir(parents=True, exist_ok=True)
    log_handle = (output / "evaluation.log").open("w", encoding="utf-8")

    def log(message: str) -> None:
        print(message, flush=True)
        print(message, file=log_handle, flush=True)

    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("evaluation requires exactly one visible CUDA GPU")
        stage1, validation, audit = audit_validation_inputs()
        if audit["asap_test_access_count"] != 0:
            raise RuntimeError("ASAP test access changed")
        device = torch.device("cuda:0")
        model, checkpoint = load_best_model(config, device)
        regression = regression_diagnostics(config, model, device)
        atomic_json(output / "regression_diagnostics.json", regression)
        tokenizer = PinnedTokenizerConfig()
        inference_diagnostics: list[dict[str, Any]] = []
        for index, row in enumerate(stage1, 1):
            source = np.load(row["generated_ids_path"], allow_pickle=False)
            if ids_sha256(source) != row["generated_token_sha256_int64_le"]:
                raise RuntimeError("canonical Stage 1 token hash changed")
            classes, candidate, inference, continuous = infer_huber(
                model, source, device=device
            )
            paths = prediction_paths(output, OBJECTIVE, row["piece_id"])
            atomic_npy(paths["root"] / "continuous_normalized_float32.npy", continuous)
            render_stage2_candidate(
                output, OBJECTIVE, row, classes, candidate, inference,
                checkpoint_hash=checkpoint["sha256"], tokenizer_config=tokenizer,
            )
            inference_diagnostics.append(inference)
            update_status(
                config, status="running", current_stage="validation_inference",
                current_piece=index, pieces_total=19,
            )
            if index == 1:
                log("FIRST_VALIDATION_CANDIDATE_PASS objective=raw_cc64_huber test_access=0")
        metrics, failures = evaluate_aligned(config, stage1, validation)
        metrics.update(
            checkpoint=checkpoint,
            input_audit=audit,
            alignment_failures=failures,
            regression_diagnostics=regression,
            inference_conversion=config["inference_conversion"],
        )
        atomic_json(output / "evaluation.json", metrics)
        atomic_json(output / "alignment_failures.json", {"failures": failures})
        update_status(
            config, status="running", current_stage="validation_evaluation_completed",
            evaluation_completed=True,
        )
        log(f"VALIDATION_EVALUATION_COMPLETE pairs={metrics['successful_pairs']} test_access=0")
        return 0
    except Exception as error:
        log(f"VALIDATION_EVALUATION_FAILED {type(error).__name__}: {error}")
        traceback.print_exc(file=log_handle)
        update_status(
            config, status="failed", current_stage="validation_evaluation_failed",
            error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc(),
        )
        return 1
    finally:
        log_handle.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("evaluation is guarded; pass --execute")
    return run(read_json(args.config))


if __name__ == "__main__":
    raise SystemExit(main())
