#!/usr/bin/env python3
"""Canonical ASAP-validation-only evaluation for one phase-3 loss candidate."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_original_pt_early_metrics import PinnedTokenizerConfig
from scripts.run_stage2_4class_validation_eval_v0 import (
    ASAP_ROOT,
    PATTERN_COUNT,
    SPLIT_CSV,
    STAGE1_MANIFEST,
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
from scripts.run_stage2_encoder_only_loss_phase3_v0 import (
    STATUS_KEYS,
    canonical_config,
    objective_directory,
    read_json,
    update_objective_status,
    update_status,
)
from src.stage2_binary.canonical_stage1 import sha256_file
from src.stage2_four_class.loss_objectives import (
    HUBER_DELTA_NORMALIZED,
    OBJECTIVES,
    RepresentativeHuberEncoderModel,
    continuous_cc64_to_classes,
    normalized_scalars_to_cc64,
)
from src.stage2_four_class.model import FourClassPedalEncoderModel
from src.stage2_four_class.validation_evaluator import (
    as_note_tokens,
    classes_to_candidate_tokens,
    classification_metrics,
    confusion_from_pairs,
    infer_four_class_pedals,
    pattern_ids,
    pattern_metrics,
    pooled_transition_counts,
)
from src.stage2_encoder_only.dataset import (
    MASK_ID,
    NON_PEDAL_FEATURES,
    generate_window_starts,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def audit_validation_inputs() -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    stage1 = read_csv(STAGE1_MANIFEST)
    validation = [row for row in read_csv(SPLIT_CSV) if row["split"] == "validation"]
    if len(stage1) != 19 or len({row["piece_id"] for row in stage1}) != 19:
        raise RuntimeError("canonical Stage 1 manifest is not 19 unique pieces")
    if len(validation) != 71 or len({row["piece_id"] for row in validation}) != 19:
        raise RuntimeError("ASAP validation is not 19 pieces / 71 performances")
    stage1_pieces = {row["piece_id"] for row in stage1}
    if stage1_pieces != {row["piece_id"] for row in validation}:
        raise RuntimeError("Stage 1 and validation piece universes differ")
    by_piece: dict[str, int] = {piece: 0 for piece in stage1_pieces}
    for row in validation:
        if row["split"] != "validation":
            raise RuntimeError("non-validation row entered evaluation")
        by_piece[row["piece_id"]] += 1
    for row in stage1:
        if row["status"] != "frozen_pass":
            raise RuntimeError("canonical Stage 1 artifact is not frozen")
        if not Path(row["canonical_midi_path"]).is_file() or not Path(row["generated_ids_path"]).is_file():
            raise FileNotFoundError(f"missing canonical Stage 1 artifact: {row['piece_id']}")
        if int(row["validation_performance_count"]) != by_piece[row["piece_id"]]:
            raise RuntimeError("validation pairing count changed")
    return stage1, validation, {
        "stage1_pieces": 19,
        "validation_performances": 71,
        "asap_test_access_count": 0,
        "original_pt_neural_inference_count": 0,
        "repedal_metric_execution_count": 0,
    }


def load_model(
    config: Mapping[str, Any], objective: str, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = objective_directory(config, objective) / "best.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    base = canonical_config(config)
    common = {"torch_dtype": torch.float32, "attn_implementation": "eager"}
    if objective == "representative_huber":
        model = RepresentativeHuberEncoderModel.from_pretrained(
            base["checkpoint_path"], delta=HUBER_DELTA_NORMALIZED, **common
        )
    else:
        model = FourClassPedalEncoderModel.from_pretrained(base["checkpoint_path"], **common)
    incompatible = model.load_state_dict(payload["model_state"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"checkpoint mismatch: {incompatible}")
    if not all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
        raise FloatingPointError("checkpoint contains non-finite model parameters")
    metadata = {
        "path": str(checkpoint),
        "objective": objective,
        "epoch": int(payload["epoch"]),
        "validation_objective": float(payload["validation_objective"]),
    }
    return model.to(device).eval(), metadata


def infer_huber(
    model: torch.nn.Module,
    source_tokens: Sequence[int] | np.ndarray,
    *,
    device: torch.device,
    window_notes: int = 512,
    stride_notes: int = 256,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], np.ndarray]:
    tokens = as_note_tokens(source_tokens)
    masked = tokens.copy()
    masked[:, NON_PEDAL_FEATURES:] = MASK_ID
    starts = generate_window_starts(len(tokens), window_notes, stride_notes)
    accumulated = torch.zeros((len(tokens), 4), dtype=torch.float32, device=device)
    contributions = torch.zeros(len(tokens), dtype=torch.float32, device=device)
    with torch.inference_mode():
        for start in starts:
            end = min(start + window_notes, len(tokens))
            input_ids = torch.from_numpy(masked[start:end].reshape(1, -1)).long().to(device)
            attention = torch.ones_like(input_ids)
            note_mask = torch.ones((1, end - start), dtype=torch.bool, device=device)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                output = model(
                    input_ids=input_ids,
                    token_attention_mask=attention,
                    note_mask=note_mask,
                )
            values = output.predictions[0, : end - start].float()
            if tuple(values.shape) != (end - start, 4) or not bool(torch.isfinite(values).all()):
                raise RuntimeError("Huber inference output is invalid")
            accumulated[start:end] += values
            contributions[start:end] += 1
    if bool(torch.any(contributions == 0)):
        raise RuntimeError("one or more notes received no regression prediction")
    normalized = accumulated / contributions[:, None]
    continuous_cc64 = normalized_scalars_to_cc64(normalized)
    classes = continuous_cc64_to_classes(continuous_cc64).cpu().numpy().astype(np.int64)
    candidate = classes_to_candidate_tokens(tokens, classes)
    values = normalized.cpu().numpy().astype(np.float32)
    diagnostics = {
        "notes": len(tokens),
        "windows": len(starts),
        "window_notes": window_notes,
        "stride_notes": stride_notes,
        "overlap_merge": "unconstrained scalar mean, clip [0,1], scale 127, half-up integer conversion, canonical-bin",
        "deterministic": True,
        "ground_truth_pedal_history_access_count": 0,
        "finite_predictions": True,
        "normalized_prediction_min": float(values.min()),
        "normalized_prediction_max": float(values.max()),
        "clip_at_0_proportion": float(np.mean(values <= 0.0)),
        "clip_at_127_proportion": float(np.mean(values >= 1.0)),
    }
    return classes, candidate, diagnostics, values


def evaluate_aligned(
    config: Mapping[str, Any], objective: str,
    stage1: Sequence[Mapping[str, str]], validation: Sequence[Mapping[str, str]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    output_root = Path(config["output_root"])
    run_output = objective_directory(config, objective) / "validation"
    shared_cache = output_root / "canonical_validation_alignment_cache"
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
            try:
                humans[identifier] = align_and_cache(
                    shared_cache, kind="human", identifier=identifier, score_path=score,
                    performance_path=ASAP_ROOT / human["performance_path"],
                )
            except Exception as error:
                failures.append({"piece": piece, "human": identifier, "reason": f"{type(error).__name__}: {error}"})
        metadata = read_json(prediction_paths(run_output, objective, piece)["metadata"])
        try:
            candidates[piece] = align_and_cache(
                run_output, kind=objective, identifier=piece, score_path=score,
                performance_path=Path(metadata["candidate_midi"]),
            )
        except Exception as error:
            failures.append({"piece": piece, "candidate": objective, "reason": f"{type(error).__name__}: {error}"})
    pairs = [
        (row["piece_id"], row["metadata_index"])
        for row in validation
        if row["metadata_index"] in humans and row["piece_id"] in candidates
    ]
    if not pairs:
        raise RuntimeError("alignment produced no successful validation pairs")
    confusion = np.zeros((4, 4), dtype=np.int64)
    candidate_histogram = np.zeros(PATTERN_COUNT, dtype=np.int64)
    target_histogram = np.zeros(PATTERN_COUNT, dtype=np.int64)
    transition = {
        scope: {key: 0 for key in ("candidate", "reference", "tp", "fp", "fn")}
        for scope in ("pooled", "up", "down")
    }
    aligned_notes = 0
    from src.stage2_four_class.validation_evaluator import canonical_classes_from_tokens
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
    classification = classification_metrics(confusion)
    patterns = pattern_metrics(candidate_histogram, target_histogram)
    transition = finalize_transition(transition)
    return {
        "objective": objective,
        "candidate_pieces": len(candidates),
        "human_performances": 71,
        "successful_pairs": len(pairs),
        "aligned_notes": aligned_notes,
        "classification": classification,
        "transition": transition,
        "patterns": {
            **patterns,
            "top_candidate_patterns": top_patterns(candidate_histogram),
            "top_target_patterns": top_patterns(target_histogram),
        },
        "asap_test_access_count": 0,
        "repedal_metric_execution_count": 0,
    }, failures


def run(config: Mapping[str, Any], objective: str) -> int:
    run_output = objective_directory(config, objective) / "validation"
    run_output.mkdir(parents=True, exist_ok=True)
    log_handle = (run_output / "evaluation.log").open("w", encoding="utf-8")

    def log(message: str) -> None:
        print(message, flush=True)
        print(message, file=log_handle, flush=True)

    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("evaluation requires exactly one visible CUDA GPU")
        stage1, validation, audit = audit_validation_inputs()
        device = torch.device("cuda:0")
        model, checkpoint = load_model(config, objective, device)
        checkpoint_hash = sha256_file(checkpoint["path"])
        tokenizer = PinnedTokenizerConfig()
        diagnostics: list[dict[str, Any]] = []
        for index, row in enumerate(stage1, 1):
            source = np.load(row["generated_ids_path"], allow_pickle=False)
            if ids_sha256(source) != row["generated_token_sha256_int64_le"]:
                raise RuntimeError("canonical Stage 1 token hash changed")
            if objective == "representative_huber":
                classes, candidate, inference, continuous = infer_huber(model, source, device=device)
                paths = prediction_paths(run_output, objective, row["piece_id"])
                atomic_npy(paths["root"] / "continuous_normalized_float32.npy", continuous)
                diagnostics.append(inference)
            else:
                classes, candidate, inference = infer_four_class_pedals(
                    model, source, architecture="encoder_only", device=device,
                )
            render_stage2_candidate(
                run_output, objective, row, classes, candidate, inference,
                checkpoint_hash=checkpoint_hash,
                tokenizer_config=tokenizer,
            )
            update_objective_status(config, objective, current_stage="validation_inference", current_piece=index, pieces_total=19)
            if index == 1:
                log(f"FIRST_VALIDATION_CANDIDATE_PASS objective={objective} test_access=0")
        metrics, failures = evaluate_aligned(config, objective, stage1, validation)
        metrics.update(checkpoint=checkpoint, input_audit=audit, alignment_failures=failures)
        if diagnostics:
            sample_counts = [item["notes"] * 4 for item in diagnostics]
            total = sum(sample_counts)
            metrics["regression_diagnostics"] = {
                "normalized_prediction_min": min(item["normalized_prediction_min"] for item in diagnostics),
                "normalized_prediction_max": max(item["normalized_prediction_max"] for item in diagnostics),
                "clip_at_0_proportion": sum(item["clip_at_0_proportion"] * count for item, count in zip(diagnostics, sample_counts)) / total,
                "clip_at_127_proportion": sum(item["clip_at_127_proportion"] * count for item, count in zip(diagnostics, sample_counts)) / total,
                "midi_integer_conversion": "nearest integer, half up",
            }
        atomic_json(run_output / "evaluation.json", metrics)
        atomic_json(run_output / "alignment_failures.json", {"failures": failures})
        log(f"VALIDATION_EVALUATION_COMPLETE objective={objective} pairs={metrics['successful_pairs']} test_access=0")
        return 0
    except Exception as error:
        log(f"VALIDATION_EVALUATION_FAILED {type(error).__name__}: {error}")
        traceback.print_exc(file=log_handle)
        update_objective_status(
            config, objective, "failed", current_stage="validation_evaluation_failed",
            error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc(),
        )
        return 1
    finally:
        log_handle.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--objective", choices=OBJECTIVES, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("evaluation is guarded; pass --execute")
    return run(read_json(args.config), args.objective)


if __name__ == "__main__":
    raise SystemExit(main())
