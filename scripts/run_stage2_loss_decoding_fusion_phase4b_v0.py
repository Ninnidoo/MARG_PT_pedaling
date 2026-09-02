#!/usr/bin/env python3
"""Decoding/fusion-only comparison over frozen Phase-3/Hybrid checkpoints."""

from __future__ import annotations

import argparse
import gc
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
    SPLIT_CSV,
    STAGE1_MANIFEST,
    align_and_cache,
    finalize_transition,
    prediction_paths,
    render_stage2_candidate,
    sum_transition,
    top_patterns,
)
from scripts.run_stage2_encoder_only_decoding_phase4_v0 import (
    atomic_json,
    atomic_npy,
    array_hash,
    direction_matrix,
    infer_final_logits,
    load_human_universe,
    now,
    read_csv,
    update_status,
)
from src.stage2_binary.canonical_stage1 import sha256_file
from src.stage2_binary.validation_evaluator import replace_pedal_tokens
from src.stage2_encoder_only.dataset import (
    MASK_ID,
    NON_PEDAL_FEATURES,
    generate_window_starts,
)
from src.stage2_four_class.decoding_strategies import (
    decode_argmax,
    decode_auxiliary_median,
    decode_ordered_median,
    decode_side_constrained,
    softmax_final_logits,
    stable_top2_indices,
)
from src.stage2_four_class.loss_objectives import (
    HUBER_DELTA_NORMALIZED,
    normalized_scalars_to_cc64,
)
from src.stage2_four_class.model import FourClassPedalEncoderModel
from src.stage2_four_class.raw_huber_aux_ce import (
    LAMBDA_CE,
    RawHuberAuxCEEncoderModel,
)
from src.stage2_four_class.representation import REPRESENTATIVES
from src.stage2_four_class.validation_evaluator import (
    PATTERN_COUNT,
    as_note_tokens,
    canonical_classes_from_tokens,
    classes_to_candidate_tokens,
    classification_metrics,
    confusion_from_pairs,
    pattern_ids,
    pattern_metrics,
    pooled_transition_counts,
)

DEFAULT_CONFIG = ROOT / "configs/stage2_loss_decoding_fusion_phase4b_v0.json"
METHODS = (
    "weighted_ce_median",
    "ntl_was_median",
    "hybrid_aux_median",
    "hybrid_side_constrained",
)
LABELS = {
    "weighted_ce_median": "Weighted CE / Median",
    "ntl_was_median": "NTL-WAS / Median",
    "hybrid_aux_median": "Hybrid Aux / Median",
    "hybrid_side_constrained": "Hybrid Side-Constrained Fusion",
}
EXPECTED_PAIRS = 70
EXPECTED_ALIGNED_NOTES = 272053
CLASS_NAMES = ("ZERO", "LOW", "HALF", "FULL")


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def load_classification_model(
    checkpoint: Path, pretrained: Path, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = FourClassPedalEncoderModel.from_pretrained(
        pretrained, torch_dtype=torch.float32, attn_implementation="eager"
    )
    incompatible = model.load_state_dict(payload["model_state"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"classification checkpoint mismatch: {incompatible}")
    if not all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
        raise FloatingPointError("classification checkpoint contains non-finite parameters")
    return model.to(device).eval(), {
        "path": str(checkpoint),
        "sha256": sha256_file(checkpoint),
        "epoch": int(payload["epoch"]),
        "validation_objective": float(payload["validation_objective"]),
    }


def load_hybrid_model(
    checkpoint: Path, pretrained: Path, device: torch.device
) -> tuple[RawHuberAuxCEEncoderModel, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = RawHuberAuxCEEncoderModel.from_pretrained(
        pretrained,
        delta=HUBER_DELTA_NORMALIZED,
        lambda_ce=LAMBDA_CE,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    incompatible = model.load_state_dict(payload["model_state"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Hybrid checkpoint mismatch: {incompatible}")
    if not all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
        raise FloatingPointError("Hybrid checkpoint contains non-finite parameters")
    return model.to(device).eval(), {
        "path": str(checkpoint),
        "sha256": sha256_file(checkpoint),
        "epoch": int(payload["epoch"]),
        "validation_objective": float(payload["validation_objective"]),
        "validation_huber": float(payload["validation_huber"]),
        "validation_ce": float(payload["validation_ce"]),
    }


def infer_hybrid_final_outputs(
    model: torch.nn.Module,
    source_tokens: np.ndarray,
    *,
    device: torch.device,
    window_notes: int,
    stride_notes: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    tokens = as_note_tokens(source_tokens)
    masked = tokens.copy()
    masked[:, NON_PEDAL_FEATURES:] = MASK_ID
    starts = generate_window_starts(len(tokens), window_notes, stride_notes)
    regression_sum = torch.zeros((len(tokens), 4), dtype=torch.float32, device=device)
    logits_sum = torch.zeros((len(tokens), 4, 4), dtype=torch.float32, device=device)
    contributions = torch.zeros(len(tokens), dtype=torch.float32, device=device)
    with torch.inference_mode():
        for start in starts:
            end = min(start + window_notes, len(tokens))
            input_ids = torch.from_numpy(masked[start:end].reshape(1, -1)).long().to(device)
            attention = torch.ones_like(input_ids)
            note_mask = torch.ones((1, end - start), dtype=torch.bool, device=device)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                output = model(
                    input_ids=input_ids,
                    token_attention_mask=attention,
                    note_mask=note_mask,
                )
            regression = output.predictions[0, : end - start].float()
            auxiliary = output.auxiliary_logits[0, : end - start].float()
            if tuple(regression.shape) != (end - start, 4):
                raise RuntimeError("Hybrid regression shape changed")
            if tuple(auxiliary.shape) != (end - start, 4, 4):
                raise RuntimeError("Hybrid auxiliary-logit shape changed")
            if not bool(torch.isfinite(regression).all() and torch.isfinite(auxiliary).all()):
                raise FloatingPointError("Hybrid final output is non-finite")
            regression_sum[start:end] += regression
            logits_sum[start:end] += auxiliary
            contributions[start:end] += 1
    if bool(torch.any(contributions == 0)):
        raise AssertionError("Hybrid final output has an uncovered note")
    regression = (regression_sum / contributions[:, None]).cpu().numpy().astype(np.float32)
    logits = (logits_sum / contributions[:, None, None]).cpu().numpy().astype(np.float32)
    return regression, logits, {
        "notes": len(tokens),
        "windows": len(starts),
        "window_notes": window_notes,
        "stride_notes": stride_notes,
        "overlap_merge": "regression scalars and auxiliary logits independently averaged before decoding",
        "finite": True,
    }


def regression_cc64_integer(normalized: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(normalized, dtype=np.float32))
    continuous = normalized_scalars_to_cc64(tensor)
    return torch.floor(continuous + 0.5).long().numpy().astype(np.int64)


def audit(
    config: Mapping[str, Any],
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any], dict[str, Any]]:
    for key in (
        "training_steps",
        "optimizer_steps",
        "checkpoint_changes",
        "asap_test_access_count",
        "repedal_execution_count",
    ):
        if int(config[key]) != 0:
            raise RuntimeError(f"forbidden nonzero configuration: {key}")
    if list(config["representatives"]) != list(REPRESENTATIVES):
        raise RuntimeError("canonical representatives changed")
    if float(config["ntl_lambda"]) != 0.3 or float(config["hybrid_lambda_ce"]) != 0.1:
        raise RuntimeError("frozen objective constants changed")
    if not math.isclose(
        float(config["huber_delta_normalized"]),
        HUBER_DELTA_NORMALIZED,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise RuntimeError("Hybrid Huber delta changed")
    stage1 = read_csv(STAGE1_MANIFEST)
    validation = [row for row in read_csv(SPLIT_CSV) if row["split"] == "validation"]
    if len(stage1) != 19 or len(validation) != 71:
        raise RuntimeError("canonical validation counts changed")
    if {row["piece_id"] for row in stage1} != {row["piece_id"] for row in validation}:
        raise RuntimeError("Stage 1/validation universes differ")
    references = {
        "weighted_ce_argmax": load_json(config["weighted_ce_evaluation"]),
        "ntl_was_argmax": load_json(config["ntl_was_evaluation"]),
        "hybrid_regression_only": load_json(config["hybrid_evaluation"]),
        "standard_ce_phase4": load_json(
            Path(config["standard_ce_phase4_root"]) / "metrics_all.json"
        ),
    }
    paths = (
        "weighted_ce_checkpoint",
        "ntl_was_checkpoint",
        "hybrid_checkpoint",
        "pretrained_checkpoint",
    )
    for key in paths:
        if not Path(config[key]).exists():
            raise FileNotFoundError(f"missing read-only artifact: {config[key]}")
    hybrid = references["hybrid_regression_only"]
    if hybrid["successful_pairs"] != EXPECTED_PAIRS or hybrid["aligned_notes"] != EXPECTED_ALIGNED_NOTES:
        raise RuntimeError("Hybrid canonical evaluation universe changed")
    provenance = {
        "training_steps": 0,
        "optimizer_steps": 0,
        "checkpoint_changes": 0,
        "asap_test_access_count": 0,
        "repedal_execution_count": 0,
        "validation_pieces": 19,
        "validation_humans": 71,
        "frozen_successful_pairs": EXPECTED_PAIRS,
        "aligned_notes": EXPECTED_ALIGNED_NOTES,
    }
    return stage1, validation, references, provenance


def cache_classification_logits(
    output: Path,
    name: str,
    model: torch.nn.Module,
    checkpoint: Mapping[str, Any],
    stage1: Sequence[Mapping[str, str]],
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    manifest = []
    for row in stage1:
        piece = row["piece_id"]
        source = np.load(row["generated_ids_path"], allow_pickle=False)
        logits, inference = infer_final_logits(
            model,
            source,
            device=device,
            window_notes=int(config["window_notes"]),
            stride_notes=int(config["stride_notes"]),
        )
        path = output / "final_logit_cache" / name / piece / "final_logits_float32.npy"
        atomic_npy(path, logits)
        values[piece] = logits
        manifest.append({
            "piece_id": piece,
            "path": str(path),
            "shape": list(logits.shape),
            "sha256": array_hash(logits),
            "checkpoint_sha256": checkpoint["sha256"],
            "inference": inference,
        })
    atomic_json(
        output / "final_logit_cache" / name / "manifest.json",
        {"model": name, "overlap_logits_before_decoding": True, "pieces": manifest},
    )
    return values


def median_diagnostics(posteriors: Sequence[np.ndarray]) -> dict[str, Any]:
    flat = np.concatenate([value.reshape(-1, 4) for value in posteriors], axis=0)
    argmax = decode_argmax(flat)
    median = decode_ordered_median(flat)
    top2 = stable_top2_indices(flat)
    low = (top2 == 1).any(axis=1) & (argmax != 1)
    half = (top2 == 2).any(axis=1) & (argmax != 2)
    return {
        "pedal_samples_once_per_stage1_note_slot": len(flat),
        "median_differs_from_argmax_fraction": float(np.mean(median != argmax)),
        "argmax_to_median_change_matrix": direction_matrix(argmax, median),
        "low_in_top2_but_not_argmax_fraction": float(low.mean()),
        "half_in_top2_but_not_argmax_fraction": float(half.mean()),
        "low_or_half_in_top2_but_not_argmax_fraction": float((low | half).mean()),
    }


def render_method(
    output: Path,
    method: str,
    row: Mapping[str, str],
    classes: np.ndarray,
    source: np.ndarray,
    checkpoint: Mapping[str, Any],
    cache_hash: str,
    tokenizer: Any,
) -> None:
    candidate = classes_to_candidate_tokens(source, classes)
    render_stage2_candidate(
        output,
        method,
        row,
        classes,
        candidate,
        {
            "decoder": LABELS[method],
            "overlap_logits_before_decoding": True,
            "final_cache_sha256": cache_hash,
            "non_pedal_tokens_preserved": True,
            "training_steps": 0,
        },
        checkpoint["sha256"],
        tokenizer,
    )


def candidate_alignment(
    output: Path,
    method: str,
    stage1: Sequence[Mapping[str, str]],
) -> dict[str, Mapping[str, Any]]:
    result = {}
    for row in stage1:
        piece = row["piece_id"]
        metadata = load_json(prediction_paths(output, method, piece)["metadata"])
        result[piece] = align_and_cache(
            output,
            kind=method,
            identifier=piece,
            score_path=Path(row["selected_score_absolute_path"]),
            performance_path=Path(metadata["candidate_midi"]),
        )
    return result


def evaluate_candidate(
    method: str,
    candidates: Mapping[str, Mapping[str, Any]],
    validation: Sequence[Mapping[str, str]],
    humans: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    confusion = np.zeros((4, 4), dtype=np.int64)
    candidate_histogram = np.zeros(PATTERN_COUNT, dtype=np.int64)
    target_histogram = np.zeros(PATTERN_COUNT, dtype=np.int64)
    transition = {
        scope: {key: 0 for key in ("candidate", "reference", "tp", "fp", "fn")}
        for scope in ("pooled", "up", "down")
    }
    aligned_notes = 0
    pairs = 0
    for row in validation:
        identifier = row["metadata_index"]
        if identifier not in humans:
            continue
        candidate = candidates[row["piece_id"]]
        target = humans[identifier]
        predicted = canonical_classes_from_tokens(candidate["tokens"])
        reference = canonical_classes_from_tokens(target["tokens"])
        if predicted.shape != reference.shape:
            raise RuntimeError("candidate/reference aligned shapes differ")
        confusion += confusion_from_pairs(predicted, reference)
        candidate_histogram += np.bincount(pattern_ids(predicted), minlength=PATTERN_COUNT)
        target_histogram += np.bincount(pattern_ids(reference), minlength=PATTERN_COUNT)
        sum_transition(
            transition,
            pooled_transition_counts(candidate["transitions"], target["transitions"]),
        )
        aligned_notes += len(predicted)
        pairs += 1
    if pairs != EXPECTED_PAIRS or aligned_notes != EXPECTED_ALIGNED_NOTES:
        raise AssertionError("frozen 70-pair evaluation universe drifted")
    patterns = pattern_metrics(candidate_histogram, target_histogram)
    return {
        "method": LABELS[method],
        "candidate_pieces": 19,
        "human_performances": 71,
        "successful_pairs": pairs,
        "aligned_notes": aligned_notes,
        "classification": classification_metrics(confusion),
        "transition": finalize_transition(transition),
        "patterns": {
            **patterns,
            "top_candidate_patterns": top_patterns(candidate_histogram),
            "top_target_patterns": top_patterns(target_histogram),
        },
        "asap_test_access_count": 0,
        "repedal_execution_count": 0,
    }


def transition_signature(value: Mapping[str, Any]) -> list[tuple[Any, ...]]:
    return [
        (
            event["direction"],
            int(event["onset_index"]),
            int(event["score_position"]),
            int(event["tick"]),
        )
        for event in value["transitions"]
    ]


def assert_hybrid_transition_invariance(
    config: Mapping[str, Any],
    stage1: Sequence[Mapping[str, str]],
    fusion: Mapping[str, Mapping[str, Any]],
    result: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(config["hybrid_validation_root"])
    per_piece = {}
    for row in stage1:
        piece = row["piece_id"]
        control = load_json(root / "alignment/raw_huber_aux_ce" / piece / "transitions.json")
        candidate = fusion[piece]["transitions"]
        control_signature = transition_signature(control)
        fusion_signature = transition_signature(candidate)
        if fusion_signature != control_signature:
            raise AssertionError(f"Hybrid transition locations changed: {piece}")
        per_piece[piece] = {
            "candidate_transitions": len(fusion_signature),
            "up": sum(item[0] == "UP" for item in fusion_signature),
            "down": sum(item[0] == "DOWN" for item in fusion_signature),
            "locations_exact": True,
        }
    if result["transition"] != reference["transition"]:
        raise AssertionError("canonical Hybrid transition P/R/F1/counts changed")
    return {
        "per_sample_off_on_exact": True,
        "candidate_transition_locations_exact": True,
        "up_locations_exact": True,
        "down_locations_exact": True,
        "canonical_transition_metrics_exact": True,
        "candidate_count": result["transition"]["pooled"]["candidate"],
        "per_piece": per_piece,
    }


def binary_confusion(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.bincount(target.reshape(-1) * 2 + predicted.reshape(-1), minlength=4).reshape(2, 2)


def within_side_diagnostics(
    config: Mapping[str, Any],
    fusion: Mapping[str, Mapping[str, Any]],
    validation: Sequence[Mapping[str, str]],
    humans: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    control_root = Path(config["hybrid_validation_root"]) / "alignment/raw_huber_aux_ce"
    side_correct_count = total = 0
    control_off = np.zeros((2, 2), dtype=np.int64)
    fusion_off = np.zeros((2, 2), dtype=np.int64)
    control_on = np.zeros((2, 2), dtype=np.int64)
    fusion_on = np.zeros((2, 2), dtype=np.int64)
    change = np.zeros((4, 4), dtype=np.int64)
    for row in validation:
        identifier = row["metadata_index"]
        if identifier not in humans:
            continue
        piece = row["piece_id"]
        control_tokens = np.load(
            control_root / piece / "aligned_tokens_int64.npy", allow_pickle=False
        )
        control = canonical_classes_from_tokens(control_tokens)
        fused = canonical_classes_from_tokens(fusion[piece]["tokens"])
        target = canonical_classes_from_tokens(humans[identifier]["tokens"])
        if control.shape != fused.shape or fused.shape != target.shape:
            raise AssertionError("within-side diagnostic shape mismatch")
        if not np.array_equal(control >= 2, fused >= 2):
            raise AssertionError("aligned fusion changed an OFF/ON bit")
        change += confusion_from_pairs(fused, control)
        side_correct = (control >= 2) == (target >= 2)
        side_correct_count += int(side_correct.sum())
        total += int(side_correct.size)
        off = side_correct & (target < 2)
        on = side_correct & (target >= 2)
        control_off += binary_confusion(control[off], target[off])
        fusion_off += binary_confusion(fused[off], target[off])
        control_on += binary_confusion(control[on] - 2, target[on] - 2)
        fusion_on += binary_confusion(fused[on] - 2, target[on] - 2)
    cross_side_changes = int(change[:2, 2:].sum() + change[2:, :2].sum())
    if cross_side_changes != 0:
        raise AssertionError("fusion produced cross-side class changes")

    def accuracy(matrix: np.ndarray) -> float:
        return float(np.trace(matrix) / matrix.sum())

    return {
        "on_off_side_accuracy": side_correct_count / total,
        "on_off_side_accuracy_identical_before_after": True,
        "side_correct_samples": side_correct_count,
        "total_samples": total,
        "off_side_correct_subset": {
            "samples": int(control_off.sum()),
            "regression_only_accuracy": accuracy(control_off),
            "fusion_accuracy": accuracy(fusion_off),
            "regression_only_confusion": control_off.tolist(),
            "fusion_confusion": fusion_off.tolist(),
        },
        "on_side_correct_subset": {
            "samples": int(control_on.sum()),
            "regression_only_accuracy": accuracy(control_on),
            "fusion_accuracy": accuracy(fusion_on),
            "regression_only_confusion": control_on.tolist(),
            "fusion_confusion": fusion_on.tolist(),
        },
        "regression_to_fusion_change_matrix": change.tolist(),
        "cross_side_change_count": cross_side_changes,
    }


def main_row(value: Mapping[str, Any]) -> tuple[float, ...]:
    c = value["classification"]
    t = value["transition"]["pooled"]
    p = value["patterns"]
    return (
        c["token_accuracy"],
        c["macro_f1"],
        t["precision"],
        t["recall"],
        t["f1"],
        float(t["candidate"]),
        p["js_divergence_base2"],
        p["intersection"],
    )


def report_text(
    provenance: Mapping[str, Any],
    checkpoints: Mapping[str, Mapping[str, Any]],
    references: Mapping[str, Any],
    results: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
) -> str:
    standard = references["standard_ce_phase4"]
    rows = [
        ("Standard CE / Argmax", standard["argmax"]),
        ("Standard CE / Median", standard["median"]),
        ("Weighted CE / Argmax", references["weighted_ce_argmax"]),
        ("Weighted CE / Median", results["weighted_ce_median"]),
        ("NTL-WAS / Argmax", references["ntl_was_argmax"]),
        ("NTL-WAS / Median", results["ntl_was_median"]),
        ("Hybrid Aux / Median", results["hybrid_aux_median"]),
        ("Hybrid Regression-only", references["hybrid_regression_only"]),
        ("Hybrid Side-Constrained Fusion", results["hybrid_side_constrained"]),
    ]
    lines = [
        "# Loss × Decoding × Fusion Phase 4B Comparison",
        "",
        "## Frozen scope",
        "",
        f"- Checkpoints: {json.dumps(checkpoints, sort_keys=True)}",
        "- Training steps 0; optimizer steps 0; checkpoint changes 0.",
        "- ASAP validation only: 19 pieces / 71 humans / 70 frozen successful pairs / 272053 aligned notes.",
        "- ASAP test access 0; Repedal execution 0.",
        "- Window 512 / stride 256; overlap logits/scalars aggregated before decoding.",
        "",
        "## Main comparison",
        "",
        "| Model / Decoding | 4C Acc | Macro F1 | Trans P | Trans R | Trans F1 | Candidates | JS | Intersection |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, value in rows:
        row = main_row(value)
        lines.append(
            f"| {label} | {row[0]:.6f} | {row[1]:.6f} | {row[2]:.6f} | {row[3]:.6f} | {row[4]:.6f} | {int(row[5])} | {row[6]:.6f} | {row[7]:.6f} |"
        )
    lines += [
        "",
        "## LOW / HALF recall",
        "",
        "| Model / Decoding | LOW Recall | HALF Recall |",
        "|---|---:|---:|",
    ]
    for label, value in rows:
        classes = value["classification"]["class_metrics"]
        lines.append(f"| {label} | {classes[1]['recall']:.6f} | {classes[2]['recall']:.6f} |")
    hybrid = references["hybrid_regression_only"]
    fusion = results["hybrid_side_constrained"]
    base_row = main_row(hybrid)
    fusion_row = main_row(fusion)
    base_classes = hybrid["classification"]["class_metrics"]
    fusion_classes = fusion["classification"]["class_metrics"]
    direct = (
        ("4C Accuracy", base_row[0], fusion_row[0]),
        ("Macro F1", base_row[1], fusion_row[1]),
        ("Transition F1", base_row[4], fusion_row[4]),
        ("Candidates", base_row[5], fusion_row[5]),
        ("JS", base_row[6], fusion_row[6]),
        ("Intersection", base_row[7], fusion_row[7]),
        ("LOW Recall", base_classes[1]["recall"], fusion_classes[1]["recall"]),
        ("HALF Recall", base_classes[2]["recall"], fusion_classes[2]["recall"]),
    )
    lines += [
        "",
        "## Hybrid direct comparison",
        "",
        "| Metric | Regression-only | Side-Constrained Fusion | Delta |",
        "|---|---:|---:|---:|",
    ]
    for label, first, second in direct:
        if label == "Candidates":
            lines.append(f"| {label} | {int(first)} | {int(second)} | {int(second-first)} |")
        else:
            lines.append(f"| {label} | {first:.6f} | {second:.6f} | {second-first:+.6f} |")
    lines += [
        "",
        "## Decoder diagnostics",
        "",
        f"- Weighted CE median: {json.dumps(diagnostics['weighted_ce_median'], sort_keys=True)}",
        f"- NTL-WAS median: {json.dumps(diagnostics['ntl_was_median'], sort_keys=True)}",
        f"- Hybrid Aux median: {json.dumps(diagnostics['hybrid_aux_median'], sort_keys=True)}",
        f"- Hybrid transition invariance: {json.dumps(diagnostics['transition_invariance'], sort_keys=True)}",
        f"- Hybrid within-side: {json.dumps(diagnostics['within_side'], sort_keys=True)}",
        "",
        "## Per-method class diagnostics",
        "",
    ]
    for method in METHODS:
        value = results[method]
        lines += [
            f"### {LABELS[method]}",
            "",
            f"- Prediction distribution: {value['classification']['prediction_class_distribution']}",
            f"- Class P/R/F1: {value['classification']['class_metrics']}",
            f"- Confusion matrix rows=human, columns=candidate: {value['classification']['confusion_matrix']}",
            f"- Transition pooled/UP/DOWN: {value['transition']}",
            f"- Top predicted 256-pattern probabilities: {value['patterns']['top_candidate_patterns']}",
            "",
        ]
    weighted_arg = main_row(references["weighted_ce_argmax"])
    weighted_med = main_row(results["weighted_ce_median"])
    ntl_arg = main_row(references["ntl_was_argmax"])
    ntl_med = main_row(results["ntl_was_median"])
    aux = main_row(results["hybrid_aux_median"])
    within = diagnostics["within_side"]
    lines += [
        "## Interpretation",
        "",
        f"- Q1 (confirmed): Weighted CE Argmax→Median changes Macro F1 {weighted_arg[1]:.6f}→{weighted_med[1]:.6f}, Transition F1 {weighted_arg[4]:.6f}→{weighted_med[4]:.6f}, JS {weighted_arg[6]:.6f}→{weighted_med[6]:.6f}, Intersection {weighted_arg[7]:.6f}→{weighted_med[7]:.6f}. LOW/HALF changes are in the table; transition density must be considered with these gains.",
        f"- Q2 (confirmed): NTL Argmax→Median changes Macro F1 {ntl_arg[1]:.6f}→{ntl_med[1]:.6f}, Transition F1 {ntl_arg[4]:.6f}→{ntl_med[4]:.6f}, JS {ntl_arg[6]:.6f}→{ntl_med[6]:.6f}. Its median-change/top-2 diagnostics show posterior geometry; metric changes determine whether that geometry is useful.",
        f"- Q3 (diagnostic): Hybrid auxiliary-only Median yields Accuracy/Macro F1/Transition F1 {aux[0]:.6f}/{aux[1]:.6f}/{aux[4]:.6f}; it does not replace canonical regression inference.",
        f"- Q4 (confirmed): fusion versus regression-only deltas are Accuracy {fusion_row[0]-base_row[0]:+.6f}, Macro F1 {fusion_row[1]-base_row[1]:+.6f}, JS {fusion_row[6]-base_row[6]:+.6f}, Intersection {fusion_row[7]-base_row[7]:+.6f}.",
        "- Q5 (hard assertion): per-sample sides, candidate transition count, UP/DOWN locations, and pooled/UP/DOWN TP/FP/FN/P/R/F1 are exact between Hybrid regression-only and fusion.",
        f"- Q6 (confirmed): on side-correct samples, OFF within-side accuracy changes {within['off_side_correct_subset']['regression_only_accuracy']:.6f}→{within['off_side_correct_subset']['fusion_accuracy']:.6f}; ON changes {within['on_side_correct_subset']['regression_only_accuracy']:.6f}→{within['on_side_correct_subset']['fusion_accuracy']:.6f}.",
    ]
    strong = (
        fusion_row[0] > base_row[0]
        or fusion_row[1] > base_row[1]
    ) and fusion_row[6] <= base_row[6] and fusion_row[7] >= base_row[7]
    if strong:
        lines.append("- Q7: observed metrics support Hybrid Side-Constrained Fusion as the conventional validation finalist: Hybrid transition behavior is exact while categorical/global metrics provide net improvement.")
    else:
        lines.append("- Q7: observed metrics do not show the required overall gain with exact transition preservation; existing Hybrid regression-only remains the conventional validation finalist.")
    lines += [
        "",
        "Confirmed values are separated from interpretation above. No composite score, new training, tuning, ASAP-test access, Repedal evaluation, or follow-up experiment was run.",
        "",
    ]
    return "\n".join(lines)


def run(config_path: Path) -> int:
    config = load_json(config_path)
    output = Path(config["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    log_handle = (output / "evaluation.log").open("a", encoding="utf-8")

    def log(message: str) -> None:
        line = f"{now()} {message}"
        print(line, flush=True)
        print(line, file=log_handle, flush=True)

    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Phase 4B requires exactly one visible CUDA GPU")
        stage1, validation, references, provenance = audit(config)
        atomic_json(output / "config.json", {**config, **provenance})
        update_status(
            output,
            status="running",
            stage="checkpoint_inference",
            training_steps=0,
            optimizer_steps=0,
            checkpoint_changes=0,
            asap_test_access_count=0,
            repedal_execution_count=0,
            report_generated=False,
            error=None,
            traceback=None,
        )
        log("PREFLIGHT_PASS pieces=19 humans=71 pairs=70 training=0 optimizer=0 test=0 repedal=0")
        device = torch.device("cuda:0")
        tokenizer = PinnedTokenizerConfig()
        checkpoints: dict[str, Mapping[str, Any]] = {}
        posterior_by_model: dict[str, dict[str, np.ndarray]] = {}
        class_specs = (
            ("weighted_ce", Path(config["weighted_ce_checkpoint"]), "weighted_ce_median"),
            ("ntl_was", Path(config["ntl_was_checkpoint"]), "ntl_was_median"),
        )
        for name, checkpoint_path, method in class_specs:
            model, checkpoint = load_classification_model(
                checkpoint_path, Path(config["pretrained_checkpoint"]), device
            )
            checkpoints[name] = checkpoint
            logits = cache_classification_logits(
                output, name, model, checkpoint, stage1, config, device
            )
            posterior_by_model[name] = {}
            existing_root = (
                Path(config["weighted_ce_evaluation"]).parent
                if name == "weighted_ce"
                else Path(config["ntl_was_evaluation"]).parent
            )
            for row in stage1:
                piece = row["piece_id"]
                posterior = softmax_final_logits(logits[piece])
                posterior_by_model[name][piece] = posterior
                source = np.load(row["generated_ids_path"], allow_pickle=False)
                argmax = decode_argmax(posterior)
                existing = np.load(
                    prediction_paths(existing_root, name if name == "weighted_ce" else "ce_ntl_was", piece)["classes"],
                    allow_pickle=False,
                )
                np.testing.assert_array_equal(argmax, existing)
                median = decode_ordered_median(posterior)
                render_method(
                    output,
                    method,
                    row,
                    median,
                    source,
                    checkpoint,
                    array_hash(logits[piece]),
                    tokenizer,
                )
            del model
            gc.collect()
            torch.cuda.empty_cache()
            log(f"CLASSIFICATION_MEDIAN_READY model={name} argmax_exact=true pieces=19")
        hybrid_model, hybrid_checkpoint = load_hybrid_model(
            Path(config["hybrid_checkpoint"]),
            Path(config["pretrained_checkpoint"]),
            device,
        )
        checkpoints["hybrid"] = hybrid_checkpoint
        hybrid_posteriors: dict[str, np.ndarray] = {}
        source_control_classes: dict[str, np.ndarray] = {}
        hybrid_manifest = []
        for row in stage1:
            piece = row["piece_id"]
            source = np.load(row["generated_ids_path"], allow_pickle=False)
            regression, logits, inference = infer_hybrid_final_outputs(
                hybrid_model,
                source,
                device=device,
                window_notes=int(config["window_notes"]),
                stride_notes=int(config["stride_notes"]),
            )
            integer = regression_cc64_integer(regression)
            control_classes = np.select(
                (integer <= 25, integer <= 63, integer <= 103),
                (0, 1, 2),
                default=3,
            ).astype(np.int64)
            control_candidate = classes_to_candidate_tokens(source, control_classes)
            control_root = Path(config["hybrid_validation_root"])
            existing_paths = prediction_paths(control_root, "raw_huber_aux_ce", piece)
            np.testing.assert_array_equal(
                regression,
                np.load(
                    existing_paths["root"] / "continuous_normalized_float32.npy",
                    allow_pickle=False,
                ),
            )
            np.testing.assert_array_equal(
                control_classes, np.load(existing_paths["classes"], allow_pickle=False)
            )
            np.testing.assert_array_equal(
                control_candidate, np.load(existing_paths["ids"], allow_pickle=False)
            )
            posterior = softmax_final_logits(logits)
            hybrid_posteriors[piece] = posterior
            source_control_classes[piece] = control_classes
            aux_median = decode_auxiliary_median(posterior)
            fusion = decode_side_constrained(integer, posterior)
            if not np.array_equal(fusion >= 2, control_classes >= 2):
                raise AssertionError("source-level Hybrid OFF/ON side changed")
            render_method(
                output,
                "hybrid_aux_median",
                row,
                aux_median,
                source,
                hybrid_checkpoint,
                array_hash(logits),
                tokenizer,
            )
            render_method(
                output,
                "hybrid_side_constrained",
                row,
                fusion,
                source,
                hybrid_checkpoint,
                array_hash(logits),
                tokenizer,
            )
            reg_path = output / "final_logit_cache/hybrid" / piece / "regression_normalized_float32.npy"
            logit_path = output / "final_logit_cache/hybrid" / piece / "auxiliary_logits_float32.npy"
            atomic_npy(reg_path, regression)
            atomic_npy(logit_path, logits)
            hybrid_manifest.append({
                "piece_id": piece,
                "regression_sha256": array_hash(regression),
                "auxiliary_logits_sha256": array_hash(logits),
                "regression_control_exact": True,
                "source_off_on_exact": True,
                "inference": inference,
            })
        atomic_json(
            output / "final_logit_cache/hybrid/manifest.json",
            {"checkpoint": hybrid_checkpoint, "pieces": hybrid_manifest},
        )
        del hybrid_model
        gc.collect()
        torch.cuda.empty_cache()
        log("HYBRID_OUTPUTS_READY regression_control_exact=true source_side_exact=true pieces=19")
        diagnostics = {
            "weighted_ce_median": median_diagnostics(list(posterior_by_model["weighted_ce"].values())),
            "ntl_was_median": median_diagnostics(list(posterior_by_model["ntl_was"].values())),
            "hybrid_aux_median": median_diagnostics(list(hybrid_posteriors.values())),
        }
        humans = load_human_universe(validation)
        results = {}
        alignments = {}
        for method in METHODS:
            update_status(output, stage="alignment_and_evaluation", current_method=method)
            alignments[method] = candidate_alignment(output, method, stage1)
            results[method] = evaluate_candidate(
                method, alignments[method], validation, humans
            )
            atomic_json(output / method / "metrics.json", results[method])
            log(f"EVALUATION_READY method={method} pairs=70")
        diagnostics["transition_invariance"] = assert_hybrid_transition_invariance(
            config,
            stage1,
            alignments["hybrid_side_constrained"],
            results["hybrid_side_constrained"],
            references["hybrid_regression_only"],
        )
        diagnostics["within_side"] = within_side_diagnostics(
            config, alignments["hybrid_side_constrained"], validation, humans
        )
        if diagnostics["within_side"]["cross_side_change_count"] != 0:
            raise AssertionError("cross-side fusion changes are nonzero")
        for key, value in diagnostics.items():
            target = (
                output / key / "diagnostics.json"
                if key in METHODS
                else output / f"{key}.json"
            )
            atomic_json(target, value)
        atomic_json(output / "metrics_all.json", results)
        report = output / "LOSS_DECODING_FUSION_COMPARISON_REPORT.md"
        temporary = report.with_name(f".{report.name}.{os.getpid()}.tmp")
        temporary.write_text(
            report_text(provenance, checkpoints, references, results, diagnostics),
            encoding="utf-8",
        )
        os.replace(temporary, report)
        update_status(
            output,
            status="completed",
            stage="completed",
            current_method="all",
            transition_invariance_pass=True,
            cross_side_change_count=0,
            training_steps=0,
            optimizer_steps=0,
            checkpoint_changes=0,
            asap_test_access_count=0,
            repedal_execution_count=0,
            report_generated=True,
            error=None,
            traceback=None,
            report_path=str(report),
            completed_at=now(),
        )
        log(f"PHASE4B_COMPLETE report={report}")
        return 0
    except BaseException as error:
        update_status(
            output,
            status="failed",
            stage="failed",
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
            training_steps=0,
            optimizer_steps=0,
            checkpoint_changes=0,
            asap_test_access_count=0,
            repedal_execution_count=0,
            report_generated=False,
        )
        log(f"PHASE4B_FAILED error={error!r}")
        log(traceback.format_exc())
        return 1
    finally:
        log_handle.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("decoding/fusion evaluation is guarded; pass --execute")
    return run(args.config)


if __name__ == "__main__":
    raise SystemExit(main())
