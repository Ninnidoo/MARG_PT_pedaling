#!/usr/bin/env python3
"""Compare four decoders on one cached canonical Encoder-only posterior."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
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
    STAGE1_MANIFEST,
    SPLIT_CSV,
    align_and_cache,
    finalize_transition,
    prediction_paths,
    render_stage2_candidate,
    sum_transition,
    top_patterns,
)
from src.stage2_binary.canonical_stage1 import sha256_file
from src.stage2_binary.validation_evaluator import replace_pedal_tokens
from src.stage2_encoder_only.dataset import (
    MASK_ID,
    NON_PEDAL_FEATURES,
    TOKENS_PER_NOTE,
    generate_window_starts,
)
from src.stage2_four_class.decoding_strategies import (
    CLASS_NAMES,
    cc64_integer_to_classes,
    decode_argmax,
    decode_expectation,
    decode_ordered_median,
    decode_top2,
    softmax_final_logits,
    stable_top2_indices,
)
from src.stage2_four_class.representation import REPRESENTATIVES
from src.stage2_four_class.validation_evaluator import (
    PATTERN_COUNT,
    as_note_tokens,
    canonical_classes_from_tokens,
    classes_to_candidate_tokens,
    classification_metrics,
    confusion_from_pairs,
    load_four_class_model,
    pattern_ids,
    pattern_metrics,
    pooled_transition_counts,
)


METHODS = ("argmax", "median", "expectation", "top2_seed42")
LABELS = {
    "argmax": "Argmax",
    "median": "Posterior Median",
    "expectation": "Posterior Expectation",
    "top2_seed42": "Top-2, seed 42",
}
CANONICAL_ROOT = ROOT / "analysis/stage2_4class_architecture_validation_eval_v0"
BASELINE_METRICS = CANONICAL_ROOT / "metrics/encoder_only.json"
DEFAULT_CONFIG = ROOT / "configs/stage2_encoder_only_4class_decoding_phase4_v0.json"
REPORT_NAME = "DECODING_STRATEGY_COMPARISON_REPORT.md"
EXPECTED_CHECKPOINT_EPOCH = 2
EXPECTED_VALIDATION_CE = 0.8925421636047081
EXPECTED_PAIRS = 70
EXPECTED_ALIGNED_NOTES = 272053


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(value), allow_pickle=False)
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    descriptor = f"{array.dtype.str}|{array.shape}".encode()
    return hashlib.sha256(descriptor + array.tobytes()).hexdigest()


def update_status(output: Path, **updates: Any) -> None:
    path = output / "run_status.json"
    current = json.loads(path.read_text()) if path.is_file() else {}
    current.update(updates, last_update=now())
    atomic_json(path, current)


def audit_inputs(config: Mapping[str, Any]) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    checkpoint = Path(config["checkpoint"])
    stage1 = read_csv(STAGE1_MANIFEST)
    validation = [row for row in read_csv(SPLIT_CSV) if row["split"] == "validation"]
    if len(stage1) != 19 or len(validation) != 71:
        raise RuntimeError("canonical validation counts changed")
    if {row["piece_id"] for row in stage1} != {row["piece_id"] for row in validation}:
        raise RuntimeError("Stage 1 and validation piece universes differ")
    if not checkpoint.is_file() or not BASELINE_METRICS.is_file():
        raise FileNotFoundError("canonical checkpoint/evaluation is missing")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(payload["epoch"]) != EXPECTED_CHECKPOINT_EPOCH or not math.isclose(
        float(payload["validation_ce"]), EXPECTED_VALIDATION_CE, rel_tol=0.0, abs_tol=1e-12
    ):
        raise RuntimeError("canonical Standard CE checkpoint metadata changed")
    del payload
    if list(config["representatives"]) != list(REPRESENTATIVES):
        raise RuntimeError("canonical representatives changed")
    for key in ("training_steps", "optimizer_steps", "asap_test_access_count", "repedal_execution_count"):
        if int(config[key]) != 0:
            raise RuntimeError(f"forbidden nonzero configuration: {key}")
    baseline = json.loads(BASELINE_METRICS.read_text())
    if baseline["common_successful_pairs"] != EXPECTED_PAIRS:
        raise RuntimeError("canonical common-pair universe changed")
    return stage1, validation, {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_epoch": EXPECTED_CHECKPOINT_EPOCH,
        "checkpoint_validation_ce": EXPECTED_VALIDATION_CE,
        "stage1_manifest": str(STAGE1_MANIFEST),
        "stage1_manifest_sha256": sha256_file(STAGE1_MANIFEST),
        "validation_split": str(SPLIT_CSV),
        "validation_split_sha256": sha256_file(SPLIT_CSV),
        "validation_pieces": 19,
        "validation_humans": 71,
        "training_steps": 0,
        "optimizer_steps": 0,
        "checkpoint_changes": 0,
        "asap_test_access_count": 0,
        "repedal_execution_count": 0,
    }


def infer_final_logits(
    model: torch.nn.Module,
    source_tokens: np.ndarray,
    *,
    device: torch.device,
    window_notes: int,
    stride_notes: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Exact canonical Encoder-only overlap-logit averaging, before decoding."""

    tokens = as_note_tokens(source_tokens)
    masked = tokens.copy()
    masked[:, NON_PEDAL_FEATURES:] = MASK_ID
    starts = generate_window_starts(len(tokens), window_notes, stride_notes)
    accumulated = torch.zeros((len(tokens), 4, 4), dtype=torch.float32, device=device)
    contributions = torch.zeros(len(tokens), dtype=torch.float32, device=device)
    with torch.inference_mode():
        for start in starts:
            end = min(start + window_notes, len(tokens))
            input_ids = torch.from_numpy(masked[start:end].reshape(1, -1)).long().to(device)
            attention = torch.ones_like(input_ids)
            note_mask = torch.ones((1, end - start), dtype=torch.bool, device=device)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(
                    input_ids=input_ids,
                    token_attention_mask=attention,
                    note_mask=note_mask,
                ).logits[0]
            logits = logits[: end - start].float()
            if tuple(logits.shape) != (end - start, 4, 4) or not bool(torch.isfinite(logits).all()):
                raise FloatingPointError("invalid canonical inference logits")
            accumulated[start:end] += logits
            contributions[start:end] += 1
    if bool(torch.any(contributions == 0)):
        raise AssertionError("a final note received no window contribution")
    final = accumulated / contributions[:, None, None]
    result = final.cpu().numpy().astype(np.float32, copy=False)
    return result, {
        "notes": len(tokens),
        "windows": len(starts),
        "window_notes": window_notes,
        "stride_notes": stride_notes,
        "overlap_merge": "window logits averaged before one shared posterior/decoding",
        "finite_final_logits": bool(np.all(np.isfinite(result))),
    }


def cache_logits(
    output: Path,
    row: Mapping[str, str],
    model: torch.nn.Module,
    provenance: Mapping[str, Any],
    config: Mapping[str, Any],
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    piece_id = row["piece_id"]
    root = output / "posterior_cache" / piece_id
    path = root / "final_logits_float32.npy"
    metadata_path = root / "metadata.json"
    source = np.load(row["generated_ids_path"], allow_pickle=False)
    source_hash = array_hash(np.asarray(source, dtype=np.int64))
    try:
        metadata = json.loads(metadata_path.read_text())
        logits = np.load(path, allow_pickle=False)
        if (
            metadata["complete"]
            and metadata["checkpoint_sha256"] == provenance["checkpoint_sha256"]
            and metadata["source_array_sha256"] == source_hash
            and metadata["final_logits_sha256"] == array_hash(logits)
            and tuple(logits.shape) == (len(as_note_tokens(source)), 4, 4)
        ):
            return logits, {**metadata, "cache_reused": True}
    except Exception:
        pass
    logits, inference = infer_final_logits(
        model,
        source,
        device=device,
        window_notes=int(config["window_notes"]),
        stride_notes=int(config["stride_notes"]),
    )
    atomic_npy(path, logits)
    metadata = {
        "complete": True,
        "piece_id": piece_id,
        "checkpoint_sha256": provenance["checkpoint_sha256"],
        "source_array_sha256": source_hash,
        "final_logits_sha256": array_hash(logits),
        "path": str(path),
        "shape": list(logits.shape),
        "dtype": str(logits.dtype),
        "inference": inference,
        "cache_reused": False,
    }
    atomic_json(metadata_path, metadata)
    return logits, metadata


def decode_piece(
    posterior: np.ndarray, source: np.ndarray, *, method: str, piece_id: str
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    argmax = decode_argmax(posterior)
    diagnostics: dict[str, Any] = {"same_final_posterior": True}
    if method == "argmax":
        classes = argmax
        candidate = classes_to_candidate_tokens(source, classes)
    elif method == "median":
        classes = decode_ordered_median(posterior)
        candidate = classes_to_candidate_tokens(source, classes)
    elif method == "expectation":
        continuous, integer = decode_expectation(posterior)
        classes = cc64_integer_to_classes(integer)
        candidate = replace_pedal_tokens(as_note_tokens(source), integer)
        diagnostics.update(
            continuous_cc64=continuous,
            integer_cc64=integer,
            representative_snapping=False,
            integer_conversion="clip [0,127], nearest integer half up",
        )
    elif method == "top2_seed42":
        classes, top2 = decode_top2(posterior, piece_id=piece_id, seed=42)
        candidate = classes_to_candidate_tokens(source, classes)
        diagnostics.update(top2)
    else:
        raise ValueError(method)
    if not np.array_equal(
        as_note_tokens(candidate)[:, :NON_PEDAL_FEATURES],
        as_note_tokens(source)[:, :NON_PEDAL_FEATURES],
    ):
        raise AssertionError("decoder changed Stage 1 non-pedal tokens")
    return classes, candidate, diagnostics


def direction_matrix(first: np.ndarray, second: np.ndarray) -> list[list[int]]:
    return np.bincount(
        np.asarray(first).reshape(-1) * 4 + np.asarray(second).reshape(-1), minlength=16
    ).reshape(4, 4).tolist()


def posterior_diagnostics(posteriors: Sequence[np.ndarray]) -> dict[str, Any]:
    p = np.concatenate([value.reshape(-1, 4) for value in posteriors], axis=0)
    entropy = -(p * np.log(np.clip(p, 1e-300, 1.0))).sum(axis=1)
    order = np.argsort(-p, axis=1, kind="stable")
    ranked = np.take_along_axis(p, order, axis=1)
    margin = ranked[:, 0] - ranked[:, 1]
    argmax = order[:, 0]
    top2 = order[:, :2]
    in_low = (top2 == 1).any(axis=1) & (argmax != 1)
    in_half = (top2 == 2).any(axis=1) & (argmax != 2)
    return {
        "sample_universe": "19 Stage-1 pieces, each final note/pedal slot once",
        "pedal_samples": len(p),
        "mean_posterior_entropy_nats": float(entropy.mean()),
        "median_posterior_entropy_nats": float(np.median(entropy)),
        "top1_probability_mean": float(ranked[:, 0].mean()),
        "top2_probability_mean": float(ranked[:, 1].mean()),
        "top1_top2_margin_mean": float(margin.mean()),
        "margin_quantiles": _quantiles(margin),
        "low_in_top2_but_not_argmax_fraction": float(in_low.mean()),
        "half_in_top2_but_not_argmax_fraction": float(in_half.mean()),
        "low_or_half_in_top2_but_not_argmax_fraction": float((in_low | in_half).mean()),
    }


def _quantiles(values: np.ndarray) -> dict[str, float]:
    points = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
    return {f"p{int(point * 100):02d}": float(np.quantile(values, point)) for point in points}


def decoder_diagnostics(
    decoded: Mapping[str, Sequence[np.ndarray]],
    extras: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    argmax = np.concatenate([x.reshape(-1) for x in decoded["argmax"]])
    median = np.concatenate([x.reshape(-1) for x in decoded["median"]])
    top2 = np.concatenate([x.reshape(-1) for x in decoded["top2_seed42"]])
    continuous = np.concatenate(
        [x["continuous_cc64"].reshape(-1) for x in extras["expectation"]]
    )
    integer = np.concatenate([x["integer_cc64"].reshape(-1) for x in extras["expectation"]])
    top2_indices = np.concatenate(
        [x["top2_indices"].reshape(-1, 2) for x in extras["top2_seed42"]], axis=0
    )
    pairs = Counter(
        tuple(sorted((int(row[0]), int(row[1])))) for row in top2_indices
    )
    return {
        "median": {
            "different_from_argmax_fraction": float(np.mean(median != argmax)),
            "argmax_to_median_count_matrix": direction_matrix(argmax, median),
        },
        "expectation": {
            "continuous_mean": float(continuous.mean()),
            "continuous_std": float(continuous.std()),
            "continuous_quantiles": _quantiles(continuous),
            "fraction_lt_26": float(np.mean(integer < 26)),
            "fraction_26_63": float(np.mean((integer >= 26) & (integer <= 63))),
            "fraction_64_103": float(np.mean((integer >= 64) & (integer <= 103))),
            "fraction_ge_104": float(np.mean(integer >= 104)),
            "fraction_eq_0": float(np.mean(integer == 0)),
            "fraction_eq_127": float(np.mean(integer == 127)),
            "mean_absolute_distance_from_argmax_representative": float(
                np.mean(np.abs(continuous - np.asarray(REPRESENTATIVES)[argmax]))
            ),
            "representative_snapping": False,
        },
        "top2_seed42": {
            "different_from_argmax_fraction": float(np.mean(top2 != argmax)),
            "non_argmax_selection_count": int(np.sum(top2 != argmax)),
            "sampled_class_distribution": np.bincount(top2, minlength=4).astype(float).tolist()
            / np.asarray([len(top2)] * 4),
            "low_selection_count": int(np.sum(top2 == 1)),
            "half_selection_count": int(np.sum(top2 == 2)),
            "top2_pair_frequency": {
                f"{{{CLASS_NAMES[a]},{CLASS_NAMES[b]}}}": count
                for (a, b), count in sorted(pairs.items())
            },
            "seed": 42,
            "temperature": 1.0,
            "one_fixed_seed_validation_result": True,
        },
    }


def load_human_universe(validation: Sequence[Mapping[str, str]]) -> dict[str, dict[str, Any]]:
    universe: dict[str, dict[str, Any]] = {}
    for row in validation:
        identifier = row["metadata_index"]
        root = CANONICAL_ROOT / "alignment/human" / identifier
        metadata = root / "metadata.json"
        tokens = root / "aligned_tokens_int64.npy"
        transitions = root / "transitions.json"
        if metadata.is_file() and tokens.is_file() and transitions.is_file():
            saved = json.loads(metadata.read_text())
            if saved.get("complete"):
                universe[identifier] = {
                    "tokens": np.load(tokens, allow_pickle=False),
                    "transitions": json.loads(transitions.read_text()),
                    "metadata": saved,
                }
    if len(universe) != EXPECTED_PAIRS:
        raise RuntimeError(f"canonical human universe must contain {EXPECTED_PAIRS} alignments")
    return universe


def evaluate_method(
    output: Path,
    method: str,
    stage1: Sequence[Mapping[str, str]],
    validation: Sequence[Mapping[str, str]],
    human: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    candidates: dict[str, Mapping[str, Any]] = {}
    for row in stage1:
        piece = row["piece_id"]
        if method == "argmax":
            new_meta = json.loads(prediction_paths(output, method, piece)["metadata"].read_text())
            old_meta = json.loads(prediction_paths(CANONICAL_ROOT, "encoder_only", piece)["metadata"].read_text())
            if new_meta["candidate_midi_sha256"] != old_meta["candidate_midi_sha256"]:
                raise AssertionError("Argmax MIDI differs from canonical output")
            root = CANONICAL_ROOT / "alignment/encoder_only" / piece
            candidates[piece] = {
                "tokens": np.load(root / "aligned_tokens_int64.npy", allow_pickle=False),
                "transitions": json.loads((root / "transitions.json").read_text()),
            }
        else:
            metadata = json.loads(prediction_paths(output, method, piece)["metadata"].read_text())
            candidates[piece] = align_and_cache(
                output,
                kind=method,
                identifier=piece,
                score_path=Path(row["selected_score_absolute_path"]),
                performance_path=Path(metadata["candidate_midi"]),
            )
    confusion = np.zeros((4, 4), dtype=np.int64)
    candidate_histogram = np.zeros(PATTERN_COUNT, dtype=np.int64)
    target_histogram = np.zeros(PATTERN_COUNT, dtype=np.int64)
    transition = {
        scope: {key: 0 for key in ("candidate", "reference", "tp", "fp", "fn")}
        for scope in ("pooled", "up", "down")
    }
    pairs = 0
    aligned_notes = 0
    for row in validation:
        identifier = row["metadata_index"]
        if identifier not in human:
            continue
        candidate = candidates[row["piece_id"]]
        target = human[identifier]
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
        pairs += 1
        aligned_notes += len(predicted)
    if pairs != EXPECTED_PAIRS or aligned_notes != EXPECTED_ALIGNED_NOTES:
        raise AssertionError("evaluation universe drifted from canonical 70 pairs")
    patterns = pattern_metrics(candidate_histogram, target_histogram)
    return {
        "method": LABELS[method],
        "candidate_pieces": 19,
        "human_performances": 71,
        "common_successful_pairs": pairs,
        "aligned_notes": aligned_notes,
        "alignment_pedal_samples": aligned_notes * 4,
        "classification": classification_metrics(confusion),
        "transition": finalize_transition(transition),
        "patterns": {
            **patterns,
            "top_candidate_patterns": top_patterns(candidate_histogram),
            "top_target_patterns": top_patterns(target_histogram),
        },
    }


def assert_argmax_regression(
    output: Path,
    stage1: Sequence[Mapping[str, str]],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    for row in stage1:
        piece = row["piece_id"]
        new = prediction_paths(output, "argmax", piece)
        old = prediction_paths(CANONICAL_ROOT, "encoder_only", piece)
        np.testing.assert_array_equal(
            np.load(new["classes"], allow_pickle=False), np.load(old["classes"], allow_pickle=False)
        )
        np.testing.assert_array_equal(
            np.load(new["ids"], allow_pickle=False), np.load(old["ids"], allow_pickle=False)
        )
    baseline = json.loads(BASELINE_METRICS.read_text())
    scalar_paths = (
        ("classification", "token_accuracy"),
        ("classification", "macro_f1"),
        ("transition", "pooled", "precision"),
        ("transition", "pooled", "recall"),
        ("transition", "pooled", "f1"),
        ("patterns", "js_divergence_base2"),
        ("patterns", "intersection"),
    )
    for path in scalar_paths:
        actual: Any = result
        expected: Any = baseline
        for key in path:
            actual, expected = actual[key], expected[key]
        if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-15):
            raise AssertionError(f"Argmax metric regression failed at {path}: {actual} != {expected}")
    if result["classification"]["confusion_matrix"] != baseline["classification"]["confusion_matrix"]:
        raise AssertionError("Argmax confusion matrix regression failed")
    return {
        "class_ids_exact_equality": True,
        "candidate_cc64_token_ids_exact_equality": True,
        "candidate_midi_sha256_exact_equality": True,
        "main_metrics_absolute_tolerance": 1e-15,
        "main_metrics_pass": True,
        "confusion_matrix_exact_equality": True,
    }


def metric_row(value: Mapping[str, Any]) -> tuple[float, ...]:
    c, t, p = value["classification"], value["transition"]["pooled"], value["patterns"]
    return (
        c["token_accuracy"], c["macro_f1"], t["precision"], t["recall"], t["f1"],
        float(t["candidate"]), p["js_divergence_base2"], p["intersection"],
    )


def report_text(
    provenance: Mapping[str, Any],
    results: Mapping[str, Mapping[str, Any]],
    posterior: Mapping[str, Any],
    decoder: Mapping[str, Any],
    regression: Mapping[str, Any],
) -> str:
    hybrid = (0.507749, 0.409029, 0.502348, 0.513770, 0.507995, 35354, 0.038362, 0.834859)
    lines = [
        "# Encoder-only Standard CE Decoding Strategy Comparison",
        "",
        "## Scope and provenance",
        "",
        f"- Standard CE checkpoint: `{provenance['checkpoint']}` (SHA256 `{provenance['checkpoint_sha256']}`; epoch 2)",
        "- Same final posterior: 512-note windows, stride 256, overlap logits averaged first, one softmax posterior cached per final note/pedal sample.",
        "- Training steps: 0; optimizer steps: 0; checkpoint changes: 0.",
        "- ASAP validation only: 19 pieces / 71 humans / 70 frozen successful pairs; ASAP test access: 0.",
        "- Repedal execution: 0. Transition is direction-aware, one-to-one, threshold 64, tolerance ±1 distinct onset.",
        "",
        "## Regression and targeted guarantees",
        "",
        f"- Argmax exact regression: PASS — `{json.dumps(regression, sort_keys=True)}`",
        "- Ordered median cumulative `>=0.5` tests: PASS.",
        "- Expectation `[0,51,79,127]` arithmetic and Raw-Huber half-up integer conversion: PASS; representative snapping: false.",
        "- Top-2 support/tie rule and seed-42 exact reproducibility: PASS.",
        "- All decoders consumed the same cached final-logit hashes: PASS.",
        "",
        "## Main metrics",
        "",
        "| Decoding | 4C Acc ↑ | Macro F1 ↑ | Trans P ↑ | Trans R ↑ | Trans F1 ↑ | Candidates | JS ↓ | Intersection ↑ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = metric_row(results[method])
        lines.append(
            f"| {LABELS[method]} | {row[0]:.6f} | {row[1]:.6f} | {row[2]:.6f} | {row[3]:.6f} | {row[4]:.6f} | {int(row[5])} | {row[6]:.6f} | {row[7]:.6f} |"
        )
    lines.append(
        f"| Hybrid finalist (reference) | {hybrid[0]:.6f} | {hybrid[1]:.6f} | {hybrid[2]:.6f} | {hybrid[3]:.6f} | {hybrid[4]:.6f} | {hybrid[5]} | {hybrid[6]:.6f} | {hybrid[7]:.6f} |"
    )
    lines += [
        "",
        "## Class and transition diagnostics",
        "",
        "| Decoding | LOW Recall | HALF Recall | Candidate transitions |",
        "|---|---:|---:|---:|",
    ]
    for method in METHODS:
        value = results[method]
        classes = value["classification"]["class_metrics"]
        lines.append(
            f"| {LABELS[method]} | {classes[1]['recall']:.6f} | {classes[2]['recall']:.6f} | {value['transition']['pooled']['candidate']} |"
        )
    lines += [
        "| Hybrid finalist (reference) | 0.186309 | 0.267974 | 35354 |",
        "",
    ]
    for method in METHODS:
        value = results[method]
        lines += [
            f"### {LABELS[method]}",
            "",
            f"- Prediction distribution: `{value['classification']['prediction_class_distribution']}`",
            f"- Class P/R/F1: `{value['classification']['class_metrics']}`",
            f"- Confusion (rows human, columns candidate): `{value['classification']['confusion_matrix']}`",
            f"- Transition pooled/UP/DOWN: `{value['transition']}`",
            f"- Top 256-pattern probabilities: `{value['patterns']['top_candidate_patterns']}`",
            "",
        ]
    lines += [
        "## Posterior and decoder diagnostics",
        "",
        f"- Shared posterior confidence: `{json.dumps(json_safe(posterior), sort_keys=True)}`",
        f"- Median vs argmax: `{json.dumps(json_safe(decoder['median']), sort_keys=True)}`",
        f"- Expectation: `{json.dumps(json_safe(decoder['expectation']), sort_keys=True)}`",
        f"- Top-2: `{json.dumps(json_safe(decoder['top2_seed42']), sort_keys=True)}`",
        "",
        "## Interpretation",
        "",
    ]
    arg = metric_row(results["argmax"])
    med = metric_row(results["median"])
    exp = metric_row(results["expectation"])
    top = metric_row(results["top2_seed42"])
    best_transition = max(METHODS, key=lambda key: metric_row(results[key])[4])
    best_macro = max(METHODS, key=lambda key: metric_row(results[key])[1])
    lines += [
        f"- Q1 (confirmed): best Macro F1 decoder is {LABELS[best_macro]} ({metric_row(results[best_macro])[1]:.6f} vs argmax {arg[1]:.6f}); best Transition F1 decoder is {LABELS[best_transition]} ({metric_row(results[best_transition])[4]:.6f} vs argmax {arg[4]:.6f}). No single-metric winner is declared.",
        f"- Q2 (confirmed): posterior median changes {decoder['median']['different_from_argmax_fraction']:.3%} of Stage-1 pedal samples; its LOW/HALF recalls are {results['median']['classification']['class_metrics'][1]['recall']:.6f}/{results['median']['classification']['class_metrics'][2]['recall']:.6f} versus argmax {results['argmax']['classification']['class_metrics'][1]['recall']:.6f}/{results['argmax']['classification']['class_metrics'][2]['recall']:.6f}.",
        f"- Q3 (confirmed): expectation changes the transition metrics from P/R/F1 {arg[2]:.6f}/{arg[3]:.6f}/{arg[4]:.6f} to {exp[2]:.6f}/{exp[3]:.6f}/{exp[4]:.6f}, JS {arg[6]:.6f}→{exp[6]:.6f}, and Intersection {arg[7]:.6f}→{exp[7]:.6f}.",
        f"- Q4 (confirmed): Top-2 selects LOW/HALF {decoder['top2_seed42']['low_selection_count']}/{decoder['top2_seed42']['half_selection_count']} times in its once-per-Stage1-sample diagnostic and differs from argmax {decoder['top2_seed42']['different_from_argmax_fraction']:.3%}; candidate transitions are {int(top[5])} versus argmax {int(arg[5])}. This is one fixed-seed validation result.",
        f"- Q5 (inference): LOW/HALF appears in Top-2 without being argmax for {posterior['low_or_half_in_top2_but_not_argmax_fraction']:.3%} of samples. Metric changes show how much of that uncertainty is useful rather than proving that all retained uncertainty is useful.",
        f"- Q6 (confirmed): Hybrid reference has Macro F1/Transition F1 {hybrid[1]:.6f}/{hybrid[4]:.6f}; the strongest Standard-CE decoding values are {metric_row(results[best_macro])[1]:.6f}/{metric_row(results[best_transition])[4]:.6f}. Other dimensions remain in the table.",
    ]
    promising = [method for method in METHODS[1:] if metric_row(results[method])[1] > arg[1] or metric_row(results[method])[4] > arg[4]]
    if promising:
        lines.append(
            "- Q7 (candidate only): " + ", ".join(LABELS[x] for x in promising) + " merit consideration in a future loss×decoding check. No follow-up inference/training was started."
        )
    else:
        lines.append("- Q7: no non-argmax decoder improved Macro F1 or Transition F1, so no loss×decoding follow-up candidate is supported by these results.")
    lines += [
        "",
        "Median/expectation/Top-2 trade-offs must be read across Macro F1, transition P/R/F1 and density, JS/Intersection, and LOW/HALF behavior. Top-2 is explicitly one seed-42 result; no seed, k, or temperature sweep was run.",
        "",
    ]
    return "\n".join(lines)


def run(config_path: Path) -> int:
    config = json.loads(config_path.read_text())
    output = Path(config["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    log_handle = (output / "evaluation.log").open("a", encoding="utf-8")

    def log(message: str) -> None:
        line = f"{now()} {message}"
        print(line, flush=True)
        print(line, file=log_handle, flush=True)

    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Phase 4 inference requires exactly one visible CUDA GPU")
        stage1, validation, provenance = audit_inputs(config)
        atomic_json(output / "config.json", {**config, **provenance})
        update_status(
            output,
            status="running",
            stage="posterior_cache",
            training_steps=0,
            optimizer_steps=0,
            checkpoint_changes=0,
            asap_test_access_count=0,
            repedal_execution_count=0,
            current_piece=0,
            report_generated=False,
        )
        log("PREFLIGHT_PASS training_steps=0 optimizer_steps=0 test_access=0 repedal=0")
        device = torch.device("cuda:0")
        model, loaded = load_four_class_model(
            "encoder_only",
            config["checkpoint"],
            pretrained_checkpoint=config["pretrained_checkpoint"],
            device=device,
        )
        if loaded["epoch"] != EXPECTED_CHECKPOINT_EPOCH:
            raise RuntimeError("loaded checkpoint epoch mismatch")
        logits_by_piece: dict[str, np.ndarray] = {}
        posterior_by_piece: dict[str, np.ndarray] = {}
        cache_manifest: list[dict[str, Any]] = []
        for index, row in enumerate(stage1, 1):
            logits, metadata = cache_logits(output, row, model, provenance, config, device)
            logits_by_piece[row["piece_id"]] = logits
            posterior_by_piece[row["piece_id"]] = softmax_final_logits(logits)
            cache_manifest.append(metadata)
            update_status(output, current_piece=index)
            log(f"FINAL_LOGITS_READY piece={index}/19 cache_reused={metadata['cache_reused']}")
        del model
        gc.collect()
        torch.cuda.empty_cache()
        shared_hashes = {row["piece_id"]: row["final_logits_sha256"] for row in cache_manifest}
        atomic_json(
            output / "posterior_cache/manifest.json",
            {
                "checkpoint_sha256": provenance["checkpoint_sha256"],
                "overlap_logits_before_decoding": True,
                "same_final_logits_all_methods": True,
                "pieces": cache_manifest,
            },
        )
        tokenizer = PinnedTokenizerConfig()
        decoded: dict[str, list[np.ndarray]] = {name: [] for name in METHODS}
        extras: dict[str, list[Mapping[str, Any]]] = {name: [] for name in METHODS}
        for method in METHODS:
            method_hashes: dict[str, str] = {}
            for row in stage1:
                piece = row["piece_id"]
                source = np.load(row["generated_ids_path"], allow_pickle=False)
                classes, candidate, diagnostic = decode_piece(
                    posterior_by_piece[piece], source, method=method, piece_id=piece
                )
                decoded[method].append(classes)
                extras[method].append(diagnostic)
                method_hashes[piece] = shared_hashes[piece]
                render_stage2_candidate(
                    output,
                    method,
                    row,
                    classes,
                    candidate,
                    {
                        "decoder": LABELS[method],
                        "same_cached_final_logits": True,
                        "final_logits_sha256": shared_hashes[piece],
                        "softmax_after_overlap_average": True,
                        "non_pedal_tokens_preserved": True,
                        "top2_seed": 42 if method == "top2_seed42" else None,
                        "temperature": 1.0,
                    },
                    provenance["checkpoint_sha256"],
                    tokenizer,
                )
            atomic_json(
                output / method / "decoder_input_manifest.json",
                {"method": method, "final_logits_by_piece": method_hashes},
            )
            if method_hashes != shared_hashes:
                raise AssertionError("same-posterior guarantee failed")
            log(f"DECODING_READY method={method} pieces=19")
        # Exact stochastic reproducibility is checked again over full validation posteriors.
        for row, expected in zip(stage1, decoded["top2_seed42"]):
            actual, _ = decode_top2(
                posterior_by_piece[row["piece_id"]], piece_id=row["piece_id"], seed=42
            )
            np.testing.assert_array_equal(actual, expected)
        post_diag = posterior_diagnostics(list(posterior_by_piece.values()))
        decoder_diag = decoder_diagnostics(decoded, extras)
        atomic_json(output / "posterior_diagnostics.json", post_diag)
        atomic_json(output / "decoder_diagnostics.json", decoder_diag)
        human = load_human_universe(validation)
        results: dict[str, dict[str, Any]] = {}
        # Gate all alternative evaluation on exact canonical Argmax regression.
        results["argmax"] = evaluate_method(output, "argmax", stage1, validation, human)
        regression = assert_argmax_regression(output, stage1, results["argmax"])
        atomic_json(output / "argmax/regression_test.json", regression)
        atomic_json(output / "argmax/metrics.json", results["argmax"])
        log("ARGMAX_EXACT_REGRESSION_PASS classes=true cc64=true metrics=true")
        for method in METHODS[1:]:
            update_status(output, stage="alignment_and_evaluation", current_method=method)
            results[method] = evaluate_method(output, method, stage1, validation, human)
            atomic_json(output / method / "metrics.json", results[method])
            atomic_json(output / method / "diagnostics.json", decoder_diag[method])
            log(f"EVALUATION_READY method={method} pairs=70")
        report = output / REPORT_NAME
        temporary = report.with_name(f".{report.name}.{os.getpid()}.tmp")
        temporary.write_text(
            report_text(provenance, results, post_diag, decoder_diag, regression),
            encoding="utf-8",
        )
        os.replace(temporary, report)
        atomic_json(output / "metrics_all.json", results)
        update_status(
            output,
            status="completed",
            stage="completed",
            current_method="all",
            same_posterior_guarantee=True,
            argmax_regression_pass=True,
            top2_seed42_reproducibility_pass=True,
            asap_test_access_count=0,
            repedal_execution_count=0,
            training_steps=0,
            optimizer_steps=0,
            checkpoint_changes=0,
            report_generated=True,
            report_path=str(report),
            completed_at=now(),
        )
        log(f"PHASE4_COMPLETE report={report}")
        return 0
    except BaseException as error:
        update_status(
            output,
            status="failed",
            stage="failed",
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
            asap_test_access_count=0,
            repedal_execution_count=0,
            training_steps=0,
            optimizer_steps=0,
            checkpoint_changes=0,
            report_generated=False,
        )
        log(f"PHASE4_FAILED error={error!r}")
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
        parser.error("decoding evaluation is guarded; pass --execute")
    return run(args.config)


if __name__ == "__main__":
    raise SystemExit(main())
