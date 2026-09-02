#!/usr/bin/env python3
"""Validation-only diagnosis and immutable final experiment lock for Stage 2."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from miditoolkit import MidiFile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.stage2_binary.validation_evaluator import (  # noqa: E402
    JOINT_PATTERNS,
    infer_cached_binary_pedals,
    joint16_histogram_from_ids,
    load_binary_stage2_checkpoint,
    normalized_joint16,
    official_pt_pedal_similarity,
)
from src.stage2_encoder_only.dataset import _PinnedTokenizerConfig  # noqa: E402
from third_party.PianistTransformer.src.utils.midi import midi_to_ids  # noqa: E402


DIAGNOSIS_ROOT = PROJECT_ROOT / "analysis/stage2_binary_v0/validation_diagnosis_v0"
LOCK_ROOT = PROJECT_ROOT / "analysis/stage2_binary_v0/final_lock_v0"
VALIDATION_ROOT = PROJECT_ROOT / "analysis/stage2_binary_v0/validation_eval_v0"
TRAIN_ROOT = PROJECT_ROOT / "analysis/stage2_binary_v0/train_v0"
CACHE_ROOT = PROJECT_ROOT / "analysis/stage2_binary_v0/train_setup_v0/shared_cache"
ENCODER_CHECKPOINT = PROJECT_ROOT / "checkpoints/pianist_transformer"
MODEL_SOURCE = PROJECT_ROOT / "src/stage2_binary/model.py"
EVALUATOR_SOURCE = PROJECT_ROOT / "src/stage2_binary/validation_evaluator.py"
TOKENIZER_SOURCE = PROJECT_ROOT / "third_party/PianistTransformer/src/utils/midi.py"
OFFICIAL_EVALUATOR_SOURCE = (
    PROJECT_ROOT / "third_party/PianistTransformer/src/evaluate/evaluate.py"
)

ARCHITECTURES = ("independent_4x2", "joint_16")
CHECKPOINTS = {
    "independent_4x2": TRAIN_ROOT / "independent_4x2/best.pt",
    "joint_16": TRAIN_ROOT / "joint_16/best.pt",
}
EXPECTED_BEST_EPOCH = {"independent_4x2": 2, "joint_16": 3}
EXPECTED_GLOBAL = {
    "independent_4x2": {
        "js_distance": 0.06991888733125748,
        "intersection": 0.9652983320723032,
    },
    "joint_16": {
        "js_distance": 0.16875023071926262,
        "intersection": 0.9193735869220325,
    },
}
GLOBAL_TOLERANCE = 1e-12
BINARY_THRESHOLD = 64


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _human_piece_histograms() -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    cache_stats = _read_json(CACHE_ROOT / "cache_statistics.json")
    split = cache_stats["splits"]["validation"]
    tokens = np.load(CACHE_ROOT / split["tokens_file"], mmap_mode="r")
    rows = _read_csv(CACHE_ROOT / split["index_file"])
    if len(rows) != 71 or tuple(tokens.shape) != (283928, 8):
        raise RuntimeError("validation human cache inventory mismatch")
    histograms: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(16, dtype=np.int64))
    metadata: dict[str, dict[str, Any]] = {}
    for row in rows:
        piece_id = row["piece_id"]
        offset = int(row["token_offset"])
        notes = int(row["notes"])
        histograms[piece_id] += joint16_histogram_from_ids(tokens[offset : offset + notes])
        item = metadata.setdefault(
            piece_id,
            {
                "piece_id": piece_id,
                "composer": row["composer"],
                "title": row["title"],
                "human_performances": 0,
                "human_notes": 0,
            },
        )
        item["human_performances"] += 1
        item["human_notes"] += notes
    if len(histograms) != 19 or sum(item["human_performances"] for item in metadata.values()) != 71:
        raise RuntimeError("human piece aggregation mismatch")
    return dict(histograms), metadata


def _original_pt_piece_histograms(
    stage1_rows: Sequence[Mapping[str, str]],
) -> tuple[dict[str, np.ndarray], dict[str, int], np.ndarray]:
    tokenizer = _PinnedTokenizerConfig()
    histograms: dict[str, np.ndarray] = {}
    note_counts: dict[str, int] = {}
    global_histogram = np.zeros(16, dtype=np.int64)
    for row in stage1_rows:
        midi_path = Path(row["generated_midi_path"])
        if not midi_path.is_file() or "validation_eval_v0/stage1_cache" not in str(midi_path):
            raise RuntimeError(f"unexpected Original PT validation cache path: {midi_path}")
        ids = midi_to_ids(tokenizer, MidiFile(str(midi_path)))
        histogram = joint16_histogram_from_ids(ids)
        piece_id = row["piece_id"]
        histograms[piece_id] = histogram
        note_counts[piece_id] = int(histogram.sum())
        global_histogram += histogram
    return histograms, note_counts, global_histogram


def _stage2_piece_histograms(
    architecture: str,
    stage1_rows: Sequence[Mapping[str, str]],
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, int], np.ndarray, int]:
    checkpoint = CHECKPOINTS[architecture]
    model = load_binary_stage2_checkpoint(
        checkpoint,
        architecture=architecture,
        encoder_checkpoint=ENCODER_CHECKPOINT,
        device=device,
    )
    histograms: dict[str, np.ndarray] = {}
    note_counts: dict[str, int] = {}
    global_histogram = np.zeros(16, dtype=np.int64)
    total_windows = 0
    for row in stage1_rows:
        ids_path = Path(row["generated_ids_path"])
        if not ids_path.is_file() or "validation_eval_v0/stage1_cache" not in str(ids_path):
            raise RuntimeError(f"unexpected Stage 1 validation ID path: {ids_path}")
        stage1_ids = np.load(ids_path)
        candidate, details = infer_cached_binary_pedals(
            model,
            stage1_ids,
            architecture=architecture,
            device=device,
            window_notes=512,
            stride_notes=256,
        )
        if not details["non_pedal_tokens_preserved"]:
            raise AssertionError("Stage 2 changed a non-pedal token")
        histogram = joint16_histogram_from_ids(candidate)
        piece_id = row["piece_id"]
        histograms[piece_id] = histogram
        note_counts[piece_id] = int(histogram.sum())
        global_histogram += histogram
        total_windows += int(details["windows"])
    del model
    torch.cuda.empty_cache()
    return histograms, note_counts, global_histogram, total_windows


def _marginal(probability: np.ndarray, slot: int) -> tuple[float, float]:
    shift = 3 - slot
    on = sum(float(probability[joint_id]) for joint_id in range(16) if (joint_id >> shift) & 1)
    return 1.0 - on, on


def _global_csv_rows(
    histograms: Mapping[str, np.ndarray],
    probabilities: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows = []
    for joint_id, pattern in enumerate(JOINT_PATTERNS):
        human = float(probabilities["human"][joint_id])
        row: dict[str, Any] = {
            "joint_id": joint_id,
            "pattern": pattern,
            "human_count": int(histograms["human"][joint_id]),
            "human_probability": human,
        }
        for candidate in ("original_pt", "independent_4x2", "joint_16"):
            probability = float(probabilities[candidate][joint_id])
            row[f"{candidate}_count"] = int(histograms[candidate][joint_id])
            row[f"{candidate}_probability"] = probability
            row[f"{candidate}_signed_error"] = probability - human
            row[f"{candidate}_absolute_error"] = abs(probability - human)
        rows.append(row)
    return rows


def _slotwise_rows(probabilities: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    rows = []
    for slot in range(4):
        human_off, human_on = _marginal(probabilities["human"], slot)
        row: dict[str, Any] = {
            "pedal_slot": f"Pedal{slot + 1}",
            "human_off_probability": human_off,
            "human_on_probability": human_on,
        }
        for candidate in ("original_pt", "independent_4x2", "joint_16"):
            off, on = _marginal(probabilities[candidate], slot)
            row[f"{candidate}_off_probability"] = off
            row[f"{candidate}_on_probability"] = on
            row[f"{candidate}_off_signed_error"] = off - human_off
            row[f"{candidate}_on_signed_error"] = on - human_on
            row[f"{candidate}_off_absolute_error"] = abs(off - human_off)
            row[f"{candidate}_on_absolute_error"] = abs(on - human_on)
        rows.append(row)
    return rows


def _piece_rows(
    human: Mapping[str, np.ndarray],
    human_metadata: Mapping[str, Mapping[str, Any]],
    candidates: Mapping[str, Mapping[str, np.ndarray]],
    candidate_notes: Mapping[str, Mapping[str, int]],
) -> list[dict[str, Any]]:
    rows = []
    for piece_id in sorted(human, key=lambda value: (
        str(human_metadata[value]["composer"]),
        str(human_metadata[value]["title"]),
        value,
    )):
        metadata = human_metadata[piece_id]
        row: dict[str, Any] = dict(metadata)
        for candidate in ("original_pt", "independent_4x2", "joint_16"):
            metric = official_pt_pedal_similarity(human[piece_id], candidates[candidate][piece_id])
            row[f"{candidate}_candidate_notes"] = candidate_notes[candidate][piece_id]
            row[f"{candidate}_js_distance"] = float(metric["js_distance_base2"])
            row[f"{candidate}_intersection"] = float(metric["histogram_intersection"])
        row["independent_js_improvement_vs_original_pt"] = (
            row["original_pt_js_distance"] - row["independent_4x2_js_distance"]
        )
        row["joint_js_improvement_vs_original_pt"] = (
            row["original_pt_js_distance"] - row["joint_16_js_distance"]
        )
        rows.append(row)
    if len(rows) != 19:
        raise RuntimeError("piece metric row count must be 19")
    return rows


def _top_pattern_errors(rows: Sequence[Mapping[str, Any]], candidate: str, count: int = 4) -> list[Mapping[str, Any]]:
    return sorted(
        rows,
        key=lambda row: float(row[f"{candidate}_absolute_error"]),
        reverse=True,
    )[:count]


def _piece_summary(rows: Sequence[Mapping[str, Any]], candidate: str) -> dict[str, Any]:
    js_values = [float(row[f"{candidate}_js_distance"]) for row in rows]
    improvement_key = (
        "independent_js_improvement_vs_original_pt"
        if candidate == "independent_4x2"
        else "joint_js_improvement_vs_original_pt"
    )
    improvements = [float(row[improvement_key]) for row in rows]
    ranked = sorted(zip(rows, improvements), key=lambda item: item[1], reverse=True)
    return {
        "improved_piece_count": sum(value > 0 for value in improvements),
        "mean_js_distance": statistics.fmean(js_values),
        "median_js_distance": statistics.median(js_values),
        "best_improvements": [
            {
                "piece_id": item[0]["piece_id"],
                "composer": item[0]["composer"],
                "title": item[0]["title"],
                "js_improvement": item[1],
            }
            for item in ranked[:3]
        ],
        "worst_improvements": [
            {
                "piece_id": item[0]["piece_id"],
                "composer": item[0]["composer"],
                "title": item[0]["title"],
                "js_improvement": item[1],
            }
            for item in ranked[-3:]
        ],
    }


def _report(
    histograms: Mapping[str, np.ndarray],
    probabilities: Mapping[str, np.ndarray],
    global_metrics: Mapping[str, Mapping[str, float]],
    global_rows: Sequence[Mapping[str, Any]],
    slot_rows: Sequence[Mapping[str, Any]],
    piece_rows: Sequence[Mapping[str, Any]],
    piece_summaries: Mapping[str, Mapping[str, Any]],
    checkpoint_hashes: Mapping[str, str],
    windows: Mapping[str, int],
) -> str:
    display = {
        "human": "Human ASAP validation",
        "original_pt": "Original PT",
        "independent_4x2": "independent 4×2",
        "joint_16": "joint 16",
    }
    lines = [
        "# Stage 2 Binary Validation Diagnosis v0",
        "",
        "## Outcome",
        "",
        "고정 ASAP validation 19 pieces / 71 human performances와 기존 Stage 1 cache만 사용해 Human, Original PT, independent 4×2 best epoch 2, joint 16 best epoch 3을 비교했다. 두 Stage 2 checkpoint는 deterministic argmax로 decode했으며 sampling, calibration, threshold 변경, Stage 1 regeneration은 수행하지 않았다. ASAP test MIDI access는 **0**이다.",
        "",
        "Global primary metric은 training report와 허용오차 `1e-12` 안에서 재현됐다. 따라서 **independent 4×2 best epoch 2를 final primary candidate**, **joint 16 best epoch 3을 fixed comparison model**로 lock한다.",
        "",
        "## Metric provenance note",
        "",
        "Human reference는 existing validation int16 cache를 사용하며 기존 histogram과 exact 일치한다. Original PT baseline은 기존 evaluation과 동일하게 cached `original_pt.mid` 19개를 official tokenizer로 재-tokenization한다. Pre-render `generated_ids_int64.npy`를 직접 집계하면 note 수는 같지만 MIDI mapping/round-trip의 pedal sampling 때문에 histogram이 달라지므로 baseline source로 사용하지 않았다. Stage 2 후보는 training checkpoint selection과 동일하게 cached Stage 1 IDs의 Pedal1–4만 교체한 결과를 직접 집계한다. 모든 Stage 2 piece에서 non-pedal token exact equality를 assert했다.",
        "",
        "## Global metric reproduction",
        "",
        "| Candidate | notes | JS distance ↓ | Intersection ↑ | expected JS | expected Intersection | result |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    expected = {
        "original_pt": {"js_distance": 0.14408183938722538, "intersection": 0.9075316431441947},
        **EXPECTED_GLOBAL,
    }
    for candidate in ("original_pt", "independent_4x2", "joint_16"):
        metric = global_metrics[candidate]
        lines.append(
            f"| {display[candidate]} | {int(histograms[candidate].sum()):,} | "
            f"{metric['js_distance']:.12f} | {metric['intersection']:.12f} | "
            f"{expected[candidate]['js_distance']:.12f} | {expected[candidate]['intersection']:.12f} | PASS |"
        )
    lines += [
        "",
        "Independent와 joint inference는 각각 "
        f"{windows['independent_4x2']} windows와 {windows['joint_16']} windows를 처리했다. Stage 1 neural inference 재실행은 0이다.",
        "",
        "## Global 16-pattern comparison",
        "",
        "Pattern order는 `0000, 0001, ..., 1111`이다. Probability는 official PT metric의 epsilon `1e-10` 추가 후 renormalization semantics를 따른다.",
        "",
        "| pattern | Human | Original PT | independent 4×2 | joint 16 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in global_rows:
        lines.append(
            f"| `{row['pattern']}` | {float(row['human_probability']):.9f} | "
            f"{float(row['original_pt_probability']):.9f} | "
            f"{float(row['independent_4x2_probability']):.9f} | "
            f"{float(row['joint_16_probability']):.9f} |"
        )
    lines += ["", "Largest pattern errors versus Human:", ""]
    for candidate in ("original_pt", "independent_4x2", "joint_16"):
        descriptions = []
        for row in _top_pattern_errors(global_rows, candidate):
            descriptions.append(
                f"`{row['pattern']}` {float(row[f'{candidate}_signed_error']):+.6f}"
            )
        lines.append(f"- {display[candidate]}: " + ", ".join(descriptions))
    lines += [
        "",
        "Signed/absolute error 전체 값은 `global_joint16_comparison.csv`에 있다.",
        "",
        "## Steady and transition-containing mass",
        "",
        "Steady mass는 `P(0000)+P(1111)`, transition-containing mass는 나머지 14개 binary pattern의 합이다.",
        "",
        "| Candidate | steady mass | transition-containing mass | transition error vs Human |",
        "|---|---:|---:|---:|",
    ]
    human_steady = float(probabilities["human"][0] + probabilities["human"][15])
    for candidate in ("human", "original_pt", "independent_4x2", "joint_16"):
        steady = float(probabilities[candidate][0] + probabilities[candidate][15])
        transition = 1.0 - steady
        human_transition = 1.0 - human_steady
        lines.append(
            f"| {display[candidate]} | {steady:.9f} | {transition:.9f} | "
            f"{transition - human_transition:+.9f} |"
        )
    lines += [
        "",
        "Original PT는 Human보다 transition-containing binary patterns를 0.060681575 적게 생성했다. Independent 4×2는 이 부족분을 0.013518324까지 줄여 Human 쪽으로 이동한 반면, joint 16의 부족분은 0.068392132로 Original PT보다 더 커졌다. 이는 음악적 pedal quality 자체가 아니라 **PT의 네 within-IOI sampling point 사이에서 binary state가 달라지는 pattern mass**에 대한 기술 통계다.",
        "",
        "## Slot-wise OFF/ON marginals",
        "",
        "| slot | Human ON | Original PT ON | independent ON | joint ON |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in slot_rows:
        lines.append(
            f"| {row['pedal_slot']} | {float(row['human_on_probability']):.9f} | "
            f"{float(row['original_pt_on_probability']):.9f} | "
            f"{float(row['independent_4x2_on_probability']):.9f} | "
            f"{float(row['joint_16_on_probability']):.9f} |"
        )
    lines += [
        "",
        "OFF/ON signed 및 absolute error는 `slotwise_binary_comparison.csv`에 있다. 이 marginal audit은 16-pattern 결과를 설명하는 supplemental diagnostic이며 selection criterion이 아니다.",
        "",
        "## Piece-level sanity analysis",
        "",
        "각 piece의 모든 Human validation performances를 aggregate하고, 동일 piece의 한 Original PT/Stage 2 candidate와 비교했다. Global selection은 변경하지 않는다.",
        "",
        "| Candidate | improved vs Original PT | mean piece JS | median piece JS |",
        "|---|---:|---:|---:|",
    ]
    for candidate in ARCHITECTURES:
        summary = piece_summaries[candidate]
        lines.append(
            f"| {display[candidate]} | {summary['improved_piece_count']}/19 | "
            f"{summary['mean_js_distance']:.9f} | {summary['median_js_distance']:.9f} |"
        )
    original_js = [float(row["original_pt_js_distance"]) for row in piece_rows]
    lines.append(
        f"| Original PT reference | — | {statistics.fmean(original_js):.9f} | {statistics.median(original_js):.9f} |"
    )
    lines += ["", "Best/worst JS improvement examples (`Original PT JS - model JS`):", ""]
    for candidate in ARCHITECTURES:
        summary = piece_summaries[candidate]
        best = "; ".join(
            f"{item['composer']} {item['title']} {item['js_improvement']:+.6f}"
            for item in summary["best_improvements"]
        )
        worst = "; ".join(
            f"{item['composer']} {item['title']} {item['js_improvement']:+.6f}"
            for item in summary["worst_improvements"]
        )
        lines += [f"- {display[candidate]} best: {best}", f"- {display[candidate]} worst: {worst}"]
    lines += [
        "",
        "전체 19-piece JS/Intersection과 improvement는 `piece_level_metrics.csv`에 있다.",
        "",
        "## Final selection and lock",
        "",
        f"- Primary: **independent 4×2 best epoch 2**, checkpoint SHA-256 `{checkpoint_hashes['independent_4x2']}`",
        f"- Fixed comparison: **joint 16 best epoch 3**, checkpoint SHA-256 `{checkpoint_hashes['joint_16']}`",
        "- Primary rule: global validation Pedal JS distance lower; numerical tie이면 Intersection higher",
        "- Binary threshold: 64; decoding: deterministic argmax",
        "- No test-time calibration, sampling, temperature change, or post-lock hyperparameter change",
        "- Machine-readable lock: `analysis/stage2_binary_v0/final_lock_v0/final_experiment_lock.json`",
        "",
        "## Integrity",
        "",
        "- Global metrics reproduce training/baseline reports: PASS",
        "- Human and Original PT saved global histograms exact: PASS",
        "- Stage 2 non-pedal token exact equality for all 19 pieces: PASS",
        "- Best checkpoint epochs and checkpoint metadata: PASS",
        "- Deterministic argmax, threshold 64, fixed 512/256 inference: PASS",
        "- Stage 1 inference regeneration: 0",
        "- ASAP test MIDI access: **0 / PASS**",
        "",
        "No training, checkpoint update, calibration, sampling experiment, threshold change, Stage 1 regeneration, test evaluation, or audio rendering was performed.",
        "",
    ]
    return "\n".join(lines)


def _lock(
    global_metrics: Mapping[str, Mapping[str, float]],
    checkpoint_hashes: Mapping[str, str],
    output_hashes: Mapping[str, str],
) -> dict[str, Any]:
    candidates: dict[str, Any] = {}
    for architecture, role in (
        ("independent_4x2", "primary"),
        ("joint_16", "fixed_comparison"),
    ):
        checkpoint_path = CHECKPOINTS[architecture]
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        run_config_path = TRAIN_ROOT / architecture / "config.json"
        run_status_path = TRAIN_ROOT / architecture / "run_status.json"
        run_config = _read_json(run_config_path)
        if int(payload["best_epoch"]) != EXPECTED_BEST_EPOCH[architecture]:
            raise RuntimeError("best checkpoint epoch changed before lock")
        candidates[architecture] = {
            "role": role,
            "architecture": architecture,
            "model_class": (
                "src.stage2_binary.model.IndependentBinaryPedalModel"
                if architecture == "independent_4x2"
                else "src.stage2_binary.model.JointBinaryPedalModel"
            ),
            "best_epoch": int(payload["best_epoch"]),
            "checkpoint_project_relative_path": str(checkpoint_path.relative_to(PROJECT_ROOT)),
            "checkpoint_container_path": "/workspace/project/" + str(checkpoint_path.relative_to(PROJECT_ROOT)),
            "checkpoint_sha256": checkpoint_hashes[architecture],
            "checkpoint_bytes": checkpoint_path.stat().st_size,
            "checkpoint_selection_verified": True,
            "run_config_project_relative_path": str(run_config_path.relative_to(PROJECT_ROOT)),
            "run_config_sha256": _sha256(run_config_path),
            "run_status_project_relative_path": str(run_status_path.relative_to(PROJECT_ROOT)),
            "run_status_sha256": _sha256(run_status_path),
            "initial_encoder_parameter_sha256": run_config["initial_encoder_parameter_sha256"],
            "training_cache_id": run_config["cache_id"],
            "validation_global_js_distance": global_metrics[architecture]["js_distance"],
            "validation_global_intersection": global_metrics[architecture]["intersection"],
            "decoding": "deterministic argmax",
            "binary_threshold": BINARY_THRESHOLD,
            "output_raw_pedal_values": [0, 127],
        }
        del payload
    return {
        "lock_version": "stage2_binary_final_lock_v0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "locked_before_asap_test",
        "selection": {
            "primary_candidate": "independent_4x2",
            "fixed_comparison_candidate": "joint_16",
            "primary_metric": "global validation PT-style Pedal JS distance (lower)",
            "secondary_tiebreak": "global validation Pedal Intersection (higher) on numerical JS tie",
            "selection_changed_by_piece_or_slot_diagnostics": False,
        },
        "candidates": candidates,
        "common_inference_lock": {
            "stage1": "official Original Pianist Transformer",
            "stage1_checkpoint_project_relative_path": "checkpoints/pianist_transformer",
            "stage1_checkpoint_model_safetensors_sha256": _sha256(ENCODER_CHECKPOINT / "model.safetensors"),
            "stage1_validation_cache_project_relative_path": "analysis/stage2_binary_v0/validation_eval_v0/stage1_cache",
            "stage1_cache_manifest_sha256": _sha256(VALIDATION_ROOT / "stage1_cache_manifest.csv"),
            "stage1_inference_regeneration": False,
            "stage2_input_note_tokens": ["Pitch", "IOI", "Velocity", "Duration", "MASK", "MASK", "MASK", "MASK"],
            "stage2_changes": ["Pedal1", "Pedal2", "Pedal3", "Pedal4"],
            "non_pedal_token_exact_equality_required": True,
            "window_notes": 512,
            "stride_notes": 256,
            "overlap_merge": "mean logits per note before argmax",
            "binary_threshold": 64,
            "binary_rule": "raw <64 -> 0; raw >=64 -> 1",
            "joint_id": "8*P1 + 4*P2 + 2*P3 + P4",
            "pattern_order": list(JOINT_PATTERNS),
            "decoding": "deterministic argmax",
            "sampling": False,
            "temperature_calibration": False,
            "official_pt_evaluator_semantics": {
                "metric": "base-2 Jensen-Shannon distance",
                "epsilon": 1e-10,
                "renormalize_after_epsilon": True,
                "intersection": "sum(min(human_probability, candidate_probability))",
            },
            "test_time_calibration": False,
            "hyperparameter_changes_after_lock": False,
        },
        "source_provenance": {
            "binary_model_source": str(MODEL_SOURCE.relative_to(PROJECT_ROOT)),
            "binary_model_source_sha256": _sha256(MODEL_SOURCE),
            "validation_evaluator_source": str(EVALUATOR_SOURCE.relative_to(PROJECT_ROOT)),
            "validation_evaluator_source_sha256": _sha256(EVALUATOR_SOURCE),
            "official_tokenizer_source": str(TOKENIZER_SOURCE.relative_to(PROJECT_ROOT)),
            "official_tokenizer_source_sha256": _sha256(TOKENIZER_SOURCE),
            "official_evaluator_source": str(OFFICIAL_EVALUATOR_SOURCE.relative_to(PROJECT_ROOT)),
            "official_evaluator_source_sha256": _sha256(OFFICIAL_EVALUATOR_SOURCE),
        },
        "validation_lock_evidence": {
            "global_metrics_reproduced": True,
            "diagnosis_outputs_sha256": dict(output_hashes),
            "original_pt_js_distance": global_metrics["original_pt"]["js_distance"],
            "original_pt_intersection": global_metrics["original_pt"]["intersection"],
            "human_validation_performances": 71,
            "validation_pieces": 19,
            "stage1_cached_candidates": 19,
        },
        "prohibited_after_lock": [
            "training or checkpoint update",
            "sampling or temperature sweep",
            "calibration",
            "threshold change",
            "class or dataset weighting",
            "architecture change",
            "Stage 1 regeneration",
        ],
        "asap_test": {
            "midi_access_count_during_validation_diagnosis": 0,
            "evaluation_performed": False,
        },
    }


def main() -> None:
    if DIAGNOSIS_ROOT.exists() or LOCK_ROOT.exists():
        raise FileExistsError("refusing to overwrite validation diagnosis or final lock")
    DIAGNOSIS_ROOT.mkdir(parents=True)
    LOCK_ROOT.mkdir(parents=True)
    baseline = _read_json(VALIDATION_ROOT / "original_pt_validation_metrics.json")
    stage1_rows = _read_csv(VALIDATION_ROOT / "stage1_cache_manifest.csv")
    if len(stage1_rows) != 19:
        raise RuntimeError("Stage 1 validation cache must contain 19 pieces")
    human_piece, human_metadata = _human_piece_histograms()
    human_global = sum(human_piece.values(), np.zeros(16, dtype=np.int64))
    if human_global.tolist() != baseline["human_histogram"]:
        raise RuntimeError("human validation histogram changed")
    original_piece, original_notes, original_global = _original_pt_piece_histograms(stage1_rows)
    if original_global.tolist() != baseline["original_pt_histogram"]:
        raise RuntimeError("Original PT validation histogram did not reproduce")
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("best-checkpoint validation diagnosis requires CUDA")
    stage2_piece: dict[str, dict[str, np.ndarray]] = {}
    stage2_notes: dict[str, dict[str, int]] = {}
    stage2_global: dict[str, np.ndarray] = {}
    windows: dict[str, int] = {}
    for architecture in ARCHITECTURES:
        payload = torch.load(CHECKPOINTS[architecture], map_location="cpu", weights_only=False)
        if int(payload["best_epoch"]) != EXPECTED_BEST_EPOCH[architecture]:
            raise RuntimeError(f"unexpected best epoch: {architecture}")
        del payload
        piece, notes, histogram, window_count = _stage2_piece_histograms(
            architecture, stage1_rows, device
        )
        stage2_piece[architecture] = piece
        stage2_notes[architecture] = notes
        stage2_global[architecture] = histogram
        windows[architecture] = window_count
    histograms = {
        "human": human_global,
        "original_pt": original_global,
        **stage2_global,
    }
    probabilities = {
        name: normalized_joint16(histogram) for name, histogram in histograms.items()
    }
    global_metrics: dict[str, dict[str, float]] = {}
    expected_metrics = {
        "original_pt": {
            "js_distance": float(baseline["metrics"]["js_distance_base2"]),
            "intersection": float(baseline["metrics"]["histogram_intersection"]),
        },
        **EXPECTED_GLOBAL,
    }
    for candidate in ("original_pt", *ARCHITECTURES):
        metric = official_pt_pedal_similarity(human_global, histograms[candidate])
        result = {
            "js_distance": float(metric["js_distance_base2"]),
            "js_divergence": float(metric["js_divergence_base2"]),
            "intersection": float(metric["histogram_intersection"]),
        }
        global_metrics[candidate] = result
        if not math.isclose(
            result["js_distance"], expected_metrics[candidate]["js_distance"],
            rel_tol=0.0, abs_tol=GLOBAL_TOLERANCE,
        ) or not math.isclose(
            result["intersection"], expected_metrics[candidate]["intersection"],
            rel_tol=0.0, abs_tol=GLOBAL_TOLERANCE,
        ):
            raise RuntimeError(
                f"global validation metric mismatch for {candidate}: "
                f"{result} expected {expected_metrics[candidate]}"
            )
    global_rows = _global_csv_rows(histograms, probabilities)
    slot_rows = _slotwise_rows(probabilities)
    candidates = {
        "original_pt": original_piece,
        **stage2_piece,
    }
    candidate_notes = {
        "original_pt": original_notes,
        **stage2_notes,
    }
    piece_rows = _piece_rows(human_piece, human_metadata, candidates, candidate_notes)
    piece_summaries = {
        architecture: _piece_summary(piece_rows, architecture)
        for architecture in ARCHITECTURES
    }
    global_path = DIAGNOSIS_ROOT / "global_joint16_comparison.csv"
    _write_csv(global_path, tuple(global_rows[0]), global_rows)
    slot_path = DIAGNOSIS_ROOT / "slotwise_binary_comparison.csv"
    _write_csv(slot_path, tuple(slot_rows[0]), slot_rows)
    piece_path = DIAGNOSIS_ROOT / "piece_level_metrics.csv"
    _write_csv(piece_path, tuple(piece_rows[0]), piece_rows)
    checkpoint_hashes = {
        architecture: _sha256(path) for architecture, path in CHECKPOINTS.items()
    }
    summary = {
        "completed": True,
        "global_metrics": global_metrics,
        "global_metrics_reproduced": True,
        "steady_transition": {
            name: {
                "steady_mass": float(probability[0] + probability[15]),
                "transition_containing_mass": float(1.0 - probability[0] - probability[15]),
            }
            for name, probability in probabilities.items()
        },
        "piece_summaries": piece_summaries,
        "checkpoint_sha256": checkpoint_hashes,
        "stage2_windows": windows,
        "stage1_inference_regenerated": False,
        "asap_test_midi_access_count": 0,
        "final_primary_candidate": "independent_4x2",
        "fixed_comparison_candidate": "joint_16",
    }
    summary_path = DIAGNOSIS_ROOT / "diagnosis_summary.json"
    _write_json(summary_path, summary)
    report_path = DIAGNOSIS_ROOT / "VALIDATION_DIAGNOSIS_REPORT.md"
    report_path.write_text(
        _report(
            histograms,
            probabilities,
            global_metrics,
            global_rows,
            slot_rows,
            piece_rows,
            piece_summaries,
            checkpoint_hashes,
            windows,
        ),
        encoding="utf-8",
    )
    evidence_paths = (global_path, slot_path, piece_path, summary_path, report_path)
    output_hashes = {
        str(path.relative_to(PROJECT_ROOT)): _sha256(path) for path in evidence_paths
    }
    lock = _lock(global_metrics, checkpoint_hashes, output_hashes)
    _write_json(LOCK_ROOT / "final_experiment_lock.json", lock)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
