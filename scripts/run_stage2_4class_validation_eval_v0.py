#!/usr/bin/env python3
"""Canonical ASAP-validation evaluation for Original PT and three 4C models."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from miditoolkit import MidiFile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_pedal_event_metric_tolerance_mini import (
    build_score_positions,
    extract_transitions,
    raw_events,
    tick_ms_fn,
)
from scripts.evaluate_original_pt_early_metrics import (
    PinnedTokenizerConfig,
    aligned_full_tokens,
)
from src.stage2_binary.canonical_stage1 import (
    assert_strict_non_cc64_equality,
    sha256_file,
    signature_sha256,
    transplant_cc64_only,
)
from src.stage2_four_class.validation_evaluator import (
    MODEL_LABELS,
    MODEL_NAMES,
    PATTERN_COUNT,
    TRANSITION_TOLERANCE,
    canonical_classes_from_tokens,
    classification_metrics,
    confusion_from_pairs,
    infer_four_class_pedals,
    load_four_class_model,
    pattern_ids,
    pattern_metrics,
    pooled_transition_counts,
)
from third_party.PianistTransformer.src import model as _official_model_package
from third_party.PianistTransformer.src import utils as _official_utils_package
from third_party.PianistTransformer.src.model import pianoformer as _official_pianoformer
from third_party.PianistTransformer.src.utils import midi as _official_midi

sys.modules.setdefault("src.model", _official_model_package)
sys.modules.setdefault("src.model.pianoformer", _official_pianoformer)
sys.modules.setdefault("src.utils", _official_utils_package)
sys.modules.setdefault("src.utils.midi", _official_midi)

from third_party.PianistTransformer.src.model.generate import map_midi
from third_party.PianistTransformer.src.utils.midi import ids_to_midi, midi_to_ids


OUTPUT_ROOT = ROOT / "analysis/stage2_4class_architecture_validation_eval_v0"
REPORT_NAME = "FOUR_MODEL_VALIDATION_EVALUATION_REPORT.md"
STAGE1_MANIFEST = ROOT / "analysis/stage2_binary_canonical_v1/canonical_validation_stage1_manifest.csv"
SPLIT_CSV = ROOT / "analysis/stage2_encoder_only_v0/asap_split.csv"
ASAP_ROOT = Path("/workspace/public/ASAP/asap-dataset-v1.1")
PRETRAINED = ROOT / "checkpoints/pianist_transformer"
ALIGNMENT_TOOL = Path("/tmp/original_pt_early_alignment/tools/AlignmentTool")
CHECKPOINTS = {
    "encoder_only": ROOT / "analysis/stage2_4class_architecture_v0/encoder_only_ce/best.pt",
    "decoder_only": ROOT / "analysis/stage2_decoder_only_4class_v0/full_train/best.pt",
    "encoder_decoder": ROOT / "analysis/stage2_4class_architecture_v0/encoder_decoder_ce/best.pt",
}
EXPECTED_CHECKPOINTS = {
    "encoder_only": (2, 0.8925421636047081),
    "decoder_only": (8, 0.17593201818653073),
    "encoder_decoder": (4, 0.20062611097252214),
}
CLASS_NAMES = ("ZERO", "LOW", "HALF", "FULL")


def now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
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


def ids_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(value, dtype="<i8").tobytes()).hexdigest()


def update_status(output: Path, **updates: Any) -> dict[str, Any]:
    path = output / "run_status.json"
    current = json.loads(path.read_text()) if path.is_file() else {}
    current.update(updates, last_update=now())
    atomic_json(path, current)
    return current


def audit_inputs() -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    stage1 = read_csv(STAGE1_MANIFEST)
    validation = [row for row in read_csv(SPLIT_CSV) if row["split"] == "validation"]
    if len(stage1) != 19 or len({row["piece_id"] for row in stage1}) != 19:
        raise RuntimeError("canonical Stage 1 manifest is not 19 unique pieces")
    if len(validation) != 71 or len({row["piece_id"] for row in validation}) != 19:
        raise RuntimeError("ASAP validation split is not 19 pieces / 71 performances")
    if {row["piece_id"] for row in stage1} != {row["piece_id"] for row in validation}:
        raise RuntimeError("Stage 1 and validation piece universes differ")
    by_piece = {piece: 0 for piece in {row["piece_id"] for row in validation}}
    for row in validation:
        by_piece[row["piece_id"]] += 1
        if row["split"] != "validation":
            raise AssertionError("non-validation row entered evaluation")
    checkpoint_metadata: dict[str, Any] = {}
    for name, checkpoint_path in CHECKPOINTS.items():
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        epoch, validation_ce = EXPECTED_CHECKPOINTS[name]
        if int(payload["epoch"]) != epoch or not math.isclose(
            float(payload["validation_ce"]), validation_ce, rel_tol=0.0, abs_tol=1e-12
        ):
            raise RuntimeError(f"unexpected canonical checkpoint metadata: {name}")
        checkpoint_metadata[name] = {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "epoch": epoch,
            "validation_ce": validation_ce,
        }
        del payload
    for row in stage1:
        if row["status"] != "frozen_pass":
            raise RuntimeError(f"non-frozen Stage 1 row: {row['piece_id']}")
        midi = Path(row["canonical_midi_path"])
        ids = Path(row["generated_ids_path"])
        if not midi.is_file() or not ids.is_file():
            raise FileNotFoundError(f"missing Stage 1 artifact: {row['piece_id']}")
        if sha256_file(midi) != row["canonical_midi_sha256"]:
            raise RuntimeError(f"Stage 1 MIDI hash mismatch: {row['piece_id']}")
        if signature_sha256(midi) != row["canonical_non_cc64_signature_sha256"]:
            raise RuntimeError(f"Stage 1 non-CC64 signature mismatch: {row['piece_id']}")
        if int(row["validation_performance_count"]) != by_piece[row["piece_id"]]:
            raise RuntimeError(f"piece pairing count mismatch: {row['piece_id']}")
    return stage1, validation, {
        "stage1_manifest": str(STAGE1_MANIFEST),
        "stage1_manifest_sha256": sha256_file(STAGE1_MANIFEST),
        "validation_split": str(SPLIT_CSV),
        "validation_split_sha256": sha256_file(SPLIT_CSV),
        "pieces": 19,
        "human_performances": 71,
        "candidate_human_pairs_per_model": 71,
        "asap_test_access_count": 0,
        "original_pt_neural_inference_count": 0,
        "checkpoints": checkpoint_metadata,
    }


def prediction_paths(output: Path, model_name: str, piece_id: str) -> dict[str, Path]:
    root = output / "predictions" / model_name / piece_id
    return {
        "root": root,
        "metadata": root / "metadata.json",
        "classes": root / "classes_int64.npy",
        "ids": root / "candidate_ids_int64.npy",
        "midi": root / "candidate.mid",
        "donor": root / "pedal_donor.mid",
    }


def valid_prediction_cache(
    paths: Mapping[str, Path], source_hash: str, checkpoint_hash: str | None
) -> bool:
    try:
        metadata = json.loads(paths["metadata"].read_text())
        required = ("classes", "ids", "midi") if checkpoint_hash else ("midi",)
        if not all(paths[name].is_file() for name in required):
            return False
        if metadata["source_ids_sha256"] != source_hash:
            return False
        if metadata.get("checkpoint_sha256") != checkpoint_hash:
            return False
        if sha256_file(paths["midi"]) != metadata["candidate_midi_sha256"]:
            return False
        if checkpoint_hash:
            classes = np.load(paths["classes"], allow_pickle=False)
            ids = np.load(paths["ids"], allow_pickle=False)
            if ids_sha256(classes) != metadata["classes_sha256"]:
                return False
            if ids_sha256(ids) != metadata["candidate_ids_sha256"]:
                return False
        return bool(metadata["complete"])
    except Exception:
        return False


def cache_original(output: Path, row: Mapping[str, str]) -> dict[str, Any]:
    paths = prediction_paths(output, "original_pt", row["piece_id"])
    source_ids = np.load(row["generated_ids_path"], allow_pickle=False)
    source_hash = ids_sha256(source_ids)
    if valid_prediction_cache(paths, source_hash, None):
        return json.loads(paths["metadata"].read_text())
    paths["root"].mkdir(parents=True, exist_ok=True)
    temporary = paths["midi"].with_name(".candidate.mid.tmp")
    shutil.copy2(row["canonical_midi_path"], temporary)
    os.replace(temporary, paths["midi"])
    metadata = {
        "complete": True,
        "model": "original_pt",
        "piece_id": row["piece_id"],
        "source_ids_sha256": source_hash,
        "checkpoint_sha256": None,
        "candidate_midi": str(paths["midi"]),
        "candidate_midi_sha256": sha256_file(paths["midi"]),
        "saved_canonical_midi_reused": True,
        "new_pt_inference": False,
        "non_pedal_identity_pass": signature_sha256(paths["midi"])
        == row["canonical_non_cc64_signature_sha256"],
    }
    if not metadata["non_pedal_identity_pass"]:
        raise AssertionError("copied Original PT MIDI identity failed")
    atomic_json(paths["metadata"], metadata)
    return metadata


def render_stage2_candidate(
    output: Path,
    model_name: str,
    row: Mapping[str, str],
    classes: np.ndarray,
    candidate_ids: np.ndarray,
    inference: Mapping[str, Any],
    checkpoint_hash: str,
    tokenizer_config: Any,
) -> dict[str, Any]:
    paths = prediction_paths(output, model_name, row["piece_id"])
    paths["root"].mkdir(parents=True, exist_ok=True)
    atomic_npy(paths["classes"], classes.astype(np.int64))
    atomic_npy(paths["ids"], candidate_ids.astype(np.int64))
    score_midi = MidiFile(row["selected_score_absolute_path"])
    score_ids = midi_to_ids(tokenizer_config, score_midi)
    performance = ids_to_midi(tokenizer_config, candidate_ids, ref=score_ids)
    mapped = map_midi(score_midi, performance)
    donor_temp = paths["donor"].with_name(".pedal_donor.mid.tmp")
    mapped.dump(str(donor_temp))
    os.replace(donor_temp, paths["donor"])
    candidate_temp = paths["midi"].with_name(".candidate.mid.tmp")
    if candidate_temp.exists():
        candidate_temp.unlink()
    identity = transplant_cc64_only(
        row["canonical_midi_path"], paths["donor"], candidate_temp
    )
    os.replace(candidate_temp, paths["midi"])
    identity = assert_strict_non_cc64_equality(
        row["canonical_midi_path"], paths["midi"]
    )
    if signature_sha256(paths["midi"]) != row["canonical_non_cc64_signature_sha256"]:
        raise AssertionError("Stage 2 candidate non-pedal MIDI signature changed")
    metadata = {
        "complete": True,
        "model": model_name,
        "piece_id": row["piece_id"],
        "source_ids_sha256": row["generated_token_sha256_int64_le"],
        "checkpoint_sha256": checkpoint_hash,
        "classes_sha256": ids_sha256(classes),
        "candidate_ids_sha256": ids_sha256(candidate_ids),
        "candidate_midi": str(paths["midi"]),
        "candidate_midi_sha256": sha256_file(paths["midi"]),
        "non_pedal_identity_pass": True,
        "midi_identity": identity,
        "inference": dict(inference),
    }
    atomic_json(paths["metadata"], metadata)
    return metadata


def alignment_paths(output: Path, kind: str, identifier: str) -> dict[str, Path]:
    root = output / "alignment" / kind / identifier
    return {
        "root": root,
        "metadata": root / "metadata.json",
        "tokens": root / "aligned_tokens_int64.npy",
        "transitions": root / "transitions.json",
        "process_log": output / "alignment" / "logs" / f"{kind}_{identifier}.log",
    }


def align_and_cache(
    output: Path,
    *,
    kind: str,
    identifier: str,
    score_path: Path,
    performance_path: Path,
) -> dict[str, Any]:
    paths = alignment_paths(output, kind, identifier)
    source_hash = sha256_file(performance_path)
    try:
        metadata = json.loads(paths["metadata"].read_text())
        if (
            metadata.get("complete")
            and metadata.get("performance_sha256") == source_hash
            and paths["tokens"].is_file()
            and paths["transitions"].is_file()
            and ids_sha256(np.load(paths["tokens"], allow_pickle=False))
            == metadata["aligned_tokens_sha256"]
        ):
            return {
                "tokens": np.load(paths["tokens"], allow_pickle=False),
                "transitions": json.loads(paths["transitions"].read_text()),
                "audit": metadata["audit"],
                "cache_reused": True,
            }
    except Exception:
        pass
    paths["root"].mkdir(parents=True, exist_ok=True)
    paths["process_log"].parent.mkdir(parents=True, exist_ok=True)
    aligned_tokens, audit = aligned_full_tokens(
        PinnedTokenizerConfig(), score_path, performance_path, ALIGNMENT_TOOL
    )
    midi = MidiFile(str(performance_path))
    onsets, controls = raw_events(midi)
    positions = build_score_positions(
        score_path,
        performance_path,
        onsets,
        ALIGNMENT_TOOL,
        paths["process_log"],
    )
    tick_to_ms = tick_ms_fn(midi)
    onset_ms = [tick_to_ms(tick) for tick in onsets]
    transitions = extract_transitions(onsets, onset_ms, controls, tick_to_ms)
    for event in transitions:
        event["score_position"] = positions[event["onset_index"]]
    if any(event["score_position"] is None for event in transitions):
        raise RuntimeError("transition could not be mapped to score position")
    atomic_npy(paths["tokens"], aligned_tokens.astype(np.int64))
    atomic_json(paths["transitions"], {"transitions": transitions})
    metadata = {
        "complete": True,
        "kind": kind,
        "identifier": identifier,
        "score_path": str(score_path),
        "performance_path": str(performance_path),
        "performance_sha256": source_hash,
        "aligned_tokens_sha256": ids_sha256(aligned_tokens),
        "aligned_notes": len(aligned_tokens),
        "transition_count": len(transitions),
        "audit": audit,
    }
    atomic_json(paths["metadata"], metadata)
    return {
        "tokens": aligned_tokens,
        "transitions": {"transitions": transitions},
        "audit": audit,
        "cache_reused": False,
    }


def sum_transition(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    for scope in ("pooled", "up", "down"):
        for key in ("candidate", "reference", "tp", "fp", "fn"):
            target[scope][key] += int(source[scope][key])


def finalize_transition(value: dict[str, Any]) -> dict[str, Any]:
    for scope in ("pooled", "up", "down"):
        row = value[scope]
        row["precision"] = row["tp"] / (row["tp"] + row["fp"]) if row["tp"] + row["fp"] else 0.0
        row["recall"] = row["tp"] / (row["tp"] + row["fn"]) if row["tp"] + row["fn"] else 0.0
        row["f1"] = 2 * row["precision"] * row["recall"] / (row["precision"] + row["recall"]) if row["precision"] + row["recall"] else 0.0
    value["pooled"]["tolerance_distinct_onsets"] = TRANSITION_TOLERANCE
    return value


def pattern_name(index: int) -> str:
    values = ((index // 64) % 4, (index // 16) % 4, (index // 4) % 4, index % 4)
    return " ".join(CLASS_NAMES[value] for value in values)


def top_patterns(histogram: np.ndarray, count: int = 20) -> list[dict[str, Any]]:
    total = int(histogram.sum())
    order = sorted(range(PATTERN_COUNT), key=lambda index: (-int(histogram[index]), index))[:count]
    return [
        {
            "pattern_id": index,
            "pattern": pattern_name(index),
            "count": int(histogram[index]),
            "probability": float(histogram[index] / total),
        }
        for index in order
    ]


def evaluate_metrics(
    output: Path,
    stage1: Sequence[Mapping[str, str]],
    validation: Sequence[Mapping[str, str]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    human_alignments: dict[str, dict[str, Any]] = {}
    candidate_alignments: dict[tuple[str, str], dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    validation_by_piece: dict[str, list[Mapping[str, str]]] = {}
    for row in validation:
        validation_by_piece.setdefault(row["piece_id"], []).append(row)
    for row in stage1:
        piece_id = row["piece_id"]
        score = Path(row["selected_score_absolute_path"])
        for human in validation_by_piece[piece_id]:
            identifier = human["metadata_index"]
            try:
                human_alignments[identifier] = align_and_cache(
                    output,
                    kind="human",
                    identifier=identifier,
                    score_path=score,
                    performance_path=ASAP_ROOT / human["performance_path"],
                )
            except Exception as error:
                failures.append({
                    "piece": piece_id,
                    "candidate_model": "shared_human",
                    "human_performance": human["performance_path"],
                    "reason": f"{type(error).__name__}: {error}",
                })
        for model_name in MODEL_NAMES:
            metadata = json.loads(
                prediction_paths(output, model_name, piece_id)["metadata"].read_text()
            )
            try:
                candidate_alignments[(model_name, piece_id)] = align_and_cache(
                    output,
                    kind=model_name,
                    identifier=piece_id,
                    score_path=score,
                    performance_path=Path(metadata["candidate_midi"]),
                )
            except Exception as error:
                failures.append({
                    "piece": piece_id,
                    "candidate_model": model_name,
                    "human_performance": "",
                    "reason": f"{type(error).__name__}: {error}",
                })
    common_pairs = []
    for human in validation:
        piece_id, identifier = human["piece_id"], human["metadata_index"]
        if identifier not in human_alignments:
            continue
        if all((model_name, piece_id) in candidate_alignments for model_name in MODEL_NAMES):
            common_pairs.append((piece_id, identifier, human))
    if not common_pairs:
        raise RuntimeError("alignment produced no common comparison pairs")
    results: dict[str, Any] = {}
    for model_name in MODEL_NAMES:
        confusion = np.zeros((4, 4), dtype=np.int64)
        candidate_histogram = np.zeros(PATTERN_COUNT, dtype=np.int64)
        target_histogram = np.zeros(PATTERN_COUNT, dtype=np.int64)
        transition = {
            scope: {key: 0 for key in ("candidate", "reference", "tp", "fp", "fn")}
            for scope in ("pooled", "up", "down")
        }
        aligned_notes = 0
        for piece_id, identifier, _ in common_pairs:
            candidate = candidate_alignments[(model_name, piece_id)]
            target = human_alignments[identifier]
            candidate_classes = canonical_classes_from_tokens(candidate["tokens"])
            target_classes = canonical_classes_from_tokens(target["tokens"])
            if candidate_classes.shape != target_classes.shape:
                raise RuntimeError("aligned candidate/reference shapes differ")
            confusion += confusion_from_pairs(candidate_classes, target_classes)
            candidate_histogram += np.bincount(
                pattern_ids(candidate_classes), minlength=PATTERN_COUNT
            )
            target_histogram += np.bincount(
                pattern_ids(target_classes), minlength=PATTERN_COUNT
            )
            counts = pooled_transition_counts(
                candidate["transitions"], target["transitions"]
            )
            sum_transition(transition, counts)
            aligned_notes += len(candidate_classes)
        classification = classification_metrics(confusion)
        patterns = pattern_metrics(candidate_histogram, target_histogram)
        transition = finalize_transition(transition)
        if not all(
            math.isfinite(float(value))
            for value in (
                classification["token_accuracy"],
                classification["macro_f1"],
                patterns["js_divergence_base2"],
                patterns["intersection"],
                transition["pooled"]["f1"],
            )
        ):
            raise FloatingPointError("non-finite canonical evaluation metric")
        results[model_name] = {
            "model": MODEL_LABELS[model_name],
            "candidate_pieces": 19,
            "human_performances": 71,
            "common_successful_pairs": len(common_pairs),
            "aligned_notes": aligned_notes,
            "alignment_pedal_samples": aligned_notes * 4,
            "classification": classification,
            "transition": transition,
            "patterns": {
                **patterns,
                "top_candidate_patterns": top_patterns(candidate_histogram),
                "top_target_patterns": top_patterns(target_histogram),
            },
        }
    return results, failures


def report_text(provenance: Mapping[str, Any], results: Mapping[str, Any], failures: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Four-Model Canonical 4-Class Validation Evaluation",
        "",
        "## A. Scope / frozen definitions",
        "",
        "- ASAP validation only: 19 pieces / 71 human performances",
        "- ASAP test access: 0",
        "- Classes: ZERO 0–25; LOW 26–63; HALF 64–103; FULL 104–127; representatives [0,51,79,127]",
        "- Transition: raw CC64 crossing at 64; UP/DOWN direction-aware one-to-one; tolerance ±1 distinct onset",
        "- Repedal: not evaluated; execution count 0",
        "",
        "## B. Model/checkpoint provenance",
        "",
        "- Original PT: saved canonical validation MIDI reused; no new PT inference",
    ]
    for name in ("encoder_only", "decoder_only", "encoder_decoder"):
        checkpoint = provenance["checkpoints"][name]
        decoding = "deterministic argmax" if name == "encoder_only" else "true free-running greedy; ground-truth history access 0"
        lines.append(
            f"- {MODEL_LABELS[name]}: `{checkpoint['path']}`; best epoch {checkpoint['epoch']}; {decoding}"
        )
    lines += [
        "",
        "## C. Input identity verification",
        "",
        "- All Stage 2 candidate token and MIDI non-pedal signatures (pitch, onset/timing, velocity, duration) matched canonical Original PT: PASS",
        "",
        "## D. Alignment accounting",
        "",
        f"- Candidate pieces: 19; human performances/pairs: 71; failed alignment records: {len(failures)}",
        f"- Common successful comparison pairs: {next(iter(results.values()))['common_successful_pairs']}",
        "- All four models use the same common alignment universe.",
        "",
        "## E. Main metrics",
        "",
        "| Model | 4C Acc ↑ | Macro F1 ↑ | Transition P ↑ | Transition R ↑ | Transition F1 ↑ | JS Div ↓ | Intersection ↑ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in MODEL_NAMES:
        value = results[name]
        classification = value["classification"]
        transition = value["transition"]["pooled"]
        patterns = value["patterns"]
        lines.append(
            f"| {MODEL_LABELS[name]} | {classification['token_accuracy']:.6f} | {classification['macro_f1']:.6f} | "
            f"{transition['precision']:.6f} | {transition['recall']:.6f} | {transition['f1']:.6f} | "
            f"{patterns['js_divergence_base2']:.6f} | {patterns['intersection']:.6f} |"
        )
    lines += ["", "## F. Class diagnostics", ""]
    for name in MODEL_NAMES:
        value = results[name]["classification"]
        lines += [
            f"### {MODEL_LABELS[name]}",
            "",
            f"- Samples: {value['pedal_samples']}; prediction distribution: `{value['prediction_class_distribution']}`",
            f"- Human distribution: `{value['target_class_distribution']}`",
            f"- Class P/R/F1: `{value['class_metrics']}`",
            f"- Confusion matrix rows=human, columns=candidate: `{value['confusion_matrix']}`",
            "",
        ]
    lines += ["## G. Transition diagnostics", ""]
    for name in MODEL_NAMES:
        value = results[name]["transition"]
        lines += [
            f"- {MODEL_LABELS[name]} pooled: `{value['pooled']}`",
            f"  - UP: `{value['up']}`",
            f"  - DOWN: `{value['down']}`",
        ]
    lines += [
        "",
        "## H. 256-pattern diagnostics",
        "",
        "JS is base-2 Jensen-Shannon divergence (not distance), lower is better. Intersection is Σ min(P,Q), higher is better.",
        "",
    ]
    for name in MODEL_NAMES:
        value = results[name]["patterns"]
        lines += [
            f"### {MODEL_LABELS[name]}",
            "",
            f"- JS divergence: {value['js_divergence_base2']:.6f}; JS distance diagnostic: {value['js_distance_base2']:.6f}; intersection: {value['intersection']:.6f}",
            f"- Top predicted patterns: `{value['top_candidate_patterns']}`",
            f"- Top human patterns: `{value['top_target_patterns']}`",
            "",
        ]
    lines += [
        "## I. Interpretation",
        "",
        "The table separately exposes four-class depth fidelity, direction-aware transition behavior, and global joint-pattern distribution. Architecture comparison uses only these free-running/canonical validation metrics; teacher-forced CE is provenance only. No follow-up loss or decoding experiment was started automatically.",
        "",
    ]
    return "\n".join(lines)


def run(output: Path) -> int:
    output.mkdir(parents=True, exist_ok=True)
    log_handle = (output / "evaluation.log").open("a", encoding="utf-8")

    def log(message: str) -> None:
        line = f"{now()} {message}"
        print(line, flush=True)
        print(line, file=log_handle, flush=True)

    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("evaluation requires exactly one visible CUDA GPU")
        stage1, validation, provenance = audit_inputs()
        manifest = {
            **provenance,
            "models": list(MODEL_NAMES),
            "class_boundaries": [[0, 25], [26, 63], [64, 103], [104, 127]],
            "representatives": [0, 51, 79, 127],
            "transition_tolerance_distinct_onsets": 1,
            "repedal_evaluated": False,
        }
        prior_manifest = output / "manifest.json"
        if prior_manifest.is_file() and json.loads(prior_manifest.read_text()) != manifest:
            raise RuntimeError("existing evaluation manifest/provenance changed")
        atomic_json(prior_manifest, manifest)
        update_status(
            output,
            status="running",
            state="preflight_pass",
            current_model="original_pt",
            current_piece=0,
            pieces_total=19,
            human_performances=71,
            asap_test_access_count=0,
            original_pt_neural_inference_count=0,
            decoder_only_ground_truth_history_access_count=0,
            encoder_decoder_ground_truth_history_access_count=0,
            repedal_metric_execution_count=0,
            report_generated=False,
            error=None,
            traceback=None,
        )
        log("PREFLIGHT_PASS validation_pieces=19 humans=71 test_access=0 original_pt_inference=0")
        for index, row in enumerate(stage1, 1):
            cache_original(output, row)
            update_status(output, current_model="original_pt", current_piece=index)
        tokenizer_config = PinnedTokenizerConfig()
        device = torch.device("cuda:0")
        first_candidate_verified = False
        for model_name in ("encoder_only", "decoder_only", "encoder_decoder"):
            checkpoint_info = provenance["checkpoints"][model_name]
            model, loaded = load_four_class_model(
                model_name,
                CHECKPOINTS[model_name],
                pretrained_checkpoint=PRETRAINED,
                device=device,
            )
            if loaded["epoch"] != checkpoint_info["epoch"]:
                raise RuntimeError("loaded checkpoint epoch changed")
            for index, row in enumerate(stage1, 1):
                paths = prediction_paths(output, model_name, row["piece_id"])
                source = np.load(row["generated_ids_path"], allow_pickle=False)
                source_hash = ids_sha256(source)
                if source_hash != row["generated_token_sha256_int64_le"]:
                    raise RuntimeError("canonical Stage 1 token hash mismatch")
                if valid_prediction_cache(paths, source_hash, checkpoint_info["sha256"]):
                    metadata = json.loads(paths["metadata"].read_text())
                    cache_state = "reused"
                else:
                    classes, candidate_ids, inference = infer_four_class_pedals(
                        model,
                        source,
                        architecture=model_name,
                        device=device,
                    )
                    if inference["ground_truth_pedal_history_access_count"] != 0:
                        raise AssertionError("ground-truth pedal history was accessed")
                    metadata = render_stage2_candidate(
                        output,
                        model_name,
                        row,
                        classes,
                        candidate_ids,
                        inference,
                        checkpoint_info["sha256"],
                        tokenizer_config,
                    )
                    cache_state = "new"
                if not metadata["non_pedal_identity_pass"]:
                    raise AssertionError("Stage 2 non-pedal identity failed")
                if not first_candidate_verified:
                    first_candidate_verified = True
                    update_status(
                        output,
                        status="running",
                        state="first_candidate_sanity_pass",
                        current_model=model_name,
                        current_piece=index,
                        first_validation_piece=row["piece_id"],
                        first_stage2_input_identity_check="PASS",
                        first_candidate_output=metadata["candidate_midi"],
                        initial_pipeline_finite=True,
                    )
                    log(
                        f"FIRST_CANDIDATE_SANITY_PASS model={model_name} piece={row['piece_id']} "
                        f"identity=PASS finite=true output={metadata['candidate_midi']}"
                    )
                update_status(
                    output,
                    status="running",
                    state="inference",
                    current_model=model_name,
                    current_piece=index,
                    candidate_cache_state=cache_state,
                )
                log(f"CANDIDATE_READY model={model_name} piece={index}/19 cache={cache_state}")
            del model
            gc.collect()
            torch.cuda.empty_cache()
        update_status(output, state="alignment", current_model="all")
        results, failures = evaluate_metrics(output, stage1, validation)
        atomic_json(output / "alignment" / "failures.json", {"failures": failures})
        for model_name, metrics in results.items():
            atomic_json(output / "metrics" / f"{model_name}.json", metrics)
        sanity = {
            "asap_test_access_count": 0,
            "all_four_models_piece_count": 19,
            "stage2_non_pedal_identity_pass": True,
            "decoder_only_ground_truth_history_access_count": 0,
            "encoder_decoder_ground_truth_history_access_count": 0,
            "predicted_class_range": [0, 3],
            "distribution_bin_count": 256,
            "transition_tolerance_distinct_onsets": 1,
            "repedal_metric_execution_count": 0,
        }
        atomic_json(output / "sanity_checks.json", sanity)
        report = output / REPORT_NAME
        temporary = report.with_name(f".{report.name}.{os.getpid()}.tmp")
        temporary.write_text(report_text(provenance, results, failures), encoding="utf-8")
        os.replace(temporary, report)
        update_status(
            output,
            status="completed",
            state="completed",
            current_model="all",
            current_piece=19,
            metrics_finite=True,
            alignment_failures=len(failures),
            report_generated=True,
            report_path=str(report),
            completed_at=now(),
        )
        log(f"EVALUATION_COMPLETE report={report}")
        return 0
    except BaseException as error:
        update_status(
            output,
            status="failed",
            state="failed",
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
            asap_test_access_count=0,
            repedal_metric_execution_count=0,
            report_generated=False,
        )
        log(f"EVALUATION_FAILED error={error!r}")
        log(traceback.format_exc())
        return 1
    finally:
        log_handle.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("evaluation is guarded; pass --execute")
    return run(args.output_root)


if __name__ == "__main__":
    raise SystemExit(main())
