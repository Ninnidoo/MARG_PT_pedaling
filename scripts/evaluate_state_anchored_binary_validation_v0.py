#!/usr/bin/env python3
"""Canonical binary validation suite for persisted Run B epoch-9 MIDI."""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from miditoolkit import MidiFile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import evaluate_stage2_4class_repedal_validation_v0 as final_repedal
from scripts import run_stage2_4class_validation_eval_v0 as canonical
from scripts.audit_pedal_event_metric_tolerance_mini import (
    extract_transitions,
    raw_events,
    tick_ms_fn,
)
from src.stage2_binary.canonical_stage1 import (
    assert_strict_non_cc64_equality,
    sha256_file,
    signature_sha256,
)
from src.stage2_encoder_only.dataset import (
    NON_PEDAL_FEATURES,
    PEDAL_TOKEN_OFFSET,
    TOKENS_PER_NOTE,
)
from src.stage2_encoder_only.evaluate_oracle import distribution_similarity
from src.stage2_four_class.validation_evaluator import pooled_transition_counts


RUN_ROOT = ROOT / "analysis/stage2_state_anchored_transition_v1_full_v0"
OUTPUT_ROOT = ROOT / "analysis/stage2_state_anchored_transition_v1_binary_validation_eval_v0"
CANONICAL_EVAL = ROOT / "analysis/stage2_4class_architecture_validation_eval_v0"
RUN_A_SCORE_POSITIONS = ROOT / "analysis/stage2_binary_2slot_D_pre_main_post_full_v1/frozen_pt_score_positions"
STAGE1_MANIFEST = ROOT / "analysis/stage2_binary_canonical_v1/canonical_validation_stage1_manifest.csv"
SPLIT_CSV = ROOT / "analysis/stage2_encoder_only_v0/asap_split.csv"
BEST_CHECKPOINT = RUN_ROOT / "best.pt"
EPOCH_METRICS = RUN_ROOT / "frozen_pt_validation_epochs/epoch_09.json"
EXPECTED_STAGE1_SHA = "f32e91da5d2196edee825236daa577f18709cf7d49e289d8f18ae8ea637b0037"
EXPECTED_BEST_EPOCH = 9
EXPECTED_TRANSITION_F1 = 0.6249756009669524
EXPECTED_EXCLUSION = "856"
SYSTEMS = ("Original PT", "Run B")
PATTERNS = tuple(f"{value:04b}" for value in range(16))
PATTERN_EPSILON = 1e-10


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def ids_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def as_note_tokens(values: np.ndarray) -> np.ndarray:
    tokens = np.asarray(values, dtype=np.int64)
    if tokens.ndim == 1:
        if not tokens.size or tokens.size % TOKENS_PER_NOTE:
            raise ValueError("token array does not contain complete notes")
        tokens = tokens.reshape(-1, TOKENS_PER_NOTE)
    if tokens.ndim != 2 or tokens.shape[1] != TOKENS_PER_NOTE or not len(tokens):
        raise ValueError("token array must have shape [N,8]")
    return tokens


def binary_pedal_values(tokens: np.ndarray) -> np.ndarray:
    values = as_note_tokens(tokens)[:, NON_PEDAL_FEATURES:] - PEDAL_TOKEN_OFFSET
    if not np.all((values >= 0) & (values <= 127)):
        raise ValueError("aligned Pedal1-4 value escaped [0,127]")
    return (values >= 64).astype(np.int64)


def represented_horizon_note_mask(tokens: np.ndarray) -> np.ndarray:
    """Keep complete Pedal1-4 notes strictly before the final distinct onset."""

    notes = as_note_tokens(tokens)
    iois = notes[:, 1] - 261
    if np.any(iois < 0):
        raise ValueError("negative PT IOI token")
    distinct_starts = np.flatnonzero(np.r_[True, iois[1:] > 0])
    if len(distinct_starts) < 2:
        raise ValueError("represented horizon requires at least two distinct onsets")
    final_onset_start = int(distinct_starts[-1])
    mask = np.arange(len(notes), dtype=np.int64) < final_onset_start
    if not mask.any() or mask[final_onset_start:].any():
        raise AssertionError("invalid [t1,tM) aligned-note mask")
    return mask


def binary_confusion(predicted: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    candidate = np.asarray(predicted, dtype=np.int64)
    reference = np.asarray(target, dtype=np.int64)
    valid = np.asarray(mask, dtype=bool)
    if candidate.shape != reference.shape or candidate.shape != valid.shape:
        raise ValueError("binary candidate/target/mask shapes differ")
    if candidate.ndim != 2 or candidate.shape[1] != 4:
        raise ValueError("binary Pedal1-4 arrays must have shape [N,4]")
    if not valid.any() or not np.all((candidate == 0) | (candidate == 1)):
        raise ValueError("invalid binary metric input")
    if not np.all((reference == 0) | (reference == 1)):
        raise ValueError("invalid binary reference")
    return np.bincount(reference[valid] * 2 + candidate[valid], minlength=4).reshape(2, 2)


def classification_metrics(confusion: np.ndarray) -> dict[str, Any]:
    matrix = np.asarray(confusion, dtype=np.int64)
    if matrix.shape != (2, 2) or np.any(matrix < 0) or matrix.sum() <= 0:
        raise ValueError("binary confusion must be a non-empty 2x2 matrix")
    classes: dict[str, Any] = {}
    for index, name in enumerate(("OFF", "ON")):
        tp = int(matrix[index, index])
        predicted = int(matrix[:, index].sum())
        support = int(matrix[index].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        classes[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "predicted_count": predicted,
        }
    total = int(matrix.sum())
    return {
        "accuracy": float(np.trace(matrix) / total),
        "macro_f1": float(np.mean([classes["OFF"]["f1"], classes["ON"]["f1"]])),
        "classes": classes,
        "confusion_matrix": matrix.tolist(),
        "pedal_samples": total,
        "target_distribution": {
            "OFF": float(matrix[0].sum() / total),
            "ON": float(matrix[1].sum() / total),
        },
        "prediction_distribution": {
            "OFF": float(matrix[:, 0].sum() / total),
            "ON": float(matrix[:, 1].sum() / total),
        },
        "threshold": "OFF <64; ON >=64",
        "zero_support_convention": "precision/recall/F1=0",
    }


def pattern_ids(bits: np.ndarray) -> np.ndarray:
    values = np.asarray(bits, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != 4 or not np.all((values == 0) | (values == 1)):
        raise ValueError("binary patterns require [N,4] values in {0,1}")
    return values @ np.asarray([8, 4, 2, 1], dtype=np.int64)


def pattern_metrics(candidate_histogram: np.ndarray, target_histogram: np.ndarray) -> dict[str, Any]:
    candidate = np.asarray(candidate_histogram, dtype=np.int64)
    target = np.asarray(target_histogram, dtype=np.int64)
    if candidate.shape != (16,) or target.shape != (16,) or candidate.sum() <= 0 or target.sum() <= 0:
        raise ValueError("binary pattern histograms must be non-empty 16-bin arrays")
    metrics = distribution_similarity(target, candidate, epsilon=PATTERN_EPSILON)
    candidate_probability = candidate.astype(np.float64) / candidate.sum() + PATTERN_EPSILON
    target_probability = target.astype(np.float64) / target.sum() + PATTERN_EPSILON
    candidate_probability /= candidate_probability.sum()
    target_probability /= target_probability.sum()
    return {
        "js_divergence_base2": float(metrics["js_divergence_base2"]),
        "js_distance_base2_diagnostic": float(metrics["js_distance_base2"]),
        "intersection": float(metrics["histogram_intersection"]),
        "candidate_histogram": candidate.tolist(),
        "target_histogram": target.tolist(),
        "candidate_distribution": candidate_probability.tolist(),
        "target_distribution": target_probability.tolist(),
        "pattern_notes": int(candidate.sum()),
        "bins": 16,
        "epsilon": PATTERN_EPSILON,
        "aggregation": "pooled aligned-note histogram before normalization",
    }


def transition_accumulator() -> dict[str, dict[str, int]]:
    return {
        scope: {key: 0 for key in ("candidate", "reference", "tp", "fp", "fn")}
        for scope in ("pooled", "up", "down")
    }


def add_transition_counts(total: dict[str, dict[str, int]], value: Mapping[str, Any]) -> None:
    for scope in ("pooled", "up", "down"):
        for key in ("candidate", "reference", "tp", "fp", "fn"):
            total[scope][key] += int(value[scope][key])


def finish_transition(total: dict[str, dict[str, int]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for scope in ("pooled", "up", "down"):
        value = total[scope]
        precision = value["tp"] / (value["tp"] + value["fp"]) if value["tp"] + value["fp"] else 0.0
        recall = value["tp"] / (value["tp"] + value["fn"]) if value["tp"] + value["fn"] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result[scope] = {**value, "precision": precision, "recall": recall, "f1": f1}
    result["pooled"]["predicted_reference_ratio"] = (
        result["pooled"]["candidate"] / result["pooled"]["reference"]
        if result["pooled"]["reference"] else 0.0
    )
    result["pooled"]["tolerance_distinct_onsets"] = 1
    return result


def mapped_raw_transitions(
    midi_path: Path,
    score_positions: Sequence[int],
    *,
    initialization_anchor: Mapping[str, Any] | None = None,
    represented_horizon_only: bool,
) -> dict[str, Any]:
    midi = MidiFile(str(midi_path))
    onsets, controls = raw_events(midi)
    if len(onsets) != len(score_positions) or len(onsets) < 2:
        raise RuntimeError("candidate onset/score-position cardinality changed")
    convert = tick_ms_fn(midi)
    transitions = extract_transitions(onsets, [convert(tick) for tick in onsets], controls, convert)
    removed_anchor = 0
    if initialization_anchor and initialization_anchor.get("initialization_anchor_inserted"):
        anchor_tick = int(initialization_anchor["initialization_anchor_tick"])
        anchor_direction = "DOWN" if int(initialization_anchor["initialization_anchor_cc64"]) >= 64 else "UP"
        for index, event in enumerate(transitions):
            if int(event["tick"]) == anchor_tick and event["direction"] == anchor_direction:
                transitions.pop(index)
                removed_anchor = 1
                break
        if removed_anchor != 1:
            raise RuntimeError(f"serialization-only S1 anchor not found in raw MIDI: {midi_path}")
    for event in transitions:
        event["score_position"] = int(score_positions[int(event["onset_index"])])
    unfiltered = list(transitions)
    if represented_horizon_only:
        left_ms, right_ms = float(convert(onsets[0])), float(convert(onsets[-1]))
        transitions = [event for event in transitions if left_ms <= float(event["time_ms"]) < right_ms]
    return {
        "transitions": transitions,
        "all_transitions": unfiltered,
        "onsets": onsets,
        "score_positions": list(map(int, score_positions)),
        "serialization_anchor_removed": removed_anchor,
    }


def transition_mapping_from_events(events: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int], dict[str, int]]:
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for event in events:
        grouped[(str(event["direction"]), int(event["tick"]))].append(event)
    result: dict[tuple[str, int], dict[str, int]] = {}
    for key, values in grouped.items():
        positions = {int(value["score_position"]) for value in values}
        onsets = {int(value["onset_index"]) for value in values}
        if len(positions) != 1 or len(onsets) != 1:
            raise RuntimeError(f"ambiguous transition mapping: {key}")
        result[key] = {"score_position": positions.pop(), "onset_index": onsets.pop()}
    return result


def filtered_human_transitions(tokens_path: Path, midi_path: Path) -> dict[str, Any]:
    payload = json.loads(tokens_path.read_text(encoding="utf-8"))
    midi = MidiFile(str(midi_path))
    onsets, _ = raw_events(midi)
    if len(onsets) < 2:
        raise RuntimeError("human reference has fewer than two distinct onsets")
    convert = tick_ms_fn(midi)
    left_ms, right_ms = float(convert(onsets[0])), float(convert(onsets[-1]))
    return {
        "transitions": [
            event for event in payload["transitions"]
            if left_ms <= float(event["time_ms"]) < right_ms
        ]
    }


def load_score_positions(stage1_row: Mapping[str, str]) -> list[int]:
    path = RUN_A_SCORE_POSITIONS / f"{stage1_row['piece_id']}.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    source = Path(stage1_row["canonical_midi_path"])
    onsets, _ = raw_events(MidiFile(str(source)))
    if value["source_sha256"] != stage1_row["canonical_midi_sha256"] or value["raw_onsets"] != onsets:
        raise RuntimeError(f"frozen score-position provenance changed: {stage1_row['piece_id']}")
    positions = [int(item) for item in value["score_positions"]]
    if len(positions) != len(onsets):
        raise RuntimeError("frozen score-position cardinality changed")
    return positions


def validate_alignment_metadata(path: Path, expected_midi: Path) -> dict[str, Any]:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    actual = Path(metadata["performance_path"])
    if not metadata.get("complete") or not actual.is_file():
        raise RuntimeError(f"alignment source mismatch: {path}")
    expected_hash = sha256_file(expected_midi)
    if metadata["performance_sha256"] != expected_hash or sha256_file(actual) != expected_hash:
        raise RuntimeError(f"alignment source hash mismatch: {path}")
    return metadata


def metric_direction(value: float, baseline: float, *, lower: bool = False, tolerance: float = 1e-12) -> str:
    delta = value - baseline
    if abs(delta) <= tolerance:
        return "maintained"
    improved = delta < 0 if lower else delta > 0
    return "improved" if improved else "degraded"


def evaluate(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    stage1 = read_csv(STAGE1_MANIFEST)
    validation = [row for row in read_csv(SPLIT_CSV) if row["split"] == "validation"]
    if len(stage1) != 19 or len(validation) != 71 or sha256_file(STAGE1_MANIFEST) != EXPECTED_STAGE1_SHA:
        raise RuntimeError("canonical Stage1/validation provenance changed")
    if {row["seed"] for row in stage1} != {"42"} or {row["status"] for row in stage1} != {"frozen_pass"}:
        raise RuntimeError("canonical Stage1 seed/status changed")

    best = torch.load(BEST_CHECKPOINT, map_location="cpu", weights_only=False)
    if int(best["epoch"]) != EXPECTED_BEST_EPOCH or int(best["seed"]) != 42:
        raise RuntimeError("best.pt epoch/seed changed")
    if abs(float(best["frozen_pt_validation"]["transition_f1"]) - EXPECTED_TRANSITION_F1) > 1e-15:
        raise RuntimeError("best.pt primary metric changed")
    del best

    epoch = json.loads(EPOCH_METRICS.read_text(encoding="utf-8"))
    identities = {row["piece_id"]: row for row in epoch["identity_rows"]}
    if int(epoch["epoch"]) != EXPECTED_BEST_EPOCH or len(identities) != 19:
        raise RuntimeError("epoch-9 validation identity provenance changed")

    inventory: dict[str, dict[str, Any]] = {}
    manifest_rows: list[dict[str, Any]] = []
    for row in stage1:
        piece = row["piece_id"]
        source = Path(row["canonical_midi_path"])
        candidate = RUN_ROOT / "frozen_pt_outputs/epoch_09" / piece / "candidate.mid"
        identity = identities[piece]
        if not source.is_file() or sha256_file(source) != row["canonical_midi_sha256"]:
            raise RuntimeError(f"canonical Stage1 source missing/changed: {piece}")
        if not candidate.is_file() or sha256_file(candidate) != identity["candidate_sha256"]:
            raise RuntimeError(f"persisted Run B epoch-9 MIDI missing/changed: {piece}")
        strict = assert_strict_non_cc64_equality(source, candidate)
        if not strict["passed"] or not identity["note_identity_exact"]:
            raise RuntimeError(f"Run B non-pedal identity failed: {piece}")
        inventory[piece] = {
            "stage1": row,
            "source": source,
            "run_b": candidate,
            "identity": identity,
            "strict": strict,
        }
        manifest_rows.append({
            "piece_id": piece,
            "best_epoch": EXPECTED_BEST_EPOCH,
            "source_frozen_pt_midi": str(source),
            "source_sha256": row["canonical_midi_sha256"],
            "run_b_candidate_midi": str(candidate),
            "run_b_candidate_sha256": identity["candidate_sha256"],
            "persistent_artifact_reused": True,
            "new_run_b_inference": False,
            "non_pedal_identity": "PASS",
            "note_count": identity["note_count"],
            "initialization_anchor_inserted": identity["initialization_anchor_inserted"],
        })
    atomic_csv(
        output / "RUN_B_BEST_VALIDATION_MANIFEST.csv",
        manifest_rows,
        tuple(manifest_rows[0]),
    )
    materialization = {
        "status": "PASS",
        "best_epoch": EXPECTED_BEST_EPOCH,
        "best_checkpoint": str(BEST_CHECKPOINT),
        "candidate_midi_source": "persisted frozen_pt_outputs/epoch_09; no rematerialization",
        "persistent_candidates_reused": 19,
        "run_b_inference_count": 0,
        "stage1_inference_regeneration": 0,
        "non_pedal_identity_pass": 19,
        "non_pedal_identity_total": 19,
        "pedal_leakage": 0,
        "asap_test_access": 0,
        "rows": manifest_rows,
    }
    atomic_json(output / "materialization_audit.json", materialization)

    validation_by_piece: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in validation:
        validation_by_piece[row["piece_id"]].append(row)
    human: dict[str, dict[str, Any]] = {}
    failures = []
    for row in validation:
        identifier = row["metadata_index"]
        root = CANONICAL_EVAL / "alignment/human" / identifier
        if not (root / "metadata.json").is_file():
            failures.append(identifier)
            continue
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        midi_path = Path(metadata["performance_path"])
        validate_alignment_metadata(root / "metadata.json", midi_path)
        if row["performance_path"].replace("\\", "/") not in str(midi_path).replace("\\", "/"):
            raise RuntimeError(f"human alignment split mapping changed: {identifier}")
        human[identifier] = {
            "row": row,
            "midi": midi_path,
            "tokens": np.load(root / "aligned_tokens_int64.npy", allow_pickle=False),
            "transitions_path": root / "transitions.json",
        }
    if failures != [EXPECTED_EXCLUSION] or len(human) != 70:
        raise RuntimeError(f"canonical aligned universe changed: failures={failures} humans={len(human)}")

    candidate_trajectories: dict[str, dict[str, dict[str, Any]]] = {system: {} for system in SYSTEMS}
    for piece, item in inventory.items():
        positions = load_score_positions(item["stage1"])
        candidate_trajectories["Original PT"][piece] = mapped_raw_transitions(
            item["source"], positions, represented_horizon_only=True,
        )
        candidate_trajectories["Run B"][piece] = mapped_raw_transitions(
            item["run_b"], positions,
            initialization_anchor=item["identity"],
            represented_horizon_only=True,
        )
    human_transition = {
        identifier: filtered_human_transitions(item["transitions_path"], item["midi"])
        for identifier, item in human.items()
    }
    transition_results: dict[str, Any] = {}
    for system in SYSTEMS:
        total = transition_accumulator()
        pairs = 0
        for piece, rows in validation_by_piece.items():
            for row in rows:
                identifier = row["metadata_index"]
                if identifier not in human:
                    continue
                counts = pooled_transition_counts(
                    {"transitions": candidate_trajectories[system][piece]["transitions"]},
                    human_transition[identifier],
                )
                add_transition_counts(total, counts)
                pairs += 1
        result = finish_transition(total)
        result.update({
            "system": system,
            "comparison_pairs": pairs,
            "modeled_horizon": "[t1,tM)",
            "source": "final materialized raw MIDI CC64",
            "threshold": "OFF <64; ON >=64",
            "serialization_anchor_metric_events": 0,
        })
        transition_results[system] = result
    run_b_transition = transition_results["Run B"]["pooled"]
    if abs(float(run_b_transition["f1"]) - EXPECTED_TRANSITION_F1) > 1e-12:
        raise RuntimeError(
            f"materialized Run B Transition F1 did not reproduce training-time metric: {run_b_transition['f1']}"
        )
    if run_b_transition["candidate"] != int(epoch["predicted_transitions"]) or run_b_transition["reference"] != int(epoch["reference_transitions"]):
        raise RuntimeError("materialized Run B Transition denominator changed")
    atomic_json(output / "transition_metrics.json", {
        "status": "PASS", "systems": transition_results,
        "training_time_expected_f1": EXPECTED_TRANSITION_F1,
        "training_time_reproduced": True, "asap_test_access": 0,
    })

    aligned_candidates: dict[str, dict[str, np.ndarray]] = {system: {} for system in SYSTEMS}
    horizon_masks: dict[str, np.ndarray] = {}
    alignment_cache_reused = 0
    alignment_cache_created = 0
    for piece, item in inventory.items():
        original_root = CANONICAL_EVAL / "alignment/original_pt" / piece
        metadata = validate_alignment_metadata(original_root / "metadata.json", item["source"])
        original_tokens = np.load(original_root / "aligned_tokens_int64.npy", allow_pickle=False)
        if ids_sha256(original_tokens) != metadata["aligned_tokens_sha256"]:
            raise RuntimeError(f"Original PT aligned token cache hash changed: {piece}")
        run_b_aligned = canonical.align_and_cache(
            output,
            kind="run_b",
            identifier=piece,
            score_path=Path(item["stage1"]["selected_score_absolute_path"]),
            performance_path=item["run_b"],
        )
        alignment_cache_reused += int(run_b_aligned["cache_reused"])
        alignment_cache_created += int(not run_b_aligned["cache_reused"])
        run_b_tokens = as_note_tokens(run_b_aligned["tokens"])
        original_tokens = as_note_tokens(original_tokens)
        if original_tokens.shape != run_b_tokens.shape or not np.array_equal(
            original_tokens[:, :NON_PEDAL_FEATURES], run_b_tokens[:, :NON_PEDAL_FEATURES]
        ):
            raise RuntimeError(f"Run B canonical aligned non-pedal grid changed: {piece}")
        aligned_candidates["Original PT"][piece] = binary_pedal_values(original_tokens)
        aligned_candidates["Run B"][piece] = binary_pedal_values(run_b_tokens)
        original_mask = represented_horizon_note_mask(original_tokens)
        run_b_mask = represented_horizon_note_mask(run_b_tokens)
        if not np.array_equal(original_mask, run_b_mask):
            raise RuntimeError(f"Original PT/Run B represented-horizon mask differs: {piece}")
        horizon_masks[piece] = run_b_mask

    state_totals = {system: np.zeros((2, 2), dtype=np.int64) for system in SYSTEMS}
    candidate_histograms = {system: np.zeros(16, dtype=np.int64) for system in SYSTEMS}
    target_histograms = {system: np.zeros(16, dtype=np.int64) for system in SYSTEMS}
    aligned_notes_before_mask = {system: 0 for system in SYSTEMS}
    represented_notes = {system: 0 for system in SYSTEMS}
    excluded_final_onset_notes = {system: 0 for system in SYSTEMS}
    pair_count = {system: 0 for system in SYSTEMS}
    for piece, rows in validation_by_piece.items():
        note_mask = horizon_masks[piece]
        sample_mask = np.repeat(note_mask[:, None], 4, axis=1)
        for row in rows:
            identifier = row["metadata_index"]
            if identifier not in human:
                continue
            target = binary_pedal_values(human[identifier]["tokens"])
            for system in SYSTEMS:
                candidate = aligned_candidates[system][piece]
                if candidate.shape != target.shape or len(note_mask) != len(candidate):
                    raise RuntimeError(f"candidate/human aligned shape changed: {system} {identifier}")
                state_totals[system] += binary_confusion(candidate, target, sample_mask)
                candidate_histograms[system] += np.bincount(
                    pattern_ids(candidate[note_mask]), minlength=16
                )
                target_histograms[system] += np.bincount(
                    pattern_ids(target[note_mask]), minlength=16
                )
                aligned_notes_before_mask[system] += len(candidate)
                represented_notes[system] += int(note_mask.sum())
                excluded_final_onset_notes[system] += int((~note_mask).sum())
                pair_count[system] += 1

    state_results = {}
    pattern_results = {}
    distribution_rows = []
    for system in SYSTEMS:
        state = classification_metrics(state_totals[system])
        state.update({
            "system": system,
            "comparison_pairs": pair_count[system],
            "aligned_notes_before_horizon_mask": aligned_notes_before_mask[system],
            "represented_pattern_notes": represented_notes[system],
            "excluded_final_onset_notes": excluded_final_onset_notes[system],
            "horizon": "[t1,tM)",
            "inclusion_mask": "all four PT Pedal1-4 sample positions belong to notes strictly before final distinct onset",
        })
        state_results[system] = state
        patterns = pattern_metrics(candidate_histograms[system], target_histograms[system])
        patterns.update({"system": system, "comparison_pairs": pair_count[system], "horizon": "[t1,tM)"})
        pattern_results[system] = patterns
        for index, pattern in enumerate(PATTERNS):
            distribution_rows.append({
                "system": system,
                "pattern_id": index,
                "pattern": " ".join("ON" if bit == "1" else "OFF" for bit in pattern),
                "candidate_count": int(candidate_histograms[system][index]),
                "target_count": int(target_histograms[system][index]),
                "candidate_probability": patterns["candidate_distribution"][index],
                "target_probability": patterns["target_distribution"][index],
            })
    atomic_json(output / "binary_state_metrics.json", {
        "status": "PASS", "systems": state_results,
        "representation": "final materialized MIDI Pedal1-4 thresholded at 64",
        "asap_test_access": 0,
    })
    atomic_json(output / "binary_16pattern_metrics.json", {
        "status": "PASS", "systems": pattern_results,
        "all_16_bins_retained": True, "asap_test_access": 0,
    })
    atomic_csv(
        output / "binary_16pattern_distribution.csv",
        distribution_rows,
        ("system", "pattern_id", "pattern", "candidate_count", "target_count", "candidate_probability", "target_probability"),
    )

    reference_repedals: dict[str, list[dict[str, Any]]] = {}
    for identifier, item in human.items():
        mapping = final_repedal.transition_mapping(item["transitions_path"], item["midi"])
        reference_repedals[identifier] = final_repedal.extract_repedals(
            item["midi"], mapping=mapping, candidate="REFERENCE",
            piece_id=item["row"]["piece_id"], performance_id=item["row"]["performance_path"], role="reference",
        )
    candidate_repedals: dict[str, dict[str, list[dict[str, Any]]]] = {system: {} for system in SYSTEMS}
    for piece, item in inventory.items():
        for system, midi_key in (("Original PT", "source"), ("Run B", "run_b")):
            trajectory = candidate_trajectories[system][piece]
            mapping = transition_mapping_from_events(trajectory["all_transitions"])
            candidate_repedals[system][piece] = final_repedal.extract_repedals(
                item[midi_key], mapping=mapping, candidate=system,
                piece_id=piece, performance_id=piece, role="candidate",
            )
    repedal_results = {}
    for system in SYSTEMS:
        predicted = reference = tp = pairs = 0
        unique_candidate = sum(len(value) for value in candidate_repedals[system].values())
        for piece, rows in validation_by_piece.items():
            for row in rows:
                identifier = row["metadata_index"]
                if identifier not in human:
                    continue
                candidate = candidate_repedals[system][piece]
                target = reference_repedals[identifier]
                matched = final_repedal.match_repedals(candidate, target)
                predicted += len(candidate)
                reference += len(target)
                tp += len(matched)
                pairs += 1
        value = final_repedal.metric_row(predicted, reference, tp)
        value.update({
            "system": system,
            "comparison_pairs": pairs,
            "unique_candidate_repedal_count": unique_candidate,
            "predicted_reference_ratio": predicted / reference if reference else 0.0,
            "evaluation_horizon": "full saved MIDI, matching frozen FINAL Repedaling evaluator",
            "definition": "strict high-low-high, excursions >=64, 0<OFF duration<=300ms, both boundaries ±1 onset",
        })
        repedal_results[system] = value
    atomic_json(output / "repedal_metrics.json", {
        "status": "PASS", "systems": repedal_results,
        "final_evaluator": str(ROOT / "scripts/evaluate_stage2_4class_repedal_validation_v0.py"),
        "asap_test_access": 0,
    })

    comparison_rows = []
    for system in SYSTEMS:
        comparison_rows.append({
            "system": system,
            "binary_accuracy": state_results[system]["accuracy"],
            "binary_macro_f1": state_results[system]["macro_f1"],
            "transition_precision": transition_results[system]["pooled"]["precision"],
            "transition_recall": transition_results[system]["pooled"]["recall"],
            "transition_f1": transition_results[system]["pooled"]["f1"],
            "repedal_precision": repedal_results[system]["precision"],
            "repedal_recall": repedal_results[system]["recall"],
            "repedal_f1": repedal_results[system]["f1"],
            "pattern_js_divergence_base2": pattern_results[system]["js_divergence_base2"],
            "pattern_intersection": pattern_results[system]["intersection"],
        })
    atomic_csv(output / "comparison_summary.csv", comparison_rows, tuple(comparison_rows[0]))

    original = comparison_rows[0]
    run_b = comparison_rows[1]
    comparisons = {
        "binary_accuracy": metric_direction(run_b["binary_accuracy"], original["binary_accuracy"]),
        "binary_macro_f1": metric_direction(run_b["binary_macro_f1"], original["binary_macro_f1"]),
        "transition_f1": metric_direction(run_b["transition_f1"], original["transition_f1"]),
        "repedal_f1": metric_direction(run_b["repedal_f1"], original["repedal_f1"]),
        "pattern_js_divergence_base2": metric_direction(
            run_b["pattern_js_divergence_base2"], original["pattern_js_divergence_base2"], lower=True
        ),
        "pattern_intersection": metric_direction(run_b["pattern_intersection"], original["pattern_intersection"]),
    }
    provenance = {
        "status": "PASS",
        "best_checkpoint": str(BEST_CHECKPOINT),
        "best_checkpoint_sha256": sha256_file(BEST_CHECKPOINT),
        "best_epoch": EXPECTED_BEST_EPOCH,
        "best_training_transition_f1": EXPECTED_TRANSITION_F1,
        "stage1_manifest": str(STAGE1_MANIFEST),
        "stage1_manifest_sha256": sha256_file(STAGE1_MANIFEST),
        "stage1_pieces": 19,
        "structural_references": 71,
        "canonical_aligned_references": 70,
        "excluded_metadata": EXPECTED_EXCLUSION,
        "seed": 42,
        "stage1_regeneration": 0,
        "run_b_inference_count": 0,
        "run_b_candidate_alignment_cache_created": alignment_cache_created,
        "run_b_candidate_alignment_cache_reused": alignment_cache_reused,
        "canonical_alignment_evaluator": str(ROOT / "scripts/run_stage2_4class_validation_eval_v0.py"),
        "canonical_alignment_evaluator_sha256": sha256_file(ROOT / "scripts/run_stage2_4class_validation_eval_v0.py"),
        "final_repedal_evaluator": str(ROOT / "scripts/evaluate_stage2_4class_repedal_validation_v0.py"),
        "final_repedal_evaluator_sha256": sha256_file(ROOT / "scripts/evaluate_stage2_4class_repedal_validation_v0.py"),
        "transition_evaluator": str(ROOT / "src/stage2_four_class/validation_evaluator.py"),
        "transition_tolerance_distinct_onsets": 1,
        "binary_threshold": 64,
        "pattern_epsilon": PATTERN_EPSILON,
        "asap_test_access": 0,
    }
    atomic_json(output / "provenance.json", provenance)

    tradeoff = (
        "Run B improves Transition F1 while lowering sampled-state Accuracy/Macro-F1; "
        "Repedal F1 and both distribution metrics improve, so the observed sacrifice is state fidelity rather than distribution or Repedaling."
        if comparisons["transition_f1"] == "improved"
        and comparisons["binary_accuracy"] == "degraded"
        and comparisons["binary_macro_f1"] == "degraded"
        and comparisons["repedal_f1"] == "improved"
        and comparisons["pattern_js_divergence_base2"] == "improved"
        and comparisons["pattern_intersection"] == "improved"
        else "See the per-metric comparison labels; no single uniform trade-off description applies."
    )
    report = f"""# Run B Binary Validation Evaluation

**Verdict: PASS**

## Provenance and integrity

- Run B candidate: persisted epoch-9 MIDI artifacts reused; best.pt inference rerun = 0.
- best.pt: epoch 9; Frozen Transition F1 `{EXPECTED_TRANSITION_F1:.12f}`.
- Frozen Stage1: `{STAGE1_MANIFEST}`; SHA `{EXPECTED_STAGE1_SHA}`; seed 42.
- Universe: 19 pieces / 71 structural references / 70 canonical aligned references; metadata 856 excluded.
- Non-pedal identity: 19/19 PASS; pedal leakage: 0; Stage1 regeneration: 0; ASAP test access: 0.

## Headline comparison

| System | Binary Acc ↑ | Binary Macro F1 ↑ | Transition F1 ↑ | Repedal F1 ↑ | 16-pattern JS ↓ | Intersection ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Original PT | {original['binary_accuracy']:.6f} | {original['binary_macro_f1']:.6f} | {original['transition_f1']:.6f} | {original['repedal_f1']:.6f} | {original['pattern_js_divergence_base2']:.6f} | {original['pattern_intersection']:.6f} |
| Run B | {run_b['binary_accuracy']:.6f} | {run_b['binary_macro_f1']:.6f} | {run_b['transition_f1']:.6f} | {run_b['repedal_f1']:.6f} | {run_b['pattern_js_divergence_base2']:.6f} | {run_b['pattern_intersection']:.6f} |

## Transition and Repedal

| System | Transition P | Transition R | Transition F1 | Pred/Ref | Repedal P | Repedal R | Repedal F1 | Repedal Pred/Ref |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original PT | {original['transition_precision']:.6f} | {original['transition_recall']:.6f} | {original['transition_f1']:.6f} | {transition_results['Original PT']['pooled']['predicted_reference_ratio']:.6f} | {original['repedal_precision']:.6f} | {original['repedal_recall']:.6f} | {original['repedal_f1']:.6f} | {repedal_results['Original PT']['predicted_reference_ratio']:.6f} |
| Run B | {run_b['transition_precision']:.6f} | {run_b['transition_recall']:.6f} | {run_b['transition_f1']:.6f} | {transition_results['Run B']['pooled']['predicted_reference_ratio']:.6f} | {run_b['repedal_precision']:.6f} | {run_b['repedal_recall']:.6f} | {run_b['repedal_f1']:.6f} | {repedal_results['Run B']['predicted_reference_ratio']:.6f} |

Transition denominator: 70 candidate-human pairs; Original PT candidate/reference events {transition_results['Original PT']['pooled']['candidate']} / {transition_results['Original PT']['pooled']['reference']}, Run B {transition_results['Run B']['pooled']['candidate']} / {transition_results['Run B']['pooled']['reference']}. Repedal denominator: 70 pairs; Original PT pooled candidate/reference episodes {repedal_results['Original PT']['candidate_repedal_count']} / {repedal_results['Original PT']['reference_repedal_count']}, Run B {repedal_results['Run B']['candidate_repedal_count']} / {repedal_results['Run B']['reference_repedal_count']}. Repedal pooled candidate counts repeat each piece candidate for each paired human reference, matching the FINAL evaluator; unique 19-piece candidate episodes are {repedal_results['Original PT']['unique_candidate_repedal_count']} / {repedal_results['Run B']['unique_candidate_repedal_count']}.

Run B materialized-MIDI Transition F1 reproduces training time: **YES** (`{run_b['transition_f1']:.12f}`). UP and DOWN diagnostics are in `transition_metrics.json`.

## Binary state and 16-pattern semantics

Pedal1–4 are sampled with the existing canonical PT score-alignment logic and thresholded as OFF `<64`, ON `>=64`. Samples from the final distinct onset onward are excluded, so Accuracy/F1 use `{state_results['Run B']['pedal_samples']}` pooled Pedal slots on `[t1,tM)`. The 16-pattern histogram uses `{pattern_results['Run B']['pattern_notes']}` complete aligned notes, pools counts globally before normalization, and retains all 16 bins. JS is base-2 divergence (not distance); epsilon `{PATTERN_EPSILON}` and zero handling are unchanged from the prior 256-pattern implementation.

Repedal intentionally retains the FINAL evaluator's full-saved-MIDI event universe; its denominator is therefore reported separately rather than forced onto the sampled-state or Transition horizon.

Run B OFF/ON F1: `{state_results['Run B']['classes']['OFF']['f1']:.6f}` / `{state_results['Run B']['classes']['ON']['f1']:.6f}`. Target OFF/ON ratio: `{state_results['Run B']['target_distribution']['OFF']:.6f}` / `{state_results['Run B']['target_distribution']['ON']:.6f}`. Predicted OFF/ON ratio: `{state_results['Run B']['prediction_distribution']['OFF']:.6f}` / `{state_results['Run B']['prediction_distribution']['ON']:.6f}`.

## Answers

1. Epoch-9 MIDI: existing persistent artifacts reused; best.pt generation rerun = 0.
2. Non-pedal identity: 19/19 PASS.
3. Run B Binary Pedal1–4 Accuracy: `{run_b['binary_accuracy']:.9f}`.
4. Run B Binary Macro F1: `{run_b['binary_macro_f1']:.9f}`.
5. Run B Transition P/R/F1: `{run_b['transition_precision']:.9f}` / `{run_b['transition_recall']:.9f}` / `{run_b['transition_f1']:.9f}`; training-time value reproduced: YES.
6. Run B Repedal P/R/F1: `{run_b['repedal_precision']:.9f}` / `{run_b['repedal_recall']:.9f}` / `{run_b['repedal_f1']:.9f}`.
7. Run B 16-pattern JS / Intersection: `{run_b['pattern_js_divergence_base2']:.9f}` / `{run_b['pattern_intersection']:.9f}`.
8. Run B vs Original PT: `{json.dumps(comparisons, sort_keys=True)}`.
9. Trade-off: {tradeoff}
10. ASAP test access: 0.

Bass Connectivity and Harmonic Muddiness were not evaluated.
"""
    report_path = output / "RUN_B_BINARY_VALIDATION_EVALUATION_REPORT.md"
    temporary = report_path.with_name(f".{report_path.name}.{os.getpid()}.tmp")
    temporary.write_text(report, encoding="utf-8")
    os.replace(temporary, report_path)
    return {
        "status": "PASS",
        "run_b": run_b,
        "original_pt": original,
        "comparisons": comparisons,
        "materialized_transition_reproduced": True,
        "asap_test_access": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("evaluation requires --execute")
    result = evaluate(args.output)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
