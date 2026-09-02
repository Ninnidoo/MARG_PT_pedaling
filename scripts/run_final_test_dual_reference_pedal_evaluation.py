#!/usr/bin/env python3
"""CPU-only final-test pedal evaluation over two independent Human universes.

This wrapper reads saved MIDI only.  It imports the frozen canonical alignment,
4-class/pattern/Transition metrics, and the latest deduplicated Repedal matcher.
It never loads a checkpoint and never performs model inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_stage2_4class_repedal_validation_v0 import (
    MAX_OFF_DURATION_MS,
    MIN_EXCURSION,
    ONSET_TOLERANCE,
    THRESHOLD,
    extract_repedals,
    match_repedals,
    maximum_cardinality_match_count,
    metric_row,
    transition_mapping,
)
from scripts.run_stage2_4class_validation_eval_v0 import (
    align_and_cache,
    alignment_paths,
    finalize_transition,
    sum_transition,
)
from src.stage2_four_class.validation_evaluator import (
    PATTERN_COUNT,
    TRANSITION_TOLERANCE,
    canonical_classes_from_tokens,
    classification_metrics,
    confusion_from_pairs,
    pattern_ids,
    pattern_metrics,
    pooled_transition_counts,
)


OUTPUT_ROOT = ROOT / "analysis/final_test_dual_reference_pedal_evaluation_v0"
AB_MANIFEST = ROOT / "analysis/final_test_run_ab_inference_v0/manifest.csv"
AB_CONFIG = ROOT / "analysis/final_test_run_ab_inference_v0/config.json"
TRIO_MANIFEST = ROOT / "analysis/final_test_encoder_only_trio_inference_v0/manifest.csv"
TRIO_CONFIG = ROOT / "analysis/final_test_encoder_only_trio_inference_v0/config.json"
TEST_MANIFEST = ROOT / "analysis/stage2_binary_canonical_v1/test_eval_v0/canonical_test_manifest.csv"
SPLIT_CSV = ROOT / "analysis/stage2_encoder_only_v0/asap_split.csv"
ASAP_ROOT = Path("/workspace/public/ASAP/asap-dataset-v1.1")
PT_HUMAN_ROOT = ROOT / "third_party/PianistTransformer/data/midis/testset/human"

EXPECTED_PIECES = 23
EXPECTED_SET_A_HUMANS = 104
EXPECTED_SET_B_HUMANS = 166
SYSTEM_ORDER = (
    "Human–Human",
    "Original PT",
    "RUN A",
    "RUN B",
    "Weighted CE",
    "CE + NTL-WAS",
    "Huber + Auxiliary CE",
)
CANDIDATE_SYSTEMS = SYSTEM_ORDER[1:]
SYSTEM_KEYS = {
    "Original PT": "original_pt",
    "RUN A": "run_a",
    "RUN B": "run_b",
    "Weighted CE": "weighted_ce",
    "CE + NTL-WAS": "ntl_was",
    "Huber + Auxiliary CE": "huber_aux_ce",
}
REFERENCE_SETS = {
    "setA": {"label": "Set A: ASAP canonical 104", "slug": "setA_asap104"},
    "setB": {"label": "Set B: PT Human 166", "slug": "setB_pt166"},
}
CLASS_NAMES = ("ZERO", "LOW", "HALF", "FULL")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def project_path(value: str | Path) -> Path:
    text = str(value)
    if text == "/workspace/project" or text.startswith("/workspace/project/"):
        return ROOT / Path(text).relative_to("/workspace/project")
    return Path(text)


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
        return None
    return value


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_text(
        path,
        json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(fields or (list(rows[0]) if rows else []))
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: ""
                    if value is None or (isinstance(value, float) and not math.isfinite(value))
                    else value
                    for key, value in row.items()
                }
            )
    os.replace(temporary, path)


def update_status(output: Path, **updates: Any) -> dict[str, Any]:
    path = output / "run_status.json"
    current = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    current.update(updates, last_update=now())
    write_json(path, current)
    return current


def note_signature(path: Path) -> tuple[tuple[int, int, int, int, int, int], ...]:
    from miditoolkit import MidiFile

    midi = MidiFile(str(path))
    rows = []
    for instrument_index, instrument in enumerate(midi.instruments):
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            rows.append(
                (
                    instrument_index,
                    int(instrument.program),
                    int(note.pitch),
                    int(note.start),
                    int(note.end),
                    int(note.velocity),
                )
            )
    return tuple(sorted(rows))


def candidate_columns() -> dict[str, tuple[str, str, str]]:
    return {
        "Original PT": (
            "canonical_stage1_midi_path",
            "canonical_stage1_midi_sha256",
            "ab",
        ),
        "RUN A": ("run_a_output_path", "run_a_output_sha256", "ab"),
        "RUN B": ("run_b_output_path", "run_b_output_sha256", "ab"),
        "Weighted CE": (
            "weighted_ce_output_path",
            "weighted_ce_output_sha256",
            "trio",
        ),
        "CE + NTL-WAS": (
            "ntl_was_output_path",
            "ntl_was_output_sha256",
            "trio",
        ),
        "Huber + Auxiliary CE": (
            "huber_aux_ce_output_path",
            "huber_aux_ce_output_sha256",
            "trio",
        ),
    }


def discover_candidates() -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, Path]],
    list[dict[str, str]],
    dict[str, Any],
]:
    ab = read_csv(AB_MANIFEST)
    trio = read_csv(TRIO_MANIFEST)
    pieces = read_csv(TEST_MANIFEST)
    ab_config = json.loads(AB_CONFIG.read_text(encoding="utf-8"))
    trio_config = json.loads(TRIO_CONFIG.read_text(encoding="utf-8"))
    if len(ab) != EXPECTED_PIECES or len(trio) != EXPECTED_PIECES or len(pieces) != EXPECTED_PIECES:
        raise RuntimeError("candidate/test manifests are not all 23 rows")
    if [int(row["num"]) for row in pieces] != list(range(EXPECTED_PIECES)):
        raise RuntimeError("canonical test manifest is not ordered nums 0..22")
    ab_by_num = {int(row["num"]): row for row in ab}
    trio_by_num = {int(row["num"]): row for row in trio}
    if set(ab_by_num) != set(range(EXPECTED_PIECES)) or set(trio_by_num) != set(range(EXPECTED_PIECES)):
        raise RuntimeError("candidate manifest numbering differs from 0..22")
    if not (
        ab_config["run_a"]["checkpoint_path"].endswith(
            "stage2_binary_2slot_D_pre_main_post_full_v1/best.pt"
        )
        and ab_config["run_a"]["modeled_regions"] == ["PRE", "MAIN", "POST"]
        and ab_config["run_b"]["checkpoint_path"].endswith(
            "stage2_state_anchored_transition_v1_full_v0/best.pt"
        )
        and "constrained 2-state Viterbi" in ab_config["run_b"]["native_frozen_semantics"]
    ):
        raise RuntimeError("RUN A/RUN B final identity mismatch")
    if float(trio_config["ntl_was"]["ntl_was_lambda"]) != 0.3:
        raise RuntimeError("NTL-WAS lambda is not 0.3")
    if trio_config["huber_aux_ce"]["inference_head"] != "regression-only":
        raise RuntimeError("Huber+Aux final MIDI was not regression-only")

    paths: dict[str, dict[str, Path]] = {system: {} for system in CANDIDATE_SYSTEMS}
    inventory: list[dict[str, Any]] = []
    identity_counts: dict[str, Counter[str]] = {
        "RUN A": Counter(row["run_a_nonpedal_identity"] for row in ab),
        "RUN B": Counter(row["run_b_nonpedal_identity"] for row in ab),
        "Weighted CE": Counter(row["weighted_ce_nonpedal_identity"] for row in trio),
        "CE + NTL-WAS": Counter(row["ntl_was_nonpedal_identity"] for row in trio),
        "Huber + Auxiliary CE": Counter(row["huber_aux_ce_nonpedal_identity"] for row in trio),
    }
    expected_run_a = {"PASS_STRICT": 13, "PASS_NATIVE_FROZEN_EOT_EXTENSION_ONLY": 10}
    if dict(identity_counts["RUN A"]) != expected_run_a:
        raise RuntimeError(f"RUN A identity accounting changed: {identity_counts['RUN A']}")
    for system in CANDIDATE_SYSTEMS[2:]:
        if dict(identity_counts[system]) != {"PASS_STRICT": 23}:
            raise RuntimeError(f"strict non-pedal identity accounting changed: {system}")

    columns = candidate_columns()
    for piece in pieces:
        num = int(piece["num"])
        if ab_by_num[num]["piece_id"] != piece["piece_id"] or trio_by_num[num]["piece_id"] != piece["piece_id"]:
            raise RuntimeError(f"candidate piece identity mismatch at num={num}")
        if (
            ab_by_num[num]["canonical_stage1_midi_sha256"]
            != trio_by_num[num]["canonical_stage1_midi_sha256"]
        ):
            raise RuntimeError(f"canonical Original PT hash differs across manifests: num={num}")
        original_path = project_path(ab_by_num[num]["canonical_stage1_midi_path"])
        original_notes = note_signature(original_path)
        for system, (path_column, hash_column, source) in columns.items():
            row = ab_by_num[num] if source == "ab" else trio_by_num[num]
            path = project_path(row[path_column])
            if not path.is_file():
                raise FileNotFoundError(path)
            actual_hash = sha256_file(path)
            if actual_hash != row[hash_column]:
                raise RuntimeError(f"candidate hash mismatch: {system} num={num}")
            if system != "Original PT" and note_signature(path) != original_notes:
                raise RuntimeError(f"note-level non-pedal identity mismatch: {system} num={num}")
            paths[system][piece["piece_id"]] = path
            inventory.append(
                {
                    "num": num,
                    "piece_id": piece["piece_id"],
                    "composer": piece["composer"],
                    "title": piece["title"],
                    "system": system,
                    "midi_path": str(path),
                    "sha256": actual_hash,
                    "note_level_nonpedal_identity": "CANONICAL_SOURCE"
                    if system == "Original PT"
                    else "PASS",
                    "prior_identity_status": "CANONICAL_SOURCE"
                    if system == "Original PT"
                    else (
                        ab_by_num[num]["run_a_nonpedal_identity"]
                        if system == "RUN A"
                        else ab_by_num[num]["run_b_nonpedal_identity"]
                        if system == "RUN B"
                        else trio_by_num[num][
                            {
                                "Weighted CE": "weighted_ce_nonpedal_identity",
                                "CE + NTL-WAS": "ntl_was_nonpedal_identity",
                                "Huber + Auxiliary CE": "huber_aux_ce_nonpedal_identity",
                            }[system]
                        ]
                    ),
                }
            )
    return inventory, paths, pieces, {
        "run_ab_config": ab_config,
        "trio_config": trio_config,
        "identity_counts": {key: dict(value) for key, value in identity_counts.items()},
    }


def discover_humans(
    pieces: Sequence[Mapping[str, str]],
) -> dict[str, list[dict[str, Any]]]:
    piece_by_id = {row["piece_id"]: row for row in pieces}
    piece_by_num = {int(row["num"]): row for row in pieces}
    set_a = []
    for row in read_csv(SPLIT_CSV):
        if row["split"] != "test":
            continue
        if row["piece_id"] not in piece_by_id:
            raise RuntimeError(f"Set A piece absent from candidate universe: {row['piece_id']}")
        path = ASAP_ROOT / row["performance_path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        piece = piece_by_id[row["piece_id"]]
        set_a.append(
            {
                "reference_set": "setA",
                "identifier": row["metadata_index"],
                "performance_id": row["performance_path"],
                "midi_path": str(path),
                "sha256": sha256_file(path),
                "num": int(piece["num"]),
                "piece_id": row["piece_id"],
                "composer": piece["composer"],
                "title": piece["title"],
            }
        )
    set_b = []
    pattern = re.compile(r"^(\d+)-(.+)\.mid$", re.IGNORECASE)
    for path in sorted(PT_HUMAN_ROOT.glob("*.mid"), key=lambda item: item.name):
        match = pattern.match(path.name)
        if not match:
            raise RuntimeError(f"unrecognized PT Human filename: {path.name}")
        num = int(match.group(1))
        if num not in piece_by_num:
            raise RuntimeError(f"PT Human prefix outside candidate numbering: {path.name}")
        piece = piece_by_num[num]
        set_b.append(
            {
                "reference_set": "setB",
                "identifier": path.stem,
                "performance_id": path.name,
                "midi_path": str(path),
                "sha256": sha256_file(path),
                "num": num,
                "piece_id": piece["piece_id"],
                "composer": piece["composer"],
                "title": piece["title"],
            }
        )
    if len(set_a) != EXPECTED_SET_A_HUMANS or len({row["piece_id"] for row in set_a}) != 23:
        raise RuntimeError("Set A is not 23 pieces / 104 Human performances")
    if len(set_b) != EXPECTED_SET_B_HUMANS or len({row["piece_id"] for row in set_b}) != 23:
        raise RuntimeError("Set B is not 23 pieces / 166 Human MIDI")
    if {row["piece_id"] for row in set_a} != set(piece_by_id):
        raise RuntimeError("Set A piece correspondence is not 23/23")
    if {row["piece_id"] for row in set_b} != set(piece_by_id):
        raise RuntimeError("Set B piece correspondence is not 23/23")
    if len({row["performance_id"] for row in set_a}) != len(set_a):
        raise RuntimeError("Set A performance identifiers are not unique")
    if len({row["performance_id"] for row in set_b}) != len(set_b):
        raise RuntimeError("Set B performance identifiers are not unique")
    return {"setA": set_a, "setB": set_b}


def freeze_representatives(
    output: Path, reference_set: str, humans: Sequence[Mapping[str, Any]]
) -> dict[str, str]:
    by_piece: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in humans:
        by_piece[str(row["piece_id"])].append(row)
    representatives: dict[str, str] = {}
    rows = []
    for piece_id in sorted(by_piece):
        ordered = sorted(
            by_piece[piece_id], key=lambda row: (str(row["performance_id"]), str(row["midi_path"]))
        )
        representative = str(ordered[0]["identifier"])
        representatives[piece_id] = representative
        for order, row in enumerate(ordered):
            rows.append(
                {
                    "reference_set": reference_set,
                    "num": row["num"],
                    "piece_id": piece_id,
                    "composer": row["composer"],
                    "title": row["title"],
                    "lexicographic_order": order,
                    "identifier": row["identifier"],
                    "performance_id": row["performance_id"],
                    "midi_path": row["midi_path"],
                    "sha256": row["sha256"],
                    "is_representative_h0": order == 0,
                    "human_human_role": "H0" if order == 0 else "REFERENCE",
                }
            )
    path = output / f"human_representatives_{reference_set}.csv"
    if path.is_file():
        existing = read_csv(path)
        normalized = [
            {key: str(value) for key, value in row.items()}
            for row in rows
        ]
        if existing != normalized:
            raise RuntimeError(f"frozen Human representative mapping changed: {path}")
    else:
        write_csv(path, rows)
    return representatives


def cpu_preflight() -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be empty or -1")
    import torch

    tensor = torch.zeros(1, device="cpu")
    result = {
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": str(tensor.device),
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_device_count": int(torch.cuda.device_count()),
        "torch_cuda_initialized": bool(torch.cuda.is_initialized()),
    }
    if result["device"] != "cpu" or result["torch_cuda_available"] or result["torch_cuda_device_count"]:
        raise RuntimeError(f"CPU-only preflight failed: {result}")
    return result


def preflight(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    cpu = cpu_preflight()
    inventory, candidate_paths, pieces, configs = discover_candidates()
    humans = discover_humans(pieces)
    write_csv(output / "candidate_inventory.csv", inventory)
    representatives = {}
    for reference_set, rows in humans.items():
        write_csv(output / f"human_inventory_{reference_set}.csv", rows)
        representatives[reference_set] = freeze_representatives(output, reference_set, rows)
    if len(representatives["setA"]) != 23 or len(representatives["setB"]) != 23:
        raise RuntimeError("representative mapping was not frozen for all 23 pieces")
    provenance = {
        "experiment": "final_test_dual_reference_pedal_evaluation_v0",
        "created_at": now(),
        "saved_midi_only": True,
        "model_inference_count": 0,
        "checkpoint_load_count": 0,
        "training_count": 0,
        "candidate_systems": list(CANDIDATE_SYSTEMS),
        "candidate_piece_count_per_system": 23,
        "candidate_manifest_sources": {
            "run_ab": str(AB_MANIFEST),
            "run_ab_sha256": sha256_file(AB_MANIFEST),
            "encoder_only_trio": str(TRIO_MANIFEST),
            "encoder_only_trio_sha256": sha256_file(TRIO_MANIFEST),
        },
        "run_a_identity": {
            "experiment": configs["run_ab_config"]["run_a"]["checkpoint_path"],
            "semantics": "PRE/MAIN/POST",
            "prior_nonpedal_identity": configs["identity_counts"]["RUN A"],
            "note_level_pitch_onset_offset_velocity_identity": "PASS 23/23",
        },
        "run_b_identity": configs["run_ab_config"]["run_b"],
        "encoder_only_identities": {
            "weighted_ce": configs["trio_config"]["weighted_ce"],
            "ntl_was": configs["trio_config"]["ntl_was"],
            "huber_aux_ce": configs["trio_config"]["huber_aux_ce"],
        },
        "reference_sets": {
            "setA": {
                "source": str(SPLIT_CSV),
                "source_sha256": sha256_file(SPLIT_CSV),
                "pieces": 23,
                "humans": 104,
                "human_human_pairs_before_alignment": 81,
            },
            "setB": {
                "source": str(PT_HUMAN_ROOT),
                "mapping": "existing PT evaluator numeric filename prefix joined exactly to canonical numbered test piece 0..22",
                "piece_match": "23/23 PASS",
                "pieces": 23,
                "humans": 166,
                "human_human_pairs_before_alignment": 143,
            },
        },
        "representative_rule": "within each independent reference set and piece, stable lexicographic (performance_id, path); first is H0; self comparison excluded",
        "representative_mappings_frozen_before_metric_computation": True,
        "class_boundaries": [[0, 25], [26, 63], [64, 103], [104, 127]],
        "pattern_bins": 256,
        "transition": {
            "threshold": 64,
            "tolerance_distinct_onsets": TRANSITION_TOLERANCE,
            "matcher": "frozen direction-aware greedy one-to-one",
        },
        "repedal": {
            "threshold": THRESHOLD,
            "minimum_excursion": MIN_EXCURSION,
            "maximum_off_duration_ms": MAX_OFF_DURATION_MS,
            "tolerance_distinct_onsets": ONSET_TOLERANCE,
            "deduplication": "same (UP tick, DOWN tick)",
            "matcher": "frozen deterministic greedy one-to-one",
        },
        "aggregation": {
            "4class": "pooled confusion",
            "pattern": "pooled candidate/reference 256-bin histograms",
            "transition": "pair TP/FP/FN micro-sum",
            "repedal": "pair TP/FP/FN micro-sum",
            "per_piece": "recomputed from only that piece's raw comparison statistics",
        },
        "evaluator_reuse": {
            "alignment": "scripts.run_stage2_4class_validation_eval_v0::align_and_cache",
            "4class_pattern_transition": "src.stage2_four_class.validation_evaluator",
            "repedal": "scripts.evaluate_stage2_4class_repedal_validation_v0 (dedup-fixed)",
        },
        "cpu_only_preflight": cpu,
    }
    existing = output / "provenance.json"
    if existing.is_file():
        old = json.loads(existing.read_text(encoding="utf-8"))
        # Timestamps may differ, but every frozen source/definition must remain identical.
        old.pop("created_at", None)
        candidate = dict(provenance)
        candidate.pop("created_at", None)
        if old != candidate:
            raise RuntimeError("existing frozen provenance differs from current preflight")
    write_json(existing, provenance)
    update_status(
        output,
        status="prepared",
        state="preflight_pass",
        candidate_systems=6,
        candidate_pieces_each=23,
        set_a_pieces=23,
        set_a_humans=104,
        set_b_pieces=23,
        set_b_humans=166,
        set_b_piece_match="23/23 PASS",
        representative_mappings_frozen=True,
        cpu_only=True,
        report_generated=False,
        error=None,
        traceback=None,
    )
    return {
        "candidate_paths": candidate_paths,
        "pieces": pieces,
        "humans": humans,
        "representatives": representatives,
        "provenance": provenance,
    }


def new_transition() -> dict[str, Any]:
    return {
        scope: {key: 0 for key in ("candidate", "reference", "tp", "fp", "fn")}
        for scope in ("pooled", "up", "down")
    }


def new_stats() -> dict[str, Any]:
    return {
        "confusion": np.zeros((4, 4), dtype=np.int64),
        "candidate_histogram": np.zeros(PATTERN_COUNT, dtype=np.int64),
        "target_histogram": np.zeros(PATTERN_COUNT, dtype=np.int64),
        "transition": new_transition(),
        "repedal_candidate": 0,
        "repedal_reference": 0,
        "repedal_tp": 0,
        "repedal_maximum_tp": 0,
        "pair_count": 0,
        "aligned_notes": 0,
    }


def update_stats(
    stats: dict[str, Any], candidate: Mapping[str, Any], reference: Mapping[str, Any]
) -> None:
    candidate_classes = candidate["classes"]
    reference_classes = reference["classes"]
    if candidate_classes.shape != reference_classes.shape:
        raise RuntimeError("candidate/reference aligned class shapes differ")
    stats["confusion"] += confusion_from_pairs(candidate_classes, reference_classes)
    stats["candidate_histogram"] += np.bincount(
        pattern_ids(candidate_classes), minlength=PATTERN_COUNT
    )
    stats["target_histogram"] += np.bincount(
        pattern_ids(reference_classes), minlength=PATTERN_COUNT
    )
    counts = pooled_transition_counts(candidate["transitions"], reference["transitions"])
    sum_transition(stats["transition"], counts)
    matched = match_repedals(candidate["repedals"], reference["repedals"])
    stats["repedal_candidate"] += len(candidate["repedals"])
    stats["repedal_reference"] += len(reference["repedals"])
    stats["repedal_tp"] += len(matched)
    stats["repedal_maximum_tp"] += maximum_cardinality_match_count(
        candidate["repedals"], reference["repedals"]
    )
    stats["pair_count"] += 1
    stats["aligned_notes"] += len(candidate_classes)


def finalize_stats(stats: Mapping[str, Any]) -> dict[str, Any]:
    if not stats["pair_count"]:
        return {
            "successful_pair_count": 0,
            "aligned_note_count": 0,
            "aligned_pedal_sample_count": 0,
            "4C_accuracy": None,
            "macro_f1": None,
            "js_divergence": None,
            "intersection": None,
            "transition_precision": None,
            "transition_recall": None,
            "transition_f1": None,
            "transition_tp": 0,
            "transition_fp": 0,
            "transition_fn": 0,
            "candidate_transition_count": 0,
            "reference_transition_count": 0,
            "repedal_precision": None,
            "repedal_recall": None,
            "repedal_f1": None,
            "repedal_tp": 0,
            "repedal_fp": 0,
            "repedal_fn": 0,
            "candidate_repedal_count": 0,
            "reference_repedal_count": 0,
            "repedal_maximum_cardinality_tp_diagnostic": 0,
            "classification": None,
            "patterns": None,
            "transition": None,
        }
    classification = classification_metrics(stats["confusion"])
    patterns = pattern_metrics(stats["candidate_histogram"], stats["target_histogram"])
    transition = finalize_transition(json.loads(json.dumps(stats["transition"])))
    repedal = metric_row(
        int(stats["repedal_candidate"]),
        int(stats["repedal_reference"]),
        int(stats["repedal_tp"]),
    )
    pooled = transition["pooled"]
    if pooled["tp"] + pooled["fp"] != pooled["candidate"]:
        raise AssertionError("Transition candidate invariant failed")
    if pooled["tp"] + pooled["fn"] != pooled["reference"]:
        raise AssertionError("Transition reference invariant failed")
    if repedal["tp"] + repedal["fp"] != repedal["candidate_repedal_count"]:
        raise AssertionError("Repedal candidate invariant failed")
    if repedal["tp"] + repedal["fn"] != repedal["reference_repedal_count"]:
        raise AssertionError("Repedal reference invariant failed")
    if not (0.0 <= patterns["js_divergence_base2"] <= 1.0 + 1e-12):
        raise AssertionError("JS divergence escaped [0,1]")
    if not (0.0 <= patterns["intersection"] <= 1.0 + 1e-12):
        raise AssertionError("intersection escaped [0,1]")
    return {
        "successful_pair_count": int(stats["pair_count"]),
        "aligned_note_count": int(stats["aligned_notes"]),
        "aligned_pedal_sample_count": int(stats["aligned_notes"] * 4),
        "4C_accuracy": classification["token_accuracy"],
        "macro_f1": classification["macro_f1"],
        "js_divergence": patterns["js_divergence_base2"],
        "intersection": patterns["intersection"],
        "transition_precision": pooled["precision"],
        "transition_recall": pooled["recall"],
        "transition_f1": pooled["f1"],
        "transition_tp": pooled["tp"],
        "transition_fp": pooled["fp"],
        "transition_fn": pooled["fn"],
        "candidate_transition_count": pooled["candidate"],
        "reference_transition_count": pooled["reference"],
        "repedal_precision": repedal["precision"],
        "repedal_recall": repedal["recall"],
        "repedal_f1": repedal["f1"],
        "repedal_tp": repedal["tp"],
        "repedal_fp": repedal["fp"],
        "repedal_fn": repedal["fn"],
        "candidate_repedal_count": repedal["candidate_repedal_count"],
        "reference_repedal_count": repedal["reference_repedal_count"],
        "repedal_maximum_cardinality_tp_diagnostic": int(stats["repedal_maximum_tp"]),
        "classification": classification,
        "patterns": patterns,
        "transition": transition,
    }


def feature_from_alignment(
    *,
    alignment: Mapping[str, Any],
    transitions_path: Path,
    midi_path: Path,
    candidate: str,
    piece_id: str,
    performance_id: str,
    role: str,
) -> dict[str, Any]:
    mapping = transition_mapping(transitions_path, midi_path)
    repedals = extract_repedals(
        midi_path,
        mapping=mapping,
        candidate=candidate,
        piece_id=piece_id,
        performance_id=performance_id,
        role=role,
    )
    return {
        "classes": canonical_classes_from_tokens(alignment["tokens"]),
        "transitions": alignment["transitions"],
        "repedals": repedals,
    }


def align_all(
    output: Path,
    discovered: Mapping[str, Any],
    log,
) -> tuple[
    dict[tuple[str, str], dict[str, Any]],
    dict[tuple[str, str], dict[str, Any]],
    dict[str, list[dict[str, Any]]],
]:
    candidate_alignments: dict[tuple[str, str], dict[str, Any]] = {}
    human_alignments: dict[tuple[str, str], dict[str, Any]] = {}
    failures: dict[str, list[dict[str, Any]]] = {"setA": [], "setB": []}
    pieces = discovered["pieces"]
    piece_by_id = {row["piece_id"]: row for row in pieces}

    first_pass = False
    total = sum(len(value) for value in discovered["humans"].values()) + 6 * 23
    progress = 0
    for reference_set in ("setA", "setB"):
        cache_root = output / "alignment_cache" / reference_set
        for human in discovered["humans"][reference_set]:
            progress += 1
            piece = piece_by_id[human["piece_id"]]
            identifier = str(human["identifier"])
            update_status(
                output,
                status="running",
                state="alignment",
                alignment_progress=progress,
                alignment_total=total,
                current_reference_set=reference_set,
                current_kind="human",
                current_piece=human["piece_id"],
                current_performance=human["performance_id"],
            )
            log(
                f"ALIGNMENT_START reference_set={reference_set} kind=human "
                f"piece={human['piece_id']} performance={human['performance_id']}"
            )
            try:
                result = align_and_cache(
                    cache_root,
                    kind="human",
                    identifier=identifier,
                    score_path=Path(piece["score_path"]),
                    performance_path=Path(human["midi_path"]),
                )
                human_alignments[(reference_set, identifier)] = result
                log(
                    f"ALIGNMENT_PASS reference_set={reference_set} kind=human "
                    f"piece={human['piece_id']} performance={human['performance_id']} "
                    f"notes={len(result['tokens'])} cache_reused={result['cache_reused']}"
                )
                if not first_pass:
                    first_pass = True
                    update_status(
                        output,
                        first_alignment_pass=True,
                        first_alignment_reference_set=reference_set,
                        first_alignment_piece=human["piece_id"],
                        first_alignment_performance=human["performance_id"],
                    )
                    log("FIRST_ALIGNMENT_PASS")
            except Exception as error:
                failures[reference_set].append(
                    {
                        "stage": "alignment",
                        "system": "Human reference",
                        "piece_id": human["piece_id"],
                        "performance_id": human["performance_id"],
                        "reason": f"{type(error).__name__}: {error}",
                    }
                )
                log(
                    f"ALIGNMENT_FAIL reference_set={reference_set} kind=human "
                    f"piece={human['piece_id']} error={error!r}"
                )

    cache_root = output / "alignment_cache" / "candidates"
    for system in CANDIDATE_SYSTEMS:
        kind = SYSTEM_KEYS[system]
        for piece in pieces:
            progress += 1
            piece_id = piece["piece_id"]
            midi_path = discovered["candidate_paths"][system][piece_id]
            update_status(
                output,
                status="running",
                state="alignment",
                alignment_progress=progress,
                alignment_total=total,
                current_reference_set="shared_candidates",
                current_kind=system,
                current_piece=piece_id,
                current_performance=str(midi_path),
            )
            log(f"ALIGNMENT_START reference_set=shared kind={system} piece={piece_id}")
            try:
                result = align_and_cache(
                    cache_root,
                    kind=kind,
                    identifier=piece_id,
                    score_path=Path(piece["score_path"]),
                    performance_path=midi_path,
                )
                candidate_alignments[(system, piece_id)] = result
                log(
                    f"ALIGNMENT_PASS reference_set=shared kind={system} piece={piece_id} "
                    f"notes={len(result['tokens'])} cache_reused={result['cache_reused']}"
                )
            except Exception as error:
                for reference_set in failures:
                    failures[reference_set].append(
                        {
                            "stage": "alignment",
                            "system": system,
                            "piece_id": piece_id,
                            "performance_id": str(midi_path),
                            "reason": f"{type(error).__name__}: {error}",
                        }
                    )
                log(f"ALIGNMENT_FAIL reference_set=shared kind={system} piece={piece_id} error={error!r}")
    return candidate_alignments, human_alignments, failures


def build_features(
    output: Path,
    discovered: Mapping[str, Any],
    candidate_alignments: Mapping[tuple[str, str], Mapping[str, Any]],
    human_alignments: Mapping[tuple[str, str], Mapping[str, Any]],
    failures: dict[str, list[dict[str, Any]]],
    log,
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    candidate_features = {}
    human_features = {}
    candidate_cache = output / "alignment_cache" / "candidates"
    for system in CANDIDATE_SYSTEMS:
        kind = SYSTEM_KEYS[system]
        for piece in discovered["pieces"]:
            piece_id = piece["piece_id"]
            alignment = candidate_alignments.get((system, piece_id))
            if alignment is None:
                continue
            midi_path = discovered["candidate_paths"][system][piece_id]
            try:
                candidate_features[(system, piece_id)] = feature_from_alignment(
                    alignment=alignment,
                    transitions_path=alignment_paths(candidate_cache, kind, piece_id)["transitions"],
                    midi_path=midi_path,
                    candidate=system,
                    piece_id=piece_id,
                    performance_id=piece_id,
                    role="candidate",
                )
            except Exception as error:
                for reference_set in failures:
                    failures[reference_set].append(
                        {
                            "stage": "feature_extraction",
                            "system": system,
                            "piece_id": piece_id,
                            "performance_id": str(midi_path),
                            "reason": f"{type(error).__name__}: {error}",
                        }
                    )
                log(f"FEATURE_FAIL system={system} piece={piece_id} error={error!r}")
    for reference_set in ("setA", "setB"):
        cache_root = output / "alignment_cache" / reference_set
        for human in discovered["humans"][reference_set]:
            identifier = str(human["identifier"])
            alignment = human_alignments.get((reference_set, identifier))
            if alignment is None:
                continue
            try:
                human_features[(reference_set, identifier)] = feature_from_alignment(
                    alignment=alignment,
                    transitions_path=alignment_paths(cache_root, "human", identifier)["transitions"],
                    midi_path=Path(human["midi_path"]),
                    candidate="Human",
                    piece_id=human["piece_id"],
                    performance_id=human["performance_id"],
                    role="reference",
                )
            except Exception as error:
                failures[reference_set].append(
                    {
                        "stage": "feature_extraction",
                        "system": "Human reference",
                        "piece_id": human["piece_id"],
                        "performance_id": human["performance_id"],
                        "reason": f"{type(error).__name__}: {error}",
                    }
                )
                log(
                    f"FEATURE_FAIL reference_set={reference_set} human={human['performance_id']} error={error!r}"
                )
    return candidate_features, human_features


def evaluate_reference_set(
    output: Path,
    reference_set: str,
    discovered: Mapping[str, Any],
    candidate_features: Mapping[tuple[str, str], Mapping[str, Any]],
    human_features: Mapping[tuple[str, str], Mapping[str, Any]],
    failures: list[dict[str, Any]],
    log,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    humans = discovered["humans"][reference_set]
    reps = discovered["representatives"][reference_set]
    pieces = discovered["pieces"]
    by_piece: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for human in humans:
        by_piece[human["piece_id"]].append(human)

    common_rows = []
    eligible_candidate_humans: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for human in humans:
        piece_id = human["piece_id"]
        feature = human_features.get((reference_set, str(human["identifier"])))
        missing_systems = [
            system for system in CANDIDATE_SYSTEMS if (system, piece_id) not in candidate_features
        ]
        shape_mismatch = []
        if feature is not None and not missing_systems:
            for system in CANDIDATE_SYSTEMS:
                if candidate_features[(system, piece_id)]["classes"].shape != feature["classes"].shape:
                    shape_mismatch.append(system)
        included = feature is not None and not missing_systems and not shape_mismatch
        reason = ""
        if feature is None:
            reason = "Human alignment/feature unavailable"
        elif missing_systems:
            reason = "candidate feature unavailable: " + ";".join(missing_systems)
        elif shape_mismatch:
            reason = "aligned shape mismatch: " + ";".join(shape_mismatch)
        if included:
            eligible_candidate_humans[piece_id].append(human)
        common_rows.append(
            {
                "reference_set": reference_set,
                "pair_type": "six_candidate_common_reference",
                "piece_id": piece_id,
                "candidate_id": "ALL_SIX_SYSTEMS",
                "reference_id": human["performance_id"],
                "included": included,
                "exclusion_reason": reason,
            }
        )

    hh_pairs: dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = defaultdict(list)
    id_to_human = {str(row["identifier"]): row for row in humans}
    for piece in pieces:
        piece_id = piece["piece_id"]
        h0_id = reps[piece_id]
        h0 = id_to_human[h0_id]
        h0_feature = human_features.get((reference_set, h0_id))
        for target in sorted(
            by_piece[piece_id], key=lambda row: (str(row["performance_id"]), str(row["midi_path"]))
        ):
            if str(target["identifier"]) == h0_id:
                continue
            target_feature = human_features.get((reference_set, str(target["identifier"])))
            included = (
                h0_feature is not None
                and target_feature is not None
                and h0_feature["classes"].shape == target_feature["classes"].shape
            )
            reason = "" if included else "H0/reference alignment, feature, or shape unavailable"
            if included:
                hh_pairs[piece_id].append((h0, target))
            common_rows.append(
                {
                    "reference_set": reference_set,
                    "pair_type": "human_human",
                    "piece_id": piece_id,
                    "candidate_id": h0["performance_id"],
                    "reference_id": target["performance_id"],
                    "included": included,
                    "exclusion_reason": reason,
                }
            )

    global_stats = {system: new_stats() for system in SYSTEM_ORDER}
    piece_stats = {
        (piece["piece_id"], system): new_stats()
        for piece in pieces
        for system in SYSTEM_ORDER
    }
    for piece in pieces:
        piece_id = piece["piece_id"]
        for human in eligible_candidate_humans[piece_id]:
            target = human_features[(reference_set, str(human["identifier"]))]
            for system in CANDIDATE_SYSTEMS:
                candidate = candidate_features[(system, piece_id)]
                update_stats(global_stats[system], candidate, target)
                update_stats(piece_stats[(piece_id, system)], candidate, target)
        h0_id = reps[piece_id]
        if (reference_set, h0_id) in human_features:
            candidate = human_features[(reference_set, h0_id)]
            for _, target_row in hh_pairs[piece_id]:
                target = human_features[(reference_set, str(target_row["identifier"]))]
                update_stats(global_stats["Human–Human"], candidate, target)
                update_stats(piece_stats[(piece_id, "Human–Human")], candidate, target)

    summary_rows = []
    per_piece_rows = []
    diagnostics_root = {
        "confusion": output / "confusion_matrices",
        "transition": output / "transition_diagnostics",
        "repedal": output / "repedal_diagnostics",
        "pattern": output / "pattern_diagnostics",
    }
    for root in diagnostics_root.values():
        root.mkdir(parents=True, exist_ok=True)
    for system in SYSTEM_ORDER:
        result = finalize_stats(global_stats[system])
        summary_rows.append({"System": system, **{key: result[key] for key in (
            "4C_accuracy", "macro_f1", "js_divergence", "intersection",
            "transition_precision", "transition_recall", "transition_f1",
            "repedal_precision", "repedal_recall", "repedal_f1",
        )}})
        slug = SYSTEM_KEYS.get(system, "human_human")
        if result["classification"] is not None:
            matrix = result["classification"]["confusion_matrix"]
            write_csv(
                diagnostics_root["confusion"] / f"{reference_set}_{slug}.csv",
                [
                    {
                        "true_class": CLASS_NAMES[index],
                        **{f"pred_{CLASS_NAMES[column].lower()}": int(matrix[index][column]) for column in range(4)},
                    }
                    for index in range(4)
                ],
            )
            write_csv(
                diagnostics_root["pattern"] / f"{reference_set}_{slug}.csv",
                [
                    {
                        "pattern_id": index,
                        "candidate_probability": result["patterns"]["candidate_distribution"][index],
                        "reference_probability": result["patterns"]["target_distribution"][index],
                    }
                    for index in range(PATTERN_COUNT)
                ],
            )
        write_json(
            diagnostics_root["transition"] / f"{reference_set}_{slug}.json",
            {"system": system, "reference_set": reference_set, "transition": result["transition"]},
        )
        write_json(
            diagnostics_root["repedal"] / f"{reference_set}_{slug}.json",
            {
                "system": system,
                "reference_set": reference_set,
                **{key: result[key] for key in (
                    "candidate_repedal_count", "reference_repedal_count", "repedal_tp",
                    "repedal_fp", "repedal_fn", "repedal_precision", "repedal_recall",
                    "repedal_f1", "repedal_maximum_cardinality_tp_diagnostic",
                )},
            },
        )
    for piece in pieces:
        piece_id = piece["piece_id"]
        for system in SYSTEM_ORDER:
            result = finalize_stats(piece_stats[(piece_id, system)])
            raw_reference_count = (
                len(by_piece[piece_id]) - 1 if system == "Human–Human" else len(by_piece[piece_id])
            )
            per_piece_rows.append(
                {
                    "reference_set": reference_set,
                    "piece_id": piece_id,
                    "composer": piece["composer"],
                    "title": piece["title"],
                    "system": system,
                    "human_reference_count": raw_reference_count,
                    **{key: result[key] for key in (
                        "successful_pair_count", "aligned_note_count", "aligned_pedal_sample_count",
                        "4C_accuracy", "macro_f1", "js_divergence", "intersection",
                        "transition_precision", "transition_recall", "transition_f1",
                        "transition_tp", "transition_fp", "transition_fn",
                        "candidate_transition_count", "reference_transition_count",
                        "repedal_precision", "repedal_recall", "repedal_f1",
                        "repedal_tp", "repedal_fp", "repedal_fn",
                        "candidate_repedal_count", "reference_repedal_count",
                    )},
                }
            )
    if len(per_piece_rows) != 23 * 7:
        raise AssertionError("per-piece output is not 23 x 7 rows")
    write_csv(output / f"common_pairs_{reference_set}.csv", common_rows)
    write_csv(
        output / f"alignment_failures_{reference_set}.csv",
        failures,
        ("stage", "system", "piece_id", "performance_id", "reason"),
    )
    log(
        f"REFERENCE_SET_COMPLETE reference_set={reference_set} "
        f"candidate_common_pairs={sum(len(value) for value in eligible_candidate_humans.values())} "
        f"human_human_pairs={sum(len(value) for value in hh_pairs.values())}"
    )
    return summary_rows, per_piece_rows, common_rows


def fmt(value: Any) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    return f"{float(value):.6f}"


def report_text(
    summaries: Mapping[str, Sequence[Mapping[str, Any]]],
    common_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    failures: Mapping[str, Sequence[Mapping[str, Any]]],
) -> str:
    lines = ["# Final Test Dual-Reference Canonical Pedal Evaluation", ""]
    for index, reference_set in enumerate(("setA", "setB"), 1):
        title = "ASAP canonical 104" if reference_set == "setA" else "PT Human 166"
        lines += [f"## Table {index} — {'Set A' if reference_set == 'setA' else 'Set B'}: {title}", ""]
        lines += [
            "| System | 4C Acc ↑ | Macro F1 ↑ | JS ↓ | Intersection ↑ | Trans P ↑ | Trans R ↑ | Trans F1 ↑ | Repedal P ↑ | Repedal R ↑ | Repedal F1 ↑ |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in summaries[reference_set]:
            lines.append(
                "| " + " | ".join(
                    [str(row["System"])]
                    + [fmt(row[key]) for key in (
                        "4C_accuracy", "macro_f1", "js_divergence", "intersection",
                        "transition_precision", "transition_recall", "transition_f1",
                        "repedal_precision", "repedal_recall", "repedal_f1",
                    )]
                ) + " |"
            )
        lines.append("")
    lines += [
        "## Reference inventory / pair accounting",
        "",
        "- Set A: 23 pieces / 104 Human performances from `asap_split.csv` test rows; never pooled with Set B.",
        "- Set B: 23 pieces / 166 bundled PT Human MIDI; numeric prefix mapping matched the candidate universe 23/23; never pooled with Set A.",
    ]
    for reference_set in ("setA", "setB"):
        candidate_rows = [row for row in common_rows[reference_set] if row["pair_type"] == "six_candidate_common_reference"]
        hh_rows = [row for row in common_rows[reference_set] if row["pair_type"] == "human_human"]
        lines.append(
            f"- {reference_set}: six-system common pairs {sum(bool(row['included']) for row in candidate_rows)}/{len(candidate_rows)}; "
            f"Human–Human pairs {sum(bool(row['included']) for row in hh_rows)}/{len(hh_rows)}; alignment/feature failures {len(failures[reference_set])}."
        )
    lines += [
        "",
        "## Human–Human representative rule",
        "",
        "Within each reference universe and piece, `(performance identifier, path)` was sorted lexicographically before any alignment or metric computation. The first entry was frozen as H0 and compared with every other Human. H0↔H0 self-comparisons are zero. The two mappings are stored separately.",
        "",
        "## Alignment failures",
        "",
        "Failures are recorded separately for Set A and Set B. Any unavailable Human comparison unit or candidate piece was excluded from all six model rows of that reference set. Human–Human retains its separate fixed-H0 pair inventory.",
        "",
        "## RUN A EOT identity provenance",
        "",
        "RUN A is the final `stage2_binary_2slot_D_pre_main_post_full_v1` PRE/MAIN/POST candidate. Prior inference verification recorded 13/23 strict identities and 10/23 frozen terminal/EOT-extension-only identities. This evaluation independently confirmed exact note count, pitch, onset, offset, and velocity identity for 23/23 before parsing pedal metrics.",
        "",
        "## Canonical artifacts",
        "",
        "- `summary_setA_asap104.csv`, `summary_setB_pt166.csv`",
        "- `per_piece_setA_asap104.csv`, `per_piece_setB_pt166.csv` (23 × 7 rows each)",
        "- `reference_set_sensitivity.csv` uses Set B minus Set A.",
        "- Confusion, Transition, Repedal, and 256-pattern diagnostics are under their respective subdirectories.",
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
        discovered = preflight(output)
        update_status(
            output,
            status="running",
            state="runner_started",
            started_at=now(),
            first_alignment_pass=False,
            report_generated=False,
        )
        log(
            "RUNNER_STARTED cpu_only=true candidates=6x23 setA=23/104 "
            "setB=23/166 setB_piece_match=23/23 representatives=frozen"
        )
        candidate_alignments, human_alignments, failures = align_all(output, discovered, log)
        update_status(output, status="running", state="feature_extraction")
        candidate_features, human_features = build_features(
            output,
            discovered,
            candidate_alignments,
            human_alignments,
            failures,
            log,
        )
        summaries = {}
        per_piece = {}
        common = {}
        for reference_set in ("setA", "setB"):
            update_status(
                output,
                status="running",
                state="metric_computation",
                current_reference_set=reference_set,
            )
            summaries[reference_set], per_piece[reference_set], common[reference_set] = evaluate_reference_set(
                output,
                reference_set,
                discovered,
                candidate_features,
                human_features,
                failures[reference_set],
                log,
            )
            slug = REFERENCE_SETS[reference_set]["slug"]
            write_csv(output / f"summary_{slug}.csv", summaries[reference_set])
            write_csv(output / f"per_piece_{slug}.csv", per_piece[reference_set])

        by_set = {
            reference_set: {row["System"]: row for row in rows}
            for reference_set, rows in summaries.items()
        }
        sensitivity = []
        for system in SYSTEM_ORDER:
            a = by_set["setA"][system]
            b = by_set["setB"][system]
            sensitivity.append(
                {
                    "system": system,
                    **{
                        f"delta_{name}": None
                        if a[key] is None or b[key] is None
                        else float(b[key]) - float(a[key])
                        for name, key in (
                            ("4C_accuracy", "4C_accuracy"),
                            ("macro_f1", "macro_f1"),
                            ("js", "js_divergence"),
                            ("intersection", "intersection"),
                            ("transition_f1", "transition_f1"),
                            ("repedal_f1", "repedal_f1"),
                        )
                    },
                }
            )
        write_csv(output / "reference_set_sensitivity.csv", sensitivity)
        report = output / "FINAL_TEST_DUAL_REFERENCE_EVALUATION.md"
        atomic_text(report, report_text(summaries, common, failures))
        update_status(
            output,
            status="completed",
            state="completed",
            completed_at=now(),
            report_generated=True,
            report_path=str(report),
            set_a_alignment_failures=len(failures["setA"]),
            set_b_alignment_failures=len(failures["setB"]),
            summary_files=2,
            per_piece_files=2,
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
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight-only", action="store_true")
    group.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    output = args.output_root.resolve()
    if args.preflight_only:
        result = preflight(output)
        print(
            json.dumps(
                {
                    "status": "prepared",
                    "output": str(output),
                    "candidate_systems": len(result["candidate_paths"]),
                    "candidate_pieces_each": 23,
                    "set_a_humans": len(result["humans"]["setA"]),
                    "set_b_humans": len(result["humans"]["setB"]),
                    "set_b_piece_match": "23/23 PASS",
                    "representatives_frozen": True,
                    "cpu_only": True,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    return run(output)


if __name__ == "__main__":
    raise SystemExit(main())
