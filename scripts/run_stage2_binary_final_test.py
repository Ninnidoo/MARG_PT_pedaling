#!/usr/bin/env python3
"""Run the locked ASAP test evaluation as one self-contained process."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import logging
import os
import random
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from miditoolkit import MidiFile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_stage2_binary_metric_fidelity import (  # noqa: E402
    _assert_nonpedal_midi_equality,
    _strict_official_similarity,
)
from scripts.build_stage2_validation_baseline import (  # noqa: E402
    PianoT5Gemma,
    _atomic_json,
    _atomic_npy,
    _fix_ownership,
    _ids_sha256,
    _seed_everything,
    _sha256,
    _write_csv,
    batch_performance_render,
    map_midi,
    midi_to_ids,
)
from src.stage2_binary.strict_midi_validation import (  # noqa: E402
    STRICT_EVALUATOR_PROVENANCE,
    PianoT5GemmaConfig,
    ids_to_midi,
)
from src.stage2_binary.validation_evaluator import (  # noqa: E402
    JOINT_PATTERNS,
    infer_cached_binary_pedals,
    joint16_histogram_from_ids,
    load_binary_stage2_checkpoint,
)
from src.stage2_encoder_only.dataset import TOKENS_PER_NOTE  # noqa: E402


LOCK_RELATIVE = Path("analysis/stage2_binary_v0/final_lock_v1/final_experiment_lock.json")
OUTPUT_RELATIVE = Path("analysis/stage2_binary_v0/test_eval_v0")
SPLIT_RELATIVE = Path("analysis/stage2_encoder_only_v0/asap_split.csv")
VALIDATION_CONFIG_RELATIVE = Path("configs/stage2_binary_validation_eval_v0.json")
VALIDATION_METRICS_RELATIVE = Path(
    "analysis/stage2_binary_v0/validation_eval_v0/original_pt_validation_metrics.json"
)
ASAP_ROOT = Path("/workspace/public/ASAP/asap-dataset-v1.1")
ASAP_METADATA = ASAP_ROOT / "metadata.csv"
STAGE1_CHECKPOINT_RELATIVE = Path("checkpoints/pianist_transformer")
EXPECTED_SPLIT_SHA256 = "d1fe379eb123ff7abca93773296f735f40a215dcd6e6363b55756383708c965b"
EXPECTED_TEST_PIECES = 23
EXPECTED_TEST_PERFORMANCES = 104
SELECTION_RULE = "max direct test-row metadata support; lexical tie-break"
ARCHITECTURES = ("independent_4x2", "joint_16")
EQUALITY_FIELDS = (
    "architecture", "piece_id", "composer", "title", "original_pt_midi",
    "stage2_candidate_midi", "original_note_count", "candidate_note_count",
    "note_count_exact", "pitch_exact", "onset_exact", "offset_exact",
    "duration_exact", "velocity_exact", "note_order_and_events_exact",
    "ticks_per_beat_exact", "instrument_layout_exact", "tempo_changes_exact",
    "time_signatures_exact", "key_signatures_exact", "non_cc64_controls_exact",
    "markers_exact", "lyrics_exact", "original_cc64_event_count",
    "candidate_cc64_event_count", "cc64_events_exact",
    "all_nonpedal_midi_exact",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _probability(histogram: Sequence[int]) -> np.ndarray:
    values = np.asarray(histogram, dtype=np.float64)
    if values.shape != (16,) or np.any(values < 0) or values.sum() <= 0:
        raise ValueError("expected a non-empty 16-bin histogram")
    result = values / values.sum()
    if not np.isclose(result.sum(), 1.0, rtol=0.0, atol=1e-15):
        raise AssertionError("empirical distribution does not sum to one")
    return result


def _load_locked_context() -> dict[str, Any]:
    lock_path = _resolve(LOCK_RELATIVE)
    lock = _read_json(lock_path)
    if lock["status"] != "definitive_strict_locked_before_asap_test":
        raise RuntimeError("final_lock_v1 is not in the definitive pre-test state")
    if lock["selection"]["primary_architecture"] != "independent_4x2":
        raise RuntimeError("locked primary architecture changed")
    if lock["selection"]["comparison_architecture"] != "joint_16":
        raise RuntimeError("locked comparison architecture changed")
    if int(lock["inference_lock"]["binary_threshold"]) != 64:
        raise RuntimeError("locked threshold is not 64")
    if lock["inference_lock"]["decoding"] != "deterministic argmax":
        raise RuntimeError("locked decoding is not deterministic argmax")
    if lock["inference_lock"]["sampling"] or lock["inference_lock"]["test_time_calibration"]:
        raise RuntimeError("locked inference unexpectedly enables sampling/calibration")
    if int(lock["inference_lock"]["window_notes"]) != 512:
        raise RuntimeError("locked Stage 2 window changed")
    if int(lock["inference_lock"]["stride_notes"]) != 256:
        raise RuntimeError("locked Stage 2 stride changed")

    verified_candidates: dict[str, dict[str, Any]] = {}
    expected_epochs = {"independent_4x2": 2, "joint_16": 3}
    for architecture in ARCHITECTURES:
        item = lock["candidates"][architecture]
        if item["architecture"] != architecture:
            raise RuntimeError(f"lock architecture mismatch: {architecture}")
        if int(item["best_epoch"]) != expected_epochs[architecture]:
            raise RuntimeError(f"locked best epoch changed: {architecture}")
        checkpoint = _resolve(item["checkpoint_project_relative_path"])
        if _sha256(checkpoint) != item["checkpoint_sha256"]:
            raise RuntimeError(f"locked checkpoint SHA-256 mismatch: {architecture}")
        config = checkpoint.parent / "config.json"
        if _sha256(config) != item["config_sha256"]:
            raise RuntimeError(f"locked config SHA-256 mismatch: {architecture}")
        verified_candidates[architecture] = {
            **item,
            "checkpoint": checkpoint,
            "config": config,
        }

    stage1_checkpoint = _resolve(STAGE1_CHECKPOINT_RELATIVE)
    stage1_model = stage1_checkpoint / "model.safetensors"
    if _sha256(stage1_model) != lock["inference_lock"][
        "stage1_checkpoint_model_safetensors_sha256"
    ]:
        raise RuntimeError("Original PT checkpoint SHA-256 mismatch")
    for source_key, path in (
        ("official_evaluator_sha256", PROJECT_ROOT / "third_party/PianistTransformer/src/evaluate/evaluate.py"),
        ("official_tokenizer_sha256", PROJECT_ROOT / "third_party/PianistTransformer/src/utils/midi.py"),
        ("official_mapping_sha256", PROJECT_ROOT / "third_party/PianistTransformer/src/model/generate.py"),
    ):
        if _sha256(path) != lock["strict_evaluator_provenance"][source_key]:
            raise RuntimeError(f"pinned official source hash mismatch: {path}")

    split = _resolve(SPLIT_RELATIVE)
    if _sha256(split) != EXPECTED_SPLIT_SHA256:
        raise RuntimeError("fixed ASAP split SHA-256 mismatch")
    validation_config = _read_json(_resolve(VALIDATION_CONFIG_RELATIVE))
    expected_validation = {
        "seed": 42,
        "asap_root": str(ASAP_ROOT),
        "asap_metadata": str(ASAP_METADATA),
        "split_csv": str(SPLIT_RELATIVE),
        "stage1_checkpoint": str(STAGE1_CHECKPOINT_RELATIVE),
        "device": "cuda:0",
        "torch_dtype": "float32",
        "temperature": 1.0,
        "top_p": 0.95,
        "max_context_length": 4096,
        "overlap_ratio": 0.5,
    }
    for key, value in expected_validation.items():
        if validation_config.get(key) != value:
            raise RuntimeError(f"validation inference config mismatch: {key}")
    validation_metrics = _read_json(_resolve(VALIDATION_METRICS_RELATIVE))
    inference = validation_metrics["inference"]
    required_inference = {
        "do_sample": True,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 50,
        "num_beams": 1,
        "repetition_penalty": 1.0,
        "max_context_length": 4096,
        "overlap_ratio": 0.5,
        "seed_reset_per_piece": True,
    }
    for key, value in required_inference.items():
        if inference.get(key) != value:
            raise RuntimeError(f"validation provenance mismatch: {key}")
    if validation_metrics["checkpoint_sha256"] != _sha256(stage1_model):
        raise RuntimeError("validation Stage 1 checkpoint provenance mismatch")

    return {
        "lock": lock,
        "lock_path": lock_path,
        "lock_sha256": _sha256(lock_path),
        "candidates": verified_candidates,
        "stage1_checkpoint": stage1_checkpoint,
        "stage1_model_sha256": _sha256(stage1_model),
        "split": split,
        "split_sha256": _sha256(split),
        "validation_config": validation_config,
        "validation_metrics": validation_metrics,
    }


def _select_test_scores(context: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    split_rows = _read_csv(context["split"])
    metadata_rows = _read_csv(ASAP_METADATA)
    counts = Counter(row["split"] for row in split_rows)
    piece_counts = {
        split: len({row["piece_id"] for row in split_rows if row["split"] == split})
        for split in ("train", "validation", "test")
    }
    if counts != Counter({"train": 892, "validation": 71, "test": 104}):
        raise RuntimeError(f"fixed ASAP split performance counts changed: {counts}")
    if piece_counts != {"train": 180, "validation": 19, "test": 23}:
        raise RuntimeError(f"fixed ASAP split piece counts changed: {piece_counts}")
    test_rows = [row for row in split_rows if row["split"] == "test"]
    grouped: dict[str, list[tuple[dict[str, str], dict[str, str]]]] = defaultdict(list)
    for split_row in test_rows:
        metadata = metadata_rows[int(split_row["metadata_index"])]
        performance = metadata["midi_performance"].replace("\\", "/")
        if performance != split_row["performance_path"].replace("\\", "/"):
            raise RuntimeError("split metadata_index does not identify test performance")
        grouped[split_row["piece_id"]].append((split_row, metadata))
    selected: list[dict[str, Any]] = []
    for piece_id, items in grouped.items():
        score_counts = Counter(
            metadata["midi_score"].replace("\\", "/") for _, metadata in items
        )
        support = max(score_counts.values())
        selected_path = min(path for path, count in score_counts.items() if count == support)
        first = items[0][0]
        absolute = (ASAP_ROOT / selected_path).resolve(strict=True)
        selected.append(
            {
                "piece_id": piece_id,
                "composer": first["composer"],
                "title": first["title"],
                "test_performance_count": len(items),
                "selected_score_path": selected_path,
                "selected_score_absolute_path": str(absolute),
                "selected_score_support": support,
                "score_candidate_count": len(score_counts),
                "alternative_candidate_score_paths": ";".join(
                    f"{path}|{score_counts[path]}" for path in sorted(score_counts)
                ),
                "selected_score_sha256": _sha256(absolute),
                "selection_rule": SELECTION_RULE,
            }
        )
    selected.sort(key=lambda row: (row["composer"], row["title"], row["piece_id"]))
    if len(selected) != EXPECTED_TEST_PIECES or len(test_rows) != EXPECTED_TEST_PERFORMANCES:
        raise RuntimeError("ASAP test scope is not exactly 23 pieces / 104 performances")
    if sum(int(row["test_performance_count"]) for row in selected) != 104:
        raise RuntimeError("selected test score support manifest does not cover 104 performances")
    return selected, test_rows


class FinalTestMidiGuard:
    def __init__(
        self,
        human_paths: Sequence[Path],
        score_paths: Sequence[Path],
        original_paths: Sequence[Path],
        candidate_paths: Sequence[Path],
    ) -> None:
        self.allowed = {
            "human_test": {path.resolve() for path in human_paths},
            "test_score": {path.resolve() for path in score_paths},
            "original_pt_cache": {path.resolve() for path in original_paths},
            "stage2_candidate": {path.resolve() for path in candidate_paths},
        }
        self.counts = {key: 0 for key in self.allowed}

    def load(self, path: Path, category: str) -> MidiFile:
        resolved = path.resolve()
        if category not in self.allowed or resolved not in self.allowed[category]:
            raise PermissionError(f"MIDI path outside locked final-test allowlist: {path}")
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        self.counts[category] += 1
        return MidiFile(str(resolved))


def _cache_paths(cache_root: Path, piece_id: str) -> dict[str, Path]:
    root = cache_root / piece_id
    return {
        "root": root,
        "ids": root / "generated_ids_int64.npy",
        "midi": root / "original_pt.mid",
        "metadata": root / "inference_metadata.json",
    }


def _stage1_generate_one(
    model: PianoT5Gemma,
    score_row: Mapping[str, Any],
    paths: Mapping[str, Path],
    context: Mapping[str, Any],
    guard: FinalTestMidiGuard,
) -> dict[str, Any]:
    paths["root"].mkdir(parents=True, exist_ok=False)
    score_path = Path(score_row["selected_score_absolute_path"])
    _seed_everything(42)
    score = guard.load(score_path, "test_score")
    score_ids = midi_to_ids(model.config, score)
    started = time.perf_counter()
    performances, generated = batch_performance_render(
        model,
        [score],
        max_context_length=4096,
        overlap_ratio=0.5,
        temperature=1.0,
        top_p=0.95,
        device="cuda:0",
    )
    runtime = time.perf_counter() - started
    if len(performances) != 1 or len(generated) != 1:
        raise RuntimeError("official PT did not return exactly one generated performance")
    generated_ids = generated[0]
    if len(generated_ids) != len(score_ids) or len(generated_ids) % TOKENS_PER_NOTE:
        raise RuntimeError("Stage 1 generated length does not match score")
    score_notes = np.asarray(score_ids, dtype=np.int64).reshape(-1, 8)
    generated_notes = np.asarray(generated_ids, dtype=np.int64).reshape(-1, 8)
    if not np.array_equal(score_notes[:, 0], generated_notes[:, 0]):
        raise RuntimeError("official pitch hard constraint failed")
    mapped = map_midi(guard.load(score_path, "test_score"), performances[0])
    temporary = paths["midi"].with_name(paths["midi"].name + ".tmp")
    mapped.dump(str(temporary))
    temporary.replace(paths["midi"])
    _atomic_npy(paths["ids"], generated_ids)
    inference = context["validation_metrics"]["inference"]
    metadata = {
        "piece_id": score_row["piece_id"],
        "composer": score_row["composer"],
        "title": score_row["title"],
        "source_score_relative_path": score_row["selected_score_path"],
        "source_score_path": str(score_path),
        "source_score_sha256": score_row["selected_score_sha256"],
        "official_pt_checkpoint_path": str(context["stage1_checkpoint"]),
        "official_pt_checkpoint_model_safetensors_sha256": context["stage1_model_sha256"],
        "seed": 42,
        "seed_reset_per_piece": True,
        "official_inference_function": "third_party.PianistTransformer.src.model.generate.batch_performance_render",
        "do_sample": True,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": int(inference["top_k"]),
        "num_beams": int(inference["num_beams"]),
        "repetition_penalty": float(inference["repetition_penalty"]),
        "max_context_length": 4096,
        "overlap_ratio": 0.5,
        "pitch_hard_constraint_verified": True,
        "score_tokens": len(score_ids),
        "generated_tokens": len(generated_ids),
        "generated_notes": len(generated_ids) // TOKENS_PER_NOTE,
        "generated_ids_sha256_int64_le": _ids_sha256(generated_ids),
        "generated_ids_path": str(paths["ids"]),
        "generated_midi_sha256": _sha256(paths["midi"]),
        "generated_midi_path": str(paths["midi"]),
        "runtime_seconds": runtime,
    }
    _atomic_json(paths["metadata"], metadata)
    return metadata


def _verify_stage1_cache(records: Sequence[Mapping[str, Any]]) -> None:
    if len(records) != 23:
        raise RuntimeError("Stage 1 cache count is not 23")
    for record in records:
        ids = np.load(record["generated_ids_path"], allow_pickle=False)
        if ids.dtype != np.int64 or ids.ndim != 1 or len(ids) % 8:
            raise RuntimeError(f"invalid Stage 1 ID cache: {record['piece_id']}")
        if _ids_sha256(ids) != record["generated_ids_sha256_int64_le"]:
            raise RuntimeError(f"Stage 1 ID hash mismatch: {record['piece_id']}")
        if _sha256(Path(record["generated_midi_path"])) != record["generated_midi_sha256"]:
            raise RuntimeError(f"Stage 1 MIDI hash mismatch: {record['piece_id']}")
        metadata = _read_json(Path(record["generated_ids_path"]).parent / "inference_metadata.json")
        if metadata["generated_ids_sha256_int64_le"] != record["generated_ids_sha256_int64_le"]:
            raise RuntimeError(f"Stage 1 metadata mismatch: {record['piece_id']}")


def _render_stage2(
    architecture: str,
    records: Sequence[Mapping[str, Any]],
    score_by_piece: Mapping[str, Mapping[str, Any]],
    context: Mapping[str, Any],
    output_root: Path,
    guard: FinalTestMidiGuard,
    equality_rows: list[dict[str, Any]],
    logger: logging.Logger,
) -> dict[str, Any]:
    device = torch.device("cuda:0")
    model = load_binary_stage2_checkpoint(
        context["candidates"][architecture]["checkpoint"],
        architecture=architecture,
        encoder_checkpoint=context["stage1_checkpoint"],
        device=device,
    )
    candidate_root = output_root / "candidate_midis" / architecture
    # Match the strict validation/audit path.  Loading this custom config via
    # from_pretrained re-injects the serialized decoder field into an
    # __init__ that already supplies it under the pinned transformers version.
    tokenizer_config = PianoT5GemmaConfig()
    total_notes = 0
    total_windows = 0
    for index, record in enumerate(records, start=1):
        piece_id = record["piece_id"]
        cached_ids = np.load(record["generated_ids_path"], allow_pickle=False)
        candidate_ids, details = infer_cached_binary_pedals(
            model,
            cached_ids,
            architecture=architecture,
            device=device,
            window_notes=512,
            stride_notes=256,
        )
        if not details["non_pedal_tokens_preserved"]:
            raise AssertionError("Stage 2 changed non-pedal token content")
        score_path = Path(score_by_piece[piece_id]["selected_score_absolute_path"])
        score_ids = midi_to_ids(tokenizer_config, guard.load(score_path, "test_score"))
        performance = ids_to_midi(tokenizer_config, candidate_ids, ref=score_ids)
        mapped = map_midi(guard.load(score_path, "test_score"), performance)
        candidate_path = candidate_root / f"{piece_id}.mid"
        mapped.dump(str(candidate_path))
        row = _assert_nonpedal_midi_equality(
            guard.load(Path(record["generated_midi_path"]), "original_pt_cache"),
            guard.load(candidate_path, "stage2_candidate"),
            metadata={
                "architecture": architecture,
                "piece_id": piece_id,
                "composer": record["composer"],
                "title": record["title"],
                "original_pt_midi": record["generated_midi_path"],
                "stage2_candidate_midi": str(candidate_path),
            },
            raise_on_mismatch=False,
        )
        equality_rows.append(row)
        _write_csv(output_root / "midi_nonpedal_equality.csv", equality_rows, EQUALITY_FIELDS)
        if not row["all_nonpedal_midi_exact"]:
            failed = [key for key in EQUALITY_FIELDS if key.endswith("_exact") and not row.get(key, True)]
            raise AssertionError(f"non-pedal MIDI mismatch: {architecture}/{piece_id}: {failed}")
        total_notes += int(details["notes"])
        total_windows += int(details["windows"])
        logger.info(
            "STAGE2 architecture=%s piece=%d/23 piece_id=%s notes=%d windows=%d",
            architecture, index, piece_id, details["notes"], details["windows"],
        )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    if len(list(candidate_root.glob("*.mid"))) != 23:
        raise RuntimeError(f"candidate MIDI count is not 23: {architecture}")
    return {"notes": total_notes, "windows": total_windows, "midi_count": 23}


def _evaluate(
    records: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, str]],
    output_root: Path,
    context: Mapping[str, Any],
    guard: FinalTestMidiGuard,
    logger: logging.Logger,
) -> dict[str, Any]:
    tokenizer_config = PianoT5GemmaConfig()
    human_global = np.zeros(16, dtype=np.int64)
    human_piece = {record["piece_id"]: np.zeros(16, dtype=np.int64) for record in records}
    human_notes = 0
    human_counts = Counter()
    for index, row in enumerate(test_rows, start=1):
        path = ASAP_ROOT / row["performance_path"]
        ids = midi_to_ids(tokenizer_config, guard.load(path, "human_test"))
        histogram = joint16_histogram_from_ids(ids)
        human_global += histogram
        human_piece[row["piece_id"]] += histogram
        human_notes += int(histogram.sum())
        human_counts[row["piece_id"]] += 1
        logger.info("HUMAN_TEST performance=%d/104 piece_id=%s", index, row["piece_id"])

    labels = ("original_pt", "independent_4x2", "joint_16")
    candidate_global = {label: np.zeros(16, dtype=np.int64) for label in labels}
    candidate_piece = {
        label: {record["piece_id"]: np.zeros(16, dtype=np.int64) for record in records}
        for label in labels
    }
    candidate_notes = Counter()
    for record in records:
        piece_id = record["piece_id"]
        paths = {
            "original_pt": Path(record["generated_midi_path"]),
            "independent_4x2": output_root / "candidate_midis/independent_4x2" / f"{piece_id}.mid",
            "joint_16": output_root / "candidate_midis/joint_16" / f"{piece_id}.mid",
        }
        categories = {
            "original_pt": "original_pt_cache",
            "independent_4x2": "stage2_candidate",
            "joint_16": "stage2_candidate",
        }
        for label in labels:
            ids = midi_to_ids(tokenizer_config, guard.load(paths[label], categories[label]))
            histogram = joint16_histogram_from_ids(ids)
            candidate_global[label] += histogram
            candidate_piece[label][piece_id] += histogram
            candidate_notes[label] += int(histogram.sum())

    global_metrics = {
        label: _strict_official_similarity(human_global, candidate_global[label])
        for label in labels
    }
    probabilities = {
        "human": _probability(human_global),
        **{label: _probability(candidate_global[label]) for label in labels},
    }
    global_rows = []
    for joint_id, pattern in enumerate(JOINT_PATTERNS):
        row: dict[str, Any] = {
            "joint_id": joint_id,
            "pattern": pattern,
            "human_count": int(human_global[joint_id]),
            "human_probability": float(probabilities["human"][joint_id]),
        }
        for label, prefix in (
            ("original_pt", "original_pt"),
            ("independent_4x2", "independent"),
            ("joint_16", "joint"),
        ):
            error = probabilities[label][joint_id] - probabilities["human"][joint_id]
            row[f"{prefix}_count"] = int(candidate_global[label][joint_id])
            row[f"{prefix}_probability"] = float(probabilities[label][joint_id])
            row[f"{prefix}_signed_error_vs_human"] = float(error)
            row[f"{prefix}_absolute_error_vs_human"] = float(abs(error))
        global_rows.append(row)
    _write_csv(
        output_root / "global_joint16_comparison.csv",
        global_rows,
        tuple(global_rows[0].keys()),
    )

    piece_rows = []
    for record in records:
        piece_id = record["piece_id"]
        row: dict[str, Any] = {
            "piece_id": piece_id,
            "composer": record["composer"],
            "title": record["title"],
            "human_performances": human_counts[piece_id],
            "human_notes": int(human_piece[piece_id].sum()),
        }
        for label, prefix in (
            ("original_pt", "original_pt"),
            ("independent_4x2", "independent"),
            ("joint_16", "joint"),
        ):
            metric = _strict_official_similarity(
                human_piece[piece_id], candidate_piece[label][piece_id]
            )
            row[f"{prefix}_candidate_notes"] = int(candidate_piece[label][piece_id].sum())
            row[f"{prefix}_js_distance"] = float(metric["js_distance_base2"])
            row[f"{prefix}_intersection"] = float(metric["histogram_intersection"])
        row["independent_js_improvement_vs_original_pt"] = (
            row["original_pt_js_distance"] - row["independent_js_distance"]
        )
        row["joint_js_improvement_vs_original_pt"] = (
            row["original_pt_js_distance"] - row["joint_js_distance"]
        )
        piece_rows.append(row)
    _write_csv(output_root / "piece_level_metrics.csv", piece_rows, tuple(piece_rows[0].keys()))

    piece_summary: dict[str, Any] = {}
    for prefix in ("original_pt", "independent", "joint"):
        values = np.asarray([row[f"{prefix}_js_distance"] for row in piece_rows])
        piece_summary[prefix] = {
            "mean_js_distance": float(values.mean()),
            "median_js_distance": float(np.median(values)),
        }
    piece_summary["independent"]["pieces_improved_vs_original_pt"] = sum(
        row["independent_js_improvement_vs_original_pt"] > 0 for row in piece_rows
    )
    piece_summary["joint"]["pieces_improved_vs_original_pt"] = sum(
        row["joint_js_improvement_vs_original_pt"] > 0 for row in piece_rows
    )

    masses = {}
    for label, probability in probabilities.items():
        steady = float(probability[0] + probability[15])
        masses[label] = {
            "steady_mass": steady,
            "transition_containing_mass": 1.0 - steady,
        }
    original_js = global_metrics["original_pt"]["js_distance_base2"]
    original_intersection = global_metrics["original_pt"]["histogram_intersection"]
    changes = {}
    for label in ("independent_4x2", "joint_16"):
        changes[label] = {
            "absolute_js_change_candidate_minus_original_pt": float(
                global_metrics[label]["js_distance_base2"] - original_js
            ),
            "relative_js_change_percent": float(
                (global_metrics[label]["js_distance_base2"] - original_js)
                / original_js
                * 100.0
            ),
            "absolute_intersection_change_candidate_minus_original_pt": float(
                global_metrics[label]["histogram_intersection"] - original_intersection
            ),
        }
    return {
        "human_histogram": human_global.tolist(),
        "candidate_histograms": {
            label: candidate_global[label].tolist() for label in labels
        },
        "global_metrics": global_metrics,
        "changes_vs_original_pt": changes,
        "piece_summary": piece_summary,
        "masses": masses,
        "human_performances": 104,
        "human_notes": human_notes,
        "candidate_notes": dict(candidate_notes),
        "piece_rows": piece_rows,
    }


def _report(
    context: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    stage1_records: Sequence[Mapping[str, Any]],
    stage2_summary: Mapping[str, Any],
    equality_rows: Sequence[Mapping[str, Any]],
    score_manifest_sha256: str,
    guard: FinalTestMidiGuard,
) -> str:
    metrics = evaluation["global_metrics"]
    changes = evaluation["changes_vs_original_pt"]
    summary = evaluation["piece_summary"]
    masses = evaluation["masses"]
    independent_improves = (
        metrics["independent_4x2"]["js_distance_base2"]
        < metrics["original_pt"]["js_distance_base2"]
    )
    lines = [
        "# Stage 2 Binary Locked ASAP Final-Test Report",
        "",
        "## Final headline result",
        "",
        "| Model | Pedal JS ↓ | Intersection ↑ |",
        "|---|---:|---:|",
    ]
    for label, display in (
        ("original_pt", "Original PT"),
        ("independent_4x2", "independent 4×2"),
        ("joint_16", "joint 16"),
    ):
        lines.append(
            f"| {display} | {metrics[label]['js_distance_base2']:.12f} | "
            f"{metrics[label]['histogram_intersection']:.12f} |"
        )
    answer = "개선했다" if independent_improves else "개선하지 못했다"
    lines += [
        "",
        f"Locked independent 4×2 Stage 2는 동일 Stage 1 non-pedal performance 위에서 "
        f"Original PT 대비 official strict Pedal JS를 **{answer}**.",
        "",
        "Test 결과를 사용한 parameter/checkpoint/inference 변경은 수행하지 않았다.",
        "",
        "## Changes versus Original PT",
        "",
        "| Model | absolute JS change | relative JS change | Intersection change |",
        "|---|---:|---:|---:|",
    ]
    for label, display in (("independent_4x2", "independent 4×2"), ("joint_16", "joint 16")):
        item = changes[label]
        lines.append(
            f"| {display} | {item['absolute_js_change_candidate_minus_original_pt']:+.12f} | "
            f"{item['relative_js_change_percent']:+.6f}% | "
            f"{item['absolute_intersection_change_candidate_minus_original_pt']:+.12f} |"
        )
    lines += [
        "",
        "## Piece-level supplemental analysis",
        "",
        "| Candidate | improved pieces | mean JS | median JS |",
        "|---|---:|---:|---:|",
        f"| Original PT | — | {summary['original_pt']['mean_js_distance']:.12f} | "
        f"{summary['original_pt']['median_js_distance']:.12f} |",
        f"| independent 4×2 | {summary['independent']['pieces_improved_vs_original_pt']}/23 | "
        f"{summary['independent']['mean_js_distance']:.12f} | "
        f"{summary['independent']['median_js_distance']:.12f} |",
        f"| joint 16 | {summary['joint']['pieces_improved_vs_original_pt']}/23 | "
        f"{summary['joint']['mean_js_distance']:.12f} | "
        f"{summary['joint']['median_js_distance']:.12f} |",
        "",
        "Piece-level 결과는 supplemental이며 model/checkpoint 선택에 사용하지 않았다.",
        "",
        "## Steady / transition-containing mass",
        "",
        "| Distribution | steady mass | transition-containing mass |",
        "|---|---:|---:|",
    ]
    for label, display in (
        ("human", "Human"),
        ("original_pt", "Original PT"),
        ("independent_4x2", "independent 4×2"),
        ("joint_16", "joint 16"),
    ):
        lines.append(
            f"| {display} | {masses[label]['steady_mass']:.12f} | "
            f"{masses[label]['transition_containing_mass']:.12f} |"
        )
    independent = context["candidates"]["independent_4x2"]
    joint = context["candidates"]["joint_16"]
    lines += [
        "",
        "## Locked provenance",
        "",
        f"- final_lock_v1: `{LOCK_RELATIVE}`",
        f"- final_lock_v1 SHA-256: `{context['lock_sha256']}`",
        f"- independent checkpoint: `{independent['checkpoint_project_relative_path']}`",
        f"- independent checkpoint SHA-256: `{independent['checkpoint_sha256']}`",
        f"- joint checkpoint: `{joint['checkpoint_project_relative_path']}`",
        f"- joint checkpoint SHA-256: `{joint['checkpoint_sha256']}`",
        f"- Original PT checkpoint: `{STAGE1_CHECKPOINT_RELATIVE}`",
        f"- Original PT model SHA-256: `{context['stage1_model_sha256']}`",
        f"- ASAP split: `{SPLIT_RELATIVE}`",
        f"- ASAP split SHA-256: `{context['split_sha256']}`",
        f"- test score manifest SHA-256: `{score_manifest_sha256}`",
        "- Stage 1: seed 42 reset per piece; multinomial; temperature 1.0; top-p 0.95; "
        "top-k 50; beams 1; repetition penalty 1.0; context 4096; overlap 0.5",
        "- Stage 2: deterministic argmax; threshold 64; 512-note windows; stride 256; "
        "overlap-logit averaging; binary 0→0 and 1→127",
        f"- strict evaluator: `{STRICT_EVALUATOR_PROVENANCE['official_function']}`; "
        f"pinned commit `{STRICT_EVALUATOR_PROVENANCE['pinned_pt_commit']}`",
        "",
        "## Integrity",
        "",
        f"- ASAP test scope: 23 pieces / 104 human performances",
        f"- selected scores: 23",
        f"- Stage 1 generated outputs: {len(stage1_records)}",
        f"- independent candidate MIDI: {stage2_summary['independent_4x2']['midi_count']}",
        f"- joint candidate MIDI: {stage2_summary['joint_16']['midi_count']}",
        f"- MIDI non-pedal equality: {sum(bool(row['all_nonpedal_midi_exact']) for row in equality_rows)}/46 PASS",
        f"- Human notes after official tokenization: {evaluation['human_notes']}",
        f"- Original PT notes after strict roundtrip: {evaluation['candidate_notes']['original_pt']}",
        f"- independent notes after strict roundtrip: {evaluation['candidate_notes']['independent_4x2']}",
        f"- joint notes after strict roundtrip: {evaluation['candidate_notes']['joint_16']}",
        f"- MIDI access counts: `{json.dumps(guard.counts, sort_keys=True)}`",
        "",
        "이 결과는 final_lock_v1 확정 이후 수행한 locked ASAP test final evaluation이며, "
        "test 결과를 사용한 parameter/checkpoint/inference 변경은 수행하지 않았다.",
        "",
        "## Stop point",
        "",
        "Final test evaluation과 locked descriptive audit만 수행했다. 추가 tuning, calibration, "
        "rerun, checkpoint reselection은 수행하지 않았다.",
        "",
    ]
    return "\n".join(lines)


def _artifact_integrity(output_root: Path) -> list[dict[str, Any]]:
    for mutable_name in ("final_test.log", "run_status.json"):
        mutable = output_root / mutable_name
        if not mutable.is_file() or mutable.stat().st_size <= 0:
            raise RuntimeError(f"missing/empty runtime artifact: {mutable}")
    required = [
        "test_score_manifest.csv",
        "stage1_cache_manifest.csv",
        "global_joint16_comparison.csv",
        "piece_level_metrics.csv",
        "midi_nonpedal_equality.csv",
        "final_test_metrics.json",
        "FINAL_TEST_REPORT.md",
    ]
    paths = [output_root / name for name in required]
    paths += sorted((output_root / "stage1_cache").glob("*/*"))
    paths += sorted((output_root / "candidate_midis/independent_4x2").glob("*.mid"))
    paths += sorted((output_root / "candidate_midis/joint_16").glob("*.mid"))
    if len(list((output_root / "stage1_cache").glob("*/generated_ids_int64.npy"))) != 23:
        raise RuntimeError("artifact check: Stage 1 ID count is not 23")
    if len(list((output_root / "stage1_cache").glob("*/original_pt.mid"))) != 23:
        raise RuntimeError("artifact check: Original PT MIDI count is not 23")
    if len(list((output_root / "stage1_cache").glob("*/inference_metadata.json"))) != 23:
        raise RuntimeError("artifact check: Stage 1 metadata count is not 23")
    if len(_read_csv(output_root / "test_score_manifest.csv")) != 23:
        raise RuntimeError("artifact check: score manifest row count is not 23")
    if len(_read_csv(output_root / "stage1_cache_manifest.csv")) != 23:
        raise RuntimeError("artifact check: cache manifest row count is not 23")
    if len(_read_csv(output_root / "global_joint16_comparison.csv")) != 16:
        raise RuntimeError("artifact check: global distribution row count is not 16")
    if len(_read_csv(output_root / "piece_level_metrics.csv")) != 23:
        raise RuntimeError("artifact check: piece metric row count is not 23")
    equality = _read_csv(output_root / "midi_nonpedal_equality.csv")
    if len(equality) != 46 or not all(row["all_nonpedal_midi_exact"] == "True" for row in equality):
        raise RuntimeError("artifact check: MIDI equality is not 46/46")
    rows = []
    for path in paths:
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"missing/empty final artifact: {path}")
        rows.append(
            {
                "project_relative_path": str(path.relative_to(PROJECT_ROOT)),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    for row in rows:
        if _sha256(_resolve(row["project_relative_path"])) != row["sha256"]:
            raise RuntimeError(f"artifact SHA verification failed: {row['project_relative_path']}")
    return rows


def run(
    output_root: Path,
    logger: logging.Logger,
    state: dict[str, Any],
    *,
    resume_after_stage1_failure: bool = False,
) -> dict[str, Any]:
    def stage(name: str, **extra: Any) -> None:
        state.update({"status": "running", "current_stage": name, "updated_at": _utc_now(), **extra})
        _atomic_json(output_root / "run_status.json", state)
        logger.info("STAGE %s", name)

    stage("lock_and_static_provenance_verification")
    context = _load_locked_context()
    scores, test_rows = _select_test_scores(context)
    score_fields = (
        "piece_id", "composer", "title", "test_performance_count",
        "selected_score_path", "selected_score_absolute_path",
        "selected_score_support", "score_candidate_count",
        "alternative_candidate_score_paths", "selected_score_sha256",
        "selection_rule",
    )
    cache_root = output_root / "stage1_cache"
    candidate_root = output_root / "candidate_midis"
    if resume_after_stage1_failure:
        existing_scores = _read_csv(output_root / "test_score_manifest.csv")
        if len(existing_scores) != 23:
            raise RuntimeError("resume refused: existing test score manifest is not 23 rows")
        expected_by_piece = {row["piece_id"]: row for row in scores}
        for existing in existing_scores:
            expected = expected_by_piece.get(existing["piece_id"])
            if expected is None:
                raise RuntimeError("resume refused: unexpected score-manifest piece")
            for key in (
                "selected_score_path",
                "selected_score_sha256",
                "selected_score_support",
                "test_performance_count",
                "selection_rule",
            ):
                if str(existing[key]) != str(expected[key]):
                    raise RuntimeError(
                        f"resume refused: score manifest provenance mismatch: "
                        f"{existing['piece_id']}/{key}"
                    )
        if not cache_root.is_dir() or not candidate_root.is_dir():
            raise RuntimeError("resume refused: expected cache/candidate directories are absent")
        for architecture in ARCHITECTURES:
            architecture_root = candidate_root / architecture
            if not architecture_root.is_dir() or any(architecture_root.iterdir()):
                raise RuntimeError(
                    f"resume refused: Stage 2 output is not pristine: {architecture}"
                )
        if (output_root / "midi_nonpedal_equality.csv").exists():
            raise RuntimeError("resume refused: MIDI equality output already exists")
    else:
        _write_csv(output_root / "test_score_manifest.csv", scores, score_fields)
        cache_root.mkdir()
        for architecture in ARCHITECTURES:
            (candidate_root / architecture).mkdir(parents=True, exist_ok=False)
    paths_by_piece = {row["piece_id"]: _cache_paths(cache_root, row["piece_id"]) for row in scores}
    candidate_paths = [
        candidate_root / architecture / f"{row['piece_id']}.mid"
        for architecture in ARCHITECTURES
        for row in scores
    ]
    guard = FinalTestMidiGuard(
        [ASAP_ROOT / row["performance_path"] for row in test_rows],
        [Path(row["selected_score_absolute_path"]) for row in scores],
        [paths_by_piece[row["piece_id"]]["midi"] for row in scores],
        candidate_paths,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("locked final test requires CUDA")
    device = torch.device("cuda:0")
    expected_uuid = context["validation_metrics"]["inference"]["device_uuid"]
    actual_uuid = str(torch.cuda.get_device_properties(device).uuid)
    if actual_uuid != expected_uuid:
        raise RuntimeError(f"GPU binding mismatch: expected {expected_uuid}, got {actual_uuid}")
    if resume_after_stage1_failure:
        stage("stage1_cache_resume_verification")
        stage1_records = _read_csv(output_root / "stage1_cache_manifest.csv")
        _verify_stage1_cache(stage1_records)
        if {row["piece_id"] for row in stage1_records} != set(paths_by_piece):
            raise RuntimeError("resume refused: Stage 1 cache piece set mismatch")
        state.update(
            {
                "initial_verification": "passed",
                "verified_lock_sha256": context["lock_sha256"],
                "verified_checkpoint_hashes": {
                    architecture: context["candidates"][architecture]["checkpoint_sha256"]
                    for architecture in ARCHITECTURES
                },
                "gpu": {
                    "device": "cuda:0",
                    "name": torch.cuda.get_device_name(device),
                    "uuid": actual_uuid,
                },
                "test_piece_count": 23,
                "test_human_performance_count": 104,
                "stage1_generated_outputs": 23,
                "stage1_neural_inference_reused": 23,
                "stage1_neural_inference_regenerated": 0,
            }
        )
        _atomic_json(output_root / "run_status.json", state)
        logger.info(
            "STAGE1_CACHE_RESUME_VERIFIED pieces=23 ids/midi/metadata_hashes=PASS "
            "stage1_regenerated=0"
        )
        logger.info("INITIAL_READY resume verified; entering independent Stage 2 inference")
    else:
        _seed_everything(42)
        model = PianoT5Gemma.from_pretrained(
            str(context["stage1_checkpoint"]),
            torch_dtype=torch.float32,
            attn_implementation="eager",
        ).to(device).eval()
        generation = model.generation_config
        if (
            int(generation.top_k),
            int(generation.num_beams),
            float(generation.repetition_penalty),
        ) != (50, 1, 1.0):
            raise RuntimeError("official generation defaults differ from validation provenance")
        stage(
            "stage1_original_pt_inference",
            initial_verification="passed",
            verified_lock_sha256=context["lock_sha256"],
            verified_checkpoint_hashes={
                architecture: context["candidates"][architecture]["checkpoint_sha256"]
                for architecture in ARCHITECTURES
            },
            gpu={
                "device": "cuda:0",
                "name": torch.cuda.get_device_name(device),
                "uuid": actual_uuid,
            },
            test_piece_count=23,
            test_human_performance_count=104,
        )
        logger.info(
            "INITIAL_VERIFICATION_PASS lock/checkpoints/config/split/sources/gpu "
            "test_scope=23/104"
        )

        stage1_records = []
        for index, score in enumerate(scores, start=1):
            if index == 1:
                logger.info(
                    "FIRST_STAGE1_INFERENCE_STARTED piece=1/23 piece_id=%s",
                    score["piece_id"],
                )
                logger.info(
                    "INITIAL_READY final-test runner entered first Stage 1 neural inference"
                )
            record = _stage1_generate_one(
                model, score, paths_by_piece[score["piece_id"]], context, guard
            )
            stage1_records.append(record)
            logger.info(
                "STAGE1_COMPLETE piece=%d/23 piece_id=%s notes=%d runtime_seconds=%.3f",
                index,
                record["piece_id"],
                record["generated_notes"],
                record["runtime_seconds"],
            )
        del model
        gc.collect()
        torch.cuda.empty_cache()
        _verify_stage1_cache(stage1_records)
        cache_fields = tuple(stage1_records[0].keys())
        _write_csv(output_root / "stage1_cache_manifest.csv", stage1_records, cache_fields)

    equality_rows: list[dict[str, Any]] = []
    score_by_piece = {row["piece_id"]: row for row in scores}
    stage2_summary = {}
    for architecture in ARCHITECTURES:
        stage(f"stage2_inference_{architecture}", stage1_generated_outputs=23)
        stage2_summary[architecture] = _render_stage2(
            architecture,
            stage1_records,
            score_by_piece,
            context,
            output_root,
            guard,
            equality_rows,
            logger,
        )
    if len(equality_rows) != 46 or not all(row["all_nonpedal_midi_exact"] for row in equality_rows):
        raise RuntimeError("MIDI non-pedal equality is not 46/46")

    stage("strict_midi_roundtrip_global_and_piece_evaluation")
    evaluation = _evaluate(
        stage1_records, test_rows, output_root, context, guard, logger
    )
    final_metrics = {
        "status": "locked_final_test_evaluated",
        "final_lock_path": str(LOCK_RELATIVE),
        "final_lock_sha256": context["lock_sha256"],
        "test_scope": {"pieces": 23, "human_performances": 104},
        "score_manifest_sha256": _sha256(output_root / "test_score_manifest.csv"),
        "stage1_generated_outputs": 23,
        "candidate_outputs": {"independent_4x2": 23, "joint_16": 23},
        "midi_nonpedal_equality": {"passed": 46, "expected": 46},
        "binary_threshold": 64,
        "stage2_decoding": "deterministic argmax",
        "global_metrics": evaluation["global_metrics"],
        "changes_vs_original_pt": evaluation["changes_vs_original_pt"],
        "piece_level_summary": evaluation["piece_summary"],
        "steady_transition_mass": evaluation["masses"],
        "note_counts": {
            "human": evaluation["human_notes"],
            **evaluation["candidate_notes"],
        },
        "histograms": {
            "human": evaluation["human_histogram"],
            **evaluation["candidate_histograms"],
        },
        "strict_evaluator_provenance": STRICT_EVALUATOR_PROVENANCE,
        "locked_candidates": {
            architecture: {
                "best_epoch": context["candidates"][architecture]["best_epoch"],
                "checkpoint_path": context["candidates"][architecture][
                    "checkpoint_project_relative_path"
                ],
                "checkpoint_sha256": context["candidates"][architecture]["checkpoint_sha256"],
            }
            for architecture in ARCHITECTURES
        },
        "stage1_provenance": {
            "checkpoint_path": str(STAGE1_CHECKPOINT_RELATIVE),
            "checkpoint_sha256": context["stage1_model_sha256"],
            "seed": 42,
            "seed_reset_per_piece": True,
            "do_sample": True,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 50,
            "num_beams": 1,
            "repetition_penalty": 1.0,
            "max_context_length": 4096,
            "overlap_ratio": 0.5,
        },
        "asap_split": {"path": str(SPLIT_RELATIVE), "sha256": context["split_sha256"]},
        "test_result_used_for_any_change": False,
    }
    _atomic_json(output_root / "final_test_metrics.json", final_metrics)
    report = _report(
        context,
        evaluation,
        stage1_records,
        stage2_summary,
        equality_rows,
        final_metrics["score_manifest_sha256"],
        guard,
    )
    temporary_report = output_root / "FINAL_TEST_REPORT.md.tmp"
    temporary_report.write_text(report, encoding="utf-8")
    temporary_report.replace(output_root / "FINAL_TEST_REPORT.md")

    stage("output_artifact_integrity_verification")
    artifact_rows = _artifact_integrity(output_root)
    _write_csv(
        output_root / "artifact_sha256_manifest.csv",
        artifact_rows,
        ("project_relative_path", "size_bytes", "sha256"),
    )
    for row in _read_csv(output_root / "artifact_sha256_manifest.csv"):
        if _sha256(_resolve(row["project_relative_path"])) != row["sha256"]:
            raise RuntimeError("artifact manifest post-write verification failed")
    return {
        "context": context,
        "stage1_records": stage1_records,
        "stage2_summary": stage2_summary,
        "equality_rows": equality_rows,
        "evaluation": evaluation,
        "artifact_count": len(artifact_rows),
        "guard_counts": guard.counts,
    }


def _configure_logger(output_root: Path) -> logging.Logger:
    logger = logging.getLogger("stage2_binary_final_test")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s")
    formatter.converter = time.gmtime
    file_handler = logging.FileHandler(output_root / "final_test.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def _static_preflight() -> dict[str, Any]:
    context = _load_locked_context()
    scores, test_rows = _select_test_scores(context)
    return {
        "status": "pass",
        "lock_sha256": context["lock_sha256"],
        "split_sha256": context["split_sha256"],
        "test_pieces": len(scores),
        "test_human_performances": len(test_rows),
        "selected_scores": len(scores),
        "checkpoint_sha256": {
            architecture: context["candidates"][architecture]["checkpoint_sha256"]
            for architecture in ARCHITECTURES
        },
        "midi_files_opened": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(OUTPUT_RELATIVE))
    parser.add_argument("--static-preflight", action="store_true")
    parser.add_argument("--resume-after-stage1-failure", action="store_true")
    args = parser.parse_args()
    if args.static_preflight:
        print(json.dumps(_static_preflight(), indent=2, sort_keys=True), flush=True)
        return
    output_root = _resolve(args.output_root)
    if output_root != _resolve(OUTPUT_RELATIVE):
        raise RuntimeError("locked final-test output root cannot be changed")
    if args.resume_after_stage1_failure:
        if not output_root.is_dir():
            raise RuntimeError("resume requested but final-test output root is absent")
        prior = _read_json(output_root / "run_status.json")
        if (
            prior.get("status") != "failed"
            or prior.get("failed_stage") != "stage2_inference_independent_4x2"
            or int(prior.get("stage1_generated_outputs", 0)) != 23
        ):
            raise RuntimeError("resume is allowed only for the verified post-Stage-1 failure")
        if (output_root / "FINAL_TEST_REPORT.md").exists():
            raise RuntimeError("resume refused because a final report already exists")
    else:
        output_root.mkdir(parents=True, exist_ok=False)
    logger = _configure_logger(output_root)
    state: dict[str, Any] = {
        "status": "running",
        "started_at": (
            prior["started_at"] if args.resume_after_stage1_failure else _utc_now()
        ),
        "updated_at": _utc_now(),
        "current_stage": "runner_initialization",
        "runner": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "output_root": str(OUTPUT_RELATIVE),
        "process_id": os.getpid(),
        "resume": bool(args.resume_after_stage1_failure),
    }
    if args.resume_after_stage1_failure:
        state["resumed_at"] = _utc_now()
        state["resumed_from_failure"] = {
            "failed_stage": prior["failed_stage"],
            "failed_at": prior["failed_at"],
            "error_type": prior["error_type"],
            "error_summary": prior["error_summary"],
        }
    _atomic_json(output_root / "run_status.json", state)
    logger.info("FINAL_TEST_RUNNER_STARTED pid=%d output=%s", os.getpid(), output_root)
    try:
        result = run(
            output_root,
            logger,
            state,
            resume_after_stage1_failure=args.resume_after_stage1_failure,
        )
        completed = {
            **state,
            "status": "completed",
            "current_stage": "completed",
            "completed_at": _utc_now(),
            "updated_at": _utc_now(),
            "stage1_generated_outputs": len(result["stage1_records"]),
            "candidate_midi_outputs": {
                architecture: result["stage2_summary"][architecture]["midi_count"]
                for architecture in ARCHITECTURES
            },
            "midi_nonpedal_equality_passed": sum(
                bool(row["all_nonpedal_midi_exact"]) for row in result["equality_rows"]
            ),
            "artifact_integrity_verified": True,
            "artifact_count": result["artifact_count"],
            "test_result_used_for_any_change": False,
        }
        _atomic_json(output_root / "run_status.json", completed)
        logger.info("FINAL_TEST_RUNNER_COMPLETED artifacts=%d", result["artifact_count"])
    except BaseException as error:
        failed_stage = state.get("current_stage", "unknown")
        logger.error(
            "FINAL_TEST_RUNNER_FAILED stage=%s error=%s\n%s",
            failed_stage,
            error,
            traceback.format_exc(),
        )
        failed = {
            **state,
            "status": "failed",
            "failed_stage": failed_stage,
            "failed_at": _utc_now(),
            "updated_at": _utc_now(),
            "error_type": type(error).__name__,
            "error_summary": str(error),
            "traceback_log": str(OUTPUT_RELATIVE / "final_test.log"),
        }
        _atomic_json(output_root / "run_status.json", failed)
        raise
    finally:
        for handler in logger.handlers:
            handler.flush()
        try:
            _fix_ownership(output_root)
        except Exception:
            logger.exception("failed to normalize output ownership")


if __name__ == "__main__":
    main()
