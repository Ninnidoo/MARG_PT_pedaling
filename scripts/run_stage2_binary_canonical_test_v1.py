#!/usr/bin/env python3
"""Run the locked canonical-v1 binary Stage-2 ASAP test evaluation."""

from __future__ import annotations

import csv
import gc
import json
import logging
import os
import statistics
import sys
import tempfile
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import mido
import numpy as np
import torch
from miditoolkit import MidiFile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_stage2_binary_metric_fidelity import _strict_official_similarity  # noqa: E402
from src.stage2_binary.canonical_stage1 import (  # noqa: E402
    assert_strict_non_cc64_equality,
    cc64_schedule,
    sha256_file,
    signature_sha256,
    transplant_cc64_only,
)
from src.stage2_binary.strict_midi_validation import (  # noqa: E402
    PianoT5GemmaConfig,
    ids_to_midi,
    map_midi,
    midi_to_ids,
)
from src.stage2_binary.validation_evaluator import (  # noqa: E402
    JOINT_PATTERNS,
    infer_cached_binary_pedals,
    joint16_histogram_from_ids,
    load_binary_stage2_checkpoint,
)


EXPERIMENT_ROOT = ROOT / "analysis/stage2_binary_canonical_v1"
OUTPUT = EXPERIMENT_ROOT / "test_eval_v0"
MAPPING_SOURCE = (
    ROOT
    / "analysis/stage2_binary_v0/canonical_stage1_pipeline_v0/canonical_stage1_manifest.csv"
)
SPLIT_PATH = ROOT / "analysis/stage2_encoder_only_v0/asap_split.csv"
ASAP_ROOT = Path("/workspace/public/ASAP/asap-dataset-v1.1")
HUMAN_PROVENANCE = ROOT / "analysis/stage2_binary_v0/test_eval_v0/final_test_metrics.json"
TRAINING_REPORT = EXPERIMENT_ROOT / "CANONICAL_BINARY_TRAINING_REPORT.md"
VALIDATION_BASELINE = EXPERIMENT_ROOT / "canonical_validation_baseline.json"
ENCODER_CHECKPOINT = ROOT / "checkpoints/pianist_transformer"
OFFICIAL_EVALUATOR = ROOT / "third_party/PianistTransformer/src/evaluate/evaluate.py"
OFFICIAL_TOKENIZER = ROOT / "third_party/PianistTransformer/src/utils/midi.py"
ARCHITECTURES = ("independent_4x2", "joint_16")
EXPECTED = {
    "independent_4x2": {
        "best_epoch": 2,
        "checkpoint_sha256": "9fc60f9b3d91942afaf56f0553fea3f2c39509f32c9965d6236de178077d65d4",
        "validation_js": 0.15008283937733968,
        "validation_intersection": 0.829325144703809,
    },
    "joint_16": {
        "best_epoch": 2,
        "checkpoint_sha256": "4d4d009b0ce9dae44c88dbb1c6e80883295146bbcd6daca3421954ff5ebdacb6",
        "validation_js": 0.20244233677774107,
        "validation_intersection": 0.7805984692558221,
    },
}
EXPECTED_HASHES = {
    "asap_split": "d1fe379eb123ff7abca93773296f735f40a215dcd6e6363b55756383708c965b",
    "mapping_source": "78d6a63421ec26b95dbc3dcd9df7c4c2dc4a469e118bf75148b5a747f5a17031",
    "human_provenance": "36db39160dee3924139b97f0198587e15965c0f50184b6a5d925b6f1c970e8f3",
    "official_evaluator": "42a6066496c570b974c3288eb3e8ac3b83e91c00add94da2a19b59c575a5e02a",
    "official_tokenizer": "2fb37eaca3d6e4f4a775eb8a59379e7e09651fda36c8146cf0067cde8ad78633",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def probability(histogram: Sequence[int]) -> np.ndarray:
    values = np.asarray(histogram, dtype=np.float64)
    if values.shape != (16,) or np.any(values < 0) or values.sum() <= 0:
        raise ValueError("invalid 16-pattern histogram")
    return values / values.sum()


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("canonical_binary_test_v1")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.FileHandler(OUTPUT / "test_eval.log"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def checkpoint_lock() -> dict[str, Any]:
    report = TRAINING_REPORT.read_text(encoding="utf-8")
    candidates: dict[str, Any] = {}
    for architecture in ARCHITECTURES:
        expected = EXPECTED[architecture]
        train_root = EXPERIMENT_ROOT / "train_v0" / architecture
        checkpoint = train_root / "best.pt"
        manifest_path = train_root / "checkpoint_manifest.csv"
        config_path = train_root / "config.json"
        rows = read_csv(manifest_path)
        best = [row for row in rows if row["kind"] == "best"]
        if len(best) != 1:
            raise RuntimeError(f"{architecture}: checkpoint manifest has no unique best row")
        row = best[0]
        if int(row["epoch"]) != expected["best_epoch"]:
            raise RuntimeError(f"{architecture}: validation-selected epoch changed")
        if row["checkpoint_sha256"] != expected["checkpoint_sha256"]:
            raise RuntimeError(f"{architecture}: manifest checkpoint hash changed")
        if abs(float(row["canonical_strict_js_distance"]) - expected["validation_js"]) > 1e-15:
            raise RuntimeError(f"{architecture}: manifest validation JS changed")
        actual_hash = sha256_file(checkpoint)
        if actual_hash != expected["checkpoint_sha256"]:
            raise RuntimeError(f"{architecture}: actual best.pt hash mismatch")
        if expected["checkpoint_sha256"] not in report:
            raise RuntimeError(f"{architecture}: training report does not contain locked hash")
        candidates[architecture] = {
            "architecture": architecture,
            "best_epoch": expected["best_epoch"],
            "checkpoint_path": str(checkpoint),
            "checkpoint_project_relative_path": str(checkpoint.relative_to(ROOT)),
            "checkpoint_sha256": actual_hash,
            "checkpoint_manifest_path": str(manifest_path),
            "checkpoint_manifest_sha256": sha256_file(manifest_path),
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "validation_strict_js": expected["validation_js"],
            "validation_intersection": expected["validation_intersection"],
        }
    baseline = json.loads(VALIDATION_BASELINE.read_text(encoding="utf-8"))
    original_validation = baseline["metrics"]
    return {
        "experiment_id": "stage2_binary_canonical_v1",
        "lock_created_at": now(),
        "selection_scope": "validation-only before canonical-v1 test evaluation",
        "selection_metric": "lowest canonical strict official PT base-2 JS distance",
        "checkpoints": candidates,
        "validation_original_pt": {
            "js_distance_base2": float(original_validation["js_distance_base2"]),
            "histogram_intersection": float(
                original_validation["histogram_intersection_official"]
            ),
        },
        "stage2_inference": {
            "window_notes": 512,
            "stride_notes": 256,
            "overlap": "logit averaging",
            "decoding": "deterministic argmax",
            "binary_threshold": 64,
            "decoded_raw_values": [0, 127],
            "pedal_input": "four MASK tokens; no pedal leakage",
        },
        "forbidden_after_lock": [
            "epoch or checkpoint reselection",
            "architecture selection from test result",
            "threshold change",
            "calibration, smoothing, or post-processing",
            "Stage 1 neural inference",
            "training",
        ],
        "training_report_path": str(TRAINING_REPORT),
        "training_report_sha256": sha256_file(TRAINING_REPORT),
        "validation_baseline_path": str(VALIDATION_BASELINE),
        "validation_baseline_sha256": sha256_file(VALIDATION_BASELINE),
    }


def canonical_manifest() -> tuple[list[dict[str, Any]], dict[str, str]]:
    if sha256_file(MAPPING_SOURCE) != EXPECTED_HASHES["mapping_source"]:
        raise RuntimeError("existing canonical piece-to-num mapping artifact changed")
    source = read_csv(MAPPING_SOURCE)
    if len(source) != 23 or sorted(int(row["num"]) for row in source) != list(range(23)):
        raise RuntimeError("canonical mapping is not exactly nums 0..22")
    if len({row["piece_id"] for row in source}) != 23:
        raise RuntimeError("canonical mapping piece IDs are not unique")
    rows: list[dict[str, Any]] = []
    protected: dict[str, str] = {}
    for row in sorted(source, key=lambda value: int(value["num"])):
        canonical = Path(row["canonical_original_pt_midi_path"])
        score = Path(row["source_score_path"])
        if not canonical.is_file() or not score.is_file():
            raise FileNotFoundError(canonical if not canonical.is_file() else score)
        canonical_hash = sha256_file(canonical)
        if canonical_hash != row["canonical_midi_sha256"]:
            raise RuntimeError(f"canonical MIDI hash changed: num={row['num']}")
        if sha256_file(score) != row["source_score_sha256"]:
            raise RuntimeError(f"canonical score hash changed: num={row['num']}")
        if signature_sha256(canonical) != row["canonical_non_cc64_signature_sha256"]:
            raise RuntimeError(f"canonical non-CC64 signature changed: num={row['num']}")
        midi = mido.MidiFile(str(canonical), clip=False)
        note_count = sum(
            1
            for track in midi.tracks
            for message in track
            if not message.is_meta and message.type == "note_on" and message.velocity > 0
        )
        if note_count != int(row["note_count"]):
            raise RuntimeError(f"canonical note count changed: num={row['num']}")
        rows.append(
            {
                "num": int(row["num"]),
                "piece_id": row["piece_id"],
                "composer": row["composer"],
                "title": row["title"],
                "score_path": str(score),
                "score_sha256": row["source_score_sha256"],
                "canonical_original_pt_midi_path": str(canonical),
                "canonical_midi_sha256": canonical_hash,
                "canonical_non_cc64_signature_sha256": row[
                    "canonical_non_cc64_signature_sha256"
                ],
                "note_count": note_count,
                "duration_seconds": float(midi.length),
                "mapping_method": row["mapping_method"],
                "mapping_source_path": str(MAPPING_SOURCE),
                "read_only_source_of_truth": True,
            }
        )
        protected[str(canonical)] = canonical_hash
    return rows, protected


def test_rows(manifest: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    if sha256_file(SPLIT_PATH) != EXPECTED_HASHES["asap_split"]:
        raise RuntimeError("ASAP split hash changed")
    rows = [row for row in read_csv(SPLIT_PATH) if row["split"] == "test"]
    pieces = {row["piece_id"] for row in rows}
    manifest_pieces = {str(row["piece_id"]) for row in manifest}
    if len(rows) != 104 or len(pieces) != 23 or pieces != manifest_pieces:
        raise RuntimeError("ASAP test split is not the mapped 104 performances / 23 pieces")
    counts = Counter(row["piece_id"] for row in rows)
    if any(count <= 0 for count in counts.values()):
        raise RuntimeError("ASAP test piece has no human performance")
    for row in rows:
        path = ASAP_ROOT / row["performance_path"]
        if not path.is_file():
            raise FileNotFoundError(path)
    return rows


def verify_fixed_provenance() -> dict[str, str]:
    paths = {
        "asap_split": SPLIT_PATH,
        "mapping_source": MAPPING_SOURCE,
        "human_provenance": HUMAN_PROVENANCE,
        "official_evaluator": OFFICIAL_EVALUATOR,
        "official_tokenizer": OFFICIAL_TOKENIZER,
    }
    actual = {name: sha256_file(path) for name, path in paths.items()}
    for name, expected in EXPECTED_HASHES.items():
        if actual[name] != expected:
            raise RuntimeError(f"pinned provenance changed: {name}")
    return actual


def infer_candidates(
    manifest: Sequence[Mapping[str, Any]],
    lock: Mapping[str, Any],
    canonical_ids: Mapping[int, np.ndarray],
    protected_canonical: Mapping[str, str],
    logger: logging.Logger,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    equality_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    config = PianoT5GemmaConfig()
    for architecture in ARCHITECTURES:
        locked = lock["checkpoints"][architecture]
        checkpoint = Path(locked["checkpoint_path"])
        if sha256_file(checkpoint) != locked["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint changed after lock: {architecture}")
        model = load_binary_stage2_checkpoint(
            checkpoint,
            architecture=architecture,
            encoder_checkpoint=ENCODER_CHECKPOINT,
            device=device,
        )
        logger.info("MODEL_LOAD_PASS architecture=%s device=%s", architecture, device)
        candidate_root = OUTPUT / "candidate_midis" / architecture
        candidate_root.mkdir(parents=True, exist_ok=False)
        notes_total = 0
        windows_total = 0
        donor_total = 0
        transplanted_total = 0
        discarded_total = 0
        with tempfile.TemporaryDirectory(prefix=f"{architecture}_donor_", dir=OUTPUT) as temp:
            temp_root = Path(temp)
            for index, row in enumerate(manifest, start=1):
                number = int(row["num"])
                piece_id = str(row["piece_id"])
                canonical_path = Path(row["canonical_original_pt_midi_path"])
                predicted_ids, details = infer_cached_binary_pedals(
                    model,
                    canonical_ids[number],
                    architecture=architecture,
                    device=device,
                    window_notes=512,
                    stride_notes=256,
                )
                if not details["non_pedal_tokens_preserved"]:
                    raise AssertionError(f"pre-render token invariant failed: {piece_id}")
                donor_path = temp_root / f"{number}_donor.mid"
                candidate_path = candidate_root / f"{number}_{architecture}.mid"
                performance = ids_to_midi(
                    config,
                    predicted_ids,
                    ref=canonical_ids[number].tolist(),
                )
                donor = map_midi(MidiFile(str(canonical_path)), performance)
                donor.dump(str(donor_path))
                transplant = transplant_cc64_only(canonical_path, donor_path, candidate_path)
                strict = assert_strict_non_cc64_equality(canonical_path, candidate_path)
                donor_count = int(transplant["donor_cc64_events"])
                transplanted_count = int(strict["candidate_cc64_events"])
                discarded = int(transplant["donor_cc64_discarded_after_canonical_eot"])
                if donor_count - discarded != transplanted_count:
                    raise AssertionError(f"CC64 transplant accounting failed: {architecture}/{piece_id}")
                equality_rows.append(
                    {
                        "architecture": architecture,
                        "num": number,
                        "piece_id": piece_id,
                        "canonical_midi_path": str(canonical_path),
                        "candidate_midi_path": str(candidate_path),
                        "canonical_sha256": row["canonical_midi_sha256"],
                        "candidate_sha256": sha256_file(candidate_path),
                        "midi_type_exact": strict["midi_type_exact"],
                        "ticks_per_beat_exact": strict["ticks_per_beat_exact"],
                        "track_count_exact": strict["track_count_exact"],
                        "all_ordered_non_cc64_events_exact": strict[
                            "all_ordered_non_cc64_events_exact"
                        ],
                        "canonical_cc64_events": strict["canonical_cc64_events"],
                        "candidate_cc64_events": strict["candidate_cc64_events"],
                        "cc64_changed": strict["cc64_changed"],
                        "status": "PASS",
                    }
                )
                audit_rows.append(
                    {
                        "architecture": architecture,
                        "num": number,
                        "piece_id": piece_id,
                        "donor_cc64_events": donor_count,
                        "transplanted_cc64_events": transplanted_count,
                        "discarded_after_canonical_eot": discarded,
                        "accounting_exact": donor_count - discarded == transplanted_count,
                        "conversion_path": "predicted Pedal1--4 0/127 -> ids_to_midi -> map_midi donor -> CC64-only transplant",
                    }
                )
                donor_total += donor_count
                transplanted_total += transplanted_count
                discarded_total += discarded
                notes_total += int(details["notes"])
                windows_total += int(details["windows"])
                logger.info(
                    "CANDIDATE_PASS architecture=%s piece=%d/23 num=%d notes=%d windows=%d donor_cc64=%d discarded=%d",
                    architecture,
                    index,
                    number,
                    details["notes"],
                    details["windows"],
                    donor_count,
                    discarded,
                )
        candidates = list(candidate_root.glob("*.mid"))
        if len(candidates) != 23:
            raise RuntimeError(f"candidate count is not 23: {architecture}")
        summaries[architecture] = {
            "candidate_midis": len(candidates),
            "input_notes": notes_total,
            "inference_windows": windows_total,
            "total_donor_cc64_events": donor_total,
            "total_transplanted_cc64_events": transplanted_total,
            "total_discarded_after_canonical_eot": discarded_total,
        }
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if len(equality_rows) != 46 or not all(row["status"] == "PASS" for row in equality_rows):
        raise AssertionError("strict non-CC64 equality did not pass 46/46")
    write_csv(OUTPUT / "midi_nonpedal_equality.csv", equality_rows)
    write_csv(OUTPUT / "cc64_transplant_audit.csv", audit_rows)
    for path, digest in protected_canonical.items():
        if sha256_file(path) != digest:
            raise AssertionError(f"canonical MIDI changed during inference: {path}")
    return equality_rows, audit_rows, {"device": str(device), **summaries}


def evaluate_metrics(
    manifest: Sequence[Mapping[str, Any]],
    split_rows: Sequence[Mapping[str, str]],
    canonical_ids: Mapping[int, np.ndarray],
    logger: logging.Logger,
) -> dict[str, Any]:
    config = PianoT5GemmaConfig()
    old = json.loads(HUMAN_PROVENANCE.read_text(encoding="utf-8"))
    expected_human = np.asarray(old["histograms"]["human"], dtype=np.int64)
    if expected_human.shape != (16,) or int(expected_human.sum()) != 331_576:
        raise RuntimeError("verified historical Human histogram inventory changed")
    human_global = np.zeros(16, dtype=np.int64)
    human_piece = {
        str(row["piece_id"]): np.zeros(16, dtype=np.int64) for row in manifest
    }
    human_counts: Counter[str] = Counter()
    for index, row in enumerate(split_rows, start=1):
        path = ASAP_ROOT / row["performance_path"]
        ids = midi_to_ids(config, MidiFile(str(path)))
        histogram = joint16_histogram_from_ids(ids)
        human_global += histogram
        human_piece[row["piece_id"]] += histogram
        human_counts[row["piece_id"]] += 1
        logger.info("HUMAN_REFERENCE_PASS performance=%d/104 piece_id=%s", index, row["piece_id"])
    if not np.array_equal(human_global, expected_human):
        raise RuntimeError("recomputed 104-performance Human histogram differs from verified provenance")

    labels = ("original_pt", "independent_4x2", "joint_16")
    global_hist = {label: np.zeros(16, dtype=np.int64) for label in labels}
    piece_hist = {
        label: {str(row["piece_id"]): np.zeros(16, dtype=np.int64) for row in manifest}
        for label in labels
    }
    for row in manifest:
        number = int(row["num"])
        piece_id = str(row["piece_id"])
        paths = {
            "independent_4x2": OUTPUT
            / "candidate_midis/independent_4x2"
            / f"{number}_independent_4x2.mid",
            "joint_16": OUTPUT / "candidate_midis/joint_16" / f"{number}_joint_16.mid",
        }
        ids_by_label = {
            "original_pt": canonical_ids[number],
            **{
                label: np.asarray(midi_to_ids(config, MidiFile(str(path))), dtype=np.int64)
                for label, path in paths.items()
            },
        }
        for label, ids in ids_by_label.items():
            histogram = joint16_histogram_from_ids(ids)
            global_hist[label] += histogram
            piece_hist[label][piece_id] += histogram

    metrics = {
        label: _strict_official_similarity(human_global, global_hist[label])
        for label in labels
    }
    probs = {
        "human": probability(human_global),
        **{label: probability(global_hist[label]) for label in labels},
    }
    global_rows: list[dict[str, Any]] = []
    for joint_id, pattern in enumerate(JOINT_PATTERNS):
        row: dict[str, Any] = {
            "joint_id": joint_id,
            "pattern": pattern,
            "human_count": int(human_global[joint_id]),
            "human_probability": float(probs["human"][joint_id]),
        }
        for label, prefix in (
            ("original_pt", "original_pt"),
            ("independent_4x2", "independent"),
            ("joint_16", "joint"),
        ):
            error = float(probs[label][joint_id] - probs["human"][joint_id])
            row[f"{prefix}_count"] = int(global_hist[label][joint_id])
            row[f"{prefix}_probability"] = float(probs[label][joint_id])
            row[f"{prefix}_signed_error_vs_human"] = error
            row[f"{prefix}_absolute_error_vs_human"] = abs(error)
        global_rows.append(row)
    write_csv(OUTPUT / "global_joint16_comparison.csv", global_rows)

    piece_rows: list[dict[str, Any]] = []
    for row in manifest:
        piece_id = str(row["piece_id"])
        output: dict[str, Any] = {
            "num": int(row["num"]),
            "piece_id": piece_id,
            "composer": row["composer"],
            "title": row["title"],
            "human_performances": human_counts[piece_id],
            "human_notes": int(human_piece[piece_id].sum()),
        }
        for label, prefix in (
            ("original_pt", "original_pt"),
            ("independent_4x2", "independent"),
            ("joint_16", "joint"),
        ):
            metric = _strict_official_similarity(human_piece[piece_id], piece_hist[label][piece_id])
            output[f"{prefix}_candidate_notes"] = int(piece_hist[label][piece_id].sum())
            output[f"{prefix}_js_distance"] = float(metric["js_distance_base2"])
            output[f"{prefix}_intersection"] = float(metric["histogram_intersection"])
        output["independent_js_improvement_original_minus_stage2"] = (
            output["original_pt_js_distance"] - output["independent_js_distance"]
        )
        output["joint_js_improvement_original_minus_stage2"] = (
            output["original_pt_js_distance"] - output["joint_js_distance"]
        )
        piece_rows.append(output)
    write_csv(OUTPUT / "piece_level_metrics.csv", piece_rows)

    summaries: dict[str, Any] = {}
    for prefix in ("original_pt", "independent", "joint"):
        values = [float(row[f"{prefix}_js_distance"]) for row in piece_rows]
        summaries[prefix] = {
            "mean_js_distance": statistics.fmean(values),
            "median_js_distance": statistics.median(values),
        }
    summaries["independent"]["pieces_improved_vs_original_pt"] = sum(
        float(row["independent_js_improvement_original_minus_stage2"]) > 0
        for row in piece_rows
    )
    summaries["joint"]["pieces_improved_vs_original_pt"] = sum(
        float(row["joint_js_improvement_original_minus_stage2"]) > 0 for row in piece_rows
    )
    masses = {}
    for label, values in probs.items():
        steady = float(values[0] + values[15])
        masses[label] = {
            "steady_mass": steady,
            "transition_containing_mass": 1.0 - steady,
        }
    original_js = float(metrics["original_pt"]["js_distance_base2"])
    original_intersection = float(metrics["original_pt"]["histogram_intersection"])
    changes = {}
    for label in ("independent_4x2", "joint_16"):
        js = float(metrics[label]["js_distance_base2"])
        intersection = float(metrics[label]["histogram_intersection"])
        changes[label] = {
            "absolute_js_change_candidate_minus_original_pt": js - original_js,
            "relative_js_change_percent": (js - original_js) / original_js * 100.0,
            "intersection_change_candidate_minus_original_pt": intersection
            - original_intersection,
            "beats_original_pt": js < original_js,
        }
    return {
        "human_histogram": human_global.tolist(),
        "histograms": {label: values.tolist() for label, values in global_hist.items()},
        "global_strict_metrics": metrics,
        "changes_vs_original_pt": changes,
        "piece_level_summary": summaries,
        "steady_transition_mass": masses,
        "human_performances": 104,
        "human_notes": int(human_global.sum()),
        "candidate_notes": {label: int(hist.sum()) for label, hist in global_hist.items()},
        "human_histogram_reproduced_verified_artifact": True,
    }


def make_report(
    result: Mapping[str, Any],
    lock: Mapping[str, Any],
    audit_rows: Sequence[Mapping[str, Any]],
) -> str:
    metrics = result["global_strict_metrics"]
    changes = result["changes_vs_original_pt"]
    pieces = result["piece_level_summary"]
    masses = result["steady_transition_mass"]
    audit_by_arch = {
        architecture: {
            "donor": sum(int(row["donor_cc64_events"]) for row in audit_rows if row["architecture"] == architecture),
            "transplanted": sum(int(row["transplanted_cc64_events"]) for row in audit_rows if row["architecture"] == architecture),
            "discarded": sum(int(row["discarded_after_canonical_eot"]) for row in audit_rows if row["architecture"] == architecture),
            "pieces_with_discard": sum(int(row["discarded_after_canonical_eot"]) > 0 for row in audit_rows if row["architecture"] == architecture),
        }
        for architecture in ARCHITECTURES
    }
    independent_generalizes = changes["independent_4x2"]["beats_original_pt"]
    joint_failure_persists = not changes["joint_16"]["beats_original_pt"]
    lines = [
        "# Canonical-v1 Binary Stage 2 Test Report",
        "",
        "## A. Locked experiment provenance",
        "",
        "This evaluation uses only the two checkpoints selected by canonical validation strict JS. The test result did not change model, epoch, threshold, decoding, or post-processing.",
        "",
        f"- Independent 4×2: epoch 2, `{lock['checkpoints']['independent_4x2']['checkpoint_sha256']}`",
        f"- Joint 16: epoch 2, `{lock['checkpoints']['joint_16']['checkpoint_sha256']}`",
        "- Stage 2 inference: 512-note windows, stride 256, overlap-logit averaging, deterministic argmax, threshold 64, decode 0/127.",
        "- Stage 1 neural inference performed in this evaluation: **0**.",
        "",
        "## B. Canonical Stage 1 test provenance",
        "",
        "The 23 immutable `outputs/midi/{num}_original_pt.mid` files were joined to ASAP pieces through the existing exact score-SHA mapping artifact. Nums 0..22 occur exactly once; filenames were not sorted into an inferred mapping. Canonical source hashes were verified before and after evaluation.",
        "",
        "The Human reference contains 104 ASAP test performances. Official `midi_to_ids`, threshold ≥64, and Pedal1-as-MSB 16-pattern semantics reproduced the previously verified Human histogram exactly. Historical histogram artifact SHA-256: `" + EXPECTED_HASHES["human_provenance"] + "`.",
        "",
        "## C. Hard non-CC64 equality",
        "",
        "**46/46 PASS.** Every Independent and Joint candidate preserves MIDI type, ticks per beat, track layout, every ordered non-CC64 channel/meta event, and end-of-track timing exactly. Only sustain CC64 may differ.",
        "",
        "## D. CC64 transplant audit",
        "",
        "| Model | donor CC64 | transplanted CC64 | discarded after canonical EOT | pieces with discard |",
        "|---|---:|---:|---:|---:|",
    ]
    for architecture, display in (("independent_4x2", "Independent 4×2"), ("joint_16", "Joint 16")):
        item = audit_by_arch[architecture]
        lines.append(f"| {display} | {item['donor']} | {item['transplanted']} | {item['discarded']} | {item['pieces_with_discard']} |")
    lines += [
        "",
        "Predicted Pedal1--4 values followed the validated `0/127 → ids_to_midi → map_midi donor → CC64-only transplant` conversion path. No new pedal timing algorithm was introduced.",
        "",
        "## E. Global headline result",
        "",
        "| Model | canonical test strict JS ↓ | Intersection ↑ |",
        "|---|---:|---:|",
        f"| Original PT canonical CPU | {metrics['original_pt']['js_distance_base2']:.12f} | {metrics['original_pt']['histogram_intersection']:.12f} |",
        f"| Independent 4×2 | {metrics['independent_4x2']['js_distance_base2']:.12f} | {metrics['independent_4x2']['histogram_intersection']:.12f} |",
        f"| Joint 16 | {metrics['joint_16']['js_distance_base2']:.12f} | {metrics['joint_16']['histogram_intersection']:.12f} |",
        "",
        "| Model | ΔJS (candidate − PT) | relative ΔJS | ΔIntersection | beats PT? |",
        "|---|---:|---:|---:|---|",
    ]
    for architecture, display in (("independent_4x2", "Independent 4×2"), ("joint_16", "Joint 16")):
        item = changes[architecture]
        lines.append(
            f"| {display} | {item['absolute_js_change_candidate_minus_original_pt']:+.12f} | "
            f"{item['relative_js_change_percent']:+.6f}% | "
            f"{item['intersection_change_candidate_minus_original_pt']:+.12f} | "
            f"{'yes' if item['beats_original_pt'] else 'no'} |"
        )
    lines += [
        "",
        "The global MIDI-roundtrip metric above is the primary result. Direct pre-render token metrics are not reported as headline results.",
        "",
        "## F. Global 16-pattern diagnostics",
        "",
        "| Distribution | steady mass | transition-containing mass |",
        "|---|---:|---:|",
        f"| Human | {masses['human']['steady_mass']:.12f} | {masses['human']['transition_containing_mass']:.12f} |",
        f"| Original PT | {masses['original_pt']['steady_mass']:.12f} | {masses['original_pt']['transition_containing_mass']:.12f} |",
        f"| Independent 4×2 | {masses['independent_4x2']['steady_mass']:.12f} | {masses['independent_4x2']['transition_containing_mass']:.12f} |",
        f"| Joint 16 | {masses['joint_16']['steady_mass']:.12f} | {masses['joint_16']['transition_containing_mass']:.12f} |",
        "",
        "Full count/probability and signed/absolute error diagnostics are in `global_joint16_comparison.csv`.",
        "",
        "## G. Piece-level supplemental analysis",
        "",
        "| Model | improved pieces | mean JS | median JS |",
        "|---|---:|---:|---:|",
        f"| Original PT | — | {pieces['original_pt']['mean_js_distance']:.12f} | {pieces['original_pt']['median_js_distance']:.12f} |",
        f"| Independent 4×2 | {pieces['independent']['pieces_improved_vs_original_pt']}/23 | {pieces['independent']['mean_js_distance']:.12f} | {pieces['independent']['median_js_distance']:.12f} |",
        f"| Joint 16 | {pieces['joint']['pieces_improved_vs_original_pt']}/23 | {pieces['joint']['mean_js_distance']:.12f} | {pieces['joint']['median_js_distance']:.12f} |",
        "",
        "Piece-level results are supplemental and were not used for model selection.",
        "",
        "## H. Validation → test generalization",
        "",
        "| Model | validation strict JS ↓ | validation vs PT | test vs PT |",
        "|---|---:|---|---|",
        f"| Original PT | {lock['validation_original_pt']['js_distance_base2']:.12f} | — | — |",
        f"| Independent 4×2 epoch 2 | {lock['checkpoints']['independent_4x2']['validation_strict_js']:.12f} | improved | {'improved' if independent_generalizes else 'did not improve'} |",
        f"| Joint 16 epoch 2 | {lock['checkpoints']['joint_16']['validation_strict_js']:.12f} | did not improve | {'did not improve' if joint_failure_persists else 'improved'} |",
        "",
        ("The Independent validation improvement is retained on canonical test." if independent_generalizes else "The Independent validation improvement is not retained on canonical test."),
        ("The Joint validation failure is also observed on canonical test." if joint_failure_persists else "The Joint validation failure is not repeated on canonical test."),
        "",
        "## I. No-retuning and scientific status",
        "",
        "No training, checkpoint reselection, architecture reselection, threshold change, calibration, smoothing, post-processing, or Stage 1 inference was performed in response to this test result.",
        "",
        "This is the canonical-v1 test evaluation performed after validation-only checkpoint selection under the revised fixed-Stage-1 protocol. No canonical-v1 model or checkpoint selection used the canonical-v1 test result.",
        "",
        "ASAP test had been accessed historically in the earlier `stage2_binary_v0` lineage under other protocols. This report therefore does not describe the data as a previously untouched test set or as first-ever ASAP test access. Earlier GPU Stage-1 and prior canonical re-evaluation results remain separate and are not mixed into this headline table.",
    ]
    return "\n".join(lines) + "\n"


def chown_output() -> None:
    owner = ROOT.stat()
    for path in [OUTPUT, *OUTPUT.rglob("*")]:
        os.chown(path, owner.st_uid, owner.st_gid)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    logger = setup_logger()
    status_path = OUTPUT / "run_status.json"
    stage = "preflight"
    write_json(
        status_path,
        {
            "status": "running",
            "started_at": now(),
            "current_stage": stage,
            "stage1_neural_inference": 0,
        },
    )
    try:
        provenance = verify_fixed_provenance()
        lock = checkpoint_lock()
        manifest, canonical_hashes = canonical_manifest()
        split_rows = test_rows(manifest)
        write_csv(OUTPUT / "canonical_test_manifest.csv", manifest)
        write_json(OUTPUT / "final_experiment_lock.json", lock)
        lock_hash = sha256_file(OUTPUT / "final_experiment_lock.json")
        logger.info("PREFLIGHT_PASS checkpoints=2 mapping=23 human_performances=104")
        stage = "canonical_tokenization"
        write_json(status_path, {"status": "running", "current_stage": stage, "stage1_neural_inference": 0})
        config = PianoT5GemmaConfig()
        canonical_ids: dict[int, np.ndarray] = {}
        for row in manifest:
            number = int(row["num"])
            canonical_ids[number] = np.asarray(
                midi_to_ids(config, MidiFile(str(row["canonical_original_pt_midi_path"]))),
                dtype=np.int64,
            )
        if set(canonical_ids) != set(range(23)):
            raise RuntimeError("canonical tokenization did not cover nums 0..22")
        logger.info("CANONICAL_TOKENIZATION_PASS pieces=23 notes=%d", sum(len(ids) // 8 for ids in canonical_ids.values()))
        stage = "stage2_inference_and_equality"
        write_json(status_path, {"status": "running", "current_stage": stage, "stage1_neural_inference": 0})
        equality_rows, audit_rows, inference = infer_candidates(
            manifest, lock, canonical_ids, canonical_hashes, logger
        )
        logger.info("NON_CC64_EQUALITY_PASS comparisons=46")
        stage = "official_strict_metrics"
        write_json(status_path, {"status": "running", "current_stage": stage, "stage1_neural_inference": 0, "non_cc64_equality": "46/46 PASS"})
        evaluation = evaluate_metrics(manifest, split_rows, canonical_ids, logger)
        if sha256_file(OUTPUT / "final_experiment_lock.json") != lock_hash:
            raise AssertionError("final experiment lock changed during test evaluation")
        for path, digest in canonical_hashes.items():
            if sha256_file(path) != digest:
                raise AssertionError(f"canonical source changed: {path}")
        for architecture in ARCHITECTURES:
            locked = lock["checkpoints"][architecture]
            if sha256_file(locked["checkpoint_path"]) != locked["checkpoint_sha256"]:
                raise AssertionError(f"locked checkpoint changed: {architecture}")
        final = {
            "status": "completed",
            "experiment_id": "stage2_binary_canonical_v1",
            "evaluation_id": "test_eval_v0",
            "completed_at": now(),
            "canonical_test_pieces": 23,
            "canonical_nums": list(range(23)),
            "canonical_stage1_neural_inference": 0,
            "candidate_outputs": {architecture: 23 for architecture in ARCHITECTURES},
            "non_cc64_equality": {"passed": len(equality_rows), "expected": 46},
            "inference": inference,
            "locked_checkpoints": lock["checkpoints"],
            "final_experiment_lock_sha256": lock_hash,
            "canonical_manifest_sha256": sha256_file(OUTPUT / "canonical_test_manifest.csv"),
            "provenance_hashes": provenance,
            "human_reference": {
                "performances": 104,
                "direct_official_midi_to_ids_loads": 104,
                "verified_prior_histogram_artifact": str(HUMAN_PROVENANCE),
                "verified_prior_histogram_artifact_sha256": provenance["human_provenance"],
                "recomputed_histogram_exact_match": True,
            },
            **evaluation,
            "cc64_transplant_totals": {
                architecture: inference[architecture] for architecture in ARCHITECTURES
            },
            "training_performed": 0,
            "checkpoint_reselection": 0,
            "calibration_or_postprocessing": 0,
            "test_result_used_for_model_selection": False,
        }
        write_json(OUTPUT / "final_test_metrics.json", final)
        (OUTPUT / "CANONICAL_BINARY_TEST_REPORT.md").write_text(
            make_report(final, lock, audit_rows), encoding="utf-8"
        )
        write_json(
            status_path,
            {
                "status": "completed",
                "completed_at": now(),
                "stage1_neural_inference": 0,
                "candidate_midis": {architecture: 23 for architecture in ARCHITECTURES},
                "non_cc64_equality": "46/46 PASS",
                "training_performed": 0,
                "checkpoint_reselection": 0,
                "calibration_or_postprocessing": 0,
            },
        )
        chown_output()
        print(json.dumps({"status": "completed", "metrics": final["global_strict_metrics"]}, indent=2))
    except BaseException as error:
        logger.error("FAILED stage=%s error=%s\n%s", stage, error, traceback.format_exc())
        write_json(
            status_path,
            {
                "status": "failed",
                "failed_at": now(),
                "failed_stage": stage,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "stage1_neural_inference": 0,
            },
        )
        chown_output()
        raise


if __name__ == "__main__":
    main()
