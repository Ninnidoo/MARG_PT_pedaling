#!/usr/bin/env python3
"""Re-evaluate locked binary Stage 2 on the canonical CPU Stage-1 MIDI bank."""

from __future__ import annotations

import csv
import gc
import json
import logging
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from miditoolkit import MidiFile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_stage2_binary_metric_fidelity import _strict_official_similarity  # noqa: E402
from src.stage2_binary.canonical_stage1 import (  # noqa: E402
    assert_strict_non_cc64_equality,
    sha256_file,
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


OUTPUT = ROOT / "analysis/stage2_binary_v0/test_eval_canonical_stage1_v1"
MANIFEST = ROOT / "analysis/stage2_binary_v0/canonical_stage1_pipeline_v0/canonical_stage1_manifest.csv"
LOCK_PATH = ROOT / "analysis/stage2_binary_v0/final_lock_v1/final_experiment_lock.json"
OLD_METRICS = ROOT / "analysis/stage2_binary_v0/test_eval_v0/final_test_metrics.json"
ARCHITECTURES = ("independent_4x2", "joint_16")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("canonical_stage1_test_eval")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.FileHandler(OUTPUT / "evaluation.log"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def evaluate(logger: logging.Logger) -> dict[str, Any]:
    manifest = read_csv(MANIFEST)
    if len(manifest) != 23 or [int(row["num"]) for row in manifest] != list(range(23)):
        raise RuntimeError("canonical manifest is not exact nums 0..22")
    canonical_hashes = {
        row["canonical_original_pt_midi_path"]: row["canonical_midi_sha256"]
        for row in manifest
    }
    for path, digest in canonical_hashes.items():
        if sha256_file(path) != digest:
            raise RuntimeError(f"canonical MIDI hash mismatch: {path}")
    lock_hash = sha256_file(LOCK_PATH)
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    old_metrics_hash = sha256_file(OLD_METRICS)
    old_metrics = json.loads(OLD_METRICS.read_text(encoding="utf-8"))
    human_histogram = np.asarray(old_metrics["histograms"]["human"], dtype=np.int64)
    if human_histogram.shape != (16,) or int(human_histogram.sum()) != 331576:
        raise RuntimeError("existing verified Human test histogram provenance changed")

    config = PianoT5GemmaConfig()
    canonical_ids: dict[int, np.ndarray] = {}
    original_histogram = np.zeros(16, dtype=np.int64)
    for row in manifest:
        number = int(row["num"])
        midi = MidiFile(row["canonical_original_pt_midi_path"])
        ids = np.asarray(midi_to_ids(config, midi), dtype=np.int64)
        canonical_ids[number] = ids
        original_histogram += joint16_histogram_from_ids(ids)
    logger.info("CANONICAL_TOKENIZATION_PASS pieces=23 notes=%d", original_histogram.sum())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    histograms: dict[str, np.ndarray] = {"original_pt": original_histogram}
    equality_rows: list[dict[str, Any]] = []
    model_summaries: dict[str, Any] = {}
    for architecture in ARCHITECTURES:
        locked = lock["candidates"][architecture]
        checkpoint = ROOT / locked["checkpoint_project_relative_path"]
        if sha256_file(checkpoint) != locked["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch: {architecture}")
        model = load_binary_stage2_checkpoint(
            checkpoint,
            architecture=architecture,
            encoder_checkpoint=ROOT / "checkpoints/pianist_transformer",
            device=device,
        )
        donor_root = OUTPUT / "pedal_donors" / architecture
        candidate_root = OUTPUT / "candidate_midis" / architecture
        donor_root.mkdir(parents=True)
        candidate_root.mkdir(parents=True)
        histogram = np.zeros(16, dtype=np.int64)
        notes_total = 0
        windows_total = 0
        for index, row in enumerate(manifest, start=1):
            number = int(row["num"])
            canonical_path = Path(row["canonical_original_pt_midi_path"])
            ids = canonical_ids[number]
            predicted_ids, details = infer_cached_binary_pedals(
                model,
                ids,
                architecture=architecture,
                device=device,
                window_notes=512,
                stride_notes=256,
            )
            donor_path = donor_root / f"{number}_pedal_donor.mid"
            candidate_path = candidate_root / f"{number}_{architecture}_cc64_only.mid"
            performance = ids_to_midi(config, predicted_ids, ref=ids.tolist())
            donor = map_midi(MidiFile(str(canonical_path)), performance)
            donor.dump(str(donor_path))
            transplant = transplant_cc64_only(canonical_path, donor_path, candidate_path)
            strict = assert_strict_non_cc64_equality(canonical_path, candidate_path)
            candidate_roundtrip_ids = midi_to_ids(config, MidiFile(str(candidate_path)))
            histogram += joint16_histogram_from_ids(candidate_roundtrip_ids)
            equality_rows.append(
                {
                    "architecture": architecture,
                    "num": number,
                    "piece_id": row["piece_id"],
                    "canonical_midi": str(canonical_path),
                    "candidate_midi": str(candidate_path),
                    "canonical_sha256": row["canonical_midi_sha256"],
                    "candidate_sha256": sha256_file(candidate_path),
                    "midi_type_exact": strict["midi_type_exact"],
                    "ticks_per_beat_exact": strict["ticks_per_beat_exact"],
                    "track_count_exact": strict["track_count_exact"],
                    "all_ordered_non_cc64_events_exact": strict["all_ordered_non_cc64_events_exact"],
                    "canonical_cc64_events": strict["canonical_cc64_events"],
                    "candidate_cc64_events": strict["candidate_cc64_events"],
                    "cc64_changed": strict["cc64_changed"],
                    "donor_cc64_discarded_after_canonical_eot": transplant[
                        "donor_cc64_discarded_after_canonical_eot"
                    ],
                    "canonical_unchanged": transplant["canonical_unchanged"],
                    "status": "PASS",
                }
            )
            notes_total += int(details["notes"])
            windows_total += int(details["windows"])
            logger.info("STAGE2_PASS architecture=%s piece=%d/23 num=%d notes=%d windows=%d", architecture, index, number, details["notes"], details["windows"])
        histograms[architecture] = histogram
        model_summaries[architecture] = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": locked["checkpoint_sha256"],
            "best_epoch": locked["best_epoch"],
            "notes": notes_total,
            "windows": windows_total,
            "candidate_midis": 23,
        }
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if len(equality_rows) != 46 or not all(row["status"] == "PASS" for row in equality_rows):
        raise AssertionError("strict canonical equality is not 46/46")
    write_csv(OUTPUT / "midi_nonpedal_equality.csv", equality_rows)

    metrics = {
        name: _strict_official_similarity(human_histogram, histogram)
        for name, histogram in histograms.items()
    }
    distribution_rows = []
    for joint_id, pattern in enumerate(JOINT_PATTERNS):
        row: dict[str, Any] = {"joint_id": joint_id, "pattern": pattern}
        for name, histogram in {"human": human_histogram, **histograms}.items():
            row[f"{name}_count"] = int(histogram[joint_id])
            row[f"{name}_probability"] = float(histogram[joint_id] / histogram.sum())
        distribution_rows.append(row)
    write_csv(OUTPUT / "global_joint16_comparison.csv", distribution_rows)
    for path, digest in canonical_hashes.items():
        if sha256_file(path) != digest:
            raise AssertionError(f"canonical MIDI changed: {path}")
    if sha256_file(LOCK_PATH) != lock_hash or sha256_file(OLD_METRICS) != old_metrics_hash:
        raise AssertionError("protected provenance artifact changed")
    return {
        "status": "PASS",
        "protocol": "canonical fixed CPU Original-PT Stage 1 + CC64-only transplant",
        "canonical_manifest": str(MANIFEST),
        "canonical_manifest_sha256": sha256_file(MANIFEST),
        "canonical_pieces": 23,
        "canonical_stage1_neural_inference": 0,
        "architectures": model_summaries,
        "non_cc64_equality": "46/46 PASS",
        "human_histogram_source": str(OLD_METRICS),
        "human_histogram_source_sha256": old_metrics_hash,
        "human_midi_reopened": 0,
        "histograms": {name: values.tolist() for name, values in {"human": human_histogram, **histograms}.items()},
        "global_strict_metrics": metrics,
        "device": str(device),
        "final_lock": str(LOCK_PATH),
        "final_lock_sha256": lock_hash,
        "training_performed": 0,
        "checkpoint_reselection": 0,
        "test_protocol_revision_after_prior_test_observation": True,
    }


def make_report(result: dict[str, Any]) -> str:
    metrics = result["global_strict_metrics"]
    old = json.loads(OLD_METRICS.read_text(encoding="utf-8"))["global_metrics"]
    original = metrics["original_pt"]
    independent = metrics["independent_4x2"]
    joint = metrics["joint_16"]
    return f"""# Canonical Fixed-Stage-1 Binary Stage 2 Re-evaluation

## Result

| Model | canonical Stage-1 strict JS ↓ | Intersection ↑ |
|---|---:|---:|
| Original PT canonical CPU MIDI | {metrics['original_pt']['js_distance_base2']:.12f} | {metrics['original_pt']['histogram_intersection']:.12f} |
| independent 4×2 locked | {metrics['independent_4x2']['js_distance_base2']:.12f} | {metrics['independent_4x2']['histogram_intersection']:.12f} |
| joint 16 locked | {metrics['joint_16']['js_distance_base2']:.12f} | {metrics['joint_16']['histogram_intersection']:.12f} |

| Model | JS change vs canonical Original PT | relative JS change | Intersection change |
|---|---:|---:|---:|
| independent 4×2 | {independent['js_distance_base2'] - original['js_distance_base2']:+.12f} | {(independent['js_distance_base2'] / original['js_distance_base2'] - 1) * 100:+.6f}% | {independent['histogram_intersection'] - original['histogram_intersection']:+.12f} |
| joint 16 | {joint['js_distance_base2'] - original['js_distance_base2']:+.12f} | {(joint['js_distance_base2'] / original['js_distance_base2'] - 1) * 100:+.6f}% | {joint['histogram_intersection'] - original['histogram_intersection']:+.12f} |

Under this revised canonical Stage-1 protocol, neither locked Stage 2 checkpoint improves the canonical Original PT global pedal metric. No model/checkpoint was reselected in response.

## Corrected protocol

- Stage 1 is the immutable 23-piece CPU/seed-42 bank `outputs/midi/{{num}}_original_pt.mid`; PT neural inference was not rerun.
- Each canonical MIDI was tokenized with official `midi_to_ids`; Pedal1--4 were masked by the existing Stage 2 helper.
- The unchanged final-lock-v1 independent epoch 2 and joint epoch 3 checkpoints were used with 512-note windows, stride 256, overlap-logit averaging, and deterministic argmax.
- `ids_to_midi` + `map_midi` produced temporary pedal donors only. Final candidates copied every canonical raw event and transplanted donor CC64 only.
- Donor CC64 strictly after canonical end-of-track was discarded (never shifted); canonical EOT remained exact and discard counts are recorded per piece.
- Strict raw MIDI non-CC64 equality: **46/46 PASS**. Canonical source hashes remained unchanged.
- Human MIDI was not reopened; the prior official 104-performance Human histogram was reused byte-for-byte from `{OLD_METRICS.relative_to(ROOT)}` (`{result['human_histogram_source_sha256']}`).

## Prior GPU Stage-1 result (not directly comparable as the same realization)

| Model | prior GPU Stage-1 JS ↓ | Intersection ↑ |
|---|---:|---:|
| Original PT | {old['original_pt']['js_distance_base2']:.12f} | {old['original_pt']['histogram_intersection']:.12f} |
| independent 4×2 | {old['independent_4x2']['js_distance_base2']:.12f} | {old['independent_4x2']['histogram_intersection']:.12f} |
| joint 16 | {old['joint_16']['js_distance_base2']:.12f} | {old['joint_16']['histogram_intersection']:.12f} |

## Scientific status

No training, checkpoint selection, threshold/decoding change, calibration, Stage 1 inference, or Human-MIDI re-evaluation occurred. This is a revised canonical Stage-1 protocol adopted after inspecting the earlier test realization, so it is reported separately from the original locked final-test artifact and does not overwrite it.
"""


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite output: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    logger = setup_logger()
    status_path = OUTPUT / "run_status.json"
    write_json(status_path, {"status": "running", "started_at": now()})
    try:
        result = evaluate(logger)
        write_json(OUTPUT / "canonical_test_metrics.json", result)
        (OUTPUT / "CANONICAL_TEST_REEVALUATION_REPORT.md").write_text(make_report(result), encoding="utf-8")
        write_json(status_path, {"status": "completed", "completed_at": now(), "non_cc64_equality": "46/46 PASS"})
        owner = ROOT.stat()
        for path in [OUTPUT, *OUTPUT.rglob("*")]:
            os.chown(path, owner.st_uid, owner.st_gid)
        print(json.dumps({"status": "completed", "metrics": result["global_strict_metrics"]}, indent=2))
    except BaseException as error:
        logger.error("FAILED %s\n%s", error, traceback.format_exc())
        write_json(status_path, {"status": "failed", "failed_at": now(), "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
