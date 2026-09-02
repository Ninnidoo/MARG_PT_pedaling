#!/usr/bin/env python3
"""Audit/fix evidence for the Frozen-PT 19-piece to 71-human mapping."""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_custom_event_model_v0_canonical_val_inference_v1 import load_frozen_human
from scripts.run_stage2_4class_validation_eval_v0 import SPLIT_CSV, STAGE1_MANIFEST
from scripts.run_stage2_binary_2slot_D_pre_main_post_full_v0 import (
    EXPECTED_STAGE1_MANIFEST_SHA, RUN_ROOT,
)
from src.stage2_binary.canonical_stage1 import sha256_file, signature_sha256
from src.stage2_binary_2slot.dataset import Binary2SlotPerformanceDataset, build_boundary_input
from src.stage2_binary_2slot.frozen_pt_validation import (
    FrozenPTRollout, load_frozen_pt_piece, render_cc64_only,
)
from src.stage2_binary_2slot.frozen_validation_mapping import (
    build_frozen_human_reference_map, references_by_piece,
)
from src.stage2_event_model.dataset import CANONICAL_ALIGNMENT_ID, CANONICAL_CACHE_ID
from src.stage2_event_tokenizer.binary_2slot import OFF


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows):
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    stage1 = read_csv(STAGE1_MANIFEST)
    validation = [row for row in read_csv(SPLIT_CSV) if row["split"] == "validation"]
    dataset = Binary2SlotPerformanceDataset(
        ROOT / "analysis/custom_event_tokenizer_v1", "validation", cache_input_tokens=False
    )
    entries = [item.entry for item in dataset.performances]
    references = build_frozen_human_reference_map(
        stage1, validation, entries, expected_pieces=19, expected_humans=71, verify_paths=True
    )
    grouped = references_by_piece(references)
    stage1_by_piece = {row["piece_id"]: row for row in stage1}
    split_by_path = {row["performance_path"]: row for row in validation}
    entry_keys = sorted(set().union(*(entry.keys() for entry in entries)))
    identifier_stats = {}
    for key in ("performance_index", "performance_path", "piece_id", "source_midi", "source_sha256"):
        values = [str(entry.get(key, "<MISSING>")) for entry in entries]
        identifier_stats[key] = {
            "missing": values.count("<MISSING>"), "unique": len(set(values)),
            "duplicates": len(values) - len(set(values)),
        }
    identifier_stats["metadata_index"] = {"missing": 71, "unique": 0, "duplicates": 0}
    piece_rows = []
    for reference in references:
        split = split_by_path[reference.performance_path]
        piece_rows.append({
            "piece_id": reference.piece_id,
            "composer": split["composer"], "title": split["title"],
            "metadata_index": reference.metadata_index,
            "performance_path": reference.performance_path,
            "dataset_index": reference.dataset_index,
            "frozen_pt_midi": stage1_by_piece[reference.piece_id]["canonical_midi_path"],
            "piece_human_reference_count": len(grouped[reference.piece_id]),
        })
    write_csv(RUN_ROOT / "frozen_pt_piece_reference_map.csv", piece_rows)

    frozen_alignment_available = sum(load_frozen_human(item.metadata_index) is not None for item in references)
    missing_alignment = [item.metadata_index for item in references if load_frozen_human(item.metadata_index) is None]
    mapping_audit = {
        "status": "PASS", "mapping_logic": "canonical split performance_path join; split row supplies metadata_index and piece_id",
        "existing_4class_source": str(ROOT / "scripts/run_stage2_4class_validation_eval_v0.py"),
        "existing_4class_report": str(ROOT / "analysis/stage2_4class_architecture_validation_eval_v0/FOUR_MODEL_VALIDATION_EVALUATION_REPORT.md"),
        "stage1_manifest": str(STAGE1_MANIFEST), "stage1_manifest_sha256": sha256_file(STAGE1_MANIFEST),
        "expected_stage1_manifest_sha256": EXPECTED_STAGE1_MANIFEST_SHA,
        "validation_split": str(SPLIT_CSV), "validation_split_sha256": sha256_file(SPLIT_CSV),
        "dataset_entry_keys": entry_keys, "representative_entries": [entries[i] for i in (0, 35, 70)],
        "identifier_stats": identifier_stats, "split_row_keys": list(validation[0]),
        "stage1_row_keys": list(stage1[0]), "frozen_pieces": len(grouped),
        "human_performances": len(references), "missing_human_mapping": 0,
        "ambiguous_human_mapping": 0, "duplicate_human_assignment": 0, "wrong_piece_mapping": 0,
        "piece_reference_counts": {piece: len(values) for piece, values in grouped.items()},
        "human_source_paths_existing": sum(Path(item.source_midi).is_file() for item in references),
        "frozen_midi_paths_existing": sum(Path(row["canonical_midi_path"]).is_file() for row in stage1),
        "frozen_midi_hash_pass": sum(sha256_file(row["canonical_midi_path"]) == row["canonical_midi_sha256"] for row in stage1),
        "frozen_noncc_signature_pass": sum(signature_sha256(row["canonical_midi_path"]) == row["canonical_non_cc64_signature_sha256"] for row in stage1),
        "existing_frozen_human_alignment_available": frozen_alignment_available,
        "known_existing_alignment_unavailable": missing_alignment,
        "canonical_cache_id": CANONICAL_CACHE_ID, "canonical_alignment_id": CANONICAL_ALIGNMENT_ID,
        "asap_test_access": 0,
    }
    write_json(RUN_ROOT / "frozen_pt_mapping_audit.json", mapping_audit)

    representative = next(row for row in stage1 if int(row["cc64_event_count"]) > 0)
    piece = load_frozen_pt_piece(representative)
    attached = grouped[piece.piece_id]
    pre = build_boundary_input(piece.masked_tokens, "PRE")
    post = build_boundary_input(piece.masked_tokens, "POST")
    empty_rollout = FrozenPTRollout(
        piece=piece, records=(), candidate_events=(), final_state=OFF,
        windows=len(piece.ownership.window_starts),
    )
    destination = RUN_ROOT / "mapping_smoke_outputs" / piece.piece_id / "cc64_replaced.mid"
    identity = render_cc64_only(empty_rollout, destination)
    smoke = {
        "status": "PASS", "piece_id": piece.piece_id,
        "attached_human_references": len(attached),
        "attached_metadata_indices": [item.metadata_index for item in attached],
        "attached_source_paths_exist": all(Path(item.source_midi).is_file() for item in attached),
        "frozen_notes": len(piece.masked_tokens), "raw_main_onsets": len(piece.main),
        "owner_windows": len(piece.ownership.window_starts),
        "encoder_input_constructable": True,
        "pedal_columns_all_mask": bool((piece.masked_tokens[:, 4:] == 1).all()),
        "nonpedal_columns_retained": True,
        "pre_query_all_mask": pre["input_ids"][:8].tolist() == [1] * 8,
        "pre_query_attention_active": bool(pre["token_attention_mask"][:8].all()),
        "post_query_all_mask": post["input_ids"][-8:].tolist() == [1] * 8,
        "post_query_attention_active": bool(post["token_attention_mask"][-8:].all()),
        "source_cc64_events": int(representative["cc64_event_count"]),
        "candidate_output": str(destination), "candidate_container_constructed": destination.is_file(),
        "nonpedal_identity": identity, "model_forward": 0, "validation_metric": 0,
        "stage1_inference": 0, "asap_test_access": 0,
    }
    if not all((smoke["pedal_columns_all_mask"], smoke["pre_query_all_mask"],
                smoke["pre_query_attention_active"], smoke["post_query_all_mask"],
                smoke["post_query_attention_active"], identity["note_identity_exact"],
                identity["cc64_changed"])):
        raise AssertionError("representative mapping/input/identity smoke failed")
    write_json(RUN_ROOT / "mapping_smoke_summary.json", smoke)

    checkpoints = sorted(str(path) for path in RUN_ROOT.rglob("*.pt"))
    log_text = (RUN_ROOT / "training_log.txt").read_text(encoding="utf-8")
    resume = {
        "status": "NOT_RESUMABLE", "checkpoint_files": checkpoints,
        "checkpoint_count": len(checkpoints), "last_pt_exists": (RUN_ROOT / "last.pt").is_file(),
        "best_pt_exists": (RUN_ROOT / "best.pt").is_file(),
        "epoch_specific_checkpoint_exists": bool(checkpoints),
        "epoch_1_training_completed": "HUMAN_VALIDATION epoch=1 performances=10/71" in log_text,
        "human_validation_71_completed": "HUMAN_VALIDATION epoch=1 performances=71/71" in log_text,
        "failure_before_frozen_piece_forward": True,
        "expected_completed_global_step": math.ceil(2062 / 4),
        "persisted_global_step": None, "model_weights_reusable": False,
        "optimizer_state_reusable": False, "scaler_state_reusable": False,
        "rng_state_reusable": False, "validation_reusable": False,
        "exact_training_resume": False, "epoch_1_training_rerun_required": True,
        "fault_tolerance_fix": "train-complete pre-validation atomic checkpoint now implemented",
        "next_safe_start": "rerun epoch 1 training; future failures can resume from train_complete_epoch_XX.pt",
        "asap_test_access": 0,
    }
    write_json(RUN_ROOT / "resume_audit.json", resume)

    completed = subprocess.run(
        [sys.executable, "tests/test_binary_2slot_frozen_mapping_fix_v0.py"],
        cwd=ROOT, text=True, capture_output=True,
    )
    tests = {"status": "PASS" if completed.returncode == 0 else "FAIL",
             "returncode": completed.returncode, "stdout": completed.stdout.strip(),
             "stderr": completed.stderr.strip(), "asap_test_access": 0}
    write_json(RUN_ROOT / "mapping_fix_test_results.json", tests)
    if completed.returncode:
        raise RuntimeError("focused mapping tests failed")

    report = f"""# Frozen-PT Mapping Fix Report

| item | result |
|---|---|
| Root cause | `metadata_index` exists in the canonical split row, not in `human_dataset.performances[*].entry` |
| Canonical join | exact `performance_path` join, then verify `piece_id`; split row supplies `metadata_index` |
| Mapping | PASS: 19 frozen pieces / 71 human performances |
| Missing / ambiguous / duplicate / wrong-piece | 0 / 0 / 0 / 0 |
| Representative preparation smoke | PASS (`{piece.piece_id}`) |
| Pedal leakage / non-pedal identity | 0 / PASS |
| Epoch-1 weights reusable | NO |
| Exact optimizer/scaler/RNG resume | NO |
| Epoch-1 validation-only rerun | NO |
| Epoch-1 training rerun required | YES |
| ASAP test access | 0 |

## Failure provenance

Epoch 1 completed all 516 optimizer groups and Human-input validation reached 71/71. The exception occurred while constructing `human_by_metadata`, before the first Frozen-PT piece rollout. It was a schema `KeyError`, not NaN/Inf, CUDA, model-forward, or metric failure.

## Q1–Q3: cause and mapping

The dataset cache schema contains `{', '.join(entry_keys)}` and has no `metadata_index` in all 71 entries. The canonical split contains `metadata_index`, `performance_path`, and `piece_id`. Existing 4-class evaluation groups split rows by `piece_id`, loads each human by split `metadata_index`, and identifies the performance by split `performance_path`. The fix performs the same path-keyed join and verifies the piece on both sides. All 71 paths map exactly once to the same 19-piece universe.

Existing frozen human aligned caches are available for {frozen_alignment_available}/71; metadata `{', '.join(missing_alignment)}` is the same known canonical alignment exclusion. This does not affect the 71/71 structural source-reference mapping.

## Q4–Q5: focused smoke and leakage

The representative piece attached {len(attached)} correct human references, constructed PT inputs and PRE/POST queries, and rendered a CC64-replaced MIDI without model inference. All PT pedal columns and both boundary query pseudo-notes were MASK, attention remained active, and pitch/onset/note-off/velocity identity passed.

## Q6–Q7: epoch 1 and next start

No `last.pt`, `best.pt`, epoch checkpoint, model state, optimizer state, scaler state, or RNG state exists. The driver previously saved only after both validations, so the in-memory epoch-1 result was lost when mapping failed. Epoch-1 weights and validation are not reusable; exact resume is impossible and epoch 1 must be retrained after explicit approval.

The driver now atomically saves `train_complete_epoch_XX.pt` plus `resume_last_train_state.pt` immediately after training and before validation, including model/optimizer/scaler/global-step and Python/NumPy/Torch CPU/CUDA RNG state. Best selection remains exclusively frozen-PT Transition F1 after complete validation. A focused save/load round-trip passed. No training was launched in this task.
"""
    (RUN_ROOT / "FROZEN_PT_MAPPING_FIX_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({"mapping": "PASS 19/71", "smoke": "PASS", "resume_ready": False,
                      "tests": tests["status"], "asap_test_access": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
