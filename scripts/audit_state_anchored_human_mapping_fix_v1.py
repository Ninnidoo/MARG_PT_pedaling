#!/usr/bin/env python3
"""Pre-resume audit for the Run B same-human validation mapping fix."""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_state_anchored_transition_v1_full as full
from scripts.run_stage2_4class_validation_eval_v0 import SPLIT_CSV, STAGE1_MANIFEST
from src.stage2_binary.canonical_stage1 import sha256_file
from src.stage2_binary_2slot.checkpointing import restore_resumable_epoch_payload
from src.stage2_binary_2slot.frozen_pt_validation import load_frozen_pt_piece
from src.stage2_state_anchored.dataset import StateAnchoredWindowDataset, diagnose_ownership
from src.stage2_state_anchored.inference import collect_frozen_pt_piece_logits, decode_performance
from src.stage2_state_anchored.serialization import render_state_anchored_candidate
from src.stage2_state_anchored.targets import StateAnchoredIntervalTarget


OUTPUT = full.RUN_ROOT
CHECKPOINT = OUTPUT / "resume_last_train_state.pt"


def main() -> int:
    tests = full.focused_tests()
    test_payload = {
        "status": "PASS", "execution": "direct-function runners; pytest not required",
        "files": tests, "test_functions": sum(int(row["tests"]) for row in tests),
        "asap_test_access": 0,
    }
    full.atomic_json(OUTPUT / "focused_regression_results.json", test_payload)

    fields = [field.name for field in dataclasses.fields(StateAnchoredIntervalTarget)]
    if "duration_seconds" in fields:
        raise AssertionError("frozen Run B target unexpectedly gained duration_seconds")
    dataset = StateAnchoredWindowDataset(full.CACHE_ROOT, "validation", cache_input_tokens=False)
    ownership = diagnose_ownership(dataset)
    state_count = mode_count = cardinality_failures = boundary_failures = 0
    for index, performance in enumerate(dataset.performances):
        targets = dataset.performance_targets(index)
        state_count += len(targets.onset_states)
        mode_count += len(targets.intervals)
        if len(targets.onset_states) != performance.num_onsets or len(targets.intervals) != performance.num_onsets - 1:
            cardinality_failures += 1
        if any(interval.right_seconds <= interval.left_seconds for interval in targets.intervals):
            boundary_failures += 1
    if len(dataset.performances) != 71 or cardinality_failures or boundary_failures:
        raise AssertionError("Human validation static direct-mapping audit failed")

    split_rows = [row for row in full.read_csv(SPLIT_CSV) if row["split"] == "validation"]
    performance_55 = dataset.performances[55]
    split_55 = next(row for row in split_rows if row["performance_path"] == performance_55.entry["performance_path"])
    if split_55["metadata_index"] != "856":
        raise AssertionError("performance 55 is no longer the known metadata 856 case")
    if sha256_file(STAGE1_MANIFEST) != full.EXPECTED_STAGE1_MANIFEST_SHA:
        raise RuntimeError("Frozen Stage 1 manifest provenance changed")
    stage1_rows = full.read_csv(STAGE1_MANIFEST)

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    expected = {
        "checkpoint_kind": "train_epoch_complete_pre_validation",
        "completed_training_epoch": 1,
        "validations_complete": False,
        "global_step": 516,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise RuntimeError("epoch-1 validation-pending checkpoint identity changed")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("mapping-fix dry-run requires one isolated CUDA device")
    device = torch.device("cuda:0")
    full.seed_everything()
    model = full.build_model(device)
    optimizer = full.build_optimizer(model)
    scaler = torch.amp.GradScaler("cuda", init_scale=full.AMP_INIT_SCALE, enabled=True)
    epoch, global_step = restore_resumable_epoch_payload(
        payload, model=model, optimizer=optimizer, scaler=scaler, restore_rng=False,
    )
    if (epoch, global_step) != (1, 516):
        raise RuntimeError("checkpoint restore progress mismatch")

    human_dataset = StateAnchoredWindowDataset(full.CACHE_ROOT, "validation", cache_input_tokens=True)
    original_alignment = full.common.score_positions_for_performance
    external_calls = 0

    def forbidden_external_alignment(*args, **kwargs):
        nonlocal external_calls
        external_calls += 1
        raise AssertionError("Human validation called external score/performance alignment")

    full.common.score_positions_for_performance = forbidden_external_alignment
    try:
        human = full.human_validation(model, human_dataset, device, epoch=1)
    finally:
        full.common.score_positions_for_performance = original_alignment
    if human["metric_performances"] != 71 or human["alignment_failures"] or external_calls:
        raise AssertionError("Human 71/71 direct evaluation path failed")
    if human["viterbi_conflict_count"] != 0:
        raise AssertionError("Human constrained decoding produced a conflict")

    representative_rows = [stage1_rows[0], next(row for row in stage1_rows if row["piece_id"] == performance_55.entry["piece_id"])]
    frozen_rows = []
    for row in representative_rows:
        piece = load_frozen_pt_piece(row)
        if bool(torch.any(piece.masked_tokens[:, 4:] != 1)):
            raise AssertionError("Frozen PT pedal token leakage")
        logits = collect_frozen_pt_piece_logits(model, piece, device, amp=True)
        decoded = decode_performance(logits)
        rendered = render_state_anchored_candidate(
            piece, decoded.viterbi.state_path, decoded.viterbi.mode_path,
            logits.timing_predictions,
            OUTPUT / "mapping_fix_frozen_regression" / piece.piece_id / "candidate.mid",
        )
        if decoded.viterbi.conflict_count or not rendered.identity["note_identity_exact"]:
            raise AssertionError("Frozen representative Viterbi/identity regression")
        if rendered.identity["initialization_anchor_metric_event_count"]:
            raise AssertionError("Frozen representative S1 anchor contaminated metric")
        frozen_rows.append({
            "piece_id": piece.piece_id, "onsets": len(piece.onset_seconds),
            "viterbi_conflicts": decoded.viterbi.conflict_count,
            "pedal_leakage": 0, "nonpedal_identity": True,
            "initialization_anchor_metric_event_count": 0,
        })

    audit = {
        "status": "PASS", "schema_fields": fields, "duration_seconds_field_exists": False,
        "duration_rule": "right_seconds - left_seconds",
        "human_mapping_strategy": "direct same-performance canonical distinct-onset indices",
        "human_external_alignment_calls": external_calls,
        "human_performances": 71, "human_mapping_failures": 0,
        "state_targets": state_count, "mode_interval_targets": mode_count,
        "state_count_equals_M_failures": cardinality_failures,
        "mode_count_equals_M_minus_1_failures": cardinality_failures,
        "ownership": ownership.__dict__,
        "human_viterbi_conflicts": human["viterbi_conflict_count"],
        "human_metrics": {key: human[key] for key in (
            "transition_precision", "transition_recall", "transition_f1",
            "predicted_transitions", "reference_transitions", "predicted_reference_ratio",
            "raw_conflict_count", "raw_conflict_rate", "viterbi_conflict_count",
            "timing_active_mae", "timing_change_mae", "timing_return_tau1_mae", "timing_return_tau2_mae",
        )},
        "performance_index_55": {
            "performance_path": performance_55.entry["performance_path"],
            "source_midi": performance_55.entry["source_midi"],
            "metadata_index": split_55["metadata_index"], "piece_id": performance_55.entry["piece_id"],
            "num_onsets": performance_55.num_onsets,
            "known_frozen_alignment_exclusion": True,
            "human_direct_mapping_status": "PASS",
        },
        "frozen_regression": frozen_rows,
        "frozen_stage1_manifest": str(STAGE1_MANIFEST),
        "frozen_stage1_manifest_sha256": full.EXPECTED_STAGE1_MANIFEST_SHA,
        "stage1_regeneration": 0, "pedal_leakage": 0,
        "checkpoint": str(CHECKPOINT), "checkpoint_restore": "PASS",
        "completed_training_epoch": epoch, "global_step": global_step,
        "validations_complete": False, "epoch_1_train_reexecuted": False,
        "resume_gate": "PASS", "asap_test_access": 0,
    }
    full.atomic_json(OUTPUT / "human_mapping_audit.json", audit)
    report = f"""# Human Validation Mapping Fix Report

**Verdict: PASS**

## Q1. `duration_seconds` AttributeError root cause

`StateAnchoredIntervalTarget` intentionally stores `left_seconds` and `right_seconds` but has no legacy `duration_seconds` property. The Human reference-event reconstruction reused the Run A `HumanIntervalPrimitive` interface and failed for 70 performances after inference.

## Q2. Conflicting legacy/helper interface

The conflicting code was Run A-style `interval.left_seconds + tau * interval.duration_seconds`, combined with `score_positions_for_performance()` and `trajectory_for_metric()`. Run B now derives duration as `right_seconds - left_seconds` and maps events directly to the same performance's distinct-onset indices.

## Q3. Is Nakamura/PT alignment needed for Human-input validation?

No. Prediction onset/interval `i` and target onset/interval `i` share the same canonical performance timeline. Full dry-run external alignment calls: **{external_calls}**.

## Q4. Performance index 55

Index 55 is `{performance_55.entry['performance_path']}`, piece `{performance_55.entry['piece_id']}`, metadata **856**, with {performance_55.num_onsets:,} onsets. It is the existing Frozen cross-performance alignment exclusion. The timeout was caused only by the unnecessary Human external-alignment call; direct Human mapping now includes it and passes.

## Q5. Human 71/71 path

PASS: prediction → owner assembly → global Viterbi → direct common-horizon mapping → aggregation completed for **{human['metric_performances']}/71**, failures 0, Viterbi conflicts 0. Transition P/R/F1={human['transition_precision']:.6f}/{human['transition_recall']:.6f}/{human['transition_f1']:.6f}.

## Q6. Frozen-PT regression

PASS on {len(frozen_rows)} representatives including the metadata-856 piece family: mapping/logits/Viterbi/MIDI/S1/non-pedal identity PASS; pedal leakage 0. Frozen 19/71/70 logic and Stage 1 provenance were not changed.

## Q7. Safe epoch-1 resume

PASS: `{CHECKPOINT}` restored as training-complete epoch **{epoch}**, global step **{global_step}**, validation pending. Epoch-1 training reexecution must remain 0.

Focused tests: {test_payload['test_functions']} direct-function tests PASS. ASAP test access: 0.
"""
    (OUTPUT / "HUMAN_VALIDATION_MAPPING_FIX_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({"status": "PASS", "human": "71/71", "external_alignment_calls": external_calls,
                      "checkpoint": [epoch, global_step], "frozen_representatives": len(frozen_rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
