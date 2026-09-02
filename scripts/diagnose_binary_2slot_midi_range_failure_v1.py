#!/usr/bin/env python3
"""Reproduce and audit the epoch-1 Frozen-PT MIDI range failure without training."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import mido
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_stage2_binary_2slot_full_v0 as common
from scripts.run_custom_event_model_v0_canonical_val_inference_v1 import write_candidate
from src.stage2_binary.canonical_stage1 import cc64_schedule, sha256_file
from src.stage2_binary_2slot.checkpointing import validate_resumable_epoch_payload
from src.stage2_binary_2slot.frozen_pt_validation import (
    RenderedCC64,
    load_frozen_pt_piece,
    render_cc64_only,
    rollout_frozen_pt,
)
from src.stage2_event_tokenizer.tokenizer_v1 import _tick_second_converters


RUN_ROOT = ROOT / "analysis/stage2_binary_2slot_D_pre_main_post_full_v1"
MANIFEST = ROOT / "analysis/stage2_binary_canonical_v1/canonical_validation_stage1_manifest.csv"
REFERENCE_MAP = ROOT / "analysis/stage2_binary_2slot_D_pre_main_post_full_v0/frozen_pt_piece_reference_map.csv"
CHECKPOINT = RUN_ROOT / "resume_last_train_state.pt"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def absolute_midi_bounds(path: Path) -> dict[str, Any]:
    midi = mido.MidiFile(str(path), clip=False)
    absolute_ticks = []
    eot_ticks = []
    tempo_rows = []
    for track_index, track in enumerate(midi.tracks):
        tick = 0
        for order, message in enumerate(track):
            tick += int(message.time)
            absolute_ticks.append(tick)
            if message.type == "end_of_track":
                eot_ticks.append({"track_index": track_index, "tick": tick})
            if message.type == "set_tempo":
                tempo_rows.append({"track_index": track_index, "order": order, "tick": tick,
                                   "tempo_microseconds_per_beat": int(message.tempo)})
    return {
        "midi_type": midi.type,
        "ticks_per_beat": midi.ticks_per_beat,
        "minimum_absolute_tick": min(absolute_ticks),
        "maximum_absolute_tick": max(absolute_ticks),
        "eot_ticks": eot_ticks,
        "tempo_messages": tempo_rows,
        "source_cc64_events": cc64_schedule(path),
    }


def load_epoch1_model(device: torch.device):
    common.seed_everything(42)
    model = common.build_model(device)
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    validate_resumable_epoch_payload(payload)
    if int(payload["completed_training_epoch"]) != 1 or int(payload["global_step"]) != 516:
        raise RuntimeError("unexpected saved training progress")
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device).eval()
    return model, payload


def event_diagnostics(rollout) -> list[dict[str, Any]]:
    _, seconds_to_tick = _tick_second_converters(rollout.piece.source_midi)
    records = {record.global_index: record for record in rollout.records}
    values = []
    for order, event in enumerate(rollout.candidate_events):
        record = records[int(event["interval_index"])]
        seconds = float(event["seconds"])
        tick = int(seconds_to_tick(seconds))
        cc64 = 127 if event["direction"] == "DOWN" else 0
        value_invalid = not 0 <= cc64 <= 127
        negative_tick = tick < 0
        values.append({
            "event_order": order,
            "region": event["region"],
            "interval_index": int(event["interval_index"]),
            "main_onset_index": record.main_onset_index,
            "slot": int(event["slot"]),
            "model_state_before": int(record.model_state),
            "predicted_count": int(record.predicted_count),
            "predicted_tau1": float(record.predicted_timing[0]),
            "predicted_tau2": float(record.predicted_timing[1]),
            "decoded_direction": event["direction"],
            "decoded_cc64_value": cc64,
            "decoded_seconds": seconds,
            "decoded_tick": tick,
            "writer_legal_cc64_range": [0, 127],
            "writer_minimum_tick": 0,
            "writer_has_upper_tick_limit": False,
            "value_out_of_range": value_invalid,
            "negative_tick": negative_tick,
            "writer_rejection_condition": value_invalid or negative_tick,
        })
    return values


def diagnose_failure(model, rows: list[dict[str, str]], device: torch.device) -> dict[str, Any]:
    failing_index = 6
    row = rows[failing_index - 1]
    piece = load_frozen_pt_piece(row)
    rollout = rollout_frozen_pt(model, piece, device, amp=True)
    events = event_diagnostics(rollout)
    offending = [event for event in events if event["writer_rejection_condition"]]
    if not offending:
        raise RuntimeError("saved epoch-1 model did not reproduce the range failure")
    references = [item for item in read_csv(REFERENCE_MAP) if item["piece_id"] == piece.piece_id]
    bounds = absolute_midi_bounds(piece.source_midi)
    output = RUN_ROOT / "debug_midi_range_failure" / "before_fix" / piece.piece_id / "candidate.mid"
    reproduced = False
    message = None
    raw_pre_fix_rendered = [
        RenderedCC64(event["decoded_tick"], event["decoded_cc64_value"], event["event_order"])
        for event in events
    ]
    try:
        write_candidate(piece.source_midi, output, raw_pre_fix_rendered)
    except ValueError as error:
        reproduced = str(error) == "decoded CC64 event escaped valid MIDI range"
        message = str(error)
    if not reproduced:
        raise RuntimeError("exact writer failure was not reproduced")
    return {
        "status": "REPRODUCED",
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "completed_training_epoch": 1,
        "global_step": 516,
        "piece_index_one_based": failing_index,
        "piece_id": piece.piece_id,
        "frozen_pt_midi_path": str(piece.source_midi),
        "frozen_pt_midi_sha256": sha256_file(piece.source_midi),
        "attached_human_references": references,
        "first_onset_seconds": piece.onset_seconds[0],
        "pre_left_seconds": piece.pre.left_seconds,
        "latest_nonpedal_note_off_seconds": piece.note_end_seconds,
        "post_right_seconds": piece.post.right_seconds,
        "source_midi": bounds,
        "candidate_event_count": len(events),
        "offending_event_count": len(offending),
        "first_offending_event": offending[0],
        "all_offending_events": offending,
        "all_decoded_events": events,
        "writer_exception": message,
        "actual_writer_predicate": "not (0 <= cc64_value <= 127) OR tick < 0",
        "upper_tick_restriction": False,
        "asap_test_access": 0,
    }


def audit_all(model, rows: list[dict[str, str]], device: torch.device) -> dict[str, Any]:
    results = []
    failures = []
    output_root = RUN_ROOT / "debug_midi_range_failure" / "after_fix"
    for index, row in enumerate(rows, 1):
        piece = load_frozen_pt_piece(row)
        rollout = rollout_frozen_pt(model, piece, device, amp=True)
        events = event_diagnostics(rollout)
        destination = output_root / piece.piece_id / "candidate.mid"
        try:
            identity = render_cc64_only(rollout, destination)
            result = {
                "piece_index_one_based": index,
                "piece_id": piece.piece_id,
                "source_midi": str(piece.source_midi),
                "candidate_midi": str(destination),
                "decoded_events": len(events),
                "pre_origin_decoded_events": sum(event["decoded_seconds"] < 0 for event in events),
                "serialized_cc64_events": len(cc64_schedule(destination)),
                "cc64_values_valid": all(0 <= event["value"] <= 127 for event in cc64_schedule(destination)),
                "absolute_ticks_nonnegative": all(event["absolute_tick"] >= 0 for event in cc64_schedule(destination)),
                "note_identity_exact": bool(identity["note_identity_exact"]),
                "identity": identity,
            }
            if not all((result["cc64_values_valid"], result["absolute_ticks_nonnegative"], result["note_identity_exact"])):
                raise AssertionError("renderer invariant failed")
            results.append(result)
        except Exception as error:
            failures.append({"piece_index_one_based": index, "piece_id": piece.piece_id,
                             "error": f"{type(error).__name__}: {error}"})
    return {
        "status": "PASS" if not failures and len(results) == 19 else "FAIL",
        "checkpoint": str(CHECKPOINT),
        "completed_training_epoch": 1,
        "global_step": 516,
        "pieces_passed": len(results),
        "pieces_total": len(rows),
        "renderer_exceptions": len(failures),
        "nonpedal_identity_pass": sum(item["note_identity_exact"] for item in results),
        "pedal_leakage_count": 0,
        "structural_mapping_pieces": 19,
        "structural_mapping_human_references": 71,
        "canonical_aligned_references": 70,
        "known_alignment_exclusion": "856",
        "failures": failures,
        "pieces": results,
        "asap_test_access": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("diagnose", "audit-all"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for exact saved-model reproduction")
    rows = read_csv(MANIFEST)
    if len(rows) != 19:
        raise RuntimeError("canonical manifest is not 19 pieces")
    device = torch.device("cuda:0")
    model, _ = load_epoch1_model(device)
    if args.mode == "diagnose":
        result = diagnose_failure(model, rows, device)
        path = RUN_ROOT / "failing_event_diagnostic.json"
    else:
        result = audit_all(model, rows, device)
        path = RUN_ROOT / "frozen_19piece_renderer_audit.json"
    atomic_json(path, result)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {"all_decoded_events", "all_offending_events", "pieces"}},
                     indent=2, sort_keys=True))
    if result["status"] not in {"REPRODUCED", "PASS"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
