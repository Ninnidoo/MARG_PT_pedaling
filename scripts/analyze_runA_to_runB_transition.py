#!/usr/bin/env python3
"""Artifact-only Run A summary and common [t1,tM) Transition evaluation."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

from miditoolkit import MidiFile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_pedal_event_metric_tolerance_mini import raw_events
from scripts.run_custom_event_model_v0_canonical_val_inference_v1 import load_frozen_human
from scripts.run_stage2_4class_validation_eval_v0 import SPLIT_CSV, STAGE1_MANIFEST, finalize_transition, sum_transition
from scripts.run_stage2_binary_2slot_full_v0 import trajectory_for_metric, transition_accumulator
from src.stage2_binary_2slot.dataset import Binary2SlotPerformanceDataset
from src.stage2_binary_2slot.frozen_pt_validation import serialized_candidate_events_for_metric
from src.stage2_binary_2slot.frozen_validation_mapping import build_frozen_human_reference_map, references_by_piece
from src.stage2_four_class.validation_evaluator import pooled_transition_counts


RUN_A = ROOT / "analysis/stage2_binary_2slot_D_pre_main_post_full_v1"
OUTPUT = ROOT / "analysis/runA_to_runB_transition"
CACHE = ROOT / "analysis/custom_event_tokenizer_v1"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def score_positions(row: dict[str, str]) -> tuple[list[int], list[int]]:
    cache = RUN_A / "frozen_pt_score_positions" / f"{row['piece_id']}.json"
    value = json.loads(cache.read_text(encoding="utf-8"))
    onsets, _ = raw_events(MidiFile(row["canonical_midi_path"]))
    if value["raw_onsets"] != onsets or value["source_sha256"] != row["canonical_midi_sha256"]:
        raise RuntimeError(f"Run A score-position cache provenance changed: {row['piece_id']}")
    return onsets, [int(item) for item in value["score_positions"]]


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frozen_rows = read_csv(RUN_A / "frozen_pt_validation_by_epoch.csv")
    human_rows = read_csv(RUN_A / "human_input_validation_by_epoch.csv")
    train_rows = read_csv(RUN_A / "training_metrics_by_epoch.csv")
    best = next(row for row in frozen_rows if int(row["epoch"]) == 8)
    human_best = next(row for row in human_rows if int(row["epoch"]) == 8)
    train_best = next(row for row in train_rows if int(row["epoch"]) == 8)
    if abs(float(best["transition_f1"]) - 0.5835638492499086) > 1e-15:
        raise RuntimeError("Run A best epoch/F1 provenance changed")

    stage1 = read_csv(STAGE1_MANIFEST)
    validation = [row for row in read_csv(SPLIT_CSV) if row["split"] == "validation"]
    dataset = Binary2SlotPerformanceDataset(CACHE, "validation", cache_input_tokens=False)
    references = build_frozen_human_reference_map(
        stage1, validation, [item.entry for item in dataset.performances],
        expected_pieces=19, expected_humans=71, verify_paths=True,
    )
    grouped = references_by_piece(references)
    total = transition_accumulator()
    pairs = 0
    excluded: list[str] = []
    piece_rows: list[dict[str, Any]] = []
    for row in stage1:
        onsets, positions = score_positions(row)
        if len(onsets) < 2:
            raise RuntimeError("common horizon requires at least two distinct onsets")
        left_tick, right_tick = int(onsets[0]), int(onsets[-1])
        midi_path = RUN_A / "frozen_pt_outputs/epoch_08" / row["piece_id"] / "candidate.mid"
        serialized = serialized_candidate_events_for_metric(midi_path)
        represented = [event for event in serialized if left_tick <= int(event["source_tick"]) < right_tick]
        candidate = trajectory_for_metric(represented, row["canonical_midi_path"], onsets, positions)
        piece_total = transition_accumulator()
        piece_pairs = 0
        for reference in grouped[row["piece_id"]]:
            frozen = load_frozen_human(reference.metadata_index)
            if frozen is None:
                excluded.append(reference.metadata_index)
                continue
            timeline = dataset.timeline(reference.dataset_index)
            left_ms = float(timeline.main[0].left_seconds) * 1000.0
            right_ms = float(timeline.main[-1].left_seconds) * 1000.0
            target = {"transitions": [
                event for event in frozen["transitions"]["transitions"]
                if left_ms <= float(event["time_ms"]) < right_ms
            ]}
            counts = pooled_transition_counts(candidate, target)
            sum_transition(total, counts)
            sum_transition(piece_total, counts)
            pairs += 1
            piece_pairs += 1
        piece_metric = finalize_transition(piece_total)["pooled"]
        piece_rows.append({"piece_id": row["piece_id"], "pairs": piece_pairs, **piece_metric})
    if pairs != 70 or sorted(excluded) != ["856"]:
        raise RuntimeError(f"canonical common-horizon population changed: pairs={pairs}, excluded={excluded}")
    common = finalize_transition(total)["pooled"]
    result = {
        "status": "PASS", "best_epoch": 8,
        "full_horizon": {
            "precision": float(best["transition_precision"]),
            "recall": float(best["transition_recall"]),
            "f1": float(best["transition_f1"]),
            "candidate": int(best["predicted_transitions"]),
            "reference": int(best["reference_transitions"]),
            "predicted_reference_ratio": float(best["predicted_reference_ratio"]),
        },
        "common_horizon": {**common, "definition": "[t1,tM), PRE/final-MAIN-tail/POST excluded"},
        "population": {"frozen_pieces": 19, "structural_references": 71, "aligned_pairs": 70, "excluded_metadata_index": "856"},
        "metric": {"direction_aware": True, "tolerance_distinct_onsets": 1, "matching": "canonical one-to-one"},
        "best_epoch_diagnostics": {
            "train_loss": float(train_best["train_loss"]),
            "human_transition_f1": float(human_best["transition_f1"]),
            "state_agreement": float(best["state_agreement"]),
            "first_main_state_agreement": float(best["first_main_state_agreement"]),
            "mismatch_fraction": float(best["mismatch_fraction"]),
            "mismatch_run_mean": float(best["mismatch_run_mean"]),
            "mismatch_run_median": float(best["mismatch_run_median"]),
            "mismatch_run_p95": float(best["mismatch_run_p95"]),
            "mismatch_run_max": int(best["mismatch_run_max"]),
            **{f"recovered_within_{h}": float(best[f"recovered_within_{h}"]) for h in (1,2,4,8)},
            "pre_sync_success": float(best["pre_sync_success"]),
            "starting_on_pre_success": float(best["starting_on_pre_success"]),
            "starting_on_state_agreement": float(best["starting_on_state_agreement"]),
            "starting_on_transition_f1": float(best["starting_on_transition_f1"]),
            "predicted_n0_n1_n2": [int(best[f"predicted_n{i}"]) for i in range(3)],
            "timing_mae": None,
            "timing_unavailable_reason": best["timing_unavailable_reason"],
        },
        "piece_metrics": piece_rows,
        "stage1_regeneration": 0, "asap_test_access": 0,
    }
    (OUTPUT / "runA_common_horizon_metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    epochs = []
    for train, human, frozen in zip(train_rows, human_rows, frozen_rows, strict=True):
        epochs.append(
            f"| {train['epoch']} | {float(train['train_loss']):.5f} | {float(human['transition_f1']):.4f} | "
            f"{float(frozen['transition_precision']):.4f} | {float(frozen['transition_recall']):.4f} | "
            f"{float(frozen['transition_f1']):.4f} | {float(frozen['state_agreement']):.4f} | "
            f"{float(frozen['first_main_state_agreement']):.4f} |"
        )
    report = f"""# Run A Short Analysis

Run A completed 10 epochs; epoch 8 is the canonical best checkpoint by Frozen-PT direction-aware Transition F1.

| epoch | train loss | Human F1 | Frozen P | Frozen R | Frozen F1 | hard state agreement | first-MAIN agreement |
|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(epochs)}

## Epoch 8

- Full PRE/MAIN/final-tail/POST horizon: P={float(best['transition_precision']):.6f}, R={float(best['transition_recall']):.6f}, F1={float(best['transition_f1']):.6f}; predicted/reference ratio={float(best['predicted_reference_ratio']):.6f}.
- Common `[t1,tM)` horizon: P={common['precision']:.6f}, R={common['recall']:.6f}, F1={common['f1']:.6f}; candidate/reference={common['candidate']}/{common['reference']}.
- Count predictions N0/N1/N2: {int(best['predicted_n0']):,}/{int(best['predicted_n1']):,}/{int(best['predicted_n2']):,}.
- Hard state agreement={float(best['state_agreement']):.6f}; mismatch={float(best['mismatch_fraction']):.6f}; mismatch runs mean/median/p95/max={float(best['mismatch_run_mean']):.3f}/{float(best['mismatch_run_median']):.1f}/{float(best['mismatch_run_p95']):.1f}/{int(best['mismatch_run_max'])}.
- Recovery within 1/2/4/8 intervals={float(best['recovered_within_1']):.4f}/{float(best['recovered_within_2']):.4f}/{float(best['recovered_within_4']):.4f}/{float(best['recovered_within_8']):.4f}.
- PRE synchronization={float(best['pre_sync_success']):.6f}; first-MAIN agreement={float(best['first_main_state_agreement']):.6f}; starting-ON PRE success={float(best['starting_on_pre_success']):.6f}; starting-ON MAIN state agreement={float(best['starting_on_state_agreement']):.6f}.
- Frozen timing MAE is unavailable: {best['timing_unavailable_reason']}.

## Conclusion

The hypothesis is supported. Run A localized transitions substantially better than its early epochs and reached balanced epoch-8 P/R, but recurrent parity did not generalize as an absolute state trajectory: hard state agreement remained {float(best['state_agreement']):.3f}, near chance, and first-MAIN agreement was only {float(best['first_main_state_agreement']):.3f}. PRE cold-start synchronization did not generalize reliably, especially for starting-ON performances ({float(best['starting_on_pre_success']):.3f}).

Epoch 9-10 Frozen F1 decreased mildly from {float(best['transition_f1']):.6f} to {float(frozen_rows[-1]['transition_f1']):.6f} while train loss continued falling, consistent with mild over-specialization, not a collapse. No new inference or Stage 1 generation was performed; common-horizon evaluation reused epoch-8 candidate MIDI and the canonical 70 aligned references. ASAP test access: 0.
"""
    (OUTPUT / "RUN_A_SHORT_ANALYSIS.md").write_text(report)
    print(json.dumps({"status": "PASS", "common": common, "pairs": pairs}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
