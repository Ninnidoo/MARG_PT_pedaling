#!/usr/bin/env python3
"""Read-only structural audits for the State-Conditioned Event Model v1 design.

This script reads the frozen tokenizer caches and frozen Custom Event v0 validation
predictions.  It does not instantiate a model, run inference, or alter tokenizer or
decoder artifacts.
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUTPUT = ROOT / "analysis/custom_event_model_v1_state_conditioning_design"
CACHE_ROOT = ROOT / "analysis/custom_event_tokenizer_v1"
RAW_ROOT = ROOT / "analysis/custom_event_v0_4state_failure_audit_v1/raw_predictions"
STATE_NAMES = ("ZERO", "LOW", "HALF", "FULL")
EVENT_NAMES = ("NONE", "SET_ZERO", "SET_LOW", "SET_HALF", "SET_FULL")
EXPECTED_CACHE_ID = "3a5520155b5db1e9a1da7f8148556aa3e1da852655c9adde25d3dbbd1d966263"
EXPECTED_RAW_DIGEST = "839c7705ac51a87d722209362756ab20677ee218de751edf5c743b029fb263e0"


def safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def atomic_json(name: str, payload: Any) -> None:
    path = OUTPUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(safe(payload), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def entropy(counts: np.ndarray) -> float:
    values = np.asarray(counts, dtype=np.float64)
    values = values[values > 0]
    if not len(values):
        return 0.0
    probability = values / values.sum()
    return float(-(probability * np.log2(probability)).sum())


def conditional_entropy(matrix: np.ndarray) -> float:
    rows = np.asarray(matrix, dtype=np.float64)
    total = rows.sum()
    return float(sum(row.sum() / total * entropy(row) for row in rows if row.sum()))


def distribution_rows(matrix: np.ndarray, columns: tuple[str, ...]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for state, row in zip(STATE_NAMES, matrix, strict=True):
        total = int(row.sum())
        result[state] = {
            "count": total,
            "rates": {
                name: float(value / total) if total else None
                for name, value in zip(columns, row, strict=True)
            },
        }
    return result


def target_prestate(events: np.ndarray, interval_start: np.ndarray) -> np.ndarray:
    result = np.empty_like(events, dtype=np.int8)
    result[:, 0] = interval_start
    for slot in range(1, events.shape[1]):
        previous = events[:, slot - 1]
        result[:, slot] = np.where(previous > 0, previous - 1, result[:, slot - 1])
    return result


def add_target_cache(
    cache: Any,
    all_matrix: np.ndarray,
    active_matrix: np.ndarray,
    slot_active: np.ndarray,
) -> dict[str, int]:
    events = cache["main_event_target"].astype(np.int8)
    starts = cache["main_interval_start_state"].astype(np.int8)
    pre = target_prestate(events, starts)
    for slot in range(6):
        np.add.at(all_matrix, (pre[:, slot], events[:, slot]), 1)
        mask = events[:, slot] > 0
        np.add.at(active_matrix, (pre[mask, slot], events[mask, slot] - 1), 1)
        np.add.at(slot_active[slot], (pre[mask, slot], events[mask, slot] - 1), 1)

    final = pre[:, -1].copy()
    final = np.where(events[:, -1] > 0, events[:, -1] - 1, final)
    if len(final) > 1 and not np.array_equal(final[:-1], starts[1:]):
        raise AssertionError("main target rollout disagrees with cached interval-start state")

    terminal_noops = 0
    current = int(final[-1])
    for event in cache["terminal_event_target"].astype(np.int8):
        if int(event) == 0:
            break
        destination = int(event) - 1
        terminal_noops += int(destination == current)
        current = destination
    return {
        "main_noops": int(np.trace(active_matrix) - getattr(add_target_cache, "_trace_before", 0)),
        "terminal_noops": terminal_noops,
    }


def target_audit(entries: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    accumulators: dict[str, dict[str, np.ndarray | int]] = {}
    for split in ("train", "validation"):
        accumulators[split] = {
            "all": np.zeros((4, 5), dtype=np.int64),
            "active": np.zeros((4, 4), dtype=np.int64),
            "slot": np.zeros((6, 4, 4), dtype=np.int64),
            "terminal_noops": 0,
        }
    for entry in entries:
        split = str(entry["split"])
        acc = accumulators[split]
        active_before = int(np.trace(acc["active"]))
        with np.load(CACHE_ROOT / entry["cache_file"], allow_pickle=False) as cache:
            events = cache["main_event_target"].astype(np.int8)
            starts = cache["main_interval_start_state"].astype(np.int8)
            pre = target_prestate(events, starts)
            for slot in range(6):
                np.add.at(acc["all"], (pre[:, slot], events[:, slot]), 1)
                mask = events[:, slot] > 0
                np.add.at(acc["active"], (pre[mask, slot], events[mask, slot] - 1), 1)
                np.add.at(acc["slot"][slot], (pre[mask, slot], events[mask, slot] - 1), 1)
            final = np.where(events[:, -1] > 0, events[:, -1] - 1, pre[:, -1])
            if len(final) > 1 and not np.array_equal(final[:-1], starts[1:]):
                raise AssertionError("main target rollout disagrees with cached interval-start state")
            current = int(final[-1])
            for event in cache["terminal_event_target"].astype(np.int8):
                if int(event) == 0:
                    break
                destination = int(event) - 1
                acc["terminal_noops"] += int(destination == current)
                current = destination
        if int(np.trace(acc["active"])) < active_before:
            raise AssertionError("impossible no-op count regression")

    def payload(all_matrix: np.ndarray, active: np.ndarray, slot: np.ndarray, terminal_noops: int) -> dict[str, Any]:
        pair_counts = {
            f"{source}->{destination}": int(active[i, j])
            for i, source in enumerate(STATE_NAMES)
            for j, destination in enumerate(STATE_NAMES)
        }
        cross = int(active[:2, 2:].sum() + active[2:, :2].sum())
        same_side = int(active[0, 1] + active[1, 0] + active[2, 3] + active[3, 2])
        noops = int(np.trace(active))
        active_count = int(active.sum())
        event_count = int(all_matrix.sum())
        h_active = entropy(active.sum(axis=0))
        h_active_given_state = conditional_entropy(active)
        h_all = entropy(all_matrix.sum(axis=0))
        h_all_given_state = conditional_entropy(all_matrix)
        return {
            "axes": {"rows": list(STATE_NAMES), "all_columns": list(EVENT_NAMES), "active_columns": list(STATE_NAMES)},
            "all_event_matrix": all_matrix,
            "active_destination_matrix": active,
            "active_destination_matrix_by_slot": slot,
            "all_event_distribution_by_prestate": distribution_rows(all_matrix, EVENT_NAMES),
            "active_destination_distribution_by_prestate": distribution_rows(active, STATE_NAMES),
            "pair_counts": pair_counts,
            "counts": {
                "all_main_slot_positions": event_count,
                "none": int(all_matrix[:, 0].sum()),
                "active_main_events": active_count,
                "main_same_state_targets": noops,
                "terminal_same_state_targets": int(terminal_noops),
                "main_plus_terminal_same_state_targets": noops + int(terminal_noops),
                "same_side_depth_changes": same_side,
                "cross_binary_threshold_destinations": cross,
            },
            "rates": {
                "none": float(all_matrix[:, 0].sum() / event_count),
                "main_same_state_given_active": float(noops / active_count),
                "same_side_depth_change_given_active": float(same_side / active_count),
                "cross_binary_threshold_given_active": float(cross / active_count),
            },
            "entropy_bits": {
                "destination_given_event_active": h_active,
                "destination_given_event_active_and_current_state": h_active_given_state,
                "active_reduction": h_active - h_active_given_state,
                "active_reduction_fraction": (h_active - h_active_given_state) / h_active,
                "event_class_including_none": h_all,
                "event_class_including_none_given_current_state": h_all_given_state,
                "including_none_reduction": h_all - h_all_given_state,
                "including_none_reduction_fraction": (h_all - h_all_given_state) / h_all,
            },
        }

    result: dict[str, Any] = {
        split: payload(acc["all"], acc["active"], acc["slot"], int(acc["terminal_noops"]))
        for split, acc in accumulators.items()
    }
    combined_all = accumulators["train"]["all"] + accumulators["validation"]["all"]
    combined_active = accumulators["train"]["active"] + accumulators["validation"]["active"]
    combined_slot = accumulators["train"]["slot"] + accumulators["validation"]["slot"]
    combined_terminal = int(accumulators["train"]["terminal_noops"]) + int(accumulators["validation"]["terminal_noops"])
    result["combined"] = payload(combined_all, combined_active, combined_slot, combined_terminal)
    result["provenance"] = {
        "cache_id": EXPECTED_CACHE_ID,
        "train_performances": sum(entry["split"] == "train" for entry in entries),
        "validation_performances": sum(entry["split"] == "validation" for entry in entries),
        "state_semantics": "pre-state immediately before each target slot in frozen canonical slot order",
        "none_handling": "NONE leaves state unchanged; all six cached positions are included in the five-way distribution",
    }
    entropy_payload = {
        "unit": "bits (base-2 Shannon entropy)",
        "definitions": {
            "active": "D is destination class conditional on Main event target != NONE",
            "including_none": "Y is the five-way Main target NONE/SET_ZERO/SET_LOW/SET_HALF/SET_FULL",
            "conditional": "empirical pre-state-weighted conditional entropy",
        },
        **{split: result[split]["entropy_bits"] for split in ("train", "validation", "combined")},
    }
    return result, entropy_payload


def confusion_metrics(matrix: np.ndarray) -> dict[str, Any]:
    total = int(matrix.sum())
    exact = int(np.trace(matrix))
    binary = int(matrix[:2, :2].sum() + matrix[2:, 2:].sum())
    return {
        "count": total,
        "exact_match_count": exact,
        "exact_match_rate": exact / total,
        "exact_mismatch_rate": 1.0 - exact / total,
        "binary_off_on_match_count": binary,
        "binary_off_on_match_rate": binary / total,
        "binary_off_on_mismatch_rate": 1.0 - binary / total,
        "confusion_gt_rows_predicted_columns": matrix,
    }


def quantiles(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "mean": float(array.mean()) if len(array) else None,
        **{
            name: float(np.quantile(array, level)) if len(array) else None
            for name, level in (("p50", .50), ("p75", .75), ("p90", .90), ("p95", .95), ("p99", .99))
        },
        "max": float(array.max()) if len(array) else None,
    }


def validation_exposure_audit(raw_manifest: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    # Imported lazily because it loads the canonical MIDI tempo-map helper.
    from scripts.run_custom_event_model_v0_canonical_val_inference_v1 import timeline_from_cache

    raw_confusion = np.zeros((4, 4), dtype=np.int64)
    raw_by_slot = np.zeros((6, 4, 4), dtype=np.int64)
    chronological_confusion = np.zeros((4, 4), dtype=np.int64)
    reorder_count = 0
    effective_count = 0
    collapsed_count = 0
    run_event_counts: list[float] = []
    run_onset_spans: list[float] = []
    run_seconds: list[float] = []

    for record in raw_manifest["records"]:
        raw_path = Path(record["raw_prediction_file"])
        cache_path = Path(record["cache_file"])
        with np.load(raw_path, allow_pickle=False) as raw, np.load(cache_path, allow_pickle=False) as cache:
            events = raw["main_event_targets"].astype(np.int8)
            starts = cache["main_interval_start_state"].astype(np.int8)
            gt_pre = target_prestate(events, starts)
            classes = raw["main_event_logits"].argmax(-1).astype(np.int8)
            timing = raw["main_timing_predictions"].astype(np.float64)

            predicted_current = int(raw["initial_logits"].argmax())
            for interval_index in range(len(events)):
                row = classes[interval_index]
                zero = np.flatnonzero(row == 0)
                active_count = int(zero[0]) if len(zero) else 6
                for slot in range(active_count):
                    oracle = int(gt_pre[interval_index, slot])
                    raw_confusion[oracle, predicted_current] += 1
                    raw_by_slot[slot, oracle, predicted_current] += 1
                    predicted_current = int(row[slot]) - 1

            timeline, _, _ = timeline_from_cache(cache)
            predicted_current = int(raw["initial_logits"].argmax())
            mismatch_positions: list[tuple[int, float, bool]] = []
            for interval_index, interval in enumerate(timeline.main_intervals):
                row = classes[interval_index]
                zero = np.flatnonzero(row == 0)
                active_count = int(zero[0]) if len(zero) else 6
                raw_tau = timing[interval_index, :active_count]
                reorder_count += int(np.sum(raw_tau[1:] < raw_tau[:-1]))
                pairs = sorted(
                    (
                        min(1.0, max(0.0, float(raw_tau[slot]))),
                        slot,
                        int(row[slot]) - 1,
                    )
                    for slot in range(active_count)
                )
                position = 0
                while position < len(pairs):
                    end = position + 1
                    while end < len(pairs) and pairs[end][0] == pairs[position][0]:
                        end += 1
                    collapsed_count += end - position - 1
                    tau, _, destination = pairs[end - 1]
                    oracle = int(starts[interval_index])
                    target_row = events[interval_index]
                    target_tau = raw["main_timing_targets"][interval_index]
                    target_active = np.flatnonzero(target_row == 0)
                    target_count = int(target_active[0]) if len(target_active) else 6
                    target_pairs = sorted(
                        (float(target_tau[slot]), slot, int(target_row[slot]) - 1)
                        for slot in range(target_count)
                    )
                    target_position = 0
                    while target_position < len(target_pairs):
                        target_end = target_position + 1
                        while target_end < len(target_pairs) and target_pairs[target_end][0] == target_pairs[target_position][0]:
                            target_end += 1
                        if target_pairs[target_position][0] >= tau:
                            break
                        oracle = target_pairs[target_end - 1][2]
                        target_position = target_end
                    chronological_confusion[oracle, predicted_current] += 1
                    seconds = interval.left_time + tau * (interval.right_time - interval.left_time)
                    mismatch_positions.append((interval_index, seconds, oracle != predicted_current))
                    effective_count += 1
                    predicted_current = destination
                    position = end

            index = 0
            while index < len(mismatch_positions):
                if not mismatch_positions[index][2]:
                    index += 1
                    continue
                end = index + 1
                while end < len(mismatch_positions) and mismatch_positions[end][2]:
                    end += 1
                first = mismatch_positions[index]
                last = mismatch_positions[end - 1]
                run_event_counts.append(float(end - index))
                run_onset_spans.append(float(last[0] - first[0]))
                run_seconds.append(float(last[1] - first[1]))
                index = end

    raw_payload = confusion_metrics(raw_confusion)
    raw_payload["by_original_slot"] = {
        f"slot{slot + 1}": confusion_metrics(raw_by_slot[slot]) for slot in range(6)
    }
    chronological_payload = confusion_metrics(chronological_confusion)
    mismatch_payload = {
        "provenance": {
            "raw_prediction_digest": EXPECTED_RAW_DIGEST,
            "validation_performances": 71,
            "modeled_onsets": 272927,
            "state_axes": {"rows": list(STATE_NAMES), "columns": list(STATE_NAMES)},
        },
        "raw_prefix_slot_order": {
            "definition": "At every frozen-v0 prefix-active predicted Main slot, compare the predicted rollout state with the target rollout pre-state attached to the same interval/slot; both rollouts use canonical slot order.",
            **raw_payload,
        },
        "decoder_chronological_order": {
            "definition": "After first-NONE, tau clip/sort, and exact-same-time collapse, compare predicted decoder pre-state with target state after target events at strictly earlier tau in the same interval.",
            "same-time_target_tie_rule": "strictly earlier target tau only; this avoids assigning an arbitrary order across target/prediction sources at equal time",
            **chronological_payload,
        },
    }
    exposure_payload = {
        "provenance": mismatch_payload["provenance"],
        "scope": "frozen Custom Event v0 validation predictions; descriptive state-source exposure only",
        "decoder_effective_prediction_positions": effective_count,
        "raw_tau_order_violations": reorder_count,
        "raw_tau_order_violation_rate_per_prefix_active_position": reorder_count / int(raw_confusion.sum()),
        "same_time_collapsed_predictions": collapsed_count,
        "mismatched_run_count": len(run_event_counts),
        "mismatch_run_quantiles": {
            "subsequent_event_decisions_including_first_wrong_position": quantiles(run_event_counts),
            "onset_interval_span_end_minus_start": quantiles(run_onset_spans),
            "seconds_end_minus_start": quantiles(run_seconds),
        },
        "interpretation_guardrail": "Runs measure how long the conditioning source would remain different at observed frozen-v0 decision positions. They do not simulate v1 logits or performance.",
    }
    return mismatch_payload, exposure_payload


def main() -> None:
    cache_manifest = json.loads((CACHE_ROOT / "cache_manifest.json").read_text(encoding="utf-8"))
    raw_manifest = json.loads((RAW_ROOT / "manifest.json").read_text(encoding="utf-8"))
    if cache_manifest["cache_id"] != EXPECTED_CACHE_ID:
        raise RuntimeError("frozen tokenizer cache identity changed")
    if raw_manifest["aggregate_prediction_digest"] != EXPECTED_RAW_DIGEST:
        raise RuntimeError("frozen raw-prediction identity changed")
    if (cache_manifest["train_count"], cache_manifest["validation_count"]) != (2062, 71):
        raise RuntimeError("frozen train/validation universe changed")
    if (raw_manifest["validation_performances"], raw_manifest["modeled_onsets"]) != (71, 272927):
        raise RuntimeError("frozen prediction universe changed")

    transitions, entropy_payload = target_audit(cache_manifest["entries"])
    mismatch, exposure = validation_exposure_audit(raw_manifest)
    atomic_json("state_transition_matrix.json", transitions)
    atomic_json("conditional_entropy.json", entropy_payload)
    atomic_json("prestate_mismatch_audit.json", mismatch)
    atomic_json("exposure_risk_summary.json", exposure)
    print(
        "DESIGN_AUDITS_COMPLETE "
        f"train_active={transitions['train']['counts']['active_main_events']} "
        f"validation_active={transitions['validation']['counts']['active_main_events']} "
        f"raw_prefix_exact={mismatch['raw_prefix_slot_order']['exact_match_rate']:.9f}"
    )


if __name__ == "__main__":
    main()
