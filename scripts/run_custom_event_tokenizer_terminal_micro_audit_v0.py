#!/usr/bin/env python3
"""Terminal capacity/timing micro-audit; no tokenizer implementation or model work."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_custom_event_tokenizer_oracle_audit_v0 import atomic_json, load_manifests, now, write_csv
from src.stage2_event_tokenizer.capacity_terminal_audit import post_noteoff_events
from src.stage2_event_tokenizer.terminal_micro_audit import (
    TERMINAL_CAPACITIES,
    TRANSFORMS,
    TerminalEvent,
    absolute_delays,
    assert_terminal_final_state_preserved,
    inter_event_gaps,
    numeric_summary,
    perturbation_rows,
    retain_terminal_last_k,
    transformed_summary,
)
from src.stage2_event_tokenizer.tokenizer import EVENT_NAMES, STATE_NAMES, encode_performance, parse_raw_midi

OUTPUT_DEFAULT = ROOT / "analysis/custom_event_tokenizer_terminal_micro_audit_v0"
V0_ROOT = ROOT / "analysis/custom_event_tokenizer_oracle_audit_v0"
V1_ROOT = ROOT / "analysis/custom_event_tokenizer_capacity_terminal_audit_v1"
V1_DELAY_STATS = V1_ROOT / "post_noteoff_delay_stats.json"
MAIN_K6_REFERENCE = {
    "event_retention": 0.99614549,
    "crossing_retention": 0.99663139,
    "token_accuracy": 0.99986352,
    "macro_f1": 0.99984563,
    "transition_recall": 0.99582770,
    "transition_f1": 0.99790949,
    "js_divergence": 0.00003691,
    "intersection": 0.99972880,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def log(output: Path, message: str) -> None:
    line = f"{now()} {message}"
    print(line, flush=True)
    with (output / "audit.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def terminal_sequence(path: str) -> tuple[TerminalEvent, ...]:
    encoded = encode_performance(parse_raw_midi(path))
    return tuple(
        TerminalEvent(event.delay_seconds, event.destination_state, event.transition_type)
        for event in post_noteoff_events(encoded)
    )


def collect_sequences(rows: list[dict[str, str]], dataset: str, output: Path) -> list[tuple[TerminalEvent, ...]]:
    result: list[tuple[TerminalEvent, ...]] = []
    started = time.monotonic()
    for index, row in enumerate(rows, 1):
        result.append(terminal_sequence(row["performance_absolute_path"]))
        if index == 1 or index % 100 == 0 or index == len(rows):
            log(output, f"terminal_extract {dataset} {index}/{len(rows)} elapsed={time.monotonic()-started:.1f}s")
    return result


def count_bin(value: int) -> str:
    return str(value) if value <= 9 else "10+"


def event_count_payload(dataset: str, sequences: list[tuple[TerminalEvent, ...]]):
    histogram = Counter(len(sequence) for sequence in sequences)
    affected_values = [len(sequence) for sequence in sequences if sequence]
    rows = []
    for denominator, total, allowed in (
        ("all_performances", len(sequences), lambda value: True),
        ("affected_performances", len(affected_values), lambda value: value > 0),
    ):
        for label in [str(value) for value in range(10)] + ["10+"]:
            count = sum(
                amount for value, amount in histogram.items()
                if allowed(value) and count_bin(value) == label
            )
            rows.append({
                "dataset": dataset,
                "denominator": denominator,
                "bin": label,
                "count": count,
                "proportion": count / total if total else 0.0,
            })
    return rows, {
        "all_performances": numeric_summary(len(sequence) for sequence in sequences),
        "affected_performances": numeric_summary(affected_values),
        "total_performances": len(sequences),
        "affected_performances_count": len(affected_values),
    }


def capacity_payload(dataset: str, sequences: list[tuple[TerminalEvent, ...]]):
    rows: list[dict[str, Any]] = []
    slots: dict[str, Any] = {}
    total_events = sum(map(len, sequences))
    affected = sum(bool(sequence) for sequence in sequences)
    before_crossings = sum(
        event.transition_type in {"OFF_TO_ON", "ON_TO_OFF"}
        for sequence in sequences for event in sequence
    )
    before_same_side = total_events - before_crossings
    for capacity in (*TERMINAL_CAPACITIES, None):
        retained_sequences = [retain_terminal_last_k(sequence, capacity) for sequence in sequences]
        for source, retained in zip(sequences, retained_sequences):
            assert_terminal_final_state_preserved(source, retained)
        retained_events = sum(map(len, retained_sequences))
        retained_crossings = sum(
            event.transition_type in {"OFF_TO_ON", "ON_TO_OFF"}
            for sequence in retained_sequences for event in sequence
        )
        overflow = sum(capacity is not None and len(sequence) > capacity for sequence in sequences)
        label = "unlimited" if capacity is None else f"k{capacity}"
        rows.append({
            "dataset": dataset,
            "capacity": "Unlimited" if capacity is None else capacity,
            "performances": len(sequences),
            "affected_performances": affected,
            "overflow_performances": overflow,
            "overflow_fraction_all": overflow / len(sequences),
            "overflow_fraction_affected": overflow / affected if affected else 0.0,
            "total_terminal_events": total_events,
            "retained_events": retained_events,
            "dropped_events": total_events - retained_events,
            "event_retention": retained_events / total_events if total_events else 1.0,
            "off_to_on_before": sum(event.transition_type == "OFF_TO_ON" for sequence in sequences for event in sequence),
            "on_to_off_before": sum(event.transition_type == "ON_TO_OFF" for sequence in sequences for event in sequence),
            "same_side_before": before_same_side,
            "retained_threshold_crossings": retained_crossings,
            "dropped_threshold_crossings": before_crossings - retained_crossings,
            "crossing_retention": retained_crossings / before_crossings if before_crossings else 1.0,
            "final_destination_preservation": 1.0,
            "last_slot_active_fraction_all": None if capacity is None else sum(len(sequence) >= capacity for sequence in retained_sequences) / len(sequences),
            "last_slot_active_fraction_affected": None if capacity is None or not affected else sum(len(sequence) >= capacity for sequence in retained_sequences) / affected,
        })
        if capacity is not None:
            counts = np.zeros((capacity, len(EVENT_NAMES)), dtype=np.int64)
            for sequence in retained_sequences:
                for index in range(capacity):
                    event_class = sequence[index].destination_state + 1 if index < len(sequence) else 0
                    counts[index, event_class] += 1
            slots[label] = {}
            for index, values in enumerate(counts):
                proportions = values / values.sum()
                slots[label][f"TerminalSlot{index+1}"] = {
                    "counts": dict(zip(EVENT_NAMES, values.tolist())),
                    "proportions": dict(zip(EVENT_NAMES, proportions.tolist())),
                    "active_fraction_all": float(1.0 - proportions[0]),
                    "active_fraction_affected": float((values.sum() - values[0]) / affected) if affected else 0.0,
                }
    return rows, slots


def timing_payload(dataset: str, sequences: list[tuple[TerminalEvent, ...]]):
    absolute = [value for sequence in sequences for value in absolute_delays(sequence)]
    gaps = [value for sequence in sequences for value in inter_event_gaps(sequence)]
    if any(value < 0 for value in absolute + gaps):
        raise AssertionError("negative terminal timing target")
    comparison = []
    coordinate_values = {"absolute": absolute, "gap": gaps}
    stats = {
        "absolute": {"overall": numeric_summary(absolute)},
        "gap": {
            "overall": numeric_summary(gaps),
            "zero_gap_count": sum(value == 0.0 for value in gaps),
            "zero_gap_fraction": sum(value == 0.0 for value in gaps) / len(gaps) if gaps else 0.0,
        },
    }
    for coordinate, values in coordinate_values.items():
        for transform in TRANSFORMS:
            row = {"dataset": dataset, "coordinate": coordinate, "transform": transform}
            row.update(transformed_summary(values, transform))
            comparison.append(row)
    for capacity in (3, 4, 6):
        for coordinate in ("absolute", "gap"):
            slot_values: list[list[float]] = [[] for _ in range(capacity)]
            for sequence in sequences:
                retained = retain_terminal_last_k(sequence, capacity)
                values = absolute_delays(retained) if coordinate == "absolute" else inter_event_gaps(retained)
                for index, value in enumerate(values):
                    slot_values[index].append(value)
            stats[coordinate][f"slot_conditioned_k{capacity}"] = {
                f"TerminalSlot{index+1}": numeric_summary(values)
                for index, values in enumerate(slot_values)
            }
    return stats, comparison


def conditioned_rows(dataset: str, sequences: list[tuple[TerminalEvent, ...]], key: str):
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for sequence in sequences:
        abs_values = absolute_delays(sequence)
        gap_values = inter_event_gaps(sequence)
        for event, absolute, gap in zip(sequence, abs_values, gap_values):
            label = f"SET_{STATE_NAMES[event.destination_state]}" if key == "class" else event.transition_type
            grouped[(label, "absolute")].append(absolute)
            grouped[(label, "gap")].append(gap)
    rows = []
    for (label, coordinate), values in sorted(grouped.items()):
        row = {"dataset": dataset, "label": label, "coordinate": coordinate}
        row.update(numeric_summary(values))
        rows.append(row)
    return rows


def write_report(output: Path, payload: Mapping[str, Any], recommendation: Mapping[str, str | int]) -> None:
    capacity = payload["terminal_capacity_curve"]
    timing = payload["delay_transform_comparison"]
    lines = [
        "# Custom Event Tokenizer Terminal Micro-Audit v0", "",
        "- Scope: post-latest-note-off effective four-state events only.",
        "- Main representation fixed: K=6, chronological last-6; no main-K resweep.",
        "- Original MIDI raw CC64 provenance; no Pedal1-4 target reconstruction.",
        "- Model/tokenizer implementation/training: 0. ASAP test access: 0. Repedal: 0.", "",
        "## Terminal event-count distribution", "",
    ]
    for dataset in ("train", "validation"):
        summary = payload["terminal_event_count_summary"][dataset]
        lines.append(f"- {dataset}: all={summary['all_performances']}; affected={summary['affected_performances']}")
    lines += ["", "## Terminal capacity curve", "",
              "| Dataset | K_T | Overflow affected % | Event retention % | Crossing retention % | Final state % | Last-slot active all % |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in capacity:
        last_slot = (
            "N/A" if row["last_slot_active_fraction_all"] is None
            else f"{100 * row['last_slot_active_fraction_all']:.4f}"
        )
        lines.append(
            f"| {row['dataset']} | {row['capacity']} | {100*row['overflow_fraction_affected']:.4f} | "
            f"{100*row['event_retention']:.4f} | {100*row['crossing_retention']:.4f} | "
            f"{100*row['final_destination_preservation']:.4f} | {last_slot} |"
        )
    lines += ["", "## Timing representation", "",
              "| Dataset | Coordinate | Transform | Median | p95 | p99 | Max | Dynamic range | Invertible |",
              "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for row in timing:
        lines.append(
            f"| {row['dataset']} | {row['coordinate']} | {row['transform']} | {row['p50']:.6f} | "
            f"{row['p95']:.6f} | {row['p99']:.6f} | {row['max']:.6f} | {row['dynamic_range']:.6f} | Yes |"
        )
    train_abs = payload["absolute_delay_stats"]["train"]["overall"]
    train_gap = payload["inter_event_gap_stats"]["train"]["overall"]
    lines += [
        "", "## Findings", "",
        f"1. Terminal counts are long-tailed; train max count is {int(payload['terminal_event_count_summary']['train']['all_performances']['max'])}.",
        f"2. Absolute raw median/p99/max: {train_abs['p50']:.6f}/{train_abs['p99']:.6f}/{train_abs['max']:.6f}s.",
        f"3. Gap raw median/p99/max: {train_gap['p50']:.6f}/{train_gap['p99']:.6f}/{train_gap['max']:.6f}s.",
        "4. Gap coordinates structurally preserve ordering by cumulative sum; absolute coordinates require sorting independently predicted delays.",
        "5. Perturbation results are conditioning diagnostics only, not predictions of model accuracy.",
        "6. Existing same-timestamp limitation remains: ~2.3617% effective events on onset; 193 cross-track/interleaved ambiguous cases.",
        "", "## Recommended terminal representation", "", "```text",
        "Terminal anchor:", "latest note-off", "",
        "Terminal capacity:", f"K_T = {recommendation.get('capacity', '?')}", "",
        "Capacity compression:", f"chronological last-{recommendation.get('capacity', 'K_T')}", "",
        "Timing coordinate:", str(recommendation.get("coordinate", "pending")), "",
        "Timing transform:", str(recommendation.get("transform", "pending")), "",
        "Event vocabulary:", "NONE / SET_ZERO / SET_LOW / SET_HALF / SET_FULL", "",
        "Inference ordering implication:", str(recommendation.get("ordering", "pending")), "",
        "Reason:", str(recommendation.get("reason", "pending final audit judgment")), "```", "",
        "## Custom Event Tokenizer Proposed Final Spec", "", "```text",
        "Initial state: state immediately before first distinct onset",
        "Main interval: distinct-onset [t_i,t_{i+1})",
        "Main slots: K = 6", "Main compression: chronological last-6",
        "Main timing: exact tau in [0,1)",
        "Event vocabulary: NONE / SET_ZERO / SET_LOW / SET_HALF / SET_FULL",
        "Terminal anchor: latest note-off", f"Terminal slots: K_T = {recommendation.get('capacity', '?')}",
        f"Terminal compression: chronological last-{recommendation.get('capacity', 'K_T')}",
        f"Terminal timing coordinate: {recommendation.get('coordinate', 'pending')}",
        f"Terminal timing transform: {recommendation.get('transform', 'pending')}",
        "Same-timestamp semantics: keep v0 deterministic semantics; documented targeted limitation",
        f"Model implementation readiness: {recommendation.get('readiness', 'NO (recommendation pending)')}", "```", "",
        f"Main K=6 read-only oracle reference: `{MAIN_K6_REFERENCE}`.", "",
        "No final tokenizer implementation, model, loss, training, ASAP test, Repedal, or full oracle rerun was performed.",
    ]
    (output / "CUSTOM_EVENT_TOKENIZER_TERMINAL_MICRO_AUDIT_V0.md").write_text("\n".join(lines) + "\n")


def save_outputs(output: Path, payload: Mapping[str, Any]) -> None:
    rows = payload["terminal_event_count_distribution"]
    write_csv(output / "terminal_event_count_distribution.csv", rows, list(rows[0]))
    rows = payload["terminal_capacity_curve"]
    write_csv(output / "terminal_capacity_curve.csv", rows, list(rows[0]))
    atomic_json(output / "terminal_slot_distributions.json", payload["terminal_slot_distributions"])
    atomic_json(output / "absolute_delay_stats.json", payload["absolute_delay_stats"])
    atomic_json(output / "inter_event_gap_stats.json", payload["inter_event_gap_stats"])
    rows = payload["delay_transform_comparison"]
    write_csv(output / "delay_transform_comparison.csv", rows, list(rows[0]))
    rows = payload["delay_transform_conditioning"]
    write_csv(output / "delay_transform_conditioning.csv", rows, list(rows[0]))
    rows = payload["terminal_class_delay_stats"]
    write_csv(output / "terminal_class_delay_stats.csv", rows, list(rows[0]))
    rows = payload["terminal_transition_type_delay_stats"]
    write_csv(output / "terminal_transition_type_delay_stats.csv", rows, list(rows[0]))
    atomic_json(output / "audit_summary.json", payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--recommended-kt", type=int, default=0)
    parser.add_argument("--coordinate", default="pending")
    parser.add_argument("--transform", default="pending")
    parser.add_argument("--ordering", default="pending")
    parser.add_argument("--reason", default="pending final audit judgment")
    parser.add_argument("--readiness", default="NO (recommendation pending)")
    args = parser.parse_args()
    output = args.output_root
    recommendation = {
        "capacity": args.recommended_kt or "?", "coordinate": args.coordinate,
        "transform": args.transform, "ordering": args.ordering,
        "reason": args.reason, "readiness": args.readiness,
    }
    if args.report_only:
        payload = json.loads((output / "audit_summary.json").read_text())
        write_report(output, payload, recommendation)
        atomic_json(output / "run_status.json", {
            "status": "completed", "recommended_terminal_capacity": args.recommended_kt,
            "recommended_coordinate": args.coordinate, "recommended_transform": args.transform,
            "training_steps": 0, "model_inference_steps": 0, "full_oracle_rerun_count": 0,
            "asap_test_access_count": 0, "repedal_execution_count": 0, "last_update": now(),
        })
        return
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    train, validation, provenance = load_manifests()
    atomic_json(output / "config.json", {
        "experiment": "custom_event_tokenizer_terminal_micro_audit_v0",
        "main_representation": "immutable v0 core; main K=6 chronological last-6 fixed",
        "terminal_capacities": list(TERMINAL_CAPACITIES) + ["Unlimited"],
        "terminal_anchor": "latest non-pedal note-off; strict event time > anchor",
        "timing_coordinates": ["absolute-from-noteoff", "positive inter-event gap"],
        "timing_transforms": ["raw", "log1p", "bounded_S_1_second"],
        "v0_root_read_only": str(V0_ROOT), "v1_root_read_only": str(V1_ROOT),
        "v1_delay_stats_sha256": sha256(V1_DELAY_STATS), "provenance": provenance,
        "training_steps": 0, "optimizer_steps": 0, "model_inference_steps": 0,
        "full_oracle_rerun_count": 0, "checkpoint_access_count": 0,
        "asap_test_access_count": 0, "repedal_execution_count": 0,
    })
    train_sequences = collect_sequences(train, "train", output)
    validation_sequences = collect_sequences(validation, "validation", output)
    expected = json.loads(V1_DELAY_STATS.read_text())
    for dataset, sequences in (("train", train_sequences), ("validation", validation_sequences)):
        values = [event.delay_seconds for sequence in sequences for event in sequence]
        if len(values) != expected[dataset]["count"]:
            raise AssertionError(f"{dataset} terminal event count does not regress to v1")
        if sum(bool(sequence) for sequence in sequences) != expected[dataset]["affected_performances"]:
            raise AssertionError(f"{dataset} affected performance count does not regress to v1")
        if abs(float(np.median(values)) - expected[dataset]["p50"]) > 1e-12:
            raise AssertionError(f"{dataset} delay median does not regress to v1")
    count_rows, count_summary = [], {}
    capacity_rows, slot_payload = [], {}
    delay_rows, absolute_stats, gap_stats = [], {}, {}
    class_rows, transition_rows = [], []
    for dataset, sequences in (("train", train_sequences), ("validation", validation_sequences)):
        rows, summary = event_count_payload(dataset, sequences)
        count_rows.extend(rows); count_summary[dataset] = summary
        rows, slots = capacity_payload(dataset, sequences)
        capacity_rows.extend(rows); slot_payload[dataset] = slots
        stats, rows = timing_payload(dataset, sequences)
        absolute_stats[dataset] = stats["absolute"]
        gap_stats[dataset] = stats["gap"]
        delay_rows.extend(rows)
        class_rows.extend(conditioned_rows(dataset, sequences, "class"))
        transition_rows.extend(conditioned_rows(dataset, sequences, "transition"))
    payload = {
        "terminal_event_count_distribution": count_rows,
        "terminal_event_count_summary": count_summary,
        "terminal_capacity_curve": capacity_rows,
        "terminal_slot_distributions": slot_payload,
        "absolute_delay_stats": absolute_stats,
        "inter_event_gap_stats": gap_stats,
        "delay_transform_comparison": delay_rows,
        "delay_transform_conditioning": perturbation_rows(),
        "terminal_class_delay_stats": class_rows,
        "terminal_transition_type_delay_stats": transition_rows,
        "v1_terminal_regression_verified": True,
        "main_k6_reference": MAIN_K6_REFERENCE,
        "same_timestamp_limitation": {"effective_events_on_onset_fraction": 0.023617, "ambiguous_cases": 193},
        "training_steps": 0, "asap_test_access_count": 0, "repedal_execution_count": 0,
    }
    save_outputs(output, payload)
    write_report(output, payload, recommendation)
    atomic_json(output / "run_status.json", {
        "status": "completed_pending_recommendation_finalization",
        "train_performances": len(train), "validation_performances": len(validation),
        "v1_terminal_regression_verified": True, "all_terminal_final_state_preservation": True,
        "training_steps": 0, "model_inference_steps": 0, "full_oracle_rerun_count": 0,
        "asap_test_access_count": 0, "repedal_execution_count": 0, "last_update": now(),
    })


if __name__ == "__main__":
    main()
