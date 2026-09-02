#!/usr/bin/env python3
"""No-model fixed-capacity and post-note-off audit for event tokenizer v0."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_custom_event_tokenizer_oracle_audit_v0 import (
    add_transition_counts,
    atomic_json,
    class_arrays,
    load_manifests,
    now,
    transition_object,
    write_csv,
)
from src.stage2_event_tokenizer.audit import (
    inverse_sqrt_mean_one,
    oracle_aggregate,
)
from src.stage2_event_tokenizer.capacity_terminal_audit import (
    TAIL_SECONDS,
    VARIANT_CAPACITY,
    VARIANT_ORDER,
    assert_final_state_preserved,
    event_count_bin,
    nearest_rank_quantiles,
    padded_slots,
    post_noteoff_events,
    retained_events,
    terminal_coverage,
    write_variant_midi,
)
from src.stage2_event_tokenizer.tokenizer import EVENT_NAMES, STATE_NAMES, encode_performance, parse_raw_midi
from src.stage2_four_class.validation_evaluator import (
    PATTERN_COUNT,
    confusion_from_pairs,
    pattern_ids,
    pattern_metrics,
    pooled_transition_counts,
)
from third_party.PianistTransformer.src.model.pianoformer import PianoT5GemmaConfig

DEFAULT_OUTPUT = ROOT / "analysis/custom_event_tokenizer_capacity_terminal_audit_v1"
V0_ROOT = ROOT / "analysis/custom_event_tokenizer_oracle_audit_v0"
V0_METRICS = V0_ROOT / "validation_oracle_metrics.json"
DATASETS = ("MAESTRO-clean", "ASAP-train", "combined")
QUANTILES = (.90, .95, .975, .99, .995, .999)
VARIANT_LABEL = {
    "k1_last1": "K=1 last-1",
    "k2_v0_odd_even": "K=2 odd/even",
    "k2_last2": "K=2 last-2",
    "k3_last3": "K=3 last-3",
    "k4_last4": "K=4 last-4",
    "k6_last6": "K=6 last-6",
    "k8_last8": "K=8 last-8",
    "unlimited": "Unlimited",
}


@dataclass
class CapacityAccumulator:
    dataset: str
    variant: str
    intervals: int = 0
    overflow_intervals: int = 0
    effective_events: int = 0
    retained_events: int = 0
    crossings_before: Counter[str] = field(default_factory=Counter)
    crossings_retained: Counter[str] = field(default_factory=Counter)
    final_state_matches: int = 0
    slot_counts: np.ndarray | None = None

    def __post_init__(self) -> None:
        capacity = VARIANT_CAPACITY[self.variant]
        if capacity is not None:
            self.slot_counts = np.zeros((capacity, len(EVENT_NAMES)), dtype=np.int64)

    def add(self, interval) -> None:
        events = tuple(interval.all_events)
        retained = retained_events(interval, self.variant)
        self.intervals += 1
        capacity = VARIANT_CAPACITY[self.variant]
        self.overflow_intervals += int(capacity is not None and len(events) > capacity)
        self.effective_events += len(events)
        self.retained_events += len(retained)
        for event in events:
            if event.source_crossing:
                self.crossings_before[event.source_crossing] += 1
        for event in retained:
            if event.source_crossing:
                self.crossings_retained[event.source_crossing] += 1
        assert_final_state_preserved(interval, self.variant)
        self.final_state_matches += 1
        if self.slot_counts is not None:
            for slot_index, event in enumerate(padded_slots(interval, self.variant)):
                self.slot_counts[slot_index, event.event_class] += 1

    def row(self) -> dict[str, Any]:
        crossing_before = sum(self.crossings_before.values())
        crossing_retained = sum(self.crossings_retained.values())
        last_active = None
        if self.slot_counts is not None:
            last_active = 1.0 - float(self.slot_counts[-1, 0] / self.slot_counts[-1].sum())
        return {
            "dataset": self.dataset,
            "variant": self.variant,
            "capacity": VARIANT_CAPACITY[self.variant],
            "intervals": self.intervals,
            "overflow_intervals": self.overflow_intervals,
            "overflow_fraction": self.overflow_intervals / self.intervals if self.intervals else 0.0,
            "effective_events": self.effective_events,
            "retained_events": self.retained_events,
            "dropped_events": self.effective_events - self.retained_events,
            "event_retention": self.retained_events / self.effective_events if self.effective_events else 1.0,
            "off_to_on_before": self.crossings_before["DOWN"],
            "on_to_off_before": self.crossings_before["UP"],
            "off_to_on_retained": self.crossings_retained["DOWN"],
            "on_to_off_retained": self.crossings_retained["UP"],
            "crossings_before": crossing_before,
            "crossings_retained": crossing_retained,
            "dropped_crossings": crossing_before - crossing_retained,
            "crossing_retention": crossing_retained / crossing_before if crossing_before else 1.0,
            "final_state_preservation": self.final_state_matches / self.intervals if self.intervals else 1.0,
            "last_slot_non_none": last_active,
        }

    def slot_payload(self) -> dict[str, Any] | None:
        if self.slot_counts is None:
            return None
        payload: dict[str, Any] = {}
        for index, counts in enumerate(self.slot_counts):
            proportions = counts / counts.sum()
            payload[f"Slot{index + 1}"] = {
                "counts": dict(zip(EVENT_NAMES, counts.tolist())),
                "proportions": dict(zip(EVENT_NAMES, proportions.tolist())),
                "none_proportion": float(proportions[0]),
                "non_none_proportion": float(1.0 - proportions[0]),
                "candidate_weights_inv_sqrt_mean1": dict(
                    zip(EVENT_NAMES, inverse_sqrt_mean_one(counts))
                ),
            }
        return payload


@dataclass
class TerminalAccumulator:
    dataset: str
    performance_count: int = 0
    delays_by_performance: list[tuple[float, ...]] = field(default_factory=list)
    destination_counts: Counter[str] = field(default_factory=Counter)
    transition_type_counts: Counter[str] = field(default_factory=Counter)
    event_count_bins: Counter[str] = field(default_factory=Counter)
    last_destination_counts: Counter[str] = field(default_factory=Counter)

    def add(self, events) -> None:
        self.performance_count += 1
        delays = tuple(event.delay_seconds for event in events)
        self.delays_by_performance.append(delays)
        count_bin = "3+" if len(events) >= 3 else str(len(events))
        self.event_count_bins[count_bin] += 1
        for event in events:
            self.destination_counts[f"SET_{STATE_NAMES[event.destination_state]}"] += 1
            self.transition_type_counts[event.transition_type] += 1
        if events:
            self.last_destination_counts[f"SET_{STATE_NAMES[events[-1].destination_state]}"] += 1

    def delay_stats(self) -> dict[str, Any]:
        values = np.asarray(
            [delay for performance in self.delays_by_performance for delay in performance],
            dtype=np.float64,
        )
        if not len(values):
            return {"count": 0}
        return {
            "count": int(len(values)),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            **{
                name: float(np.quantile(values, level))
                for name, level in (
                    ("p25", .25), ("p50", .50), ("p75", .75), ("p90", .90),
                    ("p95", .95), ("p97p5", .975), ("p99", .99), ("p99p5", .995),
                )
            },
            "max": float(values.max()),
            "affected_performances": sum(bool(values) for values in self.delays_by_performance),
            "total_performances": self.performance_count,
        }


def log(output: Path, message: str) -> None:
    line = f"{now()} {message}"
    print(line, flush=True)
    with (output / "audit.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def structural_and_terminal(train, output: Path):
    accumulators = {
        dataset: {variant: CapacityAccumulator(dataset, variant) for variant in VARIANT_ORDER}
        for dataset in DATASETS
    }
    event_counts = {dataset: Counter() for dataset in DATASETS}
    crossing_counts = {dataset: Counter() for dataset in DATASETS}
    terminal = TerminalAccumulator("train")
    started = time.monotonic()
    for index, row in enumerate(train, 1):
        encoded = encode_performance(parse_raw_midi(row["performance_absolute_path"]))
        for dataset in (row["source"], "combined"):
            for interval in encoded.intervals:
                event_counts[dataset][len(interval.all_events)] += 1
                crossing_counts[dataset][sum(bool(event.source_crossing) for event in interval.all_events)] += 1
                for variant in VARIANT_ORDER:
                    accumulators[dataset][variant].add(interval)
        terminal.add(post_noteoff_events(encoded))
        if index == 1 or index % 50 == 0 or index == len(train):
            log(output, f"structural_terminal {index}/{len(train)} elapsed={time.monotonic()-started:.1f}s")
    rows = [accumulators[dataset][variant].row() for dataset in DATASETS for variant in VARIANT_ORDER]
    if any(row["final_state_preservation"] != 1.0 for row in rows):
        raise AssertionError("one or more capacity variants changed interval final state")
    slots = {
        dataset: {
            variant: accumulators[dataset][variant].slot_payload()
            for variant in VARIANT_ORDER if variant != "unlimited"
        }
        for dataset in DATASETS
    }
    distribution_rows = []
    distribution_summary = {}
    for dataset in DATASETS:
        total = sum(event_counts[dataset].values())
        for count in range(10):
            distribution_rows.append({"dataset": dataset, "distribution": "effective_events_per_interval", "bin": str(count), "count": event_counts[dataset][count], "proportion": event_counts[dataset][count] / total})
        tail = sum(count for value, count in event_counts[dataset].items() if value >= 10)
        distribution_rows.append({"dataset": dataset, "distribution": "effective_events_per_interval", "bin": "10+", "count": tail, "proportion": tail / total})
        for count in range(5):
            distribution_rows.append({"dataset": dataset, "distribution": "crossings_per_interval", "bin": str(count), "count": crossing_counts[dataset][count], "proportion": crossing_counts[dataset][count] / total})
        crossing_tail = sum(count for value, count in crossing_counts[dataset].items() if value >= 5)
        distribution_rows.append({"dataset": dataset, "distribution": "crossings_per_interval", "bin": "5+", "count": crossing_tail, "proportion": crossing_tail / total})
        event_quantiles = nearest_rank_quantiles(event_counts[dataset], QUANTILES)
        crossing_quantiles = nearest_rank_quantiles(crossing_counts[dataset], QUANTILES)
        distribution_summary[dataset] = {
            "event_count_quantiles_nearest_rank": {**event_quantiles, "max": max(event_counts[dataset])},
            "crossing_count_quantiles_nearest_rank": {**crossing_quantiles, "max": max(crossing_counts[dataset])},
            "fraction_crossings_gt_2": sum(count for value, count in crossing_counts[dataset].items() if value > 2) / total,
            "fraction_crossings_gt_3": sum(count for value, count in crossing_counts[dataset].items() if value > 3) / total,
            "fraction_crossings_gt_4": sum(count for value, count in crossing_counts[dataset].items() if value > 4) / total,
            "quantile_convention": "nearest-rank empirical quantile",
        }
    return rows, slots, distribution_rows, distribution_summary, terminal


def evaluate_validation(validation, output: Path, terminal: TerminalAccumulator):
    config = PianoT5GemmaConfig()
    confusion = {variant: np.zeros((4, 4), dtype=np.int64) for variant in VARIANT_ORDER}
    candidate_hist = {variant: np.zeros(PATTERN_COUNT, dtype=np.int64) for variant in VARIANT_ORDER}
    target_hist = np.zeros(PATTERN_COUNT, dtype=np.int64)
    transition_counts = {
        variant: {direction: {key: 0 for key in ("tp", "fp", "fn")} for direction in ("UP", "DOWN")}
        for variant in VARIANT_ORDER
    }
    errors = {variant: [] for variant in VARIANT_ORDER}
    started = time.monotonic()
    for index, row in enumerate(validation, 1):
        source_path = Path(row["performance_absolute_path"])
        encoded = encode_performance(parse_raw_midi(source_path))
        terminal.add(post_noteoff_events(encoded))
        source_raw, source_classes = class_arrays(source_path, config)
        target_hist += np.bincount(pattern_ids(source_classes), minlength=PATTERN_COUNT)
        target_transition = transition_object(source_path)
        key = f"{int(row['metadata_index']):04d}_{source_path.stem}.mid"
        with tempfile.TemporaryDirectory(prefix="capacity_oracle_") as directory:
            temporary = Path(directory)
            paths = {
                "unlimited": V0_ROOT / "roundtrip_unlimited" / key,
                "k2_v0_odd_even": V0_ROOT / "roundtrip_two_slot" / key,
            }
            for variant in VARIANT_ORDER:
                if variant not in paths:
                    paths[variant] = write_variant_midi(encoded, temporary / f"{variant}.mid", variant)
                candidate_raw, candidate_classes = class_arrays(paths[variant], config)
                if candidate_classes.shape != source_classes.shape:
                    raise AssertionError("capacity candidate changed official note universe")
                confusion[variant] += confusion_from_pairs(candidate_classes, source_classes)
                candidate_hist[variant] += np.bincount(pattern_ids(candidate_classes), minlength=PATTERN_COUNT)
                errors[variant].extend(np.abs(candidate_raw - source_raw).reshape(-1).tolist())
                metrics = pooled_transition_counts(transition_object(paths[variant]), target_transition)
                add_transition_counts(transition_counts[variant], metrics)
        log(output, f"validation_oracle {index}/{len(validation)} elapsed={time.monotonic()-started:.1f}s {row['performance_path']}")
    metrics = {
        variant: oracle_aggregate(confusion[variant], candidate_hist[variant], target_hist, transition_counts[variant], errors[variant])
        for variant in VARIANT_ORDER
    }
    v0 = json.loads(V0_METRICS.read_text())
    for variant, reference_name in (("unlimited", "unlimited"), ("k2_v0_odd_even", "two_slot")):
        reference = v0[reference_name]
        current = metrics[variant]
        checks = (
            (current["token_accuracy"], reference["token_accuracy"]),
            (current["macro_f1"], reference["macro_f1"]),
            (current["transition"]["pooled"]["f1"], reference["transition"]["pooled"]["f1"]),
            (current["pattern"]["js_divergence_base2"], reference["pattern"]["js_divergence_base2"]),
            (current["pattern"]["intersection"], reference["pattern"]["intersection"]),
        )
        if any(not math.isclose(first, second, rel_tol=0.0, abs_tol=1e-12) for first, second in checks):
            raise AssertionError(f"{variant} failed exact v0 metric regression")
    return metrics


def oracle_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    unlimited = metrics["unlimited"]
    rows = []
    for variant in VARIANT_ORDER:
        value = metrics[variant]
        transition = value["transition"]["pooled"]
        row = {
            "variant": variant,
            "capacity": VARIANT_CAPACITY[variant],
            "token_accuracy": value["token_accuracy"],
            "macro_f1": value["macro_f1"],
            "transition_precision": transition["precision"],
            "transition_recall": transition["recall"],
            "transition_f1": transition["f1"],
            "candidate_transitions": transition["candidate"],
            "reference_transitions": transition["reference"],
            "tp": transition["tp"], "fp": transition["fp"], "fn": transition["fn"],
            "up_precision": value["transition"]["up"]["precision"],
            "up_recall": value["transition"]["up"]["recall"],
            "up_f1": value["transition"]["up"]["f1"],
            "down_precision": value["transition"]["down"]["precision"],
            "down_recall": value["transition"]["down"]["recall"],
            "down_f1": value["transition"]["down"]["f1"],
            "js_divergence": value["pattern"]["js_divergence_base2"],
            "intersection": value["pattern"]["intersection"],
            "raw_cc64_mae_diagnostic": value["raw_cc64_mae_diagnostic"],
            "class_metrics_json": json.dumps(value["class_metrics"], sort_keys=True),
        }
        for name, getter in (
            ("accuracy_delta", lambda x: x["token_accuracy"]),
            ("macro_f1_delta", lambda x: x["macro_f1"]),
            ("transition_f1_delta", lambda x: x["transition"]["pooled"]["f1"]),
            ("transition_recall_delta", lambda x: x["transition"]["pooled"]["recall"]),
            ("js_delta", lambda x: x["pattern"]["js_divergence_base2"]),
            ("intersection_delta", lambda x: x["pattern"]["intersection"]),
        ):
            row[name] = getter(value) - getter(unlimited)
        rows.append(row)
    return rows


def terminal_payload(train_terminal: TerminalAccumulator, validation_terminal: TerminalAccumulator):
    stats = {item.dataset: item.delay_stats() for item in (train_terminal, validation_terminal)}
    coverage_rows = []
    type_rows = []
    for item in (train_terminal, validation_terminal):
        for row in terminal_coverage(item.delays_by_performance, TAIL_SECONDS):
            coverage_rows.append({"dataset": item.dataset, **row})
        for distribution, counts in (
            ("destination_state", item.destination_counts),
            ("transition_type", item.transition_type_counts),
            ("post_noteoff_event_count_per_performance", item.event_count_bins),
            ("last_destination_affected_performance", item.last_destination_counts),
        ):
            total = sum(counts.values())
            for label, count in sorted(counts.items()):
                type_rows.append({"dataset": item.dataset, "distribution": distribution, "label": label, "count": count, "proportion": count / total if total else 0.0})
    return stats, coverage_rows, type_rows


def row_by_variant(rows, variant):
    return next(row for row in rows if row["variant"] == variant)


def fmt(value: Any, digits: int = 6) -> str:
    return "N/A" if value is None else f"{float(value):.{digits}f}"


def write_reports(output: Path, payload: dict[str, Any], recommended_k: int, terminal_recommendation: str) -> None:
    train_rows = payload["capacity_train_stats"]
    oracle = payload["capacity_validation_oracle"]
    combined = {row["variant"]: row for row in train_rows if row["dataset"] == "combined"}
    oracle_by = {row["variant"]: row for row in oracle}
    slots = payload["slot_distributions"]["combined"]
    capacity_lines = [
        "| Capacity | Overflow % | Event retention % | Crossing retention % | Last-slot active % |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    oracle_lines = [
        "| Capacity | 4C Acc | Macro F1 | Trans P | Trans R | Trans F1 | JS | Intersection |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    penalty_lines = [
        "| Capacity | 4C Acc Δ | Macro F1 Δ | Trans F1 Δ | Trans Recall Δ | JS Δ | Intersection Δ |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    pareto_lines = [
        "| K | Trans F1 | Trans Recall | 4C Acc | JS | Intersection | Event retention | Crossing retention | Overflow % | Last-slot active % |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant in VARIANT_ORDER:
        train = combined[variant]; value = oracle_by[variant]
        label = VARIANT_LABEL[variant]
        last_slot = "N/A" if train["last_slot_non_none"] is None else f"{100*train['last_slot_non_none']:.6f}"
        capacity_lines.append(f"| {label} | {100*train['overflow_fraction']:.6f} | {100*train['event_retention']:.6f} | {100*train['crossing_retention']:.6f} | {last_slot} |")
        oracle_lines.append(f"| {label} | {value['token_accuracy']:.6f} | {value['macro_f1']:.6f} | {value['transition_precision']:.6f} | {value['transition_recall']:.6f} | {value['transition_f1']:.6f} | {value['js_divergence']:.6f} | {value['intersection']:.6f} |")
        penalty_lines.append(f"| {label} | {value['accuracy_delta']:+.6f} | {value['macro_f1_delta']:+.6f} | {value['transition_f1_delta']:+.6f} | {value['transition_recall_delta']:+.6f} | {value['js_delta']:+.6f} | {value['intersection_delta']:+.6f} |")
        if variant not in ("k2_v0_odd_even", "unlimited"):
            pareto_lines.append(f"| {VARIANT_CAPACITY[variant]} | {value['transition_f1']:.6f} | {value['transition_recall']:.6f} | {value['token_accuracy']:.6f} | {value['js_divergence']:.6f} | {value['intersection']:.6f} | {100*train['event_retention']:.4f}% | {100*train['crossing_retention']:.4f}% | {100*train['overflow_fraction']:.4f}% | {100*train['last_slot_non_none']:.4f}% |")
    coverage_lines = ["| Tail | Train event | Train full perf | Validation event | Validation full perf |", "| ---: | ---: | ---: | ---: | ---: |"]
    coverage = payload["terminal_tail_coverage"]
    for tail in TAIL_SECONDS:
        train = next(row for row in coverage if row["dataset"] == "train" and row["tail_seconds"] == tail)
        validation = next(row for row in coverage if row["dataset"] == "validation" and row["tail_seconds"] == tail)
        coverage_lines.append(f"| {tail:.2f}s | {100*train['event_coverage']:.4f}% | {100*train['affected_performance_full_coverage']:.4f}% | {100*validation['event_coverage']:.4f}% | {100*validation['affected_performance_full_coverage']:.4f}% |")
    event_tail = payload["event_distribution_summary"]["combined"]
    k2_old, k2_new = combined["k2_v0_odd_even"], combined["k2_last2"]
    ok2, nk2 = oracle_by["k2_v0_odd_even"], oracle_by["k2_last2"]
    recommendation = "pending final audit judgment" if not recommended_k else f"K={recommended_k} with generic last-{recommended_k}"
    terminal_text = terminal_recommendation or "pending final audit judgment"
    lines = [
        "# Custom Event Tokenizer Capacity & Terminal Audit v1", "",
        "- Core v0 representation: unchanged; original v0 artifacts: read-only.",
        "- Train: 1,170 MAESTRO-clean + 892 ASAP train; validation: 71 ASAP performances.",
        "- Training/model/optimizer/checkpoint activity: 0. ASAP test access: 0. Repedal: 0.",
        "- Fixed-K rule: retain all if n<=K, otherwise retain chronological final K.", "",
        "## Capacity structural curve", "", *capacity_lines, "",
        "## Validation oracle curve", "", *oracle_lines, "",
        "## Unlimited penalty", "", *penalty_lines, "",
        "## Pareto-like capacity view", "", *pareto_lines, "",
        "## Event/crossing tail", "",
        f"- Event-count quantiles (nearest-rank): `{event_tail['event_count_quantiles_nearest_rank']}`.",
        f"- Crossing-count quantiles: `{event_tail['crossing_count_quantiles_nearest_rank']}`.",
        f"- Fractions crossing-count >2/>3/>4: {event_tail['fraction_crossings_gt_2']:.6%} / {event_tail['fraction_crossings_gt_3']:.6%} / {event_tail['fraction_crossings_gt_4']:.6%}.", "",
        "## K2 odd/even versus generic last-2", "",
        f"- Event retention: {k2_old['event_retention']:.6%} → {k2_new['event_retention']:.6%}.",
        f"- Structural crossing retention: {k2_old['crossing_retention']:.6%} → {k2_new['crossing_retention']:.6%}.",
        f"- 4C accuracy: {ok2['token_accuracy']:.6f} → {nk2['token_accuracy']:.6f}.",
        f"- Transition P/R/F1: {ok2['transition_precision']:.6f}/{ok2['transition_recall']:.6f}/{ok2['transition_f1']:.6f} → {nk2['transition_precision']:.6f}/{nk2['transition_recall']:.6f}/{nk2['transition_f1']:.6f}.",
        f"- JS/Intersection: {ok2['js_divergence']:.6f}/{ok2['intersection']:.6f} → {nk2['js_divergence']:.6f}/{nk2['intersection']:.6f}.",
        "- Final-state preservation: 100% for both.", "",
        "## Terminal delay", "",
        f"- Train: `{payload['post_noteoff_delay_stats']['train']}`.",
        f"- Validation: `{payload['post_noteoff_delay_stats']['validation']}`.", "",
        "## Fixed-tail coverage (affected performances denominator)", "", *coverage_lines, "",
        "## Data-based recommendations", "",
        f"- Minimal sufficient capacity recommendation: **{recommendation}**.",
        f"- Terminal recommendation: **{terminal_text}**.",
        "- Same-timestamp semantics: KEEP for now, with the existing targeted limitation (2.3617% effective events on onset; 193 ambiguous cross-track/interleaved cases).", "",
        "## Final proposed configuration", "",
        "```text",
        "Core representation: KEEP",
        f"Fixed capacity: {'K = ?' if not recommended_k else f'K = {recommended_k}'}",
        f"Compression: {'pending' if not recommended_k else f'generic chronological last-{recommended_k}'}",
        "Initial state: KEEP",
        f"Terminal: {terminal_text}",
        "Same-timestamp semantics: KEEP / targeted limitation documented",
        f"Ready for model implementation: {'NO (recommendation pending)' if not recommended_k or not terminal_recommendation else 'YES, after user approves this proposed spec'}",
        "```", "",
        "No tokenizer spec implementation, model, loss, tiny overfit, training, ASAP test, or Repedal execution was started.",
    ]
    (output / "CUSTOM_EVENT_TOKENIZER_CAPACITY_TERMINAL_AUDIT_V1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    k2_lines = [
        "# K2 Odd/Even vs Generic Last-2", "",
        "| Metric | K2 v0 odd/even | K2 generic last-2 | Delta |", "| --- | ---: | ---: | ---: |",
    ]
    for label, a, b in (
        ("Event retention", k2_old["event_retention"], k2_new["event_retention"]),
        ("Crossing retention", k2_old["crossing_retention"], k2_new["crossing_retention"]),
        ("4C accuracy", ok2["token_accuracy"], nk2["token_accuracy"]),
        ("Macro F1", ok2["macro_f1"], nk2["macro_f1"]),
        ("Transition F1", ok2["transition_f1"], nk2["transition_f1"]),
        ("Transition recall", ok2["transition_recall"], nk2["transition_recall"]),
        ("JS", ok2["js_divergence"], nk2["js_divergence"]),
        ("Intersection", ok2["intersection"], nk2["intersection"]),
    ):
        k2_lines.append(f"| {label} | {a:.6f} | {b:.6f} | {b-a:+.6f} |")
    k2_lines += ["", "Both variants preserve every interval final state exactly. This audit does not modify the v0 specification."]
    (output / "K2_ODD_EVEN_VS_LAST2.md").write_text("\n".join(k2_lines) + "\n", encoding="utf-8")


def save_outputs(output: Path, payload: dict[str, Any]) -> None:
    train_rows = payload["capacity_train_stats"]
    oracle = payload["capacity_validation_oracle"]
    write_csv(output / "capacity_train_stats.csv", train_rows, list(train_rows[0]))
    write_csv(output / "capacity_validation_oracle.csv", oracle, list(oracle[0]))
    atomic_json(output / "slot_distributions_by_k.json", payload["slot_distributions"])
    write_csv(output / "event_count_distribution.csv", payload["event_count_distribution"], list(payload["event_count_distribution"][0]))
    crossing_rows = []
    combined = {row["variant"]: row for row in train_rows if row["dataset"] == "combined"}
    oracle_by = {row["variant"]: row for row in oracle}
    for variant in VARIANT_ORDER:
        crossing_rows.append({
            "variant": variant,
            "structural_exact_crossing_retention": combined[variant]["crossing_retention"],
            "canonical_transition_recall_tolerance1": oracle_by[variant]["transition_recall"],
            "difference": oracle_by[variant]["transition_recall"] - combined[variant]["crossing_retention"],
        })
    write_csv(output / "crossing_capacity_curve.csv", crossing_rows, list(crossing_rows[0]))
    atomic_json(output / "post_noteoff_delay_stats.json", payload["post_noteoff_delay_stats"])
    write_csv(output / "terminal_tail_coverage.csv", payload["terminal_tail_coverage"], list(payload["terminal_tail_coverage"][0]))
    write_csv(output / "post_noteoff_event_types.csv", payload["post_noteoff_event_types"], list(payload["post_noteoff_event_types"][0]))
    atomic_json(output / "audit_summary.json", payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--recommended-k", type=int, default=0)
    parser.add_argument("--terminal-recommendation", default="")
    args = parser.parse_args()
    output = args.output_root
    if args.report_only:
        payload = json.loads((output / "audit_summary.json").read_text())
        write_reports(output, payload, args.recommended_k, args.terminal_recommendation)
        atomic_json(output / "run_status.json", {
            "status": "completed",
            "train_performances": 2062, "validation_performances": 71,
            "all_capacity_final_state_preservation": True,
            "v0_baseline_metric_regression": True,
            "recommended_k": args.recommended_k,
            "terminal_recommendation": args.terminal_recommendation,
            "asap_test_access_count": 0,
            "repedal_execution_count": 0,
            "training_steps": 0,
            "last_update": now(),
        })
        print(json.dumps({"report_only": True, "recommended_k": args.recommended_k, "terminal": args.terminal_recommendation}))
        return
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    train, validation, provenance = load_manifests()
    baseline = json.loads(V0_METRICS.read_text())
    atomic_json(output / "config.json", {
        "experiment": "custom_event_tokenizer_capacity_terminal_audit_v1",
        "core_representation_changed": False,
        "variants": list(VARIANT_ORDER),
        "fixed_k_rule": "n<=K all; n>K chronological final K",
        "v0_root_read_only": str(V0_ROOT),
        "v0_baseline_sha256": __import__("hashlib").sha256(V0_METRICS.read_bytes()).hexdigest(),
        "training_steps": 0, "model_inference_steps": 0, "optimizer_steps": 0,
        "checkpoint_access_count": 0, "asap_test_access_count": 0, "repedal_execution_count": 0,
        "provenance": provenance,
    })
    train_rows, slots, distribution_rows, distribution_summary, train_terminal = structural_and_terminal(train, output)
    validation_terminal = TerminalAccumulator("validation")
    metrics = evaluate_validation(validation, output, validation_terminal)
    validation_rows = oracle_rows(metrics)
    delay_stats, coverage_rows, type_rows = terminal_payload(train_terminal, validation_terminal)
    payload = {
        "capacity_train_stats": train_rows,
        "capacity_validation_oracle": validation_rows,
        "slot_distributions": slots,
        "event_count_distribution": distribution_rows,
        "event_distribution_summary": distribution_summary,
        "post_noteoff_delay_stats": delay_stats,
        "terminal_tail_coverage": coverage_rows,
        "post_noteoff_event_types": type_rows,
        "baseline_verified": True,
        "v0_baseline": {"unlimited": baseline["unlimited"], "two_slot": baseline["two_slot"]},
        "asap_test_access_count": 0,
        "repedal_execution_count": 0,
        "training_steps": 0,
    }
    save_outputs(output, payload)
    write_reports(output, payload, 0, "")
    atomic_json(output / "run_status.json", {
        "status": "completed_pending_recommendation_finalization",
        "train_performances": len(train), "validation_performances": len(validation),
        "all_capacity_final_state_preservation": True,
        "v0_baseline_metric_regression": True,
        "asap_test_access_count": 0, "repedal_execution_count": 0, "training_steps": 0,
        "last_update": now(),
    })
    print(json.dumps({"status": "completed_pending_recommendation", "output": str(output), "test_access": 0, "training": 0}))


if __name__ == "__main__":
    main()
