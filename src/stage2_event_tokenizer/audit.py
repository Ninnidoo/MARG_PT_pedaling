"""Statistics and oracle-metric helpers for the custom event tokenizer audit."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from src.stage2_four_class.validation_evaluator import classification_metrics, pattern_metrics

from .tokenizer import EVENT_NAMES, STATE_NAMES, EncodedPerformance


def distribution(counts: Iterable[int]) -> list[float]:
    values = np.asarray(list(counts), dtype=np.float64)
    return (values / values.sum()).tolist() if values.sum() else [0.0] * len(values)


def timing_summary(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        return {"count": 0}
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "std": float(array.std()),
        **{f"p{int(level * 100):02d}": float(np.quantile(array, level)) for level in (.01, .05, .25, .5, .75, .95, .99)},
        "min": float(array.min()),
        "max": float(array.max()),
        "tau_eq_0_fraction": float(np.mean(array == 0)),
        "tau_lt_0p05_fraction": float(np.mean(array < .05)),
        "tau_gt_0p95_fraction": float(np.mean(array > .95)),
    }


def inverse_sqrt_mean_one(counts: Iterable[int]) -> list[float | None]:
    values = np.asarray(list(counts), dtype=np.float64)
    present = values > 0
    raw = np.zeros_like(values)
    raw[present] = 1.0 / np.sqrt(values[present] / values[present].sum())
    if present.any():
        raw[present] /= raw[present].mean()
    return [float(value) if include else None for value, include in zip(raw, present)]


@dataclass
class StructuralAccumulator:
    dataset: str
    performances: int = 0
    intervals: int = 0
    raw_cc64_events: int = 0
    same_state_suppressed: int = 0
    same_timestamp_collapsed: int = 0
    effective_events_all_source: int = 0
    effective_events_modeled: int = 0
    retained_events: int = 0
    dropped_events: int = 0
    initial_counts: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.int64))
    event_bins: Counter[str] = field(default_factory=Counter)
    slot_counts: np.ndarray = field(default_factory=lambda: np.zeros((2, 5), dtype=np.int64))
    crossings_before: Counter[str] = field(default_factory=Counter)
    crossings_retained: Counter[str] = field(default_factory=Counter)
    overflow_odd: int = 0
    overflow_even: int = 0
    overflow_events_before: int = 0
    overflow_events_retained: int = 0
    overflow_events_dropped: int = 0
    overflow_crossings_before: int = 0
    overflow_crossings_retained: int = 0
    final_state_checks: int = 0
    final_state_matches: int = 0
    timing_by_slot: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    timing_by_class: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    def add(self, encoded: EncodedPerformance) -> dict[str, Any]:
        self.performances += 1
        self.raw_cc64_events += len(encoded.source.raw_cc64)
        self.same_state_suppressed += encoded.same_state_suppressed_count
        self.same_timestamp_collapsed += encoded.same_timestamp_collapsed_count
        self.effective_events_all_source += len(encoded.effective_events)
        self.initial_counts[encoded.initial_state] += 1
        overflow_intervals = dropped_crossings = source_crossings = 0
        for interval in encoded.intervals:
            self.intervals += 1
            count = len(interval.all_events)
            label = str(count) if count <= 4 else "5+"
            self.event_bins[label] += 1
            retained = tuple(slot for slot in interval.slots if slot.event_class)
            self.effective_events_modeled += count
            self.retained_events += len(retained)
            self.dropped_events += count - len(retained)
            retained_ids = {id(event) for event in retained}
            for slot_index, slot in enumerate(interval.slots):
                self.slot_counts[slot_index, slot.event_class] += 1
                if slot.event_class and slot.tau is not None:
                    self.timing_by_slot[f"Slot{slot_index + 1}"].append(slot.tau)
                    self.timing_by_class[EVENT_NAMES[slot.event_class]].append(slot.tau)
            for event in interval.all_events:
                if event.source_crossing:
                    self.crossings_before[event.source_crossing] += 1
                    source_crossings += 1
                    if id(event) in retained_ids:
                        self.crossings_retained[event.source_crossing] += 1
                    else:
                        dropped_crossings += 1
            if count >= 3:
                overflow_intervals += 1
                if count % 2:
                    self.overflow_odd += 1
                else:
                    self.overflow_even += 1
                self.overflow_events_before += count
                self.overflow_events_retained += len(retained)
                self.overflow_events_dropped += count - len(retained)
                before_cross = sum(bool(event.source_crossing) for event in interval.all_events)
                retained_cross = sum(bool(event.source_crossing) for event in retained)
                self.overflow_crossings_before += before_cross
                self.overflow_crossings_retained += retained_cross
            original_end = interval.start_state
            for event in interval.all_events:
                if event.state is not None:
                    original_end = event.state
            compressed_end = interval.start_state
            for event in interval.slots:
                if event.state is not None:
                    compressed_end = event.state
            self.final_state_checks += 1
            self.final_state_matches += int(original_end == compressed_end)
        return {
            "performance": encoded.source.path,
            "intervals": len(encoded.intervals),
            "overflow_intervals": overflow_intervals,
            "overflow_fraction": overflow_intervals / len(encoded.intervals) if encoded.intervals else 0.0,
            "source_crossings": source_crossings,
            "dropped_crossings": dropped_crossings,
            "dropped_crossing_fraction": dropped_crossings / source_crossings if source_crossings else 0.0,
        }

    def payload(self) -> dict[str, Any]:
        event_counts = [self.event_bins[str(value)] for value in range(5)] + [self.event_bins["5+"]]
        crossings_before = sum(self.crossings_before.values())
        crossings_retained = sum(self.crossings_retained.values())
        return {
            "dataset": self.dataset,
            "performances": self.performances,
            "intervals": self.intervals,
            "raw_cc64_event_count": self.raw_cc64_events,
            "same_state_suppressed_count": self.same_state_suppressed,
            "same_timestamp_collapsed_count": self.same_timestamp_collapsed,
            "effective_4state_events_all_source": self.effective_events_all_source,
            "effective_events_before_slot_cap_modeled_support": self.effective_events_modeled,
            "retained_events": self.retained_events,
            "dropped_events": self.dropped_events,
            "event_retention_fraction": self.retained_events / self.effective_events_modeled if self.effective_events_modeled else 1.0,
            "event_dropped_fraction": self.dropped_events / self.effective_events_modeled if self.effective_events_modeled else 0.0,
            "initial_state_counts": dict(zip(STATE_NAMES, self.initial_counts.tolist())),
            "initial_state_proportions": dict(zip(STATE_NAMES, distribution(self.initial_counts))),
            "precompression_interval_event_counts": dict(zip(("0", "1", "2", "3", "4", "5+"), event_counts)),
            "precompression_interval_event_proportions": dict(zip(("0", "1", "2", "3", "4", "5+"), distribution(event_counts))),
            "ge3_interval_fraction": sum(event_counts[3:]) / self.intervals if self.intervals else 0.0,
            "ge4_interval_fraction": sum(event_counts[4:]) / self.intervals if self.intervals else 0.0,
            "crossings_before": dict(self.crossings_before),
            "crossings_retained": dict(self.crossings_retained),
            "crossing_retention_fraction": crossings_retained / crossings_before if crossings_before else 1.0,
            "slot_counts": {f"Slot{slot + 1}": dict(zip(EVENT_NAMES, self.slot_counts[slot].tolist())) for slot in range(2)},
            "slot_proportions": {f"Slot{slot + 1}": dict(zip(EVENT_NAMES, distribution(self.slot_counts[slot]))) for slot in range(2)},
            "slot_candidate_weights_inv_sqrt_mean1": {f"Slot{slot + 1}": dict(zip(EVENT_NAMES, inverse_sqrt_mean_one(self.slot_counts[slot]))) for slot in range(2)},
            "initial_candidate_weights_inv_sqrt_mean1": dict(zip(STATE_NAMES, inverse_sqrt_mean_one(self.initial_counts))),
            "timing": {
                "by_slot": {name: timing_summary(values) for name, values in sorted(self.timing_by_slot.items())},
                "by_event_class": {name: timing_summary(values) for name, values in sorted(self.timing_by_class.items())},
            },
            "overflow": {
                "odd_interval_count": self.overflow_odd,
                "even_interval_count": self.overflow_even,
                "events_before": self.overflow_events_before,
                "events_retained": self.overflow_events_retained,
                "events_dropped": self.overflow_events_dropped,
                "crossings_before": self.overflow_crossings_before,
                "crossings_retained": self.overflow_crossings_retained,
            },
            "final_state_preservation": self.final_state_matches / self.final_state_checks if self.final_state_checks else 1.0,
            "final_state_checks": self.final_state_checks,
        }


def oracle_aggregate(
    confusion: np.ndarray,
    candidate_patterns: np.ndarray,
    target_patterns: np.ndarray,
    transition_counts: dict[str, dict[str, int]],
    raw_absolute_errors: list[float],
) -> dict[str, Any]:
    result = classification_metrics(confusion)
    result["pattern"] = pattern_metrics(candidate_patterns, target_patterns)
    result["transition"] = transition_metrics_from_counts(transition_counts)
    errors = np.asarray(raw_absolute_errors, dtype=np.float64)
    result["raw_cc64_mae_diagnostic"] = float(errors.mean()) if len(errors) else None
    return result


def transition_metrics_from_counts(counts: dict[str, dict[str, int]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    pooled = Counter()
    for direction in ("UP", "DOWN"):
        values = counts[direction]
        tp, fp, fn = values["tp"], values["fp"], values["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result[direction.lower()] = {**values, "precision": precision, "recall": recall, "f1": f1}
        pooled.update(values)
    tp, fp, fn = pooled["tp"], pooled["fp"], pooled["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    result["pooled"] = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "candidate": tp + fp,
        "reference": tp + fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tolerance_distinct_onsets": 1,
    }
    return result
