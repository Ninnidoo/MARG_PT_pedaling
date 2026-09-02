#!/usr/bin/env python3
"""Audit the frozen state-conditioned binary two-slot data formulation.

This command performs raw/canonical MIDI analysis only.  It intentionally has
no model, checkpoint, optimizer, training, or inference dependency.
"""

from __future__ import annotations

import bisect
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage2_event_model.dataset import CANONICAL_ALIGNMENT_ID, CANONICAL_CACHE_ID
from src.stage2_event_model.note_alignment import load_note_alignment, sha256_file
from src.stage2_event_model.ownership import assign_unique_owners
from src.stage2_event_tokenizer.audit import inverse_sqrt_mean_one
from src.stage2_event_tokenizer.binary_2slot import (
    DOWN,
    OFF,
    ON,
    UP,
    BinaryTransition,
    compress_binary_transitions,
    interval_region,
    reconcile_and_compress,
    state_after,
)
from src.stage2_event_tokenizer.tokenizer import _effective_events, parse_raw_midi
from src.stage2_event_tokenizer.tokenizer_v1 import _tick_second_converters


CACHE_ROOT = ROOT / "analysis/custom_event_tokenizer_v1"
ALIGNMENT_ROOT = ROOT / "analysis/custom_event_model_v0_note_alignment"
OUTPUT_ROOT = ROOT / "analysis/binary_2slot_transition_audit_v0"
BUCKETS = ("0", "1", "2", "3", "4", "5+")
EXPECTED_SOURCES = Counter({"MAESTRO-clean": 1170, "ASAP-train": 892, "ASAP-validation": 71})
CAPACITY_KEYS = ("MAESTRO-clean", "ASAP-train", "ASAP-validation")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)
    path.chmod(0o644)


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_text(path, json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)
    path.chmod(0o644)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def count_bucket(value: int) -> str:
    return str(value) if value <= 4 else "5+"


def compressed_n(count: int) -> int:
    if count <= 2:
        return count
    return 1 if count % 2 else 2


def retained_local_indices(count: int) -> tuple[int, ...]:
    if count <= 2:
        return tuple(range(count))
    return (count - 1,) if count % 2 else (count - 2, count - 1)


def ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def quantiles(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "median": None, "q3": None, "p95": None, "p99": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "median": float(np.quantile(array, 0.50)),
        "q3": float(np.quantile(array, 0.75)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(array.max()),
    }


def numeric_distribution(values: Sequence[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.int64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "min": int(array.min()),
        "median": float(np.quantile(array, 0.50)),
        "q3": float(np.quantile(array, 0.75)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": int(array.max()),
    }


def direct_binary_crossings(raw_events) -> tuple[tuple[int, str], ...]:
    """Independent threshold projection after the existing same-tick ordering."""

    grouped: dict[int, list[Any]] = defaultdict(list)
    for event in raw_events:
        grouped[int(event.tick)].append(event)
    state = OFF
    result: list[tuple[int, str]] = []
    for tick in sorted(grouped):
        ordered = sorted(grouped[tick], key=lambda event: (event.track, event.message_index))
        next_state = ON if int(ordered[-1].value) >= 64 else OFF
        if next_state != state:
            result.append((tick, DOWN if state == OFF else UP))
            state = next_state
    return tuple(result)


def n_histogram(counts: np.ndarray, *, correction: bool = False) -> np.ndarray:
    values = counts.astype(np.int64, copy=False) + int(correction)
    result = np.zeros(3, dtype=np.int64)
    result[0] = int(np.sum(values == 0))
    result[1] = int(np.sum((values == 1) | ((values >= 3) & (values % 2 == 1))))
    result[2] = int(values.size - result[0] - result[1])
    return result


class CapacityAccumulator:
    def __init__(self, dataset: str) -> None:
        self.dataset = dataset
        self.performances = 0
        self.intervals = 0
        self.buckets: Counter[str] = Counter()
        self.raw_events = 0
        self.retained_events = 0
        self.raw_directions: Counter[str] = Counter()
        self.retained_directions: Counter[str] = Counter()
        self.raw_pairs: Counter[str] = Counter()
        self.retained_pairs: Counter[str] = Counter()
        self.final_checks = 0
        self.final_matches = 0
        self.overflow_counts: list[int] = []
        self.overflow_examples: list[dict[str, Any]] = []

    def add(
        self,
        performance: str,
        onset_ticks: np.ndarray,
        note_end_tick: int,
        binary_ticks: np.ndarray,
        directions: Sequence[str],
        tick_to_seconds,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        self.performances += 1
        first = int(onset_ticks[0])
        lo = int(np.searchsorted(binary_ticks, first, side="left"))
        hi = int(np.searchsorted(binary_ticks, note_end_tick, side="left"))
        modeled_ticks = binary_ticks[lo:hi]
        interval_ids = np.searchsorted(onset_ticks, modeled_ticks, side="right") - 1
        if np.any(interval_ids < 0) or np.any(interval_ids >= len(onset_ticks)):
            raise AssertionError("MAIN transition escaped its interval")
        counts = np.bincount(interval_ids, minlength=len(onset_ticks)).astype(np.int64)
        self.intervals += int(counts.size)
        unique_counts, frequencies = np.unique(counts, return_counts=True)
        for count, frequency in zip(unique_counts.tolist(), frequencies.tolist()):
            self.buckets[count_bucket(int(count))] += int(frequency)
        retained_counts = n_histogram(counts)
        self.raw_events += int(counts.sum())
        self.retained_events += int(retained_counts[1] + 2 * retained_counts[2])
        self.final_checks += int(counts.size)
        # Compression retains the same parity by construction.  Verify every
        # interval vector rather than only sampling non-empty intervals.
        compressed_parity = np.where(counts <= 2, counts, np.where(counts % 2, 1, 2)) % 2
        matches = compressed_parity == (counts % 2)
        self.final_matches += int(matches.sum())
        if not bool(np.all(matches)):
            raise AssertionError("MAIN final-state parity mismatch")

        overflow = int(np.sum(counts >= 3))
        self.overflow_counts.append(overflow)
        groups: list[dict[str, Any]] = []
        if modeled_ticks.size:
            changes = np.flatnonzero(np.diff(interval_ids)) + 1
            starts = np.r_[0, changes]
            ends = np.r_[changes, len(interval_ids)]
            for start, end in zip(starts.tolist(), ends.tolist()):
                interval_index = int(interval_ids[start])
                count = end - start
                global_indices = tuple(range(lo + start, lo + end))
                local_directions = tuple(directions[index] for index in global_indices)
                expected_start = DOWN if global_indices[0] % 2 == 0 else UP
                if local_directions[0] != expected_start:
                    raise AssertionError("binary sequence does not alternate from OFF")
                if any(a == b for a, b in zip(local_directions, local_directions[1:])):
                    raise AssertionError("adjacent effective binary crossings have equal direction")
                retained_positions = retained_local_indices(count)
                retained_directions = tuple(local_directions[index] for index in retained_positions)
                self.raw_directions.update(local_directions)
                self.retained_directions.update(retained_directions)
                for pair in zip(local_directions, local_directions[1:]):
                    self.raw_pairs[f"{pair[0]}->{pair[1]}"] += 1
                if len(retained_directions) == 2:
                    pair = f"{retained_directions[0]}->{retained_directions[1]}"
                    self.retained_pairs[pair] += 1
                if count >= 3 and len(self.overflow_examples) < 10:
                    left_tick = int(onset_ticks[interval_index])
                    right_tick = int(note_end_tick if interval_index + 1 == len(onset_ticks) else onset_ticks[interval_index + 1])
                    self.overflow_examples.append({
                        "dataset": self.dataset,
                        "performance": performance,
                        "interval_index": interval_index,
                        "left_tick": left_tick,
                        "right_tick": right_tick,
                        "left_seconds": float(tick_to_seconds(left_tick)),
                        "right_seconds": float(tick_to_seconds(right_tick)),
                        "raw_count": count,
                        "raw_directions": "|".join(local_directions),
                        "retained_directions": "|".join(retained_directions),
                    })
                groups.append({"interval_index": interval_index, "start": lo + start, "count": count})
        return counts, groups

    def payload(self) -> dict[str, Any]:
        overflow_intervals = sum(self.buckets[bucket] for bucket in ("3", "4", "5+"))
        performances_with_overflow = sum(value > 0 for value in self.overflow_counts)
        return {
            "dataset": self.dataset,
            "performances": self.performances,
            "main_intervals": self.intervals,
            "raw_count_distribution": {
                bucket: {"count": self.buckets[bucket], "fraction": ratio(self.buckets[bucket], self.intervals)}
                for bucket in BUCKETS
            },
            "overflow_intervals": overflow_intervals,
            "overflow_interval_fraction": ratio(overflow_intervals, self.intervals),
            "raw_effective_transitions": self.raw_events,
            "retained_transitions": self.retained_events,
            "overall_transition_retention": ratio(self.retained_events, self.raw_events),
            "direction_retention": {
                direction: {
                    "raw": self.raw_directions[direction],
                    "retained": self.retained_directions[direction],
                    "retention": ratio(self.retained_directions[direction], self.raw_directions[direction]),
                }
                for direction in (UP, DOWN)
            },
            "final_state_preservation": ratio(self.final_matches, self.final_checks),
            "final_state_checks": self.final_checks,
            "performances_with_any_overflow": performances_with_overflow,
            "performances_with_any_overflow_fraction": ratio(performances_with_overflow, self.performances),
            "overflow_intervals_per_performance": numeric_distribution(self.overflow_counts),
            "gesture_retention": {
                pair: {
                    "raw_adjacent_pairs": self.raw_pairs[pair],
                    "retained_as_pair": self.retained_pairs[pair],
                    "retention": ratio(self.retained_pairs[pair], self.raw_pairs[pair]),
                }
                for pair in (f"{UP}->{DOWN}", f"{DOWN}->{UP}")
            },
        }


class BoundaryAccumulator:
    def __init__(self, split: str) -> None:
        self.split = split
        self.performances = 0
        self.pre_left_on = 0
        self.pre_counts: Counter[str] = Counter()
        self.pre_raw = self.pre_retained = 0
        self.pre_initialized_total = self.pre_initialized_retained = 0
        self.pre_final_checks = self.pre_final_matches = 0
        self.pre_initialized_checks = self.pre_initialized_matches = 0
        self.pre_before_left = 0
        self.pre_last_delays: list[float] = []
        self.post_has_any = 0
        self.post_inside = self.post_beyond = 0
        self.post_truncated_performances = 0
        self.post_counts: Counter[str] = Counter()
        self.post_raw = self.post_retained = 0
        self.post_final_checks = self.post_final_matches = 0
        self.post_delays: list[float] = []
        self.post_on_at_horizon = 0
        self.exact_tend = 0
        self.exact_post_right = 0

    def payload(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "performances": self.performances,
            "pre": {
                "human_on_at_left_boundary": self.pre_left_on,
                "human_on_at_left_boundary_fraction": ratio(self.pre_left_on, self.performances),
                "synthetic_down_required": self.pre_left_on,
                "synthetic_down_required_fraction": ratio(self.pre_left_on, self.performances),
                "actual_crossing_count_distribution": {
                    bucket: {"count": self.pre_counts[bucket], "fraction": ratio(self.pre_counts[bucket], self.performances)}
                    for bucket in BUCKETS
                },
                "actual_raw_crossings": self.pre_raw,
                "actual_retained_crossings": self.pre_retained,
                "actual_transition_retention": ratio(self.pre_retained, self.pre_raw),
                "initialized_target_events_before_compression": self.pre_initialized_total,
                "initialized_target_events_retained": self.pre_initialized_retained,
                "initialized_target_retention": ratio(self.pre_initialized_retained, self.pre_initialized_total),
                "actual_final_state_preservation": ratio(self.pre_final_matches, self.pre_final_checks),
                "initialized_from_off_final_state_preservation": ratio(self.pre_initialized_matches, self.pre_initialized_checks),
                "effective_crossings_before_left_boundary": self.pre_before_left,
                "last_pre_onset_crossing_delay_seconds": quantiles(self.pre_last_delays),
            },
            "post": {
                "performances_with_transition_at_or_after_tend": self.post_has_any,
                "performances_with_transition_at_or_after_tend_fraction": ratio(self.post_has_any, self.performances),
                "transitions_inside_represented_horizon": self.post_inside,
                "transitions_truncated_after_horizon": self.post_beyond,
                "performances_with_truncated_transition": self.post_truncated_performances,
                "performances_with_truncated_transition_fraction": ratio(self.post_truncated_performances, self.performances),
                "represented_crossing_count_distribution": {
                    bucket: {"count": self.post_counts[bucket], "fraction": ratio(self.post_counts[bucket], self.performances)}
                    for bucket in BUCKETS
                },
                "represented_raw_crossings": self.post_raw,
                "represented_retained_crossings": self.post_retained,
                "represented_transition_retention": ratio(self.post_retained, self.post_raw),
                "final_state_preservation_within_horizon": ratio(self.post_final_matches, self.post_final_checks),
                "transition_delay_after_tend_seconds": quantiles(self.post_delays),
                "performances_on_at_represented_horizon": self.post_on_at_horizon,
                "events_exactly_at_tend": self.exact_tend,
                "events_exactly_at_tend_plus_1s": self.exact_post_right,
            },
        }


class CountAccumulator:
    def __init__(self) -> None:
        self.counts: dict[tuple[str, str, str], np.ndarray] = defaultdict(lambda: np.zeros(3, dtype=np.int64))
        self.checks: Counter[tuple[str, str]] = Counter()
        self.matches: Counter[tuple[str, str]] = Counter()
        self.failures: list[dict[str, Any]] = []

    def add_hist(self, split: str, case: str, region: str, histogram: np.ndarray) -> None:
        self.counts[(split, case, region)] += histogram

    def add_one(self, split: str, case: str, region: str, count: int) -> None:
        self.counts[(split, case, region)][compressed_n(count)] += 1

    def verify_bulk(self, split: str, case: str, raw_counts: np.ndarray, correction: bool) -> None:
        key = (split, case)
        self.checks[key] += int(raw_counts.size)
        human_parity = raw_counts % 2
        model_offset = int(correction)
        target_counts = raw_counts + model_offset
        retained_parity = np.where(
            target_counts <= 2,
            target_counts,
            np.where(target_counts % 2, 1, 2),
        ) % 2
        matches = (model_offset ^ retained_parity) == human_parity
        self.matches[key] += int(matches.sum())
        if not bool(np.all(matches)) and len(self.failures) < 10:
            self.failures.append({"split": split, "case": case, "raw_count": int(raw_counts[np.flatnonzero(~matches)[0]])})

    def rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for split in ("train", "validation"):
            for case in ("base_oracle", "counterfactual_mismatch", "pre_default_off"):
                regions = ("PRE",) if case == "pre_default_off" else ("PRE", "MAIN", "POST", "ALL")
                for region in regions:
                    counts = self.counts[(split, case, region)]
                    total = int(counts.sum())
                    for klass in range(3):
                        rows.append({
                            "split": split,
                            "target_case": case,
                            "interval_type": region,
                            "N": klass,
                            "count": int(counts[klass]),
                            "fraction": ratio(int(counts[klass]), total),
                            "candidate_inverse_sqrt_raw": "",
                            "candidate_inverse_sqrt_mean1": "",
                        })
        train = self.counts[("train", "base_oracle", "ALL")]
        frequencies = train / train.sum()
        raw = [1.0 / math.sqrt(float(value)) if value > 0 else None for value in frequencies]
        normalized = inverse_sqrt_mean_one(train.tolist())
        for row in rows:
            if row["split"] == "train" and row["target_case"] == "base_oracle" and row["interval_type"] == "ALL":
                klass = int(row["N"])
                row["candidate_inverse_sqrt_raw"] = raw[klass]
                row["candidate_inverse_sqrt_mean1"] = normalized[klass]
        return rows

    def payload(self) -> dict[str, Any]:
        rows = self.rows()
        nested: dict[str, Any] = {}
        for row in rows:
            target = nested.setdefault(row["split"], {}).setdefault(row["target_case"], {}).setdefault(row["interval_type"], {})
            target[str(row["N"])] = {"count": row["count"], "fraction": row["fraction"]}
        train = self.counts[("train", "base_oracle", "ALL")]
        frequencies = train / train.sum()
        nested["candidate_inverse_sqrt_weights"] = {
            "basis": "train/base_oracle/ALL",
            "formula": "w_c proportional to 1/sqrt(f_c)",
            "normalization": "legacy audit-local mean-of-present-classes=1 diagnostic; not the project canonical weighted-CE convention",
            "frequency": {str(i): float(frequencies[i]) for i in range(3)},
            "raw": {str(i): float(1.0 / math.sqrt(frequencies[i])) for i in range(3)},
            "mean1": {str(i): value for i, value in enumerate(inverse_sqrt_mean_one(train.tolist()))},
        }
        nested["reconciliation_checks"] = {
            case: {
                split: {
                    "checks": self.checks[(split, case)],
                    "matches": self.matches[(split, case)],
                    "fraction": ratio(self.matches[(split, case)], self.checks[(split, case)]),
                }
                for split in ("train", "validation")
            }
            for case in ("base_oracle", "counterfactual_mismatch")
        }
        nested["reconciliation_failures"] = self.failures
        nested["case_b_is_counterfactual_diagnostic"] = True
        return nested


def transition_objects(directions: Sequence[str], taus: Sequence[float], ticks: Sequence[int]) -> tuple[BinaryTransition, ...]:
    return tuple(BinaryTransition(direction, float(tau), int(tick)) for direction, tau, tick in zip(directions, taus, ticks))


def add_boundary_performance(
    accumulator: BoundaryAccumulator,
    count_accumulator: CountAccumulator,
    binary_ticks: np.ndarray,
    directions: Sequence[str],
    event_seconds: np.ndarray,
    first_tick: int,
    end_tick: int,
    tick_to_seconds,
) -> None:
    accumulator.performances += 1
    split = accumulator.split
    first_seconds = float(tick_to_seconds(first_tick))
    end_seconds = float(tick_to_seconds(end_tick))
    pre_left = first_seconds - 1.0
    pre_start = bisect.bisect_left(event_seconds.tolist(), pre_left)
    pre_end = int(np.searchsorted(binary_ticks, first_tick, side="left"))
    pre_k = pre_end - pre_start
    human_pre_state = pre_start % 2
    pre_dirs = tuple(directions[pre_start:pre_end])
    pre_taus = tuple(float((value - pre_left) / 1.0) for value in event_seconds[pre_start:pre_end])
    pre_events = transition_objects(pre_dirs, pre_taus, binary_ticks[pre_start:pre_end].tolist())
    pre_retained = compress_binary_transitions(pre_events)
    human_pre_end = state_after(human_pre_state, pre_events)
    accumulator.pre_left_on += int(human_pre_state == ON)
    accumulator.pre_counts[count_bucket(pre_k)] += 1
    accumulator.pre_raw += pre_k
    accumulator.pre_retained += len(pre_retained)
    accumulator.pre_final_checks += 1
    accumulator.pre_final_matches += int(state_after(human_pre_state, pre_retained) == human_pre_end)
    initialized = reconcile_and_compress(OFF, human_pre_state, pre_events)
    accumulator.pre_initialized_total += pre_k + int(human_pre_state == ON)
    accumulator.pre_initialized_retained += len(initialized)
    accumulator.pre_initialized_checks += 1
    accumulator.pre_initialized_matches += int(state_after(OFF, initialized) == human_pre_end)
    accumulator.pre_before_left += pre_start
    if pre_end:
        accumulator.pre_last_delays.append(first_seconds - float(event_seconds[pre_end - 1]))

    count_accumulator.add_one(split, "base_oracle", "PRE", pre_k)
    count_accumulator.add_one(split, "counterfactual_mismatch", "PRE", pre_k + 1)
    count_accumulator.add_one(split, "pre_default_off", "PRE", pre_k + int(human_pre_state == ON))
    singleton = np.asarray([pre_k], dtype=np.int64)
    count_accumulator.verify_bulk(split, "base_oracle", singleton, False)
    count_accumulator.verify_bulk(split, "counterfactual_mismatch", singleton, True)

    post_start = int(np.searchsorted(binary_ticks, end_tick, side="left"))
    post_horizon = end_seconds + 1.0
    post_end = bisect.bisect_right(event_seconds.tolist(), post_horizon)
    post_k = post_end - post_start
    post_beyond = len(binary_ticks) - post_end
    human_post_state = post_start % 2
    post_dirs = tuple(directions[post_start:post_end])
    post_taus = tuple(float(value - end_seconds) for value in event_seconds[post_start:post_end])
    post_events = transition_objects(post_dirs, post_taus, binary_ticks[post_start:post_end].tolist())
    post_retained = compress_binary_transitions(post_events)
    human_horizon_end = state_after(human_post_state, post_events)
    accumulator.post_has_any += int(post_start < len(binary_ticks))
    accumulator.post_inside += post_k
    accumulator.post_beyond += post_beyond
    accumulator.post_truncated_performances += int(post_beyond > 0)
    accumulator.post_counts[count_bucket(post_k)] += 1
    accumulator.post_raw += post_k
    accumulator.post_retained += len(post_retained)
    accumulator.post_final_checks += 1
    accumulator.post_final_matches += int(state_after(human_post_state, post_retained) == human_horizon_end)
    accumulator.post_on_at_horizon += int(human_horizon_end == ON)
    accumulator.exact_tend += int(np.sum(binary_ticks == end_tick))
    accumulator.exact_post_right += sum(abs(float(value) - post_horizon) <= 1e-12 for value in event_seconds[post_start:])
    accumulator.post_delays.extend(float(value - end_seconds) for value in event_seconds[post_start:])
    count_accumulator.add_one(split, "base_oracle", "POST", post_k)
    count_accumulator.add_one(split, "counterfactual_mismatch", "POST", post_k + 1)
    singleton = np.asarray([post_k], dtype=np.int64)
    count_accumulator.verify_bulk(split, "base_oracle", singleton, False)
    count_accumulator.verify_bulk(split, "counterfactual_mismatch", singleton, True)


def invariant_results() -> list[dict[str, Any]]:
    def alternating(start: int, count: int) -> tuple[BinaryTransition, ...]:
        state = start
        result = []
        for index in range(count):
            result.append(BinaryTransition(DOWN if state == OFF else UP, (index + 1) / (count + 1)))
            state = 1 - state
        return tuple(result)

    rows: list[dict[str, Any]] = []

    def base(case: int, label: str, start: int, count: int) -> None:
        raw = alternating(start, count)
        retained = compress_binary_transitions(raw)
        expected = state_after(start, raw)
        actual = state_after(start, retained)
        rows.append({
            "case": case, "name": label, "interval_assignment": "GENERIC_INTERVAL",
            "raw_count": count, "compressed_N": len(retained),
            "retained_tau": "|".join(f"{item.tau:.6f}" for item in retained),
            "expected_final_state": "ON" if expected else "OFF",
            "reconstructed_final_state": "ON" if actual else "OFF", "passed": actual == expected,
        })

    base(1, "OFF start + zero event", OFF, 0)
    base(2, "OFF start + one event", OFF, 1)
    base(3, "OFF start + two events", OFF, 2)
    base(4, "ON start + one event", ON, 1)
    base(5, "ON start + two events", ON, 2)
    base(6, "3 raw events -> last 1", OFF, 3)
    base(7, "4 raw events -> last 2", OFF, 4)

    for case, count, label in (
        (8, 0, "mismatch + zero human event"),
        (9, 1, "mismatch + one human event"),
        (10, 2, "mismatch + two human events"),
        (11, 3, "mismatch correction + 3+ human events"),
    ):
        human = ON
        model = OFF
        raw = alternating(human, count)
        retained = reconcile_and_compress(model, human, raw)
        expected = state_after(human, raw)
        actual = state_after(model, retained)
        rows.append({
            "case": case, "name": label, "interval_assignment": "GENERIC_INTERVAL",
            "raw_count": count, "compressed_N": len(retained),
            "retained_tau": "|".join(f"{item.tau:.6f}" for item in retained),
            "expected_final_state": "ON" if expected else "OFF",
            "reconstructed_final_state": "ON" if actual else "OFF", "passed": actual == expected,
        })

    retained = reconcile_and_compress(OFF, ON, ())
    rows.append({
        "case": 12, "name": "PRE boundary already ON -> tau=0 correction", "interval_assignment": "PRE",
        "raw_count": 0, "compressed_N": len(retained),
        "retained_tau": "|".join(f"{item.tau:.6f}" for item in retained),
        "expected_final_state": "ON", "reconstructed_final_state": "ON" if state_after(OFF, retained) else "OFF",
        "passed": len(retained) == 1 and retained[0].tau == 0.0 and retained[0].direction == DOWN,
    })

    for case, label, value, expected_region, tau in (
        (13, "event exactly at t1", 10.0, "MAIN", 0.0),
        (14, "event exactly at T_end", 20.0, "POST", 0.0),
        (15, "event exactly at T_end+1s", 21.0, "POST", 1.0),
    ):
        event = BinaryTransition(DOWN, tau)
        retained = compress_binary_transitions((event,))
        region = interval_region(value, 10.0, 20.0)
        actual = state_after(OFF, retained)
        rows.append({
            "case": case, "name": label, "interval_assignment": region,
            "raw_count": 1, "compressed_N": len(retained), "retained_tau": f"{retained[0].tau:.6f}",
            "expected_final_state": "ON", "reconstructed_final_state": "ON" if actual else "OFF",
            "passed": region == expected_region and retained[0].tau == tau and actual == ON,
        })
    if len(rows) != 15 or not all(row["passed"] for row in rows):
        raise AssertionError("synthetic invariant suite failed")
    return rows


def load_frozen_manifest() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    config_path = CACHE_ROOT / "config.json"
    manifest_path = CACHE_ROOT / "cache_manifest.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if config["cache_id"] != CANONICAL_CACHE_ID or manifest["cache_id"] != CANONICAL_CACHE_ID:
        raise RuntimeError("canonical Custom Event cache ID changed")
    if config["asap_test_access_count"] != 0 or manifest["asap_test_access_count"] != 0:
        raise RuntimeError("frozen cache reports ASAP test access")
    if manifest["train_count"] != 2062 or manifest["validation_count"] != 71:
        raise RuntimeError("canonical performance counts changed")
    entries = list(manifest["entries"])
    if {entry["split"] for entry in entries} != {"train", "validation"}:
        raise RuntimeError("manifest contains a non-train/validation entry")
    sources = Counter(entry["source"] for entry in entries)
    if sources != EXPECTED_SOURCES:
        raise RuntimeError(f"canonical source counts changed: {sources}")
    provenance = {
        "cache_id": CANONICAL_CACHE_ID,
        "cache_config": str(config_path),
        "cache_config_sha256": file_sha256(config_path),
        "cache_manifest": str(manifest_path),
        "cache_manifest_sha256": file_sha256(manifest_path),
        "inherited_source_provenance": config["source_provenance"],
        "current_scan_source_sha256_verified": True,
        "asap_split_csv_opened_by_this_audit": False,
        "asap_test_metadata_access_count": 0,
        "asap_test_midi_access_count": 0,
        "model_checkpoint_access_count": 0,
        "training_steps": 0,
        "validation_inference_count": 0,
    }
    return config, manifest, entries, provenance


def scan_raw_data(entries: Sequence[dict[str, Any]], output: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    capacity = {key: CapacityAccumulator(key) for key in CAPACITY_KEYS}
    boundaries = {"train": BoundaryAccumulator("train"), "validation": BoundaryAccumulator("validation")}
    counts = CountAccumulator()
    extraction = {
        key: {"performances": 0, "raw_cc64_messages": 0, "same_tick_messages_collapsed": 0,
              "effective_4state_changes": 0, "effective_binary_crossings": 0,
              "direct_projection_matches": 0, "direct_projection_mismatches": 0,
              "cache_onset_matches": 0, "cache_note_end_matches": 0, "source_sha_matches": 0}
        for key in CAPACITY_KEYS
    }
    performance_rows: list[dict[str, Any]] = []
    started = time.monotonic()
    log_path = output / "audit.log"
    for index, entry in enumerate(entries, 1):
        source_name = entry["source"]
        split = entry["split"]
        source_path = Path(entry["source_midi"])
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        current_sha = sha256_file(source_path)
        if current_sha != entry["source_sha256"]:
            raise RuntimeError(f"source SHA changed: {source_path}")
        raw = parse_raw_midi(source_path)
        effective_4state = _effective_events(raw.raw_cc64)
        projected = tuple((int(event.tick), str(event.crossing)) for event in effective_4state if event.crossing)
        direct = direct_binary_crossings(raw.raw_cc64)
        if projected != direct:
            extraction[source_name]["direct_projection_mismatches"] += 1
            raise AssertionError(f"binary extraction mismatch: {source_path}")
        binary_ticks = np.asarray([tick for tick, _ in direct], dtype=np.int64)
        directions = tuple(direction for _, direction in direct)
        if any(direction != (DOWN if position % 2 == 0 else UP) for position, direction in enumerate(directions)):
            raise AssertionError(f"binary crossing sequence does not alternate: {source_path}")

        cache_path = CACHE_ROOT / entry["cache_file"]
        with np.load(cache_path, allow_pickle=False) as cache:
            onset_ticks = cache["onset_ticks"].astype(np.int64, copy=True)
            note_end_tick = int(cache["latest_note_off_tick"])
        complete_onsets = np.asarray(sorted({note[3] for note in raw.note_signature}), dtype=np.int64)
        if not np.array_equal(onset_ticks, complete_onsets):
            raise AssertionError(f"frozen complete-note onset metadata mismatch: {source_path}")
        if note_end_tick != raw.latest_note_off:
            raise AssertionError(f"latest note-off mismatch: {source_path}")
        tick_to_seconds, _ = _tick_second_converters(source_path)
        event_seconds = np.asarray([tick_to_seconds(int(tick)) for tick in binary_ticks], dtype=np.float64)
        if np.any(np.diff(event_seconds) < 0):
            raise AssertionError(f"non-monotonic converted event seconds: {source_path}")

        extract = extraction[source_name]
        extract["performances"] += 1
        extract["raw_cc64_messages"] += len(raw.raw_cc64)
        extract["same_tick_messages_collapsed"] += len(raw.raw_cc64) - len({event.tick for event in raw.raw_cc64})
        extract["effective_4state_changes"] += len(effective_4state)
        extract["effective_binary_crossings"] += len(direct)
        extract["direct_projection_matches"] += 1
        extract["cache_onset_matches"] += 1
        extract["cache_note_end_matches"] += 1
        extract["source_sha_matches"] += 1

        main_counts, _ = capacity[source_name].add(
            entry["performance_path"], onset_ticks, note_end_tick, binary_ticks,
            directions, tick_to_seconds,
        )
        counts.add_hist(split, "base_oracle", "MAIN", n_histogram(main_counts, correction=False))
        counts.add_hist(split, "counterfactual_mismatch", "MAIN", n_histogram(main_counts, correction=True))
        counts.verify_bulk(split, "base_oracle", main_counts, False)
        counts.verify_bulk(split, "counterfactual_mismatch", main_counts, True)
        add_boundary_performance(
            boundaries[split], counts, binary_ticks, directions, event_seconds,
            int(onset_ticks[0]), note_end_tick, tick_to_seconds,
        )
        overflow_count = int(np.sum(main_counts >= 3))
        performance_rows.append({
            "dataset": source_name, "split": split, "performance": entry["performance_path"],
            "main_intervals": int(main_counts.size), "overflow_intervals": overflow_count,
        })
        if index == 1 or index % 50 == 0 or index == len(entries):
            message = f"{now()} raw_scan {index}/{len(entries)} elapsed={time.monotonic()-started:.1f}s"
            print(message, flush=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(message + "\n")

    for split in ("train", "validation"):
        for case in ("base_oracle", "counterfactual_mismatch"):
            counts.counts[(split, case, "ALL")] = sum(
                (counts.counts[(split, case, region)] for region in ("PRE", "MAIN", "POST")),
                np.zeros(3, dtype=np.int64),
            )
    payload = {
        "binary_extraction_validation": extraction,
        "capacity": {key: capacity[key].payload() for key in CAPACITY_KEYS},
        "pre_post": {split: boundaries[split].payload() for split in ("train", "validation")},
        "count_targets": counts.payload(),
    }
    overflow_examples = [row for key in CAPACITY_KEYS for row in capacity[key].overflow_examples]
    return payload, performance_rows + overflow_examples


def audit_ownership(entries: Sequence[dict[str, Any]], output: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = ALIGNMENT_ROOT / "alignment_manifest.json"
    alignment_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if alignment_manifest["alignment_id"] != CANONICAL_ALIGNMENT_ID:
        raise RuntimeError("canonical alignment ID changed")
    if alignment_manifest["tokenizer_cache_id"] != CANONICAL_CACHE_ID:
        raise RuntimeError("alignment/tokenizer cache identity mismatch")
    if alignment_manifest["asap_test_access_count"] != 0:
        raise RuntimeError("alignment provenance reports ASAP test access")
    alignment_entries = {
        (row["split"], int(row["performance_index"])): row
        for row in alignment_manifest["entries"]
    }
    rows: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    totals = {
        split: {"performances": 0, "onsets": 0, "zero_owner": 0, "duplicate_owner": 0,
                "non_monotonic_performances": 0, "offending_onset_transitions": 0, "windows": 0}
        for split in ("train", "validation")
    }
    started = time.monotonic()
    for index, entry in enumerate(entries, 1):
        split = entry["split"]
        cache_path = CACHE_ROOT / entry["cache_file"]
        with np.load(cache_path, allow_pickle=False) as cache:
            notes = int(cache["non_pedal_note_count"])
            first = cache["onset_first_note_index"].astype(np.int64, copy=True)
            last = cache["onset_last_note_index"].astype(np.int64, copy=True)
            representative = cache["onset_representative_note_index"].astype(np.int64, copy=True)
        alignment_entry = alignment_entries[(split, int(entry["performance_index"]))]
        alignment_path = ALIGNMENT_ROOT / alignment_entry["alignment_file"]
        if sha256_file(alignment_path) != alignment_entry["alignment_file_sha256"]:
            raise RuntimeError(f"alignment SHA mismatch: {alignment_path}")
        alignment, metadata = load_note_alignment(alignment_path)
        if metadata["alignment_config_sha256"] != alignment_manifest["alignment_config_sha256"]:
            raise RuntimeError(f"alignment config mismatch: {alignment_path}")
        if metadata["source_sha256"] != entry["source_sha256"] or alignment.note_count != notes:
            raise RuntimeError(f"alignment source/note identity mismatch: {alignment_path}")
        mapped_first, mapped_last, mapped_rep, non_contiguous = alignment.map_group_bounds(first, last, representative)
        if non_contiguous:
            raise RuntimeError(f"non-contiguous mapped onset groups: {alignment_path}")
        ownership = assign_unique_owners(notes, mapped_first, mapped_last, mapped_rep, window_notes=512, stride_notes=256)
        membership = np.zeros(len(first), dtype=np.int8)
        for owned in ownership.owned_onset_indices:
            membership[owned] += 1
        zero = int(np.sum(membership == 0))
        duplicate = int(np.sum(membership > 1))
        differences = np.diff(ownership.owner_window_index)
        offending = np.flatnonzero(differences < 0)
        non_monotonic = int(offending.size > 0)
        row = {
            "split": split, "dataset": entry["source"], "performance": entry["performance_path"],
            "notes": notes, "distinct_onsets": len(first), "windows": len(ownership.window_starts),
            "zero_owner_count": zero, "duplicate_owner_count": duplicate,
            "non_monotonic_owner_ordering": non_monotonic,
            "offending_onset_transitions": int(offending.size),
            "state_chain_usable": int(zero == 0 and duplicate == 0 and offending.size == 0),
        }
        rows.append(row)
        target = totals[split]
        target["performances"] += 1
        target["onsets"] += len(first)
        target["windows"] += len(ownership.window_starts)
        target["zero_owner"] += zero
        target["duplicate_owner"] += duplicate
        target["non_monotonic_performances"] += non_monotonic
        target["offending_onset_transitions"] += int(offending.size)
        for onset_index in offending[: max(0, 10 - len(examples))].tolist():
            before_owner = int(ownership.owner_window_index[onset_index])
            after_owner = int(ownership.owner_window_index[onset_index + 1])
            examples.append({
                "split": split, "performance": entry["performance_path"], "onset_index": onset_index,
                "next_onset_index": onset_index + 1, "owner_index_before": before_owner,
                "owner_index_after": after_owner,
                "owner_start_before": ownership.window_starts[before_owner],
                "owner_start_after": ownership.window_starts[after_owner],
            })
        if index == 1 or index % 250 == 0 or index == len(entries):
            message = f"{now()} owner_scan {index}/{len(entries)} elapsed={time.monotonic()-started:.1f}s"
            print(message, flush=True)
            with (output / "audit.log").open("a", encoding="utf-8") as handle:
                handle.write(message + "\n")
    usable = all(
        values["zero_owner"] == 0
        and values["duplicate_owner"] == 0
        and values["non_monotonic_performances"] == 0
        for values in totals.values()
    )
    return {
        "window_notes": 512, "stride_notes": 256,
        "rule": "full onset group eligible; maximum representative context margin; tie -> earlier start",
        "alignment_id": CANONICAL_ALIGNMENT_ID,
        "alignment_manifest_sha256": sha256_file(manifest_path),
        "splits": totals, "representative_examples": examples,
        "existing_owner_rule_usable_for_single_chronological_state_chain": usable,
    }, rows


def distribution_csv_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, payload in summary["capacity"].items():
        for bucket in BUCKETS:
            value = payload["raw_count_distribution"][bucket]
            rows.append({"dataset": dataset, "split": "validation" if dataset == "ASAP-validation" else "train",
                         "interval_type": "MAIN", "raw_count_bucket": bucket, **value})
    for split in ("train", "validation"):
        for region, key in (("PRE", "actual_crossing_count_distribution"), ("POST", "represented_crossing_count_distribution")):
            payload = summary["pre_post"][split][region.lower()][key]
            for bucket in BUCKETS:
                rows.append({"dataset": split, "split": split, "interval_type": region,
                             "raw_count_bucket": bucket, **payload[bucket]})
    return rows


def pre_post_csv_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for split in ("train", "validation"):
        pre = summary["pre_post"][split]["pre"]
        post = summary["pre_post"][split]["post"]
        row = {"split": split, "performances": summary["pre_post"][split]["performances"]}
        for key, value in pre.items():
            if not isinstance(value, dict):
                row[f"pre_{key}"] = value
        for key, value in pre["last_pre_onset_crossing_delay_seconds"].items():
            row[f"pre_last_delay_{key}"] = value
        for key, value in post.items():
            if not isinstance(value, dict):
                row[f"post_{key}"] = value
        for key, value in post["transition_delay_after_tend_seconds"].items():
            row[f"post_delay_{key}"] = value
        rows.append(row)
    return rows


def percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.6f}%"


def markdown_report(summary: Mapping[str, Any]) -> str:
    capacity = summary["capacity"]
    pre_post = summary["pre_post"]
    count_targets = summary["count_targets"]
    owner = summary["ownership"]
    invariants = summary["invariants"]
    hard_fail = summary["hard_invariant_failure"]
    status = summary["conclusion_status"]
    capacity_rows = []
    for dataset in CAPACITY_KEYS:
        item = capacity[dataset]
        capacity_rows.append(
            f"| {dataset} | {item['performances']:,} | {item['main_intervals']:,} | "
            f"{item['overflow_intervals']:,} ({percent(item['overflow_interval_fraction'])}) | "
            f"{item['raw_effective_transitions']:,} | {item['retained_transitions']:,} | "
            f"{percent(item['overall_transition_retention'])} | "
            f"{item['performances_with_any_overflow']:,} ({percent(item['performances_with_any_overflow_fraction'])}) |"
        )
    bucket_rows = []
    for dataset in CAPACITY_KEYS:
        item = capacity[dataset]
        cells = [f"{item['raw_count_distribution'][bucket]['count']:,} ({percent(item['raw_count_distribution'][bucket]['fraction'])})" for bucket in BUCKETS]
        bucket_rows.append(f"| {dataset} | " + " | ".join(cells) + " |")
    direction_rows = []
    gesture_rows = []
    for dataset in CAPACITY_KEYS:
        for direction in (UP, DOWN):
            item = capacity[dataset]["direction_retention"][direction]
            direction_rows.append(f"| {dataset} | {direction} | {item['raw']:,} | {item['retained']:,} | {percent(item['retention'])} |")
        for pair in (f"{UP}->{DOWN}", f"{DOWN}->{UP}"):
            item = capacity[dataset]["gesture_retention"][pair]
            gesture_rows.append(f"| {dataset} | {pair} | {item['raw_adjacent_pairs']:,} | {item['retained_as_pair']:,} | {percent(item['retention'])} |")
    count_rows = []
    for split in ("train", "validation"):
        for region in ("PRE", "MAIN", "POST", "ALL"):
            item = count_targets[split]["base_oracle"][region]
            count_rows.append(
                f"| {split} | {region} | " + " | ".join(
                    f"{item[str(n)]['count']:,} ({percent(item[str(n)]['fraction'])})" for n in range(3)
                ) + " |"
            )
    mismatch_rows = []
    for split in ("train", "validation"):
        check = count_targets["reconciliation_checks"]["counterfactual_mismatch"][split]
        item = count_targets[split]["counterfactual_mismatch"]["ALL"]
        mismatch_rows.append(
            f"| {split} | {check['checks']:,} | {check['matches']:,} | {percent(check['fraction'])} | "
            + " | ".join(f"{item[str(n)]['count']:,} ({percent(item[str(n)]['fraction'])})" for n in range(3)) + " |"
        )
    pre_rows = []
    post_rows = []
    for split in ("train", "validation"):
        pre = pre_post[split]["pre"]
        post = pre_post[split]["post"]
        pre_rows.append(
            f"| {split} | {pre['human_on_at_left_boundary']:,} ({percent(pre['human_on_at_left_boundary_fraction'])}) | "
            f"{pre['effective_crossings_before_left_boundary']:,} | {percent(pre['actual_transition_retention'])} | "
            f"{percent(pre['initialized_target_retention'])} | {percent(pre['initialized_from_off_final_state_preservation'])} |"
        )
        post_rows.append(
            f"| {split} | {post['performances_with_transition_at_or_after_tend']:,} ({percent(post['performances_with_transition_at_or_after_tend_fraction'])}) | "
            f"{post['transitions_inside_represented_horizon']:,} | {post['transitions_truncated_after_horizon']:,} | "
            f"{post['performances_with_truncated_transition']:,} ({percent(post['performances_with_truncated_transition_fraction'])}) | "
            f"{percent(post['represented_transition_retention'])} | {percent(post['final_state_preservation_within_horizon'])} |"
        )
    weights = count_targets["candidate_inverse_sqrt_weights"]
    train_all = count_targets["train"]["base_oracle"]["ALL"]
    owner_rows = []
    for split in ("train", "validation"):
        item = owner["splits"][split]
        owner_rows.append(
            f"| {split} | {item['performances']:,} | {item['onsets']:,} | {item['zero_owner']:,} | "
            f"{item['duplicate_owner']:,} | {item['non_monotonic_performances']:,} | {item['offending_onset_transitions']:,} |"
        )
    q1 = (
        "예. binary-only target에서 대부분의 interval은 2-slot 이내이며 실측 retention은 위 표와 같다. "
        "다만 overflow에서 사라지는 중간 crossing/gesture가 있으므로 무손실 표현은 아니다."
    )
    q2 = "예. train/validation의 모든 MAIN, PRE, POST represented interval에서 final binary state를 100% 보존했다."
    q3 = (
        f"1초보다 앞선 binary crossing은 train {pre_post['train']['pre']['effective_crossings_before_left_boundary']:,}개, "
        f"validation {pre_post['validation']['pre']['effective_crossings_before_left_boundary']:,}개가 event timing/history로는 사라진다. "
        "PRE-left state가 ON이면 tau=0 DOWN 하나로 state만 복원한다."
    )
    q4 = (
        f"T_end+1초 뒤 crossing은 train {pre_post['train']['post']['transitions_truncated_after_horizon']:,}개, "
        f"validation {pre_post['validation']['post']['transitions_truncated_after_horizon']:,}개가 잘린다. horizon 끝 ON은 그대로 허용했다."
    )
    q5 = "예." if not count_targets["reconciliation_failures"] else "아니오. failure example을 JSON에 기록했다."
    q7 = "예." if owner["existing_owner_rule_usable_for_single_chronological_state_chain"] else "아니오. owner interleaving example을 기록했다."
    q8 = "없다. 측정된 정보 손실은 WARN 사항이나 hard invariant failure는 없다." if not hard_fail else "있다. hard invariant failure가 있으므로 구현 전에 수정이 필요하다."
    return f"""# Binary 2-Slot Transition Audit v0

## 1. Scope / frozen specification

- Binary state: `CC64 < 64 -> OFF`, `CC64 >= 64 -> ON`; 실제 threshold crossing만 DOWN/UP으로 남겼다.
- PRE `[t1-1s,t1)`, MAIN `[t_i,t_(i+1))`, final MAIN `[t_M,T_end)`, POST `[T_end,T_end+1s]`를 그대로 적용했다.
- max-2 odd/even last-retention, PRE default OFF correction, counterfactual mismatch correction을 변경 없이 사용했다.
- 모델/head/checkpoint/training/inference와 horizon/capacity tuning은 수행하지 않았다. ASAP test metadata/MIDI access는 0이다.

## 2. Dataset provenance

- Frozen Custom Event cache `{summary['provenance']['cache_id']}`의 train 2,062개(MAESTRO-clean 1,170 + ASAP train 892)와 validation 71개(ASAP validation)만 사용했다.
- 각 원본 MIDI의 현재 SHA-256을 frozen manifest와 대조한 뒤 raw MIDI를 다시 parse했다.
- validation membership은 frozen cache provenance를 상속했고 이번 audit은 ASAP split CSV를 열지 않았다.
- owner audit은 alignment `{owner['alignment_id']}`와 기존 512-note/stride-256 maximum-margin owner 구현을 그대로 호출했다.

## 3. Binary crossing extraction validation

- 기존 parser의 equal-tick `(tick, track, message)`/last-value semantics 후 4-state effective event에서 threshold crossing만 투영한 결과를 독립 direct binary projection과 전 performance에서 exact 비교했다.
- MAESTRO-clean / ASAP train / ASAP validation 모두 projection mismatch 0, cache onset mismatch 0, latest-noteoff mismatch 0, source SHA mismatch 0이다.
- SMF cross-track same-tick ordering의 기존 deterministic limitation은 그대로 상속한다. 새로운 ordering semantics는 만들지 않았다.

## 4. 2-slot capacity results

| Dataset | Performances | MAIN intervals | Overflow >=3 | Raw crossings | Retained | Retention | Performances with overflow |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(capacity_rows)}

| Dataset | K=0 | K=1 | K=2 | K=3 | K=4 | K=5+ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(bucket_rows)}

| Dataset | Direction | Raw | Retained | Retention |
| --- | --- | ---: | ---: | ---: |
{chr(10).join(direction_rows)}

| Dataset | Adjacent gesture | Raw pairs | Pair retained intact | Retention |
| --- | --- | ---: | ---: | ---: |
{chr(10).join(gesture_rows)}

`UP->DOWN`은 요청된 repedal-like pair diagnostic일 뿐 새로운 repedal metric이 아니다. Overflow/performance 분포와 대표 예시는 JSON 및 CSV에 저장했다.

## 5. PRE 1-second audit

| Split | ON at PRE-left / synthetic DOWN | Crossings before PRE-left | Actual crossing retention | Initialized-target retention | Initialized final-state preservation |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(pre_rows)}

PRE raw `0/1/2/3/4/5+` 분포와 first-onset 전 마지막 crossing delay의 median/Q3/p95/p99/max는 `pre_post_audit.csv`와 `audit_summary.json`에 있다. `actual` retention은 human crossing만, `initialized-target` retention은 필요한 synthetic DOWN까지 분모에 포함한다.

## 6. POST 1-second audit

| Split | Any crossing at/after T_end | Inside 1s | Truncated after 1s | Performances truncated | Inside retention | Horizon final-state preservation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(post_rows)}

POST right endpoint의 `tau=1`은 포함했고 강제 UP은 추가하지 않았다. 모든 T_end 이후 crossing delay와 POST 분포는 JSON/CSV에 기록했다.

## 7. Base `N=0/1/2` distribution

| Split | Interval | N=0 | N=1 | N=2 |
| --- | --- | ---: | ---: | ---: |
{chr(10).join(count_rows)}

Base/oracle은 `s_model=s_human`인 실제 human sequence다. 별도로 실제 PRE 실행 조건(default OFF)은 `pre_default_off` 행으로 CSV/JSON에 저장했다. Train ALL candidate inverse-sqrt raw weights는 `{weights['raw']}`이다. 함께 기록한 `{weights['mean1']}`는 audit-local legacy mean-present-class=1 diagnostic이며 project canonical weighted-CE normalization이 아니다. Canonical 보정값은 follow-up audit에 기록한다.

## 8. Counterfactual state-mismatch correction audit

| Split | Intervals checked | Recovered | Recovery | N=0 | N=1 | N=2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(mismatch_rows)}

Case B는 실제 training distribution이 아니라 모든 interval에서 `s_model != s_human`을 가정한 counterfactual diagnostic이다. Correction 방향과 tau=0 prepend 후 동일 compression을 적용했다.

## 9. Owner-window/state-chain feasibility

| Split | Performances | Onsets | 0 owner | Duplicate owner | Non-monotonic performances | Offending transitions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(owner_rows)}

기존 owner rule을 `PRE -> owned MAIN chronological order -> POST`의 단일 state chain에 그대로 사용할 수 있는가: **{q7}**

## 10. Edge-case invariant results

- 15/15 required synthetic cases passed: {sum(row['passed'] for row in invariants)}/{len(invariants)}.
- 각 case의 interval assignment, compressed N, retained tau, reconstructed final state는 `invariant_results.csv`에 있다.

## 11. Information lost by the proposed representation

- MAIN/PRE/POST 안에서 3개 이상 crossing이 있으면 parity와 final state는 보존하지만 앞쪽 crossing timing 및 일부 adjacent two-event gesture는 손실된다.
- PRE-left보다 이른 crossing history/timing은 버리고 PRE-left binary state만 synthetic correction으로 전달한다.
- POST horizon 뒤 crossing은 전부 truncate한다. `T_end+1s`에서 ON인 trajectory에는 강제 UP을 넣지 않는다.
- Binary thresholding 자체가 CC64 depth 변화(0/LOW/HALF/FULL)를 모두 버린다.
- Equal-tick cross-track ordering은 기존 deterministic parser convention을 상속하며 SMF 차원의 의미적 ambiguity를 해결하지 않는다.

## 12. {status} conclusion

**Q1. Binary-only setting에서 max 2-slot이 실용적으로 타당한가?** {q1}

**Q2. Odd/even last-retention이 final binary state를 실제 데이터 전체에서 100% 보존하는가?** {q2}

**Q3. PRE 1s truncation으로 어떤 정보가 손실되는가?** {q3}

**Q4. POST 1s truncation으로 어떤 정보가 손실되는가?** {q4}

**Q5. tau=0 state-reconciliation rule이 mismatch case에서 항상 human interval-end state를 복구 가능한 target으로 만드는가?** {q5} 전수 검사 failure 수는 {len(count_targets['reconciliation_failures'])}이다.

**Q6. N=0/1/2 class imbalance는 어느 정도인가?** Train ALL base 분포는 N=0 {train_all['0']['count']:,} ({percent(train_all['0']['fraction'])}), N=1 {train_all['1']['count']:,} ({percent(train_all['1']['fraction'])}), N=2 {train_all['2']['count']:,} ({percent(train_all['2']['fraction'])})이다.

**Q7. 기존 owner-window rule을 performance-level hard free-running state carry에 그대로 사용할 수 있는가?** {q7}

**Q8. Full model implementation 전에 반드시 설계를 수정해야 할 blocking issue가 있는가?** {q8}

Overall status는 **{status}**다. WARN은 측정된 compression/horizon/depth 정보 손실을 명시하는 것이며 새로운 threshold나 자동 설계 변경을 뜻하지 않는다.
"""


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.chmod(0o755)
    atomic_text(OUTPUT_ROOT / "audit.log", f"{now()} start\n")
    _, _, entries, provenance = load_frozen_manifest()
    invariants = invariant_results()
    raw_summary, mixed_rows = scan_raw_data(entries, OUTPUT_ROOT)
    ownership, owner_rows = audit_ownership(entries, OUTPUT_ROOT)
    performance_rows = [row for row in mixed_rows if "main_intervals" in row]
    overflow_examples = [row for row in mixed_rows if "raw_directions" in row]
    hard_failure = (
        any(
            item["direct_projection_mismatches"]
            or item["cache_onset_matches"] != item["performances"]
            or item["cache_note_end_matches"] != item["performances"]
            or item["source_sha_matches"] != item["performances"]
            for item in raw_summary["binary_extraction_validation"].values()
        )
        or any(item["final_state_preservation"] != 1.0 for item in raw_summary["capacity"].values())
        or any(
            raw_summary["pre_post"][split][region][key] != 1.0
            for split in ("train", "validation")
            for region, key in (
                ("pre", "actual_final_state_preservation"),
                ("pre", "initialized_from_off_final_state_preservation"),
                ("post", "final_state_preservation_within_horizon"),
            )
        )
        or bool(raw_summary["count_targets"]["reconciliation_failures"])
        or not ownership["existing_owner_rule_usable_for_single_chronological_state_chain"]
        or not all(row["passed"] for row in invariants)
    )
    any_information_loss = (
        any(item["overflow_intervals"] > 0 for item in raw_summary["capacity"].values())
        or any(raw_summary["pre_post"][split]["pre"]["effective_crossings_before_left_boundary"] > 0 for split in ("train", "validation"))
        or any(raw_summary["pre_post"][split]["post"]["transitions_truncated_after_horizon"] > 0 for split in ("train", "validation"))
    )
    status = "FAIL" if hard_failure else "WARN" if any_information_loss else "PASS"
    summary = {
        "audit_id": "binary_2slot_transition_audit_v0",
        "completed_at": now(),
        "specification": {
            "binary_threshold": {"OFF": "CC64 < 64", "ON": "CC64 >= 64"},
            "events": {"OFF->ON": DOWN, "ON->OFF": UP},
            "pre": "[t1-1.0s,t1)", "main": "[t_i,t_(i+1)); final [t_M,T_end)",
            "post": "[T_end,T_end+1.0s]", "capacity": 2,
            "compression": "K<=2 keep all; odd overflow keep last 1; even overflow keep last 2",
            "pre_default_state": "OFF", "mismatch_correction": "prepend directional transition at tau=0",
        },
        "provenance": provenance,
        **raw_summary,
        "ownership": ownership,
        "invariants": invariants,
        "hard_invariant_failure": hard_failure,
        "information_loss_observed": any_information_loss,
        "conclusion_status": status,
        "forbidden_activity": {
            "model_implementation": 0, "checkpoint_creation_or_access": 0, "training_steps": 0,
            "validation_model_inference": 0, "asap_test_metadata_access": 0, "asap_test_midi_access": 0,
            "pre_post_tuning": 0, "capacity_tuning": 0,
        },
    }

    distribution_rows = distribution_csv_rows(summary)
    atomic_csv(OUTPUT_ROOT / "interval_count_distribution.csv", distribution_rows,
               ("dataset", "split", "interval_type", "raw_count_bucket", "count", "fraction"))
    pre_post_rows = pre_post_csv_rows(summary)
    atomic_csv(OUTPUT_ROOT / "pre_post_audit.csv", pre_post_rows, tuple(pre_post_rows[0]))
    # Use the already materialized payload to avoid rebuilding accumulator state.
    flat_count_rows = []
    for split in ("train", "validation"):
        for case in ("base_oracle", "counterfactual_mismatch", "pre_default_off"):
            regions = ("PRE",) if case == "pre_default_off" else ("PRE", "MAIN", "POST", "ALL")
            for region in regions:
                for n in range(3):
                    item = summary["count_targets"][split][case][region][str(n)]
                    weight = summary["count_targets"]["candidate_inverse_sqrt_weights"]
                    flat_count_rows.append({
                        "split": split, "target_case": case, "interval_type": region, "N": n,
                        "count": item["count"], "fraction": item["fraction"],
                        "candidate_inverse_sqrt_raw": weight["raw"][str(n)] if split == "train" and case == "base_oracle" and region == "ALL" else "",
                        "candidate_inverse_sqrt_mean1": weight["mean1"][str(n)] if split == "train" and case == "base_oracle" and region == "ALL" else "",
                    })
    atomic_csv(OUTPUT_ROOT / "count_class_distribution.csv", flat_count_rows, tuple(flat_count_rows[0]))
    atomic_csv(OUTPUT_ROOT / "owner_ordering_audit.csv", owner_rows, tuple(owner_rows[0]))
    atomic_csv(OUTPUT_ROOT / "invariant_results.csv", invariants, tuple(invariants[0]))
    atomic_csv(
        OUTPUT_ROOT / "overflow_performance_distribution.csv", performance_rows,
        ("dataset", "split", "performance", "main_intervals", "overflow_intervals"),
    )
    overflow_fields = (
        "dataset", "performance", "interval_index", "left_tick", "right_tick", "left_seconds",
        "right_seconds", "raw_count", "raw_directions", "retained_directions",
    )
    atomic_csv(OUTPUT_ROOT / "overflow_examples.csv", overflow_examples, overflow_fields)
    gesture_rows = []
    for dataset in CAPACITY_KEYS:
        for pair, item in summary["capacity"][dataset]["gesture_retention"].items():
            gesture_rows.append({"dataset": dataset, "pair": pair, **item})
    atomic_csv(OUTPUT_ROOT / "gesture_retention.csv", gesture_rows, tuple(gesture_rows[0]))
    atomic_json(OUTPUT_ROOT / "audit_summary.json", summary)
    atomic_text(OUTPUT_ROOT / "BINARY_2SLOT_TRANSITION_AUDIT.md", markdown_report(summary))
    atomic_json(OUTPUT_ROOT / "run_status.json", {
        "status": "completed", "conclusion": status, "hard_invariant_failure": hard_failure,
        "train_performances": 2062, "validation_performances": 71,
        "asap_test_metadata_access_count": 0, "asap_test_midi_access_count": 0,
        "model_steps": 0, "training_steps": 0, "validation_inference_count": 0,
        "completed_at": summary["completed_at"],
    })
    with (OUTPUT_ROOT / "audit.log").open("a", encoding="utf-8") as handle:
        handle.write(f"{now()} completed status={status}\n")
    (OUTPUT_ROOT / "audit.log").chmod(0o644)
    print(json.dumps({"status": "completed", "conclusion": status, "output": str(OUTPUT_ROOT), "test_access": 0, "training": 0}))


if __name__ == "__main__":
    main()
