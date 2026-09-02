#!/usr/bin/env python3
"""Final no-model follow-up audit for the frozen Binary 2-Slot formulation."""

from __future__ import annotations

import ast
import bisect
import csv
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_binary_2slot_transition_audit_v0 import (
    BUCKETS,
    CACHE_ROOT,
    atomic_csv,
    atomic_json,
    atomic_text,
    direct_binary_crossings,
    file_sha256,
    load_frozen_manifest,
    now,
    percent,
    quantiles,
)
from src.stage2_event_tokenizer.binary_2slot import (
    OFF,
    ON,
    BinaryTransition,
    canonical_inverse_sqrt_weights,
    correction_transition,
    reconcile_and_compress,
    state_after,
)
from src.stage2_event_tokenizer.tokenizer import parse_raw_midi
from src.stage2_event_tokenizer.tokenizer_v1 import _tick_second_converters

PRIOR_ROOT = ROOT / "analysis/binary_2slot_transition_audit_v0"
OUTPUT_ROOT = ROOT / "analysis/binary_2slot_transition_followup_v0"
TEST_FILE = ROOT / "tests/test_binary_2slot_transition_audit_v0.py"
REGIONS = ("PRE", "MAIN", "POST", "ALL")
POLICY_CURRENT = "current_final_state_priority"
POLICY_RESERVED = "correction_reserved_parity_valid"


def count_bucket(value: int) -> str:
    return str(value) if value <= 4 else "5+"


def ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def distribution(values: Sequence[float]) -> dict[str, Any]:
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


def replay_trajectory(
    human_start: int,
    model_start: int,
    human_events: Sequence[BinaryTransition],
    target_events: Sequence[BinaryTransition],
) -> dict[str, Any]:
    """Replay both piecewise-constant trajectories over normalized [0,1]."""

    human_by_tau: dict[float, list[BinaryTransition]] = defaultdict(list)
    target_by_tau: dict[float, list[BinaryTransition]] = defaultdict(list)
    for event in human_events:
        human_by_tau[float(event.tau)].append(event)
    for event in target_events:
        target_by_tau[float(event.tau)].append(event)
    times = sorted(set(human_by_tau) | set(target_by_tau))
    human_state = int(human_start)
    target_state = int(model_start)
    cursor = 0.0
    mismatch_duration = 0.0
    first_recovery: float | None = None
    for tau in times:
        if not -1e-12 <= tau <= 1.0 + 1e-12:
            raise ValueError(f"trajectory tau outside [0,1]: {tau}")
        tau = min(1.0, max(0.0, tau))
        if human_state != target_state:
            mismatch_duration += tau - cursor
        if human_by_tau.get(tau):
            human_state = state_after(human_state, human_by_tau[tau])
        if target_by_tau.get(tau):
            target_state = state_after(target_state, target_by_tau[tau])
        if first_recovery is None and human_state == target_state:
            first_recovery = tau
        cursor = tau
    if human_state != target_state:
        mismatch_duration += 1.0 - cursor
    if first_recovery is None and human_state == target_state:
        first_recovery = cursor
    return {
        "first_recovery_tau": first_recovery,
        "mismatch_fraction": float(mismatch_duration),
        "human_end_state": human_state,
        "target_end_state": target_state,
        "final_state_match": human_state == target_state,
    }


def reserved_policy_target(
    model_start: int,
    human_events: Sequence[BinaryTransition],
) -> tuple[BinaryTransition, ...]:
    """Reserve correction; use the real slot only when odd parity is valid."""

    real = tuple(human_events[-1:]) if len(human_events) % 2 == 1 else ()
    return (correction_transition(model_start),) + real


def retained_up_down_pairs(
    human_events: Sequence[BinaryTransition], retained_real: Sequence[BinaryTransition]
) -> tuple[int, int]:
    raw_pairs = 0
    retained_pairs = 0
    retained_ticks = tuple(event.source_tick for event in retained_real)
    for left, right in zip(human_events, human_events[1:]):
        if left.direction == "UP" and right.direction == "DOWN":
            raw_pairs += 1
            if len(retained_ticks) >= 2:
                for index in range(len(retained_ticks) - 1):
                    if retained_ticks[index : index + 2] == (left.source_tick, right.source_tick):
                        retained_pairs += 1
                        break
    return raw_pairs, retained_pairs


class FollowupAccumulator:
    def __init__(self, split: str, region: str) -> None:
        self.split = split
        self.region = region
        self.intervals = 0
        self.by_k: dict[str, Counter[str]] = {bucket: Counter() for bucket in BUCKETS}
        self.correction_retained = 0
        self.correction_discarded = 0
        self.current_raw_real = 0
        self.current_retained_real = 0
        self.current_final_matches = 0
        self.current_raw_up_down = 0
        self.current_retained_up_down = 0
        self.reserved_retained_real = 0
        self.reserved_final_matches = 0
        self.reserved_retained_up_down = 0
        self.recovery_tau: list[float] = []
        self.mismatch_fraction: list[float] = []

    def add_bulk(self, human_k: int, count: int) -> None:
        if human_k not in (0, 1):
            raise ValueError("bulk path is only valid for K=0/1")
        bucket = count_bucket(human_k)
        target = self.by_k[bucket]
        target["intervals"] += count
        target["correction_retained"] += count
        target["raw_real"] += human_k * count
        target["current_retained_real"] += human_k * count
        target["reserved_retained_real"] += human_k * count
        self.intervals += count
        self.correction_retained += count
        self.current_raw_real += human_k * count
        self.current_retained_real += human_k * count
        self.reserved_retained_real += human_k * count
        self.current_final_matches += count
        self.reserved_final_matches += count

    def add_analyzed(self, analysis: Mapping[str, Any]) -> None:
        human_k = int(analysis["human_k"])
        bucket = count_bucket(human_k)
        target = self.by_k[bucket]
        target["intervals"] += 1
        target["correction_discarded"] += 1
        target["raw_real"] += human_k
        target["current_retained_real"] += int(analysis["current_retained_real"])
        target["reserved_retained_real"] += int(analysis["reserved_retained_real"])
        self.intervals += 1
        self.correction_discarded += 1
        self.current_raw_real += human_k
        self.current_retained_real += int(analysis["current_retained_real"])
        self.reserved_retained_real += int(analysis["reserved_retained_real"])
        self.current_final_matches += int(analysis["current_final_match"])
        self.reserved_final_matches += int(analysis["reserved_final_match"])
        self.current_raw_up_down += int(analysis["raw_up_down"])
        self.current_retained_up_down += int(analysis["current_retained_up_down"])
        self.reserved_retained_up_down += int(analysis["reserved_retained_up_down"])
        self.recovery_tau.append(float(analysis["first_recovery_tau"]))
        self.mismatch_fraction.append(float(analysis["mismatch_fraction"]))

    def survival_rows(self) -> list[dict[str, Any]]:
        rows = []
        for bucket in (*BUCKETS, "ALL_K"):
            if bucket == "ALL_K":
                intervals = self.intervals
                retained = self.correction_retained
                discarded = self.correction_discarded
                raw_real = self.current_raw_real
                current_real = self.current_retained_real
                reserved_real = self.reserved_retained_real
            else:
                item = self.by_k[bucket]
                intervals = item["intervals"]
                retained = item["correction_retained"]
                discarded = item["correction_discarded"]
                raw_real = item["raw_real"]
                current_real = item["current_retained_real"]
                reserved_real = item["reserved_retained_real"]
            rows.append({
                "split": self.split,
                "interval_type": self.region,
                "human_K_bucket": bucket,
                "interval_count": intervals,
                "correction_retained_count": retained,
                "correction_retained_fraction": ratio(retained, intervals),
                "correction_discarded_count": discarded,
                "correction_discarded_fraction": ratio(discarded, intervals),
                "human_raw_transition_count": raw_real,
                "current_retained_human_transition_count": current_real,
                "reserved_retained_human_transition_count": reserved_real,
            })
        return rows

    def payload(self) -> dict[str, Any]:
        return {
            "intervals": self.intervals,
            "correction_retained_count": self.correction_retained,
            "correction_retained_fraction": ratio(self.correction_retained, self.intervals),
            "correction_discarded_count": self.correction_discarded,
            "correction_discarded_fraction": ratio(self.correction_discarded, self.intervals),
            "by_human_K": {
                bucket: {
                    "intervals": self.by_k[bucket]["intervals"],
                    "correction_retained": self.by_k[bucket]["correction_retained"],
                    "correction_discarded": self.by_k[bucket]["correction_discarded"],
                    "survival_fraction": ratio(
                        self.by_k[bucket]["correction_retained"], self.by_k[bucket]["intervals"]
                    ),
                }
                for bucket in BUCKETS
            },
            "discarded_recovery_tau": distribution(self.recovery_tau),
            "discarded_state_mismatch_fraction": distribution(self.mismatch_fraction),
            "policies": {
                POLICY_CURRENT: {
                    "immediate_correction_count": self.correction_retained,
                    "immediate_correction_rate": ratio(self.correction_retained, self.intervals),
                    "human_raw_transitions": self.current_raw_real,
                    "retained_real_human_transitions": self.current_retained_real,
                    "real_transition_retention": ratio(self.current_retained_real, self.current_raw_real),
                    "final_state_matches": self.current_final_matches,
                    "final_state_preservation": ratio(self.current_final_matches, self.intervals),
                    "raw_up_down_pairs": self.current_raw_up_down,
                    "retained_up_down_pairs": self.current_retained_up_down,
                    "up_down_pair_retention": ratio(self.current_retained_up_down, self.current_raw_up_down),
                },
                POLICY_RESERVED: {
                    "immediate_correction_count": self.intervals,
                    "immediate_correction_rate": 1.0,
                    "human_raw_transitions": self.current_raw_real,
                    "retained_real_human_transitions": self.reserved_retained_real,
                    "real_transition_retention": ratio(self.reserved_retained_real, self.current_raw_real),
                    "final_state_matches": self.reserved_final_matches,
                    "final_state_preservation": ratio(self.reserved_final_matches, self.intervals),
                    "raw_up_down_pairs": self.current_raw_up_down,
                    "retained_up_down_pairs": self.reserved_retained_up_down,
                    "up_down_pair_retention": ratio(self.reserved_retained_up_down, self.current_raw_up_down),
                },
            },
        }


def analyze_discarded_interval(
    human_start: int, human_events: Sequence[BinaryTransition]
) -> dict[str, Any]:
    human_events = tuple(human_events)
    human_k = len(human_events)
    if human_k < 2:
        raise ValueError("discarded-correction analysis requires K>=2")
    model_start = 1 - human_start
    current = reconcile_and_compress(model_start, human_start, human_events)
    if any(event.synthetic for event in current):
        raise AssertionError("frozen correction unexpectedly survived K>=2")
    reserved = reserved_policy_target(model_start, human_events)
    current_replay = replay_trajectory(human_start, model_start, human_events, current)
    reserved_replay = replay_trajectory(human_start, model_start, human_events, reserved)
    if not current_replay["final_state_match"] or not reserved_replay["final_state_match"]:
        raise AssertionError("policy failed final-state preservation")
    current_real = tuple(event for event in current if not event.synthetic)
    reserved_real = tuple(event for event in reserved if not event.synthetic)
    raw_up_down, current_up_down = retained_up_down_pairs(human_events, current_real)
    _, reserved_up_down = retained_up_down_pairs(human_events, reserved_real)
    return {
        "human_k": human_k,
        "current_retained_real": len(current_real),
        "reserved_retained_real": len(reserved_real),
        "current_final_match": current_replay["final_state_match"],
        "reserved_final_match": reserved_replay["final_state_match"],
        "raw_up_down": raw_up_down,
        "current_retained_up_down": current_up_down,
        "reserved_retained_up_down": reserved_up_down,
        "first_recovery_tau": current_replay["first_recovery_tau"],
        "mismatch_fraction": current_replay["mismatch_fraction"],
    }


def event_objects(
    directions: Sequence[str], ticks: Sequence[int], taus: Sequence[float]
) -> tuple[BinaryTransition, ...]:
    return tuple(
        BinaryTransition(direction, min(1.0, max(0.0, float(tau))), int(tick))
        for direction, tick, tau in zip(directions, ticks, taus)
    )


def add_sequence(
    accumulators: Mapping[tuple[str, str], FollowupAccumulator],
    split: str,
    region: str,
    human_start: int,
    events: Sequence[BinaryTransition],
) -> None:
    targets = (accumulators[(split, region)], accumulators[(split, "ALL")])
    if len(events) <= 1:
        for target in targets:
            target.add_bulk(len(events), 1)
        return
    analysis = analyze_discarded_interval(human_start, events)
    for target in targets:
        target.add_analyzed(analysis)


def run_tests() -> dict[str, Any]:
    pytest_available = importlib.util.find_spec("pytest") is not None
    direct = subprocess.run(
        [sys.executable, str(TEST_FILE)], cwd=ROOT, text=True, capture_output=True, check=False
    )
    if direct.returncode != 0:
        raise RuntimeError(f"direct test suite failed: {direct.stderr}")
    results = ast.literal_eval(direct.stdout.strip())
    payload: dict[str, Any] = {
        "pytest_available": pytest_available,
        "execution_mode": "direct plain test-function runner",
        "command": f"python {TEST_FILE.relative_to(ROOT)}",
        "returncode": direct.returncode,
        "test_functions": results,
        "passed": len(results),
        "failed": 0,
    }
    if pytest_available:
        pytest_run = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", str(TEST_FILE.relative_to(ROOT))],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
        payload["pytest_command"] = "python -m pytest -q tests/test_binary_2slot_transition_audit_v0.py"
        payload["pytest_returncode"] = pytest_run.returncode
        payload["pytest_stdout"] = pytest_run.stdout.strip()
        if pytest_run.returncode != 0:
            raise RuntimeError(f"pytest suite failed: {pytest_run.stdout}\n{pytest_run.stderr}")
    return payload


def weight_payload(prior: Mapping[str, Any]) -> dict[str, Any]:
    distribution_rows = prior["count_targets"]["train"]["base_oracle"]["ALL"]
    counts = tuple(int(distribution_rows[str(index)]["count"]) for index in range(3))
    frequencies, raw, weights = canonical_inverse_sqrt_weights(counts)
    weighted_sum = sum(frequency * weight for frequency, weight in zip(frequencies, weights))
    if not math.isclose(weighted_sum, 1.0, rel_tol=1e-12, abs_tol=1e-12):
        raise AssertionError("canonical weight normalization failed")
    return {
        "basis": "train/base_oracle/ALL",
        "counts": {str(i): counts[i] for i in range(3)},
        "frequencies": {str(i): frequencies[i] for i in range(3)},
        "raw_inverse_sqrt_weights": {str(i): raw[i] for i in range(3)},
        "canonical_normalized_weights": {str(i): weights[i] for i in range(3)},
        "formula": "raw_w_c=1/sqrt(f_c); w_c=raw_w_c/sum_j(f_j*raw_w_j)",
        "weighted_frequency_sum": weighted_sum,
        "tolerance": {"relative": 1e-12, "absolute": 1e-12},
        "normalization_check_passed": True,
        "legacy_global_helper_modified": False,
    }


def scan(entries: Sequence[Mapping[str, Any]], output: Path) -> dict[str, Any]:
    accumulators = {
        (split, region): FollowupAccumulator(split, region)
        for split in ("train", "validation") for region in REGIONS
    }
    started = time.monotonic()
    for entry_index, entry in enumerate(entries, 1):
        split = str(entry["split"])
        source_path = Path(entry["source_midi"])
        raw = parse_raw_midi(source_path)
        direct = direct_binary_crossings(raw.raw_cc64)
        binary_ticks = np.asarray([tick for tick, _ in direct], dtype=np.int64)
        directions = tuple(direction for _, direction in direct)
        cache_path = CACHE_ROOT / str(entry["cache_file"])
        with np.load(cache_path, allow_pickle=False) as cache:
            onset_ticks = cache["onset_ticks"].astype(np.int64, copy=True)
            note_end_tick = int(cache["latest_note_off_tick"])
        tick_to_seconds, _ = _tick_second_converters(source_path)
        event_seconds = np.asarray(
            [tick_to_seconds(int(tick)) for tick in binary_ticks], dtype=np.float64
        )
        first_tick = int(onset_ticks[0])
        first_seconds = float(tick_to_seconds(first_tick))
        end_seconds = float(tick_to_seconds(note_end_tick))

        pre_start = bisect.bisect_left(event_seconds.tolist(), first_seconds - 1.0)
        pre_end = int(np.searchsorted(binary_ticks, first_tick, side="left"))
        pre_events = event_objects(
            directions[pre_start:pre_end], binary_ticks[pre_start:pre_end].tolist(),
            ((event_seconds[pre_start:pre_end] - (first_seconds - 1.0)) / 1.0).tolist(),
        )
        add_sequence(accumulators, split, "PRE", pre_start % 2, pre_events)

        post_start = int(np.searchsorted(binary_ticks, note_end_tick, side="left"))
        post_end = bisect.bisect_right(event_seconds.tolist(), end_seconds + 1.0)
        post_events = event_objects(
            directions[post_start:post_end], binary_ticks[post_start:post_end].tolist(),
            ((event_seconds[post_start:post_end] - end_seconds) / 1.0).tolist(),
        )
        add_sequence(accumulators, split, "POST", post_start % 2, post_events)

        main_lo = int(np.searchsorted(binary_ticks, first_tick, side="left"))
        main_hi = int(np.searchsorted(binary_ticks, note_end_tick, side="left"))
        modeled_ticks = binary_ticks[main_lo:main_hi]
        interval_ids = np.searchsorted(onset_ticks, modeled_ticks, side="right") - 1
        counts = np.bincount(interval_ids, minlength=len(onset_ticks)).astype(np.int64)
        for k in (0, 1):
            frequency = int(np.sum(counts == k))
            accumulators[(split, "MAIN")].add_bulk(k, frequency)
            accumulators[(split, "ALL")].add_bulk(k, frequency)
        if modeled_ticks.size:
            changes = np.flatnonzero(np.diff(interval_ids)) + 1
            starts = np.r_[0, changes]
            ends = np.r_[changes, len(interval_ids)]
            for start, end in zip(starts.tolist(), ends.tolist()):
                human_k = end - start
                if human_k < 2:
                    continue
                interval_index = int(interval_ids[start])
                left_tick = int(onset_ticks[interval_index])
                right_tick = int(
                    note_end_tick if interval_index + 1 == len(onset_ticks)
                    else onset_ticks[interval_index + 1]
                )
                left_seconds = float(tick_to_seconds(left_tick))
                right_seconds = float(tick_to_seconds(right_tick))
                duration = right_seconds - left_seconds
                if duration <= 0.0:
                    raise ValueError(f"non-positive interval seconds: {source_path}")
                global_start = main_lo + start
                global_end = main_lo + end
                events = event_objects(
                    directions[global_start:global_end],
                    binary_ticks[global_start:global_end].tolist(),
                    ((event_seconds[global_start:global_end] - left_seconds) / duration).tolist(),
                )
                analysis = analyze_discarded_interval(global_start % 2, events)
                accumulators[(split, "MAIN")].add_analyzed(analysis)
                accumulators[(split, "ALL")].add_analyzed(analysis)
        if entry_index == 1 or entry_index % 50 == 0 or entry_index == len(entries):
            message = f"{now()} followup_scan {entry_index}/{len(entries)} elapsed={time.monotonic()-started:.1f}s"
            print(message, flush=True)
            with (output / "followup.log").open("a", encoding="utf-8") as handle:
                handle.write(message + "\n")
    return {
        split: {region: accumulators[(split, region)].payload() for region in REGIONS}
        for split in ("train", "validation")
    }, [
        row
        for split in ("train", "validation")
        for region in REGIONS
        for row in accumulators[(split, region)].survival_rows()
    ]


def recovery_rows(results: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for split in ("train", "validation"):
        for region in REGIONS:
            item = results[split][region]
            recovery = item["discarded_recovery_tau"]
            mismatch = item["discarded_state_mismatch_fraction"]
            rows.append({
                "split": split, "interval_type": region,
                "correction_discarded_count": item["correction_discarded_count"],
                "recovery_tau_count": recovery["count"],
                "recovery_tau_median": recovery["median"], "recovery_tau_q3": recovery["q3"],
                "recovery_tau_p95": recovery["p95"], "recovery_tau_p99": recovery["p99"],
                "recovery_tau_max": recovery["max"],
                "mismatch_fraction_count": mismatch["count"],
                "mismatch_fraction_median": mismatch["median"], "mismatch_fraction_q3": mismatch["q3"],
                "mismatch_fraction_p95": mismatch["p95"], "mismatch_fraction_p99": mismatch["p99"],
                "mismatch_fraction_max": mismatch["max"],
            })
    return rows


def markdown(summary: Mapping[str, Any]) -> str:
    weights = summary["weight_normalization"]
    results = summary["correction_survival"]
    test = summary["tests"]
    survival_lines = []
    recovery_lines = []
    policy_lines = []
    for split in ("train", "validation"):
        for region in REGIONS:
            item = results[split][region]
            survival_lines.append(
                f"| {split} | {region} | {item['intervals']:,} | {item['correction_retained_count']:,} "
                f"({percent(item['correction_retained_fraction'])}) | {item['correction_discarded_count']:,} "
                f"({percent(item['correction_discarded_fraction'])}) |"
            )
            recovery = item["discarded_recovery_tau"]
            mismatch = item["discarded_state_mismatch_fraction"]
            recovery_lines.append(
                f"| {split} | {region} | {recovery['count']:,} | {recovery['median']} | {recovery['q3']} | "
                f"{recovery['p95']} | {recovery['p99']} | {recovery['max']} | {mismatch['median']} | "
                f"{mismatch['q3']} | {mismatch['p95']} | {mismatch['p99']} | {mismatch['max']} |"
            )
        for policy in (POLICY_CURRENT, POLICY_RESERVED):
            item = results[split]["ALL"]["policies"][policy]
            policy_lines.append(
                f"| {split} | {policy} | {percent(item['immediate_correction_rate'])} | "
                f"{item['retained_real_human_transitions']:,}/{item['human_raw_transitions']:,} "
                f"({percent(item['real_transition_retention'])}) | {percent(item['final_state_preservation'])} | "
                f"{item['retained_up_down_pairs']:,}/{item['raw_up_down_pairs']:,} "
                f"({percent(item['up_down_pair_retention'])}) |"
            )
    train_all = results["train"]["ALL"]
    validation_all = results["validation"]["ALL"]
    current_train = train_all["policies"][POLICY_CURRENT]
    reserved_train = train_all["policies"][POLICY_RESERVED]
    q1 = ", ".join(
        f"N={index}: {weights['canonical_normalized_weights'][str(index)]:.15g}"
        for index in range(3)
    )
    return f"""# Binary 2-Slot Transition Follow-up v0

## Scope and frozen inputs

- Prior audit: `{summary['provenance']['prior_audit_summary']}`; its capacity/PRE/POST scientific counts were not recomputed or changed in-place.
- Follow-up universe: canonical train 2,062 + validation 71 only. ASAP split CSV/test metadata/test MIDI access: 0.
- Frozen current policy: prepend tau=0 correction on mismatch, append human transitions, then max-2 odd/even last-retention.
- No model/head/checkpoint/training/inference, capacity change, horizon tuning, or policy adoption was performed.

## Canonical class-weight normalization correction

| N | Count | Frequency | Raw 1/sqrt(f) | Canonical normalized weight |
| ---: | ---: | ---: | ---: | ---: |
| 0 | {weights['counts']['0']:,} | {weights['frequencies']['0']:.15g} | {weights['raw_inverse_sqrt_weights']['0']:.15g} | {weights['canonical_normalized_weights']['0']:.15g} |
| 1 | {weights['counts']['1']:,} | {weights['frequencies']['1']:.15g} | {weights['raw_inverse_sqrt_weights']['1']:.15g} | {weights['canonical_normalized_weights']['1']:.15g} |
| 2 | {weights['counts']['2']:,} | {weights['frequencies']['2']:.15g} | {weights['raw_inverse_sqrt_weights']['2']:.15g} | {weights['canonical_normalized_weights']['2']:.15g} |

`sum_c f_c*w_c = {weights['weighted_frequency_sum']:.17g}`; tolerance rtol/atol `1e-12`, PASS. Legacy global helper는 수정하지 않았다.

## Correction survival

| Split | Interval | Total | Correction retained | Correction discarded |
| --- | --- | ---: | ---: | ---: |
{chr(10).join(survival_lines)}

Correction은 human K=0/1에서만 생존하고 K>=2에서 모두 discard된다. K bucket별 exact counts는 `correction_survival.csv`에 있다.

## Piecewise trajectory recovery

| Split | Interval | Discarded | Recovery tau median | Q3 | p95 | p99 | max | Mismatch fraction median | Q3 | p95 | p99 | max |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(recovery_lines)}

Recovery tau와 mismatch fraction은 first-retained-event proxy가 아니다. Human/raw와 compressed-target state를 모든 event timestamp에서 각각 재생하고, 구간별 state inequality 길이를 `[0,1]`에서 적분했다.

## Optional correction-reserved comparison

Alternative diagnostic은 correction을 slot 1에 고정한다. 남은 한 real slot은 state-valid와 final-state parity를 보존할 수 있을 때만 사용한다: odd K는 마지막 human transition 1개, even K는 0개다. 실제 tokenizer에는 채택하지 않았다.

| Split | Policy | Immediate correction | Retained real transitions | Final state | UP->DOWN pair retention |
| --- | --- | ---: | ---: | ---: | ---: |
{chr(10).join(policy_lines)}

## Unit/invariant tests

- Execution: `{test['execution_mode']}` (`{test['command']}`).
- pytest available: `{test['pytest_available']}`.
- Result: {test['passed']} test functions passed, {test['failed']} failed.
- Added coverage includes K=2 correction discard, K=5/6 compression, invalid/same-direction rejection, epsilon boundaries, same-timestamp tau=0 correction+human transition, and canonical weight normalization.

## Conclusions

**Q1. Canonical fixed class weights는?** {q1}. Weighted frequency sum은 {weights['weighted_frequency_sum']:.17g}이다.

**Q2. Counterfactual mismatch에서 correction survival은?** Train ALL {percent(train_all['correction_retained_fraction'])} ({train_all['correction_retained_count']:,}/{train_all['intervals']:,}), validation ALL {percent(validation_all['correction_retained_fraction'])} ({validation_all['correction_retained_count']:,}/{validation_all['intervals']:,})이다.

**Q3. Correction discard 시 mismatch는 언제까지 지속되는가?** Train ALL recovery tau median/Q3/p95/p99/max는 `{train_all['discarded_recovery_tau']}`이고 mismatch-duration fraction은 `{train_all['discarded_state_mismatch_fraction']}`이다. Validation 값은 본문 표와 JSON에 있다.

**Q4. Current policy의 final interval-end state preservation은 100%인가?** 예. Train/validation의 PRE/MAIN/POST/ALL 모두 100%다.

**Q5. 두 policy의 transition-information trade-off는?** Current policy는 train real-transition retention {percent(current_train['real_transition_retention'])}, UP->DOWN pair retention {percent(current_train['up_down_pair_retention'])}인 대신 immediate correction은 {percent(current_train['immediate_correction_rate'])}다. Reserved policy는 immediate correction 100%와 final state 100%를 보장하지만 real-transition retention이 {percent(reserved_train['real_transition_retention'])}, UP->DOWN pair retention이 {percent(reserved_train['up_down_pair_retention'])}로 낮아진다.

**Q6. 구현 전 blocking issue가 있는가?** Final-state correctness 측면의 blocker는 없다. 다만 correction token 자체가 K>=2에서 사라지고 piecewise mismatch가 지속되는 것은 의도적으로 수용해야 하는 measured behavior다. 자동 policy 변경은 하지 않았다.

**Q7. 추가 tests는 모두 통과했는가?** 예. {test['passed']}/{test['passed']} direct test functions가 통과했다.
"""


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.chmod(0o755)
    atomic_text(OUTPUT_ROOT / "followup.log", f"{now()} start\n")
    prior_summary_path = PRIOR_ROOT / "audit_summary.json"
    prior_report_path = PRIOR_ROOT / "BINARY_2SLOT_TRANSITION_AUDIT.md"
    prior = json.loads(prior_summary_path.read_text(encoding="utf-8"))
    if prior["hard_invariant_failure"] or prior["provenance"]["asap_test_midi_access_count"] != 0:
        raise RuntimeError("prior audit provenance/invariants are not reusable")
    tests = run_tests()
    weights = weight_payload(prior)
    _, _, entries, cache_provenance = load_frozen_manifest()
    results, survival_rows = scan(entries, OUTPUT_ROOT)
    expected = {
        "train": prior["count_targets"]["reconciliation_checks"]["counterfactual_mismatch"]["train"]["checks"],
        "validation": prior["count_targets"]["reconciliation_checks"]["counterfactual_mismatch"]["validation"]["checks"],
    }
    for split in ("train", "validation"):
        if results[split]["ALL"]["intervals"] != expected[split]:
            raise AssertionError(f"follow-up interval universe changed: {split}")
        for region in REGIONS:
            for policy in (POLICY_CURRENT, POLICY_RESERVED):
                if results[split][region]["policies"][policy]["final_state_preservation"] != 1.0:
                    raise AssertionError(f"final-state preservation failed: {split}/{region}/{policy}")
    provenance = {
        "prior_audit_summary": str(prior_summary_path),
        "prior_audit_summary_sha256": file_sha256(prior_summary_path),
        "prior_audit_report": str(prior_report_path),
        "prior_audit_report_sha256": file_sha256(prior_report_path),
        "cache_id": cache_provenance["cache_id"],
        "cache_manifest_sha256": cache_provenance["cache_manifest_sha256"],
        "asap_split_csv_opened_by_followup": False,
        "asap_test_metadata_access_count": 0,
        "asap_test_midi_access_count": 0,
        "checkpoint_access_count": 0,
        "model_steps": 0,
        "training_steps": 0,
        "validation_inference_count": 0,
    }
    summary = {
        "followup_id": "binary_2slot_transition_followup_v0",
        "completed_at": now(),
        "provenance": provenance,
        "weight_normalization": weights,
        "correction_survival": results,
        "tests": tests,
        "alternative_policy": {
            "name": POLICY_RESERVED,
            "diagnostic_only": True,
            "definition": "reserve tau=0 correction; odd human K retain last one real event, even K retain zero real events",
            "adopted": False,
        },
        "blocking_issue": False,
        "forbidden_activity": {
            "model_implementation": 0, "checkpoint_access": 0, "training_steps": 0,
            "validation_model_inference": 0, "asap_test_metadata_access": 0,
            "asap_test_midi_access": 0, "capacity_change": 0, "horizon_change": 0,
            "policy_change": 0,
        },
    }
    recovery = recovery_rows(results)
    atomic_csv(OUTPUT_ROOT / "correction_survival.csv", survival_rows, tuple(survival_rows[0]))
    atomic_csv(OUTPUT_ROOT / "correction_recovery_timing.csv", recovery, tuple(recovery[0]))
    atomic_json(OUTPUT_ROOT / "weight_normalization.json", weights)
    atomic_json(OUTPUT_ROOT / "test_results.json", tests)
    atomic_json(OUTPUT_ROOT / "followup_summary.json", summary)
    atomic_text(OUTPUT_ROOT / "BINARY_2SLOT_TRANSITION_FOLLOWUP.md", markdown(summary))
    atomic_json(OUTPUT_ROOT / "run_status.json", {
        "status": "completed", "blocking_issue": False,
        "train_intervals": results["train"]["ALL"]["intervals"],
        "validation_intervals": results["validation"]["ALL"]["intervals"],
        "asap_test_access_count": 0, "checkpoint_access_count": 0,
        "model_steps": 0, "training_steps": 0, "validation_inference_count": 0,
        "completed_at": summary["completed_at"],
    })
    with (OUTPUT_ROOT / "followup.log").open("a", encoding="utf-8") as handle:
        handle.write(f"{now()} completed\n")
    (OUTPUT_ROOT / "followup.log").chmod(0o644)
    print(json.dumps({"status": "completed", "output": str(OUTPUT_ROOT), "blocking_issue": False}))


if __name__ == "__main__":
    main()
