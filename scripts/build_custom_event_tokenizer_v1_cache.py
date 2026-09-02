#!/usr/bin/env python3
"""Build the frozen piece-level Custom Event Tokenizer v1 target cache."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_custom_event_tokenizer_oracle_audit_v0 import load_manifests, now
from src.stage2_event_tokenizer.audit import inverse_sqrt_mean_one
from src.stage2_event_tokenizer.tokenizer import EVENT_NAMES, STATE_NAMES
from src.stage2_event_tokenizer.tokenizer_v1 import (
    MAIN_CAPACITY,
    SAME_TIMESTAMP_RULE_VERSION,
    TERMINAL_CAPACITY,
    TOKENIZER_IMPLEMENTATION_VERSION,
    TOKENIZER_VERSION,
    EncodedPerformanceV1,
    apply_terminal_tokens,
    encode_midi_v1,
)
from src.stage2_event_tokenizer.tokenizer import apply_tokens

DEFAULT_OUTPUT = ROOT / "analysis/custom_event_tokenizer_v1"
PRIOR_CAPACITY = ROOT / "analysis/custom_event_tokenizer_capacity_terminal_audit_v1/capacity_train_stats.csv"
PRIOR_TERMINAL = ROOT / "analysis/custom_event_tokenizer_terminal_micro_audit_v0/terminal_capacity_curve.csv"

SPEC_PAYLOAD = {
    "tokenizer_version": TOKENIZER_VERSION,
    "tokenizer_implementation_version": TOKENIZER_IMPLEMENTATION_VERSION,
    "states": {"ZERO": [0, 25], "LOW": [26, 63], "HALF": [64, 103], "FULL": [104, 127]},
    "representatives": [0, 51, 79, 127],
    "event_vocabulary": list(EVENT_NAMES),
    "initial_state": "state immediately before first distinct onset; strict event tick < first onset",
    "main_timeline": "distinct onset; nonterminal [t_i,t_(i+1)); final [t_M,latest_noteoff] inclusive",
    "main_capacity": MAIN_CAPACITY,
    "main_compression": "chronological_last_6",
    "main_timing": "exact_tau; targets nonterminal [0,1), final [0,1]",
    "main_decoder_safety": "clip predicted tau to [0,1]",
    "terminal_anchor": "latest non-pedal note-off; strict event tick > anchor",
    "terminal_capacity": TERMINAL_CAPACITY,
    "terminal_compression": "chronological_last_4",
    "terminal_coordinate": "inter_event_gap; first retained event re-anchored to latest note-off",
    "terminal_transform": "z=log1p(gap_seconds)",
    "terminal_decoder_safety": "clamp predicted z >= 0, then gap=expm1(z)",
    "same_timestamp_rule_version": SAME_TIMESTAMP_RULE_VERSION,
    "note_order": "complete non-drum notes sorted by (onset,pitch,track,channel,noteoff,velocity)",
    "representative_note_index": "last note index in each global distinct-onset group",
    "window_ownership": "not defined in this cache version",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def scalar(value: Any):
    return np.asarray(value)


def cache_arrays(
    encoded: EncodedPerformanceV1,
    row: Mapping[str, str],
    *,
    performance_index: int,
    cache_id: str,
    source_sha256: str,
) -> dict[str, Any]:
    intervals = encoded.main_intervals
    main_event = np.asarray(
        [[slot.event_class for slot in interval.slots] for interval in intervals], dtype=np.int8
    )
    main_tau = np.asarray(
        [[0.0 if slot.tau is None else slot.tau for slot in interval.slots] for interval in intervals],
        dtype=np.float64,
    )
    main_mask = main_event != 0
    terminal_event = np.asarray([slot.event_class for slot in encoded.terminal.slots], dtype=np.int8)
    terminal_log_gap = np.asarray(
        [0.0 if slot.log1p_gap is None else slot.log1p_gap for slot in encoded.terminal.slots],
        dtype=np.float64,
    )
    terminal_mask = terminal_event != 0
    main_total = sum(len(interval.all_events) for interval in intervals)
    main_retained = int(main_mask.sum())
    main_crossings = sum(
        bool(event.source_crossing) for interval in intervals for event in interval.all_events
    )
    main_crossings_retained = sum(
        bool(slot.source_crossing) for interval in intervals for slot in interval.slots if slot.event_class
    )
    terminal_total = len(encoded.terminal.all_events)
    terminal_retained = int(terminal_mask.sum())
    terminal_crossings = sum(bool(event.source_crossing) for event in encoded.terminal.all_events)
    terminal_crossings_retained = sum(
        bool(slot.source_crossing) for slot in encoded.terminal.slots if slot.event_class
    )
    main_final_ok = all(
        apply_tokens(interval.start_state, interval.all_events)
        == apply_tokens(interval.start_state, interval.slots)
        for interval in intervals
    )
    terminal_final_ok = (
        apply_terminal_tokens(encoded.terminal.start_state, encoded.terminal.all_events)
        == apply_terminal_tokens(encoded.terminal.start_state, encoded.terminal.slots)
    )
    note_rows = encoded.note_order
    return {
        "cache_id": scalar(cache_id),
        "tokenizer_version": scalar(TOKENIZER_VERSION),
        "performance_index": scalar(performance_index),
        "performance_id": scalar(row.get("performance_path", "")),
        "piece_id": scalar(row.get("piece_id", "")),
        "source_dataset": scalar(row.get("source", "ASAP-validation")),
        "dataset_split": scalar(row.get("dataset_split", "validation")),
        "source_midi": scalar(row["performance_absolute_path"]),
        "source_sha256": scalar(source_sha256),
        "ticks_per_beat": scalar(encoded.source.ticks_per_beat),
        "latest_note_off_tick": scalar(encoded.source.latest_note_off),
        "non_pedal_note_count": scalar(encoded.non_pedal_note_count),
        "note_onset_ticks": np.asarray([note[3] for note in note_rows], dtype=np.int64),
        "note_pitch": np.asarray([note[2] for note in note_rows], dtype=np.int16),
        "note_offset_ticks": np.asarray([note[4] for note in note_rows], dtype=np.int64),
        "note_velocity": np.asarray([note[5] for note in note_rows], dtype=np.int16),
        "distinct_onset_count": scalar(len(encoded.onset_groups)),
        "raw_parser_distinct_onset_count": scalar(len(encoded.source.distinct_onsets)),
        "incomplete_note_only_onset_count": scalar(
            len(set(encoded.source.distinct_onsets) - {group.onset_tick for group in encoded.onset_groups})
        ),
        "onset_ticks": np.asarray([group.onset_tick for group in encoded.onset_groups], dtype=np.int64),
        "onset_first_note_index": np.asarray([group.first_note_index for group in encoded.onset_groups], dtype=np.int64),
        "onset_last_note_index": np.asarray([group.last_note_index for group in encoded.onset_groups], dtype=np.int64),
        "onset_representative_note_index": np.asarray(
            [group.representative_note_index for group in encoded.onset_groups], dtype=np.int64
        ),
        "initial_state": scalar(encoded.initial_state).astype(np.int8),
        "main_interval_left_tick": np.asarray([interval.left_tick for interval in intervals], dtype=np.int64),
        "main_interval_right_tick": np.asarray([interval.right_tick for interval in intervals], dtype=np.int64),
        "main_is_final_interval": np.asarray(
            [interval.is_final_main_interval for interval in intervals], dtype=np.bool_
        ),
        "main_interval_start_state": np.asarray([interval.start_state for interval in intervals], dtype=np.int8),
        "main_event_target": main_event,
        "main_tau_target": main_tau,
        "main_timing_valid_mask": main_mask,
        "terminal_event_target": terminal_event,
        "terminal_log1p_gap_target": terminal_log_gap,
        "terminal_timing_valid_mask": terminal_mask,
        "main_effective_events_before_cap": scalar(main_total),
        "main_retained_events": scalar(main_retained),
        "main_crossings_before_cap": scalar(main_crossings),
        "main_crossings_retained": scalar(main_crossings_retained),
        "terminal_effective_events_before_cap": scalar(terminal_total),
        "terminal_retained_events": scalar(terminal_retained),
        "terminal_crossings_before_cap": scalar(terminal_crossings),
        "terminal_crossings_retained": scalar(terminal_crossings_retained),
        "main_final_state_preservation": scalar(main_final_ok),
        "terminal_final_state_preservation": scalar(terminal_final_ok),
    }


def valid_existing_cache(path: Path, cache_id: str, source_sha256: str) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as cache:
            return (
                str(cache["cache_id"].item()) == cache_id
                and str(cache["source_sha256"].item()) == source_sha256
                and cache["main_event_target"].ndim == 2
                and cache["main_event_target"].shape[1] == MAIN_CAPACITY
                and cache["terminal_event_target"].shape == (TERMINAL_CAPACITY,)
                and cache["main_timing_valid_mask"].dtype == np.bool_
                and cache["terminal_timing_valid_mask"].dtype == np.bool_
            )
    except Exception:
        return False


class Stats:
    def __init__(self, split: str):
        self.split = split
        self.performances = 0
        self.notes = 0
        self.onsets = 0
        self.intervals = 0
        self.initial = np.zeros(4, dtype=np.int64)
        self.main_counts = np.zeros((MAIN_CAPACITY, len(EVENT_NAMES)), dtype=np.int64)
        self.terminal_counts = np.zeros((TERMINAL_CAPACITY, len(EVENT_NAMES)), dtype=np.int64)
        self.main_tau = [[] for _ in range(MAIN_CAPACITY)]
        self.terminal_log_gap = [[] for _ in range(TERMINAL_CAPACITY)]
        self.main_total = self.main_retained = 0
        self.main_cross = self.main_cross_retained = 0
        self.terminal_total = self.terminal_retained = 0
        self.terminal_cross = self.terminal_cross_retained = 0
        self.main_final_ok = self.terminal_final_ok = 0
        self.incomplete_note_only_onsets = 0

    def add(self, path: Path) -> None:
        with np.load(path, allow_pickle=False) as cache:
            self.performances += 1
            self.notes += int(cache["non_pedal_note_count"])
            self.onsets += int(cache["distinct_onset_count"])
            self.intervals += len(cache["main_event_target"])
            self.incomplete_note_only_onsets += int(cache["incomplete_note_only_onset_count"])
            self.initial[int(cache["initial_state"])] += 1
            for index in range(MAIN_CAPACITY):
                values = cache["main_event_target"][:, index]
                self.main_counts[index] += np.bincount(values, minlength=len(EVENT_NAMES))
                mask = cache["main_timing_valid_mask"][:, index]
                self.main_tau[index].extend(cache["main_tau_target"][:, index][mask].tolist())
            for index in range(TERMINAL_CAPACITY):
                value = int(cache["terminal_event_target"][index])
                self.terminal_counts[index, value] += 1
                if bool(cache["terminal_timing_valid_mask"][index]):
                    self.terminal_log_gap[index].append(float(cache["terminal_log1p_gap_target"][index]))
            for name in (
                "main_effective_events_before_cap", "main_retained_events",
                "main_crossings_before_cap", "main_crossings_retained",
                "terminal_effective_events_before_cap", "terminal_retained_events",
                "terminal_crossings_before_cap", "terminal_crossings_retained",
            ):
                setattr(self, {
                    "main_effective_events_before_cap": "main_total",
                    "main_retained_events": "main_retained",
                    "main_crossings_before_cap": "main_cross",
                    "main_crossings_retained": "main_cross_retained",
                    "terminal_effective_events_before_cap": "terminal_total",
                    "terminal_retained_events": "terminal_retained",
                    "terminal_crossings_before_cap": "terminal_cross",
                    "terminal_crossings_retained": "terminal_cross_retained",
                }[name], getattr(self, {
                    "main_effective_events_before_cap": "main_total",
                    "main_retained_events": "main_retained",
                    "main_crossings_before_cap": "main_cross",
                    "main_crossings_retained": "main_cross_retained",
                    "terminal_effective_events_before_cap": "terminal_total",
                    "terminal_retained_events": "terminal_retained",
                    "terminal_crossings_before_cap": "terminal_cross",
                    "terminal_crossings_retained": "terminal_cross_retained",
                }[name]) + int(cache[name]))
            self.main_final_ok += int(bool(cache["main_final_state_preservation"]))
            self.terminal_final_ok += int(bool(cache["terminal_final_state_preservation"]))

    @staticmethod
    def timing(values: list[float]) -> dict[str, float | int]:
        if not values:
            return {"valid_count": 0}
        array = np.asarray(values, dtype=np.float64)
        return {
            "valid_count": len(values), "mean": float(array.mean()),
            "p50": float(np.quantile(array, .5)), "p95": float(np.quantile(array, .95)),
        }

    def payload(self) -> dict[str, Any]:
        return {
            "split": self.split, "performances": self.performances,
            "non_pedal_notes": self.notes, "distinct_onsets": self.onsets,
            "main_intervals": self.intervals,
            "incomplete_note_only_onsets_excluded": self.incomplete_note_only_onsets,
            "initial_state": {
                name: {"count": int(self.initial[index]), "proportion": float(self.initial[index] / self.performances)}
                for index, name in enumerate(STATE_NAMES)
            },
            "main_timing_by_slot": {
                f"Slot{index+1}": self.timing(values) for index, values in enumerate(self.main_tau)
            },
            "terminal_timing_by_slot": {
                f"Slot{index+1}": self.timing(values)
                for index, values in enumerate(self.terminal_log_gap)
            },
            "main_structural": {
                "events_before_cap": self.main_total, "events_retained": self.main_retained,
                "event_retention": self.main_retained / self.main_total,
                "crossings_before_cap": self.main_cross, "crossings_retained": self.main_cross_retained,
                "crossing_retention": self.main_cross_retained / self.main_cross,
                "final_state_preservation": self.main_final_ok / self.performances,
            },
            "terminal_structural": {
                "events_before_cap": self.terminal_total, "events_retained": self.terminal_retained,
                "event_retention": self.terminal_retained / self.terminal_total,
                "crossings_before_cap": self.terminal_cross,
                "crossings_retained": self.terminal_cross_retained,
                "crossing_retention": self.terminal_cross_retained / self.terminal_cross,
                "final_state_preservation": self.terminal_final_ok / self.performances,
            },
        }


def distribution_rows(split: str, counts: np.ndarray, scope: str) -> list[dict[str, Any]]:
    rows = []
    for slot_index, values in enumerate(counts):
        total = int(values.sum())
        for class_index, name in enumerate(EVENT_NAMES):
            rows.append({
                "split": split, "scope": scope, "slot": slot_index + 1,
                "event_class": class_index, "event_name": name,
                "count": int(values[class_index]),
                "proportion": float(values[class_index] / total),
            })
    return rows


def weights_payload(stats: Stats) -> dict[str, Any]:
    return {
        "diagnostic_only": True,
        "formula": "w_c proportional to 1/sqrt(f_c), mean of present-class weights normalized to 1",
        "initial_state": dict(zip(STATE_NAMES, inverse_sqrt_mean_one(stats.initial))),
        "main_slots": {
            f"Slot{index+1}": dict(zip(EVENT_NAMES, inverse_sqrt_mean_one(values)))
            for index, values in enumerate(stats.main_counts)
        },
        "terminal_slots": {
            f"Slot{index+1}": dict(zip(EVENT_NAMES, inverse_sqrt_mean_one(values)))
            for index, values in enumerate(stats.terminal_counts)
        },
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def verify_regressions(train_stats: dict[str, Any], validation_stats: dict[str, Any]) -> dict[str, Any]:
    prior_main = next(
        row for row in read_csv(PRIOR_CAPACITY)
        if row["dataset"] == "combined" and row["variant"] == "k6_last6"
    )
    prior_terminal = {
        row["dataset"]: row for row in read_csv(PRIOR_TERMINAL) if row["capacity"] == "4"
    }
    checks = {
        "main_train_events_exact": train_stats["main_structural"]["events_before_cap"] == int(prior_main["effective_events"])
        and train_stats["main_structural"]["events_retained"] == int(prior_main["retained_events"]),
        "main_train_crossings_exact": train_stats["main_structural"]["crossings_before_cap"] == int(prior_main["crossings_before"])
        and train_stats["main_structural"]["crossings_retained"] == int(prior_main["crossings_retained"]),
        "terminal_train_exact": train_stats["terminal_structural"]["events_before_cap"] == int(prior_terminal["train"]["total_terminal_events"])
        and train_stats["terminal_structural"]["events_retained"] == int(prior_terminal["train"]["retained_events"])
        and train_stats["terminal_structural"]["crossings_retained"] == int(prior_terminal["train"]["retained_threshold_crossings"]),
        "terminal_validation_exact": validation_stats["terminal_structural"]["events_before_cap"] == int(prior_terminal["validation"]["total_terminal_events"])
        and validation_stats["terminal_structural"]["events_retained"] == int(prior_terminal["validation"]["retained_events"])
        and validation_stats["terminal_structural"]["crossings_retained"] == int(prior_terminal["validation"]["retained_threshold_crossings"]),
    }
    if not all(checks.values()):
        raise AssertionError(f"audit regression mismatch: {checks}")
    return checks


def spec_markdown(cache_id: str) -> str:
    return f"""# Custom Event Tokenizer Specification v1 (Frozen)

- Cache ID: `{cache_id}`
- Implementation version: `{TOKENIZER_IMPLEMENTATION_VERSION}`
- Raw provenance: canonical original human-performance MIDI CC64 timestamp/value events only.
- States: ZERO 0–25, LOW 26–63, HALF 64–103, FULL 104–127; representatives `[0,51,79,127]`.
- Initial: state immediately before first distinct onset; strict `< first onset`; default ZERO.
- Vocabulary: `NONE / SET_ZERO / SET_LOW / SET_HALF / SET_FULL`; same-state raw changes suppressed.
- Same timestamp: v0 deterministic `(tick,track,message_index)`, last effective state; no new ordering token.
- Main intervals: nonterminal `[t_i,t_(i+1))`; final main interval `[t_M, latest note-off]` inclusive.
- Main capacity: K=6, chronological final six on overflow; final state preserved exactly.
- Main target timing: exact tau; nonterminal `[0,1)`, final `[0,1]`; predicted tau is clipped to `[0,1]` by decoder.
- Terminal: events strictly after latest note-off; K_T=4, chronological final four.
- Terminal timing: retained first gap is re-anchored to latest note-off, later gaps are inter-event; target `log1p(gap_seconds)`.
- Terminal decoder: clamp predicted z to `>=0`, apply `expm1`, cumulatively sum from latest note-off.
- Note order: `(onset,pitch,track,channel,noteoff,velocity)`; onset representative is its last note index.
- Onset groups use complete non-drum notes. A malformed unmatched note-on has no model note representation and is excluded with an explicit cache diagnostic.
- Cache scope: one global piece-level initial target and one terminal target set; no window ownership semantics.
- Event targets are always valid; timing targets are valid iff event != NONE. Finite zero placeholders must never be consumed without the stored mask.
- Known limitation: ~2.3617% effective events exactly on onset and 193 cross-track/interleaved ambiguous cases from prior audit.
- ASAP test access: zero. Repedal: zero.
"""


def report_markdown(cache_id: str, train: dict[str, Any], validation: dict[str, Any], checks: dict[str, Any], reused: int) -> str:
    return f"""# Custom Event Tokenizer v1 Cache Report

- Cache ID: `{cache_id}`
- Train: {train['performances']} performances; validation: {validation['performances']} performances.
- Cache level: full performance/piece; window ownership not implemented.
- Reused valid cache files during this invocation: {reused}.
- ASAP test access: 0. Model/head/loss/training: 0.
- Complete-note onset diagnostic: train excluded {train[incomplete_note_only_onsets_excluded]} unmatched-note-only onset(s); validation excluded {validation[incomplete_note_only_onsets_excluded]}.

## Structural regression

| Split | Main event retention | Main crossing retention | Terminal event retention | Terminal crossing retention |
| --- | ---: | ---: | ---: | ---: |
| Train | {train['main_structural']['event_retention']:.9f} | {train['main_structural']['crossing_retention']:.9f} | {train['terminal_structural']['event_retention']:.9f} | {train['terminal_structural']['crossing_retention']:.9f} |
| Validation | {validation['main_structural']['event_retention']:.9f} | {validation['main_structural']['crossing_retention']:.9f} | {validation['terminal_structural']['event_retention']:.9f} | {validation['terminal_structural']['crossing_retention']:.9f} |

- Regression checks: `{checks}`.
- Main interval final-state preservation: train/validation 100%.
- Terminal final-state preservation: train/validation 100%.
- Timing masks are explicit; NONE timing placeholders are finite zero and invalid by mask.
- Candidate inverse-sqrt frequency weights are diagnostic only; no loss was implemented.
- Full strict verification: `cache_verification.json`; real-MIDI regression: `small_real_data_regression.json`.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_root
    train_rows, validation_rows, provenance = load_manifests()
    manifest_identity = {
        "train_manifest": provenance["train_manifest"],
        "train_manifest_sha256": provenance["train_manifest_sha256"],
        "validation_manifest": provenance["validation_membership_manifest"],
        "validation_manifest_sha256": provenance["validation_membership_manifest_sha256"],
    }
    fingerprint_payload = {"spec": SPEC_PAYLOAD, "manifests": manifest_identity}
    cache_id = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    existing_config = output / "config.json"
    if existing_config.exists() and json.loads(existing_config.read_text())["cache_id"] != cache_id:
        raise RuntimeError("existing output has a different cache ID")
    atomic_json(existing_config, {
        "cache_id": cache_id, "fingerprint_payload": fingerprint_payload,
        "source_provenance": provenance, "train_performances": len(train_rows),
        "validation_performances": len(validation_rows), "cache_format": "atomic per-performance compressed NPZ",
        "asap_test_access_count": 0, "repedal_execution_count": 0,
        "model_steps": 0, "training_steps": 0, "window_ownership_defined": False,
    })
    atomic_text(output / "CUSTOM_EVENT_TOKENIZER_SPEC_V1.md", spec_markdown(cache_id))
    manifest_entries = []
    reused = 0
    started = time.monotonic()
    for split, rows in (("train", train_rows), ("validation", validation_rows)):
        split_dir = output / "cache" / split
        for index, row in enumerate(rows):
            source_path = Path(row["performance_absolute_path"])
            source_sha = sha256_file(source_path)
            key = hashlib.sha256(row["performance_path"].encode("utf-8")).hexdigest()[:16]
            cache_path = split_dir / f"{index:04d}_{key}.npz"
            was_reused = valid_existing_cache(cache_path, cache_id, source_sha)
            if was_reused:
                reused += 1
            else:
                encoded = encode_midi_v1(source_path)
                arrays = cache_arrays(
                    encoded, row, performance_index=index, cache_id=cache_id, source_sha256=source_sha
                )
                atomic_npz(cache_path, arrays)
                if not valid_existing_cache(cache_path, cache_id, source_sha):
                    raise AssertionError(f"new cache failed strict reload: {cache_path}")
            with np.load(cache_path, allow_pickle=False) as cache:
                manifest_entries.append({
                    "split": split, "performance_index": index,
                    "source": row.get("source", "ASAP-validation"),
                    "piece_id": row.get("piece_id", ""), "performance_path": row["performance_path"],
                    "source_midi": row["performance_absolute_path"], "source_sha256": source_sha,
                    "cache_file": str(cache_path.relative_to(output)),
                    "cache_file_bytes": cache_path.stat().st_size,
                    "notes": int(cache["non_pedal_note_count"]),
                    "onsets": int(cache["distinct_onset_count"]),
                    "main_intervals": len(cache["main_event_target"]),
                    "reused": was_reused,
                })
            if index == 0 or (index + 1) % 100 == 0 or index + 1 == len(rows):
                print(f"cache {split} {index+1}/{len(rows)} elapsed={time.monotonic()-started:.1f}s", flush=True)
    train_stats_obj, validation_stats_obj = Stats("train"), Stats("validation")
    for entry in manifest_entries:
        target = train_stats_obj if entry["split"] == "train" else validation_stats_obj
        target.add(output / entry["cache_file"])
    train_stats, validation_stats = train_stats_obj.payload(), validation_stats_obj.payload()
    if train_stats["performances"] != 2062 or validation_stats["performances"] != 71:
        raise AssertionError("canonical cache performance count mismatch")
    checks = verify_regressions(train_stats, validation_stats)
    if any(
        payload["final_state_preservation"] != 1.0
        for stats in (train_stats, validation_stats)
        for payload in (stats["main_structural"], stats["terminal_structural"])
    ):
        raise AssertionError("cache final-state preservation is not exact")
    atomic_json(output / "cache_manifest.json", {
        "cache_id": cache_id, "entries": manifest_entries,
        "train_count": 2062, "validation_count": 71,
        "asap_test_access_count": 0,
    })
    atomic_json(output / "train_structural_stats.json", train_stats)
    atomic_json(output / "validation_structural_stats.json", validation_stats)
    atomic_csv(
        output / "main_slot_distributions.csv",
        distribution_rows("train", train_stats_obj.main_counts, "main")
        + distribution_rows("validation", validation_stats_obj.main_counts, "main"),
    )
    atomic_csv(
        output / "terminal_slot_distributions.csv",
        distribution_rows("train", train_stats_obj.terminal_counts, "terminal")
        + distribution_rows("validation", validation_stats_obj.terminal_counts, "terminal"),
    )
    atomic_json(output / "candidate_event_weights.json", {
        "train_frequency_only": True, **weights_payload(train_stats_obj)
    })
    atomic_text(
        output / "CUSTOM_EVENT_TOKENIZER_V1_CACHE_REPORT.md",
        report_markdown(cache_id, train_stats, validation_stats, checks, reused),
    )
    atomic_json(output / "run_status.json", {
        "status": "completed", "cache_id": cache_id,
        "train_cache_count": 2062, "validation_cache_count": 71,
        "main_regression_passed": checks["main_train_events_exact"] and checks["main_train_crossings_exact"],
        "terminal_regression_passed": checks["terminal_train_exact"] and checks["terminal_validation_exact"],
        "main_final_state_preservation": 1.0, "terminal_final_state_preservation": 1.0,
        "asap_test_access_count": 0, "repedal_execution_count": 0,
        "model_steps": 0, "training_steps": 0, "last_update": now(),
    })
    print(json.dumps({"cache_id": cache_id, "train": 2062, "validation": 71, "reused": reused}))


if __name__ == "__main__":
    main()
