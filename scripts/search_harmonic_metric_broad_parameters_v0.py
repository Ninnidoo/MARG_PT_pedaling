#!/usr/bin/env python3
"""Validation-only broad parameter search for the finalized harmonic metric.

This extends, rather than replaces, the audited v0 pilot implementation.  MIDI
event predicates, note-instance types, interval weights, canonical identity
checks, and extreme-MIDI construction are imported from that pilot.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import shutil
import sys
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mido
import numpy as np
import pandas as pd

import scripts.audit_harmonic_metric_kdyn_blow_pilot_v0 as pilot
from src.stage2_binary.canonical_stage1 import (
    assert_strict_non_cc64_equality,
    cc64_schedule,
    sha256_file,
    strict_non_cc64_signature,
)


OUTPUT = REPO_ROOT / "analysis/harmonic_metric_broad_parameter_search_v0"
WORK = REPO_ROOT / "analysis/.harmonic_metric_broad_parameter_search_v0.work"
PILOT_OUTPUT = REPO_ROOT / "analysis/harmonic_metric_kdyn_blow_pilot_v0"
STAGE1_MANIFEST = REPO_ROOT / "analysis/stage2_binary_canonical_v1/canonical_validation_stage1_manifest.csv"
CUSTOM_MANIFEST = REPO_ROOT / "analysis/custom_event_model_v0_canonical_val_inference_v1/candidate_manifest.json"
METRIC_DOCUMENT = REPO_ROOT / "Metric_harmonic_consonance.md"

ALPHAS = (0.50, 0.75, 1.00, 1.25, 1.50, 2.00)
ETAS = (0.00, 0.25, 0.50, 0.75, 0.90, 1.00)
PIVOTS = (42, 48, 54, 60)
BETAS = (0.00, 0.25, 0.50, 0.75, 1.00)
KAPPAS = (0.00, 0.10, 0.20, 0.30, 0.40, 0.50)
PRIMARY_THRESHOLD = 0.05
SENSITIVITY_THRESHOLDS = (0.02, 0.05, 0.10)
MIN_VALID_ONSETS = 20
EPSILON = pilot.EPSILON
SCRIPT_VERSION = "harmonic_metric_broad_parameter_search_v0.1"
SPLIT_RULE = "exhaustive held-out subset minimizing standardized mean+SD imbalance; lexicographic tie-break"
SPLIT_SEED = 20260820

# Exact ALWAYS_ON work is cubic in long performances.  This score-blind limit
# was fixed from note-state counts before any harmonic configuration was scored.
MAX_EXACT_ALWAYS_PAIR_ONSET_WORK = 3_000_000_000
MAX_WORKERS = 4
PAIR_CHUNK = 256

MODEL_CANONICAL = (
    "STANDARD_CE_ARGMAX",
    "STANDARD_CE_POSTERIOR_MEDIAN",
    "WEIGHTED_CE_ARGMAX",
    "HYBRID_REGRESSION_ONLY",
)
MODEL_ALL = MODEL_CANONICAL + ("CUSTOM_EVENT_V0",)
SYSTEMS = tuple(pilot.SYSTEM_SPECS) + ("CUSTOM_EVENT_V0", "HUMAN", "NO_PEDAL", "ALWAYS_ON")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def stable_id(values: Sequence[Any]) -> str:
    payload = "|".join(str(value) for value in values).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def build_cases() -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    stage1 = read_csv(STAGE1_MANIFEST)
    custom = json.loads(CUSTOM_MANIFEST.read_text(encoding="utf-8"))["candidates"]
    by_piece: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in custom:
        by_piece[str(item["piece_id"])].append(item)
    cases: list[dict[str, Any]] = []
    for row in stage1:
        piece_id = row["piece_id"]
        if piece_id not in by_piece:
            continue
        chosen = sorted(by_piece[piece_id], key=lambda item: str(item["performance_path"]))[0]
        canonical = pilot.host_to_container_path(row["canonical_midi_path"])
        descriptor = pilot.midi_descriptor(canonical)
        cases.append(
            {
                "piece_id": piece_id,
                "composer": row["composer"],
                "title": row["title"],
                "performance_id": str(chosen["performance_id"]),
                "canonical_midi_path": str(canonical),
                "human_midi_path": str(pilot.host_to_container_path(chosen["source_midi"])),
                "custom_candidate_path": str(pilot.host_to_container_path(chosen["candidate_midi"])),
                **descriptor,
            }
        )
    if len(cases) != 19:
        raise AssertionError(f"expected 19-piece common validation intersection, found {len(cases)}")
    return sorted(cases, key=lambda item: item["piece_id"]), stage1


def _apply_state_group(
    messages: Sequence[Any],
    tick: int,
    seconds: float,
    active: dict[tuple[int, int], deque[pilot.NoteInstance]],
    residual: list[pilot.NoteInstance],
    pedal_on: dict[int, bool],
    identifier: int,
    force_always: bool = False,
) -> tuple[list[pilot.NoteInstance], int, int]:
    residual_events = 0
    for message in messages:
        if pilot.is_cc64(message) and not force_always:
            channel = int(message.channel)
            new_state = int(message.value) >= 64
            if pedal_on[channel] and not new_state:
                residual = [note for note in residual if note.channel != channel]
            pedal_on[channel] = new_state
        elif pilot.is_note_on(message):
            identifier += 1
            channel, pitch = int(message.channel), int(message.note)
            active[(channel, pitch)].append(
                pilot.NoteInstance(
                    identifier=identifier,
                    channel=channel,
                    pitch=pitch,
                    velocity=int(message.velocity),
                    onset_tick=tick,
                    onset_seconds=seconds,
                )
            )
        elif pilot.is_note_off(message):
            channel, pitch = int(message.channel), int(message.note)
            key = (channel, pitch)
            if active[key]:
                note = active[key].popleft()
                if force_always or pedal_on[channel]:
                    residual.append(note)
                    residual_events += 1
        elif (
            not message.is_meta
            and message.type == "control_change"
            and int(message.control) in (120, 123)
        ):
            channel = int(message.channel)
            affected: list[pilot.NoteInstance] = []
            for key in list(active):
                if key[0] == channel:
                    affected.extend(active.pop(key))
            if force_always or (int(message.control) == 123 and pedal_on[channel]):
                residual.extend(affected)
                residual_events += len(affected)
            else:
                residual = [note for note in residual if note.channel != channel]
    return residual, identifier, residual_events


def pedal_activity(path: Path, force_always: bool = False) -> dict[str, Any]:
    groups = list(pilot.merged_tick_groups(path)[1])
    onset_times = [seconds for _, seconds, messages in groups for message in messages if pilot.is_note_on(message)]
    off_times = [seconds for _, seconds, messages in groups for message in messages if pilot.is_note_off(message)]
    if not onset_times:
        raise ValueError(f"no note onset: {path}")
    start = min(onset_times)
    end = max(off_times or onset_times)
    active: dict[tuple[int, int], deque[pilot.NoteInstance]] = defaultdict(deque)
    residual: list[pilot.NoteInstance] = []
    pedal_on: dict[int, bool] = defaultdict(bool)
    identifier = 0
    residual_events = 0
    total_onsets = 0
    valid_onsets = 0
    on_area = 0.0
    episodes = 0
    previous_time = groups[0][1]
    aggregate_on = force_always
    workloads: list[int] = []
    for tick, seconds, messages in groups:
        left, right = max(previous_time, start), min(seconds, end)
        if aggregate_on and right > left:
            on_area += right - left
        before = aggregate_on
        residual, identifier, added = _apply_state_group(
            messages, tick, seconds, active, residual, pedal_on, identifier, force_always
        )
        residual_events += added
        aggregate_on = force_always or any(pedal_on.values())
        if not before and aggregate_on and start <= seconds <= end:
            episodes += 1
        if any(pilot.is_note_on(message) for message in messages):
            total_onsets += 1
            active_count = sum(len(queue) for queue in active.values())
            pedal_count = len(residual)
            pair_count = pedal_count * active_count + pedal_count * (pedal_count - 1) // 2
            workloads.append(pair_count)
            if pair_count > 0:
                valid_onsets += 1
        previous_time = seconds
    duration = max(end - start, EPSILON)
    return {
        "total_distinct_onset_count": total_onsets,
        "CC64_ON_duty_ratio": float(on_area / duration),
        "sustain_ON_episode_count": episodes,
        "human_valid_onset_count": valid_onsets,
        "human_valid_onset_fraction": float(valid_onsets / max(total_onsets, 1)),
        "pedal_sustained_residual_note_count": residual_events,
        "always_on_pair_onset_work": int(sum(workloads)),
        "always_on_max_pair_count": int(max(workloads, default=0)),
    }


def eligibility(activity: dict[str, Any], threshold: float) -> bool:
    return bool(
        int(activity["human_valid_onset_count"]) >= MIN_VALID_ONSETS
        and float(activity["human_valid_onset_fraction"]) >= threshold
        and float(activity["CC64_ON_duty_ratio"]) >= threshold
    )


def audit_pieces(cases: Sequence[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for case in cases:
        human = pedal_activity(Path(case["human_midi_path"]))
        workload = pedal_activity(Path(case["canonical_midi_path"]), force_always=True)
        row = {**case, **human}
        row["always_on_pair_onset_work"] = workload["always_on_pair_onset_work"]
        row["always_on_max_pair_count"] = workload["always_on_max_pair_count"]
        for threshold in SENSITIVITY_THRESHOLDS:
            row[f"eligible_threshold_{threshold:.2f}"] = eligibility(row, threshold)
        row["primary_status"] = "PEDAL_ELIGIBLE" if eligibility(row, PRIMARY_THRESHOLD) else "PEDAL_SPARSE"
        failures = []
        if row["human_valid_onset_count"] < MIN_VALID_ONSETS:
            failures.append(f"valid_onsets<{MIN_VALID_ONSETS}")
        if row["human_valid_onset_fraction"] < PRIMARY_THRESHOLD:
            failures.append("valid_fraction<0.05")
        if row["CC64_ON_duty_ratio"] < PRIMARY_THRESHOLD:
            failures.append("duty_ratio<0.05")
        row["primary_reason"] = "all primary criteria pass" if not failures else "; ".join(failures)
        row["exact_compute_status"] = (
            "EXACT_COMPUTABLE"
            if row["always_on_pair_onset_work"] <= MAX_EXACT_ALWAYS_PAIR_ONSET_WORK
            else "DEFERRED_LONG_FORM"
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("piece_id").reset_index(drop=True)


def split_eligible(audit: pd.DataFrame) -> pd.DataFrame:
    eligible = audit[audit.primary_status == "PEDAL_ELIGIBLE"].copy().sort_values("piece_id")
    features = [
        "low_note_fraction",
        "velocity_range",
        "note_density_per_second",
        "human_valid_onset_fraction",
        "CC64_ON_duty_ratio",
    ]
    values = eligible[features].astype(float)
    z = (values - values.mean()) / values.std(ddof=0).replace(0.0, 1.0)
    heldout_size = int(round(len(eligible) / 3))
    best: tuple[float, tuple[str, ...], tuple[int, ...]] | None = None
    for combo in itertools.combinations(range(len(eligible)), heldout_size):
        held = z.iloc[list(combo)]
        search = z.drop(z.index[list(combo)])
        imbalance = float(np.abs(search.mean() - held.mean()).sum())
        imbalance += float(np.abs(search.std(ddof=0) - held.std(ddof=0)).sum())
        ids = tuple(eligible.iloc[list(combo)].piece_id.astype(str))
        candidate = (imbalance, ids, combo)
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    held_indices = set(best[2])
    rows = []
    for position, row in enumerate(eligible.itertuples(index=False)):
        role = "HELD_OUT_CHECK" if position in held_indices else "PARAMETER_SEARCH"
        rows.append(
            {
                "piece_id": row.piece_id,
                "performance_id": row.performance_id,
                "composer": row.composer,
                "title": row.title,
                "split_role": role,
                "exact_compute_status": row.exact_compute_status,
                "included_in_exact_analysis": row.exact_compute_status == "EXACT_COMPUTABLE",
                "split_rule": SPLIT_RULE,
                "deterministic_seed": SPLIT_SEED,
                "split_imbalance_objective": best[0],
                **{name: getattr(row, name) for name in features},
            }
        )
    return pd.DataFrame(rows).sort_values(["split_role", "piece_id"]).reset_index(drop=True)


def ensure_extremes(
    piece_ids: set[str], audit: pd.DataFrame, work: Path
) -> tuple[dict[tuple[str, str], Path], pd.DataFrame]:
    old_manifest = pd.read_csv(PILOT_OUTPUT / "reusable_extremes/manifest.csv")
    old_by_key = {(row.piece_id, row.variant): row for row in old_manifest.itertuples()}
    paths: dict[tuple[str, str], Path] = {}
    records = []
    by_piece = audit.set_index("piece_id")
    for piece_id in sorted(piece_ids):
        source = Path(by_piece.loc[piece_id, "canonical_midi_path"])
        source_hash = sha256_file(source)
        for variant, subdir in (("NO_PEDAL", "no_pedal"), ("ALWAYS_ON", "always_on")):
            key = (piece_id, variant)
            reused = False
            if key in old_by_key:
                prior = old_by_key[key]
                path = Path(prior.generated_midi_path)
                if prior.source_sha256 != source_hash or sha256_file(path) != prior.generated_sha256:
                    raise AssertionError(f"pilot extreme SHA drift: {piece_id} {variant}")
                reused = True
            else:
                path = work / "reusable_extremes" / subdir / f"{piece_id}.mid"
                if not path.exists():
                    pilot.rebuild_extreme(source, path, variant)
            identity = assert_strict_non_cc64_equality(source, path)
            note_exact = pilot.note_signature(source) == pilot.note_signature(path)
            schedule = cc64_schedule(path)
            if not identity["passed"] or not note_exact:
                raise AssertionError(f"extreme identity failure: {piece_id} {variant}")
            if variant == "NO_PEDAL" and schedule:
                raise AssertionError(f"NO_PEDAL has CC64: {path}")
            if variant == "ALWAYS_ON" and not schedule:
                raise AssertionError(f"ALWAYS_ON has no CC64: {path}")
            paths[key] = path
            records.append(
                {
                    "piece_id": piece_id,
                    "variant": variant,
                    "source_canonical_midi_path": str(source),
                    "extreme_midi_path": str(path),
                    "source_sha256": source_hash,
                    "extreme_sha256": sha256_file(path),
                    "non_CC64_exact_identity": True,
                    "note_exact_identity": note_exact,
                    "reused_from_pilot": reused,
                    "creation_script_version": prior.creation_script_version if reused else SCRIPT_VERSION,
                }
            )
    return paths, pd.DataFrame(records)


def resolve_paths(
    cases: Sequence[dict[str, Any]], piece_ids: set[str], extremes: dict[tuple[str, str], Path]
) -> tuple[dict[tuple[str, str], Path], pd.DataFrame]:
    by_piece = {case["piece_id"]: case for case in cases}
    paths: dict[tuple[str, str], Path] = {}
    rows = []
    for piece_id in sorted(piece_ids):
        case = by_piece[piece_id]
        for system, spec in pilot.SYSTEM_SPECS.items():
            path = (
                Path(case["canonical_midi_path"])
                if spec.get("path_kind") == "manifest"
                else Path(spec["root"]) / piece_id / "candidate.mid"
            )
            paths[(piece_id, system)] = path
        paths[(piece_id, "CUSTOM_EVENT_V0")] = Path(case["custom_candidate_path"])
        paths[(piece_id, "HUMAN")] = Path(case["human_midi_path"])
        paths[(piece_id, "NO_PEDAL")] = extremes[(piece_id, "NO_PEDAL")]
        paths[(piece_id, "ALWAYS_ON")] = extremes[(piece_id, "ALWAYS_ON")]
        for system in SYSTEMS:
            path = paths[(piece_id, system)]
            if not path.is_file():
                raise FileNotFoundError(path)
            rows.append(
                {
                    "piece_id": piece_id,
                    "performance_id": case["performance_id"],
                    "system": system,
                    "midi_path": str(path),
                    "sha256": sha256_file(path),
                }
            )
    return paths, pd.DataFrame(rows)


def _strengths(notes: Sequence[pilot.NoteInstance], onset_time: float, alphas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pitch = np.asarray([note.pitch for note in notes], dtype=np.int64)
    if not notes:
        return pitch, np.empty((len(alphas), 0), dtype=np.float64)
    velocity = np.asarray([note.velocity for note in notes], dtype=np.float64)
    age = np.maximum(onset_time - np.asarray([note.onset_seconds for note in notes]), 0.0)
    t60 = np.asarray(pilot.t60_seconds(pitch), dtype=np.float64)
    strength = (velocity[None, :] / 127.0) * np.power(
        10.0, -3.0 * age[None, :] / (alphas[:, None] * t60[None, :])
    )
    return pitch, strength


def _pair_block_multi(
    pitch_a: np.ndarray,
    pitch_b: np.ndarray,
    strength_a: np.ndarray,
    strength_b: np.ndarray,
    pivots: np.ndarray,
    triangular: bool,
) -> tuple[np.ndarray, np.ndarray, int]:
    alpha_count = strength_a.shape[0]
    if pitch_a.size == 0 or pitch_b.size == 0:
        return np.zeros(alpha_count), np.zeros((alpha_count, len(pivots))), 0
    intervals = pilot.interval_class(pitch_a[:, None], pitch_b[None, :])
    hall = pilot.HALL_WEIGHTS[intervals]
    if triangular:
        mask = np.triu(np.ones(hall.shape, dtype=bool), 1)
    else:
        mask = np.ones(hall.shape, dtype=bool)
    hall_values = hall[mask]
    low_pairs = []
    for pivot in pivots:
        low_a = np.clip((pivot - pitch_a.astype(float)) / (pivot - 21.0), 0.0, 1.0)
        low_b = np.clip((pivot - pitch_b.astype(float)) / (pivot - 21.0), 0.0, 1.0)
        low_pairs.append(((low_a[:, None] + low_b[None, :]) / 2.0)[mask])
    low_values = np.asarray(low_pairs)
    base = np.zeros(alpha_count, dtype=np.float64)
    low = np.zeros((alpha_count, len(pivots)), dtype=np.float64)
    for alpha_index in range(alpha_count):
        aa = strength_a[alpha_index, :, None]
        bb = strength_b[alpha_index, None, :]
        wdec = 2.0 * aa * bb / (aa + bb + EPSILON)
        contributions = wdec[mask] * hall_values
        base[alpha_index] = contributions.sum(dtype=np.float64)
        low[alpha_index] = (low_values * contributions[None, :]).sum(axis=1, dtype=np.float64)
    return base, low, int(mask.sum())


def _onset_stats_multi(
    active: Sequence[pilot.NoteInstance],
    residual: Sequence[pilot.NoteInstance],
    onset_time: float,
    alphas: np.ndarray,
    pivots: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    active_pitch, active_strength = _strengths(active, onset_time, alphas)
    pedal_pitch, pedal_strength = _strengths(residual, onset_time, alphas)
    pa_base, pa_low, n_pa = _pair_block_multi(
        pedal_pitch, active_pitch, pedal_strength, active_strength, pivots, False
    )
    pp_base = np.zeros(len(alphas), dtype=np.float64)
    pp_low = np.zeros((len(alphas), len(pivots)), dtype=np.float64)
    n_pp = 0
    count = len(residual)
    for start in range(0, count, PAIR_CHUNK):
        stop = min(start + PAIR_CHUNK, count)
        base, low, number = _pair_block_multi(
            pedal_pitch[start:stop], pedal_pitch[start:stop],
            pedal_strength[:, start:stop], pedal_strength[:, start:stop], pivots, True
        )
        pp_base += base
        pp_low += low
        n_pp += number
        if stop < count:
            base, low, number = _pair_block_multi(
                pedal_pitch[start:stop], pedal_pitch[stop:],
                pedal_strength[:, start:stop], pedal_strength[:, stop:], pivots, False
            )
            pp_base += base
            pp_low += low
            n_pp += number
    return pa_base, pa_low, pp_base, pp_low, n_pa, n_pp


def compute_sufficient_statistics(
    path: Path, alphas: Sequence[float], pivots: Sequence[int]
) -> dict[str, np.ndarray]:
    alpha_array = np.asarray(alphas, dtype=np.float64)
    pivot_array = np.asarray(pivots, dtype=np.float64)
    active: dict[tuple[int, int], deque[pilot.NoteInstance]] = defaultdict(deque)
    residual: list[pilot.NoteInstance] = []
    pedal_on: dict[int, bool] = defaultdict(bool)
    identifier = 0
    rows: dict[str, list[Any]] = defaultdict(list)
    for tick, seconds, messages in pilot.merged_tick_groups(path)[1]:
        has_onset = any(pilot.is_note_on(message) for message in messages)
        residual, identifier, _ = _apply_state_group(
            messages, tick, seconds, active, residual, pedal_on, identifier, False
        )
        if not has_onset:
            continue
        current_active = [note for queue in active.values() for note in queue]
        n_pa = len(residual) * len(current_active)
        n_pp = len(residual) * (len(residual) - 1) // 2
        if n_pa + n_pp == 0:
            continue
        pa_base, pa_low, pp_base, pp_low, checked_pa, checked_pp = _onset_stats_multi(
            current_active, residual, seconds, alpha_array, pivot_array
        )
        if checked_pa != n_pa or checked_pp != n_pp:
            raise AssertionError("pair count mismatch in sufficient-stat computation")
        rows["onset_tick"].append(tick)
        rows["onset_time"].append(seconds)
        rows["vbar"].append(float(np.mean([note.velocity for note in current_active])))
        rows["N_PA"].append(n_pa)
        rows["N_PP"].append(n_pp)
        rows["PA_base"].append(pa_base)
        rows["PA_low"].append(pa_low)
        rows["PP_base"].append(pp_base)
        rows["PP_low"].append(pp_low)
    if rows["vbar"]:
        velocities = np.asarray(rows["vbar"], dtype=np.float64)
        q10, q90 = np.quantile(velocities, [0.1, 0.9])
        d_context = (
            np.full_like(velocities, 0.5)
            if q90 == q10
            else np.clip((velocities - q10) / (q90 - q10), 0.0, 1.0)
        )
    else:
        q10 = q90 = math.nan
        d_context = np.empty(0, dtype=np.float64)
    return {
        "alpha_values": alpha_array,
        "pivot_values": pivot_array,
        "onset_tick": np.asarray(rows["onset_tick"], dtype=np.int64),
        "onset_time": np.asarray(rows["onset_time"], dtype=np.float64),
        "vbar": np.asarray(rows["vbar"], dtype=np.float64),
        "D_n": d_context,
        "N_PA": np.asarray(rows["N_PA"], dtype=np.int64),
        "N_PP": np.asarray(rows["N_PP"], dtype=np.int64),
        "PA_base": np.asarray(rows["PA_base"], dtype=np.float64).reshape(-1, len(alpha_array)),
        "PA_low": np.asarray(rows["PA_low"], dtype=np.float64).reshape(-1, len(alpha_array), len(pivot_array)),
        "PP_base": np.asarray(rows["PP_base"], dtype=np.float64).reshape(-1, len(alpha_array)),
        "PP_low": np.asarray(rows["PP_low"], dtype=np.float64).reshape(-1, len(alpha_array), len(pivot_array)),
        "q10": np.asarray([q10], dtype=np.float64),
        "q90": np.asarray([q90], dtype=np.float64),
    }


def _cache_worker(task: tuple[str, str, str, str, tuple[float, ...], tuple[int, ...]]) -> dict[str, Any]:
    piece_id, system, midi_path, cache_path, alphas, pivots = task
    destination = Path(cache_path)
    if destination.exists():
        with np.load(destination) as loaded:
            if np.array_equal(loaded["alpha_values"], np.asarray(alphas)) and np.array_equal(
                loaded["pivot_values"], np.asarray(pivots)
            ):
                return {"piece_id": piece_id, "system": system, "cache_path": cache_path, "reused": True}
    stats = compute_sufficient_statistics(Path(midi_path), alphas, pivots)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **stats)
    os.replace(temporary, destination)
    return {"piece_id": piece_id, "system": system, "cache_path": cache_path, "reused": False}


def compute_caches(
    piece_ids: Sequence[str],
    paths: dict[tuple[str, str], Path],
    cache_root: Path,
    alphas: Sequence[float],
    pivots: Sequence[int],
) -> pd.DataFrame:
    tasks = []
    for piece_id in sorted(piece_ids):
        for system in SYSTEMS:
            if system == "NO_PEDAL":
                continue
            cache_path = cache_root / f"{piece_id}__{system}.npz"
            tasks.append((piece_id, system, str(paths[(piece_id, system)]), str(cache_path), tuple(alphas), tuple(pivots)))
    rows = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_cache_worker, task): task for task in tasks}
        for completed, future in enumerate(as_completed(futures), 1):
            row = future.result()
            rows.append(row)
            print(f"CACHE {completed}/{len(tasks)} {row['piece_id']} {row['system']} reused={row['reused']}", flush=True)
    return pd.DataFrame(rows)


def load_stats(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as loaded:
        return {key: loaded[key] for key in loaded.files}


def score_stats(
    stats: dict[str, np.ndarray], alpha: float, eta: float, pivot: int, beta: float, kappa: float
) -> float:
    if len(stats["N_PA"]) == 0:
        return 0.0
    ai = int(np.flatnonzero(np.isclose(stats["alpha_values"], alpha, rtol=0, atol=1e-12))[0])
    mi = int(np.flatnonzero(np.isclose(stats["pivot_values"], pivot, rtol=0, atol=1e-12))[0])
    numerator = (
        stats["PA_base"][:, ai]
        + beta * stats["PA_low"][:, ai, mi]
        + eta * (stats["PP_base"][:, ai] + beta * stats["PP_low"][:, ai, mi])
    )
    denominator = stats["N_PA"] + eta * stats["N_PP"] + EPSILON
    hbase = numerator / denominator
    hn = hbase * (1.0 + kappa * (2.0 * stats["D_n"] - 1.0) * np.sign(hbase))
    return float(hn.mean())


def parameter_grid() -> pd.DataFrame:
    rows = []
    alpha_index = {value: index for index, value in enumerate(ALPHAS)}
    eta_index = {value: index for index, value in enumerate(ETAS)}
    pivot_index = {value: index for index, value in enumerate(PIVOTS)}
    beta_index = {value: index for index, value in enumerate(BETAS)}
    kappa_index = {value: index for index, value in enumerate(KAPPAS)}
    reference_indices = (alpha_index[1.0], eta_index[0.9], pivot_index[48], beta_index[0.0], kappa_index[0.0])
    prior_pairs = {(0.0, 0.0), (0.0, 0.1), (0.1, 0.0), (0.1, 0.1)}
    for values in itertools.product(ALPHAS, ETAS, PIVOTS, BETAS, KAPPAS):
        alpha, eta, pivot, beta, kappa = values
        indices = (alpha_index[alpha], eta_index[eta], pivot_index[pivot], beta_index[beta], kappa_index[kappa])
        rows.append(
            {
                "config_id": "coarse_" + stable_id(values),
                "alpha_decay": alpha,
                "eta_PP": eta,
                "m0": pivot,
                "beta_low": beta,
                "kappa_dyn": kappa,
                "reference_distance_grid_L1": int(sum(abs(a - b) for a, b in zip(indices, reference_indices))),
                "is_hall_decay_backbone": values == (1.0, 0.9, 48, 0.0, 0.0),
                "is_previous_pilot_reference": alpha == 1.0 and eta == 0.9 and pivot == 48 and (kappa, beta) in prior_pairs,
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) != 4320:
        raise AssertionError(f"coarse grid size {len(frame)} != 4320")
    return frame


def score_grid_for_piece_system(stats: dict[str, np.ndarray], grid: pd.DataFrame) -> np.ndarray:
    result = np.empty(len(grid), dtype=np.float64)
    grouped = grid.groupby(["alpha_decay", "m0"], sort=False)
    for (alpha, pivot), group in grouped:
        ai = int(np.flatnonzero(np.isclose(stats["alpha_values"], alpha, rtol=0, atol=1e-12))[0])
        mi = int(np.flatnonzero(np.isclose(stats["pivot_values"], pivot, rtol=0, atol=1e-12))[0])
        pa_base = stats["PA_base"][:, ai]
        pa_low = stats["PA_low"][:, ai, mi]
        pp_base = stats["PP_base"][:, ai]
        pp_low = stats["PP_low"][:, ai, mi]
        for (eta, beta), sub in group.groupby(["eta_PP", "beta_low"], sort=False):
            numerator = pa_base + beta * pa_low + eta * (pp_base + beta * pp_low)
            hbase = numerator / (stats["N_PA"] + eta * stats["N_PP"] + EPSILON)
            dynamic_slope = float(((2.0 * stats["D_n"] - 1.0) * np.abs(hbase)).mean()) if len(hbase) else 0.0
            base_mean = float(hbase.mean()) if len(hbase) else 0.0
            for row in sub.itertuples():
                result[row.Index] = base_mean + float(row.kappa_dyn) * dynamic_slope
    if not np.isfinite(result).all():
        raise AssertionError("non-finite grid score")
    return result


def score_search(
    piece_ids: Sequence[str], cache_root: Path, grid: pd.DataFrame, audit: pd.DataFrame
) -> pd.DataFrame:
    performance = audit.set_index("piece_id").performance_id.to_dict()
    rows = []
    for piece_id in sorted(piece_ids):
        for system in SYSTEMS:
            scores = (
                np.zeros(len(grid), dtype=np.float64)
                if system == "NO_PEDAL"
                else score_grid_for_piece_system(load_stats(cache_root / f"{piece_id}__{system}.npz"), grid)
            )
            rows.append(
                pd.DataFrame(
                    {
                        "piece_id": piece_id,
                        "performance_id": performance[piece_id],
                        "system": system,
                        "config_id": grid.config_id,
                        "H_piece": scores,
                    }
                )
            )
    frame = pd.concat(rows, ignore_index=True)
    if not np.isfinite(frame.H_piece).all() or not (frame[frame.system == "NO_PEDAL"].H_piece == 0.0).all():
        raise AssertionError("search score integrity failure")
    return frame
