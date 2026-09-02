#!/usr/bin/env python3
"""Pure Hall-negative evaluation using audited pedal note-state semantics."""

from __future__ import annotations

import math
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import scripts.audit_harmonic_metric_kdyn_blow_pilot_v0 as pilot
import scripts.search_harmonic_metric_broad_parameters_v0 as broad

PITCH_COUNT = 128
PITCH_AXIS = np.arange(PITCH_COUNT, dtype=np.int64)
INTERVAL_MATRIX = pilot.interval_class(PITCH_AXIS[:, None], PITCH_AXIS[None, :])
HALL_MATRIX = pilot.HALL_WEIGHTS[INTERVAL_MATRIX]
NEGATIVE_MATRIX = np.maximum(-HALL_MATRIX, 0.0)
NEGATIVE_INDICATOR = NEGATIVE_MATRIX > 0
MAX_NEGATIVE_HALL = float(NEGATIVE_MATRIX.max())
FINAL_ACCUMULATION_COEFFICIENT = 0.05


def pitch_counts(notes: Sequence[pilot.NoteInstance]) -> np.ndarray:
    """Retain note-instance multiplicity while reducing interval lookup to pitch counts."""
    return np.bincount([note.pitch for note in notes], minlength=PITCH_COUNT).astype(np.int64)


def empty_stats() -> dict[str, Any]:
    return {"pair_count": 0, "negative_pair_count": 0, "negative_mass": 0.0, "max_negative": 0.0}


def pa_stats(pedal_counts: np.ndarray, active_counts: np.ndarray) -> dict[str, Any]:
    p_idx, a_idx = np.flatnonzero(pedal_counts), np.flatnonzero(active_counts)
    if not len(p_idx) or not len(a_idx):
        return empty_stats()
    multiplicity = np.outer(pedal_counts[p_idx], active_counts[a_idx])
    negative = NEGATIVE_MATRIX[np.ix_(p_idx, a_idx)]
    indicator = NEGATIVE_INDICATOR[np.ix_(p_idx, a_idx)]
    return {
        "pair_count": int(multiplicity.sum()),
        "negative_pair_count": int(multiplicity[indicator].sum()),
        "negative_mass": float((multiplicity * negative).sum(dtype=np.float64)),
        "max_negative": float(negative[indicator].max()) if np.any(indicator) else 0.0,
    }


def pp_stats(pedal_counts: np.ndarray) -> dict[str, Any]:
    p_idx = np.flatnonzero(pedal_counts)
    if not len(p_idx):
        return empty_stats()
    counts = pedal_counts[p_idx]
    multiplicity = np.triu(np.outer(counts, counts), k=1)
    diagonal = counts * (counts - 1) // 2
    multiplicity[np.diag_indices_from(multiplicity)] = diagonal
    negative = NEGATIVE_MATRIX[np.ix_(p_idx, p_idx)]
    indicator = (multiplicity > 0) & NEGATIVE_INDICATOR[np.ix_(p_idx, p_idx)]
    return {
        "pair_count": int(multiplicity.sum()),
        "negative_pair_count": int(multiplicity[indicator].sum()),
        "negative_mass": float((multiplicity * negative).sum(dtype=np.float64)),
        "max_negative": float(negative[indicator].max()) if np.any(indicator) else 0.0,
    }


def combine(pa: dict[str, Any], pp: dict[str, Any]) -> dict[str, Any]:
    pair_count = int(pa["pair_count"] + pp["pair_count"])
    negative_count = int(pa["negative_pair_count"] + pp["negative_pair_count"])
    negative_mass = float(pa["negative_mass"] + pp["negative_mass"])
    return {
        "pair_count": pair_count,
        "negative_pair_count": negative_count,
        "negative_mass": negative_mass,
        "hall_neg_mean": negative_mass / pair_count if pair_count else 0.0,
        "hall_neg_pair_fraction": negative_count / pair_count if pair_count else 0.0,
        "hall_neg_conditional": negative_mass / negative_count if negative_count else 0.0,
        "hall_neg_max": max(float(pa["max_negative"]), float(pp["max_negative"])),
    }


def pair_statistics_from_pitch_counts(pedal_counts: np.ndarray, active_counts: np.ndarray) -> dict[str, Any]:
    pa, pp = pa_stats(pedal_counts, active_counts), pp_stats(pedal_counts)
    combined = combine(pa, pp)
    combined["PA"], combined["PP"] = {**pa, **combine(pa, empty_stats())}, {**pp, **combine(empty_stats(), pp)}
    return combined


def pair_statistics_from_pitch_lists(pedal_pitches: Sequence[int], active_pitches: Sequence[int]) -> dict[str, Any]:
    pedal = np.bincount(list(pedal_pitches), minlength=PITCH_COUNT).astype(np.int64)
    active = np.bincount(list(active_pitches), minlength=PITCH_COUNT).astype(np.int64)
    return pair_statistics_from_pitch_counts(pedal, active)


def aggregate_final_harmonic_metric(
    onset_rows: Sequence[Mapping[str, Any]],
) -> dict[str, int | float]:
    """Aggregate frozen final Harmonic Muddiness from native onset rows.

    H_mean averages Hall-negative severity only over onsets with a non-empty
    pedal-induced pair set. A_acc averages log1p(raw mass) over every onset,
    including empty-pair onsets. The empty-performance extension is all-zero.
    """

    total_onset_count = len(onset_rows)
    valid_h: list[float] = []
    accumulation: list[float] = []
    for onset_index, row in enumerate(onset_rows):
        pair_count = int(row["N_pair_n"])
        raw = float(row["negative_hall_mass_n"])
        if pair_count < 0 or not math.isfinite(raw) or raw < 0.0:
            raise ValueError(
                f"invalid harmonic onset statistics at index {onset_index}: "
                f"N_pair_n={pair_count}, negative_hall_mass_n={raw}"
            )
        if pair_count == 0:
            if raw != 0.0:
                raise ValueError(
                    f"empty Q_n must have RAW_n=0 at index {onset_index}: {raw}"
                )
        else:
            valid_h.append(raw / pair_count)
        accumulation.append(math.log1p(raw))

    valid_onset_count = len(valid_h)
    h_mean = math.fsum(valid_h) / valid_onset_count if valid_onset_count else 0.0
    a_acc = (
        math.fsum(accumulation) / total_onset_count
        if total_onset_count
        else 0.0
    )
    m_harm = h_mean + FINAL_ACCUMULATION_COEFFICIENT * a_acc
    if not all(math.isfinite(value) and value >= 0.0 for value in (h_mean, a_acc, m_harm)):
        raise ValueError("final Harmonic Muddiness aggregation must be finite and nonnegative")
    return {
        "H_mean": h_mean,
        "A_acc": a_acc,
        "M_harm": m_harm,
        "total_onset_count": total_onset_count,
        "valid_onset_count": valid_onset_count,
    }


def evaluate_midi(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate all distinct onsets without acoustic-strength or performance-intensity terms."""
    active: dict[tuple[int, int], deque[pilot.NoteInstance]] = defaultdict(deque)
    residual: list[pilot.NoteInstance] = []
    pedal_on: dict[int, bool] = defaultdict(bool)
    identifier = 0
    onset_index = 0
    rows: list[dict[str, Any]] = []
    max_active_instances = 0
    max_residual_instances = 0

    for tick, seconds, messages in pilot.merged_tick_groups(path)[1]:
        has_onset = any(pilot.is_note_on(message) for message in messages)
        residual, identifier, _ = broad._apply_state_group(
            messages, tick, seconds, active, residual, pedal_on, identifier, False
        )
        if not has_onset:
            continue
        onset_index += 1
        current_active = [note for queue in active.values() for note in queue]
        stats = pair_statistics_from_pitch_counts(pitch_counts(residual), pitch_counts(current_active))
        pa, pp = stats["PA"], stats["PP"]
        max_active_instances = max(max_active_instances, len(current_active))
        max_residual_instances = max(max_residual_instances, len(residual))
        rows.append({
            "onset_index": onset_index, "onset_tick": int(tick), "onset_time": float(seconds),
            "A_n_count": len(current_active), "P_n_count": len(residual),
            "PA_pair_count": int(pa["pair_count"]), "PP_pair_count": int(pp["pair_count"]),
            "N_pair_n": int(stats["pair_count"]), "N_neg_n": int(stats["negative_pair_count"]),
            "negative_hall_mass_n": float(stats["negative_mass"]),
            "HALL_NEG_MEAN_n": float(stats["hall_neg_mean"]),
            "HALL_NEG_PAIR_FRACTION_n": float(stats["hall_neg_pair_fraction"]),
            "HALL_NEG_CONDITIONAL_n": float(stats["hall_neg_conditional"]),
            "HALL_NEG_MAX_n": float(stats["hall_neg_max"]),
            "PA_negative_pair_count": int(pa["negative_pair_count"]),
            "PP_negative_pair_count": int(pp["negative_pair_count"]),
            "PA_negative_hall_mass_n": float(pa["negative_mass"]),
            "PP_negative_hall_mass_n": float(pp["negative_mass"]),
            "HALL_NEG_MEAN_PA_n": float(pa["hall_neg_mean"]),
            "HALL_NEG_MEAN_PP_n": float(pp["hall_neg_mean"]),
            "NEGATIVE_PAIR_FRACTION_PA_n": float(pa["hall_neg_pair_fraction"]),
            "NEGATIVE_PAIR_FRACTION_PP_n": float(pp["hall_neg_pair_fraction"]),
            "AA_pair_count": 0,
        })
    return rows, {
        "distinct_onset_count": len(rows), "max_active_note_instances": max_active_instances,
        "max_residual_note_instances": max_residual_instances,
    }


def evaluate_final_harmonic_metric(
    path: Path,
) -> tuple[dict[str, int | float], dict[str, Any]]:
    """Evaluate one MIDI and return the frozen performance-level headline."""

    onset_rows, diagnostics = evaluate_midi(path)
    return aggregate_final_harmonic_metric(onset_rows), diagnostics
