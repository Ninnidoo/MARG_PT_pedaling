#!/usr/bin/env python3
"""Exact positive/negative decomposition built on the audited harmonic scorer."""

from __future__ import annotations

import heapq
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import scripts.audit_harmonic_metric_kdyn_blow_pilot_v0 as pilot
import scripts.search_harmonic_metric_broad_parameters_v0 as broad


ETA_PP = 0.9
EPSILON = pilot.EPSILON
PAIR_CHUNK = broad.PAIR_CHUNK
INTERVAL_NAMES = {
    1: "m2", 2: "M2", 3: "m3", 4: "M3", 5: "P4", 6: "TT",
    7: "P5", 8: "m6", 9: "M6", 10: "m7", 11: "M7", 12: "P8",
}


def _new_detail() -> dict[str, Any]:
    return {
        "negative_pair_count": 0,
        "negative_eta_mass": 0.0,
        "negative_audible_mass": 0.0,
        "audible_weight_square_sum": 0.0,
        "minimum_W_dec": np.inf,
        "pair_reconstruction_max_abs_error": 0.0,
        "q_reconstruction_max_abs_error": 0.0,
        "interval_counts": Counter(),
        "interval_negative_masses": defaultdict(float),
    }


def _empty_stats() -> dict[str, float]:
    return {
        "numerator": 0.0,
        "positive": 0.0,
        "negative": 0.0,
        "wdec": 0.0,
        "count": 0.0,
    }


def _accumulate(target: dict[str, float], source: dict[str, float]) -> None:
    for key in target:
        target[key] += float(source[key])


def decompose_pair_components(
    active: Sequence[pilot.NoteInstance],
    residual: Sequence[pilot.NoteInstance],
    onset_time: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Reuse the audited pair block and add PA/PP sign/frequency detail."""
    active_pitch = np.asarray([note.pitch for note in active], dtype=np.int64)
    active_strength = pilot.note_strength(
        np.asarray([note.velocity for note in active], dtype=np.int64),
        onset_time - np.asarray([note.onset_seconds for note in active]),
        active_pitch,
    )
    residual_pitch = np.asarray([note.pitch for note in residual], dtype=np.int64)
    residual_strength = pilot.note_strength(
        np.asarray([note.velocity for note in residual], dtype=np.int64),
        onset_time - np.asarray([note.onset_seconds for note in residual]),
        residual_pitch,
    )
    positive_heap: list[tuple[float, int, dict[str, Any]]] = []
    negative_heap: list[tuple[float, int, dict[str, Any]]] = []
    serial = 0
    stats_by_type = {"PA": _empty_stats(), "PP": _empty_stats()}
    detail_by_type = {"PA": _new_detail(), "PP": _new_detail()}

    if residual and active:
        stats, serial = pilot.pair_block_statistics(
            residual_pitch, active_pitch, residual_strength, active_strength,
            1.0, False, residual, active, onset_time, "PA",
            positive_heap, negative_heap, serial, detail_by_type["PA"],
        )
        _accumulate(stats_by_type["PA"], stats)

    count = len(residual)
    for start in range(0, count, PAIR_CHUNK):
        stop = min(start + PAIR_CHUNK, count)
        block_notes = residual[start:stop]
        stats, serial = pilot.pair_block_statistics(
            residual_pitch[start:stop], residual_pitch[start:stop],
            residual_strength[start:stop], residual_strength[start:stop],
            ETA_PP, True, block_notes, block_notes, onset_time, "PP",
            positive_heap, negative_heap, serial, detail_by_type["PP"],
        )
        _accumulate(stats_by_type["PP"], stats)
        if stop < count:
            stats, serial = pilot.pair_block_statistics(
                residual_pitch[start:stop], residual_pitch[stop:],
                residual_strength[start:stop], residual_strength[stop:],
                ETA_PP, False, block_notes, residual[stop:], onset_time, "PP",
                positive_heap, negative_heap, serial, detail_by_type["PP"],
            )
            _accumulate(stats_by_type["PP"], stats)

    combined = {
        key: stats_by_type["PA"][key] + stats_by_type["PP"][key]
        for key in stats_by_type["PA"]
    }
    combined.update(
        {
            "PA": stats_by_type["PA"],
            "PP": stats_by_type["PP"],
            "PA_detail": detail_by_type["PA"],
            "PP_detail": detail_by_type["PP"],
            "pair_reconstruction_max_abs_error": max(
                detail_by_type["PA"]["pair_reconstruction_max_abs_error"],
                detail_by_type["PP"]["pair_reconstruction_max_abs_error"],
            ),
            "q_reconstruction_max_abs_error": max(
                detail_by_type["PA"]["q_reconstruction_max_abs_error"],
                detail_by_type["PP"]["q_reconstruction_max_abs_error"],
            ),
            "minimum_W_dec": min(
                detail_by_type["PA"]["minimum_W_dec"],
                detail_by_type["PP"]["minimum_W_dec"],
            ),
        }
    )
    top_negative = []
    for rank, (_, _, record) in enumerate(sorted(negative_heap, reverse=True)[:10], 1):
        top_negative.append(
            {
                "pair_rank_within_onset": rank,
                **record,
                "negative_contribution_magnitude": -float(record["contribution_beta0"]),
            }
        )
    return combined, top_negative


def _top_interval_string(pair_stats: dict[str, Any]) -> str:
    masses: defaultdict[int, float] = defaultdict(float)
    for pair_type in ("PA", "PP"):
        for interval, mass in pair_stats[f"{pair_type}_detail"]["interval_negative_masses"].items():
            masses[int(interval)] += float(mass)
    ranked = sorted(masses.items(), key=lambda item: (-item[1], item[0]))[:4]
    return ";".join(f"{INTERVAL_NAMES[interval]}:{mass:.12g}" for interval, mass in ranked)


def evaluate_midi(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Return all-onset rows, top-onset pair packets, and pair aggregates."""
    active: dict[tuple[int, int], deque[pilot.NoteInstance]] = defaultdict(deque)
    residual: list[pilot.NoteInstance] = []
    pedal_on: dict[int, bool] = defaultdict(bool)
    identifier = 0
    onset_index = 0
    onset_rows: list[dict[str, Any]] = []
    top_onset_heap: list[tuple[float, int, dict[str, Any]]] = []
    pair_aggregate = {
        pair_type: {
            "total_pair_count": 0,
            "negative_pair_count": 0,
            "eta_weighted_pair_mass": 0.0,
            "negative_eta_mass": 0.0,
            "negative_numerator_mass": 0.0,
            "interval_counts": Counter(),
            "interval_negative_masses": defaultdict(float),
        }
        for pair_type in ("PA", "PP")
    }
    max_pair_error = 0.0
    max_q_error = 0.0
    minimum_wdec = np.inf
    max_onset_error = 0.0

    for tick, seconds, messages in pilot.merged_tick_groups(path)[1]:
        has_onset = any(pilot.is_note_on(message) for message in messages)
        residual, identifier, _ = broad._apply_state_group(
            messages, tick, seconds, active, residual, pedal_on, identifier, False
        )
        if not has_onset:
            continue
        onset_index += 1
        current_active = [note for queue in active.values() for note in queue]
        pair_stats, top_pairs = decompose_pair_components(current_active, residual, seconds)
        n_pa = int(pair_stats["PA"]["count"])
        n_pp = int(pair_stats["PP"]["count"])
        valid = n_pa + n_pp > 0
        z_n = n_pa + ETA_PP * n_pp + EPSILON
        positive_mass = float(pair_stats["positive"])
        negative_mass = -float(pair_stats["negative"])
        h_pos = positive_mass / z_n
        h_neg = negative_mass / z_n
        h_base = float(pair_stats["numerator"]) / z_n
        h_net = h_pos - h_neg
        z_neg_pa = float(pair_stats["PA_detail"]["negative_eta_mass"])
        z_neg_pp = float(pair_stats["PP_detail"]["negative_eta_mass"])
        z_neg = z_neg_pa + z_neg_pp
        f_neg = z_neg / z_n
        h_neg_cond = negative_mass / (z_neg + EPSILON) if z_neg > 0 else 0.0
        h_neg_pa = -float(pair_stats["PA"]["negative"]) / z_n
        h_neg_pp = -float(pair_stats["PP"]["negative"]) / z_n
        eta_mass = float(n_pa + ETA_PP * n_pp)
        audible_pair_mass = float(pair_stats["wdec"])
        negative_audible_mass_pa = float(pair_stats["PA_detail"]["negative_audible_mass"])
        negative_audible_mass_pp = float(pair_stats["PP_detail"]["negative_audible_mass"])
        negative_audible_mass = negative_audible_mass_pa + negative_audible_mass_pp
        audible_square_sum = float(
            pair_stats["PA_detail"]["audible_weight_square_sum"]
            + pair_stats["PP_detail"]["audible_weight_square_sum"]
        )
        d_audible = negative_mass / (audible_pair_mass + EPSILON) if valid else 0.0
        f_neg_audible = negative_audible_mass / (audible_pair_mass + EPSILON) if valid else 0.0
        audible_to_raw = audible_pair_mass / (eta_mass + EPSILON) if valid else 0.0
        effective_audible_pair_count = audible_pair_mass ** 2 / (audible_square_sum + EPSILON) if valid else 0.0
        top_q = [float(pair["negative_contribution_magnitude"]) for pair in top_pairs]
        d_top1 = top_q[0] if top_q else 0.0
        d_top3 = float(np.mean(top_q[:3])) if top_q else 0.0
        d_top5 = float(np.mean(top_q[:5])) if top_q else 0.0
        audible_mass_pa = float(pair_stats["PA"]["wdec"])
        audible_mass_pp = float(pair_stats["PP"]["wdec"])
        d_mass_pa = -float(pair_stats["PA"]["negative"])
        d_mass_pp = -float(pair_stats["PP"]["negative"])
        d_audible_pa = d_mass_pa / (audible_mass_pa + EPSILON) if n_pa else 0.0
        d_audible_pp = d_mass_pp / (audible_mass_pp + EPSILON) if n_pp else 0.0
        cancellation = min(h_pos, h_neg)
        total_component = h_pos + h_neg
        negative_ratio = h_neg / (total_component + EPSILON) if total_component > 0 else 0.0
        onset_error = abs(h_base - h_net)
        max_onset_error = max(max_onset_error, onset_error)
        max_pair_error = max(max_pair_error, float(pair_stats["pair_reconstruction_max_abs_error"]))
        max_q_error = max(max_q_error, float(pair_stats["q_reconstruction_max_abs_error"]))
        minimum_wdec = min(minimum_wdec, float(pair_stats["minimum_W_dec"]))
        strongest = top_pairs[0] if top_pairs else {}
        dominant_origin = (
            "PA" if h_neg_pa > h_neg_pp else "PP" if h_neg_pp > h_neg_pa else "TIE_OR_NONE"
        )
        row = {
            "onset_index": onset_index,
            "onset_tick": int(tick),
            "onset_time": float(seconds),
            "A_n_count": len(current_active),
            "P_n_count": len(residual),
            "PA_pair_count": n_pa,
            "PP_pair_count": n_pp,
            "valid_onset": valid,
            "Z_n": z_n if valid else EPSILON,
            "raw_pair_count": n_pa + n_pp,
            "eta_mass": eta_mass,
            "audible_pair_mass": audible_pair_mass,
            "negative_audible_pair_mass": negative_audible_mass,
            "audible_weight_square_sum": audible_square_sum,
            "audible_to_raw_ratio": audible_to_raw,
            "effective_audible_pair_count": effective_audible_pair_count,
            "positive_contribution_mass": positive_mass,
            "negative_contribution_mass": negative_mass,
            "H_pos_n": h_pos,
            "H_neg_all_n": h_neg,
            "D_paircount_n": h_neg,
            "D_audible_n": d_audible,
            "D_mass_n": negative_mass,
            "D_top1_n": d_top1,
            "D_top3_mean_n": d_top3,
            "D_top5_mean_n": d_top5,
            "H_base_n": h_base,
            "H_net_n": h_net,
            "Z_neg_n": z_neg,
            "F_neg_n": f_neg,
            "F_neg_paircount_n": f_neg,
            "F_neg_audible_n": f_neg_audible,
            "H_neg_cond_n": h_neg_cond,
            "cancellation_mass_n": cancellation,
            "negative_ratio_n": negative_ratio,
            "H_neg_PA_n": h_neg_pa,
            "H_neg_PP_n": h_neg_pp,
            "D_mass_PA_n": d_mass_pa,
            "D_mass_PP_n": d_mass_pp,
            "audible_pair_mass_PA_n": audible_mass_pa,
            "audible_pair_mass_PP_n": audible_mass_pp,
            "negative_audible_mass_PA_n": negative_audible_mass_pa,
            "negative_audible_mass_PP_n": negative_audible_mass_pp,
            "D_audible_PA_n": d_audible_pa,
            "D_audible_PP_n": d_audible_pp,
            "negative_PA_pair_count": int(pair_stats["PA_detail"]["negative_pair_count"]),
            "negative_PP_pair_count": int(pair_stats["PP_detail"]["negative_pair_count"]),
            "top_negative_interval_classes": _top_interval_string(pair_stats),
            "top_negative_pair_contribution_magnitude": float(
                strongest.get("negative_contribution_magnitude", 0.0)
            ),
            "top_negative_pair_origin": strongest.get("pair_type", "NONE"),
            "dominant_negative_origin": dominant_origin,
            "onset_reconstruction_abs_error": onset_error,
            "AA_pair_count": 0,
        }
        onset_rows.append(row)

        if h_neg > 0:
            packet = {"onset_row": row, "top_pairs": top_pairs}
            item = (h_neg, onset_index, packet)
            if len(top_onset_heap) < 20:
                heapq.heappush(top_onset_heap, item)
            elif h_neg > top_onset_heap[0][0]:
                heapq.heapreplace(top_onset_heap, item)

        for pair_type, eta in (("PA", 1.0), ("PP", ETA_PP)):
            stats = pair_stats[pair_type]
            detail = pair_stats[f"{pair_type}_detail"]
            aggregate = pair_aggregate[pair_type]
            aggregate["total_pair_count"] += int(stats["count"])
            aggregate["negative_pair_count"] += int(detail["negative_pair_count"])
            aggregate["eta_weighted_pair_mass"] += eta * float(stats["count"])
            aggregate["negative_eta_mass"] += float(detail["negative_eta_mass"])
            aggregate["negative_numerator_mass"] += -float(stats["negative"])
            aggregate["interval_counts"].update(detail["interval_counts"])
            for interval, mass in detail["interval_negative_masses"].items():
                aggregate["interval_negative_masses"][int(interval)] += float(mass)

    top_packets = [item[2] for item in sorted(top_onset_heap, reverse=True)]
    diagnostics = {
        "distinct_onset_count": len(onset_rows),
        "max_pair_reconstruction_abs_error": max_pair_error,
        "max_q_reconstruction_abs_error": max_q_error,
        "minimum_W_dec": 0.0 if minimum_wdec == np.inf else minimum_wdec,
        "max_onset_reconstruction_abs_error": max_onset_error,
        "pair_aggregate": pair_aggregate,
    }
    return onset_rows, top_packets, diagnostics
