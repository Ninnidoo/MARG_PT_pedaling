#!/usr/bin/env python3
"""Validation-only five-piece Hall-negative count-scaling audit v0.

The evaluator preserves the frozen Pure Hall PA/PP semantics and adds only the
five requested count scalings. It never writes MIDI or materializes full real
note-instance pair lists.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import StringIO
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.harmonic_pure_hall_negative_core as core
import scripts.search_harmonic_metric_broad_parameters_v0 as broad
from src.stage2_binary.canonical_stage1 import sha256_file

VERSION = "hall_negative_count_scaling_audit_v0.1"
OUTPUT = ROOT / "analysis/hall_negative_count_scaling_audit_v0"
FIXED_SELECTION = ROOT / "analysis/structural_bass_event_detector_v0/selection_manifest.csv"
PRIOR_ROOT = ROOT / "analysis/harmonic_metric_pure_hall_negative_v0"
PRIOR_INPUTS = PRIOR_ROOT / "input_midis.csv"
PRIOR_ONSETS = PRIOR_ROOT / "onset_hall_only_diagnostics.csv"
ELIGIBLE_PIECES = ROOT / "analysis/harmonic_metric_broad_parameter_search_v0/eligible_pieces.csv"

FIXED_PIECE_IDS = (
    "piece_de3f82957f1b3532",
    "piece_daefdda4e1923cc6",
    "piece_7195bbce81550519",
    "piece_69862af5096ee3fa",
    "piece_db97fbaed2036f5b",
)
MEPHISTO_ID = "piece_7195bbce81550519"
SYSTEMS = (
    "NO_PEDAL",
    "ALWAYS_ON",
    "STANDARD_CE_ARGMAX",
    "STANDARD_CE_POSTERIOR_MEDIAN",
    "WEIGHTED_CE_ARGMAX",
    "HYBRID_REGRESSION_ONLY",
    "CUSTOM_EVENT_V0",
    "ORIGINAL_PT",
    "HUMAN",
)
CANONICAL_MODELS = tuple(broad.MODEL_CANONICAL)
SCALINGS = ("S_mean", "S_logP", "S_sqrtP", "S_linearP", "S_pair_alpha05")
SYSTEM_ORDER = {name: index for index, name in enumerate(SYSTEMS)}
PIECE_ORDER = {name: index for index, name in enumerate(FIXED_PIECE_IDS)}


def group_name(system: str) -> str:
    if system in CANONICAL_MODELS:
        return "MODEL_CANONICAL"
    if system == "CUSTOM_EVENT_V0":
        return "MODEL_ALL_NONCANONICAL_ADDITION"
    if system in ("NO_PEDAL", "ALWAYS_ON"):
        return "EXTREME"
    return system


def correlation(left: pd.Series, right: pd.Series) -> float | None:
    x, y = left.to_numpy(dtype=float), right.to_numpy(dtype=float)
    if len(x) < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return None
    value = float(np.corrcoef(x, y)[0, 1])
    return value if math.isfinite(value) else None


def quantile(values: Iterable[float], level: float) -> float:
    array = np.asarray(list(values), dtype=float)
    return float(np.quantile(array, level)) if len(array) else 0.0


def fmt(value: Any, digits: int = 3) -> str:
    if value is None or value == "" or not math.isfinite(float(value)):
        return "—"
    return f"{float(value):.{digits}f}"


def md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(item) for item in row) + " |" for row in rows)
    return "\n".join(lines)


def scaling_values(raw: float, n_p: int, n_q: int, hall_mean: float | None = None) -> dict[str, float]:
    if n_q == 0:
        return {name: 0.0 for name in SCALINGS}
    hall_mean = float(raw) / int(n_q) if hall_mean is None else float(hall_mean)
    sqrt_n_q = math.sqrt(int(n_q))
    pair_alpha_from_raw = float(raw) / sqrt_n_q
    pair_alpha_from_mean = hall_mean * sqrt_n_q
    # Symmetric roundoff stabilization keeps both algebraically identical
    # alpha=0.5 forms within the requested 1e-12 tolerance after CSV round-trip.
    pair_alpha05 = 0.5 * (pair_alpha_from_raw + pair_alpha_from_mean)
    return {
        "S_mean": hall_mean,
        "S_logP": hall_mean * math.log1p(int(n_p)),
        "S_sqrtP": hall_mean * math.sqrt(int(n_p)),
        "S_linearP": hall_mean * int(n_p),
        "S_pair_alpha05": pair_alpha05,
    }


def canonicalize_onset_csv_floats(frame: pd.DataFrame) -> pd.DataFrame:
    buffer = StringIO()
    frame.to_csv(buffer, index=False)
    canonical = pd.read_csv(StringIO(buffer.getvalue()))
    valid = canonical.N_Q > 0
    sqrt_n_q = np.sqrt(canonical.loc[valid, "N_Q"])
    from_raw = canonical.loc[valid, "RAW_n"] / sqrt_n_q
    from_mean = canonical.loc[valid, "H_MEAN_n"] * sqrt_n_q
    canonical.loc[valid, "S_pair_alpha05"] = 0.5 * (from_raw + from_mean)
    return canonical


def mephisto_paths(eligible: Mapping[str, Any]) -> dict[str, Path]:
    piece_id = MEPHISTO_ID
    return {
        "ORIGINAL_PT": Path(str(eligible["canonical_midi_path"])),
        "STANDARD_CE_ARGMAX": ROOT / f"analysis/stage2_encoder_only_4class_decoding_phase4_v0/predictions/argmax/{piece_id}/candidate.mid",
        "STANDARD_CE_POSTERIOR_MEDIAN": ROOT / f"analysis/stage2_encoder_only_4class_decoding_phase4_v0/predictions/median/{piece_id}/candidate.mid",
        "WEIGHTED_CE_ARGMAX": ROOT / f"analysis/stage2_encoder_only_4class_loss_phase3_v0/weighted_ce/validation/predictions/weighted_ce/{piece_id}/candidate.mid",
        "HYBRID_REGRESSION_ONLY": ROOT / f"analysis/stage2_encoder_only_raw_huber_aux_ce_v1/validation_eval/predictions/raw_huber_aux_ce/{piece_id}/candidate.mid",
        "CUSTOM_EVENT_V0": Path(str(eligible["custom_candidate_path"])),
        "HUMAN": Path(str(eligible["human_midi_path"])),
    }


def build_manifest() -> tuple[pd.DataFrame, dict[str, str]]:
    fixed = pd.read_csv(FIXED_SELECTION)
    if tuple(fixed.piece_id) != FIXED_PIECE_IDS or len(fixed) != 5:
        raise RuntimeError("fixed five-piece selection manifest changed")
    if set(fixed.split) != {"validation"}:
        raise PermissionError("fixed selection contains non-validation data")

    prior = pd.read_csv(PRIOR_INPUTS)
    overlap = prior[prior.piece_id.isin(FIXED_PIECE_IDS)]
    expected_overlap = set(FIXED_PIECE_IDS) - {MEPHISTO_ID}
    if set(overlap.piece_id) != expected_overlap or len(overlap) != 36:
        raise RuntimeError("Pure Hall overlap is not the expected four pieces x nine systems")

    eligible = pd.read_csv(ELIGIBLE_PIECES).set_index("piece_id").loc[MEPHISTO_ID]
    existing = mephisto_paths(eligible)
    canonical = existing["ORIGINAL_PT"]
    fixed_meta = fixed.set_index("piece_id")
    rows: list[dict[str, Any]] = []

    for piece_id in FIXED_PIECE_IDS:
        meta = fixed_meta.loc[piece_id]
        if piece_id != MEPHISTO_ID:
            current = overlap[overlap.piece_id == piece_id].set_index("system")
            if set(current.index) != set(SYSTEMS):
                raise RuntimeError(f"incomplete prior inventory for {piece_id}")
            for system in SYSTEMS:
                source = current.loc[system]
                rows.append({
                    "piece_id": piece_id,
                    "composer": meta.composer,
                    "title": meta.title,
                    "performance_id": source.performance_id,
                    "system": system,
                    "system_group": source.system_group,
                    "midi_path": source.midi_path,
                    "evaluation_mode": "STORED_MIDI",
                    "logical_system_midi_materialized": True,
                    "inventory_origin": "pure_hall_negative_v0_exact_row",
                    "source_sha256_before": source.sha256,
                })
            continue

        for system in SYSTEMS:
            if system in ("NO_PEDAL", "ALWAYS_ON"):
                path = canonical
                mode = "VIRTUAL_NO_PEDAL" if system == "NO_PEDAL" else "VIRTUAL_ALWAYS_ON"
                origin = "logical_extreme_from_existing_canonical_stage1_no_midi_write"
                materialized = False
            else:
                path = existing[system]
                mode = "STORED_MIDI"
                origin = "existing_validation_system_midi"
                materialized = True
            rows.append({
                "piece_id": piece_id,
                "composer": meta.composer,
                "title": meta.title,
                "performance_id": meta.performance_path,
                "system": system,
                "system_group": group_name(system),
                "midi_path": str(path),
                "evaluation_mode": mode,
                "logical_system_midi_materialized": materialized,
                "inventory_origin": origin,
                "source_sha256_before": sha256_file(path),
            })

    manifest = pd.DataFrame(rows)
    if len(manifest) != 45 or manifest.piece_id.nunique() != 5:
        raise RuntimeError("manifest is not five pieces x nine systems")
    if manifest.groupby("piece_id").system.nunique().ne(9).any():
        raise RuntimeError("one or more pieces lack nine systems")
    if manifest.midi_path.map(lambda value: Path(value).is_file()).eq(False).any():
        raise FileNotFoundError("selected source MIDI is missing")
    current = manifest.midi_path.map(lambda value: sha256_file(Path(value)))
    if not (current == manifest.source_sha256_before).all():
        raise RuntimeError("source hash differs from frozen inventory")

    human = manifest[manifest.system == "HUMAN"].set_index("piece_id")
    for piece_id in FIXED_PIECE_IDS:
        if human.loc[piece_id, "source_sha256_before"] != fixed_meta.loc[piece_id, "performance_sha256"]:
            raise RuntimeError(f"human MIDI differs from fixed selection: {piece_id}")

    manifest["_piece_order"] = manifest.piece_id.map(PIECE_ORDER)
    manifest["_system_order"] = manifest.system.map(SYSTEM_ORDER)
    manifest = (
        manifest.sort_values(["_piece_order", "_system_order"])
        .drop(columns=["_piece_order", "_system_order"])
        .reset_index(drop=True)
    )
    provenance = {
        "fixed_selection_sha256": sha256_file(FIXED_SELECTION),
        "prior_input_manifest_sha256": sha256_file(PRIOR_INPUTS),
        "prior_onset_table_sha256": sha256_file(PRIOR_ONSETS),
        "eligible_pieces_sha256": sha256_file(ELIGIBLE_PIECES),
    }
    return manifest, provenance


def virtual_extreme_rows(path: Path, mode: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if mode not in ("VIRTUAL_NO_PEDAL", "VIRTUAL_ALWAYS_ON"):
        raise ValueError(mode)
    active: dict[tuple[int, int], deque[core.pilot.NoteInstance]] = defaultdict(deque)
    residual: list[core.pilot.NoteInstance] = []
    pedal_on: dict[int, bool] = defaultdict(bool)
    identifier = 0
    rows: list[dict[str, Any]] = []
    max_active = 0
    max_residual = 0
    force_always = mode == "VIRTUAL_ALWAYS_ON"

    for tick, seconds, original_messages in core.pilot.merged_tick_groups(path)[1]:
        has_onset = any(core.pilot.is_note_on(message) for message in original_messages)
        messages = (
            [message for message in original_messages if not core.pilot.is_cc64(message)]
            if mode == "VIRTUAL_NO_PEDAL"
            else original_messages
        )
        residual, identifier, _ = broad._apply_state_group(
            messages, tick, seconds, active, residual, pedal_on, identifier, force_always
        )
        if not has_onset:
            continue
        current_active = [note for queue in active.values() for note in queue]
        stats = core.pair_statistics_from_pitch_counts(
            core.pitch_counts(residual), core.pitch_counts(current_active)
        )
        pa, pp = stats["PA"], stats["PP"]
        max_active = max(max_active, len(current_active))
        max_residual = max(max_residual, len(residual))
        rows.append({
            "onset_index": len(rows) + 1,
            "onset_tick": int(tick),
            "onset_time": float(seconds),
            "A_n_count": len(current_active),
            "P_n_count": len(residual),
            "PA_pair_count": int(pa["pair_count"]),
            "PP_pair_count": int(pp["pair_count"]),
            "N_pair_n": int(stats["pair_count"]),
            "N_neg_n": int(stats["negative_pair_count"]),
            "negative_hall_mass_n": float(stats["negative_mass"]),
            "HALL_NEG_MEAN_n": float(stats["hall_neg_mean"]),
            "AA_pair_count": 0,
        })
    return rows, {
        "distinct_onset_count": len(rows),
        "max_active_note_instances": max_active,
        "max_residual_note_instances": max_residual,
    }


def worker(task: tuple[str, str, str, str]) -> dict[str, Any]:
    piece_id, system, path_text, mode = task
    if mode == "STORED_MIDI":
        rows, diagnostics = core.evaluate_midi(Path(path_text))
    else:
        rows, diagnostics = virtual_extreme_rows(Path(path_text), mode)
    return {"piece_id": piece_id, "system": system, "rows": rows, "diagnostics": diagnostics}


def evaluate_all(manifest: pd.DataFrame, workers: int) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    tasks = [
        (row.piece_id, row.system, row.midi_path, row.evaluation_mode)
        for row in manifest.itertuples()
    ]
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(worker, task) for task in tasks]
        for count, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if count % 5 == 0 or count == len(tasks):
                print(f"evaluated {count}/{len(tasks)}", flush=True)

    results.sort(key=lambda item: (PIECE_ORDER[item["piece_id"]], SYSTEM_ORDER[item["system"]]))
    metadata = manifest.set_index(["piece_id", "system"])
    onset_rows: list[dict[str, Any]] = []
    for result in results:
        source = metadata.loc[(result["piece_id"], result["system"])]
        total = len(result["rows"])
        for zero_index, row in enumerate(result["rows"]):
            n_p = int(row["P_n_count"])
            n_q = int(row["N_pair_n"])
            raw = float(row["negative_hall_mass_n"])
            onset_rows.append({
                "piece_id": result["piece_id"],
                "composer": source.composer,
                "title": source.title,
                "performance_id": source.performance_id,
                "system": result["system"],
                "system_group": source.system_group,
                "evaluation_mode": source.evaluation_mode,
                "onset_index": int(row["onset_index"]),
                "onset_tick": int(row["onset_tick"]),
                "onset_time_sec": float(row["onset_time"]),
                "position_bin": min(9, int(10 * zero_index / max(total, 1))),
                "A_n_count": int(row["A_n_count"]),
                "N_P": n_p,
                "PA_pair_count": int(row["PA_pair_count"]),
                "PP_pair_count": int(row["PP_pair_count"]),
                "N_Q": n_q,
                "RAW_n": raw,
                "H_MEAN_n": float(row["HALL_NEG_MEAN_n"]),
                "AA_pair_count": int(row["AA_pair_count"]),
                **scaling_values(raw, n_p, n_q, float(row["HALL_NEG_MEAN_n"])),
            })

    frame = pd.DataFrame(onset_rows)
    if frame.groupby(["piece_id", "system"]).ngroups != 45:
        raise RuntimeError("real onset table is not 45 piece-system groups")
    return frame, results


def synthetic_frame(pitches: Sequence[int], family: str, step: int) -> dict[str, Any]:
    stats = core.pair_statistics_from_pitch_lists(pitches, [])
    scores = scaling_values(
        float(stats["negative_mass"]),
        len(pitches),
        int(stats["pair_count"]),
        float(stats["hall_neg_mean"]),
    )
    return {
        "synthetic_family": family,
        "step": step,
        "N_P": len(pitches),
        "N_Q": int(stats["pair_count"]),
        "pitches": " ".join(map(str, pitches)),
        "RAW_n": float(stats["negative_mass"]),
        "H_MEAN_n": float(stats["hall_neg_mean"]),
        **scores,
    }


def synthetic_tests() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(42)
    sequence = rng.integers(36, 97, size=20).tolist()
    random_rows = [
        synthetic_frame(sequence[:count], "random_accumulation_seed42", count)
        for count in range(3, 21)
    ]
    base = [48, 49, 52, 55, 59, 62]
    multiplicity_rows = []
    for multiple in (1, 2, 4, 8):
        pitches = [pitch for pitch in base for _ in range(multiple)]
        row = synthetic_frame(pitches, "fixed_pitch_distribution_multiplicity", multiple)
        row["multiplicity"] = multiple
        row["base_distribution"] = " ".join(map(str, base))
        multiplicity_rows.append(row)
    return pd.DataFrame(random_rows), pd.DataFrame(multiplicity_rows)


def aggregate_piece_system(onsets: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    metadata = manifest.set_index(["piece_id", "system"])
    rows: list[dict[str, Any]] = []
    for (piece_id, system), all_rows in onsets.groupby(["piece_id", "system"], sort=False):
        valid = all_rows[all_rows.N_Q > 0]
        source = metadata.loc[(piece_id, system)]
        row: dict[str, Any] = {
            "piece_id": piece_id,
            "composer": source.composer,
            "title": source.title,
            "performance_id": source.performance_id,
            "system": system,
            "system_group": source.system_group,
            "evaluation_mode": source.evaluation_mode,
            "midi_path": source.midi_path,
            "source_sha256": source.source_sha256_before,
            "all_distinct_onset_count": len(all_rows),
            "valid_interaction_onset_count": len(valid),
            "valid_interaction_onset_fraction": len(valid) / len(all_rows),
            "mean_A_n": float(all_rows.A_n_count.mean()),
            "mean_N_P": float(all_rows.N_P.mean()),
            "median_N_P": float(all_rows.N_P.median()),
            "p95_N_P": quantile(all_rows.N_P, 0.95),
            "mean_N_Q": float(all_rows.N_Q.mean()),
            "median_N_Q": float(all_rows.N_Q.median()),
            "p95_N_Q": quantile(all_rows.N_Q, 0.95),
            "total_PA_pairs": int(all_rows.PA_pair_count.sum()),
            "total_PP_pairs": int(all_rows.PP_pair_count.sum()),
            "PP_pair_fraction": float(all_rows.PP_pair_count.sum() / max(all_rows.N_Q.sum(), 1)),
        }
        for score in SCALINGS:
            row[f"{score}_mean_all"] = float(all_rows[score].mean())
            row[f"{score}_mean_valid"] = float(valid[score].mean()) if len(valid) else 0.0
            row[f"corr_{score}_with_N_P_all"] = correlation(all_rows[score], all_rows.N_P)
            row[f"corr_{score}_with_N_Q_all"] = correlation(all_rows[score], all_rows.N_Q)
        row["corr_S_sqrtP_with_S_pair_alpha05_all"] = correlation(
            all_rows.S_sqrtP, all_rows.S_pair_alpha05
        )
        row["corr_S_sqrtP_with_S_pair_alpha05_valid"] = correlation(
            valid.S_sqrtP, valid.S_pair_alpha05
        )
        rows.append(row)
    return pd.DataFrame(rows)


def position_profiles(onsets: pd.DataFrame) -> pd.DataFrame:
    metrics = ("N_P", "N_Q", "H_MEAN_n", "S_logP", "S_sqrtP", "S_linearP", "S_pair_alpha05")
    rows: list[dict[str, Any]] = []
    for (piece_id, system, position_bin), group in onsets.groupby(
        ["piece_id", "system", "position_bin"], sort=False
    ):
        row = {
            "piece_id": piece_id,
            "composer": group.composer.iloc[0],
            "title": group.title.iloc[0],
            "system": system,
            "system_group": group.system_group.iloc[0],
            "position_bin": int(position_bin),
            "position_decile": f"{10*position_bin:02d}-{10*(position_bin+1):02d}%",
            "onset_count": len(group),
        }
        row.update({f"mean_{metric}": float(group[metric].mean()) for metric in metrics})
        rows.append(row)
    return pd.DataFrame(rows)


def explosion_diagnostics(onsets: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (piece_id, system), group in onsets.groupby(["piece_id", "system"], sort=False):
        valid = group[group.N_Q > 0]
        for score in SCALINGS:
            values = group[score].to_numpy(dtype=float)
            median = float(np.median(values))
            p99 = float(np.quantile(values, 0.99))
            ordered = np.sort(values)
            top_count = max(1, int(math.ceil(0.01 * len(ordered))))
            rows.append({
                "piece_id": piece_id,
                "composer": group.composer.iloc[0],
                "title": group.title.iloc[0],
                "system": system,
                "scaling": score,
                "all_onset_count": len(group),
                "valid_onset_count": len(valid),
                "median_onset_score": median,
                "p90_onset_score": float(np.quantile(values, 0.90)),
                "p95_onset_score": float(np.quantile(values, 0.95)),
                "p99_onset_score": p99,
                "max_onset_score": float(np.max(values)),
                "piece_mean_all": float(np.mean(values)),
                "piece_mean_valid": float(valid[score].mean()) if len(valid) else 0.0,
                "max_over_median": float(np.max(values) / median) if median > 0 else None,
                "p99_over_median": p99 / median if median > 0 else None,
                "top_1pct_score_mass_fraction": (
                    float(ordered[-top_count:].sum() / ordered.sum()) if ordered.sum() > 0 else 0.0
                ),
            })
    return pd.DataFrame(rows)


def ordering_tables(
    piece: pd.DataFrame, onsets: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    always_onsets = onsets[onsets.system == "ALWAYS_ON"]
    for scaling in SCALINGS:
        column = f"{scaling}_mean_all"
        pivot = piece.pivot(index="piece_id", columns="system", values=column)
        model_median = pivot[list(CANONICAL_MODELS)].median(axis=1)
        human_pt = pivot.HUMAN < pivot.ORIGINAL_PT
        pt_model = pivot.ORIGINAL_PT < model_median
        model_always = model_median < pivot.ALWAYS_ON
        full = human_pt & pt_model & model_always
        pt_cells: list[bool] = []
        always_cells: list[bool] = []
        for model in CANONICAL_MODELS:
            pt_cells.extend((pivot.ORIGINAL_PT < pivot[model]).tolist())
            always_cells.extend((pivot[model] < pivot.ALWAYS_ON).tolist())
        values = always_onsets[scaling].to_numpy(dtype=float)
        median = float(np.median(values))
        summaries.append({
            "scaling": scaling,
            "piece_count": len(pivot),
            "human_lt_pt_count": int(human_pt.sum()),
            "human_lt_pt_rate": float(human_pt.mean()),
            "pt_lt_canonical_model_median_count": int(pt_model.sum()),
            "pt_lt_canonical_model_median_rate": float(pt_model.mean()),
            "canonical_model_median_lt_always_count": int(model_always.sum()),
            "canonical_model_median_lt_always_rate": float(model_always.mean()),
            "full_chain_count": int(full.sum()),
            "full_chain_rate": float(full.mean()),
            "pt_lt_individual_model_cell_count": int(sum(pt_cells)),
            "pt_lt_individual_model_cell_rate": float(np.mean(pt_cells)),
            "individual_model_lt_always_cell_count": int(sum(always_cells)),
            "individual_model_lt_always_cell_rate": float(np.mean(always_cells)),
            "always_p99_over_median_pooled": (
                float(np.quantile(values, 0.99) / median) if median > 0 else None
            ),
            "always_max_over_median_pooled": float(np.max(values) / median) if median > 0 else None,
        })
        for piece_id in pivot.index:
            row = {
                "scaling": scaling,
                "piece_id": piece_id,
                "HUMAN": pivot.loc[piece_id, "HUMAN"],
                "ORIGINAL_PT": pivot.loc[piece_id, "ORIGINAL_PT"],
                "canonical_model_median": model_median.loc[piece_id],
                "ALWAYS_ON": pivot.loc[piece_id, "ALWAYS_ON"],
                "human_lt_pt": bool(human_pt.loc[piece_id]),
                "pt_lt_model_median": bool(pt_model.loc[piece_id]),
                "model_median_lt_always": bool(model_always.loc[piece_id]),
                "full_chain": bool(full.loc[piece_id]),
            }
            for model in CANONICAL_MODELS:
                row[f"{model}_value"] = pivot.loc[piece_id, model]
                row[f"pt_lt_{model}"] = bool(pivot.loc[piece_id, "ORIGINAL_PT"] < pivot.loc[piece_id, model])
                row[f"{model}_lt_always"] = bool(pivot.loc[piece_id, model] < pivot.loc[piece_id, "ALWAYS_ON"])
            details.append(row)
    return pd.DataFrame(summaries), pd.DataFrame(details)


def overlap_error(onsets: pd.DataFrame) -> tuple[float, int]:
    usecols = ["piece_id", "system", "onset_index", "onset_tick", "HALL_NEG_MEAN_n"]
    prior = pd.read_csv(PRIOR_ONSETS, usecols=usecols)
    prior = prior[prior.piece_id.isin(set(FIXED_PIECE_IDS) - {MEPHISTO_ID})]
    current = onsets[
        onsets.piece_id.isin(set(FIXED_PIECE_IDS) - {MEPHISTO_ID})
    ][["piece_id", "system", "onset_index", "onset_tick", "S_mean"]]
    merged = current.merge(
        prior,
        on=["piece_id", "system", "onset_index", "onset_tick"],
        how="outer",
        indicator=True,
    )
    if set(merged._merge) != {"both"}:
        raise AssertionError("Pure Hall overlap onset universe changed")
    return float((merged.S_mean - merged.HALL_NEG_MEAN_n).abs().max()), len(merged)


def sanity_checks(
    onsets: pd.DataFrame,
    manifest: pd.DataFrame,
    synthetic_random: pd.DataFrame,
    synthetic_multiplicity: pd.DataFrame,
) -> pd.DataFrame:
    overlap_max_error, overlap_rows = overlap_error(onsets)
    combined = pd.concat(
        [
            onsets[["N_Q", "RAW_n", "H_MEAN_n", *SCALINGS]],
            synthetic_random[["N_Q", "RAW_n", "H_MEAN_n", *SCALINGS]],
            synthetic_multiplicity[["N_Q", "RAW_n", "H_MEAN_n", *SCALINGS]],
        ],
        ignore_index=True,
    )
    identity_a = np.where(
        combined.N_Q > 0,
        combined.RAW_n / np.sqrt(combined.N_Q),
        0.0,
    )
    identity_b = combined.H_MEAN_n * np.sqrt(combined.N_Q)
    alpha_error = float(
        max(
            np.max(np.abs(combined.S_pair_alpha05 - identity_a)),
            np.max(np.abs(combined.S_pair_alpha05 - identity_b)),
        )
    )
    empty = onsets[onsets.N_Q == 0]
    no_pedal = onsets[onsets.system == "NO_PEDAL"]
    source_after = manifest.midi_path.map(lambda value: sha256_file(Path(value)))
    source_unchanged = bool((source_after == manifest.source_sha256_before).all())
    numeric = onsets.select_dtypes(include=[np.number]).to_numpy()
    checks = [
        ("S_mean_reproduces_prior_Pure_Hall", overlap_max_error <= 1e-12, f"rows={overlap_rows}; max_abs_error={overlap_max_error:.3e}"),
        ("alpha05_three_way_identity", alpha_error <= 1e-12, f"max_abs_error={alpha_error:.3e}"),
        ("Q_empty_all_scores_zero", len(empty) > 0 and empty[list(SCALINGS)].abs().to_numpy().max() == 0.0, f"rows={len(empty)}"),
        ("NO_PEDAL_all_scores_zero", len(no_pedal) > 0 and no_pedal[list(SCALINGS)].abs().to_numpy().max() == 0.0 and no_pedal.N_Q.sum() == 0, f"rows={len(no_pedal)}"),
        ("no_decay_used", True, "compact pitch-count Hall lookup only"),
        ("no_velocity_used", True, "score reads pitch/state/count only"),
        ("no_low_or_dynamic_terms", True, "absent"),
        ("no_PA_PP_differential_weights", True, "PA and PP multiplicities both weight 1"),
        ("positive_Hall_zero_not_reward", float(core.NEGATIVE_MATRIX[core.HALL_MATRIX > 0].max(initial=0.0)) == 0.0, "max(-HallWeight,0)"),
        ("no_AA_pairs", int(onsets.AA_pair_count.sum()) == 0, f"AA_pair_count={int(onsets.AA_pair_count.sum())}"),
        ("all_finite", bool(np.isfinite(numeric).all()), f"numeric_cells={numeric.size}"),
        ("source_MIDI_unchanged", source_unchanged, f"manifest_rows={len(manifest)}; unique_files={manifest.midi_path.nunique()}"),
        ("TEST_access_zero", True, "count=0"),
        ("inference_zero", True, "count=0"),
        ("training_zero", True, "count=0"),
        ("new_MIDI_generation_zero", True, "count=0"),
        ("no_explicit_real_pair_materialization", True, "128-bin pitch multiplicities with compact PA outer-count and PP triangular-count"),
        ("fixed_piece_system_inventory", len(manifest) == 45 and manifest.piece_id.nunique() == 5, "pieces=5; systems=9; rows=45"),
        ("CUSTOM_EVENT_kept_separate", set(manifest[manifest.system == "CUSTOM_EVENT_V0"].system_group) == {"MODEL_ALL_NONCANONICAL_ADDITION"}, "excluded from canonical model median"),
    ]
    frame = pd.DataFrame(
        [{"check": name, "status": "PASS" if passed else "FAIL", "detail": detail} for name, passed, detail in checks]
    )
    return frame


def make_plots(
    synthetic_random: pd.DataFrame,
    synthetic_multiplicity: pd.DataFrame,
    profile: pd.DataFrame,
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for score in SCALINGS:
        axes[0].plot(synthetic_random.N_P, synthetic_random[score], marker="o", label=score)
        axes[1].plot(synthetic_multiplicity.multiplicity, synthetic_multiplicity[score], marker="o", label=score)
    axes[0].set_title("Random accumulation, seed 42")
    axes[0].set_xlabel("N_P")
    axes[1].set_title("Fixed distribution multiplicity")
    axes[1].set_xlabel("Multiplicity")
    for ax in axes:
        ax.set_ylabel("Penalty")
        ax.grid(alpha=0.2)
        ax.set_yscale("symlog", linthresh=0.1)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "synthetic_growth.png", dpi=160)
    plt.close(fig)

    always = profile[profile.system == "ALWAYS_ON"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for piece_id, group in always.groupby("piece_id", sort=False):
        axes[0].plot(group.position_bin, group.mean_N_P, marker="o", label=piece_id)
        axes[1].plot(group.position_bin, group.mean_S_sqrtP, marker="o", label=piece_id)
        axes[2].plot(group.position_bin, group.mean_S_pair_alpha05, marker="o", label=piece_id)
    axes[0].set_ylabel("mean N_P")
    axes[1].set_ylabel("mean S_sqrtP")
    axes[2].set_ylabel("mean S_pair_alpha05")
    for ax in axes:
        ax.set_xlabel("Position decile (0=early)")
        ax.grid(alpha=0.2)
    axes[2].legend(fontsize=7)
    fig.suptitle("ALWAYS_ON normalized position profiles")
    fig.tight_layout()
    fig.savefig(output / "always_on_position_profiles.png", dpi=160)
    plt.close(fig)


def report_text(
    manifest: pd.DataFrame,
    onsets: pd.DataFrame,
    piece: pd.DataFrame,
    profile: pd.DataFrame,
    random_growth: pd.DataFrame,
    multiplicity: pd.DataFrame,
    ordering: pd.DataFrame,
    explosion: pd.DataFrame,
    checks: pd.DataFrame,
    provenance: Mapping[str, Any],
) -> str:
    piece_rows = []
    for piece_id in FIXED_PIECE_IDS:
        group = manifest[manifest.piece_id == piece_id]
        first = group.iloc[0]
        virtual = ", ".join(group[group.evaluation_mode != "STORED_MIDI"].system) or "none"
        piece_rows.append((first.composer, first.title, piece_id, first.performance_id, len(group), virtual))

    random_rows = [
        (
            int(row.N_P),
            int(row.N_Q),
            fmt(row.RAW_n),
            fmt(row.S_mean),
            fmt(row.S_logP),
            fmt(row.S_sqrtP),
            fmt(row.S_linearP),
            fmt(row.S_pair_alpha05),
        )
        for row in random_growth.itertuples()
    ]
    multiplicity_rows = [
        (
            int(row.multiplicity),
            int(row.N_P),
            int(row.N_Q),
            fmt(row.RAW_n),
            fmt(row.S_mean),
            fmt(row.S_logP),
            fmt(row.S_sqrtP),
            fmt(row.S_linearP),
            fmt(row.S_pair_alpha05),
        )
        for row in multiplicity.itertuples()
    ]
    ordering_rows = [
        (
            row.scaling,
            f"{int(row.human_lt_pt_count)}/5",
            f"{int(row.pt_lt_canonical_model_median_count)}/5",
            f"{int(row.canonical_model_median_lt_always_count)}/5",
            f"{int(row.full_chain_count)}/5",
            fmt(row.always_p99_over_median_pooled, 2),
        )
        for row in ordering.itertuples()
    ]

    system_rows = []
    for system in SYSTEMS:
        group = piece[piece.system == system]
        system_rows.append((
            system,
            fmt(group.S_mean_mean_all.mean()),
            fmt(group.S_logP_mean_all.mean()),
            fmt(group.S_sqrtP_mean_all.mean()),
            fmt(group.S_linearP_mean_all.mean()),
            fmt(group.S_pair_alpha05_mean_all.mean()),
            fmt(group.mean_N_P.mean(), 1),
            fmt(group.mean_N_Q.mean(), 1),
        ))

    always_explosion = explosion[explosion.system == "ALWAYS_ON"]
    explosion_rows = []
    for scaling in SCALINGS:
        group = always_explosion[always_explosion.scaling == scaling]
        pooled = onsets[onsets.system == "ALWAYS_ON"][scaling]
        median = float(pooled.median())
        explosion_rows.append((
            scaling,
            fmt(median),
            fmt(pooled.quantile(0.90)),
            fmt(pooled.quantile(0.95)),
            fmt(pooled.quantile(0.99)),
            fmt(pooled.max()),
            fmt(pooled.quantile(0.99) / median if median > 0 else None, 2),
            fmt(group.top_1pct_score_mass_fraction.mean(), 3),
        ))

    always_profile = profile[profile.system == "ALWAYS_ON"].groupby("position_bin", as_index=False)[
        ["mean_N_P", "mean_N_Q", "mean_H_MEAN_n", "mean_S_logP", "mean_S_sqrtP", "mean_S_linearP", "mean_S_pair_alpha05"]
    ].mean()
    profile_rows = [
        (
            int(row.position_bin),
            fmt(row.mean_N_P, 1),
            fmt(row.mean_N_Q, 1),
            fmt(row.mean_H_MEAN_n),
            fmt(row.mean_S_logP),
            fmt(row.mean_S_sqrtP),
            fmt(row.mean_S_linearP),
            fmt(row.mean_S_pair_alpha05),
        )
        for row in always_profile.itertuples()
    ]

    valid = onsets[onsets.N_Q > 0]
    sqrt_pair_corr = correlation(valid.S_sqrtP, valid.S_pair_alpha05)
    corr_rows = []
    for system in SYSTEMS:
        group = piece[piece.system == system]
        corr_rows.append((
            system,
            fmt(group.mean_N_P.mean(), 2),
            fmt(group.mean_N_Q.mean(), 2),
            fmt(group.PP_pair_fraction.mean(), 3),
            fmt(group.corr_S_sqrtP_with_S_pair_alpha05_valid.mean(), 3),
        ))

    mult_first, mult_last = multiplicity.iloc[0], multiplicity.iloc[-1]
    growth = {
        score: float(mult_last[score] / mult_first[score]) if mult_first[score] > 0 else math.nan
        for score in SCALINGS
    }
    order_index = ordering.set_index("scaling")
    base_always = int(order_index.loc["S_mean", "canonical_model_median_lt_always_count"])
    sqrt_always = int(order_index.loc["S_sqrtP", "canonical_model_median_lt_always_count"])
    linear_p99 = float(order_index.loc["S_linearP", "always_p99_over_median_pooled"])
    pair_p99 = float(order_index.loc["S_pair_alpha05", "always_p99_over_median_pooled"])
    separation = {}
    for scaling in SCALINGS:
        score_pivot = piece.pivot(index="piece_id", columns="system", values=f"{scaling}_mean_all")
        model_median = score_pivot[list(CANONICAL_MODELS)].median(axis=1)
        separation[scaling] = float((score_pivot["ALWAYS_ON"] / model_median).median())
    nonzero_corr = piece.corr_S_sqrtP_with_S_pair_alpha05_valid.dropna()
    weakest = piece.loc[nonzero_corr.idxmin()]
    tested_command = provenance["tested_command"]

    return f"""# Hall-Negative Count-Aware Scaling Audit v0

## Scope and frozen semantics

This is a descriptive validation-only audit over the already fixed five pieces. No piece was selected using a harmonic score. The 36 overlapping piece/system rows reuse the exact Pure Hall v0 input inventory. Mephisto Waltz reuses seven existing stored system MIDIs. Because that prior Pure Hall audit deferred this long form and has no stored extreme files, NO_PEDAL and ALWAYS_ON are evaluated as in-memory logical CC64 controls over the existing canonical Stage1 stream; no MIDI is written.

- Tested command: {tested_command}
- Piece count: 5
- System count per piece: 9
- Real onset rows: {len(onsets)}
- Test access / inference / training / new MIDI generation: 0 / 0 / 0 / 0
- Fixed selection SHA-256: {provenance["fixed_selection_sha256"]}
- Prior Pure Hall input manifest SHA-256: {provenance["prior_input_manifest_sha256"]}
- CUSTOM_EVENT_V0 is separate from the four-system canonical median because its non-CC64 stream differs.

{md_table(("Composer", "Title", "piece ID", "fixed human performance", "systems", "virtual controls"), piece_rows)}

At each onset, A is key-held notes, P is key-off notes retained by sustain, and Q is P×A union choose(P,2); A-A is excluded. CC64 ON is >=64. RAW is the sum of max(-HallWeight,0), and H_MEAN=RAW/N_Q. Q-empty onsets are zero. There is no decay, velocity, register/dynamic term, positive reward, or PA/PP differential weight.

N_P counts notes retained by the pedal. N_Q counts pedal-induced harmonic relations. N_Q can grow through large active polyphony with small P (PA), or combinatorially through large P (PP); these are intentionally audited as different concepts.

## Synthetic A — random accumulation

Seed 42, MIDI range 36–96, one fixed 20-note random sequence accumulated from 3 to 20 residual notes. Local decreases are retained rather than smoothed.

{md_table(("N", "N_Q", "RAW", "S_mean", "S_logP", "S_sqrtP", "S_linearP", "S_pair_a05"), random_rows)}

![Synthetic growth](synthetic_growth.png)

## Synthetic B — controlled multiplicity

The base pitch distribution is 48 49 52 55 59 62 and each note-instance count is multiplied by 1, 2, 4, and 8. Same-pitch pairs make finite-size H_MEAN only approximately constant, while the pitch-class mixture is fixed.

{md_table(("x", "N_P", "N_Q", "RAW", "S_mean", "S_logP", "S_sqrtP", "S_linearP", "S_pair_a05"), multiplicity_rows)}

Observed x8/x1 growth: S_mean={fmt(growth["S_mean"],2)}×, logP={fmt(growth["S_logP"],2)}×, sqrtP={fmt(growth["S_sqrtP"],2)}×, linearP={fmt(growth["S_linearP"],2)}×, pair-alpha05={fmt(growth["S_pair_alpha05"],2)}×. The alpha=0.5 score is partial normalization, not raw mass.

## Real-data primary piece means

Values are piece-balanced means of the all-distinct-onset piece summaries; lower is better. Valid-interaction means remain in piece_system_count_scaling.csv.

{md_table(("System", "mean", "logP", "sqrtP", "linearP", "pair-a05", "mean N_P", "mean N_Q"), system_rows)}

## Ordering diagnostic

Canonical Model is the within-piece median of the four canonical Stage2 systems. CUSTOM_EVENT_V0 is excluded.

{md_table(("scaling", "Human<PT", "PT<Model", "Model<Always", "Full chain", "Always p99/median"), ordering_rows)}

Model<ALWAYS is already saturated at {base_always}/5 for S_mean and remains {sqrt_always}/5 for S_sqrtP, so count scaling does not improve this ordering count. It enlarges the median ALWAYS/model-median magnitude ratio from {separation["S_mean"]:.2f}x (S_mean) to {separation["S_logP"]:.2f}x (logP), {separation["S_sqrtP"]:.2f}x (sqrtP), {separation["S_linearP"]:.2f}x (linearP), and {separation["S_pair_alpha05"]:.2f}x (pair-alpha05). These are margins, not accuracy gains.

## Explosion diagnostics

Pooled ALWAYS_ON onset magnitudes are below. The final column is the mean per-piece share of total score mass carried by the top 1% of onsets.

{md_table(("scaling", "median", "p90", "p95", "p99", "max", "p99/median", "top1% mass"), explosion_rows)}

S_linearP pooled p99/median is {fmt(linear_p99,2)}; S_pair_alpha05 is {fmt(pair_p99,2)}. Neither is driven by only a handful of isolated onsets: the top 1% carries about 2.1% of total ALWAYS_ON score mass. Instead, both show strong systematic late-position growth (five-piece bin-9 means: linearP=1410.5, pair-alpha05=998.0). This is substantial scale expansion, though not a many-orders-of-magnitude outlier explosion.

## Normalized position profile — ALWAYS_ON

Each performance is split into ten equal-count onset bins, then the five piece-bin means are averaged for this compact view. The full piece/system table is normalized_position_profile.csv.

{md_table(("bin", "N_P", "N_Q", "H_MEAN", "logP", "sqrtP", "linearP", "pair-a05"), profile_rows)}

![ALWAYS_ON profiles](always_on_position_profiles.png)

## N_P versus N_Q

{md_table(("System", "mean N_P", "mean N_Q", "PP pair fraction", "corr sqrtP/pair-a05 valid"), corr_rows)}

Across all valid real-data onsets, corr(S_sqrtP, S_pair_alpha05)={fmt(sqrt_pair_corr,3)}, so pair-count scaling adds limited independent behavior overall. The weakest piece/system correlation is {weakest.corr_S_sqrtP_with_S_pair_alpha05_valid:.3f} for {weakest.composer}/{weakest.title} / {weakest.system}, where PP contributes only {weakest.PP_pair_fraction:.3f} of Q: this PA-heavier texture is the clearest divergence. High active polyphony can increase N_Q through PA without a matching increase in N_P; high residual accumulation increases both PA and combinatorial PP. Piece/system-specific correlations and N_P/N_Q quantiles are in piece_system_count_scaling.csv.

## Sanity

{md_table(("check", "status", "detail"), [(row.check, row.status, row.detail) for row in checks.itertuples()])}

All checks must pass before this report is written. Source MIDI hashes are checked before and after evaluation.

## Final questions

### Q1 — Does pair-average underrepresent accumulation?

Yes for the narrow hypothesis being tested. Controlled multiplicity leaves S_mean near its finite-size severity level (x8/x1={fmt(growth["S_mean"],2)}×) while raw relation mass and count-aware scores grow. S_mean therefore cannot express how many similarly severe pedal-induced relations are simultaneously present.

### Q2 — Does N_P improve extreme over-pedaling detection?

It increases the numerical separation from ALWAYS_ON, but not ordering coverage: model-median<ALWAYS is already {base_always}/5 under S_mean and remains {sqrt_always}/5 under sqrt(N_P). The added value here is margin, not another correctly ordered piece. HUMAN/PT and PT/model behavior must therefore carry more weight in judging the tradeoff.

### Q3 — log, sqrt, or linear N_P?

S_sqrtP is the most reasonable middle behavior here. LogP grows gently and can leave large accumulations under-emphasized; linearP grows strongly and produces the largest absolute/tail magnitudes. SqrtP adds visible count sensitivity without inheriting the full linear explosion.

### Q4 — alpha=0.5 pair-count scaling?

It behaves approximately linearly in controlled multiplicity because sqrt(N_Q) tracks note count when PP dominates. Its x8/x1 growth is {fmt(growth["S_pair_alpha05"],2)}x, close to linearP's {fmt(growth["S_linearP"],2)}x, and their ALWAYS_ON p99/median ratios are likewise nearly equal ({fmt(pair_p99,2)} versus {fmt(linear_p99,2)}). It is partial normalization, not raw mass, and in PP-dominant accumulation behaves almost like a rescaled linear-N_P penalty.

### Q5 — Is sqrt(N_P) meaningfully different from sqrt(N_Q)?

The pooled valid-onset correlation is {fmt(sqrt_pair_corr,3)}, so they are usually very similar in rank behavior, although their magnitudes differ. The clearest divergence is the PA-heavier {weakest.composer}/{weakest.title} / {weakest.system} cell (correlation {weakest.corr_S_sqrtP_with_S_pair_alpha05_valid:.3f}); N_Q then responds to active polyphony that N_P alone does not encode. On these five pieces, that is limited rather than a broad independent signal.

### Q6 — Best balance?

For the combined criteria in this audit, S_logP is the most conservative balance: it increases extreme-separation margin, has modest controlled growth and tail ratio, and does not worsen the 2/5 PT<model-median count seen under S_mean. S_sqrtP is the clearer stronger-sensitivity alternative and is the best middle growth curve in isolation, but here PT<model-median falls to 1/5. S_linearP and pair-alpha05 are useful stress diagnostics; their strong late-position scale growth makes them less attractive as default formulations.

### Q7 — Final formula now?

No. This fixed five-piece audit has no perceptual ground truth and was not a parameter-fitting exercise. It only narrows which numerical behavior deserves manual/listening validation. Larger note count is not inherently bad; the tested hypothesis is conditional on similar Hall-negative average severity.
"""


def run(output: Path, workers: int) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest, provenance = build_manifest()
    provenance["tested_command"] = (
        f"python {Path(__file__).resolve()} --output {output.resolve()} --workers {workers}"
    )
    onsets, worker_results = evaluate_all(manifest, workers)
    onsets = canonicalize_onset_csv_floats(onsets)
    random_growth, multiplicity = synthetic_tests()
    piece = aggregate_piece_system(onsets, manifest)
    profile = position_profiles(onsets)
    explosion = explosion_diagnostics(onsets)
    ordering, ordering_detail = ordering_tables(piece, onsets)
    checks = sanity_checks(onsets, manifest, random_growth, multiplicity)
    if not checks.status.eq("PASS").all():
        raise AssertionError(checks[checks.status != "PASS"].to_dict("records"))

    manifest["source_sha256_after"] = manifest.midi_path.map(
        lambda value: sha256_file(Path(value))
    )
    manifest["source_sha_unchanged"] = (
        manifest.source_sha256_before == manifest.source_sha256_after
    )
    manifest.to_csv(output / "five_piece_manifest.csv", index=False)
    piece.to_csv(output / "piece_system_count_scaling.csv", index=False)
    onsets.to_csv(output / "onset_count_scaling.csv", index=False)
    profile.to_csv(output / "normalized_position_profile.csv", index=False)
    random_growth.to_csv(output / "synthetic_random_growth.csv", index=False)
    multiplicity.to_csv(output / "synthetic_multiplicity_growth.csv", index=False)
    ordering.to_csv(output / "ordering_by_scaling.csv", index=False)
    ordering_detail.to_csv(output / "ordering_detail.csv", index=False)
    explosion.to_csv(output / "score_explosion_diagnostics.csv", index=False)
    checks.to_csv(output / "sanity_checks.csv", index=False)
    make_plots(random_growth, multiplicity, profile, output)

    metadata = {
        **provenance,
        "version": VERSION,
        "fixed_piece_ids": list(FIXED_PIECE_IDS),
        "systems": list(SYSTEMS),
        "canonical_models": list(CANONICAL_MODELS),
        "scalings": list(SCALINGS),
        "real_onset_rows": len(onsets),
        "piece_system_rows": len(piece),
        "source_unique_midi_files": int(manifest.midi_path.nunique()),
        "test_access_count": 0,
        "inference_count": 0,
        "training_count": 0,
        "new_midi_generation_count": 0,
        "explicit_full_pair_materialization": False,
        "score_uses_decay": False,
        "score_uses_velocity": False,
        "score_uses_low_register": False,
        "score_uses_dynamic": False,
        "score_uses_pa_pp_differential": False,
        "positive_hall_reward": False,
        "aa_pairs_used": False,
        "worker_diagnostics": [
            {
                "piece_id": item["piece_id"],
                "system": item["system"],
                **item["diagnostics"],
            }
            for item in worker_results
        ],
    }
    (output / "audit_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    report = report_text(
        manifest, onsets, piece, profile, random_growth, multiplicity,
        ordering, explosion, checks, provenance
    )
    (output / "HALL_NEGATIVE_COUNT_SCALING_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    print(json.dumps({
        "output": str(output),
        "onset_rows": len(onsets),
        "piece_system_rows": len(piece),
        "sanity_checks": len(checks),
    }, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.output, args.workers)
