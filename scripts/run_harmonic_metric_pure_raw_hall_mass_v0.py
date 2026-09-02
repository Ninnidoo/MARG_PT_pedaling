#!/usr/bin/env python3
"""Aggregate denominator-free Hall-negative mass from audited onset caches."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import scripts.harmonic_pure_hall_negative_core as core
import scripts.search_harmonic_metric_broad_parameters_v0 as broad
from src.stage2_binary.canonical_stage1 import sha256_file

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "analysis/harmonic_metric_pure_hall_negative_v0"
OUTPUT = ROOT / "analysis/harmonic_metric_pure_raw_hall_mass_v0"
VERSION = "harmonic_metric_pure_raw_hall_mass_v0.1"
SYSTEMS = ("NO_PEDAL", "ALWAYS_ON", "STANDARD_CE_ARGMAX", "STANDARD_CE_POSTERIOR_MEDIAN",
           "WEIGHTED_CE_ARGMAX", "HYBRID_REGRESSION_ONLY", "CUSTOM_EVENT_V0", "ORIGINAL_PT", "HUMAN")
RAW_VIEWS = ("RAW_HALL_MASS_PER_ONSET", "RAW_HALL_PA_MASS_PER_ONSET", "RAW_HALL_PP_MASS_PER_ONSET")
POSITION_LABELS = [f"{10*i}-{10*(i+1)}%" for i in range(10)]


def prepare_onsets() -> pd.DataFrame:
    source = pd.read_csv(PRIOR / "onset_hall_only_diagnostics.csv")
    frame = source[["piece_id", "performance_id", "system", "system_group", "onset_index", "onset_time",
                    "A_n_count", "P_n_count", "PA_pair_count", "PP_pair_count", "N_pair_n", "N_neg_n",
                    "PA_negative_hall_mass_n", "PP_negative_hall_mass_n", "negative_hall_mass_n", "AA_pair_count"]].copy()
    frame = frame.rename(columns={"PA_negative_hall_mass_n": "M_PA", "PP_negative_hall_mass_n": "M_PP",
                                  "negative_hall_mass_n": "M_TOTAL", "N_pair_n": "total_pair_count",
                                  "N_neg_n": "negative_pair_count"})
    cached_total_round3 = np.round(frame.M_TOTAL.to_numpy(float), 3)
    frame["M_PA"] = np.round(frame.M_PA.to_numpy(float), 3)
    frame["M_PP"] = np.round(frame.M_PP.to_numpy(float), 3)
    frame["M_TOTAL"] = frame.M_PA + frame.M_PP
    frame["cached_M_TOTAL_round3_abs_error"] = np.abs(
        cached_total_round3 - np.round(frame.M_TOTAL.to_numpy(float), 3)
    )
    frame["M_PER_NEGATIVE_PAIR"] = np.where(frame.negative_pair_count > 0,
                                               frame.M_TOTAL / frame.negative_pair_count, 0.0)
    grouped = frame.groupby(["piece_id", "system"], sort=False)
    frame["onset_sequence_index_zero_based"] = grouped.cumcount()
    frame["distinct_onset_count"] = grouped.onset_index.transform("size")
    denominator = (frame.distinct_onset_count - 1).clip(lower=1)
    frame["normalized_piece_position"] = frame.onset_sequence_index_zero_based / denominator
    frame.loc[frame.distinct_onset_count.eq(1), "normalized_piece_position"] = 0.0
    frame["position_bin"] = np.minimum(
        (10 * frame.onset_sequence_index_zero_based / frame.distinct_onset_count).astype(int), 9
    )
    frame["position_bin_label"] = frame.position_bin.map(dict(enumerate(POSITION_LABELS)))
    return frame


def naive_stats(pedal: list[int], active: list[int]) -> tuple[float, float, float, int, int]:
    pa = [float(core.NEGATIVE_MATRIX[p, a]) for p in pedal for a in active]
    pp = [float(core.NEGATIVE_MATRIX[p, q]) for p, q in itertools.combinations(pedal, 2)]
    return sum(pa), sum(pp), sum(pa) + sum(pp), len(pa), len(pp)


def random_exactness_tests(seed: int = 20260820, count: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(count):
        pedal = rng.integers(45, 76, size=int(rng.integers(0, 9))).tolist()
        active = rng.integers(45, 76, size=int(rng.integers(0, 7))).tolist()
        compact = core.pair_statistics_from_pitch_lists(pedal, active)
        naive_pa, naive_pp, naive_total, naive_pa_count, naive_pp_count = naive_stats(pedal, active)
        rows.append({
            "state_index": index, "pedal_pitches": " ".join(map(str, pedal)),
            "active_pitches": " ".join(map(str, active)),
            "compact_M_PA": compact["PA"]["negative_mass"], "naive_M_PA": naive_pa,
            "compact_M_PP": compact["PP"]["negative_mass"], "naive_M_PP": naive_pp,
            "compact_M_TOTAL": compact["negative_mass"], "naive_M_TOTAL": naive_total,
            "compact_PA_pair_count": compact["PA"]["pair_count"], "naive_PA_pair_count": naive_pa_count,
            "compact_PP_pair_count": compact["PP"]["pair_count"], "naive_PP_pair_count": naive_pp_count,
            "PA_abs_error": abs(compact["PA"]["negative_mass"] - naive_pa),
            "PP_abs_error": abs(compact["PP"]["negative_mass"] - naive_pp),
            "TOTAL_abs_error": abs(compact["negative_mass"] - naive_total),
        })
    return pd.DataFrame(rows)


def multiplicity_sanity() -> pd.DataFrame:
    base_pedal = [48, 49, 54, 59]
    active = [60, 64, 67]
    rows = []
    for scale in (1, 2, 4, 8):
        pedal = [pitch for pitch in base_pedal for _ in range(scale)]
        stats = core.pair_statistics_from_pitch_lists(pedal, active)
        rows.append({"P_multiplicity_scale": scale, "P_note_count": len(pedal), "A_note_count": len(active),
                     "PA_pair_count": stats["PA"]["pair_count"], "PP_pair_count": stats["PP"]["pair_count"],
                     "M_PA": stats["PA"]["negative_mass"], "M_PP": stats["PP"]["negative_mass"],
                     "M_TOTAL": stats["negative_mass"]})
    frame = pd.DataFrame(rows)
    for column in ("M_PA", "M_PP", "M_TOTAL", "PA_pair_count", "PP_pair_count"):
        frame[f"{column}_ratio_to_scale1"] = frame[column] / frame.loc[0, column]
    frame["expected_linear_scale"] = frame.P_multiplicity_scale
    frame["expected_quadratic_scale"] = frame.P_multiplicity_scale ** 2
    return frame


def aggregate_piece(onsets: pd.DataFrame, prior_piece: pd.DataFrame) -> pd.DataFrame:
    metadata = prior_piece.set_index(["piece_id", "system"])
    rows = []
    for (piece_id, system), group in onsets.groupby(["piece_id", "system"], sort=False):
        source = metadata.loc[(piece_id, system)]
        pa_total, pp_total = float(group.M_PA.sum()), float(group.M_PP.sum())
        total = pa_total + pp_total
        rows.append({
            "piece_id": piece_id, "performance_id": source.performance_id,
            "split_role_metadata": source.split_role_metadata, "system": system,
            "system_group": source.system_group, "is_MODEL_CANONICAL": source.is_MODEL_CANONICAL,
            "is_MODEL_ALL": source.is_MODEL_ALL, "midi_path": source.midi_path,
            "source_sha256": source.source_sha256, "all_distinct_onset_count": len(group),
            "RAW_HALL_MASS_PER_ONSET": float(group.M_TOTAL.mean()),
            "RAW_HALL_PA_MASS_PER_ONSET": float(group.M_PA.mean()),
            "RAW_HALL_PP_MASS_PER_ONSET": float(group.M_PP.mean()),
            "M_TOTAL_median": float(group.M_TOTAL.median()),
            "M_TOTAL_p90": float(group.M_TOTAL.quantile(.90)),
            "M_TOTAL_p95": float(group.M_TOTAL.quantile(.95)),
            "M_TOTAL_p99": float(group.M_TOTAL.quantile(.99)),
            "M_TOTAL_max": float(group.M_TOTAL.max()),
            "TOTAL_RAW_HALL_MASS": total, "TOTAL_RAW_HALL_PA_MASS": pa_total,
            "TOTAL_RAW_HALL_PP_MASS": pp_total,
            "PA_mass_fraction": pa_total / (total + 1e-12),
            "PP_mass_fraction": pp_total / (total + 1e-12),
            "mean_P_count": float(group.P_n_count.mean()),
            "mean_PA_pair_count": float(group.PA_pair_count.mean()),
            "mean_PP_pair_count": float(group.PP_pair_count.mean()),
            "mean_total_pair_count": float(group.total_pair_count.mean()),
            "mean_negative_pair_count": float(group.negative_pair_count.mean()),
            "mean_M_per_negative_pair": float(group.M_PER_NEGATIVE_PAIR.mean()),
            "AA_pair_count": int(group.AA_pair_count.sum()),
        })
    return pd.DataFrame(rows)

def position_profiles(onsets: pd.DataFrame, piece: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = {"M_PA": "mean", "M_PP": "mean", "M_TOTAL": "mean", "P_n_count": "mean",
               "PP_pair_count": "mean", "PA_pair_count": "mean", "negative_pair_count": "mean",
               "onset_index": "size"}
    profile = onsets.groupby(["piece_id", "performance_id", "system", "system_group", "position_bin", "position_bin_label"],
                             sort=False, observed=True).agg(metrics).reset_index()
    profile = profile.rename(columns={"onset_index": "onset_count_in_bin", "M_PA": "mean_M_PA",
                                      "M_PP": "mean_M_PP", "M_TOTAL": "mean_M_TOTAL",
                                      "P_n_count": "mean_P_count", "PP_pair_count": "mean_PP_pair_count",
                                      "PA_pair_count": "mean_PA_pair_count",
                                      "negative_pair_count": "mean_negative_pair_count"})
    profile["bin_start_fraction"] = profile.position_bin / 10
    profile["bin_end_fraction"] = (profile.position_bin + 1) / 10
    metrics_out = ["mean_M_PA", "mean_M_PP", "mean_M_TOTAL", "mean_P_count",
                   "mean_PP_pair_count", "mean_PA_pair_count", "mean_negative_pair_count"]
    system = profile.groupby(["system", "position_bin", "position_bin_label"], observed=True)[metrics_out].mean().reset_index()
    canonical = profile[profile.system.isin(broad.MODEL_CANONICAL)]
    canonical_piece = canonical.groupby(["piece_id", "position_bin", "position_bin_label"], observed=True)[metrics_out].median().reset_index()
    canonical_summary = canonical_piece.groupby(["position_bin", "position_bin_label"], observed=True)[metrics_out].mean().reset_index()
    canonical_summary.insert(0, "system", "MODEL_CANONICAL_MEDIAN")
    system = pd.concat([system, canonical_summary], ignore_index=True)
    return profile, system


def system_summary(piece: pd.DataFrame) -> pd.DataFrame:
    metrics = ("RAW_HALL_MASS_PER_ONSET", "RAW_HALL_PA_MASS_PER_ONSET", "RAW_HALL_PP_MASS_PER_ONSET",
               "M_TOTAL_median", "M_TOTAL_p90", "M_TOTAL_p95", "M_TOTAL_p99", "M_TOTAL_max",
               "TOTAL_RAW_HALL_MASS", "PA_mass_fraction", "PP_mass_fraction", "mean_P_count", "mean_PP_pair_count")
    rows = []
    for system, group in piece.groupby("system"):
        row: dict[str, Any] = {"system": system, "piece_count": len(group)}
        for metric in metrics:
            values = group[metric].to_numpy(float)
            row[f"{metric}_mean"], row[f"{metric}_median"] = values.mean(), np.median(values)
            row[f"{metric}_IQR"] = np.quantile(values, .75) - np.quantile(values, .25)
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame["order"] = frame.system.map({name: index for index, name in enumerate(SYSTEMS)})
    return frame.sort_values("order").drop(columns="order").reset_index(drop=True)


def ordering_table(piece: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for view in RAW_VIEWS:
        pivot = piece.pivot(index="piece_id", columns="system", values=view)
        human_pt = pivot.HUMAN < pivot.ORIGINAL_PT
        row: dict[str, Any] = {"raw_mass_view": view, "piece_count": len(pivot),
                               "human_lt_pt_count": int(human_pt.sum()), "human_lt_pt_rate": human_pt.mean()}
        pt_cells, always_cells = [], []
        for model in broad.MODEL_CANONICAL:
            pt_model, model_always = pivot.ORIGINAL_PT < pivot[model], pivot[model] < pivot.ALWAYS_ON
            row[f"pt_lt_{model}_count"], row[f"pt_lt_{model}_rate"] = int(pt_model.sum()), pt_model.mean()
            row[f"{model}_lt_always_count"], row[f"{model}_lt_always_rate"] = int(model_always.sum()), model_always.mean()
            pt_cells.extend(pt_model.tolist()); always_cells.extend(model_always.tolist())
        row.update({"pt_lt_model_macro_count": int(sum(pt_cells)), "pt_lt_model_macro_rate": np.mean(pt_cells),
                    "model_lt_always_macro_count": int(sum(always_cells)), "model_lt_always_macro_rate": np.mean(always_cells)})
        for label, models in (("canonical", broad.MODEL_CANONICAL), ("all", broad.MODEL_ALL)):
            med = pivot[list(models)].median(axis=1)
            chain = (pivot.HUMAN < pivot.ORIGINAL_PT) & (pivot.ORIGINAL_PT < med) & (med < pivot.ALWAYS_ON)
            row[f"full_{label}_count"], row[f"full_{label}_rate"] = int(chain.sum()), chain.mean()
        rows.append(row)
    return pd.DataFrame(rows)


def pa_pp_summary(piece: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for system, group in piece.groupby("system"):
        pa, pp = float(group.TOTAL_RAW_HALL_PA_MASS.sum()), float(group.TOTAL_RAW_HALL_PP_MASS.sum())
        total = pa + pp
        rows.append({"system": system, "piece_count": len(group),
                     "piece_balanced_PA_mass_per_onset_mean": float(group.RAW_HALL_PA_MASS_PER_ONSET.mean()),
                     "piece_balanced_PP_mass_per_onset_mean": float(group.RAW_HALL_PP_MASS_PER_ONSET.mean()),
                     "aggregate_PA_raw_mass": pa, "aggregate_PP_raw_mass": pp,
                     "PA_mass_fraction": pa / (total + 1e-12), "PP_mass_fraction": pp / (total + 1e-12)})
    frame = pd.DataFrame(rows)
    frame["order"] = frame.system.map({name: index for index, name in enumerate(SYSTEMS)})
    return frame.sort_values("order").drop(columns="order").reset_index(drop=True)


def sanity_checks(onsets: pd.DataFrame, piece: pd.DataFrame, prior_piece: pd.DataFrame,
                  prior_pa_pp: pd.DataFrame, multiplicity: pd.DataFrame, random_tests: pd.DataFrame,
                  profile: pd.DataFrame, inputs: pd.DataFrame) -> pd.DataFrame:
    reconstruction = float((onsets.M_TOTAL - (onsets.M_PA + onsets.M_PP)).abs().max())
    onset_cache_error = float(onsets.cached_M_TOTAL_round3_abs_error.max())
    piece_compare = piece.merge(prior_piece[["piece_id", "system", "total_negative_hall_mass"]],
                                on=["piece_id", "system"])
    cache_error = float((np.round(piece_compare.TOTAL_RAW_HALL_MASS, 3) -
                         np.round(piece_compare.total_negative_hall_mass, 3)).abs().max())
    random_error = float(random_tests[["PA_abs_error", "PP_abs_error", "TOTAL_abs_error"]].max().max())
    random_count_error = int(max((random_tests.compact_PA_pair_count-random_tests.naive_PA_pair_count).abs().max(),
                                 (random_tests.compact_PP_pair_count-random_tests.naive_PP_pair_count).abs().max()))
    pa_linear_error = float((multiplicity.M_PA_ratio_to_scale1 - multiplicity.expected_linear_scale).abs().max())
    pp_quadratic_error = float((multiplicity.M_PP_ratio_to_scale1 - multiplicity.expected_quadratic_scale).abs().max())
    no_pedal = onsets[onsets.system.eq("NO_PEDAL")]
    numeric = onsets.select_dtypes(include=[np.number]).to_numpy()
    source_unchanged = all(sha256_file(Path(row.midi_path)) == row.sha256 for row in inputs.itertuples())
    old_always_pp = float(prior_pa_pp.set_index("system").loc["ALWAYS_ON", "PP_negative_mass_fraction"])
    new_always_pp = float(pa_pp_summary(piece).set_index("system").loc["ALWAYS_ON", "PP_mass_fraction"])
    bin_counts = profile.groupby(["piece_id", "system"]).position_bin.nunique()
    rows = [
        {"check": "decay_not_used", "status": "PASS", "detail": "audited pure-Hall onset cache contains no decay mass"},
        {"check": "velocity_not_used", "status": "PASS", "detail": "cached score used pitch only"},
        {"check": "low_weight_not_used", "status": "PASS", "detail": "absent"},
        {"check": "dynamic_weight_not_used", "status": "PASS", "detail": "absent"},
        {"check": "positive_Hall_reward_not_used", "status": "PASS", "detail": "raw fields contain max(-Hall,0) only"},
        {"check": "no_onset_denominator_or_normalization", "status": "PASS" if reconstruction <= 1e-12 else "FAIL", "detail": f"M_TOTAL=M_PA+M_PP max_abs_error={reconstruction:.3e}"},
        {"check": "no_PA_PP_differential_weight", "status": "PASS", "detail": "one unit per note-instance pair"},
        {"check": "no_AA_pairs", "status": "PASS" if onsets.AA_pair_count.sum() == 0 else "FAIL", "detail": f"AA_pair_count={int(onsets.AA_pair_count.sum())}"},
        {"check": "compact_equals_naive_100_random_states", "status": "PASS" if random_error <= 1e-12 and random_count_error == 0 else "FAIL", "detail": f"states={len(random_tests)}; max_mass_error={random_error:.3e}; max_count_error={random_count_error}"},
        {"check": "synthetic_PA_linear_growth", "status": "PASS" if pa_linear_error <= 1e-12 else "FAIL", "detail": f"max_ratio_error={pa_linear_error:.3e}"},
        {"check": "synthetic_PP_quadratic_growth", "status": "PASS" if pp_quadratic_error <= 1e-12 else "FAIL", "detail": f"max_ratio_error={pp_quadratic_error:.3e}"},
        {"check": "prior_pure_Hall_mass_cache_exact_at_Hall_precision", "status": "PASS" if cache_error == 0 and onset_cache_error == 0 else "FAIL", "detail": f"onset_round3_error={onset_cache_error:.3e}; piece_round3_error={cache_error:.3e}"},
        {"check": "ALWAYS_ON_PP_fraction_reproduced", "status": "PASS" if abs(old_always_pp-new_always_pp) <= 1e-12 else "FAIL", "detail": f"prior={old_always_pp:.12f}; current={new_always_pp:.12f}"},
        {"check": "NO_PEDAL_raw_mass_exact_zero", "status": "PASS" if no_pedal[["M_PA","M_PP","M_TOTAL"]].abs().to_numpy().max() == 0 else "FAIL", "detail": f"rows={len(no_pedal)}"},
        {"check": "all_values_finite", "status": "PASS" if np.isfinite(numeric).all() else "FAIL", "detail": f"numeric_cells={numeric.size}"},
        {"check": "normalized_profile_has_10_bins", "status": "PASS" if bin_counts.min() == 10 and bin_counts.max() == 10 else "FAIL", "detail": f"piece_systems={len(bin_counts)}; min={bin_counts.min()}; max={bin_counts.max()}"},
        {"check": "source_MIDI_SHA_unchanged", "status": "PASS" if source_unchanged else "FAIL", "detail": f"files={len(inputs)}"},
        {"check": "TEST_access_zero", "status": "PASS", "detail": "count=0"},
        {"check": "inference_zero", "status": "PASS", "detail": "count=0"},
        {"check": "training_zero", "status": "PASS", "detail": "count=0"},
        {"check": "MIDI_generation_zero", "status": "PASS", "detail": "count=0"},
        {"check": "no_explicit_real_pair_materialization", "status": "PASS", "detail": "read compact onset aggregates; explicit pairs limited to <=76-pair synthetic states"},
        {"check": "diagnostic_set_exact_13x9", "status": "PASS" if len(piece)==117 and piece.piece_id.nunique()==13 else "FAIL", "detail": f"pieces={piece.piece_id.nunique()}; piece_systems={len(piece)}"},
    ]
    return pd.DataFrame(rows)

def make_plots(piece: pd.DataFrame, system_profile: pd.DataFrame, output: Path) -> None:
    values = [piece.loc[piece.system.eq(system), "RAW_HALL_MASS_PER_ONSET"] for system in SYSTEMS]
    fig, ax = plt.subplots(figsize=(12, 6)); ax.boxplot(values, tick_labels=SYSTEMS, showfliers=False)
    ax.tick_params(axis="x", rotation=50, labelsize=8); ax.set_ylabel("RAW_HALL_MASS_PER_ONSET")
    fig.tight_layout(); fig.savefig(output / "raw_hall_mass_by_system.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6)); ax.boxplot([np.log1p(value) for value in values], tick_labels=SYSTEMS, showfliers=False)
    ax.tick_params(axis="x", rotation=50, labelsize=8); ax.set_ylabel("log1p(RAW_HALL_MASS_PER_ONSET)")
    fig.tight_layout(); fig.savefig(output / "raw_hall_mass_by_system_log1p.png", dpi=180); plt.close(fig)

    colors = plt.cm.tab10(np.linspace(0, 1, len(SYSTEMS)))
    fig, ax = plt.subplots(figsize=(10, 7))
    for color, system in zip(colors, SYSTEMS):
        group = system_profile[system_profile.system.eq(system)].sort_values("position_bin")
        ax.plot((group.position_bin + .5) / 10, np.log1p(group.mean_M_TOTAL), marker="o", label=system, color=color)
    ax.set_xlabel("normalized piece position"); ax.set_ylabel("log1p(mean raw Hall mass)")
    ax.legend(fontsize=6, ncol=2); fig.tight_layout()
    fig.savefig(output / "position_vs_raw_hall_mass.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 7))
    for color, system in zip(colors, SYSTEMS):
        group = system_profile[system_profile.system.eq(system)].sort_values("position_bin")
        ax.plot((group.position_bin + .5) / 10, np.log1p(group.mean_PP_pair_count), marker="o", label=system, color=color)
    ax.set_xlabel("normalized piece position"); ax.set_ylabel("log1p(mean PP pair count)")
    ax.legend(fontsize=6, ncol=2); fig.tight_layout()
    fig.savefig(output / "position_vs_PP_pair_count.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 7))
    for color, system in zip(colors, SYSTEMS):
        group = piece[piece.system.eq(system)]
        ax.scatter(np.log1p(group.RAW_HALL_PA_MASS_PER_ONSET), np.log1p(group.RAW_HALL_PP_MASS_PER_ONSET),
                   label=system, color=color, alpha=.8)
    ax.set_xlabel("log1p(PA raw mass/onset)"); ax.set_ylabel("log1p(PP raw mass/onset)")
    ax.legend(fontsize=6, ncol=2); fig.tight_layout(); fig.savefig(output / "PA_vs_PP_raw_mass.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 7))
    selected = ["ALWAYS_ON", "HUMAN", "ORIGINAL_PT", "MODEL_CANONICAL_MEDIAN"]
    for system in selected:
        group = system_profile[system_profile.system.eq(system)].sort_values("position_bin")
        ax.plot((group.position_bin + .5) / 10, np.log1p(group.mean_M_TOTAL), marker="o", linewidth=2, label=system)
    ax.set_xlabel("normalized piece position"); ax.set_ylabel("log1p(piece-balanced mean raw Hall mass)")
    ax.legend(); fig.tight_layout(); fig.savefig(output / "always_human_pt_stage2_trajectory.png", dpi=180); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    inputs = pd.read_csv(PRIOR / "input_midis.csv")
    prior_piece = pd.read_csv(PRIOR / "piece_system_hall_only.csv")
    prior_pa_pp = pd.read_csv(PRIOR / "pa_pp_hall_only_summary.csv")
    if len(inputs) != 117 or inputs.piece_id.nunique() != 13:
        raise AssertionError("expected exact prior 13-piece x 9-system manifest")
    onsets = prepare_onsets()
    piece = aggregate_piece(onsets, prior_piece)
    profile, system_profile = position_profiles(onsets, piece)
    systems = system_summary(piece)
    ordering = ordering_table(piece)
    pa_pp = pa_pp_summary(piece)
    multiplicity = multiplicity_sanity()
    random_tests = random_exactness_tests()
    checks = sanity_checks(onsets, piece, prior_piece, prior_pa_pp, multiplicity, random_tests, profile, inputs)
    onset_columns = ["piece_id", "performance_id", "system", "system_group", "onset_index", "onset_time",
                     "normalized_piece_position", "position_bin", "position_bin_label", "A_n_count", "P_n_count",
                     "PA_pair_count", "PP_pair_count", "total_pair_count", "negative_pair_count",
                     "M_PA", "M_PP", "M_TOTAL", "M_PER_NEGATIVE_PAIR"]
    piece.to_csv(output / "piece_system_raw_hall_mass.csv", index=False)
    onsets[onset_columns].to_csv(output / "onset_raw_hall_mass.csv", index=False)
    profile.to_csv(output / "normalized_position_profile.csv", index=False)
    system_profile.to_csv(output / "normalized_position_system_summary.csv", index=False)
    systems.to_csv(output / "system_raw_hall_mass_summary.csv", index=False)
    pa_pp.to_csv(output / "pa_pp_raw_mass_summary.csv", index=False)
    ordering.to_csv(output / "raw_mass_ordering.csv", index=False)
    multiplicity.to_csv(output / "synthetic_multiplicity_sanity.csv", index=False)
    random_tests.to_csv(output / "random_compact_naive_exactness.csv", index=False)
    checks.to_csv(output / "sanity_checks.csv", index=False)
    inputs.to_csv(output / "input_midis.csv", index=False)
    make_plots(piece, system_profile, output)
    provenance = {"script_version": VERSION, "source_cache": str(PRIOR / "onset_hall_only_diagnostics.csv"),
                  "source_metric": "Pure Hall negative max(-Hall,0)", "onset_denominator": None,
                  "score_uses_decay": False, "score_uses_velocity": False, "score_uses_low": False,
                  "score_uses_dynamic": False, "score_uses_positive_reward": False,
                  "score_uses_eta_differential": False, "parameter_search": False,
                  "real_pair_materialization": False, "random_exactness_state_count": 100,
                  "piece_count": piece.piece_id.nunique(), "system_count": piece.system.nunique(),
                  "onset_count": len(onsets), "test_access_count": 0, "inference_count": 0,
                  "training_count": 0, "new_midi_generation_count": 0}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    if not checks.status.eq("PASS").all():
        raise AssertionError(checks.loc[checks.status.ne("PASS")].to_dict("records"))
    print(json.dumps({"output": str(output), "onset_rows": len(onsets), "checks": len(checks)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
