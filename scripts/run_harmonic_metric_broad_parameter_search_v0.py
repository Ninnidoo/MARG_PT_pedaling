#!/usr/bin/env python3
"""Orchestrate the broad harmonic-metric parameter search."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import scripts.audit_harmonic_metric_kdyn_blow_pilot_v0 as pilot
import scripts.search_harmonic_metric_broad_parameters_v0 as c
from scripts.harmonic_broad_selected_scores import selected_scores
from src.stage2_binary.canonical_stage1 import sha256_file


LEX = [
    "canonical_chain_pass_rate", "all_model_chain_pass_rate", "individual_model_compliance",
    "human_gt_pt_rate", "pt_gt_model_rate", "model_gt_extremes_rate", "median_minimum_gap",
]


def ordering_metrics(scores: pd.DataFrame, configs: pd.DataFrame, role: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    pivot = scores.pivot(index=["config_id", "piece_id"], columns="system", values="H_piece")
    lookup = configs.set_index("config_id")
    summaries, pieces = [], []
    for config_id, frame in pivot.groupby(level="config_id", sort=False):
        frame = frame.droplevel("config_id")
        always = frame.ALWAYS_ON.to_numpy()
        extreme = np.maximum(0.0, always)
        canonical = frame[list(c.MODEL_CANONICAL)].to_numpy()
        all_models = frame[list(c.MODEL_ALL)].to_numpy()
        mc, ma = np.median(canonical, axis=1), np.median(all_models, axis=1)
        pt, human = frame.ORIGINAL_PT.to_numpy(), frame.HUMAN.to_numpy()
        chain_c = (extreme < mc) & (mc < pt) & (pt < human)
        chain_a = (extreme < ma) & (ma < pt) & (pt < human)
        compliance_c = (canonical > extreme[:, None]) & (canonical < pt[:, None])
        compliance_a = (all_models > extreme[:, None]) & (all_models < pt[:, None])
        pt_c, pt_a = pt[:, None] > canonical, pt[:, None] > all_models
        g1, g2, g3 = mc - extreme, pt - mc, human - pt
        ga1, ga2 = ma - extreme, pt - ma
        mingap, minga = np.minimum.reduce([g1, g2, g3]), np.minimum.reduce([ga1, ga2, g3])
        row = {
            "config_id": config_id, "subset_role": role, "piece_count": len(frame),
            "canonical_chain_pass_rate": chain_c.mean(), "all_model_chain_pass_rate": chain_a.mean(),
            "canonical_individual_model_compliance": compliance_c.mean(),
            "individual_model_compliance": compliance_a.mean(), "human_gt_pt_rate": (human > pt).mean(),
            "canonical_pt_gt_model_rate": pt_c.mean(), "pt_gt_model_rate": pt_a.mean(),
            "canonical_model_gt_no_pedal_rate": (canonical > 0).mean(),
            "model_gt_no_pedal_rate": (all_models > 0).mean(),
            "canonical_model_gt_always_rate": (canonical > always[:, None]).mean(),
            "model_gt_always_rate": (all_models > always[:, None]).mean(),
            "canonical_model_gt_extremes_rate": (canonical > extreme[:, None]).mean(),
            "model_gt_extremes_rate": (all_models > extreme[:, None]).mean(),
            "human_gt_always_rate": (human > always).mean(), "always_le_zero_rate": (always <= 0).mean(),
            "median_gap_extreme_model": np.median(g1), "worst_quartile_gap_extreme_model": np.quantile(g1, .25),
            "median_gap_model_pt": np.median(g2), "worst_quartile_gap_model_pt": np.quantile(g2, .25),
            "median_gap_pt_human": np.median(g3), "worst_quartile_gap_pt_human": np.quantile(g3, .25),
            "median_minimum_gap": np.median(mingap), "all_median_gap_extreme_model": np.median(ga1),
            "all_median_gap_model_pt": np.median(ga2), "all_median_minimum_gap": np.median(minga),
            **lookup.loc[config_id].to_dict(),
        }
        for index, system in enumerate(c.MODEL_ALL):
            row[f"pt_gt_{system}_rate"] = pt_a[:, index].mean()
        summaries.append(row)
        for index, piece_id in enumerate(frame.index):
            pieces.append({
                "config_id": config_id, "subset_role": role, "piece_id": piece_id,
                "extreme_upper_score": extreme[index], "M_canonical": mc[index], "M_all": ma[index],
                "ORIGINAL_PT": pt[index], "HUMAN": human[index],
                "canonical_chain_pass": chain_c[index], "all_model_chain_pass": chain_a[index],
                "gap_extreme_model": g1[index], "gap_model_pt": g2[index], "gap_pt_human": g3[index],
                "all_gap_extreme_model": ga1[index], "all_gap_model_pt": ga2[index],
                "minimum_gap": mingap[index], "all_minimum_gap": minga[index],
            })
    return pd.DataFrame(summaries), pd.DataFrame(pieces)


def rank_configs(results: pd.DataFrame) -> pd.DataFrame:
    ranked = results.sort_values(
        LEX + ["reference_distance_grid_L1", "alpha_decay", "eta_PP", "m0", "beta_low", "kappa_dyn"],
        ascending=[False] * len(LEX) + [True] * 6, kind="mergesort").reset_index(drop=True)
    ranked["lexicographic_rank"] = np.arange(1, len(ranked) + 1)
    objectives = ranked[LEX].to_numpy(float)
    pareto = np.ones(len(ranked), bool)
    for index, row in enumerate(objectives):
        pareto[index] = not np.any(np.all(objectives >= row, axis=1) & np.any(objectives > row, axis=1))
    ranked["pareto_relevant"], ranked["top_100"] = pareto, ranked.lexicographic_rank <= 100
    maps = {"alpha_decay": {v: i for i, v in enumerate(c.ALPHAS)}, "eta_PP": {v: i for i, v in enumerate(c.ETAS)},
            "m0": {v: i for i, v in enumerate(c.PIVOTS)}, "beta_low": {v: i for i, v in enumerate(c.BETAS)},
            "kappa_dyn": {v: i for i, v in enumerate(c.KAPPAS)}}
    accepted, region = [], np.zeros(len(ranked), int)
    for row_index, row in ranked.iterrows():
        position = np.asarray([maps[name][row[name]] for name in maps], float)
        if not accepted or min(np.linalg.norm(position - old) for old in accepted) >= 1.75:
            accepted.append(position); region[row_index] = len(accepted)
            if len(accepted) == 20: break
    ranked["distinct_region_rank"] = region
    return ranked


def freeze_pool(ranked: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    reasons, ids = defaultdict(list), []
    def add(config_id: str, reason: str) -> None:
        reasons[config_id].append(reason)
        if config_id not in ids: ids.append(config_id)
    add(ranked.iloc[0].config_id, "SEARCH lexicographic best")
    for row in ranked[ranked.distinct_region_rank > 0].sort_values("distinct_region_rank").head(5).itertuples():
        add(row.config_id, f"SEARCH distinct region {row.distinct_region_rank}")
    for row in grid[grid.is_previous_pilot_reference].sort_values(["reference_distance_grid_L1", "kappa_dyn", "beta_low"]).itertuples():
        add(row.config_id, "required previous-pilot reference")
    for row in ranked[ranked.pareto_relevant].head(2).itertuples(): add(row.config_id, "SEARCH Pareto-relevant")
    selected = grid.set_index("config_id").loc[ids].reset_index()
    selected["freeze_order"] = np.arange(1, len(selected) + 1)
    selected["freeze_reason"] = selected.config_id.map(lambda item: "; ".join(reasons[item]))
    selected["heldout_frozen_before_scoring"] = True
    return selected


def exactness(search_ids: Sequence[str], paths: dict[tuple[str, str], Path], cache: Path) -> pd.DataFrame:
    piece_id, rows = sorted(search_ids)[0], []
    for system in ("ORIGINAL_PT", "HUMAN", "ALWAYS_ON"):
        components, _, _ = pilot.evaluate_midi_components(paths[(piece_id, system)])
        stats = c.load_stats(cache / f"{piece_id}__{system}.npz")
        for beta, kappa in ((0., 0.), (.5, .2), (1., .5)):
            naive = pilot.score_components(components, kappa, beta)[0]["H_piece"]
            optimized = c.score_stats(stats, 1., .9, 48, beta, kappa)
            difference = abs(naive - optimized)
            rows.append({"audit_type": "real_MIDI_vs_audited_pilot", "piece_id": piece_id, "system": system,
                         "eta_PP": .9, "beta_low": beta, "kappa_dyn": kappa, "naive": naive,
                         "optimized": optimized, "absolute_difference": difference,
                         "status": "PASS" if difference <= 1e-12 else "FAIL"})
    rng = np.random.default_rng(c.SPLIT_SEED)
    for trial in range(20):
        pa, pp = rng.normal(size=rng.integers(0, 12)), rng.normal(size=rng.integers(1, 15))
        pal, ppl = rng.uniform(0, 1, len(pa)), rng.uniform(0, 1, len(pp))
        eta, beta = float(rng.choice(c.ETAS)), float(rng.choice(c.BETAS))
        naive = (pa * (1 + beta * pal)).sum() + eta * (pp * (1 + beta * ppl)).sum()
        optimized = pa.sum() + beta * (pa * pal).sum() + eta * (pp.sum() + beta * (pp * ppl).sum())
        difference = abs(naive - optimized)
        rows.append({"audit_type": "synthetic_pair_sum", "piece_id": f"synthetic_{trial}", "system": "SYNTHETIC",
                     "eta_PP": eta, "beta_low": beta, "kappa_dyn": np.nan, "naive": naive,
                     "optimized": optimized, "absolute_difference": difference,
                     "status": "PASS" if difference <= 1e-12 else "FAIL"})
    frame = pd.DataFrame(rows)
    if not frame.status.eq("PASS").all(): raise AssertionError("optimized != naive")
    return frame


def mechanism(ranked: pd.DataFrame, scores: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    parameters, rows = ["alpha_decay", "eta_PP", "m0", "beta_low", "kappa_dyn"], []
    for parameter in parameters:
        for value, group in ranked.groupby(parameter):
            rows.append({"scope": "ordering", "parameter": parameter, "value": value, "system": "ALL",
                         "mean_canonical_chain_pass_rate": group.canonical_chain_pass_rate.mean(),
                         "max_canonical_chain_pass_rate": group.canonical_chain_pass_rate.max(),
                         "mean_all_model_chain_pass_rate": group.all_model_chain_pass_rate.mean(),
                         "mean_individual_model_compliance": group.individual_model_compliance.mean(), "mean_H_piece": np.nan})
    merged = scores.merge(grid[["config_id"] + parameters], on="config_id")
    for parameter in parameters:
        for row in merged.groupby([parameter, "system"], as_index=False).H_piece.mean().itertuples():
            rows.append({"scope": "system_score", "parameter": parameter, "value": getattr(row, parameter),
                         "system": row.system, "mean_canonical_chain_pass_rate": np.nan,
                         "max_canonical_chain_pass_rate": np.nan, "mean_all_model_chain_pass_rate": np.nan,
                         "mean_individual_model_compliance": np.nan, "mean_H_piece": row.H_piece})
    return pd.DataFrame(rows)


def distribution(scores: pd.DataFrame, configs: pd.DataFrame, role: str) -> pd.DataFrame:
    rows = []
    for (config_id, system), group in scores[scores.config_id.isin(configs.config_id)].groupby(["config_id", "system"]):
        values = group.H_piece.to_numpy()
        rows.append({"config_id": config_id, "subset_role": role, "system": system, "piece_count": len(values),
                     "mean_H_piece": values.mean(), "median_H_piece": np.median(values),
                     "IQR_H_piece": np.quantile(values, .75) - np.quantile(values, .25)})
    return pd.DataFrame(rows)


def md_table(frame: pd.DataFrame, columns: Sequence[str], limit: int | None = None) -> list[str]:
    shown = frame.loc[:, list(columns)].head(limit) if limit else frame.loc[:, list(columns)]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in shown.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)): values.append("NA" if math.isnan(float(value)) else f"{float(value):.6f}")
            else: values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_plots(work: Path, ranked: pd.DataFrame, mechanism_frame: pd.DataFrame,
                candidate_scores: pd.DataFrame, candidates: pd.DataFrame,
                comparison: pd.DataFrame, per_piece: pd.DataFrame) -> None:
    plot_dir = work / "plots"; plot_dir.mkdir(exist_ok=True)
    best = candidates.iloc[0].config_id
    data = candidate_scores[candidate_scores.config_id == best]
    fig, ax = plt.subplots(figsize=(14, 6))
    systems = list(c.SYSTEMS); values = [data[data.system == system].H_piece for system in systems]
    ax.boxplot(values, tick_labels=systems, showmeans=True); ax.axhline(0, color="black", linewidth=.7)
    ax.tick_params(axis="x", rotation=45); ax.set_ylabel("H_piece"); ax.set_title("Frozen top candidate system distributions")
    fig.tight_layout(); fig.savefig(plot_dir / "top_config_system_distributions.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8)); axes = axes.ravel()
    for axis, parameter in zip(axes, ["alpha_decay", "eta_PP", "m0", "beta_low", "kappa_dyn"]):
        frame = mechanism_frame[(mechanism_frame.scope == "ordering") & (mechanism_frame.parameter == parameter)]
        axis.plot(frame.value, frame.mean_canonical_chain_pass_rate, marker="o", label="canonical")
        axis.plot(frame.value, frame.mean_all_model_chain_pass_rate, marker="s", label="all")
        axis.set_title(parameter); axis.set_ylim(0, 1); axis.legend()
    axes[-1].axis("off"); fig.tight_layout(); fig.savefig(plot_dir / "parameter_partial_dependence.png", dpi=180); plt.close(fig)

    heat = ranked.groupby(["alpha_decay", "eta_PP"]).canonical_chain_pass_rate.mean().unstack()
    fig, ax = plt.subplots(figsize=(8, 6)); image = ax.imshow(heat, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(heat.columns)), heat.columns); ax.set_yticks(range(len(heat.index)), heat.index)
    ax.set_xlabel("eta_PP"); ax.set_ylabel("alpha_decay"); fig.colorbar(image, ax=ax, label="mean chain pass")
    fig.tight_layout(); fig.savefig(plot_dir / "chain_pass_rate_alpha_eta_heatmap.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(ranked.reference_distance_grid_L1, ranked.canonical_chain_pass_rate,
               c=ranked.all_model_chain_pass_rate, cmap="plasma", alpha=.6)
    ax.set_xlabel("reference distance (grid L1)"); ax.set_ylabel("canonical chain pass rate")
    fig.tight_layout(); fig.savefig(plot_dir / "parameter_reference_distance.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6)); x = np.arange(len(comparison)); width = .35
    ax.bar(x-width/2, comparison.search_canonical_chain_pass_rate, width, label="search")
    ax.bar(x+width/2, comparison.heldout_canonical_chain_pass_rate, width, label="held-out")
    ax.set_xticks(x, comparison.config_id, rotation=45, ha="right"); ax.set_ylim(0, 1); ax.legend()
    fig.tight_layout(); fig.savefig(plot_dir / "search_vs_heldout.png", dpi=180); plt.close(fig)

    chosen_piece = per_piece[per_piece.config_id == best].sort_values("piece_id")
    fig, ax = plt.subplots(figsize=(12, 6))
    for label in ["extreme_upper_score", "M_canonical", "ORIGINAL_PT", "HUMAN"]:
        ax.plot(range(len(chosen_piece)), chosen_piece[label], marker="o", label=label)
    ax.set_xticks(range(len(chosen_piece)), chosen_piece.piece_id, rotation=45, ha="right"); ax.legend()
    fig.tight_layout(); fig.savefig(plot_dir / "target_ordering_visualization.png", dpi=180); plt.close(fig)


def report(work: Path, audit: pd.DataFrame, split: pd.DataFrame, ranked: pd.DataFrame,
           heldout: pd.DataFrame, comparison: pd.DataFrame, mechanism_frame: pd.DataFrame,
           candidates: pd.DataFrame, distributions: pd.DataFrame, exact_frame: pd.DataFrame,
           threshold_frame: pd.DataFrame, refinement_performed: bool) -> None:
    eligible = audit[audit.primary_status == "PEDAL_ELIGIBLE"]
    sparse = audit[audit.primary_status == "PEDAL_SPARSE"]
    deferred = eligible[eligible.exact_compute_status == "DEFERRED_LONG_FORM"]
    best, best_held = ranked.iloc[0], heldout.sort_values(LEX, ascending=[False]*len(LEX)).iloc[0]
    mechanism_order = mechanism_frame[mechanism_frame.scope == "ordering"]
    importance = (mechanism_order.groupby("parameter").mean_canonical_chain_pass_rate.agg(lambda x: x.max()-x.min()).sort_values(ascending=False))
    target_exists = best.canonical_chain_pass_rate >= .75 and best.all_model_chain_pass_rate >= .75
    held_maintained = best_held.canonical_chain_pass_rate >= .75 and best_held.all_model_chain_pass_rate >= .75
    columns = ["alpha_decay", "eta_PP", "m0", "beta_low", "kappa_dyn", "canonical_chain_pass_rate",
               "all_model_chain_pass_rate", "individual_model_compliance", "human_gt_pt_rate", "pt_gt_model_rate",
               "model_gt_no_pedal_rate", "model_gt_always_rate", "median_gap_extreme_model",
               "median_gap_model_pt", "median_gap_pt_human"]
    lines = [
        "# Harmonic Metric Broad Parameter Search v0", "", "## Scope and integrity", "",
        "`Metric_harmonic_consonance.md` was the sole formula source. The audited pilot parser, note-instance semantics, Hall weights, and extreme construction were reused. ASAP test access, training, and new inference were all zero.", "",
        f"The full 4,320-point coarse grid was evaluated on the exact-computable PARAMETER_SEARCH pieces. The split was fixed before harmonic scoring with rule `{c.SPLIT_RULE}` and seed `{c.SPLIT_SEED}`. Held-out scores were computed only after {len(pd.read_csv(work / 'frozen_before_heldout.csv'))} configurations had been frozen from SEARCH; the final listening shortlist contains {len(candidates)} of those pre-frozen configurations.", "",
        f"Exactness audit: {len(exact_frame)}/{len(exact_frame)} PASS; maximum optimized-vs-naive absolute difference `{exact_frame.absolute_difference.max():.3e}`.", "",
        "## Table 1 — PEDAL_ELIGIBLE / PEDAL_SPARSE", "",
    ]
    table1 = audit.copy(); table1["piece"] = table1.composer + "/" + table1.title
    lines += md_table(table1, ["piece_id", "piece", "human_valid_onset_count", "human_valid_onset_fraction",
                               "CC64_ON_duty_ratio", "sustain_ON_episode_count", "primary_status", "primary_reason",
                               "exact_compute_status"])
    lines += ["", f"The 0.05 thresholds are calibration eligibility rules, not a musical definition. Primary: {len(eligible)} eligible, {len(sparse)} sparse. {len(deferred)} eligible long-form pieces were deferred by the pre-score exact-work limit of {c.MAX_EXACT_ALWAYS_PAIR_ONSET_WORK:,} pair-onsets; their IDs and workloads remain in the table and are not silently treated as failures.", "",
              "## Table 2 — Search / held-out split", ""]
    lines += md_table(split, ["piece_id", "composer", "title", "split_role", "exact_compute_status",
                              "low_note_fraction", "velocity_range", "note_density_per_second",
                              "human_valid_onset_fraction", "CC64_ON_duty_ratio"])
    backbone_id = ranked[ranked.is_hall_decay_backbone].iloc[0].config_id
    reference_distribution = distributions[(distributions.config_id == backbone_id) & (distributions.subset_role == "PARAMETER_SEARCH")]
    lines += ["", "## Table 3 — Hall+decay backbone system distribution (SEARCH)", ""]
    lines += md_table(reference_distribution, ["system", "piece_count", "mean_H_piece", "median_H_piece", "IQR_H_piece"])
    lines += ["", "## Table 4 — SEARCH top 20 configurations", ""]
    lines += md_table(ranked, columns, 20)
    lines += ["", "## Table 5 — Frozen configurations on HELD_OUT_CHECK", ""]
    lines += md_table(heldout.sort_values(LEX, ascending=[False]*len(LEX)), ["config_id"] + columns)
    lines += ["", "## Table 6 — Search vs held-out degradation", ""]
    lines += md_table(comparison, ["config_id", "search_canonical_chain_pass_rate", "heldout_canonical_chain_pass_rate",
                                   "canonical_chain_degradation", "search_all_model_chain_pass_rate",
                                   "heldout_all_model_chain_pass_rate", "all_chain_degradation"])
    effect = mechanism_order.groupby("parameter", as_index=False).agg(
        min_mean_chain=("mean_canonical_chain_pass_rate", "min"), max_mean_chain=("mean_canonical_chain_pass_rate", "max"),
        min_all_chain=("mean_all_model_chain_pass_rate", "min"), max_all_chain=("mean_all_model_chain_pass_rate", "max"))
    effect["canonical_effect_range"] = effect.max_mean_chain-effect.min_mean_chain
    lines += ["", "## Table 7 — Parameter-family effect summary", ""] + md_table(effect, effect.columns)
    lines += ["", "## Table 8 — Candidate system distributions", ""]
    lines += md_table(distributions[distributions.config_id.isin(candidates.config_id)],
                      ["config_id", "subset_role", "system", "piece_count", "mean_H_piece", "median_H_piece", "IQR_H_piece"])
    lines += ["", "## Mechanism and failure analysis", "",
              f"Partial-dependence ordering sensitivity is largest for `{importance.index[0]}` (mean canonical-chain range {importance.iloc[0]:.3f}); full data are in `parameter_mechanism_summary.csv`. eta_PP and alpha_decay system-specific movements, including ALWAYS_ON, are recorded there rather than inferred from a single optimum.", "",
              f"Optional refinement was {'performed' if refinement_performed else 'not performed'}; the predeclared trigger required both SEARCH chain rates >=0.75 and individual compliance >=0.75. This prevents local search when the formula family does not show a robust coarse capability signal.", "",
              "PEDAL_SPARSE threshold sensitivity for frozen candidates is in `pedal_sparse_sensitivity.csv`; coverage columns expose deferred long-form pieces at every threshold.", "",
              "## Final candidate configurations (not final parameters)", ""]
    lines += md_table(candidates, ["config_id", "candidate_type", "alpha_decay", "eta_PP", "m0", "beta_low", "kappa_dyn",
                                   "search_canonical_chain_pass_rate", "heldout_canonical_chain_pass_rate",
                                   "reference_distance_grid_L1", "selection_reason"])
    lines += ["", "## Answers to the research questions", "",
              f"**Q1. Formula-family capability.** {'A robust target region exists on SEARCH.' if target_exists else 'No robust full-chain region was found on SEARCH under the predeclared 0.75 criterion.'} Best canonical/all chain rates were {best.canonical_chain_pass_rate:.3f}/{best.all_model_chain_pass_rate:.3f}.", "",
              f"**Q2. Held-out maintenance.** {'The frozen signal was maintained.' if held_maintained else 'The target ordering was not robustly maintained on held-out pieces.'} Best frozen held-out canonical/all rates were {best_held.canonical_chain_pass_rate:.3f}/{best_held.all_model_chain_pass_rate:.3f}.", "",
              f"**Q3. Important parameters.** `{importance.index[0]}` had the largest marginal ordering-rate range; eta_PP and alpha_decay effects are reported system-by-system in Table 7 and the mechanism CSV.", "",
              f"**Q4. Reference distance.** The SEARCH best is {int(best.reference_distance_grid_L1)} coarse grid steps from the Hall+decay backbone; reference rows were evaluated alongside search candidates, not used as a primary objective.", "",
              "**Q5. Structural failure.** When the chain fails, the per-piece gap columns identify whether extreme<model, model<PT, or PT<HUMAN breaks. No formula term was changed and the space was not widened after held-out inspection.", "",
              f"**Q6. Listening candidates.** {len(candidates)} frozen configurations are listed above. They cover reference-near, strongest SEARCH, conservative/held-out-stable, and a distinct region where available; none is declared final.", "",
              "## Files, provenance, and caveats", "",
              "- MODEL_CANONICAL contains the four strict canonical-stream Stage 2 models; MODEL_ALL adds CUSTOM_EVENT_V0.",
              "- HUMAN and CUSTOM_EVENT_V0 are not exact canonical pedal-only comparisons.",
              "- Reused extreme source/generated SHA and strict identity are in `reusable_extremes_manifest.csv`.",
              "- The Lehtonen pitch-anchor interpolation remains the research v0 adaptation; alpha_decay only scales its T60 values.",
              "- All input paths and SHA values are in `input_midis.csv`; no TEST path was accessed.", ""]
    (work / "HARMONIC_METRIC_BROAD_PARAMETER_SEARCH_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=c.OUTPUT)
    args = parser.parse_args(); output = args.output.resolve(); work = output.parent / f".{output.name}.work"
    if output.exists(): raise FileExistsError(f"refusing to overwrite existing result: {output}")
    work.mkdir(parents=True, exist_ok=True)

    cases, _ = c.build_cases(); audit = c.audit_pieces(cases); audit.to_csv(work / "pedal_activity_audit.csv", index=False)
    eligible = audit[audit.primary_status == "PEDAL_ELIGIBLE"].copy(); eligible.to_csv(work / "eligible_pieces.csv", index=False)
    planned_split = c.split_eligible(audit).rename(columns={"split_role": "planned_split_role"})
    exact_split = c.split_eligible(
        audit[audit.exact_compute_status == "EXACT_COMPUTABLE"].copy()
    )[["piece_id", "split_role"]].rename(columns={"split_role": "exact_analysis_split_role"})
    split = planned_split.merge(exact_split, on="piece_id", how="left")
    split["split_role"] = split.exact_analysis_split_role.fillna("DEFERRED_LONG_FORM")
    split.to_csv(work / "split_manifest.csv", index=False)
    computed = set(split[split.included_in_exact_analysis].piece_id)
    threshold_extra = set(audit[(audit["eligible_threshold_0.02"]) & (audit.exact_compute_status == "EXACT_COMPUTABLE")].piece_id) - computed
    all_needed = computed | threshold_extra
    extremes, extreme_manifest = c.ensure_extremes(all_needed, audit, work)
    extreme_manifest.to_csv(work / "reusable_extremes_manifest.csv", index=False)
    paths, inputs = c.resolve_paths(cases, all_needed, extremes); inputs.to_csv(work / "input_midis.csv", index=False)
    identity_cases = [case for case in cases if case["piece_id"] in all_needed]
    identity = pd.DataFrame(pilot.identity_audit(identity_cases, paths, {row["piece_id"]: row for row in c.read_csv(c.STAGE1_MANIFEST)}))
    identity.to_csv(work / "midi_identity_audit.csv", index=False)

    search_ids = split[(split.split_role == "PARAMETER_SEARCH") & split.included_in_exact_analysis].piece_id.tolist()
    heldout_ids = split[(split.split_role == "HELD_OUT_CHECK") & split.included_in_exact_analysis].piece_id.tolist()
    grid = c.parameter_grid(); grid.to_csv(work / "coarse_parameter_grid.csv", index=False)
    search_cache = work / "sufficient_statistics/search"
    c.compute_caches(search_ids, paths, search_cache, c.ALPHAS, c.PIVOTS).to_csv(work / "search_cache_manifest.csv", index=False)
    exact_frame = exactness(search_ids, paths, search_cache); exact_frame.to_csv(work / "sufficient_statistics_exactness_audit.csv", index=False)

    search_scores = c.score_search(search_ids, search_cache, grid, audit)
    search_results, search_piece = ordering_metrics(search_scores, grid, "PARAMETER_SEARCH")
    ranked = rank_configs(search_results); ranked.to_csv(work / "coarse_search_results.csv", index=False)
    selected_top = ranked[ranked.top_100 | ranked.pareto_relevant | (ranked.distinct_region_rank > 0)]
    selected_top.to_csv(work / "top_coarse_configs.csv", index=False)

    refinement_trigger = bool(ranked.iloc[0].canonical_chain_pass_rate >= .75 and
                              ranked.iloc[0].all_model_chain_pass_rate >= .75 and
                              ranked.iloc[0].individual_model_compliance >= .75)
    refinement_columns = list(grid.columns) + ["status", "reason"]
    pd.DataFrame(columns=refinement_columns).to_csv(work / "refinement_results.csv", index=False)
    if refinement_trigger:
        # Refinement is deliberately not silently approximated. A coarse success
        # would require a separate exact alpha/m0 cache pass; this run records the
        # trigger and freezes coarse candidates instead of inspecting held-out.
        (work / "refinement_triggered_not_executed.txt").write_text(
            "Coarse trigger passed; exact local alpha/m0 expansion requires a follow-up run.\n", encoding="utf-8")

    frozen = freeze_pool(ranked, grid); frozen.to_csv(work / "frozen_before_heldout.csv", index=False)
    held_cache = work / "sufficient_statistics/heldout"
    alphas, pivots = sorted(set(frozen.alpha_decay)), sorted(set(frozen.m0.astype(int)))
    c.compute_caches(heldout_ids, paths, held_cache, alphas, pivots).to_csv(work / "heldout_cache_manifest.csv", index=False)
    held_scores = selected_scores(heldout_ids, held_cache, frozen, audit)
    held_results, held_piece = ordering_metrics(held_scores, frozen, "HELD_OUT_CHECK")
    held_results.to_csv(work / "heldout_results.csv", index=False)

    search_frozen_scores = search_scores[search_scores.config_id.isin(frozen.config_id)]
    search_frozen = ranked[ranked.config_id.isin(frozen.config_id)]
    compare = search_frozen[["config_id", "canonical_chain_pass_rate", "all_model_chain_pass_rate"]].merge(
        held_results[["config_id", "canonical_chain_pass_rate", "all_model_chain_pass_rate"]], on="config_id", suffixes=("_search", "_heldout"))
    compare = compare.rename(columns={"canonical_chain_pass_rate_search": "search_canonical_chain_pass_rate",
                                      "canonical_chain_pass_rate_heldout": "heldout_canonical_chain_pass_rate",
                                      "all_model_chain_pass_rate_search": "search_all_model_chain_pass_rate",
                                      "all_model_chain_pass_rate_heldout": "heldout_all_model_chain_pass_rate"})
    compare["canonical_chain_degradation"] = compare.search_canonical_chain_pass_rate-compare.heldout_canonical_chain_pass_rate
    compare["all_chain_degradation"] = compare.search_all_model_chain_pass_rate-compare.heldout_all_model_chain_pass_rate
    compare.to_csv(work / "search_vs_heldout_degradation.csv", index=False)

    held_ranked = held_results.sort_values(LEX + ["reference_distance_grid_L1"], ascending=[False]*len(LEX)+[True])
    candidate_ids, reasons = [], {}
    def add_candidate(config_id: str, kind: str, reason: str) -> None:
        if config_id not in candidate_ids: candidate_ids.append(config_id); reasons[config_id] = (kind, reason)
    add_candidate(ranked.iloc[0].config_id, "strongest_SEARCH", "lexicographic SEARCH best")
    reference_near = ranked[(ranked.canonical_chain_pass_rate == ranked.iloc[0].canonical_chain_pass_rate) &
                            (ranked.all_model_chain_pass_rate == ranked.iloc[0].all_model_chain_pass_rate)].sort_values("reference_distance_grid_L1").iloc[0]
    add_candidate(reference_near.config_id, "reference_near", "closest reference among best chain-rate plateau")
    add_candidate(held_ranked.iloc[0].config_id, "heldout_stable_frozen", "best held-out behavior within pre-frozen pool")
    for row in ranked[ranked.distinct_region_rank > 1].sort_values("distinct_region_rank").itertuples():
        add_candidate(row.config_id, "distinct_region", "qualitatively distinct pre-frozen coarse region")
        if len(candidate_ids) >= 4: break
    backbone = grid[grid.is_hall_decay_backbone].iloc[0]
    add_candidate(backbone.config_id, "literature_backbone", "Hall+decay reference anchor")
    candidate_ids = candidate_ids[:5]
    candidates = frozen.set_index("config_id").loc[candidate_ids].reset_index()
    candidates = candidates.merge(compare, on="config_id", how="left")
    candidates["candidate_type"] = candidates.config_id.map(lambda item: reasons[item][0])
    candidates["selection_reason"] = candidates.config_id.map(lambda item: reasons[item][1])
    candidates.to_csv(work / "final_candidate_configs.csv", index=False)

    top_scores = pd.concat([search_frozen_scores, held_scores], ignore_index=True)
    top_scores.to_csv(work / "per_system_scores_top_configs.csv", index=False)
    per_piece = pd.concat([search_piece[search_piece.config_id.isin(frozen.config_id)], held_piece], ignore_index=True)
    per_piece.to_csv(work / "per_piece_ordering.csv", index=False)
    mechanism_frame = mechanism(ranked, search_scores, grid); mechanism_frame.to_csv(work / "parameter_mechanism_summary.csv", index=False)
    distributions = pd.concat([distribution(search_frozen_scores, frozen, "PARAMETER_SEARCH"),
                               distribution(held_scores, frozen, "HELD_OUT_CHECK")], ignore_index=True)
    distributions.to_csv(work / "candidate_system_distributions.csv", index=False)

    extra_scores = []
    if threshold_extra:
        extra_cache = work / "sufficient_statistics/threshold_extra"
        c.compute_caches(sorted(threshold_extra), paths, extra_cache, alphas, pivots)
        extra_scores.append(selected_scores(sorted(threshold_extra), extra_cache, frozen, audit))
    all_frozen_scores = pd.concat([search_frozen_scores, held_scores] + extra_scores, ignore_index=True)
    threshold_rows = []
    for threshold in c.SENSITIVITY_THRESHOLDS:
        eligible_ids = set(audit[audit[f"eligible_threshold_{threshold:.2f}"]].piece_id)
        exact_ids = eligible_ids & set(all_frozen_scores.piece_id)
        subset = all_frozen_scores[all_frozen_scores.piece_id.isin(exact_ids)]
        metrics, _ = ordering_metrics(subset, frozen, f"THRESHOLD_{threshold:.2f}")
        metrics["threshold"] = threshold; metrics["eligible_piece_count_total"] = len(eligible_ids)
        metrics["eligible_piece_count_exact_scored"] = len(exact_ids)
        threshold_rows.append(metrics)
    threshold_frame = pd.concat(threshold_rows, ignore_index=True)
    threshold_frame.to_csv(work / "pedal_sparse_sensitivity.csv", index=False)

    write_plots(work, ranked, mechanism_frame, top_scores, candidates, compare, per_piece)
    report(work, audit, split, ranked, held_results, compare, mechanism_frame, candidates,
           distributions, exact_frame, threshold_frame, False)
    provenance = {
        "script_version": c.SCRIPT_VERSION, "metric_source_of_truth": str(c.METRIC_DOCUMENT),
        "metric_document_sha256": sha256_file(c.METRIC_DOCUMENT), "coarse_configuration_count": len(grid),
        "split_fixed_before_scoring": True, "heldout_used_for_parameter_generation": False,
        "heldout_candidates_frozen_before_scoring": frozen.config_id.tolist(),
        "test_set_access_count": 0, "new_inference_count": 0, "training_steps": 0,
        "exact_work_limit": c.MAX_EXACT_ALWAYS_PAIR_ONSET_WORK, "refinement_trigger": refinement_trigger,
    }
    (work / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

    required = ["pedal_activity_audit.csv", "eligible_pieces.csv", "split_manifest.csv", "coarse_parameter_grid.csv",
                "coarse_search_results.csv", "top_coarse_configs.csv", "per_piece_ordering.csv",
                "per_system_scores_top_configs.csv", "parameter_mechanism_summary.csv", "heldout_results.csv",
                "refinement_results.csv", "final_candidate_configs.csv", "HARMONIC_METRIC_BROAD_PARAMETER_SEARCH_REPORT.md"]
    missing = [name for name in required if not (work / name).is_file()]
    if missing: raise AssertionError(f"missing outputs: {missing}")
    for text_path in work.rglob("*"):
        if text_path.is_file() and text_path.suffix.lower() in {".csv", ".json", ".md", ".txt"}:
            payload = text_path.read_text(encoding="utf-8")
            text_path.write_text(payload.replace(str(work), str(output)), encoding="utf-8")
    os.rename(work, output)
    print(json.dumps({"status": "completed", "output": str(output), "eligible": len(eligible),
                      "search_exact": len(search_ids), "heldout_exact": len(heldout_ids),
                      "best_search": ranked.iloc[0].config_id, "candidates": candidate_ids}, indent=2))


if __name__ == "__main__":
    main()
