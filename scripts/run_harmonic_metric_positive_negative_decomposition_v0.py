#!/usr/bin/env python3
"""Validation-only positive/negative decomposition of the harmonic metric."""

from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import scripts.harmonic_pos_neg_decomposition_core as core
import scripts.search_harmonic_metric_broad_parameters_v0 as broad
from src.stage2_binary.canonical_stage1 import sha256_file


ROOT = Path(__file__).resolve().parents[1]
BROAD = ROOT / "analysis/harmonic_metric_broad_parameter_search_v0"
DEFAULT_OUTPUT = ROOT / "analysis/harmonic_metric_positive_negative_decomposition_v0"
SCRIPT_VERSION = "harmonic_metric_positive_negative_decomposition_v0.1"
REFERENCE_CONFIG_ID = "coarse_3ba56a105882f2d9"
MAX_WORKERS = 2
SYSTEM_ORDER = (
    "NO_PEDAL", "ALWAYS_ON", "STANDARD_CE_ARGMAX",
    "STANDARD_CE_POSTERIOR_MEDIAN", "WEIGHTED_CE_ARGMAX",
    "HYBRID_REGRESSION_ONLY", "CUSTOM_EVENT_V0", "ORIGINAL_PT", "HUMAN",
)


def _worker(task: tuple[str, str, str]) -> dict[str, Any]:
    piece_id, system, midi_path = task
    rows, packets, diagnostics = core.evaluate_midi(Path(midi_path))
    return {
        "piece_id": piece_id,
        "system": system,
        "onsets": rows,
        "packets": packets,
        "diagnostics": diagnostics,
    }


def _system_group(system: str) -> str:
    if system in broad.MODEL_CANONICAL:
        return "MODEL_CANONICAL"
    if system == "CUSTOM_EVENT_V0":
        return "MODEL_ALL_NONCANONICAL_ADDITION"
    if system in ("NO_PEDAL", "ALWAYS_ON"):
        return "EXTREME"
    return system


def _mean(frame: pd.DataFrame, column: str) -> float:
    return float(frame[column].mean()) if len(frame) else 0.0


def aggregate_pieces(onsets: pd.DataFrame, inputs: pd.DataFrame, split: pd.DataFrame) -> pd.DataFrame:
    metadata = inputs.set_index(["piece_id", "system"])
    split_map = split.set_index("piece_id").split_role.to_dict()
    rows = []
    for (piece_id, system), group in onsets.groupby(["piece_id", "system"], sort=False):
        valid = group[group.valid_onset]
        negative = valid[valid.Z_neg_n > 0]
        h_pos, h_neg, h_net = (_mean(valid, name) for name in ("H_pos_n", "H_neg_all_n", "H_net_n"))
        total = h_pos + h_neg
        cancellation_ratio = 0.0 if total == 0 else float(np.clip(1.0 - abs(h_net) / (total + core.EPSILON), 0.0, 1.0))
        h_neg_pa, h_neg_pp = _mean(valid, "H_neg_PA_n"), _mean(valid, "H_neg_PP_n")
        source = metadata.loc[(piece_id, system)]
        rows.append(
            {
                "piece_id": piece_id,
                "performance_id": source.performance_id,
                "split_role_metadata": split_map[piece_id],
                "system": system,
                "system_group": _system_group(system),
                "is_MODEL_CANONICAL": system in broad.MODEL_CANONICAL,
                "is_MODEL_ALL": system in broad.MODEL_ALL,
                "midi_path": source.midi_path,
                "source_sha256": source.sha256,
                "all_distinct_onset_count": len(group),
                "valid_onset_count": len(valid),
                "H_pos_valid": h_pos,
                "H_neg_valid": h_neg,
                "H_net_valid": h_net,
                "F_neg_valid": _mean(valid, "F_neg_n"),
                "H_neg_cond_valid": _mean(negative, "H_neg_cond_n"),
                "cancellation_valid": _mean(valid, "cancellation_mass_n"),
                "H_neg_PA_valid": h_neg_pa,
                "H_neg_PP_valid": h_neg_pp,
                "H_neg_PA_fraction": h_neg_pa / (h_neg_pa + h_neg_pp + core.EPSILON),
                "H_pos_all_onsets": _mean(group, "H_pos_n"),
                "H_neg_all_onsets": _mean(group, "H_neg_all_n"),
                "H_net_all_onsets": _mean(group, "H_net_n"),
                "negative_onset_rate": float((group.Z_neg_n > 0).mean()),
                "positive_plus_negative_valid": total,
                "abs_H_net_valid": abs(h_net),
                "cancellation_ratio_piece": cancellation_ratio,
                "PA_pair_count": int(group.PA_pair_count.sum()),
                "PP_pair_count": int(group.PP_pair_count.sum()),
                "negative_PA_pair_count": int(group.negative_PA_pair_count.sum()),
                "negative_PP_pair_count": int(group.negative_PP_pair_count.sum()),
                "max_onset_reconstruction_abs_error": float(group.onset_reconstruction_abs_error.max()),
                "AA_pair_count": int(group.AA_pair_count.sum()),
            }
        )
    return pd.DataFrame(rows)


def system_summary(piece: pd.DataFrame) -> pd.DataFrame:
    metrics = ("H_pos_valid", "H_neg_valid", "H_net_valid")
    rows = []
    for system, group in piece.groupby("system"):
        row: dict[str, Any] = {"system": system, "piece_count": len(group)}
        for metric in metrics:
            values = group[metric].to_numpy(float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_median"] = float(np.median(values))
            row[f"{metric}_IQR"] = float(np.quantile(values, .75) - np.quantile(values, .25))
        for metric in ("F_neg_valid", "H_neg_cond_valid", "negative_onset_rate"):
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_median"] = float(group[metric].median())
        for metric in ("cancellation_valid", "cancellation_ratio_piece", "H_neg_PA_valid", "H_neg_PP_valid"):
            row[f"{metric}_mean"] = float(group[metric].mean())
        denom = row["H_neg_PA_valid_mean"] + row["H_neg_PP_valid_mean"] + core.EPSILON
        row["H_neg_PA_fraction"] = row["H_neg_PA_valid_mean"] / denom
        row["H_neg_PP_fraction"] = row["H_neg_PP_valid_mean"] / denom
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame["system_order"] = frame.system.map({name: i for i, name in enumerate(SYSTEM_ORDER)})
    return frame.sort_values("system_order").drop(columns="system_order").reset_index(drop=True)


def ordering_analysis(piece: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    net = piece.pivot(index="piece_id", columns="system", values="H_net_valid")
    neg = piece.pivot(index="piece_id", columns="system", values="H_neg_valid")
    rows, details = [], []

    def record(comparison: str, target: str, net_pass: pd.Series, neg_pass: pd.Series) -> None:
        rows.append(
            {
                "comparison": comparison,
                "target_system_or_group": target,
                "piece_or_cell_count": len(net_pass),
                "net_desired_order_pass_count": int(net_pass.sum()),
                "net_desired_order_rate": float(net_pass.mean()),
                "negative_only_desired_order_pass_count": int(neg_pass.sum()),
                "negative_only_desired_order_rate": float(neg_pass.mean()),
                "negative_minus_net_rate_change": float(neg_pass.mean() - net_pass.mean()),
            }
        )
        for key in net_pass.index:
            details.append(
                {
                    "comparison": comparison,
                    "target_system_or_group": target,
                    "piece_or_cell_id": key if isinstance(key, str) else "|".join(map(str, key)),
                    "net_desired_order_pass": bool(net_pass.loc[key]),
                    "negative_only_desired_order_pass": bool(neg_pass.loc[key]),
                }
            )

    record("HUMAN_vs_ORIGINAL_PT", "ORIGINAL_PT", net.HUMAN > net.ORIGINAL_PT, neg.HUMAN < neg.ORIGINAL_PT)
    net_pt_cells, neg_pt_cells, net_always_cells, neg_always_cells = [], [], [], []
    for model in broad.MODEL_CANONICAL:
        record("ORIGINAL_PT_vs_STAGE2", model, net.ORIGINAL_PT > net[model], neg.ORIGINAL_PT < neg[model])
        record("STAGE2_vs_ALWAYS_ON", model, net[model] > net.ALWAYS_ON, neg[model] < neg.ALWAYS_ON)
        for piece_id in net.index:
            net_pt_cells.append(((piece_id, model), net.loc[piece_id, "ORIGINAL_PT"] > net.loc[piece_id, model]))
            neg_pt_cells.append(((piece_id, model), neg.loc[piece_id, "ORIGINAL_PT"] < neg.loc[piece_id, model]))
            net_always_cells.append(((piece_id, model), net.loc[piece_id, model] > net.loc[piece_id, "ALWAYS_ON"]))
            neg_always_cells.append(((piece_id, model), neg.loc[piece_id, model] < neg.loc[piece_id, "ALWAYS_ON"]))
    record("ORIGINAL_PT_vs_STAGE2_MACRO", "MODEL_CANONICAL",
           pd.Series(dict(net_pt_cells)), pd.Series(dict(neg_pt_cells)))
    record("STAGE2_vs_ALWAYS_ON_MACRO", "MODEL_CANONICAL",
           pd.Series(dict(net_always_cells)), pd.Series(dict(neg_always_cells)))

    for group_name, models in (("MODEL_CANONICAL", broad.MODEL_CANONICAL), ("MODEL_ALL", broad.MODEL_ALL)):
        net_median = net[list(models)].median(axis=1)
        neg_median = neg[list(models)].median(axis=1)
        net_chain = (net.ALWAYS_ON < net_median) & (net_median < net.ORIGINAL_PT) & (net.ORIGINAL_PT < net.HUMAN)
        neg_chain = (neg.HUMAN < neg.ORIGINAL_PT) & (neg.ORIGINAL_PT < neg_median) & (neg_median < neg.ALWAYS_ON)
        record("FULL_ORDERING", group_name, net_chain, neg_chain)
    return pd.DataFrame(rows), pd.DataFrame(details)


def negative_pair_summary(results: list[dict[str, Any]], piece: pd.DataFrame) -> pd.DataFrame:
    component_lookup = piece.set_index(["piece_id", "system"])
    rows = []
    for result in results:
        for pair_type in ("PA", "PP"):
            aggregate = result["diagnostics"]["pair_aggregate"][pair_type]
            total = int(aggregate["total_pair_count"])
            negative = int(aggregate["negative_pair_count"])
            base = {
                "piece_id": result["piece_id"], "system": result["system"], "pair_type": pair_type,
                "total_pair_count": total, "eta_weighted_pair_mass": aggregate["eta_weighted_pair_mass"],
                "total_negative_pair_count": negative, "total_negative_eta_mass": aggregate["negative_eta_mass"],
                "total_negative_numerator_mass": aggregate["negative_numerator_mass"],
                "negative_pair_count_fraction": negative / total if total else 0.0,
                "mean_valid_onset_normalized_negative_burden": component_lookup.loc[(result["piece_id"], result["system"]), f"H_neg_{pair_type}_valid"],
            }
            rows.append({**base, "negative_interval_class": "ALL", "interval_negative_pair_count": negative,
                         "interval_negative_numerator_mass": aggregate["negative_numerator_mass"]})
            for interval in range(1, 13):
                count = int(aggregate["interval_counts"].get(interval, 0))
                mass = float(aggregate["interval_negative_masses"].get(interval, 0.0))
                rows.append({**base, "negative_interval_class": core.INTERVAL_NAMES[interval],
                             "interval_negative_pair_count": count, "interval_negative_numerator_mass": mass})
    return pd.DataFrame(rows)


def cancellation_diagnostics(piece: pd.DataFrame) -> pd.DataFrame:
    frame = piece[["piece_id", "system", "H_pos_valid", "H_neg_valid", "H_net_valid",
                   "abs_H_net_valid", "cancellation_valid", "positive_plus_negative_valid",
                   "cancellation_ratio_piece"]].copy()
    nonzero = frame[~frame.system.eq("NO_PEDAL")].positive_plus_negative_valid
    threshold = float(nonzero.median())
    frame["descriptive_large_component_threshold"] = threshold
    frame["cancellation_ratio_ge_0p8"] = frame.cancellation_ratio_piece >= .8
    frame["large_components_and_high_cancellation"] = (
        (frame.positive_plus_negative_valid >= threshold) & frame.cancellation_ratio_ge_0p8
    )
    return frame


def frequency_severity(piece: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for system, group in piece.groupby("system"):
        rows.append(
            {
                "system": system,
                "piece_count": len(group),
                "mean_H_neg_valid": group.H_neg_valid.mean(),
                "mean_F_neg_valid": group.F_neg_valid.mean(),
                "mean_H_neg_cond_valid": group.H_neg_cond_valid.mean(),
                "pearson_H_neg_vs_F_neg": group.H_neg_valid.corr(group.F_neg_valid),
                "pearson_H_neg_vs_conditional_severity": group.H_neg_valid.corr(group.H_neg_cond_valid),
            }
        )
    return pd.DataFrame(rows)


def representative_pieces(piece: pd.DataFrame, split: pd.DataFrame) -> pd.DataFrame:
    wide = piece.pivot(index="piece_id", columns="system", values=["H_neg_valid", "cancellation_ratio_piece"])
    model_median = piece[piece.system.isin(broad.MODEL_CANONICAL)].pivot(
        index="piece_id", columns="system", values="H_neg_valid"
    ).median(axis=1)
    choices = [
        (wide["cancellation_ratio_piece"]["ALWAYS_ON"].idxmax(), "highest ALWAYS_ON cancellation_ratio"),
        ((wide["H_neg_valid"]["HUMAN"] - wide["H_neg_valid"]["ORIGINAL_PT"]).abs().idxmax(),
         "largest absolute HUMAN–ORIGINAL_PT negative-burden gap"),
        ((wide["H_neg_valid"]["ORIGINAL_PT"] - model_median).abs().idxmax(),
         "largest absolute ORIGINAL_PT–canonical-model-median negative-burden gap"),
    ]
    kreis = split[split.title.eq("Kreisleriana_6")]
    if len(kreis):
        choices.append((kreis.piece_id.iloc[0], "broad-sweep problem case: Schumann/Kreisleriana_6"))
    reasons: dict[str, list[str]] = {}
    for piece_id, reason in choices:
        reasons.setdefault(piece_id, []).append(reason)
    metadata = split.set_index("piece_id")
    rows = []
    for order, (piece_id, reason_list) in enumerate(reasons.items(), 1):
        rows.append({"selection_order": order, "piece_id": piece_id,
                     "performance_id": metadata.loc[piece_id, "performance_id"],
                     "composer": metadata.loc[piece_id, "composer"], "title": metadata.loc[piece_id, "title"],
                     "selection_reason": "; ".join(reason_list)})
    return pd.DataFrame(rows)


def top_failure_tables(results: list[dict[str, Any]], selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    reason = selected.set_index("piece_id").selection_reason.to_dict()
    selected_ids = set(selected.piece_id)
    onset_rows, pair_rows = [], []
    for result in results:
        if result["piece_id"] not in selected_ids:
            continue
        for onset_rank, packet in enumerate(result["packets"], 1):
            onset = packet["onset_row"]
            onset_rows.append({"piece_id": result["piece_id"], "system": result["system"],
                               "selection_reason": reason[result["piece_id"]],
                               "negative_onset_rank": onset_rank, **onset})
            for pair in packet["top_pairs"]:
                pair_rows.append({"piece_id": result["piece_id"], "system": result["system"],
                                  "selection_reason": reason[result["piece_id"]],
                                  "negative_onset_rank": onset_rank,
                                  "onset_index": onset["onset_index"], "onset_tick": onset["onset_tick"],
                                  **pair})
    return pd.DataFrame(onset_rows), pd.DataFrame(pair_rows)


def make_plots(work: Path, piece: pd.DataFrame, systems: pd.DataFrame) -> None:
    plot_dir = work / "plots"
    plot_dir.mkdir(exist_ok=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(SYSTEM_ORDER)))
    color_map = dict(zip(SYSTEM_ORDER, colors))

    fig, ax = plt.subplots(figsize=(10, 7))
    for system in SYSTEM_ORDER:
        group = piece[piece.system.eq(system)]
        ax.scatter(group.H_pos_valid, group.H_neg_valid, label=system, alpha=.75, color=color_map[system])
    ax.set(xlabel="H_pos_valid", ylabel="H_neg_valid (larger = worse)", title="Positive vs negative harmonic contribution")
    ax.legend(fontsize=7, ncol=2); fig.tight_layout(); fig.savefig(plot_dir / "system_H_pos_vs_H_neg_scatter.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    data = [piece[piece.system.eq(system)].H_neg_valid for system in SYSTEM_ORDER]
    ax.boxplot(data, tick_labels=SYSTEM_ORDER, showfliers=True)
    ax.tick_params(axis="x", rotation=40); ax.set_ylabel("H_neg_valid (larger = worse)"); ax.set_title("Negative burden by system")
    fig.tight_layout(); fig.savefig(plot_dir / "system_H_neg_boxplot.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 7))
    for system in SYSTEM_ORDER:
        group = piece[piece.system.eq(system)]
        ax.scatter(group.H_net_valid, group.H_neg_valid, label=system, alpha=.75, color=color_map[system])
    ax.axvline(0, color="black", lw=.8); ax.set(xlabel="Original Net H", ylabel="H_neg_valid (larger = worse)", title="Net score versus negative-only burden")
    ax.legend(fontsize=7, ncol=2); fig.tight_layout(); fig.savefig(plot_dir / "net_H_vs_H_neg_comparison.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    data = [piece[piece.system.eq(system)].cancellation_ratio_piece for system in SYSTEM_ORDER]
    ax.boxplot(data, tick_labels=SYSTEM_ORDER, showfliers=True)
    ax.tick_params(axis="x", rotation=40); ax.set_ylabel("cancellation_ratio_piece"); ax.set_ylim(-.03, 1.03); ax.set_title("Positive/negative cancellation by system")
    fig.tight_layout(); fig.savefig(plot_dir / "cancellation_ratio_by_system.png", dpi=180); plt.close(fig)

    pivot = piece.pivot(index="piece_id", columns="system", values="H_neg_valid")
    fig, ax = plt.subplots(figsize=(15, 7))
    x = np.arange(len(pivot))
    ax.plot(x, pivot.HUMAN, marker="o", label="HUMAN")
    ax.plot(x, pivot.ORIGINAL_PT, marker="o", label="ORIGINAL_PT")
    ax.plot(x, pivot[list(broad.MODEL_CANONICAL)].median(axis=1), marker="o", label="MODEL_CANONICAL median")
    ax.plot(x, pivot.ALWAYS_ON, marker="o", label="ALWAYS_ON")
    ax.set_xticks(x, pivot.index, rotation=45, ha="right"); ax.set_ylabel("H_neg_valid (larger = worse)")
    ax.set_title("Piece-wise negative burden"); ax.legend(); fig.tight_layout()
    fig.savefig(plot_dir / "piecewise_H_neg_ordering.png", dpi=180); plt.close(fig)


def md_table(frame: pd.DataFrame, columns: Sequence[str], limit: int | None = None) -> list[str]:
    shown = frame.loc[:, list(columns)].head(limit) if limit else frame.loc[:, list(columns)]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for values in shown.itertuples(index=False, name=None):
        cells = []
        for value in values:
            if isinstance(value, (float, np.floating)):
                cells.append("NA" if math.isnan(float(value)) else f"{float(value):.6f}")
            else:
                cells.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def write_report(work: Path, piece: pd.DataFrame, systems: pd.DataFrame, ordering: pd.DataFrame,
                 cancellation: pd.DataFrame, pair_summary: pd.DataFrame, frequency: pd.DataFrame,
                 selected: pd.DataFrame, sanity: pd.DataFrame) -> None:
    order = ordering.set_index(["comparison", "target_system_or_group"])
    human_pt = order.loc[("HUMAN_vs_ORIGINAL_PT", "ORIGINAL_PT")]
    pt_model = order.loc[("ORIGINAL_PT_vs_STAGE2_MACRO", "MODEL_CANONICAL")]
    model_always = order.loc[("STAGE2_vs_ALWAYS_ON_MACRO", "MODEL_CANONICAL")]
    full_c = order.loc[("FULL_ORDERING", "MODEL_CANONICAL")]
    full_a = order.loc[("FULL_ORDERING", "MODEL_ALL")]
    summary = systems.set_index("system")
    always = summary.loc["ALWAYS_ON"]
    human = summary.loc["HUMAN"]
    pt = summary.loc["ORIGINAL_PT"]
    always_cancel = cancellation[cancellation.system.eq("ALWAYS_ON")]
    high_cancel_rate = always_cancel.large_components_and_high_cancellation.mean()
    pair_all = pair_summary[pair_summary.negative_interval_class.eq("ALL")]
    pair_system = pair_all.groupby(["system", "pair_type"]).mean_valid_onset_normalized_negative_burden.mean().unstack(fill_value=0)
    freq = frequency.set_index("system")
    lines = [
        "# Harmonic Metric Positive / Negative Contribution Decomposition v0", "",
        "## Scope and integrity", "",
        "`Metric_harmonic_consonance.md` is the sole formula source. This audit reuses the pilot/broad note-instance parser, CC64 threshold, PA/PP construction, Hall Simple Type weights, attack-based Lehtonen decay, and the already-generated extremes. No metric term or parameter was tuned.", "",
        "Primary configuration: `alpha_decay=1.0`, `eta_PP=0.9`, `m0=48`, `beta_low=0`, `kappa_dyn=0`. The 13 broad-sweep pieces marked PEDAL_ELIGIBLE and EXACT_COMPUTABLE are treated as one diagnostic set; prior split labels are metadata only.", "",
        "`H_neg_valid` is a harm-only penalty, so larger is worse. NO_PEDAL=0 is expected and does not imply best overall pedaling quality.", "",
        "## Table 1 — Piece-balanced system summary", "",
    ]
    table1 = ["system", "H_pos_valid_mean", "H_pos_valid_median", "H_pos_valid_IQR",
              "H_neg_valid_mean", "H_neg_valid_median", "H_neg_valid_IQR",
              "H_net_valid_mean", "H_net_valid_median", "H_net_valid_IQR",
              "F_neg_valid_mean", "F_neg_valid_median", "H_neg_cond_valid_mean",
              "H_neg_cond_valid_median", "cancellation_valid_mean", "negative_onset_rate_mean",
              "H_neg_PA_valid_mean", "H_neg_PP_valid_mean"]
    lines += md_table(systems, table1)
    lines += ["", "## Table 2 — Net versus negative-only ordering", ""]
    lines += md_table(ordering, ["comparison", "target_system_or_group", "piece_or_cell_count",
        "net_desired_order_pass_count", "net_desired_order_rate",
        "negative_only_desired_order_pass_count", "negative_only_desired_order_rate",
        "negative_minus_net_rate_change"])
    lines += ["", "## Table 3 — Cancellation summary", ""]
    cancel_summary = piece.groupby("system", as_index=False).agg(
        H_pos_valid=("H_pos_valid", "mean"), H_neg_valid=("H_neg_valid", "mean"),
        abs_H_net_valid=("abs_H_net_valid", "mean"), cancellation_valid=("cancellation_valid", "mean"),
        cancellation_ratio_piece=("cancellation_ratio_piece", "mean"))
    lines += md_table(cancel_summary, list(cancel_summary.columns))
    lines += ["", "## Table 4 — PA versus PP negative burden", ""]
    pa_pp = systems[["system", "H_neg_PA_valid_mean", "H_neg_PP_valid_mean", "H_neg_PA_fraction", "H_neg_PP_fraction"]]
    lines += md_table(pa_pp, list(pa_pp.columns))
    lines += ["", "## Table 5 — Frequency versus conditional severity", ""]
    lines += md_table(frequency, list(frequency.columns))
    lines += ["", "## Table 6 — Automatically selected failure-onset pieces", ""]
    lines += md_table(selected, list(selected.columns))
    lines += ["", "## Sanity checks", ""]
    lines += md_table(sanity, list(sanity.columns))
    lines += [
        "", "## Answers to the diagnostic questions", "",
        f"**Q1. Is ALWAYS_ON hidden by cancellation?** ALWAYS_ON has mean H_pos={always.H_pos_valid_mean:.6f}, H_neg={always.H_neg_valid_mean:.6f}, net={always.H_net_valid_mean:.6f}, and mean piece cancellation ratio={always.cancellation_ratio_piece_mean:.3f}. {high_cancel_rate:.1%} of ALWAYS_ON pieces meet the descriptive joint flag (component sum at or above the non-NO_PEDAL median and cancellation ratio >=0.8). This directly quantifies whether its near-zero net is caused by large opposing components.", "",
        f"**Q2. Does negative-only recover the target penalty ordering?** The canonical full ordering changes from {full_c.net_desired_order_rate:.1%} under Net H to {full_c.negative_only_desired_order_rate:.1%} under H_neg; MODEL_ALL changes from {full_a.net_desired_order_rate:.1%} to {full_a.negative_only_desired_order_rate:.1%}.", "",
        f"**Q3. HUMAN versus PT.** Desired HUMAN>PT under net passes {int(human_pt.net_desired_order_pass_count)}/13; desired HUMAN<PT penalty passes {int(human_pt.negative_only_desired_order_pass_count)}/13. Mean H_neg is HUMAN={human.H_neg_valid_mean:.6f}, PT={pt.H_neg_valid_mean:.6f}.", "",
        f"**Q4. PT versus Stage2.** Across 52 canonical model/piece cells, desired ordering changes from {pt_model.net_desired_order_rate:.1%} to {pt_model.negative_only_desired_order_rate:.1%}. Models versus ALWAYS_ON changes from {model_always.net_desired_order_rate:.1%} to {model_always.negative_only_desired_order_rate:.1%}.", "",
        f"**Q5. PA or PP origin?** Mean ALWAYS_ON normalized burden is PA={pair_system.loc['ALWAYS_ON', 'PA']:.6f}, PP={pair_system.loc['ALWAYS_ON', 'PP']:.6f}; HUMAN is PA={pair_system.loc['HUMAN', 'PA']:.6f}, PP={pair_system.loc['HUMAN', 'PP']:.6f}; PT is PA={pair_system.loc['ORIGINAL_PT', 'PA']:.6f}, PP={pair_system.loc['ORIGINAL_PT', 'PP']:.6f}. Table 4 gives every system's component fractions.", "",
        f"**Q6. Frequency or severity?** For ALWAYS_ON, across-piece corr(H_neg,F_neg)={freq.loc['ALWAYS_ON', 'pearson_H_neg_vs_F_neg']:.3f} and corr(H_neg,conditional severity)={freq.loc['ALWAYS_ON', 'pearson_H_neg_vs_conditional_severity']:.3f}; HUMAN values are {freq.loc['HUMAN', 'pearson_H_neg_vs_F_neg']:.3f} and {freq.loc['HUMAN', 'pearson_H_neg_vs_conditional_severity']:.3f}. These are diagnostics, not causal attribution; the system means in Table 5 show which term shifts.", "",
        f"**Q7. Is cancellation the formula-family failure?** Negative-only full-chain improvement is {full_c.negative_minus_net_rate_change:+.1%} for MODEL_CANONICAL. If the resulting rate remains low, positive cancellation is not the principal explanation for the broad search failure; the separated negative burden itself does not consistently distinguish HUMAN, PT, and models.", "",
        "## Provenance and caveats", "",
        "- MODEL_CANONICAL is the same four strict canonical-stream Stage2 systems as the broad audit; MODEL_ALL adds CUSTOM_EVENT_V0.",
        "- HUMAN and CUSTOM_EVENT_V0 do not share strict canonical Stage1 timing/velocity and are reference distributions rather than pedal-only pairs.",
        "- PEDAL_SPARSE pieces and four DEFERRED_LONG_FORM pieces were not added. No approximation or pruning was introduced.",
        "- Optional frozen-config robustness was not run: this task remains a single reference-backbone decomposition and performs no parameter selection.",
        "- Every input path and unchanged SHA is recorded in `input_midis.csv`; `provenance.json` records TEST access=0, inference=0, and training=0.",
        "- `onset_decomposition.csv` includes all distinct onsets; Q-empty onsets are explicit zeros. `top_negative_onsets.csv` and `top_negative_pairs.csv` contain the selected direct-inspection examples.",
    ]
    (work / "HARMONIC_METRIC_POS_NEG_DECOMPOSITION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    work = output.parent / f".{output.name}.work"
    if output.exists() or work.exists():
        raise FileExistsError(f"refusing to overwrite existing output/work directory: {output} / {work}")
    work.mkdir(parents=True)

    split_all = pd.read_csv(BROAD / "split_manifest.csv")
    split = split_all[split_all.included_in_exact_analysis.astype(bool)].copy()
    if len(split) != 13 or not split.exact_compute_status.eq("EXACT_COMPUTABLE").all():
        raise AssertionError("expected exactly 13 broad-sweep exact-computable pieces")
    piece_ids = set(split.piece_id)
    broad_inputs = pd.read_csv(BROAD / "input_midis.csv")
    inputs = broad_inputs[broad_inputs.piece_id.isin(piece_ids)].copy()
    if len(inputs) != 13 * len(SYSTEM_ORDER) or set(inputs.system) != set(SYSTEM_ORDER):
        raise AssertionError("input intersection is not 13 pieces x 9 systems")
    inputs["current_sha256"] = inputs.midi_path.map(lambda value: sha256_file(Path(value)))
    inputs["source_sha_unchanged"] = inputs.sha256.eq(inputs.current_sha256)
    if not inputs.source_sha_unchanged.all():
        raise AssertionError("source MIDI SHA drift")
    inputs["system_group"] = inputs.system.map(_system_group)
    inputs.to_csv(work / "input_midis.csv", index=False)

    tasks = [(row.piece_id, row.system, row.midi_path) for row in inputs.itertuples()]
    results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_worker, task): task for task in tasks}
        for completed, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            print(f"DECOMPOSE {completed}/{len(tasks)} {result['piece_id']} {result['system']}", flush=True)

    onset_frames = []
    performance = inputs.set_index(["piece_id", "system"]).performance_id.to_dict()
    split_role = split.set_index("piece_id").split_role.to_dict()
    for result in results:
        frame = pd.DataFrame(result["onsets"])
        frame.insert(0, "system_group", _system_group(result["system"]))
        frame.insert(0, "split_role_metadata", split_role[result["piece_id"]])
        frame.insert(0, "performance_id", performance[(result["piece_id"], result["system"])])
        frame.insert(0, "system", result["system"])
        frame.insert(0, "piece_id", result["piece_id"])
        onset_frames.append(frame)
    onsets = pd.concat(onset_frames, ignore_index=True).sort_values(["piece_id", "system", "onset_index"])
    onsets.to_csv(work / "onset_decomposition.csv", index=False)

    piece = aggregate_pieces(onsets, inputs, split)
    reference = pd.read_csv(BROAD / "per_system_scores_top_configs.csv")
    reference = reference[reference.config_id.eq(REFERENCE_CONFIG_ID)][["piece_id", "system", "H_piece"]].drop_duplicates()
    piece = piece.merge(reference.rename(columns={"H_piece": "original_backbone_H_piece"}), on=["piece_id", "system"], how="left")
    piece["piece_backbone_reconstruction_abs_error"] = (piece.H_net_valid - piece.original_backbone_H_piece).abs()
    piece.to_csv(work / "piece_system_decomposition.csv", index=False)

    systems = system_summary(piece)
    systems.to_csv(work / "system_summary.csv", index=False)
    ordering, ordering_details = ordering_analysis(piece)
    ordering.to_csv(work / "ordering_comparison.csv", index=False)
    ordering_details.to_csv(work / "ordering_piece_details.csv", index=False)
    pair_summary = negative_pair_summary(results, piece)
    pair_summary.to_csv(work / "negative_pair_summary.csv", index=False)
    cancellation = cancellation_diagnostics(piece)
    cancellation.to_csv(work / "cancellation_diagnostics.csv", index=False)
    frequency = frequency_severity(piece)
    frequency.to_csv(work / "frequency_severity_diagnostics.csv", index=False)
    selected = representative_pieces(piece, split)
    selected.to_csv(work / "representative_piece_selection.csv", index=False)
    top_onsets, top_pairs = top_failure_tables(results, selected)
    top_onsets.to_csv(work / "top_negative_onsets.csv", index=False)
    top_pairs.to_csv(work / "top_negative_pairs.csv", index=False)

    pair_max = max(float(result["diagnostics"]["max_pair_reconstruction_abs_error"]) for result in results)
    onset_max = float(onsets.onset_reconstruction_abs_error.max())
    piece_max = float(piece.piece_backbone_reconstruction_abs_error.max())
    no_onset = onsets[onsets.system.eq("NO_PEDAL")]
    no_piece = piece[piece.system.eq("NO_PEDAL")]
    numeric_onset = onsets.select_dtypes(include=[np.number]).to_numpy()
    numeric_piece = piece.select_dtypes(include=[np.number]).to_numpy()
    test_paths = [value for value in inputs.midi_path if "/test/" in value.lower() or "test_set" in value.lower()]
    sanity_rows = [
        ("pair_c_equals_pos_minus_neg", pair_max <= 1e-12, f"max_abs_error={pair_max:.3e}"),
        ("valid_onset_Hbase_equals_Hpos_minus_Hneg", onset_max <= 1e-12, f"max_abs_error={onset_max:.3e}"),
        ("piece_backbone_equals_mean_net", piece_max <= 1e-12, f"max_abs_error={piece_max:.3e}"),
        ("NO_PEDAL_all_components_exact_zero", bool((no_onset[["H_pos_n", "H_neg_all_n", "H_net_n"]] == 0).all().all() and (no_piece[["H_pos_valid", "H_neg_valid", "H_net_valid"]] == 0).all().all()), f"onsets={len(no_onset)} pieces={len(no_piece)}"),
        ("all_scores_finite", bool(np.isfinite(numeric_onset).all() and np.isfinite(numeric_piece).all()), f"onset_numeric_cells={numeric_onset.size}; piece_numeric_cells={numeric_piece.size}"),
        ("no_AA_pairs", bool(onsets.AA_pair_count.eq(0).all()), f"AA_pair_count={int(onsets.AA_pair_count.sum())}"),
        ("pair_count_formula", bool((onsets.PA_pair_count == onsets.A_n_count * onsets.P_n_count).all() and (onsets.PP_pair_count == onsets.P_n_count * (onsets.P_n_count - 1) // 2).all()), "PA=|P|*|A|; PP=choose(|P|,2)"),
        ("source_MIDI_SHA_unchanged", bool(inputs.source_sha_unchanged.all()), f"files={len(inputs)}"),
        ("TEST_access_zero", not test_paths, "count=0"),
        ("new_inference_zero", True, "count=0"),
        ("training_zero", True, "count=0"),
        ("diagnostic_set_exact_13", len(piece_ids) == 13 and len(piece) == 117, f"pieces={len(piece_ids)} piece_systems={len(piece)}"),
    ]
    sanity = pd.DataFrame(sanity_rows, columns=["check", "passed", "detail"])
    sanity["status"] = np.where(sanity.passed, "PASS", "FAIL")
    sanity = sanity[["check", "status", "detail"]]
    sanity.to_csv(work / "sanity_checks.csv", index=False)
    if not sanity.status.eq("PASS").all():
        raise AssertionError(sanity[sanity.status.eq("FAIL")].to_dict("records"))

    make_plots(work, piece, systems)
    write_report(work, piece, systems, ordering, cancellation, pair_summary, frequency, selected, sanity)
    provenance = {
        "script_version": SCRIPT_VERSION,
        "metric_source_of_truth": "/workspace/project/Metric_harmonic_consonance.md",
        "metric_document_sha256": sha256_file(ROOT / "Metric_harmonic_consonance.md"),
        "primary_configuration": {"alpha_decay": 1.0, "eta_PP": 0.9, "m0": 48, "beta_low": 0.0, "kappa_dyn": 0.0},
        "piece_count": 13, "system_count": 9, "piece_system_count": 117,
        "parameter_search_performed": False, "optional_robustness_performed": False,
        "test_set_access_count": 0, "new_inference_count": 0, "training_steps": 0,
        "reused_broad_input_manifest": str(BROAD / "input_midis.csv"),
        "reused_extremes": True,
    }
    (work / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

    required = ["piece_system_decomposition.csv", "onset_decomposition.csv", "negative_pair_summary.csv",
                "ordering_comparison.csv", "cancellation_diagnostics.csv", "top_negative_onsets.csv",
                "top_negative_pairs.csv", "sanity_checks.csv", "HARMONIC_METRIC_POS_NEG_DECOMPOSITION_REPORT.md"]
    missing = [name for name in required if not (work / name).is_file()]
    if missing:
        raise AssertionError(f"missing required outputs: {missing}")
    for text_path in work.rglob("*"):
        if text_path.is_file() and text_path.suffix.lower() in {".csv", ".json", ".md", ".txt"}:
            payload = text_path.read_text(encoding="utf-8")
            text_path.write_text(payload.replace(str(work), str(output)), encoding="utf-8")
    os.rename(work, output)
    print(json.dumps({"status": "completed", "output": str(output), "piece_count": 13,
                      "piece_system_count": 117, "sanity_pass": int(sanity.status.eq("PASS").sum())}, indent=2))


if __name__ == "__main__":
    main()
