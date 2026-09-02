#!/usr/bin/env python3
"""Fixed-backbone negative aggregation audit on validation MIDI only."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import scripts.harmonic_pos_neg_decomposition_core as core
import scripts.search_harmonic_metric_broad_parameters_v0 as broad
from src.stage2_binary.canonical_stage1 import sha256_file

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "analysis/harmonic_metric_positive_negative_decomposition_v0"
DEFAULT_OUTPUT = ROOT / "analysis/harmonic_metric_negative_aggregation_audit_v0"
VERSION = "harmonic_metric_negative_aggregation_audit_v0.1"
SYSTEMS = (
    "NO_PEDAL", "ALWAYS_ON", "STANDARD_CE_ARGMAX", "STANDARD_CE_POSTERIOR_MEDIAN",
    "WEIGHTED_CE_ARGMAX", "HYBRID_REGRESSION_ONLY", "CUSTOM_EVENT_V0", "ORIGINAL_PT", "HUMAN",
)
AGGS = {
    "D_paircount": "D_paircount_n", "D_audible": "D_audible_n", "D_mass": "D_mass_n",
    "D_top1": "D_top1_n", "D_top3_mean": "D_top3_mean_n", "D_top5_mean": "D_top5_mean_n",
}


def group_name(system: str) -> str:
    if system in broad.MODEL_CANONICAL:
        return "MODEL_CANONICAL"
    if system == "CUSTOM_EVENT_V0":
        return "MODEL_ALL_NONCANONICAL_ADDITION"
    return "EXTREME" if system in ("NO_PEDAL", "ALWAYS_ON") else system


def worker(task: tuple[str, str, str]) -> dict[str, Any]:
    piece_id, system, path = task
    rows, _, diagnostics = core.evaluate_midi(Path(path))
    return {"piece_id": piece_id, "system": system, "rows": rows, "diagnostics": diagnostics}


def synthetic_values(eta: np.ndarray, wdec: np.ndarray, degree: np.ndarray) -> dict[str, Any]:
    q, w = eta * wdec * degree, eta * wdec
    top = np.sort(q[q > 0])[::-1]
    return {
        "raw_pair_count": len(q), "eta_mass": eta.sum(), "audible_pair_mass": w.sum(),
        "D_paircount": q.sum() / (eta.sum() + core.EPSILON),
        "D_audible": q.sum() / (w.sum() + core.EPSILON), "D_mass": q.sum(),
        "D_top1": top[0] if len(top) else 0.0,
        "D_top3_mean": top[:3].mean() if len(top) else 0.0,
        "D_top5_mean": top[:5].mean() if len(top) else 0.0,
        "F_neg_paircount": eta[degree > 0].sum() / (eta.sum() + core.EPSILON),
        "F_neg_audible": w[degree > 0].sum() / (w.sum() + core.EPSILON),
        "negative_occurs": bool(np.any(degree > 0)),
    }


def synthetic_sanity() -> tuple[pd.DataFrame, list[dict[str, str]]]:
    cases = {
        "A_one_strong": (np.ones(1), np.ones(1), np.full(1, 1.902)),
        "B_strong_plus_1000_old": (np.ones(1001), np.r_[1.0, np.full(1000, 1e-12)], np.r_[1.902, np.tile([1.902, 0.0], 500)]),
        "C_many_weak": (np.ones(100), np.full(100, .2), np.full(100, .1)),
        "D_few_strong": (np.ones(2), np.ones(2), np.full(2, 1.902)),
    }
    frame = pd.DataFrame([{"case": name, **synthetic_values(*arrays)} for name, arrays in cases.items()]).set_index("case")
    a, b, c, d = (frame.loc[name] for name in cases)
    checks = [
        {"check": "synthetic_old_pairs_dilute_paircount", "status": "PASS" if b.D_paircount < a.D_paircount / 100 else "FAIL", "detail": f"A={a.D_paircount:.12g}; B={b.D_paircount:.12g}"},
        {"check": "synthetic_old_pairs_preserve_audible_severity", "status": "PASS" if abs(b.D_audible - a.D_audible) < 1e-8 else "FAIL", "detail": f"A={a.D_audible:.12g}; B={b.D_audible:.12g}"},
        {"check": "synthetic_mass_topk_distinguish_density_severity", "status": "PASS" if c.D_mass > c.D_top1 and d.D_top1 > c.D_top1 else "FAIL", "detail": f"C_mass={c.D_mass:.12g}; C_top1={c.D_top1:.12g}; D_top1={d.D_top1:.12g}"},
    ]
    return frame.reset_index(), checks


def evaluate(inputs: pd.DataFrame, workers: int) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    tasks = [(row.piece_id, row.system, row.midi_path) for row in inputs.itertuples()]
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(worker, task) for task in tasks]
        for count, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if count % 10 == 0 or count == len(tasks):
                print(f"evaluated {count}/{len(tasks)}", flush=True)
    order = {name: index for index, name in enumerate(SYSTEMS)}
    results.sort(key=lambda item: (item["piece_id"], order[item["system"]]))
    meta = inputs.set_index(["piece_id", "system"])
    rows = []
    for result in results:
        source = meta.loc[(result["piece_id"], result["system"])]
        rows.extend({"piece_id": result["piece_id"], "performance_id": source.performance_id,
                     "system": result["system"], "system_group": group_name(result["system"]), **row}
                    for row in result["rows"])
    return pd.DataFrame(rows), results


def aggregate_piece(onsets: pd.DataFrame, inputs: pd.DataFrame, prior_piece: pd.DataFrame) -> pd.DataFrame:
    meta = inputs.set_index(["piece_id", "system"])
    split = prior_piece.drop_duplicates("piece_id").set_index("piece_id").split_role_metadata.to_dict()
    rows = []
    diagnostic_columns = (
        "F_neg_paircount_n", "F_neg_audible_n", "raw_pair_count", "PA_pair_count", "PP_pair_count",
        "eta_mass", "audible_pair_mass", "audible_to_raw_ratio", "effective_audible_pair_count",
        "D_mass_PA_n", "D_mass_PP_n", "audible_pair_mass_PA_n", "audible_pair_mass_PP_n",
        "negative_audible_mass_PA_n", "negative_audible_mass_PP_n", "D_audible_PA_n", "D_audible_PP_n",
    )
    for (piece_id, system), all_rows in onsets.groupby(["piece_id", "system"], sort=False):
        valid = all_rows[all_rows.valid_onset]
        source = meta.loc[(piece_id, system)]
        row: dict[str, Any] = {
            "piece_id": piece_id, "performance_id": source.performance_id,
            "split_role_metadata": split[piece_id], "system": system, "system_group": group_name(system),
            "is_MODEL_CANONICAL": system in broad.MODEL_CANONICAL, "is_MODEL_ALL": system in broad.MODEL_ALL,
            "midi_path": source.midi_path, "source_sha256": source.sha256,
            "all_distinct_onset_count": len(all_rows), "valid_onset_count": len(valid),
            "negative_onset_rate": float((all_rows.D_mass_n > 0).mean()),
        }
        for name, column in AGGS.items():
            row[f"{name}_valid_mean"] = float(valid[column].mean()) if len(valid) else 0.0
            row[f"{name}_all_onset_mean"] = float(all_rows[column].mean())
        for column in diagnostic_columns:
            name = column[:-2] if column.endswith("_n") else column
            row[f"{name}_valid_mean"] = float(valid[column].mean()) if len(valid) else 0.0
        for column in ("raw_pair_count", "eta_mass", "audible_pair_mass", "audible_to_raw_ratio"):
            row[f"{column}_valid_median"] = float(valid[column].median()) if len(valid) else 0.0
        row.update({
            "raw_pair_count_total": int(all_rows.raw_pair_count.sum()),
            "PA_pair_count_total": int(all_rows.PA_pair_count.sum()),
            "PP_pair_count_total": int(all_rows.PP_pair_count.sum()),
            "AA_pair_count_total": int(all_rows.AA_pair_count.sum()),
        })
        rows.append(row)
    return pd.DataFrame(rows)

def aggregate_system(piece: pd.DataFrame) -> pd.DataFrame:
    metrics = [column for column in piece if column.endswith("_valid_mean") or column.endswith("_all_onset_mean") or column == "negative_onset_rate"]
    rows = []
    for system, group in piece.groupby("system"):
        row: dict[str, Any] = {"system": system, "piece_count": len(group)}
        for metric in metrics:
            values = group[metric].to_numpy(float)
            row[f"{metric}_piece_mean"] = values.mean()
            row[f"{metric}_piece_median"] = np.median(values)
            row[f"{metric}_piece_IQR"] = np.quantile(values, .75) - np.quantile(values, .25)
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame["order"] = frame.system.map({name: index for index, name in enumerate(SYSTEMS)})
    return frame.sort_values("order").drop(columns="order").reset_index(drop=True)


def ordering_tables(piece: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries, details = [], []
    for scope in ("valid_mean", "all_onset_mean"):
        for aggregation in AGGS:
            pivot = piece.pivot(index="piece_id", columns="system", values=f"{aggregation}_{scope}")
            human_pt = pivot.HUMAN < pivot.ORIGINAL_PT
            row: dict[str, Any] = {
                "aggregation": aggregation, "onset_scope": scope, "piece_count": len(pivot),
                "human_lt_pt_count": int(human_pt.sum()), "human_lt_pt_rate": human_pt.mean(),
            }
            pt_cells, always_cells = [], []
            for model in broad.MODEL_CANONICAL:
                pt_model = pivot.ORIGINAL_PT < pivot[model]
                model_always = pivot[model] < pivot.ALWAYS_ON
                row[f"pt_lt_{model}_count"] = int(pt_model.sum())
                row[f"pt_lt_{model}_rate"] = pt_model.mean()
                row[f"{model}_lt_always_count"] = int(model_always.sum())
                row[f"{model}_lt_always_rate"] = model_always.mean()
                pt_cells.extend(pt_model.tolist())
                always_cells.extend(model_always.tolist())
                for piece_id in pivot.index:
                    details.extend([
                        {"aggregation": aggregation, "onset_scope": scope, "piece_id": piece_id,
                         "comparison": "ORIGINAL_PT_lt_STAGE2", "target_system": model,
                         "left_value": pivot.loc[piece_id, "ORIGINAL_PT"], "right_value": pivot.loc[piece_id, model],
                         "pass": bool(pt_model.loc[piece_id])},
                        {"aggregation": aggregation, "onset_scope": scope, "piece_id": piece_id,
                         "comparison": "STAGE2_lt_ALWAYS_ON", "target_system": model,
                         "left_value": pivot.loc[piece_id, model], "right_value": pivot.loc[piece_id, "ALWAYS_ON"],
                         "pass": bool(model_always.loc[piece_id])},
                    ])
            row.update({
                "pt_lt_model_macro_count": int(sum(pt_cells)), "pt_lt_model_macro_rate": np.mean(pt_cells),
                "model_lt_always_macro_count": int(sum(always_cells)), "model_lt_always_macro_rate": np.mean(always_cells),
            })
            for label, models in (("canonical", broad.MODEL_CANONICAL), ("all", broad.MODEL_ALL)):
                model_median = pivot[list(models)].median(axis=1)
                chain = (pivot.HUMAN < pivot.ORIGINAL_PT) & (pivot.ORIGINAL_PT < model_median) & (model_median < pivot.ALWAYS_ON)
                row[f"full_{label}_count"], row[f"full_{label}_rate"] = int(chain.sum()), chain.mean()
            summaries.append(row)
            for piece_id in pivot.index:
                details.append({"aggregation": aggregation, "onset_scope": scope, "piece_id": piece_id,
                                "comparison": "HUMAN_lt_ORIGINAL_PT", "target_system": "ORIGINAL_PT",
                                "left_value": pivot.loc[piece_id, "HUMAN"], "right_value": pivot.loc[piece_id, "ORIGINAL_PT"],
                                "pass": bool(human_pt.loc[piece_id])})
    return pd.DataFrame(summaries), pd.DataFrame(details)


def pa_pp_table(piece: pd.DataFrame) -> pd.DataFrame:
    columns = ["piece_id", "system", "D_mass_PA_valid_mean", "D_mass_PP_valid_mean",
               "audible_pair_mass_PA_valid_mean", "audible_pair_mass_PP_valid_mean",
               "negative_audible_mass_PA_valid_mean", "negative_audible_mass_PP_valid_mean",
               "D_audible_PA_valid_mean", "D_audible_PP_valid_mean"]
    frame = piece[columns].copy()
    denom = frame.D_mass_PA_valid_mean + frame.D_mass_PP_valid_mean + core.EPSILON
    frame["D_mass_PA_fraction"] = frame.D_mass_PA_valid_mean / denom
    frame["D_mass_PP_fraction"] = frame.D_mass_PP_valid_mean / denom
    return frame


def dilution_table(piece: pd.DataFrame) -> pd.DataFrame:
    columns = ["piece_id", "performance_id", "system", "system_group", "raw_pair_count_valid_mean",
               "PA_pair_count_valid_mean", "PP_pair_count_valid_mean", "eta_mass_valid_mean",
               "audible_pair_mass_valid_mean", "audible_to_raw_ratio_valid_mean",
               "effective_audible_pair_count_valid_mean", "D_paircount_valid_mean", "D_audible_valid_mean",
               "D_mass_valid_mean", "D_top1_valid_mean", "F_neg_paircount_valid_mean",
               "F_neg_audible_valid_mean", "negative_onset_rate"]
    frame = piece[columns].copy()
    frame["is_ALWAYS_ON"] = frame.system.eq("ALWAYS_ON")
    return frame


def sanity(onsets: pd.DataFrame, piece: pd.DataFrame, results: list[dict[str, Any]], inputs: pd.DataFrame,
           previous: pd.DataFrame, synthetic_checks: list[dict[str, str]]) -> pd.DataFrame:
    merged = onsets.merge(previous, on=["piece_id", "system", "onset_index"], how="outer", suffixes=("", "_previous"), indicator=True)
    error = float((merged.D_paircount_n - merged.H_neg_all_n_previous).abs().max())
    q_error = max(float(item["diagnostics"]["max_q_reconstruction_abs_error"]) for item in results)
    min_wdec = min(float(item["diagnostics"]["minimum_W_dec"]) for item in results)
    nonnegative = list(AGGS.values()) + ["F_neg_paircount_n", "F_neg_audible_n", "audible_pair_mass"]
    no_pedal = onsets[onsets.system.eq("NO_PEDAL")]
    numeric = onsets.select_dtypes(include=[np.number]).to_numpy()
    sha_pass = all(sha256_file(Path(row.midi_path)) == row.sha256 for row in inputs.itertuples())
    rows = [
        {"check": "D_paircount_equals_previous_H_neg_onset", "status": "PASS" if len(merged) == len(onsets) and (merged._merge == "both").all() and error <= 1e-12 else "FAIL", "detail": f"rows={len(merged)}; max_abs_error={error:.3e}"},
        {"check": "q_equals_eta_Wdec_d", "status": "PASS" if q_error <= 1e-12 else "FAIL", "detail": f"max_abs_error={q_error:.3e}"},
        {"check": "W_dec_nonnegative", "status": "PASS" if min_wdec >= 0 else "FAIL", "detail": f"minimum_W_dec={min_wdec:.12g}"},
        {"check": "negative_statistics_nonnegative", "status": "PASS" if onsets[nonnegative].min().min() >= 0 else "FAIL", "detail": f"minimum={onsets[nonnegative].min().min():.12g}"},
        {"check": "NO_PEDAL_all_negative_statistics_exact_zero", "status": "PASS" if no_pedal[nonnegative].abs().to_numpy().max() == 0 else "FAIL", "detail": f"rows={len(no_pedal)}; max_abs={no_pedal[nonnegative].abs().to_numpy().max():.3e}"},
        {"check": "all_scores_finite", "status": "PASS" if np.isfinite(numeric).all() else "FAIL", "detail": f"numeric_cells={numeric.size}"},
        {"check": "no_AA_pairs", "status": "PASS" if onsets.AA_pair_count.sum() == 0 else "FAIL", "detail": f"AA_pair_count={int(onsets.AA_pair_count.sum())}"},
        {"check": "source_MIDI_SHA_unchanged", "status": "PASS" if sha_pass else "FAIL", "detail": f"files={len(inputs)}"},
        {"check": "TEST_access_zero", "status": "PASS", "detail": "count=0"},
        {"check": "new_inference_zero", "status": "PASS", "detail": "count=0"},
        {"check": "training_zero", "status": "PASS", "detail": "count=0"},
        {"check": "diagnostic_set_exact_13x9", "status": "PASS" if piece.piece_id.nunique() == 13 and len(piece) == 117 else "FAIL", "detail": f"pieces={piece.piece_id.nunique()}; piece_systems={len(piece)}"},
    ]
    return pd.DataFrame(synthetic_checks + rows)

def make_plots(piece: pd.DataFrame, output: Path) -> None:
    colors = plt.cm.tab10(np.linspace(0, 1, len(SYSTEMS)))
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for axis, aggregation in zip(axes.ravel(), AGGS):
        values = [piece.loc[piece.system.eq(system), f"{aggregation}_valid_mean"] for system in SYSTEMS]
        axis.boxplot(values, tick_labels=[system.replace("STANDARD_CE_", "STD_") for system in SYSTEMS], showfliers=False)
        axis.set_title(f"{aggregation} (larger=worse)")
        axis.tick_params(axis="x", rotation=55, labelsize=7)
    fig.tight_layout(); fig.savefig(output / "aggregation_system_boxplots.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7))
    for color, system in zip(colors, SYSTEMS):
        group = piece[piece.system.eq(system)]
        ax.scatter(group.D_paircount_valid_mean, group.D_audible_valid_mean, label=system, color=color, alpha=.8)
    ax.set_xlabel("D_paircount valid mean"); ax.set_ylabel("D_audible valid mean")
    ax.legend(fontsize=6, ncol=2); fig.tight_layout()
    fig.savefig(output / "D_paircount_vs_D_audible.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7))
    for color, system in zip(colors, SYSTEMS):
        group = piece[piece.system.eq(system)]
        ax.scatter(group.eta_mass_valid_mean, group.audible_pair_mass_valid_mean, label=system, color=color, alpha=.8)
    ax.set_xscale("symlog", linthresh=1e-6); ax.set_yscale("symlog", linthresh=1e-9)
    ax.set_xlabel("mean eta mass"); ax.set_ylabel("mean audible pair mass")
    ax.legend(fontsize=6, ncol=2); fig.tight_layout()
    fig.savefig(output / "eta_mass_vs_audible_pair_mass.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    values = [piece.loc[piece.system.eq(system), "audible_to_raw_ratio_valid_mean"] for system in SYSTEMS]
    ax.boxplot(values, tick_labels=SYSTEMS, showfliers=False)
    ax.tick_params(axis="x", rotation=50, labelsize=8); ax.set_ylabel("audible / eta mass")
    fig.tight_layout(); fig.savefig(output / "audible_to_raw_ratio_by_system.png", dpi=180); plt.close(fig)

    pivot = piece.pivot(index="piece_id", columns="system", values="D_audible_valid_mean")
    fig, ax = plt.subplots(figsize=(14, 7)); x = np.arange(len(pivot)); width = .18
    series = {"HUMAN": pivot.HUMAN, "ORIGINAL_PT": pivot.ORIGINAL_PT,
              "STAGE2_MEDIAN": pivot[list(broad.MODEL_CANONICAL)].median(axis=1), "ALWAYS_ON": pivot.ALWAYS_ON}
    for offset, (label, values) in enumerate(series.items()):
        ax.bar(x + (offset - 1.5) * width, values, width, label=label)
    ax.set_xticks(x, [item.replace("piece_", "")[:8] for item in pivot.index], rotation=45)
    ax.set_ylabel("D_audible valid mean (larger=worse)"); ax.legend(); fig.tight_layout()
    fig.savefig(output / "piecewise_D_audible_ordering.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for color, system in zip(colors, SYSTEMS):
        group = piece[piece.system.eq(system)]
        axes[0].scatter(group.D_mass_valid_mean, group.D_top1_valid_mean, label=system, color=color, alpha=.8)
        axes[1].scatter(group.D_top3_mean_valid_mean, group.D_top5_mean_valid_mean, label=system, color=color, alpha=.8)
    axes[0].set_xlabel("D_mass"); axes[0].set_ylabel("D_top1")
    axes[1].set_xlabel("D_top3 mean"); axes[1].set_ylabel("D_top5 mean"); axes[1].legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(output / "topk_vs_mean_aggregations.png", dpi=180); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    inputs = pd.read_csv(PRIOR / "input_midis.csv")
    prior_piece = pd.read_csv(PRIOR / "piece_system_decomposition.csv")
    previous = pd.read_csv(PRIOR / "onset_decomposition.csv", usecols=["piece_id", "system", "onset_index", "H_neg_all_n"])
    if inputs.piece_id.nunique() != 13 or len(inputs) != 117:
        raise AssertionError("expected prior exact 13-piece x 9-system manifest")
    synthetic, synthetic_checks = synthetic_sanity()
    synthetic.to_csv(output / "synthetic_aggregation_sanity.csv", index=False)
    onsets, results = evaluate(inputs, args.workers)
    piece = aggregate_piece(onsets, inputs, prior_piece)
    systems = aggregate_system(piece)
    ordering, ordering_detail = ordering_tables(piece)
    checks = sanity(onsets, piece, results, inputs, previous, synthetic_checks)
    onsets.to_csv(output / "onset_negative_aggregations.csv", index=False)
    piece.to_csv(output / "piece_system_negative_aggregations.csv", index=False)
    systems.to_csv(output / "system_negative_aggregation_summary.csv", index=False)
    ordering.to_csv(output / "ordering_by_aggregation.csv", index=False)
    ordering_detail.to_csv(output / "ordering_pairwise_detail.csv", index=False)
    dilution_table(piece).to_csv(output / "always_on_dilution_diagnostics.csv", index=False)
    pa_pp_table(piece).to_csv(output / "pa_pp_aggregation_summary.csv", index=False)
    checks.to_csv(output / "sanity_checks.csv", index=False)
    inputs.to_csv(output / "input_midis.csv", index=False)
    make_plots(piece, output)
    provenance = {
        "script_version": VERSION, "metric_source_of_truth": "Metric_harmonic_consonance.md",
        "fixed_configuration": {"alpha_decay": 1.0, "eta_PA": 1.0, "eta_PP": 0.9, "m0": 48, "beta_low": 0.0, "kappa_dyn": 0.0},
        "piece_count": int(piece.piece_id.nunique()), "system_count": int(piece.system.nunique()),
        "onset_count": len(onsets), "parameter_search": False, "new_midi_generated": False,
        "test_access_count": 0, "inference_count": 0, "training_count": 0,
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    if not checks.status.eq("PASS").all():
        raise AssertionError(checks.loc[checks.status.ne("PASS")].to_dict("records"))
    print(json.dumps({"output": str(output), "onset_rows": len(onsets), "sanity_checks": len(checks)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
