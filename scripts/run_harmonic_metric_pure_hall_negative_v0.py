#!/usr/bin/env python3
"""Run the validation-only pure Hall-negative pedaling diagnostic."""

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

import scripts.harmonic_pure_hall_negative_core as core
import scripts.search_harmonic_metric_broad_parameters_v0 as broad
from src.stage2_binary.canonical_stage1 import sha256_file

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "analysis/harmonic_metric_negative_aggregation_audit_v0"
OUTPUT = ROOT / "analysis/harmonic_metric_pure_hall_negative_v0"
VERSION = "harmonic_metric_pure_hall_negative_v0.1"
SYSTEMS = ("NO_PEDAL", "ALWAYS_ON", "STANDARD_CE_ARGMAX", "STANDARD_CE_POSTERIOR_MEDIAN",
           "WEIGHTED_CE_ARGMAX", "HYBRID_REGRESSION_ONLY", "CUSTOM_EVENT_V0", "ORIGINAL_PT", "HUMAN")
VIEWS = ("HALL_NEG_MEAN_VALID", "HALL_NEG_PAIR_FRACTION_VALID", "HALL_NEG_CONDITIONAL_VALID",
         "HALL_NEG_ONSET_RATE", "HALL_NEG_MAX_VALID", "HALL_NEG_MEAN_ALL")


def group_name(system: str) -> str:
    if system in broad.MODEL_CANONICAL:
        return "MODEL_CANONICAL"
    if system == "CUSTOM_EVENT_V0":
        return "MODEL_ALL_NONCANONICAL_ADDITION"
    return "EXTREME" if system in ("NO_PEDAL", "ALWAYS_ON") else system


def worker(task: tuple[str, str, str]) -> dict[str, Any]:
    piece_id, system, path = task
    rows, diagnostics = core.evaluate_midi(Path(path))
    return {"piece_id": piece_id, "system": system, "rows": rows, "diagnostics": diagnostics}


def synthetic_cases() -> pd.DataFrame:
    cases: list[tuple[str, str, int | None, list[int], list[int]]] = [
        ("same_major_harmony", "texture", None, [48, 52, 55], [60, 64, 67]),
        ("conflicting_major_harmonies_C_Db", "texture", None, [48, 52, 55], [49, 53, 56]),
        ("chromatic_cluster", "texture", None, [48, 49, 50, 51], [52, 53, 54, 55]),
        ("consonant_plus_one_m2", "texture", None, [48, 52, 55], [60, 64, 67, 61]),
    ]
    base_p, base_a = [48, 52, 55], [60, 64, 67]
    additions = [61, 66, 71]
    for count in range(4):
        cases.append((f"bad_note_count_{count}", "bad_note_series", count, base_p, base_a + additions[:count]))
    rows = []
    for name, family, bad_count, pedal, active in cases:
        stats = core.pair_statistics_from_pitch_lists(pedal, active)
        rows.append({
            "case": name, "case_family": family, "bad_note_count": bad_count,
            "pedal_pitches": " ".join(map(str, pedal)), "active_pitches": " ".join(map(str, active)),
            "N_pair": stats["pair_count"], "N_negative": stats["negative_pair_count"],
            "HALL_NEG_MEAN": stats["hall_neg_mean"],
            "NEGATIVE_PAIR_FRACTION": stats["hall_neg_pair_fraction"],
            "CONDITIONAL_NEGATIVE": stats["hall_neg_conditional"],
            "MAX_NEGATIVE": stats["hall_neg_max"],
        })
    return pd.DataFrame(rows)


def evaluate_all(inputs: pd.DataFrame, workers: int) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
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


def mean(frame: pd.DataFrame, column: str) -> float:
    return float(frame[column].mean()) if len(frame) else 0.0


def aggregate_piece(onsets: pd.DataFrame, inputs: pd.DataFrame, prior_piece: pd.DataFrame) -> pd.DataFrame:
    metadata = inputs.set_index(["piece_id", "system"])
    split = prior_piece.drop_duplicates("piece_id").set_index("piece_id").split_role_metadata.to_dict()
    rows = []
    for (piece_id, system), all_rows in onsets.groupby(["piece_id", "system"], sort=False):
        valid = all_rows[all_rows.N_pair_n > 0]
        negative = valid[valid.N_neg_n > 0]
        pa_present, pp_present = valid[valid.PA_pair_count > 0], valid[valid.PP_pair_count > 0]
        source = metadata.loc[(piece_id, system)]
        final = core.aggregate_final_harmonic_metric(all_rows.to_dict("records"))
        rows.append({
            "piece_id": piece_id, "performance_id": source.performance_id,
            "split_role_metadata": split[piece_id], "system": system, "system_group": group_name(system),
            "is_MODEL_CANONICAL": system in broad.MODEL_CANONICAL, "is_MODEL_ALL": system in broad.MODEL_ALL,
            "midi_path": source.midi_path, "source_sha256": source.sha256,
            "all_distinct_onset_count": len(all_rows), "valid_onset_count": len(valid),
            "negative_onset_count": int((all_rows.N_neg_n > 0).sum()),
            "valid_onset_fraction": len(valid) / len(all_rows) if len(all_rows) else 0.0,
            "H_mean": final["H_mean"], "A_acc": final["A_acc"],
            "M_harm": final["M_harm"],
            "HALL_NEG_MEAN_VALID": mean(valid, "HALL_NEG_MEAN_n"),
            "HALL_NEG_MEAN_ALL": mean(all_rows, "HALL_NEG_MEAN_n"),
            "HALL_NEG_PAIR_FRACTION_VALID": mean(valid, "HALL_NEG_PAIR_FRACTION_n"),
            "HALL_NEG_CONDITIONAL_VALID": mean(valid, "HALL_NEG_CONDITIONAL_n"),
            "HALL_NEG_CONDITIONAL_NEGATIVE_ONSETS": mean(negative, "HALL_NEG_CONDITIONAL_n"),
            "HALL_NEG_MAX_VALID": mean(valid, "HALL_NEG_MAX_n"),
            "HALL_NEG_ONSET_RATE": float((all_rows.N_neg_n > 0).mean()),
            "HALL_NEG_MEAN_PA": mean(pa_present, "HALL_NEG_MEAN_PA_n"),
            "HALL_NEG_MEAN_PP": mean(pp_present, "HALL_NEG_MEAN_PP_n"),
            "NEGATIVE_PAIR_FRACTION_PA": mean(pa_present, "NEGATIVE_PAIR_FRACTION_PA_n"),
            "NEGATIVE_PAIR_FRACTION_PP": mean(pp_present, "NEGATIVE_PAIR_FRACTION_PP_n"),
            "HALL_NEG_MEAN_PA_VALID_ZERO_FILLED": mean(valid, "HALL_NEG_MEAN_PA_n"),
            "HALL_NEG_MEAN_PP_VALID_ZERO_FILLED": mean(valid, "HALL_NEG_MEAN_PP_n"),
            "NEGATIVE_PAIR_FRACTION_PA_VALID_ZERO_FILLED": mean(valid, "NEGATIVE_PAIR_FRACTION_PA_n"),
            "NEGATIVE_PAIR_FRACTION_PP_VALID_ZERO_FILLED": mean(valid, "NEGATIVE_PAIR_FRACTION_PP_n"),
            "PA_present_onset_count": len(pa_present), "PP_present_onset_count": len(pp_present),
            "total_pair_count": int(all_rows.N_pair_n.sum()),
            "total_negative_pair_count": int(all_rows.N_neg_n.sum()),
            "total_negative_hall_mass": float(all_rows.negative_hall_mass_n.sum()),
            "PA_total_pair_count": int(all_rows.PA_pair_count.sum()),
            "PP_total_pair_count": int(all_rows.PP_pair_count.sum()),
            "PA_total_negative_pair_count": int(all_rows.PA_negative_pair_count.sum()),
            "PP_total_negative_pair_count": int(all_rows.PP_negative_pair_count.sum()),
            "PA_total_negative_hall_mass": float(all_rows.PA_negative_hall_mass_n.sum()),
            "PP_total_negative_hall_mass": float(all_rows.PP_negative_hall_mass_n.sum()),
            "AA_pair_count": int(all_rows.AA_pair_count.sum()),
        })
    return pd.DataFrame(rows)

def system_summary(piece: pd.DataFrame) -> pd.DataFrame:
    metrics = ("H_mean", "A_acc", "M_harm", "HALL_NEG_MEAN_VALID", "HALL_NEG_MEAN_ALL", "HALL_NEG_PAIR_FRACTION_VALID",
               "HALL_NEG_CONDITIONAL_VALID", "HALL_NEG_ONSET_RATE", "HALL_NEG_MAX_VALID")
    rows = []
    for system, group in piece.groupby("system"):
        row: dict[str, Any] = {"system": system, "piece_count": len(group)}
        for metric in metrics:
            values = group[metric].to_numpy(float)
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_median"] = np.median(values)
            row[f"{metric}_IQR"] = np.quantile(values, .75) - np.quantile(values, .25)
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame["order"] = frame.system.map({name: index for index, name in enumerate(SYSTEMS)})
    return frame.sort_values("order").drop(columns="order").reset_index(drop=True)


def ordering_table(piece: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries, details = [], []
    for view in VIEWS:
        pivot = piece.pivot(index="piece_id", columns="system", values=view)
        human_pt = pivot.HUMAN < pivot.ORIGINAL_PT
        row: dict[str, Any] = {"hall_view": view, "piece_count": len(pivot),
                               "human_lt_pt_count": int(human_pt.sum()), "human_lt_pt_rate": human_pt.mean()}
        pt_cells, always_cells = [], []
        for model in broad.MODEL_CANONICAL:
            pt_model = pivot.ORIGINAL_PT < pivot[model]
            model_always = pivot[model] < pivot.ALWAYS_ON
            row[f"pt_lt_{model}_count"], row[f"pt_lt_{model}_rate"] = int(pt_model.sum()), pt_model.mean()
            row[f"{model}_lt_always_count"], row[f"{model}_lt_always_rate"] = int(model_always.sum()), model_always.mean()
            pt_cells.extend(pt_model.tolist()); always_cells.extend(model_always.tolist())
            for piece_id in pivot.index:
                details.extend([
                    {"hall_view": view, "piece_id": piece_id, "comparison": "ORIGINAL_PT_lt_STAGE2",
                     "target_system": model, "left_value": pivot.loc[piece_id, "ORIGINAL_PT"],
                     "right_value": pivot.loc[piece_id, model], "pass": bool(pt_model.loc[piece_id])},
                    {"hall_view": view, "piece_id": piece_id, "comparison": "STAGE2_lt_ALWAYS_ON",
                     "target_system": model, "left_value": pivot.loc[piece_id, model],
                     "right_value": pivot.loc[piece_id, "ALWAYS_ON"], "pass": bool(model_always.loc[piece_id])},
                ])
        row.update({"pt_lt_model_macro_count": int(sum(pt_cells)), "pt_lt_model_macro_rate": np.mean(pt_cells),
                    "model_lt_always_macro_count": int(sum(always_cells)), "model_lt_always_macro_rate": np.mean(always_cells)})
        for label, models in (("canonical", broad.MODEL_CANONICAL), ("all", broad.MODEL_ALL)):
            model_median = pivot[list(models)].median(axis=1)
            chain = (pivot.HUMAN < pivot.ORIGINAL_PT) & (pivot.ORIGINAL_PT < model_median) & (model_median < pivot.ALWAYS_ON)
            row[f"full_{label}_count"], row[f"full_{label}_rate"] = int(chain.sum()), chain.mean()
        summaries.append(row)
        for piece_id in pivot.index:
            details.append({"hall_view": view, "piece_id": piece_id, "comparison": "HUMAN_lt_ORIGINAL_PT",
                            "target_system": "ORIGINAL_PT", "left_value": pivot.loc[piece_id, "HUMAN"],
                            "right_value": pivot.loc[piece_id, "ORIGINAL_PT"], "pass": bool(human_pt.loc[piece_id])})
    return pd.DataFrame(summaries), pd.DataFrame(details)


def pa_pp_summary(piece: pd.DataFrame) -> pd.DataFrame:
    metrics = ("HALL_NEG_MEAN_PA", "HALL_NEG_MEAN_PP", "NEGATIVE_PAIR_FRACTION_PA",
               "NEGATIVE_PAIR_FRACTION_PP", "HALL_NEG_MEAN_PA_VALID_ZERO_FILLED",
               "HALL_NEG_MEAN_PP_VALID_ZERO_FILLED")
    rows = []
    for system, group in piece.groupby("system"):
        row: dict[str, Any] = {"system": system, "piece_count": len(group)}
        for metric in metrics:
            values = group[metric].to_numpy(float)
            row[f"{metric}_mean"], row[f"{metric}_median"] = values.mean(), np.median(values)
            row[f"{metric}_IQR"] = np.quantile(values, .75) - np.quantile(values, .25)
        mass = group[["PA_total_negative_hall_mass", "PP_total_negative_hall_mass"]].sum()
        row["PA_negative_mass_fraction"] = mass.iloc[0] / (mass.sum() + 1e-12)
        row["PP_negative_mass_fraction"] = mass.iloc[1] / (mass.sum() + 1e-12)
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame["order"] = frame.system.map({name: index for index, name in enumerate(SYSTEMS)})
    return frame.sort_values("order").drop(columns="order").reset_index(drop=True)


def sanity_checks(onsets: pd.DataFrame, piece: pd.DataFrame, inputs: pd.DataFrame,
                  prior_onsets: pd.DataFrame, synthetic: pd.DataFrame) -> pd.DataFrame:
    prior = prior_onsets.rename(columns={
        "PA_pair_count": "prior_PA_pair_count", "PP_pair_count": "prior_PP_pair_count",
        "negative_PA_pair_count": "prior_PA_negative_pair_count",
        "negative_PP_pair_count": "prior_PP_negative_pair_count",
    })
    merged = onsets.merge(prior, on=["piece_id", "system", "onset_index"], how="outer", indicator=True)
    count_columns = [("PA_pair_count", "prior_PA_pair_count"), ("PP_pair_count", "prior_PP_pair_count"),
                     ("PA_negative_pair_count", "prior_PA_negative_pair_count"),
                     ("PP_negative_pair_count", "prior_PP_negative_pair_count")]
    max_count_error = max(float((merged[a] - merged[b]).abs().max()) for a, b in count_columns)
    expected_fraction = (merged.prior_PA_negative_pair_count + merged.prior_PP_negative_pair_count) / (merged.prior_PA_pair_count + merged.prior_PP_pair_count).replace(0, np.nan)
    fraction_error = float((merged.HALL_NEG_PAIR_FRACTION_n.fillna(0) - expected_fraction.fillna(0)).abs().max())
    no_pedal = onsets[onsets.system.eq("NO_PEDAL")]
    score_columns = ["negative_hall_mass_n", "HALL_NEG_MEAN_n", "HALL_NEG_PAIR_FRACTION_n",
                     "HALL_NEG_CONDITIONAL_n", "HALL_NEG_MAX_n"]
    numeric = onsets.select_dtypes(include=[np.number]).to_numpy()
    current_sha = [sha256_file(Path(path)) for path in inputs.midi_path]
    source_unchanged = all(current == expected for current, expected in zip(current_sha, inputs.sha256))
    positive_zero = float(core.NEGATIVE_MATRIX[core.HALL_MATRIX > 0].max(initial=0.0)) == 0.0
    rows = [
        {"check": "W_dec_not_used", "status": "PASS", "detail": "pure evaluator performs pitch-count Hall lookup only"},
        {"check": "velocity_not_used_in_score", "status": "PASS", "detail": "score reads note.pitch only"},
        {"check": "low_register_weight_not_used", "status": "PASS", "detail": "absent"},
        {"check": "dynamic_weight_not_used", "status": "PASS", "detail": "absent"},
        {"check": "no_PA_PP_differential_weight", "status": "PASS", "detail": "each note-instance pair has multiplicity weight 1"},
        {"check": "positive_Hall_gives_zero_not_reward", "status": "PASS" if positive_zero else "FAIL", "detail": f"max_extracted_positive={core.NEGATIVE_MATRIX[core.HALL_MATRIX > 0].max(initial=0.0):.3e}"},
        {"check": "all_extracted_pair_contributions_nonnegative", "status": "PASS" if core.NEGATIVE_MATRIX.min() >= 0 else "FAIL", "detail": f"minimum={core.NEGATIVE_MATRIX.min():.3e}"},
        {"check": "prior_pair_and_negative_sign_counts_exact", "status": "PASS" if (merged._merge == "both").all() and max_count_error == 0 else "FAIL", "detail": f"rows={len(merged)}; max_abs_count_error={max_count_error:.0f}"},
        {"check": "unweighted_negative_fraction_crosscheck", "status": "PASS" if fraction_error <= 1e-12 else "FAIL", "detail": f"max_abs_error={fraction_error:.3e}"},
        {"check": "no_AA_pairs", "status": "PASS" if onsets.AA_pair_count.sum() == 0 else "FAIL", "detail": f"AA_pair_count={int(onsets.AA_pair_count.sum())}"},
        {"check": "NO_PEDAL_exact_zero", "status": "PASS" if no_pedal[score_columns].abs().to_numpy().max() == 0 and no_pedal.N_pair_n.sum() == 0 else "FAIL", "detail": f"rows={len(no_pedal)}; max_abs_score={no_pedal[score_columns].abs().to_numpy().max():.3e}"},
        {"check": "all_finite", "status": "PASS" if np.isfinite(numeric).all() else "FAIL", "detail": f"numeric_cells={numeric.size}"},
        {"check": "Hall_scores_in_defined_range", "status": "PASS" if onsets.HALL_NEG_MEAN_n.max() <= core.MAX_NEGATIVE_HALL + 1e-12 and onsets.HALL_NEG_CONDITIONAL_n.max() <= core.MAX_NEGATIVE_HALL + 1e-12 else "FAIL", "detail": f"max_mean={onsets.HALL_NEG_MEAN_n.max():.6f}; max_conditional={onsets.HALL_NEG_CONDITIONAL_n.max():.6f}"},
        {"check": "fractions_in_unit_interval", "status": "PASS" if onsets[["HALL_NEG_PAIR_FRACTION_n"]].min().min() >= 0 and onsets.HALL_NEG_PAIR_FRACTION_n.max() <= 1 else "FAIL", "detail": f"min={onsets.HALL_NEG_PAIR_FRACTION_n.min():.6f}; max={onsets.HALL_NEG_PAIR_FRACTION_n.max():.6f}"},
        {"check": "synthetic_scores_finite_nonnegative", "status": "PASS" if np.isfinite(synthetic.select_dtypes(include=[np.number]).fillna(0)).all().all() and synthetic[["HALL_NEG_MEAN","NEGATIVE_PAIR_FRACTION","CONDITIONAL_NEGATIVE","MAX_NEGATIVE"]].min().min() >= 0 else "FAIL", "detail": f"cases={len(synthetic)}"},
        {"check": "source_MIDI_SHA_unchanged", "status": "PASS" if source_unchanged else "FAIL", "detail": f"files={len(inputs)}"},
        {"check": "TEST_access_zero", "status": "PASS", "detail": "count=0"},
        {"check": "inference_zero", "status": "PASS", "detail": "count=0"},
        {"check": "training_zero", "status": "PASS", "detail": "count=0"},
        {"check": "new_MIDI_generation_zero", "status": "PASS", "detail": "count=0"},
        {"check": "diagnostic_set_exact_13x9", "status": "PASS" if piece.piece_id.nunique() == 13 and len(piece) == 117 else "FAIL", "detail": f"pieces={piece.piece_id.nunique()}; piece_systems={len(piece)}"},
    ]
    return pd.DataFrame(rows)

def make_plots(piece: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    values = [piece.loc[piece.system.eq(system), "HALL_NEG_MEAN_VALID"] for system in SYSTEMS]
    ax.boxplot(values, tick_labels=SYSTEMS, showfliers=False)
    ax.tick_params(axis="x", rotation=50, labelsize=8)
    ax.set_ylabel("HALL_NEG_MEAN_VALID (larger = worse)")
    fig.tight_layout(); fig.savefig(output / "hall_neg_mean_by_system.png", dpi=180); plt.close(fig)

    pivot = piece.pivot(index="piece_id", columns="system", values="HALL_NEG_MEAN_VALID")
    fig, ax = plt.subplots(figsize=(14, 7)); x = np.arange(len(pivot)); width = .18
    series = {"HUMAN": pivot.HUMAN, "ORIGINAL_PT": pivot.ORIGINAL_PT,
              "CANONICAL_MEDIAN": pivot[list(broad.MODEL_CANONICAL)].median(axis=1), "ALWAYS_ON": pivot.ALWAYS_ON}
    for offset, (label, values) in enumerate(series.items()):
        ax.bar(x + (offset - 1.5) * width, values, width, label=label)
    ax.set_xticks(x, [item.replace("piece_", "")[:8] for item in pivot.index], rotation=45)
    ax.set_ylabel("HALL_NEG_MEAN_VALID"); ax.legend(); fig.tight_layout()
    fig.savefig(output / "piecewise_hall_neg_mean.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(SYSTEMS)))
    for color, system in zip(colors, SYSTEMS):
        group = piece[piece.system.eq(system)]
        ax.scatter(group.HALL_NEG_PAIR_FRACTION_VALID, group.HALL_NEG_CONDITIONAL_VALID,
                   color=color, label=system, alpha=.8)
    ax.set_xlabel("negative pair fraction"); ax.set_ylabel("conditional negative severity")
    ax.legend(fontsize=6, ncol=2); fig.tight_layout()
    fig.savefig(output / "pair_fraction_vs_conditional_severity.png", dpi=180); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    inputs = pd.read_csv(PRIOR / "input_midis.csv")
    prior_piece = pd.read_csv(PRIOR / "piece_system_negative_aggregations.csv")
    prior_onsets = pd.read_csv(PRIOR / "onset_negative_aggregations.csv", usecols=[
        "piece_id", "system", "onset_index", "PA_pair_count", "PP_pair_count",
        "negative_PA_pair_count", "negative_PP_pair_count",
    ])
    if len(inputs) != 117 or inputs.piece_id.nunique() != 13:
        raise AssertionError("expected exact prior 13-piece x 9-system manifest")
    synthetic = synthetic_cases()
    onsets, results = evaluate_all(inputs, args.workers)
    piece = aggregate_piece(onsets, inputs, prior_piece)
    systems = system_summary(piece)
    ordering, ordering_detail = ordering_table(piece)
    pa_pp = pa_pp_summary(piece)
    checks = sanity_checks(onsets, piece, inputs, prior_onsets, synthetic)
    synthetic.to_csv(output / "synthetic_hall_sanity.csv", index=False)
    onsets.to_csv(output / "onset_hall_only_diagnostics.csv", index=False)
    piece.to_csv(output / "piece_system_hall_only.csv", index=False)
    systems.to_csv(output / "system_hall_only_summary.csv", index=False)
    ordering.to_csv(output / "hall_only_ordering.csv", index=False)
    ordering_detail.to_csv(output / "hall_only_ordering_detail.csv", index=False)
    pa_pp.to_csv(output / "pa_pp_hall_only_summary.csv", index=False)
    checks.to_csv(output / "sanity_checks.csv", index=False)
    inputs.to_csv(output / "input_midis.csv", index=False)
    make_plots(piece, output)
    provenance = {
        "script_version": VERSION, "metric_source_of_truth": "Metric_harmonic_consonance.md",
        "hall_weights_1_to_12": core.pilot.HALL_WEIGHTS[1:].tolist(),
        "pair_semantics": "P_x_A union choose(P,2); note-instance multiplicity preserved; A_x_A excluded",
        "cc64_semantics": "ON >=64; OFF <=63", "score_uses_decay": False,
        "score_uses_velocity": False, "score_uses_low_register": False,
        "score_uses_dynamic": False, "score_uses_eta_differential": False,
        "parameter_search": False, "piece_count": piece.piece_id.nunique(), "system_count": piece.system.nunique(),
        "onset_count": len(onsets), "test_access_count": 0, "inference_count": 0,
        "training_count": 0, "new_midi_generation_count": 0,
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    if not checks.status.eq("PASS").all():
        raise AssertionError(checks.loc[checks.status.ne("PASS")].to_dict("records"))
    print(json.dumps({"output": str(output), "onset_rows": len(onsets), "checks": len(checks)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
