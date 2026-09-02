#!/usr/bin/env python3
"""Finalize interpretation and independent integrity checks for decomposition v0."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/harmonic_metric_positive_negative_decomposition_v0"
EPS = 1e-12


def md_table(frame: pd.DataFrame, columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for values in frame[columns].itertuples(index=False, name=None):
        cells = []
        for value in values:
            if isinstance(value, (float, np.floating)):
                cells.append("NA" if math.isnan(float(value)) else f"{float(value):.6f}")
            else:
                cells.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def add_all_onset_summary() -> pd.DataFrame:
    piece = pd.read_csv(OUT / "piece_system_decomposition.csv")
    systems = pd.read_csv(OUT / "system_summary.csv")
    additions = []
    for system, group in piece.groupby("system"):
        row = {"system": system}
        for metric in ("H_pos_all_onsets", "H_neg_all_onsets", "H_net_all_onsets"):
            values = group[metric].to_numpy(float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_median"] = float(np.median(values))
            row[f"{metric}_IQR"] = float(np.quantile(values, .75) - np.quantile(values, .25))
        additions.append(row)
    add = pd.DataFrame(additions)
    systems = systems.drop(columns=[column for column in add.columns if column != "system" and column in systems], errors="ignore")
    systems = systems.merge(add, on="system", how="left")
    systems.to_csv(OUT / "system_summary.csv", index=False)
    return systems


def update_report(systems: pd.DataFrame) -> None:
    path = OUT / "HARMONIC_METRIC_POS_NEG_DECOMPOSITION_REPORT.md"
    text = path.read_text(encoding="utf-8")
    table_columns = ["system", "H_pos_all_onsets_mean", "H_neg_all_onsets_mean",
                     "H_net_all_onsets_mean", "negative_onset_rate_mean"]
    block = "\n".join([
        "## Table 1b — All-distinct-onset aggregation", "",
        md_table(systems, table_columns), "",
        "Q-empty onsets are zero here. ALWAYS_ON is barely diluted because negative pairs occur at almost every onset; other systems are diluted in proportion to their lower pedal-interaction coverage. This is an occurrence diagnostic and does not replace the valid-onset metric.", "",
    ])
    if "## Table 1b" not in text:
        text = text.replace("## Table 2 — Net versus negative-only ordering", block + "\n## Table 2 — Net versus negative-only ordering")

    summary = systems.set_index("system")
    always, human, pt = summary.loc["ALWAYS_ON"], summary.loc["HUMAN"], summary.loc["ORIGINAL_PT"]
    ordering = pd.read_csv(OUT / "ordering_comparison.csv").set_index(["comparison", "target_system_or_group"])
    human_pt = ordering.loc[("HUMAN_vs_ORIGINAL_PT", "ORIGINAL_PT")]
    pt_model = ordering.loc[("ORIGINAL_PT_vs_STAGE2_MACRO", "MODEL_CANONICAL")]
    model_always = ordering.loc[("STAGE2_vs_ALWAYS_ON_MACRO", "MODEL_CANONICAL")]
    full_c = ordering.loc[("FULL_ORDERING", "MODEL_CANONICAL")]
    full_a = ordering.loc[("FULL_ORDERING", "MODEL_ALL")]
    frequency = pd.read_csv(OUT / "frequency_severity_diagnostics.csv").set_index("system")
    cancellation = pd.read_csv(OUT / "cancellation_diagnostics.csv")
    high_cancel = cancellation[cancellation.system.eq("ALWAYS_ON")].large_components_and_high_cancellation.mean()

    replacements = {
        "Q1": f"**Q1. Is ALWAYS_ON hidden by cancellation?** No in the hypothesized absolute-mass sense. ALWAYS_ON has mean H_pos={always.H_pos_valid_mean:.6f}, H_neg={always.H_neg_valid_mean:.6f}, net={always.H_net_valid_mean:.6f}, and mean piece cancellation ratio={always.cancellation_ratio_piece_mean:.3f}. Its proportional cancellation is high, but both components are roughly 20–40 times smaller than the performance systems; {high_cancel:.1%} of ALWAYS_ON pieces meet the descriptive joint flag (component sum at or above the non-NO_PEDAL median and cancellation ratio >=0.8). Its near-zero net is primarily a small-contribution result, not hidden large opposing mass.",
        "Q2": f"**Q2. Does negative-only recover the target penalty ordering?** No. The canonical full ordering changes from {full_c.net_desired_order_rate:.1%} under Net H to {full_c.negative_only_desired_order_rate:.1%} under H_neg; MODEL_ALL changes from {full_a.net_desired_order_rate:.1%} to {full_a.negative_only_desired_order_rate:.1%}. Negative-only improves HUMAN versus PT but reverses the intended model-versus-ALWAYS_ON boundary.",
        "Q3": f"**Q3. HUMAN versus PT.** Desired HUMAN>PT under net passes {int(human_pt.net_desired_order_pass_count)}/13; desired HUMAN<PT penalty passes {int(human_pt.negative_only_desired_order_pass_count)}/13. Mean H_neg is HUMAN={human.H_neg_valid_mean:.6f}, PT={pt.H_neg_valid_mean:.6f}. This boundary improves by {human_pt.negative_minus_net_rate_change:+.1%}.",
        "Q4": f"**Q4. PT versus Stage2.** Across 52 canonical model/piece cells, desired ordering changes from {pt_model.net_desired_order_rate:.1%} to {pt_model.negative_only_desired_order_rate:.1%}. Models versus ALWAYS_ON changes from {model_always.net_desired_order_rate:.1%} to {model_always.negative_only_desired_order_rate:.1%}; negative-only does not improve the PT–model boundary and completely fails the model–ALWAYS boundary.",
        "Q6": f"**Q6. Frequency or severity?** ALWAYS_ON has the highest mean F_neg ({always.F_neg_valid_mean:.3f}) and negative-onset rate ({always.negative_onset_rate_mean:.3f}), but by far the lowest nonzero conditional severity ({always.H_neg_cond_valid_mean:.6f}). Across pieces, corr(H_neg,F_neg)={frequency.loc['ALWAYS_ON', 'pearson_H_neg_vs_F_neg']:.3f} while corr(H_neg,conditional severity)={frequency.loc['ALWAYS_ON', 'pearson_H_neg_vs_conditional_severity']:.3f}. HUMAN correlations are {frequency.loc['HUMAN', 'pearson_H_neg_vs_F_neg']:.3f} and {frequency.loc['HUMAN', 'pearson_H_neg_vs_conditional_severity']:.3f}. ALWAYS_ON creates negative relations almost everywhere, but attack-based decay plus pair normalization makes each relation extremely weak; severity, not frequency, controls its small H_neg.",
        "Q7": f"**Q7. Is cancellation the formula-family failure?** No. Negative-only full-chain improvement is {full_c.negative_minus_net_rate_change:+.1%} for MODEL_CANONICAL and the resulting rate is zero. Positive cancellation is not the principal explanation for the broad-search failure: the separated burden itself assigns ALWAYS_ON the smallest nonzero H_neg and only partly distinguishes HUMAN from PT/models.",
    }
    for key, replacement in replacements.items():
        text = re.sub(rf"\*\*{key}\..*?(?=\n\n\*\*Q[1-7]\.|\n\n## Provenance)", replacement, text, flags=re.S)
    path.write_text(text, encoding="utf-8")


def integrity_audit() -> pd.DataFrame:
    checks = []
    def record(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "status": "PASS" if passed else "FAIL", "detail": detail})

    piece = pd.read_csv(OUT / "piece_system_decomposition.csv")
    onset = pd.read_csv(OUT / "onset_decomposition.csv")
    top_onsets = pd.read_csv(OUT / "top_negative_onsets.csv")
    top_pairs = pd.read_csv(OUT / "top_negative_pairs.csv")
    sanity = pd.read_csv(OUT / "sanity_checks.csv")
    provenance = json.loads((OUT / "provenance.json").read_text(encoding="utf-8"))
    record("required_sanity_all_pass", sanity.status.eq("PASS").all(), f"checks={len(sanity)}")
    record("piece_system_13x9", len(piece) == 117 and piece.piece_id.nunique() == 13 and piece.system.nunique() == 9, f"rows={len(piece)}")
    record("all_distinct_onsets_saved", len(onset) > 0 and onset.groupby(["piece_id", "system"]).ngroups == 117, f"rows={len(onset)}")
    record("top_onsets_bounded_20", top_onsets.groupby(["piece_id", "system"]).size().max() <= 20, f"rows={len(top_onsets)}")
    record("top_pairs_bounded_10_per_onset", top_pairs.groupby(["piece_id", "system", "onset_index"]).size().max() <= 10, f"rows={len(top_pairs)}")
    record("all_piece_scores_finite", np.isfinite(piece.select_dtypes(include=[np.number]).to_numpy()).all(), "all numeric cells finite")
    record("TEST_inference_training_zero", provenance["test_set_access_count"] == 0 and provenance["new_inference_count"] == 0 and provenance["training_steps"] == 0, "0/0/0")
    required = ["piece_system_decomposition.csv", "onset_decomposition.csv", "negative_pair_summary.csv",
                "ordering_comparison.csv", "cancellation_diagnostics.csv", "top_negative_onsets.csv",
                "top_negative_pairs.csv", "sanity_checks.csv", "HARMONIC_METRIC_POS_NEG_DECOMPOSITION_REPORT.md"]
    plots = ["system_H_pos_vs_H_neg_scatter.png", "system_H_neg_boxplot.png", "net_H_vs_H_neg_comparison.png",
             "cancellation_ratio_by_system.png", "piecewise_H_neg_ordering.png"]
    missing = [name for name in required if not (OUT / name).is_file()]
    missing += [f"plots/{name}" for name in plots if not (OUT / "plots" / name).is_file()]
    record("required_outputs_and_plots_present", not missing, "missing=" + ",".join(missing))
    frame = pd.DataFrame(checks)
    frame.to_csv(OUT / "integrity_audit.csv", index=False)
    if not frame.status.eq("PASS").all():
        raise AssertionError(frame[frame.status.eq("FAIL")].to_dict("records"))
    return frame


def main() -> None:
    systems = add_all_onset_summary()
    update_report(systems)
    audit = integrity_audit()
    print(json.dumps({"status": "finalized", "integrity_pass": int(audit.status.eq("PASS").sum()),
                      "integrity_total": len(audit)}, indent=2))


if __name__ == "__main__":
    main()
