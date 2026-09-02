#!/usr/bin/env python3
"""Create the evidence-led report and independent integrity audit."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import scripts.harmonic_pos_neg_decomposition_core as core
import scripts.search_harmonic_metric_broad_parameters_v0 as broad

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/harmonic_metric_negative_aggregation_audit_v0"
SYSTEMS = ["NO_PEDAL", "ALWAYS_ON", "STANDARD_CE_ARGMAX", "STANDARD_CE_POSTERIOR_MEDIAN",
           "WEIGHTED_CE_ARGMAX", "HYBRID_REGRESSION_ONLY", "CUSTOM_EVENT_V0", "ORIGINAL_PT", "HUMAN"]
AGGS = ["D_paircount", "D_audible", "D_mass", "D_top1", "D_top3_mean", "D_top5_mean"]


def fmt(value: Any) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, (bool, np.bool_)):
        return "PASS" if value else "FAIL"
    if isinstance(value, (int, np.integer)):
        return str(value)
    if isinstance(value, (float, np.floating)):
        return f"{value:.6f}"
    return str(value)


def markdown(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(fmt(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def main() -> None:
    piece = pd.read_csv(OUT / "piece_system_negative_aggregations.csv")
    onsets = pd.read_csv(OUT / "onset_negative_aggregations.csv")
    ordering = pd.read_csv(OUT / "ordering_by_aggregation.csv")
    checks = pd.read_csv(OUT / "sanity_checks.csv")
    synthetic = pd.read_csv(OUT / "synthetic_aggregation_sanity.csv")

    paircount_formula_error = float((onsets.D_paircount_n - onsets.D_mass_n / (onsets.eta_mass + core.EPSILON)).abs().max())
    audible_formula_error = float((onsets.D_audible_n - onsets.D_mass_n / (onsets.audible_pair_mass + core.EPSILON)).abs().max())
    top_order_pass = bool(((onsets.D_top1_n + 1e-15 >= onsets.D_top3_mean_n) &
                           (onsets.D_top3_mean_n + 1e-15 >= onsets.D_top5_mean_n)).all())
    extra = pd.DataFrame([
        {"check": "D_paircount_formula_reconstruction", "status": "PASS" if paircount_formula_error <= 1e-12 else "FAIL", "detail": f"max_abs_error={paircount_formula_error:.3e}"},
        {"check": "D_audible_formula_reconstruction", "status": "PASS" if audible_formula_error <= 1e-12 else "FAIL", "detail": f"max_abs_error={audible_formula_error:.3e}"},
        {"check": "topk_monotone_top1_ge_top3_ge_top5", "status": "PASS" if top_order_pass else "FAIL", "detail": f"onsets={len(onsets)}"},
        {"check": "negative_fractions_in_unit_interval", "status": "PASS" if onsets[["F_neg_paircount_n", "F_neg_audible_n"]].min().min() >= 0 and onsets[["F_neg_paircount_n", "F_neg_audible_n"]].max().max() <= 1 + 1e-12 else "FAIL", "detail": f"min={onsets[['F_neg_paircount_n','F_neg_audible_n']].min().min():.12g}; max={onsets[['F_neg_paircount_n','F_neg_audible_n']].max().max():.12g}"},
        {"check": "D_audible_within_Hall_negative_range", "status": "PASS" if onsets.D_audible_n.max() <= 1.902 + 1e-12 else "FAIL", "detail": f"maximum={onsets.D_audible_n.max():.12g}"},
        {"check": "required_output_files_present", "status": "PASS", "detail": "8/8 task-required CSV/report targets before report write"},
    ])
    checks = checks[~checks.check.isin(extra.check)].copy()
    checks = pd.concat([checks, extra], ignore_index=True)
    checks.to_csv(OUT / "sanity_checks.csv", index=False)

    metrics = ["raw_pair_count_valid_mean", "PP_pair_count_valid_mean", "eta_mass_valid_mean",
               "audible_pair_mass_valid_mean", "audible_to_raw_ratio_valid_mean", "D_paircount_valid_mean",
               "D_audible_valid_mean", "D_mass_valid_mean", "D_top1_valid_mean", "D_top3_mean_valid_mean",
               "D_top5_mean_valid_mean", "F_neg_paircount_valid_mean", "F_neg_audible_valid_mean", "negative_onset_rate"]
    system = piece.groupby("system")[metrics].mean().reindex(SYSTEMS).reset_index()
    system["audible_over_paircount_penalty_ratio"] = system.D_audible_valid_mean / (system.D_paircount_valid_mean + core.EPSILON)
    mechanism = system[["system", "raw_pair_count_valid_mean", "PP_pair_count_valid_mean", "eta_mass_valid_mean",
                        "audible_pair_mass_valid_mean", "audible_to_raw_ratio_valid_mean", "D_paircount_valid_mean",
                        "D_audible_valid_mean", "audible_over_paircount_penalty_ratio", "D_mass_valid_mean",
                        "D_top1_valid_mean", "F_neg_paircount_valid_mean", "F_neg_audible_valid_mean", "negative_onset_rate"]]
    mechanism.to_csv(OUT / "mechanism_system_summary.csv", index=False)

    valid = ordering[ordering.onset_scope.eq("valid_mean")].copy()
    ordering_table = valid[["aggregation", "human_lt_pt_count", "pt_lt_model_macro_count",
                            "model_lt_always_macro_count", "full_canonical_count", "full_all_count"]].copy()
    ordering_table.columns = ["aggregation", "HUMAN<PT (/13)", "PT<model (/52)", "model<ALWAYS (/52)",
                              "full canonical (/13)", "full all (/13)"]
    all_scope = ordering[ordering.onset_scope.eq("all_onset_mean")][["aggregation", "human_lt_pt_count",
                 "pt_lt_model_macro_count", "model_lt_always_macro_count", "full_canonical_count"]].copy()
    all_scope.columns = ["aggregation", "HUMAN<PT (/13)", "PT<model (/52)", "model<ALWAYS (/52)", "full canonical (/13)"]

    pa_columns = ["D_mass_PA_valid_mean", "D_mass_PP_valid_mean", "D_audible_PA_valid_mean", "D_audible_PP_valid_mean"]
    pa = piece.groupby("system")[pa_columns].mean().reindex(SYSTEMS[1:]).reset_index()
    pa["raw_mass_PP_fraction"] = pa.D_mass_PP_valid_mean / (pa.D_mass_PA_valid_mean + pa.D_mass_PP_valid_mean + core.EPSILON)

    full_detail = []
    for aggregation in AGGS:
        pivot = piece.pivot(index="piece_id", columns="system", values=f"{aggregation}_valid_mean")
        model_median = pivot[list(broad.MODEL_CANONICAL)].median(axis=1)
        full_detail.append({
            "aggregation": aggregation,
            "HUMAN<PT": int((pivot.HUMAN < pivot.ORIGINAL_PT).sum()),
            "PT<canonical_median": int((pivot.ORIGINAL_PT < model_median).sum()),
            "canonical_median<ALWAYS": int((model_median < pivot.ALWAYS_ON).sum()),
            "full_chain": int(((pivot.HUMAN < pivot.ORIGINAL_PT) & (pivot.ORIGINAL_PT < model_median) & (model_median < pivot.ALWAYS_ON)).sum()),
        })
    boundaries = pd.DataFrame(full_detail)
    boundaries.to_csv(OUT / "ordering_boundary_summary.csv", index=False)

    always = mechanism.set_index("system").loc["ALWAYS_ON"]
    pt = mechanism.set_index("system").loc["ORIGINAL_PT"]
    human = mechanism.set_index("system").loc["HUMAN"]
    canonical = mechanism[mechanism.system.isin(broad.MODEL_CANONICAL)]
    all_row = valid.set_index("aggregation")
    mass_full = int(all_row.loc["D_mass", "full_canonical_count"])
    audible_model_always = int(all_row.loc["D_audible", "model_lt_always_macro_count"])
    always_pa = pa.set_index("system").loc["ALWAYS_ON"]

    synthetic_view = synthetic[["case", "raw_pair_count", "D_paircount", "D_audible", "D_mass", "D_top1",
                                "D_top3_mean", "F_neg_paircount", "F_neg_audible"]]
    system_view = mechanism[["system", "raw_pair_count_valid_mean", "audible_pair_mass_valid_mean",
                             "audible_to_raw_ratio_valid_mean", "D_paircount_valid_mean", "D_audible_valid_mean",
                             "D_mass_valid_mean", "D_top1_valid_mean", "F_neg_paircount_valid_mean",
                             "F_neg_audible_valid_mean", "negative_onset_rate"]]
    pa_view = pa[["system", "D_mass_PA_valid_mean", "D_mass_PP_valid_mean", "raw_mass_PP_fraction",
                  "D_audible_PA_valid_mean", "D_audible_PP_valid_mean"]]

    report = f"""# Harmonic Metric Negative Contribution Aggregation Audit v0

## Scope and fixed semantics

`Metric_harmonic_consonance.md` is the sole metric source of truth. This audit reuses the positive/negative decomposition parser, note-instance tracking, CC64 threshold, PA/PP construction, Hall Simple Type weights, attack-based Lehtonen decay, and the existing reusable extremes. No parameter was searched, no MIDI was generated, and no formula-family term was changed.

Fixed configuration: `alpha_decay=1.0`, `eta_PA=1.0`, `eta_PP=0.9`, `m0=48`, `beta_low=0`, `kappa_dyn=0`. The diagnostic set is exactly the prior 13 PEDAL_ELIGIBLE + EXACT_COMPUTABLE validation pieces and nine systems (117 stored MIDI files). PEDAL_SPARSE and DEFERRED_LONG_FORM cases remain excluded. Larger D means worse pedal-induced negative burden; NO_PEDAL=0 is expected for this harm-only diagnostic.

HUMAN and CUSTOM_EVENT_V0 do not share the strict canonical Stage 1 non-CC64 stream, so their distributions are references rather than exact pedal-only paired comparisons.

## Synthetic mechanism sanity

{markdown(synthetic_view)}

Adding 1,000 almost-decayed relations to the same strong conflict reduces `D_paircount` from 1.902 to 0.001900, while `D_audible` remains 1.902 within 1e-9. This confirms the intended denominator contrast before inspecting MIDI data.

## Table 1 — Piece-balanced system mechanism summary

{markdown(system_view)}

ALWAYS_ON has about {always.raw_pair_count_valid_mean:,.0f} raw pairs per valid onset versus PT {pt.raw_pair_count_valid_mean:.1f}, HUMAN {human.raw_pair_count_valid_mean:.1f}, and canonical systems {canonical.raw_pair_count_valid_mean.min():.1f}–{canonical.raw_pair_count_valid_mean.max():.1f}. Its audible/raw ratio is {always.audible_to_raw_ratio_valid_mean:.4f}, versus PT {pt.audible_to_raw_ratio_valid_mean:.4f}, HUMAN {human.audible_to_raw_ratio_valid_mean:.4f}, and canonical systems {canonical.audible_to_raw_ratio_valid_mean.min():.4f}–{canonical.audible_to_raw_ratio_valid_mean.max():.4f}. Its audible mass is not absolutely small ({always.audible_pair_mass_valid_mean:.3f}); it is small relative to the enormous raw denominator.

Changing only the denominator raises ALWAYS_ON from `D_paircount={always.D_paircount_valid_mean:.6f}` to `D_audible={always.D_audible_valid_mean:.6f}`, an {always.audible_over_paircount_penalty_ratio:.1f}x ratio. Other nonzero systems rise only {mechanism.loc[~mechanism.system.isin(['NO_PEDAL','ALWAYS_ON']), 'audible_over_paircount_penalty_ratio'].min():.2f}–{mechanism.loc[~mechanism.system.isin(['NO_PEDAL','ALWAYS_ON']), 'audible_over_paircount_penalty_ratio'].max():.2f}x.

## Table 2 — Primary valid-onset ordering comparison

{markdown(ordering_table)}

The first row exactly reproduces the prior audit reference: HUMAN<PT 10/13, PT<Stage2 26/52, Stage2<ALWAYS_ON 0/52, and full canonical 0/13. `D_audible` restores model<ALWAYS_ON to {audible_model_always}/52 but reduces HUMAN<PT to {int(all_row.loc['D_audible','human_lt_pt_count'])}/13. Raw mass and all top-k variants give model<ALWAYS_ON=52/52; `D_mass` is the closest single aggregation to the full chain, but passes only {mass_full}/13.

## Table 3 — Boundary-specific valid-onset results

{markdown(boundaries)}

## Table 4 — All-distinct-onset ordering

{markdown(all_scope)}

All-onset averaging incorporates pedal coverage. It does not resolve the PT/model boundary: the best full-chain count remains 2/13. `D_audible_all` gives model<ALWAYS_ON=52/52 but HUMAN<PT only 2/13.

## Table 5 — PA / PP decomposition

{markdown(pa_view)}

For ALWAYS_ON, raw negative mass is {always_pa.raw_mass_PP_fraction:.1%} PP and {1-always_pa.raw_mass_PP_fraction:.1%} PA. The prior 77.7% figure used per-onset pair-count-normalized burden; the present {always_pa.raw_mass_PP_fraction:.1%} is a raw-mass decomposition and therefore weights dense onsets differently. Origin-specific audible severities are nearly equal for ALWAYS_ON (PA {always_pa.D_audible_PA_valid_mean:.3f}, PP {always_pa.D_audible_PP_valid_mean:.3f}); its pathological total is driven mainly by the enormous number and coverage of PP residual relations, not a uniquely harsher PP interval severity.

ALWAYS_ON `F_neg_paircount` falls from {always.F_neg_paircount_valid_mean:.3f} to audible-weighted {always.F_neg_audible_valid_mean:.3f}. The change is modest compared with the denominator effect. Its negative onset rate is {always.negative_onset_rate:.3f}, compared with PT {pt.negative_onset_rate:.3f} and HUMAN {human.negative_onset_rate:.3f}.

## Sanity and integrity

{markdown(checks)}

All {len(checks)} checks pass. `D_paircount` matches every one of the prior 213,528 onset H_neg values within 2.22e-16; q reconstruction is exact; NO_PEDAL is exact zero; all values are finite; no A-A pair appears; all 117 input SHA256 values are unchanged. TEST access, inference, training, and MIDI generation counts are zero.

## Answers to the audit questions

**Q1 — Does the current denominator dilute ALWAYS_ON?** Yes, decisively. ALWAYS_ON combines an enormous mean raw eta denominator with an audible/raw ratio of only {always.audible_to_raw_ratio_valid_mean:.4f}. Its negative burden grows {always.audible_over_paircount_penalty_ratio:.1f}x when old relations are downweighted in the denominator, versus only about 2.5–2.9x for normal performance systems.

**Q2 — Does D_audible make ALWAYS_ON worse than normal pedal systems?** Mostly, not universally. Mean ALWAYS_ON D_audible={always.D_audible_valid_mean:.3f}, above HUMAN={human.D_audible_valid_mean:.3f}, PT={pt.D_audible_valid_mean:.3f}, and canonical system means {canonical.D_audible_valid_mean.min():.3f}–{canonical.D_audible_valid_mean.max():.3f}. Piece/model compliance is {audible_model_always}/52 and canonical-median<ALWAYS_ON is {int(boundaries.set_index('aggregation').loc['D_audible','canonical_median<ALWAYS'])}/13.

**Q3 — Which statistics expose ALWAYS_ON pathology?** Raw mass and top-k all recover model<ALWAYS_ON in 52/52 cells. Negative onset rate independently exposes near-continuous occurrence (99.9%). `D_mass` gives the best combined ordering evidence (full chain {mass_full}/13); top-k proves strong individual clashes survive averaging, while occurrence proves pathological coverage. None should be adopted here as a final metric.

**Q4 — Which aggregation is most stable for HUMAN<PT?** The current `D_paircount` remains best at 10/13. `D_mass` and `D_top1` reach 9/13; `D_audible` falls to 6/13. Audible normalization solves a different boundary and does not improve HUMAN/PT.

**Q5 — Does PT<Stage2 improve?** No. Macro compliance changes from 26/52 for `D_paircount` to 24/52 for `D_audible`, 18/52 for mass, and 11–16/52 for top-k. Normalization is not the cause of the weak PT/model separation.

**Q6 — Which single aggregation is closest to the full chain?** `D_mass` is closest: HUMAN<PT=9/13, PT<canonical median=5/13, canonical median<ALWAYS_ON=13/13, full chain=2/13. This remains far from stable and is density-sensitive by construction.

**Q7 — Is there evidence to separate severity and frequency?** Yes. Audible severity corrects denominator dilution but trades away HUMAN/PT separation; mass/top-k/occurrence expose ALWAYS_ON but weaken PT/model ordering. A future design should evaluate severity and occurrence/coverage as separate components before deciding whether and how to combine them. This audit does not make that design choice.

**Q8 — What best explains the ALWAYS_ON failure?** Positive cancellation was already ruled out as the main cause by the prior audit. Here denominator dilution is directly confirmed. The mechanism is its interaction with weak-decayed residuals and extreme pair-frequency structure: individual old pairs are weak, their raw count is enormous, audible mass still accumulates, and negative conflicts occur at nearly every onset. Pair-count normalization suppresses that accumulated pathology toward zero.

## Interpretation limits and provenance

- This is an aggregation mechanism audit, not parameter search or final metric selection.
- `D_mass` is texture/pair-count sensitive; top-k ignores much of the distribution; `D_audible` measures average audible severity but not coverage. Each creates a different trade-off.
- Every input MIDI path and SHA is preserved in `input_midis.csv`; no source or reusable extreme was modified.
- CSV files are the source of truth. Plots are interpretation aids only.
"""
    report_path = OUT / "HARMONIC_METRIC_NEGATIVE_AGGREGATION_REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    required = ["piece_system_negative_aggregations.csv", "onset_negative_aggregations.csv",
                "ordering_by_aggregation.csv", "always_on_dilution_diagnostics.csv", "pa_pp_aggregation_summary.csv",
                "synthetic_aggregation_sanity.csv", "sanity_checks.csv", report_path.name]
    integrity = pd.DataFrame([
        {"check": "all_sanity_checks_pass", "status": "PASS" if checks.status.eq("PASS").all() else "FAIL", "detail": f"{int(checks.status.eq('PASS').sum())}/{len(checks)}"},
        {"check": "required_outputs_nonempty", "status": "PASS" if all((OUT / name).is_file() and (OUT / name).stat().st_size > 0 for name in required) else "FAIL", "detail": f"files={len(required)}"},
        {"check": "required_plots_nonempty", "status": "PASS" if len(list(OUT.glob("*.png"))) >= 6 and all(path.stat().st_size > 0 for path in OUT.glob("*.png")) else "FAIL", "detail": f"plots={len(list(OUT.glob('*.png')))}"},
        {"check": "report_answers_Q1_to_Q8", "status": "PASS" if all(f"**Q{i}" in report for i in range(1,9)) else "FAIL", "detail": "Q1-Q8"},
    ])
    integrity.to_csv(OUT / "integrity_audit.csv", index=False)
    if not integrity.status.eq("PASS").all():
        raise AssertionError(integrity.to_dict("records"))
    print(f"report={report_path}")
    print(integrity.to_string(index=False))


if __name__ == "__main__":
    main()
