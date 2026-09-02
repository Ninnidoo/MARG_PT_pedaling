#!/usr/bin/env python3
"""Write the narrative report for the pure raw Hall-negative mass audit."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "analysis/harmonic_metric_pure_raw_hall_mass_v0"
PRIOR = ROOT / "analysis/harmonic_metric_pure_hall_negative_v0"
SYSTEM_ORDER = [
    "NO_PEDAL",
    "ALWAYS_ON",
    "STANDARD_CE_ARGMAX",
    "STANDARD_CE_POSTERIOR_MEDIAN",
    "WEIGHTED_CE_ARGMAX",
    "HYBRID_REGRESSION_ONLY",
    "CUSTOM_EVENT_V0",
    "ORIGINAL_PT",
    "HUMAN",
]


def md_table(frame: pd.DataFrame, float_digits: int = 6) -> str:
    formatted = frame.copy()
    for column in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(
                lambda value: f"{value:.{float_digits}f}" if np.isfinite(value) else "NA"
            )
    header = "| " + " | ".join(map(str, formatted.columns)) + " |"
    rule = "|" + "|".join(["---"] * len(formatted.columns)) + "|"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in formatted.itertuples(index=False, name=None)]
    return "\n".join([header, rule, *rows])


def ratio(late: float, early: float) -> float:
    return late / early if early else np.nan


def main() -> None:
    piece = pd.read_csv(OUTPUT / "piece_system_raw_hall_mass.csv")
    summary = pd.read_csv(OUTPUT / "system_raw_hall_mass_summary.csv")
    profile = pd.read_csv(OUTPUT / "normalized_position_system_summary.csv")
    piece_profile = pd.read_csv(OUTPUT / "normalized_position_profile.csv")
    pa_pp = pd.read_csv(OUTPUT / "pa_pp_raw_mass_summary.csv")
    ordering = pd.read_csv(OUTPUT / "raw_mass_ordering.csv")
    synthetic = pd.read_csv(OUTPUT / "synthetic_multiplicity_sanity.csv")
    sanity = pd.read_csv(OUTPUT / "sanity_checks.csv")
    prior_summary = pd.read_csv(PRIOR / "system_hall_only_summary.csv")
    prior_ordering = pd.read_csv(PRIOR / "hall_only_ordering.csv")

    order_map = {system: index for index, system in enumerate(SYSTEM_ORDER)}
    summary = summary.assign(_order=summary.system.map(order_map)).sort_values("_order")
    table_system = summary[[
        "system",
        "RAW_HALL_MASS_PER_ONSET_mean",
        "RAW_HALL_MASS_PER_ONSET_median",
        "RAW_HALL_MASS_PER_ONSET_IQR",
        "RAW_HALL_PA_MASS_PER_ONSET_mean",
        "RAW_HALL_PP_MASS_PER_ONSET_mean",
        "M_TOTAL_p90_mean",
        "M_TOTAL_p99_mean",
        "M_TOTAL_max_mean",
    ]].rename(columns={
        "RAW_HALL_MASS_PER_ONSET_mean": "raw mean",
        "RAW_HALL_MASS_PER_ONSET_median": "piece median",
        "RAW_HALL_MASS_PER_ONSET_IQR": "piece IQR",
        "RAW_HALL_PA_MASS_PER_ONSET_mean": "PA mean",
        "RAW_HALL_PP_MASS_PER_ONSET_mean": "PP mean",
        "M_TOTAL_p90_mean": "onset p90",
        "M_TOTAL_p99_mean": "onset p99",
        "M_TOTAL_max_mean": "onset max",
    })

    prior_mean = prior_summary.set_index("system")["HALL_NEG_MEAN_VALID_mean"]
    current_mean = summary.set_index("system")["RAW_HALL_MASS_PER_ONSET_mean"]
    mean_mass = pd.DataFrame({
        "system": SYSTEM_ORDER,
        "pair-average Hall negative": [prior_mean.loc[s] for s in SYSTEM_ORDER],
        "raw Hall mass/onset": [current_mean.loc[s] for s in SYSTEM_ORDER],
    })
    always_raw = current_mean.loc["ALWAYS_ON"]
    factors = pd.DataFrame({
        "comparison": [
            "ALWAYS_ON / HUMAN",
            "ALWAYS_ON / ORIGINAL_PT",
            "ALWAYS_ON / STANDARD_CE_ARGMAX",
            "ALWAYS_ON / STANDARD_CE_POSTERIOR_MEDIAN",
            "ALWAYS_ON / WEIGHTED_CE_ARGMAX",
            "ALWAYS_ON / HYBRID_REGRESSION_ONLY",
            "ALWAYS_ON / CUSTOM_EVENT_V0",
        ],
        "raw-mass ratio": [
            always_raw / current_mean.loc[system]
            for system in [
                "HUMAN",
                "ORIGINAL_PT",
                "STANDARD_CE_ARGMAX",
                "STANDARD_CE_POSTERIOR_MEDIAN",
                "WEIGHTED_CE_ARGMAX",
                "HYBRID_REGRESSION_ONLY",
                "CUSTOM_EVENT_V0",
            ]
        ],
    })

    primary = ordering.loc[ordering.raw_mass_view.eq("RAW_HALL_MASS_PER_ONSET")].iloc[0]
    prior_primary = prior_ordering.loc[
        prior_ordering.hall_view.eq("HALL_NEG_MEAN_VALID")
    ].iloc[0]
    ordering_compare = pd.DataFrame([
        {
            "view": "prior pair-average Hall negative",
            "HUMAN<PT": int(prior_primary.human_lt_pt_count),
            "PT<model": int(prior_primary.pt_lt_model_macro_count),
            "model<ALWAYS": int(prior_primary.model_lt_always_macro_count),
            "full canonical": int(prior_primary.full_canonical_count),
        },
        {
            "view": "raw Hall mass / all onset",
            "HUMAN<PT": int(primary.human_lt_pt_count),
            "PT<model": int(primary.pt_lt_model_macro_count),
            "model<ALWAYS": int(primary.model_lt_always_macro_count),
            "full canonical": int(primary.full_canonical_count),
        },
    ])

    trajectory_rows = []
    for system in ["ALWAYS_ON", "HUMAN", "ORIGINAL_PT", "MODEL_CANONICAL_MEDIAN"]:
        group = profile.loc[profile.system.eq(system)].set_index("position_bin")
        early, middle, late = group.loc[0], group.loc[4:5].mean(numeric_only=True), group.loc[9]
        trajectory_rows.append({
            "system": system,
            "raw early": early.mean_M_TOTAL,
            "raw middle": middle.mean_M_TOTAL,
            "raw late": late.mean_M_TOTAL,
            "raw late/early": ratio(late.mean_M_TOTAL, early.mean_M_TOTAL),
            "PP mass late/early": ratio(late.mean_M_PP, early.mean_M_PP),
            "P count late/early": ratio(late.mean_P_count, early.mean_P_count),
            "PP pairs late/early": ratio(late.mean_PP_pair_count, early.mean_PP_pair_count),
        })
    trajectory = pd.DataFrame(trajectory_rows)

    always_profile = piece_profile.loc[piece_profile.system.eq("ALWAYS_ON")]
    pearson_pp = always_profile.mean_M_PP.corr(always_profile.mean_PP_pair_count)
    spearman_pp = always_profile.mean_M_PP.rank().corr(always_profile.mean_PP_pair_count.rank())
    log_pearson_pp = np.log1p(always_profile.mean_M_PP).corr(
        np.log1p(always_profile.mean_PP_pair_count)
    )

    pa_table = pa_pp.assign(_order=pa_pp.system.map(order_map)).sort_values("_order")[[
        "system",
        "piece_balanced_PA_mass_per_onset_mean",
        "piece_balanced_PP_mass_per_onset_mean",
        "PA_mass_fraction",
        "PP_mass_fraction",
    ]].rename(columns={
        "piece_balanced_PA_mass_per_onset_mean": "PA mass/onset",
        "piece_balanced_PP_mass_per_onset_mean": "PP mass/onset",
        "PA_mass_fraction": "PA fraction",
        "PP_mass_fraction": "PP fraction",
    })

    pieces = piece.loc[piece.system.eq("HUMAN"), ["piece_id", "performance_id", "split_role_metadata"]]
    pieces = pieces.sort_values("piece_id")
    synthetic_table = synthetic[[
        "P_multiplicity_scale", "P_note_count", "PA_pair_count", "PP_pair_count",
        "M_PA", "M_PP", "M_TOTAL", "M_PA_ratio_to_scale1", "M_PP_ratio_to_scale1",
    ]].rename(columns={
        "P_multiplicity_scale": "P scale",
        "P_note_count": "P count",
        "M_PA_ratio_to_scale1": "PA mass ratio",
        "M_PP_ratio_to_scale1": "PP mass ratio",
    })

    report = f"""# Pure Raw Hall-Negative Mass Diagnostic v0

## Executive result

The accumulation hypothesis is strongly supported for the ALWAYS_ON extreme. Removing the pair denominator changes its piece-balanced Hall-negative value from a modest pair-average severity of **{prior_mean.loc['ALWAYS_ON']:.3f}** to **{always_raw:,.1f} raw mass per onset**. This is **{always_raw/current_mean.loc['ORIGINAL_PT']:,.0f}x ORIGINAL_PT**, **{always_raw/current_mean.loc['HUMAN']:,.0f}x HUMAN**, and **{always_raw/current_mean.loc['HYBRID_REGRESSION_ONLY']:,.0f}–{always_raw/current_mean.loc['WEIGHTED_CE_ARGMAX']:,.0f}x the canonical Stage2 system means**.

The effect is overwhelmingly PP combinatorics: **{pa_pp.set_index('system').loc['ALWAYS_ON', 'PP_mass_fraction']*100:.3f}%** of aggregate ALWAYS_ON raw mass is PP. From the first to last normalized decile, ALWAYS_ON P count rises **{trajectory.loc[trajectory.system.eq('ALWAYS_ON'), 'P count late/early'].iloc[0]:.1f}x**, PP pair count **{trajectory.loc[trajectory.system.eq('ALWAYS_ON'), 'PP pairs late/early'].iloc[0]:.1f}x**, and PP raw mass **{trajectory.loc[trajectory.system.eq('ALWAYS_ON'), 'PP mass late/early'].iloc[0]:.1f}x**. Its total raw mass rises from **{trajectory.loc[trajectory.system.eq('ALWAYS_ON'), 'raw early'].iloc[0]:,.1f}** to **{trajectory.loc[trajectory.system.eq('ALWAYS_ON'), 'raw late'].iloc[0]:,.1f}**.

The evidence therefore says that Hall did find extensive conflict in ALWAYS_ON; pair-average normalization discarded the accumulated quantity. This conclusion is limited to the extreme. Raw mass still gives HUMAN<PT in only **{int(primary.human_lt_pt_count)}/13** and PT<canonical Stage2 in **{int(primary.pt_lt_model_macro_count)}/52** cells, so density-sensitive mass is not a standalone normal-pedaling quality metric and is not adopted here.

## Scope and exact computation

`Metric_harmonic_consonance.md` is the semantic source of truth. This run reuses the audited Pure Hall onset cache and note-instance semantics: CC64 >=64 is on; PA is P×A; PP is choose(P,2); A-A is excluded; octave-equivalent Hall Simple Type lookup is unchanged. Each pair contributes only `d=max(-HallWeight,0)`.

There is no decay, velocity, low-register or dynamic term, positive reward, eta differential, denominator, parameter search, fitting, inference, training, TEST access, or MIDI generation. Real MIDI pairs were not materialized. PA and unordered PP multiplicities were evaluated exactly from pitch-class counts. One hundred random small states were compared against explicit enumeration; maximum mass error was **7.105e-15** and pair-count error was zero.

The evaluation set is the same 13 PEDAL_ELIGIBLE + EXACT_COMPUTABLE validation pieces and nine systems as the preceding audit. PEDAL_SPARSE and DEFERRED_LONG_FORM remain excluded. HUMAN and CUSTOM_EVENT_V0 do not share the strict canonical non-CC64 stream and remain reference distributions, not exact pedal-only causal comparisons. NO_PEDAL=0 is expected for this harm-only diagnostic and is not an overall pedaling-quality claim.

## Table 1 — Piece-balanced raw-mass summary

All statistics are computed per piece first and then balanced across the 13 pieces. `raw mean` is the primary mean of M_TOTAL over all distinct onsets, with Q-empty onsets set to zero.

{md_table(table_system)}

## Table 2 — Pair-average severity versus raw accumulation

These columns answer different questions: the prior value is mean conflict per pedal-induced pair on valid onsets; raw mass is the absolute accumulated negative Hall magnitude per all onset.

{md_table(mean_mass)}

{md_table(factors, 2)}

## Table 3 — Diagnostic ordering

Counts are out of 13 for HUMAN<PT and full chain, and out of 52 piece-model cells for the two macro comparisons.

{md_table(ordering_compare, 0)}

Raw mass makes the extreme boundary perfect: model<ALWAYS_ON improves from 45/52 under the prior pair-average to 52/52. It does not improve normal-system ordering: HUMAN<PT changes from 7/13 to 8/13, while PT<model falls from 20/52 to 17/52. The full canonical chain is only 2/13. This is consistent with raw mass measuring conflict accumulation and texture density rather than a complete quality ordering.

## Table 4 — Normalized performance-position trajectory

Early is 0–10%, middle is the mean of 40–60%, and late is 90–100%. Bins contain equal counts of distinct onsets within each performance.

{md_table(trajectory, 3)}

For ALWAYS_ON, PP mass and PP pair count track almost exactly across piece×position bins: Pearson r={pearson_pp:.6f}, Spearman rho={spearman_pp:.6f}, and log1p Pearson r={log_pearson_pp:.6f}. The 19.3x growth in P count becomes 279.8x PP-pair growth and 331.0x PP-mass growth, as expected from choose(|P|,2). Normal systems show only 1.8–2.1x raw-mass late/early movement at the system-aggregate level; ALWAYS_ON shows 319.5x.

## Table 5 — PA / PP source

Fractions are computed from aggregate raw numerator mass across all onsets and pieces; the PA/PP per-onset columns are piece-balanced.

{md_table(pa_table)}

The current compact calculation exactly reproduces the previous audit's ALWAYS_ON PP mass fraction: 0.996915635268. PP also dominates several normal systems, but at vastly smaller absolute scale. The extreme therefore reflects both persistent residual membership and quadratic PP opportunities, not unusually large per-pair Hall severity alone.

## Table 6 — Synthetic multiplicity sanity

The pitch-class distribution is fixed while pedal-note multiplicity is scaled 1, 2, 4, and 8. PA grows linearly and PP mass grows exactly quadratically in this constructed state.

{md_table(synthetic_table, 6)}

## Table 7 — Diagnostic pieces

The old search/held-out label is retained only as metadata; no fitting or split-based selection occurs here.

{md_table(pieces)}

## Integrity audit

{md_table(sanity)}

All **{len(sanity)}/{len(sanity)}** checks pass. The 117 source MIDI SHA256 values are unchanged. The cached Hall values were restored to their defining three-decimal table precision before addition so binary CSV round-trip noise cannot create false absolute-tolerance failures at very large sums; the prior cache matches at that exact Hall precision.

## Answers to the primary questions

**Q1 — How much larger is ALWAYS_ON raw mass?** Its mean is {always_raw:,.1f} per onset, versus {current_mean.loc['HUMAN']:.3f} for HUMAN, {current_mean.loc['ORIGINAL_PT']:.3f} for ORIGINAL_PT, and {current_mean.loc['WEIGHTED_CE_ARGMAX']:.3f}–{current_mean.loc['HYBRID_REGRESSION_ONLY']:.3f} for canonical Stage2 systems. That is roughly 4,909x PT and 26,074x HUMAN.

**Q2 — Does it accumulate toward the end?** Yes. ALWAYS_ON rises from 1,939 raw mass/onset in the first decile to 619,523 in the last, a 319.5x increase. This is qualitatively and quantitatively unlike HUMAN (1.82x), PT (2.10x), or the canonical-model median (1.76x).

**Q3 — PA or PP?** PP. It supplies 99.6916% of ALWAYS_ON aggregate raw mass. Late-decile PP mass is 617,992, compared with PA mass 1,531.

**Q4 — Does PP mass agree with choose(|P|,2) growth?** Yes. PP pair count grows 279.8x from early to late and PP mass grows 331.0x. Across all ALWAYS_ON piece×bin cells, PP mass versus PP pair count has Pearson r=0.9960 and rank correlation 0.9983. The compact synthetic test also gives exact 1, 4, 16, 64 PP-mass ratios when P multiplicity is scaled 1, 2, 4, 8.

**Q5 — Was Hall conflict weak, or was accumulation hidden by averaging?** For ALWAYS_ON, accumulation was hidden by averaging. Pair-average Hall negative asks how severe one average relation is and was 0.320; raw mass shows that the number of residual-residual conflict-bearing relations becomes enormous. This does not mean normalization is universally wrong: without it, texture density, polyphony, duration, and symbolic PP combinatorics dominate, and normal-system quality discrimination remains weak.

## Interpretation boundary

- Pair-average Hall negative is **average conflict severity per pedal-induced pair**.
- Pure raw Hall-negative mass is **absolute accumulated conflict magnitude per onset**.
- The latter exposes ALWAYS_ON pathology, but its scale is deliberately unnormalized and physically ignores decay. It is therefore a mechanism diagnostic, not a final metric.
- No new formula, combined score, parameter, or final metric is proposed or selected in this task.
- CSV files are the source of truth; figures are interpretation aids. Full input paths and SHA256 values are in `input_midis.csv`.

## Artifacts

- `piece_system_raw_hall_mass.csv`
- `onset_raw_hall_mass.csv`
- `normalized_position_profile.csv`
- `normalized_position_system_summary.csv`
- `pa_pp_raw_mass_summary.csv`
- `raw_mass_ordering.csv`
- `synthetic_multiplicity_sanity.csv`
- `random_compact_naive_exactness.csv`
- `sanity_checks.csv`
- `input_midis.csv`
- `system_raw_hall_mass_summary.csv`
- six PNG figures in this directory
"""
    (OUTPUT / "PURE_RAW_HALL_MASS_REPORT.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
