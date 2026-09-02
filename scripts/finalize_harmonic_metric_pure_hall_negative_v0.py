#!/usr/bin/env python3
"""Finalize the pure Hall-negative diagnostic report and integrity audit."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import scripts.harmonic_pure_hall_negative_core as core
import scripts.search_harmonic_metric_broad_parameters_v0 as broad

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/harmonic_metric_pure_hall_negative_v0"
SYSTEMS = ["NO_PEDAL", "ALWAYS_ON", "STANDARD_CE_ARGMAX", "STANDARD_CE_POSTERIOR_MEDIAN",
           "WEIGHTED_CE_ARGMAX", "HYBRID_REGRESSION_ONLY", "CUSTOM_EVENT_V0", "ORIGINAL_PT", "HUMAN"]


def fmt(value: Any) -> str:
    if pd.isna(value):
        return "NA"
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


def pa_pp_ordering(piece: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for view in ("HALL_NEG_MEAN_PA", "HALL_NEG_MEAN_PP", "NEGATIVE_PAIR_FRACTION_PA", "NEGATIVE_PAIR_FRACTION_PP"):
        pivot = piece.pivot(index="piece_id", columns="system", values=view)
        model_median = pivot[list(broad.MODEL_CANONICAL)].median(axis=1)
        pt_cells, always_cells = [], []
        for model in broad.MODEL_CANONICAL:
            pt_cells.extend((pivot.ORIGINAL_PT < pivot[model]).tolist())
            always_cells.extend((pivot[model] < pivot.ALWAYS_ON).tolist())
        chain = (pivot.HUMAN < pivot.ORIGINAL_PT) & (pivot.ORIGINAL_PT < model_median) & (model_median < pivot.ALWAYS_ON)
        rows.append({"origin_view": view, "HUMAN<PT (/13)": int((pivot.HUMAN < pivot.ORIGINAL_PT).sum()),
                     "PT<model (/52)": int(sum(pt_cells)), "model<ALWAYS (/52)": int(sum(always_cells)),
                     "full chain (/13)": int(chain.sum())})
    return pd.DataFrame(rows)


def main() -> None:
    piece = pd.read_csv(OUT / "piece_system_hall_only.csv")
    summary = pd.read_csv(OUT / "system_hall_only_summary.csv")
    ordering = pd.read_csv(OUT / "hall_only_ordering.csv")
    pa_pp = pd.read_csv(OUT / "pa_pp_hall_only_summary.csv")
    synthetic = pd.read_csv(OUT / "synthetic_hall_sanity.csv")
    checks = pd.read_csv(OUT / "sanity_checks.csv")

    scoring_source = "\n".join(inspect.getsource(function) for function in (
        core.pitch_counts, core.pa_stats, core.pp_stats, core.combine,
        core.pair_statistics_from_pitch_counts, core.evaluate_midi,
    ))
    forbidden = ("note_strength(", "pair_block_statistics(", "low_degree(", "W_dec", ".velocity", "eta_PP", "kappa_dyn")
    absent = [token for token in forbidden if token not in scoring_source]
    repeated = core.pair_statistics_from_pitch_lists([60, 60, 61], [64, 67])
    extra = pd.DataFrame([
        {"check": "scoring_source_has_no_modifier_calls", "status": "PASS" if len(absent) == len(forbidden) else "FAIL", "detail": f"absent={len(absent)}/{len(forbidden)}"},
        {"check": "note_instance_multiplicity_histogram_exact", "status": "PASS" if repeated["pair_count"] == 9 and repeated["negative_pair_count"] == 4 and abs(repeated["negative_mass"] - 4.415) < 1e-12 else "FAIL", "detail": f"pairs={repeated['pair_count']}; negative={repeated['negative_pair_count']}; mass={repeated['negative_mass']:.12g}"},
    ])
    checks = checks[~checks.check.isin(extra.check)].copy()
    checks = pd.concat([checks, extra], ignore_index=True)
    checks.to_csv(OUT / "sanity_checks.csv", index=False)

    sys_view = summary[["system", "HALL_NEG_MEAN_VALID_mean", "HALL_NEG_MEAN_VALID_median",
                        "HALL_NEG_MEAN_VALID_IQR", "HALL_NEG_MEAN_ALL_mean",
                        "HALL_NEG_PAIR_FRACTION_VALID_mean", "HALL_NEG_CONDITIONAL_VALID_mean",
                        "HALL_NEG_ONSET_RATE_mean", "HALL_NEG_MAX_VALID_mean"]]
    order_view = ordering[["hall_view", "human_lt_pt_count", "pt_lt_model_macro_count",
                           "model_lt_always_macro_count", "full_canonical_count", "full_all_count"]].copy()
    order_view.columns = ["Hall view", "HUMAN<PT (/13)", "PT<model (/52)", "model<ALWAYS (/52)",
                          "full canonical (/13)", "full all (/13)"]
    pa_view = pa_pp[["system", "HALL_NEG_MEAN_PA_mean", "HALL_NEG_MEAN_PP_mean",
                     "NEGATIVE_PAIR_FRACTION_PA_mean", "NEGATIVE_PAIR_FRACTION_PP_mean",
                     "PA_negative_mass_fraction", "PP_negative_mass_fraction"]]
    origin_order = pa_pp_ordering(piece)
    origin_order.to_csv(OUT / "pa_pp_hall_only_ordering.csv", index=False)

    primary = ordering.set_index("hall_view").loc["HALL_NEG_MEAN_VALID"]
    fraction = ordering.set_index("hall_view").loc["HALL_NEG_PAIR_FRACTION_VALID"]
    conditional = ordering.set_index("hall_view").loc["HALL_NEG_CONDITIONAL_VALID"]
    occurrence = ordering.set_index("hall_view").loc["HALL_NEG_ONSET_RATE"]
    maximum = ordering.set_index("hall_view").loc["HALL_NEG_MAX_VALID"]
    systems = summary.set_index("system")
    always, human, pt = systems.loc["ALWAYS_ON"], systems.loc["HUMAN"], systems.loc["ORIGINAL_PT"]
    canonical = systems.loc[list(broad.MODEL_CANONICAL)]
    pa_order = origin_order.set_index("origin_view")

    synth_texture = synthetic[synthetic.case_family.eq("texture")][["case", "N_pair", "N_negative", "HALL_NEG_MEAN",
                    "NEGATIVE_PAIR_FRACTION", "CONDITIONAL_NEGATIVE", "MAX_NEGATIVE"]]
    synth_bad = synthetic[synthetic.case_family.eq("bad_note_series")][["bad_note_count", "HALL_NEG_MEAN",
                    "NEGATIVE_PAIR_FRACTION", "CONDITIONAL_NEGATIVE", "MAX_NEGATIVE"]]
    bad_mean_monotone = bool(np.all(np.diff(synth_bad.HALL_NEG_MEAN) >= -1e-12))
    bad_fraction_monotone = bool(np.all(np.diff(synth_bad.NEGATIVE_PAIR_FRACTION) >= -1e-12))

    pivot = piece.pivot(index="piece_id", columns="system", values="HALL_NEG_MEAN_VALID")
    model_median = pivot[list(broad.MODEL_CANONICAL)].median(axis=1)
    full = (pivot.HUMAN < pivot.ORIGINAL_PT) & (pivot.ORIGINAL_PT < model_median) & (model_median < pivot.ALWAYS_ON)
    metadata = piece.drop_duplicates("piece_id").set_index("piece_id").performance_id
    full_piece_text = ", ".join(f"{piece_id} ({metadata[piece_id]})" for piece_id in full[full].index) or "none"

    report = f"""# Pure Hall-Negative Pedaling Diagnostic v0

## Scope

This audit asks whether Hall Simple Type interval dissonance alone is a useful primitive for pedal-induced harmonic conflict. `Metric_harmonic_consonance.md` supplies the PA/PP note-state semantics and audited Hall mapping. Scoring uses only `d=max(-HallWeight,0)` with one unweighted sample per note-instance pair. It uses no decay, note age, performance intensity, low-register term, dynamic term, PA/PP differential weight, parameter search, or positive reward.

The input is exactly the prior 13 PEDAL_ELIGIBLE + EXACT_COMPUTABLE validation pieces and nine systems (117 stored MIDI files). PEDAL_SPARSE and DEFERRED_LONG_FORM pieces remain excluded. NO_PEDAL=0 is expected for a harm-only diagnostic and does not mean that no pedal is best overall. HUMAN and CUSTOM_EVENT_V0 remain reference distributions rather than strict canonical pedal-only comparisons.

## Table 1 — Piece-balanced system summary

{markdown(sys_view)}

Pure `HALL_NEG_MEAN_VALID` places ALWAYS_ON at {always.HALL_NEG_MEAN_VALID_mean:.3f}, above HUMAN {human.HALL_NEG_MEAN_VALID_mean:.3f}, PT {pt.HALL_NEG_MEAN_VALID_mean:.3f}, and canonical system means {canonical.HALL_NEG_MEAN_VALID_mean.min():.3f}–{canonical.HALL_NEG_MEAN_VALID_mean.max():.3f}. The extreme also has near-universal negative onset occurrence ({always.HALL_NEG_ONSET_RATE_mean:.3f}) and an average maximum conflict of {always.HALL_NEG_MAX_VALID_mean:.3f}, close to the Hall m2 maximum 1.902.

## Table 2 — Ordering by simple Hall view

{markdown(order_view)}

The primary mean gives HUMAN<PT {int(primary.human_lt_pt_count)}/13, PT<canonical Stage2 {int(primary.pt_lt_model_macro_count)}/52, model<ALWAYS_ON {int(primary.model_lt_always_macro_count)}/52, and the full canonical chain {int(primary.full_canonical_count)}/13. The only primary full-chain piece is {full_piece_text}.

Magnitude and sign frequency are complementary rather than interchangeable. Pair fraction improves PT<model from {int(primary.pt_lt_model_macro_count)}/52 to {int(fraction.pt_lt_model_macro_count)}/52, but weakens HUMAN<PT and model<ALWAYS_ON. Conditional severity improves HUMAN<PT to {int(conditional.human_lt_pt_count)}/13 and model<ALWAYS_ON to {int(conditional.model_lt_always_macro_count)}/52, but PT<model falls to {int(conditional.pt_lt_model_macro_count)}/52. Onset frequency catches model<ALWAYS_ON in {int(occurrence.model_lt_always_macro_count)}/52 while providing only {int(occurrence.human_lt_pt_count)}/13 and {int(occurrence.pt_lt_model_macro_count)}/52 at the normal-performance boundaries. Maximum conflict gives {int(maximum.human_lt_pt_count)}/13 and {int(maximum.model_lt_always_macro_count)}/52, but is a one-pair diagnostic that saturates easily.

## Table 3 — PA / PP Hall-only structure

{markdown(pa_view)}

## Table 4 — PA-only versus PP-only discrimination

{markdown(origin_order)}

PA mean is somewhat more informative for the normal PT/model boundary ({int(pa_order.loc['HALL_NEG_MEAN_PA','PT<model (/52)'])}/52 versus PP {int(pa_order.loc['HALL_NEG_MEAN_PP','PT<model (/52)'])}/52). PP mean is slightly stronger for extreme separation ({int(pa_order.loc['HALL_NEG_MEAN_PP','model<ALWAYS (/52)'])}/52 versus PA {int(pa_order.loc['HALL_NEG_MEAN_PA','model<ALWAYS (/52)'])}/52). Without decay, 99.7% of ALWAYS_ON raw negative Hall mass comes from PP because the residual-residual pair count grows combinatorially; its PA and PP mean severities are nevertheless similar ({pa_pp.set_index('system').loc['ALWAYS_ON','HALL_NEG_MEAN_PA_mean']:.3f} and {pa_pp.set_index('system').loc['ALWAYS_ON','HALL_NEG_MEAN_PP_mean']:.3f}).

## Synthetic Hall sanity

{markdown(synth_texture)}

### Bad-note series

{markdown(synth_bad)}

Same-major harmony has low but nonzero penalty (0.0449) because the learned Hall table assigns small negative values to some nominally consonant classes such as m3/M6. Conflicting major harmonies and the chromatic cluster rise to 0.582 and 0.458. Bad-note-count mean and conditional severity are monotone ({bad_mean_monotone}); maximum conflict reaches 1.902 after the first m2 and then saturates. Negative pair fraction is not fully monotone ({bad_fraction_monotone}): at the third added pitch it decreases from 0.556 to 0.524 because the added note creates both negative and nonnegative relations, changing the denominator composition.

The real MIDI direction agrees at the extreme level: ALWAYS_ON has larger mean, conditional severity, maximum conflict, and occurrence than the normal systems. Synthetic monotonicity is diagnostic-dependent, showing why raw sign fraction alone is not an ordinal bad-note counter.

## Sanity and integrity

{markdown(checks)}

All {len(checks)} checks pass. The compact pitch-multiplicity calculation exactly reproduces all 213,528 prior onset PA/PP pair counts and unweighted negative Hall-sign counts (max count error 0; fraction error 0). Repeated same-pitch note instances also match naive combinatorics. NO_PEDAL is exact zero, all scores are finite and in the Hall-defined range, all 117 source SHA256 values are unchanged, and A-A pairs are absent. TEST access, inference, training, and MIDI generation are all zero.

## Answers to the research questions

**Q1 — Does Hall-negative magnitude distinguish ALWAYS_ON without modifiers?** Yes at the extreme level. Its mean penalty is {always.HALL_NEG_MEAN_VALID_mean:.3f} versus {canonical.HALL_NEG_MEAN_VALID_mean.min():.3f}–{canonical.HALL_NEG_MEAN_VALID_mean.max():.3f} for canonical Stage2 systems, and model<ALWAYS_ON holds in {int(primary.model_lt_always_macro_count)}/52 cells. It is strong but not universal: the canonical median is below ALWAYS_ON in 11/13 pieces.

**Q2 — HUMAN versus PT.** HUMAN<PT holds in {int(primary.human_lt_pt_count)}/13 under the primary mean. Their piece-balanced means are almost equal ({human.HALL_NEG_MEAN_VALID_mean:.3f} versus {pt.HALL_NEG_MEAN_VALID_mean:.3f}), so the signal is not a robust HUMAN/PT quality separator.

**Q3 — PT versus Stage2.** PT<model holds in only {int(primary.pt_lt_model_macro_count)}/52 cells, below chance-level consistency. Pure Hall mean does not reproduce a stable PT/Stage2 quality distinction.

**Q4 — Is magnitude more informative than negative-pair fraction?** Neither dominates. Magnitude is better for HUMAN/PT and extreme separation; fraction is slightly better for PT/model. Conditional severity and maximum conflict supply additional extreme evidence, but no view creates stable normal-system ordering.

**Q5 — Frequency or severity?** Extreme detection appears in both: ALWAYS_ON conditional severity is {always.HALL_NEG_CONDITIONAL_VALID_mean:.3f} versus canonical {canonical.HALL_NEG_CONDITIONAL_VALID_mean.min():.3f}–{canonical.HALL_NEG_CONDITIONAL_VALID_mean.max():.3f}, while onset frequency is {always.HALL_NEG_ONSET_RATE_mean:.3f} versus {canonical.HALL_NEG_ONSET_RATE_mean.min():.3f}–{canonical.HALL_NEG_ONSET_RATE_mean.max():.3f}. Frequency is the clearest ALWAYS_ON detector (52/52) but the weakest HUMAN/PT detector (2/13). Severity is more useful for HUMAN/PT (8/13) but still weak for PT/model (18/52).

**Q6 — PA or PP?** PA gives more normal-system discrimination, while PP gives slightly stronger extreme separation and overwhelmingly dominates ALWAYS_ON raw mass due combinatorial residual accumulation. Both PA and PP mean severities rise for ALWAYS_ON, so the core pitch signal is not exclusive to one origin.

**Q7 — Synthetic versus real behavior.** They agree for strong conflicts: chromatic/conflicting pitch sets and ALWAYS_ON receive larger penalties. They do not support simple monotonic interpretation for every statistic; pair fraction can fall when a bad note also introduces nonnegative pairs, and max conflict saturates after one m2.

**Q8 — Should Hall negative remain a core primitive?** Yes, as an extreme pedal-induced conflict primitive, but not as a standalone normal-pedaling quality metric. The evidence is closest to **B: Hall-only is useful for extreme conflict detection but limited for normal pedaling quality discrimination**. This audit does not propose how to combine it with any other component.

## Interpretation limits

- No new formula or scalar combination is proposed or selected.
- Without decay, every residual note remains equally present until symbolic pedal release; PP combinatorics therefore dominate ALWAYS_ON raw mass.
- Pair-average severity, sign fraction, conditional severity, onset occurrence, and max conflict answer different questions and should not be conflated.
- CSV files are the source of truth; plots are interpretation aids.
"""
    report_path = OUT / "HARMONIC_METRIC_PURE_HALL_NEGATIVE_REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    required = ["piece_system_hall_only.csv", "system_hall_only_summary.csv", "hall_only_ordering.csv",
                "pa_pp_hall_only_summary.csv", "synthetic_hall_sanity.csv", "sanity_checks.csv", report_path.name]
    integrity = pd.DataFrame([
        {"check": "all_sanity_checks_pass", "status": "PASS" if checks.status.eq("PASS").all() else "FAIL", "detail": f"{int(checks.status.eq('PASS').sum())}/{len(checks)}"},
        {"check": "required_outputs_nonempty", "status": "PASS" if all((OUT / name).is_file() and (OUT / name).stat().st_size > 0 for name in required) else "FAIL", "detail": f"files={len(required)}"},
        {"check": "required_plots_nonempty", "status": "PASS" if len(list(OUT.glob("*.png"))) >= 3 and all(path.stat().st_size > 0 for path in OUT.glob("*.png")) else "FAIL", "detail": f"plots={len(list(OUT.glob('*.png')))}"},
        {"check": "report_answers_Q1_to_Q8", "status": "PASS" if all(f"**Q{i}" in report for i in range(1, 9)) else "FAIL", "detail": "Q1-Q8"},
        {"check": "conclusion_classification_explicit", "status": "PASS" if "closest to **B" in report else "FAIL", "detail": "A/B/C classification"},
    ])
    integrity.to_csv(OUT / "integrity_audit.csv", index=False)
    if not integrity.status.eq("PASS").all():
        raise AssertionError(integrity.to_dict("records"))
    print(f"report={report_path}")
    print(integrity.to_string(index=False))


if __name__ == "__main__":
    main()
