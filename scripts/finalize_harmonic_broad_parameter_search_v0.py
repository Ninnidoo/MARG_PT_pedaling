#!/usr/bin/env python3
"""Finalize the broad harmonic search without selecting new parameters."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import search_harmonic_metric_broad_parameters_v0 as core


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/harmonic_metric_broad_parameter_search_v0"


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for values in frame.loc[:, columns].itertuples(index=False, name=None):
        cells = []
        for value in values:
            if isinstance(value, (float, np.floating)):
                cells.append("NA" if math.isnan(float(value)) else f"{float(value):.6f}")
            else:
                cells.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def candidate_failure_diagnostics() -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = pd.read_csv(OUT / "final_candidate_configs.csv")
    pieces = pd.read_csv(OUT / "per_piece_ordering.csv")
    selected = pieces[pieces.config_id.isin(candidates.config_id)].copy()
    selected["extreme_lt_canonical_model"] = selected.extreme_upper_score < selected.M_canonical
    selected["canonical_model_lt_pt"] = selected.M_canonical < selected.ORIGINAL_PT
    selected["extreme_lt_all_model"] = selected.extreme_upper_score < selected.M_all
    selected["all_model_lt_pt"] = selected.M_all < selected.ORIGINAL_PT
    selected["pt_lt_human"] = selected.ORIGINAL_PT < selected.HUMAN
    selected["canonical_failed_inequalities"] = selected.apply(
        lambda row: ";".join(
            name
            for name, passed in (
                ("E<M_CANONICAL", row.extreme_lt_canonical_model),
                ("M_CANONICAL<PT", row.canonical_model_lt_pt),
                ("PT<HUMAN", row.pt_lt_human),
            )
            if not passed
        )
        or "NONE",
        axis=1,
    )
    selected["all_model_failed_inequalities"] = selected.apply(
        lambda row: ";".join(
            name
            for name, passed in (
                ("E<M_ALL", row.extreme_lt_all_model),
                ("M_ALL<PT", row.all_model_lt_pt),
                ("PT<HUMAN", row.pt_lt_human),
            )
            if not passed
        )
        or "NONE",
        axis=1,
    )
    selected.to_csv(OUT / "candidate_failure_details.csv", index=False)

    rows = []
    for (config_id, role), group in selected.groupby(["config_id", "subset_role"], sort=False):
        rows.append(
            {
                "config_id": config_id,
                "subset_role": role,
                "piece_count": len(group),
                "E_lt_M_canonical_rate": group.extreme_lt_canonical_model.mean(),
                "M_canonical_lt_PT_rate": group.canonical_model_lt_pt.mean(),
                "PT_lt_HUMAN_rate": group.pt_lt_human.mean(),
                "E_lt_M_all_rate": group.extreme_lt_all_model.mean(),
                "M_all_lt_PT_rate": group.all_model_lt_pt.mean(),
                "canonical_full_chain_rate": group.canonical_chain_pass.mean(),
                "all_full_chain_rate": group.all_model_chain_pass.mean(),
                "canonical_failure_piece_ids": ";".join(group.loc[~group.canonical_chain_pass, "piece_id"]),
                "all_failure_piece_ids": ";".join(group.loc[~group.all_model_chain_pass, "piece_id"]),
            }
        )
    aggregate = pd.DataFrame(rows)
    aggregate.to_csv(OUT / "candidate_failure_summary.csv", index=False)

    backbone_id = candidates.loc[candidates.reference_distance_grid_L1.eq(0), "config_id"].iloc[0]
    mask = candidates.config_id.eq(backbone_id)
    candidates.loc[mask, "candidate_type"] = "reference_near_heldout_stable"
    candidates.loc[mask, "selection_reason"] = "Hall+decay backbone; best held-out behavior within the pre-frozen pool"
    summaries = aggregate.pivot(index="config_id", columns="subset_role")
    for role, prefix in (("PARAMETER_SEARCH", "search"), ("HELD_OUT_CHECK", "heldout")):
        candidates[f"{prefix}_E_lt_M_rate"] = candidates.config_id.map(summaries["E_lt_M_canonical_rate"][role])
        candidates[f"{prefix}_M_lt_PT_rate"] = candidates.config_id.map(summaries["M_canonical_lt_PT_rate"][role])
        candidates[f"{prefix}_PT_lt_HUMAN_rate"] = candidates.config_id.map(summaries["PT_lt_HUMAN_rate"][role])
        candidates[f"{prefix}_failure_piece_ids"] = candidates.config_id.map(summaries["canonical_failure_piece_ids"][role])
    candidates.to_csv(OUT / "final_candidate_configs.csv", index=False)
    return candidates, aggregate


def mechanism_diagnostics() -> tuple[pd.DataFrame, pd.DataFrame]:
    mechanism = pd.read_csv(OUT / "parameter_mechanism_summary.csv")
    ordering = mechanism[mechanism.scope.eq("ordering")]
    ranges = ordering.groupby("parameter", as_index=False).mean_individual_model_compliance.agg(["min", "max"]).reset_index()
    ranges["compliance_range"] = ranges["max"] - ranges["min"]
    ranges = ranges.sort_values("compliance_range", ascending=False)
    rows: list[dict[str, object]] = []
    for row in ranges.itertuples():
        rows.append({"diagnostic": "marginal_individual_compliance_range", "parameter": row.parameter,
                     "system": "ALL", "from_value": np.nan, "to_value": np.nan,
                     "value": row.compliance_range,
                     "interpretation": "secondary ordering sensitivity; full-chain rate was zero throughout"})

    system_scores = mechanism[mechanism.scope.eq("system_score")]
    for parameter, start, end in (("eta_PP", 0.9, 0.0), ("alpha_decay", 0.5, 2.0)):
        subset = system_scores[system_scores.parameter.eq(parameter)]
        first = subset[np.isclose(subset.value, start)].set_index("system").mean_H_piece
        last = subset[np.isclose(subset.value, end)].set_index("system").mean_H_piece
        for system in sorted(first.index.intersection(last.index)):
            rows.append({"diagnostic": "marginal_mean_H_change", "parameter": parameter, "system": system,
                         "from_value": start, "to_value": end, "value": last[system] - first[system],
                         "interpretation": "positive means mean H_piece increased at the endpoint"})

    grid = pd.read_csv(OUT / "coarse_parameter_grid.csv")
    audit = pd.read_csv(OUT / "pedal_activity_audit.csv")
    split = pd.read_csv(OUT / "split_manifest.csv")
    search_ids = split.loc[split.split_role.eq("PARAMETER_SEARCH"), "piece_id"].tolist()
    contrasts = [(1.0, .9, 48, 0., 0.), (1.0, .9, 48, 1., 0.),
                 (1.0, .9, 42, 1., 0.), (1.0, .9, 60, 1., 0.), (1.0, .9, 48, 0., .5)]
    masks = []
    for alpha, eta, pivot, beta, kappa in contrasts:
        masks.append(grid[np.isclose(grid.alpha_decay, alpha) & np.isclose(grid.eta_PP, eta)
                          & grid.m0.eq(pivot) & np.isclose(grid.beta_low, beta)
                          & np.isclose(grid.kappa_dyn, kappa)])
    chosen = pd.concat(masks, ignore_index=True)
    scores = core.score_search(search_ids, OUT / "sufficient_statistics/search", chosen.reset_index(drop=True), audit)
    descriptors = audit.set_index("piece_id").low_note_fraction

    def cid(pivot: int, beta: float, kappa: float) -> str:
        return chosen.loc[chosen.m0.eq(pivot) & np.isclose(chosen.beta_low, beta)
                          & np.isclose(chosen.kappa_dyn, kappa), "config_id"].iloc[0]

    contrast_rows = []
    for label, left, right in (
        ("beta_0_to_1_at_m0_48", cid(48, 0., 0.), cid(48, 1., 0.)),
        ("m0_42_to_60_at_beta_1", cid(42, 1., 0.), cid(60, 1., 0.)),
        ("kappa_0_to_0.5", cid(48, 0., 0.), cid(48, 0., .5)),
    ):
        wide = scores[scores.config_id.isin([left, right])].pivot(index=["piece_id", "system"],
                                                                  columns="config_id", values="H_piece")
        wide["delta_H"] = wide[right] - wide[left]
        wide = wide.reset_index()
        wide["contrast"] = label
        wide["low_note_fraction"] = wide.piece_id.map(descriptors)
        contrast_rows.append(wide[["contrast", "piece_id", "system", "low_note_fraction", "delta_H"]])
        if label.startswith(("beta", "m0")):
            for system, group in wide.groupby("system"):
                rows.append({"diagnostic": "low_fraction_vs_absolute_score_change_pearson_r",
                             "parameter": "beta_low" if label.startswith("beta") else "m0",
                             "system": system, "from_value": 0. if label.startswith("beta") else 42.,
                             "to_value": 1. if label.startswith("beta") else 60.,
                             "value": group.low_note_fraction.corr(group.delta_H.abs()),
                             "interpretation": "positive means larger absolute effect in low-note-heavy pieces"})
    pd.concat(contrast_rows, ignore_index=True).to_csv(OUT / "parameter_contrast_per_piece.csv", index=False)
    diagnostics = pd.DataFrame(rows)
    diagnostics.to_csv(OUT / "parameter_mechanism_diagnostics.csv", index=False)
    return diagnostics, ranges


def integrity_audit() -> pd.DataFrame:
    checks: list[dict[str, object]] = []
    def record(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "status": "PASS" if passed else "FAIL", "detail": detail})

    provenance = json.loads((OUT / "provenance.json").read_text(encoding="utf-8"))
    grid = pd.read_csv(OUT / "coarse_parameter_grid.csv")
    results = pd.read_csv(OUT / "coarse_search_results.csv")
    scores = pd.read_csv(OUT / "per_system_scores_top_configs.csv")
    exact = pd.read_csv(OUT / "sufficient_statistics_exactness_audit.csv")
    extremes = pd.read_csv(OUT / "reusable_extremes_manifest.csv")
    identities = pd.read_csv(OUT / "midi_identity_audit.csv")
    record("coarse_grid_4320_unique", len(grid) == 4320 and grid.config_id.nunique() == 4320, f"rows={len(grid)}")
    record("coarse_results_4320_unique", len(results) == 4320 and results.config_id.nunique() == 4320, f"rows={len(results)}")
    numeric = results.select_dtypes(include=[np.number]).to_numpy()
    record("all_coarse_metrics_finite", bool(np.isfinite(numeric).all()), f"numeric_cells={numeric.size}")
    no_pedal = scores[scores.system.eq("NO_PEDAL")].H_piece
    record("NO_PEDAL_exact_zero", bool(no_pedal.eq(0.).all()), f"cells={len(no_pedal)}")
    record("optimized_equals_naive", bool(exact.status.eq("PASS").all() and exact.absolute_difference.max() <= 1e-12),
           f"max_abs_diff={exact.absolute_difference.max():.3e}")
    flags = ["non_CC64_exact_identity", "note_exact_identity"]
    extreme_ok = all(extremes[column].astype(str).str.lower().eq("true").all() for column in flags)
    pilot_manifest = pd.read_csv(
        ROOT / "analysis/harmonic_metric_kdyn_blow_pilot_v0/reusable_extremes/manifest.csv"
    ).rename(columns={"generated_sha256": "pilot_extreme_sha256"})
    overlap = extremes.merge(
        pilot_manifest[["piece_id", "variant", "source_sha256", "pilot_extreme_sha256"]],
        on=["piece_id", "variant"], suffixes=("", "_pilot")
    )
    sha_ok = (
        overlap.source_sha256.eq(overlap.source_sha256_pilot).all()
        and overlap.extreme_sha256.eq(overlap.pilot_extreme_sha256).all()
    )
    record("reusable_extremes_SHA_and_identity", bool(extreme_ok and sha_ok),
           f"rows={len(extremes)}; pilot_overlap={len(overlap)}")
    canonical_rows = identities[identities.expected_canonical_identity.astype(str).str.lower().eq("true")]
    identity_ok = (
        identities.status.eq("PASS").all()
        and canonical_rows.canonical_stage1_non_CC64_exact.astype(str).str.lower().eq("true").all()
        and canonical_rows.canonical_stage1_note_exact.astype(str).str.lower().eq("true").all()
    )
    record("canonical_non_CC64_identity", bool(identity_ok),
           f"canonical_rows={len(canonical_rows)}; all_status_pass={identities.status.eq('PASS').all()}")
    record("split_fixed_before_scoring", provenance["split_fixed_before_scoring"] is True, "provenance=true")
    record("heldout_not_used_for_tuning", provenance["heldout_used_for_parameter_generation"] is False, "provenance=false")
    record("TEST_access_zero", provenance["test_set_access_count"] == 0, "count=0")
    record("new_inference_zero", provenance["new_inference_count"] == 0, "count=0")
    record("training_zero", provenance["training_steps"] == 0, "count=0")
    required = ["pedal_activity_audit.csv", "eligible_pieces.csv", "split_manifest.csv", "coarse_parameter_grid.csv",
                "coarse_search_results.csv", "top_coarse_configs.csv", "per_piece_ordering.csv",
                "per_system_scores_top_configs.csv", "parameter_mechanism_summary.csv", "heldout_results.csv",
                "refinement_results.csv", "final_candidate_configs.csv", "HARMONIC_METRIC_BROAD_PARAMETER_SEARCH_REPORT.md"]
    plots = ["top_config_system_distributions.png", "target_ordering_visualization.png",
             "parameter_partial_dependence.png", "chain_pass_rate_alpha_eta_heatmap.png",
             "search_vs_heldout.png", "parameter_reference_distance.png"]
    missing = [name for name in required if not (OUT / name).is_file()]
    missing += [f"plots/{name}" for name in plots if not (OUT / "plots" / name).is_file()]
    record("required_outputs_present", not missing, "missing=" + ",".join(missing))
    frame = pd.DataFrame(checks)
    frame.to_csv(OUT / "integrity_audit.csv", index=False)
    if not frame.status.eq("PASS").all():
        raise AssertionError(frame[frame.status.eq("FAIL")].to_dict("records"))
    return frame


def update_report(candidates: pd.DataFrame, failures: pd.DataFrame, diagnostics: pd.DataFrame,
                  ranges: pd.DataFrame, integrity: pd.DataFrame) -> None:
    path = OUT / "HARMONIC_METRIC_BROAD_PARAMETER_SEARCH_REPORT.md"
    text = path.read_text(encoding="utf-8")
    best_id = candidates.iloc[0].config_id
    best_fail = failures[failures.config_id.eq(best_id)][["subset_role", "piece_count",
        "E_lt_M_canonical_rate", "M_canonical_lt_PT_rate", "PT_lt_HUMAN_rate", "canonical_full_chain_rate"]]
    compliance = ranges[["parameter", "min", "max", "compliance_range"]].rename(
        columns={"min": "min_compliance", "max": "max_compliance"})
    eta = diagnostics[diagnostics.diagnostic.eq("marginal_mean_H_change") & diagnostics.parameter.eq("eta_PP")].set_index("system").value
    alpha = diagnostics[diagnostics.diagnostic.eq("marginal_mean_H_change") & diagnostics.parameter.eq("alpha_decay")].set_index("system").value
    canonical = ["STANDARD_CE_ARGMAX", "STANDARD_CE_POSTERIOR_MEDIAN", "WEIGHTED_CE_ARGMAX", "HYBRID_REGRESSION_ONLY"]
    mechanism_block = "\n".join([
        "## Mechanism and failure analysis", "",
        "All 4,320 SEARCH configurations had zero canonical and MODEL_ALL full-chain rate. Therefore Table 7's full-chain effect ranges are correctly all zero, but they cannot rank parameter importance. The secondary diagnostic below uses individual-model compliance instead.", "",
        "### Secondary ordering sensitivity", "", markdown_table(compliance, list(compliance.columns)), "",
        f"`kappa_dyn` has the largest marginal individual-compliance range ({ranges.iloc[0].compliance_range:.3f}), followed by `eta_PP` ({ranges.set_index('parameter').loc['eta_PP', 'compliance_range']:.3f}) and `alpha_decay` ({ranges.set_index('parameter').loc['alpha_decay', 'compliance_range']:.3f}). None changes the full-chain conclusion.", "",
        "### Dominant inequality failures", "", markdown_table(best_fail, list(best_fail.columns)), "",
        "For the lexicographic SEARCH best, `E<M` fails on 1/9 SEARCH pieces, `M<PT` fails on 3/9, and `PT<HUMAN` fails on 6/9. On HELD_OUT, `M<PT` fails on 3/4 and `PT<HUMAN` fails on all 4. Thus the primary structural bottlenecks are PT-versus-model and HUMAN-versus-PT, not suppression of the extremes.", "",
        "### Parameter mechanisms", "",
        f"Lowering `eta_PP` from 0.9 to 0 raises mean ALWAYS_ON by {eta['ALWAYS_ON']:+.6f}, raises the four canonical model means by {eta[canonical].mean():+.6f}, changes ORIGINAL_PT by {eta['ORIGINAL_PT']:+.6f}, and leaves HUMAN nearly unchanged ({eta['HUMAN']:+.6f}). This sometimes improves PT>model compliance, but it also moves ALWAYS_ON upward and never creates a full-chain region.", "",
        f"Increasing `alpha_decay` from 0.5 to 2.0 lifts every nonzero system: ALWAYS_ON {alpha['ALWAYS_ON']:+.6f}, HUMAN {alpha['HUMAN']:+.6f}, ORIGINAL_PT {alpha['ORIGINAL_PT']:+.6f}, and the canonical-model mean {alpha[canonical].mean():+.6f}. ALWAYS_ON is less sensitive in absolute H_piece than the performances, contrary to the hypothesized special long-residual sensitivity.", "",
        "The per-piece low-register contrasts and their correlation with non-pedal low-note fraction are stored in `parameter_contrast_per_piece.csv` and `parameter_mechanism_diagnostics.csv`. Effects vary by system and do not rescue the chain. The kappa contrast likewise changes secondary separation but not full-chain feasibility.", "",
        "Optional refinement was not performed; the predeclared trigger required both SEARCH chain rates >=0.75 and individual compliance >=0.75. This prevents local search when the formula family does not show a robust coarse capability signal.", "",
        "PEDAL_SPARSE threshold sensitivity for frozen candidates is in `pedal_sparse_sensitivity.csv`; coverage columns expose deferred long-form pieces at every threshold.",
    ])
    start, end = text.index("## Mechanism and failure analysis"), text.index("## Final candidate configurations")
    text = text[:start] + mechanism_block + "\n\n" + text[end:]

    columns = ["config_id", "candidate_type", "alpha_decay", "eta_PP", "m0", "beta_low", "kappa_dyn",
               "search_canonical_chain_pass_rate", "heldout_canonical_chain_pass_rate",
               "reference_distance_grid_L1", "selection_reason"]
    start, end = text.index("## Final candidate configurations"), text.index("## Answers to the research questions")
    block = "\n".join(["## Final candidate configurations (not final parameters)", "",
                        markdown_table(candidates, columns), "",
                        "These are listening controls spanning reference, strongest-SEARCH, and distinct regions—not successful calibrated optima. `candidate_failure_details.csv` records the failed systems/pieces for every candidate and subset.", "",])
    text = text[:start] + block + text[end:]

    q_start, q_end = text.index("## Answers to the research questions"), text.index("## Files, provenance, and caveats")
    q = "\n".join(["## Answers to the research questions", "",
        "**Q1. Formula-family capability.** No robust target-ordering region was found. Every one of the 4,320 SEARCH configurations had canonical and MODEL_ALL full-chain rate 0/9.", "",
        "**Q2. Held-out maintenance.** No SEARCH success existed to generalize. Among configurations frozen before held-out scoring, only the Hall+decay backbone passed on one of four held-out pieces (0.25); the other candidates passed zero. This isolated held-out pass is not evidence of stable ordering.", "",
        f"**Q3. Important parameters.** By secondary individual-model compliance, kappa_dyn had the largest marginal range ({ranges.iloc[0].compliance_range:.3f}), eta_PP was next ({ranges.set_index('parameter').loc['eta_PP', 'compliance_range']:.3f}), and alpha_decay followed ({ranges.set_index('parameter').loc['alpha_decay', 'compliance_range']:.3f}). eta_PP materially trades PT against models, while longer decay lifts all systems; neither produces the full chain.", "",
        f"**Q4. Reference distance.** The lexicographic SEARCH best is {int(candidates.iloc[0].reference_distance_grid_L1)} coarse grid steps from the Hall+decay backbone, mainly because eta_PP falls to 0, m0 rises to 60, and kappa rises to 0.3. That movement still yields no full-chain pass, so moving far from the reference is not justified by this audit.", "",
        "**Q5. Structural failure.** At the SEARCH best, PT<HUMAN fails most often (6/9), followed by M_canonical<PT (3/9) and E<M_canonical (1/9). On held-out, PT<HUMAN fails 4/4 and M_canonical<PT fails 3/4. The family generally separates models from extremes, but does not consistently express the proposed quality order among model, PT, and HUMAN.", "",
        "**Q6. Listening candidates.** Four pre-frozen configurations are retained as diagnostic listening controls: the literature backbone, the strongest SEARCH secondary-ordering point, and two distinct regions. None is a final parameter or a successful target-ordering solution; candidate-specific failures are in `candidate_failure_details.csv`.", "",])
    text = text[:q_start] + q + text[q_end:]
    text = text.rstrip() + f"\n- Final integrity audit: {len(integrity)}/{len(integrity)} PASS (`integrity_audit.csv`).\n"
    text += "- Four eligible long-form pieces were deferred by the predeclared exact pair-onset work cap; no approximation or pruning was substituted. Claims therefore apply to 13 exact-scored eligible pieces (9 SEARCH, 4 HELD_OUT), with coverage disclosed in all sensitivity tables.\n"
    path.write_text(text, encoding="utf-8")


def main() -> None:
    candidates, failures = candidate_failure_diagnostics()
    diagnostics, ranges = mechanism_diagnostics()
    integrity = integrity_audit()
    update_report(candidates, failures, diagnostics, ranges, integrity)
    print(json.dumps({"status": "finalized", "candidate_count": len(candidates),
                      "integrity_pass": int(integrity.status.eq("PASS").sum()),
                      "integrity_total": len(integrity)}, indent=2))


if __name__ == "__main__":
    main()
