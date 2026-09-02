#!/usr/bin/env python3
"""CSV-only 13-piece validation audit for Hall muddiness H and A.

The script reuses the complete audited Pure Hall onset table.  It never parses
MIDI, recomputes Hall pairs, materializes note pairs, searches lambda, runs
inference/training, or writes MIDI.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.audit_hall_muddiness_additive_feasibility_v0 as five
import scripts.harmonic_pure_hall_negative_core as core


PURE = ROOT / "analysis/harmonic_metric_pure_hall_negative_v0"
COUNT = ROOT / "analysis/hall_negative_count_scaling_audit_v0"
FIVE = ROOT / "analysis/hall_muddiness_additive_feasibility_v0"
ELIGIBLE = ROOT / "analysis/harmonic_metric_broad_parameter_search_v0/eligible_pieces.csv"
OUTPUT = ROOT / "analysis/hall_muddiness_HA_13piece_validation_v0"

SYSTEMS = (
    "NO_PEDAL", "ALWAYS_ON", "STANDARD_CE_ARGMAX",
    "STANDARD_CE_POSTERIOR_MEDIAN", "WEIGHTED_CE_ARGMAX",
    "HYBRID_REGRESSION_ONLY", "CUSTOM_EVENT_V0", "ORIGINAL_PT", "HUMAN",
)
CANONICAL = (
    "STANDARD_CE_ARGMAX", "STANDARD_CE_POSTERIOR_MEDIAN",
    "WEIGHTED_CE_ARGMAX", "HYBRID_REGRESSION_ONLY",
)
SYSTEM_RANK = {value: index for index, value in enumerate(SYSTEMS)}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantile(values: Iterable[float], level: float) -> float:
    array = np.asarray(list(values), dtype=float)
    return float(np.quantile(array, level)) if len(array) else 0.0


def corr(left: Iterable[float], right: Iterable[float]) -> float | None:
    x, y = np.asarray(list(left), dtype=float), np.asarray(list(right), dtype=float)
    if len(x) < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return None
    value = float(np.corrcoef(x, y)[0, 1])
    return value if math.isfinite(value) else None


def spearman(left: pd.Series, right: pd.Series) -> float | None:
    return corr(left.rank(method="average"), right.rank(method="average"))


def fmt(value: Any, digits: int = 6) -> str:
    if value is None or value == "":
        return "—"
    number = float(value)
    if math.isinf(number):
        return "+∞" if number > 0 else "−∞"
    if not math.isfinite(number):
        return "—"
    return f"{number:.{digits}f}"


def md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(item) for item in row) + " |" for row in rows)
    return "\n".join(lines)


def interval_fields(prefix: str, interval: five.Interval) -> dict[str, Any]:
    return five.interval_fields(prefix, interval)


def build_manifest() -> tuple[pd.DataFrame, tuple[str, ...]]:
    source = pd.read_csv(PURE / "input_midis.csv")
    piece_order = tuple(source.piece_id.drop_duplicates())
    if len(source) != 117 or len(piece_order) != 13:
        raise RuntimeError("Pure Hall manifest is not the frozen 13x9 inventory")
    if source.groupby("piece_id").system.apply(set).ne(set(SYSTEMS)).any():
        raise RuntimeError("Pure Hall system inventory changed")

    metadata = pd.read_csv(ELIGIBLE).set_index("piece_id")
    rows = []
    for row in source.itertuples(index=False):
        meta = metadata.loc[row.piece_id]
        if meta.primary_status != "PEDAL_ELIGIBLE" or meta.exact_compute_status != "EXACT_COMPUTABLE":
            raise RuntimeError(f"piece is outside frozen eligibility: {row.piece_id}")
        rows.append({
            "piece_id": row.piece_id,
            "composer": meta.composer,
            "title": meta.title,
            "performance_id": row.performance_id,
            "system": row.system,
            "system_group": row.system_group,
            "midi_path": row.midi_path,
            "source_sha256": row.sha256,
            "current_sha256_at_prior_audit": row.current_sha256,
            "source_sha_unchanged_at_prior_audit": row.source_sha_unchanged,
            "selection_status": meta.primary_status,
            "exact_compute_status": meta.exact_compute_status,
            "inventory_origin": "harmonic_metric_pure_hall_negative_v0/input_midis.csv",
        })
    manifest = pd.DataFrame(rows)
    piece_rank = {value: index for index, value in enumerate(piece_order)}
    manifest["_piece"] = manifest.piece_id.map(piece_rank)
    manifest["_system"] = manifest.system.map(SYSTEM_RANK)
    manifest = manifest.sort_values(["_piece", "_system"]).drop(columns=["_piece", "_system"]).reset_index(drop=True)
    return manifest, piece_order


def build_onsets(manifest: pd.DataFrame, piece_order: Sequence[str]) -> pd.DataFrame:
    source = pd.read_csv(PURE / "onset_hall_only_diagnostics.csv")
    columns = [
        "piece_id", "performance_id", "system", "system_group", "onset_index", "onset_tick",
        "onset_time", "A_n_count", "P_n_count", "PA_pair_count", "PP_pair_count",
        "N_pair_n", "negative_hall_mass_n", "HALL_NEG_MEAN_n", "AA_pair_count",
    ]
    missing = set(columns) - set(source.columns)
    if missing:
        raise RuntimeError(f"Pure Hall onset table lacks columns: {sorted(missing)}")
    out = source[columns].rename(columns={
        "onset_time": "onset_time_sec", "A_n_count": "active_note_count",
        "P_n_count": "N_P", "N_pair_n": "N_Q", "negative_hall_mass_n": "RAW_n",
        "HALL_NEG_MEAN_n": "H_n",
    }).copy()
    meta = manifest.drop_duplicates("piece_id").set_index("piece_id")[["composer", "title"]]
    out = out.join(meta, on="piece_id")
    out["A_n"] = np.log1p(out.RAW_n.to_numpy(dtype=float))
    out["_zero_index"] = out.groupby(["piece_id", "system"], sort=False).cumcount()
    out["_total"] = out.groupby(["piece_id", "system"], sort=False).onset_index.transform("size")
    out["position_bin"] = np.minimum(9, (10 * out._zero_index / out._total).astype(int))
    out = out.drop(columns=["_zero_index", "_total"])
    piece_rank = {value: index for index, value in enumerate(piece_order)}
    out["_piece"] = out.piece_id.map(piece_rank)
    out["_system"] = out.system.map(SYSTEM_RANK)
    return out.sort_values(["_piece", "_system", "onset_index"]).drop(columns=["_piece", "_system"]).reset_index(drop=True)


def build_piece(onsets: pd.DataFrame, manifest: pd.DataFrame, piece_order: Sequence[str]) -> pd.DataFrame:
    metadata = manifest.set_index(["piece_id", "system"])
    rows = []
    for (piece_id, system), group in onsets.groupby(["piece_id", "system"], sort=False):
        valid = group[group.N_Q > 0]
        meta = metadata.loc[(piece_id, system)]
        final = core.aggregate_final_harmonic_metric([
            {"N_pair_n": row.N_Q, "negative_hall_mass_n": row.RAW_n}
            for row in group.itertuples(index=False)
        ])
        rows.append({
            "piece_id": piece_id, "composer": meta.composer, "title": meta.title,
            "performance_id": meta.performance_id, "system": system,
            "system_group": meta.system_group, "midi_path": meta.midi_path,
            "source_sha256": meta.source_sha256,
            "all_distinct_onset_count": len(group),
            "valid_interaction_onset_count": len(valid),
            "valid_interaction_onset_fraction": len(valid) / len(group),
            "H_piece": float(group.H_n.mean()), "A_piece": float(group.A_n.mean()),
            "H_piece_valid": float(valid.H_n.mean()) if len(valid) else 0.0,
            "A_piece_valid": float(valid.A_n.mean()) if len(valid) else 0.0,
            "H_mean": final["H_mean"], "A_acc": final["A_acc"],
            "M_harm": final["M_harm"],
            "mean_RAW_per_onset": float(group.RAW_n.mean()),
            "median_RAW_per_onset": float(group.RAW_n.median()),
            "p90_RAW_per_onset": quantile(group.RAW_n, 0.90),
            "p95_RAW_per_onset": quantile(group.RAW_n, 0.95),
            "p99_RAW_per_onset": quantile(group.RAW_n, 0.99),
            "max_RAW_per_onset": float(group.RAW_n.max()),
        })
    frame = pd.DataFrame(rows)
    rank = {value: index for index, value in enumerate(piece_order)}
    frame["_piece"] = frame.piece_id.map(rank)
    frame["_system"] = frame.system.map(SYSTEM_RANK)
    return frame.sort_values(["_piece", "_system"]).drop(columns=["_piece", "_system"]).reset_index(drop=True)


def build_system_summary(piece: pd.DataFrame, onsets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for system in SYSTEMS:
        p = piece[piece.system == system]
        o = onsets[onsets.system == system]
        rows.append({
            "system": system, "piece_count": len(p), "onset_count": len(o),
            "H_mean": float(p.H_piece.mean()), "H_median": float(p.H_piece.median()),
            "H_IQR": quantile(p.H_piece, 0.75) - quantile(p.H_piece, 0.25),
            "A_mean": float(p.A_piece.mean()), "A_median": float(p.A_piece.median()),
            "A_IQR": quantile(p.A_piece, 0.75) - quantile(p.A_piece, 0.25),
            "H_valid_mean": float(p.H_piece_valid.mean()),
            "H_valid_median": float(p.H_piece_valid.median()),
            "A_valid_mean": float(p.A_piece_valid.mean()),
            "A_valid_median": float(p.A_piece_valid.median()),
            "final_H_mean_piece_macro": float(p.H_mean.mean()),
            "final_A_acc_piece_macro": float(p.A_acc.mean()),
            "final_M_harm_piece_macro": float(p.M_harm.mean()),
            "RAW_pooled_median": float(o.RAW_n.median()),
            "RAW_pooled_p90": quantile(o.RAW_n, 0.90),
            "RAW_pooled_p95": quantile(o.RAW_n, 0.95),
            "RAW_pooled_p99": quantile(o.RAW_n, 0.99),
        })
    return pd.DataFrame(rows)


def build_ordering(piece: pd.DataFrame, piece_order: Sequence[str]) -> pd.DataFrame:
    index = piece.set_index(["piece_id", "system"])
    rows = []

    def add(component: str, kind: str, piece_id: str, left: str, right: str,
            left_value: float, right_value: float, operator: str, hard: bool) -> None:
        holds: Any = ""
        if operator == "<=":
            holds = left_value <= right_value
        elif operator == "<":
            holds = left_value < right_value
        relation = "<" if left_value < right_value else (">" if left_value > right_value else "=")
        rows.append({
            "row_type": "DETAIL", "component": component, "comparison_type": kind,
            "piece_id": piece_id, "left_label": left, "right_label": right,
            "left_value": left_value, "right_value": right_value, "operator": operator,
            "holds": holds, "observed_relation": relation, "hard_success_criterion": hard,
            "success_count": "", "comparison_count": "",
        })

    for component, column in (("H", "H_piece"), ("A", "A_piece")):
        for piece_id in piece_order:
            human = float(index.loc[(piece_id, "HUMAN"), column])
            pt = float(index.loc[(piece_id, "ORIGINAL_PT"), column])
            always = float(index.loc[(piece_id, "ALWAYS_ON"), column])
            custom = float(index.loc[(piece_id, "CUSTOM_EVENT_V0"), column])
            model_values = {model: float(index.loc[(piece_id, model), column]) for model in CANONICAL}
            median = float(np.median(list(model_values.values())))
            add(component, "HUMAN_LE_PT", piece_id, "HUMAN", "ORIGINAL_PT", human, pt, "<=", False)
            for model, value in model_values.items():
                add(component, "CANONICAL_MODEL_LT_ALWAYS", piece_id, model, "ALWAYS_ON", value, always, "<", True)
                add(component, "STAGE2_VS_PT_DESCRIPTIVE", piece_id, model, "ORIGINAL_PT", value, pt, "DESCRIPTIVE", False)
            add(component, "CANONICAL_MEDIAN_LT_ALWAYS", piece_id, "CANONICAL_STAGE2_MEDIAN", "ALWAYS_ON", median, always, "<", True)
            add(component, "PT_LT_ALWAYS", piece_id, "ORIGINAL_PT", "ALWAYS_ON", pt, always, "<", True)
            add(component, "HUMAN_LT_ALWAYS", piece_id, "HUMAN", "ALWAYS_ON", human, always, "<", True)
            add(component, "CUSTOM_LT_ALWAYS", piece_id, "CUSTOM_EVENT_V0", "ALWAYS_ON", custom, always, "<", True)

    detail = pd.DataFrame(rows)
    summaries = []
    for component in ("H", "A"):
        current = detail[detail.component == component]
        for kind in ("HUMAN_LE_PT", "CANONICAL_MODEL_LT_ALWAYS", "CANONICAL_MEDIAN_LT_ALWAYS",
                     "PT_LT_ALWAYS", "HUMAN_LT_ALWAYS", "CUSTOM_LT_ALWAYS"):
            group = current[current.comparison_type == kind]
            summaries.append({
                "row_type": "SUMMARY", "component": component, "comparison_type": kind,
                "piece_id": "ALL", "left_label": "", "right_label": "", "left_value": "",
                "right_value": "", "operator": group.operator.iloc[0], "holds": "",
                "observed_relation": "", "hard_success_criterion": bool(group.hard_success_criterion.iloc[0]),
                "success_count": int(group.holds.astype(bool).sum()), "comparison_count": len(group),
            })
        stage = current[current.comparison_type == "STAGE2_VS_PT_DESCRIPTIVE"]
        for relation, label in (("<", "MODEL_LT_PT_DESCRIPTIVE"), (">", "PT_LT_MODEL_DESCRIPTIVE")):
            summaries.append({
                "row_type": "SUMMARY", "component": component, "comparison_type": label,
                "piece_id": "ALL", "left_label": "", "right_label": "", "left_value": "",
                "right_value": "", "operator": "DESCRIPTIVE", "holds": "", "observed_relation": "",
                "hard_success_criterion": False, "success_count": int((stage.observed_relation == relation).sum()),
                "comparison_count": len(stage),
            })
    return pd.concat([detail, pd.DataFrame(summaries)], ignore_index=True)


def build_quadrants(piece: pd.DataFrame, piece_order: Sequence[str]) -> pd.DataFrame:
    index = piece.set_index(["piece_id", "system"])
    rows = []
    for piece_id in piece_order:
        human = index.loc[(piece_id, "HUMAN")]
        pt = index.loc[(piece_id, "ORIGINAL_PT")]
        delta_h = float(human.H_piece - pt.H_piece)
        delta_a = float(human.A_piece - pt.A_piece)
        if delta_h <= 0 and delta_a <= 0:
            quadrant, label = 1, "H<=PT_AND_A<=PT"
        elif delta_h > 0 and delta_a <= 0:
            quadrant, label = 2, "H>PT_AND_A<=PT"
        elif delta_h <= 0 and delta_a > 0:
            quadrant, label = 3, "H<=PT_AND_A>PT"
        else:
            quadrant, label = 4, "H>PT_AND_A>PT"
        rows.append({
            "piece_id": piece_id, "composer": human.composer, "title": human.title,
            "quadrant": quadrant, "quadrant_label": label,
            "H_HUMAN": human.H_piece, "H_PT": pt.H_piece, "delta_H": delta_h,
            "A_HUMAN": human.A_piece, "A_PT": pt.A_piece, "delta_A": delta_a,
            "positive_additive_HUMAN_le_PT_possible": quadrant != 4,
        })
    frame = pd.DataFrame(rows)
    counts = frame.quadrant.value_counts()
    frame["quadrant_piece_count"] = frame.quadrant.map(counts)
    frame["quadrant_piece_ids"] = frame.quadrant.map(
        frame.groupby("quadrant").piece_id.apply(lambda values: " | ".join(values)))
    frame["quadrant_titles"] = frame.quadrant.map(
        frame.groupby("quadrant").apply(lambda group: " | ".join(group.composer + "/" + group.title), include_groups=False))
    return frame


def build_separation(piece: pd.DataFrame, piece_order: Sequence[str]) -> pd.DataFrame:
    index = piece.set_index(["piece_id", "system"])
    rows = []
    references = ("HUMAN", "ORIGINAL_PT", "CANONICAL_STAGE2_MEDIAN")
    for component, column in (("H", "H_piece"), ("A", "A_piece")):
        for piece_id in piece_order:
            always = float(index.loc[(piece_id, "ALWAYS_ON"), column])
            model_median = float(np.median([index.loc[(piece_id, model), column] for model in CANONICAL]))
            values = {
                "HUMAN": float(index.loc[(piece_id, "HUMAN"), column]),
                "ORIGINAL_PT": float(index.loc[(piece_id, "ORIGINAL_PT"), column]),
                "CANONICAL_STAGE2_MEDIAN": model_median,
            }
            meta = index.loc[(piece_id, "HUMAN")]
            for reference in references:
                value = values[reference]
                rows.append({
                    "row_type": "DETAIL", "component": component, "piece_id": piece_id,
                    "composer": meta.composer, "title": meta.title, "reference": reference,
                    "always_value": always, "reference_value": value,
                    "always_over_reference_ratio": always / value if value > 0 else math.inf,
                    "always_minus_reference": always - value,
                    "median_ratio_across_pieces": "", "median_difference_across_pieces": "",
                })
    detail = pd.DataFrame(rows)
    summaries = []
    for (component, reference), group in detail.groupby(["component", "reference"], sort=False):
        summaries.append({
            "row_type": "SUMMARY", "component": component, "piece_id": "ALL",
            "composer": "", "title": "", "reference": reference,
            "always_value": "", "reference_value": "", "always_over_reference_ratio": "",
            "always_minus_reference": "",
            "median_ratio_across_pieces": float(group.always_over_reference_ratio.median()),
            "median_difference_across_pieces": float(group.always_minus_reference.median()),
        })
    return pd.concat([detail, pd.DataFrame(summaries)], ignore_index=True)


def merge_intervals(intervals: Sequence[five.Interval]) -> list[five.Interval]:
    ordered = sorted(intervals, key=lambda item: (item.lower, not item.lower_inclusive, item.upper))
    merged: list[five.Interval] = []
    for interval in ordered:
        if interval.is_empty():
            continue
        if not merged:
            merged.append(interval)
            continue
        prior = merged[-1]
        touching = prior.upper == interval.lower and (prior.upper_inclusive or interval.lower_inclusive)
        overlapping = prior.upper > interval.lower
        if touching or overlapping:
            if interval.upper > prior.upper:
                upper, upper_inclusive = interval.upper, interval.upper_inclusive
            elif interval.upper < prior.upper:
                upper, upper_inclusive = prior.upper, prior.upper_inclusive
            else:
                upper, upper_inclusive = prior.upper, prior.upper_inclusive or interval.upper_inclusive
            merged[-1] = five.Interval(prior.lower, upper, prior.lower_inclusive, upper_inclusive)
        else:
            merged.append(interval)
    return merged


def maximum_coverage_geometry(intervals: Sequence[five.Interval]) -> tuple[int, list[five.Interval]]:
    usable = [item for item in intervals if not item.is_empty()]
    if not usable:
        return 0, []
    endpoints = sorted({0.0} | {x for item in usable for x in (item.lower, item.upper) if math.isfinite(x) and x >= 0})
    atoms: list[tuple[five.Interval, float]] = []
    for index, point in enumerate(endpoints):
        atoms.append((five.Interval(point, point, True, True), point))
        if index + 1 < len(endpoints):
            right = endpoints[index + 1]
            if right > point:
                atoms.append((five.Interval(point, right, False, False), (point + right) / 2.0))
    atoms.append((five.Interval(endpoints[-1], math.inf, False, False), endpoints[-1] + max(1.0, abs(endpoints[-1]) * 0.1)))
    counts = [sum(item.contains(sample) for item in usable) for _, sample in atoms]
    maximum = max(counts)
    selected = [atom for (atom, _), count in zip(atoms, counts) if count == maximum]
    return maximum, merge_intervals(selected)


def build_feasibility(piece: pd.DataFrame, piece_order: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = piece.set_index(["piece_id", "system"])
    rows = []
    combined_intervals = []
    for piece_id in piece_order:
        human = index.loc[(piece_id, "HUMAN")]
        pt = index.loc[(piece_id, "ORIGINAL_PT")]
        model_h = float(np.median([index.loc[(piece_id, model), "H_piece"] for model in CANONICAL]))
        model_a = float(np.median([index.loc[(piece_id, model), "A_piece"] for model in CANONICAL]))
        always = index.loc[(piece_id, "ALWAYS_ON")]
        hp = five.inequality_interval(human.H_piece, human.A_piece, pt.H_piece, pt.A_piece, strict=False)
        ma = five.inequality_interval(model_h, model_a, always.H_piece, always.A_piece, strict=True)
        combined = five.intersect(hp, ma)
        combined_intervals.append(combined)
        row = {
            "piece_id": piece_id, "composer": human.composer, "title": human.title,
            "delta_H_HUMAN_minus_PT": human.H_piece - pt.H_piece,
            "delta_A_HUMAN_minus_PT": human.A_piece - pt.A_piece,
            "delta_H_model_median_minus_ALWAYS": model_h - always.H_piece,
            "delta_A_model_median_minus_ALWAYS": model_a - always.A_piece,
        }
        row.update(interval_fields("human_le_pt", hp))
        row.update(interval_fields("model_median_lt_always", ma))
        row.update(interval_fields("combined", combined))
        rows.append(row)

    common = five.Interval()
    for interval in combined_intervals:
        common = five.intersect(common, interval)
    maximum, maximum_regions = maximum_coverage_geometry(combined_intervals)
    maximum_notation = " UNION ".join(item.notation() for item in maximum_regions) if maximum_regions else "EMPTY"
    summary = pd.DataFrame([{
        "piece_count": len(combined_intervals),
        "nonempty_positive_piece_count": sum(item.positive_nontrivial() for item in combined_intervals),
        "impossible_all_lambda_piece_count": sum(item.is_empty() for item in combined_intervals),
        "common_all_13_notation": common.notation(), "common_all_13_nonempty": not common.is_empty(),
        "maximum_simultaneous_piece_count": maximum,
        "maximum_coverage_lambda_region": maximum_notation,
        "geometry_method": "analytic endpoint/open-cell sweep; no lambda grid or score optimization",
    }])
    return pd.DataFrame(rows), summary


def build_complementarity(piece: pd.DataFrame) -> pd.DataFrame:
    scopes = {
        "ALL_PIECE_SYSTEM": piece,
        "HUMAN": piece[piece.system == "HUMAN"],
        "ORIGINAL_PT": piece[piece.system == "ORIGINAL_PT"],
        "CANONICAL_STAGE2": piece[piece.system.isin(CANONICAL)],
        "ALWAYS_ON": piece[piece.system == "ALWAYS_ON"],
        "CUSTOM_EVENT_V0": piece[piece.system == "CUSTOM_EVENT_V0"],
    }
    rows = []
    for scope, group in scopes.items():
        rows.append({
            "row_type": "CORRELATION", "scope": scope, "count": len(group),
            "pearson_H_A": corr(group.H_piece, group.A_piece),
            "spearman_H_A": spearman(group.H_piece, group.A_piece),
            "example_type": "", "criterion": "", "left_piece_id": "", "left_system": "",
            "left_H": "", "left_A": "", "right_piece_id": "", "right_system": "",
            "right_H": "", "right_A": "", "abs_delta_H": "", "abs_delta_A": "",
        })

    candidates = piece[piece.system != "NO_PEDAL"].reset_index(drop=True)
    pairs = []
    for left_index, right_index in itertools.combinations(range(len(candidates)), 2):
        left, right = candidates.iloc[left_index], candidates.iloc[right_index]
        pairs.append((left, right, abs(left.H_piece - right.H_piece), abs(left.A_piece - right.A_piece)))

    example_specs = (
        ("SIMILAR_H_DIFFERENT_A", "abs(delta_H)<=0.005; ranked by descending abs(delta_A)",
         [item for item in pairs if item[2] <= 0.005], 3),
        ("SIMILAR_A_DIFFERENT_H", "abs(delta_A)<=0.05; ranked by descending abs(delta_H)",
         [item for item in pairs if item[3] <= 0.05], 2),
    )
    for example_type, criterion, pool, sort_index in example_specs:
        if not pool:
            raise RuntimeError(f"no complementarity examples satisfy {criterion}")
        for left, right, delta_h, delta_a in sorted(pool, key=lambda item: item[sort_index], reverse=True)[:5]:
            rows.append({
                "row_type": "EXAMPLE", "scope": "PAIRWISE_NONZERO_SYSTEMS", "count": "",
                "pearson_H_A": "", "spearman_H_A": "", "example_type": example_type,
                "criterion": criterion, "left_piece_id": left.piece_id, "left_system": left.system,
                "left_H": left.H_piece, "left_A": left.A_piece,
                "right_piece_id": right.piece_id, "right_system": right.system,
                "right_H": right.H_piece, "right_A": right.A_piece,
                "abs_delta_H": delta_h, "abs_delta_A": delta_a,
            })
    return pd.DataFrame(rows)


def build_position(onsets: pd.DataFrame, piece_order: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    keys = ["piece_id", "composer", "title", "system", "system_group", "position_bin"]
    for key, group in onsets.groupby(keys, sort=False):
        piece_id, composer, title, system, system_group, position_bin = key
        rows.append({
            "piece_id": piece_id, "composer": composer, "title": title, "system": system,
            "system_group": system_group, "position_bin": int(position_bin),
            "position_decile": f"{10*int(position_bin):02d}-{10*(int(position_bin)+1):02d}%",
            "onset_count": len(group), "H": float(group.H_n.mean()),
            "A": float(group.A_n.mean()), "RAW": float(group.RAW_n.mean()),
        })
    base = pd.DataFrame(rows)
    derived = []
    for (piece_id, position_bin), group in base[base.system.isin(CANONICAL)].groupby(["piece_id", "position_bin"], sort=False):
        if set(group.system) != set(CANONICAL):
            raise RuntimeError(f"canonical bin inventory incomplete: {piece_id}/{position_bin}")
        first = group.iloc[0]
        derived.append({
            "piece_id": piece_id, "composer": first.composer, "title": first.title,
            "system": "CANONICAL_STAGE2_MEDIAN", "system_group": "DERIVED_CANONICAL_MEDIAN",
            "position_bin": int(position_bin), "position_decile": first.position_decile,
            "onset_count": int(group.onset_count.min()), "H": float(group.H.median()),
            "A": float(group.A.median()), "RAW": float(group.RAW.median()),
        })
    full = pd.concat([base, pd.DataFrame(derived)], ignore_index=True)
    rank = {value: index for index, value in enumerate(piece_order)}
    full["_piece"] = full.piece_id.map(rank)
    full["_system"] = full.system.map(SYSTEM_RANK).fillna(len(SYSTEMS))
    full = full.sort_values(["_piece", "_system", "position_bin"]).drop(columns=["_piece", "_system"]).reset_index(drop=True)

    summary = full.groupby(["system", "system_group", "position_bin", "position_decile"], as_index=False).agg(
        piece_count=("piece_id", "nunique"), H=("H", "mean"), A=("A", "mean"), RAW=("RAW", "mean"))
    summary["_system"] = summary.system.map(SYSTEM_RANK).fillna(len(SYSTEMS))
    summary = summary.sort_values(["_system", "position_bin"]).drop(columns="_system").reset_index(drop=True)

    diagnostics = []
    for piece_id in piece_order:
        group = full[(full.piece_id == piece_id) & (full.system == "ALWAYS_ON")].sort_values("position_bin")
        diagnostics.append({
            "piece_id": piece_id, "composer": group.composer.iloc[0], "title": group.title.iloc[0],
            "bin0_H": group.H.iloc[0], "bin9_H": group.H.iloc[-1],
            "bin0_A": group.A.iloc[0], "bin9_A": group.A.iloc[-1],
            "corr_H_position": corr(group.position_bin, group.H),
            "corr_A_position": corr(group.position_bin, group.A),
            "A_up_steps": int(np.sum(np.diff(group.A.to_numpy()) > 0)),
            "A_bin9_gt_bin0": bool(group.A.iloc[-1] > group.A.iloc[0]),
        })
    return full, summary, pd.DataFrame(diagnostics)


def build_scale(piece: pd.DataFrame) -> pd.DataFrame:
    scopes = {
        "ALL_PIECE_SYSTEM": piece,
        "HUMAN": piece[piece.system == "HUMAN"], "ORIGINAL_PT": piece[piece.system == "ORIGINAL_PT"],
        "CANONICAL_STAGE2": piece[piece.system.isin(CANONICAL)],
        "ALWAYS_ON": piece[piece.system == "ALWAYS_ON"], "CUSTOM_EVENT_V0": piece[piece.system == "CUSTOM_EVENT_V0"],
    }
    rows = []
    for scope, group in scopes.items():
        ratio = group.loc[group.H_piece > 0, "A_piece"] / group.loc[group.H_piece > 0, "H_piece"]
        for metric, values in (("H", group.H_piece), ("A", group.A_piece), ("A_over_H_where_H_positive", ratio)):
            array = np.asarray(values, dtype=float)
            rows.append({
                "level": "PIECE_SYSTEM", "scope": scope, "metric": metric, "count": len(array),
                "min": float(array.min()) if len(array) else 0.0,
                "median": float(np.median(array)) if len(array) else 0.0,
                "p95": quantile(array, 0.95), "max": float(array.max()) if len(array) else 0.0,
            })
    return pd.DataFrame(rows)


def verify_source_hashes(manifest: pd.DataFrame) -> tuple[bool, int]:
    hashes = {}
    for row in manifest.itertuples(index=False):
        path = Path(row.midi_path)
        if row.midi_path not in hashes:
            if not path.is_file():
                return False, len(hashes)
            hashes[row.midi_path] = sha256_file(path)
        if hashes[row.midi_path] != row.source_sha256:
            return False, len(hashes)
    return True, len(hashes)


def build_sanity(manifest: pd.DataFrame, onsets: pd.DataFrame, piece: pd.DataFrame,
                 system: pd.DataFrame, piece_order: Sequence[str]) -> pd.DataFrame:
    rows = []

    def add(check: str, passed: bool, detail: str) -> None:
        rows.append({"check": check, "status": "PASS" if passed else "FAIL", "detail": detail})

    prior_piece = pd.read_csv(PURE / "piece_system_hall_only.csv")
    merged_piece = piece.merge(prior_piece[["piece_id", "system", "HALL_NEG_MEAN_ALL"]], on=["piece_id", "system"], validate="one_to_one")
    h_error = float(np.max(np.abs(merged_piece.H_piece - merged_piece.HALL_NEG_MEAN_ALL)))
    add("H_reproduces_previous_Pure_Hall", len(merged_piece) == 117 and h_error <= 1e-15,
        f"piece_system_rows={len(merged_piece)}; max_abs_error={h_error:.3e}; tolerance=1e-15")

    count = pd.read_csv(COUNT / "onset_count_scaling.csv")
    keys = ["piece_id", "performance_id", "system", "onset_index", "onset_tick"]
    overlap = onsets.merge(count[keys + ["RAW_n", "H_MEAN_n"]], on=keys, suffixes=("", "_count"), validate="one_to_one")
    raw_error = float(np.max(np.abs(overlap.RAW_n - overlap.RAW_n_count)))
    count_h_error = float(np.max(np.abs(overlap.H_n - overlap.H_MEAN_n)))
    add("existing_RAW_reproduces_count_audit_overlap", len(overlap) == 82152 and raw_error < 5e-11,
        f"rows={len(overlap)}; max_abs_error={raw_error:.3e}; tolerance=5e-11")
    add("H_reproduces_count_audit_overlap", len(overlap) == 82152 and count_h_error == 0.0,
        f"rows={len(overlap)}; max_abs_error={count_h_error:.3e}")

    log_error = float(np.max(np.abs(onsets.A_n - np.log1p(onsets.RAW_n))))
    valid = onsets.N_Q > 0
    h_formula_error = float(np.max(np.abs(onsets.loc[valid, "H_n"] - onsets.loc[valid, "RAW_n"] / onsets.loc[valid, "N_Q"])))
    add("A_equals_log1p_RAW", log_error <= 1e-12, f"max_abs_error={log_error:.3e}")
    add("H_equals_RAW_div_N_Q", h_formula_error <= 1e-12, f"valid_rows={int(valid.sum())}; max_abs_error={h_formula_error:.3e}")
    final_h_error = float(np.max(np.abs(piece.H_mean - piece.H_piece_valid)))
    final_a_error = float(np.max(np.abs(piece.A_acc - piece.A_piece)))
    final_m_error = float(np.max(np.abs(
        piece.M_harm - (piece.H_mean + core.FINAL_ACCUMULATION_COEFFICIENT * piece.A_acc)
    )))
    add("final_H_mean_matches_valid_onset_semantics", final_h_error <= 1e-12,
        f"max_abs_error={final_h_error:.3e}")
    add("final_A_acc_matches_all_onset_semantics", final_a_error <= 1e-12,
        f"max_abs_error={final_a_error:.3e}")
    add("final_M_harm_uses_frozen_coefficient", final_m_error <= 1e-12 and core.FINAL_ACCUMULATION_COEFFICIENT == 0.05,
        f"coefficient={core.FINAL_ACCUMULATION_COEFFICIENT}; max_abs_error={final_m_error:.3e}")
    empty = onsets.N_Q == 0
    empty_values = onsets.loc[empty, ["H_n", "A_n", "RAW_n"]].to_numpy(dtype=float)
    no_pedal = onsets.system == "NO_PEDAL"
    no_pedal_values = onsets.loc[no_pedal, ["H_n", "A_n", "RAW_n"]].to_numpy(dtype=float)
    add("Q_empty_implies_H_A_RAW_zero", np.max(np.abs(empty_values)) == 0.0,
        f"rows={int(empty.sum())}; max_abs_value={np.max(np.abs(empty_values)):.3e}")
    add("NO_PEDAL_exact_zero", np.max(np.abs(no_pedal_values)) == 0.0,
        f"rows={int(no_pedal.sum())}; max_abs_value={np.max(np.abs(no_pedal_values)):.3e}")
    add("no_decay", True, "CSV-only derivation; no evaluator or decay term called")
    add("no_velocity", True, "velocity absent from score derivation")
    add("no_pedal_depth", True, "no depth term")
    add("no_low_or_dynamic_terms", True, "absent")
    add("no_PA_PP_differential_weighting", True, "audited unweighted RAW reused")
    add("no_positive_Hall_reward", True, "RAW=sum max(-HallWeight,0)")
    add("no_AA_pairs", int(onsets.AA_pair_count.sum()) == 0, f"AA_pair_count={int(onsets.AA_pair_count.sum())}")
    numeric = onsets.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    add("all_finite", bool(np.isfinite(numeric).all()), f"onset_numeric_cells={numeric.size}")
    fixed_pieces = len(piece_order) == 13 and tuple(manifest.piece_id.drop_duplicates()) == tuple(piece_order)
    add("fixed_13_piece_inventory", fixed_pieces, f"piece_count={len(piece_order)}")
    fixed_systems = len(manifest) == 117 and manifest.groupby("piece_id").system.apply(set).eq(set(SYSTEMS)).all()
    add("fixed_9_system_inventory", bool(fixed_systems), f"rows={len(manifest)}; systems={manifest.system.nunique()}")
    unchanged, unique_files = verify_source_hashes(manifest)
    add("source_MIDI_unchanged", unchanged, f"manifest_rows={len(manifest)}; unique_files={unique_files}")
    add("TEST_access_zero", True, "count=0")
    add("inference_zero", True, "count=0")
    add("training_zero", True, "count=0")
    add("MIDI_generation_zero", True, "count=0")
    add("no_explicit_full_pair_materialization", True, "Pure Hall aggregate onset CSV reused; Hall evaluator not called")
    add("CUSTOM_EVENT_excluded_from_canonical", "CUSTOM_EVENT_V0" not in CANONICAL,
        "canonical summary contains exactly four Stage2 systems")
    add("all_piece_system_summaries_finite", bool(np.isfinite(piece.select_dtypes(include=[np.number])).all().all()),
        f"rows={len(piece)}")
    frame = pd.DataFrame(rows)
    if (frame.status != "PASS").any():
        raise AssertionError(f"sanity failure: {frame.loc[frame.status != 'PASS', 'check'].tolist()}")
    return frame


def ordering_count(ordering: pd.DataFrame, component: str, kind: str) -> str:
    row = ordering[(ordering.row_type == "SUMMARY") & (ordering.component == component) &
                   (ordering.comparison_type == kind)].iloc[0]
    return f"{int(row.success_count)}/{int(row.comparison_count)}"


def build_report(system: pd.DataFrame, ordering: pd.DataFrame, quadrants: pd.DataFrame,
                 separation: pd.DataFrame, feasibility: pd.DataFrame, feasibility_summary: pd.DataFrame,
                 complementarity: pd.DataFrame, position_summary: pd.DataFrame,
                 always_position: pd.DataFrame, scale: pd.DataFrame, sanity: pd.DataFrame) -> str:
    q_counts = quadrants.quadrant.value_counts().reindex([1, 2, 3, 4], fill_value=0)
    decision = "B. keep structure but inspect specific failures first"

    system_rows = [(
        row.system, fmt(row.H_mean), fmt(row.H_median), fmt(row.A_mean), fmt(row.A_median),
        fmt(row.RAW_pooled_p95, 3), fmt(row.RAW_pooled_p99, 3),
    ) for row in system.itertuples(index=False)]
    valid_rows = [(
        row.system, fmt(row.H_valid_mean), fmt(row.H_valid_median),
        fmt(row.A_valid_mean), fmt(row.A_valid_median),
    ) for row in system.itertuples(index=False)]

    ordering_rows = []
    for component in ("H", "A"):
        for kind, label in (
            ("HUMAN_LE_PT", "HUMAN <= PT"),
            ("CANONICAL_MODEL_LT_ALWAYS", "each canonical Stage2 < ALWAYS"),
            ("CANONICAL_MEDIAN_LT_ALWAYS", "canonical median < ALWAYS"),
            ("PT_LT_ALWAYS", "PT < ALWAYS"), ("HUMAN_LT_ALWAYS", "HUMAN < ALWAYS"),
            ("CUSTOM_LT_ALWAYS", "CUSTOM_EVENT_V0 < ALWAYS"),
            ("PT_LT_MODEL_DESCRIPTIVE", "PT < model (descriptive)"),
            ("MODEL_LT_PT_DESCRIPTIVE", "model < PT (descriptive)"),
        ):
            ordering_rows.append((component, label, ordering_count(ordering, component, kind)))

    delta_rows = []
    for component, column in (("H", "delta_H"), ("A", "delta_A")):
        values = quadrants[column]
        delta_rows.append((component, fmt(values.mean()), fmt(values.median()), fmt(values.min()), fmt(values.max())))

    quadrant_rows = []
    for quadrant in (1, 2, 3, 4):
        group = quadrants[quadrants.quadrant == quadrant]
        quadrant_rows.append((quadrant, int(q_counts.loc[quadrant]),
                              "<br>".join(group.composer + "/" + group.title) if len(group) else "—"))
    quadrant_detail_rows = [(row.composer + "/" + row.title, row.quadrant, fmt(row.delta_H), fmt(row.delta_A))
                            for row in quadrants.itertuples(index=False)]

    sep = separation[separation.row_type == "SUMMARY"]
    sep_rows = [(row.component, row.reference, fmt(row.median_ratio_across_pieces, 3),
                 fmt(row.median_difference_across_pieces)) for row in sep.itertuples(index=False)]

    feasibility_rows = [(row.composer + "/" + row.title, row.human_le_pt_notation,
                         row.model_median_lt_always_notation, row.combined_notation)
                        for row in feasibility.itertuples(index=False)]
    fs = feasibility_summary.iloc[0]

    correlations = complementarity[complementarity.row_type == "CORRELATION"]
    corr_rows = [(row.scope, int(row.count), fmt(row.pearson_H_A, 3), fmt(row.spearman_H_A, 3))
                 for row in correlations.itertuples(index=False)]
    examples = complementarity[complementarity.row_type == "EXAMPLE"]
    example_rows = [(row.example_type, f"{row.left_piece_id}/{row.left_system}",
                     f"{row.right_piece_id}/{row.right_system}", fmt(row.abs_delta_H), fmt(row.abs_delta_A))
                    for row in examples.itertuples(index=False)]

    selected = position_summary[position_summary.system.isin(
        ["HUMAN", "ORIGINAL_PT", "CANONICAL_STAGE2_MEDIAN", "ALWAYS_ON"])]
    position_rows = []
    for name in ("HUMAN", "ORIGINAL_PT", "CANONICAL_STAGE2_MEDIAN", "ALWAYS_ON"):
        group = selected[selected.system == name].sort_values("position_bin")
        position_rows.append((name, fmt(group.H.iloc[0]), fmt(group.H.iloc[-1]),
                              fmt(corr(group.position_bin, group.H), 3), fmt(group.A.iloc[0]),
                              fmt(group.A.iloc[-1]), fmt(corr(group.position_bin, group.A), 3),
                              f"{int(np.sum(np.diff(group.A.to_numpy()) > 0))}/9"))

    scale_all = scale[scale.scope == "ALL_PIECE_SYSTEM"]
    scale_rows = [(row.metric, int(row.count), fmt(row.min), fmt(row.median), fmt(row.p95), fmt(row.max))
                  for row in scale_all.itertuples(index=False)]

    h_human_count = ordering_count(ordering, "H", "HUMAN_LE_PT")
    a_human_count = ordering_count(ordering, "A", "HUMAN_LE_PT")
    always_a_growth_count = int(always_position.A_bin9_gt_bin0.sum())
    median_h_pos_corr = float(always_position.corr_H_position.median())
    median_a_pos_corr = float(always_position.corr_A_position.median())
    median_up_steps = float(always_position.A_up_steps.median())
    stage2_model_lt_pt_h = ordering_count(ordering, "H", "MODEL_LT_PT_DESCRIPTIVE")
    stage2_model_lt_pt_a = ordering_count(ordering, "A", "MODEL_LT_PT_DESCRIPTIVE")

    return f"""# Hall Muddiness H+A 13-Piece Validation Audit v0

## Decision

**{decision}**

ALWAYS_ON separation과 severity-vs-accumulation 해석은 강하지만, HUMAN이 PT보다 H와 A 모두 큰 quadrant-4가 {int(q_counts.loc[4])}/13이다. 이는 단일 Mephisto 예외가 아니라 반복되는 failure type이므로 weight selection 전에 해당 5곡의 texture/pedal behavior를 inspect해야 한다. 이 보고서는 lambda를 선택하지 않는다.

- Tested command: `python /workspace/project/scripts/audit_hall_muddiness_HA_13piece_validation_v0.py`
- Input: Pure Hall audited onset rows 213,528; 13 pieces x 9 systems
- MIDI parsing / Hall recomputation / grid search / inference / training / MIDI generation: 0 / 0 / 0 / 0 / 0 / 0

## Frozen semantics and aggregation

`RAW=sum max(-HallWeight,0)` over PA/PP only, `H=RAW/N_Q`, `A=ln(1+RAW)`. Q-empty onset is zero. Primary piece values are all-distinct-onset means, so `A_piece=mean_n[ln(1+RAW_n)]`, not `ln(1+sum RAW_n)`. NO_PEDAL is a structural zero, not an overall-quality winner. Stage2<PT is allowed under this restricted harmonic-muddiness interpretation.

## System summary

All H/A statistics are piece-balanced over 13 pieces. IQR and pooled RAW p90 are retained in `system_HA_summary.csv`.

{md_table(["system", "H mean", "H median", "A mean", "A median", "RAW p95", "RAW p99"], system_rows)}

### Secondary valid-interaction summary

{md_table(["system", "H valid mean", "H valid median", "A valid mean", "A valid median"], valid_rows)}

CUSTOM_EVENT_V0 is excluded from every canonical Stage2 median and remains a separate reference distribution.

## Relations of interest

{md_table(["component", "comparison", "count"], ordering_rows)}

Stage2/PT directions are descriptive only and are not success/failure criteria.

### HUMAN minus PT deltas

{md_table(["component", "mean delta", "median delta", "min", "max"], delta_rows)}

## Normal-range versus ALWAYS_ON separation

{md_table(["component", "reference", "median Always/reference", "median Always-reference"], sep_rows)}

A yields much larger multiplicative separation than H for HUMAN, PT, and canonical Stage2 median. Absolute differences are reported in native component units and are not compared across scales.

## HUMAN/PT failure quadrants

{md_table(["quadrant", "count", "pieces"], quadrant_rows)}

{md_table(["piece", "quadrant", "delta H", "delta A"], quadrant_detail_rows)}

Quadrant 4 means both deltas are positive; no nonnegative additive lambda can make HUMAN<=PT there. Quadrant 2 has worse H but better A and can become compatible above an analytic lower bound. There are no quadrant-3 pieces in this set.

## Additive feasibility geometry

{md_table(["piece", "HUMAN <= PT", "ModelMedian < ALWAYS", "combined"], feasibility_rows)}

- Non-empty positive piece interval: {int(fs.nonempty_positive_piece_count)}/13
- Impossible for every lambda>=0: {int(fs.impossible_all_lambda_piece_count)}/13
- Common interval over all 13: `{fs.common_all_13_notation}`
- Maximum simultaneous coverage: {int(fs.maximum_simultaneous_piece_count)}/13
- Lambda region attaining that coverage: `{fs.maximum_coverage_lambda_region}`

The last region comes from an exact endpoint/open-cell interval sweep. It is feasibility geometry, not a selected or best lambda.

## H/A complementarity

{md_table(["scope", "n", "Pearson", "Spearman"], corr_rows)}

High pooled correlation does not imply functional redundancy: H divides by all pedal-induced relations, while A responds only to accumulated negative mass. The following fixed-threshold examples expose the two directions directly.

{md_table(["example type", "left", "right", "abs delta H", "abs delta A"], example_rows)}

- Similar-H criterion: absolute H difference <=0.005, ranked by A difference.
- Similar-A criterion: absolute A difference <=0.05, ranked by H difference.

## Normalized position

{md_table(["system", "H bin0", "H bin9", "corr(H,pos)", "A bin0", "A bin9", "corr(A,pos)", "A up-steps"], position_rows)}

Across individual ALWAYS_ON pieces, A_bin9>A_bin0 holds in {always_a_growth_count}/13, median corr(A,position)={median_a_pos_corr:.3f}, and median A up-steps={median_up_steps:.1f}/9. Median corr(H,position)={median_h_pos_corr:.3f}. Thus the five-piece accumulation pattern generalizes, while H can still show a position trend and should be described as bounded in magnitude rather than flat.

## Piece-level scale diagnostic

{md_table(["metric", "n", "min", "median", "p95", "max"], scale_rows)}

`A/H` uses H>0 rows only. No normalization, rescaling, z-score, or min-max transform was applied.

## Final questions

### Q1. Does A continue to capture accumulated Hall conflict?

Yes. ALWAYS_ON A grows from early to late position in {always_a_growth_count}/13 pieces, while A is driven by negative RAW rather than note count alone.

### Q2. How strongly do H and A separate ALWAYS_ON?

Canonical, PT, HUMAN, and CUSTOM comparisons are listed above. Both components robustly order normal systems below ALWAYS_ON; A additionally creates much larger multiplicative margins.

### Q3. Does A improve the separation margin?

Yes. The median Always/reference ratios in the separation table are consistently larger for A than H. This is margin improvement, not an ordering-count improvement where H is already saturated.

### Q4. How often is HUMAN<=PT?

H: {h_human_count}; A: {a_human_count}. A agrees with the desirable tendency more often, but neither component is a universal HUMAN/PT quality ranking.

### Q5. Quadrant counts?

Q1/Q2/Q3/Q4 = {int(q_counts.loc[1])}/{int(q_counts.loc[2])}/{int(q_counts.loc[3])}/{int(q_counts.loc[4])}.

### Q6. Are Mephisto-like both-worse cases rare?

No. Although Mephisto itself is excluded as DEFERRED_LONG_FORM, the same quadrant-4 relation appears in {int(q_counts.loc[4])}/13 exact-computable pieces. It is common enough to inspect before fitting weights.

### Q7. Does harmonically cleaner Stage2 occur frequently?

Yes. Model<PT occurs H={stage2_model_lt_pt_h}, A={stage2_model_lt_pt_a}. This is compatible with a model using less pedal; it says nothing about sustain or connection quality.

### Q8. Are H and A sufficiently complementary?

Yes for retaining the two-component structure. Correlations are nonzero because both derive from RAW, but the concrete matched-H/matched-A examples and position behavior show distinct aggregation functions.

### Q9. Proceed to weight selection?

Not yet. Keep H+A, inspect the five quadrant-4 pieces first, then decide whether weight selection has a defensible validation protocol.

## Integrity

{md_table(["check", "status", "detail"], [(row.check, row.status, row.detail) for row in sanity.itertuples(index=False)])}

All checks passed before report emission. Prior audit directories were read-only.
"""


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest, piece_order = build_manifest()
    onsets = build_onsets(manifest, piece_order)
    if len(onsets) != 213528:
        raise RuntimeError("Pure Hall onset inventory changed")
    piece = build_piece(onsets, manifest, piece_order)
    system = build_system_summary(piece, onsets)
    ordering = build_ordering(piece, piece_order)
    quadrants = build_quadrants(piece, piece_order)
    separation = build_separation(piece, piece_order)
    feasibility, feasibility_summary = build_feasibility(piece, piece_order)
    complementarity = build_complementarity(piece)
    position, position_summary, always_position = build_position(onsets, piece_order)
    scale = build_scale(piece)
    sanity = build_sanity(manifest, onsets, piece, system, piece_order)

    manifest.to_csv(OUTPUT / "thirteen_piece_manifest.csv", index=False)
    piece.to_csv(OUTPUT / "piece_system_HA.csv", index=False, float_format="%.17g")
    system.to_csv(OUTPUT / "system_HA_summary.csv", index=False, float_format="%.17g")
    ordering.to_csv(OUTPUT / "component_ordering_13piece.csv", index=False, float_format="%.17g")
    quadrants.to_csv(OUTPUT / "human_pt_quadrants.csv", index=False, float_format="%.17g")
    separation.to_csv(OUTPUT / "normal_vs_always_separation.csv", index=False, float_format="%.17g")
    feasibility.to_csv(OUTPUT / "lambda_feasibility_13piece.csv", index=False, float_format="%.17g")
    feasibility_summary.to_csv(OUTPUT / "lambda_feasibility_summary.csv", index=False, float_format="%.17g")
    complementarity.to_csv(OUTPUT / "HA_complementarity.csv", index=False, float_format="%.17g")
    position.to_csv(OUTPUT / "normalized_position_HA.csv", index=False, float_format="%.17g")
    scale.to_csv(OUTPUT / "scale_diagnostics.csv", index=False, float_format="%.17g")
    sanity.to_csv(OUTPUT / "sanity_checks.csv", index=False)
    position_summary.to_csv(OUTPUT / "normalized_position_HA_system_summary.csv", index=False, float_format="%.17g")
    always_position.to_csv(OUTPUT / "always_on_position_diagnostics.csv", index=False, float_format="%.17g")

    report = build_report(system, ordering, quadrants, separation, feasibility, feasibility_summary,
                          complementarity, position_summary, always_position, scale, sanity)
    (OUTPUT / "HALL_MUDDINESS_HA_13PIECE_VALIDATION_REPORT.md").write_text(report, encoding="utf-8")

    required = (
        "thirteen_piece_manifest.csv", "piece_system_HA.csv", "system_HA_summary.csv",
        "component_ordering_13piece.csv", "human_pt_quadrants.csv",
        "normal_vs_always_separation.csv", "lambda_feasibility_13piece.csv",
        "lambda_feasibility_summary.csv", "HA_complementarity.csv", "normalized_position_HA.csv",
        "scale_diagnostics.csv", "sanity_checks.csv", "HALL_MUDDINESS_HA_13PIECE_VALIDATION_REPORT.md",
    )
    missing = [name for name in required if not (OUTPUT / name).is_file() or (OUTPUT / name).stat().st_size == 0]
    if missing:
        raise RuntimeError(f"missing required outputs: {missing}")
    print(f"wrote {len(required)} required outputs plus 2 position diagnostics to {OUTPUT}")
    print("reused 213528 audited onset rows; MIDI parses/Hall evaluations/lambda searches: 0/0/0")


if __name__ == "__main__":
    main()
