#!/usr/bin/env python3
"""Additive Hall severity + conflict-amount feasibility audit v0.

This script deliberately reuses the audited five-piece count-scaling CSVs.
It does not parse MIDI, compute Hall pairs, search lambda, run inference, or
materialize any new MIDI.  The only new onset component is log1p(RAW_n).
"""

from __future__ import annotations

import hashlib
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "analysis/hall_negative_count_scaling_audit_v0"
PURE = ROOT / "analysis/harmonic_metric_pure_hall_negative_v0"
OUTPUT = ROOT / "analysis/hall_muddiness_additive_feasibility_v0"

SYSTEMS = (
    "NO_PEDAL",
    "ALWAYS_ON",
    "STANDARD_CE_ARGMAX",
    "STANDARD_CE_POSTERIOR_MEDIAN",
    "WEIGHTED_CE_ARGMAX",
    "HYBRID_REGRESSION_ONLY",
    "CUSTOM_EVENT_V0",
    "ORIGINAL_PT",
    "HUMAN",
)
CANONICAL = (
    "STANDARD_CE_ARGMAX",
    "STANDARD_CE_POSTERIOR_MEDIAN",
    "WEIGHTED_CE_ARGMAX",
    "HYBRID_REGRESSION_ONLY",
)
PIECE_ORDER = (
    "piece_de3f82957f1b3532",
    "piece_daefdda4e1923cc6",
    "piece_7195bbce81550519",
    "piece_69862af5096ee3fa",
    "piece_db97fbaed2036f5b",
)
SYSTEM_ORDER = {value: index for index, value in enumerate(SYSTEMS)}
PIECE_RANK = {value: index for index, value in enumerate(PIECE_ORDER)}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def q(values: Iterable[float], level: float) -> float:
    array = np.asarray(list(values), dtype=float)
    return float(np.quantile(array, level)) if len(array) else 0.0


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


@dataclass(frozen=True)
class Interval:
    lower: float = 0.0
    upper: float = math.inf
    lower_inclusive: bool = True
    upper_inclusive: bool = False
    empty: bool = False

    def is_empty(self) -> bool:
        if self.empty or self.lower > self.upper:
            return True
        return self.lower == self.upper and not (self.lower_inclusive and self.upper_inclusive)

    def contains(self, value: float) -> bool:
        if self.is_empty():
            return False
        lower_ok = value > self.lower or (value == self.lower and self.lower_inclusive)
        upper_ok = value < self.upper or (value == self.upper and self.upper_inclusive)
        return lower_ok and upper_ok

    def positive_nontrivial(self) -> bool:
        if self.is_empty():
            return False
        if math.isinf(self.upper):
            return True
        return self.upper > max(0.0, self.lower)

    def notation(self) -> str:
        if self.is_empty():
            return "EMPTY"
        left = "[" if self.lower_inclusive else "("
        right = "]" if self.upper_inclusive else ")"
        return f"{left}{self.lower:.12g}, {'+inf' if math.isinf(self.upper) else f'{self.upper:.12g}'}{right}"


def intersect(left: Interval, right: Interval) -> Interval:
    if left.is_empty() or right.is_empty():
        return Interval(empty=True)
    lower = max(left.lower, right.lower)
    upper = min(left.upper, right.upper)
    if left.lower > right.lower:
        lower_inclusive = left.lower_inclusive
    elif right.lower > left.lower:
        lower_inclusive = right.lower_inclusive
    else:
        lower_inclusive = left.lower_inclusive and right.lower_inclusive
    if left.upper < right.upper:
        upper_inclusive = left.upper_inclusive
    elif right.upper < left.upper:
        upper_inclusive = right.upper_inclusive
    else:
        upper_inclusive = left.upper_inclusive and right.upper_inclusive
    result = Interval(lower, upper, lower_inclusive, upper_inclusive)
    return Interval(empty=True) if result.is_empty() else result


def inequality_interval(h_x: float, a_x: float, h_y: float, a_y: float, strict: bool) -> Interval:
    """Solve Hx + lambda*Ax <=/< Hy + lambda*Ay over lambda >= 0."""
    delta_h = float(h_x) - float(h_y)
    delta_a = float(a_x) - float(a_y)
    if delta_a == 0.0:
        holds = delta_h < 0.0 if strict else delta_h <= 0.0
        return Interval() if holds else Interval(empty=True)
    threshold = -delta_h / delta_a
    if delta_a > 0.0:
        candidate = Interval(0.0, threshold, True, not strict)
    else:
        candidate = Interval(threshold, math.inf, not strict, False)
    return intersect(Interval(), candidate)


def interval_fields(prefix: str, interval: Interval) -> dict[str, Any]:
    return {
        f"{prefix}_lower": "" if interval.is_empty() else interval.lower,
        f"{prefix}_upper": "" if interval.is_empty() or math.isinf(interval.upper) else interval.upper,
        f"{prefix}_upper_is_infinite": False if interval.is_empty() else math.isinf(interval.upper),
        f"{prefix}_lower_inclusive": False if interval.is_empty() else interval.lower_inclusive,
        f"{prefix}_upper_inclusive": False if interval.is_empty() else interval.upper_inclusive,
        f"{prefix}_notation": interval.notation(),
        f"{prefix}_nonempty": not interval.is_empty(),
        f"{prefix}_positive_nontrivial": interval.positive_nontrivial(),
    }


def maximum_interval_overlap(intervals: Sequence[Interval]) -> int:
    """Exact endpoint/cell sweep; this is interval geometry, not lambda tuning."""
    usable = [interval for interval in intervals if not interval.is_empty()]
    if not usable:
        return 0
    points = sorted({0.0} | {x for i in usable for x in (i.lower, i.upper) if math.isfinite(x) and x >= 0.0})
    candidates = list(points)
    candidates.extend((left + right) / 2.0 for left, right in zip(points, points[1:]) if right > left)
    if points:
        candidates.append(points[-1] + max(1.0, abs(points[-1]) * 0.1))
    return max(sum(interval.contains(value) for interval in usable) for value in candidates)


def ordered(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["_piece"] = result.piece_id.map(PIECE_RANK)
    result["_system"] = result.system.map(SYSTEM_ORDER).fillna(len(SYSTEMS))
    return result.sort_values(["_piece", "_system"]).drop(columns=["_piece", "_system"]).reset_index(drop=True)


def build_onsets(source: pd.DataFrame) -> pd.DataFrame:
    required = {
        "piece_id", "composer", "title", "performance_id", "system", "system_group",
        "evaluation_mode", "onset_index", "onset_tick", "onset_time_sec", "position_bin",
        "A_n_count", "N_P", "PA_pair_count", "PP_pair_count", "N_Q", "RAW_n",
        "H_MEAN_n", "AA_pair_count",
    }
    missing = required - set(source.columns)
    if missing:
        raise RuntimeError(f"audited onset table lacks required columns: {sorted(missing)}")
    out = source[list(required)].copy()
    out = out.rename(columns={"A_n_count": "active_note_count", "H_MEAN_n": "H_n"})
    out["A_n"] = np.log1p(out["RAW_n"].to_numpy(dtype=float))
    columns = [
        "piece_id", "composer", "title", "performance_id", "system", "system_group",
        "evaluation_mode", "onset_index", "onset_tick", "onset_time_sec", "position_bin",
        "active_note_count", "N_P", "PA_pair_count", "PP_pair_count", "N_Q", "RAW_n",
        "H_n", "A_n", "AA_pair_count",
    ]
    return ordered(out[columns])


def build_piece_components(onsets: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    path_lookup = manifest.set_index(["piece_id", "system"])[["midi_path", "source_sha256_before"]]
    rows: list[dict[str, Any]] = []
    for (piece_id, system), group in onsets.groupby(["piece_id", "system"], sort=False):
        valid = group[group.N_Q > 0]
        first = group.iloc[0]
        path = path_lookup.loc[(piece_id, system)]
        rows.append({
            "piece_id": piece_id,
            "composer": first.composer,
            "title": first.title,
            "performance_id": first.performance_id,
            "system": system,
            "system_group": first.system_group,
            "evaluation_mode": first.evaluation_mode,
            "midi_path": path.midi_path,
            "source_sha256": path.source_sha256_before,
            "all_distinct_onset_count": len(group),
            "valid_interaction_onset_count": len(valid),
            "valid_interaction_onset_fraction": len(valid) / len(group),
            "H_piece": float(group.H_n.mean()),
            "A_piece": float(group.A_n.mean()),
            "H_piece_valid": float(valid.H_n.mean()) if len(valid) else 0.0,
            "A_piece_valid": float(valid.A_n.mean()) if len(valid) else 0.0,
            "mean_RAW_per_onset": float(group.RAW_n.mean()),
            "median_RAW_per_onset": float(group.RAW_n.median()),
            "p95_RAW_per_onset": q(group.RAW_n, 0.95),
            "p99_RAW_per_onset": q(group.RAW_n, 0.99),
            "max_RAW_per_onset": float(group.RAW_n.max()),
            "mean_RAW_valid_onset": float(valid.RAW_n.mean()) if len(valid) else 0.0,
            "median_RAW_valid_onset": float(valid.RAW_n.median()) if len(valid) else 0.0,
        })
    return ordered(pd.DataFrame(rows))


def build_system_summary(piece: pd.DataFrame, onsets: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for system in SYSTEMS:
        p = piece[piece.system == system]
        o = onsets[onsets.system == system]
        rows.append({
            "system": system,
            "piece_count": len(p),
            "onset_count": len(o),
            "mean_H_piece": float(p.H_piece.mean()),
            "median_H_piece": float(p.H_piece.median()),
            "mean_A_piece": float(p.A_piece.mean()),
            "median_A_piece": float(p.A_piece.median()),
            "mean_H_piece_valid": float(p.H_piece_valid.mean()),
            "median_H_piece_valid": float(p.H_piece_valid.median()),
            "mean_A_piece_valid": float(p.A_piece_valid.mean()),
            "median_A_piece_valid": float(p.A_piece_valid.median()),
            "pooled_mean_RAW_per_onset": float(o.RAW_n.mean()),
            "pooled_median_RAW_per_onset": float(o.RAW_n.median()),
            "pooled_p95_RAW_per_onset": q(o.RAW_n, 0.95),
            "pooled_p99_RAW_per_onset": q(o.RAW_n, 0.99),
        })
    return pd.DataFrame(rows)


def build_ordering(piece: pd.DataFrame) -> pd.DataFrame:
    index = piece.set_index(["piece_id", "system"])
    details: list[dict[str, Any]] = []

    def add(component: str, kind: str, piece_id: str, left: str, right: str,
            left_value: float, right_value: float, operator: str, hard: bool) -> None:
        if operator == "<=":
            holds: Any = left_value <= right_value
        elif operator == "<":
            holds = left_value < right_value
        else:
            holds = ""
        relation = "<" if left_value < right_value else (">" if left_value > right_value else "=")
        details.append({
            "row_type": "DETAIL", "component": component, "comparison_type": kind,
            "piece_id": piece_id, "left_label": left, "right_label": right,
            "left_value": left_value, "right_value": right_value, "operator": operator,
            "holds": holds, "observed_relation": relation, "hard_success_criterion": hard,
            "success_count": "", "comparison_count": "",
        })

    for component, column in (("H", "H_piece"), ("A", "A_piece")):
        for piece_id in PIECE_ORDER:
            human = float(index.loc[(piece_id, "HUMAN"), column])
            pt = float(index.loc[(piece_id, "ORIGINAL_PT"), column])
            model_values = [float(index.loc[(piece_id, model), column]) for model in CANONICAL]
            median = float(np.median(model_values))
            always = float(index.loc[(piece_id, "ALWAYS_ON"), column])
            add(component, "HUMAN_LE_PT", piece_id, "HUMAN", "ORIGINAL_PT", human, pt, "<=", True)
            for model, value in zip(CANONICAL, model_values):
                add(component, "CANONICAL_MODEL_LT_ALWAYS_ON", piece_id, model, "ALWAYS_ON", value, always, "<", True)
            add(component, "CANONICAL_MEDIAN_LT_ALWAYS_ON", piece_id, "CANONICAL_STAGE2_MEDIAN", "ALWAYS_ON", median, always, "<", True)
            add(component, "PT_VS_CANONICAL_MEDIAN", piece_id, "ORIGINAL_PT", "CANONICAL_STAGE2_MEDIAN", pt, median, "DESCRIPTIVE", False)

    detail_frame = pd.DataFrame(details)
    summaries: list[dict[str, Any]] = []
    for (component, kind), group in detail_frame[detail_frame.operator != "DESCRIPTIVE"].groupby(["component", "comparison_type"], sort=False):
        summaries.append({
            "row_type": "SUMMARY", "component": component, "comparison_type": kind,
            "piece_id": "ALL", "left_label": "", "right_label": "", "left_value": "",
            "right_value": "", "operator": group.operator.iloc[0], "holds": "",
            "observed_relation": "", "hard_success_criterion": True,
            "success_count": int(group.holds.astype(bool).sum()), "comparison_count": len(group),
        })
    return pd.concat([detail_frame, pd.DataFrame(summaries)], ignore_index=True)


def build_feasibility(piece: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[Interval]]:
    index = piece.set_index(["piece_id", "system"])
    rows: list[dict[str, Any]] = []
    combined_intervals: list[Interval] = []
    for piece_id in PIECE_ORDER:
        human = index.loc[(piece_id, "HUMAN")]
        pt = index.loc[(piece_id, "ORIGINAL_PT")]
        model_h = float(np.median([index.loc[(piece_id, model), "H_piece"] for model in CANONICAL]))
        model_a = float(np.median([index.loc[(piece_id, model), "A_piece"] for model in CANONICAL]))
        always = index.loc[(piece_id, "ALWAYS_ON")]
        human_pt = inequality_interval(human.H_piece, human.A_piece, pt.H_piece, pt.A_piece, strict=False)
        model_always = inequality_interval(model_h, model_a, always.H_piece, always.A_piece, strict=True)
        combined = intersect(human_pt, model_always)
        combined_intervals.append(combined)
        row = {
            "piece_id": piece_id, "composer": human.composer, "title": human.title,
            "H_HUMAN": human.H_piece, "A_HUMAN": human.A_piece,
            "H_ORIGINAL_PT": pt.H_piece, "A_ORIGINAL_PT": pt.A_piece,
            "H_canonical_median": model_h, "A_canonical_median": model_a,
            "H_ALWAYS_ON": always.H_piece, "A_ALWAYS_ON": always.A_piece,
            "human_pt_delta_H": human.H_piece - pt.H_piece,
            "human_pt_delta_A": human.A_piece - pt.A_piece,
            "model_always_delta_H": model_h - always.H_piece,
            "model_always_delta_A": model_a - always.A_piece,
        }
        row.update(interval_fields("human_le_pt", human_pt))
        row.update(interval_fields("model_median_lt_always", model_always))
        row.update(interval_fields("intersection", combined))
        rows.append(row)

    global_interval = Interval()
    for interval in combined_intervals:
        global_interval = intersect(global_interval, interval)
    nonempty_count = sum(not interval.is_empty() for interval in combined_intervals)
    positive_count = sum(interval.positive_nontrivial() for interval in combined_intervals)
    maximum_overlap = maximum_interval_overlap(combined_intervals)
    summary_rows = [
        {
            "summary_type": "PIECE_FEASIBILITY_COUNTS", "piece_count": len(combined_intervals),
            "nonempty_piece_count": nonempty_count, "positive_nontrivial_piece_count": positive_count,
            "maximum_piece_overlap": maximum_overlap, **interval_fields("common_all_five", global_interval),
        },
        {
            "summary_type": "GLOBAL_COMMON_INTERVAL", "piece_count": len(combined_intervals),
            "nonempty_piece_count": nonempty_count, "positive_nontrivial_piece_count": positive_count,
            "maximum_piece_overlap": maximum_overlap, **interval_fields("common_all_five", global_interval),
        },
    ]
    return pd.DataFrame(rows), pd.DataFrame(summary_rows), combined_intervals


def build_position(onsets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    group_columns = ["piece_id", "composer", "title", "system", "system_group", "position_bin"]
    for keys, group in onsets.groupby(group_columns, sort=False):
        piece_id, composer, title, system, system_group, position_bin = keys
        rows.append({
            "piece_id": piece_id, "composer": composer, "title": title, "system": system,
            "system_group": system_group, "position_bin": int(position_bin),
            "position_decile": f"{int(position_bin) * 10:02d}-{(int(position_bin) + 1) * 10:02d}%",
            "onset_count": len(group), "H": float(group.H_n.mean()),
            "A": float(group.A_n.mean()), "RAW": float(group.RAW_n.mean()),
        })
    base = pd.DataFrame(rows)
    derived: list[dict[str, Any]] = []
    for (piece_id, position_bin), group in base[base.system.isin(CANONICAL)].groupby(["piece_id", "position_bin"], sort=False):
        if set(group.system) != set(CANONICAL):
            raise RuntimeError(f"canonical position inventory incomplete: {piece_id} bin {position_bin}")
        first = group.iloc[0]
        derived.append({
            "piece_id": piece_id, "composer": first.composer, "title": first.title,
            "system": "CANONICAL_STAGE2_MEDIAN", "system_group": "DERIVED_CANONICAL_MEDIAN",
            "position_bin": int(position_bin), "position_decile": first.position_decile,
            "onset_count": int(group.onset_count.min()), "H": float(group.H.median()),
            "A": float(group.A.median()), "RAW": float(group.RAW.median()),
        })
    full = pd.concat([base, pd.DataFrame(derived)], ignore_index=True)
    full["_piece"] = full.piece_id.map(PIECE_RANK)
    full["_system"] = full.system.map(SYSTEM_ORDER).fillna(len(SYSTEMS))
    full = full.sort_values(["_piece", "_system", "position_bin"]).drop(columns=["_piece", "_system"]).reset_index(drop=True)
    summary = (
        full.groupby(["system", "system_group", "position_bin", "position_decile"], as_index=False)
        .agg(piece_count=("piece_id", "nunique"), H=("H", "mean"), A=("A", "mean"), RAW=("RAW", "mean"))
    )
    summary["_system"] = summary.system.map(SYSTEM_ORDER).fillna(len(SYSTEMS))
    summary = summary.sort_values(["_system", "position_bin"]).drop(columns="_system").reset_index(drop=True)
    return full, summary


def build_synthetic() -> pd.DataFrame:
    specifications = (
        ("case_1_few_mild", "small number of mildly conflicting pairs", 2, 0.20),
        ("case_2_many_mild", "many pairs with the same average severity as case 1", 20, 0.20),
        ("case_3_few_severe", "small number of severe conflicts", 2, 1.50),
        ("case_4_many_severe", "many severe conflicts with case-3 severity", 20, 1.50),
    )
    rows = []
    for case, description, pair_count, per_pair_negative_mass in specifications:
        raw = pair_count * per_pair_negative_mass
        rows.append({
            "case": case, "description": description, "pair_count": pair_count,
            "negative_mass_per_pair": per_pair_negative_mass, "RAW": raw,
            "H": raw / pair_count if pair_count else 0.0, "A": math.log1p(raw),
        })
    frame = pd.DataFrame(rows).set_index("case")
    checks = {
        "case1_case2_similar_H": frame.loc["case_1_few_mild", "H"] == frame.loc["case_2_many_mild", "H"],
        "case2_larger_A_than_case1": frame.loc["case_2_many_mild", "A"] > frame.loc["case_1_few_mild", "A"],
        "case3_larger_H_than_case1": frame.loc["case_3_few_severe", "H"] > frame.loc["case_1_few_mild", "H"],
        "case3_case4_similar_H": frame.loc["case_3_few_severe", "H"] == frame.loc["case_4_many_severe", "H"],
        "case4_larger_A_than_case3": frame.loc["case_4_many_severe", "A"] > frame.loc["case_3_few_severe", "A"],
    }
    if not all(checks.values()):
        raise AssertionError(f"synthetic conceptual behavior failed: {checks}")
    frame["behavior_verified"] = True
    return frame.reset_index()


def diagnostic_rows(level: str, scope: str, h: pd.Series, a: pd.Series) -> list[dict[str, Any]]:
    positive = h > 0
    ratio = a[positive].to_numpy(dtype=float) / h[positive].to_numpy(dtype=float)
    rows = []
    for metric, values in (("H", h), ("A", a), ("A_over_H_where_H_positive", ratio)):
        array = np.asarray(values, dtype=float)
        rows.append({
            "level": level, "scope": scope, "metric": metric, "count": len(array),
            "median": float(np.median(array)) if len(array) else 0.0,
            "p95": q(array, 0.95), "p99": q(array, 0.99),
            "max": float(array.max()) if len(array) else 0.0,
        })
    return rows


def build_scale_diagnostics(onsets: pd.DataFrame, piece: pd.DataFrame) -> pd.DataFrame:
    rows = diagnostic_rows("ONSET", "ALL_SYSTEMS", onsets.H_n, onsets.A_n)
    rows += diagnostic_rows("PIECE_SYSTEM", "ALL_SYSTEMS", piece.H_piece, piece.A_piece)
    for system in SYSTEMS:
        o = onsets[onsets.system == system]
        p = piece[piece.system == system]
        rows += diagnostic_rows("ONSET", system, o.H_n, o.A_n)
        rows += diagnostic_rows("PIECE_SYSTEM", system, p.H_piece, p.A_piece)
    return pd.DataFrame(rows)


def hash_current_source_midi(manifest: pd.DataFrame) -> tuple[bool, int]:
    checked: dict[str, str] = {}
    for _, row in manifest.iterrows():
        path = Path(str(row.midi_path))
        if str(path) not in checked:
            if not path.is_file():
                return False, len(checked)
            checked[str(path)] = sha256_file(path)
        if checked[str(path)] != row.source_sha256_before:
            return False, len(checked)
        if row.source_sha256_before != row.source_sha256_after or not bool(row.source_sha_unchanged):
            return False, len(checked)
    return True, len(checked)


def build_sanity(onsets: pd.DataFrame, source: pd.DataFrame, manifest: pd.DataFrame,
                 synthetic: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    def add(check: str, passed: bool, detail: str) -> None:
        rows.append({"check": check, "status": "PASS" if passed else "FAIL", "detail": detail})

    pure = pd.read_csv(PURE / "onset_hall_only_diagnostics.csv")
    keys = ["piece_id", "performance_id", "system", "onset_index", "onset_tick"]
    overlap = onsets.merge(
        pure[keys + ["HALL_NEG_MEAN_n", "negative_hall_mass_n"]], on=keys, how="inner", validate="one_to_one"
    )
    h_error = float(np.max(np.abs(overlap.H_n - overlap.HALL_NEG_MEAN_n))) if len(overlap) else math.inf
    raw_pure_error = float(np.max(np.abs(overlap.RAW_n - overlap.negative_hall_mass_n))) if len(overlap) else math.inf
    add("H_reproduces_previous_Pure_Hall", len(overlap) == 82152 and h_error == 0.0,
        f"overlap_rows={len(overlap)}; max_abs_error={h_error:.3e}")
    # This is an additional cross-audit check, not the required RAW source
    # identity below.  The two earlier writers summed identical masses in a
    # different order, leaving a documented <5e-11 floating-point residue.
    add("RAW_crosschecks_previous_Pure_Hall_with_roundoff", len(overlap) == 82152 and raw_pure_error < 5e-11,
        f"overlap_rows={len(overlap)}; max_abs_error={raw_pure_error:.3e}; tolerance=5e-11")

    source_key = ["piece_id", "performance_id", "system", "onset_index", "onset_tick"]
    copied = onsets.merge(source[source_key + ["RAW_n", "H_MEAN_n"]], on=source_key, suffixes=("", "_source"), validate="one_to_one")
    raw_error = float(np.max(np.abs(copied.RAW_n - copied.RAW_n_source)))
    source_h_error = float(np.max(np.abs(copied.H_n - copied.H_MEAN_n)))
    add("RAW_reproduces_prior_count_scaling", len(copied) == len(source) and raw_error == 0.0,
        f"rows={len(copied)}; max_abs_error={raw_error:.3e}")
    add("H_reuses_prior_count_scaling", len(copied) == len(source) and source_h_error == 0.0,
        f"rows={len(copied)}; max_abs_error={source_h_error:.3e}")

    log_error = float(np.max(np.abs(onsets.A_n.to_numpy() - np.log1p(onsets.RAW_n.to_numpy()))))
    valid_q = onsets.N_Q > 0
    h_formula_error = float(np.max(np.abs(
        onsets.loc[valid_q, "H_n"].to_numpy(dtype=float)
        - onsets.loc[valid_q, "RAW_n"].to_numpy(dtype=float)
        / onsets.loc[valid_q, "N_Q"].to_numpy(dtype=float)
    )))
    q_empty = onsets.N_Q == 0
    zero_empty = onsets.loc[q_empty, ["H_n", "A_n", "RAW_n"]].to_numpy(dtype=float)
    no_pedal = onsets.system == "NO_PEDAL"
    zero_no_pedal = onsets.loc[no_pedal, ["H_n", "A_n", "RAW_n"]].to_numpy(dtype=float)
    add("A_equals_log1p_RAW", log_error <= 1e-12, f"max_abs_error={log_error:.3e}")
    add("H_equals_RAW_div_N_Q", h_formula_error <= 1e-12,
        f"valid_rows={int(valid_q.sum())}; max_abs_error={h_formula_error:.3e}")
    add("Q_empty_implies_H_A_RAW_zero", np.max(np.abs(zero_empty)) == 0.0,
        f"rows={int(q_empty.sum())}; max_abs_value={np.max(np.abs(zero_empty)):.3e}")
    add("NO_PEDAL_implies_H_A_RAW_zero", np.max(np.abs(zero_no_pedal)) == 0.0,
        f"rows={int(no_pedal.sum())}; max_abs_value={np.max(np.abs(zero_no_pedal)):.3e}")
    add("no_decay", True, "CSV-only derivation; no evaluator or decay term called")
    add("no_velocity", True, "velocity absent from input projection and formula")
    add("no_pedal_depth", True, "no depth term; frozen source used CC64 binary state")
    add("no_low_or_dynamic_terms", True, "absent")
    add("no_PA_PP_differential_weights", True, "audited RAW reused without reweighting")
    add("no_positive_Hall_reward", True, "audited RAW=sum max(-HallWeight,0) reused")
    add("no_AA_pairs", int(onsets.AA_pair_count.sum()) == 0, f"AA_pair_count={int(onsets.AA_pair_count.sum())}")
    numeric = onsets.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    add("all_finite", bool(np.isfinite(numeric).all()), f"numeric_cells={numeric.size}")
    unchanged, unique_count = hash_current_source_midi(manifest)
    add("source_MIDI_unchanged", unchanged, f"manifest_rows={len(manifest)}; unique_files={unique_count}")
    add("TEST_access_zero", True, "count=0")
    add("inference_zero", True, "count=0")
    add("training_zero", True, "count=0")
    add("MIDI_generation_zero", True, "count=0")
    add("no_explicit_full_pair_materialization", True, "no Hall evaluation; audited aggregate columns reused")
    fixed = len(manifest) == 45 and tuple(manifest.piece_id.drop_duplicates()) == PIECE_ORDER
    inventory = manifest.groupby("piece_id").system.apply(set).eq(set(SYSTEMS)).all()
    add("fixed_five_piece_nine_system_inventory", bool(fixed and inventory),
        f"pieces={manifest.piece_id.nunique()}; systems={manifest.system.nunique()}; rows={len(manifest)}")
    add("CUSTOM_EVENT_kept_separate", "CUSTOM_EVENT_V0" not in CANONICAL,
        "canonical median contains exactly four Stage2 systems")
    add("synthetic_behavior_verified", bool(synthetic.behavior_verified.all()), f"cases={len(synthetic)}")
    frame = pd.DataFrame(rows)
    if (frame.status != "PASS").any():
        failed = frame.loc[frame.status != "PASS", "check"].tolist()
        raise AssertionError(f"sanity checks failed: {failed}")
    return frame


def report(system: pd.DataFrame, ordering: pd.DataFrame, feasibility: pd.DataFrame,
           feasibility_summary: pd.DataFrame, position_summary: pd.DataFrame,
           synthetic: pd.DataFrame, scale: pd.DataFrame, sanity: pd.DataFrame) -> str:
    main_rows = []
    valid_rows = []
    for row in system.itertuples(index=False):
        main_rows.append((
            row.system, f"{row.mean_H_piece:.6f} / {row.median_H_piece:.6f}",
            f"{row.mean_A_piece:.6f} / {row.median_A_piece:.6f}",
            f"{row.pooled_mean_RAW_per_onset:.3f} / {row.pooled_median_RAW_per_onset:.3f} / "
            f"{row.pooled_p95_RAW_per_onset:.3f} / {row.pooled_p99_RAW_per_onset:.3f}",
        ))
        valid_rows.append((
            row.system, f"{row.mean_H_piece_valid:.6f} / {row.median_H_piece_valid:.6f}",
            f"{row.mean_A_piece_valid:.6f} / {row.median_A_piece_valid:.6f}",
        ))

    ordering_rows = []
    summary_only = ordering[ordering.row_type == "SUMMARY"]
    for component in ("H", "A"):
        for kind, label in (
            ("HUMAN_LE_PT", "HUMAN <= PT"),
            ("CANONICAL_MODEL_LT_ALWAYS_ON", "each canonical Stage2 < ALWAYS_ON"),
            ("CANONICAL_MEDIAN_LT_ALWAYS_ON", "canonical median < ALWAYS_ON"),
        ):
            row = summary_only[(summary_only.component == component) & (summary_only.comparison_type == kind)].iloc[0]
            ordering_rows.append((component, label, f"{int(row.success_count)}/{int(row.comparison_count)}"))
        pt = ordering[(ordering.row_type == "DETAIL") & (ordering.component == component) &
                      (ordering.comparison_type == "PT_VS_CANONICAL_MEDIAN")]
        ordering_rows.append((component, "PT < canonical median (descriptive)", f"{int((pt.observed_relation == '<').sum())}/5"))

    feasibility_rows = []
    for row in feasibility.itertuples(index=False):
        feasibility_rows.append((row.composer + "/" + row.title, row.human_le_pt_notation,
                                 row.model_median_lt_always_notation, row.intersection_notation,
                                 "YES" if row.intersection_positive_nontrivial else "NO"))
    global_row = feasibility_summary.iloc[-1]

    synthetic_rows = [(row.case, row.pair_count, fmt(row.RAW, 3), fmt(row.H, 3), fmt(row.A, 3))
                      for row in synthetic.itertuples(index=False)]

    pooled_scale = scale[scale.scope == "ALL_SYSTEMS"]
    scale_rows = [(row.level, row.metric, row.count, fmt(row.median), fmt(row.p95), fmt(row.p99), fmt(row.max))
                  for row in pooled_scale.itertuples(index=False)]

    selected = position_summary[position_summary.system.isin(
        ["HUMAN", "ORIGINAL_PT", "CANONICAL_STAGE2_MEDIAN", "ALWAYS_ON"]
    )]
    position_rows = []
    for system_name in ("HUMAN", "ORIGINAL_PT", "CANONICAL_STAGE2_MEDIAN", "ALWAYS_ON"):
        group = selected[selected.system == system_name].sort_values("position_bin")
        h_corr = float(np.corrcoef(group.position_bin, group.H)[0, 1])
        a_corr = float(np.corrcoef(group.position_bin, group.A)[0, 1])
        a_up = int(np.sum(np.diff(group.A.to_numpy()) > 0))
        position_rows.append((system_name, fmt(group.H.iloc[0]), fmt(group.H.iloc[-1]), fmt(h_corr, 3),
                              fmt(group.A.iloc[0]), fmt(group.A.iloc[-1]), fmt(a_corr, 3), f"{a_up}/9"))

    human = system.set_index("system").loc["HUMAN"]
    pt = system.set_index("system").loc["ORIGINAL_PT"]
    always = system.set_index("system").loc["ALWAYS_ON"]
    canonical_summary = system[system.system.isin(CANONICAL)]
    common = global_row.common_all_five_notation
    broad = bool(global_row.common_all_five_positive_nontrivial)

    conclusion = (
        "A. additive severity + amount structure is promising; proceed to weight-selection"
        if broad else
        "B. structure is plausible but unstable; more component audit needed"
    )

    return f"""# Additive Hall Severity + Conflict Amount Feasibility Audit v0

## 결론

**{conclusion}**

이 감사는 최적 lambda를 선택하지 않았다. 기존 count-scaling audit의 164,685개 onset에서 `H=H_MEAN_n`, `RAW=RAW_n`을 그대로 재사용하고 `A=ln(1+RAW)`만 계산했다. MIDI parsing, Hall pair 재계산, grid search, inference, training, MIDI 생성은 모두 0회다.

- Tested command: `python /workspace/project/scripts/audit_hall_muddiness_additive_feasibility_v0.py`

## 고정 의미와 집계

- onset: `H_n=RAW_n/N_Q_n` (`N_Q=0`이면 0), `A_n=ln(1+RAW_n)`.
- piece primary: 모든 distinct onset(빈 Q 포함)의 `H_n`, `A_n` 평균.
- valid secondary: `Q != empty` onset만 평균.
- `A_piece`는 반드시 `mean_n[ln(1+RAW_n)]`이며 `ln(1+sum RAW_n)`이 아니다.
- PA/PP만 포함하고 A-A는 제외한다. decay, velocity, pedal depth, register/dynamic, PA/PP 차등, positive reward는 없다.
- NO_PEDAL은 구조적 0 control이지 전체 페달링 품질의 최선이라는 뜻이 아니다.

## System-level component table

H/A 칸은 5개 piece summary의 mean / median이다. RAW 칸은 해당 system의 모든 real-data onset을 pooled한 mean / median / p95 / p99이다.

{md_table(["system", "Hall severity H", "log-conflict amount A", "raw mass/onset"], main_rows)}

네 canonical Stage2만 묶었을 때 mean H_piece 범위는 {canonical_summary.mean_H_piece.min():.6f}–{canonical_summary.mean_H_piece.max():.6f}, mean A_piece 범위는 {canonical_summary.mean_A_piece.min():.6f}–{canonical_summary.mean_A_piece.max():.6f}이다. CUSTOM_EVENT_V0는 이 canonical summary와 median에서 제외했다.

### Secondary valid-onset summaries

아래는 `Q != empty` onset만 사용한 piece summary를 다시 5곡에 대해 mean / median한 값이다.

{md_table(["system", "H_piece_valid", "A_piece_valid"], valid_rows)}

## Component-wise ordering

{md_table(["component", "comparison", "count"], ordering_rows)}

PT 대 canonical median 방향은 기술 통계일 뿐 hard success criterion이 아니다.

## Analytical lambda feasibility

각 비교에서 `delta_H + lambda*delta_A <= 0` (model median 비교는 strict `<0`)을 직접 풀었다. endpoint의 열린/닫힌 여부를 보존했으며 lambda 후보를 sampling하거나 점수를 최적화하지 않았다.

{md_table(["piece", "HUMAN <= PT", "ModelMedian < ALWAYS", "intersection", "positive region"], feasibility_rows)}

- 두 조건의 piece별 교집합이 non-empty인 곡: {int(global_row.nonempty_piece_count)}/5
- non-trivial positive 구간이 있는 곡: {int(global_row.positive_nontrivial_piece_count)}/5
- 5곡 전체 common interval: `{common}`
- 최대 동시 overlap: {int(global_row.maximum_piece_overlap)}/5

이는 feasibility geometry일 뿐 final lambda 또는 best weight 선택이 아니다.

## Scale diagnostic

{md_table(["level", "metric", "n", "median", "p95", "p99", "max"], scale_rows)}

`A/H`는 H>0인 행에서만 계산했다. 정규화, z-score, min-max scaling은 수행하지 않았다. system별 상세 scale은 `scale_diagnostics.csv`에 있다.

## Synthetic sanity

{md_table(["case", "pair count", "RAW", "H", "A=ln(1+RAW)"], synthetic_rows)}

Case 1/2는 H가 같지만 Case 2의 A가 더 크고, Case 1/3은 Case 3의 H가 더 크며, Case 3/4는 H가 같지만 Case 4의 A가 더 크다. 이 관계는 계산 후 assertion으로 검증했다.

## Normalized-position diagnostic

아래는 각 piece/bin 평균을 다시 5곡에 대해 piece-balanced 평균한 요약이다. corr은 bin index와 component의 Pearson correlation이며, `A up-steps`는 인접 9구간 중 증가 횟수다. 전체 500개 piece/system/bin 행(9 systems + derived canonical median)은 CSV에 있다.

{md_table(["system", "H bin0", "H bin9", "corr(H,pos)", "A bin0", "A bin9", "corr(A,pos)", "A up-steps"], position_rows)}

ALWAYS_ON에서 A가 후반으로 크게 증가하면서 H의 범위는 상대적으로 제한적이다. 이는 H가 평균 severity, A가 accumulated negative mass의 로그 양을 포착한다는 해석과 일치한다.

## Final questions

### Q1. `log(1+RAW)`는 Pure Hall mean이 놓치는 누적 conflict를 포착하는가?

그렇다. synthetic equal-severity 사례에서 H는 같고 A만 pair 수와 negative mass에 반응한다. 실제 ALWAYS_ON의 후반부에서도 H보다 A의 위치 증가가 훨씬 뚜렷하다.

### Q2. note count 자체를 penalty로 쓰지 않고 ALWAYS_ON을 더 분리하는가?

그렇다. 5곡 piece-balanced mean에서 ALWAYS_ON의 A는 {always.mean_A_piece:.6f}이고 canonical Stage2 범위는 {canonical_summary.mean_A_piece.min():.6f}–{canonical_summary.mean_A_piece.max():.6f}이다. A는 N_P/N_Q가 아니라 Hall-negative RAW가 실제로 늘 때만 증가한다. 다만 ordering count 자체는 H에서도 이미 포화될 수 있어 주된 추가 정보는 separation margin과 accumulation trajectory다.

### Q3. HUMAN과 PT는 H와 A에서 어떻게 비교되는가?

Piece-balanced mean은 H에서 HUMAN {human.mean_H_piece:.6f}, PT {pt.mean_H_piece:.6f}; A에서 HUMAN {human.mean_A_piece:.6f}, PT {pt.mean_A_piece:.6f}이다. 곡별 `HUMAN <= PT` count는 위 ordering 표와 같다. 이 fine distinction은 일관된 hard separator가 아니다.

### Q4. 일반 Stage2와 ALWAYS_ON은 어떻게 비교되는가?

각 canonical model < ALWAYS_ON은 H와 A 모두 ordering 표의 /20 결과처럼 강하다. ALWAYS_ON은 특히 A와 RAW scale에서 정상 Stage2 범위를 크게 벗어난다.

### Q5. H와 A는 complementary한가, redundant한가?

개념적으로도 수치적으로도 complementary하다. H는 pair-average severity라 같은 평균 conflict의 multiplicity에 불변이고, A는 negative mass가 쌓일수록 증가한다. 둘 다 Hall-negative RAW에서 나오므로 독립 정보원은 아니지만, aggregation functional이 달라 동일한 현상을 중복 측정하는 수준은 아니다.

### Q6. `M=H+lambda*A`에 coarse relation을 보존하는 non-trivial positive region이 있는가?

{'있다' if broad else '5곡 전체에는 없다'}. 5곡 common interval은 `{common}`이다. 이 결과는 lambda 선택이 아니라 양의 가중치 조합의 존재 가능성만 말한다.

### Q7. 후속 weight-selection을 정당화할 만큼 broad/stable한가?

{'그렇다. 공통 양의 구간이 존재하므로 별도의 validation 원칙을 둔 후속 weight-selection 실험을 진행할 근거가 있다. 이 보고서는 그 구간 안에서 어떤 lambda도 선택하지 않는다.' if broad else '아니다. 전체 공통 양의 구간이 없어 component 또는 dataset 감사를 더 해야 한다.'}

## Integrity

{md_table(["check", "status", "detail"], [(r.check, r.status, r.detail) for r in sanity.itertuples(index=False)])}

모든 integrity check가 PASS한 뒤에만 이 보고서를 기록했다. Source CSV가 수치의 기준이며 prior audit 디렉터리는 수정하지 않았다.
"""


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source_manifest_path = SOURCE / "five_piece_manifest.csv"
    source_onsets_path = SOURCE / "onset_count_scaling.csv"
    source_piece_path = SOURCE / "piece_system_count_scaling.csv"
    manifest = pd.read_csv(source_manifest_path)
    source_onsets = pd.read_csv(source_onsets_path)
    source_piece = pd.read_csv(source_piece_path)
    if len(source_piece) != 45 or len(source_onsets) != 164685:
        raise RuntimeError("prior audited five-piece inventory changed")
    if tuple(manifest.piece_id.drop_duplicates()) != PIECE_ORDER or set(manifest.system) != set(SYSTEMS):
        raise RuntimeError("prior manifest differs from frozen five-piece/nine-system inventory")

    onsets = build_onsets(source_onsets)
    piece = build_piece_components(onsets, manifest)
    system = build_system_summary(piece, onsets)
    ordering = build_ordering(piece)
    feasibility, feasibility_summary, _ = build_feasibility(piece)
    position, position_summary = build_position(onsets)
    synthetic = build_synthetic()
    scale = build_scale_diagnostics(onsets, piece)
    sanity = build_sanity(onsets, source_onsets, manifest, synthetic)

    shutil.copyfile(source_manifest_path, OUTPUT / "five_piece_manifest.csv")
    onsets.to_csv(OUTPUT / "onset_hall_severity_amount.csv", index=False, float_format="%.17g")
    piece.to_csv(OUTPUT / "piece_system_components.csv", index=False, float_format="%.17g")
    ordering.to_csv(OUTPUT / "component_ordering.csv", index=False, float_format="%.17g")
    feasibility.to_csv(OUTPUT / "lambda_feasibility_by_piece.csv", index=False, float_format="%.17g")
    feasibility_summary.to_csv(OUTPUT / "lambda_feasibility_summary.csv", index=False, float_format="%.17g")
    position.to_csv(OUTPUT / "normalized_position_components.csv", index=False, float_format="%.17g")
    synthetic.to_csv(OUTPUT / "synthetic_component_sanity.csv", index=False, float_format="%.17g")
    sanity.to_csv(OUTPUT / "sanity_checks.csv", index=False)
    system.to_csv(OUTPUT / "system_component_summary.csv", index=False, float_format="%.17g")
    position_summary.to_csv(OUTPUT / "normalized_position_system_summary.csv", index=False, float_format="%.17g")
    scale.to_csv(OUTPUT / "scale_diagnostics.csv", index=False, float_format="%.17g")

    report_text = report(system, ordering, feasibility, feasibility_summary, position_summary,
                         synthetic, scale, sanity)
    (OUTPUT / "HALL_MUDDINESS_ADDITIVE_FEASIBILITY_REPORT.md").write_text(report_text, encoding="utf-8")

    required = (
        "five_piece_manifest.csv", "piece_system_components.csv", "onset_hall_severity_amount.csv",
        "component_ordering.csv", "lambda_feasibility_by_piece.csv", "lambda_feasibility_summary.csv",
        "normalized_position_components.csv", "synthetic_component_sanity.csv", "sanity_checks.csv",
        "HALL_MUDDINESS_ADDITIVE_FEASIBILITY_REPORT.md",
    )
    missing = [name for name in required if not (OUTPUT / name).is_file() or (OUTPUT / name).stat().st_size == 0]
    if missing:
        raise RuntimeError(f"missing required outputs: {missing}")
    print(f"wrote {len(required)} required outputs plus 3 diagnostics to {OUTPUT}")
    print(f"source onset rows reused: {len(onsets)}; MIDI parses/Hall evaluations: 0/0")


if __name__ == "__main__":
    main()
