#!/usr/bin/env python3
"""Train-only 4-class sustain-pedal boundary audit for canonical Stage 2.

The audit deliberately reuses the immutable MAESTRO+ASAP binary Stage-2 token
cache.  It never constructs a validation/test dataset and never changes the
tokenizer or cache.  Raw CC64 events are read only from the same train manifest
that produced that cache.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from miditoolkit import MidiFile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.stage2_encoder_only.dataset import (  # noqa: E402
    NON_PEDAL_FEATURES,
    PEDAL_TOKEN_OFFSET,
    PEDAL_SLOTS,
    TOKENS_PER_NOTE,
)


EXPECTED_CACHE_ID = "85a79e9d10e955b10000f72f6bbc4a29dbd2cc5a4c8cfb1277054a8d45dcb877"
EXPECTED_MANIFEST_SHA256 = "999b10e3d41bca7992b4e46ce85a70bbd2a0b313391247a53da510587efd4f20"
EXPECTED_ASAP_SPLIT_SHA256 = "d1fe379eb123ff7abca93773296f735f40a215dcd6e6363b55756383708c965b"
EXPECTED_SOURCE_PERFORMANCES = {"ASAP-train": 892, "MAESTRO-clean": 1170}
SOURCE_TO_DATASET = {"ASAP-train": "ASAP", "MAESTRO-clean": "MAESTRO"}
DATASETS = ("ASAP", "MAESTRO", "combined")
QUANTILES = (0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def weighted_quantile(counts: np.ndarray, q: float) -> float:
    """Nearest-rank quantile of an integer-valued histogram."""
    total = int(counts.sum())
    if total == 0:
        return float("nan")
    if q <= 0:
        return float(np.flatnonzero(counts)[0])
    if q >= 1:
        return float(np.flatnonzero(counts)[-1])
    rank = max(1, math.ceil(q * total))
    return float(np.searchsorted(np.cumsum(counts), rank, side="left"))


def weighted_median(counts: np.ndarray, lo: int, hi: int) -> float:
    """Usual sample median, including .5 for an even sample count."""
    local = counts[lo : hi + 1]
    total = int(local.sum())
    if total == 0:
        raise ValueError(f"empty class interval [{lo}, {hi}]")
    cumulative = np.cumsum(local)
    left_rank = (total + 1) // 2
    right_rank = (total + 2) // 2
    left = lo + int(np.searchsorted(cumulative, left_rank, side="left"))
    right = lo + int(np.searchsorted(cumulative, right_rank, side="left"))
    return (left + right) / 2.0


def weighted_value_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Nearest-rank quantile for arbitrary non-negative values and weights."""
    order = np.argsort(values)
    sorted_values, sorted_weights = values[order], weights[order]
    total = int(sorted_weights.sum())
    if total == 0:
        return float("nan")
    rank = max(1, math.ceil(q * total))
    return float(sorted_values[np.searchsorted(np.cumsum(sorted_weights), rank, side="left")])


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:.3f}%"


def fmt_num(value: float) -> str:
    return f"{value:.4f}"


def interval_counts(counts: np.ndarray, b_off: int, b_on: int) -> np.ndarray:
    return np.asarray(
        [
            counts[: b_off + 1].sum(),
            counts[b_off + 1 : 64].sum(),
            counts[64 : b_on + 1].sum(),
            counts[b_on + 1 :].sum(),
        ],
        dtype=np.int64,
    )


def candidate_metrics(
    combined: np.ndarray, asap: np.ndarray, maestro: np.ndarray, b_off: int, b_on: int
) -> dict[str, Any]:
    bounds = ((0, b_off), (b_off + 1, 63), (64, b_on), (b_on + 1, 127))
    labels = ("ZERO", "LOW", "HALF", "FULL")
    classes = interval_counts(combined, b_off, b_on)
    total = int(combined.sum())
    proportions = classes / total
    representatives = np.asarray(
        [weighted_median(combined, low, high) for low, high in bounds], dtype=float
    )
    asap_reps = [weighted_median(asap, low, high) for low, high in bounds]
    maestro_reps = [weighted_median(maestro, low, high) for low, high in bounds]
    values = np.arange(128, dtype=float)
    decoded = np.empty(128, dtype=float)
    class_ids = np.empty(128, dtype=np.int8)
    for class_id, (low, high) in enumerate(bounds):
        decoded[low : high + 1] = representatives[class_id]
        class_ids[low : high + 1] = class_id
    errors = np.abs(values - decoded)
    mae = float((combined * errors).sum() / total)
    rmse = float(np.sqrt((combined * errors**2).sum() / total))
    median_error = weighted_value_quantile(errors, combined, 0.5)
    entropy = float(-np.sum(proportions[proportions > 0] * np.log(proportions[proportions > 0])))
    invariant_ok = bool(
        np.all(class_ids[:64] < 2)
        and np.all(class_ids[64:] >= 2)
        and np.array_equal((values >= 64), (class_ids >= 2))
    )
    result: dict[str, Any] = {
        "b_off": b_off,
        "b_on": b_on,
        "zero_count": int(classes[0]),
        "low_count": int(classes[1]),
        "half_count": int(classes[2]),
        "full_count": int(classes[3]),
        "zero_proportion": float(proportions[0]),
        "low_proportion": float(proportions[1]),
        "half_proportion": float(proportions[2]),
        "full_proportion": float(proportions[3]),
        "uniform_l1_distance": float(np.abs(proportions - 0.25).sum()),
        "class_entropy_nats": entropy,
        "smallest_class_proportion": float(proportions.min()),
        "largest_class_proportion": float(proportions.max()),
        "representative_zero_combined": representatives[0],
        "representative_low_combined": representatives[1],
        "representative_half_combined": representatives[2],
        "representative_full_combined": representatives[3],
        "representative_zero_asap": asap_reps[0],
        "representative_low_asap": asap_reps[1],
        "representative_half_asap": asap_reps[2],
        "representative_full_asap": asap_reps[3],
        "representative_zero_maestro": maestro_reps[0],
        "representative_low_maestro": maestro_reps[1],
        "representative_half_maestro": maestro_reps[2],
        "representative_full_maestro": maestro_reps[3],
        "oracle_mae": mae,
        "oracle_rmse": rmse,
        "oracle_median_absolute_error": median_error,
        "oracle_within_5": float(combined[errors <= 5].sum() / total),
        "oracle_within_10": float(combined[errors <= 10].sum() / total),
        "oracle_within_20": float(combined[errors <= 20].sum() / total),
        "off_on_invariant_ok": invariant_ok,
        "off_on_mismatch_count": int(np.count_nonzero((values >= 64) != (class_ids >= 2))),
    }
    for label, (low, high) in zip(labels, bounds):
        result[f"{label.lower()}_range"] = f"{low}-{high}"
    return result


def add_pareto_flags(rows: list[dict[str, Any]]) -> None:
    # Smaller MAE and smaller L1 are both better; equal candidates are all retained.
    for row in rows:
        mae, balance = row["oracle_mae"], row["uniform_l1_distance"]
        row["pareto_optimal"] = not any(
            (other["oracle_mae"] <= mae and other["uniform_l1_distance"] <= balance)
            and (other["oracle_mae"] < mae or other["uniform_l1_distance"] < balance)
            for other in rows
        )


def distribution_rows(
    counts_by_dataset: Mapping[str, np.ndarray], distribution_type: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, counts in counts_by_dataset.items():
        for slot_index in range(counts.shape[0]):
            slot = "flattened" if slot_index == 0 else f"Pedal{slot_index}"
            hist = counts[slot_index]
            total = int(hist.sum())
            cumulative = np.cumsum(hist)
            for value, count in enumerate(hist):
                rows.append(
                    {
                        "distribution_type": distribution_type,
                        "dataset": dataset,
                        "slot": slot,
                        "value": value,
                        "count": int(count),
                        "proportion": float(count / total) if total else float("nan"),
                        "cumulative_count": int(cumulative[value]),
                        "cumulative_proportion": float(cumulative[value] / total) if total else float("nan"),
                        "total": total,
                    }
                )
    return rows


def summarize_histogram(counts: np.ndarray) -> dict[str, Any]:
    total = int(counts.sum())
    if not total:
        raise ValueError("empty distribution")
    return {
        "total": total,
        "zero": float(counts[0] / total),
        "one_to_63": float(counts[1:64].sum() / total),
        "64_to_126": float(counts[64:127].sum() / total),
        "127": float(counts[127] / total),
        "off": float(counts[:64].sum() / total),
        "on": float(counts[64:].sum() / total),
        "quantiles": {f"p{int(q * 100):02d}": weighted_quantile(counts, q) for q in QUANTILES},
        "spikes": [
            {"value": int(value), "count": int(counts[value]), "proportion": float(counts[value] / total)}
            for value in sorted(range(128), key=lambda value: (-int(counts[value]), value))[:10]
        ],
    }


def plot_histograms(raw: Mapping[str, np.ndarray], targets: Mapping[str, np.ndarray], output: Path) -> None:
    values = np.arange(128)
    colors = {"ASAP": "#1f77b4", "MAESTRO": "#ff7f0e", "combined": "#2ca02c"}
    fig, axis = plt.subplots(figsize=(13, 5.5))
    for dataset in DATASETS:
        axis.plot(values, raw[dataset] / raw[dataset].sum(), label=dataset, color=colors[dataset])
    axis.set(title="Raw CC64 event distribution (linear)", xlabel="CC64 value", ylabel="event proportion", xlim=(0, 127))
    axis.legend(); axis.grid(alpha=0.25); fig.tight_layout(); fig.savefig(output / "raw_cc64_histogram.png", dpi=180); plt.close(fig)
    fig, axis = plt.subplots(figsize=(13, 5.5))
    for dataset in DATASETS:
        axis.plot(values[1:127], raw[dataset][1:127] / raw[dataset].sum(), label=dataset, color=colors[dataset])
    axis.set_yscale("log")
    axis.set(title="Raw CC64 intermediate event distribution (log scale)", xlabel="CC64 value (1–126)", ylabel="event proportion (log)", xlim=(1, 126))
    axis.legend(); axis.grid(alpha=0.25); fig.tight_layout(); fig.savefig(output / "raw_cc64_intermediate_log.png", dpi=180); plt.close(fig)
    fig, axis = plt.subplots(figsize=(13, 5.5))
    for dataset in DATASETS:
        axis.plot(values, np.cumsum(raw[dataset]) / raw[dataset].sum(), label=dataset, color=colors[dataset])
    axis.set(title="Raw CC64 event CDF", xlabel="CC64 value", ylabel="cumulative proportion", xlim=(0, 127), ylim=(0, 1))
    axis.legend(); axis.grid(alpha=0.25); fig.tight_layout(); fig.savefig(output / "raw_cc64_cdf.png", dpi=180); plt.close(fig)
    fig, axis = plt.subplots(figsize=(13, 5.5))
    for dataset in DATASETS:
        axis.plot(values, targets[dataset][0] / targets[dataset][0].sum(), label=dataset, color=colors[dataset])
    axis.set(title="Stage 2 flattened Pedal1–4 target distribution", xlabel="raw target value", ylabel="target proportion", xlim=(0, 127))
    axis.legend(); axis.grid(alpha=0.25); fig.tight_layout(); fig.savefig(output / "pedal_target_histogram.png", dpi=180); plt.close(fig)
    fig, axis = plt.subplots(figsize=(13, 5.5))
    for dataset in DATASETS:
        axis.plot(values, np.cumsum(targets[dataset][0]) / targets[dataset][0].sum(), label=dataset, color=colors[dataset])
    axis.set(title="Stage 2 flattened Pedal1–4 target CDF", xlabel="raw target value", ylabel="cumulative proportion", xlim=(0, 127), ylim=(0, 1))
    axis.legend(); axis.grid(alpha=0.25); fig.tight_layout(); fig.savefig(output / "pedal_target_cdf.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
    for slot, axis in enumerate(axes.flat, start=1):
        for dataset in DATASETS:
            axis.plot(values, targets[dataset][slot] / targets[dataset][slot].sum(), label=dataset, color=colors[dataset])
        axis.set_title(f"Pedal{slot}")
        axis.grid(alpha=0.25)
    axes[0, 0].legend(); fig.supxlabel("raw target value"); fig.supylabel("target proportion")
    fig.suptitle("Slot-specific Stage 2 target distributions", y=1.01); fig.tight_layout(); fig.savefig(output / "pedal_target_slot_distributions.png", dpi=180, bbox_inches="tight"); plt.close(fig)


def plot_tradeoff(candidates: list[dict[str, Any]], output: Path) -> None:
    fig, axis = plt.subplots(figsize=(8.5, 6.5))
    x = [row["uniform_l1_distance"] for row in candidates]
    y = [row["oracle_mae"] for row in candidates]
    axis.scatter(x, y, s=9, alpha=0.25, label="all valid fixed-64 candidates")
    pareto = [row for row in candidates if row["pareto_optimal"]]
    axis.scatter([row["uniform_l1_distance"] for row in pareto], [row["oracle_mae"] for row in pareto], s=28, c="#d62728", label="Pareto-optimal")
    for row, label in ((next(r for r in candidates if (r["b_off"], r["b_on"]) == (31, 95)), "equal-width"), (next(r for r in candidates if (r["b_off"], r["b_on"]) == (20, 109)), "endpoint-region")):
        axis.scatter([row["uniform_l1_distance"]], [row["oracle_mae"]], s=45, c="black", marker="x")
        axis.annotate(label, (row["uniform_l1_distance"], row["oracle_mae"]), xytext=(5, 4), textcoords="offset points")
    axis.set(title="Fixed-64 candidate trade-off", xlabel="uniform class-proportion L1 distance (lower is better)", ylabel="oracle quantization MAE (lower is better)")
    axis.grid(alpha=0.25); axis.legend(); fig.tight_layout(); fig.savefig(output / "candidate_mae_balance_tradeoff.png", dpi=180); plt.close(fig)


def markdown_table(rows: Iterable[Mapping[str, Any]], headers: list[tuple[str, str]]) -> str:
    data = list(rows)
    body = ["| " + " | ".join(title for _, title in headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in data:
        body.append("| " + " | ".join(str(row[key]) for key, _ in headers) + " |")
    return "\n".join(body)


def candidate_display(row: Mapping[str, Any]) -> dict[str, str]:
    reps = ", ".join(
        f"{label}={row[f'representative_{label.lower()}_combined']:g}"
        for label in ("ZERO", "LOW", "HALF", "FULL")
    )
    return {
        "b_off": str(row["b_off"]), "b_on": str(row["b_on"]),
        "zero": fmt_pct(row["zero_proportion"]), "low": fmt_pct(row["low_proportion"]),
        "half": fmt_pct(row["half_proportion"]), "full": fmt_pct(row["full_proportion"]),
        "representatives": reps, "mae": fmt_num(row["oracle_mae"]), "rmse": fmt_num(row["oracle_rmse"]),
        "balance": f"L1={row['uniform_l1_distance']:.4f}; H={row['class_entropy_nats']:.4f}",
    }


def write_report(
    output: Path, provenance: Mapping[str, Any], raw: Mapping[str, np.ndarray], targets: Mapping[str, np.ndarray], candidates: list[dict[str, Any]], shortlist: list[dict[str, Any]]
) -> None:
    raw_summary = {name: summarize_histogram(raw[name]) for name in DATASETS}
    target_summary = {name: summarize_histogram(targets[name][0]) for name in DATASETS}
    refs = [next(row for row in candidates if (row["b_off"], row["b_on"]) == pair) for pair in ((31, 95), (20, 109))]
    best_mae = min(candidates, key=lambda row: (row["oracle_mae"], row["uniform_l1_distance"], row["b_off"], row["b_on"]))
    best_balance = min(candidates, key=lambda row: (row["uniform_l1_distance"], -row["class_entropy_nats"], row["oracle_mae"], row["b_off"], row["b_on"]))
    pareto = sorted((row for row in candidates if row["pareto_optimal"]), key=lambda row: (row["uniform_l1_distance"], row["oracle_mae"]))
    lines = [
        "# Stage 2 4-class fixed-64 boundary audit (v0)", "",
        "## Scope and provenance", "",
        "This report is a train-only boundary audit, not a model experiment and not a tokenizer change. It reuses the current canonical 2-class Stage 2 `MAESTRO-clean + ASAP-train` cache, whose tokens were produced by the pinned official Pianist Transformer `midi_to_ids` tokenizer. Pedal targets are cache token columns Pedal1–Pedal4 minus token offset 5,261.", "",
        f"- Cache ID: `{provenance['cache_id']}`; cache token SHA-256: `{provenance['token_sha256']}`.",
        f"- Train performances: {provenance['performance_counts']['ASAP']:,} ASAP + {provenance['performance_counts']['MAESTRO']:,} MAESTRO = {provenance['performance_counts']['combined']:,}; train notes: {provenance['note_count']:,}; Pedal1–4 targets: {provenance['target_count']:,}.",
        "- The immutable canonical train manifest/index defines the Stage 2 training corpus. MAESTRO retains its source-dataset split metadata, while all ASAP records are explicitly in ASAP `train` and disjoint from ASAP validation/test paths. No validation/test token array, performance index, or MIDI file was opened.",
        "- Raw CC64 is counted only from each matching train-manifest performance MIDI. It is supplementary: boundary ranking uses the tokenized Stage 2 target distribution below.", "",
        "## Raw CC64 event distribution (supplementary)", "",
        "`raw_cc64_distribution.csv` contains counts, proportions, and CDFs for every 0–127 value. The figures show the full linear histogram, an intermediate-only log-scale view, and the CDF.", "",
    ]
    raw_rows = []
    for dataset in DATASETS:
        summary = raw_summary[dataset]
        raw_rows.append({"dataset": dataset, "events": f"{summary['total']:,}", "0": fmt_pct(summary['zero']), "1–63": fmt_pct(summary['one_to_63']), "64–126": fmt_pct(summary['64_to_126']), "127": fmt_pct(summary['127']), "spikes": ", ".join(f"{x['value']} ({fmt_pct(x['proportion'])})" for x in summary['spikes'][:5])})
    lines += [markdown_table(raw_rows, [("dataset", "dataset"), ("events", "events"), ("0", "0"), ("1–63", "1–63"), ("64–126", "64–126"), ("127", "127"), ("spikes", "top spikes")]), "", "Quantiles (nearest-rank on raw event values):", ""]
    q_rows = []
    for dataset in DATASETS:
        q_rows.append({"dataset": dataset, **{key: f"{value:g}" for key, value in raw_summary[dataset]["quantiles"].items()}})
    q_headers = [("dataset", "dataset")] + [(f"p{int(q * 100):02d}", f"p{int(q * 100):02d}") for q in QUANTILES]
    lines += [markdown_table(q_rows, q_headers), "", "## Actual Stage 2 Pedal1–4 targets (primary)", "", "`pedal_target_distribution.csv` provides the 128-bin count/proportion/CDF for flattened Pedal1–4 and each slot, separately for ASAP, MAESTRO, and combined training data. The following flattened view is the primary evidence for candidate boundaries.", ""]
    target_rows = []
    for dataset in DATASETS:
        summary = target_summary[dataset]
        target_rows.append({"dataset": dataset, "targets": f"{summary['total']:,}", "0": fmt_pct(summary['zero']), "1–63": fmt_pct(summary['one_to_63']), "64–126": fmt_pct(summary['64_to_126']), "127": fmt_pct(summary['127']), "off/on": f"{fmt_pct(summary['off'])} / {fmt_pct(summary['on'])}", "spikes": ", ".join(f"{x['value']} ({fmt_pct(x['proportion'])})" for x in summary['spikes'][:5])})
    lines += [markdown_table(target_rows, [("dataset", "dataset"), ("targets", "targets"), ("0", "0"), ("1–63", "1–63"), ("64–126", "64–126"), ("127", "127"), ("off/on", "OFF / ON"), ("spikes", "top spikes")]), "", "Target quantiles (nearest-rank):", ""]
    q_rows = [{"dataset": dataset, **{key: f"{value:g}" for key, value in target_summary[dataset]["quantiles"].items()}} for dataset in DATASETS]
    lines += [markdown_table(q_rows, q_headers), "", "### Slot consistency", ""]
    slot_rows = []
    for dataset in DATASETS:
        for slot in range(1, 5):
            summary = summarize_histogram(targets[dataset][slot])
            slot_rows.append({"dataset": dataset, "slot": f"Pedal{slot}", "n": f"{summary['total']:,}", "OFF": fmt_pct(summary['off']), "ON": fmt_pct(summary['on']), "p50": f"{summary['quantiles']['p50']:g}", "0 / 127": f"{fmt_pct(summary['zero'])} / {fmt_pct(summary['127'])}"})
    lines += [markdown_table(slot_rows, [("dataset", "dataset"), ("slot", "slot"), ("n", "targets"), ("OFF", "OFF"), ("ON", "ON"), ("p50", "p50"), ("0 / 127", "0 / 127")]), "", "The slot plot and table should be used to judge a shared boundary. This audit keeps one shared candidate boundary by design, while preserving slot-level evidence instead of assuming the four slots are identical.", "", "## Fixed-64 candidate scan", "", "All 3,969 non-empty integer partitions were scanned: `b_off = 0…62`, `b_on = 64…126`. Every candidate has `ZERO=0…b_off`, `LOW=b_off+1…63`, `HALF=64…b_on`, `FULL=b_on+1…127`.", "", "The balance score is the L1 distance to uniform 25% proportions (lower is better); entropy, smallest, and largest class proportions are also in `boundary_candidates.csv`. Balance is a diagnostic only—not a choice rule.", "", "### Required and scan-selected candidates", ""]
    selected_rows = [candidate_display(row) for row in [refs[0], refs[1], best_mae, best_balance]]
    lines += [markdown_table(selected_rows, [("b_off", "b_off"), ("b_on", "b_on"), ("zero", "ZERO %"), ("low", "LOW %"), ("half", "HALF %"), ("full", "FULL %"), ("representatives", "combined median representatives"), ("mae", "Oracle MAE"), ("rmse", "RMSE"), ("balance", "Balance")]), "", f"- Lowest oracle MAE in the full scan: `b_off={best_mae['b_off']}`, `b_on={best_mae['b_on']}`.", f"- Best uniform-balance L1 in the full scan: `b_off={best_balance['b_off']}`, `b_on={best_balance['b_on']}`.", f"- Pareto-optimal candidates (MAE vs uniform L1): {len(pareto)}; see `boundary_candidates.csv` (`pareto_optimal=true`) and `candidate_mae_balance_tradeoff.png`.", "", "### Representative policy and oracle fidelity", "", "For every candidate and each class, the decoder representative is the median of combined train Pedal1–4 raw values assigned to that class—not the interval midpoint. The CSV also records ASAP-only and MAESTRO-only medians for every class. Oracle MAE/RMSE/tolerance metrics are therefore the unavoidable error of this 4-class quantizer under its own train-data medians; they are not model predictions or validation metrics.", "", "### Fixed 64 OFF/ON invariant", "", "The program verifies every candidate has zero OFF/ON mismatches: `ZERO + LOW` is always raw `<64`, while `HALF + FULL` is always raw `>=64`. Collapsing a four-class token to OFF/ON therefore exactly equals thresholding the original raw target at 64. Consequently binary transition labels at this threshold cannot change when only `b_off` or `b_on` changes; only within-side depth quantization changes.", "", "## Candidate shortlist (not a final tokenizer decision)", ""]
    short_rows = [candidate_display(row) for row in shortlist]
    lines += [markdown_table(short_rows, [("b_off", "b_off"), ("b_on", "b_on"), ("zero", "ZERO %"), ("low", "LOW %"), ("half", "HALF %"), ("full", "FULL %"), ("representatives", "representatives"), ("mae", "Oracle MAE"), ("rmse", "RMSE"), ("balance", "Balance")]), ""]
    for row in shortlist:
        difference = max(abs(row[f"representative_{label}_asap"] - row[f"representative_{label}_maestro"]) for label in ("zero", "low", "half", "full"))
        lines.append(f"- **({row['b_off']}, {row['b_on']})**: information preservation is summarized by oracle MAE {row['oracle_mae']:.3f}; class imbalance ranges from {fmt_pct(row['smallest_class_proportion'])} to {fmt_pct(row['largest_class_proportion'])}. The integer boundaries are directly interpretable under fixed 64; maximum ASAP-vs-MAESTRO class-median difference is {difference:g} CC64 units.")
    lines += ["", "No canonical boundary is selected by this audit. The combined corpus is the primary distribution for any later choice, while the separated ASAP/MAESTRO rows expose cross-dataset sensitivity.", "", "## Files", "", "- `raw_cc64_distribution.csv`: raw train-MIDI CC64 distributions.", "- `pedal_target_distribution.csv`: official-tokenizer Pedal1–4 distributions, flattened and per slot.", "- `boundary_candidates.csv`: all fixed-64 candidates with balance, representatives, oracle fidelity, Pareto flag, and invariant checks.", "- `boundary_shortlist.csv`: the 3–5 candidates summarized above.", "- `raw_cc64_*.png`, `pedal_target_*.png`, `candidate_mae_balance_tradeoff.png`: audit visualizations."]
    (output / "BOUNDARY_AUDIT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=Path("/workspace/project/analysis/stage2_binary_v0/train_setup_v0/shared_cache"))
    parser.add_argument("--manifest", type=Path, default=Path("/workspace/project/analysis/stage2_binary_v0/data_prep_v1/train_manifest.csv"))
    parser.add_argument("--asap-split", type=Path, default=Path("/workspace/project/analysis/stage2_encoder_only_v0/asap_split.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("/workspace/project/analysis/stage2_4class_boundary_audit_v0"))
    parser.add_argument("--overwrite", action="store_true", help="replace only this audit output directory")
    args = parser.parse_args()
    cache_root, manifest_path, split_path, output = (args.cache_root.resolve(), args.manifest.resolve(), args.asap_split.resolve(), args.output_dir.resolve())
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists: {output}; use --overwrite for this audit directory only")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    metadata = json.loads((cache_root / "cache_statistics.json").read_text(encoding="utf-8"))
    if not metadata.get("completed") or metadata.get("cache_id") != EXPECTED_CACHE_ID:
        raise RuntimeError("cache is not the pinned canonical binary Stage-2 cache")
    if sha256_file(manifest_path) != EXPECTED_MANIFEST_SHA256 or sha256_file(split_path) != EXPECTED_ASAP_SPLIT_SHA256:
        raise RuntimeError("train manifest or ASAP split differs from canonical provenance")
    index = read_csv(cache_root / metadata["splits"]["train"]["index_file"])
    manifest = read_csv(manifest_path)
    asap_split = read_csv(split_path)
    if len(index) != len(manifest) != sum(EXPECTED_SOURCE_PERFORMANCES.values()):
        raise RuntimeError("canonical train performance count mismatch")
    if Counter(row["source"] for row in index) != EXPECTED_SOURCE_PERFORMANCES:
        raise RuntimeError("cache sources differ from canonical corpus")
    # ``dataset_split`` retains the source dataset's own label (notably for
    # MAESTRO); canonical Stage-2 membership is defined by this immutable
    # train manifest. ASAP additionally has an explicit held-out split guard below.
    if [(row["source"], row["performance_path"]) for row in index] != [(row["source"], row["performance_path"]) for row in manifest]:
        raise RuntimeError("cache index and canonical train manifest ordering differ")
    asap_train_paths = {row["performance_path"] for row in manifest if row["source"] == "ASAP-train"}
    split_by_path = {row["performance_path"]: row["split"] for row in asap_split}
    if any(split_by_path.get(path) != "train" for path in asap_train_paths):
        raise RuntimeError("ASAP train manifest contains validation/test/unknown path")
    heldout = {row["performance_path"] for row in asap_split if row["split"] != "train"}
    if asap_train_paths & heldout:
        raise RuntimeError("ASAP held-out path reached the train manifest")
    token_path = cache_root / metadata["splits"]["train"]["tokens_file"]
    tokens = np.load(token_path, mmap_mode="r")
    if tokens.dtype != np.int16 or tokens.shape != (int(metadata["splits"]["train"]["notes"]), TOKENS_PER_NOTE):
        raise RuntimeError("invalid canonical train token array")
    target_counts = {dataset: np.zeros((5, 128), dtype=np.int64) for dataset in DATASETS}
    raw_counts = {dataset: np.zeros(128, dtype=np.int64) for dataset in DATASETS}
    for row in index:
        dataset = SOURCE_TO_DATASET.get(row["source"])
        if dataset is None:
            raise RuntimeError(f"unexpected source: {row['source']}")
        offset, notes = int(row["token_offset"]), int(row["notes"])
        pedal = tokens[offset : offset + notes, NON_PEDAL_FEATURES:].astype(np.int32) - PEDAL_TOKEN_OFFSET
        if pedal.shape != (notes, PEDAL_SLOTS) or not np.all((pedal >= 0) & (pedal <= 127)):
            raise RuntimeError(f"invalid cached pedal targets: {row['performance_path']}")
        for slot in range(PEDAL_SLOTS):
            values = pedal[:, slot]
            target_counts[dataset][slot + 1] += np.bincount(values, minlength=128)
        target_counts[dataset][0] += np.bincount(pedal.reshape(-1), minlength=128)
        midi = MidiFile(str(Path(row["performance_absolute_path"])))
        values = [change.value for instrument in midi.instruments for change in instrument.control_changes if change.number == 64]
        if values:
            raw_counts[dataset] += np.bincount(values, minlength=128)
    for dataset in ("ASAP", "MAESTRO"):
        target_counts["combined"] += target_counts[dataset]
        raw_counts["combined"] += raw_counts[dataset]
    if any(not np.array_equal(target_counts[name][0], target_counts[name][1:].sum(axis=0)) for name in DATASETS):
        raise RuntimeError("flattened Pedal1–4 count mismatch")
    candidates = [candidate_metrics(target_counts["combined"][0], target_counts["ASAP"][0], target_counts["MAESTRO"][0], b_off, b_on) for b_off in range(0, 63) for b_on in range(64, 127)]
    add_pareto_flags(candidates)
    if not all(row["off_on_invariant_ok"] and row["off_on_mismatch_count"] == 0 for row in candidates):
        raise RuntimeError("fixed-64 invariant failed")
    best_mae = min(candidates, key=lambda row: (row["oracle_mae"], row["uniform_l1_distance"], row["b_off"], row["b_on"]))
    best_balance = min(candidates, key=lambda row: (row["uniform_l1_distance"], -row["class_entropy_nats"], row["oracle_mae"], row["b_off"], row["b_on"]))
    references = [next(row for row in candidates if (row["b_off"], row["b_on"]) == pair) for pair in ((31, 95), (20, 109))]
    shortlist: list[dict[str, Any]] = []
    for row in [*references, best_mae, best_balance, *sorted((r for r in candidates if r["pareto_optimal"]), key=lambda r: (abs(r["b_off"] - 31) + abs(r["b_on"] - 95), r["oracle_mae"]))]:
        if (row["b_off"], row["b_on"]) not in {(selected["b_off"], selected["b_on"]) for selected in shortlist}:
            shortlist.append(row)
        if len(shortlist) == 5:
            break
    provenance = {
        "cache_id": metadata["cache_id"], "token_sha256": metadata["splits"]["train"]["tokens_sha256_int16_le"],
        "performance_counts": {"ASAP": int(sum(row["source"] == "ASAP-train" for row in index)), "MAESTRO": int(sum(row["source"] == "MAESTRO-clean" for row in index)), "combined": len(index)},
        "note_count": int(tokens.shape[0]), "target_count": int(target_counts["combined"][0].sum()),
    }
    raw_rows = distribution_rows({name: raw_counts[name][None, :] for name in DATASETS}, "raw_cc64_event")
    write_csv(output / "raw_cc64_distribution.csv", raw_rows)
    write_csv(output / "pedal_target_distribution.csv", distribution_rows(target_counts, "stage2_pedal_target"))
    write_csv(output / "boundary_candidates.csv", candidates)
    write_csv(output / "boundary_shortlist.csv", shortlist)
    plot_histograms(raw_counts, target_counts, output)
    plot_tradeoff(candidates, output)
    write_report(output, provenance, raw_counts, target_counts, candidates, shortlist)
    status = {"completed": True, "validation_or_test_used": False, "cache_reused": True, "candidate_count": len(candidates), "off_on_invariant_failures": 0, "provenance": provenance}
    (output / "audit_status.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
