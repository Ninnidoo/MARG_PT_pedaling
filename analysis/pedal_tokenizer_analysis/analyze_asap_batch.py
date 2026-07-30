from __future__ import annotations

import argparse
import csv
import math
import random
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from analyze_pedal_tokenizer import (
    REPEDAL_COLUMNS,
    aggregate_density_stats,
    aggregate_ioi_stats,
    analyze_one_file,
    import_miditoolkit,
    load_official_components,
    original_duration_seconds,
    repo_root_from_script,
    safe_stem,
    scan_candidate,
    write_csv,
    write_json,
)
from pedal_utils import (
    TARGET_TICKS_PER_SECOND,
    detect_repedals,
    detect_threshold_transitions,
    extract_normalized_cc64_events,
    extract_normalized_notes,
    extract_original_cc64_events,
    ioi_bins,
    match_repedals,
    match_transitions,
    midi_duration_seconds,
    ticks_to_seconds,
    token_rows_from_ids,
)


DEFAULT_RANDOM_SEED = 20260727
DEFAULT_MIN_CC64_EVENTS = 100
DEFAULT_MAX_FILES = 100

THRESHOLD_DEFINITIONS = [
    {"definition": "A", "high_threshold": 96, "low_threshold": 31, "max_duration_ms": 300},
    {"definition": "B", "high_threshold": 96, "low_threshold": 47, "max_duration_ms": 300},
    {"definition": "C", "high_threshold": 80, "low_threshold": 31, "max_duration_ms": 300},
    {"definition": "D", "high_threshold": 96, "low_threshold": 31, "max_duration_ms": 500},
]


def read_asap_metadata(asap_root: Path) -> tuple[dict[str, dict[str, str]], set[str]]:
    metadata_path = asap_root / "metadata.csv"
    performance_rows: dict[str, dict[str, str]] = {}
    score_paths: set[str] = set()
    if not metadata_path.exists():
        return performance_rows, score_paths

    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            performance = (row.get("midi_performance") or "").replace("\\", "/")
            score = (row.get("midi_score") or "").replace("\\", "/")
            if performance:
                performance_rows[performance] = row
            if score:
                score_paths.add(score)
    return performance_rows, score_paths


def is_score_midi(path: Path, relative_path: str, metadata_score_paths: set[str]) -> bool:
    name = path.name.lower()
    stem = path.stem.lower()
    if relative_path in metadata_score_paths:
        return True
    return name == "midi_score.mid" or stem.endswith("_score") or "score" in stem


def infer_metadata_from_path(relative_path: str) -> dict[str, str]:
    parts = relative_path.split("/")
    composer = parts[0] if parts else "unknown"
    folder = "/".join(parts[:-1]) if len(parts) > 1 else ""
    title = parts[-2] if len(parts) > 1 else Path(relative_path).stem
    return {
        "composer": composer,
        "title": title,
        "folder": folder,
        "midi_performance": relative_path,
    }


def discover_asap_inventory(repo_root: Path, MidiFile) -> list[dict[str, Any]]:
    asap_root = repo_root / "third_party" / "PianistTransformer" / "data" / "midis" / "asap-dataset-master"
    performance_rows, score_paths = read_asap_metadata(asap_root)
    rows: list[dict[str, Any]] = []
    if not asap_root.exists():
        return rows

    for path in sorted(asap_root.rglob("*.mid")) + sorted(asap_root.rglob("*.midi")):
        relative_path = path.relative_to(asap_root).as_posix()
        if is_score_midi(path, relative_path, score_paths):
            continue

        metadata = performance_rows.get(relative_path, infer_metadata_from_path(relative_path))
        source = "asap_metadata_performance" if relative_path in performance_rows else "asap_recursive_non_score"
        try:
            scan = scan_candidate(path.resolve(), source, MidiFile)
            if source == "asap_recursive_non_score" and int(scan.get("num_cc64_events", 0)) == 0:
                continue
            row = {
                "relative_path": relative_path,
                "composer": metadata.get("composer", ""),
                "title": metadata.get("title", ""),
                "folder": metadata.get("folder", ""),
                "candidate_source": source,
                **scan,
            }
            rows.append(row)
        except Exception as exc:
            rows.append(
                {
                    "relative_path": relative_path,
                    "composer": metadata.get("composer", ""),
                    "title": metadata.get("title", ""),
                    "folder": metadata.get("folder", ""),
                    "file": str(path.resolve()),
                    "candidate_source": source,
                    "scan_error": f"{type(exc).__name__}: {exc}",
                    "selected": False,
                }
            )
    return rows


def stratified_select(
    inventory: Sequence[dict[str, Any]],
    max_files: int,
    min_cc64_events: int,
    seed: int,
) -> tuple[list[dict[str, Any]], int, str]:
    eligible = [
        row
        for row in inventory
        if not row.get("scan_error") and int(row.get("num_cc64_events", 0)) >= min_cc64_events
    ]
    threshold_used = min_cc64_events
    reason = f"CC64 >= {min_cc64_events}"
    if not eligible:
        fallback = [
            row
            for row in inventory
            if not row.get("scan_error") and int(row.get("num_cc64_events", 0)) > 0
        ]
        eligible = fallback
        threshold_used = 1
        reason = "No files met CC64 >= 100; relaxed to CC64 > 0."

    rng = random.Random(seed)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        grouped[row.get("composer") or "unknown"].append(row)

    for group_rows in grouped.values():
        group_rows.sort(key=lambda r: (-int(r.get("num_cc64_events", 0)), str(r.get("relative_path", ""))))
        rng.shuffle(group_rows)

    selected: list[dict[str, Any]] = []
    composers = sorted(grouped)
    while len(selected) < min(max_files, len(eligible)):
        changed = False
        for composer in composers:
            if grouped[composer] and len(selected) < max_files:
                selected.append(grouped[composer].pop(0))
                changed = True
        if not changed:
            break

    selected_files = {row["file"] for row in selected}
    for row in inventory:
        row["selected"] = row.get("file") in selected_files
        row["selection_threshold_used"] = threshold_used
        row["selection_reason"] = reason if row["selected"] else ""
        row["random_seed"] = seed if row["selected"] else ""
    return selected, threshold_used, reason


def compute_repedal_counts_for_definition(
    candidate: dict[str, Any],
    components: dict[str, Any],
    MidiFile,
    definition: dict[str, Any],
    match_tolerance_ms: int,
) -> dict[str, Any]:
    midi_obj = MidiFile(candidate["file"])
    normalized_raw = components["normalize_midi"](midi_obj)
    ids = components["midi_to_ids"](components["config"], midi_obj, normalize=True)
    reconstructed = components["ids_to_midi"](components["config"], ids)
    raw_events = extract_normalized_cc64_events(normalized_raw)
    token_events = extract_normalized_cc64_events(reconstructed)

    raw_repedals = detect_repedals(
        raw_events,
        high_threshold=definition["high_threshold"],
        low_threshold=definition["low_threshold"],
        max_duration_ms=definition["max_duration_ms"],
    )
    token_repedals = detect_repedals(
        token_events,
        high_threshold=definition["high_threshold"],
        low_threshold=definition["low_threshold"],
        max_duration_ms=definition["max_duration_ms"],
    )
    matched_rows, matched_token_count = match_repedals(
        raw_repedals,
        token_repedals,
        tolerance_ms=match_tolerance_ms,
    )
    preserved = sum(1 for row in matched_rows if row["preserved"] is True)
    raw_count = len(raw_repedals)
    token_count = len(token_repedals)
    return {
        "file": candidate["file"],
        "composer": candidate.get("composer", ""),
        "definition": definition["definition"],
        "high_threshold": definition["high_threshold"],
        "low_threshold": definition["low_threshold"],
        "max_duration_ms": definition["max_duration_ms"],
        "raw_repedal_count": raw_count,
        "token_repedal_count": token_count,
        "preserved_repedal_count": preserved,
        "lost_repedal_count": raw_count - preserved,
        "matched_token_repedal_count": matched_token_count,
        "repedal_recall": preserved / raw_count if raw_count else math.nan,
        "repedal_precision": matched_token_count / token_count if token_count else math.nan,
    }


def threshold_sensitivity(
    selected: Sequence[dict[str, Any]],
    components: dict[str, Any],
    MidiFile,
    match_tolerance_ms: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_file_rows: list[dict[str, Any]] = []
    for candidate in selected:
        for definition in THRESHOLD_DEFINITIONS:
            per_file_rows.append(
                compute_repedal_counts_for_definition(candidate, components, MidiFile, definition, match_tolerance_ms)
            )

    summary_rows: list[dict[str, Any]] = []
    for definition in THRESHOLD_DEFINITIONS:
        rows = [row for row in per_file_rows if row["definition"] == definition["definition"]]
        total_raw = sum(int(row["raw_repedal_count"]) for row in rows)
        total_preserved = sum(int(row["preserved_repedal_count"]) for row in rows)
        recalls = [float(row["repedal_recall"]) for row in rows if not math.isnan(float(row["repedal_recall"]))]
        summary_rows.append(
            {
                "definition": definition["definition"],
                "high_threshold": definition["high_threshold"],
                "low_threshold": definition["low_threshold"],
                "max_duration_ms": definition["max_duration_ms"],
                "total_raw_repedal_count": total_raw,
                "total_preserved": total_preserved,
                "total_lost": total_raw - total_preserved,
                "micro_average_recall": total_preserved / total_raw if total_raw else math.nan,
                "per_file_recall_mean": float(np.mean(recalls)) if recalls else math.nan,
                "per_file_recall_median": float(np.median(recalls)) if recalls else math.nan,
                "files_with_raw_repedal": len(recalls),
                "files_without_raw_repedal": len(rows) - len(recalls),
                "operational_threshold_definition": True,
            }
        )
    return summary_rows, per_file_rows


def build_file_intervals(token_note_rows: Sequence[dict[str, Any]]) -> dict[str, list[tuple[float, float, str]]]:
    by_file: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
    bins = ioi_bins()
    for row in token_note_rows:
        start = float(row["onset_sec"])
        end = start + float(row["next_ioi_sec"])
        if end <= start:
            continue
        label = bins[-1][0]
        for bin_label, bin_start, bin_end in bins:
            if bin_start <= float(row["next_ioi_sec"]) < bin_end:
                label = bin_label
                break
        by_file[row["file"]].append((start, end, label))
    return by_file


def interval_label_at_time(intervals: Sequence[tuple[float, float, str]], time_sec: float) -> str | None:
    for start, end, label in intervals:
        if start <= time_sec < end:
            return label
    preceding = [item for item in intervals if item[0] <= time_sec]
    if preceding:
        return preceding[-1][2]
    return None


def aggregate_ioi_repedal(
    repedal_rows: Sequence[dict[str, Any]],
    transition_rows: Sequence[dict[str, Any]],
    token_note_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    intervals_by_file = build_file_intervals(token_note_rows)
    acc = {
        label: {
            "raw": 0,
            "preserved": 0,
            "transition_errors": [],
        }
        for label, _, _ in ioi_bins()
    }

    for row in repedal_rows:
        label = interval_label_at_time(intervals_by_file.get(row["file"], []), float(row["raw_release_time"]))
        if label is None:
            continue
        acc[label]["raw"] += 1
        if row["preserved"] is True or str(row["preserved"]).lower() == "true":
            acc[label]["preserved"] += 1

    for row in transition_rows:
        label = interval_label_at_time(intervals_by_file.get(row["file"], []), float(row["raw_time_sec"]))
        if label is None:
            continue
        acc[label]["transition_errors"].append(float(row["abs_error_ms"]))

    rows: list[dict[str, Any]] = []
    for label, start, end in ioi_bins():
        raw = acc[label]["raw"]
        preserved = acc[label]["preserved"]
        errors = acc[label]["transition_errors"]
        rows.append(
            {
                "ioi_bin": label,
                "ioi_start_sec": start,
                "ioi_end_sec": end,
                "local_ioi_definition": "next IOI of the note interval containing the raw repedal release time",
                "multi_ioi_gesture_rule": "represent by the IOI bin at raw release time",
                "raw_repedal_count": raw,
                "preserved_count": preserved,
                "lost_count": raw - preserved,
                "repedal_recall": preserved / raw if raw else math.nan,
                "mean_transition_timing_error_ms": float(np.mean(errors)) if errors else math.nan,
                "median_transition_timing_error_ms": float(np.median(errors)) if errors else math.nan,
                "transition_match_count": len(errors),
            }
        )
    return rows


def finite_values(rows: Sequence[dict[str, Any]], key: str) -> list[float]:
    out = []
    for row in rows:
        try:
            value = float(row[key])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isnan(value):
            out.append(value)
    return out


def bootstrap_mean_ci(values: Sequence[float], seed: int, n_bootstrap: int = 2000) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(float(np.mean(sample)))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def aggregate_statistics(summary_rows: Sequence[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    metrics = [
        "raw_intermediate_time_ratio",
        "token_intermediate_time_ratio",
        "cc64_nmae",
        "js_divergence",
        "wasserstein_distance",
        "repedal_recall",
        "median_transition_error_ms",
        "p90_transition_error_ms",
    ]
    rows: list[dict[str, Any]] = []
    total_raw = sum(int(float(row.get("raw_repedal_count", 0) or 0)) for row in summary_rows)
    total_preserved = sum(int(float(row.get("preserved_repedal_count", 0) or 0)) for row in summary_rows)
    files_without_repedal = sum(1 for row in summary_rows if int(float(row.get("raw_repedal_count", 0) or 0)) == 0)

    for metric in metrics:
        source_rows = summary_rows
        excluded = 0
        if metric == "repedal_recall":
            source_rows = [row for row in summary_rows if int(float(row.get("raw_repedal_count", 0) or 0)) > 0]
            excluded = files_without_repedal
        values = finite_values(source_rows, metric)
        ci_low, ci_high = bootstrap_mean_ci(values, seed)
        rows.append(
            {
                "metric": metric,
                "count": len(values),
                "excluded_files": excluded,
                "mean": float(np.mean(values)) if values else math.nan,
                "median": float(np.median(values)) if values else math.nan,
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 if len(values) == 1 else math.nan,
                "p25": float(np.percentile(values, 25)) if values else math.nan,
                "p75": float(np.percentile(values, 75)) if values else math.nan,
                "bootstrap_mean_ci95_low": ci_low,
                "bootstrap_mean_ci95_high": ci_high,
                "micro_value": total_preserved / total_raw if metric == "repedal_recall" and total_raw else math.nan,
                "macro_value": float(np.mean(values)) if metric == "repedal_recall" and values else math.nan,
                "total_raw_repedal_count": total_raw if metric == "repedal_recall" else "",
                "total_preserved_repedal_count": total_preserved if metric == "repedal_recall" else "",
                "total_lost_repedal_count": total_raw - total_preserved if metric == "repedal_recall" else "",
            }
        )
    return rows


def by_composer(summary_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        grouped[row.get("composer") or "unknown"].append(row)

    rows: list[dict[str, Any]] = []
    for composer, items in sorted(grouped.items()):
        total_raw = sum(int(float(row.get("raw_repedal_count", 0) or 0)) for row in items)
        total_preserved = sum(int(float(row.get("preserved_repedal_count", 0) or 0)) for row in items)
        recalls = [float(row["repedal_recall"]) for row in items if not math.isnan(float(row["repedal_recall"]))]
        rows.append(
            {
                "composer": composer,
                "num_performances": len(items),
                "raw_intermediate_time_ratio_mean": float(np.mean(finite_values(items, "raw_intermediate_time_ratio"))),
                "token_intermediate_time_ratio_mean": float(np.mean(finite_values(items, "token_intermediate_time_ratio"))),
                "cc64_nmae_mean": float(np.mean(finite_values(items, "cc64_nmae"))),
                "raw_repedal_count": total_raw,
                "preserved_repedal_count": total_preserved,
                "lost_repedal_count": total_raw - total_preserved,
                "micro_repedal_recall": total_preserved / total_raw if total_raw else math.nan,
                "macro_repedal_recall": float(np.mean(recalls)) if recalls else math.nan,
                "median_transition_error_ms_mean": float(np.mean(finite_values(items, "median_transition_error_ms"))),
                "small_n_warning": len(items) < 5,
            }
        )
    return rows


def make_aggregate_figures(
    output_dir: Path,
    summary_rows: Sequence[dict[str, Any]],
    ioi_rows: Sequence[dict[str, Any]],
) -> list[str]:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    raw_intermediate = finite_values(summary_rows, "raw_intermediate_time_ratio")
    token_intermediate = finite_values(summary_rows, "token_intermediate_time_ratio")
    n = min(len(raw_intermediate), len(token_intermediate))
    fig, ax = plt.subplots(figsize=(5, 5))
    if n:
        ax.scatter(raw_intermediate[:n], token_intermediate[:n], alpha=0.75)
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("raw intermediate time ratio")
    ax.set_ylabel("tokenized intermediate time ratio")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    path = output_dir / "asap_batch_raw_vs_token_intermediate_scatter.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))

    recalls = finite_values(
        [row for row in summary_rows if int(float(row.get("raw_repedal_count", 0) or 0)) > 0],
        "repedal_recall",
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(recalls, bins=np.linspace(0, 1, 11), edgecolor="black")
    ax.set_xlabel("per-file repedal recall")
    ax.set_ylabel("file count")
    fig.tight_layout()
    path = output_dir / "asap_batch_repedal_recall_distribution.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [row["ioi_bin"] for row in ioi_rows]
    values = [0 if math.isnan(float(row["repedal_recall"])) else float(row["repedal_recall"]) for row in ioi_rows]
    ax.bar(labels, values)
    ax.set_xlabel("IOI bin")
    ax.set_ylabel("repedal recall")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    path = output_dir / "asap_batch_ioi_bin_vs_repedal_recall.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(finite_values(summary_rows, "cc64_nmae"), bins=10, edgecolor="black")
    ax.set_xlabel("per-file CC64 NMAE")
    ax.set_ylabel("file count")
    fig.tight_layout()
    path = output_dir / "asap_batch_nmae_distribution.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(6, 4))
    raw_counts = [int(float(row.get("raw_repedal_count", 0) or 0)) for row in summary_rows]
    lost_counts = [int(float(row.get("lost_repedal_count", 0) or 0)) for row in summary_rows]
    ax.scatter(raw_counts, lost_counts, alpha=0.75)
    ax.set_xlabel("raw repedal count")
    ax.set_ylabel("lost repedal count")
    fig.tight_layout()
    path = output_dir / "asap_batch_raw_repedal_vs_lost_repedal.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))

    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch ASAP pedal-tokenizer analysis.")
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_script())
    parser.add_argument("--max-files", type=int, default=DEFAULT_MAX_FILES)
    parser.add_argument("--min-cc64-events", type=int, default=DEFAULT_MIN_CC64_EVENTS)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--grid-ms", type=float, default=10.0)
    parser.add_argument("--high-threshold", type=int, default=96)
    parser.add_argument("--low-threshold", type=int, default=31)
    parser.add_argument("--max-repedal-duration-ms", type=int, default=300)
    parser.add_argument("--repedal-match-tolerance-ms", type=int, default=150)
    parser.add_argument("--transition-threshold", type=int, default=64)
    parser.add_argument("--transition-match-tolerance-ms", type=int, default=500)
    parser.add_argument("--note-density-window-sec", type=float, default=5.0)
    parser.add_argument("--skip-per-file-figures", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.repo_root = args.repo_root.resolve()
    output_root = Path(__file__).resolve().parent / "outputs"
    csv_root = output_root / "csv"
    figures_root = output_root / "figures"
    csv_root.mkdir(parents=True, exist_ok=True)
    figures_root.mkdir(parents=True, exist_ok=True)

    MidiFile = import_miditoolkit()
    components = load_official_components(args.repo_root, require_official_config=True)

    inventory = discover_asap_inventory(args.repo_root, MidiFile)
    selected, threshold_used, selection_reason = stratified_select(
        inventory,
        max_files=args.max_files,
        min_cc64_events=args.min_cc64_events,
        seed=args.random_seed,
    )
    write_csv(csv_root / "asap_midi_inventory.csv", inventory)
    write_csv(csv_root / "selected_asap_performances.csv", selected)

    all_summary: list[dict[str, Any]] = []
    all_repedals: list[dict[str, Any]] = []
    all_token_notes: list[dict[str, Any]] = []
    all_events: list[dict[str, Any]] = []
    all_ioi: list[dict[str, Any]] = []
    all_density: list[dict[str, Any]] = []
    all_transitions: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []

    analysis_args = argparse.Namespace(
        repo_root=args.repo_root,
        grid_ms=args.grid_ms,
        high_threshold=args.high_threshold,
        low_threshold=args.low_threshold,
        max_repedal_duration_ms=args.max_repedal_duration_ms,
        repedal_match_tolerance_ms=args.repedal_match_tolerance_ms,
        transition_threshold=args.transition_threshold,
        transition_match_tolerance_ms=args.transition_match_tolerance_ms,
        note_density_window_sec=args.note_density_window_sec,
        skip_figures=args.skip_per_file_figures,
    )

    for candidate in selected:
        try:
            result = analyze_one_file(candidate, components, MidiFile, output_root, analysis_args)
            summary = result["summary"]
            for key in [
                "relative_path",
                "composer",
                "title",
                "folder",
                "num_intermediate_cc64_events",
                "selection_threshold_used",
                "selection_reason",
                "random_seed",
            ]:
                summary[key] = candidate.get(key, "")
            summary["num_raw_intermediate_cc64_events"] = candidate.get("num_intermediate_cc64_events", "")
            all_summary.append(summary)

            for collection_name, target in [
                ("repedal_rows", all_repedals),
                ("token_note_rows", all_token_notes),
                ("event_rows", all_events),
                ("ioi_rows", all_ioi),
                ("density_rows", all_density),
                ("transition_rows", all_transitions),
            ]:
                for row in result[collection_name]:
                    row["relative_path"] = candidate.get("relative_path", "")
                    row["composer"] = candidate.get("composer", "")
                    row["title"] = candidate.get("title", "")
                    row["folder"] = candidate.get("folder", "")
                    target.append(row)
        except Exception as exc:
            error_rows.append(
                {
                    "file": candidate.get("file", ""),
                    "relative_path": candidate.get("relative_path", ""),
                    "composer": candidate.get("composer", ""),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            continue

    write_csv(csv_root / "summary_per_file.csv", all_summary)
    write_csv(csv_root / "repedal_events.csv", all_repedals)
    write_csv(csv_root / "tokenized_notes.csv", all_token_notes)
    write_csv(csv_root / "cc64_events.csv", all_events)
    write_csv(csv_root / "ioi_bins.csv", all_ioi)
    write_csv(csv_root / "note_density_bins.csv", all_density)
    write_csv(csv_root / "transition_matches.csv", all_transitions)
    write_csv(
        csv_root / "error_log.csv",
        error_rows,
        fieldnames=["file", "relative_path", "composer", "error_type", "error_message", "traceback"],
    )

    sensitivity_rows: list[dict[str, Any]] = []
    sensitivity_per_file_rows: list[dict[str, Any]] = []
    successful_files = {row["file"] for row in all_summary}
    successful_selected = [candidate for candidate in selected if candidate.get("file") in successful_files]
    if successful_selected:
        sensitivity_rows, sensitivity_per_file_rows = threshold_sensitivity(
            successful_selected,
            components,
            MidiFile,
            args.repedal_match_tolerance_ms,
        )
    write_csv(csv_root / "repedal_threshold_sensitivity.csv", sensitivity_rows)
    write_csv(csv_root / "repedal_threshold_sensitivity_per_file.csv", sensitivity_per_file_rows)

    ioi_repedal_rows = aggregate_ioi_repedal(all_repedals, all_transitions, all_token_notes)
    write_csv(csv_root / "ioi_repedal_analysis.csv", ioi_repedal_rows)

    aggregate_rows = aggregate_statistics(all_summary, args.random_seed)
    write_csv(csv_root / "aggregate_statistics.csv", aggregate_rows)

    composer_rows = by_composer(all_summary) if all_summary else []
    write_csv(csv_root / "by_composer.csv", composer_rows)

    aggregate_figure_paths = make_aggregate_figures(figures_root, all_summary, ioi_repedal_rows) if all_summary else []

    write_json(
        csv_root / "asap_batch_run_metadata.json",
        {
            "repo_root": str(args.repo_root),
            "tokenizer_file": str(components["pt_root"] / "src" / "utils" / "midi.py"),
            "tokenizer_functions": ["normalize_midi", "midi_to_ids", "ids_to_midi"],
            "config_source": components["config_source"],
            "model_training_or_inference": False,
            "coordinate_system": (
                "Official normalize_midi coordinate: 120 BPM, 500 ticks/beat, "
                f"{TARGET_TICKS_PER_SECOND:.0f} ticks/sec."
            ),
            "asap_inventory_count": len(inventory),
            "selected_count": len(selected),
            "successful_count": len(all_summary),
            "failed_count": len(error_rows),
            "selection": {
                "requested_max_files": args.max_files,
                "requested_min_cc64_events": args.min_cc64_events,
                "threshold_used": threshold_used,
                "reason": selection_reason,
                "random_seed": args.random_seed,
                "deterministic_stratified_by_composer": True,
            },
            "repedal_definition_default": {
                "high_threshold": args.high_threshold,
                "low_threshold": args.low_threshold,
                "max_repedal_duration_ms": args.max_repedal_duration_ms,
                "operational_threshold_definition_only": True,
            },
            "ioi_repedal_local_context": {
                "local_ioi_definition": "next IOI of the note interval containing the raw repedal release time",
                "multi_ioi_gesture_rule": "represent by the IOI bin at raw release time",
            },
            "validation_notes": [
                "ASAP midi_score paths and filenames containing score are excluded from selection.",
                "Only CC64 controller events are treated as sustain pedal.",
                "Raw and reconstructed curves are compared in official normalized 120 BPM time.",
                "Reconstruction uses tokenized Pedal1-4 values only; no interpolation is used to invent raw pedal values.",
                "Per-file exceptions are recorded to error_log.csv and do not stop the batch.",
            ],
            "aggregate_figure_paths": aggregate_figure_paths,
        },
    )

    print("ASAP batch analysis complete.")
    print(f"Inventory performances: {len(inventory)}")
    print(f"Selected performances: {len(selected)}")
    print(f"Successful analyses: {len(all_summary)}")
    print(f"Failed analyses: {len(error_rows)}")
    print(f"Wrote CSV outputs under: {csv_root}")
    print(f"Wrote figures under: {figures_root}")
    return 0 if not error_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
