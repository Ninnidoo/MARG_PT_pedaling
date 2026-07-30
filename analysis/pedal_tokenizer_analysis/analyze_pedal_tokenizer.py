from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from pedal_utils import (
    TARGET_TICKS_PER_SECOND,
    TokenizerOnlyConfig,
    bin_label_for_value,
    detect_repedals,
    detect_threshold_transitions,
    extract_normalized_cc64_events,
    extract_normalized_notes,
    extract_original_cc64_events,
    histogram_128,
    ioi_bins,
    js_divergence,
    match_repedals,
    match_transitions,
    midi_duration_seconds,
    note_density_at_times,
    occupancy_ratios,
    sample_zero_order_hold,
    summarize_errors_ms,
    ticks_to_seconds,
    token_rows_from_ids,
    wasserstein_distance_1d,
)


SUMMARY_COLUMNS = [
    "file",
    "candidate_source",
    "duration_sec",
    "num_notes",
    "num_raw_cc64_events",
    "raw_intermediate_time_ratio",
    "token_intermediate_time_ratio",
    "cc64_mae",
    "cc64_nmae",
    "cc64_rmse",
    "js_divergence",
    "wasserstein_distance",
    "raw_repedal_count",
    "preserved_repedal_count",
    "lost_repedal_count",
    "repedal_recall",
    "repedal_precision",
    "median_transition_error_ms",
    "p90_transition_error_ms",
]

REPEDAL_COLUMNS = [
    "file",
    "candidate_source",
    "raw_release_time",
    "raw_redepress_time",
    "duration_ms",
    "minimum_cc64",
    "preserved",
    "token_release_time",
    "token_redepress_time",
    "release_timing_error_ms",
    "redepress_timing_error_ms",
]


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def add_third_party_to_path(repo_root: Path) -> Path:
    pt_root = repo_root / "third_party" / "PianistTransformer"
    if not pt_root.exists():
        raise FileNotFoundError(f"Missing Pianist Transformer repository: {pt_root}")
    sys.path.insert(0, str(pt_root))
    return pt_root


def load_official_components(repo_root: Path, require_official_config: bool = False) -> dict[str, Any]:
    pt_root = add_third_party_to_path(repo_root)

    try:
        midi_module = importlib.import_module("src.utils.midi")
    except Exception as exc:
        raise RuntimeError(
            "Could not import official tokenizer module "
            "third_party/PianistTransformer/src/utils/midi.py. "
            "The usual missing dependencies here are miditoolkit and numpy."
        ) from exc

    config_source = "TokenizerOnlyConfig fallback"
    try:
        pianoformer = importlib.import_module("src.model.pianoformer")
        config = pianoformer.PianoT5GemmaConfig()
        config_source = "official PianoT5GemmaConfig"
    except Exception as exc:
        if require_official_config:
            raise RuntimeError(
                "Could not import official PianoT5GemmaConfig. "
                "Install the repository's model dependencies or rerun without "
                "--require-official-config to use the tokenizer-only constants."
            ) from exc
        config = TokenizerOnlyConfig()
        config_source = f"TokenizerOnlyConfig fallback after {type(exc).__name__}: {exc}"

    return {
        "pt_root": pt_root,
        "config": config,
        "config_source": config_source,
        "midi_to_ids": midi_module.midi_to_ids,
        "ids_to_midi": midi_module.ids_to_midi,
        "normalize_midi": midi_module.normalize_midi,
    }


def import_miditoolkit():
    try:
        return importlib.import_module("miditoolkit").MidiFile
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency: miditoolkit. Do not install automatically; "
            "activate the project environment or install the repository "
            "requirements in an isolated environment."
        ) from exc


def safe_stem(path: Path) -> str:
    rel = str(path).replace("\\", "/")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", rel).strip("_")


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        seen: list[str] = []
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.append(key)
        fieldnames = seen
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def original_duration_seconds(midi_obj) -> float:
    max_tick = int(getattr(midi_obj, "max_tick", 0) or 0)
    for instrument in midi_obj.instruments:
        for note in instrument.notes:
            max_tick = max(max_tick, int(note.end))
        for cc in instrument.control_changes:
            max_tick = max(max_tick, int(cc.time))
    mapping = midi_obj.get_tick_to_time_mapping()
    if len(mapping) == 0:
        return 0.0
    return float(mapping[min(max_tick, len(mapping) - 1)])


def iter_asap_performance_paths(repo_root: Path) -> Iterable[tuple[Path, str]]:
    asap_root = repo_root / "third_party" / "PianistTransformer" / "data" / "midis" / "asap-dataset-master"
    metadata_path = asap_root / "metadata.csv"
    if not metadata_path.exists():
        return
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rel = row.get("midi_performance", "")
            if not rel:
                continue
            path = asap_root / rel
            if path.exists():
                yield path, "asap_performance_metadata"


def iter_testset_performance_paths(repo_root: Path) -> Iterable[tuple[Path, str]]:
    base = repo_root / "third_party" / "PianistTransformer" / "data" / "midis" / "testset" / "performance"
    if not base.exists():
        return
    for path in sorted(base.glob("*.mid")):
        yield path, "pianist_transformer_testset_performance"


def iter_testset_human_paths(repo_root: Path) -> Iterable[tuple[Path, str]]:
    base = repo_root / "third_party" / "PianistTransformer" / "data" / "midis" / "testset" / "human"
    if not base.exists():
        return
    for path in sorted(base.glob("*.mid")):
        yield path, "pianist_transformer_testset_human"


def iter_baseline_output_paths(repo_root: Path) -> Iterable[tuple[Path, str]]:
    base = repo_root / "outputs" / "midi" / "baseline"
    if not base.exists():
        return
    for path in sorted(base.rglob("*.mid")):
        yield path, "local_baseline_output"


def scan_candidate(path: Path, source: str, MidiFile) -> dict[str, Any]:
    midi_obj = MidiFile(str(path))
    events = extract_original_cc64_events(midi_obj)
    note_count = sum(len(inst.notes) for inst in midi_obj.instruments if not inst.is_drum)
    values = [event.cc64_value for event in events]
    intermediate_count = sum(1 for value in values if 0 < value < 127)
    return {
        "file": str(path),
        "candidate_source": source,
        "duration_sec": original_duration_seconds(midi_obj),
        "num_notes": note_count,
        "num_cc64_events": len(events),
        "num_intermediate_cc64_events": intermediate_count,
        "min_cc64": min(values) if values else math.nan,
        "max_cc64": max(values) if values else math.nan,
        "tracks": len(midi_obj.instruments),
        "ticks_per_beat": midi_obj.ticks_per_beat,
        "tempo_changes": len(midi_obj.tempo_changes),
        "selected": False,
    }


def collect_candidates(repo_root: Path, MidiFile, include_human: bool) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    path_sources: list[tuple[Path, str]] = []
    path_sources.extend(iter_asap_performance_paths(repo_root))
    path_sources.extend(iter_testset_performance_paths(repo_root))
    if include_human:
        path_sources.extend(iter_testset_human_paths(repo_root))
    path_sources.extend(iter_baseline_output_paths(repo_root))

    seen: set[Path] = set()
    for path, source in path_sources:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            candidates.append(scan_candidate(resolved, source, MidiFile))
        except Exception as exc:
            candidates.append(
                {
                    "file": str(resolved),
                    "candidate_source": source,
                    "scan_error": f"{type(exc).__name__}: {exc}",
                    "selected": False,
                }
            )
    return candidates


def select_candidates(candidates: list[dict[str, Any]], max_files: int, asap_only: bool) -> list[dict[str, Any]]:
    valid = [c for c in candidates if not c.get("scan_error") and int(c.get("num_cc64_events", 0)) > 0]
    asap = [c for c in valid if c["candidate_source"] == "asap_performance_metadata"]
    asap.sort(key=lambda c: int(c.get("num_intermediate_cc64_events", 0)), reverse=True)
    selected = asap[:max_files]

    if not asap_only and len(selected) < max_files:
        fallback = [
            c
            for c in valid
            if c["candidate_source"] == "pianist_transformer_testset_performance" and c not in selected
        ]
        fallback.sort(key=lambda c: int(c.get("num_cc64_events", 0)), reverse=True)
        selected.extend(fallback[: max_files - len(selected)])

    for candidate in candidates:
        candidate["selected"] = any(candidate["file"] == item["file"] for item in selected)
    return selected


def aggregate_ioi_stats(
    token_rows: Sequence[dict[str, Any]],
    grid_sec: np.ndarray,
    raw_curve: np.ndarray,
    token_curve: np.ndarray,
    repedal_rows: Sequence[dict[str, Any]],
    transition_matches: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    bins = ioi_bins()
    acc = {
        label: {
            "sample_count": 0,
            "abs_error_sum": 0.0,
            "raw_intermediate": 0,
            "token_intermediate": 0,
            "raw_repedal_count": 0,
            "preserved_repedal_count": 0,
            "transition_errors": [],
        }
        for label, _, _ in bins
    }

    intervals: list[tuple[float, float, str]] = []
    for row in token_rows:
        start = float(row["onset_sec"])
        end = start + float(row["next_ioi_sec"])
        if end <= start:
            continue
        label = bin_label_for_value(float(row["next_ioi_sec"]), bins)
        intervals.append((start, end, label))
        left = int(np.searchsorted(grid_sec, start, side="left"))
        right = int(np.searchsorted(grid_sec, end, side="left"))
        if right <= left:
            continue
        raw_slice = raw_curve[left:right]
        token_slice = token_curve[left:right]
        acc[label]["sample_count"] += int(len(raw_slice))
        acc[label]["abs_error_sum"] += float(np.sum(np.abs(raw_slice - token_slice)))
        acc[label]["raw_intermediate"] += int(np.sum((raw_slice > 0) & (raw_slice < 127)))
        acc[label]["token_intermediate"] += int(np.sum((token_slice > 0) & (token_slice < 127)))

    def label_at_time(time_sec: float) -> str | None:
        for start, end, label in intervals:
            if start <= time_sec < end:
                return label
        return None

    for row in repedal_rows:
        label = label_at_time(float(row["raw_release_time"]))
        if label is None:
            continue
        acc[label]["raw_repedal_count"] += 1
        if str(row["preserved"]).lower() == "true" or row["preserved"] is True:
            acc[label]["preserved_repedal_count"] += 1

    for match in transition_matches:
        label = label_at_time(float(match["raw_time_sec"]))
        if label is not None:
            acc[label]["transition_errors"].append(float(match["abs_error_ms"]))

    rows: list[dict[str, Any]] = []
    for label, _, _ in bins:
        item = acc[label]
        sample_count = int(item["sample_count"])
        raw_repedals = int(item["raw_repedal_count"])
        errors = item["transition_errors"]
        rows.append(
            {
                "ioi_bin": label,
                "sample_count": sample_count,
                "pedal_nmae": (item["abs_error_sum"] / sample_count / 127.0) if sample_count else math.nan,
                "raw_intermediate_time_ratio": item["raw_intermediate"] / sample_count if sample_count else math.nan,
                "token_intermediate_time_ratio": item["token_intermediate"] / sample_count if sample_count else math.nan,
                "raw_repedal_count": raw_repedals,
                "preserved_repedal_count": int(item["preserved_repedal_count"]),
                "repedal_recall": item["preserved_repedal_count"] / raw_repedals if raw_repedals else math.nan,
                "median_transition_error_ms": float(np.median(errors)) if errors else math.nan,
            }
        )
    return rows


def aggregate_density_stats(
    note_rows: Sequence[dict[str, Any]],
    grid_sec: np.ndarray,
    raw_curve: np.ndarray,
    token_curve: np.ndarray,
    repedal_rows: Sequence[dict[str, Any]],
    transition_matches: Sequence[dict[str, Any]],
    window_sec: float,
) -> list[dict[str, Any]]:
    if len(grid_sec) == 0:
        return []
    onsets = np.asarray([float(row["onset_sec"]) for row in note_rows], dtype=float)
    density = note_density_at_times(onsets, grid_sec, window_sec)
    if len(density) == 0:
        return []
    q1, q2 = np.quantile(density, [1 / 3, 2 / 3])
    if q1 == q2:
        q1 = float(np.min(density))
        q2 = float(np.max(density))

    groups = {
        "low": density <= q1,
        "middle": (density > q1) & (density <= q2),
        "high": density > q2,
    }
    rows: list[dict[str, Any]] = []

    def group_at_time(time_sec: float) -> str:
        index = int(np.searchsorted(grid_sec, time_sec, side="right") - 1)
        index = max(0, min(index, len(grid_sec) - 1))
        value = density[index]
        if value <= q1:
            return "low"
        if value <= q2:
            return "middle"
        return "high"

    for group, mask in groups.items():
        sample_count = int(np.sum(mask))
        raw_repedals = [row for row in repedal_rows if group_at_time(float(row["raw_release_time"])) == group]
        preserved = [row for row in raw_repedals if row["preserved"] is True]
        transition_errors = [
            float(row["abs_error_ms"])
            for row in transition_matches
            if group_at_time(float(row["raw_time_sec"])) == group
        ]
        rows.append(
            {
                "density_group": group,
                "window_sec": window_sec,
                "density_min": float(np.min(density[mask])) if sample_count else math.nan,
                "density_max": float(np.max(density[mask])) if sample_count else math.nan,
                "sample_count": sample_count,
                "pedal_nmae": float(np.mean(np.abs(raw_curve[mask] - token_curve[mask])) / 127.0)
                if sample_count
                else math.nan,
                "raw_repedal_count": len(raw_repedals),
                "preserved_repedal_count": len(preserved),
                "repedal_recall": len(preserved) / len(raw_repedals) if raw_repedals else math.nan,
                "median_transition_error_ms": float(np.median(transition_errors)) if transition_errors else math.nan,
            }
        )
    return rows


def make_figures(
    output_dir: Path,
    file_stem: str,
    grid_sec: np.ndarray,
    raw_curve: np.ndarray,
    token_curve: np.ndarray,
    token_rows: Sequence[dict[str, Any]],
    raw_hist: np.ndarray,
    token_hist: np.ndarray,
    raw_delta: np.ndarray,
    token_delta: np.ndarray,
    ioi_rows: Sequence[dict[str, Any]],
    repedal_rows: Sequence[dict[str, Any]],
) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency: matplotlib. Rerun with --skip-figures or install matplotlib "
            "in the isolated project environment."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    lost = [row for row in repedal_rows if row["preserved"] is False]
    if lost:
        center = float(lost[0]["raw_release_time"])
        start = max(0.0, center - 5.0)
        end = min(float(grid_sec[-1]), start + 15.0)
    else:
        nonzero = np.where(raw_curve > 0)[0]
        start = float(grid_sec[nonzero[0]]) if len(nonzero) else 0.0
        end = min(float(grid_sec[-1]), start + 10.0)
    mask = (grid_sec >= start) & (grid_sec <= end)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.step(grid_sec[mask], raw_curve[mask], where="post", label="raw")
    ax.step(grid_sec[mask], token_curve[mask], where="post", label="tokenized/reconstructed")
    sample_times: list[float] = []
    sample_values: list[float] = []
    for row in token_rows:
        for idx in range(1, 5):
            t = float(row[f"pedal{idx}_sample_time_sec"])
            if start <= t <= end:
                sample_times.append(t)
                sample_values.append(float(row[f"pedal{idx}"]))
    if sample_times:
        ax.scatter(sample_times, sample_values, s=12, c="black", alpha=0.55, label="Pedal1-4 samples")
    ax.set_xlabel("time (seconds)")
    ax.set_ylabel("CC64")
    ax.set_ylim(-5, 132)
    ax.legend(loc="best")
    fig.tight_layout()
    path = output_dir / f"{file_stem}_figure1_raw_vs_tokenized.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(128)
    ax.plot(x, raw_hist / max(1, np.sum(raw_hist)), label="raw")
    ax.plot(x, token_hist / max(1, np.sum(token_hist)), label="tokenized/reconstructed")
    ax.set_xlabel("CC64 value")
    ax.set_ylabel("time-weighted probability")
    ax.legend(loc="best")
    fig.tight_layout()
    path = output_dir / f"{file_stem}_figure2_cc64_histogram.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(10, 4))
    bins = np.arange(-128, 130, 2)
    ax.hist(raw_delta, bins=bins, alpha=0.55, label="raw", density=True)
    ax.hist(token_delta, bins=bins, alpha=0.55, label="tokenized/reconstructed", density=True)
    ax.set_xlabel("Delta CC64 per grid step")
    ax.set_ylabel("density")
    ax.legend(loc="best")
    fig.tight_layout()
    path = output_dir / f"{file_stem}_figure3_delta_distribution.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))

    labels = [row["ioi_bin"] for row in ioi_rows]
    nmae = [row["pedal_nmae"] for row in ioi_rows]
    repedal_recall = [row["repedal_recall"] for row in ioi_rows]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(labels, [0 if math.isnan(float(v)) else v for v in nmae])
    ax.set_xlabel("IOI bin")
    ax.set_ylabel("NMAE")
    fig.tight_layout()
    path = output_dir / f"{file_stem}_figure4_ioi_nmae.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(labels, [0 if math.isnan(float(v)) else v for v in repedal_recall])
    ax.set_xlabel("IOI bin")
    ax.set_ylabel("repedal recall")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    path = output_dir / f"{file_stem}_figure5_ioi_repedal_recall.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))

    if lost:
        selected_lost = lost[:3]
        fig, axes = plt.subplots(len(selected_lost), 1, figsize=(12, 3.2 * len(selected_lost)), sharey=True)
        if len(selected_lost) == 1:
            axes = [axes]
        for ax, row in zip(axes, selected_lost):
            release = float(row["raw_release_time"])
            redepress = float(row["raw_redepress_time"])
            start = max(0.0, release - 0.4)
            end = min(float(grid_sec[-1]), redepress + 0.4)
            mask = (grid_sec >= start) & (grid_sec <= end)
            ax.step(grid_sec[mask], raw_curve[mask], where="post", label="raw")
            ax.step(grid_sec[mask], token_curve[mask], where="post", label="tokenized/reconstructed")
            ax.axvline(release, color="black", linestyle="--", linewidth=1)
            ax.axvline(redepress, color="black", linestyle=":", linewidth=1)
            ax.set_ylabel("CC64")
            ax.set_ylim(-5, 132)
        axes[-1].set_xlabel("time (seconds)")
        axes[0].legend(loc="best")
        fig.tight_layout()
        path = output_dir / f"{file_stem}_figure6_lost_repedal_examples.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path))

    return paths


def analyze_one_file(
    candidate: dict[str, Any],
    components: dict[str, Any],
    MidiFile,
    output_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    path = Path(candidate["file"])
    midi_obj = MidiFile(str(path))
    normalize_midi = components["normalize_midi"]
    midi_to_ids = components["midi_to_ids"]
    ids_to_midi = components["ids_to_midi"]
    config = components["config"]

    normalized_raw_midi = normalize_midi(midi_obj)
    ids = midi_to_ids(config, midi_obj, normalize=True)
    reconstructed_midi = ids_to_midi(config, ids)

    raw_events = extract_normalized_cc64_events(normalized_raw_midi)
    token_events = extract_normalized_cc64_events(reconstructed_midi)
    normalized_notes = extract_normalized_notes(normalized_raw_midi)
    token_rows = token_rows_from_ids(config, ids, normalized_raw_midi)

    duration_sec = max(midi_duration_seconds(normalized_raw_midi), midi_duration_seconds(reconstructed_midi))
    grid_sec = np.arange(0.0, duration_sec + args.grid_ms / 1000.0, args.grid_ms / 1000.0)
    raw_curve = sample_zero_order_hold(raw_events, grid_sec)
    token_curve = sample_zero_order_hold(token_events, grid_sec)
    diff = raw_curve - token_curve

    raw_hist = histogram_128(raw_curve)
    token_hist = histogram_128(token_curve)
    raw_delta = np.diff(raw_curve)
    token_delta = np.diff(token_curve)
    raw_occ = occupancy_ratios(raw_curve)
    token_occ = occupancy_ratios(token_curve)

    raw_repedals = detect_repedals(
        raw_events,
        high_threshold=args.high_threshold,
        low_threshold=args.low_threshold,
        max_duration_ms=args.max_repedal_duration_ms,
    )
    token_repedals = detect_repedals(
        token_events,
        high_threshold=args.high_threshold,
        low_threshold=args.low_threshold,
        max_duration_ms=args.max_repedal_duration_ms,
    )
    repedal_rows, matched_token_repedals = match_repedals(
        raw_repedals,
        token_repedals,
        tolerance_ms=args.repedal_match_tolerance_ms,
    )
    for row in repedal_rows:
        row["file"] = str(path)
        row["candidate_source"] = candidate["candidate_source"]

    raw_transitions = detect_threshold_transitions(raw_events, threshold=args.transition_threshold)
    token_transitions = detect_threshold_transitions(token_events, threshold=args.transition_threshold)
    transition_matches = match_transitions(
        raw_transitions,
        token_transitions,
        tolerance_ms=args.transition_match_tolerance_ms,
    )
    transition_summary = summarize_errors_ms([float(row["abs_error_ms"]) for row in transition_matches])

    ioi_rows = aggregate_ioi_stats(token_rows, grid_sec, raw_curve, token_curve, repedal_rows, transition_matches)
    for row in ioi_rows:
        row["file"] = str(path)
        row["candidate_source"] = candidate["candidate_source"]

    density_rows = aggregate_density_stats(
        token_rows,
        grid_sec,
        raw_curve,
        token_curve,
        repedal_rows,
        transition_matches,
        window_sec=args.note_density_window_sec,
    )
    for row in density_rows:
        row["file"] = str(path)
        row["candidate_source"] = candidate["candidate_source"]

    event_rows: list[dict[str, Any]] = []
    for representation, events in [("raw_normalized", raw_events), ("tokenized_reconstructed", token_events)]:
        for event in events:
            event_rows.append(
                {
                    "file": str(path),
                    "candidate_source": candidate["candidate_source"],
                    "representation": representation,
                    "time_sec": event.time_sec,
                    "cc64_value": event.cc64_value,
                    "tick": event.tick,
                }
            )

    token_note_rows: list[dict[str, Any]] = []
    for row in token_rows:
        out = {"file": str(path), "candidate_source": candidate["candidate_source"]}
        out.update(row)
        token_note_rows.append(out)

    figure_paths: list[str] = []
    if not args.skip_figures:
        figure_paths = make_figures(
            output_root / "figures",
            safe_stem(path.relative_to(args.repo_root) if path.is_relative_to(args.repo_root) else path),
            grid_sec,
            raw_curve,
            token_curve,
            token_rows,
            raw_hist,
            token_hist,
            raw_delta,
            token_delta,
            ioi_rows,
            repedal_rows,
        )

    raw_repedal_count = len(raw_repedals)
    preserved_repedal_count = sum(1 for row in repedal_rows if row["preserved"] is True)
    lost_repedal_count = raw_repedal_count - preserved_repedal_count
    repedal_precision = matched_token_repedals / len(token_repedals) if token_repedals else math.nan
    summary = {
        "file": str(path),
        "candidate_source": candidate["candidate_source"],
        "duration_sec": duration_sec,
        "num_notes": len(normalized_notes),
        "num_tokens": len(ids),
        "num_raw_cc64_events": len(raw_events),
        "num_token_cc64_events": len(token_events),
        "raw_intermediate_time_ratio": raw_occ["intermediate_time_ratio"],
        "token_intermediate_time_ratio": token_occ["intermediate_time_ratio"],
        "raw_released_exact_ratio": raw_occ["released_exact_ratio"],
        "token_released_exact_ratio": token_occ["released_exact_ratio"],
        "raw_full_exact_ratio": raw_occ["full_exact_ratio"],
        "token_full_exact_ratio": token_occ["full_exact_ratio"],
        "raw_partial_16_111_ratio": raw_occ["partial_16_111_ratio"],
        "token_partial_16_111_ratio": token_occ["partial_16_111_ratio"],
        "cc64_mae": float(np.mean(np.abs(diff))) if len(diff) else math.nan,
        "cc64_nmae": float(np.mean(np.abs(diff)) / 127.0) if len(diff) else math.nan,
        "cc64_rmse": float(np.sqrt(np.mean(diff**2))) if len(diff) else math.nan,
        "js_divergence": js_divergence(raw_hist, token_hist),
        "wasserstein_distance": wasserstein_distance_1d(raw_hist, token_hist),
        "raw_delta_mean_abs": float(np.mean(np.abs(raw_delta))) if len(raw_delta) else math.nan,
        "token_delta_mean_abs": float(np.mean(np.abs(token_delta))) if len(token_delta) else math.nan,
        "raw_delta_median_abs": float(np.median(np.abs(raw_delta))) if len(raw_delta) else math.nan,
        "token_delta_median_abs": float(np.median(np.abs(token_delta))) if len(token_delta) else math.nan,
        "raw_repedal_count": raw_repedal_count,
        "token_repedal_count": len(token_repedals),
        "preserved_repedal_count": preserved_repedal_count,
        "lost_repedal_count": lost_repedal_count,
        "repedal_recall": preserved_repedal_count / raw_repedal_count if raw_repedal_count else math.nan,
        "repedal_precision": repedal_precision,
        "raw_transition_count": len(raw_transitions),
        "token_transition_count": len(token_transitions),
        "matched_transition_count": len(transition_matches),
        "figure_paths": ";".join(figure_paths),
    }
    summary.update(transition_summary)

    return {
        "summary": summary,
        "repedal_rows": repedal_rows,
        "token_note_rows": token_note_rows,
        "event_rows": event_rows,
        "ioi_rows": ioi_rows,
        "density_rows": density_rows,
        "transition_rows": [
            {
                "file": str(path),
                "candidate_source": candidate["candidate_source"],
                **row,
            }
            for row in transition_matches
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze CC64 information loss through the official Pianist Transformer tokenizer."
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_script())
    parser.add_argument("--midi", type=Path, nargs="*", help="Explicit MIDI files to analyze.")
    parser.add_argument("--max-files", type=int, default=10)
    parser.add_argument("--pilot", action="store_true", help="Analyze only the first selected file.")
    parser.add_argument("--asap-only", action="store_true", help="Do not fill with testset/performance fallback files.")
    parser.add_argument("--include-human", action="store_true", help="Also scan testset/human MIDI files as candidates.")
    parser.add_argument("--grid-ms", type=float, default=10.0)
    parser.add_argument("--high-threshold", type=int, default=96)
    parser.add_argument("--low-threshold", type=int, default=31)
    parser.add_argument("--max-repedal-duration-ms", type=int, default=300)
    parser.add_argument("--repedal-match-tolerance-ms", type=int, default=150)
    parser.add_argument("--transition-threshold", type=int, default=64)
    parser.add_argument("--transition-match-tolerance-ms", type=int, default=500)
    parser.add_argument("--note-density-window-sec", type=float, default=5.0)
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--require-official-config", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.repo_root = args.repo_root.resolve()
    output_root = Path(__file__).resolve().parent / "outputs"
    csv_root = output_root / "csv"

    MidiFile = import_miditoolkit()
    components = load_official_components(args.repo_root, require_official_config=args.require_official_config)

    if args.midi:
        candidates = [
            scan_candidate(path.resolve(), "explicit_user_path", MidiFile)
            for path in args.midi
        ]
        selected = candidates[: 1 if args.pilot else args.max_files]
        for candidate in candidates:
            candidate["selected"] = any(candidate["file"] == item["file"] for item in selected)
    else:
        candidates = collect_candidates(args.repo_root, MidiFile, include_human=args.include_human)
        selected = select_candidates(candidates, args.max_files, asap_only=args.asap_only)
        if args.pilot:
            selected = selected[:1]
            for candidate in candidates:
                candidate["selected"] = any(candidate["file"] == item["file"] for item in selected)

    write_csv(csv_root / "midi_candidates.csv", candidates)

    if not selected:
        print("No analyzable MIDI candidates with CC64 events were found.")
        print(f"Wrote candidate scan: {csv_root / 'midi_candidates.csv'}")
        return 2

    all_summary: list[dict[str, Any]] = []
    all_repedals: list[dict[str, Any]] = []
    all_token_notes: list[dict[str, Any]] = []
    all_events: list[dict[str, Any]] = []
    all_ioi: list[dict[str, Any]] = []
    all_density: list[dict[str, Any]] = []
    all_transitions: list[dict[str, Any]] = []

    for candidate in selected:
        result = analyze_one_file(candidate, components, MidiFile, output_root, args)
        all_summary.append(result["summary"])
        all_repedals.extend(result["repedal_rows"])
        all_token_notes.extend(result["token_note_rows"])
        all_events.extend(result["event_rows"])
        all_ioi.extend(result["ioi_rows"])
        all_density.extend(result["density_rows"])
        all_transitions.extend(result["transition_rows"])

    write_csv(csv_root / "summary_per_file.csv", all_summary, fieldnames=None)
    write_csv(csv_root / "repedal_events.csv", all_repedals, fieldnames=REPEDAL_COLUMNS)
    write_csv(csv_root / "tokenized_notes.csv", all_token_notes)
    write_csv(csv_root / "cc64_events.csv", all_events)
    write_csv(csv_root / "ioi_bins.csv", all_ioi)
    write_csv(csv_root / "note_density_bins.csv", all_density)
    write_csv(csv_root / "transition_matches.csv", all_transitions)

    write_json(
        csv_root / "analysis_run_metadata.json",
        {
            "repo_root": str(args.repo_root),
            "pianist_transformer_root": str(components["pt_root"]),
            "tokenizer_file": str(components["pt_root"] / "src" / "utils" / "midi.py"),
            "tokenizer_function": "midi_to_ids",
            "decoder_function": "ids_to_midi",
            "config_source": components["config_source"],
            "grid_ms": args.grid_ms,
            "repedal_definition": {
                "high_threshold": args.high_threshold,
                "low_threshold": args.low_threshold,
                "max_repedal_duration_ms": args.max_repedal_duration_ms,
                "match_tolerance_ms": args.repedal_match_tolerance_ms,
                "operational_definition_only": True,
            },
            "transition_threshold": args.transition_threshold,
            "selected_files": [candidate["file"] for candidate in selected],
            "model_training_or_inference": False,
            "coordinate_system": (
                "Official normalize_midi coordinate: 120 BPM, 500 ticks/beat, "
                f"{TARGET_TICKS_PER_SECOND:.0f} ticks/sec."
            ),
        },
    )

    print("Analysis complete.")
    print(f"Config source: {components['config_source']}")
    print(f"Selected files: {len(selected)}")
    print(f"Wrote: {csv_root / 'summary_per_file.csv'}")
    print(f"Wrote: {csv_root / 'repedal_events.csv'}")
    print(f"Wrote figures under: {output_root / 'figures'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
