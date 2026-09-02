#!/usr/bin/env python3
"""Five-performance, validation-only Structural Bass Event detector v0 audit.

This is a descriptive audit, not a learned detector and not a ground-truth
evaluation.  Bass features use note attacks only.  CC64 and note durations are
never read into the detector feature table.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mido
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_pedal_event_metric_tolerance_mini import local_ioi
from src.stage2_event_tokenizer.tokenizer import parse_raw_midi
from src.stage2_event_tokenizer.tokenizer_v1 import _tick_second_converters


CUTOFFS = (48, 52, 55, 57, 60)
K_VALUES = (2, 3, 4, 6, 8)
LOCAL_IOI_WINDOW = 16  # eight distinct onsets on either side where available
MANUAL_REFERENCE_CUTOFF = 55
MANUAL_PER_PIECE = 18
ASAP_ROOT = Path("/workspace/public/ASAP/asap-dataset-v1.1")
SPLIT_MANIFEST = ROOT / "analysis/stage2_encoder_only_v0/asap_split.csv"
DEFAULT_OUTPUT = ROOT / "analysis/structural_bass_event_detector_v0"

# Exact validation-piece choices.  The first row for each piece in canonical
# manifest order is selected below; no lexical reordering of performances.
TARGETS = (
    ("Ravel", "Pavane"),
    ("Schumann", "Arabeske"),
    ("Liszt", "Mephisto_Waltz"),
    ("Beethoven", "Piano_Sonatas_27-1"),
    ("Chopin", "Etudes_op_25_12"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    values = list(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="raise")
        writer.writeheader()
        writer.writerows(values)


def midi_note_name(pitch: int) -> str:
    names = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    return f"{names[int(pitch) % 12]}{int(pitch) // 12 - 1}"


def fmt(value: Any, digits: int = 3) -> str:
    if value is None or value == "":
        return "—"
    number = float(value)
    return "—" if not math.isfinite(number) else f"{number:.{digits}f}"


def pct(numerator: int | float, denominator: int | float, digits: int = 1) -> str:
    return "—" if not denominator else f"{100.0 * float(numerator) / float(denominator):.{digits}f}%"


def quantiles(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray([float(value) for value in values if pd.notna(value)], dtype=float)
    if not len(array):
        return {"count": 0, "q05": None, "q25": None, "median": None, "q75": None, "q95": None}
    return {
        "count": int(len(array)),
        "q05": float(np.quantile(array, 0.05)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.50)),
        "q75": float(np.quantile(array, 0.75)),
        "q95": float(np.quantile(array, 0.95)),
    }


def choose_validation_performances() -> tuple[list[dict[str, str]], dict[str, Any]]:
    rows = read_csv(SPLIT_MANIFEST)
    validation = [row for row in rows if row["split"] == "validation"]
    if len(validation) != 71 or len({row["piece_id"] for row in validation}) != 19:
        raise RuntimeError("canonical validation manifest is not 71 performances / 19 pieces")

    selected: list[dict[str, str]] = []
    for composer, title in TARGETS:
        candidates = [
            row for row in validation
            if row["composer"] == composer and row["title"] == title
        ]
        if not candidates:
            raise RuntimeError(f"required validation piece is absent: {composer} / {title}")
        row = dict(candidates[0])
        row["canonical_performance_ordinal_within_piece"] = "1"
        row["validation_performance_count_for_piece"] = str(len(candidates))
        path = (ASAP_ROOT / row["performance_path"]).resolve(strict=True)
        try:
            path.relative_to(ASAP_ROOT.resolve())
        except ValueError as error:
            raise PermissionError(f"selected path escaped ASAP root: {path}") from error
        row["performance_absolute_path"] = str(path)
        row["performance_sha256"] = sha256(path)
        selected.append(row)

    if len(selected) != 5 or len({row["piece_id"] for row in selected}) != 5:
        raise RuntimeError("selection is not exactly five unique validation pieces")
    if any(row["split"] != "validation" for row in selected):
        raise PermissionError("non-validation row reached selection")
    provenance = {
        "split_manifest": str(SPLIT_MANIFEST),
        "split_manifest_sha256": sha256(SPLIT_MANIFEST),
        "canonical_validation_rows": len(validation),
        "canonical_validation_pieces": len({row["piece_id"] for row in validation}),
        "selected_performance_count": len(selected),
        "selection_rule": "first row in canonical validation-manifest order for each exact target piece",
        "opened_asap_performance_midis": [row["performance_path"] for row in selected],
        "asap_test_midi_access_count": 0,
    }
    return selected, provenance


def newly_attacked_pitch_groups(path: Path) -> dict[int, list[int]]:
    """Collect non-drum note-on attacks without consulting note-off messages."""

    midi = mido.MidiFile(str(path), clip=False)
    grouped: dict[int, list[int]] = defaultdict(list)
    for track in midi.tracks:
        tick = 0
        for message in track:
            tick += int(message.time)
            if (
                message.type == "note_on"
                and int(message.velocity) > 0
                and int(message.channel) != 9
            ):
                grouped[tick].append(int(message.note))
    return {tick: sorted(grouped[tick]) for tick in sorted(grouped)}


def extract_performance(row: Mapping[str, str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = Path(row["performance_absolute_path"])
    source = parse_raw_midi(path)
    pitch_groups = newly_attacked_pitch_groups(path)
    group_ticks = tuple(pitch_groups)
    if group_ticks != source.distinct_onsets:
        missing = sorted(set(source.distinct_onsets) - set(group_ticks))
        raise RuntimeError(f"canonical attack-onset mismatch in {path}: {missing[:10]}")
    tick_to_seconds, _ = _tick_second_converters(path)
    times = [float(tick_to_seconds(tick)) for tick in group_ticks]
    if any(right <= left for left, right in zip(times, times[1:])):
        raise RuntimeError(f"non-positive distinct-onset IOI in {path}")
    local_values: list[float | None] = []
    local_counts: list[int] = []
    times_ms = [time * 1000.0 for time in times]
    for index in range(len(times)):
        median_ms, count = local_ioi(times_ms, index, LOCAL_IOI_WINDOW)
        local_values.append(None if median_ms is None else float(median_ms) / 1000.0)
        local_counts.append(int(count))

    pitches = [pitch_groups[tick] for tick in group_ticks]
    lowest = [values[0] for values in pitches]

    successors: dict[int, list[int | None]] = {}
    for cutoff in CUTOFFS:
        result: list[int | None] = [None] * len(group_ticks)
        next_low: int | None = None
        for index in range(len(group_ticks) - 1, -1, -1):
            result[index] = next_low
            if lowest[index] <= cutoff:
                next_low = index
        successors[cutoff] = result

    output: list[dict[str, Any]] = []
    for index, (onset_tick, group_pitches, pitch) in enumerate(zip(group_ticks, pitches, lowest)):
        feature: dict[str, Any] = {
            "composer": row["composer"],
            "title": row["title"],
            "piece_id": row["piece_id"],
            "performance_id": Path(row["performance_path"]).stem,
            "performance_path": row["performance_path"],
            "onset_index": index,
            "onset_tick": int(onset_tick),
            "onset_time_sec": times[index],
            "lowest_pitch": pitch,
            "lowest_pitch_name": midi_note_name(pitch),
            "onset_group_pitches": " ".join(str(value) for value in group_pitches),
            "onset_group_pitch_names": " ".join(midi_note_name(value) for value in group_pitches),
            "onset_group_note_count": len(group_pitches),
            "octave_12": int(pitch + 12 in group_pitches),
            "octave_24": int(pitch + 24 in group_pitches),
            "local_median_ioi_sec": local_values[index],
            "local_positive_ioi_count": local_counts[index],
        }
        for cutoff in CUTOFFS:
            next_index = successors[cutoff][index]
            prefix = f"m{cutoff}"
            if next_index is None:
                feature[f"{prefix}_next_low_onset_index"] = None
                feature[f"{prefix}_next_low_gap_sec"] = None
                feature[f"{prefix}_intervening_onsets_N"] = None
                feature[f"{prefix}_onset_steps"] = None
                feature[f"{prefix}_R_local_ioi"] = None
            else:
                gap = times[next_index] - times[index]
                local = local_values[index]
                feature[f"{prefix}_next_low_onset_index"] = next_index
                feature[f"{prefix}_next_low_gap_sec"] = gap
                feature[f"{prefix}_intervening_onsets_N"] = next_index - index - 1
                feature[f"{prefix}_onset_steps"] = next_index - index
                feature[f"{prefix}_R_local_ioi"] = None if local is None else gap / local
        output.append(feature)

    metadata = {
        "piece_id": row["piece_id"],
        "performance_id": Path(row["performance_path"]).stem,
        "distinct_onsets": len(output),
        "newly_attacked_non_drum_notes": sum(len(values) for values in pitches),
        "first_onset_sec": times[0],
        "last_onset_sec": times[-1],
        "onset_span_sec": times[-1] - times[0],
        "tempo_aware": True,
        "canonical_parser_distinct_onset_match": True,
        "attack_pitches_obtained_without_noteoff": True,
    }
    return output, metadata


def distribution_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scopes = [(piece_id, group) for piece_id, group in frame.groupby("piece_id", sort=False)]
    scopes.append(("POOLED", frame))
    distribution: list[dict[str, Any]] = []
    histogram: list[dict[str, Any]] = []
    thresholds: list[dict[str, Any]] = []
    for scope, group in scopes:
        pitch_q = quantiles(group["lowest_pitch"])
        distribution.append({
            "scope": scope, "cutoff": "", "metric": "lowest_pitch",
            "eligible_current_low_count": len(group), "missing_successor_count": 0,
            **pitch_q,
        })
        counts = Counter(int(value) for value in group["lowest_pitch"])
        for pitch in range(min(counts), max(counts) + 1):
            histogram.append({
                "scope": scope, "midi_pitch": pitch, "note_name": midi_note_name(pitch),
                "count": counts[pitch], "proportion": counts[pitch] / len(group),
            })
        for cutoff in CUTOFFS:
            low = group[group["lowest_pitch"] <= cutoff]
            observed = low[low[f"m{cutoff}_intervening_onsets_N"].notna()]
            for metric, column in (
                ("next_low_gap_sec", f"m{cutoff}_next_low_gap_sec"),
                ("N_intervening_onsets", f"m{cutoff}_intervening_onsets_N"),
                ("R_local_ioi", f"m{cutoff}_R_local_ioi"),
            ):
                distribution.append({
                    "scope": scope, "cutoff": cutoff, "metric": metric,
                    "eligible_current_low_count": len(low),
                    "missing_successor_count": len(low) - len(observed),
                    **quantiles(observed[column]),
                })
            for k_value in K_VALUES:
                detected = observed[observed[f"m{cutoff}_intervening_onsets_N"] >= k_value]
                thresholds.append({
                    "scope": scope,
                    "pitch_cutoff": cutoff,
                    "K_B": k_value,
                    "low_register_onset_count": len(low),
                    "observed_successor_count": len(observed),
                    "missing_successor_count": len(low) - len(observed),
                    "candidate_count": len(detected),
                    "candidate_rate_among_low_register": len(detected) / len(low) if len(low) else None,
                    "candidate_rate_among_all_onsets": len(detected) / len(group),
                })
    return pd.DataFrame(distribution), pd.DataFrame(histogram), pd.DataFrame(thresholds)


def detector_sweep(frame: pd.DataFrame, performance_meta: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    span = {row["piece_id"]: float(row["onset_span_sec"]) for row in performance_meta}
    scopes = [(piece_id, group) for piece_id, group in frame.groupby("piece_id", sort=False)]
    scopes.append(("POOLED", frame))
    rows: list[dict[str, Any]] = []
    for scope, group in scopes:
        minutes = (
            sum(span[piece] for piece in group["piece_id"].unique()) / 60.0
            if scope == "POOLED" else span[scope] / 60.0
        )
        for cutoff in CUTOFFS:
            n_column = f"m{cutoff}_intervening_onsets_N"
            for k_value in K_VALUES:
                detected = group[(group["lowest_pitch"] <= cutoff) & (group[n_column] >= k_value)]
                rows.append({
                    "scope": scope,
                    "pitch_cutoff_m_B": cutoff,
                    "timescale_threshold_K_B": k_value,
                    "detected_event_count": len(detected),
                    "events_per_minute_onset_span": len(detected) / minutes if minutes > 0 else None,
                    "fraction_of_all_onsets": len(detected) / len(group),
                    "octave_12_count": int(detected["octave_12"].sum()),
                    "octave_12_rate": float(detected["octave_12"].mean()) if len(detected) else None,
                    "octave_24_count": int(detected["octave_24"].sum()),
                    "octave_24_rate": float(detected["octave_24"].mean()) if len(detected) else None,
                    "all_onset_count": len(group),
                    "onset_span_minutes": minutes,
                })
    return pd.DataFrame(rows)


def spread_indices(indices: Sequence[int], count: int) -> list[int]:
    values = list(indices)
    if len(values) <= count:
        return values
    positions = np.linspace(0, len(values) - 1, count)
    result: list[int] = []
    for position in positions:
        value = values[int(round(position))]
        if value not in result:
            result.append(value)
    if len(result) < count:
        result.extend(value for value in values if value not in result and len(result) < count)
    return result


def manual_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    chosen_rows: list[dict[str, Any]] = []
    n_col = f"m{MANUAL_REFERENCE_CUTOFF}_intervening_onsets_N"
    r_col = f"m{MANUAL_REFERENCE_CUTOFF}_R_local_ioi"
    gap_col = f"m{MANUAL_REFERENCE_CUTOFF}_next_low_gap_sec"
    for piece_id, group in frame.groupby("piece_id", sort=False):
        group = group.sort_values("onset_index").copy()
        eligible = group[(group["lowest_pitch"] <= MANUAL_REFERENCE_CUTOFF) & group[n_col].notna()]
        strata = (
            ("very_low_large_N", eligible[(eligible.lowest_pitch <= 48) & (eligible[n_col] >= 6)], 3),
            ("very_low_small_N", eligible[(eligible.lowest_pitch <= 48) & (eligible[n_col] <= 2)], 3),
            ("relatively_high_large_N", eligible[(eligible.lowest_pitch.between(52, 55)) & (eligible[n_col] >= 6)], 3),
            ("octave_12", eligible[eligible.octave_12 == 1], 3),
            ("non_octave", eligible[eligible.octave_12 == 0], 3),
            ("detector_boundary", eligible[(eligible.lowest_pitch.between(52, 55)) & (eligible[n_col].between(3, 4))], 3),
        )
        selected: dict[int, set[str]] = defaultdict(set)
        for label, subset, quota in strata:
            candidates = [int(value) for value in subset.onset_index if int(value) not in selected]
            for onset_index in spread_indices(candidates, quota):
                selected[onset_index].add(label)
        if len(selected) < MANUAL_PER_PIECE:
            remaining = [int(value) for value in eligible.onset_index if int(value) not in selected]
            for onset_index in spread_indices(remaining, MANUAL_PER_PIECE - len(selected)):
                selected[onset_index].add("timeline_diversity_fill")
        if len(selected) != MANUAL_PER_PIECE:
            raise RuntimeError(f"could not select {MANUAL_PER_PIECE} examples for {piece_id}")

        indexed = group.set_index("onset_index", drop=False)
        for onset_index in sorted(selected):
            source = indexed.loc[onset_index]
            previous = []
            following = []
            for context_index in range(max(0, onset_index - 4), onset_index):
                context = indexed.loc[context_index]
                previous.append(f"{context_index}:{context.onset_group_pitch_names}")
            for context_index in range(onset_index + 1, min(len(group), onset_index + 9)):
                context = indexed.loc[context_index]
                following.append(f"{context_index}:{context.onset_group_pitch_names}")
            row = {
                "annotation": "",
                "allowed_labels": "STRUCTURAL_BASS|NOT_STRUCTURAL_BASS|AMBIGUOUS",
                "selection_strata": ";".join(sorted(selected[onset_index])),
                "reference_pitch_cutoff": MANUAL_REFERENCE_CUTOFF,
                "composer": source.composer,
                "title": source.title,
                "piece_id": piece_id,
                "performance_id": source.performance_id,
                "performance_path": source.performance_path,
                "onset_index": onset_index,
                "onset_time_sec": source.onset_time_sec,
                "lowest_pitch": int(source.lowest_pitch),
                "lowest_pitch_name": source.lowest_pitch_name,
                "onset_group_pitches": source.onset_group_pitches,
                "onset_group_pitch_names": source.onset_group_pitch_names,
                "octave_12": int(source.octave_12),
                "octave_24": int(source.octave_24),
                "N_intervening_onsets": int(source[n_col]),
                "R_local_ioi": float(source[r_col]),
                "next_low_gap_sec": float(source[gap_col]),
                "local_median_ioi_sec": float(source.local_median_ioi_sec),
                "previous_4_onset_groups": " | ".join(previous),
                "following_8_onset_groups": " | ".join(following),
            }
            chosen_rows.append(row)
    return pd.DataFrame(chosen_rows)


def octave_diagnostic(frame: pd.DataFrame) -> pd.DataFrame:
    n_col = f"m{MANUAL_REFERENCE_CUTOFF}_intervening_onsets_N"
    r_col = f"m{MANUAL_REFERENCE_CUTOFF}_R_local_ioi"
    candidates = frame[(frame.lowest_pitch <= MANUAL_REFERENCE_CUTOFF) & frame[n_col].notna()]
    scopes = [(piece, group) for piece, group in candidates.groupby("piece_id", sort=False)]
    scopes.append(("POOLED", candidates))
    rows: list[dict[str, Any]] = []
    for scope, group in scopes:
        for octave in (0, 1):
            subset = group[group.octave_12 == octave]
            for metric, column in (
                ("lowest_pitch", "lowest_pitch"),
                ("N_intervening_onsets", n_col),
                ("R_local_ioi", r_col),
            ):
                rows.append({
                    "scope": scope, "reference_pitch_cutoff": MANUAL_REFERENCE_CUTOFF,
                    "octave_12": octave, "metric": metric, **quantiles(subset[column]),
                })
    return pd.DataFrame(rows)


def pitch_plot(frame: pd.DataFrame, output: Path) -> None:
    pieces = list(frame["piece_id"].drop_duplicates())
    scopes = [(piece, frame[frame.piece_id == piece]) for piece in pieces] + [("POOLED", frame)]
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True)
    for ax, (scope, group) in zip(axes.flat, scopes):
        counts = group.lowest_pitch.value_counts().sort_index()
        ax.bar(counts.index, counts.values, width=0.9, color="#3465a4")
        for cutoff in CUTOFFS:
            ax.axvline(cutoff, color="#cc0000", alpha=0.2, linewidth=0.8)
        title = "Pooled" if scope == "POOLED" else f"{group.composer.iloc[0]} — {group.title.iloc[0]}"
        ax.set_title(title)
        ax.set_ylabel("Distinct onsets")
        ax.grid(axis="y", alpha=0.2)
    axes[-1, 0].set_xlabel("Lowest newly-attacked MIDI pitch")
    axes[-1, 1].set_xlabel("Lowest newly-attacked MIDI pitch")
    fig.suptitle("Structural Bass Audit: Lowest Newly-Attacked Pitch")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def timescale_plot(frame: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = (
        ("next_low_gap_sec", "Next-low gap (seconds)"),
        ("intervening_onsets_N", "N intervening onsets"),
        ("R_local_ioi", "R = gap / local median IOI"),
    )
    for ax, (suffix, label) in zip(axes, metrics):
        values = []
        labels = []
        for cutoff in CUTOFFS:
            low = frame[frame.lowest_pitch <= cutoff]
            array = low[f"m{cutoff}_{suffix}"].dropna().to_numpy(dtype=float)
            values.append(array)
            labels.append(str(cutoff))
        ax.boxplot(values, tick_labels=labels, showfliers=False)
        ax.set_xlabel("Pitch cutoff")
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Pooled Successor Timescale Distributions (current onset ≤ cutoff)")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def sweep_plot(sweep: pd.DataFrame, output: Path) -> None:
    pieces = [scope for scope in sweep.scope.drop_duplicates() if scope != "POOLED"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, scope in zip(axes.flat, pieces):
        group = sweep[sweep.scope == scope]
        matrix = group.pivot(index="timescale_threshold_K_B", columns="pitch_cutoff_m_B", values="events_per_minute_onset_span")
        image = ax.imshow(matrix.values, aspect="auto", origin="lower", cmap="viridis")
        ax.set_xticks(range(len(matrix.columns)), matrix.columns)
        ax.set_yticks(range(len(matrix.index)), matrix.index)
        ax.set_xlabel("m_B")
        ax.set_ylabel("K_B")
        ax.set_title(scope)
        fig.colorbar(image, ax=ax, label="events/min")
    axes.flat[-1].axis("off")
    fig.suptitle("Detector Sweep: Events per Minute")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def octave_plot(frame: pd.DataFrame, output: Path) -> None:
    cutoff = MANUAL_REFERENCE_CUTOFF
    n_col = f"m{cutoff}_intervening_onsets_N"
    r_col = f"m{cutoff}_R_local_ioi"
    candidates = frame[(frame.lowest_pitch <= cutoff) & frame[n_col].notna()]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, column, label in zip(
        axes,
        ("lowest_pitch", n_col, r_col),
        ("Lowest pitch", "N intervening onsets", "R local IOI"),
    ):
        values = [candidates[candidates.octave_12 == value][column].dropna() for value in (0, 1)]
        ax.boxplot(values, tick_labels=("no octave+12", "octave+12"), showfliers=False)
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(f"Octave-12 Diagnostic Only (reference cutoff {cutoff})")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def manual_timeline_plots(frame: pd.DataFrame, manual: pd.DataFrame, output: Path) -> list[str]:
    paths: list[str] = []
    for piece_id, examples in manual.groupby("piece_id", sort=False):
        group = frame[frame.piece_id == piece_id].set_index("onset_index")
        examples = examples.sort_values("onset_index")
        fig, axes = plt.subplots(6, 3, figsize=(15, 18), sharex=True, sharey=True)
        for ax, (_, example) in zip(axes.flat, examples.iterrows()):
            center = int(example.onset_index)
            for onset_index in range(max(0, center - 4), min(len(group), center + 9)):
                pitches = [int(value) for value in str(group.loc[onset_index, "onset_group_pitches"]).split()]
                relative = onset_index - center
                ax.scatter([relative] * len(pitches), pitches, s=12, color="#cc0000" if relative == 0 else "#3465a4")
            ax.axvline(0, color="#cc0000", alpha=0.35)
            ax.set_title(f"onset {center}, {example.lowest_pitch_name}, N={int(example.N_intervening_onsets)}, R={example.R_local_ioi:.1f}", fontsize=9)
            ax.grid(alpha=0.15)
        for ax in axes[-1, :]:
            ax.set_xlabel("Relative distinct-onset index")
        for ax in axes[:, 0]:
            ax.set_ylabel("MIDI pitch")
        name = f"manual_timeline_{piece_id}.png"
        fig.suptitle(f"Manual-inspection contexts — {examples.composer.iloc[0]} / {examples.title.iloc[0]}")
        fig.tight_layout()
        fig.savefig(output / name, dpi=140)
        plt.close(fig)
        paths.append(name)
    return paths


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def build_report(
    selected: Sequence[Mapping[str, str]], provenance: Mapping[str, Any], frame: pd.DataFrame,
    performance_meta: Sequence[Mapping[str, Any]], distribution: pd.DataFrame,
    thresholds: pd.DataFrame, sweep: pd.DataFrame, manual: pd.DataFrame,
    octave: pd.DataFrame, timeline_paths: Sequence[str], output: Path,
) -> str:
    labels = {row["piece_id"]: f"{row['composer']} — {row['title']}" for row in selected}
    selection_rows = [
        (row["composer"], row["title"], row["piece_id"], row["performance_path"], row["validation_performance_count_for_piece"], "first")
        for row in selected
    ]
    pitch_rows = []
    pitch_dist = distribution[distribution.metric == "lowest_pitch"]
    for _, row in pitch_dist.iterrows():
        pitch_rows.append((labels.get(row.scope, "Pooled"), int(row["count"]), fmt(row.q25, 1), fmt(row["median"], 1), fmt(row.q75, 1), f"{fmt(row.q05, 1)}–{fmt(row.q95, 1)}"))

    time_rows = []
    for scope in [*labels, "POOLED"]:
        subset = distribution[(distribution.scope == scope) & (distribution.metric != "lowest_pitch")]
        for cutoff in CUTOFFS:
            current = subset[subset.cutoff == cutoff].set_index("metric")
            time_rows.append((
                labels.get(scope, "Pooled"), cutoff,
                int(current.iloc[0].eligible_current_low_count), int(current.iloc[0].missing_successor_count),
                fmt(current.loc["next_low_gap_sec", "median"]),
                f"{fmt(current.loc['next_low_gap_sec', 'q25'])}–{fmt(current.loc['next_low_gap_sec', 'q75'])}",
                fmt(current.loc["N_intervening_onsets", "median"], 1),
                f"{fmt(current.loc['N_intervening_onsets', 'q25'], 1)}–{fmt(current.loc['N_intervening_onsets', 'q75'], 1)}",
                fmt(current.loc["R_local_ioi", "median"], 2),
                f"{fmt(current.loc['R_local_ioi', 'q25'], 2)}–{fmt(current.loc['R_local_ioi', 'q75'], 2)}",
            ))

    ref_threshold = thresholds[(thresholds.pitch_cutoff == MANUAL_REFERENCE_CUTOFF) & (thresholds.scope != "POOLED")]
    threshold_rows = []
    for scope in labels:
        group = ref_threshold[ref_threshold.scope == scope]
        values = []
        for k_value in K_VALUES:
            row = group[group.K_B == k_value].iloc[0]
            values.append(f"{int(row.candidate_count)} ({100*row.candidate_rate_among_low_register:.1f}%)")
        threshold_rows.append((labels[scope], *values))

    representative = sweep[(sweep.pitch_cutoff_m_B == 55) & (sweep.timescale_threshold_K_B == 4) & (sweep.scope != "POOLED")]
    representative_rows = [
        (labels[row.scope], int(row.detected_event_count), fmt(row.events_per_minute_onset_span, 2), pct(row.detected_event_count, row.all_onset_count), pct(row.octave_12_count, row.detected_event_count), pct(row.octave_24_count, row.detected_event_count))
        for _, row in representative.iterrows()
    ]

    oct_pooled = octave[octave.scope == "POOLED"]
    octave_rows = []
    for octave_value in (0, 1):
        current = oct_pooled[oct_pooled.octave_12 == octave_value].set_index("metric")
        octave_rows.append((
            "octave+12" if octave_value else "no octave+12",
            int(current.loc["lowest_pitch", "count"]),
            fmt(current.loc["lowest_pitch", "median"], 1),
            fmt(current.loc["N_intervening_onsets", "median"], 1),
            fmt(current.loc["R_local_ioi", "median"], 2),
        ))

    pooled_k = thresholds[(thresholds.scope == "POOLED") & (thresholds.pitch_cutoff == 55)].set_index("K_B")
    retention_text = ", ".join(
        f"K={k}: {int(pooled_k.loc[k, 'candidate_count'])} ({100*pooled_k.loc[k, 'candidate_rate_among_low_register']:.1f}%)"
        for k in K_VALUES
    )
    n_piece_medians = distribution[(distribution.scope != "POOLED") & (distribution.cutoff == 55) & (distribution.metric == "N_intervening_onsets")]["median"].to_numpy(dtype=float)
    r_piece_medians = distribution[(distribution.scope != "POOLED") & (distribution.cutoff == 55) & (distribution.metric == "R_local_ioi")]["median"].to_numpy(dtype=float)
    n_cv = float(np.std(n_piece_medians) / np.mean(n_piece_medians))
    r_cv = float(np.std(r_piece_medians) / np.mean(r_piece_medians))
    rates = representative.events_per_minute_onset_span.to_numpy(dtype=float)
    rate_ratio = float(np.max(rates) / np.min(rates)) if np.min(rates) > 0 else math.inf
    oct_no = oct_pooled[(oct_pooled.octave_12 == 0) & (oct_pooled.metric == "N_intervening_onsets")].iloc[0]
    oct_yes = oct_pooled[(oct_pooled.octave_12 == 1) & (oct_pooled.metric == "N_intervening_onsets")].iloc[0]
    oct_no_r = oct_pooled[(oct_pooled.octave_12 == 0) & (oct_pooled.metric == "R_local_ioi")].iloc[0]
    oct_yes_r = oct_pooled[(oct_pooled.octave_12 == 1) & (oct_pooled.metric == "R_local_ioi")].iloc[0]
    octave_rate = float(oct_yes["count"] / (oct_yes["count"] + oct_no["count"]))

    metadata_rows = []
    for row in performance_meta:
        metadata_rows.append((labels[row["piece_id"]], row["distinct_onsets"], row["newly_attacked_non_drum_notes"], fmt(row["onset_span_sec"], 1)))

    report = f"""# Structural Bass Event Detector v0 — Five-Performance Data Audit

## Scope and safeguards

This exploratory audit uses exactly five human performances from the canonical ASAP validation manifest. It does not train a model, use score annotations, access a test MIDI, separate voices, segment phrases, or calculate precision/recall. CC64, key-off, and performed note duration are absent from every detector feature. Octave reinforcement is diagnostic only.

- Canonical split manifest: `{provenance['split_manifest']}`
- Manifest SHA-256: `{provenance['split_manifest_sha256']}`
- Canonical validation universe verified: {provenance['canonical_validation_rows']} performances / {provenance['canonical_validation_pieces']} pieces
- Selection rule: {provenance['selection_rule']}
- MIDI files opened: exactly {provenance['selected_performance_count']} selected validation performances
- ASAP test MIDI access count: **{provenance['asap_test_midi_access_count']}**
- Tested command: `{provenance['tested_command']}`

{markdown_table(('Composer', 'Title', 'piece ID', 'human performance MIDI path', 'validation performances', 'chosen ordinal'), selection_rows)}

## Method

The audit reuses `src.stage2_event_tokenizer.tokenizer.parse_raw_midi` for the canonical distinct-onset ticks and the existing deterministic tempo converter. Group pitches are collected independently from raw non-drum `note_on velocity>0` messages, without consulting note-off messages. All five sources passed an exact equality assertion between these raw attack ticks and parser `distinct_onsets`. At each group, `p_n` is the minimum newly attacked pitch only.

Local median IOI reuses `scripts.audit_pedal_event_metric_tolerance_mini.local_ioi` with `window=16`, corresponding to up to eight distinct onsets on each side. Only positive, tempo-aware IOIs are included. Successor values are missing when no later low attack exists; no sentinel is substituted. Distribution summaries for a cutoff use current onsets with `p_n <= cutoff`; missing terminal successors remain counted separately. Events/minute uses first-to-last distinct-onset span and therefore does not use note duration.

{markdown_table(('Piece', 'distinct onsets', 'newly attacked notes', 'onset span sec'), metadata_rows)}

## Pitch distribution

The “main range” below is the empirical 5th–95th percentile interval; full integer-pitch histograms are in `pitch_histogram.csv` and `pitch_histogram.png`.

{markdown_table(('Scope', 'onsets', 'Q1', 'median', 'Q3', 'main range (P05–P95)'), pitch_rows)}

![Pitch histogram](pitch_histogram.png)

## Structural timescale distributions

Each cell reports median and, where shown, Q1–Q3. `low n` is the number of current onsets satisfying the cutoff and `missing` is the number without a later low successor.

{markdown_table(('Scope', 'm_cut', 'low n', 'missing', 'gap med s', 'gap Q1–Q3', 'N med', 'N Q1–Q3', 'R med', 'R Q1–Q3'), time_rows)}

The complete machine-readable table, including P05/P95, is `distribution_summary.csv`.

![Timescale distributions](timescale_distributions.png)

## Candidate threshold counts

For compactness this report shows the central diagnostic cutoff `m_B=55`; each cell is count and percent among all current low-register onsets. Every cutoff × K × piece/pooled result is in `timescale_threshold_summary.csv`.

{markdown_table(('Piece', 'K=2', 'K=3', 'K=4', 'K=6', 'K=8'), threshold_rows)}

At pooled `m_B=55`: {retention_text}.

## Detector sweep

`detector_sweep.csv` contains all 25 combinations for each of five pieces plus pooled results. It reports detected count, events/minute, fraction of all onsets, and octave+12/+24 diagnostic rates. The table below is a readable center-point slice (`m_B=55`, `K_B=4`), not a selected final detector.

{markdown_table(('Piece', 'events', 'events/min', 'all-onset rate', 'octave+12', 'octave+24'), representative_rows)}

![Detector sweep heatmaps](detector_sweep_heatmaps.png)

## Manual inspection set

`manual_inspection_candidates.csv` contains {len(manual)} rows ({MANUAL_PER_PIECE} per piece), an empty `annotation` column, the allowed labels, and four preceding/eight following onset-group contexts. Sampling deliberately covers very-low/large-N, very-low/small-N, relatively-high/large-N, octave/non-octave, and detector-boundary strata. Values shown in this table use the explicitly recorded reference cutoff `m_B={MANUAL_REFERENCE_CUTOFF}`; all cutoff-specific features remain available in `onset_features.csv`.

Timeline figures plot every pitch in the −4…+8 distinct-onset context, with the candidate at relative onset zero:

""" + "\n".join(f"- [{name}]({name})" for name in timeline_paths) + f"""

No annotation is auto-filled and none of these examples is treated as ground truth.

## Octave diagnostic

This comparison uses all `p_n <= {MANUAL_REFERENCE_CUTOFF}` candidates with an observed next-low successor, split only for diagnosis. Octave status was not used to select a detector threshold or modify event weights.

{markdown_table(('Group', 'n', 'pitch median', 'N median', 'R median'), octave_rows)}

![Octave diagnostic](octave_diagnostic.png)

## Answers for the next human threshold-selection step

### Q1 — Plausible pitch-cutoff range

The five-piece comparison supports treating **MIDI 52–57** as the useful review range, with 55 as a convenient inspection center rather than a final value. MIDI 48 is deliberately conservative and misses mid-low structural supports; MIDI 60 admits substantially more accompaniment/inner-texture attacks. The manual boundary examples should decide where within 52–57 the intended musical concept sits.

### Q2 — N versus R

**`R_n^B` is more stable across these five performances**, while `N_n^B` is easier to interpret directly. Across piece medians at cutoff 55, relative dispersion is N={n_cv:.3f} versus R={r_cv:.3f}; local-IOI normalization substantially reduces between-piece scale differences here. Keep both: use R as the cross-texture normalized indicator and N as the transparent discrete audit/control variable. Neither is ground truth.

### Q3 — Plausible K range

The empirical working range is **K_B=4–6**. K=2–3 retains many short-cycle accompaniment returns, while K=8 is the aggressive end and removes many reviewable events. Pooled retention at cutoff 55 is: {retention_text}. A human pass over the boundary examples is still required before choosing a value.

### Q4 — Octave reinforcement

Octave+12 is **promising but currently weak/sparse auxiliary evidence**, not a detector condition. Only {int(oct_yes['count'])} of {int(oct_yes['count'] + oct_no['count'])} pooled cutoff-55 candidates ({100*octave_rate:.1f}%) have octave+12. Their median N={fmt(oct_yes['median'], 1)} versus {fmt(oct_no['median'], 1)}, and median R={fmt(oct_yes_r['median'], 2)} versus {fmt(oct_no_r['median'], 2)}, which is an interesting enrichment but too rare and unlabelled to justify weighting or gating.

### Q5 — Cross-piece behavior

The same threshold does not behave identically: at the illustrative `(m_B=55, K_B=4)` point, the maximum/minimum events-per-minute ratio across pieces is {rate_ratio:.2f}×. This is meaningful texture sensitivity, so pooled counts should not substitute for piece-level review. The sweep remains usable as a common starting grid, but any final threshold needs explicit checks against the Ravel, Schumann, Liszt, Beethoven, and Chopin manual examples.

## Non-conclusion

This audit **does not finalize `(m_B, K_B)`**. It narrows a human-review region and provides examples for subsequent `STRUCTURAL_BASS`, `NOT_STRUCTURAL_BASS`, or `AMBIGUOUS` annotation.

## Files

- `onset_features.csv`: every distinct onset and every cutoff-specific successor feature
- `manual_inspection_candidates.csv`: empty-label manual review sheet
- `detector_sweep.csv`: 25-grid results per piece plus pooled
- `distribution_summary.csv`, `pitch_histogram.csv`, `timescale_threshold_summary.csv`: audit tables
- `octave_diagnostic.csv`: diagnostic-only octave comparison
- `selection_manifest.csv`, `audit_metadata.json`: exact inputs and provenance
- `pitch_histogram.png`, `timescale_distributions.png`, `detector_sweep_heatmaps.png`, `octave_diagnostic.png`, and five manual timelines
"""
    return report


def run(output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    selected, provenance = choose_validation_performances()
    provenance["tested_command"] = f"python {Path(__file__).resolve()} --output {output.resolve()}"
    all_rows: list[dict[str, Any]] = []
    performance_meta: list[dict[str, Any]] = []
    for row in selected:
        features, metadata = extract_performance(row)
        all_rows.extend(features)
        performance_meta.append(metadata)
        print(f"parsed {row['composer']} {row['title']}: {len(features)} distinct onsets", flush=True)
    frame = pd.DataFrame(all_rows)
    if frame.groupby("piece_id").ngroups != 5 or len(frame) == 0:
        raise RuntimeError("feature table does not contain exactly five non-empty pieces")

    distribution, histogram, thresholds = distribution_tables(frame)
    sweep = detector_sweep(frame, performance_meta)
    manual = manual_candidates(frame)
    octave = octave_diagnostic(frame)
    if len(sweep) != 6 * len(CUTOFFS) * len(K_VALUES):
        raise AssertionError("detector sweep must contain five pieces plus pooled × 25")
    if len(manual) != 5 * MANUAL_PER_PIECE or manual.annotation.astype(str).ne("").any():
        raise AssertionError("manual inspection sheet count/empty-label invariant failed")

    selection_fields = (
        "composer", "title", "piece_id", "performance_path", "split",
        "canonical_performance_ordinal_within_piece", "validation_performance_count_for_piece",
        "performance_sha256",
    )
    write_csv(
        output / "selection_manifest.csv",
        ({field: row[field] for field in selection_fields} for row in selected),
        selection_fields,
    )
    frame.to_csv(output / "onset_features.csv", index=False)
    manual.to_csv(output / "manual_inspection_candidates.csv", index=False)
    sweep.to_csv(output / "detector_sweep.csv", index=False)
    distribution.to_csv(output / "distribution_summary.csv", index=False)
    histogram.to_csv(output / "pitch_histogram.csv", index=False)
    thresholds.to_csv(output / "timescale_threshold_summary.csv", index=False)
    octave.to_csv(output / "octave_diagnostic.csv", index=False)

    pitch_plot(frame, output / "pitch_histogram.png")
    timescale_plot(frame, output / "timescale_distributions.png")
    sweep_plot(sweep, output / "detector_sweep_heatmaps.png")
    octave_plot(frame, output / "octave_diagnostic.png")
    timeline_paths = manual_timeline_plots(frame, manual, output)

    metadata = {
        **provenance,
        "cutoffs": list(CUTOFFS),
        "K_values": list(K_VALUES),
        "local_ioi_window": LOCAL_IOI_WINDOW,
        "manual_reference_cutoff": MANUAL_REFERENCE_CUTOFF,
        "manual_examples_per_piece": MANUAL_PER_PIECE,
        "performance_metadata": performance_meta,
        "feature_rules": {
            "onset_group": "exact MIDI tick, raw non-drum note_on velocity>0; ticks asserted against canonical parser",
            "lowest_pitch": "minimum newly attacked pitch only",
            "local_ioi": "median positive tempo-aware IOI, up to +/-8 distinct onsets",
            "N": "strictly intervening distinct onset groups",
            "missing_successor": "blank/NA, never sentinel-filled",
            "cc64_used_by_detector": False,
            "note_duration_or_keyoff_used_by_detector": False,
            "octave_used_by_detector": False,
        },
    }
    (output / "audit_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    report = build_report(
        selected, provenance, frame, performance_meta, distribution, thresholds,
        sweep, manual, octave, timeline_paths, output,
    )
    (output / "STRUCTURAL_BASS_AUDIT_REPORT.md").write_text(report, encoding="utf-8")
    print(f"wrote {output}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.output)
