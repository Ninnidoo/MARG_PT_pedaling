#!/usr/bin/env python3
"""Regroup the five-performance Structural Bass Event v0 audit at 20 ms.

This is a validation-only descriptive audit and manual-annotation data builder.
It does not choose detector thresholds or produce automatic labels. Detector
features use non-drum note-on attacks only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
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

from src.stage2_event_tokenizer.tokenizer_v1 import _tick_second_converters


GROUP_WINDOW_SEC = 0.020
LOCAL_RADIUS = 8
CUTOFFS = (52, 55, 57)
MANUAL_REFERENCE_CUTOFF = 57
MANUAL_PER_PIECE = 20
R_STRATA = 5
ASAP_ROOT = Path("/workspace/public/ASAP/asap-dataset-v1.1")
SPLIT_MANIFEST = ROOT / "analysis/stage2_encoder_only_v0/asap_split.csv"
OLD_OUTPUT = ROOT / "analysis/structural_bass_event_detector_v0"
DEFAULT_OUTPUT = ROOT / "analysis/structural_bass_event_detector_v0_regroup20ms"

# These exact rows, and no other ASAP performances, are permitted in this audit.
TARGETS = (
    ("Ravel", "Pavane", "piece_de3f82957f1b3532", "Ravel/Pavane/ChenS03.mid"),
    ("Schumann", "Arabeske", "piece_daefdda4e1923cc6", "Schumann/Arabeske/Min09M.mid"),
    ("Liszt", "Mephisto_Waltz", "piece_7195bbce81550519", "Liszt/Mephisto_Waltz/Avdeeva03.mid"),
    ("Beethoven", "Piano_Sonatas_27-1", "piece_69862af5096ee3fa", "Beethoven/Piano_Sonatas/27-1/Abdelmola01.mid"),
    ("Chopin", "Etudes_op_25_12", "piece_db97fbaed2036f5b", "Chopin/Etudes_op_25/12/Atzinger03.mid"),
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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def midi_note_name(pitch: int) -> str:
    names = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    return f"{names[int(pitch) % 12]}{int(pitch) // 12 - 1}"


def fmt(value: Any, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "—"
    number = float(value)
    return "—" if not math.isfinite(number) else f"{number:.{digits}f}"


def pct_change(old: float, new: float) -> float | None:
    return None if not math.isfinite(old) or old == 0 else 100.0 * (new - old) / old


def quantiles(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray([float(value) for value in values if pd.notna(value)], dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"count": 0, "q25": None, "median": None, "q75": None, "q95": None}
    return {
        "count": int(len(array)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.50)),
        "q75": float(np.quantile(array, 0.75)),
        "q95": float(np.quantile(array, 0.95)),
    }


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def choose_validation_performances() -> tuple[list[dict[str, str]], dict[str, Any]]:
    manifest_rows = read_csv(SPLIT_MANIFEST)
    validation = [row for row in manifest_rows if row["split"] == "validation"]
    if len(validation) != 71 or len({row["piece_id"] for row in validation}) != 19:
        raise RuntimeError("canonical validation manifest is not 71 performances / 19 pieces")

    selected: list[dict[str, str]] = []
    for composer, title, piece_id, performance_path in TARGETS:
        matches = [
            row for row in validation
            if row["composer"] == composer
            and row["title"] == title
            and row["piece_id"] == piece_id
            and row["performance_path"] == performance_path
        ]
        if len(matches) != 1:
            raise RuntimeError(f"expected one exact validation row, got {len(matches)}: {performance_path}")
        row = dict(matches[0])
        path = (ASAP_ROOT / performance_path).resolve(strict=True)
        try:
            path.relative_to(ASAP_ROOT.resolve())
        except ValueError as error:
            raise PermissionError(f"selected path escaped ASAP root: {path}") from error
        row["performance_absolute_path"] = str(path)
        row["performance_sha256"] = sha256(path)
        selected.append(row)

    if len(selected) != 5 or any(row["split"] != "validation" for row in selected):
        raise PermissionError("selection is not exactly five validation performances")
    provenance = {
        "split_manifest": str(SPLIT_MANIFEST),
        "split_manifest_sha256": sha256(SPLIT_MANIFEST),
        "canonical_validation_rows": len(validation),
        "canonical_validation_pieces": len({row["piece_id"] for row in validation}),
        "selected_performance_count": len(selected),
        "selection_rule": "five exact user-specified validation manifest rows",
        "opened_asap_performance_midis": [row["performance_path"] for row in selected],
        "asap_test_midi_access_count": 0,
    }
    return selected, provenance


def note_attacks(path: Path) -> list[dict[str, Any]]:
    """Read non-drum positive-velocity attacks; never inspect note-off or CC."""

    midi = mido.MidiFile(str(path), clip=False)
    tick_to_seconds, _ = _tick_second_converters(path)
    attacks: list[dict[str, Any]] = []
    for track_index, track in enumerate(midi.tracks):
        tick = 0
        for message_index, message in enumerate(track):
            tick += int(message.time)
            if (
                message.type == "note_on"
                and int(message.velocity) > 0
                and int(message.channel) != 9
            ):
                attacks.append({
                    "time_sec": float(tick_to_seconds(tick)),
                    "tick": int(tick),
                    "track_index": int(track_index),
                    "message_index": int(message_index),
                    "pitch": int(message.note),
                })
    attacks.sort(key=lambda value: (value["time_sec"], value["track_index"], value["message_index"]))
    if not attacks:
        raise RuntimeError(f"no non-drum note attacks: {path}")
    return attacks


def earliest_anchor_groups(attacks: Sequence[Mapping[str, Any]], window_sec: float = GROUP_WINDOW_SEC) -> list[list[Mapping[str, Any]]]:
    """Group attacks within an inclusive window from each earliest anchor."""

    groups: list[list[Mapping[str, Any]]] = []
    index = 0
    while index < len(attacks):
        anchor = float(attacks[index]["time_sec"])
        stop = index + 1
        while stop < len(attacks) and float(attacks[stop]["time_sec"]) - anchor <= window_sec + 1e-12:
            stop += 1
        groups.append(list(attacks[index:stop]))
        index = stop
    return groups


def assert_grouping_semantics() -> None:
    synthetic = [
        {"time_sec": time, "track_index": 0, "message_index": index, "pitch": 60 + index}
        for index, time in enumerate((0.000, 0.015, 0.030))
    ]
    result = earliest_anchor_groups(synthetic)
    if [[round(1000 * float(x["time_sec"])) for x in group] for group in result] != [[0, 15], [30]]:
        raise AssertionError("20 ms grouping accidentally uses single-linkage chaining")


def local_ioi_seconds(times: Sequence[float], onset_index: int) -> tuple[float | None, int]:
    if len(times) < 2:
        return None, 0
    start = max(0, onset_index - LOCAL_RADIUS)
    stop = min(len(times) - 1, onset_index + LOCAL_RADIUS)
    values = [
        float(times[index + 1]) - float(times[index])
        for index in range(start, stop)
        if float(times[index + 1]) - float(times[index]) > 0
    ]
    return (float(np.median(values)), len(values)) if values else (None, 0)


def extract_performance(row: Mapping[str, str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    attacks = note_attacks(Path(row["performance_absolute_path"]))
    groups = earliest_anchor_groups(attacks)
    times = [float(group[0]["time_sec"]) for group in groups]
    if any(right <= left for left, right in zip(times, times[1:])):
        raise AssertionError("group anchors must be strictly increasing")
    local = [local_ioi_seconds(times, index) for index in range(len(times))]
    pitches = [sorted(int(attack["pitch"]) for attack in group) for group in groups]
    lowest = [group[0] for group in pitches]

    successors: dict[int, list[int | None]] = {}
    for cutoff in CUTOFFS:
        values: list[int | None] = [None] * len(groups)
        next_low: int | None = None
        for index in range(len(groups) - 1, -1, -1):
            values[index] = next_low
            if lowest[index] <= cutoff:
                next_low = index
        successors[cutoff] = values

    output: list[dict[str, Any]] = []
    for index, (group, group_pitches, pitch) in enumerate(zip(groups, pitches, lowest)):
        feature: dict[str, Any] = {
            "piece": f"{row['composer']} — {row['title']}",
            "composer": row["composer"],
            "title": row["title"],
            "piece_id": row["piece_id"],
            "performance": Path(row["performance_path"]).name,
            "performance_path": row["performance_path"],
            "onset_group_index": index,
            "onset_time_sec": times[index],
            "group_last_attack_time_sec": float(group[-1]["time_sec"]),
            "group_width_ms": 1000.0 * (float(group[-1]["time_sec"]) - times[index]),
            "lowest_pitch": pitch,
            "lowest_note_name": midi_note_name(pitch),
            "group_pitches": " ".join(str(value) for value in group_pitches),
            "group_note_names": " ".join(midi_note_name(value) for value in group_pitches),
            "group_note_count": len(group_pitches),
            "local_median_ioi": local[index][0],
            "local_positive_ioi_count": local[index][1],
        }
        for cutoff in CUTOFFS:
            prefix = f"m{cutoff}"
            next_index = successors[cutoff][index]
            if next_index is None:
                feature[f"{prefix}_next_low_group_index"] = None
                feature[f"{prefix}_next_low_gap_sec"] = None
                feature[f"{prefix}_intervening_group_count"] = None
                feature[f"{prefix}_R_B"] = None
            else:
                gap = times[next_index] - times[index]
                feature[f"{prefix}_next_low_group_index"] = next_index
                feature[f"{prefix}_next_low_gap_sec"] = gap
                feature[f"{prefix}_intervening_group_count"] = next_index - index - 1
                feature[f"{prefix}_R_B"] = None if local[index][0] is None else gap / float(local[index][0])
        output.append(feature)

    exact_onset_count = len({(attack["time_sec"], attack["tick"]) for attack in attacks})
    metadata = {
        "piece_id": row["piece_id"],
        "performance": Path(row["performance_path"]).name,
        "attack_count": len(attacks),
        "exact_onset_count_from_raw_attacks": exact_onset_count,
        "regroup20ms_onset_count": len(groups),
        "merged_exact_onset_count": exact_onset_count - len(groups),
        "first_onset_sec": times[0],
        "last_onset_sec": times[-1],
        "onset_span_sec": times[-1] - times[0],
        "max_group_width_ms": max(row_["group_width_ms"] for row_ in output),
    }
    return output, metadata


def structural_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    scopes = [(piece, group) for piece, group in frame.groupby("piece_id", sort=False)]
    scopes.append(("POOLED", frame))
    rows: list[dict[str, Any]] = []
    for scope, group in scopes:
        for cutoff in CUTOFFS:
            eligible = group[group.lowest_pitch <= cutoff]
            observed = eligible[eligible[f"m{cutoff}_R_B"].notna()]
            for metric, column in (
                ("R_B", f"m{cutoff}_R_B"),
                ("N_B", f"m{cutoff}_intervening_group_count"),
                ("next_low_gap_sec", f"m{cutoff}_next_low_gap_sec"),
            ):
                rows.append({
                    "scope": scope,
                    "pitch_cutoff": cutoff,
                    "metric": metric,
                    "eligible_low_count": len(eligible),
                    "missing_successor_count": len(eligible) - len(observed),
                    **quantiles(observed[column]),
                })
    return pd.DataFrame(rows)


def exact_vs_regroup_comparison(new: pd.DataFrame, old_path: Path) -> pd.DataFrame:
    old = pd.read_csv(old_path)
    expected = {value[2] for value in TARGETS}
    if set(old.piece_id.unique()) != expected:
        raise RuntimeError("old exact-onset feature table is not the same five-piece audit")
    scopes = [(piece, old[old.piece_id == piece], new[new.piece_id == piece]) for piece in new.piece_id.drop_duplicates()]
    scopes.append(("POOLED", old, new))
    rows: list[dict[str, Any]] = []
    for scope, old_group, new_group in scopes:
        old_local = float(old_group.local_median_ioi_sec.median())
        new_local = float(new_group.local_median_ioi.median())
        for cutoff in CUTOFFS:
            old_low = old_group[old_group.lowest_pitch <= cutoff]
            new_low = new_group[new_group.lowest_pitch <= cutoff]
            old_r = quantiles(old_low[f"m{cutoff}_R_local_ioi"])
            new_r = quantiles(new_low[f"m{cutoff}_R_B"])
            old_n = quantiles(old_low[f"m{cutoff}_intervening_onsets_N"])
            new_n = quantiles(new_low[f"m{cutoff}_intervening_group_count"])
            rows.append({
                "scope": scope,
                "pitch_cutoff": cutoff,
                "old_exact_onset_group_count": len(old_group),
                "new_regroup20ms_group_count": len(new_group),
                "onset_group_count_change": len(new_group) - len(old_group),
                "onset_group_count_change_pct": pct_change(float(len(old_group)), float(len(new_group))),
                "old_local_ioi_median_sec": old_local,
                "new_local_ioi_median_sec": new_local,
                "local_ioi_median_change_pct": pct_change(old_local, new_local),
                "old_R_count": old_r["count"],
                "new_R_count": new_r["count"],
                "old_R_q25": old_r["q25"],
                "new_R_q25": new_r["q25"],
                "old_R_median": old_r["median"],
                "new_R_median": new_r["median"],
                "old_R_q75": old_r["q75"],
                "new_R_q75": new_r["q75"],
                "R_q75_change_pct": pct_change(float(old_r["q75"]), float(new_r["q75"])),
                "old_R_q95": old_r["q95"],
                "new_R_q95": new_r["q95"],
                "R_q95_change_pct": pct_change(float(old_r["q95"]), float(new_r["q95"])),
                "old_N_q25": old_n["q25"],
                "new_N_q25": new_n["q25"],
                "old_N_median": old_n["median"],
                "new_N_median": new_n["median"],
                "old_N_q75": old_n["q75"],
                "new_N_q75": new_n["q75"],
            })
    return pd.DataFrame(rows)


def pitch_band(pitch: int) -> str:
    if pitch <= 52:
        return "p_le_52"
    if pitch <= 55:
        return "p_53_55"
    return "p_56_57"


def assign_rank_strata(group: pd.DataFrame) -> pd.DataFrame:
    ordered = group.sort_values([f"m{MANUAL_REFERENCE_CUTOFF}_R_B", "onset_group_index"], kind="mergesort").copy()
    ranks = np.arange(len(ordered), dtype=int)
    ordered["R_stratum_index"] = np.minimum(R_STRATA - 1, ranks * R_STRATA // len(ordered))
    labels = ("very_small", "small_moderate", "intermediate", "moderately_large", "very_large")
    ordered["R_stratum"] = ordered.R_stratum_index.map(dict(enumerate(labels)))
    ordered["pitch_band"] = ordered.lowest_pitch.map(lambda value: pitch_band(int(value)))
    return ordered


def balanced_candidates(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    r_col = f"m{MANUAL_REFERENCE_CUTOFF}_R_B"
    n_col = f"m{MANUAL_REFERENCE_CUTOFF}_intervening_group_count"
    gap_col = f"m{MANUAL_REFERENCE_CUTOFF}_next_low_gap_sec"
    next_col = f"m{MANUAL_REFERENCE_CUTOFF}_next_low_group_index"
    target_by_band = {"p_le_52": 7, "p_53_55": 7, "p_56_57": 6}

    for piece_id, piece in frame.groupby("piece_id", sort=False):
        piece = piece.sort_values("onset_group_index").copy()
        eligible = piece[(piece.lowest_pitch <= MANUAL_REFERENCE_CUTOFF) & piece[r_col].notna()].copy()
        if len(eligible) < MANUAL_PER_PIECE:
            raise RuntimeError(f"too few manual candidates for {piece_id}: {len(eligible)}")
        eligible = assign_rank_strata(eligible)
        selected_indices: list[int] = []
        selected_band_counts: Counter[str] = Counter()
        bounds: dict[int, tuple[float, float, int]] = {}

        for stratum_index in range(R_STRATA):
            subset = eligible[eligible.R_stratum_index == stratum_index].copy()
            if len(subset) < MANUAL_PER_PIECE // R_STRATA:
                raise RuntimeError(f"R stratum too small for {piece_id}: {stratum_index}")
            bounds[stratum_index] = (float(subset[r_col].min()), float(subset[r_col].max()), len(subset))
            remaining = subset.copy()
            for slot in range(MANUAL_PER_PIECE // R_STRATA):
                target_fraction = (slot + 0.5) / (MANUAL_PER_PIECE // R_STRATA)
                onset_min = float(subset.onset_group_index.min())
                onset_max = float(subset.onset_group_index.max())
                target_onset = onset_min + target_fraction * (onset_max - onset_min)
                span = max(1.0, onset_max - onset_min)

                def candidate_key(record: pd.Series) -> tuple[float, float, int]:
                    band = str(record.pitch_band)
                    deficit = target_by_band[band] - selected_band_counts[band]
                    time_distance = abs(float(record.onset_group_index) - target_onset) / span
                    return (-float(deficit), time_distance, int(record.onset_group_index))

                chosen_position = min(range(len(remaining)), key=lambda pos: candidate_key(remaining.iloc[pos]))
                chosen = remaining.iloc[chosen_position]
                onset_index = int(chosen.onset_group_index)
                selected_indices.append(onset_index)
                selected_band_counts[str(chosen.pitch_band)] += 1
                remaining = remaining[remaining.onset_group_index != onset_index]

        if len(set(selected_indices)) != MANUAL_PER_PIECE:
            raise AssertionError(f"manual selection is not {MANUAL_PER_PIECE} unique rows for {piece_id}")
        indexed = piece.set_index("onset_group_index", drop=False)
        eligible_indexed = eligible.set_index("onset_group_index")
        for onset_index in sorted(selected_indices):
            source = indexed.loc[onset_index]
            sampling = eligible_indexed.loc[onset_index]
            previous: list[str] = []
            following: list[str] = []
            for context_index in range(max(0, onset_index - 4), onset_index):
                context = indexed.loc[context_index]
                previous.append(
                    f"{context_index}@{float(context.onset_time_sec)-float(source.onset_time_sec):+.3f}s:"
                    f"{context.group_pitches}({context.group_note_names})"
                )
            for context_index in range(onset_index + 1, min(len(piece), onset_index + 9)):
                context = indexed.loc[context_index]
                following.append(
                    f"{context_index}@{float(context.onset_time_sec)-float(source.onset_time_sec):+.3f}s:"
                    f"{context.group_pitches}({context.group_note_names})"
                )
            stratum_index = int(sampling.R_stratum_index)
            lower, upper, population = bounds[stratum_index]
            rows.append({
                "piece": source.piece,
                "piece_id": piece_id,
                "performance": source.performance,
                "performance_path": source.performance_path,
                "onset_group_index": onset_index,
                "onset_time_sec": source.onset_time_sec,
                "lowest_pitch": int(source.lowest_pitch),
                "lowest_note_name": source.lowest_note_name,
                "group_pitches": source.group_pitches,
                "group_note_names": source.group_note_names,
                "reference_pitch_cutoff": MANUAL_REFERENCE_CUTOFF,
                "R_B": float(source[r_col]),
                "N_B": int(source[n_col]),
                "next_low_gap_sec": float(source[gap_col]),
                "next_low_group_index": int(source[next_col]),
                "local_median_ioi": float(source.local_median_ioi),
                "R_stratum": sampling.R_stratum,
                "R_stratum_empirical_min": lower,
                "R_stratum_empirical_max": upper,
                "R_stratum_population": population,
                "pitch_band": sampling.pitch_band,
                "previous_4_grouped_onsets": " | ".join(previous),
                "following_8_grouped_onsets": " | ".join(following),
                "annotation": "",
                "comment": "",
            })

        selected_piece = pd.DataFrame([row for row in rows if row["piece_id"] == piece_id])
        for stratum, count in selected_piece.R_stratum.value_counts().sort_index().items():
            summary.append({"piece_id": piece_id, "dimension": "R_stratum", "category": stratum, "count": int(count)})
        for band, count in selected_piece.pitch_band.value_counts().sort_index().items():
            summary.append({"piece_id": piece_id, "dimension": "pitch_band", "category": band, "count": int(count)})

    manual = pd.DataFrame(rows)
    if len(manual) != 5 * MANUAL_PER_PIECE or manual.annotation.astype(str).ne("").any():
        raise AssertionError("manual candidate count or empty-annotation invariant failed")
    if manual.groupby(["piece_id", "R_stratum"]).size().ne(4).any():
        raise AssertionError("each piece must contribute four candidates to every R stratum")
    return manual, pd.DataFrame(summary)


def r_distribution_plot(frame: pd.DataFrame, output: Path) -> None:
    pieces = list(frame.piece_id.drop_duplicates())
    scopes = [(piece, frame[frame.piece_id == piece]) for piece in pieces] + [("POOLED", frame)]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True, sharey=True)
    colors = {52: "#3465a4", 55: "#75507b", 57: "#cc0000"}
    for ax, (scope, group) in zip(axes.flat, scopes):
        for cutoff in CUTOFFS:
            values = np.sort(group.loc[group.lowest_pitch <= cutoff, f"m{cutoff}_R_B"].dropna().to_numpy(dtype=float))
            y = np.arange(1, len(values) + 1) / len(values)
            ax.plot(values, y, label=f"m_cut={cutoff}", color=colors[cutoff], linewidth=1.2)
        title = "Pooled" if scope == "POOLED" else str(group.piece.iloc[0])
        ax.set_title(title)
        ax.set_xscale("log")
        ax.grid(alpha=0.2)
    for ax in axes[-1, :]:
        ax.set_xlabel("R_B (log scale)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Empirical CDF")
    axes.flat[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("20 ms Regrouped Structural-Timescale Distributions")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def manual_timeline_plots(frame: pd.DataFrame, manual: pd.DataFrame, output: Path) -> list[str]:
    paths: list[str] = []
    for piece_id, candidates in manual.groupby("piece_id", sort=False):
        piece = frame[frame.piece_id == piece_id].set_index("onset_group_index")
        candidates = candidates.sort_values(["R_stratum", "R_B"], kind="mergesort")
        fig, axes = plt.subplots(5, 4, figsize=(18, 18), sharey=True)
        for ax, (_, candidate) in zip(axes.flat, candidates.iterrows()):
            center = int(candidate.onset_group_index)
            center_time = float(candidate.onset_time_sec)
            for onset_index in range(max(0, center - 4), min(len(piece), center + 9)):
                context = piece.loc[onset_index]
                context_pitches = [int(value) for value in str(context.group_pitches).split()]
                relative_time = float(context.onset_time_sec) - center_time
                color = "#cc0000" if onset_index == center else "#3465a4"
                size = 23 if onset_index == center else 11
                ax.scatter([relative_time] * len(context_pitches), context_pitches, s=size, color=color)
                ax.annotate(str(onset_index - center), (relative_time, min(context_pitches)), fontsize=6, alpha=0.65)
            ax.axvline(0.0, color="#cc0000", alpha=0.4, linewidth=0.9)
            ax.set_title(
                f"g{center} {candidate.lowest_note_name} | {candidate.R_stratum}\n"
                f"R={candidate.R_B:.2f}, N={int(candidate.N_B)}",
                fontsize=8,
            )
            ax.grid(alpha=0.15)
        for ax in axes[-1, :]:
            ax.set_xlabel("Relative time (s); labels = group offset")
        for ax in axes[:, 0]:
            ax.set_ylabel("MIDI pitch")
        name = f"manual_timeline_{piece_id}.png"
        fig.suptitle(f"Manual candidates: {candidates.piece.iloc[0]} (−4/+8 grouped onsets)")
        fig.tight_layout()
        fig.savefig(output / name, dpi=140)
        plt.close(fig)
        paths.append(name)
    return paths


def stability_rows(distribution: pd.DataFrame, comparison: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cutoff in CUTOFFS:
        new_values = distribution[
            (distribution.scope != "POOLED")
            & (distribution.pitch_cutoff == cutoff)
            & (distribution.metric == "R_B")
        ]["median"].to_numpy(dtype=float)
        old_values = comparison[
            (comparison.scope != "POOLED") & (comparison.pitch_cutoff == cutoff)
        ].old_R_median.to_numpy(dtype=float)
        rows.append({
            "pitch_cutoff": cutoff,
            "old_piece_median_min": float(old_values.min()),
            "old_piece_median_max": float(old_values.max()),
            "old_piece_median_cv": float(np.std(old_values) / np.mean(old_values)),
            "new_piece_median_min": float(new_values.min()),
            "new_piece_median_max": float(new_values.max()),
            "new_piece_median_cv": float(np.std(new_values) / np.mean(new_values)),
        })
    return rows


def build_report(
    selected: Sequence[Mapping[str, str]], provenance: Mapping[str, Any], frame: pd.DataFrame,
    metadata: Sequence[Mapping[str, Any]], distribution: pd.DataFrame, comparison: pd.DataFrame,
    manual: pd.DataFrame, sampling: pd.DataFrame, timeline_paths: Sequence[str], output: Path,
) -> str:
    labels = {row["piece_id"]: f"{row['composer']} — {row['title']}" for row in selected}
    scope_order = [*labels, "POOLED"]
    selection_rows = [
        (row["composer"], row["title"], row["piece_id"], row["performance_path"])
        for row in selected
    ]
    count_rows = []
    for scope in scope_order:
        row = comparison[(comparison.scope == scope) & (comparison.pitch_cutoff == 57)].iloc[0]
        count_rows.append((
            labels.get(scope, "Pooled"), int(row.old_exact_onset_group_count),
            int(row.new_regroup20ms_group_count), fmt(row.onset_group_count_change_pct, 1) + "%",
            fmt(row.old_local_ioi_median_sec, 4), fmt(row.new_local_ioi_median_sec, 4),
            fmt(row.local_ioi_median_change_pct, 1) + "%",
        ))

    change_rows = []
    for scope in scope_order:
        for cutoff in CUTOFFS:
            row = comparison[(comparison.scope == scope) & (comparison.pitch_cutoff == cutoff)].iloc[0]
            change_rows.append((
                labels.get(scope, "Pooled"), cutoff,
                f"{fmt(row.old_R_q25, 2)}→{fmt(row.new_R_q25, 2)}",
                f"{fmt(row.old_R_median, 2)}→{fmt(row.new_R_median, 2)}",
                f"{fmt(row.old_R_q75, 2)}→{fmt(row.new_R_q75, 2)}",
                f"{fmt(row.old_R_q95, 2)}→{fmt(row.new_R_q95, 2)}",
                f"{fmt(row.old_N_q25, 1)}→{fmt(row.new_N_q25, 1)}",
                f"{fmt(row.old_N_median, 1)}→{fmt(row.new_N_median, 1)}",
                f"{fmt(row.old_N_q75, 1)}→{fmt(row.new_N_q75, 1)}",
            ))

    new_rows = []
    for scope in scope_order:
        subset = distribution[distribution.scope == scope]
        for cutoff in CUTOFFS:
            current = subset[subset.pitch_cutoff == cutoff].set_index("metric")
            new_rows.append((
                labels.get(scope, "Pooled"), cutoff,
                int(current.loc["R_B", "eligible_low_count"]),
                int(current.loc["R_B", "missing_successor_count"]),
                fmt(current.loc["R_B", "median"], 2),
                f"{fmt(current.loc['R_B', 'q25'], 2)}–{fmt(current.loc['R_B', 'q75'], 2)}",
                fmt(current.loc["R_B", "q95"], 2),
                fmt(current.loc["N_B", "median"], 1),
                f"{fmt(current.loc['N_B', 'q25'], 1)}–{fmt(current.loc['N_B', 'q75'], 1)}",
            ))

    ravel = comparison[comparison.scope == "piece_de3f82957f1b3532"]
    ravel_rows = [
        (
            int(row.pitch_cutoff), fmt(row.old_R_q75, 2), fmt(row.new_R_q75, 2),
            fmt(row.R_q75_change_pct, 1) + "%", fmt(row.old_R_q95, 2),
            fmt(row.new_R_q95, 2), fmt(row.R_q95_change_pct, 1) + "%",
        )
        for _, row in ravel.iterrows()
    ]
    pooled = comparison[comparison.scope == "POOLED"]
    pooled_tail_reduced = all(
        float(row.new_R_q75) < float(row.old_R_q75) and float(row.new_R_q95) < float(row.old_R_q95)
        for _, row in pooled.iterrows()
    )
    ravel_tail_reduced = all(
        float(row.new_R_q75) < float(row.old_R_q75) and float(row.new_R_q95) < float(row.old_R_q95)
        for _, row in ravel.iterrows()
    )

    stability = stability_rows(distribution, comparison)
    stability_table = [
        (
            row["pitch_cutoff"], f"{fmt(row['old_piece_median_min'], 2)}–{fmt(row['old_piece_median_max'], 2)}",
            fmt(row["old_piece_median_cv"], 3),
            f"{fmt(row['new_piece_median_min'], 2)}–{fmt(row['new_piece_median_max'], 2)}",
            fmt(row["new_piece_median_cv"], 3),
        )
        for row in stability
    ]
    stability_maintained = all(row["new_piece_median_cv"] <= row["old_piece_median_cv"] for row in stability)

    pitch_rows = []
    for piece_id in labels:
        group = manual[manual.piece_id == piece_id]
        pitch_rows.append((
            labels[piece_id], len(group),
            int((group.pitch_band == "p_le_52").sum()),
            int((group.pitch_band == "p_53_55").sum()),
            int((group.pitch_band == "p_56_57").sum()),
            "/".join(str(int(value)) for value in group.R_stratum.value_counts().reindex(
                ["very_small", "small_moderate", "intermediate", "moderately_large", "very_large"]
            ).fillna(0)),
        ))
    all_pitch_bands_present = manual.groupby("piece_id").pitch_band.nunique().eq(3).all()
    candidate_diverse = len(manual) == 100 and all_pitch_bands_present and manual.groupby(["piece_id", "R_stratum"]).size().eq(4).all()
    annotations_empty = manual.annotation.astype(str).eq("").all()

    report = f"""# Structural Bass Event v0 — 20 ms Regrouping Audit

## Scope and safeguards

This small audit recalculates the existing five-performance validation audit with earliest-anchor 20 ms musical-onset grouping and creates an unlabelled manual-inspection set. It does not replace or expand the original audit. No ASAP test MIDI was accessed, and no additional validation performance was added.

- Canonical split manifest: `{provenance['split_manifest']}`
- Manifest SHA-256: `{provenance['split_manifest_sha256']}`
- Inputs opened: exactly {provenance['selected_performance_count']} specified validation performances
- ASAP test MIDI access count: **{provenance['asap_test_midi_access_count']}**
- Existing exact-onset feature source: `{OLD_OUTPUT / 'onset_features.csv'}`
- Existing exact-onset feature SHA-256: `{provenance['old_onset_features_sha256']}`
- Tested command: `{provenance['tested_command']}`

{markdown_table(('Composer', 'Title', 'piece ID', 'performance'), selection_rows)}

## Method

Positive-velocity, non-drum note-on attacks are sorted by `(tempo-aware onset time, track index, message index)`, where track/message position is the MIDI-file order tie-break. Starting at the earliest unassigned attack, every still-unassigned attack in the inclusive interval `[anchor, anchor + 20 ms]` enters the group. The next group starts at the next unassigned attack. This is earliest-anchor grouping, not single-linkage chaining; a built-in assertion verifies that 0/15/30 ms becomes `{{0,15}}`, `{{30}}`.

Each group's representative time is its earliest attack and `p_n` is the minimum newly attacked pitch. The local timescale is the median of positive grouped-onset IOIs spanning up to eight groups on either side. For each cutoff 52/55/57, the next later grouped onset with `p_k <= cutoff` supplies gap, successor group index, strictly intervening group count `N_B`, and `R_B = gap / local median IOI`. A missing terminal successor remains blank/NA.

CC64, key-off, duration, velocity values, score annotation, voice separation, and octave-derived features are excluded from every computed feature and sampling decision.

## Exact onset versus 20 ms grouping

{markdown_table(('Scope', 'old groups', 'new groups', 'group Δ%', 'old local IOI med s', 'new local IOI med s', 'IOI Δ%'), count_rows)}

Each `old→new` cell below uses low-current onsets for the indicated cutoff. Complete numeric values and percentage changes are in `exact_vs_regroup20ms_comparison.csv`.

{markdown_table(('Scope', 'cut', 'R Q1', 'R median', 'R Q3', 'R P95', 'N Q1', 'N median', 'N Q3'), change_rows)}

### Ravel tail check

{markdown_table(('cut', 'old R Q3', 'new R Q3', 'Q3 Δ%', 'old R P95', 'new R P95', 'P95 Δ%'), ravel_rows)}

## New structural-timescale distributions

{markdown_table(('Scope', 'cut', 'low n', 'missing', 'R med', 'R Q1–Q3', 'R P95', 'N med', 'N Q1–Q3'), new_rows)}

![R distributions](R_B_distribution.png)

## Manual annotation candidate set

`manual_annotation_candidates.csv` contains {len(manual)} rows: exactly {MANUAL_PER_PIECE} per piece. Eligibility is `p_n <= 57` with an observed successor and local IOI. Within each piece, eligible observations are stable-sorted by empirical `R_B` and rank-partitioned into five equal-frequency strata. Four examples are selected from each stratum while balancing the three requested pitch bands and temporal position. Stratum names describe relative empirical rank only; they are not detector labels or proposed thresholds.

{markdown_table(('Piece', 'total', 'p≤52', 'p53–55', 'p56–57', 'R strata counts (low→high)'), pitch_rows)}

The `annotation` and `comment` columns are blank. Allowed human labels are:

- `STRUCTURAL_BASS`
- `NOT_STRUCTURAL_BASS`
- `AMBIGUOUS`

Timeline figures show −4/+8 grouped-onset context, use relative seconds on x, MIDI pitch on y, mark the candidate in red, and label each point column with relative group index:

""" + "\n".join(f"- [{name}]({name})" for name in timeline_paths) + f"""

## Required questions

### Q1 — Did the extreme R tail shrink?

**Pooled: {'yes' if pooled_tail_reduced else 'not uniformly'}. Ravel: {'yes' if ravel_tail_reduced else 'not uniformly'}.** This judgment requires both Q3 and P95 to decrease at all three audited cutoffs; the exact magnitudes are reported above. The grouping result is therefore measured rather than assumed.

### Q2 — Is relative stability across the five pieces maintained?

**{'Yes descriptively: dispersion decreased at every audited cutoff.' if stability_maintained else 'Not uniformly: dispersion did not decrease at every audited cutoff.'}** The comparison uses the coefficient of variation of the five piece-level R medians; it is an audit diagnostic, not a pass/fail threshold for a detector.

{markdown_table(('cut', 'old median range', 'old CV', 'new median range', 'new CV'), stability_table)}

This is a descriptive five-piece stability check, not evidence of dataset-wide invariance.

### Q3 — How do cutoffs 52/55/57 differ?

The pooled and piece-level table above gives the direct answer. Raising the cutoff changes both the eligible pitch population and which onset counts as the successor; it must not be read as a nested re-labelling of one fixed set. No cutoff is selected here.

### Q4 — Is candidate sampling sufficiently diverse?

**{'Yes for this annotation pilot.' if candidate_diverse else 'No; a diversity invariant failed.'}** Every piece contributes 4 examples from each empirical R stratum{' and all three pitch bands' if all_pitch_bands_present else ', but at least one pitch band is absent'}. The exact empirical R bounds are stored per row, and `candidate_sampling_summary.csv` provides auditable counts.

### Q5 — Is the audit ready for labels and later R_low/R_high estimation?

**{'Yes, for the requested next human-annotation step.' if candidate_diverse and annotations_empty else 'No; a candidate or empty-label invariant failed.'}** The sheet has balanced piece/R coverage, pitch/context fields, blank labels, and annotation-support plots. This means data preparation is ready; it does not mean either boundary is identifiable before labels are filled and reviewed.

## Non-goals and non-conclusion

This audit does not set `R_low` or `R_high`, calculate Bass Connectivity, evaluate pedal sustain, train a classifier, fit isotonic regression, calculate precision/recall, access ASAP test, or expand beyond the five performances.

## Files

- `onset_features_regroup20ms.csv`: every 20 ms grouped onset and cutoff-specific features
- `manual_annotation_candidates.csv`: 100 empty-label manual candidates
- `exact_vs_regroup20ms_comparison.csv`: exact-onset versus regrouped statistics
- `structural_timescale_distribution.csv`: new R/N/gap summaries
- `candidate_sampling_summary.csv`: candidate strata and pitch-band counts
- `selection_manifest.csv`, `audit_metadata.json`: exact inputs and audit provenance
- `R_B_distribution.png` and five `manual_timeline_*.png` figures
"""
    return report


def run(output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output}")
    old_features = OLD_OUTPUT / "onset_features.csv"
    if not old_features.is_file():
        raise FileNotFoundError(f"existing exact-onset audit is required: {old_features}")
    output.mkdir(parents=True, exist_ok=True)
    assert_grouping_semantics()
    selected, provenance = choose_validation_performances()
    provenance["old_onset_features_sha256"] = sha256(old_features)
    provenance["tested_command"] = f"python {Path(__file__).resolve()} --output {output.resolve()}"

    rows: list[dict[str, Any]] = []
    performance_metadata: list[dict[str, Any]] = []
    for selected_row in selected:
        features, metadata = extract_performance(selected_row)
        rows.extend(features)
        performance_metadata.append(metadata)
        print(
            f"parsed {selected_row['composer']} {selected_row['title']}: "
            f"{metadata['exact_onset_count_from_raw_attacks']} exact -> {len(features)} grouped onsets",
            flush=True,
        )
    frame = pd.DataFrame(rows)
    if frame.groupby("piece_id").ngroups != 5 or not len(frame):
        raise AssertionError("feature table must contain exactly five non-empty pieces")
    if any("octave" in column.lower() for column in frame.columns):
        raise AssertionError("octave-derived feature leaked into output")

    old_counts = pd.read_csv(old_features).groupby("piece_id", sort=False).size().to_dict()
    for metadata in performance_metadata:
        piece_id = metadata["piece_id"]
        if metadata["exact_onset_count_from_raw_attacks"] != old_counts[piece_id]:
            raise AssertionError(f"raw exact-onset count does not reproduce old audit for {piece_id}")

    distribution = structural_distribution(frame)
    comparison = exact_vs_regroup_comparison(frame, old_features)
    manual, sampling = balanced_candidates(frame)
    if any("octave" in column.lower() for column in manual.columns):
        raise AssertionError("octave-derived feature leaked into manual candidates")

    selection_fields = (
        "composer", "title", "piece_id", "performance_path", "split", "performance_sha256",
    )
    write_csv(
        output / "selection_manifest.csv",
        ({field: row[field] for field in selection_fields} for row in selected),
        selection_fields,
    )
    frame.to_csv(output / "onset_features_regroup20ms.csv", index=False)
    manual.to_csv(output / "manual_annotation_candidates.csv", index=False)
    comparison.to_csv(output / "exact_vs_regroup20ms_comparison.csv", index=False)
    distribution.to_csv(output / "structural_timescale_distribution.csv", index=False)
    sampling.to_csv(output / "candidate_sampling_summary.csv", index=False)

    r_distribution_plot(frame, output / "R_B_distribution.png")
    timeline_paths = manual_timeline_plots(frame, manual, output)
    metadata = {
        **provenance,
        "group_window_ms": 20.0,
        "group_rule": "inclusive earliest-anchor window; no single-linkage chaining",
        "attack_sort_tiebreak": "track index, then message index",
        "local_group_radius": LOCAL_RADIUS,
        "pitch_cutoffs": list(CUTOFFS),
        "manual_reference_cutoff": MANUAL_REFERENCE_CUTOFF,
        "manual_candidates_per_piece": MANUAL_PER_PIECE,
        "manual_R_strata": R_STRATA,
        "performance_metadata": performance_metadata,
        "feature_exclusions": [
            "CC64", "key-off", "performed duration", "velocity value", "score annotation",
            "voice separation", "octave-derived information",
        ],
    }
    (output / "audit_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    report = build_report(
        selected, provenance, frame, performance_metadata, distribution, comparison,
        manual, sampling, timeline_paths, output,
    )
    (output / "STRUCTURAL_BASS_REGROUP20MS_REPORT.md").write_text(report, encoding="utf-8")
    print(f"wrote {output}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.output)
