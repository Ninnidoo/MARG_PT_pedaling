#!/usr/bin/env python3
"""Plot the CC64 value histogram of one canonical PT inference MIDI."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import mido


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--midi", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--piece-label", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    values = [
        message.value
        for message in mido.MidiFile(args.midi)
        if message.type == "control_change" and message.control == 64
    ]
    if not values:
        raise RuntimeError(f"No sustain-pedal CC64 events found: {args.midi}")

    counts = Counter(values)
    total = len(values)
    intermediate = sum(counts[value] for value in range(1, 127))
    unique_values = sorted(counts)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.output_dir / "pt_cc64_value_counts.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cc64_value", "event_count", "event_ratio"])
        for value in range(128):
            writer.writerow([value, counts[value], counts[value] / total])

    metadata = {
        "source_midi": str(args.midi),
        "piece_label": args.piece_label,
        "total_cc64_events": total,
        "zero_count": counts[0],
        "zero_ratio": counts[0] / total,
        "intermediate_1_126_count": intermediate,
        "intermediate_1_126_ratio": intermediate / total,
        "full_127_count": counts[127],
        "full_127_ratio": counts[127] / total,
        "unique_cc64_values": unique_values,
    }
    metadata_path = args.output_dir / "pt_cc64_endpoint_collapse_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 14,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#28323C",
            "axes.labelcolor": "#28323C",
            "xtick.color": "#4B5563",
            "ytick.color": "#4B5563",
        }
    )
    fig, ax = plt.subplots(figsize=(16, 9), dpi=120)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x = list(range(128))
    bar_colors = ["#D9DEE5"] * 128
    bar_colors[0] = "#2176AE"
    bar_colors[127] = "#E45756"
    ax.bar(x, [counts[value] for value in x], width=1.0, color=bar_colors, linewidth=0)
    ax.axvspan(0.5, 126.5, color="#F3F4F6", alpha=0.75, zorder=-1)

    ax.set_xlim(-3.5, 130.5)
    ax.set_ylim(0, max(counts.values()) * 1.34)
    ax.set_xticks([0, 16, 32, 48, 64, 80, 96, 112, 127])
    ax.set_xlabel("Predicted sustain-pedal value (MIDI CC64)", labelpad=14)
    ax.set_ylabel("Number of CC64 events", labelpad=14)
    ax.grid(axis="y", color="#D9DEE5", linewidth=0.9, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_title(
        "Pianist Transformer predicts only pedal OFF or FULL",
        fontsize=26,
        loc="left",
        pad=30,
        color="#17202A",
    )
    ax.text(
        0,
        1.015,
        f"{args.piece_label}  |  canonical PT inference: {args.midi.name}",
        transform=ax.transAxes,
        fontsize=14,
        color="#5B6573",
        va="bottom",
    )

    for value, label, color, alignment in [
        (0, "OFF", "#2176AE", "left"),
        (127, "FULL", "#E45756", "right"),
    ]:
        count = counts[value]
        ax.text(
            value,
            count + max(counts.values()) * 0.035,
            f"{label}\n{count} events ({count / total:.1%})",
            ha=alignment,
            va="bottom",
            fontsize=15,
            fontweight="bold",
            color=color,
        )

    ax.text(
        63.5,
        max(counts.values()) * 0.44,
        "1–126: 0 events (0.0%)\nNo intermediate pedal-depth prediction",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color="#6B7280",
        bbox={
            "boxstyle": "round,pad=0.65",
            "facecolor": "white",
            "edgecolor": "#CBD1D8",
            "linewidth": 1.3,
        },
    )
    ax.text(
        1.0,
        -0.17,
        f"N = {total} CC64 events  •  observed values = {unique_values}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=12.5,
        color="#6B7280",
    )

    fig.subplots_adjust(left=0.09, right=0.97, top=0.82, bottom=0.18)
    stem = args.output_dir / "pt_cc64_endpoint_collapse_128_bins"
    fig.savefig(stem.with_suffix(".png"), dpi=180, facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)

    print(json.dumps(metadata, indent=2))
    print(f"Wrote: {stem.with_suffix('.png')}")
    print(f"Wrote: {stem.with_suffix('.svg')}")
    print(f"Wrote: {stem.with_suffix('.pdf')}")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {metadata_path}")


if __name__ == "__main__":
    main()
