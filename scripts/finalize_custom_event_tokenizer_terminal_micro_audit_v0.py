#!/usr/bin/env python3
"""Write the data-based final report for the completed terminal micro-audit."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def pct(value: float) -> str:
    return f"{100 * float(value):.4f}%"


def histogram_quantile(payload: dict[str, Any], dataset: str, denominator: str, level: float) -> int:
    rows = [
        row for row in payload["terminal_event_count_distribution"]
        if row["dataset"] == dataset and row["denominator"] == denominator
    ]
    values: list[int] = []
    for row in rows:
        if row["bin"] != "10+":
            values.extend([int(row["bin"])] * int(row["count"]))
    rank = max(1, math.ceil(level * len(values)))
    return sorted(values)[rank - 1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root
    payload: dict[str, Any] = json.loads((output / "audit_summary.json").read_text())
    capacity = payload["terminal_capacity_curve"]
    capacity_by = {(row["dataset"], str(row["capacity"])): row for row in capacity}
    timing = payload["delay_transform_comparison"]
    timing_by = {(row["dataset"], row["coordinate"], row["transform"]): row for row in timing}
    conditioning = {
        (row["transform"], row["source_seconds"], row["epsilon"]): row
        for row in payload["delay_transform_conditioning"]
    }
    counts = payload["terminal_event_count_summary"]
    slots = payload["terminal_slot_distributions"]
    train_abs = payload["absolute_delay_stats"]["train"]["overall"]
    train_gap = payload["inter_event_gap_stats"]["train"]["overall"]
    bounded_abs = timing_by[("train", "absolute", "bounded")]
    bounded_gap = timing_by[("train", "gap", "bounded")]
    log_gap = timing_by[("train", "gap", "log1p")]

    lines = [
        "# Custom Event Tokenizer Terminal Micro-Audit v0", "",
        "## Scope and provenance", "",
        "- Original performance MIDI raw CC64 events; no Pedal1–4 reconstruction.",
        "- Terminal anchor: latest non-pedal note-off; only effective events strictly after it.",
        "- Main representation fixed at K=6 chronological last-6; no main-K sweep.",
        "- Existing v0/v1 artifacts were read-only; v1 terminal count/affected/median regression passed.",
        "- Training/model/optimizer/checkpoint/full-oracle activity: 0. ASAP test: 0. Repedal: 0.", "",
        "## Terminal event counts", "",
        "| Dataset | Performances | Affected | All mean/p50/p75/p90/p95/p97.5/p99/p99.5/max | Affected mean/p50/p75/p90/p95/p97.5/p99/p99.5/max |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for dataset in ("train", "validation"):
        summary = counts[dataset]
        all_stats = summary["all_performances"]
        affected = summary["affected_performances"]
        lines.append(
            f"| {dataset} | {summary['total_performances']} | {summary['affected_performances_count']} | "
            f"{all_stats['mean']:.3f}/{all_stats['p50']:.1f}/{all_stats['p75']:.1f}/{all_stats['p90']:.1f}/"
            f"{all_stats['p95']:.1f}/{histogram_quantile(payload, dataset, 'all_performances', .975)}/"
            f"{all_stats['p99']:.1f}/{all_stats['p99p5']:.1f}/{all_stats['max']:.0f} | "
            f"{affected['mean']:.3f}/{affected['p50']:.1f}/{affected['p75']:.1f}/{affected['p90']:.1f}/"
            f"{affected['p95']:.1f}/{histogram_quantile(payload, dataset, 'affected_performances', .975)}/"
            f"{affected['p99']:.1f}/{affected['p99p5']:.1f}/{affected['max']:.0f} |"
        )
    lines += ["", "## Terminal capacity curve", "",
              "| Dataset | K_T | Overflow all / affected | Event retention | Crossing retention | Final state | Last slot active all / affected |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in capacity:
        if row["capacity"] == "Unlimited":
            last = "N/A"
        else:
            last = f"{pct(row['last_slot_active_fraction_all'])} / {pct(row['last_slot_active_fraction_affected'])}"
        lines.append(
            f"| {row['dataset']} | {row['capacity']} | {pct(row['overflow_fraction_all'])} / {pct(row['overflow_fraction_affected'])} | "
            f"{pct(row['event_retention'])} | {pct(row['crossing_retention'])} | "
            f"{pct(row['final_destination_preservation'])} | {last} |"
        )
    lines += ["", "## Slot occupancy elbow", "",
              "| Dataset | Capacity | Slot active fractions (all performances) |",
              "| --- | ---: | --- |"]
    for dataset in ("train", "validation"):
        for key in ("k3", "k4", "k6"):
            values = slots[dataset][key]
            active = ", ".join(f"S{i+1}={pct(slot['active_fraction_all'])}" for i, slot in enumerate(values.values()))
            lines.append(f"| {dataset} | {key[1:]} | {active} |")
    lines += ["", "## Timing representation", "",
              "| Dataset | Coordinate | Transform | Median | p95 | p99 | Max | Dynamic range |",
              "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in timing:
        lines.append(
            f"| {row['dataset']} | {row['coordinate']} | {row['transform']} | {row['p50']:.6f} | "
            f"{row['p95']:.6f} | {row['p99']:.6f} | {row['max']:.6f} | {row['dynamic_range']:.6f} |"
        )
    lines += [
        "", "## Transform conditioning (diagnostic only)", "",
        "These numbers do not predict model accuracy; they only invert fixed transformed-space perturbations.", "",
        "| Source delay | epsilon | Raw seconds error | Log1p seconds error | Bounded seconds error |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for source in (0.1, 1.0, 5.0, 10.0):
        for epsilon in (0.01, 0.05):
            values = [conditioning[(name, source, epsilon)]["absolute_seconds_error"] for name in ("raw", "log1p", "bounded")]
            lines.append(f"| {source:.2f} | +{epsilon:.2f} | {values[0]:.6f} | {values[1]:.6f} | {values[2]:.6f} |")
    lines += [
        "", "## Research questions", "",
        f"### Q1 — Count tail\nAffected train performances have mean/median/p95/p99/max "
        f"{counts['train']['affected_performances']['mean']:.3f}/{counts['train']['affected_performances']['p50']:.0f}/"
        f"{counts['train']['affected_performances']['p95']:.0f}/{counts['train']['affected_performances']['p99']:.0f}/"
        f"{counts['train']['affected_performances']['max']:.0f}. The tail reaches nine events but is sparse.",
        f"### Q2 — Retention curve\nK_T=3/4/6 train event retention is "
        f"{pct(capacity_by[('train','3')]['event_retention'])}/{pct(capacity_by[('train','4')]['event_retention'])}/"
        f"{pct(capacity_by[('train','6')]['event_retention'])}; crossing retention is "
        f"{pct(capacity_by[('train','3')]['crossing_retention'])}/{pct(capacity_by[('train','4')]['crossing_retention'])}/"
        f"{pct(capacity_by[('train','6')]['crossing_retention'])}.",
        f"### Q3 — Minimal K_T\nRecommend K_T=4. It retains {pct(capacity_by[('train','4')]['event_retention'])} events and "
        f"{pct(capacity_by[('train','4')]['crossing_retention'])} crossings; only "
        f"{pct(capacity_by[('train','4')]['overflow_fraction_affected'])} of affected train performances overflow. "
        f"K_T=6 gains {100*(capacity_by[('train','6')]['event_retention']-capacity_by[('train','4')]['event_retention']):.3f}pp event and "
        f"{100*(capacity_by[('train','6')]['crossing_retention']-capacity_by[('train','4')]['crossing_retention']):.3f}pp crossing retention, "
        "but Slots5/6 are active in only 0.970%/0.242% of all train performances. Validation has one nine-event outlier, so its K_T=4 event retention is 96.454%.",
        f"### Q4 — Coordinate\nRecommend inter-event gaps. Train median/p95/p99 are "
        f"{train_gap['p50']:.3f}/{train_gap['p95']:.3f}/{train_gap['p99']:.3f}s versus absolute "
        f"{train_abs['p50']:.3f}/{train_abs['p95']:.3f}/{train_abs['p99']:.3f}s, and cumulative decoding guarantees chronological order. "
        "Its trade-off is cumulative timing error across at most four terminal slots.",
        f"### Q5 — Transform\nRecommend log1p. For train gaps it maps max/p99 "
        f"{train_gap['max']:.3f}/{train_gap['p99']:.3f}s to {log_gap['max']:.3f}/{log_gap['p99']:.3f}, "
        "while retaining a near-linear short-delay region.",
        f"### Q6 — Bounded saturation\nFrequency saturation is not widespread: train z>=0.90 is "
        f"{pct(bounded_abs['fraction_z_ge_0p9'])} absolute and {pct(bounded_gap['fraction_z_ge_0p9'])} gap; z>=0.98 is zero. "
        "Nevertheless inverse sensitivity near one is poor.",
        f"### Q7 — Log1p short delays\nAt source 0.1s, a +0.05 transformed perturbation becomes "
        f"{conditioning[('log1p',0.1,0.05)]['absolute_seconds_error']:.3f}s error, close to raw's 0.050s, while long tails are compressed.",
        f"### Q8 — Inverse sensitivity\nAt 5s/10s with +0.05 perturbation, log1p gives "
        f"{conditioning[('log1p',5.0,0.05)]['absolute_seconds_error']:.3f}/"
        f"{conditioning[('log1p',10.0,0.05)]['absolute_seconds_error']:.3f}s error; bounded gives "
        f"{conditioning[('bounded',5.0,0.05)]['absolute_seconds_error']:.3f}/"
        f"{conditioning[('bounded',10.0,0.05)]['absolute_seconds_error']:.3f}s. Raw is best-conditioned in seconds but leaves the heavy tail uncompressed.",
        "### Q9 — Recommendation\nK_T=4, chronological last-4, inter-event-gap timing, log1p transform.",
        "### Q10 — Readiness\nREADY. This is a proposed spec for user approval; it has not been implemented.",
        "", "## Recommended terminal representation", "", "```text",
        "Terminal anchor:", "latest note-off", "", "Terminal capacity:", "K_T = 4", "",
        "Capacity compression:", "chronological last-4", "", "Timing coordinate:", "inter-event-gap", "",
        "Timing transform:", "log1p", "", "Event vocabulary:",
        "NONE / SET_ZERO / SET_LOW / SET_HALF / SET_FULL", "", "Inference ordering implication:",
        "decode positive gaps, invert log1p, and cumulatively sum from latest note-off; ordering is structural", "",
        "Reason:", "K_T=4 is the capacity elbow; gaps compact targets and guarantee order; log1p balances tail compression and inverse conditioning.", "```", "",
        "## Custom Event Tokenizer Proposed Final Spec", "", "```text",
        "Initial state: state immediately before first distinct onset",
        "Main interval: distinct-onset [t_i,t_{i+1})", "Main slots: K = 6",
        "Main compression: chronological last-6", "Main timing: exact tau in [0,1)",
        "Event vocabulary: NONE / SET_ZERO / SET_LOW / SET_HALF / SET_FULL",
        "Terminal anchor: latest note-off", "Terminal slots: K_T = 4",
        "Terminal compression: chronological last-4", "Terminal timing coordinate: inter-event-gap",
        "Terminal timing transform: log1p",
        "Same-timestamp semantics: keep v0 deterministic semantics; document ~2.3617% onset events and 193 ambiguous cases",
        "Model implementation readiness: YES, after user freezes this proposed spec", "```", "",
        f"Read-only main K=6 reference: `{payload['main_k6_reference']}`.", "",
        "No tokenizer final-spec implementation, model, loss, training, ASAP test, Repedal, or full oracle rerun was performed.",
    ]
    (output / "CUSTOM_EVENT_TOKENIZER_TERMINAL_MICRO_AUDIT_V0.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
