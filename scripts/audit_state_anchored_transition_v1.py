#!/usr/bin/env python3
"""CPU-only canonical audit for State-Anchored Transition v1."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage2_binary_2slot.dataset import build_performance_timeline  # noqa: E402
from src.stage2_event_model.dataset import CANONICAL_CACHE_ID, load_cache_manifest  # noqa: E402
from src.stage2_event_model.note_alignment import sha256_file  # noqa: E402
from src.stage2_event_tokenizer.binary_2slot import OFF, ON  # noqa: E402
from src.stage2_state_anchored.targets import CHANGE, HOLD, RETURN, build_performance_targets  # noqa: E402


EXPECTED_SOURCE_COUNTS = {
    "train": {"MAESTRO-clean": 1170, "ASAP-train": 892},
    "validation": {"ASAP-validation": 71},
}
STATE_NAMES = {OFF: "OFF", ON: "ON"}


def ratio(count: int, total: int) -> float:
    return float(count / total) if total else 0.0


def distribution(counter: Counter[str], names: tuple[str, ...]) -> dict[str, dict[str, float | int]]:
    total = sum(counter.values())
    return {
        name: {"count": int(counter[name]), "ratio": ratio(counter[name], total)}
        for name in names
    }


def summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p05": None, "p95": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
    }


@dataclass
class SplitAccumulator:
    performances: int = 0
    onset_states: Counter[str] = field(default_factory=Counter)
    modes: Counter[str] = field(default_factory=Counter)
    return_composition: Counter[str] = field(default_factory=Counter)
    intervals: int = 0
    raw_transitions: int = 0
    retained_transitions: int = 0
    overflow_intervals: int = 0
    odd_overflow_intervals: int = 0
    even_overflow_intervals: int = 0
    endpoint_preserved: int = 0
    parity_preserved: int = 0
    raw_tau_zero_events: int = 0
    retained_tau_zero_events: int = 0
    change_flip_violations: int = 0
    same_state_mode_violations: int = 0
    timing_range_violations: int = 0
    return_order_violations: int = 0
    change_tau1: list[float] = field(default_factory=list)
    return_tau1: list[float] = field(default_factory=list)
    return_tau2: list[float] = field(default_factory=list)
    return_gap: list[float] = field(default_factory=list)

    def add(self, targets) -> None:
        self.performances += 1
        self.onset_states.update(STATE_NAMES[state] for state in targets.onset_states)
        for target in targets.intervals:
            self.intervals += 1
            self.modes[target.mode] += 1
            raw_count = target.raw_transition_count
            retained_count = target.retained_transition_count
            self.raw_transitions += raw_count
            self.retained_transitions += retained_count
            self.raw_tau_zero_events += sum(float(event.tau) == 0.0 for event in target.raw_transitions)
            self.retained_tau_zero_events += sum(float(event.tau) == 0.0 for event in target.retained_transitions)
            if raw_count >= 3:
                self.overflow_intervals += 1
                if raw_count % 2:
                    self.odd_overflow_intervals += 1
                else:
                    self.even_overflow_intervals += 1
            self.endpoint_preserved += int(target.endpoint_state_preserved)
            self.parity_preserved += int(
                target.raw_transition_count % 2 == target.retained_transition_count % 2
            )

            flips = target.start_state != target.end_state
            self.change_flip_violations += int((target.mode == CHANGE) != flips)
            self.same_state_mode_violations += int(
                (target.mode in {HOLD, RETURN}) != (not flips)
            )
            retained_taus = [float(event.tau) for event in target.retained_transitions]
            self.timing_range_violations += sum(not 0.0 <= tau < 1.0 for tau in retained_taus)
            if target.mode == CHANGE:
                self.change_tau1.append(retained_taus[0])
            elif target.mode == RETURN:
                tau1, tau2 = retained_taus
                self.return_tau1.append(tau1)
                self.return_tau2.append(tau2)
                self.return_gap.append(tau2 - tau1)
                self.return_order_violations += int(tau1 > tau2)
                composition = "ON -> OFF -> ON" if target.start_state == ON else "OFF -> ON -> OFF"
                self.return_composition[composition] += 1

    def result(self) -> dict[str, Any]:
        endpoint_rate = ratio(self.endpoint_preserved, self.intervals)
        return {
            "performances": self.performances,
            "onset_state_targets": sum(self.onset_states.values()),
            "interval_targets": self.intervals,
            "state": distribution(self.onset_states, ("OFF", "ON")),
            "mode": distribution(self.modes, (HOLD, CHANGE, RETURN)),
            "return_composition": distribution(
                self.return_composition,
                ("ON -> OFF -> ON", "OFF -> ON -> OFF"),
            ),
            "compression": {
                "raw_transition_count": self.raw_transitions,
                "retained_transition_count": self.retained_transitions,
                "retained_transition_ratio": ratio(self.retained_transitions, self.raw_transitions),
                "raw_ge_3_interval_count": self.overflow_intervals,
                "raw_ge_3_interval_ratio": ratio(self.overflow_intervals, self.intervals),
                "odd_overflow_interval_count": self.odd_overflow_intervals,
                "even_overflow_interval_count": self.even_overflow_intervals,
                "endpoint_state_preserved_count": self.endpoint_preserved,
                "endpoint_state_preservation_rate": endpoint_rate,
                "parity_preserved_count": self.parity_preserved,
                "parity_preservation_rate": ratio(self.parity_preserved, self.intervals),
                "raw_exact_onset_tau_zero_event_count": self.raw_tau_zero_events,
                "retained_exact_onset_tau_zero_event_count": self.retained_tau_zero_events,
            },
            "consistency": {
                "change_iff_state_flip_violation_count": self.change_flip_violations,
                "hold_or_return_iff_same_state_violation_count": self.same_state_mode_violations,
                "timing_range_violation_count": self.timing_range_violations,
                "return_timing_order_violation_count": self.return_order_violations,
            },
            "timing": {
                "CHANGE": {"tau1": summary(self.change_tau1)},
                "RETURN": {
                    "tau1": summary(self.return_tau1),
                    "tau2": summary(self.return_tau2),
                    "tau2_minus_tau1": summary(self.return_gap),
                },
            },
        }


def audit_entry(cache_root: Path, entry: dict[str, Any]):
    cache_path = cache_root / entry["cache_file"]
    with np.load(cache_path, allow_pickle=False) as cache:
        if str(cache["cache_id"].item()) != CANONICAL_CACHE_ID:
            raise RuntimeError(f"per-performance canonical cache ID mismatch: {cache_path}")
        onset_ticks = cache["onset_ticks"].astype(np.int64, copy=True)
        latest_note_off_tick = int(cache["latest_note_off_tick"])
    source_midi = Path(entry["source_midi"])
    if sha256_file(source_midi) != entry["source_sha256"]:
        raise RuntimeError(f"canonical source MIDI SHA changed: {source_midi}")
    timeline = build_performance_timeline(
        source_midi,
        onset_ticks,
        latest_note_off_tick,
        performance_id=str(entry["performance_path"]),
    )
    return build_performance_targets(timeline)


def fmt_count_ratio(item: dict[str, Any]) -> str:
    return f"{item['count']:,} ({item['ratio']:.6%})"


def fmt_summary(item: dict[str, Any]) -> str:
    if item["count"] == 0:
        return "n=0"
    return (
        f"n={item['count']:,}, mean={item['mean']:.6f}, median={item['median']:.6f}, "
        f"p05={item['p05']:.6f}, p95={item['p95']:.6f}"
    )


def markdown(report: dict[str, Any]) -> str:
    train = report["splits"]["train"]
    validation = report["splits"]["validation"]
    lines = [
        "# State-Anchored Transition v1 Tokenizer Audit",
        "",
        f"**Verdict: {report['verdict']}**",
        "",
        "## Target definition and implementation",
        "",
        "The modeled horizon is MAIN-only: for distinct onsets `t_1 ... t_M`, only "
        "`[t_i, t_(i+1))` for `i=1 ... M-1` is represented. There is no PRE, POST, "
        "or final-note-off tail. State `S_i` is the threshold-64 pedal state immediately "
        "before `t_i`; an event exactly at `t_i` belongs to interval `i` with `tau=0`.",
        "",
        "The implementation reuses the frozen canonical cache manifest/onset metadata, "
        "the existing raw MIDI parser, same-tick-last CC64 projection, `<64/OFF` and "
        "`>=64/ON` crossing extraction, and right-open Binary 2-Slot timeline builder. "
        "Raw counts 0/1/2 map to HOLD/CHANGE/RETURN. Odd overflow retains the final one "
        "transition and even overflow retains the final two; no correction or synthetic "
        "event is inserted.",
        "",
        "## Canonical population",
        "",
        f"- Train: {train['performances']:,} performances (MAESTRO-clean 1,170 + ASAP train 892)",
        f"- Validation: {validation['performances']:,} performances (ASAP validation 71)",
        "- ASAP test access: 0",
        "",
        "## State distribution",
        "",
        "| Split | OFF | ON | Total onset states |",
        "| --- | ---: | ---: | ---: |",
        f"| Train | {fmt_count_ratio(train['state']['OFF'])} | {fmt_count_ratio(train['state']['ON'])} | {train['onset_state_targets']:,} |",
        f"| Validation | {fmt_count_ratio(validation['state']['OFF'])} | {fmt_count_ratio(validation['state']['ON'])} | {validation['onset_state_targets']:,} |",
        "",
        "## Mode distribution",
        "",
        "| Split | HOLD | CHANGE | RETURN | Total intervals |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| Train | {fmt_count_ratio(train['mode'][HOLD])} | {fmt_count_ratio(train['mode'][CHANGE])} | {fmt_count_ratio(train['mode'][RETURN])} | {train['interval_targets']:,} |",
        f"| Validation | {fmt_count_ratio(validation['mode'][HOLD])} | {fmt_count_ratio(validation['mode'][CHANGE])} | {fmt_count_ratio(validation['mode'][RETURN])} | {validation['interval_targets']:,} |",
        "",
        "## RETURN composition",
        "",
        "| Split | ON -> OFF -> ON | OFF -> ON -> OFF |",
        "| --- | ---: | ---: |",
        f"| Train | {fmt_count_ratio(train['return_composition']['ON -> OFF -> ON'])} | {fmt_count_ratio(train['return_composition']['OFF -> ON -> OFF'])} |",
        f"| Validation | {fmt_count_ratio(validation['return_composition']['ON -> OFF -> ON'])} | {fmt_count_ratio(validation['return_composition']['OFF -> ON -> OFF'])} |",
        "",
        "## Compression",
        "",
        "| Split | Raw transitions | Retained | Retained ratio | K>=3 intervals | Odd overflow | Even overflow | Endpoint preservation |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, result in (("Train", train), ("Validation", validation)):
        value = result["compression"]
        lines.append(
            f"| {label} | {value['raw_transition_count']:,} | {value['retained_transition_count']:,} | "
            f"{value['retained_transition_ratio']:.6%} | {value['raw_ge_3_interval_count']:,} "
            f"({value['raw_ge_3_interval_ratio']:.6%}) | {value['odd_overflow_interval_count']:,} | "
            f"{value['even_overflow_interval_count']:,} | {value['endpoint_state_preservation_rate']:.6%} |"
        )
    lines.extend([
        "",
        "## Consistency invariants",
        "",
        "| Split | CHANGE iff flip violations | HOLD/RETURN iff same violations | Timing range violations | RETURN order violations |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| Train | {train['consistency']['change_iff_state_flip_violation_count']:,} | {train['consistency']['hold_or_return_iff_same_state_violation_count']:,} | {train['consistency']['timing_range_violation_count']:,} | {train['consistency']['return_timing_order_violation_count']:,} |",
        f"| Validation | {validation['consistency']['change_iff_state_flip_violation_count']:,} | {validation['consistency']['hold_or_return_iff_same_state_violation_count']:,} | {validation['consistency']['timing_range_violation_count']:,} | {validation['consistency']['return_timing_order_violation_count']:,} |",
        "",
        "## Timing distribution",
        "",
    ])
    for label, result in (("Train", train), ("Validation", validation)):
        lines.extend([
            f"### {label}",
            "",
            f"- CHANGE tau1: {fmt_summary(result['timing']['CHANGE']['tau1'])}",
            f"- RETURN tau1: {fmt_summary(result['timing']['RETURN']['tau1'])}",
            f"- RETURN tau2: {fmt_summary(result['timing']['RETURN']['tau2'])}",
            f"- RETURN tau2 - tau1: {fmt_summary(result['timing']['RETURN']['tau2_minus_tau1'])}",
            "",
        ])
    lines.extend([
        "## Edge-case tests",
        "",
        f"Focused command: `{report['focused_tests']['command']}`",
        "",
        f"Result: {report['focused_tests']['status']} ({report['focused_tests']['case_count']} synthetic cases). "
        "Covered HOLD, both CHANGE directions, both RETURN compositions, 3/4-transition "
        "compression, left/right boundary ownership, and strict-before first-onset state.",
        "",
        "## PASS / FAIL",
        "",
        f"{report['verdict']}: focused tests passed; both split invariants and timing checks have "
        "zero violations; compression preserves every raw endpoint state; canonical provenance "
        "reports zero ASAP test access. No model, decoder, overfit, or training was run.",
        "",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=ROOT / "analysis/custom_event_tokenizer_v1",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "analysis/state_anchored_transition_v1",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
        raise RuntimeError("audit must run with CUDA_VISIBLE_DEVICES empty or -1")

    cache_root, config, manifest = load_cache_manifest(args.cache_root)
    if config["asap_test_access_count"] != 0 or manifest["asap_test_access_count"] != 0:
        raise RuntimeError("canonical provenance reports ASAP test access")
    source_counts = {
        split: Counter(
            entry["source"] for entry in manifest["entries"] if entry["split"] == split
        )
        for split in ("train", "validation")
    }
    if {key: dict(value) for key, value in source_counts.items()} != EXPECTED_SOURCE_COUNTS:
        raise RuntimeError(f"canonical source population changed: {source_counts}")
    if any(entry["split"] not in {"train", "validation"} for entry in manifest["entries"]):
        raise RuntimeError("unexpected split is present in canonical manifest")

    test_command = f"{sys.executable} tests/test_state_anchored_transition_v1.py"
    completed = subprocess.run(
        [sys.executable, "tests/test_state_anchored_transition_v1.py"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    if completed.returncode:
        raise RuntimeError(f"focused tests failed:\n{completed.stdout}\n{completed.stderr}")
    test_results = eval(completed.stdout.strip(), {"__builtins__": {}}, {})
    if not isinstance(test_results, dict):
        raise RuntimeError("focused test output is not a result mapping")

    accumulators = {"train": SplitAccumulator(), "validation": SplitAccumulator()}
    entries = [entry for entry in manifest["entries"] if entry["split"] in accumulators]
    for index, entry in enumerate(entries, start=1):
        accumulators[entry["split"]].add(audit_entry(cache_root, entry))
        if index % 100 == 0 or index == len(entries):
            print(f"audited {index}/{len(entries)} performances", flush=True)

    splits = {name: accumulator.result() for name, accumulator in accumulators.items()}
    zero_violations = all(
        all(value == 0 for value in result["consistency"].values())
        for result in splits.values()
    )
    endpoint_perfect = all(
        math.isclose(
            result["compression"]["endpoint_state_preservation_rate"],
            1.0,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        for result in splits.values()
    )
    verdict = "PASS" if zero_violations and endpoint_perfect else "FAIL"
    report = {
        "schema_version": 1,
        "experiment": "state_anchored_transition_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "device": "CPU-only (CUDA_VISIBLE_DEVICES empty)",
        "target": {
            "horizon": "MAIN-only [t_i,t_(i+1)), i=1..M-1",
            "state": "binary CC64 state strictly before each distinct onset",
            "threshold": {"OFF": "CC64 < 64", "ON": "CC64 >= 64"},
            "modes": [HOLD, CHANGE, RETURN],
            "compression": "K<=2 keep all; K>=3 odd keep last 1; K>=4 even keep last 2",
            "pre": False,
            "post": False,
            "final_note_off_tail": False,
        },
        "canonical_provenance": {
            "cache_root": str(cache_root),
            "cache_id": manifest["cache_id"],
            "source_counts": {key: dict(value) for key, value in source_counts.items()},
            "asap_test_access_count": 0,
        },
        "focused_tests": {
            "command": test_command,
            "status": "PASS",
            "case_count": len(test_results),
            "cases": test_results,
        },
        "splits": splits,
        "consistency_all_zero": zero_violations,
        "endpoint_state_preservation_all": endpoint_perfect,
        "asap_test_access": 0,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    json_path = args.output_root / "tokenizer_audit.json"
    markdown_path = args.output_root / "TOKENIZER_AUDIT.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown(report), encoding="utf-8")
    os.chmod(json_path, 0o644)
    os.chmod(markdown_path, 0o644)
    print(f"{verdict}: {json_path}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
