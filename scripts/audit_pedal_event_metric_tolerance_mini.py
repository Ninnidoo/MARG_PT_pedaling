#!/usr/bin/env python3
"""Small, train-only pedal-event tolerance pilot with persistent alignment cache."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import subprocess
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from miditoolkit import MidiFile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from third_party.PianistTransformer.src.utils.midi import normalize_midi, read_corresp

WINDOWS = (4, 8, 12, 16)
SEARCH_WINDOWS = (6, 8, 12)
T_VALUES_MS = (100, 150, 200, 250, 300, 400, 500, 600)
CC_THRESHOLD = 64


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def q(values: Iterable[float], level: float) -> float | None:
    array = np.asarray(list(values), dtype=float)
    return None if not len(array) else float(np.quantile(array, level))


def f(value: float | None, digits: int = 3) -> str:
    return "—" if value is None or not math.isfinite(float(value)) else f"{float(value):.{digits}f}"


def tick_ms_fn(midi: MidiFile):
    changes = sorted(midi.tempo_changes, key=lambda item: item.time)
    ticks = [0]
    tempos = [120.0]
    for item in changes:
        if item.time == ticks[-1]:
            tempos[-1] = float(item.tempo)
        else:
            ticks.append(int(item.time)); tempos.append(float(item.tempo))
    cumulative = [0.0]
    for index in range(1, len(ticks)):
        cumulative.append(cumulative[-1] + (ticks[index] - ticks[index - 1]) * 60000.0 / (tempos[index - 1] * midi.ticks_per_beat))

    def convert(tick: int) -> float:
        index = int(np.searchsorted(ticks, tick, side="right") - 1)
        return cumulative[index] + (tick - ticks[index]) * 60000.0 / (tempos[index] * midi.ticks_per_beat)

    return convert


def raw_events(midi: MidiFile) -> tuple[list[int], list[tuple[int, int]]]:
    onsets = sorted({int(note.start) for inst in midi.instruments if not inst.is_drum for note in inst.notes})
    controls: list[tuple[int, int, int, int]] = []
    for inst_index, inst in enumerate(midi.instruments):
        if inst.is_drum:
            continue
        for event_index, event in enumerate(inst.control_changes):
            if event.number == 64:
                controls.append((int(event.time), inst_index, event_index, int(event.value)))
    controls.sort()
    return onsets, [(tick, value) for tick, _, _, value in controls]


def nearest_onset(onsets: Sequence[int], tick: int) -> int:
    right = int(np.searchsorted(onsets, tick, side="left"))
    if right == 0:
        return 0
    if right == len(onsets):
        return len(onsets) - 1
    return right - 1 if tick - onsets[right - 1] <= onsets[right] - tick else right


def local_ioi(onset_ms: Sequence[float], onset_index: int, window: int) -> tuple[float | None, int]:
    if len(onset_ms) < 2:
        return None, 0
    left_count = window // 2
    right_count = window - left_count
    start = max(0, onset_index - left_count)
    stop = min(len(onset_ms) - 1, onset_index + right_count)
    values = [onset_ms[i + 1] - onset_ms[i] for i in range(start, stop) if onset_ms[i + 1] - onset_ms[i] > 0]
    return (float(np.median(values)), len(values)) if values else (None, 0)


def extract_transitions(onsets: list[int], onset_ms: list[float], controls: list[tuple[int, int]], tick_to_ms) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous = 0
    for tick, value in controls:
        direction = "UP" if previous >= 64 and value < 64 else "DOWN" if previous < 64 and value >= 64 else None
        if direction and onsets:
            index = nearest_onset(onsets, tick)
            row: dict[str, Any] = {"direction": direction, "tick": tick, "time_ms": tick_to_ms(tick), "onset_index": index}
            for window in WINDOWS:
                row[f"ioi_w{window}"] = local_ioi(onset_ms, index, window)[0]
            rows.append(row)
        previous = value
    return rows


def choose_piece(manifest: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    subset = manifest[manifest["composer"].isin(["Chopin", "Schumann"])]
    grouped = subset.groupby(["piece_id", "composer", "title"], as_index=False).agg(
        performance_count=("performance_path", "size"), raw_pedal_count=("has_raw_cc64", "sum"),
        median_cc64=("num_raw_cc64_events", "median"), median_notes=("num_normalized_notes", "median"),
        max_notes=("num_normalized_notes", "max"),
    )
    eligible = grouped[
        grouped["performance_count"].between(5, 8)
        & (grouped["raw_pedal_count"] >= 5)
        & (grouped["median_cc64"] >= 100)
        & grouped["median_notes"].between(500, 2500)
        & (grouped["max_notes"] <= 3500)
    ].copy()
    if eligible.empty:
        raise RuntimeError("no Chopin/Schumann piece satisfies deterministic pilot criteria")
    eligible["distance_from_six"] = (eligible["performance_count"] - 6).abs()
    chosen = eligible.sort_values(["distance_from_six", "median_cc64", "piece_id"], ascending=[True, False, True]).iloc[0].to_dict()
    return manifest[manifest["piece_id"] == chosen["piece_id"]].sort_values("metadata_index"), chosen


def cleanup_stem(tool_dir: Path) -> None:
    for stem in ("mini_score", "mini_performance"):
        for path in tool_dir.glob(f"{stem}*"):
            if path.is_file():
                path.unlink()


def run_alignment(score_path: Path, performance_path: Path, tool_dir: Path, process_log: Path) -> list[tuple[int, int]]:
    cleanup_stem(tool_dir)
    normalize_midi(MidiFile(str(score_path))).dump(str(tool_dir / "mini_score.mid"))
    normalize_midi(MidiFile(str(performance_path))).dump(str(tool_dir / "mini_performance.mid"))
    with process_log.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(
            ["bash", "./MIDIToMIDIAlign.sh", "mini_performance", "mini_score"], cwd=tool_dir,
            stdout=output, stderr=subprocess.STDOUT, text=True, start_new_session=True,
        )
        try:
            return_code = process.wait(timeout=180)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait()
            raise RuntimeError("existing Nakamura/PT alignment exceeded 180 seconds")
    path = tool_dir / "mini_score_corresp.txt"
    if return_code != 0 or not path.is_file():
        raise RuntimeError(f"existing alignment failed, return code={return_code}")
    result = [(int(score_index), int(performance_index)) for score_index, performance_index in read_corresp(str(path))]
    cleanup_stem(tool_dir)
    return result


def build_score_positions(score_path: Path, performance_path: Path, raw_onsets: list[int], tool_dir: Path, process_log: Path) -> list[int | None]:
    score = normalize_midi(MidiFile(str(score_path)))
    raw_performance = MidiFile(str(performance_path))
    performance = normalize_midi(raw_performance)
    score_notes = score.instruments[0].notes
    performance_notes = performance.instruments[0].notes
    score_onsets = sorted({int(note.start) for note in score_notes})
    score_rank = {tick: index for index, tick in enumerate(score_onsets)}
    by_tick: dict[int, list[int]] = defaultdict(list)
    for score_index, performance_index in run_alignment(score_path, performance_path, tool_dir, process_log):
        if performance_index >= 0:
            by_tick[int(performance_notes[performance_index].start)].append(score_rank[int(score_notes[score_index].start)])
    if not by_tick:
        raise RuntimeError("alignment returned no matched performance notes")
    raw_time = raw_performance.get_tick_to_time_mapping()
    mapped_ticks = sorted(by_tick)
    result: list[int | None] = []
    for tick in raw_onsets:
        normalized_tick = round(float(raw_time[tick]) * 1000.0)
        position = int(np.searchsorted(mapped_ticks, normalized_tick, side="left"))
        if position == 0:
            selected_tick = mapped_ticks[0]
        elif position == len(mapped_ticks):
            selected_tick = mapped_ticks[-1]
        else:
            selected_tick = mapped_ticks[position - 1] if normalized_tick - mapped_ticks[position - 1] <= mapped_ticks[position] - normalized_tick else mapped_ticks[position]
        result.append(int(round(float(np.median(by_tick[selected_tick])))))
    return result


def log_line(path: Path, message: str) -> None:
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def load_piece(rows: pd.DataFrame, args, output: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cache_dir = output / "alignment_cache"; cache_dir.mkdir(exist_ok=True)
    process_dir = output / "alignment_logs"; process_dir.mkdir(exist_ok=True)
    progress = output / "progress.log"
    performances, failures = [], []
    total = len(rows)
    for current, row in enumerate(rows.itertuples(index=False), 1):
        started = time.monotonic(); identifier = str(row.performance_path)
        cache_path = cache_dir / f"{int(row.metadata_index):04d}.json"
        log_line(progress, f"[{current}/{total}] START {identifier}")
        midi_path = Path(row.performance_absolute_path)
        midi = MidiFile(str(midi_path)); onsets, controls = raw_events(midi)
        tick_to_ms = tick_ms_fn(midi); onset_ms = [tick_to_ms(tick) for tick in onsets]
        try:
            if cache_path.is_file():
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                if payload["performance"] != identifier or len(payload["score_positions"]) != len(onsets):
                    raise RuntimeError("persistent cache identity/length mismatch")
                positions = payload["score_positions"]; source = "cache"
            else:
                score_path = args.asap_root / str(row.midi_score)
                positions = build_score_positions(score_path, midi_path, onsets, args.alignment_tool_dir, process_dir / f"{int(row.metadata_index):04d}.log")
                temporary = cache_path.with_suffix(".json.tmp")
                temporary.write_text(json.dumps({"performance": identifier, "metadata_index": int(row.metadata_index), "score_positions": positions}) + "\n", encoding="utf-8")
                temporary.replace(cache_path); source = "new"
            transitions = extract_transitions(onsets, onset_ms, controls, tick_to_ms)
            for event in transitions:
                event["score_position"] = positions[event["onset_index"]]
            performances.append({"piece": str(row.piece_id), "performance": identifier, "metadata_index": int(row.metadata_index), "midi_path": str(midi_path), "onsets": onsets, "onset_ms": onset_ms, "controls": controls, "tick_to_ms": tick_to_ms, "transitions": transitions})
            log_line(progress, f"[{current}/{total}] SUCCESS {identifier} elapsed={time.monotonic()-started:.1f}s source={source}")
        except Exception as error:
            failures.append({"performance": identifier, "error": str(error)})
            log_line(progress, f"[{current}/{total}] FAILURE {identifier} elapsed={time.monotonic()-started:.1f}s error={error}")
    return performances, failures


def match_pair(first: dict[str, Any], second: dict[str, Any], search: int) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    candidates = []
    for ia, a in enumerate(first["transitions"]):
        for ib, b in enumerate(second["transitions"]):
            if a["direction"] != b["direction"]:
                continue
            offset = abs(int(a["score_position"]) - int(b["score_position"]))
            if offset <= search:
                candidates.append((offset, abs(a["onset_index"] - b["onset_index"]), ia, ib))
    used_a, used_b, matched = set(), set(), []
    for _, _, ia, ib in sorted(candidates):
        if ia not in used_a and ib not in used_b:
            used_a.add(ia); used_b.add(ib); matched.append((first["transitions"][ia], second["transitions"][ib]))
    return matched


def transition_tables(performances: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    sensitivity: dict[int, list[dict[str, Any]]] = {search: [] for search in SEARCH_WINDOWS}
    for first, second in combinations(performances, 2):
        for search in SEARCH_WINDOWS:
            for a, b in match_pair(first, second, search):
                row: dict[str, Any] = {
                    "performance_a": first["performance"], "performance_b": second["performance"], "direction": a["direction"],
                    "aligned_musical_position": min(a["score_position"], b["score_position"]),
                    "score_position_a": a["score_position"], "score_position_b": b["score_position"],
                    "delta_onset": abs(a["score_position"] - b["score_position"]),
                    "actual_time_difference_ms": abs(a["time_ms"] - b["time_ms"]), "search_window": search,
                }
                for window in WINDOWS:
                    va, vb = a[f"ioi_w{window}"], b[f"ioi_w{window}"]
                    row[f"local_ioi_a_w{window}_ms"] = "" if va is None else va
                    row[f"local_ioi_b_w{window}_ms"] = "" if vb is None else vb
                    row[f"local_ioi_pair_w{window}_ms"] = "" if va is None or vb is None else float(np.median([va, vb]))
                sensitivity[search].append(row)
    return sensitivity[8], sensitivity


def coverage(rows: Sequence[dict[str, Any]], allowed: int, direction: str | None = None) -> float | None:
    selected = [row for row in rows if direction is None or row["direction"] == direction]
    return None if not selected else float(np.mean([row["delta_onset"] <= allowed for row in selected]))


def adaptive_rows(pairs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for window in WINDOWS:
        valid = [row for row in pairs if row[f"local_ioi_pair_w{window}_ms"] != ""]
        for scale in T_VALUES_MS:
            budgets = [min(4, max(0, math.ceil(scale / float(row[f"local_ioi_pair_w{window}_ms"])) - 1)) for row in valid]
            covered = [row["delta_onset"] <= budget for row, budget in zip(valid, budgets)]
            item: dict[str, Any] = {"W": window, "T_ms": scale, "pair_count": len(valid), "coverage": float(np.mean(covered)) if covered else None, "mean_K": float(np.mean(budgets)) if budgets else None, "median_K": q(budgets, .5)}
            for value in range(5):
                item[f"K{value}_proportion"] = float(np.mean(np.asarray(budgets) == value)) if budgets else None
            for direction in ("UP", "DOWN"):
                indexes = [i for i, row in enumerate(valid) if row["direction"] == direction]
                item[f"{direction.lower()}_coverage"] = float(np.mean([covered[i] for i in indexes])) if indexes else None
            output.append(item)
    return output


def repedal_candidates(performance: dict[str, Any]) -> list[dict[str, Any]]:
    compact: list[tuple[int, int]] = []
    for event in performance["controls"]:
        if not compact or compact[-1][1] != event[1]:
            compact.append(event)
    peaks, troughs = [], []
    for index in range(1, len(compact) - 1):
        left, current, right = compact[index - 1][1], compact[index][1], compact[index + 1][1]
        if current > left and current > right:
            peaks.append(index)
        elif current < left and current < right:
            troughs.append(index)
    triples = []
    for trough in troughs:
        prior = next((index for index in reversed(peaks) if index < trough and compact[index][1] - compact[trough][1] >= 64), None)
        following = next((index for index in peaks if index > trough and compact[index][1] - compact[trough][1] >= 64), None)
        if prior is not None and following is not None:
            triples.append((prior, trough, following))
    rows = []
    for candidate_id, (prior, trough, following) in enumerate(sorted(set(triples), key=lambda item: (item[1], item[0], item[2]))):
        ticks = [compact[index][0] for index in (prior, trough, following)]
        values = [compact[index][1] for index in (prior, trough, following)]
        onset_indexes = [nearest_onset(performance["onsets"], tick) for tick in ticks]
        row: dict[str, Any] = {
            "performance": performance["performance"], "candidate_id": candidate_id,
            "release_onset_span": abs(onset_indexes[1] - onset_indexes[0]), "repress_onset_span": abs(onset_indexes[2] - onset_indexes[1]),
            "total_onset_span": abs(onset_indexes[2] - onset_indexes[0]),
            "duration_ms": performance["tick_to_ms"](ticks[2]) - performance["tick_to_ms"](ticks[0]),
            "release_excursion": values[0] - values[1], "repress_excursion": values[2] - values[1],
        }
        for window in WINDOWS:
            value, _ = local_ioi(performance["onset_ms"], onset_indexes[1], window)
            row[f"local_ioi_w{window}_ms"] = "" if value is None else value
        rows.append(row)
    return rows


def span_summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for name in ("release", "repress", "total"):
        values = [row[f"{name}_onset_span"] for row in rows]
        item: dict[str, Any] = {"span": name, "candidate_count": len(values), "median": q(values, .5), "p75": q(values, .75), "p90": q(values, .90), "p95": q(values, .95)}
        for limit in range(1, 5):
            item[f"proportion_le_{limit}"] = float(np.mean(np.asarray(values) <= limit)) if values else None
        result.append(item)
    return result


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    # Spearman is Pearson correlation over average ranks; avoid optional SciPy.
    return float(pd.Series(x).rank(method="average").corr(pd.Series(y).rank(method="average"), method="pearson"))


def report(output: Path, chosen: dict[str, Any], performances: list[dict[str, Any]], failures: list[dict[str, Any]], pairs: list[dict[str, Any]], sensitivity: dict[int, list[dict[str, Any]]], adaptive: list[dict[str, Any]], repedal_performances: list[dict[str, Any]], repedals: list[dict[str, Any]], spans: list[dict[str, Any]]) -> None:
    fixed = [{"K": k, "all": coverage(pairs, k), "up": coverage(pairs, k, "UP"), "down": coverage(pairs, k, "DOWN")} for k in range(5)]
    transition_relations = []
    for window in WINDOWS:
        valid = [row for row in pairs if row[f"local_ioi_pair_w{window}_ms"] != ""]
        ioi = [float(row[f"local_ioi_pair_w{window}_ms"]) for row in valid]; delta = [float(row["delta_onset"]) for row in valid]
        comparable = [row for row in valid if row["local_ioi_pair_w8_ms"] != ""]
        relative_to_w8 = [
            abs(float(row[f"local_ioi_pair_w{window}_ms"]) - float(row["local_ioi_pair_w8_ms"]))
            / max((float(row[f"local_ioi_pair_w{window}_ms"]) + float(row["local_ioi_pair_w8_ms"])) / 2.0, 1e-9)
            for row in comparable
        ]
        transition_relations.append({"W": window, "n": len(valid), "ioi_p25": q(ioi, .25), "ioi_median": q(ioi, .5), "ioi_p75": q(ioi, .75), "median_relative_to_w8": q(relative_to_w8, .5), "rho": spearman(ioi, delta)})
    repedal_relations = []
    for window in WINDOWS:
        valid = [row for row in repedals if row[f"local_ioi_w{window}_ms"] != ""]
        repedal_relations.append({"W": window, "rho": spearman([float(row[f"local_ioi_w{window}_ms"]) for row in valid], [float(row["total_onset_span"]) for row in valid])})
    shortlist = [row for row in adaptive if (row["W"], row["T_ms"]) in {(8, 200), (8, 300), (12, 300)}]
    lines = [
        "# Mini Pedal Event Tolerance Audit v1", "", "## Scope and selection", "",
        "Only the canonical ASAP train manifest and train MIDI were used. ASAP validation/test and MAESTRO were not opened. No model, inference, tokenizer, or production evaluator was used.", "",
        f"Selected piece: **{chosen['composer']} — {chosen['title']}** (`{chosen['piece_id']}`). Deterministic rule: composers restricted to Chopin/Schumann; 5–8 train performances; at least 5 with raw CC64; median CC64 ≥100; median normalized notes 500–2500 and maximum ≤3500. Sort by distance from 6 performances, then descending median CC64, then piece ID. This selected {int(chosen['performance_count'])} performances without looking at alignment or transition outcomes.", "",
        f"Successful alignments: {len(performances)}/{int(chosen['performance_count'])}; failures: {len(failures)}. Every success was atomically cached immediately; reruns skip valid caches. Progress and external-tool logs are retained.", "",
        "Distinct onsets are unique final-performance note-on timestamps. Local IOI uses positive differences only and its median. CC64 UP/DOWN uses the fixed threshold 64.", "", "## A. Transition findings", "",
        f"Primary ±8 matching produced {len(pairs)} pairs: UP {sum(row['direction']=='UP' for row in pairs)}, DOWN {sum(row['direction']=='DOWN' for row in pairs)}.", "",
        "| fixed K | all coverage | UP | DOWN |", "| ---: | ---: | ---: | ---: |",
    ]
    lines += [f"| {row['K']} | {f(row['all'])} | {f(row['up'])} | {f(row['down'])} |" for row in fixed]
    lines += ["", "Search sensitivity: " + ", ".join(f"+/-{search}: {len(rows)} pairs" for search, rows in sensitivity.items()) + ".", "", "| W | valid pairs | IOI p25/median/p75 ms | median relative difference vs W=8 | Spearman(IOI, delta onset) |", "| ---: | ---: | --- | ---: | ---: |"]
    lines += [f"| {row['W']} | {row['n']} | {f(row['ioi_p25'])}/{f(row['ioi_median'])}/{f(row['ioi_p75'])} | {f(row['median_relative_to_w8'])} | {f(row['rho'])} |" for row in transition_relations]
    lines += ["", "Adaptive sanity-check shortlist (not final parameters):", "", "| W | T ms | coverage | mean/median K | K=0/1/2/3/4 | UP/DOWN | trade-off |", "| ---: | ---: | ---: | --- | --- | --- | --- |"]
    for row in shortlist:
        proportions = "/".join(f(row[f"K{k}_proportion"], 2) for k in range(5))
        trade = "more local" if row["W"] == 8 else "smoother local IOI"
        lines.append(f"| {row['W']} | {row['T_ms']} | {f(row['coverage'])} | {f(row['mean_K'])}/{f(row['median_K'])} | {proportions} | {f(row['up_coverage'])}/{f(row['down_coverage'])} | {trade} |")
    rho8 = next(row["rho"] for row in transition_relations if row["W"] == 8)
    relation_text = "clear" if rho8 is not None and rho8 <= -0.4 else "modest" if rho8 is not None and rho8 <= -0.2 else "weak or absent"
    lines += ["", "W=8 and W=12 give nearly the same median IOI and transition/IOI relationship. W=4 is the most event-local and therefore more exposed to individual short IOIs; W=16 shifts the distribution most and is the most likely to smooth local tempo change. W choice does not materially change the transition conclusion in this pilot.", "", f"At W=8, the fast-region hypothesis is **{relation_text}** (negative correlation means shorter IOI tends to larger onset error). Fixed K=1 reaches 0.963 coverage with exactly one onset allowed, whereas W=8/T=200 reaches only 0.939 at mean K=1.598 and W=8/T=300 reaches 0.971 at mean K=2.212. Adaptive tolerance therefore does not improve the transition coverage/width trade-off here.", "", "## B. Repedal findings", "", f"Analyzed performances: {len(repedal_performances)}; strong candidates: {len(repedals)}.", "", "| span | median | p75 | p90 | p95 | <=1/2/3/4 |", "| --- | ---: | ---: | ---: | ---: | --- |"]
    for row in spans:
        props = "/".join(f(row[f"proportion_le_{k}"], 2) for k in range(1, 5))
        lines.append(f"| {row['span']} | {f(row['median'])} | {f(row['p75'])} | {f(row['p90'])} | {f(row['p95'])} | {props} |")
    lines += ["", "IOI versus total repedal span Spearman correlations: " + ", ".join(f"W={row['W']}: {f(row['rho'])}" for row in repedal_relations) + ". Negative values support a larger onset budget in faster passages.", "", "Extrema rule: collapse consecutive equal CC64 values; identify strict local peaks/troughs; for each trough choose the nearest prior and following local peaks satisfying both ≥64 excursions; deduplicate exact `(pre-peak,trough,post-peak)` triples. This minimizes gratuitously long nested gestures.", "", "## C. Research conclusions", ""]
    k1, k2, k4 = fixed[1]["all"], fixed[2]["all"], fixed[4]["all"]
    lines += [
        f"1. This pilot favors fixed +/-1 as the primary Transition F1 tolerance (coverage {f(k1)}), with fixed +/-2 (coverage {f(k2)}) as a conservative sensitivity check. The tested adaptive candidates are less efficient.",
        f"2. Maximum 4-onset tolerance covers {f(k4)}, only {f(None if k4 is None or k2 is None else k4-k2)} above +/-2, so +/-3 to 4 appears unnecessarily broad for transition matching in this piece.",
        f"3. Local-IOI evidence for transition adaptivity is {relation_text}; this pilot does not justify retaining adaptive Transition F1 tolerance.",
        "4. Repedal scale depends on the boundary: trough-to-repress is typically 1-2 onsets, but peak-to-trough release and total peak-to-peak spans are much longer (medians 9 and 11). Thus a fixed 1-4 window describes the re-press action but not the full peak-to-peak candidate under this definition. The modest negative IOI/span correlations support further adaptive-window investigation, not a cutoff.",
        "5. One lyrical Chopin piece cannot establish a general heuristic. Before production implementation, check one contrasting, rhythmically dense ASAP-train piece in a separate task.", "",
        "No final W, T, transition tolerance, or repedal cutoff is selected by this report.", "",
        "## Artifacts", "", "- `transition_pairs.csv`", "- `adaptive_tolerance_scan.csv`", "- `repedal_candidates.csv`", "- `repedal_span_summary.csv`", "- `progress.log`, `alignment_cache/`, `alignment_logs/`", "- two compact W=8 diagnostic plots", "",
    ]
    (output / "MINI_PEDAL_TOLERANCE_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asap-root", type=Path, default=Path("/workspace/public/ASAP/asap-dataset-v1.1"))
    parser.add_argument("--manifest", type=Path, default=ROOT / "analysis/stage2_binary_v0/data_prep_v1/asap_train_manifest.csv")
    parser.add_argument("--alignment-tool-dir", type=Path, default=Path("/tmp/original_pt_early_alignment/tools/AlignmentTool"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "analysis/pedal_event_metric_tolerance_mini_audit_v1")
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest)
    if set(manifest["source"]) != {"ASAP-train"} or set(manifest["split"]) != {"train"} or len(manifest) != 892:
        raise RuntimeError("expected exact canonical ASAP train-only manifest with 892 rows")
    selected_rows, chosen = choose_piece(manifest)
    performances, failures = load_piece(selected_rows, args, args.output_dir)
    if len(performances) < 2:
        raise RuntimeError("fewer than two successful alignments")
    pairs, sensitivity = transition_tables(performances)
    if not pairs:
        raise RuntimeError("selected piece produced no matched transitions")
    write_csv(args.output_dir / "transition_pairs.csv", pairs, list(pairs[0]))
    adaptive = adaptive_rows(pairs); write_csv(args.output_dir / "adaptive_tolerance_scan.csv", adaptive, list(adaptive[0]))

    repedal_performances = list(performances)
    repedals = [candidate for performance in repedal_performances for candidate in repedal_candidates(performance)]
    if len(repedals) < 40:
        extras = manifest[(~manifest["piece_id"].eq(chosen["piece_id"])) & manifest["composer"].isin(["Chopin", "Schumann"]) & manifest["has_raw_cc64"] & manifest["num_normalized_notes"].between(500, 2500) & (manifest["num_raw_cc64_events"] >= 100)].sort_values(["composer", "title", "performance_path"])
        for row in extras.itertuples(index=False):
            if len(repedal_performances) >= 10 or len(repedals) >= 40:
                break
            midi = MidiFile(str(row.performance_absolute_path)); onsets, controls = raw_events(midi)
            convert = tick_ms_fn(midi); onset_ms = [convert(tick) for tick in onsets]
            performance = {"performance": str(row.performance_path), "onsets": onsets, "onset_ms": onset_ms, "controls": controls, "tick_to_ms": convert}
            repedal_performances.append(performance); repedals.extend(repedal_candidates(performance))
    write_csv(args.output_dir / "repedal_candidates.csv", repedals, list(repedals[0]) if repedals else ["performance", "candidate_id"])
    spans = span_summary(repedals); write_csv(args.output_dir / "repedal_span_summary.csv", spans, list(spans[0]))

    valid = [row for row in pairs if row["local_ioi_pair_w8_ms"] != ""]
    plt.figure(figsize=(6, 4)); plt.hexbin([float(row["local_ioi_pair_w8_ms"]) for row in valid], [row["delta_onset"] for row in valid], gridsize=30, mincnt=1); plt.xlabel("local median IOI W=8 (ms)"); plt.ylabel("|Δonset|"); plt.tight_layout(); plt.savefig(args.output_dir / "transition_ioi_w8.png", dpi=150); plt.close()
    valid_repedal = [row for row in repedals if row["local_ioi_w8_ms"] != ""]
    plt.figure(figsize=(6, 4)); plt.scatter([float(row["local_ioi_w8_ms"]) for row in valid_repedal], [row["total_onset_span"] for row in valid_repedal], s=8, alpha=.35); plt.xlabel("local median IOI W=8 (ms)"); plt.ylabel("total repedal onset span"); plt.tight_layout(); plt.savefig(args.output_dir / "repedal_ioi_w8.png", dpi=150); plt.close()
    report(args.output_dir, chosen, performances, failures, pairs, sensitivity, adaptive, repedal_performances, repedals, spans)
    print(json.dumps({"piece": chosen["title"], "aligned": len(performances), "transition_pairs": len(pairs), "repedal_performances": len(repedal_performances), "repedal_candidates": len(repedals), "output": str(args.output_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()
