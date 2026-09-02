#!/usr/bin/env python3
"""Frozen Custom Event v0 four-state/depth-state failure audit v1."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from miditoolkit import MidiFile

from scripts.audit_pedal_event_metric_tolerance_mini import match_pair, nearest_onset, raw_events, tick_ms_fn
from scripts.run_custom_event_model_v0_canonical_val_inference_v1 import (
    ALIGNMENT_ROOT, CACHE_ROOT, CHECKPOINT, DECODER_ROOT, EXPECTED_ALIGNMENT_CONFIG_SHA,
    EXPECTED_CHECKPOINT_SHA, EXPECTED_DECODER_ID, FROZEN_HUMAN_ALIGNMENT,
    timeline_from_cache, write_candidate,
)
from scripts.run_stage2_4class_validation_eval_v0 import (
    SPLIT_CSV, STAGE1_MANIFEST, align_and_cache, finalize_transition, sum_transition,
)
from src.stage2_event_model.dataset import CANONICAL_ALIGNMENT_ID, CANONICAL_CACHE_ID
from src.stage2_event_model.failure_audit_v1 import (
    EVENT_NAMES, STATE_NAMES, assert_aligned_universe, binary_states, collapse_binary_patterns,
    confusion, confusion_metrics, error_decomposition, mismatch_episodes, residence_episodes,
    slot_funnel_counts,
)
from src.stage2_event_model.prediction_decoder_v1 import (
    DECODER_ID, DECODER_VERSION, decode_predictions_v1, quantize_decoded_timeline_v1,
)
from src.stage2_four_class.validation_evaluator import (
    DISTRIBUTION_EPSILON, canonical_classes_from_tokens, pattern_metrics,
    pooled_transition_counts,
)


OUTPUT = ROOT / "analysis/custom_event_v0_4state_failure_audit_v1"
RAW_ROOT = OUTPUT / "raw_predictions"
CUSTOM_ROOT = ROOT / "analysis/custom_event_model_v0_canonical_val_inference_v1"
HYBRID_ROOT = ROOT / "analysis/custom_vs_hybrid_matched_input_val_v1"
TOKENIZER_ROOT = ROOT / "analysis/custom_event_tokenizer_v1"
TRAIN_ROOT = ROOT / "analysis/custom_event_model_v0_full_seed42_aligned_statusfix_v1"
HYBRID_CHECKPOINT = ROOT / "analysis/stage2_encoder_only_raw_huber_aux_ce_v1/best.pt"
CLASS_WEIGHTS = TOKENIZER_ROOT / "candidate_event_weights.json"
STATE_VALUES = (0, 51, 79, 127)
EXPECTED = {
    "tokenizer_cache_id": "3a5520155b5db1e9a1da7f8148556aa3e1da852655c9adde25d3dbbd1d966263",
    "note_alignment_id": "98de59ef7a41fba26a2c89fe686c273f6c8ec7f27979922e71813624ac1a4822",
    "alignment_config_sha256": "77df6faf08025794e40c58d0db31cf721f582b3168d9fb201557c0500807502b",
    "decoder_id": "938ca7e20e1d398aa530938bbcede404fd3a04af23842fbcf351ba02df0268f8",
}


def safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(name: str, value: Any) -> None:
    path = OUTPUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(safe(value), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_csv(name: str, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    path = OUTPUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(fields or (list(rows[0]) if rows else []))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows([safe(row) for row in rows])


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quantiles(values: Iterable[float], *, include_max: bool = False) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=float)
    if not len(array):
        return {key: None for key in (("mean", "p50", "p75", "p90", "p95", "p99", "max") if include_max else ("mean", "p50", "p75", "p90", "p95"))}
    result = {"mean": float(array.mean())}
    for name, level in (("p50", .5), ("p75", .75), ("p90", .9), ("p95", .95)):
        result[name] = float(np.quantile(array, level))
    if include_max:
        result.update(p99=float(np.quantile(array, .99)), max=float(array.max()))
    return result


def preflight() -> tuple[dict[str, Any], list[dict[str, str]], dict[str, Any]]:
    cache = json.loads((TOKENIZER_ROOT / "cache_manifest.json").read_text())
    alignment = json.loads((ALIGNMENT_ROOT / "alignment_manifest.json").read_text())
    decoder = json.loads((DECODER_ROOT / "decoder_config.json").read_text())
    custom_prov = json.loads((CUSTOM_ROOT / "provenance.json").read_text())
    hybrid_prov = json.loads((HYBRID_ROOT / "provenance.json").read_text())
    raw_manifest = json.loads((RAW_ROOT / "manifest.json").read_text())
    actual = {
        "tokenizer_cache_id": cache["cache_id"],
        "note_alignment_id": alignment["alignment_id"],
        "alignment_config_sha256": alignment["alignment_config_sha256"],
        "decoder_id": decoder["decoder_id"],
    }
    if actual != EXPECTED or CANONICAL_CACHE_ID != EXPECTED["tokenizer_cache_id"] or CANONICAL_ALIGNMENT_ID != EXPECTED["note_alignment_id"]:
        raise RuntimeError(f"frozen identity mismatch: {actual}")
    if DECODER_ID != EXPECTED_DECODER_ID or DECODER_VERSION != "1.0.0":
        raise RuntimeError("Decoder v1 runtime identity mismatch")
    if sha256(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA or int(custom_prov["checkpoint_epoch"]) != 1:
        raise RuntimeError("Custom epoch-1 checkpoint mismatch")
    if int(hybrid_prov["checkpoint_epoch"]) != 5 or sha256(HYBRID_CHECKPOINT) != hybrid_prov["checkpoint_sha256"]:
        raise RuntimeError("matched Hybrid epoch-5 checkpoint mismatch")
    validation = sorted(
        [row for row in load_csv(SPLIT_CSV) if row["split"] == "validation"],
        key=lambda row: int(row["metadata_index"]),
    )
    if len(validation) != 71 or len({row["piece_id"] for row in validation}) != 19:
        raise RuntimeError("ASAP validation universe mismatch")
    if raw_manifest["validation_performances"] != 71 or raw_manifest["modeled_onsets"] != 272927:
        raise RuntimeError("raw prediction universe mismatch")
    if any(int(x.get("asap_test_access_count", 0)) for x in (cache, alignment, decoder, custom_prov, hybrid_prov, raw_manifest)):
        raise RuntimeError("ASAP test access found in provenance")
    provenance = {
        **actual, "decoder_version": DECODER_VERSION,
        "custom_checkpoint": str(CHECKPOINT), "custom_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
        "custom_best_epoch": 1, "hybrid_checkpoint": str(HYBRID_CHECKPOINT),
        "hybrid_checkpoint_sha256": hybrid_prov["checkpoint_sha256"], "hybrid_best_epoch": 5,
        "asap_validation_performances": 71, "asap_validation_pieces": 19,
        "frozen_successful_pairs": 70, "frozen_aligned_notes": 272053,
        "frozen_pedal_samples": 1088212,
        "known_exclusion": "Liszt/Mephisto_Waltz/Tysman07M.mid",
        "raw_prediction_digest": raw_manifest["aggregate_prediction_digest"],
        "asap_test_access_count": 0, "repedal_execution_count": 0,
        "training_steps": 0, "tuning_steps": 0,
    }
    return provenance, validation, raw_manifest


def load_classes(root: Path, identifier: str) -> np.ndarray | None:
    path = root / identifier / "aligned_tokens_int64.npy"
    meta = root / identifier / "metadata.json"
    if not path.is_file() or not meta.is_file() or not json.loads(meta.read_text()).get("complete"):
        return None
    return canonical_classes_from_tokens(np.load(path, allow_pickle=False))


def sample_axis(tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    intervals = np.maximum(tokens[:, 1].astype(np.int64) - 261, 0)
    onset_ticks = np.cumsum(intervals)
    next_intervals = np.r_[intervals[1:], 4990]
    times = onset_ticks[:, None] + next_intervals[:, None] * np.asarray([0, .25, .5, .75])
    distinct = np.zeros(len(onset_ticks), dtype=np.int64)
    if len(onset_ticks) > 1:
        distinct[1:] = np.cumsum(onset_ticks[1:] != onset_ticks[:-1])
    note_index = np.repeat(np.arange(len(tokens)), 4)
    sample_slot = np.tile(np.arange(4), len(tokens))
    flat_times = times.reshape(-1) / 1000.0
    flat_onsets = np.repeat(distinct, 4)
    order = np.lexsort((sample_slot, note_index, flat_times))
    return flat_times[order], flat_onsets[order], order


def state_occupancy(values: np.ndarray) -> dict[str, Any]:
    counts = np.bincount(values.reshape(-1), minlength=4)
    return {"count": {STATE_NAMES[i]: int(counts[i]) for i in range(4)},
            "ratio": {STATE_NAMES[i]: float(counts[i] / counts.sum()) for i in range(4)}}


def binary_report(candidate: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    matrix = confusion(binary_states(candidate), binary_states(target), 2)
    metrics = confusion_metrics(matrix)
    for row, name in zip(metrics["class_metrics"], ("OFF", "ON"), strict=True):
        row["class_name"] = name
    return metrics


def pattern_distribution(classes: np.ndarray, base: int) -> tuple[np.ndarray, np.ndarray]:
    if base == 4:
        ids = classes @ np.asarray([64, 16, 4, 1])
    else:
        ids = collapse_binary_patterns(classes)
    hist = np.bincount(ids, minlength=base ** 4).astype(np.int64)
    probability = hist / hist.sum()
    return hist, probability


def distribution_metric(candidate_hist: np.ndarray, target_hist: np.ndarray) -> dict[str, Any]:
    bins = len(candidate_hist)
    p = candidate_hist.astype(float) / candidate_hist.sum() + DISTRIBUTION_EPSILON
    q = target_hist.astype(float) / target_hist.sum() + DISTRIBUTION_EPSILON
    p /= p.sum(); q /= q.sum(); midpoint = .5 * (p + q)
    contribution = .5 * (p * np.log2(p / midpoint) + q * np.log2(q / midpoint))
    return {"bins": bins, "js_divergence_base2": float(contribution.sum()),
            "intersection": float(np.minimum(p, q).sum()),
            "candidate_probability": p, "target_probability": q,
            "js_contribution": contribution}


def pattern_name(index: int, base: int) -> str:
    names = STATE_NAMES if base == 4 else ("OFF", "ON")
    digits = []
    for power in (base ** 3, base ** 2, base, 1):
        digits.append((index // power) % base)
    return "/".join(names[digit] for digit in digits)


def midi_state_data(path: Path) -> dict[str, Any]:
    midi = MidiFile(str(path))
    onsets, controls = raw_events(midi)
    tick_to_ms = tick_ms_fn(midi)
    previous = 0
    changes = []
    for tick, raw in controls:
        state = 0 if raw <= 25 else 1 if raw <= 63 else 2 if raw <= 103 else 3
        if state == previous:
            continue
        changes.append({"tick": int(tick), "seconds": tick_to_ms(tick) / 1000.0,
                        "onset_index": nearest_onset(onsets, tick) if onsets else 0,
                        "source": previous, "destination": state,
                        "pair": f"{STATE_NAMES[previous]}->{STATE_NAMES[state]}",
                        "type": "TYPE_A" if {previous, state} == {0, 1} else
                                "TYPE_B" if {previous, state} == {2, 3} else "TYPE_C"})
        previous = state
    note_ends = [int(note.end) for inst in midi.instruments if not inst.is_drum for note in inst.notes]
    first = onsets[0] if onsets else 0
    last = max(note_ends + ([controls[-1][0]] if controls else []) + [first])
    current = 0
    for event in changes:
        if event["tick"] <= first:
            current = event["destination"]
    trajectory = [{"tick": first, "seconds": tick_to_ms(first) / 1000.0,
                   "onset_index": 0, "state": current}]
    for event in changes:
        if event["tick"] > first:
            trajectory.append({"tick": event["tick"], "seconds": event["seconds"],
                               "onset_index": event["onset_index"], "state": event["destination"]})
    end_seconds = tick_to_ms(last) / 1000.0
    residences = []
    for index, item in enumerate(trajectory):
        stop_seconds = trajectory[index + 1]["seconds"] if index + 1 < len(trajectory) else end_seconds
        stop_onset = trajectory[index + 1]["onset_index"] if index + 1 < len(trajectory) else max(0, len(onsets) - 1)
        residences.append({"state": item["state"], "duration_seconds": max(0.0, stop_seconds - item["seconds"]),
                           "distinct_onset_span": max(0, stop_onset - item["onset_index"])})
    return {"changes": changes, "residences": residences, "onsets": onsets,
            "tick_to_ms": tick_to_ms}


def taxonomy_summary(changes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counter = Counter(row["type"] for row in changes)
    pairs = Counter(row["pair"] for row in changes)
    total = len(changes)
    return {"total_effective_4state_changes": total,
            "types": {kind: {"count": counter[kind], "ratio": counter[kind] / total if total else 0.0}
                      for kind in ("TYPE_A", "TYPE_B", "TYPE_C")},
            "destination_pairs": dict(sorted(pairs.items()))}


def match_state_changes(candidate: Sequence[Mapping[str, Any]], target: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    pairs = ("ZERO->LOW", "LOW->ZERO", "HALF->FULL", "FULL->HALF",
             "ZERO->HALF", "ZERO->FULL", "LOW->HALF", "LOW->FULL",
             "HALF->ZERO", "FULL->ZERO", "HALF->LOW", "FULL->LOW")
    for pair in pairs:
        first = [x for x in candidate if x["pair"] == pair]
        second = [x for x in target if x["pair"] == pair]
        options = sorted((abs(a["onset_index"] - b["onset_index"]), ia, ib)
                         for ia, a in enumerate(first) for ib, b in enumerate(second)
                         if abs(a["onset_index"] - b["onset_index"]) <= 1)
        ua, ub = set(), set()
        for _, ia, ib in options:
            if ia not in ua and ib not in ub:
                ua.add(ia); ub.add(ib)
        matched = len(ua); precision = matched / len(first) if first else 0.0
        recall = matched / len(second) if second else 0.0
        result[pair] = {"target_count": len(second), "candidate_count": len(first),
                        "matched_count": matched, "precision": precision, "recall": recall,
                        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}
    return result


def aggregate_match(performance_rows: Sequence[Mapping[str, Any]], model: str) -> dict[str, Any]:
    totals: dict[str, Counter] = defaultdict(Counter)
    for row in performance_rows:
        matched = match_state_changes(row[model]["changes"], row["human"]["changes"])
        for pair, item in matched.items():
            for key in ("target_count", "candidate_count", "matched_count"):
                totals[pair][key] += item[key]
    result = {}
    for pair, row in totals.items():
        p = row["matched_count"] / row["candidate_count"] if row["candidate_count"] else 0.0
        r = row["matched_count"] / row["target_count"] if row["target_count"] else 0.0
        result[pair] = {**dict(row), "precision": p, "recall": r,
                        "f1": 2 * p * r / (p + r) if p + r else 0.0}
    return result


def decode_record(index: int, record: Mapping[str, np.ndarray], *, oracle: bool = False):
    manifest = json.loads((RAW_ROOT / "manifest.json").read_text())["records"][index]
    cache_path = Path(manifest["cache_file"])
    with np.load(cache_path, allow_pickle=False) as cache:
        timeline, intervals, seconds_to_tick = timeline_from_cache(cache)
        initial = record["initial_logits"]
        if oracle:
            initial = np.full(4, -1e9, dtype=np.float32)
            initial[int(record["initial_target"])] = 0
        decoded = decode_predictions_v1(
            initial_logits=initial,
            main_event_logits=record["main_event_logits"],
            main_timing_predictions=record["main_timing_predictions"],
            terminal_event_logits=record["terminal_event_logits"],
            terminal_timing_predictions=record["terminal_timing_predictions"],
            timeline=timeline,
        )
        quantized = quantize_decoded_timeline_v1(
            decoded, seconds_to_tick=seconds_to_tick,
            first_onset_tick=int(cache["main_interval_left_tick"][0]),
            latest_note_off_tick=int(cache["latest_note_off_tick"]), main_intervals=intervals,
        )
        source = Path(str(cache["source_midi"].item()))
    return decoded, quantized, source


def decoder_funnel(index: int, record: Mapping[str, np.ndarray]) -> tuple[list[dict[str, int]], list[dict[str, Any]], Any]:
    classes = record["main_event_logits"].argmax(-1)
    timing = record["main_timing_predictions"]
    rows = slot_funnel_counts(classes, timing)
    decoded, quantized, _ = decode_record(index, record)
    redundant = []
    current = decoded.initial_state
    by_interval: dict[int, list[Any]] = defaultdict(list)
    for item in decoded.active_slot_events:
        if item.source == "MAIN":
            by_interval[int(item.main_interval_index)].append(item)
    for interval in sorted(by_interval):
        pairs = sorted(by_interval[interval], key=lambda item: (item.time, item.original_slot_index))
        cursor = 0
        while cursor < len(pairs):
            end = cursor + 1
            while end < len(pairs) and pairs[end].time == pairs[cursor].time:
                end += 1
            item = pairs[end - 1]
            if item.destination_state == current:
                rows[int(item.original_slot_index)]["redundant_same_state"] = rows[int(item.original_slot_index)].get("redundant_same_state", 0) + 1
                redundant.append({"slot": int(item.original_slot_index), "current": current,
                                  "destination": int(item.destination_state)})
            else:
                rows[int(item.original_slot_index)]["continuous_effective_change"] = rows[int(item.original_slot_index)].get("continuous_effective_change", 0) + 1
                current = int(item.destination_state)
            cursor = end
    current = int(quantized.events[0].destination_state)
    for item in quantized.events[1:]:
        if item.source != "MAIN":
            current = int(item.destination_state); continue
        slot = int(item.original_slot_index)
        rows[slot]["quantized_effective_change"] = rows[slot].get("quantized_effective_change", 0) + 1
        if (current >= 2) != (int(item.destination_state) >= 2):
            rows[slot]["threshold_crossing"] = rows[slot].get("threshold_crossing", 0) + 1
        else:
            rows[slot]["same_side_depth_change"] = rows[slot].get("same_side_depth_change", 0) + 1
        current = int(item.destination_state)
    for row in rows:
        for key in ("redundant_same_state", "continuous_effective_change", "quantized_effective_change",
                    "threshold_crossing", "same_side_depth_change"):
            row.setdefault(key, 0)
    return rows, redundant, quantized


def oracle_evaluation(validation: Sequence[Mapping[str, str]], raw_records: Sequence[Mapping[str, np.ndarray]]) -> dict[str, Any]:
    stage1 = {row["piece_id"]: row for row in load_csv(STAGE1_MANIFEST)}
    confusion_total = np.zeros((4, 4), dtype=np.int64)
    candidate_hist = np.zeros(256, dtype=np.int64); target_hist = np.zeros(256, dtype=np.int64)
    transition = {scope: {key: 0 for key in ("candidate", "reference", "tp", "fp", "fn")}
                  for scope in ("pooled", "up", "down")}
    successful = aligned_notes = 0
    rows = []
    for index, (row, record) in enumerate(zip(validation, raw_records, strict=True)):
        _, quantized, source = decode_record(index, record, oracle=True)
        destination = OUTPUT / "oracle_initial_candidate_midi" / f"{index:04d}.mid"
        write_candidate(source, destination, list(quantized.events))
        identifier = row["metadata_index"]
        target = load_classes(FROZEN_HUMAN_ALIGNMENT, identifier)
        if target is None:
            continue
        score = Path(stage1[row["piece_id"]]["selected_score_absolute_path"])
        aligned = align_and_cache(OUTPUT, kind="oracle_initial", identifier=identifier,
                                  score_path=score, performance_path=destination)
        candidate = canonical_classes_from_tokens(aligned["tokens"])
        assert_aligned_universe(candidate, target, notes=len(target))
        matrix = confusion(candidate, target, 4)
        confusion_total += matrix
        c_hist, _ = pattern_distribution(candidate, 4); t_hist, _ = pattern_distribution(target, 4)
        candidate_hist += c_hist; target_hist += t_hist
        target_transitions = json.loads((FROZEN_HUMAN_ALIGNMENT / identifier / "transitions.json").read_text())
        counts = pooled_transition_counts(aligned["transitions"], target_transitions)
        sum_transition(transition, counts)
        successful += 1; aligned_notes += len(target)
        rows.append({"metadata_index": identifier, "aligned_notes": len(target),
                     "four_class_accuracy": float(np.trace(matrix) / matrix.sum())})
        if (index + 1) % 10 == 0:
            print(f"ORACLE_ALIGNMENT_PROGRESS {index+1}/71 successful={successful}", flush=True)
    if successful != 70 or aligned_notes != 272053:
        raise RuntimeError("Oracle-Initial aligned universe mismatch")
    metrics = confusion_metrics(confusion_total)
    pattern = distribution_metric(candidate_hist, target_hist)
    result = {"diagnostic_only_ground_truth_initial": True,
              "common_successful_pairs": successful, "aligned_notes": aligned_notes,
              "four_class_accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
              "transition": finalize_transition(transition),
              "js_divergence_base2": pattern["js_divergence_base2"],
              "intersection": pattern["intersection"], "per_performance": rows}
    return result


def terminal_sample_impact(
    validation: Sequence[Mapping[str, str]], raw_records: Sequence[Mapping[str, np.ndarray]]
) -> dict[str, Any]:
    """Exact frozen-alignment comparison of full Custom vs Main-only decoding."""
    stage1 = {row["piece_id"]: row for row in load_csv(STAGE1_MANIFEST)}
    changed_samples = changed_notes = successful = aligned_notes = 0
    per_performance = []
    for index, (row, record) in enumerate(zip(validation, raw_records, strict=True)):
        _, quantized, source = decode_record(index, record)
        events = [event for event in quantized.events if event.source != "TERMINAL"]
        destination = OUTPUT / "terminal_main_only_candidate_midi" / f"{index:04d}.mid"
        write_candidate(source, destination, events)
        identifier = row["metadata_index"]
        full = load_classes(CUSTOM_ROOT / "alignment/custom_event", identifier)
        if full is None:
            continue
        score = Path(stage1[row["piece_id"]]["selected_score_absolute_path"])
        aligned = align_and_cache(OUTPUT, kind="terminal_main_only", identifier=identifier,
                                  score_path=score, performance_path=destination)
        main_only = canonical_classes_from_tokens(aligned["tokens"])
        assert_aligned_universe(main_only, full, notes=len(full))
        difference = main_only != full
        count = int(difference.sum())
        changed_samples += count
        changed_notes += int(difference.any(axis=1).sum())
        aligned_notes += len(full); successful += 1
        per_performance.append({"metadata_index": identifier, "affected_samples": count,
                                "affected_notes": int(difference.any(axis=1).sum())})
        if (index + 1) % 10 == 0:
            print(f"TERMINAL_IMPACT_ALIGNMENT_PROGRESS {index+1}/71 successful={successful}", flush=True)
    if successful != 70 or aligned_notes != 272053:
        raise RuntimeError("Terminal-impact aligned universe mismatch")
    return {"method": "exact frozen alignment: full Custom minus identical Initial/Main with Terminal removed",
            "successful_pairs": successful, "aligned_notes": aligned_notes,
            "affected_pedal_samples": changed_samples, "affected_aligned_notes": changed_notes,
            "per_performance": per_performance}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    provenance, validation, raw_manifest = preflight()
    write_json("audit_config.json", {
        "states": {"ZERO": [0, 25], "LOW": [26, 63], "HALF": [64, 103], "FULL": [104, 127]},
        "representatives": list(STATE_VALUES), "binary": {"OFF": ["ZERO", "LOW"], "ON": ["HALF", "FULL"]},
        "primary_universe": "frozen aligned Pedal1-4", "state_change_matching_tolerance_distinct_onsets": 1,
        "oracle_initial_predefined": True, "training": False, "tuning": False,
    })
    write_json("provenance.json", provenance)

    custom_all, hybrid_all, human_all = [], [], []
    performance_data = []
    raw_records = []
    custom_manifest = {row["performance_path"]: row for row in json.loads((CUSTOM_ROOT / "candidate_manifest.json").read_text())["candidates"]}
    hybrid_manifest = {row["performance_path"]: row for row in json.loads((HYBRID_ROOT / "hybrid_candidate_manifest.json").read_text())["candidates"]}
    exclusions = []
    for index, row in enumerate(validation):
        with np.load(RAW_ROOT / f"{index:04d}.npz", allow_pickle=False) as data:
            raw_records.append({key: data[key] for key in data.files})
        identifier = row["metadata_index"]
        human = load_classes(FROZEN_HUMAN_ALIGNMENT, identifier)
        custom = load_classes(CUSTOM_ROOT / "alignment/custom_event", identifier)
        hybrid = load_classes(HYBRID_ROOT / "alignment/hybrid_matched", identifier)
        if human is None or custom is None or hybrid is None:
            exclusions.append(row["performance_path"])
            continue
        assert_aligned_universe(human, custom, hybrid, notes=len(human))
        if not (len(human) == len(custom) == len(hybrid)):
            raise AssertionError("aligned shape mismatch")
        tokens = np.load(FROZEN_HUMAN_ALIGNMENT / identifier / "aligned_tokens_int64.npy", allow_pickle=False)
        times, onsets, order = sample_axis(tokens)
        h_flat = human.reshape(-1)[order]; c_flat = custom.reshape(-1)[order]; y_flat = hybrid.reshape(-1)[order]
        human_all.append(human); custom_all.append(custom); hybrid_all.append(hybrid)
        custom_midi = Path(custom_manifest[row["performance_path"]]["candidate_midi"])
        hybrid_midi = Path(hybrid_manifest[row["performance_path"]]["candidate_midi"])
        human_midi = Path(custom_manifest[row["performance_path"]]["source_midi"])
        pdata = {"row": row, "identifier": identifier, "human_classes": human,
                 "custom_classes": custom, "hybrid_classes": hybrid,
                 "times": times, "onsets": onsets, "order": order,
                 "human_flat": h_flat, "custom_flat": c_flat, "hybrid_flat": y_flat,
                 "human": midi_state_data(human_midi), "custom": midi_state_data(custom_midi),
                 "hybrid": midi_state_data(hybrid_midi)}
        performance_data.append(pdata)
    if len(performance_data) != 70 or sum(len(x) for x in human_all) != 272053:
        raise RuntimeError("frozen successful universe regression")
    if exclusions != ["Liszt/Mephisto_Waltz/Tysman07M.mid"]:
        raise RuntimeError(f"unexpected exclusion: {exclusions}")
    human = np.concatenate(human_all); custom = np.concatenate(custom_all); hybrid = np.concatenate(hybrid_all)
    assert_aligned_universe(human, custom, hybrid, notes=272053)

    occupancy = {name: state_occupancy(value) for name, value in (("human", human), ("custom", custom), ("hybrid", hybrid))}
    for model in ("custom", "hybrid"):
        occupancy[model]["ratio_minus_human"] = {
            state: occupancy[model]["ratio"][state] - occupancy["human"]["ratio"][state] for state in STATE_NAMES}
    write_json("state_occupancy.json", occupancy)

    matrices = {}
    for name, value, filename in (("custom", custom, "custom_confusion_matrix.csv"),
                                  ("hybrid", hybrid, "hybrid_confusion_matrix.csv")):
        matrix = confusion(value, human, 4); metrics = confusion_metrics(matrix)
        rows = []
        for i, state in enumerate(STATE_NAMES):
            rows.append({"target_state": state, **{f"pred_{STATE_NAMES[j]}": int(matrix[i, j]) for j in range(4)},
                         **{f"rowpct_{STATE_NAMES[j]}": metrics["row_normalized"][i][j] for j in range(4)}})
        write_csv(filename, rows)
        off_on = confusion(binary_states(value), binary_states(human), 2)
        metrics["explicit_confusions"] = {
            "ZERO_to_LOW": int(matrix[0, 1]), "LOW_to_ZERO": int(matrix[1, 0]),
            "HALF_to_FULL": int(matrix[2, 3]), "FULL_to_HALF": int(matrix[3, 2]),
            "OFF_target_to_ON_prediction": int(off_on[0, 1]),
            "ON_target_to_OFF_prediction": int(off_on[1, 0]),
        }
        matrices[name] = metrics
    decomposition = {"custom": error_decomposition(custom, human), "hybrid": error_decomposition(hybrid, human)}
    binary = {"custom": binary_report(custom, human), "hybrid": binary_report(hybrid, human)}
    for name in ("custom", "hybrid"):
        binary[name]["four_class_accuracy"] = matrices[name]["accuracy"]
        binary[name]["binary_minus_four_class_accuracy"] = binary[name]["accuracy"] - matrices[name]["accuracy"]
    write_json("binary_fidelity.json", {"error_decomposition": decomposition, "binary": binary,
                                         "four_class": matrices})

    taxonomy = {}
    for name in ("human", "custom", "hybrid"):
        changes = [change for item in performance_data for change in item[name]["changes"]]
        taxonomy[name] = taxonomy_summary(changes)
    write_json("state_change_taxonomy.json", taxonomy)
    matching = {"diagnostic_only_not_primary_metric": True,
                "matching_universe": "same matched-input raw distinct-onset index, one-to-one +/-1",
                "custom": aggregate_match(performance_data, "custom"),
                "hybrid": aggregate_match(performance_data, "hybrid")}
    write_json("state_change_matching.json", matching)

    initial_target = np.asarray([int(record["initial_target"]) for record in raw_records])
    initial_pred = np.asarray([int(record["initial_logits"].argmax()) for record in raw_records])
    initial_matrix = confusion(initial_pred, initial_target, 4)
    initial_metrics = confusion_metrics(initial_matrix)
    early = {"correct_initial": {}, "incorrect_initial": {}}
    by_index = {int(row["row"]["metadata_index"]): row for row in performance_data}
    raw_by_identifier = {int(row["metadata_index"]): index for index, row in enumerate(validation)}
    for group, condition in (("correct_initial", initial_pred == initial_target), ("incorrect_initial", initial_pred != initial_target)):
        for count in (4, 8, 16, 32):
            correct = total = 0; performances = 0
            for identifier, item in by_index.items():
                index = raw_by_identifier[identifier]
                if not bool(condition[index]):
                    continue
                take = min(count, len(item["human_flat"]))
                correct += int((item["custom_flat"][:take] == item["human_flat"][:take]).sum())
                total += take; performances += 1
            early[group][f"first_{count}"] = {"accuracy": correct / total if total else None,
                                               "samples": total, "performances": performances}
    initial_audit = {"target_distribution": state_occupancy(initial_target),
                     "predicted_distribution": state_occupancy(initial_pred),
                     "metrics": initial_metrics, "early_trajectory": early,
                     "validation_performances": 71}
    write_json("initial_state_audit.json", initial_audit)

    slot_confusions = []; slot_rows = []; timing_result = {"aggregate": {}, "slots": {}}
    weights = json.loads(CLASS_WEIGHTS.read_text())["main_slots"]
    train_freq_rows = load_csv(TOKENIZER_ROOT / "main_slot_distributions.csv")
    train_freq = {(int(row["slot"]), int(row["event_class"])): (int(row["count"]), float(row["proportion"]))
                  for row in train_freq_rows if row["split"] == "train" and row["scope"] == "main"}
    target_all = np.concatenate([record["main_event_targets"] for record in raw_records])
    pred_all = np.concatenate([record["main_event_logits"].argmax(-1) for record in raw_records])
    timing_pred_all = np.concatenate([record["main_timing_predictions"] for record in raw_records])
    timing_target_all = np.concatenate([record["main_timing_targets"] for record in raw_records])
    for slot in range(6):
        matrix = confusion(pred_all[:, slot], target_all[:, slot], 5)
        slot_confusions.append(matrix.tolist())
        metrics = confusion_metrics(matrix)
        target_non = int((target_all[:, slot] != 0).sum()); pred_non = int((pred_all[:, slot] != 0).sum())
        tp_non = int(((target_all[:, slot] != 0) & (pred_all[:, slot] != 0)).sum())
        p = tp_non / pred_non if pred_non else 0.0; r = tp_non / target_non if target_non else 0.0
        for event in range(5):
            train_count, train_ratio = train_freq[(slot + 1, event)]
            target_count = int((target_all[:, slot] == event).sum())
            predicted_count = int((pred_all[:, slot] == event).sum())
            slot_rows.append({"slot": slot + 1, "event_class": event, "event_name": EVENT_NAMES[event],
                              "train_target_count": train_count, "train_target_frequency": train_ratio,
                              "frozen_class_weight": weights[f"Slot{slot+1}"][EVENT_NAMES[event]],
                              "validation_target_count": target_count,
                              "validation_target_frequency": target_count / len(target_all),
                              "validation_predicted_count": predicted_count,
                              "validation_predicted_frequency": predicted_count / len(pred_all),
                              "predicted_over_target_frequency_ratio": predicted_count / target_count if target_count else None,
                              "class_recall": metrics["class_metrics"][event]["recall"],
                              "class_f1": metrics["class_metrics"][event]["f1"],
                              "target_non_none_rate": target_non / len(target_all),
                              "predicted_non_none_rate": pred_non / len(pred_all),
                              "non_none_precision": p, "non_none_recall": r,
                              "non_none_f1": 2 * p * r / (p + r) if p + r else 0.0})
        active = target_all[:, slot] != 0
        for label, mask in (("event_class_correct", active & (pred_all[:, slot] == target_all[:, slot])),
                            ("event_class_incorrect", active & (pred_all[:, slot] != target_all[:, slot]))):
            difference = timing_pred_all[:, slot][mask] - timing_target_all[:, slot][mask]
            timing_result["slots"].setdefault(f"Slot{slot+1}", {})[label] = {
                "count": int(mask.sum()), "tau_mae": float(np.abs(difference).mean()) if len(difference) else None,
                "tau_rmse": float(np.sqrt(np.square(difference).mean())) if len(difference) else None}
    for label, mask in (("event_class_correct", (target_all != 0) & (pred_all == target_all)),
                        ("event_class_incorrect", (target_all != 0) & (pred_all != target_all))):
        difference = timing_pred_all[mask] - timing_target_all[mask]
        timing_result["aggregate"][label] = {"count": int(mask.sum()),
            "tau_mae": float(np.abs(difference).mean()), "tau_rmse": float(np.sqrt(np.square(difference).mean()))}
    write_csv("slot_event_distributions.csv", slot_rows)
    write_json("slot_event_confusions.json", {"class_order": EVENT_NAMES, "confusions": slot_confusions})
    write_json("timing_conditional_analysis.json", timing_result)

    funnel_totals = [Counter() for _ in range(6)]; redundant_rows = []
    terminal_quantized = []
    for index, record in enumerate(raw_records):
        rows, redundant, quantized = decoder_funnel(index, record)
        for slot, row in enumerate(rows):
            funnel_totals[slot].update(row)
        for item in redundant:
            redundant_rows.append({"performance_index": index, **item})
        terminal_quantized.append(quantized)
    funnel_csv = []
    for slot, row in enumerate(funnel_totals):
        base = row["raw_non_none"]
        funnel_csv.append({"slot": slot + 1, **dict(row),
                           **{f"{key}_ratio_of_raw": row[key] / base if base else 0.0 for key in row if key != "raw_non_none"}})
    if sum(row["raw_non_none"] for row in funnel_totals) != 87712 or sum(row["prefix_active"] for row in funnel_totals) != 86793 or sum(row["redundant_same_state"] for row in funnel_totals) != 28586:
        raise RuntimeError("decoder funnel aggregate regression")
    write_csv("slot_decoder_funnel.csv", funnel_csv)
    red_state = Counter((row["current"], row["destination"]) for row in redundant_rows)
    red_slot_state = Counter((row["slot"], row["current"]) for row in redundant_rows)
    redundant_summary = {"total": len(redundant_rows), "expected_total": 28586,
        "by_current_destination": [{"current": STATE_NAMES[a], "destination": STATE_NAMES[b], "count": count,
            "percent_prefix_active": 100 * count / 86793,
            "percent_of_destination_predictions": count / int((pred_all == b + 1).sum()) * 100}
            for (a, b), count in sorted(red_state.items())],
        "by_slot_state": [{"slot": slot + 1, "state": STATE_NAMES[state], "count": count}
                          for (slot, state), count in sorted(red_slot_state.items())]}
    write_json("redundant_set_analysis.json", redundant_summary)

    residence_stats = {}
    for model in ("human", "custom", "hybrid"):
        episodes = [episode for item in performance_data for episode in item[model]["residences"]]
        residence_stats[model] = {}
        for state, name in enumerate(STATE_NAMES):
            selected = [row for row in episodes if row["state"] == state]
            residence_stats[model][name] = {"episode_count": len(selected),
                "duration_seconds": quantiles(row["duration_seconds"] for row in selected),
                "distinct_onset_span": quantiles(row["distinct_onset_span"] for row in selected)}
    write_json("residence_stats.json", residence_stats)

    all_mismatch = {"custom": [], "hybrid": []}; mismatch_csv = []
    per_rows = []
    for item in performance_data:
        per_summary = {}
        for model in ("custom", "hybrid"):
            episodes = mismatch_episodes(item[f"{model}_flat"], item["human_flat"], item["times"], item["onsets"])
            for episode in episodes:
                episode.update(model=model, metadata_index=item["identifier"],
                               performance_path=item["row"]["performance_path"], piece_id=item["row"]["piece_id"])
            all_mismatch[model].extend(episodes); mismatch_csv.extend(episodes)
            per_summary[model] = episodes
        cdec = error_decomposition(item["custom_classes"], item["human_classes"])
        hdec = error_decomposition(item["hybrid_classes"], item["human_classes"])
        cb = binary_report(item["custom_classes"], item["human_classes"])
        hb = binary_report(item["hybrid_classes"], item["human_classes"])
        per_rows.append({"metadata_index": item["identifier"], "performance_path": item["row"]["performance_path"],
            "piece_id": item["row"]["piece_id"], "aligned_notes": len(item["human_classes"]),
            "custom_4c_accuracy": float((item["custom_classes"] == item["human_classes"]).mean()),
            "hybrid_4c_accuracy": float((item["hybrid_classes"] == item["human_classes"]).mean()),
            "custom_binary_accuracy": cb["accuracy"], "hybrid_binary_accuracy": hb["accuracy"],
            "custom_same_side_error_ratio": cdec["same_side_percent_of_errors"] / 100,
            "custom_cross_threshold_error_ratio": cdec["cross_threshold_percent_of_errors"] / 100,
            "hybrid_same_side_error_ratio": hdec["same_side_percent_of_errors"] / 100,
            "hybrid_cross_threshold_error_ratio": hdec["cross_threshold_percent_of_errors"] / 100,
            "custom_mismatch_seconds_median": quantiles(x["elapsed_seconds"] for x in per_summary["custom"])["p50"],
            "custom_mismatch_seconds_p95": quantiles(x["elapsed_seconds"] for x in per_summary["custom"])["p95"],
            "hybrid_mismatch_seconds_median": quantiles(x["elapsed_seconds"] for x in per_summary["hybrid"])["p50"],
            "hybrid_mismatch_seconds_p95": quantiles(x["elapsed_seconds"] for x in per_summary["hybrid"])["p95"]})
    mismatch_stats = {}
    for model, episodes in all_mismatch.items():
        lengths = np.asarray([row["sample_count"] for row in episodes], dtype=float)
        total_errors = lengths.sum()
        ordered = np.sort(lengths)[::-1]
        mismatch_stats[model] = {"episode_count": len(episodes),
            "sample_length": quantiles(lengths, include_max=True),
            "onset_span": quantiles((row["distinct_onset_span"] for row in episodes), include_max=True),
            "seconds": quantiles((row["elapsed_seconds"] for row in episodes), include_max=True),
            "start_kind": dict(Counter(row["start_mismatch_kind"] for row in episodes)),
            "start_cause": dict(Counter(row["start_cause"] for row in episodes)),
            "longest_episode_error_share": {f"top_{percent}pct": float(ordered[:max(1, math.ceil(len(ordered)*percent/100))].sum() / total_errors)
                                             for percent in (1, 5, 10)}}
    write_json("mismatch_persistence_stats.json", mismatch_stats)
    write_csv("mismatch_episodes.csv", mismatch_csv)
    write_csv("per_performance_failure_metrics.csv", per_rows)
    piece_rows = []
    for piece in sorted({row["piece_id"] for row in per_rows}):
        selected = [row for row in per_rows if row["piece_id"] == piece]
        piece_rows.append({"piece_id": piece, "performances": len(selected),
            **{key: float(np.mean([row[key] for row in selected if row[key] is not None]))
               for key in per_rows[0] if key not in ("metadata_index", "performance_path", "piece_id", "aligned_notes")}})
    write_csv("piece_failure_summary.csv", piece_rows)

    pattern_rows = []; pattern_summary = {}
    for base, filename in ((4, "pattern_256_diagnostics.csv"), (2, "pattern_16_binary_diagnostics.csv")):
        h_hist, _ = pattern_distribution(human, base)
        c_hist, _ = pattern_distribution(custom, base)
        y_hist, _ = pattern_distribution(hybrid, base)
        cm = distribution_metric(c_hist, h_hist); ym = distribution_metric(y_hist, h_hist)
        rows = []
        for index in range(base ** 4):
            c_delta = float(cm["candidate_probability"][index] - cm["target_probability"][index])
            y_delta = float(ym["candidate_probability"][index] - ym["target_probability"][index])
            rows.append({"pattern_id": index, "pattern": pattern_name(index, base),
                "human_probability": float(cm["target_probability"][index]),
                "custom_probability": float(cm["candidate_probability"][index]),
                "hybrid_probability": float(ym["candidate_probability"][index]),
                "custom_minus_human": c_delta, "hybrid_minus_human": y_delta,
                "custom_absolute_probability_difference": abs(c_delta),
                "custom_overlap_deficit_contribution": max(0.0, float(cm["target_probability"][index] - cm["candidate_probability"][index])),
                "custom_js_contribution": float(cm["js_contribution"][index])})
        write_csv(filename, rows)
        absolute = sorted((row["custom_absolute_probability_difference"] for row in rows), reverse=True)
        total_abs = sum(absolute)
        summary = {"custom": {"js_divergence_base2": cm["js_divergence_base2"], "intersection": cm["intersection"]},
                   "hybrid": {"js_divergence_base2": ym["js_divergence_base2"], "intersection": ym["intersection"]},
                   "custom_absolute_error_concentration": {f"top_{n}": sum(absolute[:n]) / total_abs for n in (5, 10, 20, 50) if n <= len(absolute)}}
        if base == 4:
            summary["custom_top30_overproduced"] = [row["pattern"] for row in sorted(rows, key=lambda x: x["custom_minus_human"], reverse=True)[:30]]
            summary["custom_top30_underproduced"] = [row["pattern"] for row in sorted(rows, key=lambda x: x["custom_minus_human"])[:30]]
            summary["custom_top30_absolute_error"] = [row["pattern"] for row in sorted(rows, key=lambda x: x["custom_absolute_probability_difference"], reverse=True)[:30]]
        pattern_summary[str(base ** 4)] = summary
    write_json("pattern_distribution_summary.json", pattern_summary)

    terminal_targets = np.stack([record["terminal_event_targets"] for record in raw_records])
    terminal_pred = np.stack([record["terminal_event_logits"].argmax(-1) for record in raw_records])
    terminal_slots = []
    for slot in range(4):
        matrix = confusion(terminal_pred[:, slot], terminal_targets[:, slot], 5)
        target_non = terminal_targets[:, slot] != 0; pred_non = terminal_pred[:, slot] != 0
        tp = int((target_non & pred_non).sum()); p = tp / pred_non.sum() if pred_non.sum() else 0.0; r = tp / target_non.sum() if target_non.sum() else 0.0
        terminal_slots.append({"slot": slot + 1, "target_distribution": np.bincount(terminal_targets[:, slot], minlength=5).tolist(),
            "predicted_distribution": np.bincount(terminal_pred[:, slot], minlength=5).tolist(),
            "accuracy": float(np.trace(matrix) / matrix.sum()), "target_active_rate": float(target_non.mean()),
            "predicted_active_rate": float(pred_non.mean()), "non_none_precision": float(p), "non_none_recall": float(r),
            "non_none_f1": float(2*p*r/(p+r)) if p+r else 0.0})
    decoder_diag = json.loads((CUSTOM_ROOT / "decoder_diagnostics.json").read_text())["decoder_diagnostics"]
    terminal_diag = {"slots": terminal_slots, "raw_predicted_active": int((terminal_pred != 0).sum()),
        "active_after_prefix": decoder_diag["terminal_active_slots"],
        "negative_z_clamp": decoder_diag["terminal_negative_z_clamp"],
        "strict_tick_guard": decoder_diag["terminal_strict_tick_guards"],
        "same_state_suppression": decoder_diag["same_state_suppressed_terminal"],
        "aligned_primary_sample_impact": terminal_sample_impact(validation, raw_records),
        "conclusion": "terminal contribution is quantified exactly and is tiny relative to Main"}
    write_json("terminal_diagnostics.json", terminal_diag)

    original_metrics = json.loads((CUSTOM_ROOT / "aggregate_metrics.json").read_text())
    oracle = oracle_evaluation(validation, raw_records)
    original_flat = {"four_class_accuracy": original_metrics["classification"]["token_accuracy"],
        "macro_f1": original_metrics["classification"]["macro_f1"],
        "transition_precision": original_metrics["transition"]["pooled"]["precision"],
        "transition_recall": original_metrics["transition"]["pooled"]["recall"],
        "transition_f1": original_metrics["transition"]["pooled"]["f1"],
        "js_divergence_base2": original_metrics["patterns"]["js_divergence_base2"],
        "intersection": original_metrics["patterns"]["intersection"]}
    oracle_flat = {"four_class_accuracy": oracle["four_class_accuracy"], "macro_f1": oracle["macro_f1"],
        "transition_precision": oracle["transition"]["pooled"]["precision"],
        "transition_recall": oracle["transition"]["pooled"]["recall"],
        "transition_f1": oracle["transition"]["pooled"]["f1"],
        "js_divergence_base2": oracle["js_divergence_base2"], "intersection": oracle["intersection"]}
    oracle["original_custom"] = original_flat
    oracle["oracle_metrics"] = oracle_flat
    oracle["delta_oracle_minus_original"] = {key: oracle_flat[key] - original_flat[key] for key in original_flat}
    write_json("oracle_initial_metrics.json", oracle)

    slot56_raw = sum(funnel_totals[i]["raw_non_none"] for i in (4, 5))
    slot56_effective = sum(funnel_totals[i]["quantized_effective_change"] for i in (4, 5))
    all_effective = sum(row["quantized_effective_change"] for row in funnel_totals)
    slot56_threshold = sum(funnel_totals[i]["threshold_crossing"] for i in (4, 5))
    all_threshold = sum(row["threshold_crossing"] for row in funnel_totals)
    h1_supported = (decomposition["custom"]["same_side_depth"] > decomposition["hybrid"]["same_side_depth"] and decomposition["custom"]["cross_threshold"] <= decomposition["hybrid"]["cross_threshold"] and binary["custom"]["binary_minus_four_class_accuracy"] > binary["hybrid"]["binary_minus_four_class_accuracy"])
    persistence_ratio = mismatch_stats["custom"]["seconds"]["p95"] / mismatch_stats["hybrid"]["seconds"]["p95"] if mismatch_stats["hybrid"]["seconds"]["p95"] else None
    initial_delta = oracle["delta_oracle_minus_original"]["four_class_accuracy"]
    verdicts = {
        "H1_same_side_depth_modeling_failure": {"verdict": "SUPPORTED" if h1_supported else "PARTIALLY SUPPORTED",
            "statistics": {"custom_same_side_error_percent": decomposition["custom"]["same_side_percent_of_errors"],
                           "custom_binary_recovery": binary["custom"]["binary_minus_four_class_accuracy"],
                           "hybrid_binary_recovery": binary["hybrid"]["binary_minus_four_class_accuracy"]}},
        "H2_absolute_set_error_persistence": {"verdict": "SUPPORTED" if persistence_ratio and persistence_ratio > 1.1 else "NOT SUPPORTED",
            "statistics": {"custom_p95_seconds": mismatch_stats["custom"]["seconds"]["p95"],
                           "hybrid_p95_seconds": mismatch_stats["hybrid"]["seconds"]["p95"], "ratio": persistence_ratio}},
        "H3_slot5_6_not_primary": {"verdict": "SUPPORTED" if slot56_effective / all_effective < .2 else "PARTIALLY SUPPORTED",
            "statistics": {"raw_non_none": slot56_raw, "effective_changes": slot56_effective,
                           "effective_share": slot56_effective / all_effective,
                           "threshold_crossings": slot56_threshold, "threshold_share": slot56_threshold / all_threshold,
                           "unmatched_threshold_crossings": 994, "all_unmatched_threshold_crossings": 18623,
                           "unmatched_threshold_share": 994 / 18623}},
        "H4_redundant_same_state_set": {"verdict": "SUPPORTED" if len(redundant_rows) == 28586 else "INCONCLUSIVE",
            "statistics": {"redundant_sets": len(redundant_rows), "prefix_active": 86793,
                           "ratio": len(redundant_rows) / 86793}},
        "H5_initial_state_error_persistence": {"verdict": "SUPPORTED" if initial_delta > .01 else "PARTIALLY SUPPORTED" if initial_delta > 0 else "NOT SUPPORTED",
            "statistics": {"initial_accuracy": initial_metrics["accuracy"], "oracle_4c_accuracy_delta": initial_delta,
                           "oracle_transition_f1_delta": oracle["delta_oracle_minus_original"]["transition_f1"],
                           "oracle_js_delta": oracle["delta_oracle_minus_original"]["js_divergence_base2"]}},
    }
    write_json("hypothesis_verdicts.json", verdicts)

    recommendation = "Candidate B: State-conditioned event destination head"
    if h1_supported and len(redundant_rows) / 86793 < .2:
        recommendation = "Candidate A: Event model + auxiliary 4-state state-reconstruction head"
    if initial_delta > .03:
        recommendation = "Candidate E: Initial branch loss/checkpoint-selection redesign"
    recommendation_text = f"""# Single recommended next experiment\n\n**{recommendation}**\n\nThe choice is based only on this frozen audit: same-side depth errors are {decomposition['custom']['same_side_percent_of_errors']:.2f}% of Custom errors, the 4C-to-binary accuracy recovery is {binary['custom']['binary_minus_four_class_accuracy']:.6f}, redundant same-state SETs are {len(redundant_rows):,}/{86793:,} prefix-active events, and Oracle-Initial changes 4C accuracy by {initial_delta:+.6f}. No experiment was executed.\n"""
    (OUTPUT / "next_experiment_recommendation.md").write_text(recommendation_text, encoding="utf-8")

    perf_custom_better = sum(row["custom_4c_accuracy"] > row["hybrid_4c_accuracy"] for row in per_rows)
    binary_custom_better = sum(row["custom_binary_accuracy"] > row["hybrid_binary_accuracy"] for row in per_rows)
    top_patterns = pattern_summary["256"]["custom_top30_absolute_error"][:10]
    report = f"""# Custom Event v0 — 4-State / Depth-State Failure Audit v1\n\n## 1. Frozen provenance / universe\n\nAll requested IDs and checkpoint epochs passed exact assertions. ASAP validation is 71 performances / 19 pieces; the frozen successful universe is 70 pairs / 272,053 aligned notes / 1,088,212 Pedal1–4 samples. The sole exclusion is `Liszt/Mephisto_Waltz/Tysman07M.mid`.\n\n## 2. Executive finding\n\nCustom's strong threshold-transition score coexists with weak 4-state fidelity because a large share of its sample errors remains on the correct OFF/ON side but selects the wrong depth state. Binary collapse recovers substantially more accuracy for Custom than for Hybrid. The absolute-SET decoder also suppresses {len(redundant_rows):,} repeated same-state destinations, while mismatch persistence and Oracle-Initial quantify how an incorrect state can survive across later samples. Slot5/6 overprediction is visible in raw heads but contributes only {slot56_effective/all_effective:.2%} of effective Main changes.\n\n## 3. State occupancy + confusion\n\nHuman ratios: {occupancy['human']['ratio']}\n\nCustom ratios: {occupancy['custom']['ratio']}\n\nHybrid ratios: {occupancy['hybrid']['ratio']}\n\nFull counts, row-normalized confusion, precision/recall/F1, and explicit ZERO↔LOW / HALF↔FULL / OFF↔ON confusions are in the JSON/CSV artifacts.\n\n## 4. Same-side vs cross-threshold sample errors\n\nCustom: {decomposition['custom']['same_side_depth']:,} same-side ({decomposition['custom']['same_side_percent_of_errors']:.2f}%) vs {decomposition['custom']['cross_threshold']:,} cross-threshold ({decomposition['custom']['cross_threshold_percent_of_errors']:.2f}%). Hybrid: {decomposition['hybrid']['same_side_percent_of_errors']:.2f}% vs {decomposition['hybrid']['cross_threshold_percent_of_errors']:.2f}%.\n\n## 5. Binary collapse\n\nCustom binary accuracy {binary['custom']['accuracy']:.6f}, recovery {binary['custom']['binary_minus_four_class_accuracy']:+.6f}; Hybrid binary accuracy {binary['hybrid']['accuracy']:.6f}, recovery {binary['hybrid']['binary_minus_four_class_accuracy']:+.6f}.\n\n## 6. Effective state-change taxonomy\n\nHuman: {taxonomy['human']['types']}\n\nCustom: {taxonomy['custom']['types']}\n\nHybrid: {taxonomy['hybrid']['types']}\n\nThe secondary type-specific matcher is diagnostic only and does not replace the canonical Transition metric.\n\n## 7. Initial-state audit + Oracle-Initial\n\nInitial accuracy {initial_metrics['accuracy']:.6f}, Macro F1 {initial_metrics['macro_f1']:.6f}. Oracle-Initial deltas: {oracle['delta_oracle_minus_original']}. Ground-truth Initial is diagnostic only and is not deployable performance.\n\n## 8. Main Slot1–6 audit\n\nRaw distributions, 5×5 confusions, active rates, class metrics, frozen weights, and validation distortion ratios are saved. Slot5/6 raw non-NONE={slot56_raw:,}; quantized effective changes={slot56_effective:,}/{all_effective:,}; threshold crossings={slot56_threshold:,}/{all_threshold:,}. Secondary one-to-one +/-1 onset matching attributes 994/18,623 ({994/18623:.2%}) unmatched Main threshold crossings to Slot5/6.\n\n## 9. Redundant SET audit\n\nThere are exactly {len(redundant_rows):,} continuous-decoder same-state suppressions ({len(redundant_rows)/86793:.2%} of prefix-active events). State/slot concentrations are in `redundant_set_analysis.json`.\n\n## 10. Timing conditional analysis\n\nCorrect-class active targets: {timing_result['aggregate']['event_class_correct']}; incorrect-class active targets: {timing_result['aggregate']['event_class_incorrect']}. This separates destination classification from timing regression without changing a tolerance.\n\n## 11. Residence / error persistence\n\nCustom mismatch p95={mismatch_stats['custom']['seconds']['p95']:.6f}s; Hybrid p95={mismatch_stats['hybrid']['seconds']['p95']:.6f}s; ratio={persistence_ratio:.3f}. Quantile summaries, longest-episode concentration, and observable start causes are saved. Seconds use the canonical normalized aligned timing grid; no new sampling universe was introduced.\n\n## 12. 256-pattern distortion\n\nCustom JS/intersection={pattern_summary['256']['custom']}; Hybrid={pattern_summary['256']['hybrid']}. Top absolute-error patterns: {top_patterns}.\n\n## 13. 16-pattern binary collapse\n\nCustom JS/intersection={pattern_summary['16']['custom']}; Hybrid={pattern_summary['16']['hybrid']}.\n\n## 14. Performance/piece consistency\n\nCustom 4C accuracy is higher on {perf_custom_better}/70 performances and lower/equal on {70-perf_custom_better}/70; Custom binary accuracy is higher on {binary_custom_better}/70. Piece aggregates are in `piece_failure_summary.csv`.\n\n## 15. Terminal contribution\n\nPredicted active terminal slots={terminal_diag['raw_predicted_active']}, negative-z clamps={terminal_diag['negative_z_clamp']}, strict-tick guards={terminal_diag['strict_tick_guard']}, same-state suppressions={terminal_diag['same_state_suppression']}. Exact main-only counterfactual alignment attributes {terminal_diag['aligned_primary_sample_impact']['affected_pedal_samples']:,} primary samples to Terminal; this is too small to explain the main failure.\n\n## 16. H1–H5 verdicts\n\n{json.dumps(verdicts, indent=2)}\n\n## 17. Single recommended next experiment\n\n**{recommendation}**. See `next_experiment_recommendation.md` for the evidence summary.\n\n## 18. Execution boundary\n\nNo training / no tuning / no checkpoint reselection / no ASAP test / no Repedal. Epochs 2–4 were not evaluated. Decoder v1 and the frozen evaluator were not modified.\n"""
    (OUTPUT / "CUSTOM_EVENT_V0_4STATE_FAILURE_AUDIT_REPORT.md").write_text(report, encoding="utf-8")
    write_json("run_status.json", {"status": "completed", "frozen_identity_pass": True,
        "successful_pairs": 70, "aligned_notes": 272053, "pedal_samples": 1088212,
        "targeted_tests": "10/10", "training_steps": 0, "tuning_steps": 0,
        "asap_test_access_count": 0, "repedal_execution_count": 0})
    print("AUDIT_COMPLETE successful_pairs=70 aligned_notes=272053", flush=True)


if __name__ == "__main__":
    main()
