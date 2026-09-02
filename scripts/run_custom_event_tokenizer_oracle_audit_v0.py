#!/usr/bin/env python3
"""Run the no-training Custom Event Tokenizer v0 structural/oracle audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from miditoolkit import MidiFile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_pedal_event_metric_tolerance_mini import (
    extract_transitions,
    raw_events,
    tick_ms_fn,
)
from src.stage2_event_tokenizer.audit import StructuralAccumulator, oracle_aggregate
from src.stage2_event_tokenizer.decoder import write_roundtrip_midi
from src.stage2_event_tokenizer.tokenizer import (
    EVENT_NAMES,
    REPRESENTATIVES,
    STATE_NAMES,
    EncodedPerformance,
    encode_performance,
    parse_raw_midi,
    quantize_cc64,
)
from src.stage2_four_class.validation_evaluator import (
    PATTERN_COUNT,
    confusion_from_pairs,
    pattern_ids,
    pooled_transition_counts,
)
from third_party.PianistTransformer.src.model.pianoformer import PianoT5GemmaConfig
from third_party.PianistTransformer.src.utils.midi import midi_to_ids

TRAIN_MANIFEST = ROOT / "analysis/stage2_binary_v0/data_prep_v1/train_manifest.csv"
SPLIT_MANIFEST = ROOT / "analysis/stage2_encoder_only_v0/asap_split.csv"
ASAP_ROOT = Path("/workspace/public/ASAP/asap-dataset-v1.1")
DEFAULT_OUTPUT = ROOT / "analysis/custom_event_tokenizer_oracle_audit_v0"
SOURCE_NAMES = ("MAESTRO-clean", "ASAP-train", "combined")


def now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifests() -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    train = read_csv(TRAIN_MANIFEST)
    if len(train) != 2062 or Counter(row["source"] for row in train) != Counter(
        {"MAESTRO-clean": 1170, "ASAP-train": 892}
    ):
        raise RuntimeError("canonical training manifest is not 1170+892")
    if any(not Path(row["performance_absolute_path"]).is_file() for row in train):
        raise FileNotFoundError("canonical training manifest contains a missing source MIDI")
    # Split metadata is the immutable membership source; only validation paths are opened.
    split = read_csv(SPLIT_MANIFEST)
    validation = [row for row in split if row["split"] == "validation"]
    if len(validation) != 71 or len({row["piece_id"] for row in validation}) != 19:
        raise RuntimeError("canonical validation universe is not 19 pieces / 71 performances")
    for row in validation:
        row["performance_absolute_path"] = str(ASAP_ROOT / row["performance_path"])
        if not Path(row["performance_absolute_path"]).is_file():
            raise FileNotFoundError(row["performance_absolute_path"])
    provenance = {
        "train_manifest": str(TRAIN_MANIFEST),
        "train_manifest_sha256": sha256(TRAIN_MANIFEST),
        "validation_membership_manifest": str(SPLIT_MANIFEST),
        "validation_membership_manifest_sha256": sha256(SPLIT_MANIFEST),
        "raw_cc64_source": "original performance MIDI control_change 64 messages via mido; no Pedal1-4 target/cache reconstruction",
        "mido_order": "per-track file order preserved; equal-tick global deterministic key=(tick,track_index,message_index); last effective CC64 state wins",
        "cross_track_same_tick_limitation": "SMF defines no semantic cross-track ordering; counted as ambiguous where CC and note-on are in different tracks",
        "test_midi_access_count": 0,
    }
    return train, validation, provenance


def terminal_row(dataset: str, key: str, encoded: EncodedPerformance) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "performance": key,
        "raw_cc64_events": len(encoded.source.raw_cc64),
        "effective_4state_events": len(encoded.effective_events),
        "raw_after_last_onset": encoded.raw_events_after_last_onset,
        "effective_after_last_onset": encoded.effective_events_after_last_onset,
        "raw_after_latest_noteoff": encoded.raw_events_after_note_off,
        "effective_after_latest_noteoff": encoded.effective_events_after_note_off,
        "has_post_noteoff_effective": int(encoded.effective_events_after_note_off > 0),
        "terminal_denominator_nonpositive": int(encoded.terminal_denominator_nonpositive),
        "raw_exactly_on_onset": encoded.raw_events_exactly_on_onset,
        "effective_exactly_on_onset": encoded.effective_events_exactly_on_onset,
        "first_onset_tau0": encoded.first_onset_tau0_event_count,
        "cc_before_note_same_tick": encoded.same_time_cc_before_note,
        "cc_after_note_same_tick": encoded.same_time_cc_after_note,
        "cc_note_same_tick_ambiguous": encoded.same_time_order_ambiguous,
    }


def structural_audit(
    train: list[dict[str, str]], output: Path
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    accumulators = {name: StructuralAccumulator(name) for name in SOURCE_NAMES}
    performance_rows: list[dict[str, Any]] = []
    terminal_rows: list[dict[str, Any]] = []
    progress = output / "audit.log"
    started = time.monotonic()
    for index, row in enumerate(train, 1):
        source = row["source"]
        encoded = encode_performance(parse_raw_midi(row["performance_absolute_path"]))
        perf = accumulators[source].add(encoded)
        accumulators["combined"].add(encoded)
        perf.update(dataset=source, performance_path=row["performance_path"])
        performance_rows.append(perf)
        terminal_rows.append(terminal_row(source, row["performance_path"], encoded))
        if index == 1 or index % 50 == 0 or index == len(train):
            message = f"{now()} structural {index}/{len(train)} elapsed={time.monotonic()-started:.1f}s"
            print(message, flush=True)
            with progress.open("a", encoding="utf-8") as handle:
                handle.write(message + "\n")
    payload = {name: accumulators[name].payload() for name in SOURCE_NAMES}
    if any(value["final_state_preservation"] != 1.0 for value in payload.values()):
        raise AssertionError("final-state preservation is not exact 100%")
    return payload, performance_rows, terminal_rows


def class_arrays(path: Path, config: PianoT5GemmaConfig) -> tuple[np.ndarray, np.ndarray]:
    tokens = np.asarray(midi_to_ids(config, MidiFile(str(path))), dtype=np.int64)
    if not len(tokens) or len(tokens) % 8:
        raise ValueError(f"official tokenizer returned invalid token sequence: {path}")
    notes = tokens.reshape(-1, 8)
    raw = notes[:, 4:8] - int(config.pedal_start)
    classes = np.vectorize(quantize_cc64, otypes=[np.int64])(raw)
    return raw.astype(np.int64), classes.astype(np.int64)


def transition_object(path: Path) -> dict[str, Any]:
    midi = MidiFile(str(path))
    onsets, controls = raw_events(midi)
    converter = tick_ms_fn(midi)
    transitions = extract_transitions(onsets, [converter(tick) for tick in onsets], controls, converter)
    for event in transitions:
        event["score_position"] = event["onset_index"]
    return {"transitions": transitions}


def add_transition_counts(target: dict[str, Any], metrics: dict[str, Any]) -> None:
    for direction in ("UP", "DOWN"):
        source = metrics[direction.lower()]
        for name in ("tp", "fp", "fn"):
            target[direction][name] += int(source[name])


def exact_crossings(encoded: EncodedPerformance, *, modeled_only: bool) -> Counter[tuple[int, str]]:
    result: Counter[tuple[int, str]] = Counter()
    first = encoded.source.distinct_onsets[0]
    end = encoded.source.latest_note_off
    previous = 0
    for raw in encoded.source.raw_cc64:
        current = 1 if raw.value >= 64 else 0
        direction = "DOWN" if previous == 0 and current == 1 else "UP" if previous == 1 and current == 0 else None
        if direction and (not modeled_only or first <= raw.tick <= end):
            result[(raw.tick, direction)] += 1
        previous = current
    return result


def exact_retention(source: Counter[tuple[int, str]], candidate: Counter[tuple[int, str]]) -> dict[str, int]:
    retained = sum((source & candidate).values())
    return {
        "source_crossings": sum(source.values()),
        "candidate_crossings": sum(candidate.values()),
        "exact_retained_source_crossings": retained,
        "dropped_source_crossings": sum(source.values()) - retained,
        "new_artificial_crossings": sum(candidate.values()) - retained,
    }


def oracle_audit(
    validation: list[dict[str, str]], output: Path
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    config = PianoT5GemmaConfig()
    if int(config.pedal_start) != 5261:
        raise RuntimeError("official Pedal1-4 token offset changed")
    oracle_names = ("unlimited", "two_slot")
    confusion = {name: np.zeros((4, 4), dtype=np.int64) for name in oracle_names}
    candidate_hist = {name: np.zeros(PATTERN_COUNT, dtype=np.int64) for name in oracle_names}
    target_hist = np.zeros(PATTERN_COUNT, dtype=np.int64)
    transition_counts = {
        name: {direction: {key: 0 for key in ("tp", "fp", "fn")} for direction in ("UP", "DOWN")}
        for name in oracle_names
    }
    absolute_errors = {name: [] for name in oracle_names}
    exact_totals = {
        name: {
            support: Counter(
                source_crossings=0, candidate_crossings=0,
                exact_retained_source_crossings=0, dropped_source_crossings=0,
                new_artificial_crossings=0,
            )
            for support in ("full_source", "modeled_support")
        }
        for name in oracle_names
    }
    rows: list[dict[str, Any]] = []
    terminal_rows: list[dict[str, Any]] = []
    progress = output / "audit.log"
    for index, row in enumerate(validation, 1):
        source_path = Path(row["performance_absolute_path"])
        encoded = encode_performance(parse_raw_midi(source_path))
        key = f"{int(row['metadata_index']):04d}_{source_path.stem}"
        unlimited_path = write_roundtrip_midi(encoded, output / "roundtrip_unlimited" / f"{key}.mid", two_slot=False)
        two_slot_path = write_roundtrip_midi(encoded, output / "roundtrip_two_slot" / f"{key}.mid", two_slot=True)
        source_raw, source_classes = class_arrays(source_path, config)
        target_hist += np.bincount(pattern_ids(source_classes), minlength=PATTERN_COUNT)
        source_transition = transition_object(source_path)
        source_full = exact_crossings(encoded, modeled_only=False)
        source_modeled = exact_crossings(encoded, modeled_only=True)
        item: dict[str, Any] = {
            "metadata_index": row["metadata_index"],
            "piece_id": row["piece_id"],
            "performance_path": row["performance_path"],
            "source_note_count": len(source_classes),
        }
        for name, candidate_path in (("unlimited", unlimited_path), ("two_slot", two_slot_path)):
            raw, classes = class_arrays(candidate_path, config)
            if classes.shape != source_classes.shape:
                raise AssertionError("source/round-trip official token shapes differ")
            matrix = confusion_from_pairs(classes, source_classes)
            confusion[name] += matrix
            candidate_hist[name] += np.bincount(pattern_ids(classes), minlength=PATTERN_COUNT)
            absolute_errors[name].extend(np.abs(raw - source_raw).reshape(-1).tolist())
            transition = pooled_transition_counts(transition_object(candidate_path), source_transition)
            add_transition_counts(transition_counts[name], transition)
            candidate_encoded = encode_performance(parse_raw_midi(candidate_path))
            candidate_full = exact_crossings(candidate_encoded, modeled_only=False)
            candidate_modeled = exact_crossings(candidate_encoded, modeled_only=True)
            for support, source_crossings, candidate_crossings in (
                ("full_source", source_full, candidate_full),
                ("modeled_support", source_modeled, candidate_modeled),
            ):
                exact_totals[name][support].update(
                    exact_retention(source_crossings, candidate_crossings)
                )
            classification = matrix.trace() / matrix.sum()
            item.update({
                f"{name}_4c_accuracy": float(classification),
                f"{name}_transition_tp": int(transition["pooled"]["tp"]),
                f"{name}_transition_fp": int(transition["pooled"]["fp"]),
                f"{name}_transition_fn": int(transition["pooled"]["fn"]),
                f"{name}_raw_cc64_mae": float(np.mean(np.abs(raw - source_raw))),
            })
        rows.append(item)
        terminal_rows.append(terminal_row("ASAP-validation", row["performance_path"], encoded))
        message = f"{now()} oracle {index}/{len(validation)} {row['performance_path']}"
        print(message, flush=True)
        with progress.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")
    metrics = {
        name: oracle_aggregate(confusion[name], candidate_hist[name], target_hist, transition_counts[name], absolute_errors[name])
        for name in oracle_names
    }
    metrics["source_pattern_distribution"] = (target_hist / target_hist.sum()).tolist()
    metrics["exact_transition_retention"] = {
        name: {support: dict(values) for support, values in supports.items()}
        for name, supports in exact_totals.items()
    }
    metrics["top_source_patterns"] = [
        {"pattern_id": int(index), "probability": float(target_hist[index] / target_hist.sum())}
        for index in np.argsort(target_hist)[::-1][:20]
    ]
    return metrics, rows, terminal_rows


def distribution_rows(structural: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, payload in structural.items():
        for state in STATE_NAMES:
            rows.append({"dataset": dataset, "distribution": "initial_state", "slot": "initial", "label": state, "count": payload["initial_state_counts"][state], "proportion": payload["initial_state_proportions"][state], "candidate_weight": payload["initial_candidate_weights_inv_sqrt_mean1"][state]})
        for slot in ("Slot1", "Slot2"):
            for event in EVENT_NAMES:
                rows.append({"dataset": dataset, "distribution": "slot_label", "slot": slot, "label": event, "count": payload["slot_counts"][slot][event], "proportion": payload["slot_proportions"][slot][event], "candidate_weight": payload["slot_candidate_weights_inv_sqrt_mean1"][slot][event]})
    return rows


def metric_row(name: str, value: dict[str, Any]) -> str:
    transition = value["transition"]["pooled"]
    pattern = value["pattern"]
    return f"| {name} | {value['token_accuracy']:.6f} | {value['macro_f1']:.6f} | {transition['precision']:.6f} | {transition['recall']:.6f} | {transition['f1']:.6f} | {pattern['js_divergence_base2']:.6f} | {pattern['intersection']:.6f} |"


def build_spec(output: Path, provenance: dict[str, Any]) -> None:
    text = f"""# Custom Event Tokenizer Specification v0

- Raw source: original performance MIDI CC64 timestamp/value messages; Pedal1–4 samples are never used to reconstruct events.
- Parser ordering: `{provenance['mido_order']}`. Cross-track equal-tick ordering is deterministic for audit repeatability but semantically ambiguous.
- States: ZERO 0–25, LOW 26–63, HALF 64–103, FULL 104–127; representatives `{list(REPRESENTATIVES)}`.
- Vocabulary: `NONE / SET_ZERO / SET_LOW / SET_HALF / SET_FULL` (absolute destination state).
- Equal timestamp: stable source order, then last effective CC64 state; quantized same-state changes are suppressed.
- Timeline: distinct non-pedal note onsets; right-open `[t_i,t_(i+1))`; event at an onset belongs to that onset's interval with tau=0.
- Initial state: state immediately before the first onset (`event_time < first_onset`); MIDI default ZERO.
- Terminal interval: `[last_onset, latest_note_off]`; later events are unsupported and audited.
- Tau: exact raw float ratio, no discretization.
- Slots: 0→NONE/NONE; 1→E1/NONE; 2→E1/E2; odd 3+→last/NONE; even 4+→penultimate/last.
- Decoder: deterministic chronological SET application, no smoothing/snapping/randomization.
- Initial serialization: representative CC at `max(0, first_onset-1)` and ordered before any tau=0 SET/note-on at tick zero.
- Training/model/optimizer/checkpoint activity: zero.
- ASAP test MIDI/cache/candidate access: zero. Repedal executions: zero.
"""
    (output / "CUSTOM_EVENT_TOKENIZER_SPEC_V0.md").write_text(text, encoding="utf-8")


def build_report(
    output: Path,
    structural: dict[str, Any],
    metrics: dict[str, Any],
    terminal: list[dict[str, Any]],
    overflow_rows: list[dict[str, Any]],
    provenance: dict[str, Any],
) -> None:
    combined = structural["combined"]
    unlimited, two = metrics["unlimited"], metrics["two_slot"]
    post = [row for row in terminal if row["has_post_noteoff_effective"]]
    same_raw = sum(row["raw_exactly_on_onset"] for row in terminal)
    same_effective = sum(row["effective_exactly_on_onset"] for row in terminal)
    raw_total = sum(row["raw_cc64_events"] for row in terminal)
    effective_total = sum(row["effective_4state_events"] for row in terminal)
    ambiguous = sum(row["cc_note_same_tick_ambiguous"] for row in terminal)
    training_ready = (
        combined["final_state_preservation"] == 1.0
        and two["transition"]["pooled"]["f1"] >= unlimited["transition"]["pooled"]["f1"] - 1e-12
        and not post
    )
    recommendation = "ready" if training_ready else "needs targeted design revision"
    lines = [
        "# Custom Event Tokenizer Oracle Audit v0", "",
        "## Scope and provenance", "",
        f"- Train structural universe: 2,062 original performance MIDIs (MAESTRO-clean 1,170 + ASAP train 892).",
        f"- Oracle universe: ASAP validation 71 performances / 19 pieces, same-performance self-roundtrip.",
        f"- Raw provenance: {provenance['raw_cc64_source']}.",
        f"- Same-tick rule: {provenance['mido_order']}.",
        "- ASAP test access: **0**; Repedal execution: **0**; training/model/optimizer/checkpoint activity: **0**.", "",
        "## Structural compression", "",
        "| Dataset | Intervals | 0 events | 1 | 2 | 3 | 4 | 5+ | ≥3 fraction | Event retention | Crossing retention |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset in SOURCE_NAMES:
        value = structural[dataset]; counts = value["precompression_interval_event_counts"]
        lines.append(f"| {dataset} | {value['intervals']} | {counts['0']} | {counts['1']} | {counts['2']} | {counts['3']} | {counts['4']} | {counts['5+']} | {value['ge3_interval_fraction']:.6f} | {value['event_retention_fraction']:.6f} | {value['crossing_retention_fraction']:.6f} |")
    lines += ["", "## Initial state", "", "| Dataset | ZERO | LOW | HALF | FULL |", "| --- | ---: | ---: | ---: | ---: |"]
    for dataset in SOURCE_NAMES:
        counts = structural[dataset]["initial_state_counts"]
        lines.append(f"| {dataset} | {counts['ZERO']} | {counts['LOW']} | {counts['HALF']} | {counts['FULL']} |")
    lines += ["", "## Combined slot distributions", "", "| Slot | NONE | SET_ZERO | SET_LOW | SET_HALF | SET_FULL |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for slot in ("Slot1", "Slot2"):
        counts = combined["slot_counts"][slot]
        lines.append(f"| {slot} | {counts['NONE']} | {counts['SET_ZERO']} | {counts['SET_LOW']} | {counts['SET_HALF']} | {counts['SET_FULL']} |")
    lines += ["", "## Worst overflow performances", "", "### Highest >=3-event interval fraction", "", "| Dataset | Performance | Intervals | Overflow intervals | Fraction |", "| --- | --- | ---: | ---: | ---: |"]
    for row in sorted(overflow_rows, key=lambda value: (-float(value["overflow_fraction"]), value["performance_path"]))[:10]:
        lines.append(f"| {row['dataset']} | `{row['performance_path']}` | {row['intervals']} | {row['overflow_intervals']} | {float(row['overflow_fraction']):.6f} |")
    lines += ["", "### Highest dropped-crossing fraction", "", "| Dataset | Performance | Source crossings | Dropped | Fraction |", "| --- | --- | ---: | ---: | ---: |"]
    for row in sorted(overflow_rows, key=lambda value: (-float(value["dropped_crossing_fraction"]), value["performance_path"]))[:10]:
        lines.append(f"| {row['dataset']} | `{row['performance_path']}` | {row['source_crossings']} | {row['dropped_crossings']} | {float(row['dropped_crossing_fraction']):.6f} |")
    lines += ["", "## Oracle ceiling", "", "| Oracle | 4C Acc ↑ | Macro F1 ↑ | Trans P ↑ | Trans R ↑ | Trans F1 ↑ | JS ↓ | Intersection ↑ |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |", metric_row("Unlimited Event", unlimited), metric_row("Actual Two-Slot", two), "", "## Slot-cap penalty", "", "| Metric | Unlimited | Two-Slot | Δ (Two−Unlimited) |", "| --- | ---: | ---: | ---: |"]
    for label, getter in (
        ("4C Accuracy", lambda x: x["token_accuracy"]),
        ("Transition F1", lambda x: x["transition"]["pooled"]["f1"]),
        ("JS", lambda x: x["pattern"]["js_divergence_base2"]),
        ("Intersection", lambda x: x["pattern"]["intersection"]),
    ):
        first, second = getter(unlimited), getter(two)
        lines.append(f"| {label} | {first:.6f} | {second:.6f} | {second-first:+.6f} |")
    lines += ["", "## Top source patterns and oracle probabilities", "", "| Pattern ID | Source | Unlimited | Two-Slot |", "| ---: | ---: | ---: | ---: |"]
    source_distribution = metrics["source_pattern_distribution"]
    unlimited_distribution = unlimited["pattern"]["candidate_distribution"]
    two_distribution = two["pattern"]["candidate_distribution"]
    for pattern in sorted(range(256), key=lambda index: source_distribution[index], reverse=True)[:10]:
        lines.append(f"| {pattern} | {source_distribution[pattern]:.6f} | {unlimited_distribution[pattern]:.6f} | {two_distribution[pattern]:.6f} |")
    lines += [
        "", "## Coverage and invariants", "",
        f"- Interval final-state preservation: **{combined['final_state_preservation']:.6%}** across {combined['final_state_checks']:,} interval checks (hard invariant PASS).",
        f"- Modeled effective events: {combined['effective_events_before_slot_cap_modeled_support']:,}; retained {combined['retained_events']:,}; dropped {combined['dropped_events']:,}.",
        f"- Source crossings retained as slots: {sum(combined['crossings_retained'].values()):,}/{sum(combined['crossings_before'].values()):,}.",
        f"- Post-latest-note-off effective events occur in {len(post):,} source performances across train+validation diagnostics; these events are outside v0 support.",
        f"- Exact-on-onset raw CC events: {same_raw:,}/{raw_total:,} ({same_raw/raw_total:.6%}); effective events: {same_effective:,}/{effective_total:,} ({same_effective/effective_total:.6%}).",
        f"- First-onset tau=0 effective events: {sum(row['first_onset_tau0'] for row in terminal):,}; cross-track/interleaved same-tick ambiguity count: {ambiguous:,}.",
        f"- Unlimited full-source exact transition diagnostic: `{metrics['exact_transition_retention']['unlimited']['full_source']}`.",
        f"- Two-slot modeled-support exact transition diagnostic: `{metrics['exact_transition_retention']['two_slot']['modeled_support']}`.",
        "", "## Answers", "",
        f"1. Absolute four-state SET representation ceiling is quantified by the Unlimited row; its remaining gap is quantization/tie/terminal behavior, not model error.",
        f"2. The two-slot cap retains {combined['event_retention_fraction']:.6%} of modeled events and {combined['crossing_retention_fraction']:.6%} of source crossing events as slots.",
        f"3. Overflow frequency is reported source-wise above and per performance in `overflow_diagnostics.csv`.",
        "4. The odd/even rule preserved every modeled interval final state exactly (100%).",
        f"5. Actual Two-Slot transition F1 ceiling is {two['transition']['pooled']['f1']:.6f}; exact diagnostics separate slot loss from unsupported terminal events.",
        "6. Initial-state imbalance and diagnostic inverse-sqrt weights are in the JSON/CSV; no loss was selected or applied.",
        "7. Slot1/Slot2 imbalance and candidate weights are diagnostic only.",
        "8. Exact tau summaries by slot/state, including boundary concentration, are in `train_structural_stats.json`.",
        f"9. Post-note-off coverage is explicit above and in `terminal_coverage_diagnostics.csv`.",
        f"10. Recommendation: **{recommendation}**. This is based on observed oracle/coverage facts; the tokenizer was not automatically redesigned.",
        "", "## Interpretation boundary", "",
        "Confirmed metrics/statistics are separated from the readiness recommendation. No arbitrary accuracy pass threshold, model implementation, loss, training, inference, ASAP test access, Repedal metric, or automatic tokenizer redesign was performed.",
    ]
    (output / "CUSTOM_EVENT_TOKENIZER_ORACLE_AUDIT_V0.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_root
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty audit root: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "roundtrip_unlimited").mkdir()
    (output / "roundtrip_two_slot").mkdir()
    train, validation, provenance = load_manifests()
    config_payload = {
        "experiment": "custom_event_tokenizer_oracle_audit_v0",
        "train_performances": 2062,
        "validation_performances": 71,
        "training_steps": 0,
        "model_inference_steps": 0,
        "optimizer_steps": 0,
        "checkpoint_access_count": 0,
        "asap_test_access_count": 0,
        "repedal_execution_count": 0,
        "provenance": provenance,
    }
    atomic_json(output / "config.json", config_payload)
    build_spec(output, provenance)
    structural, overflow_rows, terminal_train = structural_audit(train, output)
    atomic_json(output / "train_structural_stats.json", structural)
    write_csv(output / "train_event_distributions.csv", distribution_rows(structural), ["dataset", "distribution", "slot", "label", "count", "proportion", "candidate_weight"])
    write_csv(output / "overflow_diagnostics.csv", sorted(overflow_rows, key=lambda row: (-row["overflow_fraction"], -row["dropped_crossing_fraction"], row["performance_path"])), ["dataset", "performance_path", "intervals", "overflow_intervals", "overflow_fraction", "source_crossings", "dropped_crossings", "dropped_crossing_fraction"])
    metrics, validation_rows, terminal_validation = oracle_audit(validation, output)
    metrics.update({"validation_performances": 71, "same_performance_self_roundtrip": True, "asap_test_access_count": 0, "repedal_execution_count": 0})
    atomic_json(output / "validation_oracle_metrics.json", metrics)
    write_csv(output / "validation_per_performance_oracle.csv", validation_rows, list(validation_rows[0]))
    terminal_rows = terminal_train + terminal_validation
    write_csv(output / "terminal_coverage_diagnostics.csv", terminal_rows, list(terminal_rows[0]))
    build_report(output, structural, metrics, terminal_rows, overflow_rows, provenance)
    atomic_json(output / "run_status.json", {
        "status": "completed",
        "train_performances": 2062,
        "validation_performances": 71,
        "final_state_preservation": structural["combined"]["final_state_preservation"],
        "non_pedal_identity": "exact for all serialized validation roundtrips",
        "asap_test_access_count": 0,
        "repedal_execution_count": 0,
        "training_steps": 0,
        "last_update": now(),
    })
    print(json.dumps({"status": "completed", "output": str(output), "test_access": 0, "training_steps": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
