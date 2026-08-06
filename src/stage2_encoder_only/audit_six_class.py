"""Train-only audit of the fixed endpoint-aware six-class pedal encoding.

This command intentionally loads no model and never opens validation/test MIDI.
It reuses :class:`Stage2PedalDataset` so the pinned Pianist Transformer tokenizer
and its immutable in-memory performance cache are the source of all Pedal1--4
targets.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import socket
import subprocess
import time
import traceback
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from miditoolkit import MidiFile

from .dataset import NON_PEDAL_FEATURES, PEDAL_TOKEN_OFFSET, Stage2PedalDataset


CLASS_NAMES = ("ZERO", "LOW", "MID_LOW", "MID_HIGH", "HIGH", "FULL")
CLASS_RANGES = ((0, 0), (1, 31), (32, 63), (64, 95), (96, 126), (127, 127))
SLOT_NAMES = ("Pedal1", "Pedal2", "Pedal3", "Pedal4")
DEFAULT_ASAP_ROOT = Path("/workspace/public/ASAP/asap-dataset-v1.1")
DEFAULT_SPLIT_CSV = Path("/workspace/project/analysis/stage2_encoder_only_v0/asap_split.csv")
DEFAULT_OUTPUT = Path("/workspace/project/analysis/stage2_pedal_6class_audit_v0")
PRIMARY_REPEDAL_MAX_SAMPLES = 4
REPEDAL_SENSITIVITY_SAMPLES = (1, 2, 4, 8)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def finite(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_text(path: Path, text: str) -> None:
    temporary = path.parent / f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_text(path, json.dumps(finite(payload), ensure_ascii=False, indent=2) + "\n")


def atomic_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    temporary = path.parent / f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: finite(row.get(key)) for key in fieldnames})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_git_metadata(project_root: Path) -> dict[str, Any]:
    """Capture useful metadata without requiring git inside the container."""

    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "commit": None, "error": f"{type(exc).__name__}: {exc}"}
    if result.returncode:
        return {
            "available": False,
            "commit": None,
            "error": result.stderr.strip() or f"git exited {result.returncode}",
        }
    return {"available": True, "commit": result.stdout.strip(), "error": None}


def classify(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.size and (int(values.min()) < 0 or int(values.max()) > 127):
        raise ValueError("pedal value outside [0, 127]")
    classes = np.empty(values.shape, dtype=np.uint8)
    classes[values == 0] = 0
    classes[(values >= 1) & (values <= 31)] = 1
    classes[(values >= 32) & (values <= 63)] = 2
    classes[(values >= 64) & (values <= 95)] = 3
    classes[(values >= 96) & (values <= 126)] = 4
    classes[values == 127] = 5
    return classes


def weighted_quantile(values: np.ndarray, counts: np.ndarray, quantile: float) -> float | None:
    counts = np.asarray(counts, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)
    total = int(counts.sum())
    if total == 0:
        return None
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")

    def order_statistic(index: int) -> float:
        cumulative = np.cumsum(counts)
        return float(values[int(np.searchsorted(cumulative, index + 1, side="left"))])

    position = (total - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    low_value = order_statistic(lower)
    high_value = order_statistic(upper)
    return low_value + (high_value - low_value) * (position - lower)


def weighted_summary(values: np.ndarray, counts: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.int64)
    total = int(counts.sum())
    if total == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "q01": None,
            "q05": None,
            "q25": None,
            "q75": None,
            "q95": None,
            "q99": None,
        }
    positive = counts > 0
    mean = float(np.average(values, weights=counts))
    variance = float(np.average((values - mean) ** 2, weights=counts))
    return {
        "count": total,
        "mean": mean,
        "median": weighted_quantile(values, counts, 0.5),
        "std": math.sqrt(variance),
        "min": float(values[positive][0]),
        "max": float(values[positive][-1]),
        "q01": weighted_quantile(values, counts, 0.01),
        "q05": weighted_quantile(values, counts, 0.05),
        "q25": weighted_quantile(values, counts, 0.25),
        "q75": weighted_quantile(values, counts, 0.75),
        "q95": weighted_quantile(values, counts, 0.95),
        "q99": weighted_quantile(values, counts, 0.99),
    }


def vector_summary(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"count": 0, "mean": None, "median": None, "std": None, "min": None, "max": None,
                "q05": None, "q25": None, "q75": None, "q95": None}
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
        "q05": float(np.quantile(array, 0.05)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "q95": float(np.quantile(array, 0.95)),
    }


def histogram_class_counts(histogram: np.ndarray) -> np.ndarray:
    return np.asarray(
        [int(histogram[low : high + 1].sum()) for low, high in CLASS_RANGES],
        dtype=np.int64,
    )


def histogram_bin_statistics(histogram: np.ndarray) -> list[dict[str, Any]]:
    values = np.arange(128, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for class_id, ((low, high), name) in enumerate(zip(CLASS_RANGES, CLASS_NAMES)):
        counts = histogram[low : high + 1]
        bin_values = values[low : high + 1]
        summary = weighted_summary(bin_values, counts)
        rows.append(
            {
                "class_id": class_id,
                "class_name": name,
                "range": f"{low}" if low == high else f"{low}-{high}",
                **{key: summary[key] for key in ("count", "mean", "median", "std", "min", "max")},
                "unique_value_count": int(np.count_nonzero(counts)),
            }
        )
    return rows


def distribution_rows(source: str, unit: str, counts: np.ndarray) -> list[dict[str, Any]]:
    total = float(np.asarray(counts).sum())
    rows = []
    for class_id, (name, bounds, count) in enumerate(zip(CLASS_NAMES, CLASS_RANGES, counts)):
        low, high = bounds
        rows.append(
            {
                "source": source,
                "unit": unit,
                "class_id": class_id,
                "class_name": name,
                "range": f"{low}" if low == high else f"{low}-{high}",
                "count": float(count) if unit == "seconds" else int(count),
                "ratio": float(count / total) if total else None,
            }
        )
    return rows


def raw_cc64_histograms(midi_path: Path) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Return raw event counts and real-time held-value occupancy.

    Events are all CC64 messages on non-drum instruments. Occupancy starts at
    value 0 at time zero and ends at the later of the last non-drum note end or
    last CC64 event. Simultaneous events retain source instrument/list order;
    the last event at a tick controls the following interval.
    """

    midi = MidiFile(str(midi_path))
    events: list[tuple[int, int, int, int]] = []
    horizon_tick = 0
    for instrument_index, instrument in enumerate(midi.instruments):
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            horizon_tick = max(horizon_tick, int(note.end))
        for event_index, control in enumerate(instrument.control_changes):
            if int(control.number) != 64:
                continue
            value = int(control.value)
            if not 0 <= value <= 127:
                raise ValueError(f"CC64 outside [0,127]: {midi_path}")
            tick = int(control.time)
            events.append((tick, instrument_index, event_index, value))
            horizon_tick = max(horizon_tick, tick)
    events.sort(key=lambda item: (item[0], item[1], item[2]))
    event_values = np.fromiter((event[3] for event in events), dtype=np.int64)
    event_hist = np.bincount(event_values, minlength=128).astype(np.int64)

    occupancy = np.zeros(128, dtype=np.float64)
    if horizon_tick > 0:
        mapping = midi.get_tick_to_time_mapping()
        if horizon_tick >= len(mapping):
            raise ValueError(f"tick/time mapping shorter than horizon: {midi_path}")
        previous_tick = 0
        previous_value = 0
        index = 0
        while index < len(events):
            tick = events[index][0]
            occupancy[previous_value] += float(mapping[tick] - mapping[previous_tick])
            while index < len(events) and events[index][0] == tick:
                previous_value = events[index][3]
                index += 1
            previous_tick = tick
        occupancy[previous_value] += float(mapping[horizon_tick] - mapping[previous_tick])
    return event_hist, occupancy, len(events), float(occupancy.sum())


def class_run_data(classes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not len(classes):
        return np.empty(0, dtype=np.uint8), np.empty(0, dtype=np.int64)
    starts = np.r_[0, np.flatnonzero(classes[1:] != classes[:-1]) + 1]
    ends = np.r_[starts[1:], len(classes)]
    return classes[starts], ends - starts


def state_run_data(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return class_run_data(np.asarray(states, dtype=np.uint8))


def count_repedal_patterns(states: np.ndarray, max_low_samples: int) -> int:
    run_states, run_lengths = state_run_data(states)
    if len(run_states) < 3:
        return 0
    return sum(
        1
        for index in range(1, len(run_states) - 1)
        if run_states[index - 1] == 1
        and run_states[index] == 0
        and run_states[index + 1] == 1
        and int(run_lengths[index]) <= max_low_samples
    )


def count_extreme_repedal_patterns(classes: np.ndarray, max_low_samples: int) -> int:
    # 1=HIGH/FULL (>=96), 0=ZERO/LOW (<=31), 2=middle region.
    states = np.full(classes.shape, 2, dtype=np.uint8)
    states[classes <= 1] = 0
    states[classes >= 4] = 1
    run_states, run_lengths = state_run_data(states)
    if len(run_states) < 3:
        return 0
    return sum(
        1
        for index in range(1, len(run_states) - 1)
        if run_states[index - 1] == 1
        and run_states[index] == 0
        and run_states[index + 1] == 1
        and int(run_lengths[index]) <= max_low_samples
    )


def error_metrics_from_histogram(histogram: np.ndarray, representatives: np.ndarray) -> dict[str, Any]:
    values = np.arange(128, dtype=np.float64)
    classes = classify(values.astype(np.int16))
    errors = np.abs(values - representatives[classes])
    order = np.argsort(errors, kind="stable")
    summary = weighted_summary(errors[order], histogram[order])
    total = int(histogram.sum())
    return {
        "mae": summary["mean"],
        "maximum_absolute_error": summary["max"],
        "error_quantiles": {key: summary[key] for key in ("q01", "q05", "q25", "median", "q75", "q95", "q99")},
        "tolerance_accuracy": {
            "plus_minus_5": float(histogram[errors <= 5].sum() / total) if total else None,
            "plus_minus_10": float(histogram[errors <= 10].sum() / total) if total else None,
            "plus_minus_20": float(histogram[errors <= 20].sum() / total) if total else None,
        },
    }


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    def cell(value: Any) -> str:
        if value is None:
            return "N/A"
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value)

    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asap-root", type=Path, default=DEFAULT_ASAP_ROOT)
    parser.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    asap_root = args.asap_root.resolve()
    split_csv = args.split_csv.resolve()
    output_dir = args.output_dir.resolve()
    project_root = Path("/workspace/project").resolve()
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:12]}"
    pid = os.getpid()

    # Atomic mkdir is the duplicate-run lock. Completed and failed directories
    # are deliberately retained and cannot be silently overwritten.
    output_dir.mkdir(parents=False, exist_ok=False)
    config: dict[str, Any] = {
        "run_id": run_id,
        "pid": pid,
        "started_at_utc": utc_now(),
        "hostname": socket.gethostname(),
        "asap_root": str(asap_root),
        "split_csv": str(split_csv),
        "split_csv_sha256": file_sha256(split_csv),
        "split": "train",
        "data_access_policy": "Only rows whose split field is exactly 'train' are loaded; validation/test MIDI are never opened.",
        "tokenizer": "Pinned Pianist Transformer midi_to_ids via Stage2PedalDataset(cache_mode='preload')",
        "dataset_cache": "Stage2PedalDataset immutable int16 in-memory per-performance preload cache",
        "class_definition": [
            {"class_id": index, "name": name, "minimum": bounds[0], "maximum": bounds[1]}
            for index, (name, bounds) in enumerate(zip(CLASS_NAMES, CLASS_RANGES))
        ],
        "representative_policy": "ZERO=0; FULL=127; each interior class=global train Pedal1-4 median; shared by all slots",
        "transition_definition": "Within each performance only, flatten row-major as note-major Pedal1,Pedal2,Pedal3,Pedal4; an original transition is adjacent values differing; preservation means the class also differs at that same boundary.",
        "repedal_definition": f"HIGH->LOW->HIGH state runs with LOW run <= {PRIMARY_REPEDAL_MAX_SAMPLES} flattened pedal samples; threshold and extreme-bin variants plus 1/2/4/8-sample sensitivity are reported.",
        "model_training": False,
        "git": safe_git_metadata(project_root),
    }
    atomic_json(output_dir / "config.json", config)
    status: dict[str, Any] = {
        "run_id": run_id,
        "pid": pid,
        "state": "running",
        "started_at_utc": config["started_at_utc"],
        "completed_at_utc": None,
        "elapsed_seconds": None,
        "model_training_performed": False,
        "validation_test_midi_accessed": False,
        "error": None,
    }
    atomic_json(output_dir / "audit_status.json", status)

    try:
        dataset = Stage2PedalDataset(
            asap_root,
            split_csv,
            "train",
            window_notes=512,
            stride_notes=256,
            return_metadata=True,
            cache_mode="preload",
        )
        if not dataset.performances or any(row["split"] != "train" for row in dataset.performances):
            raise AssertionError("dataset contains non-train performances")
        if dataset.cached_performance_count != dataset.performance_count:
            raise AssertionError("not all train performances were cached")

        token_hist = np.zeros(128, dtype=np.int64)
        slot_hists = np.zeros((4, 128), dtype=np.int64)
        raw_event_hist = np.zeros(128, dtype=np.int64)
        raw_occupancy_hist = np.zeros(128, dtype=np.float64)
        per_performance_rows: list[dict[str, Any]] = []
        total_notes = 0
        total_raw_events = 0

        for performance_index, row in enumerate(dataset.performances, start=1):
            cached = dataset._token_cache[row["performance_path"]]
            pedal_values = cached[:, NON_PEDAL_FEATURES:].astype(np.int16) - PEDAL_TOKEN_OFFSET
            if pedal_values.shape[1] != 4:
                raise AssertionError("expected four pedal slots")
            total_notes += len(pedal_values)
            flat = pedal_values.reshape(-1)
            token_hist += np.bincount(flat, minlength=128).astype(np.int64)
            for slot in range(4):
                slot_hists[slot] += np.bincount(pedal_values[:, slot], minlength=128).astype(np.int64)
            class_counts = np.bincount(classify(flat), minlength=6).astype(np.int64)
            for class_id, (name, count) in enumerate(zip(CLASS_NAMES, class_counts)):
                per_performance_rows.append(
                    {
                        "performance_path": row["performance_path"],
                        "piece_id": row["piece_id"],
                        "composer": row["composer"],
                        "title": row["title"],
                        "class_id": class_id,
                        "class_name": name,
                        "count": int(count),
                        "ratio": float(count / len(flat)),
                        "token_count": int(len(flat)),
                    }
                )

            midi_path = (asap_root / row["performance_path"]).resolve()
            try:
                midi_path.relative_to(asap_root)
            except ValueError as exc:
                raise ValueError("performance path escapes ASAP root") from exc
            event_hist, occupancy_hist, event_count, _ = raw_cc64_histograms(midi_path)
            expected_raw_events = int(row["num_raw_cc64_events"])
            if event_count != expected_raw_events:
                raise AssertionError(
                    f"raw CC64 count mismatch for {row['performance_path']}: {event_count} != {expected_raw_events}"
                )
            raw_event_hist += event_hist
            raw_occupancy_hist += occupancy_hist
            total_raw_events += event_count
            if performance_index % 100 == 0 or performance_index == dataset.performance_count:
                print(f"distribution audit performances={performance_index}/{dataset.performance_count}", flush=True)

        total_targets = total_notes * 4
        if int(token_hist.sum()) != total_targets or not np.array_equal(slot_hists.sum(axis=0), token_hist):
            raise AssertionError("token histogram accounting failed")
        if int(raw_event_hist.sum()) != total_raw_events:
            raise AssertionError("raw event histogram accounting failed")

        token_class_counts = histogram_class_counts(token_hist)
        raw_event_class_counts = histogram_class_counts(raw_event_hist)
        raw_occupancy_class_seconds = np.asarray(
            [raw_occupancy_hist[low : high + 1].sum() for low, high in CLASS_RANGES]
        )
        slot_class_counts = np.stack([histogram_class_counts(hist) for hist in slot_hists])

        bin_stats = histogram_bin_statistics(token_hist)
        representatives = np.asarray(
            [0.0, bin_stats[1]["median"], bin_stats[2]["median"], bin_stats[3]["median"], bin_stats[4]["median"], 127.0],
            dtype=np.float64,
        )
        if any(value is None for value in representatives.tolist()):
            raise ValueError("cannot derive a representative from an empty interior class")
        if not np.all(np.diff(representatives) > 0):
            raise AssertionError("representatives are not strictly ordered")

        slot_bin_stats = [histogram_bin_statistics(slot_hists[slot]) for slot in range(4)]
        representative_payload = {
            "source_split": "train",
            "global_shared_representatives": {
                name: float(representatives[index]) for index, name in enumerate(CLASS_NAMES)
            },
            "slot_medians_reported_only_not_used_for_reconstruction": {
                SLOT_NAMES[slot]: {
                    CLASS_NAMES[class_id]: slot_bin_stats[slot][class_id]["median"] for class_id in range(6)
                }
                for slot in range(4)
            },
            "all_slots_use_same_global_representatives": True,
        }

        overall_oracle = error_metrics_from_histogram(token_hist, representatives)
        intermediate_hist = token_hist.copy()
        intermediate_hist[0] = 0
        intermediate_hist[127] = 0
        intermediate_oracle = error_metrics_from_histogram(intermediate_hist, representatives)
        slot_oracles = {
            SLOT_NAMES[slot]: error_metrics_from_histogram(slot_hists[slot], representatives) for slot in range(4)
        }
        class_within_mae: dict[str, Any] = {}
        values = np.arange(128, dtype=np.float64)
        for class_id, (name, bounds) in enumerate(zip(CLASS_NAMES, CLASS_RANGES)):
            low, high = bounds
            counts = token_hist[low : high + 1]
            errors = np.abs(values[low : high + 1] - representatives[class_id])
            class_within_mae[name] = float(np.average(errors, weights=counts)) if counts.sum() else None

        performance_maes: list[float] = []
        transition_records: list[dict[str, Any]] = []
        aggregate_transitions = Counter()
        class_run_histograms = [Counter() for _ in range(6)]
        repedal_sensitivity = {
            str(threshold): {"binary_original": 0, "binary_quantized": 0, "extreme_original": 0, "extreme_quantized": 0}
            for threshold in REPEDAL_SENSITIVITY_SAMPLES
        }

        for row in dataset.performances:
            cached = dataset._token_cache[row["performance_path"]]
            original = (cached[:, NON_PEDAL_FEATURES:].astype(np.int16) - PEDAL_TOKEN_OFFSET).reshape(-1)
            classes = classify(original)
            reconstructed = representatives[classes]
            errors = np.abs(original.astype(np.float64) - reconstructed)
            performance_maes.append(float(errors.mean()))

            original_change = original[1:] != original[:-1]
            class_change = classes[1:] != classes[:-1]
            preserved = original_change & class_change
            removed = original_change & ~class_change
            introduced = ~original_change & class_change
            introduced_count = int(introduced.sum())
            if introduced_count:
                raise AssertionError("quantization introduced a transition")
            original_count = int(original_change.sum())
            class_count = int(class_change.sum())
            preserved_count = int(preserved.sum())
            if class_count != preserved_count:
                raise AssertionError("class transition accounting failed")

            original_binary = original >= 64
            quantized_binary = classes >= 3
            if not np.array_equal(original_binary, quantized_binary):
                raise AssertionError("binary sustain state changed under quantization")
            binary_original_change = original_binary[1:] != original_binary[:-1]
            binary_quantized_change = quantized_binary[1:] != quantized_binary[:-1]
            binary_original_count = int(binary_original_change.sum())
            binary_quantized_count = int(binary_quantized_change.sum())

            original_direction = np.sign(original[1:].astype(np.int16) - original[:-1].astype(np.int16))
            class_direction = np.sign(classes[1:].astype(np.int16) - classes[:-1].astype(np.int16))
            direction_agree = int((original_direction[preserved] == class_direction[preserved]).sum())

            transition_records.append(
                {
                    "performance_path": row["performance_path"],
                    "original_value_transitions": original_count,
                    "class_transitions": class_count,
                    "preserved_transitions": preserved_count,
                    "removed_transitions": int(removed.sum()),
                    "newly_introduced_transitions": introduced_count,
                    "preservation_ratio": float(preserved_count / original_count) if original_count else None,
                    "binary_transitions": binary_original_count,
                    "binary_preservation_ratio": float(binary_quantized_count / binary_original_count) if binary_original_count else None,
                    "direction_agreement": float(direction_agree / preserved_count) if preserved_count else None,
                }
            )
            aggregate_transitions.update(
                original=original_count,
                quantized=class_count,
                preserved=preserved_count,
                removed=int(removed.sum()),
                introduced=introduced_count,
                binary_original=binary_original_count,
                binary_quantized=binary_quantized_count,
                direction_agree=direction_agree,
            )

            run_classes, run_lengths = class_run_data(classes)
            for class_id in range(6):
                for length in run_lengths[run_classes == class_id]:
                    class_run_histograms[class_id][int(length)] += 1

            for threshold in REPEDAL_SENSITIVITY_SAMPLES:
                binary_original_patterns = count_repedal_patterns(original_binary, threshold)
                binary_quantized_patterns = count_repedal_patterns(quantized_binary, threshold)
                extreme_original_patterns = count_extreme_repedal_patterns(classes, threshold)
                # Extreme groups are class-defined and therefore identical after reconstruction.
                extreme_quantized_patterns = count_extreme_repedal_patterns(classify(reconstructed), threshold)
                repedal_sensitivity[str(threshold)]["binary_original"] += binary_original_patterns
                repedal_sensitivity[str(threshold)]["binary_quantized"] += binary_quantized_patterns
                repedal_sensitivity[str(threshold)]["extreme_original"] += extreme_original_patterns
                repedal_sensitivity[str(threshold)]["extreme_quantized"] += extreme_quantized_patterns

        transition_ratios = [record["preservation_ratio"] for record in transition_records if record["preservation_ratio"] is not None]
        binary_ratios = [record["binary_preservation_ratio"] for record in transition_records if record["binary_preservation_ratio"] is not None]
        direction_ratios = [record["direction_agreement"] for record in transition_records if record["direction_agreement"] is not None]
        original_total = aggregate_transitions["original"]
        binary_total = aggregate_transitions["binary_original"]
        preserved_total = aggregate_transitions["preserved"]
        class_run_stats = {}
        for class_id, name in enumerate(CLASS_NAMES):
            lengths = np.asarray(sorted(class_run_histograms[class_id]), dtype=np.float64)
            counts = np.asarray([class_run_histograms[class_id][int(length)] for length in lengths], dtype=np.int64)
            class_run_stats[name] = weighted_summary(lengths, counts)

        transition_payload = {
            "not_model_performance": True,
            "sequence_order": "Per performance, note-major row-major Pedal1->Pedal2->Pedal3->Pedal4; no cross-performance boundary.",
            "definitions": {
                "original_value_transition": "Adjacent original values differ (delta != 0).",
                "quantized_class_transition": "Adjacent fixed six-class IDs differ.",
                "preserved": "Original and class transition occur at the same adjacent boundary.",
                "removed": "Original transition occurs but both values map to the same class.",
                "newly_introduced": "Class transition occurs without an original value transition.",
                "binary_state": "UP-like <64 versus DOWN-like >=64.",
                "direction_agreement": "Among preserved transitions, sign(original value delta) equals sign(class-ID delta).",
            },
            "overall": {
                "original_nonzero_value_transition_count": original_total,
                "quantized_class_transition_count": aggregate_transitions["quantized"],
                "preserved_transition_count": preserved_total,
                "transition_preservation_ratio": float(preserved_total / original_total) if original_total else None,
                "transitions_removed_by_quantization": aggregate_transitions["removed"],
                "transitions_newly_introduced": aggregate_transitions["introduced"],
                "binary_sustain_transition_count": binary_total,
                "binary_sustain_quantized_transition_count": aggregate_transitions["binary_quantized"],
                "binary_sustain_transition_preservation_ratio": float(aggregate_transitions["binary_quantized"] / binary_total) if binary_total else None,
                "transition_direction_agreement": float(aggregate_transitions["direction_agree"] / preserved_total) if preserved_total else None,
            },
            "per_performance_distribution": {
                "transition_preservation_ratio": vector_summary(transition_ratios),
                "binary_transition_preservation_ratio": vector_summary(binary_ratios),
                "direction_agreement": vector_summary(direction_ratios),
                "performances_without_original_value_transition": dataset.performance_count - len(transition_ratios),
                "performances_without_binary_transition": dataset.performance_count - len(binary_ratios),
            },
            "class_run_length_samples": class_run_stats,
            "repedal_like_patterns": {
                "primary_max_low_run_samples": PRIMARY_REPEDAL_MAX_SAMPLES,
                "binary_definition": ">=64 -> <64 -> >=64 state runs",
                "extreme_definition": ">=96 -> <=31 -> >=96 state runs",
                "sensitivity_by_max_low_run_samples": repedal_sensitivity,
            },
            "per_performance": transition_records,
        }

        oracle_payload = {
            "not_model_performance": True,
            "interpretation": "Information loss caused solely by replacing each original train Pedal target with its fixed-class train-derived representative.",
            "overall_micro": overall_oracle,
            "intermediate_only_original_1_to_126": intermediate_oracle,
            "per_slot": slot_oracles,
            "performance_macro_mae": vector_summary(performance_maes),
            "class_within_bin_mae": class_within_mae,
        }

        exact_rows = []
        exact_sources = [
            ("raw_cc64_event", "events", raw_event_hist),
            ("raw_cc64_time_occupancy", "seconds", raw_occupancy_hist),
            ("tokenized_pedal_all", "tokens", token_hist),
            *[(f"tokenized_{SLOT_NAMES[slot].lower()}", "tokens", slot_hists[slot]) for slot in range(4)],
        ]
        for source, unit, histogram in exact_sources:
            total = float(histogram.sum())
            for value, count in enumerate(histogram):
                exact_rows.append(
                    {"source": source, "unit": unit, "value": value,
                     "count": float(count) if unit == "seconds" else int(count),
                     "ratio": float(count / total) if total else None}
                )
        atomic_csv(output_dir / "exact_value_histogram.csv", ("source", "unit", "value", "count", "ratio"), exact_rows)

        overall_distribution_rows = []
        overall_distribution_rows.extend(distribution_rows("raw_cc64_event", "events", raw_event_class_counts))
        overall_distribution_rows.extend(distribution_rows("raw_cc64_time_occupancy", "seconds", raw_occupancy_class_seconds))
        overall_distribution_rows.extend(distribution_rows("tokenized_pedal_all", "tokens", token_class_counts))
        atomic_csv(
            output_dir / "class_distribution_overall.csv",
            ("source", "unit", "class_id", "class_name", "range", "count", "ratio"),
            overall_distribution_rows,
        )

        slot_rows = []
        for slot, slot_name in enumerate(SLOT_NAMES):
            total = int(slot_class_counts[slot].sum())
            for class_id, (name, bounds, count) in enumerate(zip(CLASS_NAMES, CLASS_RANGES, slot_class_counts[slot])):
                low, high = bounds
                slot_rows.append(
                    {"slot": slot_name, "class_id": class_id, "class_name": name,
                     "range": f"{low}" if low == high else f"{low}-{high}", "count": int(count),
                     "ratio": float(count / total)}
                )
        atomic_csv(
            output_dir / "class_distribution_per_slot.csv",
            ("slot", "class_id", "class_name", "range", "count", "ratio"),
            slot_rows,
        )
        atomic_csv(
            output_dir / "class_distribution_per_performance.csv",
            ("performance_path", "piece_id", "composer", "title", "class_id", "class_name", "count", "ratio", "token_count"),
            per_performance_rows,
        )

        bin_rows = []
        for record in bin_stats:
            bin_rows.append({"scope": "overall", "slot": "all", **record})
        for slot, slot_name in enumerate(SLOT_NAMES):
            for record in slot_bin_stats[slot]:
                bin_rows.append({"scope": "slot", "slot": slot_name, **record})
        atomic_csv(
            output_dir / "bin_statistics.csv",
            ("scope", "slot", "class_id", "class_name", "range", "count", "mean", "median", "std", "min", "max", "unique_value_count"),
            bin_rows,
        )
        atomic_json(output_dir / "representative_values.json", representative_payload)
        atomic_json(output_dir / "quantization_oracle_metrics.json", oracle_payload)
        atomic_json(output_dir / "transition_preservation_metrics.json", transition_payload)

        per_class_performance_summaries = {}
        for class_id, name in enumerate(CLASS_NAMES):
            ratios = [row["ratio"] for row in per_performance_rows if row["class_id"] == class_id]
            per_class_performance_summaries[name] = vector_summary(ratios)

        raw_token_comparison = []
        raw_total = int(raw_event_class_counts.sum())
        token_total = int(token_class_counts.sum())
        for class_id, name in enumerate(CLASS_NAMES):
            raw_ratio = float(raw_event_class_counts[class_id] / raw_total) if raw_total else None
            token_ratio = float(token_class_counts[class_id] / token_total)
            raw_token_comparison.append(
                (name, raw_ratio, token_ratio, token_ratio - raw_ratio if raw_ratio is not None else None,
                 token_ratio / raw_ratio if raw_ratio else None)
            )

        all_classes_nonempty = bool(np.all(token_class_counts > 0))
        proceed = bool(
            all_classes_nonempty
            and aggregate_transitions["introduced"] == 0
            and aggregate_transitions["binary_quantized"] == binary_total
            and float(overall_oracle["mae"]) <= 16.0
        )
        rarest_id = int(np.argmin(token_class_counts))
        dominant_id = int(np.argmax(token_class_counts))

        token_distribution_table = markdown_table(
            ("Class", "Overall", *SLOT_NAMES),
            [
                (
                    name,
                    float(token_class_counts[class_id] / token_total),
                    *[float(slot_class_counts[slot, class_id] / slot_class_counts[slot].sum()) for slot in range(4)],
                )
                for class_id, name in enumerate(CLASS_NAMES)
            ],
        )
        performance_summary_table = markdown_table(
            ("Class", "mean", "median", "std", "q05", "q25", "q75", "q95"),
            [
                (name, *[per_class_performance_summaries[name][key] for key in ("mean", "median", "std", "q05", "q25", "q75", "q95")])
                for name in CLASS_NAMES
            ],
        )
        representative_table = markdown_table(
            ("Class", "Global representative", *[f"{slot} median" for slot in SLOT_NAMES]),
            [
                (name, representatives[class_id], *[slot_bin_stats[slot][class_id]["median"] for slot in range(4)])
                for class_id, name in enumerate(CLASS_NAMES)
            ],
        )
        raw_summary = {
            "zero": int(raw_event_hist[0]),
            "full": int(raw_event_hist[127]),
            "intermediate": int(raw_event_hist[1:127].sum()),
        }
        report = f"""# Stage 2 Endpoint-aware 6-class Pedal Audit v0

## 1. 목적과 가설

flat 128-class Pedal1–4 target을 고정된 endpoint-aware 6-class로 바꿀 때의 class imbalance, 값 정보 손실, transition 보존성을 모델 학습 전에 측정했다. 가설은 ZERO/FULL을 독립 endpoint로 유지하고 네 intermediate 구간을 쓰면 binary sustain/repedal 구조를 보존하면서 128-class 분류보다 통제 가능한 target을 제공한다는 것이다. 이 문서의 oracle은 **모델 성능이 아니라 표현 자체의 정보 손실**이다.

## 2. 데이터 및 split

- ASAP v1.1 root: `{asap_root}`
- CE baseline split CSV: `{split_csv}` (SHA-256 `{config['split_csv_sha256']}`)
- 사용 범위: train 행만, {dataset.performance_count:,} performances / {len({row['piece_id'] for row in dataset.performances}):,} pieces
- normalized notes: {total_notes:,}; Pedal1–4 targets: {total_targets:,}
- tokenizer/cache: 기존 `Stage2PedalDataset(cache_mode='preload')`와 pinned `midi_to_ids`; validation/test MIDI 접근 없음
- run_id: `{run_id}`; PID: {pid}; 모델 학습 없음

## 3. Raw CC64 분포

non-drum instrument의 원본 CC64 event {total_raw_events:,}개를 셌다. event 기준 0={raw_summary['zero']:,} ({raw_summary['zero']/raw_total:.6%}), 127={raw_summary['full']:,} ({raw_summary['full']/raw_total:.6%}), 1–126={raw_summary['intermediate']:,} ({raw_summary['intermediate']/raw_total:.6%})이다. 6-class event count/ratio와 time-occupancy(초) 분포는 `class_distribution_overall.csv`, exact 0–127 histogram은 `exact_value_histogram.csv`에 있다. Occupancy는 time 0에서 value 0으로 시작하며 마지막 non-drum note end 또는 마지막 CC64 중 늦은 시점까지, 동시 event에서는 source order상 마지막 event가 이후 구간을 지배하도록 계산했다.

{markdown_table(('Class','Raw event ratio','Raw occupancy ratio'), [(name, raw_event_class_counts[i]/raw_total if raw_total else None, raw_occupancy_class_seconds[i]/raw_occupancy_class_seconds.sum() if raw_occupancy_class_seconds.sum() else None) for i,name in enumerate(CLASS_NAMES)])}

## 4. Tokenized Pedal1–4 분포

{token_distribution_table}

Performance별 비율 분포(각 performance를 동일 가중):

{performance_summary_table}

## 5. 6-class class imbalance

가장 드문 class는 {CLASS_NAMES[rarest_id]} ({token_class_counts[rarest_id]:,}, {token_class_counts[rarest_id]/token_total:.6%}), 가장 많은 class는 {CLASS_NAMES[dominant_id]} ({token_class_counts[dominant_id]:,}, {token_class_counts[dominant_id]/token_total:.6%})이다. max/min count ratio는 {token_class_counts[dominant_id]/token_class_counts[rarest_id]:.3f}다. 모든 class는 train에 존재한다: {all_classes_nonempty}.

## 6. Train-derived representative values

ZERO=0, FULL=127로 고정했고, interior representative는 모든 train Pedal1–4 target을 합친 bin별 global median이다. 모든 slot이 동일한 global representative를 사용한다. Slot median은 비교용이며 선택에 사용하지 않았다.

{representative_table}

Bin 내부 mean/median/std/min/max/unique count는 `bin_statistics.csv`에 있다.

## 7. Quantization oracle error

- overall micro MAE: {overall_oracle['mae']:.6f}
- intermediate-only (원래 값 1–126) MAE: {intermediate_oracle['mae']:.6f}
- performance-macro MAE: {oracle_payload['performance_macro_mae']['mean']:.6f} (median {oracle_payload['performance_macro_mae']['median']:.6f})
- tolerance accuracy: ±5 {overall_oracle['tolerance_accuracy']['plus_minus_5']:.6%}, ±10 {overall_oracle['tolerance_accuracy']['plus_minus_10']:.6%}, ±20 {overall_oracle['tolerance_accuracy']['plus_minus_20']:.6%}
- maximum absolute error: {overall_oracle['maximum_absolute_error']:.6f}
- per-slot MAE: {', '.join(f'{slot}={slot_oracles[slot]["mae"]:.6f}' for slot in SLOT_NAMES)}

Class별 within-bin MAE와 error quantile 전체는 `quantization_oracle_metrics.json`에 있다.

## 8. Transition preservation

Sequence는 performance별로 `(note 1 Pedal1→4, note 2 Pedal1→4, ...)` 순서로 flatten했고 performance 경계를 연결하지 않았다. Original transition은 인접 exact value가 다른 경우, class transition은 인접 6-class ID가 다른 경우다.

- original nonzero value transitions: {original_total:,}
- quantized/preserved class transitions: {aggregate_transitions['quantized']:,}
- preservation ratio: {preserved_total/original_total:.6%}
- removed: {aggregate_transitions['removed']:,}; newly introduced: {aggregate_transitions['introduced']:,}
- binary `<64`/`>=64` transitions: {binary_total:,}; preservation: {aggregate_transitions['binary_quantized']/binary_total:.6%}
- preserved transition direction agreement: {aggregate_transitions['direction_agree']/preserved_total:.6%}
- performance-macro preservation: mean {transition_payload['per_performance_distribution']['transition_preservation_ratio']['mean']:.6%}, median {transition_payload['per_performance_distribution']['transition_preservation_ratio']['median']:.6%}
- primary short repedal-like (`>=64→<64→>=64`, low run ≤{PRIMARY_REPEDAL_MAX_SAMPLES} samples): original {repedal_sensitivity[str(PRIMARY_REPEDAL_MAX_SAMPLES)]['binary_original']:,}, quantized {repedal_sensitivity[str(PRIMARY_REPEDAL_MAX_SAMPLES)]['binary_quantized']:,}

Class run-length와 1/2/4/8-sample repedal 민감도는 `transition_preservation_metrics.json`에 있다.

## 9. Slot별 차이

Slot별 class ratio 차이는 위 표와 `class_distribution_per_slot.csv`에, slot별 median은 대표값 표에, slot별 oracle MAE는 Section 7과 JSON에 기록했다. 대표값은 slot별 차이에 맞춰 조정하지 않았다.

## 10. Raw와 tokenized 비교 및 위험 요소

{markdown_table(('Class','Raw event ratio','Token ratio','Token - raw (pp as ratio)','Token/raw'), raw_token_comparison)}

- Token sampling은 raw **event frequency**가 아니라 note-IOI 내부 네 지점의 held state를 표본화하므로 event ratio와 직접 같은 모집단이 아니다. 위 차이는 sampling/시간 점유 효과를 함께 반영한다.
- 6-class 내부의 모든 exact 변화는 제거되므로 exact-value transition 보존율은 {preserved_total/original_total:.6%}에 그친다. 반면 `<64`/`>=64` 경계는 class 경계와 일치해 100% 보존된다.
- 가장 드문 {CLASS_NAMES[rarest_id]} class의 imbalance는 controlled training에서 class별 recall, macro-F1, confusion을 반드시 별도로 보게 만든다.
- 마지막 note의 Pedal2–4는 원 tokenizer의 고정 4,990-tick tail sampling을 그대로 포함한다. 이것은 새 audit이 만든 현상이 아니라 기존 target grammar의 특성이다.
- 짧은 repedal 통계의 “short”는 실제 millisecond가 아니라 flattened sample 개수다. IOI가 가변이므로 timing 품질의 대용치로 해석하면 안 된다.

## 11. Controlled training 진행 결론

**{'진행 가능' if proceed else '현재 정의로 진행 보류'}** (representation audit 기준). 판정 조건은 모든 train class 존재, newly introduced transition=0, binary transition 100% 보존, overall oracle MAE≤16이었다. 이 결론은 모델이 해당 class를 잘 학습한다는 뜻이 아니다. 다음 실험은 이 고정 경계/대표값을 변경하지 않고 CE baseline과 동일 split·초기화·학습 budget을 사용하는 controlled comparison이어야 하며, validation은 모델 선택/평가에만 쓰고 대표값 재추정에는 쓰지 않아야 한다.
"""
        atomic_text(output_dir / "SIX_CLASS_AUDIT_REPORT.md", report)

        config.update(
            {
                "train_performance_count": dataset.performance_count,
                "train_piece_count": len({row["piece_id"] for row in dataset.performances}),
                "normalized_note_count": total_notes,
                "tokenized_pedal_target_count": total_targets,
                "raw_cc64_event_count": total_raw_events,
                "cached_performance_count": dataset.cached_performance_count,
                "cached_token_count_all_8_columns": dataset.cached_token_count,
                "cached_nbytes": dataset.cached_nbytes,
                "controlled_training_audit_decision": "proceed" if proceed else "hold",
                "completed_at_utc": utc_now(),
            }
        )
        atomic_json(output_dir / "config.json", config)
        required = {
            "config.json", "audit_status.json", "exact_value_histogram.csv",
            "class_distribution_overall.csv", "class_distribution_per_slot.csv",
            "class_distribution_per_performance.csv", "bin_statistics.csv",
            "representative_values.json", "quantization_oracle_metrics.json",
            "transition_preservation_metrics.json", "SIX_CLASS_AUDIT_REPORT.md",
        }
        missing = sorted(name for name in required if not (output_dir / name).is_file())
        if missing:
            raise AssertionError(f"missing required outputs: {missing}")
        status.update(
            state="completed",
            completed_at_utc=utc_now(),
            elapsed_seconds=time.perf_counter() - started,
            train_performance_count=dataset.performance_count,
            train_tokenized_pedal_target_count=total_targets,
            required_output_files_verified=True,
        )
        atomic_json(output_dir / "audit_status.json", status)
        print(json.dumps({"output_dir": str(output_dir), "status": status}, indent=2), flush=True)
    except BaseException as exc:
        status.update(
            state="failed",
            completed_at_utc=utc_now(),
            elapsed_seconds=time.perf_counter() - started,
            error={"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
        )
        atomic_json(output_dir / "audit_status.json", status)
        raise


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
