"""Train-only audit of the fixed endpoint-aware five-class pedal candidate."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import socket
import time
import traceback
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .audit_six_class import (
    REPEDAL_SENSITIVITY_SAMPLES,
    SLOT_NAMES,
    atomic_csv,
    atomic_json,
    atomic_text,
    class_run_data,
    count_repedal_patterns,
    file_sha256,
    markdown_table,
    safe_git_metadata,
    utc_now,
    vector_summary,
    weighted_summary,
)
from .dataset import NON_PEDAL_FEATURES, PEDAL_TOKEN_OFFSET, Stage2PedalDataset


CLASS_NAMES = ("ZERO", "LOW", "MID", "HIGH", "FULL")
CLASS_RANGES = ((0, 0), (1, 63), (64, 95), (96, 126), (127, 127))
DEFAULT_ASAP_ROOT = Path("/workspace/public/ASAP/asap-dataset-v1.1")
DEFAULT_SPLIT_CSV = Path("/workspace/project/analysis/stage2_encoder_only_v0/asap_split.csv")
DEFAULT_SIX_CLASS_DIR = Path("/workspace/project/analysis/stage2_pedal_6class_audit_v0")
DEFAULT_OUTPUT = Path("/workspace/project/analysis/stage2_pedal_5class_audit_v0")
PRIMARY_REPEDAL_MAX_SAMPLES = 4
ADOPTION_CRITERIA = {
    "overall_oracle_mae_max": 4.0,
    "intermediate_only_oracle_mae_max": 10.0,
    "plus_minus_10_accuracy_min": 0.85,
    "binary_transition_preservation_required": 1.0,
    "short_repedal_preservation_required": 1.0,
    "all_five_classes_present_required": True,
    "new_transitions_required": 0,
}


def classify_five(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.size and (int(values.min()) < 0 or int(values.max()) > 127):
        raise ValueError("pedal value outside [0, 127]")
    classes = np.empty(values.shape, dtype=np.uint8)
    classes[values == 0] = 0
    classes[(values >= 1) & (values <= 63)] = 1
    classes[(values >= 64) & (values <= 95)] = 2
    classes[(values >= 96) & (values <= 126)] = 3
    classes[values == 127] = 4
    return classes


def class_counts(histogram: np.ndarray) -> np.ndarray:
    return np.asarray(
        [int(histogram[low : high + 1].sum()) for low, high in CLASS_RANGES],
        dtype=np.int64,
    )


def bin_statistics(histogram: np.ndarray) -> list[dict[str, Any]]:
    values = np.arange(128, dtype=np.float64)
    rows = []
    for class_id, (name, bounds) in enumerate(zip(CLASS_NAMES, CLASS_RANGES)):
        low, high = bounds
        counts = histogram[low : high + 1]
        summary = weighted_summary(values[low : high + 1], counts)
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


def oracle_from_histogram(histogram: np.ndarray, representatives: np.ndarray) -> dict[str, Any]:
    values = np.arange(128, dtype=np.float64)
    classes = classify_five(values.astype(np.int16))
    errors = np.abs(values - representatives[classes])
    order = np.argsort(errors, kind="stable")
    summary = weighted_summary(errors[order], histogram[order])
    total = int(histogram.sum())
    return {
        "mae": summary["mean"],
        "maximum_absolute_error": summary["max"],
        "error_quantiles": {
            key: summary[key] for key in ("q01", "q05", "q25", "median", "q75", "q95", "q99")
        },
        "tolerance_accuracy": {
            "plus_minus_5": float(histogram[errors <= 5].sum() / total) if total else None,
            "plus_minus_10": float(histogram[errors <= 10].sum() / total) if total else None,
            "plus_minus_20": float(histogram[errors <= 20].sum() / total) if total else None,
        },
    }


def load_six_class_baseline(directory: Path, split_hash: str) -> dict[str, Any]:
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    status = json.loads((directory / "audit_status.json").read_text(encoding="utf-8"))
    oracle = json.loads((directory / "quantization_oracle_metrics.json").read_text(encoding="utf-8"))
    transitions = json.loads((directory / "transition_preservation_metrics.json").read_text(encoding="utf-8"))
    representatives = json.loads((directory / "representative_values.json").read_text(encoding="utf-8"))
    if status["state"] != "completed" or config["split"] != "train":
        raise ValueError("six-class baseline is not a completed train audit")
    if config["split_csv_sha256"] != split_hash:
        raise ValueError("five/six-class split CSV hash mismatch")
    distribution = []
    with (directory / "class_distribution_overall.csv").open(newline="", encoding="utf-8") as handle:
        distribution = [row for row in csv.DictReader(handle) if row["source"] == "tokenized_pedal_all"]
    six_counts = np.asarray([int(row["count"]) for row in distribution], dtype=np.int64)
    if len(six_counts) != 6 or np.any(six_counts <= 0):
        raise ValueError("invalid six-class distribution")
    primary = transitions["repedal_like_patterns"]["sensitivity_by_max_low_run_samples"][str(PRIMARY_REPEDAL_MAX_SAMPLES)]
    return {
        "source_directory": str(directory),
        "run_id": config["run_id"],
        "train_performance_count": config["train_performance_count"],
        "token_count": config["tokenized_pedal_target_count"],
        "class_count": 6,
        "class_counts": six_counts.tolist(),
        "max_min_imbalance_ratio": float(six_counts.max() / six_counts.min()),
        "representative_values": list(representatives["global_shared_representatives"].values()),
        "overall_oracle_mae": oracle["overall_micro"]["mae"],
        "intermediate_only_oracle_mae": oracle["intermediate_only_original_1_to_126"]["mae"],
        "performance_macro_mae": oracle["performance_macro_mae"]["mean"],
        "tolerance_accuracy": oracle["overall_micro"]["tolerance_accuracy"],
        "maximum_absolute_error": oracle["overall_micro"]["maximum_absolute_error"],
        "error_quantiles": oracle["overall_micro"]["error_quantiles"],
        "exact_transition_preservation": transitions["overall"]["transition_preservation_ratio"],
        "binary_transition_preservation": transitions["overall"]["binary_sustain_transition_preservation_ratio"],
        "transition_direction_agreement": transitions["overall"]["transition_direction_agreement"],
        "new_transitions": transitions["overall"]["transitions_newly_introduced"],
        "short_repedal_original": primary["binary_original"],
        "short_repedal_quantized": primary["binary_quantized"],
        "short_repedal_preservation": (
            primary["binary_quantized"] / primary["binary_original"] if primary["binary_original"] else None
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asap-root", type=Path, default=DEFAULT_ASAP_ROOT)
    parser.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT_CSV)
    parser.add_argument("--six-class-dir", type=Path, default=DEFAULT_SIX_CLASS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    asap_root = args.asap_root.resolve()
    split_csv = args.split_csv.resolve()
    six_class_dir = args.six_class_dir.resolve()
    output_dir = args.output_dir.resolve()
    split_hash = file_sha256(split_csv)
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:12]}"
    pid = os.getpid()
    output_dir.mkdir(parents=False, exist_ok=False)

    config: dict[str, Any] = {
        "run_id": run_id,
        "pid": pid,
        "started_at_utc": utc_now(),
        "hostname": socket.gethostname(),
        "asap_root": str(asap_root),
        "split_csv": str(split_csv),
        "split_csv_sha256": split_hash,
        "split": "train",
        "data_access_policy": "Only train rows are loaded; validation/test MIDI are never opened.",
        "six_class_baseline_directory_read_only": str(six_class_dir),
        "tokenizer": "Pinned Pianist Transformer midi_to_ids via Stage2PedalDataset(cache_mode='preload')",
        "class_definition": [
            {"class_id": index, "name": name, "minimum": bounds[0], "maximum": bounds[1]}
            for index, (name, bounds) in enumerate(zip(CLASS_NAMES, CLASS_RANGES))
        ],
        "representative_policy": "ZERO=0; FULL=127; interior classes use global train Pedal1-4 median shared by all slots.",
        "adoption_criteria": ADOPTION_CRITERIA,
        "model_training": False,
        "git": safe_git_metadata(Path("/workspace/project")),
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
        "existing_six_class_outputs_modified": False,
        "error": None,
    }
    atomic_json(output_dir / "audit_status.json", status)

    try:
        six = load_six_class_baseline(six_class_dir, split_hash)
        dataset = Stage2PedalDataset(
            asap_root, split_csv, "train", window_notes=512, stride_notes=256,
            return_metadata=True, cache_mode="preload",
        )
        if any(row["split"] != "train" for row in dataset.performances):
            raise AssertionError("non-train row loaded")
        if dataset.performance_count != six["train_performance_count"]:
            raise AssertionError("five/six-class performance count mismatch")

        histogram = np.zeros(128, dtype=np.int64)
        slot_histograms = np.zeros((4, 128), dtype=np.int64)
        performance_distribution_rows = []
        total_notes = 0
        for index, row in enumerate(dataset.performances, 1):
            pedals = dataset._token_cache[row["performance_path"]][:, NON_PEDAL_FEATURES:].astype(np.int16) - PEDAL_TOKEN_OFFSET
            flat = pedals.reshape(-1)
            total_notes += len(pedals)
            histogram += np.bincount(flat, minlength=128).astype(np.int64)
            for slot in range(4):
                slot_histograms[slot] += np.bincount(pedals[:, slot], minlength=128).astype(np.int64)
            counts = np.bincount(classify_five(flat), minlength=5).astype(np.int64)
            for class_id, (name, count) in enumerate(zip(CLASS_NAMES, counts)):
                performance_distribution_rows.append(
                    {
                        "performance_path": row["performance_path"], "piece_id": row["piece_id"],
                        "composer": row["composer"], "title": row["title"], "class_id": class_id,
                        "class_name": name, "count": int(count), "ratio": float(count / len(flat)),
                        "token_count": int(len(flat)),
                    }
                )
            if index % 100 == 0 or index == dataset.performance_count:
                print(f"five-class distribution performances={index}/{dataset.performance_count}", flush=True)

        total_targets = total_notes * 4
        if int(histogram.sum()) != total_targets or not np.array_equal(slot_histograms.sum(axis=0), histogram):
            raise AssertionError("histogram accounting failed")
        if total_targets != six["token_count"]:
            raise AssertionError("five/six-class token count mismatch")

        overall_counts = class_counts(histogram)
        slot_counts = np.stack([class_counts(slot_histograms[slot]) for slot in range(4)])
        global_bin_stats = bin_statistics(histogram)
        slot_bin_stats = [bin_statistics(slot_histograms[slot]) for slot in range(4)]
        representatives = np.asarray(
            [0.0, global_bin_stats[1]["median"], global_bin_stats[2]["median"], global_bin_stats[3]["median"], 127.0],
            dtype=np.float64,
        )
        if not np.all(np.diff(representatives) > 0):
            raise AssertionError("representatives are not strictly ordered")

        overall_oracle = oracle_from_histogram(histogram, representatives)
        intermediate_histogram = histogram.copy()
        intermediate_histogram[[0, 127]] = 0
        intermediate_oracle = oracle_from_histogram(intermediate_histogram, representatives)
        slot_oracles = {
            SLOT_NAMES[slot]: oracle_from_histogram(slot_histograms[slot], representatives) for slot in range(4)
        }
        performance_maes = []
        transition_records = []
        transition_totals = Counter()
        repedal_sensitivity = {
            str(limit): {"original": 0, "quantized": 0} for limit in REPEDAL_SENSITIVITY_SAMPLES
        }
        class_run_histograms = [Counter() for _ in CLASS_NAMES]

        for row in dataset.performances:
            original = (
                dataset._token_cache[row["performance_path"]][:, NON_PEDAL_FEATURES:].astype(np.int16)
                - PEDAL_TOKEN_OFFSET
            ).reshape(-1)
            classes = classify_five(original)
            reconstructed = representatives[classes]
            performance_maes.append(float(np.abs(original - reconstructed).mean()))
            original_change = original[1:] != original[:-1]
            quantized_change = classes[1:] != classes[:-1]
            preserved = original_change & quantized_change
            removed = original_change & ~quantized_change
            introduced = ~original_change & quantized_change
            if np.any(introduced):
                raise AssertionError("quantization introduced a transition")
            original_count = int(original_change.sum())
            preserved_count = int(preserved.sum())
            binary_original = original >= 64
            binary_quantized = classes >= 2
            if not np.array_equal(binary_original, binary_quantized):
                raise AssertionError("binary state changed")
            binary_count = int((binary_original[1:] != binary_original[:-1]).sum())
            binary_quantized_count = int((binary_quantized[1:] != binary_quantized[:-1]).sum())
            original_direction = np.sign(original[1:].astype(np.int16) - original[:-1].astype(np.int16))
            class_direction = np.sign(classes[1:].astype(np.int16) - classes[:-1].astype(np.int16))
            direction_agree = int((original_direction[preserved] == class_direction[preserved]).sum())
            record = {
                "performance_path": row["performance_path"],
                "original_value_transitions": original_count,
                "class_transitions": int(quantized_change.sum()),
                "preserved_transitions": preserved_count,
                "removed_transitions": int(removed.sum()),
                "newly_introduced_transitions": int(introduced.sum()),
                "preservation_ratio": float(preserved_count / original_count) if original_count else None,
                "binary_transitions": binary_count,
                "binary_preservation_ratio": float(binary_quantized_count / binary_count) if binary_count else None,
                "direction_agreement": float(direction_agree / preserved_count) if preserved_count else None,
            }
            transition_records.append(record)
            transition_totals.update(
                original=original_count, quantized=int(quantized_change.sum()), preserved=preserved_count,
                removed=int(removed.sum()), introduced=int(introduced.sum()), binary_original=binary_count,
                binary_quantized=binary_quantized_count, direction_agree=direction_agree,
            )
            run_classes, run_lengths = class_run_data(classes)
            for class_id in range(5):
                for length in run_lengths[run_classes == class_id]:
                    class_run_histograms[class_id][int(length)] += 1
            for limit in REPEDAL_SENSITIVITY_SAMPLES:
                repedal_sensitivity[str(limit)]["original"] += count_repedal_patterns(binary_original, limit)
                repedal_sensitivity[str(limit)]["quantized"] += count_repedal_patterns(binary_quantized, limit)

        original_total = transition_totals["original"]
        preserved_total = transition_totals["preserved"]
        binary_total = transition_totals["binary_original"]
        primary_repedal = repedal_sensitivity[str(PRIMARY_REPEDAL_MAX_SAMPLES)]
        short_repedal_preservation = (
            primary_repedal["quantized"] / primary_repedal["original"] if primary_repedal["original"] else None
        )
        class_run_stats = {}
        for class_id, name in enumerate(CLASS_NAMES):
            lengths = np.asarray(sorted(class_run_histograms[class_id]), dtype=np.float64)
            counts = np.asarray([class_run_histograms[class_id][int(length)] for length in lengths], dtype=np.int64)
            class_run_stats[name] = weighted_summary(lengths, counts)

        oracle_payload = {
            "not_model_performance": True,
            "overall_micro": overall_oracle,
            "intermediate_only_original_1_to_126": intermediate_oracle,
            "per_slot": slot_oracles,
            "performance_macro_mae": vector_summary(performance_maes),
        }
        transition_payload = {
            "not_model_performance": True,
            "sequence_order": "Per performance, note-major Pedal1->Pedal2->Pedal3->Pedal4; no cross-performance boundary.",
            "overall": {
                "original_nonzero_value_transition_count": original_total,
                "quantized_class_transition_count": transition_totals["quantized"],
                "preserved_transition_count": preserved_total,
                "transition_preservation_ratio": float(preserved_total / original_total) if original_total else None,
                "transitions_removed_by_quantization": transition_totals["removed"],
                "transitions_newly_introduced": transition_totals["introduced"],
                "binary_sustain_transition_count": binary_total,
                "binary_sustain_quantized_transition_count": transition_totals["binary_quantized"],
                "binary_sustain_transition_preservation_ratio": float(transition_totals["binary_quantized"] / binary_total) if binary_total else None,
                "transition_direction_agreement": float(transition_totals["direction_agree"] / preserved_total) if preserved_total else None,
            },
            "per_performance_distribution": {
                "transition_preservation_ratio": vector_summary([r["preservation_ratio"] for r in transition_records if r["preservation_ratio"] is not None]),
                "binary_transition_preservation_ratio": vector_summary([r["binary_preservation_ratio"] for r in transition_records if r["binary_preservation_ratio"] is not None]),
                "direction_agreement": vector_summary([r["direction_agreement"] for r in transition_records if r["direction_agreement"] is not None]),
            },
            "class_run_length_samples": class_run_stats,
            "repedal_like_patterns": {
                "definition": ">=64 -> <64 -> >=64 state runs",
                "primary_max_low_run_samples": PRIMARY_REPEDAL_MAX_SAMPLES,
                "primary_original": primary_repedal["original"],
                "primary_quantized": primary_repedal["quantized"],
                "primary_preservation_ratio": short_repedal_preservation,
                "sensitivity_by_max_low_run_samples": repedal_sensitivity,
            },
            "per_performance": transition_records,
        }

        five_imbalance = float(overall_counts.max() / overall_counts.min())
        all_present = bool(np.all(overall_counts > 0))
        checks = {
            "overall_oracle_mae_le_4": overall_oracle["mae"] <= ADOPTION_CRITERIA["overall_oracle_mae_max"],
            "intermediate_only_oracle_mae_le_10": intermediate_oracle["mae"] <= ADOPTION_CRITERIA["intermediate_only_oracle_mae_max"],
            "plus_minus_10_accuracy_ge_85_percent": overall_oracle["tolerance_accuracy"]["plus_minus_10"] >= ADOPTION_CRITERIA["plus_minus_10_accuracy_min"],
            "binary_transition_preservation_100_percent": transition_payload["overall"]["binary_sustain_transition_preservation_ratio"] == 1.0,
            "short_repedal_preservation_100_percent": short_repedal_preservation == 1.0,
            "all_five_classes_present": all_present,
            "no_new_transitions": transition_totals["introduced"] == 0,
        }
        adopted = all(checks.values())
        five = {
            "run_id": run_id, "train_performance_count": dataset.performance_count,
            "token_count": total_targets, "class_count": 5, "class_counts": overall_counts.tolist(),
            "max_min_imbalance_ratio": five_imbalance,
            "representative_values": representatives.tolist(),
            "overall_oracle_mae": overall_oracle["mae"],
            "intermediate_only_oracle_mae": intermediate_oracle["mae"],
            "performance_macro_mae": oracle_payload["performance_macro_mae"]["mean"],
            "tolerance_accuracy": overall_oracle["tolerance_accuracy"],
            "maximum_absolute_error": overall_oracle["maximum_absolute_error"],
            "error_quantiles": overall_oracle["error_quantiles"],
            "exact_transition_preservation": transition_payload["overall"]["transition_preservation_ratio"],
            "binary_transition_preservation": transition_payload["overall"]["binary_sustain_transition_preservation_ratio"],
            "transition_direction_agreement": transition_payload["overall"]["transition_direction_agreement"],
            "new_transitions": transition_totals["introduced"],
            "short_repedal_original": primary_repedal["original"],
            "short_repedal_quantized": primary_repedal["quantized"],
            "short_repedal_preservation": short_repedal_preservation,
        }
        comparison_payload = {
            "same_train_split_sha256": split_hash,
            "five_class": five,
            "six_class": six,
            "five_class_adoption_criteria": ADOPTION_CRITERIA,
            "five_class_criterion_checks": checks,
            "all_criteria_passed": adopted,
            "recommendation": "prioritize_5class_for_controlled_training" if adopted else "retain_6class",
        }

        exact_rows = []
        for source, hist in [("tokenized_pedal_all", histogram), *[(f"tokenized_{SLOT_NAMES[s].lower()}", slot_histograms[s]) for s in range(4)]]:
            total = int(hist.sum())
            for value, count in enumerate(hist):
                exact_rows.append({"source": source, "unit": "tokens", "value": value, "count": int(count), "ratio": float(count / total)})
        atomic_csv(output_dir / "exact_value_histogram.csv", ("source", "unit", "value", "count", "ratio"), exact_rows)
        overall_rows = []
        for class_id, (name, bounds, count) in enumerate(zip(CLASS_NAMES, CLASS_RANGES, overall_counts)):
            low, high = bounds
            overall_rows.append({"class_id": class_id, "class_name": name, "range": f"{low}" if low == high else f"{low}-{high}", "count": int(count), "ratio": float(count / total_targets)})
        atomic_csv(output_dir / "class_distribution_overall.csv", ("class_id", "class_name", "range", "count", "ratio"), overall_rows)
        slot_rows = []
        for slot, slot_name in enumerate(SLOT_NAMES):
            for class_id, (name, bounds, count) in enumerate(zip(CLASS_NAMES, CLASS_RANGES, slot_counts[slot])):
                low, high = bounds
                slot_rows.append({"slot": slot_name, "class_id": class_id, "class_name": name, "range": f"{low}" if low == high else f"{low}-{high}", "count": int(count), "ratio": float(count / total_notes)})
        atomic_csv(output_dir / "class_distribution_per_slot.csv", ("slot", "class_id", "class_name", "range", "count", "ratio"), slot_rows)
        atomic_csv(output_dir / "class_distribution_per_performance.csv", ("performance_path", "piece_id", "composer", "title", "class_id", "class_name", "count", "ratio", "token_count"), performance_distribution_rows)
        bin_rows = [{"scope": "overall", "slot": "all", **row} for row in global_bin_stats]
        for slot, slot_name in enumerate(SLOT_NAMES):
            bin_rows.extend({"scope": "slot", "slot": slot_name, **row} for row in slot_bin_stats[slot])
        atomic_csv(output_dir / "bin_statistics.csv", ("scope", "slot", "class_id", "class_name", "range", "count", "mean", "median", "std", "min", "max", "unique_value_count"), bin_rows)
        atomic_json(output_dir / "representative_values.json", {
            "source_split": "train",
            "global_shared_representatives": {name: float(representatives[i]) for i, name in enumerate(CLASS_NAMES)},
            "slot_medians_reported_only_not_used_for_reconstruction": {SLOT_NAMES[s]: {name: slot_bin_stats[s][i]["median"] for i, name in enumerate(CLASS_NAMES)} for s in range(4)},
            "all_slots_use_same_global_representatives": True,
        })
        atomic_json(output_dir / "quantization_oracle_metrics.json", oracle_payload)
        atomic_json(output_dir / "transition_preservation_metrics.json", transition_payload)
        atomic_json(output_dir / "comparison_5class_vs_6class.json", comparison_payload)

        distribution_table = markdown_table(
            ("Class", "Overall", *SLOT_NAMES),
            [(name, overall_counts[i] / total_targets, *[slot_counts[s, i] / total_notes for s in range(4)]) for i, name in enumerate(CLASS_NAMES)],
        )
        comparison_table = markdown_table(
            ("Metric", "5-class", "6-class", "5-class criterion"),
            [
                ("Max/min imbalance", five_imbalance, six["max_min_imbalance_ratio"], "descriptive"),
                ("Representatives", str(five["representative_values"]), str(six["representative_values"]), "train medians"),
                ("Overall oracle MAE", five["overall_oracle_mae"], six["overall_oracle_mae"], "<=4.0"),
                ("Intermediate-only MAE", five["intermediate_only_oracle_mae"], six["intermediate_only_oracle_mae"], "<=10.0"),
                ("Performance-macro MAE", five["performance_macro_mae"], six["performance_macro_mae"], "descriptive"),
                ("±5 accuracy", five["tolerance_accuracy"]["plus_minus_5"], six["tolerance_accuracy"]["plus_minus_5"], "descriptive"),
                ("±10 accuracy", five["tolerance_accuracy"]["plus_minus_10"], six["tolerance_accuracy"]["plus_minus_10"], ">=0.85"),
                ("±20 accuracy", five["tolerance_accuracy"]["plus_minus_20"], six["tolerance_accuracy"]["plus_minus_20"], "descriptive"),
                ("Maximum error", five["maximum_absolute_error"], six["maximum_absolute_error"], "descriptive"),
                ("Exact transition preservation", five["exact_transition_preservation"], six["exact_transition_preservation"], "descriptive"),
                ("Binary transition preservation", five["binary_transition_preservation"], six["binary_transition_preservation"], "==1.0"),
                ("Direction agreement", five["transition_direction_agreement"], six["transition_direction_agreement"], "descriptive"),
                ("Short repedal preservation", five["short_repedal_preservation"], six["short_repedal_preservation"], "==1.0"),
                ("New transitions", five["new_transitions"], six["new_transitions"], "==0"),
            ],
        )
        criteria_table = markdown_table(("Criterion", "Pass"), [(name, passed) for name, passed in checks.items()])
        report = f"""# Stage 2 Endpoint-aware 5-class Candidate Audit v0

## Scope

기존 6-class audit와 동일한 CE baseline split, pinned tokenizer, immutable preload cache, performance별 note-major Pedal1→4 flatten 순서를 사용했다. ASAP train {dataset.performance_count:,} performances / {total_targets:,} Pedal targets만 사용했으며 모델 학습과 validation/test MIDI 접근은 없었다. 6-class output은 읽기 전용 비교 baseline으로만 사용했고 수정하지 않았다.

## 5-class distribution

{distribution_table}

Max/min imbalance ratio는 {five_imbalance:.6f}이며, 모든 class가 train에 존재한다: {all_present}.

## Representatives

Global train-derived shared representatives: `{representatives.tolist()}`. ZERO/FULL은 0/127 고정이고 interior는 해당 train bin의 global median이다. Slot median은 `representative_values.json`에 비교용으로만 기록했다.

## Side-by-side: 5-class vs 6-class

{comparison_table}

5-class overall error quantiles: `{json.dumps(overall_oracle['error_quantiles'], ensure_ascii=False)}`.

Exact transition은 같은 performance 안에서 인접 exact value가 달라지는 boundary 중 class도 달라지는 비율이다. Binary state는 `<64`/`>=64`, short repedal-like는 `>=64→<64→>=64`에서 low run ≤{PRIMARY_REPEDAL_MAX_SAMPLES} flattened samples로 6-class audit와 동일하다. 5-class short pattern은 {primary_repedal['original']:,}/{primary_repedal['quantized']:,}로 보존됐다.

## Adoption criteria

{criteria_table}

## Conclusion

**{'5-class를 controlled training의 우선 표현으로 추천한다.' if adopted else '5-class가 채택 기준을 통과하지 못했으므로 기존 6-class를 유지한다.'}** 모든 기준 통과: {adopted}. 이 결론은 표현 audit 결과이며 모델 성능을 의미하지 않는다.
"""
        atomic_text(output_dir / "FIVE_CLASS_AUDIT_REPORT.md", report)

        config.update({
            "train_performance_count": dataset.performance_count,
            "train_piece_count": len({row["piece_id"] for row in dataset.performances}),
            "normalized_note_count": total_notes,
            "tokenized_pedal_target_count": total_targets,
            "cached_performance_count": dataset.cached_performance_count,
            "cached_nbytes": dataset.cached_nbytes,
            "all_adoption_criteria_passed": adopted,
            "recommendation": comparison_payload["recommendation"],
            "completed_at_utc": utc_now(),
        })
        atomic_json(output_dir / "config.json", config)
        required = {
            "config.json", "audit_status.json", "exact_value_histogram.csv",
            "class_distribution_overall.csv", "class_distribution_per_slot.csv",
            "class_distribution_per_performance.csv", "bin_statistics.csv",
            "representative_values.json", "quantization_oracle_metrics.json",
            "transition_preservation_metrics.json", "comparison_5class_vs_6class.json",
            "FIVE_CLASS_AUDIT_REPORT.md",
        }
        missing = sorted(name for name in required if not (output_dir / name).is_file())
        if missing:
            raise AssertionError(f"missing required outputs: {missing}")
        status.update(
            state="completed", completed_at_utc=utc_now(), elapsed_seconds=time.perf_counter() - started,
            train_performance_count=dataset.performance_count,
            train_tokenized_pedal_target_count=total_targets,
            all_adoption_criteria_passed=adopted,
            recommendation=comparison_payload["recommendation"], required_output_files_verified=True,
        )
        atomic_json(output_dir / "audit_status.json", status)
        print(json.dumps({"output_dir": str(output_dir), "status": status}, indent=2), flush=True)
    except BaseException as exc:
        status.update(
            state="failed", completed_at_utc=utc_now(), elapsed_seconds=time.perf_counter() - started,
            error={"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
        )
        atomic_json(output_dir / "audit_status.json", status)
        raise


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
