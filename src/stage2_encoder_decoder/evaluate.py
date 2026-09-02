"""Validation-only teacher-forced and free-running encoder-decoder evaluation."""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from src.stage2_encoder_only.dataset import (
    NON_PEDAL_FEATURES,
    PEDAL_SLOTS,
    PEDAL_TOKEN_OFFSET,
    Stage2PedalDataset,
    generate_window_starts,
)
from src.stage2_encoder_only.evaluate_five_class import (
    EXPECTED_NOTES,
    EXPECTED_PERFORMANCES,
    EXPECTED_PIECES,
    EXPECTED_WINDOWS,
    _atomic_csv,
    _atomic_json,
    _flat_metric_row,
    classification_metrics,
    decoded_value_metrics,
    renderer_aware_metrics,
)
from src.stage2_encoder_only.evaluate_oracle import _make_window_sample, _visible_gpu_identity
from src.stage2_encoder_only.five_class import (
    CLASS_NAMES,
    NUM_CLASSES,
    REPRESENTATIVES,
    average_five_class_logits,
    classify_pedal_values,
    five_class_collate_fn,
)

from .model import EncoderDecoderFiveClassModel


BASELINE_ROOT = Path(
    "/workspace/project/analysis/stage2_encoder_only_5class_weighted_sqrt_probability_v0"
)
TEACHER_MODEL = "encoder_decoder_teacher_forced"
FREE_MODEL = "encoder_decoder_greedy_free_running"


def _slice_batch(batch: Mapping[str, Any], start: int, end: int) -> dict[str, Any]:
    size = int(batch["input_ids"].shape[0])
    return {
        key: (
            value[start:end]
            if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == size
            else value[start:end]
            if isinstance(value, list) and len(value) == size
            else value
        )
        for key, value in batch.items()
    }


def _load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    configuration = checkpoint["configuration"]
    model = EncoderDecoderFiveClassModel.from_pretrained_encoder(
        configuration["checkpoint_path"],
        decoder_init_seed=int(configuration["decoder_init_seed"]),
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    model.set_class_weights(configuration["loss_configuration"]["class_weights"])
    incompatible = model.load_state_dict(checkpoint["model_state"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("encoder-decoder checkpoint state mismatch")
    return model.to(device).eval(), configuration, {
        "checkpoint": str(checkpoint_path),
        "best_epoch": int(checkpoint["best_epoch"]),
        "best_validation_loss": float(checkpoint["best_validation_loss"]),
        "architecture": configuration["architecture"],
    }


def infer_performance(
    model: EncoderDecoderFiveClassModel,
    tokens: np.ndarray,
    device: torch.device,
    micro_batch_size: int,
) -> dict[str, Any]:
    notes = len(tokens)
    starts = generate_window_starts(notes, 512, 256)
    teacher_windows: list[tuple[int, np.ndarray]] = []
    free_windows: list[tuple[int, np.ndarray]] = []
    with torch.inference_mode():
        for offset in range(0, len(starts), micro_batch_size):
            current = starts[offset : offset + micro_batch_size]
            samples = [_make_window_sample(tokens, start, min(start + 512, notes)) for start in current]
            cpu_batch = five_class_collate_fn(samples)
            for inner in range(0, len(current), micro_batch_size):
                batch = _slice_batch(cpu_batch, inner, min(inner + micro_batch_size, len(current)))
                gpu = {
                    key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
                    for key, value in batch.items()
                }
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                    teacher = model(
                        gpu["input_ids"], gpu["token_attention_mask"],
                        gpu["pedal_targets"], gpu["note_mask"],
                    )
                    free = model(
                        gpu["input_ids"], gpu["token_attention_mask"],
                        note_mask=gpu["note_mask"], decode_mode="greedy",
                    )
                if not bool(torch.isfinite(teacher.logits).all()) or not bool(torch.isfinite(free.logits).all()):
                    raise FloatingPointError("validation logits are non-finite")
                teacher_np = teacher.logits.float().cpu().numpy()
                free_np = free.logits.float().cpu().numpy()
                for index, start in enumerate(current):
                    length = min(512, notes - start)
                    teacher_windows.append((start, teacher_np[index, :length].copy()))
                    free_windows.append((start, free_np[index, :length].copy()))
    teacher_logits, teacher_count = average_five_class_logits(notes, teacher_windows)
    free_logits, free_count = average_five_class_logits(notes, free_windows)
    if not np.array_equal(teacher_count, free_count):
        raise RuntimeError("teacher/free overlap coverage differs")
    return {
        "teacher_predictions": teacher_logits.argmax(axis=-1).astype(np.int64),
        "free_predictions": free_logits.argmax(axis=-1).astype(np.int64),
        "contribution_count": teacher_count,
        "num_windows": len(starts),
    }


def _mode_metrics(
    predictions: Sequence[np.ndarray],
    target_classes: Sequence[np.ndarray],
    raw_targets: Sequence[np.ndarray],
    oracle: Mapping[str, Any],
    provenance: Mapping[str, Any],
    model_name: str,
) -> dict[str, Any]:
    classification = classification_metrics(np.concatenate(predictions), np.concatenate(target_classes))
    renderer = renderer_aware_metrics(list(zip(predictions, target_classes)))
    decoded = decoded_value_metrics(list(zip(predictions, raw_targets)))
    # Reuse the historical row calculator for metric definitions, but its
    # presentation label registry only knows the encoder-only model names.
    flat = _flat_metric_row(
        "endpoint_aware_5class_argmax",
        classification,
        renderer,
        decoded,
        oracle,
        provenance,
    )
    flat["model"] = model_name
    flat["label"] = (
        "Encoder-decoder teacher forced"
        if model_name == TEACHER_MODEL
        else "Encoder-decoder greedy free running"
    )
    flat["is_new_model"] = model_name == FREE_MODEL
    return {
        "classification": classification,
        "renderer": renderer,
        "decoded": decoded,
        "flat": flat,
    }


def _metric_gap(teacher: Mapping[str, Any], free: Mapping[str, Any]) -> dict[str, float]:
    tf = teacher["flat"]
    fr = free["flat"]
    keys = (
        "five_class_token_accuracy",
        "five_class_macro_f1",
        "nonendpoint_endpoint_collapse_ratio",
        "canonical_decoded_overall_mae",
        "canonical_decoded_on_region_mae",
        "canonical_decoded_subthreshold_mae",
        "off_on_state_accuracy",
        "off_on_state_macro_f1",
        "binary_transition_f1",
        "short_repedal_f1",
    )
    return {f"free_minus_teacher_{key}": float(fr[key]) - float(tf[key]) for key in keys}


def _baseline_row() -> dict[str, Any]:
    path = BASELINE_ROOT / "validation" / "validation_comparison.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("model") == "endpoint_aware_5class_argmax":
                return dict(row)
    raise RuntimeError("corrected weighted encoder-only baseline row is missing")


def evaluate(output_dir: str | Path) -> dict[str, Any]:
    started = time.perf_counter()
    root = Path(output_dir).resolve()
    configuration = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if torch.cuda.device_count() != 1:
        raise RuntimeError("validation requires exactly one visible CUDA device")
    device = torch.device("cuda:0")
    gpu = _visible_gpu_identity()
    if gpu["uuid"] != configuration["expected_gpu_uuid"]:
        raise RuntimeError("validation GPU UUID mismatch")
    dataset = Stage2PedalDataset(
        configuration["asap_root"], configuration["split_csv"], "validation",
        window_notes=512, stride_notes=256, return_metadata=True, cache_mode="preload",
    )
    pieces = len({row["piece_id"] for row in dataset.performances})
    notes = sum(int(row["num_normalized_notes"]) for row in dataset.performances)
    if (dataset.performance_count, pieces, notes, dataset.window_count) != (
        EXPECTED_PERFORMANCES, EXPECTED_PIECES, EXPECTED_NOTES, EXPECTED_WINDOWS
    ):
        raise RuntimeError("fixed validation accounting mismatch")
    if any(row["split"] != "validation" for row in dataset.performances):
        raise RuntimeError("non-validation row reached evaluation")
    raw_targets = []
    target_classes = []
    for row in dataset.performances:
        tokens = dataset._token_cache[row["performance_path"]]
        raw = tokens[:, NON_PEDAL_FEATURES:].astype(np.int64) - PEDAL_TOKEN_OFFSET
        raw_targets.append(raw)
        target_classes.append(np.asarray(classify_pedal_values(raw), dtype=np.int64))
    oracle = decoded_value_metrics(list(zip(target_classes, raw_targets)))
    model, checkpoint_config, provenance = _load_model(root / "best.pt", device)
    teacher_predictions = []
    free_predictions = []
    coverage = {"performance_count": 0, "window_count": 0, "minimum_contributions": math.inf, "maximum_contributions": 0}
    micro = int(configuration["micro_batch_size"])
    for index, row in enumerate(dataset.performances):
        result = infer_performance(model, dataset._token_cache[row["performance_path"]], device, micro)
        contributions = result["contribution_count"]
        teacher_predictions.append(result["teacher_predictions"])
        free_predictions.append(result["free_predictions"])
        coverage["performance_count"] += 1
        coverage["window_count"] += int(result["num_windows"])
        coverage["minimum_contributions"] = min(coverage["minimum_contributions"], int(contributions.min()))
        coverage["maximum_contributions"] = max(coverage["maximum_contributions"], int(contributions.max()))
        if index == 0 or (index + 1) % 10 == 0 or index + 1 == dataset.performance_count:
            print(f"ENCODER_DECODER_VALIDATION performances={index + 1}/{dataset.performance_count}", flush=True)
    del model
    torch.cuda.empty_cache()
    teacher = _mode_metrics(
        teacher_predictions, target_classes, raw_targets, oracle,
        {**provenance, "decoder": "teacher-forced argmax after overlap-averaged step logits"},
        TEACHER_MODEL,
    )
    free = _mode_metrics(
        free_predictions, target_classes, raw_targets, oracle,
        {**provenance, "decoder": "greedy autoregressive argmax after overlap-averaged generated-step logits"},
        FREE_MODEL,
    )
    gap = _metric_gap(teacher, free)
    baseline = _baseline_row()
    validation = root / "validation"
    validation.mkdir(parents=True, exist_ok=True)
    comparison_rows = [baseline, teacher["flat"], free["flat"]]
    _atomic_csv(validation / "validation_comparison.csv", comparison_rows)
    per_class_rows = []
    confusion_rows = []
    distribution_rows = []
    for name, metrics in ((TEACHER_MODEL, teacher), (FREE_MODEL, free)):
        classification = metrics["classification"]
        matrix = np.asarray(classification["confusion_matrix"])
        for class_id, class_name in enumerate(CLASS_NAMES):
            per_class_rows.append({
                "model": name, "class_id": class_id, "class_name": class_name,
                "precision": float(classification["precision"][class_id]),
                "recall": float(classification["recall"][class_id]),
                "f1": float(classification["f1"][class_id]),
                "support": int(classification["support"][class_id]),
                "predicted_count": int(classification["predicted_count"][class_id]),
            })
            confusion_rows.append({
                "model": name, "true_class": class_name,
                **{f"pred_{CLASS_NAMES[column].lower()}": int(matrix[class_id, column]) for column in range(NUM_CLASSES)},
            })
            distribution_rows.append({
                "model": name, "class_id": class_id, "class_name": class_name,
                "target_ratio": float(classification["target_distribution"][class_id]),
                "predicted_ratio": float(classification["predicted_distribution"][class_id]),
            })
    _atomic_csv(validation / "per_class_metrics.csv", per_class_rows)
    _atomic_csv(validation / "five_class_confusion.csv", confusion_rows)
    _atomic_csv(validation / "class_distributions.csv", distribution_rows)
    _atomic_json(validation / "renderer_aware_metrics.json", {
        TEACHER_MODEL: teacher["renderer"], FREE_MODEL: free["renderer"]
    })
    result = {
        "completed": True,
        "split": "validation",
        "test_rows_used": 0,
        "primary_result": FREE_MODEL,
        "performance_count": dataset.performance_count,
        "piece_count": pieces,
        "note_count": notes,
        "window_count": dataset.window_count,
        "gpu": gpu,
        "coverage": coverage,
        "teacher_forced": {key: value for key, value in teacher.items() if key != "flat"},
        "free_running": {key: value for key, value in free.items() if key != "flat"},
        "teacher_forced_vs_free_running_gap": gap,
        "comparison": {
            "corrected_weighted_encoder_only": baseline,
            TEACHER_MODEL: teacher["flat"],
            FREE_MODEL: free["flat"],
        },
        "checkpoint_configuration_matches": checkpoint_config == configuration,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(validation / "evaluation.json", result)
    print("ENCODER_DECODER_VALIDATION_COMPLETE", flush=True)
    return result
