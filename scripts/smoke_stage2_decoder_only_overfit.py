#!/usr/bin/env python3
"""Training-only pedal-rich tiny overfit for Decoder-only Stage 2 v0."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_stage2_4class_architecture_v0 import read_json, sha256_file
from src.stage2_decoder_only.model import FourClassPedalDecoderOnlyModel
from src.stage2_decoder_only.training import build_decoder_only_optimizer
from src.stage2_encoder_only.dataset import PEDAL_TOKEN_OFFSET
from src.stage2_encoder_only.train import atomic_torch_save, get_gpu_identity
from src.stage2_encoder_only.training import set_deterministic_seed
from src.stage2_four_class.representation import (
    CLASS_NAMES,
    IGNORE_INDEX,
    NUM_CLASSES,
    SharedFourClassWindowDataset,
    classify_cc64,
    four_class_collate_fn,
)


SOURCES = ("MAESTRO-clean", "ASAP-train")
WINDOWS_PER_SOURCE = 2
SELECTION_RULE = (
    "Within each source, require all four classes and max class ratio <0.90; "
    "rank by OFF/ON boundary crossings descending, canonical class entropy "
    "descending, short ON->OFF->ON runs descending, then window_id ascending; "
    "greedily keep distinct performances. Select two MAESTRO-clean and two "
    "ASAP-train windows."
)


def atomic_json(path: Path, value: Mapping[str, Any] | list[Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def short_on_off_on_count(states: np.ndarray, maximum_off_samples: int = 4) -> int:
    """Selection-only run count; not a canonical evaluation metric."""

    if states.size < 3:
        return 0
    changes = np.flatnonzero(states[1:] != states[:-1]) + 1
    boundaries = np.concatenate(([0], changes, [states.size]))
    count = 0
    for run in range(1, len(boundaries) - 2):
        start, end = int(boundaries[run]), int(boundaries[run + 1])
        if (
            not bool(states[start])
            and bool(states[boundaries[run - 1]])
            and bool(states[boundaries[run + 1]])
            and end - start <= maximum_off_samples
        ):
            count += 1
    return count


def window_statistics(
    dataset: SharedFourClassWindowDataset, window_id: int
) -> dict[str, Any]:
    performance_index, start, end = (
        int(value) for value in dataset.windows[window_id]
    )
    record = dataset.records[performance_index]
    offset = int(record["token_offset"])
    raw = (
        np.asarray(dataset.tokens[offset + start : offset + end, 4:], dtype=np.int64)
        - PEDAL_TOKEN_OFFSET
    )
    classes = np.asarray(classify_cc64(raw), dtype=np.int64)
    flat = classes.reshape(-1)
    counts = np.bincount(flat, minlength=NUM_CLASSES)
    distribution = counts / counts.sum()
    positive = distribution[distribution > 0]
    entropy = float(-(positive * np.log(positive)).sum())
    states = flat >= 2
    crossings = int(np.count_nonzero(states[1:] != states[:-1]))
    return {
        "window_id": int(window_id),
        "source": record["source"],
        "dataset_split": record["dataset_split"],
        "performance_index": performance_index,
        "performance_path": record["performance_path"],
        "window_start_note": start,
        "window_end_note": end,
        "notes": end - start,
        "target_tokens": int(flat.size),
        "class_counts": {
            CLASS_NAMES[index]: int(counts[index]) for index in range(NUM_CLASSES)
        },
        "class_distribution": {
            CLASS_NAMES[index]: float(distribution[index])
            for index in range(NUM_CLASSES)
        },
        "class_coverage": int(np.count_nonzero(counts)),
        "class_entropy_nats": entropy,
        "maximum_class_ratio": float(distribution.max()),
        "off_on_boundary_crossings": crossings,
        "short_on_off_on_runs_max4_samples": short_on_off_on_count(states),
    }


def select_windows(dataset: SharedFourClassWindowDataset) -> list[dict[str, Any]]:
    candidates: dict[str, list[dict[str, Any]]] = {source: [] for source in SOURCES}
    for window_id in range(len(dataset)):
        performance_index = int(dataset.windows[window_id][0])
        record = dataset.records[performance_index]
        source = record["source"]
        if source not in candidates:
            raise RuntimeError(f"non-canonical source reached train cache: {source}")
        # Filter on provenance before reading target tokens.
        if record["dataset_split"].lower() != "train":
            continue
        statistics = window_statistics(dataset, window_id)
        if (
            statistics["class_coverage"] == NUM_CLASSES
            and statistics["maximum_class_ratio"] < 0.90
        ):
            candidates[source].append(statistics)

    selected: list[dict[str, Any]] = []
    for source in SOURCES:
        ranked = sorted(
            candidates[source],
            key=lambda item: (
                -item["off_on_boundary_crossings"],
                -item["class_entropy_nats"],
                -item["short_on_off_on_runs_max4_samples"],
                item["window_id"],
            ),
        )
        used_performances: set[int] = set()
        source_selected = []
        for item in ranked:
            if item["performance_index"] in used_performances:
                continue
            source_selected.append(item)
            used_performances.add(item["performance_index"])
            if len(source_selected) == WINDOWS_PER_SOURCE:
                break
        if len(source_selected) != WINDOWS_PER_SOURCE:
            raise RuntimeError(f"insufficient pedal-rich windows for {source}")
        selected.extend(source_selected)
    return selected


def combined_target_summary(selected: list[dict[str, Any]]) -> dict[str, Any]:
    counts = np.asarray(
        [
            sum(item["class_counts"][name] for item in selected)
            for name in CLASS_NAMES
        ],
        dtype=np.int64,
    )
    total = int(counts.sum())
    return {
        "window_count": len(selected),
        "notes": int(sum(item["notes"] for item in selected)),
        "target_tokens": total,
        "class_counts": {
            name: int(counts[index]) for index, name in enumerate(CLASS_NAMES)
        },
        "class_distribution": {
            name: float(counts[index] / total)
            for index, name in enumerate(CLASS_NAMES)
        },
        "off_on_boundary_crossings": int(
            sum(item["off_on_boundary_crossings"] for item in selected)
        ),
        "short_on_off_on_runs_max4_samples": int(
            sum(item["short_on_off_on_runs_max4_samples"] for item in selected)
        ),
    }


def longest_constant_run(values: np.ndarray) -> int:
    if not values.size:
        return 0
    changes = np.flatnonzero(values[1:] != values[:-1]) + 1
    boundaries = np.concatenate(([0], changes, [values.size]))
    return int(np.diff(boundaries).max())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/stage2_4class_architecture_v0.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "analysis/stage2_decoder_only_4class_v0/smoke_overfit",
    )
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--minimum-epochs", type=int, default=10)
    arguments = parser.parse_args()
    if arguments.max_epochs <= 0 or arguments.minimum_epochs <= 0:
        raise ValueError("epoch limits must be positive")
    if arguments.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite smoke output: {arguments.output_dir}")
    work_dir = arguments.output_dir.with_name(
        f".{arguments.output_dir.name}.work.{os.getpid()}"
    )
    work_dir.mkdir(parents=True)
    predictions_dir = work_dir / "predictions"
    predictions_dir.mkdir()
    log_handle = (work_dir / "train.log").open("w", encoding="utf-8")

    def log(message: str) -> None:
        print(message, flush=True)
        print(message, file=log_handle, flush=True)

    config = read_json(arguments.config)
    if int(config["seed"]) != 42 or int(config["decoder_only_num_layers"]) != 2:
        raise RuntimeError("canonical seed/layer count changed")
    if config["loss"] != "standard_unweighted_cross_entropy":
        raise RuntimeError("canonical loss changed")
    if config["class_weighting"] is not None or config["scheduled_sampling"]:
        raise RuntimeError("non-canonical training option enabled")
    if int(config["window_notes"]) != 512 or int(config["stride_notes"]) != 256:
        raise RuntimeError("canonical window pipeline changed")
    checkpoint = Path(config["checkpoint_path"])
    checkpoint_file = checkpoint / "model.safetensors"
    if sha256_file(checkpoint_file) != config["expected_pretrained_sha256"]:
        raise RuntimeError("pretrained checkpoint hash mismatch")

    dataset = SharedFourClassWindowDataset(config["cache_root"], "train")
    if dataset.cache_id != config["expected_cache_id"]:
        raise RuntimeError("canonical train cache ID mismatch")
    if set(record["source"] for record in dataset.records) != set(SOURCES):
        raise RuntimeError("canonical train source inventory changed")
    selected = select_windows(dataset)
    target_summary = combined_target_summary(selected)
    selected_payload = {
        "cache_id": dataset.cache_id,
        "cache_split": "train",
        "record_dataset_split_filter": "train only before target-statistic access",
        "selection_rule": SELECTION_RULE,
        "tie_break": "ascending canonical train window_id",
        "source_quota": {source: WINDOWS_PER_SOURCE for source in SOURCES},
        "asap_validation_access_count": 0,
        "asap_test_access_count": 0,
        "selected_windows": selected,
        "combined": target_summary,
    }
    atomic_json(work_dir / "selected_windows.json", selected_payload)
    log(
        "SELECTION_COMPLETE "
        f"windows={len(selected)} ids={[item['window_id'] for item in selected]} "
        f"crossings={target_summary['off_on_boundary_crossings']}"
    )

    gpu = get_gpu_identity()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("smoke training requires exactly one visible GPU")
    if gpu["uuid"] != config["expected_gpu_uuid"]:
        raise RuntimeError(f"assigned GPU UUID mismatch: {gpu['uuid']}")
    device = torch.device("cuda:0")
    set_deterministic_seed(42)
    model = FourClassPedalDecoderOnlyModel.from_pretrained_performance_embeddings(
        checkpoint,
        num_layers=2,
        init_seed=42,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    ).to(device)
    optimizer = build_decoder_only_optimizer(
        model,
        pretrained_lr=float(config["encoder_lr"]),
        fresh_lr=float(config["head_lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=True, init_scale=float(config["amp_init_scale"])
    )
    samples = [dataset[item["window_id"]] for item in selected]
    run_config = {
        "experiment": "decoder_only_4class_tiny_overfit_smoke",
        "architecture": {
            "hidden_size": model.hidden_size,
            "ffn_size": 3072,
            "head_dimension": 128,
            "layers": len(model.transformer.layers),
            "performance_prefix": "non-pedal only",
            "cross_attention": False,
        },
        "seed": 42,
        "loss": "standard_unweighted_cross_entropy",
        "class_weighting": None,
        "scheduled_sampling": False,
        "optimizer": "AdamW",
        "pretrained_representation_lr": float(config["encoder_lr"]),
        "fresh_prefix_lm_lr": float(config["head_lr"]),
        "weight_decay": float(config["weight_decay"]),
        "max_grad_norm": float(config["max_grad_norm"]),
        "micro_batch_size": 1,
        "maximum_epochs": int(arguments.max_epochs),
        "minimum_epochs": int(arguments.minimum_epochs),
        "stop_condition": "epoch>=minimum and teacher_forced_CE<=0.03 and accuracy>=0.995",
        "selected_window_ids": [item["window_id"] for item in selected],
        "cache_id": dataset.cache_id,
        "cache_split": "train",
        "training_sources": ["MAESTRO-clean/train", "ASAP-train/train"],
        "asap_validation_access_count": 0,
        "asap_test_access_count": 0,
        "checkpoint_sha256": config["expected_pretrained_sha256"],
        "parameter_count": model.parameter_count,
        "gpu": gpu,
    }
    atomic_json(work_dir / "config.json", run_config)
    log(
        f"MODEL_READY parameters={model.parameter_count} gpu={gpu['uuid']} "
        f"epochs_max={arguments.max_epochs}"
    )

    epoch_rows: list[dict[str, Any]] = []
    gradients_finite = True
    optimizer_steps = 0
    started = time.perf_counter()
    for epoch in range(1, arguments.max_epochs + 1):
        model.train()
        numerator_sum = 0.0
        denominator_sum = 0
        correct = 0
        for sample in samples:
            batch = move_batch(four_class_collate_fn([sample]), device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=True):
                output = model(
                    input_ids=batch["input_ids"],
                    token_attention_mask=batch["token_attention_mask"],
                    pedal_targets=batch["pedal_targets"],
                    note_mask=batch["note_mask"],
                )
            if output.loss is None or not bool(torch.isfinite(output.loss)):
                raise FloatingPointError("non-finite teacher-forced loss")
            scale_before = float(scaler.get_scale())
            scaler.scale(output.loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(config["max_grad_norm"]),
                error_if_nonfinite=False,
            )
            finite = bool(torch.isfinite(grad_norm))
            gradients_finite &= finite
            if not finite:
                raise FloatingPointError("non-finite smoke gradient")
            scaler.step(optimizer)
            scaler.update()
            if float(scaler.get_scale()) < scale_before:
                raise FloatingPointError("AMP skipped a smoke optimizer step")
            optimizer_steps += 1
            targets = batch["pedal_targets"].view(batch["input_ids"].shape[0], -1)
            valid = targets != IGNORE_INDEX
            numerator_sum += float(output.loss_numerator.detach().float().item())
            denominator_sum += int(valid.sum().item())
            correct += int(((output.logits.argmax(dim=-1) == targets) & valid).sum().item())
        row = {
            "epoch": epoch,
            "optimizer_steps": optimizer_steps,
            "teacher_forced_ce": numerator_sum / denominator_sum,
            "teacher_forced_token_accuracy": correct / denominator_sum,
            "gradient_finite": gradients_finite,
            "seconds": time.perf_counter() - started,
        }
        epoch_rows.append(row)
        if epoch == 1 or epoch % 5 == 0:
            log(
                f"EPOCH epoch={epoch} steps={optimizer_steps} "
                f"ce={row['teacher_forced_ce']:.8f} "
                f"accuracy={row['teacher_forced_token_accuracy']:.8f}"
            )
        if (
            epoch >= arguments.minimum_epochs
            and row["teacher_forced_ce"] <= 0.03
            and row["teacher_forced_token_accuracy"] >= 0.995
        ):
            log(f"OVERFIT_STOP epoch={epoch} reason=success_condition")
            break

    model.eval()
    tf_numerator = 0.0
    tf_denominator = 0
    tf_correct = 0
    target_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    prediction_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    free_correct = 0
    free_notes = 0
    free_exact_notes = 0
    collapse_diagnostics = []
    for item, sample in zip(selected, samples):
        batch = move_batch(four_class_collate_fn([sample]), device)
        with torch.no_grad(), torch.amp.autocast(
            "cuda", dtype=torch.float16, enabled=True
        ):
            teacher = model(
                input_ids=batch["input_ids"],
                token_attention_mask=batch["token_attention_mask"],
                pedal_targets=batch["pedal_targets"],
                note_mask=batch["note_mask"],
            )
            generated = model.generate(
                batch["input_ids"], batch["token_attention_mask"], batch["note_mask"]
            )
        valid_notes = batch["note_mask"].bool()
        targets = batch["pedal_targets"][valid_notes].view(-1, 4)
        predicted = generated[valid_notes].view(-1, 4)
        if predicted.shape != targets.shape or predicted.shape[1] != 4:
            raise RuntimeError("free-running output shape/slot indexing mismatch")
        if not bool(((predicted >= 0) & (predicted < NUM_CLASSES)).all()):
            raise RuntimeError("free-running class escaped [0,3]")
        flat_targets = targets.reshape(-1)
        flat_predicted = predicted.reshape(-1)
        tf_numerator += float(teacher.loss_numerator.detach().float().item())
        tf_denominator += int(flat_targets.numel())
        tf_correct += int(
            (teacher.logits[:, : flat_targets.numel()].argmax(dim=-1).reshape(-1) == flat_targets).sum().item()
        )
        free_correct += int((flat_predicted == flat_targets).sum().item())
        free_notes += int(targets.shape[0])
        free_exact_notes += int((predicted == targets).all(dim=-1).sum().item())
        target_counts += np.bincount(flat_targets.cpu().numpy(), minlength=NUM_CLASSES)
        prediction_counts += np.bincount(
            flat_predicted.cpu().numpy(), minlength=NUM_CLASSES
        )
        prediction_array = flat_predicted.cpu().numpy()
        diagnostic = {
            "window_id": item["window_id"],
            "tokens": int(prediction_array.size),
            "unique_predicted_classes": sorted(
                int(value) for value in np.unique(prediction_array)
            ),
            "single_class_collapse": bool(np.unique(prediction_array).size == 1),
            "longest_constant_prediction_run": longest_constant_run(
                prediction_array
            ),
            "longest_constant_run_ratio": float(
                longest_constant_run(prediction_array) / prediction_array.size
            ),
        }
        collapse_diagnostics.append(diagnostic)
        atomic_json(
            predictions_dir / f"window_{item['window_id']}.json",
            {
                **diagnostic,
                "target_note_major_pedal1_to_4": targets.cpu().tolist(),
                "prediction_note_major_pedal1_to_4": predicted.cpu().tolist(),
            },
        )

    total_tokens = int(target_counts.sum())
    final = {
        "status": "pass",
        "selection": target_summary,
        "training": {
            "epochs": int(epoch_rows[-1]["epoch"]),
            "optimizer_steps": optimizer_steps,
            "gradient_finite": gradients_finite,
            "loss_finite": all(math.isfinite(row["teacher_forced_ce"]) for row in epoch_rows),
            "initial_teacher_forced_ce": epoch_rows[0]["teacher_forced_ce"],
            "initial_teacher_forced_token_accuracy": epoch_rows[0]["teacher_forced_token_accuracy"],
            "final_train_epoch_ce": epoch_rows[-1]["teacher_forced_ce"],
            "final_train_epoch_token_accuracy": epoch_rows[-1]["teacher_forced_token_accuracy"],
            "teacher_forced_eval_ce": tf_numerator / tf_denominator,
            "teacher_forced_eval_token_accuracy": tf_correct / tf_denominator,
            "epoch_history": epoch_rows,
        },
        "free_running": {
            "ground_truth_history_provided": False,
            "token_accuracy": free_correct / total_tokens,
            "exact_note_accuracy": free_exact_notes / free_notes,
            "output_tokens": total_tokens,
            "output_notes": free_notes,
            "class_range_valid": True,
            "target_class_counts": {
                name: int(target_counts[index]) for index, name in enumerate(CLASS_NAMES)
            },
            "target_class_distribution": {
                name: float(target_counts[index] / total_tokens)
                for index, name in enumerate(CLASS_NAMES)
            },
            "prediction_class_counts": {
                name: int(prediction_counts[index]) for index, name in enumerate(CLASS_NAMES)
            },
            "prediction_class_distribution": {
                name: float(prediction_counts[index] / total_tokens)
                for index, name in enumerate(CLASS_NAMES)
            },
            "collapse_diagnostics": collapse_diagnostics,
        },
        "indexing_checks": {
            "flatten_order": "note-major Pedal1,Pedal2,Pedal3,Pedal4",
            "expected_tokens": total_tokens,
            "actual_tokens": total_tokens,
            "shape_and_length_match": True,
            "padding_aggregated": False,
            "nan_or_inf": False,
        },
        "maximum_cuda_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "full_training_started": False,
        "asap_validation_access_count": 0,
        "asap_test_access_count": 0,
    }
    overfit_succeeded = (
        final["training"]["teacher_forced_eval_ce"] <= 0.05
        and final["training"]["teacher_forced_eval_token_accuracy"] >= 0.99
    )
    if not overfit_succeeded:
        final["status"] = "fail_teacher_forced_overfit"
    if any(item["single_class_collapse"] for item in collapse_diagnostics):
        final["status"] = "fail_free_running_single_class_collapse"
    atomic_json(work_dir / "metrics.json", final)
    atomic_torch_save(
        {
            "artifact_use": "smoke_test_only_not_for_full_training",
            "model_state": model.state_dict(),
            "configuration": run_config,
            "metrics": {key: value for key, value in final.items() if key != "training"},
            "epochs": int(epoch_rows[-1]["epoch"]),
            "optimizer_steps": optimizer_steps,
        },
        work_dir / "smoke_final.pt",
    )
    log(
        f"SMOKE_COMPLETE status={final['status']} epochs={epoch_rows[-1]['epoch']} "
        f"tf_ce={final['training']['teacher_forced_eval_ce']:.8f} "
        f"tf_acc={final['training']['teacher_forced_eval_token_accuracy']:.8f} "
        f"free_acc={final['free_running']['token_accuracy']:.8f} "
        f"free_exact={final['free_running']['exact_note_accuracy']:.8f}"
    )
    log_handle.close()
    arguments.output_dir.parent.mkdir(parents=True, exist_ok=True)
    work_dir.rename(arguments.output_dir)
    return 0 if final["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
