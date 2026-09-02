#!/usr/bin/env python3
"""Train and orchestrate the canonical raw-CC64 Encoder-only Huber experiment."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_stage2_4class_architecture_v0 import (
    atomic_json,
    parameter_hash,
    preflight as canonical_preflight,
    read_json,
)
from src.stage2_binary.full_training import SharedBinaryWindowDataset, training_window_order
from src.stage2_binary.training import build_binary_optimizer
from src.stage2_encoder_only.dataset import PEDAL_TOKEN_OFFSET
from src.stage2_encoder_only.train import atomic_torch_save, get_gpu_identity
from src.stage2_encoder_only.training import set_deterministic_seed
from src.stage2_four_class.loss_objectives import (
    HUBER_DELTA_CC64,
    HUBER_DELTA_NORMALIZED,
    continuous_cc64_to_classes,
    normalized_scalars_to_cc64,
)
from src.stage2_four_class.raw_cc64_huber import (
    RawCC64HuberEncoderModel,
    make_raw_cc64_loader,
)
from src.stage2_four_class.representation import IGNORE_INDEX, classify_cc64


REPRESENTATIVE_BASELINE = {
    "accuracy": 0.4774428144515958,
    "macro_f1": 0.3961580585415642,
    "transition_precision": 0.471488,
    "transition_recall": 0.521667,
    "transition_f1": 0.495310,
    "candidate_transitions": 38247,
    "low_recall": 0.21681967974806618,
    "half_recall": 0.28151326461200354,
    "js_divergence": 0.042943,
    "intersection": 0.810684,
    "best_epoch": 6,
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_config(config: Mapping[str, Any]) -> dict[str, Any]:
    base = read_json(config["canonical_config"])
    if Path(base["output_root"]) == Path(config["output_root"]):
        raise RuntimeError("raw run output overlaps canonical architecture output")
    return base


def update_status(config: Mapping[str, Any], **updates: Any) -> dict[str, Any]:
    output = Path(config["output_root"])
    path = output / "run_status.json"
    current = read_json(path) if path.is_file() else {}
    current.update(updates, last_update=now(), asap_test_access_count=0)
    atomic_json(path, current)
    return current


def scan_training_raw_targets(base: Mapping[str, Any]) -> dict[str, Any]:
    cache_root = Path(base["cache_root"])
    stats = read_json(cache_root / "cache_statistics.json")
    tokens = np.load(
        cache_root / stats["splits"]["train"]["tokens_file"], mmap_mode="r"
    )
    counts = np.zeros(128, dtype=np.int64)
    for start in range(0, len(tokens), 250_000):
        raw = np.asarray(tokens[start : start + 250_000, 4:], dtype=np.int64)
        raw -= PEDAL_TOKEN_OFFSET
        if raw.min() < 0 or raw.max() > 127:
            raise RuntimeError("canonical cache contains an invalid raw CC64 target")
        counts += np.bincount(raw.reshape(-1), minlength=128)
    present = np.flatnonzero(counts)
    representatives = {0, 51, 79, 127}
    non_representatives = [int(value) for value in present if int(value) not in representatives]
    total = int(counts.sum())
    expected = int(stats["splits"]["train"]["notes"]) * 4
    if total != expected or len(present) <= 4 or not non_representatives:
        raise RuntimeError("raw target diversity audit failed")
    return {
        "source": "canonical training cache raw pedal tokens only",
        "total_targets": total,
        "unique_raw_values": int(len(present)),
        "minimum": int(present.min()),
        "maximum": int(present.max()),
        "present_values": present.tolist(),
        "non_representative_unique_values": len(non_representatives),
        "non_representative_examples": non_representatives[:24],
        "representative_quantization_applied": False,
        "validation_frequency_access": 0,
        "test_access_count": 0,
    }


def preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    if config.get("objective") != "raw_cc64_huber":
        raise RuntimeError("objective must be raw_cc64_huber")
    if float(config["huber_delta_cc64"]) != HUBER_DELTA_CC64 or not math.isclose(
        float(config["huber_delta_normalized"]), HUBER_DELTA_NORMALIZED,
        rel_tol=0.0, abs_tol=1e-15,
    ):
        raise RuntimeError("canonical Huber delta changed")
    if int(config["asap_test_access"]) != 0:
        raise RuntimeError("ASAP test access must be zero")
    if bool(config["representative_checkpoint_used_for_initialization"]):
        raise RuntimeError("representative checkpoint initialization is forbidden")
    base = canonical_config(config)
    audit = canonical_preflight(base)
    raw_audit = scan_training_raw_targets(base)
    representative_root = Path(config["representative_huber_root"])
    representative_status = read_json(representative_root / "run_status.json")
    representative_config = read_json(representative_root / "config.json")
    representative_eval = read_json(config["representative_huber_evaluation"])
    if representative_status.get("status") != "completed" or int(
        representative_status["best_epoch"]
    ) != 6:
        raise RuntimeError("Representative Huber baseline is not the completed epoch-6 run")
    classification = representative_eval["classification"]
    transition = representative_eval["transition"]["pooled"]
    patterns = representative_eval["patterns"]
    checks = {
        "accuracy": classification["token_accuracy"],
        "macro_f1": classification["macro_f1"],
        "candidate_transitions": transition["candidate"],
    }
    if not math.isclose(checks["accuracy"], REPRESENTATIVE_BASELINE["accuracy"], abs_tol=1e-12):
        raise RuntimeError("Representative Huber accuracy changed")
    if not math.isclose(checks["macro_f1"], REPRESENTATIVE_BASELINE["macro_f1"], abs_tol=1e-12):
        raise RuntimeError("Representative Huber Macro F1 changed")
    if int(checks["candidate_transitions"]) != REPRESENTATIVE_BASELINE["candidate_transitions"]:
        raise RuntimeError("Representative Huber transition count changed")
    if representative_config["architecture"] != "PT pretrained 10-layer encoder + four independent Linear(768,1) heads":
        raise RuntimeError("Representative Huber architecture provenance changed")
    if float(representative_config["huber_delta_normalized"]) != HUBER_DELTA_NORMALIZED:
        raise RuntimeError("Representative Huber delta provenance changed")
    return {
        "canonical_audit": audit,
        "raw_target_audit": raw_audit,
        "representative_baseline": {
            **REPRESENTATIVE_BASELINE,
            "root": str(representative_root),
            "checkpoint_read_for_initialization": False,
            "initial_head_parameter_sha256": representative_config[
                "initial_head_parameter_sha256"
            ],
            "successful_validation_pairs": representative_eval["successful_pairs"],
            "aligned_notes": representative_eval["aligned_notes"],
            "excluded_alignment_failures": representative_eval["alignment_failures"],
            "transition_exact": transition,
            "patterns_exact": {
                "js_divergence": patterns["js_divergence_base2"],
                "intersection": patterns["intersection"],
            },
        },
    }


def build_fresh_model(base: Mapping[str, Any]) -> RawCC64HuberEncoderModel:
    model = RawCC64HuberEncoderModel.from_pretrained(
        base["checkpoint_path"],
        delta=HUBER_DELTA_NORMALIZED,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    if len(model.encoder.layers) != 10 or model.hidden_size != 768:
        raise RuntimeError("pretrained PT encoder architecture changed")
    if len(model.regression_heads) != 4 or any(
        head.in_features != 768 or head.out_features != 1
        for head in model.regression_heads
    ):
        raise RuntimeError("raw Huber scalar-head architecture changed")
    return model


def move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def run_epoch(
    model: RawCC64HuberEncoderModel,
    loader: Any,
    *,
    device: torch.device,
    amp_enabled: bool,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    max_grad_norm: float = 1.0,
    step_callback: Any = None,
    max_batches: int | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    training = optimizer is not None
    if training and scaler is None:
        raise ValueError("training requires GradScaler")
    model.train(training)
    huber_sum = absolute_sum = squared_sum = count = 0.0
    prediction_sum = prediction_square_sum = 0.0
    target_sum = target_square_sum = 0.0
    prediction_min, prediction_max = math.inf, -math.inf
    clip_zero = clip_full = 0
    correct = exact = notes = 0
    optimizer_steps = amp_skips = 0
    for step, cpu_batch in enumerate(loader, 1):
        if max_batches is not None and step > max_batches:
            break
        batch = move_batch(cpu_batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        use_amp = bool(amp_enabled and device.type == "cuda")
        with torch.set_grad_enabled(training), torch.amp.autocast(
            device_type=device.type,
            dtype=torch.float16 if use_amp else torch.bfloat16,
            enabled=use_amp,
        ):
            output = model(
                input_ids=batch["input_ids"],
                token_attention_mask=batch["token_attention_mask"],
                note_mask=batch["note_mask"],
                pedal_targets=batch["pedal_targets"],
            )
        if output.loss is None or output.loss_numerator is None or output.loss_denominator is None:
            raise RuntimeError("raw Huber model omitted objective components")
        if not bool(torch.isfinite(output.loss)) or not bool(torch.isfinite(output.loss_numerator)):
            raise FloatingPointError("raw Huber loss is non-finite")
        if training:
            if use_amp:
                scaler.scale(output.loss).backward()
                scaler.unscale_(optimizer)
            else:
                output.loss.backward()
            parameters = [value for value in model.parameters() if value.requires_grad]
            gradients = [value.grad for value in parameters if value.grad is not None]
            grad_norm = torch.nn.utils.get_total_norm(gradients, norm_type=2.0)
            grad_norm_value = float(grad_norm.detach().float().item())
            finite_gradient = math.isfinite(grad_norm_value)
            scale_before = float(scaler.get_scale()) if use_amp else None
            if finite_gradient:
                torch.nn.utils.clip_grads_with_norm_(
                    parameters, max_norm=float(max_grad_norm), total_norm=grad_norm
                )
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
                scale_after = float(scaler.get_scale())
                applied = finite_gradient and scale_after >= float(scale_before)
                if not applied:
                    amp_skips += 1
            else:
                if not finite_gradient:
                    raise FloatingPointError("non-finite raw Huber gradient without AMP")
                optimizer.step()
                scale_after = None
                applied = True
            if applied:
                optimizer_steps += 1
            if step_callback is not None:
                raw = batch["pedal_targets"]
                valid_raw = raw[raw != IGNORE_INDEX]
                unique = torch.unique(valid_raw)
                non_representative = unique[
                    (unique != 0) & (unique != 51) & (unique != 79) & (unique != 127)
                ]
                step_callback(step, {
                    "loss": float(output.loss.detach().float().item()),
                    "gradient_norm": grad_norm_value,
                    "optimizer_step_applied": applied,
                    "optimizer_steps": optimizer_steps,
                    "amp_skipped_steps": amp_skips,
                    "scale_before": scale_before,
                    "scale_after": scale_after,
                    "raw_unique_count": int(unique.numel()),
                    "raw_unique_examples": unique[:24].detach().cpu().tolist(),
                    "non_representative_unique_count": int(non_representative.numel()),
                    "non_representative_examples": non_representative[:24].detach().cpu().tolist(),
                })
        valid = batch["pedal_targets"] != IGNORE_INDEX
        raw_targets = batch["pedal_targets"][valid].float()
        raw_unclipped = output.predictions.detach()[valid].float() * 127.0
        raw_predictions = raw_unclipped.clamp(0.0, 127.0)
        errors = raw_predictions - raw_targets
        batch_count = float(valid.sum().item())
        huber_sum += float(output.loss_numerator.detach().float().item())
        absolute_sum += float(errors.abs().sum().item())
        squared_sum += float(errors.square().sum().item())
        count += batch_count
        prediction_sum += float(raw_predictions.sum().item())
        prediction_square_sum += float(raw_predictions.square().sum().item())
        target_sum += float(raw_targets.sum().item())
        target_square_sum += float(raw_targets.square().sum().item())
        prediction_min = min(prediction_min, float(raw_unclipped.min().item()))
        prediction_max = max(prediction_max, float(raw_unclipped.max().item()))
        clip_zero += int((raw_unclipped <= 0).sum().item())
        clip_full += int((raw_unclipped >= 127).sum().item())
        predicted_classes = continuous_cc64_to_classes(raw_predictions)
        target_classes = classify_cc64(batch["pedal_targets"], allow_ignore=True)[valid]
        correct += int((predicted_classes == target_classes).sum().item())
        valid_notes = valid.all(dim=-1)
        predicted_full = torch.full_like(batch["pedal_targets"], IGNORE_INDEX)
        predicted_full[valid] = predicted_classes
        exact += int((((predicted_full == classify_cc64(batch["pedal_targets"], allow_ignore=True)) | ~valid).all(dim=-1) & valid_notes).sum().item())
        notes += int(valid_notes.sum().item())
    if count <= 0 or notes <= 0:
        raise RuntimeError("raw Huber epoch was empty")
    prediction_mean = prediction_sum / count
    target_mean = target_sum / count
    metrics = {
        "huber": huber_sum / count,
        "raw_cc64_mae": absolute_sum / count,
        "raw_cc64_rmse": math.sqrt(squared_sum / count),
        "prediction_mean": prediction_mean,
        "prediction_std": math.sqrt(max(0.0, prediction_square_sum / count - prediction_mean ** 2)),
        "target_mean": target_mean,
        "target_std": math.sqrt(max(0.0, target_square_sum / count - target_mean ** 2)),
        "raw_prediction_min_before_clipping": prediction_min,
        "raw_prediction_max_before_clipping": prediction_max,
        "clip_at_0_proportion": clip_zero / count,
        "clip_at_127_proportion": clip_full / count,
        "canonical_4class_accuracy_diagnostic": correct / count,
        "exact_note_accuracy_diagnostic": exact / notes,
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise FloatingPointError("raw Huber epoch metric is non-finite")
    return metrics, {"optimizer_steps": optimizer_steps, "amp_skipped_steps": amp_skips}


def write_metrics(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def checkpoint_payload(
    model: RawCC64HuberEncoderModel,
    run_config: Mapping[str, Any],
    row: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "model_state": model.state_dict(),
        "configuration": dict(run_config),
        "objective": "raw_cc64_huber",
        "epoch": int(row["epoch"]),
        "validation_objective": float(row["validation_huber"]),
        "validation_raw_cc64_mae": float(row["validation_raw_cc64_mae"]),
        "checkpoint_selection": "minimum validation raw-CC64 Huber objective",
    }


def train(config: Mapping[str, Any]) -> int:
    output = Path(config["output_root"])
    if any((output / name).exists() for name in ("best.pt", "last.pt", "config.json", "metrics.csv", "train.log")):
        raise FileExistsError("refusing to overwrite raw Huber full-run artifacts")
    output.mkdir(parents=True, exist_ok=True)
    log_handle = (output / "train.log").open("w", encoding="utf-8")

    def log(message: str) -> None:
        line = f"{now()} {message}"
        print(line, flush=True)
        print(line, file=log_handle, flush=True)

    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("training requires exactly one visible CUDA GPU")
        device = torch.device("cuda:0")
        gpu = get_gpu_identity()
        base = canonical_config(config)
        if gpu["uuid"] != base["expected_gpu_uuid"]:
            raise RuntimeError(f"assigned GPU UUID mismatch: {gpu['uuid']}")
        phase = preflight(config)
        audit = phase["canonical_audit"]
        raw_audit = phase["raw_target_audit"]
        log(f"CACHE_LOAD_PASS cache_id={audit['cache_id']} train=1170+892 validation=71 test_access=0")
        log(
            f"RAW_TARGET_DIVERSITY_PASS unique={raw_audit['unique_raw_values']} "
            f"non_representative={raw_audit['non_representative_unique_values']} "
            f"examples={raw_audit['non_representative_examples']} quantization=false"
        )
        set_deterministic_seed(int(base["seed"]))
        model = build_fresh_model(base)
        encoder_hash = parameter_hash(model.encoder.parameters())
        head_hash = parameter_hash(model.prediction_head_parameters())
        if encoder_hash != base["expected_initial_encoder_sha256"]:
            raise RuntimeError("fresh pretrained encoder parameter hash mismatch")
        expected_head_hash = phase["representative_baseline"]["initial_head_parameter_sha256"]
        if head_hash != expected_head_hash:
            raise RuntimeError("fresh scalar head initialization differs from Representative Huber")
        run_config = {
            **base,
            **dict(config),
            "architecture_name": "encoder_only",
            "architecture": "PT pretrained 10-layer encoder + four independent Linear(768,1) heads",
            "output_shape": "B,N,4",
            "target_dtype": "cached raw integer CC64, normalized float target=y_raw/127",
            "target_quantization": None,
            "training_output_activation": "none (unconstrained scalar)",
            "fresh_initialization_seed": int(base["seed"]),
            "initialization_source": base["checkpoint_path"],
            "representative_checkpoint_loaded_for_initialization": False,
            "initial_encoder_parameter_sha256": encoder_hash,
            "initial_head_parameter_sha256": head_hash,
            "canonical_cache_audit": audit,
            "raw_target_audit": raw_audit,
            "representative_baseline": phase["representative_baseline"],
            "micro_batch_size": int(base["micro_batch_size"]["encoder_only"]),
            "gradient_accumulation_steps": int(base["gradient_accumulation_steps"]["encoder_only"]),
            "gpu": gpu,
            "asap_test_access": 0,
        }
        atomic_json(output / "config.json", run_config)
        update_status(
            config, status="running", current_stage="model_initialized",
            fresh_seed=42, gpu=gpu, cache_id=audit["cache_id"],
            raw_target_unique_values=raw_audit["unique_raw_values"],
            raw_target_quantization=False,
            representative_checkpoint_used_for_initialization=False,
        )
        log(
            f"MODEL_INITIALIZED objective=raw_cc64_huber seed=42 "
            f"encoder_sha256={encoder_hash} head_sha256={head_hash} representative_checkpoint_used=false"
        )
        train_dataset = SharedBinaryWindowDataset(base["cache_root"], "train")
        validation_dataset = SharedBinaryWindowDataset(base["cache_root"], "validation")
        model.to(device)
        optimizer = build_binary_optimizer(
            model, encoder_lr=float(base["encoder_lr"]), head_lr=float(base["head_lr"]),
            weight_decay=float(base["weight_decay"]),
        )
        scaler = torch.amp.GradScaler(
            "cuda", enabled=bool(base["amp_enabled"]), init_scale=float(base["amp_init_scale"])
        )
        validation_loader = make_raw_cc64_loader(
            validation_dataset, batch_size=int(base["micro_batch_size"]["encoder_only"]),
            pin_memory=bool(base["pin_memory"]),
        )
        rows: list[dict[str, Any]] = []
        best = early_best = math.inf
        best_epoch = stale = 0
        first_step_logged = False
        for epoch in range(1, int(base["max_epochs"]) + 1):
            started = time.perf_counter()
            order = training_window_order(len(train_dataset), int(base["seed"]), epoch)
            loader = make_raw_cc64_loader(
                train_dataset, batch_size=int(base["micro_batch_size"]["encoder_only"]),
                pin_memory=bool(base["pin_memory"]), order=order,
            )

            def callback(step: int, details: Mapping[str, Any]) -> None:
                nonlocal first_step_logged
                if not first_step_logged and details["optimizer_step_applied"]:
                    loss = float(details["loss"])
                    if not math.isfinite(loss) or int(details["non_representative_unique_count"]) <= 0:
                        raise FloatingPointError("first raw Huber step or target diversity is invalid")
                    first_step_logged = True
                    update_status(
                        config, status="running", current_stage="first_optimizer_step_pass",
                        first_optimizer_step=True, first_step_epoch=epoch, first_step=step,
                        first_step_loss=loss,
                        first_step_gradient_norm=float(details["gradient_norm"]),
                        first_batch_raw_unique_count=int(details["raw_unique_count"]),
                        first_batch_raw_unique_examples=details["raw_unique_examples"],
                        first_batch_non_representative_unique_count=int(details["non_representative_unique_count"]),
                        first_batch_non_representative_examples=details["non_representative_examples"],
                    )
                    log(
                        f"FIRST_OPTIMIZER_STEP_PASS epoch={epoch} step={step} "
                        f"loss={loss:.9f} grad_norm={float(details['gradient_norm']):.6f} "
                        f"raw_unique={details['raw_unique_count']} "
                        f"non_representative_unique={details['non_representative_unique_count']}"
                    )
                elif step % int(base["progress_interval"]) == 0:
                    update_status(config, status="running", current_stage="training", current_epoch=epoch, current_step=step)
                    log(f"TRAIN_PROGRESS epoch={epoch} step={step} loss={float(details['loss']):.9f}")

            train_metrics, checks = run_epoch(
                model, loader, device=device, amp_enabled=bool(base["amp_enabled"]),
                optimizer=optimizer, scaler=scaler,
                max_grad_norm=float(base["max_grad_norm"]), step_callback=callback,
            )
            if not first_step_logged:
                raise RuntimeError("finite first raw Huber optimizer step was not verified")
            validation_metrics, _ = run_epoch(
                model, validation_loader, device=device,
                amp_enabled=bool(base["amp_enabled"]),
            )
            row: dict[str, Any] = {
                "epoch": epoch,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"validation_{key}": value for key, value in validation_metrics.items()},
                "encoder_lr": float(optimizer.param_groups[0]["lr"]),
                "head_lr": float(optimizer.param_groups[1]["lr"]),
                "optimizer_steps": checks["optimizer_steps"],
                "amp_skipped_steps": checks["amp_skipped_steps"],
                "epoch_seconds": time.perf_counter() - started,
            }
            rows.append(row)
            write_metrics(output / "metrics.csv", rows)
            candidate = float(row["validation_huber"])
            if candidate < best:
                best, best_epoch = candidate, epoch
                atomic_torch_save(checkpoint_payload(model, run_config, row), output / "best.pt")
            if candidate < early_best - float(base["early_stopping_min_delta"]):
                early_best, stale = candidate, 0
            else:
                stale += 1
            atomic_torch_save(checkpoint_payload(model, run_config, row), output / "last.pt")
            update_status(
                config, status="running", current_stage="epoch_completed",
                current_epoch=epoch, best_epoch=best_epoch,
                best_validation_objective=best,
                validation_huber=float(row["validation_huber"]),
                validation_raw_cc64_mae=float(row["validation_raw_cc64_mae"]),
            )
            log(
                f"EPOCH_COMPLETE epoch={epoch} train_huber={float(row['train_huber']):.9f} "
                f"validation_huber={candidate:.9f} "
                f"validation_raw_cc64_mae={float(row['validation_raw_cc64_mae']):.6f} "
                f"best_epoch={best_epoch}"
            )
            if stale >= int(base["early_stopping_patience"]):
                log(f"EARLY_STOP epoch={epoch}")
                break
        if not all((output / name).is_file() for name in ("best.pt", "last.pt", "config.json", "metrics.csv", "train.log")):
            raise RuntimeError("required raw Huber training artifacts are missing")
        update_status(
            config, status="trained", current_stage="training_completed",
            training_completed=True, best_epoch=best_epoch,
            best_validation_objective=best,
        )
        log(f"TRAINING_COMPLETE best_epoch={best_epoch} best_validation_huber={best:.9f}")
        return 0
    except Exception as error:
        log(f"TRAINING_FAILED {type(error).__name__}: {error}")
        traceback.print_exc(file=log_handle)
        update_status(
            config, status="failed", current_stage="training_failed",
            error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc(),
        )
        return 1
    finally:
        log_handle.close()


def smoke(config: Mapping[str, Any]) -> int:
    output = Path(config["output_root"])
    smoke_root = output / "preflight_smoke"
    if smoke_root.exists():
        raise FileExistsError(f"refusing to overwrite smoke output: {smoke_root}")
    output.mkdir(parents=True, exist_ok=True)
    phase = preflight(config)
    base = canonical_config(config)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("smoke requires exactly one visible CUDA GPU")
    if get_gpu_identity()["uuid"] != base["expected_gpu_uuid"]:
        raise RuntimeError("smoke GPU UUID mismatch")
    device = torch.device("cuda:0")
    set_deterministic_seed(int(base["seed"]))
    model = build_fresh_model(base).to(device)
    before = parameter_hash(model.prediction_head_parameters())
    if before != phase["representative_baseline"]["initial_head_parameter_sha256"]:
        raise RuntimeError("smoke fresh head differs from Representative Huber initialization")
    optimizer = build_binary_optimizer(
        model, encoder_lr=float(base["encoder_lr"]), head_lr=float(base["head_lr"]),
        weight_decay=float(base["weight_decay"]),
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=bool(base["amp_enabled"]), init_scale=float(base["amp_init_scale"])
    )
    train_dataset = SharedBinaryWindowDataset(base["cache_root"], "train")
    validation_dataset = SharedBinaryWindowDataset(base["cache_root"], "validation")
    first_details: dict[str, Any] = {}
    train_loader = make_raw_cc64_loader(
        train_dataset, batch_size=1, pin_memory=bool(base["pin_memory"]), order=[0],
    )
    train_metrics, checks = run_epoch(
        model, train_loader, device=device, amp_enabled=bool(base["amp_enabled"]),
        optimizer=optimizer, scaler=scaler, max_grad_norm=float(base["max_grad_norm"]),
        step_callback=lambda _step, details: first_details.update(details), max_batches=1,
    )
    after = parameter_hash(model.prediction_head_parameters())
    validation_loader = make_raw_cc64_loader(
        validation_dataset, batch_size=1, pin_memory=bool(base["pin_memory"]), order=[0],
    )
    validation_metrics, _ = run_epoch(
        model, validation_loader, device=device,
        amp_enabled=bool(base["amp_enabled"]), max_batches=1,
    )
    smoke_root.mkdir(parents=True)
    checkpoint = smoke_root / "raw_cc64_huber.pt"
    atomic_torch_save({"model_state": model.state_dict(), "objective": "raw_cc64_huber"}, checkpoint)
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(loaded["model_state"], strict=True)
    if not (
        checks["optimizer_steps"] == 1 and before != after
        and math.isfinite(train_metrics["huber"])
        and math.isfinite(validation_metrics["huber"])
        and int(first_details.get("non_representative_unique_count", 0)) > 0
        and not incompatible.missing_keys and not incompatible.unexpected_keys
    ):
        raise RuntimeError("raw CC64 Huber smoke failed")
    result = {
        "status": "passed",
        "finite_forward": True,
        "finite_huber_loss": train_metrics["huber"],
        "finite_backward": True,
        "optimizer_step": True,
        "checkpoint_serialization": True,
        "checkpoint_reload": True,
        "finite_validation_huber": validation_metrics["huber"],
        "head_parameters_updated": True,
        "fresh_head_matches_representative_initialization": True,
        "representative_checkpoint_used_for_initialization": False,
        "raw_batch_unique_count": first_details["raw_unique_count"],
        "raw_batch_unique_examples": first_details["raw_unique_examples"],
        "raw_batch_non_representative_unique_count": first_details[
            "non_representative_unique_count"
        ],
        "raw_batch_non_representative_examples": first_details[
            "non_representative_examples"
        ],
        "target_quantization": False,
        "checkpoint": str(checkpoint),
        "asap_test_access_count": 0,
        "canonical_audit": phase["canonical_audit"],
        "raw_target_audit": phase["raw_target_audit"],
    }
    atomic_json(smoke_root / "smoke_results.json", result)
    return 0


def run_process(command: Sequence[str], log_handle: Any) -> int:
    process = subprocess.Popen(
        list(command), cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        print(line, end="", file=log_handle, flush=True)
    return process.wait()


def overnight(config: Mapping[str, Any], config_path: Path) -> int:
    output = Path(config["output_root"])
    smoke_path = output / "preflight_smoke" / "smoke_results.json"
    if not smoke_path.is_file() or read_json(smoke_path).get("status") != "passed":
        raise RuntimeError("full training is blocked until raw Huber smoke passes")
    if any((output / name).exists() for name in ("best.pt", "last.pt", "config.json", "metrics.csv", "train.log", "overnight.log")):
        raise FileExistsError("refusing to overwrite raw Huber full-run output")
    update_status(
        config, experiment_id=config["experiment_id"], status="running",
        current_stage="starting", training_completed=False,
        evaluation_completed=False, training_report_generated=False,
        comparison_report_generated=False, fresh_seed=42,
        representative_checkpoint_used_for_initialization=False,
        started_at=now(),
    )
    with (output / "overnight.log").open("w", encoding="utf-8") as log_handle:
        train_command = [
            sys.executable, str(Path(__file__).resolve()), "--config", str(config_path), "--train"
        ]
        train_code = run_process(train_command, log_handle)
        if train_code != 0:
            update_status(config, status="failed", current_stage="training_failed", train_return_code=train_code)
            return train_code
        update_status(config, status="running", current_stage="validation_evaluation")
        eval_command = [
            sys.executable,
            str(ROOT / "scripts/evaluate_stage2_encoder_only_raw_cc64_huber_v0.py"),
            "--config", str(config_path), "--execute",
        ]
        eval_code = run_process(eval_command, log_handle)
        if eval_code != 0:
            update_status(config, status="failed", current_stage="validation_evaluation_failed", evaluation_return_code=eval_code)
            return eval_code
        report_command = [
            sys.executable,
            str(ROOT / "scripts/report_stage2_encoder_only_raw_cc64_huber_v0.py"),
            "--config", str(config_path), "--execute",
        ]
        report_code = run_process(report_command, log_handle)
        final = "completed" if report_code == 0 else "failed"
        update_status(
            config, status=final, current_stage=final,
            evaluation_completed=True, evaluation_return_code=0,
            reports_generated=report_code == 0, report_return_code=report_code,
            finished_at=now(),
        )
        return report_code


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--overnight", action="store_true")
    args = parser.parse_args()
    config = read_json(args.config)
    try:
        if args.preflight:
            print(json.dumps(preflight(config), indent=2, sort_keys=True))
            return 0
        if args.smoke:
            return smoke(config)
        if args.train:
            return train(config)
        return overnight(config, args.config.resolve())
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
