#!/usr/bin/env python3
"""Sequential canonical Encoder-only four-class loss comparison."""

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
from src.stage2_binary.full_training import training_window_order
from src.stage2_binary.training import build_binary_optimizer
from src.stage2_encoder_only.dataset import PEDAL_TOKEN_OFFSET
from src.stage2_encoder_only.train import atomic_torch_save, get_gpu_identity
from src.stage2_encoder_only.training import set_deterministic_seed
from src.stage2_four_class.loss_objectives import (
    HUBER_DELTA_CC64,
    HUBER_DELTA_NORMALIZED,
    NTL_WAS_LAMBDA,
    OBJECTIVES,
    FourClassObjectiveEncoderModel,
    RepresentativeHuberEncoderModel,
    continuous_cc64_to_classes,
    inverse_sqrt_class_weights,
    normalized_scalars_to_cc64,
)
from src.stage2_four_class.representation import (
    CLASS_BOUNDS,
    CLASS_NAMES,
    IGNORE_INDEX,
    REPRESENTATIVES,
    SharedFourClassWindowDataset,
    classify_cc64,
    make_four_class_loader,
)


STATUS_KEYS = {
    "weighted_ce": "weighted_ce",
    "ce_ntl_was": "ntl_was",
    "representative_huber": "huber",
}
REPORT_NAMES = {
    "weighted_ce": "WEIGHTED_CE_TRAINING_REPORT.md",
    "ce_ntl_was": "NTL_WAS_TRAINING_REPORT.md",
    "representative_huber": "HUBER_REGRESSION_TRAINING_REPORT.md",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def objective_directory(config: Mapping[str, Any], objective: str) -> Path:
    return Path(config["output_root"]) / config["objective_directories"][objective]


def update_status(config: Mapping[str, Any], **updates: Any) -> dict[str, Any]:
    path = Path(config["output_root"]) / "overnight_status.json"
    current = read_json(path) if path.is_file() else {}
    current.update(updates, last_update=now())
    atomic_json(path, current)
    return current


def update_objective_status(
    config: Mapping[str, Any], objective: str, status: str | None = None, **updates: Any
) -> dict[str, Any]:
    path = Path(config["output_root"]) / "overnight_status.json"
    current = read_json(path)
    key = STATUS_KEYS[objective]
    row = dict(current[key])
    if status is not None:
        row["status"] = status
    row.update(updates, last_update=now())
    current[key] = row
    current["last_update"] = now()
    atomic_json(path, current)
    return current


def update_run_status(output: Path, **updates: Any) -> dict[str, Any]:
    path = output / "run_status.json"
    current = read_json(path) if path.is_file() else {}
    current.update(updates, last_update=now(), asap_test_access_count=0)
    atomic_json(path, current)
    return current


def canonical_config(config: Mapping[str, Any]) -> dict[str, Any]:
    value = read_json(config["canonical_config"])
    if value["output_root"] == config["output_root"]:
        raise RuntimeError("phase-3 output root must not overlap the canonical CE baseline")
    return value


def training_class_statistics(base: Mapping[str, Any]) -> dict[str, Any]:
    """Read only the canonical train token cache and derive loss weights."""

    cache_root = Path(base["cache_root"])
    stats = read_json(cache_root / "cache_statistics.json")
    token_path = cache_root / stats["splits"]["train"]["tokens_file"]
    tokens = np.load(token_path, mmap_mode="r")
    counts = np.zeros(4, dtype=np.int64)
    for start in range(0, len(tokens), 250_000):
        raw = np.asarray(tokens[start : start + 250_000, 4:], dtype=np.int64)
        raw -= PEDAL_TOKEN_OFFSET
        classes = np.asarray(classify_cc64(raw), dtype=np.int64)
        counts += np.bincount(classes.reshape(-1), minlength=4)
    raw_weights, normalized = inverse_sqrt_class_weights(counts)
    total = int(counts.sum())
    if total != int(stats["splits"]["train"]["notes"]) * 4:
        raise RuntimeError("training class count does not match canonical cache notes")
    return {
        "source_split": "canonical training cache only",
        "validation_frequency_access": 0,
        "test_frequency_access": 0,
        "class_names": list(CLASS_NAMES),
        "counts": counts.tolist(),
        "total": total,
        "proportions": (counts / total).tolist(),
        "raw_inverse_sqrt_weights": raw_weights.tolist(),
        "normalized_mean_one_weights": normalized.tolist(),
        "mean_normalized_weight": float(normalized.mean()),
    }


def phase3_preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    if tuple(config["objectives"]) != OBJECTIVES:
        raise RuntimeError("objective order must be Weighted CE, NTL-WAS, Huber")
    if float(config["ntl_was_lambda"]) != NTL_WAS_LAMBDA:
        raise RuntimeError("NTL-WAS lambda changed")
    if float(config["huber_delta_cc64"]) != HUBER_DELTA_CC64 or not math.isclose(
        float(config["huber_delta_normalized"]), HUBER_DELTA_NORMALIZED,
        rel_tol=0.0, abs_tol=1e-15,
    ):
        raise RuntimeError("Huber delta changed")
    if int(config["asap_test_access"]) != 0:
        raise RuntimeError("ASAP test access must remain zero")
    base = canonical_config(config)
    audit = canonical_preflight(base)
    weights = training_class_statistics(base)
    baseline = Path(config["canonical_baseline_checkpoint"])
    if not baseline.is_file():
        raise FileNotFoundError(baseline)
    payload = torch.load(baseline, map_location="cpu", weights_only=False)
    if int(payload["epoch"]) != 2 or not math.isclose(
        float(payload["validation_ce"]), 0.8925421636047081,
        rel_tol=0.0, abs_tol=1e-12,
    ):
        raise RuntimeError("canonical Standard CE baseline metadata changed")
    del payload
    return {"canonical_audit": audit, "training_class_statistics": weights}


def build_model(
    base: Mapping[str, Any], objective: str, weights: Sequence[float]
) -> torch.nn.Module:
    common = {"torch_dtype": torch.float32, "attn_implementation": "eager"}
    if objective == "representative_huber":
        model = RepresentativeHuberEncoderModel.from_pretrained(
            base["checkpoint_path"], delta=HUBER_DELTA_NORMALIZED, **common
        )
    else:
        model = FourClassObjectiveEncoderModel.from_pretrained(
            base["checkpoint_path"],
            objective=objective,
            class_weights=weights if objective == "weighted_ce" else None,
            ntl_lambda=NTL_WAS_LAMBDA,
            **common,
        )
    if len(model.encoder.layers) != 10 or model.hidden_size != 768:
        raise RuntimeError("pretrained PT encoder architecture changed")
    heads = (
        model.regression_heads
        if objective == "representative_huber"
        else model.classification_heads
    )
    expected_features = 1 if objective == "representative_huber" else 4
    if len(heads) != 4 or any(head.in_features != 768 or head.out_features != expected_features for head in heads):
        raise RuntimeError("objective head architecture changed")
    return model


def move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def prediction_classes(objective: str, predictions: torch.Tensor) -> torch.Tensor:
    if objective == "representative_huber":
        return continuous_cc64_to_classes(normalized_scalars_to_cc64(predictions))
    return predictions.argmax(dim=-1)


def run_epoch(
    model: torch.nn.Module,
    loader: Any,
    *,
    objective: str,
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
    objective_numerator = objective_denominator = 0.0
    component_numerators: dict[str, float] = {}
    component_denominators: dict[str, float] = {}
    correct = exact = notes = target_count = 0
    optimizer_steps = amp_skips = 0
    continuous_min = math.inf
    continuous_max = -math.inf
    clipped_zero = clipped_full = 0
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
            raise RuntimeError("objective output omitted loss components")
        if not bool(torch.isfinite(output.loss)) or not bool(torch.isfinite(output.loss_numerator)):
            raise FloatingPointError("objective loss is non-finite")
        if training:
            if use_amp:
                scaler.scale(output.loss).backward()
                scaler.unscale_(optimizer)
            else:
                output.loss.backward()
            gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
            grad_norm = torch.nn.utils.get_total_norm(gradients, norm_type=2.0)
            grad_norm_value = float(grad_norm.detach().float().item())
            finite_gradient = math.isfinite(grad_norm_value)
            scale_before = float(scaler.get_scale()) if use_amp else None
            if finite_gradient:
                torch.nn.utils.clip_grads_with_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=float(max_grad_norm), total_norm=grad_norm,
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
                    raise FloatingPointError("non-finite gradient without AMP")
                optimizer.step()
                scale_after = None
                applied = True
            if applied:
                optimizer_steps += 1
            if step_callback is not None:
                step_callback(step, {
                    "loss": float(output.loss.detach().float().item()),
                    "gradient_norm": grad_norm_value,
                    "optimizer_step_applied": applied,
                    "optimizer_steps": optimizer_steps,
                    "amp_skipped_steps": amp_skips,
                    "scale_before": scale_before,
                    "scale_after": scale_after,
                })
        numerator = float(output.loss_numerator.detach().float().item())
        denominator = float(output.loss_denominator.detach().float().item())
        objective_numerator += numerator
        objective_denominator += denominator
        for name, value in output.component_numerators.items():
            component_numerators[name] = component_numerators.get(name, 0.0) + float(
                value.detach().float().item()
            )
            component_denominators[name] = component_denominators.get(name, 0.0) + float(
                output.component_denominators[name].detach().float().item()
            )
        targets = batch["pedal_targets"]
        valid = targets != IGNORE_INDEX
        predicted = prediction_classes(objective, output.predictions.detach())
        valid_notes = valid.all(dim=-1)
        correct += int(((predicted == targets) & valid).sum().item())
        exact += int(((((predicted == targets) | ~valid).all(dim=-1)) & valid_notes).sum().item())
        notes += int(valid_notes.sum().item())
        target_count += int(valid.sum().item())
        if objective == "representative_huber":
            values = output.predictions.detach()[valid].float()
            continuous_min = min(continuous_min, float(values.min().item()))
            continuous_max = max(continuous_max, float(values.max().item()))
            clipped_zero += int((values <= 0).sum().item())
            clipped_full += int((values >= 1).sum().item())
    if objective_denominator <= 0 or target_count <= 0 or notes <= 0:
        raise RuntimeError("empty epoch accounting")
    metrics = {
        "objective": objective_numerator / objective_denominator,
        "token_accuracy": correct / target_count,
        "exact_note_accuracy": exact / notes,
    }
    for name, value in component_numerators.items():
        metrics[name] = value / component_denominators[name]
    if objective == "representative_huber":
        metrics.update(
            continuous_prediction_min=continuous_min,
            continuous_prediction_max=continuous_max,
            clip_at_0_proportion=clipped_zero / target_count,
            clip_at_127_proportion=clipped_full / target_count,
        )
    if not all(math.isfinite(value) for value in metrics.values()):
        raise FloatingPointError("non-finite epoch metric")
    return metrics, {"optimizer_steps": optimizer_steps, "amp_skipped_steps": amp_skips}


def checkpoint_payload(
    model: torch.nn.Module, run_config: Mapping[str, Any], row: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "model_state": model.state_dict(),
        "configuration": dict(run_config),
        "objective": run_config["objective"],
        "epoch": int(row["epoch"]),
        "validation_objective": float(row["validation_objective"]),
        "checkpoint_selection": "minimum validation training objective",
    }


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


def run_config_for(
    config: Mapping[str, Any], base: Mapping[str, Any], objective: str,
    audit: Mapping[str, Any], weights: Mapping[str, Any], gpu: Mapping[str, Any],
    model: torch.nn.Module,
) -> dict[str, Any]:
    regression = objective == "representative_huber"
    return {
        **base,
        "experiment_id": config["experiment_id"],
        "output_root": str(objective_directory(config, objective)),
        "objective": objective,
        "architecture_name": "encoder_only",
        "architecture": (
            "PT pretrained 10-layer encoder + four independent Linear(768,1) heads"
            if regression else
            "PT pretrained 10-layer encoder + four independent Linear(768,4) heads"
        ),
        "output_shape": "B,N,4" if regression else "B,N,4,4",
        "fresh_initialization_seed": int(base["seed"]),
        "checkpoint_selection": "minimum_validation_training_objective",
        "class_names": list(CLASS_NAMES),
        "class_boundaries": [list(value) for value in CLASS_BOUNDS],
        "representatives": list(REPRESENTATIVES),
        "ntl_was_lambda": NTL_WAS_LAMBDA if objective == "ce_ntl_was" else None,
        "ntl_was_formula": (
            "sum_c softmax(logits)_c * abs(V_c-V_y)/127" if objective == "ce_ntl_was" else None
        ),
        "total_loss_formula": "CE - 0.3 * NTL_WAS" if objective == "ce_ntl_was" else None,
        "training_class_statistics": weights if objective == "weighted_ce" else None,
        "class_weights": weights["normalized_mean_one_weights"] if objective == "weighted_ce" else None,
        "regression_target": "canonical class -> [0,51,79,127] / 127" if regression else None,
        "huber_delta_cc64": HUBER_DELTA_CC64 if regression else None,
        "huber_delta_normalized": HUBER_DELTA_NORMALIZED if regression else None,
        "training_output_activation": "none (unconstrained scalar)" if regression else None,
        "inference_conversion": config["regression_midi_integer_conversion"] if regression else "argmax",
        "micro_batch_size": int(base["micro_batch_size"]["encoder_only"]),
        "gradient_accumulation_steps": int(base["gradient_accumulation_steps"]["encoder_only"]),
        "initial_encoder_parameter_sha256": parameter_hash(model.encoder.parameters()),
        "initial_head_parameter_sha256": parameter_hash(model.prediction_head_parameters()),
        "canonical_cache_audit": audit,
        "gpu": dict(gpu),
        "asap_test_access": 0,
        "standard_ce_retrained": False,
    }


def train_objective(config: Mapping[str, Any], objective: str) -> int:
    output = objective_directory(config, objective)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite objective output: {output}")
    output.mkdir(parents=True)
    update_run_status(
        output, objective=objective, status="running", current_stage="initializing",
        started_at=now(), report_generated=False, evaluation_completed=False,
    )
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
        phase = phase3_preflight(config)
        audit = phase["canonical_audit"]
        weights = phase["training_class_statistics"]
        log(f"CACHE_LOAD_PASS cache_id={audit['cache_id']} train=1170+892 validation=71 test_access=0")
        set_deterministic_seed(int(base["seed"]))
        model = build_model(base, objective, weights["normalized_mean_one_weights"])
        encoder_hash = parameter_hash(model.encoder.parameters())
        if encoder_hash != base["expected_initial_encoder_sha256"]:
            raise RuntimeError("pretrained encoder parameter hash mismatch")
        run_config = run_config_for(config, base, objective, audit, weights, gpu, model)
        atomic_json(output / "config.json", run_config)
        if objective == "weighted_ce":
            atomic_json(output / "class_weights.json", weights)
        log(
            f"MODEL_INITIALIZED objective={objective} seed=42 encoder_sha256={encoder_hash} "
            f"head_sha256={run_config['initial_head_parameter_sha256']}"
        )
        train_dataset = SharedFourClassWindowDataset(base["cache_root"], "train")
        validation_dataset = SharedFourClassWindowDataset(base["cache_root"], "validation")
        model.to(device)
        optimizer = build_binary_optimizer(
            model, encoder_lr=float(base["encoder_lr"]), head_lr=float(base["head_lr"]),
            weight_decay=float(base["weight_decay"]),
        )
        scaler = torch.amp.GradScaler(
            "cuda", enabled=bool(base["amp_enabled"]), init_scale=float(base["amp_init_scale"])
        )
        validation_loader = make_four_class_loader(
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
            loader = make_four_class_loader(
                train_dataset, batch_size=int(base["micro_batch_size"]["encoder_only"]),
                pin_memory=bool(base["pin_memory"]), order=order,
            )

            def callback(step: int, details: Mapping[str, Any]) -> None:
                nonlocal first_step_logged
                if not first_step_logged and details["optimizer_step_applied"]:
                    loss = float(details["loss"])
                    if not math.isfinite(loss):
                        raise FloatingPointError("initial loss is non-finite")
                    first_step_logged = True
                    update_objective_status(
                        config, objective, "running", first_optimizer_step=True,
                        first_step_epoch=epoch, first_step=step, first_step_loss=loss,
                        first_step_gradient_norm=float(details["gradient_norm"]),
                        fresh_seed=42, cache_id=audit["cache_id"], gpu=gpu,
                    )
                    update_run_status(
                        output, status="running", current_stage="first_optimizer_step_pass",
                        first_optimizer_step=True, first_step_epoch=epoch, first_step=step,
                        first_step_loss=loss, first_step_gradient_norm=float(details["gradient_norm"]),
                        fresh_seed=42, cache_id=audit["cache_id"], gpu=gpu,
                    )
                    log(
                        f"FIRST_OPTIMIZER_STEP_PASS objective={objective} epoch={epoch} "
                        f"step={step} loss={loss:.9f} grad_norm={float(details['gradient_norm']):.6f}"
                    )
                elif step % int(base["progress_interval"]) == 0:
                    update_objective_status(
                        config, objective, "running", current_epoch=epoch, current_step=step,
                    )
                    log(f"TRAIN_PROGRESS objective={objective} epoch={epoch} step={step} loss={float(details['loss']):.9f}")

            train_metrics, checks = run_epoch(
                model, loader, objective=objective, device=device,
                amp_enabled=bool(base["amp_enabled"]), optimizer=optimizer, scaler=scaler,
                max_grad_norm=float(base["max_grad_norm"]), step_callback=callback,
            )
            if not first_step_logged:
                raise RuntimeError("finite first optimizer step was not verified")
            validation_metrics, _ = run_epoch(
                model, validation_loader, objective=objective, device=device,
                amp_enabled=bool(base["amp_enabled"]),
            )
            row: dict[str, Any] = {
                "epoch": epoch,
                "train_objective": train_metrics["objective"],
                "validation_objective": validation_metrics["objective"],
                "train_token_accuracy": train_metrics["token_accuracy"],
                "validation_token_accuracy": validation_metrics["token_accuracy"],
                "train_exact_note_accuracy": train_metrics["exact_note_accuracy"],
                "validation_exact_note_accuracy": validation_metrics["exact_note_accuracy"],
                "optimizer_steps": checks["optimizer_steps"],
                "amp_skipped_steps": checks["amp_skipped_steps"],
                "epoch_seconds": time.perf_counter() - started,
            }
            for name, value in train_metrics.items():
                if name not in {"objective", "token_accuracy", "exact_note_accuracy"}:
                    row[f"train_{name}"] = value
            for name, value in validation_metrics.items():
                if name not in {"objective", "token_accuracy", "exact_note_accuracy"}:
                    row[f"validation_{name}"] = value
            rows.append(row)
            write_metrics(output / "metrics.csv", rows)
            candidate = float(row["validation_objective"])
            if candidate < best:
                best, best_epoch = candidate, epoch
                atomic_torch_save(checkpoint_payload(model, run_config, row), output / "best.pt")
            if candidate < early_best - float(base["early_stopping_min_delta"]):
                early_best, stale = candidate, 0
            else:
                stale += 1
            atomic_torch_save(checkpoint_payload(model, run_config, row), output / "last.pt")
            update_objective_status(
                config, objective, "running", current_stage="epoch_completed",
                current_epoch=epoch, best_epoch=best_epoch, best_validation_objective=best,
                validation_objective=candidate,
            )
            update_run_status(
                output, status="running", current_stage="epoch_completed",
                current_epoch=epoch, best_epoch=best_epoch,
                best_validation_objective=best, validation_objective=candidate,
            )
            log(
                f"EPOCH_COMPLETE objective={objective} epoch={epoch} "
                f"train_objective={float(row['train_objective']):.9f} "
                f"validation_objective={candidate:.9f} best_epoch={best_epoch}"
            )
            if stale >= int(base["early_stopping_patience"]):
                log(f"EARLY_STOP objective={objective} epoch={epoch}")
                break
        if not all((output / name).is_file() for name in ("best.pt", "last.pt", "config.json", "metrics.csv", "train.log")):
            raise RuntimeError("required training artifacts are missing")
        update_objective_status(
            config, objective, "trained", best_epoch=best_epoch,
            best_validation_objective=best, training_completed=True,
        )
        update_run_status(
            output, status="trained", current_stage="training_completed",
            best_epoch=best_epoch, best_validation_objective=best,
            training_completed=True, finished_at=now(),
        )
        log(f"TRAINING_COMPLETE objective={objective} best_epoch={best_epoch} best_validation_objective={best:.9f}")
        return 0
    except Exception as error:
        log(f"TRAINING_FAILED {type(error).__name__}: {error}")
        traceback.print_exc(file=log_handle)
        update_objective_status(
            config, objective, "failed", current_stage="training_failed",
            error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc(),
        )
        update_run_status(
            output, status="failed", current_stage="training_failed",
            error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc(),
            finished_at=now(),
        )
        return 20 if any(term in str(error).lower() for term in ("cache", "asap", "gpu", "cuda", "non-finite model")) else 1
    finally:
        log_handle.close()


def smoke_all(config: Mapping[str, Any]) -> int:
    output_root = Path(config["output_root"])
    smoke_root = output_root / "preflight_smoke"
    if smoke_root.exists():
        raise FileExistsError(f"refusing to overwrite smoke output: {smoke_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    phase = phase3_preflight(config)
    base = canonical_config(config)
    weights = phase["training_class_statistics"]
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("smoke requires exactly one visible CUDA GPU")
    device = torch.device("cuda:0")
    if get_gpu_identity()["uuid"] != base["expected_gpu_uuid"]:
        raise RuntimeError("smoke GPU UUID mismatch")
    train_dataset = SharedFourClassWindowDataset(base["cache_root"], "train")
    validation_dataset = SharedFourClassWindowDataset(base["cache_root"], "validation")
    smoke_root.mkdir(parents=True)
    results: dict[str, Any] = {}
    for objective in OBJECTIVES:
        set_deterministic_seed(int(base["seed"]))
        model = build_model(base, objective, weights["normalized_mean_one_weights"]).to(device)
        optimizer = build_binary_optimizer(
            model, encoder_lr=float(base["encoder_lr"]), head_lr=float(base["head_lr"]),
            weight_decay=float(base["weight_decay"]),
        )
        scaler = torch.amp.GradScaler(
            "cuda", enabled=bool(base["amp_enabled"]), init_scale=float(base["amp_init_scale"])
        )
        before = parameter_hash(model.prediction_head_parameters())
        train_loader = make_four_class_loader(
            train_dataset, batch_size=1, pin_memory=bool(base["pin_memory"]), order=[0],
        )
        train_metrics, checks = run_epoch(
            model, train_loader, objective=objective, device=device,
            amp_enabled=bool(base["amp_enabled"]), optimizer=optimizer, scaler=scaler,
            max_grad_norm=float(base["max_grad_norm"]), max_batches=1,
        )
        after = parameter_hash(model.prediction_head_parameters())
        validation_loader = make_four_class_loader(
            validation_dataset, batch_size=1, pin_memory=bool(base["pin_memory"]), order=[0],
        )
        validation_metrics, _ = run_epoch(
            model, validation_loader, objective=objective, device=device,
            amp_enabled=bool(base["amp_enabled"]), max_batches=1,
        )
        checkpoint = smoke_root / f"{objective}.pt"
        atomic_torch_save({"model_state": model.state_dict(), "objective": objective}, checkpoint)
        loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
        incompatible = model.load_state_dict(loaded["model_state"], strict=True)
        passed = (
            checks["optimizer_steps"] == 1 and before != after
            and math.isfinite(train_metrics["objective"])
            and math.isfinite(validation_metrics["objective"])
            and not incompatible.missing_keys and not incompatible.unexpected_keys
        )
        if not passed:
            raise RuntimeError(f"smoke failed: {objective}")
        results[objective] = {
            "status": "passed", "finite_train_loss": train_metrics["objective"],
            "finite_validation_loss": validation_metrics["objective"],
            "finite_gradients": True, "optimizer_step": True,
            "head_parameters_updated": True, "checkpoint_load": True,
            "checkpoint": str(checkpoint),
        }
        del loaded, model, optimizer, scaler
        gc.collect()
        torch.cuda.empty_cache()
    atomic_json(smoke_root / "smoke_results.json", {
        "status": "passed", "objectives": results, "asap_test_access_count": 0,
        "canonical_audit": phase["canonical_audit"],
        "training_class_statistics": weights,
    })
    return 0


def individual_report(config: Mapping[str, Any], objective: str, status: str) -> None:
    output = objective_directory(config, objective)
    cfg = read_json(output / "config.json") if (output / "config.json").is_file() else {}
    rows = list(csv.DictReader((output / "metrics.csv").open())) if (output / "metrics.csv").is_file() else []
    state = read_json(Path(config["output_root"]) / "overnight_status.json")[STATUS_KEYS[objective]]
    evaluation = read_json(output / "validation" / "evaluation.json") if (output / "validation" / "evaluation.json").is_file() else None
    definitions = {
        "weighted_ce": "Weighted cross entropy with train-cache-only mean-one inverse-sqrt frequency weights.",
        "ce_ntl_was": "CE - 0.3 × Σ_c p_c |V_c-V_y|/127, V=[0,51,79,127].",
        "representative_huber": "Unconstrained four-scalar representative regression with Huber δ=14/127.",
    }
    lines = [
        f"# {objective} Training Report", "", f"- Final status: {status}",
        f"- Objective: {definitions[objective]}",
        f"- Architecture: {cfg.get('architecture', 'initialization failed')}",
        "- Dataset/cache: canonical MAESTRO-clean 1,170 + ASAP train 892; ASAP validation 71",
        f"- Seed: {cfg.get('fresh_initialization_seed', 42)}",
        f"- Optimizer/LR: AdamW; encoder {cfg.get('encoder_lr', 1e-5)}; head {cfg.get('head_lr', 1e-4)}",
        f"- Best epoch: {state.get('best_epoch')}",
        f"- Best validation objective: {state.get('best_validation_objective')}",
        f"- Checkpoints: `{output / 'best.pt'}`, `{output / 'last.pt'}`",
        "- ASAP test access count: 0", "", "## Epoch table", "",
    ]
    if rows:
        keys = ["epoch", "train_objective", "validation_objective", "validation_ce", "validation_ntl_was", "validation_huber", "validation_representative_mae_cc64"]
        keys = [key for key in keys if key in rows[0]]
        lines += ["| " + " | ".join(keys) + " |", "|" + "|".join("---:" for _ in keys) + "|"]
        for row in rows:
            lines.append("| " + " | ".join(row[key] for key in keys) + " |")
    if evaluation:
        cls = evaluation["classification"]
        trans = evaluation["transition"]["pooled"]
        patt = evaluation["patterns"]
        lines += [
            "", "## Canonical ASAP-validation evaluation", "",
            f"- 4C accuracy: {cls['token_accuracy']:.6f}; Macro F1: {cls['macro_f1']:.6f}",
            f"- Transition P/R/F1: {trans['precision']:.6f} / {trans['recall']:.6f} / {trans['f1']:.6f}",
            f"- JS divergence / Intersection: {patt['js_divergence_base2']:.6f} / {patt['intersection']:.6f}",
            f"- Prediction distribution: `{cls['prediction_class_distribution']}`",
            f"- Class P/R/F1: `{cls['class_metrics']}`",
            f"- Confusion matrix: `{cls['confusion_matrix']}`",
            f"- Candidate/reference transitions: {trans['candidate']} / {trans['reference']}; TP/FP/FN: {trans['tp']}/{trans['fp']}/{trans['fn']}",
            f"- Top 256-pattern probabilities: `{patt['top_candidate_patterns']}`",
        ]
    temporary = output / f".{REPORT_NAMES[objective]}.{os.getpid()}.tmp"
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, output / REPORT_NAMES[objective])
    update_run_status(
        output, status=status, report_generated=True,
        evaluation_completed=evaluation is not None,
    )


def aggregate_report(config: Mapping[str, Any]) -> None:
    baseline = read_json(config["canonical_baseline_metrics"])
    names = {
        "weighted_ce": "Weighted CE",
        "ce_ntl_was": "CE + NTL-WAS λ=0.3",
        "representative_huber": "Representative Huber δ=14",
    }
    values: dict[str, Any] = {"standard_ce": baseline}
    for objective in OBJECTIVES:
        path = objective_directory(config, objective) / "validation" / "evaluation.json"
        values[objective] = read_json(path) if path.is_file() else None
    lines = [
        "# Encoder-only 4-Class Loss Phase 3 Comparison", "",
        "ASAP validation only; test access count 0; frozen canonical evaluator; Repedal excluded.", "",
        "| Objective | 4C Acc ↑ | Macro F1 ↑ | Transition P ↑ | Transition R ↑ | Transition F1 ↑ | JS ↓ | Intersection ↑ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = [("standard_ce", "Standard CE")] + [(key, names[key]) for key in OBJECTIVES]
    for key, label in labels:
        value = values[key]
        if value is None:
            lines.append(f"| {label} | ... | ... | ... | ... | ... | ... | ... |")
            continue
        cls, trans, patt = value["classification"], value["transition"]["pooled"], value["patterns"]
        lines.append(
            f"| {label} | {cls['token_accuracy']:.6f} | {cls['macro_f1']:.6f} | {trans['precision']:.6f} | {trans['recall']:.6f} | {trans['f1']:.6f} | {patt['js_divergence_base2']:.6f} | {patt['intersection']:.6f} |"
        )
    status = read_json(Path(config["output_root"]) / "overnight_status.json")
    lines += ["", "| Objective | LOW Recall | HALF Recall | Candidate Transitions | Best Epoch | Status |", "|---|---:|---:|---:|---:|---|"]
    for key, label in labels:
        value = values[key]
        state = {"best_epoch": 2, "status": "completed"} if key == "standard_ce" else status[STATUS_KEYS[key]]
        if value is None:
            lines.append(f"| {label} | ... | ... | ... | {state.get('best_epoch', '...')} | {state.get('status')} |")
        else:
            metrics = value["classification"]["class_metrics"]
            lines.append(f"| {label} | {metrics[1]['recall']:.6f} | {metrics[2]['recall']:.6f} | {value['transition']['pooled']['candidate']} | {state.get('best_epoch')} | {state.get('status')} |")
    lines += [
        "", "## Interpretation scope", "",
        "This report is limited to intermediate-depth performance, class-imbalance effects, numerical-distance-aware loss effects, regression-formulation effects, spurious-transition changes, and global pattern-distribution changes. It starts no follow-up experiment.",
    ]
    path = Path(config["output_root"]) / "LOSS_PHASE3_COMPARISON_REPORT.md"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, path)


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
    output_root = Path(config["output_root"])
    smoke = output_root / "preflight_smoke" / "smoke_results.json"
    if not smoke.is_file() or read_json(smoke).get("status") != "passed":
        raise RuntimeError("full pipeline is blocked until all smoke tests pass")
    if any(objective_directory(config, name).exists() for name in OBJECTIVES):
        raise FileExistsError("refusing to overwrite an existing phase-3 objective directory")
    status = {
        "overall_status": "running", "current_run": "weighted_ce", "current_stage": "starting",
        "weighted_ce": {"status": "pending", "best_epoch": None, "best_validation_objective": None, "report_generated": False, "evaluation_completed": False},
        "ntl_was": {"status": "pending", "best_epoch": None, "best_validation_objective": None, "report_generated": False, "evaluation_completed": False},
        "huber": {"status": "pending", "best_epoch": None, "best_validation_objective": None, "report_generated": False, "evaluation_completed": False},
        "aggregate_report_generated": False, "test_access_count": 0,
        "sequential": True, "single_gpu": True, "started_at": now(), "last_update": now(),
    }
    atomic_json(output_root / "overnight_status.json", status)
    log_path = output_root / "overnight.log"
    critical_failure = False
    with log_path.open("w", encoding="utf-8") as log_handle:
        for objective in OBJECTIVES:
            update_status(config, current_run=objective, current_stage="training")
            update_objective_status(config, objective, "running", started_at=now(), report_generated=False, evaluation_completed=False)
            print(f"{now()} OVERNIGHT_START objective={objective}", file=log_handle, flush=True)
            train_command = [sys.executable, str(Path(__file__).resolve()), "--config", str(config_path), "--train-objective", objective]
            code = run_process(train_command, log_handle)
            if code != 0:
                update_objective_status(config, objective, "failed", return_code=code, finished_at=now())
                individual_report(config, objective, "failed")
                update_objective_status(config, objective, report_generated=True)
                if code == 20:
                    critical_failure = True
                    break
                continue
            update_status(config, current_stage="validation_evaluation")
            eval_command = [sys.executable, str(ROOT / "scripts/evaluate_stage2_encoder_only_loss_phase3_v0.py"), "--config", str(config_path), "--objective", objective, "--execute"]
            eval_code = run_process(eval_command, log_handle)
            final_status = "completed" if eval_code == 0 else "failed"
            update_objective_status(
                config, objective, final_status, evaluation_completed=eval_code == 0,
                evaluation_return_code=eval_code, finished_at=now(),
            )
            individual_report(config, objective, final_status)
            update_objective_status(config, objective, report_generated=True)
        aggregate_report(config)
        update_status(
            config, overall_status="failed_shared_critical" if critical_failure else "completed",
            current_run=None, current_stage="completed", aggregate_report_generated=True,
            test_access_count=0, finished_at=now(),
        )
    return 20 if critical_failure else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--smoke-all", action="store_true")
    mode.add_argument("--train-objective", choices=OBJECTIVES)
    mode.add_argument("--overnight", action="store_true")
    args = parser.parse_args()
    config = read_json(args.config)
    try:
        if args.preflight:
            print(json.dumps(phase3_preflight(config), indent=2, sort_keys=True))
            return 0
        if args.smoke_all:
            return smoke_all(config)
        if args.train_objective:
            return train_objective(config, args.train_objective)
        return overnight(config, args.config.resolve())
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
