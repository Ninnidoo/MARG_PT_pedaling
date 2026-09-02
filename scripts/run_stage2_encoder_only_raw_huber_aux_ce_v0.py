#!/usr/bin/env python3
"""Train/orchestrate Raw CC64 Huber plus training-only standard CE."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_stage2_4class_architecture_v0 import atomic_json, parameter_hash, read_json
from scripts.run_stage2_encoder_only_raw_cc64_huber_v0 import (
    canonical_config,
    make_raw_cc64_loader,
    now,
    preflight as raw_preflight,
    scan_training_raw_targets,
    update_status,
)
from src.stage2_binary.full_training import SharedBinaryWindowDataset, training_window_order
from src.stage2_binary.training import build_binary_optimizer
from src.stage2_encoder_only.train import atomic_torch_save, get_gpu_identity
from src.stage2_encoder_only.training import set_deterministic_seed
from src.stage2_four_class.loss_objectives import (
    HUBER_DELTA_CC64,
    HUBER_DELTA_NORMALIZED,
    continuous_cc64_to_classes,
)
from src.stage2_four_class.raw_huber_aux_ce import (
    LAMBDA_CE,
    RawHuberAuxCEEncoderModel,
)
from src.stage2_four_class.representation import IGNORE_INDEX, classify_cc64


RAW_BASELINE = {
    "accuracy": 0.49231951127170076,
    "macro_f1": 0.40341265695775064,
    "transition_precision": 0.497066,
    "transition_recall": 0.512063,
    "transition_f1": 0.504453,
    "candidate_transitions": 35611,
    "low_recall": 0.21127851248913707,
    "half_recall": 0.27161713149495054,
    "js_divergence": 0.038859,
    "intersection": 0.825369,
    "best_epoch": 6,
}

MAX_CONSECUTIVE_AMP_OVERFLOWS = 3


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _nonfinite_model_parameter_names(model: torch.nn.Module) -> list[str]:
    return [name for name, value in model.named_parameters() if not bool(torch.isfinite(value).all())]


def _nonfinite_gradient_diagnostics(
    model: RawHuberAuxCEEncoderModel,
) -> dict[str, Any]:
    groups = {id(parameter): "encoder" for parameter in model.encoder.parameters()}
    groups.update({id(parameter): "regression_heads" for parameter in model.regression_heads.parameters()})
    groups.update({id(parameter): "auxiliary_heads" for parameter in model.classification_heads.parameters()})
    names: list[str] = []
    affected: set[str] = set()
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
            names.append(name)
            affected.add(groups[id(parameter)])
    return {
        "offending_parameter_count": len(names),
        "offending_parameter_names": names,
        "affected_groups": sorted(affected),
    }


def preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    if config.get("objective") != "raw_huber_aux_ce":
        raise RuntimeError("objective must be raw_huber_aux_ce")
    if float(config["lambda_ce"]) != LAMBDA_CE:
        raise RuntimeError("lambda_ce must be exactly 0.1")
    if float(config["huber_delta_cc64"]) != HUBER_DELTA_CC64 or not math.isclose(
        float(config["huber_delta_normalized"]), HUBER_DELTA_NORMALIZED,
        rel_tol=0.0, abs_tol=1e-15,
    ):
        raise RuntimeError("canonical Huber delta changed")
    if int(config["asap_test_access"]) != 0 or bool(config["repedal_evaluated"]):
        raise RuntimeError("ASAP test and Repedal must remain unused")
    forbidden = (
        "raw_huber_checkpoint_used_for_initialization",
        "representative_checkpoint_used_for_initialization",
        "classification_checkpoint_used_for_initialization",
    )
    if any(bool(config[key]) for key in forbidden):
        raise RuntimeError("candidate checkpoint warm-start is forbidden")
    previous = Path(config.get("failed_hybrid_v0_root", ""))
    if previous and previous == Path(config["output_root"]):
        raise RuntimeError("clean v1 output overlaps failed v0 provenance")
    raw_config = dict(config)
    raw_config["objective"] = "raw_cc64_huber"
    raw_config["representative_checkpoint_used_for_initialization"] = False
    phase = raw_preflight(raw_config)
    status = read_json(Path(config["raw_huber_root"]) / "run_status.json")
    evaluation = read_json(config["raw_huber_evaluation"])
    classification = evaluation["classification"]
    transition = evaluation["transition"]["pooled"]
    patterns = evaluation["patterns"]
    if status.get("status") != "completed" or int(status["best_epoch"]) != 6:
        raise RuntimeError("completed Raw Huber epoch-6 baseline is unavailable")
    exact = {
        "accuracy": classification["token_accuracy"],
        "macro_f1": classification["macro_f1"],
        "candidate_transitions": transition["candidate"],
    }
    if not math.isclose(exact["accuracy"], RAW_BASELINE["accuracy"], abs_tol=1e-12):
        raise RuntimeError("Raw Huber baseline accuracy changed")
    if not math.isclose(exact["macro_f1"], RAW_BASELINE["macro_f1"], abs_tol=1e-12):
        raise RuntimeError("Raw Huber baseline Macro F1 changed")
    if int(exact["candidate_transitions"]) != RAW_BASELINE["candidate_transitions"]:
        raise RuntimeError("Raw Huber baseline transition count changed")
    phase["raw_baseline"] = {
        **RAW_BASELINE,
        "root": config["raw_huber_root"],
        "checkpoint_read_for_initialization": False,
        "successful_validation_pairs": evaluation["successful_pairs"],
        "aligned_notes": evaluation["aligned_notes"],
        "transition_exact": transition,
        "patterns_exact": patterns,
    }
    return phase


def build_fresh_model(base: Mapping[str, Any]) -> RawHuberAuxCEEncoderModel:
    model = RawHuberAuxCEEncoderModel.from_pretrained(
        base["checkpoint_path"], delta=HUBER_DELTA_NORMALIZED,
        lambda_ce=LAMBDA_CE, torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    if len(model.encoder.layers) != 10 or model.hidden_size != 768:
        raise RuntimeError("pretrained encoder architecture changed")
    if len(model.regression_heads) != 4 or any(h.out_features != 1 for h in model.regression_heads):
        raise RuntimeError("four scalar-head architecture changed")
    if len(model.classification_heads) != 4 or any(h.out_features != 4 for h in model.classification_heads):
        raise RuntimeError("four auxiliary classification heads are invalid")
    return model


def move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def _class_metrics(confusion: np.ndarray) -> tuple[float, float, list[float]]:
    total = int(confusion.sum())
    accuracy = float(np.trace(confusion) / total)
    f1s: list[float] = []
    for index in range(4):
        tp = float(confusion[index, index])
        fp = float(confusion[:, index].sum() - tp)
        fn = float(confusion[index, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return accuracy, float(np.mean(f1s)), f1s


def run_epoch(
    model: RawHuberAuxCEEncoderModel, loader: Any, *, device: torch.device,
    amp_enabled: bool, optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None, max_grad_norm: float = 1.0,
    step_callback: Any = None, max_batches: int | None = None,
    max_consecutive_amp_overflows: int = MAX_CONSECUTIVE_AMP_OVERFLOWS,
    initial_consecutive_amp_overflows: int = 0,
    inspect_finite_gradient_groups: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    training = optimizer is not None
    model.train(training)
    sums = {key: 0.0 for key in ("huber", "ce", "total", "raw_cc64_absolute_error", "raw_cc64_squared_error")}
    count = 0.0
    confusion = np.zeros((4, 4), dtype=np.int64)
    optimizer_steps = optimizer_attempts = amp_skips = amp_overflow_events = 0
    consecutive_overflows = int(initial_consecutive_amp_overflows)
    maximum_consecutive_overflows = consecutive_overflows
    if training:
        corrupt = _nonfinite_model_parameter_names(model)
        if corrupt:
            raise FloatingPointError("non-finite model parameter(s): " + ", ".join(corrupt[:8]))
    for step, cpu_batch in enumerate(loader, 1):
        if max_batches is not None and step > max_batches:
            break
        batch = move_batch(cpu_batch, device)
        if training:
            assert scaler is not None
            optimizer.zero_grad(set_to_none=True)
        use_autocast = bool(amp_enabled and device.type == "cuda")
        with torch.set_grad_enabled(training), torch.amp.autocast(
            device_type=device.type, dtype=torch.float16 if use_autocast else torch.bfloat16,
            enabled=use_autocast,
        ):
            output = model(
                input_ids=batch["input_ids"], token_attention_mask=batch["token_attention_mask"],
                note_mask=batch["note_mask"], pedal_targets=batch["pedal_targets"],
            )
        if output.loss is None or output.loss_denominator is None:
            raise FloatingPointError("hybrid objective is missing")
        for component in ("huber", "ce", "total"):
            value = output.component_losses.get(component)
            if value is None or not bool(torch.isfinite(value)):
                raise FloatingPointError(f"hybrid {component} objective is non-finite")
        grad_norm_value = 0.0
        applied = False
        gradient_groups: dict[str, bool] = {}
        overflow_diagnostics = {
            "offending_parameter_count": 0,
            "offending_parameter_names": [],
            "affected_groups": [],
        }
        overflow_event = False
        scale_before = scale_after = None
        if training:
            optimizer_attempts += 1
            if amp_enabled:
                scaler.scale(output.loss).backward(); scaler.unscale_(optimizer)
            else:
                output.loss.backward()
            parameters = [p for p in model.parameters() if p.requires_grad]
            gradients = [p.grad for p in parameters if p.grad is not None]
            grad_norm = torch.nn.utils.get_total_norm(gradients, norm_type=2.0)
            grad_norm_value = float(grad_norm.detach().float().item())
            gradient_finite = math.isfinite(grad_norm_value)
            if gradient_finite:
                torch.nn.utils.clip_grads_with_norm_(parameters, max_norm=max_grad_norm, total_norm=grad_norm)
                if inspect_finite_gradient_groups:
                    for name, group in (
                        ("encoder", model.encoder.parameters()),
                        ("regression", model.regression_heads.parameters()),
                        ("classification", model.classification_heads.parameters()),
                    ):
                        group_gradients = [p.grad for p in group if p.grad is not None]
                        gradient_groups[name] = bool(group_gradients) and all(
                            bool(torch.isfinite(g).all()) for g in group_gradients
                        ) and any(bool(torch.count_nonzero(g)) for g in group_gradients)
            else:
                overflow_diagnostics = _nonfinite_gradient_diagnostics(model)
            if amp_enabled:
                scale_before = float(scaler.get_scale())
                if not gradient_finite and not overflow_diagnostics["offending_parameter_count"]:
                    raise FloatingPointError("non-finite aggregate gradient norm without a non-finite gradient tensor")
                scaler.step(optimizer); scaler.update()
                scale_after = float(scaler.get_scale())
                overflow_event = not gradient_finite
                applied = gradient_finite and scale_after >= scale_before
                if overflow_event and scale_after < scale_before:
                    amp_skips += 1
            else:
                if not gradient_finite:
                    raise FloatingPointError("non-finite gradient without AMP")
                optimizer.step(); applied = True
            if applied:
                optimizer_steps += 1
                consecutive_overflows = 0
            elif overflow_event:
                amp_overflow_events += 1
                consecutive_overflows += 1
                maximum_consecutive_overflows = max(maximum_consecutive_overflows, consecutive_overflows)
                corrupt = _nonfinite_model_parameter_names(model)
                if corrupt:
                    raise FloatingPointError("non-finite model parameter(s) after AMP overflow: " + ", ".join(corrupt[:8]))
        denominator = float(output.loss_denominator.detach().float().item())
        for key in sums:
            sums[key] += float(output.component_numerators[key].detach().float().item())
        count += denominator
        valid = batch["pedal_targets"] != IGNORE_INDEX
        target_classes = classify_cc64(batch["pedal_targets"], allow_ignore=True)[valid]
        assert output.auxiliary_logits is not None
        predicted_classes = output.auxiliary_logits.detach().argmax(dim=-1)[valid]
        encoded = (target_classes * 4 + predicted_classes).detach().cpu().numpy()
        confusion += np.bincount(encoded, minlength=16).reshape(4, 4)
        if step_callback is not None:
            raw = batch["pedal_targets"][valid]
            unique = torch.unique(raw)
            nonrep = unique[(unique != 0) & (unique != 51) & (unique != 79) & (unique != 127)]
            step_callback(step, {
                "huber": float(output.component_losses["huber"].detach().float().item()),
                "ce": float(output.component_losses["ce"].detach().float().item()),
                "total": float(output.component_losses["total"].detach().float().item()),
                "gradient_norm": grad_norm_value, "optimizer_step_applied": applied,
                "gradient_groups_finite_nonzero": gradient_groups,
                "optimizer_steps": optimizer_steps,
                "optimizer_attempts": optimizer_attempts,
                "amp_overflow_event": overflow_event,
                "amp_overflow_events": amp_overflow_events,
                "consecutive_amp_overflows": consecutive_overflows,
                "max_consecutive_amp_overflows": maximum_consecutive_overflows,
                "scale_before": scale_before, "scale_after": scale_after,
                "scaler_step_update_executed": bool(training and amp_enabled),
                "optimizer_step_skipped": bool(overflow_event and scale_after is not None and scale_before is not None and scale_after < scale_before),
                **overflow_diagnostics,
                "raw_unique_count": int(unique.numel()), "non_representative_unique_count": int(nonrep.numel()),
                "raw_unique_examples": unique[:24].detach().cpu().tolist(),
                "canonical_class_examples": torch.unique(target_classes).detach().cpu().tolist(),
            })
        if overflow_event and consecutive_overflows >= int(max_consecutive_amp_overflows):
            raise FloatingPointError(
                f"persistent AMP gradient overflow: {consecutive_overflows} consecutive optimizer attempts"
            )
    if count <= 0:
        raise RuntimeError("hybrid epoch was empty")
    accuracy, macro_f1, f1s = _class_metrics(confusion)
    metrics: dict[str, Any] = {
        "huber": sums["huber"] / count, "ce": sums["ce"] / count,
        "total": sums["total"] / count,
        "raw_cc64_mae": sums["raw_cc64_absolute_error"] / count,
        "raw_cc64_rmse": math.sqrt(sums["raw_cc64_squared_error"] / count),
        "auxiliary_accuracy": accuracy, "auxiliary_macro_f1": macro_f1,
        "auxiliary_class_f1": f1s,
        "auxiliary_prediction_class_distribution": (confusion.sum(axis=0) / confusion.sum()).tolist(),
    }
    if not math.isclose(metrics["total"], metrics["huber"] + LAMBDA_CE * metrics["ce"], rel_tol=1e-6, abs_tol=2e-6):
        raise AssertionError("epoch total != Huber + 0.1*CE")
    return metrics, {
        "optimizer_steps": optimizer_steps,
        "optimizer_attempts": optimizer_attempts,
        "amp_skipped_steps": amp_skips,
        "amp_overflow_events": amp_overflow_events,
        "consecutive_amp_overflows": consecutive_overflows,
        "max_consecutive_amp_overflows": maximum_consecutive_overflows,
        "grad_scaler_scale": float(scaler.get_scale()) if training and amp_enabled else None,
    }


def write_metrics(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, path)


def checkpoint_payload(
    model: RawHuberAuxCEEncoderModel,
    run_config: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    best_epoch: int,
    best_validation_total: float,
    early_stopping_counter: int,
    global_optimizer_step: int,
    global_optimizer_attempt: int,
    amp_overflow_events_total: int,
    max_consecutive_amp_overflows: int,
    overflow_events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "grad_scaler_state": scaler.state_dict(),
        "rng_state": capture_rng_state(),
        "configuration": dict(run_config),
        "objective": "raw_huber_aux_ce", "epoch": int(row["epoch"]),
        "global_optimizer_step": int(global_optimizer_step),
        "global_optimizer_attempt": int(global_optimizer_attempt),
        "best_epoch": int(best_epoch),
        "best_validation_total": float(best_validation_total),
        "early_stopping_counter": int(early_stopping_counter),
        "amp_overflow_events_total": int(amp_overflow_events_total),
        "max_consecutive_amp_overflows": int(max_consecutive_amp_overflows),
        "amp_overflow_events": [dict(event) for event in overflow_events],
        "validation_objective": float(row["validation_total"]),
        "validation_huber": float(row["validation_huber"]),
        "validation_ce": float(row["validation_ce"]),
        "validation_raw_cc64_mae": float(row["validation_raw_cc64_mae"]),
        "checkpoint_selection": "minimum validation Huber + 0.1*CE",
    }


def restore_training_checkpoint(
    checkpoint_path: Path,
    model: RawHuberAuxCEEncoderModel,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required = {
        "model_state", "optimizer_state", "grad_scaler_state", "rng_state",
        "epoch", "global_optimizer_step", "best_epoch", "best_validation_total",
        "early_stopping_counter", "amp_overflow_events_total",
        "max_consecutive_amp_overflows", "configuration",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise KeyError(f"Hybrid checkpoint missing continuation state: {missing}")
    incompatible = model.load_state_dict(checkpoint["model_state"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict Hybrid checkpoint load failed: {incompatible}")
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scaler.load_state_dict(checkpoint["grad_scaler_state"])
    restore_rng_state(checkpoint["rng_state"])
    return checkpoint


def _initialization(config: Mapping[str, Any], base: Mapping[str, Any], phase: Mapping[str, Any]) -> tuple[RawHuberAuxCEEncoderModel, dict[str, str]]:
    set_deterministic_seed(int(base["seed"]))
    model = build_fresh_model(base)
    hashes = {
        "encoder": parameter_hash(model.encoder.parameters()),
        "regression": parameter_hash(model.regression_heads.parameters()),
        "classification": parameter_hash(model.classification_heads.parameters()),
    }
    raw_config = read_json(Path(config["raw_huber_root"]) / "config.json")
    if hashes["encoder"] != base["expected_initial_encoder_sha256"]:
        raise RuntimeError("fresh pretrained encoder hash mismatch")
    if hashes["regression"] != raw_config["initial_head_parameter_sha256"]:
        raise RuntimeError("fresh regression heads differ from Raw Huber seed-42 initialization")
    if len(set(hashes.values())) != 3:
        raise RuntimeError("head initialization hashes unexpectedly collide")
    return model, hashes


def train(config: Mapping[str, Any]) -> int:
    output = Path(config["output_root"])
    protected = ("best.pt", "last.pt", "config.json", "metrics.csv", "train.log")
    if any((output / name).exists() for name in protected):
        raise FileExistsError("refusing to overwrite hybrid full-run artifacts")
    output.mkdir(parents=True, exist_ok=True)
    log_handle = (output / "train.log").open("w", encoding="utf-8")
    def log(message: str) -> None:
        line = f"{now()} {message}"; print(line, flush=True); print(line, file=log_handle, flush=True)
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("training requires exactly one visible CUDA GPU")
        device = torch.device("cuda:0"); gpu = get_gpu_identity(); base = canonical_config(config)
        if gpu["uuid"] != base["expected_gpu_uuid"]: raise RuntimeError("assigned GPU UUID mismatch")
        phase = preflight(config); audit = phase["canonical_audit"]; raw_audit = phase["raw_target_audit"]
        log(f"CACHE_LOAD_PASS cache_id={audit['cache_id']} train=1170+892 validation=71 test_access=0 repedal=0")
        log(f"TARGETS_PASS raw_unique={raw_audit['unique_raw_values']} regression_quantization=false auxiliary=canonical_classes")
        model, hashes = _initialization(config, base, phase)
        run_config = {**base, **dict(config),
            "architecture": "PT pretrained 10-layer encoder + 4x Linear(768,1) regression + 4x Linear(768,4) auxiliary",
            "regression_output_shape": "B,N,4", "auxiliary_output_shape": "B,N,4,4",
            "fresh_initialization_seed": 42, "initialization_source": base["checkpoint_path"],
            "initial_parameter_sha256": hashes, "raw_target_audit": raw_audit,
            "canonical_cache_audit": audit, "gpu": gpu,
            "all_candidate_checkpoint_warm_starts": False, "asap_test_access": 0,
            "amp_overflow_policy": "recover isolated unscaled-gradient overflow; fail at 3 consecutive attempts",
            "checkpoint_schema": "model+optimizer+GradScaler+training counters+Python/NumPy/Torch CPU/CUDA RNG",
            "failed_v0_checkpoint_loaded": False,
            "modeling_hyperparameters_changed_from_v0": False,
        }
        atomic_json(output / "config.json", run_config)
        update_status(config, status="running", current_stage="model_initialized", fresh_seed=42,
            gpu=gpu, cache_id=audit["cache_id"], raw_target_quantization=False,
            lambda_ce=LAMBDA_CE, all_candidate_checkpoint_warm_starts=False)
        log(f"MODEL_INITIALIZED seed=42 encoder={hashes['encoder']} regression={hashes['regression']} auxiliary={hashes['classification']} warm_start=false lambda_ce=0.1")
        train_dataset = SharedBinaryWindowDataset(base["cache_root"], "train")
        validation_dataset = SharedBinaryWindowDataset(base["cache_root"], "validation")
        model.to(device)
        optimizer = build_binary_optimizer(model, encoder_lr=float(base["encoder_lr"]), head_lr=float(base["head_lr"]), weight_decay=float(base["weight_decay"]))
        scaler = torch.amp.GradScaler("cuda", enabled=bool(base["amp_enabled"]), init_scale=float(base["amp_init_scale"]))
        validation_loader = make_raw_cc64_loader(validation_dataset, batch_size=int(base["micro_batch_size"]["encoder_only"]), pin_memory=bool(base["pin_memory"]))
        rows: list[dict[str, Any]] = []
        best = early_best = math.inf
        best_epoch = stale = 0
        first = False
        global_optimizer_step = global_optimizer_attempt = 0
        amp_overflow_events_total = max_consecutive_overflows = consecutive_overflows = 0
        overflow_events: list[dict[str, Any]] = []
        update_status(
            config, status="running", current_stage="training", current_epoch=1,
            global_optimizer_step=0, best_epoch=0, best_validation_total=None,
            early_stopping_counter=0, amp_overflow_events_total=0,
            amp_overflow_events_current_epoch=0, max_consecutive_amp_overflows=0,
            last_overflow=None, forward_nonfinite_count=0,
            parameter_nonfinite_count=0, checkpoint_has_optimizer_state=False,
            checkpoint_has_scaler_state=False, checkpoint_has_rng_state=False,
            test_access_count=0,
        )
        for epoch in range(1, int(base["max_epochs"]) + 1):
            started = time.perf_counter(); order = training_window_order(len(train_dataset), int(base["seed"]), epoch)
            loader = make_raw_cc64_loader(train_dataset, batch_size=int(base["micro_batch_size"]["encoder_only"]), pin_memory=bool(base["pin_memory"]), order=order)
            steps_before_epoch = global_optimizer_step
            attempts_before_epoch = global_optimizer_attempt
            overflows_before_epoch = amp_overflow_events_total
            def callback(step: int, details: Mapping[str, Any]) -> None:
                nonlocal first, amp_overflow_events_total, max_consecutive_overflows
                current_global_step = steps_before_epoch + int(details["optimizer_steps"])
                current_global_attempt = attempts_before_epoch + int(details["optimizer_attempts"])
                if details["amp_overflow_event"]:
                    amp_overflow_events_total = overflows_before_epoch + int(details["amp_overflow_events"])
                    max_consecutive_overflows = max(max_consecutive_overflows, int(details["max_consecutive_amp_overflows"]))
                    event = {
                        "epoch": epoch, "step": step,
                        "global_optimizer_attempt": current_global_attempt,
                        "huber": float(details["huber"]), "ce": float(details["ce"]),
                        "total": float(details["total"]), "grad_norm": float(details["gradient_norm"]),
                        "scale_before": float(details["scale_before"]),
                        "scale_after": float(details["scale_after"]),
                        "affected_groups": details["affected_groups"],
                        "offending_parameter_names": details["offending_parameter_names"],
                        "offending_parameter_count": int(details["offending_parameter_count"]),
                        "overflow_events_total": amp_overflow_events_total,
                        "consecutive_overflows": int(details["consecutive_amp_overflows"]),
                        "scaler_step_update_executed": bool(details["scaler_step_update_executed"]),
                        "optimizer_step_skipped": bool(details["optimizer_step_skipped"]),
                    }
                    overflow_events.append(event)
                    update_status(
                        config, status="running", current_stage="amp_overflow_recovered",
                        current_epoch=epoch, current_step=step,
                        global_optimizer_step=current_global_step,
                        global_optimizer_attempt=current_global_attempt,
                        best_epoch=best_epoch,
                        best_validation_total=None if math.isinf(best) else best,
                        early_stopping_counter=stale,
                        amp_overflow_events_total=amp_overflow_events_total,
                        amp_overflow_events_current_epoch=int(details["amp_overflow_events"]),
                        max_consecutive_amp_overflows=max_consecutive_overflows,
                        last_overflow=event, forward_nonfinite_count=0,
                        parameter_nonfinite_count=0, test_access_count=0,
                    )
                    log("AMP_OVERFLOW_EVENT " + json.dumps(event, sort_keys=True))
                if not first and details["optimizer_step_applied"]:
                    values = [float(details[key]) for key in ("huber", "ce", "total")]
                    if not all(math.isfinite(v) for v in values) or int(details["non_representative_unique_count"]) <= 0:
                        raise FloatingPointError("invalid first hybrid step")
                    first = True
                    update_status(config, status="running", current_stage="first_optimizer_step_pass", first_optimizer_step=True,
                        first_step_epoch=epoch, first_step=step, first_step_huber=values[0], first_step_ce=values[1], first_step_total=values[2],
                        first_step_gradient_norm=float(details["gradient_norm"]), first_batch_raw_unique_count=int(details["raw_unique_count"]),
                        first_batch_raw_examples=details["raw_unique_examples"], first_batch_canonical_classes=details["canonical_class_examples"],
                        global_optimizer_step=current_global_step,
                        first_step_grad_scaler_scale=float(details["scale_after"]),
                        amp_overflow_events_total=amp_overflow_events_total,
                        test_access_count=0)
                    log(f"FIRST_OPTIMIZER_STEP_PASS epoch={epoch} step={step} huber={values[0]:.9f} ce={values[1]:.9f} total={values[2]:.9f} grad_norm={float(details['gradient_norm']):.6f} scaler={float(details['scale_after']):.1f} raw_unique={details['raw_unique_count']} classes={details['canonical_class_examples']}")
                elif step % int(base["progress_interval"]) == 0:
                    update_status(config, status="running", current_stage="training", current_epoch=epoch, current_step=step,
                        global_optimizer_step=current_global_step, global_optimizer_attempt=current_global_attempt,
                        amp_overflow_events_total=amp_overflow_events_total, best_epoch=best_epoch,
                        best_validation_total=None if math.isinf(best) else best, early_stopping_counter=stale)
                    log(f"TRAIN_PROGRESS epoch={epoch} step={step} total={float(details['total']):.9f}")
            train_metrics, checks = run_epoch(
                model, loader, device=device, amp_enabled=bool(base["amp_enabled"]),
                optimizer=optimizer, scaler=scaler, max_grad_norm=float(base["max_grad_norm"]),
                step_callback=callback, max_consecutive_amp_overflows=MAX_CONSECUTIVE_AMP_OVERFLOWS,
                initial_consecutive_amp_overflows=consecutive_overflows,
            )
            if not first: raise RuntimeError("first finite optimizer step not verified")
            global_optimizer_step += int(checks["optimizer_steps"])
            global_optimizer_attempt += int(checks["optimizer_attempts"])
            amp_overflow_events_total = overflows_before_epoch + int(checks["amp_overflow_events"])
            consecutive_overflows = int(checks["consecutive_amp_overflows"])
            max_consecutive_overflows = max(max_consecutive_overflows, int(checks["max_consecutive_amp_overflows"]))
            validation_metrics, _ = run_epoch(model, validation_loader, device=device, amp_enabled=bool(base["amp_enabled"]))
            row = {"epoch": epoch, **{f"train_{k}": v for k,v in train_metrics.items()}, **{f"validation_{k}": v for k,v in validation_metrics.items()},
                "encoder_lr": float(optimizer.param_groups[0]["lr"]), "head_lr": float(optimizer.param_groups[1]["lr"]),
                "optimizer_steps": checks["optimizer_steps"], "optimizer_attempts": checks["optimizer_attempts"],
                "global_optimizer_step": global_optimizer_step, "global_optimizer_attempt": global_optimizer_attempt,
                "amp_skipped_steps": checks["amp_skipped_steps"], "amp_overflow_events": checks["amp_overflow_events"],
                "max_consecutive_amp_overflows": checks["max_consecutive_amp_overflows"],
                "grad_scaler_scale": checks["grad_scaler_scale"], "epoch_seconds": time.perf_counter()-started}
            rows.append(row); write_metrics(output / "metrics.csv", rows); candidate = float(row["validation_total"])
            improved = candidate < best
            if improved: best, best_epoch = candidate, epoch
            if candidate < early_best - float(base["early_stopping_min_delta"]): early_best, stale = candidate, 0
            else: stale += 1
            payload = checkpoint_payload(
                model, run_config, row, optimizer=optimizer, scaler=scaler,
                best_epoch=best_epoch, best_validation_total=best,
                early_stopping_counter=stale, global_optimizer_step=global_optimizer_step,
                global_optimizer_attempt=global_optimizer_attempt,
                amp_overflow_events_total=amp_overflow_events_total,
                max_consecutive_amp_overflows=max_consecutive_overflows,
                overflow_events=overflow_events,
            )
            atomic_torch_save(payload, output / "last.pt")
            if improved: atomic_torch_save(payload, output / "best.pt")
            update_status(config, status="running", current_stage="epoch_completed", current_epoch=epoch, best_epoch=best_epoch,
                best_validation_objective=best, best_validation_total=best, early_stopping_counter=stale,
                global_optimizer_step=global_optimizer_step, global_optimizer_attempt=global_optimizer_attempt,
                validation_huber=row["validation_huber"], validation_ce=row["validation_ce"], validation_total=row["validation_total"],
                amp_overflow_events_total=amp_overflow_events_total,
                amp_overflow_events_current_epoch=int(checks["amp_overflow_events"]),
                max_consecutive_amp_overflows=max_consecutive_overflows,
                checkpoint_has_optimizer_state=True, checkpoint_has_scaler_state=True,
                checkpoint_has_rng_state=True, current_grad_scaler_scale=float(scaler.get_scale()), test_access_count=0)
            log(f"EPOCH_COMPLETE epoch={epoch} train_huber={row['train_huber']:.9f} train_ce={row['train_ce']:.9f} train_total={row['train_total']:.9f} validation_huber={row['validation_huber']:.9f} validation_ce={row['validation_ce']:.9f} validation_total={candidate:.9f} best_epoch={best_epoch} amp_overflows={checks['amp_overflow_events']} scaler={float(scaler.get_scale()):.1f}")
            if stale >= int(base["early_stopping_patience"]): log(f"EARLY_STOP epoch={epoch}"); break
        update_status(config, status="trained", current_stage="training_completed", training_completed=True, best_epoch=best_epoch, best_validation_objective=best,
            best_validation_total=best, early_stopping_counter=stale, global_optimizer_step=global_optimizer_step,
            global_optimizer_attempt=global_optimizer_attempt, amp_overflow_events_total=amp_overflow_events_total,
            max_consecutive_amp_overflows=max_consecutive_overflows, overflow_events=overflow_events,
            forward_nonfinite_count=0, parameter_nonfinite_count=0, test_access_count=0)
        log(f"TRAINING_COMPLETE best_epoch={best_epoch} best_validation_total={best:.9f}"); return 0
    except Exception as error:
        log(f"TRAINING_FAILED {type(error).__name__}: {error}"); traceback.print_exc(file=log_handle)
        current = read_json(output / "run_status.json") if (output / "run_status.json").is_file() else {}
        message = str(error)
        forward_count = int(current.get("forward_nonfinite_count", 0))
        parameter_count = int(current.get("parameter_nonfinite_count", 0))
        if "objective is non-finite" in message: forward_count += 1
        if "non-finite model parameter" in message: parameter_count += 1
        update_status(config, status="failed", current_stage="training_failed", error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(), forward_nonfinite_count=forward_count,
            parameter_nonfinite_count=parameter_count, test_access_count=0); return 1
    finally: log_handle.close()


def smoke(config: Mapping[str, Any]) -> int:
    output = Path(config["output_root"]); smoke_root = output / "preflight_smoke"
    if smoke_root.exists(): raise FileExistsError(f"refusing to overwrite smoke: {smoke_root}")
    output.mkdir(parents=True, exist_ok=True); phase = preflight(config); base = canonical_config(config)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1: raise RuntimeError("smoke requires one CUDA GPU")
    if get_gpu_identity()["uuid"] != base["expected_gpu_uuid"]: raise RuntimeError("smoke GPU mismatch")
    device = torch.device("cuda:0"); model, hashes = _initialization(config, base, phase); model.to(device)
    before = parameter_hash(model.prediction_head_parameters())
    optimizer = build_binary_optimizer(model, encoder_lr=float(base["encoder_lr"]), head_lr=float(base["head_lr"]), weight_decay=float(base["weight_decay"]))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(base["amp_enabled"]), init_scale=float(base["amp_init_scale"]))
    train_data = SharedBinaryWindowDataset(base["cache_root"], "train"); val_data = SharedBinaryWindowDataset(base["cache_root"], "validation"); details: dict[str, Any] = {}
    train_metrics, checks = run_epoch(model, make_raw_cc64_loader(train_data, batch_size=1, pin_memory=bool(base["pin_memory"]), order=[0]), device=device, amp_enabled=bool(base["amp_enabled"]), optimizer=optimizer, scaler=scaler, max_grad_norm=float(base["max_grad_norm"]), step_callback=lambda _s,d: details.update(d), max_batches=1, inspect_finite_gradient_groups=True)
    validation_metrics, _ = run_epoch(model, make_raw_cc64_loader(val_data, batch_size=1, pin_memory=bool(base["pin_memory"]), order=[0]), device=device, amp_enabled=bool(base["amp_enabled"]), max_batches=1)
    after = parameter_hash(model.prediction_head_parameters()); smoke_root.mkdir(parents=True)
    checkpoint = smoke_root / "raw_huber_aux_ce.pt"
    row = {"epoch": 1, "validation_total": validation_metrics["total"],
        "validation_huber": validation_metrics["huber"], "validation_ce": validation_metrics["ce"],
        "validation_raw_cc64_mae": validation_metrics["raw_cc64_mae"]}
    payload = checkpoint_payload(model, dict(config), row, optimizer=optimizer, scaler=scaler,
        best_epoch=1, best_validation_total=validation_metrics["total"], early_stopping_counter=0,
        global_optimizer_step=1, global_optimizer_attempt=1, amp_overflow_events_total=0,
        max_consecutive_amp_overflows=0, overflow_events=[])
    atomic_torch_save(payload, checkpoint)
    loaded_model, _ = _initialization(config, base, phase)
    loaded_optimizer = build_binary_optimizer(loaded_model, encoder_lr=float(base["encoder_lr"]), head_lr=float(base["head_lr"]), weight_decay=float(base["weight_decay"]))
    loaded_scaler = torch.amp.GradScaler("cuda", enabled=bool(base["amp_enabled"]), init_scale=float(base["amp_init_scale"]))
    loaded = restore_training_checkpoint(checkpoint, loaded_model, loaded_optimizer, loaded_scaler)
    required = {"model_state","optimizer_state","grad_scaler_state","rng_state","epoch","global_optimizer_step","best_epoch","best_validation_total","early_stopping_counter","amp_overflow_events_total","max_consecutive_amp_overflows","configuration"}
    if checks["optimizer_steps"] != 1 or before == after or not required.issubset(loaded) or int(details["non_representative_unique_count"]) <= 0 or not all(details["gradient_groups_finite_nonzero"].values()):
        raise RuntimeError("hybrid smoke failed")
    if not all(math.isfinite(float(metrics[k])) for metrics in (train_metrics, validation_metrics) for k in ("huber","ce","total")):
        raise RuntimeError("hybrid smoke loss is non-finite")
    result = {"status":"passed", "finite_huber":train_metrics["huber"], "finite_ce":train_metrics["ce"], "finite_total":train_metrics["total"],
        "finite_backward":True, "gradient_groups_finite_nonzero":details["gradient_groups_finite_nonzero"], "optimizer_step":True, "checkpoint_serialization":True, "checkpoint_reload":True,
        "validation_huber":validation_metrics["huber"], "validation_ce":validation_metrics["ce"], "validation_total":validation_metrics["total"],
        "raw_batch_unique_count":details["raw_unique_count"], "canonical_class_examples":details["canonical_class_examples"],
        "regression_inference_uses_auxiliary":False, "fresh_initialization_hashes":hashes, "checkpoint":str(checkpoint),
        "checkpoint_has_model_state":True, "checkpoint_has_optimizer_state":True,
        "checkpoint_has_scaler_state":True, "checkpoint_has_rng_state":True,
        "checkpoint_strict_restore":True, "restored_global_optimizer_step":loaded["global_optimizer_step"],
        "restored_grad_scaler_scale":float(loaded_scaler.get_scale()),
        "asap_test_access_count":0, "repedal_execution_count":0}
    atomic_json(smoke_root / "smoke_results.json", result); return 0


def run_process(command: Sequence[str], log_handle: Any) -> int:
    process = subprocess.Popen(list(command), cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert process.stdout is not None
    for line in process.stdout: print(line, end="", flush=True); print(line, end="", file=log_handle, flush=True)
    return process.wait()


def overnight(config: Mapping[str, Any], config_path: Path) -> int:
    output = Path(config["output_root"]); smoke_path = output / "preflight_smoke/smoke_results.json"
    if not smoke_path.is_file() or read_json(smoke_path).get("status") != "passed": raise RuntimeError("full run blocked until smoke passes")
    if any((output/name).exists() for name in ("best.pt","last.pt","config.json","metrics.csv","train.log","overnight.log")): raise FileExistsError("refusing to overwrite full-run output")
    update_status(config, experiment_id=config["experiment_id"], status="running", current_stage="starting", training_completed=False, evaluation_completed=False, reports_generated=False, fresh_seed=42, all_candidate_checkpoint_warm_starts=False, started_at=now())
    with (output / "overnight.log").open("w", encoding="utf-8") as log_handle:
        commands = [
            [sys.executable, str(Path(__file__).resolve()), "--config", str(config_path), "--train"],
            [sys.executable, str(ROOT/"scripts/evaluate_stage2_encoder_only_raw_huber_aux_ce_v0.py"), "--config", str(config_path), "--execute"],
            [sys.executable, str(ROOT/"scripts/report_stage2_encoder_only_raw_huber_aux_ce_v0.py"), "--config", str(config_path), "--execute"],
        ]
        stages = ("training", "validation_evaluation", "report_generation")
        for stage, command in zip(stages, commands):
            update_status(config, status="running", current_stage=stage)
            code = run_process(command, log_handle)
            if code != 0:
                update_status(config, status="failed", current_stage=f"{stage}_failed", return_code=code); return code
        update_status(config, status="completed", current_stage="completed", evaluation_completed=True, reports_generated=True, finished_at=now()); return 0


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    for name in ("preflight","smoke","train","overnight"): modes.add_argument(f"--{name}", action="store_true")
    args = parser.parse_args(); config = read_json(args.config)
    try:
        if args.preflight: print(json.dumps(preflight(config), indent=2, sort_keys=True)); return 0
        if args.smoke: return smoke(config)
        if args.train: return train(config)
        return overnight(config, args.config.resolve())
    finally:
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == "__main__": raise SystemExit(main())
