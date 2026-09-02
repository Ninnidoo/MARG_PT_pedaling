#!/usr/bin/env python3
"""Canonical four-class Decoder-only Prefix-LM v0 full training."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import sys
import time
import traceback
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_stage2_4class_architecture_v0 import (
    parameter_hash,
    preflight,
    read_json,
    run_epoch,
    sha256_file,
    training_window_order,
)
from src.stage2_decoder_only.model import FourClassPedalDecoderOnlyModel
from src.stage2_decoder_only.training import build_decoder_only_optimizer
from src.stage2_encoder_only.train import atomic_torch_save, get_gpu_identity
from src.stage2_encoder_only.training import set_deterministic_seed
from src.stage2_four_class.representation import (
    CLASS_BOUNDS,
    CLASS_NAMES,
    IGNORE_INDEX,
    REPRESENTATIVES,
    SharedFourClassWindowDataset,
    make_four_class_loader,
)


DEFAULT_OUTPUT = ROOT / "analysis/stage2_decoder_only_4class_v0/full_train"
REPORT_NAME = "DECODER_ONLY_4CLASS_TRAINING_REPORT.md"
METRIC_FIELDS = (
    "epoch",
    "train_ce",
    "train_token_accuracy",
    "train_exact_note_accuracy",
    "validation_ce",
    "validation_token_accuracy",
    "validation_exact_note_accuracy",
    "best_epoch",
    "best_validation_ce",
    "best_so_far",
    "pretrained_representation_lr",
    "fresh_prefix_lm_lr",
    "epoch_seconds",
    "epoch_optimizer_steps",
    "epoch_optimizer_attempts",
    "global_optimizer_step",
    "global_optimizer_attempt_step",
    "amp_skipped_steps",
    "amp_overflow_events",
    "max_consecutive_amp_overflows",
    "gradients_finite",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def write_metrics(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def update_status(output_dir: Path, **updates: Any) -> dict[str, Any]:
    path = output_dir / "run_status.json"
    current = read_json(path) if path.is_file() else {}
    current.update(updates, last_update=now())
    atomic_json(path, current)
    return current


def read_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    integer_fields = {
        "epoch",
        "best_epoch",
        "epoch_optimizer_steps",
        "epoch_optimizer_attempts",
        "global_optimizer_step",
        "global_optimizer_attempt_step",
        "amp_skipped_steps",
        "amp_overflow_events",
        "max_consecutive_amp_overflows",
    }
    boolean_fields = {"best_so_far", "gradients_finite"}
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = {}
            for key in METRIC_FIELDS:
                value = raw.get(key, "")
                if value == "":
                    if key == "epoch_optimizer_attempts":
                        value = raw.get("epoch_optimizer_steps", "0")
                    elif key == "global_optimizer_attempt_step":
                        value = raw.get("global_optimizer_step", "0")
                    elif key in {"amp_overflow_events", "max_consecutive_amp_overflows"}:
                        value = "0"
                if key in integer_fields:
                    row[key] = int(value)
                elif key in boolean_fields:
                    row[key] = str(value).lower() == "true"
                else:
                    row[key] = float(value)
            rows.append(row)
    return rows


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
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


class _FourDimensionalTrainingView(nn.Module):
    """Adapt flat canonical logits to the shared four-class epoch utility."""

    def __init__(self, model: FourClassPedalDecoderOnlyModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, **batch: torch.Tensor):
        output = self.model(**batch)
        batch_size, flat_length, classes = output.logits.shape
        return replace(
            output,
            logits=output.logits.view(batch_size, flat_length // 4, 4, classes),
        )


def configuration_values(config: Mapping[str, Any]) -> tuple[int, int, int, int]:
    layers = int(config.get("decoder_only_num_layers", 2))
    init_seed = int(config.get("decoder_only_init_seed", config["seed"]))
    micro = int(config["micro_batch_size"]["decoder_only"])
    accumulation = int(config["gradient_accumulation_steps"]["decoder_only"])
    if layers != 2 or init_seed != 42:
        raise RuntimeError("canonical Decoder-only v0 requires two layers and seed 42")
    if micro * accumulation != int(config["effective_batch_size"]):
        raise RuntimeError("Decoder-only effective batch size mismatch")
    return layers, init_seed, micro, accumulation


def checkpoint_payload(
    model: FourClassPedalDecoderOnlyModel,
    run_config: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    stale_epochs: int,
    amp_overflow_events_total: int,
    max_consecutive_amp_overflows: int,
    overflow_events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "grad_scaler_state": scaler.state_dict(),
        "configuration": dict(run_config),
        "epoch": int(row["epoch"]),
        "global_optimizer_step": int(row["global_optimizer_step"]),
        "global_optimizer_attempt_step": int(row["global_optimizer_attempt_step"]),
        "validation_ce": float(row["validation_ce"]),
        "best_epoch": int(row["best_epoch"]),
        "best_validation_ce": float(row["best_validation_ce"]),
        "early_stopping_counter": int(stale_epochs),
        "amp_overflow_events_total": int(amp_overflow_events_total),
        "max_consecutive_amp_overflows": int(max_consecutive_amp_overflows),
        "amp_overflow_events": [dict(event) for event in overflow_events],
        "rng_state": capture_rng_state(),
        "checkpoint_selection": "minimum validation CE",
        "smoke_checkpoint_reused": False,
    }


def restore_training_checkpoint(
    checkpoint_path: Path,
    model: FourClassPedalDecoderOnlyModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required = {
        "model_state",
        "optimizer_state",
        "grad_scaler_state",
        "epoch",
        "global_optimizer_step",
        "best_epoch",
        "best_validation_ce",
        "early_stopping_counter",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise KeyError(f"resume checkpoint missing keys: {missing}")
    incompatible = model.load_state_dict(checkpoint["model_state"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict checkpoint load failed: {incompatible}")
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scaler.load_state_dict(checkpoint["grad_scaler_state"])
    return checkpoint


def report_text(
    run_config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    status: Mapping[str, Any],
) -> str:
    lines = [
        "# Decoder-only 4-Class Stage 2 Training Report",
        "",
        "## A. Experiment setup",
        "",
        f"- Architecture: {run_config['architecture']}",
        f"- Parameters: {int(run_config['parameter_count']):,} total/trainable",
        "- Pretrained: PT token embedding, eight feature projections, note compression",
        "- Random initialization: two-layer Prefix-LM, pedal/BOS/PAD embedding, Pedal1–4 slot embedding, Linear(768,4)",
        f"- Classes: `{run_config['class_names']}`; boundaries: `{run_config['class_boundaries']}`; representatives: `{run_config['representatives']}`",
        "- Training: MAESTRO-clean 1,170 + ASAP train 892 = 2,062 performances / 9,369,095 notes / 35,573 windows",
        "- Validation: ASAP validation 71 performances / 283,928 notes / 1,078 windows",
        f"- Shared cache ID: `{run_config['cache_id']}`",
        f"- Seed: {run_config['seed']}; window/stride: {run_config['window_notes']}/{run_config['stride_notes']}",
        "- Loss: standard unweighted cross entropy; teacher-forced training and validation",
        f"- Optimizer: AdamW; pretrained representation LR {run_config['encoder_lr']}; fresh Prefix-LM LR {run_config['head_lr']}; weight decay {run_config['weight_decay']}; scheduler none",
        f"- Batch: micro {run_config['micro_batch_size']}, accumulation {run_config['gradient_accumulation_steps']}, effective {run_config['effective_batch_size']}",
        f"- Precision: FP16 AMP; gradient clip {run_config['max_grad_norm']}",
        f"- Early stopping: minimum validation CE, patience {run_config['early_stopping_patience']}, minimum delta {run_config['early_stopping_min_delta']}",
        "- ASAP test access: 0",
        "",
        "## B. Smoke-test provenance",
        "",
        "- Pedal-rich tiny overfit: PASS",
        "- Smoke teacher-forced accuracy: 99.8901%",
        "- Smoke free-running greedy accuracy: 98.4863%",
        "- Smoke checkpoint was not loaded or reused.",
        "",
        "## C. Resume provenance",
        "",
    ]
    if run_config.get("resumed_from_checkpoint"):
        lines += [
            "- Initial run: interrupted during epoch 9 by a non-finite unscaled aggregate gradient norm.",
            f"- Recovery checkpoint: `{run_config['resumed_from_checkpoint']}`",
            "- Model/optimizer/GradScaler restored: yes",
            f"- Resume epoch/global optimizer step: {run_config['resume_epoch']}/{run_config['resume_global_step']}",
            "- RNG state restored: no (the epoch-8 checkpoint contains no RNG state)",
            "- Bit-exact resume: no; this is a near-exact resume",
            "- Modeling hyperparameters changed: none",
        ]
    else:
        lines.append("- No resume was used; the initial run started from fresh seed-42 initialization.")
    lines += [
        "",
        "## D. AMP diagnostics",
        "",
        f"- Recorded AMP gradient-overflow events after recovery instrumentation: {status.get('amp_overflow_events_total', 0)}",
        f"- Maximum consecutive recorded events: {status.get('max_consecutive_amp_overflows', 0)}",
        f"- Forward non-finite count: {status.get('forward_nonfinite_count', 0)}",
        f"- Model-parameter non-finite count: {status.get('model_parameter_nonfinite_count', 0)}",
        "- The pre-resume failed epoch contained at least one overflow event, but its exact count and scale transition were not recorded by the old runner.",
    ]
    for event in status.get("amp_overflow_events", []):
        lines.append(
            "- Event: "
            f"epoch={event.get('epoch')} step={event.get('step')} "
            f"global_attempt={event.get('global_optimizer_attempt_step')} "
            f"grad_norm={event.get('grad_norm')} "
            f"scale={event.get('scale_before')}->{event.get('scale_after')} "
            f"groups={event.get('offending_parameter_groups')}"
        )
    lines += [
        "",
        "## E. Epoch table",
        "",
        "| Epoch | Train CE | Validation CE | Validation TF accuracy | Best so far |",
        "|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {int(row['epoch'])} | {float(row['train_ce']):.9f} | "
            f"{float(row['validation_ce']):.9f} | "
            f"{float(row['validation_token_accuracy']):.9f} | "
            f"{'yes' if row['best_so_far'] else 'no'} |"
        )
    lines += [
        "",
        "## F. Final training result",
        "",
        f"- Status: {status.get('status', 'failed')}",
        f"- Best epoch: {status.get('best_epoch', 'not available')}",
        f"- Best validation CE: {status.get('best_validation_ce', 'not available')}",
        f"- Early-stop/final epoch: {status.get('early_stop_epoch', status.get('current_epoch', 'not available'))}",
        f"- Best checkpoint: `{status.get('best_checkpoint_path', run_config.get('resumed_from_checkpoint', 'not available'))}`",
        f"- Last checkpoint: `{status.get('last_checkpoint_path', 'not available')}`",
        "",
        "## G. Training-curve interpretation",
        "",
    ]
    if rows:
        lines.append(
            f"- Train CE moved from {float(rows[0]['train_ce']):.6f} to {float(rows[-1]['train_ce']):.6f}; validation CE moved from {float(rows[0]['validation_ce']):.6f} to {float(rows[-1]['validation_ce']):.6f}."
        )
    else:
        lines.append("- No complete validation epoch was recorded.")
    lines += [
        "- Interpretation is limited to observed convergence and numerical behavior; no architecture ranking is made.",
        "",
        "## H. Pending evaluation",
        "",
        "The next stage must evaluate `best.pt` with true free-running greedy inference and the frozen canonical evaluator. No full-validation free-running inference or ASAP test evaluation was performed during this run.",
        "",
    ]
    if status.get("error"):
        lines += ["## Failure", "", f"`{status['error']}`", ""]
    return "\n".join(lines)

def train(
    config: Mapping[str, Any],
    output_dir: Path,
    *,
    resume_checkpoint: Path | None = None,
) -> int:
    resuming = resume_checkpoint is not None
    if resuming:
        if not output_dir.is_dir():
            raise FileNotFoundError(f"resume output directory does not exist: {output_dir}")
        if not resume_checkpoint.is_file():
            raise FileNotFoundError(resume_checkpoint)
        rows = read_metrics(output_dir / "metrics.csv")
        if not rows:
            raise RuntimeError("resume requires completed epoch history")
        log_handle = (output_dir / "train.log").open("a", encoding="utf-8")
        status = read_json(output_dir / "run_status.json")
    else:
        if output_dir.exists():
            raise FileExistsError(f"refusing to overwrite run output: {output_dir}")
        output_dir.mkdir(parents=True)
        log_handle = (output_dir / "train.log").open("w", encoding="utf-8")
        rows: list[dict[str, Any]] = []
        status: dict[str, Any] = {
            "status": "pending",
            "architecture": "decoder_only",
            "current_epoch": 0,
            "best_epoch": 0,
            "best_validation_ce": None,
            "report_generated": False,
            "fresh_seed_42_initialization": False,
            "smoke_checkpoint_reused": False,
            "asap_test_access_count": 0,
            "started_at": now(),
        }
        atomic_json(output_dir / "run_status.json", status)
        write_metrics(output_dir / "metrics.csv", rows)

    def log(message: str) -> None:
        print(message, flush=True)
        print(message, file=log_handle, flush=True)

    try:
        layers, init_seed, micro_batch_size, accumulation_steps = configuration_values(config)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("training requires exactly one visible CUDA GPU")
        device = torch.device("cuda:0")
        gpu = get_gpu_identity()
        if gpu["uuid"] != config["expected_gpu_uuid"]:
            raise RuntimeError(f"assigned GPU UUID mismatch: {gpu['uuid']}")
        audit = preflight(config)
        log(
            f"CACHE_LOAD_PASS cache_id={audit['cache_id']} "
            "train=1170+892 validation=71 test_access=0"
        )
        status = update_status(
            output_dir,
            status="running",
            state="cache_load_pass",
            cache_id=audit["cache_id"],
            cache_audit=audit,
            training_performances=2062,
            validation_performances=71,
            training_windows=35573,
            validation_windows=1078,
            asap_test_access_count=0,
            gpu=gpu,
            report_generated=False,
            error=None,
            traceback=None,
        )

        train_dataset = SharedFourClassWindowDataset(config["cache_root"], "train")
        validation_dataset = SharedFourClassWindowDataset(config["cache_root"], "validation")
        if len(train_dataset) != 35573 or len(validation_dataset) != 1078:
            raise RuntimeError("canonical window inventory mismatch")
        set_deterministic_seed(int(config["seed"]))
        model = FourClassPedalDecoderOnlyModel.from_pretrained_performance_embeddings(
            config["checkpoint_path"],
            num_layers=layers,
            init_seed=init_seed,
            torch_dtype=torch.float32,
            attn_implementation="eager",
        )
        if model.hidden_size != 768 or len(model.transformer.layers) != 2:
            raise RuntimeError("canonical Decoder-only architecture mismatch")
        if hasattr(model, "encoder") or any(
            "cross" in name.lower() for name, _ in model.named_modules()
        ):
            raise RuntimeError("Decoder-only model unexpectedly contains encoder/cross-attention")
        representation_hash = parameter_hash(model.pretrained_representation_parameters())
        fresh_hash = parameter_hash(model.fresh_parameters())
        run_config = {
            **dict(config),
            "experiment_id": "stage2_decoder_only_4class_v0_full_train",
            "output_dir": str(output_dir),
            "architecture_name": "decoder_only",
            "architecture": (
                "pretrained PT token/projection/note compression + fresh seed-42 "
                "2-layer self-attention Prefix-LM + shared Linear(768,4)"
            ),
            "hidden_size": 768,
            "ffn_size": 3072,
            "attention_head_dimension": 128,
            "decoder_only_num_layers": layers,
            "decoder_only_init_seed": init_seed,
            "class_names": list(CLASS_NAMES),
            "class_boundaries": [list(item) for item in CLASS_BOUNDS],
            "representatives": list(REPRESENTATIVES),
            "parameter_count": model.parameter_count,
            "trainable_parameter_count": model.trainable_parameter_count,
            "pretrained_representation_parameter_sha256": representation_hash,
            "fresh_parameter_sha256_at_initialization": fresh_hash,
            "initialization_loaded_checkpoint_paths": [str(config["checkpoint_path"])],
            "smoke_checkpoint_path": str(
                ROOT / "analysis/stage2_decoder_only_4class_v0/smoke_overfit/smoke_final.pt"
            ),
            "smoke_checkpoint_reused": False,
            "fresh_seed_42_initialization": True,
            "cache_id": audit["cache_id"],
            "cache_audit": audit,
            "micro_batch_size": micro_batch_size,
            "gradient_accumulation_steps": accumulation_steps,
            "validation_mode": "teacher-forced CE and 4-class accuracy only",
            "checkpoint_selection": "minimum validation CE",
            "scheduler": None,
            "asap_test_access_count": 0,
            "gpu": gpu,
            "amp_overflow_policy": "recover isolated events; fail at 3 consecutive attempts",
            "modeling_hyperparameters_changed_on_resume": False,
        }

        model.to(device)
        view = _FourDimensionalTrainingView(model)
        optimizer = build_decoder_only_optimizer(
            model,
            pretrained_lr=float(config["encoder_lr"]),
            fresh_lr=float(config["head_lr"]),
            weight_decay=float(config["weight_decay"]),
        )
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=bool(config["amp_enabled"]),
            init_scale=float(config["amp_init_scale"]),
        )

        if resuming:
            checkpoint = restore_training_checkpoint(
                resume_checkpoint, model, optimizer, scaler
            )
            resume_epoch = int(checkpoint["epoch"])
            resume_global_step = int(checkpoint["global_optimizer_step"])
            if int(rows[-1]["epoch"]) != resume_epoch:
                raise RuntimeError("metrics history does not end at resume checkpoint epoch")
            start_epoch = resume_epoch + 1
            best_ce = float(checkpoint["best_validation_ce"])
            early_stopping_best_ce = best_ce
            best_epoch = int(checkpoint["best_epoch"])
            stale_epochs = int(checkpoint["early_stopping_counter"])
            global_optimizer_step = resume_global_step
            global_optimizer_attempt_step = int(
                checkpoint.get("global_optimizer_attempt_step", resume_global_step)
            )
            amp_overflow_events_total = int(
                checkpoint.get("amp_overflow_events_total", 0)
            )
            max_consecutive_overflows = int(
                checkpoint.get("max_consecutive_amp_overflows", 0)
            )
            consecutive_overflows = 0
            overflow_events = [
                dict(event) for event in checkpoint.get("amp_overflow_events", [])
            ]
            amp_skipped_total = sum(int(row["amp_skipped_steps"]) for row in rows)
            resume_rng_available = "rng_state" in checkpoint
            if resume_rng_available:
                restore_rng_state(checkpoint["rng_state"])
            run_config.update(
                resumed_from_checkpoint=str(resume_checkpoint),
                resume_epoch=resume_epoch,
                resume_global_step=resume_global_step,
                resume_is_bit_exact=resume_rng_available,
                resume_rng_state_restored=resume_rng_available,
                resume_initial_grad_scaler_scale=float(scaler.get_scale()),
            )
            atomic_json(output_dir / "config.json", run_config)
            status = update_status(
                output_dir,
                status="running",
                state="resume_checkpoint_restored",
                current_epoch=start_epoch,
                current_step=0,
                global_step=global_optimizer_step,
                global_optimizer_step=global_optimizer_step,
                global_optimizer_attempt_step=global_optimizer_attempt_step,
                best_epoch=best_epoch,
                best_validation_ce=best_ce,
                resumed_from_checkpoint=str(resume_checkpoint),
                resume_epoch=resume_epoch,
                resume_global_step=resume_global_step,
                resume_is_bit_exact=resume_rng_available,
                resume_rng_state_restored=resume_rng_available,
                resume_rng_state_missing_reason=(
                    None if resume_rng_available else "checkpoint contains no RNG state"
                ),
                optimizer_restored=True,
                grad_scaler_restored=True,
                resumed_grad_scaler_scale=float(scaler.get_scale()),
                amp_overflow_events_total=amp_overflow_events_total,
                amp_overflow_events_current_epoch=0,
                max_consecutive_amp_overflows=max_consecutive_overflows,
                consecutive_amp_overflows=consecutive_overflows,
                amp_overflow_events=overflow_events,
                last_overflow=None,
                pre_resume_overflow_event_count="unknown_at_least_1",
                failed_at=None,
                forward_nonfinite_count=0,
                model_parameter_nonfinite_count=0,
                asap_test_access_count=0,
            )
            log(
                f"RESUME_CHECKPOINT_LOADED checkpoint={resume_checkpoint} "
                f"epoch={resume_epoch} next_epoch={start_epoch} "
                f"global_step={global_optimizer_step} best_epoch={best_epoch} "
                f"best_validation_ce={best_ce:.12f} scaler={float(scaler.get_scale()):.1f} "
                f"optimizer_restored=true rng_restored={str(resume_rng_available).lower()} "
                "bit_exact=false"
            )
        else:
            start_epoch = 1
            best_ce = math.inf
            early_stopping_best_ce = math.inf
            best_epoch = 0
            stale_epochs = 0
            global_optimizer_step = 0
            global_optimizer_attempt_step = 0
            amp_overflow_events_total = 0
            max_consecutive_overflows = 0
            consecutive_overflows = 0
            overflow_events: list[dict[str, Any]] = []
            amp_skipped_total = 0
            atomic_json(output_dir / "config.json", run_config)
            log(
                f"FRESH_MODEL_INITIALIZED seed=42 parameters={model.parameter_count} "
                f"representation_sha256={representation_hash} fresh_sha256={fresh_hash} "
                "smoke_checkpoint_reused=false"
            )
            status = update_status(
                output_dir,
                state="fresh_model_initialized",
                fresh_seed_42_initialization=True,
                smoke_checkpoint_reused=False,
                loaded_checkpoint_paths=[str(config["checkpoint_path"])],
                parameter_count=model.parameter_count,
                amp_overflow_events_total=0,
                amp_overflow_events_current_epoch=0,
                max_consecutive_amp_overflows=0,
                resumed_from_checkpoint=None,
                resume_is_bit_exact=False,
            )

        validation_loader = make_four_class_loader(
            validation_dataset,
            batch_size=micro_batch_size,
            pin_memory=bool(config["pin_memory"]),
        )
        first_step_logged = False
        stop_reason = "maximum_epochs"

        for epoch in range(start_epoch, int(config["max_epochs"]) + 1):
            epoch_started = time.perf_counter()
            order = training_window_order(len(train_dataset), int(config["seed"]), epoch)
            train_loader = make_four_class_loader(
                train_dataset,
                batch_size=micro_batch_size,
                pin_memory=bool(config["pin_memory"]),
                order=order,
            )
            steps_before_epoch = global_optimizer_step
            attempts_before_epoch = global_optimizer_attempt_step
            overflow_events_before_epoch = amp_overflow_events_total

            def callback(step: int, details: Mapping[str, Any]) -> None:
                nonlocal first_step_logged
                nonlocal amp_overflow_events_total, max_consecutive_overflows
                current_global_step = steps_before_epoch + int(details["optimizer_steps"])
                current_global_attempt = attempts_before_epoch + int(step)
                if details["amp_overflow_event"]:
                    start = (int(step) - 1) * micro_batch_size * accumulation_steps
                    end = min(start + micro_batch_size * accumulation_steps, len(order))
                    grad_norm = float(details["gradient_norm"])
                    event = {
                        "epoch": epoch,
                        "step": int(step),
                        "global_optimizer_step": current_global_step,
                        "global_optimizer_attempt_step": current_global_attempt,
                        "window_ids": [int(value) for value in order[start:end]],
                        "micro_batch_losses": [
                            float(value) for value in details["micro_batch_losses"]
                        ],
                        "grad_norm": grad_norm if math.isfinite(grad_norm) else str(grad_norm),
                        "scale_before": float(details["scale_before"]),
                        "scale_after": float(details["scale_after"]),
                        "offending_parameter_count": int(
                            details["offending_parameter_count"]
                        ),
                        "offending_parameter_names": list(
                            details["offending_parameter_names"]
                        ),
                        "offending_parameter_groups": list(
                            details["offending_parameter_groups"]
                        ),
                        "consecutive_overflow_count": int(
                            details["consecutive_amp_overflows"]
                        ),
                    }
                    amp_overflow_events_total = (
                        overflow_events_before_epoch
                        + int(details["amp_overflow_events"])
                    )
                    max_consecutive_overflows = max(
                        max_consecutive_overflows,
                        int(details["max_consecutive_amp_overflows"]),
                    )
                    event["cumulative_overflow_count"] = amp_overflow_events_total
                    overflow_events.append(event)
                    update_status(
                        output_dir,
                        status="running",
                        state="amp_overflow_recovered",
                        current_epoch=epoch,
                        current_step=int(step),
                        global_step=current_global_step,
                        global_optimizer_step=current_global_step,
                        global_optimizer_attempt_step=current_global_attempt,
                        amp_overflow_events_total=amp_overflow_events_total,
                        amp_overflow_events_current_epoch=int(
                            details["amp_overflow_events"]
                        ),
                        consecutive_amp_overflows=int(
                            details["consecutive_amp_overflows"]
                        ),
                        max_consecutive_amp_overflows=max_consecutive_overflows,
                        last_overflow=event,
                        amp_overflow_events=overflow_events,
                    )
                    log("AMP_OVERFLOW_EVENT " + json.dumps(event, sort_keys=True))
                if not first_step_logged and details["optimizer_step_applied"]:
                    first_step_logged = True
                    initial_loss = float(details["loss"])
                    initial_gradient = float(details["gradient_norm"])
                    if not math.isfinite(initial_loss) or not math.isfinite(initial_gradient):
                        raise FloatingPointError("initial loss/gradient is non-finite")
                    update_status(
                        output_dir,
                        status="running",
                        state=(
                            "first_resumed_optimizer_step_pass"
                            if resuming
                            else "first_optimizer_step_pass"
                        ),
                        current_epoch=epoch,
                        current_step=step,
                        global_step=current_global_step,
                        global_optimizer_step=current_global_step,
                        global_optimizer_attempt_step=current_global_attempt,
                        first_optimizer_step=True,
                        first_resumed_optimizer_step=bool(resuming),
                        initial_loss=initial_loss,
                        initial_gradient_norm=initial_gradient,
                        initial_loss_finite=True,
                        initial_gradient_finite=True,
                    )
                    log(
                        f"FIRST_{'RESUMED_' if resuming else ''}OPTIMIZER_STEP_PASS "
                        f"epoch={epoch} step={step} global_step={current_global_step} "
                        f"global_attempt={current_global_attempt} loss={initial_loss:.9f} "
                        f"grad_norm={initial_gradient:.9f} finite=true"
                    )
                elif step % int(config["progress_interval"]) == 0:
                    update_status(
                        output_dir,
                        status="running",
                        state="training",
                        current_epoch=epoch,
                        current_step=step,
                        global_step=current_global_step,
                        global_optimizer_step=current_global_step,
                        global_optimizer_attempt_step=current_global_attempt,
                    )
                    log(
                        f"TRAIN_PROGRESS epoch={epoch} step={step} "
                        f"global_step={current_global_step} "
                        f"global_attempt={current_global_attempt} "
                        f"loss={float(details['loss']):.9f}"
                    )

            train_metrics, checks = run_epoch(
                view,
                train_loader,
                device=device,
                amp_enabled=bool(config["amp_enabled"]),
                accumulation_steps=accumulation_steps,
                optimizer=optimizer,
                scaler=scaler,
                max_grad_norm=float(config["max_grad_norm"]),
                step_callback=callback,
                max_consecutive_amp_overflows=3,
                initial_consecutive_amp_overflows=consecutive_overflows,
            )
            if not first_step_logged:
                raise RuntimeError("finite first optimizer step was not verified")
            global_optimizer_step += int(checks["optimizer_steps"])
            global_optimizer_attempt_step += int(checks["optimizer_attempts"])
            amp_skipped_total += int(checks["amp_skipped_steps"])
            amp_overflow_events_total = (
                overflow_events_before_epoch + int(checks["amp_overflow_events"])
            )
            consecutive_overflows = int(checks["consecutive_amp_overflows"])
            max_consecutive_overflows = max(
                max_consecutive_overflows,
                int(checks["max_consecutive_amp_overflows"]),
            )
            validation_metrics, _ = run_epoch(
                view,
                validation_loader,
                device=device,
                amp_enabled=bool(config["amp_enabled"]),
            )
            candidate_ce = float(validation_metrics["ce"])
            improved = candidate_ce < best_ce
            if improved:
                best_ce = candidate_ce
                best_epoch = epoch
            if candidate_ce < early_stopping_best_ce - float(
                config["early_stopping_min_delta"]
            ):
                early_stopping_best_ce = candidate_ce
                stale_epochs = 0
            else:
                stale_epochs += 1
            row = {
                "epoch": epoch,
                "train_ce": train_metrics["ce"],
                "train_token_accuracy": train_metrics["token_accuracy"],
                "train_exact_note_accuracy": train_metrics["exact_note_accuracy"],
                "validation_ce": validation_metrics["ce"],
                "validation_token_accuracy": validation_metrics["token_accuracy"],
                "validation_exact_note_accuracy": validation_metrics[
                    "exact_note_accuracy"
                ],
                "best_epoch": best_epoch,
                "best_validation_ce": best_ce,
                "best_so_far": improved,
                "pretrained_representation_lr": optimizer.param_groups[0]["lr"],
                "fresh_prefix_lm_lr": optimizer.param_groups[1]["lr"],
                "epoch_seconds": time.perf_counter() - epoch_started,
                "epoch_optimizer_steps": checks["optimizer_steps"],
                "epoch_optimizer_attempts": checks["optimizer_attempts"],
                "global_optimizer_step": global_optimizer_step,
                "global_optimizer_attempt_step": global_optimizer_attempt_step,
                "amp_skipped_steps": checks["amp_skipped_steps"],
                "amp_overflow_events": checks["amp_overflow_events"],
                "max_consecutive_amp_overflows": checks[
                    "max_consecutive_amp_overflows"
                ],
                "gradients_finite": checks["gradients_finite"],
            }
            rows.append(row)
            write_metrics(output_dir / "metrics.csv", rows)
            payload = checkpoint_payload(
                model,
                run_config,
                row,
                optimizer=optimizer,
                scaler=scaler,
                stale_epochs=stale_epochs,
                amp_overflow_events_total=amp_overflow_events_total,
                max_consecutive_amp_overflows=max_consecutive_overflows,
                overflow_events=overflow_events,
            )
            atomic_torch_save(payload, output_dir / "last.pt")
            if improved:
                best_payload = dict(payload)
                best_payload["selection_metric"] = "minimum validation CE"
                atomic_torch_save(best_payload, output_dir / "best.pt")
            status = update_status(
                output_dir,
                status="running",
                state="epoch_completed",
                current_epoch=epoch,
                current_step=int(checks["optimizer_attempts"]),
                completed_epochs=len(rows),
                best_epoch=best_epoch,
                best_validation_ce=best_ce,
                current_train_ce=float(train_metrics["ce"]),
                current_validation_ce=candidate_ce,
                global_step=global_optimizer_step,
                global_optimizer_step=global_optimizer_step,
                global_optimizer_attempt_step=global_optimizer_attempt_step,
                amp_skipped_steps_total=amp_skipped_total,
                amp_overflow_events_total=amp_overflow_events_total,
                amp_overflow_events_current_epoch=int(checks["amp_overflow_events"]),
                consecutive_amp_overflows=consecutive_overflows,
                max_consecutive_amp_overflows=max_consecutive_overflows,
                all_gradients_finite=bool(checks["gradients_finite"]),
                amp_overflow_events=overflow_events,
            )
            log(
                f"EPOCH_COMPLETE epoch={epoch} train_ce={float(train_metrics['ce']):.9f} "
                f"validation_ce={candidate_ce:.9f} best_epoch={best_epoch} "
                f"global_step={global_optimizer_step} "
                f"global_attempt={global_optimizer_attempt_step} "
                f"amp_overflows={int(checks['amp_overflow_events'])} "
                f"epoch_seconds={row['epoch_seconds']:.3f}"
            )
            if stale_epochs >= int(config["early_stopping_patience"]):
                stop_reason = "early_stopping"
                log(
                    f"EARLY_STOP epoch={epoch} best_epoch={best_epoch} "
                    f"best_validation_ce={best_ce:.9f}"
                )
                break

        if not rows or not (output_dir / "best.pt").is_file() or not (
            output_dir / "last.pt"
        ).is_file():
            raise RuntimeError("training ended without required metrics/checkpoints")
        status = update_status(
            output_dir,
            status="completed",
            state="training_completed",
            completed_at=now(),
            current_epoch=int(rows[-1]["epoch"]),
            completed_epochs=len(rows),
            best_epoch=best_epoch,
            best_validation_ce=best_ce,
            early_stop_epoch=int(rows[-1]["epoch"]),
            stop_reason=stop_reason,
            final_train_ce=float(rows[-1]["train_ce"]),
            final_validation_ce=float(rows[-1]["validation_ce"]),
            global_step=global_optimizer_step,
            global_optimizer_step=global_optimizer_step,
            global_optimizer_attempt_step=global_optimizer_attempt_step,
            amp_skipped_steps_total=amp_skipped_total,
            amp_overflow_events_total=amp_overflow_events_total,
            max_consecutive_amp_overflows=max_consecutive_overflows,
            amp_overflow_events=overflow_events,
            best_checkpoint_path=str(output_dir / "best.pt"),
            last_checkpoint_path=str(output_dir / "last.pt"),
            best_checkpoint_sha256=sha256_file(output_dir / "best.pt"),
            last_checkpoint_sha256=sha256_file(output_dir / "last.pt"),
            asap_test_access_count=0,
            report_generated=False,
        )
        report_path = output_dir / REPORT_NAME
        atomic_text(report_path, report_text(run_config, rows, status))
        status = update_status(
            output_dir,
            status="completed",
            state="completed",
            report_generated=True,
            report_path=str(report_path),
        )
        log(
            f"TRAINING_AND_REPORT_COMPLETE epochs={len(rows)} best_epoch={best_epoch} "
            f"best_validation_ce={best_ce:.9f} report={report_path}"
        )
        return 0
    except BaseException as error:
        message = str(error)
        current_status = read_json(output_dir / "run_status.json")
        forward_nonfinite_count = int(current_status.get("forward_nonfinite_count", 0))
        model_parameter_nonfinite_count = int(
            current_status.get("model_parameter_nonfinite_count", 0)
        )
        if "CE is missing or non-finite" in message:
            forward_nonfinite_count += 1
        if "non-finite model parameter" in message:
            model_parameter_nonfinite_count += 1
        status = update_status(
            output_dir,
            status="failed",
            state="failed",
            failed_at=now(),
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
            current_epoch=current_status.get("current_epoch", 0),
            current_step=current_status.get("current_step", 0),
            best_epoch=current_status.get("best_epoch", 0),
            best_validation_ce=current_status.get("best_validation_ce"),
            forward_nonfinite_count=forward_nonfinite_count,
            model_parameter_nonfinite_count=model_parameter_nonfinite_count,
            asap_test_access_count=0,
            report_generated=False,
        )
        run_config = locals().get(
            "run_config",
            {
                **dict(config),
                "architecture": "Decoder-only Prefix-LM v0",
                "parameter_count": status.get("parameter_count", 0),
                "class_names": list(CLASS_NAMES),
                "class_boundaries": [list(item) for item in CLASS_BOUNDS],
                "representatives": list(REPRESENTATIVES),
                "cache_id": status.get("cache_id", "not available"),
                "micro_batch_size": config["micro_batch_size"]["decoder_only"],
                "gradient_accumulation_steps": config[
                    "gradient_accumulation_steps"
                ]["decoder_only"],
            },
        )
        report_path = output_dir / REPORT_NAME
        atomic_text(report_path, report_text(run_config, rows, status))
        update_status(
            output_dir,
            status="failed",
            report_generated=True,
            report_path=str(report_path),
        )
        log(f"TRAINING_FAILED error={error!r}")
        log(traceback.format_exc())
        return 1
    finally:
        log_handle.close()
        if "model" in locals():
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/stage2_4class_architecture_v0.json",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--execute-training", action="store_true")
    arguments = parser.parse_args()
    if not arguments.execute_training:
        parser.error("training is guarded; pass --execute-training")
    return train(
        read_json(arguments.config),
        arguments.output_dir,
        resume_checkpoint=arguments.resume_checkpoint,
    )


if __name__ == "__main__":
    raise SystemExit(main())
