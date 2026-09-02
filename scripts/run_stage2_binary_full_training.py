#!/usr/bin/env python3
"""Full binary Stage 2 runner. This script is prepared but not launched by setup."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.stage2_binary.full_training import (  # noqa: E402
    PedalMetricEarlyStopping,
    SharedBinaryWindowDataset,
    evaluate_stage1_cache_distribution,
    make_binary_loader,
    order_sha256,
    run_binary_epoch,
    training_window_order,
)
from src.stage2_binary.model import (  # noqa: E402
    IndependentBinaryPedalModel,
    JointBinaryPedalModel,
)
from src.stage2_binary.training import build_binary_optimizer  # noqa: E402
from src.stage2_encoder_only.train import (  # noqa: E402
    atomic_torch_save,
    capture_rng_states,
    get_gpu_identity,
    restore_rng_states,
)
from src.stage2_encoder_only.training import set_deterministic_seed  # noqa: E402


MODEL_CLASSES = {
    "independent_4x2": IndependentBinaryPedalModel,
    "joint_16": JointBinaryPedalModel,
}
METRIC_COLUMNS = (
    "epoch",
    "train_loss",
    "validation_loss",
    "validation_binary_accuracy",
    "validation_exact_pattern_accuracy",
    "validation_pedal_js_distance",
    "validation_pedal_intersection",
    "epoch_time_seconds",
    "global_optimizer_step",
    "encoder_learning_rate",
    "head_learning_rate",
    "peak_gpu_memory_bytes",
)


def _load_config(path: str) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    return json.loads(config_path.read_text(encoding="utf-8"))


def _parameter_hash(parameters: Any) -> str:
    digest = hashlib.sha256()
    with torch.no_grad():
        for parameter in parameters:
            value = parameter.detach().cpu().contiguous()
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    config: dict[str, Any],
    epoch: int,
    global_step: int,
    stopping: PedalMetricEarlyStopping,
) -> dict[str, Any]:
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "grad_scaler_state": scaler.state_dict(),
        "configuration": config,
        "completed_epoch": epoch,
        "global_optimizer_step": global_step,
        "best_epoch": stopping.best_epoch,
        "best_validation_js_distance": stopping.best_js_distance,
        "best_validation_intersection": stopping.best_intersection,
        "early_stopping_counter": stopping.counter,
    }
    payload.update(capture_rng_states())
    return payload


def train(
    config: dict[str, Any],
    architecture: str,
    resume: str | None = None,
) -> None:
    if architecture not in MODEL_CLASSES:
        raise ValueError(f"unknown architecture: {architecture}")
    if int(config["gradient_accumulation_steps"]) != 1:
        raise ValueError("this fixed v0 setup requires gradient_accumulation_steps=1")
    output_dir = Path(config["output_root"]) / architecture
    resume_path = Path(resume).resolve() if resume is not None else None
    if resume_path is None:
        if output_dir.exists():
            raise FileExistsError(f"refusing to overwrite run directory: {output_dir}")
        output_dir.mkdir(parents=True)
    elif not output_dir.is_dir() or not resume_path.is_file():
        raise FileNotFoundError("resume run directory or checkpoint is missing")
    log_handle = (output_dir / "train.log").open(
        "a" if resume_path is not None else "w",
        encoding="utf-8",
    )

    def log(message: str) -> None:
        print(message, flush=True)
        print(message, file=log_handle, flush=True)

    status = {
        "architecture": architecture,
        "status": "running",
        "full_training_started": True,
        "asap_test_access_count": 0,
    }
    if resume_path is not None:
        status["resumed_from"] = str(resume_path)
        status["previous_status"] = json.loads(
            (output_dir / "run_status.json").read_text(encoding="utf-8")
        )
    _write_json(output_dir / "run_status.json", status)
    try:
        set_deterministic_seed(int(config["seed"]))
        gpu = get_gpu_identity()
        if gpu["uuid"] != config["expected_gpu_uuid"]:
            raise RuntimeError(f"unexpected GPU: {gpu}")
        device = torch.device("cuda:0")
        train_dataset = SharedBinaryWindowDataset(config["cache_root"], "train")
        validation_dataset = SharedBinaryWindowDataset(config["cache_root"], "validation")
        if train_dataset.cache_id != validation_dataset.cache_id:
            raise AssertionError("train/validation cache IDs differ")
        if resume_path is None:
            run_config = dict(config)
            run_config.update(
                architecture=architecture,
                cache_id=train_dataset.cache_id,
                output_dir=str(output_dir),
                full_training_started=True,
                encoder_initialization="official Pianist Transformer pretrained encoder",
                epoch_window_order_sha256={
                    str(epoch): order_sha256(
                        training_window_order(
                            len(train_dataset), int(config["seed"]), epoch
                        )
                    )
                    for epoch in range(1, int(config["max_epochs"]) + 1)
                },
            )
        else:
            run_config = json.loads(
                (output_dir / "config.json").read_text(encoding="utf-8")
            )
            if run_config["architecture"] != architecture:
                raise RuntimeError("resume architecture does not match run config")
            if run_config["cache_id"] != train_dataset.cache_id:
                raise RuntimeError("resume cache ID does not match")
        model = MODEL_CLASSES[architecture].from_pretrained(
            config["checkpoint_path"],
            torch_dtype=torch.float32,
            attn_implementation="eager",
        )
        if resume_path is None:
            run_config["initial_encoder_parameter_sha256"] = _parameter_hash(
                model.encoder.parameters()
            )
            run_config["parameter_count"] = model.parameter_count
            run_config["trainable_parameter_count"] = model.trainable_parameter_count
            _write_json(output_dir / "config.json", run_config)
        model.to(device)
        optimizer = build_binary_optimizer(
            model,
            encoder_lr=float(config["encoder_lr"]),
            head_lr=float(config["head_lr"]),
            weight_decay=float(config["weight_decay"]),
        )
        scaler = torch.amp.GradScaler(
            "cuda", enabled=bool(config["amp_enabled"]), init_scale=float(config["amp_init_scale"])
        )
        stopping = PedalMetricEarlyStopping(
            patience=int(config["early_stopping_patience"]),
            tie_tolerance=float(config["js_tie_tolerance"]),
        )
        start_epoch = 1
        global_step = 0
        amp_skipped_steps_total = 0
        if resume_path is not None:
            checkpoint = torch.load(
                resume_path,
                map_location="cpu",
                weights_only=False,
            )
            model.load_state_dict(checkpoint["model_state"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scaler.load_state_dict(checkpoint["grad_scaler_state"])
            stopping.best_js_distance = float(
                checkpoint["best_validation_js_distance"]
            )
            stopping.best_intersection = float(
                checkpoint["best_validation_intersection"]
            )
            stopping.best_epoch = int(checkpoint["best_epoch"])
            stopping.counter = int(checkpoint["early_stopping_counter"])
            start_epoch = int(checkpoint["completed_epoch"]) + 1
            global_step = int(checkpoint["global_optimizer_step"])
            restore_rng_states(checkpoint)
            log(
                f"resumed checkpoint={resume_path} start_epoch={start_epoch} "
                f"global_step={global_step} dynamic_amp_scale={scaler.get_scale()}"
            )
            del checkpoint
        validation_loader = make_binary_loader(
            validation_dataset,
            batch_size=int(config["batch_size"]),
            pin_memory=bool(config["pin_memory"]),
        )
        human = json.loads(
            Path(config["original_pt_validation_metrics"]).read_text(encoding="utf-8")
        )["human_histogram"]
        metrics_path = output_dir / "metrics.csv"
        metrics_mode = "a" if resume_path is not None else "w"
        with metrics_path.open(
            metrics_mode, newline="", encoding="utf-8"
        ) as metrics_handle:
            writer = csv.DictWriter(metrics_handle, fieldnames=METRIC_COLUMNS)
            if metrics_mode == "w":
                writer.writeheader()
            for epoch in range(start_epoch, int(config["max_epochs"]) + 1):
                epoch_started = time.perf_counter()
                torch.cuda.reset_peak_memory_stats(device)
                order = training_window_order(len(train_dataset), int(config["seed"]), epoch)
                train_loader = make_binary_loader(
                    train_dataset,
                    batch_size=int(config["batch_size"]),
                    pin_memory=bool(config["pin_memory"]),
                    order=order,
                )
                train_metrics, train_checks = run_binary_epoch(
                    model,
                    train_loader,
                    architecture=architecture,
                    device=device,
                    amp_enabled=bool(config["amp_enabled"]),
                    optimizer=optimizer,
                    scaler=scaler,
                    max_grad_norm=float(config["max_grad_norm"]),
                )
                global_step += int(train_checks["optimizer_steps"])
                amp_skipped_steps_total += int(
                    train_checks["amp_skipped_steps"]
                )
                validation_metrics, _ = run_binary_epoch(
                    model,
                    validation_loader,
                    architecture=architecture,
                    device=device,
                    amp_enabled=bool(config["amp_enabled"]),
                )
                distribution = evaluate_stage1_cache_distribution(
                    model,
                    architecture=architecture,
                    stage1_manifest_csv=config["stage1_cache_manifest"],
                    human_histogram=human,
                    device=device,
                )
                improved, should_stop = stopping.update(
                    distribution["js_distance"], distribution["intersection"], epoch
                )
                row = {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "validation_loss": validation_metrics["loss"],
                    "validation_binary_accuracy": validation_metrics["binary_accuracy"],
                    "validation_exact_pattern_accuracy": validation_metrics["exact_pattern_accuracy"],
                    "validation_pedal_js_distance": distribution["js_distance"],
                    "validation_pedal_intersection": distribution["intersection"],
                    "epoch_time_seconds": time.perf_counter() - epoch_started,
                    "global_optimizer_step": global_step,
                    "encoder_learning_rate": optimizer.param_groups[0]["lr"],
                    "head_learning_rate": optimizer.param_groups[1]["lr"],
                    "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
                }
                if not all(math.isfinite(float(row[key])) for key in METRIC_COLUMNS if key != "epoch"):
                    raise FloatingPointError("non-finite epoch metric")
                writer.writerow(row)
                metrics_handle.flush()
                os.fsync(metrics_handle.fileno())
                checkpoint = _checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    config=run_config,
                    epoch=epoch,
                    global_step=global_step,
                    stopping=stopping,
                )
                atomic_torch_save(checkpoint, output_dir / "last.pt")
                if improved:
                    atomic_torch_save(
                        {
                            "model_state": model.state_dict(),
                            "configuration": run_config,
                            "best_epoch": epoch,
                            "best_validation_js_distance": distribution["js_distance"],
                            "best_validation_intersection": distribution["intersection"],
                        },
                        output_dir / "best.pt",
                    )
                log(json.dumps(row, sort_keys=True))
                if should_stop:
                    log(f"early stopping at epoch {epoch}; best epoch {stopping.best_epoch}")
                    break
        status.update(
            status="completed",
            completed_epochs=epoch,
            best_epoch=stopping.best_epoch,
            best_validation_js_distance=stopping.best_js_distance,
            best_validation_intersection=stopping.best_intersection,
            amp_skipped_steps_after_resume=amp_skipped_steps_total,
        )
        _write_json(output_dir / "run_status.json", status)
    except BaseException as error:
        status.update(status="failed", error=repr(error))
        _write_json(output_dir / "run_status.json", status)
        raise
    finally:
        log_handle.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage2_binary_full_training_v0.json")
    parser.add_argument("--architecture", required=True, choices=tuple(MODEL_CLASSES))
    parser.add_argument("--resume")
    parser.add_argument(
        "--confirm-full-training",
        action="store_true",
        help="Required guard; this setup task deliberately does not pass it.",
    )
    args = parser.parse_args()
    if not args.confirm_full_training:
        raise SystemExit("refusing to start full training without --confirm-full-training")
    train(_load_config(args.config), args.architecture, resume=args.resume)


if __name__ == "__main__":
    main()
