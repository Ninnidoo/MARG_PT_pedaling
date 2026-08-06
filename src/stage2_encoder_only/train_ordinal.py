"""Training entry point for CE plus squared-CDF ordinal Stage 2 models."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
from torch.utils.data import DataLoader

from .dataset import Stage2PedalDataset, stage2_pedal_collate_fn
from .model import Stage2PedalEncoderModel
from .ordinal_loss import squared_cdf_ordinal_loss
from .train import (
    EarlyStopping,
    GracefulStop,
    PINNED_PT_COMMIT,
    atomic_torch_save,
    build_best_checkpoint,
    build_last_checkpoint,
    deterministic_train_order,
    get_gpu_identity,
    make_loader,
    prepare_output_directory,
    restore_training_state,
    write_configuration,
)
from .training import (
    _gradient_norm,
    _parameters_with_gradients,
    _validate_finite_gradients,
    build_optimizer,
    calculate_pedal_metrics,
    create_grad_scaler,
    move_batch_to_device,
    set_deterministic_seed,
)


LOSS_CONFIGURATION = {
    "name": "cross_entropy_plus_squared_cdf_cramer_ordinal",
    "classes": 128,
    "thresholds": 127,
    "ignore_index": -100,
    "cdf_dtype": "float32",
    "reduction": "mean_over_valid_targets_and_thresholds",
}
ORDINAL_METRIC_NAMES = [
    "total_loss",
    "ce_loss",
    "ordinal_loss",
    "pedal_token_accuracy",
    "exact_note_accuracy",
    "pedal1_accuracy",
    "pedal2_accuracy",
    "pedal3_accuracy",
    "pedal4_accuracy",
    "pedal_value_mae",
    "valid_target_count",
    "valid_note_count",
    "encoder_gradient_norm",
    "head_gradient_norm",
    "total_gradient_norm",
    "encoder_learning_rate",
    "head_learning_rate",
    "amp_scale",
]


def parameter_hash(model: torch.nn.Module) -> str:
    """Stable hash of all initial named parameters before an optimizer update."""

    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode("utf-8"))
        value = parameter.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _forward_without_builtin_ce(
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    amp_enabled: bool,
) -> Any:
    device_type = batch["input_ids"].device.type
    use_amp = amp_enabled and device_type == "cuda"
    with torch.amp.autocast(
        device_type=device_type,
        dtype=torch.float16 if device_type == "cuda" else torch.bfloat16,
        enabled=use_amp,
    ):
        return model(
            input_ids=batch["input_ids"],
            token_attention_mask=batch["token_attention_mask"],
            pedal_targets=None,
            note_mask=batch.get("note_mask"),
        )


def _learning_rates(optimizer: torch.optim.Optimizer) -> tuple[float, float]:
    rates = {group.get("group_name"): float(group["lr"]) for group in optimizer.param_groups}
    return float(rates.get("encoder", 0.0)), float(rates.get("heads", 0.0))


def ordinal_train_step(
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
    lambda_ordinal: float,
    scaler: torch.amp.GradScaler | None = None,
    amp_enabled: bool = False,
    max_grad_norm: float = 1.0,
) -> dict[str, float | int]:
    """Run one checked update and report CE, ordinal, and total separately."""

    if max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = _forward_without_builtin_ce(model, batch, amp_enabled)
    losses = squared_cdf_ordinal_loss(
        output.logits, batch["pedal_targets"], lambda_ordinal
    )
    if not all(bool(torch.isfinite(value)) for value in losses):
        raise FloatingPointError("non-finite ordinal training loss")

    use_amp = amp_enabled and batch["input_ids"].device.type == "cuda"
    if scaler is None:
        scaler = create_grad_scaler(use_amp, batch["input_ids"].device.type)
    if use_amp:
        scaler.scale(losses.total).backward()
        scaler.unscale_(optimizer)
    else:
        losses.total.backward()

    encoder_parameters = list(model.encoder.parameters())
    head_parameters = list(model.classification_heads.parameters())
    trainable_encoder = [p for p in encoder_parameters if p.requires_grad]
    if trainable_encoder and not _parameters_with_gradients(trainable_encoder):
        raise RuntimeError("trainable encoder received no gradients")
    for slot, head in enumerate(model.classification_heads):
        if not _parameters_with_gradients(head.parameters()):
            raise RuntimeError(f"prediction head {slot + 1} received no gradients")
    trainable = [p for p in model.parameters() if p.requires_grad]
    _validate_finite_gradients(trainable)
    encoder_gradient_norm = _gradient_norm(encoder_parameters)
    head_gradient_norm = _gradient_norm(head_parameters)
    total_gradient_norm = float(
        torch.nn.utils.clip_grad_norm_(
            trainable, max_grad_norm, error_if_nonfinite=True
        ).item()
    )
    optimizer.step() if not use_amp else scaler.step(optimizer)
    if use_amp:
        scaler.update()
    encoder_lr, head_lr = _learning_rates(optimizer)
    metrics = calculate_pedal_metrics(output.logits.detach(), batch["pedal_targets"])
    metrics.pop("loss")
    metrics.update(
        total_loss=float(losses.total.detach()),
        ce_loss=float(losses.ce.detach()),
        ordinal_loss=float(losses.ordinal.detach()),
        encoder_gradient_norm=encoder_gradient_norm,
        head_gradient_norm=head_gradient_norm,
        total_gradient_norm=total_gradient_norm,
        encoder_learning_rate=encoder_lr,
        head_learning_rate=head_lr,
        amp_scale=float(scaler.get_scale()) if use_amp else 1.0,
    )
    return metrics


@torch.no_grad()
def ordinal_evaluation_step(
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    lambda_ordinal: float,
    amp_enabled: bool = False,
) -> dict[str, float | int]:
    model.eval()
    output = _forward_without_builtin_ce(model, batch, amp_enabled)
    losses = squared_cdf_ordinal_loss(
        output.logits, batch["pedal_targets"], lambda_ordinal
    )
    metrics = calculate_pedal_metrics(output.logits, batch["pedal_targets"])
    metrics.pop("loss")
    metrics.update(
        total_loss=float(losses.total),
        ce_loss=float(losses.ce),
        ordinal_loss=float(losses.ordinal),
        encoder_gradient_norm=0.0,
        head_gradient_norm=0.0,
        total_gradient_norm=0.0,
        encoder_learning_rate=0.0,
        head_learning_rate=0.0,
        amp_scale=1.0,
    )
    if not all(
        math.isfinite(float(metrics[name]))
        for name in ORDINAL_METRIC_NAMES
    ):
        raise FloatingPointError("non-finite ordinal evaluation metric")
    return metrics


class OrdinalMetricAccumulator:
    def __init__(self) -> None:
        self.targets = 0
        self.notes = 0
        self.weighted = {name: 0.0 for name in ORDINAL_METRIC_NAMES}
        self.last_amp_scale = 1.0

    def update(self, metrics: Mapping[str, float | int]) -> None:
        targets = int(metrics["valid_target_count"])
        notes = int(metrics["valid_note_count"])
        if targets <= 0 or targets != 4 * notes:
            raise ValueError("expected four valid pedal targets per valid note")
        self.targets += targets
        self.notes += notes
        target_weighted = {
            "total_loss", "ce_loss", "ordinal_loss", "pedal_token_accuracy",
            "pedal_value_mae", "encoder_gradient_norm", "head_gradient_norm",
            "total_gradient_norm", "encoder_learning_rate", "head_learning_rate",
        }
        note_weighted = {
            "exact_note_accuracy", "pedal1_accuracy", "pedal2_accuracy",
            "pedal3_accuracy", "pedal4_accuracy",
        }
        for name in target_weighted:
            value = float(metrics[name])
            if not math.isfinite(value):
                raise FloatingPointError(f"non-finite metric {name}")
            self.weighted[name] += value * targets
        for name in note_weighted:
            value = float(metrics[name])
            if not math.isfinite(value):
                raise FloatingPointError(f"non-finite metric {name}")
            self.weighted[name] += value * notes
        self.last_amp_scale = float(metrics["amp_scale"])

    def compute(self) -> dict[str, float | int]:
        if not self.targets:
            raise ValueError("empty epoch")
        result: dict[str, float | int] = {}
        for name in (
            "total_loss", "ce_loss", "ordinal_loss", "pedal_token_accuracy",
            "pedal_value_mae", "encoder_gradient_norm", "head_gradient_norm",
            "total_gradient_norm", "encoder_learning_rate", "head_learning_rate",
        ):
            result[name] = self.weighted[name] / self.targets
        for name in (
            "exact_note_accuracy", "pedal1_accuracy", "pedal2_accuracy",
            "pedal3_accuracy", "pedal4_accuracy",
        ):
            result[name] = self.weighted[name] / self.notes
        result.update(
            valid_target_count=self.targets,
            valid_note_count=self.notes,
            amp_scale=self.last_amp_scale,
        )
        return result


def run_ordinal_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    training: bool,
    lambda_ordinal: float,
    amp_enabled: bool,
    stop: GracefulStop,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    max_grad_norm: float = 1.0,
    global_step: int = 0,
) -> tuple[dict[str, float | int] | None, int, bool]:
    accumulator = OrdinalMetricAccumulator()
    for batch_index, cpu_batch in enumerate(loader, 1):
        batch = move_batch_to_device(cpu_batch, device)
        if training:
            if optimizer is None or scaler is None:
                raise ValueError("training requires optimizer and scaler")
            metrics = ordinal_train_step(
                model, batch, optimizer, lambda_ordinal, scaler,
                amp_enabled, max_grad_norm,
            )
            global_step += 1
        else:
            metrics = ordinal_evaluation_step(
                model, batch, lambda_ordinal, amp_enabled
            )
        accumulator.update(metrics)
        if batch_index % 100 == 0:
            print(
                f"{'train' if training else 'validation'} batch "
                f"{batch_index}/{len(loader)} total={metrics['total_loss']:.6f} "
                f"ce={metrics['ce_loss']:.6f} ordinal={metrics['ordinal_loss']:.6f} "
                f"global_step={global_step}", flush=True,
            )
        if stop.requested:
            return None, global_step, False
    return accumulator.compute(), global_step, True


def _metric_row(
    epoch: int,
    global_step: int,
    train_metrics: Mapping[str, Any],
    validation_metrics: Mapping[str, Any],
    seconds: float,
    peak_memory: int,
) -> dict[str, Any]:
    row = {
        "epoch": epoch,
        "global_optimizer_step": global_step,
        "epoch_seconds": seconds,
        "peak_gpu_memory_bytes": peak_memory,
    }
    for prefix, metrics in (("train", train_metrics), ("validation", validation_metrics)):
        for name in ORDINAL_METRIC_NAMES:
            row[f"{prefix}_{name}"] = metrics[name]
    return row


def train(
    configuration: dict[str, Any],
    epoch_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Train one lambda independently and return its completed summary."""

    output_dir = Path(configuration["output_dir"]).resolve()
    resume = Path(configuration["resume"]).resolve() if configuration.get("resume") else None
    prepare_output_directory(output_dir, resume, True)
    for key in ("asap_root", "split_csv", "checkpoint_path"):
        configuration[key] = str(Path(configuration[key]).resolve())
    configuration["output_dir"] = str(output_dir)
    configuration["pianist_transformer_commit"] = PINNED_PT_COMMIT
    configuration["loss_configuration"] = dict(LOSS_CONFIGURATION)

    set_deterministic_seed(int(configuration["seed"]))
    stop = GracefulStop()
    stop.install()
    gpu = get_gpu_identity()
    if gpu["uuid"] != configuration["expected_gpu_uuid"]:
        raise RuntimeError(f"GPU UUID mismatch: expected {configuration['expected_gpu_uuid']}, got {gpu['uuid']}")
    print("ORDINAL RUN START", json.dumps(configuration, sort_keys=True), flush=True)
    print(f"GPU uuid={gpu['uuid']} model={gpu['name']}", flush=True)

    preload_start = time.perf_counter()
    train_dataset = Stage2PedalDataset(
        configuration["asap_root"], configuration["split_csv"], "train",
        window_notes=512, stride_notes=256, cache_mode="preload",
    )
    validation_dataset = Stage2PedalDataset(
        configuration["asap_root"], configuration["split_csv"], "validation",
        window_notes=512, stride_notes=256, cache_mode="preload",
    )
    print(
        f"preload seconds={time.perf_counter()-preload_start:.3f} "
        f"train_performances={train_dataset.performance_count} "
        f"train_windows={train_dataset.window_count} "
        f"validation_performances={validation_dataset.performance_count} "
        f"validation_windows={validation_dataset.window_count}", flush=True,
    )

    device = torch.device("cuda:0")
    model = Stage2PedalEncoderModel.from_pretrained(
        configuration["checkpoint_path"], freeze_encoder=False,
        torch_dtype=torch.float32, attn_implementation="eager",
    ).to(device)
    initial_hash = parameter_hash(model)
    configuration["initial_parameter_sha256"] = initial_hash
    config_path = output_dir / "config.json"
    if resume is None:
        write_configuration(config_path, configuration)
    else:
        saved_config = json.loads(config_path.read_text(encoding="utf-8"))
        for key in (
            "lambda_ordinal", "seed", "batch_size", "encoder_lr", "head_lr",
            "weight_decay", "max_grad_norm", "initial_parameter_sha256",
        ):
            if saved_config.get(key) != configuration.get(key):
                raise ValueError(f"incompatible resume configuration: {key}")
        configuration = saved_config
    print(f"initial_parameter_sha256={initial_hash}", flush=True)

    optimizer = build_optimizer(
        model, float(configuration["encoder_lr"]), float(configuration["head_lr"]),
        float(configuration["weight_decay"]),
    )
    scaler = create_grad_scaler(
        bool(configuration["amp_enabled"]), "cuda", float(configuration["amp_init_scale"])
    )
    early = EarlyStopping(
        int(configuration["early_stopping_patience"]),
        float(configuration["early_stopping_min_delta"]),
    )
    start_epoch, global_step = 1, 0
    if resume is not None:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        start_epoch, global_step = restore_training_state(checkpoint, model, optimizer, scaler, early)
        print(f"resumed checkpoint={resume} start_epoch={start_epoch}", flush=True)

    validation_loader = make_loader(
        validation_dataset, int(configuration["batch_size"]), bool(configuration["pin_memory"])
    )
    fields = ["epoch", "global_optimizer_step", "epoch_seconds", "peak_gpu_memory_bytes"] + [
        f"{prefix}_{name}" for prefix in ("train", "validation") for name in ORDINAL_METRIC_NAMES
    ]
    metrics_path = output_dir / "metrics.csv"
    append = resume is not None and metrics_path.exists()
    metrics_handle = metrics_path.open("a" if append else "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(metrics_handle, fieldnames=fields)
    if not append:
        writer.writeheader()
        metrics_handle.flush()
    run_start = time.perf_counter()
    completed_epoch = start_epoch - 1
    stop_reason = "max_epochs"
    try:
        for epoch in range(start_epoch, int(configuration["max_epochs"]) + 1):
            torch.cuda.reset_peak_memory_stats(device)
            epoch_start = time.perf_counter()
            order = deterministic_train_order(len(train_dataset), int(configuration["seed"]), epoch)
            train_loader = make_loader(
                train_dataset, int(configuration["batch_size"]),
                bool(configuration["pin_memory"]), order,
            )
            print(f"epoch {epoch} start train_batches={len(train_loader)} validation_batches={len(validation_loader)}", flush=True)
            train_metrics, global_step, complete = run_ordinal_epoch(
                model, train_loader, device, True, float(configuration["lambda_ordinal"]),
                bool(configuration["amp_enabled"]), stop, optimizer, scaler,
                float(configuration["max_grad_norm"]), global_step,
            )
            if not complete:
                stop_reason = "signal"
                break
            validation_metrics, global_step, complete = run_ordinal_epoch(
                model, validation_loader, device, False, float(configuration["lambda_ordinal"]),
                bool(configuration["amp_enabled"]), stop, global_step=global_step,
            )
            if not complete:
                stop_reason = "signal"
                break
            completed_epoch = epoch
            improved, should_stop = early.update(float(validation_metrics["total_loss"]), epoch)
            peak_memory = int(torch.cuda.max_memory_allocated(device))
            row = _metric_row(
                epoch, global_step, train_metrics, validation_metrics,
                time.perf_counter() - epoch_start, peak_memory,
            )
            writer.writerow(row)
            metrics_handle.flush()
            os.fsync(metrics_handle.fileno())
            atomic_torch_save(
                build_last_checkpoint(model, optimizer, scaler, completed_epoch, global_step, early, configuration),
                output_dir / "last.pt",
            )
            if improved:
                atomic_torch_save(
                    build_best_checkpoint(model, configuration, early.best_epoch, early.best_loss),
                    output_dir / "best.pt",
                )
            callback_row = dict(row)
            callback_row.update(best_epoch=early.best_epoch, best_validation_loss=early.best_loss)
            if epoch_callback:
                epoch_callback(callback_row)
            print(
                f"epoch {epoch} complete train={train_metrics} validation={validation_metrics} "
                f"seconds={row['epoch_seconds']:.3f} peak_gpu_bytes={peak_memory} "
                f"improved={improved} early_counter={early.counter}", flush=True,
            )
            if should_stop:
                stop_reason = "early_stopping"
                break
            if stop.requested:
                stop_reason = "signal"
                break
    finally:
        if completed_epoch >= 0 and stop_reason == "signal":
            atomic_torch_save(
                build_last_checkpoint(model, optimizer, scaler, completed_epoch, global_step, early, configuration),
                output_dir / "last.pt",
            )
        metrics_handle.close()
    if stop_reason == "signal":
        raise KeyboardInterrupt("ordinal training interrupted safely")
    result = {
        "completed_epoch": completed_epoch,
        "global_optimizer_step": global_step,
        "best_epoch": early.best_epoch,
        "best_validation_loss": early.best_loss,
        "initial_parameter_sha256": initial_hash,
        "runtime_seconds": time.perf_counter() - run_start,
        "stop_reason": stop_reason,
    }
    print("ORDINAL RUN COMPLETE", json.dumps(result, sort_keys=True), flush=True)
    return result


def default_configuration(output_dir: str, lambda_ordinal: float) -> dict[str, Any]:
    return {
        "asap_root": "/workspace/public/ASAP/asap-dataset-v1.1",
        "split_csv": "/workspace/project/analysis/stage2_encoder_only_v0/asap_split.csv",
        "checkpoint_path": "/workspace/project/checkpoints/pianist_transformer",
        "output_dir": output_dir,
        "resume": None,
        "batch_size": 16,
        "max_epochs": 20,
        "seed": 20260710,
        "early_stopping_patience": 4,
        "early_stopping_min_delta": 1e-4,
        "encoder_lr": 1e-5,
        "head_lr": 1e-4,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "amp_init_scale": 1024.0,
        "amp_enabled": True,
        "freeze_encoder": False,
        "pin_memory": True,
        "num_workers": 0,
        "allow_existing_log_dir": True,
        "expected_gpu_uuid": "GPU-6982dbee-fbaf-f359-d7ef-a22d0e83400b",
        "lambda_ordinal": lambda_ordinal,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lambda-ordinal", required=True, type=float)
    parser.add_argument("--resume")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = default_configuration(args.output_dir, args.lambda_ordinal)
    config["resume"] = args.resume
    train(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
