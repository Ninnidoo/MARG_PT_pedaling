"""Controlled weighted training for the Stage 2 PT encoder-decoder model."""

from __future__ import annotations

import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.stage2_encoder_only.dataset import (
    PEDAL_SLOTS,
    Stage2PedalDataset,
)
from src.stage2_encoder_only.five_class import (
    CLASS_BOUNDS,
    CLASS_NAMES,
    NUM_CLASSES,
    REPRESENTATIVES,
    five_class_collate_fn,
)
from src.stage2_encoder_only.train import (
    EarlyStopping,
    atomic_torch_save,
    build_best_checkpoint,
    build_last_checkpoint,
    deterministic_train_order,
    get_gpu_identity,
)
from src.stage2_encoder_only.train_five_class import (
    DEFAULT_ASAP_ROOT,
    DEFAULT_CHECKPOINT,
    DEFAULT_SPLIT_CSV,
    EXPECTED_GPU_UUID,
    PEDAL_RICH_PERFORMANCE,
    _atomic_json,
    _make_window_sample,
    parameter_hash,
    train_global_class_weight_payload,
)
from src.stage2_encoder_only.training import build_optimizer, create_grad_scaler, set_deterministic_seed

from .model import EncoderDecoderFiveClassModel, ScheduledSamplingHistory


Callback = Callable[[dict[str, Any]], None]
DEFAULT_OUTPUT = "/workspace/project/analysis/stage2_encoder_decoder_5class_weighted_v0"
TEACHER_FORCING_SCHEDULE = {
    1: 1.00,
    2: 1.00,
    3: 0.90,
    4: 0.80,
    5: 0.70,
}


def teacher_forcing_probability(epoch: int) -> float:
    if int(epoch) < 1:
        raise ValueError("epoch must be positive")
    return TEACHER_FORCING_SCHEDULE.get(int(epoch), 0.60)


def default_configuration(output_dir: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    return {
        "output_dir": str(output_dir),
        "asap_root": DEFAULT_ASAP_ROOT,
        "split_csv": DEFAULT_SPLIT_CSV,
        "checkpoint_path": DEFAULT_CHECKPOINT,
        "seed": 42,
        "decoder_init_seed": 42,
        "effective_batch_size": 16,
        "micro_batch_size": None,
        "gradient_accumulation_steps": None,
        "max_epochs": 20,
        "early_stopping_patience": 4,
        "early_stopping_min_delta": 1e-4,
        "early_stopping_enabled": True,
        "encoder_lr": 1e-5,
        "decoder_lr": 1e-4,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "amp_enabled": True,
        "amp_init_scale": 1024.0,
        "window_notes": 512,
        "stride_notes": 256,
        "num_workers": 0,
        "pin_memory": True,
        "progress_interval": 100,
        "expected_gpu_uuid": EXPECTED_GPU_UUID,
        "expected_train_performances": 892,
        "expected_validation_performances": 71,
        "expected_train_pieces": 180,
        "expected_validation_pieces": 19,
        "class_names": list(CLASS_NAMES),
        "class_boundaries": [list(item) for item in CLASS_BOUNDS],
        "representatives": REPRESENTATIVES.tolist(),
        "architecture": (
            "official pretrained unfrozen PT encoder (10x768) + fresh PT-native "
            "2-layer causal decoder (hidden 768, FFN 3072, head dim 128) + "
            "Pedal1-4 slot embeddings + shared Linear(768,5)"
        ),
        "loss_configuration": {
            "name": "class_weighted_cross_entropy_inverse_sqrt_train_global",
            "class_weighting": "global_train_pedal1_4_inverse_sqrt_probability",
            "ignore_index": -100,
            "label_smoothing": 0.0,
        },
        "model_selection": "teacher_forced_validation_weighted_ce",
        "pipeline_splits": ["train", "validation"],
        "test_split_passed_to_pipeline": False,
        "scheduled_sampling_enabled": False,
        "scheduled_sampling_seed": 42,
    }


def _notify(callback: Callback | None, **payload: Any) -> None:
    if callback is not None:
        callback(dict(payload))


def _make_model(configuration: Mapping[str, Any]) -> EncoderDecoderFiveClassModel:
    model = EncoderDecoderFiveClassModel.from_pretrained_encoder(
        configuration["checkpoint_path"],
        decoder_init_seed=int(configuration["decoder_init_seed"]),
        freeze_encoder=False,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    weights = configuration["loss_configuration"].get("class_weights")
    if weights is None:
        raise RuntimeError("train-only class weights are missing")
    model.set_class_weights(weights)
    if len(model.encoder.layers) != 10 or len(model.decoder.layers) != 2:
        raise RuntimeError("registered 10-layer encoder / 2-layer decoder architecture mismatch")
    if model.hidden_size != 768 or model.output_head.in_features != 768:
        raise RuntimeError("registered hidden size mismatch")
    return model


def preload_datasets(
    configuration: dict[str, Any], callback: Callback | None = None
) -> tuple[Stage2PedalDataset, Stage2PedalDataset]:
    datasets = []
    for split in ("train", "validation"):
        _notify(callback, stage="preloading", split=split)
        started = time.perf_counter()
        dataset = Stage2PedalDataset(
            configuration["asap_root"],
            configuration["split_csv"],
            split,
            window_notes=int(configuration["window_notes"]),
            stride_notes=int(configuration["stride_notes"]),
            cache_mode="preload",
        )
        seconds = time.perf_counter() - started
        configuration[f"{split}_preload_seconds"] = seconds
        configuration[f"{split}_performance_count"] = dataset.performance_count
        configuration[f"{split}_piece_count"] = len(
            {row["piece_id"] for row in dataset.performances}
        )
        configuration[f"{split}_window_count"] = dataset.window_count
        if any(row["split"] != split for row in dataset.performances):
            raise RuntimeError(f"non-{split} row reached the pipeline")
        _notify(callback, stage="preloading", split=split, completed=True, seconds=seconds)
        datasets.append(dataset)
    expected = {
        "train_performance_count": int(configuration["expected_train_performances"]),
        "validation_performance_count": int(configuration["expected_validation_performances"]),
        "train_piece_count": int(configuration["expected_train_pieces"]),
        "validation_piece_count": int(configuration["expected_validation_pieces"]),
    }
    for key, value in expected.items():
        if int(configuration[key]) != value:
            raise RuntimeError(f"fixed split accounting mismatch for {key}")
    return datasets[0], datasets[1]


def apply_train_only_weights(
    configuration: dict[str, Any], train_dataset: Stage2PedalDataset, output_dir: Path
) -> dict[str, Any]:
    payload = train_global_class_weight_payload(train_dataset)
    if payload["source_split"] != "train" or payload["test_split_accessed"] is not False:
        raise RuntimeError("class weight provenance is not train-only")
    configuration["loss_configuration"] = {
        **configuration["loss_configuration"],
        "class_weights": payload["weights"],
        "normalization": "sum_c p_c w_c = 1",
    }
    _atomic_json(output_dir / "class_weights.json", payload)
    return payload


def _slice_batch(batch: Mapping[str, Any], start: int, end: int) -> dict[str, Any]:
    size = int(batch["input_ids"].shape[0])
    result: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == size:
            result[key] = value[start:end]
        elif isinstance(value, list) and len(value) == size:
            result[key] = value[start:end]
        else:
            result[key] = value
    return result


def _move(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _forward(
    model: EncoderDecoderFiveClassModel,
    batch: Mapping[str, Any],
    amp: bool,
    history: ScheduledSamplingHistory | None = None,
):
    with torch.amp.autocast(
        device_type=batch["input_ids"].device.type,
        dtype=torch.float16,
        enabled=amp and batch["input_ids"].device.type == "cuda",
    ):
        if history is not None:
            return model.loss_from_scheduled_history(
                input_ids=batch["input_ids"],
                token_attention_mask=batch["token_attention_mask"],
                pedal_targets=batch["pedal_targets"],
                note_mask=batch["note_mask"],
                history=history,
            )
        return model(
            input_ids=batch["input_ids"],
            token_attention_mask=batch["token_attention_mask"],
            pedal_targets=batch["pedal_targets"],
            note_mask=batch["note_mask"],
        )


def select_micro_batch_size(
    model: EncoderDecoderFiveClassModel,
    sample: Mapping[str, Any],
    device: torch.device,
    configuration: Mapping[str, Any],
) -> dict[str, Any]:
    effective = int(configuration["effective_batch_size"])
    candidates = [value for value in (16, 8, 4, 2, 1) if value <= effective and effective % value == 0]
    attempts: list[dict[str, Any]] = []
    selected = None
    amp = bool(configuration["amp_enabled"])
    for candidate in candidates:
        cpu_batch = five_class_collate_fn([sample] * candidate)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        model.zero_grad(set_to_none=True)
        try:
            batch = _move(cpu_batch, device)
            output = _forward(model, batch, amp)
            output.loss.backward()
            finite = bool(torch.isfinite(output.loss)) and all(
                p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()
            )
            peak = int(torch.cuda.max_memory_allocated(device))
            attempts.append({"micro_batch_size": candidate, "oom": False, "finite": finite, "peak_gpu_memory_bytes": peak})
            if finite:
                selected = candidate
                break
        except torch.cuda.OutOfMemoryError:
            attempts.append({"micro_batch_size": candidate, "oom": True, "finite": False})
        finally:
            model.zero_grad(set_to_none=True)
            del cpu_batch
            torch.cuda.empty_cache()
    if selected is None:
        raise RuntimeError("GPU memory smoke failed even at micro-batch 1")
    return {
        "passed": True,
        "effective_batch_size": effective,
        "selected_micro_batch_size": selected,
        "gradient_accumulation_steps": effective // selected,
        "attempts": attempts,
        "test_split_accessed": False,
    }


def _gradient_gate(model: EncoderDecoderFiveClassModel) -> dict[str, bool]:
    groups = {
        "encoder": model.encoder.parameters(),
        "decoder": model.decoder.parameters(),
        "slot_embedding": model.slot_embeddings.parameters(),
        "output": model.output_head.parameters(),
    }
    result = {}
    for name, parameters in groups.items():
        gradients = [p.grad for p in parameters if p.grad is not None]
        result[name] = bool(
            gradients
            and all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
            and sum(float(gradient.abs().sum().item()) for gradient in gradients) > 0.0
        )
    return result


def optimizer_step(
    model: EncoderDecoderFiveClassModel,
    cpu_batch: Mapping[str, Any],
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    *,
    micro_batch_size: int,
    amp_enabled: bool,
    max_grad_norm: float,
    teacher_forcing_probability_value: float = 1.0,
    scheduled_sampling_seed: int = 0,
) -> dict[str, Any]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    targets = cpu_batch["pedal_targets"]
    active = targets != -100
    global_denominator = model.class_weights.cpu()[targets[active]].sum().to(device)
    numerator_total = 0.0
    correct = valid_count = 0
    batch_size = int(cpu_batch["input_ids"].shape[0])
    probability = float(teacher_forcing_probability_value)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("teacher forcing probability must be in [0,1]")
    history: ScheduledSamplingHistory | None = None
    full_batch: dict[str, Any] | None = None
    if probability < 1.0:
        full_batch = _move(cpu_batch, device)
        generator = torch.Generator(device=device)
        generator.manual_seed(int(scheduled_sampling_seed))
        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled and device.type == "cuda",
        ):
            history = model.scheduled_sampling_history(
                full_batch["input_ids"],
                full_batch["token_attention_mask"],
                full_batch["pedal_targets"],
                full_batch["note_mask"],
                probability,
                generator=generator,
            )
        ground_truth_history_count = history.ground_truth_count
        predicted_history_count = history.predicted_count
    else:
        lengths = active.reshape(batch_size, -1).sum(dim=1)
        ground_truth_history_count = int(
            torch.clamp(lengths - 1, min=0).sum().item()
        )
        predicted_history_count = 0
    for start in range(0, batch_size, micro_batch_size):
        end = min(start + micro_batch_size, batch_size)
        if full_batch is None or history is None:
            batch = _move(_slice_batch(cpu_batch, start, end), device)
            micro_history = None
        else:
            batch = _slice_batch(full_batch, start, end)
            micro_history = ScheduledSamplingHistory(
                decoder_input_ids=history.decoder_input_ids[start:end],
                decoder_attention_mask=history.decoder_attention_mask[start:end],
                teacher_forcing_mask=history.teacher_forcing_mask[start:end],
                predictions=history.predictions[start:end],
            )
        output = _forward(model, batch, amp_enabled, micro_history)
        if output.loss_numerator is None or not bool(torch.isfinite(output.loss_numerator)):
            raise FloatingPointError("training weighted CE numerator is non-finite")
        scaled = output.loss_numerator / global_denominator.to(output.loss_numerator.dtype)
        scaler.scale(scaled).backward()
        numerator_total += float(output.loss_numerator.detach().float().item())
        mask = batch["pedal_targets"] != -100
        predictions = output.logits.detach().argmax(dim=-1)
        correct += int(((predictions == batch["pedal_targets"]) & mask).sum().item())
        valid_count += int(mask.sum().item())
    scaler.unscale_(optimizer)
    gradients = _gradient_gate(model)
    if not all(gradients.values()):
        raise RuntimeError(f"finite/nonzero gradient gate failed: {gradients}")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    norm = float(torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm, error_if_nonfinite=True).item())
    scaler.step(optimizer)
    scaler.update()
    denominator_value = float(global_denominator.item())
    history_total = ground_truth_history_count + predicted_history_count
    return {
        "loss": numerator_total / denominator_value,
        "token_accuracy": correct / valid_count,
        "valid_targets": valid_count,
        "gradient_norm": norm,
        "gradient_groups": gradients,
        "teacher_forcing_probability": probability,
        "ground_truth_history_count": ground_truth_history_count,
        "predicted_history_count": predicted_history_count,
        "ground_truth_history_ratio": (
            ground_truth_history_count / history_total if history_total else 1.0
        ),
        "predicted_history_ratio": (
            predicted_history_count / history_total if history_total else 0.0
        ),
    }


@torch.no_grad()
def validation_epoch(
    model: EncoderDecoderFiveClassModel,
    loader: DataLoader,
    device: torch.device,
    micro_batch_size: int,
    amp_enabled: bool,
) -> dict[str, float | int]:
    model.eval()
    numerator = denominator = 0.0
    correct = valid_count = 0
    for cpu_batch in loader:
        batch_size = int(cpu_batch["input_ids"].shape[0])
        for start in range(0, batch_size, micro_batch_size):
            batch = _move(_slice_batch(cpu_batch, start, min(start + micro_batch_size, batch_size)), device)
            output = _forward(model, batch, amp_enabled)
            numerator += float(output.loss_numerator.float().item())
            denominator += float(output.loss_denominator.float().item())
            mask = batch["pedal_targets"] != -100
            correct += int(((output.logits.argmax(dim=-1) == batch["pedal_targets"]) & mask).sum().item())
            valid_count += int(mask.sum().item())
    return {"loss": numerator / denominator, "token_accuracy": correct / valid_count, "valid_targets": valid_count}


def pedal_rich_tiny_overfit(
    configuration: Mapping[str, Any],
    train_dataset: Stage2PedalDataset,
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    row = next(item for item in train_dataset.performances if item["performance_path"] == PEDAL_RICH_PERFORMANCE)
    tokens = train_dataset._token_cache[row["performance_path"]]
    selected = None
    for length in (16, 24, 32, 48, 64):
        for start in range(0, len(tokens) - length + 1):
            sample = _make_window_sample(tokens, start, start + length)
            classes = five_class_collate_fn([sample])["pedal_targets"]
            counts = torch.bincount(classes.reshape(-1), minlength=NUM_CLASSES)
            if bool(torch.all(counts > 0)):
                selected = (start, length, sample, counts)
                break
        if selected is not None:
            break
    if selected is None:
        raise RuntimeError("pedal-rich train performance has no tiny all-class segment")
    start, length, sample, counts = selected
    set_deterministic_seed(int(configuration["seed"]))
    model = _make_model(configuration).to(device)
    optimizer = build_optimizer(
        model,
        encoder_lr=1e-4,
        head_lr=1e-3,
        weight_decay=0.0,
    )
    scaler = create_grad_scaler(True, "cuda", float(configuration["amp_init_scale"]))
    cpu_batch = five_class_collate_fn([sample])
    losses = []
    final_accuracy = 0.0
    gradients = {}
    for step in range(1, 201):
        metrics = optimizer_step(
            model, cpu_batch, device, optimizer, scaler,
            micro_batch_size=1, amp_enabled=True, max_grad_norm=1.0,
        )
        losses.append(float(metrics["loss"]))
        final_accuracy = float(metrics["token_accuracy"])
        gradients = dict(metrics["gradient_groups"])
        if final_accuracy >= 0.99 and step >= 10:
            break
    with torch.no_grad():
        batch = _move(cpu_batch, device)
        output = _forward(model, batch, True)
        predictions = output.logits.argmax(dim=-1)
        predicted_counts = torch.bincount(predictions.reshape(-1), minlength=NUM_CLASSES)
        final_accuracy = float((predictions == batch["pedal_targets"]).float().mean().item())
    result = {
        "passed": bool(
            final_accuracy >= 0.99
            and torch.all(predicted_counts > 0)
            and all(gradients.values())
            and all(math.isfinite(value) for value in losses)
        ),
        "source_split": "train",
        "test_split_accessed": False,
        "performance_path": PEDAL_RICH_PERFORMANCE,
        "window_start": start,
        "note_count": length,
        "target_class_counts": counts.tolist(),
        "predicted_class_counts": predicted_counts.cpu().tolist(),
        "steps": len(losses),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "final_teacher_forced_accuracy": final_accuracy,
        "gradient_groups_finite_nonzero": gradients,
    }
    _atomic_json(output_dir / "tests" / "pedal_rich_tiny_overfit.json", result)
    del model, optimizer, scaler
    torch.cuda.empty_cache()
    if not result["passed"]:
        raise RuntimeError(f"pedal-rich tiny overfit failed: {result}")
    return result


def _loader(dataset, effective_batch_size: int, pin_memory: bool, order=None) -> DataLoader:
    source = Subset(dataset, order) if order is not None else dataset
    return DataLoader(
        source,
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
        collate_fn=five_class_collate_fn,
    )


def train(
    configuration: dict[str, Any],
    train_dataset: Stage2PedalDataset,
    validation_dataset: Stage2PedalDataset,
    *,
    progress_callback: Callback | None = None,
) -> dict[str, Any]:
    output_dir = Path(configuration["output_dir"]).resolve()
    device = torch.device("cuda:0")
    set_deterministic_seed(int(configuration["seed"]))
    model = _make_model(configuration)
    initial_hashes = {
        "encoder_initial_sha256": parameter_hash(model.encoder),
        "decoder_initial_sha256": parameter_hash(model.decoder),
        "slot_embedding_initial_sha256": parameter_hash(model.slot_embeddings),
        "output_head_initial_sha256": parameter_hash(model.output_head),
    }
    reference_hashes = configuration.get("reference_initial_hashes")
    if reference_hashes is not None and dict(reference_hashes) != initial_hashes:
        raise RuntimeError(
            f"initialization differs from reference run: expected={reference_hashes} "
            f"observed={initial_hashes}"
        )
    configuration.update(initial_hashes)
    _atomic_json(output_dir / "config.json", configuration)
    model.to(device)
    optimizer = build_optimizer(
        model,
        encoder_lr=float(configuration["encoder_lr"]),
        head_lr=float(configuration["decoder_lr"]),
        weight_decay=float(configuration["weight_decay"]),
    )
    scaler = create_grad_scaler(
        bool(configuration["amp_enabled"]), "cuda", float(configuration["amp_init_scale"])
    )
    early = EarlyStopping(
        int(configuration["early_stopping_patience"]),
        float(configuration["early_stopping_min_delta"]),
    )
    effective = int(configuration["effective_batch_size"])
    micro = int(configuration["micro_batch_size"])
    validation_loader = _loader(validation_dataset, effective, bool(configuration["pin_memory"]))
    metrics_path = output_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "epoch", "global_optimizer_step", "train_loss", "train_token_accuracy",
            "teacher_forcing_probability", "actual_ground_truth_history_ratio",
            "actual_predicted_history_ratio",
            "validation_teacher_forced_weighted_ce",
            "validation_teacher_forced_token_accuracy", "epoch_seconds",
            "encoder_learning_rate", "decoder_learning_rate", "is_best_so_far",
            "improved",
            "peak_gpu_memory_bytes",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        global_step = 0
        completed_epoch = 0
        stop_reason = "max_epochs"
        run_started = time.perf_counter()
        for epoch in range(1, int(configuration["max_epochs"]) + 1):
            epoch_started = time.perf_counter()
            torch.cuda.reset_peak_memory_stats(device)
            order = deterministic_train_order(len(train_dataset), int(configuration["seed"]), epoch)
            loader = _loader(train_dataset, effective, bool(configuration["pin_memory"]), order)
            loss_sum = accuracy_weighted = targets_sum = 0.0
            ground_truth_history_sum = predicted_history_sum = 0
            probability = (
                teacher_forcing_probability(epoch)
                if bool(configuration.get("scheduled_sampling_enabled", False))
                else 1.0
            )
            for batch_index, cpu_batch in enumerate(loader, start=1):
                metrics = optimizer_step(
                    model, cpu_batch, device, optimizer, scaler,
                    micro_batch_size=micro,
                    amp_enabled=bool(configuration["amp_enabled"]),
                    max_grad_norm=float(configuration["max_grad_norm"]),
                    teacher_forcing_probability_value=probability,
                    scheduled_sampling_seed=(
                        int(configuration.get("scheduled_sampling_seed", configuration["seed"]))
                        + epoch * 1_000_000
                        + batch_index
                    ),
                )
                global_step += 1
                count = int(metrics["valid_targets"])
                loss_sum += float(metrics["loss"]) * count
                accuracy_weighted += float(metrics["token_accuracy"]) * count
                targets_sum += count
                ground_truth_history_sum += int(metrics["ground_truth_history_count"])
                predicted_history_sum += int(metrics["predicted_history_count"])
                if batch_index == 1 or batch_index % int(configuration["progress_interval"]) == 0:
                    update = {
                        "stage": "training", "epoch": epoch, "batch": batch_index,
                        "batches": len(loader), "global_step": global_step,
                        "loss": float(metrics["loss"]), "micro_batch_size": micro,
                        "teacher_forcing_probability": probability,
                        "ground_truth_history_ratio": metrics["ground_truth_history_ratio"],
                        "predicted_history_ratio": metrics["predicted_history_ratio"],
                    }
                    _notify(progress_callback, **update)
                    print(
                        f"training epoch={epoch} batch={batch_index}/{len(loader)} "
                        f"loss={metrics['loss']:.6f} global_step={global_step} "
                        f"micro_batch={micro} p_tf={probability:.2f} "
                        f"actual_gt_history={metrics['ground_truth_history_ratio']:.6f} "
                        f"actual_pred_history={metrics['predicted_history_ratio']:.6f}",
                        flush=True,
                    )
            validation = validation_epoch(
                model, validation_loader, device, micro, bool(configuration["amp_enabled"])
            )
            completed_epoch = epoch
            improved, should_stop = early.update(float(validation["loss"]), epoch)
            history_total = ground_truth_history_sum + predicted_history_sum
            row = {
                "epoch": epoch,
                "global_optimizer_step": global_step,
                "train_loss": loss_sum / targets_sum,
                "train_token_accuracy": accuracy_weighted / targets_sum,
                "teacher_forcing_probability": probability,
                "actual_ground_truth_history_ratio": (
                    ground_truth_history_sum / history_total if history_total else 1.0
                ),
                "actual_predicted_history_ratio": (
                    predicted_history_sum / history_total if history_total else 0.0
                ),
                "validation_teacher_forced_weighted_ce": validation["loss"],
                "validation_teacher_forced_token_accuracy": validation["token_accuracy"],
                "encoder_learning_rate": float(optimizer.param_groups[0]["lr"]),
                "decoder_learning_rate": float(optimizer.param_groups[-1]["lr"]),
                "is_best_so_far": bool(improved),
                "improved": bool(improved),
                "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
                "epoch_seconds": time.perf_counter() - epoch_started,
            }
            writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())
            atomic_torch_save(
                build_last_checkpoint(model, optimizer, scaler, epoch, global_step, early, configuration),
                output_dir / "last.pt",
            )
            if improved:
                atomic_torch_save(
                    build_best_checkpoint(model, configuration, early.best_epoch, early.best_loss),
                    output_dir / "best.pt",
                )
            print(f"epoch {epoch} complete metrics={json.dumps(row, sort_keys=True)} improved={improved}", flush=True)
            if should_stop and bool(configuration.get("early_stopping_enabled", True)):
                stop_reason = "early_stopping"
                break
    result = {
        "completed_epoch": completed_epoch,
        "global_optimizer_step": global_step,
        "best_epoch": early.best_epoch,
        "best_validation_loss": early.best_loss,
        "best_checkpoint_rule": "minimum teacher-forced validation weighted CE",
        "stop_reason": stop_reason,
        "early_stopped": stop_reason == "early_stopping",
        "early_stopping_enabled": bool(configuration.get("early_stopping_enabled", True)),
        "runtime_seconds": time.perf_counter() - run_started,
        "effective_batch_size": effective,
        "micro_batch_size": micro,
        "gradient_accumulation_steps": effective // micro,
        "scheduled_sampling_enabled": bool(
            configuration.get("scheduled_sampling_enabled", False)
        ),
        "teacher_forcing_schedule": (
            configuration.get("teacher_forcing_schedule")
            if configuration.get("scheduled_sampling_enabled", False)
            else None
        ),
    }
    print("ENCODER_DECODER_TRAINING_COMPLETE " + json.dumps(result, sort_keys=True), flush=True)
    del model, optimizer, scaler
    torch.cuda.empty_cache()
    return result


def prepare_preflight(
    configuration: dict[str, Any],
    train_dataset: Stage2PedalDataset,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    device = torch.device("cuda:0")
    model = _make_model(configuration).to(device)
    smoke = select_micro_batch_size(model, train_dataset[0], device, configuration)
    configuration["micro_batch_size"] = smoke["selected_micro_batch_size"]
    configuration["gradient_accumulation_steps"] = smoke["gradient_accumulation_steps"]
    _atomic_json(output_dir / "tests" / "gpu_memory_smoke.json", smoke)
    del model
    torch.cuda.empty_cache()
    overfit = pedal_rich_tiny_overfit(configuration, train_dataset, output_dir, device)
    return smoke, overfit
