"""Focused training utilities for Stage 2 pedal classification."""

from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torch import nn


DEFAULT_SEED = 20260710
IGNORE_INDEX = -100
PEDAL_SLOTS = 4


def set_deterministic_seed(seed: int = DEFAULT_SEED) -> torch.Generator:
    """Seed Python, NumPy, PyTorch, and a returned DataLoader generator."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def build_optimizer(
    model: nn.Module,
    encoder_lr: float = 1e-5,
    head_lr: float = 1e-4,
    weight_decay: float = 0.01,
) -> torch.optim.AdamW:
    """Construct AdamW with complete, disjoint encoder and head groups."""

    if encoder_lr <= 0 or head_lr <= 0:
        raise ValueError("encoder_lr and head_lr must be positive")
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    encoder_parameters = [
        p for p in model.encoder.parameters() if p.requires_grad
    ]
    head_parameters = [
        p for p in model.classification_heads.parameters() if p.requires_grad
    ]
    grouped = encoder_parameters + head_parameters
    grouped_ids = [id(p) for p in grouped]
    trainable_ids = [id(p) for p in model.parameters() if p.requires_grad]
    if len(grouped_ids) != len(set(grouped_ids)):
        raise ValueError("optimizer parameter groups contain duplicates")
    if set(grouped_ids) != set(trainable_ids):
        raise ValueError("optimizer parameter groups omit trainable parameters")

    groups: list[dict[str, Any]] = []
    if encoder_parameters:
        groups.append(
            {
                "params": encoder_parameters,
                "lr": encoder_lr,
                "weight_decay": weight_decay,
                "group_name": "encoder",
            }
        )
    if head_parameters:
        groups.append(
            {
                "params": head_parameters,
                "lr": head_lr,
                "weight_decay": weight_decay,
                "group_name": "heads",
            }
        )
    if not groups:
        raise ValueError("model has no trainable encoder or head parameters")
    return torch.optim.AdamW(groups)


def create_grad_scaler(
    enabled: bool,
    device: str = "cuda",
    amp_init_scale: float = 1024.0,
) -> torch.amp.GradScaler:
    """Create a GradScaler using the installed torch.amp API."""

    if amp_init_scale <= 0:
        raise ValueError("amp_init_scale must be positive")
    return torch.amp.GradScaler(device=device, enabled=enabled, init_scale=amp_init_scale)


def move_batch_to_device(
    batch: Mapping[str, Any],
    device: torch.device | str,
) -> dict[str, Any]:
    """Move model inputs to a device while preserving metadata."""

    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def calculate_pedal_metrics(
    logits: torch.Tensor,
    pedal_targets: torch.Tensor,
    loss: torch.Tensor | float | None = None,
) -> dict[str, float | int]:
    """Calculate pedal metrics over targets not equal to -100."""

    if logits.ndim != 4 or logits.shape[-2] != PEDAL_SLOTS:
        raise ValueError("logits must have shape [B, N, 4, classes]")
    if tuple(pedal_targets.shape) != tuple(logits.shape[:-1]):
        raise ValueError("pedal_targets shape must match logits [B, N, 4]")

    predictions = logits.argmax(dim=-1)
    valid = pedal_targets != IGNORE_INDEX
    valid_target_count = int(valid.sum().item())
    valid_notes = valid.any(dim=-1)
    valid_note_count = int(valid_notes.sum().item())
    if valid_target_count == 0:
        raise ValueError("metrics require at least one valid pedal target")

    correct = (predictions == pedal_targets) & valid
    token_accuracy = float(correct.sum().item() / valid_target_count)
    note_correct = ((predictions == pedal_targets) | ~valid).all(dim=-1)
    exact_note_accuracy = float(
        (note_correct & valid_notes).sum().item() / valid_note_count
    )
    slot_accuracies = []
    for slot in range(PEDAL_SLOTS):
        slot_valid = valid[..., slot]
        slot_count = int(slot_valid.sum().item())
        slot_accuracies.append(
            float(correct[..., slot].sum().item() / slot_count)
            if slot_count
            else float("nan")
        )
    mae = float(
        (
            predictions[valid].to(torch.float32)
            - pedal_targets[valid].to(torch.float32)
        )
        .abs()
        .mean()
        .item()
    )
    if isinstance(loss, torch.Tensor):
        loss_value = float(loss.detach().item())
    elif loss is None:
        loss_value = float("nan")
    else:
        loss_value = float(loss)

    return {
        "loss": loss_value,
        "pedal_token_accuracy": token_accuracy,
        "exact_note_accuracy": exact_note_accuracy,
        "pedal1_accuracy": slot_accuracies[0],
        "pedal2_accuracy": slot_accuracies[1],
        "pedal3_accuracy": slot_accuracies[2],
        "pedal4_accuracy": slot_accuracies[3],
        "pedal_value_mae": mae,
        "valid_target_count": valid_target_count,
        "valid_note_count": valid_note_count,
    }


def _parameters_with_gradients(parameters: Any) -> list[nn.Parameter]:
    return [parameter for parameter in parameters if parameter.grad is not None]


def _gradient_norm(parameters: Any) -> float:
    with_gradients = _parameters_with_gradients(parameters)
    if not with_gradients:
        return 0.0
    norms = torch.stack([p.grad.detach().norm(2) for p in with_gradients])
    return float(norms.norm(2).item())


def _validate_finite_gradients(parameters: Any) -> None:
    for parameter in parameters:
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
            raise FloatingPointError("non-finite gradient detected")


def _model_forward(
    model: nn.Module,
    batch: Mapping[str, Any],
    amp_enabled: bool,
) -> Any:
    input_ids = batch["input_ids"]
    device_type = input_ids.device.type
    use_amp = amp_enabled and device_type == "cuda"
    with torch.amp.autocast(
        device_type=device_type,
        dtype=torch.float16 if device_type == "cuda" else torch.bfloat16,
        enabled=use_amp,
    ):
        return model(
            input_ids=input_ids,
            token_attention_mask=batch["token_attention_mask"],
            pedal_targets=batch["pedal_targets"],
            note_mask=batch.get("note_mask"),
        )


def train_step(
    model: nn.Module,
    batch: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None = None,
    amp_enabled: bool = False,
    max_grad_norm: float = 1.0,
) -> dict[str, float | int]:
    """Run one finite checked, optionally AMP-scaled optimizer step."""

    if max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = _model_forward(model, batch, amp_enabled)
    if output.loss is None or not bool(torch.isfinite(output.loss)):
        raise FloatingPointError("training loss is missing or non-finite")

    use_amp = amp_enabled and batch["input_ids"].device.type == "cuda"
    if scaler is None:
        scaler = create_grad_scaler(
            enabled=use_amp,
            device=batch["input_ids"].device.type,
        )
    if use_amp:
        scaler.scale(output.loss).backward()
        scaler.unscale_(optimizer)
    else:
        output.loss.backward()

    encoder_parameters = list(model.encoder.parameters())
    head_parameters = list(model.classification_heads.parameters())
    trainable_encoder = [p for p in encoder_parameters if p.requires_grad]
    if trainable_encoder and not _parameters_with_gradients(trainable_encoder):
        raise RuntimeError("trainable encoder received no gradients")
    for slot, head in enumerate(model.classification_heads):
        if not _parameters_with_gradients(head.parameters()):
            raise RuntimeError(f"prediction head {slot + 1} received no gradients")

    all_trainable = [p for p in model.parameters() if p.requires_grad]
    _validate_finite_gradients(all_trainable)
    encoder_gradient_norm = _gradient_norm(encoder_parameters)
    head_gradient_norm = _gradient_norm(head_parameters)
    total_gradient_norm = float(
        torch.nn.utils.clip_grad_norm_(
            all_trainable,
            max_norm=max_grad_norm,
            error_if_nonfinite=True,
        ).item()
    )
    clipped_gradient_norm = _gradient_norm(all_trainable)
    _validate_finite_gradients(all_trainable)

    if use_amp:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

    metrics = calculate_pedal_metrics(
        output.logits.detach(),
        batch["pedal_targets"],
        output.loss,
    )
    metrics.update(
        {
            "encoder_gradient_norm": encoder_gradient_norm,
            "head_gradient_norm": head_gradient_norm,
            "total_gradient_norm": total_gradient_norm,
            "clipped_gradient_norm": clipped_gradient_norm,
        }
    )
    return metrics


def evaluation_step(
    model: nn.Module,
    batch: Mapping[str, Any],
    amp_enabled: bool = False,
) -> dict[str, float | int]:
    """Evaluate one batch without gradients or parameter changes."""

    model.eval()
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        output = _model_forward(model, batch, amp_enabled)
    if output.loss is None or not bool(torch.isfinite(output.loss)):
        raise FloatingPointError("evaluation loss is missing or non-finite")
    metrics = calculate_pedal_metrics(
        output.logits,
        batch["pedal_targets"],
        output.loss,
    )
    model.zero_grad(set_to_none=True)
    return metrics
