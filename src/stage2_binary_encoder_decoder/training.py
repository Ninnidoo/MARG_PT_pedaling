"""Teacher-forced training utilities for the canonical binary encoder-decoder."""

from __future__ import annotations

import math
from itertools import islice
from typing import Any, Callable, Mapping

import torch
from torch.utils.data import DataLoader

from src.stage2_binary.full_training import BinaryMetricAccumulator, binary_batch_metrics
from src.stage2_encoder_only.training import move_batch_to_device

from .model import BinaryPedalEncoderDecoderModel


def _groups(loader: DataLoader, size: int):
    if size <= 0:
        raise ValueError("gradient accumulation steps must be positive")
    iterator = iter(loader)
    while True:
        group = list(islice(iterator, size))
        if not group:
            return
        yield group


def teacher_forced_forward(
    model: BinaryPedalEncoderDecoderModel,
    batch: Mapping[str, Any],
    *,
    amp_enabled: bool,
):
    use_amp = bool(amp_enabled and batch["input_ids"].device.type == "cuda")
    with torch.amp.autocast(
        device_type=batch["input_ids"].device.type,
        dtype=torch.float16 if use_amp else torch.bfloat16,
        enabled=use_amp,
    ):
        return model(
            input_ids=batch["input_ids"],
            token_attention_mask=batch["token_attention_mask"],
            binary_targets=batch["binary_targets"],
            note_mask=batch["note_mask"],
            decode_mode="teacher_forced",
        )


def run_teacher_forced_validation(
    model: BinaryPedalEncoderDecoderModel,
    loader: DataLoader,
    *,
    device: torch.device,
    amp_enabled: bool,
) -> dict[str, float | int]:
    model.eval()
    accumulator = BinaryMetricAccumulator("independent_4x2")
    with torch.inference_mode():
        for cpu_batch in loader:
            batch = move_batch_to_device(cpu_batch, device)
            output = teacher_forced_forward(model, batch, amp_enabled=amp_enabled)
            if output.loss is None or not bool(torch.isfinite(output.loss)):
                raise FloatingPointError("teacher-forced validation loss is invalid")
            accumulator.update(
                binary_batch_metrics(
                    output.logits,
                    batch["binary_targets"],
                    output.loss,
                    "independent_4x2",
                )
            )
    return accumulator.compute()


def run_accumulated_training_epoch(
    model: BinaryPedalEncoderDecoderModel,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    amp_enabled: bool,
    accumulation_steps: int,
    max_grad_norm: float,
    max_consecutive_amp_skips: int,
    step_callback: Callable[[int, Mapping[str, Any]], None] | None = None,
) -> tuple[dict[str, float | int], dict[str, Any]]:
    """Accumulate exact valid-token CE over micro-batches before each step."""

    model.train()
    accumulator = BinaryMetricAccumulator("independent_4x2")
    optimizer_steps = 0
    attempted_steps = 0
    skipped_steps = 0
    consecutive_skips = 0
    maximum_consecutive_skips = 0
    micro_batches = 0
    encoder_gradient_present = False
    head_gradient_present = False
    use_amp = bool(amp_enabled and device.type == "cuda")

    for cpu_group in _groups(loader, accumulation_steps):
        group_denominator = sum(
            int((batch["binary_targets"] != -100).sum().item())
            for batch in cpu_group
        )
        if group_denominator <= 0:
            raise ValueError("gradient-accumulation group has no valid targets")
        optimizer.zero_grad(set_to_none=True)
        group_loss_numerator = 0.0
        for cpu_batch in cpu_group:
            batch = move_batch_to_device(cpu_batch, device)
            output = teacher_forced_forward(model, batch, amp_enabled=amp_enabled)
            if (
                output.loss is None
                or output.loss_numerator is None
                or not bool(torch.isfinite(output.loss))
            ):
                raise FloatingPointError("teacher-forced training loss is invalid")
            scaled_loss = output.loss_numerator / float(group_denominator)
            if use_amp:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            group_loss_numerator += float(output.loss_numerator.detach().item())
            accumulator.update(
                binary_batch_metrics(
                    output.logits.detach(),
                    batch["binary_targets"],
                    output.loss,
                    "independent_4x2",
                )
            )
            micro_batches += 1

        if use_amp:
            scaler.unscale_(optimizer)
        total_gradient_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            max_norm=max_grad_norm,
            error_if_nonfinite=False,
        )
        gradient_norm_finite = bool(torch.isfinite(total_gradient_norm))
        if attempted_steps == 0:
            encoder_gradient_present = any(
                parameter.grad is not None and bool(torch.any(parameter.grad != 0))
                for parameter in model.encoder.parameters()
                if parameter.requires_grad
            )
            head_gradient_present = any(
                parameter.grad is not None and bool(torch.any(parameter.grad != 0))
                for parameter in model.prediction_head_parameters()
                if parameter.requires_grad
            )
            if not encoder_gradient_present or not head_gradient_present:
                raise RuntimeError("first accumulated step did not reach encoder and decoder/head")

        scale_before = float(scaler.get_scale()) if use_amp else None
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
            scale_after = float(scaler.get_scale())
            stepped = gradient_norm_finite and scale_after >= float(scale_before)
        else:
            if not gradient_norm_finite:
                raise FloatingPointError("non-finite gradient norm without AMP")
            optimizer.step()
            scale_after = None
            stepped = True
        attempted_steps += 1
        if stepped:
            optimizer_steps += 1
            consecutive_skips = 0
        else:
            skipped_steps += 1
            consecutive_skips += 1
            maximum_consecutive_skips = max(maximum_consecutive_skips, consecutive_skips)
            if consecutive_skips >= max_consecutive_amp_skips:
                raise FloatingPointError(
                    f"persistent AMP overflow: {consecutive_skips} consecutive skipped steps"
                )
        details = {
            "attempted_steps": attempted_steps,
            "optimizer_steps": optimizer_steps,
            "amp_skipped_steps": skipped_steps,
            "micro_batches": micro_batches,
            "micro_batches_in_step": len(cpu_group),
            "valid_targets_in_step": group_denominator,
            "loss": group_loss_numerator / group_denominator,
            "gradient_norm": float(total_gradient_norm.detach().item()),
            "gradient_norm_finite": gradient_norm_finite,
            "amp_scale_before": scale_before,
            "amp_scale_after": scale_after,
            "optimizer_step_applied": stepped,
        }
        if not math.isfinite(details["loss"]):
            raise FloatingPointError("non-finite accumulated training loss")
        if step_callback is not None:
            step_callback(attempted_steps, details)

    return accumulator.compute(), {
        "micro_batches": micro_batches,
        "attempted_steps": attempted_steps,
        "optimizer_steps": optimizer_steps,
        "amp_skipped_steps": skipped_steps,
        "maximum_consecutive_amp_skips": maximum_consecutive_skips,
        "encoder_gradient_present": encoder_gradient_present,
        "head_gradient_present": head_gradient_present,
    }
