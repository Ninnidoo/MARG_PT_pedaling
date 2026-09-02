"""Shared optimization helpers for the two binary Stage 2 heads."""

from __future__ import annotations

from typing import Any

import torch

from .model import BinaryStage2EncoderBase


def build_binary_optimizer(
    model: BinaryStage2EncoderBase,
    *,
    encoder_lr: float = 1e-4,
    head_lr: float = 1e-3,
    weight_decay: float = 0.0,
) -> torch.optim.AdamW:
    """Build complete, disjoint encoder/head parameter groups."""

    if encoder_lr <= 0 or head_lr <= 0:
        raise ValueError("learning rates must be positive")
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    encoder_parameters = [
        parameter for parameter in model.encoder.parameters() if parameter.requires_grad
    ]
    head_parameters = [
        parameter
        for parameter in model.prediction_head_parameters()
        if parameter.requires_grad
    ]
    grouped = encoder_parameters + head_parameters
    grouped_ids = [id(parameter) for parameter in grouped]
    trainable_ids = [
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    ]
    if len(grouped_ids) != len(set(grouped_ids)):
        raise ValueError("optimizer parameter groups contain duplicates")
    if set(grouped_ids) != set(trainable_ids):
        raise ValueError("optimizer groups do not cover all trainable parameters")
    groups: list[dict[str, Any]] = [
        {
            "params": encoder_parameters,
            "lr": encoder_lr,
            "weight_decay": weight_decay,
            "group_name": "encoder",
        },
        {
            "params": head_parameters,
            "lr": head_lr,
            "weight_decay": weight_decay,
            "group_name": "head",
        },
    ]
    return torch.optim.AdamW(groups)
