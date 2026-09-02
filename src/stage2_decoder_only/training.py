"""Training utilities specific to canonical Decoder-only Stage 2 v0."""

from __future__ import annotations

import torch

from .model import FourClassPedalDecoderOnlyModel


def build_decoder_only_optimizer(
    model: FourClassPedalDecoderOnlyModel,
    *,
    pretrained_lr: float,
    fresh_lr: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    """Use the established lower LR for pretrained representation weights."""

    if pretrained_lr <= 0 or fresh_lr <= 0:
        raise ValueError("learning rates must be positive")
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    pretrained = [
        parameter
        for parameter in model.pretrained_representation_parameters()
        if parameter.requires_grad
    ]
    fresh = [
        parameter for parameter in model.fresh_parameters() if parameter.requires_grad
    ]
    grouped = pretrained + fresh
    grouped_ids = [id(parameter) for parameter in grouped]
    trainable_ids = [
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    ]
    if len(grouped_ids) != len(set(grouped_ids)):
        raise ValueError("optimizer parameter groups contain duplicates")
    if set(grouped_ids) != set(trainable_ids):
        raise ValueError("optimizer groups do not cover all trainable parameters")
    return torch.optim.AdamW(
        [
            {
                "params": pretrained,
                "lr": pretrained_lr,
                "weight_decay": weight_decay,
                "group_name": "pretrained_performance_representation",
            },
            {
                "params": fresh,
                "lr": fresh_lr,
                "weight_decay": weight_decay,
                "group_name": "fresh_prefix_lm",
            },
        ]
    )
