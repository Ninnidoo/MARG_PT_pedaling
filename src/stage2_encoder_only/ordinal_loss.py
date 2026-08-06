"""Squared-CDF (Cramer-style) auxiliary loss for ordered pedal values."""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F


NUM_PEDAL_CLASSES = 128
IGNORE_INDEX = -100


class OrdinalLosses(NamedTuple):
    """The three scalar objectives used by ordinal Stage 2 training."""

    total: torch.Tensor
    ce: torch.Tensor
    ordinal: torch.Tensor


def ordinal_target_cdf(
    targets: torch.Tensor,
    num_classes: int = NUM_PEDAL_CLASSES,
) -> torch.Tensor:
    """Return F_y(r)=1[r >= y] for thresholds 0..num_classes-2."""

    if num_classes < 2:
        raise ValueError("num_classes must be at least two")
    thresholds = torch.arange(
        num_classes - 1, device=targets.device, dtype=torch.long
    )
    return thresholds >= targets.to(torch.long).unsqueeze(-1)


def squared_cdf_ordinal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    lambda_ordinal: float,
    ignore_index: int = IGNORE_INDEX,
) -> OrdinalLosses:
    """Compute unweighted CE plus the squared-CDF ordinal auxiliary loss.

    Softmax, cumulative probabilities, and squared differences are always
    evaluated in float32, including when the forward pass used AMP.
    """

    if logits.ndim < 2 or logits.shape[-1] != NUM_PEDAL_CLASSES:
        raise ValueError("logits must end in 128 pedal classes")
    if tuple(logits.shape[:-1]) != tuple(targets.shape):
        raise ValueError("target shape must equal logits shape without classes")
    if lambda_ordinal < 0:
        raise ValueError("lambda_ordinal must be non-negative")

    flat_logits = logits.reshape(-1, NUM_PEDAL_CLASSES)
    flat_targets = targets.reshape(-1).to(torch.long)
    valid = flat_targets != ignore_index
    if not bool(valid.any()):
        raise ValueError("loss requires at least one valid pedal target")

    valid_logits = flat_logits[valid]
    valid_targets = flat_targets[valid]
    # Keep the existing unweighted CE semantics and average over valid targets.
    ce = F.cross_entropy(valid_logits.to(torch.float32), valid_targets)

    probabilities = torch.softmax(valid_logits.to(torch.float32), dim=-1)
    predicted_cdf = probabilities.cumsum(dim=-1)[..., :-1]
    target_cdf = ordinal_target_cdf(valid_targets).to(torch.float32)
    ordinal = (predicted_cdf - target_cdf).square().mean()
    total = ce + float(lambda_ordinal) * ordinal
    return OrdinalLosses(total=total, ce=ce, ordinal=ordinal)
