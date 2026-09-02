"""Opt-in parity state-consistency objective for Binary 2-Slot diagnostics."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def required_toggle_targets(
    current_states: torch.Tensor,
    human_next_states: torch.Tensor,
) -> torch.Tensor:
    """Return OFF/ON XOR targets: one iff this interval must toggle."""

    if current_states.shape != human_next_states.shape:
        raise ValueError("current and human-next states must have identical shape")
    if not bool(torch.all((current_states == 0) | (current_states == 1))):
        raise ValueError("current states must be binary")
    if not bool(torch.all((human_next_states == 0) | (human_next_states == 1))):
        raise ValueError("human-next states must be binary")
    return torch.bitwise_xor(current_states.long(), human_next_states.long())


def state_consistency_loss(
    count_logits: torch.Tensor,
    current_states: torch.Tensor,
    human_next_states: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Parity NLL derived only from Count logits, with no soft recurrence.

    N=1 is the toggle class. N=0 and N=2 jointly represent no-toggle.
    The log-space implementation is exactly BCE(P(N=1), required_toggle)
    while remaining stable for extreme logits.
    """

    if count_logits.shape[:-1] != current_states.shape or count_logits.shape[-1] != 3:
        raise ValueError("count logits must be [...,3] and states must match leading dimensions")
    required = required_toggle_targets(current_states, human_next_states)
    log_probabilities = F.log_softmax(count_logits, dim=-1)
    log_toggle = log_probabilities[..., 1]
    log_no_toggle = torch.logsumexp(log_probabilities[..., (0, 2)], dim=-1)
    losses = torch.where(required.bool(), -log_toggle, -log_no_toggle)
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    if reduction == "mean":
        return losses.mean()
    raise ValueError("reduction must be none, sum, or mean")
