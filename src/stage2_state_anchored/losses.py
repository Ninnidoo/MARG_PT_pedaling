"""Frozen State/Mode/Timing objective for State-Anchored v1."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .config import IGNORE_INDEX, MODE_CLASS_WEIGHTS
from .model import StateAnchoredModelOutput


@dataclass(frozen=True)
class StateAnchoredLossConfig:
    timing_beta: float = 0.1
    lambda_state: float = 1.0
    lambda_mode: float = 1.0
    lambda_timing: float = 1.0

    def __post_init__(self) -> None:
        if self.timing_beta != 0.1:
            raise ValueError("State-Anchored v1 freezes timing beta=0.1")
        if (self.lambda_state, self.lambda_mode, self.lambda_timing) != (1.0, 1.0, 1.0):
            raise ValueError("State-Anchored v1 freezes all loss coefficients to 1")


@dataclass
class StateAnchoredLossOutput:
    total_loss: torch.Tensor
    state_loss: torch.Tensor
    mode_loss: torch.Tensor
    timing_loss: torch.Tensor
    valid_state_targets: int
    valid_mode_targets: int
    valid_timing_targets: int


class StateAnchoredCriterion(nn.Module):
    def __init__(self, config: StateAnchoredLossConfig | None = None) -> None:
        super().__init__()
        self.config = config or StateAnchoredLossConfig()
        self.register_buffer("mode_class_weights", torch.tensor(MODE_CLASS_WEIGHTS, dtype=torch.float32))

    @staticmethod
    def _zero(reference: torch.Tensor) -> torch.Tensor:
        return reference.sum() * 0.0

    def forward(self, output: StateAnchoredModelOutput, batch: dict[str, torch.Tensor]) -> StateAnchoredLossOutput:
        leading = output.state_logits.shape[:-1]
        if output.state_logits.shape != (*leading, 2) or output.mode_logits.shape != (*leading, 3) or output.timing_predictions.shape != (*leading, 2):
            raise ValueError("State-Anchored output shape mismatch")
        state_targets, mode_targets = batch["state_targets"], batch["mode_targets"]
        timing_targets = batch["timing_targets"]
        state_valid = batch["state_valid_mask"].bool()
        mode_valid = batch["mode_valid_mask"].bool()
        timing_mask = batch["timing_mask"].bool()
        if tuple(state_targets.shape) != leading or tuple(mode_targets.shape) != leading:
            raise ValueError("State/Mode targets must match output leading shape")
        if tuple(state_valid.shape) != leading or tuple(mode_valid.shape) != leading:
            raise ValueError("State/Mode valid masks must match output leading shape")
        if tuple(timing_targets.shape) != (*leading, 2) or tuple(timing_mask.shape) != (*leading, 2):
            raise ValueError("Timing target/mask shape mismatch")
        if not torch.equal(state_targets != IGNORE_INDEX, state_valid) or not torch.equal(mode_targets != IGNORE_INDEX, mode_valid):
            raise ValueError("target padding differs from valid mask")
        if bool(torch.any((state_targets[state_valid] < 0) | (state_targets[state_valid] > 1))):
            raise ValueError("State target outside OFF/ON")
        if bool(torch.any((mode_targets[mode_valid] < 0) | (mode_targets[mode_valid] > 2))):
            raise ValueError("Mode target outside HOLD/CHANGE/RETURN")
        expected_timing = torch.zeros_like(timing_mask)
        expected_timing[..., 0] = mode_valid & (mode_targets >= 1)
        expected_timing[..., 1] = mode_valid & (mode_targets == 2)
        if not torch.equal(timing_mask, expected_timing):
            raise ValueError("Timing mask is inconsistent with Mode targets")
        state_loss = (
            F.cross_entropy(output.state_logits[state_valid], state_targets[state_valid])
            if bool(state_valid.any()) else self._zero(output.state_logits)
        )
        if bool(mode_valid.any()):
            selected_modes = mode_targets[mode_valid]
            mode_sum = F.cross_entropy(output.mode_logits[mode_valid], selected_modes, weight=self.mode_class_weights, reduction="sum")
            mode_loss = mode_sum / self.mode_class_weights[selected_modes].sum()
        else:
            mode_loss = self._zero(output.mode_logits)
        timing_loss = (
            F.smooth_l1_loss(output.timing_predictions[timing_mask], timing_targets[timing_mask], beta=self.config.timing_beta, reduction="mean")
            if bool(timing_mask.any()) else self._zero(output.timing_predictions)
        )
        total = state_loss + mode_loss + timing_loss
        if not all(bool(torch.isfinite(value)) for value in (state_loss, mode_loss, timing_loss, total)):
            raise FloatingPointError("State-Anchored objective is non-finite")
        return StateAnchoredLossOutput(total, state_loss, mode_loss, timing_loss, int(state_valid.sum()), int(mode_valid.sum()), int(timing_mask.sum()))
