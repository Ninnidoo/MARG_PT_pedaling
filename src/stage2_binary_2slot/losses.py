"""Frozen count/timing objective with explicit MAIN/BOUNDARY aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn


FIXED_COUNT_WEIGHTS = (
    0.7577816791288143,
    2.246984238727598,
    4.428054342343362,
)


@dataclass(frozen=True)
class Binary2SlotLossConfig:
    boundary_loss_weight: float
    timing_beta: float = 0.1

    def __post_init__(self) -> None:
        if not float(self.boundary_loss_weight) >= 0.0:
            raise ValueError("boundary_loss_weight must be explicitly non-negative")
        if float(self.timing_beta) != 0.1:
            raise ValueError("Binary 2-Slot v0 freezes timing beta=0.1")


@dataclass
class Binary2SlotLossOutput:
    total_loss: torch.Tensor
    main_loss: torch.Tensor
    boundary_loss: torch.Tensor
    main_count_loss: torch.Tensor
    main_timing_loss: torch.Tensor
    boundary_count_loss: torch.Tensor
    boundary_timing_loss: torch.Tensor
    counts: dict[str, int]


class Binary2SlotCriterion(nn.Module):
    def __init__(self, config: Binary2SlotLossConfig) -> None:
        super().__init__()
        self.config = config
        self.register_buffer("count_weights", torch.tensor(FIXED_COUNT_WEIGHTS, dtype=torch.float32))

    @staticmethod
    def _zero(reference: torch.Tensor) -> torch.Tensor:
        return reference.sum() * 0.0

    def _group_loss(
        self,
        count_logits: torch.Tensor,
        timing_predictions: torch.Tensor,
        target_counts: torch.Tensor,
        timing_targets: torch.Tensor,
        timing_mask: torch.Tensor,
        selected: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not bool(selected.any()):
            return self._zero(count_logits), self._zero(timing_predictions)
        counts = target_counts[selected]
        if bool(torch.any((counts < 0) | (counts > 2))):
            raise ValueError("target N must be 0, 1, or 2")
        count_loss = F.cross_entropy(
            count_logits[selected], counts, weight=self.count_weights, reduction="mean"
        )
        active = timing_mask[selected]
        if bool(active.any()):
            timing_loss = F.smooth_l1_loss(
                timing_predictions[selected][active],
                timing_targets[selected][active],
                beta=self.config.timing_beta,
                reduction="mean",
            )
        else:
            timing_loss = self._zero(timing_predictions)
        return count_loss, timing_loss

    def forward(
        self,
        count_logits: torch.Tensor,
        timing_predictions: torch.Tensor,
        target_counts: torch.Tensor,
        timing_targets: torch.Tensor,
        timing_mask: torch.Tensor,
        regions: Sequence[str],
    ) -> Binary2SlotLossOutput:
        interval_count = count_logits.shape[0]
        if tuple(count_logits.shape) != (interval_count, 3):
            raise ValueError("count logits must have shape [I,3]")
        if tuple(timing_predictions.shape) != (interval_count, 2):
            raise ValueError("timing predictions must have shape [I,2]")
        if tuple(target_counts.shape) != (interval_count,):
            raise ValueError("target counts must have shape [I]")
        if tuple(timing_targets.shape) != (interval_count, 2) or tuple(timing_mask.shape) != (interval_count, 2):
            raise ValueError("timing targets/mask must have shape [I,2]")
        if len(regions) != interval_count or any(region not in {"PRE", "MAIN", "POST"} for region in regions):
            raise ValueError("regions must label every interval PRE/MAIN/POST")
        expected_mask = torch.stack((target_counts >= 1, target_counts >= 2), dim=-1)
        if not torch.equal(timing_mask.bool(), expected_mask):
            raise ValueError("timing mask must be derived from target N")
        main_mask = torch.tensor([region == "MAIN" for region in regions], device=count_logits.device)
        boundary_mask = ~main_mask
        main_count, main_timing = self._group_loss(
            count_logits, timing_predictions, target_counts, timing_targets, timing_mask.bool(), main_mask
        )
        boundary_count, boundary_timing = self._group_loss(
            count_logits, timing_predictions, target_counts, timing_targets, timing_mask.bool(), boundary_mask
        )
        main = main_count + main_timing
        boundary = boundary_count + boundary_timing
        total = main + float(self.config.boundary_loss_weight) * boundary
        values = (total, main, boundary, main_count, main_timing, boundary_count, boundary_timing)
        if not all(bool(torch.isfinite(value)) for value in values):
            raise FloatingPointError("Binary 2-Slot objective is non-finite")
        return Binary2SlotLossOutput(
            total_loss=total,
            main_loss=main,
            boundary_loss=boundary,
            main_count_loss=main_count,
            main_timing_loss=main_timing,
            boundary_count_loss=boundary_count,
            boundary_timing_loss=boundary_timing,
            counts={
                "main_intervals": int(main_mask.sum().item()),
                "boundary_intervals": int(boundary_mask.sum().item()),
                "active_main_timing_slots": int((timing_mask.bool() & main_mask.unsqueeze(-1)).sum().item()),
                "active_boundary_timing_slots": int((timing_mask.bool() & boundary_mask.unsqueeze(-1)).sum().item()),
            },
        )
