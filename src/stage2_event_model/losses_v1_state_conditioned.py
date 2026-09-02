"""Frozen B3-S objective: v0 losses plus detached semantic pre-state CE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .dataset import IGNORE_INDEX, MAIN_SLOTS
from .losses import CustomEventCriterion, CustomEventLossOutput
from .model_v1_state_conditioned import CustomEventStateConditionedOutput, STATE_CLASSES


STATE_LOSS_COEFFICIENT = 1.0


@dataclass
class CustomEventStateConditionedLossOutput:
    total_loss: torch.Tensor
    legacy_selection_loss: torch.Tensor
    main_state_ce: torch.Tensor
    initial_ce: torch.Tensor
    main_event_ce: torch.Tensor
    main_timing_huber: torch.Tensor
    terminal_event_ce: torch.Tensor
    terminal_timing_huber: torch.Tensor
    per_main_slot_state_loss: tuple[torch.Tensor, ...]
    per_main_slot_event_loss: tuple[torch.Tensor, ...]
    per_main_slot_timing_loss: tuple[torch.Tensor, ...]
    per_terminal_slot_event_loss: tuple[torch.Tensor, ...]
    per_terminal_slot_timing_loss: tuple[torch.Tensor, ...]
    valid_target_counts: dict[str, Any]


class CustomEventStateConditionedCriterion(CustomEventCriterion):
    """Keep every v0 loss exact and add unit-weight, slot-macro state CE."""

    def forward(
        self,
        output: CustomEventStateConditionedOutput,
        batch: dict[str, torch.Tensor],
    ) -> CustomEventStateConditionedLossOutput:
        legacy: CustomEventLossOutput = super().forward(output, batch)
        owned = batch["owned_onset_mask"].bool()
        targets = batch["main_pre_state_targets"]
        losses = []
        counts = []
        for slot in range(MAIN_SLOTS):
            valid = owned & (targets[:, :, slot] != IGNORE_INDEX)
            if not bool(valid.any()):
                loss = output.main_state_logits[:, :, slot].sum() * 0.0
            else:
                selected = targets[:, :, slot][valid]
                if bool(((selected < 0) | (selected >= STATE_CLASSES)).any()):
                    raise ValueError("pre-state target outside [0,3]")
                loss = F.cross_entropy(output.main_state_logits[:, :, slot][valid], selected)
            losses.append(loss)
            counts.append(int(valid.sum().item()))
        state_ce = torch.stack(losses).mean()
        total = legacy.total_loss + STATE_LOSS_COEFFICIENT * state_ce
        values = (total, legacy.total_loss, state_ce, *losses)
        if not all(bool(torch.isfinite(value)) for value in values):
            raise FloatingPointError("B3-S objective is non-finite")
        counts_payload = dict(legacy.valid_target_counts)
        counts_payload["main_state_by_slot"] = counts
        return CustomEventStateConditionedLossOutput(
            total_loss=total,
            legacy_selection_loss=legacy.total_loss,
            main_state_ce=state_ce,
            initial_ce=legacy.initial_ce,
            main_event_ce=legacy.main_event_ce,
            main_timing_huber=legacy.main_timing_huber,
            terminal_event_ce=legacy.terminal_event_ce,
            terminal_timing_huber=legacy.terminal_timing_huber,
            per_main_slot_state_loss=tuple(losses),
            per_main_slot_event_loss=legacy.per_main_slot_event_loss,
            per_main_slot_timing_loss=legacy.per_main_slot_timing_loss,
            per_terminal_slot_event_loss=legacy.per_terminal_slot_event_loss,
            per_terminal_slot_timing_loss=legacy.per_terminal_slot_timing_loss,
            valid_target_counts=counts_payload,
        )
