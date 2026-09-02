"""Raw-CC64 target variant of the canonical four-scalar Huber encoder model."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from src.stage2_encoder_only.dataset import (
    PEDAL_SLOTS,
    stage2_pedal_collate_fn,
)

from .loss_objectives import (
    HUBER_DELTA_NORMALIZED,
    ObjectiveOutput,
    RepresentativeHuberEncoderModel,
    normalized_scalars_to_cc64,
)
from .representation import IGNORE_INDEX


def raw_cc64_to_normalized_targets(raw_targets: torch.Tensor) -> torch.Tensor:
    """Preserve every raw integer CC64 value and normalize it by 127."""

    if raw_targets.ndim < 1 or raw_targets.shape[-1] != PEDAL_SLOTS:
        raise ValueError("raw CC64 targets must have shape [...,4]")
    if raw_targets.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise TypeError("raw CC64 targets must use an integer dtype")
    valid = raw_targets != IGNORE_INDEX
    illegal = valid & ((raw_targets < 0) | (raw_targets > 127))
    if bool(illegal.any()):
        raise ValueError("raw CC64 targets must be in [0,127] or -100")
    normalized = raw_targets.to(torch.float32) / 127.0
    return normalized.masked_fill(~valid, float(IGNORE_INDEX))


def make_raw_cc64_loader(
    dataset: Dataset[dict[str, Any]],
    *,
    batch_size: int,
    pin_memory: bool,
    order: Sequence[int] | None = None,
) -> DataLoader:
    """Reuse the immutable cache and baseline collator without class quantization."""

    source: Dataset = Subset(dataset, list(order)) if order is not None else dataset
    return DataLoader(
        source,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=bool(pin_memory),
        collate_fn=stage2_pedal_collate_fn,
    )


class RawCC64HuberEncoderModel(RepresentativeHuberEncoderModel):
    """Same encoder/Linear(H,1)x4 architecture, with direct raw CC64 targets."""

    target_name = "pedal_targets"

    def forward(
        self,
        input_ids: torch.Tensor,
        token_attention_mask: torch.Tensor,
        pedal_targets: torch.Tensor | None = None,
        note_mask: torch.Tensor | None = None,
    ) -> ObjectiveOutput:
        hidden, batch_size, note_count = self._encode(
            input_ids, token_attention_mask, note_mask
        )
        if pedal_targets is not None:
            self._validate_mask_alignment(
                pedal_targets,
                note_mask,
                (batch_size, note_count, PEDAL_SLOTS),
                128,
            )
        states = self.dropout(hidden)
        values = torch.cat([head(states) for head in self.regression_heads], dim=-1)
        expected = (batch_size, note_count, PEDAL_SLOTS)
        if tuple(values.shape) != expected:
            raise AssertionError(f"raw regression output must have shape {expected}")
        if pedal_targets is None:
            return ObjectiveOutput(predictions=values, encoder_hidden_states=hidden)

        targets = raw_cc64_to_normalized_targets(pedal_targets).to(values.dtype)
        valid = pedal_targets != IGNORE_INDEX
        huber_sum = F.smooth_l1_loss(
            values[valid], targets[valid], beta=self.delta, reduction="sum"
        )
        count = valid.sum().to(values.dtype)
        clipped_cc64 = normalized_scalars_to_cc64(values[valid].float())
        target_cc64 = pedal_targets[valid].float()
        absolute_error = (clipped_cc64 - target_cc64).abs()
        squared_error = (clipped_cc64 - target_cc64).square()
        loss = huber_sum / count
        if not bool(torch.isfinite(loss)) or float(count.item()) <= 0.0:
            raise FloatingPointError("raw CC64 Huber objective is invalid")
        return ObjectiveOutput(
            predictions=values,
            loss=loss,
            loss_numerator=huber_sum,
            loss_denominator=count,
            component_numerators={
                "huber": huber_sum,
                "raw_cc64_absolute_error": absolute_error.sum(),
                "raw_cc64_squared_error": squared_error.sum(),
            },
            component_denominators={
                "huber": count,
                "raw_cc64_absolute_error": count,
                "raw_cc64_squared_error": count,
            },
            encoder_hidden_states=hidden,
        )
