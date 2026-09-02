"""Raw-CC64 Huber regression with training-only canonical CE supervision."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import chain
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from src.stage2_encoder_only.dataset import PEDAL_SLOTS

from .loss_objectives import HUBER_DELTA_NORMALIZED, ObjectiveOutput
from .raw_cc64_huber import RawCC64HuberEncoderModel, raw_cc64_to_normalized_targets
from .representation import IGNORE_INDEX, NUM_CLASSES, classify_cc64


LAMBDA_CE = 0.1


@dataclass
class RawHuberAuxCEOutput(ObjectiveOutput):
    """The inherited ``predictions`` field is always regression-only output."""

    auxiliary_logits: torch.Tensor | None = None
    component_losses: dict[str, torch.Tensor] = field(default_factory=dict)


class RawHuberAuxCEEncoderModel(RawCC64HuberEncoderModel):
    """Canonical scalar heads plus four fresh training-only 4-class heads."""

    def __init__(
        self,
        encoder: nn.Module,
        hidden_size: int | None = None,
        dropout: float | None = None,
        *,
        delta: float = HUBER_DELTA_NORMALIZED,
        lambda_ce: float = LAMBDA_CE,
    ) -> None:
        super().__init__(encoder, hidden_size, dropout, delta=delta)
        self.lambda_ce = float(lambda_ce)
        if self.lambda_ce != LAMBDA_CE:
            raise ValueError("canonical auxiliary CE coefficient must be 0.1")
        self.classification_heads = nn.ModuleList(
            [nn.Linear(self.hidden_size, NUM_CLASSES) for _ in range(PEDAL_SLOTS)]
        )

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        *,
        delta: float = HUBER_DELTA_NORMALIZED,
        lambda_ce: float = LAMBDA_CE,
        dropout: float | None = None,
        **pretrained_kwargs: object,
    ) -> "RawHuberAuxCEEncoderModel":
        from third_party.PianistTransformer.src.model.pianoformer import PianoT5Gemma

        full_model = PianoT5Gemma.from_pretrained(str(checkpoint_path), **pretrained_kwargs)
        encoder = full_model.get_encoder()
        hidden_size = int(encoder.config.hidden_size)
        full_model.model.encoder = None
        del full_model
        return cls(
            encoder=encoder,
            hidden_size=hidden_size,
            dropout=dropout,
            delta=delta,
            lambda_ce=lambda_ce,
        )

    def prediction_head_parameters(self) -> Iterable[nn.Parameter]:
        return chain(
            self.regression_heads.parameters(), self.classification_heads.parameters()
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        token_attention_mask: torch.Tensor,
        pedal_targets: torch.Tensor | None = None,
        note_mask: torch.Tensor | None = None,
    ) -> RawHuberAuxCEOutput:
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
        logits = torch.stack(
            [head(states) for head in self.classification_heads], dim=2
        )
        if tuple(values.shape) != (batch_size, note_count, PEDAL_SLOTS):
            raise AssertionError("regression output must have shape [B,N,4]")
        if tuple(logits.shape) != (batch_size, note_count, PEDAL_SLOTS, NUM_CLASSES):
            raise AssertionError("auxiliary logits must have shape [B,N,4,4]")
        if pedal_targets is None:
            return RawHuberAuxCEOutput(
                predictions=values,
                auxiliary_logits=logits,
                encoder_hidden_states=hidden,
            )

        valid = pedal_targets != IGNORE_INDEX
        count = valid.sum().to(values.dtype)
        if float(count.item()) <= 0.0:
            raise FloatingPointError("hybrid objective has no valid targets")
        regression_targets = raw_cc64_to_normalized_targets(pedal_targets).to(values.dtype)
        class_targets = classify_cc64(pedal_targets, allow_ignore=True)
        huber_sum = F.smooth_l1_loss(
            values[valid], regression_targets[valid], beta=self.delta, reduction="sum"
        )
        ce_sum = F.cross_entropy(
            logits.reshape(-1, NUM_CLASSES),
            class_targets.reshape(-1),
            ignore_index=IGNORE_INDEX,
            reduction="sum",
        )
        total_sum = huber_sum + self.lambda_ce * ce_sum
        component_losses = {
            "huber": huber_sum / count,
            "ce": ce_sum / count,
            "total": total_sum / count,
        }
        if not all(bool(torch.isfinite(value)) for value in component_losses.values()):
            raise FloatingPointError("raw Huber + auxiliary CE objective is non-finite")
        clipped_cc64 = values[valid].float().clamp(0.0, 1.0) * 127.0
        target_cc64 = pedal_targets[valid].float()
        return RawHuberAuxCEOutput(
            predictions=values,
            auxiliary_logits=logits,
            loss=component_losses["total"],
            loss_numerator=total_sum,
            loss_denominator=count,
            component_numerators={
                "huber": huber_sum,
                "ce": ce_sum,
                "total": total_sum,
                "raw_cc64_absolute_error": (clipped_cc64 - target_cc64).abs().sum(),
                "raw_cc64_squared_error": (clipped_cc64 - target_cc64).square().sum(),
            },
            component_denominators={
                key: count
                for key in (
                    "huber", "ce", "total", "raw_cc64_absolute_error",
                    "raw_cc64_squared_error",
                )
            },
            component_losses=component_losses,
            encoder_hidden_states=hidden,
        )
