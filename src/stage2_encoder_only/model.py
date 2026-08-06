"""Encoder-only Stage 2 sustain-pedal classifier."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


TOKENS_PER_NOTE = 8
PEDAL_SLOTS = 4
PEDAL_NUM_CLASSES = 128
IGNORE_INDEX = -100


@dataclass
class Stage2PedalOutput:
    """Structured output for Stage 2 pedal classification."""

    logits: torch.Tensor
    loss: torch.Tensor | None = None
    hidden_states: torch.Tensor | None = None


class Stage2PedalEncoderModel(nn.Module):
    """Apply four independent pedal heads to pretrained PT encoder states."""

    def __init__(
        self,
        encoder: nn.Module,
        hidden_size: int | None = None,
        freeze_encoder: bool = False,
        dropout: float | None = None,
    ) -> None:
        super().__init__()
        if hidden_size is None:
            encoder_config = getattr(encoder, "config", None)
            hidden_size = getattr(encoder_config, "hidden_size", None)
        if hidden_size is None or hidden_size <= 0:
            raise ValueError("hidden_size must be positive or available on encoder.config")

        encoder_config = getattr(encoder, "config", None)
        if dropout is None:
            dropout = float(getattr(encoder_config, "dropout_rate", 0.0))
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be in [0, 1]")

        self.encoder = encoder
        self.hidden_size = int(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classification_heads = nn.ModuleList(
            [
                nn.Linear(self.hidden_size, PEDAL_NUM_CLASSES)
                for _ in range(PEDAL_SLOTS)
            ]
        )
        self.set_encoder_frozen(freeze_encoder)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str | Path,
        freeze_encoder: bool = False,
        dropout: float | None = None,
        **pretrained_kwargs: Any,
    ) -> "Stage2PedalEncoderModel":
        """Load the official full PT checkpoint and retain only its encoder."""

        from third_party.PianistTransformer.src.model.pianoformer import (
            PianoT5Gemma,
        )

        full_model = PianoT5Gemma.from_pretrained(
            str(checkpoint_path),
            **pretrained_kwargs,
        )
        encoder = full_model.get_encoder()
        hidden_size = int(encoder.config.hidden_size)

        # Detach the retained encoder so deleting the full wrapper releases the
        # decoder, LM head, and the wrapper's unused embedding module.
        full_model.model.encoder = None
        del full_model
        return cls(
            encoder=encoder,
            hidden_size=hidden_size,
            freeze_encoder=freeze_encoder,
            dropout=dropout,
        )

    def set_encoder_frozen(self, freeze: bool) -> None:
        """Enable or disable encoder parameter gradients."""

        self.freeze_encoder = freeze
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(not freeze)

    @property
    def trainable_encoder_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.encoder.parameters()
            if parameter.requires_grad
        )

    @property
    def prediction_head_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.classification_heads.parameters())

    @staticmethod
    def compute_loss(
        logits: torch.Tensor,
        pedal_targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute unweighted cross-entropy, ignoring padded target slots."""

        return F.cross_entropy(
            logits.reshape(-1, PEDAL_NUM_CLASSES),
            pedal_targets.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        token_attention_mask: torch.Tensor,
        pedal_targets: torch.Tensor | None = None,
        note_mask: torch.Tensor | None = None,
    ) -> Stage2PedalOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [B, N * 8]")
        batch_size, flat_length = input_ids.shape
        if flat_length % TOKENS_PER_NOTE:
            raise ValueError("input_ids flat token length must be divisible by 8")
        if token_attention_mask.shape != input_ids.shape:
            raise ValueError("token_attention_mask must have the same shape as input_ids")

        note_count = flat_length // TOKENS_PER_NOTE
        expected_targets_shape = (batch_size, note_count, PEDAL_SLOTS)
        expected_note_mask_shape = (batch_size, note_count)
        if pedal_targets is not None and tuple(pedal_targets.shape) != expected_targets_shape:
            raise ValueError(
                f"pedal_targets must have shape {expected_targets_shape}"
            )
        if note_mask is not None and tuple(note_mask.shape) != expected_note_mask_shape:
            raise ValueError(f"note_mask must have shape {expected_note_mask_shape}")

        if pedal_targets is not None:
            valid_targets = (pedal_targets == IGNORE_INDEX) | (
                (pedal_targets >= 0) & (pedal_targets < PEDAL_NUM_CLASSES)
            )
            if not bool(torch.all(valid_targets)):
                raise ValueError("pedal_targets must be in [0, 127] or -100")
        if pedal_targets is not None and note_mask is not None:
            expected_valid = note_mask.to(dtype=torch.bool).unsqueeze(-1).expand_as(
                pedal_targets
            )
            actual_valid = pedal_targets != IGNORE_INDEX
            if not torch.equal(expected_valid, actual_valid):
                raise ValueError(
                    "note_mask and pedal_targets -100 padding are inconsistent"
                )

        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=token_attention_mask,
        )
        hidden_states = getattr(encoder_output, "last_hidden_state", None)
        if hidden_states is None:
            raise ValueError("encoder output must provide last_hidden_state")
        expected_hidden_shape = (batch_size, note_count, self.hidden_size)
        if tuple(hidden_states.shape) != expected_hidden_shape:
            raise ValueError(
                "encoder last_hidden_state must have shape "
                f"{expected_hidden_shape}, got {tuple(hidden_states.shape)}"
            )

        dropped_states = self.dropout(hidden_states)
        logits = torch.stack(
            [head(dropped_states) for head in self.classification_heads],
            dim=2,
        )
        expected_logits_shape = (
            batch_size,
            note_count,
            PEDAL_SLOTS,
            PEDAL_NUM_CLASSES,
        )
        if tuple(logits.shape) != expected_logits_shape:
            raise ValueError(
                f"logits must have shape {expected_logits_shape}, "
                f"got {tuple(logits.shape)}"
            )

        loss = None
        if pedal_targets is not None:
            loss = self.compute_loss(logits, pedal_targets)
        return Stage2PedalOutput(
            logits=logits,
            loss=loss,
            hidden_states=hidden_states,
        )
