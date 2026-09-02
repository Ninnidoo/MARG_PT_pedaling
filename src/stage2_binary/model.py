"""Binary encoder-only Stage 2 models sharing the official PT encoder path."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch import nn


TOKENS_PER_NOTE = 8
PEDAL_SLOTS = 4
BINARY_CLASSES = 2
JOINT_CLASSES = 16
IGNORE_INDEX = -100


@dataclass
class BinaryStage2Output:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    hidden_states: torch.Tensor | None = None


class BinaryStage2EncoderBase(nn.Module):
    """Common official-encoder loading and input/mask validation."""

    target_name: str

    def __init__(
        self,
        encoder: nn.Module,
        hidden_size: int | None = None,
        dropout: float | None = None,
    ) -> None:
        super().__init__()
        encoder_config = getattr(encoder, "config", None)
        resolved_hidden_size = hidden_size or getattr(
            encoder_config, "hidden_size", None
        )
        if resolved_hidden_size is None or int(resolved_hidden_size) <= 0:
            raise ValueError(
                "hidden_size must be positive or available on encoder.config"
            )
        if dropout is None:
            dropout = float(getattr(encoder_config, "dropout_rate", 0.0))
        if not 0.0 <= float(dropout) <= 1.0:
            raise ValueError("dropout must be in [0,1]")
        self.encoder = encoder
        self.hidden_size = int(resolved_hidden_size)
        self.dropout = nn.Dropout(float(dropout))
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(True)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str | Path,
        dropout: float | None = None,
        **pretrained_kwargs: Any,
    ) -> "BinaryStage2EncoderBase":
        """Load the official full PT checkpoint and retain its encoder only."""

        from third_party.PianistTransformer.src.model.pianoformer import (
            PianoT5Gemma,
        )

        full_model = PianoT5Gemma.from_pretrained(
            str(checkpoint_path), **pretrained_kwargs
        )
        encoder = full_model.get_encoder()
        hidden_size = int(encoder.config.hidden_size)
        full_model.model.encoder = None
        del full_model
        return cls(encoder=encoder, hidden_size=hidden_size, dropout=dropout)

    @property
    def encoder_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.encoder.parameters())

    @property
    def prediction_head_parameter_count(self) -> int:
        return sum(
            parameter.numel() for parameter in self.prediction_head_parameters()
        )

    @property
    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def prediction_head_parameters(self) -> Iterable[nn.Parameter]:
        raise NotImplementedError

    def _encode(
        self,
        input_ids: torch.Tensor,
        token_attention_mask: torch.Tensor,
        note_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, int, int]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [B,N*8]")
        batch_size, flat_length = input_ids.shape
        if flat_length % TOKENS_PER_NOTE:
            raise ValueError("input_ids length must contain complete 8-token notes")
        if token_attention_mask.shape != input_ids.shape:
            raise ValueError("token_attention_mask must match input_ids")
        note_count = flat_length // TOKENS_PER_NOTE
        if note_mask is not None and tuple(note_mask.shape) != (
            batch_size,
            note_count,
        ):
            raise ValueError("note_mask must have shape [B,N]")
        encoded = self.encoder(
            input_ids=input_ids,
            attention_mask=token_attention_mask,
        )
        hidden_states = getattr(encoded, "last_hidden_state", None)
        expected = (batch_size, note_count, self.hidden_size)
        if hidden_states is None or tuple(hidden_states.shape) != expected:
            actual = None if hidden_states is None else tuple(hidden_states.shape)
            raise ValueError(
                f"encoder last_hidden_state must have shape {expected}, got {actual}"
            )
        return hidden_states, batch_size, note_count

    @staticmethod
    def _validate_mask_alignment(
        targets: torch.Tensor,
        note_mask: torch.Tensor | None,
        expected_shape: tuple[int, ...],
        upper_bound: int,
    ) -> None:
        if tuple(targets.shape) != expected_shape:
            raise ValueError(f"targets must have shape {expected_shape}")
        valid_values = (targets == IGNORE_INDEX) | (
            (targets >= 0) & (targets < upper_bound)
        )
        if not bool(torch.all(valid_values)):
            raise ValueError(
                f"targets must be in [0,{upper_bound - 1}] or {IGNORE_INDEX}"
            )
        if not bool(torch.any(targets != IGNORE_INDEX)):
            raise ValueError("loss requires at least one valid target")
        if note_mask is not None:
            expected_valid = note_mask.to(dtype=torch.bool)
            while expected_valid.ndim < targets.ndim:
                expected_valid = expected_valid.unsqueeze(-1)
            expected_valid = expected_valid.expand_as(targets)
            if not torch.equal(expected_valid, targets != IGNORE_INDEX):
                raise ValueError("note_mask and target padding are inconsistent")


class IndependentBinaryPedalModel(BinaryStage2EncoderBase):
    """Official PT encoder plus four independent Linear(H,2) heads."""

    target_name = "binary_targets"

    def __init__(
        self,
        encoder: nn.Module,
        hidden_size: int | None = None,
        dropout: float | None = None,
    ) -> None:
        super().__init__(encoder, hidden_size, dropout)
        self.classification_heads = nn.ModuleList(
            [nn.Linear(self.hidden_size, BINARY_CLASSES) for _ in range(PEDAL_SLOTS)]
        )

    def prediction_head_parameters(self) -> Iterable[nn.Parameter]:
        return self.classification_heads.parameters()

    @staticmethod
    def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits.reshape(-1, BINARY_CLASSES),
            targets.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        token_attention_mask: torch.Tensor,
        binary_targets: torch.Tensor | None = None,
        note_mask: torch.Tensor | None = None,
    ) -> BinaryStage2Output:
        hidden, batch_size, note_count = self._encode(
            input_ids, token_attention_mask, note_mask
        )
        if binary_targets is not None:
            self._validate_mask_alignment(
                binary_targets,
                note_mask,
                (batch_size, note_count, PEDAL_SLOTS),
                BINARY_CLASSES,
            )
        states = self.dropout(hidden)
        logits = torch.stack(
            [head(states) for head in self.classification_heads], dim=2
        )
        expected = (batch_size, note_count, PEDAL_SLOTS, BINARY_CLASSES)
        if tuple(logits.shape) != expected:
            raise AssertionError(f"independent logits must have shape {expected}")
        loss = (
            None
            if binary_targets is None
            else self.compute_loss(logits, binary_targets)
        )
        return BinaryStage2Output(logits=logits, loss=loss, hidden_states=hidden)


class JointBinaryPedalModel(BinaryStage2EncoderBase):
    """Official PT encoder plus one Linear(H,16) joint-pattern head."""

    target_name = "joint_targets"

    def __init__(
        self,
        encoder: nn.Module,
        hidden_size: int | None = None,
        dropout: float | None = None,
    ) -> None:
        super().__init__(encoder, hidden_size, dropout)
        self.classification_head = nn.Linear(self.hidden_size, JOINT_CLASSES)

    def prediction_head_parameters(self) -> Iterable[nn.Parameter]:
        return self.classification_head.parameters()

    @staticmethod
    def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits.reshape(-1, JOINT_CLASSES),
            targets.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        token_attention_mask: torch.Tensor,
        joint_targets: torch.Tensor | None = None,
        note_mask: torch.Tensor | None = None,
    ) -> BinaryStage2Output:
        hidden, batch_size, note_count = self._encode(
            input_ids, token_attention_mask, note_mask
        )
        if joint_targets is not None:
            self._validate_mask_alignment(
                joint_targets,
                note_mask,
                (batch_size, note_count),
                JOINT_CLASSES,
            )
        logits = self.classification_head(self.dropout(hidden))
        expected = (batch_size, note_count, JOINT_CLASSES)
        if tuple(logits.shape) != expected:
            raise AssertionError(f"joint logits must have shape {expected}")
        loss = (
            None if joint_targets is None else self.compute_loss(logits, joint_targets)
        )
        return BinaryStage2Output(logits=logits, loss=loss, hidden_states=hidden)
