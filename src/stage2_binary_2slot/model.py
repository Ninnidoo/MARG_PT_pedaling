"""Stateless pretrained-PT wrapper and shared Binary 2-Slot heads."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn

from src.stage2_binary.model import BinaryStage2EncoderBase


COUNT_CLASSES = 3
TIMING_SLOTS = 2
HEAD_INIT_SEED = 42


@dataclass
class Binary2SlotHeadOutput:
    count_logits: torch.Tensor
    timing_predictions: torch.Tensor
    conditioned_states: torch.Tensor


@dataclass
class Binary2SlotMainEncoding:
    encoder_hidden_states: torch.Tensor
    owned_onset_hidden_states: torch.Tensor


class StateConditionedBinary2SlotModel(BinaryStage2EncoderBase):
    """Official PT encoder plus scalar-state-conditioned shared linear heads.

    The model intentionally stores neither human targets nor a current pedal
    state. State lifetime and hard recurrence belong to the rollout/trainer.
    """

    target_name = "binary_2slot_online_targets"

    def __init__(
        self,
        encoder: nn.Module,
        hidden_size: int | None = None,
        dropout: float | None = None,
    ) -> None:
        super().__init__(encoder, hidden_size, dropout)
        conditioned_size = self.hidden_size + 1
        self.count_head = nn.Linear(conditioned_size, COUNT_CLASSES)
        self.timing_head_1 = nn.Linear(conditioned_size, 1)
        self.timing_head_2 = nn.Linear(conditioned_size, 1)
        self.head_init_seed: int | None = None
        self.checkpoint_path: str | None = None

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str | Path,
        dropout: float | None = None,
        *,
        head_init_seed: int = HEAD_INIT_SEED,
        **pretrained_kwargs: Any,
    ) -> "StateConditionedBinary2SlotModel":
        if int(head_init_seed) != HEAD_INIT_SEED:
            raise ValueError("Binary 2-Slot v0 freezes head_init_seed=42")
        from third_party.PianistTransformer.src.model.pianoformer import PianoT5Gemma

        full_model = PianoT5Gemma.from_pretrained(str(checkpoint_path), **pretrained_kwargs)
        encoder = full_model.get_encoder()
        hidden_size = int(encoder.config.hidden_size)
        full_model.model.encoder = None
        del full_model
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(HEAD_INIT_SEED)
            model = cls(encoder=encoder, hidden_size=hidden_size, dropout=dropout)
        model.head_init_seed = HEAD_INIT_SEED
        model.checkpoint_path = str(Path(checkpoint_path).resolve())
        if model.prediction_head_parameter_count != 3850:
            raise AssertionError("Binary 2-Slot head parameter count must equal 3,850")
        return model

    def prediction_head_parameters(self) -> Iterable[nn.Parameter]:
        for module in (self.count_head, self.timing_head_1, self.timing_head_2):
            yield from module.parameters()

    @property
    def head_parameter_counts(self) -> dict[str, int]:
        return {
            "count": sum(parameter.numel() for parameter in self.count_head.parameters()),
            "timing_1": sum(parameter.numel() for parameter in self.timing_head_1.parameters()),
            "timing_2": sum(parameter.numel() for parameter in self.timing_head_2.parameters()),
            "total": self.prediction_head_parameter_count,
        }

    @staticmethod
    def _assert_no_human_pedal_leakage(
        input_ids: torch.Tensor, token_attention_mask: torch.Tensor
    ) -> None:
        if input_ids.ndim != 2 or input_ids.shape != token_attention_mask.shape:
            raise ValueError("input IDs and attention mask must have matching [B,N*8] shape")
        if input_ids.shape[1] % 8:
            raise ValueError("input IDs must contain complete PT notes")
        notes = input_ids.reshape(input_ids.shape[0], -1, 8)
        active = token_attention_mask.reshape(input_ids.shape[0], -1, 8).bool().any(dim=-1)
        if bool(torch.any(notes[:, :, 4:][active] != 1)):
            raise AssertionError("active encoder input exposes human Pedal1-4 tokens")

    @staticmethod
    def _gather_positions(
        hidden: torch.Tensor,
        positions: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if positions.shape != valid_mask.shape or positions.ndim not in {1, 2}:
            raise ValueError("positions/mask must have matching [B] or [B,O] shape")
        if positions.shape[0] != hidden.shape[0]:
            raise ValueError("position batch differs from encoder batch")
        note_count = hidden.shape[1]
        if bool(torch.any(valid_mask & ((positions < 0) | (positions >= note_count)))):
            raise ValueError("valid gather position lies outside encoder sequence")
        safe = positions.clamp(min=0, max=max(0, note_count - 1))
        if positions.ndim == 1:
            return hidden[torch.arange(hidden.shape[0], device=hidden.device), safe]
        return torch.gather(
            hidden,
            1,
            safe.unsqueeze(-1).expand(-1, -1, hidden.shape[-1]),
        )

    def encode_main(
        self,
        input_ids: torch.Tensor,
        token_attention_mask: torch.Tensor,
        note_mask: torch.Tensor,
        owned_representative_positions: torch.Tensor,
        owned_onset_mask: torch.Tensor,
    ) -> Binary2SlotMainEncoding:
        self._assert_no_human_pedal_leakage(input_ids, token_attention_mask)
        hidden, _, _ = self._encode(input_ids, token_attention_mask, note_mask)
        owned = self._gather_positions(hidden, owned_representative_positions, owned_onset_mask)
        return Binary2SlotMainEncoding(hidden, owned)

    def encode_boundary(
        self,
        input_ids: torch.Tensor,
        token_attention_mask: torch.Tensor,
        note_mask: torch.Tensor,
        query_positions: torch.Tensor,
    ) -> torch.Tensor:
        self._assert_no_human_pedal_leakage(input_ids, token_attention_mask)
        hidden, batch_size, _ = self._encode(input_ids, token_attention_mask, note_mask)
        if tuple(query_positions.shape) != (batch_size,):
            raise ValueError("boundary query positions must have shape [B]")
        valid = torch.ones_like(query_positions, dtype=torch.bool)
        return self._gather_positions(hidden, query_positions, valid)

    def condition_and_predict(
        self,
        hidden_states: torch.Tensor,
        current_states: torch.Tensor,
    ) -> Binary2SlotHeadOutput:
        expected_state_shape = hidden_states.shape[:-1]
        if tuple(current_states.shape) != tuple(expected_state_shape):
            raise ValueError("current state shape must match hidden leading dimensions")
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError("hidden state dimension differs from pretrained encoder")
        if not bool(torch.all((current_states == 0) | (current_states == 1))):
            raise ValueError("current state must contain only OFF=0/ON=1")
        # Dropout applies only to h. The hard scalar state remains untouched.
        conditioned = torch.cat(
            (self.dropout(hidden_states), current_states.to(hidden_states.dtype).unsqueeze(-1)),
            dim=-1,
        )
        if conditioned.shape[-1] != self.hidden_size + 1:
            raise AssertionError("conditioned representation must have dimension H+1")
        timing = torch.cat(
            (self.timing_head_1(conditioned), self.timing_head_2(conditioned)), dim=-1
        )
        return Binary2SlotHeadOutput(
            count_logits=self.count_head(conditioned),
            timing_predictions=timing,
            conditioned_states=conditioned,
        )
