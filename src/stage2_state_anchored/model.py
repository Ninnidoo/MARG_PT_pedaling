"""Pretrained-PT encoder and parallel State-Anchored v1 heads."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn

from src.stage2_binary.model import BinaryStage2EncoderBase

from .config import HIDDEN_SIZE, SEED


@dataclass
class StateAnchoredModelOutput:
    state_logits: torch.Tensor
    mode_logits: torch.Tensor
    timing_predictions: torch.Tensor
    encoder_hidden_states: torch.Tensor
    owned_onset_hidden_states: torch.Tensor


class StateAnchoredTransitionModelV1(BinaryStage2EncoderBase):
    """PT encoder with four direct, non-recurrent linear prediction heads."""

    target_name = "state_anchored_transition_v1"

    def __init__(self, encoder: nn.Module, hidden_size: int | None = None, dropout: float | None = None) -> None:
        super().__init__(encoder, hidden_size, dropout)
        self.state_head = nn.Linear(self.hidden_size, 2)
        self.mode_head = nn.Linear(self.hidden_size, 3)
        self.timing_head_1 = nn.Linear(self.hidden_size, 1)
        self.timing_head_2 = nn.Linear(self.hidden_size, 1)
        self.head_init_seed: int | None = None
        self.checkpoint_path: str | None = None

    @classmethod
    def from_pretrained(cls, checkpoint_path: str | Path, dropout: float | None = None, *, head_init_seed: int = SEED, **pretrained_kwargs: Any) -> "StateAnchoredTransitionModelV1":
        if int(head_init_seed) != SEED:
            raise ValueError("State-Anchored v1 freezes head_init_seed=42")
        from third_party.PianistTransformer.src.model.pianoformer import PianoT5Gemma

        full_model = PianoT5Gemma.from_pretrained(str(checkpoint_path), **pretrained_kwargs)
        encoder = full_model.get_encoder()
        hidden_size = int(encoder.config.hidden_size)
        full_model.model.encoder = None
        del full_model
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(SEED)
            model = cls(encoder=encoder, hidden_size=hidden_size, dropout=dropout)
        model.head_init_seed = SEED
        model.checkpoint_path = str(Path(checkpoint_path).resolve())
        if hidden_size == HIDDEN_SIZE and model.prediction_head_parameter_count != 5_383:
            raise AssertionError("State-Anchored v1 head parameter count must equal 5,383")
        return model

    def prediction_head_parameters(self) -> Iterable[nn.Parameter]:
        for module in (self.state_head, self.mode_head, self.timing_head_1, self.timing_head_2):
            yield from module.parameters()

    @property
    def head_parameter_counts(self) -> dict[str, int]:
        return {
            "state": sum(parameter.numel() for parameter in self.state_head.parameters()),
            "mode": sum(parameter.numel() for parameter in self.mode_head.parameters()),
            "timing_1": sum(parameter.numel() for parameter in self.timing_head_1.parameters()),
            "timing_2": sum(parameter.numel() for parameter in self.timing_head_2.parameters()),
            "total": self.prediction_head_parameter_count,
        }

    @staticmethod
    def _assert_no_human_pedal_leakage(input_ids: torch.Tensor, token_attention_mask: torch.Tensor) -> None:
        if input_ids.ndim != 2 or input_ids.shape != token_attention_mask.shape:
            raise ValueError("input IDs/mask must have matching [B,N*8] shape")
        if input_ids.shape[1] % 8:
            raise ValueError("input IDs must contain complete PT notes")
        notes = input_ids.reshape(input_ids.shape[0], -1, 8)
        active = token_attention_mask.reshape(input_ids.shape[0], -1, 8).bool().any(-1)
        if bool(torch.any(notes[:, :, 4:][active] != 1)):
            raise AssertionError("active encoder input exposes human Pedal1-4 tokens")

    @staticmethod
    def _gather_owned(hidden: torch.Tensor, positions: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if positions.ndim != 2 or positions.shape != valid_mask.shape:
            raise ValueError("owned positions/mask must have matching [B,O] shape")
        if positions.shape[0] != hidden.shape[0]:
            raise ValueError("owned position batch differs from encoder batch")
        note_count = hidden.shape[1]
        if bool(torch.any(valid_mask & ((positions < 0) | (positions >= note_count)))):
            raise ValueError("valid representative position lies outside note window")
        safe = positions.clamp(min=0, max=max(0, note_count - 1))
        return torch.gather(hidden, 1, safe.unsqueeze(-1).expand(-1, -1, hidden.shape[-1]))

    def forward(self, input_ids: torch.Tensor, token_attention_mask: torch.Tensor, note_mask: torch.Tensor, owned_representative_positions: torch.Tensor, owned_onset_mask: torch.Tensor) -> StateAnchoredModelOutput:
        self._assert_no_human_pedal_leakage(input_ids, token_attention_mask)
        hidden, batch_size, _ = self._encode(input_ids, token_attention_mask, note_mask)
        if owned_representative_positions.shape[0] != batch_size:
            raise ValueError("owned representative batch mismatch")
        owned_hidden = self._gather_owned(hidden, owned_representative_positions, owned_onset_mask.bool())
        # All heads see only h_i. No human/predicted state enters the graph.
        states = self.dropout(owned_hidden)
        state_logits = self.state_head(states)
        mode_logits = self.mode_head(states)
        timing = torch.cat((self.timing_head_1(states), self.timing_head_2(states)), dim=-1)
        expected = (batch_size, owned_hidden.shape[1])
        if tuple(state_logits.shape) != (*expected, 2) or tuple(mode_logits.shape) != (*expected, 3) or tuple(timing.shape) != (*expected, 2):
            raise AssertionError("State-Anchored head output shape mismatch")
        return StateAnchoredModelOutput(state_logits, mode_logits, timing, hidden, owned_hidden)
