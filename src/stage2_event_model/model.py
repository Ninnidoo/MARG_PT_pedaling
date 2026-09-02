"""Pretrained-PT Encoder-only Custom Event Model v0."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Iterable

import torch
from torch import nn

from src.stage2_binary.model import BinaryStage2EncoderBase

from .dataset import EVENT_CLASSES, INITIAL_CLASSES, MAIN_SLOTS, TERMINAL_SLOTS


@dataclass
class CustomEventModelOutput:
    main_event_logits: torch.Tensor
    main_timing_predictions: torch.Tensor
    initial_logits: torch.Tensor
    terminal_event_logits: torch.Tensor
    terminal_timing_predictions: torch.Tensor
    encoder_hidden_states: torch.Tensor
    owned_onset_hidden_states: torch.Tensor


class CustomEventEncoderModelV0(BinaryStage2EncoderBase):
    """Parallel Initial + Main K6 + Terminal K4 heads over PT note states."""

    target_name = "custom_event_targets"

    def __init__(
        self,
        encoder: nn.Module,
        hidden_size: int | None = None,
        dropout: float | None = None,
    ) -> None:
        super().__init__(encoder, hidden_size, dropout)
        self.initial_head = nn.Linear(self.hidden_size, INITIAL_CLASSES)
        self.main_event_heads = nn.ModuleList(
            [nn.Linear(self.hidden_size, EVENT_CLASSES) for _ in range(MAIN_SLOTS)]
        )
        self.main_timing_heads = nn.ModuleList(
            [nn.Linear(self.hidden_size, 1) for _ in range(MAIN_SLOTS)]
        )
        self.terminal_event_heads = nn.ModuleList(
            [nn.Linear(self.hidden_size, EVENT_CLASSES) for _ in range(TERMINAL_SLOTS)]
        )
        self.terminal_timing_heads = nn.ModuleList(
            [nn.Linear(self.hidden_size, 1) for _ in range(TERMINAL_SLOTS)]
        )
        self.head_init_seed: int | None = None

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str | Path,
        dropout: float | None = None,
        *,
        head_init_seed: int = 42,
        **pretrained_kwargs: Any,
    ) -> "CustomEventEncoderModelV0":
        from third_party.PianistTransformer.src.model.pianoformer import PianoT5Gemma

        full_model = PianoT5Gemma.from_pretrained(
            str(checkpoint_path), **pretrained_kwargs
        )
        encoder = full_model.get_encoder()
        hidden_size = int(encoder.config.hidden_size)
        full_model.model.encoder = None
        del full_model
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(head_init_seed))
            model = cls(encoder=encoder, hidden_size=hidden_size, dropout=dropout)
        model.head_init_seed = int(head_init_seed)
        return model

    def prediction_head_parameters(self) -> Iterable[nn.Parameter]:
        for module in (
            self.initial_head,
            self.main_event_heads,
            self.main_timing_heads,
            self.terminal_event_heads,
            self.terminal_timing_heads,
        ):
            yield from module.parameters()

    @property
    def head_parameter_counts(self) -> dict[str, int]:
        return {
            "initial": sum(parameter.numel() for parameter in self.initial_head.parameters()),
            "main_event": sum(parameter.numel() for parameter in self.main_event_heads.parameters()),
            "main_timing": sum(parameter.numel() for parameter in self.main_timing_heads.parameters()),
            "terminal_event": sum(parameter.numel() for parameter in self.terminal_event_heads.parameters()),
            "terminal_timing": sum(parameter.numel() for parameter in self.terminal_timing_heads.parameters()),
        }

    @staticmethod
    def _gather_positions(
        hidden: torch.Tensor,
        positions: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        name: str,
    ) -> torch.Tensor:
        if positions.shape != valid_mask.shape:
            raise ValueError(f"{name} positions and mask must have the same shape")
        note_count = hidden.shape[1]
        if bool(torch.any(valid_mask & ((positions < 0) | (positions >= note_count)))):
            raise ValueError(f"valid {name} position lies outside the note window")
        safe = positions.clamp(min=0, max=max(0, note_count - 1))
        if positions.ndim == 1:
            return hidden[torch.arange(hidden.shape[0], device=hidden.device), safe]
        if positions.ndim != 2:
            raise ValueError(f"{name} positions must be [B] or [B,O]")
        return torch.gather(
            hidden,
            1,
            safe.unsqueeze(-1).expand(-1, -1, hidden.shape[-1]),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        token_attention_mask: torch.Tensor,
        note_mask: torch.Tensor,
        owned_representative_positions: torch.Tensor,
        owned_onset_mask: torch.Tensor,
        initial_representative_positions: torch.Tensor,
        initial_valid_mask: torch.Tensor,
        terminal_representative_positions: torch.Tensor,
        terminal_valid_mask: torch.Tensor,
    ) -> CustomEventModelOutput:
        hidden, batch_size, _ = self._encode(
            input_ids, token_attention_mask, note_mask
        )
        if owned_representative_positions.shape[0] != batch_size:
            raise ValueError("owned representative batch mismatch")
        owned_hidden = self._gather_positions(
            hidden,
            owned_representative_positions,
            owned_onset_mask,
            name="owned onset",
        )
        initial_hidden = self._gather_positions(
            hidden,
            initial_representative_positions,
            initial_valid_mask,
            name="initial",
        )
        terminal_hidden = self._gather_positions(
            hidden,
            terminal_representative_positions,
            terminal_valid_mask,
            name="terminal",
        )
        owned_states = self.dropout(owned_hidden)
        initial_states = self.dropout(initial_hidden)
        terminal_states = self.dropout(terminal_hidden)
        main_event_logits = torch.stack(
            [head(owned_states) for head in self.main_event_heads], dim=2
        )
        main_timing = torch.stack(
            [head(owned_states).squeeze(-1) for head in self.main_timing_heads],
            dim=2,
        )
        terminal_event_logits = torch.stack(
            [head(terminal_states) for head in self.terminal_event_heads], dim=1
        )
        terminal_timing = torch.stack(
            [head(terminal_states).squeeze(-1) for head in self.terminal_timing_heads],
            dim=1,
        )
        expected_owned = (batch_size, owned_hidden.shape[1])
        if tuple(main_event_logits.shape) != (*expected_owned, MAIN_SLOTS, EVENT_CLASSES):
            raise AssertionError("main event output shape mismatch")
        if tuple(main_timing.shape) != (*expected_owned, MAIN_SLOTS):
            raise AssertionError("main timing output shape mismatch")
        if tuple(terminal_event_logits.shape) != (
            batch_size,
            TERMINAL_SLOTS,
            EVENT_CLASSES,
        ):
            raise AssertionError("terminal event output shape mismatch")
        if tuple(terminal_timing.shape) != (batch_size, TERMINAL_SLOTS):
            raise AssertionError("terminal timing output shape mismatch")
        return CustomEventModelOutput(
            main_event_logits=main_event_logits,
            main_timing_predictions=main_timing,
            initial_logits=self.initial_head(initial_states),
            terminal_event_logits=terminal_event_logits,
            terminal_timing_predictions=terminal_timing,
            encoder_hidden_states=hidden,
            owned_onset_hidden_states=owned_hidden,
        )
