"""Custom Event Model v1 B3-S: detached soft semantic pre-state features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn

from .dataset import EVENT_CLASSES, INITIAL_CLASSES, MAIN_SLOTS, TERMINAL_SLOTS
from .model import CustomEventEncoderModelV0


STATE_CLASSES = 4
STATE_FEATURE_DIM = 4
MODEL_VERSION = "1.0.0-b3s"


@dataclass
class CustomEventStateConditionedOutput:
    main_event_logits: torch.Tensor
    main_timing_predictions: torch.Tensor
    main_state_logits: torch.Tensor
    main_state_posteriors: torch.Tensor
    initial_logits: torch.Tensor
    terminal_event_logits: torch.Tensor
    terminal_timing_predictions: torch.Tensor
    encoder_hidden_states: torch.Tensor
    owned_onset_hidden_states: torch.Tensor


class CustomEventEncoderModelV1StateConditioned(CustomEventEncoderModelV0):
    """Frozen B3-S formulation without hard state rollout or teacher forcing."""

    target_name = "custom_event_state_conditioned_targets"
    model_version = MODEL_VERSION

    def __init__(
        self,
        encoder: nn.Module,
        hidden_size: int | None = None,
        dropout: float | None = None,
    ) -> None:
        # Construct every legacy module first in exact v0 order.
        super().__init__(encoder, hidden_size, dropout)
        legacy_main_event_heads = self.main_event_heads
        expanded = nn.ModuleList(
            [nn.Linear(self.hidden_size + STATE_FEATURE_DIM, EVENT_CLASSES) for _ in range(MAIN_SLOTS)]
        )
        with torch.no_grad():
            for old, new in zip(legacy_main_event_heads, expanded, strict=True):
                new.weight[:, : self.hidden_size].copy_(old.weight)
                new.weight[:, self.hidden_size :].zero_()
                new.bias.copy_(old.bias)
        self.main_event_heads = expanded
        self.main_state_heads = nn.ModuleList(
            [nn.Linear(self.hidden_size, STATE_CLASSES) for _ in range(MAIN_SLOTS)]
        )

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str | Path,
        dropout: float | None = None,
        *,
        head_init_seed: int = 42,
        **pretrained_kwargs: Any,
    ) -> "CustomEventEncoderModelV1StateConditioned":
        model = super().from_pretrained(
            checkpoint_path,
            dropout,
            head_init_seed=head_init_seed,
            **pretrained_kwargs,
        )
        if not isinstance(model, cls):
            raise AssertionError("v1 pretrained constructor returned the wrong class")
        return model

    def prediction_head_parameters(self) -> Iterable[nn.Parameter]:
        for module in (
            self.initial_head,
            self.main_event_heads,
            self.main_timing_heads,
            self.terminal_event_heads,
            self.terminal_timing_heads,
            self.main_state_heads,
        ):
            yield from module.parameters()

    @property
    def head_parameter_counts(self) -> dict[str, int]:
        counts = super().head_parameter_counts
        counts["main_state"] = sum(
            parameter.numel() for parameter in self.main_state_heads.parameters()
        )
        return counts

    def _main_predictions(
        self, owned_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state_input = owned_states.detach()
        state_logits = torch.stack(
            [head(state_input) for head in self.main_state_heads], dim=2
        )
        state_posteriors = torch.softmax(state_logits.float(), dim=-1).to(
            state_logits.dtype
        )
        event_logits = self._event_predictions(owned_states, state_posteriors)
        tensors = (state_logits, state_posteriors, event_logits)
        if not all(bool(torch.isfinite(value).all()) for value in tensors):
            raise FloatingPointError("B3-S Main state/event prediction is non-finite")
        return state_logits, state_posteriors, event_logits

    def _event_predictions(
        self, owned_states: torch.Tensor, state_posteriors: torch.Tensor
    ) -> torch.Tensor:
        """Apply detached posterior features; exposed for connectivity tests."""

        expected = (*owned_states.shape[:-1], MAIN_SLOTS, STATE_CLASSES)
        if tuple(state_posteriors.shape) != expected:
            raise ValueError(f"state_posteriors must have shape {expected}")
        return torch.stack(
            [
                head(
                    torch.cat(
                        (owned_states, state_posteriors[:, :, slot].detach()), dim=-1
                    )
                )
                for slot, head in enumerate(self.main_event_heads)
            ],
            dim=2,
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
    ) -> CustomEventStateConditionedOutput:
        hidden, batch_size, _ = self._encode(input_ids, token_attention_mask, note_mask)
        if owned_representative_positions.shape[0] != batch_size:
            raise ValueError("owned representative batch mismatch")
        owned_hidden = self._gather_positions(
            hidden, owned_representative_positions, owned_onset_mask, name="owned onset"
        )
        initial_hidden = self._gather_positions(
            hidden, initial_representative_positions, initial_valid_mask, name="initial"
        )
        terminal_hidden = self._gather_positions(
            hidden, terminal_representative_positions, terminal_valid_mask, name="terminal"
        )
        owned_states = self.dropout(owned_hidden)
        initial_states = self.dropout(initial_hidden)
        terminal_states = self.dropout(terminal_hidden)
        state_logits, state_posteriors, main_event_logits = self._main_predictions(owned_states)
        main_timing = torch.stack(
            [head(owned_states).squeeze(-1) for head in self.main_timing_heads], dim=2
        )
        terminal_event_logits = torch.stack(
            [head(terminal_states) for head in self.terminal_event_heads], dim=1
        )
        terminal_timing = torch.stack(
            [head(terminal_states).squeeze(-1) for head in self.terminal_timing_heads], dim=1
        )
        expected = (batch_size, owned_hidden.shape[1])
        if tuple(state_logits.shape) != (*expected, MAIN_SLOTS, STATE_CLASSES):
            raise AssertionError("Main state logits must have shape [B,O,6,4]")
        if tuple(state_posteriors.shape) != tuple(state_logits.shape):
            raise AssertionError("Main state posterior shape mismatch")
        if tuple(main_event_logits.shape) != (*expected, MAIN_SLOTS, EVENT_CLASSES):
            raise AssertionError("Main event logits must have shape [B,O,6,5]")
        if tuple(main_timing.shape) != (*expected, MAIN_SLOTS):
            raise AssertionError("Main timing shape changed")
        if tuple(terminal_event_logits.shape) != (batch_size, TERMINAL_SLOTS, EVENT_CLASSES):
            raise AssertionError("Terminal event shape changed")
        if tuple(terminal_timing.shape) != (batch_size, TERMINAL_SLOTS):
            raise AssertionError("Terminal timing shape changed")
        return CustomEventStateConditionedOutput(
            main_event_logits=main_event_logits,
            main_timing_predictions=main_timing,
            main_state_logits=state_logits,
            main_state_posteriors=state_posteriors,
            initial_logits=self.initial_head(initial_states),
            terminal_event_logits=terminal_event_logits,
            terminal_timing_predictions=terminal_timing,
            encoder_hidden_states=hidden,
            owned_onset_hidden_states=owned_hidden,
        )
