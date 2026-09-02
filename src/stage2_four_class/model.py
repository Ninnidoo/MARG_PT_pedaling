"""Minimal canonical four-class adapters over the verified Stage 2 models."""

from __future__ import annotations

from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch import nn

from src.stage2_binary.model import BinaryStage2EncoderBase
from src.stage2_encoder_decoder.model import (
    EncoderDecoderFiveClassModel,
    EncoderDecoderFiveClassOutput,
    flatten_pedal_targets,
)
from src.stage2_encoder_only.dataset import PEDAL_SLOTS

from .representation import IGNORE_INDEX, NUM_CLASSES


BOS_ID = NUM_CLASSES
PAD_ID = NUM_CLASSES + 1
DECODER_VOCAB_SIZE = NUM_CLASSES + 2


def shift_right_four_class_targets(
    flat_targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create canonical ``BOS,y_1,...,y_(T-1)`` four-class history."""

    if flat_targets.ndim != 2 or flat_targets.shape[1] < 1:
        raise ValueError("flat targets must have non-empty shape [B,4N]")
    valid = flat_targets != IGNORE_INDEX
    illegal = valid & ((flat_targets < 0) | (flat_targets >= NUM_CLASSES))
    if bool(illegal.any()):
        raise ValueError("targets must be in [0,3] or -100")
    decoder_inputs = torch.full_like(flat_targets, PAD_ID)
    decoder_inputs[:, 0] = BOS_ID
    previous = flat_targets[:, :-1]
    decoder_inputs[:, 1:] = torch.where(
        previous == IGNORE_INDEX,
        torch.full_like(previous, PAD_ID),
        previous,
    )
    decoder_inputs = torch.where(
        valid, decoder_inputs, torch.full_like(decoder_inputs, PAD_ID)
    )
    return decoder_inputs, valid


class FourClassPedalEncoderModel(BinaryStage2EncoderBase):
    """Verified PT encoder path plus four independent Linear(H,4) heads."""

    target_name = "pedal_targets"

    def __init__(
        self,
        encoder: nn.Module,
        hidden_size: int | None = None,
        dropout: float | None = None,
    ) -> None:
        super().__init__(encoder, hidden_size, dropout)
        self.classification_heads = nn.ModuleList(
            [nn.Linear(self.hidden_size, NUM_CLASSES) for _ in range(PEDAL_SLOTS)]
        )

    def prediction_head_parameters(self) -> Iterable[nn.Parameter]:
        return self.classification_heads.parameters()

    @staticmethod
    def compute_loss(
        logits: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        numerator = F.cross_entropy(
            logits.reshape(-1, NUM_CLASSES),
            targets.reshape(-1),
            ignore_index=IGNORE_INDEX,
            reduction="sum",
        )
        denominator = (targets != IGNORE_INDEX).sum().to(logits.dtype)
        if not bool(torch.isfinite(numerator)) or float(denominator.item()) <= 0.0:
            raise FloatingPointError("unweighted four-class CE components are invalid")
        return numerator / denominator, numerator, denominator

    def forward(
        self,
        input_ids: torch.Tensor,
        token_attention_mask: torch.Tensor,
        pedal_targets: torch.Tensor | None = None,
        note_mask: torch.Tensor | None = None,
    ) -> EncoderDecoderFiveClassOutput:
        hidden, batch_size, note_count = self._encode(
            input_ids, token_attention_mask, note_mask
        )
        if pedal_targets is not None:
            self._validate_mask_alignment(
                pedal_targets,
                note_mask,
                (batch_size, note_count, PEDAL_SLOTS),
                NUM_CLASSES,
            )
        states = self.dropout(hidden)
        logits = torch.stack(
            [head(states) for head in self.classification_heads], dim=2
        )
        expected = (batch_size, note_count, PEDAL_SLOTS, NUM_CLASSES)
        if tuple(logits.shape) != expected:
            raise AssertionError(f"encoder-only logits must have shape {expected}")
        if pedal_targets is None:
            return EncoderDecoderFiveClassOutput(logits=logits, encoder_hidden_states=hidden)
        loss, numerator, denominator = self.compute_loss(logits, pedal_targets)
        return EncoderDecoderFiveClassOutput(
            logits=logits,
            loss=loss,
            loss_numerator=numerator,
            loss_denominator=denominator,
            encoder_hidden_states=hidden,
        )


class FourClassPedalEncoderDecoderModel(EncoderDecoderFiveClassModel):
    """The verified 5-class causal model with only its class vocabulary changed."""

    target_name = "pedal_targets"

    def __init__(
        self,
        encoder: nn.Module,
        decoder_config: Any,
        *,
        decoder_init_seed: int = 42,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__(
            encoder,
            decoder_config,
            decoder_init_seed=decoder_init_seed,
            freeze_encoder=freeze_encoder,
        )
        old_embedding = self.decoder.embed_tokens
        old_head = self.output_head
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(decoder_init_seed) + 4)
            embedding = nn.Embedding(NUM_CLASSES + 2, self.hidden_size)
            head = nn.Linear(self.hidden_size, NUM_CLASSES)
            nn.init.normal_(
                embedding.weight,
                mean=0.0,
                std=float(old_embedding.weight.detach().std().item()),
            )
            nn.init.normal_(
                head.weight,
                mean=0.0,
                std=float(old_head.weight.detach().std().item()),
            )
            nn.init.zeros_(head.bias)
        self.decoder.embed_tokens = embedding
        self.output_head = head
        self.num_classes = NUM_CLASSES
        self.bos_id = NUM_CLASSES
        self.pad_id = NUM_CLASSES + 1
        self.decoder_vocab_size = NUM_CLASSES + 2
        self.decoder.config.vocab_size = self.decoder_vocab_size
        self.decoder.config.bos_token_id = self.bos_id
        self.decoder.config.pad_token_id = self.pad_id
        self.decoder.config.eos_token_id = self.pad_id
        self.class_weights = torch.ones(NUM_CLASSES)

    def prediction_head_parameters(self) -> Iterable[nn.Parameter]:
        for module in (self.decoder, self.slot_embeddings, self.output_head):
            yield from module.parameters()

    def _shift_right(self, flat_targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return shift_right_four_class_targets(flat_targets)

    def _loss_components(
        self, flat_logits: torch.Tensor, flat_targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        numerator = F.cross_entropy(
            flat_logits.reshape(-1, NUM_CLASSES),
            flat_targets.reshape(-1),
            ignore_index=IGNORE_INDEX,
            reduction="sum",
        )
        denominator = (flat_targets != IGNORE_INDEX).sum().to(flat_logits.dtype)
        if not bool(torch.isfinite(numerator)) or float(denominator.item()) <= 0.0:
            raise FloatingPointError("unweighted four-class CE components are invalid")
        return numerator / denominator, numerator, denominator

    def teacher_forced(
        self,
        encoder_hidden: torch.Tensor,
        note_mask: torch.Tensor,
        pedal_targets: torch.Tensor,
    ) -> EncoderDecoderFiveClassOutput:
        flat_targets = flatten_pedal_targets(pedal_targets)
        decoder_inputs, decoder_mask = self._shift_right(flat_targets)
        decoded = self.decoder(
            inputs_embeds=self._decoder_embeddings(decoder_inputs),
            attention_mask=decoder_mask,
            encoder_hidden_states=encoder_hidden,
            encoder_attention_mask=note_mask,
            use_cache=False,
            return_dict=True,
        )
        flat_logits = self.output_head(decoded.last_hidden_state)
        loss, numerator, denominator = self._loss_components(flat_logits, flat_targets)
        batch, notes, _ = pedal_targets.shape
        return EncoderDecoderFiveClassOutput(
            logits=flat_logits.view(batch, notes, PEDAL_SLOTS, NUM_CLASSES),
            loss=loss,
            loss_numerator=numerator,
            loss_denominator=denominator,
            encoder_hidden_states=encoder_hidden,
            decoder_hidden_states=decoded.last_hidden_state,
            decoder_input_ids=decoder_inputs,
        )

    @torch.no_grad()
    def greedy_from_encoder(
        self, encoder_hidden: torch.Tensor, note_mask: torch.Tensor
    ) -> EncoderDecoderFiveClassOutput:
        batch, notes, _ = encoder_hidden.shape
        lengths = note_mask.long().sum(dim=1) * PEDAL_SLOTS
        maximum = int(lengths.max().item())
        if maximum <= 0:
            raise ValueError("greedy decoding requires at least one valid note")
        previous = torch.full(
            (batch, 1), self.bos_id, dtype=torch.long, device=encoder_hidden.device
        )
        logits_steps: list[torch.Tensor] = []
        past = None
        for step in range(maximum):
            decoded = self.decoder(
                inputs_embeds=self._decoder_embeddings(previous, start_position=step),
                attention_mask=(
                    torch.arange(step + 1, device=encoder_hidden.device).unsqueeze(0)
                    < lengths.unsqueeze(1)
                ),
                position_ids=torch.tensor(
                    [[step]], dtype=torch.long, device=encoder_hidden.device
                ),
                past_key_values=past,
                encoder_hidden_states=encoder_hidden,
                encoder_attention_mask=note_mask,
                use_cache=True,
                cache_position=torch.tensor(
                    [step], dtype=torch.long, device=encoder_hidden.device
                ),
                return_dict=True,
            )
            step_logits = self.output_head(decoded.last_hidden_state[:, -1])
            if not bool(torch.isfinite(step_logits).all()):
                raise FloatingPointError("greedy decoder produced non-finite logits")
            logits_steps.append(step_logits)
            prediction = step_logits.argmax(dim=-1)
            previous = torch.where(
                (step + 1) < lengths,
                prediction,
                torch.full_like(prediction, self.pad_id),
            ).unsqueeze(1)
            past = decoded.past_key_values
        flat_logits = torch.stack(logits_steps, dim=1)
        full_length = notes * PEDAL_SLOTS
        if maximum < full_length:
            flat_logits = torch.cat(
                (
                    flat_logits,
                    flat_logits.new_zeros(
                        (batch, full_length - maximum, NUM_CLASSES)
                    ),
                ),
                dim=1,
            )
        return EncoderDecoderFiveClassOutput(
            logits=flat_logits.view(batch, notes, PEDAL_SLOTS, NUM_CLASSES),
            encoder_hidden_states=encoder_hidden,
        )
