"""PT-native causal encoder-decoder for binary Pedal1--4 prediction."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch import nn

from src.stage2_binary.dataset import IGNORE_INDEX
from src.stage2_encoder_only.dataset import PEDAL_SLOTS, TOKENS_PER_NOTE


BINARY_CLASSES = 2
BOS_ID = BINARY_CLASSES
PAD_ID = BINARY_CLASSES + 1
DECODER_VOCAB_SIZE = BINARY_CLASSES + 2
DECODER_INIT_SEED = 42


def flatten_binary_pedal_targets(targets: torch.Tensor) -> torch.Tensor:
    """Flatten [B,N,P1..P4] as P1_1,P2_1,P3_1,P4_1,P1_2,... ."""

    if targets.ndim != 3 or targets.shape[-1] != PEDAL_SLOTS:
        raise ValueError("binary pedal targets must have shape [B,N,4]")
    legal = (targets == 0) | (targets == 1) | (targets == IGNORE_INDEX)
    if not bool(torch.all(legal)):
        raise ValueError("binary targets must be 0, 1, or ignore_index")
    return targets.contiguous().view(targets.shape[0], -1)


def unflatten_binary_sequence(
    flat_tokens: torch.Tensor, *, note_count: int | None = None
) -> torch.Tensor:
    """Restore a note-major flat binary sequence to [B,N,P1..P4]."""

    if flat_tokens.ndim != 2 or flat_tokens.shape[1] == 0:
        raise ValueError("flat binary sequence must have non-empty shape [B,4N]")
    if flat_tokens.shape[1] % PEDAL_SLOTS:
        raise ValueError("flat binary sequence length must be divisible by four")
    resolved_notes = flat_tokens.shape[1] // PEDAL_SLOTS
    if note_count is not None and int(note_count) != resolved_notes:
        raise ValueError("requested note count disagrees with flat sequence length")
    if not bool(torch.all((flat_tokens == 0) | (flat_tokens == 1))):
        raise ValueError("reconstructed decoder output must contain binary classes only")
    return flat_tokens.contiguous().view(flat_tokens.shape[0], resolved_notes, PEDAL_SLOTS)


def shift_right_binary_targets(
    flat_targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build fixed-length teacher inputs BOS,y_1,...,y_(L-1)."""

    if flat_targets.ndim != 2 or flat_targets.shape[1] < 1:
        raise ValueError("flat targets must have non-empty shape [B,4N]")
    legal = (
        (flat_targets == 0)
        | (flat_targets == 1)
        | (flat_targets == IGNORE_INDEX)
    )
    if not bool(torch.all(legal)):
        raise ValueError("flat binary targets must be 0, 1, or ignore_index")
    valid = flat_targets != IGNORE_INDEX
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


@dataclass
class BinaryEncoderDecoderOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    loss_numerator: torch.Tensor | None = None
    loss_denominator: torch.Tensor | None = None
    encoder_hidden_states: torch.Tensor | None = None
    decoder_hidden_states: torch.Tensor | None = None
    decoder_input_ids: torch.Tensor | None = None
    generated_ids: torch.Tensor | None = None


class BinaryPedalEncoderDecoderModel(nn.Module):
    """Official PT encoder plus the existing native two-layer causal decoder."""

    target_name = "binary_targets"

    def __init__(
        self,
        encoder: nn.Module,
        decoder_config: Any,
        *,
        decoder_init_seed: int = DECODER_INIT_SEED,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        from transformers.models.t5gemma.modeling_t5gemma import T5GemmaDecoder

        self.encoder = encoder
        self.hidden_size = int(getattr(encoder.config, "hidden_size"))
        config = copy.deepcopy(decoder_config)
        if int(getattr(config, "hidden_size")) != self.hidden_size:
            raise ValueError("encoder and decoder hidden sizes differ")
        config.vocab_size = DECODER_VOCAB_SIZE
        config.pad_token_id = PAD_ID
        config.bos_token_id = BOS_ID
        # Generation is fixed-length; PAD occupies the unused EOS config slot.
        config.eos_token_id = PAD_ID
        config.is_decoder = True
        config.cross_attention_hidden_size = self.hidden_size
        config.num_hidden_layers = 2
        self.decoder_init_seed = int(decoder_init_seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.decoder_init_seed)
            self.decoder = T5GemmaDecoder(config)
            self.slot_embeddings = nn.Embedding(PEDAL_SLOTS, self.hidden_size)
            self.output_head = nn.Linear(self.hidden_size, BINARY_CLASSES)
            nn.init.normal_(
                self.slot_embeddings.weight,
                mean=0.0,
                std=float(config.initializer_range),
            )
            nn.init.normal_(
                self.output_head.weight,
                mean=0.0,
                std=float(config.initializer_range),
            )
            nn.init.zeros_(self.output_head.bias)
        self.set_encoder_frozen(freeze_encoder)

    @classmethod
    def from_pretrained_encoder(
        cls,
        checkpoint_path: str | Path,
        *,
        decoder_init_seed: int = DECODER_INIT_SEED,
        freeze_encoder: bool = False,
        **pretrained_kwargs: Any,
    ) -> "BinaryPedalEncoderDecoderModel":
        from third_party.PianistTransformer.src.model.pianoformer import PianoT5Gemma

        full_model = PianoT5Gemma.from_pretrained(
            str(checkpoint_path), **pretrained_kwargs
        )
        encoder = full_model.get_encoder()
        decoder_config = copy.deepcopy(full_model.config.decoder)
        full_model.model.encoder = None
        del full_model
        return cls(
            encoder,
            decoder_config,
            decoder_init_seed=decoder_init_seed,
            freeze_encoder=freeze_encoder,
        )

    def set_encoder_frozen(self, freeze: bool) -> None:
        self.freeze_encoder = bool(freeze)
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(not freeze)

    def prediction_head_parameters(self) -> Iterable[nn.Parameter]:
        for module in (self.decoder, self.slot_embeddings, self.output_head):
            yield from module.parameters()

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def _encode(
        self, input_ids: torch.Tensor, token_attention_mask: torch.Tensor
    ) -> torch.Tensor:
        encoded = self.encoder(
            input_ids=input_ids, attention_mask=token_attention_mask
        )
        hidden = getattr(encoded, "last_hidden_state", None)
        if hidden is None or hidden.ndim != 3 or hidden.shape[-1] != self.hidden_size:
            raise RuntimeError("encoder returned invalid note states")
        return hidden

    def _decoder_embeddings(
        self, token_ids: torch.Tensor, *, start_position: int = 0
    ) -> torch.Tensor:
        positions = torch.arange(
            start_position,
            start_position + token_ids.shape[1],
            device=token_ids.device,
        )
        slots = positions.remainder(PEDAL_SLOTS).unsqueeze(0).expand(
            token_ids.shape[0], -1
        )
        return self.decoder.embed_tokens(token_ids) + self.slot_embeddings(slots)

    @staticmethod
    def _loss_components(
        flat_logits: torch.Tensor, flat_targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        numerator = F.cross_entropy(
            flat_logits.reshape(-1, BINARY_CLASSES),
            flat_targets.reshape(-1),
            ignore_index=IGNORE_INDEX,
            reduction="sum",
        )
        denominator = (flat_targets != IGNORE_INDEX).sum().to(flat_logits.dtype)
        if not bool(torch.isfinite(numerator)) or float(denominator.item()) <= 0.0:
            raise FloatingPointError("unweighted binary CE components are invalid")
        return numerator / denominator, numerator, denominator

    def teacher_forced(
        self,
        encoder_hidden: torch.Tensor,
        note_mask: torch.Tensor,
        binary_targets: torch.Tensor,
    ) -> BinaryEncoderDecoderOutput:
        flat_targets = flatten_binary_pedal_targets(binary_targets)
        decoder_inputs, decoder_mask = shift_right_binary_targets(flat_targets)
        decoded = self.decoder(
            inputs_embeds=self._decoder_embeddings(decoder_inputs),
            attention_mask=decoder_mask,
            encoder_hidden_states=encoder_hidden,
            encoder_attention_mask=note_mask,
            use_cache=False,
            return_dict=True,
        )
        flat_logits = self.output_head(decoded.last_hidden_state)
        loss, numerator, denominator = self._loss_components(
            flat_logits, flat_targets
        )
        batch, notes, _ = binary_targets.shape
        return BinaryEncoderDecoderOutput(
            logits=flat_logits.view(
                batch, notes, PEDAL_SLOTS, BINARY_CLASSES
            ),
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
    ) -> BinaryEncoderDecoderOutput:
        """Fixed-length greedy rollout using only BOS and prior predictions."""

        batch, notes, _ = encoder_hidden.shape
        lengths = note_mask.long().sum(dim=1) * PEDAL_SLOTS
        maximum = int(lengths.max().item())
        if maximum <= 0:
            raise ValueError("greedy inference requires at least one valid note")
        previous = torch.full(
            (batch, 1), BOS_ID, dtype=torch.long, device=encoder_hidden.device
        )
        logits_steps: list[torch.Tensor] = []
        input_steps: list[torch.Tensor] = []
        generated_steps: list[torch.Tensor] = []
        past = None
        for step in range(maximum):
            input_steps.append(previous[:, 0].clone())
            decoded = self.decoder(
                inputs_embeds=self._decoder_embeddings(
                    previous, start_position=step
                ),
                attention_mask=(
                    torch.arange(step + 1, device=encoder_hidden.device)
                    .unsqueeze(0)
                    .lt(lengths.unsqueeze(1))
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
            active_current = step < lengths
            generated_steps.append(
                torch.where(
                    active_current,
                    prediction,
                    torch.full_like(prediction, PAD_ID),
                )
            )
            active_next = (step + 1) < lengths
            previous = torch.where(
                active_next,
                prediction,
                torch.full_like(prediction, PAD_ID),
            ).unsqueeze(1)
            past = decoded.past_key_values
        flat_logits = torch.stack(logits_steps, dim=1)
        decoder_inputs = torch.stack(input_steps, dim=1)
        generated = torch.stack(generated_steps, dim=1)
        full_length = notes * PEDAL_SLOTS
        if maximum < full_length:
            flat_logits = torch.cat(
                (
                    flat_logits,
                    flat_logits.new_zeros(
                        (batch, full_length - maximum, BINARY_CLASSES)
                    ),
                ),
                dim=1,
            )
            decoder_inputs = F.pad(
                decoder_inputs, (0, full_length - maximum), value=PAD_ID
            )
            generated = F.pad(
                generated, (0, full_length - maximum), value=PAD_ID
            )
        return BinaryEncoderDecoderOutput(
            logits=flat_logits.view(
                batch, notes, PEDAL_SLOTS, BINARY_CLASSES
            ),
            encoder_hidden_states=encoder_hidden,
            decoder_input_ids=decoder_inputs,
            generated_ids=generated.view(batch, notes, PEDAL_SLOTS),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        token_attention_mask: torch.Tensor,
        binary_targets: torch.Tensor | None = None,
        note_mask: torch.Tensor | None = None,
        *,
        decode_mode: str = "teacher_forced",
    ) -> BinaryEncoderDecoderOutput:
        if input_ids.ndim != 2 or input_ids.shape[1] % TOKENS_PER_NOTE:
            raise ValueError("input_ids must have shape [B,N*8]")
        batch, flat_tokens = input_ids.shape
        notes = flat_tokens // TOKENS_PER_NOTE
        if token_attention_mask.shape != input_ids.shape:
            raise ValueError("token attention mask shape mismatch")
        if note_mask is None:
            note_mask = token_attention_mask.view(
                batch, notes, TOKENS_PER_NOTE
            ).any(dim=-1)
        if note_mask.shape != (batch, notes):
            raise ValueError("note_mask must have shape [B,N]")
        hidden = self._encode(input_ids, token_attention_mask)
        if hidden.shape[:2] != (batch, notes):
            raise RuntimeError("encoder note-state count mismatch")
        if decode_mode == "greedy":
            if binary_targets is not None:
                raise ValueError("free-running greedy inference forbids targets")
            return self.greedy_from_encoder(hidden, note_mask.bool())
        if decode_mode != "teacher_forced" or binary_targets is None:
            raise ValueError("teacher_forced mode requires binary_targets")
        if binary_targets.shape != (batch, notes, PEDAL_SLOTS):
            raise ValueError("binary_targets must have shape [B,N,4]")
        expected = note_mask.bool().unsqueeze(-1).expand_as(binary_targets)
        if not torch.equal(expected, binary_targets != IGNORE_INDEX):
            raise ValueError("note mask and padded targets are inconsistent")
        return self.teacher_forced(hidden, note_mask.bool(), binary_targets)
