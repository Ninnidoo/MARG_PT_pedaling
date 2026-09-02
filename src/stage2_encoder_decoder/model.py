"""PT-native encoder-decoder model for flattened five-class pedal tokens."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from src.stage2_encoder_only.dataset import PEDAL_SLOTS, TOKENS_PER_NOTE
from src.stage2_encoder_only.five_class import (
    IGNORE_INDEX,
    NUM_CLASSES,
    validate_class_weights,
)


BOS_ID = NUM_CLASSES
PAD_ID = NUM_CLASSES + 1
DECODER_VOCAB_SIZE = NUM_CLASSES + 2
DECODER_INIT_SEED = 42


def flatten_pedal_targets(targets: torch.Tensor) -> torch.Tensor:
    """Flatten [B,N,P1..P4] in note-major Pedal1->Pedal4 order."""

    if targets.ndim != 3 or targets.shape[-1] != PEDAL_SLOTS:
        raise ValueError("pedal targets must have shape [B,N,4]")
    return targets.contiguous().view(targets.shape[0], -1)


def shift_right_pedal_targets(flat_targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Create BOS,y_1,...,y_(L-1) and its valid-position attention mask."""

    if flat_targets.ndim != 2 or flat_targets.shape[1] < 1:
        raise ValueError("flat targets must have non-empty shape [B,4N]")
    valid = flat_targets != IGNORE_INDEX
    legal = valid & ((flat_targets < 0) | (flat_targets >= NUM_CLASSES))
    if bool(legal.any()):
        raise ValueError("targets must be in [0,4] or -100")
    decoder_inputs = torch.full_like(flat_targets, PAD_ID)
    decoder_inputs[:, 0] = BOS_ID
    previous = flat_targets[:, :-1]
    decoder_inputs[:, 1:] = torch.where(
        previous == IGNORE_INDEX,
        torch.full_like(previous, PAD_ID),
        previous,
    )
    decoder_inputs = torch.where(valid, decoder_inputs, torch.full_like(decoder_inputs, PAD_ID))
    return decoder_inputs, valid


@dataclass
class EncoderDecoderFiveClassOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    loss_numerator: torch.Tensor | None = None
    loss_denominator: torch.Tensor | None = None
    encoder_hidden_states: torch.Tensor | None = None
    decoder_hidden_states: torch.Tensor | None = None
    decoder_input_ids: torch.Tensor | None = None
    teacher_forcing_mask: torch.Tensor | None = None
    history_predictions: torch.Tensor | None = None


@dataclass
class ScheduledSamplingHistory:
    """Detached autoregressive decoder history used by the training loss pass."""

    decoder_input_ids: torch.Tensor
    decoder_attention_mask: torch.Tensor
    teacher_forcing_mask: torch.Tensor
    predictions: torch.Tensor

    @property
    def valid_history_mask(self) -> torch.Tensor:
        result = self.decoder_attention_mask.clone()
        result[:, 0] = False
        return result

    @property
    def ground_truth_count(self) -> int:
        valid = self.valid_history_mask
        return int((self.teacher_forcing_mask & valid).sum().item())

    @property
    def predicted_count(self) -> int:
        valid = self.valid_history_mask
        return int((~self.teacher_forcing_mask & valid).sum().item())


class _DecoderTrainableGroups:
    """Non-registering optimizer facade over decoder-side modules."""

    def __init__(self, modules: Sequence[nn.Module]) -> None:
        self._modules = tuple(modules)

    def __iter__(self) -> Iterator[nn.Module]:
        return iter(self._modules)

    def __len__(self) -> int:
        return len(self._modules)

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        for module in self._modules:
            yield from module.parameters(recurse=recurse)


class EncoderDecoderFiveClassModel(nn.Module):
    """Pretrained PT encoder plus a freshly initialized native two-layer decoder."""

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
        if self.hidden_size != 768 and int(getattr(decoder_config, "hidden_size")) != self.hidden_size:
            raise ValueError("encoder and decoder hidden sizes differ")
        config = copy.deepcopy(decoder_config)
        config.vocab_size = DECODER_VOCAB_SIZE
        config.pad_token_id = PAD_ID
        config.bos_token_id = BOS_ID
        config.eos_token_id = PAD_ID
        config.is_decoder = True
        config.cross_attention_hidden_size = self.hidden_size
        config.num_hidden_layers = 2
        self.decoder_init_seed = int(decoder_init_seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.decoder_init_seed)
            self.decoder = T5GemmaDecoder(config)
            self.slot_embeddings = nn.Embedding(PEDAL_SLOTS, self.hidden_size)
            self.output_head = nn.Linear(self.hidden_size, NUM_CLASSES)
            nn.init.normal_(self.slot_embeddings.weight, mean=0.0, std=float(config.initializer_range))
            nn.init.normal_(self.output_head.weight, mean=0.0, std=float(config.initializer_range))
            nn.init.zeros_(self.output_head.bias)
        self.register_buffer("class_weights", torch.ones(NUM_CLASSES), persistent=False)
        self.set_encoder_frozen(freeze_encoder)

    @property
    def classification_heads(self) -> _DecoderTrainableGroups:
        # Existing optimizer/training helpers call this historical attribute.
        return _DecoderTrainableGroups(
            (self.decoder, self.slot_embeddings, self.output_head)
        )

    @classmethod
    def from_pretrained_encoder(
        cls,
        checkpoint_path: str | Path,
        *,
        decoder_init_seed: int = DECODER_INIT_SEED,
        freeze_encoder: bool = False,
        **pretrained_kwargs: Any,
    ) -> "EncoderDecoderFiveClassModel":
        from third_party.PianistTransformer.src.model.pianoformer import PianoT5Gemma

        full_model = PianoT5Gemma.from_pretrained(str(checkpoint_path), **pretrained_kwargs)
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

    def set_class_weights(self, values: torch.Tensor | Sequence[float] | None) -> None:
        weights = validate_class_weights(values)
        self.class_weights.copy_(
            torch.ones_like(self.class_weights) if weights is None else weights
        )

    def _encode(
        self, input_ids: torch.Tensor, token_attention_mask: torch.Tensor
    ) -> torch.Tensor:
        encoded = self.encoder(input_ids=input_ids, attention_mask=token_attention_mask)
        hidden = getattr(encoded, "last_hidden_state", None)
        if hidden is None or hidden.ndim != 3 or hidden.shape[-1] != self.hidden_size:
            raise RuntimeError("encoder returned invalid note states")
        return hidden

    def _decoder_embeddings(self, token_ids: torch.Tensor, start_position: int = 0) -> torch.Tensor:
        positions = torch.arange(
            start_position,
            start_position + token_ids.shape[1],
            device=token_ids.device,
        )
        slots = positions.remainder(PEDAL_SLOTS).unsqueeze(0).expand(token_ids.shape[0], -1)
        return self.decoder.embed_tokens(token_ids) + self.slot_embeddings(slots)

    def _loss_components(
        self, flat_logits: torch.Tensor, flat_targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weights = self.class_weights.to(device=flat_logits.device, dtype=flat_logits.dtype)
        numerator = F.cross_entropy(
            flat_logits.reshape(-1, NUM_CLASSES),
            flat_targets.reshape(-1),
            ignore_index=IGNORE_INDEX,
            weight=weights,
            reduction="sum",
        )
        active = flat_targets != IGNORE_INDEX
        denominator = weights[flat_targets[active]].sum()
        if not bool(torch.isfinite(numerator)) or not bool(torch.isfinite(denominator)):
            raise FloatingPointError("weighted CE components are non-finite")
        if float(denominator.detach().item()) <= 0.0:
            raise ValueError("weighted CE has no valid targets")
        return numerator / denominator, numerator, denominator

    def teacher_forced(
        self,
        encoder_hidden: torch.Tensor,
        note_mask: torch.Tensor,
        pedal_targets: torch.Tensor,
    ) -> EncoderDecoderFiveClassOutput:
        flat_targets = flatten_pedal_targets(pedal_targets)
        decoder_inputs, decoder_mask = shift_right_pedal_targets(flat_targets)
        embedded = self._decoder_embeddings(decoder_inputs)
        decoded = self.decoder(
            inputs_embeds=embedded,
            attention_mask=decoder_mask,
            encoder_hidden_states=encoder_hidden,
            encoder_attention_mask=note_mask,
            use_cache=False,
            return_dict=True,
        )
        flat_logits = self.output_head(decoded.last_hidden_state)
        loss, numerator, denominator = self._loss_components(flat_logits, flat_targets)
        batch, notes, _ = pedal_targets.shape
        logits = flat_logits.view(batch, notes, PEDAL_SLOTS, NUM_CLASSES)
        return EncoderDecoderFiveClassOutput(
            logits=logits,
            loss=loss,
            loss_numerator=numerator,
            loss_denominator=denominator,
            encoder_hidden_states=encoder_hidden,
            decoder_hidden_states=decoded.last_hidden_state,
            decoder_input_ids=decoder_inputs,
        )

    @torch.no_grad()
    def build_scheduled_sampling_history(
        self,
        encoder_hidden: torch.Tensor,
        note_mask: torch.Tensor,
        pedal_targets: torch.Tensor,
        teacher_forcing_probability: float,
        *,
        generator: torch.Generator | None = None,
    ) -> ScheduledSamplingHistory:
        """Construct BOS/mixed history strictly left-to-right with no future GT.

        The rollout is detached because argmax history tokens are discrete.  A
        separate causal forward computes the exact weighted-CE gradient
        conditional on this completed history.
        """

        probability = float(teacher_forcing_probability)
        if not 0.0 <= probability <= 1.0:
            raise ValueError("teacher_forcing_probability must be in [0,1]")
        flat_targets = flatten_pedal_targets(pedal_targets)
        decoder_mask = flat_targets != IGNORE_INDEX
        batch, full_length = flat_targets.shape
        lengths = decoder_mask.long().sum(dim=1)
        if bool((lengths <= 0).any()):
            raise ValueError("scheduled sampling requires at least one valid target")
        decoder_inputs = torch.full_like(flat_targets, PAD_ID)
        decoder_inputs[:, 0] = BOS_ID
        teacher_mask = torch.zeros_like(decoder_mask)
        predictions = torch.full_like(flat_targets, PAD_ID)
        if probability == 1.0:
            shifted, shifted_mask = shift_right_pedal_targets(flat_targets)
            teacher_mask[:, 1:] = shifted_mask[:, 1:]
            return ScheduledSamplingHistory(
                decoder_input_ids=shifted.detach(),
                decoder_attention_mask=shifted_mask.detach(),
                teacher_forcing_mask=teacher_mask.detach(),
                predictions=predictions.detach(),
            )

        previous = decoder_inputs[:, :1]
        past = None
        maximum = int(lengths.max().item())
        for step in range(maximum):
            embedded = self._decoder_embeddings(previous, start_position=step)
            prefix_mask = (
                torch.arange(step + 1, device=encoder_hidden.device).unsqueeze(0)
                < lengths.unsqueeze(1)
            )
            position = torch.tensor(
                [[step]], dtype=torch.long, device=encoder_hidden.device
            )
            cache_position = torch.tensor(
                [step], dtype=torch.long, device=encoder_hidden.device
            )
            decoded = self.decoder(
                inputs_embeds=embedded,
                attention_mask=prefix_mask,
                position_ids=position,
                past_key_values=past,
                encoder_hidden_states=encoder_hidden,
                encoder_attention_mask=note_mask,
                use_cache=True,
                cache_position=cache_position,
                return_dict=True,
            )
            step_logits = self.output_head(decoded.last_hidden_state[:, -1])
            if not bool(torch.isfinite(step_logits).all()):
                raise FloatingPointError(
                    "scheduled-sampling rollout produced non-finite logits"
                )
            prediction = step_logits.argmax(dim=-1).detach()
            predictions[:, step] = prediction
            active_next = (step + 1) < lengths
            if step + 1 < full_length:
                if probability == 0.0:
                    use_teacher = torch.zeros(
                        batch, dtype=torch.bool, device=encoder_hidden.device
                    )
                else:
                    random_values = torch.rand(
                        batch,
                        device=encoder_hidden.device,
                        generator=generator,
                    )
                    use_teacher = random_values < probability
                use_teacher &= active_next
                selected = torch.where(
                    use_teacher,
                    flat_targets[:, step],
                    prediction,
                ).detach()
                decoder_inputs[:, step + 1] = torch.where(
                    active_next,
                    selected,
                    torch.full_like(selected, PAD_ID),
                )
                teacher_mask[:, step + 1] = use_teacher
                previous = decoder_inputs[:, step + 1 : step + 2]
            past = decoded.past_key_values
        if not torch.equal(
            decoder_inputs.masked_fill(decoder_mask, PAD_ID),
            torch.full_like(decoder_inputs, PAD_ID),
        ):
            raise RuntimeError("PAD positions were populated in scheduled history")
        return ScheduledSamplingHistory(
            decoder_input_ids=decoder_inputs.detach(),
            decoder_attention_mask=decoder_mask.detach(),
            teacher_forcing_mask=teacher_mask.detach(),
            predictions=predictions.detach(),
        )

    @torch.no_grad()
    def scheduled_sampling_history(
        self,
        input_ids: torch.Tensor,
        token_attention_mask: torch.Tensor,
        pedal_targets: torch.Tensor,
        note_mask: torch.Tensor,
        teacher_forcing_probability: float,
        *,
        generator: torch.Generator | None = None,
    ) -> ScheduledSamplingHistory:
        encoder_hidden = self._encode(input_ids, token_attention_mask)
        return self.build_scheduled_sampling_history(
            encoder_hidden,
            note_mask.bool(),
            pedal_targets,
            teacher_forcing_probability,
            generator=generator,
        )

    def loss_from_scheduled_history(
        self,
        input_ids: torch.Tensor,
        token_attention_mask: torch.Tensor,
        pedal_targets: torch.Tensor,
        note_mask: torch.Tensor,
        history: ScheduledSamplingHistory | torch.Tensor,
    ) -> EncoderDecoderFiveClassOutput:
        """Compute causal weighted CE for an already detached mixed history."""

        encoder_hidden = self._encode(input_ids, token_attention_mask)
        flat_targets = flatten_pedal_targets(pedal_targets)
        decoder_mask = flat_targets != IGNORE_INDEX
        decoder_inputs = (
            history.decoder_input_ids
            if isinstance(history, ScheduledSamplingHistory)
            else history
        )
        if decoder_inputs.shape != flat_targets.shape:
            raise ValueError("scheduled decoder history shape mismatch")
        if decoder_inputs.requires_grad:
            raise ValueError("scheduled decoder history must be detached")
        if not torch.equal(
            decoder_inputs.masked_select(~decoder_mask),
            torch.full_like(decoder_inputs.masked_select(~decoder_mask), PAD_ID),
        ):
            raise ValueError("scheduled history has non-PAD values in padded positions")
        embedded = self._decoder_embeddings(decoder_inputs)
        decoded = self.decoder(
            inputs_embeds=embedded,
            attention_mask=decoder_mask,
            encoder_hidden_states=encoder_hidden,
            encoder_attention_mask=note_mask.bool(),
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
            teacher_forcing_mask=(
                history.teacher_forcing_mask
                if isinstance(history, ScheduledSamplingHistory)
                else None
            ),
            history_predictions=(
                history.predictions
                if isinstance(history, ScheduledSamplingHistory)
                else None
            ),
        )

    @torch.no_grad()
    def greedy_from_encoder(
        self, encoder_hidden: torch.Tensor, note_mask: torch.Tensor
    ) -> EncoderDecoderFiveClassOutput:
        batch, notes, _ = encoder_hidden.shape
        lengths = note_mask.long().sum(dim=1) * PEDAL_SLOTS
        maximum = int(lengths.max().item())
        previous = torch.full((batch, 1), BOS_ID, dtype=torch.long, device=encoder_hidden.device)
        logits_steps: list[torch.Tensor] = []
        past = None
        for step in range(maximum):
            embedded = self._decoder_embeddings(previous, start_position=step)
            prefix_mask = torch.arange(step + 1, device=encoder_hidden.device).unsqueeze(0) < lengths.unsqueeze(1)
            position = torch.tensor([[step]], dtype=torch.long, device=encoder_hidden.device)
            cache_position = torch.tensor([step], dtype=torch.long, device=encoder_hidden.device)
            decoded = self.decoder(
                inputs_embeds=embedded,
                attention_mask=prefix_mask,
                position_ids=position,
                past_key_values=past,
                encoder_hidden_states=encoder_hidden,
                encoder_attention_mask=note_mask,
                use_cache=True,
                cache_position=cache_position,
                return_dict=True,
            )
            step_logits = self.output_head(decoded.last_hidden_state[:, -1])
            if not bool(torch.isfinite(step_logits).all()):
                raise FloatingPointError("greedy decoder produced non-finite logits")
            logits_steps.append(step_logits)
            prediction = step_logits.argmax(dim=-1)
            active_next = (step + 1) < lengths
            previous = torch.where(
                active_next,
                prediction,
                torch.full_like(prediction, PAD_ID),
            ).unsqueeze(1)
            past = decoded.past_key_values
        flat_logits = torch.stack(logits_steps, dim=1)
        full_length = notes * PEDAL_SLOTS
        if maximum < full_length:
            padding = flat_logits.new_zeros((batch, full_length - maximum, NUM_CLASSES))
            flat_logits = torch.cat((flat_logits, padding), dim=1)
        logits = flat_logits.view(batch, notes, PEDAL_SLOTS, NUM_CLASSES)
        return EncoderDecoderFiveClassOutput(logits=logits, encoder_hidden_states=encoder_hidden)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_attention_mask: torch.Tensor,
        pedal_targets: torch.Tensor | None = None,
        note_mask: torch.Tensor | None = None,
        *,
        decode_mode: str = "teacher_forced",
    ) -> EncoderDecoderFiveClassOutput:
        if input_ids.ndim != 2 or input_ids.shape[1] % TOKENS_PER_NOTE:
            raise ValueError("input_ids must have shape [B,N*8]")
        batch, flat_tokens = input_ids.shape
        notes = flat_tokens // TOKENS_PER_NOTE
        if token_attention_mask.shape != input_ids.shape:
            raise ValueError("token attention mask shape mismatch")
        if note_mask is None:
            note_mask = token_attention_mask.view(batch, notes, TOKENS_PER_NOTE).any(dim=-1)
        if note_mask.shape != (batch, notes):
            raise ValueError("note_mask must have shape [B,N]")
        hidden = self._encode(input_ids, token_attention_mask)
        if hidden.shape[:2] != (batch, notes):
            raise RuntimeError("encoder note-state count mismatch")
        if decode_mode == "greedy":
            return self.greedy_from_encoder(hidden, note_mask.bool())
        if decode_mode != "teacher_forced" or pedal_targets is None:
            raise ValueError("teacher_forced mode requires pedal_targets")
        if pedal_targets.shape != (batch, notes, PEDAL_SLOTS):
            raise ValueError("pedal_targets must have shape [B,N,4]")
        expected = note_mask.bool().unsqueeze(-1).expand_as(pedal_targets)
        if not torch.equal(expected, pedal_targets != IGNORE_INDEX):
            raise ValueError("note mask and padded targets are inconsistent")
        return self.teacher_forced(hidden, note_mask.bool(), pedal_targets)
