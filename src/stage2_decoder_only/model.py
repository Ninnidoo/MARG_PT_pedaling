"""Canonical four-class Decoder-only Stage 2 Prefix-LM.

Only the pretrained PT token/feature/note-compression module is retained.
The two-layer T5Gemma self-attention stack is fresh and has no encoder or
cross-attention. Performance notes form a bidirectional prefix; flattened
Pedal1--4 inputs form the causal suffix.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import DynamicCache
from transformers.models.t5gemma.modeling_t5gemma import (
    T5GemmaEncoderLayer,
    T5GemmaPreTrainedModel,
    T5GemmaRMSNorm,
    T5GemmaRotaryEmbedding,
)

from src.stage2_encoder_decoder.model import flatten_pedal_targets
from src.stage2_encoder_only.dataset import MASK_ID, PEDAL_SLOTS, TOKENS_PER_NOTE
from src.stage2_four_class.model import (
    BOS_ID,
    DECODER_VOCAB_SIZE,
    PAD_ID,
    shift_right_four_class_targets,
)
from src.stage2_four_class.representation import IGNORE_INDEX, NUM_CLASSES


DEFAULT_NUM_LAYERS = 2
DEFAULT_INIT_SEED = 42


def build_prefix_lm_attention_mask(
    note_mask: torch.Tensor,
    autoregressive_mask: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return additive ``[B,1,N+T,N+T]`` Prefix-LM attention mask.

    Zero entries are allowed and ``finfo(dtype).min`` entries are blocked.
    Padding follows the established Stage 2 key-padding convention: padded
    prefix and autoregressive positions can never be used as keys.
    """

    if note_mask.ndim != 2:
        raise ValueError("note_mask must have shape [B,N]")
    if autoregressive_mask.ndim != 2:
        raise ValueError("autoregressive_mask must have shape [B,T]")
    if note_mask.shape[0] != autoregressive_mask.shape[0]:
        raise ValueError("prefix and autoregressive masks must share batch size")
    if not dtype.is_floating_point:
        raise TypeError("attention mask dtype must be floating point")

    prefix_valid = note_mask.bool()
    ar_valid = autoregressive_mask.bool()
    batch, prefix_length = prefix_valid.shape
    ar_length = ar_valid.shape[1]
    total_length = prefix_length + ar_length
    device = note_mask.device

    allowed = torch.zeros(
        (batch, total_length, total_length), dtype=torch.bool, device=device
    )
    allowed[:, :prefix_length, :prefix_length] = prefix_valid[:, None, :]
    allowed[:, prefix_length:, :prefix_length] = prefix_valid[:, None, :]
    causal = torch.ones(
        (ar_length, ar_length), dtype=torch.bool, device=device
    ).tril()
    allowed[:, prefix_length:, prefix_length:] = (
        causal.unsqueeze(0) & ar_valid[:, None, :]
    )

    additive = torch.zeros(
        (batch, 1, total_length, total_length), dtype=dtype, device=device
    )
    return additive.masked_fill(~allowed.unsqueeze(1), torch.finfo(dtype).min)


def _prefix_only_attention_mask(
    note_mask: torch.Tensor, *, dtype: torch.dtype
) -> torch.Tensor:
    valid = note_mask.bool()
    batch, length = valid.shape
    allowed = valid[:, None, None, :].expand(batch, 1, length, length)
    result = torch.zeros(
        (batch, 1, length, length), dtype=dtype, device=note_mask.device
    )
    return result.masked_fill(~allowed, torch.finfo(dtype).min)


def _generation_step_attention_mask(
    note_mask: torch.Tensor,
    ar_key_mask: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    key_valid = torch.cat((note_mask.bool(), ar_key_mask.bool()), dim=1)
    result = torch.zeros(
        (key_valid.shape[0], 1, 1, key_valid.shape[1]),
        dtype=dtype,
        device=note_mask.device,
    )
    return result.masked_fill(
        ~key_valid[:, None, None, :], torch.finfo(dtype).min
    )


class _FreshPrefixLMStack(T5GemmaPreTrainedModel):
    """Fresh T5Gemma encoder layers used strictly as self-attention blocks."""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.layers = nn.ModuleList(
            [
                T5GemmaEncoderLayer(config, layer_index)
                for layer_index in range(config.num_hidden_layers)
            ]
        )
        self.norm = T5GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = T5GemmaRotaryEmbedding(config=config)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.post_init()


@dataclass
class DecoderOnlyStage2Output:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    loss_numerator: torch.Tensor | None = None
    loss_denominator: torch.Tensor | None = None
    prefix_hidden_states: torch.Tensor | None = None
    hidden_states: torch.Tensor | None = None
    decoder_input_ids: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None

    @property
    def note_logits(self) -> torch.Tensor:
        """Expose ``[B,N,4,4]`` for existing overlap-logit averaging utilities."""

        batch, flat_length, classes = self.logits.shape
        if flat_length % PEDAL_SLOTS or classes != NUM_CLASSES:
            raise ValueError("flat pedal logits cannot be restored to note slots")
        return self.logits.view(
            batch, flat_length // PEDAL_SLOTS, PEDAL_SLOTS, classes
        )


class FourClassPedalDecoderOnlyModel(nn.Module):
    """Pretrained PT note embeddings plus a fresh Prefix-LM Transformer."""

    target_name = "pedal_targets"

    def __init__(
        self,
        performance_embeddings: nn.Module,
        transformer_config: Any,
        *,
        num_layers: int = DEFAULT_NUM_LAYERS,
        init_seed: int = DEFAULT_INIT_SEED,
    ) -> None:
        super().__init__()
        if int(num_layers) <= 0:
            raise ValueError("num_layers must be positive")
        self.hidden_size = int(getattr(performance_embeddings, "hidden_size"))
        config = copy.deepcopy(transformer_config)
        if int(getattr(config, "hidden_size")) != self.hidden_size:
            raise ValueError("performance embedding and Transformer hidden sizes differ")
        config.num_hidden_layers = int(num_layers)
        source_layer_types = list(getattr(config, "layer_types", ["full_attention"]))
        if not source_layer_types:
            source_layer_types = ["full_attention"]
        config.layer_types = [
            source_layer_types[index % len(source_layer_types)]
            for index in range(int(num_layers))
        ]
        config.vocab_size = DECODER_VOCAB_SIZE
        config.pad_token_id = PAD_ID
        config.bos_token_id = BOS_ID
        config.eos_token_id = PAD_ID
        config.is_decoder = False
        config.use_cache = True
        config._attn_implementation = "eager"

        self.performance_embeddings = performance_embeddings
        self.num_layers = int(num_layers)
        self.init_seed = int(init_seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.init_seed)
            self.transformer = _FreshPrefixLMStack(config)
            self.pedal_embeddings = nn.Embedding(
                DECODER_VOCAB_SIZE, self.hidden_size, padding_idx=PAD_ID
            )
            self.slot_embeddings = nn.Embedding(PEDAL_SLOTS, self.hidden_size)
            self.output_head = nn.Linear(self.hidden_size, NUM_CLASSES)
            initializer_range = float(config.initializer_range)
            nn.init.normal_(
                self.pedal_embeddings.weight, mean=0.0, std=initializer_range
            )
            with torch.no_grad():
                self.pedal_embeddings.weight[PAD_ID].zero_()
            nn.init.normal_(
                self.slot_embeddings.weight, mean=0.0, std=initializer_range
            )
            nn.init.normal_(self.output_head.weight, mean=0.0, std=initializer_range)
            nn.init.zeros_(self.output_head.bias)

    @classmethod
    def from_pretrained_performance_embeddings(
        cls,
        checkpoint_path: str | Path,
        *,
        num_layers: int = DEFAULT_NUM_LAYERS,
        init_seed: int = DEFAULT_INIT_SEED,
        **pretrained_kwargs: Any,
    ) -> "FourClassPedalDecoderOnlyModel":
        """Load only PT token embedding/projection/note-compression weights."""

        from third_party.PianistTransformer.src.model.pianoformer import PianoT5Gemma

        full_model = PianoT5Gemma.from_pretrained(
            str(checkpoint_path), **pretrained_kwargs
        )
        performance_embeddings = full_model.get_encoder().embeddings
        transformer_config = copy.deepcopy(full_model.config.decoder)
        full_model.model.encoder.embeddings = None
        del full_model
        return cls(
            performance_embeddings,
            transformer_config,
            num_layers=num_layers,
            init_seed=init_seed,
        )

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

    def pretrained_representation_parameters(self) -> Iterable[nn.Parameter]:
        return self.performance_embeddings.parameters()

    def fresh_parameters(self) -> Iterable[nn.Parameter]:
        for module in (
            self.transformer,
            self.pedal_embeddings,
            self.slot_embeddings,
            self.output_head,
        ):
            yield from module.parameters()

    def _validate_inputs(
        self,
        input_ids: torch.Tensor,
        token_attention_mask: torch.Tensor,
        note_mask: torch.Tensor,
    ) -> tuple[int, int]:
        if input_ids.ndim != 2 or input_ids.shape[1] % TOKENS_PER_NOTE:
            raise ValueError("input_ids must have shape [B,N*8]")
        if token_attention_mask.shape != input_ids.shape:
            raise ValueError("token_attention_mask must match input_ids")
        batch, token_length = input_ids.shape
        notes = token_length // TOKENS_PER_NOTE
        if tuple(note_mask.shape) != (batch, notes):
            raise ValueError("note_mask must have shape [B,N]")
        if bool((note_mask.long().sum(dim=1) <= 0).any()):
            raise ValueError("each sample must contain at least one valid note")
        expected_token_mask = note_mask.bool().repeat_interleave(
            TOKENS_PER_NOTE, dim=1
        )
        if not torch.equal(token_attention_mask.bool(), expected_token_mask):
            raise ValueError("token_attention_mask and note_mask are inconsistent")
        return batch, notes

    def build_performance_prefix(
        self,
        input_ids: torch.Tensor,
        token_attention_mask: torch.Tensor,
        note_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Embed only Pitch/IOI/Velocity/Duration plus four MASK tokens."""

        batch, notes = self._validate_inputs(
            input_ids, token_attention_mask, note_mask
        )
        note_tokens = input_ids.view(batch, notes, TOKENS_PER_NOTE)
        safe_tokens = note_tokens.clone()
        valid_pedals = note_mask.bool().unsqueeze(-1).expand(-1, -1, PEDAL_SLOTS)
        pedal_tokens = safe_tokens[:, :, 4:]
        pedal_tokens[valid_pedals] = MASK_ID
        prefix = self.performance_embeddings(safe_tokens.reshape(batch, -1))
        expected = (batch, notes, self.hidden_size)
        if tuple(prefix.shape) != expected:
            raise RuntimeError(f"performance prefix must have shape {expected}")
        return prefix * note_mask.to(prefix.dtype).unsqueeze(-1)

    def _pedal_input_embeddings(
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
        return self.pedal_embeddings(token_ids) + self.slot_embeddings(slots)

    @staticmethod
    def _run_layer(
        layer: T5GemmaEncoderLayer,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        *,
        cache: DynamicCache | None = None,
        cache_position: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        states = layer.pre_self_attn_layernorm(hidden_states)
        states, _ = layer.self_attn(
            hidden_states=states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=cache,
            cache_position=cache_position,
        )
        states = layer.post_self_attn_layernorm(states)
        hidden_states = residual + layer.dropout(states)
        residual = hidden_states
        states = layer.pre_feedforward_layernorm(hidden_states)
        states = layer.mlp(states)
        states = layer.post_feedforward_layernorm(states)
        return residual + layer.dropout(states)

    def _run_stack(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        *,
        cache: DynamicCache | None = None,
        cache_position: torch.Tensor | None = None,
    ) -> torch.Tensor:
        position_embeddings = self.transformer.rotary_emb(
            hidden_states, position_ids
        )
        states = hidden_states * torch.tensor(
            math.sqrt(self.hidden_size),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        states = self.transformer.dropout(states)
        for layer in self.transformer.layers:
            states = self._run_layer(
                layer,
                states,
                position_embeddings,
                attention_mask,
                cache=cache,
                cache_position=cache_position,
            )
        states = self.transformer.norm(states)
        return self.transformer.dropout(states)

    @staticmethod
    def _loss_components(
        logits: torch.Tensor, flat_targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        numerator = F.cross_entropy(
            logits.reshape(-1, NUM_CLASSES),
            flat_targets.reshape(-1),
            ignore_index=IGNORE_INDEX,
            reduction="sum",
        )
        denominator = (flat_targets != IGNORE_INDEX).sum().to(logits.dtype)
        if not bool(torch.isfinite(numerator)) or float(denominator.item()) <= 0.0:
            raise FloatingPointError("unweighted four-class CE components are invalid")
        return numerator / denominator, numerator, denominator

    def forward(
        self,
        input_ids: torch.Tensor,
        token_attention_mask: torch.Tensor,
        pedal_targets: torch.Tensor,
        note_mask: torch.Tensor,
    ) -> DecoderOnlyStage2Output:
        prefix = self.build_performance_prefix(
            input_ids, token_attention_mask, note_mask
        )
        batch, notes, _ = prefix.shape
        expected_targets = (batch, notes, PEDAL_SLOTS)
        if tuple(pedal_targets.shape) != expected_targets:
            raise ValueError(f"pedal_targets must have shape {expected_targets}")
        expected_valid = note_mask.bool().unsqueeze(-1).expand_as(pedal_targets)
        if not torch.equal(pedal_targets != IGNORE_INDEX, expected_valid):
            raise ValueError("note_mask and target padding are inconsistent")
        flat_targets = flatten_pedal_targets(pedal_targets)
        decoder_inputs, ar_mask = shift_right_four_class_targets(flat_targets)
        ar_embeddings = self._pedal_input_embeddings(decoder_inputs)
        combined = torch.cat((prefix, ar_embeddings), dim=1)
        positions = torch.arange(
            combined.shape[1], device=combined.device
        ).unsqueeze(0)
        attention_mask = build_prefix_lm_attention_mask(
            note_mask, ar_mask, dtype=combined.dtype
        )
        hidden = self._run_stack(combined, attention_mask, positions)
        ar_hidden = hidden[:, notes:]
        logits = self.output_head(ar_hidden)
        expected_logits = (batch, notes * PEDAL_SLOTS, NUM_CLASSES)
        if tuple(logits.shape) != expected_logits:
            raise AssertionError(f"pedal logits must have shape {expected_logits}")
        loss, numerator, denominator = self._loss_components(logits, flat_targets)
        return DecoderOnlyStage2Output(
            logits=logits,
            loss=loss,
            loss_numerator=numerator,
            loss_denominator=denominator,
            prefix_hidden_states=prefix,
            hidden_states=ar_hidden,
            decoder_input_ids=decoder_inputs,
            attention_mask=attention_mask,
        )

    @torch.no_grad()
    def _generate_free_running(
        self,
        input_ids: torch.Tensor,
        token_attention_mask: torch.Tensor,
        note_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return greedy classes and their true free-running step logits."""

        prefix = self.build_performance_prefix(
            input_ids, token_attention_mask, note_mask
        )
        batch, notes, _ = prefix.shape
        prefix_positions = torch.arange(notes, device=prefix.device).unsqueeze(0)
        prefix_attention = _prefix_only_attention_mask(
            note_mask, dtype=prefix.dtype
        )
        cache = DynamicCache()
        self._run_stack(
            prefix,
            prefix_attention,
            prefix_positions,
            cache=cache,
            cache_position=torch.arange(notes, device=prefix.device),
        )

        lengths = note_mask.long().sum(dim=1) * PEDAL_SLOTS
        full_length = notes * PEDAL_SLOTS
        previous = torch.full(
            (batch, 1), BOS_ID, dtype=torch.long, device=prefix.device
        )
        predictions = torch.zeros(
            (batch, full_length), dtype=torch.long, device=prefix.device
        )
        logits_steps: list[torch.Tensor] = []
        for step in range(full_length):
            active = step < lengths
            current_key_mask = (
                torch.arange(step + 1, device=prefix.device).unsqueeze(0)
                < lengths.unsqueeze(1)
            )
            step_attention = _generation_step_attention_mask(
                note_mask, current_key_mask, dtype=prefix.dtype
            )
            step_hidden = self._pedal_input_embeddings(
                previous, start_position=step
            )
            absolute_position = notes + step
            decoded = self._run_stack(
                step_hidden,
                step_attention,
                torch.tensor(
                    [[absolute_position]], dtype=torch.long, device=prefix.device
                ),
                cache=cache,
                cache_position=torch.tensor(
                    [absolute_position], dtype=torch.long, device=prefix.device
                ),
            )
            step_logits = self.output_head(decoded[:, -1])
            if not bool(torch.isfinite(step_logits).all()):
                raise FloatingPointError("greedy generation produced non-finite logits")
            logits_steps.append(step_logits)
            prediction = step_logits.argmax(dim=-1)
            predictions[:, step] = torch.where(
                active, prediction, torch.zeros_like(prediction)
            )
            previous = torch.where(
                (step + 1) < lengths,
                prediction,
                torch.full_like(prediction, PAD_ID),
            ).unsqueeze(1)
        classes = predictions.view(batch, notes, PEDAL_SLOTS)
        logits = torch.stack(logits_steps, dim=1).view(
            batch, notes, PEDAL_SLOTS, NUM_CLASSES
        )
        return classes, logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        token_attention_mask: torch.Tensor,
        note_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Greedily generate class IDs as ``[B,N,4]`` without targets."""

        predictions, _ = self._generate_free_running(
            input_ids, token_attention_mask, note_mask
        )
        return predictions

    @torch.no_grad()
    def generate_with_logits(
        self,
        input_ids: torch.Tensor,
        token_attention_mask: torch.Tensor,
        note_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Expose free-running classes/logits for canonical overlap averaging."""

        return self._generate_free_running(
            input_ids, token_attention_mask, note_mask
        )
