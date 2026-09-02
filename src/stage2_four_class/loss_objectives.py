"""Loss-only variants for the frozen canonical four-class encoder architecture."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.stage2_binary.model import BinaryStage2EncoderBase
from src.stage2_encoder_only.dataset import PEDAL_SLOTS

from .model import FourClassPedalEncoderModel
from .representation import IGNORE_INDEX, NUM_CLASSES, REPRESENTATIVES


OBJECTIVES = ("weighted_ce", "ce_ntl_was", "representative_huber")
NTL_WAS_LAMBDA = 0.3
HUBER_DELTA_CC64 = 14.0
HUBER_DELTA_NORMALIZED = HUBER_DELTA_CC64 / 127.0


@dataclass
class ObjectiveOutput:
    """Training output with additive numerators for exact epoch aggregation."""

    predictions: torch.Tensor
    loss: torch.Tensor | None = None
    loss_numerator: torch.Tensor | None = None
    loss_denominator: torch.Tensor | None = None
    component_numerators: Mapping[str, torch.Tensor] = field(default_factory=dict)
    component_denominators: Mapping[str, torch.Tensor] = field(default_factory=dict)
    encoder_hidden_states: torch.Tensor | None = None

    @property
    def logits(self) -> torch.Tensor:
        return self.predictions


def inverse_sqrt_class_weights(
    counts: Sequence[int] | np.ndarray | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return raw ``1/sqrt(f)`` and mean-one normalized weights."""

    values = torch.as_tensor(counts, dtype=torch.float64)
    if tuple(values.shape) != (NUM_CLASSES,) or not bool(torch.all(values > 0)):
        raise ValueError("counts must contain four positive training frequencies")
    raw = values.rsqrt()
    normalized = raw / raw.mean()
    return raw, normalized


def representative_distance_matrix(
    *, device: torch.device | None = None, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    values = torch.tensor(REPRESENTATIVES, device=device, dtype=dtype)
    return (values[:, None] - values[None, :]).abs() / 127.0


def ntl_was_per_target(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Expected normalized representative CC64 distance for every target."""

    if logits.shape[:-1] != targets.shape or logits.shape[-1] != NUM_CLASSES:
        raise ValueError("logits/targets must have shapes [...,4] and [...]")
    valid = targets != IGNORE_INDEX
    illegal = valid & ((targets < 0) | (targets >= NUM_CLASSES))
    if bool(illegal.any()):
        raise ValueError("targets must be in [0,3] or -100")
    safe_targets = targets.masked_fill(~valid, 0)
    distances = representative_distance_matrix(device=logits.device, dtype=logits.dtype)
    target_distances = distances[:, safe_targets].movedim(0, -1)
    values = (logits.softmax(dim=-1) * target_distances).sum(dim=-1)
    return values.masked_fill(~valid, 0.0)


def classes_to_normalized_representatives(targets: torch.Tensor) -> torch.Tensor:
    """Map canonical classes to [0,51,79,127]/127, preserving ignore values."""

    valid = targets != IGNORE_INDEX
    illegal = valid & ((targets < 0) | (targets >= NUM_CLASSES))
    if bool(illegal.any()):
        raise ValueError("targets must be in [0,3] or -100")
    safe = targets.masked_fill(~valid, 0)
    representatives = torch.tensor(
        REPRESENTATIVES, dtype=torch.float32, device=targets.device
    )
    mapped = representatives[safe] / 127.0
    return mapped.masked_fill(~valid, float(IGNORE_INDEX))


def normalized_scalars_to_cc64(values: torch.Tensor) -> torch.Tensor:
    """Inference conversion: clip to [0,1], then scale continuously to CC64."""

    if not torch.is_floating_point(values):
        raise TypeError("regression predictions must be floating point")
    return values.clamp(0.0, 1.0) * 127.0


def continuous_cc64_to_classes(values: torch.Tensor) -> torch.Tensor:
    """Canonical boundary mapping for continuous clipped CC64 predictions."""

    if not torch.is_floating_point(values):
        raise TypeError("continuous CC64 predictions must be floating point")
    if not bool(torch.isfinite(values).all()):
        raise FloatingPointError("continuous CC64 predictions are non-finite")
    clipped = values.clamp(0.0, 127.0)
    # The integer convention for MIDI writing is nearest integer, half up.
    midi_integer = torch.floor(clipped + 0.5).to(torch.long)
    classes = torch.empty_like(midi_integer)
    classes[midi_integer <= 25] = 0
    classes[(midi_integer >= 26) & (midi_integer <= 63)] = 1
    classes[(midi_integer >= 64) & (midi_integer <= 103)] = 2
    classes[midi_integer >= 104] = 3
    return classes


class FourClassObjectiveEncoderModel(FourClassPedalEncoderModel):
    """Canonical Linear(768,4) heads with a selected loss formulation."""

    def __init__(
        self,
        encoder: nn.Module,
        hidden_size: int | None = None,
        dropout: float | None = None,
        *,
        objective: str,
        class_weights: Sequence[float] | torch.Tensor | None = None,
        ntl_lambda: float = NTL_WAS_LAMBDA,
    ) -> None:
        if objective not in {"standard_ce", "weighted_ce", "ce_ntl_was"}:
            raise ValueError(f"unsupported classification objective: {objective}")
        super().__init__(encoder, hidden_size, dropout)
        self.objective = objective
        self.ntl_lambda = float(ntl_lambda)
        if self.ntl_lambda != NTL_WAS_LAMBDA:
            raise ValueError("canonical NTL-WAS lambda must be 0.3")
        weights = (
            torch.ones(NUM_CLASSES, dtype=torch.float32)
            if class_weights is None
            else torch.as_tensor(class_weights, dtype=torch.float32)
        )
        if tuple(weights.shape) != (NUM_CLASSES,) or not bool(torch.all(weights > 0)):
            raise ValueError("class_weights must contain four positive values")
        if objective == "weighted_ce" and not torch.isclose(
            weights.mean(), torch.tensor(1.0), atol=1e-6, rtol=0.0
        ):
            raise ValueError("weighted CE weights must have mean one")
        # Non-persistent: classification checkpoints remain load-compatible with
        # the frozen canonical FourClassPedalEncoderModel evaluator.
        self.register_buffer("objective_class_weights", weights, persistent=False)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        *,
        objective: str,
        class_weights: Sequence[float] | torch.Tensor | None = None,
        ntl_lambda: float = NTL_WAS_LAMBDA,
        dropout: float | None = None,
        **pretrained_kwargs: object,
    ) -> "FourClassObjectiveEncoderModel":
        from third_party.PianistTransformer.src.model.pianoformer import PianoT5Gemma

        full_model = PianoT5Gemma.from_pretrained(
            str(checkpoint_path), **pretrained_kwargs
        )
        encoder = full_model.get_encoder()
        hidden_size = int(encoder.config.hidden_size)
        full_model.model.encoder = None
        del full_model
        return cls(
            encoder=encoder,
            hidden_size=hidden_size,
            dropout=dropout,
            objective=objective,
            class_weights=class_weights,
            ntl_lambda=ntl_lambda,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        token_attention_mask: torch.Tensor,
        pedal_targets: torch.Tensor | None = None,
        note_mask: torch.Tensor | None = None,
    ) -> ObjectiveOutput:
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
            return ObjectiveOutput(predictions=logits, encoder_hidden_states=hidden)

        flat_logits = logits.reshape(-1, NUM_CLASSES)
        flat_targets = pedal_targets.reshape(-1)
        valid = flat_targets != IGNORE_INDEX
        count = valid.sum().to(logits.dtype)
        ce_sum = F.cross_entropy(
            flat_logits,
            flat_targets,
            ignore_index=IGNORE_INDEX,
            reduction="sum",
        )
        components = {"ce": ce_sum}
        denominators = {"ce": count}
        if self.objective == "weighted_ce":
            weighted_sum = F.cross_entropy(
                flat_logits,
                flat_targets,
                weight=self.objective_class_weights.to(logits.dtype),
                ignore_index=IGNORE_INDEX,
                reduction="sum",
            )
            safe_targets = flat_targets.masked_fill(~valid, 0)
            weight_denominator = self.objective_class_weights[safe_targets[valid]].sum().to(
                logits.dtype
            )
            numerator, denominator = weighted_sum, weight_denominator
            components["weighted_ce"] = weighted_sum
            denominators["weighted_ce"] = weight_denominator
        elif self.objective == "ce_ntl_was":
            ntl_sum = ntl_was_per_target(logits, pedal_targets).sum()
            numerator, denominator = ce_sum - self.ntl_lambda * ntl_sum, count
            components["ntl_was"] = ntl_sum
            components["total"] = numerator
            denominators["ntl_was"] = count
            denominators["total"] = count
        else:
            numerator, denominator = ce_sum, count
        loss = numerator / denominator
        if not bool(torch.isfinite(loss)) or float(denominator.item()) <= 0.0:
            raise FloatingPointError("classification objective is invalid")
        return ObjectiveOutput(
            predictions=logits,
            loss=loss,
            loss_numerator=numerator,
            loss_denominator=denominator,
            component_numerators=components,
            component_denominators=denominators,
            encoder_hidden_states=hidden,
        )


class RepresentativeHuberEncoderModel(BinaryStage2EncoderBase):
    """Canonical PT encoder plus four independent unconstrained Linear(H,1) heads."""

    target_name = "pedal_targets"

    def __init__(
        self,
        encoder: nn.Module,
        hidden_size: int | None = None,
        dropout: float | None = None,
        *,
        delta: float = HUBER_DELTA_NORMALIZED,
    ) -> None:
        super().__init__(encoder, hidden_size, dropout)
        self.delta = float(delta)
        if self.delta != HUBER_DELTA_NORMALIZED:
            raise ValueError("canonical normalized Huber delta must be 14/127")
        self.regression_heads = nn.ModuleList(
            [nn.Linear(self.hidden_size, 1) for _ in range(PEDAL_SLOTS)]
        )

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        *,
        delta: float = HUBER_DELTA_NORMALIZED,
        dropout: float | None = None,
        **pretrained_kwargs: object,
    ) -> "RepresentativeHuberEncoderModel":
        from third_party.PianistTransformer.src.model.pianoformer import PianoT5Gemma

        full_model = PianoT5Gemma.from_pretrained(
            str(checkpoint_path), **pretrained_kwargs
        )
        encoder = full_model.get_encoder()
        hidden_size = int(encoder.config.hidden_size)
        full_model.model.encoder = None
        del full_model
        return cls(
            encoder=encoder,
            hidden_size=hidden_size,
            dropout=dropout,
            delta=delta,
        )

    def prediction_head_parameters(self) -> Iterable[nn.Parameter]:
        return self.regression_heads.parameters()

    def forward(
        self,
        input_ids: torch.Tensor,
        token_attention_mask: torch.Tensor,
        pedal_targets: torch.Tensor | None = None,
        note_mask: torch.Tensor | None = None,
    ) -> ObjectiveOutput:
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
        values = torch.cat([head(states) for head in self.regression_heads], dim=-1)
        expected = (batch_size, note_count, PEDAL_SLOTS)
        if tuple(values.shape) != expected:
            raise AssertionError(f"regression output must have shape {expected}")
        if pedal_targets is None:
            return ObjectiveOutput(predictions=values, encoder_hidden_states=hidden)

        targets = classes_to_normalized_representatives(pedal_targets).to(values.dtype)
        valid = pedal_targets != IGNORE_INDEX
        huber_sum = F.smooth_l1_loss(
            values[valid], targets[valid], beta=self.delta, reduction="sum"
        )
        count = valid.sum().to(values.dtype)
        clipped_cc64 = normalized_scalars_to_cc64(values[valid].float())
        target_cc64 = targets[valid].float() * 127.0
        mae_sum = (clipped_cc64 - target_cc64).abs().sum()
        loss = huber_sum / count
        if not bool(torch.isfinite(loss)) or float(count.item()) <= 0.0:
            raise FloatingPointError("representative Huber objective is invalid")
        return ObjectiveOutput(
            predictions=values,
            loss=loss,
            loss_numerator=huber_sum,
            loss_denominator=count,
            component_numerators={"huber": huber_sum, "representative_mae_cc64": mae_sum},
            component_denominators={"huber": count, "representative_mae_cc64": count},
            encoder_hidden_states=hidden,
        )
