"""Fixed endpoint-aware five-class representation for Stage 2 pedaling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .dataset import PEDAL_SLOTS, TOKENS_PER_NOTE, stage2_pedal_collate_fn


CLASS_NAMES = ("ZERO", "LOW", "MID", "HIGH", "FULL")
CLASS_BOUNDS = ((0, 0), (1, 63), (64, 95), (96, 126), (127, 127))
REPRESENTATIVES = np.asarray([0, 32, 80, 111, 127], dtype=np.int64)
NUM_CLASSES = 5
IGNORE_INDEX = -100


def _validate_integer_array(values: np.ndarray | torch.Tensor, name: str) -> None:
    if values.ndim < 1 or values.shape[-1] != PEDAL_SLOTS:
        raise ValueError(f"{name} must have shape [..., 4]")
    if isinstance(values, torch.Tensor):
        if values.dtype not in {
            torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8,
        }:
            raise TypeError(f"{name} must use an integer dtype")
    elif not np.issubdtype(values.dtype, np.integer):
        raise TypeError(f"{name} must use an integer dtype")


def classify_pedal_values(
    values: np.ndarray | torch.Tensor,
    *,
    allow_ignore: bool = False,
) -> np.ndarray | torch.Tensor:
    """Map raw CC64 values to the fixed classes without data-derived policy."""

    if not isinstance(values, (np.ndarray, torch.Tensor)):
        raise TypeError("values must be a NumPy array or torch Tensor")
    _validate_integer_array(values, "values")
    valid = (values >= 0) & (values <= 127)
    if allow_ignore:
        valid = valid | (values == IGNORE_INDEX)
    if isinstance(values, torch.Tensor):
        if not bool(torch.all(valid)):
            raise ValueError("values must be in [0,127] or the allowed ignore index")
        classes = torch.empty_like(values, dtype=torch.long)
        classes[values == 0] = 0
        classes[(values >= 1) & (values <= 63)] = 1
        classes[(values >= 64) & (values <= 95)] = 2
        classes[(values >= 96) & (values <= 126)] = 3
        classes[values == 127] = 4
        if allow_ignore:
            classes[values == IGNORE_INDEX] = IGNORE_INDEX
        return classes
    if not bool(np.all(valid)):
        raise ValueError("values must be in [0,127] or the allowed ignore index")
    classes = np.empty(values.shape, dtype=np.int64)
    classes[values == 0] = 0
    classes[(values >= 1) & (values <= 63)] = 1
    classes[(values >= 64) & (values <= 95)] = 2
    classes[(values >= 96) & (values <= 126)] = 3
    classes[values == 127] = 4
    if allow_ignore:
        classes[values == IGNORE_INDEX] = IGNORE_INDEX
    return classes


def decode_five_classes(
    classes: np.ndarray | torch.Tensor,
    *,
    allow_ignore: bool = False,
) -> np.ndarray | torch.Tensor:
    """Decode class IDs with the canonical interval-midpoint representatives."""

    if not isinstance(classes, (np.ndarray, torch.Tensor)):
        raise TypeError("classes must be a NumPy array or torch Tensor")
    _validate_integer_array(classes, "classes")
    valid = (classes >= 0) & (classes < NUM_CLASSES)
    if allow_ignore:
        valid = valid | (classes == IGNORE_INDEX)
    if isinstance(classes, torch.Tensor):
        if not bool(torch.all(valid)):
            raise ValueError("classes must be in [0,4] or the allowed ignore index")
        table = torch.as_tensor(REPRESENTATIVES, dtype=torch.long, device=classes.device)
        decoded = torch.full_like(classes, IGNORE_INDEX, dtype=torch.long)
        active = classes != IGNORE_INDEX if allow_ignore else torch.ones_like(classes, dtype=torch.bool)
        decoded[active] = table[classes[active].long()]
        return decoded
    if not bool(np.all(valid)):
        raise ValueError("classes must be in [0,4] or the allowed ignore index")
    decoded = np.full(classes.shape, IGNORE_INDEX, dtype=np.int64)
    active = classes != IGNORE_INDEX if allow_ignore else np.ones(classes.shape, dtype=bool)
    decoded[active] = REPRESENTATIVES[classes[active].astype(np.int64)]
    return decoded


def five_class_collate_fn(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Reuse baseline padding/masking, then replace only raw pedal targets."""

    batch = stage2_pedal_collate_fn(samples)
    batch["pedal_targets"] = classify_pedal_values(
        batch["pedal_targets"], allow_ignore=True
    )
    return batch


def average_five_class_logits(
    num_notes: int,
    windows: Sequence[tuple[int, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Average raw logits across windows before the single argmax decode."""

    if num_notes <= 0 or not windows:
        raise ValueError("num_notes and windows must be non-empty")
    starts = [int(start) for start, _ in windows]
    if len(starts) != len(set(starts)):
        raise ValueError("window starts must be unique")
    total = np.zeros((num_notes, PEDAL_SLOTS, NUM_CLASSES), dtype=np.float64)
    count = np.zeros(num_notes, dtype=np.int64)
    for start, logits in sorted(windows, key=lambda item: int(item[0])):
        values = np.asarray(logits)
        if values.ndim != 3 or values.shape[1:] != (PEDAL_SLOTS, NUM_CLASSES):
            raise ValueError("window logits must have shape [notes,4,5]")
        end = int(start) + len(values)
        if int(start) < 0 or end > num_notes or not len(values):
            raise ValueError("window lies outside the performance")
        if not np.isfinite(values).all():
            raise FloatingPointError("window logits are non-finite")
        total[int(start) : end] += values.astype(np.float64, copy=False)
        count[int(start) : end] += 1
    if np.any(count < 1) or count[0] < 1 or count[-1] < 1:
        raise RuntimeError("overlap reconstruction left notes uncovered")
    averaged = total / count[:, None, None]
    if not np.isfinite(averaged).all():
        raise FloatingPointError("averaged logits are non-finite")
    return averaged, count


@dataclass
class FiveClassPedalOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    hidden_states: torch.Tensor | None = None


class FiveClassPedalEncoderModel(nn.Module):
    """Official PT encoder with four independent Linear(768,5) heads."""

    def __init__(
        self,
        encoder: nn.Module,
        hidden_size: int | None = None,
        freeze_encoder: bool = False,
        dropout: float | None = None,
    ) -> None:
        super().__init__()
        encoder_config = getattr(encoder, "config", None)
        if hidden_size is None:
            hidden_size = getattr(encoder_config, "hidden_size", None)
        if hidden_size is None or int(hidden_size) <= 0:
            raise ValueError("hidden_size must be positive or provided by encoder.config")
        if dropout is None:
            dropout = float(getattr(encoder_config, "dropout_rate", 0.0))
        if not 0.0 <= float(dropout) <= 1.0:
            raise ValueError("dropout must be in [0,1]")
        self.encoder = encoder
        self.hidden_size = int(hidden_size)
        self.dropout = nn.Dropout(float(dropout))
        self.classification_heads = nn.ModuleList(
            [nn.Linear(self.hidden_size, NUM_CLASSES) for _ in range(PEDAL_SLOTS)]
        )
        self.set_encoder_frozen(freeze_encoder)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str | Path,
        freeze_encoder: bool = False,
        dropout: float | None = None,
        **pretrained_kwargs: Any,
    ) -> "FiveClassPedalEncoderModel":
        from third_party.PianistTransformer.src.model.pianoformer import PianoT5Gemma

        full_model = PianoT5Gemma.from_pretrained(str(checkpoint_path), **pretrained_kwargs)
        encoder = full_model.get_encoder()
        hidden_size = int(encoder.config.hidden_size)
        full_model.model.encoder = None
        del full_model
        return cls(encoder, hidden_size, freeze_encoder, dropout)

    def set_encoder_frozen(self, freeze: bool) -> None:
        self.freeze_encoder = bool(freeze)
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(not freeze)

    @staticmethod
    def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits.reshape(-1, NUM_CLASSES),
            targets.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        token_attention_mask: torch.Tensor,
        pedal_targets: torch.Tensor | None = None,
        note_mask: torch.Tensor | None = None,
    ) -> FiveClassPedalOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [B,N*8]")
        batch_size, flat_length = input_ids.shape
        if flat_length % TOKENS_PER_NOTE:
            raise ValueError("input length must be divisible by eight")
        if token_attention_mask.shape != input_ids.shape:
            raise ValueError("token_attention_mask shape mismatch")
        note_count = flat_length // TOKENS_PER_NOTE
        if pedal_targets is not None and pedal_targets.shape != (
            batch_size, note_count, PEDAL_SLOTS
        ):
            raise ValueError("pedal_targets must have shape [B,N,4]")
        if note_mask is not None and note_mask.shape != (batch_size, note_count):
            raise ValueError("note_mask must have shape [B,N]")
        if pedal_targets is not None:
            valid = (pedal_targets == IGNORE_INDEX) | (
                (pedal_targets >= 0) & (pedal_targets < NUM_CLASSES)
            )
            if not bool(torch.all(valid)):
                raise ValueError("pedal targets must be in [0,4] or -100")
        if pedal_targets is not None and note_mask is not None:
            expected = note_mask.bool().unsqueeze(-1).expand_as(pedal_targets)
            if not torch.equal(expected, pedal_targets != IGNORE_INDEX):
                raise ValueError("note mask and padded targets are inconsistent")
        encoded = self.encoder(input_ids=input_ids, attention_mask=token_attention_mask)
        hidden = getattr(encoded, "last_hidden_state", None)
        if hidden is None or hidden.shape != (batch_size, note_count, self.hidden_size):
            raise ValueError("encoder last_hidden_state has an invalid shape")
        dropped = self.dropout(hidden)
        logits = torch.stack([head(dropped) for head in self.classification_heads], dim=2)
        if logits.shape != (batch_size, note_count, PEDAL_SLOTS, NUM_CLASSES):
            raise RuntimeError("five-class logits have an invalid shape")
        loss = self.compute_loss(logits, pedal_targets) if pedal_targets is not None else None
        return FiveClassPedalOutput(logits=logits, loss=loss, hidden_states=hidden)
