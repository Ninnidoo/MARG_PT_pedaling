"""Encoder-only coarse-to-fine pedal heads and their exact v0 objective."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import torch
import torch.nn.functional as F
from torch import nn


IGNORE_INDEX = -100
PEDAL_SLOTS = 4
REGION_ZERO, REGION_INTERMEDIATE, REGION_FULL = 0, 1, 2
SMOOTH_L1_BETA = 0.05


class CoarseToFineLosses(NamedTuple):
    total: torch.Tensor
    region_ce: torch.Tensor
    depth_smooth_l1: torch.Tensor
    intermediate_count: int


@dataclass
class CoarseToFineOutput:
    region_logits: torch.Tensor
    depth_logits: torch.Tensor
    hidden_states: torch.Tensor


def pedal_regions(targets: torch.Tensor, ignore_index: int = IGNORE_INDEX) -> torch.Tensor:
    """Map 0/1..126/127 to ZERO/INTERMEDIATE/FULL, preserving -100."""

    if not bool(torch.all((targets == ignore_index) | ((targets >= 0) & (targets <= 127)))):
        raise ValueError("pedal targets must be in [0,127] or ignore_index")
    regions = torch.full_like(targets, ignore_index)
    valid = targets != ignore_index
    regions[valid & (targets == 0)] = REGION_ZERO
    regions[valid & (targets == 127)] = REGION_FULL
    regions[valid & (targets >= 1) & (targets <= 126)] = REGION_INTERMEDIATE
    return regions


def normalized_depth_targets(targets: torch.Tensor) -> torch.Tensor:
    return (targets.to(torch.float32) - 1.0) / 125.0


def decode_pedals(region_logits: torch.Tensor, depth_logits: torch.Tensor) -> torch.Tensor:
    """Decode raw outputs only after region/depth logits have been averaged."""

    if region_logits.shape[:-1] != depth_logits.shape or region_logits.shape[-1] != 3:
        raise ValueError("region logits [...,3] and depth logits [...] must align")
    regions = region_logits.argmax(dim=-1)
    intermediate = (1 + torch.round(125.0 * torch.sigmoid(depth_logits))).clamp(1, 126).to(torch.long)
    return torch.where(
        regions == REGION_ZERO,
        torch.zeros_like(intermediate),
        torch.where(regions == REGION_FULL, torch.full_like(intermediate, 127), intermediate),
    )


def coarse_to_fine_loss(
    region_logits: torch.Tensor,
    depth_logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
) -> CoarseToFineLosses:
    if tuple(region_logits.shape[:-1]) != tuple(targets.shape) or tuple(depth_logits.shape) != tuple(targets.shape):
        raise ValueError("region/depth outputs must match target shape")
    if region_logits.shape[-1] != 3:
        raise ValueError("region logits must have three classes")
    regions = pedal_regions(targets, ignore_index)
    valid = targets != ignore_index
    if not bool(valid.any()):
        raise ValueError("loss requires valid pedal targets")
    region_ce = F.cross_entropy(region_logits.reshape(-1, 3).float(), regions.reshape(-1), ignore_index=ignore_index)
    intermediate = regions == REGION_INTERMEDIATE
    count = int(intermediate.sum())
    if count:
        depth = F.smooth_l1_loss(
            torch.sigmoid(depth_logits[intermediate]).float(),
            normalized_depth_targets(targets[intermediate]),
            beta=SMOOTH_L1_BETA,
            reduction="mean",
        )
    else:
        # Preserve a zero gradient path for a valid all-endpoint batch.
        depth = depth_logits.sum() * 0.0
    return CoarseToFineLosses(region_ce + depth, region_ce, depth, count)


class Stage2CoarseToFineModel(nn.Module):
    """Official PT encoder with four independent region and depth heads."""

    def __init__(self, encoder: nn.Module, hidden_size: int | None = None, dropout: float | None = None) -> None:
        super().__init__()
        config = getattr(encoder, "config", None)
        self.encoder = encoder
        self.hidden_size = int(hidden_size or getattr(config, "hidden_size", 0))
        if self.hidden_size <= 0:
            raise ValueError("encoder hidden size is required")
        self.dropout = nn.Dropout(float(getattr(config, "dropout_rate", 0.0) if dropout is None else dropout))
        self.region_heads = nn.ModuleList([nn.Linear(self.hidden_size, 3) for _ in range(PEDAL_SLOTS)])
        self.depth_heads = nn.ModuleList([nn.Linear(self.hidden_size, 1) for _ in range(PEDAL_SLOTS)])
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(True)

    @classmethod
    def from_pretrained(cls, checkpoint_path: str | Path, **kwargs: Any) -> "Stage2CoarseToFineModel":
        from third_party.PianistTransformer.src.model.pianoformer import PianoT5Gemma
        full = PianoT5Gemma.from_pretrained(str(checkpoint_path), **kwargs)
        encoder = full.get_encoder()
        full.model.encoder = None
        del full
        return cls(encoder)

    def forward(self, input_ids: torch.Tensor, token_attention_mask: torch.Tensor, note_mask: torch.Tensor | None = None) -> CoarseToFineOutput:
        batch, flat = input_ids.shape
        if flat % 8 or token_attention_mask.shape != input_ids.shape:
            raise ValueError("input and attention must be aligned complete notes")
        notes = flat // 8
        if note_mask is not None and tuple(note_mask.shape) != (batch, notes):
            raise ValueError("note_mask shape mismatch")
        encoded = self.encoder(input_ids=input_ids, attention_mask=token_attention_mask)
        hidden = getattr(encoded, "last_hidden_state", None)
        if hidden is None or tuple(hidden.shape) != (batch, notes, self.hidden_size):
            raise ValueError("encoder must return [B,N,H] last_hidden_state")
        states = self.dropout(hidden)
        region = torch.stack([head(states) for head in self.region_heads], dim=2)
        depth = torch.stack([head(states).squeeze(-1) for head in self.depth_heads], dim=2)
        return CoarseToFineOutput(region, depth, hidden)
