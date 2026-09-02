"""Canonical four-class CC64 representation and immutable-cache adapter."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from src.stage2_binary.full_training import SharedBinaryWindowDataset
from src.stage2_encoder_only.dataset import PEDAL_SLOTS, stage2_pedal_collate_fn


CLASS_NAMES = ("ZERO", "LOW", "HALF", "FULL")
CLASS_BOUNDS = ((0, 25), (26, 63), (64, 103), (104, 127))
REPRESENTATIVES = (0, 51, 79, 127)
NUM_CLASSES = 4
IGNORE_INDEX = -100


def classify_cc64(
    values: np.ndarray | torch.Tensor,
    *,
    allow_ignore: bool = False,
) -> np.ndarray | torch.Tensor:
    """Map raw CC64 to the frozen canonical 0..3 class IDs."""

    if not isinstance(values, (np.ndarray, torch.Tensor)):
        raise TypeError("values must be a NumPy array or torch Tensor")
    if values.ndim < 1 or values.shape[-1] != PEDAL_SLOTS:
        raise ValueError("values must have shape [...,4]")
    if isinstance(values, torch.Tensor):
        if values.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise TypeError("values must use an integer dtype")
    elif not np.issubdtype(values.dtype, np.integer):
        raise TypeError("values must use an integer dtype")
    valid = (values >= 0) & (values <= 127)
    if allow_ignore:
        valid = valid | (values == IGNORE_INDEX)
    all_valid = bool(torch.all(valid)) if isinstance(valid, torch.Tensor) else bool(np.all(valid))
    if not all_valid:
        raise ValueError("CC64 values must be in [0,127] or the allowed ignore index")
    if isinstance(values, torch.Tensor):
        result = torch.empty_like(values, dtype=torch.long)
    else:
        result = np.empty(values.shape, dtype=np.int64)
    result[(values >= 0) & (values <= 25)] = 0
    result[(values >= 26) & (values <= 63)] = 1
    result[(values >= 64) & (values <= 103)] = 2
    result[(values >= 104) & (values <= 127)] = 3
    if allow_ignore:
        result[values == IGNORE_INDEX] = IGNORE_INDEX
    return result


def four_class_collate_fn(
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Reuse established Stage 2 padding/masking and change targets only."""

    batch = stage2_pedal_collate_fn(samples)
    targets = classify_cc64(batch["pedal_targets"], allow_ignore=True)
    active = targets != IGNORE_INDEX
    if bool(active.any()) and (
        int(targets[active].min()) < 0 or int(targets[active].max()) >= NUM_CLASSES
    ):
        raise AssertionError("canonical targets escaped [0,3]")
    batch["pedal_targets"] = targets
    return batch


class SharedFourClassWindowDataset(SharedBinaryWindowDataset):
    """Read the already-built combined raw-token cache without modifying it."""


def make_four_class_loader(
    dataset: Dataset[dict[str, Any]],
    *,
    batch_size: int,
    pin_memory: bool,
    order: Sequence[int] | None = None,
) -> DataLoader:
    source: Dataset = Subset(dataset, list(order)) if order is not None else dataset
    return DataLoader(
        source,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=bool(pin_memory),
        collate_fn=four_class_collate_fn,
    )
