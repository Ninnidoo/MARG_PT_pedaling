"""Frozen-cache dataset adapter for B3-S semantic pre-state supervision."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .dataset import (
    IGNORE_INDEX,
    MAIN_SLOTS,
    CustomEventWindowDataset,
    custom_event_collate_fn,
)


def derive_main_pre_state_targets(
    initial_state: int,
    main_event_targets: np.ndarray,
    main_interval_start_states: np.ndarray,
) -> np.ndarray:
    """Derive the state immediately before every frozen Main target slot."""

    events = np.asarray(main_event_targets)
    starts = np.asarray(main_interval_start_states)
    if events.ndim != 2 or events.shape[1] != MAIN_SLOTS:
        raise ValueError("main_event_targets must have shape [I,6]")
    if starts.shape != (events.shape[0],):
        raise ValueError("main_interval_start_states must have shape [I]")
    if not events.shape[0]:
        raise ValueError("pre-state derivation requires at least one Main interval")
    if np.any((events < 0) | (events > 4)):
        raise ValueError("Main event target outside [0,4]")
    if np.any((starts < 0) | (starts > 3)) or not 0 <= int(initial_state) <= 3:
        raise ValueError("state target outside [0,3]")
    if int(starts[0]) != int(initial_state):
        raise AssertionError("first Main interval does not start at Initial target")
    for row in events:
        none = np.flatnonzero(row == 0)
        if len(none) and np.any(row[int(none[0]) + 1 :] != 0):
            raise AssertionError("frozen first-NONE target grammar violated")

    pre = np.empty(events.shape, dtype=np.int8)
    pre[:, 0] = starts.astype(np.int8, copy=False)
    for slot in range(1, MAIN_SLOTS):
        previous = events[:, slot - 1]
        pre[:, slot] = np.where(previous == 0, pre[:, slot - 1], previous - 1)
    final = np.where(events[:, -1] == 0, pre[:, -1], events[:, -1] - 1)
    if len(final) > 1 and not np.array_equal(final[:-1], starts[1:]):
        raise AssertionError("derived Main final state disagrees with next interval start")
    return np.ascontiguousarray(pre)


class CustomEventStateConditionedWindowDataset(CustomEventWindowDataset):
    """v0 owner windows plus runtime-derived B3-S pre-state labels."""

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = super().__getitem__(index)
        performance_index, _ = self.windows[index]
        performance = self.performances[performance_index]
        owned = sample["owned_global_onset_indices"].numpy()
        with np.load(performance.cache_path, allow_pickle=False) as cache:
            pre = derive_main_pre_state_targets(
                int(cache["initial_state"]),
                cache["main_event_target"],
                cache["main_interval_start_state"],
            )
        sample["main_pre_state_targets"] = torch.from_numpy(
            pre[owned].astype(np.int64, copy=True)
        ).long()
        return sample


def custom_event_state_conditioned_collate_fn(
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    batch = custom_event_collate_fn(samples)
    batch_size, max_owned = batch["owned_onset_mask"].shape
    targets = torch.full(
        (batch_size, max_owned, MAIN_SLOTS), IGNORE_INDEX, dtype=torch.long
    )
    for batch_index, sample in enumerate(samples):
        count = len(sample["main_pre_state_targets"])
        if tuple(sample["main_pre_state_targets"].shape) != (count, MAIN_SLOTS):
            raise ValueError("main_pre_state_targets must have shape [O,6]")
        targets[batch_index, :count] = sample["main_pre_state_targets"]
    if not torch.equal(targets != IGNORE_INDEX, batch["owned_onset_mask"].unsqueeze(-1).expand_as(targets)):
        raise AssertionError("pre-state padding does not match unique ownership")
    batch["main_pre_state_targets"] = targets
    return batch
