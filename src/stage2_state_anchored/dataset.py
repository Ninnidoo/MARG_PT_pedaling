"""Canonical 512/256 window integration for State-Anchored Transition v1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from src.stage2_binary_2slot.dataset import (
    Binary2SlotPerformanceDataset,
    binary_2slot_collate_fn,
)

from .config import IGNORE_INDEX, STRIDE_NOTES, WINDOW_NOTES
from .targets import MODE_NAMES, StateAnchoredPerformanceTargets, build_performance_targets


MODE_TO_ID = {name: index for index, name in enumerate(MODE_NAMES)}


@dataclass(frozen=True)
class OwnershipDiagnostics:
    split: str
    performances: int
    state_targets: int
    mode_interval_targets: int
    state_zero_owner: int
    state_duplicate_owner: int
    mode_zero_owner: int
    mode_duplicate_owner: int
    performance_boundary_leakage: int

    @property
    def passed(self) -> bool:
        return not any(
            (
                self.state_zero_owner,
                self.state_duplicate_owner,
                self.mode_zero_owner,
                self.mode_duplicate_owner,
                self.performance_boundary_leakage,
            )
        )


def build_window_supervision(
    targets: StateAnchoredPerformanceTargets,
    owned_onset_indices: torch.Tensor | np.ndarray | Sequence[int],
) -> dict[str, torch.Tensor]:
    """Attach State to every owner onset and Mode/Timing to owner indices < M-1."""

    owned = torch.as_tensor(owned_onset_indices, dtype=torch.long)
    if owned.ndim != 1:
        raise ValueError("owned onset indices must be one-dimensional")
    onset_count = len(targets.onset_states)
    if bool(torch.any((owned < 0) | (owned >= onset_count))):
        raise ValueError("owned onset index lies outside its performance")
    if len(owned) > 1 and bool(torch.any(owned[1:] <= owned[:-1])):
        raise ValueError("owned onset indices must be strictly chronological")

    state_targets = torch.tensor(
        [targets.onset_states[int(index)] for index in owned], dtype=torch.long
    )
    mode_targets = torch.full((len(owned),), IGNORE_INDEX, dtype=torch.long)
    timing_targets = torch.zeros((len(owned), 2), dtype=torch.float32)
    timing_mask = torch.zeros((len(owned), 2), dtype=torch.bool)
    mode_valid = torch.zeros((len(owned),), dtype=torch.bool)
    for local_index, global_index_tensor in enumerate(owned):
        global_index = int(global_index_tensor)
        if global_index >= len(targets.intervals):
            continue
        target = targets.intervals[global_index]
        if target.onset_index != global_index:
            raise AssertionError("interval target index differs from owner onset")
        mode_targets[local_index] = MODE_TO_ID[target.mode]
        timing_targets[local_index] = torch.tensor(target.timing_targets)
        timing_mask[local_index] = torch.tensor(target.timing_mask)
        mode_valid[local_index] = True
    return {
        "state_targets": state_targets,
        "state_valid_mask": torch.ones(len(owned), dtype=torch.bool),
        "mode_targets": mode_targets,
        "mode_valid_mask": mode_valid,
        "timing_targets": timing_targets,
        "timing_mask": timing_mask,
    }


class StateAnchoredWindowDataset(Binary2SlotPerformanceDataset):
    """Read-only adapter over the canonical Binary 2-Slot data infrastructure."""

    def __init__(
        self,
        cache_root,
        split: str,
        *,
        window_notes: int = WINDOW_NOTES,
        stride_notes: int = STRIDE_NOTES,
        cache_input_tokens: bool = False,
        entry_limit: int | None = None,
        performance_indices: Sequence[int] | None = None,
        alignment_root=None,
    ) -> None:
        kwargs: dict[str, Any] = {
            "window_notes": window_notes,
            "stride_notes": stride_notes,
            "cache_input_tokens": cache_input_tokens,
            "entry_limit": entry_limit,
            "performance_indices": performance_indices,
        }
        if alignment_root is not None:
            kwargs["alignment_root"] = alignment_root
        super().__init__(cache_root, split, **kwargs)
        self._state_anchored_target_cache: dict[int, StateAnchoredPerformanceTargets] = {}

    def performance_targets(self, performance_index: int) -> StateAnchoredPerformanceTargets:
        if performance_index not in self._state_anchored_target_cache:
            self._state_anchored_target_cache[performance_index] = build_performance_targets(
                self.timeline(performance_index)
            )
        return self._state_anchored_target_cache[performance_index]

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = super().__getitem__(index)
        performance_index = int(sample["metadata"]["performance_index"])
        supervision = build_window_supervision(
            self.performance_targets(performance_index),
            sample["owned_global_onset_indices"],
        )
        # Do not expose the parent legacy final [t_M,T_end) raw primitive.
        # RUN B returns only the frozen M State and M-1 interval tensors.
        sample.pop("human_intervals")
        return {**sample, **supervision}


def state_anchored_collate_fn(
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Pad canonical windows and aligned State/Mode/Timing owner targets."""

    compatibility_samples = [{**sample, "human_intervals": ()} for sample in samples]
    batch = binary_2slot_collate_fn(compatibility_samples)
    batch.pop("human_intervals")
    batch_size, max_owned = batch["owned_onset_mask"].shape
    state_targets = torch.full((batch_size, max_owned), IGNORE_INDEX, dtype=torch.long)
    mode_targets = torch.full((batch_size, max_owned), IGNORE_INDEX, dtype=torch.long)
    timing_targets = torch.zeros((batch_size, max_owned, 2), dtype=torch.float32)
    timing_mask = torch.zeros((batch_size, max_owned, 2), dtype=torch.bool)
    state_valid = torch.zeros((batch_size, max_owned), dtype=torch.bool)
    mode_valid = torch.zeros((batch_size, max_owned), dtype=torch.bool)
    for batch_index, sample in enumerate(samples):
        count = len(sample["state_targets"])
        state_targets[batch_index, :count] = sample["state_targets"]
        mode_targets[batch_index, :count] = sample["mode_targets"]
        timing_targets[batch_index, :count] = sample["timing_targets"]
        timing_mask[batch_index, :count] = sample["timing_mask"]
        state_valid[batch_index, :count] = sample["state_valid_mask"]
        mode_valid[batch_index, :count] = sample["mode_valid_mask"]
    if not torch.equal(state_valid, batch["owned_onset_mask"].bool()):
        raise AssertionError("State supervision must cover every owned onset exactly once")
    if not torch.equal(state_targets != IGNORE_INDEX, state_valid):
        raise AssertionError("State target padding differs from valid mask")
    if not torch.equal(mode_targets != IGNORE_INDEX, mode_valid):
        raise AssertionError("Mode target padding differs from valid mask")
    if bool(torch.any(timing_mask & ~mode_valid.unsqueeze(-1))):
        raise AssertionError("invalid Mode placeholder has active timing supervision")
    batch.update(
        state_targets=state_targets,
        state_valid_mask=state_valid,
        mode_targets=mode_targets,
        mode_valid_mask=mode_valid,
        timing_targets=timing_targets,
        timing_mask=timing_mask,
    )
    return batch


def diagnose_ownership(dataset: StateAnchoredWindowDataset) -> OwnershipDiagnostics:
    """Count zero/duplicate owners without parsing MIDI or materializing targets."""

    state_zero = state_duplicate = mode_zero = mode_duplicate = leakage = 0
    state_targets = mode_targets = 0
    for performance in dataset.performances:
        onset_count = int(performance.num_onsets)
        membership = np.zeros(onset_count, dtype=np.int64)
        for owned in performance.ownership.owned_onset_indices:
            membership[owned] += 1
        state_zero += int(np.sum(membership == 0))
        state_duplicate += int(np.sum(membership > 1))
        state_targets += onset_count
        interval_membership = membership[:-1]
        mode_zero += int(np.sum(interval_membership == 0))
        mode_duplicate += int(np.sum(interval_membership > 1))
        mode_targets += max(0, onset_count - 1)
        # The final onset is intentionally State-only; no cross-performance
        # successor is ever manufactured for it.
        leakage += int(len(interval_membership) != max(0, onset_count - 1))
    result = OwnershipDiagnostics(
        split=dataset.split,
        performances=len(dataset.performances),
        state_targets=state_targets,
        mode_interval_targets=mode_targets,
        state_zero_owner=state_zero,
        state_duplicate_owner=state_duplicate,
        mode_zero_owner=mode_zero,
        mode_duplicate_owner=mode_duplicate,
        performance_boundary_leakage=leakage,
    )
    if not result.passed:
        raise AssertionError(f"State-Anchored ownership diagnostic failed: {result}")
    return result
