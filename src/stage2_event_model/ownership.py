"""Deterministic 512/256 supervision ownership for piece-level onset targets."""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.stage2_encoder_only.dataset import generate_window_starts


@dataclass(frozen=True)
class OwnershipResult:
    window_starts: tuple[int, ...]
    window_ends: tuple[int, ...]
    owner_window_index: np.ndarray
    owner_margin: np.ndarray
    owned_onset_indices: tuple[np.ndarray, ...]
    owned_local_representative_indices: tuple[np.ndarray, ...]
    split_chord_group_count: int
    split_chord_window_group_pairs: int


def assign_unique_owners(
    num_notes: int,
    first_note_indices: Sequence[int] | np.ndarray,
    last_note_indices: Sequence[int] | np.ndarray,
    representative_note_indices: Sequence[int] | np.ndarray,
    *,
    window_notes: int = 512,
    stride_notes: int = 256,
) -> OwnershipResult:
    """Assign every complete onset group to its maximum-margin eligible window."""

    first = np.asarray(first_note_indices, dtype=np.int64)
    last = np.asarray(last_note_indices, dtype=np.int64)
    representative = np.asarray(representative_note_indices, dtype=np.int64)
    if first.ndim != 1 or last.shape != first.shape or representative.shape != first.shape:
        raise ValueError("onset metadata must be equal-length one-dimensional arrays")
    if num_notes <= 0 or len(first) == 0:
        raise ValueError("ownership requires notes and onset groups")
    if np.any(first < 0) or np.any(first > last) or np.any(last >= num_notes):
        raise ValueError("onset group note bounds are invalid")
    if np.any((representative < first) | (representative > last)):
        raise ValueError("representative note must lie inside its onset group")

    starts = tuple(generate_window_starts(num_notes, window_notes, stride_notes))
    ends = tuple(min(start + window_notes, num_notes) for start in starts)
    owners = np.full(len(first), -1, dtype=np.int64)
    margins = np.full(len(first), -1, dtype=np.int64)
    split_groups = 0
    split_pairs = 0

    for onset_index, (group_first, group_last, rep) in enumerate(
        zip(first.tolist(), last.tolist(), representative.tolist())
    ):
        # Eligible: start <= first and start + window_notes > last.
        candidate_left = bisect.bisect_right(starts, group_last - window_notes)
        candidate_right = bisect.bisect_right(starts, group_first)
        eligible: list[int] = []
        for window_index in range(candidate_left, candidate_right):
            if starts[window_index] <= group_first and group_last < ends[window_index]:
                eligible.append(window_index)
        if not eligible:
            raise ValueError(
                f"onset {onset_index} [{group_first},{group_last}] has no complete owner"
            )
        best = max(
            eligible,
            key=lambda index: (
                min(rep - starts[index], (ends[index] - 1) - rep),
                -starts[index],
            ),
        )
        owners[onset_index] = best
        margins[onset_index] = min(
            rep - starts[best], (ends[best] - 1) - rep
        )

        intersect_left = bisect.bisect_right(starts, group_first - window_notes)
        intersect_right = bisect.bisect_right(starts, group_last)
        rejected = 0
        for window_index in range(intersect_left, intersect_right):
            intersects = starts[window_index] <= group_last and ends[window_index] > group_first
            contains = starts[window_index] <= group_first and group_last < ends[window_index]
            if intersects and not contains:
                rejected += 1
        if rejected:
            split_groups += 1
            split_pairs += rejected

    if np.any(owners < 0):
        raise AssertionError("zero-owner onset escaped validation")
    owned: list[np.ndarray] = []
    local: list[np.ndarray] = []
    for window_index, start in enumerate(starts):
        indices = np.flatnonzero(owners == window_index).astype(np.int64)
        positions = representative[indices] - start
        if np.any(positions < 0) or np.any(positions >= ends[window_index] - start):
            raise AssertionError("owned representative is outside its owner window")
        if np.any(first[indices] < start) or np.any(last[indices] >= ends[window_index]):
            raise AssertionError("owner window does not contain the full onset group")
        owned.append(indices)
        local.append(positions.astype(np.int64))
    if sum(len(indices) for indices in owned) != len(first):
        raise AssertionError("ownership does not partition onset groups exactly once")
    return OwnershipResult(
        window_starts=starts,
        window_ends=ends,
        owner_window_index=owners,
        owner_margin=margins,
        owned_onset_indices=tuple(owned),
        owned_local_representative_indices=tuple(local),
        split_chord_group_count=split_groups,
        split_chord_window_group_pairs=split_pairs,
    )
