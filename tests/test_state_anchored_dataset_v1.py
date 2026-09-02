#!/usr/bin/env python3
"""Focused synthetic ownership/window tests for State-Anchored v1."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage2_binary_2slot.dataset import HumanIntervalPrimitive, PerformanceTimeline  # noqa: E402
from src.stage2_event_model.ownership import assign_unique_owners  # noqa: E402
from src.stage2_state_anchored.config import IGNORE_INDEX  # noqa: E402
from src.stage2_state_anchored.dataset import (  # noqa: E402
    build_window_supervision,
    state_anchored_collate_fn,
)
from src.stage2_state_anchored.targets import build_performance_targets  # noqa: E402


def performance(name: str, onset_count: int):
    main = tuple(
        HumanIntervalPrimitive("MAIN", index + 1, index, float(index), float(index + 1), 0, ())
        for index in range(onset_count)
    )
    return build_performance_targets(PerformanceTimeline(name, f"{name}.mid", None, main, None))  # type: ignore[arg-type]


def overlapping_ownership(onset_count: int):
    first = np.arange(onset_count, dtype=np.int64) * 2
    return assign_unique_owners(
        num_notes=onset_count * 2,
        first_note_indices=first,
        last_note_indices=first + 1,
        representative_note_indices=first + 1,
        window_notes=8,
        stride_notes=4,
    )


def test_synthetic_overlapping_windows_state_exactly_once() -> None:
    targets = performance("p", 9)
    ownership = overlapping_ownership(9)
    counts = np.zeros(9, dtype=np.int64)
    for owned in ownership.owned_onset_indices:
        supervision = build_window_supervision(targets, owned)
        assert int(supervision["state_valid_mask"].sum()) == len(owned)
        counts[owned] += 1
    assert np.array_equal(counts, np.ones(9, dtype=np.int64))


def test_synthetic_overlapping_windows_mode_exactly_once() -> None:
    targets = performance("p", 9)
    ownership = overlapping_ownership(9)
    counts = np.zeros(8, dtype=np.int64)
    for owned in ownership.owned_onset_indices:
        supervision = build_window_supervision(targets, owned)
        for index, valid in zip(owned, supervision["mode_valid_mask"]):
            if bool(valid):
                counts[int(index)] += 1
    assert np.array_equal(counts, np.ones(8, dtype=np.int64))


def test_performance_boundary_has_no_interval_leakage() -> None:
    for targets in (performance("left", 3), performance("right", 4)):
        owned = torch.arange(len(targets.onset_states))
        supervision = build_window_supervision(targets, owned)
        assert int(supervision["mode_valid_mask"].sum()) == len(targets.onset_states) - 1
        assert int(supervision["mode_targets"][-1]) == IGNORE_INDEX


def test_collate_padding_and_target_masks() -> None:
    def sample(targets, owned, notes):
        supervision = build_window_supervision(targets, torch.tensor(owned))
        tokens = torch.full((notes, 8), 7, dtype=torch.long)
        tokens[:, 4:] = 1
        return {
            "input_ids": tokens.reshape(-1),
            "note_mask": torch.ones(notes, dtype=torch.bool),
            "owned_global_onset_indices": torch.tensor(owned),
            "owned_representative_positions": torch.tensor(list(range(len(owned)))),
            "human_intervals": (),
            "metadata": {"performance_index": 0},
            **supervision,
        }

    batch = state_anchored_collate_fn((sample(performance("a", 3), [0, 1, 2], 4), sample(performance("b", 2), [0, 1], 3)))
    assert batch["state_targets"].shape == (2, 3)
    assert batch["mode_targets"].shape == (2, 3)
    assert batch["timing_targets"].shape == (2, 3, 2)
    assert batch["state_targets"][1, 2] == IGNORE_INDEX
    assert not bool(batch["timing_mask"].any())


def run_directly() -> dict[str, str]:
    tests = {name: value for name, value in globals().items() if name.startswith("test_")}
    result = {}
    for name, function in sorted(tests.items()):
        function()
        result[name] = "passed"
    return result


if __name__ == "__main__":
    print(run_directly())
