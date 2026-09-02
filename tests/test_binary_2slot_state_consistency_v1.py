#!/usr/bin/env python3
"""Focused executable tests for the opt-in parity state loss."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage2_binary_2slot.state_consistency import (
    required_toggle_targets,
    state_consistency_loss,
)


def logits(probabilities):
    return torch.tensor([[math.log(value) for value in probabilities]], dtype=torch.float64)


def loss(probabilities, current, human_next):
    return float(state_consistency_loss(
        logits(probabilities), torch.tensor([current]), torch.tensor([human_next])
    ))


def test_off_to_off_no_toggle_and_p1_monotonic():
    assert required_toggle_targets(torch.tensor([0]), torch.tensor([0])).item() == 0
    assert loss((0.7, 0.2, 0.1), 0, 0) < loss((0.5, 0.4, 0.1), 0, 0)


def test_off_to_on_toggle_and_p1_monotonic():
    assert required_toggle_targets(torch.tensor([0]), torch.tensor([1])).item() == 1
    assert loss((0.5, 0.4, 0.1), 0, 1) < loss((0.7, 0.2, 0.1), 0, 1)


def test_on_to_on_requires_no_toggle():
    assert required_toggle_targets(torch.tensor([1]), torch.tensor([1])).item() == 0


def test_on_to_off_requires_toggle():
    assert required_toggle_targets(torch.tensor([1]), torch.tensor([0])).item() == 1


def test_p0_p2_redistribution_with_fixed_p1_is_invariant():
    first = loss((0.7, 0.2, 0.1), 0, 0)
    second = loss((0.1, 0.2, 0.7), 0, 0)
    assert math.isclose(first, second, rel_tol=0.0, abs_tol=1e-12)


def run_directly():
    tests = {
        name: value for name, value in globals().items()
        if name.startswith("test_") and callable(value)
    }
    result = {}
    for name, function in sorted(tests.items()):
        function()
        result[name] = "passed"
    return result


if __name__ == "__main__":
    print(run_directly())
