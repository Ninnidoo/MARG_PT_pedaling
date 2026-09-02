#!/usr/bin/env python3
"""Focused raw/Viterbi/timing decoder tests for State-Anchored v1."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage2_state_anchored.decoding import (  # noqa: E402
    CHANGE_ID,
    HOLD_ID,
    RETURN_ID,
    decode_timings,
    raw_head_decode,
    state_mode_conflicts,
)
from src.stage2_state_anchored.viterbi import constrained_viterbi_decode  # noqa: E402
from src.stage2_state_anchored.inference import (  # noqa: E402
    assemble_performance_owner_logits,
    decode_performance,
)


def decode(states, modes):
    return constrained_viterbi_decode(torch.tensor(states, dtype=torch.float32), torch.tensor(modes, dtype=torch.float32))


def test_off_off_hold() -> None:
    result = decode([[8, -8], [8, -8]], [[7, 0, -2]])
    assert result.state_path.tolist() == [0, 0] and result.mode_path.tolist() == [HOLD_ID]


def test_on_on_return() -> None:
    result = decode([[-8, 8], [-8, 8]], [[0, -3, 7]])
    assert result.state_path.tolist() == [1, 1] and result.mode_path.tolist() == [RETURN_ID]


def test_off_on_forces_change() -> None:
    result = decode([[20, -20], [-20, 20]], [[10, 0, -10]])
    assert result.state_path.tolist() == [0, 1] and result.mode_path.tolist() == [CHANGE_ID]


def test_on_off_forces_change() -> None:
    result = decode([[-20, 20], [20, -20]], [[10, 0, -10]])
    assert result.state_path.tolist() == [1, 0] and result.mode_path.tolist() == [CHANGE_ID]


def test_raw_argmax_conflict_still_yields_legal_path() -> None:
    states = torch.tensor([[9.0, 0.0], [0.0, 9.0]])
    modes = torch.tensor([[9.0, 0.0, -1.0]])
    raw = raw_head_decode(states, modes, torch.tensor([[1.2, -0.2]]))
    assert raw.conflict_count == 1
    decoded = constrained_viterbi_decode(states, modes)
    assert decoded.conflict_count == 0


def test_multiple_interval_global_optimum_matches_exhaustive_search() -> None:
    states = torch.tensor([[1.2, 0.0], [0.1, 0.8], [1.0, 0.2], [0.0, 0.9]])
    modes = torch.tensor([[1.0, 0.7, -0.5], [-0.3, 1.2, 0.2], [0.5, 0.4, 1.1]])
    decoded = constrained_viterbi_decode(states, modes)
    state_logp, mode_logp = F.log_softmax(states, -1), F.log_softmax(modes, -1)
    best = None
    for path in itertools.product((0, 1), repeat=len(states)):
        score = sum(float(state_logp[i, value]) for i, value in enumerate(path))
        chosen = []
        for index, (left, right) in enumerate(zip(path, path[1:])):
            mode = CHANGE_ID if left != right else max((HOLD_ID, RETURN_ID), key=lambda item: float(mode_logp[index, item]))
            chosen.append(mode)
            score += float(mode_logp[index, mode])
        candidate = (score, tuple(-value for value in path), path, tuple(chosen))
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    assert tuple(decoded.state_path.tolist()) == best[2]
    assert tuple(decoded.mode_path.tolist()) == best[3]


def test_decoded_conflict_zero_and_deterministic() -> None:
    states = torch.zeros((5, 2))
    modes = torch.zeros((4, 3))
    first = constrained_viterbi_decode(states, modes)
    second = constrained_viterbi_decode(states, modes)
    assert state_mode_conflicts(first.state_path, first.mode_path)["count"] == 0
    assert torch.equal(first.state_path, second.state_path) and torch.equal(first.mode_path, second.mode_path)


def test_timing_clamp_and_return_chronology_policy() -> None:
    decoded = decode_timings(torch.tensor([[0.5, 0.7], [1.4, -0.2], [0.9, 0.2]]), [HOLD_ID, CHANGE_ID, RETURN_ID])
    assert decoded[0] == () and decoded[1] == (1.0,)
    assert abs(decoded[2][0] - 0.2) < 1e-6 and abs(decoded[2][1] - 0.9) < 1e-6


def test_overlapping_window_owner_logits_assemble_exactly_once_globally() -> None:
    chunks = [
        ([0, 1], torch.tensor([[2.0, 0.0], [0.0, 2.0]]), torch.zeros((2, 3)), torch.zeros((2, 2))),
        ([2, 3], torch.tensor([[2.0, 0.0], [0.0, 2.0]]), torch.zeros((2, 3)), torch.zeros((2, 2))),
    ]
    logits = assemble_performance_owner_logits(4, chunks)
    assert logits.state_logits.shape == (4, 2)
    assert logits.mode_logits.shape == (3, 3) and logits.timing_predictions.shape == (3, 2)
    assert decode_performance(logits).viterbi.conflict_count == 0


def test_duplicate_owner_logits_fail_fast() -> None:
    chunks = [
        ([0, 1], torch.zeros((2, 2)), torch.zeros((2, 3)), torch.zeros((2, 2))),
        ([1, 2], torch.zeros((2, 2)), torch.zeros((2, 3)), torch.zeros((2, 2))),
    ]
    try:
        assemble_performance_owner_logits(3, chunks)
    except AssertionError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("duplicate owner logits were accepted")


def run_directly() -> dict[str, str]:
    tests = {name: value for name, value in globals().items() if name.startswith("test_")}
    result = {}
    for name, function in sorted(tests.items()):
        function()
        result[name] = "passed"
    return result


if __name__ == "__main__":
    print(run_directly())
