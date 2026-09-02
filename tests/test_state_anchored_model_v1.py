#!/usr/bin/env python3
"""Focused model/loss tests for State-Anchored Transition v1."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage2_state_anchored.config import IGNORE_INDEX, MODE_CLASS_WEIGHTS  # noqa: E402
from src.stage2_state_anchored.losses import StateAnchoredCriterion  # noqa: E402
from src.stage2_state_anchored.model import (  # noqa: E402
    StateAnchoredModelOutput,
    StateAnchoredTransitionModelV1,
)


class FakeEncoder(nn.Module):
    def __init__(self, hidden_size: int = 8) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size, dropout_rate=0.0, num_hidden_layers=2)
        self.embedding = nn.Embedding(32, hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        batch, flat = input_ids.shape
        hidden = self.embedding(input_ids).reshape(batch, flat // 8, 8, -1).mean(2)
        return SimpleNamespace(last_hidden_state=hidden)


def make_model() -> StateAnchoredTransitionModelV1:
    torch.manual_seed(42)
    return StateAnchoredTransitionModelV1(FakeEncoder(), hidden_size=8, dropout=0.0)


def model_batch():
    input_ids = torch.randint(2, 20, (2, 4 * 8))
    input_ids.reshape(2, 4, 8)[:, :, 4:] = 1
    return {
        "input_ids": input_ids,
        "token_attention_mask": torch.ones_like(input_ids),
        "note_mask": torch.ones((2, 4), dtype=torch.bool),
        "owned_representative_positions": torch.tensor([[0, 3], [1, -1]]),
        "owned_onset_mask": torch.tensor([[True, True], [True, False]]),
    }


def loss_batch():
    return {
        "state_targets": torch.tensor([[0, 1, 0]]),
        "state_valid_mask": torch.tensor([[True, True, True]]),
        "mode_targets": torch.tensor([[0, 1, 2]]),
        "mode_valid_mask": torch.tensor([[True, True, True]]),
        "timing_targets": torch.tensor([[[0.0, 0.0], [0.2, 0.0], [0.3, 0.8]]]),
        "timing_mask": torch.tensor([[[False, False], [True, False], [True, True]]]),
    }


def independent_output(requires_grad: bool = True) -> StateAnchoredModelOutput:
    state = torch.tensor([[[0.3, -0.2], [-0.1, 0.4], [0.2, -0.3]]], requires_grad=requires_grad)
    mode = torch.tensor([[[0.4, -0.2, -0.3], [-0.2, 0.3, -0.1], [-0.4, -0.2, 0.5]]], requires_grad=requires_grad)
    timing = torch.tensor([[[0.9, 0.7], [0.7, 0.8], [0.9, 0.1]]], requires_grad=requires_grad)
    hidden = torch.zeros((1, 3, 8))
    return StateAnchoredModelOutput(state, mode, timing, hidden, hidden)


def test_output_tensor_shapes_and_direct_heads() -> None:
    model = make_model()
    output = model(**model_batch())
    assert output.state_logits.shape == (2, 2, 2)
    assert output.mode_logits.shape == (2, 2, 3)
    assert output.timing_predictions.shape == (2, 2, 2)
    assert model.prediction_head_parameter_count == 63
    assert not hasattr(model, "state_embedding") and not hasattr(model, "recurrent_state")


def test_padding_and_invalid_target_masking() -> None:
    output = independent_output()
    batch = loss_batch()
    batch["state_targets"] = torch.tensor([[0, 1, IGNORE_INDEX]])
    batch["state_valid_mask"] = torch.tensor([[True, True, False]])
    batch["mode_targets"] = torch.tensor([[0, 1, IGNORE_INDEX]])
    batch["mode_valid_mask"] = torch.tensor([[True, True, False]])
    batch["timing_mask"] = torch.tensor([[[False, False], [True, False], [False, False]]])
    result = StateAnchoredCriterion()(output, batch)
    assert result.valid_state_targets == 2 and result.valid_mode_targets == 2
    assert result.valid_timing_targets == 1


def test_state_only_final_onset_batch_has_zero_mode_and_timing_loss() -> None:
    output = independent_output()
    batch = loss_batch()
    batch["mode_targets"][:] = IGNORE_INDEX
    batch["mode_valid_mask"][:] = False
    batch["timing_mask"][:] = False
    result = StateAnchoredCriterion()(output, batch)
    assert result.valid_state_targets == 3 and result.valid_mode_targets == 0
    assert result.mode_loss == 0 and result.timing_loss == 0
    assert torch.isfinite(result.total_loss)


def test_timing_masks_and_gradients_for_hold_change_return() -> None:
    output = independent_output()
    result = StateAnchoredCriterion()(output, loss_batch())
    result.total_loss.backward()
    gradient = output.timing_predictions.grad
    assert torch.equal(gradient[0, 0], torch.zeros(2))
    assert gradient[0, 1, 0] != 0 and gradient[0, 1, 1] == 0
    assert gradient[0, 2, 0] != 0 and gradient[0, 2, 1] != 0


def test_state_mode_total_losses_are_finite_and_weights_are_frozen() -> None:
    result = StateAnchoredCriterion()(independent_output(), loss_batch())
    assert all(torch.isfinite(value) for value in (result.state_loss, result.mode_loss, result.timing_loss, result.total_loss))
    assert MODE_CLASS_WEIGHTS == (0.3056523825702673, 0.9072671241146008, 1.7870804933151319)
    assert abs(sum(MODE_CLASS_WEIGHTS) / 3.0 - 1.0) < 1e-12


def test_backward_custom_heads_have_finite_gradients() -> None:
    model = make_model()
    output = model(**model_batch())
    batch = {
        "state_targets": torch.tensor([[0, 1], [1, IGNORE_INDEX]]),
        "state_valid_mask": torch.tensor([[True, True], [True, False]]),
        "mode_targets": torch.tensor([[0, 1], [2, IGNORE_INDEX]]),
        "mode_valid_mask": torch.tensor([[True, True], [True, False]]),
        "timing_targets": torch.tensor([[[0.0, 0.0], [0.2, 0.0]], [[0.3, 0.8], [0.0, 0.0]]]),
        "timing_mask": torch.tensor([[[False, False], [True, False]], [[True, True], [False, False]]]),
    }
    loss = StateAnchoredCriterion()(output, batch).total_loss
    loss.backward()
    for parameter in model.prediction_head_parameters():
        assert parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())


def run_directly() -> dict[str, str]:
    tests = {name: value for name, value in globals().items() if name.startswith("test_")}
    result = {}
    for name, function in sorted(tests.items()):
        function()
        result[name] = "passed"
    return result


if __name__ == "__main__":
    print(run_directly())
