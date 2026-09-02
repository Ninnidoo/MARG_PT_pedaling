"""Targeted tests for the canonical encoder-only four-class loss phase."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.stage2_four_class.loss_objectives import (
    HUBER_DELTA_NORMALIZED,
    NTL_WAS_LAMBDA,
    FourClassObjectiveEncoderModel,
    RepresentativeHuberEncoderModel,
    classes_to_normalized_representatives,
    continuous_cc64_to_classes,
    inverse_sqrt_class_weights,
    normalized_scalars_to_cc64,
    ntl_was_per_target,
    representative_distance_matrix,
)
from src.stage2_four_class.model import FourClassPedalEncoderModel
from src.stage2_four_class.representation import IGNORE_INDEX, REPRESENTATIVES


class DummyEncoder(nn.Module):
    def __init__(self, hidden_size: int = 12) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size, dropout_rate=0.0)
        self.embedding = nn.Embedding(6000, hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        batch, flat = input_ids.shape
        notes = flat // 8
        values = input_ids.view(batch, notes, 8)[:, :, 0]
        return SimpleNamespace(last_hidden_state=self.embedding(values))


def batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(17)
    input_ids = torch.randint(0, 5000, (2, 24), dtype=torch.long)
    attention = torch.ones_like(input_ids, dtype=torch.bool)
    note_mask = torch.tensor([[True, True, True], [True, True, False]])
    targets = torch.tensor(
        [
            [[0, 1, 2, 3], [3, 2, 1, 0], [0, 0, 3, 3]],
            [[1, 1, 2, 2], [3, 0, 2, 1], [IGNORE_INDEX] * 4],
        ],
        dtype=torch.long,
    )
    return input_ids, attention, note_mask, targets


def assert_finite_backward(model: nn.Module) -> None:
    input_ids, attention, note_mask, targets = batch()
    output = model(input_ids, attention, targets, note_mask)
    assert output.loss is not None and torch.isfinite(output.loss)
    output.loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(value).all() for value in gradients)


def test_weighted_ce_formula_shape_and_backward() -> None:
    counts = torch.tensor([10433898, 4447400, 7039938, 15555144])
    raw, normalized = inverse_sqrt_class_weights(counts)
    assert torch.allclose(raw, counts.double().rsqrt(), atol=0.0, rtol=0.0)
    assert torch.isclose(normalized.mean(), torch.tensor(1.0, dtype=torch.float64))
    model = FourClassObjectiveEncoderModel(
        DummyEncoder(), objective="weighted_ce", class_weights=normalized
    )
    input_ids, attention, note_mask, targets = batch()
    output = model(input_ids, attention, targets, note_mask)
    assert output.logits.shape == (2, 3, 4, 4)
    expected = F.cross_entropy(
        output.logits.reshape(-1, 4),
        targets.reshape(-1),
        weight=normalized.float(),
        ignore_index=IGNORE_INDEX,
    )
    assert torch.allclose(output.loss, expected, atol=1e-7, rtol=1e-7)
    assert_finite_backward(model)


def test_ntl_was_analytical_representative_distance_and_total() -> None:
    distances = representative_distance_matrix(dtype=torch.float64)
    expected = torch.tensor(
        [
            [0, 51, 79, 127],
            [51, 0, 28, 76],
            [79, 28, 0, 48],
            [127, 76, 48, 0],
        ],
        dtype=torch.float64,
    ) / 127.0
    assert torch.equal(distances, expected)

    target = torch.tensor([1], dtype=torch.long)
    exact = torch.log(torch.tensor([[0.01, 0.97, 0.01, 0.01]], dtype=torch.float64))
    near = torch.log(torch.tensor([[0.01, 0.48, 0.50, 0.01]], dtype=torch.float64))
    far = torch.log(torch.tensor([[0.01, 0.48, 0.01, 0.50]], dtype=torch.float64))
    exact_ntl = ntl_was_per_target(exact, target)
    near_ntl = ntl_was_per_target(near, target)
    far_ntl = ntl_was_per_target(far, target)
    assert exact_ntl < near_ntl < far_ntl
    manual = (0.01 * 51 + 0.48 * 0 + 0.01 * 28 + 0.50 * 76) / 127
    assert torch.allclose(far_ntl, torch.tensor([manual], dtype=torch.float64))

    logits = torch.log(torch.tensor([[[[0.1, 0.2, 0.3, 0.4]]]], dtype=torch.float64))
    targets = torch.tensor([[[2]]])
    ce = F.cross_entropy(logits.reshape(-1, 4), targets.reshape(-1))
    ntl = (0.1 * 79 + 0.2 * 28 + 0.3 * 0 + 0.4 * 48) / 127
    total = ce - NTL_WAS_LAMBDA * ntl
    model = FourClassObjectiveEncoderModel(DummyEncoder(), objective="ce_ntl_was")
    result = model
    # Direct formulation check avoids coupling the hand calculation to encoder weights.
    direct = ce - NTL_WAS_LAMBDA * ntl_was_per_target(logits, targets).mean()
    assert torch.allclose(direct, total)
    assert result.ntl_lambda == 0.3
    assert_finite_backward(model)


def test_representative_huber_mapping_heads_boundaries_and_backward() -> None:
    classes = torch.tensor([[[0, 1, 2, 3]]])
    normalized = classes_to_normalized_representatives(classes)
    assert torch.equal(
        normalized,
        torch.tensor(REPRESENTATIVES, dtype=torch.float32).view(1, 1, 4) / 127.0,
    )
    model = RepresentativeHuberEncoderModel(DummyEncoder())
    assert len(model.regression_heads) == 4
    assert all(head.out_features == 1 for head in model.regression_heads)
    assert model.delta == 14 / 127
    input_ids, attention, note_mask, targets = batch()
    output = model(input_ids, attention, targets, note_mask)
    assert output.predictions.shape == (2, 3, 4)
    assert HUBER_DELTA_NORMALIZED == 14 / 127
    assert_finite_backward(model)

    scaled = normalized_scalars_to_cc64(
        torch.tensor([-1.0, 0.0, 25 / 127, 26 / 127, 63 / 127, 64 / 127,
                      103 / 127, 104 / 127, 1.0, 2.0])
    )
    assert torch.equal(scaled[[0, -1]], torch.tensor([0.0, 127.0]))
    assert torch.equal(
        continuous_cc64_to_classes(scaled),
        torch.tensor([0, 0, 0, 1, 1, 2, 2, 3, 3, 3]),
    )


def test_standard_ce_shared_path_regression_is_exact() -> None:
    torch.manual_seed(9)
    canonical = FourClassPedalEncoderModel(DummyEncoder())
    shared = FourClassObjectiveEncoderModel(DummyEncoder(), objective="standard_ce")
    shared.load_state_dict(canonical.state_dict(), strict=True)
    input_ids, attention, note_mask, targets = batch()
    canonical_output = canonical(input_ids, attention, targets, note_mask)
    shared_output = shared(input_ids, attention, targets, note_mask)
    assert torch.equal(canonical_output.logits, shared_output.logits)
    assert torch.equal(canonical_output.loss, shared_output.loss)
    assert torch.equal(canonical_output.loss_numerator, shared_output.loss_numerator)
    assert torch.equal(canonical_output.loss_denominator, shared_output.loss_denominator)
