"""Targeted raw-target tests for canonical Encoder-only Huber regression."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from src.stage2_four_class.loss_objectives import (
    HUBER_DELTA_NORMALIZED,
    RepresentativeHuberEncoderModel,
    continuous_cc64_to_classes,
    normalized_scalars_to_cc64,
)
from src.stage2_four_class.raw_cc64_huber import (
    RawCC64HuberEncoderModel,
    raw_cc64_to_normalized_targets,
)


class DummyEncoder(nn.Module):
    def __init__(self, hidden_size: int = 12) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size, dropout_rate=0.0)
        self.embedding = nn.Embedding(6000, hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        batch, flat = input_ids.shape
        notes = flat // 8
        first_tokens = input_ids.view(batch, notes, 8)[:, :, 0]
        return SimpleNamespace(last_hidden_state=self.embedding(first_tokens))


def test_raw_target_preservation_without_representative_quantization() -> None:
    raw = torch.tensor([[0, 18, 51, 57], [72, 79, 96, 121], [127, 25, 26, 63]])
    normalized = raw_cc64_to_normalized_targets(raw)
    assert torch.equal(normalized, raw.float() / 127.0)
    non_representatives = torch.tensor([18, 57, 72, 96, 121])
    converted = raw_cc64_to_normalized_targets(
        non_representatives.view(-1, 1).expand(-1, 4)
    )[:, 0]
    assert torch.equal(converted, non_representatives.float() / 127.0)
    assert len(torch.unique(converted)) == len(non_representatives)


def test_boundary_targets_remain_distinct_during_training() -> None:
    raw = torch.tensor([[25, 26, 63, 64], [103, 104, 0, 127]])
    normalized = raw_cc64_to_normalized_targets(raw)
    assert torch.equal(normalized, raw.float() / 127.0)
    assert len(torch.unique(normalized[0])) == 4
    assert normalized[0, 0] != normalized[0, 1]
    assert normalized[0, 2] != normalized[0, 3]
    assert normalized[1, 0] != normalized[1, 1]


def test_raw_huber_uses_same_delta_and_formulation_as_representative_huber() -> None:
    torch.manual_seed(31)
    representative = RepresentativeHuberEncoderModel(DummyEncoder())
    raw_model = RawCC64HuberEncoderModel(DummyEncoder())
    raw_model.load_state_dict(representative.state_dict(), strict=True)
    input_ids = torch.randint(0, 5000, (2, 24), dtype=torch.long)
    attention = torch.ones_like(input_ids)
    note_mask = torch.ones((2, 3), dtype=torch.bool)
    classes = torch.tensor(
        [[[0, 1, 2, 3], [3, 2, 1, 0], [1, 1, 2, 2]]] * 2,
        dtype=torch.long,
    )
    representatives = torch.tensor([0, 51, 79, 127], dtype=torch.long)
    raw_targets = representatives[classes]
    representative_output = representative(input_ids, attention, classes, note_mask)
    raw_output = raw_model(input_ids, attention, raw_targets, note_mask)
    assert representative.delta == raw_model.delta == HUBER_DELTA_NORMALIZED == 14 / 127
    assert torch.equal(representative_output.predictions, raw_output.predictions)
    assert torch.equal(representative_output.loss, raw_output.loss)


def test_unconstrained_inference_reuses_clip_scale_round_and_bins() -> None:
    normalized = torch.tensor([-0.25, 0.0, 0.5, 1.0, 1.25])
    cc64 = normalized_scalars_to_cc64(normalized)
    assert torch.equal(cc64, torch.tensor([0.0, 0.0, 63.5, 127.0, 127.0]))
    assert torch.equal(
        continuous_cc64_to_classes(cc64), torch.tensor([0, 0, 2, 3, 3])
    )
    boundaries = torch.tensor([25.0, 26.0, 63.0, 64.0, 103.0, 104.0])
    assert torch.equal(
        continuous_cc64_to_classes(boundaries), torch.tensor([0, 1, 1, 2, 2, 3])
    )


def test_raw_model_four_scalar_heads_finite_backward() -> None:
    model = RawCC64HuberEncoderModel(DummyEncoder())
    assert len(model.regression_heads) == 4
    assert all(head.in_features == 12 and head.out_features == 1 for head in model.regression_heads)
    input_ids = torch.randint(0, 5000, (2, 24), dtype=torch.long)
    attention = torch.ones_like(input_ids)
    note_mask = torch.ones((2, 3), dtype=torch.bool)
    raw_targets = torch.tensor(
        [
            [[0, 18, 57, 121], [25, 26, 63, 64], [72, 79, 96, 127]],
            [[5, 37, 88, 109], [12, 55, 103, 104], [1, 40, 80, 126]],
        ],
        dtype=torch.long,
    )
    output = model(input_ids, attention, raw_targets, note_mask)
    assert output.predictions.shape == (2, 3, 4)
    assert output.loss is not None and torch.isfinite(output.loss)
    output.loss.backward()
    gradients = [value.grad for value in model.parameters() if value.grad is not None]
    assert gradients and all(torch.isfinite(value).all() for value in gradients)
