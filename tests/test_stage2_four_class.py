from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from src.stage2_four_class.model import FourClassPedalEncoderModel
from src.stage2_four_class.representation import (
    CLASS_BOUNDS,
    NUM_CLASSES,
    REPRESENTATIVES,
    classify_cc64,
)


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


def test_canonical_boundaries_and_representatives() -> None:
    values = np.asarray([[25, 26, 63, 64], [103, 104, 127, 0]], dtype=np.int64)
    expected = np.asarray([[0, 1, 1, 2], [2, 3, 3, 0]], dtype=np.int64)
    assert np.array_equal(classify_cc64(values), expected)
    assert CLASS_BOUNDS == ((0, 25), (26, 63), (64, 103), (104, 127))
    assert REPRESENTATIVES == (0, 51, 79, 127)


def test_encoder_only_four_heads_unweighted_ce_backward() -> None:
    model = FourClassPedalEncoderModel(DummyEncoder())
    input_ids = torch.randint(0, 5000, (2, 24), dtype=torch.long)
    attention = torch.ones_like(input_ids, dtype=torch.bool)
    note_mask = torch.ones((2, 3), dtype=torch.bool)
    targets = torch.randint(0, NUM_CLASSES, (2, 3, 4), dtype=torch.long)
    output = model(input_ids, attention, targets, note_mask)
    assert output.logits.shape == (2, 3, 4, 4)
    assert output.loss is not None and torch.isfinite(output.loss)
    output.loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert not hasattr(model, "class_weights")
