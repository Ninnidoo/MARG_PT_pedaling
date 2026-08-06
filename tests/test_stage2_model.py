from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from src.stage2_encoder_only.model import (
    PEDAL_NUM_CLASSES,
    PEDAL_SLOTS,
    Stage2PedalEncoderModel,
)


class StubEncoder(nn.Module):
    def __init__(self, hidden_size: int = 16, dropout_rate: float = 0.2) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
        )
        self.embedding = nn.Embedding(256, hidden_size)
        self.projection = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> SimpleNamespace:
        del attention_mask
        batch_size, flat_length = input_ids.shape
        embedded = self.embedding(input_ids).view(
            batch_size,
            flat_length // 8,
            8,
            self.config.hidden_size,
        )
        hidden = self.projection(embedded.mean(dim=2))
        return SimpleNamespace(last_hidden_state=hidden)


class Stage2ModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20260710)

    @staticmethod
    def inputs(
        batch_size: int = 2,
        notes: int = 3,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ids = torch.randint(0, 256, (batch_size, notes * 8))
        attention_mask = torch.ones_like(input_ids)
        targets = torch.randint(0, PEDAL_NUM_CLASSES, (batch_size, notes, 4))
        note_mask = torch.ones((batch_size, notes), dtype=torch.bool)
        return input_ids, attention_mask, targets, note_mask

    @staticmethod
    def make_model(
        freeze_encoder: bool = False,
        dropout: float | None = None,
    ) -> Stage2PedalEncoderModel:
        return Stage2PedalEncoderModel(
            StubEncoder(),
            freeze_encoder=freeze_encoder,
            dropout=dropout,
        )

    def test_output_shape_and_four_independent_heads(self) -> None:
        model = self.make_model(dropout=0.0)
        input_ids, attention_mask, _, _ = self.inputs()
        output = model(input_ids, attention_mask)

        self.assertEqual(tuple(output.logits.shape), (2, 3, 4, 128))
        self.assertEqual(tuple(output.hidden_states.shape), (2, 3, 16))
        self.assertEqual(len(model.classification_heads), PEDAL_SLOTS)
        self.assertEqual(
            len({id(head) for head in model.classification_heads}),
            PEDAL_SLOTS,
        )
        for head in model.classification_heads:
            self.assertEqual(head.in_features, 16)
            self.assertEqual(head.out_features, PEDAL_NUM_CLASSES)

    def test_loss_matches_flattened_cross_entropy(self) -> None:
        model = self.make_model(dropout=0.0)
        input_ids, attention_mask, targets, note_mask = self.inputs()
        targets[:, -1] = -100
        note_mask[:, -1] = False
        output = model(input_ids, attention_mask, targets, note_mask)
        expected = F.cross_entropy(
            output.logits.reshape(-1, 128),
            targets.reshape(-1),
            ignore_index=-100,
        )
        self.assertTrue(torch.allclose(output.loss, expected))

    def test_padded_logits_do_not_change_loss(self) -> None:
        model = self.make_model()
        logits = torch.randn(1, 2, 4, 128)
        targets = torch.randint(0, 128, (1, 2, 4))
        targets[:, 1] = -100
        changed = logits.clone()
        changed[:, 1] = torch.randn_like(changed[:, 1]) * 10000
        self.assertTrue(
            torch.allclose(
                model.compute_loss(logits, targets),
                model.compute_loss(changed, targets),
            )
        )

    def test_shape_validation(self) -> None:
        model = self.make_model()
        input_ids, attention_mask, targets, note_mask = self.inputs()
        invalid_calls = [
            lambda: model(input_ids.unsqueeze(0), attention_mask),
            lambda: model(input_ids[:, :-1], attention_mask[:, :-1]),
            lambda: model(input_ids, attention_mask[:, :-1]),
            lambda: model(input_ids, attention_mask, targets[:, :, :3]),
            lambda: model(input_ids, attention_mask, targets, note_mask[:, :-1]),
        ]
        for invalid_call in invalid_calls:
            with self.subTest(call=invalid_call):
                with self.assertRaises(ValueError):
                    invalid_call()

    def test_note_mask_and_target_padding_must_agree(self) -> None:
        model = self.make_model()
        input_ids, attention_mask, targets, note_mask = self.inputs()
        note_mask[:, -1] = False
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            model(input_ids, attention_mask, targets, note_mask)

    def test_frozen_encoder_has_no_gradients_but_heads_do(self) -> None:
        model = self.make_model(freeze_encoder=True, dropout=0.0)
        input_ids, attention_mask, targets, note_mask = self.inputs()
        output = model(input_ids, attention_mask, targets, note_mask)
        output.loss.backward()

        self.assertTrue(
            all(not parameter.requires_grad for parameter in model.encoder.parameters())
        )
        self.assertTrue(
            all(parameter.grad is None for parameter in model.encoder.parameters())
        )
        self.assertTrue(
            all(
                parameter.grad is not None
                for parameter in model.classification_heads.parameters()
            )
        )

    def test_fine_tuned_encoder_and_heads_receive_gradients(self) -> None:
        model = self.make_model(freeze_encoder=False, dropout=0.0)
        input_ids, attention_mask, targets, note_mask = self.inputs()
        output = model(input_ids, attention_mask, targets, note_mask)
        output.loss.backward()

        self.assertTrue(
            all(parameter.requires_grad for parameter in model.encoder.parameters())
        )
        self.assertTrue(
            any(parameter.grad is not None for parameter in model.encoder.parameters())
        )
        self.assertTrue(
            all(
                parameter.grad is not None
                for parameter in model.classification_heads.parameters()
            )
        )

    def test_eval_mode_is_deterministic(self) -> None:
        model = self.make_model(dropout=0.5).eval()
        input_ids, attention_mask, _, _ = self.inputs()
        with torch.no_grad():
            first = model(input_ids, attention_mask).logits
            second = model(input_ids, attention_mask).logits
        self.assertTrue(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
