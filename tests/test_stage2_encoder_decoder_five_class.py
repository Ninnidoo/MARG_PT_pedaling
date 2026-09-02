"""Synthetic tests for the weighted Stage 2 encoder-decoder model."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from torch import nn
from transformers import T5GemmaModuleConfig

from src.stage2_encoder_decoder.model import (
    BOS_ID,
    PAD_ID,
    EncoderDecoderFiveClassModel,
    flatten_pedal_targets,
    shift_right_pedal_targets,
)
from src.stage2_encoder_only.five_class import IGNORE_INDEX


class TinyEncoder(nn.Module):
    def __init__(self, hidden_size: int = 32) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embedding = nn.Embedding(64, hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        batch, length = input_ids.shape
        hidden = self.embedding(input_ids).view(batch, length // 8, 8, -1).mean(dim=2)
        return SimpleNamespace(last_hidden_state=hidden)


def make_model() -> EncoderDecoderFiveClassModel:
    config = T5GemmaModuleConfig(
        vocab_size=7,
        hidden_size=32,
        cross_attention_hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        pad_token_id=PAD_ID,
        bos_token_id=BOS_ID,
        eos_token_id=PAD_ID,
        dropout_rate=0.0,
        attention_dropout=0.0,
    )
    return EncoderDecoderFiveClassModel(TinyEncoder(), config, decoder_init_seed=42)


def make_batch(notes: int = 3):
    input_ids = torch.arange(notes * 8).remainder(63).view(1, -1).long()
    attention = torch.ones_like(input_ids)
    note_mask = torch.ones((1, notes), dtype=torch.bool)
    targets = torch.arange(notes * 4).remainder(5).view(1, notes, 4).long()
    return input_ids, attention, note_mask, targets


class PedalSequenceTest(unittest.TestCase):
    def test_flatten_order_and_shift_right(self) -> None:
        targets = torch.tensor([[[0, 1, 2, 3], [4, 3, 2, 1]]])
        flattened = flatten_pedal_targets(targets)
        self.assertEqual(flattened.tolist(), [[0, 1, 2, 3, 4, 3, 2, 1]])
        shifted, mask = shift_right_pedal_targets(flattened)
        self.assertEqual(shifted.tolist(), [[BOS_ID, 0, 1, 2, 3, 4, 3, 2]])
        self.assertTrue(bool(mask.all()))

    def test_bos_pad_and_padding_mask(self) -> None:
        flat = torch.tensor([[0, 1, 2, 3, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]])
        shifted, mask = shift_right_pedal_targets(flat)
        self.assertEqual(shifted[0, 0].item(), BOS_ID)
        self.assertEqual(shifted[0, 4:].tolist(), [PAD_ID] * 4)
        self.assertEqual(mask.tolist(), [[True, True, True, True, False, False, False, False]])
        model = make_model()
        logits = torch.randn(1, 8, 5)
        loss, numerator, denominator = model._loss_components(logits, flat)
        reference = torch.nn.functional.cross_entropy(logits[:, :4].reshape(-1, 5), flat[:, :4].reshape(-1))
        self.assertTrue(torch.allclose(loss, reference, atol=1e-6))
        self.assertTrue(bool(torch.isfinite(numerator)))
        self.assertGreater(denominator.item(), 0.0)

    def test_slot_embedding_alignment(self) -> None:
        model = make_model()
        ids = torch.zeros((1, 8), dtype=torch.long)
        combined = model._decoder_embeddings(ids)
        token = model.decoder.embed_tokens(ids)
        actual = combined - token
        for position in range(8):
            self.assertTrue(torch.allclose(actual[0, position], model.slot_embeddings.weight[position % 4], atol=1e-7))


class DecoderBehaviorTest(unittest.TestCase):
    def test_causal_mask_blocks_future_token_leakage(self) -> None:
        model = make_model().eval()
        input_ids, attention, note_mask, targets = make_batch(3)
        changed = targets.clone()
        changed.view(1, -1)[0, 5] = (changed.view(1, -1)[0, 5] + 1) % 5
        with torch.no_grad():
            first = model(input_ids, attention, targets, note_mask).logits.view(1, -1, 5)
            second = model(input_ids, attention, changed, note_mask).logits.view(1, -1, 5)
        # y_5 enters the shifted input only at position 6; positions <=5 must match.
        self.assertTrue(torch.equal(first[:, :6], second[:, :6]))

    def test_encoder_decoder_output_gradients_are_finite_nonzero(self) -> None:
        model = make_model().train()
        input_ids, attention, note_mask, targets = make_batch(3)
        output = model(input_ids, attention, targets, note_mask)
        self.assertTrue(bool(torch.isfinite(output.loss)))
        output.loss.backward()
        groups = {
            "encoder": model.encoder.parameters(),
            "decoder": model.decoder.parameters(),
            "output": model.output_head.parameters(),
        }
        for name, parameters in groups.items():
            gradients = [p.grad for p in parameters if p.grad is not None]
            self.assertTrue(gradients, name)
            self.assertTrue(all(bool(torch.isfinite(g).all()) for g in gradients), name)
            self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0.0, name)

    def test_greedy_generation_shape_and_note_slot_order(self) -> None:
        model = make_model().eval()
        input_ids, attention, note_mask, _ = make_batch(2)
        output = model(input_ids, attention, note_mask=note_mask, decode_mode="greedy")
        self.assertEqual(tuple(output.logits.shape), (1, 2, 4, 5))
        self.assertTrue(bool(torch.isfinite(output.logits).all()))
        self.assertEqual(output.logits.view(1, 8, 5).shape[1], 2 * 4)


if __name__ == "__main__":
    unittest.main()
