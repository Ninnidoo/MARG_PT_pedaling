"""Focused synthetic tests for the binary encoder-decoder scaffold."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from transformers import T5GemmaModuleConfig

from src.stage2_binary.dataset import (
    IGNORE_INDEX,
    binarize_raw_pedals,
    make_masked_binary_sample,
)
from src.stage2_binary.training import build_binary_optimizer
from src.stage2_binary_encoder_decoder.model import (
    BOS_ID,
    PAD_ID,
    BinaryPedalEncoderDecoderModel,
    flatten_binary_pedal_targets,
    shift_right_binary_targets,
    unflatten_binary_sequence,
)
from src.stage2_encoder_only.dataset import MASK_ID, PEDAL_TOKEN_OFFSET


class TinyEncoder(nn.Module):
    def __init__(self, hidden_size: int = 32) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embedding = nn.Embedding(256, hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        batch, length = input_ids.shape
        hidden = self.embedding(input_ids).view(
            batch, length // 8, 8, -1
        ).mean(dim=2)
        return SimpleNamespace(last_hidden_state=hidden)


def make_model() -> BinaryPedalEncoderDecoderModel:
    configuration = T5GemmaModuleConfig(
        vocab_size=4,
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
    return BinaryPedalEncoderDecoderModel(
        TinyEncoder(), configuration, decoder_init_seed=42
    )


def make_batch(notes: int = 3):
    input_ids = torch.arange(notes * 8).remainder(255).view(1, -1).long()
    token_attention_mask = torch.ones_like(input_ids)
    note_mask = torch.ones((1, notes), dtype=torch.bool)
    targets = torch.tensor(
        [[[0, 1, 0, 1], [1, 1, 0, 0], [1, 0, 1, 0]]],
        dtype=torch.long,
    )[:, :notes]
    return input_ids, token_attention_mask, note_mask, targets


class BinaryTargetAndSequenceTest(unittest.TestCase):
    def test_binary_boundary_and_note_major_flatten_order(self) -> None:
        raw = torch.tensor(
            [[[0, 63, 64, 127], [127, 64, 63, 0]]], dtype=torch.long
        )
        binary = binarize_raw_pedals(raw)
        self.assertEqual(
            binary.tolist(), [[[0, 0, 1, 1], [1, 1, 0, 0]]]
        )
        flattened = flatten_binary_pedal_targets(binary)
        self.assertEqual(flattened.tolist(), [[0, 0, 1, 1, 1, 1, 0, 0]])

    def test_teacher_forcing_one_step_shift_and_padding(self) -> None:
        flat = torch.tensor(
            [[0, 1, 1, 0, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]]
        )
        inputs, mask = shift_right_binary_targets(flat)
        self.assertEqual(inputs.tolist(), [[BOS_ID, 0, 1, 1, PAD_ID, PAD_ID, PAD_ID, PAD_ID]])
        self.assertEqual(
            mask.tolist(), [[True, True, True, True, False, False, False, False]]
        )
        # Target y_t is absent from decoder input position t; only y_(t-1) enters.
        self.assertEqual(inputs[0, 1:4].tolist(), flat[0, :3].tolist())

    def test_flat_output_reconstructs_exact_pedal_slots(self) -> None:
        flat = torch.tensor([[0, 1, 0, 1, 1, 0, 1, 0]])
        restored = unflatten_binary_sequence(flat, note_count=2)
        self.assertEqual(
            restored.tolist(), [[[0, 1, 0, 1], [1, 0, 1, 0]]]
        )

    def test_existing_binary_masking_helper_hides_all_pedals(self) -> None:
        raw = np.asarray([[0, 63, 64, 127], [127, 0, 64, 1]], dtype=np.int64)
        tokens = np.asarray(
            [
                [10, 20, 30, 40, *(raw[0] + PEDAL_TOKEN_OFFSET)],
                [11, 21, 31, 41, *(raw[1] + PEDAL_TOKEN_OFFSET)],
            ],
            dtype=np.int64,
        )
        sample = make_masked_binary_sample(tokens, 0, 2)
        masked = sample["input_ids"].reshape(2, 8)
        self.assertTrue(torch.equal(masked[:, :4], torch.from_numpy(tokens[:, :4])))
        self.assertTrue(bool(torch.all(masked[:, 4:] == MASK_ID)))


class BinaryDecoderCausalityTest(unittest.TestCase):
    def test_teacher_forced_causal_mask_blocks_future_ground_truth(self) -> None:
        model = make_model().eval()
        input_ids, attention, note_mask, targets = make_batch(3)
        changed = targets.clone()
        changed.view(1, -1)[0, 5] = 1 - changed.view(1, -1)[0, 5]
        with torch.no_grad():
            first = model(
                input_ids, attention, targets, note_mask
            ).logits.view(1, -1, 2)
            second = model(
                input_ids, attention, changed, note_mask
            ).logits.view(1, -1, 2)
        # y_5 first enters shifted history at position 6.
        self.assertTrue(torch.equal(first[:, :6], second[:, :6]))

    def test_free_running_history_is_bos_then_own_predictions(self) -> None:
        model = make_model().eval()
        input_ids, attention, note_mask, targets = make_batch(3)
        with torch.no_grad():
            output = model(
                input_ids,
                attention,
                note_mask=note_mask,
                decode_mode="greedy",
            )
        generated = output.generated_ids.view(1, -1)
        history = output.decoder_input_ids
        self.assertEqual(history[0, 0].item(), BOS_ID)
        self.assertTrue(torch.equal(history[:, 1:], generated[:, :-1]))
        self.assertTrue(bool(torch.all((generated == 0) | (generated == 1))))
        with self.assertRaisesRegex(ValueError, "forbids targets"):
            model(
                input_ids,
                attention,
                binary_targets=targets,
                note_mask=note_mask,
                decode_mode="greedy",
            )

    def test_teacher_forced_loss_gradients_and_optimizer_groups(self) -> None:
        model = make_model().train()
        input_ids, attention, note_mask, targets = make_batch(3)
        optimizer = build_binary_optimizer(
            model, encoder_lr=1e-5, head_lr=1e-4, weight_decay=0.01
        )
        output = model(input_ids, attention, targets, note_mask)
        self.assertTrue(bool(torch.isfinite(output.loss)))
        output.loss.backward()
        for name, parameters in (
            ("encoder", model.encoder.parameters()),
            ("decoder", model.decoder.parameters()),
            ("slot", model.slot_embeddings.parameters()),
            ("output", model.output_head.parameters()),
        ):
            gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
            self.assertTrue(gradients, name)
            self.assertTrue(all(bool(torch.isfinite(value).all()) for value in gradients), name)
            self.assertGreater(sum(float(value.abs().sum()) for value in gradients), 0.0, name)
        self.assertEqual(
            [group["group_name"] for group in optimizer.param_groups],
            ["encoder", "head"],
        )


if __name__ == "__main__":
    unittest.main()
