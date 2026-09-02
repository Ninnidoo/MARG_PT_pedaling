"""Targeted synthetic tests for canonical Decoder-only Stage 2 v0."""

from __future__ import annotations

import unittest

import torch
from transformers import T5GemmaModuleConfig

from third_party.PianistTransformer.src.model.pianoformer import PianoEncoderEmbeddings
from src.stage2_decoder_only.model import (
    FourClassPedalDecoderOnlyModel,
    build_prefix_lm_attention_mask,
)
from src.stage2_four_class.model import (
    BOS_ID,
    PAD_ID,
    shift_right_four_class_targets,
)
from src.stage2_four_class.representation import IGNORE_INDEX


HIDDEN_SIZE = 32
VOCAB_SIZE = 128


def make_model() -> FourClassPedalDecoderOnlyModel:
    config = T5GemmaModuleConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=3,
        dropout_rate=0.0,
        attention_dropout=0.0,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=4096,
    )
    embeddings = PianoEncoderEmbeddings(config)
    return FourClassPedalDecoderOnlyModel(
        embeddings, config, num_layers=2, init_seed=42
    )


def make_batch(notes: int = 3, *, batch_size: int = 2):
    values = torch.arange(batch_size * notes * 8).remainder(VOCAB_SIZE - 2) + 2
    input_ids = values.view(batch_size, notes, 8).long()
    note_mask = torch.ones((batch_size, notes), dtype=torch.bool)
    if batch_size > 1:
        note_mask[1, -1] = False
        input_ids[1, -1] = 0
    token_attention_mask = note_mask.repeat_interleave(8, dim=1).long()
    targets = torch.arange(batch_size * notes * 4).remainder(4)
    targets = targets.view(batch_size, notes, 4).long()
    targets = targets.masked_fill(~note_mask.unsqueeze(-1), IGNORE_INDEX)
    return input_ids.view(batch_size, -1), token_attention_mask, note_mask, targets


def test_a_prefix_lm_mask_element_semantics_and_padding() -> None:
    note_mask = torch.tensor([[True, True, False]])
    ar_mask = torch.tensor([[True, True, True, False]])
    mask = build_prefix_lm_attention_mask(note_mask, ar_mask)
    assert mask.shape == (1, 1, 7, 7)
    allowed = mask[0, 0] == 0

    # 1. Prefix query -> every valid prefix key.
    assert allowed[0, :3].tolist() == [True, True, False]
    assert allowed[1, :3].tolist() == [True, True, False]
    # 2. Prefix query -> every autoregressive key blocked.
    assert not bool(allowed[:3, 3:].any())
    # 3. Pedal query -> every valid prefix key.
    assert allowed[4, :3].tolist() == [True, True, False]
    # 4/5. Pedal query -> past/current allowed, future blocked.
    assert allowed[4, 3:].tolist() == [True, True, False, False]
    assert allowed[5, 3:].tolist() == [True, True, True, False]
    # 6. Padded prefix and padded pedal positions are never keys.
    assert not bool(allowed[:, 2].any())
    assert not bool(allowed[:, 6].any())


def test_a2_actual_forward_has_no_future_pedal_leakage() -> None:
    model = make_model().eval()
    input_ids, attention, note_mask, targets = make_batch(3, batch_size=1)
    changed = targets.clone()
    changed.view(1, -1)[0, 5] = (changed.view(1, -1)[0, 5] + 1) % 4
    with torch.no_grad():
        first = model(input_ids, attention, targets, note_mask).logits
        second = model(input_ids, attention, changed, note_mask).logits
    # y_5 first enters shifted history at input position 6.
    assert torch.equal(first[:, :6], second[:, :6])


def test_b_performance_prefix_never_uses_pedal_values_or_targets() -> None:
    model = make_model().eval()
    input_ids, attention, note_mask, targets = make_batch(3, batch_size=1)
    changed_inputs = input_ids.view(1, 3, 8).clone()
    changed_inputs[:, :, 4:] = torch.tensor([17, 33, 79, 127])
    changed_targets = (targets + 1).remainder(4)
    with torch.no_grad():
        first = model.build_performance_prefix(input_ids, attention, note_mask)
        second = model.build_performance_prefix(
            changed_inputs.view(1, -1), attention, note_mask
        )
    assert torch.equal(first, second)
    assert not torch.equal(targets, changed_targets)


def test_c_teacher_forced_shift_right_is_exactly_one_position() -> None:
    flat_targets = torch.tensor(
        [[0, 1, 2, 3, 3, 2, 1, 0], [3, 1, 2, 0, -100, -100, -100, -100]]
    )
    decoder_inputs, valid = shift_right_four_class_targets(flat_targets)
    assert decoder_inputs[0].tolist() == [BOS_ID, 0, 1, 2, 3, 3, 2, 1]
    assert decoder_inputs[1].tolist() == [BOS_ID, 3, 1, 2, PAD_ID, PAD_ID, PAD_ID, PAD_ID]
    assert torch.equal(decoder_inputs[:, 1:4], flat_targets[:, :3])
    assert valid[1].tolist() == [True, True, True, True, False, False, False, False]


def test_d_prefix_and_pedal_logit_shapes() -> None:
    model = make_model().eval()
    input_ids, attention, note_mask, targets = make_batch(3)
    with torch.no_grad():
        prefix = model.build_performance_prefix(input_ids, attention, note_mask)
        output = model(input_ids, attention, targets, note_mask)
    assert prefix.shape == (2, 3, HIDDEN_SIZE)
    assert output.logits.shape == (2, 12, 4)
    assert output.attention_mask is not None
    assert output.attention_mask.shape == (2, 1, 15, 15)
    assert not any("cross_attn" in name for name, _ in model.named_modules())
    assert not hasattr(model, "encoder")


def test_e_teacher_forced_forward_backward_has_finite_loss_and_gradients() -> None:
    model = make_model().train()
    input_ids, attention, note_mask, targets = make_batch(2)
    output = model(input_ids, attention, targets, note_mask)
    assert output.loss is not None and bool(torch.isfinite(output.loss))
    output.loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0.0


def test_f_target_free_greedy_generation_shape_and_vocabulary() -> None:
    model = make_model().eval()
    input_ids, attention, note_mask, _ = make_batch(2)
    generated = model.generate(input_ids, attention, note_mask)
    generated_with_logits, logits = model.generate_with_logits(
        input_ids, attention, note_mask
    )
    assert generated.shape == (2, 2, 4)
    assert torch.equal(generated, generated_with_logits)
    assert logits.shape == (2, 2, 4, 4)
    valid = note_mask.unsqueeze(-1).expand_as(generated_with_logits)
    assert torch.equal(
        generated_with_logits.masked_select(valid),
        logits.argmax(dim=-1).masked_select(valid),
    )
    assert bool(torch.isfinite(logits).all())
    assert bool(((generated >= 0) & (generated <= 3)).all())


class DecoderOnlyTargetedTests(unittest.TestCase):
    def test_a_mask(self) -> None:
        test_a_prefix_lm_mask_element_semantics_and_padding()

    def test_b_no_pedal_leakage(self) -> None:
        test_b_performance_prefix_never_uses_pedal_values_or_targets()

    def test_a2_no_future_pedal_leakage(self) -> None:
        test_a2_actual_forward_has_no_future_pedal_leakage()

    def test_c_shift_right(self) -> None:
        test_c_teacher_forced_shift_right_is_exactly_one_position()

    def test_d_shapes(self) -> None:
        test_d_prefix_and_pedal_logit_shapes()

    def test_e_forward_backward(self) -> None:
        test_e_teacher_forced_forward_backward_has_finite_loss_and_gradients()

    def test_f_free_running(self) -> None:
        test_f_target_free_greedy_generation_shape_and_vocabulary()


if __name__ == "__main__":
    unittest.main()
