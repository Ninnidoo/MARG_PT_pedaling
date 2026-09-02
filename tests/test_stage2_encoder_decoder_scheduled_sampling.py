"""Correctness tests for exact autoregressive scheduled-sampling history."""

from __future__ import annotations

import unittest

import torch

from src.stage2_encoder_decoder.model import (
    BOS_ID,
    PAD_ID,
    flatten_pedal_targets,
    shift_right_pedal_targets,
)
from src.stage2_encoder_decoder.train import teacher_forcing_probability
from tests.test_stage2_encoder_decoder_five_class import make_batch, make_model


def _generator(seed: int) -> torch.Generator:
    result = torch.Generator(device="cpu")
    result.manual_seed(seed)
    return result


class ScheduledSamplingScheduleTest(unittest.TestCase):
    def test_epoch_probabilities_are_exact(self) -> None:
        actual = [teacher_forcing_probability(epoch) for epoch in range(1, 10)]
        self.assertEqual(actual, [1.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.6, 0.6, 0.6])
        with self.assertRaises(ValueError):
            teacher_forcing_probability(0)


class ScheduledSamplingHistoryTest(unittest.TestCase):
    def test_probability_one_matches_parallel_teacher_forcing_objective(self) -> None:
        model = make_model().eval()
        input_ids, attention, note_mask, targets = make_batch(3)
        with torch.no_grad():
            baseline = model(input_ids, attention, targets, note_mask)
            history = model.scheduled_sampling_history(
                input_ids, attention, targets, note_mask, 1.0, generator=_generator(7)
            )
            scheduled = model.loss_from_scheduled_history(
                input_ids, attention, targets, note_mask, history
            )
        shifted, valid = shift_right_pedal_targets(flatten_pedal_targets(targets))
        self.assertTrue(torch.equal(history.decoder_input_ids, shifted))
        self.assertTrue(torch.equal(history.decoder_attention_mask, valid))
        self.assertTrue(torch.equal(baseline.logits, scheduled.logits))
        self.assertTrue(torch.equal(baseline.loss, scheduled.loss))

    def test_prediction_is_detached_and_enters_next_history_position(self) -> None:
        model = make_model().eval()
        input_ids, attention, note_mask, targets = make_batch(3)
        history = model.scheduled_sampling_history(
            input_ids, attention, targets, note_mask, 0.0, generator=_generator(11)
        )
        self.assertFalse(history.decoder_input_ids.requires_grad)
        self.assertFalse(history.predictions.requires_grad)
        self.assertEqual(history.decoder_input_ids[0, 0].item(), BOS_ID)
        self.assertTrue(
            torch.equal(
                history.decoder_input_ids[:, 1:],
                history.predictions[:, :-1],
            )
        )
        self.assertEqual(history.ground_truth_count, 0)
        self.assertEqual(history.predicted_count, targets.numel() - 1)

    def test_no_future_ground_truth_leakage_when_predictions_drive_history(self) -> None:
        model = make_model().eval()
        input_ids, attention, note_mask, targets = make_batch(3)
        changed = targets.clone()
        changed.view(1, -1)[0, 5:] = torch.tensor([4, 4, 4, 4, 4, 4, 4])
        first = model.scheduled_sampling_history(
            input_ids, attention, targets, note_mask, 0.0, generator=_generator(19)
        )
        second = model.scheduled_sampling_history(
            input_ids, attention, changed, note_mask, 0.0, generator=_generator(19)
        )
        self.assertTrue(torch.equal(first.predictions, second.predictions))
        self.assertTrue(torch.equal(first.decoder_input_ids, second.decoder_input_ids))

    def test_synthetic_decision_mask_constructs_exact_mixed_history(self) -> None:
        model = make_model().eval()
        input_ids, attention, note_mask, targets = make_batch(4)
        flat = flatten_pedal_targets(targets)
        history = model.scheduled_sampling_history(
            input_ids, attention, targets, note_mask, 0.5, generator=_generator(1234)
        )
        expected = torch.full_like(flat, PAD_ID)
        expected[:, 0] = BOS_ID
        for position in range(1, flat.shape[1]):
            expected[:, position] = torch.where(
                history.teacher_forcing_mask[:, position],
                flat[:, position - 1],
                history.predictions[:, position - 1],
            )
        self.assertTrue(torch.equal(history.decoder_input_ids, expected))
        self.assertGreater(history.ground_truth_count, 0)
        self.assertGreater(history.predicted_count, 0)

    def test_cached_rollout_predictions_match_causal_parallel_loss_pass(self) -> None:
        """The detached rollout is not an approximation of the token objective.

        The reference PT decoder has zero dropout.  Given the completed mixed
        history, native cached step decoding and the causal parallel decoder
        therefore produce the same logits/predictions at every target step.
        """

        model = make_model().eval()
        self.assertEqual(float(model.decoder.config.attention_dropout), 0.0)
        self.assertEqual(float(model.decoder.config.dropout_rate), 0.0)
        input_ids, attention, note_mask, targets = make_batch(4)
        history = model.scheduled_sampling_history(
            input_ids, attention, targets, note_mask, 0.5, generator=_generator(314)
        )
        with torch.no_grad():
            output = model.loss_from_scheduled_history(
                input_ids, attention, targets, note_mask, history
            )
        parallel_predictions = output.logits.view(1, -1, 5).argmax(dim=-1)
        self.assertTrue(torch.equal(parallel_predictions, history.predictions))

    def test_rng_is_seed_reproducible(self) -> None:
        model = make_model().eval()
        input_ids, attention, note_mask, targets = make_batch(4)
        first = model.scheduled_sampling_history(
            input_ids, attention, targets, note_mask, 0.6, generator=_generator(42)
        )
        second = model.scheduled_sampling_history(
            input_ids, attention, targets, note_mask, 0.6, generator=_generator(42)
        )
        self.assertTrue(torch.equal(first.teacher_forcing_mask, second.teacher_forcing_mask))
        self.assertTrue(torch.equal(first.decoder_input_ids, second.decoder_input_ids))

    def test_bos_pad_loss_mask_and_note_slot_order(self) -> None:
        model = make_model().eval()
        input_ids, attention, note_mask, targets = make_batch(2)
        targets[:, 1] = -100
        note_mask[:, 1] = False
        history = model.scheduled_sampling_history(
            input_ids, attention, targets, note_mask, 0.5, generator=_generator(6)
        )
        self.assertEqual(history.decoder_input_ids[0, 0].item(), BOS_ID)
        self.assertEqual(history.decoder_input_ids[0, 4:].tolist(), [PAD_ID] * 4)
        self.assertFalse(bool(history.decoder_attention_mask[0, 4:].any()))
        flat = flatten_pedal_targets(targets)
        self.assertEqual(flat[0, :4].tolist(), targets[0, 0].tolist())
        with torch.no_grad():
            output = model.loss_from_scheduled_history(
                input_ids, attention, targets, note_mask, history
            )
        self.assertTrue(bool(torch.isfinite(output.loss)))
        self.assertEqual(tuple(output.logits.shape), (1, 2, 4, 5))


class ScheduledSamplingGradientTest(unittest.TestCase):
    def test_finite_loss_gradients_and_optimizer_update(self) -> None:
        model = make_model().train()
        input_ids, attention, note_mask, targets = make_batch(4)
        history = model.scheduled_sampling_history(
            input_ids, attention, targets, note_mask, 0.5, generator=_generator(99)
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        before = model.output_head.weight.detach().clone()
        output = model.loss_from_scheduled_history(
            input_ids, attention, targets, note_mask, history
        )
        self.assertTrue(bool(torch.isfinite(output.loss)))
        output.loss.backward()
        groups = {
            "encoder": model.encoder.parameters(),
            "decoder": model.decoder.parameters(),
            "slot": model.slot_embeddings.parameters(),
            "output": model.output_head.parameters(),
        }
        for name, parameters in groups.items():
            gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
            self.assertTrue(gradients, name)
            self.assertTrue(all(bool(torch.isfinite(value).all()) for value in gradients), name)
            self.assertGreater(sum(float(value.abs().sum()) for value in gradients), 0.0, name)
        optimizer.step()
        self.assertFalse(torch.equal(before, model.output_head.weight.detach()))


if __name__ == "__main__":
    unittest.main()
