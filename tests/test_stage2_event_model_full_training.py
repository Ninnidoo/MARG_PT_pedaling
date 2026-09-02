"""Targeted exact-objective and canonical-config tests for full training."""

from __future__ import annotations

import unittest

import torch

from scripts.run_custom_event_model_v0_full_seed42 import EXPECTED_WEIGHT_SHA, configuration
from src.stage2_event_model.full_training import group_denominators, normalized_components, raw_loss_terms
from src.stage2_event_model.losses import CustomEventCriterion, CustomEventLossConfig
from src.stage2_event_model.model import CustomEventModelOutput


def batch(seed: int) -> tuple[dict[str, torch.Tensor], CustomEventModelOutput]:
    generator = torch.Generator().manual_seed(seed)
    main_targets = torch.tensor([[[0, 1, 2, 0, 3, 4], [1, 0, 3, 4, 0, 2]]])
    terminal_targets = torch.tensor([[1, 0, 3, 4]])
    payload = {
        "initial_valid_mask": torch.tensor([True]),
        "initial_targets": torch.tensor([seed % 4]),
        "owned_onset_mask": torch.tensor([[True, True]]),
        "main_event_targets": main_targets,
        "main_timing_valid_mask": main_targets != 0,
        "main_timing_targets": torch.rand((1, 2, 6), generator=generator),
        "terminal_valid_mask": torch.tensor([True]),
        "terminal_event_targets": terminal_targets,
        "terminal_timing_valid_mask": terminal_targets != 0,
        "terminal_timing_targets": torch.rand((1, 4), generator=generator),
    }
    output = CustomEventModelOutput(
        main_event_logits=torch.randn((1, 2, 6, 5), generator=generator),
        main_timing_predictions=torch.randn((1, 2, 6), generator=generator),
        initial_logits=torch.randn((1, 4), generator=generator),
        terminal_event_logits=torch.randn((1, 4, 5), generator=generator),
        terminal_timing_predictions=torch.randn((1, 4), generator=generator),
        encoder_hidden_states=torch.empty(0),
        owned_onset_hidden_states=torch.empty(0),
    )
    return payload, output


def concatenate_batches(first, second):
    return {key: torch.cat((first[key], second[key]), dim=0) for key in first}


def concatenate_outputs(first, second):
    return CustomEventModelOutput(
        main_event_logits=torch.cat((first.main_event_logits, second.main_event_logits)),
        main_timing_predictions=torch.cat((first.main_timing_predictions, second.main_timing_predictions)),
        initial_logits=torch.cat((first.initial_logits, second.initial_logits)),
        terminal_event_logits=torch.cat((first.terminal_event_logits, second.terminal_event_logits)),
        terminal_timing_predictions=torch.cat((first.terminal_timing_predictions, second.terminal_timing_predictions)),
        encoder_hidden_states=torch.empty(0),
        owned_onset_hidden_states=torch.empty(0),
    )


class FullTrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        main = torch.tensor([[0.8, 1.2, 1.1, 0.9, 1.0]] * 6)
        terminal = torch.tensor([[0.7, 1.3, 1.1, 0.8, 1.1]] * 4)
        self.criterion = CustomEventCriterion(main, terminal, CustomEventLossConfig())

    def test_effective_batch_accumulation_matches_full_batch_objective(self) -> None:
        first_batch, first_output = batch(7)
        second_batch, second_output = batch(9)
        denominator = group_denominators(
            [first_batch, second_batch],
            self.criterion.main_event_weights,
            self.criterion.terminal_event_weights,
        )
        first = normalized_components(raw_loss_terms(first_output, first_batch, self.criterion), denominator, self.criterion.config)
        second = normalized_components(raw_loss_terms(second_output, second_batch, self.criterion), denominator, self.criterion.config)
        canonical = self.criterion(
            concatenate_outputs(first_output, second_output),
            concatenate_batches(first_batch, second_batch),
        )
        self.assertTrue(torch.allclose(first["total"] + second["total"], canonical.total_loss, atol=1e-6))

    def test_frozen_configuration(self) -> None:
        config = configuration(4)
        self.assertEqual(config["cache_id"], "3a5520155b5db1e9a1da7f8148556aa3e1da852655c9adde25d3dbbd1d966263")
        self.assertEqual(config["event_weight_sha256"], EXPECTED_WEIGHT_SHA)
        self.assertEqual(config["seed"], 42)
        self.assertEqual(config["micro_batch_size"], 4)
        self.assertEqual(config["gradient_accumulation_steps"], 4)
        self.assertEqual(config["effective_batch_size"], 16)
        self.assertEqual(config["max_epochs"], 10)
        self.assertEqual(config["early_stopping_patience"], 3)
        self.assertIsNone(config["scheduler"])
        self.assertEqual(set(config["loss"].values()), {0.1, 1.0})
        self.assertEqual(config["asap_test_access_count"], 0)


if __name__ == "__main__":
    unittest.main()
