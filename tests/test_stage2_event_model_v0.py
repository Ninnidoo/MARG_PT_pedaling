from __future__ import annotations

import copy
import types
import unittest
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn

from src.stage2_event_model.dataset import (
    CANONICAL_CACHE_ID,
    CustomEventWindowDataset,
    custom_event_collate_fn,
)
from src.stage2_event_model.losses import (
    CustomEventCriterion,
    CustomEventLossConfig,
    load_slot_event_weights,
)
from src.stage2_event_model.model import CustomEventEncoderModelV0
from src.stage2_event_model.ownership import assign_unique_owners


ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = ROOT / "analysis/custom_event_tokenizer_v1"
WEIGHTS = CACHE_ROOT / "candidate_event_weights.json"


class FakeEncoder(nn.Module):
    def __init__(self, hidden_size: int = 16):
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=hidden_size, dropout_rate=0.0)
        self.embedding = nn.Embedding(6000, hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        batch, flat = input_ids.shape
        notes = flat // 8
        hidden = self.embedding(input_ids.view(batch, notes, 8)[:, :, 0])
        return types.SimpleNamespace(last_hidden_state=hidden)


class OwnershipTests(unittest.TestCase):
    def test_exact_one_owner_512_256(self):
        first = torch.arange(0, 1200, 3).numpy()
        result = assign_unique_owners(1200, first, first, first)
        self.assertEqual(sum(map(len, result.owned_onset_indices)), len(first))
        self.assertTrue((result.owner_window_index >= 0).all())

    def test_split_chord_window_is_rejected(self):
        result = assign_unique_owners(800, [510], [513], [513])
        self.assertNotEqual(int(result.owner_window_index[0]), 0)
        self.assertEqual(result.split_chord_group_count, 1)

    def test_maximum_margin_owner(self):
        result = assign_unique_owners(1024, [500], [500], [500])
        owner = int(result.owner_window_index[0])
        self.assertEqual(result.window_starts[owner], 256)

    def test_tie_uses_smaller_start(self):
        result = assign_unique_owners(
            10, [3], [3], [3], window_notes=5, stride_notes=2
        )
        self.assertEqual(result.window_starts[int(result.owner_window_index[0])], 0)

    def test_short_piece_initial_terminal_same_window(self):
        result = assign_unique_owners(4, [0, 2], [1, 3], [1, 3])
        self.assertEqual(result.window_starts, (0,))
        self.assertEqual(result.owner_window_index.tolist(), [0, 0])


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = CustomEventWindowDataset(
            CACHE_ROOT, "train", entry_limit=1, cache_input_tokens=True
        )

    def test_cache_id_and_no_target_duplication(self):
        self.assertEqual(self.dataset.cache_config["cache_id"], CANONICAL_CACHE_ID)
        perf = self.dataset.performances[0]
        self.assertEqual(
            sum(map(len, perf.ownership.owned_onset_indices)), perf.num_onsets
        )
        self.assertEqual(
            len(set(perf.ownership.owner_window_index.tolist())), 
            len([x for x in perf.ownership.owned_onset_indices if len(x)]),
        )

    def test_initial_terminal_valid_once_and_global_targets(self):
        perf = self.dataset.performances[0]
        first_owner = int(perf.ownership.owner_window_index[0])
        last_owner = int(perf.ownership.owner_window_index[-1])
        first_dataset_index = self.dataset.windows.index((0, first_owner))
        last_dataset_index = self.dataset.windows.index((0, last_owner))
        samples = [self.dataset[first_dataset_index], self.dataset[last_dataset_index]]
        self.assertTrue(bool(samples[0]["initial_valid_mask"]))
        self.assertTrue(bool(samples[1]["terminal_valid_mask"]))
        for sample in samples:
            self.assertEqual(sample["main_event_targets"].shape[1:], (6,))
            self.assertEqual(sample["main_timing_targets"].shape[1:], (6,))
            self.assertTrue(
                torch.equal(
                    sample["main_timing_valid_mask"],
                    sample["main_event_targets"] != 0,
                )
            )
            import numpy as np
            with np.load(perf.cache_path, allow_pickle=False) as cache:
                expected = torch.from_numpy(
                    cache["main_event_target"][
                        sample["owned_global_onset_indices"].numpy()
                    ].astype("int64", copy=True)
                )
            self.assertTrue(torch.equal(sample["main_event_targets"], expected))
        batch = custom_event_collate_fn(samples)
        self.assertEqual(int(batch["initial_valid_mask"].sum()), 1)
        self.assertEqual(int(batch["terminal_valid_mask"].sum()), 1)
        self.assertEqual(batch["main_event_targets"].shape[-1], 6)
        self.assertTrue(
            torch.equal(
                batch["owned_onset_mask"],
                batch["owned_global_onset_indices"] >= 0,
            )
        )


def synthetic_batch() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.arange(2 * 4 * 8).reshape(2, 4 * 8) % 100,
        "token_attention_mask": torch.ones(2, 4 * 8, dtype=torch.long),
        "note_mask": torch.ones(2, 4, dtype=torch.bool),
        "owned_representative_positions": torch.tensor([[1, 3], [0, -1]]),
        "owned_onset_mask": torch.tensor([[True, True], [True, False]]),
        "main_event_targets": torch.tensor(
            [
                [[0, 1, 0, 2, 0, 3], [0, 0, 1, 0, 2, 0]],
                [[4, 0, 0, 0, 0, 0], [-100, -100, -100, -100, -100, -100]],
            ]
        ),
        "main_timing_targets": torch.full((2, 2, 6), 0.25),
        "main_timing_valid_mask": torch.tensor(
            [
                [[False, True, False, True, False, True], [False, False, True, False, True, False]],
                [[True, False, False, False, False, False], [False] * 6],
            ]
        ),
        "initial_targets": torch.tensor([1, 3]),
        "initial_valid_mask": torch.tensor([True, False]),
        "initial_representative_positions": torch.tensor([1, -1]),
        "terminal_event_targets": torch.tensor([[0, 1, 0, 2], [4, 4, 4, 4]]),
        "terminal_timing_targets": torch.full((2, 4), 0.2),
        "terminal_timing_valid_mask": torch.tensor(
            [[False, True, False, True], [True, True, True, True]]
        ),
        "terminal_valid_mask": torch.tensor([True, False]),
        "terminal_representative_positions": torch.tensor([3, -1]),
    }


class ModelLossTests(unittest.TestCase):
    def setUp(self):
        self.batch = synthetic_batch()
        self.model = CustomEventEncoderModelV0(FakeEncoder(), hidden_size=16, dropout=0.0)
        main, terminal, self.provenance = load_slot_event_weights(WEIGHTS)
        self.criterion = CustomEventCriterion(main, terminal, CustomEventLossConfig())

    def forward(self):
        return self.model(
            input_ids=self.batch["input_ids"],
            token_attention_mask=self.batch["token_attention_mask"],
            note_mask=self.batch["note_mask"],
            owned_representative_positions=self.batch["owned_representative_positions"],
            owned_onset_mask=self.batch["owned_onset_mask"],
            initial_representative_positions=self.batch["initial_representative_positions"],
            initial_valid_mask=self.batch["initial_valid_mask"],
            terminal_representative_positions=self.batch["terminal_representative_positions"],
            terminal_valid_mask=self.batch["terminal_valid_mask"],
        )

    def test_shapes_and_weight_loading(self):
        output = self.forward()
        self.assertEqual(output.main_event_logits.shape, (2, 2, 6, 5))
        self.assertEqual(output.main_timing_predictions.shape, (2, 2, 6))
        self.assertEqual(output.initial_logits.shape, (2, 4))
        self.assertEqual(output.terminal_event_logits.shape, (2, 4, 5))
        self.assertEqual(output.terminal_timing_predictions.shape, (2, 4))
        self.assertAlmostEqual(
            float(self.criterion.main_event_weights[0, 0]), 0.2523498163940072
        )
        self.assertTrue(self.provenance["train_frequency_only"])

    def test_none_timing_placeholder_is_excluded(self):
        output = self.forward()
        first = self.criterion(output, self.batch).main_timing_huber
        changed = copy.deepcopy(self.batch)
        invalid = ~changed["main_timing_valid_mask"]
        changed["main_timing_targets"][invalid] = 1e6
        second = self.criterion(output, changed).main_timing_huber
        self.assertTrue(torch.equal(first, second))
        terminal_first = self.criterion(output, self.batch).terminal_timing_huber
        terminal_changed = copy.deepcopy(self.batch)
        terminal_invalid = ~terminal_changed["terminal_timing_valid_mask"]
        terminal_changed["terminal_timing_targets"][terminal_invalid] = 1e6
        terminal_second = self.criterion(
            output, terminal_changed
        ).terminal_timing_huber
        self.assertTrue(torch.equal(terminal_first, terminal_second))

    def test_non_none_timing_changes_loss_and_empty_slot_is_zero(self):
        output = self.forward()
        first = self.criterion(output, self.batch)
        changed = copy.deepcopy(self.batch)
        changed["main_timing_targets"][0, 0, 1] += 2.0
        second = self.criterion(output, changed)
        self.assertFalse(torch.equal(first.main_timing_huber, second.main_timing_huber))
        empty = copy.deepcopy(self.batch)
        empty["main_timing_valid_mask"][:, :, 5] = False
        loss = self.criterion(output, empty)
        self.assertEqual(float(loss.per_main_slot_timing_loss[5].detach()), 0.0)
        self.assertTrue(torch.isfinite(loss.per_main_slot_timing_loss[5]))

    def test_macro_slot_average_and_masks(self):
        output = self.forward()
        loss = self.criterion(output, self.batch)
        self.assertTrue(
            torch.allclose(
                loss.main_event_ce,
                torch.stack(loss.per_main_slot_event_loss).mean(),
            )
        )
        self.assertEqual(loss.valid_target_counts["initial"], 1)
        self.assertEqual(loss.valid_target_counts["terminal_event_by_slot"], [1] * 4)

    def test_initial_and_terminal_invalid_windows_do_not_contribute(self):
        output = self.forward()
        baseline = self.criterion(output, self.batch)
        initial_logits = output.initial_logits.clone()
        initial_logits[1] = torch.tensor([1e4, -1e4, 1e4, -1e4])
        terminal_logits = output.terminal_event_logits.clone()
        terminal_logits[1] = 1e4
        changed = replace(
            output,
            initial_logits=initial_logits,
            terminal_event_logits=terminal_logits,
        )
        altered = self.criterion(changed, self.batch)
        self.assertTrue(torch.equal(baseline.initial_ce, altered.initial_ce))
        self.assertTrue(
            torch.equal(baseline.terminal_event_ce, altered.terminal_event_ce)
        )

    def test_forward_backward_gradients_finite(self):
        output = self.forward()
        loss = self.criterion(output, self.batch)
        loss.total_loss.backward()
        gradients = [
            parameter.grad
            for parameter in self.model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(bool(torch.isfinite(gradient).all()) for gradient in gradients))
        self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0.0)


if __name__ == "__main__":
    unittest.main()
