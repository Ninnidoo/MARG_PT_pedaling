from __future__ import annotations

import hashlib
import json
import types
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.stage2_event_model.dataset import CANONICAL_ALIGNMENT_ID, CANONICAL_CACHE_ID
from src.stage2_event_model.dataset_v1_state_conditioned import (
    CustomEventStateConditionedWindowDataset,
    custom_event_state_conditioned_collate_fn,
    derive_main_pre_state_targets,
)
from src.stage2_event_model.losses import CustomEventLossConfig, load_slot_event_weights
from src.stage2_event_model.losses_v1_state_conditioned import (
    STATE_LOSS_COEFFICIENT,
    CustomEventStateConditionedCriterion,
)
from src.stage2_event_model.model import CustomEventEncoderModelV0
from src.stage2_event_model.model_v1_state_conditioned import (
    MODEL_VERSION,
    CustomEventEncoderModelV1StateConditioned,
)
from src.stage2_event_model.prediction_decoder_v1 import (
    DECODER_ID,
    DECODER_VERSION,
    MainIntervalBoundary,
    PerformanceTimeline,
    decode_predictions_v1,
)


ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = ROOT / "analysis/custom_event_tokenizer_v1"
WEIGHTS = CACHE_ROOT / "candidate_event_weights.json"
SUBSET_MANIFEST = (
    ROOT
    / "analysis/custom_event_model_v1_state_conditioned_impl"
    / "tiny_overfit_subset_manifest.json"
)
EXPECTED_WEIGHT_SHA = "26da9ede3373bd3df5c3f28c259e29f173a0920e080eb3a27cdbc5c790ed5b55"
EXPECTED_ALIGNMENT_SHA = "77df6faf08025794e40c58d0db31cf721f582b3168d9fb201557c0500807502b"
EXPECTED_DECODER_ID = "938ca7e20e1d398aa530938bbcede404fd3a04af23842fbcf351ba02df0268f8"


class FakeEncoder(nn.Module):
    def __init__(self, hidden_size: int = 16):
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=hidden_size, dropout_rate=0.0)
        self.embedding = nn.Embedding(128, hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        batch, flat = input_ids.shape
        notes = flat // 8
        hidden = self.embedding(input_ids.view(batch, notes, 8)[:, :, 0])
        return types.SimpleNamespace(last_hidden_state=hidden)


def synthetic_batch() -> dict[str, torch.Tensor]:
    events = torch.tensor(
        [
            [[2, 1, 3, 4, 2, 1], [1, 2, 0, 0, 0, 0]],
            [[4, 3, 2, 0, 0, 0], [-100, -100, -100, -100, -100, -100]],
        ],
        dtype=torch.long,
    )
    pre = torch.tensor(
        [
            [[0, 1, 0, 2, 3, 1], [0, 0, 1, 1, 1, 1]],
            [[2, 3, 2, 1, 1, 1], [-100, -100, -100, -100, -100, -100]],
        ],
        dtype=torch.long,
    )
    owned = torch.tensor([[True, True], [True, False]])
    terminal = torch.tensor([[1, 2, 3, 4], [0, 0, 0, 0]])
    return {
        "input_ids": torch.arange(2 * 4 * 8).reshape(2, 4 * 8) % 100,
        "token_attention_mask": torch.ones(2, 4 * 8, dtype=torch.long),
        "note_mask": torch.ones(2, 4, dtype=torch.bool),
        "owned_representative_positions": torch.tensor([[1, 3], [0, -1]]),
        "owned_onset_mask": owned,
        "main_event_targets": events,
        "main_pre_state_targets": pre,
        "main_timing_targets": torch.full((2, 2, 6), 0.25),
        "main_timing_valid_mask": (events != 0) & owned.unsqueeze(-1),
        "initial_targets": torch.tensor([1, 3]),
        "initial_valid_mask": torch.tensor([True, False]),
        "initial_representative_positions": torch.tensor([1, -1]),
        "terminal_event_targets": terminal,
        "terminal_timing_targets": torch.full((2, 4), 0.2),
        "terminal_timing_valid_mask": terminal != 0,
        "terminal_valid_mask": torch.tensor([True, False]),
        "terminal_representative_positions": torch.tensor([3, -1]),
    }


def forward(model, batch):
    return model(
        input_ids=batch["input_ids"],
        token_attention_mask=batch["token_attention_mask"],
        note_mask=batch["note_mask"],
        owned_representative_positions=batch["owned_representative_positions"],
        owned_onset_mask=batch["owned_onset_mask"],
        initial_representative_positions=batch["initial_representative_positions"],
        initial_valid_mask=batch["initial_valid_mask"],
        terminal_representative_positions=batch["terminal_representative_positions"],
        terminal_valid_mask=batch["terminal_valid_mask"],
    )


def group_grad(module: nn.Module) -> tuple[int, float, bool]:
    gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
    return (
        len(gradients),
        sum(float(value.detach().abs().sum()) for value in gradients),
        all(bool(torch.isfinite(value).all()) for value in gradients),
    )


def module_sha(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(array.dtype.str.encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


class PreStateTargetTests(unittest.TestCase):
    def test_exact_state_machine_and_post_none_labels(self):
        events = np.asarray(
            [[2, 1, 3, 4, 0, 0], [2, 0, 0, 0, 0, 0], [1, 2, 0, 0, 0, 0]],
            dtype=np.int8,
        )
        starts = np.asarray([0, 3, 1], dtype=np.int8)
        expected = np.asarray(
            [[0, 1, 0, 2, 3, 3], [3, 1, 1, 1, 1, 1], [1, 0, 1, 1, 1, 1]],
            dtype=np.int8,
        )
        first = derive_main_pre_state_targets(0, events, starts)
        second = derive_main_pre_state_targets(0, events.copy(), starts.copy())
        self.assertTrue(np.array_equal(first, expected))
        self.assertEqual(first.tobytes(), second.tobytes())

    def test_first_none_violation_and_bad_continuity_fail(self):
        with self.assertRaises(AssertionError):
            derive_main_pre_state_targets(
                0, np.asarray([[0, 2, 0, 0, 0, 0]]), np.asarray([0])
            )
        with self.assertRaises(AssertionError):
            derive_main_pre_state_targets(
                0,
                np.asarray([[2, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]),
                np.asarray([0, 3]),
            )


class DatasetCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = CustomEventStateConditionedWindowDataset(
            CACHE_ROOT, "train", entry_limit=1, cache_input_tokens=True
        )

    def test_cache_alignment_ownership_and_repeat_determinism(self):
        self.assertEqual(self.dataset.cache_config["cache_id"], CANONICAL_CACHE_ID)
        alignment = json.loads(
            (ROOT / "analysis/custom_event_model_v0_note_alignment/alignment_manifest.json").read_text()
        )
        self.assertEqual(alignment["alignment_id"], CANONICAL_ALIGNMENT_ID)
        self.assertEqual(alignment["alignment_config_sha256"], EXPECTED_ALIGNMENT_SHA)
        performance = self.dataset.performances[0]
        self.assertEqual(
            sum(map(len, performance.ownership.owned_onset_indices)), performance.num_onsets
        )
        first = self.dataset[0]
        second = self.dataset[0]
        self.assertTrue(torch.equal(first["main_pre_state_targets"], second["main_pre_state_targets"]))
        self.assertEqual(
            first["main_pre_state_targets"].numpy().tobytes(),
            second["main_pre_state_targets"].numpy().tobytes(),
        )
        batch = custom_event_state_conditioned_collate_fn([first])
        self.assertTrue(
            torch.equal(
                batch["main_pre_state_targets"] != -100,
                batch["owned_onset_mask"].unsqueeze(-1).expand_as(
                    batch["main_pre_state_targets"]
                ),
            )
        )


class SelectorArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = json.loads(SUBSET_MANIFEST.read_text(encoding="utf-8"))

    def test_train_only_provenance_and_complete_coverage(self):
        payload = self.payload
        self.assertEqual(payload["cache_id"], CANONICAL_CACHE_ID)
        self.assertEqual(payload["alignment_id"], CANONICAL_ALIGNMENT_ID)
        self.assertEqual(payload["split"], "train")
        self.assertEqual(payload["seed"], 42)
        self.assertEqual(payload["missing_required"], [])
        self.assertLessEqual(set(payload["required_coverage"]), set(payload["covered"]))
        self.assertEqual(payload["validation_target_access_count"], 0)
        self.assertEqual(payload["validation_model_inference_count"], 0)
        self.assertEqual(payload["asap_test_access_count"], 0)
        self.assertEqual(payload["repedal_execution_count"], 0)
        self.assertLessEqual(len(payload["selected"]), 16)
        self.assertEqual(len(payload["selection_sha256"]), 64)

    def test_every_selected_window_adds_coverage_and_stats_are_nonzero(self):
        covered: set[str] = set()
        for expected_order, row in enumerate(self.payload["selected"]):
            self.assertEqual(row["selection_order"], expected_order)
            newly = set(row["newly_covered"])
            self.assertTrue(newly)
            self.assertTrue(newly.isdisjoint(covered))
            covered.update(row["coverage"])
        stats = self.payload["summary"]
        self.assertEqual(stats["windows"], len(self.payload["selected"]))
        self.assertEqual(
            set(stats["initial_state_supervision_counts"]),
            {"ZERO", "LOW", "HALF", "FULL"},
        )
        self.assertGreaterEqual(stats["same_state_main_target_count"], 1)
        self.assertTrue(
            all(value > 0 for value in stats["per_slot_active_counts"].values())
        )
        self.assertTrue(
            all(value > 0 for value in stats["terminal_slot_active_counts"].values())
        )


class ArchitectureGradientTests(unittest.TestCase):
    def setUp(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            self.model = CustomEventEncoderModelV1StateConditioned(
                FakeEncoder(), hidden_size=16, dropout=0.0
            )
        self.batch = synthetic_batch()
        main, terminal, provenance = load_slot_event_weights(WEIGHTS)
        self.assertEqual(provenance["sha256"], EXPECTED_WEIGHT_SHA)
        self.criterion = CustomEventStateConditionedCriterion(
            main, terminal, CustomEventLossConfig()
        )

    def losses(self):
        return self.criterion(forward(self.model, self.batch), self.batch)

    def assert_group(self, module, *, expected_nonzero):
        count, magnitude, finite = group_grad(module)
        self.assertTrue(finite)
        if expected_nonzero:
            self.assertGreater(count, 0)
            self.assertGreater(magnitude, 0.0)
        else:
            self.assertEqual(magnitude, 0.0)

    def test_shapes_q_normalization_and_decoder_interface(self):
        output = forward(self.model, self.batch)
        self.assertEqual(output.main_state_logits.shape, (2, 2, 6, 4))
        self.assertEqual(output.main_state_posteriors.shape, (2, 2, 6, 4))
        self.assertTrue(torch.isfinite(output.main_state_posteriors).all())
        self.assertTrue(
            torch.allclose(
                output.main_state_posteriors.sum(-1), torch.ones(2, 2, 6)
            )
        )
        self.assertTrue(all(head.in_features == 20 for head in self.model.main_event_heads))
        self.assertEqual(output.main_event_logits.shape, (2, 2, 6, 5))
        self.assertEqual(output.main_timing_predictions.shape, (2, 2, 6))
        self.assertEqual(output.initial_logits.shape, (2, 4))
        self.assertEqual(output.terminal_event_logits.shape, (2, 4, 5))
        self.assertEqual(output.terminal_timing_predictions.shape, (2, 4))
        timeline = PerformanceTimeline(
            0.0,
            2.0,
            (
                MainIntervalBoundary(0, 0.0, 1.0, False),
                MainIntervalBoundary(1, 1.0, 2.0, True),
            ),
        )
        decoded = decode_predictions_v1(
            initial_logits=output.initial_logits[0],
            main_event_logits=output.main_event_logits[0],
            main_timing_predictions=output.main_timing_predictions[0],
            terminal_event_logits=output.terminal_event_logits[0],
            terminal_timing_predictions=output.terminal_timing_predictions[0],
            timeline=timeline,
        )
        self.assertIsInstance(decoded.diagnostics, dict)
        self.assertEqual(DECODER_VERSION, "1.0.0")
        self.assertEqual(DECODER_ID, EXPECTED_DECODER_ID)

    def test_state_ce_gradient_isolation(self):
        self.model.zero_grad(set_to_none=True)
        self.losses().main_state_ce.backward()
        self.assert_group(self.model.encoder, expected_nonzero=False)
        self.assert_group(self.model.main_state_heads, expected_nonzero=True)
        self.assert_group(self.model.main_event_heads, expected_nonzero=False)
        self.assert_group(self.model.main_timing_heads, expected_nonzero=False)

    def test_event_ce_gradient_isolation(self):
        self.model.zero_grad(set_to_none=True)
        self.losses().main_event_ce.backward()
        self.assert_group(self.model.encoder, expected_nonzero=True)
        self.assert_group(self.model.main_state_heads, expected_nonzero=False)
        self.assert_group(self.model.main_event_heads, expected_nonzero=True)
        self.assert_group(self.model.main_timing_heads, expected_nonzero=False)

    def test_timing_gradient_isolation(self):
        self.model.zero_grad(set_to_none=True)
        self.losses().main_timing_huber.backward()
        self.assert_group(self.model.encoder, expected_nonzero=True)
        self.assert_group(self.model.main_state_heads, expected_nonzero=False)
        self.assert_group(self.model.main_event_heads, expected_nonzero=False)
        self.assert_group(self.model.main_timing_heads, expected_nonzero=True)

    def test_combined_gradients_are_finite_and_routed(self):
        self.model.zero_grad(set_to_none=True)
        self.losses().total_loss.backward()
        for module in (
            self.model.encoder,
            self.model.initial_head,
            self.model.main_state_heads,
            self.model.main_event_heads,
            self.model.main_timing_heads,
            self.model.terminal_event_heads,
            self.model.terminal_timing_heads,
        ):
            self.assert_group(module, expected_nonzero=True)

    def test_q_feature_connectivity_with_nonzero_state_columns(self):
        hidden = torch.randn(1, 2, 16)
        q_zero = torch.zeros(1, 2, 6, 4)
        q_one = torch.zeros_like(q_zero)
        q_zero[..., 0] = 1.0
        q_one[..., 1] = 1.0
        with torch.no_grad():
            for head in self.model.main_event_heads:
                head.weight[:, -4:] = torch.arange(
                    20, dtype=head.weight.dtype
                ).reshape(5, 4)
        first = self.model._event_predictions(hidden, q_zero)
        second = self.model._event_predictions(hidden, q_one)
        self.assertFalse(torch.equal(first, second))

    def test_nonfinite_main_feature_fails_fast(self):
        with self.assertRaises(FloatingPointError):
            self.model._main_predictions(torch.full((1, 1, 16), float("nan")))

    def test_loss_separates_optimization_and_selection(self):
        loss = self.losses()
        self.assertEqual(STATE_LOSS_COEFFICIENT, 1.0)
        self.assertTrue(
            torch.allclose(
                loss.total_loss, loss.legacy_selection_loss + loss.main_state_ce
            )
        )
        self.assertEqual(loss.valid_target_counts["main_state_by_slot"], [3] * 6)


class InitializationAccountingTests(unittest.TestCase):
    def build(self, cls):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            return cls(FakeEncoder(768), hidden_size=768, dropout=0.0)

    def test_parameter_delta_and_legacy_initialization(self):
        v0 = self.build(CustomEventEncoderModelV0)
        v1 = self.build(CustomEventEncoderModelV1StateConditioned)
        self.assertEqual(v1.parameter_count - v0.parameter_count, 18576)
        self.assertEqual(v1.main_event_heads[0].in_features, 772)
        for old, new in zip(v0.main_event_heads, v1.main_event_heads, strict=True):
            self.assertTrue(torch.equal(old.weight, new.weight[:, :768]))
            self.assertTrue(torch.equal(old.bias, new.bias))
            self.assertEqual(float(new.weight[:, 768:].detach().abs().sum()), 0.0)
        for name in (
            "initial_head",
            "main_timing_heads",
            "terminal_event_heads",
            "terminal_timing_heads",
        ):
            self.assertEqual(module_sha(getattr(v0, name)), module_sha(getattr(v1, name)))

    def test_step_zero_v0_output_regression(self):
        v0 = self.build(CustomEventEncoderModelV0)
        v1 = self.build(CustomEventEncoderModelV1StateConditioned)
        batch = synthetic_batch()
        v0_output = forward(v0, batch)
        v1_output = forward(v1, batch)
        self.assertTrue(
            torch.allclose(
                v0_output.main_event_logits,
                v1_output.main_event_logits,
                rtol=1e-6,
                atol=1e-6,
            )
        )
        for name in (
            "initial_logits",
            "main_timing_predictions",
            "terminal_event_logits",
            "terminal_timing_predictions",
            "encoder_hidden_states",
            "owned_onset_hidden_states",
        ):
            self.assertTrue(
                torch.equal(getattr(v0_output, name), getattr(v1_output, name)),
                name,
            )

    def test_same_seed_is_deterministic(self):
        first = self.build(CustomEventEncoderModelV1StateConditioned)
        second = self.build(CustomEventEncoderModelV1StateConditioned)
        self.assertEqual(module_sha(first), module_sha(second))
        self.assertEqual(first.model_version, MODEL_VERSION)


if __name__ == "__main__":
    unittest.main()
