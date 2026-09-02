from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage2_binary_2slot.dataset import (
    HumanIntervalPrimitive,
    PerformanceMajorBatchSampler,
    PerformanceMajorSampler,
    assert_masked_real_tokens,
    binary_2slot_collate_fn,
    build_boundary_input,
    mask_human_pedal_features,
)
from src.stage2_binary_2slot.losses import (
    FIXED_COUNT_WEIGHTS,
    Binary2SlotCriterion,
    Binary2SlotLossConfig,
)
from src.stage2_binary_2slot.model import StateConditionedBinary2SlotModel
from src.stage2_binary_2slot.rollout import (
    PerformanceStateStore,
    build_online_target,
    decode_prediction,
    next_binary_state,
)
from src.stage2_binary_2slot.trainer import StatefulRolloutEngine
from src.stage2_event_model.ownership import assign_unique_owners
from src.stage2_event_tokenizer.binary_2slot import (
    DOWN,
    OFF,
    ON,
    UP,
    BinaryTransition,
    state_after,
)


class FakeEncoder(nn.Module):
    def __init__(self, hidden_size: int = 768) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=hidden_size, dropout_rate=0.0, num_hidden_layers=2
        )
        self.embedding = nn.Embedding(32, hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        batch, flat = input_ids.shape
        notes = flat // 8
        values = self.embedding(input_ids).reshape(batch, notes, 8, -1).mean(dim=2)
        return SimpleNamespace(last_hidden_state=values)


def alternating(start: int, count: int) -> tuple[BinaryTransition, ...]:
    state = start
    result = []
    for index in range(count):
        result.append(
            BinaryTransition(
                DOWN if state == OFF else UP,
                (index + 1) / (count + 1),
            )
        )
        state = 1 - state
    return tuple(result)


def interval(
    region: str,
    index: int,
    human_state: int = OFF,
    count: int = 0,
) -> HumanIntervalPrimitive:
    return HumanIntervalPrimitive(
        region=region,
        global_index=index + 1,
        main_onset_index=index if region == "MAIN" else None,
        left_seconds=float(index),
        right_seconds=float(index + 1),
        human_start_state=human_state,
        human_transitions=alternating(human_state, count),
    )


def make_model() -> StateConditionedBinary2SlotModel:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = StateConditionedBinary2SlotModel(FakeEncoder(), hidden_size=768, dropout=0.0)
    model.head_init_seed = 42
    return model


def test_pedal_features_are_replaced_by_mask() -> None:
    raw = torch.tensor([[5, 261, 133, 262, 5261, 5262, 5263, 5264]])
    masked = mask_human_pedal_features(raw)
    assert torch.equal(masked[:, :4], raw[:, :4])
    assert torch.equal(masked[:, 4:], torch.ones((1, 4), dtype=torch.long))
    assert_masked_real_tokens(masked)


def test_pre_boundary_pseudo_note_and_attention() -> None:
    real = mask_human_pedal_features(torch.full((4, 8), 7))
    value = build_boundary_input(real, "PRE")
    notes = value["input_ids"].reshape(-1, 8)
    assert value["query_position"] == 0
    assert torch.equal(notes[0], torch.ones(8, dtype=torch.long))
    assert torch.equal(notes[1:], real)
    assert bool(torch.all(value["token_attention_mask"] == 1))
    assert bool(torch.all(value["note_mask"]))


def test_post_boundary_short_query_position() -> None:
    real = mask_human_pedal_features(torch.full((7, 8), 9))
    value = build_boundary_input(real, "POST")
    notes = value["input_ids"].reshape(-1, 8)
    assert value["query_position"] == 7
    assert value["real_context_notes"] == 7
    assert torch.equal(notes[-1], torch.ones(8, dtype=torch.long))
    assert torch.equal(notes[:-1], real)


def test_boundary_caps_at_512_and_requires_no_midi_object() -> None:
    real = mask_human_pedal_features(torch.full((700, 8), 6))
    pre = build_boundary_input(real, "PRE")
    post = build_boundary_input(real, "POST")
    assert pre["input_ids"].numel() == post["input_ids"].numel() == 512 * 8
    assert pre["real_context_notes"] == post["real_context_notes"] == 511
    assert post["query_position"] == 511


def test_collate_preserves_pedal_mask_and_no_targets() -> None:
    tokens = mask_human_pedal_features(torch.full((3, 8), 7)).reshape(-1)
    sample = {
        "input_ids": tokens,
        "note_mask": torch.ones(3, dtype=torch.bool),
        "owned_global_onset_indices": torch.tensor([0, 1]),
        "owned_representative_positions": torch.tensor([0, 2]),
        "human_intervals": (interval("MAIN", 0), interval("MAIN", 1)),
        "metadata": {"performance_index": 0},
    }
    batch = binary_2slot_collate_fn((sample,))
    assert "main_event_targets" not in batch and "main_timing_targets" not in batch
    reshaped = batch["input_ids"].reshape(1, 3, 8)
    assert bool(torch.all(reshaped[:, :, 4:] == 1))


def test_model_shapes_parameter_count_and_statelessness() -> None:
    model = make_model()
    tokens = torch.randint(1, 20, (2, 4 * 8))
    tokens.reshape(2, 4, 8)[:, :, 4:] = 1
    mask = torch.ones_like(tokens)
    note_mask = torch.ones((2, 4), dtype=torch.bool)
    encoding = model.encode_main(
        tokens, mask, note_mask,
        torch.tensor([[0, 3], [1, 2]]),
        torch.ones((2, 2), dtype=torch.bool),
    )
    assert encoding.encoder_hidden_states.shape == (2, 4, 768)
    assert encoding.owned_onset_hidden_states.shape == (2, 2, 768)
    output = model.condition_and_predict(
        encoding.owned_onset_hidden_states, torch.tensor([[0, 1], [1, 0]])
    )
    assert output.conditioned_states.shape == (2, 2, 769)
    assert output.count_logits.shape == (2, 2, 3)
    assert output.timing_predictions.shape == (2, 2, 2)
    assert model.prediction_head_parameter_count == 3850
    assert model.head_parameter_counts == {
        "count": 2310, "timing_1": 770, "timing_2": 770, "total": 3850
    }
    assert not hasattr(model, "initial_head")
    assert not hasattr(model, "terminal_head")
    assert not hasattr(model, "current_state")

    leaked = tokens.clone()
    leaked.reshape(2, 4, 8)[0, 0, 4] = 7
    try:
        model.encode_main(
            leaked, mask, note_mask,
            torch.tensor([[0, 3], [1, 2]]),
            torch.ones((2, 2), dtype=torch.bool),
        )
    except AssertionError:
        pass
    else:
        raise AssertionError("model must reject active human Pedal1-4 input")


def test_boundary_encoder_returns_hidden_size() -> None:
    model = make_model()
    tokens = mask_human_pedal_features(torch.full((5, 8), 7))
    boundary = build_boundary_input(tokens, "POST")
    hidden = model.encode_boundary(
        boundary["input_ids"].unsqueeze(0),
        boundary["token_attention_mask"].unsqueeze(0),
        boundary["note_mask"].unsqueeze(0),
        torch.tensor([boundary["query_position"]]),
    )
    assert hidden.shape == (1, 768)


def test_hard_state_parity_and_direction_decode() -> None:
    assert next_binary_state(OFF, 0) == OFF
    assert next_binary_state(OFF, 1) == ON
    assert next_binary_state(OFF, 2) == OFF
    assert next_binary_state(ON, 1) == OFF
    assert next_binary_state(ON, 2) == ON
    decoded = decode_prediction(OFF, 2, (1.4, -0.2))
    assert [event.tau for event in decoded.transitions] == [0.0, 1.0]
    assert [event.direction for event in decoded.transitions] == [DOWN, UP]
    assert decoded.next_state == OFF
    assert decode_prediction(ON, 1, (0.3, 0.9)).transitions[0].direction == UP


def test_online_reconciliation_match_and_mismatch_k2_frozen() -> None:
    matched = build_online_target(OFF, interval("MAIN", 0, OFF, 1))
    assert matched.count == 1 and not matched.correction_required
    mismatch_interval = interval("MAIN", 0, ON, 2)
    mismatch = build_online_target(OFF, mismatch_interval)
    assert mismatch.correction_required
    assert not mismatch.correction_retained
    assert mismatch.count == 1
    assert mismatch.retained_real_human_transitions == 1
    assert state_after(OFF, mismatch.retained_transitions) == mismatch_interval.human_end_state


def test_target_count_controls_timing_mask_and_huber_is_finite() -> None:
    criterion = Binary2SlotCriterion(Binary2SlotLossConfig(boundary_loss_weight=1.0))
    assert tuple(float(v) for v in criterion.count_weights.tolist()) == tuple(
        float(torch.tensor(v)) for v in FIXED_COUNT_WEIGHTS
    )
    logits = torch.randn(4, 3, requires_grad=True)
    timing = torch.randn(4, 2, requires_grad=True)
    counts = torch.tensor([0, 1, 2, 1])
    targets = torch.tensor([[0.0, 0.0], [0.2, 0.0], [0.3, 0.8], [0.5, 0.0]])
    active = torch.tensor([[False, False], [True, False], [True, True], [True, False]])
    loss = criterion(logits, timing, counts, targets, active, ("PRE", "MAIN", "MAIN", "POST"))
    assert bool(torch.isfinite(loss.total_loss))
    assert loss.counts["main_intervals"] == 2
    assert loss.counts["boundary_intervals"] == 2
    loss.total_loss.backward()
    assert bool(torch.isfinite(logits.grad).all())
    assert bool(torch.isfinite(timing.grad).all())


def test_loss_config_requires_explicit_boundary_weight() -> None:
    try:
        Binary2SlotLossConfig()  # type: ignore[call-arg]
    except TypeError:
        pass
    else:
        raise AssertionError("boundary_loss_weight must be explicit")


def test_unique_owner_and_monotonic_chronology() -> None:
    first = np.array([0, 2, 300, 520])
    last = np.array([1, 3, 301, 521])
    rep = last.copy()
    ownership = assign_unique_owners(700, first, last, rep, window_notes=512, stride_notes=256)
    membership = np.zeros(len(first), dtype=int)
    flattened = []
    for owned in ownership.owned_onset_indices:
        membership[owned] += 1
        flattened.extend(owned.tolist())
    assert np.all(membership == 1)
    assert np.all(np.diff(ownership.owner_window_index) >= 0)
    assert flattened == list(range(len(first)))


def test_performance_major_sampler_keeps_window_order() -> None:
    class FakeDataset:
        performances = (object(), object())
        windows = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]

        def __len__(self) -> int:
            return len(self.windows)

    fake = FakeDataset()
    sampler = PerformanceMajorSampler(fake, shuffle=True, seed=42)
    order = list(sampler)
    grouped = [[fake.windows[i][1] for i in order if fake.windows[i][0] == p] for p in (0, 1)]
    assert grouped == [[0, 1, 2], [0, 1]]

    batch_sampler = PerformanceMajorBatchSampler(
        fake, batch_size=2, shuffle_performances=True, seed=42
    )
    batches = list(batch_sampler)
    assert len(batches) == 3
    for batch in batches:
        performance_ids = {fake.windows[index][0] for index in batch}
        assert len(performance_ids) == 1
        window_ids = [fake.windows[index][1] for index in batch]
        assert window_ids == sorted(window_ids)


def test_state_store_survives_chunks_and_requires_pre_main_post() -> None:
    store = PerformanceStateStore()
    assert store.begin("p") == OFF
    store.record_interval("p", "PRE", 1)
    assert store.current("p") == ON
    store.record_interval("p", "MAIN", 2, main_onset_index=0)
    assert store.current("p") == ON
    # This read represents a micro-batch/accumulation boundary: no reset API is invoked.
    assert store.current("p") == ON
    store.record_interval("p", "MAIN", 1, main_onset_index=1)
    store.record_interval("p", "POST", 0)
    assert store.finish("p", expected_main_count=2) == OFF


def test_stateful_rollout_engine_online_targets_and_lifecycle() -> None:
    model = make_model()
    engine = StatefulRolloutEngine(model)
    pre = interval("PRE", -1, OFF, 0)
    mains = (interval("MAIN", 0, OFF, 1), interval("MAIN", 1, ON, 0))
    post = interval("POST", 2, ON, 1)
    pre_chunk = engine.begin_performance("p", torch.randn(768), pre)
    main_a = engine.process_main_chunk("p", torch.randn(1, 768), mains[:1])
    assert engine.states.current("p") in {OFF, ON}
    main_b = engine.process_main_chunk("p", torch.randn(1, 768), mains[1:])
    post_chunk = engine.end_performance("p", torch.randn(768), post, expected_main_count=2)
    assert pre_chunk.regions == ("PRE",)
    assert main_a.regions + main_b.regions == ("MAIN", "MAIN")
    assert post_chunk.regions == ("POST",)
    assert engine.diagnostics.as_dict()["region_counts"] == {"PRE": 1, "MAIN": 2, "POST": 1}


def run_direct_test_suite() -> dict[str, str]:
    results: dict[str, str] = {}
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            value()
            results[name] = "passed"
    return results


if __name__ == "__main__":
    print(run_direct_test_suite())
