#!/usr/bin/env python3
"""Correctness-gated train-only coverage memorization for frozen B3-S v1."""

from __future__ import annotations

import json
import math
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_custom_event_model_v0_tiny_overfit import (
    atomic_csv,
    atomic_json,
    event_summary,
    gradient_norm,
    move_batch,
    seed_everything,
    timing_summary,
)
from src.stage2_event_model.dataset import CANONICAL_ALIGNMENT_ID, CANONICAL_CACHE_ID
from src.stage2_event_model.dataset_v1_state_conditioned import (
    CustomEventStateConditionedWindowDataset,
    custom_event_state_conditioned_collate_fn,
)
from src.stage2_event_model.losses import CustomEventLossConfig, load_slot_event_weights
from src.stage2_event_model.losses_v1_state_conditioned import (
    STATE_LOSS_COEFFICIENT,
    CustomEventStateConditionedCriterion,
)
from src.stage2_event_model.model_v1_state_conditioned import (
    MODEL_VERSION,
    CustomEventEncoderModelV1StateConditioned,
)
from src.stage2_event_model.prediction_decoder_v1 import DECODER_ID, DECODER_VERSION


CACHE_ROOT = ROOT / "analysis/custom_event_tokenizer_v1"
OUTPUT_ROOT = ROOT / "analysis/custom_event_model_v1_state_conditioned_impl"
SUBSET_PATH = OUTPUT_ROOT / "tiny_overfit_subset_manifest.json"
CHECKPOINT = ROOT / "checkpoints/pianist_transformer"
WEIGHTS = CACHE_ROOT / "candidate_event_weights.json"
TEST_MODULE = "tests.test_custom_event_model_v1_state_conditioned"
STATE_NAMES = ("ZERO", "LOW", "HALF", "FULL")
SEED = 42
MIN_STEPS = 1000
MAX_STEPS = 12000
LOG_INTERVAL = 25
ENCODER_LR = 1e-5
HEAD_LR = 1e-4
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
EXPECTED_V0_PARAMETERS = 103_320_640
EXPECTED_V1_PARAMETERS = 103_339_216
EXPECTED_DELTA = 18_576
REQUIRED_TRANSITIONS = (
    "ZERO_LOW",
    "LOW_ZERO",
    "HALF_FULL",
    "FULL_HALF",
    "OFF_ON",
    "ON_OFF",
    "ZERO_ON",
    "LOW_ON",
    "ON_ZERO",
    "ON_LOW",
)


def forward_model(model: CustomEventEncoderModelV1StateConditioned, batch: dict[str, Any]):
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


def run_correctness_tests() -> dict[str, Any]:
    command = [sys.executable, "-m", "unittest", TEST_MODULE, "-v"]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    payload = {
        "command": command,
        "returncode": completed.returncode,
        "passed": completed.returncode == 0,
        "output": completed.stdout,
    }
    atomic_json(OUTPUT_ROOT / "correctness_test_results.json", payload)
    if completed.returncode:
        raise RuntimeError("targeted B3-S correctness tests failed before tiny-overfit")
    return payload


def load_selected_samples() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    subset = json.loads(SUBSET_PATH.read_text(encoding="utf-8"))
    if subset["cache_id"] != CANONICAL_CACHE_ID:
        raise RuntimeError("tiny subset cache ID changed")
    if subset["alignment_id"] != CANONICAL_ALIGNMENT_ID:
        raise RuntimeError("tiny subset alignment ID changed")
    if subset["split"] != "train" or subset["missing_required"]:
        raise RuntimeError("tiny subset is not a complete train-only coverage set")
    forbidden = (
        "validation_target_access_count",
        "validation_model_inference_count",
        "asap_test_access_count",
        "repedal_execution_count",
    )
    if any(int(subset[key]) != 0 for key in forbidden):
        raise RuntimeError("tiny subset provenance reports forbidden access")
    selected = subset["selected"]
    last_entry = max(int(row["performance_index"]) for row in selected)
    dataset = CustomEventStateConditionedWindowDataset(
        CACHE_ROOT,
        "train",
        entry_limit=last_entry + 1,
        cache_input_tokens=True,
    )
    lookup = {pair: index for index, pair in enumerate(dataset.windows)}
    samples = []
    for row in selected:
        pair = (int(row["performance_index"]), int(row["window_index"]))
        sample = dataset[lookup[pair]]
        metadata = sample["metadata"]
        if metadata["performance_path"] != row["performance_path"]:
            raise AssertionError("tiny subset performance identity mismatch")
        if metadata["window_start_note"] != row["window_start_note"]:
            raise AssertionError("tiny subset window identity mismatch")
        if sample["owned_global_onset_indices"].tolist() != row["owned_global_onset_indices"]:
            raise AssertionError("tiny subset ownership identity mismatch")
        samples.append(sample)
    return subset, samples


def assert_coverage_masks(subset: dict[str, Any], samples: list[dict[str, Any]]) -> None:
    if not set(subset["required_coverage"]).issubset(subset["covered"]):
        raise AssertionError("required selector coverage is incomplete")
    initial_states: set[int] = set()
    terminal_active = [0] * 4
    main_active = [0] * 6
    state_counts = [0] * 4
    for sample in samples:
        if not torch.equal(sample["main_timing_valid_mask"], sample["main_event_targets"] != 0):
            raise AssertionError("Main NONE timing would contribute")
        if not torch.equal(
            sample["terminal_timing_valid_mask"], sample["terminal_event_targets"] != 0
        ):
            raise AssertionError("Terminal NONE timing would contribute")
        if bool(sample["initial_valid_mask"]):
            initial_states.add(int(sample["initial_target"]))
        if bool(sample["terminal_valid_mask"]):
            for slot in range(4):
                terminal_active[slot] += int(sample["terminal_event_targets"][slot] != 0)
        for slot in range(6):
            main_active[slot] += int((sample["main_event_targets"][:, slot] != 0).sum())
        for state in range(4):
            state_counts[state] += int((sample["main_pre_state_targets"] == state).sum())
    if initial_states != {0, 1, 2, 3}:
        raise AssertionError("tiny subset does not exercise all Initial states")
    if min(main_active) < 1 or min(terminal_active) < 1 or min(state_counts) < 1:
        raise AssertionError("tiny subset does not exercise every required branch")


def parameter_accounting(model: CustomEventEncoderModelV1StateConditioned) -> dict[str, Any]:
    actual_v1 = model.parameter_count
    inferred_v0 = actual_v1 - EXPECTED_DELTA
    main_v0 = 6 * (768 * 5 + 5)
    main_v1 = int(sum(parameter.numel() for parameter in model.main_event_heads.parameters()))
    state = int(sum(parameter.numel() for parameter in model.main_state_heads.parameters()))
    payload = {
        "hidden_size": model.hidden_size,
        "v0_expected_total_parameters": EXPECTED_V0_PARAMETERS,
        "v1_expected_total_parameters": EXPECTED_V1_PARAMETERS,
        "v1_actual_total_parameters": actual_v1,
        "v0_inferred_total_parameters": inferred_v0,
        "expected_delta": EXPECTED_DELTA,
        "actual_delta": actual_v1 - inferred_v0,
        "expanded_main_event_head_delta": main_v1 - main_v0,
        "main_state_head_parameters": state,
        "fixed_identity_embedding_parameters": 0,
        "head_parameter_counts": model.head_parameter_counts,
        "main_event_in_features": [head.in_features for head in model.main_event_heads],
        "passed": (
            actual_v1 == EXPECTED_V1_PARAMETERS
            and inferred_v0 == EXPECTED_V0_PARAMETERS
            and main_v1 - main_v0 == 120
            and state == 18_456
            and all(head.in_features == 772 for head in model.main_event_heads)
        ),
    }
    atomic_json(OUTPUT_ROOT / "parameter_accounting.json", payload)
    if not payload["passed"]:
        raise AssertionError("B3-S parameter accounting failed")
    return payload


def gradient_groups(
    model: CustomEventEncoderModelV1StateConditioned,
) -> dict[str, Iterable[torch.nn.Parameter]]:
    return {
        "encoder": model.encoder.parameters(),
        "initial_head": model.initial_head.parameters(),
        "main_state_heads": model.main_state_heads.parameters(),
        "main_event_heads": model.main_event_heads.parameters(),
        "main_timing_heads": model.main_timing_heads.parameters(),
        "terminal_event_heads": model.terminal_event_heads.parameters(),
        "terminal_timing_heads": model.terminal_timing_heads.parameters(),
    }


def gradient_snapshot(model: CustomEventEncoderModelV1StateConditioned) -> dict[str, Any]:
    result = {}
    for name, parameters in gradient_groups(model).items():
        norm, finite, tensor_count = gradient_norm(parameters)
        result[name] = {
            "norm": norm,
            "finite": finite,
            "gradient_tensor_count": tensor_count,
        }
    return result


def verify_gradient_isolation(
    model: CustomEventEncoderModelV1StateConditioned,
    criterion: CustomEventStateConditionedCriterion,
    batches: list[dict[str, Any]],
) -> dict[str, Any]:
    expected_nonzero = {
        "state_ce_only": {"main_state_heads"},
        "main_event_ce_only": {"encoder", "main_event_heads"},
        "main_timing_only": {"encoder", "main_timing_heads"},
        "combined": {
            "encoder",
            "initial_head",
            "main_state_heads",
            "main_event_heads",
            "main_timing_heads",
            "terminal_event_heads",
            "terminal_timing_heads",
        },
    }
    cases = {}
    model.eval()
    for case in expected_nonzero:
        model.zero_grad(set_to_none=True)
        selected_batches = batches if case == "combined" else batches[:1]
        for batch in selected_batches:
            loss = criterion(forward_model(model, batch), batch)
            selected = {
                "state_ce_only": loss.main_state_ce,
                "main_event_ce_only": loss.main_event_ce,
                "main_timing_only": loss.main_timing_huber,
                "combined": loss.total_loss,
            }[case]
            selected.backward()
        snapshot = gradient_snapshot(model)
        checks = {}
        for name, values in snapshot.items():
            if name in expected_nonzero[case]:
                checks[name] = (
                    values["gradient_tensor_count"] > 0
                    and values["norm"] > 0.0
                    and values["finite"]
                )
            else:
                checks[name] = values["norm"] == 0.0 and values["finite"]
        cases[case] = {
            "expected_nonzero": sorted(expected_nonzero[case]),
            "groups": snapshot,
            "checks": checks,
            "passed": all(checks.values()),
        }
    model.zero_grad(set_to_none=True)
    payload = {
        "detach_boundaries": {
            "encoder_to_state_head": "stop-gradient",
            "state_posterior_to_event_head": "stop-gradient",
        },
        "cases": cases,
        "all_cases_passed": all(row["passed"] for row in cases.values()),
        "all_recorded_gradients_finite": all(
            values["finite"] for row in cases.values() for values in row["groups"].values()
        ),
    }
    atomic_json(OUTPUT_ROOT / "gradient_isolation_results.json", payload)
    if not payload["all_cases_passed"]:
        raise AssertionError("B3-S gradient isolation failed; tiny-overfit is blocked")
    return payload


def class_summary(
    targets: list[torch.Tensor],
    predictions: list[torch.Tensor],
    names: tuple[str, ...],
) -> dict[str, Any]:
    target = torch.cat([value.reshape(-1).cpu() for value in targets])
    prediction = torch.cat([value.reshape(-1).cpu() for value in predictions])
    correct = int((target == prediction).sum())
    total = int(target.numel())
    class_accuracy = {}
    for index, name in enumerate(names):
        mask = target == index
        class_accuracy[name] = (
            float(((prediction == index) & mask).sum() / mask.sum()) if bool(mask.any()) else None
        )
    return {
        "target_count": total,
        "correct_count": correct,
        "accuracy": float(correct / total) if total else 0.0,
        "class_accuracy": class_accuracy,
    }


def transition_categories(pre: int, destination: int) -> set[str]:
    labels: set[str] = set()
    if (pre, destination) == (0, 1):
        labels.add("ZERO_LOW")
    if (pre, destination) == (1, 0):
        labels.add("LOW_ZERO")
    if (pre, destination) == (2, 3):
        labels.add("HALF_FULL")
    if (pre, destination) == (3, 2):
        labels.add("FULL_HALF")
    if pre < 2 <= destination:
        labels |= {"OFF_ON", "ZERO_ON" if pre == 0 else "LOW_ON"}
    if destination < 2 <= pre:
        labels |= {"ON_OFF", "ON_ZERO" if destination == 0 else "ON_LOW"}
    return labels


def evaluate(
    model: CustomEventEncoderModelV1StateConditioned,
    criterion: CustomEventStateConditionedCriterion,
    batches: list[dict[str, Any]],
) -> dict[str, Any]:
    model.eval()
    loss_keys = (
        "total_loss",
        "legacy_selection_loss",
        "main_state_ce",
        "initial_ce",
        "main_event_ce",
        "main_timing_huber",
        "terminal_event_ce",
        "terminal_timing_huber",
    )
    losses = {key: [] for key in loss_keys}
    state_targets = [[] for _ in range(6)]
    state_predictions = [[] for _ in range(6)]
    main_targets = [[] for _ in range(6)]
    main_predictions = [[] for _ in range(6)]
    main_time_targets = [[] for _ in range(6)]
    main_time_predictions = [[] for _ in range(6)]
    initial_targets = []
    initial_predictions = []
    terminal_targets = [[] for _ in range(4)]
    terminal_predictions = [[] for _ in range(4)]
    terminal_time_targets = [[] for _ in range(4)]
    terminal_time_predictions = [[] for _ in range(4)]
    transition = {
        name: {"target_count": 0, "correct_count": 0} for name in REQUIRED_TRANSITIONS
    }
    same_state = {
        "target_same_state_set_count": 0,
        "target_same_state_set_correct_count": 0,
        "predicted_target_semantic_same_state_set_count": 0,
        "redundant_predicted_target_semantic_same_state_set_count": 0,
    }
    q_confidence = []
    examples = []
    with torch.no_grad():
        for batch in batches:
            output = forward_model(model, batch)
            loss = criterion(output, batch)
            for key in loss_keys:
                value = getattr(loss, key)
                if not bool(torch.isfinite(value)):
                    raise FloatingPointError(f"non-finite evaluation loss: {key}")
                losses[key].append(float(value))
            valid = batch["owned_onset_mask"]
            q_confidence.append(output.main_state_posteriors.max(-1).values[valid])
            for slot in range(6):
                target_state = batch["main_pre_state_targets"][:, :, slot][valid]
                predicted_state = output.main_state_logits[:, :, slot].argmax(-1)[valid]
                target_event = batch["main_event_targets"][:, :, slot][valid]
                predicted_event = output.main_event_logits[:, :, slot].argmax(-1)[valid]
                state_targets[slot].append(target_state)
                state_predictions[slot].append(predicted_state)
                main_targets[slot].append(target_event)
                main_predictions[slot].append(predicted_event)
                timing_valid = batch["main_timing_valid_mask"][:, :, slot] & valid
                main_time_targets[slot].append(
                    batch["main_timing_targets"][:, :, slot][timing_valid]
                )
                main_time_predictions[slot].append(
                    output.main_timing_predictions[:, :, slot][timing_valid]
                )
                for pre, target, prediction in zip(
                    target_state.cpu().tolist(),
                    target_event.cpu().tolist(),
                    predicted_event.cpu().tolist(),
                    strict=True,
                ):
                    target_same = target > 0 and target - 1 == pre
                    predicted_same = prediction > 0 and prediction - 1 == pre
                    if target_same:
                        same_state["target_same_state_set_count"] += 1
                        same_state["target_same_state_set_correct_count"] += int(
                            prediction == target
                        )
                    if predicted_same:
                        same_state["predicted_target_semantic_same_state_set_count"] += 1
                        if not target_same:
                            same_state[
                                "redundant_predicted_target_semantic_same_state_set_count"
                            ] += 1
                    if target > 0:
                        for name in transition_categories(pre, target - 1):
                            transition[name]["target_count"] += 1
                            transition[name]["correct_count"] += int(prediction == target)
            initial_valid = batch["initial_valid_mask"]
            if bool(initial_valid.any()):
                initial_targets.append(batch["initial_targets"][initial_valid])
                initial_predictions.append(output.initial_logits.argmax(-1)[initial_valid])
            terminal_valid = batch["terminal_valid_mask"]
            for slot in range(4):
                if bool(terminal_valid.any()):
                    terminal_targets[slot].append(
                        batch["terminal_event_targets"][:, slot][terminal_valid]
                    )
                    terminal_predictions[slot].append(
                        output.terminal_event_logits[:, slot].argmax(-1)[terminal_valid]
                    )
                timing_valid = batch["terminal_timing_valid_mask"][:, slot] & terminal_valid
                if bool(timing_valid.any()):
                    terminal_time_targets[slot].append(
                        batch["terminal_timing_targets"][:, slot][timing_valid]
                    )
                    terminal_time_predictions[slot].append(
                        output.terminal_timing_predictions[:, slot][timing_valid]
                    )
            if len(examples) < 3:
                examples.append(
                    {
                        "performance_path": batch["metadata"][0]["performance_path"],
                        "window_index": batch["metadata"][0]["window_index"],
                        "state_target_first": batch["main_pre_state_targets"][0, :3].cpu().tolist(),
                        "state_prediction_first": output.main_state_logits[0, :3]
                        .argmax(-1)
                        .cpu()
                        .tolist(),
                        "event_target_first": batch["main_event_targets"][0, :3].cpu().tolist(),
                        "event_prediction_first": output.main_event_logits[0, :3]
                        .argmax(-1)
                        .cpu()
                        .tolist(),
                    }
                )
    for row in transition.values():
        row["accuracy"] = (
            float(row["correct_count"] / row["target_count"]) if row["target_count"] else 0.0
        )
    return {
        "losses": {key: statistics.fmean(values) for key, values in losses.items()},
        "state": class_summary(
            [value for rows in state_targets for value in rows],
            [value for rows in state_predictions for value in rows],
            STATE_NAMES,
        ),
        "state_by_slot": {
            f"Slot{slot + 1}": class_summary(
                state_targets[slot], state_predictions[slot], STATE_NAMES
            )
            for slot in range(6)
        },
        "state_posterior_mean_max_probability": float(torch.cat(q_confidence).mean()),
        "initial": class_summary(initial_targets, initial_predictions, STATE_NAMES),
        "main_event": event_summary(
            [value for rows in main_targets for value in rows],
            [value for rows in main_predictions for value in rows],
        ),
        "main_event_by_slot": {
            f"Slot{slot + 1}": event_summary(main_targets[slot], main_predictions[slot])
            for slot in range(6)
        },
        "main_timing": timing_summary(
            [value for rows in main_time_targets for value in rows],
            [value for rows in main_time_predictions for value in rows],
        ),
        "main_timing_by_slot": {
            f"Slot{slot + 1}": timing_summary(
                main_time_targets[slot], main_time_predictions[slot]
            )
            for slot in range(6)
        },
        "terminal_event": event_summary(
            [value for rows in terminal_targets for value in rows],
            [value for rows in terminal_predictions for value in rows],
        ),
        "terminal_event_by_slot": {
            f"Slot{slot + 1}": event_summary(
                terminal_targets[slot], terminal_predictions[slot]
            )
            for slot in range(4)
        },
        "terminal_timing": timing_summary(
            [value for rows in terminal_time_targets for value in rows],
            [value for rows in terminal_time_predictions for value in rows],
            seconds=True,
        ),
        "terminal_timing_by_slot": {
            f"Slot{slot + 1}": timing_summary(
                terminal_time_targets[slot],
                terminal_time_predictions[slot],
                seconds=True,
            )
            for slot in range(4)
        },
        "transition_coverage": transition,
        "same_state_diagnostic": same_state,
        "examples": examples,
    }


def flatten_log(step: int, elapsed: float, metrics: dict[str, Any]) -> dict[str, Any]:
    row = {
        "step": step,
        "elapsed_seconds": elapsed,
        **metrics["losses"],
        "state_accuracy": metrics["state"]["accuracy"],
        "initial_accuracy": metrics["initial"]["accuracy"],
        "main_event_accuracy": metrics["main_event"]["accuracy"],
        "main_non_none_recall": metrics["main_event"]["non_none_recall"],
        "main_timing_mae": metrics["main_timing"]["mae"],
        "terminal_event_accuracy": metrics["terminal_event"]["accuracy"],
        "terminal_timing_log1p_mae": metrics["terminal_timing"]["mae"],
    }
    for slot in range(1, 7):
        row[f"state_slot{slot}_accuracy"] = metrics["state_by_slot"][f"Slot{slot}"][
            "accuracy"
        ]
        row[f"main_slot{slot}_event_accuracy"] = metrics["main_event_by_slot"][
            f"Slot{slot}"
        ]["accuracy"]
        row[f"main_slot{slot}_non_none_recall"] = metrics["main_event_by_slot"][
            f"Slot{slot}"
        ]["non_none_recall"]
        row[f"main_slot{slot}_timing_mae"] = metrics["main_timing_by_slot"][
            f"Slot{slot}"
        ]["mae"]
    return row


def q_value_connectivity(model: CustomEventEncoderModelV1StateConditioned) -> dict[str, Any]:
    model.eval()
    parameter = next(model.parameters())
    hidden = torch.zeros(
        (1, 1, model.hidden_size), device=parameter.device, dtype=parameter.dtype
    )
    q_zero = torch.zeros((1, 1, 6, 4), device=hidden.device, dtype=hidden.dtype)
    q_one = torch.zeros_like(q_zero)
    q_zero[..., 0] = 1.0
    q_one[..., 1] = 1.0
    with torch.no_grad():
        delta = (
            model._event_predictions(hidden, q_zero)
            - model._event_predictions(hidden, q_one)
        ).abs()
    return {
        "comparison": "q=ZERO basis versus q=LOW basis at identical hidden input",
        "max_absolute_event_logit_delta": float(delta.max()),
        "finite": bool(torch.isfinite(delta).all()),
        "passed": bool(torch.isfinite(delta).all()) and float(delta.max()) > 0.0,
    }


def success_assessment(
    initial: dict[str, Any],
    final: dict[str, Any],
    gradients: dict[str, Any],
    connectivity: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "all_losses_finite": all(math.isfinite(value) for value in final["losses"].values()),
        "gradient_isolation_exact": gradients["all_cases_passed"],
        "state_accuracy_at_least_0_99": final["state"]["accuracy"] >= 0.99,
        "main_event_accuracy_at_least_0_99": final["main_event"]["accuracy"] >= 0.99,
        "initial_accuracy_at_least_0_99": final["initial"]["accuracy"] >= 0.99,
        "terminal_event_accuracy_at_least_0_99": final["terminal_event"]["accuracy"] >= 0.99,
        "all_main_slots_non_none_recall_at_least_0_99": all(
            row["non_none_recall"] >= 0.99 for row in final["main_event_by_slot"].values()
        ),
        "required_transition_directions_memorized": all(
            final["transition_coverage"][name]["target_count"] > 0
            and final["transition_coverage"][name]["accuracy"] >= 0.99
            for name in REQUIRED_TRANSITIONS
        ),
        "main_timing_memorized": (
            final["main_timing"]["mae"] <= 0.05
            and final["main_timing"]["mae"] < initial["main_timing"]["mae"]
        ),
        "terminal_timing_memorized": (
            final["terminal_timing"]["mae"] <= 0.05
            and final["terminal_timing"]["mae"] < initial["terminal_timing"]["mae"]
        ),
        "no_all_none_collapse": not final["main_event"]["all_none_prediction"],
        "q_to_event_value_connectivity": connectivity["passed"],
    }
    passed = all(checks.values())
    return {
        "checks": checks,
        "passed": passed,
        "verdict": (
            "READY FOR B3-S FULL TRAINING"
            if passed
            else "NEEDS IMPLEMENTATION FIX BEFORE FULL TRAINING"
        ),
        "full_training_started": False,
    }


def write_reports(
    configuration: dict[str, Any],
    accounting: dict[str, Any],
    gradients: dict[str, Any],
    metrics: dict[str, Any],
    decoder: dict[str, Any],
) -> None:
    final = metrics["final"]
    assessment = metrics["success_assessment"]
    implementation = f"""# Custom Event Model v1 B3-S — Implementation Report

## Frozen implementation

- Model: {MODEL_VERSION}; six Linear(768,4) state heads and six Linear(772,5) Main event heads.
- Conditioning: softmax(state_logits) as a fixed 4D posterior feature.
- Detach boundaries: encoder→state head and q→event head.
- State target: runtime derivation from the frozen train cache.
- State objective: unweighted Slot1–6 macro CE, fixed coefficient {STATE_LOSS_COEFFICIENT}.
- Legacy checkpoint-selection objective excludes state CE.

## Correctness gates

- Targeted tests: PASS ({TEST_MODULE}).
- Parameter accounting: {'PASS' if accounting['passed'] else 'FAIL'}; v0→v1 delta {accounting['actual_delta']:,}.
- Gradient isolation A/B/C/D: {'PASS' if gradients['all_cases_passed'] else 'FAIL'}.
- Decoder ID/version unchanged: {DECODER_ID} / {DECODER_VERSION}.
- Tokenizer cache ID unchanged: {CANONICAL_CACHE_ID}.
- Alignment ID unchanged: {CANONICAL_ALIGNMENT_ID}.

No validation inference, ASAP test, Repedal, full training, tokenizer rebuild,
decoder change, or candidate MIDI generation was performed.
"""
    tiny = f"""# Custom Event Model v1 B3-S — Coverage Tiny-Overfit Report

## Scope

Train-only deterministic selector {configuration['selection_sha256']} chose
{configuration['subset_windows']} owner windows from
{configuration['subset_performances']} performances.

## Final memorization

- State accuracy: {final['state']['accuracy']:.6f}
- Main event accuracy: {final['main_event']['accuracy']:.6f}
- Main non-NONE recall: {final['main_event']['non_none_recall']:.6f}
- Main timing MAE: {final['main_timing']['mae']:.6f}
- Initial accuracy: {final['initial']['accuracy']:.6f}
- Terminal event accuracy: {final['terminal_event']['accuracy']:.6f}
- Terminal timing log1p MAE: {final['terminal_timing']['mae']:.6f}
- q→event value connectivity: {'PASS' if decoder['q_to_event_connectivity']['passed'] else 'FAIL'}
- All-NONE collapse: {final['main_event']['all_none_prediction']}
- Target same-state SET count / correct: {final['same_state_diagnostic']['target_same_state_set_count']} / {final['same_state_diagnostic']['target_same_state_set_correct_count']}
- Redundant predicted target-semantic same-state SET: {final['same_state_diagnostic']['redundant_predicted_target_semantic_same_state_set_count']}

## Final verdict

**{assessment['verdict']}**

Full training was not started.
"""
    (OUTPUT_ROOT / "CUSTOM_EVENT_MODEL_V1_IMPLEMENTATION_REPORT.md").write_text(
        implementation, encoding="utf-8"
    )
    (OUTPUT_ROOT / "CUSTOM_EVENT_MODEL_V1_TINY_OVERFIT_REPORT.md").write_text(
        tiny, encoding="utf-8"
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("B3-S tiny-overfit requires the assigned CUDA GPU")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    correctness = run_correctness_tests()
    seed_everything(SEED)
    device = torch.device("cuda:0")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("container must expose exactly one GPU")
    subset, samples = load_selected_samples()
    assert_coverage_masks(subset, samples)
    batches = [
        move_batch(custom_event_state_conditioned_collate_fn([sample]), device)
        for sample in samples
    ]
    loss_config = CustomEventLossConfig()
    main_weights, terminal_weights, weight_provenance = load_slot_event_weights(WEIGHTS)
    criterion = CustomEventStateConditionedCriterion(
        main_weights, terminal_weights, loss_config
    ).to(device)
    model = CustomEventEncoderModelV1StateConditioned.from_pretrained(
        CHECKPOINT, head_init_seed=SEED
    ).to(device)
    if model.hidden_size != 768:
        raise RuntimeError("canonical pretrained hidden size changed")
    accounting = parameter_accounting(model)
    configuration = {
        "status": "correctness_gating",
        "model_version": MODEL_VERSION,
        "candidate": "B3-S",
        "cache_id": CANONICAL_CACHE_ID,
        "alignment_id": CANONICAL_ALIGNMENT_ID,
        "decoder_id": DECODER_ID,
        "decoder_version": DECODER_VERSION,
        "selection_sha256": subset["selection_sha256"],
        "subset_windows": len(subset["selected"]),
        "subset_performances": subset["summary"]["performances"],
        "subset_owned_onsets": subset["summary"]["owned_onsets"],
        "seed": SEED,
        "max_optimizer_steps": MAX_STEPS,
        "minimum_optimizer_steps": MIN_STEPS,
        "early_stop_policy": "first 25-step evaluation passing every frozen memorization gate",
        "logging_interval": LOG_INTERVAL,
        "precision": "FP32",
        "micro_batch_size": 1,
        "gradient_accumulation": 1,
        "optimizer": "AdamW",
        "encoder_lr": ENCODER_LR,
        "head_lr": HEAD_LR,
        "weight_decay": WEIGHT_DECAY,
        "max_gradient_norm": MAX_GRAD_NORM,
        "scheduler": None,
        "loss_config": asdict(loss_config),
        "state_loss_coefficient": STATE_LOSS_COEFFICIENT,
        "state_loss_weighting": "unweighted slot macro-average",
        "legacy_selection_excludes_state_ce": True,
        "head_initialization": "fresh deterministic seed-42 B3-S initialization",
        "warm_start": False,
        "full_training": False,
        "validation_target_access_count": 0,
        "validation_model_inference_count": 0,
        "asap_test_access_count": 0,
        "repedal_execution_count": 0,
        "candidate_midi_generation_count": 0,
        "tokenizer_change": False,
        "decoder_change": False,
        "event_weight_provenance": weight_provenance,
        "correctness_test_passed": correctness["passed"],
        "cuda_device": torch.cuda.get_device_name(device),
    }
    atomic_json(OUTPUT_ROOT / "implementation_config.json", configuration)
    gradients = verify_gradient_isolation(model, criterion, batches)
    configuration["status"] = "tiny_overfit_running"
    configuration["gradient_isolation_passed"] = True
    atomic_json(OUTPUT_ROOT / "implementation_config.json", configuration)
    optimizer = torch.optim.AdamW(
        [
            {"params": list(model.encoder.parameters()), "lr": ENCODER_LR},
            {"params": list(model.prediction_head_parameters()), "lr": HEAD_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    torch.cuda.reset_peak_memory_stats(device)
    initial = evaluate(model, criterion, batches)
    started = time.monotonic()
    log_rows = [flatten_log(0, 0.0, initial)]
    gradient_rows = []
    actual_steps = 0
    early_stopped = False
    for step in range(1, MAX_STEPS + 1):
        actual_steps = step
        batch = batches[(step - 1) % len(batches)]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(forward_model(model, batch), batch)
        if not bool(torch.isfinite(loss.total_loss)):
            raise FloatingPointError(f"non-finite tiny loss at step {step}")
        loss.total_loss.backward()
        snapshot = {"step": step, "groups": gradient_snapshot(model)}
        if not all(
            row["finite"]
            for row in snapshot["groups"].values()
            if row["gradient_tensor_count"] > 0
        ):
            raise FloatingPointError(f"non-finite tiny gradient at step {step}")
        if step <= len(batches) or step % LOG_INTERVAL == 0:
            gradient_rows.append(snapshot)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), MAX_GRAD_NORM, error_if_nonfinite=True
        )
        optimizer.step()
        if step % LOG_INTERVAL == 0 or step == MAX_STEPS:
            current = evaluate(model, criterion, batches)
            log_rows.append(flatten_log(step, time.monotonic() - started, current))
            atomic_csv(OUTPUT_ROOT / "tiny_overfit_training_log.csv", log_rows)
            if step >= MIN_STEPS and success_assessment(
                initial,
                current,
                gradients,
                q_value_connectivity(model),
            )["passed"]:
                early_stopped = True
                break
    final = evaluate(model, criterion, batches)
    elapsed = time.monotonic() - started
    connectivity = q_value_connectivity(model)
    decoder = {
        "decoder_id": DECODER_ID,
        "decoder_version": DECODER_VERSION,
        "decoder_implementation_changed": False,
        "decoder_facing_interface": {
            "initial_logits": "[4]",
            "main_event_logits": "[I,6,5]",
            "main_timing_predictions": "[I,6]",
            "terminal_event_logits": "[4,5]",
            "terminal_timing_predictions": "[4]",
            "state_logits_passed_to_decoder": False,
        },
        "interface_unit_test_passed": correctness["passed"],
        "q_to_event_connectivity": connectivity,
        "same_state_pre_decoder_diagnostic": final["same_state_diagnostic"],
        "partial_window_decoder_execution_count": 0,
        "candidate_midi_generation_count": 0,
    }
    assessment = success_assessment(initial, final, gradients, connectivity)
    metrics = {
        "status": "completed",
        "optimizer_steps": actual_steps,
        "maximum_optimizer_steps": MAX_STEPS,
        "early_stopped_on_all_gates": early_stopped,
        "elapsed_seconds": elapsed,
        "initial": initial,
        "final": final,
        "success_assessment": assessment,
        "all_forward_loss_gradient_values_finite": True,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "full_training_started": False,
        "validation_model_inference_count": 0,
        "asap_test_access_count": 0,
        "repedal_execution_count": 0,
        "candidate_midi_generation_count": 0,
    }
    transition_payload = {
        "required_directions": list(REQUIRED_TRANSITIONS),
        "initial": initial["transition_coverage"],
        "final": final["transition_coverage"],
        "all_required_present": all(
            final["transition_coverage"][name]["target_count"] > 0
            for name in REQUIRED_TRANSITIONS
        ),
        "all_required_memorized": assessment["checks"][
            "required_transition_directions_memorized"
        ],
        "same_state_diagnostic": final["same_state_diagnostic"],
    }
    atomic_json(OUTPUT_ROOT / "tiny_overfit_metrics.json", metrics)
    atomic_json(
        OUTPUT_ROOT / "tiny_overfit_transition_coverage.json", transition_payload
    )
    atomic_json(OUTPUT_ROOT / "tiny_overfit_decoder_diagnostics.json", decoder)
    atomic_json(
        OUTPUT_ROOT / "tiny_overfit_gradient_diagnostics.json",
        {"snapshots": gradient_rows, "all_recorded_gradients_finite": True},
    )
    configuration["status"] = "completed"
    configuration["optimizer_steps"] = actual_steps
    configuration["early_stopped_on_all_gates"] = early_stopped
    configuration["elapsed_seconds"] = elapsed
    configuration["final_verdict"] = assessment["verdict"]
    atomic_json(OUTPUT_ROOT / "implementation_config.json", configuration)
    write_reports(configuration, accounting, gradients, metrics, decoder)
    print(
        json.dumps(
            {
                "verdict": assessment["verdict"],
                "state_accuracy": final["state"]["accuracy"],
                "main_event_accuracy": final["main_event"]["accuracy"],
                "main_timing_mae": final["main_timing"]["mae"],
                "elapsed_seconds": elapsed,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
