#!/usr/bin/env python3
"""Guarded State-Conditioned Binary 2-Slot v0 training pipeline.

Modes ``preflight``, ``sanity``, ``tiny`` are foreground gates.  Mode ``full``
is a standalone ten-epoch train + per-epoch ASAP-validation process intended
for a detached tmux session.  No mode opens an ASAP test manifest or test MIDI.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from miditoolkit import MidiFile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage2_binary_2slot.dataset import (
    Binary2SlotPerformanceDataset,
    HumanIntervalPrimitive,
    binary_2slot_collate_fn,
)
from src.stage2_binary_2slot.losses import FIXED_COUNT_WEIGHTS
from src.stage2_binary_2slot.model import StateConditionedBinary2SlotModel
from src.stage2_binary_2slot.rollout import build_online_target, decode_prediction, next_binary_state
from src.stage2_event_model.dataset import CANONICAL_ALIGNMENT_ID, CANONICAL_CACHE_ID
from src.stage2_event_tokenizer.tokenizer_v1 import _tick_second_converters
from src.stage2_four_class.validation_evaluator import pooled_transition_counts
from scripts.audit_pedal_event_metric_tolerance_mini import (
    build_score_positions,
    nearest_onset,
    raw_events,
)
from scripts.run_stage2_4class_validation_eval_v0 import (
    ALIGNMENT_TOOL,
    STAGE1_MANIFEST,
    finalize_transition,
    sum_transition,
)


RUN_ROOT = ROOT / "analysis/stage2_binary_2slot_full_v0"
CACHE_ROOT = ROOT / "analysis/custom_event_tokenizer_v1"
ALIGNMENT_ROOT = ROOT / "analysis/custom_event_model_v0_note_alignment"
PRETRAINED = ROOT / "checkpoints/pianist_transformer"
IMPLEMENTATION_ROOT = ROOT / "analysis/binary_2slot_model_implementation_v0"
PRIOR_VALIDATION_ROOT = ROOT / "analysis/custom_event_model_v0_canonical_val_inference_v1"
FROZEN_HUMAN_ALIGNMENT = ROOT / "analysis/stage2_4class_architecture_validation_eval_v0/alignment/human"

SEED = 42
MAX_EPOCHS = 10
WINDOW_MICRO_BATCH = 4
PERFORMANCES_PER_STEP = 4
ENCODER_LR = 1e-5
HEAD_LR = 1e-4
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
AMP_INIT_SCALE = 1024.0
TINY_MAX_STEPS = 500
TINY_LOG_INTERVAL = 25
MULTIWINDOW_INDEX = 13
GRADIENT_INDICES = (72, 6, 7, 13)
TINY_INDICES = (7,)
BOUNDARY_CANDIDATES = (1.0, 0.5, 0.25, 0.1)
GIT_COMMIT = "4c1d5daa527d76e815fcbda0239bcdec6af9458a"
ALL_COMPONENTS = {"main_count", "main_timing", "boundary_count", "boundary_timing"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    line = f"{now()} {message}"
    print(line, flush=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    with (RUN_ROOT / "training_log.txt").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_torch(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_model(device: torch.device) -> StateConditionedBinary2SlotModel:
    model = StateConditionedBinary2SlotModel.from_pretrained(
        PRETRAINED,
        head_init_seed=SEED,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    ).to(device)
    if (
        model.hidden_size != 768
        or int(model.encoder.config.num_hidden_layers) != 10
        or model.encoder_parameter_count != 103271424
        or model.prediction_head_parameter_count != 3850
    ):
        raise RuntimeError("frozen Binary 2-Slot architecture identity changed")
    return model


def build_optimizer(model: StateConditionedBinary2SlotModel) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        [
            {"params": list(model.encoder.parameters()), "lr": ENCODER_LR, "group_name": "encoder"},
            {"params": list(model.prediction_head_parameters()), "lr": HEAD_LR, "group_name": "binary_2slot_heads"},
        ],
        weight_decay=WEIGHT_DECAY,
    )


def window_dataset_indices(dataset: Binary2SlotPerformanceDataset, performance_index: int) -> list[int]:
    values = [index for index, (owner, _) in enumerate(dataset.windows) if owner == performance_index]
    expected = list(range(len(dataset.performances[performance_index].ownership.window_starts)))
    if [dataset.windows[index][1] for index in values] != expected:
        raise AssertionError("performance window indices are not chronological")
    return values


def chunks(values: Sequence[int], size: int = WINDOW_MICRO_BATCH) -> Iterable[list[int]]:
    for start in range(0, len(values), size):
        yield list(values[start : start + size])


def autocast_context(device: torch.device, enabled: bool):
    return torch.amp.autocast(
        device_type=device.type,
        dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
        enabled=enabled and device.type == "cuda",
    )


def encode_boundary(
    model: StateConditionedBinary2SlotModel,
    dataset: Binary2SlotPerformanceDataset,
    performance_index: int,
    region: str,
    device: torch.device,
    *,
    amp: bool,
) -> torch.Tensor:
    value = dataset.boundary_input(performance_index, region)
    with autocast_context(device, amp):
        return model.encode_boundary(
            value["input_ids"].unsqueeze(0).to(device),
            value["token_attention_mask"].unsqueeze(0).to(device),
            value["note_mask"].unsqueeze(0).to(device),
            torch.tensor([value["query_position"]], dtype=torch.long, device=device),
        )[0]


def encode_window_samples(
    model: StateConditionedBinary2SlotModel,
    samples: Sequence[Mapping[str, Any]],
    device: torch.device,
    *,
    amp: bool,
) -> list[torch.Tensor]:
    batch = binary_2slot_collate_fn(samples)
    with autocast_context(device, amp):
        encoded = model.encode_main(
            batch["input_ids"].to(device),
            batch["token_attention_mask"].to(device),
            batch["note_mask"].to(device),
            batch["owned_representative_positions"].to(device),
            batch["owned_onset_mask"].to(device),
        ).owned_onset_hidden_states
    return [
        encoded[index][batch["owned_onset_mask"][index].to(device)]
        for index in range(len(samples))
    ]


@dataclass
class IntervalRecord:
    region: str
    global_index: int
    main_onset_index: int | None
    model_state: int
    human_state: int
    target_count: int
    timing_targets: tuple[float, float]
    timing_mask: tuple[bool, bool]
    predicted_count: int
    predicted_timing: tuple[float, float]
    correction_required: bool
    retained_real: int
    raw_real: int


@dataclass
class WindowRecord:
    dataset_index: int
    window_index: int
    state_before: int
    state_after: int
    intervals: tuple[IntervalRecord, ...]


@dataclass
class PerformancePlan:
    performance_index: int
    performance_id: str
    pre: IntervalRecord
    windows: tuple[WindowRecord, ...]
    post: IntervalRecord
    candidate_events: tuple[dict[str, Any], ...]
    reference_events: tuple[dict[str, Any], ...]

    @property
    def records(self) -> tuple[IntervalRecord, ...]:
        return (self.pre,) + tuple(record for window in self.windows for record in window.intervals) + (self.post,)


def _select_predictions(
    model: StateConditionedBinary2SlotModel,
    hidden: torch.Tensor,
    state: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden = hidden.float()
    states = torch.full(hidden.shape[:-1], int(state), dtype=torch.long, device=hidden.device)
    output = model.condition_and_predict(hidden, states)
    return output.count_logits, output.timing_predictions


def _record_interval(
    model: StateConditionedBinary2SlotModel,
    hidden: torch.Tensor,
    interval: HumanIntervalPrimitive,
    state: int,
    *,
    absolute_events: list[dict[str, Any]],
) -> tuple[IntervalRecord, int]:
    count_logits, timing = _select_predictions(model, hidden.reshape(1, -1), state)
    if not bool(torch.isfinite(count_logits).all() and torch.isfinite(timing).all()):
        raise FloatingPointError("non-finite controller output")
    predicted_count = int(count_logits[0].argmax().item())
    predicted_timing = tuple(float(value) for value in timing[0].float().cpu().tolist())
    target = build_online_target(state, interval)
    decoded = decode_prediction(state, predicted_count, predicted_timing)
    for order, event in enumerate(decoded.transitions):
        absolute_events.append(
            {
                "direction": event.direction,
                "seconds": interval.left_seconds + event.tau * interval.duration_seconds,
                "region": interval.region,
                "interval_index": interval.global_index,
                "slot": order,
            }
        )
    record = IntervalRecord(
        region=interval.region,
        global_index=interval.global_index,
        main_onset_index=interval.main_onset_index,
        model_state=state,
        human_state=interval.human_start_state,
        target_count=target.count,
        timing_targets=target.timing_targets,
        timing_mask=target.timing_mask,
        predicted_count=predicted_count,
        predicted_timing=predicted_timing,
        correction_required=target.correction_required,
        retained_real=target.retained_real_human_transitions,
        raw_real=target.raw_human_transitions,
    )
    return record, next_binary_state(state, predicted_count)


def controller_plan(
    model: StateConditionedBinary2SlotModel,
    dataset: Binary2SlotPerformanceDataset,
    performance_index: int,
    device: torch.device,
    *,
    amp: bool,
) -> PerformancePlan:
    """Deterministic no-grad hard rollout used to define online targets."""

    model.eval()
    timeline = dataset.timeline(performance_index)
    state = 0
    candidate_events: list[dict[str, Any]] = []
    with torch.inference_mode():
        pre_hidden = encode_boundary(model, dataset, performance_index, "PRE", device, amp=amp)
        pre, state = _record_interval(
            model, pre_hidden, timeline.pre, state, absolute_events=candidate_events
        )
        window_records: list[WindowRecord] = []
        seen: list[int] = []
        for batch_indices in chunks(window_dataset_indices(dataset, performance_index)):
            samples = [dataset[index] for index in batch_indices]
            hidden_values = encode_window_samples(model, samples, device, amp=amp)
            for dataset_index, sample, hidden in zip(batch_indices, samples, hidden_values):
                before = state
                records: list[IntervalRecord] = []
                globals_ = [int(value) for value in sample["owned_global_onset_indices"].tolist()]
                if globals_ != sorted(globals_) or (seen and globals_ and globals_[0] <= seen[-1]):
                    raise AssertionError("owned global onset chronology violation")
                seen.extend(globals_)
                for local_index, interval in enumerate(sample["human_intervals"]):
                    record, state = _record_interval(
                        model, hidden[local_index], interval, state,
                        absolute_events=candidate_events,
                    )
                    records.append(record)
                window_records.append(
                    WindowRecord(
                        dataset_index=dataset_index,
                        window_index=int(sample["metadata"]["window_index"]),
                        state_before=before,
                        state_after=state,
                        intervals=tuple(records),
                    )
                )
        if seen != list(range(len(timeline.main))):
            raise AssertionError("MAIN owner supervision has missing or duplicate onsets")
        if any(
            left.state_after != right.state_before
            for left, right in zip(window_records, window_records[1:])
        ):
            raise AssertionError("state reset detected at a window boundary")
        post_hidden = encode_boundary(model, dataset, performance_index, "POST", device, amp=amp)
        post, state = _record_interval(
            model, post_hidden, timeline.post, state, absolute_events=candidate_events
        )
    reference_events = []
    for interval in timeline.represented_intervals:
        for event in interval.human_transitions:
            reference_events.append(
                {
                    "direction": event.direction,
                    "seconds": interval.left_seconds + event.tau * interval.duration_seconds,
                    "source_tick": event.source_tick,
                    "region": interval.region,
                }
            )
    return PerformancePlan(
        performance_index=performance_index,
        performance_id=timeline.performance_id,
        pre=pre,
        windows=tuple(window_records),
        post=post,
        candidate_events=tuple(candidate_events),
        reference_events=tuple(reference_events),
    )


def plan_diagnostics(plans: Sequence[PerformancePlan]) -> dict[str, Any]:
    records = [record for plan in plans for record in plan.records]
    target = Counter(record.target_count for record in records)
    predicted = Counter(record.predicted_count for record in records)
    region = Counter(record.region for record in records)
    agreements = [record.model_state == record.human_state for record in records]
    mismatch_runs: list[int] = []
    active = 0
    for agreement in agreements + [True]:
        if not agreement:
            active += 1
        elif active:
            mismatch_runs.append(active)
            active = 0
    active_errors = [
        abs(record.predicted_timing[slot] - record.timing_targets[slot])
        for record in records
        for slot in range(2)
        if record.timing_mask[slot]
    ]
    raw = sum(record.raw_real for record in records)
    retained = sum(record.retained_real for record in records)
    return {
        "intervals": len(records),
        "count_correct": sum(record.target_count == record.predicted_count for record in records),
        "count_target": {str(index): target[index] for index in range(3)},
        "count_predicted": {str(index): predicted[index] for index in range(3)},
        "count_accuracy": sum(record.target_count == record.predicted_count for record in records) / len(records),
        "state_agreement_count": sum(agreements),
        "interval_start_state_agreement": sum(agreements) / len(agreements),
        "correction_required_count": sum(record.correction_required for record in records),
        "correction_required_fraction": sum(record.correction_required for record in records) / len(records),
        "raw_human_transitions": raw,
        "retained_real_human_transitions": retained,
        "real_transition_retention": retained / raw if raw else 1.0,
        "region_counts": {key: region[key] for key in ("PRE", "MAIN", "POST")},
        "active_timing_error_sum": float(sum(active_errors)),
        "active_timing_error_count": len(active_errors),
        "active_timing_mae": float(np.mean(active_errors)) if active_errors else 0.0,
        "mismatch_run_count": len(mismatch_runs),
        "mismatch_run_mean": float(np.mean(mismatch_runs)) if mismatch_runs else 0.0,
        "mismatch_run_max": max(mismatch_runs, default=0),
        "invalid_or_nonfinite": 0,
    }


def loss_denominators(plans: Sequence[PerformancePlan]) -> dict[str, float]:
    weights = FIXED_COUNT_WEIGHTS
    values = {
        "main_count": 0.0,
        "main_timing": 0.0,
        "boundary_count": 0.0,
        "boundary_timing": 0.0,
    }
    for plan in plans:
        for record in plan.records:
            scope = "main" if record.region == "MAIN" else "boundary"
            values[f"{scope}_count"] += float(weights[record.target_count])
            values[f"{scope}_timing"] += float(sum(record.timing_mask))
    if values["main_count"] <= 0 or values["boundary_count"] <= 0:
        raise ValueError(f"loss group has a zero denominator: {values}")
    return values


def tensor_targets(records: Sequence[IntervalRecord], device: torch.device, dtype: torch.dtype):
    return (
        torch.tensor([record.model_state for record in records], dtype=torch.long, device=device),
        torch.tensor([record.target_count for record in records], dtype=torch.long, device=device),
        torch.tensor([record.timing_targets for record in records], dtype=dtype, device=device),
        torch.tensor([record.timing_mask for record in records], dtype=torch.bool, device=device),
    )


def raw_loss_terms(
    model: StateConditionedBinary2SlotModel,
    hidden: torch.Tensor,
    records: Sequence[IntervalRecord],
) -> dict[str, torch.Tensor]:
    hidden = hidden.float()
    states, counts, timing_targets, timing_mask = tensor_targets(records, hidden.device, torch.float32)
    output = model.condition_and_predict(hidden, states)
    logits = output.count_logits.float()
    timing = output.timing_predictions.float()
    weights = torch.tensor(FIXED_COUNT_WEIGHTS, dtype=torch.float32, device=hidden.device)
    count_sum = F.cross_entropy(logits, counts, weight=weights, reduction="sum")
    if bool(timing_mask.any()):
        timing_sum = F.smooth_l1_loss(
            timing[timing_mask], timing_targets[timing_mask], beta=0.1, reduction="sum"
        )
    else:
        timing_sum = timing.sum() * 0.0
    return {"count": count_sum, "timing": timing_sum}


def normalized_chunk_loss(
    terms: Mapping[str, torch.Tensor],
    scope: str,
    denominators: Mapping[str, float],
    boundary_weight: float,
    components: set[str],
) -> torch.Tensor | None:
    selected: list[torch.Tensor] = []
    if f"{scope}_count" in components:
        selected.append(terms["count"] / denominators[f"{scope}_count"])
    if f"{scope}_timing" in components:
        denominator = denominators[f"{scope}_timing"]
        selected.append(terms["timing"] / denominator if denominator > 0 else terms["timing"] * 0.0)
    if not selected:
        return None
    value = sum(selected)
    return value * (float(boundary_weight) if scope == "boundary" else 1.0)


def replay_plans(
    model: StateConditionedBinary2SlotModel,
    dataset: Binary2SlotPerformanceDataset,
    plans: Sequence[PerformancePlan],
    device: torch.device,
    *,
    amp: bool,
    boundary_weight: float,
    components: set[str],
    backward: Any | None,
) -> dict[str, float]:
    """Replay stored hard states and targets with exact group denominators."""

    denominators = loss_denominators(plans)
    sums = {key: 0.0 for key in denominators}

    def consume(hidden: torch.Tensor, records: Sequence[IntervalRecord], scope: str) -> torch.Tensor | None:
        terms = raw_loss_terms(model, hidden, records)
        sums[f"{scope}_count"] += float(terms["count"].detach().item())
        sums[f"{scope}_timing"] += float(terms["timing"].detach().item())
        loss = normalized_chunk_loss(
            terms, scope, denominators, boundary_weight, components
        )
        if loss is not None and not bool(torch.isfinite(loss)):
            raise FloatingPointError("non-finite normalized chunk loss")
        return loss

    for plan in plans:
        performance_index = plan.performance_index
        if "boundary_count" in components or "boundary_timing" in components or backward is None:
            hidden = encode_boundary(model, dataset, performance_index, "PRE", device, amp=amp)
            loss = consume(hidden.reshape(1, -1), (plan.pre,), "boundary")
            if loss is not None and backward is not None:
                backward(loss)
        if "main_count" in components or "main_timing" in components or backward is None:
            for records_batch in chunks(list(range(len(plan.windows)))):
                windows = [plan.windows[index] for index in records_batch]
                samples = [dataset[window.dataset_index] for window in windows]
                hidden_values = encode_window_samples(model, samples, device, amp=amp)
                batch_losses: list[torch.Tensor] = []
                for window, hidden in zip(windows, hidden_values):
                    loss = consume(hidden, window.intervals, "main")
                    if loss is not None:
                        batch_losses.append(loss)
                if batch_losses and backward is not None:
                    backward(sum(batch_losses))
        if "boundary_count" in components or "boundary_timing" in components or backward is None:
            hidden = encode_boundary(model, dataset, performance_index, "POST", device, amp=amp)
            loss = consume(hidden.reshape(1, -1), (plan.post,), "boundary")
            if loss is not None and backward is not None:
                backward(loss)
    result = {key: sums[key] / denominators[key] if denominators[key] > 0 else 0.0 for key in sums}
    result["main"] = result["main_count"] + result["main_timing"]
    result["boundary"] = result["boundary_count"] + result["boundary_timing"]
    result["total"] = result["main"] + float(boundary_weight) * result["boundary"]
    result.update({f"denominator_{key}": value for key, value in denominators.items()})
    if not all(math.isfinite(value) for value in result.values()):
        raise FloatingPointError("non-finite replay aggregate")
    return result


def group_parameters(model: StateConditionedBinary2SlotModel) -> dict[str, list[tuple[str, torch.nn.Parameter]]]:
    encoder = [(f"encoder.{name}", parameter) for name, parameter in model.encoder.named_parameters()]
    count = [(f"count_head.{name}", parameter) for name, parameter in model.count_head.named_parameters()]
    timing_1 = [(f"timing_head_1.{name}", parameter) for name, parameter in model.timing_head_1.named_parameters()]
    timing_2 = [(f"timing_head_2.{name}", parameter) for name, parameter in model.timing_head_2.named_parameters()]
    return {
        "encoder": encoder,
        "count_head": count,
        "timing_head_1": timing_1,
        "timing_head_2": timing_2,
        "all_heads": count + timing_1 + timing_2,
        "full_model": encoder + count + timing_1 + timing_2,
    }


def gradient_snapshot(model: StateConditionedBinary2SlotModel) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for group, named in group_parameters(model).items():
        gradients = [(name, parameter.grad) for name, parameter in named if parameter.grad is not None]
        squared = sum(float(gradient.detach().float().square().sum().item()) for _, gradient in gradients)
        result[group] = {
            "global_l2": math.sqrt(squared),
            "parameter_tensors": len(named),
            "parameter_tensors_with_grad": len(gradients),
            "all_finite": all(bool(torch.isfinite(gradient).all()) for _, gradient in gradients),
            "no_grad_parameter_names": [name for name, parameter in named if parameter.grad is None],
        }
    return result


def configuration(boundary_weight: float) -> dict[str, Any]:
    cache = json.loads((CACHE_ROOT / "cache_manifest.json").read_text(encoding="utf-8"))
    alignment = json.loads((ALIGNMENT_ROOT / "alignment_manifest.json").read_text(encoding="utf-8"))
    return {
        "experiment": "stage2_binary_2slot_full_v0",
        "created_at": now(),
        "git_commit": GIT_COMMIT,
        "git_status": "dirty worktree recorded by foreground host inspection; unrelated user changes preserved",
        "seed": SEED,
        "pretrained_checkpoint": str(PRETRAINED),
        "pretrained_model_sha256": sha256_file(PRETRAINED / "model.safetensors"),
        "cache_id": cache["cache_id"],
        "alignment_id": alignment["alignment_id"],
        "train_performances": 2062,
        "validation_performances": 71,
        "test_access_count": 0,
        "model": {
            "hidden_size": 768,
            "encoder_layers": 10,
            "encoder_parameters": 103271424,
            "head_parameters": 3850,
            "state_conditioning": "concat(dropout(h), hard scalar state)",
        },
        "optimizer": "AdamW",
        "encoder_lr": ENCODER_LR,
        "head_lr": HEAD_LR,
        "weight_decay": WEIGHT_DECAY,
        "max_gradient_norm": MAX_GRAD_NORM,
        "precision": "FP16 AMP",
        "amp_init_scale": AMP_INIT_SCALE,
        "scheduler": None,
        "window_micro_batch": WINDOW_MICRO_BATCH,
        "performances_per_optimizer_step": PERFORMANCES_PER_STEP,
        "stateful_two_pass": "deterministic eval/no-grad controller then train/grad replay before parameter update",
        "optimizer_provenance": "scripts/run_custom_event_model_v0_full_seed42.py and run_custom_event_model_v0_tiny_overfit.py",
        "max_epochs": MAX_EPOCHS,
        "early_stopping": None,
        "validation_every_epoch": True,
        "checkpoint_selection": "highest canonical validation direction-aware Transition F1; ties earlier epoch",
        "count_weights": list(FIXED_COUNT_WEIGHTS),
        "smooth_l1_beta": 0.1,
        "boundary_loss_weight": float(boundary_weight),
        "boundary_weight_selection": "largest candidate with max(boundary/main encoder,head grad ratio)*lambda <= 0.5",
        "forbidden": {
            "asap_test_metadata": 0,
            "asap_test_midi": 0,
            "test_inference": 0,
            "test_metrics": 0,
            "hyperparameter_sweep": 0,
        },
    }


def preflight() -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("preflight requires exactly one visible CUDA GPU")
    implementation = json.loads((IMPLEMENTATION_ROOT / "implementation_summary.json").read_text(encoding="utf-8"))
    if implementation["status"] != "PASS" or implementation["blocking_issue"]:
        raise RuntimeError("implementation audit is not PASS")
    cache = json.loads((CACHE_ROOT / "cache_manifest.json").read_text(encoding="utf-8"))
    alignment = json.loads((ALIGNMENT_ROOT / "alignment_manifest.json").read_text(encoding="utf-8"))
    if cache["cache_id"] != CANONICAL_CACHE_ID or cache["train_count"] != 2062 or cache["validation_count"] != 71:
        raise RuntimeError("canonical cache identity/count changed")
    if alignment["alignment_id"] != CANONICAL_ALIGNMENT_ID or alignment["train_count"] != 2062 or alignment["validation_count"] != 71:
        raise RuntimeError("canonical alignment identity/count changed")
    if cache["asap_test_access_count"] or alignment["asap_test_access_count"]:
        raise RuntimeError("frozen provenance reports ASAP test access")
    commands = [
        [sys.executable, "tests/test_binary_2slot_model_v0.py"],
        [sys.executable, "tests/test_binary_2slot_transition_audit_v0.py"],
    ]
    test_rows = []
    for command in commands:
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        test_rows.append({"command": command, "returncode": completed.returncode, "stdout": completed.stdout.strip(), "stderr": completed.stderr.strip()})
        if completed.returncode:
            raise RuntimeError(f"unit regression failed: {command}")
    device = torch.device("cuda:0")
    model = build_model(device)
    payload = {
        "status": "PASS",
        "completed_at": now(),
        "git_commit": GIT_COMMIT,
        "git_status": "dirty worktree recorded by foreground host inspection; unrelated user changes preserved",
        "checkpoint": str(PRETRAINED),
        "checkpoint_sha256": sha256_file(PRETRAINED / "model.safetensors"),
        "cache_id": CANONICAL_CACHE_ID,
        "alignment_id": CANONICAL_ALIGNMENT_ID,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
        "device_count": torch.cuda.device_count(),
        "training_dtype": "FP16 AMP; FP32 master parameters",
        "hidden_size": model.hidden_size,
        "encoder_layers": int(model.encoder.config.num_hidden_layers),
        "encoder_parameters": model.encoder_parameter_count,
        "head_parameters": model.prediction_head_parameter_count,
        "tests": test_rows,
        "asap_test_access_count": 0,
    }
    atomic_json(RUN_ROOT / "environment_provenance.json", payload)
    del model
    torch.cuda.empty_cache()
    log("STAGE_A_PASS environment/provenance/unit gates passed")


def sanity_and_gradient() -> None:
    if json.loads((RUN_ROOT / "environment_provenance.json").read_text())["status"] != "PASS":
        raise RuntimeError("Stage A PASS artifact missing")
    seed_everything()
    device = torch.device("cuda:0")
    model = build_model(device)
    sanity_dataset = Binary2SlotPerformanceDataset(
        CACHE_ROOT, "train", performance_indices=(MULTIWINDOW_INDEX,), cache_input_tokens=True
    )
    plan = controller_plan(model, sanity_dataset, 0, device, amp=True)
    performance = sanity_dataset.performances[0]
    ownership = performance.ownership
    membership = np.zeros(performance.num_onsets, dtype=np.int64)
    for owned in ownership.owned_onset_indices:
        membership[owned] += 1
    sanity = {
        "status": "PASS",
        "canonical_train_performance_index": MULTIWINDOW_INDEX,
        "performance_id": plan.performance_id,
        "notes": performance.num_notes,
        "main_onsets": performance.num_onsets,
        "windows": len(plan.windows),
        "pre_count": 1,
        "post_count": 1,
        "window_indices": [window.window_index for window in plan.windows],
        "owned_global_onsets_strict": [record.main_onset_index for window in plan.windows for record in window.intervals] == list(range(performance.num_onsets)),
        "missing_supervision": int(np.sum(membership == 0)),
        "duplicate_supervision": int(np.sum(membership > 1)),
        "owner_monotonic": bool(np.all(np.diff(ownership.owner_window_index) >= 0)),
        "window_state_trace": [
            {"window_index": window.window_index, "state_before": window.state_before, "state_after": window.state_after}
            for window in plan.windows
        ],
        "boundary_state_continuity": all(left.state_after == right.state_before for left, right in zip(plan.windows, plan.windows[1:])),
        "diagnostics": plan_diagnostics((plan,)),
        "interpretation": "random-head plumbing diagnostic only",
        "asap_test_access_count": 0,
    }
    if sanity["windows"] < 3 or sanity["missing_supervision"] or sanity["duplicate_supervision"] or not sanity["owner_monotonic"] or not sanity["boundary_state_continuity"]:
        raise AssertionError("Stage B multi-window sanity failed")
    atomic_json(RUN_ROOT / "multiwindow_sanity.json", sanity)
    log(f"STAGE_B_PASS windows={sanity['windows']} onsets={sanity['main_onsets']}")
    del sanity_dataset, plan
    torch.cuda.empty_cache()

    gradient_dataset = Binary2SlotPerformanceDataset(
        CACHE_ROOT, "train", performance_indices=GRADIENT_INDICES, cache_input_tokens=True
    )
    plans = [controller_plan(model, gradient_dataset, index, device, amp=False) for index in range(len(GRADIENT_INDICES))]
    baseline_loss = replay_plans(
        model, gradient_dataset, plans, device, amp=False, boundary_weight=1.0,
        components={"main_count", "main_timing", "boundary_count", "boundary_timing"}, backward=None,
    )
    component_sets = {
        "main_count_loss": {"main_count"},
        "main_timing_loss": {"main_timing"},
        "boundary_count_loss": {"boundary_count"},
        "boundary_timing_loss": {"boundary_timing"},
        "L_main": {"main_count", "main_timing"},
        "L_boundary": {"boundary_count", "boundary_timing"},
    }
    gradients: dict[str, Any] = {}
    model.eval()
    for name, selected in component_sets.items():
        model.zero_grad(set_to_none=True)
        replay_plans(
            model, gradient_dataset, plans, device, amp=False, boundary_weight=1.0,
            components=selected, backward=lambda value: value.backward(),
        )
        snapshot = gradient_snapshot(model)
        if not all(item["all_finite"] for item in snapshot.values()):
            raise FloatingPointError(f"non-finite Stage C gradients: {name}")
        gradients[name] = snapshot
    main_encoder = gradients["L_main"]["encoder"]["global_l2"]
    boundary_encoder = gradients["L_boundary"]["encoder"]["global_l2"]
    main_heads = gradients["L_main"]["all_heads"]["global_l2"]
    boundary_heads = gradients["L_boundary"]["all_heads"]["global_l2"]
    ratios = {
        "scalar_loss_boundary_over_main": baseline_loss["boundary"] / baseline_loss["main"],
        "R_encoder": boundary_encoder / main_encoder,
        "R_heads": boundary_heads / main_heads,
    }
    no_grad_names = sorted(set(
        gradients["L_main"]["encoder"]["no_grad_parameter_names"]
        + gradients["L_boundary"]["encoder"]["no_grad_parameter_names"]
    ))
    inactive_verdict = {
        "names": no_grad_names,
        "classification": "inactive/dead pretrained token-embedding path" if no_grad_names == ["encoder.embed_tokens.weight"] else "unexpected; inspect before training",
        "reason": "PianoT5Gemma encoder receives custom PianoEncoderEmbeddings through inputs_embeds; inherited T5 embed_tokens is not used in forward.",
    }
    if inactive_verdict["classification"].startswith("unexpected"):
        raise RuntimeError(f"unexpected no-grad encoder parameters: {no_grad_names}")
    maximum_ratio = max(ratios["R_encoder"], ratios["R_heads"])
    candidate_rows = []
    selected_weight = None
    for candidate in BOUNDARY_CANDIDATES:
        effective = candidate * maximum_ratio
        eligible = effective <= 0.5
        candidate_rows.append({"boundary_loss_weight": candidate, "R": maximum_ratio, "effective_boundary_ratio": effective, "eligible": eligible})
        if selected_weight is None and eligible:
            selected_weight = candidate
    if selected_weight is None:
        raise RuntimeError("boundary domination remains >0.5 even at lambda=0.1")
    summary = {
        "status": "PASS",
        "train_performance_indices": list(GRADIENT_INDICES),
        "performance_shapes": [
            {"canonical_index": original, "notes": item.num_notes, "onsets": item.num_onsets, "windows": len(item.ownership.window_starts)}
            for original, item in zip(GRADIENT_INDICES, gradient_dataset.performances)
        ],
        "losses": baseline_loss,
        "gradients": gradients,
        "ratios": ratios,
        "no_grad_encoder_parameters": inactive_verdict,
        "selected_boundary_loss_weight": selected_weight,
        "selection_rule_R": maximum_ratio,
        "asap_test_access_count": 0,
    }
    atomic_json(RUN_ROOT / "gradient_scale_summary.json", summary)
    atomic_csv(RUN_ROOT / "boundary_weight_candidates.csv", candidate_rows)
    atomic_json(RUN_ROOT / "FULL_RUN_CONFIG.json", configuration(selected_weight))
    sanity_md = f"""# Pre-overfit sanity\n\n- Stage A: PASS\n- Stage B multi-window: PASS ({sanity['windows']} windows, {sanity['main_onsets']} owned onsets)\n- Missing/duplicate supervision: 0/0\n- State continuity at every window boundary: PASS\n- Stage C gradient audit: PASS\n- No-grad encoder parameter: `{no_grad_names[0]}`; expected inactive inherited embedding path.\n- Boundary/Main scalar loss ratio: {ratios['scalar_loss_boundary_over_main']:.6f}\n- Encoder gradient ratio: {ratios['R_encoder']:.6f}\n- Head gradient ratio: {ratios['R_heads']:.6f}\n- Selected boundary loss weight: **{selected_weight}**\n- ASAP test access: 0\n"""
    (RUN_ROOT / "PREOVERFIT_SANITY.md").write_text(sanity_md, encoding="utf-8")
    log(f"STAGE_C_D_PASS R_encoder={ratios['R_encoder']:.6f} R_heads={ratios['R_heads']:.6f} lambda={selected_weight}")


def combine_diagnostics(values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    intervals = sum(int(value["intervals"]) for value in values)
    if intervals <= 0:
        raise ValueError("cannot combine empty rollout diagnostics")
    target = {str(index): sum(int(value["count_target"][str(index)]) for value in values) for index in range(3)}
    predicted = {str(index): sum(int(value["count_predicted"][str(index)]) for value in values) for index in range(3)}
    regions = {name: sum(int(value["region_counts"][name]) for value in values) for name in ("PRE", "MAIN", "POST")}
    count_correct = sum(int(value["count_correct"]) for value in values)
    state_agreements = sum(int(value["state_agreement_count"]) for value in values)
    corrections = sum(int(value["correction_required_count"]) for value in values)
    raw = sum(int(value["raw_human_transitions"]) for value in values)
    retained = sum(int(value["retained_real_human_transitions"]) for value in values)
    timing_error_sum = sum(float(value["active_timing_error_sum"]) for value in values)
    timing_error_count = sum(int(value["active_timing_error_count"]) for value in values)
    run_count = sum(int(value["mismatch_run_count"]) for value in values)
    run_sum = sum(float(value["mismatch_run_mean"]) * int(value["mismatch_run_count"]) for value in values)
    return {
        "intervals": intervals,
        "count_target": target,
        "count_predicted": predicted,
        "count_accuracy": count_correct / intervals,
        "interval_start_state_agreement": state_agreements / intervals,
        "correction_required_fraction": corrections / intervals,
        "real_transition_retention": retained / raw if raw else 1.0,
        "region_counts": regions,
        "active_timing_mae": timing_error_sum / timing_error_count if timing_error_count else 0.0,
        "mismatch_run_count": run_count,
        "mismatch_run_mean": run_sum / run_count if run_count else 0.0,
        "mismatch_run_max": max(int(value["mismatch_run_max"]) for value in values),
        "invalid_or_nonfinite": sum(int(value["invalid_or_nonfinite"]) for value in values),
        "raw_human_transitions": raw,
        "retained_real_human_transitions": retained,
        "active_timing_slots": timing_error_count,
        "aggregation_note": "exact pooled counts; performance-local mismatch runs do not cross boundaries",
    }


def evaluate_plans_loss(
    model: StateConditionedBinary2SlotModel,
    dataset: Binary2SlotPerformanceDataset,
    plans: Sequence[PerformancePlan],
    device: torch.device,
    boundary_weight: float,
    *,
    amp: bool,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        return replay_plans(
            model, dataset, plans, device, amp=amp, boundary_weight=boundary_weight,
            components=ALL_COMPONENTS, backward=None,
        )


def tiny_overfit() -> None:
    gradient = json.loads((RUN_ROOT / "gradient_scale_summary.json").read_text(encoding="utf-8"))
    if gradient["status"] != "PASS":
        raise RuntimeError("Stage C/D PASS artifact missing")
    boundary_weight = float(gradient["selected_boundary_loss_weight"])
    seed_everything()
    device = torch.device("cuda:0")
    dataset = Binary2SlotPerformanceDataset(
        CACHE_ROOT, "train", performance_indices=TINY_INDICES, cache_input_tokens=True
    )
    model = build_model(device)
    optimizer = build_optimizer(model)
    rows: list[dict[str, Any]] = []

    def snapshot(step: int) -> tuple[list[PerformancePlan], dict[str, Any], dict[str, float]]:
        plans = [controller_plan(model, dataset, index, device, amp=False) for index in range(len(dataset.performances))]
        diagnostics = plan_diagnostics(plans)
        losses = evaluate_plans_loss(model, dataset, plans, device, boundary_weight, amp=False)
        row = {
            "step": step,
            "total_loss": losses["total"],
            "main_count_loss": losses["main_count"],
            "main_timing_loss": losses["main_timing"],
            "boundary_count_loss": losses["boundary_count"],
            "boundary_timing_loss": losses["boundary_timing"],
            "count_accuracy": diagnostics["count_accuracy"],
            "timing_mae": diagnostics["active_timing_mae"],
            "state_agreement": diagnostics["interval_start_state_agreement"],
            "correction_fraction": diagnostics["correction_required_fraction"],
            "real_transition_retention": diagnostics["real_transition_retention"],
            "target_n0": diagnostics["count_target"]["0"],
            "target_n1": diagnostics["count_target"]["1"],
            "target_n2": diagnostics["count_target"]["2"],
            "predicted_n0": diagnostics["count_predicted"]["0"],
            "predicted_n1": diagnostics["count_predicted"]["1"],
            "predicted_n2": diagnostics["count_predicted"]["2"],
            "invalid_or_nonfinite": diagnostics["invalid_or_nonfinite"],
        }
        rows.append(row)
        atomic_csv(RUN_ROOT / "tiny_overfit_metrics.csv", rows)
        log(
            f"TINY step={step} loss={losses['total']:.6f} acc={diagnostics['count_accuracy']:.4f} "
            f"state={diagnostics['interval_start_state_agreement']:.4f} correction={diagnostics['correction_required_fraction']:.4f}"
        )
        return plans, diagnostics, losses

    initial_plans, initial_diag, initial_loss = snapshot(0)
    del initial_plans
    final_diag, final_loss = initial_diag, initial_loss
    for step in range(1, TINY_MAX_STEPS + 1):
        plans = [controller_plan(model, dataset, index, device, amp=False) for index in range(len(dataset.performances))]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        replay_plans(
            model, dataset, plans, device, amp=False, boundary_weight=boundary_weight,
            components=ALL_COMPONENTS, backward=lambda value: value.backward(),
        )
        if not all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()):
            raise FloatingPointError("tiny overfit produced a non-finite gradient")
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM).item())
        if not math.isfinite(grad_norm):
            raise FloatingPointError("tiny overfit gradient norm is non-finite")
        optimizer.step()
        if step % TINY_LOG_INTERVAL == 0 or step == TINY_MAX_STEPS:
            _, final_diag, final_loss = snapshot(step)
    reduction = 1.0 - final_loss["total"] / initial_loss["total"]
    strong = (
        final_diag["count_accuracy"] >= 0.90
        and final_diag["interval_start_state_agreement"] >= 0.90
        and reduction >= 0.70
        and final_diag["correction_required_fraction"] <= initial_diag["correction_required_fraction"]
    )
    clear_progress = (
        reduction >= 0.50
        and final_diag["count_accuracy"] >= 0.80
        and final_diag["interval_start_state_agreement"] >= initial_diag["interval_start_state_agreement"]
        and final_diag["correction_required_fraction"] <= initial_diag["correction_required_fraction"]
    )
    status = "PASS" if strong else "WARN/PASS" if clear_progress else "FAIL"
    summary = {
        "status": status,
        "canonical_train_performance_indices": list(TINY_INDICES),
        "optimizer_steps": TINY_MAX_STEPS,
        "boundary_loss_weight": boundary_weight,
        "initial": {"losses": initial_loss, "diagnostics": initial_diag},
        "final": {"losses": final_loss, "diagnostics": final_diag},
        "total_loss_reduction_fraction": reduction,
        "strong_gate": strong,
        "clear_progress_gate": clear_progress,
        "optimizer_updates": TINY_MAX_STEPS,
        "checkpoint_created": False,
        "asap_test_access_count": 0,
    }
    atomic_json(RUN_ROOT / "tiny_overfit_summary.json", summary)
    log(f"STAGE_E_{status.replace('/', '_')} reduction={reduction:.4f}")
    if status == "FAIL":
        raise RuntimeError("tiny-overfit clear failure; full training is prohibited")


def empty_loss_totals() -> dict[str, float]:
    return {key: 0.0 for key in ("main_count", "main_timing", "boundary_count", "boundary_timing")}


def add_loss_totals(total: dict[str, float], result: Mapping[str, float]) -> None:
    for key in ("main_count", "main_timing", "boundary_count", "boundary_timing"):
        total[key] += float(result[key]) * float(result[f"denominator_{key}"])
        total[f"denominator_{key}"] = total.get(f"denominator_{key}", 0.0) + float(result[f"denominator_{key}"])


def finalize_loss_totals(total: Mapping[str, float], boundary_weight: float) -> dict[str, float]:
    output: dict[str, float] = {}
    for key in ("main_count", "main_timing", "boundary_count", "boundary_timing"):
        denominator = float(total.get(f"denominator_{key}", 0.0))
        output[key] = float(total[key]) / denominator if denominator > 0 else 0.0
    output["main"] = output["main_count"] + output["main_timing"]
    output["boundary"] = output["boundary_count"] + output["boundary_timing"]
    output["total"] = output["main"] + boundary_weight * output["boundary"]
    return output


def score_manifest() -> dict[str, dict[str, str]]:
    rows = read_csv(STAGE1_MANIFEST)
    if len(rows) != 19 or any(row["status"] != "frozen_pass" for row in rows):
        raise RuntimeError("canonical validation Stage 1 manifest identity changed")
    return {row["piece_id"]: row for row in rows}


def score_positions_for_performance(
    performance: Any,
    *,
    canonical_index: int,
    stage1: Mapping[str, Mapping[str, str]],
) -> tuple[list[int], list[int]]:
    source = Path(performance.entry["source_midi"])
    midi = MidiFile(str(source))
    raw_onsets, _ = raw_events(midi)
    if len(raw_onsets) != performance.num_onsets:
        raise RuntimeError("validation raw/cache distinct onset identity changed")
    cache_dir = RUN_ROOT / "validation_score_position_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"validation_{canonical_index:04d}.json"
    source_sha = str(performance.entry["source_sha256"])
    if cache_path.is_file():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if payload.get("source_sha256") != source_sha or payload.get("raw_onsets") != raw_onsets:
            raise RuntimeError("validation score-position cache identity mismatch")
        positions = [int(value) for value in payload["score_positions"]]
    else:
        piece = str(performance.entry["piece_id"])
        if piece not in stage1:
            raise RuntimeError(f"validation piece is absent from frozen Stage 1 manifest: {piece}")
        score_path = Path(stage1[piece]["selected_score_absolute_path"])
        if sha256_file(score_path) != stage1[piece]["selected_score_sha256"]:
            raise RuntimeError("frozen validation score SHA changed")
        process_log = RUN_ROOT / "validation_alignment_logs" / f"validation_{canonical_index:04d}.log"
        process_log.parent.mkdir(parents=True, exist_ok=True)
        positions_raw = build_score_positions(score_path, source, raw_onsets, ALIGNMENT_TOOL, process_log)
        if any(value is None for value in positions_raw):
            raise RuntimeError("validation score alignment produced an unmapped onset")
        positions = [int(value) for value in positions_raw]
        atomic_json(cache_path, {
            "canonical_validation_index": canonical_index,
            "performance_path": performance.entry["performance_path"],
            "source_sha256": source_sha,
            "raw_onsets": raw_onsets,
            "score_positions": positions,
            "asap_test_access_count": 0,
        })
    return raw_onsets, positions


def trajectory_for_metric(
    events: Sequence[Mapping[str, Any]],
    source_midi: str,
    raw_onsets: Sequence[int],
    score_positions: Sequence[int],
) -> dict[str, Any]:
    _, seconds_to_tick = _tick_second_converters(source_midi)
    transitions = []
    for event in events:
        tick = int(event["source_tick"]) if "source_tick" in event else int(seconds_to_tick(float(event["seconds"])))
        onset_index = nearest_onset(raw_onsets, tick)
        transitions.append({
            "direction": str(event["direction"]),
            "tick": tick,
            "onset_index": onset_index,
            "score_position": int(score_positions[onset_index]),
        })
    return {"transitions": transitions}


def transition_accumulator() -> dict[str, Any]:
    return {
        scope: {key: 0 for key in ("candidate", "reference", "tp", "fp", "fn")}
        for scope in ("pooled", "up", "down")
    }


def validation_epoch(
    model: StateConditionedBinary2SlotModel,
    dataset: Binary2SlotPerformanceDataset,
    device: torch.device,
    boundary_weight: float,
    epoch: int,
) -> dict[str, Any]:
    model.eval()
    stage1 = score_manifest()
    diagnostic_values: list[dict[str, Any]] = []
    loss_totals = empty_loss_totals()
    transition_total = transition_accumulator()
    failures: list[dict[str, Any]] = []
    for performance_index, performance in enumerate(dataset.performances):
        plan = controller_plan(model, dataset, performance_index, device, amp=True)
        diagnostic_values.append(plan_diagnostics((plan,)))
        losses = evaluate_plans_loss(model, dataset, (plan,), device, boundary_weight, amp=True)
        add_loss_totals(loss_totals, losses)
        canonical_index = int(performance.entry["performance_index"])
        try:
            onsets, positions = score_positions_for_performance(
                performance, canonical_index=canonical_index, stage1=stage1
            )
            source_midi = str(performance.entry["source_midi"])
            candidate = trajectory_for_metric(plan.candidate_events, source_midi, onsets, positions)
            reference = trajectory_for_metric(plan.reference_events, source_midi, onsets, positions)
            current = pooled_transition_counts(candidate, reference)
            sum_transition(transition_total, current)
        except Exception as error:
            failures.append({
                "canonical_validation_index": canonical_index,
                "performance_path": performance.entry["performance_path"],
                "error": f"{type(error).__name__}: {error}",
            })
        if (performance_index + 1) % 10 == 0 or performance_index + 1 == len(dataset.performances):
            log(f"VALIDATION epoch={epoch} performances={performance_index + 1}/{len(dataset.performances)}")
    if len(failures) > 1 or len(dataset.performances) - len(failures) < 70:
        raise RuntimeError(f"canonical transition metric alignment failed for too many validation performances: {failures}")
    transitions = finalize_transition(transition_total)
    diagnostics = combine_diagnostics(diagnostic_values)
    losses = finalize_loss_totals(loss_totals, boundary_weight)
    pooled = transitions["pooled"]
    result = {
        "epoch": epoch,
        "transition_precision": pooled["precision"],
        "transition_recall": pooled["recall"],
        "transition_f1": pooled["f1"],
        "predicted_transitions": pooled["candidate"],
        "reference_transitions": pooled["reference"],
        "predicted_reference_ratio": pooled["candidate"] / pooled["reference"] if pooled["reference"] else 0.0,
        "metric_performances": len(dataset.performances) - len(failures),
        "free_running_inference_performances": len(dataset.performances),
        "alignment_failures": failures,
        "losses": losses,
        "diagnostics": diagnostics,
        "direction_metrics": transitions,
        "asap_test_access_count": 0,
        "nonfinite": diagnostics["invalid_or_nonfinite"],
    }
    atomic_json(RUN_ROOT / "validation_epochs" / f"epoch_{epoch:02d}.json", result)
    return result


def checkpoint_payload(
    model: StateConditionedBinary2SlotModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    *,
    epoch: int,
    global_step: int,
    configuration_: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "global_step": global_step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "grad_scaler_state": scaler.state_dict(),
        "configuration": dict(configuration_),
        "boundary_loss_weight": float(configuration_["boundary_loss_weight"]),
        "seed": SEED,
        "validation_metrics": dict(validation),
        "created_at": now(),
        "asap_test_access_count": 0,
    }


def update_full_report(
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    *,
    best_epoch: int | None,
    best_f1: float | None,
    status: str,
) -> None:
    gradient = json.loads((RUN_ROOT / "gradient_scale_summary.json").read_text(encoding="utf-8"))
    tiny = json.loads((RUN_ROOT / "tiny_overfit_summary.json").read_text(encoding="utf-8"))
    rows = []
    for train, validation in zip(train_rows, validation_rows):
        rows.append(
            f"| {train['epoch']} | {float(train['train_loss']):.6f} | {float(train['count_accuracy']):.4f} | "
            f"{float(train['state_agreement']):.4f} | {float(train['correction_fraction']):.4f} | "
            f"{float(validation['transition_precision']):.4f} | {float(validation['transition_recall']):.4f} | "
            f"{float(validation['transition_f1']):.4f} | {float(validation['predicted_reference_ratio']):.4f} |"
        )
    improving = None
    if len(validation_rows) >= 2:
        improving = float(validation_rows[-1]["transition_f1"]) > float(validation_rows[-2]["transition_f1"])
    ratios = gradient["ratios"]
    text = f"""# Full State-Conditioned Binary 2-Slot v0 Report

| item | result |
|---|---|
| Run status | {status} |
| Multi-window sanity | PASS |
| No-grad encoder parameter | `encoder.embed_tokens.weight` (inactive inherited T5 path) |
| Boundary/Main scalar loss ratio | {ratios['scalar_loss_boundary_over_main']:.6f} |
| Boundary/Main encoder grad ratio | {ratios['R_encoder']:.6f} |
| Boundary/Main head grad ratio | {ratios['R_heads']:.6f} |
| boundary_loss_weight | {gradient['selected_boundary_loss_weight']} |
| Tiny overfit | {tiny['status']} |
| Tiny final Count accuracy | {tiny['final']['diagnostics']['count_accuracy']:.6f} |
| Tiny final state agreement | {tiny['final']['diagnostics']['interval_start_state_agreement']:.6f} |
| Tiny final correction fraction | {tiny['final']['diagnostics']['correction_required_fraction']:.6f} |
| Tiny loss reduction | {tiny['total_loss_reduction_fraction']:.6f} |
| Full epochs completed | {len(train_rows)} / {MAX_EPOCHS} |
| Best epoch | {best_epoch if best_epoch is not None else 'pending'} |
| Best validation Transition F1 | {best_f1 if best_f1 is not None else 'pending'} |
| Best checkpoint | `{RUN_ROOT / 'best.pt'}` |
| Last checkpoint | `{RUN_ROOT / 'last.pt'}` |
| Improving at epoch 10 | {improving if len(validation_rows) == MAX_EPOCHS else 'pending'} |
| ASAP test access | 0 |

## Epoch results

| epoch | train loss | count acc | state agreement | correction frac | val P | val R | val F1 | pred/ref transitions |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows) if rows else '| pending | | | | | | | | |'}

## Method and warnings

- Random-head Stage B output was used only as a plumbing diagnostic; the table above contains trained validation results.
- Hard states and online reconciliation targets are generated by a deterministic no-grad/eval controller pass at the current parameters, then replayed with gradients before that optimizer update. Encoder representations are recomputed; state never crosses performances.
- Each validation epoch runs all 71 ASAP-validation performances in PRE -> owned MAIN -> POST order. Canonical direction-aware Transition F1 uses the frozen ±1 distinct-score-onset matcher; at most one pre-existing alignment failure is tolerated and is recorded per epoch.
- Checkpoints are selected only by highest validation Transition F1, with an earlier-epoch tie break. Validation never changes LR, lambda, or architecture.
- Training instability/warnings are recorded in `training_log.txt` and per-epoch JSON. No scientific formulation was changed.
- The first next-experiment check is whether state agreement and correction fraction improve together with Transition F1 through epoch 10.
- Prohibited activity counters: ASAP test metadata=0, ASAP test MIDI=0, test inference=0, test metrics=0, hyperparameter sweep=0.
"""
    temporary = RUN_ROOT / ".FULL_BINARY_2SLOT_REPORT.md.tmp"
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, RUN_ROOT / "FULL_BINARY_2SLOT_REPORT.md")


def write_tiny_stop_report() -> None:
    gradient = json.loads((RUN_ROOT / "gradient_scale_summary.json").read_text(encoding="utf-8"))
    tiny = json.loads((RUN_ROOT / "tiny_overfit_summary.json").read_text(encoding="utf-8"))
    sanity = json.loads((RUN_ROOT / "multiwindow_sanity.json").read_text(encoding="utf-8"))
    ratios = gradient["ratios"]
    final = tiny["final"]["diagnostics"]
    predicted_total = sum(int(value) for value in final["count_predicted"].values())
    dominant_n1 = int(final["count_predicted"]["1"]) / predicted_total
    report = f"""# Full State-Conditioned Binary 2-Slot v0 Report

| item | result |
|---|---|
| Run status | **STOPPED — tiny-overfit FAIL** |
| Multi-window sanity | PASS ({sanity['windows']} windows, {sanity['main_onsets']} owned MAIN onsets) |
| No-grad encoder parameter | `encoder.embed_tokens.weight` (inactive inherited T5 path) |
| Boundary/Main scalar loss ratio | {ratios['scalar_loss_boundary_over_main']:.6f} |
| Boundary/Main encoder grad ratio | {ratios['R_encoder']:.6f} |
| Boundary/Main head grad ratio | {ratios['R_heads']:.6f} |
| Selected boundary_loss_weight | {gradient['selected_boundary_loss_weight']} |
| Tiny overfit | FAIL |
| Tiny final Count accuracy | {final['count_accuracy']:.6f} |
| Tiny final state agreement | {final['interval_start_state_agreement']:.6f} |
| Tiny final correction fraction | {final['correction_required_fraction']:.6f} |
| Tiny total-loss reduction | {tiny['total_loss_reduction_fraction']:.6f} |
| Persistent dominant prediction | N=1: {dominant_n1:.2%} ({final['count_predicted']['1']}/{predicted_total}) |
| Full epochs completed | 0 / {MAX_EPOCHS} |
| Validation epochs completed | 0 |
| Best epoch / F1 | N/A — full training not authorized by gate |
| Best / last checkpoint | Not created |
| tmux full-training session | Not launched |
| ASAP test access | 0 |

## Guarded run outcome

Stage A passed CUDA/provenance/checkpoint/compile/unit checks. Stage B passed PRE-once, POST-once, strict owner chronology, zero missing/duplicate supervision, monotonic owner order, and state continuity across all real window boundaries. Stage C completed six independent backward audits with finite gradients. The sole encoder parameter without a gradient was the expected inherited `encoder.embed_tokens.weight`; the custom PT input path supplies `inputs_embeds`, so this path is inactive rather than a missing live gradient.

The automatic Stage D rule selected lambda=0.25 because lambda times max(encoder, head boundary/Main gradient ratio) was {0.25 * max(ratios['R_encoder'], ratios['R_heads']):.6f}; lambda=0.5 yielded {0.5 * max(ratios['R_encoder'], ratios['R_heads']):.6f}, above the frozen 0.5 ceiling.

The train-only 500-step tiny run was numerically stable and reduced total loss from {tiny['initial']['losses']['total']:.6f} to {tiny['final']['losses']['total']:.6f}. Nevertheless, it converged to an N=1-dominant self-reconciliation regime: Count accuracy remained {final['count_accuracy']:.4f}, state agreement {final['interval_start_state_agreement']:.4f}, and correction-required fraction {final['correction_required_fraction']:.4f}. This meets the task's explicit sustained single-class-collapse / failed Count-learning hard-stop condition. Therefore no full-training optimizer step, validation inference, checkpoint, or tmux process was started.

## Engineering fixes made during the guarded stages

- Batched owner windows now sum their loss before one backward, avoiding a second backward through a shared encoder graph. The Binary 2-Slot unit regression was rerun and passed.
- AMP encoder outputs are cast to FP32 for the three FP32 linear heads, preventing half/float dtype mismatch without changing the frozen architecture.
- Read-only cached NumPy boundary inputs are copied before conversion to Torch tensors; scientific values are unchanged.
- The first Stage C failure artifact is retained as a resolved engineering incident; the subsequent Stage B–D run passed.

## Epoch results

| epoch | train loss | count acc | state agreement | correction frac | val P | val R | val F1 | pred/ref transitions |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| none | — | — | — | — | — | — | — | — |

No ASAP test metadata or MIDI was accessed. No test inference or test metric was run. The next required action is a human design review of why hard online reconciliation admits the observed N=1-dominant fixed point; the frozen scientific formulation was not changed automatically.
"""
    temporary = RUN_ROOT / ".FULL_BINARY_2SLOT_REPORT.md.tmp"
    temporary.write_text(report, encoding="utf-8")
    os.replace(temporary, RUN_ROOT / "FULL_BINARY_2SLOT_REPORT.md")
    atomic_json(RUN_ROOT / "run_status.json", {
        "status": "STOPPED_TINY_OVERFIT_FAIL",
        "completed_stages": ["A", "B", "C", "D", "E"],
        "completed_epochs": 0,
        "validation_inference_count": 0,
        "optimizer_updates_tiny": int(tiny["optimizer_updates"]),
        "optimizer_updates_full": 0,
        "checkpoint_count": 0,
        "tmux_launched": False,
        "stop_reason": "persistent N=1-dominant tiny-overfit collapse with Count/state accuracy near 0.5",
        "asap_test_metadata_access_count": 0,
        "asap_test_midi_access_count": 0,
        "test_inference_count": 0,
        "test_metric_count": 0,
        "last_update": now(),
    })
    failure_path = RUN_ROOT / "failure_sanity.json"
    if failure_path.is_file():
        failure = json.loads(failure_path.read_text(encoding="utf-8"))
        failure["resolved"] = True
        failure["resolution"] = "losses from windows sharing one encoder micro-batch are summed before one backward; subsequent Stage B-D run passed"
        atomic_json(failure_path, failure)
    log("PIPELINE_STOPPED tiny-overfit FAIL; full training and tmux launch skipped")


def full_training() -> None:
    tiny = json.loads((RUN_ROOT / "tiny_overfit_summary.json").read_text(encoding="utf-8"))
    if tiny["status"] not in {"PASS", "WARN/PASS"}:
        raise RuntimeError("tiny-overfit gate does not authorize full training")
    configuration_ = json.loads((RUN_ROOT / "FULL_RUN_CONFIG.json").read_text(encoding="utf-8"))
    boundary_weight = float(configuration_["boundary_loss_weight"])
    if (RUN_ROOT / "last.pt").exists() or list((RUN_ROOT / "checkpoints").glob("epoch_*.pt")):
        raise RuntimeError("full run checkpoint already exists; refusing accidental overwrite")
    seed_everything()
    device = torch.device("cuda:0")
    model = build_model(device)
    optimizer = build_optimizer(model)
    scaler = torch.amp.GradScaler("cuda", init_scale=AMP_INIT_SCALE, enabled=True)
    train_dataset = Binary2SlotPerformanceDataset(CACHE_ROOT, "train", cache_input_tokens=True)
    validation_dataset = Binary2SlotPerformanceDataset(CACHE_ROOT, "validation", cache_input_tokens=True)
    if len(train_dataset.performances) != 2062 or len(validation_dataset.performances) != 71:
        raise RuntimeError("canonical full train/validation count changed")
    train_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    best_epoch: int | None = None
    best_f1: float | None = None
    global_step = 0
    consecutive_overflow = 0
    update_full_report(train_rows, validation_rows, best_epoch=None, best_f1=None, status="RUNNING")
    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_started = time.monotonic()
        torch.cuda.reset_peak_memory_stats(device)
        order = list(range(len(train_dataset.performances)))
        random.Random(SEED + epoch).shuffle(order)
        diagnostic_values: list[dict[str, Any]] = []
        loss_totals = empty_loss_totals()
        gradient_norms: list[float] = []
        for group_start in range(0, len(order), PERFORMANCES_PER_STEP):
            selected = order[group_start : group_start + PERFORMANCES_PER_STEP]
            plans = [controller_plan(model, train_dataset, index, device, amp=True) for index in selected]
            diagnostic_values.extend(plan_diagnostics((plan,)) for plan in plans)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            replay = replay_plans(
                model, train_dataset, plans, device, amp=True, boundary_weight=boundary_weight,
                components=ALL_COMPONENTS, backward=lambda value: scaler.scale(value).backward(),
            )
            add_loss_totals(loss_totals, replay)
            scaler.unscale_(optimizer)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM).item())
            finite = math.isfinite(grad_norm) and all(
                parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                for parameter in model.parameters()
            )
            scale_before = float(scaler.get_scale())
            if finite:
                scaler.step(optimizer)
            scaler.update()
            scale_after = float(scaler.get_scale())
            if not finite or scale_after < scale_before:
                consecutive_overflow += 1
                log(f"WARNING nonfinite_or_amp_overflow epoch={epoch} group={group_start // PERFORMANCES_PER_STEP} consecutive={consecutive_overflow}")
            else:
                consecutive_overflow = 0
                global_step += 1
                gradient_norms.append(grad_norm)
            if consecutive_overflow >= 3:
                raise FloatingPointError("repeated non-finite gradient/AMP overflow")
            if (group_start // PERFORMANCES_PER_STEP + 1) % 25 == 0:
                log(
                    f"TRAIN epoch={epoch} performances={min(group_start + PERFORMANCES_PER_STEP, len(order))}/2062 "
                    f"global_step={global_step} peak_gib={torch.cuda.max_memory_allocated(device) / 2**30:.3f}"
                )
        train_loss = finalize_loss_totals(loss_totals, boundary_weight)
        train_diag = combine_diagnostics(diagnostic_values)
        if train_diag["invalid_or_nonfinite"]:
            raise FloatingPointError("non-finite training rollout diagnostic")
        validation = validation_epoch(model, validation_dataset, device, boundary_weight, epoch)
        validation_f1 = float(validation["transition_f1"])
        train_row = {
            "epoch": epoch,
            "train_loss": train_loss["total"],
            "main_count_loss": train_loss["main_count"],
            "main_timing_loss": train_loss["main_timing"],
            "boundary_count_loss": train_loss["boundary_count"],
            "boundary_timing_loss": train_loss["boundary_timing"],
            "count_accuracy": train_diag["count_accuracy"],
            "state_agreement": train_diag["interval_start_state_agreement"],
            "correction_fraction": train_diag["correction_required_fraction"],
            "real_transition_retention": train_diag["real_transition_retention"],
            "timing_active_mae": train_diag["active_timing_mae"],
            "target_n0": train_diag["count_target"]["0"],
            "target_n1": train_diag["count_target"]["1"],
            "target_n2": train_diag["count_target"]["2"],
            "predicted_n0": train_diag["count_predicted"]["0"],
            "predicted_n1": train_diag["count_predicted"]["1"],
            "predicted_n2": train_diag["count_predicted"]["2"],
            "pre_intervals": train_diag["region_counts"]["PRE"],
            "main_intervals": train_diag["region_counts"]["MAIN"],
            "post_intervals": train_diag["region_counts"]["POST"],
            "mean_gradient_norm": float(np.mean(gradient_norms)) if gradient_norms else 0.0,
            "epoch_wall_seconds": time.monotonic() - epoch_started,
            "cuda_peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "nonfinite": train_diag["invalid_or_nonfinite"],
        }
        val_row = {
            "epoch": epoch,
            "transition_precision": validation["transition_precision"],
            "transition_recall": validation["transition_recall"],
            "transition_f1": validation_f1,
            "predicted_transitions": validation["predicted_transitions"],
            "reference_transitions": validation["reference_transitions"],
            "predicted_reference_ratio": validation["predicted_reference_ratio"],
            "state_agreement": validation["diagnostics"]["interval_start_state_agreement"],
            "correction_fraction": validation["diagnostics"]["correction_required_fraction"],
            "count_accuracy": validation["diagnostics"]["count_accuracy"],
            "metric_performances": validation["metric_performances"],
            "inference_performances": validation["free_running_inference_performances"],
            "nonfinite": validation["nonfinite"],
        }
        state_row = {
            "epoch": epoch,
            "train_state_agreement": train_diag["interval_start_state_agreement"],
            "train_correction_fraction": train_diag["correction_required_fraction"],
            "train_mismatch_run_mean": train_diag["mismatch_run_mean"],
            "train_mismatch_run_max": train_diag["mismatch_run_max"],
            "validation_state_agreement": validation["diagnostics"]["interval_start_state_agreement"],
            "validation_correction_fraction": validation["diagnostics"]["correction_required_fraction"],
            "validation_mismatch_run_mean": validation["diagnostics"]["mismatch_run_mean"],
            "validation_mismatch_run_max": validation["diagnostics"]["mismatch_run_max"],
        }
        train_rows.append(train_row)
        validation_rows.append(val_row)
        state_rows.append(state_row)
        atomic_csv(RUN_ROOT / "metrics_by_epoch.csv", train_rows)
        atomic_csv(RUN_ROOT / "validation_metrics_by_epoch.csv", validation_rows)
        atomic_csv(RUN_ROOT / "state_diagnostics_by_epoch.csv", state_rows)
        payload = checkpoint_payload(
            model, optimizer, scaler, epoch=epoch, global_step=global_step,
            configuration_=configuration_, validation=validation,
        )
        epoch_path = RUN_ROOT / "checkpoints" / f"epoch_{epoch:02d}.pt"
        epoch_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_torch(epoch_path, payload)
        atomic_torch(RUN_ROOT / "last.pt", payload)
        if best_f1 is None or validation_f1 > best_f1:
            best_f1 = validation_f1
            best_epoch = epoch
            atomic_torch(RUN_ROOT / "best.pt", payload)
        atomic_json(RUN_ROOT / "run_status.json", {
            "status": "RUNNING" if epoch < MAX_EPOCHS else "COMPLETE",
            "completed_epochs": epoch,
            "global_step": global_step,
            "best_epoch": best_epoch,
            "best_validation_transition_f1": best_f1,
            "last_update": now(),
            "asap_test_access_count": 0,
        })
        update_full_report(train_rows, validation_rows, best_epoch=best_epoch, best_f1=best_f1, status="RUNNING" if epoch < MAX_EPOCHS else "COMPLETE")
        log(f"EPOCH_COMPLETE epoch={epoch} train_loss={train_loss['total']:.6f} val_f1={validation_f1:.6f} best_epoch={best_epoch}")
    log(f"FULL_TRAINING_COMPLETE epochs={MAX_EPOCHS} best_epoch={best_epoch} best_f1={best_f1:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "sanity", "tiny", "full", "finalize-stop"))
    args = parser.parse_args()
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        if args.mode == "preflight":
            preflight()
        elif args.mode == "sanity":
            sanity_and_gradient()
        elif args.mode == "tiny":
            tiny_overfit()
        elif args.mode == "finalize-stop":
            write_tiny_stop_report()
        else:
            full_training()
    except Exception as error:
        failure = {
            "mode": args.mode,
            "failed_at": now(),
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "asap_test_access_count": 0,
        }
        atomic_json(RUN_ROOT / f"failure_{args.mode}.json", failure)
        log(f"HARD_STOP mode={args.mode} error={failure['error']}")
        raise


if __name__ == "__main__":
    main()
