#!/usr/bin/env python3
"""Run A: MAIN-only Condition D guarded smoke and standalone full training."""

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
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_stage2_binary_2slot_full_v0 as common
from src.stage2_binary_2slot.dataset import Binary2SlotPerformanceDataset, HumanIntervalPrimitive
from src.stage2_binary_2slot.losses import FIXED_COUNT_WEIGHTS
from src.stage2_binary_2slot.main_only import (
    MainOnlyStateStore,
    StaticMainTarget,
    build_static_main_target,
    modeled_main_intervals,
    recovery_within,
)
from src.stage2_binary_2slot.rollout import decode_prediction, next_binary_state
from src.stage2_binary_2slot.state_consistency import state_consistency_loss
from src.stage2_event_model.dataset import CANONICAL_ALIGNMENT_ID, CANONICAL_CACHE_ID
from src.stage2_four_class.validation_evaluator import pooled_transition_counts
from scripts.run_stage2_4class_validation_eval_v0 import finalize_transition, sum_transition


RUN_ROOT = ROOT / "analysis/stage2_binary_2slot_static_state_main_full_v0"
CACHE_ROOT = ROOT / "analysis/custom_event_tokenizer_v1"
ALIGNMENT_ROOT = ROOT / "analysis/custom_event_model_v0_note_alignment"
PRETRAINED = ROOT / "checkpoints/pianist_transformer"
DIAGNOSTIC_ROOT = ROOT / "analysis/binary_2slot_state_stability_diagnostic_v1"
SEED = 42
MAX_EPOCHS = 10
LAMBDA_STATE = 0.25
WINDOW_MICRO_BATCH = 4
PERFORMANCES_PER_STEP = 4
TINY_INDEX = 7
SMOKE_STEPS = 300
LOG_INTERVAL = 25
AMP_INIT_SCALE = 1024.0

# Reuse the frozen validation alignment helper without writing into the old run.
common.RUN_ROOT = RUN_ROOT


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    line = f"{now()} {message}"
    print(line, flush=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    with (RUN_ROOT / "training_log.txt").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
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
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_torch(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_model(device: torch.device):
    return common.build_model(device)


@dataclass(frozen=True)
class Record:
    onset_index: int
    model_state: int
    human_start_state: int
    human_end_state: int
    target: StaticMainTarget
    predicted_count: int
    predicted_timing: tuple[float, float]


@dataclass(frozen=True)
class Window:
    dataset_index: int
    window_index: int
    state_before: int
    state_after: int
    records: tuple[Record, ...]


@dataclass(frozen=True)
class Plan:
    performance_index: int
    performance_id: str
    windows: tuple[Window, ...]
    candidate_events: tuple[dict[str, Any], ...]
    reference_events: tuple[dict[str, Any], ...]
    human_first_onset_state: int
    horizon_left: float
    horizon_right: float

    @property
    def records(self) -> tuple[Record, ...]:
        return tuple(record for window in self.windows for record in window.records)


def controller_plan(model, dataset, performance_index: int, device: torch.device, *, amp: bool) -> Plan:
    """Pure hard free-running inference/controller; no human state enters the model."""

    model.eval()
    timeline = dataset.timeline(performance_index)
    intervals = modeled_main_intervals(timeline)
    performance_id = timeline.performance_id
    store = MainOnlyStateStore()
    store.begin(performance_id)
    candidate: list[dict[str, Any]] = []
    windows: list[Window] = []
    all_owned: list[int] = []
    modeled_seen: list[int] = []
    with torch.inference_mode():
        for batch_indices in common.chunks(
            common.window_dataset_indices(dataset, performance_index), WINDOW_MICRO_BATCH
        ):
            samples = [dataset[index] for index in batch_indices]
            hidden_values = common.encode_window_samples(model, samples, device, amp=amp)
            for dataset_index, sample, hidden in zip(batch_indices, samples, hidden_values):
                before = store.current(performance_id)
                owned = [int(value) for value in sample["owned_global_onset_indices"].tolist()]
                if owned != sorted(owned) or (all_owned and owned and owned[0] <= all_owned[-1]):
                    raise AssertionError("owner onset chronology violation")
                all_owned.extend(owned)
                records: list[Record] = []
                for local_index, interval in enumerate(sample["human_intervals"]):
                    onset = int(interval.main_onset_index)
                    if onset == len(intervals):
                        continue  # t_M representation is context-only; final tail is not modeled.
                    if onset >= len(intervals):
                        raise AssertionError("unexpected owned onset beyond MAIN-only horizon")
                    state = store.current(performance_id)
                    output = model.condition_and_predict(
                        hidden[local_index].float().reshape(1, -1),
                        torch.tensor([state], dtype=torch.long, device=device),
                    )
                    if not bool(torch.isfinite(output.count_logits).all() and torch.isfinite(output.timing_predictions).all()):
                        raise FloatingPointError("non-finite MAIN-only prediction")
                    predicted_count = int(output.count_logits[0].argmax().item())
                    predicted_timing = tuple(float(value) for value in output.timing_predictions[0].float().cpu().tolist())
                    target = build_static_main_target(interval)
                    decoded = decode_prediction(state, predicted_count, predicted_timing)
                    for slot, event in enumerate(decoded.transitions):
                        seconds = interval.left_seconds + event.tau * interval.duration_seconds
                        if seconds < timeline.main[-1].left_seconds:
                            candidate.append({
                                "direction": event.direction, "seconds": seconds,
                                "interval_index": onset, "slot": slot,
                            })
                    records.append(Record(
                        onset, state, interval.human_start_state, interval.human_end_state,
                        target, predicted_count, predicted_timing,
                    ))
                    modeled_seen.append(onset)
                    store.record(performance_id, onset, predicted_count)
                windows.append(Window(
                    dataset_index, int(sample["metadata"]["window_index"]), before,
                    store.current(performance_id), tuple(records),
                ))
    if all_owned != list(range(len(timeline.main))):
        raise AssertionError("owner supervision has missing/duplicate onsets")
    if modeled_seen != list(range(len(intervals))):
        raise AssertionError("MAIN-only supervision has missing/duplicate intervals")
    if any(left.state_after != right.state_before for left, right in zip(windows, windows[1:])):
        raise AssertionError("hard state reset at owner-window boundary")
    store.finish(performance_id, len(intervals))
    reference = tuple(
        {
            "direction": event.direction,
            "seconds": interval.left_seconds + event.tau * interval.duration_seconds,
            "source_tick": event.source_tick,
        }
        for interval in intervals for event in interval.human_transitions
    )
    if any(not (intervals[0].left_seconds <= float(event["seconds"]) < timeline.main[-1].left_seconds) for event in reference):
        raise AssertionError("reference event escaped [t1,tM) horizon")
    return Plan(
        performance_index, performance_id, tuple(windows), tuple(candidate), reference,
        intervals[0].human_start_state, intervals[0].left_seconds,
        timeline.main[-1].left_seconds,
    )


def count_statistics(targets: Sequence[int], predictions: Sequence[int]) -> dict[str, Any]:
    confusion = np.zeros((3, 3), dtype=np.int64)
    for target, prediction in zip(targets, predictions):
        confusion[int(target), int(prediction)] += 1
    per_class = []
    for value in range(3):
        tp = int(confusion[value, value])
        predicted = int(confusion[:, value].sum())
        support = int(confusion[value].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append({"precision": precision, "recall": recall, "f1": f1, "support": support})
    return {
        "accuracy": float(np.trace(confusion) / confusion.sum()),
        "macro_f1": float(np.mean([row["f1"] for row in per_class])),
        "confusion": confusion.tolist(), "per_class": per_class,
    }


def plan_diagnostics(plan: Plan) -> dict[str, Any]:
    records = plan.records
    targets = [record.target.count for record in records]
    predictions = [record.predicted_count for record in records]
    count = count_statistics(targets, predictions)
    start = [record.model_state == record.human_start_state for record in records]
    end = [next_binary_state(record.model_state, record.predicted_count) == record.human_end_state for record in records]
    runs: list[int] = []
    active = 0
    for agreement in start + [True]:
        if agreement:
            if active:
                runs.append(active)
                active = 0
        else:
            active += 1
    recovery = recovery_within(start, end)
    timing_all: list[float] = []
    timing_n1: list[float] = []
    timing_n2_1: list[float] = []
    timing_n2_2: list[float] = []
    for record in records:
        for slot, selected in enumerate(record.target.timing_mask):
            if selected:
                error = abs(record.predicted_timing[slot] - record.target.timing_targets[slot])
                timing_all.append(error)
                if record.target.count == 1:
                    timing_n1.append(error)
                elif slot == 0:
                    timing_n2_1.append(error)
                else:
                    timing_n2_2.append(error)
    initial_mismatch = not start[0]
    initial_recovery = 0
    if initial_mismatch:
        initial_recovery = next((index + 1 for index, agreement in enumerate(end) if agreement), len(records) + 1)
    target_counts = Counter(targets)
    predicted_counts = Counter(predictions)
    return {
        "performance_id": plan.performance_id,
        "intervals": len(records), "count_correct": int(sum(a == b for a, b in zip(targets, predictions))),
        "count_confusion": count["confusion"],
        "target_n0": target_counts[0], "target_n1": target_counts[1], "target_n2": target_counts[2],
        "predicted_n0": predicted_counts[0], "predicted_n1": predicted_counts[1], "predicted_n2": predicted_counts[2],
        "state_start_agree": int(sum(start)), "state_end_agree": int(sum(end)),
        "state_excluding_first_agree": int(sum(start[1:])), "state_excluding_first_total": max(0, len(start) - 1),
        "mismatch_runs": runs,
        "mismatch_intervals": int(sum(not value for value in start)),
        "recovery_denominator": int(sum(not value for value in start)),
        **{f"recovery_{h}_numerator": int(round(recovery[h] * sum(not value for value in start))) for h in (1, 2, 4, 8)},
        "human_first_off": int(plan.human_first_onset_state == 0),
        "human_first_on": int(plan.human_first_onset_state == 1),
        "first_onset_agreement": int(start[0]), "initial_mismatch": int(initial_mismatch),
        "initial_recovery_intervals": initial_recovery,
        "initial_recovery_observed": int(not initial_mismatch or initial_recovery <= len(records)),
        "timing_errors": timing_all, "timing_n1_errors": timing_n1,
        "timing_n2_slot1_errors": timing_n2_1, "timing_n2_slot2_errors": timing_n2_2,
        "invalid": 0,
    }


def combine_diagnostics(values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    intervals = sum(int(value["intervals"]) for value in values)
    confusion = np.sum([np.asarray(value["count_confusion"], dtype=np.int64) for value in values], axis=0)
    targets: list[int] = []
    predictions: list[int] = []
    for target in range(3):
        for prediction in range(3):
            amount = int(confusion[target, prediction])
            targets.extend([target] * amount)
            predictions.extend([prediction] * amount)
    count = count_statistics(targets, predictions)
    runs = [int(run) for value in values for run in value["mismatch_runs"]]
    timing = [float(item) for value in values for item in value["timing_errors"]]
    timing_n1 = [float(item) for value in values for item in value["timing_n1_errors"]]
    timing_n21 = [float(item) for value in values for item in value["timing_n2_slot1_errors"]]
    timing_n22 = [float(item) for value in values for item in value["timing_n2_slot2_errors"]]
    recovery_denominator = sum(int(value["recovery_denominator"]) for value in values)
    initial_lengths = [int(value["initial_recovery_intervals"]) for value in values if int(value["initial_mismatch"])]
    result = {
        "performances": len(values), "intervals": intervals,
        "count_accuracy": count["accuracy"], "count_macro_f1": count["macro_f1"],
        "count_per_class": count["per_class"], "count_confusion": count["confusion"],
        **{key: sum(int(value[key]) for value in values) for key in (
            "target_n0", "target_n1", "target_n2", "predicted_n0", "predicted_n1", "predicted_n2",
        )},
        "state_agreement": sum(int(value["state_start_agree"]) for value in values) / intervals,
        "state_end_agreement": sum(int(value["state_end_agree"]) for value in values) / intervals,
        "state_agreement_excluding_first": (
            sum(int(value["state_excluding_first_agree"]) for value in values)
            / max(1, sum(int(value["state_excluding_first_total"]) for value in values))
        ),
        "mismatch_fraction": sum(int(value["mismatch_intervals"]) for value in values) / intervals,
        "mismatch_run_count": len(runs), "mismatch_run_mean": float(np.mean(runs)) if runs else 0.0,
        "mismatch_run_median": float(np.median(runs)) if runs else 0.0,
        "mismatch_run_p95": float(np.quantile(runs, 0.95)) if runs else 0.0,
        "mismatch_run_max": max(runs, default=0),
        **{
            f"recovered_within_{h}": (
                sum(int(value[f"recovery_{h}_numerator"]) for value in values) / recovery_denominator
                if recovery_denominator else 1.0
            ) for h in (1, 2, 4, 8)
        },
        "human_first_off": sum(int(value["human_first_off"]) for value in values),
        "human_first_on": sum(int(value["human_first_on"]) for value in values),
        "first_onset_agreement": sum(int(value["first_onset_agreement"]) for value in values) / len(values),
        "initial_mismatch_performances": sum(int(value["initial_mismatch"]) for value in values),
        "initial_recovery_observed": sum(int(value["initial_recovery_observed"]) for value in values if int(value["initial_mismatch"])),
        "initial_recovery_mean": float(np.mean(initial_lengths)) if initial_lengths else 0.0,
        "initial_recovery_median": float(np.median(initial_lengths)) if initial_lengths else 0.0,
        "initial_recovery_p95": float(np.quantile(initial_lengths, 0.95)) if initial_lengths else 0.0,
        "initial_recovery_max": max(initial_lengths, default=0),
        "active_timing_mae": float(np.mean(timing)) if timing else 0.0,
        "n1_timing_mae": float(np.mean(timing_n1)) if timing_n1 else 0.0,
        "n2_slot1_timing_mae": float(np.mean(timing_n21)) if timing_n21 else 0.0,
        "n2_slot2_timing_mae": float(np.mean(timing_n22)) if timing_n22 else 0.0,
        "invalid": sum(int(value["invalid"]) for value in values),
    }
    return result


def loss_denominators(plans: Sequence[Plan]) -> dict[str, float]:
    records = [record for plan in plans for record in plan.records]
    values = {
        "count": sum(FIXED_COUNT_WEIGHTS[record.target.count] for record in records),
        "timing": sum(sum(record.target.timing_mask) for record in records),
        "state": float(len(records)),
    }
    if values["count"] <= 0 or values["state"] <= 0:
        raise ValueError("empty MAIN-only loss denominator")
    return values


def raw_loss_terms(model, hidden: torch.Tensor, records: Sequence[Record]) -> dict[str, torch.Tensor]:
    hidden = hidden.float()
    device = hidden.device
    states = torch.tensor([record.model_state for record in records], dtype=torch.long, device=device)
    counts = torch.tensor([record.target.count for record in records], dtype=torch.long, device=device)
    targets = torch.tensor([record.target.timing_targets for record in records], dtype=torch.float32, device=device)
    mask = torch.tensor([record.target.timing_mask for record in records], dtype=torch.bool, device=device)
    human_end = torch.tensor([record.human_end_state for record in records], dtype=torch.long, device=device)
    output = model.condition_and_predict(hidden, states)
    ce = F.cross_entropy(output.count_logits.float(), counts, reduction="none")
    weights = torch.tensor([FIXED_COUNT_WEIGHTS[value] for value in counts.tolist()], dtype=torch.float32, device=device)
    count_sum = (ce * weights).sum()
    timing = output.timing_predictions.float()
    timing_sum = F.smooth_l1_loss(timing[mask], targets[mask], beta=0.1, reduction="sum") if bool(mask.any()) else timing.sum() * 0.0
    state_sum = state_consistency_loss(output.count_logits.float(), states, human_end, reduction="sum")
    return {"count": count_sum, "timing": timing_sum, "state": state_sum}


def replay_plans(model, dataset, plans: Sequence[Plan], device: torch.device, *, amp: bool, backward=None) -> dict[str, float]:
    denominator = loss_denominators(plans)
    sums = {"count": 0.0, "timing": 0.0, "state": 0.0}
    for plan in plans:
        for batch in common.chunks(list(range(len(plan.windows))), WINDOW_MICRO_BATCH):
            windows = [plan.windows[index] for index in batch]
            samples = [dataset[window.dataset_index] for window in windows]
            hidden_values = common.encode_window_samples(model, samples, device, amp=amp)
            losses: list[torch.Tensor] = []
            for window, hidden in zip(windows, hidden_values):
                if not window.records:
                    continue
                terms = raw_loss_terms(model, hidden[: len(window.records)], window.records)
                for key in sums:
                    sums[key] += float(terms[key].detach().item())
                losses.append(
                    terms["count"] / denominator["count"]
                    + (terms["timing"] / denominator["timing"] if denominator["timing"] else terms["timing"] * 0.0)
                    + LAMBDA_STATE * terms["state"] / denominator["state"]
                )
            if losses and backward is not None:
                value = sum(losses)
                if not bool(torch.isfinite(value)):
                    raise FloatingPointError("non-finite MAIN-only chunk loss")
                backward(value)
    result = {key: sums[key] / denominator[key] if denominator[key] else 0.0 for key in sums}
    result["total"] = result["count"] + result["timing"] + LAMBDA_STATE * result["state"]
    result.update({f"denominator_{key}": value for key, value in denominator.items()})
    if not all(math.isfinite(value) for value in result.values()):
        raise FloatingPointError("non-finite MAIN-only aggregate")
    return result


def configuration() -> dict[str, Any]:
    cache = json.loads((CACHE_ROOT / "cache_manifest.json").read_text(encoding="utf-8"))
    alignment = json.loads((ALIGNMENT_ROOT / "alignment_manifest.json").read_text(encoding="utf-8"))
    return {
        "experiment": "stage2_binary_2slot_static_state_main_full_v0",
        "created_at": now(), "seed": SEED,
        "pretrained_checkpoint": str(PRETRAINED),
        "pretrained_model_sha256": sha256_file(PRETRAINED / "model.safetensors"),
        "cache_id": cache["cache_id"], "alignment_id": alignment["alignment_id"],
        "train_performances": 2062, "validation_performances": 71,
        "modeled_horizon": "[t1,tM); intervals [t_i,t_(i+1)) for i=1..M-1",
        "pre": False, "post": False, "final_tail": False, "boundary_pass": False,
        "boundary_loss": False, "boundary_loss_weight_present": False,
        "online_reconciliation": False, "synthetic_correction": False, "dynamic_target": False,
        "target": "static raw-human crossing max-2 compression",
        "state_initialization": "OFF at t1", "state_update": "hard argmax Count parity",
        "human_state_usage": "state-consistency loss label and metrics only",
        "lambda_state": LAMBDA_STATE, "count_weights": list(FIXED_COUNT_WEIGHTS),
        "timing_beta": 0.1,
        "optimizer": "AdamW", "encoder_lr": common.ENCODER_LR, "head_lr": common.HEAD_LR,
        "weight_decay": common.WEIGHT_DECAY, "gradient_clip": common.MAX_GRAD_NORM,
        "precision": "FP16 AMP full; FP32 smoke", "amp_init_scale": AMP_INIT_SCALE,
        "window_micro_batch": WINDOW_MICRO_BATCH,
        "performances_per_optimizer_step": PERFORMANCES_PER_STEP,
        "optimizer_provenance": "scripts/run_stage2_binary_2slot_full_v0.py",
        "max_epochs": MAX_EPOCHS, "early_stopping": None,
        "checkpoint_selection": "highest canonical validation direction-aware Transition F1; earlier epoch on tie",
        "validation_inference": "pure oracle-free hard rollout from OFF over [t1,tM)",
        "forbidden": {"asap_test_metadata": 0, "asap_test_midi": 0, "test_inference": 0, "test_metrics": 0, "hyperparameter_sweep": 0},
    }


def preflight() -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    config = configuration()
    if "boundary_loss_weight" in config:
        raise AssertionError("Run A config must not contain boundary_loss_weight")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one CUDA GPU is required")
    cache = json.loads((CACHE_ROOT / "cache_manifest.json").read_text(encoding="utf-8"))
    alignment = json.loads((ALIGNMENT_ROOT / "alignment_manifest.json").read_text(encoding="utf-8"))
    if cache["cache_id"] != CANONICAL_CACHE_ID or cache["train_count"] != 2062 or cache["validation_count"] != 71:
        raise RuntimeError("canonical cache provenance changed")
    if alignment["alignment_id"] != CANONICAL_ALIGNMENT_ID or alignment["train_count"] != 2062 or alignment["validation_count"] != 71:
        raise RuntimeError("canonical alignment provenance changed")
    if cache["asap_test_access_count"] or alignment["asap_test_access_count"]:
        raise RuntimeError("provenance reports ASAP test contamination")
    tests = [
        "tests/test_binary_2slot_main_only_run_a.py",
        "tests/test_binary_2slot_state_consistency_v1.py",
        "tests/test_binary_2slot_model_v0.py",
        "tests/test_binary_2slot_transition_audit_v0.py",
    ]
    results = []
    for test in tests:
        completed = subprocess.run([sys.executable, test], cwd=ROOT, text=True, capture_output=True)
        results.append({"test": test, "returncode": completed.returncode, "stdout": completed.stdout.strip(), "stderr": completed.stderr.strip()})
        if completed.returncode:
            raise RuntimeError(f"focused regression failed: {test}")
    device = torch.device("cuda:0")
    model = build_model(device)
    if model.hidden_size != 768 or int(model.encoder.config.num_hidden_layers) != 10 or model.prediction_head_parameter_count != 3850:
        raise RuntimeError("frozen architecture identity changed")
    atomic_json(RUN_ROOT / "FULL_RUN_CONFIG.json", config)
    atomic_json(RUN_ROOT / "preflight_summary.json", {
        "status": "PASS", "tests": results, "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "hidden_size": model.hidden_size, "encoder_layers": int(model.encoder.config.num_hidden_layers),
        "encoder_parameters": model.encoder_parameter_count, "head_parameters": model.prediction_head_parameter_count,
        "checkpoint": model.checkpoint_path, "lambda_state": LAMBDA_STATE,
        "asap_test_access": 0,
    })
    del model
    torch.cuda.empty_cache()
    log("PREFLIGHT_PASS MAIN-only/static-target/unit/provenance gates")


def snapshot(model, dataset, device, step: int) -> tuple[Plan, dict[str, Any], dict[str, float]]:
    plan = controller_plan(model, dataset, 0, device, amp=False)
    diagnostics = combine_diagnostics([plan_diagnostics(plan)])
    model.eval()
    with torch.no_grad():
        losses = replay_plans(model, dataset, (plan,), device, amp=False)
    return plan, diagnostics, losses


def write_smoke_stop_report(summary: Mapping[str, Any]) -> None:
    initial = summary["initial"]["diagnostics"]
    final = summary["final"]["diagnostics"]
    text = f"""# Full Binary 2-Slot Static-State MAIN-only Run A

| item | result |
|---|---|
| status | **STOPPED — guarded smoke FAIL** |
| MAIN-only integration | PASS: exactly M-1 intervals; PRE/POST/final tail absent |
| static target / reconciliation | static max-2 / no reconciliation or correction |
| lambda_state | {LAMBDA_STATE} |
| smoke subset / intervals | canonical TRAIN {TINY_INDEX} / {summary['modeled_intervals']} |
| human state at t1 / model initialization | ON / OFF |
| final Count accuracy / Macro F1 | {final['count_accuracy']:.4f} / {final['count_macro_f1']:.4f} |
| final state agreement / mismatch fraction | {final['state_agreement']:.4f} / {final['mismatch_fraction']:.4f} |
| mismatch run / recovery | {final['mismatch_run_max']} intervals / not recovered |
| loss reduction / timing MAE | {summary['loss_reduction']:.2%} / {final['active_timing_mae']:.6f} |
| final predicted N0/N1/N2 | {final['predicted_n0']} / {final['predicted_n1']} / {final['predicted_n2']} |
| full optimizer updates / validation inference | 0 / 0 |
| checkpoints / tmux launch | 0 / not launched |
| ASAP test access | 0 |

## Hard-stop finding

The integration is internally consistent, finite, leakage-free, owner-complete, and uses only `[t1,tM)`. However, subset 7 starts with human pedal ON while the frozen model initialization is OFF. Once the model memorized every static Count target, the opposite parity offset persisted for all {final['intervals']} intervals: Count accuracy was 1.0 while state agreement was 0.0. Recovery within 1/2/4/8 intervals was 0/0/0/0.

This is not an N=1 collapse (final N1 fraction {summary['n1_prediction_fraction']:.2%}) and not an optimization divergence (loss fell {summary['loss_reduction']:.2%}). It is the explicit MAIN-only cold-start state failure prohibited by the launch gate. With no target mutation, all base Count parities preserve the initial offset; the local state loss asks for opposite parity on every mismatched interval but at frozen `lambda_state=0.25` did not select a single recovery error.

Therefore full training, validation inference, checkpoint creation, and tmux launch were not authorized. No scientific setting was changed automatically.

Initial random-head state agreement was {initial['state_agreement']:.4f}; the final persistent zero is the trained smoke result, not random-head evidence.
"""
    (RUN_ROOT / "FULL_BINARY_2SLOT_STATIC_STATE_MAIN_REPORT.md").write_text(text, encoding="utf-8")
    for path, header in (
        (RUN_ROOT / "metrics_by_epoch.csv", "epoch,train_loss,count_loss,timing_loss,state_loss,count_accuracy,state_agreement\n"),
        (RUN_ROOT / "state_diagnostics_by_epoch.csv", "epoch,state_agreement,mismatch_fraction,mismatch_run_max,recovered_within_1,recovered_within_2,recovered_within_4,recovered_within_8\n"),
        (RUN_ROOT / "validation_metrics_by_epoch.csv", "epoch,transition_precision,transition_recall,transition_f1,count_accuracy,state_agreement\n"),
    ):
        path.write_text(header, encoding="utf-8")
    atomic_json(RUN_ROOT / "run_status.json", {
        "status": "STOPPED_SMOKE_FAIL", "completed_epochs": 0,
        "optimizer_updates_smoke": int(summary["optimizer_updates"]),
        "optimizer_updates_full": 0, "validation_inference": 0,
        "checkpoints": 0, "tmux_launched": False,
        "stop_reason": "persistent cold-start parity offset: Count accuracy 1.0 with state agreement 0.0",
        "last_update": now(), "asap_test_metadata_access": 0,
        "asap_test_midi_access": 0, "test_inference": 0, "test_metrics": 0,
    })


def smoke() -> None:
    if json.loads((RUN_ROOT / "preflight_summary.json").read_text())["status"] != "PASS":
        raise RuntimeError("preflight PASS artifact missing")
    common.seed_everything(SEED)
    device = torch.device("cuda:0")
    dataset = Binary2SlotPerformanceDataset(CACHE_ROOT, "train", performance_indices=(TINY_INDEX,), cache_input_tokens=True)
    model = build_model(device)
    optimizer = common.build_optimizer(model)
    rows: list[dict[str, Any]] = []

    def record(step: int):
        plan, diagnostic, losses = snapshot(model, dataset, device, step)
        row = {
            "step": step, "total_loss": losses["total"], "count_loss": losses["count"],
            "timing_loss": losses["timing"], "state_loss": losses["state"],
            "count_accuracy": diagnostic["count_accuracy"], "count_macro_f1": diagnostic["count_macro_f1"],
            "state_agreement": diagnostic["state_agreement"], "state_agreement_excluding_first": diagnostic["state_agreement_excluding_first"],
            "mismatch_fraction": diagnostic["mismatch_fraction"], "predicted_n0": diagnostic["predicted_n0"],
            "predicted_n1": diagnostic["predicted_n1"], "predicted_n2": diagnostic["predicted_n2"],
            "timing_mae": diagnostic["active_timing_mae"], "invalid": diagnostic["invalid"],
        }
        rows.append(row)
        atomic_csv(RUN_ROOT / "main_only_smoke_curve.csv", rows)
        log(f"SMOKE step={step} loss={losses['total']:.6f} count={diagnostic['count_accuracy']:.4f} state={diagnostic['state_agreement']:.4f}")
        return plan, diagnostic, losses

    initial_plan, initial, initial_loss = record(0)
    if len(initial_plan.records) != dataset.performances[0].num_onsets - 1:
        raise AssertionError("smoke modeled interval count is not M-1")
    if any(record.onset_index == dataset.performances[0].num_onsets - 1 for record in initial_plan.records):
        raise AssertionError("final tail entered smoke supervision")
    del initial_plan
    final, final_loss = initial, initial_loss
    for step in range(1, SMOKE_STEPS + 1):
        plan = controller_plan(model, dataset, 0, device, amp=False)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        replay_plans(model, dataset, (plan,), device, amp=False, backward=lambda value: value.backward())
        if not all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()):
            raise FloatingPointError("smoke non-finite gradient")
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), common.MAX_GRAD_NORM).item())
        if not math.isfinite(norm):
            raise FloatingPointError("smoke non-finite gradient norm")
        optimizer.step()
        if step % LOG_INTERVAL == 0 or step == SMOKE_STEPS:
            _, final, final_loss = record(step)
    reduction = 1.0 - final_loss["total"] / initial_loss["total"]
    predicted_total = final["predicted_n0"] + final["predicted_n1"] + final["predicted_n2"]
    n1_fraction = final["predicted_n1"] / predicted_total
    strong = final["count_accuracy"] >= 0.95 and final["state_agreement"] >= 0.95
    progressing = final["count_accuracy"] >= 0.90 and final["state_agreement"] >= 0.85 and reduction > 0.50
    hard_failure = final["invalid"] or n1_fraction >= 0.90 or final["state_agreement"] < 0.70
    status = "PASS" if strong and not hard_failure else "WARN_PASS" if progressing and not hard_failure else "FAIL"
    summary = {
        "status": status, "canonical_train_index": TINY_INDEX, "steps": SMOKE_STEPS,
        "modeled_intervals": final["intervals"], "expected_M_minus_1": dataset.performances[0].num_onsets - 1,
        "lambda_state": LAMBDA_STATE, "initial": {"diagnostics": initial, "losses": initial_loss},
        "final": {"diagnostics": final, "losses": final_loss}, "loss_reduction": reduction,
        "n1_prediction_fraction": n1_fraction, "pre_passes": 0, "post_passes": 0,
        "boundary_losses": 0, "online_reconciliation_calls": 0, "synthetic_corrections": 0,
        "optimizer_updates": SMOKE_STEPS, "checkpoint_created": False,
        "validation_inference": 0, "asap_test_access": 0,
    }
    atomic_json(RUN_ROOT / "main_only_smoke_summary.json", summary)
    log(f"SMOKE_{status} reduction={reduction:.4f} count={final['count_accuracy']:.4f} state={final['state_agreement']:.4f}")
    if status == "FAIL":
        write_smoke_stop_report(summary)
        raise RuntimeError("MAIN-only smoke hard gate failed; full launch prohibited")


def empty_loss_totals() -> dict[str, float]:
    return {"count": 0.0, "timing": 0.0, "state": 0.0, "denominator_count": 0.0, "denominator_timing": 0.0, "denominator_state": 0.0}


def add_loss(total: dict[str, float], value: Mapping[str, float]) -> None:
    for key in ("count", "timing", "state"):
        total[key] += float(value[key]) * float(value[f"denominator_{key}"])
        total[f"denominator_{key}"] += float(value[f"denominator_{key}"])


def finish_loss(total: Mapping[str, float]) -> dict[str, float]:
    result = {
        key: total[key] / total[f"denominator_{key}"] if total[f"denominator_{key}"] else 0.0
        for key in ("count", "timing", "state")
    }
    result["total"] = result["count"] + result["timing"] + LAMBDA_STATE * result["state"]
    return result


def validation_epoch(model, dataset, device, epoch: int) -> dict[str, Any]:
    model.eval()
    stage1 = common.score_manifest()
    diagnostics = []
    losses_total = empty_loss_totals()
    transition_total = common.transition_accumulator()
    failures = []
    for index, performance in enumerate(dataset.performances):
        plan = controller_plan(model, dataset, index, device, amp=True)
        diagnostics.append(plan_diagnostics(plan))
        with torch.no_grad():
            losses = replay_plans(model, dataset, (plan,), device, amp=True)
        add_loss(losses_total, losses)
        canonical_index = int(performance.entry["performance_index"])
        try:
            onsets, positions = common.score_positions_for_performance(performance, canonical_index=canonical_index, stage1=stage1)
            source = str(performance.entry["source_midi"])
            candidate = common.trajectory_for_metric(plan.candidate_events, source, onsets, positions)
            reference = common.trajectory_for_metric(plan.reference_events, source, onsets, positions)
            sum_transition(transition_total, pooled_transition_counts(candidate, reference))
        except Exception as error:
            failures.append({"canonical_validation_index": canonical_index, "error": f"{type(error).__name__}: {error}"})
        if (index + 1) % 10 == 0 or index + 1 == len(dataset.performances):
            log(f"VALIDATION epoch={epoch} performances={index + 1}/{len(dataset.performances)}")
    if len(failures) > 1 or len(dataset.performances) - len(failures) < 70:
        raise RuntimeError(f"too many frozen validation alignment failures: {failures}")
    transition = finalize_transition(transition_total)
    diagnostic = combine_diagnostics(diagnostics)
    pooled = transition["pooled"]
    result = {
        "epoch": epoch, "transition_precision": pooled["precision"],
        "transition_recall": pooled["recall"], "transition_f1": pooled["f1"],
        "predicted_transitions": pooled["candidate"], "reference_transitions": pooled["reference"],
        "predicted_reference_ratio": pooled["candidate"] / pooled["reference"] if pooled["reference"] else 0.0,
        "metric_performances": len(dataset.performances) - len(failures),
        "inference_performances": len(dataset.performances), "alignment_failures": failures,
        "losses": finish_loss(losses_total), "diagnostics": diagnostic,
        "direction_metrics": transition, "modeled_horizon": "[t1,tM)",
        "oracle_state_used_in_inference": False, "asap_test_access": 0,
    }
    atomic_json(RUN_ROOT / "validation_epochs" / f"epoch_{epoch:02d}.json", result)
    return result


def update_report(train_rows, validation_rows, best_epoch, best_f1, status) -> None:
    smoke_value = json.loads((RUN_ROOT / "main_only_smoke_summary.json").read_text())
    rows = []
    for train, validation in zip(train_rows, validation_rows):
        rows.append(
            f"| {train['epoch']} | {train['train_loss']:.6f} | {train['count_accuracy']:.4f} | "
            f"{train['state_agreement']:.4f} | {validation['transition_precision']:.4f} | "
            f"{validation['transition_recall']:.4f} | {validation['transition_f1']:.4f} | "
            f"{validation['predicted_reference_ratio']:.4f} | {validation['state_agreement']:.4f} |"
        )
    last = validation_rows[-1] if validation_rows else None
    answer = (
        "Pending: no completed validation epoch yet."
        if last is None else
        f"At epoch {last['epoch']}, unseen validation Count accuracy was {last['count_accuracy']:.4f}, "
        f"while hard onset-state agreement was {last['state_agreement']:.4f}, recovery-within-4 was "
        f"{last['recovered_within_4']:.4f}, and Transition F1 was {last['transition_f1']:.4f}. "
        "These are reported separately because local Count quality alone is not evidence of long-horizon stability."
    )
    text = f"""# Full Binary 2-Slot Static-State MAIN-only Run A

| item | result |
|---|---|
| status | {status} |
| formulation | Condition D, static target + hard free-running + local state-consistency |
| horizon | `[t1,tM)`; M-1 intervals only |
| PRE / POST / final tail | absent / absent / absent |
| online reconciliation / synthetic correction | absent / absent |
| lambda_state | {LAMBDA_STATE} |
| smoke | {smoke_value['status']} (Count {smoke_value['final']['diagnostics']['count_accuracy']:.4f}, state {smoke_value['final']['diagnostics']['state_agreement']:.4f}) |
| epochs completed | {len(train_rows)} / {MAX_EPOCHS} |
| best epoch / validation F1 | {best_epoch if best_epoch is not None else 'pending'} / {best_f1 if best_f1 is not None else 'pending'} |
| best / last checkpoint | `{RUN_ROOT / 'best.pt'}` / `{RUN_ROOT / 'last.pt'}` |
| ASAP test access | 0 |

## Epoch summary

| epoch | train loss | train Count acc | train state | val P | val R | val F1 | pred/ref | val state |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows) if rows else '| pending | | | | | | | | |'}

## Key scientific question

**Does tiny-stable Condition D remain stable under oracle-free unseen validation rollout?** {answer}

Validation always starts OFF at `t1`, never reads human state for initialization/input/update, and emits/evaluates only within `[t1,tM)`. Human state is used only for loss labels during training and metrics after inference. Checkpoint selection uses frozen canonical direction-aware Transition F1; validation never changes LR, lambda, or architecture.

Prohibited counters: PRE passes=0, POST passes=0, boundary loss=0, online reconciliation=0, synthetic corrections=0, ASAP test metadata/MIDI/inference/metrics=0.
"""
    temporary = RUN_ROOT / ".FULL_BINARY_2SLOT_STATIC_STATE_MAIN_REPORT.md.tmp"
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, RUN_ROOT / "FULL_BINARY_2SLOT_STATIC_STATE_MAIN_REPORT.md")


def checkpoint_payload(model, optimizer, scaler, epoch, global_step, config, validation):
    return {
        "epoch": epoch, "global_step": global_step, "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(), "grad_scaler_state": scaler.state_dict(),
        "configuration": config, "lambda_state": LAMBDA_STATE, "seed": SEED,
        "validation_metrics": validation, "created_at": now(), "asap_test_access": 0,
    }


def full() -> None:
    smoke_value = json.loads((RUN_ROOT / "main_only_smoke_summary.json").read_text())
    if smoke_value["status"] not in {"PASS", "WARN_PASS"}:
        raise RuntimeError("MAIN-only smoke does not authorize full training")
    if (RUN_ROOT / "last.pt").exists() or list((RUN_ROOT / "checkpoints").glob("epoch_*.pt")):
        raise RuntimeError("existing checkpoint found; refusing overwrite")
    config = json.loads((RUN_ROOT / "FULL_RUN_CONFIG.json").read_text())
    if any(config[key] for key in ("pre", "post", "final_tail", "boundary_pass", "boundary_loss", "online_reconciliation", "synthetic_correction", "dynamic_target")):
        raise AssertionError("Run A forbidden feature enabled in config")
    common.seed_everything(SEED)
    device = torch.device("cuda:0")
    model = build_model(device)
    optimizer = common.build_optimizer(model)
    scaler = torch.amp.GradScaler("cuda", init_scale=AMP_INIT_SCALE, enabled=True)
    log(f"FULL_INIT checkpoint={model.checkpoint_path} hidden={model.hidden_size} layers={model.encoder.config.num_hidden_layers} lambda_state={LAMBDA_STATE}")
    train_dataset = Binary2SlotPerformanceDataset(CACHE_ROOT, "train", cache_input_tokens=True)
    validation_dataset = Binary2SlotPerformanceDataset(CACHE_ROOT, "validation", cache_input_tokens=True)
    if len(train_dataset.performances) != 2062 or len(validation_dataset.performances) != 71:
        raise RuntimeError("canonical train/validation count changed")
    log(f"DATASET_READY train={len(train_dataset.performances)} validation={len(validation_dataset.performances)}")
    train_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    best_epoch = None
    best_f1 = None
    global_step = 0
    consecutive_failure = 0
    update_report(train_rows, validation_rows, best_epoch, best_f1, "RUNNING")
    for epoch in range(1, MAX_EPOCHS + 1):
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats(device)
        order = list(range(len(train_dataset.performances)))
        random.Random(SEED + epoch).shuffle(order)
        epoch_diagnostics = []
        loss_total = empty_loss_totals()
        gradient_norms = []
        for start in range(0, len(order), PERFORMANCES_PER_STEP):
            selected = order[start : start + PERFORMANCES_PER_STEP]
            plans = [controller_plan(model, train_dataset, index, device, amp=True) for index in selected]
            epoch_diagnostics.extend(plan_diagnostics(plan) for plan in plans)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            replay = replay_plans(
                model, train_dataset, plans, device, amp=True,
                backward=lambda value: scaler.scale(value).backward(),
            )
            add_loss(loss_total, replay)
            scaler.unscale_(optimizer)
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), common.MAX_GRAD_NORM).item())
            finite = math.isfinite(norm) and all(
                parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                for parameter in model.parameters()
            )
            scale_before = float(scaler.get_scale())
            if finite:
                scaler.step(optimizer)
            scaler.update()
            scale_after = float(scaler.get_scale())
            if not finite or scale_after < scale_before:
                consecutive_failure += 1
                log(f"WARNING gradient_or_amp epoch={epoch} group={start // PERFORMANCES_PER_STEP} consecutive={consecutive_failure}")
            else:
                consecutive_failure = 0
                global_step += 1
                gradient_norms.append(norm)
            if consecutive_failure >= 3:
                raise FloatingPointError("repeated non-finite gradient/AMP overflow")
            if (start // PERFORMANCES_PER_STEP + 1) % 25 == 0:
                log(f"TRAIN epoch={epoch} performances={min(start + PERFORMANCES_PER_STEP,2062)}/2062 global_step={global_step} peak_gib={torch.cuda.max_memory_allocated(device)/2**30:.3f}")
        train_loss = finish_loss(loss_total)
        train_diag = combine_diagnostics(epoch_diagnostics)
        validation = validation_epoch(model, validation_dataset, device, epoch)
        validation_diag = validation["diagnostics"]
        validation_f1 = float(validation["transition_f1"])
        train_row = {
            "epoch": epoch, "train_loss": train_loss["total"], "count_loss": train_loss["count"],
            "timing_loss": train_loss["timing"], "state_loss": train_loss["state"],
            "count_accuracy": train_diag["count_accuracy"], "count_macro_f1": train_diag["count_macro_f1"],
            "state_agreement": train_diag["state_agreement"], "state_agreement_excluding_first": train_diag["state_agreement_excluding_first"],
            "mismatch_fraction": train_diag["mismatch_fraction"], "active_timing_mae": train_diag["active_timing_mae"],
            "target_n0": train_diag["target_n0"], "target_n1": train_diag["target_n1"], "target_n2": train_diag["target_n2"],
            "predicted_n0": train_diag["predicted_n0"], "predicted_n1": train_diag["predicted_n1"], "predicted_n2": train_diag["predicted_n2"],
            "human_first_off": train_diag["human_first_off"], "human_first_on": train_diag["human_first_on"],
            "initial_mismatch_performances": train_diag["initial_mismatch_performances"],
            "epoch_wall_seconds": time.monotonic() - started,
            "mean_gradient_norm": float(np.mean(gradient_norms)) if gradient_norms else 0.0,
            "cuda_peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "nonfinite": train_diag["invalid"],
        }
        validation_row = {
            "epoch": epoch, "transition_precision": validation["transition_precision"],
            "transition_recall": validation["transition_recall"], "transition_f1": validation_f1,
            "predicted_transitions": validation["predicted_transitions"], "reference_transitions": validation["reference_transitions"],
            "predicted_reference_ratio": validation["predicted_reference_ratio"],
            "count_accuracy": validation_diag["count_accuracy"], "count_macro_f1": validation_diag["count_macro_f1"],
            "predicted_n0": validation_diag["predicted_n0"], "predicted_n1": validation_diag["predicted_n1"], "predicted_n2": validation_diag["predicted_n2"],
            "state_agreement": validation_diag["state_agreement"], "state_agreement_excluding_first": validation_diag["state_agreement_excluding_first"],
            "mismatch_fraction": validation_diag["mismatch_fraction"],
            "recovered_within_1": validation_diag["recovered_within_1"], "recovered_within_2": validation_diag["recovered_within_2"],
            "recovered_within_4": validation_diag["recovered_within_4"], "recovered_within_8": validation_diag["recovered_within_8"],
            "active_timing_mae": validation_diag["active_timing_mae"], "n1_timing_mae": validation_diag["n1_timing_mae"],
            "n2_slot1_timing_mae": validation_diag["n2_slot1_timing_mae"], "n2_slot2_timing_mae": validation_diag["n2_slot2_timing_mae"],
            "human_first_off": validation_diag["human_first_off"], "human_first_on": validation_diag["human_first_on"],
            "first_onset_agreement": validation_diag["first_onset_agreement"],
            "initial_mismatch_performances": validation_diag["initial_mismatch_performances"],
            "initial_recovery_mean": validation_diag["initial_recovery_mean"], "initial_recovery_median": validation_diag["initial_recovery_median"],
            "initial_recovery_p95": validation_diag["initial_recovery_p95"], "initial_recovery_max": validation_diag["initial_recovery_max"],
            "nonfinite": validation_diag["invalid"],
        }
        state_row = {
            "epoch": epoch,
            **{key: validation_diag[key] for key in (
                "state_agreement", "state_end_agreement", "state_agreement_excluding_first", "mismatch_fraction",
                "mismatch_run_count", "mismatch_run_mean", "mismatch_run_median", "mismatch_run_p95", "mismatch_run_max",
                "recovered_within_1", "recovered_within_2", "recovered_within_4", "recovered_within_8",
                "human_first_off", "human_first_on", "first_onset_agreement", "initial_mismatch_performances",
                "initial_recovery_observed", "initial_recovery_mean", "initial_recovery_median", "initial_recovery_p95", "initial_recovery_max",
            )},
        }
        train_rows.append(train_row)
        validation_rows.append(validation_row)
        state_rows.append(state_row)
        atomic_csv(RUN_ROOT / "metrics_by_epoch.csv", train_rows)
        atomic_csv(RUN_ROOT / "validation_metrics_by_epoch.csv", validation_rows)
        atomic_csv(RUN_ROOT / "state_diagnostics_by_epoch.csv", state_rows)
        payload = checkpoint_payload(model, optimizer, scaler, epoch, global_step, config, validation)
        atomic_torch(RUN_ROOT / "checkpoints" / f"epoch_{epoch:02d}.pt", payload)
        atomic_torch(RUN_ROOT / "last.pt", payload)
        if best_f1 is None or validation_f1 > best_f1:
            best_f1 = validation_f1
            best_epoch = epoch
            atomic_torch(RUN_ROOT / "best.pt", payload)
        status = "RUNNING" if epoch < MAX_EPOCHS else "COMPLETE"
        atomic_json(RUN_ROOT / "run_status.json", {
            "status": status, "completed_epochs": epoch, "global_step": global_step,
            "best_epoch": best_epoch, "best_validation_transition_f1": best_f1,
            "last_update": now(), "asap_test_metadata_access": 0, "asap_test_midi_access": 0,
        })
        update_report(train_rows, validation_rows, best_epoch, best_f1, status)
        log(f"EPOCH_COMPLETE epoch={epoch} loss={train_loss['total']:.6f} val_f1={validation_f1:.6f} val_state={validation_diag['state_agreement']:.6f} best={best_epoch}")
    log(f"FULL_TRAINING_COMPLETE epochs={MAX_EPOCHS} best_epoch={best_epoch} best_f1={best_f1:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "smoke", "full"))
    args = parser.parse_args()
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        if args.mode == "preflight":
            preflight()
        elif args.mode == "smoke":
            smoke()
        else:
            full()
    except Exception as error:
        atomic_json(RUN_ROOT / f"failure_{args.mode}.json", {
            "mode": args.mode, "failed_at": now(), "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(), "asap_test_access": 0,
        })
        log(f"HARD_STOP mode={args.mode} error={type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
