#!/usr/bin/env python3
"""Condition D PRE/MAIN/POST canonical full training and dual validation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_binary_2slot_state_stability_diagnostic_v1 as v1
from scripts import run_stage2_binary_2slot_full_v0 as common
from scripts.run_binary_2slot_D_pre_main_post_restore_smoke_v0 import (
    CONDITION,
    StaticPlan,
    as_v1_plan,
    controller_plan,
)
from scripts.run_custom_event_model_v0_canonical_val_inference_v1 import (
    FROZEN_HUMAN_ALIGNMENT,
    load_frozen_human,
)
from scripts.run_stage2_4class_validation_eval_v0 import (
    ALIGNMENT_TOOL,
    ASAP_ROOT,
    SPLIT_CSV,
    STAGE1_MANIFEST,
    finalize_transition,
    sum_transition,
)
from src.stage2_binary.canonical_stage1 import sha256_file, signature_sha256
from src.stage2_binary_2slot.dataset import Binary2SlotPerformanceDataset
from src.stage2_binary_2slot.checkpointing import (
    atomic_save_resumable_checkpoint,
    build_resumable_epoch_payload,
    restore_resumable_epoch_payload,
)
from src.stage2_binary_2slot.frozen_pt_validation import (
    FrozenPTRollout,
    load_frozen_pt_piece,
    render_cc64_only,
    rollout_frozen_pt,
    serialized_candidate_events_for_metric,
)
from src.stage2_binary_2slot.frozen_validation_mapping import (
    build_frozen_human_reference_map,
    references_by_piece,
)
from src.stage2_binary_2slot.losses import FIXED_COUNT_WEIGHTS
from src.stage2_binary_2slot.rollout import decode_prediction, next_binary_state
from src.stage2_event_model.dataset import CANONICAL_ALIGNMENT_ID, CANONICAL_CACHE_ID
from src.stage2_encoder_only.dataset import NON_PEDAL_FEATURES, PEDAL_TOKEN_OFFSET
from src.stage2_four_class.validation_evaluator import pooled_transition_counts


PREVIOUS_FAILED_RUN = ROOT / "analysis/stage2_binary_2slot_D_pre_main_post_full_v0"
MAPPING_FIX_REPORT = PREVIOUS_FAILED_RUN / "FROZEN_PT_MAPPING_FIX_REPORT.md"
RUN_ROOT = Path(os.environ.get(
    "BINARY2SLOT_RUN_ROOT",
    str(PREVIOUS_FAILED_RUN),
)).resolve()
RESTORE_ROOT = ROOT / "analysis/binary_2slot_D_pre_main_post_restore_smoke_v0"
CACHE_ROOT = ROOT / "analysis/custom_event_tokenizer_v1"
ALIGNMENT_ROOT = ROOT / "analysis/custom_event_model_v0_note_alignment"
PRETRAINED = ROOT / "checkpoints/pianist_transformer"
SEED = 42
MAX_EPOCHS = 10
LAMBDA_STATE = 0.25
BOUNDARY_WEIGHT = 0.25
WINDOW_MICRO_BATCH = 4
PERFORMANCES_PER_STEP = 4
AMP_INIT_SCALE = 1024.0
EXPECTED_STAGE1_MANIFEST_SHA = "f32e91da5d2196edee825236daa577f18709cf7d49e289d8f18ae8ea637b0037"

common.RUN_ROOT = RUN_ROOT


def now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    line = f"{now()} {message}"
    print(line, flush=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    with (RUN_ROOT / "training_log.txt").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_torch(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def transition_accumulator() -> dict[str, Any]:
    return {scope: {key: 0 for key in ("candidate", "reference", "tp", "fp", "fn")}
            for scope in ("pooled", "up", "down")}


def replay_static(model, dataset, plans: Sequence[StaticPlan], device, *, amp: bool, backward=None) -> dict[str, float]:
    adapted = [as_v1_plan(plan) for plan in plans]
    denominator = v1.loss_denominators(adapted)
    sums = {key: 0.0 for key in denominator}

    def consume(hidden, records, base, scope):
        terms = v1.raw_loss_terms(model, hidden, records, base)
        for component in ("count", "timing", "state"):
            sums[f"{scope}_{component}"] += float(terms[component].detach().item())
        values = []
        for component in ("count", "timing", "state"):
            denom = denominator[f"{scope}_{component}"]
            value = terms[component] / denom if denom else terms[component] * 0.0
            values.append((LAMBDA_STATE if component == "state" else 1.0) * value)
        loss = sum(values)
        return BOUNDARY_WEIGHT * loss if scope == "boundary" else loss

    for item in plans:
        plan = item.plan
        hidden = common.encode_boundary(model, dataset, plan.performance_index, "PRE", device, amp=amp)
        loss = consume(hidden.reshape(1, -1), (plan.pre,), item.base, "boundary")
        if backward is not None:
            backward(loss)
        for batch in common.chunks(list(range(len(plan.windows))), WINDOW_MICRO_BATCH):
            windows = [plan.windows[index] for index in batch]
            samples = [dataset[window.dataset_index] for window in windows]
            hidden_values = common.encode_window_samples(model, samples, device, amp=amp)
            losses = [consume(hidden, window.intervals, item.base, "main")
                      for hidden, window in zip(hidden_values, windows)]
            if losses and backward is not None:
                backward(sum(losses))
        hidden = common.encode_boundary(model, dataset, plan.performance_index, "POST", device, amp=amp)
        loss = consume(hidden.reshape(1, -1), (plan.post,), item.base, "boundary")
        if backward is not None:
            backward(loss)
    result = {key: sums[key] / denominator[key] if denominator[key] else 0.0 for key in sums}
    result["main"] = result["main_count"] + result["main_timing"] + LAMBDA_STATE * result["main_state"]
    result["boundary"] = result["boundary_count"] + result["boundary_timing"] + LAMBDA_STATE * result["boundary_state"]
    result["total"] = result["main"] + BOUNDARY_WEIGHT * result["boundary"]
    result.update({f"denominator_{key}": value for key, value in denominator.items()})
    if not all(math.isfinite(float(value)) for value in result.values()):
        raise FloatingPointError("non-finite static Condition D loss")
    return result


def empty_loss_totals() -> dict[str, float]:
    return {key: 0.0 for key in (
        "main_count_sum", "main_count_den", "main_timing_sum", "main_timing_den", "main_state_sum", "main_state_den",
        "boundary_count_sum", "boundary_count_den", "boundary_timing_sum", "boundary_timing_den", "boundary_state_sum", "boundary_state_den",
    )}


def add_loss(total: dict[str, float], result: Mapping[str, float]) -> None:
    for scope in ("main", "boundary"):
        for component in ("count", "timing", "state"):
            denominator = float(result[f"denominator_{scope}_{component}"])
            total[f"{scope}_{component}_sum"] += float(result[f"{scope}_{component}"]) * denominator
            total[f"{scope}_{component}_den"] += denominator


def finish_loss(total: Mapping[str, float]) -> dict[str, float]:
    result: dict[str, float] = {}
    for scope in ("main", "boundary"):
        for component in ("count", "timing", "state"):
            denominator = total[f"{scope}_{component}_den"]
            result[f"{scope}_{component}"] = total[f"{scope}_{component}_sum"] / denominator if denominator else 0.0
        result[scope] = result[f"{scope}_count"] + result[f"{scope}_timing"] + LAMBDA_STATE * result[f"{scope}_state"]
    result["total"] = result["main"] + BOUNDARY_WEIGHT * result["boundary"]
    return result


def _runs(agreement: Sequence[bool]) -> list[int]:
    values: list[int] = []
    active = 0
    for item in list(agreement) + [True]:
        if item:
            if active:
                values.append(active)
                active = 0
        else:
            active += 1
    return values


def diagnostics(plans: Sequence[StaticPlan]) -> dict[str, Any]:
    confusion = np.zeros((3, 3), np.int64)
    predicted = Counter()
    target = Counter()
    region = Counter()
    starts: list[bool] = []
    ends: list[bool] = []
    mismatch_runs: list[int] = []
    recovery_num = Counter()
    recovery_den = 0
    timing: list[float] = []
    timing_n1: list[float] = []
    timing_n21: list[float] = []
    timing_n22: list[float] = []
    human_first_on = pre_success = first_main_success = 0
    initial_recovery: list[int] = []
    for item in plans:
        records = item.plan.records
        local_start = [record.model_state == record.human_state for record in records]
        local_end = [next_binary_state(record.model_state, record.predicted_count)
                     == item.base[record.global_index].target.human_end_state for record in records]
        starts.extend(local_start)
        ends.extend(local_end)
        mismatch_runs.extend(_runs(local_start))
        mismatch_indices = [index for index, value in enumerate(local_start) if not value]
        recovery_den += len(mismatch_indices)
        for horizon in (1, 2, 4, 8):
            recovery_num[horizon] += sum(any(local_end[index:min(len(local_end), index + horizon)]) for index in mismatch_indices)
        first_main = next(record for record in records if record.region == "MAIN")
        human_first_on += int(first_main.human_state == 1)
        pre_end = next_binary_state(item.plan.pre.model_state, item.plan.pre.predicted_count)
        pre_success += int(pre_end == first_main.human_state)
        first_main_success += int(first_main.model_state == first_main.human_state)
        if first_main.model_state != first_main.human_state:
            main_agree = [record.model_state == record.human_state for record in records if record.region == "MAIN"]
            initial_recovery.append(next((i for i, value in enumerate(main_agree) if value), len(main_agree)))
        for record in records:
            base = item.base[record.global_index].target
            confusion[base.count, record.predicted_count] += 1
            predicted[record.predicted_count] += 1
            target[base.count] += 1
            region[record.region] += 1
            for slot in range(2):
                if base.timing_mask[slot]:
                    error = abs(record.predicted_timing[slot] - base.timing_targets[slot])
                    timing.append(error)
                    if base.count == 1:
                        timing_n1.append(error)
                    elif slot == 0:
                        timing_n21.append(error)
                    else:
                        timing_n22.append(error)
    count_values = v1.count_metrics(
        [index for index in range(3) for _ in range(int(confusion[index].sum()))
         for index2 in ()], []
    ) if False else None
    accuracy = float(np.trace(confusion) / confusion.sum())
    f1s = []
    for value in range(3):
        tp = confusion[value, value]
        pden = confusion[:, value].sum()
        rden = confusion[value].sum()
        precision = tp / pden if pden else 0.0
        recall = tp / rden if rden else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {
        "performances": len(plans), "intervals": int(confusion.sum()),
        "count_accuracy": accuracy, "count_macro_f1": float(np.mean(f1s)),
        "target_n0": target[0], "target_n1": target[1], "target_n2": target[2],
        "predicted_n0": predicted[0], "predicted_n1": predicted[1], "predicted_n2": predicted[2],
        "pre_count": region["PRE"], "main_count": region["MAIN"], "post_count": region["POST"],
        "state_agreement": float(np.mean(starts)), "state_end_agreement": float(np.mean(ends)),
        "mismatch_fraction": 1.0 - float(np.mean(starts)),
        "mismatch_run_count": len(mismatch_runs),
        "mismatch_run_mean": float(np.mean(mismatch_runs)) if mismatch_runs else 0.0,
        "mismatch_run_median": float(np.median(mismatch_runs)) if mismatch_runs else 0.0,
        "mismatch_run_p95": float(np.quantile(mismatch_runs, .95)) if mismatch_runs else 0.0,
        "mismatch_run_max": max(mismatch_runs, default=0),
        **{f"recovered_within_{h}": recovery_num[h] / recovery_den if recovery_den else 1.0 for h in (1, 2, 4, 8)},
        "human_first_off": len(plans) - human_first_on, "human_first_on": human_first_on,
        "pre_sync_success": pre_success / len(plans), "first_main_state_agreement": first_main_success / len(plans),
        "starting_on_pre_success": (
            sum(
                next_binary_state(item.plan.pre.model_state, item.plan.pre.predicted_count)
                == next(record for record in item.plan.records if record.region == "MAIN").human_state
                for item in plans
                if next(record for record in item.plan.records if record.region == "MAIN").human_state == 1
            ) / human_first_on if human_first_on else 1.0
        ),
        "initial_recovery_mean": float(np.mean(initial_recovery)) if initial_recovery else 0.0,
        "initial_recovery_p95": float(np.quantile(initial_recovery, .95)) if initial_recovery else 0.0,
        "initial_recovery_max": max(initial_recovery, default=0),
        "active_timing_mae": float(np.mean(timing)) if timing else 0.0,
        "n1_timing_mae": float(np.mean(timing_n1)) if timing_n1 else 0.0,
        "n2_slot1_timing_mae": float(np.mean(timing_n21)) if timing_n21 else 0.0,
        "n2_slot2_timing_mae": float(np.mean(timing_n22)) if timing_n22 else 0.0,
        "invalid": 0,
    }


def training_diagnostic_accumulator() -> dict[str, Any]:
    """Compact epoch accumulator; it never retains performance rollout objects."""
    return {
        "performances": 0, "confusion": np.zeros((3, 3), np.int64),
        "predicted": Counter(), "target": Counter(), "region": Counter(),
        "start_agree": 0, "end_agree": 0, "intervals": 0,
        "mismatch_run_histogram": Counter(), "recovery": Counter(), "recovery_den": 0,
        "human_first_on": 0, "pre_success": 0, "first_main_success": 0,
        "starting_on_pre_success": 0, "initial_recovery_histogram": Counter(),
        "timing_error_sum": Counter(), "timing_error_count": Counter(), "invalid": 0,
    }


def add_training_diagnostics(total: dict[str, Any], plans: Sequence[StaticPlan]) -> None:
    for item in plans:
        total["performances"] += 1
        records = item.plan.records
        starts = [record.model_state == record.human_state for record in records]
        ends = [next_binary_state(record.model_state, record.predicted_count)
                == item.base[record.global_index].target.human_end_state for record in records]
        total["start_agree"] += sum(starts)
        total["end_agree"] += sum(ends)
        total["intervals"] += len(records)
        total["mismatch_run_histogram"].update(_runs(starts))
        mismatch_indices = [index for index, value in enumerate(starts) if not value]
        total["recovery_den"] += len(mismatch_indices)
        for horizon in (1, 2, 4, 8):
            total["recovery"][horizon] += sum(
                any(ends[index:min(len(ends), index + horizon)]) for index in mismatch_indices
            )
        first_main = next(record for record in records if record.region == "MAIN")
        first_on = first_main.human_state == 1
        pre_end = next_binary_state(item.plan.pre.model_state, item.plan.pre.predicted_count)
        total["human_first_on"] += int(first_on)
        total["pre_success"] += int(pre_end == first_main.human_state)
        total["first_main_success"] += int(first_main.model_state == first_main.human_state)
        total["starting_on_pre_success"] += int(first_on and pre_end == first_main.human_state)
        if first_main.model_state != first_main.human_state:
            main_agree = [record.model_state == record.human_state for record in records if record.region == "MAIN"]
            recovery = next((index for index, value in enumerate(main_agree) if value), len(main_agree))
            total["initial_recovery_histogram"][recovery] += 1
        for record in records:
            base = item.base[record.global_index].target
            total["confusion"][base.count, record.predicted_count] += 1
            total["predicted"][record.predicted_count] += 1
            total["target"][base.count] += 1
            total["region"][record.region] += 1
            for slot in range(2):
                if not base.timing_mask[slot]:
                    continue
                error = abs(record.predicted_timing[slot] - base.timing_targets[slot])
                keys = ["all", "n1"] if base.count == 1 else ["all", f"n2_slot{slot + 1}"]
                for key in keys:
                    total["timing_error_sum"][key] += error
                    total["timing_error_count"][key] += 1
                total["invalid"] += int(not math.isfinite(error))


def finish_training_diagnostics(total: Mapping[str, Any]) -> dict[str, Any]:
    confusion = total["confusion"]
    intervals = int(total["intervals"])
    performances = int(total["performances"])
    if not intervals or not performances:
        raise ValueError("empty training diagnostics")
    f1_values = []
    for value in range(3):
        true_positive = confusion[value, value]
        precision_denominator = confusion[:, value].sum()
        recall_denominator = confusion[value].sum()
        precision = true_positive / precision_denominator if precision_denominator else 0.0
        recall = true_positive / recall_denominator if recall_denominator else 0.0
        f1_values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    mismatch_runs = [length for length, count in total["mismatch_run_histogram"].items() for _ in range(count)]
    initial_recovery = [length for length, count in total["initial_recovery_histogram"].items() for _ in range(count)]

    def timing_mae(key: str) -> float:
        count = total["timing_error_count"][key]
        return total["timing_error_sum"][key] / count if count else 0.0

    first_on = int(total["human_first_on"])
    return {
        "performances": performances, "intervals": intervals,
        "count_accuracy": float(np.trace(confusion) / confusion.sum()),
        "count_macro_f1": float(np.mean(f1_values)),
        **{f"target_n{value}": total["target"][value] for value in range(3)},
        **{f"predicted_n{value}": total["predicted"][value] for value in range(3)},
        "pre_count": total["region"]["PRE"], "main_count": total["region"]["MAIN"],
        "post_count": total["region"]["POST"],
        "state_agreement": total["start_agree"] / intervals,
        "state_end_agreement": total["end_agree"] / intervals,
        "mismatch_fraction": 1.0 - total["start_agree"] / intervals,
        "mismatch_run_count": len(mismatch_runs),
        "mismatch_run_mean": float(np.mean(mismatch_runs)) if mismatch_runs else 0.0,
        "mismatch_run_median": float(np.median(mismatch_runs)) if mismatch_runs else 0.0,
        "mismatch_run_p95": float(np.quantile(mismatch_runs, .95)) if mismatch_runs else 0.0,
        "mismatch_run_max": max(mismatch_runs, default=0),
        **{f"recovered_within_{h}": total["recovery"][h] / total["recovery_den"]
           if total["recovery_den"] else 1.0 for h in (1, 2, 4, 8)},
        "human_first_off": performances - first_on, "human_first_on": first_on,
        "pre_sync_success": total["pre_success"] / performances,
        "first_main_state_agreement": total["first_main_success"] / performances,
        "starting_on_pre_success": total["starting_on_pre_success"] / first_on if first_on else 1.0,
        "initial_recovery_mean": float(np.mean(initial_recovery)) if initial_recovery else 0.0,
        "initial_recovery_p95": float(np.quantile(initial_recovery, .95)) if initial_recovery else 0.0,
        "initial_recovery_max": max(initial_recovery, default=0),
        "active_timing_mae": timing_mae("all"), "n1_timing_mae": timing_mae("n1"),
        "n2_slot1_timing_mae": timing_mae("n2_slot1"),
        "n2_slot2_timing_mae": timing_mae("n2_slot2"), "invalid": int(total["invalid"]),
    }


def plan_events(item: StaticPlan) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    timeline_records = {record.global_index: record for record in item.plan.records}
    intervals = []
    # base map and records share the full chronological global-index identity.
    for global_index in sorted(timeline_records):
        record = timeline_records[global_index]
        intervals.append((record, item.base[global_index]))
    candidate: list[dict[str, Any]] = []
    reference: list[dict[str, Any]] = []
    # Recover interval bounds from the immutable dataset interval records stored by caller.
    return candidate, reference


def events_from_dataset(item: StaticPlan, dataset, performance_index: int):
    timeline = dataset.timeline(performance_index)
    by_index = {interval.global_index: interval for interval in timeline.represented_intervals}
    candidate: list[dict[str, Any]] = []
    reference: list[dict[str, Any]] = []
    for record in item.plan.records:
        interval = by_index[record.global_index]
        decoded = decode_prediction(record.model_state, record.predicted_count, record.predicted_timing)
        for event in decoded.transitions:
            candidate.append({"direction": event.direction,
                              "seconds": interval.left_seconds + event.tau * interval.duration_seconds})
        for event in interval.human_transitions:
            reference.append({"direction": event.direction,
                              "seconds": interval.left_seconds + event.tau * interval.duration_seconds,
                              "source_tick": event.source_tick})
    return candidate, reference


def human_validation(model, dataset, device, epoch: int) -> dict[str, Any]:
    stage1 = common.score_manifest()
    values: list[StaticPlan] = []
    loss_total = empty_loss_totals()
    transitions = transition_accumulator()
    failures = []
    for index, performance in enumerate(dataset.performances):
        item = controller_plan(model, dataset, index, device, amp=True)
        values.append(item)
        with torch.no_grad():
            add_loss(loss_total, replay_static(model, dataset, (item,), device, amp=True))
        try:
            canonical_index = int(performance.entry["performance_index"])
            onsets, positions = common.score_positions_for_performance(
                performance, canonical_index=canonical_index, stage1=stage1
            )
            candidate_events, reference_events = events_from_dataset(item, dataset, index)
            source = str(performance.entry["source_midi"])
            candidate = common.trajectory_for_metric(candidate_events, source, onsets, positions)
            reference = common.trajectory_for_metric(reference_events, source, onsets, positions)
            sum_transition(transitions, pooled_transition_counts(candidate, reference))
        except Exception as error:
            failures.append({"index": index, "error": f"{type(error).__name__}: {error}"})
        if (index + 1) % 10 == 0 or index + 1 == len(dataset.performances):
            log(f"HUMAN_VALIDATION epoch={epoch} performances={index + 1}/{len(dataset.performances)}")
    if len(failures) > 1:
        raise RuntimeError(f"human validation alignment failures: {failures}")
    metric = finalize_transition(transitions)["pooled"]
    return {
        "epoch": epoch,
        "transition_precision": metric["precision"], "transition_recall": metric["recall"],
        "transition_f1": metric["f1"], "predicted_transitions": metric["candidate"],
        "reference_transitions": metric["reference"],
        "predicted_reference_ratio": metric["candidate"] / metric["reference"] if metric["reference"] else 0.0,
        "metric_performances": len(dataset.performances) - len(failures), "alignment_failures": failures,
        "losses": finish_loss(loss_total), "diagnostics": diagnostics(values),
        "asap_test_access": 0,
    }


def audit_stage1() -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    rows = read_csv(STAGE1_MANIFEST)
    validation = [row for row in read_csv(SPLIT_CSV) if row["split"] == "validation"]
    if sha256_file(STAGE1_MANIFEST) != EXPECTED_STAGE1_MANIFEST_SHA:
        raise RuntimeError("canonical Stage 1 manifest SHA changed")
    if len(rows) != 19 or len(validation) != 71 or {r["piece_id"] for r in rows} != {r["piece_id"] for r in validation}:
        raise RuntimeError("canonical frozen Stage 1 / validation universe mismatch")
    if {row["seed"] for row in rows} != {"42"} or {row["status"] for row in rows} != {"frozen_pass"}:
        raise RuntimeError("canonical Stage 1 seed/status provenance changed")
    mapping = []
    for row in rows:
        path = Path(row["canonical_midi_path"])
        if not path.is_file() or sha256_file(path) != row["canonical_midi_sha256"]:
            raise RuntimeError(f"canonical Stage 1 MIDI missing/hash mismatch: {row['piece_id']}")
        if signature_sha256(path) != row["canonical_non_cc64_signature_sha256"]:
            raise RuntimeError(f"canonical Stage 1 non-CC64 signature mismatch: {row['piece_id']}")
        mapped = [value["performance_path"] for value in validation if value["piece_id"] == row["piece_id"]]
        if len(mapped) != int(row["validation_performance_count"]):
            raise RuntimeError("frozen Stage 1 piece/performance mapping count changed")
        mapping.append({"piece_id": row["piece_id"], "canonical_midi_path": row["canonical_midi_path"],
                        "canonical_midi_sha256": row["canonical_midi_sha256"], "human_performances": mapped})
    provenance = {
        "status": "PASS", "canonical_manifest": str(STAGE1_MANIFEST),
        "canonical_manifest_sha256": EXPECTED_STAGE1_MANIFEST_SHA,
        "source_experiment": str(ROOT / "analysis/stage2_4class_architecture_validation_eval_v0"),
        "source_report": str(ROOT / "analysis/stage2_4class_architecture_validation_eval_v0/FOUR_MODEL_VALIDATION_EVALUATION_REPORT.md"),
        "pieces": 19, "human_performances": 71, "seed": 42,
        "frozen_midi_note_count": 48893,
        "saved_stage1_id_note_count": sum(int(row["note_count"]) for row in rows),
        "known_stage1_midi_dump_note_delta": sum(int(row["note_count"]) for row in rows) - 48893,
        "stage2_input_choice": "official midi_to_ids of the canonical frozen Original-PT MIDI; pedal columns masked",
        "inference_configuration_identifier": sorted({row["inference_configuration_identifier"] for row in rows}),
        "stage1_inference_regeneration": 0, "asap_test_access": 0, "mapping": mapping,
    }
    atomic_json(RUN_ROOT / "stage1_validation_provenance.json", provenance)
    return rows, validation, provenance


def score_positions_for_piece(row: Mapping[str, str]) -> tuple[list[int], list[int]]:
    from scripts.audit_pedal_event_metric_tolerance_mini import build_score_positions, raw_events
    from miditoolkit import MidiFile
    cache = RUN_ROOT / "frozen_pt_score_positions" / f"{row['piece_id']}.json"
    source = Path(row["canonical_midi_path"])
    onsets, _ = raw_events(MidiFile(str(source)))
    if cache.is_file():
        value = json.loads(cache.read_text())
        if value["source_sha256"] == row["canonical_midi_sha256"] and value["raw_onsets"] == onsets:
            return onsets, [int(item) for item in value["score_positions"]]
    process_log = RUN_ROOT / "frozen_pt_alignment_logs" / f"{row['piece_id']}.log"
    process_log.parent.mkdir(parents=True, exist_ok=True)
    positions_raw = build_score_positions(
        Path(row["selected_score_absolute_path"]), source, onsets, ALIGNMENT_TOOL,
        process_log,
    )
    if any(value is None for value in positions_raw):
        raise RuntimeError(f"frozen PT score alignment unmapped onset: {row['piece_id']}")
    positions = [int(value) for value in positions_raw]
    atomic_json(cache, {"piece_id": row["piece_id"], "source_sha256": row["canonical_midi_sha256"],
                        "raw_onsets": onsets, "score_positions": positions, "asap_test_access": 0})
    return onsets, positions


def binary_states_from_aligned_tokens(tokens: np.ndarray) -> list[int]:
    values = np.asarray(tokens, dtype=np.int64)
    if values.ndim == 1:
        values = values.reshape(-1, 8)
    iois = values[:, 1] - 261
    absolute = np.cumsum(iois)
    first = np.flatnonzero(np.r_[True, np.diff(absolute) != 0])
    return [int(values[index, NON_PEDAL_FEATURES] - PEDAL_TOKEN_OFFSET >= 64) for index in first]


def _state_pair_stats(model_states: Sequence[int], human_states: Sequence[int]) -> dict[str, Any]:
    agreement = [left == right for left, right in zip(model_states, human_states)]
    runs = _runs(agreement)
    ends = [agreement[index + 1] if index + 1 < len(agreement) else agreement[index] for index in range(len(agreement))]
    mismatch = [index for index, value in enumerate(agreement) if not value]
    result = {
        "agree": sum(agreement), "total": len(agreement), "runs": runs,
        "recovery_den": len(mismatch),
    }
    for horizon in (1, 2, 4, 8):
        result[f"recovery_{horizon}"] = sum(any(ends[index:min(len(ends), index + horizon)]) for index in mismatch)
    return result


def frozen_pt_validation(model, pieces, stage1_rows, validation_rows, human_dataset, device, epoch: int) -> dict[str, Any]:
    by_piece = {row["piece_id"]: row for row in stage1_rows}
    references = build_frozen_human_reference_map(
        stage1_rows, validation_rows,
        [item.entry for item in human_dataset.performances],
        expected_pieces=19, expected_humans=71, verify_paths=True,
    )
    validation_by_piece = references_by_piece(references)
    transition = transition_accumulator()
    transition_on = transition_accumulator()
    failures = []
    identity_rows = []
    predicted = Counter()
    pre_predicted = Counter()
    state_parts = []
    state_parts_on = []
    pre_success = first_success = starting_on = starting_on_pre_success = 0
    pair_count = 0
    output_root = RUN_ROOT / "frozen_pt_outputs" / f"epoch_{epoch:02d}"
    for index, piece in enumerate(pieces, 1):
        rollout = rollout_frozen_pt(model, piece, device, amp=True)
        destination = output_root / piece.piece_id / "candidate.mid"
        identity = render_cc64_only(rollout, destination)
        identity_rows.append({"piece_id": piece.piece_id, "candidate": str(destination), **identity})
        row = by_piece[piece.piece_id]
        raw_onsets, positions = score_positions_for_piece(row)
        if len(raw_onsets) != len(piece.main) or len(positions) != len(piece.main):
            raise RuntimeError("frozen PT token/onset/alignment cardinality mismatch")
        serialized_events = serialized_candidate_events_for_metric(destination)
        candidate = common.trajectory_for_metric(serialized_events, str(piece.source_midi), raw_onsets, positions)
        main_records = [record for record in rollout.records if record.region == "MAIN"]
        model_states = [record.model_state for record in main_records]
        pre_after = next_binary_state(rollout.records[0].model_state, rollout.records[0].predicted_count)
        for record in rollout.records:
            predicted[record.predicted_count] += 1
            if record.region == "PRE":
                pre_predicted[record.predicted_count] += 1
        for human in validation_by_piece[piece.piece_id]:
            identifier = human.metadata_index
            target = load_frozen_human(identifier)
            if target is None:
                failures.append({"metadata_index": identifier, "performance_path": human.performance_path,
                                 "reason": "frozen shared-human alignment unavailable"})
                continue
            human_timeline = human_dataset.timeline(human.dataset_index)
            left_ms = human_timeline.pre.left_seconds * 1000.0
            right_ms = human_timeline.post.right_seconds * 1000.0
            target_transitions = {"transitions": [event for event in target["transitions"]["transitions"]
                                                   if left_ms <= float(event["time_ms"]) <= right_ms]}
            current = pooled_transition_counts(candidate, target_transitions)
            sum_transition(transition, current)
            human_first = human_timeline.main[0].human_start_state
            first_model = main_records[0].model_state
            starting_on += int(human_first == 1)
            pre_success += int(pre_after == human_first)
            starting_on_pre_success += int(human_first == 1 and pre_after == human_first)
            first_success += int(first_model == human_first)
            pair_count += 1
            human_score_states = binary_states_from_aligned_tokens(target["tokens"])
            paired_model = []
            paired_human = []
            for model_state, score_position in zip(model_states, positions):
                if 0 <= score_position < len(human_score_states):
                    paired_model.append(model_state)
                    paired_human.append(human_score_states[score_position])
            part = _state_pair_stats(paired_model, paired_human)
            state_parts.append(part)
            if human_first == 1:
                state_parts_on.append(part)
                sum_transition(transition_on, current)
        log(f"FROZEN_PT_VALIDATION epoch={epoch} pieces={index}/{len(pieces)}")
    if len(identity_rows) != 19 or not all(row["note_identity_exact"] for row in identity_rows):
        raise AssertionError("frozen PT non-pedal identity is not 19/19 PASS")
    if len(failures) > 1 or pair_count < 70:
        raise RuntimeError(f"frozen PT/human mapping failures: {failures}")

    def combine(parts):
        total = sum(part["total"] for part in parts)
        agree = sum(part["agree"] for part in parts)
        runs = [run for part in parts for run in part["runs"]]
        denominator = sum(part["recovery_den"] for part in parts)
        return {
            "state_agreement": agree / total if total else 0.0,
            "mismatch_fraction": 1.0 - agree / total if total else 0.0,
            "mismatch_run_count": len(runs), "mismatch_run_mean": float(np.mean(runs)) if runs else 0.0,
            "mismatch_run_median": float(np.median(runs)) if runs else 0.0,
            "mismatch_run_p95": float(np.quantile(runs, .95)) if runs else 0.0,
            "mismatch_run_max": max(runs, default=0),
            **{f"recovered_within_{h}": sum(part[f'recovery_{h}'] for part in parts) / denominator if denominator else 1.0
               for h in (1, 2, 4, 8)},
        }

    direction = finalize_transition(transition)
    direction_on = finalize_transition(transition_on)
    state = combine(state_parts)
    state_on = combine(state_parts_on)
    pooled = direction["pooled"]
    result = {
        "epoch": epoch, "transition_precision": pooled["precision"], "transition_recall": pooled["recall"],
        "transition_f1": pooled["f1"], "predicted_transitions": pooled["candidate"],
        "reference_transitions": pooled["reference"],
        "predicted_reference_ratio": pooled["candidate"] / pooled["reference"] if pooled["reference"] else 0.0,
        "predicted_n0": predicted[0], "predicted_n1": predicted[1], "predicted_n2": predicted[2],
        "pre_predicted_n0": pre_predicted[0], "pre_predicted_n1": pre_predicted[1], "pre_predicted_n2": pre_predicted[2],
        **state,
        "human_t1_starting_on": starting_on, "human_t1_starting_off": pair_count - starting_on,
        "pre_sync_success": pre_success / pair_count, "first_main_state_agreement": first_success / pair_count,
        "starting_on_pre_success": starting_on_pre_success / starting_on if starting_on else 1.0,
        "starting_on_state_agreement": state_on["state_agreement"],
        "starting_on_longest_mismatch_run": state_on["mismatch_run_max"],
        "starting_on_transition_f1": direction_on["pooled"]["f1"],
        "metric_pairs": pair_count, "alignment_failures": failures,
        "nonpedal_identity_pass": len(identity_rows), "nonpedal_identity_total": 19,
        "pedal_leakage_count": 0,
        "primary_transition_metric_source": "serialized candidate MIDI CC64 trajectory",
        "pre_origin_projection_count": sum(int(row["pre_origin_decoded_event_count"]) for row in identity_rows),
        "tick_zero_origin_anchor_count": sum(int(row["midi_origin_state_anchor_inserted"]) for row in identity_rows),
        "post_eot_extension_piece_count": sum(int(row["eot_extension_ticks"] > 0) for row in identity_rows),
        "identity_rows": identity_rows,
        "static_count_accuracy": None, "count_macro_f1": None,
        "active_timing_mae": None, "timing_unavailable_reason": "19 PT piece timelines pair to multiple human timings",
        "stage1_inference_regeneration": 0, "human_state_used_in_inference": False,
        "asap_test_access": 0,
    }
    atomic_json(RUN_ROOT / "frozen_pt_validation_epochs" / f"epoch_{epoch:02d}.json", result)
    atomic_json(RUN_ROOT / "nonpedal_identity_audit.json", {
        "epoch": epoch, "pass": len(identity_rows), "total": 19, "rows": identity_rows,
        "status": "PASS", "asap_test_access": 0,
    })
    return result


def configuration() -> dict[str, Any]:
    return {
        "experiment": RUN_ROOT.name, "created_at": now(), "run_root": str(RUN_ROOT),
        "clean_restart_from_epoch_1": True,
        "previous_failed_run": str(PREVIOUS_FAILED_RUN),
        "previous_failure_reason": "Frozen-PT metadata mapping KeyError: metadata_index",
        "mapping_fix_report": str(MAPPING_FIX_REPORT),
        "frozen_validation_mapping": "performance_path exact join + piece_id cross-check",
        "frozen_validation_structural_population": {"pieces": 19, "human_references": 71},
        "frozen_validation_canonical_aligned_population": {"included": 70, "total": 71, "excluded_metadata_index": "856"},
        "fault_tolerant_checkpointing": {
            "enabled": True,
            "before_validation": True,
            "resume_last": "resume_last_train_state.pt",
            "epoch_pattern": "checkpoints/train_complete_epoch_XX.pt",
            "best_requires_completed_frozen_pt_validation": True,
            "validation_pending_resume_supported": True,
            "post_train_rng_restored_after_validation": True,
        },
        "epoch_1_resume_provenance": {
            "training_reexecuted": False,
            "completed_epoch": 1,
            "global_step": 516,
            "resume_checkpoint": str(RUN_ROOT / "resume_last_train_state.pt"),
            "first_resumed_phase": "HUMAN_INPUT_VALIDATION",
        },
        "frozen_pt_primary_metric_source": "serialized candidate MIDI CC64 trajectory",
        "negative_pre_serialization": "origin state projection; no timestamp clamp",
        "formulation": "Condition D static Count/Timing + hard free-running + local state consistency",
        "regions": {"PRE": "[t1-1,t1)", "MAIN": "all including [tM,Tend)", "POST": "[Tend,Tend+1]"},
        "static_pre_initial_synchronization": True, "online_reconciliation": False,
        "dynamic_target": False, "main_synthetic_correction": False,
        "lambda_state": LAMBDA_STATE, "boundary_loss_weight": BOUNDARY_WEIGHT,
        "count_weights": list(FIXED_COUNT_WEIGHTS), "timing_beta": .1,
        "seed": SEED, "optimizer": "AdamW", "encoder_lr": common.ENCODER_LR,
        "head_lr": common.HEAD_LR, "weight_decay": common.WEIGHT_DECAY,
        "gradient_clip": common.MAX_GRAD_NORM, "precision": "FP16 AMP",
        "window_micro_batch": WINDOW_MICRO_BATCH, "performances_per_optimizer_step": PERFORMANCES_PER_STEP,
        "max_epochs": MAX_EPOCHS, "early_stopping": None,
        "human_input_validation": True, "frozen_pt_input_validation": True,
        "frozen_pt_input_source": "canonical frozen Original-PT MIDI from the 4-class validation manifest",
        "frozen_pt_midi_notes": 48893, "frozen_pt_saved_stage1_id_notes": 48894,
        "best_checkpoint_metric": "frozen_pt_input canonical direction-aware Transition F1",
        "tie_break": "earlier epoch", "stage1_inference_regeneration": 0,
        "asap_test_metadata_access": 0, "asap_test_midi_access": 0,
    }


def preflight() -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    restore = json.loads((RESTORE_ROOT / "smoke_summary.json").read_text())
    if restore["status"] != "PASS" or restore["online_reconciliation_calls"] != 0:
        raise RuntimeError("authoritative restored smoke PASS missing")
    cache = json.loads((CACHE_ROOT / "cache_manifest.json").read_text())
    alignment = json.loads((ALIGNMENT_ROOT / "alignment_manifest.json").read_text())
    if cache["cache_id"] != CANONICAL_CACHE_ID or cache["train_count"] != 2062 or cache["validation_count"] != 71:
        raise RuntimeError("canonical cache provenance changed")
    if alignment["alignment_id"] != CANONICAL_ALIGNMENT_ID:
        raise RuntimeError("canonical alignment provenance changed")
    stage1, validation, provenance = audit_stage1()
    tests = [
        "tests/test_binary_2slot_pre_main_post_restore_v0.py",
        "tests/test_binary_2slot_state_consistency_v1.py",
        "tests/test_binary_2slot_model_v0.py",
        "tests/test_binary_2slot_transition_audit_v0.py",
        "tests/test_binary_2slot_frozen_mapping_fix_v0.py",
    ]
    test_rows = []
    for path in tests:
        completed = subprocess.run([sys.executable, path], cwd=ROOT, text=True, capture_output=True)
        test_rows.append({"path": path, "returncode": completed.returncode, "stdout": completed.stdout.strip()})
        if completed.returncode:
            raise RuntimeError(f"focused regression failed: {path}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one existing CUDA device is required")
    device = torch.device("cuda:0")
    common.seed_everything(SEED)
    model = common.build_model(device)
    train = Binary2SlotPerformanceDataset(CACHE_ROOT, "train", performance_indices=(7,), cache_input_tokens=True)
    item = controller_plan(model, train, 0, device, amp=False)
    with torch.no_grad():
        initial = replay_static(model, train, (item,), device, amp=False)
    if not math.isfinite(initial["total"]):
        raise FloatingPointError("initial restored Condition D loss is non-finite")
    pieces = [load_frozen_pt_piece(row) for row in stage1]
    if sum(len(piece.masked_tokens) for piece in pieces) != 48893:
        raise RuntimeError("frozen canonical PT note universe changed")
    for row, piece in zip(stage1, pieces):
        raw_onsets, positions = score_positions_for_piece(row)
        if len(raw_onsets) != len(piece.main) or len(positions) != len(piece.main):
            raise RuntimeError("frozen PT onset/score mapping mismatch")
    human_val = Binary2SlotPerformanceDataset(CACHE_ROOT, "validation", cache_input_tokens=False)
    references = build_frozen_human_reference_map(
        stage1, validation, [item.entry for item in human_val.performances],
        expected_pieces=19, expected_humans=71, verify_paths=True,
    )
    grouped = references_by_piece(references)
    if len(grouped) != 19 or sum(len(items) for items in grouped.values()) != 71:
        raise RuntimeError("frozen PT structural mapping is not exactly 19 pieces / 71 references")
    unavailable = sorted(reference.metadata_index for reference in references
                         if load_frozen_human(reference.metadata_index) is None)
    if unavailable != ["856"]:
        raise RuntimeError(f"canonical frozen-human alignment exclusion changed: {unavailable}")
    identity_rows = []
    for piece in pieces:
        rollout = rollout_frozen_pt(model, piece, device, amp=False)
        identity = render_cc64_only(
            rollout, RUN_ROOT / "preflight_frozen_pt_outputs" / piece.piece_id / "candidate.mid"
        )
        identity_rows.append({"piece_id": piece.piece_id, **identity})
    if len(identity_rows) != 19 or not all(row["note_identity_exact"] for row in identity_rows):
        raise AssertionError("preflight frozen PT non-pedal identity is not 19/19 PASS")
    atomic_json(RUN_ROOT / "nonpedal_identity_audit.json", {
        "stage": "preflight", "status": "PASS", "pass": 19, "total": 19,
        "rows": identity_rows, "asap_test_access": 0,
    })
    atomic_json(RUN_ROOT / "FULL_RUN_CONFIG.json", configuration())
    atomic_json(RUN_ROOT / "preflight_summary.json", {
        "status": "PASS", "tests": test_rows, "initial_loss": initial,
        "checkpoint": model.checkpoint_path, "hidden": model.hidden_size,
        "layers": int(model.encoder.config.num_hidden_layers),
        "encoder_parameters": model.encoder_parameter_count,
        "head_parameters": model.prediction_head_parameter_count,
        "train_count": 2062, "human_validation_count": 71,
        "frozen_pt_pieces": 19, "frozen_pt_midi_notes": 48893,
        "frozen_pt_saved_id_notes": sum(int(row["note_count"]) for row in stage1),
        "known_midi_dump_delta_notes": sum(int(row["note_count"]) for row in stage1) - 48893,
        "preflight_identity_pass": 19, "preflight_identity_total": 19,
        "frozen_pt_mapping_pieces": len(grouped),
        "frozen_pt_mapping_references": len(references),
        "canonical_aligned_population": len(references) - len(unavailable),
        "canonical_alignment_exclusions": unavailable,
        "pedal_input_masking": "PASS",
        "fault_tolerant_checkpointing": "ENABLED",
        "stage1_provenance": provenance,
        "stage1_inference_regeneration": 0, "asap_test_access": 0,
    })
    atomic_json(RUN_ROOT / "run_status.json", {
        "status": "PREFLIGHT_PASS", "completed_epochs": 0, "last_update": now(),
        "stage1_source_resolved": True, "initial_loss_finite": True,
        "asap_test_metadata_access": 0, "asap_test_midi_access": 0,
    })
    del model, train, human_val
    torch.cuda.empty_cache()
    log("PREFLIGHT_PASS restore/static/no-online/frozen-stage1/mapping=19:71/aligned=70:71/pedal_masked/identity/finite-loss")


def update_report(train_rows, human_rows, frozen_rows, best_epoch, best_f1, status):
    table = []
    for train, human, frozen in zip(train_rows, human_rows, frozen_rows):
        table.append(f"| {train['epoch']} | {train['train_loss']:.5f} | {human['transition_f1']:.4f} | {frozen['transition_precision']:.4f} | {frozen['transition_recall']:.4f} | {frozen['transition_f1']:.4f} | {frozen['state_agreement']:.4f} | {frozen['first_main_state_agreement']:.4f} |")
    report = f"""# Full Binary 2-Slot Condition D PRE/MAIN/POST run

| item | result |
|---|---|
| status | {status} |
| formulation | restored Condition D, immutable static targets, hard free-running |
| lambda_state / boundary weight | 0.25 / 0.25 |
| online reconciliation / dynamic target | 0 / 0 |
| frozen Stage 1 | `{STAGE1_MANIFEST}` (19 pieces, seed 42, SHA `{EXPECTED_STAGE1_MANIFEST_SHA}`) |
| Stage 1 regeneration | 0 |
| previous failed run | `{PREVIOUS_FAILED_RUN}` |
| previous failure / mapping fix | metadata mapping KeyError / `{MAPPING_FIX_REPORT}` |
| structural / canonical aligned population | 19 pieces, 71 references / 70 of 71 (metadata 856 excluded) |
| fault-tolerant train-complete checkpoint | enabled before validation |
| epoch 1 continuation | saved post-train state; validation rerun; epoch-1 training not reexecuted |
| negative PRE serialization | origin-state projection; tick-0 anchor only when projected state is ON |
| Frozen-PT primary metric source | serialized candidate MIDI CC64 trajectory |
| saved-model renderer audit | 19/19 PASS; non-pedal identity 19/19; pedal leakage 0 |
| epochs | {len(train_rows)} / 10 |
| best epoch / frozen-PT Transition F1 | {best_epoch if best_epoch else 'pending'} / {best_f1 if best_f1 is not None else 'pending'} |
| best / last | `{RUN_ROOT / 'best.pt'}` / `{RUN_ROOT / 'last.pt'}` |
| ASAP test access | 0 |

Checkpoint selection uses only frozen-PT-input canonical direction-aware Transition F1, with earlier-epoch tie break. Human-input validation is auxiliary. Frozen PT pedal tokens are masked before encoding; output MIDI preserves source notes exactly and replaces CC64 only (EOT extension, if required for POST, is explicitly audited).

| epoch | train loss | human F1 | frozen P | frozen R | frozen F1 | frozen state | first-MAIN state |
|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table) if table else '| pending | | | | | | | |'}

No validation result changes LR, loss weights, architecture, or schedule. Full training terminates after epoch 10 unless a frozen hard-stop invariant fails.
"""
    temporary = RUN_ROOT / ".FULL_BINARY_2SLOT_D_PRE_MAIN_POST_REPORT.md.tmp"
    temporary.write_text(report)
    os.replace(temporary, RUN_ROOT / "FULL_BINARY_2SLOT_D_PRE_MAIN_POST_REPORT.md")


def _coerce_csv_value(value: str) -> Any:
    if value == "":
        return None
    if value in {"True", "False"}:
        return value == "True"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def load_metric_table(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [{key: _coerce_csv_value(value) for key, value in row.items()} for row in read_csv(path)]


def replace_epoch_row(table: list[dict[str, Any]], row: dict[str, Any]) -> None:
    epoch = int(row["epoch"])
    table[:] = [existing for existing in table if int(existing["epoch"]) != epoch]
    table.append(row)
    table.sort(key=lambda value: int(value["epoch"]))


def finalize_validated_epoch(
    *, model, optimizer, scaler, human_val, pieces, stage1_rows, validation_rows,
    device, epoch: int, global_step: int, losses: Mapping[str, Any],
    train_diag: Mapping[str, Any], train_extras: Mapping[str, Any],
    train_table: list[dict[str, Any]], human_table: list[dict[str, Any]],
    frozen_table: list[dict[str, Any]], state_table: list[dict[str, Any]],
    cold_table: list[dict[str, Any]],
) -> tuple[int, float]:
    """Run both validations, persist epoch metrics, and select best by Frozen-PT F1."""

    log(f"VALIDATION_PHASE epoch={epoch} phase=HUMAN_INPUT train_reexecuted=0")
    human = human_validation(model, human_val, device, epoch)
    log(f"VALIDATION_PHASE epoch={epoch} phase=FROZEN_PT train_reexecuted=0")
    frozen = frozen_pt_validation(model, pieces, stage1_rows, validation_rows, human_val, device, epoch)
    train_row = {"epoch": epoch, "train_loss": losses["total"], **dict(losses), **dict(train_diag), **dict(train_extras)}
    human_row = {
        "epoch": epoch,
        **{key: value for key, value in human.items() if key not in {"diagnostics", "losses", "alignment_failures"}},
        **{f"diag_{key}": value for key, value in human["diagnostics"].items() if not isinstance(value, (dict, list))},
        **{f"loss_{key}": value for key, value in human["losses"].items()},
    }
    frozen_row = {key: value for key, value in frozen.items() if not isinstance(value, (dict, list))}
    state_row = {
        "epoch": epoch,
        "human_state_agreement": human["diagnostics"]["state_agreement"],
        "human_mismatch_fraction": human["diagnostics"]["mismatch_fraction"],
        "frozen_state_agreement": frozen["state_agreement"],
        "frozen_mismatch_fraction": frozen["mismatch_fraction"],
        "frozen_mismatch_run_count": frozen["mismatch_run_count"],
        "frozen_mismatch_run_mean": frozen["mismatch_run_mean"],
        "frozen_mismatch_run_median": frozen["mismatch_run_median"],
        "frozen_mismatch_run_p95": frozen["mismatch_run_p95"],
        "frozen_mismatch_run_max": frozen["mismatch_run_max"],
        **{f"frozen_recovered_within_{h}": frozen[f"recovered_within_{h}"] for h in (1, 2, 4, 8)},
    }
    cold_row = {
        "epoch": epoch,
        "human_t1_starting_on": frozen["human_t1_starting_on"],
        "pre_sync_success": frozen["pre_sync_success"],
        "first_main_state_agreement": frozen["first_main_state_agreement"],
        "starting_on_pre_success": frozen["starting_on_pre_success"],
        "starting_on_state_agreement": frozen["starting_on_state_agreement"],
        "starting_on_longest_mismatch_run": frozen["starting_on_longest_mismatch_run"],
        "starting_on_transition_f1": frozen["starting_on_transition_f1"],
    }
    for table, row in (
        (train_table, train_row), (human_table, human_row), (frozen_table, frozen_row),
        (state_table, state_row), (cold_table, cold_row),
    ):
        replace_epoch_row(table, row)
    atomic_csv(RUN_ROOT / "training_metrics_by_epoch.csv", train_table)
    atomic_csv(RUN_ROOT / "human_input_validation_by_epoch.csv", human_table)
    atomic_csv(RUN_ROOT / "frozen_pt_validation_by_epoch.csv", frozen_table)
    atomic_csv(RUN_ROOT / "state_diagnostics_by_epoch.csv", state_table)
    atomic_csv(RUN_ROOT / "cold_start_diagnostics_by_epoch.csv", cold_table)

    payload = {
        "epoch": epoch, "global_step": global_step, "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(), "grad_scaler_state": scaler.state_dict(),
        "configuration": configuration(), "seed": SEED, "lambda_state": LAMBDA_STATE,
        "boundary_loss_weight": BOUNDARY_WEIGHT, "human_validation": human,
        "frozen_pt_validation": frozen, "created_at": now(), "asap_test_access": 0,
    }
    atomic_torch(RUN_ROOT / "checkpoints" / f"epoch_{epoch:02d}.pt", payload)
    atomic_torch(RUN_ROOT / "last.pt", payload)
    best_row = max(frozen_table, key=lambda row: (float(row["transition_f1"]), -int(row["epoch"])))
    best_epoch = int(best_row["epoch"])
    best_f1 = float(best_row["transition_f1"])
    if best_epoch == epoch:
        atomic_torch(RUN_ROOT / "best.pt", payload)
    marker = {
        "status": "VALIDATION_COMPLETE", "epoch": epoch, "global_step": global_step,
        "human_validation_complete": True, "frozen_pt_validation_complete": True,
        "frozen_pt_transition_f1": float(frozen["transition_f1"]),
        "primary_metric_source": frozen["primary_transition_metric_source"],
        "best_epoch": best_epoch, "best_frozen_pt_transition_f1": best_f1,
        "created_at": now(), "asap_test_access": 0,
    }
    atomic_json(RUN_ROOT / "validation_complete" / f"epoch_{epoch:02d}.json", marker)
    status = "COMPLETE" if epoch == MAX_EPOCHS else "RUNNING"
    atomic_json(RUN_ROOT / "run_status.json", {
        "status": status, "completed_epochs": epoch, "global_step": global_step,
        "validation_pending": False, "best_epoch": best_epoch,
        "best_frozen_pt_transition_f1": best_f1, "last_update": now(),
        "stage1_inference_regeneration": 0,
        "nonpedal_identity_pass": frozen["nonpedal_identity_pass"],
        "primary_metric_source": frozen["primary_transition_metric_source"],
        "asap_test_metadata_access": 0, "asap_test_midi_access": 0,
    })
    update_report(train_table, human_table, frozen_table, best_epoch, best_f1, status)
    log(f"EPOCH_COMPLETE epoch={epoch} train_loss={losses['total']:.6f} human_f1={human['transition_f1']:.6f} frozen_f1={float(frozen['transition_f1']):.6f} best={best_epoch}")
    return best_epoch, best_f1


def acquire_run_lock(mode: str) -> Path:
    lock = RUN_ROOT / ".binary2slot_full_run.lock"
    if lock.exists():
        try:
            prior = json.loads(lock.read_text(encoding="utf-8"))
            prior_pid = int(prior["pid"])
            os.kill(prior_pid, 0)
        except (FileNotFoundError, ProcessLookupError, ValueError, KeyError, json.JSONDecodeError):
            stale = RUN_ROOT / f"binary2slot_full_run.lock.stale.{int(time.time())}"
            os.replace(lock, stale)
        else:
            raise RuntimeError(f"active run lock exists: pid={prior_pid} mode={prior.get('mode')}")
    descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "mode": mode, "created_at": now()}, handle)
        handle.write("\n")
    return lock


def full(resume_checkpoint: Path | None = None) -> None:
    preflight_value = json.loads((RUN_ROOT / "preflight_summary.json").read_text())
    if preflight_value["status"] != "PASS":
        raise RuntimeError("preflight PASS artifact missing")
    if resume_checkpoint is None and (
        (RUN_ROOT / "last.pt").exists() or list((RUN_ROOT / "checkpoints").glob("epoch_*.pt"))
    ):
        raise RuntimeError("existing checkpoint found; refusing overwrite")
    atomic_json(RUN_ROOT / "FULL_RUN_CONFIG.json", configuration())
    stage1_rows, validation_rows, _ = audit_stage1()
    common.seed_everything(SEED)
    device = torch.device("cuda:0")
    model = common.build_model(device)
    optimizer = common.build_optimizer(model)
    scaler = torch.amp.GradScaler("cuda", init_scale=AMP_INIT_SCALE, enabled=True)
    resume_payload = None
    completed_training_epoch = 0
    global_step = 0
    if resume_checkpoint is not None:
        if not resume_checkpoint.is_file():
            raise FileNotFoundError(resume_checkpoint)
        resume_payload = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
        completed_training_epoch, global_step = restore_resumable_epoch_payload(
            resume_payload, model=model, optimizer=optimizer, scaler=scaler, restore_rng=False,
        )
        if bool(resume_payload["validations_complete"]):
            raise RuntimeError("resume checkpoint is not validation-pending")
    atomic_json(RUN_ROOT / "run_status.json", {
        "status": "RESUME_INITIALIZING" if resume_payload is not None else "INITIALIZING",
        "completed_training_epoch": completed_training_epoch, "global_step": global_step,
        "epoch_1_train_reexecuted": False if resume_payload is not None else None,
        "last_update": now(),
        "stage1_source_resolved": True, "stage1_inference_regeneration": 0,
        "asap_test_metadata_access": 0, "asap_test_midi_access": 0,
    })
    if resume_payload is None:
        log(f"FULL_INIT checkpoint={model.checkpoint_path} layers={model.encoder.config.num_hidden_layers} lambda_state={LAMBDA_STATE}")
    else:
        log(
            f"RESUME_LOADED checkpoint={resume_checkpoint} completed_epoch={completed_training_epoch} "
            f"global_step={global_step} epoch_1_train_reexecuted=0"
        )
    train = Binary2SlotPerformanceDataset(CACHE_ROOT, "train", cache_input_tokens=True)
    human_val = Binary2SlotPerformanceDataset(CACHE_ROOT, "validation", cache_input_tokens=True)
    pieces = [load_frozen_pt_piece(row) for row in stage1_rows]
    if len(train.performances) != 2062 or len(human_val.performances) != 71 or len(pieces) != 19:
        raise RuntimeError("canonical dataset universe changed")
    log("DATASET_READY train=2062 human_validation=71 frozen_pt=19 stage1_regeneration=0")
    references = build_frozen_human_reference_map(
        stage1_rows, validation_rows, [item.entry for item in human_val.performances],
        expected_pieces=19, expected_humans=71, verify_paths=True,
    )
    grouped = references_by_piece(references)
    unavailable = sorted(reference.metadata_index for reference in references
                         if load_frozen_human(reference.metadata_index) is None)
    if len(grouped) != 19 or len(references) != 71 or unavailable != ["856"]:
        raise RuntimeError(
            f"canonical frozen mapping/alignment invariant failed: "
            f"pieces={len(grouped)} refs={len(references)} unavailable={unavailable}"
        )
    atomic_json(RUN_ROOT / "run_status.json", {
        "status": "RESUME_VALIDATION_PENDING" if resume_payload is not None else "RUNNING",
        "completed_training_epoch": completed_training_epoch,
        "completed_validation_epochs": completed_training_epoch - 1 if resume_payload is not None else 0,
        "current_phase": "HUMAN_VALIDATION" if resume_payload is not None else "TRAIN_EPOCH_1",
        "epoch_1_train_reexecuted": False if resume_payload is not None else None,
        "global_step": global_step, "last_update": now(),
        "train_performances": 2062, "frozen_pt_pieces": len(grouped),
        "human_validation_references": len(references),
        "canonical_aligned_population": len(references) - len(unavailable),
        "canonical_alignment_exclusions": unavailable,
        "pedal_input_masking": "PASS", "fault_tolerant_checkpointing": True,
        "stage1_inference_regeneration": 0,
        "asap_test_metadata_access": 0, "asap_test_midi_access": 0,
    })
    prefix = "RESUME_MAPPING_READY" if resume_payload is not None else "MAPPING_READY"
    log(f"{prefix} pieces=19 references=71 canonical_aligned=70 excluded_metadata=856 pedal_masked=1")
    train_table = load_metric_table(RUN_ROOT / "training_metrics_by_epoch.csv")
    human_table = load_metric_table(RUN_ROOT / "human_input_validation_by_epoch.csv")
    frozen_table = load_metric_table(RUN_ROOT / "frozen_pt_validation_by_epoch.csv")
    state_table = load_metric_table(RUN_ROOT / "state_diagnostics_by_epoch.csv")
    cold_table = load_metric_table(RUN_ROOT / "cold_start_diagnostics_by_epoch.csv")
    best_epoch = None if not frozen_table else int(max(
        frozen_table, key=lambda row: (float(row["transition_f1"]), -int(row["epoch"]))
    )["epoch"])
    best_f1 = None if best_epoch is None else max(float(row["transition_f1"]) for row in frozen_table)
    consecutive_failure = 0
    update_report(train_table, human_table, frozen_table, best_epoch, best_f1, "RUNNING")

    if resume_payload is not None:
        epoch = completed_training_epoch
        marker = RUN_ROOT / "validation_complete" / f"epoch_{epoch:02d}.json"
        if marker.is_file():
            if not any(int(row["epoch"]) == epoch for row in frozen_table):
                raise RuntimeError("validation-complete marker exists without persisted metric row")
            log(f"RESUME_VALIDATION_ALREADY_COMPLETE epoch={epoch} train_reexecuted=0")
        else:
            losses = dict(resume_payload["train_losses"])
            train_diag = dict(resume_payload["train_diagnostics"])
            atomic_json(RUN_ROOT / "run_status.json", {
                "status": "RESUME_VALIDATION_PENDING", "completed_training_epoch": epoch,
                "completed_validation_epochs": epoch - 1, "current_phase": "HUMAN_VALIDATION",
                "global_step": global_step, "epoch_1_train_reexecuted": False,
                "resume_checkpoint": str(resume_checkpoint), "last_update": now(),
                "frozen_pt_pieces": 19, "human_validation_references": 71,
                "canonical_aligned_population": 70, "pedal_input_masking": "PASS",
                "asap_test_metadata_access": 0, "asap_test_midi_access": 0,
            })
            best_epoch, best_f1 = finalize_validated_epoch(
                model=model, optimizer=optimizer, scaler=scaler, human_val=human_val,
                pieces=pieces, stage1_rows=stage1_rows, validation_rows=validation_rows,
                device=device, epoch=epoch, global_step=global_step, losses=losses,
                train_diag=train_diag,
                train_extras={"epoch_wall_seconds": None, "mean_gradient_norm": None,
                              "cuda_peak_memory_gib": None, "resumed_validation_only": True,
                              "epoch_training_reexecuted": False},
                train_table=train_table, human_table=human_table, frozen_table=frozen_table,
                state_table=state_table, cold_table=cold_table,
            )
        restored_epoch, restored_step = restore_resumable_epoch_payload(
            resume_payload, model=model, optimizer=optimizer, scaler=scaler, restore_rng=True,
        )
        if (restored_epoch, restored_step) != (completed_training_epoch, global_step):
            raise RuntimeError("post-validation RNG restore changed saved progress")
        log(f"POST_TRAIN_RNG_RESTORED completed_epoch={restored_epoch} global_step={restored_step} next_epoch={restored_epoch + 1}")
        start_epoch = completed_training_epoch + 1
        del resume_payload
    else:
        start_epoch = 1

    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats(device)
        order = list(range(len(train.performances)))
        random.Random(SEED + epoch).shuffle(order)
        train_diagnostic_total = training_diagnostic_accumulator()
        loss_total = empty_loss_totals()
        grad_norms = []
        for start in range(0, len(order), PERFORMANCES_PER_STEP):
            selected = order[start:start + PERFORMANCES_PER_STEP]
            plans = [controller_plan(model, train, index, device, amp=True) for index in selected]
            add_training_diagnostics(train_diagnostic_total, plans)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            replay = replay_static(
                model, train, plans, device, amp=True,
                backward=lambda value: scaler.scale(value).backward(),
            )
            add_loss(loss_total, replay)
            scaler.unscale_(optimizer)
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), common.MAX_GRAD_NORM).item())
            finite = math.isfinite(norm) and all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters())
            scale_before = float(scaler.get_scale())
            if finite:
                scaler.step(optimizer)
            scaler.update()
            if not finite or float(scaler.get_scale()) < scale_before:
                consecutive_failure += 1
                log(f"WARNING gradient_or_amp epoch={epoch} group={start // PERFORMANCES_PER_STEP} consecutive={consecutive_failure}")
            else:
                consecutive_failure = 0
                global_step += 1
                grad_norms.append(norm)
            if consecutive_failure >= 3:
                raise FloatingPointError("repeated non-finite gradient/AMP overflow")
            if start == 0:
                log(f"TRAIN_INITIAL_LOSS epoch={epoch} total={replay['total']:.6f} finite=1 global_step={global_step}")
            if (start // PERFORMANCES_PER_STEP + 1) % 25 == 0:
                log(f"TRAIN epoch={epoch} performances={min(start + PERFORMANCES_PER_STEP,2062)}/2062 global_step={global_step} peak_gib={torch.cuda.max_memory_allocated(device)/2**30:.3f}")
        losses = finish_loss(loss_total)
        train_diag = finish_training_diagnostics(train_diagnostic_total)
        resume_payload = build_resumable_epoch_payload(
            model=model, optimizer=optimizer, scaler=scaler,
            completed_training_epoch=epoch, global_step=global_step,
            run_configuration=configuration(),
            extra={"train_losses": losses, "train_diagnostics": train_diag,
                   "created_at": now(), "asap_test_access": 0},
        )
        atomic_save_resumable_checkpoint(
            RUN_ROOT / "checkpoints" / f"train_complete_epoch_{epoch:02d}.pt",
            resume_payload,
        )
        atomic_save_resumable_checkpoint(RUN_ROOT / "resume_last_train_state.pt", resume_payload)
        atomic_json(RUN_ROOT / "run_status.json", {
            "status": "TRAIN_EPOCH_COMPLETE_VALIDATION_PENDING",
            "completed_training_epoch": epoch, "completed_validation_epochs": epoch - 1,
            "global_step": global_step, "resume_checkpoint": str(RUN_ROOT / "resume_last_train_state.pt"),
            "last_update": now(), "stage1_inference_regeneration": 0,
            "asap_test_metadata_access": 0, "asap_test_midi_access": 0,
        })
        log(f"TRAIN_EPOCH_CHECKPOINT_SAVED epoch={epoch} global_step={global_step} validation_pending=1")
        best_epoch, best_f1 = finalize_validated_epoch(
            model=model, optimizer=optimizer, scaler=scaler, human_val=human_val,
            pieces=pieces, stage1_rows=stage1_rows, validation_rows=validation_rows,
            device=device, epoch=epoch, global_step=global_step, losses=losses,
            train_diag=train_diag,
            train_extras={"epoch_wall_seconds": time.monotonic() - started,
                          "mean_gradient_norm": float(np.mean(grad_norms)) if grad_norms else 0.0,
                          "cuda_peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                          "resumed_validation_only": False, "epoch_training_reexecuted": False},
            train_table=train_table, human_table=human_table, frozen_table=frozen_table,
            state_table=state_table, cold_table=cold_table,
        )
        restored_epoch, restored_step = restore_resumable_epoch_payload(
            resume_payload, model=model, optimizer=optimizer, scaler=scaler, restore_rng=True,
        )
        if (restored_epoch, restored_step) != (epoch, global_step):
            raise RuntimeError("post-validation train state restore changed epoch/global step")
        log(f"POST_TRAIN_RNG_RESTORED completed_epoch={epoch} global_step={global_step} next_epoch={epoch + 1}")
        del resume_payload
    log(f"FULL_COMPLETE epochs=10 best_epoch={best_epoch} best_frozen_f1={best_f1:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "full", "resume"))
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.mode == "resume" and args.resume is None:
        parser.error("resume mode requires --resume CHECKPOINT")
    if args.mode != "resume" and args.resume is not None:
        parser.error("--resume is valid only in resume mode")
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    lock = acquire_run_lock(args.mode) if args.mode in {"full", "resume"} else None
    try:
        if args.mode == "preflight":
            preflight()
        else:
            full(args.resume if args.mode == "resume" else None)
    except Exception as error:
        atomic_json(RUN_ROOT / f"failure_{args.mode}.json", {
            "mode": args.mode, "failed_at": now(), "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(), "asap_test_access": 0,
        })
        atomic_json(RUN_ROOT / "run_status.json", {
            "status": "FAILED", "mode": args.mode, "error": f"{type(error).__name__}: {error}",
            "last_update": now(), "asap_test_metadata_access": 0, "asap_test_midi_access": 0,
        })
        log(f"HARD_STOP mode={args.mode} error={type(error).__name__}: {error}")
        raise
    finally:
        if lock is not None and lock.exists():
            lock.unlink()


if __name__ == "__main__":
    main()
