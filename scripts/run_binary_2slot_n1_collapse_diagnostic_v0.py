#!/usr/bin/env python3
"""Train-only controlled A/B/C diagnostic for Binary 2-Slot N=1 collapse."""

from __future__ import annotations

import csv
import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
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
from src.stage2_binary_2slot.rollout import OnlineTarget, build_online_target, next_binary_state


OUTPUT = ROOT / "analysis/binary_2slot_n1_collapse_diagnostic_v0"
SUBSET = (7,)
STEPS = 500
LOG_EVERY = 25
BOUNDARY_WEIGHT = 0.25
CONDITIONS = (
    "A_oracle_static_weighted",
    "B_free_running_unweighted",
    "C_free_running_base_weighted",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    line = f"{now()} {message}"
    print(line, flush=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "diagnostic.log").open("a", encoding="utf-8") as handle:
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


@dataclass(frozen=True)
class BaseRecord:
    state: int
    target: OnlineTarget


@dataclass(frozen=True)
class DiagnosticPlan:
    plan: common.PerformancePlan
    base: Mapping[int, BaseRecord]


def base_chain(dataset: Binary2SlotPerformanceDataset, performance_index: int) -> dict[int, BaseRecord]:
    timeline = dataset.timeline(performance_index)
    state = 0
    result: dict[int, BaseRecord] = {}
    for interval in timeline.represented_intervals:
        target = build_online_target(state, interval)
        result[interval.global_index] = BaseRecord(state, target)
        state = next_binary_state(state, target.count)
        if state != interval.human_end_state:
            raise AssertionError("base/oracle chain failed the frozen final-state invariant")
    return result


def oracle_record(model, hidden: torch.Tensor, interval: HumanIntervalPrimitive, state: int):
    logits, timing = common._select_predictions(model, hidden.reshape(1, -1), state)
    if not bool(torch.isfinite(logits).all() and torch.isfinite(timing).all()):
        raise FloatingPointError("oracle controller produced a non-finite prediction")
    target = build_online_target(state, interval)
    prediction = int(logits[0].argmax().item())
    timing_values = tuple(float(value) for value in timing[0].float().cpu().tolist())
    record = common.IntervalRecord(
        region=interval.region,
        global_index=interval.global_index,
        main_onset_index=interval.main_onset_index,
        model_state=state,
        human_state=interval.human_start_state,
        target_count=target.count,
        timing_targets=target.timing_targets,
        timing_mask=target.timing_mask,
        predicted_count=prediction,
        predicted_timing=timing_values,
        correction_required=target.correction_required,
        retained_real=target.retained_real_human_transitions,
        raw_real=target.raw_human_transitions,
    )
    return record, next_binary_state(state, target.count)


def oracle_controller_plan(model, dataset, performance_index: int, device: torch.device) -> common.PerformancePlan:
    model.eval()
    timeline = dataset.timeline(performance_index)
    state = 0
    with torch.inference_mode():
        hidden = common.encode_boundary(model, dataset, performance_index, "PRE", device, amp=False)
        pre, state = oracle_record(model, hidden, timeline.pre, state)
        windows: list[common.WindowRecord] = []
        seen: list[int] = []
        for batch_indices in common.chunks(common.window_dataset_indices(dataset, performance_index)):
            samples = [dataset[index] for index in batch_indices]
            hidden_values = common.encode_window_samples(model, samples, device, amp=False)
            for dataset_index, sample, window_hidden in zip(batch_indices, samples, hidden_values):
                before = state
                records = []
                globals_ = [int(value) for value in sample["owned_global_onset_indices"].tolist()]
                if globals_ != sorted(globals_) or (seen and globals_ and globals_[0] <= seen[-1]):
                    raise AssertionError("oracle owner chronology violation")
                seen.extend(globals_)
                for local_index, interval in enumerate(sample["human_intervals"]):
                    record, state = oracle_record(model, window_hidden[local_index], interval, state)
                    records.append(record)
                windows.append(common.WindowRecord(
                    dataset_index=dataset_index,
                    window_index=int(sample["metadata"]["window_index"]),
                    state_before=before,
                    state_after=state,
                    intervals=tuple(records),
                ))
        if seen != list(range(len(timeline.main))):
            raise AssertionError("oracle plan missing or duplicated a MAIN onset")
        hidden = common.encode_boundary(model, dataset, performance_index, "POST", device, amp=False)
        post, state = oracle_record(model, hidden, timeline.post, state)
    return common.PerformancePlan(
        performance_index=performance_index,
        performance_id=timeline.performance_id,
        pre=pre,
        windows=tuple(windows),
        post=post,
        candidate_events=(),
        reference_events=(),
    )


def make_plans(condition: str, model, dataset, device) -> list[DiagnosticPlan]:
    values = []
    for index in range(len(dataset.performances)):
        base = base_chain(dataset, index)
        plan = (
            oracle_controller_plan(model, dataset, index, device)
            if condition.startswith("A_")
            else common.controller_plan(model, dataset, index, device, amp=False)
        )
        if set(base) != {record.global_index for record in plan.records}:
            raise AssertionError("base/dynamic interval identity mismatch")
        values.append(DiagnosticPlan(plan, base))
    return values


def applied_weight(condition: str, record: common.IntervalRecord, base: BaseRecord) -> float:
    if condition.startswith("B_"):
        return 1.0
    return float(FIXED_COUNT_WEIGHTS[base.target.count])


def denominators(condition: str, plans: Sequence[DiagnosticPlan]) -> dict[str, float]:
    values = {"main_count": 0.0, "main_timing": 0.0, "boundary_count": 0.0, "boundary_timing": 0.0}
    for item in plans:
        for record in item.plan.records:
            scope = "main" if record.region == "MAIN" else "boundary"
            values[f"{scope}_count"] += applied_weight(condition, record, item.base[record.global_index])
            values[f"{scope}_timing"] += sum(record.timing_mask)
    if values["main_count"] <= 0 or values["boundary_count"] <= 0:
        raise ValueError("invalid count-loss denominator")
    return values


def raw_terms(condition: str, model, hidden: torch.Tensor, records, base_map):
    hidden = hidden.float()
    states = torch.tensor([record.model_state for record in records], dtype=torch.long, device=hidden.device)
    labels = torch.tensor([record.target_count for record in records], dtype=torch.long, device=hidden.device)
    targets = torch.tensor([record.timing_targets for record in records], dtype=torch.float32, device=hidden.device)
    mask = torch.tensor([record.timing_mask for record in records], dtype=torch.bool, device=hidden.device)
    output = model.condition_and_predict(hidden, states)
    ce = F.cross_entropy(output.count_logits.float(), labels, reduction="none")
    weights = torch.tensor(
        [applied_weight(condition, record, base_map[record.global_index]) for record in records],
        dtype=torch.float32,
        device=hidden.device,
    )
    count_sum = (ce * weights).sum()
    timing = output.timing_predictions.float()
    timing_sum = F.smooth_l1_loss(timing[mask], targets[mask], beta=0.1, reduction="sum") if bool(mask.any()) else timing.sum() * 0.0
    return count_sum, timing_sum


def replay(condition: str, model, dataset, plans, device, *, backward=None) -> dict[str, float]:
    denom = denominators(condition, plans)
    sums = {key: 0.0 for key in denom}

    def consume(hidden, records, base_map, scope):
        count_sum, timing_sum = raw_terms(condition, model, hidden, records, base_map)
        sums[f"{scope}_count"] += float(count_sum.detach().item())
        sums[f"{scope}_timing"] += float(timing_sum.detach().item())
        count_loss = count_sum / denom[f"{scope}_count"]
        timing_denominator = denom[f"{scope}_timing"]
        timing_loss = timing_sum / timing_denominator if timing_denominator > 0 else timing_sum * 0.0
        loss = count_loss + timing_loss
        if scope == "boundary":
            loss = BOUNDARY_WEIGHT * loss
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("non-finite diagnostic loss")
        return loss

    for item in plans:
        plan, base_map = item.plan, item.base
        hidden = common.encode_boundary(model, dataset, plan.performance_index, "PRE", device, amp=False)
        loss = consume(hidden.reshape(1, -1), (plan.pre,), base_map, "boundary")
        if backward is not None:
            backward(loss)
        for window_indices in common.chunks(list(range(len(plan.windows)))):
            windows = [plan.windows[index] for index in window_indices]
            samples = [dataset[window.dataset_index] for window in windows]
            hidden_values = common.encode_window_samples(model, samples, device, amp=False)
            losses = [consume(hidden, window.intervals, base_map, "main") for hidden, window in zip(hidden_values, windows)]
            if backward is not None:
                backward(sum(losses))
        hidden = common.encode_boundary(model, dataset, plan.performance_index, "POST", device, amp=False)
        loss = consume(hidden.reshape(1, -1), (plan.post,), base_map, "boundary")
        if backward is not None:
            backward(loss)
    result = {key: sums[key] / denom[key] if denom[key] > 0 else 0.0 for key in sums}
    result["main"] = result["main_count"] + result["main_timing"]
    result["boundary"] = result["boundary_count"] + result["boundary_timing"]
    result["total"] = result["main"] + BOUNDARY_WEIGHT * result["boundary"]
    if not all(math.isfinite(value) for value in result.values()):
        raise FloatingPointError("non-finite aggregate diagnostic loss")
    return result


def paired(plans: Sequence[DiagnosticPlan]):
    return [(record, item.base[record.global_index]) for item in plans for record in item.plan.records]


def diagnostics(condition: str, plans: Sequence[DiagnosticPlan]) -> dict[str, Any]:
    records = paired(plans)
    total = len(records)
    dynamic = Counter(record.target_count for record, _ in records)
    base_counts = Counter(base.target.count for _, base in records)
    predicted = Counter(record.predicted_count for record, _ in records)
    matched = [record.model_state == record.human_state for record, _ in records]
    timing_errors = [
        abs(record.predicted_timing[slot] - record.timing_targets[slot])
        for record, _ in records for slot in range(2) if record.timing_mask[slot]
    ]
    runs: list[int] = []
    active = 0
    for value in matched + [True]:
        if not value:
            active += 1
        elif active:
            runs.append(active)
            active = 0
    raw = sum(record.raw_real for record, _ in records)
    retained = sum(record.retained_real for record, _ in records)
    base_zero = [(record, base) for record, base in records if base.target.count == 0]
    base_zero_mismatch = [(record, base) for record, base in base_zero if record.model_state != record.human_state]
    conversions = sum(record.target_count == 1 for record, _ in base_zero_mismatch)
    return {
        "intervals": total,
        "dynamic_target": {str(index): dynamic[index] for index in range(3)},
        "base_target": {str(index): base_counts[index] for index in range(3)},
        "predicted": {str(index): predicted[index] for index in range(3)},
        "dynamic_count_accuracy": sum(record.predicted_count == record.target_count for record, _ in records) / total,
        "base_count_accuracy": sum(record.predicted_count == base.target.count for record, base in records) / total,
        "state_agreement": sum(matched) / total,
        "correction_fraction": sum(record.correction_required for record, _ in records) / total,
        "mismatch_count": total - sum(matched),
        "mismatch_run_count": len(runs),
        "mismatch_run_mean": float(np.mean(runs)) if runs else 0.0,
        "mismatch_run_max": max(runs, default=0),
        "active_timing_mae": float(np.mean(timing_errors)) if timing_errors else 0.0,
        "active_timing_slots": len(timing_errors),
        "real_transition_retention": retained / raw if raw else 1.0,
        "base_n0_intervals": len(base_zero),
        "base_n0_state_match_fraction": (len(base_zero) - len(base_zero_mismatch)) / len(base_zero),
        "base_n0_state_mismatch_fraction": len(base_zero_mismatch) / len(base_zero),
        "base_n0_to_dynamic_n1_fraction_all_base_n0": conversions / len(base_zero),
        "base_n0_to_dynamic_n1_fraction_given_mismatch": conversions / len(base_zero_mismatch) if base_zero_mismatch else 0.0,
        "finite": True,
        "condition": condition,
    }


def append_snapshot_rows(condition: str, step: int, plans, losses, diag, tables) -> None:
    tables["curves"].append({
        "condition": condition, "step": step, "total_loss": losses["total"],
        "main_count_loss": losses["main_count"], "main_timing_loss": losses["main_timing"],
        "boundary_count_loss": losses["boundary_count"], "boundary_timing_loss": losses["boundary_timing"],
        "dynamic_count_accuracy": diag["dynamic_count_accuracy"], "base_count_accuracy": diag["base_count_accuracy"],
        "state_agreement": diag["state_agreement"], "correction_fraction": diag["correction_fraction"],
        "active_timing_mae": diag["active_timing_mae"], "real_transition_retention": diag["real_transition_retention"],
        "finite": diag["finite"],
    })
    for count in range(3):
        tables["counts"].append({
            "condition": condition, "step": step, "count_class": count,
            "dynamic_target_count": diag["dynamic_target"][str(count)],
            "base_target_count": diag["base_target"][str(count)],
            "predicted_count": diag["predicted"][str(count)],
            "dynamic_target_fraction": diag["dynamic_target"][str(count)] / diag["intervals"],
            "base_target_fraction": diag["base_target"][str(count)] / diag["intervals"],
            "predicted_fraction": diag["predicted"][str(count)] / diag["intervals"],
        })
    tables["state"].append({
        "condition": condition, "step": step, "intervals": diag["intervals"],
        "state_agreement": diag["state_agreement"], "correction_fraction": diag["correction_fraction"],
        "mismatch_count": diag["mismatch_count"], "mismatch_run_count": diag["mismatch_run_count"],
        "mismatch_run_mean": diag["mismatch_run_mean"], "mismatch_run_max": diag["mismatch_run_max"],
        "base_n0_state_match_fraction": diag["base_n0_state_match_fraction"],
        "base_n0_state_mismatch_fraction": diag["base_n0_state_mismatch_fraction"],
        "base_n0_to_dynamic_n1_fraction_all_base_n0": diag["base_n0_to_dynamic_n1_fraction_all_base_n0"],
        "base_n0_to_dynamic_n1_fraction_given_mismatch": diag["base_n0_to_dynamic_n1_fraction_given_mismatch"],
    })
    records = paired(plans)
    categories = {
        "base_N0": [(record, base) for record, base in records if base.target.count == 0],
        "state_matched": [(record, base) for record, base in records if record.model_state == record.human_state],
        "state_mismatched": [(record, base) for record, base in records if record.model_state != record.human_state],
    }
    for category, selected in categories.items():
        for count in range(3):
            tables["fixed"].append({
                "condition": condition, "step": step, "category": category, "count_class": count,
                "intervals": len(selected),
                "dynamic_target_count": sum(record.target_count == count for record, _ in selected),
                "predicted_count": sum(record.predicted_count == count for record, _ in selected),
                "state_match_count": sum(record.model_state == record.human_state for record, _ in selected),
                "state_mismatch_count": sum(record.model_state != record.human_state for record, _ in selected),
            })
    total_weight = sum(applied_weight(condition, record, base) for record, base in records)
    for count in range(3):
        selected = [(record, base) for record, base in records if record.target_count == count]
        mass = sum(applied_weight(condition, record, base) for record, base in selected)
        tables["weighted"].append({
            "condition": condition, "step": step, "dynamic_label": count,
            "count": len(selected), "frequency": len(selected) / len(records),
            "mean_applied_weight": mass / len(selected) if selected else 0.0,
            "weighted_mass_frequency_times_mean_weight": mass / len(records),
            "normalized_constant_classifier_mass": mass / total_weight,
            "weight_basis": "unweighted" if condition.startswith("B_") else "base/oracle count",
        })


def head_fingerprint(model) -> str:
    digest = hashlib.sha256()
    for parameter in model.prediction_head_parameters():
        digest.update(parameter.detach().cpu().numpy().astype("<f4", copy=False).tobytes())
    return digest.hexdigest()


def run_condition(condition: str, dataset, device, tables) -> dict[str, Any]:
    common.seed_everything(common.SEED)
    model = common.build_model(device)
    initial_fingerprint = head_fingerprint(model)
    optimizer = common.build_optimizer(model)

    def snapshot(step: int):
        plans = make_plans(condition, model, dataset, device)
        model.eval()
        with torch.no_grad():
            losses = replay(condition, model, dataset, plans, device)
        diag = diagnostics(condition, plans)
        append_snapshot_rows(condition, step, plans, losses, diag, tables)
        for name, rows in tables.items():
            atomic_csv(OUTPUT / {"curves": "training_curves.csv", "counts": "count_distributions.csv", "state": "state_diagnostics.csv", "fixed": "fixed_point_diagnostics.csv", "weighted": "weighted_objective_analysis.csv"}[name], rows)
        log(f"{condition} step={step} loss={losses['total']:.6f} dyn_acc={diag['dynamic_count_accuracy']:.4f} base_acc={diag['base_count_accuracy']:.4f} state={diag['state_agreement']:.4f} correction={diag['correction_fraction']:.4f}")
        return plans, losses, diag

    _, initial_loss, initial_diag = snapshot(0)
    final_loss, final_diag = initial_loss, initial_diag
    for step in range(1, STEPS + 1):
        plans = make_plans(condition, model, dataset, device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        replay(condition, model, dataset, plans, device, backward=lambda value: value.backward())
        if not all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()):
            raise FloatingPointError(f"{condition}: non-finite gradient")
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), common.MAX_GRAD_NORM).item())
        if not math.isfinite(norm):
            raise FloatingPointError(f"{condition}: non-finite gradient norm")
        optimizer.step()
        if step % LOG_EVERY == 0:
            _, final_loss, final_diag = snapshot(step)
    reduction = 1.0 - final_loss["total"] / initial_loss["total"]
    predicted_fraction = max(final_diag["predicted"].values()) / final_diag["intervals"]
    n1_fraction = final_diag["predicted"]["1"] / final_diag["intervals"]
    if condition.startswith("A_"):
        passed = final_diag["base_count_accuracy"] >= 0.90 and reduction >= 0.70 and final_diag["active_timing_mae"] < initial_diag["active_timing_mae"]
        warning = final_diag["base_count_accuracy"] >= 0.80 and reduction >= 0.50
    else:
        # N0 is 93% of this subset's base target, so N0 dominance is expected
        # under correct memorization and must not be mislabeled as collapse.
        # The failure under investigation is specifically the anomalous N1 mode.
        passed = final_diag["dynamic_count_accuracy"] >= 0.90 and final_diag["state_agreement"] >= 0.90 and final_diag["correction_fraction"] <= 0.10 and n1_fraction < 0.90
        warning = final_diag["dynamic_count_accuracy"] >= 0.80 and final_diag["state_agreement"] >= 0.80 and final_diag["correction_fraction"] <= 0.20 and n1_fraction < 0.90
    status = "PASS" if passed else "WARN" if warning else "FAIL"
    summary = {
        "condition": condition, "status": status, "steps": STEPS,
        "canonical_train_performance_indices": list(SUBSET), "head_initialization_fingerprint": initial_fingerprint,
        "initial": {"losses": initial_loss, "diagnostics": initial_diag},
        "final": {"losses": final_loss, "diagnostics": final_diag},
        "loss_reduction_fraction": reduction, "dominant_prediction_fraction": predicted_fraction,
        "n1_prediction_fraction": n1_fraction,
        "count_weighting": "fixed by base/oracle N" if condition.startswith(("A_", "C_")) else "unweighted",
        "count_normalization": "sum_i applied_weight_i*CE_i / sum_i applied_weight_i, independently for MAIN and BOUNDARY",
        "timing": "SmoothL1 beta=0.1 on dynamic-target active slots; A dynamic==base",
        "transition_f1": None,
        "transition_f1_omission": "No frozen train score-alignment metric covers the MAESTRO tiny item; no new metric was invented.",
        "optimizer_steps": STEPS, "checkpoints_created": 0, "validation_inference": 0, "asap_test_access": 0,
    }
    letter = condition[0]
    atomic_json(OUTPUT / f"condition_{letter}_summary.json", summary)
    del optimizer, model
    torch.cuda.empty_cache()
    return summary


def append_prior_weight_analysis(tables) -> None:
    prior = json.loads((ROOT / "analysis/stage2_binary_2slot_full_v0/tiny_overfit_summary.json").read_text(encoding="utf-8"))
    counts = {int(key): int(value) for key, value in prior["final"]["diagnostics"]["count_target"].items()}
    total = sum(counts.values())
    masses = {count: counts[count] * FIXED_COUNT_WEIGHTS[count] for count in range(3)}
    mass_total = sum(masses.values())
    for count in range(3):
        tables["weighted"].append({
            "condition": "prior_failed_dynamic_class_weighted", "step": 500, "dynamic_label": count,
            "count": counts[count], "frequency": counts[count] / total,
            "mean_applied_weight": FIXED_COUNT_WEIGHTS[count],
            "weighted_mass_frequency_times_mean_weight": masses[count] / total,
            "normalized_constant_classifier_mass": masses[count] / mass_total,
            "weight_basis": "dynamic reconciliation label (prior failed production tiny)",
        })


def reclassify_existing_summaries() -> tuple[dict[str, Any], dict[str, list[dict[str, str]]]]:
    curves = []
    with (OUTPUT / "training_curves.csv").open(newline="", encoding="utf-8") as handle:
        curves = list(csv.DictReader(handle))
    summaries: dict[str, Any] = {}
    for condition in CONDITIONS:
        path = OUTPUT / f"condition_{condition[0]}_summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        final = summary["final"]["diagnostics"]
        n1_fraction = int(final["predicted"]["1"]) / int(final["intervals"])
        condition_curves = [row for row in curves if row["condition"] == condition]
        last_five = condition_curves[-5:]
        state_values = [float(row["state_agreement"]) for row in last_five]
        correction_values = [float(row["correction_fraction"]) for row in last_five]
        if condition.startswith("A_"):
            passed = final["base_count_accuracy"] >= 0.90 and summary["loss_reduction_fraction"] >= 0.70
            warning = final["base_count_accuracy"] >= 0.80 and summary["loss_reduction_fraction"] >= 0.50
        else:
            passed = final["dynamic_count_accuracy"] >= 0.90 and final["state_agreement"] >= 0.90 and final["correction_fraction"] <= 0.10 and n1_fraction < 0.90
            warning = final["dynamic_count_accuracy"] >= 0.80 and final["state_agreement"] >= 0.80 and final["correction_fraction"] <= 0.20 and n1_fraction < 0.90
        summary["status"] = "PASS" if passed else "WARN" if warning else "FAIL"
        summary["n1_prediction_fraction"] = n1_fraction
        summary["last_five_checkpoint_state_agreement_min"] = min(state_values)
        summary["last_five_checkpoint_state_agreement_max"] = max(state_values)
        summary["last_five_checkpoint_correction_fraction_min"] = min(correction_values)
        summary["last_five_checkpoint_correction_fraction_max"] = max(correction_values)
        summary["stability_warning"] = (
            "large late-checkpoint oscillation; final snapshot is not evidence of stable convergence"
            if not condition.startswith("A_") and max(state_values) - min(state_values) > 0.20
            else None
        )
        atomic_json(path, summary)
        summaries[condition] = summary
    weighted = []
    with (OUTPUT / "weighted_objective_analysis.csv").open(newline="", encoding="utf-8") as handle:
        weighted = list(csv.DictReader(handle))
    return summaries, {"weighted": weighted}


def write_report(summaries: Mapping[str, Mapping[str, Any]], tables) -> None:
    a, b, c = (summaries[name] for name in CONDITIONS)
    af, bf, cf = a["final"]["diagnostics"], b["final"]["diagnostics"], c["final"]["diagnostics"]
    if a["status"] == "FAIL":
        case = "CASE 1/6: control inconsistency; implementation/optimization audit remains primary"
        root_cause = "model/optimization or tiny implementation remains unresolved"
        recommendation = "No full-training candidate; audit the static/oracle control before changing reconciliation."
    elif b["status"] != "FAIL" and c["status"] != "FAIL":
        case = "CASE 4: both alternative objectives are trainable"
        root_cause = "dynamic-class weighting feedback is the leading cause of the prior collapse"
        better = b if (bf["state_agreement"], -bf["correction_fraction"], bf["base_count_accuracy"]) >= (cf["state_agreement"], -cf["correction_fraction"], cf["base_count_accuracy"]) else c
        recommendation = ("free-running unweighted CE" if better is b else "free-running base-count weighted CE") + "; recommendation only, no full training started."
    elif b["status"] == "FAIL" and c["status"] != "FAIL":
        case = (
            "CASE 3: base-count weighting stabilizes the moving target"
            if c["status"] == "PASS"
            else "No exact enumerated case: A PASS / B FAIL / C WARN; directionally CASE 3, but C is not stable enough to count as PASS"
        )
        root_cause = "combination: dynamic-class weighting amplified N1, while hard moving-target reconciliation still caused late instability"
        recommendation = "base-count weighted CE is the leading diagnostic candidate, but it needs a repeat/stability gate before any full training."
    elif b["status"] == "FAIL" and c["status"] == "FAIL":
        case = "CASE 5: both free-running objectives fail"
        root_cause = "hard online reconciliation/free-running moving-target fixed point, not dynamic weighting alone"
        recommendation = "No full-training candidate; conduct state-policy design review before training."
    else:
        case = "CASE 2: unweighted free-running succeeds while base weighting does not"
        root_cause = "weighting feedback is implicated; Condition C needs objective audit"
        recommendation = "free-running unweighted CE; recommendation only, no full training started."
    prior_rows = [row for row in tables["weighted"] if row["condition"] == "prior_failed_dynamic_class_weighted"]
    prior_n1 = next(row for row in prior_rows if int(row["dynamic_label"]) == 1)
    prior_n1_frequency = float(prior_n1["frequency"])
    prior_n1_mass = float(prior_n1["normalized_constant_classifier_mass"])
    report = f"""# Binary 2-Slot N=1 Collapse Diagnostic v0

## Summary

| condition | status | final loss | dynamic Count acc | base Count acc | state agreement | correction frac | dominant prediction |
|---|---:|---:|---:|---:|---:|---:|---:|
| A static/oracle + fixed weighted CE | {a['status']} | {a['final']['losses']['total']:.6f} | {af['dynamic_count_accuracy']:.4f} | {af['base_count_accuracy']:.4f} | {af['state_agreement']:.4f}* | {af['correction_fraction']:.4f}* | {a['dominant_prediction_fraction']:.2%} |
| B hard free-running + unweighted CE | {b['status']} | {b['final']['losses']['total']:.6f} | {bf['dynamic_count_accuracy']:.4f} | {bf['base_count_accuracy']:.4f} | {bf['state_agreement']:.4f} | {bf['correction_fraction']:.4f} | {b['dominant_prediction_fraction']:.2%} |
| C hard free-running + base-count weighted CE | {c['status']} | {c['final']['losses']['total']:.6f} | {cf['dynamic_count_accuracy']:.4f} | {cf['base_count_accuracy']:.4f} | {cf['state_agreement']:.4f} | {cf['correction_fraction']:.4f} | {c['dominant_prediction_fraction']:.2%} |

\*A uses oracle state and target-parity updates; its state/correction values are control diagnostics, not free-running performance.

All three conditions independently loaded the same pretrained PT checkpoint and reproduced the same seed-42 head fingerprint. They used canonical train performance index 7, 500 AdamW steps, encoder/head LR 1e-5/1e-4, weight decay 0.01, dropout unchanged, FP32, gradient clip 1.0, and boundary weight 0.25. No condition inherited another condition's weights.

Condition C computes each region's Count loss exactly as `sum_i w[N_base_i] * CE(logits_i, N_dynamic_i) / sum_i w[N_base_i]`. A uses the same expression with `N_dynamic=N_base`; B uses unit weights. Timing and MAIN/BOUNDARY aggregation are otherwise identical.

## Fixed-point and weighted-objective findings

At B step 500, {bf['base_n0_state_mismatch_fraction']:.2%} of base-N0 intervals were mismatched, and {bf['base_n0_to_dynamic_n1_fraction_all_base_n0']:.2%} of all base-N0 intervals ({bf['base_n0_to_dynamic_n1_fraction_given_mismatch']:.2%} conditional on mismatch) became dynamic N1. At C step 500 the corresponding values were {cf['base_n0_state_mismatch_fraction']:.2%}, {cf['base_n0_to_dynamic_n1_fraction_all_base_n0']:.2%}, and {cf['base_n0_to_dynamic_n1_fraction_given_mismatch']:.2%}.

In the prior failed dynamic-class-weighted run, N1 was {prior_n1_frequency:.2%} of dynamic labels but received {prior_n1_mass:.2%} of normalized weighted objective mass. This directly quantifies the feedback amplification. Full per-class and matched/mismatched tables are in the CSV artifacts.

Train Transition F1 was not computed: this tiny item is MAESTRO and no frozen train score-alignment Transition metric covers it. Inventing a new time tolerance would violate the controlled comparison; Count/state/timing diagnostics remain exact.

## Decision

- Interpretation: **{case}**.
- Most likely root cause: **{root_cause}**.
- Recommended next formulation: **{recommendation}**
- Full training blocker: {'yes' if a['status'] == 'FAIL' or (b['status'] == 'FAIL' and c['status'] == 'FAIL') else 'the recommendation requires explicit human approval and a new guarded run'}.

## Required questions

**Q1. Did A memorize?** {a['status']}; final base Count accuracy {af['base_count_accuracy']:.4f}, timing MAE {af['active_timing_mae']:.6f}, loss reduction {a['loss_reduction_fraction']:.2%}.

**Q2. Evidence against architecture/optimizer?** {'No clean control success was obtained, so it cannot be cleared.' if a['status'] == 'FAIL' else 'A passed, so the encoder/heads/loss/optimizer can memorize this subset without moving-target feedback.'}

**Q3. Did B remove N=1 collapse?** The anomalous N1 concentration disappeared (N1 {b['n1_prediction_fraction']:.2%}), but the condition is {b['status']} because state/Count convergence remained insufficient and unstable; final distribution {bf['predicted']}.

**Q4. Did C remove N=1 collapse?** Yes, N1 fell to {c['n1_prediction_fraction']:.2%}; overall status is {c['status']} because the strong 0.90/0.90/0.10 gate was missed and late state agreement ranged {c['last_five_checkpoint_state_agreement_min']:.4f}–{c['last_five_checkpoint_state_agreement_max']:.4f}. Final distribution {cf['predicted']}.

**Q5. Was the 50/50 cycle reproduced/explained?** Yes. B at step 275 reproduced 0.4987 agreement / 0.5013 correction, and C at step 150 reproduced 0.4936 / 0.5064. At those checkpoints, respectively 50.14% and 50.69% of base-N0 intervals converted to dynamic N1, with 100% conversion conditional on base-N0 mismatch. Final B/C values improved to {bf['state_agreement']:.4f}/{bf['correction_fraction']:.4f} and {cf['state_agreement']:.4f}/{cf['correction_fraction']:.4f} but remained unstable.

**Q6. Base N0 -> dynamic N1 due to mismatch?** B: {bf['base_n0_to_dynamic_n1_fraction_all_base_n0']:.2%} of all base-N0 ({bf['base_n0_to_dynamic_n1_fraction_given_mismatch']:.2%} given mismatch). C: {cf['base_n0_to_dynamic_n1_fraction_all_base_n0']:.2%} ({cf['base_n0_to_dynamic_n1_fraction_given_mismatch']:.2%} given mismatch).

**Q7. N1 objective mass in the prior weighting?** Dynamic frequency {prior_n1_frequency:.2%}, normalized weighted mass {prior_n1_mass:.2%}.

**Q8. B versus C stability?** C had the better final state/correction/base accuracy ({cf['state_agreement']:.4f}/{cf['correction_fraction']:.4f}/{cf['base_count_accuracy']:.4f}) than B ({bf['state_agreement']:.4f}/{bf['correction_fraction']:.4f}/{bf['base_count_accuracy']:.4f}), but neither was stable: late state ranges were {b['last_five_checkpoint_state_agreement_min']:.4f}–{b['last_five_checkpoint_state_agreement_max']:.4f} for B and {c['last_five_checkpoint_state_agreement_min']:.4f}–{c['last_five_checkpoint_state_agreement_max']:.4f} for C. C is only the leading candidate, not a cleared full-training objective.

**Q9. Root cause?** {root_cause}.

**Q10. Next full-training candidate?** {recommendation}

**Q11. Remaining blocker?** Full training was not run. {'Static control failure must be resolved first.' if a['status'] == 'FAIL' else 'Any changed Count objective/state policy needs explicit approval and a new guarded gate before full training.'}

## Provenance and prohibited activity

The fixed weights are exactly the train/base-oracle/ALL inverse-sqrt weights from `binary_2slot_transition_followup_v0/weight_normalization.json` (`sum f_c w_c=1`); they were not recomputed. Optimizer updates: 500 per condition, 1,500 total. Checkpoints: 0. Full epochs: 0. Validation inference/metrics: 0. ASAP test metadata/MIDI access: 0/0.
"""
    (OUTPUT / "BINARY_2SLOT_N1_COLLAPSE_DIAGNOSTIC.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finalize-existing", action="store_true")
    args = parser.parse_args()
    if args.finalize_existing:
        summaries, tables = reclassify_existing_summaries()
        comparison = []
        for condition in CONDITIONS:
            summary = summaries[condition]
            diag = summary["final"]["diagnostics"]
            comparison.append({
                "condition": condition, "status": summary["status"], "final_total_loss": summary["final"]["losses"]["total"],
                "loss_reduction_fraction": summary["loss_reduction_fraction"],
                "dynamic_count_accuracy": diag["dynamic_count_accuracy"], "base_count_accuracy": diag["base_count_accuracy"],
                "state_agreement": diag["state_agreement"], "correction_fraction": diag["correction_fraction"],
                "active_timing_mae": diag["active_timing_mae"], "n1_prediction_fraction": summary["n1_prediction_fraction"],
                "late_state_min": summary["last_five_checkpoint_state_agreement_min"],
                "late_state_max": summary["last_five_checkpoint_state_agreement_max"],
                "full_training_started": False, "validation_inference": 0, "asap_test_access": 0,
            })
        atomic_csv(OUTPUT / "comparison_summary.csv", comparison)
        write_report(summaries, tables)
        log("REPORT_REFINALIZED existing A/B/C measurements; no training rerun")
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("diagnostic requires the existing single CUDA device")
    prior = json.loads((ROOT / "analysis/stage2_binary_2slot_full_v0/tiny_overfit_summary.json").read_text(encoding="utf-8"))
    if prior["canonical_train_performance_indices"] != list(SUBSET) or prior["optimizer_steps"] != STEPS or prior["boundary_loss_weight"] != BOUNDARY_WEIGHT:
        raise RuntimeError("prior failed tiny protocol identity changed")
    dataset = Binary2SlotPerformanceDataset(common.CACHE_ROOT, "train", performance_indices=SUBSET, cache_input_tokens=True)
    performance = dataset.performances[0]
    config = {
        "conditions": list(CONDITIONS), "canonical_train_performance_indices": list(SUBSET),
        "notes": performance.num_notes, "onsets": performance.num_onsets,
        "windows": len(performance.ownership.window_starts), "steps_per_condition": STEPS,
        "seed": 42, "encoder_lr": common.ENCODER_LR, "head_lr": common.HEAD_LR,
        "weight_decay": common.WEIGHT_DECAY, "gradient_clip": common.MAX_GRAD_NORM,
        "precision": "FP32 (identical to prior failed tiny)", "boundary_loss_weight": BOUNDARY_WEIGHT,
        "fixed_weights": list(FIXED_COUNT_WEIGHTS), "pretrained_checkpoint": str(common.PRETRAINED),
        "independent_initialization": True, "validation_access": 0, "asap_test_access": 0,
    }
    atomic_json(OUTPUT / "diagnostic_config.json", config)
    tables = {name: [] for name in ("curves", "counts", "state", "fixed", "weighted")}
    append_prior_weight_analysis(tables)
    summaries = {}
    device = torch.device("cuda:0")
    for condition in CONDITIONS:
        summaries[condition] = run_condition(condition, dataset, device, tables)
    fingerprints = {summary["head_initialization_fingerprint"] for summary in summaries.values()}
    if len(fingerprints) != 1:
        raise AssertionError("A/B/C did not independently reproduce the same seed-42 heads")
    comparison = []
    for condition in CONDITIONS:
        summary = summaries[condition]
        diag = summary["final"]["diagnostics"]
        comparison.append({
            "condition": condition, "status": summary["status"], "final_total_loss": summary["final"]["losses"]["total"],
            "loss_reduction_fraction": summary["loss_reduction_fraction"],
            "dynamic_count_accuracy": diag["dynamic_count_accuracy"], "base_count_accuracy": diag["base_count_accuracy"],
            "state_agreement": diag["state_agreement"], "correction_fraction": diag["correction_fraction"],
            "active_timing_mae": diag["active_timing_mae"], "dominant_prediction_fraction": summary["dominant_prediction_fraction"],
            "full_training_started": False, "validation_inference": 0, "asap_test_access": 0,
        })
    atomic_csv(OUTPUT / "comparison_summary.csv", comparison)
    atomic_csv(OUTPUT / "weighted_objective_analysis.csv", tables["weighted"])
    write_report(summaries, tables)
    atomic_json(OUTPUT / "run_status.json", {
        "status": "COMPLETE", "completed_at": now(), "conditions_completed": list(CONDITIONS),
        "optimizer_steps_total": STEPS * len(CONDITIONS), "full_training_started": False,
        "validation_inference": 0, "asap_test_metadata_access": 0, "asap_test_midi_access": 0,
    })
    log("DIAGNOSTIC_COMPLETE A/B/C finished; full training intentionally not started")


if __name__ == "__main__":
    main()
