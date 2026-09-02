#!/usr/bin/env python3
"""TRAIN-only C-vs-D state-stability diagnostic for Binary 2-Slot."""

from __future__ import annotations

import csv
import json
import math
import os
import sys
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
from scripts import run_binary_2slot_n1_collapse_diagnostic_v0 as v0
from src.stage2_binary_2slot.dataset import Binary2SlotPerformanceDataset, HumanIntervalPrimitive
from src.stage2_binary_2slot.losses import FIXED_COUNT_WEIGHTS
from src.stage2_binary_2slot.rollout import build_online_target, next_binary_state
from src.stage2_binary_2slot.state_consistency import state_consistency_loss


OUTPUT = ROOT / "analysis/binary_2slot_state_stability_diagnostic_v1"
SUBSETS = (7, 6, 13)
CONDITIONS = ("C_online_base_weighted", "D_static_state_consistency")
STEPS = 1000
BOUNDARY_WEIGHT = 0.25
LAMBDA_CANDIDATES = (1.0, 0.5, 0.25, 0.1)
LATE_START = 801
WRITE_EVERY = 25


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
class StabilityPlan:
    plan: common.PerformancePlan
    base: Mapping[int, v0.BaseRecord]
    count_probabilities: Mapping[int, tuple[float, float, float]]


def make_record(
    condition: str,
    model,
    hidden: torch.Tensor,
    interval: HumanIntervalPrimitive,
    state: int,
    base: v0.BaseRecord,
):
    logits, timing = common._select_predictions(model, hidden.reshape(1, -1), state)
    if not bool(torch.isfinite(logits).all() and torch.isfinite(timing).all()):
        raise FloatingPointError("non-finite stability controller output")
    probabilities = tuple(float(value) for value in logits[0].softmax(dim=-1).float().cpu().tolist())
    prediction = int(logits[0].argmax().item())
    target = build_online_target(state, interval) if condition.startswith("C_") else base.target
    if condition.startswith("D_") and (
        target.count != base.target.count
        or target.timing_targets != base.target.timing_targets
        or target.timing_mask != base.target.timing_mask
    ):
        raise AssertionError("Condition D target is not static base/oracle")
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
        predicted_timing=tuple(float(value) for value in timing[0].float().cpu().tolist()),
        correction_required=target.correction_required if condition.startswith("C_") else False,
        retained_real=target.retained_real_human_transitions,
        raw_real=target.raw_human_transitions,
    )
    return record, probabilities, next_binary_state(state, prediction)


def controller_plan(condition: str, model, dataset, performance_index: int, device: torch.device) -> StabilityPlan:
    model.eval()
    timeline = dataset.timeline(performance_index)
    base_map = v0.base_chain(dataset, performance_index)
    state = 0
    probabilities: dict[int, tuple[float, float, float]] = {}
    with torch.inference_mode():
        hidden = common.encode_boundary(model, dataset, performance_index, "PRE", device, amp=False)
        pre, probability, state = make_record(condition, model, hidden, timeline.pre, state, base_map[timeline.pre.global_index])
        probabilities[pre.global_index] = probability
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
                    raise AssertionError("stability owner chronology violation")
                seen.extend(globals_)
                for local_index, interval in enumerate(sample["human_intervals"]):
                    record, probability, state = make_record(
                        condition, model, window_hidden[local_index], interval, state,
                        base_map[interval.global_index],
                    )
                    records.append(record)
                    probabilities[record.global_index] = probability
                windows.append(common.WindowRecord(
                    dataset_index=dataset_index,
                    window_index=int(sample["metadata"]["window_index"]),
                    state_before=before,
                    state_after=state,
                    intervals=tuple(records),
                ))
        if seen != list(range(len(timeline.main))):
            raise AssertionError("stability plan missing or duplicated a MAIN onset")
        if any(left.state_after != right.state_before for left, right in zip(windows, windows[1:])):
            raise AssertionError("state reset at a window boundary")
        hidden = common.encode_boundary(model, dataset, performance_index, "POST", device, amp=False)
        post, probability, state = make_record(condition, model, hidden, timeline.post, state, base_map[timeline.post.global_index])
        probabilities[post.global_index] = probability
    plan = common.PerformancePlan(
        performance_index=performance_index,
        performance_id=timeline.performance_id,
        pre=pre,
        windows=tuple(windows),
        post=post,
        candidate_events=(),
        reference_events=(),
    )
    if set(probabilities) != set(base_map) or len(plan.records) != len(base_map):
        raise AssertionError("stability base/probability/interval identity mismatch")
    return StabilityPlan(plan, base_map, probabilities)


def loss_denominators(plans: Sequence[StabilityPlan]) -> dict[str, float]:
    values = {
        "main_count": 0.0, "main_timing": 0.0, "main_state": 0.0,
        "boundary_count": 0.0, "boundary_timing": 0.0, "boundary_state": 0.0,
    }
    for item in plans:
        for record in item.plan.records:
            scope = "main" if record.region == "MAIN" else "boundary"
            values[f"{scope}_count"] += FIXED_COUNT_WEIGHTS[item.base[record.global_index].target.count]
            values[f"{scope}_timing"] += sum(record.timing_mask)
            values[f"{scope}_state"] += 1.0
    if values["main_count"] <= 0 or values["boundary_count"] <= 0:
        raise ValueError("zero Count denominator")
    return values


def raw_loss_terms(model, hidden, records, base_map):
    hidden = hidden.float()
    device = hidden.device
    states = torch.tensor([record.model_state for record in records], dtype=torch.long, device=device)
    labels = torch.tensor([record.target_count for record in records], dtype=torch.long, device=device)
    timing_targets = torch.tensor([record.timing_targets for record in records], dtype=torch.float32, device=device)
    timing_mask = torch.tensor([record.timing_mask for record in records], dtype=torch.bool, device=device)
    human_end = torch.tensor([base_map[record.global_index].target.human_end_state for record in records], dtype=torch.long, device=device)
    output = model.condition_and_predict(hidden, states)
    ce = F.cross_entropy(output.count_logits.float(), labels, reduction="none")
    weights = torch.tensor(
        [FIXED_COUNT_WEIGHTS[base_map[record.global_index].target.count] for record in records],
        dtype=torch.float32, device=device,
    )
    count_sum = (ce * weights).sum()
    timing = output.timing_predictions.float()
    timing_sum = F.smooth_l1_loss(timing[timing_mask], timing_targets[timing_mask], beta=0.1, reduction="sum") if bool(timing_mask.any()) else timing.sum() * 0.0
    state_sum = state_consistency_loss(output.count_logits.float(), states, human_end, reduction="sum")
    return {"count": count_sum, "timing": timing_sum, "state": state_sum}


def replay(
    condition: str,
    model,
    dataset,
    plans: Sequence[StabilityPlan],
    device,
    *,
    lambda_state: float,
    components: set[str] = frozenset(("count", "timing", "state")),
    backward=None,
) -> dict[str, float]:
    denom = loss_denominators(plans)
    sums = {key: 0.0 for key in denom}

    def consume(hidden, records, base_map, scope):
        terms = raw_loss_terms(model, hidden, records, base_map)
        selected = []
        for component in ("count", "timing", "state"):
            sums[f"{scope}_{component}"] += float(terms[component].detach().item())
            if component not in components or (component == "state" and condition.startswith("C_")):
                continue
            denominator = denom[f"{scope}_{component}"]
            normalized = terms[component] / denominator if denominator > 0 else terms[component] * 0.0
            if component == "state":
                normalized = float(lambda_state) * normalized
            selected.append(normalized)
        if not selected:
            return None
        loss = sum(selected)
        if scope == "boundary":
            loss = BOUNDARY_WEIGHT * loss
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("non-finite C/D replay loss")
        return loss

    for item in plans:
        plan = item.plan
        hidden = common.encode_boundary(model, dataset, plan.performance_index, "PRE", device, amp=False)
        loss = consume(hidden.reshape(1, -1), (plan.pre,), item.base, "boundary")
        if loss is not None and backward is not None:
            backward(loss)
        for indices in common.chunks(list(range(len(plan.windows)))):
            windows = [plan.windows[index] for index in indices]
            samples = [dataset[window.dataset_index] for window in windows]
            hidden_values = common.encode_window_samples(model, samples, device, amp=False)
            losses = [consume(hidden, window.intervals, item.base, "main") for hidden, window in zip(hidden_values, windows)]
            losses = [loss for loss in losses if loss is not None]
            if losses and backward is not None:
                backward(sum(losses))
        hidden = common.encode_boundary(model, dataset, plan.performance_index, "POST", device, amp=False)
        loss = consume(hidden.reshape(1, -1), (plan.post,), item.base, "boundary")
        if loss is not None and backward is not None:
            backward(loss)
    result = {key: sums[key] / denom[key] if denom[key] > 0 else 0.0 for key in sums}
    result["main"] = result["main_count"] + result["main_timing"] + (lambda_state * result["main_state"] if condition.startswith("D_") else 0.0)
    result["boundary"] = result["boundary_count"] + result["boundary_timing"] + (lambda_state * result["boundary_state"] if condition.startswith("D_") else 0.0)
    result["total"] = result["main"] + BOUNDARY_WEIGHT * result["boundary"]
    if not all(math.isfinite(value) for value in result.values()):
        raise FloatingPointError("non-finite C/D aggregate")
    return result


def count_metrics(targets: Sequence[int], predictions: Sequence[int]) -> dict[str, Any]:
    confusion = np.zeros((3, 3), dtype=np.int64)
    for target, prediction in zip(targets, predictions):
        confusion[int(target), int(prediction)] += 1
    rows = []
    for count in range(3):
        tp = int(confusion[count, count])
        predicted = int(confusion[:, count].sum())
        support = int(confusion[count].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"precision": precision, "recall": recall, "f1": f1, "support": support})
    return {
        "accuracy": float(np.trace(confusion) / confusion.sum()),
        "macro_f1": float(np.mean([row["f1"] for row in rows])),
        "classes": rows,
        "confusion": confusion.tolist(),
    }


def recovery_statistics(start_agreement: Sequence[bool], end_agreement: Sequence[bool]) -> dict[str, float]:
    mismatch_indices = [index for index, value in enumerate(start_agreement) if not value]
    return recovery_statistics_for_indices(mismatch_indices, end_agreement)


def recovery_statistics_for_indices(indices: Sequence[int], end_agreement: Sequence[bool]) -> dict[str, float]:
    result: dict[str, float] = {}
    for horizon in (1, 2, 4):
        recovered = 0
        for index in indices:
            stop = min(len(end_agreement), index + horizon)
            recovered += int(any(end_agreement[index:stop]))
        result[f"recovered_within_{horizon}"] = recovered / len(indices) if indices else 1.0
    never = 0
    for index in indices:
        never += int(not any(end_agreement[index:]))
    result["remain_mismatched"] = never / len(indices) if indices else 0.0
    return result


def metrics(condition: str, plans: Sequence[StabilityPlan]) -> dict[str, Any]:
    records = [(record, item.base[record.global_index], item.count_probabilities[record.global_index]) for item in plans for record in item.plan.records]
    base_targets = [base.target.count for _, base, _ in records]
    predictions = [record.predicted_count for record, _, _ in records]
    base_count = count_metrics(base_targets, predictions)
    dynamic_count = count_metrics([record.target_count for record, _, _ in records], predictions)
    start_agreement = [record.model_state == record.human_state for record, _, _ in records]
    end_agreement = [next_binary_state(record.model_state, record.predicted_count) == base.target.human_end_state for record, base, _ in records]
    runs: list[int] = []
    active = 0
    for agreement in start_agreement + [True]:
        if not agreement:
            active += 1
        elif active:
            runs.append(active)
            active = 0
    timing_errors = [
        abs(record.predicted_timing[slot] - record.timing_targets[slot])
        for record, _, _ in records for slot in range(2) if record.timing_mask[slot]
    ]
    required_toggle = [record.model_state ^ base.target.human_end_state for record, base, _ in records]
    predicted_parity = [record.predicted_count % 2 for record, _, _ in records]
    toggle_probabilities = [probability[1] for _, _, probability in records]
    conflicts = [
        (base.target.count % 2) != required
        for (_, base, _), required in zip(records, required_toggle)
    ]
    conflict_indices = [index for index, value in enumerate(conflicts) if value]
    nonconflict_indices = [index for index, value in enumerate(conflicts) if not value]
    recovery = recovery_statistics(start_agreement, end_agreement)
    conflict_end = [end_agreement[index] for index in conflict_indices]
    # Recovery horizons are measured on the complete chronological trajectory,
    # not on a conflict-only subsequence (which would shorten the intervals).
    conflict_recovery = recovery_statistics_for_indices(conflict_indices, end_agreement)
    raw = sum(record.raw_real for record, _, _ in records)
    retained = sum(record.retained_real for record, _, _ in records)
    base_zero = [(record, base) for record, base, _ in records if base.target.count == 0]
    base_zero_conversion = sum(record.target_count == 1 and record.model_state != record.human_state for record, base in base_zero)
    return {
        "intervals": len(records),
        "base_count_accuracy": base_count["accuracy"], "base_count_macro_f1": base_count["macro_f1"],
        **{
            f"base_n{count}_{metric}": base_count["classes"][count][metric]
            for count in range(3) for metric in ("precision", "recall", "f1", "support")
        },
        "dynamic_count_accuracy": dynamic_count["accuracy"],
        "predicted_n0": predictions.count(0), "predicted_n1": predictions.count(1), "predicted_n2": predictions.count(2),
        "dynamic_target_n0": sum(record.target_count == 0 for record, _, _ in records),
        "dynamic_target_n1": sum(record.target_count == 1 for record, _, _ in records),
        "dynamic_target_n2": sum(record.target_count == 2 for record, _, _ in records),
        "interval_start_state_agreement": float(np.mean(start_agreement)),
        "interval_end_state_agreement": float(np.mean(end_agreement)),
        "state_mismatch_fraction": 1.0 - float(np.mean(start_agreement)),
        "hard_next_state_accuracy": float(np.mean(end_agreement)),
        "mismatch_run_count": len(runs), "mismatch_run_mean": float(np.mean(runs)) if runs else 0.0,
        "mismatch_run_median": float(np.median(runs)) if runs else 0.0,
        "mismatch_run_p95": float(np.quantile(runs, 0.95)) if runs else 0.0,
        "mismatch_run_max": max(runs, default=0),
        "active_timing_mae": float(np.mean(timing_errors)) if timing_errors else 0.0,
        "correction_fraction": sum(record.correction_required for record, _, _ in records) / len(records) if condition.startswith("C_") else 0.0,
        "base_n0_to_dynamic_n1": base_zero_conversion / len(base_zero) if base_zero and condition.startswith("C_") else 0.0,
        "required_toggle_0": required_toggle.count(0), "required_toggle_1": required_toggle.count(1),
        "predicted_hard_parity_accuracy": float(np.mean(np.asarray(required_toggle) == np.asarray(predicted_parity))),
        "mean_p_toggle_required": float(np.mean([value for value, required in zip(toggle_probabilities, required_toggle) if required])) if any(required_toggle) else 0.0,
        "mean_p_toggle_no_toggle": float(np.mean([value for value, required in zip(toggle_probabilities, required_toggle) if not required])) if not all(required_toggle) else 0.0,
        **recovery,
        "conflict_fraction": len(conflict_indices) / len(records),
        "conflict_base_n0": sum(base_targets[index] == 0 for index in conflict_indices),
        "conflict_base_n1": sum(base_targets[index] == 1 for index in conflict_indices),
        "conflict_base_n2": sum(base_targets[index] == 2 for index in conflict_indices),
        "conflict_pred_n0": sum(predictions[index] == 0 for index in conflict_indices),
        "conflict_pred_n1": sum(predictions[index] == 1 for index in conflict_indices),
        "conflict_pred_n2": sum(predictions[index] == 2 for index in conflict_indices),
        "conflict_count_accuracy": float(np.mean([predictions[index] == base_targets[index] for index in conflict_indices])) if conflict_indices else 1.0,
        "nonconflict_count_accuracy": float(np.mean([predictions[index] == base_targets[index] for index in nonconflict_indices])) if nonconflict_indices else 1.0,
        "conflict_next_state_agreement": float(np.mean(conflict_end)) if conflict_end else 1.0,
        "conflict_recovered_within_1": conflict_recovery["recovered_within_1"],
        "conflict_recovered_within_2": conflict_recovery["recovered_within_2"],
        "conflict_recovered_within_4": conflict_recovery["recovered_within_4"],
        "conflict_remain_mismatched": conflict_recovery["remain_mismatched"],
        "real_transition_retention": retained / raw if raw else 1.0,
        "finite": True,
    }


def gradient_audit(dataset, device) -> float:
    common.seed_everything(common.SEED)
    model = common.build_model(device)
    plans = [controller_plan("D_static_state_consistency", model, dataset, 0, device)]
    gradients = {}
    losses = {}
    for component in ("count", "state", "timing"):
        model.eval()
        model.zero_grad(set_to_none=True)
        result = replay(
            "D_static_state_consistency", model, dataset, plans, device,
            lambda_state=1.0, components={component}, backward=lambda value: value.backward(),
        )
        gradients[component] = common.gradient_snapshot(model)
        losses[component] = {
            "main": result[f"main_{component}"],
            "boundary": result[f"boundary_{component}"],
            "boundary_weight": BOUNDARY_WEIGHT,
            "audited_objective": result[f"main_{component}"] + BOUNDARY_WEIGHT * result[f"boundary_{component}"],
        }
        if not all(group["all_finite"] for group in gradients[component].values()):
            raise FloatingPointError(f"non-finite {component} gradient audit")
    count_encoder = gradients["count"]["encoder"]["global_l2"]
    count_heads = gradients["count"]["all_heads"]["global_l2"]
    ratios = {
        "R_state_encoder": gradients["state"]["encoder"]["global_l2"] / count_encoder,
        "R_state_heads": gradients["state"]["all_heads"]["global_l2"] / count_heads,
    }
    maximum = max(ratios.values())
    candidates = []
    selected = None
    for value in LAMBDA_CANDIDATES:
        effective = value * maximum
        eligible = effective <= 0.5
        candidates.append({"lambda_state": value, "R": maximum, "effective_ratio": effective, "eligible": eligible})
        if selected is None and eligible:
            selected = value
    if selected is None:
        atomic_json(OUTPUT / "state_loss_gradient_scale.json", {
            "status": "STOP", "ratios": ratios, "candidates": candidates,
            "reason": "lambda=0.1 still exceeds effective ratio 0.5",
        })
        raise RuntimeError("state gradient dominates even at lambda=0.1")
    atomic_json(OUTPUT / "state_loss_gradient_scale.json", {
        "status": "PASS", "subset_index": SUBSETS[0], "losses": losses,
        "gradients": gradients, "ratios": ratios, "R": maximum,
        "candidates": candidates, "selected_lambda_state": selected,
        "selection_rule": "largest lambda in [1,.5,.25,.1] with lambda*max(R_encoder,R_heads)<=.5",
        "optimizer_steps": 0, "asap_test_access": 0,
    })
    del model
    torch.cuda.empty_cache()
    log(f"GRADIENT_AUDIT_PASS R={maximum:.6f} lambda_state={selected}")
    return float(selected)


def distribution(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()), "median": float(np.median(array)),
        "std": float(array.std(ddof=0)), "min": float(array.min()), "max": float(array.max()),
        "p10": float(np.quantile(array, 0.10)), "p90": float(np.quantile(array, 0.90)),
    }


def classify_run(late: Mapping[str, Mapping[str, float]], rows: Sequence[Mapping[str, Any]]) -> str:
    base = late["base_count_accuracy"]
    macro = late["base_count_macro_f1"]
    state = late["interval_start_state_agreement"]
    mismatch = late["state_mismatch_fraction"]
    n1_fractions = [float(row["predicted_n1"]) / float(row["intervals"]) for row in rows]
    persistent_n1 = float(np.median(n1_fractions)) >= 0.90
    strong = (
        base["median"] >= 0.95 and macro["median"] >= 0.85
        and state["median"] >= 0.95 and state["p10"] >= 0.90
        and mismatch["median"] <= 0.05 and not persistent_n1
    )
    warning = base["median"] >= 0.90 and state["median"] >= 0.85 and not persistent_n1
    return "PASS" if strong else "WARN" if warning else "FAIL"


def aggregate_conflict_trajectory(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    conflict_counts = [
        sum(int(row[f"conflict_base_n{count}"]) for count in range(3))
        for row in rows
    ]
    total_conflicts = sum(conflict_counts)
    total_interval_observations = sum(int(row["intervals"]) for row in rows)
    result: dict[str, float | int] = {
        "optimizer_states": len(rows),
        "interval_observations": total_interval_observations,
        "conflict_observations": total_conflicts,
        "conflict_fraction": total_conflicts / total_interval_observations,
    }
    for horizon in (1, 2, 4):
        result[f"recovered_within_{horizon}"] = (
            sum(
                conflicts * float(row[f"conflict_recovered_within_{horizon}"])
                for conflicts, row in zip(conflict_counts, rows)
            ) / total_conflicts if total_conflicts else 1.0
        )
    result["remain_mismatched"] = (
        sum(
            conflicts * float(row["conflict_remain_mismatched"])
            for conflicts, row in zip(conflict_counts, rows)
        ) / total_conflicts if total_conflicts else 0.0
    )
    return result


def run_one(condition: str, canonical_index: int, lambda_state: float, device, curve_rows, mismatch_rows, conflict_rows):
    dataset = Binary2SlotPerformanceDataset(common.CACHE_ROOT, "train", performance_indices=(canonical_index,), cache_input_tokens=True)
    common.seed_everything(common.SEED)
    model = common.build_model(device)
    optimizer = common.build_optimizer(model)
    fingerprint = v0.head_fingerprint(model)
    run_rows: list[dict[str, Any]] = []
    for optimizer_step in range(1, STEPS + 1):
        plans = [controller_plan(condition, model, dataset, 0, device)]
        diagnostic = metrics(condition, plans)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = replay(
            condition, model, dataset, plans, device,
            lambda_state=lambda_state if condition.startswith("D_") else 0.0,
            backward=lambda value: value.backward(),
        )
        if not all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()):
            raise FloatingPointError(f"{condition}/{canonical_index}: non-finite gradient")
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), common.MAX_GRAD_NORM).item())
        if not math.isfinite(norm):
            raise FloatingPointError(f"{condition}/{canonical_index}: non-finite gradient norm")
        optimizer.step()
        row = {
            "condition": condition, "subset_index": canonical_index, "step": optimizer_step - 1,
            "total_loss": losses["total"], "main_count_loss": losses["main_count"],
            "main_timing_loss": losses["main_timing"], "main_state_loss": losses["main_state"] if condition.startswith("D_") else 0.0,
            "boundary_count_loss": losses["boundary_count"], "boundary_timing_loss": losses["boundary_timing"],
            "boundary_state_loss": losses["boundary_state"] if condition.startswith("D_") else 0.0,
            **diagnostic,
        }
        run_rows.append(row)
        curve_rows.append(row)
        if (optimizer_step - 1) % WRITE_EVERY == 0:
            mismatch_rows.append({
                "condition": condition, "subset_index": canonical_index, "step": optimizer_step - 1,
                **{key: diagnostic[key] for key in (
                    "mismatch_run_count", "mismatch_run_mean", "mismatch_run_median", "mismatch_run_p95", "mismatch_run_max",
                    "recovered_within_1", "recovered_within_2", "recovered_within_4", "remain_mismatched",
                )},
            })
            if condition.startswith("D_"):
                conflict_rows.append({
                    "subset_index": canonical_index, "step": optimizer_step - 1,
                    **{key: diagnostic[key] for key in diagnostic if key.startswith("conflict_")},
                })
            atomic_csv(OUTPUT / ("training_curves_C.csv" if condition.startswith("C_") else "training_curves_D.csv"), [value for value in curve_rows if value["condition"] == condition])
            atomic_csv(OUTPUT / "state_mismatch_runs.csv", mismatch_rows)
            atomic_csv(OUTPUT / "condition_D_conflict_intervals.csv", conflict_rows)
            log(
                f"{condition} subset={canonical_index} step={optimizer_step - 1} "
                f"base_acc={diagnostic['base_count_accuracy']:.4f} macro={diagnostic['base_count_macro_f1']:.4f} "
                f"state={diagnostic['interval_start_state_agreement']:.4f} mismatch={diagnostic['state_mismatch_fraction']:.4f}"
            )
    plans = [controller_plan(condition, model, dataset, 0, device)]
    diagnostic = metrics(condition, plans)
    model.eval()
    with torch.no_grad():
        losses = replay(condition, model, dataset, plans, device, lambda_state=lambda_state if condition.startswith("D_") else 0.0)
    final_row = {
        "condition": condition, "subset_index": canonical_index, "step": STEPS,
        "total_loss": losses["total"], "main_count_loss": losses["main_count"],
        "main_timing_loss": losses["main_timing"], "main_state_loss": losses["main_state"] if condition.startswith("D_") else 0.0,
        "boundary_count_loss": losses["boundary_count"], "boundary_timing_loss": losses["boundary_timing"],
        "boundary_state_loss": losses["boundary_state"] if condition.startswith("D_") else 0.0,
        **diagnostic,
    }
    run_rows.append(final_row)
    curve_rows.append(final_row)
    mismatch_rows.append({
        "condition": condition, "subset_index": canonical_index, "step": STEPS,
        **{key: diagnostic[key] for key in (
            "mismatch_run_count", "mismatch_run_mean", "mismatch_run_median", "mismatch_run_p95", "mismatch_run_max",
            "recovered_within_1", "recovered_within_2", "recovered_within_4", "remain_mismatched",
        )},
    })
    if condition.startswith("D_"):
        conflict_rows.append({"subset_index": canonical_index, "step": STEPS, **{key: diagnostic[key] for key in diagnostic if key.startswith("conflict_")}})
    late_rows = [row for row in run_rows if int(row["step"]) >= LATE_START]
    if len(late_rows) != 200:
        raise AssertionError(f"late window must contain exactly 200 optimizer states, got {len(late_rows)}")
    late = {
        metric: distribution([float(row[metric]) for row in late_rows])
        for metric in (
            "base_count_accuracy", "base_count_macro_f1", "interval_start_state_agreement",
            "state_mismatch_fraction", "active_timing_mae",
        )
    }
    status = classify_run(late, late_rows)
    performance = dataset.performances[0]
    summary = {
        "condition": condition, "subset_index": canonical_index, "performance_id": dataset.timeline(0).performance_id,
        "notes": performance.num_notes, "onsets": performance.num_onsets,
        "windows": len(performance.ownership.window_starts), "status": status,
        "head_initialization_fingerprint": fingerprint, "steps": STEPS,
        "late": late, "final": final_row,
        "all_training_conflict": aggregate_conflict_trajectory(run_rows),
        "full_training": 0, "validation_inference": 0, "asap_test_access": 0,
    }
    del optimizer, model, dataset
    torch.cuda.empty_cache()
    log(f"RUN_COMPLETE {condition} subset={canonical_index} status={status} late_state_median={late['interval_start_state_agreement']['median']:.4f}")
    return summary


def aggregate_status(runs: Sequence[Mapping[str, Any]]) -> str:
    counts = Counter(run["status"] for run in runs)
    if counts["PASS"] >= 2 and counts["FAIL"] == 0:
        return "PASS"
    if counts["FAIL"] >= 2:
        return "FAIL"
    return "WARN"


def write_report(c_runs, d_runs, lambda_state, config):
    c_status, d_status = aggregate_status(c_runs), aggregate_status(d_runs)
    c_state = [run["late"]["interval_start_state_agreement"]["median"] for run in c_runs]
    d_state = [run["late"]["interval_start_state_agreement"]["median"] for run in d_runs]
    d_count = [run["late"]["base_count_accuracy"]["median"] for run in d_runs]
    d_macro = [run["late"]["base_count_macro_f1"]["median"] for run in d_runs]
    d_conflict = [run["all_training_conflict"]["conflict_fraction"] for run in d_runs]
    d_total_conflicts = sum(run["all_training_conflict"]["conflict_observations"] for run in d_runs)
    d_total_intervals = sum(run["all_training_conflict"]["interval_observations"] for run in d_runs)
    d_aggregate_conflict = d_total_conflicts / d_total_intervals
    d_aggregate_recovery = {
        horizon: sum(
            run["all_training_conflict"]["conflict_observations"]
            * run["all_training_conflict"][f"recovered_within_{horizon}"]
            for run in d_runs
        ) / d_total_conflicts
        for horizon in (1, 2, 4)
    }
    if c_status == "PASS" and d_status != "PASS":
        interpretation = "CASE E: C passes; D is insufficient"
        recommendation = "Condition C"
    elif d_status == "PASS" and c_status != "PASS":
        interpretation = "CASE D: D passes while C remains unstable"
        recommendation = "Condition D"
    elif c_status == "PASS" and d_status == "PASS":
        interpretation = "CASE C: both pass; prefer the more stable/static objective"
        recommendation = "Condition D" if np.median(d_state) >= np.median(c_state) else "Condition C"
    elif c_status == "PASS":
        interpretation = "CASE A: C remains viable and D is not better"
        recommendation = "Condition C"
    else:
        interpretation = "CASE F: neither formulation earned an aggregate PASS"
        recommendation = "none; state-policy design review"
    table_rows = []
    for run in c_runs + d_runs:
        table_rows.append(
            f"| {run['condition'][0]} | {run['subset_index']} | {run['status']} | "
            f"{run['late']['base_count_accuracy']['median']:.4f} | {run['late']['base_count_macro_f1']['median']:.4f} | "
            f"{run['late']['interval_start_state_agreement']['median']:.4f} | {run['late']['interval_start_state_agreement']['p10']:.4f} | "
            f"{run['late']['interval_start_state_agreement']['min']:.4f} | {run['late']['active_timing_mae']['median']:.4f} |"
        )
    report = f"""# Binary 2-Slot State-Stability Diagnostic v1

## Summary

| formulation | aggregate status | individual statuses | late state median by subset |
|---|---:|---|---|
| C: online reconciliation + base-count weights | {c_status} | {', '.join(run['status'] for run in c_runs)} | {', '.join(f'{value:.4f}' for value in c_state)} |
| D: static targets + local state consistency | {d_status} | {', '.join(run['status'] for run in d_runs)} | {', '.join(f'{value:.4f}' for value in d_state)} |

Selected `lambda_state={lambda_state}` by the frozen initial-gradient rule. The same train-only subsets {list(SUBSETS)}, pretrained checkpoint, seed-42 heads, optimizer, LR, FP32 mode, 1,000 steps, and boundary weight 0.25 were used independently for C and D. No weights were transferred.

| formulation | subset | status | late base acc median | late macro F1 median | late state median | late state p10 | late state min | late timing MAE median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table_rows)}

## Scientific interpretation

Condition D keeps the actual hard model state as head input and updates it only with argmax Count parity, exactly as inference. Count/Timing labels are static base targets, so model errors never mutate later labels. Its auxiliary loss does use the human interval-end state during training to provide a local differentiable parity signal; therefore D does not eliminate every train/inference information difference and does not create a soft recurrent state.

Interpretation: **{interpretation}**. Recommended next full-training formulation: **{recommendation}**. No full training was started.

## Required questions

**Q1. Was C instability reproduced?** Aggregate {c_status}; individual statuses {', '.join(run['status'] for run in c_runs)}, with late state medians {', '.join(f'{value:.4f}' for value in c_state)}. See per-run p10/min/std in `late_stability_summary.csv`.

**Q2. Selected lambda_state?** {lambda_state}, chosen only from the initial gradient ratio; no performance sweep was run.

**Q3. Can D memorize static Count/Timing under hard free-running?** Aggregate {d_status}; late median Count accuracies {', '.join(f'{value:.4f}' for value in d_count)}, Macro F1 {', '.join(f'{value:.4f}' for value in d_macro)}. Timing medians are tabulated above.

**Q4. Is D hard state stable late?** Late state medians were {', '.join(f'{value:.4f}' for value in d_state)}; full p10/min/std determine the aggregate gate.

**Q5. Does D show a 50/50 attractor?** {'Yes or recurrently unstable; D did not pass.' if d_status == 'FAIL' else 'No persistent aggregate 50/50 attractor met the FAIL criterion, though WARN runs remain blockers.' if d_status == 'WARN' else 'No; all three runs cleared the stability gate.'}

**Q6. Count/state conflict frequency?** Across all 1,001 optimizer states per run, interval-weighted D fractions by subset were {', '.join(f'{value:.2%}' for value in d_conflict)}; the combined optimizer-state×interval fraction was {d_aggregate_conflict:.2%}. At the final state all three subsets had zero conflict intervals.

**Q7. Conflict recovery within 1/2/4 intervals?** Conflict-observation-weighted rates across the complete training trajectories were {d_aggregate_recovery[1]:.2%} / {d_aggregate_recovery[2]:.2%} / {d_aggregate_recovery[4]:.2%}. These include the deliberately retained unstable initialization period; by the final state no conflict remained. Per-step/subset values are in `condition_D_conflict_intervals.csv`.

**Q8. Does State loss recreate synthetic N1?** D never mutates Count labels. Compare predicted N1, conflict Count accuracy, and required-toggle probabilities in the detailed curves; the aggregate verdict is {d_status}.

**Q9. Better late stability?** Median across subset medians: C={np.median(c_state):.4f}, D={np.median(d_state):.4f}. Recommendation also respects Macro F1, mismatch runs, timing, and static-objective simplicity.

**Q10. Recommended formulation?** {recommendation}.

**Q11. Remaining blocker?** {'No diagnostic blocker, but explicit authorization and a guarded full-run config are still required.' if recommendation != 'none; state-policy design review' and (c_status == 'PASS' or d_status == 'PASS') else 'Yes. Neither formulation has earned a stable aggregate PASS; do not full train.'}

## Provenance and prohibited activity

Six independent train-only runs completed: 6,000 optimizer steps total. Full-training steps/epochs: 0. Validation inference/metrics: 0. ASAP test metadata/MIDI: 0/0. Checkpoints and checkpoint selection: 0. State-loss lambda candidates were evaluated only by the mandated initial gradient-scale calculation, never by training performance.
"""
    (OUTPUT / "BINARY_2SLOT_STATE_STABILITY_DIAGNOSTIC.md").write_text(report, encoding="utf-8")
    return c_status, d_status, recommendation


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("existing single CUDA device is required")
    device = torch.device("cuda:0")
    selections = []
    for index in SUBSETS:
        dataset = Binary2SlotPerformanceDataset(common.CACHE_ROOT, "train", performance_indices=(index,), cache_input_tokens=False)
        performance = dataset.performances[0]
        selections.append({
            "canonical_train_index": index, "performance_id": dataset.timeline(0).performance_id,
            "notes": performance.num_notes, "onsets": performance.num_onsets,
            "owner_windows": len(performance.ownership.window_starts),
            "selection_provenance": "subset 7 reuses v0; 6/13 were pre-existing gradient-audit indices, chosen before model results for different multi-window lengths",
        })
        del dataset
    if any(item["owner_windows"] < 2 for item in selections):
        raise RuntimeError("all stability subsets must be multi-window")
    config = {
        "experiment": "binary_2slot_state_stability_diagnostic_v1", "created_at": now(),
        "subsets": selections, "conditions": list(CONDITIONS), "steps_per_run": STEPS,
        "runs": 6, "seed": 42, "boundary_loss_weight": BOUNDARY_WEIGHT,
        "encoder_lr": common.ENCODER_LR, "head_lr": common.HEAD_LR,
        "weight_decay": common.WEIGHT_DECAY, "gradient_clip": common.MAX_GRAD_NORM,
        "precision": "FP32, matching prior tiny diagnostics", "fixed_weights": list(FIXED_COUNT_WEIGHTS),
        "state_loss_candidates": list(LAMBDA_CANDIDATES), "late_window_steps": [LATE_START, STEPS],
        "full_training": 0, "validation_inference": 0, "asap_test_access": 0,
    }
    atomic_json(OUTPUT / "run_config.json", config)
    audit_dataset = Binary2SlotPerformanceDataset(common.CACHE_ROOT, "train", performance_indices=(SUBSETS[0],), cache_input_tokens=True)
    lambda_state = gradient_audit(audit_dataset, device)
    config["selected_lambda_state"] = lambda_state
    atomic_json(OUTPUT / "run_config.json", config)
    del audit_dataset
    curve_rows: list[dict[str, Any]] = []
    mismatch_rows: list[dict[str, Any]] = []
    conflict_rows: list[dict[str, Any]] = []
    summaries: dict[str, list[dict[str, Any]]] = {condition: [] for condition in CONDITIONS}
    for condition in CONDITIONS:
        for subset in SUBSETS:
            summaries[condition].append(run_one(condition, subset, lambda_state, device, curve_rows, mismatch_rows, conflict_rows))
    fingerprints = {run["head_initialization_fingerprint"] for runs in summaries.values() for run in runs}
    if len(fingerprints) != 1:
        raise AssertionError("six runs did not independently reproduce seed-42 heads")
    c_runs, d_runs = summaries[CONDITIONS[0]], summaries[CONDITIONS[1]]
    atomic_csv(OUTPUT / "training_curves_C.csv", [row for row in curve_rows if row["condition"].startswith("C_")])
    atomic_csv(OUTPUT / "training_curves_D.csv", [row for row in curve_rows if row["condition"].startswith("D_")])
    atomic_csv(OUTPUT / "state_mismatch_runs.csv", mismatch_rows)
    atomic_csv(OUTPUT / "condition_D_conflict_intervals.csv", conflict_rows)
    run_fields = []
    late_rows = []
    for run in c_runs + d_runs:
        run_fields.append({
            "condition": run["condition"], "subset_index": run["subset_index"], "performance_id": run["performance_id"],
            "notes": run["notes"], "onsets": run["onsets"], "windows": run["windows"], "status": run["status"],
            "final_base_count_accuracy": run["final"]["base_count_accuracy"],
            "final_macro_f1": run["final"]["base_count_macro_f1"],
            "final_state_agreement": run["final"]["interval_start_state_agreement"],
            "final_timing_mae": run["final"]["active_timing_mae"],
            "all_training_conflict_fraction": run["all_training_conflict"]["conflict_fraction"],
            "all_training_conflict_recovery_1": run["all_training_conflict"]["recovered_within_1"],
            "all_training_conflict_recovery_2": run["all_training_conflict"]["recovered_within_2"],
            "all_training_conflict_recovery_4": run["all_training_conflict"]["recovered_within_4"],
        })
        for metric, values in run["late"].items():
            late_rows.append({"condition": run["condition"], "subset_index": run["subset_index"], "metric": metric, **values})
    atomic_csv(OUTPUT / "condition_C_runs.csv", [row for row in run_fields if row["condition"].startswith("C_")])
    atomic_csv(OUTPUT / "condition_D_runs.csv", [row for row in run_fields if row["condition"].startswith("D_")])
    atomic_csv(OUTPUT / "late_stability_summary.csv", late_rows)
    c_status, d_status, recommendation = write_report(c_runs, d_runs, lambda_state, config)
    atomic_csv(OUTPUT / "comparison_summary.csv", [{
        "formulation": "C", "aggregate_status": c_status,
        "pass_runs": sum(run["status"] == "PASS" for run in c_runs),
        "warn_runs": sum(run["status"] == "WARN" for run in c_runs),
        "fail_runs": sum(run["status"] == "FAIL" for run in c_runs),
        "late_state_median_across_runs": float(np.median([run["late"]["interval_start_state_agreement"]["median"] for run in c_runs])),
        "recommended": recommendation == "Condition C",
    }, {
        "formulation": "D", "aggregate_status": d_status,
        "pass_runs": sum(run["status"] == "PASS" for run in d_runs),
        "warn_runs": sum(run["status"] == "WARN" for run in d_runs),
        "fail_runs": sum(run["status"] == "FAIL" for run in d_runs),
        "late_state_median_across_runs": float(np.median([run["late"]["interval_start_state_agreement"]["median"] for run in d_runs])),
        "recommended": recommendation == "Condition D",
    }])
    atomic_json(OUTPUT / "run_status.json", {
        "status": "COMPLETE", "completed_at": now(), "C_status": c_status, "D_status": d_status,
        "selected_lambda_state": lambda_state, "recommendation": recommendation,
        "optimizer_steps": 6000, "full_training": 0, "validation_inference": 0,
        "asap_test_metadata_access": 0, "asap_test_midi_access": 0, "checkpoints": 0,
    })
    log(f"DIAGNOSTIC_COMPLETE C={c_status} D={d_status} recommendation={recommendation}; no full training")


if __name__ == "__main__":
    main()
