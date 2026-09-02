#!/usr/bin/env python3
"""Restore and verify Condition D with static PRE + MAIN + POST targets.

This is a train-only focused diagnostic.  It never runs validation inference,
touches the ASAP test split, creates checkpoints, or launches full training.
"""

from __future__ import annotations

import argparse
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_binary_2slot_state_stability_diagnostic_v1 as v1
from scripts import run_stage2_binary_2slot_full_v0 as common
from src.stage2_binary_2slot.dataset import Binary2SlotPerformanceDataset, HumanIntervalPrimitive
from src.stage2_binary_2slot.rollout import next_binary_state
from src.stage2_binary_2slot.static_targets import (
    StaticIntervalTarget,
    build_static_pre_target,
    build_static_timeline_targets,
)
from src.stage2_event_tokenizer.binary_2slot import OFF


OUTPUT = ROOT / "analysis/binary_2slot_D_pre_main_post_restore_smoke_v0"
TINY_INDEX = 7
STEPS = 1000
LOG_EVERY = 25
LATE_START = 801
LAMBDA_STATE = 0.25
BOUNDARY_WEIGHT = 0.25
CONDITION = "D_static_pre_main_post_restore"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    line = f"{now()} {message}"
    print(line, flush=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "restore_smoke.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
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
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


@dataclass(frozen=True)
class StaticBaseRecord:
    state: int
    target: StaticIntervalTarget


@dataclass(frozen=True)
class StaticPlan:
    plan: common.PerformancePlan
    base: Mapping[int, StaticBaseRecord]
    count_probabilities: Mapping[int, tuple[float, float, float]]


def static_base_chain(
    dataset: Binary2SlotPerformanceDataset, performance_index: int
) -> dict[int, StaticBaseRecord]:
    """Construct a model-independent target chain; no online target API is used."""

    timeline = dataset.timeline(performance_index)
    targets = build_static_timeline_targets(timeline)
    state = OFF
    result: dict[int, StaticBaseRecord] = {}
    for interval, target in zip(timeline.represented_intervals, targets):
        result[interval.global_index] = StaticBaseRecord(state, target)
        state = next_binary_state(state, target.count)
        if state != interval.human_end_state:
            raise AssertionError("static base chain failed interval-end invariant")
    return result


def make_record(model, hidden, interval: HumanIntervalPrimitive, state: int, base: StaticBaseRecord):
    logits, timing = common._select_predictions(model, hidden.reshape(1, -1), state)
    if not bool(torch.isfinite(logits).all() and torch.isfinite(timing).all()):
        raise FloatingPointError("non-finite restored Condition D prediction")
    target = base.target
    predicted = int(logits[0].argmax().item())
    probabilities = tuple(float(value) for value in logits[0].softmax(-1).float().cpu().tolist())
    record = common.IntervalRecord(
        region=interval.region,
        global_index=interval.global_index,
        main_onset_index=interval.main_onset_index,
        model_state=state,
        human_state=interval.human_start_state,
        target_count=target.count,
        timing_targets=target.timing_targets,
        timing_mask=target.timing_mask,
        predicted_count=predicted,
        predicted_timing=tuple(float(value) for value in timing[0].float().cpu().tolist()),
        correction_required=False,
        retained_real=target.retained_real_human_transitions,
        raw_real=target.raw_human_transitions,
    )
    return record, probabilities, next_binary_state(state, predicted)


def controller_plan(
    model,
    dataset,
    performance_index: int,
    device: torch.device,
    *,
    amp: bool = False,
) -> StaticPlan:
    """Oracle-free hard state lifecycle PRE -> owned MAIN -> POST."""

    model.eval()
    timeline = dataset.timeline(performance_index)
    base = static_base_chain(dataset, performance_index)
    state = OFF
    probabilities: dict[int, tuple[float, float, float]] = {}
    with torch.inference_mode():
        hidden = common.encode_boundary(model, dataset, performance_index, "PRE", device, amp=amp)
        pre, probability, state = make_record(model, hidden, timeline.pre, state, base[timeline.pre.global_index])
        probabilities[pre.global_index] = probability
        windows: list[common.WindowRecord] = []
        seen: list[int] = []
        for indices in common.chunks(common.window_dataset_indices(dataset, performance_index)):
            samples = [dataset[index] for index in indices]
            hidden_values = common.encode_window_samples(model, samples, device, amp=amp)
            for dataset_index, sample, window_hidden in zip(indices, samples, hidden_values):
                before = state
                records = []
                globals_ = [int(value) for value in sample["owned_global_onset_indices"].tolist()]
                if globals_ != sorted(globals_) or (seen and globals_ and globals_[0] <= seen[-1]):
                    raise AssertionError("restored owner chronology violation")
                seen.extend(globals_)
                for local_index, interval in enumerate(sample["human_intervals"]):
                    record, probability, state = make_record(
                        model, window_hidden[local_index], interval, state, base[interval.global_index]
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
            raise AssertionError("restored MAIN supervision has missing/duplicate onsets")
        if any(left.state_after != right.state_before for left, right in zip(windows, windows[1:])):
            raise AssertionError("restored state reset at owner-window boundary")
        hidden = common.encode_boundary(model, dataset, performance_index, "POST", device, amp=amp)
        post, probability, state = make_record(model, hidden, timeline.post, state, base[timeline.post.global_index])
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
    if len(plan.records) != len(base) or set(probabilities) != set(base):
        raise AssertionError("restored interval/base/probability identity mismatch")
    if plan.windows and plan.pre is not None:
        pre_end = next_binary_state(plan.pre.model_state, plan.pre.predicted_count)
        if pre_end != plan.windows[0].state_before:
            raise AssertionError("state reset between PRE and first MAIN window")
        final_main_end = plan.windows[-1].state_after
        if final_main_end != plan.post.model_state:
            raise AssertionError("state reset between final MAIN and POST")
    return StaticPlan(plan, base, probabilities)


def as_v1_plan(item: StaticPlan) -> v1.StabilityPlan:
    # v1 replay/metrics are structural and use only state/target fields shared by
    # StaticBaseRecord.  Target creation remains entirely in this script/module.
    return v1.StabilityPlan(item.plan, item.base, item.count_probabilities)  # type: ignore[arg-type]


def runs(values: Sequence[bool]) -> list[int]:
    result: list[int] = []
    active = 0
    for value in list(values) + [True]:
        if value:
            if active:
                result.append(active)
                active = 0
        else:
            active += 1
    return result


def recovery(start: Sequence[bool], end: Sequence[bool]) -> dict[str, float]:
    indices = [index for index, value in enumerate(start) if not value]
    result: dict[str, float] = {}
    for horizon in (1, 2, 4, 8):
        result[f"recovered_within_{horizon}"] = (
            sum(any(end[index:min(len(end), index + horizon)]) for index in indices) / len(indices)
            if indices else 1.0
        )
    result["remain_mismatched"] = (
        sum(not any(end[index:]) for index in indices) / len(indices) if indices else 0.0
    )
    return result


def regional_metrics(item: StaticPlan, region: str) -> dict[str, Any]:
    values = [record for record in item.plan.records if record.region == region]
    targets = [item.base[record.global_index].target.count for record in values]
    predictions = [record.predicted_count for record in values]
    count = v1.count_metrics(targets, predictions)
    start = [record.model_state == record.human_state for record in values]
    end = [
        next_binary_state(record.model_state, record.predicted_count)
        == item.base[record.global_index].target.human_end_state
        for record in values
    ]
    mismatch_runs = runs(start)
    timing = [
        abs(record.predicted_timing[slot] - record.timing_targets[slot])
        for record in values for slot in range(2) if record.timing_mask[slot]
    ]
    return {
        "intervals": len(values),
        "count_accuracy": count["accuracy"],
        "count_macro_f1": count["macro_f1"],
        "state_agreement": float(np.mean(start)),
        "end_state_agreement": float(np.mean(end)),
        "mismatch_fraction": 1.0 - float(np.mean(start)),
        "mismatch_run_count": len(mismatch_runs),
        "mismatch_run_mean": float(np.mean(mismatch_runs)) if mismatch_runs else 0.0,
        "mismatch_run_median": float(np.median(mismatch_runs)) if mismatch_runs else 0.0,
        "mismatch_run_p95": float(np.quantile(mismatch_runs, 0.95)) if mismatch_runs else 0.0,
        "mismatch_run_max": max(mismatch_runs, default=0),
        "timing_mae": float(np.mean(timing)) if timing else 0.0,
        "predicted_n0": predictions.count(0),
        "predicted_n1": predictions.count(1),
        "predicted_n2": predictions.count(2),
        **recovery(start, end),
    }


def snapshot(item: StaticPlan, losses: Mapping[str, float], step: int) -> dict[str, Any]:
    overall = v1.metrics(CONDITION, [as_v1_plan(item)])
    pre = regional_metrics(item, "PRE")
    main = regional_metrics(item, "MAIN")
    post = regional_metrics(item, "POST")
    pre_record = item.plan.pre
    pre_target = item.base[pre_record.global_index].target
    pre_end = next_binary_state(pre_record.model_state, pre_record.predicted_count)
    first_main = next(record for record in item.plan.records if record.region == "MAIN")
    row: dict[str, Any] = {
        "step": step,
        "total_loss": losses["total"],
        "main_loss": losses["main"],
        "boundary_loss": losses["boundary"],
        "main_count_loss": losses["main_count"],
        "main_timing_loss": losses["main_timing"],
        "main_state_loss": losses["main_state"],
        "boundary_count_loss": losses["boundary_count"],
        "boundary_timing_loss": losses["boundary_timing"],
        "boundary_state_loss": losses["boundary_state"],
        "base_count_accuracy": overall["base_count_accuracy"],
        "base_count_macro_f1": overall["base_count_macro_f1"],
        "state_agreement": overall["interval_start_state_agreement"],
        "timing_mae": overall["active_timing_mae"],
        "predicted_n0": overall["predicted_n0"],
        "predicted_n1": overall["predicted_n1"],
        "predicted_n2": overall["predicted_n2"],
        "pre_left_human_state": item.plan.pre.human_state,
        "pre_static_correction_inserted": pre_target.static_initial_correction_inserted,
        "pre_target_count": pre_record.target_count,
        "pre_predicted_count": pre_record.predicted_count,
        "pre_result_state_t1": pre_end,
        "human_state_t1": pre_target.human_end_state,
        "pre_sync_accuracy": float(pre_end == pre_target.human_end_state),
        "first_main_state_agreement": float(first_main.model_state == first_main.human_state),
    }
    for prefix, metrics in (("main", main), ("post", post), ("pre", pre)):
        for key, value in metrics.items():
            row[f"{prefix}_{key}"] = value
    if not all(math.isfinite(float(value)) for value in row.values() if isinstance(value, (int, float))):
        raise FloatingPointError("non-finite restored smoke snapshot")
    return row


def audit_split(split: str) -> dict[str, Any]:
    dataset = Binary2SlotPerformanceDataset(common.CACHE_ROOT, split, cache_input_tokens=False)
    corrections = 0
    retained = 0
    violations: list[dict[str, Any]] = []
    pre_raw = Counter()
    for index in range(len(dataset.performances)):
        timeline = dataset.timeline(index)
        try:
            targets = build_static_timeline_targets(timeline)
            pre_target = targets[0]
            result = next_binary_state(OFF, pre_target.count)
            if result != timeline.pre.human_end_state:
                raise AssertionError("PRE state at t1 differs from human")
            corrections += int(pre_target.static_initial_correction_inserted)
            retained += int(pre_target.static_initial_correction_retained)
            pre_raw[len(timeline.pre.human_transitions)] += 1
        except Exception as error:  # Preserve representative failures in artifact.
            violations.append({
                "index": index,
                "performance_id": timeline.performance_id,
                "error": f"{type(error).__name__}: {error}",
            })
            if len(violations) >= 20:
                break
    result = {
        "split": split,
        "performances": len(dataset.performances),
        "pre_static_correction_inserted": corrections,
        "pre_static_correction_inserted_fraction": corrections / len(dataset.performances),
        "pre_static_correction_retained": retained,
        "pre_raw_transition_counts": {str(key): value for key, value in sorted(pre_raw.items())},
        "pre_invariant_violations": len(violations),
        "violation_examples": violations,
    }
    del dataset
    return result


def pre_target_audit() -> dict[str, Any]:
    result = {
        "definition": "OFF + optional static PRE-left synchronization + actual PRE crossings, then frozen max-2",
        "online_reconciliation_calls": 0,
        "train": audit_split("train"),
        "validation": audit_split("validation"),
        "asap_test_access": 0,
    }
    subset = Binary2SlotPerformanceDataset(
        common.CACHE_ROOT, "train", performance_indices=(TINY_INDEX,), cache_input_tokens=False
    )
    timeline = subset.timeline(0)
    target = build_static_pre_target(timeline.pre)
    state_t1 = next_binary_state(OFF, target.count)
    result["subset_7"] = {
        "canonical_train_index": TINY_INDEX,
        "performance_id": timeline.performance_id,
        "pre_left_human_state": timeline.pre.human_start_state,
        "static_correction_inserted": target.static_initial_correction_inserted,
        "static_correction_retained": target.static_initial_correction_retained,
        "pre_raw_transition_count": len(timeline.pre.human_transitions),
        "pre_compression_input_count": len(timeline.pre.human_transitions) + int(target.static_initial_correction_inserted),
        "pre_retained_transition_count": target.count,
        "pre_retained_parity": target.count % 2,
        "pre_retained_taus": [event.tau for event in target.retained_transitions],
        "pre_retained_directions": [event.direction for event in target.retained_transitions],
        "oracle_decoded_state_at_t1": state_t1,
        "human_state_immediately_before_t1": timeline.pre.human_end_state,
        "invariant_pass": state_t1 == timeline.pre.human_end_state,
        "human_state_at_first_main_start": timeline.main[0].human_start_state,
    }
    del subset
    if result["train"]["pre_invariant_violations"] or result["validation"]["pre_invariant_violations"]:
        raise AssertionError("canonical PRE target invariant failed")
    if not result["subset_7"]["invariant_pass"]:
        raise AssertionError("subset 7 PRE target invariant failed")
    atomic_json(OUTPUT / "pre_target_audit.json", result)
    return result


def distribution(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()), "median": float(np.median(array)),
        "std": float(array.std(ddof=0)), "min": float(array.min()),
        "max": float(array.max()), "p10": float(np.quantile(array, 0.1)),
        "p90": float(np.quantile(array, 0.9)),
    }


def run_smoke(device: torch.device) -> dict[str, Any]:
    def reject_online_reconciliation(*args, **kwargs):
        raise AssertionError("online reconciliation is prohibited in restored Condition D")

    # v1 is reused only for its verified loss/metric reductions. Turn its
    # legacy online-target symbol into a runtime tripwire for this smoke.
    v1.build_online_target = reject_online_reconciliation
    dataset = Binary2SlotPerformanceDataset(
        common.CACHE_ROOT, "train", performance_indices=(TINY_INDEX,), cache_input_tokens=True
    )
    common.seed_everything(common.SEED)
    model = common.build_model(device)
    optimizer = common.build_optimizer(model)
    rows: list[dict[str, Any]] = []
    for optimizer_step in range(1, STEPS + 1):
        plan = controller_plan(model, dataset, 0, device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = v1.replay(
            CONDITION,
            model,
            dataset,
            [as_v1_plan(plan)],
            device,
            lambda_state=LAMBDA_STATE,
            backward=lambda value: value.backward(),
        )
        if not all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()):
            raise FloatingPointError("non-finite restored smoke gradient")
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), common.MAX_GRAD_NORM).item())
        if not math.isfinite(norm):
            raise FloatingPointError("non-finite restored smoke gradient norm")
        row = snapshot(plan, losses, optimizer_step - 1)
        row["gradient_norm"] = norm
        rows.append(row)
        optimizer.step()
        if (optimizer_step - 1) % LOG_EVERY == 0:
            atomic_csv(OUTPUT / "smoke_metrics.csv", rows)
            log(
                f"step={optimizer_step - 1} loss={row['total_loss']:.6f} "
                f"main_acc={row['main_count_accuracy']:.4f} main_state={row['main_state_agreement']:.4f} "
                f"pre_sync={row['pre_sync_accuracy']:.0f}"
            )

    plan = controller_plan(model, dataset, 0, device)
    model.eval()
    with torch.no_grad():
        losses = v1.replay(CONDITION, model, dataset, [as_v1_plan(plan)], device, lambda_state=LAMBDA_STATE)
    final = snapshot(plan, losses, STEPS)
    final["gradient_norm"] = rows[-1]["gradient_norm"]
    rows.append(final)
    atomic_csv(OUTPUT / "smoke_metrics.csv", rows)
    late = [row for row in rows if int(row["step"]) >= LATE_START]
    if len(late) != 200:
        raise AssertionError(f"expected 200 late optimizer states, got {len(late)}")
    late_summary = {
        key: distribution([float(row[key]) for row in late])
        for key in (
            "main_count_accuracy", "main_count_macro_f1", "main_state_agreement",
            "main_mismatch_fraction", "main_timing_mae", "pre_sync_accuracy",
            "first_main_state_agreement",
        )
    }
    n1_fraction = float(np.median([row["predicted_n1"] / (row["main_intervals"] + 2) for row in late]))
    strong = (
        late_summary["main_count_accuracy"]["median"] >= 0.95
        and late_summary["main_count_macro_f1"]["median"] >= 0.85
        and late_summary["main_state_agreement"]["median"] >= 0.95
        and late_summary["main_state_agreement"]["p10"] >= 0.90
        and late_summary["first_main_state_agreement"]["median"] == 1.0
        and n1_fraction < 0.90
    )
    warning = (
        late_summary["main_count_accuracy"]["median"] >= 0.90
        and late_summary["main_state_agreement"]["median"] >= 0.85
        and late_summary["first_main_state_agreement"]["median"] >= 0.90
        and n1_fraction < 0.90
    )
    status = "PASS" if strong else "WARN" if warning else "FAIL"
    summary = {
        "status": status,
        "condition": CONDITION,
        "canonical_train_index": TINY_INDEX,
        "performance_id": dataset.timeline(0).performance_id,
        "steps": STEPS,
        "lambda_state": LAMBDA_STATE,
        "boundary_loss_weight": BOUNDARY_WEIGHT,
        "optimizer": "AdamW",
        "encoder_lr": common.ENCODER_LR,
        "head_lr": common.HEAD_LR,
        "weight_decay": common.WEIGHT_DECAY,
        "gradient_clip": common.MAX_GRAD_NORM,
        "initial": rows[0],
        "final": final,
        "late": late_summary,
        "late_predicted_n1_fraction_median": n1_fraction,
        "cold_start_global_parity_offset_present": late_summary["main_state_agreement"]["median"] < 0.5,
        "persistent_50_50_attractor": 0.45 <= late_summary["main_state_agreement"]["median"] <= 0.55,
        "persistent_n1_collapse": n1_fraction >= 0.90,
        "online_reconciliation_calls": 0,
        "dynamic_target_count": 0,
        "optimizer_steps": STEPS,
        "full_training_steps": 0,
        "validation_inference": 0,
        "asap_test_access": 0,
    }
    atomic_json(OUTPUT / "smoke_summary.json", summary)
    del optimizer, model, dataset
    torch.cuda.empty_cache()
    return summary


def write_report(config: Mapping[str, Any], audit: Mapping[str, Any], smoke: Mapping[str, Any]) -> None:
    subset = audit["subset_7"]
    final = smoke["final"]
    late = smoke["late"]
    old_path = ROOT / "analysis/stage2_binary_2slot_static_state_main_full_v0/main_only_smoke_summary.json"
    old = json.loads(old_path.read_text(encoding="utf-8")) if old_path.exists() else {}
    old_final = old.get("final", old)
    old_diagnostics = old_final.get("diagnostics", old_final)
    report = f"""# Binary 2-Slot Condition D PRE/MAIN/POST Restore Smoke v0

## Outcome

| check | result |
|---|---:|
| status | **{smoke['status']}** |
| subset | canonical TRAIN 7 |
| optimizer steps | {smoke['steps']} |
| late MAIN Count accuracy median | {late['main_count_accuracy']['median']:.6f} |
| late MAIN Macro F1 median | {late['main_count_macro_f1']['median']:.6f} |
| late MAIN state agreement median / p10 | {late['main_state_agreement']['median']:.6f} / {late['main_state_agreement']['p10']:.6f} |
| late first-MAIN agreement median | {late['first_main_state_agreement']['median']:.6f} |
| final MAIN timing MAE | {final['main_timing_mae']:.6f} |
| online reconciliation / dynamic target | 0 / 0 |
| full training / validation inference / ASAP test | 0 / 0 / 0 |

## Frozen restoration

The restored path predicts PRE, every owned MAIN interval including the final `[t_M,T_end)` tail, and POST with one continuous hard state. Count/Timing targets are immutable static human targets. The state-consistency loss is the previously verified parity loss with `lambda_state=0.25`; total loss is `MAIN + 0.25 * BOUNDARY`, with Count, Timing, and `0.25 * State` inside each region. Human state is used only for static target construction, the local loss label, and diagnostics.

PRE uses a dataset-fixed initial synchronization event only when the human state at `t1-1s` is ON. It is prepended before real PRE events and undergoes ordinary max-2 compression without reservation. Inference receives no human state and performs no forced correction.

## Canonical PRE target audit

Train: {audit['train']['performances']} performances, {audit['train']['pre_static_correction_inserted']} static corrections, {audit['train']['pre_invariant_violations']} violations. Validation target-only audit: {audit['validation']['performances']} performances, {audit['validation']['pre_static_correction_inserted']} corrections, {audit['validation']['pre_invariant_violations']} violations. ASAP test was not accessed.

Subset 7 (`{subset['performance_id']}`): PRE-left human state={subset['pre_left_human_state']}, correction inserted={subset['static_correction_inserted']}, raw PRE crossings={subset['pre_raw_transition_count']}, retained Count={subset['pre_retained_transition_count']}, retained parity={subset['pre_retained_parity']}, oracle PRE result at t1={subset['oracle_decoded_state_at_t1']}, human state at t1={subset['human_state_immediately_before_t1']}. Invariant: **{subset['invariant_pass']}**.

## Cold-start comparison

The prior MAIN-only run began OFF while subset 7 human `t1` state was ON, producing Count accuracy {old_diagnostics.get('count_accuracy', old_diagnostics.get('main_count_accuracy', 'n/a'))} but state agreement {old_diagnostics.get('state_agreement', old_diagnostics.get('main_state_agreement', 'n/a'))}. That reproduced a global parity-offset cold start rather than a Count-capacity failure.

After restoring PRE, final PRE prediction Count={final['pre_predicted_count']} against target={final['pre_target_count']}; resulting t1 state={final['pre_result_state_t1']} and human t1 state={final['human_state_t1']}. First-MAIN agreement={final['first_main_state_agreement']:.6f}. Final MAIN Count accuracy={final['main_count_accuracy']:.6f}, Macro F1={final['main_count_macro_f1']:.6f}, state agreement={final['main_state_agreement']:.6f}, mismatch fraction={final['main_mismatch_fraction']:.6f}. The late median state agreement is {late['main_state_agreement']['median']:.6f}; 50/50 attractor={smoke['persistent_50_50_attractor']}, N=1 collapse={smoke['persistent_n1_collapse']}.

## Required conclusions

1. **MAIN-only root cause reproduced/resolved?** The previous OFF-vs-ON initialization mismatch is documented above. Restored PRE {'resolved the learned cold-start parity offset.' if smoke['status'] == 'PASS' else 'did not fully resolve the cold-start trajectory under the gate.'}
2. **Was a static PRE correction generated on subset 7?** {subset['static_correction_inserted']}; exact retained details are in `pre_target_audit.json`.
3. **Was first MAIN synchronized?** Final={bool(final['first_main_state_agreement'])}; late median={late['first_main_state_agreement']['median']:.6f}.
4. **Was hard free-running MAIN stable?** Late state median/p10={late['main_state_agreement']['median']:.6f}/{late['main_state_agreement']['p10']:.6f}, mismatch-run maximum at final={final['main_mismatch_run_max']}.
5. **Was online reconciliation used?** No. Static builders accept no model state or prediction; runtime counters are zero.
6. **Proceed to full training?** **{smoke['status']}**. {'The focused diagnostic found no blocker; full training still requires a separate user task.' if smoke['status'] == 'PASS' else 'Do not launch full training until the reported warning/failure is reviewed.'}

## Provenance and stop point

Seed 42, pretrained PT encoder, AdamW encoder/head LR `1e-5/1e-4`, weight decay `0.01`, gradient clip `1.0`, FP32, fixed Count weights, `lambda_state=0.25`, and boundary weight `0.25` match the prior diagnostic. This task ran exactly one 1,000-step TRAIN-only smoke. It created no checkpoint, ran no full training, ran no validation inference/metric, and accessed no ASAP test data.
"""
    (OUTPUT / "BINARY_2SLOT_D_PRE_MAIN_POST_RESTORE_REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("audit", "smoke", "all"), default="all")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    config = {
        "experiment": "binary_2slot_D_pre_main_post_restore_smoke_v0",
        "created_at": now(),
        "canonical_train_subset": [TINY_INDEX],
        "steps": STEPS,
        "seed": common.SEED,
        "lambda_state": LAMBDA_STATE,
        "boundary_loss_weight": BOUNDARY_WEIGHT,
        "count_weights": [0.7577816791288143, 2.246984238727598, 4.428054342343362],
        "encoder_lr": common.ENCODER_LR,
        "head_lr": common.HEAD_LR,
        "weight_decay": common.WEIGHT_DECAY,
        "gradient_clip": common.MAX_GRAD_NORM,
        "precision": "FP32, matching Condition D diagnostic v1",
        "timeline": ["PRE", "all MAIN including final tail", "POST"],
        "target": "model-independent static base; PRE-only MIDI-derived initial synchronization",
        "online_reconciliation": False,
        "dynamic_target": False,
        "full_training": 0,
        "validation_inference": 0,
        "asap_test_access": 0,
    }
    atomic_json(OUTPUT / "RESTORE_CONFIG.json", config)
    audit = pre_target_audit() if args.mode in {"audit", "all"} else json.loads((OUTPUT / "pre_target_audit.json").read_text())
    if args.mode == "audit":
        log("TARGET_AUDIT_PASS; stopping before smoke")
        return
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("one existing CUDA device is required for focused smoke")
    smoke = run_smoke(torch.device("cuda:0"))
    write_report(config, audit, smoke)
    log(f"RESTORE_SMOKE_COMPLETE status={smoke['status']}; no full training")


if __name__ == "__main__":
    main()
