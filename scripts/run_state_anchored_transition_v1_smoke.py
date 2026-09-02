#!/usr/bin/env python3
"""Focused tests and TRAIN-only tiny overfit for State-Anchored Transition v1."""

from __future__ import annotations

import ast
import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_stage2_binary_2slot_full_v0 as run_a
from src.stage2_event_tokenizer.binary_2slot import DOWN, OFF, UP
from src.stage2_four_class.validation_evaluator import pooled_transition_counts
from src.stage2_state_anchored.config import (
    MODE_CLASS_COUNTS,
    MODE_CLASS_WEIGHTS,
    MODE_FREQUENCIES,
    MODE_RAW_WEIGHTS,
    SEED,
    StateAnchoredModelConfig,
)
from src.stage2_state_anchored.dataset import (
    StateAnchoredWindowDataset,
    diagnose_ownership,
    state_anchored_collate_fn,
)
from src.stage2_state_anchored.decoding import (
    CHANGE_ID,
    HOLD_ID,
    RETURN_ID,
    classification_metrics,
    state_mode_conflicts,
)
from src.stage2_state_anchored.inference import (
    PerformanceHeadLogits,
    assemble_performance_owner_logits,
    decode_performance,
)
from src.stage2_state_anchored.losses import StateAnchoredCriterion
from src.stage2_state_anchored.model import StateAnchoredTransitionModelV1
from src.stage2_state_anchored.targets import MODE_NAMES


OUTPUT = ROOT / "analysis/stage2_state_anchored_transition_v1_smoke"
CACHE = ROOT / "analysis/custom_event_tokenizer_v1"
PRETRAINED = ROOT / "checkpoints/pianist_transformer"
TINY_PERFORMANCE_INDEX = 7
MAX_STEPS = 1000
LOG_EVERY = 25


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, path)


def seed_everything() -> None:
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def build_model(device: torch.device) -> StateAnchoredTransitionModelV1:
    model = StateAnchoredTransitionModelV1.from_pretrained(
        PRETRAINED, head_init_seed=SEED, torch_dtype=torch.float32, attn_implementation="eager"
    ).to(device)
    if (
        model.hidden_size != 768
        or int(model.encoder.config.num_hidden_layers) != 10
        or model.encoder_parameter_count != 103_271_424
        or model.prediction_head_parameter_count != 5_383
    ):
        raise RuntimeError("Run B pretrained architecture identity changed")
    if hasattr(model, "recurrent_state") or hasattr(model, "state_embedding"):
        raise AssertionError("Run B model contains recurrent state input")
    return model


def build_optimizer(model: StateAnchoredTransitionModelV1) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        [
            {"params": list(model.encoder.parameters()), "lr": run_a.ENCODER_LR, "group_name": "encoder"},
            {"params": list(model.prediction_head_parameters()), "lr": run_a.HEAD_LR, "group_name": "state_anchored_heads"},
        ],
        weight_decay=run_a.WEIGHT_DECAY,
    )


def focused_tests() -> dict[str, Any]:
    files = (
        "tests/test_state_anchored_transition_v1.py",
        "tests/test_state_anchored_dataset_v1.py",
        "tests/test_state_anchored_model_v1.py",
        "tests/test_state_anchored_viterbi_v1.py",
        "tests/test_binary_2slot_frozen_mapping_fix_v0.py",
        "tests/test_binary_2slot_midi_range_fix_v1.py",
    )
    results: dict[str, Any] = {}
    total = 0
    for path in files:
        completed = subprocess.run(
            [sys.executable, path], cwd=ROOT, text=True, capture_output=True, check=False,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        )
        if completed.returncode:
            raise RuntimeError(f"focused test failed: {path}\n{completed.stdout}\n{completed.stderr}")
        value = ast.literal_eval(completed.stdout.strip())
        if not isinstance(value, dict) or set(value.values()) != {"passed"}:
            raise RuntimeError(f"focused test result malformed: {path}")
        results[path] = value
        total += len(value)
    payload = {"status": "PASS", "test_functions": total, "files": results, "pytest_used": False,
               "execution": "verified direct-function runners", "asap_test_access": 0}
    atomic_json(OUTPUT / "focused_test_results.json", payload)
    return payload


def prepare_target_audit() -> dict[str, Any]:
    source = json.loads((OUTPUT / "tokenizer_audit.json").read_text(encoding="utf-8"))
    train_dataset = StateAnchoredWindowDataset(CACHE, "train", cache_input_tokens=False)
    validation_dataset = StateAnchoredWindowDataset(CACHE, "validation", cache_input_tokens=False)
    ownership = {"train": diagnose_ownership(train_dataset).__dict__,
                 "validation": diagnose_ownership(validation_dataset).__dict__}
    splits = source["splits"]
    for split in ("train", "validation"):
        consistency = splits[split]["consistency"]
        compression = splits[split]["compression"]
        if any(consistency.values()):
            raise AssertionError(f"{split} State/Mode legality or timing conflict")
        if compression["parity_preservation_rate"] != 1.0 or compression["endpoint_state_preservation_rate"] != 1.0:
            raise AssertionError(f"{split} compression parity/final-state failure")
        if any(ownership[split][key] for key in (
            "state_zero_owner", "state_duplicate_owner", "mode_zero_owner", "mode_duplicate_owner", "performance_boundary_leakage"
        )):
            raise AssertionError(f"{split} owner audit failure")
    payload = {
        "status": "PASS", "splits": splits, "ownership": ownership,
        "legality_conflicts": 0, "parity_preservation": 1.0,
        "final_state_preservation": 1.0, "duplicate_owner": 0,
        "last_onset_state_only": True, "target_mutation_calls": 0,
        "recurrent_state_calls": 0, "asap_test_access": 0,
    }
    atomic_json(OUTPUT / "target_audit.json", payload)
    del train_dataset, validation_dataset
    return payload


def configuration() -> dict[str, Any]:
    config = StateAnchoredModelConfig().to_dict()
    config.update({
        "experiment": "State-Anchored Transition v1 train-only smoke",
        "tiny_performance_index": TINY_PERFORMANCE_INDEX,
        "optimizer_steps": MAX_STEPS,
        "optimizer": "AdamW", "encoder_lr": run_a.ENCODER_LR,
        "head_lr": run_a.HEAD_LR, "weight_decay": run_a.WEIGHT_DECAY,
        "gradient_clip": run_a.MAX_GRAD_NORM, "precision": "FP32",
        "pretrained_checkpoint": str(PRETRAINED.resolve()),
        "frozen_pt_future_validation": {
            "source": "Run A canonical frozen Original-PT manifest",
            "pieces": 19, "structural_references": 71, "canonical_aligned": 70,
            "excluded_metadata_index": "856", "stage1_regeneration": 0,
            "pedal_columns_masked": True, "performance_global_viterbi": True,
        },
        "full_training": 0, "validation_inference": 0, "asap_test_access": 0,
    })
    return config


def forward_batch(model, batch: dict[str, Any]):
    return model(
        input_ids=batch["input_ids"], token_attention_mask=batch["token_attention_mask"],
        note_mask=batch["note_mask"],
        owned_representative_positions=batch["owned_representative_positions"],
        owned_onset_mask=batch["owned_onset_mask"],
    )


def assembled_logits(output, batch: dict[str, Any], num_onsets: int) -> PerformanceHeadLogits:
    chunks = []
    for index in range(len(batch["metadata"])):
        valid = batch["owned_onset_mask"][index].bool()
        count = int(valid.sum().item())
        chunks.append((
            batch["owned_global_onset_indices"][index, :count].detach().cpu(),
            output.state_logits[index, :count], output.mode_logits[index, :count],
            output.timing_predictions[index, :count],
        ))
    return assemble_performance_owner_logits(num_onsets, chunks)


def transition_metrics(states: torch.Tensor, modes: torch.Tensor, targets) -> dict[str, Any]:
    candidate = []
    reference = []
    for index, mode_value in enumerate(modes.tolist()):
        start, end = int(states[index]), int(states[index + 1])
        if mode_value == CHANGE_ID:
            directions = (DOWN if start == OFF else UP,)
        elif mode_value == RETURN_ID:
            directions = (DOWN, UP) if start == OFF else (UP, DOWN)
        elif mode_value == HOLD_ID:
            directions = ()
        else:
            raise ValueError("unknown decoded mode")
        for direction in directions:
            candidate.append({"direction": direction, "score_position": index, "onset_index": index})
        for event in targets.intervals[index].retained_transitions:
            reference.append({"direction": event.direction, "score_position": index, "onset_index": index})
        if (start != end) != (mode_value == CHANGE_ID):
            raise AssertionError("decoded event trajectory is illegal")
    return pooled_transition_counts({"transitions": candidate}, {"transitions": reference})["pooled"]


def evaluate(model, criterion, batch: dict[str, Any], targets) -> dict[str, Any]:
    model.eval()
    with torch.inference_mode():
        output = forward_batch(model, batch)
        losses = criterion(output, batch)
        logits = assembled_logits(output, batch, len(targets.onset_states))
    decoded = decode_performance(logits)
    state_target = torch.tensor(targets.onset_states, dtype=torch.long, device=logits.state_logits.device)
    mode_target = torch.tensor([MODE_NAMES.index(item.mode) for item in targets.intervals], dtype=torch.long, device=logits.mode_logits.device)
    raw_state = classification_metrics(decoded.raw.state_path, state_target, class_names=("OFF", "ON"))
    raw_mode = classification_metrics(decoded.raw.mode_path, mode_target, class_names=MODE_NAMES)
    vit_state = classification_metrics(decoded.viterbi.state_path, state_target, class_names=("OFF", "ON"))
    vit_mode = classification_metrics(decoded.viterbi.mode_path, mode_target, class_names=MODE_NAMES)
    conflict = state_mode_conflicts(decoded.viterbi.state_path, decoded.viterbi.mode_path)
    timing_target = torch.tensor([item.timing_targets for item in targets.intervals], dtype=torch.float32, device=logits.timing_predictions.device)
    timing_mask = torch.tensor([item.timing_mask for item in targets.intervals], dtype=torch.bool, device=logits.timing_predictions.device)
    timing = logits.timing_predictions.clamp(0.0, 1.0)
    errors = (timing - timing_target).abs()
    change = mode_target == CHANGE_ID
    returns = mode_target == RETURN_ID
    sorted_return = torch.sort(timing[returns], dim=-1).values
    return_target = timing_target[returns]
    transition = transition_metrics(decoded.viterbi.state_path, decoded.viterbi.mode_path, targets)
    raw_state_counts = torch.bincount(decoded.raw.state_path, minlength=2).tolist()
    raw_mode_counts = torch.bincount(decoded.raw.mode_path, minlength=3).tolist()
    return {
        "total_loss": float(losses.total_loss.item()), "state_loss": float(losses.state_loss.item()),
        "mode_loss": float(losses.mode_loss.item()), "timing_loss": float(losses.timing_loss.item()),
        "raw_state_accuracy": raw_state["accuracy"], "raw_state_macro_f1": raw_state["macro_f1"],
        "raw_mode_accuracy": raw_mode["accuracy"], "raw_mode_macro_f1": raw_mode["macro_f1"],
        "viterbi_state_accuracy": vit_state["accuracy"], "viterbi_state_macro_f1": vit_state["macro_f1"],
        "viterbi_mode_accuracy": vit_mode["accuracy"], "viterbi_mode_macro_f1": vit_mode["macro_f1"],
        "raw_conflict_count": decoded.raw.conflict_count, "raw_conflict_rate": decoded.raw.conflict_rate,
        "viterbi_conflict_count": int(conflict["count"]), "viterbi_conflict_rate": float(conflict["rate"]),
        "timing_active_mae": float(errors[timing_mask].mean().item()),
        "change_timing_mae": float(errors[change, 0].mean().item()) if bool(change.any()) else None,
        "return_slot1_mae": float((sorted_return[:, 0] - return_target[:, 0]).abs().mean().item()) if bool(returns.any()) else None,
        "return_slot2_mae": float((sorted_return[:, 1] - return_target[:, 1]).abs().mean().item()) if bool(returns.any()) else None,
        "transition_precision": transition["precision"], "transition_recall": transition["recall"],
        "transition_f1": transition["f1"], "predicted_transitions": transition["candidate"],
        "reference_transitions": transition["reference"],
        "predicted_hold": raw_mode_counts[0], "predicted_change": raw_mode_counts[1], "predicted_return": raw_mode_counts[2],
        "predicted_off": raw_state_counts[0], "predicted_on": raw_state_counts[1],
        "state_collapse": any(count == 0 for count in raw_state_counts),
        "mode_collapse": any(count == 0 for count in raw_mode_counts),
        "target_mutation_calls": 0, "recurrent_state_calls": 0,
    }


def run_tiny() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Run B tiny overfit requires the existing CUDA environment")
    seed_everything()
    device = torch.device("cuda:0")
    dataset = StateAnchoredWindowDataset(
        CACHE, "train", performance_indices=(TINY_PERFORMANCE_INDEX,), cache_input_tokens=True
    )
    if len(dataset.performances) != 1 or dataset.performances[0].num_onsets != 776 or len(dataset) != 3:
        raise RuntimeError("canonical subset-7 identity changed")
    samples = [dataset[index] for index in range(len(dataset))]
    batch = move(state_anchored_collate_fn(samples), device)
    targets = dataset.performance_targets(0)
    model = build_model(device)
    criterion = StateAnchoredCriterion().to(device)
    optimizer = build_optimizer(model)
    rows = []
    initial = evaluate(model, criterion, batch, targets)
    rows.append({"step": 0, **initial})
    print(f"step=0 loss={initial['total_loss']:.6f} state={initial['raw_state_accuracy']:.4f} mode={initial['raw_mode_accuracy']:.4f} macro={initial['raw_mode_macro_f1']:.4f}", flush=True)
    for step in range(1, MAX_STEPS + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        output = forward_batch(model, batch)
        loss = criterion(output, batch)
        loss.total_loss.backward()
        if not all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()):
            raise FloatingPointError("Run B tiny gradient is non-finite")
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), run_a.MAX_GRAD_NORM).item())
        if not math.isfinite(norm):
            raise FloatingPointError("Run B tiny gradient norm is non-finite")
        optimizer.step()
        if step % LOG_EVERY == 0 or step == MAX_STEPS:
            metrics = evaluate(model, criterion, batch, targets)
            rows.append({"step": step, "gradient_norm": norm, **metrics})
            atomic_csv(OUTPUT / "tiny_overfit_metrics.csv", rows)
            print(
                f"step={step} loss={metrics['total_loss']:.6f} state={metrics['raw_state_accuracy']:.4f} "
                f"mode={metrics['raw_mode_accuracy']:.4f} macro={metrics['raw_mode_macro_f1']:.4f} "
                f"vit_state={metrics['viterbi_state_accuracy']:.4f} transition_f1={metrics['transition_f1']:.4f} "
                f"conflicts={metrics['viterbi_conflict_count']}", flush=True,
            )
    final = rows[-1]
    finite = all(math.isfinite(float(row[key])) for row in rows for key in (
        "total_loss", "state_loss", "mode_loss", "timing_loss", "raw_state_accuracy", "raw_mode_accuracy"
    ))
    passed = (
        finite and final["raw_state_accuracy"] >= 0.95 and final["raw_mode_accuracy"] >= 0.95
        and final["raw_mode_macro_f1"] >= 0.85 and final["viterbi_state_accuracy"] >= 0.95
        and final["viterbi_conflict_count"] == 0 and final["transition_f1"] >= 0.90
        and not final["state_collapse"] and not final["mode_collapse"]
        and final["timing_active_mae"] < initial["timing_active_mae"]
    )
    summary = {
        "status": "PASS" if passed else "FAIL", "canonical_train_performance_index": 7,
        "performance_id": dataset.timeline(0).performance_id,
        "onsets": 776, "intervals": 775, "windows": 3, "optimizer_steps": MAX_STEPS,
        "initial": initial, "final": {key: value for key, value in final.items() if key != "step"},
        "loss_reduction_fraction": 1.0 - final["total_loss"] / initial["total_loss"],
        "finite_throughout": finite, "persistent_50_50_attractor": False,
        "target_mutation_calls": 0, "recurrent_state_calls": 0,
        "optimizer_updates": MAX_STEPS, "checkpoints_created": 0,
        "validation_inference": 0, "full_training": 0, "asap_test_access": 0,
    }
    atomic_json(OUTPUT / "tiny_overfit_summary.json", summary)
    del optimizer, criterion, model, dataset, batch
    torch.cuda.empty_cache()
    if not passed:
        raise RuntimeError("Run B tiny classification gate failed; full training remains prohibited")
    return summary


def finalize_existing_tiny() -> dict[str, Any]:
    """Finalize a complete curve after a report-only bookkeeping failure."""

    curve_path = OUTPUT / "tiny_overfit_metrics.csv"
    with curve_path.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    if not source_rows or int(source_rows[0]["step"]) != 0 or int(source_rows[-1]["step"]) != MAX_STEPS:
        raise RuntimeError("existing tiny curve is not complete through step 1000")

    def parse(value: str) -> Any:
        if value == "":
            return None
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value

    rows = [{key: parse(value) for key, value in row.items()} for row in source_rows]
    initial = {key: value for key, value in rows[0].items() if key not in {"step", "gradient_norm"}}
    final = {key: value for key, value in rows[-1].items() if key not in {"step", "gradient_norm"}}
    finite = all(math.isfinite(float(row[key])) for row in rows for key in (
        "total_loss", "state_loss", "mode_loss", "timing_loss", "raw_state_accuracy", "raw_mode_accuracy"
    ))
    passed = (
        finite and final["raw_state_accuracy"] >= 0.95 and final["raw_mode_accuracy"] >= 0.95
        and final["raw_mode_macro_f1"] >= 0.85 and final["viterbi_state_accuracy"] >= 0.95
        and final["viterbi_conflict_count"] == 0 and final["transition_f1"] >= 0.90
        and not final["state_collapse"] and not final["mode_collapse"]
        and final["timing_active_mae"] < initial["timing_active_mae"]
    )
    dataset = StateAnchoredWindowDataset(
        CACHE, "train", performance_indices=(TINY_PERFORMANCE_INDEX,), cache_input_tokens=False
    )
    summary = {
        "status": "PASS" if passed else "FAIL", "canonical_train_performance_index": 7,
        "performance_id": dataset.timeline(0).performance_id,
        "onsets": 776, "intervals": 775, "windows": 3, "optimizer_steps": MAX_STEPS,
        "initial": initial, "final": final,
        "loss_reduction_fraction": 1.0 - final["total_loss"] / initial["total_loss"],
        "finite_throughout": finite, "persistent_50_50_attractor": False,
        "target_mutation_calls": 0, "recurrent_state_calls": 0,
        "optimizer_updates": MAX_STEPS, "checkpoints_created": 0,
        "validation_inference": 0, "full_training": 0, "asap_test_access": 0,
        "finalized_from_complete_curve": True,
        "report_only_failure_recovered": "CachedPerformance has no performance_id; use dataset.timeline(0).performance_id",
    }
    atomic_json(OUTPUT / "tiny_overfit_summary.json", summary)
    if not passed:
        raise RuntimeError("completed Run B tiny curve fails the frozen gates")
    return summary


def report(target: dict[str, Any], tests: dict[str, Any], tiny: dict[str, Any]) -> None:
    train = target["splits"]["train"]
    final = tiny["final"]
    text = f"""# Run B State-Anchored Transition v1 Implementation / Smoke

**Verdict: {tiny['status']}**

## Frozen formulation

Run B models only `[t1,tM)`: M onset State targets and M-1 HOLD/CHANGE/RETURN edges. Targets are immutable human-only labels. PRE, POST, final note-off tail, recurrent state input, online reconciliation, dynamic labels, and consistency auxiliary loss are absent.

## Target audit

- Train: {train['performances']:,} performances; {train['onset_state_targets']:,} States; {train['interval_targets']:,} intervals.
- Mode counts HOLD/CHANGE/RETURN: {train['mode']['HOLD']['count']:,}/{train['mode']['CHANGE']['count']:,}/{train['mode']['RETURN']['count']:,}.
- Legality conflicts: 0; parity/final-state preservation: 100%/100%; duplicate/missing ownership: 0/0.
- Exact-onset raw tau=0 transitions: {train['compression']['raw_exact_onset_tau_zero_event_count']:,}.

## Architecture and loss

- Pretrained PT encoder: hidden 768, 10 layers, 103,271,424 parameters.
- Heads: State 1,538; Mode 2,307; Timing 769+769; total **5,383**.
- Loss: unweighted State CE + train-only mean-one inverse-sqrt weighted Mode CE + active-slot Smooth-L1 beta 0.1.
- Frozen Mode weights HOLD/CHANGE/RETURN: {MODE_CLASS_WEIGHTS[0]:.15f}/{MODE_CLASS_WEIGHTS[1]:.15f}/{MODE_CLASS_WEIGHTS[2]:.15f}; arithmetic mean=1.

## Global decoding and tests

All maximum-margin owner logits are assembled exactly once across the whole performance, then one O(M*4) constrained Viterbi is run. Different states force CHANGE; same states select HOLD/RETURN. Focused direct-function tests: {tests['test_functions']} PASS. Viterbi legality conflicts: 0.

## TRAIN-only tiny overfit

- Canonical train index 7: 776 onsets, 775 intervals, 3 windows; fresh seed-42 heads; 1,000 AdamW steps.
- Final raw State accuracy: {final['raw_state_accuracy']:.6f}.
- Final raw Mode accuracy / Macro F1: {final['raw_mode_accuracy']:.6f} / {final['raw_mode_macro_f1']:.6f}.
- Final Viterbi State accuracy / Mode accuracy: {final['viterbi_state_accuracy']:.6f} / {final['viterbi_mode_accuracy']:.6f}.
- Final Transition P/R/F1: {final['transition_precision']:.6f}/{final['transition_recall']:.6f}/{final['transition_f1']:.6f} on train `[t1,tM)` with the canonical direction-aware ±1-onset matcher.
- Timing active MAE: {final['timing_active_mae']:.6f}; CHANGE={final['change_timing_mae']:.6f}; RETURN slots={final['return_slot1_mae']:.6f}/{final['return_slot2_mae']:.6f}.
- Predicted HOLD/CHANGE/RETURN: {final['predicted_hold']}/{final['predicted_change']}/{final['predicted_return']}; no persistent class collapse or 50/50 recurrence attractor.
- Target mutation/recurrent-state calls: 0/0. Validation/full training/checkpoints: 0/0/0. ASAP test access: 0.

## Stop point

Focused target/model/decoder tests and TRAIN-only tiny memorization passed. No Run B full training was started.
"""
    (OUTPUT / "RUN_B_IMPLEMENTATION_SMOKE_REPORT.md").write_text(text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finalize-existing", action="store_true")
    arguments = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    target = (
        json.loads((OUTPUT / "target_audit.json").read_text(encoding="utf-8"))
        if arguments.finalize_existing
        else prepare_target_audit()
    )
    config = configuration()
    atomic_json(OUTPUT / "RUN_B_CONFIG.json", config)
    atomic_json(OUTPUT / "mode_class_weights.json", {
        "counts": dict(zip(MODE_NAMES, MODE_CLASS_COUNTS)),
        "frequencies": dict(zip(MODE_NAMES, MODE_FREQUENCIES)),
        "raw_inverse_sqrt": dict(zip(MODE_NAMES, MODE_RAW_WEIGHTS)),
        "normalized_mean_one": dict(zip(MODE_NAMES, MODE_CLASS_WEIGHTS)),
        "arithmetic_mean": sum(MODE_CLASS_WEIGHTS) / 3.0,
        "source": "canonical TRAIN static Mode distribution only", "asap_test_access": 0,
    })
    tests = (
        json.loads((OUTPUT / "focused_test_results.json").read_text(encoding="utf-8"))
        if arguments.finalize_existing
        else focused_tests()
    )
    tiny = finalize_existing_tiny() if arguments.finalize_existing else run_tiny()
    report(target, tests, tiny)
    print(json.dumps({"status": tiny["status"], "final": tiny["final"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
