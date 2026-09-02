#!/usr/bin/env python3
"""Bounded train-only memorization diagnostic for Custom Event Model v0."""

from __future__ import annotations

import csv
import json
import math
import os
import random
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage2_event_model.dataset import (
    CANONICAL_CACHE_ID,
    CustomEventWindowDataset,
    custom_event_collate_fn,
)
from src.stage2_event_model.losses import (
    CustomEventCriterion,
    CustomEventLossConfig,
    load_slot_event_weights,
)
from src.stage2_event_model.model import CustomEventEncoderModelV0
from src.stage2_event_tokenizer.tokenizer import EVENT_NAMES

CACHE_ROOT = ROOT / "analysis/custom_event_tokenizer_v1"
OUTPUT_ROOT = ROOT / "analysis/custom_event_model_v0_tiny_overfit"
SUBSET_PATH = OUTPUT_ROOT / "tiny_subset_manifest.json"
CHECKPOINT = ROOT / "checkpoints/pianist_transformer"
WEIGHTS = CACHE_ROOT / "candidate_event_weights.json"
SEED = 42
MAX_STEPS = 1000
LOG_INTERVAL = 25
ENCODER_LR = 1e-5
HEAD_LR = 1e-4
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty training log")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def forward_model(model: CustomEventEncoderModelV0, batch: dict[str, Any]):
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


def load_selected_samples() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    subset = json.loads(SUBSET_PATH.read_text(encoding="utf-8"))
    if subset["cache_id"] != CANONICAL_CACHE_ID or subset["split"] != "train":
        raise RuntimeError("tiny subset cache/split mismatch")
    if subset["validation_access_count"] != 0 or subset["asap_test_access_count"] != 0:
        raise RuntimeError("tiny subset reports forbidden validation/test access")
    selected = subset["selected"]
    last_entry = max(int(row["manifest_entry_index"]) for row in selected)
    dataset = CustomEventWindowDataset(
        CACHE_ROOT,
        "train",
        entry_limit=last_entry + 1,
        cache_input_tokens=True,
    )
    lookup = {pair: index for index, pair in enumerate(dataset.windows)}
    samples = []
    for row in selected:
        pair = (int(row["manifest_entry_index"]), int(row["window_index"]))
        sample = dataset[lookup[pair]]
        metadata = sample["metadata"]
        if metadata["performance_path"] != row["performance_path"]:
            raise AssertionError("tiny subset performance identity mismatch")
        if metadata["window_start_note"] != row["window_start_note"]:
            raise AssertionError("tiny subset window identity mismatch")
        samples.append(sample)
    return subset, samples


def assert_masks(samples: list[dict[str, Any]]) -> None:
    initial = 0
    terminal = 0
    main_non_none = [0] * 6
    terminal_non_none = [0] * 4
    for sample in samples:
        initial += int(sample["initial_valid_mask"])
        terminal += int(sample["terminal_valid_mask"])
        if not torch.equal(
            sample["main_timing_valid_mask"], sample["main_event_targets"] != 0
        ):
            raise AssertionError("NONE main timing would contribute")
        if not torch.equal(
            sample["terminal_timing_valid_mask"],
            sample["terminal_event_targets"] != 0,
        ):
            raise AssertionError("NONE terminal timing would contribute")
        for slot in range(6):
            main_non_none[slot] += int(
                (sample["main_event_targets"][:, slot] != 0).sum()
            )
        if sample["terminal_valid_mask"]:
            for slot in range(4):
                terminal_non_none[slot] += int(
                    sample["terminal_event_targets"][slot] != 0
                )
    if initial < 1 or terminal < 1 or min(main_non_none) < 1 or min(terminal_non_none) < 1:
        raise AssertionError("tiny subset does not cover every required supervision path")


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def event_summary(targets: list[torch.Tensor], predictions: list[torch.Tensor]) -> dict[str, Any]:
    target = torch.cat([value.reshape(-1).cpu() for value in targets])
    prediction = torch.cat([value.reshape(-1).cpu() for value in predictions])
    correct = int((target == prediction).sum())
    total = int(target.numel())
    target_positive = target != 0
    predicted_positive = prediction != 0
    tp = int((target_positive & predicted_positive).sum())
    fp = int((~target_positive & predicted_positive).sum())
    fn = int((target_positive & ~predicted_positive).sum())
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    recalls = {}
    for index, name in enumerate(EVENT_NAMES):
        mask = target == index
        recalls[name] = safe_div(int(((prediction == index) & mask).sum()), int(mask.sum()))
    return {
        "target_count": total,
        "correct_count": correct,
        "accuracy": safe_div(correct, total),
        "non_none_precision": precision,
        "non_none_recall": recall,
        "non_none_f1": f1,
        "target_non_none_count": int(target_positive.sum()),
        "predicted_non_none_count": int(predicted_positive.sum()),
        "all_none_prediction": bool(int(predicted_positive.sum()) == 0),
        "class_recall": recalls,
    }


def timing_summary(
    targets: list[torch.Tensor], predictions: list[torch.Tensor], *, seconds: bool = False
) -> dict[str, Any]:
    if not targets:
        return {"valid_count": 0, "mae": 0.0, "rmse": 0.0}
    target = torch.cat([value.reshape(-1).float().cpu() for value in targets])
    prediction = torch.cat([value.reshape(-1).float().cpu() for value in predictions])
    difference = prediction - target
    result = {
        "valid_count": int(target.numel()),
        "mae": float(difference.abs().mean()),
        "rmse": float(torch.sqrt(torch.mean(difference.square()))),
    }
    if seconds:
        target_seconds = torch.expm1(target.clamp(min=0))
        prediction_seconds = torch.expm1(prediction.clamp(min=0))
        result["seconds_gap_mae"] = float((prediction_seconds - target_seconds).abs().mean())
    return result


def evaluate(
    model: CustomEventEncoderModelV0,
    criterion: CustomEventCriterion,
    batches: list[dict[str, Any]],
) -> dict[str, Any]:
    model.eval()
    loss_keys = (
        "total_loss",
        "initial_ce",
        "main_event_ce",
        "main_timing_huber",
        "terminal_event_ce",
        "terminal_timing_huber",
    )
    loss_values = {key: [] for key in loss_keys}
    per_main_event_losses = [[] for _ in range(6)]
    per_main_timing_losses = [[] for _ in range(6)]
    per_terminal_event_losses = [[] for _ in range(4)]
    per_terminal_timing_losses = [[] for _ in range(4)]
    initial_targets: list[torch.Tensor] = []
    initial_predictions: list[torch.Tensor] = []
    main_targets = [[] for _ in range(6)]
    main_predictions = [[] for _ in range(6)]
    terminal_targets = [[] for _ in range(4)]
    terminal_predictions = [[] for _ in range(4)]
    main_time_targets = [[] for _ in range(6)]
    main_time_predictions = [[] for _ in range(6)]
    terminal_time_targets = [[] for _ in range(4)]
    terminal_time_predictions = [[] for _ in range(4)]
    examples = []
    with torch.no_grad():
        for batch in batches:
            output = forward_model(model, batch)
            loss = criterion(output, batch)
            for key in loss_keys:
                loss_values[key].append(float(getattr(loss, key)))
            for slot in range(6):
                per_main_event_losses[slot].append(float(loss.per_main_slot_event_loss[slot]))
                per_main_timing_losses[slot].append(float(loss.per_main_slot_timing_loss[slot]))
                valid = batch["owned_onset_mask"]
                target = batch["main_event_targets"][:, :, slot][valid]
                prediction = output.main_event_logits[:, :, slot].argmax(-1)[valid]
                main_targets[slot].append(target)
                main_predictions[slot].append(prediction)
                timing_valid = batch["main_timing_valid_mask"][:, :, slot] & valid
                main_time_targets[slot].append(batch["main_timing_targets"][:, :, slot][timing_valid])
                main_time_predictions[slot].append(output.main_timing_predictions[:, :, slot][timing_valid])
            initial_valid = batch["initial_valid_mask"]
            if bool(initial_valid.any()):
                initial_targets.append(batch["initial_targets"][initial_valid])
                initial_predictions.append(output.initial_logits.argmax(-1)[initial_valid])
            terminal_valid = batch["terminal_valid_mask"]
            for slot in range(4):
                per_terminal_event_losses[slot].append(float(loss.per_terminal_slot_event_loss[slot]))
                per_terminal_timing_losses[slot].append(float(loss.per_terminal_slot_timing_loss[slot]))
                if bool(terminal_valid.any()):
                    terminal_targets[slot].append(batch["terminal_event_targets"][:, slot][terminal_valid])
                    terminal_predictions[slot].append(output.terminal_event_logits[:, slot].argmax(-1)[terminal_valid])
                timing_valid = batch["terminal_timing_valid_mask"][:, slot] & terminal_valid
                if bool(timing_valid.any()):
                    terminal_time_targets[slot].append(batch["terminal_timing_targets"][:, slot][timing_valid])
                    terminal_time_predictions[slot].append(output.terminal_timing_predictions[:, slot][timing_valid])
            if len(examples) < 4:
                first_owned = batch["owned_onset_mask"][0]
                examples.append(
                    {
                        "performance_path": batch["metadata"][0]["performance_path"],
                        "window_index": batch["metadata"][0]["window_index"],
                        "main_target_first": batch["main_event_targets"][0][first_owned][:3].cpu().tolist(),
                        "main_prediction_first": output.main_event_logits[0][first_owned][:3].argmax(-1).cpu().tolist(),
                        "terminal_valid": bool(terminal_valid[0]),
                        "terminal_target": batch["terminal_event_targets"][0].cpu().tolist() if bool(terminal_valid[0]) else None,
                        "terminal_prediction": output.terminal_event_logits[0].argmax(-1).cpu().tolist() if bool(terminal_valid[0]) else None,
                    }
                )
    losses = {key: statistics.fmean(values) for key, values in loss_values.items()}
    main_slot_events = [event_summary(main_targets[slot], main_predictions[slot]) for slot in range(6)]
    terminal_slot_events = [event_summary(terminal_targets[slot], terminal_predictions[slot]) for slot in range(4)]
    combined_main = event_summary(
        [value for rows in main_targets for value in rows],
        [value for rows in main_predictions for value in rows],
    )
    combined_terminal = event_summary(
        [value for rows in terminal_targets for value in rows],
        [value for rows in terminal_predictions for value in rows],
    )
    initial_correct = sum(int((a == b).sum()) for a, b in zip(initial_targets, initial_predictions))
    initial_count = sum(int(value.numel()) for value in initial_targets)
    return {
        "losses": losses,
        "initial": {
            "target_count": initial_count,
            "correct_count": initial_correct,
            "accuracy": safe_div(initial_correct, initial_count),
        },
        "main_event": combined_main,
        "main_event_by_slot": {
            f"Slot{slot + 1}": {
                **main_slot_events[slot],
                "weighted_ce": statistics.fmean(per_main_event_losses[slot]),
            }
            for slot in range(6)
        },
        "terminal_event": combined_terminal,
        "terminal_event_by_slot": {
            f"Slot{slot + 1}": {
                **terminal_slot_events[slot],
                "weighted_ce": statistics.fmean(per_terminal_event_losses[slot]),
            }
            for slot in range(4)
        },
        "main_timing": timing_summary(
            [value for rows in main_time_targets for value in rows],
            [value for rows in main_time_predictions for value in rows],
        ),
        "main_timing_by_slot": {
            f"Slot{slot + 1}": {
                **timing_summary(main_time_targets[slot], main_time_predictions[slot]),
                "huber": statistics.fmean(per_main_timing_losses[slot]),
            }
            for slot in range(6)
        },
        "terminal_timing": timing_summary(
            [value for rows in terminal_time_targets for value in rows],
            [value for rows in terminal_time_predictions for value in rows],
            seconds=True,
        ),
        "terminal_timing_by_slot": {
            f"Slot{slot + 1}": {
                **timing_summary(
                    terminal_time_targets[slot],
                    terminal_time_predictions[slot],
                    seconds=True,
                ),
                "huber": statistics.fmean(per_terminal_timing_losses[slot]),
            }
            for slot in range(4)
        },
        "examples": examples,
    }


def flatten_log(step: int, elapsed: float, metrics: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {"step": step, "elapsed_seconds": elapsed}
    row.update(metrics["losses"])
    row.update(
        {
            "initial_accuracy": metrics["initial"]["accuracy"],
            "main_event_accuracy": metrics["main_event"]["accuracy"],
            "main_non_none_precision": metrics["main_event"]["non_none_precision"],
            "main_non_none_recall": metrics["main_event"]["non_none_recall"],
            "main_non_none_f1": metrics["main_event"]["non_none_f1"],
            "terminal_event_accuracy": metrics["terminal_event"]["accuracy"],
            "terminal_non_none_precision": metrics["terminal_event"]["non_none_precision"],
            "terminal_non_none_recall": metrics["terminal_event"]["non_none_recall"],
            "terminal_non_none_f1": metrics["terminal_event"]["non_none_f1"],
            "main_timing_mae": metrics["main_timing"]["mae"],
            "terminal_timing_log1p_mae": metrics["terminal_timing"]["mae"],
            "terminal_timing_seconds_mae": metrics["terminal_timing"].get("seconds_gap_mae", 0.0),
        }
    )
    for scope, count in (("main", 6), ("terminal", 4)):
        for slot in range(1, count + 1):
            event = metrics[f"{scope}_event_by_slot"][f"Slot{slot}"]
            timing = metrics[f"{scope}_timing_by_slot"][f"Slot{slot}"]
            row[f"{scope}_slot{slot}_event_accuracy"] = event["accuracy"]
            row[f"{scope}_slot{slot}_non_none_recall"] = event["non_none_recall"]
            row[f"{scope}_slot{slot}_timing_mae"] = timing["mae"]
    return row


def gradient_norm(parameters) -> tuple[float, bool, int]:
    total = 0.0
    count = 0
    finite = True
    for parameter in parameters:
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        count += 1
        finite = finite and bool(torch.isfinite(gradient).all())
        total += float(gradient.float().square().sum())
    return math.sqrt(total), finite, count


def gradient_snapshot(model: CustomEventEncoderModelV0, step: int) -> dict[str, Any]:
    groups = {
        "encoder": model.encoder.parameters(),
        "initial_head": model.initial_head.parameters(),
        "main_event_heads": model.main_event_heads.parameters(),
        "main_timing_heads": model.main_timing_heads.parameters(),
        "terminal_event_heads": model.terminal_event_heads.parameters(),
        "terminal_timing_heads": model.terminal_timing_heads.parameters(),
    }
    result: dict[str, Any] = {"step": step}
    for name, parameters in groups.items():
        norm, finite, tensor_count = gradient_norm(parameters)
        result[name] = {
            "norm": norm,
            "finite": finite,
            "gradient_tensor_count": tensor_count,
        }
    result["main_event_slots"] = {}
    result["main_timing_slots"] = {}
    for slot in range(6):
        result["main_event_slots"][f"Slot{slot + 1}"] = gradient_norm(model.main_event_heads[slot].parameters())[0]
        result["main_timing_slots"][f"Slot{slot + 1}"] = gradient_norm(model.main_timing_heads[slot].parameters())[0]
    result["terminal_event_slots"] = {}
    result["terminal_timing_slots"] = {}
    for slot in range(4):
        result["terminal_event_slots"][f"Slot{slot + 1}"] = gradient_norm(model.terminal_event_heads[slot].parameters())[0]
        result["terminal_timing_slots"][f"Slot{slot + 1}"] = gradient_norm(model.terminal_timing_heads[slot].parameters())[0]
    return result


def loss_scale_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "total_loss",
        "initial_ce",
        "main_event_ce",
        "main_timing_huber",
        "terminal_event_ce",
        "terminal_timing_huber",
    )
    early = [row for row in rows if 0 < int(row["step"]) <= 200]
    return {
        key: {
            "initial": float(rows[0][key]),
            "median_early": float(statistics.median(float(row[key]) for row in early)),
            "final": float(rows[-1][key]),
            "minimum": float(min(float(row[key]) for row in rows)),
        }
        for key in keys
    }


def atomic_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("tiny-overfit requires the available CUDA GPU")
    seed_everything(SEED)
    device = torch.device("cuda:0")
    subset, samples = load_selected_samples()
    assert_masks(samples)
    batches = [move_batch(custom_event_collate_fn([sample]), device) for sample in samples]
    loss_config = CustomEventLossConfig()
    main_weights, terminal_weights, weight_provenance = load_slot_event_weights(WEIGHTS)
    criterion = CustomEventCriterion(main_weights, terminal_weights, loss_config).to(device)
    model = CustomEventEncoderModelV0.from_pretrained(CHECKPOINT, head_init_seed=SEED).to(device)
    if model.hidden_size != 768:
        raise RuntimeError("canonical pretrained hidden size changed")
    encoder_parameters = list(model.encoder.parameters())
    head_parameters = list(model.prediction_head_parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": ENCODER_LR},
            {"params": head_parameters, "lr": HEAD_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    config = {
        "status": "running",
        "cache_id": CANONICAL_CACHE_ID,
        "selection_sha256": subset["selection_sha256"],
        "seed": SEED,
        "max_optimizer_steps": MAX_STEPS,
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
        "pretrained_checkpoint": str(CHECKPOINT),
        "pretrained_loading_path": "PianoT5Gemma.from_pretrained -> get_encoder",
        "head_initialization": "fresh deterministic seed-42",
        "warm_start": False,
        "full_training": False,
        "validation_model_inference_count": 0,
        "asap_test_access_count": 0,
        "repedal_execution_count": 0,
        "candidate_midi_generation_count": 0,
        "event_weight_provenance": weight_provenance,
        "cuda_device": torch.cuda.get_device_name(device),
    }
    atomic_json(OUTPUT_ROOT / "config.json", config)
    initial_metrics = evaluate(model, criterion, batches)
    started = time.monotonic()
    log_rows = [flatten_log(0, 0.0, initial_metrics)]
    gradient_rows = []
    selected_gradient_steps = set(range(1, len(batches) + 1))
    selected_gradient_steps |= set(range(MAX_STEPS - len(batches) + 1, MAX_STEPS + 1))
    for step in range(1, MAX_STEPS + 1):
        batch = batches[(step - 1) % len(batches)]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = forward_model(model, batch)
        loss = criterion(output, batch)
        if not bool(torch.isfinite(loss.total_loss)):
            raise FloatingPointError(f"non-finite tiny loss at step {step}")
        loss.total_loss.backward()
        snapshot = gradient_snapshot(model, step)
        if not all(
            value["finite"]
            for key, value in snapshot.items()
            if key not in {"step", "main_event_slots", "main_timing_slots", "terminal_event_slots", "terminal_timing_slots"}
            and value["gradient_tensor_count"] > 0
        ):
            raise FloatingPointError(f"non-finite tiny gradient at step {step}")
        if step in selected_gradient_steps or step % LOG_INTERVAL == 0:
            gradient_rows.append(snapshot)
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM, error_if_nonfinite=True)
        optimizer.step()
        if step % LOG_INTERVAL == 0 or step == MAX_STEPS:
            metrics = evaluate(model, criterion, batches)
            log_rows.append(flatten_log(step, time.monotonic() - started, metrics))
            atomic_csv(OUTPUT_ROOT / "training_log.csv", log_rows)
    final_metrics = evaluate(model, criterion, batches)
    elapsed = time.monotonic() - started
    loss_scale = loss_scale_diagnostics(log_rows)
    gradient_payload = {
        "snapshots": gradient_rows,
        "all_recorded_gradients_finite": True,
        "first_cycle_steps": list(range(1, len(batches) + 1)),
        "last_cycle_steps": list(range(MAX_STEPS - len(batches) + 1, MAX_STEPS + 1)),
        "note": "zero/absent branch gradients on a step are expected when that train-only window does not own that branch",
    }
    checkpoint_payload = {
        "artifact_purpose": "tiny-overfit diagnostic only; not a scientific final model",
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step": MAX_STEPS,
        "seed": SEED,
        "config": config,
        "subset_manifest": subset,
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
    }
    atomic_checkpoint(OUTPUT_ROOT / "checkpoint.pt", checkpoint_payload)
    result = {
        "status": "completed",
        "cache_id": CANONICAL_CACHE_ID,
        "selection_sha256": subset["selection_sha256"],
        "subset_summary": subset["summary"],
        "optimizer_steps": MAX_STEPS,
        "training_steps": MAX_STEPS,
        "precision": "FP32",
        "elapsed_seconds": elapsed,
        "initial": initial_metrics,
        "final": final_metrics,
        "loss_scale_diagnostics": loss_scale,
        "gradient_diagnostics_path": str(OUTPUT_ROOT / "gradient_diagnostics.json"),
        "none_timing_excluded": True,
        "all_forward_loss_gradient_values_finite": True,
        "full_training_started": False,
        "validation_model_inference_count": 0,
        "asap_test_access_count": 0,
        "repedal_execution_count": 0,
        "candidate_midi_generation_count": 0,
        "checkpoint_purpose": "diagnostic only",
    }
    atomic_json(OUTPUT_ROOT / "final_metrics.json", result)
    atomic_json(OUTPUT_ROOT / "loss_scale_diagnostics.json", loss_scale)
    atomic_json(OUTPUT_ROOT / "gradient_diagnostics.json", gradient_payload)
    config["status"] = "completed"
    config["optimizer_steps"] = MAX_STEPS
    config["elapsed_seconds"] = elapsed
    atomic_json(OUTPUT_ROOT / "config.json", config)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
