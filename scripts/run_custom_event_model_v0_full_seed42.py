#!/usr/bin/env python3
"""Canonical seed-42 full training for frozen Custom Event Model v0."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import itertools
import json
import math
import os
import random
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage2_event_model.dataset import (
    CANONICAL_ALIGNMENT_ID,
    CANONICAL_CACHE_ID,
    CustomEventWindowDataset,
    custom_event_collate_fn,
    load_cache_manifest,
)
from src.stage2_event_model.full_training import (
    LossDenominators,
    ValidationDiagnostics,
    add_raw_terms,
    batch_denominators,
    capture_rng_state,
    components_from_sums,
    empty_raw_sums,
    group_denominators,
    normalized_components,
    raw_loss_terms,
    restore_rng_state,
)
from src.stage2_event_model.losses import (
    CustomEventCriterion,
    CustomEventLossConfig,
    load_slot_event_weights,
)
from src.stage2_event_model.model import CustomEventEncoderModelV0
from src.stage2_event_model.status_logging import write_failure_status, write_run_status

CACHE_ROOT = ROOT / "analysis/custom_event_tokenizer_v1"
MODEL_REPORT_ROOT = ROOT / "analysis/custom_event_model_v0"
TINY_ROOT = ROOT / "analysis/custom_event_model_v0_tiny_overfit"
OUTPUT_ROOT = ROOT / "analysis/custom_event_model_v0_full_seed42_aligned_statusfix_v1"
ALIGNMENT_ROOT = ROOT / "analysis/custom_event_model_v0_note_alignment"
PRETRAINED = ROOT / "checkpoints/pianist_transformer"
WEIGHTS = CACHE_ROOT / "candidate_event_weights.json"
OWNERSHIP = MODEL_REPORT_ROOT / "window_ownership_stats.json"
TINY_SUBSET = TINY_ROOT / "tiny_subset_manifest.json"
EXPECTED_WEIGHT_SHA = "26da9ede3373bd3df5c3f28c259e29f173a0920e080eb3a27cdbc5c790ed5b55"
EXPECTED_ALIGNMENT_CONFIG_SHA = "77df6faf08025794e40c58d0db31cf721f582b3168d9fb201557c0500807502b"
SEED = 42
MAX_EPOCHS = 10
PATIENCE = 3
MIN_DELTA = 1e-4
MAX_CONSECUTIVE_AMP_OVERFLOWS = 3


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
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
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_torch(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def log(message: str) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    line = f"{now()} {message}"
    with (OUTPUT_ROOT / "training.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
    print(line, flush=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init(worker_id: int) -> None:
    worker_seed = (SEED + worker_id) % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def configuration(micro_batch: int) -> dict[str, Any]:
    accumulation = 16 // micro_batch
    config = {
        "experiment": "custom_event_model_v0_full_seed42_aligned_statusfix_v1",
        "cache_id": CANONICAL_CACHE_ID,
        "note_alignment_id": CANONICAL_ALIGNMENT_ID,
        "note_alignment_config_sha256": EXPECTED_ALIGNMENT_CONFIG_SHA,
        "event_weight_sha256": EXPECTED_WEIGHT_SHA,
        "seed": SEED,
        "pretrained_checkpoint": str(PRETRAINED),
        "pretrained_loading": "PianoT5Gemma.from_pretrained -> get_encoder",
        "head_initialization": "fresh seed-42",
        "warm_start": False,
        "architecture": {
            "hidden_size": 768,
            "encoder_parameters": 103271424,
            "total_parameters": 103320640,
            "initial": "Linear(768,4)",
            "main_event": "6 independent Linear(768,5)",
            "main_timing": "6 independent Linear(768,1)",
            "terminal_event": "4 independent Linear(768,5)",
            "terminal_timing": "4 independent Linear(768,1)",
        },
        "window_notes": 512,
        "stride_notes": 256,
        "ownership_rule": "full chord containment; maximum representative margin; tie smaller window start",
        "train_performances": 2062,
        "validation_performances": 71,
        "train_windows": 35573,
        "validation_windows": 1078,
        "micro_batch_size": micro_batch,
        "gradient_accumulation_steps": accumulation,
        "effective_batch_size": micro_batch * accumulation,
        "optimizer": "AdamW",
        "encoder_lr": 1e-5,
        "head_lr": 1e-4,
        "weight_decay": 0.01,
        "max_gradient_norm": 1.0,
        "precision": "FP16 AMP",
        "amp_init_scale": 1024.0,
        "max_consecutive_amp_overflows": MAX_CONSECUTIVE_AMP_OVERFLOWS,
        "scheduler": None,
        "max_epochs": MAX_EPOCHS,
        "validation_cadence": "every epoch",
        "early_stopping_patience": PATIENCE,
        "early_stopping_min_delta": MIN_DELTA,
        "schedule_provenance": "canonical 4-class Encoder-only Stage2: max 10 epochs, validation every epoch, patience 3, min_delta 1e-4, no scheduler",
        "checkpoint_selection": "minimum validation total objective",
        "loss": asdict(CustomEventLossConfig()),
        "num_workers": 0,
        "pin_memory": True,
        "input_token_cache": "single-process in-memory per-performance official PT tokens",
        "asap_test_access_count": 0,
        "repedal_execution_count": 0,
        "validation_candidate_inference_count": 0,
        "frozen_evaluator_execution_count": 0,
    }
    fingerprint_source = {key: value for key, value in config.items() if key not in {"config_fingerprint"}}
    config["config_fingerprint"] = hashlib.sha256(
        json.dumps(fingerprint_source, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return config


def verify_frozen_artifacts() -> dict[str, Any]:
    _, cache_config, manifest = load_cache_manifest(CACHE_ROOT)
    if manifest["train_count"] != 2062 or manifest["validation_count"] != 71:
        raise RuntimeError("canonical tokenizer cache counts changed")
    if cache_config["asap_test_access_count"] != 0:
        raise RuntimeError("cache reports ASAP test access")
    alignment_manifest = json.loads((ALIGNMENT_ROOT / "alignment_manifest.json").read_text(encoding="utf-8"))
    alignment_config = json.loads((ALIGNMENT_ROOT / "alignment_config.json").read_text(encoding="utf-8"))
    alignment_stats = json.loads((ALIGNMENT_ROOT / "alignment_stats.json").read_text(encoding="utf-8"))
    if alignment_manifest["alignment_id"] != CANONICAL_ALIGNMENT_ID or alignment_stats["alignment_id"] != CANONICAL_ALIGNMENT_ID:
        raise RuntimeError("canonical note-alignment ID mismatch")
    if alignment_manifest["alignment_config_sha256"] != EXPECTED_ALIGNMENT_CONFIG_SHA or alignment_config["alignment_config_sha256"] != EXPECTED_ALIGNMENT_CONFIG_SHA:
        raise RuntimeError("canonical alignment config SHA mismatch")
    if alignment_manifest["tokenizer_cache_id"] != CANONICAL_CACHE_ID or alignment_config["tokenizer_cache_id"] != CANONICAL_CACHE_ID:
        raise RuntimeError("alignment/tokenizer cache identity mismatch")
    if alignment_manifest["asap_test_access_count"] != 0 or alignment_stats["asap_test_access_count"] != 0:
        raise RuntimeError("alignment reports ASAP test access")
    ownership = json.loads(OWNERSHIP.read_text(encoding="utf-8"))
    expected = {
        "train": (35573, 9009549, 2062, 2062),
        "validation": (1078, 272927, 71, 71),
    }
    for split, values in expected.items():
        row = ownership[split]
        actual = (
            int(row["total_windows"]),
            int(row["owned_onset_groups"]),
            int(row["initial_supervision_count"]),
            int(row["terminal_supervision_count"]),
        )
        if actual != values or row["zero_owner_count"] != 0 or row["duplicate_owner_count"] != 0:
            raise RuntimeError(f"frozen ownership mismatch for {split}: {actual}")
    tiny = json.loads((TINY_ROOT / "final_metrics.json").read_text(encoding="utf-8"))
    if tiny["status"] != "completed" or tiny["final"]["main_event"]["accuracy"] != 1.0:
        raise RuntimeError("READY tiny-overfit artifact unavailable")
    main_weights, terminal_weights, provenance = load_slot_event_weights(WEIGHTS)
    if provenance["sha256"] != EXPECTED_WEIGHT_SHA:
        raise RuntimeError("frozen event-weight SHA mismatch")
    return {
        "cache_id": CANONICAL_CACHE_ID,
        "note_alignment_id": CANONICAL_ALIGNMENT_ID,
        "note_alignment_config_sha256": EXPECTED_ALIGNMENT_CONFIG_SHA,
        "train_count": manifest["train_count"],
        "validation_count": manifest["validation_count"],
        "ownership": ownership,
        "tiny_overfit_status": "READY FOR FULL TRAINING",
        "event_weight_provenance": provenance,
        "main_weights": main_weights,
        "terminal_weights": terminal_weights,
        "asap_test_access_count": 0,
    }


def build_model(device: torch.device) -> CustomEventEncoderModelV0:
    model = CustomEventEncoderModelV0.from_pretrained(
        PRETRAINED,
        head_init_seed=SEED,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    ).to(device)
    if model.hidden_size != 768 or model.encoder_parameter_count != 103271424 or model.parameter_count != 103320640:
        raise RuntimeError("frozen Custom Event Model architecture changed")
    return model


def build_optimizer(model: CustomEventEncoderModelV0) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        [
            {"params": list(model.encoder.parameters()), "lr": 1e-5, "group_name": "encoder"},
            {"params": list(model.prediction_head_parameters()), "lr": 1e-4, "group_name": "custom_heads"},
        ],
        weight_decay=0.01,
    )


def move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def forward_model(model: CustomEventEncoderModelV0, batch: Mapping[str, Any]):
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


def selected_preflight_samples() -> list[dict[str, Any]]:
    payload = json.loads(TINY_SUBSET.read_text(encoding="utf-8"))
    selected = payload["selected"]
    last_entry = max(int(row["manifest_entry_index"]) for row in selected)
    dataset = CustomEventWindowDataset(CACHE_ROOT, "train", entry_limit=last_entry + 1, cache_input_tokens=True)
    lookup = {pair: index for index, pair in enumerate(dataset.windows)}
    return [dataset[lookup[(int(row["manifest_entry_index"]), int(row["window_index"]))]] for row in selected]


def preflight() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("canonical preflight requires CUDA")
    frozen = verify_frozen_artifacts()
    samples = selected_preflight_samples()
    device = torch.device("cuda:0")
    selected_micro = None
    attempt_details = []
    for micro in (4, 2, 1):
        seed_everything(SEED)
        model = None
        optimizer = None
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            model = build_model(device)
            optimizer = build_optimizer(model)
            scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=1024.0)
            main_weights = frozen["main_weights"]
            terminal_weights = frozen["terminal_weights"]
            criterion = CustomEventCriterion(main_weights, terminal_weights, CustomEventLossConfig()).to(device)
            component_seen = {key: False for key in ("initial", "main_event", "main_timing", "terminal_event", "terminal_timing")}
            all_gradients_finite = True
            for offset in range(0, len(samples), micro):
                batch = move_batch(custom_event_collate_fn(samples[offset : offset + micro]), device)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=True):
                    output = forward_model(model, batch)
                    loss = criterion(output, batch)
                components = {
                    "initial": loss.initial_ce,
                    "main_event": loss.main_event_ce,
                    "main_timing": loss.main_timing_huber,
                    "terminal_event": loss.terminal_event_ce,
                    "terminal_timing": loss.terminal_timing_huber,
                }
                if not all(bool(torch.isfinite(value)) for value in (loss.total_loss, *components.values())):
                    raise FloatingPointError("preflight objective is non-finite")
                for key, value in components.items():
                    component_seen[key] |= bool(float(value.detach().float()) != 0.0)
                scaler.scale(loss.total_loss).backward()
                scaler.unscale_(optimizer)
                gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
                norm = torch.nn.utils.get_total_norm(gradients)
                all_gradients_finite &= math.isfinite(float(norm.detach().float()))
            if not all_gradients_finite or not all(component_seen.values()):
                raise RuntimeError(f"preflight branch/gradient failure: {component_seen}")
            selected_micro = micro
            attempt_details.append({
                "micro_batch": micro,
                "status": "passed",
                "all_five_components_exercised": True,
                "all_gradients_finite": True,
                "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
                "optimizer_groups": [
                    {"name": group["group_name"], "lr": group["lr"]}
                    for group in optimizer.param_groups
                ],
            })
            break
        except torch.cuda.OutOfMemoryError as error:
            attempt_details.append({"micro_batch": micro, "status": "oom", "error": str(error)})
        finally:
            del optimizer
            del model
            gc.collect()
            torch.cuda.empty_cache()
    if selected_micro is None:
        raise RuntimeError("micro batch 1 did not fit canonical GPU")
    config = configuration(selected_micro)
    payload = {
        "status": "passed",
        "completed_at": now(),
        "cache_id": CANONICAL_CACHE_ID,
        "note_alignment_id": CANONICAL_ALIGNMENT_ID,
        "note_alignment_config_sha256": EXPECTED_ALIGNMENT_CONFIG_SHA,
        "event_weight_sha256": EXPECTED_WEIGHT_SHA,
        "tiny_overfit_status": frozen["tiny_overfit_status"],
        "train_windows": frozen["ownership"]["train"]["total_windows"],
        "validation_windows": frozen["ownership"]["validation"]["total_windows"],
        "train_owned_onsets": frozen["ownership"]["train"]["owned_onset_groups"],
        "validation_owned_onsets": frozen["ownership"]["validation"]["owned_onset_groups"],
        "zero_owner_count": 0,
        "duplicate_owner_count": 0,
        "model_parameters": 103320640,
        "selected_micro_batch": selected_micro,
        "gradient_accumulation_steps": 16 // selected_micro,
        "effective_batch_size": 16,
        "attempts": attempt_details,
        "canonical_training_uses_fresh_model_after_preflight": True,
        "configuration": config,
        "asap_test_access_count": 0,
    }
    atomic_json(OUTPUT_ROOT / "preflight.json", payload)
    atomic_json(OUTPUT_ROOT / "config.json", config)
    print(json.dumps(payload, sort_keys=True))
    return payload


def make_loader(dataset: CustomEventWindowDataset, config: Mapping[str, Any], epoch: int, training: bool) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(SEED + int(epoch) if training else SEED)
    return DataLoader(
        dataset,
        batch_size=int(config["micro_batch_size"]),
        shuffle=training,
        generator=generator,
        num_workers=0,
        worker_init_fn=worker_init,
        pin_memory=True,
        drop_last=False,
        collate_fn=custom_event_collate_fn,
    )


def nonfinite_parameters(model: torch.nn.Module) -> list[str]:
    return [name for name, parameter in model.named_parameters() if not bool(torch.isfinite(parameter).all())]


def gradient_overflow_details(model: CustomEventEncoderModelV0) -> dict[str, Any]:
    groups = {id(parameter): "encoder" for parameter in model.encoder.parameters()}
    for parameter in model.prediction_head_parameters():
        groups[id(parameter)] = "custom_heads"
    names = []
    affected = set()
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
            names.append(name)
            affected.add(groups.get(id(parameter), "unknown"))
    return {"offending_parameter_names": names, "offending_parameter_count": len(names), "affected_groups": sorted(affected)}


def merge_denominators(destination: LossDenominators, source: LossDenominators) -> None:
    destination.add_(source)


def run_epoch(
    model: CustomEventEncoderModelV0,
    loader: DataLoader,
    criterion: CustomEventCriterion,
    *,
    device: torch.device,
    accumulation: int,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    epoch: int,
    global_state: dict[str, int],
    status: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    training = optimizer is not None
    model.train(training)
    if training and scaler is None:
        raise ValueError("training requires GradScaler")
    if training:
        corrupt = nonfinite_parameters(model)
        if corrupt:
            raise FloatingPointError("non-finite model parameters: " + ", ".join(corrupt[:8]))
    epoch_sums = empty_raw_sums()
    epoch_denominators = LossDenominators.zero()
    diagnostics = None if training else ValidationDiagnostics()
    amp_events = 0
    max_consecutive = global_state["consecutive_amp_overflows"]
    first_step: dict[str, Any] | None = None
    iterator = iter(loader)
    group_index = 0
    while True:
        cpu_group = list(itertools.islice(iterator, accumulation if training else 1))
        if not cpu_group:
            break
        group_index += 1
        group_denom = group_denominators(cpu_group, criterion.main_event_weights.cpu(), criterion.terminal_event_weights.cpu())
        group_sums = empty_raw_sums()
        if training:
            optimizer.zero_grad(set_to_none=True)
        for cpu_batch in cpu_group:
            batch = move_batch(cpu_batch, device)
            with torch.set_grad_enabled(training), torch.amp.autocast("cuda", dtype=torch.float16, enabled=True):
                output = forward_model(model, batch)
                terms = raw_loss_terms(output, batch, criterion)
                components = normalized_components(terms, group_denom, criterion.config)
            tensors = (
                output.initial_logits,
                output.main_event_logits,
                output.main_timing_predictions,
                output.terminal_event_logits,
                output.terminal_timing_predictions,
                *components.values(),
            )
            if not all(bool(torch.isfinite(value).all()) for value in tensors):
                raise FloatingPointError("non-finite forward/objective")
            if training:
                scaler.scale(components["total"]).backward()
            add_raw_terms(group_sums, terms)
            add_raw_terms(epoch_sums, terms)
            batch_denom = batch_denominators(cpu_batch, criterion.main_event_weights.cpu(), criterion.terminal_event_weights.cpu())
            merge_denominators(epoch_denominators, batch_denom)
            if diagnostics is not None:
                diagnostics.update(output, batch)
        group_components = components_from_sums(group_sums, group_denom, criterion.config)
        if training:
            global_state["optimizer_attempt"] += 1
            scaler.unscale_(optimizer)
            parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
            gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
            grad_norm = torch.nn.utils.get_total_norm(gradients)
            grad_norm_value = float(grad_norm.detach().float())
            finite_gradient = math.isfinite(grad_norm_value)
            scale_before = float(scaler.get_scale())
            overflow_details = {"offending_parameter_count": 0, "offending_parameter_names": [], "affected_groups": []}
            if finite_gradient:
                torch.nn.utils.clip_grads_with_norm_(parameters, max_norm=1.0, total_norm=grad_norm)
            else:
                overflow_details = gradient_overflow_details(model)
                if not overflow_details["offending_parameter_count"]:
                    raise FloatingPointError("non-finite grad norm without offending tensor")
            scaler.step(optimizer)
            scaler.update()
            scale_after = float(scaler.get_scale())
            applied = finite_gradient and scale_after >= scale_before
            overflow = not finite_gradient
            if applied:
                global_state["optimizer_step"] += 1
                global_state["consecutive_amp_overflows"] = 0
            elif overflow:
                amp_events += 1
                global_state["amp_overflow_count"] += 1
                global_state["consecutive_amp_overflows"] += 1
                global_state["max_consecutive_amp_overflows"] = max(
                    global_state["max_consecutive_amp_overflows"],
                    global_state["consecutive_amp_overflows"],
                )
                max_consecutive = max(max_consecutive, global_state["consecutive_amp_overflows"])
                last_overflow = {
                    "epoch": epoch,
                    "epoch_optimizer_attempt": group_index,
                    "global_optimizer_attempt": global_state["optimizer_attempt"],
                    "gradient_norm": grad_norm_value,
                    "scale_before": scale_before,
                    "scale_after": scale_after,
                    **overflow_details,
                }
                status["last_overflow"] = last_overflow
                status.update(
                    epoch=epoch,
                    global_optimizer_attempt=global_state["optimizer_attempt"],
                    global_optimizer_step=global_state["optimizer_step"],
                    latest_train_total=float(group_components["total"]),
                    amp_overflow_count=global_state["amp_overflow_count"],
                    max_consecutive_amp_overflow=global_state["max_consecutive_amp_overflows"],
                    updated_at=now(),
                )
                write_run_status(OUTPUT_ROOT / "run_status.json", status)
                log(f"AMP_OVERFLOW epoch={epoch} attempt={group_index} scale={scale_before}->{scale_after} affected={overflow_details['affected_groups']}")
                corrupt = nonfinite_parameters(model)
                if corrupt:
                    raise FloatingPointError("parameter corruption after overflow: " + ", ".join(corrupt[:8]))
                if global_state["consecutive_amp_overflows"] >= MAX_CONSECUTIVE_AMP_OVERFLOWS:
                    raise FloatingPointError("3 consecutive AMP gradient overflows")
            if first_step is None and applied:
                first_step = {
                    "epoch": epoch,
                    "global_optimizer_attempt": global_state["optimizer_attempt"],
                    "global_optimizer_step": global_state["optimizer_step"],
                    "gradient_norm": grad_norm_value,
                    "grad_scaler_scale": scale_after,
                    **{key: float(group_components[key]) for key in ("total", "initial", "main_event", "main_timing", "terminal_event", "terminal_timing")},
                }
                status.update(
                    status="running",
                    stage="training",
                    epoch=epoch,
                    global_optimizer_attempt=global_state["optimizer_attempt"],
                    global_optimizer_step=global_state["optimizer_step"],
                    latest_train_total=float(group_components["total"]),
                    first_finite_optimizer_step=first_step,
                    updated_at=now(),
                )
                write_run_status(OUTPUT_ROOT / "run_status.json", status)
                log(
                    "FIRST_OPTIMIZER_STEP_FINITE "
                    + " ".join(f"{key}={first_step[key]:.9f}" for key in ("total", "initial", "main_event", "main_timing", "terminal_event", "terminal_timing"))
                    + f" grad_norm={grad_norm_value:.6f} scaler={scale_after:.1f}"
                )
            if group_index % 100 == 0:
                status.update(
                    epoch=epoch,
                    global_optimizer_attempt=global_state["optimizer_attempt"],
                    global_optimizer_step=global_state["optimizer_step"],
                    latest_train_total=float(group_components["total"]),
                    amp_overflow_count=global_state["amp_overflow_count"],
                    max_consecutive_amp_overflow=global_state["max_consecutive_amp_overflows"],
                    updated_at=now(),
                )
                write_run_status(OUTPUT_ROOT / "run_status.json", status)
                log(f"TRAIN_PROGRESS epoch={epoch} attempt={group_index} global_step={global_state['optimizer_step']} total={group_components['total']:.9f}")
    metrics = components_from_sums(epoch_sums, epoch_denominators, criterion.config)
    checks = {
        "optimizer_attempts": group_index if training else 0,
        "amp_overflow_events": amp_events,
        "max_consecutive_amp_overflows": max_consecutive,
        "grad_scaler_scale": float(scaler.get_scale()) if training else None,
    }
    return metrics, checks, diagnostics.result() if diagnostics is not None else None


def flatten_epoch(epoch: int, losses: Mapping[str, Any], seconds: float, extra: Mapping[str, Any]) -> dict[str, Any]:
    row = {
        "epoch": epoch,
        "total": losses["total"],
        "initial_ce": losses["initial"],
        "main_event_ce": losses["main_event"],
        "main_timing_huber": losses["main_timing"],
        "terminal_event_ce": losses["terminal_event"],
        "terminal_timing_huber": losses["terminal_timing"],
        "epoch_seconds": seconds,
    }
    for key in ("main_event_by_slot", "main_timing_by_slot", "terminal_event_by_slot", "terminal_timing_by_slot"):
        for index, value in enumerate(losses[key], 1):
            row[f"{key}_slot{index}"] = value
    row.update(extra)
    return row


def checkpoint_payload(
    model, optimizer, scaler, config, epoch, global_state, best_epoch, best_val, early_counter
) -> dict[str, Any]:
    if best_epoch is None or best_val is None or not math.isfinite(float(best_val)):
        raise ValueError("checkpoint requires a finite validated best objective")
    if not math.isfinite(float(scaler.get_scale())):
        raise FloatingPointError("checkpoint GradScaler scale is non-finite")
    return {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "grad_scaler_state": scaler.state_dict(),
        "epoch": epoch,
        "global_optimizer_attempt": global_state["optimizer_attempt"],
        "global_optimizer_step": global_state["optimizer_step"],
        "best_epoch": best_epoch,
        "best_validation_total": best_val,
        "early_stopping_counter": early_counter,
        "amp_overflow_count": global_state["amp_overflow_count"],
        "max_consecutive_amp_overflows": global_state["max_consecutive_amp_overflows"],
        "rng_state": capture_rng_state(),
        "seed": SEED,
        "tokenizer_cache_id": CANONICAL_CACHE_ID,
        "note_alignment_id": CANONICAL_ALIGNMENT_ID,
        "note_alignment_config_sha256": EXPECTED_ALIGNMENT_CONFIG_SHA,
        "event_weight_sha256": EXPECTED_WEIGHT_SHA,
        "configuration": dict(config),
        "config_fingerprint": config["config_fingerprint"],
    }


def report(config, train_rows, validation_rows, validation_slots, status) -> None:
    best = min(validation_rows, key=lambda row: float(row["total"]))
    best_diag = next(row["diagnostics"] for row in validation_slots if row["epoch"] == best["epoch"])
    main5 = best_diag["main_event_by_slot"][4]
    main6 = best_diag["main_event_by_slot"][5]
    terminal4 = best_diag["terminal_event_by_slot"][3]
    component_names = ("initial_ce", "main_event_ce", "main_timing_huber", "terminal_event_ce", "terminal_timing_huber")
    lines = [
        "# Custom Event Model v0 Canonical Training Report",
        "",
        f"- Status: `{status['status']}`",
        f"- Cache ID: `{CANONICAL_CACHE_ID}`",
        f"- Note-alignment ID: `{CANONICAL_ALIGNMENT_ID}`",
        f"- Alignment config SHA: `{EXPECTED_ALIGNMENT_CONFIG_SHA}`",
        f"- Seed: {SEED}",
        f"- Best epoch: {best['epoch']}",
        f"- Best validation total: {float(best['total']):.9f}",
        f"- Micro/accumulation/effective batch: {config['micro_batch_size']}/{config['gradient_accumulation_steps']}/16",
        f"- AMP overflow events: {status['amp_overflow_count']}",
        "- ASAP test access: 0",
        "- Candidate MIDI/frozen evaluator/Repedal: not executed",
        "",
        "## Epoch objectives",
        "",
        "| Epoch | Train total | Validation total | Init | Main event | Main time | Terminal event | Terminal time |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for train, validation in zip(train_rows, validation_rows):
        lines.append(
            f"| {train['epoch']} | {float(train['total']):.6f} | {float(validation['total']):.6f} | "
            + " | ".join(f"{float(validation[key]):.6f}" for key in component_names)
            + " |"
        )
    lines += [
        "",
        "## Best-epoch diagnostics",
        "",
        f"- Main event accuracy/non-NONE F1: {best_diag['main_event']['accuracy']:.6f} / {best_diag['main_event']['non_none_f1']:.6f}",
        f"- Main Slot5 predicted/target non-NONE: {main5['predicted_non_none_count']}/{main5['target_non_none_count']} (F1 {main5['non_none_f1']:.6f})",
        f"- Main Slot6 predicted/target non-NONE: {main6['predicted_non_none_count']}/{main6['target_non_none_count']} (F1 {main6['non_none_f1']:.6f})",
        f"- Terminal event accuracy/non-NONE F1: {best_diag['terminal_event']['accuracy']:.6f} / {best_diag['terminal_event']['non_none_f1']:.6f}",
        f"- Terminal Slot4 predicted/target non-NONE: {terminal4['predicted_non_none_count']}/{terminal4['target_non_none_count']} (F1 {terminal4['non_none_f1']:.6f})",
        f"- Main raw tau MAE/RMSE: {best_diag['main_timing']['mae']:.6f} / {best_diag['main_timing']['rmse']:.6f}",
        f"- Terminal log-gap MAE/RMSE: {best_diag['terminal_timing']['mae']:.6f} / {best_diag['terminal_timing']['rmse']:.6f}",
        f"- Terminal decoded seconds-gap MAE: {best_diag['terminal_timing']['decoded_seconds_gap_mae']:.6f}",
        "",
        "## Interpretation",
        "",
        "Best checkpoint selection used only minimum ASAP-validation target total objective. No lambda, beta, weight, architecture, or tokenizer changes were made. Component histories and sparse-slot non-NONE ratios above determine whether any branch collapsed; no automatic corrective experiment was launched.",
        "",
        "Readiness for canonical validation inference must be judged from the completed loss/diagnostic table. This runner does not generate MIDI or execute the frozen evaluator.",
        "",
        "Recommendation: **READY FOR CANONICAL VALIDATION INFERENCE** if the run status is completed and the best checkpoint loads successfully; otherwise **NEEDS TARGETED TRAINING FIX**. No inference was started automatically.",
    ]
    (OUTPUT_ROOT / "CUSTOM_EVENT_MODEL_V0_TRAINING_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def train(resume_path: str | None) -> None:
    preflight_path = OUTPUT_ROOT / "preflight.json"
    if not preflight_path.exists():
        raise RuntimeError("foreground preflight artifact missing")
    preflight_payload = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight_payload["status"] != "passed":
        raise RuntimeError("canonical preflight did not pass")
    config = configuration(int(preflight_payload["selected_micro_batch"]))
    stored_config = json.loads((OUTPUT_ROOT / "config.json").read_text(encoding="utf-8"))
    if stored_config["config_fingerprint"] != config["config_fingerprint"]:
        raise RuntimeError("preflight/training config fingerprint mismatch")
    frozen = verify_frozen_artifacts()
    seed_everything(SEED)
    device = torch.device("cuda:0")
    status = {
        "status": "initializing",
        "tokenizer_cache_id": CANONICAL_CACHE_ID,
        "note_alignment_id": CANONICAL_ALIGNMENT_ID,
        "note_alignment_config_sha256": EXPECTED_ALIGNMENT_CONFIG_SHA,
        "event_weight_sha256": EXPECTED_WEIGHT_SHA,
        "seed": SEED,
        "stage": "dataset_load",
        "epoch": 0,
        "global_optimizer_attempt": 0,
        "global_optimizer_step": 0,
        "best_epoch": None,
        "best_val_total": None,
        "latest_train_total": None,
        "latest_val_total": None,
        "latest_validation_components": None,
        "amp_overflow_count": 0,
        "max_consecutive_amp_overflow": 0,
        "failure_reason": None,
        "asap_test_access_count": 0,
        "repedal_execution_count": 0,
        "candidate_inference_count": 0,
        "updated_at": now(),
    }
    write_run_status(OUTPUT_ROOT / "run_status.json", status)
    log("CANONICAL_RUN_START fresh_seed42=true warm_start=false cache_id=" + CANONICAL_CACHE_ID + " alignment_id=" + CANONICAL_ALIGNMENT_ID)
    try:
        train_dataset = CustomEventWindowDataset(CACHE_ROOT, "train", cache_input_tokens=True)
        validation_dataset = CustomEventWindowDataset(CACHE_ROOT, "validation", cache_input_tokens=True)
        if len(train_dataset) != 35573 or len(validation_dataset) != 1078:
            raise RuntimeError("canonical window count mismatch")
        model = build_model(device)
        optimizer = build_optimizer(model)
        scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=1024.0)
        criterion = CustomEventCriterion(frozen["main_weights"], frozen["terminal_weights"], CustomEventLossConfig()).to(device)
        start_epoch = 1
        best_epoch = None
        best_val: float | None = None
        early_best: float | None = None
        early_counter = 0
        global_state = {"optimizer_attempt": 0, "optimizer_step": 0, "amp_overflow_count": 0, "consecutive_amp_overflows": 0, "max_consecutive_amp_overflows": 0}
        train_rows: list[dict[str, Any]] = []
        validation_rows: list[dict[str, Any]] = []
        validation_slots: list[dict[str, Any]] = []
        if resume_path:
            checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
            if checkpoint["config_fingerprint"] != config["config_fingerprint"] or checkpoint["tokenizer_cache_id"] != CANONICAL_CACHE_ID or checkpoint["note_alignment_id"] != CANONICAL_ALIGNMENT_ID or checkpoint["note_alignment_config_sha256"] != EXPECTED_ALIGNMENT_CONFIG_SHA or checkpoint["event_weight_sha256"] != EXPECTED_WEIGHT_SHA:
                raise RuntimeError("resume checkpoint identity mismatch")
            model.load_state_dict(checkpoint["model_state"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scaler.load_state_dict(checkpoint["grad_scaler_state"])
            restore_rng_state(checkpoint["rng_state"])
            start_epoch = int(checkpoint["epoch"]) + 1
            best_epoch = int(checkpoint["best_epoch"])
            best_val = float(checkpoint["best_validation_total"])
            early_best = best_val
            early_counter = int(checkpoint["early_stopping_counter"])
            global_state.update(
                optimizer_attempt=int(checkpoint["global_optimizer_attempt"]),
                optimizer_step=int(checkpoint["global_optimizer_step"]),
                amp_overflow_count=int(checkpoint["amp_overflow_count"]),
                max_consecutive_amp_overflows=int(checkpoint["max_consecutive_amp_overflows"]),
            )
            for path, destination in ((OUTPUT_ROOT / "train_epoch_metrics.csv", train_rows), (OUTPUT_ROOT / "validation_epoch_metrics.csv", validation_rows)):
                if path.exists():
                    with path.open(encoding="utf-8", newline="") as handle:
                        destination.extend(csv.DictReader(handle))
            if (OUTPUT_ROOT / "validation_slot_metrics.json").exists():
                validation_slots = json.loads((OUTPUT_ROOT / "validation_slot_metrics.json").read_text(encoding="utf-8"))["epochs"]
            log(f"RESUME epoch={start_epoch} checkpoint={resume_path}")
        else:
            for artifact in (OUTPUT_ROOT / "best.pt", OUTPUT_ROOT / "last.pt"):
                if artifact.exists():
                    raise RuntimeError(f"fresh run refuses existing checkpoint: {artifact}")
        status.update(status="running", stage="training", updated_at=now())
        write_run_status(OUTPUT_ROOT / "run_status.json", status)
        stop_reason = "max_epochs"
        for epoch in range(start_epoch, MAX_EPOCHS + 1):
            epoch_started = time.monotonic()
            train_loader = make_loader(train_dataset, config, epoch, True)
            train_losses, train_checks, first_step = run_epoch(
                model, train_loader, criterion, device=device,
                accumulation=int(config["gradient_accumulation_steps"]),
                optimizer=optimizer, scaler=scaler, epoch=epoch,
                global_state=global_state, status=status,
            )
            train_seconds = time.monotonic() - epoch_started
            train_row = flatten_epoch(epoch, train_losses, train_seconds, {
                "global_optimizer_attempt": global_state["optimizer_attempt"],
                "global_optimizer_step": global_state["optimizer_step"],
                "amp_overflow_events": train_checks["amp_overflow_events"],
                "grad_scaler_scale": train_checks["grad_scaler_scale"],
                "encoder_lr": optimizer.param_groups[0]["lr"],
                "head_lr": optimizer.param_groups[1]["lr"],
            })
            train_rows.append(train_row)
            atomic_csv(OUTPUT_ROOT / "train_epoch_metrics.csv", train_rows)
            status.update(stage="validation", epoch=epoch, latest_train_total=train_losses["total"], updated_at=now())
            write_run_status(OUTPUT_ROOT / "run_status.json", status)
            validation_started = time.monotonic()
            validation_loader = make_loader(validation_dataset, config, epoch, False)
            validation_losses, _, diagnostics = run_epoch(
                model, validation_loader, criterion, device=device,
                accumulation=1, optimizer=None, scaler=None, epoch=epoch,
                global_state=global_state, status=status,
            )
            validation_seconds = time.monotonic() - validation_started
            assert diagnostics is not None
            validation_row = flatten_epoch(epoch, validation_losses, validation_seconds, {
                "main_event_accuracy": diagnostics["main_event"]["accuracy"],
                "main_non_none_precision": diagnostics["main_event"]["non_none_precision"],
                "main_non_none_recall": diagnostics["main_event"]["non_none_recall"],
                "main_non_none_f1": diagnostics["main_event"]["non_none_f1"],
                "main_predicted_non_none": diagnostics["main_event"]["predicted_non_none_count"],
                "main_target_non_none": diagnostics["main_event"]["target_non_none_count"],
                "terminal_event_accuracy": diagnostics["terminal_event"]["accuracy"],
                "terminal_non_none_precision": diagnostics["terminal_event"]["non_none_precision"],
                "terminal_non_none_recall": diagnostics["terminal_event"]["non_none_recall"],
                "terminal_non_none_f1": diagnostics["terminal_event"]["non_none_f1"],
                "terminal_predicted_non_none": diagnostics["terminal_event"]["predicted_non_none_count"],
                "terminal_target_non_none": diagnostics["terminal_event"]["target_non_none_count"],
                "main_tau_mae": diagnostics["main_timing"]["mae"],
                "main_tau_rmse": diagnostics["main_timing"]["rmse"],
                "main_tau_clipped_mae": diagnostics["main_timing"]["clipped_mae"],
                "terminal_log_gap_mae": diagnostics["terminal_timing"]["mae"],
                "terminal_log_gap_rmse": diagnostics["terminal_timing"]["rmse"],
                "terminal_seconds_gap_mae": diagnostics["terminal_timing"]["decoded_seconds_gap_mae"],
            })
            validation_rows.append(validation_row)
            validation_slots.append({"epoch": epoch, "diagnostics": diagnostics})
            atomic_csv(OUTPUT_ROOT / "validation_epoch_metrics.csv", validation_rows)
            atomic_json(OUTPUT_ROOT / "validation_slot_metrics.json", {"event_names": ["NONE", "SET_ZERO", "SET_LOW", "SET_HALF", "SET_FULL"], "epochs": validation_slots})
            candidate = float(validation_losses["total"])
            if not math.isfinite(candidate) or not all(
                math.isfinite(float(validation_losses[key]))
                for key in ("initial", "main_event", "main_timing", "terminal_event", "terminal_timing")
            ):
                raise FloatingPointError("non-finite aggregate validation objective")
            if early_best is None or candidate < early_best - MIN_DELTA:
                early_best = candidate
                early_counter = 0
            else:
                early_counter += 1
            if best_val is None or candidate < best_val:
                best_val = candidate
                best_epoch = epoch
                atomic_torch(OUTPUT_ROOT / "best.pt", checkpoint_payload(model, optimizer, scaler, config, epoch, global_state, best_epoch, best_val, early_counter))
            atomic_torch(OUTPUT_ROOT / "last.pt", checkpoint_payload(model, optimizer, scaler, config, epoch, global_state, best_epoch, best_val, early_counter))
            status.update(
                status="running", stage="epoch_complete", epoch=epoch,
                global_optimizer_attempt=global_state["optimizer_attempt"],
                global_optimizer_step=global_state["optimizer_step"],
                best_epoch=best_epoch, best_val_total=best_val,
                latest_train_total=train_losses["total"], latest_val_total=candidate,
                latest_validation_components={key: validation_losses[key] for key in ("initial", "main_event", "main_timing", "terminal_event", "terminal_timing")},
                early_stopping_counter=early_counter,
                amp_overflow_count=global_state["amp_overflow_count"],
                max_consecutive_amp_overflow=global_state["max_consecutive_amp_overflows"],
                updated_at=now(),
            )
            write_run_status(OUTPUT_ROOT / "run_status.json", status)
            log(f"EPOCH_COMPLETE epoch={epoch} train_total={train_losses['total']:.9f} validation_total={candidate:.9f} best_epoch={best_epoch} early_counter={early_counter} amp_overflows={global_state['amp_overflow_count']}")
            if early_counter >= PATIENCE:
                stop_reason = "early_stopping"
                break
        status.update(status="completed", stage="training_complete", stop_reason=stop_reason, updated_at=now())
        write_run_status(OUTPUT_ROOT / "run_status.json", status)
        report(config, train_rows, validation_rows, validation_slots, status)
        log(f"TRAINING_COMPLETE best_epoch={best_epoch} best_val_total={best_val:.9f} inference_started=false")
    except Exception as error:
        primary_traceback = traceback.format_exc()
        logging_error = write_failure_status(
            OUTPUT_ROOT / "run_status.json",
            status,
            failure_reason=f"{type(error).__name__}: {error}",
            primary_traceback=primary_traceback,
            updated_at=now(),
            emergency_log_path=OUTPUT_ROOT / "status_emergency.log",
        )
        try:
            suffix = "" if logging_error is None else f" status_write_error={logging_error!r}"
            log(f"TRAINING_FAILED {type(error).__name__}: {error}{suffix}")
        except Exception:
            print(primary_traceback, file=sys.stderr, flush=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "train"))
    parser.add_argument("--resume")
    arguments = parser.parse_args()
    if arguments.mode == "preflight":
        preflight()
    else:
        train(arguments.resume)


if __name__ == "__main__":
    main()
