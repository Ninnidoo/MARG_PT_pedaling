#!/usr/bin/env python3
"""Canonical seed-42 full training for frozen Custom Event Model v1 B3-S.

This is a deliberately small adaptation of the canonical v0 runner.  The v0
data order, optimizer, AMP recovery, status persistence, early stopping, and
legacy checkpoint-selection objective remain authoritative.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_custom_event_model_v0_full_seed42 as base
from src.stage2_event_model.dataset import IGNORE_INDEX, MAIN_SLOTS
from src.stage2_event_model.dataset_v1_state_conditioned import (
    CustomEventStateConditionedWindowDataset,
    custom_event_state_conditioned_collate_fn,
)
from src.stage2_event_model.losses import CustomEventLossConfig
from src.stage2_event_model.losses_v1_state_conditioned import (
    STATE_LOSS_COEFFICIENT,
    CustomEventStateConditionedCriterion,
)
from src.stage2_event_model.model_v1_state_conditioned import (
    MODEL_VERSION,
    STATE_CLASSES,
    CustomEventEncoderModelV1StateConditioned,
)
from src.stage2_event_model.prediction_decoder_v1 import DECODER_ID, DECODER_VERSION


OUTPUT_ROOT = ROOT / "analysis/custom_event_model_v1_b3s_full_seed42"
IMPLEMENTATION_ROOT = ROOT / "analysis/custom_event_model_v1_state_conditioned_impl"
EXPECTED_MODEL_PARAMETERS = 103_339_216
EXPECTED_DECODER_ID = "938ca7e20e1d398aa530938bbcede404fd3a04af23842fbcf351ba02df0268f8"
EXPECTED_SOURCE_SHA256 = {
    "src/stage2_event_model/model_v1_state_conditioned.py": "5d83ddb66f6a1c0af7b23749afff2e7358d9cc7b1c260f1c4861e79eca8588b7",
    "src/stage2_event_model/dataset_v1_state_conditioned.py": "418b229ed8a5f62f43d2b83c3b8feba49da185397585002ee7d9f824ad500c67",
    "src/stage2_event_model/losses_v1_state_conditioned.py": "4aaec824e9ac9ad1528489b3c6a7254eb9ff1e8ceacbfdc2401e73064a09b8fa",
}
STATE_NAMES = ("ZERO", "LOW", "HALF", "FULL")
EVENT_NAMES = ("NONE", "SET_ZERO", "SET_LOW", "SET_HALF", "SET_FULL")

_v0_configuration = base.configuration
_v0_verify_frozen_artifacts = base.verify_frozen_artifacts
_v0_preflight = base.preflight
_v0_atomic_torch = base.atomic_torch
_v0_flatten_epoch = base.flatten_epoch
_v0_checkpoint_payload = base.checkpoint_payload
_head_initialization_sha256: str | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def log(message: str) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    line = f"{base.now()} {message}"
    with (OUTPUT_ROOT / "train.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
    print(line, flush=True)


def configuration(micro_batch: int) -> dict[str, Any]:
    if micro_batch != 4:
        raise RuntimeError("canonical B3-S requires micro batch 4; no fallback is authorized")
    config = _v0_configuration(micro_batch)
    config.update(
        experiment="custom_event_model_v1_b3s_full_seed42",
        model_version=MODEL_VERSION,
        candidate="B3-S",
        decoder_id=EXPECTED_DECODER_ID,
        decoder_version=DECODER_VERSION,
        state_loss_coefficient=STATE_LOSS_COEFFICIENT,
        state_loss_weighting="unweighted Slot1-6 macro-average",
        optimization_objective="legacy_selection_total + main_state_ce",
        checkpoint_selection="minimum legacy v0 validation objective; main_state_ce excluded",
        epoch_checkpoint_retention="model-state-only checkpoints/epoch_XX.pt",
        architecture={
            "hidden_size": 768,
            "encoder_parameters": 103_271_424,
            "total_parameters": EXPECTED_MODEL_PARAMETERS,
            "initial": "Linear(768,4)",
            "main_state": "6 independent Linear(768,4), input stopgrad(h)",
            "main_event": "6 independent Linear(772,5), concat(h, stopgrad(q))",
            "state_feature": "fixed 4D posterior q",
            "main_timing": "6 independent Linear(768,1)",
            "terminal_event": "4 independent Linear(768,5)",
            "terminal_timing": "4 independent Linear(768,1)",
        },
    )
    config["source_sha256"] = dict(EXPECTED_SOURCE_SHA256)
    fingerprint_source = {key: value for key, value in config.items() if key != "config_fingerprint"}
    config["config_fingerprint"] = hashlib.sha256(
        json.dumps(fingerprint_source, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return config


def verify_frozen_artifacts() -> dict[str, Any]:
    frozen = _v0_verify_frozen_artifacts()
    if DECODER_VERSION != "1.0.0" or DECODER_ID != EXPECTED_DECODER_ID:
        raise RuntimeError("frozen Decoder v1 identity mismatch")
    for relative, expected in EXPECTED_SOURCE_SHA256.items():
        actual = sha256_file(ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"B3-S implementation source changed: {relative}: {actual}")
    implementation = json.loads(
        (IMPLEMENTATION_ROOT / "implementation_config.json").read_text(encoding="utf-8")
    )
    accounting = json.loads(
        (IMPLEMENTATION_ROOT / "parameter_accounting.json").read_text(encoding="utf-8")
    )
    gradients = json.loads(
        (IMPLEMENTATION_ROOT / "gradient_isolation_results.json").read_text(encoding="utf-8")
    )
    tiny = json.loads(
        (IMPLEMENTATION_ROOT / "tiny_overfit_metrics.json").read_text(encoding="utf-8")
    )
    if (
        implementation.get("status") != "completed"
        or implementation.get("final_verdict") != "READY FOR B3-S FULL TRAINING"
        or not implementation.get("correctness_test_passed")
        or not accounting.get("passed")
        or accounting.get("actual_delta") != 18_576
        or accounting.get("v1_actual_total_parameters") != EXPECTED_MODEL_PARAMETERS
        or not gradients.get("all_cases_passed")
        or tiny.get("success_assessment", {}).get("verdict") != "READY FOR B3-S FULL TRAINING"
    ):
        raise RuntimeError("frozen B3-S implementation readiness artifact mismatch")
    frozen.update(
        model_version=MODEL_VERSION,
        decoder_id=DECODER_ID,
        decoder_version=DECODER_VERSION,
        b3s_implementation_status="READY FOR B3-S FULL TRAINING",
        b3s_source_sha256=dict(EXPECTED_SOURCE_SHA256),
        parameter_delta=18_576,
        model_parameters=EXPECTED_MODEL_PARAMETERS,
        gradient_isolation="A/B/C/D PASS",
    )
    return frozen


def _head_digest(model: CustomEventEncoderModelV1StateConditioned) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not name.startswith("encoder."):
            digest.update(name.encode("utf-8"))
            digest.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def build_model(device: torch.device) -> CustomEventEncoderModelV1StateConditioned:
    global _head_initialization_sha256
    model = CustomEventEncoderModelV1StateConditioned.from_pretrained(
        base.PRETRAINED,
        head_init_seed=base.SEED,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    ).to(device)
    if (
        model.hidden_size != 768
        or model.encoder_parameter_count != 103_271_424
        or model.parameter_count != EXPECTED_MODEL_PARAMETERS
        or MODEL_VERSION != "1.0.0-b3s"
    ):
        raise RuntimeError("frozen Custom Event Model v1 B3-S architecture changed")
    digest = _head_digest(model)
    if _head_initialization_sha256 is not None and digest != _head_initialization_sha256:
        raise RuntimeError("seed-42 B3-S head initialization is not deterministic")
    preflight_path = OUTPUT_ROOT / "preflight.json"
    if preflight_path.exists():
        stored = json.loads(preflight_path.read_text(encoding="utf-8"))
        expected = stored.get("head_initialization_sha256")
        if expected is not None and digest != expected:
            raise RuntimeError("training initialization contradicts preflight hash")
    _head_initialization_sha256 = digest
    return model


def build_optimizer(model: CustomEventEncoderModelV1StateConditioned) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        [
            {"params": list(model.encoder.parameters()), "lr": 1e-5, "group_name": "encoder"},
            {"params": list(model.prediction_head_parameters()), "lr": 1e-4, "group_name": "custom_heads"},
        ],
        weight_decay=0.01,
    )


@dataclass
class B3SRawTerms:
    legacy: base.RawLossTerms
    main_state: list[torch.Tensor]


def state_counts(batch: Mapping[str, torch.Tensor]) -> list[int]:
    owned = batch["owned_onset_mask"].bool()
    targets = batch["main_pre_state_targets"]
    return [int((owned & (targets[:, :, slot] != IGNORE_INDEX)).sum().item()) for slot in range(MAIN_SLOTS)]


def raw_terms(output: Any, batch: Mapping[str, torch.Tensor], criterion: Any) -> B3SRawTerms:
    legacy = base.raw_loss_terms(output, batch, criterion)
    owned = batch["owned_onset_mask"].bool()
    targets = batch["main_pre_state_targets"]
    state: list[torch.Tensor] = []
    for slot in range(MAIN_SLOTS):
        valid = owned & (targets[:, :, slot] != IGNORE_INDEX)
        if bool(valid.any()):
            selected = targets[:, :, slot][valid]
            if bool(((selected < 0) | (selected >= STATE_CLASSES)).any()):
                raise ValueError("pre-state target outside [0,3]")
            state.append(F.cross_entropy(output.main_state_logits[:, :, slot][valid], selected, reduction="sum"))
        else:
            state.append(output.main_state_logits[:, :, slot].sum() * 0.0)
    return B3SRawTerms(legacy, state)


def normalized_b3s(
    terms: B3SRawTerms,
    denominators: base.LossDenominators,
    counts: Sequence[int],
    config: CustomEventLossConfig,
) -> dict[str, torch.Tensor]:
    legacy = base.normalized_components(terms.legacy, denominators, config)
    slots = [value / count if count else value.sum() * 0.0 for value, count in zip(terms.main_state, counts)]
    state = torch.stack(slots).mean()
    return {
        **legacy,
        "legacy_total": legacy["total"],
        "main_state": state,
        "optimization_total": legacy["total"] + STATE_LOSS_COEFFICIENT * state,
    }


def empty_b3s_sums() -> dict[str, Any]:
    result = base.empty_raw_sums()
    result["main_state"] = [0.0] * MAIN_SLOTS
    return result


def add_b3s_sums(destination: dict[str, Any], terms: B3SRawTerms) -> None:
    base.add_raw_terms(destination, terms.legacy)
    for slot, value in enumerate(terms.main_state):
        destination["main_state"][slot] += float(value.detach().float())


def b3s_components_from_sums(
    sums: Mapping[str, Any], denominators: base.LossDenominators, counts: Sequence[int], config: Any
) -> dict[str, Any]:
    legacy = base.components_from_sums(sums, denominators, config)
    state_slots = [float(value / count) if count else 0.0 for value, count in zip(sums["main_state"], counts)]
    state = float(sum(state_slots) / MAIN_SLOTS)
    result = {
        **legacy,
        "legacy_total": legacy["total"],
        "main_state": state,
        "main_state_by_slot": state_slots,
        "optimization_total": legacy["total"] + STATE_LOSS_COEFFICIENT * state,
    }
    if not all(math.isfinite(value) for value in result.values() if isinstance(value, float)):
        raise FloatingPointError("non-finite B3-S epoch loss component")
    return result


class B3SValidationDiagnostics(base.ValidationDiagnostics):
    def __init__(self) -> None:
        super().__init__()
        self.state_confusions = [np.zeros((STATE_CLASSES, STATE_CLASSES), dtype=np.int64) for _ in range(MAIN_SLOTS)]

    def update(self, output: Any, batch: Mapping[str, torch.Tensor]) -> None:
        super().update(output, batch)
        owned = batch["owned_onset_mask"].bool()
        targets = batch["main_pre_state_targets"]
        for slot in range(MAIN_SLOTS):
            valid = owned & (targets[:, :, slot] != IGNORE_INDEX)
            self._add_confusion(
                self.state_confusions[slot],
                targets[:, :, slot][valid],
                output.main_state_logits[:, :, slot].argmax(-1)[valid],
                STATE_CLASSES,
            )

    @staticmethod
    def _state_metrics(confusion: np.ndarray) -> dict[str, Any]:
        total = int(confusion.sum())
        target = confusion.sum(axis=1)
        predicted = confusion.sum(axis=0)
        f1s = []
        recalls = []
        for index in range(STATE_CLASSES):
            tp = int(confusion[index, index])
            precision = tp / int(predicted[index]) if predicted[index] else 0.0
            recall = tp / int(target[index]) if target[index] else 0.0
            recalls.append(recall)
            f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        return {
            "target_count": total,
            "accuracy": float(np.trace(confusion) / total) if total else 0.0,
            "macro_f1": float(sum(f1s) / STATE_CLASSES),
            "class_f1": dict(zip(STATE_NAMES, f1s)),
            "class_recall": dict(zip(STATE_NAMES, recalls)),
            "target_distribution": {name: {"count": int(target[i]), "rate": float(target[i] / total) if total else 0.0} for i, name in enumerate(STATE_NAMES)},
            "prediction_distribution": {name: {"count": int(predicted[i]), "rate": float(predicted[i] / total) if total else 0.0} for i, name in enumerate(STATE_NAMES)},
            "confusion": confusion.tolist(),
        }

    def result(self) -> dict[str, Any]:
        result = super().result()
        pooled = sum(self.state_confusions, np.zeros((STATE_CLASSES, STATE_CLASSES), dtype=np.int64))
        result["main_state"] = self._state_metrics(pooled)
        result["main_state_by_slot"] = [self._state_metrics(value) for value in self.state_confusions]
        main_confusion = np.asarray(result["main_event"]["confusion"], dtype=np.int64)
        result["main_destination_frequency"] = {
            EVENT_NAMES[index]: {
                "target": int(main_confusion[index, :].sum()),
                "predicted": int(main_confusion[:, index].sum()),
            }
            for index in range(1, len(EVENT_NAMES))
        }
        return result


def run_epoch(
    model: Any, loader: Any, criterion: Any, *, device: torch.device, accumulation: int,
    optimizer: torch.optim.Optimizer | None, scaler: torch.amp.GradScaler | None,
    epoch: int, global_state: dict[str, int], status: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    training = optimizer is not None
    model.train(training)
    if training and scaler is None:
        raise ValueError("training requires GradScaler")
    if training:
        corrupt = base.nonfinite_parameters(model)
        if corrupt:
            raise FloatingPointError("non-finite model parameters: " + ", ".join(corrupt[:8]))
    epoch_sums = empty_b3s_sums()
    epoch_denominators = base.LossDenominators.zero()
    epoch_state_counts = [0] * MAIN_SLOTS
    diagnostics = None if training else B3SValidationDiagnostics()
    amp_events = 0
    max_consecutive = global_state["consecutive_amp_overflows"]
    first_step = None
    iterator = iter(loader)
    group_index = 0
    while True:
        cpu_group = list(itertools.islice(iterator, accumulation if training else 1))
        if not cpu_group:
            break
        group_index += 1
        group_denom = base.group_denominators(cpu_group, criterion.main_event_weights.cpu(), criterion.terminal_event_weights.cpu())
        group_state_counts = [sum(state_counts(batch)[slot] for batch in cpu_group) for slot in range(MAIN_SLOTS)]
        group_sums = empty_b3s_sums()
        if training:
            optimizer.zero_grad(set_to_none=True)
        for cpu_batch in cpu_group:
            batch = base.move_batch(cpu_batch, device)
            with torch.set_grad_enabled(training), torch.amp.autocast("cuda", dtype=torch.float16, enabled=True):
                output = base.forward_model(model, batch)
                terms = raw_terms(output, batch, criterion)
                components = normalized_b3s(terms, group_denom, group_state_counts, criterion.config)
            tensors = (
                output.initial_logits, output.main_state_logits, output.main_event_logits,
                output.main_timing_predictions, output.terminal_event_logits,
                output.terminal_timing_predictions, *components.values(),
            )
            if not all(bool(torch.isfinite(value).all()) for value in tensors):
                raise FloatingPointError("non-finite forward/objective")
            if training:
                scaler.scale(components["optimization_total"]).backward()
            add_b3s_sums(group_sums, terms)
            add_b3s_sums(epoch_sums, terms)
            batch_denom = base.batch_denominators(cpu_batch, criterion.main_event_weights.cpu(), criterion.terminal_event_weights.cpu())
            epoch_denominators.add_(batch_denom)
            counts = state_counts(cpu_batch)
            epoch_state_counts = [a + b for a, b in zip(epoch_state_counts, counts)]
            if diagnostics is not None:
                diagnostics.update(output, batch)
        group_components = b3s_components_from_sums(group_sums, group_denom, group_state_counts, criterion.config)
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
                overflow_details = base.gradient_overflow_details(model)
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
                global_state["max_consecutive_amp_overflows"] = max(global_state["max_consecutive_amp_overflows"], global_state["consecutive_amp_overflows"])
                max_consecutive = max(max_consecutive, global_state["consecutive_amp_overflows"])
                status["last_overflow"] = {
                    "epoch": epoch, "epoch_optimizer_attempt": group_index,
                    "global_optimizer_attempt": global_state["optimizer_attempt"],
                    "gradient_norm": grad_norm_value, "scale_before": scale_before,
                    "scale_after": scale_after, **overflow_details,
                }
                status.update(
                    epoch=epoch, global_optimizer_attempt=global_state["optimizer_attempt"],
                    global_optimizer_step=global_state["optimizer_step"],
                    latest_train_total=float(group_components["optimization_total"]),
                    latest_train_legacy_total=float(group_components["legacy_total"]),
                    latest_train_state_ce=float(group_components["main_state"]),
                    amp_overflow_count=global_state["amp_overflow_count"],
                    max_consecutive_amp_overflow=global_state["max_consecutive_amp_overflows"], updated_at=base.now(),
                )
                base.write_run_status(OUTPUT_ROOT / "run_status.json", status)
                log(f"AMP_OVERFLOW epoch={epoch} attempt={group_index} scale={scale_before}->{scale_after} affected={overflow_details['affected_groups']}")
                corrupt = base.nonfinite_parameters(model)
                if corrupt:
                    raise FloatingPointError("parameter corruption after overflow: " + ", ".join(corrupt[:8]))
                if global_state["consecutive_amp_overflows"] >= base.MAX_CONSECUTIVE_AMP_OVERFLOWS:
                    raise FloatingPointError("3 consecutive AMP gradient overflows")
            if first_step is None and applied:
                first_step = {
                    "epoch": epoch, "global_optimizer_attempt": global_state["optimizer_attempt"],
                    "global_optimizer_step": global_state["optimizer_step"], "gradient_norm": grad_norm_value,
                    "grad_scaler_scale": scale_after,
                    **{key: float(group_components[key]) for key in ("optimization_total", "legacy_total", "main_state", "initial", "main_event", "main_timing", "terminal_event", "terminal_timing")},
                }
                status.update(
                    status="running", stage="training", epoch=epoch,
                    global_optimizer_attempt=global_state["optimizer_attempt"],
                    global_optimizer_step=global_state["optimizer_step"],
                    latest_train_total=float(group_components["optimization_total"]),
                    latest_train_legacy_total=float(group_components["legacy_total"]),
                    latest_train_state_ce=float(group_components["main_state"]),
                    first_finite_optimizer_step=first_step, updated_at=base.now(),
                )
                base.write_run_status(OUTPUT_ROOT / "run_status.json", status)
                log("FIRST_OPTIMIZER_STEP_FINITE " + " ".join(f"{key}={first_step[key]:.9f}" for key in ("optimization_total", "legacy_total", "main_state", "initial", "main_event", "main_timing", "terminal_event", "terminal_timing")) + f" grad_norm={grad_norm_value:.6f} scaler={scale_after:.1f}")
            if group_index % 100 == 0:
                status.update(
                    epoch=epoch, global_optimizer_attempt=global_state["optimizer_attempt"],
                    global_optimizer_step=global_state["optimizer_step"],
                    latest_train_total=float(group_components["optimization_total"]),
                    latest_train_legacy_total=float(group_components["legacy_total"]),
                    latest_train_state_ce=float(group_components["main_state"]),
                    amp_overflow_count=global_state["amp_overflow_count"],
                    max_consecutive_amp_overflow=global_state["max_consecutive_amp_overflows"], updated_at=base.now(),
                )
                base.write_run_status(OUTPUT_ROOT / "run_status.json", status)
                log(f"TRAIN_PROGRESS epoch={epoch} attempt={group_index} global_step={global_state['optimizer_step']} optimization_total={group_components['optimization_total']:.9f} legacy_total={group_components['legacy_total']:.9f} state_ce={group_components['main_state']:.9f}")
    metrics = b3s_components_from_sums(epoch_sums, epoch_denominators, epoch_state_counts, criterion.config)
    checks = {
        "optimizer_attempts": group_index if training else 0,
        "amp_overflow_events": amp_events,
        "max_consecutive_amp_overflows": max_consecutive,
        "grad_scaler_scale": float(scaler.get_scale()) if training else None,
    }
    return metrics, checks, diagnostics.result() if diagnostics is not None else None


def flatten_epoch(epoch: int, losses: Mapping[str, Any], seconds: float, extra: Mapping[str, Any]) -> dict[str, Any]:
    row = _v0_flatten_epoch(epoch, losses, seconds, extra)
    row["optimization_total"] = losses["optimization_total"]
    row["legacy_selection_total"] = losses["legacy_total"]
    row["main_state_ce"] = losses["main_state"]
    for index, value in enumerate(losses["main_state_by_slot"], 1):
        row[f"main_state_by_slot_slot{index}"] = value
    return row


def atomic_torch(path: Path, payload: Mapping[str, Any]) -> None:
    _v0_atomic_torch(path, payload)
    if path == OUTPUT_ROOT / "last.pt":
        epoch = int(payload["epoch"])
        diagnostic = {
            "model_state": payload["model_state"], "epoch": epoch,
            "model_version": MODEL_VERSION, "seed": base.SEED,
            "tokenizer_cache_id": base.CANONICAL_CACHE_ID,
            "note_alignment_id": base.CANONICAL_ALIGNMENT_ID,
            "note_alignment_config_sha256": base.EXPECTED_ALIGNMENT_CONFIG_SHA,
            "decoder_id": EXPECTED_DECODER_ID,
            "event_weight_sha256": base.EXPECTED_WEIGHT_SHA,
            "config_fingerprint": payload["config_fingerprint"],
            "checkpoint_kind": "epoch_diagnostic_model_state_only",
        }
        (OUTPUT_ROOT / "checkpoints").mkdir(parents=True, exist_ok=True)
        _v0_atomic_torch(OUTPUT_ROOT / "checkpoints" / f"epoch_{epoch:02d}.pt", diagnostic)


def checkpoint_payload(*args: Any, **kwargs: Any) -> dict[str, Any]:
    payload = _v0_checkpoint_payload(*args, **kwargs)
    payload.update(model_version=MODEL_VERSION, decoder_id=EXPECTED_DECODER_ID, state_loss_coefficient=STATE_LOSS_COEFFICIENT)
    return payload


def _checkpoint_records() -> list[dict[str, Any]]:
    paths = [OUTPUT_ROOT / "best.pt", OUTPUT_ROOT / "last.pt", *sorted((OUTPUT_ROOT / "checkpoints").glob("epoch_*.pt"))]
    return [{"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in paths if path.exists()]


def report(config: Mapping[str, Any], train_rows: list[dict[str, Any]], validation_rows: list[dict[str, Any]], validation_slots: list[dict[str, Any]], status: Mapping[str, Any]) -> None:
    best = min(validation_rows, key=lambda row: float(row["total"]))
    best_diag = next(row["diagnostics"] for row in validation_slots if row["epoch"] == best["epoch"])
    combined = []
    for train, validation in zip(train_rows, validation_rows):
        combined.append({
            "epoch": int(train["epoch"]),
            "train_optimization_total": float(train["optimization_total"]),
            "train_legacy_total": float(train["legacy_selection_total"]),
            "val_legacy_selection_total": float(validation["legacy_selection_total"]),
            "val_main_state_ce": float(validation["main_state_ce"]),
            "val_initial_ce": float(validation["initial_ce"]),
            "val_main_event_ce": float(validation["main_event_ce"]),
            "val_main_timing_loss": float(validation["main_timing_huber"]),
            "val_terminal_event_ce": float(validation["terminal_event_ce"]),
            "val_terminal_timing_loss": float(validation["terminal_timing_huber"]),
        })
    base.atomic_csv(OUTPUT_ROOT / "metrics.csv", combined)
    checkpoints = _checkpoint_records()
    best_payload = {"epoch": int(best["epoch"]), "diagnostics": best_diag}
    base.atomic_json(OUTPUT_ROOT / "best_epoch_diagnostics.json", best_payload)
    summary = {
        "status": status["status"], "model_version": MODEL_VERSION,
        "model_parameters": EXPECTED_MODEL_PARAMETERS, "best_epoch": int(best["epoch"]),
        "best_validation_legacy_selection_total": float(best["total"]),
        "stop_reason": status.get("stop_reason"),
        "early_stopping_epoch": int(status["epoch"]) if status.get("stop_reason") == "early_stopping" else None,
        "epochs_completed": len(validation_rows), "amp_overflow_count": status["amp_overflow_count"],
        "max_consecutive_amp_overflow": status["max_consecutive_amp_overflow"],
        "checkpoints": checkpoints,
        "asap_test_access_count": 0, "repedal_execution_count": 0,
        "canonical_validation_inference_count": 0, "candidate_midi_generation_count": 0,
    }
    base.atomic_json(OUTPUT_ROOT / "training_summary.json", summary)
    state = best_diag["main_state"]
    event = best_diag["main_event"]
    lines = [
        "# Custom Event Model v1 B3-S Canonical Training Report", "",
        "## Frozen provenance", "",
        f"- Model/version: B3-S / `{MODEL_VERSION}`", f"- Parameters: {EXPECTED_MODEL_PARAMETERS:,}",
        f"- Tokenizer cache ID: `{base.CANONICAL_CACHE_ID}`", f"- Alignment ID: `{base.CANONICAL_ALIGNMENT_ID}`",
        f"- Alignment config SHA: `{base.EXPECTED_ALIGNMENT_CONFIG_SHA}`", f"- Decoder: `{DECODER_VERSION}` / `{EXPECTED_DECODER_ID}`",
        f"- Event-weight SHA: `{base.EXPECTED_WEIGHT_SHA}`", f"- Seed/head initialization SHA: 42 / `{_head_initialization_sha256}`", "",
        "## Data and training configuration", "",
        "- Train: 2,062 performances / 35,573 windows / 9,009,549 owned onsets",
        "- Validation: 71 performances / 1,078 windows / 272,927 owned onsets (19 pieces)",
        f"- AdamW; encoder/head LR 1e-5/1e-4; weight decay 0.01; grad clip 1.0",
        f"- FP16 AMP; micro/accumulation/effective batch {config['micro_batch_size']}/{config['gradient_accumulation_steps']}/16; no scheduler",
        "- Maximum epochs 10; patience 3; min delta 1e-4",
        "- Optimization adds unit-weight unweighted slot-macro state CE; selection is the legacy v0 total and excludes state CE.", "",
        "## Epoch objectives", "",
        "| epoch | train optimization | train legacy | val legacy selection | val state CE | val main event | val main timing | val initial | val terminal event | val terminal timing |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in combined:
        lines.append("| {epoch} | {train_optimization_total:.6f} | {train_legacy_total:.6f} | {val_legacy_selection_total:.6f} | {val_main_state_ce:.6f} | {val_main_event_ce:.6f} | {val_main_timing_loss:.6f} | {val_initial_ce:.6f} | {val_terminal_event_ce:.6f} | {val_terminal_timing_loss:.6f} |".format(**row))
    lines += [
        "", "## Selection and numerical summary", "",
        f"- Best epoch: {int(best['epoch'])}; legacy validation selection total: {float(best['total']):.9f}",
        f"- Stop reason / early-stopping epoch: {status.get('stop_reason')} / {summary['early_stopping_epoch']}",
        f"- AMP overflow events / maximum consecutive: {status['amp_overflow_count']} / {status['max_consecutive_amp_overflow']}", "",
        "## Best-checkpoint state-head diagnostics", "",
        f"- Overall accuracy / Macro F1: {state['accuracy']:.6f} / {state['macro_f1']:.6f}",
        f"- Target distribution: `{json.dumps(state['target_distribution'], sort_keys=True)}`",
        f"- Prediction distribution: `{json.dumps(state['prediction_distribution'], sort_keys=True)}`",
    ]
    for slot, slot_state in enumerate(best_diag["main_state_by_slot"], 1):
        lines.append(f"- Slot{slot} accuracy / Macro F1: {slot_state['accuracy']:.6f} / {slot_state['macro_f1']:.6f}; recalls `{json.dumps(slot_state['class_recall'], sort_keys=True)}`")
    lines += [
        "", "## Best-checkpoint Main-event target-space diagnostics", "",
        f"- Overall accuracy / non-NONE precision/recall/F1: {event['accuracy']:.6f} / {event['non_none_precision']:.6f} / {event['non_none_recall']:.6f} / {event['non_none_f1']:.6f}",
        f"- Predicted active / target active: {event['predicted_non_none_count']} / {event['target_non_none_count']}",
        f"- Destination target/predicted frequency: `{json.dumps(best_diag['main_destination_frequency'], sort_keys=True)}`",
    ]
    for slot, slot_event in enumerate(best_diag["main_event_by_slot"], 1):
        lines.append(f"- Slot{slot} predicted active / target active / F1: {slot_event['predicted_non_none_count']} / {slot_event['target_non_none_count']} / {slot_event['non_none_f1']:.6f}")
    lines += ["", "## Checkpoints", ""]
    lines.extend(f"- `{row['path']}` — `{row['sha256']}` ({row['bytes']} bytes)" for row in checkpoints)
    lines += [
        "", "## Scientific boundary", "",
        "This report makes no v0-versus-v1 final-performance superiority claim.",
        "- No canonical MIDI validation inference", "- No matched Hybrid comparison",
        "- No ASAP test", "- No Repedal", "- No candidate MIDI or 4C/Transition/JS/Intersection evaluation",
    ]
    (OUTPUT_ROOT / "CUSTOM_EVENT_MODEL_V1_B3S_TRAINING_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def preflight() -> dict[str, Any]:
    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
        raise RuntimeError(f"fresh canonical output root is not empty: {OUTPUT_ROOT}")
    payload = _v0_preflight()
    if int(payload["selected_micro_batch"]) != 4:
        raise RuntimeError("micro batch 4 did not fit; canonical config changes are not authorized")
    payload.update(
        model_version=MODEL_VERSION, model_parameters=EXPECTED_MODEL_PARAMETERS,
        decoder_id=DECODER_ID, decoder_version=DECODER_VERSION,
        head_initialization_sha256=_head_initialization_sha256,
        state_loss_finite_and_enabled=True,
        checkpoint_selection_excludes_state_ce=True,
    )
    base.atomic_json(OUTPUT_ROOT / "preflight.json", payload)
    provenance = verify_frozen_artifacts()
    provenance.update(
        created_at=base.now(), seed=base.SEED,
        head_initialization_sha256=_head_initialization_sha256,
        train_windows=35_573, validation_windows=1_078,
        train_performances=2_062, validation_performances=71,
        asap_test_access_count=0, repedal_execution_count=0,
        canonical_validation_inference_count=0,
    )
    provenance.pop("main_weights", None)
    provenance.pop("terminal_weights", None)
    base.atomic_json(OUTPUT_ROOT / "provenance.json", provenance)
    return payload


def install_v1_adaptation() -> None:
    base.OUTPUT_ROOT = OUTPUT_ROOT
    base.CustomEventWindowDataset = CustomEventStateConditionedWindowDataset
    base.custom_event_collate_fn = custom_event_state_conditioned_collate_fn
    base.CustomEventCriterion = CustomEventStateConditionedCriterion
    base.CustomEventEncoderModelV0 = CustomEventEncoderModelV1StateConditioned
    base.configuration = configuration
    base.verify_frozen_artifacts = verify_frozen_artifacts
    base.build_model = build_model
    base.build_optimizer = build_optimizer
    base.run_epoch = run_epoch
    base.flatten_epoch = flatten_epoch
    base.atomic_torch = atomic_torch
    base.checkpoint_payload = checkpoint_payload
    base.report = report
    base.log = log


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "train"))
    parser.add_argument("--resume")
    arguments = parser.parse_args()
    install_v1_adaptation()
    if arguments.mode == "preflight":
        preflight()
    else:
        base.train(arguments.resume)


if __name__ == "__main__":
    main()
