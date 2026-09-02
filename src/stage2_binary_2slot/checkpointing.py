"""Fault-tolerant train-epoch checkpoints saved before validation."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


REQUIRED_RESUME_KEYS = frozenset({
    "checkpoint_kind", "completed_training_epoch", "validations_complete",
    "global_step", "model_state", "optimizer_state", "grad_scaler_state",
    "python_random_state", "numpy_random_state", "torch_cpu_rng_state",
    "torch_cuda_rng_state_all", "run_configuration",
})


def build_resumable_epoch_payload(
    *, model: torch.nn.Module, optimizer: torch.optim.Optimizer, scaler: Any,
    completed_training_epoch: int, global_step: int,
    run_configuration: Mapping[str, Any], extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if completed_training_epoch < 1 or global_step < 1:
        raise ValueError("resumable checkpoint requires completed training progress")
    payload = {
        "checkpoint_kind": "train_epoch_complete_pre_validation",
        "completed_training_epoch": int(completed_training_epoch),
        "validations_complete": False,
        "global_step": int(global_step),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "grad_scaler_state": scaler.state_dict(),
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "run_configuration": dict(run_configuration),
    }
    if extra:
        payload.update(dict(extra))
    validate_resumable_epoch_payload(payload)
    return payload


def validate_resumable_epoch_payload(payload: Mapping[str, Any]) -> None:
    missing = REQUIRED_RESUME_KEYS - set(payload)
    if missing:
        raise ValueError(f"resumable checkpoint missing keys: {sorted(missing)}")
    if payload["checkpoint_kind"] != "train_epoch_complete_pre_validation":
        raise ValueError("unexpected resumable checkpoint kind")
    if bool(payload["validations_complete"]):
        raise ValueError("pre-validation checkpoint cannot claim completed validation")


def atomic_save_resumable_checkpoint(path: str | Path, payload: Mapping[str, Any]) -> None:
    validate_resumable_epoch_payload(payload)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, destination)


def restore_resumable_epoch_payload(
    payload: Mapping[str, Any], *, model: torch.nn.Module,
    optimizer: torch.optim.Optimizer, scaler: Any, restore_rng: bool = True,
) -> tuple[int, int]:
    validate_resumable_epoch_payload(payload)
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    scaler.load_state_dict(payload["grad_scaler_state"])
    if restore_rng:
        random.setstate(payload["python_random_state"])
        np.random.set_state(payload["numpy_random_state"])
        torch.set_rng_state(payload["torch_cpu_rng_state"])
        if torch.cuda.is_available() and payload["torch_cuda_rng_state_all"]:
            torch.cuda.set_rng_state_all(payload["torch_cuda_rng_state_all"])
    return int(payload["completed_training_epoch"]), int(payload["global_step"])
