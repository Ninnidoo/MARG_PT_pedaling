#!/usr/bin/env python3
"""Sequential canonical four-class Stage 2 architecture training."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage2_binary.full_training import training_window_order
from src.stage2_binary.training import build_binary_optimizer
from src.stage2_encoder_only.dataset import PEDAL_TOKEN_OFFSET
from src.stage2_encoder_only.train import atomic_torch_save, get_gpu_identity
from src.stage2_encoder_only.training import set_deterministic_seed
from src.stage2_four_class.model import (
    FourClassPedalEncoderDecoderModel,
    FourClassPedalEncoderModel,
)
from src.stage2_four_class.representation import (
    CLASS_BOUNDS,
    CLASS_NAMES,
    IGNORE_INDEX,
    NUM_CLASSES,
    REPRESENTATIVES,
    SharedFourClassWindowDataset,
    classify_cc64,
    make_four_class_loader,
)


ARCHITECTURES = ("encoder_only", "encoder_decoder")
MODEL_CLASSES = {
    "encoder_only": FourClassPedalEncoderModel,
    "encoder_decoder": FourClassPedalEncoderDecoderModel,
}
OUTPUT_NAMES = {
    "encoder_only": "encoder_only_ce",
    "encoder_decoder": "encoder_decoder_ce",
}
METRIC_FIELDS = (
    "epoch",
    "train_ce",
    "train_token_accuracy",
    "train_exact_note_accuracy",
    "validation_ce",
    "validation_token_accuracy",
    "validation_exact_note_accuracy",
    "optimizer_steps",
    "amp_skipped_steps",
    "epoch_seconds",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parameter_hash(parameters: Any) -> str:
    digest = hashlib.sha256()
    with torch.no_grad():
        for parameter in parameters:
            value = parameter.detach().cpu().contiguous()
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def update_arch_status(
    output_root: Path, architecture: str, status: str | None = None, **details: Any
) -> dict[str, Any]:
    path = output_root / "run_status.json"
    payload = read_json(path)
    current = dict(payload[architecture])
    if status is not None:
        current["status"] = status
    current.update(details, updated_at=now())
    payload[architecture] = current
    payload["updated_at"] = now()
    atomic_json(path, payload)
    return payload


def verify_config(config: Mapping[str, Any]) -> None:
    expected = {
        "seed": 42,
        "decoder_init_seed": 42,
        "window_notes": 512,
        "stride_notes": 256,
        "effective_batch_size": 16,
        "optimizer": "AdamW",
        "encoder_lr": 1e-5,
        "head_lr": 1e-4,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "amp_enabled": True,
        "max_epochs": 10,
        "early_stopping_patience": 3,
        "early_stopping_min_delta": 1e-4,
        "checkpoint_selection": "minimum_validation_ce",
        "loss": "standard_unweighted_cross_entropy",
        "scheduled_sampling": False,
        "class_weighting": None,
    }
    changed = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if changed:
        raise RuntimeError(f"controlled four-class configuration changed: {changed}")
    if config.get("representatives") != list(REPRESENTATIVES):
        raise RuntimeError("canonical representatives changed")
    if config.get("class_boundaries") != [list(item) for item in CLASS_BOUNDS]:
        raise RuntimeError("canonical class boundaries changed")
    for architecture in ARCHITECTURES:
        micro = int(config["micro_batch_size"][architecture])
        accumulation = int(config["gradient_accumulation_steps"][architecture])
        if micro * accumulation != int(config["effective_batch_size"]):
            raise RuntimeError(f"effective batch mismatch for {architecture}")


def csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    """Verify immutable cache, manifests, splits, and boundary policy."""

    verify_config(config)
    cache_root = Path(config["cache_root"])
    stats_path = cache_root / "cache_statistics.json"
    stats = read_json(stats_path)
    if not stats.get("completed") or stats.get("cache_id") != config["expected_cache_id"]:
        raise RuntimeError("canonical shared cache completion/ID mismatch")
    if int(stats["window_notes"]) != 512 or int(stats["stride_notes"]) != 256:
        raise RuntimeError("shared cache window/stride mismatch")

    expected_inventory = {
        "train": (2062, 9369095, 35573),
        "validation": (71, 283928, 1078),
    }
    for split, expected in expected_inventory.items():
        actual = tuple(
            int(stats["splits"][split][key])
            for key in ("performances", "notes", "windows")
        )
        if actual != expected:
            raise RuntimeError(f"{split} cache inventory mismatch: {actual}")

    sources = stats["splits"]["train"]["source_statistics"]
    if int(sources["MAESTRO-clean"]["performances"]) != 1170:
        raise RuntimeError("MAESTRO-clean performance count mismatch")
    if int(sources["ASAP-train"]["performances"]) != 892:
        raise RuntimeError("ASAP-train performance count mismatch")
    validation_sources = stats["splits"]["validation"]["source_statistics"]
    if set(validation_sources) != {"ASAP-validation"} or int(
        validation_sources["ASAP-validation"]["performances"]
    ) != 71:
        raise RuntimeError("ASAP validation inventory mismatch")

    manifest_rows = csv_rows(config["train_manifest"])
    if len(manifest_rows) != 2062:
        raise RuntimeError("train manifest row count mismatch")
    train_source_counts: dict[str, int] = {}
    asap_validation_rows = 0
    asap_test_rows = 0
    for row in manifest_rows:
        train_source_counts[row["source"]] = train_source_counts.get(row["source"], 0) + 1
        if row["source"].startswith("ASAP") and (
            row["dataset_split"].lower() == "validation"
            or row["source"] == "ASAP-validation"
        ):
            asap_validation_rows += 1
        if row["source"].startswith("ASAP") and (
            row["dataset_split"].lower() == "test"
            or row["source"] == "ASAP-test"
        ):
            asap_test_rows += 1
    if train_source_counts != {"MAESTRO-clean": 1170, "ASAP-train": 892}:
        raise RuntimeError(f"unexpected train sources: {train_source_counts}")
    if asap_validation_rows or asap_test_rows:
        raise RuntimeError("ASAP validation/test leaked into train manifest")

    train_index = csv_rows(cache_root / stats["splits"]["train"]["index_file"])
    validation_index = csv_rows(
        cache_root / stats["splits"]["validation"]["index_file"]
    )
    train_index_leaks = sum(
        row["source"].startswith("ASAP")
        and (
            row["dataset_split"].lower() in {"validation", "test"}
            or row["source"] in {"ASAP-validation", "ASAP-test"}
        )
        for row in train_index
    )
    validation_index_test_rows = sum(
        row["source"].startswith("ASAP")
        and (row["dataset_split"].lower() == "test" or row["source"] == "ASAP-test")
        for row in validation_index
    )
    if train_index_leaks or validation_index_test_rows:
        raise RuntimeError("ASAP held-out rows leaked into cache indices")
    if not all(
        row["source"] == "ASAP-validation"
        and row["dataset_split"].lower() == "validation"
        for row in validation_index
    ):
        raise RuntimeError("validation cache contains a non-validation row")

    hashes = {}
    for name, expected in config["expected_cache_sha256"].items():
        path = cache_root / name
        digest = sha256_file(path)
        hashes[name] = digest
        if digest != expected:
            raise RuntimeError(f"immutable cache hash mismatch: {path}")

    boundaries = np.asarray([[25, 26, 63, 64], [103, 104, 127, 0]], dtype=np.int64)
    boundary_classes = classify_cc64(boundaries)
    expected_boundary_classes = np.asarray(
        [[0, 1, 1, 2], [2, 3, 3, 0]], dtype=np.int64
    )
    if not np.array_equal(boundary_classes, expected_boundary_classes):
        raise RuntimeError("canonical boundary edge test failed")

    target_ranges: dict[str, list[int]] = {}
    for split in ("train", "validation"):
        token_path = cache_root / stats["splits"][split]["tokens_file"]
        tokens = np.load(token_path, mmap_mode="r")
        minimum = NUM_CLASSES
        maximum = -1
        for start in range(0, len(tokens), 250_000):
            raw = np.asarray(tokens[start : start + 250_000, 4:], dtype=np.int64)
            raw -= PEDAL_TOKEN_OFFSET
            classes = classify_cc64(raw)
            minimum = min(minimum, int(classes.min()))
            maximum = max(maximum, int(classes.max()))
        if (minimum, maximum) != (0, 3):
            raise RuntimeError(f"{split} target range mismatch: {(minimum, maximum)}")
        target_ranges[split] = [minimum, maximum]

    checkpoint = Path(config["checkpoint_path"]) / "model.safetensors"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if sha256_file(checkpoint) != config["expected_pretrained_sha256"]:
        raise RuntimeError("pretrained checkpoint hash mismatch")
    return {
        "cache_id": stats["cache_id"],
        "train_performances": 2062,
        "train_source_counts": train_source_counts,
        "validation_performances": 71,
        "train_manifest_asap_validation_rows": asap_validation_rows,
        "train_manifest_asap_test_rows": asap_test_rows,
        "train_cache_heldout_rows": train_index_leaks,
        "validation_cache_test_rows": validation_index_test_rows,
        "target_class_ranges": target_ranges,
        "boundary_test": {
            "25": 0,
            "26": 1,
            "63": 1,
            "64": 2,
            "103": 2,
            "104": 3,
            "127": 3,
        },
        "cache_sha256": hashes,
        "asap_test_access_count": 0,
    }


def forward_model(
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    *,
    amp_enabled: bool,
) -> Any:
    use_amp = bool(amp_enabled and batch["input_ids"].device.type == "cuda")
    with torch.amp.autocast(
        device_type=batch["input_ids"].device.type,
        dtype=torch.float16 if use_amp else torch.bfloat16,
        enabled=use_amp,
    ):
        return model(
            input_ids=batch["input_ids"],
            token_attention_mask=batch["token_attention_mask"],
            note_mask=batch["note_mask"],
            pedal_targets=batch["pedal_targets"],
        )


def batch_counts(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, int]:
    if tuple(logits.shape) != (*targets.shape, NUM_CLASSES):
        raise ValueError("four-class output shape mismatch")
    valid = targets != IGNORE_INDEX
    valid_notes = valid.all(dim=-1)
    prediction = logits.argmax(dim=-1)
    return {
        "targets": int(valid.sum().item()),
        "correct": int(((prediction == targets) & valid).sum().item()),
        "notes": int(valid_notes.sum().item()),
        "exact": int(
            ((((prediction == targets) | ~valid).all(dim=-1)) & valid_notes)
            .sum()
            .item()
        ),
    }


def loader_groups(loader: Any, size: int):
    group = []
    for batch in loader:
        group.append(batch)
        if len(group) == size:
            yield group
            group = []
    if group:
        yield group


def _nonfinite_model_parameter_names(model: torch.nn.Module) -> list[str]:
    """Return corrupt parameter names without copying tensors off device."""

    return [
        name
        for name, parameter in model.named_parameters()
        if not bool(torch.isfinite(parameter.detach()).all())
    ]


def _nonfinite_gradient_diagnostics(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> dict[str, Any]:
    """Collect parameter-level diagnostics only after an overflow is detected."""

    parameter_names = {id(parameter): name for name, parameter in model.named_parameters()}
    group_by_parameter = {
        id(parameter): str(group.get("group_name", f"group_{group_index}"))
        for group_index, group in enumerate(optimizer.param_groups)
        for parameter in group["params"]
    }
    offending_names: list[str] = []
    offending_groups: set[str] = set()
    for parameter in model.parameters():
        gradient = parameter.grad
        if gradient is None or bool(torch.isfinite(gradient.detach()).all()):
            continue
        offending_names.append(parameter_names.get(id(parameter), "<unnamed>"))
        offending_groups.add(group_by_parameter.get(id(parameter), "<unknown>"))
    return {
        "offending_parameter_count": len(offending_names),
        "offending_parameter_names": offending_names,
        "offending_parameter_groups": sorted(offending_groups),
    }


def run_epoch(
    model: torch.nn.Module,
    loader: Any,
    *,
    device: torch.device,
    amp_enabled: bool,
    accumulation_steps: int = 1,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    max_grad_norm: float = 1.0,
    step_callback: Any = None,
    max_consecutive_amp_overflows: int | None = None,
    initial_consecutive_amp_overflows: int = 0,
) -> tuple[dict[str, float], dict[str, Any]]:
    training = optimizer is not None
    if training and scaler is None:
        raise ValueError("training requires a GradScaler")
    if max_consecutive_amp_overflows is not None and max_consecutive_amp_overflows < 1:
        raise ValueError("max_consecutive_amp_overflows must be positive")
    if initial_consecutive_amp_overflows < 0:
        raise ValueError("initial_consecutive_amp_overflows must be non-negative")
    model.train(training)
    numerator_total = 0.0
    denominator_total = 0
    correct = exact = notes = targets_count = 0
    optimizer_steps = amp_skips = amp_overflow_events = 0
    consecutive_amp_overflows = int(initial_consecutive_amp_overflows)
    maximum_consecutive_amp_overflows = consecutive_amp_overflows
    gradient_finite = True
    if training:
        corrupt_parameters = _nonfinite_model_parameter_names(model)
        if corrupt_parameters:
            raise FloatingPointError(
                "non-finite model parameter(s): " + ", ".join(corrupt_parameters[:8])
            )
    for attempted_step, cpu_group in enumerate(
        loader_groups(loader, int(accumulation_steps)), start=1
    ):
        if training:
            optimizer.zero_grad(set_to_none=True)
        group_denominator = sum(
            int((batch["pedal_targets"] != IGNORE_INDEX).sum().item())
            for batch in cpu_group
        )
        group_numerator = 0.0
        micro_batch_losses: list[float] = []
        group_metrics = {"correct": 0, "exact": 0, "notes": 0, "targets": 0}
        for cpu_batch in cpu_group:
            batch = move_batch(cpu_batch, device)
            with torch.set_grad_enabled(training):
                output = forward_model(model, batch, amp_enabled=amp_enabled)
            if (
                output.loss is None
                or output.loss_numerator is None
                or not bool(torch.isfinite(output.loss))
                or not bool(torch.isfinite(output.loss_numerator))
            ):
                raise FloatingPointError("four-class CE is missing or non-finite")
            if training:
                scaled_loss = output.loss_numerator / float(group_denominator)
                if amp_enabled:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
            value = float(output.loss_numerator.detach().float().item())
            group_numerator += value
            micro_batch_losses.append(float(output.loss.detach().float().item()))
            counts = batch_counts(output.logits.detach(), batch["pedal_targets"])
            for key, count in counts.items():
                group_metrics[key] += count
        optimizer_step_applied = False
        grad_norm_value = math.nan
        scale_before = scale_after = None
        overflow_diagnostics: dict[str, Any] = {
            "offending_parameter_count": 0,
            "offending_parameter_names": [],
            "offending_parameter_groups": [],
        }
        amp_overflow_event = False
        if training:
            if amp_enabled:
                scaler.unscale_(optimizer)
            parameters = [p for p in model.parameters() if p.requires_grad]
            gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
            grad_norm = torch.nn.utils.get_total_norm(gradients, norm_type=2.0)
            grad_norm_value = float(grad_norm.detach().float().item())
            current_finite = math.isfinite(grad_norm_value)
            gradient_finite &= current_finite
            if current_finite:
                torch.nn.utils.clip_grads_with_norm_(
                    parameters, max_norm=float(max_grad_norm), total_norm=grad_norm
                )
            else:
                overflow_diagnostics = _nonfinite_gradient_diagnostics(model, optimizer)
            if amp_enabled:
                scale_before = float(scaler.get_scale())
                if not current_finite and not overflow_diagnostics[
                    "offending_parameter_count"
                ]:
                    raise FloatingPointError(
                        "non-finite aggregate gradient norm without a non-finite "
                        "gradient tensor"
                    )
                scaler.step(optimizer)
                scaler.update()
                scale_after = float(scaler.get_scale())
                optimizer_step_applied = current_finite and scale_after >= scale_before
                amp_overflow_event = not current_finite
                if scale_after < scale_before:
                    amp_skips += 1
            else:
                if not current_finite:
                    raise FloatingPointError("non-finite gradient without AMP")
                optimizer.step()
                optimizer_step_applied = True
            if optimizer_step_applied:
                optimizer_steps += 1
                consecutive_amp_overflows = 0
            elif amp_overflow_event:
                amp_overflow_events += 1
                consecutive_amp_overflows += 1
                maximum_consecutive_amp_overflows = max(
                    maximum_consecutive_amp_overflows, consecutive_amp_overflows
                )
                corrupt_parameters = _nonfinite_model_parameter_names(model)
                if corrupt_parameters:
                    raise FloatingPointError(
                        "non-finite model parameter(s) after AMP overflow: "
                        + ", ".join(corrupt_parameters[:8])
                    )
        numerator_total += group_numerator
        denominator_total += group_denominator
        correct += group_metrics["correct"]
        exact += group_metrics["exact"]
        notes += group_metrics["notes"]
        targets_count += group_metrics["targets"]
        if step_callback is not None:
            step_callback(
                attempted_step,
                {
                    "loss": group_numerator / group_denominator,
                    "gradient_norm": grad_norm_value,
                    "optimizer_step_applied": optimizer_step_applied,
                    "optimizer_steps": optimizer_steps,
                    "amp_skipped_steps": amp_skips,
                    "optimizer_attempts": attempted_step,
                    "amp_overflow_event": amp_overflow_event,
                    "amp_overflow_events": amp_overflow_events,
                    "consecutive_amp_overflows": consecutive_amp_overflows,
                    "max_consecutive_amp_overflows": maximum_consecutive_amp_overflows,
                    "scale_before": scale_before,
                    "scale_after": scale_after,
                    "micro_batch_losses": micro_batch_losses,
                    **overflow_diagnostics,
                },
            )
        if (
            amp_overflow_event
            and max_consecutive_amp_overflows is not None
            and consecutive_amp_overflows >= max_consecutive_amp_overflows
        ):
            raise FloatingPointError(
                f"persistent AMP gradient overflow: {consecutive_amp_overflows} "
                "consecutive optimizer attempts"
            )
    if not denominator_total or targets_count != denominator_total or not notes:
        raise RuntimeError("empty or inconsistent epoch accounting")
    metrics = {
        "ce": numerator_total / denominator_total,
        "token_accuracy": correct / targets_count,
        "exact_note_accuracy": exact / notes,
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise FloatingPointError("non-finite epoch metric")
    return metrics, {
        "optimizer_steps": optimizer_steps,
        "optimizer_attempts": attempted_step,
        "amp_skipped_steps": amp_skips,
        "amp_overflow_events": amp_overflow_events,
        "consecutive_amp_overflows": consecutive_amp_overflows,
        "max_consecutive_amp_overflows": maximum_consecutive_amp_overflows,
        "gradients_finite": gradient_finite,
    }


def write_metrics(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def checkpoint_payload(
    model: torch.nn.Module,
    run_config: Mapping[str, Any],
    row: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "model_state": model.state_dict(),
        "configuration": dict(run_config),
        "epoch": int(row["epoch"]),
        "validation_ce": float(row["validation_ce"]),
        "checkpoint_selection": "minimum_validation_ce",
    }


def train_architecture(config: Mapping[str, Any], architecture: str) -> int:
    output_root = Path(config["output_root"])
    output_dir = output_root / OUTPUT_NAMES[architecture]
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite run output: {output_dir}")
    output_dir.mkdir(parents=True)
    log_handle = (output_dir / "train.log").open("w", encoding="utf-8")

    def log(message: str) -> None:
        print(message, flush=True)
        print(message, file=log_handle, flush=True)

    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("training requires exactly one visible CUDA GPU")
        device = torch.device("cuda:0")
        gpu = get_gpu_identity()
        if gpu["uuid"] != config["expected_gpu_uuid"]:
            raise RuntimeError(
                f"assigned GPU UUID mismatch: {gpu['uuid']}"
            )
        audit = preflight(config)
        log(
            f"CACHE_LOAD_PASS cache_id={audit['cache_id']} "
            f"train=1170+892 validation=71 test_access=0"
        )

        set_deterministic_seed(int(config["seed"]))
        train_dataset = SharedFourClassWindowDataset(config["cache_root"], "train")
        validation_dataset = SharedFourClassWindowDataset(
            config["cache_root"], "validation"
        )
        model_class = MODEL_CLASSES[architecture]
        if architecture == "encoder_only":
            model = model_class.from_pretrained(
                config["checkpoint_path"],
                torch_dtype=torch.float32,
                attn_implementation="eager",
            )
        else:
            model = model_class.from_pretrained_encoder(
                config["checkpoint_path"],
                decoder_init_seed=int(config["decoder_init_seed"]),
                freeze_encoder=False,
                torch_dtype=torch.float32,
                attn_implementation="eager",
            )
        if len(model.encoder.layers) != 10 or model.hidden_size != 768:
            raise RuntimeError("pretrained PT encoder architecture mismatch")
        if architecture == "encoder_decoder":
            if len(model.decoder.layers) != 2:
                raise RuntimeError("causal decoder must have two layers")
            if model.output_head.out_features != NUM_CLASSES:
                raise RuntimeError("encoder-decoder output vocabulary is not four")
            if model.decoder.embed_tokens.num_embeddings != NUM_CLASSES + 2:
                raise RuntimeError("encoder-decoder input vocabulary is not 4+BOS+PAD")
        else:
            if any(head.out_features != NUM_CLASSES for head in model.classification_heads):
                raise RuntimeError("encoder-only output dimension is not four")

        encoder_hash = parameter_hash(model.encoder.parameters())
        if encoder_hash != config["expected_initial_encoder_sha256"]:
            raise RuntimeError("pretrained encoder parameter hash mismatch")
        run_config = dict(config)
        run_config.update(
            architecture_name=architecture,
            architecture=(
                "PT pretrained 10-layer encoder + four independent Linear(768,4) heads"
                if architecture == "encoder_only"
                else "PT pretrained 10-layer encoder + fresh 2-layer causal decoder + shared Linear(768,4)"
            ),
            class_names=list(CLASS_NAMES),
            class_boundaries=[list(item) for item in CLASS_BOUNDS],
            representatives=list(REPRESENTATIVES),
            representative_usage="MIDI decoding and diagnostics only; not training targets",
            output_classes=NUM_CLASSES,
            output_shape="B,N,4,4",
            decoder_vocabulary=(
                {"classes": [0, 1, 2, 3], "BOS": 4, "PAD": 5}
                if architecture == "encoder_decoder"
                else None
            ),
            loss_configuration={
                "name": "standard_unweighted_cross_entropy",
                "class_weights": None,
                "label_smoothing": 0.0,
                "ignore_index": IGNORE_INDEX,
            },
            model_selection="minimum validation CE",
            validation_mode=(
                "teacher_forced CE for checkpoint selection"
                if architecture == "encoder_decoder"
                else "encoder-only CE for checkpoint selection"
            ),
            teacher_forcing=architecture == "encoder_decoder",
            scheduled_sampling=False,
            argmax_decoding_for_future_evaluation=True,
            micro_batch_size=int(config["micro_batch_size"][architecture]),
            gradient_accumulation_steps=int(
                config["gradient_accumulation_steps"][architecture]
            ),
            initial_encoder_parameter_sha256=encoder_hash,
            cache_audit=audit,
            gpu=gpu,
            source_reuse={
                "encoder_only_5class": "src/stage2_encoder_only/five_class.py",
                "encoder_decoder_5class": "src/stage2_encoder_decoder/model.py",
                "combined_cache": "src/stage2_binary/full_training.py::SharedBinaryWindowDataset",
            },
        )
        atomic_json(output_dir / "config.json", run_config)
        log(
            f"MODEL_INITIALIZED architecture={architecture} "
            f"encoder_sha256={encoder_hash} output_classes=4"
        )

        model.to(device)
        optimizer = build_binary_optimizer(
            model,
            encoder_lr=float(config["encoder_lr"]),
            head_lr=float(config["head_lr"]),
            weight_decay=float(config["weight_decay"]),
        )
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=bool(config["amp_enabled"]),
            init_scale=float(config["amp_init_scale"]),
        )
        validation_loader = make_four_class_loader(
            validation_dataset,
            batch_size=int(config["micro_batch_size"][architecture]),
            pin_memory=bool(config["pin_memory"]),
        )
        rows: list[dict[str, Any]] = []
        best_ce = math.inf
        early_stopping_best_ce = math.inf
        best_epoch = 0
        stale_epochs = 0
        first_step_logged = False

        for epoch in range(1, int(config["max_epochs"]) + 1):
            started = time.perf_counter()
            order = training_window_order(len(train_dataset), int(config["seed"]), epoch)
            train_loader = make_four_class_loader(
                train_dataset,
                batch_size=int(config["micro_batch_size"][architecture]),
                pin_memory=bool(config["pin_memory"]),
                order=order,
            )

            def callback(step: int, details: Mapping[str, Any]) -> None:
                nonlocal first_step_logged
                if not first_step_logged and details["optimizer_step_applied"]:
                    first_step_logged = True
                    loss = float(details["loss"])
                    if not math.isfinite(loss):
                        raise FloatingPointError("initial loss is non-finite")
                    update_arch_status(
                        output_root,
                        architecture,
                        "running",
                        state="first_optimizer_step_pass",
                        first_optimizer_step=True,
                        first_step_epoch=epoch,
                        first_step=step,
                        first_step_loss=loss,
                        first_step_gradient_norm=float(details["gradient_norm"]),
                        gpu=gpu,
                        cache_id=audit["cache_id"],
                    )
                    log(
                        "FIRST_OPTIMIZER_STEP_PASS "
                        f"architecture={architecture} epoch={epoch} step={step} "
                        f"loss={loss:.9f} grad_norm={float(details['gradient_norm']):.6f}"
                    )
                elif step % int(config["progress_interval"]) == 0:
                    update_arch_status(
                        output_root,
                        architecture,
                        "running",
                        state="training",
                        current_epoch=epoch,
                        current_step=step,
                    )
                    log(
                        f"TRAIN_PROGRESS architecture={architecture} epoch={epoch} "
                        f"step={step} loss={float(details['loss']):.9f}"
                    )

            train_metrics, checks = run_epoch(
                model,
                train_loader,
                device=device,
                amp_enabled=bool(config["amp_enabled"]),
                accumulation_steps=int(
                    config["gradient_accumulation_steps"][architecture]
                ),
                optimizer=optimizer,
                scaler=scaler,
                max_grad_norm=float(config["max_grad_norm"]),
                step_callback=callback,
            )
            if not first_step_logged:
                raise RuntimeError("finite first optimizer step was not verified")
            validation_metrics, _ = run_epoch(
                model,
                validation_loader,
                device=device,
                amp_enabled=bool(config["amp_enabled"]),
            )
            row = {
                "epoch": epoch,
                "train_ce": train_metrics["ce"],
                "train_token_accuracy": train_metrics["token_accuracy"],
                "train_exact_note_accuracy": train_metrics["exact_note_accuracy"],
                "validation_ce": validation_metrics["ce"],
                "validation_token_accuracy": validation_metrics["token_accuracy"],
                "validation_exact_note_accuracy": validation_metrics[
                    "exact_note_accuracy"
                ],
                "optimizer_steps": checks["optimizer_steps"],
                "amp_skipped_steps": checks["amp_skipped_steps"],
                "epoch_seconds": time.perf_counter() - started,
            }
            rows.append(row)
            write_metrics(output_dir / "metrics.csv", rows)
            candidate_ce = float(row["validation_ce"])
            if candidate_ce < best_ce:
                best_ce = candidate_ce
                best_epoch = epoch
                atomic_torch_save(
                    checkpoint_payload(model, run_config, row),
                    output_dir / "best.pt",
                )
            if candidate_ce < early_stopping_best_ce - float(
                config["early_stopping_min_delta"]
            ):
                early_stopping_best_ce = candidate_ce
                stale_epochs = 0
            else:
                stale_epochs += 1
            atomic_torch_save(
                checkpoint_payload(model, run_config, row),
                output_dir / "last.pt",
            )
            update_arch_status(
                output_root,
                architecture,
                "running",
                state="epoch_completed",
                current_epoch=epoch,
                validation_ce=float(row["validation_ce"]),
                best_epoch=best_epoch,
                best_validation_ce=best_ce,
            )
            log(
                f"EPOCH_COMPLETE architecture={architecture} epoch={epoch} "
                f"train_ce={float(row['train_ce']):.9f} "
                f"validation_ce={float(row['validation_ce']):.9f} "
                f"best_epoch={best_epoch}"
            )
            if stale_epochs >= int(config["early_stopping_patience"]):
                log(f"EARLY_STOP architecture={architecture} epoch={epoch}")
                break
        if not (output_dir / "best.pt").is_file() or not (output_dir / "last.pt").is_file():
            raise RuntimeError("required checkpoints were not written")
        log(
            f"TRAINING_COMPLETE architecture={architecture} "
            f"best_epoch={best_epoch} best_validation_ce={best_ce:.9f}"
        )
        return 0
    except Exception:
        log("TRAINING_FAILED")
        traceback.print_exc()
        traceback.print_exc(file=log_handle)
        return 1
    finally:
        log_handle.close()


def sequential(config: Mapping[str, Any], config_path: Path) -> int:
    output_root = Path(config["output_root"])
    status_path = output_root / "run_status.json"
    if status_path.exists():
        raise FileExistsError(f"refusing duplicate canonical run: {status_path}")
    if any((output_root / name).exists() for name in OUTPUT_NAMES.values()):
        raise FileExistsError(f"refusing duplicate architecture outputs: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_json(
        status_path,
        {
            "experiment_id": config["experiment_id"],
            "encoder_only": {"status": "pending"},
            "encoder_decoder": {"status": "pending"},
            "started_at": now(),
            "updated_at": now(),
            "sequential": True,
            "single_gpu": True,
        },
    )
    log_path = output_root / "sequential_train.log"
    with log_path.open("w", encoding="utf-8") as sequential_log:
        for architecture in ARCHITECTURES:
            update_arch_status(
                output_root,
                architecture,
                "running",
                state="starting",
                started_at=now(),
            )
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--config",
                str(config_path.resolve()),
                "--architecture",
                architecture,
            ]
            message = f"SEQUENTIAL_START architecture={architecture}"
            print(message, flush=True)
            print(message, file=sequential_log, flush=True)
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                print(line, end="", file=sequential_log, flush=True)
            return_code = process.wait()
            if return_code != 0:
                update_arch_status(
                    output_root,
                    architecture,
                    "failed",
                    state="process_failed",
                    return_code=return_code,
                    finished_at=now(),
                )
                message = (
                    f"SEQUENTIAL_STOP architecture={architecture} "
                    f"return_code={return_code}"
                )
                print(message, flush=True)
                print(message, file=sequential_log, flush=True)
                return return_code
            update_arch_status(
                output_root,
                architecture,
                "completed",
                state="completed",
                return_code=0,
                finished_at=now(),
            )
            message = f"SEQUENTIAL_COMPLETE architecture={architecture}"
            print(message, flush=True)
            print(message, file=sequential_log, flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--architecture", choices=ARCHITECTURES)
    mode.add_argument("--sequential", action="store_true")
    arguments = parser.parse_args()
    config = read_json(arguments.config)
    try:
        if arguments.sequential:
            return sequential(config, arguments.config)
        return train_architecture(config, str(arguments.architecture))
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
