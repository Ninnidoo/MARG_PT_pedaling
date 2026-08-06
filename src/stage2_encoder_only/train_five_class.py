"""Controlled training utilities for the fixed endpoint-aware five-class head.

This module deliberately keeps the original encoder-only CE experiment's data,
optimizer, AMP, windowing, and early-stopping controls.  Only Pedal1--4 target
mapping and the four output dimensions change from 128 to five classes.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .dataset import (
    MASK_ID,
    NON_PEDAL_FEATURES,
    PEDAL_SLOTS,
    PEDAL_TOKEN_OFFSET,
    Stage2PedalDataset,
)
from .five_class import (
    CLASS_BOUNDS,
    CLASS_NAMES,
    NUM_CLASSES,
    REPRESENTATIVES,
    FiveClassPedalEncoderModel,
    classify_pedal_values,
    decode_five_classes,
    five_class_collate_fn,
)
from .train import (
    EarlyStopping,
    EpochMetricAccumulator,
    GracefulStop,
    METRIC_NAMES,
    PINNED_PT_COMMIT,
    atomic_torch_save,
    build_best_checkpoint,
    build_last_checkpoint,
    deterministic_train_order,
    get_gpu_identity,
    restore_training_state,
)
from .training import (
    build_optimizer,
    create_grad_scaler,
    evaluation_step,
    move_batch_to_device,
    set_deterministic_seed,
    train_step,
)


DEFAULT_ASAP_ROOT = "/workspace/public/ASAP/asap-dataset-v1.1"
DEFAULT_SPLIT_CSV = (
    "/workspace/project/analysis/stage2_encoder_only_v0/asap_split.csv"
)
DEFAULT_CHECKPOINT = "/workspace/project/checkpoints/pianist_transformer"
EXPECTED_GPU_UUID = "GPU-6982dbee-fbaf-f359-d7ef-a22d0e83400b"
PEDAL_RICH_PERFORMANCE = "Schubert/Piano_Sonatas/664-3/Lin07.mid"
PEDAL_RICH_WINDOW_STARTS = (0, 256, 512, 768)
REPRESENTATIVE_POLICY = "canonical_interval_midpoint"
LOSS_CONFIGURATION = {
    "name": "unweighted_cross_entropy",
    "classes": NUM_CLASSES,
    "ignore_index": -100,
    "class_weighting": None,
    "label_smoothing": 0.0,
}

Callback = Callable[[dict[str, Any]], None]


def parameter_hash(module: torch.nn.Module) -> str:
    """Return a stable SHA-256 over a module's named parameter values."""

    digest = hashlib.sha256()
    for name, parameter in module.named_parameters():
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def default_configuration(output_dir: str | Path) -> dict[str, Any]:
    """Return the pre-registered five-class controlled-training settings."""

    return {
        "asap_root": DEFAULT_ASAP_ROOT,
        "split_csv": DEFAULT_SPLIT_CSV,
        "checkpoint_path": DEFAULT_CHECKPOINT,
        "output_dir": str(output_dir),
        "resume": None,
        "batch_size": 16,
        "max_epochs": 20,
        "seed": 20260710,
        "early_stopping_patience": 4,
        "early_stopping_min_delta": 1e-4,
        "encoder_lr": 1e-5,
        "head_lr": 1e-4,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "amp_init_scale": 1024.0,
        "amp_enabled": True,
        "freeze_encoder": False,
        "pin_memory": True,
        "num_workers": 0,
        "window_notes": 512,
        "stride_notes": 256,
        "progress_interval": 100,
        "expected_gpu_uuid": EXPECTED_GPU_UUID,
        "expected_train_performances": 892,
        "expected_validation_performances": 71,
        "expected_train_pieces": 180,
        "expected_validation_pieces": 19,
        "class_names": list(CLASS_NAMES),
        "class_boundaries": [list(bounds) for bounds in CLASS_BOUNDS],
        "representatives": REPRESENTATIVES.tolist(),
        "representative_policy": REPRESENTATIVE_POLICY,
        "loss_configuration": dict(LOSS_CONFIGURATION),
        "architecture": (
            "official pretrained PianoT5GemmaEncoder, unfrozen, hidden_size=768, "
            "four independent Linear(768,5) heads"
        ),
        "scheduler": None,
        "pianist_transformer_commit": PINNED_PT_COMMIT,
        "pipeline_splits": ["train", "validation"],
        "test_split_passed_to_pipeline": False,
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_target(
    output_dir_or_path: str | Path, filename: str
) -> tuple[Path, Path]:
    requested = Path(output_dir_or_path)
    if requested.suffix.lower() == ".json":
        target = requested
        root = requested.parent.parent if requested.parent.name == "tests" else requested.parent
    else:
        root = requested
        target = root / "tests" / filename
    return root.resolve(), target.resolve()


def _configuration_for_root(root: Path) -> dict[str, Any]:
    configuration = default_configuration(root)
    config_path = root / "config.json"
    if config_path.is_file():
        saved = json.loads(config_path.read_text(encoding="utf-8"))
        configuration.update(saved)
    return configuration


def _raw_targets(tokens: np.ndarray) -> np.ndarray:
    targets = (
        np.asarray(tokens)[:, NON_PEDAL_FEATURES:].astype(np.int64)
        - PEDAL_TOKEN_OFFSET
    )
    if targets.ndim != 2 or targets.shape[1] != PEDAL_SLOTS:
        raise ValueError("tokenized targets must have shape [notes,4]")
    if targets.size and (targets.min() < 0 or targets.max() > 127):
        raise ValueError("tokenized pedal target outside [0,127]")
    return targets


def _weighted_quantile_from_histogram(histogram: np.ndarray, quantile: float) -> float:
    total = int(histogram.sum())
    if total <= 0:
        raise ValueError("quantile requires a non-empty histogram")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0,1]")
    cumulative = np.cumsum(histogram)

    def order_statistic(index: int) -> float:
        return float(np.searchsorted(cumulative, index + 1, side="left"))

    position = quantile * (total - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    low_value = order_statistic(lower)
    high_value = order_statistic(upper)
    return low_value + (high_value - low_value) * (position - lower)


def canonical_train_oracle(output_dir_or_path: str | Path) -> dict[str, Any]:
    """Compute and atomically save the canonical oracle from train MIDI only."""

    root, target = _artifact_target(
        output_dir_or_path, "canonical_oracle_train.json"
    )
    configuration = _configuration_for_root(root)
    dataset = Stage2PedalDataset(
        configuration["asap_root"],
        configuration["split_csv"],
        "train",
        window_notes=int(configuration.get("window_notes", 512)),
        stride_notes=int(configuration.get("stride_notes", 256)),
        cache_mode="preload",
    )
    if any(row.get("split") != "train" for row in dataset.performances):
        raise RuntimeError("non-train row reached the canonical train oracle")

    class_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    error_histogram = np.zeros(128, dtype=np.int64)
    slot_error_sums = np.zeros(PEDAL_SLOTS, dtype=np.float64)
    slot_counts = np.zeros(PEDAL_SLOTS, dtype=np.int64)
    tolerance_counts = {5: 0, 10: 0, 20: 0}
    region_error_sums = {"on": 0.0, "subthreshold": 0.0, "zero": 0.0}
    region_counts = {"on": 0, "subthreshold": 0, "zero": 0}
    performance_maes: list[float] = []
    note_count = 0

    for row in dataset.performances:
        raw = _raw_targets(dataset._token_cache[row["performance_path"]])
        classes = classify_pedal_values(raw)
        decoded = decode_five_classes(classes)
        errors = np.abs(decoded.astype(np.int64) - raw)
        note_count += len(raw)
        class_counts += np.bincount(classes.reshape(-1), minlength=NUM_CLASSES)
        error_histogram += np.bincount(errors.reshape(-1), minlength=128)
        slot_error_sums += errors.sum(axis=0)
        slot_counts += len(raw)
        for tolerance in tolerance_counts:
            tolerance_counts[tolerance] += int((errors <= tolerance).sum())
        masks = {
            "on": raw >= 64,
            "subthreshold": (raw >= 1) & (raw <= 63),
            "zero": raw == 0,
        }
        for name, mask in masks.items():
            region_error_sums[name] += float(errors[mask].sum())
            region_counts[name] += int(mask.sum())
        performance_maes.append(float(errors.mean()))

    target_count = int(class_counts.sum())
    if target_count != note_count * PEDAL_SLOTS:
        raise RuntimeError("canonical train oracle target count mismatch")
    if not target_count:
        raise RuntimeError("canonical train oracle is empty")
    quantiles = {
        name: _weighted_quantile_from_histogram(error_histogram, quantile)
        for name, quantile in (
            ("q01", 0.01), ("q05", 0.05), ("q25", 0.25),
            ("median", 0.50), ("q75", 0.75), ("q95", 0.95),
            ("q99", 0.99),
        )
    }
    payload: dict[str, Any] = {
        "source_split": "train",
        "test_split_accessed": False,
        "representative_policy": REPRESENTATIVE_POLICY,
        "class_names": list(CLASS_NAMES),
        "class_boundaries": [list(bounds) for bounds in CLASS_BOUNDS],
        "representatives": REPRESENTATIVES.tolist(),
        "performance_count": dataset.performance_count,
        "piece_count": len({row["piece_id"] for row in dataset.performances}),
        "note_count": note_count,
        "target_count": target_count,
        "class_counts": class_counts.tolist(),
        "class_distribution": (class_counts / target_count).tolist(),
        "all_five_classes_present": bool(np.all(class_counts > 0)),
        "overall_mae": float(
            np.dot(np.arange(len(error_histogram)), error_histogram) / target_count
        ),
        "performance_macro_mae": float(np.mean(performance_maes)),
        "median_performance_mae": float(np.median(performance_maes)),
        "on_region_mae": float(region_error_sums["on"] / region_counts["on"]),
        "subthreshold_region_mae": float(
            region_error_sums["subthreshold"] / region_counts["subthreshold"]
        ),
        "zero_target_mae": float(
            region_error_sums["zero"] / region_counts["zero"]
        ),
        "per_slot_mae": (slot_error_sums / slot_counts).tolist(),
        "tolerance_accuracy": {
            f"plus_minus_{tolerance}": float(count / target_count)
            for tolerance, count in tolerance_counts.items()
        },
        "maximum_absolute_error": int(np.flatnonzero(error_histogram)[-1]),
        "error_quantiles": quantiles,
        "split_csv_sha256": _sha256_file(Path(configuration["split_csv"])),
    }
    _atomic_json(target, payload)
    return payload


def _make_window_sample(
    full_tokens: np.ndarray, start: int, end: int
) -> dict[str, Any]:
    original = np.asarray(full_tokens)[start:end]
    if len(original) != end - start or not len(original):
        raise ValueError("overfit window lies outside the performance")
    input_tokens = original.copy()
    input_tokens[:, NON_PEDAL_FEATURES:] = MASK_ID
    return {
        "input_ids": torch.from_numpy(input_tokens.reshape(-1)).long(),
        "pedal_targets": torch.from_numpy(_raw_targets(original).copy()).long(),
        "note_mask": torch.ones(len(original), dtype=torch.bool),
        "metadata": {"window_start_note": start, "window_end_note": end},
    }


def _overfit_measure(
    model: FiveClassPedalEncoderModel,
    samples: Sequence[Mapping[str, Any]],
    device: torch.device,
    amp_enabled: bool,
    oracle_mae: float,
) -> dict[str, Any]:
    logits_chunks: list[torch.Tensor] = []
    class_chunks: list[torch.Tensor] = []
    raw_chunks: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(samples), 2):
            selected = samples[offset : offset + 2]
            cpu_batch = five_class_collate_fn(selected)
            raw = torch.stack(
                [sample["pedal_targets"] for sample in selected], dim=0
            )
            batch = move_batch_to_device(cpu_batch, device)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                enabled=amp_enabled and device.type == "cuda",
            ):
                output = model(
                    input_ids=batch["input_ids"],
                    token_attention_mask=batch["token_attention_mask"],
                    pedal_targets=None,
                    note_mask=batch["note_mask"],
                )
            logits_chunks.append(output.logits.float())
            class_chunks.append(batch["pedal_targets"])
            raw_chunks.append(raw.to(device))

    logits = torch.cat(logits_chunks, dim=0)
    targets = torch.cat(class_chunks, dim=0)
    raw_targets = torch.cat(raw_chunks, dim=0)
    valid = targets != -100
    predictions = logits.argmax(dim=-1)
    decoded = decode_five_classes(predictions)
    loss = FiveClassPedalEncoderModel.compute_loss(logits, targets)
    correct = (predictions == targets) & valid
    valid_notes = valid.any(dim=-1)
    exact = ((predictions == targets) | ~valid).all(dim=-1)
    target_counts = torch.bincount(targets[valid], minlength=NUM_CLASSES)
    predicted_counts = torch.bincount(predictions[valid], minlength=NUM_CLASSES)
    recalls: dict[str, float] = {}
    for class_id, name in enumerate(CLASS_NAMES):
        class_mask = valid & (targets == class_id)
        recalls[name] = float(
            ((predictions == class_id) & class_mask).sum().item()
            / class_mask.sum().item()
        )
    decoded_mae = float(
        (decoded[valid].float() - raw_targets[valid].float()).abs().mean().item()
    )
    return {
        "ce_loss": float(loss.item()),
        "token_accuracy": float(correct.sum().item() / valid.sum().item()),
        "exact_note_accuracy": float(
            (exact & valid_notes).sum().item() / valid_notes.sum().item()
        ),
        "representative_decoded_mae": decoded_mae,
        "quantization_oracle_mae": float(oracle_mae),
        "decoded_mae_gap": float(decoded_mae - oracle_mae),
        "per_class_recall": recalls,
        "target_class_counts": target_counts.cpu().tolist(),
        "predicted_class_counts": predicted_counts.cpu().tolist(),
        "all_five_classes_predicted": bool(torch.all(predicted_counts > 0).item()),
    }


def pedal_rich_overfit(output_dir: str | Path) -> dict[str, Any]:
    """Run and save the train-only Lin07 four-window GPU overfit gate."""

    output_root = Path(output_dir).resolve()
    artifact = output_root / "tests" / "pedal_rich_overfit.json"
    configuration = _configuration_for_root(output_root)
    gpu = get_gpu_identity()
    if gpu["uuid"] != configuration.get("expected_gpu_uuid", EXPECTED_GPU_UUID):
        raise RuntimeError(
            f"GPU UUID mismatch: expected {configuration.get('expected_gpu_uuid')}, "
            f"got {gpu['uuid']}"
        )
    set_deterministic_seed(int(configuration.get("seed", 20260710)))
    dataset = Stage2PedalDataset(
        configuration["asap_root"],
        configuration["split_csv"],
        "train",
        window_notes=512,
        stride_notes=256,
        cache_mode="preload",
    )
    if any(row.get("split") != "train" for row in dataset.performances):
        raise RuntimeError("non-train row reached the overfit gate")
    try:
        row = next(
            item
            for item in dataset.performances
            if item["performance_path"] == PEDAL_RICH_PERFORMANCE
        )
    except StopIteration as exc:
        raise RuntimeError("fixed Lin07 overfit performance is not in train") from exc
    tokens = dataset._token_cache[row["performance_path"]]
    samples = [
        _make_window_sample(tokens, start, start + 512)
        for start in PEDAL_RICH_WINDOW_STARTS
    ]
    raw_targets = torch.cat(
        [sample["pedal_targets"].reshape(-1) for sample in samples]
    ).numpy()
    raw_matrix = raw_targets.reshape(-1, PEDAL_SLOTS)
    target_classes = classify_pedal_values(raw_matrix)
    class_counts = np.bincount(target_classes.reshape(-1), minlength=NUM_CLASSES)
    if np.any(class_counts == 0):
        raise RuntimeError(
            f"fixed Lin07 four-window subset lacks a class: {class_counts.tolist()}"
        )
    oracle_decoded = decode_five_classes(target_classes)
    oracle_mae = float(np.abs(oracle_decoded - raw_matrix).mean())

    device = torch.device("cuda:0")
    set_deterministic_seed(int(configuration.get("seed", 20260710)))
    model = FiveClassPedalEncoderModel.from_pretrained(
        configuration["checkpoint_path"],
        freeze_encoder=False,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    ).to(device)
    if model.hidden_size != 768:
        raise RuntimeError(f"expected hidden size 768, got {model.hidden_size}")
    optimizer = build_optimizer(
        model, encoder_lr=1e-4, head_lr=1e-3, weight_decay=0.0
    )
    scaler = create_grad_scaler(True, "cuda", 1024.0)
    active_name, active_parameter = next(
        (name, parameter)
        for name, parameter in model.encoder.named_parameters()
        if parameter.requires_grad and "embed_tokens" not in name and parameter.ndim >= 2
    )
    encoder_before = active_parameter.detach().clone()
    heads_before = {
        name: parameter.detach().clone()
        for name, parameter in model.classification_heads.named_parameters()
    }
    torch.cuda.reset_peak_memory_stats(device)
    initial = _overfit_measure(model, samples, device, True, oracle_mae)
    encoder_gradient_ok = True
    head_gradients_ok = True
    losses_finite = math.isfinite(float(initial["ce_loss"]))
    steps = 0
    oom = False
    try:
        for step in range(1, 301):
            offset = 0 if step % 2 else 2
            cpu_batch = five_class_collate_fn(samples[offset : offset + 2])
            batch = move_batch_to_device(cpu_batch, device)
            metrics = train_step(
                model,
                batch,
                optimizer,
                scaler=scaler,
                amp_enabled=True,
                max_grad_norm=1.0,
            )
            losses_finite &= math.isfinite(float(metrics["loss"]))
            encoder_gradient_ok &= (
                active_parameter.grad is not None
                and bool(torch.isfinite(active_parameter.grad).all())
                and float(active_parameter.grad.abs().sum().item()) > 0.0
            )
            head_gradients_ok &= all(
                any(
                    parameter.grad is not None
                    and bool(torch.isfinite(parameter.grad).all())
                    and float(parameter.grad.abs().sum().item()) > 0.0
                    for parameter in head.parameters()
                )
                for head in model.classification_heads
            )
            steps = step
            if step % 5 == 0:
                current = _overfit_measure(model, samples, device, True, oracle_mae)
                print(
                    f"five-class overfit step={step} "
                    f"metrics={json.dumps(current, sort_keys=True)}",
                    flush=True,
                )
                if (
                    current["token_accuracy"] >= 0.995
                    and current["exact_note_accuracy"] >= 0.98
                    and abs(float(current["decoded_mae_gap"])) <= 0.25
                    and current["all_five_classes_predicted"]
                ):
                    break
        final = _overfit_measure(model, samples, device, True, oracle_mae)
    except torch.cuda.OutOfMemoryError:
        oom = True
        final = initial

    encoder_changed = not torch.equal(encoder_before, active_parameter.detach())
    head_groups_changed = [
        all(
            not torch.equal(
                heads_before[f"{slot}.{name}"], parameter.detach()
            )
            for name, parameter in head.named_parameters()
        )
        for slot, head in enumerate(model.classification_heads)
    ]
    all_metrics_finite = all(
        math.isfinite(float(value))
        for group in (initial, final)
        for key, value in group.items()
        if key not in {
            "per_class_recall", "target_class_counts",
            "predicted_class_counts", "all_five_classes_predicted",
        }
    ) and all(
        math.isfinite(value)
        for group in (initial, final)
        for value in group["per_class_recall"].values()
    )
    passed = bool(
        not oom
        and final["token_accuracy"] >= 0.995
        and final["exact_note_accuracy"] >= 0.98
        and abs(float(final["decoded_mae_gap"])) <= 0.25
        and final["all_five_classes_predicted"]
        and encoder_gradient_ok
        and head_gradients_ok
        and encoder_changed
        and all(head_groups_changed)
        and losses_finite
        and all_metrics_finite
    )
    result: dict[str, Any] = {
        "passed": passed,
        "source_split": "train",
        "test_split_accessed": False,
        "performance_path": PEDAL_RICH_PERFORMANCE,
        "window_starts": list(PEDAL_RICH_WINDOW_STARTS),
        "target_class_counts": class_counts.tolist(),
        "target_class_distribution": (class_counts / class_counts.sum()).tolist(),
        "all_five_classes_present": bool(np.all(class_counts > 0)),
        "subset_canonical_quantization_oracle_mae": oracle_mae,
        "initial": initial,
        "final": final,
        "steps": steps,
        "active_encoder_parameter": active_name,
        "active_encoder_changed": encoder_changed,
        "encoder_gradient_finite_nonzero": encoder_gradient_ok,
        "every_head_received_finite_nonzero_gradient": head_gradients_ok,
        "every_head_parameter_group_changed": all(head_groups_changed),
        "head_groups_changed": head_groups_changed,
        "all_losses_finite": losses_finite,
        "all_metrics_finite": all_metrics_finite,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "oom": oom,
        "gpu": gpu,
        "overfit_optimizer": {
            "name": "AdamW", "encoder_lr": 1e-4, "head_lr": 1e-3,
            "weight_decay": 0.0, "max_grad_norm": 1.0,
        },
    }
    _atomic_json(artifact, result)
    del model, optimizer, scaler, dataset
    torch.cuda.empty_cache()
    if oom:
        raise RuntimeError("OOM during five-class pedal-rich overfit gate")
    if not passed:
        raise RuntimeError(f"five-class pedal-rich overfit gate failed: {result}")
    return result


def _make_loader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    pin_memory: bool,
    order: list[int] | None = None,
) -> DataLoader:
    source = Subset(dataset, order) if order is not None else dataset
    return DataLoader(
        source,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
        collate_fn=five_class_collate_fn,
    )


def _notify(callback: Callback | None, **payload: Any) -> None:
    if callback is not None:
        callback(dict(payload))


def _run_epoch(
    model: FiveClassPedalEncoderModel,
    loader: DataLoader,
    device: torch.device,
    *,
    training: bool,
    amp_enabled: bool,
    stop: GracefulStop,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    max_grad_norm: float = 1.0,
    global_step: int = 0,
    progress_interval: int = 100,
    progress_callback: Callback | None = None,
    epoch: int,
) -> tuple[dict[str, float | int] | None, int, bool]:
    accumulator = EpochMetricAccumulator()
    for batch_index, cpu_batch in enumerate(loader, start=1):
        batch = move_batch_to_device(cpu_batch, device)
        if training:
            if optimizer is None or scaler is None:
                raise ValueError("training requires optimizer and scaler")
            metrics = train_step(
                model,
                batch,
                optimizer,
                scaler=scaler,
                amp_enabled=amp_enabled,
                max_grad_norm=max_grad_norm,
            )
            global_step += 1
        else:
            metrics = evaluation_step(model, batch, amp_enabled=amp_enabled)
        accumulator.update(metrics)
        if batch_index == 1 or batch_index % progress_interval == 0:
            update = {
                "stage": "training" if training else "validation",
                "epoch": epoch,
                "batch": batch_index,
                "batches": len(loader),
                "global_step": global_step,
                "loss": float(metrics["loss"]),
            }
            _notify(progress_callback, **update)
            print(
                f"{update['stage']} epoch={epoch} batch={batch_index}/{len(loader)} "
                f"loss={metrics['loss']:.6f} global_step={global_step}",
                flush=True,
            )
        if stop.requested:
            return None, global_step, False
    return accumulator.compute(), global_step, True


_COMPATIBILITY_KEYS = (
    "seed", "batch_size", "max_epochs", "encoder_lr", "head_lr",
    "weight_decay", "max_grad_norm", "amp_init_scale", "amp_enabled",
    "freeze_encoder", "pin_memory", "num_workers", "window_notes",
    "stride_notes", "early_stopping_patience", "early_stopping_min_delta",
    "class_names", "class_boundaries", "representatives",
    "representative_policy", "scheduler",
)


def _validate_scientific_configuration(configuration: Mapping[str, Any]) -> None:
    if list(configuration.get("class_names", [])) != list(CLASS_NAMES):
        raise ValueError("five-class names are not the fixed definition")
    if configuration.get("class_boundaries") != [list(x) for x in CLASS_BOUNDS]:
        raise ValueError("five-class boundaries are not the fixed definition")
    if list(configuration.get("representatives", [])) != REPRESENTATIVES.tolist():
        raise ValueError("representatives must be canonical [0,32,80,111,127]")
    if configuration.get("representative_policy") != REPRESENTATIVE_POLICY:
        raise ValueError("invalid representative policy")
    if configuration.get("scheduler") not in (None, "none"):
        raise ValueError("the controlled baseline must not use a scheduler")
    if int(configuration.get("num_workers", 0)) != 0:
        raise ValueError("the controlled baseline requires num_workers=0")
    if bool(configuration.get("freeze_encoder", False)):
        raise ValueError("the controlled baseline requires an unfrozen encoder")


def _prepare_training_directory(output_dir: Path, resume: Path | None) -> None:
    if resume is not None:
        if not output_dir.is_dir() or not resume.is_file():
            raise FileNotFoundError("resume output directory or checkpoint is missing")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    allowed = {
        ".run.lock", "run_status.json", "train.log", "tests", "config.json"
    }
    unexpected = {path.name for path in output_dir.iterdir()} - allowed
    if unexpected:
        raise FileExistsError(
            f"refusing to overwrite output directory entries: {sorted(unexpected)}"
        )


def _merge_configuration(
    supplied: Mapping[str, Any], output_dir: Path, resume: Path | None
) -> dict[str, Any]:
    config_path = output_dir / "config.json"
    saved = (
        json.loads(config_path.read_text(encoding="utf-8"))
        if config_path.is_file()
        else {}
    )
    if resume is not None:
        if not saved:
            raise FileNotFoundError("resume requires the original config.json")
        for key in _COMPATIBILITY_KEYS:
            if key in supplied and key in saved and supplied[key] != saved[key]:
                raise ValueError(f"incompatible resume configuration: {key}")
        merged = dict(saved)
        merged["resume"] = str(resume)
    else:
        merged = dict(saved)
        merged.update(supplied)
    defaults = default_configuration(output_dir)
    for key, value in defaults.items():
        merged.setdefault(key, value)
    merged["output_dir"] = str(output_dir)
    for key in ("asap_root", "split_csv", "checkpoint_path"):
        merged[key] = str(Path(merged[key]).resolve())
    merged["resume"] = str(resume) if resume is not None else None
    merged.update(
        class_names=list(CLASS_NAMES),
        class_boundaries=[list(bounds) for bounds in CLASS_BOUNDS],
        representatives=REPRESENTATIVES.tolist(),
        representative_policy=REPRESENTATIVE_POLICY,
        loss_configuration=dict(LOSS_CONFIGURATION),
        scheduler=None,
        pianist_transformer_commit=PINNED_PT_COMMIT,
        pipeline_splits=["train", "validation"],
        test_split_passed_to_pipeline=False,
    )
    _validate_scientific_configuration(merged)
    return merged


def _validate_counts(
    configuration: Mapping[str, Any],
    train_dataset: Stage2PedalDataset,
    validation_dataset: Stage2PedalDataset,
) -> dict[str, int]:
    if any(row.get("split") != "train" for row in train_dataset.performances):
        raise RuntimeError("non-train row reached the training dataset")
    if any(
        row.get("split") != "validation"
        for row in validation_dataset.performances
    ):
        raise RuntimeError("non-validation row reached the validation dataset")
    counts = {
        "train_performance_count": train_dataset.performance_count,
        "validation_performance_count": validation_dataset.performance_count,
        "train_piece_count": len(
            {row["piece_id"] for row in train_dataset.performances}
        ),
        "validation_piece_count": len(
            {row["piece_id"] for row in validation_dataset.performances}
        ),
    }
    expected = {
        "train_performance_count": int(configuration["expected_train_performances"]),
        "validation_performance_count": int(
            configuration["expected_validation_performances"]
        ),
        "train_piece_count": int(configuration["expected_train_pieces"]),
        "validation_piece_count": int(configuration["expected_validation_pieces"]),
    }
    if counts != expected:
        raise RuntimeError(f"piece-wise split count mismatch: {counts} != {expected}")
    return counts


def _validate_resume_checkpoint(
    checkpoint: Mapping[str, Any], configuration: Mapping[str, Any]
) -> None:
    required = {
        "model_state", "optimizer_state", "grad_scaler_state",
        "completed_epoch", "global_optimizer_step", "best_validation_loss",
        "early_stopping_counter", "python_rng_state", "numpy_rng_state",
        "torch_cpu_rng_state", "torch_cuda_rng_state",
    }
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"checkpoint is not an exact epoch boundary: {sorted(missing)}")
    if "scheduler_state" in checkpoint:
        raise ValueError("unexpected scheduler state in no-scheduler experiment")
    checkpoint_config = checkpoint.get("configuration", {})
    for key in _COMPATIBILITY_KEYS:
        if key in checkpoint_config and checkpoint_config[key] != configuration[key]:
            raise ValueError(f"checkpoint configuration mismatch: {key}")


def _validate_resume_metrics(metrics_path: Path, completed_epoch: int) -> None:
    if completed_epoch == 0:
        return
    if not metrics_path.is_file():
        raise FileNotFoundError("resume checkpoint has no metrics.csv")
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or int(rows[-1]["epoch"]) != completed_epoch:
        raise RuntimeError("metrics.csv is not aligned to the resume epoch")


def _early_stopping_already_satisfied(early_stopping: EarlyStopping) -> bool:
    """Return whether an exact-boundary resume has exhausted its patience."""

    return early_stopping.counter >= early_stopping.patience


def train(
    config: dict[str, Any],
    epoch_callback: Callback | None = None,
    progress_callback: Callback | None = None,
) -> dict[str, Any]:
    """Train the controlled five-class model and return a completion summary."""

    supplied = dict(config)
    output_dir = Path(supplied["output_dir"]).resolve()
    resume = Path(supplied["resume"]).resolve() if supplied.get("resume") else None
    _prepare_training_directory(output_dir, resume)
    configuration = _merge_configuration(supplied, output_dir, resume)
    config_path = output_dir / "config.json"
    if resume is None:
        _atomic_json(config_path, configuration)

    stop = GracefulStop()
    stop.install()
    set_deterministic_seed(int(configuration["seed"]))
    gpu = get_gpu_identity()
    if gpu["uuid"] != configuration["expected_gpu_uuid"]:
        raise RuntimeError(
            f"GPU UUID mismatch: expected {configuration['expected_gpu_uuid']}, "
            f"got {gpu['uuid']}"
        )
    configuration.update(
        gpu_uuid=gpu["uuid"],
        gpu_name=gpu["name"],
        container_visible_cuda_index=0,
        pytorch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        split_csv_sha256=_sha256_file(Path(configuration["split_csv"])),
        dataset_cache_mode="preload",
        validation_training_loss_overlap_policy=(
            "window-level CE for checkpoint selection; final metrics are reconstructed"
        ),
    )
    _atomic_json(config_path, configuration)
    print("FIVE-CLASS RUN START", json.dumps(configuration, sort_keys=True), flush=True)
    _notify(progress_callback, stage="preloading", split="train")
    preload_start = time.perf_counter()
    train_dataset = Stage2PedalDataset(
        configuration["asap_root"],
        configuration["split_csv"],
        "train",
        window_notes=int(configuration["window_notes"]),
        stride_notes=int(configuration["stride_notes"]),
        cache_mode="preload",
    )
    train_preload_seconds = time.perf_counter() - preload_start
    _notify(
        progress_callback,
        stage="preloading",
        split="train",
        completed=True,
        seconds=train_preload_seconds,
    )
    _notify(progress_callback, stage="preloading", split="validation")
    preload_start = time.perf_counter()
    validation_dataset = Stage2PedalDataset(
        configuration["asap_root"],
        configuration["split_csv"],
        "validation",
        window_notes=int(configuration["window_notes"]),
        stride_notes=int(configuration["stride_notes"]),
        cache_mode="preload",
    )
    validation_preload_seconds = time.perf_counter() - preload_start
    counts = _validate_counts(configuration, train_dataset, validation_dataset)
    configuration.update(
        **counts,
        train_window_count=train_dataset.window_count,
        validation_window_count=validation_dataset.window_count,
        train_preload_seconds=train_preload_seconds,
        validation_preload_seconds=validation_preload_seconds,
    )
    _atomic_json(config_path, configuration)
    _notify(
        progress_callback,
        stage="preloading",
        split="validation",
        completed=True,
        seconds=validation_preload_seconds,
    )
    print(
        "datasets "
        f"train_performances={train_dataset.performance_count} "
        f"train_windows={train_dataset.window_count} "
        f"validation_performances={validation_dataset.performance_count} "
        f"validation_windows={validation_dataset.window_count}",
        flush=True,
    )

    set_deterministic_seed(int(configuration["seed"]))
    model = FiveClassPedalEncoderModel.from_pretrained(
        configuration["checkpoint_path"],
        freeze_encoder=False,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    if model.hidden_size != 768 or len(model.classification_heads) != PEDAL_SLOTS:
        raise RuntimeError("five-class model architecture differs from registration")
    encoder_hash = parameter_hash(model.encoder)
    head_hashes = [parameter_hash(head) for head in model.classification_heads]
    model_hash = parameter_hash(model)
    configuration.update(
        encoder_initial_sha256=encoder_hash,
        four_head_initial_sha256=head_hashes,
        initial_parameter_sha256=model_hash,
    )
    if resume is not None:
        saved = json.loads(config_path.read_text(encoding="utf-8"))
        if saved.get("encoder_initial_sha256") != encoder_hash:
            raise RuntimeError("resume encoder initialization hash mismatch")
        if saved.get("four_head_initial_sha256") != head_hashes:
            raise RuntimeError("resume head initialization hash mismatch")
    else:
        _atomic_json(config_path, configuration)

    device = torch.device("cuda:0")
    model = model.to(device)
    optimizer = build_optimizer(
        model,
        encoder_lr=float(configuration["encoder_lr"]),
        head_lr=float(configuration["head_lr"]),
        weight_decay=float(configuration["weight_decay"]),
    )
    scaler = create_grad_scaler(
        bool(configuration["amp_enabled"]),
        "cuda",
        float(configuration["amp_init_scale"]),
    )
    early = EarlyStopping(
        int(configuration["early_stopping_patience"]),
        float(configuration["early_stopping_min_delta"]),
    )
    start_epoch, global_step = 1, 0
    if resume is not None:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        _validate_resume_checkpoint(checkpoint, configuration)
        _validate_resume_metrics(
            output_dir / "metrics.csv", int(checkpoint["completed_epoch"])
        )
        start_epoch, global_step = restore_training_state(
            checkpoint, model, optimizer, scaler, early
        )
        print(
            f"resumed exact epoch checkpoint={resume} start_epoch={start_epoch} "
            f"global_step={global_step}",
            flush=True,
        )

    validation_loader = _make_loader(
        validation_dataset,
        int(configuration["batch_size"]),
        bool(configuration["pin_memory"]),
    )
    metric_fields = [
        "epoch", "global_optimizer_step", "epoch_seconds",
        "peak_gpu_memory_bytes",
    ] + [
        f"{prefix}_{name}"
        for prefix in ("train", "validation")
        for name in METRIC_NAMES
    ]
    metrics_path = output_dir / "metrics.csv"
    append = resume is not None and metrics_path.exists()
    metrics_handle = metrics_path.open(
        "a" if append else "w", newline="", encoding="utf-8"
    )
    writer = csv.DictWriter(metrics_handle, fieldnames=metric_fields)
    if not append:
        writer.writeheader()
        metrics_handle.flush()
        os.fsync(metrics_handle.fileno())

    run_start = time.perf_counter()
    completed_epoch = start_epoch - 1
    patience_already_exhausted = bool(
        resume is not None and _early_stopping_already_satisfied(early)
    )
    stop_reason = (
        "early_stopping_already_satisfied"
        if patience_already_exhausted
        else "max_epochs"
    )
    try:
        epoch_range = (
            ()
            if patience_already_exhausted
            else range(
                start_epoch, int(configuration["max_epochs"]) + 1
            )
        )
        for epoch in epoch_range:
            torch.cuda.reset_peak_memory_stats(device)
            epoch_start = time.perf_counter()
            order = deterministic_train_order(
                len(train_dataset), int(configuration["seed"]), epoch
            )
            train_loader = _make_loader(
                train_dataset,
                int(configuration["batch_size"]),
                bool(configuration["pin_memory"]),
                order,
            )
            print(
                f"epoch {epoch} start train_batches={len(train_loader)} "
                f"validation_batches={len(validation_loader)}",
                flush=True,
            )
            train_metrics, global_step_after_train, complete = _run_epoch(
                model,
                train_loader,
                device,
                training=True,
                amp_enabled=bool(configuration["amp_enabled"]),
                stop=stop,
                optimizer=optimizer,
                scaler=scaler,
                max_grad_norm=float(configuration["max_grad_norm"]),
                global_step=global_step,
                progress_interval=int(configuration["progress_interval"]),
                progress_callback=progress_callback,
                epoch=epoch,
            )
            if not complete:
                stop_reason = "signal_mid_epoch_prior_boundary_preserved"
                break
            global_step = global_step_after_train
            validation_metrics, _, complete = _run_epoch(
                model,
                validation_loader,
                device,
                training=False,
                amp_enabled=bool(configuration["amp_enabled"]),
                stop=stop,
                global_step=global_step,
                progress_interval=int(configuration["progress_interval"]),
                progress_callback=progress_callback,
                epoch=epoch,
            )
            if not complete:
                stop_reason = "signal_mid_epoch_prior_boundary_preserved"
                break
            completed_epoch = epoch
            improved, should_stop = early.update(
                float(validation_metrics["loss"]), epoch
            )
            peak_memory = int(torch.cuda.max_memory_allocated(device))
            row: dict[str, Any] = {
                "epoch": epoch,
                "global_optimizer_step": global_step,
                "epoch_seconds": time.perf_counter() - epoch_start,
                "peak_gpu_memory_bytes": peak_memory,
            }
            for prefix, metrics in (
                ("train", train_metrics), ("validation", validation_metrics)
            ):
                for name in METRIC_NAMES:
                    row[f"{prefix}_{name}"] = metrics[name]
            writer.writerow(row)
            metrics_handle.flush()
            os.fsync(metrics_handle.fileno())
            atomic_torch_save(
                build_last_checkpoint(
                    model, optimizer, scaler, completed_epoch, global_step,
                    early, configuration,
                ),
                output_dir / "last.pt",
            )
            if improved:
                atomic_torch_save(
                    build_best_checkpoint(
                        model, configuration, early.best_epoch, early.best_loss
                    ),
                    output_dir / "best.pt",
                )
            callback_row = dict(row)
            callback_row.update(
                best_epoch=early.best_epoch,
                best_validation_loss=early.best_loss,
                early_stopping_counter=early.counter,
                elapsed_seconds=time.perf_counter() - run_start,
            )
            _notify(epoch_callback, **callback_row)
            print(
                f"epoch {epoch} complete train={train_metrics} "
                f"validation={validation_metrics} improved={improved} "
                f"early_counter={early.counter} peak_gpu_bytes={peak_memory}",
                flush=True,
            )
            if should_stop:
                stop_reason = "early_stopping"
                break
            if stop.requested:
                stop_reason = "signal_at_epoch_boundary"
                break
    finally:
        metrics_handle.flush()
        metrics_handle.close()

    if stop_reason == "signal_mid_epoch_prior_boundary_preserved":
        raise KeyboardInterrupt(
            "five-class training interrupted; prior exact epoch checkpoint preserved"
        )
    result: dict[str, Any] = {
        "completed_epoch": completed_epoch,
        "global_optimizer_step": global_step,
        "best_epoch": early.best_epoch,
        "best_validation_loss": early.best_loss,
        "runtime_seconds": time.perf_counter() - run_start,
        "stop_reason": stop_reason,
        "early_stopped": stop_reason in {"early_stopping", "early_stopping_already_satisfied"},
        "encoder_initial_sha256": encoder_hash,
        "four_head_initial_sha256": head_hashes,
        "initial_parameter_sha256": model_hash,
        **counts,
    }
    print("FIVE-CLASS RUN COMPLETE", json.dumps(result, sort_keys=True), flush=True)
    return result

