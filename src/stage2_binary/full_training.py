"""Shared cached data and full-training utilities for binary Stage 2."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from src.stage2_encoder_only.train import deterministic_train_order
from src.stage2_encoder_only.training import move_batch_to_device

from .dataset import (
    IGNORE_INDEX,
    binary_stage2_collate_fn,
    decode_joint_targets,
    make_masked_binary_sample,
)
from .validation_evaluator import (
    infer_cached_binary_pedals,
    joint16_histogram_from_ids,
    official_pt_pedal_similarity,
)


ARCHITECTURES = ("independent_4x2", "joint_16")
METRIC_FIELDS = (
    "loss",
    "binary_accuracy",
    "exact_pattern_accuracy",
    "valid_notes",
    "valid_binary_targets",
)


class SharedBinaryWindowDataset(Dataset[dict[str, Any]]):
    """Read one immutable int16 token cache shared by both head architectures."""

    def __init__(self, cache_root: str | Path, split: str) -> None:
        if split not in {"train", "validation"}:
            raise ValueError("cache split must be train or validation")
        self.cache_root = Path(cache_root)
        self.split = split
        metadata_path = self.cache_root / "cache_statistics.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not metadata.get("completed"):
            raise RuntimeError("shared binary cache is not marked complete")
        split_metadata = metadata["splits"][split]
        self.cache_metadata = metadata
        self.split_metadata = split_metadata
        self.tokens = np.load(
            self.cache_root / split_metadata["tokens_file"], mmap_mode="r"
        )
        self.windows = np.load(
            self.cache_root / split_metadata["windows_file"], mmap_mode="r"
        )
        with (self.cache_root / split_metadata["index_file"]).open(
            newline="", encoding="utf-8"
        ) as handle:
            self.records = list(csv.DictReader(handle))
        if self.tokens.dtype != np.int16 or self.windows.dtype != np.int32:
            raise TypeError("cache dtypes must be int16 tokens and int32 windows")
        if tuple(self.tokens.shape) != (int(split_metadata["notes"]), 8):
            raise ValueError("token cache shape disagrees with metadata")
        if tuple(self.windows.shape) != (int(split_metadata["windows"]), 3):
            raise ValueError("window cache shape disagrees with metadata")
        if len(self.records) != int(split_metadata["performances"]):
            raise ValueError("performance index disagrees with metadata")

    @property
    def cache_id(self) -> str:
        return str(self.cache_metadata["cache_id"])

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        performance_index, start, end = (
            int(value) for value in self.windows[index]
        )
        record = self.records[performance_index]
        offset = int(record["token_offset"])
        note_count = int(record["notes"])
        if not (0 <= start < end <= note_count):
            raise ValueError("cached window is outside its performance")
        tokens = self.tokens[offset : offset + note_count]
        return make_masked_binary_sample(
            tokens,
            start,
            end,
            metadata={
                "source": record["source"],
                "dataset_split": record["dataset_split"],
                "performance_path": record["performance_path"],
                "performance_index": performance_index,
                "cache_id": self.cache_id,
            },
        )


def make_binary_loader(
    dataset: Dataset[dict[str, Any]],
    *,
    batch_size: int,
    pin_memory: bool,
    order: Sequence[int] | None = None,
) -> DataLoader:
    source: Dataset = Subset(dataset, list(order)) if order is not None else dataset
    return DataLoader(
        source,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
        collate_fn=binary_stage2_collate_fn,
    )


def training_window_order(length: int, seed: int, epoch: int) -> list[int]:
    """Expose the existing encoder-only deterministic order unchanged."""

    return deterministic_train_order(length, seed, epoch)


def order_sha256(order: Sequence[int]) -> str:
    values = np.asarray(order, dtype="<i8")
    return hashlib.sha256(values.tobytes()).hexdigest()


def binary_predictions(logits: torch.Tensor, architecture: str) -> torch.Tensor:
    if architecture == "independent_4x2":
        if logits.ndim != 4 or tuple(logits.shape[-2:]) != (4, 2):
            raise ValueError("independent logits must have shape [B,N,4,2]")
        return logits.argmax(dim=-1)
    if architecture == "joint_16":
        if logits.ndim != 3 or logits.shape[-1] != 16:
            raise ValueError("joint logits must have shape [B,N,16]")
        return decode_joint_targets(logits.argmax(dim=-1))
    raise ValueError(f"unsupported architecture: {architecture}")


def binary_batch_metrics(
    logits: torch.Tensor,
    binary_targets: torch.Tensor,
    loss: torch.Tensor | float,
    architecture: str,
) -> dict[str, float | int]:
    predicted = binary_predictions(logits, architecture)
    if predicted.shape != binary_targets.shape:
        raise ValueError("decoded prediction and binary target shapes differ")
    valid = binary_targets != IGNORE_INDEX
    valid_notes = valid.all(dim=-1)
    target_count = int(valid.sum().item())
    note_count = int(valid_notes.sum().item())
    if not target_count or target_count != note_count * 4:
        raise ValueError("binary metrics require four valid bits per valid note")
    correct = (predicted == binary_targets) & valid
    exact = ((predicted == binary_targets) | ~valid).all(dim=-1) & valid_notes
    loss_value = float(loss.detach().item()) if isinstance(loss, torch.Tensor) else float(loss)
    values: dict[str, float | int] = {
        "loss": loss_value,
        "binary_accuracy": float(correct.sum().item() / target_count),
        "exact_pattern_accuracy": float(exact.sum().item() / note_count),
        "valid_notes": note_count,
        "valid_binary_targets": target_count,
    }
    if not all(math.isfinite(float(values[name])) for name in METRIC_FIELDS[:3]):
        raise FloatingPointError("non-finite binary metric")
    return values


class BinaryMetricAccumulator:
    def __init__(self, architecture: str) -> None:
        if architecture not in ARCHITECTURES:
            raise ValueError("unknown architecture")
        self.architecture = architecture
        self.loss_weight = 0
        self.loss_sum = 0.0
        self.notes = 0
        self.targets = 0
        self.correct_bits = 0.0
        self.correct_patterns = 0.0

    def update(self, metrics: Mapping[str, float | int]) -> None:
        notes = int(metrics["valid_notes"])
        targets = int(metrics["valid_binary_targets"])
        loss_weight = targets if self.architecture == "independent_4x2" else notes
        self.loss_weight += loss_weight
        self.loss_sum += float(metrics["loss"]) * loss_weight
        self.notes += notes
        self.targets += targets
        self.correct_bits += float(metrics["binary_accuracy"]) * targets
        self.correct_patterns += float(metrics["exact_pattern_accuracy"]) * notes

    def compute(self) -> dict[str, float | int]:
        if not self.notes or not self.targets or not self.loss_weight:
            raise ValueError("cannot aggregate empty binary metrics")
        return {
            "loss": self.loss_sum / self.loss_weight,
            "binary_accuracy": self.correct_bits / self.targets,
            "exact_pattern_accuracy": self.correct_patterns / self.notes,
            "valid_notes": self.notes,
            "valid_binary_targets": self.targets,
        }


def forward_binary_model(
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    *,
    amp_enabled: bool,
) -> Any:
    input_ids = batch["input_ids"]
    use_amp = bool(amp_enabled and input_ids.device.type == "cuda")
    with torch.amp.autocast(
        device_type=input_ids.device.type,
        dtype=torch.float16 if input_ids.device.type == "cuda" else torch.bfloat16,
        enabled=use_amp,
    ):
        return model(
            input_ids=input_ids,
            token_attention_mask=batch["token_attention_mask"],
            note_mask=batch["note_mask"],
            **{model.target_name: batch[model.target_name]},
        )


def run_binary_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    architecture: str,
    device: torch.device,
    amp_enabled: bool,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    max_grad_norm: float = 1.0,
    max_batches: int | None = None,
    batch_callback: Callable[[int, Mapping[str, Any]], None] | None = None,
    max_consecutive_amp_skips: int | None = None,
) -> tuple[dict[str, float | int], dict[str, Any]]:
    training = optimizer is not None
    if training and scaler is None:
        raise ValueError("training requires a GradScaler")
    model.train(training)
    accumulator = BinaryMetricAccumulator(architecture)
    encoder_gradient = False
    head_gradient = False
    finite_gradients = True
    optimizer_steps = 0
    amp_skipped_steps = 0
    consecutive_amp_skips = 0
    maximum_consecutive_amp_skips = 0
    batches = 0
    for cpu_batch in loader:
        batch = move_batch_to_device(cpu_batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = forward_binary_model(model, batch, amp_enabled=amp_enabled)
        if output.loss is None or not bool(torch.isfinite(output.loss)):
            raise FloatingPointError("binary loss is missing or non-finite")
        if training:
            use_amp = bool(amp_enabled and device.type == "cuda")
            if use_amp:
                scaler.scale(output.loss).backward()
                scaler.unscale_(optimizer)
            else:
                output.loss.backward()
            encoder_grads = [p.grad for p in model.encoder.parameters() if p.requires_grad]
            head_grads = [p.grad for p in model.prediction_head_parameters()]
            encoder_gradient |= any(g is not None and bool(torch.any(g != 0)) for g in encoder_grads)
            head_gradient |= any(g is not None and bool(torch.any(g != 0)) for g in head_grads)
            gradients = [g for g in encoder_grads + head_grads if g is not None]
            if not gradients:
                raise RuntimeError("training produced no gradients")
            total_gradient_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=max_grad_norm,
                error_if_nonfinite=False,
            )
            gradient_norm_finite = bool(torch.isfinite(total_gradient_norm))
            finite_gradients &= gradient_norm_finite
            if use_amp:
                scale_before = float(scaler.get_scale())
                scaler.step(optimizer)
                scaler.update()
                scale_after = float(scaler.get_scale())
                if gradient_norm_finite and scale_after >= scale_before:
                    optimizer_steps += 1
                    consecutive_amp_skips = 0
                else:
                    # GradScaler skips the optimizer update and backs off its
                    # dynamic scale when unscale_ finds an overflow. This is
                    # the standard AMP recovery path, not a config change.
                    amp_skipped_steps += 1
                    consecutive_amp_skips += 1
                    maximum_consecutive_amp_skips = max(
                        maximum_consecutive_amp_skips, consecutive_amp_skips
                    )
                    if (
                        max_consecutive_amp_skips is not None
                        and consecutive_amp_skips >= max_consecutive_amp_skips
                    ):
                        raise FloatingPointError(
                            "persistent AMP overflow: "
                            f"{consecutive_amp_skips} consecutive skipped steps"
                        )
            else:
                if not gradient_norm_finite:
                    raise FloatingPointError("non-finite gradient norm without AMP")
                optimizer.step()
                optimizer_steps += 1
        metrics = binary_batch_metrics(
            output.logits.detach(), batch["binary_targets"], output.loss, architecture
        )
        accumulator.update(metrics)
        batches += 1
        if batch_callback is not None:
            batch_callback(
                batches,
                {
                    "loss": float(metrics["loss"]),
                    "optimizer_steps": optimizer_steps if training else 0,
                    "amp_skipped_steps": amp_skipped_steps if training else 0,
                    "consecutive_amp_skips": consecutive_amp_skips if training else 0,
                    "gradient_norm_finite": gradient_norm_finite if training else None,
                    "amp_scale_before": scale_before if training and use_amp else None,
                    "amp_scale_after": scale_after if training and use_amp else None,
                },
            )
        if max_batches is not None and batches >= max_batches:
            break
    return accumulator.compute(), {
        "batches": batches,
        "optimizer_steps": optimizer_steps if training else 0,
        "amp_skipped_steps": amp_skipped_steps if training else 0,
        "maximum_consecutive_amp_skips": (
            maximum_consecutive_amp_skips if training else 0
        ),
        "encoder_gradient_present": encoder_gradient if training else None,
        "head_gradient_present": head_gradient if training else None,
        "finite_gradients": finite_gradients if training else None,
    }


@dataclass
class PedalMetricEarlyStopping:
    patience: int
    tie_tolerance: float = 1e-12
    best_js_distance: float = math.inf
    best_intersection: float = -math.inf
    best_epoch: int = 0
    counter: int = 0

    def update(self, js_distance: float, intersection: float, epoch: int) -> tuple[bool, bool]:
        if not (math.isfinite(js_distance) and math.isfinite(intersection)):
            raise FloatingPointError("non-finite validation distribution metric")
        better_js = js_distance < self.best_js_distance - self.tie_tolerance
        tied_js = abs(js_distance - self.best_js_distance) <= self.tie_tolerance
        better_tie = tied_js and intersection > self.best_intersection + self.tie_tolerance
        improved = better_js or better_tie
        if improved:
            self.best_js_distance = js_distance
            self.best_intersection = intersection
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
        return improved, self.counter >= self.patience


def evaluate_stage1_cache_distribution(
    model: torch.nn.Module,
    *,
    architecture: str,
    stage1_manifest_csv: str | Path,
    human_histogram: Sequence[int],
    device: torch.device,
    max_pieces: int | None = None,
) -> dict[str, Any]:
    """Run Stage 2 only on immutable Stage 1 ID arrays; no MIDI is opened."""

    with Path(stage1_manifest_csv).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 19:
        raise ValueError("Stage 1 validation cache manifest must contain 19 pieces")
    selected = rows if max_pieces is None else rows[:max_pieces]
    if not selected:
        raise ValueError("distribution validation requires at least one cached piece")
    candidate_histogram = np.zeros(16, dtype=np.int64)
    notes = 0
    windows = 0
    non_pedal_preserved = True
    for row in selected:
        ids = np.load(row["generated_ids_path"])
        candidate_ids, details = infer_cached_binary_pedals(
            model,
            ids,
            architecture=architecture,
            device=device,
            window_notes=512,
            stride_notes=256,
        )
        candidate_histogram += joint16_histogram_from_ids(candidate_ids)
        notes += int(details["notes"])
        windows += int(details["windows"])
        non_pedal_preserved &= bool(details["non_pedal_tokens_preserved"])
    similarity = official_pt_pedal_similarity(human_histogram, candidate_histogram)
    if not non_pedal_preserved:
        raise AssertionError("cached validation changed non-pedal Stage 1 tokens")
    return {
        "pieces": len(selected),
        "notes": notes,
        "windows": windows,
        "candidate_histogram": candidate_histogram.tolist(),
        "js_distance": float(similarity["js_distance_base2"]),
        "js_divergence": float(similarity["js_divergence_base2"]),
        "intersection": float(similarity["histogram_intersection"]),
        "non_pedal_tokens_exact": non_pedal_preserved,
        "stage1_inference_rerun": False,
        "midi_access_count": 0,
    }
