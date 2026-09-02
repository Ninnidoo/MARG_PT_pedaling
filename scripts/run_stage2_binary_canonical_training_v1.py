#!/usr/bin/env python3
"""Sequential fresh binary Stage-2 training on frozen canonical validation MIDI."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
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

from src.stage2_binary.canonical_stage1 import sha256_file  # noqa: E402
from src.stage2_binary.canonical_validation import (  # noqa: E402
    evaluate_canonical_stage1,
    verify_canonical_bank,
)
from src.stage2_binary.full_training import (  # noqa: E402
    PedalMetricEarlyStopping,
    SharedBinaryWindowDataset,
    make_binary_loader,
    order_sha256,
    run_binary_epoch,
    training_window_order,
)
from src.stage2_binary.model import (  # noqa: E402
    IndependentBinaryPedalModel,
    JointBinaryPedalModel,
)
from src.stage2_binary.training import build_binary_optimizer  # noqa: E402
from src.stage2_encoder_only.train import (  # noqa: E402
    atomic_torch_save,
    capture_rng_states,
    get_gpu_identity,
)
from src.stage2_encoder_only.training import set_deterministic_seed  # noqa: E402


ARCHITECTURES = ("independent_4x2", "joint_16")
MODEL_CLASSES = {
    "independent_4x2": IndependentBinaryPedalModel,
    "joint_16": JointBinaryPedalModel,
}
METRIC_COLUMNS = (
    "epoch",
    "train_loss",
    "train_binary_accuracy",
    "train_exact_pattern_accuracy",
    "human_validation_loss",
    "human_validation_binary_accuracy",
    "human_validation_exact_pattern_accuracy",
    "canonical_strict_js_distance",
    "canonical_strict_intersection",
    "canonical_js_absolute_change_vs_original_pt",
    "canonical_js_relative_change_vs_original_pt",
    "canonical_intersection_change_vs_original_pt",
    "canonical_beats_original_pt",
    "canonical_steady_mass",
    "canonical_transition_containing_mass",
    "original_pt_strict_js_distance",
    "original_pt_strict_intersection",
    "canonical_non_cc64_equality_count",
    "epoch_time_seconds",
    "global_optimizer_step",
    "amp_skipped_optimizer_steps",
    "maximum_consecutive_amp_skips",
    "encoder_learning_rate",
    "head_learning_rate",
    "peak_gpu_memory_bytes",
)
CHECKPOINT_COLUMNS = (
    "kind",
    "epoch",
    "checkpoint_path",
    "checkpoint_sha256",
    "checkpoint_bytes",
    "canonical_strict_js_distance",
    "canonical_strict_intersection",
)
EQUALITY_COLUMNS = (
    "architecture",
    "epoch",
    "piece_id",
    "canonical_sha256",
    "candidate_sha256",
    "midi_type_exact",
    "ticks_per_beat_exact",
    "track_count_exact",
    "all_ordered_non_cc64_events_exact",
    "canonical_cc64_events",
    "candidate_cc64_events",
    "cc64_changed",
    "donor_cc64_discarded_after_canonical_eot",
    "status",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _sha256(path: str | Path) -> str:
    return sha256_file(path)


def _parameter_hash(parameters: Any) -> str:
    digest = hashlib.sha256()
    with torch.no_grad():
        for parameter in parameters:
            value = parameter.detach().cpu().contiguous()
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _input_hash(
    dataset: SharedBinaryWindowDataset,
    order: Sequence[int],
    batch_size: int,
) -> str:
    digest = hashlib.sha256()
    for index in order[:batch_size]:
        values = dataset[int(index)]["input_ids"].numpy().astype("<i8", copy=False)
        digest.update(values.tobytes())
    return digest.hexdigest()


def _status_path(config: Mapping[str, Any]) -> Path:
    return Path(config["experiment_root"]) / "training_run_status.json"


def _update_status(config: Mapping[str, Any], **updates: Any) -> dict[str, Any]:
    path = _status_path(config)
    current = _load_json(path) if path.is_file() else {}
    current.update(updates, updated_at=_now())
    _atomic_json(path, current)
    return current


def _pt_git_head(repository: Path) -> str:
    head = (repository / ".git/HEAD").read_text(encoding="utf-8").strip()
    if not head.startswith("ref: "):
        return head
    ref = repository / ".git" / head.removeprefix("ref: ")
    if ref.is_file():
        return ref.read_text(encoding="utf-8").strip()
    packed = repository / ".git/packed-refs"
    for line in packed.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith(("#", "^")):
            digest, name = line.split(" ", 1)
            if name == head.removeprefix("ref: "):
                return digest
    raise RuntimeError("cannot resolve pinned PT git revision")


def _verify_source_configuration(config: Mapping[str, Any]) -> None:
    source = _load_json(config["validated_training_config_source"])
    fixed = (
        "seed",
        "cache_root",
        "checkpoint_path",
        "architectures",
        "expected_train_performances",
        "expected_train_notes",
        "expected_train_windows",
        "expected_validation_performances",
        "expected_validation_pieces",
        "expected_validation_notes",
        "expected_validation_windows",
        "window_notes",
        "stride_notes",
        "optimizer",
        "encoder_lr",
        "head_lr",
        "weight_decay",
        "max_grad_norm",
        "amp_enabled",
        "amp_init_scale",
        "batch_size",
        "gradient_accumulation_steps",
        "effective_batch_size",
        "num_workers",
        "pin_memory",
        "max_epochs",
        "early_stopping_patience",
        "js_tie_tolerance",
    )
    differences = {
        key: {"validated": source.get(key), "canonical": config.get(key)}
        for key in fixed
        if source.get(key) != config.get(key)
    }
    if differences:
        raise RuntimeError(f"canonical run changed validated binary configuration: {differences}")


def preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    """Verify every frozen input without opening any ASAP test MIDI."""

    experiment_root = Path(config["experiment_root"])
    train_root = Path(config["output_root"])
    protected = (
        Path(config["canonical_manifest"]),
        Path(config["canonical_baseline"]),
        Path(config["canonical_stage1_report"]),
        Path(config["experiment_protocol"]),
    )
    if train_root.exists():
        raise FileExistsError(f"refusing to overwrite canonical training root: {train_root}")
    for output in (
        experiment_root / "canonical_validation_trajectory.csv",
        experiment_root / "model_comparison_validation.csv",
        experiment_root / "CANONICAL_BINARY_TRAINING_REPORT.md",
    ):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite canonical training artifact: {output}")
    _verify_source_configuration(config)
    expected_hashes = config["expected_file_sha256"]
    hash_results: dict[str, str] = {}
    for name, item in config["provenance_files"].items():
        path = Path(item)
        digest = _sha256(path)
        hash_results[name] = digest
        if digest != expected_hashes[name]:
            raise RuntimeError(f"provenance hash mismatch for {name}: {path}")
    if _pt_git_head(ROOT / "third_party/PianistTransformer") != config["pinned_pt_commit"]:
        raise RuntimeError("Pianist Transformer revision changed")

    cache = _load_json(Path(config["cache_root"]) / "cache_statistics.json")
    if not cache.get("completed") or cache.get("cache_id") != config["expected_cache_id"]:
        raise RuntimeError("shared cache completion/ID mismatch")
    train_cache = cache["splits"]["train"]
    validation_cache = cache["splits"]["validation"]
    if (
        int(train_cache["performances"]) != int(config["expected_train_performances"])
        or int(train_cache["notes"]) != int(config["expected_train_notes"])
        or int(train_cache["windows"]) != int(config["expected_train_windows"])
        or int(validation_cache["performances"]) != int(config["expected_validation_performances"])
        or int(validation_cache["notes"]) != int(config["expected_validation_notes"])
        or int(validation_cache["windows"]) != int(config["expected_validation_windows"])
    ):
        raise RuntimeError("shared cache inventory mismatch")
    if train_cache["source_statistics"] != config["expected_train_source_statistics"]:
        raise RuntimeError("natural-concatenation training source inventory changed")
    train_dataset = SharedBinaryWindowDataset(config["cache_root"], "train")
    validation_dataset = SharedBinaryWindowDataset(config["cache_root"], "validation")
    if train_dataset.cache_id != validation_dataset.cache_id:
        raise RuntimeError("train and validation cache IDs differ")
    first_order = training_window_order(len(train_dataset), int(config["seed"]), 1)
    first_order_hash = order_sha256(first_order)
    first_batch_hash = _input_hash(train_dataset, first_order, int(config["batch_size"]))
    if first_order_hash != config["expected_epoch1_window_order_sha256"]:
        raise RuntimeError("epoch-1 training order changed")
    if first_batch_hash != config["expected_first_training_batch_input_sha256"]:
        raise RuntimeError("first masked training batch changed")

    canonical = verify_canonical_bank(
        config["canonical_manifest"],
        config["canonical_baseline"],
        tolerance=float(config["js_tie_tolerance"]),
    )
    if abs(canonical["js_distance"] - float(config["original_pt_strict_js_distance"])) > float(config["js_tie_tolerance"]):
        raise RuntimeError("configured Original-PT canonical JS mismatch")
    if abs(canonical["intersection"] - float(config["original_pt_strict_intersection"])) > float(config["js_tie_tolerance"]):
        raise RuntimeError("configured Original-PT canonical Intersection mismatch")
    for path in protected:
        if not path.is_file():
            raise FileNotFoundError(path)
    return {
        "cache_pass": True,
        "canonical_bank_pass": True,
        "cache_id": train_dataset.cache_id,
        "train_windows": len(train_dataset),
        "validation_windows": len(validation_dataset),
        "epoch1_window_order_sha256": first_order_hash,
        "first_training_batch_input_sha256": first_batch_hash,
        "canonical": canonical,
        "provenance_hashes": hash_results,
        "protected_artifact_hashes": {str(path): _sha256(path) for path in protected},
        "asap_test_midi_access_count": 0,
    }


def _epoch_checkpoint_payload(
    model: torch.nn.Module,
    run_config: Mapping[str, Any],
    row: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "model_state": model.state_dict(),
        "configuration": dict(run_config),
        "epoch": int(row["epoch"]),
        "canonical_strict_js_distance": float(row["canonical_strict_js_distance"]),
        "canonical_strict_intersection": float(row["canonical_strict_intersection"]),
    }


def _last_checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    run_config: Mapping[str, Any],
    row: Mapping[str, Any],
    stopping: PedalMetricEarlyStopping,
    global_step: int,
    amp_skipped_total: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "grad_scaler_state": scaler.state_dict(),
        "configuration": dict(run_config),
        "completed_epoch": int(row["epoch"]),
        "global_optimizer_step": global_step,
        "best_epoch": stopping.best_epoch,
        "best_canonical_strict_js_distance": stopping.best_js_distance,
        "best_canonical_strict_intersection": stopping.best_intersection,
        "early_stopping_counter": stopping.counter,
        "amp_skipped_optimizer_steps_total": amp_skipped_total,
    }
    payload.update(capture_rng_states())
    return payload


def _checkpoint_manifest(
    output_dir: Path,
    epoch_rows: Sequence[Mapping[str, Any]],
    metrics_rows: Sequence[Mapping[str, Any]],
    best_epoch: int,
) -> list[dict[str, Any]]:
    metric_by_epoch = {int(row["epoch"]): row for row in metrics_rows}
    rows: list[dict[str, Any]] = []
    for item in epoch_rows:
        epoch = int(item["epoch"])
        metric = metric_by_epoch[epoch]
        path = Path(item["checkpoint_path"])
        rows.append(
            {
                "kind": "epoch",
                "epoch": epoch,
                "checkpoint_path": str(path),
                "checkpoint_sha256": _sha256(path),
                "checkpoint_bytes": path.stat().st_size,
                "canonical_strict_js_distance": metric["canonical_strict_js_distance"],
                "canonical_strict_intersection": metric["canonical_strict_intersection"],
            }
        )
    if best_epoch not in metric_by_epoch:
        raise RuntimeError("selected best epoch is absent from metrics")
    for kind, epoch in (("best", best_epoch), ("last", int(metrics_rows[-1]["epoch"]))):
        path = output_dir / f"{kind}.pt"
        metric = metric_by_epoch[epoch]
        rows.append(
            {
                "kind": kind,
                "epoch": epoch,
                "checkpoint_path": str(path),
                "checkpoint_sha256": _sha256(path),
                "checkpoint_bytes": path.stat().st_size,
                "canonical_strict_js_distance": metric["canonical_strict_js_distance"],
                "canonical_strict_intersection": metric["canonical_strict_intersection"],
            }
        )
    return rows


def train_architecture(
    config: Mapping[str, Any],
    architecture: str,
    preflight_result: Mapping[str, Any],
    trajectory_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    output_dir = Path(config["output_root"]) / architecture
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite architecture output: {output_dir}")
    output_dir.mkdir(parents=True)
    log_handle = (output_dir / "train.log").open("w", encoding="utf-8")

    def log(message: str) -> None:
        print(message, flush=True)
        print(message, file=log_handle, flush=True)

    architecture_status: dict[str, Any] = {
        "architecture": architecture,
        "status": "running",
        "fresh_start_from_official_pretrained_encoder": True,
        "canonical_validation_only": True,
        "stage1_inference_regeneration": 0,
        "asap_test_midi_access_count": 0,
        "started_at": _now(),
    }
    _atomic_json(output_dir / "run_status.json", architecture_status)
    try:
        set_deterministic_seed(int(config["seed"]))
        device = torch.device("cuda:0")
        train_dataset = SharedBinaryWindowDataset(config["cache_root"], "train")
        validation_dataset = SharedBinaryWindowDataset(config["cache_root"], "validation")
        epoch_orders = {
            str(epoch): order_sha256(
                training_window_order(len(train_dataset), int(config["seed"]), epoch)
            )
            for epoch in range(1, int(config["max_epochs"]) + 1)
        }
        model = MODEL_CLASSES[architecture].from_pretrained(
            config["checkpoint_path"],
            torch_dtype=torch.float32,
            attn_implementation="eager",
        )
        initial_encoder_hash = _parameter_hash(model.encoder.parameters())
        if initial_encoder_hash != config["expected_initial_encoder_sha256"]:
            raise RuntimeError("fresh pretrained encoder hash mismatch")
        run_config = dict(config)
        run_config.update(
            architecture=architecture,
            output_dir=str(output_dir),
            initial_encoder_parameter_sha256=initial_encoder_hash,
            epoch_window_order_sha256=epoch_orders,
            cache_id=train_dataset.cache_id,
            parameter_count=model.parameter_count,
            trainable_parameter_count=model.trainable_parameter_count,
            input_grammar=["Pitch", "IOI", "Velocity", "Duration", "MASK", "MASK", "MASK", "MASK"],
            loss=(
                "unweighted cross entropy over four independent binary heads"
                if architecture == "independent_4x2"
                else "unweighted 16-class joint-pattern cross entropy"
            ),
            canonical_validation=(
                "official midi_to_ids -> masked Stage 2 -> deterministic argmax -> "
                "CC64-only transplant -> strict non-CC64 equality -> official metric"
            ),
        )
        _atomic_json(output_dir / "config.json", run_config)
        log(
            f"ENCODER_LOAD_PASS architecture={architecture} "
            f"encoder_sha256={initial_encoder_hash}"
        )
        if architecture == ARCHITECTURES[0]:
            _update_status(
                config,
                state="independent_encoder_load_pass",
                pretrained_encoder_load_pass=True,
                initial_encoder_parameter_sha256=initial_encoder_hash,
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
        stopping = PedalMetricEarlyStopping(
            patience=int(config["early_stopping_patience"]),
            tie_tolerance=float(config["js_tie_tolerance"]),
        )
        validation_loader = make_binary_loader(
            validation_dataset,
            batch_size=int(config["batch_size"]),
            pin_memory=bool(config["pin_memory"]),
        )
        baseline_js = float(config["original_pt_strict_js_distance"])
        baseline_intersection = float(config["original_pt_strict_intersection"])
        metrics_rows: list[dict[str, Any]] = []
        equality_rows: list[dict[str, Any]] = []
        epoch_checkpoint_rows: list[dict[str, Any]] = []
        global_step = 0
        amp_skipped_total = 0
        maximum_consecutive_skips = 0
        metrics_path = output_dir / "metrics.csv"
        with metrics_path.open("w", newline="", encoding="utf-8") as metrics_handle:
            writer = csv.DictWriter(metrics_handle, fieldnames=METRIC_COLUMNS)
            writer.writeheader()
            for epoch in range(1, int(config["max_epochs"]) + 1):
                epoch_started = time.perf_counter()
                torch.cuda.reset_peak_memory_stats(device)
                order = training_window_order(len(train_dataset), int(config["seed"]), epoch)
                train_loader = make_binary_loader(
                    train_dataset,
                    batch_size=int(config["batch_size"]),
                    pin_memory=bool(config["pin_memory"]),
                    order=order,
                )

                def batch_callback(batch: int, details: Mapping[str, Any]) -> None:
                    if batch == 1:
                        finite = math.isfinite(float(details["loss"]))
                        log(
                            f"FIRST_TRAIN_STEP_PASS architecture={architecture} epoch={epoch} "
                            f"loss={float(details['loss']):.9f} finite={finite} "
                            f"optimizer_steps={details['optimizer_steps']}"
                        )
                        if architecture == ARCHITECTURES[0] and epoch == 1:
                            _update_status(
                                config,
                                state="independent_first_training_step_pass",
                                status="running",
                                independent_first_training_step_pass=True,
                                first_step_loss=float(details["loss"]),
                                first_step_finite=finite,
                            )

                train_metrics, train_checks = run_binary_epoch(
                    model,
                    train_loader,
                    architecture=architecture,
                    device=device,
                    amp_enabled=bool(config["amp_enabled"]),
                    optimizer=optimizer,
                    scaler=scaler,
                    max_grad_norm=float(config["max_grad_norm"]),
                    batch_callback=batch_callback,
                    max_consecutive_amp_skips=int(config["max_consecutive_amp_skips"]),
                )
                global_step += int(train_checks["optimizer_steps"])
                amp_skipped_total += int(train_checks["amp_skipped_steps"])
                maximum_consecutive_skips = max(
                    maximum_consecutive_skips,
                    int(train_checks["maximum_consecutive_amp_skips"]),
                )
                human_metrics, _ = run_binary_epoch(
                    model,
                    validation_loader,
                    architecture=architecture,
                    device=device,
                    amp_enabled=bool(config["amp_enabled"]),
                )
                distribution = evaluate_canonical_stage1(
                    model,
                    architecture=architecture,
                    manifest_path=config["canonical_manifest"],
                    baseline_path=config["canonical_baseline"],
                    device=device,
                    temporary_root=output_dir / "canonical_validation_tmp",
                    event_logger=log,
                )
                improved, should_stop = stopping.update(
                    distribution["strict_js_distance"],
                    distribution["strict_intersection"],
                    epoch,
                )
                row: dict[str, Any] = {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "train_binary_accuracy": train_metrics["binary_accuracy"],
                    "train_exact_pattern_accuracy": train_metrics["exact_pattern_accuracy"],
                    "human_validation_loss": human_metrics["loss"],
                    "human_validation_binary_accuracy": human_metrics["binary_accuracy"],
                    "human_validation_exact_pattern_accuracy": human_metrics[
                        "exact_pattern_accuracy"
                    ],
                    "canonical_strict_js_distance": distribution["strict_js_distance"],
                    "canonical_strict_intersection": distribution["strict_intersection"],
                    "canonical_js_absolute_change_vs_original_pt": distribution[
                        "strict_js_distance"
                    ]
                    - baseline_js,
                    "canonical_js_relative_change_vs_original_pt": (
                        distribution["strict_js_distance"] / baseline_js - 1.0
                    ),
                    "canonical_intersection_change_vs_original_pt": distribution[
                        "strict_intersection"
                    ]
                    - baseline_intersection,
                    "canonical_beats_original_pt": distribution["strict_js_distance"]
                    < baseline_js - float(config["js_tie_tolerance"]),
                    "canonical_steady_mass": distribution["steady_mass"],
                    "canonical_transition_containing_mass": distribution[
                        "transition_containing_mass"
                    ],
                    "original_pt_strict_js_distance": baseline_js,
                    "original_pt_strict_intersection": baseline_intersection,
                    "canonical_non_cc64_equality_count": distribution[
                        "non_cc64_equality_count"
                    ],
                    "epoch_time_seconds": time.perf_counter() - epoch_started,
                    "global_optimizer_step": global_step,
                    "amp_skipped_optimizer_steps": train_checks["amp_skipped_steps"],
                    "maximum_consecutive_amp_skips": train_checks[
                        "maximum_consecutive_amp_skips"
                    ],
                    "encoder_learning_rate": optimizer.param_groups[0]["lr"],
                    "head_learning_rate": optimizer.param_groups[1]["lr"],
                    "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
                }
                numeric_fields = [
                    key for key in METRIC_COLUMNS if key not in {"epoch", "canonical_beats_original_pt"}
                ]
                if not all(math.isfinite(float(row[key])) for key in numeric_fields):
                    raise FloatingPointError("non-finite canonical epoch metric")
                if int(row["canonical_non_cc64_equality_count"]) != 19:
                    raise AssertionError("canonical non-CC64 equality gate is not 19/19")
                writer.writerow(row)
                metrics_handle.flush()
                os.fsync(metrics_handle.fileno())
                metrics_rows.append(row)
                for equality in distribution["equality_rows"]:
                    equality_rows.append(
                        {"architecture": architecture, "epoch": epoch, **equality}
                    )
                _atomic_csv(
                    output_dir / "canonical_non_cc64_equality.csv",
                    EQUALITY_COLUMNS,
                    equality_rows,
                )

                epoch_path = output_dir / f"epoch_{epoch:03d}.pt"
                atomic_torch_save(
                    _epoch_checkpoint_payload(model, run_config, row),
                    epoch_path,
                )
                epoch_checkpoint_rows.append(
                    {"epoch": epoch, "checkpoint_path": str(epoch_path)}
                )
                atomic_torch_save(
                    _last_checkpoint_payload(
                        model,
                        optimizer,
                        scaler,
                        run_config,
                        row,
                        stopping,
                        global_step,
                        amp_skipped_total,
                    ),
                    output_dir / "last.pt",
                )
                if improved:
                    atomic_torch_save(
                        {
                            "model_state": model.state_dict(),
                            "configuration": run_config,
                            "best_epoch": epoch,
                            "best_canonical_strict_js_distance": distribution[
                                "strict_js_distance"
                            ],
                            "best_canonical_strict_intersection": distribution[
                                "strict_intersection"
                            ],
                        },
                        output_dir / "best.pt",
                    )
                trajectory = {"architecture": architecture, **row}
                trajectory_rows.append(trajectory)
                _atomic_csv(
                    Path(config["experiment_root"]) / "canonical_validation_trajectory.csv",
                    ("architecture", *METRIC_COLUMNS),
                    trajectory_rows,
                )
                log("EPOCH " + json.dumps(row, sort_keys=True))
                _update_status(
                    config,
                    state=f"{architecture}_training",
                    status="running",
                    current_architecture=architecture,
                    current_epoch=epoch,
                    current_canonical_strict_js=distribution["strict_js_distance"],
                    current_canonical_strict_intersection=distribution[
                        "strict_intersection"
                    ],
                    canonical_non_cc64_equality="19/19 PASS",
                )
                if should_stop:
                    log(
                        f"EARLY_STOP architecture={architecture} epoch={epoch} "
                        f"best_epoch={stopping.best_epoch}"
                    )
                    break

        checkpoint_rows = _checkpoint_manifest(
            output_dir,
            epoch_checkpoint_rows,
            metrics_rows,
            stopping.best_epoch,
        )
        _atomic_csv(
            output_dir / "checkpoint_manifest.csv",
            CHECKPOINT_COLUMNS,
            checkpoint_rows,
        )
        best_row = next(row for row in metrics_rows if int(row["epoch"]) == stopping.best_epoch)
        architecture_status.update(
            status="completed",
            completed_at=_now(),
            completed_epochs=len(metrics_rows),
            best_epoch=stopping.best_epoch,
            best_canonical_strict_js_distance=stopping.best_js_distance,
            best_canonical_strict_intersection=stopping.best_intersection,
            best_checkpoint_sha256=_sha256(output_dir / "best.pt"),
            last_checkpoint_sha256=_sha256(output_dir / "last.pt"),
            epoch_checkpoints_saved=len(epoch_checkpoint_rows),
            checkpoint_manifest_entries=len(checkpoint_rows),
            amp_skipped_optimizer_steps_total=amp_skipped_total,
            maximum_consecutive_amp_skips=maximum_consecutive_skips,
            initial_encoder_parameter_sha256=initial_encoder_hash,
            epoch1_window_order_sha256=epoch_orders["1"],
            canonical_non_cc64_equality_count=len(equality_rows),
            stage1_inference_regeneration=0,
            asap_test_midi_access_count=0,
        )
        _atomic_json(output_dir / "run_status.json", architecture_status)
        log(
            f"ARCHITECTURE_COMPLETE architecture={architecture} "
            f"epochs={len(metrics_rows)} best_epoch={stopping.best_epoch} "
            f"canonical_js={stopping.best_js_distance:.15f}"
        )
        return {
            "architecture": architecture,
            "output_dir": str(output_dir),
            "metrics": metrics_rows,
            "best": best_row,
            "status": architecture_status,
            "checkpoint_manifest": checkpoint_rows,
        }
    except BaseException as error:
        architecture_status.update(
            status="failed",
            failed_at=_now(),
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
        _atomic_json(output_dir / "run_status.json", architecture_status)
        log(f"ARCHITECTURE_FAILED architecture={architecture} error={error!r}")
        log(traceback.format_exc())
        raise
    finally:
        log_handle.close()
        if "model" in locals():
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _report(config: Mapping[str, Any], runs: Mapping[str, Mapping[str, Any]]) -> str:
    labels = {"independent_4x2": "Independent 4×2", "joint_16": "Joint 16"}
    lines = [
        "# Canonical Binary Stage 2 Training Report",
        "",
        "## A. Experiment setup",
        "",
        "This controlled run retrains both existing binary hypotheses from the same official pretrained Pianist Transformer encoder. The training pool, natural concatenation, unweighted losses, optimizer, seed, and budget are unchanged from the validated binary configuration.",
        "",
        "- Training data: MAESTRO-clean 1,170 + ASAP train 892 = 2,062 performances / 9,369,095 notes / 35,573 windows",
        f"- Shared cache ID: `{config['expected_cache_id']}`",
        "- Seed: 42; window/stride: 512/256; batch/effective batch: 16/16",
        "- AdamW: encoder LR 1e-5, head LR 1e-4, weight decay 0.01; gradient clip 1.0; AMP enabled",
        "- Maximum 10 epochs; canonical strict-JS early stopping patience 3",
        f"- Frozen canonical Stage 1 manifest: `{config['canonical_manifest']}`",
        f"- Original PT baseline: JS `{float(config['original_pt_strict_js_distance']):.15f}`, Intersection `{float(config['original_pt_strict_intersection']):.15f}`",
        "- Stage 2 input: `[Pitch, IOI, Velocity, Duration, MASK, MASK, MASK, MASK]`",
        "- Final validation candidates are canonical MIDI plus transplanted CC64 only; all non-CC64 raw MIDI/meta events pass exact equality before metrics.",
        "",
    ]
    for section, architecture in (("B", "independent_4x2"), ("C", "joint_16")):
        lines += [
            f"## {section}. {labels[architecture]} trajectory",
            "",
            "| Epoch | Train loss | Human-val loss | Human binary acc | Human exact-pattern acc | Canonical JS ↓ | Intersection ↑ | ΔJS | Relative ΔJS | ΔIntersection | Beats PT? |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for row in runs[architecture]["metrics"]:
            lines.append(
                f"| {row['epoch']} | {float(row['train_loss']):.9f} | {float(row['human_validation_loss']):.9f} | {float(row['human_validation_binary_accuracy']):.9f} | {float(row['human_validation_exact_pattern_accuracy']):.9f} | {float(row['canonical_strict_js_distance']):.12f} | {float(row['canonical_strict_intersection']):.12f} | {float(row['canonical_js_absolute_change_vs_original_pt']):+.12f} | {float(row['canonical_js_relative_change_vs_original_pt']):+.6%} | {float(row['canonical_intersection_change_vs_original_pt']):+.12f} | {'yes' if row['canonical_beats_original_pt'] else 'no'} |"
            )
        lines.append("")
    lines += [
        "## D. Best validation result",
        "",
        "| Model | Best epoch | Strict JS ↓ | Intersection ↑ | Beats Original PT? |",
        "|---|---:|---:|---:|---|",
        f"| Original PT | — | {float(config['original_pt_strict_js_distance']):.12f} | {float(config['original_pt_strict_intersection']):.12f} | — |",
    ]
    for architecture in ARCHITECTURES:
        best = runs[architecture]["best"]
        lines.append(
            f"| {labels[architecture]} | {best['epoch']} | {float(best['canonical_strict_js_distance']):.12f} | {float(best['canonical_strict_intersection']):.12f} | {'yes' if best['canonical_beats_original_pt'] else 'no'} |"
        )
    lines += ["", "## E. Locked checkpoint hashes", ""]
    for architecture in ARCHITECTURES:
        status = runs[architecture]["status"]
        lines += [
            f"- {labels[architecture]} `best.pt`: `{status['best_checkpoint_sha256']}`",
            f"- {labels[architecture]} `last.pt`: `{status['last_checkpoint_sha256']}`",
            f"- Per-epoch and alias hashes: `{runs[architecture]['output_dir']}/checkpoint_manifest.csv`",
        ]
    lines += [
        "",
        "## F. Test isolation",
        "",
        "ASAP test was not used for training, model selection, checkpoint selection, calibration, or evaluation in this stage.",
        "",
        "- ASAP test human MIDI access: 0",
        "- Canonical test-bank evaluation: not performed",
        "- Test JS / Intersection: not computed",
        "- Test audio: not generated",
        "- Stage 1 validation neural inference regeneration: 0",
    ]
    return "\n".join(lines) + "\n"


def _finalize(
    config: Mapping[str, Any],
    runs: Mapping[str, Mapping[str, Any]],
    preflight_result: Mapping[str, Any],
) -> None:
    root = Path(config["experiment_root"])
    comparison: list[dict[str, Any]] = [
        {
            "model": "Original PT",
            "best_epoch": "",
            "canonical_strict_js_distance": config["original_pt_strict_js_distance"],
            "canonical_strict_intersection": config["original_pt_strict_intersection"],
            "js_change_vs_original_pt": 0.0,
            "intersection_change_vs_original_pt": 0.0,
            "beats_original_pt": "",
            "best_checkpoint_path": "",
            "best_checkpoint_sha256": "",
        }
    ]
    for architecture in ARCHITECTURES:
        run = runs[architecture]
        best = run["best"]
        comparison.append(
            {
                "model": architecture,
                "best_epoch": best["epoch"],
                "canonical_strict_js_distance": best["canonical_strict_js_distance"],
                "canonical_strict_intersection": best["canonical_strict_intersection"],
                "js_change_vs_original_pt": best[
                    "canonical_js_absolute_change_vs_original_pt"
                ],
                "intersection_change_vs_original_pt": best[
                    "canonical_intersection_change_vs_original_pt"
                ],
                "beats_original_pt": best["canonical_beats_original_pt"],
                "best_checkpoint_path": str(Path(run["output_dir"]) / "best.pt"),
                "best_checkpoint_sha256": run["status"]["best_checkpoint_sha256"],
            }
        )
    fields = tuple(comparison[0])
    _atomic_csv(root / "model_comparison_validation.csv", fields, comparison)
    (root / "CANONICAL_BINARY_TRAINING_REPORT.md").write_text(
        _report(config, runs), encoding="utf-8"
    )
    _update_status(
        config,
        status="completed",
        state="completed",
        completed_at=_now(),
        completed_architectures=list(ARCHITECTURES),
        preflight=preflight_result,
        results={
            architecture: {
                "best_epoch": int(runs[architecture]["best"]["epoch"]),
                "canonical_strict_js_distance": float(
                    runs[architecture]["best"]["canonical_strict_js_distance"]
                ),
                "canonical_strict_intersection": float(
                    runs[architecture]["best"]["canonical_strict_intersection"]
                ),
                "best_checkpoint_sha256": runs[architecture]["status"][
                    "best_checkpoint_sha256"
                ],
            }
            for architecture in ARCHITECTURES
        },
        asap_test_midi_access_count=0,
        asap_test_evaluation_performed=False,
        stage1_inference_regeneration=0,
    )


def run(config: Mapping[str, Any]) -> None:
    experiment_root = Path(config["experiment_root"])
    experiment_root.mkdir(parents=True, exist_ok=True)
    status_path = _status_path(config)
    if status_path.exists():
        raise FileExistsError(f"refusing to overwrite training status: {status_path}")
    _atomic_json(
        status_path,
        {
            "experiment_id": config["experiment_id"],
            "status": "running",
            "state": "preflight",
            "started_at": _now(),
            "pid": os.getpid(),
            "architectures": list(ARCHITECTURES),
            "asap_test_midi_access_count": 0,
            "asap_test_evaluation_performed": False,
            "stage1_inference_regeneration": 0,
        },
    )
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("canonical training requires exactly one visible CUDA GPU")
        gpu = get_gpu_identity()
        if gpu["uuid"] != config["expected_gpu_uuid"] or gpu["name"] != config["expected_gpu_name"]:
            raise RuntimeError(f"unexpected assigned GPU: {gpu}")
        preflight_result = preflight(config)
        Path(config["output_root"]).mkdir(parents=True)
        _update_status(
            config,
            state="preflight_pass",
            preflight=preflight_result,
            shared_cache_verification_pass=True,
            canonical_validation_bank_verification_pass=True,
            gpu=gpu,
        )
        print(
            f"CACHE_PASS cache_id={preflight_result['cache_id']} "
            f"train_windows={preflight_result['train_windows']}",
            flush=True,
        )
        print(
            f"CANONICAL_BANK_PASS pieces={preflight_result['canonical']['pieces']} "
            f"js={preflight_result['canonical']['js_distance']:.15f}",
            flush=True,
        )
        print("PREFLIGHT_PASS", flush=True)
        trajectory_rows: list[dict[str, Any]] = []
        runs: dict[str, Mapping[str, Any]] = {}
        for architecture in ARCHITECTURES:
            _update_status(
                config,
                state=f"{architecture}_starting",
                current_architecture=architecture,
            )
            runs[architecture] = train_architecture(
                config,
                architecture,
                preflight_result,
                trajectory_rows,
            )
        _finalize(config, runs, preflight_result)
        print("RUNNER_COMPLETED", flush=True)
    except BaseException as error:
        failure = {
            "status": "failed",
            "state": "failed",
            "failed_at": _now(),
            "failed_stage": (
                _load_json(status_path).get("state", "unknown")
                if status_path.is_file()
                else "before_status"
            ),
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "asap_test_midi_access_count": 0,
            "asap_test_evaluation_performed": False,
        }
        if status_path.is_file():
            _update_status(config, **failure)
        print(f"RUNNER_FAILED error={error!r}", flush=True)
        print(traceback.format_exc(), flush=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/stage2_binary_canonical_training_v1.json",
    )
    parser.add_argument("--confirm-canonical-training", action="store_true")
    args = parser.parse_args()
    if not args.confirm_canonical_training:
        raise SystemExit("refusing to start without --confirm-canonical-training")
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    run(_load_json(config_path))


if __name__ == "__main__":
    main()
