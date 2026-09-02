#!/usr/bin/env python3
"""Fresh definitive binary Stage 2 training with strict MIDI validation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
from src.stage2_binary.strict_midi_validation import (  # noqa: E402
    STRICT_EVALUATOR_PROVENANCE,
    evaluate_strict_midi_roundtrip,
    strict_original_pt_reference,
)
from src.stage2_binary.training import build_binary_optimizer  # noqa: E402
from src.stage2_encoder_only.train import (  # noqa: E402
    atomic_torch_save,
    capture_rng_states,
    get_gpu_identity,
)
from src.stage2_encoder_only.training import set_deterministic_seed  # noqa: E402


MODEL_CLASSES = {
    "independent_4x2": IndependentBinaryPedalModel,
    "joint_16": JointBinaryPedalModel,
}
FIXED_KEYS = (
    "seed",
    "train_manifest",
    "asap_split_artifact",
    "asap_root",
    "cache_root",
    "checkpoint_path",
    "stage1_cache_manifest",
    "original_pt_validation_metrics",
    "architectures",
    "expected_train_source_performances",
    "expected_train_performances",
    "expected_train_notes",
    "expected_validation_performances",
    "expected_validation_pieces",
    "expected_validation_notes",
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
    "expected_gpu_uuid",
    "expected_gpu_name",
    "expected_gpu_memory_mib",
)
METRIC_COLUMNS = (
    "epoch",
    "train_loss",
    "validation_loss",
    "validation_binary_accuracy",
    "validation_exact_pattern_accuracy",
    "strict_validation_pedal_js_distance",
    "strict_validation_pedal_intersection",
    "direct_token_diagnostic_js_distance",
    "direct_token_diagnostic_intersection",
    "epoch_time_seconds",
    "global_optimizer_step",
    "amp_skipped_optimizer_steps",
    "maximum_consecutive_amp_skips",
    "encoder_learning_rate",
    "head_learning_rate",
    "peak_gpu_memory_bytes",
)
CHECKPOINT_MANIFEST_FIELDS = (
    "epoch",
    "checkpoint_path",
    "checkpoint_sha256",
    "checkpoint_bytes",
    "strict_validation_pedal_js_distance",
    "strict_validation_pedal_intersection",
    "direct_token_diagnostic_js_distance",
    "direct_token_diagnostic_intersection",
)
POSTHOC_FIELDS = (
    "epoch",
    "checkpoint_path",
    "recorded_strict_js_distance",
    "posthoc_strict_js_distance",
    "strict_js_absolute_difference",
    "recorded_strict_intersection",
    "posthoc_strict_intersection",
    "strict_intersection_absolute_difference",
    "recorded_direct_js_distance",
    "posthoc_direct_js_distance",
    "direct_js_absolute_difference",
    "recorded_direct_intersection",
    "posthoc_direct_intersection",
    "direct_intersection_absolute_difference",
    "nonpedal_midi_exact",
    "result",
)


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_csv(
    path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]
) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    dataset: SharedBinaryWindowDataset, order: Sequence[int], batch_size: int
) -> str:
    digest = hashlib.sha256()
    for index in order[:batch_size]:
        tensor = dataset[int(index)]["input_ids"].numpy().astype("<i8", copy=False)
        digest.update(tensor.tobytes())
    return digest.hexdigest()


def _fixed_config_check(config: Mapping[str, Any]) -> None:
    historical = _load_json(PROJECT_ROOT / "configs/stage2_binary_full_training_v0.json")
    mismatches = {
        key: (historical.get(key), config.get(key))
        for key in FIXED_KEYS
        if historical.get(key) != config.get(key)
    }
    if mismatches:
        raise RuntimeError(f"strict run changed a fixed training setting: {mismatches}")
    if config["output_root"] == historical["output_root"]:
        raise RuntimeError("strict run must not overwrite the historical output root")
    if int(config["gradient_accumulation_steps"]) != 1:
        raise RuntimeError("fixed gradient accumulation must remain one")


def _last_checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    config: Mapping[str, Any],
    epoch: int,
    global_step: int,
    stopping: PedalMetricEarlyStopping,
    amp_skipped_steps_total: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "grad_scaler_state": scaler.state_dict(),
        "configuration": dict(config),
        "completed_epoch": epoch,
        "global_optimizer_step": global_step,
        "best_epoch": stopping.best_epoch,
        "best_strict_validation_js_distance": stopping.best_js_distance,
        "best_strict_validation_intersection": stopping.best_intersection,
        "early_stopping_counter": stopping.counter,
        "amp_skipped_optimizer_steps_total": amp_skipped_steps_total,
    }
    payload.update(capture_rng_states())
    return payload


def _epoch_checkpoint_payload(
    *,
    model: torch.nn.Module,
    config: Mapping[str, Any],
    row: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "model_state": model.state_dict(),
        "configuration": dict(config),
        "epoch": int(row["epoch"]),
        "strict_validation_js_distance": float(
            row["strict_validation_pedal_js_distance"]
        ),
        "strict_validation_intersection": float(
            row["strict_validation_pedal_intersection"]
        ),
        "direct_token_diagnostic_js_distance": float(
            row["direct_token_diagnostic_js_distance"]
        ),
        "direct_token_diagnostic_intersection": float(
            row["direct_token_diagnostic_intersection"]
        ),
    }


def _posthoc_verify(
    *,
    model: torch.nn.Module,
    architecture: str,
    checkpoint_rows: Sequence[Mapping[str, Any]],
    metric_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    human_histogram: Sequence[int],
    device: torch.device,
    output_dir: Path,
    log: Any,
) -> list[dict[str, Any]]:
    metrics_by_epoch = {int(row["epoch"]): row for row in metric_rows}
    posthoc: list[dict[str, Any]] = []
    for item in checkpoint_rows:
        epoch = int(item["epoch"])
        checkpoint_path = Path(str(item["checkpoint_path"]))
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state"], strict=True)
        del payload
        distribution = evaluate_strict_midi_roundtrip(
            model,
            architecture=architecture,
            stage1_manifest_csv=config["stage1_cache_manifest"],
            human_histogram=human_histogram,
            device=device,
            temporary_root=output_dir / "strict_validation_tmp",
        )
        recorded = metrics_by_epoch[epoch]
        differences = {
            "strict_js": abs(
                float(recorded["strict_validation_pedal_js_distance"])
                - distribution["strict_js_distance"]
            ),
            "strict_intersection": abs(
                float(recorded["strict_validation_pedal_intersection"])
                - distribution["strict_intersection"]
            ),
            "direct_js": abs(
                float(recorded["direct_token_diagnostic_js_distance"])
                - distribution["direct_js_distance"]
            ),
            "direct_intersection": abs(
                float(recorded["direct_token_diagnostic_intersection"])
                - distribution["direct_intersection"]
            ),
        }
        passed = max(differences.values()) <= float(config["js_tie_tolerance"])
        if not passed or not distribution["nonpedal_midi_exact"]:
            raise RuntimeError(f"post-hoc strict metric mismatch at epoch {epoch}")
        row = {
            "epoch": epoch,
            "checkpoint_path": str(checkpoint_path),
            "recorded_strict_js_distance": recorded[
                "strict_validation_pedal_js_distance"
            ],
            "posthoc_strict_js_distance": distribution["strict_js_distance"],
            "strict_js_absolute_difference": differences["strict_js"],
            "recorded_strict_intersection": recorded[
                "strict_validation_pedal_intersection"
            ],
            "posthoc_strict_intersection": distribution["strict_intersection"],
            "strict_intersection_absolute_difference": differences[
                "strict_intersection"
            ],
            "recorded_direct_js_distance": recorded[
                "direct_token_diagnostic_js_distance"
            ],
            "posthoc_direct_js_distance": distribution["direct_js_distance"],
            "direct_js_absolute_difference": differences["direct_js"],
            "recorded_direct_intersection": recorded[
                "direct_token_diagnostic_intersection"
            ],
            "posthoc_direct_intersection": distribution["direct_intersection"],
            "direct_intersection_absolute_difference": differences[
                "direct_intersection"
            ],
            "nonpedal_midi_exact": distribution["nonpedal_midi_exact"],
            "result": "PASS",
        }
        posthoc.append(row)
        log(f"POSTHOC epoch={epoch} strict_js={distribution['strict_js_distance']:.15f} PASS")
    _write_csv(output_dir / "posthoc_strict_metrics.csv", POSTHOC_FIELDS, posthoc)
    return posthoc


def train(config: dict[str, Any], architecture: str) -> None:
    if architecture not in MODEL_CLASSES:
        raise ValueError(f"unknown architecture: {architecture}")
    _fixed_config_check(config)
    output_dir = Path(config["output_root"]) / architecture
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite strict run: {output_dir}")
    output_dir.mkdir(parents=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir()
    log_handle = (output_dir / "train.log").open("w", encoding="utf-8")

    def log(message: str) -> None:
        print(message, flush=True)
        print(message, file=log_handle, flush=True)

    status: dict[str, Any] = {
        "architecture": architecture,
        "status": "running",
        "fresh_start_from_official_pretrained_encoder": True,
        "strict_midi_roundtrip_selection": True,
        "stage1_inference_regeneration": 0,
        "asap_test_access_count": 0,
    }
    _write_json(output_dir / "run_status.json", status)
    try:
        set_deterministic_seed(int(config["seed"]))
        gpu = get_gpu_identity()
        if gpu["uuid"] != config["expected_gpu_uuid"]:
            raise RuntimeError(f"unexpected GPU: {gpu}")
        device = torch.device("cuda:0")
        train_dataset = SharedBinaryWindowDataset(config["cache_root"], "train")
        validation_dataset = SharedBinaryWindowDataset(
            config["cache_root"], "validation"
        )
        if train_dataset.cache_id != config["expected_cache_id"]:
            raise RuntimeError("strict run cache ID mismatch")
        if train_dataset.cache_id != validation_dataset.cache_id:
            raise RuntimeError("train/validation cache IDs differ")
        if len(train_dataset) != int(config["expected_train_windows"]):
            raise RuntimeError("training window count mismatch")
        if len(validation_dataset) != int(config["expected_validation_windows"]):
            raise RuntimeError("validation window count mismatch")
        epoch_orders = {
            str(epoch): order_sha256(
                training_window_order(len(train_dataset), int(config["seed"]), epoch)
            )
            for epoch in range(1, int(config["max_epochs"]) + 1)
        }
        first_order = training_window_order(
            len(train_dataset), int(config["seed"]), 1
        )
        first_batch_hash = _input_hash(
            train_dataset, first_order, int(config["batch_size"])
        )
        if epoch_orders["1"] != config["expected_epoch1_window_order_sha256"]:
            raise RuntimeError("epoch-1 window order hash differs from historical run")
        if first_batch_hash != config["expected_first_training_batch_input_sha256"]:
            raise RuntimeError("first training batch hash differs from historical setup")

        model = MODEL_CLASSES[architecture].from_pretrained(
            config["checkpoint_path"],
            torch_dtype=torch.float32,
            attn_implementation="eager",
        )
        initial_encoder_hash = _parameter_hash(model.encoder.parameters())
        if initial_encoder_hash != config["expected_initial_encoder_sha256"]:
            raise RuntimeError("initial pretrained encoder hash mismatch")
        run_config = dict(config)
        run_config.update(
            architecture=architecture,
            output_dir=str(output_dir),
            full_training_started=True,
            cache_id=train_dataset.cache_id,
            epoch_window_order_sha256=epoch_orders,
            first_training_batch_input_sha256=first_batch_hash,
            initial_encoder_parameter_sha256=initial_encoder_hash,
            parameter_count=model.parameter_count,
            trainable_parameter_count=model.trainable_parameter_count,
            encoder_initialization="official Pianist Transformer pretrained encoder",
            strict_evaluator_provenance=STRICT_EVALUATOR_PROVENANCE,
            strict_runner_source_sha256=_sha256(Path(__file__)),
            strict_validation_source_sha256=_sha256(
                PROJECT_ROOT / "src/stage2_binary/strict_midi_validation.py"
            ),
        )
        _write_json(output_dir / "config.json", run_config)
        log(
            f"START architecture={architecture} gpu={gpu['name']} cache={train_dataset.cache_id} "
            f"train_windows={len(train_dataset)} strict_validation=True"
        )
        log(
            f"REPRO encoder={initial_encoder_hash} epoch1_order={epoch_orders['1']} "
            f"first_batch={first_batch_hash}"
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
        human_histogram = _load_json(config["original_pt_validation_metrics"])[
            "human_histogram"
        ]
        original_reference = strict_original_pt_reference(
            stage1_manifest_csv=config["stage1_cache_manifest"],
            human_histogram=human_histogram,
        )
        if (
            abs(
                original_reference["strict_js_distance"]
                - float(config["original_pt_strict_validation_js_distance"])
            )
            > float(config["js_tie_tolerance"])
            or abs(
                original_reference["strict_intersection"]
                - float(config["original_pt_strict_validation_intersection"])
            )
            > float(config["js_tie_tolerance"])
        ):
            raise RuntimeError("Original PT strict baseline did not reproduce")
        log(
            "STRICT_BASELINE "
            f"js={original_reference['strict_js_distance']:.15f} "
            f"intersection={original_reference['strict_intersection']:.15f}"
        )

        global_step = 0
        amp_skipped_total = 0
        metrics_rows: list[dict[str, Any]] = []
        checkpoint_rows: list[dict[str, Any]] = []
        metrics_path = output_dir / "metrics.csv"
        with metrics_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=METRIC_COLUMNS)
            writer.writeheader()
            for epoch in range(1, int(config["max_epochs"]) + 1):
                epoch_started = time.perf_counter()
                torch.cuda.reset_peak_memory_stats(device)
                order = training_window_order(
                    len(train_dataset), int(config["seed"]), epoch
                )
                train_loader = make_binary_loader(
                    train_dataset,
                    batch_size=int(config["batch_size"]),
                    pin_memory=bool(config["pin_memory"]),
                    order=order,
                )
                previous_skip_count = 0

                def batch_callback(batch: int, details: Mapping[str, Any]) -> None:
                    nonlocal previous_skip_count
                    current_skips = int(details["amp_skipped_steps"])
                    if batch == 1:
                        log(
                            f"FIRST_BATCH epoch={epoch} loss={details['loss']:.9f} "
                            f"finite={math.isfinite(float(details['loss']))} "
                            f"amp_scale={details['amp_scale_after']}"
                        )
                    if current_skips > previous_skip_count:
                        log(
                            f"AMP_SKIP epoch={epoch} batch={batch} "
                            f"scale_before={details['amp_scale_before']} "
                            f"scale_after={details['amp_scale_after']}"
                        )
                    previous_skip_count = current_skips

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
                    max_consecutive_amp_skips=int(
                        config["max_consecutive_amp_skips"]
                    ),
                )
                global_step += int(train_checks["optimizer_steps"])
                amp_skipped_total += int(train_checks["amp_skipped_steps"])
                validation_metrics, _ = run_binary_epoch(
                    model,
                    validation_loader,
                    architecture=architecture,
                    device=device,
                    amp_enabled=bool(config["amp_enabled"]),
                )
                distribution = evaluate_strict_midi_roundtrip(
                    model,
                    architecture=architecture,
                    stage1_manifest_csv=config["stage1_cache_manifest"],
                    human_histogram=human_histogram,
                    device=device,
                    temporary_root=output_dir / "strict_validation_tmp",
                )
                improved, should_stop = stopping.update(
                    distribution["strict_js_distance"],
                    distribution["strict_intersection"],
                    epoch,
                )
                row = {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "validation_loss": validation_metrics["loss"],
                    "validation_binary_accuracy": validation_metrics[
                        "binary_accuracy"
                    ],
                    "validation_exact_pattern_accuracy": validation_metrics[
                        "exact_pattern_accuracy"
                    ],
                    "strict_validation_pedal_js_distance": distribution[
                        "strict_js_distance"
                    ],
                    "strict_validation_pedal_intersection": distribution[
                        "strict_intersection"
                    ],
                    "direct_token_diagnostic_js_distance": distribution[
                        "direct_js_distance"
                    ],
                    "direct_token_diagnostic_intersection": distribution[
                        "direct_intersection"
                    ],
                    "epoch_time_seconds": time.perf_counter() - epoch_started,
                    "global_optimizer_step": global_step,
                    "amp_skipped_optimizer_steps": train_checks[
                        "amp_skipped_steps"
                    ],
                    "maximum_consecutive_amp_skips": train_checks[
                        "maximum_consecutive_amp_skips"
                    ],
                    "encoder_learning_rate": optimizer.param_groups[0]["lr"],
                    "head_learning_rate": optimizer.param_groups[1]["lr"],
                    "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
                }
                if not all(
                    math.isfinite(float(row[key]))
                    for key in METRIC_COLUMNS
                    if key != "epoch"
                ):
                    raise FloatingPointError("non-finite strict epoch metric")
                writer.writerow(row)
                handle.flush()
                os.fsync(handle.fileno())
                metrics_rows.append(row)

                epoch_path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
                atomic_torch_save(
                    _epoch_checkpoint_payload(
                        model=model,
                        config=run_config,
                        row=row,
                    ),
                    epoch_path,
                )
                checkpoint_row = {
                    "epoch": epoch,
                    "checkpoint_path": str(epoch_path),
                    "checkpoint_sha256": _sha256(epoch_path),
                    "checkpoint_bytes": epoch_path.stat().st_size,
                    "strict_validation_pedal_js_distance": distribution[
                        "strict_js_distance"
                    ],
                    "strict_validation_pedal_intersection": distribution[
                        "strict_intersection"
                    ],
                    "direct_token_diagnostic_js_distance": distribution[
                        "direct_js_distance"
                    ],
                    "direct_token_diagnostic_intersection": distribution[
                        "direct_intersection"
                    ],
                }
                checkpoint_rows.append(checkpoint_row)
                _write_csv(
                    output_dir / "epoch_checkpoint_manifest.csv",
                    CHECKPOINT_MANIFEST_FIELDS,
                    checkpoint_rows,
                )
                atomic_torch_save(
                    _last_checkpoint_payload(
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        config=run_config,
                        epoch=epoch,
                        global_step=global_step,
                        stopping=stopping,
                        amp_skipped_steps_total=amp_skipped_total,
                    ),
                    output_dir / "last.pt",
                )
                if improved:
                    atomic_torch_save(
                        {
                            "model_state": model.state_dict(),
                            "configuration": run_config,
                            "best_epoch": epoch,
                            "best_strict_validation_js_distance": distribution[
                                "strict_js_distance"
                            ],
                            "best_strict_validation_intersection": distribution[
                                "strict_intersection"
                            ],
                        },
                        output_dir / "best.pt",
                    )
                log(json.dumps(row, sort_keys=True))
                if should_stop:
                    log(
                        f"EARLY_STOP epoch={epoch} strict_best_epoch={stopping.best_epoch}"
                    )
                    break

        posthoc = _posthoc_verify(
            model=model,
            architecture=architecture,
            checkpoint_rows=checkpoint_rows,
            metric_rows=metrics_rows,
            config=config,
            human_histogram=human_histogram,
            device=device,
            output_dir=output_dir,
            log=log,
        )
        recomputed = PedalMetricEarlyStopping(
            patience=int(config["early_stopping_patience"]),
            tie_tolerance=float(config["js_tie_tolerance"]),
        )
        for row in metrics_rows:
            recomputed.update(
                float(row["strict_validation_pedal_js_distance"]),
                float(row["strict_validation_pedal_intersection"]),
                int(row["epoch"]),
            )
        if recomputed.best_epoch != stopping.best_epoch:
            raise RuntimeError("post-hoc strict best epoch differs from training selection")
        status.update(
            status="completed",
            completed_epochs=len(metrics_rows),
            early_stopping_epoch=int(metrics_rows[-1]["epoch"]),
            strict_best_epoch=stopping.best_epoch,
            strict_best_validation_js_distance=stopping.best_js_distance,
            strict_best_validation_intersection=stopping.best_intersection,
            amp_skipped_optimizer_steps_total=amp_skipped_total,
            maximum_consecutive_amp_skips=max(
                int(row["maximum_consecutive_amp_skips"]) for row in metrics_rows
            ),
            epoch_checkpoints_saved=len(checkpoint_rows),
            posthoc_epoch_metrics_verified=len(posthoc),
            posthoc_tolerance=float(config["js_tie_tolerance"]),
            initial_encoder_parameter_sha256=initial_encoder_hash,
            epoch1_window_order_sha256=epoch_orders["1"],
            first_training_batch_input_sha256=first_batch_hash,
            cache_id=train_dataset.cache_id,
            stage1_inference_regeneration=0,
            asap_test_access_count=0,
        )
        _write_json(output_dir / "run_status.json", status)
        log(
            f"COMPLETE architecture={architecture} epochs={len(metrics_rows)} "
            f"strict_best_epoch={stopping.best_epoch} "
            f"strict_js={stopping.best_js_distance:.15f}"
        )
    except BaseException as error:
        status.update(status="failed", error=repr(error))
        _write_json(output_dir / "run_status.json", status)
        raise
    finally:
        log_handle.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/stage2_binary_strict_full_training_v1.json"
    )
    parser.add_argument("--architecture", required=True, choices=tuple(MODEL_CLASSES))
    parser.add_argument("--confirm-definitive-training", action="store_true")
    args = parser.parse_args()
    if not args.confirm_definitive_training:
        raise SystemExit("refusing to start without --confirm-definitive-training")
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    train(_load_json(config_path), args.architecture)


if __name__ == "__main__":
    main()
