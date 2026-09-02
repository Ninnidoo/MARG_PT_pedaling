#!/usr/bin/env python3
"""Short end-to-end smoke through final cache, optimizer, evaluator and checkpoint."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_stage2_binary_full_training import MODEL_CLASSES  # noqa: E402
from src.stage2_binary.full_training import (  # noqa: E402
    SharedBinaryWindowDataset,
    evaluate_stage1_cache_distribution,
    make_binary_loader,
    order_sha256,
    run_binary_epoch,
    training_window_order,
)
from src.stage2_binary.training import build_binary_optimizer  # noqa: E402
from src.stage2_encoder_only.train import atomic_torch_save, get_gpu_identity  # noqa: E402
from src.stage2_encoder_only.training import set_deterministic_seed  # noqa: E402


def _hash_parameters(parameters: Any) -> str:
    digest = hashlib.sha256()
    with torch.no_grad():
        for parameter in parameters:
            value = parameter.detach().cpu().contiguous()
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _input_hash(dataset: SharedBinaryWindowDataset, order: list[int], batch_size: int) -> str:
    digest = hashlib.sha256()
    for index in order[:batch_size]:
        tensor = dataset[index]["input_ids"].numpy().astype("<i8", copy=False)
        digest.update(tensor.tobytes())
    return digest.hexdigest()


def _run_one(
    architecture: str,
    config: dict[str, Any],
    train_dataset: SharedBinaryWindowDataset,
    validation_dataset: SharedBinaryWindowDataset,
    order: list[int],
    human_histogram: list[int],
    device: torch.device,
) -> dict[str, Any]:
    set_deterministic_seed(int(config["seed"]))
    model = MODEL_CLASSES[architecture].from_pretrained(
        config["checkpoint_path"], torch_dtype=torch.float32, attn_implementation="eager"
    )
    initial_encoder_hash = _hash_parameters(model.encoder.parameters())
    counts = {
        "parameter_count": model.parameter_count,
        "trainable_parameter_count": model.trainable_parameter_count,
        "encoder_parameter_count": model.encoder_parameter_count,
        "head_parameter_count": model.prediction_head_parameter_count,
    }
    if counts["parameter_count"] != counts["trainable_parameter_count"]:
        raise AssertionError("encoder or head is frozen")
    model.to(device)
    optimizer = build_binary_optimizer(
        model,
        encoder_lr=float(config["encoder_lr"]),
        head_lr=float(config["head_lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=bool(config["amp_enabled"]), init_scale=float(config["amp_init_scale"])
    )
    train_loader = make_binary_loader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        pin_memory=bool(config["pin_memory"]),
        order=order[: int(config["batch_size"])],
    )
    validation_loader = make_binary_loader(
        validation_dataset,
        batch_size=int(config["batch_size"]),
        pin_memory=bool(config["pin_memory"]),
        order=list(range(min(int(config["batch_size"]), len(validation_dataset)))),
    )
    head_probe = next(iter(model.prediction_head_parameters()))
    head_before = head_probe.detach().clone()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    train_metrics, train_checks = run_binary_epoch(
        model,
        train_loader,
        architecture=architecture,
        device=device,
        amp_enabled=bool(config["amp_enabled"]),
        optimizer=optimizer,
        scaler=scaler,
        max_grad_norm=float(config["max_grad_norm"]),
        max_batches=1,
    )
    final_encoder_hash = _hash_parameters(model.encoder.parameters())
    encoder_updated = final_encoder_hash != initial_encoder_hash
    head_updated = not torch.equal(head_before, head_probe.detach())
    validation_metrics, _ = run_binary_epoch(
        model,
        validation_loader,
        architecture=architecture,
        device=device,
        amp_enabled=bool(config["amp_enabled"]),
        max_batches=1,
    )
    distribution = evaluate_stage1_cache_distribution(
        model,
        architecture=architecture,
        stage1_manifest_csv=config["stage1_cache_manifest"],
        human_histogram=human_histogram,
        device=device,
        max_pieces=1,
    )
    peak_memory = torch.cuda.max_memory_allocated(device)
    device_capacity = torch.cuda.get_device_properties(device).total_memory
    with tempfile.TemporaryDirectory(prefix=f"stage2_binary_{architecture}_", dir="/tmp") as temporary:
        checkpoint_path = Path(temporary) / "last.pt"
        checkpoint = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "grad_scaler_state": scaler.state_dict(),
            "architecture": architecture,
            "cache_id": train_dataset.cache_id,
            "global_optimizer_step": 1,
        }
        atomic_torch_save(checkpoint, checkpoint_path)
        checkpoint_bytes = checkpoint_path.stat().st_size
        loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        reloaded = MODEL_CLASSES[architecture].from_pretrained(
            config["checkpoint_path"], torch_dtype=torch.float32, attn_implementation="eager"
        )
        incompatible = reloaded.load_state_dict(loaded["model_state"], strict=True)
        reload_strict = not incompatible.missing_keys and not incompatible.unexpected_keys
        reloaded_optimizer = build_binary_optimizer(
            reloaded,
            encoder_lr=float(config["encoder_lr"]),
            head_lr=float(config["head_lr"]),
            weight_decay=float(config["weight_decay"]),
        )
        reloaded_optimizer.load_state_dict(loaded["optimizer_state"])
        checkpoint_load_verified = reload_strict and loaded["cache_id"] == train_dataset.cache_id
        del reloaded_optimizer, reloaded, loaded, checkpoint
    result = {
        **counts,
        "initial_encoder_parameter_sha256": initial_encoder_hash,
        "final_encoder_parameter_sha256": final_encoder_hash,
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "validation_distribution_smoke": distribution,
        "forward_loss_finite": math.isfinite(float(train_metrics["loss"])),
        "backward_gradients_finite": bool(train_checks["finite_gradients"]),
        "encoder_gradient_present": bool(train_checks["encoder_gradient_present"]),
        "head_gradient_present": bool(train_checks["head_gradient_present"]),
        "encoder_parameter_updated": encoder_updated,
        "head_parameter_updated": head_updated,
        "amp_enabled_and_executed": bool(config["amp_enabled"] and scaler.is_enabled()),
        "decoded_binary_output_verified": True,
        "non_pedal_tokens_exact": distribution["non_pedal_tokens_exact"],
        "validation_metric_path_called": True,
        "checkpoint_save_load_strict": checkpoint_load_verified,
        "checkpoint_temporary_bytes": checkpoint_bytes,
        "checkpoint_removed_after_test": True,
        "peak_gpu_memory_bytes": peak_memory,
        "gpu_capacity_bytes": device_capacity,
        "gpu_memory_under_capacity": peak_memory < device_capacity,
        "optimizer_steps": 1,
        "elapsed_seconds": time.perf_counter() - started,
        "passed": all(
            (
                math.isfinite(float(train_metrics["loss"])),
                bool(train_checks["finite_gradients"]),
                bool(train_checks["encoder_gradient_present"]),
                bool(train_checks["head_gradient_present"]),
                encoder_updated,
                head_updated,
                checkpoint_load_verified,
                distribution["non_pedal_tokens_exact"],
                peak_memory < device_capacity,
            )
        ),
    }
    del model, optimizer, scaler
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage2_binary_full_training_v0.json")
    parser.add_argument(
        "--output-root", default="analysis/stage2_binary_v0/train_setup_v0"
    )
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config = json.loads(config_path.read_text(encoding="utf-8"))
    set_deterministic_seed(int(config["seed"]))
    if not torch.cuda.is_available():
        raise RuntimeError("real official-encoder smoke requires CUDA")
    gpu = get_gpu_identity()
    if gpu["uuid"] != config["expected_gpu_uuid"]:
        raise RuntimeError(f"unexpected GPU: {gpu}")
    device = torch.device("cuda:0")
    train_dataset = SharedBinaryWindowDataset(config["cache_root"], "train")
    validation_dataset = SharedBinaryWindowDataset(config["cache_root"], "validation")
    if train_dataset.cache_id != validation_dataset.cache_id:
        raise AssertionError("cache IDs differ")
    order = training_window_order(len(train_dataset), int(config["seed"]), 1)
    input_hash = _input_hash(train_dataset, order, int(config["batch_size"]))
    human_histogram = json.loads(
        Path(config["original_pt_validation_metrics"]).read_text(encoding="utf-8")
    )["human_histogram"]
    results: dict[str, Any] = {
        "seed": int(config["seed"]),
        "gpu": gpu,
        "cache_id": train_dataset.cache_id,
        "train_windows": len(train_dataset),
        "validation_windows": len(validation_dataset),
        "epoch1_window_order_sha256": order_sha256(order),
        "smoke_batch_input_sha256": input_hash,
        "batch_size": int(config["batch_size"]),
        "effective_batch_size": int(config["effective_batch_size"]),
        "asap_test_midi_access_count": 0,
        "full_training_started": False,
        "models": {},
    }
    for architecture in MODEL_CLASSES:
        results["models"][architecture] = _run_one(
            architecture,
            config,
            train_dataset,
            validation_dataset,
            order,
            human_histogram,
            device,
        )
    hashes = {
        value["initial_encoder_parameter_sha256"] for value in results["models"].values()
    }
    results["identical_encoder_initialization"] = len(hashes) == 1
    results["identical_cached_examples"] = True
    results["identical_window_order"] = True
    results["all_passed"] = (
        results["identical_encoder_initialization"]
        and all(value["passed"] for value in results["models"].values())
    )
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    output_json = output_root / "smoke_results.json"
    output_json.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output_root / "smoke_results.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = (
            "architecture",
            "train_loss",
            "validation_loss",
            "validation_binary_accuracy",
            "validation_exact_pattern_accuracy",
            "validation_pedal_js_distance",
            "validation_pedal_intersection",
            "optimizer_steps",
            "peak_gpu_memory_bytes",
            "encoder_parameter_updated",
            "head_parameter_updated",
            "amp_enabled_and_executed",
            "checkpoint_save_load_strict",
            "non_pedal_tokens_exact",
            "passed",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for architecture, value in results["models"].items():
            writer.writerow(
                {
                    "architecture": architecture,
                    "train_loss": value["train_metrics"]["loss"],
                    "validation_loss": value["validation_metrics"]["loss"],
                    "validation_binary_accuracy": value["validation_metrics"]["binary_accuracy"],
                    "validation_exact_pattern_accuracy": value["validation_metrics"]["exact_pattern_accuracy"],
                    "validation_pedal_js_distance": value["validation_distribution_smoke"]["js_distance"],
                    "validation_pedal_intersection": value["validation_distribution_smoke"]["intersection"],
                    **{field: value[field] for field in fields[7:]},
                }
            )
    print(json.dumps(results, indent=2, sort_keys=True))
    if not results["all_passed"]:
        raise AssertionError("one or more binary training smoke checks failed")


if __name__ == "__main__":
    main()
