"""Run the controlled exact scheduled-sampling encoder-decoder experiment."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import socket
import sys
import traceback
import unittest
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from src.stage2_encoder_only.run_five_class import (
    AtomicRunStatus,
    Tee,
    acquire_run_lock,
    atomic_json,
    initial_run_status,
    kst_now,
    release_run_lock,
    utc_now,
)
from src.stage2_encoder_only.train import get_gpu_identity

from .evaluate import FREE_MODEL, evaluate
from .scheduled_sampling_experiment import analyze_and_report, benchmark
from .train import (
    apply_train_only_weights,
    default_configuration,
    preload_datasets,
    prepare_preflight,
    train,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REFERENCE = PROJECT_ROOT / "analysis" / "stage2_encoder_decoder_5class_weighted_v0"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "analysis"
    / "stage2_encoder_decoder_5class_weighted_scheduled_sampling_v0"
)
DEFAULT_TMUX = "stage2-encoder-decoder-5class-weighted-scheduled-sampling-v0"
TEST_MODULES = (
    "tests.test_stage2_encoder_decoder_scheduled_sampling",
    "tests.test_stage2_encoder_decoder_five_class",
    "tests.test_stage2_weighted_five_class",
    "tests.test_stage2_five_class",
    "tests.test_stage2_five_class_runtime",
)
INITIAL_HASH_KEYS = (
    "encoder_initial_sha256",
    "decoder_initial_sha256",
    "slot_embedding_initial_sha256",
    "output_head_initial_sha256",
)
CONTROLLED_KEYS = (
    "asap_root", "split_csv", "checkpoint_path", "seed", "decoder_init_seed",
    "effective_batch_size", "micro_batch_size", "gradient_accumulation_steps",
    "max_epochs", "early_stopping_patience", "early_stopping_min_delta",
    "encoder_lr", "decoder_lr", "weight_decay", "max_grad_norm", "amp_enabled",
    "amp_init_scale", "window_notes", "stride_notes", "num_workers", "pin_memory",
    "progress_interval", "expected_gpu_uuid", "expected_train_performances",
    "expected_validation_performances", "expected_train_pieces",
    "expected_validation_pieces", "class_names", "class_boundaries",
    "representatives", "architecture", "loss_configuration", "model_selection",
    "pipeline_splits", "test_split_passed_to_pipeline",
)
SCHEDULE = {
    "epoch_1": 1.0,
    "epoch_2": 1.0,
    "epoch_3": 0.9,
    "epoch_4": 0.8,
    "epoch_5": 0.7,
    "epoch_6_and_later": 0.6,
}


def _configuration(output: Path) -> dict[str, Any]:
    reference = json.loads((REFERENCE / "config.json").read_text(encoding="utf-8"))
    configuration = default_configuration(output)
    for key in CONTROLLED_KEYS:
        if key not in reference:
            raise KeyError(f"reference configuration missing controlled key: {key}")
        configuration[key] = reference[key]
    configuration.update(
        output_dir=str(output),
        early_stopping_enabled=True,
        scheduled_sampling_enabled=True,
        scheduled_sampling_seed=int(reference["seed"]),
        teacher_forcing_schedule=SCHEDULE,
        scheduled_sampling_semantics=(
            "exact detached left-to-right greedy mixed-history rollout; native KV cache; "
            "causal parallel gradient pass conditional on that history"
        ),
        scheduled_sampling_decision_granularity="independent per decoder step and sample",
        scheduled_sampling_prediction="greedy argmax detached",
        resume=None,
        reference_run=str(REFERENCE),
        experiment="encoder_decoder_weighted_5class_scheduled_sampling",
        scheduler=reference.get("scheduler"),
        free_running_validation=True,
        validation_modes=["teacher_forced", "greedy_free_running"],
        midi_inference=False,
        rendering=False,
        reference_initial_hashes={key: reference[key] for key in INITIAL_HASH_KEYS},
    )
    assertions = {
        "seed": 42,
        "decoder_init_seed": 42,
        "effective_batch_size": 16,
        "micro_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "max_epochs": 20,
        "window_notes": 512,
        "stride_notes": 256,
        "early_stopping_patience": 4,
        "early_stopping_min_delta": 1e-4,
    }
    for key, expected in assertions.items():
        if configuration[key] != expected:
            raise RuntimeError(
                f"reference controlled setting differs for {key}: "
                f"expected={expected!r} observed={configuration[key]!r}"
            )
    if configuration.get("scheduler") is not None:
        raise RuntimeError("reference scheduler unexpectedly changed")
    return configuration


def run_tests(output: Path) -> dict[str, Any]:
    suite = unittest.TestLoader().loadTestsFromNames(TEST_MODULES)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    payload = {
        "passed": result.wasSuccessful(),
        "total_tests": result.testsRun,
        "failures": [f"{test.id()}: {detail}" for test, detail in result.failures],
        "errors": [f"{test.id()}: {detail}" for test, detail in result.errors],
        "skipped": [f"{test.id()}: {reason}" for test, reason in result.skipped],
        "modules": list(TEST_MODULES),
        "asap_test_split_accessed": False,
        "excluded_real_dataset_test": "tests.test_stage2_dataset",
    }
    atomic_json(output / "tests" / "test_results.json", payload)
    if not result.wasSuccessful():
        raise RuntimeError("scheduled-sampling correctness/regression tests failed")
    return payload


def run(output_dir: str | Path) -> int:
    output = Path(output_dir).resolve()
    if not (REFERENCE / "run_status.json").is_file():
        raise FileNotFoundError("completed encoder-decoder reference run is missing")
    reference_status = json.loads((REFERENCE / "run_status.json").read_text(encoding="utf-8"))
    if not reference_status.get("completed") or reference_status.get("stage") != "complete":
        raise RuntimeError("reference encoder-decoder run is not complete")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite scheduled-sampling output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "tests").mkdir(exist_ok=True)
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:12]}"
    tmux_name = os.environ.get("STAGE2_TMUX_SESSION", DEFAULT_TMUX)
    lock = acquire_run_lock(output, run_id, container_pid=os.getpid())
    status = AtomicRunStatus(
        output / "run_status.json",
        initial_run_status(run_id, None, os.getpid(), tmux_name),
    )
    log_path = output / "train.log"
    outcome = "failed"
    failure_stage = "initializing"
    try:
        with log_path.open("a", encoding="utf-8") as log:
            with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
                configuration = _configuration(output)
                gpu = get_gpu_identity()
                if torch.cuda.device_count() != 1 or gpu["uuid"] != configuration["expected_gpu_uuid"]:
                    raise RuntimeError(f"assigned single-GPU runtime mismatch: {gpu}")
                configuration.update(
                    gpu_uuid=gpu["uuid"], gpu_name=gpu["name"],
                    container_visible_cuda_index=0,
                    pytorch_version=torch.__version__, cuda_version=torch.version.cuda,
                    tmux_session_name=tmux_name, container_hostname=socket.gethostname(),
                    run_id=run_id,
                )
                atomic_json(output / "config.json", configuration)
                atomic_json(output / "scheduled_sampling_config.json", {
                    "enabled": True,
                    "schedule": SCHEDULE,
                    "seed": configuration["scheduled_sampling_seed"],
                    "decision_granularity": configuration["scheduled_sampling_decision_granularity"],
                    "prediction": configuration["scheduled_sampling_prediction"],
                    "semantics": configuration["scheduled_sampling_semantics"],
                    "future_ground_truth_visible": False,
                    "decoder_dropout": 0.0,
                    "asap_test_split_accessed": False,
                })

                failure_stage = "testing"
                status.update(stage="testing", gpu_uuid=gpu["uuid"])
                tests = run_tests(output)

                failure_stage = "preloading"
                status.update(stage="preloading")

                def progress(update: Mapping[str, Any]) -> None:
                    mapped = "preloading" if update.get("stage") == "preloading" else "training"
                    status.update(
                        stage=mapped,
                        current_epoch=int(update.get("epoch", status.payload.get("current_epoch", 0))),
                        current_step=int(update.get("global_step", status.payload.get("current_step", 0))),
                        latest_finite_loss=update.get("loss", status.payload.get("latest_finite_loss")),
                        current_batch=update.get("batch"), total_batches=update.get("batches"),
                        preload_split=update.get("split"),
                        teacher_forcing_probability=update.get("teacher_forcing_probability"),
                        actual_ground_truth_history_ratio=update.get("ground_truth_history_ratio"),
                        actual_predicted_history_ratio=update.get("predicted_history_ratio"),
                    )

                train_dataset, validation_dataset = preload_datasets(configuration, progress)
                weights = apply_train_only_weights(configuration, train_dataset, output)
                reference_weights = json.loads((REFERENCE / "class_weights.json").read_text(encoding="utf-8"))
                if weights["class_counts"] != reference_weights["class_counts"] or not np.allclose(
                    weights["weights"], reference_weights["weights"], rtol=0.0, atol=0.0
                ):
                    raise RuntimeError("train-only class weights differ from reference run")
                atomic_json(output / "config.json", configuration)

                failure_stage = "overfit"
                status.update(stage="overfit", preflight_phase="gpu_memory_and_pedal_rich_overfit")
                memory_smoke, overfit = prepare_preflight(configuration, train_dataset, output)
                if int(configuration["micro_batch_size"]) != 4:
                    raise RuntimeError("GPU smoke selected a different micro-batch than reference")
                atomic_json(output / "config.json", configuration)

                failure_stage = "benchmarking"
                status.update(stage="overfit", preflight_phase="scheduled_sampling_runtime_benchmark")
                runtime = benchmark(configuration, train_dataset, output, REFERENCE)

                failure_stage = "training"
                status.update(stage="training", benchmark_passed=True)
                training = train(
                    configuration, train_dataset, validation_dataset,
                    progress_callback=progress,
                )
                status.update(
                    stage="validation", current_epoch=training["completed_epoch"],
                    current_step=training["global_optimizer_step"],
                    best_epoch=training["best_epoch"],
                    best_validation_loss=training["best_validation_loss"],
                    early_stopped=training["early_stopped"],
                )

                failure_stage = "validation"
                evaluation = evaluate(output)
                if evaluation["split"] != "validation" or evaluation["test_rows_used"] != 0:
                    raise RuntimeError("evaluation did not prove validation-only scope")

                failure_stage = "reporting"
                status.update(stage="reporting")
                comparison = analyze_and_report(
                    output, REFERENCE, configuration, tests, memory_smoke,
                    overfit, runtime, training, evaluation,
                )
                required = (
                    "config.json", "scheduled_sampling_config.json", "class_weights.json",
                    "tests/test_results.json", "tests/gpu_memory_smoke.json",
                    "tests/pedal_rich_tiny_overfit.json", "tests/runtime_benchmark.json",
                    "best.pt", "last.pt", "metrics.csv", "validation/evaluation.json",
                    "validation/validation_comparison.csv", "validation/per_class_metrics.csv",
                    "validation/five_class_confusion.csv", "validation/class_distributions.csv",
                    "scheduled_sampling_comparison.json", "scheduled_sampling_comparison.csv",
                    "SCHEDULED_SAMPLING_5CLASS_REPORT.md",
                )
                missing = [
                    name for name in required
                    if not (output / name).is_file() or (output / name).stat().st_size == 0
                ]
                if missing:
                    raise RuntimeError(f"scheduled-sampling artifacts missing: {missing}")
                status.update(
                    stage="complete", completed=True, failed=False,
                    end_time_utc=utc_now(), end_time_kst=kst_now(),
                    asap_test_midi_accessed=False,
                    primary_validation_result=FREE_MODEL,
                    class_weights=weights["weights"],
                    scheduled_sampling_comparison=comparison["interpretation"],
                )
                print("SCHEDULED_SAMPLING_PIPELINE_COMPLETE", flush=True)
                outcome = "complete"
                return 0
    except BaseException as exc:
        status.fail(
            failure_stage, exc, end_time_utc=utc_now(), end_time_kst=kst_now(),
            best_checkpoint_exists=(output / "best.pt").is_file(),
            last_checkpoint_exists=(output / "last.pt").is_file(),
            asap_test_midi_accessed=False,
        )
        with log_path.open("a", encoding="utf-8") as log:
            log.write(traceback.format_exc())
        raise
    finally:
        release_run_lock(lock, outcome)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    arguments = parser.parse_args()
    return run(arguments.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
