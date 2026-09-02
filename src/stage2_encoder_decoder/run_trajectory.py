"""Run the controlled no-early-stop 20-epoch trajectory experiment."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import traceback
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

from .run import run_tests
from .train import apply_train_only_weights, default_configuration, preload_datasets, train
from .trajectory import analyze


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REFERENCE = PROJECT_ROOT / "analysis" / "stage2_encoder_decoder_5class_weighted_v0"
DEFAULT_OUTPUT = PROJECT_ROOT / "analysis" / "stage2_encoder_decoder_5class_weighted_no_early_stop_20ep_v0"
DEFAULT_TMUX = "stage2-encoder-decoder-5class-weighted-no-early-stop-20ep-v0"
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


def _configuration(output: Path) -> dict[str, Any]:
    reference = json.loads((REFERENCE / "config.json").read_text(encoding="utf-8"))
    configuration = default_configuration(output)
    for key in CONTROLLED_KEYS:
        if key not in reference:
            raise KeyError(f"reference configuration missing controlled key: {key}")
        configuration[key] = reference[key]
    configuration.update(
        output_dir=str(output),
        early_stopping_enabled=False,
        resume=None,
        reference_run=str(REFERENCE),
        experiment="20_epoch_teacher_forced_trajectory_no_early_stop",
        scheduler=None,
        free_running_validation=False,
        midi_inference=False,
        rendering=False,
        reference_initial_hashes={key: reference[key] for key in INITIAL_HASH_KEYS},
    )
    if int(configuration["max_epochs"]) != 20:
        raise RuntimeError("reference max_epochs is not 20")
    if int(configuration["effective_batch_size"]) != 16:
        raise RuntimeError("reference effective batch size changed")
    return configuration


def run(output_dir: str | Path) -> int:
    output = Path(output_dir).resolve()
    if not (REFERENCE / "run_status.json").is_file():
        raise FileNotFoundError("completed reference run is missing")
    reference_status = json.loads((REFERENCE / "run_status.json").read_text(encoding="utf-8"))
    if not reference_status.get("completed") or reference_status.get("stage") != "complete":
        raise RuntimeError("reference encoder-decoder run is not complete")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite trajectory output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "tests").mkdir(exist_ok=True)
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:12]}"
    tmux_name = os.environ.get("STAGE2_TMUX_SESSION", DEFAULT_TMUX)
    lock = acquire_run_lock(output, run_id, container_pid=os.getpid())
    status = AtomicRunStatus(
        output / "run_status.json",
        initial_run_status(run_id, None, os.getpid(), tmux_name),
    )
    outcome = "failed"
    failure_stage = "initializing"
    log_path = output / "train.log"
    try:
        with log_path.open("a", encoding="utf-8") as log:
            with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
                configuration = _configuration(output)
                gpu = get_gpu_identity()
                if torch.cuda.device_count() != 1 or gpu["uuid"] != configuration["expected_gpu_uuid"]:
                    raise RuntimeError(f"assigned single-GPU runtime mismatch: {gpu}")
                configuration.update(
                    gpu_uuid=gpu["uuid"], gpu_name=gpu["name"],
                    container_visible_cuda_index=0, pytorch_version=torch.__version__,
                    cuda_version=torch.version.cuda, tmux_session_name=tmux_name,
                    run_id=run_id,
                )
                atomic_json(output / "config.json", configuration)

                failure_stage = "testing"
                status.update(stage="testing", gpu_uuid=gpu["uuid"])
                tests = run_tests(output)
                if not tests["passed"]:
                    raise RuntimeError("trajectory regression tests failed")

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
                    )

                train_dataset, validation_dataset = preload_datasets(configuration, progress)
                weights = apply_train_only_weights(configuration, train_dataset, output)
                reference_weights = json.loads((REFERENCE / "class_weights.json").read_text(encoding="utf-8"))
                if weights["class_counts"] != reference_weights["class_counts"] or not np.allclose(
                    weights["weights"], reference_weights["weights"], rtol=0.0, atol=0.0
                ):
                    raise RuntimeError("train-only weights differ from reference run")
                atomic_json(output / "config.json", configuration)

                failure_stage = "training"
                status.update(stage="training")
                training = train(
                    configuration, train_dataset, validation_dataset,
                    progress_callback=progress,
                )
                if training["completed_epoch"] != 20 or training["early_stopped"]:
                    raise RuntimeError(f"trajectory did not complete all 20 epochs: {training}")
                last = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
                if int(last["completed_epoch"]) != 20:
                    raise RuntimeError("last.pt is not the epoch-20 checkpoint")

                failure_stage = "reporting"
                status.update(
                    stage="reporting", current_epoch=20,
                    current_step=training["global_optimizer_step"],
                    best_epoch=training["best_epoch"],
                    best_validation_loss=training["best_validation_loss"],
                )
                trajectory = analyze(output, REFERENCE, training)
                required = (
                    "metrics.csv", "trajectory.csv", "trajectory_analysis.json",
                    "NO_EARLY_STOP_20EP_REPORT.md", "best.pt", "last.pt",
                )
                missing = [name for name in required if not (output / name).is_file() or (output / name).stat().st_size == 0]
                if missing:
                    raise RuntimeError(f"trajectory artifacts missing: {missing}")
                status.update(
                    stage="complete", completed=True, failed=False,
                    end_time_utc=utc_now(), end_time_kst=kst_now(),
                    best_epoch=trajectory["best_validation_epoch"],
                    best_validation_loss=trajectory["best_validation_weighted_ce"],
                    epoch_20_validation_loss=trajectory["epoch_20_validation_weighted_ce"],
                    early_stopping_enabled=False, asap_test_midi_accessed=False,
                    free_running_validation=False,
                )
                print("NO_EARLY_STOP_20EP_TRAJECTORY_COMPLETE", flush=True)
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
