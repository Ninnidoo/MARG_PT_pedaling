"""One-process pipeline for the weighted Stage 2 encoder-decoder experiment."""

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

import torch

from src.stage2_encoder_only.run_five_class import (
    AtomicRunStatus,
    Tee,
    acquire_run_lock,
    atomic_json,
    atomic_text,
    initial_run_status,
    kst_now,
    release_run_lock,
    utc_now,
)
from src.stage2_encoder_only.train import get_gpu_identity

from .evaluate import FREE_MODEL, TEACHER_MODEL, evaluate
from .train import (
    apply_train_only_weights,
    default_configuration,
    pedal_rich_tiny_overfit,
    preload_datasets,
    prepare_preflight,
    train,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "analysis" / "stage2_encoder_decoder_5class_weighted_v0"
DEFAULT_TMUX = "stage2-encoder-decoder-5class-weighted-v0"
REPORT_NAME = "ENCODER_DECODER_5CLASS_REPORT.md"
TEST_MODULES = (
    "tests.test_stage2_encoder_decoder_five_class",
    "tests.test_stage2_weighted_five_class",
    "tests.test_stage2_five_class",
    "tests.test_stage2_five_class_runtime",
)


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str)


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
        raise RuntimeError("required encoder-decoder synthetic tests failed")
    return payload


def render_report(
    configuration: Mapping[str, Any],
    tests: Mapping[str, Any],
    memory_smoke: Mapping[str, Any],
    overfit: Mapping[str, Any],
    training: Mapping[str, Any],
    evaluation: Mapping[str, Any],
) -> str:
    baseline = evaluation["comparison"]["corrected_weighted_encoder_only"]
    teacher = evaluation["comparison"][TEACHER_MODEL]
    free = evaluation["comparison"][FREE_MODEL]
    requested = (
        "five_class_token_accuracy", "five_class_macro_f1",
        "nonendpoint_endpoint_collapse_ratio",
        "canonical_decoded_overall_mae", "canonical_decoded_on_region_mae",
        "canonical_decoded_subthreshold_mae", "off_on_state_accuracy",
        "off_on_state_macro_f1", "binary_transition_f1", "short_repedal_f1",
    )
    comparison = {
        key: {
            "corrected_weighted_encoder_only": baseline.get(key),
            "encoder_decoder_teacher_forced": teacher.get(key),
            "encoder_decoder_greedy_free_running": free.get(key),
        }
        for key in requested
    }
    return f"""# Stage 2 Encoder-Decoder Weighted 5-class Report

## Scope

Corrected weighted encoder-only baseline과 데이터, five-class target, train-only inverse-sqrt
weighted CE, optimizer, window/stride, early stopping을 유지하고 Stage 2 architecture만
PT-native encoder-decoder로 변경했다. ASAP test, MAESTRO, 23-score inference는 사용하지 않았다.

## Architecture

{configuration['architecture']}

- Encoder: pretrained official PT encoder, 10 layers, hidden 768, unfrozen
- Decoder: fresh seed `{configuration['decoder_init_seed']}`, 2 causal layers, hidden 768,
  FFN 3072, head dimension 128, native rotary position/norm/dropout
- Sequence: `P1_1,P2_1,P3_1,P4_1,...`; teacher input is shifted right with BOS
- Decoder vocabulary: five classes + BOS + PAD; shared `Linear(768,5)` output

## Controlled configuration

```json
{_json(configuration)}
```

## Tests

```json
{_json(tests)}
```

## GPU memory smoke and effective batch

```json
{_json(memory_smoke)}
```

## Pedal-rich tiny overfit

```json
{_json(overfit)}
```

## Training

Best checkpoint criterion is minimum teacher-forced validation weighted CE.

```json
{_json(training)}
```

## Primary validation comparison

Greedy free-running autoregressive decoding is the primary encoder-decoder result.

```json
{_json(comparison)}
```

## Teacher-forced versus free-running gap

```json
{_json(evaluation['teacher_forced_vs_free_running_gap'])}
```

## Free-running per-class precision/recall/F1 and distribution

```json
{_json(evaluation['free_running']['classification'])}
```

## Free-running OFF/ON, transition timing, and short-repedal metrics

```json
{_json(evaluation['free_running']['renderer'])}
```

## Free-running decoded MAE

```json
{_json(evaluation['free_running']['decoded'])}
```

## Validation-only provenance

- Split: `{evaluation['split']}`
- Test rows used: `{evaluation['test_rows_used']}`
- Performances/pieces/notes/windows: `{evaluation['performance_count']}` /
  `{evaluation['piece_count']}` / `{evaluation['note_count']}` / `{evaluation['window_count']}`
- Class weights source: train Pedal1–4 only; `Σ p_c w_c = 1`
"""


def run(output_dir: str | Path) -> int:
    output = Path(output_dir).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite encoder-decoder output: {output}")
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
                gpu = get_gpu_identity()
                if torch.cuda.device_count() != 1 or gpu["uuid"] != default_configuration(output)["expected_gpu_uuid"]:
                    raise RuntimeError(f"assigned single-GPU runtime mismatch: {gpu}")
                configuration = default_configuration(output)
                configuration.update({
                    "gpu_uuid": gpu["uuid"], "gpu_name": gpu["name"],
                    "container_visible_cuda_index": 0,
                    "pytorch_version": torch.__version__, "cuda_version": torch.version.cuda,
                    "tmux_session_name": tmux_name,
                    "container_hostname": socket.gethostname(),
                    "run_id": run_id,
                })
                atomic_json(output / "config.json", configuration)
                status.update(stage="testing", gpu_uuid=gpu["uuid"])
                failure_stage = "testing"
                tests = run_tests(output)

                failure_stage = "preloading"
                status.update(stage="preloading")
                def progress(update: Mapping[str, Any]) -> None:
                    stage = "preloading" if update.get("stage") == "preloading" else "training"
                    status.update(
                        stage=stage,
                        current_epoch=int(update.get("epoch", status.payload.get("current_epoch", 0))),
                        current_step=int(update.get("global_step", status.payload.get("current_step", 0))),
                        latest_finite_loss=update.get("loss", status.payload.get("latest_finite_loss")),
                        current_batch=update.get("batch"), total_batches=update.get("batches"),
                        preload_split=update.get("split"),
                    )
                train_dataset, validation_dataset = preload_datasets(configuration, progress)
                weights = apply_train_only_weights(configuration, train_dataset, output)
                atomic_json(output / "config.json", configuration)

                failure_stage = "overfit"
                status.update(stage="overfit")
                memory_smoke, overfit = prepare_preflight(configuration, train_dataset, output)
                atomic_json(output / "config.json", configuration)

                failure_stage = "training"
                status.update(stage="training")
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
                report = render_report(configuration, tests, memory_smoke, overfit, training, evaluation)
                atomic_text(output / REPORT_NAME, report)
                required = (
                    "config.json", "class_weights.json", "best.pt", "last.pt", "metrics.csv",
                    "validation/evaluation.json", "validation/validation_comparison.csv",
                    "validation/per_class_metrics.csv", "validation/five_class_confusion.csv",
                    "validation/class_distributions.csv", REPORT_NAME,
                )
                missing = [name for name in required if not (output / name).is_file() or (output / name).stat().st_size == 0]
                if missing:
                    raise RuntimeError(f"required artifacts missing: {missing}")
                status.update(
                    stage="complete", completed=True, failed=False,
                    end_time_utc=utc_now(), end_time_kst=kst_now(),
                    asap_test_midi_accessed=False,
                    primary_validation_result=FREE_MODEL,
                    class_weights=weights["weights"],
                )
                print("ENCODER_DECODER_PIPELINE_COMPLETE", flush=True)
                outcome = "complete"
                return 0
    except BaseException as exc:
        status.fail(
            failure_stage, exc, end_time_utc=utc_now(), end_time_kst=kst_now(),
            best_checkpoint_exists=(output / "best.pt").is_file(),
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
