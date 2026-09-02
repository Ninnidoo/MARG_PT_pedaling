"""Finish validation/reporting after a recoverable encoder-decoder pipeline error."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from src.stage2_encoder_only.run_five_class import atomic_json, atomic_text, kst_now, utc_now

from .evaluate import FREE_MODEL, evaluate
from .run import REPORT_NAME, render_report


def repair(output_dir: str | Path) -> None:
    root = Path(output_dir).resolve()
    configuration = json.loads((root / "config.json").read_text(encoding="utf-8"))
    test_results = json.loads((root / "tests" / "test_results.json").read_text(encoding="utf-8"))
    memory_smoke = json.loads((root / "tests" / "gpu_memory_smoke.json").read_text(encoding="utf-8"))
    overfit = json.loads((root / "tests" / "pedal_rich_tiny_overfit.json").read_text(encoding="utf-8"))
    with (root / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("cannot repair run without completed epoch metrics")
    checkpoint = torch.load(root / "best.pt", map_location="cpu", weights_only=False)
    training = {
        "completed_epoch": int(rows[-1]["epoch"]),
        "global_optimizer_step": int(rows[-1]["global_optimizer_step"]),
        "best_epoch": int(checkpoint["best_epoch"]),
        "best_validation_loss": float(checkpoint["best_validation_loss"]),
        "best_checkpoint_rule": "minimum teacher-forced validation weighted CE",
        "stop_reason": "early_stopping",
        "early_stopped": True,
        "effective_batch_size": int(configuration["effective_batch_size"]),
        "micro_batch_size": int(configuration["micro_batch_size"]),
        "gradient_accumulation_steps": int(configuration["gradient_accumulation_steps"]),
        "repair": "validation/report rerun only; no training or checkpoint modification",
    }
    evaluation = evaluate(root)
    if evaluation["split"] != "validation" or evaluation["test_rows_used"] != 0:
        raise RuntimeError("repair evaluation did not preserve validation-only scope")
    atomic_text(
        root / REPORT_NAME,
        render_report(configuration, test_results, memory_smoke, overfit, training, evaluation),
    )
    status_path = root / "run_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status.update(
        stage="complete",
        completed=True,
        failed=False,
        failure_stage=None,
        exception_type=None,
        exception_message=None,
        end_time_utc=utc_now(),
        end_time_kst=kst_now(),
        primary_validation_result=FREE_MODEL,
        asap_test_midi_accessed=False,
        validation_repair=True,
    )
    atomic_json(status_path, status)
    print("ENCODER_DECODER_VALIDATION_REPAIR_COMPLETE", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    arguments = parser.parse_args()
    repair(arguments.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
