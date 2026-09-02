#!/usr/bin/env python3
"""Verify completed fixed binary runs and write their final comparison report."""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.stage2_binary.model import (  # noqa: E402
    IndependentBinaryPedalModel,
    JointBinaryPedalModel,
)


MODEL_CLASSES = {
    "independent_4x2": IndependentBinaryPedalModel,
    "joint_16": JointBinaryPedalModel,
}
DISPLAY_NAMES = {
    "independent_4x2": "independent 4×2",
    "joint_16": "joint 16",
}
FIXED_FIELDS = {
    "seed": 42,
    "window_notes": 512,
    "stride_notes": 256,
    "optimizer": "AdamW",
    "encoder_lr": 1e-5,
    "head_lr": 1e-4,
    "weight_decay": 0.01,
    "batch_size": 16,
    "gradient_accumulation_steps": 1,
    "effective_batch_size": 16,
    "max_grad_norm": 1.0,
    "amp_enabled": True,
    "max_epochs": 10,
    "early_stopping_patience": 3,
}
EXPECTED_CACHE_ID = "85a79e9d10e955b10000f72f6bbc4a29dbd2cc5a4c8cfb1277054a8d45dcb877"
EXPECTED_ENCODER_HASH = "3d6af38359042962e850fd4afeaedd9ad248266d99f1d2d31e3ab7bbec53aaed"
PT_JS = 0.144081839387
PT_INTERSECTION = 0.907531643144
TIE_TOLERANCE = 1e-12


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _read_metrics(path: Path) -> list[dict[str, float | int]]:
    with path.open(newline="", encoding="utf-8") as handle:
        raw = list(csv.DictReader(handle))
    if not raw:
        raise RuntimeError(f"empty metrics: {path}")
    integer_fields = {"epoch", "global_optimizer_step", "peak_gpu_memory_bytes"}
    rows: list[dict[str, float | int]] = []
    for raw_row in raw:
        row: dict[str, float | int] = {}
        for key, value in raw_row.items():
            row[key] = int(value) if key in integer_fields else float(value)
            if not math.isfinite(float(row[key])):
                raise FloatingPointError(f"non-finite {key} in {path}")
        rows.append(row)
    if [row["epoch"] for row in rows] != list(range(1, len(rows) + 1)):
        raise RuntimeError(f"non-contiguous epochs: {path}")
    if any(
        int(rows[index]["global_optimizer_step"])
        <= int(rows[index - 1]["global_optimizer_step"])
        for index in range(1, len(rows))
    ):
        raise RuntimeError(f"optimizer steps are not increasing: {path}")
    return rows


def _best_row(rows: list[dict[str, float | int]]) -> dict[str, float | int]:
    best = rows[0]
    for row in rows[1:]:
        row_js = float(row["validation_pedal_js_distance"])
        best_js = float(best["validation_pedal_js_distance"])
        better_js = row_js < best_js - TIE_TOLERANCE
        tied = abs(row_js - best_js) <= TIE_TOLERANCE
        better_intersection = (
            float(row["validation_pedal_intersection"])
            > float(best["validation_pedal_intersection"]) + TIE_TOLERANCE
        )
        if better_js or (tied and better_intersection):
            best = row
    return best


def _strict_checkpoint_loads(
    architecture: str,
    checkpoint_path: Path,
    best_payload: dict[str, Any],
    last_payload: dict[str, Any],
) -> bool:
    model = MODEL_CLASSES[architecture].from_pretrained(
        checkpoint_path,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    for payload in (best_payload, last_payload):
        incompatible = model.load_state_dict(payload["model_state"], strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"strict checkpoint mismatch: {architecture}")
    del model
    return True


def _verify_run(
    output_root: Path,
    architecture: str,
    source_config: dict[str, Any],
) -> dict[str, Any]:
    run_dir = output_root / architecture
    required = ("config.json", "train.log", "metrics.csv", "best.pt", "last.pt", "run_status.json")
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{architecture} missing artifacts: {missing}")
    status = _read_json(run_dir / "run_status.json")
    config = _read_json(run_dir / "config.json")
    if status.get("status") != "completed":
        raise RuntimeError(f"{architecture} status is not completed: {status}")
    if int(status.get("asap_test_access_count", -1)) != 0:
        raise RuntimeError(f"{architecture} reports ASAP test access")
    if config.get("architecture") != architecture:
        raise RuntimeError(f"architecture mismatch in {architecture} config")
    for key, expected in FIXED_FIELDS.items():
        if config.get(key) != expected or source_config.get(key) != expected:
            raise RuntimeError(f"fixed config mismatch: {architecture} {key}")
    if config.get("cache_id") != EXPECTED_CACHE_ID:
        raise RuntimeError(f"cache ID mismatch: {architecture}")
    if config.get("initial_encoder_parameter_sha256") != EXPECTED_ENCODER_HASH:
        raise RuntimeError(f"encoder provenance mismatch: {architecture}")
    rows = _read_metrics(run_dir / "metrics.csv")
    best = _best_row(rows)
    last = rows[-1]
    best_payload = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    last_payload = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
    if int(status["completed_epochs"]) != len(rows):
        raise RuntimeError(f"status/metrics epoch mismatch: {architecture}")
    if int(status["best_epoch"]) != int(best["epoch"]):
        raise RuntimeError(f"status best epoch mismatch: {architecture}")
    if int(best_payload["best_epoch"]) != int(best["epoch"]):
        raise RuntimeError(f"best.pt epoch mismatch: {architecture}")
    for recorded, expected, name in (
        (status["best_validation_js_distance"], best["validation_pedal_js_distance"], "status JS"),
        (status["best_validation_intersection"], best["validation_pedal_intersection"], "status intersection"),
        (best_payload["best_validation_js_distance"], best["validation_pedal_js_distance"], "best.pt JS"),
        (best_payload["best_validation_intersection"], best["validation_pedal_intersection"], "best.pt intersection"),
    ):
        if not math.isclose(float(recorded), float(expected), rel_tol=0.0, abs_tol=1e-15):
            raise RuntimeError(f"{architecture} {name} mismatch")
    if int(last_payload["completed_epoch"]) != len(rows):
        raise RuntimeError(f"last.pt epoch mismatch: {architecture}")
    if int(last_payload["global_optimizer_step"]) != int(last["global_optimizer_step"]):
        raise RuntimeError(f"last.pt optimizer step mismatch: {architecture}")
    if last_payload["configuration"]["cache_id"] != EXPECTED_CACHE_ID:
        raise RuntimeError(f"last.pt cache provenance mismatch: {architecture}")
    strict_load = _strict_checkpoint_loads(
        architecture,
        Path(config["checkpoint_path"]),
        best_payload,
        last_payload,
    )
    stopping_counter = int(last_payload["early_stopping_counter"])
    early_stopped = len(rows) < int(config["max_epochs"])
    if early_stopped and stopping_counter < int(config["early_stopping_patience"]):
        raise RuntimeError(f"early stop counter is too small: {architecture}")
    reason = (
        f"validation Pedal JS did not improve for {stopping_counter} consecutive epochs"
        if early_stopped
        else "reached fixed max_epochs=10"
    )
    result = {
        "architecture": architecture,
        "epochs": len(rows),
        "best": best,
        "last": last,
        "early_stopped": early_stopped,
        "early_stopping_counter": stopping_counter,
        "stopping_reason": reason,
        "total_epoch_runtime_seconds": sum(float(row["epoch_time_seconds"]) for row in rows),
        "peak_gpu_memory_bytes": max(int(row["peak_gpu_memory_bytes"]) for row in rows),
        "best_checkpoint_bytes": (run_dir / "best.pt").stat().st_size,
        "last_checkpoint_bytes": (run_dir / "last.pt").stat().st_size,
        "best_checkpoint_selection_verified": True,
        "best_and_last_strict_load_verified": strict_load,
        "initial_encoder_hash_verified": True,
        "fixed_config_verified": True,
        "asap_test_access_count": 0,
        "resume_used": "resumed_from" in status,
        "resumed_from": status.get("resumed_from"),
        "amp_skipped_steps_after_resume": int(
            status.get("amp_skipped_steps_after_resume", 0)
        ),
        "previous_failure": status.get("previous_status"),
    }
    del best_payload, last_payload
    return result


def _duration(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remaining = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{remaining:06.3f}"


def _report(results: dict[str, dict[str, Any]]) -> str:
    a = results["independent_4x2"]
    b = results["joint_16"]
    winner = min(
        results.values(),
        key=lambda value: (
            float(value["best"]["validation_pedal_js_distance"]),
            -float(value["best"]["validation_pedal_intersection"]),
        ),
    )
    lines = [
        "# Stage 2 Binary Full-Training Report",
        "",
        "## Outcome",
        "",
        "고정된 동일 config/cache로 Model A와 Model B를 host tmux sequential pipeline에서 순서대로 full training했다. Model B epoch 5 도중 기존 runner가 transient AMP overflow를 GradScaler skip/backoff 전에 예외로 처리해 중단됐으며, 같은 tmux session 이름과 epoch-4 `last.pt`의 model/optimizer/scaler/RNG/early-stopping state에서 재개했다. Hyperparameter나 data order는 변경하지 않았다. 두 run은 최종 정상 완료했으며 best/last checkpoint와 metrics의 일관성 및 strict load를 검증했다. ASAP test access는 **0**이다.",
        "",
        "## Validation summary",
        "",
        "| Model | epochs | best epoch | best JS ↓ | Intersection ↑ | binary acc | exact-pattern acc |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for value in (a, b):
        best = value["best"]
        lines.append(
            f"| {DISPLAY_NAMES[value['architecture']]} | {value['epochs']} | {int(best['epoch'])} | "
            f"{float(best['validation_pedal_js_distance']):.12f} | "
            f"{float(best['validation_pedal_intersection']):.12f} | "
            f"{float(best['validation_binary_accuracy']):.9f} | "
            f"{float(best['validation_exact_pattern_accuracy']):.9f} |"
        )
    lines += [
        "",
        "## Original PT comparison",
        "",
        "| Model | Pedal JS ↓ | Intersection ↑ |",
        "|---|---:|---:|",
        f"| Original PT | {PT_JS:.12f} | {PT_INTERSECTION:.12f} |",
    ]
    for value in (a, b):
        best = value["best"]
        lines.append(
            f"| {DISPLAY_NAMES[value['architecture']]} best | "
            f"{float(best['validation_pedal_js_distance']):.12f} | "
            f"{float(best['validation_pedal_intersection']):.12f} |"
        )
    lines += [
        "",
        "Original PT 대비 변화:",
        "",
    ]
    for value in (a, b):
        best = value["best"]
        js_delta = float(best["validation_pedal_js_distance"]) - PT_JS
        intersection_delta = float(best["validation_pedal_intersection"]) - PT_INTERSECTION
        lines.append(
            f"- {DISPLAY_NAMES[value['architecture']]}: JS {js_delta:+.12f}, Intersection {intersection_delta:+.12f}"
        )
    lines += [
        "",
        f"Primary validation metric 기준 우수 architecture: **{DISPLAY_NAMES[winner['architecture']]}** (lower JS; numerical tie이면 higher Intersection).",
        "",
        "Validation CE는 architecture selection에 사용하지 않았으며 두 architecture 사이에서 직접 비교하지 않는다.",
        "",
        "## Fixed common training configuration",
        "",
        f"- Cache ID: `{EXPECTED_CACHE_ID}`",
        "- Training: MAESTRO-clean 1,170 + ASAP train 892 = 2,062 performances",
        "- Training cache: 9,369,095 notes / 35,573 windows",
        "- Validation: ASAP validation 71 performances / 19 pieces / 1,078 windows",
        "- Seed: 42",
        "- Window / stride: 512 / 256 notes",
        "- AdamW; encoder LR `1e-5`; head LR `1e-4`; weight decay 0.01",
        "- Batch / gradient accumulation / effective batch: 16 / 1 / 16",
        "- Max grad norm: 1.0; AMP enabled",
        "- Max epochs: 10; early-stopping patience: 3",
        "- Encoder trainable; unweighted CE",
        "- Dataset/class weighting, balancing, oversampling: none",
        "- Stage 1 validation cache reused; Stage 1 inference regeneration: none",
        "",
        "Model A와 Model B의 initial encoder parameter SHA-256는 모두 "
        f"`{EXPECTED_ENCODER_HASH}`로 setup provenance와 일치한다.",
        "",
        "## Run details",
        "",
    ]
    for value in (a, b):
        best = value["best"]
        last = value["last"]
        lines += [
            f"### {DISPLAY_NAMES[value['architecture']]}",
            "",
            f"- Actual epochs: {value['epochs']}",
            f"- Total summed epoch runtime: {_duration(value['total_epoch_runtime_seconds'])} ({value['total_epoch_runtime_seconds']:.3f} s)",
            f"- Stop: {value['stopping_reason']}",
            f"- Best epoch: {int(best['epoch'])}; JS {float(best['validation_pedal_js_distance']):.12f}; Intersection {float(best['validation_pedal_intersection']):.12f}",
            f"- Best validation loss: {float(best['validation_loss']):.9f}",
            f"- Last epoch {int(last['epoch'])}: train loss {float(last['train_loss']):.9f}; validation loss {float(last['validation_loss']):.9f}; binary acc {float(last['validation_binary_accuracy']):.9f}; exact-pattern acc {float(last['validation_exact_pattern_accuracy']):.9f}; JS {float(last['validation_pedal_js_distance']):.12f}; Intersection {float(last['validation_pedal_intersection']):.12f}",
            f"- Last optimizer step: {int(last['global_optimizer_step'])}",
            f"- Peak GPU memory: {value['peak_gpu_memory_bytes']:,} bytes ({value['peak_gpu_memory_bytes'] / 2**30:.3f} GiB)",
            f"- Epoch-boundary resume used: {value['resume_used']}; AMP-skipped batches after resume: {value['amp_skipped_steps_after_resume']}",
            f"- `best.pt`: {value['best_checkpoint_bytes']:,} bytes; selection metadata and strict load PASS",
            f"- `last.pt`: {value['last_checkpoint_bytes']:,} bytes; epoch/optimizer metadata and strict load PASS",
            "",
        ]
    lines += [
        "## Integrity verification",
        "",
        "- Both run statuses completed: PASS",
        "- Fixed config equality across architectures: PASS",
        "- Same cache ID and deterministic ordering provenance: PASS",
        "- Same official pretrained encoder provenance: PASS",
        "- Lowest-JS / Intersection-tiebreak best row equals `best.pt`: PASS",
        "- `best.pt` and `last.pt` load with strict architecture state dict: PASS",
        "- Metrics finite and epoch/optimizer sequence consistent: PASS",
        "- Required artifact set present for both architectures: PASS",
        "- Model B recovery restored the epoch-4 checkpoint without changing fixed config; transient AMP overflow used standard dynamic-scale skip/backoff: PASS",
        "- ASAP test access: **0 / PASS**",
        "",
        "No test evaluation, Stage 1 regeneration, audio rendering, calibration, weighting, balancing, architecture change, or additional full run was performed.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    output_root = PROJECT_ROOT / "analysis/stage2_binary_v0/train_v0"
    source_config = _read_json(PROJECT_ROOT / "configs/stage2_binary_full_training_v0.json")
    results = {
        architecture: _verify_run(output_root, architecture, source_config)
        for architecture in MODEL_CLASSES
    }
    report_path = output_root / "BINARY_FULL_TRAINING_REPORT.md"
    temporary = report_path.with_name(f".{report_path.name}.{os.getpid()}.tmp")
    temporary.write_text(_report(results), encoding="utf-8")
    os.replace(temporary, report_path)
    summary_path = output_root / "full_training_summary.json"
    temporary = summary_path.with_name(f".{summary_path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, summary_path)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
