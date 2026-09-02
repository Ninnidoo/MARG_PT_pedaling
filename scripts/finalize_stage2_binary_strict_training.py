#!/usr/bin/env python3
"""Verify definitive strict runs, write the report, and create final_lock_v1."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = PROJECT_ROOT / "analysis/stage2_binary_v0/train_strict_v1"
LOCK_ROOT = PROJECT_ROOT / "analysis/stage2_binary_v0/final_lock_v1"
HISTORICAL_ROOT = PROJECT_ROOT / "analysis/stage2_binary_v0/train_v0"
ARCHITECTURES = ("independent_4x2", "joint_16")
TOLERANCE = 1e-12
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


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _best_row(rows: Sequence[Mapping[str, str]]) -> Mapping[str, str]:
    best = rows[0]
    for row in rows[1:]:
        js = float(row["strict_validation_pedal_js_distance"])
        best_js = float(best["strict_validation_pedal_js_distance"])
        intersection = float(row["strict_validation_pedal_intersection"])
        best_intersection = float(best["strict_validation_pedal_intersection"])
        if js < best_js - TOLERANCE or (
            abs(js - best_js) <= TOLERANCE
            and intersection > best_intersection + TOLERANCE
        ):
            best = row
    return best


def _trajectory_comparison(
    architecture: str, strict_rows: Sequence[Mapping[str, str]]
) -> dict[str, Any]:
    historical = _read_csv(HISTORICAL_ROOT / architecture / "metrics.csv")
    historical_by_epoch = {int(row["epoch"]): row for row in historical}
    comparisons = []
    for row in strict_rows:
        epoch = int(row["epoch"])
        if epoch not in historical_by_epoch:
            continue
        old = historical_by_epoch[epoch]
        differences = {
            "train_loss": abs(float(row["train_loss"]) - float(old["train_loss"])),
            "validation_loss": abs(
                float(row["validation_loss"]) - float(old["validation_loss"])
            ),
            "direct_js": abs(
                float(row["direct_token_diagnostic_js_distance"])
                - float(old["validation_pedal_js_distance"])
            ),
            "direct_intersection": abs(
                float(row["direct_token_diagnostic_intersection"])
                - float(old["validation_pedal_intersection"])
            ),
        }
        comparisons.append({"epoch": epoch, **differences})
    if not comparisons:
        raise RuntimeError(f"no overlapping historical trajectory for {architecture}")
    maxima = {
        key: max(float(row[key]) for row in comparisons)
        for key in ("train_loss", "validation_loss", "direct_js", "direct_intersection")
    }
    return {
        "overlapping_epochs": len(comparisons),
        "per_epoch": comparisons,
        "maximum_absolute_difference": maxima,
        "exact_within_1e-12": max(maxima.values()) <= TOLERANCE,
    }


def _validate_run(architecture: str) -> dict[str, Any]:
    root = TRAIN_ROOT / architecture
    required = (
        "config.json",
        "train.log",
        "metrics.csv",
        "best.pt",
        "last.pt",
        "run_status.json",
        "epoch_checkpoint_manifest.csv",
        "posthoc_strict_metrics.csv",
    )
    if not all((root / name).is_file() for name in required):
        raise RuntimeError(f"missing strict run artifact for {architecture}")
    config = _read_json(root / "config.json")
    status = _read_json(root / "run_status.json")
    metrics = _read_csv(root / "metrics.csv")
    manifest = _read_csv(root / "epoch_checkpoint_manifest.csv")
    posthoc = _read_csv(root / "posthoc_strict_metrics.csv")
    if status["status"] != "completed" or status["asap_test_access_count"] != 0:
        raise RuntimeError(f"strict run status failed for {architecture}")
    if not metrics or len(metrics) != len(manifest) or len(metrics) != len(posthoc):
        raise RuntimeError(f"epoch artifact counts differ for {architecture}")
    if len(metrics) != int(status["completed_epochs"]):
        raise RuntimeError(f"completed epoch count differs for {architecture}")
    if [int(row["epoch"]) for row in metrics] != list(range(1, len(metrics) + 1)):
        raise RuntimeError(f"non-contiguous strict metrics for {architecture}")
    if not all(row["result"] == "PASS" for row in posthoc):
        raise RuntimeError(f"post-hoc strict verification failed for {architecture}")
    for row in posthoc:
        differences = [
            float(row[name])
            for name in (
                "strict_js_absolute_difference",
                "strict_intersection_absolute_difference",
                "direct_js_absolute_difference",
                "direct_intersection_absolute_difference",
            )
        ]
        if max(differences) > TOLERANCE or row["nonpedal_midi_exact"] != "True":
            raise RuntimeError(f"post-hoc tolerance failed for {architecture}")
    for row in manifest:
        path = Path(row["checkpoint_path"])
        if not path.is_file() or _sha256(path) != row["checkpoint_sha256"]:
            raise RuntimeError(f"epoch checkpoint hash mismatch: {path}")
    best = _best_row(metrics)
    if int(best["epoch"]) != int(status["strict_best_epoch"]):
        raise RuntimeError(f"strict best epoch mismatch for {architecture}")
    if abs(
        float(best["strict_validation_pedal_js_distance"])
        - float(status["strict_best_validation_js_distance"])
    ) > TOLERANCE:
        raise RuntimeError(f"strict best JS mismatch for {architecture}")
    if config["cache_id"] != "85a79e9d10e955b10000f72f6bbc4a29dbd2cc5a4c8cfb1277054a8d45dcb877":
        raise RuntimeError("cache ID changed")
    if config["initial_encoder_parameter_sha256"] != "3d6af38359042962e850fd4afeaedd9ad248266d99f1d2d31e3ab7bbec53aaed":
        raise RuntimeError("initial encoder hash changed")
    if config["epoch_window_order_sha256"]["1"] != "6dac2e18773c3d1fc0b907eabcfcc2bf1f168f7505034d86541ec839259439a7":
        raise RuntimeError("epoch-1 order hash changed")
    if config["first_training_batch_input_sha256"] != "603db239a6d14188ab80a258ef05cf4562139f8dd9bc3760e927203e5b5e6d5a":
        raise RuntimeError("first batch hash changed")
    trajectory = _trajectory_comparison(architecture, metrics)
    return {
        "root": root,
        "config": config,
        "status": status,
        "metrics": metrics,
        "manifest": manifest,
        "posthoc": posthoc,
        "best": best,
        "best_checkpoint_sha256": _sha256(root / "best.pt"),
        "last_checkpoint_sha256": _sha256(root / "last.pt"),
        "trajectory": trajectory,
    }


def _report(runs: Mapping[str, Mapping[str, Any]], primary: str) -> str:
    labels = {"independent_4x2": "independent 4×2", "joint_16": "joint 16"}
    lines = [
        "# Stage 2 Binary Strict Full-Training Report",
        "",
        "## Outcome",
        "",
        "Model A와 Model B를 기존 fixed cache/config/seed에서 official pretrained encoder로부터 fresh-start하여 순차 재학습했다. 각 epoch의 checkpoint selection과 early stopping은 official PT MIDI-roundtrip strict JS 및 no-post-epsilon-renormalization Intersection만 사용했다. Direct-token metric은 diagnostic으로만 기록했다.",
        "",
        "| Model | epochs | strict best epoch | strict JS ↓ | Intersection ↑ | binary acc | exact-pattern acc |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for architecture in ARCHITECTURES:
        run = runs[architecture]
        best = run["best"]
        lines.append(
            f"| {labels[architecture]} | {len(run['metrics'])} | {best['epoch']} | {float(best['strict_validation_pedal_js_distance']):.12f} | {float(best['strict_validation_pedal_intersection']):.12f} | {float(best['validation_binary_accuracy']):.9f} | {float(best['validation_exact_pattern_accuracy']):.9f} |"
        )
    lines += [
        "",
        "| Model | strict JS ↓ | Intersection ↑ |",
        "|---|---:|---:|",
        "| Original PT | 0.144081839387 | 0.907531644596 |",
    ]
    for architecture in ARCHITECTURES:
        best = runs[architecture]["best"]
        lines.append(
            f"| {labels[architecture]} strict best | {float(best['strict_validation_pedal_js_distance']):.12f} | {float(best['strict_validation_pedal_intersection']):.12f} |"
        )
    lines += [
        "",
        f"Primary architecture under strict global validation JS: **{labels[primary]}**.",
        "",
        "## Fixed training configuration",
        "",
        "- Training: MAESTRO-clean 1,170 + ASAP train 892 = 2,062 performances",
        "- Cache: 9,369,095 notes / 35,573 windows; ID `85a79e9d10e955b10000f72f6bbc4a29dbd2cc5a4c8cfb1277054a8d45dcb877`",
        "- Validation: ASAP validation 71 performances / 19 pieces; existing Stage 1 cache only",
        "- Seed 42; 512-note window; stride 256",
        "- AdamW; encoder LR 1e-5; head LR 1e-4; weight decay 0.01",
        "- Batch/effective batch 16/16; gradient accumulation 1; AMP enabled; max grad norm 1.0",
        "- Max epochs 10; strict-JS early-stopping patience 3",
        "- Encoder trainable; unweighted CE; no class/dataset weighting, balancing, or oversampling",
        "",
        "All fixed keys were programmatically compared with `configs/stage2_binary_full_training_v0.json` before either run.",
        "",
        "## Reproducibility",
        "",
        "Both architectures reproduced:",
        "",
        "- Initial encoder SHA-256: `3d6af38359042962e850fd4afeaedd9ad248266d99f1d2d31e3ab7bbec53aaed`",
        "- Epoch-1 window order SHA-256: `6dac2e18773c3d1fc0b907eabcfcc2bf1f168f7505034d86541ec839259439a7`",
        "- First training batch input SHA-256: `603db239a6d14188ab80a258ef05cf4562139f8dd9bc3760e927203e5b5e6d5a`",
        "- Same seed/cache/order across architectures",
        "",
    ]
    for architecture in ARCHITECTURES:
        trajectory = runs[architecture]["trajectory"]
        maximum = trajectory["maximum_absolute_difference"]
        lines.append(
            f"- {labels[architecture]} historical overlap: {trajectory['overlapping_epochs']} epochs; max |Δ train loss| `{maximum['train_loss']:.3e}`, validation loss `{maximum['validation_loss']:.3e}`, direct JS `{maximum['direct_js']:.3e}`, direct Intersection `{maximum['direct_intersection']:.3e}`; 1e-12 consistency: **{'PASS' if trajectory['exact_within_1e-12'] else 'WARN'}**"
        )
    lines += [
        "",
        "## Strict per-epoch trajectories",
        "",
    ]
    for architecture in ARCHITECTURES:
        run = runs[architecture]
        lines += [
            f"### {labels[architecture]}",
            "",
            "| Epoch | train loss | val loss | strict JS | strict Intersection | direct JS diagnostic | direct Intersection | AMP skips |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in run["metrics"]:
            lines.append(
                f"| {row['epoch']} | {float(row['train_loss']):.9f} | {float(row['validation_loss']):.9f} | {float(row['strict_validation_pedal_js_distance']):.12f} | {float(row['strict_validation_pedal_intersection']):.12f} | {float(row['direct_token_diagnostic_js_distance']):.12f} | {float(row['direct_token_diagnostic_intersection']):.12f} | {row['amp_skipped_optimizer_steps']} |"
            )
        status = run["status"]
        lines += [
            "",
            f"- Strict best epoch: {status['strict_best_epoch']} (old provisional best epoch: {'2' if architecture == 'independent_4x2' else '3'}; {'same' if int(status['strict_best_epoch']) == (2 if architecture == 'independent_4x2' else 3) else 'different'})",
            f"- Early stopping/completion epoch: {status['early_stopping_epoch']}",
            f"- AMP skipped optimizer steps: {status['amp_skipped_optimizer_steps_total']}; maximum consecutive: {status['maximum_consecutive_amp_skips']}",
            f"- Saved/post-hoc verified epoch checkpoints: {status['epoch_checkpoints_saved']}/{status['posthoc_epoch_metrics_verified']}",
            f"- `best.pt` SHA-256: `{run['best_checkpoint_sha256']}`",
            f"- `last.pt` SHA-256: `{run['last_checkpoint_sha256']}`",
            "",
        ]
    lines += [
        "## Previous direct-selected run comparison",
        "",
        "- Old independent epoch 2 strict reference: JS 0.072306914395 / Intersection 0.962942038782",
        "- Old joint epoch 3 strict reference: JS 0.150083204504 / Intersection 0.921791174179",
        "- The old values were comparison-only and did not alter this run's config or stopping decisions.",
        "",
        "## AMP behavior",
        "",
        "A focused CUDA test verified finite forward loss, optimizer-step skip, parameter non-update, and GradScaler backoff on an injected overflow. The same path logs and continues transient skips for both architectures. Eight consecutive skips are treated as persistent numerical failure and abort the run.",
        "",
        "## Strict validation integrity",
        "",
        "- Every training-time strict metric was reproduced post-hoc from its saved epoch checkpoint within 1e-12.",
        "- Strict path: cached Stage 1 IDs → Stage 2 Pedal1–4 replacement → official ids_to_midi/ref → map_midi → MIDI dump/reload → midi_to_ids → threshold 64 → global 16-pattern metric.",
        "- All 19 piece evaluations required exact non-pedal MIDI equality.",
        "- Original PT strict reference reproduced: JS 0.144081839387 / Intersection 0.907531644596.",
        "- Stage 1 inference regeneration: 0",
        "- ASAP test MIDI access: **0 / PASS**",
        "",
        "## Final lock",
        "",
        "A new `analysis/stage2_binary_v0/final_lock_v1/final_experiment_lock.json` was created. `final_lock_v0` was not overwritten.",
        "",
        "## Stop point",
        "",
        "Definitive strict training, per-epoch post-hoc validation, report, and lock only. No ASAP test evaluation, calibration, sampling, threshold change, or architecture/hyperparameter experiment was performed.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    if LOCK_ROOT.exists():
        raise FileExistsError(f"refusing to overwrite final lock v1: {LOCK_ROOT}")
    runs = {architecture: _validate_run(architecture) for architecture in ARCHITECTURES}
    first_config = runs[ARCHITECTURES[0]]["config"]
    second_config = runs[ARCHITECTURES[1]]["config"]
    mismatches = {
        key: (first_config.get(key), second_config.get(key))
        for key in FIXED_KEYS
        if first_config.get(key) != second_config.get(key)
    }
    if mismatches:
        raise RuntimeError(f"architecture configs differ: {mismatches}")
    primary = min(
        ARCHITECTURES,
        key=lambda architecture: float(
            runs[architecture]["best"]["strict_validation_pedal_js_distance"]
        ),
    )
    comparison = next(item for item in ARCHITECTURES if item != primary)
    report = _report(runs, primary)
    report_path = TRAIN_ROOT / "BINARY_STRICT_FULL_TRAINING_REPORT.md"
    report_path.write_text(report, encoding="utf-8")

    def candidate(architecture: str, role: str) -> dict[str, Any]:
        run = runs[architecture]
        best = run["best"]
        checkpoint = run["root"] / "best.pt"
        return {
            "role": role,
            "architecture": architecture,
            "model_class": (
                "src.stage2_binary.model.IndependentBinaryPedalModel"
                if architecture == "independent_4x2"
                else "src.stage2_binary.model.JointBinaryPedalModel"
            ),
            "best_epoch": int(best["epoch"]),
            "checkpoint_project_relative_path": str(
                checkpoint.relative_to(PROJECT_ROOT)
            ),
            "checkpoint_sha256": run["best_checkpoint_sha256"],
            "strict_validation_js_distance": float(
                best["strict_validation_pedal_js_distance"]
            ),
            "strict_validation_intersection": float(
                best["strict_validation_pedal_intersection"]
            ),
            "validation_binary_accuracy": float(best["validation_binary_accuracy"]),
            "validation_exact_pattern_accuracy": float(
                best["validation_exact_pattern_accuracy"]
            ),
            "config_sha256": _sha256(run["root"] / "config.json"),
            "epoch_checkpoint_manifest_sha256": _sha256(
                run["root"] / "epoch_checkpoint_manifest.csv"
            ),
            "posthoc_strict_metrics_sha256": _sha256(
                run["root"] / "posthoc_strict_metrics.csv"
            ),
        }

    lock = {
        "lock_version": "stage2_binary_strict_final_lock_v1",
        "status": "definitive_strict_locked_before_asap_test",
        "selection": {
            "primary_architecture": primary,
            "comparison_architecture": comparison,
            "primary_metric": "strict official MIDI-roundtrip global validation Pedal JS distance (lower)",
            "secondary_tiebreak": "strict official Intersection (higher) when abs(JS difference) <= 1e-12",
        },
        "candidates": {
            primary: candidate(primary, "primary"),
            comparison: candidate(comparison, "fixed_comparison"),
        },
        "inference_lock": {
            "stage1": "official Original Pianist Transformer cached validation/test inference",
            "stage1_cache_project_relative_path": "analysis/stage2_binary_v0/validation_eval_v0/stage1_cache",
            "stage1_cache_manifest_sha256": _sha256(
                PROJECT_ROOT
                / "analysis/stage2_binary_v0/validation_eval_v0/stage1_cache_manifest.csv"
            ),
            "stage1_checkpoint_model_safetensors_sha256": _sha256(
                PROJECT_ROOT / "checkpoints/pianist_transformer/model.safetensors"
            ),
            "stage2_changes": ["Pedal1", "Pedal2", "Pedal3", "Pedal4"],
            "binary_threshold": 64,
            "binary_rule": "raw <64 -> 0; raw >=64 -> 1",
            "decoding": "deterministic argmax",
            "window_notes": 512,
            "stride_notes": 256,
            "strict_midi_roundtrip": "ids_to_midi(ref=score_ids) -> map_midi -> dump/reload -> midi_to_ids",
            "non_pedal_midi_exact_equality_required": [
                "note count/order",
                "pitch",
                "onset",
                "offset/duration",
                "velocity",
                "tempo/time mapping",
                "all non-CC64 events",
            ],
            "sampling": False,
            "temperature_calibration": False,
            "test_time_calibration": False,
            "parameter_changes_after_lock": False,
        },
        "strict_evaluator_provenance": first_config["strict_evaluator_provenance"],
        "training_provenance": {
            "seed": 42,
            "cache_id": first_config["cache_id"],
            "initial_encoder_parameter_sha256": first_config[
                "initial_encoder_parameter_sha256"
            ],
            "epoch1_window_order_sha256": first_config[
                "epoch_window_order_sha256"
            ]["1"],
            "first_training_batch_input_sha256": first_config[
                "first_training_batch_input_sha256"
            ],
            "strict_training_report_sha256": _sha256(report_path),
        },
        "original_pt_strict_validation_reference": {
            "js_distance": 0.14408183938722538,
            "intersection": 0.9075316445962454,
        },
        "asap_test": {
            "midi_access_count": 0,
            "evaluation_performed": False,
        },
    }
    LOCK_ROOT.mkdir(parents=True)
    _write_json(LOCK_ROOT / "final_experiment_lock.json", lock)
    print(
        json.dumps(
            {
                "completed": True,
                "primary": primary,
                "primary_best_epoch": lock["candidates"][primary]["best_epoch"],
                "primary_strict_js": lock["candidates"][primary][
                    "strict_validation_js_distance"
                ],
                "comparison": comparison,
                "asap_test_midi_access_count": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
