"""Benchmark and report helpers for the scheduled-sampling controlled run."""

from __future__ import annotations

import csv
import json
import math
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Mapping

import torch

from src.stage2_encoder_only.five_class import CLASS_NAMES, five_class_collate_fn
from src.stage2_encoder_only.run_five_class import atomic_json, atomic_text
from src.stage2_encoder_only.training import build_optimizer, create_grad_scaler, set_deterministic_seed

from .evaluate import FREE_MODEL, TEACHER_MODEL
from .train import _make_model, optimizer_step


def _gpu_sampler(stop: threading.Event, samples: list[dict[str, float]]) -> None:
    while not stop.is_set():
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits"],
                check=True, capture_output=True, text=True, timeout=5,
            )
            utilization, memory = result.stdout.strip().splitlines()[0].split(",")
            samples.append({
                "utilization_percent": float(utilization.strip()),
                "memory_mib": float(memory.strip()),
            })
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            pass
        stop.wait(0.5)


def _timed_step(model, cpu_batch, device, optimizer, scaler,
                configuration: Mapping[str, Any], probability: float, seed: int) -> dict[str, Any]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    samples: list[dict[str, float]] = []
    stop = threading.Event()
    sampler = threading.Thread(target=_gpu_sampler, args=(stop, samples), daemon=True)
    sampler.start()
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    metrics = optimizer_step(
        model, cpu_batch, device, optimizer, scaler,
        micro_batch_size=int(configuration["micro_batch_size"]),
        amp_enabled=bool(configuration["amp_enabled"]),
        max_grad_norm=float(configuration["max_grad_norm"]),
        teacher_forcing_probability_value=probability,
        scheduled_sampling_seed=seed,
    )
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    stop.set()
    sampler.join(timeout=5)
    return {
        "teacher_forcing_probability": probability,
        "optimizer_step_seconds": seconds,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "gpu_utilization_samples": len(samples),
        "mean_gpu_utilization_percent": (
            statistics.fmean(item["utilization_percent"] for item in samples) if samples else None
        ),
        "maximum_gpu_utilization_percent": (
            max(item["utilization_percent"] for item in samples) if samples else None
        ),
        "maximum_nvidia_smi_memory_mib": (
            max(item["memory_mib"] for item in samples) if samples else None
        ),
        "loss": float(metrics["loss"]),
        "token_accuracy": float(metrics["token_accuracy"]),
        "ground_truth_history_ratio": float(metrics["ground_truth_history_ratio"]),
        "predicted_history_ratio": float(metrics["predicted_history_ratio"]),
        "gradient_groups_finite_nonzero": metrics["gradient_groups"],
        "finite": bool(
            math.isfinite(float(metrics["loss"])) and math.isfinite(float(metrics["gradient_norm"]))
        ),
    }


def benchmark(configuration: Mapping[str, Any], train_dataset,
              output_dir: str | Path, baseline_dir: str | Path) -> dict[str, Any]:
    """Benchmark one full effective-batch update for each training path."""

    root = Path(output_dir).resolve()
    baseline = Path(baseline_dir).resolve()
    device = torch.device("cuda:0")
    sample = None
    for index in range(len(train_dataset)):
        candidate = train_dataset[index]
        if candidate["note_mask"].numel() == 512:
            sample = candidate
            break
    if sample is None:
        raise RuntimeError("no full 512-note train window for runtime benchmark")
    effective = int(configuration["effective_batch_size"])
    cpu_batch = five_class_collate_fn([sample] * effective)
    set_deterministic_seed(int(configuration["seed"]))
    model = _make_model(configuration).to(device)
    optimizer = build_optimizer(
        model, encoder_lr=float(configuration["encoder_lr"]),
        head_lr=float(configuration["decoder_lr"]),
        weight_decay=float(configuration["weight_decay"]),
    )
    scaler = create_grad_scaler(
        bool(configuration["amp_enabled"]), "cuda", float(configuration["amp_init_scale"])
    )
    teacher = _timed_step(
        model, cpu_batch, device, optimizer, scaler, configuration, 1.0,
        int(configuration["scheduled_sampling_seed"]),
    )
    scheduled = _timed_step(
        model, cpu_batch, device, optimizer, scaler, configuration, 0.9,
        int(configuration["scheduled_sampling_seed"]) + 3_000_001,
    )
    batches = math.ceil(len(train_dataset) / effective)
    with (baseline / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        baseline_rows = list(csv.DictReader(handle))
    reference_epoch_seconds = statistics.median(
        float(row["epoch_seconds"]) for row in baseline_rows[:2]
    )
    validation_seconds = max(
        0.0, reference_epoch_seconds - teacher["optimizer_step_seconds"] * batches
    )
    scheduled_epoch_seconds = scheduled["optimizer_step_seconds"] * batches + validation_seconds
    reference_stop_epoch = int(baseline_rows[-1]["epoch"])
    projected_reference_stop = (
        2 * reference_epoch_seconds
        + max(reference_stop_epoch - 2, 0) * scheduled_epoch_seconds
    )
    projected_max_20 = 2 * reference_epoch_seconds + 18 * scheduled_epoch_seconds
    utilization = scheduled["mean_gpu_utilization_percent"]
    runnable = bool(
        scheduled["finite"]
        and all(scheduled["gradient_groups_finite_nonzero"].values())
        and (utilization is None or utilization >= 5.0)
        and projected_reference_stop <= 4 * 24 * 3600
    )
    result = {
        "passed": runnable,
        "semantics": (
            "detached exact left-to-right autoregressive history rollout with KV cache; "
            "one causal parallel gradient pass conditional on the completed mixed history"
        ),
        "window_notes": 512,
        "decoder_steps": 2048,
        "effective_batch_size": effective,
        "micro_batch_size": int(configuration["micro_batch_size"]),
        "gradient_accumulation_steps": int(configuration["gradient_accumulation_steps"]),
        "train_batches_per_epoch": batches,
        "teacher_forced_step": teacher,
        "scheduled_sampling_step": scheduled,
        "reference_epoch_1_2_seconds": reference_epoch_seconds,
        "estimated_scheduled_sampling_epoch_seconds": scheduled_epoch_seconds,
        "estimated_scheduled_sampling_epoch_hours": scheduled_epoch_seconds / 3600,
        "projected_total_if_stops_at_reference_epoch": reference_stop_epoch,
        "projected_total_reference_stop_hours": projected_reference_stop / 3600,
        "projected_total_max_20_hours": projected_max_20 / 3600,
        "decision": "start full run" if runnable else "investigate runtime/utilization before run",
        "asap_test_split_accessed": False,
    }
    atomic_json(root / "tests" / "runtime_benchmark.json", result)
    del model, optimizer, scaler, cpu_batch
    torch.cuda.empty_cache()
    if not runnable:
        raise RuntimeError(f"scheduled-sampling runtime benchmark failed: {result}")
    return result


def _high_to_full(classification: Mapping[str, Any]) -> dict[str, Any]:
    count = int(classification["confusion_matrix"][3][4])
    support = int(classification["support"][3])
    return {"count": count, "rate": count / support if support else float("nan")}


def _mode_summary(evaluation: Mapping[str, Any], mode_key: str,
                  comparison_key: str) -> dict[str, Any]:
    mode = evaluation[mode_key]
    classification = mode["classification"]
    renderer = mode["renderer"]
    decoded = mode["decoded"]
    flat = evaluation["comparison"][comparison_key]
    timing = renderer["transition_timing_sample_offset"]
    return {
        "token_accuracy": float(classification["token_accuracy"]),
        "macro_f1": float(classification["macro_f1"]),
        "weighted_f1": float(classification["weighted_f1"]),
        "per_class": {
            name: {
                "precision": float(classification["precision"][index]),
                "recall": float(classification["recall"][index]),
                "f1": float(classification["f1"][index]),
                "support": int(classification["support"][index]),
                "predicted_count": int(classification["predicted_count"][index]),
                "target_ratio": float(classification["target_distribution"][index]),
                "predicted_ratio": float(classification["predicted_distribution"][index]),
            }
            for index, name in enumerate(CLASS_NAMES)
        },
        "high_to_full": _high_to_full(classification),
        "nonendpoint_endpoint_collapse_ratio": float(
            classification["nonendpoint_endpoint_collapse_ratio"]
        ),
        "overall_decoded_mae": float(decoded["overall_micro_mae"]),
        "on_region_mae": float(decoded["on_region_mae"]),
        "off_on_accuracy": float(flat["off_on_state_accuracy"]),
        "binary_transition_precision": float(flat["binary_transition_precision"]),
        "binary_transition_recall": float(flat["binary_transition_recall"]),
        "binary_transition_f1": float(flat["binary_transition_f1"]),
        "transition_tolerance_f1": {
            tolerance: float(timing[f"plus_minus_{tolerance}_samples"]["combined"]["f1"])
            for tolerance in (1, 2, 4)
        },
        "short_repedal_precision": float(renderer["short_repedal_like"]["precision"]),
        "short_repedal_recall": float(renderer["short_repedal_like"]["recall"]),
        "short_repedal_f1": float(renderer["short_repedal_like"]["f1"]),
    }


def _write_comparison_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze_and_report(
    output_dir: str | Path,
    baseline_dir: str | Path,
    configuration: Mapping[str, Any],
    tests: Mapping[str, Any],
    memory_smoke: Mapping[str, Any],
    overfit: Mapping[str, Any],
    runtime: Mapping[str, Any],
    training: Mapping[str, Any],
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(output_dir).resolve()
    baseline_root = Path(baseline_dir).resolve()
    baseline_eval = json.loads(
        (baseline_root / "validation/evaluation.json").read_text(encoding="utf-8")
    )
    baseline_status = json.loads(
        (baseline_root / "run_status.json").read_text(encoding="utf-8")
    )
    summaries = {
        "baseline_teacher_forced": _mode_summary(baseline_eval, "teacher_forced", TEACHER_MODEL),
        "baseline_free_running": _mode_summary(baseline_eval, "free_running", FREE_MODEL),
        "scheduled_teacher_forced": _mode_summary(evaluation, "teacher_forced", TEACHER_MODEL),
        "scheduled_free_running": _mode_summary(evaluation, "free_running", FREE_MODEL),
    }
    btf, bfr = summaries["baseline_teacher_forced"], summaries["baseline_free_running"]
    stf, sfr = summaries["scheduled_teacher_forced"], summaries["scheduled_free_running"]
    baseline_gap = btf["token_accuracy"] - bfr["token_accuracy"]
    scheduled_gap = stf["token_accuracy"] - sfr["token_accuracy"]
    interpretation = {
        "free_running_token_accuracy_improved": sfr["token_accuracy"] > bfr["token_accuracy"],
        "teacher_free_accuracy_gap_reduced": scheduled_gap < baseline_gap,
        "free_running_macro_f1_improved": sfr["macro_f1"] > bfr["macro_f1"],
        "low_recall_improved": sfr["per_class"]["LOW"]["recall"] > bfr["per_class"]["LOW"]["recall"],
        "mid_recall_improved": sfr["per_class"]["MID"]["recall"] > bfr["per_class"]["MID"]["recall"],
        "high_recall_improved": sfr["per_class"]["HIGH"]["recall"] > bfr["per_class"]["HIGH"]["recall"],
        "high_to_full_collapse_reduced": sfr["high_to_full"]["rate"] < bfr["high_to_full"]["rate"],
        "nonendpoint_endpoint_collapse_reduced": (
            sfr["nonendpoint_endpoint_collapse_ratio"]
            < bfr["nonendpoint_endpoint_collapse_ratio"]
        ),
        "binary_transition_f1_improved": sfr["binary_transition_f1"] > bfr["binary_transition_f1"],
        "short_repedal_f1_improved": sfr["short_repedal_f1"] > bfr["short_repedal_f1"],
        "teacher_metric_tradeoff_for_free_running_gain": bool(
            stf["token_accuracy"] < btf["token_accuracy"]
            and sfr["token_accuracy"] > bfr["token_accuracy"]
        ),
    }
    interpretation["trajectory_modeling_meaningful"] = bool(
        interpretation["free_running_token_accuracy_improved"]
        and interpretation["teacher_free_accuracy_gap_reduced"]
        and (
            interpretation["free_running_macro_f1_improved"]
            or interpretation["binary_transition_f1_improved"]
            or interpretation["short_repedal_f1_improved"]
        )
    )
    result = {
        "baseline": {
            "run": str(baseline_root),
            "best_epoch": int(baseline_status["best_epoch"]),
            "best_validation_teacher_forced_weighted_ce": float(
                baseline_status["best_validation_loss"]
            ),
            "teacher_free_accuracy_gap": baseline_gap,
        },
        "scheduled_sampling": {
            "run": str(root),
            "best_epoch": int(training["best_epoch"]),
            "best_validation_teacher_forced_weighted_ce": float(training["best_validation_loss"]),
            "teacher_free_accuracy_gap": scheduled_gap,
        },
        "metrics": summaries,
        "interpretation": interpretation,
        "asap_test_split_accessed": False,
    }
    atomic_json(root / "scheduled_sampling_comparison.json", result)
    rows = []
    for name, values in summaries.items():
        rows.append({
            "model_mode": name,
            "token_accuracy": values["token_accuracy"],
            "macro_f1": values["macro_f1"],
            "weighted_f1": values["weighted_f1"],
            "low_recall": values["per_class"]["LOW"]["recall"],
            "low_f1": values["per_class"]["LOW"]["f1"],
            "mid_recall": values["per_class"]["MID"]["recall"],
            "mid_f1": values["per_class"]["MID"]["f1"],
            "high_recall": values["per_class"]["HIGH"]["recall"],
            "high_f1": values["per_class"]["HIGH"]["f1"],
            "high_to_full_rate": values["high_to_full"]["rate"],
            "endpoint_collapse_ratio": values["nonendpoint_endpoint_collapse_ratio"],
            "overall_mae": values["overall_decoded_mae"],
            "on_region_mae": values["on_region_mae"],
            "off_on_accuracy": values["off_on_accuracy"],
            "binary_transition_f1": values["binary_transition_f1"],
            "transition_f1_pm1": values["transition_tolerance_f1"][1],
            "transition_f1_pm2": values["transition_tolerance_f1"][2],
            "transition_f1_pm4": values["transition_tolerance_f1"][4],
            "short_repedal_f1": values["short_repedal_f1"],
        })
    _write_comparison_csv(root / "scheduled_sampling_comparison.csv", rows)
    plot = {"created": False}
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def read_metrics(path: Path):
            with path.open(newline="", encoding="utf-8") as handle:
                return list(csv.DictReader(handle))

        baseline_rows = read_metrics(baseline_root / "metrics.csv")
        scheduled_rows = read_metrics(root / "metrics.csv")
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        for rows_, label in ((baseline_rows, "100% TF"), (scheduled_rows, "scheduled")):
            epochs = [int(row["epoch"]) for row in rows_]
            axes[0].plot(epochs, [float(row["train_loss"]) for row in rows_], marker="o", label=label)
            axes[1].plot(
                epochs,
                [float(row["validation_teacher_forced_weighted_ce"]) for row in rows_],
                marker="o", label=label,
            )
        axes[0].set_title("Train weighted CE")
        axes[1].set_title("Validation teacher-forced weighted CE")
        for axis in axes:
            axis.set_xlabel("Epoch")
            axis.grid(alpha=0.25)
            axis.legend()
        figure.tight_layout()
        path = root / "learning_curves.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        plot = {"created": True, "path": str(path)}
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        plot = {"created": False, "reason": f"{type(exc).__name__}: {exc}"}
    result["learning_curve_plot"] = plot
    atomic_json(root / "scheduled_sampling_comparison.json", result)

    report = f"""# Encoder-Decoder Weighted 5-class Scheduled Sampling Report

## Controlled scope

Reference: {baseline_root}. Architecture, initialization, ASAP train/validation split,
five classes/representatives, train-only weights, weighted CE, optimizer/LRs, effective
batch, AMP, clipping, window/stride, early stopping and checkpoint selection are copied
from the completed baseline. The only intended training change is decoder history:
100% teacher forcing becomes the configured scheduled-sampling schedule. ASAP test,
MAESTRO, score inference, audio rendering and a new tokenizer were not used.

## Exact history semantics

For p_tf < 1, history is rolled out strictly left-to-right with greedy detached predictions
and no future ground truth. The encoder/history rollout is no-grad with native KV cache.
The completed discrete history is then held fixed while one causal parallel forward computes
the same weighted CE and full encoder/decoder/output gradients. Epochs 1-2 use the unchanged
baseline parallel teacher-forcing path.

## Schedule

epoch1=1.00, epoch2=1.00, epoch3=0.90, epoch4=0.80, epoch5=0.70, epoch6+=0.60

## Preflight

Tests: {tests}

Memory smoke: {memory_smoke}

Pedal-rich overfit: {overfit}

Runtime benchmark: {runtime}

## Training

{training}

## Baseline versus scheduled sampling

{result}

## Interpretation

1. Free-running token accuracy improved: {interpretation['free_running_token_accuracy_improved']}
2. Teacher-forced/free-running gap reduced: {interpretation['teacher_free_accuracy_gap_reduced']}
3. Free-running macro F1 improved: {interpretation['free_running_macro_f1_improved']}
4. LOW/MID/HIGH recall improved: {interpretation['low_recall_improved']} /
   {interpretation['mid_recall_improved']} / {interpretation['high_recall_improved']}
5. HIGH→FULL collapse reduced: {interpretation['high_to_full_collapse_reduced']}
6. Nonendpoint→endpoint collapse reduced: {interpretation['nonendpoint_endpoint_collapse_reduced']}
7. Binary transition F1 improved: {interpretation['binary_transition_f1_improved']}
8. Short repedal-like F1 improved: {interpretation['short_repedal_f1_improved']}
9. Teacher metric cost with free-running gain: {interpretation['teacher_metric_tradeoff_for_free_running_gain']}
10. Meaningful trajectory-modeling improvement: {interpretation['trajectory_modeling_meaningful']}
"""
    atomic_text(root / "SCHEDULED_SAMPLING_5CLASS_REPORT.md", report)
    return result


__all__ = ["analyze_and_report", "benchmark"]
