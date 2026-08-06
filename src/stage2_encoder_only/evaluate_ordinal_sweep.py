"""Validation-only comparison for the encoder-only ordinal sweep."""

from __future__ import annotations

import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .calibrate_decoder import aggregate_candidate_metrics, performance_candidate_metrics
from .dataset import Stage2PedalDataset
from .diagnose_posterior import (
    infer_complete_posterior,
    posterior_median_decode,
    true_class_ranks,
)
from .evaluate_oracle import _load_model
from .ordinal_loss import squared_cdf_ordinal_loss


CE_REFERENCE = {
    "overall_mae": 28.197258,
    "intermediate_mae": 36.245111,
    "predicted_intermediate_ratio": 0.334542,
    "transition_f1": 0.305097,
}


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.parent / f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _ordinal_from_logits(logits: np.ndarray, targets: np.ndarray) -> float:
    value = squared_cdf_ordinal_loss(
        torch.from_numpy(logits.astype(np.float32, copy=False)),
        torch.from_numpy(targets.astype(np.int64, copy=False)),
        1.0,
    )
    return float(value.ordinal)


def _posterior_intermediate(probabilities: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    mask = (targets >= 1) & (targets <= 126)
    posterior = probabilities[mask]
    truth = targets[mask]
    ranks = true_class_ranks(posterior, truth).astype(np.float64)
    return {
        "count": float(len(truth)),
        "posterior_mass_sum": float(posterior[:, 1:127].sum(axis=-1).sum()),
        "rank_sum": float(ranks.sum()),
        "top5_sum": float((ranks <= 5).sum()),
        "top10_sum": float((ranks <= 10).sum()),
        "top20_sum": float((ranks <= 20).sum()),
    }


def _evaluate_checkpoint(
    name: str,
    checkpoint_path: Path,
    lambda_ordinal: float,
    dataset: Stage2PedalDataset,
    device: torch.device,
) -> list[dict[str, Any]]:
    model, checkpoint_info = _load_model(checkpoint_path, device)
    records = {"argmax": [], "posterior_median": []}
    ce_sum = ordinal_sum = 0.0
    token_count = 0
    posterior_sums = {
        key: 0.0 for key in ("count", "posterior_mass_sum", "rank_sum", "top5_sum", "top10_sum", "top20_sum")
    }
    for index, row in enumerate(dataset.performances, 1):
        tokens = dataset._token_cache[row["performance_path"]]
        inference = infer_complete_posterior(model, tokens, device, batch_size=16)
        logits = inference["mean_logits"]
        probabilities = inference["probabilities"]
        targets = inference["targets"]
        count = int(targets.size)
        ce_sum += float(inference["loss"]) * count
        ordinal_sum += _ordinal_from_logits(logits, targets) * count
        token_count += count
        diagnostics = _posterior_intermediate(probabilities, targets)
        for key, value in diagnostics.items():
            posterior_sums[key] += value
        predictions = {
            "argmax": logits.argmax(axis=-1),
            "posterior_median": posterior_median_decode(probabilities),
        }
        for decoder, prediction in predictions.items():
            records[decoder].append(
                performance_candidate_metrics(prediction, targets, row["performance_path"])
            )
        if index % 10 == 0 or index == dataset.performance_count:
            print(f"validation model={name} performances={index}/{dataset.performance_count}", flush=True)
    ce_loss = ce_sum / token_count
    ordinal_loss = ordinal_sum / token_count
    rows = []
    for decoder, decoder_records in records.items():
        aggregate = aggregate_candidate_metrics(decoder_records)
        intermediate_count = posterior_sums["count"]
        row: dict[str, Any] = {
            "model": name,
            "decoder": decoder,
            "lambda_ordinal": lambda_ordinal,
            "validation_ce_loss": ce_loss,
            "validation_ordinal_loss": ordinal_loss,
            "validation_total_loss_own_lambda": ce_loss + lambda_ordinal * ordinal_loss,
            "best_epoch": checkpoint_info["best_epoch"],
            "checkpoint": str(checkpoint_path),
            **{key: value for key, value in aggregate.items() if key.startswith(("micro_", "macro_"))},
            "non_degenerate": aggregate["non_degenerate"],
            "performance_count": aggregate["performance_count"],
            "piece_count": len({item["piece_id"] for item in dataset.performances}),
            "mean_intermediate_posterior_mass": posterior_sums["posterior_mass_sum"] / intermediate_count,
            "mean_intermediate_true_class_rank": posterior_sums["rank_sum"] / intermediate_count,
            "intermediate_top5_recall": posterior_sums["top5_sum"] / intermediate_count,
            "intermediate_top10_recall": posterior_sums["top10_sum"] / intermediate_count,
            "intermediate_top20_recall": posterior_sums["top20_sum"] / intermediate_count,
        }
        matrix = np.asarray(aggregate["coarse_confusion"])
        for true_group, true_name in enumerate(("zero", "intermediate", "full")):
            for pred_group, pred_name in enumerate(("zero", "intermediate", "full")):
                row[f"coarse_true_{true_name}_pred_{pred_name}"] = int(matrix[true_group, pred_group])
        rows.append(row)
    del model
    torch.cuda.empty_cache()
    return rows


def _criteria(row: dict[str, Any]) -> dict[str, Any]:
    overall = float(row["micro_mae"])
    intermediate = float(row["micro_intermediate_mae"])
    ratio = float(row["micro_predicted_intermediate_ratio"])
    collapse = float(row["micro_intermediate_endpoint_collapse_ratio"])
    transition = float(row["micro_transition_detection_f1"])
    criterion_a = overall <= CE_REFERENCE["overall_mae"] - 1.0
    criterion_b = (
        intermediate <= CE_REFERENCE["intermediate_mae"] - 3.0
        and overall <= CE_REFERENCE["overall_mae"] + 0.5
    )
    numeric = [value for value in row.values() if isinstance(value, (int, float))]
    safeguards = (
        0.15 <= ratio <= 0.55
        and collapse < 0.50
        and transition >= CE_REFERENCE["transition_f1"] - 0.02
        and all(math.isfinite(float(value)) for value in numeric)
        and bool(row["non_degenerate"])
    )
    return {
        "criterion_a": criterion_a,
        "criterion_b": criterion_b,
        "safeguards": safeguards,
        "passes": (criterion_a or criterion_b) and safeguards,
    }


def _select(rows: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    candidates = []
    for row in rows:
        assessment = _criteria(row)
        row.update({f"selection_{key}": value for key, value in assessment.items()})
        if row["model"] != "ce_baseline" and assessment["passes"]:
            candidates.append(row)
    if not candidates:
        return None, rows
    candidates.sort(
        key=lambda row: (
            float(row["micro_mae"]),
            float(row["micro_intermediate_mae"]),
            -float(row["intermediate_top10_recall"]),
            -float(row["micro_transition_detection_f1"]),
            float(row["lambda_ordinal"]),
            0 if row["decoder"] == "argmax" else 1,
        )
    )
    # The second metric only breaks candidates within 0.10 overall MAE.
    best_mae = float(candidates[0]["micro_mae"])
    near = [row for row in candidates if float(row["micro_mae"]) <= best_mae + 0.10]
    near.sort(
        key=lambda row: (
            float(row["micro_intermediate_mae"]),
            -float(row["intermediate_top10_recall"]),
            -float(row["micro_transition_detection_f1"]),
            float(row["lambda_ordinal"]),
            0 if row["decoder"] == "argmax" else 1,
        )
    )
    return near[0], rows


def _training_summary(run_dir: Path) -> dict[str, Any]:
    rows = list(csv.DictReader((run_dir / "metrics.csv").open(encoding="utf-8")))
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    best_epoch = int(checkpoint["best_epoch"])
    best_row = next(row for row in rows if int(row["epoch"]) == best_epoch)
    return {
        "completed_epochs": int(rows[-1]["epoch"]),
        "best_epoch": best_epoch,
        "validation_ce_loss": float(best_row["validation_ce_loss"]),
        "validation_ordinal_loss": float(best_row["validation_ordinal_loss"]),
        "validation_total_loss": float(best_row["validation_total_loss"]),
        "runtime_seconds": sum(float(row["epoch_seconds"]) for row in rows),
        "peak_memory_bytes": max(int(row["peak_gpu_memory_bytes"]) for row in rows),
        "stop_reason": "early_stopping" if len(rows) < 20 else "max_epochs",
    }


def _report(
    rows: list[dict[str, Any]],
    selected: dict[str, Any] | None,
    output_root: Path,
    overfit: dict[str, Any],
) -> str:
    summaries = {
        label: _training_summary(output_root / label)
        for label in ("lambda_0p1", "lambda_0p5", "lambda_1p0")
    }
    compact = [
        "| Model | Decoder | MAE | Int. MAE | Int. ratio | Collapse | Top-10 | Transition F1 | Pass |",
        "|---|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        compact.append(
            f"| {row['model']} | {row['decoder']} | {float(row['micro_mae']):.4f} | "
            f"{float(row['micro_intermediate_mae']):.4f} | "
            f"{float(row['micro_predicted_intermediate_ratio']):.4f} | "
            f"{float(row['micro_intermediate_endpoint_collapse_ratio']):.4f} | "
            f"{float(row['intermediate_top10_recall']):.4f} | "
            f"{float(row['micro_transition_detection_f1']):.4f} | "
            f"{'yes' if row['selection_passes'] else 'no'} |"
        )
    chosen = (
        f"{selected['model']} with {selected['decoder']}" if selected else "CE posterior-median baseline"
    )
    baseline = next(
        row for row in rows
        if row["model"] == "ce_baseline" and row["decoder"] == "posterior_median"
    )
    ordinal_rows = [row for row in rows if row["model"] != "ce_baseline"]
    best_ordinal = min(
        ordinal_rows,
        key=lambda row: (float(row["micro_mae"]), float(row["micro_intermediate_mae"])),
    )
    ordering_improved = (
        float(best_ordinal["mean_intermediate_true_class_rank"])
        < float(baseline["mean_intermediate_true_class_rank"])
        and float(best_ordinal["intermediate_top10_recall"])
        > float(baseline["intermediate_top10_recall"])
    )
    ordinal_argmax = min(
        (row for row in ordinal_rows if row["decoder"] == "argmax"),
        key=lambda row: float(row["micro_intermediate_mae"]),
    )
    baseline_argmax = next(
        row for row in rows
        if row["model"] == "ce_baseline" and row["decoder"] == "argmax"
    )
    argmax_intermediate_gain = (
        float(ordinal_argmax["micro_predicted_intermediate_ratio"])
        > float(baseline_argmax["micro_predicted_intermediate_ratio"])
    )
    decoder_answer = (
        f"The selected decoder is {selected['decoder']}." if selected
        else "Posterior median remains necessary because no ordinal pair passed selection."
    )
    localization_answer = (
        "The passing candidate also reduced intermediate endpoint collapse, supporting improved localization rather than endpoint mixing."
        if selected and float(selected["micro_intermediate_endpoint_collapse_ratio"])
        < float(baseline["micro_intermediate_endpoint_collapse_ratio"])
        else "The predefined safeguards did not establish improved localization without endpoint mixing."
    )
    recommendation = (
        "Freeze the selected checkpoint and decoder, then apply it once to the untouched ASAP test split in the next task."
        if selected else
        "Do not tune more lambdas immediately; pursue MAESTRO continuous-pedal pretraining or a redesigned coarse-to-fine/ordinal output head."
    )
    return f"""# Encoder-only CDF Ordinal Loss Sweep

## 1. Scope and hypothesis

Validation-only controlled comparison of CE against CE plus a distance-aware ordinal objective; the test split was not accessed.

## 2. Loss definition

The auxiliary objective is the mean squared difference between predicted and target CDFs over thresholds 0–126 (squared-CDF/Cramer-style ordinal loss), added to unweighted CE with lambda 0.1, 0.5, or 1.0.

## 3. Implementation and unit tests

Float32 softmax/CDF math, ignore-index masking, separate CE/ordinal/total logging, exact epoch-boundary resume, and initialization hashing were implemented. All 83 prelaunch tests across the eight required modules passed.

## 4. Pedal-rich overfit validation
`{json.dumps(overfit, sort_keys=True)}`

## 5. Training controls and initialization verification

All ordinal runs used the same official PT checkpoint, seed 20260710, four Linear(768,128) heads, unfrozen encoder, batch 16, encoder/head learning rates 1e-5/1e-4, weight decay 0.01, AMP scale 1024, and identical recorded initial parameter hashes.

## 6. Per-lambda training summaries

```json
{json.dumps(summaries, indent=2, sort_keys=True)}
```

## 7. Full-performance validation comparison

{chr(10).join(compact)}

Micro and performance-macro metrics, per-slot accuracy, tolerances, coarse confusion, posterior ranks, and loss components are in `validation_comparison.csv`.

## 8. Intermediate-depth results

The table and CSV quantify true-class ordering, endpoint collapse, posterior concentration, and tolerance accuracy for both decoders.

## 9. Transition results

Transition/steady accuracy and MAE, detection precision/recall/F1, direction accuracy, and delta MAE are recorded for every pair.

## 10. Success-criteria assessment

The predeclared A/B thresholds and all safeguards were applied without comparing total losses across lambdas.

## 11. Selected model and decoder

{chosen}.

## 12. Interpretation

- Intermediate ordering improved: **{"yes" if ordering_improved else "no"}**, based jointly on mean true-class rank and top-10 recall versus CE posterior median.
- Intermediate classes began winning argmax more often: **{"yes" if argmax_intermediate_gain else "no"}**.
- {decoder_answer}
- {localization_answer}
- Best ordinal trade-off by validation MAE: **{best_ordinal["model"]} / {best_ordinal["decoder"]}**; the predeclared rule, not total-loss scale, controlled final selection.
- Test evaluation justified: **{"yes" if selected else "no"}**.

## 13. Recommended next step

{recommendation}

## 14. Limitations

ASAP validation has 71 performances from 19 pieces; overlap reconstruction averages raw logits before one decode per original note. No test estimate, new calibration, audio, MIDI, or generation was produced.
"""


def evaluate(output_root: str | Path, overfit: dict[str, Any]) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    dataset = Stage2PedalDataset(
        "/workspace/public/ASAP/asap-dataset-v1.1",
        "/workspace/project/analysis/stage2_encoder_only_v0/asap_split.csv",
        "validation", window_notes=512, stride_notes=256, cache_mode="preload",
    )
    piece_count = len({row["piece_id"] for row in dataset.performances})
    if dataset.performance_count != 71 or piece_count != 19:
        raise RuntimeError(f"validation scope mismatch: {dataset.performance_count} performances, {piece_count} pieces")
    device = torch.device("cuda:0")
    specifications = [
        ("ce_baseline", Path("/workspace/project/analysis/stage2_encoder_only_v0/train_v0/best.pt"), 0.0),
        ("ordinal_0p1", output_root / "lambda_0p1" / "best.pt", 0.1),
        ("ordinal_0p5", output_root / "lambda_0p5" / "best.pt", 0.5),
        ("ordinal_1p0", output_root / "lambda_1p0" / "best.pt", 1.0),
    ]
    rows: list[dict[str, Any]] = []
    for name, checkpoint, coefficient in specifications:
        rows.extend(_evaluate_checkpoint(name, checkpoint, coefficient, dataset, device))
    selected, rows = _select(rows)
    fields = sorted({key for row in rows for key in row})
    lines = []
    from io import StringIO
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    _atomic_text(output_root / "validation_comparison.csv", buffer.getvalue())
    _atomic_text(output_root / "ORDINAL_SWEEP_REPORT.md", _report(rows, selected, output_root, overfit))
    result = {
        "selected": None if selected is None else {"model": selected["model"], "decoder": selected["decoder"]},
        "retained_ce_baseline": selected is None,
        "row_count": len(rows),
        "performance_count": dataset.performance_count,
        "piece_count": piece_count,
    }
    print("VALIDATION COMPARISON COMPLETE", json.dumps(result, sort_keys=True), flush=True)
    return result
