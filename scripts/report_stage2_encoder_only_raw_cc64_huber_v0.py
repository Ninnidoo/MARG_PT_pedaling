#!/usr/bin/env python3
"""Generate raw-CC64 Huber training, validation, and baseline-comparison reports."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_stage2_encoder_only_raw_cc64_huber_v0 import read_json, update_status


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value.rstrip() + "\n", encoding="utf-8")
    os.replace(temporary, path)


def metric_row(value: Mapping[str, Any]) -> tuple[Any, ...]:
    classification = value["classification"]
    transition = value["transition"]["pooled"]
    patterns = value["patterns"]
    return (
        classification["token_accuracy"], classification["macro_f1"],
        transition["precision"], transition["recall"], transition["f1"],
        patterns["js_divergence_base2"], patterns["intersection"],
    )


def training_report(
    config: Mapping[str, Any], status: Mapping[str, Any], rows: list[dict[str, str]]
) -> str:
    output = Path(config["output_root"])
    lines = [
        "# Raw CC64 Huber Training Report", "",
        "- Objective: Smooth L1/Huber on cached raw Pedal1–4 CC64 values normalized by 127",
        "- Target quantization: none; no canonical-class or representative mapping in training targets",
        "- Architecture: pretrained PT 10-layer encoder + four independent Linear(768,1) heads",
        "- Training output: unconstrained scalar; no sigmoid and no loss-time clipping",
        "- Huber delta: 14 CC64; normalized delta 14/127",
        "- Data: MAESTRO-clean 1,170 + ASAP train 892; ASAP validation 71",
        "- Seed: 42; fresh pretrained initialization; Representative Huber checkpoint initialization: false",
        "- Optimizer: AdamW; encoder LR 1e-5; head LR 1e-4; weight decay 0.01",
        "- Window/stride: 512/256; micro/effective batch: 16/16; accumulation: 1; FP16 AMP",
        f"- Best epoch: {status.get('best_epoch')}",
        f"- Best validation Huber: {status.get('best_validation_objective')}",
        f"- Checkpoints: `{output / 'best.pt'}`, `{output / 'last.pt'}`",
        "- ASAP test access count: 0", "", "## Epochs", "",
        "| Epoch | Train Huber | Validation Huber | Validation Raw MAE | Validation Raw RMSE | Encoder LR | Head LR | Seconds |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['epoch']} | {float(row['train_huber']):.9f} | {float(row['validation_huber']):.9f} | "
            f"{float(row['validation_raw_cc64_mae']):.6f} | {float(row['validation_raw_cc64_rmse']):.6f} | "
            f"{float(row['encoder_lr']):.8g} | {float(row['head_lr']):.8g} | {float(row['epoch_seconds']):.1f} |"
        )
    return "\n".join(lines)


def validation_report(raw: Mapping[str, Any], baseline: Mapping[str, Any]) -> str:
    classification = raw["classification"]
    transition = raw["transition"]
    patterns = raw["patterns"]
    regression = raw["regression_diagnostics"]
    errors = classification["error_diagnostics"]
    return "\n".join([
        "# Raw CC64 Huber Canonical Validation Report", "",
        "- ASAP validation only; ASAP test access count: 0",
        "- Frozen 4-class boundaries: ZERO 0–25, LOW 26–63, HALF 64–103, FULL 104–127",
        "- Transition: CC64 64 crossing, direction-aware one-to-one, tolerance ±1 distinct onset",
        "- Repedal metric execution count: 0",
        f"- Common Representative-Huber universe: {raw['successful_pairs']} pairs / {raw['aligned_notes']} aligned notes",
        "", "## Main metrics", "",
        "| 4C Acc | Macro F1 | Transition P | Transition R | Transition F1 | JS | Intersection |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        f"| {classification['token_accuracy']:.6f} | {classification['macro_f1']:.6f} | "
        f"{transition['pooled']['precision']:.6f} | {transition['pooled']['recall']:.6f} | "
        f"{transition['pooled']['f1']:.6f} | {patterns['js_divergence_base2']:.6f} | {patterns['intersection']:.6f} |",
        "", "## Regression diagnostics", "",
        f"- Raw CC64 MAE/RMSE: {regression['raw_cc64_mae']:.6f} / {regression['raw_cc64_rmse']:.6f}",
        f"- Prediction mean/std: {regression['prediction_mean']:.6f} / {regression['prediction_std']:.6f}",
        f"- Target mean/std: {regression['target_mean']:.6f} / {regression['target_std']:.6f}",
        f"- Prediction quantiles: `{regression['prediction_quantiles']}`",
        f"- Pre-clip min/max: {regression['raw_prediction_minimum_before_clipping']:.6f} / {regression['raw_prediction_maximum_before_clipping']:.6f}",
        f"- Clip-at-0 / clip-at-127: {regression['clip_at_0_proportion']:.6f} / {regression['clip_at_127_proportion']:.6f}",
        f"- Absolute error by target range: `{regression['absolute_error_by_target_range']}`",
        "", "## Classification diagnostics", "",
        f"- Prediction distribution: `{classification['prediction_class_distribution']}`",
        f"- Class P/R/F1: `{classification['class_metrics']}`",
        f"- Confusion matrix (rows target, columns prediction): `{classification['confusion_matrix']}`",
        f"- Adjacent directional errors: `{errors['adjacent_directional_counts']}`",
        f"- Same-side errors: `{errors['same_on_off_side_error']}`",
        f"- 64-boundary-crossing errors: `{errors['boundary_64_crossing_error']}`",
        "", "## Transition diagnostics", "",
        f"- Pooled: `{transition['pooled']}`",
        f"- UP: `{transition['up']}`",
        f"- DOWN: `{transition['down']}`",
        f"- Representative Huber baseline candidate transitions: {baseline['transition']['pooled']['candidate']}",
        "", "## Pattern diagnostics", "",
        f"- Top predicted patterns: `{patterns['top_candidate_patterns']}`",
        f"- Top target patterns: `{patterns['top_target_patterns']}`",
    ])


def comparison_report(
    raw: Mapping[str, Any], baseline: Mapping[str, Any], status: Mapping[str, Any]
) -> str:
    raw_main = metric_row(raw)
    base_main = metric_row(baseline)
    raw_classes = raw["classification"]["class_metrics"]
    base_classes = baseline["classification"]["class_metrics"]
    raw_transition = raw["transition"]["pooled"]
    base_transition = baseline["transition"]["pooled"]
    raw_patterns = raw["patterns"]
    base_patterns = baseline["patterns"]
    errors = raw["classification"]["error_diagnostics"]
    intermediate_better = raw_classes[1]["recall"] > base_classes[1]["recall"] and raw_classes[2]["recall"] > base_classes[2]["recall"]
    transition_better = raw_transition["f1"] > base_transition["f1"]
    patterns_better = raw_patterns["js_divergence_base2"] < base_patterns["js_divergence_base2"] and raw_patterns["intersection"] > base_patterns["intersection"]
    overall_evidence = raw_main[0] > base_main[0] or raw_main[1] > base_main[1] or intermediate_better
    return "\n".join([
        "# Raw CC64 vs Representative Huber", "",
        "ASAP validation only; identical architecture/training schedule/inference conversion; target representation is the intended independent variable. ASAP test access count: 0.",
        "", "| Objective | 4C Acc ↑ | Macro F1 ↑ | Trans P ↑ | Trans R ↑ | Trans F1 ↑ | JS ↓ | Intersection ↑ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| Representative Huber | {base_main[0]:.6f} | {base_main[1]:.6f} | {base_main[2]:.6f} | {base_main[3]:.6f} | {base_main[4]:.6f} | {base_main[5]:.6f} | {base_main[6]:.6f} |",
        f"| Raw CC64 Huber | {raw_main[0]:.6f} | {raw_main[1]:.6f} | {raw_main[2]:.6f} | {raw_main[3]:.6f} | {raw_main[4]:.6f} | {raw_main[5]:.6f} | {raw_main[6]:.6f} |",
        "", "| Objective | LOW Recall | HALF Recall | Candidate Transitions | Best Epoch |",
        "|---|---:|---:|---:|---:|",
        f"| Representative Huber | {base_classes[1]['recall']:.6f} | {base_classes[2]['recall']:.6f} | {base_transition['candidate']} | 6 |",
        f"| Raw CC64 Huber | {raw_classes[1]['recall']:.6f} | {raw_classes[2]['recall']:.6f} | {raw_transition['candidate']} | {status['best_epoch']} |",
        "", "## Q1. Intermediate depth", "",
        f"Confirmed: LOW recall changed by {raw_classes[1]['recall'] - base_classes[1]['recall']:+.6f}; HALF recall by {raw_classes[2]['recall'] - base_classes[2]['recall']:+.6f}. Both improved: {intermediate_better}.",
        "", "## Q2. Transition behavior", "",
        f"Confirmed: Transition F1 changed by {raw_transition['f1'] - base_transition['f1']:+.6f}, precision by {raw_transition['precision'] - base_transition['precision']:+.6f}, and candidate count by {raw_transition['candidate'] - base_transition['candidate']:+d}. Improved F1: {transition_better}.",
        "", "## Q3. Global 256-pattern distribution", "",
        f"Confirmed: JS changed by {raw_patterns['js_divergence_base2'] - base_patterns['js_divergence_base2']:+.6f}; Intersection by {raw_patterns['intersection'] - base_patterns['intersection']:+.6f}. Both improved: {patterns_better}.",
        "", "## Q4. Confusion pattern", "",
        f"Confirmed raw-model counts: adjacent `{errors['adjacent_undirected_counts']}`, same-side `{errors['same_on_off_side_error']}`, 64-crossing `{errors['boundary_64_crossing_error']}`. These counts identify which errors accompanied the accuracy change; causal attribution remains an inference.",
        "", "## Q5. Evidence for continuous raw targets", "",
        f"Confirmed: the run changed training targets from representatives to unquantized cached raw CC64 while holding the model and schedule fixed. Validation evidence favoring raw targets on at least one depth/main metric: {overall_evidence}. Inference: any gain is consistent with reduced target quantization, but one validation run does not establish a general causal advantage.",
        "", "No hybrid loss, delta sweep, smoothing, temporal loss, architecture change, ASAP-test evaluation, or Repedal experiment was started.",
    ])


def run(config: Mapping[str, Any]) -> int:
    output = Path(config["output_root"])
    status = read_json(output / "run_status.json")
    raw = read_json(output / "validation_eval" / "evaluation.json")
    baseline = read_json(config["representative_huber_evaluation"])
    with (output / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    atomic_text(
        output / "RAW_CC64_HUBER_TRAINING_REPORT.md",
        training_report(config, status, rows),
    )
    atomic_text(
        output / "validation_eval" / "RAW_CC64_HUBER_VALIDATION_REPORT.md",
        validation_report(raw, baseline),
    )
    atomic_text(
        output / "RAW_CC64_VS_REPRESENTATIVE_HUBER_REPORT.md",
        comparison_report(raw, baseline, status),
    )
    update_status(
        config, training_report_generated=True,
        validation_report_generated=True, comparison_report_generated=True,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("report generation is guarded; pass --execute")
    return run(read_json(args.config))


if __name__ == "__main__":
    raise SystemExit(main())
