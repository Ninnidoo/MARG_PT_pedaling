#!/usr/bin/env python3
"""Generate training, validation, and comparison reports for Phase 3-B v0."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from scripts.run_stage2_4class_architecture_v0 import read_json
from scripts.run_stage2_encoder_only_raw_cc64_huber_v0 import update_status


BASELINES = [
    ("Standard CE", .569449, .396136, .354366, .571714, .437535, .068187, .815823),
    ("Weighted CE", .533587, .408260, .340975, .602783, .435565, .039055, .858292),
    ("Representative Huber", .477443, .396158, .471488, .521667, .495310, .042943, .810684),
    ("Raw CC64 Huber", .492320, .403413, .497066, .512063, .504453, .038859, .825369),
]


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text.rstrip()+"\n", encoding="utf-8"); os.replace(temporary, path)


def main_values(value: Mapping[str, Any]) -> tuple[float, ...]:
    c=value["classification"]; t=value["transition"]["pooled"]; p=value["patterns"]
    return c["token_accuracy"],c["macro_f1"],t["precision"],t["recall"],t["f1"],p["js_divergence_base2"],p["intersection"]


def training_report(config: Mapping[str, Any], status: Mapping[str, Any], rows: list[dict[str,str]]) -> str:
    output=Path(config["output_root"]); lines=["# Raw Huber + Auxiliary CE Clean Rerun Report","",
        "- Primary target/loss: unquantized cached raw CC64 / 127, Smooth L1 delta 14/127",
        "- Auxiliary target/loss: canonical bins 0–25/26–63/64–103/104–127, unweighted Standard CE",
        "- Total objective: Huber + 0.1 × CE; checkpoint selection: minimum validation total",
        "- Architecture: pretrained PT encoder + four Linear(768,1) regression heads + four fresh Linear(768,4) auxiliary heads",
        "- Inference: regression output only; auxiliary candidate-MIDI usage count 0",
        "- Fresh seed 42 initialization; no Raw/Representative/CE/Weighted checkpoint warm-start",
        "- Data: MAESTRO-clean 1,170 + ASAP train 892; ASAP validation 71",
        "- AdamW: encoder LR 1e-5, both head types LR 1e-4, weight decay 0.01; batch 16; accumulation 1; FP16 AMP",
        "- Hybrid v0: interrupted during epoch 4 because a non-finite unscaled gradient was fatal before scaler.step/update",
        "- v0 audit: objectives/model/checkpoints finite; optimizer/GradScaler/RNG absent; exact original event cause not provable",
        "- Hybrid v1: clean seed-42 rerun from official PT pretrained initialization; old checkpoint warm-start false",
        "- Modeling hyperparameters changed from v0: none; AMP recovery/checkpoint infrastructure corrected",
        f"- Best epoch: {status.get('best_epoch')}; best validation total: {status.get('best_validation_objective')}",
        f"- AMP overflow events / max consecutive: {status.get('amp_overflow_events_total', 0)} / {status.get('max_consecutive_amp_overflows', 0)}",
        f"- Forward/model-parameter non-finite counts: {status.get('forward_nonfinite_count', 0)} / {status.get('parameter_nonfinite_count', 0)}",
        f"- AMP event details: `{status.get('overflow_events', [])}`",
        "- Checkpoint continuation state: model, optimizer, GradScaler, counters, Python/NumPy/Torch CPU/CUDA RNG",
        f"- Checkpoints: `{output/'best.pt'}`, `{output/'last.pt'}`","- ASAP test access: 0; Repedal execution: 0","","## Epochs","",
        "| Epoch | Train Huber | Train CE | Train Total | Val Huber | Val CE | Val Total | Val Raw MAE | AMP overflows | Scaler | Seconds |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows: lines.append(f"| {r['epoch']} | {float(r['train_huber']):.9f} | {float(r['train_ce']):.9f} | {float(r['train_total']):.9f} | {float(r['validation_huber']):.9f} | {float(r['validation_ce']):.9f} | {float(r['validation_total']):.9f} | {float(r['validation_raw_cc64_mae']):.6f} | {int(r['amp_overflow_events'])} | {float(r['grad_scaler_scale']):.1f} | {float(r['epoch_seconds']):.1f} |")
    return "\n".join(lines)


def validation_report(value: Mapping[str, Any]) -> str:
    c=value["classification"]; t=value["transition"]; p=value["patterns"]; r=value["regression_diagnostics"]; a=value["auxiliary_head_diagnostics"]; e=c["error_diagnostics"]
    m=main_values(value)
    return "\n".join(["# Raw Huber + Auxiliary CE Canonical Validation Report","",
        "ASAP validation only. Candidate MIDI was generated exclusively from regression scalars; auxiliary logits were never used.",
        "ASAP test access: 0. Repedal execution: 0.","","## Main metrics","",
        "| 4C Acc | Macro F1 | Transition P | Transition R | Transition F1 | JS | Intersection |","|---:|---:|---:|---:|---:|---:|---:|",
        f"| {m[0]:.6f} | {m[1]:.6f} | {m[2]:.6f} | {m[3]:.6f} | {m[4]:.6f} | {m[5]:.6f} | {m[6]:.6f} |","",
        "## Regression diagnostics","",f"- `{r}`","","## Regression-derived classification diagnostics","",
        f"- Prediction distribution: `{c['prediction_class_distribution']}`",f"- Class P/R/F1: `{c['class_metrics']}`",f"- Confusion matrix: `{c['confusion_matrix']}`",
        f"- Adjacent errors: `{e['adjacent_directional_counts']}`",f"- Same ON/OFF side: `{e['same_on_off_side_error']}`",f"- 64-boundary crossing: `{e['boundary_64_crossing_error']}`","",
        "## Transition diagnostics","",f"- Pooled: `{t['pooled']}`",f"- UP: `{t['up']}`",f"- DOWN: `{t['down']}`","",
        "## Auxiliary-head diagnostic (not candidate evaluation)","",f"- Accuracy/Macro F1: {a['accuracy']:.6f} / {a['macro_f1']:.6f}",f"- Prediction distribution: `{a['prediction_class_distribution']}`",
        "","## Pattern diagnostics","",f"- Top predicted: `{p['top_candidate_patterns']}`",f"- Top target: `{p['top_target_patterns']}`"])


def comparison_report(hybrid: Mapping[str, Any], raw: Mapping[str, Any], status: Mapping[str, Any]) -> str:
    hm=main_values(hybrid); rm=main_values(raw); hc=hybrid["classification"]; rc=raw["classification"]; ht=hybrid["transition"]["pooled"]; rt=raw["transition"]["pooled"]
    hlow=hc["class_metrics"][1]["recall"]; hhalf=hc["class_metrics"][2]["recall"]; rlow=rc["class_metrics"][1]["recall"]; rhalf=rc["class_metrics"][2]["recall"]
    rows=[f"| {name} | {a:.6f} | {f:.6f} | {tp:.6f} | {tr:.6f} | {tf:.6f} | {js:.6f} | {inter:.6f} |" for name,a,f,tp,tr,tf,js,inter in BASELINES]
    rows.append(f"| **Raw Huber + Aux CE λ=0.1** | {hm[0]:.6f} | {hm[1]:.6f} | {hm[2]:.6f} | {hm[3]:.6f} | {hm[4]:.6f} | {hm[5]:.6f} | {hm[6]:.6f} |")
    accuracy_gain=hm[0]-rm[0]; macro_gain=hm[1]-rm[1]; transition_gain=hm[4]-rm[4]; js_change=hm[5]-rm[5]; intersection_change=hm[6]-rm[6]
    return "\n".join(["# Raw Huber vs Auxiliary CE Clean Rerun v1","","ASAP validation only; ASAP test access 0; Repedal execution 0. Auxiliary logits were excluded from candidate inference.","",
        "| Objective | 4C Acc ↑ | Macro F1 ↑ | Trans P ↑ | Trans R ↑ | Trans F1 ↑ | JS ↓ | Intersection ↑ |","|---|---:|---:|---:|---:|---:|---:|---:|",*rows,"",
        "| Objective | LOW Recall | HALF Recall | Candidate Transitions |","|---|---:|---:|---:|",f"| Raw CC64 Huber | {rlow:.6f} | {rhalf:.6f} | {rt['candidate']} |",f"| Raw Huber + Aux CE | {hlow:.6f} | {hhalf:.6f} | {ht['candidate']} |","",
        "## Q1. 4-class Accuracy / Macro F1","",f"Confirmed: changes versus Raw Huber are {accuracy_gain:+.6f} Accuracy and {macro_gain:+.6f} Macro F1.","",
        "## Q2. LOW/HALF discrimination","",f"Confirmed: LOW recall {hlow-rlow:+.6f}; HALF recall {hhalf-rhalf:+.6f}. Confusion counts are reported in the validation report.","",
        "## Q3. Transition advantage","",f"Confirmed: Transition P/R/F1 changes are {ht['precision']-rt['precision']:+.6f}/{ht['recall']-rt['recall']:+.6f}/{transition_gain:+.6f}; candidate count change {ht['candidate']-rt['candidate']:+d}.","",
        "## Q4. Global pattern distribution","",f"Confirmed: JS change {js_change:+.6f}; Intersection change {intersection_change:+.6f}.","",
        "## Q5. AMP recovery","",f"Confirmed: {status.get('amp_overflow_events_total', 0)} AMP gradient-overflow event(s); maximum consecutive {status.get('max_consecutive_amp_overflows', 0)}. Event details: `{status.get('overflow_events', [])}`.","",
        "## Q6. Net value of auxiliary classification","",f"Confirmed metrics show the observed discrimination/transition/distribution trade-off above. Inference: practical benefit is supported only if categorical gains are not offset by unacceptable transition or distribution degradation; this single λ=0.1 validation run does not establish sensitivity or causality.","",
        "No λ/delta sweep, hybrid follow-up, temporal loss, architecture change, ASAP-test evaluation, or Repedal evaluation was started."])


def run(config: Mapping[str, Any]) -> int:
    output=Path(config["output_root"]); status=read_json(output/"run_status.json"); hybrid=read_json(output/"validation_eval/evaluation.json"); raw=read_json(config["raw_huber_evaluation"])
    with (output/"metrics.csv").open(newline="",encoding="utf-8") as handle: rows=list(csv.DictReader(handle))
    clean = training_report(config,status,rows)
    atomic_text(output/"RAW_HUBER_AUX_CE_TRAINING_REPORT.md", clean)
    atomic_text(output/"RAW_HUBER_AUX_CE_CLEAN_RERUN_REPORT.md", clean)
    atomic_text(output/"validation_eval/RAW_HUBER_AUX_CE_VALIDATION_REPORT.md", validation_report(hybrid))
    atomic_text(output/"RAW_HUBER_VS_AUX_CE_COMPARISON_REPORT.md", comparison_report(hybrid,raw,status))
    update_status(config, training_report_generated=True, validation_report_generated=True, comparison_report_generated=True); return 0


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",type=Path,required=True); parser.add_argument("--execute",action="store_true"); args=parser.parse_args()
    if not args.execute: raise SystemExit("report generation is guarded; pass --execute")
    return run(read_json(args.config))


if __name__ == "__main__": raise SystemExit(main())
