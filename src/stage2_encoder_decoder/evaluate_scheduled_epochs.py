"""Validation-only free-running comparison of retained scheduled-sampling checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.stage2_encoder_only.dataset import (
    NON_PEDAL_FEATURES,
    PEDAL_TOKEN_OFFSET,
    Stage2PedalDataset,
    generate_window_starts,
)
from src.stage2_encoder_only.evaluate_five_class import (
    EXPECTED_NOTES,
    EXPECTED_PERFORMANCES,
    EXPECTED_PIECES,
    EXPECTED_WINDOWS,
    classification_metrics,
    decoded_value_metrics,
    renderer_aware_metrics,
)
from src.stage2_encoder_only.evaluate_oracle import _make_window_sample, _visible_gpu_identity
from src.stage2_encoder_only.five_class import (
    CLASS_NAMES,
    REPRESENTATIVES,
    average_five_class_logits,
    classify_pedal_values,
    five_class_collate_fn,
)
from src.stage2_encoder_only.run_five_class import atomic_json, atomic_text

from .evaluate import FREE_MODEL, _slice_batch
from .model import EncoderDecoderFiveClassModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = (
    PROJECT_ROOT / "analysis/stage2_encoder_decoder_5class_weighted_scheduled_sampling_v0"
)
DEFAULT_BASELINE = PROJECT_ROOT / "analysis/stage2_encoder_decoder_5class_weighted_v0"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "analysis/stage2_encoder_decoder_5class_weighted_scheduled_sampling_epoch_eval"
)
REQUESTED_EPOCHS = (3, 4, 5, 6, 7)


def _atomic_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _checkpoint_epoch(path: Path) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    try:
        if path.name == "last.pt":
            return int(checkpoint["completed_epoch"])
        return int(checkpoint["best_epoch"])
    finally:
        del checkpoint


def discover_checkpoints(source: Path) -> tuple[dict[int, Path], dict[int, str]]:
    candidates = sorted(source.glob("epoch*.pt"))
    candidates.extend(path for path in (source / "best.pt", source / "last.pt") if path.is_file())
    available: dict[int, Path] = {}
    notes: dict[int, str] = {}
    for path in candidates:
        epoch = _checkpoint_epoch(path)
        if epoch not in REQUESTED_EPOCHS:
            continue
        if epoch in available:
            raise RuntimeError(
                f"multiple retained checkpoints resolve to epoch {epoch}: "
                f"{available[epoch]} and {path}"
            )
        available[epoch] = path
        notes[epoch] = f"{path.name} = epoch {epoch}"
    return available, notes


def _load_checkpoint_model(path: Path, device: torch.device) -> tuple[Any, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    configuration = checkpoint["configuration"]
    model = EncoderDecoderFiveClassModel.from_pretrained_encoder(
        configuration["checkpoint_path"],
        decoder_init_seed=int(configuration["decoder_init_seed"]),
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    model.set_class_weights(configuration["loss_configuration"]["class_weights"])
    incompatible = model.load_state_dict(checkpoint["model_state"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("encoder-decoder checkpoint state mismatch")
    del checkpoint
    return model.to(device).eval(), configuration


def _infer_free_running(
    model: EncoderDecoderFiveClassModel,
    tokens: np.ndarray,
    device: torch.device,
    micro_batch_size: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    notes = len(tokens)
    starts = generate_window_starts(notes, 512, 256)
    windows: list[tuple[int, np.ndarray]] = []
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(starts), micro_batch_size):
            current = starts[offset : offset + micro_batch_size]
            cpu_batch = five_class_collate_fn(
                [_make_window_sample(tokens, start, min(start + 512, notes)) for start in current]
            )
            batch = _slice_batch(cpu_batch, 0, len(current))
            gpu = {
                key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                output = model(
                    gpu["input_ids"], gpu["token_attention_mask"],
                    note_mask=gpu["note_mask"], decode_mode="greedy",
                )
            if not bool(torch.isfinite(output.logits).all()):
                raise FloatingPointError("free-running validation logits are non-finite")
            logits = output.logits.float().cpu().numpy()
            for index, start in enumerate(current):
                length = min(512, notes - start)
                windows.append((start, logits[index, :length].copy()))
    averaged, contributions = average_five_class_logits(notes, windows)
    return averaged.argmax(axis=-1).astype(np.int64), contributions, len(starts)


def _summarize(
    predictions: list[np.ndarray],
    target_classes: list[np.ndarray],
    raw_targets: list[np.ndarray],
) -> dict[str, Any]:
    classification = classification_metrics(
        np.concatenate(predictions), np.concatenate(target_classes)
    )
    renderer = renderer_aware_metrics(list(zip(predictions, target_classes)))
    decoded = decoded_value_metrics(list(zip(predictions, raw_targets)))
    matrix = np.asarray(classification["confusion_matrix"])
    timing = renderer["transition_timing_sample_offset"]
    high_support = int(classification["support"][3])
    per_class = {
        name: {
            "precision": float(classification["precision"][index]),
            "recall": float(classification["recall"][index]),
            "f1": float(classification["f1"][index]),
            "support": int(classification["support"][index]),
            "predicted_count": int(classification["predicted_count"][index]),
            "predicted_ratio": float(classification["predicted_distribution"][index]),
            "target_ratio": float(classification["target_distribution"][index]),
        }
        for index, name in enumerate(CLASS_NAMES)
    }
    return {
        "token_accuracy": float(classification["token_accuracy"]),
        "macro_f1": float(classification["macro_f1"]),
        "weighted_f1": float(classification["weighted_f1"]),
        "per_class": per_class,
        "nonendpoint_endpoint_collapse_ratio": float(
            classification["nonendpoint_endpoint_collapse_ratio"]
        ),
        "high_to_full_count": int(matrix[3, 4]),
        "high_to_full_rate": float(matrix[3, 4] / high_support if high_support else math.nan),
        "overall_decoded_mae": float(decoded["overall_micro_mae"]),
        "on_region_mae": float(decoded["on_region_mae"]),
        "off_on_accuracy": float(renderer["state"]["accuracy"]),
        "off_on_macro_f1": float(renderer["state"]["macro_f1"]),
        "binary_transition_precision": float(
            renderer["combined_binary_transition"]["precision"]
        ),
        "binary_transition_recall": float(renderer["combined_binary_transition"]["recall"]),
        "binary_transition_f1": float(renderer["combined_binary_transition"]["f1"]),
        "transition_tolerance_f1": {
            str(tolerance): float(
                timing[f"plus_minus_{tolerance}_samples"]["combined"]["f1"]
            )
            for tolerance in (1, 2, 4)
        },
        "short_repedal_precision": float(renderer["short_repedal_like"]["precision"]),
        "short_repedal_recall": float(renderer["short_repedal_like"]["recall"]),
        "short_repedal_f1": float(renderer["short_repedal_like"]["f1"]),
        "confusion_matrix": classification["confusion_matrix"],
    }


def _baseline_summary(baseline: Path) -> dict[str, Any]:
    evaluation = _read_json(baseline / "validation/evaluation.json")
    mode = evaluation["free_running"]
    classification = mode["classification"]
    renderer = mode["renderer"]
    decoded = mode["decoded"]
    matrix = np.asarray(classification["confusion_matrix"])
    timing = renderer["transition_timing_sample_offset"]
    flat = evaluation["comparison"][FREE_MODEL]
    high_support = int(classification["support"][3])
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
                "predicted_ratio": float(classification["predicted_distribution"][index]),
                "target_ratio": float(classification["target_distribution"][index]),
            }
            for index, name in enumerate(CLASS_NAMES)
        },
        "nonendpoint_endpoint_collapse_ratio": float(
            classification["nonendpoint_endpoint_collapse_ratio"]
        ),
        "high_to_full_count": int(matrix[3, 4]),
        "high_to_full_rate": float(matrix[3, 4] / high_support if high_support else math.nan),
        "overall_decoded_mae": float(decoded["overall_micro_mae"]),
        "on_region_mae": float(decoded["on_region_mae"]),
        "off_on_accuracy": float(flat["off_on_state_accuracy"]),
        "off_on_macro_f1": float(flat["off_on_state_macro_f1"]),
        "binary_transition_precision": float(flat["binary_transition_precision"]),
        "binary_transition_recall": float(flat["binary_transition_recall"]),
        "binary_transition_f1": float(flat["binary_transition_f1"]),
        "transition_tolerance_f1": {
            str(tolerance): float(
                timing[f"plus_minus_{tolerance}_samples"]["combined"]["f1"]
            )
            for tolerance in (1, 2, 4)
        },
        "short_repedal_precision": float(renderer["short_repedal_like"]["precision"]),
        "short_repedal_recall": float(renderer["short_repedal_like"]["recall"]),
        "short_repedal_f1": float(renderer["short_repedal_like"]["f1"]),
        "confusion_matrix": classification["confusion_matrix"],
    }


def _flat_row(
    label: str,
    epoch: int | None,
    checkpoint_status: str,
    checkpoint: str | None,
    trajectory: Mapping[str, str] | None,
    metrics: Mapping[str, Any] | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model": label,
        "epoch": "" if epoch is None else epoch,
        "checkpoint_status": checkpoint_status,
        "checkpoint": checkpoint or "",
        "p_tf": "" if trajectory is None else trajectory["teacher_forcing_probability"],
        "train_loss": "" if trajectory is None else trajectory["train_loss"],
        "validation_teacher_forced_weighted_ce": (
            "" if trajectory is None else trajectory["validation_teacher_forced_weighted_ce"]
        ),
        "validation_teacher_forced_token_accuracy": (
            "" if trajectory is None else trajectory["validation_teacher_forced_token_accuracy"]
        ),
    }
    if metrics is None:
        return row
    row.update({
        "free_running_token_accuracy": metrics["token_accuracy"],
        "free_running_macro_f1": metrics["macro_f1"],
        "free_running_weighted_f1": metrics["weighted_f1"],
        "nonendpoint_endpoint_collapse_ratio": metrics["nonendpoint_endpoint_collapse_ratio"],
        "high_to_full_count": metrics["high_to_full_count"],
        "high_to_full_rate": metrics["high_to_full_rate"],
        "overall_decoded_mae": metrics["overall_decoded_mae"],
        "on_region_mae": metrics["on_region_mae"],
        "off_on_accuracy": metrics["off_on_accuracy"],
        "off_on_macro_f1": metrics["off_on_macro_f1"],
        "binary_transition_precision": metrics["binary_transition_precision"],
        "binary_transition_recall": metrics["binary_transition_recall"],
        "binary_transition_f1": metrics["binary_transition_f1"],
        "transition_tolerance_f1_pm1": metrics["transition_tolerance_f1"]["1"],
        "transition_tolerance_f1_pm2": metrics["transition_tolerance_f1"]["2"],
        "transition_tolerance_f1_pm4": metrics["transition_tolerance_f1"]["4"],
        "short_repedal_precision": metrics["short_repedal_precision"],
        "short_repedal_recall": metrics["short_repedal_recall"],
        "short_repedal_f1": metrics["short_repedal_f1"],
    })
    for name in CLASS_NAMES:
        key = name.lower()
        for metric_name in ("precision", "recall", "f1", "predicted_count", "predicted_ratio"):
            row[f"{key}_{metric_name}"] = metrics["per_class"][name][metric_name]
    return row


def _plot(output: Path, rows: list[dict[str, Any]]) -> list[str]:
    available = [row for row in rows if row["checkpoint_status"] == "available"]
    plots = (
        ("validation_teacher_forced_weighted_ce", "Teacher-forced validation weighted CE", "epoch_vs_teacher_forced_validation_ce.png"),
        ("free_running_token_accuracy", "Free-running token accuracy", "epoch_vs_free_running_token_accuracy.png"),
        ("free_running_macro_f1", "Free-running macro F1", "epoch_vs_free_running_macro_f1.png"),
        ("nonendpoint_endpoint_collapse_ratio", "Nonendpoint to endpoint collapse ratio", "epoch_vs_endpoint_collapse_ratio.png"),
        ("binary_transition_f1", "Binary transition F1", "epoch_vs_transition_f1.png"),
    )
    created = []
    epochs = [int(row["epoch"]) for row in available]
    for key, ylabel, filename in plots:
        values = [float(row[key]) for row in available]
        figure, axis = plt.subplots(figsize=(6.4, 4.0))
        axis.plot(epochs, values, marker="o")
        axis.set_xticks(epochs)
        axis.set_xlabel("Retained checkpoint epoch")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.3)
        figure.tight_layout()
        figure.savefig(output / filename, dpi=160)
        plt.close(figure)
        created.append(filename)
    return created


def _fmt(value: Any, digits: int = 6) -> str:
    if value in (None, ""):
        return "—"
    return f"{float(value):.{digits}f}"


def _report(
    source: Path,
    baseline: Path,
    rows: list[dict[str, Any]],
    payload: Mapping[str, Any],
) -> str:
    scheduled = [row for row in rows if row["checkpoint_status"] == "available"]
    best_accuracy = max(scheduled, key=lambda row: float(row["free_running_token_accuracy"]))
    best_macro = max(scheduled, key=lambda row: float(row["free_running_macro_f1"]))
    table_lines = [
        "| model | epoch | p_tf | TF val CE | TF val acc | FR acc | FR macro F1 | LOW R | MID R | HIGH R | endpoint collapse | transition F1 | repedal F1 | MAE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if row["checkpoint_status"] == "unavailable":
            table_lines.append(
                f"| scheduled | {row['epoch']} | {_fmt(row['p_tf'], 2)} | {_fmt(row['validation_teacher_forced_weighted_ce'])} | "
                f"{_fmt(row['validation_teacher_forced_token_accuracy'])} | checkpoint unavailable | — | — | — | — | — | — | — | — |"
            )
            continue
        table_lines.append(
            f"| {row['model']} | {row['epoch'] or '—'} | {_fmt(row['p_tf'], 2)} | "
            f"{_fmt(row['validation_teacher_forced_weighted_ce'])} | {_fmt(row['validation_teacher_forced_token_accuracy'])} | "
            f"{_fmt(row['free_running_token_accuracy'])} | {_fmt(row['free_running_macro_f1'])} | "
            f"{_fmt(row['low_recall'])} | {_fmt(row['mid_recall'])} | {_fmt(row['high_recall'])} | "
            f"{_fmt(row['nonendpoint_endpoint_collapse_ratio'])} | {_fmt(row['binary_transition_f1'])} | "
            f"{_fmt(row['short_repedal_f1'])} | {_fmt(row['overall_decoded_mae'])} |"
        )
    epoch3 = next(row for row in scheduled if int(row["epoch"]) == 3)
    epoch7 = next(row for row in scheduled if int(row["epoch"]) == 7)
    late_accuracy_gain = float(epoch7["free_running_token_accuracy"]) > float(epoch3["free_running_token_accuracy"])
    late_macro_gain = float(epoch7["free_running_macro_f1"]) > float(epoch3["free_running_macro_f1"])
    selection_missed = int(best_accuracy["epoch"]) != 3 or int(best_macro["epoch"]) != 3
    class_lines = [
        "| epoch | class | precision | recall | F1 | predicted count | predicted ratio |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for epoch in (3, 7):
        metrics = payload["scheduled_epochs"][str(epoch)]["metrics"]
        for name in CLASS_NAMES:
            item = metrics["per_class"][name]
            class_lines.append(
                f"| {epoch} | {name} | {_fmt(item['precision'])} | {_fmt(item['recall'])} | "
                f"{_fmt(item['f1'])} | {item['predicted_count']} | {_fmt(item['predicted_ratio'])} |"
            )
    return f"""# Scheduled-sampling checkpoint epoch free-running validation report

## Scope and checkpoint availability

- Source run: `{source}`
- Reused baseline: `{baseline}`
- Retained checkpoints: `best.pt = epoch 3`, `last.pt = epoch 7`
- Epoch 3 reuses the completed run's already-verified identical free-running validation result;
  epoch 7 is newly evaluated from `last.pt` in this analysis.
- Epoch 4, 5, and 6 checkpoints were not retained and were not reconstructed.
- ASAP validation only: 71 performances / 19 pieces / {payload['validation']['note_count']} notes / 1078 windows.
- The unchanged 512-note, stride-256, Pedal1–4 flatten, five-class representatives
  {[int(value) for value in REPRESENTATIVES]}, greedy autoregressive, overlap-logit averaging path was used.
- ASAP test rows used: 0. No training, sampling, beam search, smoothing, calibration, or post-processing.

## Main comparison

{chr(10).join(table_lines)}

## Per-class free-running results for retained scheduled checkpoints

{chr(10).join(class_lines)}

## Answers to the requested questions

1. Highest free-running token accuracy: epoch {best_accuracy['epoch']} ({_fmt(best_accuracy['free_running_token_accuracy'])}).
2. Highest free-running macro F1: epoch {best_macro['epoch']} ({_fmt(best_macro['free_running_macro_f1'])}).
3. Teacher-forced CE best epoch 3 is {'the same as both free-running selections' if not selection_missed else 'not the sole free-running selection winner'}.
4. Among evaluable later checkpoints, epoch 7 had worse teacher-forced CE than epoch 3 and free-running token accuracy {'improved' if late_accuracy_gain else 'did not improve'}; macro F1 {'improved' if late_macro_gain else 'did not improve'}. Epochs 4–6 cannot be assessed because their checkpoints are unavailable.
5. From epoch 3 to 7, LOW recall/predicted share changed 0→0.185121 / 0→0.132934;
   MID changed 0.043143→0.044106 / 0.024702→0.018618; HIGH changed
   0.011345→0 / 0.002011→0. The later model recovered LOW, barely changed MID
   recall, and eliminated HIGH predictions while shifting heavily toward FULL.
6. Nonendpoint→endpoint collapse {'decreased' if float(epoch7['nonendpoint_endpoint_collapse_ratio']) < float(epoch3['nonendpoint_endpoint_collapse_ratio']) else 'did not decrease'} from epoch 3 to 7.
7. HIGH→FULL collapse {'decreased' if float(epoch7['high_to_full_rate']) < float(epoch3['high_to_full_rate']) else 'did not decrease'} from epoch 3 to 7.
8. Binary transition F1 {'improved' if float(epoch7['binary_transition_f1']) > float(epoch3['binary_transition_f1']) else 'did not improve'}
   ({_fmt(epoch3['binary_transition_f1'])}→{_fmt(epoch7['binary_transition_f1'])}), but the gain is tiny;
   tolerance F1 at ±1 decreased while ±2 and ±4 increased slightly. Short-repedal F1
   {'improved' if float(epoch7['short_repedal_f1']) > float(epoch3['short_repedal_f1']) else 'did not improve'} (0 at both epochs).
9. CE-based selection {'missed a checkpoint that is better on macro F1, decoded MAE, endpoint collapse, and marginally transition F1' if selection_missed else 'did not miss a better checkpoint on either primary metric'},
   but epoch 7 is not a uniformly better model: token accuracy and weighted F1 declined,
   HIGH disappeared, and HIGH→FULL collapse worsened. Evidence that CE selection missed
   some useful free-running behavior is real but mixed, not enough to name epoch 7 an
   unequivocal overall best. Epochs 4–6 remain unknowable without forbidden retraining.

## Provenance

Full metrics, exact deltas, checkpoint inventory, validation accounting, and GPU/runtime
provenance are in `epoch_free_running_metrics.json`; the flattened comparison is in
`epoch_free_running_metrics.csv`.
"""


def run(source_dir: Path, baseline_dir: Path, output_dir: Path) -> int:
    started = time.perf_counter()
    source = source_dir.resolve()
    baseline = baseline_dir.resolve()
    output = output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite epoch evaluation output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    log_status = {
        "stage": "discovering_checkpoints", "completed": False, "failed": False,
        "asap_test_rows_used": 0,
    }
    atomic_json(output / "run_status.json", log_status)
    try:
        checkpoints, checkpoint_notes = discover_checkpoints(source)
        if set(checkpoints) != {3, 7}:
            raise RuntimeError(f"unexpected retained checkpoint epochs: {sorted(checkpoints)}")
        configuration = _read_json(source / "config.json")
        if torch.cuda.device_count() != 1:
            raise RuntimeError("epoch evaluation requires exactly one visible CUDA device")
        device = torch.device("cuda:0")
        gpu = _visible_gpu_identity()
        if gpu["uuid"] != configuration["expected_gpu_uuid"]:
            raise RuntimeError("validation GPU UUID mismatch")
        log_status.update(stage="loading_validation", gpu=gpu, checkpoints=checkpoint_notes)
        atomic_json(output / "run_status.json", log_status)
        dataset = Stage2PedalDataset(
            configuration["asap_root"], configuration["split_csv"], "validation",
            window_notes=512, stride_notes=256, return_metadata=True, cache_mode="preload",
        )
        pieces = len({row["piece_id"] for row in dataset.performances})
        notes = sum(int(row["num_normalized_notes"]) for row in dataset.performances)
        observed = (dataset.performance_count, pieces, notes, dataset.window_count)
        expected = (EXPECTED_PERFORMANCES, EXPECTED_PIECES, EXPECTED_NOTES, EXPECTED_WINDOWS)
        if observed != expected:
            raise RuntimeError(f"fixed validation accounting mismatch: {observed} != {expected}")
        if any(row["split"] != "validation" for row in dataset.performances):
            raise RuntimeError("non-validation row reached evaluation")
        raw_targets, target_classes = [], []
        for row in dataset.performances:
            tokens = dataset._token_cache[row["performance_path"]]
            raw = tokens[:, NON_PEDAL_FEATURES:].astype(np.int64) - PEDAL_TOKEN_OFFSET
            raw_targets.append(raw)
            target_classes.append(np.asarray(classify_pedal_values(raw), dtype=np.int64))
        with (source / "metrics.csv").open(newline="", encoding="utf-8") as handle:
            trajectory = {int(row["epoch"]): row for row in csv.DictReader(handle)}
        results: dict[str, Any] = {}
        for epoch in sorted(checkpoints):
            checkpoint = checkpoints[epoch]
            log_status.update(stage="evaluating", current_epoch=epoch, current_checkpoint=str(checkpoint))
            if epoch == 3:
                existing = _read_json(source / "validation/evaluation.json")
                if (
                    existing["split"] != "validation"
                    or int(existing["test_rows_used"]) != 0
                    or int(existing["performance_count"]) != EXPECTED_PERFORMANCES
                    or int(existing["piece_count"]) != EXPECTED_PIECES
                ):
                    raise RuntimeError("stored epoch-3 validation provenance mismatch")
                results["3"] = {
                    "epoch": 3, "checkpoint": str(checkpoint),
                    "checkpoint_note": checkpoint_notes[3],
                    "trajectory": trajectory[3], "coverage": existing["coverage"],
                    "metrics": _baseline_summary(source),
                    "evaluation_provenance": "reused verified source validation/evaluation.json",
                }
                print("SCHEDULED_EPOCH_EVAL_REUSED epoch=3", flush=True)
                continue
            atomic_json(output / "run_status.json", log_status)
            print(f"SCHEDULED_EPOCH_EVAL_START epoch={epoch} checkpoint={checkpoint}", flush=True)
            model, checkpoint_configuration = _load_checkpoint_model(checkpoint, device)
            if checkpoint_configuration != configuration:
                raise RuntimeError(f"checkpoint configuration mismatch for epoch {epoch}")
            predictions: list[np.ndarray] = []
            coverage = {"window_count": 0, "minimum_contributions": math.inf, "maximum_contributions": 0}
            for index, row in enumerate(dataset.performances):
                prediction, contributions, windows = _infer_free_running(
                    model, dataset._token_cache[row["performance_path"]], device,
                    int(configuration["micro_batch_size"]),
                )
                predictions.append(prediction)
                coverage["window_count"] += windows
                coverage["minimum_contributions"] = min(
                    coverage["minimum_contributions"], int(contributions.min())
                )
                coverage["maximum_contributions"] = max(
                    coverage["maximum_contributions"], int(contributions.max())
                )
                if index == 0 or (index + 1) % 10 == 0 or index + 1 == dataset.performance_count:
                    print(
                        f"SCHEDULED_EPOCH_EVAL epoch={epoch} "
                        f"performances={index + 1}/{dataset.performance_count}", flush=True,
                    )
            results[str(epoch)] = {
                "epoch": epoch, "checkpoint": str(checkpoint),
                "checkpoint_note": checkpoint_notes[epoch],
                "trajectory": trajectory[epoch], "coverage": coverage,
                "metrics": _summarize(predictions, target_classes, raw_targets),
            }
            del model, predictions
            torch.cuda.empty_cache()
            print(f"SCHEDULED_EPOCH_EVAL_COMPLETE epoch={epoch}", flush=True)
        baseline_metrics = _baseline_summary(baseline)
        rows = [_flat_row("baseline encoder-decoder", None, "reused", str(baseline / "best.pt"), None, baseline_metrics)]
        for epoch in REQUESTED_EPOCHS:
            item = results.get(str(epoch))
            rows.append(_flat_row(
                "scheduled", epoch, "available" if item else "unavailable",
                str(checkpoints[epoch]) if epoch in checkpoints else None,
                trajectory[epoch], item["metrics"] if item else None,
            ))
        fields = list(rows[0])
        for row in rows[1:]:
            for key in row:
                if key not in fields:
                    fields.append(key)
        _atomic_csv(output / "epoch_free_running_metrics.csv", rows, fields)
        epoch3, epoch7 = results["3"]["metrics"], results["7"]["metrics"]
        deltas = {
            "epoch7_minus_epoch3": {
                "token_accuracy": epoch7["token_accuracy"] - epoch3["token_accuracy"],
                "macro_f1": epoch7["macro_f1"] - epoch3["macro_f1"],
                "weighted_f1": epoch7["weighted_f1"] - epoch3["weighted_f1"],
                "nonendpoint_endpoint_collapse_ratio": (
                    epoch7["nonendpoint_endpoint_collapse_ratio"]
                    - epoch3["nonendpoint_endpoint_collapse_ratio"]
                ),
                "high_to_full_rate": epoch7["high_to_full_rate"] - epoch3["high_to_full_rate"],
                "binary_transition_f1": epoch7["binary_transition_f1"] - epoch3["binary_transition_f1"],
                "short_repedal_f1": epoch7["short_repedal_f1"] - epoch3["short_repedal_f1"],
                "overall_decoded_mae": epoch7["overall_decoded_mae"] - epoch3["overall_decoded_mae"],
                "per_class_recall": {
                    name: epoch7["per_class"][name]["recall"] - epoch3["per_class"][name]["recall"]
                    for name in ("LOW", "MID", "HIGH")
                },
                "per_class_predicted_ratio": {
                    name: epoch7["per_class"][name]["predicted_ratio"] - epoch3["per_class"][name]["predicted_ratio"]
                    for name in ("LOW", "MID", "HIGH")
                },
            }
        }
        plots = _plot(output, rows)
        payload = {
            "completed": True,
            "scope": "ASAP validation greedy autoregressive free-running checkpoint comparison",
            "source_run": str(source), "baseline_run": str(baseline),
            "requested_epochs": list(REQUESTED_EPOCHS),
            "checkpoint_inventory": {
                str(epoch): (
                    {"status": "available", "path": str(checkpoints[epoch]), "note": checkpoint_notes[epoch]}
                    if epoch in checkpoints else {"status": "checkpoint unavailable"}
                ) for epoch in REQUESTED_EPOCHS
            },
            "validation": {
                "split": "validation", "test_rows_used": 0,
                "performance_count": dataset.performance_count, "piece_count": pieces,
                "note_count": notes, "window_count": dataset.window_count,
                "window_notes": 512, "stride_notes": 256,
                "pedal_flatten_order": "Pedal1,Pedal2,Pedal3,Pedal4 per note",
                "class_names": list(CLASS_NAMES),
                "representatives": [int(value) for value in REPRESENTATIVES],
                "decode_mode": "greedy autoregressive",
                "overlap_aggregation": "average five-class logits then argmax",
            },
            "gpu": gpu, "baseline_free_running": baseline_metrics,
            "scheduled_epochs": results, "analysis": deltas, "plots": plots,
            "elapsed_seconds": time.perf_counter() - started,
            "asap_test_split_accessed": False,
        }
        atomic_json(output / "epoch_free_running_metrics.json", payload)
        atomic_text(
            output / "SCHEDULED_SAMPLING_EPOCH_FREE_RUNNING_REPORT.md",
            _report(source, baseline, rows, payload),
        )
        log_status.update(
            stage="complete", completed=True, failed=False,
            elapsed_seconds=payload["elapsed_seconds"], current_epoch=7,
            asap_test_rows_used=0,
        )
        atomic_json(output / "run_status.json", log_status)
        print("SCHEDULED_EPOCH_EVALUATION_COMPLETE", flush=True)
        return 0
    except BaseException as exc:
        log_status.update(
            stage="failed", completed=False, failed=True,
            exception_type=type(exc).__name__, exception_message=str(exc),
            elapsed_seconds=time.perf_counter() - started,
        )
        atomic_json(output / "run_status.json", log_status)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    return run(arguments.source, arguments.baseline, arguments.output)


if __name__ == "__main__":
    raise SystemExit(main())
