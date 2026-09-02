"""Analyze and report the full 20-epoch teacher-forced learning trajectory."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.stage2_encoder_only.run_five_class import atomic_json, atomic_text


TRAJECTORY_FIELDS = (
    "epoch",
    "train_loss",
    "train_token_accuracy",
    "validation_teacher_forced_weighted_ce",
    "validation_teacher_forced_token_accuracy",
    "epoch_seconds",
    "encoder_learning_rate",
    "decoder_learning_rate",
    "is_best_so_far",
)


def _read_metrics(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"empty metrics file: {path}")
    return rows


def _write_trajectory(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(TRAJECTORY_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in TRAJECTORY_FIELDS})
    temporary.replace(path)


def _float(row: Mapping[str, Any], key: str) -> float:
    return float(row[key])


def analyze(
    output_dir: str | Path,
    baseline_dir: str | Path,
    training: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(output_dir).resolve()
    baseline_root = Path(baseline_dir).resolve()
    rows = _read_metrics(root / "metrics.csv")
    baseline_rows = _read_metrics(baseline_root / "metrics.csv")
    if [int(row["epoch"]) for row in rows] != list(range(1, 21)):
        raise RuntimeError("trajectory must contain exactly epochs 1 through 20")
    _write_trajectory(root / "trajectory.csv", rows)

    validation = [_float(row, "validation_teacher_forced_weighted_ce") for row in rows]
    train_loss = [_float(row, "train_loss") for row in rows]
    best_index = min(range(len(rows)), key=lambda index: validation[index])
    best_epoch = best_index + 1
    best_loss = validation[best_index]
    rise_steps = [
        {"from_epoch": index, "to_epoch": index + 1, "delta": validation[index] - validation[index - 1]}
        for index in range(max(best_epoch, 1), len(rows))
        if validation[index] > validation[index - 1]
    ]
    later_declines = [
        {"from_epoch": index, "to_epoch": index + 1, "delta": validation[index] - validation[index - 1]}
        for index in range(max(best_epoch, 1), len(rows))
        if validation[index] < validation[index - 1]
    ]
    overfitting_steps = [
        {
            "from_epoch": index,
            "to_epoch": index + 1,
            "train_loss_delta": train_loss[index] - train_loss[index - 1],
            "validation_loss_delta": validation[index] - validation[index - 1],
        }
        for index in range(1, len(rows))
        if train_loss[index] < train_loss[index - 1]
        and validation[index] > validation[index - 1]
    ]
    baseline_validation = [
        _float(row, "validation_teacher_forced_weighted_ce") for row in baseline_rows
    ]
    baseline_best_index = min(
        range(len(baseline_rows)), key=lambda index: baseline_validation[index]
    )
    shared = min(len(rows), len(baseline_rows))
    shared_epoch_deltas = [
        {
            "epoch": index + 1,
            "train_loss_delta_new_minus_baseline": train_loss[index]
            - _float(baseline_rows[index], "train_loss"),
            "validation_loss_delta_new_minus_baseline": validation[index]
            - baseline_validation[index],
        }
        for index in range(shared)
    ]
    baseline_stop_epoch = int(baseline_rows[-1]["epoch"])
    stopped_too_early = bool(
        best_epoch > baseline_stop_epoch
        and best_loss < baseline_validation[baseline_best_index]
    )
    checkpoint_matches_global_best = bool(
        int(training["best_epoch"]) == best_epoch
        and abs(float(training["best_validation_loss"]) - best_loss) <= 1e-12
    )
    analysis = {
        "epoch_count": len(rows),
        "best_validation_epoch": best_epoch,
        "best_validation_weighted_ce": best_loss,
        "epoch_20_validation_weighted_ce": validation[-1],
        "epoch_20_train_loss": train_loss[-1],
        "best_checkpoint_matches_global_minimum": checkpoint_matches_global_best,
        "post_best_validation_rise_steps": rise_steps,
        "post_best_validation_decline_steps": later_declines,
        "rose_then_declined_after_best": bool(rise_steps and later_declines),
        "overfitting_steps_train_down_validation_up": overfitting_steps,
        "overfitting_region_present": bool(overfitting_steps),
        "baseline_early_stopped_epoch": baseline_stop_epoch,
        "baseline_best_epoch": baseline_best_index + 1,
        "baseline_best_validation_weighted_ce": baseline_validation[baseline_best_index],
        "shared_epoch_deltas": shared_epoch_deltas,
        "early_stopping_was_too_early_for_global_best": stopped_too_early,
        "decision_rule": (
            "true only if the 20-epoch run reaches a strictly lower validation minimum "
            "after the baseline early-stop epoch"
        ),
        "asap_test_split_accessed": False,
    }
    atomic_json(root / "trajectory_analysis.json", analysis)

    plot_status: dict[str, Any]
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = list(range(1, 21))
        figure, axis = plt.subplots(figsize=(9, 5.5))
        axis.plot(epochs, train_loss, marker="o", label="train weighted CE")
        axis.plot(epochs, validation, marker="o", label="validation teacher-forced weighted CE")
        axis.axvline(best_epoch, color="tab:green", linestyle="--", alpha=0.7, label=f"best epoch {best_epoch}")
        axis.axvline(baseline_stop_epoch, color="tab:red", linestyle=":", alpha=0.7, label=f"baseline stop epoch {baseline_stop_epoch}")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        axis.set_title("Stage 2 encoder-decoder weighted 5-class trajectory")
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(root / "train_validation_loss_curve.png", dpi=160)
        plt.close(figure)
        plot_status = {"created": True, "path": str(root / "train_validation_loss_curve.png")}
    except (ImportError, RuntimeError, OSError, ValueError) as exc:
        plot_status = {"created": False, "reason": f"{type(exc).__name__}: {exc}"}
    analysis["plot"] = plot_status
    atomic_json(root / "trajectory_analysis.json", analysis)

    report = f"""# Encoder-Decoder Weighted 5-class: No Early Stop 20-Epoch Trajectory

## Scope

기존 completed encoder-decoder weighted 5-class run과 동일한 초기화 및 training
configuration으로 처음부터 다시 학습했다. Early stopping 종료 동작만 비활성화했으며,
best checkpoint selection은 validation teacher-forced weighted CE 기준으로 유지했다.
ASAP test와 free-running full validation은 사용하지 않았다.

## Summary

- Completed epochs: `20`
- Best validation epoch/loss: `{best_epoch}` / `{best_loss}`
- Epoch 20 validation loss: `{validation[-1]}`
- Epoch 20 train loss: `{train_loss[-1]}`
- Best checkpoint matches global minimum: `{checkpoint_matches_global_best}`
- Validation rose and later declined after best: `{analysis['rose_then_declined_after_best']}`
- Train-down/validation-up overfitting region present: `{analysis['overfitting_region_present']}`
- Baseline stopped at epoch: `{baseline_stop_epoch}`
- Baseline early stopping was too early for the observed global best: `{stopped_too_early}`

## Post-best rises

```json
{json.dumps(rise_steps, indent=2, sort_keys=True)}
```

## Post-best later declines

```json
{json.dumps(later_declines, indent=2, sort_keys=True)}
```

## Overfitting intervals

The following intervals have decreasing train loss and increasing validation loss.

```json
{json.dumps(overfitting_steps, indent=2, sort_keys=True)}
```

## Existing early-stopped trajectory comparison

```json
{json.dumps(shared_epoch_deltas, indent=2, sort_keys=True)}
```

## Artifacts

- `trajectory.csv`: epochs 1–20
- `trajectory_analysis.json`: machine-readable analysis
- `train_validation_loss_curve.png`: `{plot_status}`
- `best.pt`: lowest validation weighted CE checkpoint
- `last.pt`: epoch 20 checkpoint
"""
    atomic_text(root / "NO_EARLY_STOP_20EP_REPORT.md", report)
    return analysis


__all__ = ["TRAJECTORY_FIELDS", "analyze"]
