"""Raw-head metrics, conflict diagnostics, and timing decoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from .targets import MODE_NAMES


HOLD_ID, CHANGE_ID, RETURN_ID = range(3)


@dataclass(frozen=True)
class RawHeadPrediction:
    state_path: torch.Tensor
    mode_path: torch.Tensor
    timing_events: tuple[tuple[float, ...], ...]
    conflict_count: int
    conflict_rate: float


def state_mode_conflicts(state_path: torch.Tensor, mode_path: torch.Tensor) -> dict[str, int | float]:
    states = torch.as_tensor(state_path, dtype=torch.long)
    modes = torch.as_tensor(mode_path, dtype=torch.long, device=states.device)
    if states.ndim != 1 or modes.ndim != 1 or len(modes) != max(0, len(states) - 1):
        raise ValueError("State/Mode path lengths must be M and M-1")
    flips = states[:-1] != states[1:]
    missing_change = flips & (modes != CHANGE_ID)
    spurious_change = ~flips & (modes == CHANGE_ID)
    count = int((missing_change | spurious_change).sum().item())
    return {
        "count": count,
        "rate": float(count / len(modes)) if len(modes) else 0.0,
        "flip_but_not_change": int(missing_change.sum().item()),
        "same_but_change": int(spurious_change.sum().item()),
    }


def decode_timings(timing_predictions: torch.Tensor, mode_path: torch.Tensor | Sequence[int]) -> tuple[tuple[float, ...], ...]:
    """Clamp to [0,1]; sort RETURN pairs while retaining both head values."""

    values = torch.as_tensor(timing_predictions).detach().float()
    modes = torch.as_tensor(mode_path, dtype=torch.long, device=values.device)
    if values.ndim != 2 or values.shape != (len(modes), 2):
        raise ValueError("timing predictions must have shape [M-1,2]")
    values = values.clamp(0.0, 1.0)
    result: list[tuple[float, ...]] = []
    for pair, mode in zip(values, modes):
        mode_id = int(mode.item())
        if mode_id == HOLD_ID:
            result.append(())
        elif mode_id == CHANGE_ID:
            result.append((float(pair[0].item()),))
        elif mode_id == RETURN_ID:
            ordered = torch.sort(pair).values
            result.append((float(ordered[0].item()), float(ordered[1].item())))
        else:
            raise ValueError("Mode path contains an unknown class")
    return tuple(result)


def raw_head_decode(state_logits: torch.Tensor, mode_logits: torch.Tensor, timing_predictions: torch.Tensor) -> RawHeadPrediction:
    if state_logits.ndim != 2 or state_logits.shape[-1] != 2:
        raise ValueError("state logits must have shape [M,2]")
    if mode_logits.ndim != 2 or mode_logits.shape != (len(state_logits) - 1, 3):
        raise ValueError("mode logits must have shape [M-1,3]")
    state_path = state_logits.argmax(-1)
    mode_path = mode_logits.argmax(-1)
    conflict = state_mode_conflicts(state_path, mode_path)
    return RawHeadPrediction(state_path, mode_path, decode_timings(timing_predictions, mode_path), int(conflict["count"]), float(conflict["rate"]))


def classification_metrics(predictions: torch.Tensor, targets: torch.Tensor, *, class_names: Sequence[str], valid_mask: torch.Tensor | None = None) -> dict[str, Any]:
    predicted = torch.as_tensor(predictions, dtype=torch.long).reshape(-1)
    reference = torch.as_tensor(targets, dtype=torch.long, device=predicted.device).reshape(-1)
    if predicted.shape != reference.shape:
        raise ValueError("prediction/target shape mismatch")
    if valid_mask is not None:
        selected = torch.as_tensor(valid_mask, dtype=torch.bool, device=predicted.device).reshape(-1)
        if selected.shape != predicted.shape:
            raise ValueError("valid mask shape mismatch")
        predicted, reference = predicted[selected], reference[selected]
    classes = len(class_names)
    if not len(reference):
        raise ValueError("classification metrics require at least one target")
    if bool(torch.any((predicted < 0) | (predicted >= classes))) or bool(torch.any((reference < 0) | (reference >= classes))):
        raise ValueError("classification value outside class vocabulary")
    matrix = torch.bincount(reference * classes + predicted, minlength=classes * classes).reshape(classes, classes)
    rows = []
    for index, name in enumerate(class_names):
        true_positive = int(matrix[index, index].item())
        support = int(matrix[index].sum().item())
        predicted_count = int(matrix[:, index].sum().item())
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"class_id": index, "class_name": name, "precision": precision, "recall": recall, "f1": f1, "support": support, "predicted": predicted_count})
    return {
        "accuracy": float(matrix.diag().sum().item() / matrix.sum().item()),
        "macro_f1": float(sum(row["f1"] for row in rows) / classes),
        "class_metrics": rows,
        "confusion_matrix": matrix.cpu().tolist(),
    }


def raw_head_diagnostics(state_logits: torch.Tensor, mode_logits: torch.Tensor, state_targets: torch.Tensor, mode_targets: torch.Tensor) -> dict[str, Any]:
    prediction = raw_head_decode(state_logits, mode_logits, torch.zeros((len(mode_logits), 2), device=mode_logits.device))
    return {
        "state": classification_metrics(prediction.state_path, state_targets, class_names=("OFF", "ON")),
        "mode": classification_metrics(prediction.mode_path, mode_targets, class_names=MODE_NAMES),
        "conflict_count": prediction.conflict_count,
        "conflict_rate": prediction.conflict_rate,
    }
