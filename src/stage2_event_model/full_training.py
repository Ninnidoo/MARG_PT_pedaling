"""Exact effective-batch objective and diagnostics for canonical full training."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .dataset import EVENT_CLASSES, IGNORE_INDEX, MAIN_SLOTS, TERMINAL_SLOTS
from .losses import CustomEventCriterion, CustomEventLossConfig
from .model import CustomEventModelOutput


@dataclass
class LossDenominators:
    initial: int
    main_event: list[float]
    main_timing: list[int]
    terminal_event: list[float]
    terminal_timing: list[int]

    @classmethod
    def zero(cls) -> "LossDenominators":
        return cls(0, [0.0] * MAIN_SLOTS, [0] * MAIN_SLOTS, [0.0] * TERMINAL_SLOTS, [0] * TERMINAL_SLOTS)

    def add_(self, other: "LossDenominators") -> None:
        self.initial += other.initial
        for destination, source in (
            (self.main_event, other.main_event),
            (self.main_timing, other.main_timing),
            (self.terminal_event, other.terminal_event),
            (self.terminal_timing, other.terminal_timing),
        ):
            for index, value in enumerate(source):
                destination[index] += value


@dataclass
class RawLossTerms:
    initial: torch.Tensor
    main_event: list[torch.Tensor]
    main_timing: list[torch.Tensor]
    terminal_event: list[torch.Tensor]
    terminal_timing: list[torch.Tensor]


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def batch_denominators(
    batch: Mapping[str, torch.Tensor],
    main_weights: torch.Tensor,
    terminal_weights: torch.Tensor,
) -> LossDenominators:
    result = LossDenominators.zero()
    result.initial = int(batch["initial_valid_mask"].sum().item())
    owned = batch["owned_onset_mask"].bool()
    main_targets = batch["main_event_targets"]
    for slot in range(MAIN_SLOTS):
        event_valid = owned & (main_targets[:, :, slot] != IGNORE_INDEX)
        selected = main_targets[:, :, slot][event_valid]
        result.main_event[slot] = float(main_weights[slot][selected].sum().item())
        timing_valid = (
            batch["main_timing_valid_mask"][:, :, slot].bool()
            & (main_targets[:, :, slot] != 0)
            & owned
        )
        result.main_timing[slot] = int(timing_valid.sum().item())
    terminal_valid = batch["terminal_valid_mask"].bool()
    terminal_targets = batch["terminal_event_targets"]
    for slot in range(TERMINAL_SLOTS):
        selected = terminal_targets[:, slot][terminal_valid]
        result.terminal_event[slot] = float(terminal_weights[slot][selected].sum().item())
        timing_valid = (
            batch["terminal_timing_valid_mask"][:, slot].bool()
            & (terminal_targets[:, slot] != 0)
            & terminal_valid
        )
        result.terminal_timing[slot] = int(timing_valid.sum().item())
    return result


def group_denominators(
    batches: Sequence[Mapping[str, torch.Tensor]],
    main_weights: torch.Tensor,
    terminal_weights: torch.Tensor,
) -> LossDenominators:
    result = LossDenominators.zero()
    for batch in batches:
        result.add_(batch_denominators(batch, main_weights, terminal_weights))
    return result


def _zero(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def raw_loss_terms(
    output: CustomEventModelOutput,
    batch: Mapping[str, torch.Tensor],
    criterion: CustomEventCriterion,
) -> RawLossTerms:
    initial_valid = batch["initial_valid_mask"].bool()
    initial = (
        F.cross_entropy(
            output.initial_logits[initial_valid],
            batch["initial_targets"][initial_valid],
            reduction="sum",
        )
        if bool(initial_valid.any())
        else _zero(output.initial_logits)
    )
    owned = batch["owned_onset_mask"].bool()
    main_targets = batch["main_event_targets"]
    main_event: list[torch.Tensor] = []
    main_timing: list[torch.Tensor] = []
    for slot in range(MAIN_SLOTS):
        event_valid = owned & (main_targets[:, :, slot] != IGNORE_INDEX)
        if bool(event_valid.any()):
            main_event.append(
                F.cross_entropy(
                    output.main_event_logits[:, :, slot][event_valid],
                    main_targets[:, :, slot][event_valid],
                    weight=criterion.main_event_weights[slot],
                    reduction="sum",
                )
            )
        else:
            main_event.append(_zero(output.main_event_logits[:, :, slot]))
        timing_valid = (
            batch["main_timing_valid_mask"][:, :, slot].bool()
            & (main_targets[:, :, slot] != 0)
            & owned
        )
        if bool(timing_valid.any()):
            main_timing.append(
                F.smooth_l1_loss(
                    output.main_timing_predictions[:, :, slot][timing_valid],
                    batch["main_timing_targets"][:, :, slot][timing_valid],
                    beta=criterion.config.beta_main,
                    reduction="sum",
                )
            )
        else:
            main_timing.append(_zero(output.main_timing_predictions[:, :, slot]))
    terminal_valid = batch["terminal_valid_mask"].bool()
    terminal_targets = batch["terminal_event_targets"]
    terminal_event: list[torch.Tensor] = []
    terminal_timing: list[torch.Tensor] = []
    for slot in range(TERMINAL_SLOTS):
        if bool(terminal_valid.any()):
            terminal_event.append(
                F.cross_entropy(
                    output.terminal_event_logits[:, slot][terminal_valid],
                    terminal_targets[:, slot][terminal_valid],
                    weight=criterion.terminal_event_weights[slot],
                    reduction="sum",
                )
            )
        else:
            terminal_event.append(_zero(output.terminal_event_logits[:, slot]))
        timing_valid = (
            batch["terminal_timing_valid_mask"][:, slot].bool()
            & (terminal_targets[:, slot] != 0)
            & terminal_valid
        )
        if bool(timing_valid.any()):
            terminal_timing.append(
                F.smooth_l1_loss(
                    output.terminal_timing_predictions[:, slot][timing_valid],
                    batch["terminal_timing_targets"][:, slot][timing_valid],
                    beta=criterion.config.beta_terminal,
                    reduction="sum",
                )
            )
        else:
            terminal_timing.append(_zero(output.terminal_timing_predictions[:, slot]))
    return RawLossTerms(initial, main_event, main_timing, terminal_event, terminal_timing)


def normalized_components(
    terms: RawLossTerms,
    denominators: LossDenominators,
    config: CustomEventLossConfig,
) -> dict[str, torch.Tensor]:
    initial = terms.initial / denominators.initial if denominators.initial else _zero(terms.initial)

    def macro(values: Sequence[torch.Tensor], counts: Sequence[float]) -> torch.Tensor:
        normalized = [
            value / float(count) if count else _zero(value)
            for value, count in zip(values, counts)
        ]
        return torch.stack(normalized).mean()

    main_event = macro(terms.main_event, denominators.main_event)
    main_timing = macro(terms.main_timing, denominators.main_timing)
    terminal_event = macro(terms.terminal_event, denominators.terminal_event)
    terminal_timing = macro(terms.terminal_timing, denominators.terminal_timing)
    total = (
        config.lambda_init * initial
        + config.lambda_main_event * main_event
        + config.lambda_main_time * main_timing
        + config.lambda_terminal_event * terminal_event
        + config.lambda_terminal_time * terminal_timing
    )
    return {
        "total": total,
        "initial": initial,
        "main_event": main_event,
        "main_timing": main_timing,
        "terminal_event": terminal_event,
        "terminal_timing": terminal_timing,
    }


def add_raw_terms(destination: dict[str, Any], terms: RawLossTerms) -> None:
    destination["initial"] += float(terms.initial.detach().float())
    for key, values in (
        ("main_event", terms.main_event),
        ("main_timing", terms.main_timing),
        ("terminal_event", terms.terminal_event),
        ("terminal_timing", terms.terminal_timing),
    ):
        for index, value in enumerate(values):
            destination[key][index] += float(value.detach().float())


def empty_raw_sums() -> dict[str, Any]:
    return {
        "initial": 0.0,
        "main_event": [0.0] * MAIN_SLOTS,
        "main_timing": [0.0] * MAIN_SLOTS,
        "terminal_event": [0.0] * TERMINAL_SLOTS,
        "terminal_timing": [0.0] * TERMINAL_SLOTS,
    }


def components_from_sums(
    sums: Mapping[str, Any], denominators: LossDenominators, config: CustomEventLossConfig
) -> dict[str, Any]:
    def macro(key: str, counts: Sequence[float]) -> tuple[float, list[float]]:
        slots = [
            float(value / count) if count else 0.0
            for value, count in zip(sums[key], counts)
        ]
        return float(sum(slots) / len(slots)), slots

    initial = float(sums["initial"] / denominators.initial) if denominators.initial else 0.0
    main_event, main_event_slots = macro("main_event", denominators.main_event)
    main_timing, main_timing_slots = macro("main_timing", denominators.main_timing)
    terminal_event, terminal_event_slots = macro("terminal_event", denominators.terminal_event)
    terminal_timing, terminal_timing_slots = macro("terminal_timing", denominators.terminal_timing)
    total = (
        config.lambda_init * initial
        + config.lambda_main_event * main_event
        + config.lambda_main_time * main_timing
        + config.lambda_terminal_event * terminal_event
        + config.lambda_terminal_time * terminal_timing
    )
    result = {
        "total": total,
        "initial": initial,
        "main_event": main_event,
        "main_timing": main_timing,
        "terminal_event": terminal_event,
        "terminal_timing": terminal_timing,
        "main_event_by_slot": main_event_slots,
        "main_timing_by_slot": main_timing_slots,
        "terminal_event_by_slot": terminal_event_slots,
        "terminal_timing_by_slot": terminal_timing_slots,
    }
    if not all(math.isfinite(value) for value in result.values() if isinstance(value, float)):
        raise FloatingPointError("non-finite epoch loss component")
    return result


class ValidationDiagnostics:
    def __init__(self) -> None:
        self.initial_confusion = np.zeros((4, 4), dtype=np.int64)
        self.main_confusions = [np.zeros((EVENT_CLASSES, EVENT_CLASSES), dtype=np.int64) for _ in range(MAIN_SLOTS)]
        self.terminal_confusions = [np.zeros((EVENT_CLASSES, EVENT_CLASSES), dtype=np.int64) for _ in range(TERMINAL_SLOTS)]
        self.main_abs = [0.0] * MAIN_SLOTS
        self.main_sq = [0.0] * MAIN_SLOTS
        self.main_abs_clipped = [0.0] * MAIN_SLOTS
        self.main_count = [0] * MAIN_SLOTS
        self.terminal_abs = [0.0] * TERMINAL_SLOTS
        self.terminal_sq = [0.0] * TERMINAL_SLOTS
        self.terminal_seconds_abs = [0.0] * TERMINAL_SLOTS
        self.terminal_count = [0] * TERMINAL_SLOTS

    @staticmethod
    def _add_confusion(confusion: np.ndarray, target: torch.Tensor, prediction: torch.Tensor, classes: int) -> None:
        encoded = (target.long() * classes + prediction.long()).detach().cpu().numpy()
        confusion += np.bincount(encoded, minlength=classes * classes).reshape(classes, classes)

    def update(self, output: CustomEventModelOutput, batch: Mapping[str, torch.Tensor]) -> None:
        initial_valid = batch["initial_valid_mask"].bool()
        if bool(initial_valid.any()):
            self._add_confusion(
                self.initial_confusion,
                batch["initial_targets"][initial_valid],
                output.initial_logits.argmax(-1)[initial_valid],
                4,
            )
        owned = batch["owned_onset_mask"].bool()
        main_targets = batch["main_event_targets"]
        for slot in range(MAIN_SLOTS):
            event_valid = owned & (main_targets[:, :, slot] != IGNORE_INDEX)
            self._add_confusion(
                self.main_confusions[slot],
                main_targets[:, :, slot][event_valid],
                output.main_event_logits[:, :, slot].argmax(-1)[event_valid],
                EVENT_CLASSES,
            )
            timing_valid = batch["main_timing_valid_mask"][:, :, slot].bool() & (main_targets[:, :, slot] != 0) & owned
            if bool(timing_valid.any()):
                prediction = output.main_timing_predictions[:, :, slot][timing_valid].detach().float()
                target = batch["main_timing_targets"][:, :, slot][timing_valid].detach().float()
                difference = prediction - target
                self.main_abs[slot] += float(difference.abs().sum())
                self.main_sq[slot] += float(difference.square().sum())
                self.main_abs_clipped[slot] += float((prediction.clamp(0, 1) - target).abs().sum())
                self.main_count[slot] += int(target.numel())
        terminal_valid = batch["terminal_valid_mask"].bool()
        terminal_targets = batch["terminal_event_targets"]
        for slot in range(TERMINAL_SLOTS):
            if bool(terminal_valid.any()):
                self._add_confusion(
                    self.terminal_confusions[slot],
                    terminal_targets[:, slot][terminal_valid],
                    output.terminal_event_logits[:, slot].argmax(-1)[terminal_valid],
                    EVENT_CLASSES,
                )
            timing_valid = batch["terminal_timing_valid_mask"][:, slot].bool() & (terminal_targets[:, slot] != 0) & terminal_valid
            if bool(timing_valid.any()):
                prediction = output.terminal_timing_predictions[:, slot][timing_valid].detach().float()
                target = batch["terminal_timing_targets"][:, slot][timing_valid].detach().float()
                difference = prediction - target
                self.terminal_abs[slot] += float(difference.abs().sum())
                self.terminal_sq[slot] += float(difference.square().sum())
                seconds_prediction = torch.expm1(prediction.clamp(min=0))
                seconds_target = torch.expm1(target)
                self.terminal_seconds_abs[slot] += float((seconds_prediction - seconds_target).abs().sum())
                self.terminal_count[slot] += int(target.numel())

    @staticmethod
    def _event_metrics(confusion: np.ndarray) -> dict[str, Any]:
        total = int(confusion.sum())
        correct = int(np.trace(confusion))
        target_non_none = int(confusion[1:, :].sum())
        predicted_non_none = int(confusion[:, 1:].sum())
        binary_tp = int(confusion[1:, 1:].sum())
        precision = binary_tp / predicted_non_none if predicted_non_none else 0.0
        recall = binary_tp / target_non_none if target_non_none else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        class_recall = [
            float(confusion[index, index] / confusion[index].sum()) if confusion[index].sum() else 0.0
            for index in range(confusion.shape[0])
        ]
        return {
            "target_count": total,
            "correct_count": correct,
            "accuracy": correct / total if total else 0.0,
            "non_none_precision": precision,
            "non_none_recall": recall,
            "non_none_f1": f1,
            "target_non_none_count": target_non_none,
            "predicted_non_none_count": predicted_non_none,
            "predicted_target_non_none_ratio": predicted_non_none / target_non_none if target_non_none else 0.0,
            "class_recall": class_recall,
            "confusion": confusion.tolist(),
        }

    def result(self) -> dict[str, Any]:
        main_pooled = sum(self.main_confusions, np.zeros((EVENT_CLASSES, EVENT_CLASSES), dtype=np.int64))
        terminal_pooled = sum(self.terminal_confusions, np.zeros((EVENT_CLASSES, EVENT_CLASSES), dtype=np.int64))

        def timing(abs_values, sq_values, counts, clipped=None, seconds=None):
            total_count = sum(counts)
            result = {
                "valid_count": total_count,
                "mae": sum(abs_values) / total_count if total_count else 0.0,
                "rmse": math.sqrt(sum(sq_values) / total_count) if total_count else 0.0,
                "per_slot_valid_count": counts,
                "per_slot_mae": [value / count if count else 0.0 for value, count in zip(abs_values, counts)],
            }
            if clipped is not None:
                result["clipped_mae"] = sum(clipped) / total_count if total_count else 0.0
                result["per_slot_clipped_mae"] = [value / count if count else 0.0 for value, count in zip(clipped, counts)]
            if seconds is not None:
                result["decoded_seconds_gap_mae"] = sum(seconds) / total_count if total_count else 0.0
                result["per_slot_decoded_seconds_gap_mae"] = [value / count if count else 0.0 for value, count in zip(seconds, counts)]
            return result

        return {
            "initial": self._event_metrics(self.initial_confusion),
            "main_event": self._event_metrics(main_pooled),
            "main_event_by_slot": [self._event_metrics(value) for value in self.main_confusions],
            "terminal_event": self._event_metrics(terminal_pooled),
            "terminal_event_by_slot": [self._event_metrics(value) for value in self.terminal_confusions],
            "main_timing": timing(self.main_abs, self.main_sq, self.main_count, clipped=self.main_abs_clipped),
            "terminal_timing": timing(self.terminal_abs, self.terminal_sq, self.terminal_count, seconds=self.terminal_seconds_abs),
        }
