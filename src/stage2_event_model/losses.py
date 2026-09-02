"""Masked macro-slot objective for Custom Event Model v0."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from src.stage2_event_tokenizer.tokenizer import EVENT_NAMES

from .dataset import EVENT_CLASSES, IGNORE_INDEX, MAIN_SLOTS, TERMINAL_SLOTS
from .model import CustomEventModelOutput


@dataclass(frozen=True)
class CustomEventLossConfig:
    beta_main: float = 0.1
    beta_terminal: float = 0.1
    lambda_init: float = 1.0
    lambda_main_event: float = 1.0
    lambda_main_time: float = 1.0
    lambda_terminal_event: float = 1.0
    lambda_terminal_time: float = 1.0

    def __post_init__(self) -> None:
        if self.beta_main <= 0 or self.beta_terminal <= 0:
            raise ValueError("Huber beta values must be positive")
        if any(value < 0 for key, value in asdict(self).items() if key.startswith("lambda_")):
            raise ValueError("loss coefficients must be non-negative")


@dataclass
class CustomEventLossOutput:
    total_loss: torch.Tensor
    initial_ce: torch.Tensor
    main_event_ce: torch.Tensor
    main_timing_huber: torch.Tensor
    terminal_event_ce: torch.Tensor
    terminal_timing_huber: torch.Tensor
    per_main_slot_event_loss: tuple[torch.Tensor, ...]
    per_main_slot_timing_loss: tuple[torch.Tensor, ...]
    per_terminal_slot_event_loss: tuple[torch.Tensor, ...]
    per_terminal_slot_timing_loss: tuple[torch.Tensor, ...]
    valid_target_counts: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_slot_event_weights(
    path: str | Path,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not payload.get("train_frequency_only") or not payload.get("diagnostic_only"):
        raise RuntimeError("event weights are not the frozen train-frequency candidates")

    def slots(scope: str, count: int) -> torch.Tensor:
        rows = []
        for index in range(count):
            values = payload[f"{scope}_slots"][f"Slot{index + 1}"]
            row = []
            for name in EVENT_NAMES:
                value = values[name]
                # A null weight denotes a zero-frequency train class. It cannot
                # contribute to the canonical training loss; store explicit 0.
                row.append(0.0 if value is None else float(value))
            present = [value for value in row if value > 0]
            if not present or abs(sum(present) / len(present) - 1.0) > 1e-6:
                raise RuntimeError(f"{scope} Slot{index + 1} weights are not mean-normalized")
            rows.append(row)
        return torch.tensor(rows, dtype=torch.float32)

    main = slots("main", MAIN_SLOTS)
    terminal = slots("terminal", TERMINAL_SLOTS)
    provenance = {
        "path": str(source.resolve()),
        "sha256": _sha256(source),
        "formula": payload["formula"],
        "train_frequency_only": True,
        "null_zero_frequency_weight_convention": "stored as 0; no canonical train target has that class",
    }
    return main, terminal, provenance


class CustomEventCriterion(nn.Module):
    def __init__(
        self,
        main_event_weights: torch.Tensor,
        terminal_event_weights: torch.Tensor,
        config: CustomEventLossConfig | None = None,
    ) -> None:
        super().__init__()
        if tuple(main_event_weights.shape) != (MAIN_SLOTS, EVENT_CLASSES):
            raise ValueError("main event weights must have shape [6,5]")
        if tuple(terminal_event_weights.shape) != (TERMINAL_SLOTS, EVENT_CLASSES):
            raise ValueError("terminal event weights must have shape [4,5]")
        self.register_buffer("main_event_weights", main_event_weights.float().clone())
        self.register_buffer(
            "terminal_event_weights", terminal_event_weights.float().clone()
        )
        self.config = config or CustomEventLossConfig()

    @staticmethod
    def _zero(reference: torch.Tensor) -> torch.Tensor:
        return reference.sum() * 0.0

    def _weighted_ce(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        if not bool(valid.any()):
            return self._zero(logits)
        selected_targets = targets[valid]
        if bool(((selected_targets < 0) | (selected_targets >= EVENT_CLASSES)).any()):
            raise ValueError("event target outside [0,4]")
        selected_weights = weights[selected_targets]
        denominator = selected_weights.sum()
        if float(denominator.detach().item()) <= 0:
            return self._zero(logits)
        numerator = F.cross_entropy(
            logits[valid],
            selected_targets,
            weight=weights,
            reduction="sum",
        )
        return numerator / denominator

    def _huber(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        valid: torch.Tensor,
        beta: float,
    ) -> torch.Tensor:
        if not bool(valid.any()):
            return self._zero(predictions)
        return F.smooth_l1_loss(
            predictions[valid],
            targets[valid],
            beta=beta,
            reduction="mean",
        )

    def forward(self, output: CustomEventModelOutput, batch: dict[str, torch.Tensor]) -> CustomEventLossOutput:
        initial_valid = batch["initial_valid_mask"].bool()
        if bool(initial_valid.any()):
            initial_ce = F.cross_entropy(
                output.initial_logits[initial_valid],
                batch["initial_targets"][initial_valid],
            )
        else:
            initial_ce = self._zero(output.initial_logits)

        owned = batch["owned_onset_mask"].bool()
        main_targets = batch["main_event_targets"]
        main_timing_mask = (
            batch["main_timing_valid_mask"].bool()
            & (main_targets != 0)
            & owned.unsqueeze(-1)
        )
        main_event_losses = []
        main_timing_losses = []
        main_event_counts = []
        main_timing_counts = []
        for slot in range(MAIN_SLOTS):
            event_valid = owned & (main_targets[:, :, slot] != IGNORE_INDEX)
            timing_valid = main_timing_mask[:, :, slot]
            main_event_losses.append(
                self._weighted_ce(
                    output.main_event_logits[:, :, slot],
                    main_targets[:, :, slot],
                    event_valid,
                    self.main_event_weights[slot],
                )
            )
            main_timing_losses.append(
                self._huber(
                    output.main_timing_predictions[:, :, slot],
                    batch["main_timing_targets"][:, :, slot],
                    timing_valid,
                    self.config.beta_main,
                )
            )
            main_event_counts.append(int(event_valid.sum().item()))
            main_timing_counts.append(int(timing_valid.sum().item()))
        main_event_ce = torch.stack(main_event_losses).mean()
        main_timing_huber = torch.stack(main_timing_losses).mean()

        terminal_valid = batch["terminal_valid_mask"].bool()
        terminal_targets = batch["terminal_event_targets"]
        terminal_timing_mask = (
            batch["terminal_timing_valid_mask"].bool()
            & (terminal_targets != 0)
            & terminal_valid.unsqueeze(-1)
        )
        terminal_event_losses = []
        terminal_timing_losses = []
        terminal_event_counts = []
        terminal_timing_counts = []
        for slot in range(TERMINAL_SLOTS):
            event_valid = terminal_valid
            timing_valid = terminal_timing_mask[:, slot]
            terminal_event_losses.append(
                self._weighted_ce(
                    output.terminal_event_logits[:, slot],
                    terminal_targets[:, slot],
                    event_valid,
                    self.terminal_event_weights[slot],
                )
            )
            terminal_timing_losses.append(
                self._huber(
                    output.terminal_timing_predictions[:, slot],
                    batch["terminal_timing_targets"][:, slot],
                    timing_valid,
                    self.config.beta_terminal,
                )
            )
            terminal_event_counts.append(int(event_valid.sum().item()))
            terminal_timing_counts.append(int(timing_valid.sum().item()))
        terminal_event_ce = torch.stack(terminal_event_losses).mean()
        terminal_timing_huber = torch.stack(terminal_timing_losses).mean()
        cfg = self.config
        total = (
            cfg.lambda_init * initial_ce
            + cfg.lambda_main_event * main_event_ce
            + cfg.lambda_main_time * main_timing_huber
            + cfg.lambda_terminal_event * terminal_event_ce
            + cfg.lambda_terminal_time * terminal_timing_huber
        )
        components = (
            total,
            initial_ce,
            main_event_ce,
            main_timing_huber,
            terminal_event_ce,
            terminal_timing_huber,
        )
        if not all(bool(torch.isfinite(value)) for value in components):
            raise FloatingPointError("Custom Event Model objective is non-finite")
        return CustomEventLossOutput(
            total_loss=total,
            initial_ce=initial_ce,
            main_event_ce=main_event_ce,
            main_timing_huber=main_timing_huber,
            terminal_event_ce=terminal_event_ce,
            terminal_timing_huber=terminal_timing_huber,
            per_main_slot_event_loss=tuple(main_event_losses),
            per_main_slot_timing_loss=tuple(main_timing_losses),
            per_terminal_slot_event_loss=tuple(terminal_event_losses),
            per_terminal_slot_timing_loss=tuple(terminal_timing_losses),
            valid_target_counts={
                "initial": int(initial_valid.sum().item()),
                "main_event_by_slot": main_event_counts,
                "main_timing_by_slot": main_timing_counts,
                "terminal_event_by_slot": terminal_event_counts,
                "terminal_timing_by_slot": terminal_timing_counts,
            },
        )
