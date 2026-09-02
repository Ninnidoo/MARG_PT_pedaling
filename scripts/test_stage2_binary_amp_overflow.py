#!/usr/bin/env python3
"""Focused CUDA test for GradScaler overflow skip/backoff behavior."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.stage2_binary.full_training import run_binary_epoch  # noqa: E402


class _FiniteForwardInfiniteBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, value: torch.Tensor) -> torch.Tensor:
        return value.detach().new_tensor(1.0)

    @staticmethod
    def backward(ctx: object, gradient: torch.Tensor) -> tuple[torch.Tensor]:
        return (torch.full_like(gradient, float("inf")),)


class _OverflowProbeModel(torch.nn.Module):
    target_name = "binary_targets"

    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Linear(1, 4)
        self.head = torch.nn.Linear(4, 8)

    def prediction_head_parameters(self):
        return self.head.parameters()

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        token_attention_mask: torch.Tensor,
        note_mask: torch.Tensor,
        binary_targets: torch.Tensor,
    ) -> SimpleNamespace:
        del token_attention_mask
        batch, notes = note_mask.shape
        values = input_ids.reshape(batch, notes, 8)[..., :1].float() / 128.0
        hidden = self.encoder(values)
        logits = self.head(hidden).reshape(batch, notes, 4, 2)
        finite_probe = F.cross_entropy(
            logits.reshape(-1, 2), binary_targets.reshape(-1)
        )
        anchor = sum(parameter.sum() for parameter in self.parameters())
        overflow_loss = _FiniteForwardInfiniteBackward.apply(anchor)
        loss = finite_probe.detach() * 0.0 + overflow_loss
        return SimpleNamespace(logits=logits, loss=loss)


def _batch() -> dict[str, torch.Tensor]:
    batch = 2
    notes = 4
    return {
        "input_ids": torch.arange(batch * notes * 8).reshape(batch, notes * 8),
        "token_attention_mask": torch.ones((batch, notes * 8), dtype=torch.long),
        "note_mask": torch.ones((batch, notes), dtype=torch.bool),
        "binary_targets": torch.zeros((batch, notes, 4), dtype=torch.long),
    }


def _parameters_equal(
    before: list[torch.Tensor], model: torch.nn.Module
) -> bool:
    return all(
        torch.equal(first, second.detach())
        for first, second in zip(before, model.parameters(), strict=True)
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("focused AMP overflow test requires CUDA")
    device = torch.device("cuda:0")
    torch.manual_seed(42)
    model = _OverflowProbeModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=1024.0)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    scale_before = float(scaler.get_scale())
    metrics, checks = run_binary_epoch(
        model,
        [_batch()],
        architecture="independent_4x2",
        device=device,
        amp_enabled=True,
        optimizer=optimizer,
        scaler=scaler,
        max_grad_norm=1.0,
        max_consecutive_amp_skips=8,
    )
    scale_after = float(scaler.get_scale())
    skip_path_passed = bool(
        metrics["loss"] == 1.0
        and checks["optimizer_steps"] == 0
        and checks["amp_skipped_steps"] == 1
        and checks["maximum_consecutive_amp_skips"] == 1
        and scale_after < scale_before
        and _parameters_equal(before, model)
    )
    if not skip_path_passed:
        raise AssertionError("GradScaler skip/backoff path did not behave as expected")

    torch.manual_seed(42)
    persistent_model = _OverflowProbeModel().to(device)
    persistent_optimizer = torch.optim.AdamW(persistent_model.parameters(), lr=1e-3)
    persistent_scaler = torch.amp.GradScaler(
        "cuda", enabled=True, init_scale=1024.0
    )
    persistent_abort = False
    try:
        run_binary_epoch(
            persistent_model,
            [_batch(), _batch()],
            architecture="independent_4x2",
            device=device,
            amp_enabled=True,
            optimizer=persistent_optimizer,
            scaler=persistent_scaler,
            max_grad_norm=1.0,
            max_consecutive_amp_skips=2,
        )
    except FloatingPointError as error:
        persistent_abort = "persistent AMP overflow" in str(error)
    if not persistent_abort:
        raise AssertionError("persistent overflow guard did not abort")

    result = {
        "completed": True,
        "skip_path_passed": skip_path_passed,
        "finite_unscaled_loss": True,
        "optimizer_step_skipped": True,
        "parameters_unchanged_on_skip": True,
        "scale_before": scale_before,
        "scale_after": scale_after,
        "scale_backoff": scale_after < scale_before,
        "persistent_overflow_abort_passed": persistent_abort,
        "seed": 42,
        "asap_test_midi_access_count": 0,
    }
    output = (
        PROJECT_ROOT
        / "analysis/stage2_binary_v0/train_strict_v1/amp_overflow_focused_test.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
