"""Targeted AMP recovery and Decoder-only checkpoint-resume tests."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from torch.nn import functional as F

from scripts.run_stage2_4class_architecture_v0 import read_json, run_epoch
from scripts.run_stage2_decoder_only_v0 import (
    restore_training_checkpoint,
    update_status,
)
from src.stage2_decoder_only.model import FourClassPedalDecoderOnlyModel
from src.stage2_decoder_only.training import build_decoder_only_optimizer


BEST = ROOT / "analysis/stage2_decoder_only_4class_v0/full_train/best.pt"
CONFIG = ROOT / "configs/stage2_4class_architecture_v0.json"


class TinyFourClassModel(nn.Module):
    def __init__(self, *, nonfinite_loss: bool = False) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.25))
        self.nonfinite_loss = nonfinite_loss

    def forward(self, **batch: torch.Tensor) -> SimpleNamespace:
        targets = batch["pedal_targets"]
        leading = self.weight.expand(*targets.shape)
        zeros = torch.zeros_like(leading)
        logits = torch.stack((leading, zeros, zeros, zeros), dim=-1)
        loss_numerator = F.cross_entropy(
            logits.reshape(-1, 4), targets.reshape(-1), reduction="sum"
        )
        loss = loss_numerator / targets.numel()
        if self.nonfinite_loss:
            loss = loss * torch.tensor(float("nan"))
            loss_numerator = loss_numerator * torch.tensor(float("nan"))
        return SimpleNamespace(
            logits=logits,
            loss=loss,
            loss_numerator=loss_numerator,
        )


class SyntheticGradScaler:
    """Public GradScaler-shaped test double with controlled overflow attempts."""

    def __init__(self, overflow_attempts: set[int]) -> None:
        self.overflow_attempts = set(overflow_attempts)
        self.attempt = 0
        self._scale = 1024.0
        self.step_calls = 0
        self.update_calls = 0
        self._found_nonfinite = False

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        self.attempt += 1
        self._found_nonfinite = self.attempt in self.overflow_attempts
        if self._found_nonfinite:
            parameter = optimizer.param_groups[0]["params"][0]
            parameter.grad.fill_(float("inf"))

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        self.step_calls += 1
        if not self._found_nonfinite:
            optimizer.step()

    def update(self) -> None:
        self.update_calls += 1
        if self._found_nonfinite:
            self._scale *= 0.5

    def get_scale(self) -> float:
        return self._scale


def make_loader(steps: int) -> list[dict[str, torch.Tensor]]:
    batch = {
        "input_ids": torch.ones((1, 8), dtype=torch.long),
        "token_attention_mask": torch.ones((1, 8), dtype=torch.long),
        "note_mask": torch.ones((1, 1), dtype=torch.bool),
        "pedal_targets": torch.zeros((1, 1, 4), dtype=torch.long),
    }
    return [{key: value.clone() for key, value in batch.items()} for _ in range(steps)]


def make_optimizer(model: nn.Module) -> torch.optim.SGD:
    return torch.optim.SGD(
        [{"params": list(model.parameters()), "lr": 0.1, "group_name": "fresh_prefix_lm"}]
    )


def test_a_normal_finite_step_is_unchanged() -> None:
    model = TinyFourClassModel()
    optimizer = make_optimizer(model)
    scaler = SyntheticGradScaler(set())
    before = model.weight.detach().clone()
    _, checks = run_epoch(
        model,
        make_loader(1),
        device=torch.device("cpu"),
        amp_enabled=True,
        optimizer=optimizer,
        scaler=scaler,
        max_consecutive_amp_overflows=3,
    )
    assert checks["optimizer_steps"] == 1
    assert checks["optimizer_attempts"] == 1
    assert checks["amp_overflow_events"] == 0
    assert checks["consecutive_amp_overflows"] == 0
    assert scaler.step_calls == scaler.update_calls == 1
    assert not torch.equal(before, model.weight.detach())


def test_b_isolated_overflow_recovers_logs_status_and_resets() -> None:
    temporary = tempfile.TemporaryDirectory()
    tmp_path = Path(temporary.name)
    model = TinyFourClassModel()
    optimizer = make_optimizer(model)
    scaler = SyntheticGradScaler({1})
    events: list[dict[str, object]] = []
    log_path = tmp_path / "train.log"

    def callback(step: int, details: dict[str, object]) -> None:
        if details["amp_overflow_event"]:
            event = {"step": step, **details}
            events.append(event)
            update_status(
                tmp_path,
                status="running",
                amp_overflow_events_total=len(events),
                last_overflow=event,
            )
            log_path.write_text(json.dumps(event, default=str) + "\n")

    _, checks = run_epoch(
        model,
        make_loader(2),
        device=torch.device("cpu"),
        amp_enabled=True,
        optimizer=optimizer,
        scaler=scaler,
        max_consecutive_amp_overflows=3,
        step_callback=callback,
    )
    assert scaler.step_calls == scaler.update_calls == 2
    assert checks["optimizer_steps"] == 1
    assert checks["amp_overflow_events"] == 1
    assert checks["amp_skipped_steps"] == 1
    assert checks["consecutive_amp_overflows"] == 0
    assert checks["max_consecutive_amp_overflows"] == 1
    assert events[0]["offending_parameter_count"] == 1
    assert events[0]["offending_parameter_groups"] == ["fresh_prefix_lm"]
    assert events[0]["scale_before"] == 1024.0
    assert events[0]["scale_after"] == 512.0
    status = read_json(tmp_path / "run_status.json")
    assert status["amp_overflow_events_total"] == 1
    assert status["last_overflow"]["step"] == 1
    assert "amp_overflow_event" in log_path.read_text()
    temporary.cleanup()


def test_c_three_consecutive_overflows_fail_after_third_event() -> None:
    model = TinyFourClassModel()
    optimizer = make_optimizer(model)
    scaler = SyntheticGradScaler({1, 2, 3})
    events: list[int] = []

    def callback(step: int, details: dict[str, object]) -> None:
        if details["amp_overflow_event"]:
            events.append(step)

    try:
        run_epoch(
            model,
            make_loader(4),
            device=torch.device("cpu"),
            amp_enabled=True,
            optimizer=optimizer,
            scaler=scaler,
            max_consecutive_amp_overflows=3,
            step_callback=callback,
        )
    except FloatingPointError as error:
        assert "3 consecutive" in str(error)
    else:
        raise AssertionError("persistent overflow did not fail")
    assert events == [1, 2, 3]
    assert scaler.step_calls == scaler.update_calls == 3


def test_d_nonfinite_forward_loss_remains_immediately_fatal() -> None:
    model = TinyFourClassModel(nonfinite_loss=True)
    optimizer = make_optimizer(model)
    scaler = SyntheticGradScaler(set())
    try:
        run_epoch(
            model,
            make_loader(1),
            device=torch.device("cpu"),
            amp_enabled=True,
            optimizer=optimizer,
            scaler=scaler,
            max_consecutive_amp_overflows=3,
        )
    except FloatingPointError as error:
        assert "CE is missing or non-finite" in str(error)
    else:
        raise AssertionError("non-finite forward loss did not fail")
    assert scaler.step_calls == scaler.update_calls == 0


def test_e_epoch8_best_checkpoint_resume_dry_load() -> None:
    config = read_json(CONFIG)
    model = FourClassPedalDecoderOnlyModel.from_pretrained_performance_embeddings(
        config["checkpoint_path"],
        num_layers=2,
        init_seed=42,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    optimizer = build_decoder_only_optimizer(
        model,
        pretrained_lr=float(config["encoder_lr"]),
        fresh_lr=float(config["head_lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=True, init_scale=float(config["amp_init_scale"])
    )
    checkpoint = restore_training_checkpoint(BEST, model, optimizer, scaler)
    assert checkpoint["epoch"] == 8
    assert checkpoint["global_optimizer_step"] == 17792
    assert checkpoint["best_epoch"] == 8
    assert math.isclose(
        checkpoint["best_validation_ce"], 0.17593201818653073, rel_tol=0.0, abs_tol=1e-15
    )
    assert checkpoint["early_stopping_counter"] == 0
    assert len(optimizer.state) == 44
    assert scaler.get_scale() == 262144.0
    assert "rng_state" not in checkpoint


class AMPRecoveryTargetedTests(unittest.TestCase):
    def test_a(self) -> None:
        test_a_normal_finite_step_is_unchanged()

    def test_b(self) -> None:
        test_b_isolated_overflow_recovers_logs_status_and_resets()

    def test_c(self) -> None:
        test_c_three_consecutive_overflows_fail_after_third_event()

    def test_d(self) -> None:
        test_d_nonfinite_forward_loss_remains_immediately_fatal()

    def test_e(self) -> None:
        test_e_epoch8_best_checkpoint_resume_dry_load()


if __name__ == "__main__":
    unittest.main()
