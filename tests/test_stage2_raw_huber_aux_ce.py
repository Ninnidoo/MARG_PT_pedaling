"""Targeted tests for Raw CC64 Huber + auxiliary standard CE."""

from __future__ import annotations

import tempfile
from types import SimpleNamespace
from pathlib import Path

import torch
from torch import nn

from scripts.run_stage2_encoder_only_raw_huber_aux_ce_v0 import (
    checkpoint_payload,
    restore_training_checkpoint,
    run_epoch,
)
from src.stage2_binary.training import build_binary_optimizer

from src.stage2_four_class.loss_objectives import (
    HUBER_DELTA_NORMALIZED,
    continuous_cc64_to_classes,
    normalized_scalars_to_cc64,
)
from src.stage2_four_class.raw_cc64_huber import (
    RawCC64HuberEncoderModel,
    raw_cc64_to_normalized_targets,
)
from src.stage2_four_class.raw_huber_aux_ce import (
    LAMBDA_CE,
    RawHuberAuxCEEncoderModel,
)
from src.stage2_four_class.representation import classify_cc64


class DummyEncoder(nn.Module):
    def __init__(self, hidden_size: int = 12) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size, dropout_rate=0.0)
        self.embedding = nn.Embedding(6000, hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        batch, flat = input_ids.shape
        notes = flat // 8
        first = input_ids.view(batch, notes, 8)[:, :, 0]
        return SimpleNamespace(last_hidden_state=self.embedding(first))


class SyntheticGradScaler:
    def __init__(self, overflow_attempts: set[int]) -> None:
        self.overflow_attempts = set(overflow_attempts)
        self.attempt = self.step_calls = self.update_calls = 0
        self.scale_value = 1024.0
        self.found_nonfinite = False

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        self.attempt += 1
        self.found_nonfinite = self.attempt in self.overflow_attempts
        if self.found_nonfinite:
            optimizer.param_groups[0]["params"][0].grad.fill_(float("inf"))

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        self.step_calls += 1
        if not self.found_nonfinite:
            optimizer.step()

    def update(self) -> None:
        self.update_calls += 1
        if self.found_nonfinite:
            self.scale_value *= 0.5

    def get_scale(self) -> float:
        return self.scale_value

    def state_dict(self) -> dict[str, float | int]:
        return {"scale": self.scale_value, "attempt": self.attempt}

    def load_state_dict(self, state: dict[str, float | int]) -> None:
        self.scale_value = float(state["scale"])
        self.attempt = int(state["attempt"])


def batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs = torch.randint(0, 5000, (2, 24), dtype=torch.long)
    attention = torch.ones_like(inputs)
    mask = torch.ones((2, 3), dtype=torch.bool)
    raw = torch.tensor([
        [[0, 18, 57, 121], [25, 26, 63, 64], [72, 79, 96, 127]],
        [[5, 37, 88, 109], [12, 55, 103, 104], [1, 40, 80, 126]],
    ])
    return inputs, attention, mask, raw


def loader(steps: int) -> list[dict[str, torch.Tensor]]:
    inputs, attention, mask, raw = batch()
    value = {"input_ids": inputs, "token_attention_mask": attention, "note_mask": mask, "pedal_targets": raw}
    return [{key: tensor.clone() for key, tensor in value.items()} for _ in range(steps)]


def optimizer_for(model: RawHuberAuxCEEncoderModel) -> torch.optim.Optimizer:
    return build_binary_optimizer(model, encoder_lr=1e-5, head_lr=1e-4, weight_decay=0.01)


def test_amp_a_finite_normal_step() -> None:
    model = RawHuberAuxCEEncoderModel(DummyEncoder())
    optimizer = optimizer_for(model); scaler = SyntheticGradScaler(set()); details: dict[str, object] = {}
    metrics, checks = run_epoch(model, loader(1), device=torch.device("cpu"), amp_enabled=True,
        optimizer=optimizer, scaler=scaler, step_callback=lambda _step, value: details.update(value),
        inspect_finite_gradient_groups=True)
    assert all(torch.isfinite(torch.tensor(metrics[key])) for key in ("huber", "ce", "total"))
    assert checks["optimizer_steps"] == checks["optimizer_attempts"] == 1
    assert details["optimizer_steps"] == details["optimizer_attempts"] == 1
    assert checks["amp_overflow_events"] == 0
    assert all(details["gradient_groups_finite_nonzero"].values())
    assert scaler.step_calls == scaler.update_calls == 1


def test_amp_b_isolated_overflow_recovers_and_resets() -> None:
    model = RawHuberAuxCEEncoderModel(DummyEncoder())
    optimizer = optimizer_for(model); scaler = SyntheticGradScaler({1}); events: list[dict[str, object]] = []
    _, checks = run_epoch(model, loader(2), device=torch.device("cpu"), amp_enabled=True,
        optimizer=optimizer, scaler=scaler,
        step_callback=lambda _step, value: events.append(dict(value)) if value["amp_overflow_event"] else None)
    assert checks["optimizer_steps"] == 1 and checks["optimizer_attempts"] == 2
    assert checks["amp_overflow_events"] == 1 and checks["consecutive_amp_overflows"] == 0
    assert checks["max_consecutive_amp_overflows"] == 1
    assert scaler.step_calls == scaler.update_calls == 2
    assert events[0]["scale_before"] == 1024.0 and events[0]["scale_after"] == 512.0
    assert events[0]["affected_groups"] == ["encoder"]


def test_amp_c_three_consecutive_overflows_are_fatal() -> None:
    model = RawHuberAuxCEEncoderModel(DummyEncoder())
    optimizer = optimizer_for(model); scaler = SyntheticGradScaler({1, 2, 3})
    try:
        run_epoch(model, loader(4), device=torch.device("cpu"), amp_enabled=True,
            optimizer=optimizer, scaler=scaler, max_consecutive_amp_overflows=3)
    except FloatingPointError as error:
        assert "3 consecutive" in str(error)
    else:
        raise AssertionError("persistent overflow did not fail")
    assert scaler.step_calls == scaler.update_calls == 3


def test_amp_d_nonfinite_component_is_immediately_fatal() -> None:
    model = RawHuberAuxCEEncoderModel(DummyEncoder())
    original_forward = model.forward
    def nonfinite_forward(*args: object, **kwargs: object):
        output = original_forward(*args, **kwargs)
        output.component_losses["ce"] = torch.tensor(float("nan"))
        return output
    model.forward = nonfinite_forward  # type: ignore[method-assign]
    scaler = SyntheticGradScaler(set())
    try:
        run_epoch(model, loader(1), device=torch.device("cpu"), amp_enabled=True,
            optimizer=optimizer_for(model), scaler=scaler)
    except FloatingPointError as error:
        assert "ce objective is non-finite" in str(error)
    else:
        raise AssertionError("non-finite CE did not fail")
    assert scaler.step_calls == scaler.update_calls == 0


def test_amp_e_parameter_corruption_is_immediately_fatal() -> None:
    model = RawHuberAuxCEEncoderModel(DummyEncoder())
    with torch.no_grad(): next(model.parameters()).fill_(float("inf"))
    try:
        run_epoch(model, loader(1), device=torch.device("cpu"), amp_enabled=True,
            optimizer=optimizer_for(model), scaler=SyntheticGradScaler(set()))
    except FloatingPointError as error:
        assert "non-finite model parameter" in str(error)
    else:
        raise AssertionError("parameter corruption did not fail")


def test_amp_f_checkpoint_is_complete_and_restorable() -> None:
    model = RawHuberAuxCEEncoderModel(DummyEncoder()); optimizer = optimizer_for(model); scaler = SyntheticGradScaler(set())
    metrics, _ = run_epoch(model, loader(1), device=torch.device("cpu"), amp_enabled=True, optimizer=optimizer, scaler=scaler)
    row = {"epoch": 1, "validation_total": metrics["total"], "validation_huber": metrics["huber"],
        "validation_ce": metrics["ce"], "validation_raw_cc64_mae": metrics["raw_cc64_mae"]}
    payload = checkpoint_payload(model, {"seed": 42}, row, optimizer=optimizer, scaler=scaler,
        best_epoch=1, best_validation_total=metrics["total"], early_stopping_counter=0,
        global_optimizer_step=1, global_optimizer_attempt=1, amp_overflow_events_total=0,
        max_consecutive_amp_overflows=0, overflow_events=[])
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "checkpoint.pt"; torch.save(payload, path)
        restored_model = RawHuberAuxCEEncoderModel(DummyEncoder()); restored_optimizer = optimizer_for(restored_model); restored_scaler = SyntheticGradScaler(set())
        restored = restore_training_checkpoint(path, restored_model, restored_optimizer, restored_scaler)
    required = {"model_state","optimizer_state","grad_scaler_state","rng_state","epoch","global_optimizer_step",
        "best_epoch","best_validation_total","early_stopping_counter","amp_overflow_events_total",
        "max_consecutive_amp_overflows","configuration"}
    assert required.issubset(restored)
    assert set(restored["rng_state"]) == {"python", "numpy", "torch_cpu", "torch_cuda"}
    assert restored["global_optimizer_step"] == 1 and len(restored_optimizer.state) > 0
    assert restored_scaler.get_scale() == scaler.get_scale()


def test_dual_head_shapes_and_independent_targets() -> None:
    model = RawHuberAuxCEEncoderModel(DummyEncoder())
    inputs, attention, mask, raw = batch()
    output = model(inputs, attention, raw, mask)
    assert output.predictions.shape == (2, 3, 4)
    assert output.auxiliary_logits is not None
    assert output.auxiliary_logits.shape == (2, 3, 4, 4)
    assert torch.equal(raw_cc64_to_normalized_targets(raw), raw.float() / 127.0)
    boundaries = torch.tensor(
        [[25, 26, 63, 64], [103, 104, 0, 127]], dtype=torch.long
    )
    assert torch.equal(
        classify_cc64(boundaries),
        torch.tensor([[0, 1, 1, 2], [2, 3, 0, 3]]),
    )
    assert len(torch.unique(raw_cc64_to_normalized_targets(raw))) > 4


def test_hybrid_arithmetic_and_canonical_constants() -> None:
    model = RawHuberAuxCEEncoderModel(DummyEncoder())
    inputs, attention, mask, raw = batch()
    output = model(inputs, attention, raw, mask)
    assert model.delta == HUBER_DELTA_NORMALIZED == 14 / 127
    assert model.lambda_ce == LAMBDA_CE == 0.1
    assert torch.allclose(
        output.component_losses["total"],
        output.component_losses["huber"] + 0.1 * output.component_losses["ce"],
        rtol=0.0,
        atol=1e-7,
    )
    assert torch.equal(output.loss, output.component_losses["total"])


def test_all_head_and_encoder_gradients_and_aux_encoder_contribution() -> None:
    model = RawHuberAuxCEEncoderModel(DummyEncoder())
    inputs, attention, mask, raw = batch()
    output = model(inputs, attention, raw, mask)
    assert output.loss is not None and torch.isfinite(output.loss)
    output.loss.backward()
    groups = [
        list(model.encoder.parameters()),
        list(model.regression_heads.parameters()),
        list(model.classification_heads.parameters()),
    ]
    for parameters in groups:
        gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
        assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
        assert any(torch.count_nonzero(gradient) > 0 for gradient in gradients)
    model.zero_grad(set_to_none=True)
    ce_only = model(inputs, attention, raw, mask).component_losses["ce"]
    ce_only.backward()
    encoder_gradients = [p.grad for p in model.encoder.parameters() if p.grad is not None]
    assert encoder_gradients and any(torch.count_nonzero(g) > 0 for g in encoder_gradients)


def test_regression_inference_is_independent_of_auxiliary_logits() -> None:
    torch.manual_seed(42)
    model = RawHuberAuxCEEncoderModel(DummyEncoder()).eval()
    inputs, attention, mask, _raw = batch()
    before = model(inputs, attention, note_mask=mask).predictions.detach().clone()
    before_cc64 = normalized_scalars_to_cc64(before)
    with torch.no_grad():
        for head in model.classification_heads:
            head.weight.fill_(1234.0)
            head.bias.fill_(-987.0)
    after = model(inputs, attention, note_mask=mask).predictions.detach()
    assert torch.equal(before, after)
    assert torch.equal(before_cc64, normalized_scalars_to_cc64(after))


def test_existing_raw_huber_regression_path_is_unchanged() -> None:
    torch.manual_seed(42)
    hybrid = RawHuberAuxCEEncoderModel(DummyEncoder()).eval()
    raw_model = RawCC64HuberEncoderModel(DummyEncoder()).eval()
    raw_model.encoder.load_state_dict(hybrid.encoder.state_dict(), strict=True)
    raw_model.regression_heads.load_state_dict(hybrid.regression_heads.state_dict(), strict=True)
    inputs, attention, mask, raw = batch()
    hybrid_output = hybrid(inputs, attention, raw, mask)
    raw_output = raw_model(inputs, attention, raw, mask)
    assert torch.equal(hybrid_output.predictions, raw_output.predictions)
    assert torch.equal(
        hybrid_output.component_losses["huber"], raw_output.loss
    )
    cc64 = normalized_scalars_to_cc64(hybrid_output.predictions)
    assert torch.equal(
        continuous_cc64_to_classes(cc64), continuous_cc64_to_classes(normalized_scalars_to_cc64(raw_output.predictions))
    )
