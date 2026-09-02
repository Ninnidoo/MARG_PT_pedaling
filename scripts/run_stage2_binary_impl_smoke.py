#!/usr/bin/env python3
"""Short implementation-only overfit check for both binary Stage 2 heads."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.stage2_binary.dataset import (  # noqa: E402
    BinaryManifestSubsetDataset,
    binary_stage2_collate_fn,
    decode_joint_targets,
)
from src.stage2_binary.model import (  # noqa: E402
    IndependentBinaryPedalModel,
    JointBinaryPedalModel,
)
from src.stage2_binary.training import build_binary_optimizer  # noqa: E402
from src.stage2_encoder_only.training import set_deterministic_seed  # noqa: E402


MODEL_CLASSES = {
    "independent_4x2": IndependentBinaryPedalModel,
    "joint_16": JointBinaryPedalModel,
}


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _tensor_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _parameter_digest(parameters: Any) -> str:
    digest = hashlib.sha256()
    with torch.no_grad():
        for parameter in parameters:
            value = parameter.detach().cpu().contiguous()
            digest.update(str(tuple(value.shape)).encode())
            digest.update(str(value.dtype).encode())
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _forward(model: torch.nn.Module, batch: dict[str, Any], amp: bool) -> Any:
    context = (
        torch.amp.autocast("cuda", dtype=torch.float16)
        if amp
        else nullcontext()
    )
    with context:
        return model(
            input_ids=batch["input_ids"],
            token_attention_mask=batch["token_attention_mask"],
            note_mask=batch["note_mask"],
            **{model.target_name: batch[model.target_name]},
        )


def _evaluate(
    model: torch.nn.Module,
    batches: list[dict[str, Any]],
    amp: bool,
) -> dict[str, float | int]:
    model.eval()
    loss_sum = 0.0
    valid_notes = 0
    valid_bits = 0
    correct_bits = 0
    correct_patterns = 0
    with torch.no_grad():
        for batch in batches:
            output = _forward(model, batch, amp)
            if output.loss is None or not bool(torch.isfinite(output.loss)):
                raise FloatingPointError("evaluation loss is missing or non-finite")
            note_count = int(batch["note_mask"].sum().item())
            loss_sum += float(output.loss.item()) * note_count
            valid_notes += note_count
            binary_targets = batch["binary_targets"]
            if model.target_name == "binary_targets":
                predicted_bits = output.logits.argmax(dim=-1)
            else:
                predicted_bits = decode_joint_targets(output.logits.argmax(dim=-1))
            valid = binary_targets != -100
            correct = (predicted_bits == binary_targets) & valid
            valid_bits += int(valid.sum().item())
            correct_bits += int(correct.sum().item())
            valid_note_mask = valid.all(dim=-1)
            exact = ((predicted_bits == binary_targets) | ~valid).all(dim=-1)
            correct_patterns += int((exact & valid_note_mask).sum().item())
    return {
        "loss": loss_sum / valid_notes,
        "binary_accuracy": correct_bits / valid_bits,
        "exact_pattern_accuracy": correct_patterns / valid_notes,
        "valid_notes": valid_notes,
        "valid_binary_targets": valid_bits,
    }


def _load_batches(config: dict[str, Any], device: torch.device) -> tuple[list[dict[str, Any]], str]:
    dataset = BinaryManifestSubsetDataset(
        _resolve(config["train_manifest"]),
        [config["fixture_path"]],
        window_notes=config["window_notes"],
        window_starts=config["window_starts"],
        required_source=config["fixture_source"],
    )
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=binary_stage2_collate_fn,
    )
    batches = [_move_batch(batch, device) for batch in loader]
    flat_inputs = torch.cat([batch["input_ids"].cpu() for batch in batches])
    return batches, _tensor_digest(flat_inputs)


def _verify_fixture_split(config: dict[str, Any]) -> None:
    with _resolve(config["asap_split_artifact"]).open(
        newline="", encoding="utf-8"
    ) as handle:
        matches = [
            row
            for row in csv.DictReader(handle)
            if row["performance_path"] == config["fixture_path"]
        ]
    if len(matches) != 1 or matches[0]["split"] != "train":
        raise RuntimeError("smoke fixture is not uniquely in the existing ASAP train split")


def _run_model(
    name: str,
    config: dict[str, Any],
    batches: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    set_deterministic_seed(config["seed"])
    model = MODEL_CLASSES[name].from_pretrained(
        _resolve(config["checkpoint"]),
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    initial_encoder_digest = _parameter_digest(model.encoder.parameters())
    counts = {
        "parameters": model.parameter_count,
        "trainable_parameters": model.trainable_parameter_count,
        "encoder_parameters": model.encoder_parameter_count,
        "head_parameters": model.prediction_head_parameter_count,
    }
    if counts["parameters"] != counts["trainable_parameters"]:
        raise AssertionError("the official encoder and prediction head must all be trainable")
    model.to(device)
    optimizer = build_binary_optimizer(
        model,
        encoder_lr=config["encoder_lr"],
        head_lr=config["head_lr"],
        weight_decay=config["weight_decay"],
    )
    amp = bool(config["amp"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=1024.0)
    initial = _evaluate(model, batches, amp)
    history = []
    encoder_gradient_verified = False
    finite_gradients_verified = False
    model.train()
    for step in range(config["max_steps"]):
        batch = batches[step % len(batches)]
        optimizer.zero_grad(set_to_none=True)
        output = _forward(model, batch, amp)
        if output.loss is None or not bool(torch.isfinite(output.loss)):
            raise FloatingPointError("training loss is missing or non-finite")
        scaler.scale(output.loss).backward()
        scaler.unscale_(optimizer)
        encoder_grads = [p.grad for p in model.encoder.parameters() if p.requires_grad]
        head_grads = [p.grad for p in model.prediction_head_parameters()]
        encoder_gradient_verified = encoder_gradient_verified or any(
            grad is not None and bool(torch.any(grad != 0)) for grad in encoder_grads
        )
        gradients = [grad for grad in encoder_grads + head_grads if grad is not None]
        finite_gradients_verified = finite_gradients_verified or (
            bool(gradients) and all(bool(torch.isfinite(grad).all()) for grad in gradients)
        )
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=config["max_grad_norm"],
            error_if_nonfinite=True,
        )
        scaler.step(optimizer)
        scaler.update()
        history.append(float(output.loss.detach().item()))
    final = _evaluate(model, batches, amp)
    final_encoder_digest = _parameter_digest(model.encoder.parameters())
    if not encoder_gradient_verified or not finite_gradients_verified:
        raise AssertionError("finite nonzero encoder gradients were not observed")
    if final_encoder_digest == initial_encoder_digest:
        raise AssertionError("encoder parameters did not change")
    if not final["loss"] < initial["loss"] * 0.5:
        raise AssertionError("short overfit loss did not decrease by at least 50%")
    if not final["binary_accuracy"] > initial["binary_accuracy"]:
        raise AssertionError("short overfit binary accuracy did not improve")
    return {
        **counts,
        "initial_encoder_sha256": initial_encoder_digest,
        "final_encoder_sha256": final_encoder_digest,
        "encoder_gradient_verified": encoder_gradient_verified,
        "finite_gradients_verified": finite_gradients_verified,
        "initial": initial,
        "final": final,
        "steps": config["max_steps"],
        "first_training_loss": history[0],
        "last_training_loss": history[-1],
        "passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/stage2_binary_impl_smoke_v0.json",
    )
    parser.add_argument(
        "--output",
        default="analysis/stage2_binary_v0/model_impl_v0/overfit_results.json",
    )
    args = parser.parse_args()
    config = json.loads(_resolve(args.config).read_text(encoding="utf-8"))
    set_deterministic_seed(config["seed"])
    if not torch.cuda.is_available():
        raise RuntimeError("official-encoder smoke requires the assigned CUDA GPU")
    device = torch.device("cuda:0")
    _verify_fixture_split(config)
    batches, input_digest = _load_batches(config, device)
    results: dict[str, Any] = {
        "seed": config["seed"],
        "device": torch.cuda.get_device_name(device),
        "checkpoint": str(_resolve(config["checkpoint"])),
        "train_manifest": str(_resolve(config["train_manifest"])),
        "asap_split_artifact": str(_resolve(config["asap_split_artifact"])),
        "fixture": config["fixture_path"],
        "fixture_source": config["fixture_source"],
        "window_notes": config["window_notes"],
        "window_starts": config["window_starts"],
        "performance_count": 1,
        "window_count": len(config["window_starts"]),
        "input_sha256": input_digest,
        "validation_midi_tokenized": False,
        "smoke_script_test_midi_accessed": False,
        "full_training_started": False,
        "models": {},
    }
    for name in MODEL_CLASSES:
        results["models"][name] = _run_model(
            name, config, batches, device
        )
        torch.cuda.empty_cache()
    hashes = {
        model_result["initial_encoder_sha256"]
        for model_result in results["models"].values()
    }
    results["same_encoder_initialization_verified"] = len(hashes) == 1
    if not results["same_encoder_initialization_verified"]:
        raise AssertionError("models did not begin from identical encoder parameters")
    if not all(math.isfinite(result["final"]["loss"]) for result in results["models"].values()):
        raise FloatingPointError("non-finite final loss")
    output_path = _resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
