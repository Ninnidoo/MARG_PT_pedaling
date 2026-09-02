#!/usr/bin/env python3
"""One real pretrained-encoder forward/backward smoke; no optimizer or training."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage2_event_model.dataset import (
    CANONICAL_CACHE_ID,
    CustomEventWindowDataset,
    custom_event_collate_fn,
)
from src.stage2_event_model.losses import (
    CustomEventCriterion,
    CustomEventLossConfig,
    load_slot_event_weights,
)
from src.stage2_event_model.model import CustomEventEncoderModelV0

CACHE_ROOT = ROOT / "analysis/custom_event_tokenizer_v1"
OUTPUT_ROOT = ROOT / "analysis/custom_event_model_v0"
CHECKPOINT = ROOT / "checkpoints/pianist_transformer"
WEIGHTS = CACHE_ROOT / "candidate_event_weights.json"
SMOKE_MANIFEST_INDEX = 72


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def finite(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor).all())


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("canonical pretrained smoke requires the available CUDA GPU")
    device = torch.device("cuda:0")
    dataset = CustomEventWindowDataset(
        CACHE_ROOT,
        "train",
        entry_limit=SMOKE_MANIFEST_INDEX + 1,
        cache_input_tokens=True,
    )
    performance = dataset.performances[SMOKE_MANIFEST_INDEX]
    if len(performance.ownership.window_starts) != 1:
        raise AssertionError("chosen smoke performance must fit one canonical window")
    dataset_index = dataset.windows.index((SMOKE_MANIFEST_INDEX, 0))
    sample = dataset[dataset_index]
    if not bool(sample["initial_valid_mask"]) or not bool(sample["terminal_valid_mask"]):
        raise AssertionError("single smoke window must own initial and terminal targets")
    if not bool(sample["terminal_timing_valid_mask"].any()):
        raise AssertionError("smoke sample must exercise terminal timing")
    batch = custom_event_collate_fn([sample])
    for key, value in list(batch.items()):
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)

    model = CustomEventEncoderModelV0.from_pretrained(
        CHECKPOINT,
        head_init_seed=42,
    ).to(device)
    if model.hidden_size != 768:
        raise RuntimeError(f"pretrained hidden size changed: {model.hidden_size}")
    main_weights, terminal_weights, weight_provenance = load_slot_event_weights(WEIGHTS)
    loss_config = CustomEventLossConfig()
    criterion = CustomEventCriterion(
        main_weights, terminal_weights, loss_config
    ).to(device)
    model.train()
    output = model(
        input_ids=batch["input_ids"],
        token_attention_mask=batch["token_attention_mask"],
        note_mask=batch["note_mask"],
        owned_representative_positions=batch["owned_representative_positions"],
        owned_onset_mask=batch["owned_onset_mask"],
        initial_representative_positions=batch["initial_representative_positions"],
        initial_valid_mask=batch["initial_valid_mask"],
        terminal_representative_positions=batch["terminal_representative_positions"],
        terminal_valid_mask=batch["terminal_valid_mask"],
    )
    loss = criterion(output, batch)
    tensors = {
        "main_event_logits": output.main_event_logits,
        "main_timing_predictions": output.main_timing_predictions,
        "initial_logits": output.initial_logits,
        "terminal_event_logits": output.terminal_event_logits,
        "terminal_timing_predictions": output.terminal_timing_predictions,
        "total_loss": loss.total_loss,
        "initial_ce": loss.initial_ce,
        "main_event_ce": loss.main_event_ce,
        "main_timing_huber": loss.main_timing_huber,
        "terminal_event_ce": loss.terminal_event_ce,
        "terminal_timing_huber": loss.terminal_timing_huber,
    }
    if not all(finite(value) for value in tensors.values()):
        raise FloatingPointError("smoke forward produced non-finite values")
    loss.total_loss.backward()
    groups = {
        "encoder": list(model.encoder.parameters()),
        "initial_head": list(model.initial_head.parameters()),
        "main_event_heads": list(model.main_event_heads.parameters()),
        "main_timing_heads": list(model.main_timing_heads.parameters()),
        "terminal_event_heads": list(model.terminal_event_heads.parameters()),
        "terminal_timing_heads": list(model.terminal_timing_heads.parameters()),
    }
    gradient_summary = {}
    for name, parameters in groups.items():
        gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
        gradient_summary[name] = {
            "gradient_tensor_count": len(gradients),
            "all_finite": bool(gradients)
            and all(bool(torch.isfinite(gradient).all()) for gradient in gradients),
            "absolute_sum": sum(float(gradient.detach().abs().sum()) for gradient in gradients),
        }
    if not all(
        value["all_finite"] and value["absolute_sum"] > 0
        for value in gradient_summary.values()
    ):
        raise FloatingPointError(f"smoke gradient failure: {gradient_summary}")
    head_counts = model.head_parameter_counts
    payload = {
        "status": "passed",
        "cache_id": CANONICAL_CACHE_ID,
        "performance_path": sample["metadata"]["performance_path"],
        "notes": int(batch["note_mask"].sum().item()),
        "owned_onsets": int(batch["owned_onset_mask"].sum().item()),
        "initial_valid_count": int(batch["initial_valid_mask"].sum().item()),
        "terminal_valid_count": int(batch["terminal_valid_mask"].sum().item()),
        "pretrained_checkpoint": str(CHECKPOINT),
        "pretrained_loading_path": "PianoT5Gemma.from_pretrained -> get_encoder",
        "hidden_size": model.hidden_size,
        "head_init_seed": model.head_init_seed,
        "encoder_parameter_count": model.encoder_parameter_count,
        "head_parameter_counts": head_counts,
        "prediction_head_parameter_count": model.prediction_head_parameter_count,
        "total_parameter_count": model.parameter_count,
        "forward_all_finite": True,
        "losses": {name: float(value.detach()) for name, value in tensors.items() if "logits" not in name and "predictions" not in name},
        "valid_target_counts": loss.valid_target_counts,
        "gradient_groups": gradient_summary,
        "backward_all_finite": True,
        "cuda_device": torch.cuda.get_device_name(device),
        "cuda_max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "event_weight_provenance": weight_provenance,
        "loss_config": asdict(loss_config),
        "training_steps": 0,
        "optimizer_created": False,
        "optimizer_steps": 0,
        "asap_test_access_count": 0,
    }
    atomic_json(OUTPUT_ROOT / "forward_smoke.json", payload)
    atomic_json(
        OUTPUT_ROOT / "loss_config.json",
        {
            **asdict(loss_config),
            "initial_ce": "unweighted standard CE",
            "main_event_ce": "slot-specific frozen train-frequency weighted CE; macro average over 6 slots",
            "main_timing": "Smooth L1 over event!=NONE AND timing_valid; macro average over 6 slots",
            "terminal_event_ce": "slot-specific frozen train-frequency weighted CE; macro average over 4 slots",
            "terminal_timing": "Smooth L1 over event!=NONE AND timing_valid; macro average over 4 slots",
            "empty_timing_slot": "finite differentiable zero",
            "event_weight_provenance": weight_provenance,
            "training_steps": 0,
            "optimizer_steps": 0,
        },
    )
    config_path = OUTPUT_ROOT / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(
        {
            "actual_hidden_size": model.hidden_size,
            "head_init_seed": model.head_init_seed,
            "encoder_parameter_count": model.encoder_parameter_count,
            "prediction_head_parameter_count": model.prediction_head_parameter_count,
            "head_parameter_counts": head_counts,
            "total_parameter_count": model.parameter_count,
            "event_weight_provenance": weight_provenance,
        }
    )
    atomic_json(config_path, config)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
