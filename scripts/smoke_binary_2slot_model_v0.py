#!/usr/bin/env python3
"""Read-only canonical-train smoke for Binary 2-Slot Model v0.

Loads the pretrained PT encoder, runs one complete small train performance as
PRE -> owned MAIN -> POST, computes the provisional smoke-only boundary-weight
1.0 objective, and performs exactly one backward call. It creates no optimizer
and writes no checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage2_binary_2slot.dataset import (
    Binary2SlotPerformanceDataset,
    binary_2slot_collate_fn,
)
from src.stage2_binary_2slot.losses import Binary2SlotCriterion, Binary2SlotLossConfig
from src.stage2_binary_2slot.model import StateConditionedBinary2SlotModel
from src.stage2_binary_2slot.trainer import RolloutChunk, StatefulRolloutEngine


CACHE_ROOT = ROOT / "analysis/custom_event_tokenizer_v1"
CHECKPOINT = ROOT / "checkpoints/pianist_transformer"
OUTPUT = ROOT / "analysis/binary_2slot_model_implementation_v0"
SMOKE_TRAIN_PERFORMANCE_INDEX = 72


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def boundary_forward(model, value: dict, device: torch.device) -> torch.Tensor:
    return model.encode_boundary(
        value["input_ids"].unsqueeze(0).to(device),
        value["token_attention_mask"].unsqueeze(0).to(device),
        value["note_mask"].unsqueeze(0).to(device),
        torch.tensor([value["query_position"]], dtype=torch.long, device=device),
    )[0]


def combine(chunks: tuple[RolloutChunk, ...]) -> tuple:
    return (
        torch.cat([chunk.count_logits for chunk in chunks]),
        torch.cat([chunk.timing_predictions for chunk in chunks]),
        torch.cat([chunk.target_counts for chunk in chunks]),
        torch.cat([chunk.timing_targets for chunk in chunks]),
        torch.cat([chunk.timing_mask for chunk in chunks]),
        tuple(region for chunk in chunks for region in chunk.regions),
    )


def grad_summary(parameters) -> dict[str, object]:
    parameters = list(parameters)
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    finite = bool(gradients) and all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    return {
        "parameters": len(parameters),
        "parameters_with_grad": len(gradients),
        "all_gradients_finite": finite,
        "max_abs_gradient": max((float(gradient.detach().abs().max().item()) for gradient in gradients), default=None),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    dataset = Binary2SlotPerformanceDataset(
        CACHE_ROOT,
        "train",
        performance_indices=(SMOKE_TRAIN_PERFORMANCE_INDEX,),
        cache_input_tokens=True,
    )
    if len(dataset) != 1:
        raise AssertionError("selected smoke performance must fit exactly one owner window")
    sample = dataset[0]
    timeline = dataset.timeline(0)
    if len(sample["human_intervals"]) != len(timeline.main):
        raise AssertionError("smoke owner window must contain every MAIN interval")
    batch = binary_2slot_collate_fn((sample,))
    pre_input = dataset.boundary_input(0, "PRE")
    post_input = dataset.boundary_input(0, "POST")

    model = StateConditionedBinary2SlotModel.from_pretrained(
        CHECKPOINT, head_init_seed=42
    ).to(device)
    model.train()
    model.zero_grad(set_to_none=True)
    criterion = Binary2SlotCriterion(
        Binary2SlotLossConfig(boundary_loss_weight=1.0)
    ).to(device)

    pre_hidden = boundary_forward(model, pre_input, device)
    main_encoding = model.encode_main(
        batch["input_ids"].to(device),
        batch["token_attention_mask"].to(device),
        batch["note_mask"].to(device),
        batch["owned_representative_positions"].to(device),
        batch["owned_onset_mask"].to(device),
    )
    owned_mask = batch["owned_onset_mask"][0].to(device)
    main_hidden = main_encoding.owned_onset_hidden_states[0][owned_mask]
    post_hidden = boundary_forward(model, post_input, device)

    engine = StatefulRolloutEngine(model)
    pre_chunk = engine.begin_performance(timeline.performance_id, pre_hidden, timeline.pre)
    main_chunk = engine.process_main_chunk(
        timeline.performance_id, main_hidden, sample["human_intervals"]
    )
    post_chunk = engine.end_performance(
        timeline.performance_id,
        post_hidden,
        timeline.post,
        expected_main_count=len(timeline.main),
    )
    values = combine((pre_chunk, main_chunk, post_chunk))
    loss = criterion(*values)
    if not bool(torch.isfinite(loss.total_loss)):
        raise FloatingPointError("smoke loss is not finite")
    loss.total_loss.backward()  # exactly one backward; no optimizer exists.

    encoder_grad = grad_summary(model.encoder.parameters())
    count_grad = grad_summary(model.count_head.parameters())
    timing_1_grad = grad_summary(model.timing_head_1.parameters())
    timing_2_grad = grad_summary(model.timing_head_2.parameters())
    if not all(item["all_gradients_finite"] for item in (encoder_grad, count_grad, timing_1_grad, timing_2_grad)):
        raise FloatingPointError("one or more model gradient groups are absent/non-finite")

    config = model.encoder.config
    checkpoint_weights = CHECKPOINT / "model.safetensors"
    parameter_summary = {
        "checkpoint_path": str(CHECKPOINT.resolve()),
        "checkpoint_model_sha256": sha256(checkpoint_weights),
        "hidden_size": model.hidden_size,
        "loaded_encoder_layer_count": int(config.num_hidden_layers),
        "encoder_parameter_count": model.encoder_parameter_count,
        "new_head_parameter_count": model.prediction_head_parameter_count,
        "head_parameter_counts": model.head_parameter_counts,
        "head_initialization_seed": model.head_init_seed,
        "total_parameter_count": model.parameter_count,
        "trainable_parameter_count": model.trainable_parameter_count,
    }
    smoke = {
        "status": "PASS",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "canonical_data": {
            "split": "train",
            "canonical_performance_index": SMOKE_TRAIN_PERFORMANCE_INDEX,
            "performance_id": timeline.performance_id,
            "notes": dataset.performances[0].num_notes,
            "owner_windows": len(dataset),
            "main_intervals": len(timeline.main),
            "asap_test_access_count": 0,
        },
        "forwards": {
            "main_batch": 1,
            "pre_boundary": 1,
            "post_boundary": 1,
            "main_input_shape": list(batch["input_ids"].shape),
            "main_encoder_shape": list(main_encoding.encoder_hidden_states.shape),
            "owned_main_hidden_shape": list(main_hidden.shape),
            "pre_input_shape": [1, pre_input["input_ids"].numel()],
            "pre_query_position": pre_input["query_position"],
            "pre_hidden_shape": list(pre_hidden.shape),
            "post_input_shape": [1, post_input["input_ids"].numel()],
            "post_query_position": post_input["query_position"],
            "post_hidden_shape": list(post_hidden.shape),
        },
        "loss": {
            "boundary_loss_weight": 1.0,
            "boundary_loss_weight_status": "smoke-only provisional; not a frozen research choice",
            "total": float(loss.total_loss.detach().item()),
            "main": float(loss.main_loss.detach().item()),
            "boundary": float(loss.boundary_loss.detach().item()),
            "main_count": float(loss.main_count_loss.detach().item()),
            "main_timing": float(loss.main_timing_loss.detach().item()),
            "boundary_count": float(loss.boundary_count_loss.detach().item()),
            "boundary_timing": float(loss.boundary_timing_loss.detach().item()),
            "finite": True,
            "counts": loss.counts,
        },
        "rollout_diagnostics": engine.diagnostics.as_dict(),
        "gradients": {
            "backward_calls": 1,
            "encoder": encoder_grad,
            "count_head": count_grad,
            "timing_head_1": timing_1_grad,
            "timing_head_2": timing_2_grad,
        },
        "cuda_peak_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
        "prohibited_activity_counts": {
            "optimizer_objects_created": 0,
            "optimizer_steps": 0,
            "training_steps": 0,
            "epochs": 0,
            "checkpoints_created": 0,
            "validation_model_inference": 0,
            "validation_transition_f1": 0,
            "asap_test_metadata_access": 0,
            "asap_test_midi_access": 0,
        },
    }
    (OUTPUT / "parameter_summary.json").write_text(
        json.dumps(parameter_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (OUTPUT / "smoke_test_results.json").write_text(
        json.dumps(smoke, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", "loss": smoke["loss"], "parameters": parameter_summary, "cuda_peak_memory_bytes": smoke["cuda_peak_memory_bytes"]}, indent=2))


if __name__ == "__main__":
    main()
