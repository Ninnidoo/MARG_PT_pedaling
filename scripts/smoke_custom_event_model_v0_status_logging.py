#!/usr/bin/env python3
"""One aligned real-batch forward/backward plus pre-validation status smoke."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from src.stage2_event_model.dataset import CANONICAL_ALIGNMENT_ID, CustomEventWindowDataset, custom_event_collate_fn
from src.stage2_event_model.losses import CustomEventCriterion, CustomEventLossConfig, load_slot_event_weights
from src.stage2_event_model.model import CustomEventEncoderModelV0
from src.stage2_event_model.status_logging import atomic_write_strict_json, write_run_status

ROOT = Path("/workspace/project")
CACHE = ROOT / "analysis/custom_event_tokenizer_v1"
OUTPUT = ROOT / "analysis/custom_event_model_v0_status_logging_fix"
PRETRAINED = ROOT / "checkpoints/pianist_transformer"


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("real canonical smoke requires CUDA")
    dataset = CustomEventWindowDataset(CACHE, "train", entry_limit=1, cache_input_tokens=True)
    sample = dataset[0]
    batch = custom_event_collate_fn([sample])
    device = torch.device("cuda:0")
    batch = {key: (value.to(device) if isinstance(value, torch.Tensor) else value) for key, value in batch.items()}
    model = CustomEventEncoderModelV0.from_pretrained(
        PRETRAINED, head_init_seed=42, torch_dtype=torch.float32, attn_implementation="eager"
    ).to(device)
    main_weights, terminal_weights, _ = load_slot_event_weights(CACHE / "candidate_event_weights.json")
    criterion = CustomEventCriterion(main_weights, terminal_weights, CustomEventLossConfig()).to(device)
    output = model(
        input_ids=batch["input_ids"], token_attention_mask=batch["token_attention_mask"],
        note_mask=batch["note_mask"],
        owned_representative_positions=batch["owned_representative_positions"],
        owned_onset_mask=batch["owned_onset_mask"],
        initial_representative_positions=batch["initial_representative_positions"],
        initial_valid_mask=batch["initial_valid_mask"],
        terminal_representative_positions=batch["terminal_representative_positions"],
        terminal_valid_mask=batch["terminal_valid_mask"],
    )
    loss = criterion(output, batch)
    tensors = (
        output.main_event_logits, output.main_timing_predictions,
        output.initial_logits, output.terminal_event_logits,
        output.terminal_timing_predictions, loss.total_loss, loss.initial_ce,
        loss.main_event_ce, loss.main_timing_huber,
        loss.terminal_event_ce, loss.terminal_timing_huber,
    )
    if not all(bool(torch.isfinite(value).all()) for value in tensors):
        raise FloatingPointError("real smoke forward/loss non-finite")
    loss.total_loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    gradients_finite = bool(gradients) and all(bool(torch.isfinite(value).all()) for value in gradients)
    if not gradients_finite:
        raise FloatingPointError("real smoke gradient non-finite")

    status = {
        "status": "running", "stage": "training", "epoch": 1,
        "global_optimizer_attempt": 0, "global_optimizer_step": 0,
        "best_epoch": None, "best_val_total": None,
        "latest_train_total": float(loss.total_loss.detach()),
        "latest_val_total": None, "latest_validation_components": None,
        "amp_overflow_count": 0, "max_consecutive_amp_overflow": 0,
        "failure_reason": None, "note_alignment_id": CANONICAL_ALIGNMENT_ID,
        "asap_test_access_count": 0,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    status_path = OUTPUT / "smoke_run_status.json"
    write_run_status(status_path, status)
    readback = json.loads(status_path.read_text(encoding="utf-8"))
    result = {
        "status": "PASS", "precision": "FP32 diagnostic",
        "performance_path": sample["metadata"]["performance_path"],
        "window_index": sample["metadata"]["window_index"],
        "note_alignment_id": CANONICAL_ALIGNMENT_ID,
        "forward_finite": True, "loss_finite": True,
        "backward_finite": True, "gradients_finite": gradients_finite,
        "total_loss": float(loss.total_loss.detach()),
        "best_val_total_readback": readback["best_val_total"],
        "latest_val_total_readback": readback["latest_val_total"],
        "strict_json_readback": True, "optimizer_created": False,
        "optimizer_steps": 0, "training_steps": 0,
        "asap_test_access_count": 0,
    }
    atomic_write_strict_json(OUTPUT / "smoke_result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
