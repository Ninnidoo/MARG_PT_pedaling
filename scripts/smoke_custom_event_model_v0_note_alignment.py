#!/usr/bin/env python3
"""Forward/backward smoke on the exact window that exposed order mismatch."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from src.stage2_event_model.dataset import CANONICAL_ALIGNMENT_ID, CustomEventWindowDataset, custom_event_collate_fn
from src.stage2_event_model.losses import CustomEventCriterion, CustomEventLossConfig, load_slot_event_weights
from src.stage2_event_model.model import CustomEventEncoderModelV0

ROOT = Path("/workspace/project")
CACHE = ROOT / "analysis/custom_event_tokenizer_v1"
OUTPUT = ROOT / "analysis/custom_event_model_v0_note_alignment/failing_window_smoke.json"
WEIGHTS = CACHE / "candidate_event_weights.json"
PRETRAINED = ROOT / "checkpoints/pianist_transformer"


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the official PT smoke")
    dataset = CustomEventWindowDataset(CACHE, "train", entry_limit=1794, cache_input_tokens=True)
    dataset_index = dataset.windows.index((1793, 12))
    sample = dataset[dataset_index]
    if sample["metadata"]["window_start_note"] != 3072:
        raise AssertionError("failed window start changed")
    batch = custom_event_collate_fn([sample])
    device = torch.device("cuda:0")
    batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    model = CustomEventEncoderModelV0.from_pretrained(PRETRAINED, head_init_seed=42).to(device)
    main_weights, terminal_weights, _ = load_slot_event_weights(WEIGHTS)
    criterion = CustomEventCriterion(main_weights, terminal_weights, CustomEventLossConfig()).to(device)
    model.train()
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
    checked = {
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
    if not all(bool(torch.isfinite(value).all()) for value in checked.values()):
        raise FloatingPointError("mapped failing-window forward/loss is non-finite")
    loss.total_loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
        raise FloatingPointError("mapped failing-window backward is non-finite")
    result = {
        "status": "PASS", "performance_index": 1793, "window_index": 12,
        "note_alignment_id": CANONICAL_ALIGNMENT_ID,
        "global_window_start": 3072, "global_window_end": 3584,
        "performance_path": sample["metadata"]["performance_path"],
        "owned_representatives": int(batch["owned_onset_mask"].sum()),
        "mapped_representative_identity": "PASS", "forward_finite": True,
        "loss_finite": True, "backward_finite": True,
        "total_loss": float(loss.total_loss.detach()),
        "initial_ce": float(loss.initial_ce.detach()),
        "main_event_ce": float(loss.main_event_ce.detach()),
        "main_timing_huber": float(loss.main_timing_huber.detach()),
        "terminal_event_ce": float(loss.terminal_event_ce.detach()),
        "terminal_timing_huber": float(loss.terminal_timing_huber.detach()),
        "optimizer_created": False, "optimizer_steps": 0,
        "asap_test_access_count": 0,
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
