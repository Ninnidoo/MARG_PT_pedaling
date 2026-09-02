#!/usr/bin/env python3
"""Read-only frozen validation forward extraction for the four-state audit."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.run_custom_event_model_v0_canonical_val_inference_v1 import (
    CACHE_ROOT,
    PRETRAINED,
    move_batch,
    preflight,
)
from src.stage2_event_model.dataset import CustomEventWindowDataset, custom_event_collate_fn
from src.stage2_event_model.model import CustomEventEncoderModelV0


OUTPUT = ROOT / "analysis/custom_event_v0_4state_failure_audit_v1/raw_predictions"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.dtype.str.encode() + str(array.shape).encode() + array.tobytes()).hexdigest()


def main() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("frozen extraction requires exactly one visible CUDA device")
    provenance = preflight()
    checkpoint = provenance.pop("checkpoint_payload")
    dataset = CustomEventWindowDataset(CACHE_ROOT, "validation", cache_input_tokens=True)
    if len(dataset) != 1078 or len(dataset.performances) != 71:
        raise RuntimeError("frozen validation window/performance universe changed")
    model = CustomEventEncoderModelV0.from_pretrained(
        PRETRAINED, head_init_seed=42, torch_dtype=torch.float32, attn_implementation="eager"
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    del checkpoint
    device = torch.device("cuda:0")
    model.to(device).eval()
    records: list[dict[str, Any]] = []
    for performance in dataset.performances:
        with np.load(performance.cache_path, allow_pickle=False) as cache:
            records.append({
                "main_event_logits": np.empty((performance.num_onsets, 6, 5), np.float32),
                "main_timing_predictions": np.empty((performance.num_onsets, 6), np.float32),
                "main_event_targets": cache["main_event_target"].astype(np.int8),
                "main_timing_targets": cache["main_tau_target"].astype(np.float32),
                "main_timing_valid_mask": cache["main_timing_valid_mask"].astype(bool),
                "initial_logits": None,
                "initial_target": int(cache["initial_state"]),
                "terminal_event_logits": None,
                "terminal_timing_predictions": None,
                "terminal_event_targets": cache["terminal_event_target"].astype(np.int8),
                "terminal_timing_targets": cache["terminal_log1p_gap_target"].astype(np.float32),
                "terminal_timing_valid_mask": cache["terminal_timing_valid_mask"].astype(bool),
                "seen": np.zeros(performance.num_onsets, dtype=bool),
            })
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0,
                        collate_fn=custom_event_collate_fn)
    windows = owned_total = initial_total = terminal_total = 0
    with torch.inference_mode():
        for batch in loader:
            batch = move_batch(batch, device)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=True):
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
            tensors = (output.main_event_logits, output.main_timing_predictions,
                       output.initial_logits, output.terminal_event_logits,
                       output.terminal_timing_predictions)
            if not all(bool(torch.isfinite(item).all()) for item in tensors):
                raise FloatingPointError("non-finite frozen prediction")
            for batch_index, metadata in enumerate(batch["metadata"]):
                performance_index = int(metadata["performance_index"])
                record = records[performance_index]
                count = int(batch["owned_onset_mask"][batch_index].sum())
                owned = batch["owned_global_onset_indices"][batch_index, :count].cpu().numpy()
                if bool(record["seen"][owned].any()):
                    raise AssertionError("duplicate unique owner")
                record["main_event_logits"][owned] = output.main_event_logits[batch_index, :count].float().cpu().numpy()
                record["main_timing_predictions"][owned] = output.main_timing_predictions[batch_index, :count].float().cpu().numpy()
                record["seen"][owned] = True
                owned_total += count
                if bool(batch["initial_valid_mask"][batch_index]):
                    record["initial_logits"] = output.initial_logits[batch_index].float().cpu().numpy()
                    initial_total += 1
                if bool(batch["terminal_valid_mask"][batch_index]):
                    record["terminal_event_logits"] = output.terminal_event_logits[batch_index].float().cpu().numpy()
                    record["terminal_timing_predictions"] = output.terminal_timing_predictions[batch_index].float().cpu().numpy()
                    terminal_total += 1
                windows += 1
    if (windows, owned_total, initial_total, terminal_total) != (1078, 272927, 71, 71):
        raise AssertionError("frozen extraction accounting mismatch")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    digests = []
    for index, (performance, record) in enumerate(zip(dataset.performances, records, strict=True)):
        if not bool(record.pop("seen").all()) or record["initial_logits"] is None:
            raise AssertionError("incomplete performance prediction")
        destination = OUTPUT / f"{index:04d}.npz"
        temporary = destination.with_name("." + destination.name + ".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **record)
        os.replace(temporary, destination)
        digest = array_sha256(record["main_event_logits"])
        digests.append(digest)
        manifest.append({
            "performance_index": index,
            "performance_path": performance.entry["performance_path"],
            "performance_id": performance.entry["performance_path"],
            "piece_id": performance.entry["piece_id"],
            "cache_file": str(performance.cache_path),
            "raw_prediction_file": str(destination),
            "main_event_logits_sha256": digest,
            "modeled_onsets": performance.num_onsets,
        })
    aggregate = hashlib.sha256("".join(digests).encode()).hexdigest()
    atomic_json(OUTPUT / "manifest.json", {
        "provenance": provenance,
        "validation_performances": 71,
        "validation_windows": 1078,
        "modeled_onsets": 272927,
        "optimizer_steps": 0,
        "training_steps": 0,
        "candidate_generation_count": 0,
        "asap_test_access_count": 0,
        "repedal_execution_count": 0,
        "aggregate_prediction_digest": aggregate,
        "records": manifest,
    })
    print(f"RAW_EXTRACTION_COMPLETE performances=71 onsets=272927 digest={aggregate}", flush=True)


if __name__ == "__main__":
    main()
