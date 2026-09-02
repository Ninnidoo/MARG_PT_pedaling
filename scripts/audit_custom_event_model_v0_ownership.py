#!/usr/bin/env python3
"""Full train/validation ownership audit without model inference or training."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage2_event_model.dataset import (
    CANONICAL_CACHE_ID,
    load_cache_manifest,
)
from src.stage2_event_model.ownership import assign_unique_owners

CACHE_ROOT = ROOT / "analysis/custom_event_tokenizer_v1"
OUTPUT_ROOT = ROOT / "analysis/custom_event_model_v0"


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def distribution(values: np.ndarray) -> dict[str, float | int]:
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": int(values.min()),
        "p05": float(np.quantile(values, 0.05)),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "max": int(values.max()),
    }


def audit_split(root: Path, entries: list[dict], split: str) -> dict:
    total_windows = 0
    total_groups = 0
    owned_groups = 0
    zero_owner = 0
    duplicate_owner = 0
    split_groups = 0
    split_pairs = 0
    initial_count = 0
    terminal_count = 0
    owned_counts: list[np.ndarray] = []
    margins: list[np.ndarray] = []
    digest = hashlib.sha256()
    for entry_index, entry in enumerate(entries, 1):
        path = root / entry["cache_file"]
        with np.load(path, allow_pickle=False) as cache:
            notes = int(cache["non_pedal_note_count"])
            first = cache["onset_first_note_index"].astype(np.int64, copy=True)
            last = cache["onset_last_note_index"].astype(np.int64, copy=True)
            representative = cache["onset_representative_note_index"].astype(
                np.int64, copy=True
            )
        result = assign_unique_owners(notes, first, last, representative)
        membership = np.zeros(len(first), dtype=np.int8)
        for indices in result.owned_onset_indices:
            membership[indices] += 1
        local_zero = int(np.sum(membership == 0))
        local_duplicate = int(np.sum(membership > 1))
        if local_zero or local_duplicate:
            raise AssertionError(
                f"ownership failure {entry[performance_path]}: "
                f"zero={local_zero} duplicate={local_duplicate}"
            )
        window_counts = np.asarray(
            [len(indices) for indices in result.owned_onset_indices], dtype=np.int64
        )
        total_windows += len(result.window_starts)
        total_groups += len(first)
        owned_groups += int(membership.sum())
        zero_owner += local_zero
        duplicate_owner += local_duplicate
        split_groups += result.split_chord_group_count
        split_pairs += result.split_chord_window_group_pairs
        initial_count += 1
        terminal_count += 1
        owned_counts.append(window_counts)
        margins.append(result.owner_margin)
        digest.update(split.encode("utf-8"))
        digest.update(entry["cache_file"].encode("utf-8"))
        digest.update(result.owner_window_index.tobytes())
        if entry_index == 1 or entry_index % 250 == 0 or entry_index == len(entries):
            print(f"ownership {split} {entry_index}/{len(entries)}", flush=True)
    owned_values = np.concatenate(owned_counts)
    margin_values = np.concatenate(margins)
    if owned_groups != total_groups or zero_owner or duplicate_owner:
        raise AssertionError("full ownership invariants failed")
    return {
        "split": split,
        "performances": len(entries),
        "total_windows": total_windows,
        "total_distinct_onset_groups": total_groups,
        "owned_onset_groups": owned_groups,
        "zero_owner_count": zero_owner,
        "duplicate_owner_count": duplicate_owner,
        "groups_rejected_from_some_windows_due_split_chord": split_groups,
        "rejected_split_chord_window_group_pairs": split_pairs,
        "initial_supervision_count": initial_count,
        "terminal_supervision_count": terminal_count,
        "owned_onset_count_per_window": distribution(owned_values),
        "owner_context_margin": distribution(margin_values),
        "ownership_sha256": digest.hexdigest(),
    }


def main() -> None:
    root, config, manifest = load_cache_manifest(CACHE_ROOT)
    by_split = {
        split: [entry for entry in manifest["entries"] if entry["split"] == split]
        for split in ("train", "validation")
    }
    payload = {
        "cache_id": CANONICAL_CACHE_ID,
        "window_notes": 512,
        "stride_notes": 256,
        "ownership_rule": "eligible full group; maximum representative context margin; tie smaller window start",
        "train": audit_split(root, by_split["train"], "train"),
        "validation": audit_split(root, by_split["validation"], "validation"),
        "asap_test_access_count": 0,
        "training_steps": 0,
        "optimizer_steps": 0,
    }
    if payload["train"]["initial_supervision_count"] != 2062:
        raise AssertionError("train initial supervision count mismatch")
    if payload["validation"]["initial_supervision_count"] != 71:
        raise AssertionError("validation initial supervision count mismatch")
    if payload["train"]["terminal_supervision_count"] != 2062:
        raise AssertionError("train terminal supervision count mismatch")
    if payload["validation"]["terminal_supervision_count"] != 71:
        raise AssertionError("validation terminal supervision count mismatch")
    atomic_json(OUTPUT_ROOT / "window_ownership_stats.json", payload)
    atomic_json(
        OUTPUT_ROOT / "config.json",
        {
            "experiment": "custom_event_model_v0_implementation_only",
            "tokenizer_cache_id": CANONICAL_CACHE_ID,
            "tokenizer_cache_config": str(CACHE_ROOT / "config.json"),
            "pretrained_checkpoint": str(ROOT / "checkpoints/pianist_transformer"),
            "encoder_loading": "BinaryStage2EncoderBase.from_pretrained -> PianoT5Gemma.from_pretrained -> get_encoder",
            "expected_hidden_size": 768,
            "window_notes": 512,
            "stride_notes": 256,
            "input_features": ["Pitch", "IOI", "Velocity", "Duration"],
            "pedal_input_mask_id": 1,
            "training_steps": 0,
            "optimizer_steps": 0,
            "asap_test_access_count": 0,
        },
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
