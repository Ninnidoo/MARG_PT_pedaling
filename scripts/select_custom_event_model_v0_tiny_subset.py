#!/usr/bin/env python3
"""Deterministic train-only coverage selection for tiny memorization."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage2_event_model.dataset import CANONICAL_CACHE_ID, load_cache_manifest
from src.stage2_event_model.ownership import assign_unique_owners
from src.stage2_event_tokenizer.tokenizer import EVENT_NAMES

CACHE_ROOT = ROOT / "analysis/custom_event_tokenizer_v1"
OUTPUT_ROOT = ROOT / "analysis/custom_event_model_v0_tiny_overfit"
MAX_WINDOWS = 16


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def requirements() -> tuple[set[str], set[str]]:
    required = {"initial", "terminal"}
    required |= {f"main_slot_{slot}_non_none" for slot in range(1, 7)}
    required |= {f"terminal_slot_{slot}_non_none" for slot in range(1, 5)}
    desired = set(required)
    desired |= {f"main_class_{name}" for name in EVENT_NAMES[1:]}
    desired |= {f"terminal_class_{name}" for name in EVENT_NAMES[1:]}
    return required, desired


def coverage_for_window(
    window_index: int,
    owner: np.ndarray,
    owned: np.ndarray,
    main_targets: np.ndarray,
    terminal_targets: np.ndarray,
) -> set[str]:
    coverage: set[str] = set()
    if window_index == int(owner[0]):
        coverage.add("initial")
    terminal_valid = window_index == int(owner[-1])
    if terminal_valid:
        coverage.add("terminal")
    selected_main = main_targets[owned]
    for slot in range(6):
        values = selected_main[:, slot]
        if np.any(values != 0):
            coverage.add(f"main_slot_{slot + 1}_non_none")
    for class_index, name in enumerate(EVENT_NAMES[1:], 1):
        if np.any(selected_main == class_index):
            coverage.add(f"main_class_{name}")
        if terminal_valid and np.any(terminal_targets == class_index):
            coverage.add(f"terminal_class_{name}")
    if terminal_valid:
        for slot in range(4):
            if terminal_targets[slot] != 0:
                coverage.add(f"terminal_slot_{slot + 1}_non_none")
    return coverage


def summarize(selected: list[dict]) -> dict:
    initial_count = 0
    owned_count = 0
    main = [
        {"event_target_count": 0, "non_none_count": 0, "timing_valid_count": 0, "class_counts": Counter()}
        for _ in range(6)
    ]
    terminal = [
        {"event_target_count": 0, "non_none_count": 0, "timing_valid_count": 0, "class_counts": Counter()}
        for _ in range(4)
    ]
    terminal_window_count = 0
    for row in selected:
        path = CACHE_ROOT / row["cache_file"]
        owned = np.asarray(row["owned_global_onset_indices"], dtype=np.int64)
        owned_count += len(owned)
        with np.load(path, allow_pickle=False) as cache:
            initial_count += int(row["initial_valid"])
            values = cache["main_event_target"][owned]
            masks = cache["main_timing_valid_mask"][owned]
            for slot in range(6):
                slot_values = values[:, slot]
                main[slot]["event_target_count"] += len(slot_values)
                main[slot]["non_none_count"] += int(np.sum(slot_values != 0))
                main[slot]["timing_valid_count"] += int(masks[:, slot].sum())
                main[slot]["class_counts"].update(map(int, slot_values.tolist()))
            if row["terminal_valid"]:
                terminal_window_count += 1
                values = cache["terminal_event_target"]
                masks = cache["terminal_timing_valid_mask"]
                for slot in range(4):
                    value = int(values[slot])
                    terminal[slot]["event_target_count"] += 1
                    terminal[slot]["non_none_count"] += int(value != 0)
                    terminal[slot]["timing_valid_count"] += int(masks[slot])
                    terminal[slot]["class_counts"][value] += 1
    for rows in (main, terminal):
        for row in rows:
            row["class_counts"] = {
                EVENT_NAMES[index]: int(row["class_counts"].get(index, 0))
                for index in range(5)
            }
    return {
        "performances": len({row["performance_path"] for row in selected}),
        "windows": len(selected),
        "owned_onsets": owned_count,
        "initial_target_count": initial_count,
        "terminal_supervision_window_count": terminal_window_count,
        "main_slots": {f"Slot{index + 1}": row for index, row in enumerate(main)},
        "terminal_slots": {
            f"Slot{index + 1}": row for index, row in enumerate(terminal)
        },
    }


def main() -> None:
    root, _, manifest = load_cache_manifest(CACHE_ROOT)
    train_entries = [entry for entry in manifest["entries"] if entry["split"] == "train"]
    required, desired = requirements()
    covered: set[str] = set()
    selected: list[dict] = []
    for manifest_index, entry in enumerate(train_entries):
        path = root / entry["cache_file"]
        with np.load(path, allow_pickle=False) as cache:
            notes = int(cache["non_pedal_note_count"])
            first = cache["onset_first_note_index"].astype(np.int64, copy=True)
            last = cache["onset_last_note_index"].astype(np.int64, copy=True)
            representative = cache["onset_representative_note_index"].astype(
                np.int64, copy=True
            )
            main_targets = cache["main_event_target"].astype(np.int64, copy=True)
            terminal_targets = cache["terminal_event_target"].astype(np.int64, copy=True)
        ownership = assign_unique_owners(notes, first, last, representative)
        for window_index, owned in enumerate(ownership.owned_onset_indices):
            coverage = coverage_for_window(
                window_index,
                ownership.owner_window_index,
                owned,
                main_targets,
                terminal_targets,
            )
            new = sorted((coverage & desired) - covered)
            if not new:
                continue
            selected.append(
                {
                    "selection_order": len(selected),
                    "manifest_entry_index": manifest_index,
                    "performance_index": entry["performance_index"],
                    "performance_path": entry["performance_path"],
                    "piece_id": entry["piece_id"],
                    "cache_file": entry["cache_file"],
                    "window_index": window_index,
                    "window_start_note": ownership.window_starts[window_index],
                    "window_end_note": ownership.window_ends[window_index],
                    "owned_onset_count": len(owned),
                    "owned_global_onset_indices": owned.tolist(),
                    "initial_valid": window_index == int(ownership.owner_window_index[0]),
                    "terminal_valid": window_index == int(ownership.owner_window_index[-1]),
                    "newly_covered_requirements": new,
                    "all_window_coverage": sorted(coverage),
                }
            )
            covered.update(coverage)
            if len(selected) > MAX_WINDOWS:
                raise RuntimeError("required tiny coverage exceeded 16 windows")
            if desired <= covered:
                break
        if desired <= covered:
            break
    missing_required = sorted(required - covered)
    if missing_required:
        raise RuntimeError(f"tiny subset missing required paths: {missing_required}")
    summary = summarize(selected)
    for slot in range(1, 7):
        if summary["main_slots"][f"Slot{slot}"]["non_none_count"] < 1:
            raise AssertionError(f"Main Slot{slot} lacks non-NONE coverage")
    for slot in range(1, 5):
        if summary["terminal_slots"][f"Slot{slot}"]["non_none_count"] < 1:
            raise AssertionError(f"Terminal Slot{slot} lacks non-NONE coverage")
    signature = hashlib.sha256(
        json.dumps(selected, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    payload = {
        "cache_id": CANONICAL_CACHE_ID,
        "split": "train",
        "selection_rule": "stable train manifest/window scan; add only windows covering an unmet required or desired branch/class; stop when all desired are covered",
        "max_windows": MAX_WINDOWS,
        "required_coverage": sorted(required),
        "desired_coverage": sorted(desired),
        "covered": sorted(covered),
        "missing_desired": sorted(desired - covered),
        "selected": selected,
        "summary": summary,
        "selection_sha256": signature,
        "seed": 42,
        "validation_access_count": 0,
        "asap_test_access_count": 0,
    }
    atomic_json(OUTPUT_ROOT / "tiny_subset_manifest.json", payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
