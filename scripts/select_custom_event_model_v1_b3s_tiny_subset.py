#!/usr/bin/env python3
"""Deterministic train-only transition/branch coverage selection for B3-S."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage2_event_model.dataset import CANONICAL_ALIGNMENT_ID, CANONICAL_CACHE_ID
from src.stage2_event_model.dataset_v1_state_conditioned import (
    CustomEventStateConditionedWindowDataset,
    derive_main_pre_state_targets,
)
from src.stage2_event_tokenizer.tokenizer import EVENT_NAMES


CACHE_ROOT = ROOT / "analysis/custom_event_tokenizer_v1"
OUTPUT_ROOT = ROOT / "analysis/custom_event_model_v1_state_conditioned_impl"
STATE_NAMES = ("ZERO", "LOW", "HALF", "FULL")
SEED = 42
MAX_WINDOWS = 16


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def requirements() -> set[str]:
    required = {f"main_slot_{slot}_non_none" for slot in range(1, 7)}
    required |= {f"terminal_slot_{slot}_non_none" for slot in range(1, 5)}
    required |= {f"initial_state_{name}" for name in STATE_NAMES}
    required |= {
        "transition_ZERO_LOW",
        "transition_LOW_ZERO",
        "transition_HALF_FULL",
        "transition_FULL_HALF",
        "transition_OFF_ON",
        "transition_ON_OFF",
        "transition_ZERO_ON",
        "transition_LOW_ON",
        "transition_ON_ZERO",
        "transition_ON_LOW",
        "same_state_main_set",
    }
    return required


def transition_labels(pre: int, destination: int) -> set[str]:
    labels = {f"pair_{STATE_NAMES[pre]}_{STATE_NAMES[destination]}"}
    if (pre, destination) == (0, 1):
        labels.add("transition_ZERO_LOW")
    if (pre, destination) == (1, 0):
        labels.add("transition_LOW_ZERO")
    if (pre, destination) == (2, 3):
        labels.add("transition_HALF_FULL")
    if (pre, destination) == (3, 2):
        labels.add("transition_FULL_HALF")
    if pre < 2 <= destination:
        labels.add("transition_OFF_ON")
        labels.add("transition_ZERO_ON" if pre == 0 else "transition_LOW_ON")
    if destination < 2 <= pre:
        labels.add("transition_ON_OFF")
        labels.add("transition_ON_ZERO" if destination == 0 else "transition_ON_LOW")
    if pre == destination:
        labels.add("same_state_main_set")
    return labels


def main() -> None:
    dataset = CustomEventStateConditionedWindowDataset(
        CACHE_ROOT, "train", cache_input_tokens=False
    )
    if len(dataset.performances) != 2062 or len(dataset) != 35573:
        raise RuntimeError("canonical train ownership universe changed")
    required = requirements()
    candidates: list[dict[str, Any]] = []
    oracle_examples: dict[str, dict[str, Any]] = {}
    for performance_index, performance in enumerate(dataset.performances):
        with np.load(performance.cache_path, allow_pickle=False) as cache:
            events = cache["main_event_target"].astype(np.int8, copy=True)
            pre = derive_main_pre_state_targets(
                int(cache["initial_state"]),
                events,
                cache["main_interval_start_state"],
            )
            initial_state = int(cache["initial_state"])
            terminal = cache["terminal_event_target"].astype(np.int8, copy=True)
        ownership = performance.ownership
        first_owner = int(ownership.owner_window_index[0])
        last_owner = int(ownership.owner_window_index[-1])
        for window_index, owned in enumerate(ownership.owned_onset_indices):
            coverage: set[str] = set()
            selected_events = events[owned]
            selected_pre = pre[owned]
            for slot in range(6):
                active = selected_events[:, slot] > 0
                if bool(active.any()):
                    coverage.add(f"main_slot_{slot + 1}_non_none")
                for local in np.flatnonzero(active):
                    source = int(selected_pre[local, slot])
                    destination = int(selected_events[local, slot]) - 1
                    labels = transition_labels(source, destination)
                    coverage.update(labels)
                    for label in labels & required:
                        oracle_examples.setdefault(
                            label,
                            {
                                "performance_index": performance_index,
                                "performance_path": performance.entry["performance_path"],
                                "window_index": window_index,
                                "global_onset_index": int(owned[local]),
                                "slot": slot + 1,
                                "pre_state": STATE_NAMES[source],
                                "destination": STATE_NAMES[destination],
                                "verified_immediately_before_target": True,
                            },
                        )
            if window_index == first_owner:
                coverage.add(f"initial_state_{STATE_NAMES[initial_state]}")
            if window_index == last_owner:
                for slot, event in enumerate(terminal, 1):
                    if int(event) != 0:
                        coverage.add(f"terminal_slot_{slot}_non_none")
            relevant = coverage & required
            if relevant:
                candidates.append(
                    {
                        "performance_index": performance_index,
                        "performance_path": performance.entry["performance_path"],
                        "piece_id": performance.entry["piece_id"],
                        "cache_file": performance.entry["cache_file"],
                        "alignment_file": str(performance.alignment_path.relative_to(ROOT)),
                        "window_index": window_index,
                        "window_start_note": int(ownership.window_starts[window_index]),
                        "window_end_note": int(ownership.window_ends[window_index]),
                        "owned_onset_count": int(len(owned)),
                        "owned_global_onset_indices": owned.tolist(),
                        "initial_valid": window_index == first_owner,
                        "terminal_valid": window_index == last_owner,
                        "coverage": sorted(coverage),
                    }
                )

    selected: list[dict[str, Any]] = []
    uncovered = set(required)
    available = list(candidates)
    while uncovered:
        ranked = sorted(
            available,
            key=lambda row: (
                -len(set(row["coverage"]) & uncovered),
                int(row["owned_onset_count"]),
                int(row["performance_index"]),
                int(row["window_index"]),
            ),
        )
        if not ranked or not (set(ranked[0]["coverage"]) & uncovered):
            raise RuntimeError(f"uncovered B3-S tiny requirements: {sorted(uncovered)}")
        chosen = ranked[0]
        new = sorted(set(chosen["coverage"]) & uncovered)
        selected.append({**chosen, "selection_order": len(selected), "newly_covered": new})
        uncovered.difference_update(new)
        available.remove(chosen)
        if len(selected) > MAX_WINDOWS:
            raise RuntimeError("B3-S coverage requires more than 16 windows")

    transition_matrix = np.zeros((4, 4), dtype=np.int64)
    state_counts = np.zeros(4, dtype=np.int64)
    per_slot_active = np.zeros(6, dtype=np.int64)
    terminal_active = np.zeros(4, dtype=np.int64)
    initial_counts = Counter()
    same_state = 0
    owned_total = 0
    for row in selected:
        owned = np.asarray(row["owned_global_onset_indices"], dtype=np.int64)
        owned_total += len(owned)
        with np.load(CACHE_ROOT / row["cache_file"], allow_pickle=False) as cache:
            events = cache["main_event_target"][owned].astype(np.int8)
            full_pre = derive_main_pre_state_targets(
                int(cache["initial_state"]),
                cache["main_event_target"],
                cache["main_interval_start_state"],
            )
            pre = full_pre[owned]
            if row["initial_valid"]:
                initial_counts[STATE_NAMES[int(cache["initial_state"])]] += 1
            if row["terminal_valid"]:
                terminal_active += (cache["terminal_event_target"] != 0).astype(np.int64)
        for slot in range(6):
            state_counts += np.bincount(pre[:, slot], minlength=4)
            active = events[:, slot] > 0
            per_slot_active[slot] += int(active.sum())
            np.add.at(transition_matrix, (pre[active, slot], events[active, slot] - 1), 1)
        same_state += int(np.trace(transition_matrix)) - same_state

    selected_core = [
        {
            key: row[key]
            for key in (
                "performance_index", "performance_path", "window_index",
                "window_start_note", "window_end_note", "owned_global_onset_indices",
            )
        }
        for row in selected
    ]
    signature = hashlib.sha256(
        json.dumps(selected_core, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    stats = {
        "performances": len({row["performance_index"] for row in selected}),
        "windows": len(selected),
        "owned_onsets": owned_total,
        "per_slot_active_counts": {
            f"Slot{slot + 1}": int(value) for slot, value in enumerate(per_slot_active)
        },
        "pre_state_distribution_all_six_slots": {
            STATE_NAMES[index]: int(value) for index, value in enumerate(state_counts)
        },
        "destination_transition_matrix": transition_matrix.tolist(),
        "transition_axes": list(STATE_NAMES),
        "same_state_main_target_count": int(np.trace(transition_matrix)),
        "initial_state_supervision_counts": dict(initial_counts),
        "terminal_slot_active_counts": {
            f"Slot{slot + 1}": int(value) for slot, value in enumerate(terminal_active)
        },
    }
    payload = {
        "cache_id": CANONICAL_CACHE_ID,
        "alignment_id": CANONICAL_ALIGNMENT_ID,
        "split": "train",
        "seed": SEED,
        "selection_rule": "global deterministic greedy set cover; maximize newly covered frozen requirements, then fewer owned onsets, then lower performance/window index",
        "required_coverage": sorted(required),
        "covered": sorted(set().union(*(set(row["coverage"]) for row in selected))),
        "missing_required": [],
        "selected": selected,
        "selection_sha256": signature,
        "summary": stats,
        "oracle_transition_examples": {
            key: oracle_examples[key] for key in sorted(required) if key in oracle_examples
        },
        "validation_target_access_count": 0,
        "validation_model_inference_count": 0,
        "asap_test_access_count": 0,
        "repedal_execution_count": 0,
    }
    atomic_json(OUTPUT_ROOT / "tiny_overfit_subset_manifest.json", payload)
    atomic_json(OUTPUT_ROOT / "tiny_overfit_subset_stats.json", stats)
    print(
        f"B3S_SUBSET_SELECTED windows={len(selected)} performances={stats['performances']} "
        f"owned={owned_total} sha256={signature}"
    )


if __name__ == "__main__":
    main()
