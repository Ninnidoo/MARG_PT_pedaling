#!/usr/bin/env python3
"""Traverse every canonical window through alignment/targets/collation, no model."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from src.stage2_event_model.dataset import (
    CANONICAL_ALIGNMENT_ID,
    CustomEventWindowDataset,
    custom_event_collate_fn,
)

PROJECT = Path("/workspace/project")
CACHE_ROOT = PROJECT / "analysis/custom_event_tokenizer_v1"
OUTPUT = PROJECT / "analysis/custom_event_model_v0_note_alignment/full_loader_dry_run.json"
TINY = PROJECT / "analysis/custom_event_model_v0_tiny_overfit/tiny_subset_manifest.json"
EXPECTED_TINY_SHA = "0f03efd352df6f2c8ea43950d0a9dda63a6dd2e60471aebad2da61d5d5859144"


def main() -> None:
    started = time.time()
    result = {
        "note_alignment_id": CANONICAL_ALIGNMENT_ID,
        "asap_test_access_count": 0,
        "training_steps": 0,
        "optimizer_steps": 0,
        "splits": {},
    }
    datasets = {}
    for split, expected_windows in (("train", 35573), ("validation", 1078)):
        dataset = CustomEventWindowDataset(
            CACHE_ROOT, split, cache_input_tokens=True
        )
        datasets[split] = dataset
        failures = []
        owned = 0
        initial = 0
        terminal = 0
        pending = []
        for dataset_index in range(len(dataset)):
            try:
                sample = dataset[dataset_index]
                pending.append(sample)
                owned += len(sample["owned_global_onset_indices"])
                initial += int(sample["initial_valid_mask"])
                terminal += int(sample["terminal_valid_mask"])
                performance_index, window_index = dataset.windows[dataset_index]
                performance = dataset.performances[performance_index]
                start = performance.ownership.window_starts[window_index]
                with np.load(performance.cache_path, allow_pickle=False) as cache:
                    owned_global = sample["owned_global_onset_indices"].numpy()
                    cache_rep = cache["onset_representative_note_index"][owned_global]
                expected_local = performance.alignment.cache_to_pt[cache_rep] - start
                if not np.array_equal(
                    expected_local, sample["owned_representative_positions"].numpy()
                ):
                    raise AssertionError("mapped representative/hidden lookup mismatch")
                if len(pending) == 8 or dataset_index + 1 == len(dataset):
                    batch = custom_event_collate_fn(pending)
                    if int(batch["owned_onset_mask"].sum()) != sum(
                        len(item["owned_global_onset_indices"]) for item in pending
                    ):
                        raise AssertionError("collation duplicated or dropped owned targets")
                    pending.clear()
            except Exception as error:  # collect context, then fail after writing artifact
                failures.append({"dataset_index": dataset_index, "error": repr(error)})
                break
            if (dataset_index + 1) % 5000 == 0:
                print(f"{split} dry-run {dataset_index+1}/{len(dataset)}", flush=True)
        result["splits"][split] = {
            "performances": len(dataset.performances),
            "windows_expected": expected_windows,
            "windows_traversed": 0 if failures else len(dataset),
            "owned_onsets": owned,
            "initial_supervision": initial,
            "terminal_supervision": terminal,
            "loader_alignment_failures": failures,
        }
        if failures or len(dataset) != expected_windows:
            OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            raise RuntimeError(f"{split} loader dry-run failed")

    tiny = json.loads(TINY.read_text(encoding="utf-8"))
    if tiny["selection_sha256"] != EXPECTED_TINY_SHA:
        raise RuntimeError("frozen tiny subset selection SHA changed")
    train = datasets["train"]
    tiny_checked = 0
    for selected in tiny["selected"]:
        key = (int(selected["performance_index"]), int(selected["window_index"]))
        dataset_index = train.windows.index(key)
        sample = train[dataset_index]
        if sample["metadata"]["note_alignment_id"] != CANONICAL_ALIGNMENT_ID:
            raise AssertionError("tiny subset did not use canonical alignment")
        tiny_checked += 1
    result["tiny_subset_regression"] = {
        "selection_sha256": EXPECTED_TINY_SHA,
        "windows_checked": tiny_checked,
        "target_and_representative_alignment": "PASS",
    }
    result["first_failure_window_reload"] = {
        "dataset_index": 31168,
        "performance_index": int(train.windows[31168][0]),
        "window_index": int(train.windows[31168][1]),
        "no_order_assertion_failure": True,
        "mapped_representative_identity": "PASS",
    }
    result["elapsed_seconds"] = time.time() - started
    result["status"] = "PASS"
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
