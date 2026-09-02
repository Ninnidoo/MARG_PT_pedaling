#!/usr/bin/env python3
"""Strict structural validation for the canonical tokenizer v1 cache."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "analysis/custom_event_tokenizer_v1"


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.root
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "cache_manifest.json").read_text(encoding="utf-8"))
    assert config["asap_test_access_count"] == 0
    assert config["training_steps"] == config["model_steps"] == 0
    assert manifest["cache_id"] == config["cache_id"]
    assert manifest["train_count"] == 2062
    assert manifest["validation_count"] == 71
    assert len(manifest["entries"]) == 2133

    split_counts = {"train": 0, "validation": 0}
    main_intervals = 0
    terminal_sets = 0
    for entry in manifest["entries"]:
        split = entry["split"]
        split_counts[split] += 1
        path = root / entry["cache_file"]
        assert path.is_file()
        with np.load(path, allow_pickle=False) as cache:
            assert str(cache["cache_id"].item()) == config["cache_id"]
            main_event = cache["main_event_target"]
            main_tau = cache["main_tau_target"]
            main_mask = cache["main_timing_valid_mask"]
            terminal_event = cache["terminal_event_target"]
            terminal_gap = cache["terminal_log1p_gap_target"]
            terminal_mask = cache["terminal_timing_valid_mask"]
            assert main_event.ndim == 2 and main_event.shape[1] == 6
            assert main_tau.shape == main_event.shape == main_mask.shape
            assert terminal_event.shape == terminal_gap.shape == terminal_mask.shape == (4,)
            assert np.array_equal(main_mask, main_event != 0)
            assert np.array_equal(terminal_mask, terminal_event != 0)
            assert np.all(main_tau[~main_mask] == 0)
            assert np.all(terminal_gap[~terminal_mask] == 0)
            assert np.all(np.isfinite(main_tau)) and np.all(np.isfinite(terminal_gap))
            assert np.all((main_tau[main_mask] >= 0) & (main_tau[main_mask] <= 1))
            assert np.all(terminal_gap[terminal_mask] >= 0)
            final_flags = cache["main_is_final_interval"]
            assert final_flags.shape == (len(main_event),)
            assert int(final_flags.sum()) == 1 and bool(final_flags[-1])
            assert int(cache["main_interval_right_tick"][-1]) == int(cache["latest_note_off_tick"])
            assert np.all(cache["onset_representative_note_index"] == cache["onset_last_note_index"])
            assert bool(cache["main_final_state_preservation"])
            assert bool(cache["terminal_final_state_preservation"])
            main_intervals += len(main_event)
            terminal_sets += 1
    assert split_counts == {"train": 2062, "validation": 71}
    payload = {
        "status": "passed",
        "cache_id": config["cache_id"],
        "cache_files_checked": 2133,
        "split_counts": split_counts,
        "main_intervals_checked": main_intervals,
        "terminal_target_sets_checked": terminal_sets,
        "mandatory_masks_exact": True,
        "finite_placeholders_verified": True,
        "final_main_interval_verified": True,
        "representative_note_index_is_last": True,
        "main_final_state_preservation": 1.0,
        "terminal_final_state_preservation": 1.0,
        "asap_test_access_count": 0,
        "model_steps": 0,
        "training_steps": 0,
    }
    atomic_json(root / "cache_verification.json", payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
