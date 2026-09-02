#!/usr/bin/env python3
"""Structural oracle and train-only smoke for frozen Prediction Decoder v1."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.stage2_event_model.dataset import (
    CANONICAL_ALIGNMENT_ID,
    CANONICAL_CACHE_ID,
    CustomEventWindowDataset,
    custom_event_collate_fn,
)
from src.stage2_event_model.model import CustomEventEncoderModelV0
from src.stage2_event_model.prediction_decoder_v1 import (
    DECODER_CONFIG,
    DECODER_ID,
    DECODER_VERSION,
    MainIntervalBoundary,
    PerformanceTimeline,
    decode_predictions_v1,
)
from src.stage2_event_tokenizer.tokenizer_v1 import _tick_second_converters


ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = ROOT / "analysis/custom_event_tokenizer_v1"
CHECKPOINT = ROOT / "analysis/custom_event_model_v0_full_seed42_aligned_statusfix_v1/best.pt"
OUTPUT = ROOT / "analysis/custom_event_prediction_decoder_v1"
PRETRAINED = ROOT / "checkpoints/pianist_transformer"
EXPECTED_ALIGNMENT_CONFIG_SHA = "77df6faf08025794e40c58d0db31cf721f582b3168d9fb201557c0500807502b"
EXPECTED_WEIGHT_SHA = "26da9ede3373bd3df5c3f28c259e29f173a0920e080eb3a27cdbc5c790ed5b55"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def target_logits(targets: np.ndarray, classes: int) -> np.ndarray:
    targets = np.asarray(targets, dtype=np.int64)
    result = np.full((*targets.shape, classes), -20.0, dtype=np.float32)
    np.put_along_axis(result, targets[..., None], 20.0, axis=-1)
    return result


def timeline_from_cache(cache: Any) -> PerformanceTimeline:
    tick_to_seconds, _ = _tick_second_converters(str(cache["source_midi"].item()))
    left = cache["main_interval_left_tick"].astype(np.int64)
    right = cache["main_interval_right_tick"].astype(np.int64)
    final = cache["main_is_final_interval"].astype(bool)
    intervals = tuple(
        MainIntervalBoundary(
            index,
            tick_to_seconds(int(left[index])),
            tick_to_seconds(int(right[index])),
            bool(final[index]),
        )
        for index in range(len(left))
    )
    return PerformanceTimeline(
        first_onset_time=intervals[0].left_time,
        latest_note_off_time=intervals[-1].right_time,
        main_intervals=intervals,
    )


def prefix_valid(values: np.ndarray) -> bool:
    for row in np.atleast_2d(values):
        zeros = np.flatnonzero(row == 0)
        if len(zeros) and bool(np.any(row[int(zeros[0]) + 1:] != 0)):
            return False
    return True


def expected_final_state(initial: int, main: np.ndarray, terminal: np.ndarray) -> int:
    state = int(initial)
    for row in main:
        for event_class in row:
            if int(event_class) == 0:
                break
            state = int(event_class) - 1
    for event_class in terminal:
        if int(event_class) == 0:
            break
        state = int(event_class) - 1
    return state


def oracle_roundtrip() -> dict[str, Any]:
    manifest = json.loads((CACHE_ROOT / "cache_manifest.json").read_text(encoding="utf-8"))
    if manifest["cache_id"] != CANONICAL_CACHE_ID:
        raise RuntimeError("frozen tokenizer cache ID mismatch")
    totals: dict[str, Counter] = {"train": Counter(), "validation": Counter()}
    failures: list[dict[str, Any]] = []
    for entry in manifest["entries"]:
        split = entry["split"]
        if split not in totals:
            raise RuntimeError(f"forbidden split in canonical cache: {split}")
        cache_path = CACHE_ROOT / entry["cache_file"]
        with np.load(cache_path, allow_pickle=False) as cache:
            if str(cache["cache_id"].item()) != CANONICAL_CACHE_ID:
                raise RuntimeError(f"per-performance cache ID mismatch: {cache_path}")
            initial = int(cache["initial_state"])
            main = cache["main_event_target"].astype(np.int64)
            taus = cache["main_tau_target"].astype(np.float64)
            main_mask = cache["main_timing_valid_mask"].astype(bool)
            terminal = cache["terminal_event_target"].astype(np.int64)
            terminal_z = cache["terminal_log1p_gap_target"].astype(np.float64)
            terminal_mask = cache["terminal_timing_valid_mask"].astype(bool)
            main_prefix = prefix_valid(main)
            terminal_prefix = prefix_valid(terminal[None])
            chronological = all(
                bool(np.all(np.diff(taus[index][main_mask[index]]) >= 0.0))
                for index in range(len(main))
            )
            nonnegative_terminal = bool(np.all(terminal_z[terminal_mask] >= 0.0))
            decoded = decode_predictions_v1(
                initial_logits=target_logits(np.asarray(initial), 4),
                main_event_logits=target_logits(main, 5),
                main_timing_predictions=taus,
                terminal_event_logits=target_logits(terminal, 5),
                terminal_timing_predictions=terminal_z,
                timeline=timeline_from_cache(cache),
            )
            expected = expected_final_state(initial, main, terminal)
            okay = (
                main_prefix and terminal_prefix and chronological and nonnegative_terminal
                and decoded.final_pedal_state == expected
                and bool(cache["main_final_state_preservation"])
                and bool(cache["terminal_final_state_preservation"])
            )
            if not okay:
                failures.append({
                    "split": split,
                    "performance_index": int(cache["performance_index"]),
                    "performance_id": str(cache["performance_id"].item()),
                    "main_prefix": main_prefix,
                    "terminal_prefix": terminal_prefix,
                    "main_tau_chronological": chronological,
                    "terminal_gap_nonnegative": nonnegative_terminal,
                    "expected_final_state": expected,
                    "decoded_final_state": decoded.final_pedal_state,
                })
            stats = totals[split]
            stats["performances"] += 1
            stats["main_intervals"] += len(main)
            stats["main_active_target_set_slots"] += int((main != 0).sum())
            stats["terminal_active_target_set_slots"] += int((terminal != 0).sum())
            stats["effective_state_changing_events"] += len(decoded.emitted_events)
            stats["same_state_suppressed"] += (
                decoded.diagnostics["same_state_suppressed_main"]
                + decoded.diagnostics["same_state_suppressed_terminal"]
            )
            stats["same_time_collapsed"] += decoded.diagnostics["main_same_time_collapsed"]
            stats["prefix_failures"] += int(not main_prefix) + int(not terminal_prefix)
            stats["chronology_failures"] += int(not chronological)
            stats["negative_terminal_gap_failures"] += int(not nonnegative_terminal)
            stats["final_state_mismatches"] += int(decoded.final_pedal_state != expected)
    if failures:
        raise AssertionError(f"decoder oracle failures: {failures[:3]}")
    result = {
        "decoder_version": DECODER_VERSION,
        "decoder_id": DECODER_ID,
        "tokenizer_cache_id": CANONICAL_CACHE_ID,
        "asap_validation_metric_access_count": 0,
        "asap_test_access_count": 0,
        "repedal_execution_count": 0,
        "splits": {split: dict(counter) for split, counter in totals.items()},
        "all_ground_truth_prefix_valid": True,
        "all_main_active_tau_chronological": True,
        "all_terminal_gaps_nonnegative": True,
        "final_state_preservation_exact": True,
        "failures": [],
    }
    atomic_json(OUTPUT / "oracle_roundtrip_stats.json", result)
    return result


def load_checkpoint_safely() -> dict[str, Any]:
    from numpy.core.multiarray import _reconstruct

    safe = [_reconstruct, np.ndarray, np.dtype, type(np.dtype(np.uint32)), type(np.dtype(np.float64))]
    with torch.serialization.safe_globals(safe):
        checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    expected = {
        "epoch": 1,
        "best_epoch": 1,
        "tokenizer_cache_id": CANONICAL_CACHE_ID,
        "note_alignment_id": CANONICAL_ALIGNMENT_ID,
        "note_alignment_config_sha256": EXPECTED_ALIGNMENT_CONFIG_SHA,
        "event_weight_sha256": EXPECTED_WEIGHT_SHA,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise RuntimeError(f"canonical checkpoint {key} mismatch")
    return checkpoint


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


def train_smoke() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("canonical best-checkpoint smoke requires CUDA")
    dataset = CustomEventWindowDataset(CACHE_ROOT, "train", entry_limit=16, cache_input_tokens=True)
    # Deterministic, not performance-selected: shortest two among the stable first 16 train entries.
    selected = sorted(
        range(len(dataset.performances)),
        key=lambda index: (len(dataset.performances[index].ownership.window_starts), index),
    )[:2]
    checkpoint = load_checkpoint_safely()
    model = CustomEventEncoderModelV0.from_pretrained(
        PRETRAINED, head_init_seed=42, torch_dtype=torch.float32, attn_implementation="eager"
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    del checkpoint
    device = torch.device("cuda:0")
    model.to(device).eval()
    rows = []
    with torch.no_grad():
        for performance_index in selected:
            performance = dataset.performances[performance_index]
            onset_count = performance.num_onsets
            main_logits = np.empty((onset_count, 6, 5), dtype=np.float32)
            main_timing = np.empty((onset_count, 6), dtype=np.float32)
            seen = np.zeros(onset_count, dtype=bool)
            initial_logits = None
            terminal_logits = None
            terminal_timing = None
            window_count = 0
            for dataset_index, (owner_performance, _) in enumerate(dataset.windows):
                if owner_performance != performance_index:
                    continue
                sample = dataset[dataset_index]
                batch = move_batch(custom_event_collate_fn([sample]), device)
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
                owned = sample["owned_global_onset_indices"].numpy()
                count = len(owned)
                main_logits[owned] = output.main_event_logits[0, :count].float().cpu().numpy()
                main_timing[owned] = output.main_timing_predictions[0, :count].float().cpu().numpy()
                if bool(sample["initial_valid_mask"]):
                    initial_logits = output.initial_logits[0].float().cpu().numpy()
                if bool(sample["terminal_valid_mask"]):
                    terminal_logits = output.terminal_event_logits[0].float().cpu().numpy()
                    terminal_timing = output.terminal_timing_predictions[0].float().cpu().numpy()
                seen[owned] = True
                window_count += 1
            if not bool(seen.all()) or initial_logits is None or terminal_logits is None or terminal_timing is None:
                raise AssertionError("train smoke did not aggregate exactly one prediction per target")
            with np.load(performance.cache_path, allow_pickle=False) as cache:
                decoded = decode_predictions_v1(
                    initial_logits=initial_logits,
                    main_event_logits=main_logits,
                    main_timing_predictions=main_timing,
                    terminal_event_logits=terminal_logits,
                    terminal_timing_predictions=terminal_timing,
                    timeline=timeline_from_cache(cache),
                )
                repeated = decode_predictions_v1(
                    initial_logits=initial_logits,
                    main_event_logits=main_logits,
                    main_timing_predictions=main_timing,
                    terminal_event_logits=terminal_logits,
                    terminal_timing_predictions=terminal_timing,
                    timeline=timeline_from_cache(cache),
                )
                if decoded.to_dict() != repeated.to_dict():
                    raise AssertionError("prediction decoder is not deterministic")
                if not all(math.isfinite(item.time) for item in decoded.events):
                    raise AssertionError("decoded train smoke time is non-finite")
                rows.append({
                    "performance_index": performance_index,
                    "performance_id": str(cache["performance_id"].item()),
                    "source_dataset": str(cache["source_dataset"].item()),
                    "windows": window_count,
                    "modeled_onsets": onset_count,
                    "decoded_events_including_initial": len(decoded.events),
                    "final_pedal_state": decoded.final_pedal_state,
                    "diagnostics": dict(decoded.diagnostics),
                    "deterministic_exact": True,
                    "all_outputs_finite": True,
                })
    del model
    torch.cuda.empty_cache()
    aggregate = Counter()
    for row in rows:
        aggregate.update(row["diagnostics"])
    result = {
        "decoder_version": DECODER_VERSION,
        "decoder_id": DECODER_ID,
        "checkpoint": str(CHECKPOINT),
        "checkpoint_best_epoch": 1,
        "selection_rule": "shortest two performances among stable first 16 canonical train entries",
        "performances": rows,
        "aggregate_diagnostics": dict(aggregate),
        "all_predictions_finite": True,
        "all_decodes_deterministic": True,
        "asap_validation_inference_count": 0,
        "asap_validation_metric_access_count": 0,
        "asap_test_access_count": 0,
        "repedal_execution_count": 0,
    }
    atomic_json(OUTPUT / "train_smoke_diagnostics.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("oracle", "smoke", "all"), default="all", nargs="?")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUTPUT / "decoder_config.json", {
        **DECODER_CONFIG,
        "decoder_id": DECODER_ID,
        "tokenizer_cache_id": CANONICAL_CACHE_ID,
        "note_alignment_id": CANONICAL_ALIGNMENT_ID,
        "canonical_checkpoint": str(CHECKPOINT),
        "canonical_checkpoint_best_epoch": 1,
        "asap_validation_metric_access_count": 0,
        "asap_test_access_count": 0,
        "repedal_execution_count": 0,
    })
    result: dict[str, Any] = {}
    if args.mode in {"oracle", "all"}:
        result["oracle"] = oracle_roundtrip()
    if args.mode in {"smoke", "all"}:
        result["train_smoke"] = train_smoke()
    atomic_json(OUTPUT / "decoder_verification.json", {
        "decoder_version": DECODER_VERSION,
        "decoder_id": DECODER_ID,
        "status": "passed",
        **result,
    })
    print(json.dumps({"status": "passed", "decoder_id": DECODER_ID,
                      "completed": sorted(result)}, sort_keys=True))


if __name__ == "__main__":
    main()
