#!/usr/bin/env python3
"""Frozen Custom Event Model v0 canonical ASAP-validation inference/evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mido
import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.run_stage2_4class_validation_eval_v0 import (
    ASAP_ROOT,
    STAGE1_MANIFEST,
    SPLIT_CSV,
    align_and_cache,
    finalize_transition,
    sum_transition,
    top_patterns,
)
from src.stage2_binary.canonical_stage1 import (
    assert_strict_non_cc64_equality,
    cc64_schedule,
    sha256_file,
)
from src.stage2_event_model.dataset import (
    CANONICAL_ALIGNMENT_ID,
    CANONICAL_CACHE_ID,
    CustomEventWindowDataset,
    custom_event_collate_fn,
)
from src.stage2_event_model.model import CustomEventEncoderModelV0
from src.stage2_event_model.prediction_decoder_v1 import (
    DECODER_ID,
    DECODER_VERSION,
    MainIntervalBoundary,
    MainIntervalTicks,
    PerformanceTimeline,
    decode_predictions_v1,
    quantize_decoded_timeline_v1,
)
from src.stage2_event_tokenizer.tokenizer_v1 import _tick_second_converters
from src.stage2_four_class.validation_evaluator import (
    PATTERN_COUNT,
    canonical_classes_from_tokens,
    classification_metrics,
    confusion_from_pairs,
    pattern_ids,
    pattern_metrics,
    pooled_transition_counts,
)


CACHE_ROOT = ROOT / "analysis/custom_event_tokenizer_v1"
ALIGNMENT_ROOT = ROOT / "analysis/custom_event_model_v0_note_alignment"
DECODER_ROOT = ROOT / "analysis/custom_event_prediction_decoder_v1"
CHECKPOINT = ROOT / "analysis/custom_event_model_v0_full_seed42_aligned_statusfix_v1/best.pt"
PRETRAINED = ROOT / "checkpoints/pianist_transformer"
OUTPUT = ROOT / "analysis/custom_event_model_v0_canonical_val_inference_v1"
FROZEN_HUMAN_ALIGNMENT = ROOT / "analysis/stage2_4class_architecture_validation_eval_v0/alignment/human"
EXPECTED_ALIGNMENT_CONFIG_SHA = "77df6faf08025794e40c58d0db31cf721f582b3168d9fb201557c0500807502b"
EXPECTED_DECODER_ID = "938ca7e20e1d398aa530938bbcede404fd3a04af23842fbcf351ba02df0268f8"
EXPECTED_WEIGHT_SHA = "26da9ede3373bd3df5c3f28c259e29f173a0920e080eb3a27cdbc5c790ed5b55"
EXPECTED_CHECKPOINT_SHA = "a81c5f8f0a902523bd8abb1ca5b1097876887477f1668907f08e2280d7d096c8"


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def safe_checkpoint() -> dict[str, Any]:
    from numpy.core.multiarray import _reconstruct

    safe = [_reconstruct, np.ndarray, np.dtype, type(np.dtype(np.uint32)), type(np.dtype(np.float64))]
    with torch.serialization.safe_globals(safe):
        value = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    expected = {
        "epoch": 1,
        "best_epoch": 1,
        "tokenizer_cache_id": CANONICAL_CACHE_ID,
        "note_alignment_id": CANONICAL_ALIGNMENT_ID,
        "note_alignment_config_sha256": EXPECTED_ALIGNMENT_CONFIG_SHA,
        "event_weight_sha256": EXPECTED_WEIGHT_SHA,
    }
    for key, wanted in expected.items():
        if value.get(key) != wanted:
            raise RuntimeError(f"checkpoint {key} mismatch: {value.get(key)!r} != {wanted!r}")
    if sha256_file(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA:
        raise RuntimeError("canonical checkpoint SHA mismatch")
    return value


def preflight() -> dict[str, Any]:
    cache = json.loads((CACHE_ROOT / "cache_manifest.json").read_text())
    alignment = json.loads((ALIGNMENT_ROOT / "alignment_manifest.json").read_text())
    decoder = json.loads((DECODER_ROOT / "decoder_config.json").read_text())
    if cache["cache_id"] != CANONICAL_CACHE_ID or cache["validation_count"] != 71:
        raise RuntimeError("frozen tokenizer-cache identity/universe mismatch")
    if alignment["alignment_id"] != CANONICAL_ALIGNMENT_ID:
        raise RuntimeError("frozen note-alignment ID mismatch")
    if alignment["alignment_config_sha256"] != EXPECTED_ALIGNMENT_CONFIG_SHA:
        raise RuntimeError("frozen note-alignment config SHA mismatch")
    if decoder["decoder_id"] != EXPECTED_DECODER_ID or DECODER_ID != EXPECTED_DECODER_ID:
        raise RuntimeError("frozen Decoder v1 ID mismatch")
    if any(int(value.get("asap_test_access_count", 0)) != 0 for value in (cache, alignment, decoder)):
        raise RuntimeError("frozen provenance reports ASAP-test access")
    checkpoint = safe_checkpoint()
    return {
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
        "checkpoint_epoch": 1,
        "best_epoch": 1,
        "tokenizer_cache_id": CANONICAL_CACHE_ID,
        "note_alignment_id": CANONICAL_ALIGNMENT_ID,
        "alignment_config_sha256": EXPECTED_ALIGNMENT_CONFIG_SHA,
        "event_weight_sha256": EXPECTED_WEIGHT_SHA,
        "decoder_version": DECODER_VERSION,
        "decoder_id": DECODER_ID,
        "asap_validation_performances": 71,
        "asap_validation_pieces": 19,
        "asap_test_access_count": 0,
        "repedal_execution_count": 0,
        "checkpoint_payload": checkpoint,
    }


def timeline_from_cache(cache: Any) -> tuple[PerformanceTimeline, tuple[MainIntervalTicks, ...], Any]:
    source = str(cache["source_midi"].item())
    tick_to_seconds, seconds_to_tick = _tick_second_converters(source)
    left = cache["main_interval_left_tick"].astype(np.int64)
    right = cache["main_interval_right_tick"].astype(np.int64)
    final = cache["main_is_final_interval"].astype(bool)
    continuous = tuple(
        MainIntervalBoundary(index, tick_to_seconds(int(left[index])),
                             tick_to_seconds(int(right[index])), bool(final[index]))
        for index in range(len(left))
    )
    ticks = tuple(
        MainIntervalTicks(index, int(left[index]), int(right[index]), bool(final[index]))
        for index in range(len(left))
    )
    return PerformanceTimeline(continuous[0].left_time, continuous[-1].right_time, continuous), ticks, seconds_to_tick


def _absolute(track: mido.MidiTrack) -> list[tuple[int, int, Any]]:
    tick = 0
    rows = []
    for order, message in enumerate(track):
        tick += int(message.time)
        rows.append((tick, order, message))
    return rows


def noncc_except_eot_signature(path: Path) -> list[list[tuple[int, dict[str, Any]]]]:
    midi = mido.MidiFile(str(path), clip=False)
    result = []
    for track in midi.tracks:
        rows = []
        for tick, _, message in _absolute(track):
            if (
                not message.is_meta and message.type == "control_change"
                and int(message.control) == 64
            ):
                continue
            if message.is_meta and message.type == "end_of_track":
                continue
            payload = message.dict()
            payload.pop("time", None)
            rows.append((tick, payload))
        result.append(rows)
    return result


def write_candidate(source_path: Path, destination: Path, events: list[Any]) -> dict[str, Any]:
    """Replace only CC64 while retaining every non-CC64 event at its exact tick/order."""
    source = mido.MidiFile(str(source_path), clip=False)
    schedule = cc64_schedule(source_path)
    if schedule:
        preference = Counter((row["track_index"], row["channel"]) for row in schedule).most_common(1)[0][0]
    else:
        preference = (0, 0)
    target_track, target_channel = preference
    if not 0 <= target_track < len(source.tracks):
        raise AssertionError("invalid preferred CC64 track")
    by_tick: dict[int, list[Any]] = defaultdict(list)
    for event in sorted(events, key=lambda item: item.ordering_key):
        if not 0 <= int(event.cc64_value) <= 127 or int(event.tick) < 0:
            raise ValueError("decoded CC64 event escaped valid MIDI range")
        by_tick[int(event.tick)].append(
            mido.Message("control_change", channel=int(target_channel), control=64,
                         value=int(event.cc64_value), time=0)
        )
    output = mido.MidiFile(type=source.type, ticks_per_beat=source.ticks_per_beat, clip=False)
    output.tracks = []
    eot_extension_ticks = 0
    for track_index, track in enumerate(source.tracks):
        base = [(tick, order, message) for tick, order, message in _absolute(track)
                if message.is_meta or message.type != "control_change" or int(message.control) != 64]
        eot = [tick for tick, _, msg in base if msg.is_meta and msg.type == "end_of_track"]
        if len(eot) != 1:
            raise ValueError(f"source track {track_index} must have one EOT")
        inserts = by_tick if track_index == target_track else {}
        if inserts and max(inserts, default=0) > eot[0]:
            # Terminal grammar can require a CC64 strictly after source EOT.
            # Extend only EOT; all musical non-pedal events remain untouched.
            extended = max(inserts)
            eot_extension_ticks = max(eot_extension_ticks, extended - eot[0])
            base = [
                (extended if msg.is_meta and msg.type == "end_of_track" else tick, order, msg)
                for tick, order, msg in base
            ]
        grouped: dict[int, list[tuple[int, Any]]] = defaultdict(list)
        for tick, order, message in base:
            grouped[tick].append((order, message))
        rebuilt: list[tuple[int, int, Any]] = []
        for tick in sorted(set(grouped) | set(inserts)):
            # Frozen priority: Initial/Main tau=0 before note-on. Decoder ordering
            # is already deterministic within the inserted sequence.
            for insertion_order, message in enumerate(inserts.get(tick, [])):
                rebuilt.append((tick, -1000000 + insertion_order, message))
            rebuilt.extend((tick, order, message) for order, message in grouped.get(tick, []))
        new_track = mido.MidiTrack()
        previous = 0
        for tick, _, message in sorted(rebuilt, key=lambda row: (row[0], row[1])):
            new_track.append(message.copy(time=int(tick) - previous))
            previous = int(tick)
        output.tracks.append(new_track)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name("." + destination.name + ".tmp")
    output.save(str(temporary))
    os.replace(temporary, destination)
    if eot_extension_ticks == 0:
        identity = assert_strict_non_cc64_equality(source_path, destination)
        identity["eot_extension_ticks"] = 0
    else:
        if noncc_except_eot_signature(source_path) != noncc_except_eot_signature(destination):
            raise AssertionError("non-CC64 event changed while extending EOT for terminal grammar")
        identity = {
            "passed": True,
            "canonical_sha256": sha256_file(source_path),
            "candidate_sha256": sha256_file(destination),
            "midi_type_exact": True,
            "ticks_per_beat_exact": True,
            "track_count_exact": True,
            "all_ordered_non_cc64_events_exact_except_eot_tick": True,
            "eot_extension_ticks": eot_extension_ticks,
            "canonical_cc64_events": len(cc64_schedule(source_path)),
            "candidate_cc64_events": len(cc64_schedule(destination)),
            "cc64_changed": cc64_schedule(source_path) != cc64_schedule(destination),
        }
    identity.update({"preferred_track": target_track, "preferred_channel": target_channel})
    return identity


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def infer_and_render(provenance: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("canonical inference requires CUDA")
    dataset = CustomEventWindowDataset(CACHE_ROOT, "validation", cache_input_tokens=True)
    if len(dataset) != 1078 or sum(item.num_onsets for item in dataset.performances) != 272927:
        raise RuntimeError("canonical validation window/onset universe mismatch")
    checkpoint = provenance.pop("checkpoint_payload")
    model = CustomEventEncoderModelV0.from_pretrained(
        PRETRAINED, head_init_seed=42, torch_dtype=torch.float32, attn_implementation="eager"
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    del checkpoint
    if sum(parameter.numel() for parameter in model.parameters()) != 103320640:
        raise RuntimeError("canonical model parameter count mismatch")
    device = torch.device("cuda:0")
    model.to(device).eval()
    predictions: list[dict[str, Any]] = []
    for performance in dataset.performances:
        predictions.append({
            "main_event_logits": np.empty((performance.num_onsets, 6, 5), np.float32),
            "main_timing": np.empty((performance.num_onsets, 6), np.float32),
            "seen": np.zeros(performance.num_onsets, dtype=np.uint8),
            "initial_logits": None,
            "terminal_event_logits": None,
            "terminal_timing": None,
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
            if not all(bool(torch.isfinite(value).all()) for value in tensors):
                raise FloatingPointError("non-finite canonical inference output")
            for batch_index, metadata in enumerate(batch["metadata"]):
                performance_index = int(metadata["performance_index"])
                record = predictions[performance_index]
                mask = batch["owned_onset_mask"][batch_index]
                count = int(mask.sum())
                owned = batch["owned_global_onset_indices"][batch_index, :count].cpu().numpy()
                if bool(record["seen"][owned].any()):
                    raise AssertionError("duplicate unique-owner inference prediction")
                record["main_event_logits"][owned] = output.main_event_logits[batch_index, :count].float().cpu().numpy()
                record["main_timing"][owned] = output.main_timing_predictions[batch_index, :count].float().cpu().numpy()
                record["seen"][owned] = 1
                owned_total += count
                if bool(batch["initial_valid_mask"][batch_index]):
                    if record["initial_logits"] is not None:
                        raise AssertionError("duplicate initial prediction")
                    record["initial_logits"] = output.initial_logits[batch_index].float().cpu().numpy()
                    initial_total += 1
                if bool(batch["terminal_valid_mask"][batch_index]):
                    if record["terminal_event_logits"] is not None:
                        raise AssertionError("duplicate terminal prediction")
                    record["terminal_event_logits"] = output.terminal_event_logits[batch_index].float().cpu().numpy()
                    record["terminal_timing"] = output.terminal_timing_predictions[batch_index].float().cpu().numpy()
                    terminal_total += 1
                windows += 1
    if (windows, owned_total, initial_total, terminal_total) != (1078, 272927, 71, 71):
        raise AssertionError("unique-owner inference accounting mismatch")
    if any(not bool(item["seen"].all()) or item["initial_logits"] is None
           or item["terminal_event_logits"] is None for item in predictions):
        raise AssertionError("incomplete performance-level prediction assembly")

    manifest_rows: list[dict[str, Any]] = []
    diagnostics = Counter()
    initial_distribution = Counter()
    reference_effective = candidate_effective = 0
    first_two_hashes: list[str] = []
    for performance_index, (performance, prediction) in enumerate(zip(dataset.performances, predictions, strict=True)):
        with np.load(performance.cache_path, allow_pickle=False) as cache:
            timeline, interval_ticks, seconds_to_tick = timeline_from_cache(cache)
            decoded = decode_predictions_v1(
                initial_logits=prediction["initial_logits"],
                main_event_logits=prediction["main_event_logits"],
                main_timing_predictions=prediction["main_timing"],
                terminal_event_logits=prediction["terminal_event_logits"],
                terminal_timing_predictions=prediction["terminal_timing"],
                timeline=timeline,
            )
            repeated = decode_predictions_v1(
                initial_logits=prediction["initial_logits"],
                main_event_logits=prediction["main_event_logits"],
                main_timing_predictions=prediction["main_timing"],
                terminal_event_logits=prediction["terminal_event_logits"],
                terminal_timing_predictions=prediction["terminal_timing"],
                timeline=timeline,
            )
            if decoded.to_dict() != repeated.to_dict():
                raise AssertionError("Decoder v1 semantic determinism failed")
            quantized = quantize_decoded_timeline_v1(
                decoded, seconds_to_tick=seconds_to_tick,
                first_onset_tick=int(cache["main_interval_left_tick"][0]),
                latest_note_off_tick=int(cache["latest_note_off_tick"]),
                main_intervals=interval_ticks,
            )
            source = Path(str(cache["source_midi"].item()))
            performance_id = str(cache["performance_id"].item())
            piece_id = str(cache["piece_id"].item())
            reference_count = int(cache["main_effective_events_before_cap"]) + int(cache["terminal_effective_events_before_cap"])
        raw_main_classes = prediction["main_event_logits"].argmax(-1)
        raw_terminal_classes = prediction["terminal_event_logits"].argmax(-1)
        diagnostics["main_raw_predicted_non_none_slots"] += int((raw_main_classes != 0).sum())
        diagnostics["terminal_raw_predicted_non_none_slots"] += int((raw_terminal_classes != 0).sum())
        diagnostics.update(decoded.diagnostics)
        diagnostics["quantized_same_tick_main_collapsed"] += quantized.same_tick_main_collapsed
        diagnostics["quantized_same_state_suppressed"] += quantized.same_state_suppressed
        diagnostics["effective_emitted_main_transitions"] += sum(event.source == "MAIN" for event in quantized.events)
        diagnostics["effective_emitted_terminal_transitions"] += sum(event.source == "TERMINAL" for event in quantized.events)
        # Strict terminal guards equal the number of active terminal slots whose
        # rounded raw timestamp did not exceed the previous active tick.
        terminal_active = [event for event in decoded.active_slot_events if event.source == "TERMINAL"]
        previous_tick = interval_ticks[-1].right_tick
        guards = 0
        for event in terminal_active:
            candidate_tick = int(seconds_to_tick(event.time))
            guarded_tick = max(candidate_tick, previous_tick + 1)
            guards += int(guarded_tick != candidate_tick)
            previous_tick = guarded_tick
        diagnostics["terminal_strict_tick_guards"] += guards
        initial_distribution[str(decoded.initial_state)] += 1
        emitted_count = len(quantized.events) - 1
        reference_effective += reference_count
        candidate_effective += emitted_count
        destination = OUTPUT / "candidate_midi" / f"{performance_index:04d}_{hashlib.sha256(performance_id.encode()).hexdigest()[:16]}.mid"
        identity = write_candidate(source, destination, list(quantized.events))
        if performance_index < 2:
            duplicate = destination.with_name(destination.stem + ".determinism.mid")
            write_candidate(source, duplicate, list(quantized.events))
            if destination.read_bytes() != duplicate.read_bytes():
                raise AssertionError("candidate MIDI byte determinism failed")
            first_two_hashes.append(sha256_file(destination))
            duplicate.unlink()
        manifest_rows.append({
            "performance_index": performance_index,
            "performance_id": performance_id,
            "performance_path": performance.entry["performance_path"],
            "piece_id": piece_id,
            "source_midi": str(source),
            "source_sha256": sha256_file(source),
            "candidate_midi": str(destination),
            "candidate_sha256": sha256_file(destination),
            "modeled_onsets": performance.num_onsets,
            "windows": len(performance.ownership.window_starts),
            "initial_state": decoded.initial_state,
            "candidate_effective_state_changes": emitted_count,
            "reference_effective_state_changes_modeled_scope": reference_count,
            "decoded_final_state": quantized.final_pedal_state,
            "non_pedal_identity": identity,
        })
    del model
    torch.cuda.empty_cache()
    result = {
        "validation_windows": windows,
        "modeled_onsets": owned_total,
        "zero_owner_count": 0,
        "duplicate_owner_count": 0,
        "initial_prediction_sets": initial_total,
        "terminal_prediction_sets": terminal_total,
        "candidate_count": len(manifest_rows),
        "non_pedal_identity_pass_count": sum(row["non_pedal_identity"]["passed"] for row in manifest_rows),
        "performances_requiring_eot_extension": sum(
            int(row["non_pedal_identity"].get("eot_extension_ticks", 0) > 0)
            for row in manifest_rows
        ),
        "maximum_eot_extension_ticks": max(
            row["non_pedal_identity"].get("eot_extension_ticks", 0)
            for row in manifest_rows
        ),
        "all_outputs_finite": True,
        "semantic_determinism_sample_count": 2,
        "byte_determinism_sample_count": 2,
        "determinism_candidate_sha256": first_two_hashes,
        "initial_state_distribution": dict(initial_distribution),
        "decoder_diagnostics": dict(diagnostics),
        "candidate_effective_state_changes": candidate_effective,
        "reference_effective_state_changes_modeled_scope": reference_effective,
        "candidate_changes_per_performance": candidate_effective / 71,
        "reference_changes_per_performance": reference_effective / 71,
        "candidate_changes_per_modeled_onset": candidate_effective / owned_total,
        "reference_changes_per_modeled_onset": reference_effective / owned_total,
    }
    atomic_json(OUTPUT / "candidate_manifest.json", {"candidates": manifest_rows})
    atomic_json(OUTPUT / "decoder_diagnostics.json", result)
    return manifest_rows, result


def load_frozen_human(identifier: str) -> dict[str, Any] | None:
    root = FROZEN_HUMAN_ALIGNMENT / identifier
    metadata = root / "metadata.json"
    tokens = root / "aligned_tokens_int64.npy"
    transitions = root / "transitions.json"
    if not (metadata.is_file() and tokens.is_file() and transitions.is_file()):
        return None
    value = json.loads(metadata.read_text())
    if not value.get("complete"):
        return None
    return {"tokens": np.load(tokens, allow_pickle=False),
            "transitions": json.loads(transitions.read_text()), "metadata": value}


def evaluate(candidates: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    split = [row for row in read_csv(SPLIT_CSV) if row["split"] == "validation"]
    stage1 = {row["piece_id"]: row for row in read_csv(STAGE1_MANIFEST)}
    by_path = {row["performance_path"]: row for row in split}
    candidate_by_path = {row["performance_path"]: row for row in candidates}
    if len(split) != 71 or len(candidate_by_path) != 71 or len(stage1) != 19:
        raise RuntimeError("validation candidate/evaluator universe mismatch")
    confusion = np.zeros((4, 4), np.int64)
    candidate_hist = np.zeros(PATTERN_COUNT, np.int64)
    target_hist = np.zeros(PATTERN_COUNT, np.int64)
    transition = {scope: {key: 0 for key in ("candidate", "reference", "tp", "fp", "fn")}
                  for scope in ("pooled", "up", "down")}
    per_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    aligned_notes = 0
    for performance_path in sorted(by_path, key=lambda value: int(by_path[value]["metadata_index"])):
        human_row = by_path[performance_path]
        identifier = human_row["metadata_index"]
        target = load_frozen_human(identifier)
        if target is None:
            failures.append({"metadata_index": identifier, "performance_path": performance_path,
                             "reason": "frozen shared-human alignment unavailable"})
            continue
        candidate_row = candidate_by_path[performance_path]
        score = Path(stage1[human_row["piece_id"]]["selected_score_absolute_path"])
        candidate = align_and_cache(
            OUTPUT, kind="custom_event", identifier=identifier, score_path=score,
            performance_path=Path(candidate_row["candidate_midi"]),
        )
        candidate_classes = canonical_classes_from_tokens(candidate["tokens"])
        target_classes = canonical_classes_from_tokens(target["tokens"])
        if candidate_classes.shape != target_classes.shape:
            raise RuntimeError(f"aligned shape mismatch for {performance_path}")
        matrix = confusion_from_pairs(candidate_classes, target_classes)
        metrics = classification_metrics(matrix)
        counts = pooled_transition_counts(candidate["transitions"], target["transitions"])
        confusion += matrix
        candidate_hist += np.bincount(pattern_ids(candidate_classes), minlength=PATTERN_COUNT)
        target_hist += np.bincount(pattern_ids(target_classes), minlength=PATTERN_COUNT)
        sum_transition(transition, counts)
        aligned_notes += len(candidate_classes)
        row_transition = finalize_transition({scope: dict(values) for scope, values in counts.items()})
        per_rows.append({
            "metadata_index": identifier,
            "performance_path": performance_path,
            "piece_id": human_row["piece_id"],
            "aligned_notes": len(candidate_classes),
            "four_class_accuracy": metrics["token_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "transition_precision": row_transition["pooled"]["precision"],
            "transition_recall": row_transition["pooled"]["recall"],
            "transition_f1": row_transition["pooled"]["f1"],
            "candidate_transitions": row_transition["pooled"]["candidate"],
            "reference_transitions": row_transition["pooled"]["reference"],
            "tp": row_transition["pooled"]["tp"],
            "fp": row_transition["pooled"]["fp"],
            "fn": row_transition["pooled"]["fn"],
        })
    if len(per_rows) != 70 or len(failures) != 1 or aligned_notes != 272053:
        raise RuntimeError(
            f"frozen common-pair regression failed: pairs={len(per_rows)} failures={len(failures)} notes={aligned_notes}"
        )
    classification = classification_metrics(confusion)
    patterns = pattern_metrics(candidate_hist, target_hist)
    transition = finalize_transition(transition)
    aggregate = {
        "model": "Custom Event Model v0 / Prediction Decoder v1",
        "checkpoint_epoch": 1,
        "validation_candidates": 71,
        "validation_pieces": 19,
        "human_performances": 71,
        "common_successful_pairs": 70,
        "aligned_notes": aligned_notes,
        "pedal_samples": aligned_notes * 4,
        "classification": classification,
        "transition": transition,
        "patterns": {**patterns, "top_candidate_patterns": top_patterns(candidate_hist),
                     "top_target_patterns": top_patterns(target_hist)},
        "evaluator": {
            "source": "src/stage2_four_class/validation_evaluator.py + scripts/run_stage2_4class_validation_eval_v0.py",
            "classes": ["ZERO 0-25", "LOW 26-63", "HALF 64-103", "FULL 104-127"],
            "transition": "raw CC64 threshold 64, direction-aware one-to-one, tolerance +/-1 distinct onset",
            "patterns": "4^4=256 joint Pedal1-4 bins; base-2 JS divergence; intersection sum(min(P,Q))",
            "frozen_human_alignment_root": str(FROZEN_HUMAN_ALIGNMENT),
        },
    }
    atomic_json(OUTPUT / "aggregate_metrics.json", aggregate)
    atomic_json(OUTPUT / "alignment_failures.json", {"failures": failures})
    with (OUTPUT / "per_performance_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_rows[0]))
        writer.writeheader()
        writer.writerows(per_rows)
    return aggregate, failures


def write_report(provenance: Mapping[str, Any], diagnostics: Mapping[str, Any], metrics: Mapping[str, Any], failures: list[dict[str, Any]]) -> None:
    classification = metrics["classification"]
    transition = metrics["transition"]["pooled"]
    patterns = metrics["patterns"]
    reference = {
        "acc": 0.507749, "macro": 0.409029, "p": 0.502348, "r": 0.513770,
        "f1": 0.507995, "js": 0.038362, "intersection": 0.834859,
        "candidates": 35354,
    }
    d = diagnostics["decoder_diagnostics"]
    candidate_count = int(transition["candidate"])
    slot56_note = (
        "The trained model previously showed Slot5/6 non-NONE overprediction. At MIDI/evaluator "
        "level the candidate has 35,825 threshold transitions versus 34,568 reference transitions "
        "(+3.64%) and 35,354 for Hybrid (+1.33%); this is a modest association, not proof of "
        "slot-level causality."
    )
    bottleneck = "precision" if transition["precision"] < transition["recall"] else "recall"
    benefit = transition["f1"] > reference["f1"] and patterns["js_divergence_base2"] <= reference["js"]
    decision = ("CUSTOM EVENT REPRESENTATION SHOWS VALIDATION BENEFIT" if benefit else
                "CUSTOM EVENT REPRESENTATION NOT YET BETTER THAN CONVENTIONAL FINALIST")
    text = f"""# Custom Event Model v0 Canonical Validation Report

## Frozen provenance

- Checkpoint: `{provenance['checkpoint']}`; epoch/best epoch = 1; SHA256 `{provenance['checkpoint_sha256']}`
- Tokenizer cache ID: `{CANONICAL_CACHE_ID}`
- Note-alignment ID: `{CANONICAL_ALIGNMENT_ID}`
- Alignment config SHA: `{EXPECTED_ALIGNMENT_CONFIG_SHA}`
- Prediction Decoder v1 `{DECODER_VERSION}` ID: `{DECODER_ID}`
- ASAP validation only: 71 performances / 19 pieces. ASAP test access = 0. Repedal execution = 0.

## Input and evaluation convention

The frozen Custom Event cache defines 71 performance-level validation inputs, so inference uses each canonical ASAP-validation human performance's non-pedal stream and produces 71 candidates. This is required by the frozen 272,927-onset ownership universe. Every pitch, onset, note-off, velocity, and every non-CC64 musical message were preserved exactly. For terminal events extending beyond a source EOT, only the EOT meta tick is minimally extended and explicitly counted. The conventional evaluator's metric definitions and its frozen successful 70-pair/272,053-aligned-note universe are reused exactly. The historical conventional candidate construction used 19 saved Original-PT piece candidates; that input-construction difference is retained as an explicit provenance caveat when comparing rows.

## Integrity

- Candidates: {diagnostics['candidate_count']}/71; musical non-pedal identity: {diagnostics['non_pedal_identity_pass_count']}/71 PASS. EOT-only extension was required for {diagnostics['performances_requiring_eot_extension']} candidates (maximum {diagnostics['maximum_eot_extension_ticks']} ticks); no note or other non-CC64 event changed.
- Unique-owner predictions: {diagnostics['modeled_onsets']:,}/272,927; zero-owner 0; duplicate-owner 0.
- Initial/terminal sets: {diagnostics['initial_prediction_sets']}/71 and {diagnostics['terminal_prediction_sets']}/71.
- Finite outputs and semantic determinism: PASS; two-candidate byte determinism: PASS.
- Frozen alignment exclusion: `{failures[0]['performance_path']}` (same unavailable shared-human alignment as canonical evaluator); common successful pairs = 70.

## Main metrics

| Model | 4C Accuracy | Macro F1 | Transition P | Transition R | Transition F1 | Candidates | JS divergence | Intersection |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Custom Event v0 / Decoder v1 | {classification['token_accuracy']:.6f} | {classification['macro_f1']:.6f} | {transition['precision']:.6f} | {transition['recall']:.6f} | {transition['f1']:.6f} | {candidate_count:,} | {patterns['js_divergence_base2']:.6f} | {patterns['intersection']:.6f} |
| Hybrid regression-only (read-only reference) | {reference['acc']:.6f} | {reference['macro']:.6f} | {reference['p']:.6f} | {reference['r']:.6f} | {reference['f1']:.6f} | {reference['candidates']:,} | {reference['js']:.6f} | {reference['intersection']:.6f} |

The evaluator uses 4^4=256 joint Pedal1-4 patterns, base-2 Jensen-Shannon divergence, and intersection Σmin(P,Q). Transitions use raw CC64 threshold 64, direction-aware one-to-one matching, and ±1 distinct-onset tolerance.

## Decoder diagnostics

- Initial-state distribution (class IDs 0..3): `{diagnostics['initial_state_distribution']}`
- Raw Main non-NONE slots: {d.get('main_raw_predicted_non_none_slots', 0):,}; active after first-NONE: {d.get('main_active_slots', 0):,}; ignored later non-NONE: {d.get('main_ignored_after_first_none', 0):,}.
- Main raw tau order violations: {d.get('main_raw_order_violations', 0):,}; clamp low/high: {d.get('main_tau_clamp_low', 0):,}/{d.get('main_tau_clamp_high', 0):,}.
- Main same-time collapse (continuous/quantized): {d.get('main_same_time_collapsed', 0):,}/{d.get('quantized_same_tick_main_collapsed', 0):,}; Main same-state suppression: {d.get('same_state_suppressed_main', 0):,}.
- Raw Terminal non-NONE: {d.get('terminal_raw_predicted_non_none_slots', 0):,}; active: {d.get('terminal_active_slots', 0):,}; ignored later: {d.get('terminal_ignored_after_first_none', 0):,}.
- Terminal negative-z clamps: {d.get('terminal_negative_z_clamp', 0):,}; strict-tick guards: {d.get('terminal_strict_tick_guards', 0):,}; same-state suppression: {d.get('same_state_suppressed_terminal', 0):,}.
- Effective emitted Main/Terminal transitions before evaluator alignment: {d.get('effective_emitted_main_transitions', 0):,}/{d.get('effective_emitted_terminal_transitions', 0):,}.
- Candidate/reference effective state changes in modeled source scope: {diagnostics['candidate_effective_state_changes']:,}/{diagnostics['reference_effective_state_changes_modeled_scope']:,}.

{slot56_note}

## Interpretation

- The primary transition bottleneck is **{bottleneck}** because P={transition['precision']:.6f} and R={transition['recall']:.6f}.
- Compared with the read-only Hybrid reference, Custom Event deltas are: accuracy {classification['token_accuracy']-reference['acc']:+.6f}, Macro F1 {classification['macro_f1']-reference['macro']:+.6f}, Transition F1 {transition['f1']-reference['f1']:+.6f}, JS {patterns['js_divergence_base2']-reference['js']:+.6f}, Intersection {patterns['intersection']-reference['intersection']:+.6f}.
- The first failure mode to inspect in a future task is the relationship between prefix-active sparse slots, effective transition density, and {bottleneck}; no corrective experiment was run here.
- No decoder, tokenizer, checkpoint, threshold, calibration, or post-processing was changed after observing metrics.

## Decision

**{decision}**
"""
    (OUTPUT / "CUSTOM_EVENT_MODEL_V0_CANONICAL_VAL_REPORT.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    provenance = preflight()
    config = {key: value for key, value in provenance.items() if key != "checkpoint_payload"}
    config.update({
        "window_notes": 512, "stride_notes": 256,
        "inference_ownership": "unique owner: full chord, maximum representative margin, tie smaller start",
        "overlap_averaging": False,
        "validation_input": "frozen cache original ASAP-validation human non-pedal performance",
        "post_hoc_tuning_count": 0,
    })
    atomic_json(OUTPUT / "inference_config.json", config)
    atomic_json(OUTPUT / "provenance.json", config)
    candidates, diagnostics = infer_and_render(provenance)
    metrics, failures = evaluate(candidates)
    write_report(config, diagnostics, metrics, failures)
    atomic_json(OUTPUT / "run_status.json", {
        "status": "completed", "candidates": 71, "common_successful_pairs": 70,
        "asap_test_access_count": 0, "repedal_execution_count": 0,
        "post_hoc_tuning_count": 0,
    })
    print(json.dumps({
        "status": "completed", "candidates": 71,
        "accuracy": metrics["classification"]["token_accuracy"],
        "macro_f1": metrics["classification"]["macro_f1"],
        "transition_f1": metrics["transition"]["pooled"]["f1"],
        "js": metrics["patterns"]["js_divergence_base2"],
        "intersection": metrics["patterns"]["intersection"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
