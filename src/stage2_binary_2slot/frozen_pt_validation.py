"""Input-only frozen Original-PT validation adapter for Binary 2-Slot.

This module tokenizes already-saved canonical Original-PT MIDI.  It never runs
Stage 1 inference and never reads a human pedal state during model inference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from miditoolkit import MidiFile

from src.stage2_binary.strict_midi_validation import PianoT5GemmaConfig, midi_to_ids
from src.stage2_encoder_only.dataset import MASK_ID, NON_PEDAL_FEATURES
from src.stage2_event_model.ownership import OwnershipResult, assign_unique_owners
from src.stage2_event_tokenizer.binary_2slot import OFF
from src.stage2_event_tokenizer.tokenizer_v1 import _tick_second_converters
from third_party.PianistTransformer.src.utils.midi import normalize_midi

from .dataset import build_boundary_input
from .rollout import decode_prediction, next_binary_state
from src.stage2_binary.canonical_stage1 import cc64_schedule


TIMING_START = 261


@dataclass(frozen=True)
class FrozenPTInterval:
    region: str
    global_index: int
    main_onset_index: int | None
    left_seconds: float
    right_seconds: float

    @property
    def duration_seconds(self) -> float:
        return self.right_seconds - self.left_seconds


@dataclass(frozen=True)
class FrozenPTPiece:
    piece_id: str
    source_midi: Path
    masked_tokens: torch.Tensor
    ownership: OwnershipResult
    onset_seconds: tuple[float, ...]
    note_end_seconds: float
    pre: FrozenPTInterval
    main: tuple[FrozenPTInterval, ...]
    post: FrozenPTInterval

    @property
    def intervals(self) -> tuple[FrozenPTInterval, ...]:
        return (self.pre,) + self.main + (self.post,)


@dataclass(frozen=True)
class FrozenPTRecord:
    region: str
    global_index: int
    main_onset_index: int | None
    model_state: int
    predicted_count: int
    predicted_timing: tuple[float, float]
    state_after: int


@dataclass(frozen=True)
class FrozenPTRollout:
    piece: FrozenPTPiece
    records: tuple[FrozenPTRecord, ...]
    candidate_events: tuple[dict[str, Any], ...]
    final_state: int
    windows: int


@dataclass(frozen=True)
class RenderedCC64:
    tick: int
    cc64_value: int
    order: int

    @property
    def ordering_key(self) -> tuple[int, int]:
        return (self.tick, self.order)


def project_decoded_cc64_to_midi_origin(
    events: Sequence[Mapping[str, Any]],
    seconds_to_tick,
) -> tuple[list[RenderedCC64], dict[str, Any]]:
    """Project a conceptual decoded trajectory onto MIDI's nonnegative domain.

    PRE can start before MIDI tick zero when the first note starts before one
    second. Moving every negative event to tick zero would destroy timing and
    can create spurious same-tick toggles. Preserve the conceptual events for
    metric evaluation, and serialize their state at the MIDI origin as one
    deterministic boundary anchor when that state is ON.
    """

    rendered: list[RenderedCC64] = []
    pre_origin: list[tuple[int, Mapping[str, Any]]] = []
    for order, event in enumerate(events):
        seconds = float(event["seconds"])
        if not math.isfinite(seconds):
            raise ValueError("decoded CC64 event has non-finite time")
        if seconds < 0.0:
            if event["region"] != "PRE":
                raise ValueError("negative decoded CC64 event outside PRE")
            pre_origin.append((order, event))
            continue
        tick = int(seconds_to_tick(seconds))
        if tick < 0:
            raise ValueError("nonnegative decoded CC64 time converted to negative MIDI tick")
        rendered.append(RenderedCC64(
            tick,
            127 if event["direction"] == "DOWN" else 0,
            order,
        ))

    state_at_origin = OFF
    for _, event in sorted(pre_origin, key=lambda item: (float(item[1]["seconds"]), item[0])):
        expected = "DOWN" if state_at_origin == OFF else "UP"
        if event["direction"] != expected:
            raise ValueError("pre-origin decoded direction is inconsistent with binary state")
        state_at_origin ^= 1
    anchor_inserted = state_at_origin != OFF
    if anchor_inserted:
        rendered.append(RenderedCC64(0, 127, -1))
    rendered.sort(key=lambda event: event.ordering_key)
    return rendered, {
        "midi_time_domain_left_seconds": 0.0,
        "pre_origin_decoded_event_count": len(pre_origin),
        "pre_origin_first_seconds": min((float(event["seconds"]) for _, event in pre_origin), default=None),
        "pre_origin_last_seconds": max((float(event["seconds"]) for _, event in pre_origin), default=None),
        "state_at_midi_origin": int(state_at_origin),
        "midi_origin_state_anchor_inserted": bool(anchor_inserted),
        "midi_origin_state_anchor_cc64": 127 if anchor_inserted else None,
        "negative_event_timestamp_clamp_used": False,
    }


def _as_note_tokens(ids: Sequence[int] | np.ndarray) -> np.ndarray:
    value = np.asarray(ids, dtype=np.int64)
    if value.ndim == 1:
        if not value.size or value.size % 8:
            raise ValueError("frozen PT token sequence is not complete eight-token notes")
        value = value.reshape(-1, 8)
    if value.ndim != 2 or value.shape[1] != 8 or not len(value):
        raise ValueError("frozen PT tokens must have shape [N,8]")
    return value


def _raw_groups_in_pt_order(midi: MidiFile) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, ...], tuple[float, ...], float]:
    """Map raw distinct-onset groups into official normalized PT note order.

    This mirrors the provenance-carrying alignment used by the existing Custom
    Event dataset.  Raw onsets that round to one PT tick remain distinct target
    intervals, while their representative notes are gathered from PT order.
    """

    raw_notes = []
    source_order = 0
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            raw_notes.append({
                "source_order": source_order, "start_raw": int(note.start),
                "end_raw": int(note.end), "pitch": int(note.pitch),
                "velocity": int(note.velocity),
            })
            source_order += 1
    canonical = sorted(
        raw_notes,
        key=lambda item: (item["start_raw"], item["pitch"], item["end_raw"], item["velocity"], item["source_order"]),
    )
    raw_onsets = tuple(sorted({item["start_raw"] for item in canonical}))
    onset_to_group = {tick: index for index, tick in enumerate(raw_onsets)}
    tick_to_time = midi.get_tick_to_time_mapping()
    converted = []
    for item in raw_notes:
        start = round(float(tick_to_time[item["start_raw"]]) * 1000.0)
        end = round(float(tick_to_time[item["end_raw"]]) * 1000.0)
        if start >= end:
            end = start + 1
        converted.append({
            **item, "start": start, "end": end,
            "raw_group": onset_to_group[item["start_raw"]],
        })
    by_pitch: dict[int, list[dict[str, int]]] = defaultdict(list)
    for item in converted:
        by_pitch[item["pitch"]].append(item)
    retained = []
    dropped = []
    for pitch in sorted(by_pitch):
        notes = sorted(by_pitch[pitch], key=lambda item: item["start"])
        for index in range(len(notes) - 1):
            current, following = notes[index], notes[index + 1]
            if current["end"] >= following["start"]:
                current["end"] = following["start"]
                if current["start"] >= current["end"]:
                    current["dropped"] = 1
                    dropped.append(current["source_order"])
        retained.extend(item for item in notes if not item.get("dropped"))
    retained.sort(key=lambda item: (item["start"], item["pitch"]))
    official = normalize_midi(midi).instruments[0].notes
    simulated = [(item["start"], item["pitch"], item["end"], item["velocity"]) for item in retained]
    official_identity = [(int(note.start), int(note.pitch), int(note.end), int(note.velocity)) for note in official]
    if dropped or simulated != official_identity or len(retained) != len(raw_notes):
        raise RuntimeError("frozen PT raw-note alignment differs from official normalization")
    source_to_pt = {item["source_order"]: index for index, item in enumerate(retained)}
    grouped: list[list[dict[str, int]]] = [[] for _ in raw_onsets]
    for item in canonical:
        grouped[onset_to_group[item["start_raw"]]].append(item)
    first = np.asarray([min(source_to_pt[item["source_order"]] for item in group) for group in grouped], dtype=np.int64)
    last = np.asarray([max(source_to_pt[item["source_order"]] for item in group) for group in grouped], dtype=np.int64)
    representative = np.asarray([source_to_pt[group[-1]["source_order"]] for group in grouped], dtype=np.int64)
    onset_seconds = tuple(float(tick_to_time[tick]) for tick in raw_onsets)
    note_end_seconds = max(float(tick_to_time[item["end_raw"]]) for item in raw_notes)
    return first, last, representative, raw_onsets, onset_seconds, note_end_seconds


def load_frozen_pt_piece(row: Mapping[str, str]) -> FrozenPTPiece:
    """Tokenize one saved canonical MIDI and construct unique onset owners."""

    source = Path(row["canonical_midi_path"])
    source_midi = MidiFile(str(source))
    tokens = _as_note_tokens(midi_to_ids(PianoT5GemmaConfig(), source_midi))
    source_note_count = sum(len(instrument.notes) for instrument in source_midi.instruments)
    if source_note_count != len(tokens):
        raise RuntimeError(
            f"frozen PT MIDI/token note count changed for {row['piece_id']}: "
            f"{source_note_count} != {len(tokens)}"
        )
    masked = tokens.copy()
    masked[:, NON_PEDAL_FEATURES:] = MASK_ID
    if not np.all(masked[:, NON_PEDAL_FEATURES:] == MASK_ID):
        raise AssertionError("frozen Original-PT pedal tokens leaked into Stage 2 input")

    first, last, representative, _, onset_seconds, note_end = _raw_groups_in_pt_order(source_midi)
    ownership = assign_unique_owners(len(tokens), first, last, representative)
    if np.any(np.diff(ownership.owner_window_index) < 0):
        raise AssertionError("frozen PT owner ordering is non-monotonic")
    flattened = [int(value) for values in ownership.owned_onset_indices for value in values]
    if flattened != list(range(len(first))):
        raise AssertionError("frozen PT onset ownership is incomplete or duplicated")

    if note_end <= onset_seconds[-1]:
        raise ValueError("frozen PT final MAIN interval is not positive")
    pre = FrozenPTInterval("PRE", 0, None, onset_seconds[0] - 1.0, onset_seconds[0])
    main = tuple(
        FrozenPTInterval(
            "MAIN",
            index + 1,
            index,
            onset,
            note_end if index + 1 == len(onset_seconds) else onset_seconds[index + 1],
        )
        for index, onset in enumerate(onset_seconds)
    )
    post = FrozenPTInterval("POST", len(main) + 1, None, note_end, note_end + 1.0)
    return FrozenPTPiece(
        str(row["piece_id"]), source, torch.from_numpy(masked.copy()).long(), ownership,
        onset_seconds, note_end, pre, main, post,
    )


def _predict(model, hidden: torch.Tensor, state: int) -> tuple[int, tuple[float, float], int]:
    output = model.condition_and_predict(
        hidden.float().reshape(1, -1),
        torch.tensor([state], dtype=torch.long, device=hidden.device),
    )
    if not bool(torch.isfinite(output.count_logits).all() and torch.isfinite(output.timing_predictions).all()):
        raise FloatingPointError("non-finite frozen PT prediction")
    count = int(output.count_logits[0].argmax().item())
    timing = tuple(float(value) for value in output.timing_predictions[0].float().cpu().tolist())
    return count, timing, next_binary_state(state, count)


def _record(
    model,
    hidden: torch.Tensor,
    interval: FrozenPTInterval,
    state: int,
    events: list[dict[str, Any]],
) -> tuple[FrozenPTRecord, int]:
    count, timing, state_after = _predict(model, hidden, state)
    decoded = decode_prediction(state, count, timing)
    for slot, event in enumerate(decoded.transitions):
        events.append({
            "direction": event.direction,
            "seconds": interval.left_seconds + event.tau * interval.duration_seconds,
            "region": interval.region,
            "interval_index": interval.global_index,
            "slot": slot,
        })
    return FrozenPTRecord(
        interval.region, interval.global_index, interval.main_onset_index,
        state, count, timing, state_after,
    ), state_after


def rollout_frozen_pt(model, piece: FrozenPTPiece, device: torch.device, *, amp: bool = True) -> FrozenPTRollout:
    """Pure hard rollout from OFF over PRE, all MAIN, and POST."""

    model.eval()
    state = OFF
    records: list[FrozenPTRecord] = []
    events: list[dict[str, Any]] = []
    masked = piece.masked_tokens
    with torch.inference_mode():
        pre_input = build_boundary_input(masked, "PRE")
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp and device.type == "cuda"):
            hidden = model.encode_boundary(
                pre_input["input_ids"].reshape(1, -1).to(device),
                pre_input["token_attention_mask"].reshape(1, -1).to(device),
                pre_input["note_mask"].reshape(1, -1).to(device),
                torch.tensor([pre_input["query_position"]], dtype=torch.long, device=device),
            )[0]
        record, state = _record(model, hidden, piece.pre, state, events)
        records.append(record)

        seen: list[int] = []
        for window_index, (start, end) in enumerate(zip(piece.ownership.window_starts, piece.ownership.window_ends)):
            owned = piece.ownership.owned_onset_indices[window_index]
            local = piece.ownership.owned_local_representative_indices[window_index]
            if not len(owned):
                continue
            notes = masked[start:end]
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp and device.type == "cuda"):
                encoded = model.encode_main(
                    notes.reshape(1, -1).to(device),
                    torch.ones((1, notes.numel()), dtype=torch.long, device=device),
                    torch.ones((1, len(notes)), dtype=torch.bool, device=device),
                    torch.from_numpy(local).reshape(1, -1).long().to(device),
                    torch.ones((1, len(local)), dtype=torch.bool, device=device),
                ).owned_onset_hidden_states[0]
            for position, onset_index in enumerate(owned.tolist()):
                interval = piece.main[int(onset_index)]
                record, state = _record(model, encoded[position], interval, state, events)
                records.append(record)
                seen.append(int(onset_index))
        if seen != list(range(len(piece.main))):
            raise AssertionError("frozen PT MAIN onset supervision is missing or duplicated")

        post_input = build_boundary_input(masked, "POST")
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp and device.type == "cuda"):
            hidden = model.encode_boundary(
                post_input["input_ids"].reshape(1, -1).to(device),
                post_input["token_attention_mask"].reshape(1, -1).to(device),
                post_input["note_mask"].reshape(1, -1).to(device),
                torch.tensor([post_input["query_position"]], dtype=torch.long, device=device),
            )[0]
        record, state = _record(model, hidden, piece.post, state, events)
        records.append(record)
    if [record.global_index for record in records] != list(range(len(records))):
        raise AssertionError("frozen PT rollout interval chronology changed")
    return FrozenPTRollout(piece, tuple(records), tuple(events), state, len(piece.ownership.window_starts))


def render_cc64_only(rollout: FrozenPTRollout, destination: str | Path) -> dict[str, Any]:
    """Replace CC64 while preserving every source note exactly."""

    from scripts.run_custom_event_model_v0_canonical_val_inference_v1 import write_candidate

    _, seconds_to_tick = _tick_second_converters(rollout.piece.source_midi)
    rendered, origin_projection = project_decoded_cc64_to_midi_origin(
        rollout.candidate_events, seconds_to_tick,
    )
    destination = Path(destination)
    if destination.exists():
        destination.unlink()
    identity = write_candidate(rollout.piece.source_midi, destination, rendered)
    source_midi = MidiFile(str(rollout.piece.source_midi))
    candidate_midi = MidiFile(str(destination))
    source_notes = sorted(
        (note.pitch, note.start, note.end, note.velocity)
        for instrument in source_midi.instruments for note in instrument.notes
    )
    candidate_notes = sorted(
        (note.pitch, note.start, note.end, note.velocity)
        for instrument in candidate_midi.instruments for note in instrument.notes
    )
    if source_notes != candidate_notes:
        raise AssertionError("frozen PT candidate changed pitch/onset/note-off/velocity")
    identity.update({
        "note_identity_exact": True,
        "note_count": len(source_notes),
        "pitch_onset_noteoff_velocity_exact": True,
        **origin_projection,
    })
    return identity


def serialized_candidate_events_for_metric(candidate_midi: str | Path) -> list[dict[str, Any]]:
    """Return the exact serializable CC64 trajectory used by canonical metrics."""

    events = []
    previous_state = OFF
    for row in sorted(cc64_schedule(candidate_midi), key=lambda item: (
        int(item["absolute_tick"]), int(item["track_index"]), int(item["source_order"]),
    )):
        value = int(row["value"])
        if value not in (0, 127):
            raise ValueError("Binary 2-Slot candidate MIDI contains a non-binary CC64 value")
        state = int(value >= 64)
        if state == previous_state:
            raise ValueError("Binary 2-Slot candidate MIDI contains a redundant same-state CC64 event")
        events.append({
            "direction": "DOWN" if state else "UP",
            "source_tick": int(row["absolute_tick"]),
        })
        previous_state = state
    return events
