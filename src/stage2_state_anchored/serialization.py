"""Run B event reconstruction and MIDI-only serialization support.

The serialization-only S1 anchor is intentionally kept separate from the
modeled transition list.  Metrics consume ``modeled_events`` directly, never
the CC64 schedule written for MIDI playback.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from miditoolkit import MidiFile

from src.stage2_binary_2slot.frozen_pt_validation import FrozenPTPiece, RenderedCC64
from src.stage2_event_tokenizer.binary_2slot import DOWN, OFF, ON, UP
from src.stage2_event_tokenizer.tokenizer_v1 import _tick_second_converters

from .decoding import CHANGE_ID, HOLD_ID, RETURN_ID, decode_timings


@dataclass(frozen=True)
class StateAnchoredRenderResult:
    destination: Path
    modeled_events: tuple[dict[str, Any], ...]
    identity: dict[str, Any]


def reconstruct_modeled_events(
    onset_seconds: Sequence[float],
    state_path: Sequence[int] | torch.Tensor,
    mode_path: Sequence[int] | torch.Tensor,
    timing_predictions: torch.Tensor,
) -> tuple[dict[str, Any], ...]:
    """Reconstruct only the M-1 modeled interval events on ``[t1,tM)``."""

    onsets = tuple(float(value) for value in onset_seconds)
    states = torch.as_tensor(state_path, dtype=torch.long).cpu()
    modes = torch.as_tensor(mode_path, dtype=torch.long).cpu()
    if len(onsets) < 1 or len(states) != len(onsets) or len(modes) != len(onsets) - 1:
        raise ValueError("Run B event reconstruction requires M states and M-1 modes")
    if any(right <= left for left, right in zip(onsets, onsets[1:])):
        raise ValueError("distinct onset times must be strictly chronological")
    timings = decode_timings(timing_predictions, modes)
    events: list[dict[str, Any]] = []
    for index, (left, right, mode, taus) in enumerate(
        zip(onsets[:-1], onsets[1:], modes.tolist(), timings, strict=True)
    ):
        start = int(states[index])
        end = int(states[index + 1])
        if start not in (OFF, ON) or end not in (OFF, ON):
            raise ValueError("decoded state escaped OFF/ON")
        if mode == HOLD_ID:
            directions: tuple[str, ...] = ()
        elif mode == CHANGE_ID:
            if start == end:
                raise AssertionError("CHANGE edge has equal endpoint states")
            directions = (DOWN if start == OFF else UP,)
        elif mode == RETURN_ID:
            if start != end:
                raise AssertionError("RETURN edge has different endpoint states")
            directions = (DOWN, UP) if start == OFF else (UP, DOWN)
        else:
            raise ValueError("unknown Run B Mode")
        if len(directions) != len(taus):
            raise AssertionError("Mode/timing cardinality mismatch")
        duration = right - left
        for slot, (direction, tau) in enumerate(zip(directions, taus, strict=True)):
            seconds = left + float(tau) * duration
            events.append({
                "direction": direction,
                "seconds": seconds,
                "interval_index": index,
                "slot": slot,
                "tau": float(tau),
                "serialization_only": False,
            })
    return tuple(events)


def serialization_schedule(
    piece: FrozenPTPiece,
    initial_state: int,
    modeled_events: Sequence[Mapping[str, Any]],
) -> tuple[list[RenderedCC64], dict[str, Any]]:
    """Build the playback schedule, with at most one S1 absolute-state anchor."""

    if initial_state not in (OFF, ON):
        raise ValueError("S1 must be OFF or ON")
    _, seconds_to_tick = _tick_second_converters(piece.source_midi)
    first_tick = int(seconds_to_tick(float(piece.onset_seconds[0])))
    rendered: list[RenderedCC64] = []
    if initial_state == ON:
        rendered.append(RenderedCC64(first_tick, 127, -1))
    for order, event in enumerate(modeled_events):
        seconds = float(event["seconds"])
        if not float(piece.onset_seconds[0]) <= seconds <= float(piece.onset_seconds[-1]):
            raise ValueError("modeled event escaped the Run B endpoint domain")
        direction = str(event["direction"])
        if direction not in (DOWN, UP):
            raise ValueError("modeled transition direction is not binary")
        rendered.append(RenderedCC64(
            int(seconds_to_tick(seconds)), 127 if direction == DOWN else 0, order,
        ))
    rendered.sort(key=lambda event: event.ordering_key)
    return rendered, {
        "s1_state": int(initial_state),
        "initialization_anchor_inserted": bool(initial_state == ON),
        "initialization_anchor_count": int(initial_state == ON),
        "initialization_anchor_tick": first_tick if initial_state == ON else None,
        "initialization_anchor_cc64": 127 if initial_state == ON else None,
        "initialization_anchor_metric_event_count": 0,
        "metric_source": "modeled Run B events before MIDI serialization",
    }


def render_state_anchored_candidate(
    piece: FrozenPTPiece,
    state_path: Sequence[int] | torch.Tensor,
    mode_path: Sequence[int] | torch.Tensor,
    timing_predictions: torch.Tensor,
    destination: str | Path,
) -> StateAnchoredRenderResult:
    """Replace CC64 only, preserving every Frozen-PT non-pedal note event."""

    from scripts.run_custom_event_model_v0_canonical_val_inference_v1 import write_candidate

    states = torch.as_tensor(state_path, dtype=torch.long).cpu()
    modeled = reconstruct_modeled_events(
        piece.onset_seconds, states, mode_path, timing_predictions,
    )
    schedule, anchor = serialization_schedule(piece, int(states[0]), modeled)
    output = Path(destination)
    identity = write_candidate(piece.source_midi, output, schedule)
    source = MidiFile(str(piece.source_midi))
    candidate = MidiFile(str(output))
    source_notes = sorted(
        (note.pitch, note.start, note.end, note.velocity)
        for instrument in source.instruments for note in instrument.notes
    )
    candidate_notes = sorted(
        (note.pitch, note.start, note.end, note.velocity)
        for instrument in candidate.instruments for note in instrument.notes
    )
    if source_notes != candidate_notes:
        raise AssertionError("Run B candidate changed Frozen-PT non-pedal notes")
    identity.update({
        "note_identity_exact": True,
        "note_count": len(source_notes),
        "pitch_onset_noteoff_velocity_exact": True,
        "modeled_transition_count": len(modeled),
        **anchor,
    })
    return StateAnchoredRenderResult(output, modeled, identity)
