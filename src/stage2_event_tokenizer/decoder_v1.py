"""Deterministic decoder for frozen Custom Event Tokenizer v1."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import mido

from .tokenizer import REPRESENTATIVES, parse_raw_midi
from .tokenizer_v1 import EncodedPerformanceV1, _tick_second_converters


@dataclass(frozen=True)
class DecodedControlV1:
    tick: int
    value: int
    kind: str
    slot_index: int | None


def safe_main_tau(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def safe_terminal_gap_from_log(value: float) -> float:
    return math.expm1(max(0.0, float(value)))


def decode_controls_v1(
    encoded: EncodedPerformanceV1,
    *,
    main_event_classes: Sequence[Sequence[int]] | None = None,
    main_taus: Sequence[Sequence[float]] | None = None,
    terminal_event_classes: Sequence[int] | None = None,
    terminal_log1p_gaps: Sequence[float] | None = None,
) -> tuple[DecodedControlV1, ...]:
    """Decode targets or future predictions using canonical safety clamps."""

    first_onset = encoded.onset_groups[0].onset_tick
    controls = [
        DecodedControlV1(max(0, first_onset - 1), REPRESENTATIVES[encoded.initial_state], "initial", None)
    ]
    if main_event_classes is None:
        main_event_classes = tuple(
            tuple(slot.event_class for slot in interval.slots) for interval in encoded.main_intervals
        )
    if main_taus is None:
        main_taus = tuple(
            tuple(0.0 if slot.tau is None else slot.tau for slot in interval.slots)
            for interval in encoded.main_intervals
        )
    if len(main_event_classes) != len(encoded.main_intervals) or len(main_taus) != len(encoded.main_intervals):
        raise ValueError("main prediction interval count mismatch")
    for interval, classes, taus in zip(encoded.main_intervals, main_event_classes, main_taus):
        if len(classes) != 6 or len(taus) != 6:
            raise ValueError("each main interval requires six event/timing slots")
        for slot_index, (event_class, tau) in enumerate(zip(classes, taus)):
            event_class = int(event_class)
            if event_class == 0:
                continue
            if not 1 <= event_class <= 4:
                raise ValueError(f"invalid main event class: {event_class}")
            clipped = safe_main_tau(float(tau))
            tick = int(round(interval.left_tick + clipped * (interval.right_tick - interval.left_tick)))
            controls.append(
                DecodedControlV1(tick, REPRESENTATIVES[event_class - 1], "main", slot_index)
            )

    if terminal_event_classes is None:
        terminal_event_classes = tuple(slot.event_class for slot in encoded.terminal.slots)
    if terminal_log1p_gaps is None:
        terminal_log1p_gaps = tuple(
            0.0 if slot.log1p_gap is None else slot.log1p_gap for slot in encoded.terminal.slots
        )
    if len(terminal_event_classes) != 4 or len(terminal_log1p_gaps) != 4:
        raise ValueError("terminal prediction requires four event/timing slots")
    _, seconds_to_tick = _tick_second_converters(encoded.source.path)
    current_seconds = encoded.terminal.anchor_seconds
    for slot_index, (event_class, transformed_gap) in enumerate(
        zip(terminal_event_classes, terminal_log1p_gaps)
    ):
        event_class = int(event_class)
        if event_class == 0:
            continue
        if not 1 <= event_class <= 4:
            raise ValueError(f"invalid terminal event class: {event_class}")
        current_seconds += safe_terminal_gap_from_log(float(transformed_gap))
        controls.append(
            DecodedControlV1(
                seconds_to_tick(current_seconds),
                REPRESENTATIVES[event_class - 1],
                "terminal",
                slot_index,
            )
        )
    return tuple(controls)


def write_roundtrip_midi_v1(encoded: EncodedPerformanceV1, output_path: str | Path) -> Path:
    source_path = Path(encoded.source.path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    midi = mido.MidiFile(str(source_path), clip=False)
    controls = decode_controls_v1(encoded)
    rebuilt_tracks: list[mido.MidiTrack] = []
    for track_index, track in enumerate(midi.tracks):
        absolute = 0
        rows = []
        for order, message in enumerate(track):
            absolute += int(message.time)
            is_cc64 = (
                message.type == "control_change"
                and int(message.channel) != 9
                and int(message.control) == 64
            )
            if not is_cc64:
                rows.append((absolute, 0, order, message.copy(time=0)))
        if track_index == encoded.source.preferred_cc_track:
            for order, control in enumerate(controls):
                priority = -2 if control.kind == "initial" else -1
                rows.append(
                    (
                        control.tick,
                        priority,
                        order,
                        mido.Message(
                            "control_change",
                            channel=encoded.source.preferred_cc_channel,
                            control=64,
                            value=control.value,
                            time=0,
                        ),
                    )
                )
        rows.sort(key=lambda item: (item[0], item[1], item[2]))
        rebuilt = mido.MidiTrack()
        previous = 0
        for tick, _, _, message in rows:
            rebuilt.append(message.copy(time=tick - previous))
            previous = tick
        rebuilt_tracks.append(rebuilt)
    midi.tracks.clear()
    midi.tracks.extend(rebuilt_tracks)
    midi.save(str(output))
    reconstructed = parse_raw_midi(output)
    if reconstructed.note_signature != encoded.source.note_signature:
        raise AssertionError("v1 round-trip changed non-pedal note signature")
    return output
