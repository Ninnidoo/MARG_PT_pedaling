"""Deterministic decoder and MIDI round-trip serializer for event-token oracles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mido

from .tokenizer import EncodedPerformance, EventToken, REPRESENTATIVES, parse_raw_midi


@dataclass(frozen=True)
class DecodedControl:
    tick: int
    value: int
    kind: str
    interval_index: int | None


def _event_tick(start: int, end: int, event: EventToken) -> int:
    if event.tau is None:
        raise ValueError("NONE has no decoded timestamp")
    tick = int(round(start + event.tau * (end - start)))
    if event.source_tick is not None and tick != event.source_tick:
        raise AssertionError(
            f"exact tau did not reconstruct source tick: {tick} != {event.source_tick}"
        )
    return tick


def decode_event_controls(
    encoded: EncodedPerformance, *, two_slot: bool
) -> tuple[DecodedControl, ...]:
    """Decode initial state followed by unlimited or retained absolute SET events."""

    first = encoded.source.distinct_onsets[0]
    initial_tick = max(0, first - 1)
    controls = [
        DecodedControl(initial_tick, REPRESENTATIVES[encoded.initial_state], "initial", None)
    ]
    for interval in encoded.intervals:
        events = (
            tuple(slot for slot in interval.slots if slot.event_class)
            if two_slot
            else interval.all_events
        )
        for event in events:
            if event.state is None:
                continue
            controls.append(
                DecodedControl(
                    _event_tick(interval.start_tick, interval.end_tick, event),
                    REPRESENTATIVES[event.state],
                    "event",
                    interval.index,
                )
            )
    return tuple(controls)


def write_roundtrip_midi(
    encoded: EncodedPerformance,
    output_path: str | Path,
    *,
    two_slot: bool,
) -> Path:
    """Copy the source MIDI exactly except for non-drum CC64 messages."""

    source_path = Path(encoded.source.path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    midi = mido.MidiFile(str(source_path), clip=False)
    target_track = encoded.source.preferred_cc_track
    target_channel = encoded.source.preferred_cc_channel
    generated = decode_event_controls(encoded, two_slot=two_slot)

    rebuilt_tracks: list[mido.MidiTrack] = []
    for track_index, track in enumerate(midi.tracks):
        absolute = 0
        rows: list[tuple[int, int, int, mido.Message | mido.MetaMessage]] = []
        for order, message in enumerate(track):
            absolute += int(message.time)
            is_source_cc64 = (
                message.type == "control_change"
                and int(message.channel) != 9
                and int(message.control) == 64
            )
            if not is_source_cc64:
                rows.append((absolute, 0, order, message.copy(time=0)))
        if track_index == target_track:
            for order, control in enumerate(generated):
                # Initial SET precedes tau=0 SET, and both precede source notes.
                priority = -2 if control.kind == "initial" else -1
                rows.append(
                    (
                        control.tick,
                        priority,
                        order,
                        mido.Message(
                            "control_change",
                            channel=target_channel,
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
        raise AssertionError("round-trip MIDI changed the non-pedal note signature")
    return output
