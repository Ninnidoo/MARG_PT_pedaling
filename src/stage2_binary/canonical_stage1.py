"""Canonical ASAP-test Stage-1 and CC64-only MIDI helpers."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import mido


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized(value: Any) -> Any:
    if isinstance(value, bytes):
        return list(value)
    if isinstance(value, tuple):
        return [_normalized(item) for item in value]
    if isinstance(value, list):
        return [_normalized(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalized(item) for key, item in sorted(value.items())}
    return value


def _is_cc64(message: mido.Message | mido.MetaMessage) -> bool:
    return (
        not message.is_meta
        and message.type == "control_change"
        and int(message.control) == 64
    )


def _absolute_events(track: mido.MidiTrack) -> list[tuple[int, int, Any]]:
    absolute = 0
    events = []
    for order, message in enumerate(track):
        absolute += int(message.time)
        events.append((absolute, order, message))
    return events


def strict_non_cc64_signature(midi_path: str | Path) -> dict[str, Any]:
    """Return every ordered raw MIDI/meta event except sustain CC64."""

    midi = mido.MidiFile(str(midi_path), clip=False)
    tracks: list[list[dict[str, Any]]] = []
    for track in midi.tracks:
        messages = []
        for absolute, _, message in _absolute_events(track):
            if _is_cc64(message):
                continue
            payload = message.dict()
            payload.pop("time", None)
            messages.append(
                {"absolute_tick": absolute, "message": _normalized(payload)}
            )
        tracks.append(messages)
    return {
        "midi_type": midi.type,
        "ticks_per_beat": midi.ticks_per_beat,
        "tracks": tracks,
    }


def cc64_schedule(midi_path: str | Path) -> list[dict[str, int]]:
    midi = mido.MidiFile(str(midi_path), clip=False)
    schedule: list[dict[str, int]] = []
    for track_index, track in enumerate(midi.tracks):
        non_cc64_seen_at_tick: Counter[int] = Counter()
        for absolute, order, message in _absolute_events(track):
            if _is_cc64(message):
                schedule.append(
                    {
                        "track_index": track_index,
                        "absolute_tick": absolute,
                        "channel": int(message.channel),
                        "value": int(message.value),
                        "source_order": order,
                        "non_cc64_before_at_tick": non_cc64_seen_at_tick[absolute],
                    }
                )
            else:
                non_cc64_seen_at_tick[absolute] += 1
    return schedule


def assert_strict_non_cc64_equality(
    canonical_path: str | Path,
    candidate_path: str | Path,
    *,
    require_cc64_difference: bool = False,
) -> dict[str, Any]:
    """Hard-fail if any raw event other than sustain CC64 changed."""

    canonical = strict_non_cc64_signature(canonical_path)
    candidate = strict_non_cc64_signature(candidate_path)
    if canonical != candidate:
        checks = {
            "midi_type": canonical["midi_type"] == candidate["midi_type"],
            "ticks_per_beat": canonical["ticks_per_beat"]
            == candidate["ticks_per_beat"],
            "track_count": len(canonical["tracks"]) == len(candidate["tracks"]),
            "ordered_non_cc64_tracks": canonical["tracks"] == candidate["tracks"],
        }
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"canonical non-CC64 MIDI invariant failed: {failed}")
    canonical_cc64 = cc64_schedule(canonical_path)
    candidate_cc64 = cc64_schedule(candidate_path)
    changed = canonical_cc64 != candidate_cc64
    if require_cc64_difference and not changed:
        raise AssertionError("replacement was required to change CC64, but it did not")
    return {
        "passed": True,
        "canonical_sha256": sha256_file(canonical_path),
        "candidate_sha256": sha256_file(candidate_path),
        "midi_type_exact": True,
        "ticks_per_beat_exact": True,
        "track_count_exact": True,
        "all_ordered_non_cc64_events_exact": True,
        "canonical_cc64_events": len(canonical_cc64),
        "candidate_cc64_events": len(candidate_cc64),
        "cc64_changed": changed,
    }


def _events_by_tick(
    events: Iterable[tuple[int, int, Any]], *, include_cc64: bool
) -> dict[int, list[Any]]:
    grouped: dict[int, list[Any]] = defaultdict(list)
    for absolute, _, message in events:
        if include_cc64 == _is_cc64(message):
            grouped[absolute].append(message)
    return grouped


def transplant_cc64_only(
    canonical_path: str | Path,
    pedal_donor_path: str | Path,
    candidate_path: str | Path,
    *,
    require_cc64_difference: bool = False,
    discard_cc64_after_eot: bool = True,
) -> dict[str, Any]:
    """Copy donor CC64 events into a canonical MIDI without rebuilding it.

    Both files must share MIDI type, ticks-per-beat, and track count. Canonical
    non-CC64 messages are copied verbatim and retain their absolute ticks and
    relative order. Donor CC64 order at a shared tick is positioned using its
    count of preceding non-CC64 donor messages.
    """

    canonical_path = Path(canonical_path)
    donor_path = Path(pedal_donor_path)
    candidate_path = Path(candidate_path)
    if candidate_path.exists():
        raise FileExistsError(f"refusing to overwrite candidate: {candidate_path}")
    canonical_hash_before = sha256_file(canonical_path)
    canonical = mido.MidiFile(str(canonical_path), clip=False)
    donor = mido.MidiFile(str(donor_path), clip=False)
    if (
        canonical.type != donor.type
        or canonical.ticks_per_beat != donor.ticks_per_beat
        or len(canonical.tracks) != len(donor.tracks)
    ):
        raise ValueError("donor MIDI header/track layout is incompatible with canonical MIDI")

    output = copy.deepcopy(canonical)
    discarded_after_eot = 0
    for track_index, (canonical_track, donor_track) in enumerate(
        zip(canonical.tracks, donor.tracks, strict=True)
    ):
        canonical_events = _absolute_events(canonical_track)
        donor_events = _absolute_events(donor_track)
        base_by_tick = _events_by_tick(canonical_events, include_cc64=False)
        donor_cc_by_tick = _events_by_tick(donor_events, include_cc64=True)
        donor_slots: dict[int, list[tuple[int, Any]]] = defaultdict(list)
        non_cc_seen: Counter[int] = Counter()
        for absolute, _, message in donor_events:
            if _is_cc64(message):
                donor_slots[absolute].append((non_cc_seen[absolute], message))
            else:
                non_cc_seen[absolute] += 1

        canonical_eot = [
            absolute
            for absolute, _, message in canonical_events
            if message.is_meta and message.type == "end_of_track"
        ]
        if len(canonical_eot) != 1:
            raise ValueError(f"canonical track {track_index} must have exactly one EOT")
        beyond_eot = [tick for tick in donor_slots if tick > canonical_eot[0]]
        if beyond_eot and not discard_cc64_after_eot:
            raise ValueError(f"donor CC64 extends beyond canonical EOT in track {track_index}")
        for tick in beyond_eot:
            discarded_after_eot += len(donor_slots.pop(tick))

        rebuilt: list[tuple[int, Any]] = []
        all_ticks = sorted(set(base_by_tick) | set(donor_slots))
        for absolute in all_ticks:
            base = base_by_tick.get(absolute, [])
            by_slot: dict[int, list[Any]] = defaultdict(list)
            eot_index = next(
                (i for i, msg in enumerate(base) if msg.is_meta and msg.type == "end_of_track"),
                len(base),
            )
            for slot, message in donor_slots.get(absolute, []):
                by_slot[min(slot, eot_index)].append(message)
            for slot in range(len(base) + 1):
                for message in by_slot.get(slot, []):
                    rebuilt.append((absolute, message.copy(time=0)))
                if slot < len(base):
                    rebuilt.append((absolute, base[slot].copy(time=0)))

        new_track = mido.MidiTrack()
        previous = 0
        for absolute, message in rebuilt:
            new_track.append(message.copy(time=absolute - previous))
            previous = absolute
        output.tracks[track_index] = new_track

    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(str(candidate_path))
    if sha256_file(canonical_path) != canonical_hash_before:
        raise AssertionError("canonical MIDI was modified during CC64 transplant")
    result = assert_strict_non_cc64_equality(
        canonical_path,
        candidate_path,
        require_cc64_difference=require_cc64_difference,
    )
    result.update(
        {
            "canonical_path": str(canonical_path),
            "pedal_donor_path": str(donor_path),
            "candidate_path": str(candidate_path),
            "donor_cc64_events": len(cc64_schedule(donor_path)),
            "donor_cc64_discarded_after_canonical_eot": discarded_after_eot,
            "canonical_unchanged": True,
        }
    )
    return result


def signature_sha256(midi_path: str | Path) -> str:
    payload = json.dumps(
        strict_non_cc64_signature(midi_path), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
