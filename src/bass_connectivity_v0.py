"""Fixed MIDI-only Structural Bass detector and Bass Connectivity v0.

This module deliberately has no model, torch, audio, or CUDA dependency.  Event
ordering follows the merged-track, sequential same-timestamp convention used by
the Harmonic Muddiness validation code.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import mido


GROUP_WINDOW_SECONDS = 0.020
LOCAL_IOI_RADIUS = 8
NEXT_LOW_MAX_PITCH = 57
R_LOW = 1.5
R_HIGH = 3.0
BETA_OCTAVE = 0.2
STRUCTURAL_BASS_THRESHOLD = 0.5


@dataclass(slots=True)
class NoteAttack:
    identifier: int
    channel: int
    pitch: int
    velocity: int
    onset_tick: int
    onset_seconds: float
    event_order: int
    keyoff_tick: int | None = None
    keyoff_seconds: float | None = None
    keyoff_event_order: int | None = None
    pedal_on_at_keyoff: bool | None = None
    first_pedal_off_tick: int | None = None
    first_pedal_off_seconds: float | None = None
    first_pedal_off_event_order: int | None = None
    keyoff_source: str | None = None


@dataclass(slots=True)
class OnsetGroup:
    index: int
    anchor_seconds: float
    anchor_tick: int
    attacks: list[NoteAttack]
    pitches: tuple[int, ...]
    lowest_pitch: int
    lowest_attack: NoteAttack
    local_median_ioi: float | None = None
    local_positive_ioi_count: int = 0
    next_low_group_index: int | None = None
    next_low_gap_seconds: float | None = None
    r_b: float | None = None
    lowness: float = 0.0
    structural_timescale: float | None = None
    octave_bonus: int = 0
    salience: float | None = None
    is_structural_bass: bool = False


@dataclass(slots=True)
class ParsedMidi:
    attacks: list[NoteAttack]
    diagnostics: dict[str, int]


@dataclass(slots=True)
class TransitionResult:
    status: str
    score: float | None
    same_bass: bool
    gap_seconds: float | None
    pedal_off_seconds: float | None
    keyoff_at_pedal_off: bool
    no_subsequent_pedal_off: bool


def is_cc64(message: Any) -> bool:
    return (
        not message.is_meta
        and message.type == "control_change"
        and int(message.control) == 64
    )


def is_note_on(message: Any) -> bool:
    return (
        not message.is_meta
        and message.type == "note_on"
        and int(message.velocity) > 0
    )


def is_note_off(message: Any) -> bool:
    return not message.is_meta and (
        message.type == "note_off"
        or (message.type == "note_on" and int(message.velocity) == 0)
    )


def merged_tick_groups(path: Path) -> tuple[int, list[tuple[int, float, list[Any]]]]:
    """Tempo-aware merged groups, matching Harmonic Muddiness semantics."""

    midi = mido.MidiFile(str(path), clip=False)
    merged = mido.merge_tracks(midi.tracks)
    tempo = 500_000
    tick = 0
    seconds = 0.0
    groups: list[tuple[int, float, list[Any]]] = []
    current_tick: int | None = None
    current_seconds = 0.0
    messages: list[Any] = []
    for message in merged:
        delta = int(message.time)
        tick += delta
        seconds += mido.tick2second(delta, midi.ticks_per_beat, tempo)
        if current_tick is None or tick != current_tick:
            if current_tick is not None:
                groups.append((current_tick, current_seconds, messages))
            current_tick = tick
            current_seconds = seconds
            messages = []
        messages.append(message.copy(time=0))
        if message.is_meta and message.type == "set_tempo":
            tempo = int(message.tempo)
    if current_tick is not None:
        groups.append((current_tick, current_seconds, messages))
    return midi.ticks_per_beat, groups


def parse_midi(path: Path) -> ParsedMidi:
    """Parse attacks, physical key-offs, and CC64 release semantics.

    At a shared timestamp, messages are processed sequentially in merged order.
    Thus a CC64-off before a note-off means pedal-off at key-off, whereas a
    CC64-off later at that timestamp is the note's first subsequent pedal-off.
    """

    _, tick_groups = merged_tick_groups(path)
    active: dict[tuple[int, int], deque[NoteAttack]] = defaultdict(deque)
    pending_pedal_release: dict[int, list[NoteAttack]] = defaultdict(list)
    pedal_on: dict[int, bool] = defaultdict(bool)
    attacks: list[NoteAttack] = []
    diagnostics: Counter[str] = Counter()
    identifier = 0
    event_order = 0

    def close_note(note: NoteAttack, tick: int, seconds: float, source: str, *, pedal_can_hold: bool = True) -> None:
        note.keyoff_tick = tick
        note.keyoff_seconds = seconds
        note.keyoff_event_order = event_order
        note.keyoff_source = source
        note.pedal_on_at_keyoff = bool(pedal_on[note.channel] and pedal_can_hold)
        if note.pedal_on_at_keyoff:
            pending_pedal_release[note.channel].append(note)

    for tick, seconds, messages in tick_groups:
        for message in messages:
            event_order += 1
            if is_cc64(message):
                channel = int(message.channel)
                new_state = int(message.value) >= 64
                if pedal_on[channel] and not new_state:
                    for note in pending_pedal_release[channel]:
                        note.first_pedal_off_tick = tick
                        note.first_pedal_off_seconds = seconds
                        note.first_pedal_off_event_order = event_order
                    diagnostics["notes_assigned_pedal_off"] += len(pending_pedal_release[channel])
                    pending_pedal_release[channel].clear()
                pedal_on[channel] = new_state
                diagnostics["cc64_events"] += 1
                continue

            if is_note_on(message):
                channel = int(message.channel)
                if channel == 9:
                    diagnostics["drum_note_on_ignored"] += 1
                    continue
                identifier += 1
                note = NoteAttack(
                    identifier=identifier,
                    channel=channel,
                    pitch=int(message.note),
                    velocity=int(message.velocity),
                    onset_tick=tick,
                    onset_seconds=seconds,
                    event_order=event_order,
                )
                attacks.append(note)
                active[(channel, note.pitch)].append(note)
                continue

            if is_note_off(message):
                channel, pitch = int(message.channel), int(message.note)
                if channel == 9:
                    diagnostics["drum_note_off_ignored"] += 1
                    continue
                key = (channel, pitch)
                if not active[key]:
                    diagnostics["unmatched_note_off"] += 1
                    continue
                close_note(active[key].popleft(), tick, seconds, "note_off")
                continue

            if (
                not message.is_meta
                and message.type == "control_change"
                and int(message.control) in (120, 123)
            ):
                channel = int(message.channel)
                control = int(message.control)
                affected = [
                    note
                    for key in list(active)
                    if key[0] == channel
                    for note in active.pop(key)
                ]
                for note in affected:
                    close_note(
                        note,
                        tick,
                        seconds,
                        f"cc{control}",
                        pedal_can_hold=(control == 123),
                    )
                if control == 120:
                    pending_pedal_release[channel].clear()
                diagnostics[f"cc{control}_events"] += 1
                diagnostics[f"cc{control}_notes_closed"] += len(affected)

    diagnostics["unmatched_active_note_on"] = sum(len(queue) for queue in active.values())
    diagnostics["no_subsequent_pedal_off_notes"] = sum(
        len(notes) for notes in pending_pedal_release.values()
    )
    if not attacks:
        raise RuntimeError(f"no non-drum note attacks: {path}")
    attacks.sort(key=lambda note: (note.onset_seconds, note.event_order))
    return ParsedMidi(attacks=attacks, diagnostics=dict(diagnostics))


def earliest_anchor_groups(
    attacks: Sequence[NoteAttack], window_seconds: float = GROUP_WINDOW_SECONDS
) -> list[list[NoteAttack]]:
    """Inclusive earliest-anchor grouping; deliberately not single linkage."""

    groups: list[list[NoteAttack]] = []
    index = 0
    while index < len(attacks):
        anchor = attacks[index].onset_seconds
        stop = index + 1
        while (
            stop < len(attacks)
            and attacks[stop].onset_seconds - anchor <= window_seconds + 1e-12
        ):
            stop += 1
        groups.append(list(attacks[index:stop]))
        index = stop
    return groups


def local_ioi_seconds(times: Sequence[float], onset_index: int) -> tuple[float | None, int]:
    """Median positive IOI over the existing audit's +/-8 group convention."""

    if len(times) < 2:
        return None, 0
    start = max(0, onset_index - LOCAL_IOI_RADIUS)
    stop = min(len(times) - 1, onset_index + LOCAL_IOI_RADIUS)
    values = [
        float(times[i + 1]) - float(times[i])
        for i in range(start, stop)
        if float(times[i + 1]) - float(times[i]) > 0
    ]
    return (float(statistics.median(values)), len(values)) if values else (None, 0)


def lowness(pitch: int) -> float:
    if pitch <= 52:
        return 1.0
    if pitch < 57:
        return (57.0 - pitch) / 5.0
    return 0.0


def structural_timescale(r_b: float) -> float:
    return min(1.0, max(0.0, (r_b - R_LOW) / (R_HIGH - R_LOW)))


def detect_structural_bass(parsed: ParsedMidi) -> list[OnsetGroup]:
    raw_groups = earliest_anchor_groups(parsed.attacks)
    times = [group[0].onset_seconds for group in raw_groups]
    if any(right <= left for left, right in zip(times, times[1:])):
        raise AssertionError("20-ms group anchors must be strictly increasing")

    groups: list[OnsetGroup] = []
    for index, attacks in enumerate(raw_groups):
        pitches = tuple(sorted(note.pitch for note in attacks))
        lowest_pitch = pitches[0]
        lowest_candidates = [note for note in attacks if note.pitch == lowest_pitch]
        lowest_attack = min(lowest_candidates, key=lambda note: (note.onset_seconds, note.event_order))
        local, count = local_ioi_seconds(times, index)
        groups.append(
            OnsetGroup(
                index=index,
                anchor_seconds=times[index],
                anchor_tick=attacks[0].onset_tick,
                attacks=attacks,
                pitches=pitches,
                lowest_pitch=lowest_pitch,
                lowest_attack=lowest_attack,
                local_median_ioi=local,
                local_positive_ioi_count=count,
            )
        )

    next_low: int | None = None
    for index in range(len(groups) - 1, -1, -1):
        group = groups[index]
        group.next_low_group_index = next_low
        if next_low is not None:
            group.next_low_gap_seconds = groups[next_low].anchor_seconds - group.anchor_seconds
        if group.lowest_pitch <= NEXT_LOW_MAX_PITCH:
            next_low = index

    for group in groups:
        group.lowness = lowness(group.lowest_pitch)
        group.octave_bonus = int(group.lowest_pitch + 12 in group.pitches)
        if group.next_low_gap_seconds is None or group.local_median_ioi is None:
            # Validated audit behavior: terminal/insufficient-context values stay
            # missing; they are never replaced by a sentinel or called structural.
            continue
        group.r_b = group.next_low_gap_seconds / group.local_median_ioi
        group.structural_timescale = structural_timescale(group.r_b)
        group.salience = min(
            1.0,
            max(
                0.0,
                group.lowness
                * group.structural_timescale
                * (1.0 + BETA_OCTAVE * group.octave_bonus),
            ),
        )
        group.is_structural_bass = group.salience >= STRUCTURAL_BASS_THRESHOLD
    return groups


def connectivity_score(
    *,
    keyoff_seconds: float | None,
    next_onset_seconds: float,
    current_pitch: int,
    next_pitch: int,
    pedal_on_at_keyoff: bool | None,
    first_pedal_off_seconds: float | None,
) -> TransitionResult:
    """Evaluate one consecutive Structural Bass pair under fixed v0 rules."""

    same_bass = current_pitch == next_pitch
    if keyoff_seconds is None or pedal_on_at_keyoff is None:
        return TransitionResult(
            "excluded_missing_keyoff", None, same_bass, None, None, False, False
        )
    if keyoff_seconds >= next_onset_seconds:
        return TransitionResult(
            "excluded_keyoff_at_or_after_next_bass",
            None,
            same_bass,
            next_onset_seconds - keyoff_seconds,
            first_pedal_off_seconds,
            False,
            first_pedal_off_seconds is None,
        )

    gap = next_onset_seconds - keyoff_seconds
    if not pedal_on_at_keyoff:
        return TransitionResult(
            "valid_keyoff_at_pedal_off", 0.0, same_bass, gap, None, True, False
        )

    if first_pedal_off_seconds is None:
        return TransitionResult(
            "valid_no_subsequent_pedal_off",
            1.0 if same_bass else 0.0,
            same_bass,
            gap,
            None,
            False,
            True,
        )

    pedal_off = first_pedal_off_seconds
    if same_bass and pedal_off >= next_onset_seconds:
        score = 1.0
    else:
        sigma = gap / 2.0 if pedal_off <= next_onset_seconds else gap / 4.0
        score = math.exp(-((pedal_off - next_onset_seconds) ** 2) / (2.0 * sigma**2))
    return TransitionResult(
        "valid_scored", score, same_bass, gap, pedal_off, False, False
    )


def structural_groups(groups: Iterable[OnsetGroup]) -> list[OnsetGroup]:
    return [group for group in groups if group.is_structural_bass]

