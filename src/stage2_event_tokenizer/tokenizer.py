"""Absolute-state CC64 event tokenizer used by the v0 representation audit.

Raw MIDI messages are read with :mod:`mido` so that file track/message order is
available.  Equal-tick CC64 messages are ordered by ``(track, message_index)``;
their last value is the deterministic effective value.  Ordering between
different MIDI tracks is not semantically defined by SMF and is therefore
reported as ambiguous by the audit rather than represented by a new token.
"""

from __future__ import annotations

import bisect
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import mido

STATE_NAMES = ("ZERO", "LOW", "HALF", "FULL")
EVENT_NAMES = ("NONE", "SET_ZERO", "SET_LOW", "SET_HALF", "SET_FULL")
REPRESENTATIVES = (0, 51, 79, 127)


def quantize_cc64(value: int) -> int:
    """Map raw CC64 to the immutable canonical four-state index."""

    value = int(value)
    if not 0 <= value <= 127:
        raise ValueError(f"CC64 value outside [0,127]: {value}")
    if value <= 25:
        return 0
    if value <= 63:
        return 1
    if value <= 103:
        return 2
    return 3


def crossing_direction(before: int, after: int) -> str | None:
    """Return the frozen evaluator's historical direction naming."""

    if before < 2 <= after:
        return "DOWN"
    if before >= 2 > after:
        return "UP"
    return None


@dataclass(frozen=True)
class RawCC64Event:
    tick: int
    value: int
    track: int
    channel: int
    message_index: int
    onset_relation: str


@dataclass(frozen=True)
class EffectiveEvent:
    tick: int
    state: int
    value: int
    source_count_at_tick: int
    source_track: int
    source_channel: int
    source_message_index: int
    previous_state: int
    crossing: str | None
    onset_relation: str


@dataclass(frozen=True)
class EventToken:
    event_class: int  # NONE=0, SET state s = s+1
    tau: float | None
    source_tick: int | None = None
    source_crossing: str | None = None

    @property
    def state(self) -> int | None:
        return None if self.event_class == 0 else self.event_class - 1


NONE_TOKEN = EventToken(0, None, None, None)


@dataclass(frozen=True)
class IntervalTokens:
    index: int
    start_tick: int
    end_tick: int
    terminal: bool
    start_state: int
    all_events: tuple[EventToken, ...]
    slots: tuple[EventToken, EventToken]


@dataclass(frozen=True)
class RawMidiPerformance:
    path: str
    ticks_per_beat: int
    distinct_onsets: tuple[int, ...]
    latest_note_off: int
    note_signature: tuple[tuple[int, int, int, int, int, int], ...]
    raw_cc64: tuple[RawCC64Event, ...]
    preferred_cc_track: int
    preferred_cc_channel: int


@dataclass(frozen=True)
class EncodedPerformance:
    source: RawMidiPerformance
    initial_state: int
    effective_events: tuple[EffectiveEvent, ...]
    intervals: tuple[IntervalTokens, ...]
    same_state_suppressed_count: int
    same_timestamp_collapsed_count: int
    raw_events_exactly_on_onset: int
    effective_events_exactly_on_onset: int
    first_onset_tau0_event_count: int
    same_time_cc_before_note: int
    same_time_cc_after_note: int
    same_time_order_ambiguous: int
    raw_events_after_last_onset: int
    effective_events_after_last_onset: int
    raw_events_after_note_off: int
    effective_events_after_note_off: int
    terminal_denominator_nonpositive: bool

    @property
    def unlimited(self) -> tuple[tuple[EventToken, ...], ...]:
        return tuple(interval.all_events for interval in self.intervals)

    @property
    def two_slot(self) -> tuple[tuple[EventToken, ...], ...]:
        return tuple(
            tuple(slot for slot in interval.slots if slot.event_class != 0)
            for interval in self.intervals
        )


def _absolute_track_messages(midi: mido.MidiFile):
    for track_index, track in enumerate(midi.tracks):
        tick = 0
        for message_index, message in enumerate(track):
            tick += int(message.time)
            yield track_index, message_index, tick, message


def parse_raw_midi(path: str | Path) -> RawMidiPerformance:
    """Parse notes and original CC64 messages without normalizing timestamps."""

    midi_path = Path(path)
    midi = mido.MidiFile(str(midi_path), clip=False)
    onsets: set[int] = set()
    note_on_positions: dict[tuple[int, int], list[int]] = defaultdict(list)
    open_notes: dict[tuple[int, int, int], deque[tuple[int, int]]] = defaultdict(deque)
    notes: list[tuple[int, int, int, int, int, int]] = []
    raw_pending: list[tuple[int, int, int, int, int]] = []
    first_note_target: tuple[int, int] | None = None

    for track, message_index, tick, message in _absolute_track_messages(midi):
        if not hasattr(message, "channel") or int(message.channel) == 9:
            continue
        channel = int(message.channel)
        if message.type == "note_on" and int(message.velocity) > 0:
            onsets.add(tick)
            note_on_positions[(track, tick)].append(message_index)
            if first_note_target is None:
                first_note_target = (track, channel)
            open_notes[(track, channel, int(message.note))].append(
                (tick, int(message.velocity))
            )
        elif message.type == "note_off" or (
            message.type == "note_on" and int(message.velocity) == 0
        ):
            key = (track, channel, int(message.note))
            if open_notes[key]:
                start, velocity = open_notes[key].popleft()
                notes.append((track, channel, int(message.note), start, tick, velocity))
        elif message.type == "control_change" and int(message.control) == 64:
            raw_pending.append((tick, track, message_index, channel, int(message.value)))

    if not onsets or not notes:
        raise ValueError(f"performance has no complete non-drum notes: {midi_path}")
    distinct_onsets = tuple(sorted(onsets))
    onset_set = set(distinct_onsets)
    raw_events: list[RawCC64Event] = []
    for tick, track, message_index, channel, value in sorted(raw_pending):
        relation = "not_on_onset"
        if tick in onset_set:
            positions = note_on_positions.get((track, tick), [])
            if not positions:
                relation = "ambiguous_cross_track"
            elif message_index < min(positions):
                relation = "before_note_on"
            elif message_index > max(positions):
                relation = "after_note_on"
            else:
                relation = "ambiguous_interleaved"
        raw_events.append(
            RawCC64Event(tick, value, track, channel, message_index, relation)
        )
    preferred = (
        (raw_events[0].track, raw_events[0].channel)
        if raw_events
        else (first_note_target or (0, 0))
    )
    return RawMidiPerformance(
        path=str(midi_path),
        ticks_per_beat=int(midi.ticks_per_beat),
        distinct_onsets=distinct_onsets,
        latest_note_off=max(note[4] for note in notes),
        note_signature=tuple(sorted(notes)),
        raw_cc64=tuple(raw_events),
        preferred_cc_track=int(preferred[0]),
        preferred_cc_channel=int(preferred[1]),
    )


def _effective_events(raw_events: Sequence[RawCC64Event]) -> tuple[EffectiveEvent, ...]:
    groups: dict[int, list[RawCC64Event]] = defaultdict(list)
    for event in raw_events:
        groups[event.tick].append(event)
    state = 0
    result: list[EffectiveEvent] = []
    for tick in sorted(groups):
        ordered = sorted(groups[tick], key=lambda event: (event.track, event.message_index))
        final = ordered[-1]
        next_state = quantize_cc64(final.value)
        if next_state != state:
            result.append(
                EffectiveEvent(
                    tick=tick,
                    state=next_state,
                    value=final.value,
                    source_count_at_tick=len(ordered),
                    source_track=final.track,
                    source_channel=final.channel,
                    source_message_index=final.message_index,
                    previous_state=state,
                    crossing=crossing_direction(state, next_state),
                    onset_relation=final.onset_relation,
                )
            )
            state = next_state
    return tuple(result)


def compress_events(events: Sequence[EventToken]) -> tuple[EventToken, EventToken]:
    """Apply the specified exact odd/even two-slot overflow rule."""

    events = tuple(events)
    count = len(events)
    if count == 0:
        return NONE_TOKEN, NONE_TOKEN
    if count == 1:
        return events[0], NONE_TOKEN
    if count == 2:
        return events[0], events[1]
    if count % 2:
        return events[-1], NONE_TOKEN
    return events[-2], events[-1]


def apply_tokens(state: int, events: Iterable[EventToken]) -> int:
    for event in events:
        if event.state is not None:
            state = event.state
    return state


def encode_performance(source: RawMidiPerformance) -> EncodedPerformance:
    effective = _effective_events(source.raw_cc64)
    first_onset = source.distinct_onsets[0]
    last_onset = source.distinct_onsets[-1]
    initial_state = 0
    for event in effective:
        if event.tick < first_onset:
            initial_state = event.state
        else:
            break

    by_interval: list[list[EffectiveEvent]] = [list() for _ in source.distinct_onsets]
    for event in effective:
        if event.tick < first_onset or event.tick > source.latest_note_off:
            continue
        interval_index = bisect.bisect_right(source.distinct_onsets, event.tick) - 1
        if interval_index < 0:
            raise AssertionError("strictly pre-onset event escaped initial state")
        by_interval[interval_index].append(event)

    intervals: list[IntervalTokens] = []
    state = initial_state
    for index, start in enumerate(source.distinct_onsets):
        terminal = index == len(source.distinct_onsets) - 1
        end = source.latest_note_off if terminal else source.distinct_onsets[index + 1]
        denominator = end - start
        tokens: list[EventToken] = []
        for event in by_interval[index]:
            if denominator < 0 or (denominator == 0 and event.tick != start):
                raise ValueError("event cannot be represented by non-positive interval")
            tau = 0.0 if denominator == 0 else (event.tick - start) / denominator
            if terminal:
                if not (-1e-12 <= tau <= 1.0 + 1e-12):
                    raise ValueError(f"terminal tau outside [0,1]: {tau}")
            elif not (-1e-12 <= tau < 1.0):
                raise ValueError(f"right-open interval tau outside [0,1): {tau}")
            tokens.append(EventToken(event.state + 1, float(tau), event.tick, event.crossing))
        slots = compress_events(tokens)
        original_end = apply_tokens(state, tokens)
        compressed_end = apply_tokens(state, slots)
        if original_end != compressed_end:
            raise AssertionError("two-slot compression changed interval final state")
        intervals.append(
            IntervalTokens(index, start, end, terminal, state, tuple(tokens), slots)
        )
        state = original_end

    onset_set = set(source.distinct_onsets)
    raw_on_onset = [event for event in source.raw_cc64 if event.tick in onset_set]
    effective_on_onset = [event for event in effective if event.tick in onset_set]
    return EncodedPerformance(
        source=source,
        initial_state=initial_state,
        effective_events=effective,
        intervals=tuple(intervals),
        same_state_suppressed_count=len(source.raw_cc64) - len(effective),
        same_timestamp_collapsed_count=len(source.raw_cc64) - len({event.tick for event in source.raw_cc64}),
        raw_events_exactly_on_onset=len(raw_on_onset),
        effective_events_exactly_on_onset=len(effective_on_onset),
        first_onset_tau0_event_count=sum(event.tick == first_onset for event in effective),
        same_time_cc_before_note=sum(event.onset_relation == "before_note_on" for event in raw_on_onset),
        same_time_cc_after_note=sum(event.onset_relation == "after_note_on" for event in raw_on_onset),
        same_time_order_ambiguous=sum(event.onset_relation.startswith("ambiguous") for event in raw_on_onset),
        raw_events_after_last_onset=sum(event.tick > last_onset for event in source.raw_cc64),
        effective_events_after_last_onset=sum(event.tick > last_onset for event in effective),
        raw_events_after_note_off=sum(event.tick > source.latest_note_off for event in source.raw_cc64),
        effective_events_after_note_off=sum(event.tick > source.latest_note_off for event in effective),
        terminal_denominator_nonpositive=source.latest_note_off <= last_onset,
    )
