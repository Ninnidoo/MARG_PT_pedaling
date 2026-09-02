"""Frozen Custom Event Tokenizer v1.

This versioned implementation preserves the v0 parser and equal-timestamp
semantics while replacing the audit-only two-slot projection with the frozen
main-K6 and terminal-K4 representation.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import mido

from .tokenizer import (
    EVENT_NAMES,
    NONE_TOKEN,
    REPRESENTATIVES,
    STATE_NAMES,
    EffectiveEvent,
    EventToken,
    RawMidiPerformance,
    _effective_events,
    apply_tokens,
    parse_raw_midi,
    quantize_cc64,
)

TOKENIZER_VERSION = "custom_event_tokenizer_v1"
TOKENIZER_IMPLEMENTATION_VERSION = "1.0.1"
SAME_TIMESTAMP_RULE_VERSION = "v0_tick_track_message_last_effective_state"
MAIN_CAPACITY = 6
TERMINAL_CAPACITY = 4


class NonPositiveFinalMainIntervalError(ValueError):
    pass


@dataclass(frozen=True)
class OnsetGroup:
    onset_tick: int
    first_note_index: int
    last_note_index: int
    representative_note_index: int


@dataclass(frozen=True)
class MainIntervalV1:
    index: int
    left_tick: int
    right_tick: int
    is_final_main_interval: bool
    start_state: int
    all_events: tuple[EventToken, ...]
    slots: tuple[EventToken, ...]


@dataclass(frozen=True)
class TerminalEventTokenV1:
    event_class: int
    log1p_gap: float | None
    source_tick: int | None = None
    source_crossing: str | None = None

    @property
    def state(self) -> int | None:
        return None if self.event_class == 0 else self.event_class - 1


NONE_TERMINAL_TOKEN = TerminalEventTokenV1(0, None, None, None)


@dataclass(frozen=True)
class TerminalTokensV1:
    anchor_tick: int
    anchor_seconds: float
    start_state: int
    all_events: tuple[TerminalEventTokenV1, ...]
    slots: tuple[TerminalEventTokenV1, ...]


@dataclass(frozen=True)
class EncodedPerformanceV1:
    source: RawMidiPerformance
    note_order: tuple[tuple[int, int, int, int, int, int], ...]
    onset_groups: tuple[OnsetGroup, ...]
    initial_state: int
    effective_events: tuple[EffectiveEvent, ...]
    main_intervals: tuple[MainIntervalV1, ...]
    terminal: TerminalTokensV1

    @property
    def non_pedal_note_count(self) -> int:
        return len(self.note_order)


def retain_last(events: Sequence, capacity: int) -> tuple:
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    values = tuple(events)
    return values if len(values) <= capacity else values[-capacity:]


def pad_slots(events: Sequence, capacity: int, none_token) -> tuple:
    retained = retain_last(events, capacity)
    return retained + (none_token,) * (capacity - len(retained))


def canonical_note_order(source: RawMidiPerformance) -> tuple[tuple[int, int, int, int, int, int], ...]:
    # Raw note signature rows are (track, channel, pitch, onset, noteoff, velocity).
    return tuple(
        sorted(source.note_signature, key=lambda row: (row[3], row[2], row[0], row[1], row[4], row[5]))
    )


def onset_groups(source: RawMidiPerformance) -> tuple[OnsetGroup, ...]:
    notes = canonical_note_order(source)
    result: list[OnsetGroup] = []
    index = 0
    while index < len(notes):
        onset = notes[index][3]
        end = index
        while end + 1 < len(notes) and notes[end + 1][3] == onset:
            end += 1
        result.append(OnsetGroup(onset, index, end, end))
        index = end + 1
    return tuple(result)


def _tick_second_converters(path: str | Path):
    midi = mido.MidiFile(str(path), clip=False)
    tempo_messages: list[tuple[int, int, int, int]] = []
    for track_index, track in enumerate(midi.tracks):
        tick = 0
        for message_index, message in enumerate(track):
            tick += int(message.time)
            if message.type == "set_tempo":
                tempo_messages.append((tick, track_index, message_index, int(message.tempo)))
    grouped: dict[int, int] = {0: 500000}
    for tick, _, _, tempo in sorted(tempo_messages):
        grouped[tick] = tempo
    ticks = sorted(grouped)
    tempos = [grouped[tick] for tick in ticks]
    seconds = [0.0]
    for index in range(1, len(ticks)):
        seconds.append(
            seconds[-1]
            + mido.tick2second(ticks[index] - ticks[index - 1], midi.ticks_per_beat, tempos[index - 1])
        )

    def tick_to_seconds(tick: int) -> float:
        index = bisect.bisect_right(ticks, int(tick)) - 1
        return seconds[index] + mido.tick2second(
            int(tick) - ticks[index], midi.ticks_per_beat, tempos[index]
        )

    def seconds_to_tick(value: float) -> int:
        index = bisect.bisect_right(seconds, float(value)) - 1
        index = max(0, min(index, len(ticks) - 1))
        tick_delta = float(value) - seconds[index]
        ticks_per_second = midi.ticks_per_beat * 1_000_000.0 / tempos[index]
        return int(round(ticks[index] + tick_delta * ticks_per_second))

    return tick_to_seconds, seconds_to_tick


def apply_terminal_tokens(state: int, events: Iterable[TerminalEventTokenV1]) -> int:
    for event in events:
        if event.state is not None:
            state = event.state
    return state


def encode_performance_v1(source: RawMidiPerformance) -> EncodedPerformanceV1:
    effective = _effective_events(source.raw_cc64)
    groups = onset_groups(source)
    # v0 records every note_on in distinct_onsets even if a malformed source
    # leaves that note unmatched. A model target cannot give such an onset a
    # representative note index, so v1 uses complete non-pedal notes.
    onset_ticks = tuple(group.onset_tick for group in groups)
    first_onset = onset_ticks[0]
    last_onset = onset_ticks[-1]
    if source.latest_note_off <= last_onset:
        raise NonPositiveFinalMainIntervalError(
            f"latest note-off {source.latest_note_off} <= last onset {last_onset}: {source.path}"
        )

    initial_state = 0
    for event in effective:
        if event.tick < first_onset:
            initial_state = event.state
        else:
            break

    by_interval: list[list[EffectiveEvent]] = [list() for _ in onset_ticks]
    terminal_effective: list[EffectiveEvent] = []
    for event in effective:
        if event.tick < first_onset:
            continue
        if event.tick > source.latest_note_off:
            terminal_effective.append(event)
            continue
        interval_index = bisect.bisect_right(onset_ticks, event.tick) - 1
        if interval_index < 0:
            raise AssertionError("pre-onset event escaped initial state")
        by_interval[interval_index].append(event)

    intervals: list[MainIntervalV1] = []
    state = initial_state
    for index, left in enumerate(onset_ticks):
        final = index == len(onset_ticks) - 1
        right = source.latest_note_off if final else onset_ticks[index + 1]
        denominator = right - left
        if denominator <= 0:
            raise NonPositiveFinalMainIntervalError(
                f"non-positive main interval [{left},{right}]: {source.path}"
            )
        tokens: list[EventToken] = []
        for event in by_interval[index]:
            tau = (event.tick - left) / denominator
            if final:
                if not (-1e-12 <= tau <= 1.0 + 1e-12):
                    raise ValueError(f"final main tau outside [0,1]: {tau}")
            elif not (-1e-12 <= tau < 1.0):
                raise ValueError(f"non-terminal tau outside [0,1): {tau}")
            tokens.append(EventToken(event.state + 1, float(tau), event.tick, event.crossing))
        retained = retain_last(tokens, MAIN_CAPACITY)
        if apply_tokens(state, tokens) != apply_tokens(state, retained):
            raise AssertionError("main last-6 compression changed interval final state")
        intervals.append(
            MainIntervalV1(
                index=index,
                left_tick=left,
                right_tick=right,
                is_final_main_interval=final,
                start_state=state,
                all_events=tuple(tokens),
                slots=pad_slots(tokens, MAIN_CAPACITY, NONE_TOKEN),
            )
        )
        state = apply_tokens(state, tokens)

    tick_to_seconds, _ = _tick_second_converters(source.path)
    anchor_seconds = tick_to_seconds(source.latest_note_off)
    terminal_all: list[TerminalEventTokenV1] = []
    previous_seconds = anchor_seconds
    for event in terminal_effective:
        event_seconds = tick_to_seconds(event.tick)
        gap = event_seconds - previous_seconds
        if gap < -1e-12:
            raise AssertionError("terminal events are not chronological")
        terminal_all.append(
            TerminalEventTokenV1(
                event.state + 1,
                math.log1p(max(0.0, gap)),
                event.tick,
                event.crossing,
            )
        )
        previous_seconds = event_seconds

    retained_effective = retain_last(terminal_effective, TERMINAL_CAPACITY)
    terminal_retained: list[TerminalEventTokenV1] = []
    previous_seconds = anchor_seconds
    for event in retained_effective:
        event_seconds = tick_to_seconds(event.tick)
        gap = event_seconds - previous_seconds
        if gap < -1e-12:
            raise AssertionError("retained terminal events are not chronological")
        terminal_retained.append(
            TerminalEventTokenV1(event.state + 1, math.log1p(max(0.0, gap)), event.tick, event.crossing)
        )
        previous_seconds = event_seconds
    if apply_terminal_tokens(state, terminal_all) != apply_terminal_tokens(state, terminal_retained):
        raise AssertionError("terminal last-4 compression changed final state")
    terminal = TerminalTokensV1(
        anchor_tick=source.latest_note_off,
        anchor_seconds=anchor_seconds,
        start_state=state,
        all_events=tuple(terminal_all),
        slots=tuple(terminal_retained) + (NONE_TERMINAL_TOKEN,) * (TERMINAL_CAPACITY - len(terminal_retained)),
    )
    return EncodedPerformanceV1(
        source=source,
        note_order=canonical_note_order(source),
        onset_groups=groups,
        initial_state=initial_state,
        effective_events=effective,
        main_intervals=tuple(intervals),
        terminal=terminal,
    )


def encode_midi_v1(path: str | Path) -> EncodedPerformanceV1:
    return encode_performance_v1(parse_raw_midi(path))
