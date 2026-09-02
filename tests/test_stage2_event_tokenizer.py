from __future__ import annotations

import random
from pathlib import Path

import mido
import pytest

from src.stage2_event_tokenizer.decoder import decode_event_controls, write_roundtrip_midi
from src.stage2_event_tokenizer.tokenizer import (
    EventToken,
    apply_tokens,
    compress_events,
    encode_performance,
    parse_raw_midi,
    quantize_cc64,
)


def _midi(
    path: Path,
    *,
    controls: list[tuple[int, int]],
    note_onsets: tuple[int, ...] = (10, 20, 30),
    note_end: int = 40,
    cc_before_note_at_equal_tick: bool = True,
) -> Path:
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    rows: list[tuple[int, int, mido.Message]] = []
    for order, (tick, value) in enumerate(controls):
        rows.append(
            (
                tick,
                0 if cc_before_note_at_equal_tick else 2,
                mido.Message("control_change", channel=0, control=64, value=value),
            )
        )
    for index, tick in enumerate(note_onsets):
        rows.append((tick, 1, mido.Message("note_on", channel=0, note=60 + index, velocity=70 + index)))
        rows.append((note_end, 1, mido.Message("note_off", channel=0, note=60 + index, velocity=0)))
    rows.sort(key=lambda row: (row[0], row[1]))
    last = 0
    for tick, _, message in rows:
        track.append(message.copy(time=tick - last))
        last = tick
    midi.tracks.append(track)
    midi.save(path)
    return path


def test_canonical_state_boundaries() -> None:
    assert [quantize_cc64(value) for value in (0, 25, 26, 63, 64, 103, 104, 127)] == [
        0,
        0,
        1,
        1,
        2,
        2,
        3,
        3,
    ]


def test_initial_strictness_first_onset_and_right_open(tmp_path: Path) -> None:
    path = _midi(tmp_path / "timing.mid", controls=[(5, 51), (10, 79), (20, 127)])
    encoded = encode_performance(parse_raw_midi(path))
    assert encoded.initial_state == 1
    assert [token.source_tick for token in encoded.intervals[0].all_events] == [10]
    assert encoded.intervals[0].all_events[0].tau == 0.0
    assert [token.source_tick for token in encoded.intervals[1].all_events] == [20]
    assert encoded.intervals[1].all_events[0].tau == 0.0
    assert encoded.first_onset_tau0_event_count == 1


def test_same_timestamp_last_effective_and_same_state_suppression(tmp_path: Path) -> None:
    path = _midi(
        tmp_path / "same.mid",
        controls=[(0, 8), (1, 18), (10, 51), (10, 79), (11, 90)],
    )
    encoded = encode_performance(parse_raw_midi(path))
    assert [(event.tick, event.state) for event in encoded.effective_events] == [(10, 2)]
    assert encoded.same_timestamp_collapsed_count == 1
    assert encoded.same_state_suppressed_count == 4
    assert encoded.raw_events_exactly_on_onset == 2
    assert encoded.effective_events_exactly_on_onset == 1


@pytest.mark.parametrize(
    ("count", "retained_indexes"),
    [(0, []), (1, [0]), (2, [0, 1]), (3, [2]), (4, [2, 3]), (5, [4]), (6, [4, 5])],
)
def test_exact_slot_and_overflow_rules(count: int, retained_indexes: list[int]) -> None:
    events = [EventToken((index % 4) + 1, index / 10, index) for index in range(count)]
    retained = [event for event in compress_events(events) if event.event_class]
    assert [event.source_tick for event in retained] == retained_indexes


def test_random_final_state_preservation() -> None:
    rng = random.Random(42)
    for count in range(30):
        for _ in range(100):
            start = rng.randrange(4)
            state = start
            events = []
            for index in range(count):
                choices = [value for value in range(4) if value != state]
                state = rng.choice(choices)
                events.append(EventToken(state + 1, index / max(count, 1), index))
            assert apply_tokens(start, events) == apply_tokens(start, compress_events(events))


def test_unlimited_exact_tick_roundtrip_and_nonpedal_identity(tmp_path: Path) -> None:
    path = _midi(
        tmp_path / "source.mid",
        controls=[(0, 51), (10, 79), (11, 127), (12, 51), (19, 0), (20, 79), (40, 127), (41, 0)],
    )
    encoded = encode_performance(parse_raw_midi(path))
    unlimited = decode_event_controls(encoded, two_slot=False)
    modeled_ticks = [event.tick for event in encoded.effective_events if 10 <= event.tick <= 40]
    assert [control.tick for control in unlimited if control.kind == "event"] == modeled_ticks
    output = write_roundtrip_midi(encoded, tmp_path / "roundtrip.mid", two_slot=False)
    assert parse_raw_midi(output).note_signature == encoded.source.note_signature
    assert encoded.raw_events_after_note_off == 1
    assert encoded.effective_events_after_note_off == 1


def test_no_cc_and_cc_only_before_first_onset(tmp_path: Path) -> None:
    no_cc = encode_performance(parse_raw_midi(_midi(tmp_path / "none.mid", controls=[])))
    assert no_cc.initial_state == 0
    assert all(not interval.all_events for interval in no_cc.intervals)
    before = encode_performance(
        parse_raw_midi(_midi(tmp_path / "before.mid", controls=[(1, 121)]))
    )
    assert before.initial_state == 3
    assert all(not interval.all_events for interval in before.intervals)


def test_same_tick_note_order_diagnostic(tmp_path: Path) -> None:
    before = encode_performance(
        parse_raw_midi(_midi(tmp_path / "before_note.mid", controls=[(10, 127)]))
    )
    after = encode_performance(
        parse_raw_midi(
            _midi(
                tmp_path / "after_note.mid",
                controls=[(10, 127)],
                cc_before_note_at_equal_tick=False,
            )
        )
    )
    assert before.same_time_cc_before_note == 1
    assert after.same_time_cc_after_note == 1


def test_short_terminal_interval_and_exact_noteoff(tmp_path: Path) -> None:
    encoded = encode_performance(
        parse_raw_midi(
            _midi(
                tmp_path / "terminal.mid",
                controls=[(30, 51), (31, 127)],
                note_onsets=(10, 20, 30),
                note_end=31,
            )
        )
    )
    terminal = encoded.intervals[-1]
    assert [token.tau for token in terminal.all_events] == [0.0, 1.0]
    assert not encoded.terminal_denominator_nonpositive
