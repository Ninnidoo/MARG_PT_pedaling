from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import mido

from src.stage2_event_tokenizer.decoder_v1 import (
    decode_controls_v1,
    safe_main_tau,
    safe_terminal_gap_from_log,
    write_roundtrip_midi_v1,
)
from src.stage2_event_tokenizer.tokenizer import apply_tokens, parse_raw_midi, quantize_cc64
from src.stage2_event_tokenizer.tokenizer_v1 import (
    MAIN_CAPACITY,
    TERMINAL_CAPACITY,
    EventToken,
    NonPositiveFinalMainIntervalError,
    apply_terminal_tokens,
    encode_midi_v1,
    pad_slots,
    retain_last,
)


def fixture_midi(path: Path, *, zero_length_final_note: bool = False) -> Path:
    midi = mido.MidiFile(ticks_per_beat=100)
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
    rows = [
        (50, mido.Message("control_change", channel=0, control=64, value=51, time=0)),
        (100, mido.Message("note_on", channel=0, note=60, velocity=80, time=0)),
        (100, mido.Message("control_change", channel=0, control=64, value=79, time=0)),
        (180, mido.Message("note_off", channel=0, note=60, velocity=0, time=0)),
        (200, mido.Message("note_on", channel=0, note=64, velocity=70, time=0)),
        (200, mido.Message("note_on", channel=0, note=67, velocity=75, time=0)),
        (200, mido.Message("control_change", channel=0, control=64, value=0, time=0)),
        (200 if zero_length_final_note else 300, mido.Message("note_off", channel=0, note=64, velocity=0, time=0)),
        (200 if zero_length_final_note else 300, mido.Message("note_off", channel=0, note=67, velocity=0, time=0)),
    ]
    if not zero_length_final_note:
        rows += [
            (300, mido.Message("control_change", channel=0, control=64, value=127, time=0)),
            (310, mido.Message("control_change", channel=0, control=64, value=0, time=0)),
            (320, mido.Message("control_change", channel=0, control=64, value=79, time=0)),
            (330, mido.Message("control_change", channel=0, control=64, value=127, time=0)),
            (340, mido.Message("control_change", channel=0, control=64, value=51, time=0)),
            (350, mido.Message("control_change", channel=0, control=64, value=0, time=0)),
            (360, mido.Message("control_change", channel=0, control=64, value=79, time=0)),
        ]
    rows.sort(key=lambda row: row[0])
    previous = 0
    for tick, message in rows:
        track.append(message.copy(time=tick - previous))
        previous = tick
    midi.tracks.append(track)
    midi.save(path)
    return path


class EventTokenizerV1Tests(unittest.TestCase):
    def test_state_boundaries(self) -> None:
        expected = {0: 0, 25: 0, 26: 1, 63: 1, 64: 2, 103: 2, 104: 3, 127: 3}
        self.assertEqual({value: quantize_cc64(value) for value in expected}, expected)

    def test_initial_interval_and_terminal_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            encoded = encode_midi_v1(fixture_midi(Path(directory) / "fixture.mid"))
            self.assertEqual(encoded.initial_state, 1)
            self.assertEqual(encoded.main_intervals[0].all_events[0].tau, 0.0)
            self.assertEqual(encoded.main_intervals[1].all_events[0].tau, 0.0)
            final = encoded.main_intervals[-1]
            self.assertTrue(final.is_final_main_interval)
            self.assertEqual(final.all_events[-1].source_tick, 300)
            self.assertEqual(final.all_events[-1].tau, 1.0)
            self.assertTrue(all(slot.source_tick is None or slot.source_tick > 300 for slot in encoded.terminal.slots))

    def test_main_last6_and_final_state(self) -> None:
        events = tuple(EventToken((index % 4) + 1, index / 10, index) for index in range(8))
        retained = retain_last(events, MAIN_CAPACITY)
        self.assertEqual([event.source_tick for event in retained], list(range(2, 8)))
        self.assertEqual(apply_tokens(0, events), apply_tokens(0, retained))
        self.assertEqual(len(pad_slots(events, MAIN_CAPACITY, EventToken(0, None))), 6)

    def test_terminal_last4_reanchor_and_final_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            encoded = encode_midi_v1(fixture_midi(Path(directory) / "fixture.mid"))
            terminal = encoded.terminal
            retained_ticks = [slot.source_tick for slot in terminal.slots if slot.event_class]
            self.assertEqual(retained_ticks, [330, 340, 350, 360])
            first_gap = math.expm1(terminal.slots[0].log1p_gap or 0.0)
            self.assertAlmostEqual(first_gap, 0.15, places=10)
            self.assertEqual(
                apply_terminal_tokens(terminal.start_state, terminal.all_events),
                apply_terminal_tokens(terminal.start_state, terminal.slots),
            )
            self.assertEqual(len(terminal.slots), TERMINAL_CAPACITY)

    def test_log1p_roundtrip_and_decoder_safety(self) -> None:
        for gap in (0.0, 0.05, 0.5, 5.0, 26.704):
            self.assertAlmostEqual(math.expm1(math.log1p(gap)), gap, places=11)
        self.assertEqual(safe_terminal_gap_from_log(-10.0), 0.0)
        self.assertEqual(safe_main_tau(-0.2), 0.0)
        self.assertEqual(safe_main_tau(1.2), 1.0)

    def test_onset_group_representative_is_last_note(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            encoded = encode_midi_v1(fixture_midi(Path(directory) / "fixture.mid"))
            chord = encoded.onset_groups[1]
            self.assertEqual(chord.last_note_index, chord.representative_note_index)
            self.assertEqual(chord.last_note_index - chord.first_note_index + 1, 2)

    def test_nonpositive_final_interval_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = fixture_midi(Path(directory) / "zero.mid", zero_length_final_note=True)
            with self.assertRaises(NonPositiveFinalMainIntervalError):
                encode_midi_v1(path)

    def test_decoder_and_nonpedal_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = fixture_midi(Path(directory) / "fixture.mid")
            encoded = encode_midi_v1(source)
            controls = decode_controls_v1(encoded)
            terminal_ticks = [control.tick for control in controls if control.kind == "terminal"]
            self.assertEqual(terminal_ticks, [330, 340, 350, 360])
            output = write_roundtrip_midi_v1(encoded, Path(directory) / "roundtrip.mid")
            self.assertEqual(parse_raw_midi(source).note_signature, parse_raw_midi(output).note_signature)


if __name__ == "__main__":
    unittest.main()
