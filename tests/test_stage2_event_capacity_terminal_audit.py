from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import mido

from src.stage2_event_tokenizer.capacity_terminal_audit import (
    CAPACITIES,
    VARIANT_ORDER,
    assert_final_state_preserved,
    post_noteoff_events,
    retain_last_k,
    retained_events,
    terminal_coverage,
)
from src.stage2_event_tokenizer.tokenizer import EventToken, encode_performance, parse_raw_midi


def tokens(count: int) -> tuple[EventToken, ...]:
    return tuple(EventToken((index % 4) + 1, index / max(count, 1), index) for index in range(count))


def interval(count: int, start_state: int = 0):
    values = tokens(count)
    if count == 0:
        old_slots = (EventToken(0, None), EventToken(0, None))
    elif count == 1:
        old_slots = (values[0], EventToken(0, None))
    elif count == 2:
        old_slots = values
    elif count % 2:
        old_slots = (values[-1], EventToken(0, None))
    else:
        old_slots = values[-2:]
    return SimpleNamespace(all_events=values, slots=old_slots, start_state=start_state)


def post_noteoff_midi(path: Path) -> Path:
    midi = mido.MidiFile(ticks_per_beat=100)
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
    track.append(mido.Message("note_on", channel=0, note=60, velocity=80, time=0))
    track.append(mido.Message("note_off", channel=0, note=60, velocity=0, time=100))
    track.append(mido.Message("control_change", channel=0, control=64, value=127, time=10))
    track.append(mido.Message("control_change", channel=0, control=64, value=0, time=10))
    midi.tracks.append(track)
    midi.save(path)
    return path


class CapacityTerminalAuditTests(unittest.TestCase):
    def test_last_k_exact_for_all_capacities(self) -> None:
        for capacity in CAPACITIES:
            for count in range(13):
                values = tokens(count)
                expected = values if count <= capacity else values[-capacity:]
                self.assertEqual(retain_last_k(values, capacity), expected)

    def test_last_k_chronological_order(self) -> None:
        values = tokens(10)
        for capacity in CAPACITIES:
            retained = retain_last_k(values, capacity)
            self.assertEqual([event.source_tick for event in retained], list(range(10 - capacity, 10)))

    def test_all_variants_preserve_final_state(self) -> None:
        for count in range(20):
            item = interval(count)
            for variant in VARIANT_ORDER:
                assert_final_state_preserved(item, variant)

    def test_unlimited_exact_event_sequence(self) -> None:
        item = interval(11)
        self.assertEqual(retained_events(item, "unlimited"), item.all_events)

    def test_k2_v0_odd_even_regression(self) -> None:
        self.assertEqual([event.source_tick for event in retained_events(interval(3), "k2_v0_odd_even")], [2])
        self.assertEqual([event.source_tick for event in retained_events(interval(4), "k2_v0_odd_even")], [2, 3])
        self.assertEqual([event.source_tick for event in retained_events(interval(5), "k2_v0_odd_even")], [4])
        self.assertEqual([event.source_tick for event in retained_events(interval(6), "k2_v0_odd_even")], [4, 5])

    def test_post_noteoff_delays_are_nonnegative(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = parse_raw_midi(post_noteoff_midi(Path(directory) / "tail.mid"))
            events = post_noteoff_events(encode_performance(source))
            self.assertEqual(len(events), 2)
            self.assertTrue(all(event.delay_seconds >= 0 for event in events))
            self.assertAlmostEqual(events[0].delay_seconds, 0.05, places=9)
            self.assertAlmostEqual(events[1].delay_seconds, 0.10, places=9)

    def test_terminal_coverage_monotonic(self) -> None:
        rows = terminal_coverage(((0.02, 0.3), (0.1,), (), (1.2,)), (0.05, 0.1, 0.5, 2.0))
        events = [float(row["event_coverage"]) for row in rows]
        performances = [float(row["affected_performance_full_coverage"]) for row in rows]
        self.assertEqual(events, sorted(events))
        self.assertEqual(performances, sorted(performances))
        self.assertEqual(events[-1], 1.0)
        self.assertEqual(performances[-1], 1.0)


if __name__ == "__main__":
    unittest.main()
