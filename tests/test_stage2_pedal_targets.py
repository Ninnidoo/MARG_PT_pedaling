from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import mido


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analyze_stage2_pedal_targets.py"
)
SPEC = importlib.util.spec_from_file_location("stage2_pedal_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def cc(tick: int, value: int, time_sec: float | None = None) -> object:
    return audit.CCEvent(
        tick=tick,
        time_sec=float(tick if time_sec is None else time_sec),
        track_index=0,
        message_index=tick,
        channel=0,
        value=value,
    )


class Stage2PedalTargetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.baseline = audit.STATE_MACHINES[0]
        self.narrow = audit.STATE_MACHINES[1]
        self.onsets = [0, 100, 200]

    def assigned_types(self, events: list[object]) -> list[list[str]]:
        transitions = audit.detect_transitions(events, self.baseline)
        assignments, _, _ = audit.assign_transition_intervals(
            transitions, self.onsets
        )
        return [
            [transitions[index].event_type for index in interval]
            for interval in assignments
        ]

    def test_no_transition(self) -> None:
        self.assertEqual(self.assigned_types([cc(10, 10), cc(50, 40)]), [[], []])

    def test_one_down(self) -> None:
        self.assertEqual(self.assigned_types([cc(50, 70)]), [["DOWN"], []])

    def test_one_up(self) -> None:
        self.assertEqual(
            self.assigned_types([cc(-10, 70), cc(50, 20)]),
            [["UP"], []],
        )

    def test_up_down_repedal(self) -> None:
        self.assertEqual(
            self.assigned_types([cc(-10, 100), cc(30, 20), cc(60, 100)]),
            [["UP", "DOWN"], []],
        )

    def test_three_transitions_overflow(self) -> None:
        self.assertEqual(
            self.assigned_types([cc(10, 100), cc(30, 20), cc(60, 100)]),
            [["DOWN", "UP", "DOWN"], []],
        )

    def test_event_at_onset_boundary_belongs_to_next_ioi(self) -> None:
        self.assertEqual(
            self.assigned_types([cc(100, 100)]),
            [[], ["DOWN"]],
        )

    def test_simultaneous_note_onsets_are_grouped(self) -> None:
        ticks = sorted({0, 0, 100, 100, 200})
        self.assertEqual(ticks, [0, 100, 200])
        assignments, _, _ = audit.assign_transition_intervals([], ticks)
        self.assertEqual(len(assignments), 2)

    def test_63_65_jitter_crosses_baseline(self) -> None:
        events = [cc(10, 63), cc(20, 65), cc(30, 63), cc(40, 65)]
        self.assertEqual(
            self.assigned_types(events),
            [["DOWN", "UP", "DOWN"], []],
        )

    def test_narrow_hysteresis_state_transition(self) -> None:
        events = [cc(10, 65), cc(20, 69), cc(30, 63), cc(40, 60)]
        transitions = audit.detect_transitions(events, self.narrow)
        self.assertEqual(
            [event.event_type for event in transitions], ["DOWN", "UP"]
        )
        self.assertEqual([event.tick for event in transitions], [20, 40])

    def test_hysteresis_timing_matches_latest_crossing_in_gesture(self) -> None:
        events = [
            cc(10, 65),
            cc(20, 63),
            cc(30, 65),
            cc(40, 69),
            cc(50, 63),
            cc(60, 65),
            cc(70, 60),
        ]
        baseline = audit.detect_transitions(events, self.baseline)
        narrow = audit.detect_transitions(events, self.narrow)
        matches = audit.ordered_timing_matches(baseline, narrow)
        self.assertEqual(matches["matched_count"], 2)
        self.assertEqual(matches["unmatched_baseline_count"], 4)
        self.assertEqual(matches["unmatched_comparison_count"], 0)
        self.assertEqual(matches["_signed_ms"], [10_000.0, 0.0])

    def test_tempo_change_midi_uses_real_seconds(self) -> None:
        midi = mido.MidiFile(type=1, ticks_per_beat=480)
        tempo_track = mido.MidiTrack()
        note_track = mido.MidiTrack()
        midi.tracks.extend([tempo_track, note_track])
        tempo_track.append(mido.MetaMessage("set_tempo", tempo=500_000, time=0))
        tempo_track.append(
            mido.MetaMessage("set_tempo", tempo=1_000_000, time=480)
        )
        note_track.append(
            mido.Message("note_on", note=60, velocity=80, channel=0, time=0)
        )
        note_track.append(
            mido.Message(
                "control_change",
                control=64,
                value=100,
                channel=0,
                time=480,
            )
        )
        note_track.append(
            mido.Message("note_on", note=62, velocity=80, channel=0, time=480)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tempo_change.mid"
            midi.save(path)
            parsed = audit.parse_midi(path)
        self.assertAlmostEqual(parsed.cc_events[0].time_sec, 0.5, places=9)
        self.assertAlmostEqual(parsed.note_onset_times_sec[1], 1.5, places=9)


if __name__ == "__main__":
    unittest.main()
