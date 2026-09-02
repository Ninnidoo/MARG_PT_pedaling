from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import mido

from src.stage2_binary.canonical_stage1 import (
    assert_strict_non_cc64_equality,
    cc64_schedule,
    sha256_file,
    transplant_cc64_only,
)


def _write_fixture(path: Path, cc64: list[tuple[int, int]]) -> None:
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    metadata = mido.MidiTrack()
    metadata.extend(
        [
            mido.MetaMessage("track_name", name="Metadata", time=0),
            mido.MetaMessage("set_tempo", tempo=500000, time=0),
            mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0),
            mido.MetaMessage("key_signature", key="C", time=0),
            mido.MetaMessage("marker", text="A", time=120),
            mido.MetaMessage("lyrics", text="fixture", time=120),
            mido.MetaMessage("end_of_track", time=720),
        ]
    )
    notes = mido.MidiTrack()
    notes.extend(
        [
            mido.MetaMessage("track_name", name="Piano", time=0),
            mido.Message("program_change", channel=0, program=0, time=0),
            mido.Message("control_change", channel=0, control=1, value=23, time=0),
            mido.Message("pitchwheel", channel=0, pitch=32, time=0),
            mido.Message("note_on", channel=0, note=60, velocity=91, time=0),
            mido.Message("note_off", channel=0, note=60, velocity=0, time=480),
            mido.Message("note_on", channel=0, note=64, velocity=73, time=0),
            mido.Message("note_off", channel=0, note=64, velocity=0, time=480),
            mido.MetaMessage("end_of_track", time=480),
        ]
    )
    # Insert CC64 at absolute ticks while retaining all fixture events.
    absolute = 0
    expanded = []
    pending = list(cc64)
    for message in notes:
        next_absolute = absolute + message.time
        while pending and pending[0][0] <= next_absolute:
            tick, value = pending.pop(0)
            expanded.append(mido.Message("control_change", channel=0, control=64, value=value, time=tick - absolute))
            absolute = tick
        expanded.append(message.copy(time=next_absolute - absolute))
        absolute = next_absolute
    notes[:] = expanded
    midi.tracks.extend([metadata, notes])
    midi.save(str(path))


class CanonicalStage1Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_synthetic_cc64_transplant_preserves_every_non_cc64_event(self) -> None:
        canonical = self.root / "canonical.mid"
        donor = self.root / "donor.mid"
        candidate = self.root / "candidate.mid"
        _write_fixture(canonical, [(0, 127), (720, 0)])
        _write_fixture(donor, [(120, 127), (840, 0)])
        before = sha256_file(canonical)
        result = transplant_cc64_only(
            canonical, donor, candidate, require_cc64_difference=True
        )
        self.assertTrue(result["all_ordered_non_cc64_events_exact"])
        self.assertTrue(result["cc64_changed"])
        self.assertEqual(cc64_schedule(candidate), cc64_schedule(donor))
        self.assertEqual(sha256_file(canonical), before)

    def test_strict_equality_hard_fails_for_velocity_change(self) -> None:
        canonical = self.root / "canonical.mid"
        candidate = self.root / "mutated.mid"
        _write_fixture(canonical, [(0, 127), (720, 0)])
        midi = mido.MidiFile(str(canonical))
        note = next(message for track in midi.tracks for message in track if message.type == "note_on")
        note.velocity += 1
        midi.save(str(candidate))
        with self.assertRaisesRegex(AssertionError, "non-CC64 MIDI invariant"):
            assert_strict_non_cc64_equality(canonical, candidate)

    def test_no_pedal_canonical_accepts_donor_schedule(self) -> None:
        canonical = self.root / "no_pedal.mid"
        donor = self.root / "donor.mid"
        candidate = self.root / "candidate.mid"
        _write_fixture(canonical, [])
        _write_fixture(donor, [(0, 127), (480, 0)])
        result = transplant_cc64_only(
            canonical, donor, candidate, require_cc64_difference=True
        )
        self.assertEqual(result["canonical_cc64_events"], 0)
        self.assertEqual(result["candidate_cc64_events"], 2)
        self.assertTrue(result["all_ordered_non_cc64_events_exact"])


if __name__ == "__main__":
    unittest.main()
