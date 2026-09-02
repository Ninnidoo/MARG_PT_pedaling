from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
from miditoolkit import Instrument, MidiFile, Note, TempoChange

from src.stage2_event_model.note_alignment import (
    NoteAlignmentError,
    build_note_alignment,
    load_note_alignment,
    save_note_alignment,
)
from src.stage2_event_model.ownership import assign_unique_owners
from src.stage2_event_tokenizer.tokenizer import parse_raw_midi
from src.stage2_event_tokenizer.tokenizer_v1 import canonical_note_order


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class NoteAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def write_case(self, notes, tempo=500000):
        path = self.root / "case.mid"
        midi = MidiFile(ticks_per_beat=480)
        midi.tempo_changes = [TempoChange(60_000_000 / tempo, 0)]
        instrument = Instrument(0, is_drum=False, name="Piano")
        instrument.notes = [Note(v, p, s, e) for p, s, e, v in notes]
        midi.instruments = [instrument]
        midi.dump(str(path))
        rows = canonical_note_order(parse_raw_midi(path))
        cache = self.root / "cache.npz"
        np.savez_compressed(
            cache,
            note_onset_ticks=np.asarray([r[3] for r in rows], dtype=np.int64),
            note_pitch=np.asarray([r[2] for r in rows], dtype=np.int16),
            note_offset_ticks=np.asarray([r[4] for r in rows], dtype=np.int64),
            note_velocity=np.asarray([r[5] for r in rows], dtype=np.int16),
        )
        return path, cache, rows

    def test_identity_order_and_inverse_bijection(self):
        midi, cache, _ = self.write_case([(60, 0, 100, 80), (64, 120, 200, 70)])
        alignment = build_note_alignment(midi, cache)
        self.assertEqual(alignment.cache_to_pt.tolist(), [0, 1])
        self.assertEqual(alignment.pt_to_cache.tolist(), [0, 1])

    def test_same_normalized_onset_permutation(self):
        # 300000 us/beat gives 0.625 target ticks/source tick. Raw ticks 12/13
        # both banker's-round to PT tick 8; PT then sorts the lower pitch first.
        midi, cache, _ = self.write_case(
            [(80, 12, 100, 80), (40, 13, 101, 70)], tempo=300000
        )
        alignment = build_note_alignment(midi, cache)
        self.assertEqual(alignment.cache_to_pt.tolist(), [1, 0])
        self.assertEqual(alignment.stats["moved_across_raw_onsets"], 2)
        self.assertEqual(alignment.stats["moved_across_normalized_onsets"], 0)

    def test_duplicate_pitch_is_disambiguated_by_full_identity(self):
        midi, cache, _ = self.write_case([(60, 0, 40, 80), (60, 100, 170, 50)])
        alignment = build_note_alignment(midi, cache)
        self.assertEqual(alignment.note_count, 2)
        self.assertEqual(alignment.stats["ambiguous_mappings"], 0)

    def test_ambiguous_full_identity_fails_explicitly(self):
        midi, cache, _ = self.write_case([(60, 0, 100, 80), (60, 0, 100, 80)])
        with self.assertRaisesRegex(NoteAlignmentError, "ambiguous common raw"):
            build_note_alignment(midi, cache)

    def test_missing_note_fails_explicitly(self):
        midi, cache, rows = self.write_case([(60, 0, 100, 80), (64, 120, 200, 70)])
        np.savez_compressed(
            cache,
            note_onset_ticks=np.asarray([rows[0][3]]), note_pitch=np.asarray([rows[0][2]]),
            note_offset_ticks=np.asarray([rows[0][4]]), note_velocity=np.asarray([rows[0][5]]),
        )
        with self.assertRaisesRegex(NoteAlignmentError, "canonical raw parser"):
            build_note_alignment(midi, cache)

    def test_representative_mapping_group_containment_and_ownership(self):
        midi, cache, _ = self.write_case(
            [(80, 12, 100, 80), (40, 13, 101, 70), (70, 200, 260, 60)],
            tempo=300000,
        )
        alignment = build_note_alignment(midi, cache)
        first = np.asarray([0, 1, 2])
        mapped_first, mapped_last, mapped_rep, noncontiguous = alignment.map_group_bounds(
            first, first, first
        )
        self.assertEqual(mapped_rep.tolist(), [1, 0, 2])
        self.assertEqual(noncontiguous, 0)
        ownership = assign_unique_owners(3, mapped_first, mapped_last, mapped_rep)
        self.assertEqual(sum(map(len, ownership.owned_onset_indices)), 3)

    def test_serialization_reload_and_frozen_cache_unchanged(self):
        midi, cache, _ = self.write_case([(60, 0, 100, 80), (64, 120, 200, 70)])
        before = digest(cache)
        alignment = build_note_alignment(midi, cache)
        output = self.root / "mapping.npz"
        save_note_alignment(output, alignment, {"stats": alignment.stats, "tokenizer_cache_id": "x"})
        loaded, metadata = load_note_alignment(output)
        self.assertTrue(np.array_equal(loaded.cache_to_pt, alignment.cache_to_pt))
        self.assertEqual(metadata["tokenizer_cache_id"], "x")
        self.assertEqual(digest(cache), before)


if __name__ == "__main__":
    unittest.main()
