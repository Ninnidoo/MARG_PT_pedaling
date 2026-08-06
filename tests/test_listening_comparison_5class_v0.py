from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from miditoolkit import ControlChange, Instrument, MidiFile, Note, TempoChange

from src.listening_comparison_5class_v0 import (
    PEDAL_TOKEN_OFFSET,
    build_argument_parser,
    infer_five_class_pedals,
    select_first_valid_score,
    verify_midi_control,
)
from src.stage2_encoder_only.five_class import FiveClassPedalOutput


class _DummyFiveClassModel:
    def __call__(self, input_ids, token_attention_mask, note_mask):
        notes = input_ids.shape[1] // 8
        logits = torch.full((1, notes, 4, 5), -10.0, device=input_ids.device)
        for slot in range(4):
            logits[:, :, slot, slot + 1] = 10.0
        return FiveClassPedalOutput(logits=logits)


def _write_midi(path: Path, cc64: list[tuple[int, int]]) -> None:
    midi = MidiFile(ticks_per_beat=480)
    instrument = Instrument(program=0, is_drum=False, name="Piano")
    instrument.notes = [Note(velocity=90, pitch=60, start=0, end=480)]
    instrument.control_changes = [
        ControlChange(number=64, value=value, time=time) for time, value in cc64
    ]
    midi.instruments.append(instrument)
    midi.tempo_changes.append(TempoChange(120.0, 0))
    midi.dump(str(path))


class ListeningComparisonTests(unittest.TestCase):
    def test_cli_seed_defaults_to_42(self) -> None:
        args = build_argument_parser().parse_args(
            ["--score", "score.mid", "--output-midi-dir", "m", "--output-audio-dir", "a"]
        )
        self.assertEqual(args.seed, 42)

    def test_select_first_valid_score_uses_lexical_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "0.mid").write_bytes(b"invalid")
            _write_midi(root / "10.mid", [])
            _write_midi(root / "2.mid", [])
            self.assertEqual(select_first_valid_score(root).name, "10.mid")

    def test_inference_changes_only_pedals_to_representatives(self) -> None:
        note = [65, 300, 200, 400, 5261, 5262, 5263, 5264]
        original = np.asarray(note * 513, dtype=np.int64).reshape(-1, 8)
        predicted, details = infer_five_class_pedals(
            _DummyFiveClassModel(), original.reshape(-1).tolist(), torch.device("cpu")
        )
        result = np.asarray(predicted).reshape(-1, 8)
        np.testing.assert_array_equal(result[:, :4], original[:, :4])
        expected = np.asarray([32, 80, 111, 127]) + PEDAL_TOKEN_OFFSET
        np.testing.assert_array_equal(result[:, 4:], np.tile(expected, (513, 1)))
        self.assertEqual(details["windows"], 2)

    def test_midi_control_accepts_only_cc64_difference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, second = root / "a.mid", root / "b.mid"
            _write_midi(first, [(0, 0), (240, 127)])
            _write_midi(second, [(0, 32), (240, 111)])
            result = verify_midi_control(first, second)
            self.assertTrue(result["passed"])
            self.assertTrue(result["only_sustain_cc64_may_differ"])
            self.assertTrue(result["sustain_cc64_differs"])


    def test_midi_control_allows_identical_cc64(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, second = root / "a.mid", root / "b.mid"
            _write_midi(first, [])
            _write_midi(second, [])
            result = verify_midi_control(first, second)
            self.assertTrue(result["passed"])
            self.assertFalse(result["sustain_cc64_differs"])


if __name__ == "__main__":
    unittest.main()
