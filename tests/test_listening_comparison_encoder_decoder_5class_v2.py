from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from miditoolkit import ControlChange, Instrument, MidiFile, Note, TempoChange

from src.listening_comparison_encoder_decoder_5class_v2 import (
    MASK_ID,
    MODEL_A_SUFFIX,
    MODEL_B_SUFFIX,
    PEDAL_TOKEN_OFFSET,
    _intermediate_sanity,
    infer_encoder_decoder_pedals,
    remove_sustain_cc64,
)


class _DummyGreedyModel:
    def __call__(
        self,
        input_ids,
        token_attention_mask,
        pedal_targets=None,
        note_mask=None,
        *,
        decode_mode="teacher_forced",
    ):
        del token_attention_mask, pedal_targets
        assert decode_mode == "greedy"
        batch = input_ids.shape[0]
        notes = input_ids.shape[1] // 8
        logits = torch.full((batch, notes, 4, 5), -9.0, device=input_ids.device)
        for slot in range(4):
            logits[:, :, slot, slot + 1] = 9.0
        if note_mask is not None:
            logits = logits.masked_fill(~note_mask[:, :, None, None], 0.0)
        return SimpleNamespace(logits=logits)


def _midi() -> MidiFile:
    midi = MidiFile(ticks_per_beat=480)
    instrument = Instrument(program=0, is_drum=False, name="Piano")
    instrument.notes = [
        Note(velocity=91, pitch=60, start=0, end=360),
        Note(velocity=74, pitch=67, start=480, end=900),
    ]
    instrument.control_changes = [
        ControlChange(number=1, value=12, time=10),
        ControlChange(number=64, value=127, time=20),
        ControlChange(number=64, value=0, time=700),
    ]
    midi.instruments.append(instrument)
    midi.tempo_changes.append(TempoChange(120.0, 0))
    return midi


class EncoderDecoderListeningTests(unittest.TestCase):
    def test_requested_suffixes_are_exact(self) -> None:
        self.assertEqual(MODEL_A_SUFFIX, "stage2_5class_v2")
        self.assertEqual(MODEL_B_SUFFIX, "stage2_5class_v2_fulltrain")

    def test_remove_sustain_preserves_notes_and_other_cc(self) -> None:
        original = _midi()
        cleaned = remove_sustain_cc64(original)
        self.assertEqual(len(original.instruments[0].notes), len(cleaned.instruments[0].notes))
        self.assertEqual(
            [(n.pitch, n.start, n.end, n.velocity) for n in original.instruments[0].notes],
            [(n.pitch, n.start, n.end, n.velocity) for n in cleaned.instruments[0].notes],
        )
        self.assertEqual(
            [(event.number, event.value, event.time) for event in cleaned.instruments[0].control_changes],
            [(1, 12, 10)],
        )
        self.assertEqual(sum(event.number == 64 for event in original.instruments[0].control_changes), 2)

    def test_greedy_overlap_changes_only_four_pedal_tokens(self) -> None:
        note = [65, 300, 200, 400, 5261, 5262, 5263, 5264]
        source = np.asarray(note * 513, dtype=np.int64).reshape(-1, 8)
        predicted, details = infer_encoder_decoder_pedals(
            _DummyGreedyModel(), source.reshape(-1).tolist(), torch.device("cpu"), 4
        )
        result = np.asarray(predicted, dtype=np.int64).reshape(-1, 8)
        np.testing.assert_array_equal(result[:, :4], source[:, :4])
        expected = np.asarray([32, 80, 111, 127]) + PEDAL_TOKEN_OFFSET
        np.testing.assert_array_equal(result[:, 4:], np.tile(expected, (513, 1)))
        self.assertEqual(details["windows"], 2)

    def test_intermediate_sanity_requires_no_cc64_and_masked_input(self) -> None:
        original = _midi()
        pedal_free = remove_sustain_cc64(original)
        generated = np.asarray(
            [65, 300, 200, 400, 5261, 5261, 5261, 5261] * 2,
            dtype=np.int64,
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pedal_free.mid"
            pedal_free.dump(str(path))
            checks = _intermediate_sanity(original, path, generated)
        self.assertTrue(checks["passed"])
        self.assertTrue(checks["stage2_all_pedal_positions_masked"])
        self.assertEqual(MASK_ID, 1)


if __name__ == "__main__":
    unittest.main()
