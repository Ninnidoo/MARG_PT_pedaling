from __future__ import annotations

import importlib.util
from pathlib import Path

from miditoolkit import Instrument, MidiFile
from miditoolkit.midi.containers import ControlChange, Note


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/evaluate_stage2_4class_repedal_validation_v0.py"
SPEC = importlib.util.spec_from_file_location("repedal_validation", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def make_midi(path: Path, controls: list[tuple[int, int]]) -> None:
    midi = MidiFile(ticks_per_beat=1000)
    instrument = Instrument(program=0)
    instrument.notes = [Note(velocity=80, pitch=60, start=tick, end=tick + 10) for tick in range(0, 1000, 100)]
    instrument.control_changes = [ControlChange(number=64, value=value, time=tick) for tick, value in controls]
    midi.instruments = [instrument]
    midi.dump(str(path))


def mapped(up_tick: int, down_tick: int, up_position: int = 2, down_position: int = 3):
    return {
        ("UP", up_tick): {"score_position": up_position, "onset_index": up_position},
        ("DOWN", down_tick): {"score_position": down_position, "onset_index": down_position},
    }


def test_canonical_extraction_uses_destination_timestamps_and_collapses_equal(tmp_path: Path):
    path = tmp_path / "gesture.mid"
    make_midi(path, [(0, 10), (50, 100), (75, 100), (100, 63), (150, 20), (250, 64), (300, 110), (350, 0)])
    rows = MODULE.extract_repedals(
        path,
        mapping=mapped(100, 250),
        candidate="test",
        piece_id="piece",
        performance_id="performance",
        role="candidate",
    )
    assert len(rows) == 1
    assert rows[0]["release_excursion"] == 80
    assert rows[0]["repress_excursion"] == 90
    assert rows[0]["up_tick"] == 100
    assert rows[0]["down_tick"] == 250
    assert rows[0]["off_duration_ms"] == 75.0  # default tempo: 0.5 ms/tick


def test_rejects_weak_and_too_slow_gestures(tmp_path: Path):
    weak = tmp_path / "weak.mid"
    make_midi(weak, [(0, 10), (50, 80), (100, 20), (150, 64), (200, 80)])
    assert MODULE.extract_repedals(
        weak, mapping=mapped(100, 150), candidate="x", piece_id="p", performance_id="i", role="candidate"
    ) == []
    slow = tmp_path / "slow.mid"
    make_midi(slow, [(0, 0), (100, 127), (200, 0), (900, 127), (950, 0)])
    assert MODULE.extract_repedals(
        slow, mapping=mapped(200, 900), candidate="x", piece_id="p", performance_id="i", role="candidate"
    ) == []


def event(up: int, down: int, local: int = 0):
    return {
        "up_score_position": up,
        "down_score_position": down,
        "up_onset_index": up + local,
        "down_onset_index": down + local,
    }


def test_two_boundaries_required_and_matching_is_one_to_one():
    predicted = [event(10, 20), event(10, 20, 1), event(30, 40)]
    reference = [event(11, 21), event(30, 42)]
    matched = MODULE.match_repedals(predicted, reference)
    assert len(matched) == 1
    assert matched[0][1] == 0


def test_zero_prediction_convention():
    assert MODULE.metric_row(0, 7, 0) == {
        "reference_repedal_count": 7,
        "candidate_repedal_count": 0,
        "tp": 0,
        "fp": 0,
        "fn": 7,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }
