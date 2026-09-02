#!/usr/bin/env python3
"""Focused Run B S1 serialization/metric-separation regressions."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from miditoolkit import Instrument, MidiFile, Note

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage2_binary_2slot.frozen_pt_validation import (  # noqa: E402
    FrozenPTInterval,
    FrozenPTPiece,
)
from src.stage2_state_anchored.decoding import CHANGE_ID, HOLD_ID, RETURN_ID  # noqa: E402
from src.stage2_state_anchored.serialization import (  # noqa: E402
    reconstruct_modeled_events,
    render_state_anchored_candidate,
    serialization_schedule,
)


def _piece(root: Path) -> FrozenPTPiece:
    path = root / "source.mid"
    midi = MidiFile(ticks_per_beat=480)
    piano = Instrument(program=0)
    piano.notes = [
        Note(velocity=80, pitch=60, start=0, end=240),
        Note(velocity=90, pitch=64, start=480, end=720),
    ]
    midi.instruments = [piano]
    midi.dump(str(path))
    main = (
        FrozenPTInterval("MAIN", 1, 0, 0.0, 0.5),
        FrozenPTInterval("MAIN", 2, 1, 0.5, 0.75),
    )
    return FrozenPTPiece(
        "synthetic", path, torch.ones((2, 8), dtype=torch.long), None,
        (0.0, 0.5), 0.75,
        FrozenPTInterval("PRE", 0, None, -1.0, 0.0), main,
        FrozenPTInterval("POST", 3, None, 0.75, 1.75),
    )


def test_s1_off_has_no_initialization_down() -> None:
    with tempfile.TemporaryDirectory() as value:
        piece = _piece(Path(value))
        schedule, audit = serialization_schedule(piece, 0, ())
        assert schedule == []
        assert audit["initialization_anchor_count"] == 0
        assert audit["initialization_anchor_metric_event_count"] == 0


def test_s1_on_has_exactly_one_serialization_only_anchor() -> None:
    with tempfile.TemporaryDirectory() as value:
        piece = _piece(Path(value))
        schedule, audit = serialization_schedule(piece, 1, ())
        assert len(schedule) == 1 and schedule[0].cc64_value == 127
        assert audit["initialization_anchor_count"] == 1
        assert audit["initialization_anchor_metric_event_count"] == 0


def test_s1_anchor_is_not_a_modeled_change_and_tau_zero_change_is_retained() -> None:
    with tempfile.TemporaryDirectory() as value:
        root = Path(value)
        piece = _piece(root)
        events = reconstruct_modeled_events(
            piece.onset_seconds, [1, 0], [CHANGE_ID], torch.tensor([[0.0, 0.9]])
        )
        assert len(events) == 1 and events[0]["direction"] == "UP" and events[0]["tau"] == 0.0
        schedule, audit = serialization_schedule(piece, 1, events)
        assert [item.cc64_value for item in schedule] == [127, 0]
        assert schedule[0].tick == schedule[1].tick
        assert audit["initialization_anchor_metric_event_count"] == 0
        rendered = render_state_anchored_candidate(
            piece, [1, 0], [CHANGE_ID], torch.tensor([[0.0, 0.9]]), root / "candidate.mid"
        )
        assert len(rendered.modeled_events) == 1
        assert rendered.identity["initialization_anchor_count"] == 1
        assert rendered.identity["note_identity_exact"]


def test_mode_reconstruction_and_return_chronology() -> None:
    events = reconstruct_modeled_events(
        (0.0, 1.0, 2.0, 3.0), [0, 0, 0, 1],
        [HOLD_ID, RETURN_ID, CHANGE_ID],
        torch.tensor([[0.2, 0.4], [0.8, 0.1], [1.2, -0.5]]),
    )
    assert [(item["interval_index"], item["direction"]) for item in events] == [
        (1, "DOWN"), (1, "UP"), (2, "DOWN")
    ]
    assert np.allclose([item["tau"] for item in events], [0.1, 0.8, 1.0])


def run_directly() -> dict[str, str]:
    tests = {name: value for name, value in globals().items() if name.startswith("test_")}
    result = {}
    for name, function in sorted(tests.items()):
        function()
        result[name] = "passed"
    return result


if __name__ == "__main__":
    print(run_directly())
