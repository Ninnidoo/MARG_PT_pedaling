#!/usr/bin/env python3
"""Focused synthetic tests for State-Anchored Transition v1 targets."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import mido

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage2_binary_2slot.dataset import (  # noqa: E402
    HumanIntervalPrimitive,
    build_performance_timeline,
)
from src.stage2_event_tokenizer.binary_2slot import (  # noqa: E402
    DOWN,
    OFF,
    ON,
    UP,
    BinaryTransition,
)
from src.stage2_state_anchored.targets import (  # noqa: E402
    CHANGE,
    HOLD,
    RETURN,
    build_interval_target,
    build_performance_targets,
)


def transitions(start: int, taus: tuple[float, ...]) -> tuple[BinaryTransition, ...]:
    state = start
    result = []
    for tau in taus:
        result.append(BinaryTransition(DOWN if state == OFF else UP, tau))
        state = 1 - state
    return tuple(result)


def target(start: int, taus: tuple[float, ...]):
    raw = transitions(start, taus)
    interval = HumanIntervalPrimitive("MAIN", 1, 0, 10.0, 20.0, start, raw)
    return build_interval_target(interval, next_onset_state=start ^ (len(raw) % 2))


def test_hold() -> None:
    value = target(OFF, ())
    assert value.mode == HOLD
    assert value.timing_mask == (False, False)


def test_off_to_on_change() -> None:
    value = target(OFF, (0.25,))
    assert value.mode == CHANGE and value.start_state == OFF and value.end_state == ON
    assert value.timing_targets == (0.25, 0.0) and value.timing_mask == (True, False)


def test_on_to_off_change() -> None:
    value = target(ON, (0.25,))
    assert value.mode == CHANGE and value.start_state == ON and value.end_state == OFF


def test_on_off_on_return() -> None:
    value = target(ON, (0.2, 0.8))
    assert value.mode == RETURN and value.start_state == value.end_state == ON
    assert tuple(event.direction for event in value.retained_transitions) == (UP, DOWN)


def test_off_on_off_return() -> None:
    value = target(OFF, (0.2, 0.8))
    assert value.mode == RETURN and value.start_state == value.end_state == OFF
    assert tuple(event.direction for event in value.retained_transitions) == (DOWN, UP)


def test_three_transitions_retain_last_one() -> None:
    value = target(OFF, (0.1, 0.4, 0.9))
    assert value.mode == CHANGE
    assert value.retained_transitions == value.raw_transitions[-1:]


def test_four_transitions_retain_last_two() -> None:
    value = target(ON, (0.1, 0.3, 0.6, 0.9))
    assert value.mode == RETURN
    assert value.retained_transitions == value.raw_transitions[-2:]


def test_five_transitions_retain_last_one() -> None:
    value = target(OFF, (0.1, 0.2, 0.4, 0.7, 0.9))
    assert value.mode == CHANGE
    assert value.retained_transitions == value.raw_transitions[-1:]


def test_six_transitions_retain_last_two() -> None:
    value = target(ON, (0.05, 0.15, 0.3, 0.5, 0.7, 0.95))
    assert value.mode == RETURN
    assert value.retained_transitions == value.raw_transitions[-2:]


def test_targets_are_immutable_and_prediction_independent() -> None:
    value = target(OFF, (0.25,))
    assert value.mode == CHANGE and value.start_state == OFF and value.end_state == ON
    try:
        value.mode = HOLD  # type: ignore[misc]
    except Exception:
        pass
    else:
        raise AssertionError("frozen target dataclass accepted mutation")


def _write_boundary_midi(path: Path) -> None:
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    absolute = [
        (0, mido.MetaMessage("set_tempo", tempo=500_000)),
        (50, mido.Message("control_change", channel=0, control=64, value=127)),
        (100, mido.Message("note_on", channel=0, note=60, velocity=80)),
        (100, mido.Message("control_change", channel=0, control=64, value=0)),
        (150, mido.Message("note_off", channel=0, note=60, velocity=0)),
        (200, mido.Message("note_on", channel=0, note=62, velocity=80)),
        (200, mido.Message("control_change", channel=0, control=64, value=127)),
        (250, mido.Message("note_off", channel=0, note=62, velocity=0)),
        (300, mido.Message("note_on", channel=0, note=64, velocity=80)),
        (360, mido.Message("note_off", channel=0, note=64, velocity=0)),
    ]
    previous = 0
    for tick, message in absolute:
        message.time = tick - previous
        track.append(message)
        previous = tick
    midi.save(path)


def test_boundaries_and_strict_before_first_state() -> None:
    with tempfile.TemporaryDirectory() as directory:
        midi_path = Path(directory) / "boundaries.mid"
        _write_boundary_midi(midi_path)
        timeline = build_performance_timeline(
            midi_path,
            (100, 200, 300),
            360,
            performance_id="synthetic-boundaries",
        )
    result = build_performance_targets(timeline)
    assert result.onset_states == (ON, OFF, ON)
    assert len(result.intervals) == 2
    first, second = result.intervals
    assert first.timing_targets[0] == 0.0
    assert tuple(event.source_tick for event in first.raw_transitions) == (100,)
    assert tuple(event.source_tick for event in second.raw_transitions) == (200,)
    assert second.timing_targets[0] == 0.0


def test_rejects_tau_at_right_boundary() -> None:
    interval = HumanIntervalPrimitive(
        "MAIN", 1, 0, 0.0, 1.0, OFF, (BinaryTransition(DOWN, 1.0),)
    )
    try:
        build_interval_target(interval, next_onset_state=ON)
    except AssertionError as error:
        assert "tau < 1" in str(error)
    else:
        raise AssertionError("right-boundary tau=1 must not be accepted in MAIN")


def run_directly() -> dict[str, str]:
    tests = {name: value for name, value in globals().items() if name.startswith("test_")}
    result = {}
    for name, function in sorted(tests.items()):
        function()
        result[name] = "passed"
    return result


if __name__ == "__main__":
    print(run_directly())
