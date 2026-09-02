#!/usr/bin/env python3
"""Focused regressions for Frozen-PT PRE-origin and POST-tail rendering."""

from __future__ import annotations

import tempfile
from pathlib import Path
import sys

import mido

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_custom_event_model_v0_canonical_val_inference_v1 import (
    noncc_except_eot_signature,
    write_candidate,
)
from src.stage2_binary.canonical_stage1 import cc64_schedule
from src.stage2_binary_2slot.frozen_pt_validation import (
    RenderedCC64,
    project_decoded_cc64_to_midi_origin,
    serialized_candidate_events_for_metric,
)


def millisecond_tick(seconds: float) -> int:
    return int(round(seconds * 1000.0))


def event(seconds: float, direction: str, region: str = "PRE", interval_index: int = 0):
    return {
        "seconds": seconds,
        "direction": direction,
        "region": region,
        "interval_index": interval_index,
        "slot": 0,
    }


def test_exact_failed_negative_pre_event_becomes_origin_state_anchor_without_clamp():
    source = [event(-0.4475204348564148, "DOWN")]
    rendered, audit = project_decoded_cc64_to_midi_origin(source, millisecond_tick)
    assert [(item.tick, item.cc64_value) for item in rendered] == [(0, 127)]
    assert audit["pre_origin_decoded_event_count"] == 1
    assert audit["state_at_midi_origin"] == 1
    assert audit["midi_origin_state_anchor_inserted"] is True
    assert audit["negative_event_timestamp_clamp_used"] is False
    assert source[0]["seconds"] == -0.4475204348564148


def test_two_pre_origin_toggles_reduce_to_default_off_without_spurious_tick_zero_events():
    rendered, audit = project_decoded_cc64_to_midi_origin(
        [event(-0.8, "DOWN"), event(-0.2, "UP")], millisecond_tick,
    )
    assert rendered == []
    assert audit["state_at_midi_origin"] == 0
    assert audit["midi_origin_state_anchor_inserted"] is False


def test_pre_time_boundaries_for_t1_greater_equal_and_less_than_one_second():
    after_origin, after_audit = project_decoded_cc64_to_midi_origin(
        [event(0.5, "DOWN")], millisecond_tick,
    )
    at_origin, at_audit = project_decoded_cc64_to_midi_origin(
        [event(0.0, "DOWN")], millisecond_tick,
    )
    before_origin, before_audit = project_decoded_cc64_to_midi_origin(
        [event(-0.25, "DOWN")], millisecond_tick,
    )
    assert [(item.tick, item.cc64_value) for item in after_origin] == [(500, 127)]
    assert after_audit["pre_origin_decoded_event_count"] == 0
    assert [(item.tick, item.cc64_value) for item in at_origin] == [(0, 127)]
    assert at_audit["midi_origin_state_anchor_inserted"] is False
    assert [(item.tick, item.cc64_value) for item in before_origin] == [(0, 127)]
    assert before_audit["midi_origin_state_anchor_inserted"] is True


def test_negative_event_outside_pre_is_rejected():
    try:
        project_decoded_cc64_to_midi_origin([event(-0.1, "DOWN", "MAIN", 1)], millisecond_tick)
    except ValueError as error:
        assert "outside PRE" in str(error)
    else:
        raise AssertionError("negative MAIN event was accepted")


def _write_source(path: Path) -> None:
    midi = mido.MidiFile(type=1, ticks_per_beat=480, clip=False)
    tempo = mido.MidiTrack()
    tempo.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
    tempo.append(mido.MetaMessage("end_of_track", time=480))
    notes = mido.MidiTrack()
    notes.append(mido.Message("note_on", channel=0, note=60, velocity=90, time=0))
    notes.append(mido.Message("note_off", channel=0, note=60, velocity=0, time=240))
    notes.append(mido.MetaMessage("end_of_track", time=240))
    midi.tracks = [tempo, notes]
    midi.save(str(path))


def test_positive_post_event_after_original_eot_serializes_with_valid_chronology():
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "source.mid"
        candidate = Path(directory) / "candidate.mid"
        _write_source(source)
        before = noncc_except_eot_signature(source)
        identity = write_candidate(source, candidate, [RenderedCC64(1200, 127, 0)])
        schedule = cc64_schedule(candidate)
        assert schedule == [{
            "track_index": 0, "absolute_tick": 1200, "channel": 0, "value": 127,
            "source_order": 1, "non_cc64_before_at_tick": 0,
        }]
        assert identity["eot_extension_ticks"] == 720
        assert before == noncc_except_eot_signature(candidate)
        midi = mido.MidiFile(str(candidate), clip=False)
        for track in midi.tracks:
            assert all(message.time >= 0 for message in track)


def test_projected_values_and_order_are_valid_and_chronological():
    rendered, _ = project_decoded_cc64_to_midi_origin(
        [event(-0.5, "DOWN"), event(0.25, "UP", "MAIN", 1),
         event(2.0, "DOWN", "POST", 2)],
        millisecond_tick,
    )
    assert [(item.tick, item.cc64_value) for item in rendered] == [
        (0, 127), (250, 0), (2000, 127),
    ]
    assert all(0 <= item.cc64_value <= 127 and item.tick >= 0 for item in rendered)
    assert [item.ordering_key for item in rendered] == sorted(item.ordering_key for item in rendered)


def test_serialized_candidate_metric_trajectory_matches_origin_projection():
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "source.mid"
        candidate = Path(directory) / "candidate.mid"
        _write_source(source)
        rendered, audit = project_decoded_cc64_to_midi_origin(
            [event(-0.4475204348564148, "DOWN"),
             event(0.25, "UP", "MAIN", 1),
             event(2.0, "DOWN", "POST", 2)],
            millisecond_tick,
        )
        write_candidate(source, candidate, rendered)
        assert audit["midi_origin_state_anchor_inserted"] is True
        assert serialized_candidate_events_for_metric(candidate) == [
            {"direction": "DOWN", "source_tick": 0},
            {"direction": "UP", "source_tick": 250},
            {"direction": "DOWN", "source_tick": 2000},
        ]


def main():
    tests = sorted((name, value) for name, value in globals().items() if name.startswith("test_"))
    results = {}
    for name, test in tests:
        test()
        results[name] = "passed"
    print(results)


if __name__ == "__main__":
    main()
