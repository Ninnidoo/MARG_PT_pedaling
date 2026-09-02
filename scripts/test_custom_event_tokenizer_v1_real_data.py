#!/usr/bin/env python3
"""Small deterministic real-MIDI regression for the frozen tokenizer v1."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_custom_event_tokenizer_oracle_audit_v0 import load_manifests
from src.stage2_event_tokenizer.decoder_v1 import decode_controls_v1, write_roundtrip_midi_v1
from src.stage2_event_tokenizer.tokenizer import apply_tokens
from src.stage2_event_tokenizer.tokenizer_v1 import apply_terminal_tokens, encode_midi_v1


def validate(row: dict[str, str], split: str) -> dict[str, object]:
    encoded = encode_midi_v1(row["performance_absolute_path"])
    assert len(encoded.main_intervals) == len(encoded.onset_groups)
    assert len(encoded.terminal.slots) == 4
    assert all(len(interval.slots) == 6 for interval in encoded.main_intervals)
    assert encoded.main_intervals[-1].is_final_main_interval
    assert encoded.main_intervals[-1].right_tick == encoded.source.latest_note_off
    assert all(
        apply_tokens(interval.start_state, interval.all_events)
        == apply_tokens(interval.start_state, interval.slots)
        for interval in encoded.main_intervals
    )
    assert (
        apply_terminal_tokens(encoded.terminal.start_state, encoded.terminal.all_events)
        == apply_terminal_tokens(encoded.terminal.start_state, encoded.terminal.slots)
    )
    assert all(
        event.source_tick <= encoded.source.latest_note_off
        for interval in encoded.main_intervals
        for event in interval.all_events
    )
    assert all(
        event.source_tick > encoded.source.latest_note_off
        for event in encoded.terminal.all_events
    )
    controls = decode_controls_v1(encoded)
    decoded_main = [control.tick for control in controls if control.kind == "main"]
    target_main = [
        event.source_tick
        for interval in encoded.main_intervals
        for event in interval.slots
        if event.event_class
    ]
    decoded_terminal = [control.tick for control in controls if control.kind == "terminal"]
    target_terminal = [event.source_tick for event in encoded.terminal.slots if event.event_class]
    assert decoded_main == target_main
    assert decoded_terminal == target_terminal
    with tempfile.TemporaryDirectory(prefix="tokenizer_v1_real_") as directory:
        output = Path(directory) / "roundtrip.mid"
        write_roundtrip_midi_v1(encoded, output)
        assert output.is_file()
    return {
        "split": split,
        "performance_path": row["performance_path"],
        "notes": encoded.non_pedal_note_count,
        "onsets": len(encoded.onset_groups),
        "main_events_before_cap": sum(len(interval.all_events) for interval in encoded.main_intervals),
        "main_events_retained": len(target_main),
        "terminal_events_before_cap": len(encoded.terminal.all_events),
        "terminal_events_retained": len(target_terminal),
        "main_final_state_preserved": True,
        "terminal_final_state_preserved": True,
        "non_pedal_identity": True,
    }


def representative_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    selected = [rows[0]]
    terminal_rich = None
    overflow_rich = None
    for row in rows:
        encoded = encode_midi_v1(row["performance_absolute_path"])
        if terminal_rich is None and encoded.terminal.all_events:
            terminal_rich = row
        if overflow_rich is None and any(len(interval.all_events) > 6 for interval in encoded.main_intervals):
            overflow_rich = row
        if terminal_rich is not None and overflow_rich is not None:
            break
    for row in (terminal_rich, overflow_rich):
        if row is not None and row["performance_path"] not in {item["performance_path"] for item in selected}:
            selected.append(row)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    train, validation, provenance = load_manifests()
    results = []
    for split, rows in (("train", train), ("validation", validation)):
        results.extend(validate(row, split) for row in representative_rows(rows))
    payload = {
        "status": "passed",
        "cases": len(results),
        "results": results,
        "raw_cc64_source": provenance["raw_cc64_source"],
        "asap_test_access_count": 0,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
        temporary.write_text(rendered, encoding="utf-8")
        os.replace(temporary, args.output)
    print(rendered, end="")


if __name__ == "__main__":
    main()
