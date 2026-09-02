#!/usr/bin/env python3
"""CPU-only canonical repedaling evaluation over saved validation MIDI files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from miditoolkit import MidiFile


ROOT = Path(__file__).resolve().parents[1]
SOURCE_EVAL = ROOT / "analysis/stage2_4class_architecture_validation_eval_v0"
EXTREME_MANIFEST = (
    ROOT / "analysis/harmonic_metric_broad_parameter_search_v0/reusable_extremes_manifest.csv"
)
PILOT_EXTREME_MANIFEST = ROOT / "analysis/harmonic_metric_kdyn_blow_pilot_v0/reusable_extremes/manifest.csv"
HUMAN_INVENTORY = ROOT / "analysis/harmonic_metric_broad_parameter_search_v0/input_midis.csv"
SPLIT_CSV = ROOT / "analysis/stage2_encoder_only_v0/asap_split.csv"
ASAP_ROOT = Path("/workspace/public/ASAP/asap-dataset-v1.1")
OUTPUT_ROOT = ROOT / "analysis/stage2_4class_repedal_validation_v0"

THRESHOLD = 64
MIN_EXCURSION = 64
MAX_OFF_DURATION_MS = 300.0
ONSET_TOLERANCE = 1
MODEL_LABELS = {
    "original_pt": "Original PT",
    "encoder_only": "Encoder-only 4C",
    "decoder_only": "Decoder-only 4C FR",
    "encoder_decoder": "Encoder-Decoder 4C FR",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def container_path(path: str | Path) -> Path:
    value = str(path)
    if value == "/workspace/project" or value.startswith("/workspace/project/"):
        return ROOT / Path(value).relative_to("/workspace/project")
    return Path(value)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def raw_midi_trajectory(midi: MidiFile) -> tuple[list[int], list[dict[str, int]]]:
    """Use the exact raw-event ordering and distinct-onset definition of Transition F1."""

    onsets = sorted(
        {int(note.start) for instrument in midi.instruments if not instrument.is_drum for note in instrument.notes}
    )
    controls: list[tuple[int, int, int, int]] = []
    for instrument_index, instrument in enumerate(midi.instruments):
        if instrument.is_drum:
            continue
        for event_index, event in enumerate(instrument.control_changes):
            if int(event.number) == 64:
                controls.append(
                    (int(event.time), instrument_index, event_index, int(event.value))
                )
    controls.sort()
    compact: list[dict[str, int]] = []
    for tick, instrument_index, event_index, value in controls:
        if not compact or compact[-1]["value"] != value:
            compact.append(
                {
                    "tick": tick,
                    "value": value,
                    "instrument_index": instrument_index,
                    "event_index": event_index,
                }
            )
    return onsets, compact


def tick_ms_fn(midi: MidiFile):
    changes = sorted(midi.tempo_changes, key=lambda item: item.time)
    ticks = [0]
    tempos = [120.0]
    for item in changes:
        if int(item.time) == ticks[-1]:
            tempos[-1] = float(item.tempo)
        else:
            ticks.append(int(item.time))
            tempos.append(float(item.tempo))
    cumulative = [0.0]
    for index in range(1, len(ticks)):
        cumulative.append(
            cumulative[-1]
            + (ticks[index] - ticks[index - 1])
            * 60000.0
            / (tempos[index - 1] * midi.ticks_per_beat)
        )

    def convert(tick: int) -> float:
        index = int(np.searchsorted(ticks, tick, side="right") - 1)
        return cumulative[index] + (tick - ticks[index]) * 60000.0 / (
            tempos[index] * midi.ticks_per_beat
        )

    return convert


def transition_mapping(path: Path, expected_midi: Path) -> dict[tuple[str, int], dict[str, int]]:
    metadata_path = path.parent / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not metadata.get("complete"):
        raise RuntimeError(f"incomplete frozen alignment: {metadata_path}")
    if container_path(metadata["performance_path"]).resolve() != expected_midi.resolve():
        raise RuntimeError(f"alignment MIDI path changed: {metadata_path}")
    if metadata["performance_sha256"] != sha256_file(expected_midi):
        raise RuntimeError(f"alignment MIDI hash changed: {expected_midi}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for event in payload["transitions"]:
        grouped[(str(event["direction"]), int(event["tick"]))].append(event)
    result: dict[tuple[str, int], dict[str, int]] = {}
    for key, events in grouped.items():
        positions = {int(event["score_position"]) for event in events}
        onsets = {int(event["onset_index"]) for event in events}
        if len(positions) != 1 or len(onsets) != 1:
            raise RuntimeError(f"ambiguous frozen transition mapping for {key}: {path}")
        result[key] = {
            "score_position": positions.pop(),
            "onset_index": onsets.pop(),
        }
    return result


def deduplicate_repedal_episodes(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Collapse extrema triples that resolve to the same canonical OFF episode."""

    grouped: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["up_tick"]), int(row["down_tick"]))].append(row)

    def representative_key(row: Mapping[str, Any]) -> tuple[int, int, int, int, int]:
        return (
            int(row["trough_value"]),
            -min(int(row["release_excursion"]), int(row["repress_excursion"])),
            int(row["trough_tick"]),
            int(row["prior_peak_tick"]),
            int(row["following_peak_tick"]),
        )

    selected = [
        dict(min(group, key=representative_key))
        for _, group in sorted(grouped.items())
    ]
    for repedal_id, row in enumerate(selected):
        row["repedal_id"] = repedal_id
    return selected


def extract_repedals(
    midi_path: Path,
    *,
    mapping: Mapping[tuple[str, int], Mapping[str, int]] | None,
    candidate: str,
    piece_id: str,
    performance_id: str,
    role: str,
) -> list[dict[str, Any]]:
    """Extract frozen strong high-low-high repedals from a raw MIDI CC64 stream."""

    midi = MidiFile(str(midi_path))
    _, compact = raw_midi_trajectory(midi)
    convert = tick_ms_fn(midi)
    peaks: list[int] = []
    troughs: list[int] = []
    for index in range(1, len(compact) - 1):
        left = compact[index - 1]["value"]
        value = compact[index]["value"]
        right = compact[index + 1]["value"]
        if value > left and value > right:
            peaks.append(index)
        elif value < left and value < right:
            troughs.append(index)

    triples: set[tuple[int, int, int]] = set()
    for trough in troughs:
        trough_value = compact[trough]["value"]
        prior = next(
            (
                peak
                for peak in reversed(peaks)
                if peak < trough and compact[peak]["value"] - trough_value >= MIN_EXCURSION
            ),
            None,
        )
        following = next(
            (
                peak
                for peak in peaks
                if peak > trough and compact[peak]["value"] - trough_value >= MIN_EXCURSION
            ),
            None,
        )
        if prior is not None and following is not None:
            triples.add((prior, trough, following))

    rows: list[dict[str, Any]] = []
    for triple_index, (prior, trough, following) in enumerate(
        sorted(triples, key=lambda item: (item[1], item[0], item[2]))
    ):
        up_crossings = [
            index
            for index in range(prior + 1, trough + 1)
            if compact[index - 1]["value"] >= THRESHOLD
            and compact[index]["value"] < THRESHOLD
        ]
        down_crossings = [
            index
            for index in range(trough + 1, following + 1)
            if compact[index - 1]["value"] < THRESHOLD
            and compact[index]["value"] >= THRESHOLD
        ]
        if not up_crossings or not down_crossings:
            continue
        # The OFF episode containing the selected trough fixes a unique boundary pair.
        up = up_crossings[-1]
        down = down_crossings[0]
        up_tick = compact[up]["tick"]
        down_tick = compact[down]["tick"]
        up_ms = float(convert(up_tick))
        down_ms = float(convert(down_tick))
        duration_ms = down_ms - up_ms
        if not (0.0 < duration_ms <= MAX_OFF_DURATION_MS):
            continue
        if mapping is None:
            up_map = down_map = None
        else:
            try:
                up_map = mapping[("UP", up_tick)]
                down_map = mapping[("DOWN", down_tick)]
            except KeyError as error:
                raise RuntimeError(
                    f"accepted crossing absent from frozen Transition mapping: {midi_path} {error}"
                ) from error
        release = compact[prior]["value"] - compact[trough]["value"]
        repress = compact[following]["value"] - compact[trough]["value"]
        row = {
            "role": role,
            "candidate": candidate,
            "piece_id": piece_id,
            "performance_id": performance_id,
            "midi_path": str(midi_path),
            "repedal_id": triple_index,
            "prior_peak_index": prior,
            "trough_index": trough,
            "following_peak_index": following,
            "prior_peak_tick": compact[prior]["tick"],
            "trough_tick": compact[trough]["tick"],
            "following_peak_tick": compact[following]["tick"],
            "prior_peak_value": compact[prior]["value"],
            "trough_value": compact[trough]["value"],
            "following_peak_value": compact[following]["value"],
            "release_excursion": release,
            "repress_excursion": repress,
            "up_tick": up_tick,
            "down_tick": down_tick,
            "up_time_ms": up_ms,
            "down_time_ms": down_ms,
            "off_duration_ms": duration_ms,
            "up_onset_index": "" if up_map is None else int(up_map["onset_index"]),
            "down_onset_index": "" if down_map is None else int(down_map["onset_index"]),
            "up_score_position": "" if up_map is None else int(up_map["score_position"]),
            "down_score_position": "" if down_map is None else int(down_map["score_position"]),
        }
        if release < MIN_EXCURSION or repress < MIN_EXCURSION:
            raise AssertionError("weak gesture escaped canonical extraction")
        rows.append(row)
    return deduplicate_repedal_episodes(rows)


def match_repedals(
    candidate: Sequence[Mapping[str, Any]], reference: Sequence[Mapping[str, Any]]
) -> list[tuple[int, int]]:
    """Greedy one-to-one matching using the Transition matcher's narrowest-first rule."""

    possible: list[tuple[int, int, int, int, int, int]] = []
    for candidate_index, predicted in enumerate(candidate):
        for reference_index, target in enumerate(reference):
            up_delta = abs(
                int(predicted["up_score_position"]) - int(target["up_score_position"])
            )
            down_delta = abs(
                int(predicted["down_score_position"])
                - int(target["down_score_position"])
            )
            if up_delta <= ONSET_TOLERANCE and down_delta <= ONSET_TOLERANCE:
                local_delta = abs(
                    int(predicted["up_onset_index"]) - int(target["up_onset_index"])
                ) + abs(
                    int(predicted["down_onset_index"])
                    - int(target["down_onset_index"])
                )
                possible.append(
                    (
                        max(up_delta, down_delta),
                        up_delta + down_delta,
                        local_delta,
                        candidate_index,
                        reference_index,
                        len(possible),
                    )
                )
    used_candidate: set[int] = set()
    used_reference: set[int] = set()
    matched: list[tuple[int, int]] = []
    for _, _, _, candidate_index, reference_index, _ in sorted(possible):
        if candidate_index not in used_candidate and reference_index not in used_reference:
            used_candidate.add(candidate_index)
            used_reference.add(reference_index)
            matched.append((candidate_index, reference_index))
    return matched


def maximum_cardinality_match_count(
    candidate: Sequence[Mapping[str, Any]], reference: Sequence[Mapping[str, Any]]
) -> int:
    """Return diagnostic maximum-cardinality TP under the frozen boundary edges."""

    adjacency: list[list[int]] = []
    for predicted in candidate:
        adjacent = []
        for reference_index, target in enumerate(reference):
            if (
                abs(
                    int(predicted["up_score_position"])
                    - int(target["up_score_position"])
                )
                <= ONSET_TOLERANCE
                and abs(
                    int(predicted["down_score_position"])
                    - int(target["down_score_position"])
                )
                <= ONSET_TOLERANCE
            ):
                adjacent.append(reference_index)
        adjacency.append(adjacent)

    matched_candidate_by_reference = [-1] * len(reference)

    def augment(candidate_index: int, seen_references: set[int]) -> bool:
        for reference_index in adjacency[candidate_index]:
            if reference_index in seen_references:
                continue
            seen_references.add(reference_index)
            previous_candidate = matched_candidate_by_reference[reference_index]
            if previous_candidate == -1 or augment(previous_candidate, seen_references):
                matched_candidate_by_reference[reference_index] = candidate_index
                return True
        return False

    return sum(augment(index, set()) for index in range(len(candidate)))


def metric_row(predicted: int, reference: int, tp: int) -> dict[str, Any]:
    fp = predicted - tp
    fn = reference - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "reference_repedal_count": reference,
        "candidate_repedal_count": predicted,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def quantiles(values: Iterable[float]) -> dict[str, float | str]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        return {"median": "", "p75": "", "p90": "", "p95": ""}
    return {
        "median": float(np.quantile(array, 0.50)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
    }


def discover() -> dict[str, Any]:
    source_manifest_path = SOURCE_EVAL / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_failures = json.loads(
        (SOURCE_EVAL / "alignment/failures.json").read_text(encoding="utf-8")
    )["failures"]
    models = [str(value) for value in source_manifest["models"]]
    if models != ["original_pt", "encoder_only", "decoder_only", "encoder_decoder"]:
        raise RuntimeError(f"unexpected frozen model inventory: {models}")

    model_midis: dict[str, dict[str, Path]] = {}
    source_hashes: dict[Path, str] = {}
    for model in models:
        by_piece: dict[str, Path] = {}
        for metadata_path in sorted((SOURCE_EVAL / "predictions" / model).glob("*/metadata.json")):
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            midi_path = container_path(metadata["candidate_midi"])
            piece_id = str(metadata["piece_id"])
            if not metadata.get("complete") or not midi_path.is_file():
                raise RuntimeError(f"incomplete saved candidate: {metadata_path}")
            actual_hash = sha256_file(midi_path)
            if actual_hash != metadata["candidate_midi_sha256"]:
                raise RuntimeError(f"saved candidate hash changed: {midi_path}")
            by_piece[piece_id] = midi_path
            source_hashes[midi_path] = actual_hash
        model_midis[model] = by_piece

    extreme_rows = read_csv(EXTREME_MANIFEST) + read_csv(PILOT_EXTREME_MANIFEST)
    extremes: dict[str, dict[str, Path]] = {"ALWAYS_OFF": {}, "ALWAYS_ON": {}}
    for row in extreme_rows:
        name = "ALWAYS_OFF" if row["variant"] == "NO_PEDAL" else row["variant"]
        if name not in extremes:
            continue
        path = container_path(row.get("extreme_midi_path") or row["generated_midi_path"])
        expected_hash = row.get("extreme_sha256") or row["generated_sha256"]
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"saved extreme missing or changed: {path}")
        if row["non_CC64_exact_identity"] != "True" or row["note_exact_identity"] != "True":
            raise RuntimeError(f"saved extreme failed prior identity audit: {path}")
        extremes[name][row["piece_id"]] = path
        source_hashes[path] = expected_hash

    human_inventory: dict[str, dict[str, str]] = {}
    for row in read_csv(HUMAN_INVENTORY):
        if row["system"] == "HUMAN":
            human_inventory[row["piece_id"]] = row
    validation = [row for row in read_csv(SPLIT_CSV) if row["split"] == "validation"]
    failed_performances = {row["human_performance"] for row in source_failures}
    successful = [row for row in validation if row["performance_path"] not in failed_performances]
    by_identifier = {row["metadata_index"]: row for row in successful}
    if len(validation) != 71 or len(successful) != 70:
        raise RuntimeError("frozen validation universe is no longer 71 rows / 70 successes")
    human_alignments: dict[str, dict[str, Any]] = {}
    for identifier, row in by_identifier.items():
        metadata_path = SOURCE_EVAL / "alignment/human" / identifier / "metadata.json"
        transitions_path = metadata_path.parent / "transitions.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        midi_path = container_path(metadata["performance_path"])
        expected = ASAP_ROOT / row["performance_path"]
        if midi_path.resolve() != expected.resolve() or metadata["performance_sha256"] != sha256_file(expected):
            raise RuntimeError(f"frozen human alignment/source mismatch: {identifier}")
        human_alignments[identifier] = {
            "row": row,
            "midi": expected,
            "transitions": transitions_path,
        }
        source_hashes[expected] = metadata["performance_sha256"]

    piece_sets = [set(value) for value in model_midis.values()]
    piece_sets += [set(extremes["ALWAYS_OFF"]), set(extremes["ALWAYS_ON"]), set(human_inventory)]
    common_pieces = sorted(set.intersection(*piece_sets))
    if not common_pieces:
        raise RuntimeError("saved candidate sets have no common validation pieces")
    common_pairs = [row for row in successful if row["piece_id"] in common_pieces]
    return {
        "source_manifest_path": source_manifest_path,
        "source_manifest": source_manifest,
        "source_failures": source_failures,
        "models": models,
        "model_midis": model_midis,
        "extremes": extremes,
        "human_inventory": human_inventory,
        "human_alignments": human_alignments,
        "common_pieces": common_pieces,
        "common_pairs": common_pairs,
        "source_hashes": source_hashes,
    }


def run(output: Path) -> int:
    if "torch" in sys.modules:
        raise RuntimeError("CPU-only repedal evaluator must not import torch")
    discovered = discover()
    output.mkdir(parents=True, exist_ok=True)

    reference_events: dict[str, list[dict[str, Any]]] = {}
    all_event_rows: list[dict[str, Any]] = []
    common_ids = {row["metadata_index"] for row in discovered["common_pairs"]}
    for identifier in sorted(common_ids, key=int):
        item = discovered["human_alignments"][identifier]
        mapping = transition_mapping(item["transitions"], item["midi"])
        row = item["row"]
        events = extract_repedals(
            item["midi"],
            mapping=mapping,
            candidate="REFERENCE",
            piece_id=row["piece_id"],
            performance_id=row["performance_path"],
            role="reference",
        )
        reference_events[identifier] = events
        all_event_rows.extend(events)

    candidate_specs: list[tuple[str, dict[str, Path], dict[str, str]]] = []
    human_paths = {
        piece: container_path(row["midi_path"])
        for piece, row in discovered["human_inventory"].items()
    }
    human_ids = {
        piece: row["performance_id"] for piece, row in discovered["human_inventory"].items()
    }
    candidate_specs.append(("Human", human_paths, human_ids))
    candidate_specs.append(
        ("ALWAYS_ON", discovered["extremes"]["ALWAYS_ON"], {piece: piece for piece in discovered["common_pieces"]})
    )
    candidate_specs.append(
        ("ALWAYS_OFF", discovered["extremes"]["ALWAYS_OFF"], {piece: piece for piece in discovered["common_pieces"]})
    )
    for model in discovered["models"]:
        candidate_specs.append(
            (MODEL_LABELS[model], discovered["model_midis"][model], {piece: piece for piece in discovered["common_pieces"]})
        )

    candidate_events: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for candidate, paths, identifiers in candidate_specs:
        for piece in discovered["common_pieces"]:
            midi_path = paths[piece]
            if candidate == "Human":
                identifier = next(
                    row["metadata_index"]
                    for row in discovered["common_pairs"]
                    if row["performance_path"] == identifiers[piece]
                )
                transition_path = discovered["human_alignments"][identifier]["transitions"]
            elif candidate in {"ALWAYS_ON", "ALWAYS_OFF"}:
                transition_path = None
            else:
                model = next(key for key, label in MODEL_LABELS.items() if label == candidate)
                transition_path = SOURCE_EVAL / "alignment" / model / piece / "transitions.json"
            mapping = None if transition_path is None else transition_mapping(transition_path, midi_path)
            events = extract_repedals(
                midi_path,
                mapping=mapping,
                candidate=candidate,
                piece_id=piece,
                performance_id=identifiers[piece],
                role="candidate",
            )
            candidate_events[(candidate, piece)] = events
            all_event_rows.extend(events)

    summary_rows: list[dict[str, Any]] = []
    per_performance: list[dict[str, Any]] = []
    matching_diagnostics: list[dict[str, Any]] = []
    reference_by_piece: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in discovered["common_pairs"]:
        reference_by_piece[row["piece_id"]].append(row)
    for candidate, _, identifiers in candidate_specs:
        total_predicted = total_reference = total_tp = comparison_pairs = 0
        for piece in discovered["common_pieces"]:
            predicted = candidate_events[(candidate, piece)]
            for reference_row in reference_by_piece[piece]:
                if candidate == "Human" and reference_row["performance_path"] == identifiers[piece]:
                    continue
                target = reference_events[reference_row["metadata_index"]]
                matched = match_repedals(predicted, target)
                greedy_tp = len(matched)
                maximum_cardinality_tp = maximum_cardinality_match_count(
                    predicted, target
                )
                matching_diagnostics.append(
                    {
                        "candidate": candidate,
                        "piece_id": piece,
                        "candidate_performance_id": identifiers[piece],
                        "reference_performance_id": reference_row["performance_path"],
                        "greedy_tp": greedy_tp,
                        "maximum_cardinality_tp": maximum_cardinality_tp,
                        "tp_gap": maximum_cardinality_tp - greedy_tp,
                    }
                )
                metrics = metric_row(len(predicted), len(target), greedy_tp)
                total_predicted += len(predicted)
                total_reference += len(target)
                total_tp += len(matched)
                comparison_pairs += 1
                per_performance.append(
                    {
                        "candidate": candidate,
                        "piece_id": piece,
                        "candidate_performance_id": identifiers[piece],
                        "reference_performance_id": reference_row["performance_path"],
                        **metrics,
                    }
                )
        metrics = metric_row(total_predicted, total_reference, total_tp)
        unique_events = [
            event
            for piece in discovered["common_pieces"]
            for event in candidate_events[(candidate, piece)]
        ]
        duration = quantiles(event["off_duration_ms"] for event in unique_events)
        summary_rows.append(
            {
                "candidate": candidate,
                "num_performances": len(discovered["common_pieces"]),
                "num_comparison_pairs": comparison_pairs,
                **metrics,
                "unique_candidate_repedal_count": len(unique_events),
                "candidate_off_duration_median_ms": duration["median"],
                "candidate_off_duration_p75_ms": duration["p75"],
                "candidate_off_duration_p90_ms": duration["p90"],
                "candidate_off_duration_p95_ms": duration["p95"],
            }
        )

    unique_reference_events = [event for values in reference_events.values() for event in values]
    reference_duration = quantiles(event["off_duration_ms"] for event in unique_reference_events)
    event_fields = [
        "role", "candidate", "piece_id", "performance_id", "midi_path", "repedal_id",
        "prior_peak_index", "trough_index", "following_peak_index", "prior_peak_tick",
        "trough_tick", "following_peak_tick", "prior_peak_value", "trough_value",
        "following_peak_value", "release_excursion", "repress_excursion", "up_tick",
        "down_tick", "up_time_ms", "down_time_ms", "off_duration_ms", "up_onset_index",
        "down_onset_index", "up_score_position", "down_score_position",
    ]
    summary_fields = [
        "candidate", "num_performances", "num_comparison_pairs", "reference_repedal_count",
        "candidate_repedal_count", "tp", "fp", "fn", "precision", "recall", "f1",
        "unique_candidate_repedal_count", "candidate_off_duration_median_ms",
        "candidate_off_duration_p75_ms", "candidate_off_duration_p90_ms",
        "candidate_off_duration_p95_ms",
    ]
    per_fields = [
        "candidate", "piece_id", "candidate_performance_id", "reference_performance_id",
        "reference_repedal_count", "candidate_repedal_count", "tp", "fp", "fn",
        "precision", "recall", "f1",
    ]
    matching_diagnostic_fields = [
        "candidate", "piece_id", "candidate_performance_id",
        "reference_performance_id", "greedy_tp", "maximum_cardinality_tp",
        "tp_gap",
    ]
    write_csv(output / "repedal_validation_summary.csv", summary_rows, summary_fields)
    write_csv(output / "repedal_per_performance.csv", per_performance, per_fields)
    write_csv(output / "repedal_events.csv", all_event_rows, event_fields)
    write_csv(
        output / "greedy_vs_maximum_matching.csv",
        matching_diagnostics,
        matching_diagnostic_fields,
    )

    always_on = next(row for row in summary_rows if row["candidate"] == "ALWAYS_ON")
    always_off = next(row for row in summary_rows if row["candidate"] == "ALWAYS_OFF")
    accepted_valid = all(
        int(event["release_excursion"]) >= MIN_EXCURSION
        and int(event["repress_excursion"]) >= MIN_EXCURSION
        and event["up_tick"] != ""
        and event["down_tick"] != ""
        and 0.0 < float(event["off_duration_ms"]) <= MAX_OFF_DURATION_MS
        for event in all_event_rows
    )
    episode_counts: dict[tuple[str, str, str, int, int], int] = defaultdict(int)
    for event in all_event_rows:
        episode_counts[
            (
                str(event["midi_path"]),
                str(event["role"]),
                str(event["candidate"]),
                int(event["up_tick"]),
                int(event["down_tick"]),
            )
        ] += 1
    duplicate_episode_count = sum(
        count - 1 for count in episode_counts.values() if count > 1
    )
    pairs_with_tp_gap = sum(int(row["tp_gap"]) > 0 for row in matching_diagnostics)
    total_tp_gap = sum(int(row["tp_gap"]) for row in matching_diagnostics)
    unchanged = all(sha256_file(path) == digest for path, digest in discovered["source_hashes"].items())
    sanity = {
        "always_on_unique_candidate_repedal_count_zero": always_on["unique_candidate_repedal_count"] == 0,
        "always_off_unique_candidate_repedal_count_zero": always_off["unique_candidate_repedal_count"] == 0,
        "all_accepted_candidates_pass_frozen_definition": accepted_valid,
        "same_extraction_function_for_human_and_models": True,
        "frozen_transition_distinct_onset_mapping_reused": True,
        "duplicate_up_down_episode_count_zero": duplicate_episode_count == 0,
        "greedy_vs_maximum_pairs_with_tp_gap": pairs_with_tp_gap,
        "greedy_vs_maximum_total_tp_gap": total_tp_gap,
        "source_midi_hashes_unchanged": unchanged,
        "new_midi_files_created": 0,
        "model_inference_count": 0,
        "checkpoint_load_count": 0,
        "gpu_operation_count": 0,
        "asap_test_access_count": 0,
        "threshold": THRESHOLD,
        "minimum_excursion": MIN_EXCURSION,
        "maximum_off_duration_ms": MAX_OFF_DURATION_MS,
        "onset_tolerance": ONSET_TOLERANCE,
    }
    if not all(value is True for key, value in sanity.items() if isinstance(value, bool)):
        raise AssertionError(f"one or more sanity checks failed: {sanity}")
    write_json(output / "sanity_checks.json", sanity)

    inventory_rows = []
    inventory_rows.append({"candidate": "Human", "discovered_midi_count": len(discovered["human_inventory"]), "evaluated_midi_count": len(discovered["common_pieces"]), "source": str(HUMAN_INVENTORY)})
    inventory_rows.append({"candidate": "ALWAYS_ON", "discovered_midi_count": len(discovered["extremes"]["ALWAYS_ON"]), "evaluated_midi_count": len(discovered["common_pieces"]), "source": f"{EXTREME_MANIFEST}; {PILOT_EXTREME_MANIFEST}"})
    inventory_rows.append({"candidate": "ALWAYS_OFF", "discovered_midi_count": len(discovered["extremes"]["ALWAYS_OFF"]), "evaluated_midi_count": len(discovered["common_pieces"]), "source": f"{EXTREME_MANIFEST}; {PILOT_EXTREME_MANIFEST}"})
    for model in discovered["models"]:
        inventory_rows.append({"candidate": MODEL_LABELS[model], "discovered_midi_count": len(discovered["model_midis"][model]), "evaluated_midi_count": len(discovered["common_pieces"]), "source": str(SOURCE_EVAL / "predictions" / model)})
    write_csv(output / "candidate_inventory.csv", inventory_rows, ("candidate", "discovered_midi_count", "evaluated_midi_count", "source"))

    source_manifest_hash = sha256_file(discovered["source_manifest_path"])
    provenance = {
        "source_evaluation": str(SOURCE_EVAL),
        "execution_command": "docker exec -u 1004:1004 -e CUDA_VISIBLE_DEVICES= ilkyun-marg-pedaling-dev bash -lc \"cd /workspace/project && python scripts/evaluate_stage2_4class_repedal_validation_v0.py --execute\"",
        "source_report": str(SOURCE_EVAL / "FOUR_MODEL_VALIDATION_EVALUATION_REPORT.md"),
        "source_manifest": str(discovered["source_manifest_path"]),
        "source_manifest_sha256": source_manifest_hash,
        "source_common_successful_pairs": 70,
        "availability_filtered_common_pieces": discovered["common_pieces"],
        "availability_filtered_candidate_human_pairs": len(discovered["common_pairs"]),
        "human_baseline_source": str(HUMAN_INVENTORY),
        "human_self_comparisons": 0,
        "aggregation": "micro sums of TP/FP/FN over candidate-human comparison pairs",
        "repedal_matching": "greedy one-to-one, narrowest score-position boundary offsets first",
        "headline_matching": "frozen greedy one-to-one matching used for TP and Precision/Recall/F1",
        "maximum_cardinality_matching": "diagnostic only; never used for headline TP or Precision/Recall/F1",
    }
    write_json(output / "provenance.json", provenance)

    lines = [
        "# Canonical Repedal Validation Report",
        "",
        "## Scope and discovered candidate sets",
        "",
        "This CPU-only evaluation parsed saved validation MIDI files and added only the frozen Repedaling metric. It performed no inference, MIDI generation, checkpoint load, GPU operation, threshold tuning, or recomputation of prior 4C/Transition/JS metrics.",
        "",
        "| Candidate set | MIDI found | MIDI evaluated | Existing source |",
        "|---|---:|---:|---|",
    ]
    for row in inventory_rows:
        lines.append(f"| {row['candidate']} | {row['discovered_midi_count']} | {row['evaluated_midi_count']} | `{row['source']}` |")
    lines += [
        "",
        "## Reused evaluation universe",
        "",
        f"- Source universe: the frozen canonical 4C validation evaluation, 19 pieces / 71 Human performances / 70 successful candidate-Human pairs; its sole frozen alignment failure remains excluded.",
        f"- Common availability universe: {len(discovered['common_pieces'])} pieces / {len(discovered['common_pairs'])} candidate-Human pairs. This is the source 70-pair universe filtered only to pieces present in every reused candidate inventory (fixed Human representative plus both saved extremes); no MIDI was generated to fill the {19 - len(discovered['common_pieces'])} unavailable pieces.",
        "- All non-Human rows use exactly this same common pair universe and micro-aggregate TP/FP/FN as the Transition evaluator does.",
        "- Human baseline reuses the previously fixed representative Human MIDI per piece from `input_midis.csv` and compares it with the other successful Human performances for that piece. The identical performance is excluded, so Human is never compared with itself.",
        "",
        "## Frozen canonical definition",
        "",
        "Consecutive equal raw CC64 values are collapsed. Strict local peaks/troughs are detected; each trough uses the nearest prior and following peaks independently satisfying excursion >=64. Inside that triple, the last >=64 to <64 crossing before the trough and the first <64 to >=64 crossing after it define the OFF episode containing the trough. Event timestamps are the destination CC64 event timestamps, without interpolation, and only 0 < OFF duration <=300 ms is accepted.",
        "Multiple qualifying extrema triples that resolve to the same `(UP tick, DOWN tick)` are collapsed into one canonical repedal event using the frozen deterministic representative rule.",
        "",
        "Both UP and DOWN boundaries are mapped with the frozen Transition evaluator's cached distinct-onset score positions. A match requires both boundary offsets <=1 and uses deterministic greedy one-to-one matching, prioritizing the narrowest boundary offsets, consistent with the existing Transition matcher's greedy convention. Headline Precision/Recall/F1 continues to use this frozen greedy TP; maximum-cardinality TP is diagnostic only.",
        "",
        "## Results",
        "",
        "Counts below are micro counts over comparison pairs, matching the prior Transition aggregation (a piece-level candidate count is repeated for each Human reference paired with that piece). `Unique pred` is the diagnostic count over saved candidate MIDI files before pair repetition.",
        "",
        "| Candidate | Ref count | Pred count | Unique pred | TP | FP | FN | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['candidate']} | {row['reference_repedal_count']} | {row['candidate_repedal_count']} | {row['unique_candidate_repedal_count']} | {row['tp']} | {row['fp']} | {row['fn']} | {row['precision']:.6f} | {row['recall']:.6f} | {row['f1']:.6f} |"
        )
    lines += [
        "",
        "## OFF-duration diagnostics",
        "",
        "These diagnostics do not alter the frozen 300 ms threshold.",
        "",
        "| Candidate | Median ms | p75 | p90 | p95 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        def show(key: str) -> str:
            value = row[key]
            return "—" if value == "" else f"{float(value):.3f}"
        lines.append(
            f"| {row['candidate']} | {show('candidate_off_duration_median_ms')} | {show('candidate_off_duration_p75_ms')} | {show('candidate_off_duration_p90_ms')} | {show('candidate_off_duration_p95_ms')} |"
        )
    lines.append(
        f"| Human/reference (unique) | {reference_duration['median']:.3f} | {reference_duration['p75']:.3f} | {reference_duration['p90']:.3f} | {reference_duration['p95']:.3f} |"
    )
    lines += [
        "",
        "## Sanity checks",
        "",
        f"- ALWAYS_ON candidate repedals = {always_on['unique_candidate_repedal_count']}: PASS",
        f"- ALWAYS_OFF candidate repedals = {always_off['unique_candidate_repedal_count']}: PASS",
        "- Every accepted event has release/repress excursion >=64, both raw threshold crossings, and 0 < OFF duration <=300 ms: PASS",
        f"- Duplicate canonical `(UP tick, DOWN tick)` episodes after deduplication = {duplicate_episode_count}: {'PASS' if duplicate_episode_count == 0 else 'FAIL'}",
        f"- Greedy vs maximum-cardinality TP gap: {pairs_with_tp_gap} comparison pair(s), total gap {total_tp_gap}. Headline scores retain frozen greedy matching.",
        "- Human and model MIDI use the same extraction function: PASS",
        "- Frozen Transition distinct-onset mappings and source MIDI hashes were verified and reused: PASS",
        "- Source MIDI unchanged; new MIDI count 0; model inference/checkpoint/GPU/ASAP-test access counts all 0: PASS",
        "",
        "## Artifacts",
        "",
        "- `repedal_validation_summary.csv`",
        "- `repedal_per_performance.csv`",
        "- `repedal_events.csv`",
        "- `greedy_vs_maximum_matching.csv`",
        "- `candidate_inventory.csv`",
        "- `sanity_checks.json` and `provenance.json`",
        "- Tested command is recorded in `provenance.json`; it hides CUDA and invokes only this CPU parser/evaluator.",
        "",
    ]
    atomic_text(output / "REPEDAL_VALIDATION_REPORT.md", "\n".join(lines))
    print(json.dumps({"output": str(output), "summary": summary_rows, "sanity": sanity}, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("evaluation is guarded; pass --execute")
    return run(args.output_root)


if __name__ == "__main__":
    raise SystemExit(main())
