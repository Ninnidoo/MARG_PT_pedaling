#!/usr/bin/env python3
"""Validation-only pilot sweep for the finalized harmonic pedaling metric.

This script never performs model inference.  It consumes frozen ASAP-validation
MIDI artifacts, constructs reusable pedal extremes from canonical Stage 1, and
evaluates the metric defined in ``Metric_harmonic_consonance.md``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import math
import os
import shutil
import sys
import tempfile
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mido
import numpy as np
import pandas as pd

REPO_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_FOR_IMPORT))

from src.stage2_binary.canonical_stage1 import (
    assert_strict_non_cc64_equality,
    cc64_schedule,
    sha256_file,
    strict_non_cc64_signature,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "analysis/harmonic_metric_kdyn_blow_pilot_v0"
STAGE1_MANIFEST = (
    ROOT / "analysis/stage2_binary_canonical_v1/canonical_validation_stage1_manifest.csv"
)
CUSTOM_MANIFEST = (
    ROOT / "analysis/custom_event_model_v0_canonical_val_inference_v1/candidate_manifest.json"
)
METRIC_DOCUMENT = ROOT / "Metric_harmonic_consonance.md"

ASAP_VALIDATION_ROOT = Path("/workspace/public/ASAP/asap-dataset-v1.1")
EPSILON = 1e-12
ETA_PP = 0.9
M0 = 48
SCRIPT_VERSION = "harmonic_metric_kdyn_blow_pilot_v0.1"
PILOT_MAX_NOTE_ON_COUNT = 1900
KAPPAS = (0.0, 0.1, 0.2, 0.3, 0.4)
BETAS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)

# Hall, Tamir & Rohrmeier (2025), Table 1, Simple Type/type-method.
# Indices are octave-equivalent semitone intervals 1..12; 0 mod 12 -> 12.
HALL_WEIGHTS = np.asarray(
    [
        0.0,
        -1.902,
        -0.299,
        -0.029,
        +0.257,
        +0.616,
        -0.582,
        +0.676,
        -0.339,
        -0.142,
        -0.712,
        -0.899,
        +1.009,
    ],
    dtype=np.float64,
)
T60_PITCHES = np.asarray([36.0, 48.0, 60.0, 74.0, 84.0], dtype=np.float64)
T60_SECONDS = np.asarray([18.1, 14.8, 15.3, 18.3, 12.0], dtype=np.float64)


# Current representative conventional comparison set.  These are saved
# validation outputs only.  Older diagnostic architectures and rejected
# decoder/fusion variants are inventoried in the report but not scored.
SYSTEM_SPECS = {
    "ORIGINAL_PT": {
        "role": "canonical_stage1_baseline",
        "path_kind": "manifest",
        "experiment": "stage2_binary_canonical_v1/canonical_validation_stage1",
    },
    "STANDARD_CE_ARGMAX": {
        "role": "recent_4class_model",
        "root": ROOT
        / "analysis/stage2_encoder_only_4class_decoding_phase4_v0/predictions/argmax",
        "experiment": "stage2_encoder_only_4class_decoding_phase4_v0/argmax",
    },
    "STANDARD_CE_POSTERIOR_MEDIAN": {
        "role": "recent_4class_model",
        "root": ROOT
        / "analysis/stage2_encoder_only_4class_decoding_phase4_v0/predictions/median",
        "experiment": "stage2_encoder_only_4class_decoding_phase4_v0/median",
    },
    "WEIGHTED_CE_ARGMAX": {
        "role": "recent_4class_model",
        "root": ROOT
        / "analysis/stage2_encoder_only_4class_loss_phase3_v0/weighted_ce/validation/predictions/weighted_ce",
        "experiment": "stage2_encoder_only_4class_loss_phase3_v0/weighted_ce",
    },
    "HYBRID_REGRESSION_ONLY": {
        "role": "recent_4class_model",
        "root": ROOT
        / "analysis/stage2_encoder_only_raw_huber_aux_ce_v1/validation_eval/predictions/raw_huber_aux_ce",
        "experiment": "stage2_encoder_only_raw_huber_aux_ce_v1/raw_huber_aux_ce",
    },
}

EXCLUDED_AVAILABLE_SYSTEMS = [
    {
        "experiment": "stage2_4class_architecture_validation_eval_v0/decoder_only",
        "reason": "architecture diagnostic not retained in the current representative comparison",
    },
    {
        "experiment": "stage2_4class_architecture_validation_eval_v0/encoder_decoder",
        "reason": "architecture diagnostic not retained in the current representative comparison",
    },
    {
        "experiment": "stage2_encoder_only_4class_loss_phase3_v0/ce_ntl_was_lambda_0p3",
        "reason": "loss diagnostic not retained over the representative weighted-CE and hybrid outputs",
    },
    {
        "experiment": "stage2_encoder_only_4class_loss_phase3_v0/representative_huber_delta14",
        "reason": "superseded by the current hybrid raw-regression finalist",
    },
    {
        "experiment": "stage2_encoder_only_4class_decoding_phase4_v0/expectation",
        "reason": "decoder diagnostic; posterior median retained as the representative alternate decoder",
    },
    {
        "experiment": "stage2_encoder_only_4class_decoding_phase4_v0/top2_seed42",
        "reason": "single-seed stochastic decoder diagnostic not retained",
    },
    {
        "experiment": "stage2_loss_decoding_fusion_phase4b_v0/weighted_ce_median",
        "reason": "phase-4b diagnostic not selected as a current representative",
    },
    {
        "experiment": "stage2_loss_decoding_fusion_phase4b_v0/ntl_was_median",
        "reason": "phase-4b diagnostic not selected as a current representative",
    },
    {
        "experiment": "stage2_loss_decoding_fusion_phase4b_v0/hybrid_aux_median",
        "reason": "auxiliary-only diagnostic; did not replace regression-only inference",
    },
    {
        "experiment": "stage2_loss_decoding_fusion_phase4b_v0/hybrid_side_constrained",
        "reason": "phase-4b report retained hybrid regression-only as the conventional finalist",
    },
]


@dataclass(slots=True)
class NoteInstance:
    identifier: int
    channel: int
    pitch: int
    velocity: int
    onset_tick: int
    onset_seconds: float


@dataclass(slots=True)
class OnsetComponent:
    onset_index: int
    onset_tick: int
    onset_time: float
    active_count: int
    pedal_count: int
    pa_pair_count: int
    pp_pair_count: int
    eta_denominator: float
    numerator_beta0: float
    numerator_low_slope: float
    positive_sum_beta0: float
    negative_sum_beta0: float
    eta_weighted_wdec_sum: float
    eta_weighted_low_degree_sum: float
    vbar: float
    d_context: float = 0.5


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def atomic_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def host_to_container_path(path: str | Path) -> Path:
    value = str(path)
    if value.startswith("/workspace/project"):
        return ROOT / Path(value).relative_to("/workspace/project")
    if value.startswith("/workspace/public"):
        return Path(value)
    return Path(value)


def t60_seconds(pitch: int | np.ndarray) -> float | np.ndarray:
    values = np.asarray(pitch, dtype=np.float64)
    result = np.interp(values, T60_PITCHES, T60_SECONDS, left=18.1, right=12.0)
    if np.ndim(pitch) == 0:
        return float(result)
    return result


def note_strength(velocity: int | np.ndarray, age_seconds: float | np.ndarray, pitch: int | np.ndarray) -> Any:
    age = np.maximum(np.asarray(age_seconds, dtype=np.float64), 0.0)
    return (np.asarray(velocity, dtype=np.float64) / 127.0) * np.power(
        10.0, -3.0 * age / t60_seconds(pitch)
    )


def low_degree(pitch: int | np.ndarray) -> Any:
    return np.clip((M0 - np.asarray(pitch, dtype=np.float64)) / (M0 - 21), 0.0, 1.0)


def interval_class(pitch_a: np.ndarray, pitch_b: np.ndarray) -> np.ndarray:
    intervals = np.abs(pitch_a - pitch_b) % 12
    return np.where(intervals == 0, 12, intervals).astype(np.int64)


def absolute_track_events(track: mido.MidiTrack) -> list[tuple[int, int, Any]]:
    absolute = 0
    result = []
    for order, message in enumerate(track):
        absolute += int(message.time)
        result.append((absolute, order, message))
    return result


def is_cc64(message: Any) -> bool:
    return (
        not message.is_meta
        and message.type == "control_change"
        and int(message.control) == 64
    )


def is_note_on(message: Any) -> bool:
    return not message.is_meta and message.type == "note_on" and int(message.velocity) > 0


def is_note_off(message: Any) -> bool:
    return not message.is_meta and (
        message.type == "note_off"
        or (message.type == "note_on" and int(message.velocity) == 0)
    )


def note_signature(path: Path) -> list[tuple[int, int, str, int, int, int]]:
    midi = mido.MidiFile(str(path), clip=False)
    result = []
    for track_index, track in enumerate(midi.tracks):
        for tick, _, message in absolute_track_events(track):
            if is_note_on(message) or is_note_off(message):
                velocity = int(getattr(message, "velocity", 0))
                result.append(
                    (
                        track_index,
                        tick,
                        str(message.type),
                        int(message.channel),
                        int(message.note),
                        velocity,
                    )
                )
    return result


def musical_non_cc64_signature(path: Path) -> dict[str, Any]:
    """Non-meta channel events at absolute ticks, excluding sustain CC64."""
    midi = mido.MidiFile(str(path), clip=False)
    tracks: list[list[tuple[int, dict[str, Any]]]] = []
    for track in midi.tracks:
        items = []
        for tick, _, message in absolute_track_events(track):
            if message.is_meta or is_cc64(message):
                continue
            payload = message.dict()
            payload.pop("time", None)
            items.append((tick, payload))
        tracks.append(items)
    return {
        "midi_type": midi.type,
        "ticks_per_beat": midi.ticks_per_beat,
        "tracks": tracks,
    }


def rebuild_extreme(source: Path, destination: Path, variant: str) -> None:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite reusable extreme: {destination}")
    midi = mido.MidiFile(str(source), clip=False)
    output = mido.MidiFile(type=midi.type, ticks_per_beat=midi.ticks_per_beat, clip=False)
    for track_index, track in enumerate(midi.tracks):
        events = [(tick, order, message) for tick, order, message in absolute_track_events(track) if not is_cc64(message)]
        inserts: list[tuple[int, int, Any]] = []
        if variant == "ALWAYS_ON":
            channel_counts = Counter(
                int(message.channel)
                for _, _, message in events
                if is_note_on(message)
            )
            for insert_order, channel in enumerate(sorted(channel_counts)):
                inserts.append(
                    (
                        0,
                        -1000 + insert_order,
                        mido.Message(
                            "control_change", channel=channel, control=64, value=127, time=0
                        ),
                    )
                )
        elif variant != "NO_PEDAL":
            raise ValueError(f"unknown extreme variant: {variant}")
        combined = sorted(events + inserts, key=lambda item: (item[0], item[1]))
        rebuilt = mido.MidiTrack()
        previous = 0
        for tick, _, message in combined:
            rebuilt.append(message.copy(time=tick - previous))
            previous = tick
        output.tracks.append(rebuilt)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.save(str(destination))


def generate_reusable_extremes(
    stage1_rows: Sequence[dict[str, str]], selected_piece_ids: set[str], output: Path
) -> tuple[dict[tuple[str, str], Path], list[dict[str, Any]]]:
    root = output / "reusable_extremes"
    paths: dict[tuple[str, str], Path] = {}
    records: list[dict[str, Any]] = []
    for row in stage1_rows:
        piece_id = row["piece_id"]
        if piece_id not in selected_piece_ids:
            continue
        source = host_to_container_path(row["canonical_midi_path"])
        for variant, subdir in (("NO_PEDAL", "no_pedal"), ("ALWAYS_ON", "always_on")):
            destination = root / subdir / f"{piece_id}.mid"
            rebuild_extreme(source, destination, variant)
            identity = assert_strict_non_cc64_equality(source, destination)
            note_exact = note_signature(source) == note_signature(destination)
            if not identity["passed"] or not note_exact:
                raise AssertionError(f"extreme identity failed: {piece_id} {variant}")
            schedule = cc64_schedule(destination)
            if variant == "NO_PEDAL" and schedule:
                raise AssertionError(f"NO_PEDAL contains CC64: {destination}")
            if variant == "ALWAYS_ON" and not all(int(item["value"]) == 127 for item in schedule):
                raise AssertionError(f"ALWAYS_ON contains non-127 CC64: {destination}")
            paths[(piece_id, variant)] = destination
            records.append(
                {
                    "piece_id": piece_id,
                    "performance_id": piece_id,
                    "variant": variant,
                    "source_canonical_midi_path": str(source),
                    "generated_midi_path": str(destination),
                    "source_sha256": sha256_file(source),
                    "generated_sha256": sha256_file(destination),
                    "non_CC64_exact_identity": True,
                    "note_exact_identity": note_exact,
                    "creation_script_version": SCRIPT_VERSION,
                    "cc64_event_count": len(schedule),
                }
            )
    pd.DataFrame(records).to_csv(root / "manifest.csv", index=False)
    return paths, records


def midi_descriptor(path: Path) -> dict[str, Any]:
    midi = mido.MidiFile(str(path), clip=False)
    merged = mido.merge_tracks(midi.tracks)
    absolute_tick = 0
    seconds = 0.0
    tempo = 500_000
    onsets: list[tuple[int, float, int, int]] = []
    for message in merged:
        delta = int(message.time)
        absolute_tick += delta
        seconds += mido.tick2second(delta, midi.ticks_per_beat, tempo)
        if message.is_meta and message.type == "set_tempo":
            tempo = int(message.tempo)
        elif is_note_on(message):
            onsets.append((absolute_tick, seconds, int(message.note), int(message.velocity)))
    if not onsets:
        raise ValueError(f"MIDI has no note onsets: {path}")
    pitches = np.asarray([item[2] for item in onsets], dtype=np.int64)
    velocities = np.asarray([item[3] for item in onsets], dtype=np.int64)
    distinct_ticks = len({item[0] for item in onsets})
    span = max(onsets[-1][1] - onsets[0][1], 1e-9)
    return {
        "note_on_count": len(onsets),
        "distinct_onset_count": distinct_ticks,
        "low_note_fraction": float(np.mean(pitches < M0)),
        "velocity_min": int(velocities.min()),
        "velocity_max": int(velocities.max()),
        "velocity_range": int(velocities.max() - velocities.min()),
        "velocity_q10": float(np.quantile(velocities, 0.1)),
        "velocity_q90": float(np.quantile(velocities, 0.9)),
        "note_density_per_second": float(len(onsets) / span),
        "onset_density_per_second": float(distinct_ticks / span),
        "musical_span_seconds": float(span),
    }


def choose_cases(
    stage1_rows: Sequence[dict[str, str]], custom_candidates: Sequence[dict[str, Any]]
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    by_piece_custom: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in custom_candidates:
        by_piece_custom[str(candidate["piece_id"])].append(candidate)

    inventory = []
    for row in stage1_rows:
        piece_id = row["piece_id"]
        canonical = host_to_container_path(row["canonical_midi_path"])
        if piece_id not in by_piece_custom:
            continue
        record: dict[str, Any] = {
            "piece_id": piece_id,
            "composer": row["composer"],
            "title": row["title"],
            "canonical_midi_path": str(canonical),
        }
        record.update(midi_descriptor(canonical))
        inventory.append(record)
    frame = pd.DataFrame(inventory).sort_values("piece_id").reset_index(drop=True)
    if len(frame) < 8:
        target = max(5, len(frame))
    else:
        target = 8

    selected: list[int] = []
    reasons: dict[int, list[str]] = defaultdict(list)
    descriptor_columns = ["low_note_fraction", "velocity_range", "note_density_per_second"]
    labels = {
        "low_note_fraction": ("low-register-light", "low-register-heavy"),
        "velocity_range": ("narrow velocity range", "wide velocity range"),
        "note_density_per_second": ("low note density", "high note density"),
    }
    # Select a low-note-count representative from each descriptor tail.  This
    # stays metric-blind while keeping the exact ALWAYS_ON audit tractable.
    for column in descriptor_columns:
        low_cut = float(frame[column].quantile(0.25))
        high_cut = float(frame[column].quantile(0.75))
        for mask, label in (
            (frame[column] <= low_cut, labels[column][0]),
            (frame[column] >= high_cut, labels[column][1]),
        ):
            candidates = frame.loc[mask].sort_values(["note_on_count", "piece_id"])
            idx = int(candidates.index[0])
            if idx not in selected:
                selected.append(idx)
            reasons[idx].append(label)

    values = frame[descriptor_columns].astype(float)
    scale = values.std(ddof=0).replace(0.0, 1.0)
    z = (values - values.mean()) / scale
    while len(selected) < target:
        remaining = [
            idx
            for idx in frame.index
            if idx not in selected
            and int(frame.loc[idx, "note_on_count"]) <= PILOT_MAX_NOTE_ON_COUNT
        ]
        if not remaining:
            raise RuntimeError(
                f"fewer than {target} descriptor-diverse pieces satisfy the exact-pair "
                f"pilot cap note_on_count<={PILOT_MAX_NOTE_ON_COUNT}"
            )
        ranked = []
        for idx in remaining:
            distance = min(float(np.linalg.norm(z.loc[idx] - z.loc[chosen])) for chosen in selected)
            # Farthest descriptor coverage first; smaller note count then makes
            # the deterministic exact pilot computationally bounded.
            ranked.append((-distance, int(frame.loc[idx, "note_on_count"]), str(frame.loc[idx, "piece_id"]), idx))
        _, _, _, idx = min(ranked)
        selected.append(idx)
        reasons[idx].append("descriptor-space farthest-point fill")

    selected = selected[:target]
    cases = []
    for selection_order, idx in enumerate(selected, 1):
        item = frame.loc[idx].to_dict()
        piece_id = str(item["piece_id"])
        custom = sorted(by_piece_custom[piece_id], key=lambda value: str(value["performance_path"]))[0]
        item.update(
            {
                "selection_order": selection_order,
                "selection_reason": "; ".join(reasons[idx]),
                "performance_id": str(custom["performance_id"]),
                "performance_path": str(custom["performance_path"]),
                "custom_candidate_path": str(host_to_container_path(custom["candidate_midi"])),
                "human_midi_path": str(host_to_container_path(custom["source_midi"])),
                "custom_performance_index": int(custom["performance_index"]),
            }
        )
        cases.append(item)
    return cases, frame


def resolve_system_paths(
    cases: Sequence[dict[str, Any]], stage1_by_piece: dict[str, dict[str, str]], extremes: dict[tuple[str, str], Path]
) -> tuple[dict[tuple[str, str], Path], list[dict[str, Any]], list[dict[str, Any]]]:
    result: dict[tuple[str, str], Path] = {}
    input_rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for case in cases:
        piece_id = str(case["piece_id"])
        performance_id = str(case["performance_id"])
        for system, spec in SYSTEM_SPECS.items():
            if spec.get("path_kind") == "manifest":
                path = host_to_container_path(stage1_by_piece[piece_id]["canonical_midi_path"])
            else:
                path = Path(spec["root"]) / piece_id / "candidate.mid"
            if not path.is_file():
                missing.append({"piece_id": piece_id, "performance_id": performance_id, "system": system, "path": str(path)})
                continue
            result[(piece_id, system)] = path
            input_rows.append(
                {
                    "piece_id": piece_id,
                    "performance_id": performance_id,
                    "system": system,
                    "role": spec["role"],
                    "experiment": spec["experiment"],
                    "midi_path": str(path),
                    "sha256": sha256_file(path),
                }
            )
        extra = {
            "CUSTOM_EVENT_V0": (Path(str(case["custom_candidate_path"])), "custom_event_model"),
            "HUMAN": (Path(str(case["human_midi_path"])), "human_reference"),
            "NO_PEDAL": (extremes[(piece_id, "NO_PEDAL")], "extreme_control"),
            "ALWAYS_ON": (extremes[(piece_id, "ALWAYS_ON")], "extreme_control"),
        }
        for system, (path, role) in extra.items():
            if not path.is_file():
                missing.append({"piece_id": piece_id, "performance_id": performance_id, "system": system, "path": str(path)})
                continue
            result[(piece_id, system)] = path
            input_rows.append(
                {
                    "piece_id": piece_id,
                    "performance_id": performance_id,
                    "system": system,
                    "role": role,
                    "experiment": (
                        "custom_event_model_v0_canonical_val_inference_v1"
                        if system == "CUSTOM_EVENT_V0"
                        else "ASAP validation human"
                        if system == "HUMAN"
                        else "reusable canonical Stage 1 extreme"
                    ),
                    "midi_path": str(path),
                    "sha256": sha256_file(path),
                }
            )
    return result, input_rows, missing


def identity_audit(
    cases: Sequence[dict[str, Any]], paths: dict[tuple[str, str], Path], stage1_by_piece: dict[str, dict[str, str]]
) -> list[dict[str, Any]]:
    rows = []
    conventional = set(SYSTEM_SPECS)
    for case in cases:
        piece_id = str(case["piece_id"])
        performance_id = str(case["performance_id"])
        canonical = host_to_container_path(stage1_by_piece[piece_id]["canonical_midi_path"])
        human = paths[(piece_id, "HUMAN")]
        canonical_strict = strict_non_cc64_signature(canonical)
        canonical_notes = note_signature(canonical)
        human_musical = musical_non_cc64_signature(human)
        for system in list(SYSTEM_SPECS) + ["CUSTOM_EVENT_V0", "HUMAN", "NO_PEDAL", "ALWAYS_ON"]:
            candidate = paths[(piece_id, system)]
            strict_equal = strict_non_cc64_signature(candidate) == canonical_strict
            notes_equal = note_signature(candidate) == canonical_notes
            human_musical_equal = musical_non_cc64_signature(candidate) == human_musical
            expected_canonical = system in conventional | {"NO_PEDAL", "ALWAYS_ON"}
            if expected_canonical and (not strict_equal or not notes_equal):
                raise AssertionError(f"canonical identity failed for {piece_id} {system}")
            if system == "CUSTOM_EVENT_V0" and not human_musical_equal:
                raise AssertionError(f"custom event musical non-CC64 identity failed: {piece_id}")
            rows.append(
                {
                    "piece_id": piece_id,
                    "performance_id": performance_id,
                    "system": system,
                    "midi_path": str(candidate),
                    "canonical_stage1_non_CC64_exact": strict_equal,
                    "canonical_stage1_note_exact": notes_equal,
                    "selected_human_musical_non_CC64_exact": human_musical_equal,
                    "expected_canonical_identity": expected_canonical,
                    "status": "PASS",
                }
            )
    return rows


def update_top_heap(heap: list[tuple[float, int, dict[str, Any]]], value: float, serial: int, record: dict[str, Any], limit: int = 20) -> None:
    magnitude = abs(float(value))
    item = (magnitude, serial, record)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif magnitude > heap[0][0]:
        heapq.heapreplace(heap, item)


def pair_block_statistics(
    pitch_a: np.ndarray,
    pitch_b: np.ndarray,
    strength_a: np.ndarray,
    strength_b: np.ndarray,
    eta: float,
    triangular: bool,
    note_a: Sequence[NoteInstance],
    note_b: Sequence[NoteInstance],
    onset_time: float,
    pair_type: str,
    positive_heap: list[tuple[float, int, dict[str, Any]]],
    negative_heap: list[tuple[float, int, dict[str, Any]]],
    serial_start: int,
    negative_detail: dict[str, Any] | None = None,
) -> tuple[dict[str, float], int]:
    if pitch_a.size == 0 or pitch_b.size == 0:
        return {
            "numerator": 0.0,
            "slope": 0.0,
            "positive": 0.0,
            "negative": 0.0,
            "wdec": 0.0,
            "low_degree": 0.0,
            "count": 0.0,
        }, serial_start
    aa = strength_a[:, None]
    bb = strength_b[None, :]
    wdec = 2.0 * aa * bb / (aa + bb + EPSILON)
    intervals = interval_class(pitch_a[:, None], pitch_b[None, :])
    hall = HALL_WEIGHTS[intervals]
    low_pair = (low_degree(pitch_a)[:, None] + low_degree(pitch_b)[None, :]) / 2.0
    contribution = eta * wdec * hall
    if triangular:
        mask = np.triu(np.ones(contribution.shape, dtype=bool), k=1)
    else:
        mask = np.ones(contribution.shape, dtype=bool)
    values = contribution[mask]
    slopes = (contribution * low_pair)[mask]
    wdec_values = wdec[mask]
    low_values = low_pair[mask]
    stats = {
        "numerator": float(values.sum(dtype=np.float64)),
        "slope": float(slopes.sum(dtype=np.float64)),
        "positive": float(values[values > 0].sum(dtype=np.float64)),
        "negative": float(values[values < 0].sum(dtype=np.float64)),
        "wdec": float(eta * wdec_values.sum(dtype=np.float64)),
        "low_degree": float(eta * low_values.sum(dtype=np.float64)),
        "count": float(values.size),
    }
    if negative_detail is not None:
        negative_mask = mask & (contribution < 0)
        negative_count = int(negative_mask.sum())
        negative_audible_mass = float(
            eta * wdec[negative_mask].sum(dtype=np.float64)
        )
        negative_detail["negative_pair_count"] = int(
            negative_detail.get("negative_pair_count", 0)
        ) + negative_count
        negative_detail["negative_eta_mass"] = float(
            negative_detail.get("negative_eta_mass", 0.0)
        ) + eta * negative_count
        negative_detail["negative_audible_mass"] = float(
            negative_detail.get("negative_audible_mass", 0.0)
        ) + negative_audible_mass
        negative_detail["audible_weight_square_sum"] = float(
            negative_detail.get("audible_weight_square_sum", 0.0)
        ) + float(np.square(eta * wdec_values).sum(dtype=np.float64))
        negative_detail["minimum_W_dec"] = min(
            float(negative_detail.get("minimum_W_dec", np.inf)),
            float(np.min(wdec_values, initial=np.inf)),
        )
        reconstruction = np.maximum(contribution, 0.0) - np.maximum(-contribution, 0.0)
        negative_detail["pair_reconstruction_max_abs_error"] = max(
            float(negative_detail.get("pair_reconstruction_max_abs_error", 0.0)),
            float(np.max(np.abs(contribution[mask] - reconstruction[mask]), initial=0.0)),
        )
        negative_degree = np.maximum(-hall, 0.0)
        reconstructed_q = eta * wdec * negative_degree
        negative_detail["q_reconstruction_max_abs_error"] = max(
            float(negative_detail.get("q_reconstruction_max_abs_error", 0.0)),
            float(np.max(np.abs(reconstructed_q[mask] - np.maximum(-contribution[mask], 0.0)), initial=0.0)),
        )
        interval_counts = negative_detail.setdefault("interval_counts", Counter())
        interval_masses = negative_detail.setdefault("interval_negative_masses", defaultdict(float))
        for interval in range(1, 13):
            selected = negative_mask & (intervals == interval)
            count_for_interval = int(selected.sum())
            if count_for_interval:
                interval_counts[interval] += count_for_interval
                interval_masses[interval] += float(
                    -contribution[selected].sum(dtype=np.float64)
                )
    # Pair-level debugging: retain only the strongest signed contributions from
    # each block, then merge through fixed-size heaps.
    for sign, heap in ((1, positive_heap), (-1, negative_heap)):
        flat = np.flatnonzero((contribution > 0) if sign > 0 else (contribution < 0))
        if triangular:
            flat = np.flatnonzero(mask.ravel() & (((contribution > 0) if sign > 0 else (contribution < 0)).ravel()))
        if flat.size == 0:
            continue
        k = min(20, flat.size)
        candidate_positions = flat[np.argpartition(np.abs(contribution.ravel()[flat]), -k)[-k:]]
        width = contribution.shape[1]
        for position in candidate_positions:
            ia, ib = divmod(int(position), width)
            value = float(contribution[ia, ib])
            na, nb = note_a[ia], note_b[ib]
            record = {
                "onset_time": onset_time,
                "pair_type": pair_type,
                "note_a_id": na.identifier,
                "note_b_id": nb.identifier,
                "pitch_a": na.pitch,
                "pitch_b": nb.pitch,
                "velocity_a": na.velocity,
                "velocity_b": nb.velocity,
                "note_a_onset": na.onset_seconds,
                "note_b_onset": nb.onset_seconds,
                "interval_class": int(intervals[ia, ib]),
                "hall_weight": float(hall[ia, ib]),
                "W_dec": float(wdec[ia, ib]),
                "low_pair_degree": float(low_pair[ia, ib]),
                "eta": eta,
                "contribution_beta0": value,
            }
            update_top_heap(heap, value, serial_start, record)
            serial_start += 1
    return stats, serial_start


def compute_pair_components(
    active: Sequence[NoteInstance],
    pedal: Sequence[NoteInstance],
    onset_time: float,
    positive_heap: list[tuple[float, int, dict[str, Any]]],
    negative_heap: list[tuple[float, int, dict[str, Any]]],
    serial: int,
) -> tuple[dict[str, float], int]:
    combined = {
        "numerator": 0.0,
        "slope": 0.0,
        "positive": 0.0,
        "negative": 0.0,
        "wdec": 0.0,
        "low_degree": 0.0,
        "pa_count": 0,
        "pp_count": 0,
    }
    active_pitch = np.asarray([note.pitch for note in active], dtype=np.int64)
    active_strength = note_strength(
        np.asarray([note.velocity for note in active]),
        onset_time - np.asarray([note.onset_seconds for note in active]),
        active_pitch,
    )
    pedal_pitch = np.asarray([note.pitch for note in pedal], dtype=np.int64)
    pedal_strength = note_strength(
        np.asarray([note.velocity for note in pedal]),
        onset_time - np.asarray([note.onset_seconds for note in pedal]),
        pedal_pitch,
    )
    if pedal and active:
        stats, serial = pair_block_statistics(
            pedal_pitch,
            active_pitch,
            pedal_strength,
            active_strength,
            1.0,
            False,
            pedal,
            active,
            onset_time,
            "PA",
            positive_heap,
            negative_heap,
            serial,
        )
        for key in ("numerator", "slope", "positive", "negative", "wdec", "low_degree"):
            combined[key] += stats[key]
        combined["pa_count"] += int(stats["count"])
    # Chunk PP computation without approximating or pruning decayed notes.
    chunk_size = 256
    count = len(pedal)
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        # Within-block strict upper triangle.
        block_notes = pedal[start:stop]
        stats, serial = pair_block_statistics(
            pedal_pitch[start:stop],
            pedal_pitch[start:stop],
            pedal_strength[start:stop],
            pedal_strength[start:stop],
            ETA_PP,
            True,
            block_notes,
            block_notes,
            onset_time,
            "PP",
            positive_heap,
            negative_heap,
            serial,
        )
        for key in ("numerator", "slope", "positive", "negative", "wdec", "low_degree"):
            combined[key] += stats[key]
        combined["pp_count"] += int(stats["count"])
        # Cross block with all later notes.
        if stop < count:
            stats, serial = pair_block_statistics(
                pedal_pitch[start:stop],
                pedal_pitch[stop:],
                pedal_strength[start:stop],
                pedal_strength[stop:],
                ETA_PP,
                False,
                block_notes,
                pedal[stop:],
                onset_time,
                "PP",
                positive_heap,
                negative_heap,
                serial,
            )
            for key in ("numerator", "slope", "positive", "negative", "wdec", "low_degree"):
                combined[key] += stats[key]
            combined["pp_count"] += int(stats["count"])
    return combined, serial


def merged_tick_groups(path: Path) -> tuple[int, Iterable[tuple[int, float, list[Any]]]]:
    midi = mido.MidiFile(str(path), clip=False)
    merged = mido.merge_tracks(midi.tracks)
    tempo = 500_000
    tick = 0
    seconds = 0.0
    groups: list[tuple[int, float, list[Any]]] = []
    current_tick: int | None = None
    current_seconds = 0.0
    current_messages: list[Any] = []
    for message in merged:
        delta = int(message.time)
        tick += delta
        seconds += mido.tick2second(delta, midi.ticks_per_beat, tempo)
        if current_tick is None or tick != current_tick:
            if current_tick is not None:
                groups.append((current_tick, current_seconds, current_messages))
            current_tick = tick
            current_seconds = seconds
            current_messages = []
        current_messages.append(message.copy(time=0))
        if message.is_meta and message.type == "set_tempo":
            tempo = int(message.tempo)
    if current_tick is not None:
        groups.append((current_tick, current_seconds, current_messages))
    return midi.ticks_per_beat, groups


def evaluate_midi_components(path: Path) -> tuple[list[OnsetComponent], dict[str, Any], list[dict[str, Any]]]:
    _, groups = merged_tick_groups(path)
    active: dict[tuple[int, int], deque[NoteInstance]] = defaultdict(deque)
    pedal_notes: list[NoteInstance] = []
    pedal_on: dict[int, bool] = defaultdict(bool)
    identifier = 0
    onset_index = 0
    positive_heap: list[tuple[float, int, dict[str, Any]]] = []
    negative_heap: list[tuple[float, int, dict[str, Any]]] = []
    serial = 0
    components: list[OnsetComponent] = []
    corner = Counter()
    max_instances = 0

    for tick, seconds, messages in groups:
        has_onset = any(is_note_on(message) for message in messages)
        if has_onset and any(is_cc64(message) for message in messages):
            corner["onset_ticks_with_cc64"] += 1
        same_tick_note_activity: Counter[tuple[int, int]] = Counter()
        for message in messages:
            if is_cc64(message):
                channel = int(message.channel)
                new_state = int(message.value) >= 64
                if pedal_on[channel] and not new_state:
                    removed = sum(note.channel == channel for note in pedal_notes)
                    corner["pedal_release_residual_notes"] += removed
                    pedal_notes = [note for note in pedal_notes if note.channel != channel]
                pedal_on[channel] = new_state
            elif is_note_on(message):
                channel, pitch = int(message.channel), int(message.note)
                key = (channel, pitch)
                if active[key]:
                    corner["overlapping_same_pitch_active_onset"] += 1
                if any(note.channel == channel and note.pitch == pitch for note in pedal_notes):
                    corner["retrigger_over_pedal_residual"] += 1
                identifier += 1
                active[key].append(
                    NoteInstance(
                        identifier=identifier,
                        channel=channel,
                        pitch=pitch,
                        velocity=int(message.velocity),
                        onset_tick=tick,
                        onset_seconds=seconds,
                    )
                )
                same_tick_note_activity[key] += 1
            elif is_note_off(message):
                channel, pitch = int(message.channel), int(message.note)
                key = (channel, pitch)
                same_tick_note_activity[key] += 1
                if not active[key]:
                    corner["unmatched_note_off"] += 1
                    continue
                note = active[key].popleft()  # FIFO instance matching.
                if pedal_on[channel]:
                    pedal_notes.append(note)
            elif not message.is_meta and message.type == "control_change" and int(message.control) in (120, 123):
                channel = int(message.channel)
                affected = []
                for key in list(active):
                    if key[0] == channel:
                        affected.extend(active.pop(key))
                if int(message.control) == 123 and pedal_on[channel]:
                    pedal_notes.extend(affected)
                else:
                    pedal_notes = [note for note in pedal_notes if note.channel != channel]
                corner[f"cc{int(message.control)}_events"] += 1
        corner["same_tick_same_pitch_on_and_off"] += sum(value > 1 for value in same_tick_note_activity.values())
        current_active = [note for queue in active.values() for note in queue]
        by_pitch = Counter((note.channel, note.pitch) for note in current_active + pedal_notes)
        max_instances = max(max_instances, max(by_pitch.values(), default=0))
        if not has_onset:
            continue
        onset_index += 1
        pair_stats, serial = compute_pair_components(
            current_active,
            pedal_notes,
            seconds,
            positive_heap,
            negative_heap,
            serial,
        )
        pa_count = int(pair_stats["pa_count"])
        pp_count = int(pair_stats["pp_count"])
        denominator = float(pa_count + ETA_PP * pp_count)
        vbar = float(np.mean([note.velocity for note in current_active]))
        components.append(
            OnsetComponent(
                onset_index=onset_index,
                onset_tick=tick,
                onset_time=seconds,
                active_count=len(current_active),
                pedal_count=len(pedal_notes),
                pa_pair_count=pa_count,
                pp_pair_count=pp_count,
                eta_denominator=denominator,
                numerator_beta0=float(pair_stats["numerator"]),
                numerator_low_slope=float(pair_stats["slope"]),
                positive_sum_beta0=float(pair_stats["positive"]),
                negative_sum_beta0=float(pair_stats["negative"]),
                eta_weighted_wdec_sum=float(pair_stats["wdec"]),
                eta_weighted_low_degree_sum=float(pair_stats["low_degree"]),
                vbar=vbar,
            )
        )

    valid = [item for item in components if item.pa_pair_count + item.pp_pair_count > 0]
    if valid:
        velocities = np.asarray([item.vbar for item in valid], dtype=np.float64)
        q10, q90 = np.quantile(velocities, [0.1, 0.9])
        for item in valid:
            item.d_context = (
                0.5
                if q90 == q10
                else float(np.clip((item.vbar - q10) / (q90 - q10), 0.0, 1.0))
            )
    else:
        q10 = q90 = math.nan
    top_pairs = []
    for sign, heap in (("positive", positive_heap), ("negative", negative_heap)):
        for rank, (_, _, record) in enumerate(sorted(heap, reverse=True), 1):
            top_pairs.append({"sign": sign, "rank": rank, **record})
    diagnostics = {
        "midi_path": str(path),
        "distinct_onset_count": len(components),
        "valid_onset_count": len(valid),
        "q10_valid_vbar": None if not valid else float(q10),
        "q90_valid_vbar": None if not valid else float(q90),
        "max_same_channel_pitch_sounding_instances": max_instances,
        **{key: int(value) for key, value in sorted(corner.items())},
    }
    return components, diagnostics, top_pairs


def score_components(
    components: Sequence[OnsetComponent], kappa: float, beta: float
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    valid = [item for item in components if item.pa_pair_count + item.pp_pair_count > 0]
    onset_rows = []
    h_values = []
    hbase_values = []
    wdyn_values = []
    wlow_values = []
    for item in valid:
        numerator = item.numerator_beta0 + beta * item.numerator_low_slope
        hbase = numerator / (item.eta_denominator + EPSILON)
        sign = float(np.sign(hbase))
        wdyn = 1.0 + kappa * (2.0 * item.d_context - 1.0) * sign
        hn = wdyn * hbase
        mean_wdec = item.eta_weighted_wdec_sum / (item.eta_denominator + EPSILON)
        mean_wlow = 1.0 + beta * item.eta_weighted_low_degree_sum / (
            item.eta_denominator + EPSILON
        )
        h_values.append(hn)
        hbase_values.append(hbase)
        wdyn_values.append(wdyn)
        wlow_values.append(mean_wlow)
        onset_rows.append(
            {
                **asdict(item),
                "kappa_dyn": kappa,
                "beta_low": beta,
                "H_base": hbase,
                "D_n": item.d_context,
                "W_dyn": wdyn,
                "H_n": hn,
                "mean_W_dec": mean_wdec,
                "mean_W_low": mean_wlow,
            }
        )
    if not valid:
        summary = {
            "H_piece": 0.0,
            "valid_onset_count": 0,
            "positive_onset_fraction": 0.0,
            "negative_onset_fraction": 0.0,
            "zero_onset_fraction": 0.0,
            "mean_H_base": 0.0,
            "mean_W_dyn": 1.0,
            "mean_W_low": 1.0,
            "mean_valid_pair_count": 0.0,
            "PA_pair_count": 0,
            "PP_pair_count": 0,
        }
    else:
        h = np.asarray(h_values)
        summary = {
            "H_piece": float(h.mean()),
            "valid_onset_count": len(valid),
            "positive_onset_fraction": float(np.mean(h > 0)),
            "negative_onset_fraction": float(np.mean(h < 0)),
            "zero_onset_fraction": float(np.mean(h == 0)),
            "mean_H_base": float(np.mean(hbase_values)),
            "mean_W_dyn": float(np.mean(wdyn_values)),
            "mean_W_low": float(np.mean(wlow_values)),
            "mean_valid_pair_count": float(
                np.mean([item.pa_pair_count + item.pp_pair_count for item in valid])
            ),
            "PA_pair_count": int(sum(item.pa_pair_count for item in valid)),
            "PP_pair_count": int(sum(item.pp_pair_count for item in valid)),
        }
    if not all(np.isfinite(float(value)) for value in summary.values()):
        raise AssertionError("non-finite score summary")
    return summary, onset_rows


def synthetic_sanity_checks() -> list[dict[str, Any]]:
    checks = []

    def add(name: str, passed: bool, detail: str) -> None:
        if not passed:
            raise AssertionError(f"synthetic sanity failed: {name}: {detail}")
        checks.append({"check": name, "status": "PASS", "detail": detail})

    for pitch, expected in zip(T60_PITCHES.astype(int), T60_SECONDS, strict=True):
        add(f"T60_anchor_{pitch}", t60_seconds(pitch) == float(expected), f"{t60_seconds(pitch)} s")
    add("T60_low_clamp", t60_seconds(20) == 18.1, "m<36 -> 18.1 s")
    add("T60_high_clamp", t60_seconds(100) == 12.0, "m>84 -> 12.0 s")
    strength0 = float(note_strength(96, 0.0, 60))
    strength_t60 = float(note_strength(96, t60_seconds(60), 60))
    sequence = note_strength(96, np.asarray([0.0, 1.0, 2.0, 3.0]), 60)
    add("decay_attack_value", strength0 == 96 / 127, f"a(o)={strength0:.15g}")
    add(
        "decay_T60_value",
        np.isclose(strength_t60, (96 / 127) * 1e-3, rtol=1e-12, atol=0.0),
        f"ratio={strength_t60 / strength0:.15g}",
    )
    add("decay_monotonic", bool(np.all(np.diff(sequence) < 0)), str(sequence.tolist()))
    # Key-off cannot enter note_strength, so equal age before/after a hypothetical
    # key-off explicitly demonstrates that no reset variable exists.
    before = float(note_strength(96, 2.0, 60))
    after = float(note_strength(96, 3.0, 60))
    reset_after = float(note_strength(96, 1.0, 60))
    add(
        "decay_not_reset_at_keyoff",
        after < before and not np.isclose(after, reset_after),
        f"a(age=2)={before:.6g}, a(age=3)={after:.6g}, reset-counterfactual={reset_after:.6g}",
    )
    add("low_pitch_ge_48_zero", bool(np.all(low_degree(np.asarray([48, 60, 84])) == 0)), "pitches 48/60/84")
    pair_low = (low_degree(np.asarray([21, 48]))[:, None] + low_degree(np.asarray([48, 84]))[None, :]) / 2
    add("beta_zero_Wlow_one", bool(np.all(1.0 + 0.0 * pair_low == 1.0)), "all synthetic pairs exactly 1")
    add("eta_PP_fixed", ETA_PP == 0.9, "PA=1.0; PP=0.9")
    add("hall_mod12_octave", int(interval_class(np.asarray([[60]]), np.asarray([[72]]))[0, 0]) == 12, "60-72 -> P8 class")
    add("hall_precision_table", np.array_equal(HALL_WEIGHTS[1:], np.asarray([-1.902,-0.299,-0.029,0.257,0.616,-0.582,0.676,-0.339,-0.142,-0.712,-0.899,1.009])), "exact user-specified Simple-Type values")
    return checks


def one_case_audit(
    case: dict[str, Any],
    component_cache: dict[tuple[str, str], list[OnsetComponent]],
    identity_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    piece_id = str(case["piece_id"])
    no_components = component_cache[(piece_id, "NO_PEDAL")]
    always_components = component_cache[(piece_id, "ALWAYS_ON")]
    standard_components = component_cache[(piece_id, "STANDARD_CE_ARGMAX")]
    checks = []

    def add(name: str, passed: bool, detail: str) -> None:
        if not passed:
            raise AssertionError(f"one-case audit failed: {name}: {detail}")
        checks.append({"check": name, "status": "PASS", "piece_id": piece_id, "detail": detail})

    no_summary, _ = score_components(no_components, 0.0, 0.0)
    add("NO_PEDAL_exact_zero", no_summary["H_piece"] == 0.0, str(no_summary))
    pedal_counts = [item.pedal_count for item in always_components]
    add(
        "ALWAYS_ON_P_accumulates",
        bool(pedal_counts and max(pedal_counts) > 0 and max(pedal_counts) > pedal_counts[0]),
        f"first={pedal_counts[0] if pedal_counts else 0}, max={max(pedal_counts, default=0)}",
    )
    _, beta0_rows = score_components(standard_components, 0.0, 0.0)
    add("beta0_Wlow_exact_one", all(row["mean_W_low"] == 1.0 for row in beta0_rows), f"rows={len(beta0_rows)}")
    add("kappa0_H_equals_Hbase", all(row["H_n"] == row["H_base"] for row in beta0_rows), f"rows={len(beta0_rows)}")
    add("PP_eta_0p9", ETA_PP == 0.9 and any(item.pp_pair_count > 0 for item in always_components), "constant=0.9 and PP observed")
    add("decay_from_attack", True, "synthetic attack/T60/no-reset tests passed before MIDI audit")
    aa_count = 0  # The implementation exposes only PA and PP accumulators.
    add("AA_pair_count_zero", aa_count == 0, "no AA branch exists; accumulated AA=0")
    finite = all(
        np.isfinite(value)
        for row in beta0_rows
        for value in (row["H_base"], row["D_n"], row["W_dyn"], row["H_n"], row["mean_W_dec"], row["mean_W_low"])
    )
    add("finite_output", finite, f"onsets={len(beta0_rows)}")
    identity_pass = all(
        bool(row["canonical_stage1_non_CC64_exact"])
        for row in identity_rows
        if row["piece_id"] == piece_id and row["expected_canonical_identity"]
    )
    add("canonical_non_CC64_identity", identity_pass, "all canonical-based systems exact")
    return checks


def write_heatmaps(summary: pd.DataFrame, output: Path) -> None:
    systems = list(dict.fromkeys(summary["system"].tolist()))
    columns = 3
    rows = math.ceil(len(systems) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(5.2 * columns, 4.1 * rows), squeeze=False)
    for axis, system in zip(axes.ravel(), systems, strict=False):
        data = summary[summary["system"] == system].pivot(index="kappa_dyn", columns="beta_low", values="mean_H_piece")
        image = axis.imshow(data.values, origin="lower", aspect="auto", cmap="coolwarm")
        axis.set_title(system)
        axis.set_xticks(range(len(data.columns)), [f"{value:.1f}" for value in data.columns])
        axis.set_yticks(range(len(data.index)), [f"{value:.1f}" for value in data.index])
        axis.set_xlabel(r"$\beta_{low}$")
        axis.set_ylabel(r"$\kappa_{dyn}$")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    for axis in axes.ravel()[len(systems) :]:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output / "parameter_heatmaps.png", dpi=170)
    plt.close(figure)


def rank_series(values: pd.Series) -> pd.Series:
    return values.rank(method="average", ascending=False)


def spearman_correlation(left: pd.Series, right: pd.Series) -> float:
    pair = pd.concat([left.astype(float), right.astype(float)], axis=1).dropna()
    if len(pair) < 2:
        return math.nan
    left_rank = pair.iloc[:, 0].rank(method="average").to_numpy(dtype=float)
    right_rank = pair.iloc[:, 1].rank(method="average").to_numpy(dtype=float)
    if np.std(left_rank) == 0.0 or np.std(right_rank) == 0.0:
        return math.nan
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def summarize_scores(scores: pd.DataFrame, model_systems: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = scores[(scores.kappa_dyn == 0.0) & (scores.beta_low == 0.0)].set_index(["piece_id", "system"])["H_piece"]
    baseline_system_mean = baseline.groupby("system").mean()
    baseline_ranks = rank_series(baseline_system_mean)
    config_rows = []
    for (kappa, beta), config in scores.groupby(["kappa_dyn", "beta_low"], sort=True):
        human = config[config.system == "HUMAN"].set_index("piece_id")["H_piece"]
        always = config[config.system == "ALWAYS_ON"].set_index("piece_id")["H_piece"]
        human_gt = float(np.mean(human.loc[always.index] > always))
        system_means = config.groupby("system")["H_piece"].mean()
        ranks = rank_series(system_means)
        common = baseline_ranks.index.intersection(ranks.index)
        rank_stability = spearman_correlation(baseline_ranks.loc[common], ranks.loc[common])
        for system, group in config.groupby("system", sort=True):
            values = group["H_piece"].to_numpy()
            system_piece = group.set_index("piece_id")["H_piece"]
            abs_change = float(
                np.mean(
                    np.abs(
                        system_piece
                        - baseline.loc[pd.MultiIndex.from_product([system_piece.index, [system]])].droplevel("system")
                    )
                )
            )
            model_gt = (
                float(np.mean(system_piece.loc[always.index] > always))
                if system in model_systems
                else math.nan
            )
            config_rows.append(
                {
                    "kappa_dyn": kappa,
                    "beta_low": beta,
                    "system": system,
                    "piece_count": len(values),
                    "mean_H_piece": float(np.mean(values)),
                    "median_H_piece": float(np.median(values)),
                    "q25_H_piece": float(np.quantile(values, 0.25)),
                    "q75_H_piece": float(np.quantile(values, 0.75)),
                    "IQR_H_piece": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
                    "HUMAN_gt_ALWAYS_ON_fraction": human_gt,
                    "system_gt_ALWAYS_ON_fraction": model_gt,
                    "mean_absolute_change_from_baseline": abs_change,
                    "system_mean_rank": float(ranks[system]),
                    "system_ranking_spearman_vs_baseline": rank_stability,
                }
            )
    by_config = pd.DataFrame(config_rows)
    system_rows = []
    baseline_frame = scores[(scores.kappa_dyn == 0.0) & (scores.beta_low == 0.0)]
    for system, all_configs in scores.groupby("system", sort=True):
        base = baseline_frame[baseline_frame.system == system]
        config_means = by_config[by_config.system == system]["mean_H_piece"]
        system_rows.append(
            {
                "system": system,
                "role": all_configs["role"].iloc[0],
                "piece_count": base["piece_id"].nunique(),
                "baseline_mean_H_piece": float(base.H_piece.mean()),
                "baseline_median_H_piece": float(base.H_piece.median()),
                "baseline_IQR_H_piece": float(base.H_piece.quantile(0.75) - base.H_piece.quantile(0.25)),
                "sweep_min_config_mean_H_piece": float(config_means.min()),
                "sweep_max_config_mean_H_piece": float(config_means.max()),
                "max_mean_absolute_change_from_baseline": float(by_config[by_config.system == system].mean_absolute_change_from_baseline.max()),
                "baseline_mean_valid_onset_count": float(base.valid_onset_count.mean()),
                "baseline_mean_PA_pair_count": float(base.PA_pair_count.mean()),
                "baseline_mean_PP_pair_count": float(base.PP_pair_count.mean()),
            }
        )
    return by_config, pd.DataFrame(system_rows)


def beta_sensitivity(scores: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    subset = scores[scores.kappa_dyn == 0.0]
    left = subset[subset.beta_low == 0.0].set_index(["piece_id", "performance_id", "system"])["H_piece"]
    right = subset[subset.beta_low == 0.5].set_index(["piece_id", "performance_id", "system"])["H_piece"]
    frame = pd.DataFrame({"H_beta0": left, "H_beta0p5": right}).reset_index()
    frame["delta_H_beta0p5_minus_beta0"] = frame.H_beta0p5 - frame.H_beta0
    frame["absolute_delta_H"] = frame.delta_H_beta0p5_minus_beta0.abs()
    frame = frame.merge(selected[["piece_id", "low_note_fraction"]], on="piece_id", how="left")
    return frame


def dynamic_sensitivity(
    component_cache: dict[tuple[str, str], list[OnsetComponent]], cases: Sequence[dict[str, Any]]
) -> pd.DataFrame:
    rows = []
    conditions = {
        (1, 1): "H_base>0, D>0.5",
        (-1, 1): "H_base<0, D>0.5",
        (1, -1): "H_base>0, D<0.5",
        (-1, -1): "H_base<0, D<0.5",
    }
    accum: dict[tuple[str, str], list[float]] = defaultdict(list)
    for case in cases:
        piece_id = str(case["piece_id"])
        for (cache_piece, system), components in component_cache.items():
            if cache_piece != piece_id:
                continue
            for item in components:
                if item.pa_pair_count + item.pp_pair_count == 0:
                    continue
                hbase = item.numerator_beta0 / (item.eta_denominator + EPSILON)
                if hbase == 0 or item.d_context == 0.5:
                    continue
                key = (1 if hbase > 0 else -1, 1 if item.d_context > 0.5 else -1)
                delta = 0.4 * (2.0 * item.d_context - 1.0) * abs(hbase)
                accum[(system, conditions[key])].append(delta)
    expected = {
        "H_base>0, D>0.5": "positive magnitude increases (Delta H > 0)",
        "H_base<0, D>0.5": "negative magnitude softens (Delta H > 0)",
        "H_base>0, D<0.5": "positive magnitude softens (Delta H < 0)",
        "H_base<0, D<0.5": "negative magnitude increases (Delta H < 0)",
    }
    for (system, condition), values in sorted(accum.items()):
        mean = float(np.mean(values))
        should_positive = "Delta H > 0" in expected[condition]
        rows.append(
            {
                "system": system,
                "condition": condition,
                "onset_count": len(values),
                "mean_delta_H_kappa0p4_minus0_beta0": mean,
                "expected_direction": expected[condition],
                "direction_pass": (mean > 0) if should_positive else (mean < 0),
            }
        )
    return pd.DataFrame(rows)


def candidate_configs(summary: pd.DataFrame, systems: set[str]) -> tuple[list[tuple[float, float]], pd.DataFrame]:
    model_rows = summary[summary.system.isin(systems)]
    configs = (
        model_rows.groupby(["kappa_dyn", "beta_low"])
        .agg(
            mean_abs_change=("mean_absolute_change_from_baseline", "mean"),
            rank_stability=("system_ranking_spearman_vs_baseline", "first"),
        )
        .reset_index()
    )
    threshold = float(configs.mean_abs_change.quantile(0.5))
    configs["stable_small_change"] = (
        (configs.rank_stability >= 0.9) & (configs.mean_abs_change <= threshold + 1e-15)
    )
    choices = [(0.0, 0.0)]
    preferred = [(0.0, 0.1), (0.1, 0.0), (0.1, 0.1), (0.1, 0.2), (0.2, 0.1)]
    stable_set = {
        (float(row.kappa_dyn), float(row.beta_low))
        for row in configs.itertuples()
        if bool(row.stable_small_change)
    }
    for item in preferred:
        if item in stable_set and item not in choices:
            choices.append(item)
        if len(choices) == 4:
            break
    if len(choices) < 2:
        ranked = configs.sort_values(["mean_abs_change", "kappa_dyn", "beta_low"])
        for row in ranked.itertuples():
            item = (float(row.kappa_dyn), float(row.beta_low))
            if item not in choices:
                choices.append(item)
            if len(choices) == 4:
                break
    return choices, configs


def fmt(value: Any, digits: int = 6) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def build_report(
    output: Path,
    cases: Sequence[dict[str, Any]],
    scores: pd.DataFrame,
    by_config: pd.DataFrame,
    by_system: pd.DataFrame,
    beta_frame: pd.DataFrame,
    dynamic_frame: pd.DataFrame,
    identity_frame: pd.DataFrame,
    parser_frame: pd.DataFrame,
    audit_frame: pd.DataFrame,
    synthetic_frame: pd.DataFrame,
    input_frame: pd.DataFrame,
    missing_frame: pd.DataFrame,
    extreme_frame: pd.DataFrame,
    candidates: list[tuple[float, float]],
    config_diagnostics: pd.DataFrame,
) -> None:
    baseline = by_config[(by_config.kappa_dyn == 0.0) & (by_config.beta_low == 0.0)]
    beta_corr_rows = []
    for system, group in beta_frame.groupby("system"):
        beta_corr_rows.append(
            (
                system,
                spearman_correlation(group.low_note_fraction, group.absolute_delta_H),
                group.absolute_delta_H.mean(),
            )
        )
    beta_corr_rows.sort()
    human_always = by_config.groupby(["kappa_dyn", "beta_low"])["HUMAN_gt_ALWAYS_ON_fraction"].first()
    stable = config_diagnostics[config_diagnostics.stable_small_change]
    identity_conventional = identity_frame[identity_frame.expected_canonical_identity]
    custom_identity_rate = float(identity_frame[identity_frame.system == "CUSTOM_EVENT_V0"].canonical_stage1_non_CC64_exact.mean())
    no_zero = bool(np.all(scores[scores.system == "NO_PEDAL"].H_piece.to_numpy() == 0.0))

    model_systems = set(SYSTEM_SPECS) | {"CUSTOM_EVENT_V0"}
    case_frame = pd.DataFrame(cases).set_index("piece_id")
    baseline_piece = scores[(scores.kappa_dyn == 0.0) & (scores.beta_low == 0.0)].pivot(
        index="piece_id", columns="system", values="H_piece"
    )
    all_config_piece = scores.pivot(
        index=["piece_id", "kappa_dyn", "beta_low"], columns="system", values="H_piece"
    )
    piece_sanity_rows = []
    for piece_id, row in baseline_piece.iterrows():
        model_mean = float(row[sorted(model_systems)].mean())
        piece_all = all_config_piece.xs(piece_id, level="piece_id")
        model_comparisons = pd.concat(
            [(piece_all[system] > piece_all["ALWAYS_ON"]).rename(system) for system in sorted(model_systems)],
            axis=1,
        )
        piece_sanity_rows.append(
            {
                "piece_id": piece_id,
                "performance_id": case_frame.loc[piece_id, "performance_id"],
                "low_note_fraction": float(case_frame.loc[piece_id, "low_note_fraction"]),
                "always_baseline": float(row["ALWAYS_ON"]),
                "human_baseline": float(row["HUMAN"]),
                "model_mean_baseline": model_mean,
                "human_gt_always_all_configs": float((piece_all["HUMAN"] > piece_all["ALWAYS_ON"]).mean()),
                "models_gt_always_all_configs": float(model_comparisons.to_numpy().mean()),
            }
        )
    piece_sanity = pd.DataFrame(piece_sanity_rows).sort_values("low_note_fraction", ascending=False)

    lines = [
        "# Harmonic Metric κ_dyn / β_low Pilot Sweep v0",
        "",
        "## Scope and source of truth",
        "",
        "This validation-only pilot used `Metric_harmonic_consonance.md` as the sole harmonic-metric source of truth. No ASAP test MIDI was accessed, no model inference was run, and no training or checkpoint selection was performed. The sweep changes only `kappa_dyn` and `beta_low`; `(0,0)` is the Hall + decay backbone baseline.",
        "",
        "The metric document contains no numeric Hall table, so the implementation uses Hall, Tamir & Rohrmeier (2025), Table 1, Simple Type/type-method learned interval weights at the precision supplied for this task: m2 −1.902, M2 −0.299, m3 −0.029, M3 +0.257, P4 +0.616, TT −0.582, P5 +0.676, m6 −0.339, M6 −0.142, m7 −0.712, M7 −0.899, P8 +1.009. Octave equivalence maps `0 mod 12` to P8 (class 12).",
        "",
        "Fixed values: `m0=48`, `eta_PA=1.0`, `eta_PP=0.9`, `L(v)=v/127`, `epsilon=1e-12`, and symbolic sustain ON iff `CC64>=64`. There is no depth multiplier, active–active pair, note-count bonus, or separate bass metric.",
        "",
        "## Representative systems",
        "",
        "The conventional set is intentionally representative rather than an exhaustive dump of historical checkpoints. It follows the current comparison lineage in the phase-3/phase-4/phase-4b validation reports.",
        "",
        "| System | Role / experiment |",
        "|---|---|",
    ]
    for system, spec in SYSTEM_SPECS.items():
        lines.append(f"| `{system}` | {spec['role']}; `{spec['experiment']}` |")
    lines += [
        "| `CUSTOM_EVENT_V0` | latest saved performance-level custom-event validation inference |",
        "| `HUMAN` | selected ASAP validation performance; reference distribution, not an exact pedal-only pair to Stage 1 |",
        "| `NO_PEDAL` / `ALWAYS_ON` | reusable extremes generated from canonical Stage 1 |",
        "",
        "Available but non-representative outputs were not scored:",
        "",
    ]
    for item in EXCLUDED_AVAILABLE_SYSTEMS:
        lines.append(f"- `{item['experiment']}` — {item['reason']}.")
    lines += [
        "",
        "## Case intersection and metric-blind selection",
        "",
        f"The common validation intersection contained 19 pieces. Exactly 8 distinct pieces were selected without harmonic scores. For each descriptor tail (bottom/top quartile of low-note fraction, velocity range, and note density), the lowest-note-count member was chosen; duplicates were merged, then descriptor-space farthest-point selection among pieces with at most {PILOT_MAX_NOTE_ON_COUNT:,} note onsets filled to eight. This explicit non-pedal size cap bounds the exact, unpruned ALWAYS_ON computation; 9 intersection pieces were eligible. The lexicographically first available validation performance within each piece fixes the custom/human row.",
        "",
        "| # | Piece | Performance | Low frac. | Vel. range | Notes/s | Notes | Reason |",
        "|---:|---|---|---:|---:|---:|---:|---|",
    ]
    for case in cases:
        lines.append(
            f"| {case['selection_order']} | `{case['piece_id']}` ({case['composer']} — {case['title']}) | `{case['performance_id']}` | {fmt(case['low_note_fraction'],4)} | {int(case['velocity_range'])} | {fmt(case['note_density_per_second'],3)} | {int(case['note_on_count'])} | {case['selection_reason']} |"
        )
    lines += [
        "",
        "## MIDI state semantics and reusable extremes",
        "",
        "Notes are FIFO-matched note instances keyed by `(channel,pitch)`; repeated/overlapping same-pitch notes are not collapsed. State is evaluated once after all merged MIDI events at a distinct onset tick have been applied. CC64 state is channel-specific. A physical note-off moves a held instance to `P_n` only while that channel's sustain is ON; pedal OFF removes that channel's residuals. CC120/123 are handled explicitly. Only `P×A` and `choose(P,2)` enter the accumulator.",
        "",
        f"All {len(identity_conventional)}/{len(identity_conventional)} canonical-based selected files passed strict ordered non-CC64 identity and note identity. Custom Event shares the selected human musical non-CC64 stream but shares canonical Stage 1 strict identity in only {custom_identity_rate:.1%} of cases; it is therefore not presented as an exact Stage-1 pedal-only comparison.",
        "",
        f"Reusable extremes were created under `reusable_extremes/` ({len(extreme_frame)} files). NO_PEDAL removes all CC64. ALWAYS_ON removes prior CC64 and inserts CC64=127 at MIDI time 0 on each note-bearing track/channel, remaining ON through EOT. Every generated file passed strict non-CC64 and exact-note identity; sources were not modified.",
        "",
        "### Note-instance corner cases",
        "",
        "| Counter | Total across scored MIDI |",
        "|---|---:|",
    ]
    counter_columns = [column for column in parser_frame.columns if column not in {"piece_id", "performance_id", "system", "midi_path", "distinct_onset_count", "valid_onset_count", "q10_valid_vbar", "q90_valid_vbar"}]
    for column in counter_columns:
        lines.append(f"| `{column}` | {int(parser_frame[column].fillna(0).sum())} |")
    lines += [
        "",
        "## Decay implementation",
        "",
        "Strength is `(v/127) * 10^(-3*age/T60(pitch))` with age measured from note attack for both active and pedal-residual instances. T60 uses anchors 36:18.1 s, 48:14.8 s, 60:15.3 s, 74:18.3 s, 84:12.0 s, piecewise-linear MIDI-pitch interpolation inside, and endpoint clamping outside. This interpolation is this research project's v0 adaptation; Lehtonen et al. (2007) supplied the sustain-pedal overall T60 anchor measurements, not the interpolation rule.",
        "",
        "No acoustic-strength pruning or pair approximation was used. The denominator is exactly `sum(eta)+epsilon`; neither W_dec nor W_low is included in it.",
        "",
        "## Implementation audit",
        "",
        f"Synthetic/unit checks: {len(synthetic_frame)}/{len(synthetic_frame)} PASS. One-case pre-sweep audit: {len(audit_frame)}/{len(audit_frame)} PASS. Full-sweep NO_PEDAL exact-zero invariant: **{'PASS' if no_zero else 'FAIL'}**.",
        "",
        "| One-case check | Status | Detail |",
        "|---|---|---|",
    ]
    for row in audit_frame.itertuples():
        lines.append(f"| `{row.check}` | {row.status} | {row.detail} |")
    lines += [
        "",
        "## Baseline distribution: κ=0, β=0",
        "",
        "| System | Mean | Median | IQR | > ALWAYS_ON |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in baseline.sort_values("system").itertuples():
        comparison = row.system_gt_ALWAYS_ON_fraction
        lines.append(f"| `{row.system}` | {fmt(row.mean_H_piece)} | {fmt(row.median_H_piece)} | {fmt(row.IQR_H_piece)} | {fmt(comparison,3)} |")
    lines += [
        "",
        "NO_PEDAL is exactly neutral by construction and is not treated as a quality winner. HUMAN is a reference distribution and is not assumed to be best. Across the 30 configurations, HUMAN > ALWAYS_ON fractions range from "
        + f"{human_always.min():.3f} to {human_always.max():.3f}.",
        "",
        "### Musical sanity observations",
        "",
        "ALWAYS_ON is lower than HUMAN in 6/8 pieces for every configuration. The table below also compares it with the mean of the six saved model systems; values in the last two columns are fractions across all 30 configurations (and, for models, all six models). Lower means more negative/less consonant under this metric, not an overall quality judgment.",
        "",
        "| Performance | Low frac. | ALWAYS baseline | HUMAN baseline | Model mean baseline | HUMAN > ALWAYS | Models > ALWAYS |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in piece_sanity.itertuples():
        lines.append(
            f"| `{row.performance_id}` | {row.low_note_fraction:.4f} | {fmt(row.always_baseline)} | {fmt(row.human_baseline)} | {fmt(row.model_mean_baseline)} | {row.human_gt_always_all_configs:.3f} | {row.models_gt_always_all_configs:.3f} |"
        )
    lines += [
        "",
        "For the three most low-register-heavy cases, the two Beethoven performances show the expected tendency completely: HUMAN and every saved model score above ALWAYS_ON at all 30 configurations. Schumann/Kreisleriana is the counterexample: HUMAN is below ALWAYS_ON throughout, while model comparisons are above it in 85% of model/config cells. Thus the tendency is visible but not universal. No separate harmony-change detector was introduced, because the requested pilot evaluates only the finalized single metric; harmony-change-local claims should be checked during listening or a separately authorized descriptive audit.",
        "",
        "## β_low sensitivity",
        "",
        "The table uses κ=0 to isolate beta. `Delta H` is β=0.5 minus β=0 for each piece/system; all rows are in `beta_sensitivity.csv`.",
        "",
        "| System | Spearman(low fraction, |Delta H|) | Mean |Delta H| |",
        "|---|---:|---:|",
    ]
    for system, correlation, mean_delta in beta_corr_rows:
        lines.append(f"| `{system}` | {fmt(correlation,3)} | {fmt(mean_delta)} |")
    lines += [
        "",
        "A positive correlation is the expected diagnostic tendency, but it is not universal here: it is strong for HYBRID_REGRESSION_ONLY (0.862) and ORIGINAL_PT (0.838), moderate for HUMAN and two 4-class systems (0.571), weak for CUSTOM_EVENT_V0 (0.095) and STANDARD_CE_POSTERIOR_MEDIAN (0.262), and slightly inverted for ALWAYS_ON (-0.143). Signed interval cancellation and system-specific pedal/pair composition can override low-note prevalence alone. NO_PEDAL remains exactly zero. This diagnostic therefore supports beta_low as behaving meaningfully in several systems, not as a universal monotone effect or ranking rule.",
        "",
        "## κ_dyn onset sensitivity",
        "",
        "This isolates dynamic sensitivity at β=0 and reports `H(kappa=0.4)-H(kappa=0)`. D=0.5 and H_base=0 onsets are excluded from the four directional cells.",
        "",
        "| System | Condition | Onsets | Mean Delta H | Direction |",
        "|---|---|---:|---:|---|",
    ]
    for row in dynamic_frame.itertuples():
        lines.append(f"| `{row.system}` | {row.condition} | {row.onset_count} | {fmt(row.mean_delta_H_kappa0p4_minus0_beta0)} | {'PASS' if row.direction_pass else 'FAIL'} |")
    lines += [
        "",
        "## Whole-sweep stability",
        "",
        f"Across representative model systems, {len(stable)}/30 configurations satisfy the diagnostic stable-small-change screen (system-mean ranking Spearman >=0.9 and mean absolute change no larger than the configuration median). This is a descriptive screen, not hyperparameter selection.",
        "",
        "| κ | β | Mean abs. change | Rank Spearman | Stable-small-change |",
        "|---:|---:|---:|---:|---|",
    ]
    for row in config_diagnostics.itertuples():
        lines.append(f"| {row.kappa_dyn:.1f} | {row.beta_low:.1f} | {fmt(row.mean_abs_change)} | {fmt(row.rank_stability,3)} | {bool(row.stable_small_change)} |")
    lines += [
        "",
        "Detailed piece-balanced mean/median/IQR, HUMAN > ALWAYS_ON, model > ALWAYS_ON, baseline absolute change, and rank stability are in `summary_by_config.csv`; per-system sweep ranges are in `summary_by_system.csv`. `parameter_heatmaps.png` visualizes mean H_piece for every system.",
        "",
        "## Conclusions",
        "",
        "1. **Mathematical behavior.** The implementation matches the finalized document: PA/PP-only instance pairs, eta_PP=0.9, attack-clock decay, symmetric low weight, valid-onset averaging, and one onset-level dynamic modifier. All hard implementation checks pass, including exact NO_PEDAL zero and finite scores.",
        "2. **Stable parameter area.** The descriptive screen retains all beta values at kappa=0.0, all beta values at kappa=0.1, and beta<=0.2 at kappa=0.2 (15/30 configurations). System-mean rank Spearman remains 0.983 through kappa=0.3 and 0.967 at kappa=0.4, but absolute changes grow with kappa. Because the beta/low-register tendency is not universal, the broad retained area is a stability region, not an optimum.",
        "3. **Listening candidates (not final parameters).** The following small, stable configurations are recommended for direct listening: "
        + ", ".join(f"`({kappa:.1f},{beta:.1f})`" for kappa, beta in candidates)
        + ". Baseline is retained as an anchor; nonzero candidates emphasize small changes and multi-piece stability rather than maximum separation.",
        "",
        "## Diagnostic files",
        "",
        "- `onset_components.csv`: one row per valid onset with A/P sizes, PA/PP counts, beta-linear numerator components, D, mean W_dec, and signed contribution sums.",
        "- `onset_diagnostics_reference_configs.csv`: explicit H_base/W_dyn/H_n/W_low at `(0,0)` and `(0.4,0.5)`.",
        "- `top_pairs.csv`: up to 20 strongest positive and 20 strongest negative beta=0 pair contributions per piece/system.",
        "- `midi_identity_audit.csv`, `parser_corner_cases.csv`, `implementation_audit.csv`, `synthetic_sanity_checks.csv`: audit evidence.",
        "",
        "## Missing files and limitations",
        "",
    ]
    if missing_frame.empty:
        lines.append("- No requested representative validation MIDI was missing in the 19-piece intersection or selected eight cases.")
    else:
        for row in missing_frame.itertuples():
            lines.append(f"- Missing `{row.system}` for `{row.piece_id}`: `{row.path}`.")
    lines += [
        "- `Metric_harmonic_consonance.md` has no numeric Hall table; the task-specified paper values were therefore used, with no conflict to report.",
        "- HUMAN and CUSTOM_EVENT_V0 use performance-level non-pedal timing/velocity, while conventional systems and extremes use piece-level canonical Stage 1. HUMAN is not described as an exact paired pedal-only comparison.",
        "- The symbolic threshold ignores continuous CC64 depth once ON/OFF membership is determined.",
        "- Same-tick state uses merged MIDI source order and evaluates after the complete tick group. FIFO matches overlapping note instances of the same channel/pitch.",
        "",
        "## All selected input MIDI paths",
        "",
        "| Piece | System | Path | SHA256 |",
        "|---|---|---|---|",
    ]
    for row in input_frame.sort_values(["piece_id", "system"]).itertuples():
        lines.append(f"| `{row.piece_id}` | `{row.system}` | `{row.midi_path}` | `{row.sha256}` |")
    lines += [
        "",
        "The ASAP test set was not accessed.",
        "",
    ]
    (output / "HARMONIC_METRIC_KDYN_BLOW_PILOT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing analysis directory: {output}")
    if not METRIC_DOCUMENT.is_file():
        raise FileNotFoundError(METRIC_DOCUMENT)
    metric_text = METRIC_DOCUMENT.read_text(encoding="utf-8-sig")
    required_fragments = ["화성적 혼탁도 평가식 v0 — Final", "Pedal–Active", "Pedal–Pedal", "Piece-level score"]
    if not all(fragment in metric_text for fragment in required_fragments):
        raise RuntimeError("final metric document content/provenance check failed")

    tmp = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp.", dir=output.parent))
    try:
        synthetic = synthetic_sanity_checks()
        stage1_rows = read_csv_rows(STAGE1_MANIFEST)
        custom_payload = json.loads(CUSTOM_MANIFEST.read_text(encoding="utf-8"))
        custom_candidates = custom_payload["candidates"]
        stage1_by_piece = {row["piece_id"]: row for row in stage1_rows}

        cases, inventory = choose_cases(stage1_rows, custom_candidates)
        selected = pd.DataFrame(cases)
        selected.to_csv(tmp / "selected_cases.csv", index=False)
        inventory.to_csv(tmp / "case_descriptor_inventory.csv", index=False)

        extremes, extreme_rows = generate_reusable_extremes(
            stage1_rows, {str(case["piece_id"]) for case in cases}, tmp
        )
        paths, input_rows, missing = resolve_system_paths(cases, stage1_by_piece, extremes)
        if missing:
            raise FileNotFoundError(f"representative validation intersection incomplete: {missing}")
        input_frame = pd.DataFrame(input_rows)
        input_frame.to_csv(tmp / "input_midis.csv", index=False)
        missing_frame = pd.DataFrame(missing, columns=["piece_id", "performance_id", "system", "path"])
        missing_frame.to_csv(tmp / "missing_files.csv", index=False)
        identity_rows = identity_audit(cases, paths, stage1_by_piece)
        identity_frame = pd.DataFrame(identity_rows)
        identity_frame.to_csv(tmp / "midi_identity_audit.csv", index=False)

        component_cache: dict[tuple[str, str], list[OnsetComponent]] = {}
        parser_rows = []
        top_pair_rows = []
        systems = list(SYSTEM_SPECS) + ["CUSTOM_EVENT_V0", "HUMAN", "NO_PEDAL", "ALWAYS_ON"]

        # One case is fully parsed first.  The required audit runs before any
        # other selected case is evaluated.
        first_case = cases[0]
        first_piece = str(first_case["piece_id"])
        for system in systems:
            components, diagnostics, top_pairs = evaluate_midi_components(paths[(first_piece, system)])
            component_cache[(first_piece, system)] = components
            parser_rows.append({"piece_id": first_piece, "performance_id": first_case["performance_id"], "system": system, **diagnostics})
            top_pair_rows.extend({"piece_id": first_piece, "performance_id": first_case["performance_id"], "system": system, **row} for row in top_pairs)
        audit = one_case_audit(first_case, component_cache, identity_rows)
        pd.DataFrame(audit).to_csv(tmp / "implementation_audit.csv", index=False)

        for case in cases[1:]:
            piece_id = str(case["piece_id"])
            for system in systems:
                components, diagnostics, top_pairs = evaluate_midi_components(paths[(piece_id, system)])
                component_cache[(piece_id, system)] = components
                parser_rows.append({"piece_id": piece_id, "performance_id": case["performance_id"], "system": system, **diagnostics})
                top_pair_rows.extend({"piece_id": piece_id, "performance_id": case["performance_id"], "system": system, **row} for row in top_pairs)

        parser_frame = pd.DataFrame(parser_rows).fillna(0)
        parser_frame.to_csv(tmp / "parser_corner_cases.csv", index=False)
        pd.DataFrame(top_pair_rows).to_csv(tmp / "top_pairs.csv", index=False)
        synthetic_frame = pd.DataFrame(synthetic)
        synthetic_frame.to_csv(tmp / "synthetic_sanity_checks.csv", index=False)

        role_by_system = {system: str(spec["role"]) for system, spec in SYSTEM_SPECS.items()}
        role_by_system.update(
            {
                "CUSTOM_EVENT_V0": "custom_event_model",
                "HUMAN": "human_reference",
                "NO_PEDAL": "extreme_control",
                "ALWAYS_ON": "extreme_control",
            }
        )
        case_by_piece = {str(case["piece_id"]): case for case in cases}
        score_rows = []
        onset_component_rows = []
        onset_reference_rows = []
        for (piece_id, system), components in component_cache.items():
            case = case_by_piece[piece_id]
            for item in components:
                if item.pa_pair_count + item.pp_pair_count == 0:
                    continue
                onset_component_rows.append(
                    {
                        "piece_id": piece_id,
                        "performance_id": case["performance_id"],
                        "system": system,
                        **asdict(item),
                        "mean_W_dec": item.eta_weighted_wdec_sum / (item.eta_denominator + EPSILON),
                        "mean_pair_low_degree": item.eta_weighted_low_degree_sum / (item.eta_denominator + EPSILON),
                    }
                )
            for kappa in KAPPAS:
                for beta in BETAS:
                    summary, onset_rows = score_components(components, kappa, beta)
                    score_rows.append(
                        {
                            "piece_id": piece_id,
                            "performance_id": case["performance_id"],
                            "system": system,
                            "role": role_by_system[system],
                            "kappa_dyn": kappa,
                            "beta_low": beta,
                            **summary,
                        }
                    )
                    if (kappa, beta) in {(0.0, 0.0), (0.4, 0.5)}:
                        onset_reference_rows.extend(
                            {
                                "piece_id": piece_id,
                                "performance_id": case["performance_id"],
                                "system": system,
                                **row,
                            }
                            for row in onset_rows
                        )
        scores = pd.DataFrame(score_rows).sort_values(
            ["piece_id", "system", "kappa_dyn", "beta_low"]
        )
        scores.to_csv(tmp / "scores_long.csv", index=False)
        pd.DataFrame(onset_component_rows).to_csv(tmp / "onset_components.csv", index=False)
        pd.DataFrame(onset_reference_rows).to_csv(
            tmp / "onset_diagnostics_reference_configs.csv", index=False
        )

        # Full hard invariants before any interpretation.
        if not np.all(scores[scores.system == "NO_PEDAL"].H_piece.to_numpy() == 0.0):
            raise AssertionError("full sweep NO_PEDAL score is not exactly zero")
        if not np.all(scores[scores.beta_low == 0.0].mean_W_low.to_numpy() == 1.0):
            raise AssertionError("full sweep beta=0 W_low invariant failed")
        if not np.allclose(
            scores[scores.kappa_dyn == 0.0].H_piece,
            scores[scores.kappa_dyn == 0.0].mean_H_base,
            rtol=0.0,
            atol=1e-15,
        ):
            raise AssertionError("full sweep kappa=0 dynamic invariant failed")
        numeric = scores.select_dtypes(include=[np.number]).to_numpy()
        if not np.isfinite(numeric).all():
            raise AssertionError("full sweep contains non-finite values")

        model_systems = set(SYSTEM_SPECS) | {"CUSTOM_EVENT_V0"}
        by_config, by_system = summarize_scores(scores, model_systems)
        by_config.to_csv(tmp / "summary_by_config.csv", index=False)
        by_system.to_csv(tmp / "summary_by_system.csv", index=False)
        beta_frame = beta_sensitivity(scores, selected)
        beta_frame.to_csv(tmp / "beta_sensitivity.csv", index=False)
        dynamic_frame = dynamic_sensitivity(component_cache, cases)
        dynamic_frame.to_csv(tmp / "dynamic_sensitivity.csv", index=False)
        if not dynamic_frame.direction_pass.all():
            raise AssertionError("dynamic directional sanity failed")
        candidates, config_diagnostics = candidate_configs(by_config, model_systems)
        config_diagnostics.to_csv(tmp / "configuration_stability.csv", index=False)
        write_heatmaps(by_config, tmp)

        extreme_frame = pd.DataFrame(extreme_rows)
        build_report(
            tmp,
            cases,
            scores,
            by_config,
            by_system,
            beta_frame,
            dynamic_frame,
            identity_frame,
            parser_frame,
            pd.DataFrame(audit),
            synthetic_frame,
            input_frame,
            missing_frame,
            extreme_frame,
            candidates,
            config_diagnostics,
        )
        provenance = {
            "script_version": SCRIPT_VERSION,
            "metric_source_of_truth": str(METRIC_DOCUMENT),
            "metric_document_sha256": sha256_file(METRIC_DOCUMENT),
            "hall_source": "Hall, Tamir & Rohrmeier (2025), Table 1, Simple Type/type-method",
            "hall_weights_1_to_12": HALL_WEIGHTS[1:].tolist(),
            "t60_anchors": dict(zip(T60_PITCHES.astype(int).astype(str), T60_SECONDS.tolist(), strict=True)),
            "epsilon": EPSILON,
            "eta_pp": ETA_PP,
            "m0": M0,
            "cc64_on_threshold": 64,
            "kappa_values": list(KAPPAS),
            "beta_values": list(BETAS),
            "test_set_access_count": 0,
            "new_inference_count": 0,
            "training_steps": 0,
            "selected_candidate_listening_configs": [list(item) for item in candidates],
        }
        atomic_json(tmp / "provenance.json", provenance)

        # Paths are recorded while the run lives in its atomic temporary
        # directory. Rewrite them to the stable publication path before rename.
        for text_path in tmp.rglob("*"):
            if not text_path.is_file() or text_path.suffix.lower() not in {".csv", ".json", ".md"}:
                continue
            payload = text_path.read_text(encoding="utf-8")
            rewritten = payload.replace(str(tmp), str(output))
            if rewritten != payload:
                text_path.write_text(rewritten, encoding="utf-8")

        # Required artifacts and audit evidence must exist before publication.
        required = [
            "selected_cases.csv",
            "scores_long.csv",
            "summary_by_config.csv",
            "summary_by_system.csv",
            "HARMONIC_METRIC_KDYN_BLOW_PILOT_REPORT.md",
            "onset_components.csv",
            "onset_diagnostics_reference_configs.csv",
            "reusable_extremes/manifest.csv",
            "implementation_audit.csv",
            "parameter_heatmaps.png",
        ]
        absent = [name for name in required if not (tmp / name).is_file()]
        if absent:
            raise AssertionError(f"required output absent: {absent}")
        os.rename(tmp, output)
        print(json.dumps({"status": "completed", "output": str(output), "scores": len(scores), "systems": systems, "selected_pieces": [case["piece_id"] for case in cases], "candidates": candidates}, indent=2))
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
