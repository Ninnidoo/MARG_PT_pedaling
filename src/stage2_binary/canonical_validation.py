"""Canonical validation-only Stage-2 evaluation on frozen Original-PT MIDI."""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from miditoolkit import MidiFile

from scripts.audit_stage2_binary_metric_fidelity import _strict_official_similarity

from .canonical_stage1 import (
    assert_strict_non_cc64_equality,
    sha256_file,
    signature_sha256,
    strict_non_cc64_signature,
    transplant_cc64_only,
)
from .strict_midi_validation import PianoT5GemmaConfig, ids_to_midi, map_midi, midi_to_ids
from .validation_evaluator import infer_cached_binary_pedals, joint16_histogram_from_ids


EXPECTED_PIECES = 19
EXPECTED_CANONICAL_NOTES = 48_893
EXPECTED_HUMAN_NOTES = 283_928
EXPECTED_WINDOWS = 182


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def read_canonical_manifest(path: str | Path) -> list[dict[str, str]]:
    """Read the frozen validation manifest and reject test/non-canonical paths."""

    manifest_path = Path(path)
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != EXPECTED_PIECES or len({row["piece_id"] for row in rows}) != EXPECTED_PIECES:
        raise RuntimeError("canonical validation requires exactly 19 unique pieces")
    required = {
        "piece_id",
        "canonical_midi_path",
        "canonical_midi_sha256",
        "canonical_non_cc64_signature_sha256",
        "generated_ids_path",
        "generated_token_sha256_int64_le",
        "status",
    }
    if not required.issubset(rows[0]):
        raise RuntimeError(f"canonical manifest is missing fields: {sorted(required - set(rows[0]))}")
    for row in rows:
        midi_path = row["canonical_midi_path"]
        ids_path = row["generated_ids_path"]
        if "stage2_binary_canonical_v1/canonical_validation_stage1/" not in midi_path:
            raise PermissionError(f"non-canonical validation MIDI path: {midi_path}")
        if "stage2_binary_canonical_v1/canonical_validation_stage1/" not in ids_path:
            raise PermissionError(f"non-canonical generated-ID path: {ids_path}")
        if "/outputs/midi/" in midi_path or "/test" in midi_path.lower():
            raise PermissionError(f"ASAP test path is forbidden: {midi_path}")
        if row["status"] != "frozen_pass":
            raise RuntimeError(f"canonical piece is not frozen_pass: {row['piece_id']}")
    return rows


def verify_canonical_bank(
    manifest_path: str | Path,
    baseline_path: str | Path,
    *,
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Verify frozen hashes and reproduce the pinned baseline from MIDI files."""

    rows = read_canonical_manifest(manifest_path)
    baseline = _read_json(baseline_path)
    if not baseline.get("completed") or int(baseline.get("validation_piece_count", -1)) != EXPECTED_PIECES:
        raise RuntimeError("canonical baseline is not complete for 19 pieces")
    human = np.asarray(baseline["human_histogram"], dtype=np.int64)
    expected_candidate = np.asarray(baseline["canonical_original_pt_histogram"], dtype=np.int64)
    if human.shape != (16,) or int(human.sum()) != EXPECTED_HUMAN_NOTES:
        raise RuntimeError("canonical human histogram provenance mismatch")
    if expected_candidate.shape != (16,) or int(expected_candidate.sum()) != EXPECTED_CANONICAL_NOTES:
        raise RuntimeError("canonical Original-PT histogram provenance mismatch")

    tokenizer_config = PianoT5GemmaConfig()
    reproduced = np.zeros(16, dtype=np.int64)
    generated_id_notes = 0
    canonical_hashes: dict[str, str] = {}
    for row in rows:
        midi_path = Path(row["canonical_midi_path"])
        ids_path = Path(row["generated_ids_path"])
        if not midi_path.is_file() or not ids_path.is_file():
            raise FileNotFoundError(midi_path if not midi_path.is_file() else ids_path)
        midi_hash = sha256_file(midi_path)
        if midi_hash != row["canonical_midi_sha256"]:
            raise RuntimeError(f"canonical MIDI hash mismatch: {midi_path}")
        if signature_sha256(midi_path) != row["canonical_non_cc64_signature_sha256"]:
            raise RuntimeError(f"canonical non-CC64 signature mismatch: {midi_path}")
        generated = np.load(ids_path, allow_pickle=False)
        generated_hash = hashlib.sha256(
            np.asarray(generated, dtype="<i8").tobytes()
        ).hexdigest()
        if generated_hash != row["generated_token_sha256_int64_le"]:
            raise RuntimeError(f"canonical generated-ID hash mismatch: {ids_path}")
        generated_id_notes += int(np.asarray(generated).size // 8)
        canonical_ids = midi_to_ids(tokenizer_config, MidiFile(str(midi_path)))
        reproduced += joint16_histogram_from_ids(canonical_ids)
        canonical_hashes[str(midi_path)] = midi_hash
    if generated_id_notes != 48_894:
        raise RuntimeError(f"pre-render generated-ID note count mismatch: {generated_id_notes}")
    if not np.array_equal(reproduced, expected_candidate):
        raise RuntimeError("canonical MIDI histogram differs from frozen baseline")
    metric = _strict_official_similarity(human, reproduced)
    baseline_metric = baseline["metrics"]
    if abs(metric["js_distance_base2"] - float(baseline_metric["js_distance_base2"])) > tolerance:
        raise RuntimeError("canonical baseline JS did not reproduce")
    if abs(
        metric["histogram_intersection"]
        - float(baseline_metric["histogram_intersection_official"])
    ) > tolerance:
        raise RuntimeError("canonical baseline Intersection did not reproduce")
    return {
        "pieces": len(rows),
        "canonical_notes": int(reproduced.sum()),
        "generated_id_notes": generated_id_notes,
        "human_notes": int(human.sum()),
        "canonical_histogram": reproduced.tolist(),
        "human_histogram": human.tolist(),
        "js_distance": float(metric["js_distance_base2"]),
        "intersection": float(metric["histogram_intersection"]),
        "canonical_hashes": canonical_hashes,
        "asap_test_midi_access_count": 0,
        "stage1_inference_regeneration": 0,
        "passed": True,
    }


def _first_non_cc64_difference(canonical_path: Path, candidate_path: Path) -> dict[str, Any]:
    canonical = strict_non_cc64_signature(canonical_path)
    candidate = strict_non_cc64_signature(candidate_path)
    summary: dict[str, Any] = {
        "midi_type_equal": canonical["midi_type"] == candidate["midi_type"],
        "ticks_per_beat_equal": canonical["ticks_per_beat"] == candidate["ticks_per_beat"],
        "canonical_track_count": len(canonical["tracks"]),
        "candidate_track_count": len(candidate["tracks"]),
    }
    for track_index, (left, right) in enumerate(
        zip(canonical["tracks"], candidate["tracks"])
    ):
        if left != right:
            summary.update(
                first_different_track=track_index,
                canonical_event_count=len(left),
                candidate_event_count=len(right),
            )
            for event_index, (left_event, right_event) in enumerate(zip(left, right)):
                if left_event != right_event:
                    summary.update(
                        first_different_event=event_index,
                        canonical_event=left_event,
                        candidate_event=right_event,
                    )
                    break
            break
    return summary


def evaluate_canonical_stage1(
    model: torch.nn.Module,
    *,
    architecture: str,
    manifest_path: str | Path,
    baseline_path: str | Path,
    device: torch.device,
    temporary_root: str | Path,
    event_logger: Callable[[str], None] | None = None,
    inference_function: Callable[..., tuple[np.ndarray, dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Run deterministic Stage 2 and strict CC64-only MIDI validation on 19 pieces."""

    rows = read_canonical_manifest(manifest_path)
    baseline = _read_json(baseline_path)
    human = np.asarray(baseline["human_histogram"], dtype=np.int64)
    expected_baseline = np.asarray(
        baseline["canonical_original_pt_histogram"], dtype=np.int64
    )
    if int(human.sum()) != EXPECTED_HUMAN_NOTES or int(expected_baseline.sum()) != EXPECTED_CANONICAL_NOTES:
        raise RuntimeError("canonical baseline inventory changed")

    root = Path(temporary_root)
    root.mkdir(parents=True, exist_ok=True)
    tokenizer_config = PianoT5GemmaConfig()
    candidate_histogram = np.zeros(16, dtype=np.int64)
    original_histogram = np.zeros(16, dtype=np.int64)
    equality_rows: list[dict[str, Any]] = []
    windows = 0
    input_notes = 0
    canonical_hashes_before = {
        row["canonical_midi_path"]: sha256_file(row["canonical_midi_path"])
        for row in rows
    }
    for row in rows:
        if canonical_hashes_before[row["canonical_midi_path"]] != row["canonical_midi_sha256"]:
            raise RuntimeError(f"canonical MIDI hash mismatch before epoch metric: {row['piece_id']}")

    with tempfile.TemporaryDirectory(prefix="canonical_stage2_", dir=root) as temporary:
        temporary_path = Path(temporary)
        for index, row in enumerate(rows, start=1):
            piece_id = row["piece_id"]
            canonical_path = Path(row["canonical_midi_path"])
            canonical_ids = np.asarray(
                midi_to_ids(tokenizer_config, MidiFile(str(canonical_path))),
                dtype=np.int64,
            )
            original_histogram += joint16_histogram_from_ids(canonical_ids)
            if inference_function is None:
                predicted_ids, details = infer_cached_binary_pedals(
                    model,
                    canonical_ids,
                    architecture=architecture,
                    device=device,
                    window_notes=512,
                    stride_notes=256,
                )
            else:
                predicted_ids, details = inference_function(
                    model,
                    canonical_ids,
                    device=device,
                    window_notes=512,
                    stride_notes=256,
                )
            if not details["non_pedal_tokens_preserved"]:
                raise AssertionError(f"pre-render non-pedal token mismatch: {piece_id}")
            donor_path = temporary_path / f"{piece_id}.donor.mid"
            candidate_path = temporary_path / f"{piece_id}.candidate.mid"
            performance = ids_to_midi(
                tokenizer_config,
                predicted_ids,
                ref=canonical_ids.tolist(),
            )
            donor = map_midi(MidiFile(str(canonical_path)), performance)
            donor.dump(str(donor_path))
            try:
                transplant = transplant_cc64_only(
                    canonical_path,
                    donor_path,
                    candidate_path,
                )
                strict = assert_strict_non_cc64_equality(
                    canonical_path,
                    candidate_path,
                )
            except BaseException as error:
                difference = (
                    _first_non_cc64_difference(canonical_path, candidate_path)
                    if candidate_path.is_file()
                    else {"candidate_created": False}
                )
                raise AssertionError(
                    "canonical non-CC64 gate failed: "
                    + json.dumps(
                        {"piece_id": piece_id, "error": repr(error), "diff": difference},
                        sort_keys=True,
                    )
                ) from error
            candidate_ids = midi_to_ids(
                tokenizer_config,
                MidiFile(str(candidate_path)),
            )
            candidate_histogram += joint16_histogram_from_ids(candidate_ids)
            input_notes += int(details["notes"])
            windows += int(details["windows"])
            equality_rows.append(
                {
                    "piece_id": piece_id,
                    "canonical_sha256": row["canonical_midi_sha256"],
                    "candidate_sha256": sha256_file(candidate_path),
                    "midi_type_exact": strict["midi_type_exact"],
                    "ticks_per_beat_exact": strict["ticks_per_beat_exact"],
                    "track_count_exact": strict["track_count_exact"],
                    "all_ordered_non_cc64_events_exact": strict[
                        "all_ordered_non_cc64_events_exact"
                    ],
                    "canonical_cc64_events": strict["canonical_cc64_events"],
                    "candidate_cc64_events": strict["candidate_cc64_events"],
                    "cc64_changed": strict["cc64_changed"],
                    "donor_cc64_discarded_after_canonical_eot": transplant[
                        "donor_cc64_discarded_after_canonical_eot"
                    ],
                    "status": "PASS",
                }
            )
            if event_logger is not None:
                event_logger(
                    f"CANONICAL_PIECE_PASS architecture={architecture} "
                    f"piece={index}/{EXPECTED_PIECES} piece_id={piece_id} "
                    f"notes={details['notes']} windows={details['windows']}"
                )

    if not np.array_equal(original_histogram, expected_baseline):
        raise RuntimeError("epoch canonical Original-PT histogram changed")
    if input_notes != EXPECTED_CANONICAL_NOTES or int(candidate_histogram.sum()) != EXPECTED_CANONICAL_NOTES:
        raise RuntimeError(
            f"canonical candidate note count mismatch: input={input_notes} "
            f"roundtrip={int(candidate_histogram.sum())}"
        )
    if windows != EXPECTED_WINDOWS:
        raise RuntimeError(f"canonical validation window count mismatch: {windows}")
    if len(equality_rows) != EXPECTED_PIECES or not all(
        row["status"] == "PASS" for row in equality_rows
    ):
        raise AssertionError("canonical non-CC64 equality is not 19/19 PASS")
    for path, digest in canonical_hashes_before.items():
        if sha256_file(path) != digest:
            raise AssertionError(f"canonical MIDI changed during validation: {path}")

    metric = _strict_official_similarity(human, candidate_histogram)
    probability = candidate_histogram.astype(np.float64) / candidate_histogram.sum()
    steady = float(probability[0] + probability[15])
    return {
        "pieces": EXPECTED_PIECES,
        "windows": windows,
        "input_notes": input_notes,
        "strict_notes": int(candidate_histogram.sum()),
        "strict_candidate_histogram": candidate_histogram.tolist(),
        "strict_js_distance": float(metric["js_distance_base2"]),
        "strict_js_divergence": float(metric["js_divergence_base2"]),
        "strict_intersection": float(metric["histogram_intersection"]),
        "steady_mass": steady,
        "transition_containing_mass": 1.0 - steady,
        "non_cc64_equality": True,
        "non_cc64_equality_count": len(equality_rows),
        "equality_rows": equality_rows,
        "stage1_inference_regeneration": 0,
        "asap_test_midi_access_count": 0,
    }
