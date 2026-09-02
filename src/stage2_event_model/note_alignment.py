"""Exact frozen-cache note to official PT input-note alignment.

The frozen cache sorts raw MIDI notes, whereas PianistTransformer rounds
tempo-normalized timestamps before sorting. Adjacent raw onsets can therefore
collapse onto one PT tick and permute without changing the modeled note set.
This module reproduces official normalization while carrying cache indices as
provenance. It never changes either sequence or any frozen target.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from miditoolkit import MidiFile

from src.stage2_event_tokenizer.tokenizer import parse_raw_midi
from src.stage2_event_tokenizer.tokenizer_v1 import canonical_note_order
from third_party.PianistTransformer.src.utils.midi import normalize_midi

ALIGNMENT_IMPLEMENTATION_VERSION = "custom_event_cache_to_pt_v1.1"
PT_TARGET_TICKS_PER_BEAT = 500
PT_TARGET_TEMPO = 120


class NoteAlignmentError(ValueError):
    """The frozen cache and official PT modeled notes are not bijective."""


@dataclass(frozen=True)
class NoteAlignment:
    cache_to_pt: np.ndarray
    pt_to_cache: np.ndarray
    normalized_onset: np.ndarray
    normalized_pitch: np.ndarray
    normalized_offset: np.ndarray
    normalized_velocity: np.ndarray
    cache_raw_onset: np.ndarray
    stats: Mapping[str, Any]

    @property
    def note_count(self) -> int:
        return int(len(self.cache_to_pt))

    def map_group_bounds(
        self,
        first: Sequence[int] | np.ndarray,
        last: Sequence[int] | np.ndarray,
        representative: Sequence[int] | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Map frozen cache onset groups into official PT index space."""
        first_array = np.asarray(first, dtype=np.int64)
        last_array = np.asarray(last, dtype=np.int64)
        representative_array = np.asarray(representative, dtype=np.int64)
        if not (
            first_array.shape == last_array.shape == representative_array.shape
            and first_array.ndim == 1
        ):
            raise NoteAlignmentError("onset metadata shape mismatch")
        mapped_first = np.empty_like(first_array)
        mapped_last = np.empty_like(last_array)
        mapped_representative = self.cache_to_pt[representative_array]
        non_contiguous = 0
        for index, (left, right) in enumerate(zip(first_array, last_array)):
            positions = np.sort(self.cache_to_pt[int(left) : int(right) + 1])
            if np.any(positions < 0):
                raise NoteAlignmentError(f"unmapped cache note in onset group {index}")
            mapped_first[index] = int(positions[0])
            mapped_last[index] = int(positions[-1])
            expected = np.arange(positions[0], positions[0] + len(positions), dtype=np.int64)
            if not np.array_equal(positions, expected):
                non_contiguous += 1
            normalized = self.normalized_onset[positions]
            if np.any(normalized != normalized[0]):
                raise NoteAlignmentError(f"cache onset group {index} maps to multiple PT onsets")
            rep = int(mapped_representative[index])
            if not mapped_first[index] <= rep <= mapped_last[index]:
                raise NoteAlignmentError(f"mapped representative outside group {index}")
        return mapped_first, mapped_last, mapped_representative, non_contiguous


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cache_arrays(cache_path: str | Path) -> dict[str, np.ndarray]:
    with np.load(cache_path, allow_pickle=False) as cache:
        return {
            "onset": cache["note_onset_ticks"].astype(np.int64, copy=True),
            "pitch": cache["note_pitch"].astype(np.int64, copy=True),
            "offset": cache["note_offset_ticks"].astype(np.int64, copy=True),
            "velocity": cache["note_velocity"].astype(np.int64, copy=True),
        }


def _source_to_cache_indices(
    source_midi: str | Path,
    frozen: Mapping[str, np.ndarray],
) -> tuple[list[tuple[Any, int]], dict[str, int]]:
    """Associate official source traversal with cache indices by raw identity.

    Common identity is ``(onset,pitch,noteoff,velocity)``. Exact duplicates in
    these shared fields are rejected as ambiguous instead of being guessed.
    """
    raw = parse_raw_midi(source_midi)
    ordered = canonical_note_order(raw)
    expected = np.asarray([(r[3], r[2], r[4], r[5]) for r in ordered], dtype=np.int64)
    actual = np.column_stack(
        (frozen["onset"], frozen["pitch"], frozen["offset"], frozen["velocity"])
    )
    if expected.shape != actual.shape or not np.array_equal(expected, actual):
        raise NoteAlignmentError("frozen cache notes differ from canonical raw parser")

    buckets: dict[tuple[int, int, int, int], deque[int]] = defaultdict(deque)
    for cache_index, identity in enumerate(map(tuple, actual.tolist())):
        buckets[identity].append(cache_index)
    duplicate_keys = sum(len(indices) > 1 for indices in buckets.values())
    duplicate_notes = sum(len(indices) for indices in buckets.values() if len(indices) > 1)
    if duplicate_keys:
        raise NoteAlignmentError(
            "ambiguous common raw note identities: "
            f"keys={duplicate_keys}, notes={duplicate_notes}"
        )

    midi = MidiFile(str(source_midi))
    result: list[tuple[Any, int]] = []
    unmatched_source = 0
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            identity = (int(note.start), int(note.pitch), int(note.end), int(note.velocity))
            if not buckets[identity]:
                unmatched_source += 1
                continue
            result.append((note, buckets[identity].popleft()))
    unmatched_cache = sum(len(indices) for indices in buckets.values())
    if unmatched_source or unmatched_cache or len(result) != len(actual):
        raise NoteAlignmentError(
            "raw source/cache note identity mismatch: "
            f"unmatched_source={unmatched_source}, unmatched_cache={unmatched_cache}"
        )
    return result, {
        "duplicate_raw_identity_keys": int(duplicate_keys),
        "notes_in_duplicate_raw_identities": int(duplicate_notes),
    }


def build_note_alignment(source_midi: str | Path, cache_path: str | Path) -> NoteAlignment:
    """Build an exact cache-index <-> official-PT-index bijection."""
    frozen = _cache_arrays(cache_path)
    source_pairs, duplicate_stats = _source_to_cache_indices(source_midi, frozen)
    midi = MidiFile(str(source_midi))
    tick_to_time = midi.get_tick_to_time_mapping()
    factor = PT_TARGET_TICKS_PER_BEAT * (PT_TARGET_TEMPO / 60.0)

    converted: list[dict[str, int]] = []
    for source_order, (note, cache_index) in enumerate(source_pairs):
        start = round(float(tick_to_time[int(note.start)]) * factor)
        end = round(float(tick_to_time[int(note.end)]) * factor)
        if start >= end:
            end = start + 1
        converted.append({
            "cache_index": int(cache_index), "source_order": int(source_order),
            "start": int(start), "end": int(end), "pitch": int(note.pitch),
            "velocity": int(note.velocity),
        })

    by_pitch: dict[int, list[dict[str, int]]] = defaultdict(list)
    for note in converted:
        by_pitch[note["pitch"]].append(note)
    retained: list[dict[str, int]] = []
    dropped: list[int] = []
    for pitch in sorted(by_pitch):
        notes = sorted(by_pitch[pitch], key=lambda item: item["start"])
        for index in range(len(notes) - 1):
            current, following = notes[index], notes[index + 1]
            if current["end"] >= following["start"]:
                current["end"] = following["start"]
                if current["start"] >= current["end"]:
                    dropped.append(current["cache_index"])
                    current["dropped"] = 1
        retained.extend(note for note in notes if not note.get("dropped"))
    retained.sort(key=lambda item: (item["start"], item["pitch"]))

    official = normalize_midi(MidiFile(str(source_midi))).instruments[0].notes
    simulated = [(n["start"], n["pitch"], n["end"], n["velocity"]) for n in retained]
    official_identity = [
        (int(n.start), int(n.pitch), int(n.end), int(n.velocity)) for n in official
    ]
    if simulated != official_identity:
        raise NoteAlignmentError("provenance-carrying normalization differs from official PT")

    cache_count = len(frozen["pitch"])
    if dropped or len(retained) != cache_count:
        raise NoteAlignmentError(f"official PT dropped {len(dropped)} frozen cache notes")
    cache_to_pt = np.full(cache_count, -1, dtype=np.int64)
    pt_to_cache = np.full(cache_count, -1, dtype=np.int64)
    for pt_index, note in enumerate(retained):
        cache_index = note["cache_index"]
        if cache_to_pt[cache_index] >= 0:
            raise NoteAlignmentError(f"duplicate mapping for cache note {cache_index}")
        cache_to_pt[cache_index] = pt_index
        pt_to_cache[pt_index] = cache_index
    if np.any(cache_to_pt < 0) or np.any(pt_to_cache < 0):
        raise NoteAlignmentError("mapping is not total")
    if not np.array_equal(pt_to_cache[cache_to_pt], np.arange(cache_count)):
        raise NoteAlignmentError("mapping is not bijective")

    norm_onset = np.asarray([n["start"] for n in retained], dtype=np.int64)
    norm_pitch = np.asarray([n["pitch"] for n in retained], dtype=np.int16)
    norm_offset = np.asarray([n["end"] for n in retained], dtype=np.int64)
    norm_velocity = np.asarray([n["velocity"] for n in retained], dtype=np.int16)
    moved = np.flatnonzero(cache_to_pt != np.arange(cache_count))
    same_raw = 0
    same_normalized = 0
    for cache_index in moved.tolist():
        pt_index = int(cache_to_pt[cache_index])
        displaced_cache = int(pt_to_cache[cache_index])
        if frozen["onset"][cache_index] == frozen["onset"][displaced_cache]:
            same_raw += 1
        if norm_onset[pt_index] == norm_onset[cache_index]:
            same_normalized += 1
    stats = {
        **duplicate_stats,
        "note_count": int(cache_count),
        "identity_order": bool(not moved.size),
        "moved_notes": int(moved.size),
        "maximum_displacement": int(np.max(np.abs(cache_to_pt - np.arange(cache_count))) if cache_count else 0),
        "moved_within_same_raw_onset": int(same_raw),
        "moved_across_raw_onsets": int(moved.size - same_raw),
        "moved_within_same_normalized_onset": int(same_normalized),
        "moved_across_normalized_onsets": int(moved.size - same_normalized),
        "unmapped_cache_notes": 0, "unmapped_pt_notes": 0, "ambiguous_mappings": 0,
    }
    return NoteAlignment(
        cache_to_pt, pt_to_cache, norm_onset, norm_pitch, norm_offset,
        norm_velocity, frozen["onset"], stats,
    )


def save_note_alignment(path: str | Path, alignment: NoteAlignment, metadata: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dict(metadata), sort_keys=True, separators=(",", ":"))
    fd, temporary_name = tempfile.mkstemp(dir=destination.parent, prefix=destination.name + ".", suffix=".tmp")
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream, cache_to_pt=alignment.cache_to_pt, pt_to_cache=alignment.pt_to_cache,
                normalized_onset=alignment.normalized_onset,
                normalized_pitch=alignment.normalized_pitch,
                normalized_offset=alignment.normalized_offset,
                normalized_velocity=alignment.normalized_velocity,
                cache_raw_onset=alignment.cache_raw_onset,
                metadata_json=np.asarray(payload),
            )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_note_alignment(path: str | Path) -> tuple[NoteAlignment, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        cache_to_pt = data["cache_to_pt"].astype(np.int64, copy=True)
        pt_to_cache = data["pt_to_cache"].astype(np.int64, copy=True)
        alignment = NoteAlignment(
            cache_to_pt, pt_to_cache,
            data["normalized_onset"].astype(np.int64, copy=True),
            data["normalized_pitch"].astype(np.int16, copy=True),
            data["normalized_offset"].astype(np.int64, copy=True),
            data["normalized_velocity"].astype(np.int16, copy=True),
            data["cache_raw_onset"].astype(np.int64, copy=True),
            metadata.get("stats", {}),
        )
    if not np.array_equal(pt_to_cache[cache_to_pt], np.arange(len(cache_to_pt))):
        raise NoteAlignmentError(f"serialized mapping is not bijective: {path}")
    return alignment, metadata


def alignment_file_sha256(path: str | Path) -> str:
    return sha256_file(path)
