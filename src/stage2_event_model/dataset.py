"""Cache-backed 512/256 dataset for Custom Event Model v0."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from miditoolkit import MidiFile
from torch.utils.data import Dataset

from src.stage2_encoder_only.dataset import (
    MASK_ID,
    NON_PEDAL_FEATURES,
    PAD_ID,
    TOKENS_PER_NOTE,
    _PinnedTokenizerConfig,
)
from third_party.PianistTransformer.src.utils.midi import midi_to_ids

from .ownership import OwnershipResult, assign_unique_owners
from .note_alignment import NoteAlignment, load_note_alignment, sha256_file


CANONICAL_CACHE_ID = "3a5520155b5db1e9a1da7f8148556aa3e1da852655c9adde25d3dbbd1d966263"
CANONICAL_ALIGNMENT_ID = "98de59ef7a41fba26a2c89fe686c273f6c8ec7f27979922e71813624ac1a4822"
DEFAULT_ALIGNMENT_ROOT = Path("/workspace/project/analysis/custom_event_model_v0_note_alignment")
MAIN_SLOTS = 6
TERMINAL_SLOTS = 4
EVENT_CLASSES = 5
INITIAL_CLASSES = 4
IGNORE_INDEX = -100


@dataclass(frozen=True)
class CachedPerformance:
    entry: Mapping[str, Any]
    cache_path: Path
    ownership: OwnershipResult
    num_notes: int
    num_onsets: int
    alignment: NoteAlignment
    alignment_path: Path


def load_cache_manifest(cache_root: str | Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    root = Path(cache_root)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "cache_manifest.json").read_text(encoding="utf-8"))
    if config["cache_id"] != CANONICAL_CACHE_ID or manifest["cache_id"] != CANONICAL_CACHE_ID:
        raise RuntimeError("Custom Event Tokenizer cache ID mismatch")
    if config["asap_test_access_count"] != 0 or manifest["asap_test_access_count"] != 0:
        raise RuntimeError("cache provenance reports ASAP test access")
    if manifest["train_count"] != 2062 or manifest["validation_count"] != 71:
        raise RuntimeError("canonical cache performance counts changed")
    return root, config, manifest


def performance_from_entry(
    root: Path,
    entry: Mapping[str, Any],
    *,
    window_notes: int,
    stride_notes: int,
    alignment_root: Path,
    alignment_entry: Mapping[str, Any],
) -> CachedPerformance:
    cache_path = root / entry["cache_file"]
    with np.load(cache_path, allow_pickle=False) as cache:
        if str(cache["cache_id"].item()) != CANONICAL_CACHE_ID:
            raise RuntimeError(f"per-performance cache ID mismatch: {cache_path}")
        notes = int(cache["non_pedal_note_count"])
        first = cache["onset_first_note_index"].astype(np.int64, copy=True)
        last = cache["onset_last_note_index"].astype(np.int64, copy=True)
        representative = cache["onset_representative_note_index"].astype(np.int64, copy=True)
    alignment_path = alignment_root / alignment_entry["alignment_file"]
    alignment, alignment_metadata = load_note_alignment(alignment_path)
    if sha256_file(alignment_path) != alignment_entry["alignment_file_sha256"]:
        raise RuntimeError(f"alignment artifact SHA mismatch: {alignment_path}")
    if alignment_metadata["alignment_config_sha256"] != alignment_entry["alignment_config_sha256"]:
        raise RuntimeError(f"alignment configuration mismatch: {alignment_path}")
    if alignment_metadata["tokenizer_cache_id"] != CANONICAL_CACHE_ID:
        raise RuntimeError(f"alignment/cache identity mismatch: {alignment_path}")
    if alignment_metadata["source_sha256"] != entry["source_sha256"]:
        raise RuntimeError(f"alignment/source identity mismatch: {alignment_path}")
    if alignment.note_count != notes:
        raise RuntimeError(f"alignment note count mismatch: {alignment_path}")
    mapped_first, mapped_last, mapped_representative, non_contiguous = alignment.map_group_bounds(
        first, last, representative
    )
    if non_contiguous:
        raise RuntimeError(f"non-contiguous mapped onset groups: {alignment_path}")
    ownership = assign_unique_owners(
        notes,
        mapped_first,
        mapped_last,
        mapped_representative,
        window_notes=window_notes,
        stride_notes=stride_notes,
    )
    return CachedPerformance(
        entry=entry,
        cache_path=cache_path,
        ownership=ownership,
        num_notes=notes,
        num_onsets=len(first),
        alignment=alignment,
        alignment_path=alignment_path,
    )


class CustomEventWindowDataset(Dataset[dict[str, Any]]):
    """Read frozen global targets and attach them to their unique owner windows."""

    def __init__(
        self,
        cache_root: str | Path,
        split: str,
        *,
        window_notes: int = 512,
        stride_notes: int = 256,
        cache_input_tokens: bool = False,
        entry_limit: int | None = None,
        alignment_root: str | Path = DEFAULT_ALIGNMENT_ROOT,
    ) -> None:
        if split not in {"train", "validation"}:
            raise ValueError("only train and validation caches are allowed")
        self.root, self.cache_config, manifest = load_cache_manifest(cache_root)
        entries = [entry for entry in manifest["entries"] if entry["split"] == split]
        if entry_limit is not None:
            entries = entries[: int(entry_limit)]
        if not entries:
            raise ValueError(f"no cache entries for split {split}")
        self.split = split
        self.alignment_root = Path(alignment_root)
        alignment_manifest = json.loads(
            (self.alignment_root / "alignment_manifest.json").read_text(encoding="utf-8")
        )
        if alignment_manifest["alignment_id"] != CANONICAL_ALIGNMENT_ID:
            raise RuntimeError("canonical note-alignment ID mismatch")
        if alignment_manifest["tokenizer_cache_id"] != CANONICAL_CACHE_ID:
            raise RuntimeError("note-alignment tokenizer-cache ID mismatch")
        if alignment_manifest["asap_test_access_count"] != 0:
            raise RuntimeError("note-alignment provenance reports ASAP test access")
        if alignment_manifest["train_count"] != 2062 or alignment_manifest["validation_count"] != 71:
            raise RuntimeError("canonical note-alignment performance counts changed")
        alignment_entries = {
            (item["split"], int(item["performance_index"])): {
                **item,
                "alignment_config_sha256": alignment_manifest["alignment_config_sha256"],
            }
            for item in alignment_manifest["entries"]
        }
        self.window_notes = int(window_notes)
        self.stride_notes = int(stride_notes)
        self.cache_input_tokens = bool(cache_input_tokens)
        self._tokenizer_config = _PinnedTokenizerConfig()
        self._token_cache: dict[int, np.ndarray] = {}
        self.performances = [
            performance_from_entry(
                self.root,
                entry,
                window_notes=self.window_notes,
                stride_notes=self.stride_notes,
                alignment_root=self.alignment_root,
                alignment_entry=alignment_entries[(split, int(entry["performance_index"]))],
            )
            for entry in entries
        ]
        self.windows: list[tuple[int, int]] = [
            (performance_index, window_index)
            for performance_index, performance in enumerate(self.performances)
            for window_index in range(len(performance.ownership.window_starts))
        ]

    def __len__(self) -> int:
        return len(self.windows)

    def _load_input_tokens(self, performance_index: int) -> np.ndarray:
        if performance_index in self._token_cache:
            return self._token_cache[performance_index]
        performance = self.performances[performance_index]
        ids = np.asarray(
            midi_to_ids(
                self._tokenizer_config,
                MidiFile(str(performance.entry["source_midi"])),
            ),
            dtype=np.int64,
        )
        if ids.size % TOKENS_PER_NOTE:
            raise ValueError("official PT tokenizer returned incomplete notes")
        tokens = ids.reshape(-1, TOKENS_PER_NOTE)
        with np.load(performance.cache_path, allow_pickle=False) as cache:
            if len(tokens) != int(cache["non_pedal_note_count"]):
                raise ValueError("PT input note count differs from frozen target cache")
            alignment = performance.alignment
            # PT order stays untouched. Verify its modeled identity against the
            # separately audited permutation instead of assuming cache order.
            if not np.array_equal(tokens[:, 0] - 5, alignment.normalized_pitch):
                raise ValueError("PT pitch sequence differs from audited alignment")
            if not np.array_equal(tokens[:, 2] - 133, alignment.normalized_velocity):
                raise ValueError("PT velocity sequence differs from audited alignment")
            expected_duration = np.clip(
                alignment.normalized_offset - alignment.normalized_onset,
                self._tokenizer_config.valid_id_range[3][0] - self._tokenizer_config.timing_start,
                self._tokenizer_config.valid_id_range[3][1] - 1 - self._tokenizer_config.timing_start,
            )
            if not np.array_equal(tokens[:, 3] - self._tokenizer_config.timing_start, expected_duration):
                raise ValueError("PT duration sequence differs from audited alignment")
            mapped_pitch = (tokens[:, 0] - 5)[alignment.cache_to_pt]
            mapped_velocity = (tokens[:, 2] - 133)[alignment.cache_to_pt]
            if not np.array_equal(mapped_pitch, cache["note_pitch"]):
                raise ValueError("cache->PT mapped pitch identity mismatch")
            if not np.array_equal(mapped_velocity, cache["note_velocity"]):
                raise ValueError("cache->PT mapped velocity identity mismatch")
        tokens = tokens.astype(np.int16, copy=False)
        tokens.setflags(write=False)
        if self.cache_input_tokens:
            self._token_cache[performance_index] = tokens
        return tokens

    def __getitem__(self, index: int) -> dict[str, Any]:
        performance_index, window_index = self.windows[index]
        performance = self.performances[performance_index]
        ownership = performance.ownership
        start = ownership.window_starts[window_index]
        end = ownership.window_ends[window_index]
        tokens = self._load_input_tokens(performance_index)[start:end].copy()
        tokens[:, NON_PEDAL_FEATURES:] = MASK_ID
        owned = ownership.owned_onset_indices[window_index]
        local_representatives = ownership.owned_local_representative_indices[window_index]
        first_owner = int(ownership.owner_window_index[0])
        last_owner = int(ownership.owner_window_index[-1])
        with np.load(performance.cache_path, allow_pickle=False) as cache:
            sample = {
                "input_ids": torch.from_numpy(tokens.reshape(-1).copy()).long(),
                "note_mask": torch.ones(len(tokens), dtype=torch.bool),
                "owned_global_onset_indices": torch.from_numpy(owned.copy()).long(),
                "owned_representative_positions": torch.from_numpy(
                    local_representatives.copy()
                ).long(),
                "main_event_targets": torch.from_numpy(
                    cache["main_event_target"][owned].astype(np.int64, copy=True)
                ).long(),
                "main_timing_targets": torch.from_numpy(
                    cache["main_tau_target"][owned].astype(np.float32, copy=True)
                ),
                "main_timing_valid_mask": torch.from_numpy(
                    cache["main_timing_valid_mask"][owned].copy()
                ).bool(),
                "initial_target": torch.tensor(int(cache["initial_state"]), dtype=torch.long),
                "initial_valid_mask": torch.tensor(window_index == first_owner),
                "initial_representative_position": torch.tensor(
                    int(performance.alignment.cache_to_pt[int(cache["onset_representative_note_index"][0])]) - start
                    if window_index == first_owner
                    else -1,
                    dtype=torch.long,
                ),
                "terminal_event_targets": torch.from_numpy(
                    cache["terminal_event_target"].astype(np.int64, copy=True)
                ).long(),
                "terminal_timing_targets": torch.from_numpy(
                    cache["terminal_log1p_gap_target"].astype(np.float32, copy=True)
                ),
                "terminal_timing_valid_mask": torch.from_numpy(
                    cache["terminal_timing_valid_mask"].copy()
                ).bool(),
                "terminal_valid_mask": torch.tensor(window_index == last_owner),
                "terminal_representative_position": torch.tensor(
                    int(performance.alignment.cache_to_pt[int(cache["onset_representative_note_index"][-1])]) - start
                    if window_index == last_owner
                    else -1,
                    dtype=torch.long,
                ),
                "metadata": {
                    "split": self.split,
                    "performance_index": performance_index,
                    "performance_path": performance.entry["performance_path"],
                    "cache_file": str(performance.cache_path),
                    "window_index": window_index,
                    "window_start_note": start,
                    "window_end_note": end,
                    "total_notes": performance.num_notes,
                    "note_alignment_id": CANONICAL_ALIGNMENT_ID,
                    "note_alignment_file": str(performance.alignment_path),
                },
            }
        if not torch.equal(
            sample["main_timing_valid_mask"], sample["main_event_targets"] != 0
        ):
            raise AssertionError("frozen main event/timing mask mismatch")
        if not torch.equal(
            sample["terminal_timing_valid_mask"],
            sample["terminal_event_targets"] != 0,
        ):
            raise AssertionError("frozen terminal event/timing mask mismatch")
        return sample


def custom_event_collate_fn(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("cannot collate an empty batch")
    batch_size = len(samples)
    note_counts = [sample["input_ids"].numel() // TOKENS_PER_NOTE for sample in samples]
    owned_counts = [len(sample["owned_global_onset_indices"]) for sample in samples]
    max_notes = max(note_counts)
    max_owned = max(owned_counts, default=0)
    input_ids = torch.full((batch_size, max_notes * TOKENS_PER_NOTE), PAD_ID, dtype=torch.long)
    token_attention_mask = torch.zeros_like(input_ids)
    note_mask = torch.zeros((batch_size, max_notes), dtype=torch.bool)
    owned_mask = torch.zeros((batch_size, max_owned), dtype=torch.bool)
    owned_global = torch.full((batch_size, max_owned), -1, dtype=torch.long)
    owned_positions = torch.full((batch_size, max_owned), -1, dtype=torch.long)
    main_event = torch.full((batch_size, max_owned, MAIN_SLOTS), IGNORE_INDEX, dtype=torch.long)
    main_timing = torch.zeros((batch_size, max_owned, MAIN_SLOTS), dtype=torch.float32)
    main_timing_mask = torch.zeros((batch_size, max_owned, MAIN_SLOTS), dtype=torch.bool)
    for batch_index, (sample, notes, owned_count) in enumerate(
        zip(samples, note_counts, owned_counts)
    ):
        length = notes * TOKENS_PER_NOTE
        input_ids[batch_index, :length] = sample["input_ids"]
        token_attention_mask[batch_index, :length] = 1
        note_mask[batch_index, :notes] = True
        if owned_count:
            owned_mask[batch_index, :owned_count] = True
            owned_global[batch_index, :owned_count] = sample["owned_global_onset_indices"]
            owned_positions[batch_index, :owned_count] = sample["owned_representative_positions"]
            main_event[batch_index, :owned_count] = sample["main_event_targets"]
            main_timing[batch_index, :owned_count] = sample["main_timing_targets"]
            main_timing_mask[batch_index, :owned_count] = sample["main_timing_valid_mask"]
    return {
        "input_ids": input_ids,
        "token_attention_mask": token_attention_mask,
        "note_mask": note_mask,
        "owned_onset_mask": owned_mask,
        "owned_global_onset_indices": owned_global,
        "owned_representative_positions": owned_positions,
        "main_event_targets": main_event,
        "main_timing_targets": main_timing,
        "main_timing_valid_mask": main_timing_mask,
        "initial_targets": torch.stack([sample["initial_target"] for sample in samples]),
        "initial_valid_mask": torch.stack([sample["initial_valid_mask"] for sample in samples]).bool(),
        "initial_representative_positions": torch.stack(
            [sample["initial_representative_position"] for sample in samples]
        ),
        "terminal_event_targets": torch.stack(
            [sample["terminal_event_targets"] for sample in samples]
        ),
        "terminal_timing_targets": torch.stack(
            [sample["terminal_timing_targets"] for sample in samples]
        ),
        "terminal_timing_valid_mask": torch.stack(
            [sample["terminal_timing_valid_mask"] for sample in samples]
        ),
        "terminal_valid_mask": torch.stack(
            [sample["terminal_valid_mask"] for sample in samples]
        ).bool(),
        "terminal_representative_positions": torch.stack(
            [sample["terminal_representative_position"] for sample in samples]
        ),
        "metadata": [sample["metadata"] for sample in samples],
    }
