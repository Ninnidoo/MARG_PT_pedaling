"""On-demand ASAP dataset for masked sustain-pedal prediction."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from miditoolkit import MidiFile
from torch.utils.data import Dataset

from third_party.PianistTransformer.src.utils.midi import midi_to_ids


PAD_ID = 0
MASK_ID = 1
PEDAL_TOKEN_OFFSET = 5261
PEDAL_NUM_CLASSES = 128
TOKENS_PER_NOTE = 8
NON_PEDAL_FEATURES = 4
PEDAL_SLOTS = 4


class _PinnedTokenizerConfig:
    """Minimal configuration consumed by the pinned PT ``midi_to_ids``."""

    pitch_start = 5
    velocity_start = 133
    timing_start = 261
    pedal_start = PEDAL_TOKEN_OFFSET
    valid_id_range = [
        (5, 133),
        (261, 5252),
        (133, 261),
        (261, 5261),
        *((PEDAL_TOKEN_OFFSET, PEDAL_TOKEN_OFFSET + PEDAL_NUM_CLASSES),) * 4,
    ]


def generate_window_starts(
    num_notes: int, window_notes: int = 512, stride_notes: int = 256
) -> list[int]:
    """Return deterministic note-aligned window starts, including the tail."""

    if num_notes < 0:
        raise ValueError("num_notes must be non-negative")
    if window_notes <= 0 or stride_notes <= 0:
        raise ValueError("window_notes and stride_notes must be positive")
    if num_notes <= window_notes:
        return [0]

    starts = list(range(0, num_notes - window_notes + 1, stride_notes))
    tail_start = num_notes - window_notes
    if starts[-1] != tail_start:
        starts.append(tail_start)
    return starts


class Stage2PedalDataset(Dataset[dict[str, Any]]):
    """Return note-aligned windows with optional performance token preloading.

    Preload in the main process before creating DataLoader workers. Linux
    fork-based workers may share the read-only arrays through copy-on-write;
    spawn-based workers may duplicate the cache in each worker process.
    """

    def __init__(
        self,
        asap_root: str | Path,
        split_csv: str | Path,
        split: str,
        window_notes: int = 512,
        stride_notes: int = 256,
        return_metadata: bool = True,
        cache_mode: str = "none",
    ) -> None:
        if split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")
        if window_notes <= 0 or stride_notes <= 0:
            raise ValueError("window_notes and stride_notes must be positive")
        if cache_mode not in {"none", "preload"}:
            raise ValueError("cache_mode must be none or preload")

        self.asap_root = Path(asap_root).resolve()
        self.split_csv = Path(split_csv)
        self.split = split
        self.window_notes = window_notes
        self.stride_notes = stride_notes
        self.return_metadata = return_metadata
        self.cache_mode = cache_mode
        self._tokenizer_config = _PinnedTokenizerConfig()
        self._token_cache: dict[str, np.ndarray] = {}

        with self.split_csv.open(newline="", encoding="utf-8") as handle:
            rows = [row for row in csv.DictReader(handle) if row["split"] == split]
        if not rows:
            raise ValueError(f"split contains no performances: {split}")
        self.performances = rows

        self._windows: list[tuple[int, int, int]] = []
        for row_index, row in enumerate(rows):
            total_notes = int(row["num_normalized_notes"])
            for start in generate_window_starts(
                total_notes, self.window_notes, self.stride_notes
            ):
                end = min(start + self.window_notes, total_notes)
                self._windows.append((row_index, start, end))

        if self.cache_mode == "preload":
            self.preload_cache()

    @property
    def performance_count(self) -> int:
        return len(self.performances)

    @property
    def window_count(self) -> int:
        return len(self._windows)

    @property
    def cached_performance_count(self) -> int:
        return len(self._token_cache)

    @property
    def cached_token_count(self) -> int:
        return sum(tokens.size for tokens in self._token_cache.values())

    @property
    def cached_nbytes(self) -> int:
        return sum(tokens.nbytes for tokens in self._token_cache.values())

    def __len__(self) -> int:
        return self.window_count

    def _load_performance_tokens(self, row: Mapping[str, str]) -> np.ndarray:
        midi_path = (self.asap_root / row["performance_path"]).resolve()
        try:
            midi_path.relative_to(self.asap_root)
        except ValueError as exc:
            raise ValueError("performance path escapes ASAP root") from exc
        if not midi_path.is_file():
            raise FileNotFoundError(midi_path)

        ids = midi_to_ids(self._tokenizer_config, MidiFile(str(midi_path)))
        tokens = np.asarray(ids, dtype=np.int64)
        if tokens.size % TOKENS_PER_NOTE:
            raise ValueError("token count is not divisible by eight")
        tokens = tokens.reshape(-1, TOKENS_PER_NOTE)
        pedal_tokens = tokens[:, NON_PEDAL_FEATURES:]
        if not np.all(
            (pedal_tokens >= PEDAL_TOKEN_OFFSET)
            & (pedal_tokens < PEDAL_TOKEN_OFFSET + PEDAL_NUM_CLASSES)
        ):
            raise ValueError("pedal token outside pinned vocabulary range")
        expected_notes = int(row["num_normalized_notes"])
        if len(tokens) != expected_notes:
            raise ValueError(
                f"tokenized note count {len(tokens)} differs from split CSV "
                f"count {expected_notes}"
            )
        tokens = tokens.astype(np.int16, copy=False)
        tokens.setflags(write=False)
        return tokens

    def preload_cache(self) -> None:
        """Tokenize and retain every selected performance exactly once."""

        for row in self.performances:
            performance_path = row["performance_path"]
            if performance_path not in self._token_cache:
                self._token_cache[performance_path] = self._load_performance_tokens(
                    row
                )

    def _get_performance_tokens(self, row: Mapping[str, str]) -> np.ndarray:
        if self.cache_mode == "preload":
            return self._token_cache[row["performance_path"]]
        return self._load_performance_tokens(row)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row_index, start, end = self._windows[index]
        row = self.performances[row_index]
        original = self._get_performance_tokens(row)[start:end]
        input_tokens = original.copy()
        input_tokens[:, NON_PEDAL_FEATURES:] = MASK_ID
        pedal_targets = original[:, NON_PEDAL_FEATURES:] - PEDAL_TOKEN_OFFSET

        if not np.array_equal(
            input_tokens[:, :NON_PEDAL_FEATURES],
            original[:, :NON_PEDAL_FEATURES],
        ):
            raise AssertionError("masking modified a non-pedal feature")
        if not np.all((pedal_targets >= 0) & (pedal_targets < PEDAL_NUM_CLASSES)):
            raise ValueError("pedal target outside [0, 127]")

        sample: dict[str, Any] = {
            "input_ids": torch.from_numpy(input_tokens.reshape(-1)).long(),
            "pedal_targets": torch.from_numpy(pedal_targets.copy()).long(),
            "note_mask": torch.ones(len(original), dtype=torch.bool),
        }
        if self.return_metadata:
            sample["metadata"] = {
                "performance_path": row["performance_path"],
                "piece_id": row["piece_id"],
                "composer": row["composer"],
                "title": row["title"],
                "split": row["split"],
                "window_start_note": start,
                "window_end_note": end,
                "total_notes": int(row["num_normalized_notes"]),
            }
        return sample


def stage2_pedal_collate_fn(
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Pad a batch strictly at complete eight-token note boundaries."""

    if not samples:
        raise ValueError("cannot collate an empty batch")
    note_counts = []
    for sample in samples:
        input_ids = sample["input_ids"]
        if input_ids.numel() % TOKENS_PER_NOTE:
            raise ValueError("sample input length is not divisible by eight")
        notes = input_ids.numel() // TOKENS_PER_NOTE
        if tuple(sample["pedal_targets"].shape) != (notes, PEDAL_SLOTS):
            raise ValueError("pedal_targets shape does not match input notes")
        if tuple(sample["note_mask"].shape) != (notes,):
            raise ValueError("note_mask shape does not match input notes")
        note_counts.append(notes)

    batch_size, max_notes = len(samples), max(note_counts)
    input_ids = torch.full(
        (batch_size, max_notes * TOKENS_PER_NOTE), PAD_ID, dtype=torch.long
    )
    pedal_targets = torch.full(
        (batch_size, max_notes, PEDAL_SLOTS), -100, dtype=torch.long
    )
    note_mask = torch.zeros((batch_size, max_notes), dtype=torch.bool)
    token_attention_mask = torch.zeros(
        (batch_size, max_notes * TOKENS_PER_NOTE), dtype=torch.long
    )
    metadata = []

    for batch_index, (sample, notes) in enumerate(zip(samples, note_counts)):
        real_tokens = notes * TOKENS_PER_NOTE
        input_ids[batch_index, :real_tokens] = sample["input_ids"]
        pedal_targets[batch_index, :notes] = sample["pedal_targets"]
        note_mask[batch_index, :notes] = sample["note_mask"]
        token_attention_mask[batch_index, :real_tokens] = 1
        metadata.append(sample.get("metadata"))

    return {
        "input_ids": input_ids,
        "pedal_targets": pedal_targets,
        "note_mask": note_mask,
        "token_attention_mask": token_attention_mask,
        "metadata": metadata,
    }
