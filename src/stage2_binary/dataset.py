"""Small manifest-subset pipeline for binary Stage 2 implementation checks."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from miditoolkit import MidiFile
from torch.utils.data import Dataset

from src.stage2_encoder_only.dataset import (
    MASK_ID,
    NON_PEDAL_FEATURES,
    PEDAL_SLOTS,
    PEDAL_TOKEN_OFFSET,
    TOKENS_PER_NOTE,
    _PinnedTokenizerConfig,
    stage2_pedal_collate_fn,
)
from third_party.PianistTransformer.src.utils.midi import midi_to_ids


BINARY_THRESHOLD = 64
IGNORE_INDEX = -100
JOINT_WEIGHTS = (8, 4, 2, 1)


def _require_integer_tensor(values: torch.Tensor, name: str) -> None:
    if values.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise TypeError(f"{name} must have an integer dtype")


def binarize_raw_pedals(
    raw_values: torch.Tensor,
    *,
    allow_ignore: bool = False,
) -> torch.Tensor:
    """Apply the official evaluator boundary: raw <64 OFF, >=64 ON."""

    if raw_values.ndim < 1 or raw_values.shape[-1] != PEDAL_SLOTS:
        raise ValueError("raw_values must have shape [...,4]")
    _require_integer_tensor(raw_values, "raw_values")
    valid = (raw_values >= 0) & (raw_values <= 127)
    if allow_ignore:
        valid = valid | (raw_values == IGNORE_INDEX)
    if not bool(torch.all(valid)):
        raise ValueError("raw_values must be in [0,127] or the allowed ignore index")
    binary = (raw_values >= BINARY_THRESHOLD).to(torch.long)
    if allow_ignore:
        binary = binary.masked_fill(raw_values == IGNORE_INDEX, IGNORE_INDEX)
    return binary


def encode_joint_targets(
    binary_targets: torch.Tensor,
    *,
    allow_ignore: bool = False,
) -> torch.Tensor:
    """Encode P1..P4 as 8*P1+4*P2+2*P3+P4."""

    if binary_targets.ndim < 1 or binary_targets.shape[-1] != PEDAL_SLOTS:
        raise ValueError("binary_targets must have shape [...,4]")
    _require_integer_tensor(binary_targets, "binary_targets")
    bit_valid = (binary_targets == 0) | (binary_targets == 1)
    ignored = binary_targets == IGNORE_INDEX
    if allow_ignore:
        whole_ignored = ignored.all(dim=-1)
        partial_ignored = ignored.any(dim=-1) & ~whole_ignored
        if bool(partial_ignored.any()):
            raise ValueError("joint target cannot contain partially ignored bits")
        if not bool(torch.all(bit_valid | ignored)):
            raise ValueError("binary targets must be bits or ignore_index")
    elif not bool(torch.all(bit_valid)):
        raise ValueError("binary targets must be in {0,1}")
    weights = torch.tensor(
        JOINT_WEIGHTS,
        dtype=torch.long,
        device=binary_targets.device,
    )
    safe = binary_targets.masked_fill(ignored, 0).to(torch.long)
    joint = (safe * weights).sum(dim=-1)
    if allow_ignore:
        joint = joint.masked_fill(ignored.all(dim=-1), IGNORE_INDEX)
    return joint


def decode_joint_targets(
    joint_targets: torch.Tensor,
    *,
    allow_ignore: bool = False,
) -> torch.Tensor:
    """Decode joint IDs back to exact P1..P4 bits."""

    _require_integer_tensor(joint_targets, "joint_targets")
    valid = (joint_targets >= 0) & (joint_targets < 16)
    if allow_ignore:
        valid = valid | (joint_targets == IGNORE_INDEX)
    if not bool(torch.all(valid)):
        raise ValueError("joint targets must be in [0,15] or the allowed ignore index")
    ignored = joint_targets == IGNORE_INDEX
    safe = joint_targets.masked_fill(ignored, 0).to(torch.long)
    decoded = torch.stack(
        tuple((safe >> shift) & 1 for shift in (3, 2, 1, 0)),
        dim=-1,
    )
    if allow_ignore:
        decoded = decoded.masked_fill(ignored.unsqueeze(-1), IGNORE_INDEX)
    return decoded


def binary_stage2_collate_fn(
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Reuse baseline padding, then derive both equivalent binary targets."""

    batch = stage2_pedal_collate_fn(samples)
    raw_targets = batch.pop("pedal_targets")
    binary_targets = binarize_raw_pedals(raw_targets, allow_ignore=True)
    joint_targets = encode_joint_targets(binary_targets, allow_ignore=True)
    decoded = decode_joint_targets(joint_targets, allow_ignore=True)
    if not torch.equal(binary_targets, decoded):
        raise AssertionError("binary/joint round trip failed during collation")
    batch["binary_targets"] = binary_targets
    batch["joint_targets"] = joint_targets
    return batch


def make_masked_binary_sample(
    full_tokens: np.ndarray,
    start: int,
    end: int,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create one baseline-compatible window without changing non-pedal IDs."""

    tokens = np.asarray(full_tokens)
    if tokens.ndim != 2 or tokens.shape[1] != TOKENS_PER_NOTE:
        raise ValueError("full_tokens must have shape [notes,8]")
    if start < 0 or end <= start or end > len(tokens):
        raise ValueError("window lies outside tokenized performance")
    original = tokens[start:end]
    input_tokens = original.copy()
    input_tokens[:, NON_PEDAL_FEATURES:] = MASK_ID
    raw_targets = original[:, NON_PEDAL_FEATURES:] - PEDAL_TOKEN_OFFSET
    if not np.array_equal(
        input_tokens[:, :NON_PEDAL_FEATURES],
        original[:, :NON_PEDAL_FEATURES],
    ):
        raise AssertionError("masking changed a non-pedal input token")
    if not np.all(input_tokens[:, NON_PEDAL_FEATURES:] == MASK_ID):
        raise AssertionError("not every pedal input position was masked")
    if not np.all((raw_targets >= 0) & (raw_targets <= 127)):
        raise ValueError("official pedal target outside [0,127]")
    payload = dict(metadata or {})
    payload.update(window_start_note=start, window_end_note=end)
    return {
        "input_ids": torch.from_numpy(input_tokens.reshape(-1)).long(),
        # The baseline collator consumes this raw field; the binary collator
        # replaces it with binary_targets and joint_targets.
        "pedal_targets": torch.from_numpy(raw_targets.copy()).long(),
        "note_mask": torch.ones(end - start, dtype=torch.bool),
        "metadata": payload,
    }


class BinaryManifestSubsetDataset(Dataset[dict[str, Any]]):
    """Tokenize only explicitly selected manifest rows for implementation tests."""

    def __init__(
        self,
        manifest_csv: str | Path,
        selected_performance_paths: Sequence[str],
        *,
        window_notes: int,
        window_starts: Sequence[int],
        required_source: str | None = None,
    ) -> None:
        if not selected_performance_paths:
            raise ValueError("selected_performance_paths must be non-empty")
        if len(set(selected_performance_paths)) != len(selected_performance_paths):
            raise ValueError("selected performance paths must be unique")
        if window_notes <= 0 or not window_starts:
            raise ValueError("window_notes and window_starts must be non-empty")
        if any(start < 0 for start in window_starts):
            raise ValueError("window starts must be non-negative")
        self.manifest_csv = Path(manifest_csv)
        with self.manifest_csv.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        by_path: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            by_path.setdefault(row["performance_path"], []).append(row)
        selected_rows = []
        for performance_path in selected_performance_paths:
            candidates = by_path.get(performance_path, [])
            if required_source is not None:
                candidates = [
                    row for row in candidates if row["source"] == required_source
                ]
            if len(candidates) != 1:
                raise ValueError(
                    f"expected one selected manifest row for {performance_path}, "
                    f"found {len(candidates)}"
                )
            selected_rows.append(candidates[0])
        self.rows = selected_rows
        self.window_notes = int(window_notes)
        self.window_starts = tuple(int(start) for start in window_starts)
        self._token_arrays: dict[str, np.ndarray] = {}
        self._windows: list[tuple[str, int, int, dict[str, str]]] = []
        config = _PinnedTokenizerConfig()
        for row in self.rows:
            absolute_path = Path(row["performance_absolute_path"])
            if not absolute_path.is_file():
                raise FileNotFoundError(absolute_path)
            ids = midi_to_ids(config, MidiFile(str(absolute_path)))
            tokens = np.asarray(ids, dtype=np.int64)
            if tokens.size % TOKENS_PER_NOTE:
                raise ValueError("official token count is not divisible by eight")
            tokens = tokens.reshape(-1, TOKENS_PER_NOTE).astype(np.int16)
            tokens.setflags(write=False)
            self._token_arrays[row["performance_path"]] = tokens
            for start in self.window_starts:
                end = start + self.window_notes
                if end > len(tokens):
                    raise ValueError(
                        f"selected window exceeds {row['performance_path']}"
                    )
                self._windows.append((row["performance_path"], start, end, row))

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        performance_path, start, end, row = self._windows[index]
        return make_masked_binary_sample(
            self._token_arrays[performance_path],
            start,
            end,
            metadata={
                "source": row["source"],
                "dataset_split": row["dataset_split"],
                "performance_path": performance_path,
            },
        )
