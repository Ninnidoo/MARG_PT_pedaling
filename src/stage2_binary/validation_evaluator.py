"""Reusable PT-style pedal-only validation helpers for binary Stage 2."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from miditoolkit import MidiFile

from src.stage2_encoder_only.dataset import (
    MASK_ID,
    NON_PEDAL_FEATURES,
    PEDAL_TOKEN_OFFSET,
    TOKENS_PER_NOTE,
    generate_window_starts,
)
from src.stage2_encoder_only.evaluate_oracle import distribution_similarity
from third_party.PianistTransformer.src.utils.midi import midi_to_ids

from .dataset import (
    BINARY_THRESHOLD,
    decode_joint_targets,
    encode_joint_targets,
)
from .model import IndependentBinaryPedalModel, JointBinaryPedalModel


JOINT_PATTERNS = tuple(f"{joint_id:04b}" for joint_id in range(16))
PT_PATTERN_EPSILON = 1e-10


def _as_note_tokens(token_ids: Sequence[int] | np.ndarray) -> np.ndarray:
    array = np.asarray(token_ids, dtype=np.int64)
    if array.ndim == 1:
        if array.size == 0 or array.size % TOKENS_PER_NOTE:
            raise ValueError("token sequence must contain complete 8-token notes")
        array = array.reshape(-1, TOKENS_PER_NOTE)
    if array.ndim != 2 or array.shape[1] != TOKENS_PER_NOTE or len(array) == 0:
        raise ValueError("tokens must have shape [notes,8]")
    return array


def raw_pedal_values_from_ids(
    token_ids: Sequence[int] | np.ndarray,
) -> np.ndarray:
    """Extract official raw 0--127 pedal values from PT note tokens."""

    notes = _as_note_tokens(token_ids)
    values = notes[:, NON_PEDAL_FEATURES:] - PEDAL_TOKEN_OFFSET
    if not bool(np.all((values >= 0) & (values <= 127))):
        raise ValueError("pedal tokens lie outside the official 0--127 range")
    return values


def joint_ids_from_raw_pedals(raw_values: np.ndarray) -> np.ndarray:
    values = np.asarray(raw_values)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("raw pedal values must have shape [notes,4]")
    if not bool(np.all((values >= 0) & (values <= 127))):
        raise ValueError("raw pedal values must be in [0,127]")
    bits = torch.from_numpy((values >= BINARY_THRESHOLD).astype(np.int64))
    return encode_joint_targets(bits).numpy()


def joint16_histogram_from_ids(
    token_ids: Sequence[int] | np.ndarray,
) -> np.ndarray:
    joint_ids = joint_ids_from_raw_pedals(raw_pedal_values_from_ids(token_ids))
    return np.bincount(joint_ids, minlength=16).astype(np.int64)


def joint16_histogram_from_midis(
    midi_paths: Iterable[str | Path],
    tokenizer_config: Any,
) -> tuple[np.ndarray, int, int]:
    """Tokenize allowed MIDI paths with official PT and aggregate globally."""

    histogram = np.zeros(16, dtype=np.int64)
    performances = 0
    notes = 0
    for midi_path in midi_paths:
        token_ids = midi_to_ids(tokenizer_config, MidiFile(str(midi_path)))
        current = joint16_histogram_from_ids(token_ids)
        histogram += current
        performances += 1
        notes += len(token_ids) // TOKENS_PER_NOTE
    return histogram, performances, notes


def normalized_joint16(histogram: Sequence[int]) -> np.ndarray:
    values = np.asarray(histogram, dtype=np.float64)
    if values.shape != (16,) or bool(np.any(values < 0)) or values.sum() <= 0:
        raise ValueError("joint histogram must be non-empty, non-negative, 16-bin")
    probabilities = values / values.sum() + PT_PATTERN_EPSILON
    probabilities /= probabilities.sum()
    if not np.isclose(probabilities.sum(), 1.0, rtol=0.0, atol=1e-12):
        raise AssertionError("joint distribution does not sum to one")
    return probabilities


def official_pt_pedal_similarity(
    human_histogram: Sequence[int],
    candidate_histogram: Sequence[int],
) -> dict[str, float]:
    """Reuse the vetted PT-style base-2 metric helper from Stage 2 audit."""

    metrics = distribution_similarity(
        np.asarray(human_histogram, dtype=np.int64),
        np.asarray(candidate_histogram, dtype=np.int64),
        epsilon=PT_PATTERN_EPSILON,
    )
    if not all(np.isfinite(value) for value in metrics.values()):
        raise FloatingPointError("non-finite PT-style pedal distribution metric")
    return metrics


def replace_pedal_tokens(
    stage1_token_ids: Sequence[int] | np.ndarray,
    raw_pedal_values: np.ndarray,
) -> np.ndarray:
    """Replace only Pedal1--4 and assert exact non-pedal preservation."""

    original = _as_note_tokens(stage1_token_ids)
    raw = np.asarray(raw_pedal_values, dtype=np.int64)
    if raw.shape != (len(original), 4):
        raise ValueError("replacement pedal values must have shape [notes,4]")
    if not bool(np.all((raw >= 0) & (raw <= 127))):
        raise ValueError("replacement pedal values must be in [0,127]")
    replaced = original.copy()
    replaced[:, NON_PEDAL_FEATURES:] = raw + PEDAL_TOKEN_OFFSET
    if not np.array_equal(
        original[:, :NON_PEDAL_FEATURES],
        replaced[:, :NON_PEDAL_FEATURES],
    ):
        raise AssertionError("Stage 2 replacement changed non-pedal tokens")
    return replaced.reshape(-1)


def binary_logits_to_raw_pedals(
    logits: torch.Tensor,
    architecture: str,
) -> np.ndarray:
    """Decode either binary model head to renderer-compatible 0/127 values."""

    if architecture == "independent_4x2":
        if logits.ndim != 3 or tuple(logits.shape[-2:]) != (4, 2):
            raise ValueError("independent logits must have shape [N,4,2]")
        bits = logits.argmax(dim=-1)
    elif architecture == "joint_16":
        if logits.ndim != 2 or logits.shape[-1] != 16:
            raise ValueError("joint logits must have shape [N,16]")
        bits = decode_joint_targets(logits.argmax(dim=-1))
    else:
        raise ValueError(f"unsupported binary architecture: {architecture}")
    if not bool(torch.all((bits == 0) | (bits == 1))):
        raise AssertionError("decoded binary model output is not binary")
    return (bits.cpu().numpy().astype(np.int64) * 127)


def infer_cached_binary_pedals(
    model: torch.nn.Module,
    stage1_token_ids: Sequence[int] | np.ndarray,
    *,
    architecture: str,
    device: torch.device,
    window_notes: int = 512,
    stride_notes: int = 256,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Window a cached Stage 1 sequence and overlap-average Stage 2 logits."""

    tokens = _as_note_tokens(stage1_token_ids)
    masked = tokens.copy()
    masked[:, NON_PEDAL_FEATURES:] = MASK_ID
    if not np.array_equal(
        masked[:, :NON_PEDAL_FEATURES], tokens[:, :NON_PEDAL_FEATURES]
    ):
        raise AssertionError("masking changed Stage 1 non-pedal tokens")
    starts = generate_window_starts(len(tokens), window_notes, stride_notes)
    output_shape = (len(tokens), 4, 2) if architecture == "independent_4x2" else (len(tokens), 16)
    accumulated = torch.zeros(output_shape, dtype=torch.float32, device=device)
    contributions = torch.zeros(len(tokens), dtype=torch.float32, device=device)
    model.eval()
    with torch.inference_mode():
        for start in starts:
            end = min(start + window_notes, len(tokens))
            input_ids = torch.from_numpy(masked[start:end].reshape(1, -1)).long().to(device)
            attention = torch.ones_like(input_ids)
            note_mask = torch.ones((1, end - start), dtype=torch.bool, device=device)
            output = model(
                input_ids=input_ids,
                token_attention_mask=attention,
                note_mask=note_mask,
            )
            logits = output.logits[0].to(torch.float32)
            if tuple(logits.shape) != tuple(accumulated[start:end].shape):
                raise ValueError("binary Stage 2 output shape does not match architecture")
            accumulated[start:end] += logits
            contributions[start:end] += 1
    if bool(torch.any(contributions == 0)):
        raise AssertionError("one or more cached Stage 1 notes received no prediction")
    divisor = contributions.view((-1,) + (1,) * (accumulated.ndim - 1))
    mean_logits = accumulated / divisor
    raw = binary_logits_to_raw_pedals(mean_logits, architecture)
    candidate = replace_pedal_tokens(tokens, raw)
    return candidate, {
        "notes": len(tokens),
        "windows": len(starts),
        "window_notes": window_notes,
        "stride_notes": stride_notes,
        "non_pedal_tokens_preserved": bool(
            np.array_equal(
                candidate.reshape(-1, 8)[:, :4],
                tokens[:, :4],
            )
        ),
    }


def load_binary_stage2_checkpoint(
    checkpoint_path: str | Path,
    *,
    architecture: str,
    encoder_checkpoint: str | Path,
    device: torch.device,
) -> torch.nn.Module:
    """Load a future Model A/B state on the same official PT encoder."""

    model_classes = {
        "independent_4x2": IndependentBinaryPedalModel,
        "joint_16": JointBinaryPedalModel,
    }
    if architecture not in model_classes:
        raise ValueError(f"unsupported binary architecture: {architecture}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = payload.get("model_state", payload) if isinstance(payload, Mapping) else payload
    if not isinstance(state, Mapping):
        raise TypeError("binary checkpoint must be a state dict or contain model_state")
    model = model_classes[architecture].from_pretrained(
        encoder_checkpoint,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("binary Stage 2 checkpoint state-dict mismatch")
    if not all(parameter.requires_grad for parameter in model.encoder.parameters()):
        raise AssertionError("loaded binary Stage 2 encoder is unexpectedly frozen")
    return model.to(device).eval()


def select_validation_scores(
    split_rows: Sequence[Mapping[str, str]],
    metadata_rows: Sequence[Mapping[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Resolve one directly linked score per validation piece deterministically."""

    validation_rows = [row for row in split_rows if row["split"] == "validation"]
    grouped: dict[str, list[tuple[Mapping[str, str], Mapping[str, str]]]] = defaultdict(list)
    for split_row in validation_rows:
        metadata_row = metadata_rows[int(split_row["metadata_index"])]
        performance = metadata_row["midi_performance"].replace("\\", "/")
        if performance != split_row["performance_path"].replace("\\", "/"):
            raise ValueError("ASAP split metadata_index does not identify its performance")
        grouped[split_row["piece_id"]].append((split_row, metadata_row))
    selected: list[dict[str, Any]] = []
    for piece_id, items in grouped.items():
        score_counts = Counter(
            metadata_row["midi_score"].replace("\\", "/")
            for _, metadata_row in items
        )
        maximum = max(score_counts.values())
        score_path = min(
            path for path, count in score_counts.items() if count == maximum
        )
        first = items[0][0]
        selected.append(
            {
                "piece_id": piece_id,
                "composer": first["composer"],
                "title": first["title"],
                "validation_performance_count": len(items),
                "selected_score_path": score_path,
                "selected_score_support": maximum,
                "score_candidate_count": len(score_counts),
                "score_candidates": ";".join(
                    f"{path}|{score_counts[path]}" for path in sorted(score_counts)
                ),
                "selection_rule": "max direct validation-row metadata support; lexical tie-break",
            }
        )
    selected.sort(key=lambda row: (row["composer"], row["title"], row["piece_id"]))
    return selected, validation_rows


def read_validation_selection(
    split_csv: str | Path,
    metadata_csv: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    with Path(split_csv).open(newline="", encoding="utf-8") as handle:
        split_rows = list(csv.DictReader(handle))
    with Path(metadata_csv).open(newline="", encoding="utf-8") as handle:
        metadata_rows = list(csv.DictReader(handle))
    return select_validation_scores(split_rows, metadata_rows)
