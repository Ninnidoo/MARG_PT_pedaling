"""Performance-global owner assembly and constrained Run B inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import torch

from src.stage2_binary_2slot.frozen_pt_validation import FrozenPTPiece

from .decoding import RawHeadPrediction, raw_head_decode
from .viterbi import ViterbiDecodeResult, constrained_viterbi_decode


@dataclass(frozen=True)
class PerformanceHeadLogits:
    state_logits: torch.Tensor
    mode_logits: torch.Tensor
    timing_predictions: torch.Tensor


@dataclass(frozen=True)
class PerformanceDecode:
    raw: RawHeadPrediction
    viterbi: ViterbiDecodeResult


def assemble_performance_owner_logits(
    num_onsets: int,
    owner_chunks: Iterable[tuple[Sequence[int] | torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
) -> PerformanceHeadLogits:
    """Assemble each owned onset once, then expose M State and M-1 edge heads."""

    if num_onsets < 1:
        raise ValueError("performance must contain at least one onset")
    state: torch.Tensor | None = None
    mode: torch.Tensor | None = None
    timing: torch.Tensor | None = None
    seen = torch.zeros(num_onsets, dtype=torch.int64)
    for indices_value, state_chunk, mode_chunk, timing_chunk in owner_chunks:
        indices = torch.as_tensor(indices_value, dtype=torch.long).cpu()
        if indices.ndim != 1 or bool(torch.any((indices < 0) | (indices >= num_onsets))):
            raise ValueError("owned onset index lies outside performance")
        if len(indices) > 1 and bool(torch.any(indices[1:] <= indices[:-1])):
            raise ValueError("owned onset indices are not strictly chronological")
        if tuple(state_chunk.shape) != (len(indices), 2):
            raise ValueError("State owner chunk must have shape [O,2]")
        if tuple(mode_chunk.shape) != (len(indices), 3) or tuple(timing_chunk.shape) != (len(indices), 2):
            raise ValueError("Mode/Timing owner chunk shape mismatch")
        if state is None:
            device = state_chunk.device
            state = torch.empty((num_onsets, 2), dtype=state_chunk.dtype, device=device)
            mode = torch.empty((num_onsets, 3), dtype=mode_chunk.dtype, device=mode_chunk.device)
            timing = torch.empty((num_onsets, 2), dtype=timing_chunk.dtype, device=timing_chunk.device)
        if bool(torch.any(seen[indices] != 0)):
            raise AssertionError("duplicate owner logits in performance-global assembly")
        state[indices.to(state.device)] = state_chunk
        mode[indices.to(mode.device)] = mode_chunk
        timing[indices.to(timing.device)] = timing_chunk
        seen[indices] += 1
    if state is None or mode is None or timing is None or not bool(torch.all(seen == 1)):
        missing = torch.flatnonzero(seen == 0).tolist()
        duplicate = torch.flatnonzero(seen > 1).tolist()
        raise AssertionError(f"owner assembly is incomplete: missing={missing[:10]} duplicate={duplicate[:10]}")
    return PerformanceHeadLogits(state, mode[:-1], timing[:-1])


def decode_performance(logits: PerformanceHeadLogits) -> PerformanceDecode:
    raw = raw_head_decode(logits.state_logits, logits.mode_logits, logits.timing_predictions)
    viterbi = constrained_viterbi_decode(logits.state_logits, logits.mode_logits)
    return PerformanceDecode(raw, viterbi)


def collect_frozen_pt_piece_logits(
    model: Any, piece: FrozenPTPiece, device: torch.device, *, amp: bool = True,
) -> PerformanceHeadLogits:
    """Run the shared PT encoder over canonical frozen windows; no human pedal/state."""

    model.eval()
    chunks = []
    with torch.inference_mode():
        for window_index, (start, end) in enumerate(
            zip(piece.ownership.window_starts, piece.ownership.window_ends, strict=True)
        ):
            owned = piece.ownership.owned_onset_indices[window_index]
            local = piece.ownership.owned_local_representative_indices[window_index]
            if not len(owned):
                continue
            notes = piece.masked_tokens[start:end]
            if bool(torch.any(notes[:, 4:] != 1)):
                raise AssertionError("frozen PT pedal tokens leaked into Run B input")
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp and device.type == "cuda"):
                output = model(
                    input_ids=notes.reshape(1, -1).to(device),
                    token_attention_mask=torch.ones((1, notes.numel()), dtype=torch.long, device=device),
                    note_mask=torch.ones((1, len(notes)), dtype=torch.bool, device=device),
                    owned_representative_positions=torch.from_numpy(local).reshape(1, -1).long().to(device),
                    owned_onset_mask=torch.ones((1, len(local)), dtype=torch.bool, device=device),
                )
            tensors = (output.state_logits[0], output.mode_logits[0], output.timing_predictions[0])
            if not all(bool(torch.isfinite(value).all()) for value in tensors):
                raise FloatingPointError("non-finite frozen PT Run B logits")
            chunks.append((owned, *tensors))
    return assemble_performance_owner_logits(len(piece.onset_seconds), chunks)

