"""Free-running window inference compatible with canonical CC64 transplant."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from src.stage2_binary.dataset import make_masked_binary_sample
from src.stage2_binary.validation_evaluator import replace_pedal_tokens
from src.stage2_encoder_only.dataset import (
    NON_PEDAL_FEATURES,
    TOKENS_PER_NOTE,
    generate_window_starts,
)

from .model import BINARY_CLASSES, BinaryPedalEncoderDecoderModel


def infer_binary_encoder_decoder_pedals(
    model: BinaryPedalEncoderDecoderModel,
    stage1_token_ids: Sequence[int] | np.ndarray,
    *,
    device: torch.device,
    window_notes: int = 512,
    stride_notes: int = 256,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run independent-BOS windows and average free-running step logits."""

    tokens = np.asarray(stage1_token_ids, dtype=np.int64)
    if tokens.ndim == 1:
        if tokens.size == 0 or tokens.size % TOKENS_PER_NOTE:
            raise ValueError("Stage 1 IDs must contain complete eight-token notes")
        tokens = tokens.reshape(-1, TOKENS_PER_NOTE)
    if tokens.ndim != 2 or tokens.shape[1] != TOKENS_PER_NOTE or len(tokens) == 0:
        raise ValueError("Stage 1 IDs must have shape [N,8]")
    starts = generate_window_starts(len(tokens), window_notes, stride_notes)
    accumulated = torch.zeros(
        (len(tokens), 4, BINARY_CLASSES), dtype=torch.float32, device=device
    )
    contributions = torch.zeros(len(tokens), dtype=torch.float32, device=device)
    model.eval()
    with torch.inference_mode():
        for start in starts:
            end = min(start + window_notes, len(tokens))
            sample = make_masked_binary_sample(tokens, start, end)
            input_ids = sample["input_ids"].unsqueeze(0).to(device)
            token_attention_mask = torch.ones_like(input_ids)
            note_mask = sample["note_mask"].unsqueeze(0).to(device)
            output = model(
                input_ids=input_ids,
                token_attention_mask=token_attention_mask,
                note_mask=note_mask,
                decode_mode="greedy",
            )
            logits = output.logits[0, : end - start].float()
            if tuple(logits.shape) != (end - start, 4, BINARY_CLASSES):
                raise RuntimeError("binary encoder-decoder window logits shape mismatch")
            accumulated[start:end] += logits
            contributions[start:end] += 1
    if bool(torch.any(contributions == 0)):
        raise AssertionError("one or more notes received no window prediction")
    mean_logits = accumulated / contributions[:, None, None]
    bits = mean_logits.argmax(dim=-1).cpu().numpy().astype(np.int64)
    raw = bits * 127
    candidate = replace_pedal_tokens(tokens, raw)
    if not np.array_equal(
        candidate.reshape(-1, TOKENS_PER_NOTE)[:, :NON_PEDAL_FEATURES],
        tokens[:, :NON_PEDAL_FEATURES],
    ):
        raise AssertionError("binary encoder-decoder changed non-pedal tokens")
    return candidate, {
        "notes": len(tokens),
        "windows": len(starts),
        "window_notes": window_notes,
        "stride_notes": stride_notes,
        "decoder_sequence_order": "note-major P1,P2,P3,P4",
        "per_window_start": "independent BOS",
        "overlap_merge": "raw free-running generated-step logit average then argmax",
        "ground_truth_decoder_history_used": False,
        "deterministic_greedy": True,
        "non_pedal_tokens_preserved": True,
    }


def load_binary_encoder_decoder_checkpoint(
    checkpoint_path: str | Path,
    *,
    encoder_checkpoint: str | Path,
    device: torch.device,
) -> tuple[BinaryPedalEncoderDecoderModel, Mapping[str, Any]]:
    """Load the future scaffold checkpoint with strict state compatibility."""

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or "model_state" not in payload:
        raise TypeError("checkpoint must contain model_state and configuration")
    configuration = payload.get("configuration", {})
    model = BinaryPedalEncoderDecoderModel.from_pretrained_encoder(
        encoder_checkpoint,
        decoder_init_seed=int(configuration.get("decoder_init_seed", 42)),
        freeze_encoder=False,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    incompatible = model.load_state_dict(payload["model_state"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("binary encoder-decoder checkpoint state mismatch")
    return model.to(device).eval(), configuration
