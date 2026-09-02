"""Canonical four-class Decoder-only Stage 2 Prefix-LM."""

from .model import (
    DecoderOnlyStage2Output,
    FourClassPedalDecoderOnlyModel,
    build_prefix_lm_attention_mask,
)

__all__ = [
    "DecoderOnlyStage2Output",
    "FourClassPedalDecoderOnlyModel",
    "build_prefix_lm_attention_mask",
]
