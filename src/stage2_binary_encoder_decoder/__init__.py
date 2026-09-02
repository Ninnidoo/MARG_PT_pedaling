"""Binary Pedal1--4 encoder-decoder scaffold for canonical Stage 2."""

from .inference import infer_binary_encoder_decoder_pedals
from .model import (
    BINARY_CLASSES,
    BOS_ID,
    DECODER_VOCAB_SIZE,
    PAD_ID,
    BinaryPedalEncoderDecoderModel,
    flatten_binary_pedal_targets,
    shift_right_binary_targets,
    unflatten_binary_sequence,
)

__all__ = [
    "BINARY_CLASSES",
    "BOS_ID",
    "DECODER_VOCAB_SIZE",
    "PAD_ID",
    "BinaryPedalEncoderDecoderModel",
    "flatten_binary_pedal_targets",
    "infer_binary_encoder_decoder_pedals",
    "shift_right_binary_targets",
    "unflatten_binary_sequence",
]
