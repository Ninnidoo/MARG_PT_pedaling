"""Binary Stage 2 encoder-only models and shared target pipeline."""

from .dataset import (
    BinaryManifestSubsetDataset,
    binary_stage2_collate_fn,
    binarize_raw_pedals,
    decode_joint_targets,
    encode_joint_targets,
)
from .model import IndependentBinaryPedalModel, JointBinaryPedalModel
from .training import build_binary_optimizer
from .validation_evaluator import (
    binary_logits_to_raw_pedals,
    infer_cached_binary_pedals,
    load_binary_stage2_checkpoint,
    official_pt_pedal_similarity,
    replace_pedal_tokens,
)

__all__ = [
    "BinaryManifestSubsetDataset",
    "IndependentBinaryPedalModel",
    "JointBinaryPedalModel",
    "binary_stage2_collate_fn",
    "binarize_raw_pedals",
    "build_binary_optimizer",
    "binary_logits_to_raw_pedals",
    "infer_cached_binary_pedals",
    "load_binary_stage2_checkpoint",
    "official_pt_pedal_similarity",
    "replace_pedal_tokens",
    "decode_joint_targets",
    "encode_joint_targets",
]
