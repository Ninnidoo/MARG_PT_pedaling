"""Training-ready configuration without starting full training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.stage2_binary.full_training import SharedBinaryWindowDataset
from src.stage2_binary.training import build_binary_optimizer

from .model import BinaryPedalEncoderDecoderModel


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = ROOT / "analysis/stage2_binary_encoder_decoder_canonical_v1"


def default_training_configuration() -> dict[str, Any]:
    """Freeze the architecture-only comparison settings for a future run."""

    return {
        "experiment_id": "stage2_binary_encoder_decoder_canonical_v1",
        "reference_encoder_only_experiment": str(
            ROOT / "analysis/stage2_binary_canonical_v1"
        ),
        "output_root": str(EXPERIMENT_ROOT / "train_v0"),
        "checkpoint_path": str(ROOT / "checkpoints/pianist_transformer"),
        "cache_root": str(
            ROOT / "analysis/stage2_binary_v0/train_setup_v0/shared_cache"
        ),
        "expected_cache_id": "85a79e9d10e955b10000f72f6bbc4a29dbd2cc5a4c8cfb1277054a8d45dcb877",
        "canonical_validation_manifest": str(
            ROOT
            / "analysis/stage2_binary_canonical_v1/canonical_validation_stage1_manifest.csv"
        ),
        "canonical_validation_baseline": str(
            ROOT
            / "analysis/stage2_binary_canonical_v1/canonical_validation_baseline.json"
        ),
        "asap_split": str(
            ROOT / "analysis/stage2_encoder_only_v0/asap_split.csv"
        ),
        "seed": 42,
        "decoder_init_seed": 42,
        "window_notes": 512,
        "stride_notes": 256,
        "effective_batch_size": 16,
        "micro_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "optimizer": "AdamW",
        "encoder_lr": 1e-5,
        "decoder_head_lr": 1e-4,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "amp_enabled": True,
        "max_epochs": 10,
        "early_stopping_patience": 3,
        "checkpoint_selection": "minimum free-running canonical strict validation JS",
        "teacher_forced_validation": "diagnostic only",
        "target": "unweighted binary Pedal1--4 cross entropy; threshold 64",
        "decoder_sequence": "note-major P1,P2,P3,P4",
        "training_history": "standard teacher forcing only",
        "scheduled_sampling_enabled": False,
        "canonical_validation_decode": "fixed-length greedy free-running argmax",
        "window_overlap_policy": (
            "each 512-note window starts independently from BOS; average raw "
            "free-running step logits over stride-256 overlap; argmax once"
        ),
        "test_access_before_checkpoint_lock": False,
        "full_training_started": False,
    }


def verify_shared_cache() -> dict[str, Any]:
    configuration = default_training_configuration()
    train = SharedBinaryWindowDataset(configuration["cache_root"], "train")
    validation = SharedBinaryWindowDataset(
        configuration["cache_root"], "validation"
    )
    if train.cache_id != configuration["expected_cache_id"]:
        raise RuntimeError("shared binary cache ID changed")
    expected = {
        "train_performances": 2062,
        "train_notes": 9_369_095,
        "train_windows": 35_573,
        "validation_performances": 71,
        "validation_notes": 283_928,
        "validation_windows": 1_078,
    }
    actual = {
        "train_performances": len(train.records),
        "train_notes": int(train.tokens.shape[0]),
        "train_windows": len(train),
        "validation_performances": len(validation.records),
        "validation_notes": int(validation.tokens.shape[0]),
        "validation_windows": len(validation),
    }
    if actual != expected:
        raise RuntimeError(f"shared cache inventory mismatch: {actual}")
    return {"cache_id": train.cache_id, **actual}


def build_training_optimizer(
    model: BinaryPedalEncoderDecoderModel,
):
    configuration = default_training_configuration()
    return build_binary_optimizer(
        model,
        encoder_lr=float(configuration["encoder_lr"]),
        head_lr=float(configuration["decoder_head_lr"]),
        weight_decay=float(configuration["weight_decay"]),
    )
