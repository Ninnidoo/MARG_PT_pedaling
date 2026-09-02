"""Frozen reproducibility configuration for State-Anchored Transition v1."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .targets import MODE_NAMES


SEED = 42
HIDDEN_SIZE = 768
WINDOW_NOTES = 512
STRIDE_NOTES = 256
IGNORE_INDEX = -100
STATE_NAMES = ("OFF", "ON")
MODE_CLASS_COUNTS = (7_882_290, 894_618, 230_579)
_MODE_TOTAL = sum(MODE_CLASS_COUNTS)
MODE_FREQUENCIES = tuple(count / _MODE_TOTAL for count in MODE_CLASS_COUNTS)
MODE_RAW_WEIGHTS = tuple(1.0 / frequency**0.5 for frequency in MODE_FREQUENCIES)
_MODE_RAW_MEAN = sum(MODE_RAW_WEIGHTS) / len(MODE_RAW_WEIGHTS)
MODE_CLASS_WEIGHTS = tuple(weight / _MODE_RAW_MEAN for weight in MODE_RAW_WEIGHTS)


@dataclass(frozen=True)
class StateAnchoredModelConfig:
    architecture: str = "pretrained_pt_encoder_four_linear_heads"
    target_representation: str = "state_anchored_transition_v1_main_only"
    hidden_size: int = HIDDEN_SIZE
    state_classes: tuple[str, ...] = STATE_NAMES
    mode_classes: tuple[str, ...] = MODE_NAMES
    timing_beta: float = 0.1
    state_loss: str = "unweighted_cross_entropy"
    mode_loss: str = "train_frequency_inverse_sqrt_weighted_cross_entropy"
    mode_class_counts: tuple[int, ...] = MODE_CLASS_COUNTS
    mode_class_frequencies: tuple[float, ...] = MODE_FREQUENCIES
    mode_class_weights: tuple[float, ...] = MODE_CLASS_WEIGHTS
    mode_weight_normalization: str = "arithmetic mean across HOLD/CHANGE/RETURN equals 1"
    timing_loss: str = "masked_smooth_l1_mean_over_valid_targets"
    window_notes: int = WINDOW_NOTES
    stride_notes: int = STRIDE_NOTES
    seed: int = SEED
    pre_enabled: bool = False
    post_enabled: bool = False
    final_note_off_tail_enabled: bool = False
    recurrent_state_input: bool = False
    viterbi_lambda_state: float = 1.0
    viterbi_lambda_mode: float = 1.0
    timing_decode_policy: str = "clamp_[0,1];_RETURN_sort_preserving_pair_membership"

    def __post_init__(self) -> None:
        if self.hidden_size != HIDDEN_SIZE:
            raise ValueError("State-Anchored v1 freezes hidden_size=768")
        if self.timing_beta != 0.1:
            raise ValueError("State-Anchored v1 freezes timing beta=0.1")
        if self.seed != SEED:
            raise ValueError("State-Anchored v1 freezes seed=42")
        if self.viterbi_lambda_state != 1.0 or self.viterbi_lambda_mode != 1.0:
            raise ValueError("State-Anchored v1 freezes both Viterbi coefficients to 1")
        if any((self.pre_enabled, self.post_enabled, self.final_note_off_tail_enabled)):
            raise ValueError("State-Anchored v1 is MAIN-only")
        arithmetic_mean = sum(self.mode_class_weights) / len(self.mode_class_weights)
        if abs(arithmetic_mean - 1.0) > 1e-12:
            raise AssertionError("Mode weights do not have arithmetic mean one")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return output
