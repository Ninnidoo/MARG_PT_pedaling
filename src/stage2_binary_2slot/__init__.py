"""State-conditioned Binary 2-Slot Transition Model v0.

The package is intentionally separate from the frozen Stage 2 binary and
Custom Event v0 implementations.
"""

from .dataset import (
    Binary2SlotPerformanceDataset,
    HumanIntervalPrimitive,
    PerformanceMajorBatchSampler,
    PerformanceMajorSampler,
    build_boundary_input,
    binary_2slot_collate_fn,
    mask_human_pedal_features,
)
from .losses import Binary2SlotCriterion, Binary2SlotLossConfig
from .model import StateConditionedBinary2SlotModel
from .rollout import (
    PerformanceStateStore,
    build_online_target,
    decode_prediction,
    next_binary_state,
)
from .trainer import StatefulRolloutEngine

__all__ = [
    "Binary2SlotCriterion",
    "Binary2SlotLossConfig",
    "Binary2SlotPerformanceDataset",
    "HumanIntervalPrimitive",
    "PerformanceMajorSampler",
    "PerformanceMajorBatchSampler",
    "PerformanceStateStore",
    "StateConditionedBinary2SlotModel",
    "StatefulRolloutEngine",
    "binary_2slot_collate_fn",
    "build_boundary_input",
    "build_online_target",
    "mask_human_pedal_features",
    "decode_prediction",
    "next_binary_state",
]
