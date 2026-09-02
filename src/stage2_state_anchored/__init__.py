"""State-Anchored Transition targets for Stage 2 pedal modeling."""

from .targets import (
    CHANGE,
    HOLD,
    MODE_NAMES,
    RETURN,
    StateAnchoredIntervalTarget,
    StateAnchoredPerformanceTargets,
    build_interval_target,
    build_performance_targets,
)
from .inference import (
    PerformanceDecode,
    PerformanceHeadLogits,
    assemble_performance_owner_logits,
    decode_performance,
)

__all__ = [
    "CHANGE",
    "HOLD",
    "MODE_NAMES",
    "RETURN",
    "StateAnchoredIntervalTarget",
    "StateAnchoredPerformanceTargets",
    "build_interval_target",
    "build_performance_targets",
    "PerformanceDecode",
    "PerformanceHeadLogits",
    "assemble_performance_owner_logits",
    "decode_performance",
]
