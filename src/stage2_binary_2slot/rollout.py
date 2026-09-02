"""Online reconciliation and hard free-running state APIs."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import torch

from src.stage2_event_tokenizer.binary_2slot import (
    DOWN,
    OFF,
    ON,
    UP,
    BinaryTransition,
    reconcile_and_compress,
    state_after,
)

from .dataset import HumanIntervalPrimitive


def next_binary_state(state: int, count: int) -> int:
    if state not in {OFF, ON}:
        raise ValueError("state must be OFF or ON")
    if count not in {0, 1, 2}:
        raise ValueError("count must be 0, 1, or 2")
    return int(state) ^ (int(count) % 2)


@dataclass(frozen=True)
class OnlineTarget:
    count: int
    timing_targets: tuple[float, float]
    timing_mask: tuple[bool, bool]
    retained_transitions: tuple[BinaryTransition, ...]
    correction_required: bool
    correction_retained: bool
    retained_real_human_transitions: int
    raw_human_transitions: int
    human_end_state: int


def build_online_target(model_state: int, interval: HumanIntervalPrimitive) -> OnlineTarget:
    """Construct the frozen target from the current hard model state."""

    correction_required = int(model_state) != interval.human_start_state
    retained = reconcile_and_compress(
        int(model_state), interval.human_start_state, interval.human_transitions
    )
    count = len(retained)
    values = [float(event.tau) for event in retained]
    timing_targets = tuple((values + [0.0, 0.0])[:2])
    timing_mask = (count >= 1, count >= 2)
    expected_end = interval.human_end_state
    if state_after(int(model_state), retained) != expected_end:
        raise AssertionError("online reconciliation failed final-state invariant")
    return OnlineTarget(
        count=count,
        timing_targets=(float(timing_targets[0]), float(timing_targets[1])),
        timing_mask=timing_mask,
        retained_transitions=retained,
        correction_required=correction_required,
        correction_retained=any(event.synthetic for event in retained),
        retained_real_human_transitions=sum(not event.synthetic for event in retained),
        raw_human_transitions=len(interval.human_transitions),
        human_end_state=expected_end,
    )


@dataclass(frozen=True)
class DecodedTransition:
    direction: str
    tau: float


@dataclass(frozen=True)
class DecodedPrediction:
    count: int
    transitions: tuple[DecodedTransition, ...]
    next_state: int


def decode_prediction(
    current_state: int,
    count: int,
    timing_predictions: Sequence[float] | torch.Tensor,
) -> DecodedPrediction:
    """Clamp/sort timing and derive alternating directions from hard state."""

    if current_state not in {OFF, ON} or count not in {0, 1, 2}:
        raise ValueError("invalid current state or count")
    if isinstance(timing_predictions, torch.Tensor):
        values = timing_predictions.detach().reshape(-1).tolist()
    else:
        values = list(timing_predictions)
    if len(values) < 2:
        raise ValueError("two timing predictions are required")
    taus = [min(1.0, max(0.0, float(value))) for value in values[:count]]
    if count == 2:
        taus.sort()
    state = current_state
    decoded: list[DecodedTransition] = []
    for tau in taus:
        direction = DOWN if state == OFF else UP
        decoded.append(DecodedTransition(direction, tau))
        state = ON if state == OFF else OFF
    expected = next_binary_state(current_state, count)
    if state != expected:
        raise AssertionError("decoded transition parity disagrees with state update")
    return DecodedPrediction(count, tuple(decoded), expected)


class PerformanceStateStore:
    """Trainer-owned state with explicit PRE/MAIN/POST lifecycle checks."""

    def __init__(self) -> None:
        self._states: dict[str, int] = {}
        self._phase: dict[str, str] = {}
        self._last_main: dict[str, int] = {}

    def begin(self, performance_id: str) -> int:
        if performance_id in self._states:
            raise RuntimeError("performance has already begun")
        self._states[performance_id] = OFF
        self._phase[performance_id] = "NEEDS_PRE"
        self._last_main[performance_id] = -1
        return OFF

    def current(self, performance_id: str) -> int:
        if performance_id not in self._states:
            raise KeyError("performance state has not begun")
        return self._states[performance_id]

    def record_interval(self, performance_id: str, region: str, predicted_count: int, *, main_onset_index: int | None = None) -> int:
        state = self.current(performance_id)
        phase = self._phase[performance_id]
        if region == "PRE":
            if phase != "NEEDS_PRE" or main_onset_index is not None:
                raise RuntimeError("PRE must occur exactly once before MAIN")
            self._phase[performance_id] = "MAIN"
        elif region == "MAIN":
            if phase != "MAIN" or main_onset_index is None:
                raise RuntimeError("MAIN requires an active performance and onset index")
            expected = self._last_main[performance_id] + 1
            if int(main_onset_index) != expected:
                raise RuntimeError(f"MAIN onset must be strictly chronological; expected {expected}")
            self._last_main[performance_id] = int(main_onset_index)
        elif region == "POST":
            if phase != "MAIN" or main_onset_index is not None:
                raise RuntimeError("POST must occur once after MAIN")
            self._phase[performance_id] = "DONE"
        else:
            raise ValueError("region must be PRE, MAIN, or POST")
        self._states[performance_id] = next_binary_state(state, int(predicted_count))
        return self._states[performance_id]

    def finish(self, performance_id: str, *, expected_main_count: int) -> int:
        state = self.current(performance_id)
        if self._phase[performance_id] != "DONE":
            raise RuntimeError("performance ended without POST")
        if self._last_main[performance_id] + 1 != int(expected_main_count):
            raise RuntimeError("performance did not supervise every MAIN onset")
        del self._states[performance_id]
        del self._phase[performance_id]
        del self._last_main[performance_id]
        return state


@dataclass
class RolloutDiagnostics:
    target_counts: Counter[int] = field(default_factory=Counter)
    predicted_counts: Counter[int] = field(default_factory=Counter)
    region_counts: Counter[str] = field(default_factory=Counter)
    intervals: int = 0
    state_agreements: int = 0
    state_mismatches: int = 0
    corrections_required: int = 0
    raw_human_transitions: int = 0
    retained_real_human_transitions: int = 0
    invalid_or_nonfinite: int = 0

    def add(
        self,
        interval: HumanIntervalPrimitive,
        model_state: int,
        target: OnlineTarget,
        predicted_count: int,
        tensors: Iterable[torch.Tensor] = (),
    ) -> None:
        self.intervals += 1
        self.target_counts[target.count] += 1
        self.predicted_counts[int(predicted_count)] += 1
        self.region_counts[interval.region] += 1
        agreement = int(model_state) == interval.human_start_state
        self.state_agreements += int(agreement)
        self.state_mismatches += int(not agreement)
        self.corrections_required += int(target.correction_required)
        self.raw_human_transitions += target.raw_human_transitions
        self.retained_real_human_transitions += target.retained_real_human_transitions
        self.invalid_or_nonfinite += sum(not bool(torch.isfinite(value).all()) for value in tensors)

    def as_dict(self) -> dict[str, object]:
        denominator = max(1, self.intervals)
        transition_denominator = max(1, self.raw_human_transitions)
        return {
            "intervals": self.intervals,
            "target_N_counts": {str(key): self.target_counts[key] for key in range(3)},
            "predicted_N_counts": {str(key): self.predicted_counts[key] for key in range(3)},
            "correction_required_fraction": self.corrections_required / denominator,
            "interval_start_state_agreement": self.state_agreements / denominator,
            "state_mismatch_count": self.state_mismatches,
            "real_human_transition_retention": self.retained_real_human_transitions / transition_denominator,
            "region_counts": {key: self.region_counts[key] for key in ("PRE", "MAIN", "POST")},
            "invalid_or_nonfinite_count": self.invalid_or_nonfinite,
        }
