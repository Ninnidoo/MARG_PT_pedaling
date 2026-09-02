"""Frozen binary two-slot transition primitives used by the v0 audit.

This module deliberately contains no model, loss, or training code.  MIDI
parsing remains owned by :mod:`src.stage2_event_tokenizer.tokenizer`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence


OFF = 0
ON = 1
DOWN = "DOWN"
UP = "UP"


@dataclass(frozen=True)
class BinaryTransition:
    direction: str
    tau: float
    source_tick: int | None = None
    synthetic: bool = False

    def __post_init__(self) -> None:
        if self.direction not in {DOWN, UP}:
            raise ValueError(f"invalid binary transition: {self.direction}")
        if not 0.0 <= float(self.tau) <= 1.0:
            raise ValueError(f"tau outside [0,1]: {self.tau}")


def state_after(state: int, transitions: Iterable[BinaryTransition]) -> int:
    """Apply direction-aware binary transitions and return the final state."""

    current = int(state)
    if current not in {OFF, ON}:
        raise ValueError(f"invalid binary state: {state}")
    for transition in transitions:
        expected = DOWN if current == OFF else UP
        if transition.direction != expected:
            raise ValueError(
                f"transition {transition.direction} cannot follow state {current}; "
                f"expected {expected}"
            )
        current = ON if current == OFF else OFF
    return current


def compress_binary_transitions(
    transitions: Sequence[BinaryTransition],
) -> tuple[BinaryTransition, ...]:
    """Apply the frozen max-2 parity-preserving last-retention rule."""

    values = tuple(transitions)
    count = len(values)
    if count <= 2:
        return values
    if count % 2:
        return values[-1:]
    return values[-2:]


def correction_transition(model_state: int) -> BinaryTransition:
    """Return the tau-zero transition that toggles the model state."""

    if model_state not in {OFF, ON}:
        raise ValueError(f"invalid binary state: {model_state}")
    return BinaryTransition(DOWN if model_state == OFF else UP, 0.0, synthetic=True)


def reconcile_and_compress(
    model_state: int,
    human_state: int,
    human_transitions: Sequence[BinaryTransition],
) -> tuple[BinaryTransition, ...]:
    """Prepend the frozen mismatch correction, then apply max-2 compression."""

    if model_state not in {OFF, ON} or human_state not in {OFF, ON}:
        raise ValueError("model and human states must be OFF or ON")
    sequence = tuple(human_transitions)
    # Validate the human sequence before constructing a counterfactual target.
    state_after(human_state, sequence)
    if model_state != human_state:
        sequence = (correction_transition(model_state),) + sequence
    retained = compress_binary_transitions(sequence)
    if state_after(model_state, retained) != state_after(human_state, human_transitions):
        raise AssertionError("reconciled target does not recover the human interval-end state")
    return retained


def interval_region(time_seconds: float, first_onset: float, note_end: float) -> str:
    """Assign a time to the frozen PRE/MAIN/POST/outside regions."""

    value = float(time_seconds)
    if first_onset - 1.0 <= value < first_onset:
        return "PRE"
    if first_onset <= value < note_end:
        return "MAIN"
    if note_end <= value <= note_end + 1.0:
        return "POST"
    return "OUTSIDE"


def canonical_inverse_sqrt_weights(
    counts: Sequence[int | float],
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """Return frequencies, raw weights, and canonical weighted-mean-one weights.

    This Binary 2-Slot-specific helper intentionally does not alter the legacy
    ``inverse_sqrt_mean_one`` helper used by earlier experiments.
    """

    values = tuple(float(value) for value in counts)
    if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("canonical inverse-sqrt weights require finite positive counts")
    total = sum(values)
    frequencies = tuple(value / total for value in values)
    raw = tuple(1.0 / math.sqrt(value) for value in frequencies)
    denominator = sum(frequency * weight for frequency, weight in zip(frequencies, raw))
    normalized = tuple(weight / denominator for weight in raw)
    weighted_mean = sum(
        frequency * weight for frequency, weight in zip(frequencies, normalized)
    )
    if not math.isclose(weighted_mean, 1.0, rel_tol=1e-12, abs_tol=1e-12):
        raise AssertionError("canonical class weights do not have frequency-weighted mean one")
    return frequencies, raw, normalized
