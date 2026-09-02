"""Terminal-only structural diagnostics for the immutable event tokenizer v0."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

TERMINAL_CAPACITIES = (1, 2, 3, 4, 6, 8)
TRANSFORMS = ("raw", "log1p", "bounded")
QUANTILES = (
    ("p01", 0.01),
    ("p05", 0.05),
    ("p25", 0.25),
    ("p50", 0.50),
    ("p75", 0.75),
    ("p90", 0.90),
    ("p95", 0.95),
    ("p97p5", 0.975),
    ("p99", 0.99),
    ("p99p5", 0.995),
)


@dataclass(frozen=True)
class TerminalEvent:
    delay_seconds: float
    destination_state: int
    transition_type: str


def retain_terminal_last_k(
    events: Sequence[TerminalEvent], capacity: int | None
) -> tuple[TerminalEvent, ...]:
    values = tuple(events)
    if capacity is None:
        return values
    if capacity <= 0:
        raise ValueError("terminal capacity must be positive")
    return values if len(values) <= capacity else values[-capacity:]


def assert_terminal_final_state_preserved(
    events: Sequence[TerminalEvent], retained: Sequence[TerminalEvent]
) -> None:
    if events and (not retained or retained[-1].destination_state != events[-1].destination_state):
        raise AssertionError("terminal compression changed final destination state")


def absolute_delays(events: Sequence[TerminalEvent]) -> tuple[float, ...]:
    values = tuple(float(event.delay_seconds) for event in events)
    if any(value < -1e-12 for value in values):
        raise AssertionError("negative terminal delay")
    return tuple(max(0.0, value) for value in values)


def inter_event_gaps(events: Sequence[TerminalEvent]) -> tuple[float, ...]:
    """Use note-off for retained slot 1, then the preceding retained event."""

    delays = absolute_delays(events)
    if not delays:
        return ()
    gaps = (delays[0],) + tuple(delays[index] - delays[index - 1] for index in range(1, len(delays)))
    if any(value < -1e-12 for value in gaps):
        raise AssertionError("negative terminal inter-event gap")
    return tuple(max(0.0, value) for value in gaps)


def transform_delay(value: float, transform: str) -> float:
    value = float(value)
    if value < 0:
        raise ValueError("delay must be nonnegative")
    if transform == "raw":
        return value
    if transform == "log1p":
        return math.log1p(value)
    if transform == "bounded":
        return value / (value + 1.0)
    raise KeyError(transform)


def inverse_delay(value: float, transform: str) -> float:
    value = float(value)
    if transform == "raw":
        return value
    if transform == "log1p":
        return math.expm1(value)
    if transform == "bounded":
        if value >= 1.0:
            return math.inf
        return value / (1.0 - value)
    raise KeyError(transform)


def numeric_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(tuple(float(value) for value in values), dtype=np.float64)
    if not len(array):
        return {"count": 0}
    result: dict[str, float | int | None] = {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }
    result.update({name: float(np.quantile(array, level)) for name, level in QUANTILES})
    result["dynamic_range"] = float(array.max() - array.min())
    median = float(result["p50"])
    result["p99_over_median"] = float(result["p99"]) / median if median > 0 else None
    return result


def transformed_summary(values: Iterable[float], transform: str) -> dict[str, float | int | None]:
    raw = tuple(float(value) for value in values)
    result = numeric_summary(transform_delay(value, transform) for value in raw)
    if transform == "bounded" and raw:
        transformed = np.asarray([transform_delay(value, transform) for value in raw])
        for threshold in (0.90, 0.95, 0.98, 0.99):
            result[f"fraction_z_ge_{str(threshold).replace('.', 'p')}"] = float(
                np.mean(transformed >= threshold)
            )
    return result


def perturbation_rows(
    points: Sequence[float] = (0.05, 0.10, 0.25, 0.50, 1.0, 2.0, 5.0, 10.0),
    epsilons: Sequence[float] = (0.01, 0.05),
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for transform in TRANSFORMS:
        for source in points:
            target = transform_delay(source, transform)
            for epsilon in epsilons:
                for sign in (-1.0, 1.0):
                    perturbation = sign * epsilon
                    decoded = inverse_delay(target + perturbation, transform)
                    rows.append(
                        {
                            "transform": transform,
                            "source_seconds": float(source),
                            "epsilon": float(perturbation),
                            "transformed_target": float(target),
                            "decoded_seconds": float(decoded),
                            "signed_seconds_error": float(decoded - source),
                            "absolute_seconds_error": float(abs(decoded - source)),
                        }
                    )
    return rows


def retention_is_monotonic(events: Sequence[TerminalEvent]) -> bool:
    retained = [len(retain_terminal_last_k(events, capacity)) for capacity in TERMINAL_CAPACITIES]
    return retained == sorted(retained)
