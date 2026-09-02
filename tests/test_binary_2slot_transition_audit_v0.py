from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage2_event_tokenizer.binary_2slot import (
    DOWN,
    OFF,
    ON,
    UP,
    BinaryTransition,
    canonical_inverse_sqrt_weights,
    compress_binary_transitions,
    correction_transition,
    interval_region,
    reconcile_and_compress,
    state_after,
)


def transitions(start: int, count: int) -> tuple[BinaryTransition, ...]:
    state = start
    result = []
    for index in range(count):
        result.append(BinaryTransition(DOWN if state == OFF else UP, (index + 1) / (count + 1)))
        state = 1 - state
    return tuple(result)


def test_base_parity_cases() -> None:
    for start, count, expected_end, expected_n in (
        (OFF, 0, OFF, 0),
        (OFF, 1, ON, 1),
        (OFF, 2, OFF, 2),
        (ON, 1, OFF, 1),
        (ON, 2, ON, 2),
        (OFF, 3, ON, 1),
        (OFF, 4, OFF, 2),
        (OFF, 5, ON, 1),
        (OFF, 6, OFF, 2),
    ):
        raw = transitions(start, count)
        retained = compress_binary_transitions(raw)
        assert len(retained) == expected_n
        assert state_after(start, raw) == expected_end
        assert state_after(start, retained) == expected_end
        if count == 5:
            assert retained == raw[-1:]
        if count == 6:
            assert retained == raw[-2:]


def test_mismatch_correction_recovers_human_end() -> None:
    for human_start, count in ((ON, 0), (ON, 1), (ON, 2), (ON, 3), (OFF, 4)):
        model_start = 1 - human_start
        raw = transitions(human_start, count)
        retained = reconcile_and_compress(model_start, human_start, raw)
        assert state_after(model_start, retained) == state_after(human_start, raw)


def test_pre_already_on_has_synthetic_down_at_zero() -> None:
    retained = reconcile_and_compress(OFF, ON, ())
    assert retained == (BinaryTransition(DOWN, 0.0, synthetic=True),)


def test_frozen_boundaries() -> None:
    epsilon = 1e-9
    assert interval_region(9.0, 10.0, 20.0) == "PRE"
    assert interval_region(9.0 - epsilon, 10.0, 20.0) == "OUTSIDE"
    assert interval_region(10.0 - epsilon, 10.0, 20.0) == "PRE"
    assert interval_region(10.0, 10.0, 20.0) == "MAIN"
    assert interval_region(20.0 - epsilon, 10.0, 20.0) == "MAIN"
    assert interval_region(20.0, 10.0, 20.0) == "POST"
    assert interval_region(21.0, 10.0, 20.0) == "POST"
    assert interval_region(21.000001, 10.0, 20.0) == "OUTSIDE"


def test_mismatch_k2_discards_correction_under_frozen_policy() -> None:
    human = transitions(ON, 2)
    retained = reconcile_and_compress(OFF, ON, human)
    assert all(not event.synthetic for event in retained)
    assert retained == human[-1:]
    assert state_after(OFF, retained) == state_after(ON, human)


def expect_value_error(callback) -> None:
    try:
        callback()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_direction_state_validation() -> None:
    expect_value_error(lambda: state_after(OFF, (BinaryTransition(UP, 0.5),)))
    expect_value_error(lambda: state_after(ON, (BinaryTransition(DOWN, 0.5),)))
    expect_value_error(lambda: state_after(OFF, (BinaryTransition(DOWN, 0.2), BinaryTransition(DOWN, 0.4))))


def test_correction_and_human_transition_can_share_tau_zero() -> None:
    human = (BinaryTransition(UP, 0.0),)
    retained = reconcile_and_compress(OFF, ON, human)
    assert len(retained) == 2
    assert tuple(event.tau for event in retained) == (0.0, 0.0)
    assert tuple(event.synthetic for event in retained) == (True, False)
    assert state_after(OFF, retained) == state_after(ON, human) == OFF


def test_canonical_inverse_sqrt_weight_normalization() -> None:
    frequencies, raw, weights = canonical_inverse_sqrt_weights((7885843, 896884, 230946))
    assert abs(sum(frequencies) - 1.0) <= 1e-15
    assert len(raw) == len(weights) == 3
    assert abs(sum(frequency * weight for frequency, weight in zip(frequencies, weights)) - 1.0) <= 1e-12
    expect_value_error(lambda: canonical_inverse_sqrt_weights((1, 0, 2)))


def run_direct_test_suite() -> dict[str, str]:
    results: dict[str, str] = {}
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            value()
            results[name] = "passed"
    return results


if __name__ == "__main__":
    print(run_direct_test_suite())
