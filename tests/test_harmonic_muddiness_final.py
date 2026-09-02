from __future__ import annotations

import math

from scripts.harmonic_pure_hall_negative_core import (
    FINAL_ACCUMULATION_COEFFICIENT,
    aggregate_final_harmonic_metric,
)


def onset(raw: float, pair_count: int) -> dict[str, float | int]:
    return {"negative_hall_mass_n": raw, "N_pair_n": pair_count}


def test_all_empty_q_is_exact_zero():
    result = aggregate_final_harmonic_metric([onset(0.0, 0), onset(0.0, 0)])
    assert result == {
        "H_mean": 0.0,
        "A_acc": 0.0,
        "M_harm": 0.0,
        "total_onset_count": 2,
        "valid_onset_count": 0,
    }


def test_valid_h_ignores_empty_q_but_accumulation_includes_it():
    result = aggregate_final_harmonic_metric([onset(2.0, 2), onset(0.0, 0)])
    assert result["H_mean"] == 1.0
    assert result["valid_onset_count"] == 1
    assert result["total_onset_count"] == 2
    assert math.isclose(result["A_acc"], math.log(3.0) / 2.0, rel_tol=0.0, abs_tol=1e-15)


def test_final_combination_matches_frozen_additive_formula():
    result = aggregate_final_harmonic_metric([onset(2.0, 2), onset(4.0, 1)])
    expected_h = (1.0 + 4.0) / 2.0
    expected_a = (math.log(3.0) + math.log(5.0)) / 2.0
    assert math.isclose(result["H_mean"], expected_h, rel_tol=0.0, abs_tol=1e-15)
    assert math.isclose(result["A_acc"], expected_a, rel_tol=0.0, abs_tol=1e-15)
    assert math.isclose(result["M_harm"], expected_h + 0.05 * expected_a, rel_tol=0.0, abs_tol=1e-15)


def test_final_accumulation_coefficient_is_frozen_at_0p05():
    assert FINAL_ACCUMULATION_COEFFICIENT == 0.05
    result = aggregate_final_harmonic_metric([onset(1.0, 1)])
    assert math.isclose(
        result["M_harm"] - result["H_mean"],
        0.05 * result["A_acc"],
        abs_tol=1e-15,
    )
