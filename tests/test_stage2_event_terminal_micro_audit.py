from __future__ import annotations

import math
import unittest

from src.stage2_event_tokenizer.terminal_micro_audit import (
    TERMINAL_CAPACITIES,
    TerminalEvent,
    absolute_delays,
    assert_terminal_final_state_preserved,
    inter_event_gaps,
    inverse_delay,
    retain_terminal_last_k,
    retention_is_monotonic,
    transform_delay,
)


def events(delays: tuple[float, ...]) -> tuple[TerminalEvent, ...]:
    return tuple(
        TerminalEvent(delay, index % 4, "OFF_TO_ON" if index % 2 else "ON_TO_OFF")
        for index, delay in enumerate(delays)
    )


class TerminalMicroAuditTests(unittest.TestCase):
    def test_zero_and_one_terminal_event(self) -> None:
        self.assertEqual(retain_terminal_last_k((), 1), ())
        one = events((0.25,))
        self.assertEqual(retain_terminal_last_k(one, 1), one)

    def test_last_k_exact_and_chronological(self) -> None:
        values = events((0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9))
        for capacity in TERMINAL_CAPACITIES:
            expected = values if len(values) <= capacity else values[-capacity:]
            self.assertEqual(retain_terminal_last_k(values, capacity), expected)

    def test_final_terminal_state_preservation(self) -> None:
        values = events(tuple(index / 10 for index in range(1, 11)))
        for capacity in TERMINAL_CAPACITIES:
            retained = retain_terminal_last_k(values, capacity)
            assert_terminal_final_state_preserved(values, retained)
            self.assertEqual(values[-1].destination_state, retained[-1].destination_state)

    def test_absolute_delay_exact(self) -> None:
        self.assertEqual(absolute_delays(events((0.1, 0.4, 1.2))), (0.1, 0.4, 1.2))

    def test_gap_anchor_and_relative_gaps(self) -> None:
        actual = inter_event_gaps(events((0.4, 0.7, 1.5)))
        for observed, expected in zip(actual, (0.4, 0.3, 0.8)):
            self.assertAlmostEqual(observed, expected, places=12)

    def test_compressed_first_gap_reanchors_to_noteoff(self) -> None:
        original = events((0.1, 0.2, 0.5, 0.9, 1.4, 2.0))
        retained = retain_terminal_last_k(original, 3)
        self.assertEqual(absolute_delays(retained), (0.9, 1.4, 2.0))
        gaps = inter_event_gaps(retained)
        self.assertAlmostEqual(gaps[0], 0.9, places=12)
        self.assertAlmostEqual(gaps[1], 0.5, places=12)
        self.assertAlmostEqual(gaps[2], 0.6, places=12)

    def test_transform_roundtrips(self) -> None:
        for transform in ("raw", "log1p", "bounded"):
            for value in (0.0, 0.05, 0.5, 2.0, 10.0, 26.704):
                self.assertAlmostEqual(
                    inverse_delay(transform_delay(value, transform), transform), value, places=10
                )

    def test_no_negative_delay_or_gap(self) -> None:
        with self.assertRaises(AssertionError):
            absolute_delays(events((-0.1,)))
        with self.assertRaises(AssertionError):
            inter_event_gaps(events((0.3, 0.2)))
        self.assertEqual(inter_event_gaps(events((0.3, 0.3))), (0.3, 0.0))

    def test_retention_monotonic(self) -> None:
        for count in range(15):
            values = events(tuple(float(index) for index in range(count)))
            self.assertTrue(retention_is_monotonic(values))


if __name__ == "__main__":
    unittest.main()
