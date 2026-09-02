from __future__ import annotations

import math
import unittest

import numpy as np

from src.stage2_event_model.prediction_decoder_v1 import (
    DECODER_ID,
    MainIntervalBoundary,
    MainIntervalTicks,
    PerformanceTimeline,
    PredictionDecodeError,
    decode_predictions_v1,
    enforce_main_tick_boundary,
    quantize_decoded_timeline_v1,
    strict_terminal_tick,
)


def logits(classes: list[int], width: int) -> np.ndarray:
    result = np.full((len(classes), width), -10.0)
    for index, target in enumerate(classes):
        result[index, target] = 10.0
    return result


def timeline(count: int = 1) -> PerformanceTimeline:
    return PerformanceTimeline(
        first_onset_time=0.0,
        latest_note_off_time=float(count),
        main_intervals=tuple(
            MainIntervalBoundary(i, float(i), float(i + 1), i == count - 1)
            for i in range(count)
        ),
    )


def decode(
    *, initial: int = 0, main_classes: list[list[int]] | None = None,
    main_timing: list[list[float]] | None = None,
    terminal_classes: list[int] | None = None,
    terminal_timing: list[float] | None = None,
):
    main_classes = main_classes or [[0] * 6]
    count = len(main_classes)
    main_timing = main_timing or [[0.0] * 6 for _ in range(count)]
    terminal_classes = terminal_classes or [0] * 4
    terminal_timing = terminal_timing or [0.0] * 4
    return decode_predictions_v1(
        initial_logits=logits([initial], 4)[0],
        main_event_logits=np.stack([logits(row, 5) for row in main_classes]),
        main_timing_predictions=np.asarray(main_timing),
        terminal_event_logits=logits(terminal_classes, 5),
        terminal_timing_predictions=np.asarray(terminal_timing),
        timeline=timeline(count),
    )


class PredictionDecoderV1Tests(unittest.TestCase):
    def test_decoder_id_is_frozen_sha(self) -> None:
        self.assertEqual(len(DECODER_ID), 64)

    def test_initial_argmax_representative_and_priority(self) -> None:
        result = decode(initial=2, main_classes=[[4, 0, 0, 0, 0, 0]])
        self.assertEqual(result.initial_state, 2)
        self.assertEqual(result.initial_event.cc64_value, 79)
        self.assertEqual(result.initial_event.time, result.active_slot_events[0].time)
        self.assertLess(result.initial_event.ordering_key, result.active_slot_events[0].ordering_key)
        tied = decode_predictions_v1(
            initial_logits=np.zeros(4), main_event_logits=logits([0] * 6, 5)[None],
            main_timing_predictions=np.zeros((1, 6)),
            terminal_event_logits=logits([0] * 4, 5), terminal_timing_predictions=np.zeros(4),
            timeline=timeline(),
        )
        self.assertEqual(tied.initial_state, 0)

    def test_main_first_none_terminates(self) -> None:
        result = decode(main_classes=[[3, 1, 0, 4, 2, 0]])
        self.assertEqual(result.diagnostics["main_active_slots"], 2)
        self.assertEqual(result.diagnostics["main_ignored_after_first_none"], 2)
        self.assertEqual([item.original_slot_index for item in result.active_slot_events], [0, 1])
        self.assertEqual(decode(main_classes=[[0, 4, 4, 4, 4, 4]]).diagnostics["main_active_slots"], 0)

    def test_main_pair_sort_clamp_and_equal_tau_order(self) -> None:
        result = decode(
            main_classes=[[3, 1, 4, 0, 0, 0]],
            main_timing=[[0.72, -0.2, 1.4, 0, 0, 0]],
        )
        self.assertEqual([item.destination_state for item in result.active_slot_events], [0, 2, 3])
        self.assertEqual([item.tau for item in result.active_slot_events], [0.0, 0.72, 1.0])
        self.assertEqual(result.diagnostics["main_tau_clamp_low"], 1)
        self.assertEqual(result.diagnostics["main_tau_clamp_high"], 1)
        equal = decode(main_classes=[[2, 4, 0, 0, 0, 0]],
                       main_timing=[[0.5, 0.5, 0, 0, 0, 0]])
        self.assertEqual(equal.active_slot_events[-1].original_slot_index, 1)

    def test_same_time_last_set_and_same_state_suppression(self) -> None:
        result = decode(initial=2, main_classes=[[1, 3, 0, 0, 0, 0]],
                        main_timing=[[0.5, 0.5, 0, 0, 0, 0]])
        self.assertEqual(result.diagnostics["main_same_time_collapsed"], 1)
        self.assertEqual(result.diagnostics["same_state_suppressed_main"], 1)
        self.assertEqual([event.source for event in result.events], ["INITIAL"])
        self.assertEqual(result.final_pedal_state, 2)

    def test_state_continuity_across_intervals(self) -> None:
        result = decode(
            initial=2,
            main_classes=[[1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [4, 0, 0, 0, 0, 0]],
        )
        self.assertEqual([item.destination_state for item in result.emitted_events], [0, 3])
        self.assertEqual(result.diagnostics["same_state_suppressed_main"], 1)
        self.assertEqual(result.final_pedal_state, 3)

    def test_main_interval_tick_boundaries(self) -> None:
        self.assertEqual(enforce_main_tick_boundary(20, left_tick=10, right_tick=20,
                                                    is_final_main_interval=False), 19)
        self.assertEqual(enforce_main_tick_boundary(20, left_tick=10, right_tick=20,
                                                    is_final_main_interval=True), 20)
        self.assertEqual(enforce_main_tick_boundary(2, left_tick=10, right_tick=20,
                                                    is_final_main_interval=False), 10)

    def test_terminal_prefix_gap_cumulative_and_no_sort(self) -> None:
        result = decode(
            initial=0,
            terminal_classes=[1, 3, 0, 4],
            terminal_timing=[math.log1p(0.3), math.log1p(0.2), 5.0, 0.0],
        )
        terminal = [item for item in result.active_slot_events if item.source == "TERMINAL"]
        self.assertEqual([item.original_slot_index for item in terminal], [0, 1])
        self.assertAlmostEqual(terminal[0].time, 1.3)
        self.assertAlmostEqual(terminal[1].time, 1.5)
        self.assertEqual(result.diagnostics["terminal_ignored_after_first_none"], 1)
        self.assertEqual(result.diagnostics["same_state_suppressed_terminal"], 1)
        self.assertEqual([item.original_slot_index for item in result.emitted_events], [1])

    def test_terminal_negative_z_clamp(self) -> None:
        result = decode(terminal_classes=[3, 0, 0, 0], terminal_timing=[-2.0, 0, 0, 0])
        terminal = [item for item in result.active_slot_events if item.source == "TERMINAL"]
        self.assertEqual(terminal[0].terminal_gap_seconds, 0.0)
        self.assertEqual(result.diagnostics["terminal_negative_z_clamp"], 1)

    def test_terminal_discrete_strictness_includes_noop_clock(self) -> None:
        result = decode(initial=0, terminal_classes=[1, 3, 0, 0], terminal_timing=[0, 0, 0, 0])
        quantized = quantize_decoded_timeline_v1(
            result, seconds_to_tick=lambda _: 100, first_onset_tick=0,
            latest_note_off_tick=100,
            main_intervals=[MainIntervalTicks(0, 0, 100, True)],
        )
        terminal = [item for item in quantized.events if item.source == "TERMINAL"]
        self.assertEqual([item.tick for item in terminal], [102])
        self.assertEqual(strict_terminal_tick(100, 100), 101)

    def test_quantized_same_tick_last_destination(self) -> None:
        result = decode(main_classes=[[2, 4, 0, 0, 0, 0]],
                        main_timing=[[0.2, 0.21, 0, 0, 0, 0]])
        quantized = quantize_decoded_timeline_v1(
            result, seconds_to_tick=lambda _: 5, first_onset_tick=0,
            latest_note_off_tick=10,
            main_intervals=[MainIntervalTicks(0, 0, 10, True)],
        )
        main = [item for item in quantized.events if item.source == "MAIN"]
        self.assertEqual(len(main), 1)
        self.assertEqual(main[0].destination_state, 3)
        self.assertEqual(quantized.same_tick_main_collapsed, 1)

    def test_nonfinite_logits_and_timing_fail(self) -> None:
        base = dict(
            initial_logits=logits([0], 4)[0],
            main_event_logits=logits([0] * 6, 5)[None],
            main_timing_predictions=np.zeros((1, 6)),
            terminal_event_logits=logits([0] * 4, 5),
            terminal_timing_predictions=np.zeros(4), timeline=timeline(),
        )
        for key, index, value in (
            ("initial_logits", (0,), np.nan),
            ("main_event_logits", (0, 0, 0), np.inf),
            ("main_timing_predictions", (0, 5), -np.inf),
            ("terminal_event_logits", (3, 2), np.nan),
            ("terminal_timing_predictions", (3,), np.inf),
        ):
            corrupted = {name: np.array(item, copy=True) if isinstance(item, np.ndarray) else item
                         for name, item in base.items()}
            corrupted[key][index] = value
            with self.subTest(key=key), self.assertRaises(PredictionDecodeError):
                decode_predictions_v1(**corrupted)

    def test_structure_is_deterministic(self) -> None:
        kwargs = dict(main_classes=[[3, 1, 0, 4, 0, 0]],
                      main_timing=[[0.7, 0.2, 0, 0, 0, 0]],
                      terminal_classes=[4, 0, 0, 0], terminal_timing=[0.4, 0, 0, 0])
        self.assertEqual(decode(**kwargs).to_dict(), decode(**kwargs).to_dict())


if __name__ == "__main__":
    unittest.main()
