"""Trainer-facing hard-state rollout core.

This module performs no optimizer step and owns no DataLoader worker state. A
future training driver may keep one engine across micro-batches and gradient
accumulation boundaries; state changes only at represented intervals and is
removed only after POST.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .dataset import HumanIntervalPrimitive
from .model import Binary2SlotHeadOutput, StateConditionedBinary2SlotModel
from .rollout import OnlineTarget, PerformanceStateStore, RolloutDiagnostics, build_online_target


@dataclass
class RolloutChunk:
    count_logits: torch.Tensor
    timing_predictions: torch.Tensor
    target_counts: torch.Tensor
    timing_targets: torch.Tensor
    timing_mask: torch.Tensor
    regions: tuple[str, ...]
    online_targets: tuple[OnlineTarget, ...]
    predicted_counts: tuple[int, ...]


class StatefulRolloutEngine:
    """External performance state store plus sequential shared-head execution."""

    def __init__(self, model: StateConditionedBinary2SlotModel) -> None:
        self.model = model
        self.states = PerformanceStateStore()
        self.diagnostics = RolloutDiagnostics()

    def begin_performance(
        self, performance_id: str, pre_hidden: torch.Tensor, pre_interval: HumanIntervalPrimitive
    ) -> RolloutChunk:
        if pre_interval.region != "PRE":
            raise ValueError("begin_performance requires PRE")
        self.states.begin(performance_id)
        return self._rollout(performance_id, pre_hidden.reshape(1, -1), (pre_interval,))

    def process_main_chunk(
        self,
        performance_id: str,
        hidden_states: torch.Tensor,
        intervals: Sequence[HumanIntervalPrimitive],
    ) -> RolloutChunk:
        if any(interval.region != "MAIN" for interval in intervals):
            raise ValueError("process_main_chunk accepts only MAIN intervals")
        return self._rollout(performance_id, hidden_states, tuple(intervals))

    def end_performance(
        self,
        performance_id: str,
        post_hidden: torch.Tensor,
        post_interval: HumanIntervalPrimitive,
        *,
        expected_main_count: int,
    ) -> RolloutChunk:
        if post_interval.region != "POST":
            raise ValueError("end_performance requires POST")
        result = self._rollout(performance_id, post_hidden.reshape(1, -1), (post_interval,))
        self.states.finish(performance_id, expected_main_count=expected_main_count)
        return result

    def _rollout(
        self,
        performance_id: str,
        hidden_states: torch.Tensor,
        intervals: Sequence[HumanIntervalPrimitive],
    ) -> RolloutChunk:
        if hidden_states.ndim != 2 or hidden_states.shape[0] != len(intervals):
            raise ValueError("hidden states must have shape [I,H] for the supplied intervals")
        if not intervals:
            empty_logits = hidden_states.new_empty((0, 3))
            empty_timing = hidden_states.new_empty((0, 2))
            return RolloutChunk(
                empty_logits,
                empty_timing,
                torch.empty(0, dtype=torch.long, device=hidden_states.device),
                empty_timing,
                torch.empty((0, 2), dtype=torch.bool, device=hidden_states.device),
                (), (), (),
            )
        head_outputs: list[Binary2SlotHeadOutput] = []
        targets: list[OnlineTarget] = []
        predicted: list[int] = []
        for hidden, interval in zip(hidden_states, intervals):
            state = self.states.current(performance_id)
            target = build_online_target(state, interval)
            state_tensor = torch.tensor([state], dtype=torch.long, device=hidden.device)
            output = self.model.condition_and_predict(hidden.unsqueeze(0), state_tensor)
            predicted_count = int(torch.argmax(output.count_logits[0]).detach().item())
            self.diagnostics.add(
                interval, state, target, predicted_count,
                tensors=(output.count_logits, output.timing_predictions),
            )
            self.states.record_interval(
                performance_id, interval.region, predicted_count,
                main_onset_index=interval.main_onset_index,
            )
            head_outputs.append(output)
            targets.append(target)
            predicted.append(predicted_count)
        device = hidden_states.device
        return RolloutChunk(
            count_logits=torch.cat([output.count_logits for output in head_outputs], dim=0),
            timing_predictions=torch.cat([output.timing_predictions for output in head_outputs], dim=0),
            target_counts=torch.tensor([target.count for target in targets], dtype=torch.long, device=device),
            timing_targets=torch.tensor([target.timing_targets for target in targets], dtype=hidden_states.dtype, device=device),
            timing_mask=torch.tensor([target.timing_mask for target in targets], dtype=torch.bool, device=device),
            regions=tuple(interval.region for interval in intervals),
            online_targets=tuple(targets),
            predicted_counts=tuple(predicted),
        )
