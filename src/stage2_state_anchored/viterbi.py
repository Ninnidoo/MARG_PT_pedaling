"""Two-state constrained Viterbi decoder for State-Anchored v1."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .decoding import CHANGE_ID, HOLD_ID, RETURN_ID, state_mode_conflicts


@dataclass(frozen=True)
class ViterbiDecodeResult:
    state_path: torch.Tensor
    mode_path: torch.Tensor
    score: torch.Tensor
    conflict_count: int


def constrained_viterbi_decode(state_logits: torch.Tensor, mode_logits: torch.Tensor, *, lambda_state: float = 1.0, lambda_mode: float = 1.0) -> ViterbiDecodeResult:
    """Return the globally best legal State and Mode paths in log space."""

    if lambda_state != 1.0 or lambda_mode != 1.0:
        raise ValueError("State-Anchored v1 freezes lambda_state=lambda_mode=1")
    if state_logits.ndim != 2 or state_logits.shape[-1] != 2 or len(state_logits) < 1:
        raise ValueError("state logits must have shape [M,2] with M>=1")
    expected_modes = max(0, len(state_logits) - 1)
    if mode_logits.ndim != 2 or tuple(mode_logits.shape) != (expected_modes, 3):
        raise ValueError("mode logits must have shape [M-1,3]")
    state_logp = F.log_softmax(state_logits.float(), dim=-1)
    mode_logp = F.log_softmax(mode_logits.float(), dim=-1)
    length = len(state_logits)
    if length == 1:
        state = torch.argmax(state_logp[0]).reshape(1)
        return ViterbiDecodeResult(state, torch.empty(0, dtype=torch.long, device=state.device), state_logp[0, state[0]], 0)
    dp = state_logp[0].clone()
    previous_states = torch.empty((length - 1, 2), dtype=torch.long, device=state_logits.device)
    chosen_modes = torch.empty((length - 1, 2), dtype=torch.long, device=state_logits.device)
    for index in range(length - 1):
        next_dp = torch.full_like(dp, -torch.inf)
        for next_state in (0, 1):
            best_score = None
            best_previous = 0
            best_mode = HOLD_ID
            for previous_state in (0, 1):
                if previous_state != next_state:
                    mode = CHANGE_ID
                else:
                    same_scores = mode_logp[index, [HOLD_ID, RETURN_ID]]
                    mode = (HOLD_ID, RETURN_ID)[int(torch.argmax(same_scores).item())]
                candidate = dp[previous_state] + state_logp[index + 1, next_state] + mode_logp[index, mode]
                if best_score is None or bool(candidate > best_score):
                    best_score, best_previous, best_mode = candidate, previous_state, mode
            next_dp[next_state] = best_score
            previous_states[index, next_state] = best_previous
            chosen_modes[index, next_state] = best_mode
        dp = next_dp
    final_state = int(torch.argmax(dp).item())
    states = torch.empty(length, dtype=torch.long, device=state_logits.device)
    modes = torch.empty(length - 1, dtype=torch.long, device=state_logits.device)
    states[-1] = final_state
    for index in range(length - 2, -1, -1):
        next_state = int(states[index + 1].item())
        modes[index] = chosen_modes[index, next_state]
        states[index] = previous_states[index, next_state]
    conflict = state_mode_conflicts(states, modes)
    if int(conflict["count"]) != 0:
        raise AssertionError("constrained decoder produced an illegal path")
    return ViterbiDecodeResult(states, modes, dp[final_state], 0)
