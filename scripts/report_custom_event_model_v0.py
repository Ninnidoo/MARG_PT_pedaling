#!/usr/bin/env python3
"""Generate the implementation-only Custom Event Model v0 report."""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "analysis/custom_event_model_v0"


def read(name: str) -> dict:
    return json.loads((OUTPUT / name).read_text(encoding="utf-8"))


def main() -> None:
    ownership = read("window_ownership_stats.json")
    smoke = read("forward_smoke.json")
    config = read("config.json")
    loss = read("loss_config.json")
    train = ownership["train"]
    validation = ownership["validation"]
    text = f"""# Custom Event Model v0 Implementation Report

## Scope and provenance

- Frozen tokenizer cache: `{ownership["cache_id"]}`
- Cache universe: train 2,062; validation 71; ASAP test access 0.
- Input: official PT Pitch/IOI/Velocity/Duration tokens; all pedal input tokens are MASK.
- Windowing: 512 notes / stride 256 using the stable Stage 2 tail-window generator.
- Pretrained loading: `{config["encoder_loading"]}`
- Checkpoint: `{config["pretrained_checkpoint"]}`
- No model checkpoint warm-start, optimizer, training, candidate MIDI, or validation inference.

## Ownership

| Split | Windows | Onsets | Owned | Zero owner | Duplicate owner | Split-chord groups |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | {train["total_windows"]} | {train["total_distinct_onset_groups"]} | {train["owned_onset_groups"]} | {train["zero_owner_count"]} | {train["duplicate_owner_count"]} | {train["groups_rejected_from_some_windows_due_split_chord"]} |
| Validation | {validation["total_windows"]} | {validation["total_distinct_onset_groups"]} | {validation["owned_onset_groups"]} | {validation["zero_owner_count"]} | {validation["duplicate_owner_count"]} | {validation["groups_rejected_from_some_windows_due_split_chord"]} |

- Train initial/terminal supervision: {train["initial_supervision_count"]} / {train["terminal_supervision_count"]}.
- Validation initial/terminal supervision: {validation["initial_supervision_count"]} / {validation["terminal_supervision_count"]}.
- Owner choice: full chord required; maximum representative margin; exact ties choose smaller window start.
- Owned-onsets/window train mean/p50/p95: {train["owned_onset_count_per_window"]["mean"]:.3f} / {train["owned_onset_count_per_window"]["p50"]:.1f} / {train["owned_onset_count_per_window"]["p95"]:.1f}.
- Owner margin train mean/p50/p95: {train["owner_context_margin"]["mean"]:.3f} / {train["owner_context_margin"]["p50"]:.1f} / {train["owner_context_margin"]["p95"]:.1f}.

## Architecture

- PT Encoder-only hidden size: {smoke["hidden_size"]}.
- Initial: one `Linear(768,4)`.
- Main: six independent `Linear(768,5)` event heads and six independent `Linear(768,1)` timing heads.
- Terminal: four independent `Linear(768,5)` event heads and four independent `Linear(768,1)` timing heads.
- Terminal uses the cached final-onset last-note hidden state.
- Slots are parallel and non-autoregressive.

| Parameters | Count |
| --- | ---: |
| Encoder | {smoke["encoder_parameter_count"]} |
| Initial head | {smoke["head_parameter_counts"]["initial"]} |
| Main event heads | {smoke["head_parameter_counts"]["main_event"]} |
| Main timing heads | {smoke["head_parameter_counts"]["main_timing"]} |
| Terminal event heads | {smoke["head_parameter_counts"]["terminal_event"]} |
| Terminal timing heads | {smoke["head_parameter_counts"]["terminal_timing"]} |
| All heads | {smoke["prediction_head_parameter_count"]} |
| Total | {smoke["total_parameter_count"]} |

## Objective

- Initial: unweighted CE, first-onset owner only.
- Main event: frozen train-frequency slot-specific weighted CE, macro-average over six slots.
- Main timing: Smooth L1 beta={loss["beta_main"]}, only `event != NONE AND timing_valid`, macro-average over six slots.
- Terminal event: frozen train-frequency slot-specific weighted CE, macro-average over four slots, terminal owner only.
- Terminal timing: Smooth L1 beta={loss["beta_terminal"]}, only `event != NONE AND timing_valid`, macro-average over four slots.
- Total is the positive weighted sum of all five components; all lambdas are 1.0.
- Empty timing slots return a differentiable finite zero.
- Weight SHA256: `{smoke["event_weight_provenance"]["sha256"]}`.
- Null zero-frequency terminal weights are stored as zero; canonical train has no corresponding targets.

## Real pretrained forward/backward smoke

- Performance: `{smoke["performance_path"]}`
- Notes / owned onsets: {smoke["notes"]} / {smoke["owned_onsets"]}
- Losses: init {smoke["losses"]["initial_ce"]:.6f}, main event {smoke["losses"]["main_event_ce"]:.6f}, main time {smoke["losses"]["main_timing_huber"]:.6f}, terminal event {smoke["losses"]["terminal_event_ce"]:.6f}, terminal time {smoke["losses"]["terminal_timing_huber"]:.6f}, total {smoke["losses"]["total_loss"]:.6f}.
- All logits, scalar predictions, and losses finite: yes.
- Encoder and every head gradient group finite and non-zero: yes.
- GPU: {smoke["cuda_device"]}; peak allocated {smoke["cuda_max_memory_allocated_bytes"]} bytes.
- Optimizer created: no. Optimizer steps: 0. Training steps: 0.

## Frozen boundary

Tokenizer/cache files were read-only. Window ownership does not alter global interval targets. No tiny overfit, full training, model validation inference, ASAP test, or Repedal evaluation was run.
"""
    path = OUTPUT / "CUSTOM_EVENT_MODEL_V0_IMPLEMENTATION_REPORT.md"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)
    print(path)


if __name__ == "__main__":
    main()
