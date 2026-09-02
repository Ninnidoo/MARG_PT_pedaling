# Human Validation Mapping Fix Report

**Verdict: PASS**

## Q1. `duration_seconds` AttributeError root cause

`StateAnchoredIntervalTarget` intentionally stores `left_seconds` and `right_seconds` but has no legacy `duration_seconds` property. The Human reference-event reconstruction reused the Run A `HumanIntervalPrimitive` interface and failed for 70 performances after inference.

## Q2. Conflicting legacy/helper interface

The conflicting code was Run A-style `interval.left_seconds + tau * interval.duration_seconds`, combined with `score_positions_for_performance()` and `trajectory_for_metric()`. Run B now derives duration as `right_seconds - left_seconds` and maps events directly to the same performance's distinct-onset indices.

## Q3. Is Nakamura/PT alignment needed for Human-input validation?

No. Prediction onset/interval `i` and target onset/interval `i` share the same canonical performance timeline. Full dry-run external alignment calls: **0**.

## Q4. Performance index 55

Index 55 is `Liszt/Mephisto_Waltz/Tysman07M.mid`, piece `piece_7195bbce81550519`, metadata **856**, with 9,835 onsets. It is the existing Frozen cross-performance alignment exclusion. The timeout was caused only by the unnecessary Human external-alignment call; direct Human mapping now includes it and passes.

## Q5. Human 71/71 path

PASS: prediction → owner assembly → global Viterbi → direct common-horizon mapping → aggregation completed for **71/71**, failures 0, Viterbi conflicts 0. Transition P/R/F1=0.604805/0.516324/0.557073.

## Q6. Frozen-PT regression

PASS on 2 representatives including the metadata-856 piece family: mapping/logits/Viterbi/MIDI/S1/non-pedal identity PASS; pedal leakage 0. Frozen 19/71/70 logic and Stage 1 provenance were not changed.

## Q7. Safe epoch-1 resume

PASS: `/workspace/project/analysis/stage2_state_anchored_transition_v1_full_v0/resume_last_train_state.pt` restored as training-complete epoch **1**, global step **516**, validation pending. Epoch-1 training reexecution must remain 0.

Focused tests: 54 direct-function tests PASS. ASAP test access: 0.
