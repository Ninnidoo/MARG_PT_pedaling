# State-Anchored Transition v1 Model Implementation

**Verdict: PASS**

## Reused infrastructure

- Frozen State-Anchored MAIN-only target builder and audit representation (unchanged).
- Canonical Custom Event cache manifest and train/validation split only.
- Binary 2-Slot raw MIDI/timeline adapter, official PT Pitch/IOI/Velocity/Duration construction, and human-pedal masking.
- Existing 512-note/stride-256 maximum-margin onset ownership and canonical last-note representative mapping.
- Official `PianoT5Gemma.from_pretrained(...).get_encoder()` loading path and note-level encoder output.

## Dataset and ownership

State is attached to every uniquely owned onset. Mode/Timing are attached to the same owner for onset index `i < M-1`; the final onset is State-only, so no interval crosses a performance boundary.

| Split | Performances | State targets | Mode intervals | State zero/duplicate | Mode zero/duplicate | Boundary leakage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 2,062 | 9,009,549 | 9,007,487 | 0/0 | 0/0 | 0 |
| Validation | 71 | 272,927 | 272,856 | 0/0 | 0/0 | 0 |

## Architecture and parameter counts

The pretrained PT encoder emits `h_i` with hidden size 768. Four direct linear heads predict State (2), Mode (3), tau1 (1), and tau2 (1). Predicted or human State is never fed into another onset's network input.

- Encoder parameters: 103,271,424
- State head: 1,538
- Mode head: 2,307
- Timing heads: 769 + 769
- All custom heads: 5,383
- Total parameters: 103,276,807

## Loss and exact Mode weights

`L = State CE + weighted Mode CE + masked Smooth-L1(beta=0.1)`. State CE is unweighted. Timing is averaged over valid scalar targets only.

- HOLD: 0.757827986646530
- CHANGE: 2.249458722476316
- RETURN: 4.430849191607236
- Normalization: `sum_c train_frequency_c * weight_c = 1`; validation is not used.

## Raw heads and constrained Viterbi

Raw helpers provide State/Mode argmax, accuracy, macro F1, class precision/recall/F1, and both illegal conflict types. Viterbi maximizes State plus Mode log-probability with fixed lambdas 1/1. Different-state edges force CHANGE; same-state edges select the higher of HOLD/RETURN and preserve that Mode in backtracking.

Timing predictions are clamped to `[0,1]`. CHANGE uses head 1 only; RETURN retains both head identities as a pair and sorts their values chronologically.

## Focused tests

- Result: PASS (27 cases across 4 focused files).
- Covered tensor shapes, padding masks, HOLD/CHANGE/RETURN timing gradients, finite State/weighted-Mode/total losses, finite head gradients, all requested Viterbi legal transitions, raw conflict recovery, global optimum, determinism, decoded conflict zero, ownership exactly-once, and boundary isolation.

## Canonical CPU sanity

- Result: PASS
- Device: CPU; `torch.cuda.is_available()=False` with CUDA hidden.
- Sample: train performance index 0, window 0, 512 notes, 374 owned onsets.
- Losses: State=0.397486, Mode=1.390025, Timing=0.487410, Total=2.274922; all finite.
- Raw conflict: 47; constrained decoded conflict: 0.
- Official encoder backward was not run on CPU; custom-head finite backward is covered by focused tests.

## Safety and split gates

- PRE/POST/final-note-off tail: disabled.
- ASAP test access: 0.
- GPU use: 0.
- Tiny overfit/full training: not run.
- RUN A code/process/tmux/output: not modified by implementation or sanity commands; external liveness is checked separately.

## PASS / FAIL

PASS: dataset integration, model, loss, raw decoding, constrained Viterbi, focused tests, full ownership diagnostics, and canonical CPU forward/loss/decode sanity all passed.
