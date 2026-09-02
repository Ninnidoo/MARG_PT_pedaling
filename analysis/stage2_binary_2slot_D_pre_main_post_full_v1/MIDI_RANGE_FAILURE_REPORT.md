# Binary 2-Slot Frozen-PT MIDI Range Failure Report

## Outcome

| Item | Result |
|---|---|
| Root cause | PRE conceptual time before MIDI origin was passed directly to the strict MIDI writer |
| Failure class | Negative absolute tick; not CC64 value, upper tick, overflow, or chronology |
| Exact piece | 6/19, `piece_b45364ab6817194a` |
| Minimal fix | Project pre-origin PRE trajectory to one tick-0 absolute-state anchor; do not clamp individual timestamps |
| Focused tests | PASS, 22/22 |
| Epoch-1 saved-model renderer audit | PASS, 19/19 |
| Non-pedal note identity | PASS, 19/19 |
| Pedal leakage | 0 |
| Epoch-1 training state | RESUME_READY, epoch 1 / global step 516 |
| Training performed in this task | 0 |
| ASAP test access | 0 |

## Preserved failure state

The original run artifacts, traceback, five successful Frozen-PT candidates, and both 1.2 GB train-complete checkpoints were retained. The log proves:

- epoch 1 training completed at global step 516;
- `train_complete_epoch_01.pt` and `resume_last_train_state.pt` were atomically saved before validation;
- Human-input validation reached 71/71;
- Frozen-PT pieces 1 through 5 completed;
- piece 6 failed before its candidate destination was written;
- the failure was not NaN, CUDA, mapping, ownership, or training failure.

Checkpoint SHA-256 values remained unchanged throughout diagnosis:

- train-complete epoch 1: `077b63441f73b51caf324cb02a24364ae89d1e044eb2208188c467a36a68ae6b`
- resume-last: `435eab6e1da6672090552c25c22302bbe6f4958f6added4b22a231ed1b2e64e0`

## Exact reproduction

The epoch-1 saved model reproduced the error without training.

| Field | Value |
|---|---|
| Piece index / ID | 6 / `piece_b45364ab6817194a` |
| Frozen MIDI | `analysis/stage2_binary_canonical_v1/canonical_validation_stage1/piece_b45364ab6817194a/original_pt.mid` |
| Human references | metadata 443 (`ChenS01.mid`), 444 (`Na06M.mid`) |
| First onset `t1` | 0.0 s |
| PRE | `[-1.0, 0.0)` s |
| Latest non-pedal note-off `Tend` | 384.6606976880003 s |
| POST right | 385.6606976880003 s |
| MIDI ticks per beat | 500 |
| Initial tempo | 500,000 microseconds/beat |
| Source absolute tick range | 0 to 616,239 |

The first and only illegal event was:

| Field | Value |
|---|---|
| Region / interval / slot | PRE / 0 / 0 |
| Model state before | OFF (0) |
| Predicted Count | 1 |
| Predicted tau1 / tau2 | 0.5524795651 / 0.6261833906 (tau2 inactive) |
| Direction / CC64 | DOWN / 127 |
| Absolute seconds | -0.4475204349 |
| Absolute MIDI tick | -448 |
| Trigger | `tick < 0` |

The value 127 is legal. At the initial tempo, 500 ticks/beat and 0.5 seconds/beat give 1,000 ticks/second, which explains the rounded tick -448.

## Writer range audit

`write_candidate()` rejects exactly:

```text
not (0 <= cc64_value <= 127) OR absolute_tick < 0
```

It has no source-maximum-tick upper bound. Positive POST predictions after the original EOT are legal: the writer moves only EOT to the latest inserted CC64 tick while preserving every other non-CC64 event. The regression and saved-model audit exercised five such EOT extensions successfully.

Therefore the failure categories are:

- CC64 value: no;
- negative tick: **yes**;
- upper/source-max tick: no such restriction;
- integer/delta overflow: no;
- chronology violation: no;
- other: Frozen-PT renderer lacked a MIDI-origin projection for conceptual negative PRE time.

## Minimal semantic fix

The scientific PRE interval remains `[t1-1s,t1)`, the model rollout remains unchanged, and the conceptual negative-time event remains in `rollout.candidate_events` for metric evaluation. The fix is serialization-only:

1. replay decoded PRE events strictly before 0 seconds from the frozen OFF initial state;
2. determine the binary state immediately at MIDI origin;
3. if that state is ON, write one DOWN/127 state anchor at tick 0 before any real tick-0 prediction;
4. if it is OFF, write no anchor;
5. serialize every nonnegative decoded event at its original converted tick.

This is not `tick=max(0,tick)`: individual negative timestamps are never moved or duplicated. Multiple pre-origin toggles collapse to their only serializable sufficient statistic—the absolute state at time zero. Negative MAIN/POST events still fail fast.

The model prediction was valid within its frozen PRE domain, and the generic MIDI writer was correct to reject a negative tick. The defect was the Frozen-PT serialization adapter between those domains.

## Regression and 19-piece audit

Focused tests passed 22/22 across the new origin-projection suite, existing mapping/checkpoint suite, and frozen PRE/MAIN/POST suite. They cover the exact failed event, CC range, nonnegative delta times, chronology, t1 above/equal/below one second, negative non-PRE rejection, positive POST after source EOT, unchanged ownership, pedal masking, and checkpoint round-trip.

Using the exact epoch-1 model:

- 19/19 pieces rendered;
- renderer exceptions: 0;
- non-pedal note identity: 19/19;
- invalid CC64 values/ticks: 0;
- pedal leakage: 0;
- conceptual pre-origin events: 3 across 3 pieces;
- tick-0 state anchors: 3;
- pieces requiring legitimate EOT extension: 5;
- mapping remained 19 pieces / 71 references, canonical aligned population 70/71 (metadata 856 exclusion unchanged).

## Concurrency audit

Conclusion: **NOT SUPPORTED BY EVIDENCE**. One driver command and traceback appear in the tmux pane, no duplicate Binary 2-Slot process remained, only the five logged successful destinations existed, the failing destination was never written, source MIDI SHA matched its frozen manifest, and all relevant source mtimes predated launch. Concurrent edits elsewhere in the dirty repository do not explain the deterministic saved-checkpoint reproduction.

## Resume readiness

Epoch-1 model, optimizer, AMP scaler, Python/NumPy/Torch CPU/CUDA RNG, epoch, and global step are present. Loading the actual model+optimizer+scaler succeeded with epoch 1, global step 516, 134 optimizer states, AMP scale 1024, and 135 model-state keys. Epoch-1 training must not be rerun.

Human validation completed in the log but its aggregate return value was never persisted because the driver crashed before epoch finalization. It must therefore be rerun. Frozen-PT validation must also be rerun. The safe next task is:

1. load `resume_last_train_state.pt`;
2. rerun epoch-1 Human validation;
3. rerun epoch-1 Frozen-PT validation;
4. persist epoch-1 metrics and select `best.pt` from completed Frozen-PT F1;
5. restore the saved post-training RNG state immediately before continuing;
6. resume epoch 2 with the saved optimizer/scaler.

The checkpoint is sufficient, but the current full driver still needs an explicit validation-pending resume control path in the next task. No training was resumed here.

## Required questions

**Q1. Which piece failed?** Piece 6, `piece_b45364ab6817194a`.

**Q2. Which event was invalid?** PRE interval 0, OFF start, predicted N=1, DOWN=127 at tau1 0.5524795651, absolute -0.4475204349 seconds / tick -448.

**Q3. What was invalid?** Negative tick only.

**Q4. Which region?** PRE. POST was not the cause.

**Q5. Invalid prediction or overly strict writer?** Neither in isolation: the prediction is valid in conceptual PRE time and the writer correctly rejects negative MIDI ticks. The Frozen-PT renderer omitted the required domain projection.

**Q6. Was concurrent Codex work causal?** No evidence supports that conclusion.

**Q7. Does rendering pass after the minimal fix?** Yes, 19/19 with zero renderer exceptions and 19/19 note identity.

**Q8. Is epoch-1 state exactly resumable?** Yes. Model, optimizer, scaler, and all required RNG states load successfully at epoch 1/global step 516.

**Q9. Can the next task continue from epoch-1 validation?** Yes. Rerun both validations, finalize epoch 1, then resume epoch 2; do not retrain epoch 1.

## Modified and added files

- `src/stage2_binary_2slot/frozen_pt_validation.py`: MIDI-origin projection in the renderer only.
- `tests/test_binary_2slot_midi_range_fix_v1.py`: focused projection and POST-tail regressions.
- `scripts/diagnose_binary_2slot_midi_range_failure_v1.py`: saved-model reproduction and 19-piece audit driver.
- `failing_event_diagnostic.json`: exact pre-fix saved-model event record.
- `frozen_19piece_renderer_audit.json`: post-fix 19-piece evidence.
- `concurrency_audit.json`, `regression_test_results.json`, `epoch_01_resume_readiness.json`: supporting provenance.
