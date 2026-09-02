# Binary 2-Slot Condition D PRE/MAIN/POST Restore Smoke v0

## Outcome

| check | result |
|---|---:|
| status | **PASS** |
| subset | canonical TRAIN 7 |
| optimizer steps | 1000 |
| late MAIN Count accuracy median | 1.000000 |
| late MAIN Macro F1 median | 1.000000 |
| late MAIN state agreement median / p10 | 1.000000 / 1.000000 |
| late first-MAIN agreement median | 1.000000 |
| final MAIN timing MAE | 0.009124 |
| online reconciliation / dynamic target | 0 / 0 |
| full training / validation inference / ASAP test | 0 / 0 / 0 |

## Frozen restoration

The restored path predicts PRE, every owned MAIN interval including the final `[t_M,T_end)` tail, and POST with one continuous hard state. Count/Timing targets are immutable static human targets. The state-consistency loss is the previously verified parity loss with `lambda_state=0.25`; total loss is `MAIN + 0.25 * BOUNDARY`, with Count, Timing, and `0.25 * State` inside each region. Human state is used only for static target construction, the local loss label, and diagnostics.

PRE uses a dataset-fixed initial synchronization event only when the human state at `t1-1s` is ON. It is prepended before real PRE events and undergoes ordinary max-2 compression without reservation. Inference receives no human state and performs no forced correction.

## Canonical PRE target audit

Train: 2062 performances, 324 static corrections, 0 violations. Validation target-only audit: 71 performances, 5 corrections, 0 violations. ASAP test was not accessed.

Subset 7 (`2009/MIDI-Unprocessed_02_R1_2009_03-06_ORIG_MID--AUDIO_02_R1_2009_02_R1_2009_04_WAV.midi`): PRE-left human state=1, correction inserted=True, raw PRE crossings=0, retained Count=1, retained parity=1, oracle PRE result at t1=1, human state at t1=1. Invariant: **True**.

## Cold-start comparison

The prior MAIN-only run began OFF while subset 7 human `t1` state was ON, producing Count accuracy 1.0 but state agreement 0.0. That reproduced a global parity-offset cold start rather than a Count-capacity failure.

After restoring PRE, final PRE prediction Count=1 against target=1; resulting t1 state=1 and human t1 state=1. First-MAIN agreement=1.000000. Final MAIN Count accuracy=1.000000, Macro F1=1.000000, state agreement=1.000000, mismatch fraction=0.000000. The late median state agreement is 1.000000; 50/50 attractor=False, N=1 collapse=False.

## Required conclusions

1. **MAIN-only root cause reproduced/resolved?** The previous OFF-vs-ON initialization mismatch is documented above. Restored PRE resolved the learned cold-start parity offset.
2. **Was a static PRE correction generated on subset 7?** True; exact retained details are in `pre_target_audit.json`.
3. **Was first MAIN synchronized?** Final=True; late median=1.000000.
4. **Was hard free-running MAIN stable?** Late state median/p10=1.000000/1.000000, mismatch-run maximum at final=0.
5. **Was online reconciliation used?** No. Static builders accept no model state or prediction; runtime counters are zero.
6. **Proceed to full training?** **PASS**. The focused diagnostic found no blocker; full training still requires a separate user task.

## Provenance and stop point

Seed 42, pretrained PT encoder, AdamW encoder/head LR `1e-5/1e-4`, weight decay `0.01`, gradient clip `1.0`, FP32, fixed Count weights, `lambda_state=0.25`, and boundary weight `0.25` match the prior diagnostic. This task ran exactly one 1,000-step TRAIN-only smoke. It created no checkpoint, ran no full training, ran no validation inference/metric, and accessed no ASAP test data.
