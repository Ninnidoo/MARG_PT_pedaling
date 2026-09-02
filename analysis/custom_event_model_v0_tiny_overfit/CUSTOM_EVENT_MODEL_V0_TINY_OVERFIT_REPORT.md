# Custom Event Model v0 Tiny-Overfit Report

## Scope and invariants

This is a bounded, train-only plumbing and memorization diagnostic. It is not a canonical model run and its checkpoint is not a scientific final checkpoint.

- Frozen tokenizer cache ID: `3a5520155b5db1e9a1da7f8148556aa3e1da852655c9adde25d3dbbd1d966263`
- Subset split: canonical training only
- Seed: 42
- ASAP validation model inference: 0
- ASAP test access: 0
- Repedal execution: 0
- Candidate MIDI generation: 0
- Full canonical training: not started
- Tokenizer/model architecture/loss coefficients: unchanged

## Deterministic subset

The canonical train manifest and 512/256 windows were scanned in stable order. A window was retained only when it added an unmet supervision branch or event-class coverage requirement; scanning stopped after every desired requirement was covered.

- Selection SHA256: `0f03efd352df6f2c8ea43950d0a9dda63a6dd2e60471aebad2da61d5d5859144`
- Performances: 3
- Windows: 4
- Owned onset groups: 1,323
- Initial targets: 1
- Terminal-supervised windows: 3

| Performance | Window | Notes | Owned onsets | Initial | Terminal |
|---|---:|---:|---:|---:|---:|
| `2018/MIDI-Unprocessed_Chamber3_MID--AUDIO_10_R3_2018_wav--1.midi` | 0 | 0–512 | 374 | yes | no |
| same performance | 15 | 3685–4197 | 301 | no | yes |
| `2004/MIDI-Unprocessed_XP_21_R1_2004_01_ORIG_MID--AUDIO_21_R1_2004_01_Track01_wav.midi` | 23 | 5804–6316 | 318 | no | yes |
| `2006/MIDI-Unprocessed_14_R1_2006_01-05_ORIG_MID--AUDIO_14_R1_2006_03_Track03_wav.midi` | 27 | 6822–7334 | 330 | no | yes |

All Main Slot1–6 and Terminal Slot1–4 paths have at least one non-NONE event and a timing target. All four SET destination classes occur in the selected main subset and in the selected terminal subset.

| Slot | Main targets | Main non-NONE | Terminal targets | Terminal non-NONE |
|---|---:|---:|---:|---:|
| 1 | 1,323 | 270 | 3 | 3 |
| 2 | 1,323 | 133 | 3 | 3 |
| 3 | 1,323 | 77 | 3 | 2 |
| 4 | 1,323 | 50 | 3 | 1 |
| 5 | 1,323 | 23 | — | — |
| 6 | 1,323 | 7 | — | — |

## Optimization configuration

- Encoder: official pretrained Pianist Transformer encoder, hidden size 768
- Heads: fresh seed-42 initialization
- Warm start from Stage2/CE/Huber/Hybrid checkpoint: no
- Optimizer: AdamW
- Encoder LR: `1e-5`
- New-head LR: `1e-4`
- Weight decay: `0.01`
- Max gradient norm: `1.0`
- Precision: FP32
- Micro/effective batch: 1/1
- Scheduler: none
- Optimizer steps: 1,000
- Runtime: 89.645 seconds on NVIDIA GeForce RTX 2080 Ti

The loss is the unmodified five-component sum: unweighted initial CE, six-slot macro-averaged weighted main event CE, masked main Smooth L1 with beta 0.1, four-slot macro-averaged weighted terminal event CE, and masked terminal Smooth L1 with beta 0.1. Every lambda is 1.0. Frozen slot weights came from piece-level canonical train frequencies, not from the tiny subset.

## Loss and exact memorization

| Metric | Initial | Final |
|---|---:|---:|
| Total loss | 4.446343 | 0.022406 |
| Initial CE | 0.161053 | 0.000006 |
| Main event CE | 2.116661 | 0.010129 |
| Main timing Huber | 0.705649 | 0.006709 |
| Terminal event CE | 1.026166 | 0.000507 |
| Terminal timing Huber | 0.436815 | 0.005056 |
| Main exact event accuracy | 0.093474 | 1.000000 |
| Main non-NONE precision | 0.071409 | 1.000000 |
| Main non-NONE recall | 0.933929 | 1.000000 |
| Main non-NONE F1 | 0.132674 | 1.000000 |
| Terminal exact event accuracy | 0.416667 | 1.000000 |
| Terminal non-NONE F1 | 0.888889 | 1.000000 |

Final exact counts were 7,938/7,938 main event targets and 12/12 terminal event targets. Predictions did not collapse to NONE: the final predicted and target non-NONE counts were exactly 560 for main and 9 for terminal. All train-present event classes reached recall 1.0 in both branches.

### Per-slot event result

| Branch/slot | Non-NONE targets | Initial accuracy | Final accuracy | Final non-NONE recall |
|---|---:|---:|---:|---:|
| Main 1 | 270 | 0.068027 | 1.000000 | 1.000000 |
| Main 2 | 133 | 0.064248 | 1.000000 | 1.000000 |
| Main 3 | 77 | 0.172336 | 1.000000 | 1.000000 |
| Main 4 | 50 | 0.097506 | 1.000000 | 1.000000 |
| Main 5 | 23 | 0.014361 | 1.000000 | 1.000000 |
| Main 6 | 7 | 0.144369 | 1.000000 | 1.000000 |
| Terminal 1 | 3 | 1.000000 | 1.000000 | 1.000000 |
| Terminal 2 | 3 | 0.000000 | 1.000000 | 1.000000 |
| Terminal 3 | 2 | 0.000000 | 1.000000 | 1.000000 |
| Terminal 4 | 1 | 0.666667 | 1.000000 | 1.000000 |

Sparse Main Slot5/6 and Terminal Slot4 therefore received usable supervision and memorized their covered non-NONE targets.

## Timing result

Only `event != NONE AND timing_valid_mask` targets contributed. NONE placeholder values were excluded throughout.

| Timing metric | Initial | Final |
|---|---:|---:|
| Main tau MAE, 560 targets | 0.832683 | 0.023160 |
| Main tau RMSE | 1.032123 | 0.038406 |
| Terminal log1p-gap MAE, 9 targets | 0.824782 | 0.026538 |
| Terminal log1p-gap RMSE | 0.965392 | 0.042525 |
| Terminal decoded seconds-gap MAE | 1.391952 s | 0.046510 s |

| Slot | Main valid | Main MAE initial→final | Terminal valid | Terminal log1p MAE initial→final |
|---|---:|---:|---:|---:|
| 1 | 270 | 0.711730 → 0.012025 | 3 | 0.731718 → 0.018506 |
| 2 | 133 | 1.273066 → 0.022182 | 3 | 0.944228 → 0.017470 |
| 3 | 77 | 0.757648 → 0.011413 | 2 | 0.774582 → 0.010527 |
| 4 | 50 | 0.432062 → 0.099741 | 1 | 0.846030 → 0.109859 |
| 5 | 23 | 0.855009 → 0.015518 | — | — |
| 6 | 7 | 0.744347 → 0.078556 | — | — |

Every timing head improved substantially. The sparsest targets, Main Slot6 and Terminal Slot4, remain noisier than common slots at the final step but are neither flat nor disconnected.

## Loss-scale and gradient diagnosis

| Component | Initial | Early median | Final | Minimum |
|---|---:|---:|---:|---:|
| Initial CE | 0.161053 | 0.000172 | 0.000006 | 0.000006 |
| Main event CE | 2.116661 | 0.416198 | 0.010129 | 0.010129 |
| Main timing | 0.705649 | 0.018026 | 0.006709 | 0.000645 |
| Terminal event CE | 1.026166 | 0.009150 | 0.000507 | 0.000507 |
| Terminal timing | 0.436815 | 0.016802 | 0.005056 | 0.001791 |

Main event CE is initially the largest component, but no branch is flat or starved. All five components fall sharply. This tiny diagnostic does not show a clear failure that warrants changing the fixed lambda=1 configuration before a controlled full run. Component scales should still be monitored under the canonical data distribution; no coefficient was tuned here.

All recorded gradients were finite. The first four cyclic steps demonstrated non-zero encoder/main gradients and, on the three terminal-owner windows, non-zero terminal event and timing gradients. At the final cycle, event gradients were small after exact memorization while timing gradients remained active, including Terminal Slot4 on its owning window. This is expected from the sparse continuous targets rather than evidence of a disconnected branch.

## Answers to the diagnostic questions

1. The model memorized all selected event targets exactly.
2. There was no all-NONE collapse; non-NONE counts and class recalls exactly matched targets at the end.
3. Main Slot1–6 all learned, including Slot5/6.
4. Terminal Slot1–4 all learned, including the single covered Slot4 event.
5. Main tau MAE decreased from 0.832683 to 0.023160.
6. Terminal log1p-gap MAE decreased from 0.824782 to 0.026538; decoded gap MAE fell to 0.046510 seconds.
7. The five-component lambda=1 setup is not manifestly broken in this plumbing test.
8. Frozen slot weighting did not prevent exact memorization and did not cause NONE collapse.
9. Recommendation: **READY FOR FULL TRAINING**. This means the pipeline is technically learnable, not that validation performance is known.

No full training, validation inference, ASAP test access, Repedal evaluation, or candidate generation followed this diagnostic.
