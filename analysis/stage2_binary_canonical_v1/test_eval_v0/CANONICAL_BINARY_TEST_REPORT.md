# Canonical-v1 Binary Stage 2 Test Report

## A. Locked experiment provenance

This evaluation uses only the two checkpoints selected by canonical validation strict JS. The test result did not change model, epoch, threshold, decoding, or post-processing.

- Independent 4×2: epoch 2, `9fc60f9b3d91942afaf56f0553fea3f2c39509f32c9965d6236de178077d65d4`
- Joint 16: epoch 2, `4d4d009b0ce9dae44c88dbb1c6e80883295146bbcd6daca3421954ff5ebdacb6`
- Stage 2 inference: 512-note windows, stride 256, overlap-logit averaging, deterministic argmax, threshold 64, decode 0/127.
- Stage 1 neural inference performed in this evaluation: **0**.

## B. Canonical Stage 1 test provenance

The 23 immutable `outputs/midi/{num}_original_pt.mid` files were joined to ASAP pieces through the existing exact score-SHA mapping artifact. Nums 0..22 occur exactly once; filenames were not sorted into an inferred mapping. Canonical source hashes were verified before and after evaluation.

The Human reference contains 104 ASAP test performances. Official `midi_to_ids`, threshold ≥64, and Pedal1-as-MSB 16-pattern semantics reproduced the previously verified Human histogram exactly. Historical histogram artifact SHA-256: `36db39160dee3924139b97f0198587e15965c0f50184b6a5d925b6f1c970e8f3`.

## C. Hard non-CC64 equality

**46/46 PASS.** Every Independent and Joint candidate preserves MIDI type, ticks per beat, track layout, every ordered non-CC64 channel/meta event, and end-of-track timing exactly. Only sustain CC64 may differ.

## D. CC64 transplant audit

| Model | donor CC64 | transplanted CC64 | discarded after canonical EOT | pieces with discard |
|---|---:|---:|---:|---:|
| Independent 4×2 | 9712 | 9710 | 2 | 2 |
| Joint 16 | 5562 | 5562 | 0 | 0 |

Predicted Pedal1--4 values followed the validated `0/127 → ids_to_midi → map_midi donor → CC64-only transplant` conversion path. No new pedal timing algorithm was introduced.

## E. Global headline result

| Model | canonical test strict JS ↓ | Intersection ↑ |
|---|---:|---:|
| Original PT canonical CPU | 0.146432600425 | 0.847466876476 |
| Independent 4×2 | 0.185597746584 | 0.788894290429 |
| Joint 16 | 0.250117450050 | 0.730238116374 |

| Model | ΔJS (candidate − PT) | relative ΔJS | ΔIntersection | beats PT? |
|---|---:|---:|---:|---|
| Independent 4×2 | +0.039165146159 | +26.746193% | -0.058572586047 | no |
| Joint 16 | +0.103684849624 | +70.807217% | -0.117228760103 | no |

The global MIDI-roundtrip metric above is the primary result. Direct pre-render token metrics are not reported as headline results.

## F. Global 16-pattern diagnostics

| Distribution | steady mass | transition-containing mass |
|---|---:|---:|
| Human | 0.904962964750 | 0.095037035250 |
| Original PT | 0.953234960272 | 0.046765039728 |
| Independent 4×2 | 0.933987352035 | 0.066012647965 |
| Joint 16 | 0.977039078969 | 0.022960921031 |

Full count/probability and signed/absolute error diagnostics are in `global_joint16_comparison.csv`.

## G. Piece-level supplemental analysis

| Model | improved pieces | mean JS | median JS |
|---|---:|---:|---:|
| Original PT | — | 0.188965944717 | 0.147473669414 |
| Independent 4×2 | 8/23 | 0.227900139600 | 0.161630310133 |
| Joint 16 | 2/23 | 0.279349024902 | 0.233455725071 |

Piece-level results are supplemental and were not used for model selection.

## H. Validation → test generalization

| Model | validation strict JS ↓ | validation vs PT | test vs PT |
|---|---:|---|---|
| Original PT | 0.161498978464 | — | — |
| Independent 4×2 epoch 2 | 0.150082839377 | improved | did not improve |
| Joint 16 epoch 2 | 0.202442336778 | did not improve | did not improve |

The Independent validation improvement is not retained on canonical test.
The Joint validation failure is also observed on canonical test.

## I. No-retuning and scientific status

No training, checkpoint reselection, architecture reselection, threshold change, calibration, smoothing, post-processing, or Stage 1 inference was performed in response to this test result.

This is the canonical-v1 test evaluation performed after validation-only checkpoint selection under the revised fixed-Stage-1 protocol. No canonical-v1 model or checkpoint selection used the canonical-v1 test result.

ASAP test had been accessed historically in the earlier `stage2_binary_v0` lineage under other protocols. This report therefore does not describe the data as a previously untouched test set or as first-ever ASAP test access. Earlier GPU Stage-1 and prior canonical re-evaluation results remain separate and are not mixed into this headline table.
