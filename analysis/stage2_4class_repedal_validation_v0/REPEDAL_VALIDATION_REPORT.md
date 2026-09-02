# Canonical Repedal Validation Report

## Scope and discovered candidate sets

This CPU-only evaluation parsed saved validation MIDI files and added only the frozen Repedaling metric. It performed no inference, MIDI generation, checkpoint load, GPU operation, threshold tuning, or recomputation of prior 4C/Transition/JS metrics.

| Candidate set | MIDI found | MIDI evaluated | Existing source |
|---|---:|---:|---|
| Human | 14 | 14 | `/workspace/project/analysis/harmonic_metric_broad_parameter_search_v0/input_midis.csv` |
| ALWAYS_ON | 15 | 14 | `/workspace/project/analysis/harmonic_metric_broad_parameter_search_v0/reusable_extremes_manifest.csv; /workspace/project/analysis/harmonic_metric_kdyn_blow_pilot_v0/reusable_extremes/manifest.csv` |
| ALWAYS_OFF | 15 | 14 | `/workspace/project/analysis/harmonic_metric_broad_parameter_search_v0/reusable_extremes_manifest.csv; /workspace/project/analysis/harmonic_metric_kdyn_blow_pilot_v0/reusable_extremes/manifest.csv` |
| Original PT | 19 | 14 | `/workspace/project/analysis/stage2_4class_architecture_validation_eval_v0/predictions/original_pt` |
| Encoder-only 4C | 19 | 14 | `/workspace/project/analysis/stage2_4class_architecture_validation_eval_v0/predictions/encoder_only` |
| Decoder-only 4C FR | 19 | 14 | `/workspace/project/analysis/stage2_4class_architecture_validation_eval_v0/predictions/decoder_only` |
| Encoder-Decoder 4C FR | 19 | 14 | `/workspace/project/analysis/stage2_4class_architecture_validation_eval_v0/predictions/encoder_decoder` |

## Reused evaluation universe

- Source universe: the frozen canonical 4C validation evaluation, 19 pieces / 71 Human performances / 70 successful candidate-Human pairs; its sole frozen alignment failure remains excluded.
- Common availability universe: 14 pieces / 45 candidate-Human pairs. This is the source 70-pair universe filtered only to pieces present in every reused candidate inventory (fixed Human representative plus both saved extremes); no MIDI was generated to fill the 5 unavailable pieces.
- All non-Human rows use exactly this same common pair universe and micro-aggregate TP/FP/FN as the Transition evaluator does.
- Human baseline reuses the previously fixed representative Human MIDI per piece from `input_midis.csv` and compares it with the other successful Human performances for that piece. The identical performance is excluded, so Human is never compared with itself.

## Frozen canonical definition

Consecutive equal raw CC64 values are collapsed. Strict local peaks/troughs are detected; each trough uses the nearest prior and following peaks independently satisfying excursion >=64. Inside that triple, the last >=64 to <64 crossing before the trough and the first <64 to >=64 crossing after it define the OFF episode containing the trough. Event timestamps are the destination CC64 event timestamps, without interpolation, and only 0 < OFF duration <=300 ms is accepted.

Both UP and DOWN boundaries are mapped with the frozen Transition evaluator's cached distinct-onset score positions. A match requires both boundary offsets <=1 and uses deterministic greedy one-to-one matching, prioritizing the narrowest boundary offsets, consistent with the existing Transition matcher's greedy convention.

## Results

Counts below are micro counts over comparison pairs, matching the prior Transition aggregation (a piece-level candidate count is repeated for each Human reference paired with that piece). `Unique pred` is the diagnostic count over saved candidate MIDI files before pair repetition.

| Candidate | Ref count | Pred count | Unique pred | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Human | 4522 | 5590 | 1949 | 2937 | 2653 | 1585 | 0.525403 | 0.649491 | 0.580894 |
| ALWAYS_ON | 6471 | 0 | 0 | 0 | 0 | 6471 | 0.000000 | 0.000000 | 0.000000 |
| ALWAYS_OFF | 6471 | 0 | 0 | 0 | 0 | 6471 | 0.000000 | 0.000000 | 0.000000 |
| Original PT | 6471 | 3782 | 981 | 2858 | 924 | 3613 | 0.755685 | 0.441663 | 0.557495 |
| Encoder-only 4C | 6471 | 6580 | 1941 | 3024 | 3556 | 3447 | 0.459574 | 0.467316 | 0.463413 |
| Decoder-only 4C FR | 6471 | 7464 | 1772 | 925 | 6539 | 5546 | 0.123928 | 0.142945 | 0.132759 |
| Encoder-Decoder 4C FR | 6471 | 40 | 10 | 9 | 31 | 6462 | 0.225000 | 0.001391 | 0.002765 |

## OFF-duration diagnostics

These diagnostics do not alter the frozen 300 ms threshold.

| Candidate | Median ms | p75 | p90 | p95 |
|---|---:|---:|---:|---:|
| Human | 118.857 | 159.375 | 216.994 | 247.262 |
| ALWAYS_ON | — | — | — | — |
| ALWAYS_OFF | — | — | — | — |
| Original PT | 101.887 | 150.303 | 208.286 | 244.034 |
| Encoder-only 4C | 94.918 | 150.790 | 210.882 | 244.095 |
| Decoder-only 4C FR | 36.250 | 69.253 | 133.948 | 179.914 |
| Encoder-Decoder 4C FR | 39.258 | 54.561 | 57.378 | 61.764 |
| Human/reference (unique) | 117.708 | 158.921 | 217.949 | 256.410 |

## Sanity checks

- ALWAYS_ON candidate repedals = 0: PASS
- ALWAYS_OFF candidate repedals = 0: PASS
- Every accepted event has release/repress excursion >=64, both raw threshold crossings, and 0 < OFF duration <=300 ms: PASS
- Human and model MIDI use the same extraction function: PASS
- Frozen Transition distinct-onset mappings and source MIDI hashes were verified and reused: PASS
- Source MIDI unchanged; new MIDI count 0; model inference/checkpoint/GPU/ASAP-test access counts all 0: PASS

## Artifacts

- `repedal_validation_summary.csv`
- `repedal_per_performance.csv`
- `repedal_events.csv`
- `candidate_inventory.csv`
- `sanity_checks.json` and `provenance.json`
- Tested command is recorded in `provenance.json`; it hides CUDA and invokes only this CPU parser/evaluator.
