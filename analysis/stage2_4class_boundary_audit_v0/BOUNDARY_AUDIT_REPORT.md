# Stage 2 4-class fixed-64 boundary audit (v0)

## Scope and provenance

This report is a train-only boundary audit, not a model experiment and not a tokenizer change. It reuses the current canonical 2-class Stage 2 `MAESTRO-clean + ASAP-train` cache, whose tokens were produced by the pinned official Pianist Transformer `midi_to_ids` tokenizer. Pedal targets are cache token columns Pedal1–Pedal4 minus token offset 5,261.

- Cache ID: `85a79e9d10e955b10000f72f6bbc4a29dbd2cc5a4c8cfb1277054a8d45dcb877`; cache token SHA-256: `5a629707795d9e4968f98a751768dca84b0f46600832714bb15fde4d46274ecd`.
- Train performances: 892 ASAP + 1,170 MAESTRO = 2,062; train notes: 9,369,095; Pedal1–4 targets: 37,476,380.
- The immutable canonical train manifest/index defines the Stage 2 training corpus. MAESTRO retains its source-dataset split metadata, while all ASAP records are explicitly in ASAP `train` and disjoint from ASAP validation/test paths. No validation/test token array, performance index, or MIDI file was opened.
- Raw CC64 is counted only from each matching train-manifest performance MIDI. It is supplementary: boundary ranking uses the tokenized Stage 2 target distribution below.

## Raw CC64 event distribution (supplementary)

`raw_cc64_distribution.csv` contains counts, proportions, and CDFs for every 0–127 value. The figures show the full linear histogram, an intermediate-only log-scale view, and the CDF.

| dataset | events | 0 | 1–63 | 64–126 | 127 | top spikes |
| --- | --- | --- | --- | --- | --- | --- |
| ASAP | 3,848,103 | 3.713% | 32.729% | 59.881% | 3.677% | 0 (3.713%), 127 (3.677%), 64 (2.320%), 65 (2.217%), 63 (2.118%) |
| MAESTRO | 8,383,275 | 3.996% | 30.269% | 61.713% | 4.022% | 127 (4.022%), 0 (3.996%), 64 (2.483%), 65 (2.324%), 66 (2.182%) |
| combined | 12,231,378 | 3.907% | 31.043% | 61.137% | 3.913% | 127 (3.913%), 0 (3.907%), 64 (2.432%), 65 (2.290%), 63 (2.157%) |

Quantiles (nearest-rank on raw event values):

| dataset | p00 | p01 | p05 | p10 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ASAP | 0 | 0 | 18 | 35 | 56 | 70 | 88 | 107 | 119 | 127 | 127 |
| MAESTRO | 0 | 0 | 18 | 38 | 58 | 71 | 90 | 108 | 121 | 127 | 127 |
| combined | 0 | 0 | 18 | 37 | 57 | 71 | 89 | 108 | 120 | 127 | 127 |

## Actual Stage 2 Pedal1–4 targets (primary)

`pedal_target_distribution.csv` provides the 128-bin count/proportion/CDF for flattened Pedal1–4 and each slot, separately for ASAP, MAESTRO, and combined training data. The following flattened view is the primary evidence for candidate boundaries.

| dataset | targets | 0 | 1–63 | 64–126 | 127 | OFF / ON | top spikes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ASAP | 11,954,456 | 24.892% | 14.713% | 24.127% | 36.269% | 39.604% / 60.396% | 127 (36.269%), 0 (24.892%), 64 (0.813%), 63 (0.755%), 65 (0.751%) |
| MAESTRO | 25,521,924 | 26.945% | 12.812% | 23.306% | 36.936% | 39.757% / 60.243% | 127 (36.936%), 0 (26.945%), 64 (0.831%), 65 (0.757%), 63 (0.722%) |
| combined | 37,476,380 | 26.290% | 13.419% | 23.568% | 36.723% | 39.708% / 60.292% | 127 (36.723%), 0 (26.290%), 64 (0.825%), 65 (0.755%), 63 (0.733%) |

Target quantiles (nearest-rank):

| dataset | p00 | p01 | p05 | p10 | p25 | p50 | p75 | p90 | p95 | p99 | p100 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ASAP | 0 | 0 | 0 | 0 | 9 | 81 | 127 | 127 | 127 | 127 | 127 |
| MAESTRO | 0 | 0 | 0 | 0 | 0 | 81 | 127 | 127 | 127 | 127 | 127 |
| combined | 0 | 0 | 0 | 0 | 0 | 81 | 127 | 127 | 127 | 127 | 127 |

### Slot consistency

| dataset | slot | targets | OFF | ON | p50 | 0 / 127 |
| --- | --- | --- | --- | --- | --- | --- |
| ASAP | Pedal1 | 2,988,614 | 39.760% | 60.240% | 81 | 24.979% / 35.915% |
| ASAP | Pedal2 | 2,988,614 | 40.103% | 59.897% | 80 | 25.054% / 35.917% |
| ASAP | Pedal3 | 2,988,614 | 39.351% | 60.649% | 82 | 24.773% / 36.570% |
| ASAP | Pedal4 | 2,988,614 | 39.204% | 60.796% | 82 | 24.760% / 36.674% |
| MAESTRO | Pedal1 | 6,380,481 | 40.005% | 59.995% | 81 | 27.112% / 36.448% |
| MAESTRO | Pedal2 | 6,380,481 | 40.319% | 59.681% | 80 | 27.143% / 36.489% |
| MAESTRO | Pedal3 | 6,380,481 | 39.437% | 60.563% | 82 | 26.768% / 37.338% |
| MAESTRO | Pedal4 | 6,380,481 | 39.267% | 60.733% | 83 | 26.757% / 37.471% |
| combined | Pedal1 | 9,369,095 | 39.927% | 60.073% | 81 | 26.431% / 36.278% |
| combined | Pedal2 | 9,369,095 | 40.250% | 59.750% | 80 | 26.476% / 36.306% |
| combined | Pedal3 | 9,369,095 | 39.410% | 60.590% | 82 | 26.132% / 37.093% |
| combined | Pedal4 | 9,369,095 | 39.247% | 60.753% | 83 | 26.120% / 37.217% |

The slot plot and table should be used to judge a shared boundary. This audit keeps one shared candidate boundary by design, while preserving slot-level evidence instead of assuming the four slots are identical.

## Fixed-64 candidate scan

All 3,969 non-empty integer partitions were scanned: `b_off = 0…62`, `b_on = 64…126`. Every candidate has `ZERO=0…b_off`, `LOW=b_off+1…63`, `HALF=64…b_on`, `FULL=b_on+1…127`.

The balance score is the L1 distance to uniform 25% proportions (lower is better); entropy, smallest, and largest class proportions are also in `boundary_candidates.csv`. Balance is a diagnostic only—not a choice rule.

### Required and scan-selected candidates

| b_off | b_on | ZERO % | LOW % | HALF % | FULL % | combined median representatives | Oracle MAE | RMSE | Balance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 31 | 95 | 28.736% | 10.972% | 16.290% | 44.002% | ZERO=0, LOW=53, HALF=77, FULL=127 | 3.9469 | 8.0645 | L1=0.4548; H=1.2576 |
| 20 | 109 | 27.267% | 12.442% | 20.264% | 40.027% | ZERO=0, LOW=51, HALF=81, FULL=127 | 3.8453 | 7.5726 | L1=0.3459; H=1.3036 |
| 25 | 103 | 27.841% | 11.867% | 18.785% | 41.507% | ZERO=0, LOW=51, HALF=79, FULL=127 | 3.7371 | 7.2881 | L1=0.3870; H=1.2880 |
| 0 | 126 | 26.290% | 13.419% | 23.568% | 36.723% | ZERO=0, LOW=49, HALF=84, FULL=127 | 4.9647 | 10.1748 | L1=0.2603; H=1.3292 |

- Lowest oracle MAE in the full scan: `b_off=25`, `b_on=102` and `b_on=103` tie; the table/shortlist uses `b_on=103` for its lower balance L1.
- Best uniform-balance L1 in the full scan: `b_off=0`, `b_on=126`.
- Pareto-optimal candidates (MAE vs uniform L1): 209; see `boundary_candidates.csv` (`pareto_optimal=true`) and `candidate_mae_balance_tradeoff.png`.

### Representative policy and oracle fidelity

For every candidate and each class, the decoder representative is the median of combined train Pedal1–4 raw values assigned to that class—not the interval midpoint. The CSV also records ASAP-only and MAESTRO-only medians for every class. Oracle MAE/RMSE/tolerance metrics are therefore the unavoidable error of this 4-class quantizer under its own train-data medians; they are not model predictions or validation metrics.

### Fixed 64 OFF/ON invariant

The program verifies every candidate has zero OFF/ON mismatches: `ZERO + LOW` is always raw `<64`, while `HALF + FULL` is always raw `>=64`. Collapsing a four-class token to OFF/ON therefore exactly equals thresholding the original raw target at 64. Consequently binary transition labels at this threshold cannot change when only `b_off` or `b_on` changes; only within-side depth quantization changes.

## Candidate shortlist (not a final tokenizer decision)

| b_off | b_on | ZERO % | LOW % | HALF % | FULL % | representatives | Oracle MAE | RMSE | Balance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 31 | 95 | 28.736% | 10.972% | 16.290% | 44.002% | ZERO=0, LOW=53, HALF=77, FULL=127 | 3.9469 | 8.0645 | L1=0.4548; H=1.2576 |
| 20 | 109 | 27.267% | 12.442% | 20.264% | 40.027% | ZERO=0, LOW=51, HALF=81, FULL=127 | 3.8453 | 7.5726 | L1=0.3459; H=1.3036 |
| 25 | 103 | 27.841% | 11.867% | 18.785% | 41.507% | ZERO=0, LOW=51, HALF=79, FULL=127 | 3.7371 | 7.2881 | L1=0.3870; H=1.2880 |
| 0 | 126 | 26.290% | 13.419% | 23.568% | 36.723% | ZERO=0, LOW=49, HALF=84, FULL=127 | 4.9647 | 10.1748 | L1=0.2603; H=1.3292 |
| 24 | 103 | 27.726% | 11.982% | 18.785% | 41.507% | ZERO=0, LOW=51, HALF=79, FULL=127 | 3.7383 | 7.2921 | L1=0.3847; H=1.2890 |

- **(31, 95)**: information preservation is summarized by oracle MAE 3.947; class imbalance ranges from 10.972% to 44.002%. The integer boundaries are directly interpretable under fixed 64; maximum ASAP-vs-MAESTRO class-median difference is 0 CC64 units.
- **(20, 109)**: information preservation is summarized by oracle MAE 3.845; class imbalance ranges from 12.442% to 40.027%. The integer boundaries are directly interpretable under fixed 64; maximum ASAP-vs-MAESTRO class-median difference is 1 CC64 units.
- **(25, 103)**: information preservation is summarized by oracle MAE 3.737; class imbalance ranges from 11.867% to 41.507%. The integer boundaries are directly interpretable under fixed 64; maximum ASAP-vs-MAESTRO class-median difference is 1 CC64 units.
- **(0, 126)**: information preservation is summarized by oracle MAE 4.965; class imbalance ranges from 13.419% to 36.723%. The integer boundaries are directly interpretable under fixed 64; maximum ASAP-vs-MAESTRO class-median difference is 2 CC64 units.
- **(24, 103)**: information preservation is summarized by oracle MAE 3.738; class imbalance ranges from 11.982% to 41.507%. The integer boundaries are directly interpretable under fixed 64; maximum ASAP-vs-MAESTRO class-median difference is 1 CC64 units.

No canonical boundary is selected by this audit. The combined corpus is the primary distribution for any later choice, while the separated ASAP/MAESTRO rows expose cross-dataset sensitivity.

## Files

- `raw_cc64_distribution.csv`: raw train-MIDI CC64 distributions.
- `pedal_target_distribution.csv`: official-tokenizer Pedal1–4 distributions, flattened and per slot.
- `boundary_candidates.csv`: all fixed-64 candidates with balance, representatives, oracle fidelity, Pareto flag, and invariant checks.
- `boundary_shortlist.csv`: the 3–5 candidates summarized above.
- `raw_cc64_*.png`, `pedal_target_*.png`, `candidate_mae_balance_tradeoff.png`: audit visualizations.
