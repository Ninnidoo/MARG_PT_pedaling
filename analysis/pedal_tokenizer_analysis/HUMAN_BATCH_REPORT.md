# Human Reference Performance Pedal-Tokenizer Batch Report

Date: 2026-07-27

## Scope

This analysis measures how much sustain-pedal information in the Pianist
Transformer repository's human ground-truth/reference performance MIDI survives
the official tokenizer round trip.

Input:

- `third_party/PianistTransformer/data/midis/testset/human/`
- 166 MIDI files inventoried
- 165 analyzed
- `3-1.mid` excluded from pedal analysis because it has no CC64 events

The provenance label used here is "repository-provided human
ground-truth/reference performance MIDI." The repository evidence does not
establish that these files are unprocessed raw sensor captures.

No model prediction, model inference, or training was run.

## Verified Command And Environment

```powershell
& 'C:\Users\eagle\AppData\Local\Programs\Python\Python311\python.exe' 'analysis\pedal_tokenizer_analysis\analyze_human_batch.py'
```

The command completed with exit code 0:

- inventory: 166
- eligible: 165
- successful: 165
- failed: 0

Environment used:

- Python 3.11.2
- torch 2.11.0
- mido 1.3.3
- numpy 1.26.4
- scipy 1.17.1
- matplotlib 3.8.4
- miditoolkit 1.0.1

No package was installed or upgraded for this batch. `pretty_midi` is not
installed, but it is not imported by this analysis or by the tokenizer path
used here.

## Official Tokenizer Path

The analysis imports these functions from
`third_party/PianistTransformer/src/utils/midi.py`:

- `normalize_midi()`
- `midi_to_ids()`
- `ids_to_midi()`

It uses the official `PianoT5GemmaConfig`. Raw and reconstructed CC64 are
compared on the normalized 120 BPM, 500 ticks/beat axis (1000 ticks/second) on
a shared 10 ms grid.

`normalize_midi(original)` creates the raw normalized comparison trajectory.
Separately, `midi_to_ids(original, normalize=True)` invokes the official
normalization once inside tokenization. A previously normalized MIDI is not
normalized again.

## Data Totals

- Inventory notes, including excluded `3-1.mid`: 582,082
- Analyzed notes: 581,558
- Raw CC64 events: 659,048
- Raw intermediate CC64 events (values 1-126): 597,506
- Raw intermediate event ratio: 90.662%
- Non-CC64 controllers are inventoried but never interpreted as pedal.

## A. Raw MIDI Information

The raw representation is the original CC64 event stream after official tempo
normalization. It remains a step-event trajectory; the analysis uses zero-order
hold only to read the state on a comparison grid. It does not linearly
interpolate or invent intermediate pedal movements.

The high 90.662% intermediate *event* ratio reflects dense controller sampling.
It must not be directly compared with the token-slot ratio because the raw MIDI
and tokenizer have very different event rates.

## B. Tokenizer Representation

The tokenizer produced exactly four pedal values per analyzed note:

- total pedal tokens: 2,326,232
- value 0: 749,131
- value 127: 848,693
- value 1-126: 728,408
- intermediate token-slot ratio: 31.313%

CC64 depth values 0-127 are representable. The principal bottleneck is temporal:
only Pedal1-4 are sampled per next-note IOI. There were 18,789 note rows with
next IOI equal to zero; their four official pedal samples occur at one
timestamp and they contribute no duration interval to local-IOI attribution.

## C. Reconstruction Approximation

`ids_to_midi()` places the four token values back at their per-IOI positions and
deduplicates consecutive equal values. Raw versus reconstructed trajectory
metrics are:

| Metric | File Mean | File Median | Dataset-Micro |
| --- | ---: | ---: | ---: |
| Raw intermediate time ratio | 0.34432 | 0.32300 | 0.33552 |
| Reconstructed intermediate time ratio | 0.34485 | 0.31940 | 0.33614 |
| Reconstructed minus raw ratio | +0.00053 | +0.00069 | +0.00062 |
| CC64 NMAE | 0.05111 | 0.04193 | 0.05480 |
| CC64 RMSE | 19.130 | 17.298 | 21.485 |
| JS divergence | 0.006669 | 0.004601 | not pooled |
| Wasserstein distance | 0.8431 | 0.6158 | not pooled |

The file-level mean intermediate-time difference 95% bootstrap CI is
[-0.00173, 0.00266]. Aggregate occupancy is therefore close, but this does not
mean that individual pedal gestures or their timing are preserved.

## Repedal Preservation

Operational definition:

- high: CC64 >= 96
- low: CC64 <= 31
- maximum release-to-redepress duration: 300 ms
- raw/reconstructed landmark matching tolerance: 150 ms

This is an analysis definition, not a finalized musicological annotation.

Results:

- raw repedal candidates: 15,549
- preserved: 13,256
- lost: 2,293
- micro recall: 0.85253
- macro recall: 0.82879
- median per-file recall: 0.86364
- micro precision: 0.96837
- files with at least one raw candidate: 153
- files excluded from macro recall because raw count was zero: 12

Matching is ordered and one-to-one. It maximizes the number of matches, then
minimizes the summed release and redepress landmark error.

### Threshold Sensitivity

| Definition | High | Low | Max ms | Raw | Preserved | Lost | Micro Recall | Macro Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 96 | 31 | 300 | 15,549 | 13,256 | 2,293 | 0.8525 | 0.8288 |
| B | 96 | 47 | 300 | 17,551 | 15,192 | 2,359 | 0.8656 | 0.8536 |
| C | 80 | 31 | 300 | 17,844 | 15,247 | 2,597 | 0.8545 | 0.8236 |
| D | 96 | 31 | 500 | 19,006 | 16,142 | 2,864 | 0.8493 | 0.8396 |

The absolute counts change, but micro recall stays in the 0.849-0.866 range.
The main preservation conclusion is not dependent on one tested threshold set.

## Transition Timing

A transition is a crossing of CC64 threshold 64:

- down: below 64 to at least 64
- up: at least 64 to below 64

Raw and reconstructed transitions of the same direction are matched using
ordered one-to-one dynamic matching within 500 ms. The objective first
maximizes match count, then minimizes total absolute timing error.

- raw transitions: 80,902
- reconstructed transitions: 77,283
- matched: 76,630
- unmatched raw: 4,272
- matched rate: 94.72%
- median matched absolute error: 16 ms
- p90: 92 ms
- p95: 139 ms

Signed errors are retained in the event CSV. Unmatched raw transitions are
counted as failures but excluded from the matched-error distribution.

The old greedy pilot matcher and corrected ordered matcher produced identical
Shi05M pilot results: 199 matched, median 38 ms, p90 83 ms. The corrected method
also reports 11 unmatched raw transitions.

## IOI Dependence

Repedal local IOI is the next positive tokenizer note interval containing the
raw pedal-valley/minimum timestamp. Zero-IOI simultaneous-onset rows do not
create time intervals. Only 2 of 15,549 candidates fell in the artificial
terminal 4990 ms interval, so the terminal sentinel does not drive the result.

| Local IOI | Raw | Preserved | Lost | Recall | Transition Median ms | Transition p90 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| <100 ms | 5,421 | 5,233 | 188 | 0.9653 | 6 | 17 |
| 100-250 ms | 6,196 | 5,883 | 313 | 0.9495 | 17 | 35 |
| 250-500 ms | 1,941 | 1,435 | 506 | 0.7393 | 44 | 85 |
| 500-1000 ms | 1,485 | 668 | 817 | 0.4498 | 100 | 170 |
| >=1000 ms | 506 | 37 | 469 | 0.0731 | 200 | 368 |

Recall decreases monotonically across the requested IOI bins, while transition
timing error increases.

The exploratory univariate logistic regression is:

`preserved ~ local_IOI_ms / 100`

- coefficient per additional 100 ms: -0.53175 log odds
- cluster-robust SE by MIDI file: 0.02209
- odds ratio: 0.5876
- 95% odds-ratio CI: [0.5627, 0.6136]
- p-value: 5.50e-128
- events: 15,549 in 153 file clusters

This is a strong association in this dataset, not a causal estimate. Score,
performer/take, terminal behavior, and other musical context are not controlled.

## Score And Take Variability

The full score-level table is in `human_by_score.csv`.

- Score 3 has the clearest consistently difficult group: 5 analyzed takes,
  mean NMAE 0.1413, micro repedal recall 0.2949.
- Score 20 has the largest per-take recall standard deviation among scores
  with at least 5 analyzed takes: 0.1514, despite micro recall 0.9197.
- Scores 8 and 9 also show substantial take-level recall variation
  (standard deviations 0.1001 and 0.0974).
- Score 6 has 35 takes, mean NMAE 0.08175, and micro recall 0.8107.

This confirms that loss varies with both score context and take, although the
analysis does not identify performer identity or a causal source of variation.

### Worst Files By NMAE

1. `3-5.mid`: 0.17954
2. `3-4.mid`: 0.17788
3. `3-3.mid`: 0.13611
4. `18-0.mid`: 0.12281
5. `2-0.mid`: 0.11440

### Worst Files By Repedal Recall

1. `15-0.mid`: 0.0000 (1 raw candidate)
2. `3-0.mid`: 0.1000 (10 candidates)
3. `2-2.mid`: 0.2000 (5 candidates)
4. `3-4.mid`: 0.2258 (31 candidates)
5. `3-3.mid`: 0.2558 (43 candidates)

The first item is a small-denominator outlier; score 3 is the more substantial
and repeatable difficult case.

### Best Files By Repedal Recall

The following reached recall 1.0: `11-1.mid` (31 candidates), `14-0.mid` (13),
`16-0.mid` (4), `5-0.mid` (3), and `10-1.mid` (3). Counts are reported because
perfect recall on a few events is less informative than perfect recall on 31.

## Comparison With Shi05M Pilot

| Metric | Shi05M | Human File-Level Mean |
| --- | ---: | ---: |
| Raw intermediate time ratio | 0.53594 | 0.34432 |
| Reconstructed intermediate time ratio | 0.52849 | 0.34485 |
| CC64 NMAE | 0.05923 | 0.05111 |
| JS divergence | 0.01189 | 0.00667 |
| Repedal recall | 0.5625 | 0.82879 macro |
| Transition median error | 38 ms | 16 ms global |
| Transition p90 error | 83 ms | 92 ms global |

The pilot was broadly representative of the NMAE scale and showed the same
kind of local information loss. It was not representative of dataset-level
repedal recall, raw intermediate occupancy, or median transition timing. The
single-file pilot therefore demonstrated the mechanism but was too pessimistic
to estimate average repedal preservation.

## Main Interpretation

Pedal depth occupancy and pedal gesture preservation are different outcomes.
The tokenizer preserves aggregate intermediate-pedal *time occupancy* closely,
and it can encode intermediate depth values. At the same time, it loses 14.75%
of operational raw repedal candidates and 5.28% of threshold transitions.

The strongest observed limitation is temporal resolution: longer local IOIs
are associated with sharply lower repedal recall and larger transition timing
error. This is consistent with the four-samples-per-IOI design, but this
observational analysis alone does not prove that IOI length is the sole cause.

## Outputs

Primary CSV:

- `outputs/csv/human_midi_inventory.csv`
- `outputs/csv/human_summary_per_file.csv`
- `outputs/csv/human_repedal_events.csv`
- `outputs/csv/human_repedal_threshold_sensitivity.csv`
- `outputs/csv/human_ioi_repedal_analysis.csv`
- `outputs/csv/human_aggregate_statistics.csv`
- `outputs/csv/human_by_score.csv`

Additional validation and analysis:

- `outputs/csv/human_transition_events.csv`
- `outputs/csv/human_ioi_logistic_regression.csv`
- `outputs/csv/human_ioi_logistic_binned.csv`
- `outputs/csv/human_repedal_threshold_sensitivity_per_file.csv`
- `outputs/csv/human_vs_shi05m_pilot_comparison.csv`
- `outputs/csv/human_analysis_errors.csv`
- `outputs/csv/human_validation_report.json`
- `outputs/csv/human_batch_run_metadata.json`

Figures are under `outputs/figures/pedal_tokenizer_analysis/`, including the
ten requested aggregate views, one five-file lost-repedal panel, and five
individual lost-repedal examples.
