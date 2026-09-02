# Hall Muddiness H+A 13-Piece Validation Audit v0

## Decision

**B. keep structure but inspect specific failures first**

ALWAYS_ON separation과 severity-vs-accumulation 해석은 강하지만, HUMAN이 PT보다 H와 A 모두 큰 quadrant-4가 5/13이다. 이는 단일 Mephisto 예외가 아니라 반복되는 failure type이므로 weight selection 전에 해당 5곡의 texture/pedal behavior를 inspect해야 한다. 이 보고서는 lambda를 선택하지 않는다.

- Tested command: `python /workspace/project/scripts/audit_hall_muddiness_HA_13piece_validation_v0.py`
- Input: Pure Hall audited onset rows 213,528; 13 pieces x 9 systems
- MIDI parsing / Hall recomputation / grid search / inference / training / MIDI generation: 0 / 0 / 0 / 0 / 0 / 0

## Frozen semantics and aggregation

`RAW=sum max(-HallWeight,0)` over PA/PP only, `H=RAW/N_Q`, `A=ln(1+RAW)`. Q-empty onset is zero. Primary piece values are all-distinct-onset means, so `A_piece=mean_n[ln(1+RAW_n)]`, not `ln(1+sum RAW_n)`. NO_PEDAL is a structural zero, not an overall-quality winner. Stage2<PT is allowed under this restricted harmonic-muddiness interpretation.

## System summary

All H/A statistics are piece-balanced over 13 pieces. IQR and pooled RAW p90 are retained in `system_HA_summary.csv`.

| system | H mean | H median | A mean | A median | RAW p95 | RAW p99 |
| --- | --- | --- | --- | --- | --- | --- |
| NO_PEDAL | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000 | 0.000 |
| ALWAYS_ON | 0.319929 | 0.321777 | 11.109987 | 11.276790 | 930262.485 | 1112002.717 |
| STANDARD_CE_ARGMAX | 0.075738 | 0.070345 | 0.816834 | 0.435369 | 52.426 | 231.157 |
| STANDARD_CE_POSTERIOR_MEDIAN | 0.072721 | 0.063180 | 0.785651 | 0.381565 | 48.620 | 220.966 |
| WEIGHTED_CE_ARGMAX | 0.069135 | 0.064188 | 0.683421 | 0.375839 | 34.350 | 140.368 |
| HYBRID_REGRESSION_ONLY | 0.071654 | 0.053961 | 0.841607 | 0.330262 | 62.428 | 443.893 |
| CUSTOM_EVENT_V0 | 0.142620 | 0.154199 | 1.227567 | 1.141728 | 68.276 | 507.366 |
| ORIGINAL_PT | 0.101888 | 0.110331 | 1.238177 | 1.281313 | 141.610 | 1024.131 |
| HUMAN | 0.120629 | 0.116662 | 1.044679 | 1.065679 | 38.790 | 121.245 |

### Secondary valid-interaction summary

| system | H valid mean | H valid median | A valid mean | A valid median |
| --- | --- | --- | --- | --- |
| NO_PEDAL | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| ALWAYS_ON | 0.320297 | 0.322235 | 11.122491 | 11.283194 |
| STANDARD_CE_ARGMAX | 0.232901 | 0.231680 | 1.725092 | 1.597740 |
| STANDARD_CE_POSTERIOR_MEDIAN | 0.227816 | 0.227198 | 1.640924 | 1.513332 |
| WEIGHTED_CE_ARGMAX | 0.240969 | 0.234293 | 1.608839 | 1.686708 |
| HYBRID_REGRESSION_ONLY | 0.225839 | 0.235227 | 1.733271 | 1.600450 |
| CUSTOM_EVENT_V0 | 0.253005 | 0.263591 | 2.151531 | 2.084667 |
| ORIGINAL_PT | 0.220236 | 0.227255 | 2.085374 | 2.170683 |
| HUMAN | 0.218644 | 0.231554 | 1.674984 | 1.599902 |

CUSTOM_EVENT_V0 is excluded from every canonical Stage2 median and remains a separate reference distribution.

## Relations of interest

| component | comparison | count |
| --- | --- | --- |
| H | HUMAN <= PT | 3/13 |
| H | each canonical Stage2 < ALWAYS | 52/52 |
| H | canonical median < ALWAYS | 13/13 |
| H | PT < ALWAYS | 13/13 |
| H | HUMAN < ALWAYS | 13/13 |
| H | CUSTOM_EVENT_V0 < ALWAYS | 13/13 |
| H | PT < model (descriptive) | 18/52 |
| H | model < PT (descriptive) | 34/52 |
| A | HUMAN <= PT | 8/13 |
| A | each canonical Stage2 < ALWAYS | 52/52 |
| A | canonical median < ALWAYS | 13/13 |
| A | PT < ALWAYS | 13/13 |
| A | HUMAN < ALWAYS | 13/13 |
| A | CUSTOM_EVENT_V0 < ALWAYS | 13/13 |
| A | PT < model (descriptive) | 17/52 |
| A | model < PT (descriptive) | 35/52 |

Stage2/PT directions are descriptive only and are not success/failure criteria.

### HUMAN minus PT deltas

| component | mean delta | median delta | min | max |
| --- | --- | --- | --- | --- |
| H | 0.018741 | 0.008252 | -0.010822 | 0.090787 |
| A | -0.193498 | -0.080691 | -1.099133 | 0.492378 |

## Normal-range versus ALWAYS_ON separation

| component | reference | median Always/reference | median Always-reference |
| --- | --- | --- | --- |
| H | HUMAN | 2.622 | 0.189221 |
| H | ORIGINAL_PT | 2.772 | 0.227591 |
| H | CANONICAL_STAGE2_MEDIAN | 5.189 | 0.266740 |
| A | HUMAN | 11.024 | 10.535767 |
| A | ORIGINAL_PT | 9.168 | 10.519003 |
| A | CANONICAL_STAGE2_MEDIAN | 31.462 | 10.748506 |

A yields much larger multiplicative separation than H for HUMAN, PT, and canonical Stage2 median. Absolute differences are reported in native component units and are not compared across scales.

## HUMAN/PT failure quadrants

| quadrant | count | pieces |
| --- | --- | --- |
| 1 | 3 | Beethoven/Piano_Sonatas_9-3<br>Schubert/Piano_Sonatas_664-1<br>Ravel/Pavane |
| 2 | 5 | Schumann/Kreisleriana_6<br>Scriabin/Etudes_op_8_11<br>Beethoven/Piano_Sonatas_27-1<br>Schumann/Arabeske<br>Chopin/Etudes_op_25_12 |
| 3 | 0 | — |
| 4 | 5 | Bach/Prelude_bwv_862<br>Haydn/Keyboard_Sonatas_46-1<br>Beethoven/Piano_Sonatas_29-2<br>Haydn/Keyboard_Sonatas_39-2<br>Haydn/Keyboard_Sonatas_32-1 |

| piece | quadrant | delta H | delta A |
| --- | --- | --- | --- |
| Bach/Prelude_bwv_862 | 4 | 0.006962 | 0.034250 |
| Haydn/Keyboard_Sonatas_46-1 | 4 | 0.013674 | 0.059793 |
| Beethoven/Piano_Sonatas_9-3 | 1 | -0.002272 | -0.060139 |
| Schumann/Kreisleriana_6 | 2 | 0.027513 | -0.734538 |
| Scriabin/Etudes_op_8_11 | 2 | 0.000224 | -1.063319 |
| Beethoven/Piano_Sonatas_27-1 | 2 | 0.025586 | -0.171420 |
| Schubert/Piano_Sonatas_664-1 | 1 | -0.010822 | -0.215634 |
| Beethoven/Piano_Sonatas_29-2 | 4 | 0.027271 | 0.329222 |
| Schumann/Arabeske | 2 | 0.008252 | -0.080691 |
| Chopin/Etudes_op_25_12 | 2 | 0.006331 | -0.235255 |
| Ravel/Pavane | 1 | -0.003764 | -1.099133 |
| Haydn/Keyboard_Sonatas_39-2 | 4 | 0.090787 | 0.492378 |
| Haydn/Keyboard_Sonatas_32-1 | 4 | 0.053889 | 0.229017 |

Quadrant 4 means both deltas are positive; no nonnegative additive lambda can make HUMAN<=PT there. Quadrant 2 has worse H but better A and can become compatible above an analytic lower bound. There are no quadrant-3 pieces in this set.

## Additive feasibility geometry

| piece | HUMAN <= PT | ModelMedian < ALWAYS | combined |
| --- | --- | --- | --- |
| Bach/Prelude_bwv_862 | EMPTY | [0, +inf) | EMPTY |
| Haydn/Keyboard_Sonatas_46-1 | EMPTY | [0, +inf) | EMPTY |
| Beethoven/Piano_Sonatas_9-3 | [0, +inf) | [0, +inf) | [0, +inf) |
| Schumann/Kreisleriana_6 | [0.0374555673317, +inf) | [0, +inf) | [0.0374555673317, +inf) |
| Scriabin/Etudes_op_8_11 | [0.000210376611634, +inf) | [0, +inf) | [0.000210376611634, +inf) |
| Beethoven/Piano_Sonatas_27-1 | [0.1492606497, +inf) | [0, +inf) | [0.1492606497, +inf) |
| Schubert/Piano_Sonatas_664-1 | [0, +inf) | [0, +inf) | [0, +inf) |
| Beethoven/Piano_Sonatas_29-2 | EMPTY | [0, +inf) | EMPTY |
| Schumann/Arabeske | [0.102272855566, +inf) | [0, +inf) | [0.102272855566, +inf) |
| Chopin/Etudes_op_25_12 | [0.0269105671766, +inf) | [0, +inf) | [0.0269105671766, +inf) |
| Ravel/Pavane | [0, +inf) | [0, +inf) | [0, +inf) |
| Haydn/Keyboard_Sonatas_39-2 | EMPTY | [0, +inf) | EMPTY |
| Haydn/Keyboard_Sonatas_32-1 | EMPTY | [0, +inf) | EMPTY |

- Non-empty positive piece interval: 8/13
- Impossible for every lambda>=0: 5/13
- Common interval over all 13: `EMPTY`
- Maximum simultaneous coverage: 8/13
- Lambda region attaining that coverage: `[0.1492606497, +inf)`

The last region comes from an exact endpoint/open-cell interval sweep. It is feasibility geometry, not a selected or best lambda.

## H/A complementarity

| scope | n | Pearson | Spearman |
| --- | --- | --- | --- |
| ALL_PIECE_SYSTEM | 117 | 0.853 | 0.960 |
| HUMAN | 13 | 0.840 | 0.890 |
| ORIGINAL_PT | 13 | 0.914 | 0.918 |
| CANONICAL_STAGE2 | 52 | 0.885 | 0.959 |
| ALWAYS_ON | 13 | 0.204 | 0.209 |
| CUSTOM_EVENT_V0 | 13 | 0.296 | 0.385 |

High pooled correlation does not imply functional redundancy: H divides by all pedal-induced relations, while A responds only to accumulated negative mass. The following fixed-threshold examples expose the two directions directly.

| example type | left | right | abs delta H | abs delta A |
| --- | --- | --- | --- | --- |
| SIMILAR_H_DIFFERENT_A | piece_3056e93b3e51b646/ALWAYS_ON | piece_daefdda4e1923cc6/ALWAYS_ON | 0.004918 | 2.552003 |
| SIMILAR_H_DIFFERENT_A | piece_08aa5ff2ada461e0/ALWAYS_ON | piece_e38184a2e9753bdc/ALWAYS_ON | 0.004962 | 2.224786 |
| SIMILAR_H_DIFFERENT_A | piece_3056e93b3e51b646/CUSTOM_EVENT_V0 | piece_de3f82957f1b3532/ORIGINAL_PT | 0.001717 | 2.203739 |
| SIMILAR_H_DIFFERENT_A | piece_08aa5ff2ada461e0/ALWAYS_ON | piece_de3f82957f1b3532/ALWAYS_ON | 0.002758 | 1.854830 |
| SIMILAR_H_DIFFERENT_A | piece_18921bc81d708d16/ALWAYS_ON | piece_3056e93b3e51b646/ALWAYS_ON | 0.002597 | 1.837414 |
| SIMILAR_A_DIFFERENT_H | piece_3056e93b3e51b646/HUMAN | piece_ab42b317dcb38fc5/CUSTOM_EVENT_V0 | 0.107955 | 0.007713 |
| SIMILAR_A_DIFFERENT_H | piece_3056e93b3e51b646/CUSTOM_EVENT_V0 | piece_de3f82957f1b3532/HYBRID_REGRESSION_ONLY | 0.102619 | 0.002900 |
| SIMILAR_A_DIFFERENT_H | piece_ab42b317dcb38fc5/HUMAN | piece_daefdda4e1923cc6/CUSTOM_EVENT_V0 | 0.081448 | 0.020592 |
| SIMILAR_A_DIFFERENT_H | piece_3056e93b3e51b646/HUMAN | piece_69862af5096ee3fa/CUSTOM_EVENT_V0 | 0.080505 | 0.016457 |
| SIMILAR_A_DIFFERENT_H | piece_db97fbaed2036f5b/ORIGINAL_PT | piece_de3f82957f1b3532/HUMAN | 0.078863 | 0.001577 |

- Similar-H criterion: absolute H difference <=0.005, ranked by A difference.
- Similar-A criterion: absolute A difference <=0.05, ranked by H difference.

## Normalized position

| system | H bin0 | H bin9 | corr(H,pos) | A bin0 | A bin9 | corr(A,pos) | A up-steps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HUMAN | 0.106249 | 0.112774 | 0.125 | 0.853706 | 1.033932 | 0.169 | 4/9 |
| ORIGINAL_PT | 0.110050 | 0.106975 | -0.402 | 1.214215 | 1.647891 | -0.069 | 3/9 |
| CANONICAL_STAGE2_MEDIAN | 0.079932 | 0.075237 | -0.374 | 0.849042 | 0.960044 | -0.330 | 3/9 |
| ALWAYS_ON | 0.266737 | 0.334775 | 0.838 | 6.403014 | 13.046968 | 0.906 | 9/9 |

Across individual ALWAYS_ON pieces, A_bin9>A_bin0 holds in 13/13, median corr(A,position)=0.907, and median A up-steps=9.0/9. Median corr(H,position)=0.700. Thus the five-piece accumulation pattern generalizes, while H can still show a position trend and should be described as bounded in magnitude rather than flat.

## Piece-level scale diagnostic

| metric | n | min | median | p95 | max |
| --- | --- | --- | --- | --- | --- |
| H | 117 | 0.000000 | 0.098571 | 0.322157 | 0.364988 |
| A | 117 | 0.000000 | 0.753491 | 11.296690 | 12.089873 |
| A_over_H_where_H_positive | 104 | 2.336710 | 8.634841 | 35.931528 | 39.462274 |

`A/H` uses H>0 rows only. No normalization, rescaling, z-score, or min-max transform was applied.

## Final questions

### Q1. Does A continue to capture accumulated Hall conflict?

Yes. ALWAYS_ON A grows from early to late position in 13/13 pieces, while A is driven by negative RAW rather than note count alone.

### Q2. How strongly do H and A separate ALWAYS_ON?

Canonical, PT, HUMAN, and CUSTOM comparisons are listed above. Both components robustly order normal systems below ALWAYS_ON; A additionally creates much larger multiplicative margins.

### Q3. Does A improve the separation margin?

Yes. The median Always/reference ratios in the separation table are consistently larger for A than H. This is margin improvement, not an ordering-count improvement where H is already saturated.

### Q4. How often is HUMAN<=PT?

H: 3/13; A: 8/13. A agrees with the desirable tendency more often, but neither component is a universal HUMAN/PT quality ranking.

### Q5. Quadrant counts?

Q1/Q2/Q3/Q4 = 3/5/0/5.

### Q6. Are Mephisto-like both-worse cases rare?

No. Although Mephisto itself is excluded as DEFERRED_LONG_FORM, the same quadrant-4 relation appears in 5/13 exact-computable pieces. It is common enough to inspect before fitting weights.

### Q7. Does harmonically cleaner Stage2 occur frequently?

Yes. Model<PT occurs H=34/52, A=35/52. This is compatible with a model using less pedal; it says nothing about sustain or connection quality.

### Q8. Are H and A sufficiently complementary?

Yes for retaining the two-component structure. Correlations are nonzero because both derive from RAW, but the concrete matched-H/matched-A examples and position behavior show distinct aggregation functions.

### Q9. Proceed to weight selection?

Not yet. Keep H+A, inspect the five quadrant-4 pieces first, then decide whether weight selection has a defensible validation protocol.

## Integrity

| check | status | detail |
| --- | --- | --- |
| H_reproduces_previous_Pure_Hall | PASS | piece_system_rows=117; max_abs_error=9.714e-17; tolerance=1e-15 |
| existing_RAW_reproduces_count_audit_overlap | PASS | rows=82152; max_abs_error=2.910e-11; tolerance=5e-11 |
| H_reproduces_count_audit_overlap | PASS | rows=82152; max_abs_error=0.000e+00 |
| A_equals_log1p_RAW | PASS | max_abs_error=0.000e+00 |
| H_equals_RAW_div_N_Q | PASS | valid_rows=97824; max_abs_error=2.220e-16 |
| Q_empty_implies_H_A_RAW_zero | PASS | rows=115704; max_abs_value=0.000e+00 |
| NO_PEDAL_exact_zero | PASS | rows=23362; max_abs_value=0.000e+00 |
| no_decay | PASS | CSV-only derivation; no evaluator or decay term called |
| no_velocity | PASS | velocity absent from score derivation |
| no_pedal_depth | PASS | no depth term |
| no_low_or_dynamic_terms | PASS | absent |
| no_PA_PP_differential_weighting | PASS | audited unweighted RAW reused |
| no_positive_Hall_reward | PASS | RAW=sum max(-HallWeight,0) |
| no_AA_pairs | PASS | AA_pair_count=0 |
| all_finite | PASS | onset_numeric_cells=2775864 |
| fixed_13_piece_inventory | PASS | piece_count=13 |
| fixed_9_system_inventory | PASS | rows=117; systems=9 |
| source_MIDI_unchanged | PASS | manifest_rows=117; unique_files=117 |
| TEST_access_zero | PASS | count=0 |
| inference_zero | PASS | count=0 |
| training_zero | PASS | count=0 |
| MIDI_generation_zero | PASS | count=0 |
| no_explicit_full_pair_materialization | PASS | Pure Hall aggregate onset CSV reused; Hall evaluator not called |
| CUSTOM_EVENT_excluded_from_canonical | PASS | canonical summary contains exactly four Stage2 systems |
| all_piece_system_summaries_finite | PASS | rows=117 |

All checks passed before report emission. Prior audit directories were read-only.
