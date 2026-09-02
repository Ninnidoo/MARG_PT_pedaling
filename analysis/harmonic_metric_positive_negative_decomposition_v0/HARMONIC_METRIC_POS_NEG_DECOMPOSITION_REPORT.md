# Harmonic Metric Positive / Negative Contribution Decomposition v0

## Scope and integrity

`Metric_harmonic_consonance.md` is the sole formula source. This audit reuses the pilot/broad note-instance parser, CC64 threshold, PA/PP construction, Hall Simple Type weights, attack-based Lehtonen decay, and the already-generated extremes. No metric term or parameter was tuned.

Primary configuration: `alpha_decay=1.0`, `eta_PP=0.9`, `m0=48`, `beta_low=0`, `kappa_dyn=0`. The 13 broad-sweep pieces marked PEDAL_ELIGIBLE and EXACT_COMPUTABLE are treated as one diagnostic set; prior split labels are metadata only.

`H_neg_valid` is a harm-only penalty, so larger is worse. NO_PEDAL=0 is expected and does not imply best overall pedaling quality.

## Table 1 — Piece-balanced system summary

| system | H_pos_valid_mean | H_pos_valid_median | H_pos_valid_IQR | H_neg_valid_mean | H_neg_valid_median | H_neg_valid_IQR | H_net_valid_mean | H_net_valid_median | H_net_valid_IQR | F_neg_valid_mean | F_neg_valid_median | H_neg_cond_valid_mean | H_neg_cond_valid_median | cancellation_valid_mean | negative_onset_rate_mean | H_neg_PA_valid_mean | H_neg_PP_valid_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NO_PEDAL | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| ALWAYS_ON | 0.004615 | 0.003677 | 0.002659 | 0.003531 | 0.002981 | 0.002381 | 0.001084 | 0.001022 | 0.000944 | 0.565252 | 0.570379 | 0.006908 | 0.005656 | 0.003326 | 0.998840 | 0.000788 | 0.002743 |
| STANDARD_CE_ARGMAX | 0.139954 | 0.138086 | 0.047949 | 0.091115 | 0.090316 | 0.038886 | 0.048839 | 0.046414 | 0.059340 | 0.494721 | 0.495948 | 0.170739 | 0.175075 | 0.061238 | 0.364817 | 0.060987 | 0.030128 |
| STANDARD_CE_POSTERIOR_MEDIAN | 0.140684 | 0.137030 | 0.047184 | 0.092420 | 0.083918 | 0.047347 | 0.048264 | 0.047048 | 0.063640 | 0.500062 | 0.495592 | 0.172547 | 0.171428 | 0.061212 | 0.356176 | 0.064298 | 0.028122 |
| WEIGHTED_CE_ARGMAX | 0.140844 | 0.137708 | 0.041638 | 0.096608 | 0.088698 | 0.044838 | 0.044236 | 0.027861 | 0.072196 | 0.500516 | 0.502820 | 0.179173 | 0.175127 | 0.061491 | 0.337265 | 0.066387 | 0.030221 |
| HYBRID_REGRESSION_ONLY | 0.138944 | 0.134954 | 0.033816 | 0.094192 | 0.101372 | 0.048447 | 0.044752 | 0.030313 | 0.061816 | 0.497300 | 0.503720 | 0.176880 | 0.172521 | 0.059981 | 0.344489 | 0.062436 | 0.031755 |
| CUSTOM_EVENT_V0 | 0.117550 | 0.107196 | 0.035087 | 0.090263 | 0.086237 | 0.030525 | 0.027288 | 0.014589 | 0.032600 | 0.514679 | 0.530869 | 0.167221 | 0.156081 | 0.062292 | 0.556402 | 0.053644 | 0.036619 |
| ORIGINAL_PT | 0.138011 | 0.128619 | 0.039751 | 0.088831 | 0.086045 | 0.032808 | 0.049180 | 0.042574 | 0.057230 | 0.490955 | 0.503153 | 0.175789 | 0.161243 | 0.059793 | 0.449684 | 0.056412 | 0.032418 |
| HUMAN | 0.123313 | 0.115887 | 0.058905 | 0.081064 | 0.075424 | 0.032111 | 0.042249 | 0.032615 | 0.031854 | 0.498695 | 0.503528 | 0.158489 | 0.146487 | 0.053986 | 0.530253 | 0.054985 | 0.026079 |

## Table 1b — All-distinct-onset aggregation

| system | H_pos_all_onsets_mean | H_neg_all_onsets_mean | H_net_all_onsets_mean | negative_onset_rate_mean |
|---|---|---|---|---|
| NO_PEDAL | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| ALWAYS_ON | 0.004609 | 0.003527 | 0.001083 | 0.998840 |
| STANDARD_CE_ARGMAX | 0.053108 | 0.028239 | 0.024869 | 0.364817 |
| STANDARD_CE_POSTERIOR_MEDIAN | 0.052431 | 0.027634 | 0.024797 | 0.356176 |
| WEIGHTED_CE_ARGMAX | 0.050374 | 0.026785 | 0.023589 | 0.337265 |
| HYBRID_REGRESSION_ONLY | 0.050208 | 0.026034 | 0.024174 | 0.344489 |
| CUSTOM_EVENT_V0 | 0.067543 | 0.049601 | 0.017943 | 0.556402 |
| ORIGINAL_PT | 0.060377 | 0.035893 | 0.024484 | 0.449684 |
| HUMAN | 0.061176 | 0.039043 | 0.022133 | 0.530253 |

Q-empty onsets are zero here. ALWAYS_ON is barely diluted because negative pairs occur at almost every onset; other systems are diluted in proportion to their lower pedal-interaction coverage. This is an occurrence diagnostic and does not replace the valid-onset metric.

## Table 2 — Net versus negative-only ordering

| comparison | target_system_or_group | piece_or_cell_count | net_desired_order_pass_count | net_desired_order_rate | negative_only_desired_order_pass_count | negative_only_desired_order_rate | negative_minus_net_rate_change |
|---|---|---|---|---|---|---|---|
| HUMAN_vs_ORIGINAL_PT | ORIGINAL_PT | 13 | 4 | 0.307692 | 10 | 0.769231 | 0.461538 |
| ORIGINAL_PT_vs_STAGE2 | STANDARD_CE_ARGMAX | 13 | 7 | 0.538462 | 6 | 0.461538 | -0.076923 |
| STAGE2_vs_ALWAYS_ON | STANDARD_CE_ARGMAX | 13 | 12 | 0.923077 | 0 | 0.000000 | -0.923077 |
| ORIGINAL_PT_vs_STAGE2 | STANDARD_CE_POSTERIOR_MEDIAN | 13 | 6 | 0.461538 | 7 | 0.538462 | 0.076923 |
| STAGE2_vs_ALWAYS_ON | STANDARD_CE_POSTERIOR_MEDIAN | 13 | 10 | 0.769231 | 0 | 0.000000 | -0.769231 |
| ORIGINAL_PT_vs_STAGE2 | WEIGHTED_CE_ARGMAX | 13 | 7 | 0.538462 | 7 | 0.538462 | 0.000000 |
| STAGE2_vs_ALWAYS_ON | WEIGHTED_CE_ARGMAX | 13 | 11 | 0.846154 | 0 | 0.000000 | -0.846154 |
| ORIGINAL_PT_vs_STAGE2 | HYBRID_REGRESSION_ONLY | 13 | 6 | 0.461538 | 6 | 0.461538 | 0.000000 |
| STAGE2_vs_ALWAYS_ON | HYBRID_REGRESSION_ONLY | 13 | 10 | 0.769231 | 0 | 0.000000 | -0.769231 |
| ORIGINAL_PT_vs_STAGE2_MACRO | MODEL_CANONICAL | 52 | 26 | 0.500000 | 26 | 0.500000 | 0.000000 |
| STAGE2_vs_ALWAYS_ON_MACRO | MODEL_CANONICAL | 52 | 43 | 0.826923 | 0 | 0.000000 | -0.826923 |
| FULL_ORDERING | MODEL_CANONICAL | 13 | 1 | 0.076923 | 0 | 0.000000 | -0.076923 |
| FULL_ORDERING | MODEL_ALL | 13 | 1 | 0.076923 | 0 | 0.000000 | -0.076923 |

## Table 3 — Cancellation summary

| system | H_pos_valid | H_neg_valid | abs_H_net_valid | cancellation_valid | cancellation_ratio_piece |
|---|---|---|---|---|---|
| ALWAYS_ON | 0.004615 | 0.003531 | 0.001174 | 0.003326 | 0.864642 |
| CUSTOM_EVENT_V0 | 0.117550 | 0.090263 | 0.028734 | 0.062292 | 0.867040 |
| HUMAN | 0.123313 | 0.081064 | 0.042518 | 0.053986 | 0.802396 |
| HYBRID_REGRESSION_ONLY | 0.138944 | 0.094192 | 0.049935 | 0.059981 | 0.780219 |
| NO_PEDAL | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| ORIGINAL_PT | 0.138011 | 0.088831 | 0.049180 | 0.059793 | 0.792623 |
| STANDARD_CE_ARGMAX | 0.139954 | 0.091115 | 0.058195 | 0.061238 | 0.749177 |
| STANDARD_CE_POSTERIOR_MEDIAN | 0.140684 | 0.092420 | 0.058300 | 0.061212 | 0.750181 |
| WEIGHTED_CE_ARGMAX | 0.140844 | 0.096608 | 0.056649 | 0.061491 | 0.756460 |

## Table 4 — PA versus PP negative burden

| system | H_neg_PA_valid_mean | H_neg_PP_valid_mean | H_neg_PA_fraction | H_neg_PP_fraction |
|---|---|---|---|---|
| NO_PEDAL | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| ALWAYS_ON | 0.000788 | 0.002743 | 0.223234 | 0.776766 |
| STANDARD_CE_ARGMAX | 0.060987 | 0.030128 | 0.669339 | 0.330661 |
| STANDARD_CE_POSTERIOR_MEDIAN | 0.064298 | 0.028122 | 0.695712 | 0.304288 |
| WEIGHTED_CE_ARGMAX | 0.066387 | 0.030221 | 0.687180 | 0.312820 |
| HYBRID_REGRESSION_ONLY | 0.062436 | 0.031755 | 0.662864 | 0.337136 |
| CUSTOM_EVENT_V0 | 0.053644 | 0.036619 | 0.594306 | 0.405694 |
| ORIGINAL_PT | 0.056412 | 0.032418 | 0.635054 | 0.364946 |
| HUMAN | 0.054985 | 0.026079 | 0.678295 | 0.321705 |

## Table 5 — Frequency versus conditional severity

| system | piece_count | mean_H_neg_valid | mean_F_neg_valid | mean_H_neg_cond_valid | pearson_H_neg_vs_F_neg | pearson_H_neg_vs_conditional_severity |
|---|---|---|---|---|---|---|
| ALWAYS_ON | 13 | 0.003531 | 0.565252 | 0.006908 | -0.234077 | 0.984258 |
| CUSTOM_EVENT_V0 | 13 | 0.090263 | 0.514679 | 0.167221 | 0.657588 | 0.973045 |
| HUMAN | 13 | 0.081064 | 0.498695 | 0.158489 | 0.568493 | 0.804324 |
| HYBRID_REGRESSION_ONLY | 13 | 0.094192 | 0.497300 | 0.176880 | 0.775822 | 0.959486 |
| NO_PEDAL | 13 | 0.000000 | 0.000000 | 0.000000 | NA | NA |
| ORIGINAL_PT | 13 | 0.088831 | 0.490955 | 0.175789 | 0.660582 | 0.870091 |
| STANDARD_CE_ARGMAX | 13 | 0.091115 | 0.494721 | 0.170739 | 0.826666 | 0.967570 |
| STANDARD_CE_POSTERIOR_MEDIAN | 13 | 0.092420 | 0.500062 | 0.172547 | 0.889368 | 0.953105 |
| WEIGHTED_CE_ARGMAX | 13 | 0.096608 | 0.500516 | 0.179173 | 0.789350 | 0.974337 |

## Table 6 — Automatically selected failure-onset pieces

| selection_order | piece_id | performance_id | composer | title | selection_reason |
|---|---|---|---|---|---|
| 1 | piece_e7ad71b998a8d477 | Haydn/Keyboard_Sonatas/32-1/Pavlovic02.mid | Haydn | Keyboard_Sonatas_32-1 | highest ALWAYS_ON cancellation_ratio |
| 2 | piece_de3f82957f1b3532 | Ravel/Pavane/ChenS03.mid | Ravel | Pavane | largest absolute HUMAN–ORIGINAL_PT negative-burden gap |
| 3 | piece_0a888a7d5fbd06e6 | Haydn/Keyboard_Sonatas/46-1/Bach01.mid | Haydn | Keyboard_Sonatas_46-1 | largest absolute ORIGINAL_PT–canonical-model-median negative-burden gap |
| 4 | piece_3056e93b3e51b646 | Schumann/Kreisleriana/6/JohannsonP07.mid | Schumann | Kreisleriana_6 | broad-sweep problem case: Schumann/Kreisleriana_6 |

## Sanity checks

| check | status | detail |
|---|---|---|
| pair_c_equals_pos_minus_neg | PASS | max_abs_error=0.000e+00 |
| valid_onset_Hbase_equals_Hpos_minus_Hneg | PASS | max_abs_error=1.665e-16 |
| piece_backbone_equals_mean_net | PASS | max_abs_error=9.975e-17 |
| NO_PEDAL_all_components_exact_zero | PASS | onsets=23362 pieces=13 |
| all_scores_finite | PASS | onset_numeric_cells=5551728; piece_numeric_cells=3042 |
| no_AA_pairs | PASS | AA_pair_count=0 |
| pair_count_formula | PASS | PA=\|P\|*\|A\|; PP=choose(\|P\|,2) |
| source_MIDI_SHA_unchanged | PASS | files=117 |
| TEST_access_zero | PASS | count=0 |
| new_inference_zero | PASS | count=0 |
| training_zero | PASS | count=0 |
| diagnostic_set_exact_13 | PASS | pieces=13 piece_systems=117 |

## Answers to the diagnostic questions

**Q1. Is ALWAYS_ON hidden by cancellation?** No in the hypothesized absolute-mass sense. ALWAYS_ON has mean H_pos=0.004615, H_neg=0.003531, net=0.001084, and mean piece cancellation ratio=0.865. Its proportional cancellation is high, but both components are roughly 20–40 times smaller than the performance systems; 0.0% of ALWAYS_ON pieces meet the descriptive joint flag (component sum at or above the non-NO_PEDAL median and cancellation ratio >=0.8). Its near-zero net is primarily a small-contribution result, not hidden large opposing mass.

**Q2. Does negative-only recover the target penalty ordering?** No. The canonical full ordering changes from 7.7% under Net H to 0.0% under H_neg; MODEL_ALL changes from 7.7% to 0.0%. Negative-only improves HUMAN versus PT but reverses the intended model-versus-ALWAYS_ON boundary.

**Q3. HUMAN versus PT.** Desired HUMAN>PT under net passes 4/13; desired HUMAN<PT penalty passes 10/13. Mean H_neg is HUMAN=0.081064, PT=0.088831. This boundary improves by +46.2%.

**Q4. PT versus Stage2.** Across 52 canonical model/piece cells, desired ordering changes from 50.0% to 50.0%. Models versus ALWAYS_ON changes from 82.7% to 0.0%; negative-only does not improve the PT–model boundary and completely fails the model–ALWAYS boundary.

**Q5. PA or PP origin?** Mean ALWAYS_ON normalized burden is PA=0.000788, PP=0.002743; HUMAN is PA=0.054985, PP=0.026079; PT is PA=0.056412, PP=0.032418. Table 4 gives every system's component fractions.

**Q6. Frequency or severity?** ALWAYS_ON has the highest mean F_neg (0.565) and negative-onset rate (0.999), but by far the lowest nonzero conditional severity (0.006908). Across pieces, corr(H_neg,F_neg)=-0.234 while corr(H_neg,conditional severity)=0.984. HUMAN correlations are 0.568 and 0.804. ALWAYS_ON creates negative relations almost everywhere, but attack-based decay plus pair normalization makes each relation extremely weak; severity, not frequency, controls its small H_neg.

**Q7. Is cancellation the formula-family failure?** No. Negative-only full-chain improvement is -7.7% for MODEL_CANONICAL and the resulting rate is zero. Positive cancellation is not the principal explanation for the broad-search failure: the separated burden itself assigns ALWAYS_ON the smallest nonzero H_neg and only partly distinguishes HUMAN from PT/models.

## Provenance and caveats

- MODEL_CANONICAL is the same four strict canonical-stream Stage2 systems as the broad audit; MODEL_ALL adds CUSTOM_EVENT_V0.
- HUMAN and CUSTOM_EVENT_V0 do not share strict canonical Stage1 timing/velocity and are reference distributions rather than pedal-only pairs.
- PEDAL_SPARSE pieces and four DEFERRED_LONG_FORM pieces were not added. No approximation or pruning was introduced.
- Optional frozen-config robustness was not run: this task remains a single reference-backbone decomposition and performs no parameter selection.
- Every input path and unchanged SHA is recorded in `input_midis.csv`; `provenance.json` records TEST access=0, inference=0, and training=0.
- `onset_decomposition.csv` includes all distinct onsets; Q-empty onsets are explicit zeros. `top_negative_onsets.csv` and `top_negative_pairs.csv` contain the selected direct-inspection examples.
- Independent final integrity audit: 8/8 PASS (`integrity_audit.csv`).
