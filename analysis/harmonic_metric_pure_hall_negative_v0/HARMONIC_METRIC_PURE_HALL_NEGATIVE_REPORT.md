# Pure Hall-Negative Pedaling Diagnostic v0

## Scope

This audit asks whether Hall Simple Type interval dissonance alone is a useful primitive for pedal-induced harmonic conflict. `Metric_harmonic_consonance.md` supplies the PA/PP note-state semantics and audited Hall mapping. Scoring uses only `d=max(-HallWeight,0)` with one unweighted sample per note-instance pair. It uses no decay, note age, performance intensity, low-register term, dynamic term, PA/PP differential weight, parameter search, or positive reward.

The input is exactly the prior 13 PEDAL_ELIGIBLE + EXACT_COMPUTABLE validation pieces and nine systems (117 stored MIDI files). PEDAL_SPARSE and DEFERRED_LONG_FORM pieces remain excluded. NO_PEDAL=0 is expected for a harm-only diagnostic and does not mean that no pedal is best overall. HUMAN and CUSTOM_EVENT_V0 remain reference distributions rather than strict canonical pedal-only comparisons.

## Table 1 — Piece-balanced system summary

| system | HALL_NEG_MEAN_VALID_mean | HALL_NEG_MEAN_VALID_median | HALL_NEG_MEAN_VALID_IQR | HALL_NEG_MEAN_ALL_mean | HALL_NEG_PAIR_FRACTION_VALID_mean | HALL_NEG_CONDITIONAL_VALID_mean | HALL_NEG_ONSET_RATE_mean | HALL_NEG_MAX_VALID_mean |
|---|---|---|---|---|---|---|---|---|
| NO_PEDAL | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| ALWAYS_ON | 0.320297 | 0.322235 | 0.034321 | 0.319929 | 0.565246 | 0.565151 | 0.998840 | 1.896360 |
| STANDARD_CE_ARGMAX | 0.232901 | 0.231680 | 0.079594 | 0.075738 | 0.494446 | 0.429571 | 0.364817 | 1.019674 |
| STANDARD_CE_POSTERIOR_MEDIAN | 0.227816 | 0.227198 | 0.084478 | 0.072721 | 0.499828 | 0.418365 | 0.356176 | 0.976509 |
| WEIGHTED_CE_ARGMAX | 0.240969 | 0.234293 | 0.097102 | 0.069135 | 0.500400 | 0.438191 | 0.337265 | 0.974013 |
| HYBRID_REGRESSION_ONLY | 0.225839 | 0.235227 | 0.077976 | 0.071654 | 0.497825 | 0.399961 | 0.344489 | 0.984963 |
| CUSTOM_EVENT_V0 | 0.253005 | 0.263591 | 0.052563 | 0.142620 | 0.514486 | 0.456133 | 0.556402 | 1.254724 |
| ORIGINAL_PT | 0.220236 | 0.227255 | 0.056715 | 0.101888 | 0.490440 | 0.417713 | 0.449684 | 1.098582 |
| HUMAN | 0.218644 | 0.231554 | 0.052374 | 0.120629 | 0.498466 | 0.406150 | 0.530253 | 1.041154 |

Pure `HALL_NEG_MEAN_VALID` places ALWAYS_ON at 0.320, above HUMAN 0.219, PT 0.220, and canonical system means 0.226–0.241. The extreme also has near-universal negative onset occurrence (0.999) and an average maximum conflict of 1.896, close to the Hall m2 maximum 1.902.

## Table 2 — Ordering by simple Hall view

| Hall view | HUMAN<PT (/13) | PT<model (/52) | model<ALWAYS (/52) | full canonical (/13) | full all (/13) |
|---|---|---|---|---|---|
| HALL_NEG_MEAN_VALID | 7 | 20 | 45 | 1 | 1 |
| HALL_NEG_PAIR_FRACTION_VALID | 6 | 24 | 41 | 2 | 2 |
| HALL_NEG_CONDITIONAL_VALID | 8 | 18 | 48 | 2 | 1 |
| HALL_NEG_ONSET_RATE | 2 | 16 | 52 | 2 | 2 |
| HALL_NEG_MAX_VALID | 9 | 21 | 52 | 2 | 2 |
| HALL_NEG_MEAN_ALL | 3 | 18 | 52 | 1 | 1 |

The primary mean gives HUMAN<PT 7/13, PT<canonical Stage2 20/52, model<ALWAYS_ON 45/52, and the full canonical chain 1/13. The only primary full-chain piece is piece_3056e93b3e51b646 (Schumann/Kreisleriana/6/JohannsonP07.mid).

Magnitude and sign frequency are complementary rather than interchangeable. Pair fraction improves PT<model from 20/52 to 24/52, but weakens HUMAN<PT and model<ALWAYS_ON. Conditional severity improves HUMAN<PT to 8/13 and model<ALWAYS_ON to 48/52, but PT<model falls to 18/52. Onset frequency catches model<ALWAYS_ON in 52/52 while providing only 2/13 and 16/52 at the normal-performance boundaries. Maximum conflict gives 9/13 and 52/52, but is a one-pair diagnostic that saturates easily.

## Table 3 — PA / PP Hall-only structure

| system | HALL_NEG_MEAN_PA_mean | HALL_NEG_MEAN_PP_mean | NEGATIVE_PAIR_FRACTION_PA_mean | NEGATIVE_PAIR_FRACTION_PP_mean | PA_negative_mass_fraction | PP_negative_mass_fraction |
|---|---|---|---|---|---|---|
| NO_PEDAL | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| ALWAYS_ON | 0.324320 | 0.320144 | 0.569185 | 0.565376 | 0.003084 | 0.996916 |
| STANDARD_CE_ARGMAX | 0.241769 | 0.236507 | 0.498419 | 0.467117 | 0.207283 | 0.792717 |
| STANDARD_CE_POSTERIOR_MEDIAN | 0.241098 | 0.202290 | 0.503040 | 0.468449 | 0.210646 | 0.789354 |
| WEIGHTED_CE_ARGMAX | 0.247584 | 0.228828 | 0.501137 | 0.471671 | 0.223477 | 0.776523 |
| HYBRID_REGRESSION_ONLY | 0.236906 | 0.236780 | 0.492486 | 0.534229 | 0.130751 | 0.869249 |
| CUSTOM_EVENT_V0 | 0.267127 | 0.234902 | 0.518135 | 0.505043 | 0.138614 | 0.861386 |
| ORIGINAL_PT | 0.241428 | 0.179314 | 0.500048 | 0.450675 | 0.122069 | 0.877931 |
| HUMAN | 0.236565 | 0.192846 | 0.502558 | 0.488400 | 0.283040 | 0.716960 |

## Table 4 — PA-only versus PP-only discrimination

| origin_view | HUMAN<PT (/13) | PT<model (/52) | model<ALWAYS (/52) | full chain (/13) |
|---|---|---|---|---|
| HALL_NEG_MEAN_PA | 7 | 26 | 46 | 1 |
| HALL_NEG_MEAN_PP | 6 | 21 | 48 | 1 |
| NEGATIVE_PAIR_FRACTION_PA | 7 | 25 | 43 | 2 |
| NEGATIVE_PAIR_FRACTION_PP | 6 | 22 | 43 | 2 |

PA mean is somewhat more informative for the normal PT/model boundary (26/52 versus PP 21/52). PP mean is slightly stronger for extreme separation (48/52 versus PA 46/52). Without decay, 99.7% of ALWAYS_ON raw negative Hall mass comes from PP because the residual-residual pair count grows combinatorially; its PA and PP mean severities are nevertheless similar (0.324 and 0.320).

## Synthetic Hall sanity

| case | N_pair | N_negative | HALL_NEG_MEAN | NEGATIVE_PAIR_FRACTION | CONDITIONAL_NEGATIVE | MAX_NEGATIVE |
|---|---|---|---|---|---|---|
| same_major_harmony | 12 | 4 | 0.044917 | 0.333333 | 0.134750 | 0.339000 |
| conflicting_major_harmonies_C_Db | 12 | 8 | 0.582000 | 0.666667 | 0.873000 | 1.902000 |
| chromatic_cluster | 22 | 14 | 0.458364 | 0.636364 | 0.720286 | 1.902000 |
| consonant_plus_one_m2 | 15 | 7 | 0.211000 | 0.466667 | 0.452143 | 1.902000 |

### Bad-note series

| bad_note_count | HALL_NEG_MEAN | NEGATIVE_PAIR_FRACTION | CONDITIONAL_NEGATIVE | MAX_NEGATIVE |
|---|---|---|---|---|
| 0.000000 | 0.044917 | 0.333333 | 0.134750 | 0.339000 |
| 1.000000 | 0.211000 | 0.466667 | 0.452143 | 1.902000 |
| 2.000000 | 0.274722 | 0.555556 | 0.494500 | 1.902000 |
| 3.000000 | 0.278286 | 0.523810 | 0.531273 | 1.902000 |

Same-major harmony has low but nonzero penalty (0.0449) because the learned Hall table assigns small negative values to some nominally consonant classes such as m3/M6. Conflicting major harmonies and the chromatic cluster rise to 0.582 and 0.458. Bad-note-count mean and conditional severity are monotone (True); maximum conflict reaches 1.902 after the first m2 and then saturates. Negative pair fraction is not fully monotone (False): at the third added pitch it decreases from 0.556 to 0.524 because the added note creates both negative and nonnegative relations, changing the denominator composition.

The real MIDI direction agrees at the extreme level: ALWAYS_ON has larger mean, conditional severity, maximum conflict, and occurrence than the normal systems. Synthetic monotonicity is diagnostic-dependent, showing why raw sign fraction alone is not an ordinal bad-note counter.

## Sanity and integrity

| check | status | detail |
|---|---|---|
| W_dec_not_used | PASS | pure evaluator performs pitch-count Hall lookup only |
| velocity_not_used_in_score | PASS | score reads note.pitch only |
| low_register_weight_not_used | PASS | absent |
| dynamic_weight_not_used | PASS | absent |
| no_PA_PP_differential_weight | PASS | each note-instance pair has multiplicity weight 1 |
| positive_Hall_gives_zero_not_reward | PASS | max_extracted_positive=0.000e+00 |
| all_extracted_pair_contributions_nonnegative | PASS | minimum=0.000e+00 |
| prior_pair_and_negative_sign_counts_exact | PASS | rows=213528; max_abs_count_error=0 |
| unweighted_negative_fraction_crosscheck | PASS | max_abs_error=0.000e+00 |
| no_AA_pairs | PASS | AA_pair_count=0 |
| NO_PEDAL_exact_zero | PASS | rows=23362; max_abs_score=0.000e+00 |
| all_finite | PASS | numeric_cells=4911144 |
| Hall_scores_in_defined_range | PASS | max_mean=1.902000; max_conditional=1.902000 |
| fractions_in_unit_interval | PASS | min=0.000000; max=1.000000 |
| synthetic_scores_finite_nonnegative | PASS | cases=8 |
| source_MIDI_SHA_unchanged | PASS | files=117 |
| TEST_access_zero | PASS | count=0 |
| inference_zero | PASS | count=0 |
| training_zero | PASS | count=0 |
| new_MIDI_generation_zero | PASS | count=0 |
| diagnostic_set_exact_13x9 | PASS | pieces=13; piece_systems=117 |
| scoring_source_has_no_modifier_calls | PASS | absent=7/7 |
| note_instance_multiplicity_histogram_exact | PASS | pairs=9; negative=4; mass=4.415 |

All 23 checks pass. The compact pitch-multiplicity calculation exactly reproduces all 213,528 prior onset PA/PP pair counts and unweighted negative Hall-sign counts (max count error 0; fraction error 0). Repeated same-pitch note instances also match naive combinatorics. NO_PEDAL is exact zero, all scores are finite and in the Hall-defined range, all 117 source SHA256 values are unchanged, and A-A pairs are absent. TEST access, inference, training, and MIDI generation are all zero.

## Answers to the research questions

**Q1 — Does Hall-negative magnitude distinguish ALWAYS_ON without modifiers?** Yes at the extreme level. Its mean penalty is 0.320 versus 0.226–0.241 for canonical Stage2 systems, and model<ALWAYS_ON holds in 45/52 cells. It is strong but not universal: the canonical median is below ALWAYS_ON in 11/13 pieces.

**Q2 — HUMAN versus PT.** HUMAN<PT holds in 7/13 under the primary mean. Their piece-balanced means are almost equal (0.219 versus 0.220), so the signal is not a robust HUMAN/PT quality separator.

**Q3 — PT versus Stage2.** PT<model holds in only 20/52 cells, below chance-level consistency. Pure Hall mean does not reproduce a stable PT/Stage2 quality distinction.

**Q4 — Is magnitude more informative than negative-pair fraction?** Neither dominates. Magnitude is better for HUMAN/PT and extreme separation; fraction is slightly better for PT/model. Conditional severity and maximum conflict supply additional extreme evidence, but no view creates stable normal-system ordering.

**Q5 — Frequency or severity?** Extreme detection appears in both: ALWAYS_ON conditional severity is 0.565 versus canonical 0.400–0.438, while onset frequency is 0.999 versus 0.337–0.365. Frequency is the clearest ALWAYS_ON detector (52/52) but the weakest HUMAN/PT detector (2/13). Severity is more useful for HUMAN/PT (8/13) but still weak for PT/model (18/52).

**Q6 — PA or PP?** PA gives more normal-system discrimination, while PP gives slightly stronger extreme separation and overwhelmingly dominates ALWAYS_ON raw mass due combinatorial residual accumulation. Both PA and PP mean severities rise for ALWAYS_ON, so the core pitch signal is not exclusive to one origin.

**Q7 — Synthetic versus real behavior.** They agree for strong conflicts: chromatic/conflicting pitch sets and ALWAYS_ON receive larger penalties. They do not support simple monotonic interpretation for every statistic; pair fraction can fall when a bad note also introduces nonnegative pairs, and max conflict saturates after one m2.

**Q8 — Should Hall negative remain a core primitive?** Yes, as an extreme pedal-induced conflict primitive, but not as a standalone normal-pedaling quality metric. The evidence is closest to **B: Hall-only is useful for extreme conflict detection but limited for normal pedaling quality discrimination**. This audit does not propose how to combine it with any other component.

## Interpretation limits

- No new formula or scalar combination is proposed or selected.
- Without decay, every residual note remains equally present until symbolic pedal release; PP combinatorics therefore dominate ALWAYS_ON raw mass.
- Pair-average severity, sign fraction, conditional severity, onset occurrence, and max conflict answer different questions and should not be conflated.
- CSV files are the source of truth; plots are interpretation aids.
