# Harmonic Metric Negative Contribution Aggregation Audit v0

## Scope and fixed semantics

`Metric_harmonic_consonance.md` is the sole metric source of truth. This audit reuses the positive/negative decomposition parser, note-instance tracking, CC64 threshold, PA/PP construction, Hall Simple Type weights, attack-based Lehtonen decay, and the existing reusable extremes. No parameter was searched, no MIDI was generated, and no formula-family term was changed.

Fixed configuration: `alpha_decay=1.0`, `eta_PA=1.0`, `eta_PP=0.9`, `m0=48`, `beta_low=0`, `kappa_dyn=0`. The diagnostic set is exactly the prior 13 PEDAL_ELIGIBLE + EXACT_COMPUTABLE validation pieces and nine systems (117 stored MIDI files). PEDAL_SPARSE and DEFERRED_LONG_FORM cases remain excluded. Larger D means worse pedal-induced negative burden; NO_PEDAL=0 is expected for this harm-only diagnostic.

HUMAN and CUSTOM_EVENT_V0 do not share the strict canonical Stage 1 non-CC64 stream, so their distributions are references rather than exact pedal-only paired comparisons.

## Synthetic mechanism sanity

| case | raw_pair_count | D_paircount | D_audible | D_mass | D_top1 | D_top3_mean | F_neg_paircount | F_neg_audible |
|---|---|---|---|---|---|---|---|---|
| A_one_strong | 1 | 1.902000 | 1.902000 | 1.902000 | 1.902000 | 1.902000 | 1.000000 | 1.000000 |
| B_strong_plus_1000_old | 1001 | 0.001900 | 1.902000 | 1.902000 | 1.902000 | 0.634000 | 0.500500 | 1.000000 |
| C_many_weak | 100 | 0.020000 | 0.100000 | 2.000000 | 0.020000 | 0.020000 | 1.000000 | 1.000000 |
| D_few_strong | 2 | 1.902000 | 1.902000 | 3.804000 | 1.902000 | 1.902000 | 1.000000 | 1.000000 |

Adding 1,000 almost-decayed relations to the same strong conflict reduces `D_paircount` from 1.902 to 0.001900, while `D_audible` remains 1.902 within 1e-9. This confirms the intended denominator contrast before inspecting MIDI data.

## Table 1 — Piece-balanced system mechanism summary

| system | raw_pair_count_valid_mean | audible_pair_mass_valid_mean | audible_to_raw_ratio_valid_mean | D_paircount_valid_mean | D_audible_valid_mean | D_mass_valid_mean | D_top1_valid_mean | F_neg_paircount_valid_mean | F_neg_audible_valid_mean | negative_onset_rate |
|---|---|---|---|---|---|---|---|---|---|---|
| NO_PEDAL | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| ALWAYS_ON | 680218.981167 | 500.316377 | 0.013472 | 0.003531 | 0.295410 | 145.493189 | 0.775693 | 0.565252 | 0.538374 | 0.998840 |
| STANDARD_CE_ARGMAX | 120.781649 | 29.754229 | 0.398645 | 0.091115 | 0.236317 | 5.707084 | 0.425122 | 0.494721 | 0.497332 | 0.364817 |
| STANDARD_CE_POSTERIOR_MEDIAN | 118.192825 | 29.510190 | 0.407669 | 0.092420 | 0.235792 | 5.518356 | 0.417986 | 0.500062 | 0.504893 | 0.356176 |
| WEIGHTED_CE_ARGMAX | 104.257927 | 27.473816 | 0.405691 | 0.096608 | 0.246179 | 5.497670 | 0.413163 | 0.500516 | 0.502780 | 0.337265 |
| HYBRID_REGRESSION_ONLY | 187.235164 | 33.079030 | 0.402015 | 0.094192 | 0.232152 | 6.492684 | 0.426823 | 0.497300 | 0.501081 | 0.344489 |
| CUSTOM_EVENT_V0 | 247.407460 | 41.225459 | 0.346028 | 0.090263 | 0.262318 | 11.329184 | 0.521382 | 0.514679 | 0.520003 | 0.556402 |
| ORIGINAL_PT | 277.617000 | 49.333483 | 0.386004 | 0.088831 | 0.227754 | 11.122317 | 0.485849 | 0.490955 | 0.496245 | 0.449684 |
| HUMAN | 72.558570 | 20.558281 | 0.358717 | 0.081064 | 0.227890 | 4.066338 | 0.407455 | 0.498695 | 0.503747 | 0.530253 |

ALWAYS_ON has about 680,219 raw pairs per valid onset versus PT 277.6, HUMAN 72.6, and canonical systems 104.3–187.2. Its audible/raw ratio is 0.0135, versus PT 0.3860, HUMAN 0.3587, and canonical systems 0.3986–0.4077. Its audible mass is not absolutely small (500.316); it is small relative to the enormous raw denominator.

Changing only the denominator raises ALWAYS_ON from `D_paircount=0.003531` to `D_audible=0.295410`, an 83.7x ratio. Other nonzero systems rise only 2.46–2.91x.

## Table 2 — Primary valid-onset ordering comparison

| aggregation | HUMAN<PT (/13) | PT<model (/52) | model<ALWAYS (/52) | full canonical (/13) | full all (/13) |
|---|---|---|---|---|---|
| D_paircount | 10 | 26 | 0 | 0 | 0 |
| D_audible | 6 | 24 | 42 | 0 | 0 |
| D_mass | 9 | 18 | 52 | 2 | 2 |
| D_top1 | 9 | 11 | 52 | 1 | 1 |
| D_top3_mean | 8 | 16 | 52 | 1 | 1 |
| D_top5_mean | 8 | 14 | 52 | 1 | 1 |

The first row exactly reproduces the prior audit reference: HUMAN<PT 10/13, PT<Stage2 26/52, Stage2<ALWAYS_ON 0/52, and full canonical 0/13. `D_audible` restores model<ALWAYS_ON to 42/52 but reduces HUMAN<PT to 6/13. Raw mass and all top-k variants give model<ALWAYS_ON=52/52; `D_mass` is the closest single aggregation to the full chain, but passes only 2/13.

## Table 3 — Boundary-specific valid-onset results

| aggregation | HUMAN<PT | PT<canonical_median | canonical_median<ALWAYS | full_chain |
|---|---|---|---|---|
| D_paircount | 10 | 7 | 0 | 0 |
| D_audible | 6 | 6 | 10 | 0 |
| D_mass | 9 | 5 | 13 | 2 |
| D_top1 | 9 | 3 | 13 | 1 |
| D_top3_mean | 8 | 4 | 13 | 1 |
| D_top5_mean | 8 | 4 | 13 | 1 |

## Table 4 — All-distinct-onset ordering

| aggregation | HUMAN<PT (/13) | PT<model (/52) | model<ALWAYS (/52) | full canonical (/13) |
|---|---|---|---|---|
| D_paircount | 6 | 17 | 13 | 0 |
| D_audible | 2 | 17 | 52 | 1 |
| D_mass | 8 | 16 | 52 | 2 |
| D_top1 | 7 | 15 | 52 | 2 |
| D_top3_mean | 8 | 15 | 52 | 2 |
| D_top5_mean | 8 | 12 | 52 | 2 |

All-onset averaging incorporates pedal coverage. It does not resolve the PT/model boundary: the best full-chain count remains 2/13. `D_audible_all` gives model<ALWAYS_ON=52/52 but HUMAN<PT only 2/13.

## Table 5 — PA / PP decomposition

| system | D_mass_PA_valid_mean | D_mass_PP_valid_mean | raw_mass_PP_fraction | D_audible_PA_valid_mean | D_audible_PP_valid_mean |
|---|---|---|---|---|---|
| ALWAYS_ON | 11.981534 | 133.511655 | 0.917649 | 0.295341 | 0.295287 |
| STANDARD_CE_ARGMAX | 1.714694 | 3.992390 | 0.699550 | 0.243806 | 0.192587 |
| STANDARD_CE_POSTERIOR_MEDIAN | 1.659825 | 3.858531 | 0.699218 | 0.247550 | 0.162049 |
| WEIGHTED_CE_ARGMAX | 1.665792 | 3.831878 | 0.697000 | 0.251586 | 0.184235 |
| HYBRID_REGRESSION_ONLY | 1.830875 | 4.661809 | 0.718010 | 0.241289 | 0.191044 |
| CUSTOM_EVENT_V0 | 2.388231 | 8.940954 | 0.789197 | 0.274111 | 0.213690 |
| ORIGINAL_PT | 2.447746 | 8.674571 | 0.779925 | 0.246351 | 0.164067 |
| HUMAN | 1.468392 | 2.597945 | 0.638891 | 0.244158 | 0.170744 |

For ALWAYS_ON, raw negative mass is 91.8% PP and 8.2% PA. The prior 77.7% figure used per-onset pair-count-normalized burden; the present 91.8% is a raw-mass decomposition and therefore weights dense onsets differently. Origin-specific audible severities are nearly equal for ALWAYS_ON (PA 0.295, PP 0.295); its pathological total is driven mainly by the enormous number and coverage of PP residual relations, not a uniquely harsher PP interval severity.

ALWAYS_ON `F_neg_paircount` falls from 0.565 to audible-weighted 0.538. The change is modest compared with the denominator effect. Its negative onset rate is 0.999, compared with PT 0.450 and HUMAN 0.530.

## Sanity and integrity

| check | status | detail |
|---|---|---|
| synthetic_old_pairs_dilute_paircount | PASS | A=1.902; B=0.00190009990105 |
| synthetic_old_pairs_preserve_audible_severity | PASS | A=1.902; B=1.90199999905 |
| synthetic_mass_topk_distinguish_density_severity | PASS | C_mass=2; C_top1=0.02; D_top1=1.902 |
| D_paircount_equals_previous_H_neg_onset | PASS | rows=213528; max_abs_error=2.220e-16 |
| q_equals_eta_Wdec_d | PASS | max_abs_error=0.000e+00 |
| W_dec_nonnegative | PASS | minimum_W_dec=0 |
| negative_statistics_nonnegative | PASS | minimum=0 |
| NO_PEDAL_all_negative_statistics_exact_zero | PASS | rows=23362; max_abs=0.000e+00 |
| all_scores_finite | PASS | numeric_cells=10462872 |
| no_AA_pairs | PASS | AA_pair_count=0 |
| source_MIDI_SHA_unchanged | PASS | files=117 |
| TEST_access_zero | PASS | count=0 |
| new_inference_zero | PASS | count=0 |
| training_zero | PASS | count=0 |
| diagnostic_set_exact_13x9 | PASS | pieces=13; piece_systems=117 |
| D_paircount_formula_reconstruction | PASS | max_abs_error=2.220e-16 |
| D_audible_formula_reconstruction | PASS | max_abs_error=2.338e-15 |
| topk_monotone_top1_ge_top3_ge_top5 | PASS | onsets=213528 |
| negative_fractions_in_unit_interval | PASS | min=0; max=1 |
| D_audible_within_Hall_negative_range | PASS | maximum=1.902 |
| required_output_files_present | PASS | 8/8 task-required CSV/report targets before report write |

All 21 checks pass. `D_paircount` matches every one of the prior 213,528 onset H_neg values within 2.22e-16; q reconstruction is exact; NO_PEDAL is exact zero; all values are finite; no A-A pair appears; all 117 input SHA256 values are unchanged. TEST access, inference, training, and MIDI generation counts are zero.

## Answers to the audit questions

**Q1 — Does the current denominator dilute ALWAYS_ON?** Yes, decisively. ALWAYS_ON combines an enormous mean raw eta denominator with an audible/raw ratio of only 0.0135. Its negative burden grows 83.7x when old relations are downweighted in the denominator, versus only about 2.5–2.9x for normal performance systems.

**Q2 — Does D_audible make ALWAYS_ON worse than normal pedal systems?** Mostly, not universally. Mean ALWAYS_ON D_audible=0.295, above HUMAN=0.228, PT=0.228, and canonical system means 0.232–0.246. Piece/model compliance is 42/52 and canonical-median<ALWAYS_ON is 10/13.

**Q3 — Which statistics expose ALWAYS_ON pathology?** Raw mass and top-k all recover model<ALWAYS_ON in 52/52 cells. Negative onset rate independently exposes near-continuous occurrence (99.9%). `D_mass` gives the best combined ordering evidence (full chain 2/13); top-k proves strong individual clashes survive averaging, while occurrence proves pathological coverage. None should be adopted here as a final metric.

**Q4 — Which aggregation is most stable for HUMAN<PT?** The current `D_paircount` remains best at 10/13. `D_mass` and `D_top1` reach 9/13; `D_audible` falls to 6/13. Audible normalization solves a different boundary and does not improve HUMAN/PT.

**Q5 — Does PT<Stage2 improve?** No. Macro compliance changes from 26/52 for `D_paircount` to 24/52 for `D_audible`, 18/52 for mass, and 11–16/52 for top-k. Normalization is not the cause of the weak PT/model separation.

**Q6 — Which single aggregation is closest to the full chain?** `D_mass` is closest: HUMAN<PT=9/13, PT<canonical median=5/13, canonical median<ALWAYS_ON=13/13, full chain=2/13. This remains far from stable and is density-sensitive by construction.

**Q7 — Is there evidence to separate severity and frequency?** Yes. Audible severity corrects denominator dilution but trades away HUMAN/PT separation; mass/top-k/occurrence expose ALWAYS_ON but weaken PT/model ordering. A future design should evaluate severity and occurrence/coverage as separate components before deciding whether and how to combine them. This audit does not make that design choice.

**Q8 — What best explains the ALWAYS_ON failure?** Positive cancellation was already ruled out as the main cause by the prior audit. Here denominator dilution is directly confirmed. The mechanism is its interaction with weak-decayed residuals and extreme pair-frequency structure: individual old pairs are weak, their raw count is enormous, audible mass still accumulates, and negative conflicts occur at nearly every onset. Pair-count normalization suppresses that accumulated pathology toward zero.

## Interpretation limits and provenance

- This is an aggregation mechanism audit, not parameter search or final metric selection.
- `D_mass` is texture/pair-count sensitive; top-k ignores much of the distribution; `D_audible` measures average audible severity but not coverage. Each creates a different trade-off.
- Every input MIDI path and SHA is preserved in `input_midis.csv`; no source or reusable extreme was modified.
- CSV files are the source of truth. Plots are interpretation aids only.
