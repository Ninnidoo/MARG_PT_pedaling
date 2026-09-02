# Harmonic Metric Broad Parameter Search v0

## Scope and integrity

`Metric_harmonic_consonance.md` was the sole formula source. The audited pilot parser, note-instance semantics, Hall weights, and extreme construction were reused. ASAP test access, training, and new inference were all zero.

The full 4,320-point coarse grid was evaluated on the exact-computable PARAMETER_SEARCH pieces. The split was fixed before harmonic scoring with rule `exhaustive held-out subset minimizing standardized mean+SD imbalance; lexicographic tie-break` and seed `20260820`. Held-out scores were computed only after 7 configurations had been frozen from SEARCH; the final listening shortlist contains 4 of those pre-frozen configurations.

Exactness audit: 29/29 PASS; maximum optimized-vs-naive absolute difference `3.553e-15`.

## Evaluated system inventory

| System | Representative experiment / source |
|---|---|
| `ORIGINAL_PT` | canonical Stage 1 baseline; `analysis/stage2_binary_canonical_v1/canonical_validation_stage1/` |
| `STANDARD_CE_ARGMAX` | `analysis/stage2_encoder_only_4class_decoding_phase4_v0/predictions/argmax/` |
| `STANDARD_CE_POSTERIOR_MEDIAN` | `analysis/stage2_encoder_only_4class_decoding_phase4_v0/predictions/median/` |
| `WEIGHTED_CE_ARGMAX` | `analysis/stage2_encoder_only_4class_loss_phase3_v0/weighted_ce/validation/predictions/weighted_ce/` |
| `HYBRID_REGRESSION_ONLY` | `analysis/stage2_encoder_only_raw_huber_aux_ce_v1/validation_eval/predictions/raw_huber_aux_ce/` |
| `CUSTOM_EVENT_V0` | `analysis/custom_event_model_v0_canonical_val_inference_v1/candidate_midi/` |
| `HUMAN` | corresponding ASAP validation performance MIDI |
| `NO_PEDAL`, `ALWAYS_ON` | canonical Stage 1-derived reusable extremes; exact locations and SHA in `reusable_extremes_manifest.csv` |

These are the representative outputs retained by `Progress.md` and the previous pilot, rather than an expansion over historical checkpoints. Every concrete input MIDI path and SHA is recorded in `input_midis.csv`.

## Table 1 — PEDAL_ELIGIBLE / PEDAL_SPARSE

| piece_id | piece | human_valid_onset_count | human_valid_onset_fraction | CC64_ON_duty_ratio | sustain_ON_episode_count | primary_status | primary_reason | exact_compute_status |
|---|---|---|---|---|---|---|---|---|
| piece_08aa5ff2ada461e0 | Bach/Prelude_bwv_862 | 37 | 0.055807 | 0.105803 | 14 | PEDAL_ELIGIBLE | all primary criteria pass | EXACT_COMPUTABLE |
| piece_0a888a7d5fbd06e6 | Haydn/Keyboard_Sonatas_46-1 | 167 | 0.062664 | 0.109495 | 98 | PEDAL_ELIGIBLE | all primary criteria pass | EXACT_COMPUTABLE |
| piece_18921bc81d708d16 | Beethoven/Piano_Sonatas_9-3 | 417 | 0.236261 | 0.339865 | 107 | PEDAL_ELIGIBLE | all primary criteria pass | EXACT_COMPUTABLE |
| piece_3056e93b3e51b646 | Schumann/Kreisleriana_6 | 620 | 0.899855 | 0.835670 | 126 | PEDAL_ELIGIBLE | all primary criteria pass | EXACT_COMPUTABLE |
| piece_4b31472ad4376d9e | Liszt/Gran_Etudes_de_Paganini_6_Theme_and_Variations | 1892 | 0.409258 | 0.479233 | 204 | PEDAL_ELIGIBLE | all primary criteria pass | DEFERRED_LONG_FORM |
| piece_555ff6cd3a13efb9 | Scriabin/Etudes_op_8_11 | 1261 | 0.961128 | 0.944052 | 126 | PEDAL_ELIGIBLE | all primary criteria pass | EXACT_COMPUTABLE |
| piece_69862af5096ee3fa | Beethoven/Piano_Sonatas_27-1 | 1577 | 0.622828 | 0.782678 | 379 | PEDAL_ELIGIBLE | all primary criteria pass | EXACT_COMPUTABLE |
| piece_7195bbce81550519 | Liszt/Mephisto_Waltz | 5459 | 0.599100 | 0.672599 | 467 | PEDAL_ELIGIBLE | all primary criteria pass | DEFERRED_LONG_FORM |
| piece_759b324c1cceac54 | Schubert/Piano_Sonatas_664-1 | 1601 | 0.687124 | 0.741182 | 296 | PEDAL_ELIGIBLE | all primary criteria pass | EXACT_COMPUTABLE |
| piece_901ec1910fe9fa9e | Liszt/Concert_Etude_S145_2 | 1305 | 0.437186 | 0.399994 | 138 | PEDAL_ELIGIBLE | all primary criteria pass | DEFERRED_LONG_FORM |
| piece_ab42b317dcb38fc5 | Beethoven/Piano_Sonatas_29-2 | 709 | 0.388493 | 0.431327 | 149 | PEDAL_ELIGIBLE | all primary criteria pass | EXACT_COMPUTABLE |
| piece_b06ab236f236fd76 | Bach/Prelude_bwv_888 | 31 | 0.048666 | 0.045601 | 14 | PEDAL_SPARSE | valid_fraction<0.05; duty_ratio<0.05 | EXACT_COMPUTABLE |
| piece_b45364ab6817194a | Beethoven/Piano_Sonatas_8-1 | 1653 | 0.388850 | 0.540995 | 347 | PEDAL_ELIGIBLE | all primary criteria pass | DEFERRED_LONG_FORM |
| piece_daefdda4e1923cc6 | Schumann/Arabeske | 2071 | 0.821825 | 0.876902 | 366 | PEDAL_ELIGIBLE | all primary criteria pass | EXACT_COMPUTABLE |
| piece_db97fbaed2036f5b | Chopin/Etudes_op_25_12 | 2048 | 0.803137 | 0.831716 | 109 | PEDAL_ELIGIBLE | all primary criteria pass | EXACT_COMPUTABLE |
| piece_de3f82957f1b3532 | Ravel/Pavane | 1601 | 0.933528 | 0.937122 | 156 | PEDAL_ELIGIBLE | all primary criteria pass | EXACT_COMPUTABLE |
| piece_e38184a2e9753bdc | Haydn/Keyboard_Sonatas_39-2 | 1025 | 0.436728 | 0.503639 | 252 | PEDAL_ELIGIBLE | all primary criteria pass | EXACT_COMPUTABLE |
| piece_e7ad71b998a8d477 | Haydn/Keyboard_Sonatas_32-1 | 527 | 0.252879 | 0.393902 | 369 | PEDAL_ELIGIBLE | all primary criteria pass | EXACT_COMPUTABLE |
| piece_f10784ae79f0f210 | Bach/Prelude_bwv_856 | 4 | 0.005000 | 0.009516 | 1 | PEDAL_SPARSE | valid_onsets<20; valid_fraction<0.05; duty_ratio<0.05 | EXACT_COMPUTABLE |

The 0.05 thresholds are calibration eligibility rules, not a musical definition. Primary: 17 eligible, 2 sparse. 4 eligible long-form pieces were deferred by the pre-score exact-work limit of 3,000,000,000 pair-onsets; their IDs and workloads remain in the table and are not silently treated as failures.

## Table 2 — Search / held-out split

| piece_id | composer | title | split_role | exact_compute_status | low_note_fraction | velocity_range | note_density_per_second | human_valid_onset_fraction | CC64_ON_duty_ratio |
|---|---|---|---|---|---|---|---|---|---|
| piece_0a888a7d5fbd06e6 | Haydn | Keyboard_Sonatas_46-1 | PARAMETER_SEARCH | EXACT_COMPUTABLE | 0.047961 | 88 | 9.969389 | 0.062664 | 0.109495 |
| piece_759b324c1cceac54 | Schubert | Piano_Sonatas_664-1 | PARAMETER_SEARCH | EXACT_COMPUTABLE | 0.135180 | 94 | 8.078022 | 0.687124 | 0.741182 |
| piece_ab42b317dcb38fc5 | Beethoven | Piano_Sonatas_29-2 | HELD_OUT_CHECK | EXACT_COMPUTABLE | 0.190678 | 104 | 13.672878 | 0.388493 | 0.431327 |
| piece_daefdda4e1923cc6 | Schumann | Arabeske | HELD_OUT_CHECK | EXACT_COMPUTABLE | 0.081298 | 84 | 9.584953 | 0.821825 | 0.876902 |
| piece_db97fbaed2036f5b | Chopin | Etudes_op_25_12 | PARAMETER_SEARCH | EXACT_COMPUTABLE | 0.157935 | 86 | 20.497643 | 0.803137 | 0.831716 |
| piece_e7ad71b998a8d477 | Haydn | Keyboard_Sonatas_32-1 | PARAMETER_SEARCH | EXACT_COMPUTABLE | 0.062124 | 74 | 8.385977 | 0.252879 | 0.393902 |
| piece_08aa5ff2ada461e0 | Bach | Prelude_bwv_862 | HELD_OUT_CHECK | EXACT_COMPUTABLE | 0.041667 | 69 | 6.581780 | 0.055807 | 0.105803 |
| piece_18921bc81d708d16 | Beethoven | Piano_Sonatas_9-3 | PARAMETER_SEARCH | EXACT_COMPUTABLE | 0.206723 | 89 | 10.029213 | 0.236261 | 0.339865 |
| piece_3056e93b3e51b646 | Schumann | Kreisleriana_6 | PARAMETER_SEARCH | EXACT_COMPUTABLE | 0.186557 | 87 | 5.574831 | 0.899855 | 0.835670 |
| piece_4b31472ad4376d9e | Liszt | Gran_Etudes_de_Paganini_6_Theme_and_Variations | DEFERRED_LONG_FORM | DEFERRED_LONG_FORM | 0.124268 | 94 | 16.703720 | 0.409258 | 0.479233 |
| piece_555ff6cd3a13efb9 | Scriabin | Etudes_op_8_11 | PARAMETER_SEARCH | EXACT_COMPUTABLE | 0.108147 | 85 | 7.219821 | 0.961128 | 0.944052 |
| piece_69862af5096ee3fa | Beethoven | Piano_Sonatas_27-1 | PARAMETER_SEARCH | EXACT_COMPUTABLE | 0.118631 | 96 | 8.961817 | 0.622828 | 0.782678 |
| piece_7195bbce81550519 | Liszt | Mephisto_Waltz | DEFERRED_LONG_FORM | DEFERRED_LONG_FORM | 0.119780 | 94 | 19.491314 | 0.599100 | 0.672599 |
| piece_901ec1910fe9fa9e | Liszt | Concert_Etude_S145_2 | DEFERRED_LONG_FORM | DEFERRED_LONG_FORM | 0.104992 | 89 | 18.950877 | 0.437186 | 0.399994 |
| piece_b45364ab6817194a | Beethoven | Piano_Sonatas_8-1 | DEFERRED_LONG_FORM | DEFERRED_LONG_FORM | 0.176714 | 102 | 11.948550 | 0.388850 | 0.540995 |
| piece_de3f82957f1b3532 | Ravel | Pavane | HELD_OUT_CHECK | EXACT_COMPUTABLE | 0.102974 | 85 | 11.285184 | 0.933528 | 0.937122 |
| piece_e38184a2e9753bdc | Haydn | Keyboard_Sonatas_39-2 | PARAMETER_SEARCH | EXACT_COMPUTABLE | 0.018156 | 79 | 13.065639 | 0.436728 | 0.503639 |

## Table 3 — Hall+decay backbone system distribution (SEARCH)

| system | piece_count | mean_H_piece | median_H_piece | IQR_H_piece |
|---|---|---|---|---|
| ALWAYS_ON | 9 | 0.001140 | 0.000498 | 0.000881 |
| CUSTOM_EVENT_V0 | 9 | 0.021120 | 0.009753 | 0.011567 |
| HUMAN | 9 | 0.032656 | 0.020634 | 0.021715 |
| HYBRID_REGRESSION_ONLY | 9 | 0.035699 | 0.020630 | 0.051226 |
| NO_PEDAL | 9 | 0.000000 | 0.000000 | 0.000000 |
| ORIGINAL_PT | 9 | 0.042318 | 0.021711 | 0.042617 |
| STANDARD_CE_ARGMAX | 9 | 0.037630 | 0.026072 | 0.031742 |
| STANDARD_CE_POSTERIOR_MEDIAN | 9 | 0.042660 | 0.027162 | 0.036744 |
| WEIGHTED_CE_ARGMAX | 9 | 0.032278 | 0.027555 | 0.044644 |

## Table 4 — SEARCH top 20 configurations

| alpha_decay | eta_PP | m0 | beta_low | kappa_dyn | canonical_chain_pass_rate | all_model_chain_pass_rate | individual_model_compliance | human_gt_pt_rate | pt_gt_model_rate | model_gt_no_pedal_rate | model_gt_always_rate | median_gap_extreme_model | median_gap_model_pt | median_gap_pt_human |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.000000 | 0.000000 | 60 | 0.250000 | 0.300000 | 0.000000 | 0.000000 | 0.555556 | 0.333333 | 0.688889 | 0.866667 | 0.866667 | 0.035672 | 0.003802 | -0.007644 |
| 1.000000 | 0.000000 | 60 | 0.500000 | 0.300000 | 0.000000 | 0.000000 | 0.555556 | 0.333333 | 0.688889 | 0.866667 | 0.866667 | 0.035593 | 0.003760 | -0.008118 |
| 0.750000 | 0.000000 | 60 | 0.500000 | 0.300000 | 0.000000 | 0.000000 | 0.555556 | 0.333333 | 0.688889 | 0.866667 | 0.866667 | 0.035021 | 0.003820 | -0.007751 |
| 0.750000 | 0.000000 | 60 | 0.750000 | 0.300000 | 0.000000 | 0.000000 | 0.555556 | 0.333333 | 0.688889 | 0.866667 | 0.866667 | 0.034965 | 0.003856 | -0.008213 |
| 0.750000 | 0.000000 | 60 | 1.000000 | 0.300000 | 0.000000 | 0.000000 | 0.555556 | 0.333333 | 0.688889 | 0.866667 | 0.866667 | 0.034912 | 0.003829 | -0.008675 |
| 1.000000 | 0.000000 | 48 | 0.000000 | 0.400000 | 0.000000 | 0.000000 | 0.555556 | 0.333333 | 0.666667 | 0.888889 | 0.888889 | 0.039973 | 0.002496 | -0.007775 |
| 1.000000 | 0.000000 | 42 | 0.000000 | 0.400000 | 0.000000 | 0.000000 | 0.555556 | 0.333333 | 0.666667 | 0.888889 | 0.888889 | 0.039973 | 0.002496 | -0.007775 |
| 1.000000 | 0.000000 | 54 | 0.000000 | 0.400000 | 0.000000 | 0.000000 | 0.555556 | 0.333333 | 0.666667 | 0.888889 | 0.888889 | 0.039973 | 0.002496 | -0.007775 |
| 1.000000 | 0.000000 | 60 | 0.000000 | 0.400000 | 0.000000 | 0.000000 | 0.555556 | 0.333333 | 0.666667 | 0.888889 | 0.888889 | 0.039973 | 0.002496 | -0.007775 |
| 1.000000 | 0.000000 | 42 | 0.250000 | 0.400000 | 0.000000 | 0.000000 | 0.555556 | 0.333333 | 0.666667 | 0.888889 | 0.888889 | 0.039768 | 0.002476 | -0.007815 |
| 1.000000 | 0.000000 | 42 | 0.500000 | 0.400000 | 0.000000 | 0.000000 | 0.555556 | 0.333333 | 0.666667 | 0.888889 | 0.888889 | 0.039565 | 0.002456 | -0.007855 |
| 1.000000 | 0.000000 | 48 | 0.250000 | 0.400000 | 0.000000 | 0.000000 | 0.555556 | 0.333333 | 0.666667 | 0.888889 | 0.888889 | 0.039778 | 0.002544 | -0.007921 |
| 1.000000 | 0.000000 | 54 | 0.250000 | 0.400000 | 0.000000 | 0.000000 | 0.555556 | 0.333333 | 0.666667 | 0.888889 | 0.888889 | 0.039875 | 0.002676 | -0.008062 |
| 1.000000 | 0.000000 | 48 | 0.500000 | 0.400000 | 0.000000 | 0.000000 | 0.555556 | 0.333333 | 0.666667 | 0.888889 | 0.888889 | 0.039586 | 0.002595 | -0.008066 |
| 1.000000 | 0.000000 | 60 | 0.250000 | 0.400000 | 0.000000 | 0.000000 | 0.555556 | 0.333333 | 0.666667 | 0.888889 | 0.888889 | 0.039889 | 0.002773 | -0.008270 |
| 0.750000 | 0.000000 | 48 | 0.000000 | 0.400000 | 0.000000 | 0.000000 | 0.555556 | 0.333333 | 0.666667 | 0.888889 | 0.888889 | 0.039281 | 0.002359 | -0.007447 |
| 0.750000 | 0.000000 | 42 | 0.000000 | 0.400000 | 0.000000 | 0.000000 | 0.555556 | 0.333333 | 0.666667 | 0.888889 | 0.888889 | 0.039281 | 0.002359 | -0.007447 |
| 0.750000 | 0.000000 | 54 | 0.000000 | 0.400000 | 0.000000 | 0.000000 | 0.555556 | 0.333333 | 0.666667 | 0.888889 | 0.888889 | 0.039281 | 0.002359 | -0.007447 |
| 0.750000 | 0.000000 | 60 | 0.000000 | 0.400000 | 0.000000 | 0.000000 | 0.555556 | 0.333333 | 0.666667 | 0.888889 | 0.888889 | 0.039281 | 0.002359 | -0.007447 |
| 0.750000 | 0.000000 | 42 | 0.250000 | 0.400000 | 0.000000 | 0.000000 | 0.555556 | 0.333333 | 0.666667 | 0.888889 | 0.888889 | 0.039079 | 0.002333 | -0.007493 |

## Table 5 — Frozen configurations on HELD_OUT_CHECK

| config_id | alpha_decay | eta_PP | m0 | beta_low | kappa_dyn | canonical_chain_pass_rate | all_model_chain_pass_rate | individual_model_compliance | human_gt_pt_rate | pt_gt_model_rate | model_gt_no_pedal_rate | model_gt_always_rate | median_gap_extreme_model | median_gap_model_pt | median_gap_pt_human |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| coarse_3ba56a105882f2d9 | 1.000000 | 0.900000 | 48 | 0.000000 | 0.000000 | 0.250000 | 0.250000 | 0.250000 | 0.500000 | 0.350000 | 0.900000 | 0.900000 | 0.073168 | -0.024463 | 0.000130 |
| coarse_c4e9a2e91deac68a | 1.000000 | 0.900000 | 48 | 0.000000 | 0.100000 | 0.000000 | 0.000000 | 0.250000 | 0.250000 | 0.350000 | 0.900000 | 0.900000 | 0.074593 | -0.025076 | -0.004495 |
| coarse_e68f8f8f950ca85e | 0.750000 | 0.000000 | 42 | 1.000000 | 0.400000 | 0.000000 | 0.000000 | 0.250000 | 0.000000 | 0.350000 | 0.900000 | 0.900000 | 0.074022 | -0.034336 | -0.010673 |
| coarse_6126800833517b41 | 1.000000 | 0.000000 | 48 | 0.000000 | 0.400000 | 0.000000 | 0.000000 | 0.250000 | 0.000000 | 0.350000 | 0.900000 | 0.900000 | 0.076209 | -0.034688 | -0.010724 |
| coarse_725f36d961960bd2 | 1.000000 | 0.000000 | 42 | 0.500000 | 0.400000 | 0.000000 | 0.000000 | 0.250000 | 0.000000 | 0.350000 | 0.900000 | 0.900000 | 0.076380 | -0.034781 | -0.011340 |
| coarse_0bf6487694edea18 | 1.000000 | 0.000000 | 60 | 0.250000 | 0.300000 | 0.000000 | 0.000000 | 0.250000 | 0.000000 | 0.350000 | 0.950000 | 0.900000 | 0.075490 | -0.034881 | -0.010152 |
| coarse_118995f41cabf329 | 0.750000 | 0.000000 | 60 | 0.750000 | 0.300000 | 0.000000 | 0.000000 | 0.250000 | 0.000000 | 0.350000 | 0.900000 | 0.900000 | 0.075600 | -0.035770 | -0.010399 |

## Table 6 — Search vs held-out degradation

| config_id | search_canonical_chain_pass_rate | heldout_canonical_chain_pass_rate | canonical_chain_degradation | search_all_model_chain_pass_rate | heldout_all_model_chain_pass_rate | all_chain_degradation |
|---|---|---|---|---|---|---|
| coarse_0bf6487694edea18 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| coarse_118995f41cabf329 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| coarse_6126800833517b41 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| coarse_725f36d961960bd2 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| coarse_e68f8f8f950ca85e | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| coarse_3ba56a105882f2d9 | 0.000000 | 0.250000 | -0.250000 | 0.000000 | 0.250000 | -0.250000 |
| coarse_c4e9a2e91deac68a | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

## Table 7 — Parameter-family effect summary

| parameter | min_mean_chain | max_mean_chain | min_all_chain | max_all_chain | canonical_effect_range |
|---|---|---|---|---|---|
| alpha_decay | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| beta_low | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| eta_PP | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| kappa_dyn | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| m0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

## Table 8 — Candidate system distributions

| config_id | subset_role | system | piece_count | mean_H_piece | median_H_piece | IQR_H_piece |
|---|---|---|---|---|---|---|
| coarse_0bf6487694edea18 | PARAMETER_SEARCH | ALWAYS_ON | 9 | 0.002069 | 0.000840 | 0.001357 |
| coarse_0bf6487694edea18 | PARAMETER_SEARCH | CUSTOM_EVENT_V0 | 9 | 0.026528 | 0.012232 | 0.021036 |
| coarse_0bf6487694edea18 | PARAMETER_SEARCH | HUMAN | 9 | 0.039642 | 0.025476 | 0.028906 |
| coarse_0bf6487694edea18 | PARAMETER_SEARCH | HYBRID_REGRESSION_ONLY | 9 | 0.045938 | 0.024655 | 0.034374 |
| coarse_0bf6487694edea18 | PARAMETER_SEARCH | NO_PEDAL | 9 | 0.000000 | 0.000000 | 0.000000 |
| coarse_0bf6487694edea18 | PARAMETER_SEARCH | ORIGINAL_PT | 9 | 0.045930 | 0.028683 | 0.026272 |
| coarse_0bf6487694edea18 | PARAMETER_SEARCH | STANDARD_CE_ARGMAX | 9 | 0.049204 | 0.033046 | 0.044893 |
| coarse_0bf6487694edea18 | PARAMETER_SEARCH | STANDARD_CE_POSTERIOR_MEDIAN | 9 | 0.052320 | 0.043161 | 0.045117 |
| coarse_0bf6487694edea18 | PARAMETER_SEARCH | WEIGHTED_CE_ARGMAX | 9 | 0.044325 | 0.025123 | 0.033373 |
| coarse_118995f41cabf329 | PARAMETER_SEARCH | ALWAYS_ON | 9 | 0.002083 | 0.000998 | 0.001047 |
| coarse_118995f41cabf329 | PARAMETER_SEARCH | CUSTOM_EVENT_V0 | 9 | 0.026542 | 0.011222 | 0.021440 |
| coarse_118995f41cabf329 | PARAMETER_SEARCH | HUMAN | 9 | 0.039955 | 0.024786 | 0.027166 |
| coarse_118995f41cabf329 | PARAMETER_SEARCH | HYBRID_REGRESSION_ONLY | 9 | 0.045987 | 0.022739 | 0.033101 |
| coarse_118995f41cabf329 | PARAMETER_SEARCH | NO_PEDAL | 9 | 0.000000 | 0.000000 | 0.000000 |
| coarse_118995f41cabf329 | PARAMETER_SEARCH | ORIGINAL_PT | 9 | 0.046778 | 0.029276 | 0.025899 |
| coarse_118995f41cabf329 | PARAMETER_SEARCH | STANDARD_CE_ARGMAX | 9 | 0.049562 | 0.032309 | 0.044732 |
| coarse_118995f41cabf329 | PARAMETER_SEARCH | STANDARD_CE_POSTERIOR_MEDIAN | 9 | 0.052516 | 0.041817 | 0.044939 |
| coarse_118995f41cabf329 | PARAMETER_SEARCH | WEIGHTED_CE_ARGMAX | 9 | 0.044829 | 0.024835 | 0.031997 |
| coarse_3ba56a105882f2d9 | PARAMETER_SEARCH | ALWAYS_ON | 9 | 0.001140 | 0.000498 | 0.000881 |
| coarse_3ba56a105882f2d9 | PARAMETER_SEARCH | CUSTOM_EVENT_V0 | 9 | 0.021120 | 0.009753 | 0.011567 |
| coarse_3ba56a105882f2d9 | PARAMETER_SEARCH | HUMAN | 9 | 0.032656 | 0.020634 | 0.021715 |
| coarse_3ba56a105882f2d9 | PARAMETER_SEARCH | HYBRID_REGRESSION_ONLY | 9 | 0.035699 | 0.020630 | 0.051226 |
| coarse_3ba56a105882f2d9 | PARAMETER_SEARCH | NO_PEDAL | 9 | 0.000000 | 0.000000 | 0.000000 |
| coarse_3ba56a105882f2d9 | PARAMETER_SEARCH | ORIGINAL_PT | 9 | 0.042318 | 0.021711 | 0.042617 |
| coarse_3ba56a105882f2d9 | PARAMETER_SEARCH | STANDARD_CE_ARGMAX | 9 | 0.037630 | 0.026072 | 0.031742 |
| coarse_3ba56a105882f2d9 | PARAMETER_SEARCH | STANDARD_CE_POSTERIOR_MEDIAN | 9 | 0.042660 | 0.027162 | 0.036744 |
| coarse_3ba56a105882f2d9 | PARAMETER_SEARCH | WEIGHTED_CE_ARGMAX | 9 | 0.032278 | 0.027555 | 0.044644 |
| coarse_6126800833517b41 | PARAMETER_SEARCH | ALWAYS_ON | 9 | 0.001976 | 0.000747 | 0.001289 |
| coarse_6126800833517b41 | PARAMETER_SEARCH | CUSTOM_EVENT_V0 | 9 | 0.027020 | 0.013736 | 0.019592 |
| coarse_6126800833517b41 | PARAMETER_SEARCH | HUMAN | 9 | 0.040656 | 0.026684 | 0.027172 |
| coarse_6126800833517b41 | PARAMETER_SEARCH | HYBRID_REGRESSION_ONLY | 9 | 0.045651 | 0.025450 | 0.034382 |
| coarse_6126800833517b41 | PARAMETER_SEARCH | NO_PEDAL | 9 | 0.000000 | 0.000000 | 0.000000 |
| coarse_6126800833517b41 | PARAMETER_SEARCH | ORIGINAL_PT | 9 | 0.046553 | 0.028293 | 0.026157 |
| coarse_6126800833517b41 | PARAMETER_SEARCH | STANDARD_CE_ARGMAX | 9 | 0.049640 | 0.036800 | 0.047127 |
| coarse_6126800833517b41 | PARAMETER_SEARCH | STANDARD_CE_POSTERIOR_MEDIAN | 9 | 0.052629 | 0.047356 | 0.047015 |
| coarse_6126800833517b41 | PARAMETER_SEARCH | WEIGHTED_CE_ARGMAX | 9 | 0.044693 | 0.028449 | 0.035555 |
| coarse_0bf6487694edea18 | HELD_OUT_CHECK | ALWAYS_ON | 4 | 0.002150 | 0.002728 | 0.000762 |
| coarse_0bf6487694edea18 | HELD_OUT_CHECK | CUSTOM_EVENT_V0 | 4 | 0.035200 | 0.033482 | 0.025211 |
| coarse_0bf6487694edea18 | HELD_OUT_CHECK | HUMAN | 4 | 0.060968 | 0.051529 | 0.039352 |
| coarse_0bf6487694edea18 | HELD_OUT_CHECK | HYBRID_REGRESSION_ONLY | 4 | 0.077010 | 0.073141 | 0.040053 |
| coarse_0bf6487694edea18 | HELD_OUT_CHECK | NO_PEDAL | 4 | 0.000000 | 0.000000 | 0.000000 |
| coarse_0bf6487694edea18 | HELD_OUT_CHECK | ORIGINAL_PT | 4 | 0.070892 | 0.063608 | 0.048030 |
| coarse_0bf6487694edea18 | HELD_OUT_CHECK | STANDARD_CE_ARGMAX | 4 | 0.078597 | 0.076837 | 0.042436 |
| coarse_0bf6487694edea18 | HELD_OUT_CHECK | STANDARD_CE_POSTERIOR_MEDIAN | 4 | 0.065396 | 0.076488 | 0.050995 |
| coarse_0bf6487694edea18 | HELD_OUT_CHECK | WEIGHTED_CE_ARGMAX | 4 | 0.080744 | 0.082476 | 0.048642 |
| coarse_118995f41cabf329 | HELD_OUT_CHECK | ALWAYS_ON | 4 | 0.001875 | 0.002313 | 0.000775 |
| coarse_118995f41cabf329 | HELD_OUT_CHECK | CUSTOM_EVENT_V0 | 4 | 0.035057 | 0.033180 | 0.025380 |
| coarse_118995f41cabf329 | HELD_OUT_CHECK | HUMAN | 4 | 0.059607 | 0.048757 | 0.038233 |
| coarse_118995f41cabf329 | HELD_OUT_CHECK | HYBRID_REGRESSION_ONLY | 4 | 0.077880 | 0.072974 | 0.043048 |
| coarse_118995f41cabf329 | HELD_OUT_CHECK | NO_PEDAL | 4 | 0.000000 | 0.000000 | 0.000000 |
| coarse_118995f41cabf329 | HELD_OUT_CHECK | ORIGINAL_PT | 4 | 0.069276 | 0.059187 | 0.044026 |
| coarse_118995f41cabf329 | HELD_OUT_CHECK | STANDARD_CE_ARGMAX | 4 | 0.079933 | 0.077027 | 0.043076 |
| coarse_118995f41cabf329 | HELD_OUT_CHECK | STANDARD_CE_POSTERIOR_MEDIAN | 4 | 0.067105 | 0.076376 | 0.051364 |
| coarse_118995f41cabf329 | HELD_OUT_CHECK | WEIGHTED_CE_ARGMAX | 4 | 0.082186 | 0.082546 | 0.049247 |
| coarse_3ba56a105882f2d9 | HELD_OUT_CHECK | ALWAYS_ON | 4 | 0.000959 | 0.001179 | 0.000333 |
| coarse_3ba56a105882f2d9 | HELD_OUT_CHECK | CUSTOM_EVENT_V0 | 4 | 0.041166 | 0.037980 | 0.027402 |
| coarse_3ba56a105882f2d9 | HELD_OUT_CHECK | HUMAN | 4 | 0.063833 | 0.060846 | 0.047677 |
| coarse_3ba56a105882f2d9 | HELD_OUT_CHECK | HYBRID_REGRESSION_ONLY | 4 | 0.065122 | 0.069741 | 0.047366 |
| coarse_3ba56a105882f2d9 | HELD_OUT_CHECK | NO_PEDAL | 4 | 0.000000 | 0.000000 | 0.000000 |
| coarse_3ba56a105882f2d9 | HELD_OUT_CHECK | ORIGINAL_PT | 4 | 0.064620 | 0.061837 | 0.037947 |
| coarse_3ba56a105882f2d9 | HELD_OUT_CHECK | STANDARD_CE_ARGMAX | 4 | 0.074060 | 0.074555 | 0.034041 |
| coarse_3ba56a105882f2d9 | HELD_OUT_CHECK | STANDARD_CE_POSTERIOR_MEDIAN | 4 | 0.060875 | 0.073102 | 0.045594 |
| coarse_3ba56a105882f2d9 | HELD_OUT_CHECK | WEIGHTED_CE_ARGMAX | 4 | 0.071142 | 0.077538 | 0.045230 |
| coarse_6126800833517b41 | HELD_OUT_CHECK | ALWAYS_ON | 4 | 0.002115 | 0.002685 | 0.000738 |
| coarse_6126800833517b41 | HELD_OUT_CHECK | CUSTOM_EVENT_V0 | 4 | 0.034342 | 0.032949 | 0.025128 |
| coarse_6126800833517b41 | HELD_OUT_CHECK | HUMAN | 4 | 0.058958 | 0.050562 | 0.037599 |
| coarse_6126800833517b41 | HELD_OUT_CHECK | HYBRID_REGRESSION_ONLY | 4 | 0.074693 | 0.073681 | 0.041471 |
| coarse_6126800833517b41 | HELD_OUT_CHECK | NO_PEDAL | 4 | 0.000000 | 0.000000 | 0.000000 |
| coarse_6126800833517b41 | HELD_OUT_CHECK | ORIGINAL_PT | 4 | 0.071562 | 0.066647 | 0.050803 |
| coarse_6126800833517b41 | HELD_OUT_CHECK | STANDARD_CE_ARGMAX | 4 | 0.077086 | 0.077127 | 0.041000 |
| coarse_6126800833517b41 | HELD_OUT_CHECK | STANDARD_CE_POSTERIOR_MEDIAN | 4 | 0.064122 | 0.076937 | 0.048830 |
| coarse_6126800833517b41 | HELD_OUT_CHECK | WEIGHTED_CE_ARGMAX | 4 | 0.080268 | 0.082745 | 0.045335 |

## Mechanism and failure analysis

All 4,320 SEARCH configurations had zero canonical and MODEL_ALL full-chain rate. Therefore Table 7's full-chain effect ranges are correctly all zero, but they cannot rank parameter importance. The secondary diagnostic below uses individual-model compliance instead.

### Secondary ordering sensitivity

| parameter | min_compliance | max_compliance | compliance_range |
|---|---|---|---|
| kappa_dyn | 0.443272 | 0.483302 | 0.040031 |
| eta_PP | 0.451420 | 0.484599 | 0.033179 |
| alpha_decay | 0.451512 | 0.468580 | 0.017068 |
| beta_low | 0.455350 | 0.467181 | 0.011831 |
| m0 | 0.458045 | 0.465123 | 0.007078 |

`kappa_dyn` has the largest marginal individual-compliance range (0.040), followed by `eta_PP` (0.033) and `alpha_decay` (0.017). None changes the full-chain conclusion.

### Dominant inequality failures

| subset_role | piece_count | E_lt_M_canonical_rate | M_canonical_lt_PT_rate | PT_lt_HUMAN_rate | canonical_full_chain_rate |
|---|---|---|---|---|---|
| PARAMETER_SEARCH | 9 | 0.888889 | 0.555556 | 0.333333 | 0.000000 |
| HELD_OUT_CHECK | 4 | 1.000000 | 0.250000 | 0.000000 | 0.000000 |

For the lexicographic SEARCH best, `E<M` fails on 1/9 SEARCH pieces, `M<PT` fails on 3/9, and `PT<HUMAN` fails on 6/9. On HELD_OUT, `M<PT` fails on 3/4 and `PT<HUMAN` fails on all 4. Thus the primary structural bottlenecks are PT-versus-model and HUMAN-versus-PT, not suppression of the extremes.

### Parameter mechanisms

Lowering `eta_PP` from 0.9 to 0 raises mean ALWAYS_ON by +0.000799, raises the four canonical model means by +0.004254, changes ORIGINAL_PT by -0.004466, and leaves HUMAN nearly unchanged (+0.000088). This sometimes improves PT>model compliance, but it also moves ALWAYS_ON upward and never creates a full-chain region.

Increasing `alpha_decay` from 0.5 to 2.0 lifts every nonzero system: ALWAYS_ON +0.000840, HUMAN +0.011924, ORIGINAL_PT +0.013713, and the canonical-model mean +0.015691. ALWAYS_ON is less sensitive in absolute H_piece than the performances, contrary to the hypothesized special long-residual sensitivity.

The per-piece low-register contrasts and their correlation with non-pedal low-note fraction are stored in `parameter_contrast_per_piece.csv` and `parameter_mechanism_diagnostics.csv`. Effects vary by system and do not rescue the chain. The kappa contrast likewise changes secondary separation but not full-chain feasibility.

Optional refinement was not performed; the predeclared trigger required both SEARCH chain rates >=0.75 and individual compliance >=0.75. This prevents local search when the formula family does not show a robust coarse capability signal.

PEDAL_SPARSE threshold sensitivity for frozen candidates is in `pedal_sparse_sensitivity.csv`; coverage columns expose deferred long-form pieces at every threshold.

## Final candidate configurations (not final parameters)

| config_id | candidate_type | alpha_decay | eta_PP | m0 | beta_low | kappa_dyn | search_canonical_chain_pass_rate | heldout_canonical_chain_pass_rate | reference_distance_grid_L1 | selection_reason |
|---|---|---|---|---|---|---|---|---|---|---|
| coarse_0bf6487694edea18 | strongest_SEARCH | 1.000000 | 0.000000 | 60 | 0.250000 | 0.300000 | 0.000000 | 0.000000 | 10 | lexicographic SEARCH best |
| coarse_3ba56a105882f2d9 | reference_near_heldout_stable | 1.000000 | 0.900000 | 48 | 0.000000 | 0.000000 | 0.000000 | 0.250000 | 0 | Hall+decay backbone; best held-out behavior within the pre-frozen pool |
| coarse_118995f41cabf329 | distinct_region | 0.750000 | 0.000000 | 60 | 0.750000 | 0.300000 | 0.000000 | 0.000000 | 13 | qualitatively distinct pre-frozen coarse region |
| coarse_6126800833517b41 | distinct_region | 1.000000 | 0.000000 | 48 | 0.000000 | 0.400000 | 0.000000 | 0.000000 | 8 | qualitatively distinct pre-frozen coarse region |

These are listening controls spanning reference, strongest-SEARCH, and distinct regions—not successful calibrated optima. `candidate_failure_details.csv` records the failed systems/pieces for every candidate and subset.

## Answers to the research questions

**Q1. Formula-family capability.** No robust target-ordering region was found. Every one of the 4,320 SEARCH configurations had canonical and MODEL_ALL full-chain rate 0/9.

**Q2. Held-out maintenance.** No SEARCH success existed to generalize. Among configurations frozen before held-out scoring, only the Hall+decay backbone passed on one of four held-out pieces (0.25); the other candidates passed zero. This isolated held-out pass is not evidence of stable ordering.

**Q3. Important parameters.** By secondary individual-model compliance, kappa_dyn had the largest marginal range (0.040), eta_PP was next (0.033), and alpha_decay followed (0.017). eta_PP materially trades PT against models, while longer decay lifts all systems; neither produces the full chain.

**Q4. Reference distance.** The lexicographic SEARCH best is 10 coarse grid steps from the Hall+decay backbone, mainly because eta_PP falls to 0, m0 rises to 60, and kappa rises to 0.3. That movement still yields no full-chain pass, so moving far from the reference is not justified by this audit.

**Q5. Structural failure.** At the SEARCH best, PT<HUMAN fails most often (6/9), followed by M_canonical<PT (3/9) and E<M_canonical (1/9). On held-out, PT<HUMAN fails 4/4 and M_canonical<PT fails 3/4. The family generally separates models from extremes, but does not consistently express the proposed quality order among model, PT, and HUMAN.

**Q6. Listening candidates.** Four pre-frozen configurations are retained as diagnostic listening controls: the literature backbone, the strongest SEARCH secondary-ordering point, and two distinct regions. None is a final parameter or a successful target-ordering solution; candidate-specific failures are in `candidate_failure_details.csv`.

## Files, provenance, and caveats

- MODEL_CANONICAL contains the four strict canonical-stream Stage 2 models; MODEL_ALL adds CUSTOM_EVENT_V0.
- HUMAN and CUSTOM_EVENT_V0 are not exact canonical pedal-only comparisons.
- Reused extreme source/generated SHA and strict identity are in `reusable_extremes_manifest.csv`.
- The Lehtonen pitch-anchor interpolation remains the research v0 adaptation; alpha_decay only scales its T60 values.
- All input paths and SHA values are in `input_midis.csv`; no TEST path was accessed.
- Final integrity audit: 13/13 PASS (`integrity_audit.csv`).
- Four eligible long-form pieces were deferred by the predeclared exact pair-onset work cap; no approximation or pruning was substituted. Claims therefore apply to 13 exact-scored eligible pieces (9 SEARCH, 4 HELD_OUT), with coverage disclosed in all sensitivity tables.
