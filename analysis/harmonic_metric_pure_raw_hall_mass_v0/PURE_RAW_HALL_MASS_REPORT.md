# Pure Raw Hall-Negative Mass Diagnostic v0

## Executive result

The accumulation hypothesis is strongly supported for the ALWAYS_ON extreme. Removing the pair denominator changes its piece-balanced Hall-negative value from a modest pair-average severity of **0.320** to **227,719.4 raw mass per onset**. This is **4,909x ORIGINAL_PT**, **26,074x HUMAN**, and **7,320–29,505x the canonical Stage2 system means**.

The effect is overwhelmingly PP combinatorics: **99.692%** of aggregate ALWAYS_ON raw mass is PP. From the first to last normalized decile, ALWAYS_ON P count rises **19.3x**, PP pair count **279.8x**, and PP raw mass **331.0x**. Its total raw mass rises from **1,939.0** to **619,522.6**.

The evidence therefore says that Hall did find extensive conflict in ALWAYS_ON; pair-average normalization discarded the accumulated quantity. This conclusion is limited to the extreme. Raw mass still gives HUMAN<PT in only **8/13** and PT<canonical Stage2 in **17/52** cells, so density-sensitive mass is not a standalone normal-pedaling quality metric and is not adopted here.

## Scope and exact computation

`Metric_harmonic_consonance.md` is the semantic source of truth. This run reuses the audited Pure Hall onset cache and note-instance semantics: CC64 >=64 is on; PA is P×A; PP is choose(P,2); A-A is excluded; octave-equivalent Hall Simple Type lookup is unchanged. Each pair contributes only `d=max(-HallWeight,0)`.

There is no decay, velocity, low-register or dynamic term, positive reward, eta differential, denominator, parameter search, fitting, inference, training, TEST access, or MIDI generation. Real MIDI pairs were not materialized. PA and unordered PP multiplicities were evaluated exactly from pitch-class counts. One hundred random small states were compared against explicit enumeration; maximum mass error was **7.105e-15** and pair-count error was zero.

The evaluation set is the same 13 PEDAL_ELIGIBLE + EXACT_COMPUTABLE validation pieces and nine systems as the preceding audit. PEDAL_SPARSE and DEFERRED_LONG_FORM remain excluded. HUMAN and CUSTOM_EVENT_V0 do not share the strict canonical non-CC64 stream and remain reference distributions, not exact pedal-only causal comparisons. NO_PEDAL=0 is expected for this harm-only diagnostic and is not an overall pedaling-quality claim.

## Table 1 — Piece-balanced raw-mass summary

All statistics are computed per piece first and then balanced across the 13 pieces. `raw mean` is the primary mean of M_TOTAL over all distinct onsets, with Q-empty onsets set to zero.

| system | raw mean | piece median | piece IQR | PA mean | PP mean | onset p90 | onset p99 | onset max |
|---|---|---|---|---|---|---|---|---|
| NO_PEDAL | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| ALWAYS_ON | 227719.385626 | 197300.053956 | 247612.059811 | 758.463914 | 226960.921712 | 556612.977408 | 671642.498925 | 687089.413000 |
| STANDARD_CE_ARGMAX | 11.904270 | 3.358724 | 15.673439 | 2.544879 | 9.359391 | 25.131646 | 189.879291 | 352.659231 |
| STANDARD_CE_POSTERIOR_MEDIAN | 10.970086 | 3.370620 | 11.200915 | 2.391841 | 8.578245 | 22.685377 | 176.857805 | 323.873231 |
| WEIGHTED_CE_ARGMAX | 7.717914 | 2.394103 | 7.693440 | 1.837234 | 5.880679 | 14.964477 | 130.794322 | 257.000615 |
| HYBRID_REGRESSION_ONLY | 31.107419 | 4.229905 | 19.006214 | 3.692141 | 27.415278 | 64.000969 | 520.717803 | 721.099538 |
| CUSTOM_EVENT_V0 | 40.747368 | 9.657458 | 20.983393 | 4.021231 | 36.726137 | 128.422400 | 587.659288 | 838.647923 |
| ORIGINAL_PT | 46.387963 | 8.981223 | 42.376888 | 5.449019 | 40.938945 | 114.924177 | 705.349431 | 1023.037846 |
| HUMAN | 8.733685 | 5.320964 | 12.464833 | 2.673607 | 6.060077 | 18.808677 | 108.733349 | 244.045077 |

## Table 2 — Pair-average severity versus raw accumulation

These columns answer different questions: the prior value is mean conflict per pedal-induced pair on valid onsets; raw mass is the absolute accumulated negative Hall magnitude per all onset.

| system | pair-average Hall negative | raw Hall mass/onset |
|---|---|---|
| NO_PEDAL | 0.000000 | 0.000000 |
| ALWAYS_ON | 0.320297 | 227719.385626 |
| STANDARD_CE_ARGMAX | 0.232901 | 11.904270 |
| STANDARD_CE_POSTERIOR_MEDIAN | 0.227816 | 10.970086 |
| WEIGHTED_CE_ARGMAX | 0.240969 | 7.717914 |
| HYBRID_REGRESSION_ONLY | 0.225839 | 31.107419 |
| CUSTOM_EVENT_V0 | 0.253005 | 40.747368 |
| ORIGINAL_PT | 0.220236 | 46.387963 |
| HUMAN | 0.218644 | 8.733685 |

| comparison | raw-mass ratio |
|---|---|
| ALWAYS_ON / HUMAN | 26073.69 |
| ALWAYS_ON / ORIGINAL_PT | 4909.02 |
| ALWAYS_ON / STANDARD_CE_ARGMAX | 19129.22 |
| ALWAYS_ON / STANDARD_CE_POSTERIOR_MEDIAN | 20758.21 |
| ALWAYS_ON / WEIGHTED_CE_ARGMAX | 29505.30 |
| ALWAYS_ON / HYBRID_REGRESSION_ONLY | 7320.42 |
| ALWAYS_ON / CUSTOM_EVENT_V0 | 5588.57 |

## Table 3 — Diagnostic ordering

Counts are out of 13 for HUMAN<PT and full chain, and out of 52 piece-model cells for the two macro comparisons.

| view | HUMAN<PT | PT<model | model<ALWAYS | full canonical |
|---|---|---|---|---|
| prior pair-average Hall negative | 7 | 20 | 45 | 1 |
| raw Hall mass / all onset | 8 | 17 | 52 | 2 |

Raw mass makes the extreme boundary perfect: model<ALWAYS_ON improves from 45/52 under the prior pair-average to 52/52. It does not improve normal-system ordering: HUMAN<PT changes from 7/13 to 8/13, while PT<model falls from 20/52 to 17/52. The full canonical chain is only 2/13. This is consistent with raw mass measuring conflict accumulation and texture density rather than a complete quality ordering.

## Table 4 — Normalized performance-position trajectory

Early is 0–10%, middle is the mean of 40–60%, and late is 90–100%. Bins contain equal counts of distinct onsets within each performance.

| system | raw early | raw middle | raw late | raw late/early | PP mass late/early | P count late/early | PP pairs late/early |
|---|---|---|---|---|---|---|---|
| ALWAYS_ON | 1938.961 | 172808.678 | 619522.611 | 319.513 | 330.971 | 19.282 | 279.813 |
| HUMAN | 4.957 | 6.432 | 9.023 | 1.820 | 1.927 | 1.656 | 3.572 |
| ORIGINAL_PT | 33.455 | 31.118 | 70.377 | 2.104 | 2.151 | 1.804 | 2.300 |
| MODEL_CANONICAL_MEDIAN | 13.749 | 5.716 | 24.145 | 1.756 | 1.870 | 1.665 | 2.669 |

For ALWAYS_ON, PP mass and PP pair count track almost exactly across piece×position bins: Pearson r=0.996034, Spearman rho=0.998258, and log1p Pearson r=0.998944. The 19.3x growth in P count becomes 279.8x PP-pair growth and 331.0x PP-mass growth, as expected from choose(|P|,2). Normal systems show only 1.8–2.1x raw-mass late/early movement at the system-aggregate level; ALWAYS_ON shows 319.5x.

## Table 5 — PA / PP source

Fractions are computed from aggregate raw numerator mass across all onsets and pieces; the PA/PP per-onset columns are piece-balanced.

| system | PA mass/onset | PP mass/onset | PA fraction | PP fraction |
|---|---|---|---|---|
| NO_PEDAL | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| ALWAYS_ON | 758.463914 | 226960.921712 | 0.003084 | 0.996916 |
| STANDARD_CE_ARGMAX | 2.544879 | 9.359391 | 0.207283 | 0.792717 |
| STANDARD_CE_POSTERIOR_MEDIAN | 2.391841 | 8.578245 | 0.210646 | 0.789354 |
| WEIGHTED_CE_ARGMAX | 1.837234 | 5.880679 | 0.223477 | 0.776523 |
| HYBRID_REGRESSION_ONLY | 3.692141 | 27.415278 | 0.130751 | 0.869249 |
| CUSTOM_EVENT_V0 | 4.021231 | 36.726137 | 0.138614 | 0.861386 |
| ORIGINAL_PT | 5.449019 | 40.938945 | 0.122069 | 0.877931 |
| HUMAN | 2.673607 | 6.060077 | 0.283040 | 0.716960 |

The current compact calculation exactly reproduces the previous audit's ALWAYS_ON PP mass fraction: 0.996915635268. PP also dominates several normal systems, but at vastly smaller absolute scale. The extreme therefore reflects both persistent residual membership and quadratic PP opportunities, not unusually large per-pair Hall severity alone.

## Table 6 — Synthetic multiplicity sanity

The pitch-class distribution is fixed while pedal-note multiplicity is scaled 1, 2, 4, and 8. PA grows linearly and PP mass grows exactly quadratically in this constructed state.

| P scale | P count | PA_pair_count | PP_pair_count | M_PA | M_PP | M_TOTAL | PA mass ratio | PP mass ratio |
|---|---|---|---|---|---|---|---|---|
| 1 | 4 | 12 | 6 | 6.947000 | 4.095000 | 11.042000 | 1.000000 | 1.000000 |
| 2 | 8 | 24 | 28 | 13.894000 | 16.380000 | 30.274000 | 2.000000 | 4.000000 |
| 4 | 16 | 48 | 120 | 27.788000 | 65.520000 | 93.308000 | 4.000000 | 16.000000 |
| 8 | 32 | 96 | 496 | 55.576000 | 262.080000 | 317.656000 | 8.000000 | 64.000000 |

## Table 7 — Diagnostic pieces

The old search/held-out label is retained only as metadata; no fitting or split-based selection occurs here.

| piece_id | performance_id | split_role_metadata |
|---|---|---|
| piece_08aa5ff2ada461e0 | Bach/Prelude/bwv_862/Song04M.mid | HELD_OUT_CHECK |
| piece_0a888a7d5fbd06e6 | Haydn/Keyboard_Sonatas/46-1/Bach01.mid | PARAMETER_SEARCH |
| piece_18921bc81d708d16 | Beethoven/Piano_Sonatas/9-3/Tysman05M.mid | PARAMETER_SEARCH |
| piece_3056e93b3e51b646 | Schumann/Kreisleriana/6/JohannsonP07.mid | PARAMETER_SEARCH |
| piece_555ff6cd3a13efb9 | Scriabin/Etudes_op_8/11/Shi03.mid | PARAMETER_SEARCH |
| piece_69862af5096ee3fa | Beethoven/Piano_Sonatas/27-1/Abdelmola01.mid | PARAMETER_SEARCH |
| piece_759b324c1cceac54 | Schubert/Piano_Sonatas/664-1/BuiJL06M.mid | PARAMETER_SEARCH |
| piece_ab42b317dcb38fc5 | Beethoven/Piano_Sonatas/29-2/ChowK03.mid | HELD_OUT_CHECK |
| piece_daefdda4e1923cc6 | Schumann/Arabeske/Min09M.mid | HELD_OUT_CHECK |
| piece_db97fbaed2036f5b | Chopin/Etudes_op_25/12/Atzinger03.mid | PARAMETER_SEARCH |
| piece_de3f82957f1b3532 | Ravel/Pavane/ChenS03.mid | HELD_OUT_CHECK |
| piece_e38184a2e9753bdc | Haydn/Keyboard_Sonatas/39-2/Yarden07.mid | PARAMETER_SEARCH |
| piece_e7ad71b998a8d477 | Haydn/Keyboard_Sonatas/32-1/Pavlovic02.mid | PARAMETER_SEARCH |

## Integrity audit

| check | status | detail |
|---|---|---|
| decay_not_used | PASS | audited pure-Hall onset cache contains no decay mass |
| velocity_not_used | PASS | cached score used pitch only |
| low_weight_not_used | PASS | absent |
| dynamic_weight_not_used | PASS | absent |
| positive_Hall_reward_not_used | PASS | raw fields contain max(-Hall,0) only |
| no_onset_denominator_or_normalization | PASS | M_TOTAL=M_PA+M_PP max_abs_error=0.000e+00 |
| no_PA_PP_differential_weight | PASS | one unit per note-instance pair |
| no_AA_pairs | PASS | AA_pair_count=0 |
| compact_equals_naive_100_random_states | PASS | states=100; max_mass_error=7.105e-15; max_count_error=0 |
| synthetic_PA_linear_growth | PASS | max_ratio_error=0.000e+00 |
| synthetic_PP_quadratic_growth | PASS | max_ratio_error=0.000e+00 |
| prior_pure_Hall_mass_cache_exact_at_Hall_precision | PASS | onset_round3_error=0.000e+00; piece_round3_error=0.000e+00 |
| ALWAYS_ON_PP_fraction_reproduced | PASS | prior=0.996915635268; current=0.996915635268 |
| NO_PEDAL_raw_mass_exact_zero | PASS | rows=23362 |
| all_values_finite | PASS | numeric_cells=3843504 |
| normalized_profile_has_10_bins | PASS | piece_systems=117; min=10; max=10 |
| source_MIDI_SHA_unchanged | PASS | files=117 |
| TEST_access_zero | PASS | count=0 |
| inference_zero | PASS | count=0 |
| training_zero | PASS | count=0 |
| MIDI_generation_zero | PASS | count=0 |
| no_explicit_real_pair_materialization | PASS | read compact onset aggregates; explicit pairs limited to <=76-pair synthetic states |
| diagnostic_set_exact_13x9 | PASS | pieces=13; piece_systems=117 |

All **23/23** checks pass. The 117 source MIDI SHA256 values are unchanged. The cached Hall values were restored to their defining three-decimal table precision before addition so binary CSV round-trip noise cannot create false absolute-tolerance failures at very large sums; the prior cache matches at that exact Hall precision.

## Answers to the primary questions

**Q1 — How much larger is ALWAYS_ON raw mass?** Its mean is 227,719.4 per onset, versus 8.734 for HUMAN, 46.388 for ORIGINAL_PT, and 7.718–31.107 for canonical Stage2 systems. That is roughly 4,909x PT and 26,074x HUMAN.

**Q2 — Does it accumulate toward the end?** Yes. ALWAYS_ON rises from 1,939 raw mass/onset in the first decile to 619,523 in the last, a 319.5x increase. This is qualitatively and quantitatively unlike HUMAN (1.82x), PT (2.10x), or the canonical-model median (1.76x).

**Q3 — PA or PP?** PP. It supplies 99.6916% of ALWAYS_ON aggregate raw mass. Late-decile PP mass is 617,992, compared with PA mass 1,531.

**Q4 — Does PP mass agree with choose(|P|,2) growth?** Yes. PP pair count grows 279.8x from early to late and PP mass grows 331.0x. Across all ALWAYS_ON piece×bin cells, PP mass versus PP pair count has Pearson r=0.9960 and rank correlation 0.9983. The compact synthetic test also gives exact 1, 4, 16, 64 PP-mass ratios when P multiplicity is scaled 1, 2, 4, 8.

**Q5 — Was Hall conflict weak, or was accumulation hidden by averaging?** For ALWAYS_ON, accumulation was hidden by averaging. Pair-average Hall negative asks how severe one average relation is and was 0.320; raw mass shows that the number of residual-residual conflict-bearing relations becomes enormous. This does not mean normalization is universally wrong: without it, texture density, polyphony, duration, and symbolic PP combinatorics dominate, and normal-system quality discrimination remains weak.

## Interpretation boundary

- Pair-average Hall negative is **average conflict severity per pedal-induced pair**.
- Pure raw Hall-negative mass is **absolute accumulated conflict magnitude per onset**.
- The latter exposes ALWAYS_ON pathology, but its scale is deliberately unnormalized and physically ignores decay. It is therefore a mechanism diagnostic, not a final metric.
- No new formula, combined score, parameter, or final metric is proposed or selected in this task.
- CSV files are the source of truth; figures are interpretation aids. Full input paths and SHA256 values are in `input_midis.csv`.

## Artifacts

- `piece_system_raw_hall_mass.csv`
- `onset_raw_hall_mass.csv`
- `normalized_position_profile.csv`
- `normalized_position_system_summary.csv`
- `pa_pp_raw_mass_summary.csv`
- `raw_mass_ordering.csv`
- `synthetic_multiplicity_sanity.csv`
- `random_compact_naive_exactness.csv`
- `sanity_checks.csv`
- `input_midis.csv`
- `system_raw_hall_mass_summary.csv`
- six PNG figures in this directory
