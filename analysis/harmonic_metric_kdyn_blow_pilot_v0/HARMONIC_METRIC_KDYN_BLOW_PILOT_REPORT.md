# Harmonic Metric κ_dyn / β_low Pilot Sweep v0

## Scope and source of truth

This validation-only pilot used `Metric_harmonic_consonance.md` as the sole harmonic-metric source of truth. No ASAP test MIDI was accessed, no model inference was run, and no training or checkpoint selection was performed. The sweep changes only `kappa_dyn` and `beta_low`; `(0,0)` is the Hall + decay backbone baseline.

The metric document contains no numeric Hall table, so the implementation uses Hall, Tamir & Rohrmeier (2025), Table 1, Simple Type/type-method learned interval weights at the precision supplied for this task: m2 −1.902, M2 −0.299, m3 −0.029, M3 +0.257, P4 +0.616, TT −0.582, P5 +0.676, m6 −0.339, M6 −0.142, m7 −0.712, M7 −0.899, P8 +1.009. Octave equivalence maps `0 mod 12` to P8 (class 12).

Fixed values: `m0=48`, `eta_PA=1.0`, `eta_PP=0.9`, `L(v)=v/127`, `epsilon=1e-12`, and symbolic sustain ON iff `CC64>=64`. There is no depth multiplier, active–active pair, note-count bonus, or separate bass metric.

## Representative systems

The conventional set is intentionally representative rather than an exhaustive dump of historical checkpoints. It follows the current comparison lineage in the phase-3/phase-4/phase-4b validation reports.

| System | Role / experiment |
|---|---|
| `ORIGINAL_PT` | canonical_stage1_baseline; `stage2_binary_canonical_v1/canonical_validation_stage1` |
| `STANDARD_CE_ARGMAX` | recent_4class_model; `stage2_encoder_only_4class_decoding_phase4_v0/argmax` |
| `STANDARD_CE_POSTERIOR_MEDIAN` | recent_4class_model; `stage2_encoder_only_4class_decoding_phase4_v0/median` |
| `WEIGHTED_CE_ARGMAX` | recent_4class_model; `stage2_encoder_only_4class_loss_phase3_v0/weighted_ce` |
| `HYBRID_REGRESSION_ONLY` | recent_4class_model; `stage2_encoder_only_raw_huber_aux_ce_v1/raw_huber_aux_ce` |
| `CUSTOM_EVENT_V0` | latest saved performance-level custom-event validation inference |
| `HUMAN` | selected ASAP validation performance; reference distribution, not an exact pedal-only pair to Stage 1 |
| `NO_PEDAL` / `ALWAYS_ON` | reusable extremes generated from canonical Stage 1 |

Available but non-representative outputs were not scored:

- `stage2_4class_architecture_validation_eval_v0/decoder_only` — architecture diagnostic not retained in the current representative comparison.
- `stage2_4class_architecture_validation_eval_v0/encoder_decoder` — architecture diagnostic not retained in the current representative comparison.
- `stage2_encoder_only_4class_loss_phase3_v0/ce_ntl_was_lambda_0p3` — loss diagnostic not retained over the representative weighted-CE and hybrid outputs.
- `stage2_encoder_only_4class_loss_phase3_v0/representative_huber_delta14` — superseded by the current hybrid raw-regression finalist.
- `stage2_encoder_only_4class_decoding_phase4_v0/expectation` — decoder diagnostic; posterior median retained as the representative alternate decoder.
- `stage2_encoder_only_4class_decoding_phase4_v0/top2_seed42` — single-seed stochastic decoder diagnostic not retained.
- `stage2_loss_decoding_fusion_phase4b_v0/weighted_ce_median` — phase-4b diagnostic not selected as a current representative.
- `stage2_loss_decoding_fusion_phase4b_v0/ntl_was_median` — phase-4b diagnostic not selected as a current representative.
- `stage2_loss_decoding_fusion_phase4b_v0/hybrid_aux_median` — auxiliary-only diagnostic; did not replace regression-only inference.
- `stage2_loss_decoding_fusion_phase4b_v0/hybrid_side_constrained` — phase-4b report retained hybrid regression-only as the conventional finalist.

## Case intersection and metric-blind selection

The common validation intersection contained 19 pieces. Exactly 8 distinct pieces were selected without harmonic scores. For each descriptor tail (bottom/top quartile of low-note fraction, velocity range, and note density), the lowest-note-count member was chosen; duplicates were merged, then descriptor-space farthest-point selection among pieces with at most 1,900 note onsets filled to eight. This explicit non-pedal size cap bounds the exact, unpruned ALWAYS_ON computation; 9 intersection pieces were eligible. The lexicographically first available validation performance within each piece fixes the custom/human row.

| # | Piece | Performance | Low frac. | Vel. range | Notes/s | Notes | Reason |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | `piece_b06ab236f236fd76` (Bach — Prelude_bwv_888) | `Bach/Prelude/bwv_888/ZhaoK01M.mid` | 0.0610 | 64 | 6.520 | 688 | low-register-light; narrow velocity range; low note density |
| 2 | `piece_3056e93b3e51b646` (Schumann — Kreisleriana_6) | `Schumann/Kreisleriana/6/JohannsonP07.mid` | 0.1866 | 87 | 5.575 | 729 | low-register-heavy |
| 3 | `piece_ab42b317dcb38fc5` (Beethoven — Piano_Sonatas_29-2) | `Beethoven/Piano_Sonatas/29-2/ChowK03.mid` | 0.1907 | 104 | 13.673 | 1888 | wide velocity range |
| 4 | `piece_f10784ae79f0f210` (Bach — Prelude_bwv_856) | `Bach/Prelude/bwv_856/LuoJ01M.mid` | 0.0794 | 68 | 15.389 | 1045 | high note density |
| 5 | `piece_de3f82957f1b3532` (Ravel — Pavane) | `Ravel/Pavane/ChenS03.mid` | 0.1030 | 85 | 11.285 | 1816 | descriptor-space farthest-point fill |
| 6 | `piece_18921bc81d708d16` (Beethoven — Piano_Sonatas_9-3) | `Beethoven/Piano_Sonatas/9-3/Tysman05M.mid` | 0.2067 | 89 | 10.029 | 1785 | descriptor-space farthest-point fill |
| 7 | `piece_e7ad71b998a8d477` (Haydn — Keyboard_Sonatas_32-1) | `Haydn/Keyboard_Sonatas/32-1/Pavlovic02.mid` | 0.0621 | 74 | 8.386 | 1497 | descriptor-space farthest-point fill |
| 8 | `piece_555ff6cd3a13efb9` (Scriabin — Etudes_op_8_11) | `Scriabin/Etudes_op_8/11/Shi03.mid` | 0.1081 | 85 | 7.220 | 1387 | descriptor-space farthest-point fill |

## MIDI state semantics and reusable extremes

Notes are FIFO-matched note instances keyed by `(channel,pitch)`; repeated/overlapping same-pitch notes are not collapsed. State is evaluated once after all merged MIDI events at a distinct onset tick have been applied. CC64 state is channel-specific. A physical note-off moves a held instance to `P_n` only while that channel's sustain is ON; pedal OFF removes that channel's residuals. CC120/123 are handled explicitly. Only `P×A` and `choose(P,2)` enter the accumulator.

All 56/56 canonical-based selected files passed strict ordered non-CC64 identity and note identity. Custom Event shares the selected human musical non-CC64 stream but shares canonical Stage 1 strict identity in only 0.0% of cases; it is therefore not presented as an exact Stage-1 pedal-only comparison.

Reusable extremes were created under `reusable_extremes/` (16 files). NO_PEDAL removes all CC64. ALWAYS_ON removes prior CC64 and inserts CC64=127 at MIDI time 0 on each note-bearing track/channel, remaining ON through EOT. Every generated file passed strict non-CC64 and exact-note identity; sources were not modified.

### Note-instance corner cases

| Counter | Total across scored MIDI |
|---|---:|
| `max_same_channel_pitch_sounding_instances` | 1288 |
| `overlapping_same_pitch_active_onset` | 350 |
| `pedal_release_residual_notes` | 31958 |
| `retrigger_over_pedal_residual` | 23406 |
| `same_tick_same_pitch_on_and_off` | 466 |
| `onset_ticks_with_cc64` | 5560 |

## Decay implementation

Strength is `(v/127) * 10^(-3*age/T60(pitch))` with age measured from note attack for both active and pedal-residual instances. T60 uses anchors 36:18.1 s, 48:14.8 s, 60:15.3 s, 74:18.3 s, 84:12.0 s, piecewise-linear MIDI-pitch interpolation inside, and endpoint clamping outside. This interpolation is this research project's v0 adaptation; Lehtonen et al. (2007) supplied the sustain-pedal overall T60 anchor measurements, not the interpolation rule.

No acoustic-strength pruning or pair approximation was used. The denominator is exactly `sum(eta)+epsilon`; neither W_dec nor W_low is included in it.

## Implementation audit

Synthetic/unit checks: 16/16 PASS. One-case pre-sweep audit: 9/9 PASS. Full-sweep NO_PEDAL exact-zero invariant: **PASS**.

| One-case check | Status | Detail |
|---|---|---|
| `NO_PEDAL_exact_zero` | PASS | {'H_piece': 0.0, 'valid_onset_count': 0, 'positive_onset_fraction': 0.0, 'negative_onset_fraction': 0.0, 'zero_onset_fraction': 0.0, 'mean_H_base': 0.0, 'mean_W_dyn': 1.0, 'mean_W_low': 1.0, 'mean_valid_pair_count': 0.0, 'PA_pair_count': 0, 'PP_pair_count': 0} |
| `ALWAYS_ON_P_accumulates` | PASS | first=0, max=683 |
| `beta0_Wlow_exact_one` | PASS | rows=8 |
| `kappa0_H_equals_Hbase` | PASS | rows=8 |
| `PP_eta_0p9` | PASS | constant=0.9 and PP observed |
| `decay_from_attack` | PASS | synthetic attack/T60/no-reset tests passed before MIDI audit |
| `AA_pair_count_zero` | PASS | no AA branch exists; accumulated AA=0 |
| `finite_output` | PASS | onsets=8 |
| `canonical_non_CC64_identity` | PASS | all canonical-based systems exact |

## Baseline distribution: κ=0, β=0

| System | Mean | Median | IQR | > ALWAYS_ON |
|---|---:|---:|---:|---:|
| `ALWAYS_ON` | 0.000744 | 0.000839 | 0.001174 | NA |
| `CUSTOM_EVENT_V0` | 0.017469 | 0.012171 | 0.020362 | 0.875 |
| `HUMAN` | 0.032074 | 0.029173 | 0.033854 | NA |
| `HYBRID_REGRESSION_ONLY` | 0.024917 | 0.011615 | 0.069144 | 0.750 |
| `NO_PEDAL` | 0.000000 | 0.000000 | 0.000000 | NA |
| `ORIGINAL_PT` | 0.009634 | 0.009052 | 0.022587 | 0.875 |
| `STANDARD_CE_ARGMAX` | 0.036130 | 0.036986 | 0.077770 | 0.875 |
| `STANDARD_CE_POSTERIOR_MEDIAN` | 0.035177 | 0.040564 | 0.084964 | 0.750 |
| `WEIGHTED_CE_ARGMAX` | 0.036217 | 0.027708 | 0.084659 | 0.750 |

NO_PEDAL is exactly neutral by construction and is not treated as a quality winner. HUMAN is a reference distribution and is not assumed to be best. Across the 30 configurations, HUMAN > ALWAYS_ON fractions range from 0.750 to 0.750.

### Musical sanity observations

ALWAYS_ON is lower than HUMAN in 6/8 pieces for every configuration. The table below also compares it with the mean of the six saved model systems; values in the last two columns are fractions across all 30 configurations (and, for models, all six models). Lower means more negative/less consonant under this metric, not an overall quality judgment.

| Performance | Low frac. | ALWAYS baseline | HUMAN baseline | Model mean baseline | HUMAN > ALWAYS | Models > ALWAYS |
|---|---:|---:|---:|---:|---:|---:|
| `Beethoven/Piano_Sonatas/9-3/Tysman05M.mid` | 0.2067 | 0.000482 | 0.035963 | 0.036021 | 1.000 | 1.000 |
| `Beethoven/Piano_Sonatas/29-2/ChowK03.mid` | 0.1907 | 0.001352 | 0.101025 | 0.105411 | 1.000 | 1.000 |
| `Schumann/Kreisleriana/6/JohannsonP07.mid` | 0.1866 | 0.000253 | -0.001751 | 0.000074 | 0.000 | 0.850 |
| `Scriabin/Etudes_op_8/11/Shi03.mid` | 0.1081 | 0.002122 | 0.014248 | 0.021567 | 1.000 | 1.000 |
| `Ravel/Pavane/ChenS03.mid` | 0.1030 | 0.001196 | 0.032615 | 0.067467 | 1.000 | 1.000 |
| `Bach/Prelude/bwv_856/LuoJ01M.mid` | 0.0794 | -0.000826 | 0.068520 | -0.000191 | 1.000 | 0.989 |
| `Haydn/Keyboard_Sonatas/32-1/Pavlovic02.mid` | 0.0621 | -0.000010 | 0.025731 | 0.043935 | 1.000 | 0.833 |
| `Bach/Prelude/bwv_888/ZhaoK01M.mid` | 0.0610 | 0.001386 | -0.019757 | -0.061560 | 0.000 | 0.167 |

For the three most low-register-heavy cases, the two Beethoven performances show the expected tendency completely: HUMAN and every saved model score above ALWAYS_ON at all 30 configurations. Schumann/Kreisleriana is the counterexample: HUMAN is below ALWAYS_ON throughout, while model comparisons are above it in 85% of model/config cells. Thus the tendency is visible but not universal. No separate harmony-change detector was introduced, because the requested pilot evaluates only the finalized single metric; harmony-change-local claims should be checked during listening or a separately authorized descriptive audit.

## β_low sensitivity

The table uses κ=0 to isolate beta. `Delta H` is β=0.5 minus β=0 for each piece/system; all rows are in `beta_sensitivity.csv`.

| System | Spearman(low fraction, |Delta H|) | Mean |Delta H| |
|---|---:|---:|
| `ALWAYS_ON` | -0.143 | 0.000014 |
| `CUSTOM_EVENT_V0` | 0.095 | 0.000890 |
| `HUMAN` | 0.571 | 0.001289 |
| `HYBRID_REGRESSION_ONLY` | 0.862 | 0.001418 |
| `NO_PEDAL` | NA | 0.000000 |
| `ORIGINAL_PT` | 0.838 | 0.000963 |
| `STANDARD_CE_ARGMAX` | 0.571 | 0.001464 |
| `STANDARD_CE_POSTERIOR_MEDIAN` | 0.262 | 0.001586 |
| `WEIGHTED_CE_ARGMAX` | 0.571 | 0.001530 |

A positive correlation is the expected diagnostic tendency, but it is not universal here: it is strong for HYBRID_REGRESSION_ONLY (0.862) and ORIGINAL_PT (0.838), moderate for HUMAN and two 4-class systems (0.571), weak for CUSTOM_EVENT_V0 (0.095) and STANDARD_CE_POSTERIOR_MEDIAN (0.262), and slightly inverted for ALWAYS_ON (-0.143). Signed interval cancellation and system-specific pedal/pair composition can override low-note prevalence alone. NO_PEDAL remains exactly zero. This diagnostic therefore supports beta_low as behaving meaningfully in several systems, not as a universal monotone effect or ranking rule.

## κ_dyn onset sensitivity

This isolates dynamic sensitivity at β=0 and reports `H(kappa=0.4)-H(kappa=0)`. D=0.5 and H_base=0 onsets are excluded from the four directional cells.

| System | Condition | Onsets | Mean Delta H | Direction |
|---|---|---:|---:|---|
| `ALWAYS_ON` | H_base<0, D<0.5 | 2053 | -0.000164 | PASS |
| `ALWAYS_ON` | H_base<0, D>0.5 | 2486 | 0.000159 | PASS |
| `ALWAYS_ON` | H_base>0, D<0.5 | 2627 | -0.000760 | PASS |
| `ALWAYS_ON` | H_base>0, D>0.5 | 2820 | 0.000232 | PASS |
| `CUSTOM_EVENT_V0` | H_base<0, D<0.5 | 1252 | -0.016151 | PASS |
| `CUSTOM_EVENT_V0` | H_base<0, D>0.5 | 1206 | 0.016470 | PASS |
| `CUSTOM_EVENT_V0` | H_base>0, D<0.5 | 1851 | -0.018704 | PASS |
| `CUSTOM_EVENT_V0` | H_base>0, D>0.5 | 1395 | 0.028337 | PASS |
| `HUMAN` | H_base<0, D<0.5 | 858 | -0.010792 | PASS |
| `HUMAN` | H_base<0, D>0.5 | 856 | 0.018820 | PASS |
| `HUMAN` | H_base>0, D<0.5 | 2020 | -0.015902 | PASS |
| `HUMAN` | H_base>0, D>0.5 | 1434 | 0.024665 | PASS |
| `HYBRID_REGRESSION_ONLY` | H_base<0, D<0.5 | 363 | -0.009768 | PASS |
| `HYBRID_REGRESSION_ONLY` | H_base<0, D>0.5 | 431 | 0.018624 | PASS |
| `HYBRID_REGRESSION_ONLY` | H_base>0, D<0.5 | 1218 | -0.016067 | PASS |
| `HYBRID_REGRESSION_ONLY` | H_base>0, D>0.5 | 948 | 0.028860 | PASS |
| `ORIGINAL_PT` | H_base<0, D<0.5 | 464 | -0.009032 | PASS |
| `ORIGINAL_PT` | H_base<0, D>0.5 | 631 | 0.018284 | PASS |
| `ORIGINAL_PT` | H_base>0, D<0.5 | 1511 | -0.012636 | PASS |
| `ORIGINAL_PT` | H_base>0, D>0.5 | 1208 | 0.021498 | PASS |
| `STANDARD_CE_ARGMAX` | H_base<0, D<0.5 | 355 | -0.009385 | PASS |
| `STANDARD_CE_ARGMAX` | H_base<0, D>0.5 | 483 | 0.022581 | PASS |
| `STANDARD_CE_ARGMAX` | H_base>0, D<0.5 | 1156 | -0.016317 | PASS |
| `STANDARD_CE_ARGMAX` | H_base>0, D>0.5 | 943 | 0.030579 | PASS |
| `STANDARD_CE_POSTERIOR_MEDIAN` | H_base<0, D<0.5 | 340 | -0.010707 | PASS |
| `STANDARD_CE_POSTERIOR_MEDIAN` | H_base<0, D>0.5 | 486 | 0.021012 | PASS |
| `STANDARD_CE_POSTERIOR_MEDIAN` | H_base>0, D<0.5 | 1161 | -0.017148 | PASS |
| `STANDARD_CE_POSTERIOR_MEDIAN` | H_base>0, D>0.5 | 941 | 0.030279 | PASS |
| `WEIGHTED_CE_ARGMAX` | H_base<0, D<0.5 | 309 | -0.010825 | PASS |
| `WEIGHTED_CE_ARGMAX` | H_base<0, D>0.5 | 461 | 0.022922 | PASS |
| `WEIGHTED_CE_ARGMAX` | H_base>0, D<0.5 | 1064 | -0.017541 | PASS |
| `WEIGHTED_CE_ARGMAX` | H_base>0, D>0.5 | 861 | 0.030408 | PASS |

## Whole-sweep stability

Across representative model systems, 15/30 configurations satisfy the diagnostic stable-small-change screen (system-mean ranking Spearman >=0.9 and mean absolute change no larger than the configuration median). This is a descriptive screen, not hyperparameter selection.

| κ | β | Mean abs. change | Rank Spearman | Stable-small-change |
|---:|---:|---:|---:|---|
| 0.0 | 0.0 | 0.000000 | 1.000 | True |
| 0.0 | 0.1 | 0.000262 | 1.000 | True |
| 0.0 | 0.2 | 0.000523 | 1.000 | True |
| 0.0 | 0.3 | 0.000785 | 1.000 | True |
| 0.0 | 0.4 | 0.001047 | 1.000 | True |
| 0.0 | 0.5 | 0.001308 | 1.000 | True |
| 0.1 | 0.0 | 0.001767 | 0.983 | True |
| 0.1 | 0.1 | 0.001959 | 0.983 | True |
| 0.1 | 0.2 | 0.002174 | 0.983 | True |
| 0.1 | 0.3 | 0.002404 | 0.983 | True |
| 0.1 | 0.4 | 0.002638 | 0.983 | True |
| 0.1 | 0.5 | 0.002878 | 0.983 | True |
| 0.2 | 0.0 | 0.003534 | 0.983 | True |
| 0.2 | 0.1 | 0.003723 | 0.983 | True |
| 0.2 | 0.2 | 0.003921 | 0.983 | True |
| 0.2 | 0.3 | 0.004124 | 0.983 | False |
| 0.2 | 0.4 | 0.004350 | 0.983 | False |
| 0.2 | 0.5 | 0.004582 | 0.983 | False |
| 0.3 | 0.0 | 0.005300 | 0.983 | False |
| 0.3 | 0.1 | 0.005492 | 0.983 | False |
| 0.3 | 0.2 | 0.005684 | 0.983 | False |
| 0.3 | 0.3 | 0.005885 | 0.983 | False |
| 0.3 | 0.4 | 0.006089 | 0.983 | False |
| 0.3 | 0.5 | 0.006299 | 0.983 | False |
| 0.4 | 0.0 | 0.007067 | 0.967 | False |
| 0.4 | 0.1 | 0.007261 | 0.967 | False |
| 0.4 | 0.2 | 0.007454 | 0.967 | False |
| 0.4 | 0.3 | 0.007648 | 0.967 | False |
| 0.4 | 0.4 | 0.007852 | 0.967 | False |
| 0.4 | 0.5 | 0.008057 | 0.967 | False |

Detailed piece-balanced mean/median/IQR, HUMAN > ALWAYS_ON, model > ALWAYS_ON, baseline absolute change, and rank stability are in `summary_by_config.csv`; per-system sweep ranges are in `summary_by_system.csv`. `parameter_heatmaps.png` visualizes mean H_piece for every system.

## Conclusions

1. **Mathematical behavior.** The implementation matches the finalized document: PA/PP-only instance pairs, eta_PP=0.9, attack-clock decay, symmetric low weight, valid-onset averaging, and one onset-level dynamic modifier. All hard implementation checks pass, including exact NO_PEDAL zero and finite scores.
2. **Stable parameter area.** The descriptive screen retains all beta values at kappa=0.0, all beta values at kappa=0.1, and beta<=0.2 at kappa=0.2 (15/30 configurations). System-mean rank Spearman remains 0.983 through kappa=0.3 and 0.967 at kappa=0.4, but absolute changes grow with kappa. Because the beta/low-register tendency is not universal, the broad retained area is a stability region, not an optimum.
3. **Listening candidates (not final parameters).** The following small, stable configurations are recommended for direct listening: `(0.0,0.0)`, `(0.0,0.1)`, `(0.1,0.0)`, `(0.1,0.1)`. Baseline is retained as an anchor; nonzero candidates emphasize small changes and multi-piece stability rather than maximum separation.

## Diagnostic files

- `onset_components.csv`: one row per valid onset with A/P sizes, PA/PP counts, beta-linear numerator components, D, mean W_dec, and signed contribution sums.
- `onset_diagnostics_reference_configs.csv`: explicit H_base/W_dyn/H_n/W_low at `(0,0)` and `(0.4,0.5)`.
- `top_pairs.csv`: up to 20 strongest positive and 20 strongest negative beta=0 pair contributions per piece/system.
- `midi_identity_audit.csv`, `parser_corner_cases.csv`, `implementation_audit.csv`, `synthetic_sanity_checks.csv`: audit evidence.

## Missing files and limitations

- No requested representative validation MIDI was missing in the 19-piece intersection or selected eight cases.
- `Metric_harmonic_consonance.md` has no numeric Hall table; the task-specified paper values were therefore used, with no conflict to report.
- HUMAN and CUSTOM_EVENT_V0 use performance-level non-pedal timing/velocity, while conventional systems and extremes use piece-level canonical Stage 1. HUMAN is not described as an exact paired pedal-only comparison.
- The symbolic threshold ignores continuous CC64 depth once ON/OFF membership is determined.
- Same-tick state uses merged MIDI source order and evaluates after the complete tick group. FIFO matches overlapping note instances of the same channel/pitch.

## All selected input MIDI paths

| Piece | System | Path | SHA256 |
|---|---|---|---|
| `piece_18921bc81d708d16` | `ALWAYS_ON` | `/workspace/project/analysis/harmonic_metric_kdyn_blow_pilot_v0/reusable_extremes/always_on/piece_18921bc81d708d16.mid` | `2475dea4e6dae0c78ea76c7be3db50071f704a0e6a4509871ee959931841aab0` |
| `piece_18921bc81d708d16` | `CUSTOM_EVENT_V0` | `/workspace/project/analysis/custom_event_model_v0_canonical_val_inference_v1/candidate_midi/0013_3e23b8995de55693.mid` | `d524943ca8e0fa56c1d50f6492b20327fd10e1836e48b1c63c0da6af73f449aa` |
| `piece_18921bc81d708d16` | `HUMAN` | `/workspace/public/ASAP/asap-dataset-v1.1/Beethoven/Piano_Sonatas/9-3/Tysman05M.mid` | `cc3da12dd1c6dfceac556f121461fc1bcf8cd5fbf01e655c7a2657f231791cb5` |
| `piece_18921bc81d708d16` | `HYBRID_REGRESSION_ONLY` | `/workspace/project/analysis/stage2_encoder_only_raw_huber_aux_ce_v1/validation_eval/predictions/raw_huber_aux_ce/piece_18921bc81d708d16/candidate.mid` | `69e18900fa3cb3810d520fb27ac6c5cf5610472c2d0ec017fa82cd781530a018` |
| `piece_18921bc81d708d16` | `NO_PEDAL` | `/workspace/project/analysis/harmonic_metric_kdyn_blow_pilot_v0/reusable_extremes/no_pedal/piece_18921bc81d708d16.mid` | `d90d3dfe341b21d3819ec957bad6c70abe85084a15c792be0c646b696163f7a5` |
| `piece_18921bc81d708d16` | `ORIGINAL_PT` | `/workspace/project/analysis/stage2_binary_canonical_v1/canonical_validation_stage1/piece_18921bc81d708d16/original_pt.mid` | `64f3d683e862bbc7f41fb255c1996bfa455052d77601e9ff7e76600799aaf748` |
| `piece_18921bc81d708d16` | `STANDARD_CE_ARGMAX` | `/workspace/project/analysis/stage2_encoder_only_4class_decoding_phase4_v0/predictions/argmax/piece_18921bc81d708d16/candidate.mid` | `31292f713b4702428260561dccd355205959147ebe2abb6e5354ee81eb03b69a` |
| `piece_18921bc81d708d16` | `STANDARD_CE_POSTERIOR_MEDIAN` | `/workspace/project/analysis/stage2_encoder_only_4class_decoding_phase4_v0/predictions/median/piece_18921bc81d708d16/candidate.mid` | `95ba609b7c7d4b2ae5ce4e8033dc3ab4243bc6410d079def93f2650793fdd257` |
| `piece_18921bc81d708d16` | `WEIGHTED_CE_ARGMAX` | `/workspace/project/analysis/stage2_encoder_only_4class_loss_phase3_v0/weighted_ce/validation/predictions/weighted_ce/piece_18921bc81d708d16/candidate.mid` | `6e66e7fea8cb1de4b16179bc9d5bd35f9728d6eaf573aab7a6fad666fe587073` |
| `piece_3056e93b3e51b646` | `ALWAYS_ON` | `/workspace/project/analysis/harmonic_metric_kdyn_blow_pilot_v0/reusable_extremes/always_on/piece_3056e93b3e51b646.mid` | `ae2800306896d44f8f9f900f0ae76a9e9b8fc6650657d353558c8db6e3599f0a` |
| `piece_3056e93b3e51b646` | `CUSTOM_EVENT_V0` | `/workspace/project/analysis/custom_event_model_v0_canonical_val_inference_v1/candidate_midi/0065_fc67068dd61b26bc.mid` | `a05ff84bcf4c9d71a5df0ef0525f96b72bf2437ab4e7e218ddd5a7791895c9c7` |
| `piece_3056e93b3e51b646` | `HUMAN` | `/workspace/public/ASAP/asap-dataset-v1.1/Schumann/Kreisleriana/6/JohannsonP07.mid` | `95d8b965727979f2091ec687821523b2d25245ffce862b76ac0071abd26e7a11` |
| `piece_3056e93b3e51b646` | `HYBRID_REGRESSION_ONLY` | `/workspace/project/analysis/stage2_encoder_only_raw_huber_aux_ce_v1/validation_eval/predictions/raw_huber_aux_ce/piece_3056e93b3e51b646/candidate.mid` | `807ef880fb764ec74d5b8e343ebf605de26b1aff7d02135c6a8be301d1fd3913` |
| `piece_3056e93b3e51b646` | `NO_PEDAL` | `/workspace/project/analysis/harmonic_metric_kdyn_blow_pilot_v0/reusable_extremes/no_pedal/piece_3056e93b3e51b646.mid` | `b9ad03081bda41cdc3d36af0be2a658bea56ce92b4d8d1c15a02f44576c3199d` |
| `piece_3056e93b3e51b646` | `ORIGINAL_PT` | `/workspace/project/analysis/stage2_binary_canonical_v1/canonical_validation_stage1/piece_3056e93b3e51b646/original_pt.mid` | `849f64935dc844e8419cdca339ca9be4874e10230820a94f163708108346c21f` |
| `piece_3056e93b3e51b646` | `STANDARD_CE_ARGMAX` | `/workspace/project/analysis/stage2_encoder_only_4class_decoding_phase4_v0/predictions/argmax/piece_3056e93b3e51b646/candidate.mid` | `2974cf578b60e75e0085a1780987f39ae83fe36d4ece35abd13ac80b85a5d83e` |
| `piece_3056e93b3e51b646` | `STANDARD_CE_POSTERIOR_MEDIAN` | `/workspace/project/analysis/stage2_encoder_only_4class_decoding_phase4_v0/predictions/median/piece_3056e93b3e51b646/candidate.mid` | `e73fa6235eff99f093bb6e49b1e4a69d1d2ff13c41ed699fd068fb6c5b9fb982` |
| `piece_3056e93b3e51b646` | `WEIGHTED_CE_ARGMAX` | `/workspace/project/analysis/stage2_encoder_only_4class_loss_phase3_v0/weighted_ce/validation/predictions/weighted_ce/piece_3056e93b3e51b646/candidate.mid` | `15e638d8e40d2618e92279ed3f0dbb2f96f89da8364e0d5e4176b4735f913997` |
| `piece_555ff6cd3a13efb9` | `ALWAYS_ON` | `/workspace/project/analysis/harmonic_metric_kdyn_blow_pilot_v0/reusable_extremes/always_on/piece_555ff6cd3a13efb9.mid` | `117977144af9a126cbb3f9a80f07cc59b5ac37bbd487f98c345bca31ff95bccf` |
| `piece_555ff6cd3a13efb9` | `CUSTOM_EVENT_V0` | `/workspace/project/analysis/custom_event_model_v0_canonical_val_inference_v1/candidate_midi/0068_31810356da755956.mid` | `01b09fed12525e29f7e938ffe67dbb114e64bbd3ac15580527dbfe1102194806` |
| `piece_555ff6cd3a13efb9` | `HUMAN` | `/workspace/public/ASAP/asap-dataset-v1.1/Scriabin/Etudes_op_8/11/Shi03.mid` | `3d3c0230868fc6548c42862077455889a44ab817524886f60a79a3bd188390ee` |
| `piece_555ff6cd3a13efb9` | `HYBRID_REGRESSION_ONLY` | `/workspace/project/analysis/stage2_encoder_only_raw_huber_aux_ce_v1/validation_eval/predictions/raw_huber_aux_ce/piece_555ff6cd3a13efb9/candidate.mid` | `37c982efe77fce86b3ce7ef3164da0fc3a756da89eb63231bd684712d3ab1eee` |
| `piece_555ff6cd3a13efb9` | `NO_PEDAL` | `/workspace/project/analysis/harmonic_metric_kdyn_blow_pilot_v0/reusable_extremes/no_pedal/piece_555ff6cd3a13efb9.mid` | `a1fc54eb442c6ba11b72b1d71d9fefafacdd6167da2cc4b67c76a4ba2b16e896` |
| `piece_555ff6cd3a13efb9` | `ORIGINAL_PT` | `/workspace/project/analysis/stage2_binary_canonical_v1/canonical_validation_stage1/piece_555ff6cd3a13efb9/original_pt.mid` | `3bfa454309844c2eea155224b7b5c80f1ec1769e87401dea830445afd80b0090` |
| `piece_555ff6cd3a13efb9` | `STANDARD_CE_ARGMAX` | `/workspace/project/analysis/stage2_encoder_only_4class_decoding_phase4_v0/predictions/argmax/piece_555ff6cd3a13efb9/candidate.mid` | `2d40c0a196127fb6e4ef2999049a520adff84a7c0a96f736c69e1e3fdf1386b4` |
| `piece_555ff6cd3a13efb9` | `STANDARD_CE_POSTERIOR_MEDIAN` | `/workspace/project/analysis/stage2_encoder_only_4class_decoding_phase4_v0/predictions/median/piece_555ff6cd3a13efb9/candidate.mid` | `cd9459bac69df5d400059da597ebd072acf29e5cf1864c5647dc7723cc066be0` |
| `piece_555ff6cd3a13efb9` | `WEIGHTED_CE_ARGMAX` | `/workspace/project/analysis/stage2_encoder_only_4class_loss_phase3_v0/weighted_ce/validation/predictions/weighted_ce/piece_555ff6cd3a13efb9/candidate.mid` | `d8aab117bf056d459eaedd19612b751e40b3fb1cdc739078f1630ab945b7b55f` |
| `piece_ab42b317dcb38fc5` | `ALWAYS_ON` | `/workspace/project/analysis/harmonic_metric_kdyn_blow_pilot_v0/reusable_extremes/always_on/piece_ab42b317dcb38fc5.mid` | `3fafad62f9eb96c19f5155788882b2af4726dc9f4a6bff2837097d355b248329` |
| `piece_ab42b317dcb38fc5` | `CUSTOM_EVENT_V0` | `/workspace/project/analysis/custom_event_model_v0_canonical_val_inference_v1/candidate_midi/0010_5eb68e9a6b4e37f0.mid` | `894c074c4caf89164e06f8a6e5c6660428bd024073acf5658f8d50df705fa19f` |
| `piece_ab42b317dcb38fc5` | `HUMAN` | `/workspace/public/ASAP/asap-dataset-v1.1/Beethoven/Piano_Sonatas/29-2/ChowK03.mid` | `8b9cbde74ed4ecc3c2701097f5feee57ebc35bcfc3086ce1a5b942a5771661c9` |
| `piece_ab42b317dcb38fc5` | `HYBRID_REGRESSION_ONLY` | `/workspace/project/analysis/stage2_encoder_only_raw_huber_aux_ce_v1/validation_eval/predictions/raw_huber_aux_ce/piece_ab42b317dcb38fc5/candidate.mid` | `9333aac46919e1569f783d697b3e412b9e4b79cfc584aba0677ff2ece5ff194c` |
| `piece_ab42b317dcb38fc5` | `NO_PEDAL` | `/workspace/project/analysis/harmonic_metric_kdyn_blow_pilot_v0/reusable_extremes/no_pedal/piece_ab42b317dcb38fc5.mid` | `d01786767d97f3bf45229a27c9f15b80f1d60e2e0ad315341aa6996568e36a40` |
| `piece_ab42b317dcb38fc5` | `ORIGINAL_PT` | `/workspace/project/analysis/stage2_binary_canonical_v1/canonical_validation_stage1/piece_ab42b317dcb38fc5/original_pt.mid` | `f4f9024bc29e3c68f1241d9109f8ff0d4ec2b13ee74e87ca24dde288b0b6dd2e` |
| `piece_ab42b317dcb38fc5` | `STANDARD_CE_ARGMAX` | `/workspace/project/analysis/stage2_encoder_only_4class_decoding_phase4_v0/predictions/argmax/piece_ab42b317dcb38fc5/candidate.mid` | `5b11bb949a893bf8c21780242b071065387ebb6e2dd2e443d810e0ea0d280ea2` |
| `piece_ab42b317dcb38fc5` | `STANDARD_CE_POSTERIOR_MEDIAN` | `/workspace/project/analysis/stage2_encoder_only_4class_decoding_phase4_v0/predictions/median/piece_ab42b317dcb38fc5/candidate.mid` | `15199657ece4a76cd4b813e5cde4d51d2d7b0598ace599c17586e9effd5f540c` |
| `piece_ab42b317dcb38fc5` | `WEIGHTED_CE_ARGMAX` | `/workspace/project/analysis/stage2_encoder_only_4class_loss_phase3_v0/weighted_ce/validation/predictions/weighted_ce/piece_ab42b317dcb38fc5/candidate.mid` | `f049bddea8d186851fb43d3236ed92857988087a299431f4f357d048aa68f3da` |
| `piece_b06ab236f236fd76` | `ALWAYS_ON` | `/workspace/project/analysis/harmonic_metric_kdyn_blow_pilot_v0/reusable_extremes/always_on/piece_b06ab236f236fd76.mid` | `476500f7c0b23312e881c049e55ae7797bb75f6742d6559d8f51f3478524823a` |
| `piece_b06ab236f236fd76` | `CUSTOM_EVENT_V0` | `/workspace/project/analysis/custom_event_model_v0_canonical_val_inference_v1/candidate_midi/0002_2f32579335955a87.mid` | `aaf9dce455c6083bb026e8b2d85cf0e0803af4c94991c0b9f46d98c67a0fad4e` |
| `piece_b06ab236f236fd76` | `HUMAN` | `/workspace/public/ASAP/asap-dataset-v1.1/Bach/Prelude/bwv_888/ZhaoK01M.mid` | `f1679b6684c917ad5b68ea629520ba2ec2cf57873a68549d93c03419a5e3c7d5` |
| `piece_b06ab236f236fd76` | `HYBRID_REGRESSION_ONLY` | `/workspace/project/analysis/stage2_encoder_only_raw_huber_aux_ce_v1/validation_eval/predictions/raw_huber_aux_ce/piece_b06ab236f236fd76/candidate.mid` | `18015bcb8b70bdec3895e3954708b073e246bc9a06864768378816d5b7debc67` |
| `piece_b06ab236f236fd76` | `NO_PEDAL` | `/workspace/project/analysis/harmonic_metric_kdyn_blow_pilot_v0/reusable_extremes/no_pedal/piece_b06ab236f236fd76.mid` | `e46f941b375326bb5ea123a19d0f85b89b3b9b550725c6156e3ab000fbdffbd4` |
| `piece_b06ab236f236fd76` | `ORIGINAL_PT` | `/workspace/project/analysis/stage2_binary_canonical_v1/canonical_validation_stage1/piece_b06ab236f236fd76/original_pt.mid` | `a0d5cd9fb69e24f836da07d4c9e610f54a39ff6c81f323159429802dbfa5278e` |
| `piece_b06ab236f236fd76` | `STANDARD_CE_ARGMAX` | `/workspace/project/analysis/stage2_encoder_only_4class_decoding_phase4_v0/predictions/argmax/piece_b06ab236f236fd76/candidate.mid` | `7b57bef58f7b558edba5245ef1acd5f30a096411fefd4db316a1fc03c753c2eb` |
| `piece_b06ab236f236fd76` | `STANDARD_CE_POSTERIOR_MEDIAN` | `/workspace/project/analysis/stage2_encoder_only_4class_decoding_phase4_v0/predictions/median/piece_b06ab236f236fd76/candidate.mid` | `f902d544653a222a78a41fd2844badd49632653ae47bf39cdd29b0fe9a674bd0` |
| `piece_b06ab236f236fd76` | `WEIGHTED_CE_ARGMAX` | `/workspace/project/analysis/stage2_encoder_only_4class_loss_phase3_v0/weighted_ce/validation/predictions/weighted_ce/piece_b06ab236f236fd76/candidate.mid` | `227174e244b5fb26292866d87f19b204a637f0e2f160547f7fc24cc307756eb6` |
| `piece_de3f82957f1b3532` | `ALWAYS_ON` | `/workspace/project/analysis/harmonic_metric_kdyn_blow_pilot_v0/reusable_extremes/always_on/piece_de3f82957f1b3532.mid` | `ca70f31efb983823efd632d81ceeb996ad6672ec7701c5d9b693976b7077e966` |
| `piece_de3f82957f1b3532` | `CUSTOM_EVENT_V0` | `/workspace/project/analysis/custom_event_model_v0_canonical_val_inference_v1/candidate_midi/0057_02e9e365a0e76c67.mid` | `6c7582e2b72e100200d383f457002aff5bb547d554004b02aadeb9d905218669` |
| `piece_de3f82957f1b3532` | `HUMAN` | `/workspace/public/ASAP/asap-dataset-v1.1/Ravel/Pavane/ChenS03.mid` | `d0d00566c9a6aacbb06261720878dbcbb3943ceade9f86ebc7a53c8a0b2836f4` |
| `piece_de3f82957f1b3532` | `HYBRID_REGRESSION_ONLY` | `/workspace/project/analysis/stage2_encoder_only_raw_huber_aux_ce_v1/validation_eval/predictions/raw_huber_aux_ce/piece_de3f82957f1b3532/candidate.mid` | `c8987ed79d04f630e80fe5f1d30c1d27667966f23fbd3f9bf3a57821efa106f5` |
| `piece_de3f82957f1b3532` | `NO_PEDAL` | `/workspace/project/analysis/harmonic_metric_kdyn_blow_pilot_v0/reusable_extremes/no_pedal/piece_de3f82957f1b3532.mid` | `24cfb150883557340c8ace8f992833f6664113ed48cc613edcfc8bfdc11fe2e3` |
| `piece_de3f82957f1b3532` | `ORIGINAL_PT` | `/workspace/project/analysis/stage2_binary_canonical_v1/canonical_validation_stage1/piece_de3f82957f1b3532/original_pt.mid` | `921ec36ea43a03a29a1d54a764512ae4d6eebefd3f93b55bddf36a7b4af5b6e6` |
| `piece_de3f82957f1b3532` | `STANDARD_CE_ARGMAX` | `/workspace/project/analysis/stage2_encoder_only_4class_decoding_phase4_v0/predictions/argmax/piece_de3f82957f1b3532/candidate.mid` | `e243203778b879574854237c4a81dae40cdeef5794e73cd93387e44ce65829be` |
| `piece_de3f82957f1b3532` | `STANDARD_CE_POSTERIOR_MEDIAN` | `/workspace/project/analysis/stage2_encoder_only_4class_decoding_phase4_v0/predictions/median/piece_de3f82957f1b3532/candidate.mid` | `0448658de9e13a11d981eb9e47decf36670bb0d1fdf268af9a0edc357e40f512` |
| `piece_de3f82957f1b3532` | `WEIGHTED_CE_ARGMAX` | `/workspace/project/analysis/stage2_encoder_only_4class_loss_phase3_v0/weighted_ce/validation/predictions/weighted_ce/piece_de3f82957f1b3532/candidate.mid` | `d1943092e527d1ae10273c35264ad912a0cc60e090e98c743917c0ba646dbcaf` |
| `piece_e7ad71b998a8d477` | `ALWAYS_ON` | `/workspace/project/analysis/harmonic_metric_kdyn_blow_pilot_v0/reusable_extremes/always_on/piece_e7ad71b998a8d477.mid` | `545b0e1ad8c5b9685af8f4141eab32a2354ad6beaf29601d4506a6d5b0445ffc` |
| `piece_e7ad71b998a8d477` | `CUSTOM_EVENT_V0` | `/workspace/project/analysis/custom_event_model_v0_canonical_val_inference_v1/candidate_midi/0023_e5c4d8f3fecea654.mid` | `a919b53128a04b20ecbe9cc6884520655f3bea2bc11b98e998ca579b1199e6e7` |
| `piece_e7ad71b998a8d477` | `HUMAN` | `/workspace/public/ASAP/asap-dataset-v1.1/Haydn/Keyboard_Sonatas/32-1/Pavlovic02.mid` | `0ce3aed85c88d24d7e9429bb93bc054c509aab983cadf67122da5381098ecc62` |
| `piece_e7ad71b998a8d477` | `HYBRID_REGRESSION_ONLY` | `/workspace/project/analysis/stage2_encoder_only_raw_huber_aux_ce_v1/validation_eval/predictions/raw_huber_aux_ce/piece_e7ad71b998a8d477/candidate.mid` | `9796dcc87ecdf80425de5fb7fa5d2f48c44eff5e3e4186e04abdeee4bbf311b1` |
| `piece_e7ad71b998a8d477` | `NO_PEDAL` | `/workspace/project/analysis/harmonic_metric_kdyn_blow_pilot_v0/reusable_extremes/no_pedal/piece_e7ad71b998a8d477.mid` | `9b711b715f2a26a94257584cdfadc0ab0b2a276fbea4e86617ce93353dd9ace0` |
| `piece_e7ad71b998a8d477` | `ORIGINAL_PT` | `/workspace/project/analysis/stage2_binary_canonical_v1/canonical_validation_stage1/piece_e7ad71b998a8d477/original_pt.mid` | `1bc838580130004df5823589d401f9fca21025bca5958967029321a81073019a` |
| `piece_e7ad71b998a8d477` | `STANDARD_CE_ARGMAX` | `/workspace/project/analysis/stage2_encoder_only_4class_decoding_phase4_v0/predictions/argmax/piece_e7ad71b998a8d477/candidate.mid` | `67966360305af15fe2081847de93633a71638c88696d3244e4bf04dfd175d3d5` |
| `piece_e7ad71b998a8d477` | `STANDARD_CE_POSTERIOR_MEDIAN` | `/workspace/project/analysis/stage2_encoder_only_4class_decoding_phase4_v0/predictions/median/piece_e7ad71b998a8d477/candidate.mid` | `0528bc50678be66d6e73d6f65f499112ed0d0ee6f00143e07a2cc000fe2678f5` |
| `piece_e7ad71b998a8d477` | `WEIGHTED_CE_ARGMAX` | `/workspace/project/analysis/stage2_encoder_only_4class_loss_phase3_v0/weighted_ce/validation/predictions/weighted_ce/piece_e7ad71b998a8d477/candidate.mid` | `b769430f0cff65c9ed05f92c7c23b0dab518cf746161f8ee6eecc2d8f3498a2e` |
| `piece_f10784ae79f0f210` | `ALWAYS_ON` | `/workspace/project/analysis/harmonic_metric_kdyn_blow_pilot_v0/reusable_extremes/always_on/piece_f10784ae79f0f210.mid` | `61cd97aefdae6434e3b5c97e4f303a5b0aa3a79b023d90ad67a678ae232a7067` |
| `piece_f10784ae79f0f210` | `CUSTOM_EVENT_V0` | `/workspace/project/analysis/custom_event_model_v0_canonical_val_inference_v1/candidate_midi/0000_f69b41563e17d414.mid` | `8567085bb94ece19c35f5b3c267473661e75d45547dd0ccc54cda096844cde15` |
| `piece_f10784ae79f0f210` | `HUMAN` | `/workspace/public/ASAP/asap-dataset-v1.1/Bach/Prelude/bwv_856/LuoJ01M.mid` | `912f0757abbc008a326a70c9c249f4636bf2350de06d0c5430be0cc3836b4b71` |
| `piece_f10784ae79f0f210` | `HYBRID_REGRESSION_ONLY` | `/workspace/project/analysis/stage2_encoder_only_raw_huber_aux_ce_v1/validation_eval/predictions/raw_huber_aux_ce/piece_f10784ae79f0f210/candidate.mid` | `02a845df0b2da237edde2d902bb0f8106ac92124594c5c433d8aa487dd8e19da` |
| `piece_f10784ae79f0f210` | `NO_PEDAL` | `/workspace/project/analysis/harmonic_metric_kdyn_blow_pilot_v0/reusable_extremes/no_pedal/piece_f10784ae79f0f210.mid` | `02a845df0b2da237edde2d902bb0f8106ac92124594c5c433d8aa487dd8e19da` |
| `piece_f10784ae79f0f210` | `ORIGINAL_PT` | `/workspace/project/analysis/stage2_binary_canonical_v1/canonical_validation_stage1/piece_f10784ae79f0f210/original_pt.mid` | `02a845df0b2da237edde2d902bb0f8106ac92124594c5c433d8aa487dd8e19da` |
| `piece_f10784ae79f0f210` | `STANDARD_CE_ARGMAX` | `/workspace/project/analysis/stage2_encoder_only_4class_decoding_phase4_v0/predictions/argmax/piece_f10784ae79f0f210/candidate.mid` | `8d4a20f402c2f6abd8b6c13b52c5f6ba044e697d9c65cccdc2f147b7cf8aaf25` |
| `piece_f10784ae79f0f210` | `STANDARD_CE_POSTERIOR_MEDIAN` | `/workspace/project/analysis/stage2_encoder_only_4class_decoding_phase4_v0/predictions/median/piece_f10784ae79f0f210/candidate.mid` | `448f37362bc53d2eba023468da5c31dda881e54e2066021d2fcdc38060675ee6` |
| `piece_f10784ae79f0f210` | `WEIGHTED_CE_ARGMAX` | `/workspace/project/analysis/stage2_encoder_only_4class_loss_phase3_v0/weighted_ce/validation/predictions/weighted_ce/piece_f10784ae79f0f210/candidate.mid` | `f4b3f477d99034ec2986f54b94d0f1e8db51d72a589c93786df69bf8a6da6175` |

The ASAP test set was not accessed.
