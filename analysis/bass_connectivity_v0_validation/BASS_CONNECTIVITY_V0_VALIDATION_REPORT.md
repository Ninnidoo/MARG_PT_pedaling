# Bass Connectivity v0 — ASAP Validation Evaluation

Generated: `2026-08-25T08:25:50.017525+00:00`

## Scope and provenance

This is a measurement-only sanity evaluation over pre-existing ASAP **validation** MIDI. Neural inference, training, checkpoint loading/execution, audio rendering, parameter search, and GPU/CUDA use were all zero. ASAP test MIDI access was zero. No MIDI was generated or modified.

- Human: 71 existing validation performances across 19 pieces from `/workspace/project/analysis/stage2_encoder_only_v0/asap_split.csv`.
- Original PT: 19 frozen canonical Stage-1 MIDI files from `/workspace/project/analysis/stage2_binary_canonical_v1/canonical_validation_stage1_manifest.csv`.
- Canonical 4-class systems recovered from the Harmonic Muddiness system inventory and its source reports: STANDARD_CE_ARGMAX, STANDARD_CE_POSTERIOR_MEDIAN, and WEIGHTED_CE_ARGMAX (19 existing pieces each). Source roots: `/workspace/project/analysis/stage2_encoder_only_4class_decoding_phase4_v0` and `/workspace/project/analysis/stage2_encoder_only_4class_loss_phase3_v0`.
- Existing Harmonic Muddiness extremes from `/workspace/project/analysis/harmonic_metric_pure_hall_negative_v0/input_midis.csv`: NO_PEDAL and ALWAYS_ON, 13 pieces each. The six unavailable pieces per control remain `missing`; no controls were synthesized.
- Excluded deliberately: HYBRID_REGRESSION_ONLY (regression) and CUSTOM_EVENT_V0 (noncanonical custom-event system). The Harmonic Muddiness report explicitly excludes CUSTOM_EVENT_V0 from its canonical Stage2 median, and the request forbids mixing regression systems.

## Exact fixed metric

Non-drum positive-velocity note attacks are sorted in merged MIDI event order and grouped by an inclusive earliest-anchor window of 20 ms (not single linkage). Each group uses its earliest time and lowest newly attacked pitch `p_n`. Held/sustained notes do not enter detection.

`L_n=1` for `p_n<=52`, `(57-p_n)/5` for `52<p_n<57`, and `0` for `p_n>=57`. The next-low attack is the first later group with `p<=57`. The local timescale is the median positive grouped IOI in the existing ±8-group convention. `R_B=Delta_B/local_IOI`, `S_n=clip((R_B-1.5)/1.5,0,1)`, and `O_oct=1` only when `p_n+12` is in the same group. `B_n=clip(L_n*S_n*(1+0.2*O_oct),0,1)`; Structural Bass iff `B_n>=0.5`. Terminal next-low and insufficient-timescale values remain missing and are never replaced by sentinels or detected.

Only consecutive Structural Bass pairs are considered. The final bass is excluded. A pair is a valid pedal opportunity only when physical `keyoff < next onset`; otherwise it is counted separately. At key-off, channel-specific `CC64>=64` is ON. Same-timestamp messages follow the Harmonic Muddiness merged sequential order and same-pitch notes use FIFO instance pairing. If pedal is OFF at key-off, `C_r=0`. Otherwise the first later OFF event is used. For different pitches the Gaussian has `sigma_L=d/2`, `sigma_R=d/4`; no later OFF gives 0. For exactly equal pitches, release at/after the next onset (including no later OFF) gives 1; early release uses the left Gaussian. No octave-equivalent exception and no decay/velocity weighting are used.

Performance scores average valid `C_r`; no-opportunity performances are NA. Human performances are averaged within piece, then all scored pieces receive equal weight in the primary system macro. The pooled transition-weighted mean is diagnostic only. Reported standard deviations are population standard deviations.

## Required synthetic logic tests

| Test | Expected | Actual | Result |
|---|---:|---:|---|
| 1_keyoff_at_pedal_off | 0 | 0.0 | PASS |
| 2_different_off_at_next | 1 | 1.0 | PASS |
| 3_different_left_half_gap | 0.60653066 | 0.6065306597126334 | PASS |
| 4_different_right_quarter_gap | 0.60653066 | 0.6065306597126334 | PASS |
| 5_same_connected_through_next | 1 | 1.0 | PASS |
| 6_no_subsequent_off | different=0;same=1 | different=0;same=1 | PASS |
| 7_keyoff_at_next_excluded | excluded | excluded_keyoff_at_or_after_next_bass | PASS |
| 8_earliest_anchor_grouping | [[0,15],[30]] | [[0,15],[30]] | PASS |

All 8 required tests passed.

## Main system comparison

Primary score is the piece-balanced macro mean.

| System | Perf. | Pieces | Structural bass | Valid | Excluded key-held | No opp. perf. | C mean | C median | C std | Micro | Piece macro |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Human | 71 | 19 | 13798 | 13182 | 545 | 0 | 0.385267 | 0.222336 | 0.406744 | 0.385267 | **0.326718** |
| Original PT | 19 | 19 | 1927 | 1810 | 98 | 0 | 0.225754 | 0.000000 | 0.353200 | 0.225754 | **0.187593** |
| Standard CE — argmax (4-class) | 19 | 19 | 1927 | 1810 | 98 | 0 | 0.266195 | 0.000000 | 0.382762 | 0.266195 | **0.220004** |
| Standard CE — posterior median (4-class) | 19 | 19 | 1927 | 1810 | 98 | 0 | 0.261888 | 0.000000 | 0.382560 | 0.261888 | **0.214062** |
| Weighted CE — argmax (4-class) | 19 | 19 | 1927 | 1810 | 98 | 0 | 0.255273 | 0.000000 | 0.376975 | 0.255273 | **0.208099** |
| No pedal / Always off | 13 | 13 | 1271 | 1167 | 91 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | **0.000000** |
| Always on | 13 | 13 | 1271 | 1167 | 91 | 0 | 0.263925 | 0.000000 | 0.440759 | 0.263925 | **0.251024** |

## Same-bass versus different-bass

| System | Same n | Same mean | Same median | Different n | Different mean | Different median | Pedal OFF at key-off | No later pedal OFF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Human | 3299 | 0.542639 | 0.634900 | 9883 | 0.332735 | 0.175979 | 3931 | 0 |
| Original PT | 453 | 0.331333 | 0.000000 | 1357 | 0.190509 | 0.000000 | 966 | 8 |
| Standard CE — argmax (4-class) | 453 | 0.308001 | 0.000000 | 1357 | 0.252239 | 0.000000 | 956 | 3 |
| Standard CE — posterior median (4-class) | 453 | 0.305845 | 0.000000 | 1357 | 0.247215 | 0.000000 | 982 | 3 |
| Weighted CE — argmax (4-class) | 453 | 0.287681 | 0.000000 | 1357 | 0.244454 | 0.000000 | 1012 | 2 |
| No pedal / Always off | 308 | 0.000000 | 0.000000 | 859 | 0.000000 | 0.000000 | 1167 | 0 |
| Always on | 308 | 1.000000 | 1.000000 | 859 | 0.000000 | 0.000000 | 0 | 1167 |

## Per-piece comparison

| Piece | Human | Original PT | Standard CE — argmax (4-class) | Standard CE — posterior median (4-class) | Weighted CE — argmax (4-class) | No pedal / Always off | Always on |
|---|---:|---:|---:|---:|---:|---:|---:|
| Bach — Prelude_bwv_862 (`piece_08aa5ff2ada461e0`) | 0.015742 | 0.016887 | 0.030682 | 0.029176 | 0.028986 | 0.000000 | 0.111111 |
| Haydn — Keyboard_Sonatas_46-1 (`piece_0a888a7d5fbd06e6`) | 0.092752 | 0.001423 | 0.008561 | 0.013860 | 0.008887 | 0.000000 | 0.496894 |
| Beethoven — Piano_Sonatas_9-3 (`piece_18921bc81d708d16`) | 0.173228 | 0.108695 | 0.192651 | 0.192572 | 0.169781 | 0.000000 | 0.102362 |
| Schumann — Kreisleriana_6 (`piece_3056e93b3e51b646`) | 0.554355 | 0.259302 | 0.210154 | 0.230094 | 0.227323 | 0.000000 | 0.186441 |
| Liszt — Gran_Etudes_de_Paganini_6_Theme_and_Variations (`piece_4b31472ad4376d9e`) | 0.284258 | 0.162696 | 0.151652 | 0.153204 | 0.143923 | NA | NA |
| Scriabin — Etudes_op_8_11 (`piece_555ff6cd3a13efb9`) | 0.557828 | 0.475906 | 0.511895 | 0.506820 | 0.497236 | 0.000000 | 0.384615 |
| Beethoven — Piano_Sonatas_27-1 (`piece_69862af5096ee3fa`) | 0.368087 | 0.219012 | 0.257542 | 0.259574 | 0.217774 | 0.000000 | 0.212389 |
| Liszt — Mephisto_Waltz (`piece_7195bbce81550519`) | 0.404451 | 0.229273 | 0.317836 | 0.318895 | 0.309573 | NA | NA |
| Schubert — Piano_Sonatas_664-1 (`piece_759b324c1cceac54`) | 0.515151 | 0.374009 | 0.444302 | 0.447414 | 0.453072 | 0.000000 | 0.229167 |
| Liszt — Concert_Etude_S145_2 (`piece_901ec1910fe9fa9e`) | 0.335315 | 0.183813 | 0.134098 | 0.099590 | 0.098260 | NA | NA |
| Beethoven — Piano_Sonatas_29-2 (`piece_ab42b317dcb38fc5`) | 0.317498 | 0.055882 | 0.118079 | 0.124425 | 0.134523 | 0.000000 | 0.233333 |
| Bach — Prelude_bwv_888 (`piece_b06ab236f236fd76`) | 0.118572 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | NA | NA |
| Beethoven — Piano_Sonatas_8-1 (`piece_b45364ab6817194a`) | 0.277960 | 0.136708 | 0.166423 | 0.152680 | 0.158791 | NA | NA |
| Schumann — Arabeske (`piece_daefdda4e1923cc6`) | 0.568459 | 0.528159 | 0.539490 | 0.509799 | 0.522877 | 0.000000 | 0.256637 |
| Chopin — Etudes_op_25_12 (`piece_db97fbaed2036f5b`) | 0.489341 | 0.478523 | 0.482733 | 0.482999 | 0.481425 | 0.000000 | 0.357143 |
| Ravel — Pavane (`piece_de3f82957f1b3532`) | 0.607552 | 0.256909 | 0.275443 | 0.283787 | 0.261386 | 0.000000 | 0.154762 |
| Haydn — Keyboard_Sonatas_39-2 (`piece_e38184a2e9753bdc`) | 0.324526 | 0.045115 | 0.170408 | 0.088369 | 0.089263 | 0.000000 | 0.205128 |
| Haydn — Keyboard_Sonatas_32-1 (`piece_e7ad71b998a8d477`) | 0.182058 | 0.031952 | 0.168122 | 0.173917 | 0.150806 | 0.000000 | 0.333333 |
| Bach — Prelude_bwv_856 (`piece_f10784ae79f0f210`) | 0.020502 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | NA | NA |

## Cross-system invariants

All `102` available canonical-variant comparisons passed strict non-CC64 event equality, exact Structural Bass set equality, and exact valid-opportunity set equality. Failures: `0`. HUMAN is intentionally excluded because it is a distinct performance.

## Missing systems and anomalies

There are `12` expected-but-missing inputs: six NO_PEDAL and six ALWAYS_ON piece variants. No primary HUMAN/ORIGINAL_PT/4-class MIDI is missing.

- NO_PEDAL: `piece_4b31472ad4376d9e`, `piece_7195bbce81550519`, `piece_901ec1910fe9fa9e`, `piece_b06ab236f236fd76`, `piece_b45364ab6817194a`, `piece_f10784ae79f0f210`
- ALWAYS_ON: `piece_4b31472ad4376d9e`, `piece_7195bbce81550519`, `piece_901ec1910fe9fa9e`, `piece_b06ab236f236fd76`, `piece_b45364ab6817194a`, `piece_f10784ae79f0f210`
- Structural-pair missing-keyoff exclusions: `0`.
- Detected groups with duplicate lowest-pitch attacks (deterministic earliest merged-event representative used): `0`.
- Parser diagnostics: CC120=`0`, CC123=`0`, unmatched note-off=`10`. These are exposed in `performance_scores.csv` rather than hidden.

## Measurement-only sanity interpretation

- NO_PEDAL piece macro is `0.000000` (micro `0.000000`); its key-off-at-pedal-OFF count is `1167` of `1167` valid opportunities.
- ALWAYS_ON same-bass mean is `1.000000` and different-bass mean is `0.000000`, directly testing the fixed no-OFF exception/asymmetry.
- On the common 19-piece universe, the measured primary ordering is: Human (0.326718) > Standard CE — argmax (4-class) (0.220004) > Standard CE — posterior median (4-class) (0.214062) > Weighted CE — argmax (4-class) (0.208099) > Original PT (0.187593).
- ALWAYS_ON is based on only the 13 pieces with existing control MIDI, so its macro is not presented as a common-19-piece rank against the main systems.
- No coefficient, threshold, event handling, or post-processing was changed after seeing this ordering.

## Output files and reproducibility

- `input_manifest.csv`: every expected input, including missing controls and frozen hashes.
- `structural_bass_events.csv`: all detected fixed-v0 Structural Bass events.
- `transition_diagnostics.csv`: every consecutive pair, including exclusions and score reasons.
- `performance_scores.csv`, `piece_scores.csv`, `system_summary.csv`: the three aggregation levels.
- `cross_system_invariants.csv`, `sanity_tests.csv`: identity and required logic audits.

Tested command (inside the project container, with CUDA hidden):

```bash
CUDA_VISIBLE_DEVICES='' python scripts/evaluate_bass_connectivity_v0_validation.py
CUDA_VISIBLE_DEVICES='' python -c "import runpy; d=runpy.run_path('tests/test_bass_connectivity_v0.py'); [d[n]() for n in sorted(d) if n.startswith('test_')]"
```

Run totals: existing MIDI opened=`173`, ASAP validation human MIDI opened=`71`, ASAP test MIDI opened=`0`, inference=`0`, training=`0`, GPU/CUDA=`0`, generated MIDI=`0`.
