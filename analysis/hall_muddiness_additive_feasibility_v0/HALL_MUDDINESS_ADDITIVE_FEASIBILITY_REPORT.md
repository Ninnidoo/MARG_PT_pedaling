# Additive Hall Severity + Conflict Amount Feasibility Audit v0

## 결론

**B. structure is plausible but unstable; more component audit needed**

이 감사는 최적 lambda를 선택하지 않았다. 기존 count-scaling audit의 164,685개 onset에서 `H=H_MEAN_n`, `RAW=RAW_n`을 그대로 재사용하고 `A=ln(1+RAW)`만 계산했다. MIDI parsing, Hall pair 재계산, grid search, inference, training, MIDI 생성은 모두 0회다.

- Tested command: `python /workspace/project/scripts/audit_hall_muddiness_additive_feasibility_v0.py`

## 고정 의미와 집계

- onset: `H_n=RAW_n/N_Q_n` (`N_Q=0`이면 0), `A_n=ln(1+RAW_n)`.
- piece primary: 모든 distinct onset(빈 Q 포함)의 `H_n`, `A_n` 평균.
- valid secondary: `Q != empty` onset만 평균.
- `A_piece`는 반드시 `mean_n[ln(1+RAW_n)]`이며 `ln(1+sum RAW_n)`이 아니다.
- PA/PP만 포함하고 A-A는 제외한다. decay, velocity, pedal depth, register/dynamic, PA/PP 차등, positive reward는 없다.
- NO_PEDAL은 구조적 0 control이지 전체 페달링 품질의 최선이라는 뜻이 아니다.

## System-level component table

H/A 칸은 5개 piece summary의 mean / median이다. RAW 칸은 해당 system의 모든 real-data onset을 pooled한 mean / median / p95 / p99이다.

| system | Hall severity H | log-conflict amount A | raw mass/onset |
| --- | --- | --- | --- |
| NO_PEDAL | 0.000000 / 0.000000 | 0.000000 / 0.000000 | 0.000 / 0.000 / 0.000 / 0.000 |
| ALWAYS_ON | 0.330383 / 0.330424 | 12.318609 / 11.914534 | 3560546.282 / 694492.533 / 16715251.368 / 20078009.089 |
| STANDARD_CE_ARGMAX | 0.101246 / 0.111996 | 1.261407 / 1.019695 | 43.691 / 0.200 / 145.542 / 950.828 |
| STANDARD_CE_POSTERIOR_MEDIAN | 0.099781 / 0.104831 | 1.254621 / 0.938981 | 43.597 / 0.171 / 148.283 / 912.124 |
| WEIGHTED_CE_ARGMAX | 0.091888 / 0.098571 | 1.102659 / 0.825028 | 33.750 / 0.000 / 95.350 / 699.612 |
| HYBRID_REGRESSION_ONLY | 0.099079 / 0.104116 | 1.311463 / 1.138829 | 63.484 / 0.029 / 222.324 / 1340.724 |
| CUSTOM_EVENT_V0 | 0.147551 / 0.154199 | 1.527817 / 1.399957 | 130.216 / 2.329 / 604.113 / 3068.402 |
| ORIGINAL_PT | 0.144645 / 0.137397 | 1.968120 / 1.510917 | 131.849 / 1.678 / 374.910 / 3541.841 |
| HUMAN | 0.159380 / 0.162983 | 1.696775 / 1.740691 | 50.402 / 2.831 / 121.563 / 796.926 |

네 canonical Stage2만 묶었을 때 mean H_piece 범위는 0.091888–0.101246, mean A_piece 범위는 1.102659–1.311463이다. CUSTOM_EVENT_V0는 이 canonical summary와 median에서 제외했다.

### Secondary valid-onset summaries

아래는 `Q != empty` onset만 사용한 piece summary를 다시 5곡에 대해 mean / median한 값이다.

| system | H_piece_valid | A_piece_valid |
| --- | --- | --- |
| NO_PEDAL | 0.000000 / 0.000000 | 0.000000 / 0.000000 |
| ALWAYS_ON | 0.330687 / 0.331094 | 12.329891 / 11.938682 |
| STANDARD_CE_ARGMAX | 0.201519 / 0.183046 | 2.227172 / 2.391277 |
| STANDARD_CE_POSTERIOR_MEDIAN | 0.200445 / 0.182832 | 2.212417 / 2.335585 |
| WEIGHTED_CE_ARGMAX | 0.199879 / 0.176891 | 2.072096 / 2.293297 |
| HYBRID_REGRESSION_ONLY | 0.205055 / 0.189937 | 2.361640 / 2.446027 |
| CUSTOM_EVENT_V0 | 0.233715 / 0.233357 | 2.387623 / 2.737214 |
| ORIGINAL_PT | 0.215769 / 0.227255 | 2.800960 / 2.462604 |
| HUMAN | 0.215953 / 0.208278 | 2.259036 / 2.406285 |

## Component-wise ordering

| component | comparison | count |
| --- | --- | --- |
| H | HUMAN <= PT | 1/5 |
| H | each canonical Stage2 < ALWAYS_ON | 20/20 |
| H | canonical median < ALWAYS_ON | 5/5 |
| H | PT < canonical median (descriptive) | 2/5 |
| A | HUMAN <= PT | 4/5 |
| A | each canonical Stage2 < ALWAYS_ON | 20/20 |
| A | canonical median < ALWAYS_ON | 5/5 |
| A | PT < canonical median (descriptive) | 2/5 |

PT 대 canonical median 방향은 기술 통계일 뿐 hard success criterion이 아니다.

## Analytical lambda feasibility

각 비교에서 `delta_H + lambda*delta_A <= 0` (model median 비교는 strict `<0`)을 직접 풀었다. endpoint의 열린/닫힌 여부를 보존했으며 lambda 후보를 sampling하거나 점수를 최적화하지 않았다.

| piece | HUMAN <= PT | ModelMedian < ALWAYS | intersection | positive region |
| --- | --- | --- | --- | --- |
| Ravel/Pavane | [0, +inf) | [0, +inf) | [0, +inf) | YES |
| Schumann/Arabeske | [0.102272855566, +inf) | [0, +inf) | [0.102272855566, +inf) | YES |
| Liszt/Mephisto_Waltz | EMPTY | [0, +inf) | EMPTY | NO |
| Beethoven/Piano_Sonatas_27-1 | [0.1492606497, +inf) | [0, +inf) | [0.1492606497, +inf) | YES |
| Chopin/Etudes_op_25_12 | [0.0269105671766, +inf) | [0, +inf) | [0.0269105671766, +inf) | YES |

- 두 조건의 piece별 교집합이 non-empty인 곡: 4/5
- non-trivial positive 구간이 있는 곡: 4/5
- 5곡 전체 common interval: `EMPTY`
- 최대 동시 overlap: 4/5

이는 feasibility geometry일 뿐 final lambda 또는 best weight 선택이 아니다.

## Scale diagnostic

| level | metric | n | median | p95 | p99 | max |
| --- | --- | --- | --- | --- | --- | --- |
| ONSET | H | 164685 | 0.040486 | 0.402583 | 0.584914 | 1.902000 |
| ONSET | A | 164685 | 0.616266 | 13.669834 | 16.437472 | 16.856731 |
| ONSET | A_over_H_where_H_positive | 89613 | 16.228114 | 51.081295 | 90.126517 | 313.134898 |
| PIECE_SYSTEM | H | 45 | 0.113224 | 0.325515 | 0.363614 | 0.364988 |
| PIECE_SYSTEM | A | 45 | 1.340773 | 11.902024 | 13.568526 | 14.730324 |
| PIECE_SYSTEM | A_over_H_where_H_positive | 40 | 11.234312 | 38.147873 | 39.942398 | 40.706735 |

`A/H`는 H>0인 행에서만 계산했다. 정규화, z-score, min-max scaling은 수행하지 않았다. system별 상세 scale은 `scale_diagnostics.csv`에 있다.

## Synthetic sanity

| case | pair count | RAW | H | A=ln(1+RAW) |
| --- | --- | --- | --- | --- |
| case_1_few_mild | 2 | 0.400 | 0.200 | 0.336 |
| case_2_many_mild | 20 | 4.000 | 0.200 | 1.609 |
| case_3_few_severe | 2 | 3.000 | 1.500 | 1.386 |
| case_4_many_severe | 20 | 30.000 | 1.500 | 3.434 |

Case 1/2는 H가 같지만 Case 2의 A가 더 크고, Case 1/3은 Case 3의 H가 더 크며, Case 3/4는 H가 같지만 Case 4의 A가 더 크다. 이 관계는 계산 후 assertion으로 검증했다.

## Normalized-position diagnostic

아래는 각 piece/bin 평균을 다시 5곡에 대해 piece-balanced 평균한 요약이다. corr은 bin index와 component의 Pearson correlation이며, `A up-steps`는 인접 9구간 중 증가 횟수다. 전체 500개 piece/system/bin 행(9 systems + derived canonical median)은 CSV에 있다.

| system | H bin0 | H bin9 | corr(H,pos) | A bin0 | A bin9 | corr(A,pos) | A up-steps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HUMAN | 0.125905 | 0.152700 | 0.649 | 1.273363 | 1.646179 | 0.588 | 6/9 |
| ORIGINAL_PT | 0.119969 | 0.135508 | -0.107 | 1.271559 | 2.422891 | 0.135 | 5/9 |
| CANONICAL_STAGE2_MEDIAN | 0.104431 | 0.091483 | 0.294 | 1.180428 | 1.155937 | -0.007 | 5/9 |
| ALWAYS_ON | 0.262567 | 0.355436 | 0.917 | 7.491484 | 14.300938 | 0.906 | 9/9 |

ALWAYS_ON에서 A가 후반으로 크게 증가하면서 H의 범위는 상대적으로 제한적이다. 이는 H가 평균 severity, A가 accumulated negative mass의 로그 양을 포착한다는 해석과 일치한다.

## Final questions

### Q1. `log(1+RAW)`는 Pure Hall mean이 놓치는 누적 conflict를 포착하는가?

그렇다. synthetic equal-severity 사례에서 H는 같고 A만 pair 수와 negative mass에 반응한다. 실제 ALWAYS_ON의 후반부에서도 H보다 A의 위치 증가가 훨씬 뚜렷하다.

### Q2. note count 자체를 penalty로 쓰지 않고 ALWAYS_ON을 더 분리하는가?

그렇다. 5곡 piece-balanced mean에서 ALWAYS_ON의 A는 12.318609이고 canonical Stage2 범위는 1.102659–1.311463이다. A는 N_P/N_Q가 아니라 Hall-negative RAW가 실제로 늘 때만 증가한다. 다만 ordering count 자체는 H에서도 이미 포화될 수 있어 주된 추가 정보는 separation margin과 accumulation trajectory다.

### Q3. HUMAN과 PT는 H와 A에서 어떻게 비교되는가?

Piece-balanced mean은 H에서 HUMAN 0.159380, PT 0.144645; A에서 HUMAN 1.696775, PT 1.968120이다. 곡별 `HUMAN <= PT` count는 위 ordering 표와 같다. 이 fine distinction은 일관된 hard separator가 아니다.

### Q4. 일반 Stage2와 ALWAYS_ON은 어떻게 비교되는가?

각 canonical model < ALWAYS_ON은 H와 A 모두 ordering 표의 /20 결과처럼 강하다. ALWAYS_ON은 특히 A와 RAW scale에서 정상 Stage2 범위를 크게 벗어난다.

### Q5. H와 A는 complementary한가, redundant한가?

개념적으로도 수치적으로도 complementary하다. H는 pair-average severity라 같은 평균 conflict의 multiplicity에 불변이고, A는 negative mass가 쌓일수록 증가한다. 둘 다 Hall-negative RAW에서 나오므로 독립 정보원은 아니지만, aggregation functional이 달라 동일한 현상을 중복 측정하는 수준은 아니다.

### Q6. `M=H+lambda*A`에 coarse relation을 보존하는 non-trivial positive region이 있는가?

5곡 전체에는 없다. 5곡 common interval은 `EMPTY`이다. 이 결과는 lambda 선택이 아니라 양의 가중치 조합의 존재 가능성만 말한다.

### Q7. 후속 weight-selection을 정당화할 만큼 broad/stable한가?

아니다. 전체 공통 양의 구간이 없어 component 또는 dataset 감사를 더 해야 한다.

## Integrity

| check | status | detail |
| --- | --- | --- |
| H_reproduces_previous_Pure_Hall | PASS | overlap_rows=82152; max_abs_error=0.000e+00 |
| RAW_crosschecks_previous_Pure_Hall_with_roundoff | PASS | overlap_rows=82152; max_abs_error=2.910e-11; tolerance=5e-11 |
| RAW_reproduces_prior_count_scaling | PASS | rows=164685; max_abs_error=0.000e+00 |
| H_reuses_prior_count_scaling | PASS | rows=164685; max_abs_error=0.000e+00 |
| A_equals_log1p_RAW | PASS | max_abs_error=0.000e+00 |
| H_equals_RAW_div_N_Q | PASS | valid_rows=92073; max_abs_error=4.441e-16 |
| Q_empty_implies_H_A_RAW_zero | PASS | rows=72612; max_abs_value=0.000e+00 |
| NO_PEDAL_implies_H_A_RAW_zero | PASS | rows=18261; max_abs_value=0.000e+00 |
| no_decay | PASS | CSV-only derivation; no evaluator or decay term called |
| no_velocity | PASS | velocity absent from input projection and formula |
| no_pedal_depth | PASS | no depth term; frozen source used CC64 binary state |
| no_low_or_dynamic_terms | PASS | absent |
| no_PA_PP_differential_weights | PASS | audited RAW reused without reweighting |
| no_positive_Hall_reward | PASS | audited RAW=sum max(-HallWeight,0) reused |
| no_AA_pairs | PASS | AA_pair_count=0 |
| all_finite | PASS | numeric_cells=2140905 |
| source_MIDI_unchanged | PASS | manifest_rows=45; unique_files=43 |
| TEST_access_zero | PASS | count=0 |
| inference_zero | PASS | count=0 |
| training_zero | PASS | count=0 |
| MIDI_generation_zero | PASS | count=0 |
| no_explicit_full_pair_materialization | PASS | no Hall evaluation; audited aggregate columns reused |
| fixed_five_piece_nine_system_inventory | PASS | pieces=5; systems=9; rows=45 |
| CUSTOM_EVENT_kept_separate | PASS | canonical median contains exactly four Stage2 systems |
| synthetic_behavior_verified | PASS | cases=4 |

모든 integrity check가 PASS한 뒤에만 이 보고서를 기록했다. Source CSV가 수치의 기준이며 prior audit 디렉터리는 수정하지 않았다.
