# Binary 2-Slot Transition Audit v0

## 1. Scope / frozen specification

- Binary state: `CC64 < 64 -> OFF`, `CC64 >= 64 -> ON`; 실제 threshold crossing만 DOWN/UP으로 남겼다.
- PRE `[t1-1s,t1)`, MAIN `[t_i,t_(i+1))`, final MAIN `[t_M,T_end)`, POST `[T_end,T_end+1s]`를 그대로 적용했다.
- max-2 odd/even last-retention, PRE default OFF correction, counterfactual mismatch correction을 변경 없이 사용했다.
- 모델/head/checkpoint/training/inference와 horizon/capacity tuning은 수행하지 않았다. ASAP test metadata/MIDI access는 0이다.

## 2. Dataset provenance

- Frozen Custom Event cache `3a5520155b5db1e9a1da7f8148556aa3e1da852655c9adde25d3dbbd1d966263`의 train 2,062개(MAESTRO-clean 1,170 + ASAP train 892)와 validation 71개(ASAP validation)만 사용했다.
- 각 원본 MIDI의 현재 SHA-256을 frozen manifest와 대조한 뒤 raw MIDI를 다시 parse했다.
- validation membership은 frozen cache provenance를 상속했고 이번 audit은 ASAP split CSV를 열지 않았다.
- owner audit은 alignment `98de59ef7a41fba26a2c89fe686c273f6c8ec7f27979922e71813624ac1a4822`와 기존 512-note/stride-256 maximum-margin owner 구현을 그대로 호출했다.

## 3. Binary crossing extraction validation

- 기존 parser의 equal-tick `(tick, track, message)`/last-value semantics 후 4-state effective event에서 threshold crossing만 투영한 결과를 독립 direct binary projection과 전 performance에서 exact 비교했다.
- MAESTRO-clean / ASAP train / ASAP validation 모두 projection mismatch 0, cache onset mismatch 0, latest-noteoff mismatch 0, source SHA mismatch 0이다.
- SMF cross-track same-tick ordering의 기존 deterministic limitation은 그대로 상속한다. 새로운 ordering semantics는 만들지 않았다.

## 4. 2-slot capacity results

| Dataset | Performances | MAIN intervals | Overflow >=3 | Raw crossings | Retained | Retention | Performances with overflow |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MAESTRO-clean | 1,170 | 6,127,542 | 7,004 (0.114304%) | 954,171 | 939,987 | 98.513474% | 849 (72.564103%) |
| ASAP-train | 892 | 2,882,007 | 2,774 (0.096252%) | 422,652 | 417,044 | 98.673140% | 527 (59.080717%) |
| ASAP-validation | 71 | 272,927 | 266 (0.097462%) | 35,376 | 34,840 | 98.484848% | 48 (67.605634%) |

| Dataset | K=0 | K=1 | K=2 | K=3 | K=4 | K=5+ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MAESTRO-clean | 5,351,812 (87.340274%) | 605,616 (9.883506%) | 163,110 (2.661916%) | 5,812 (0.094850%) | 1,122 (0.018311%) | 70 (0.001142%) |
| ASAP-train | 2,531,606 (87.841771%) | 281,446 (9.765625%) | 66,181 (2.296351%) | 2,301 (0.079840%) | 454 (0.015753%) | 19 (0.000659%) |
| ASAP-validation | 244,105 (89.439667%) | 22,595 (8.278771%) | 5,961 (2.184101%) | 207 (0.075844%) | 57 (0.020885%) | 2 (0.000733%) |

| Dataset | Direction | Raw | Retained | Retention |
| --- | --- | ---: | ---: | ---: |
| MAESTRO-clean | UP | 477,003 | 469,911 | 98.513217% |
| MAESTRO-clean | DOWN | 477,168 | 470,076 | 98.513731% |
| ASAP-train | UP | 211,228 | 208,424 | 98.672524% |
| ASAP-train | DOWN | 211,424 | 208,620 | 98.673755% |
| ASAP-validation | UP | 17,673 | 17,405 | 98.483562% |
| ASAP-validation | DOWN | 17,703 | 17,435 | 98.486132% |

| Dataset | Adjacent gesture | Raw pairs | Pair retained intact | Retention |
| --- | --- | ---: | ---: | ---: |
| MAESTRO-clean | UP->DOWN | 146,187 | 139,095 | 95.148679% |
| MAESTRO-clean | DOWN->UP | 32,254 | 25,162 | 78.012030% |
| ASAP-train | UP->DOWN | 59,589 | 56,785 | 95.294434% |
| ASAP-train | DOWN->UP | 12,662 | 9,858 | 77.854999% |
| ASAP-validation | UP->DOWN | 5,364 | 5,096 | 95.003729% |
| ASAP-validation | DOWN->UP | 1,190 | 922 | 77.478992% |

`UP->DOWN`은 요청된 repedal-like pair diagnostic일 뿐 새로운 repedal metric이 아니다. Overflow/performance 분포와 대표 예시는 JSON 및 CSV에 저장했다.

## 5. PRE 1-second audit

| Split | ON at PRE-left / synthetic DOWN | Crossings before PRE-left | Actual crossing retention | Initialized-target retention | Initialized final-state preservation |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 324 (15.712900%) | 330 | 96.839729% | 96.694215% | 100.000000% |
| validation | 5 (7.042254%) | 9 | 100.000000% | 92.857143% | 100.000000% |

PRE raw `0/1/2/3/4/5+` 분포와 first-onset 전 마지막 crossing delay의 median/Q3/p95/p99/max는 `pre_post_audit.csv`와 `audit_summary.json`에 있다. `actual` retention은 human crossing만, `initialized-target` retention은 필요한 synthetic DOWN까지 분모에 포함한다.

## 6. POST 1-second audit

| Split | Any crossing at/after T_end | Inside 1s | Truncated after 1s | Performances truncated | Inside retention | Horizon final-state preservation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 1,281 (62.124151%) | 889 | 423 | 411 (19.932105%) | 99.775028% | 100.000000% |
| validation | 51 (71.830986%) | 37 | 19 | 16 (22.535211%) | 100.000000% | 100.000000% |

POST right endpoint의 `tau=1`은 포함했고 강제 UP은 추가하지 않았다. 모든 T_end 이후 crossing delay와 POST 분포는 JSON/CSV에 기록했다.

## 7. Base `N=0/1/2` distribution

| Split | Interval | N=0 | N=1 | N=2 |
| --- | --- | ---: | ---: | ---: |
| train | PRE | 1,245 (60.378274%) | 776 (37.633366%) | 41 (1.988361%) |
| train | MAIN | 7,883,418 (87.500695%) | 895,231 (9.936469%) | 230,900 (2.562836%) |
| train | POST | 1,180 (57.225994%) | 877 (42.531523%) | 5 (0.242483%) |
| train | ALL | 7,885,843 (87.487565%) | 896,884 (9.950261%) | 230,946 (2.562174%) |
| validation | PRE | 50 (70.422535%) | 19 (26.760563%) | 2 (2.816901%) |
| validation | MAIN | 244,105 (89.439667%) | 22,804 (8.355348%) | 6,018 (2.204985%) |
| validation | POST | 34 (47.887324%) | 37 (52.112676%) | 0 (0.000000%) |
| validation | ALL | 244,189 (89.423918%) | 22,860 (8.371510%) | 6,020 (2.204571%) |

Base/oracle은 `s_model=s_human`인 실제 human sequence다. 별도로 실제 PRE 실행 조건(default OFF)은 `pre_default_off` 행으로 CSV/JSON에 저장했다. Train ALL candidate inverse-sqrt weights의 raw 값은 `{'0': 1.0691209407068252, '1': 3.170171527271109, '2': 6.247347691791226}`이다. 함께 기록한 `{'0': 0.3058522818800601, '1': 0.9069172239073644, '2': 1.7872304942125754}`는 audit-local legacy mean-present-class=1 diagnostic이며 project canonical weighted-CE normalization이 아니다. Canonical 보정값은 follow-up audit에 기록한다.

## 8. Counterfactual state-mismatch correction audit

| Split | Intervals checked | Recovered | Recovery | N=0 | N=1 | N=2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 9,013,673 | 9,013,673 | 100.000000% | 0 (0.000000%) | 8,116,789 (90.049739%) | 896,884 (9.950261%) |
| validation | 273,069 | 273,069 | 100.000000% | 0 (0.000000%) | 250,209 (91.628490%) | 22,860 (8.371510%) |

Case B는 실제 training distribution이 아니라 모든 interval에서 `s_model != s_human`을 가정한 counterfactual diagnostic이다. Correction 방향과 tau=0 prepend 후 동일 compression을 적용했다.

## 9. Owner-window/state-chain feasibility

| Split | Performances | Onsets | 0 owner | Duplicate owner | Non-monotonic performances | Offending transitions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 2,062 | 9,009,549 | 0 | 0 | 0 | 0 |
| validation | 71 | 272,927 | 0 | 0 | 0 | 0 |

기존 owner rule을 `PRE -> owned MAIN chronological order -> POST`의 단일 state chain에 그대로 사용할 수 있는가: **예.**

## 10. Edge-case invariant results

- 15/15 required synthetic cases passed: 15/15.
- 각 case의 interval assignment, compressed N, retained tau, reconstructed final state는 `invariant_results.csv`에 있다.

## 11. Information lost by the proposed representation

- MAIN/PRE/POST 안에서 3개 이상 crossing이 있으면 parity와 final state는 보존하지만 앞쪽 crossing timing 및 일부 adjacent two-event gesture는 손실된다.
- PRE-left보다 이른 crossing history/timing은 버리고 PRE-left binary state만 synthetic correction으로 전달한다.
- POST horizon 뒤 crossing은 전부 truncate한다. `T_end+1s`에서 ON인 trajectory에는 강제 UP을 넣지 않는다.
- Binary thresholding 자체가 CC64 depth 변화(0/LOW/HALF/FULL)를 모두 버린다.
- Equal-tick cross-track ordering은 기존 deterministic parser convention을 상속하며 SMF 차원의 의미적 ambiguity를 해결하지 않는다.

## 12. WARN conclusion

**Q1. Binary-only setting에서 max 2-slot이 실용적으로 타당한가?** 예. binary-only target에서 대부분의 interval은 2-slot 이내이며 실측 retention은 위 표와 같다. 다만 overflow에서 사라지는 중간 crossing/gesture가 있으므로 무손실 표현은 아니다.

**Q2. Odd/even last-retention이 final binary state를 실제 데이터 전체에서 100% 보존하는가?** 예. train/validation의 모든 MAIN, PRE, POST represented interval에서 final binary state를 100% 보존했다.

**Q3. PRE 1s truncation으로 어떤 정보가 손실되는가?** 1초보다 앞선 binary crossing은 train 330개, validation 9개가 event timing/history로는 사라진다. PRE-left state가 ON이면 tau=0 DOWN 하나로 state만 복원한다.

**Q4. POST 1s truncation으로 어떤 정보가 손실되는가?** T_end+1초 뒤 crossing은 train 423개, validation 19개가 잘린다. horizon 끝 ON은 그대로 허용했다.

**Q5. tau=0 state-reconciliation rule이 mismatch case에서 항상 human interval-end state를 복구 가능한 target으로 만드는가?** 예. 전수 검사 failure 수는 0이다.

**Q6. N=0/1/2 class imbalance는 어느 정도인가?** Train ALL base 분포는 N=0 7,885,843 (87.487565%), N=1 896,884 (9.950261%), N=2 230,946 (2.562174%)이다.

**Q7. 기존 owner-window rule을 performance-level hard free-running state carry에 그대로 사용할 수 있는가?** 예.

**Q8. Full model implementation 전에 반드시 설계를 수정해야 할 blocking issue가 있는가?** 없다. 측정된 정보 손실은 WARN 사항이나 hard invariant failure는 없다.

Overall status는 **WARN**다. WARN은 측정된 compression/horizon/depth 정보 손실을 명시하는 것이며 새로운 threshold나 자동 설계 변경을 뜻하지 않는다.
