# Binary 2-Slot Transition Follow-up v0

## Scope and frozen inputs

- Prior audit: `/workspace/project/analysis/binary_2slot_transition_audit_v0/audit_summary.json`; its capacity/PRE/POST scientific counts were not recomputed or changed in-place.
- Follow-up universe: canonical train 2,062 + validation 71 only. ASAP split CSV/test metadata/test MIDI access: 0.
- Frozen current policy: prepend tau=0 correction on mismatch, append human transitions, then max-2 odd/even last-retention.
- No model/head/checkpoint/training/inference, capacity change, horizon tuning, or policy adoption was performed.

## Canonical class-weight normalization correction

| N | Count | Frequency | Raw 1/sqrt(f) | Canonical normalized weight |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 7,885,843 | 0.87487564725279 | 1.06912094070683 | 0.757781679128814 |
| 1 | 896,884 | 0.0995026111996741 | 3.17017152727111 | 2.2469842387276 |
| 2 | 230,946 | 0.0256217415475356 | 6.24734769179123 | 4.42805434234336 |

`sum_c f_c*w_c = 1`; tolerance rtol/atol `1e-12`, PASS. Legacy global helper는 수정하지 않았다.

## Correction survival

| Split | Interval | Total | Correction retained | Correction discarded |
| --- | --- | ---: | ---: | ---: |
| train | PRE | 2,062 | 2,008 (97.381183%) | 54 (2.618817%) |
| train | MAIN | 9,009,549 | 8,770,480 (97.346493%) | 239,069 (2.653507%) |
| train | POST | 2,062 | 2,056 (99.709020%) | 6 (0.290980%) |
| train | ALL | 9,013,673 | 8,774,544 (97.347042%) | 239,129 (2.652958%) |
| validation | PRE | 71 | 69 (97.183099%) | 2 (2.816901%) |
| validation | MAIN | 272,927 | 266,700 (97.718438%) | 6,227 (2.281562%) |
| validation | POST | 71 | 71 (100.000000%) | 0 (0.000000%) |
| validation | ALL | 273,069 | 266,840 (97.718892%) | 6,229 (2.281108%) |

Correction은 human K=0/1에서만 생존하고 K>=2에서 모두 discard된다. K bucket별 exact counts는 `correction_survival.csv`에 있다.

## Piecewise trajectory recovery

| Split | Interval | Discarded | Recovery tau median | Q3 | p95 | p99 | max | Mismatch fraction median | Q3 | p95 | p99 | max |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | PRE | 54 | 0.18072916666666666 | 0.4586334296875001 | 0.71484375 | 0.8922958508749999 | 0.9708333333333333 | 0.20937499999999998 | 0.4676564557291668 | 0.71484375 | 0.8922958508749999 | 0.9708333333333333 |
| train | MAIN | 239,069 | 0.11346863468634381 | 0.21868365180468635 | 0.45945945945928884 | 0.6802721088435557 | 0.9865157717312751 | 0.11532385466034827 | 0.2222222222222957 | 0.46938775510156083 | 0.6893439246837298 | 0.9865157717312751 |
| train | POST | 6 | 0.029296875 | 0.05283449218751457 | 0.25474019531247905 | 0.3073575390624683 | 0.3205118749999656 | 0.029296875 | 0.05283449218751457 | 0.25474019531247905 | 0.3073575390624683 | 0.3205118749999656 |
| train | ALL | 239,129 | 0.11347517730331326 | 0.21874999999981348 | 0.45962732919268334 | 0.680320914058561 | 0.9865157717312751 | 0.11532385466035092 | 0.2222222222223839 | 0.4693877551026299 | 0.68956526611534 | 0.9865157717312751 |
| validation | PRE | 2 | 0.30818670833333334 | 0.3773842291666667 | 0.4327422458333333 | 0.44381384916666666 | 0.44658175 | 0.30818670833333334 | 0.3773842291666667 | 0.4327422458333333 | 0.44381384916666666 | 0.44658175 |
| validation | MAIN | 6,227 | 0.1334922526817548 | 0.2533536808855186 | 0.5336160013455665 | 0.7071737786023815 | 0.9163636363636493 | 0.1367673179395981 | 0.25848967613453666 | 0.542173728363513 | 0.7094991818606947 | 0.9163636363636493 |
| validation | POST | 0 | None | None | None | None | None | None | None | None | None | None |
| validation | ALL | 6,229 | 0.13355048859931873 | 0.25339366515837103 | 0.5336033864094795 | 0.7071499928643099 | 0.9163636363636493 | 0.13698630136985876 | 0.25862068965512497 | 0.5420197515131372 | 0.7094495091164204 | 0.9163636363636493 |

Recovery tau와 mismatch fraction은 first-retained-event proxy가 아니다. Human/raw와 compressed-target state를 모든 event timestamp에서 각각 재생하고, 구간별 state inequality 길이를 `[0,1]`에서 적분했다.

## Optional correction-reserved comparison

Alternative diagnostic은 correction을 slot 1에 고정한다. 남은 한 real slot은 state-valid와 final-state parity를 보존할 수 있을 때만 사용한다: odd K는 마지막 human transition 1개, even K는 0개다. 실제 tokenizer에는 채택하지 않았다.

| Split | Policy | Immediate correction | Retained real transitions | Final state | UP->DOWN pair retention |
| --- | --- | ---: | ---: | ---: | ---: |
| train | current_final_state_priority | 97.347042% | 1,136,013/1,378,598 (82.403500%) | 100.000000% | 2,206/205,798 (1.071925%) |
| train | correction_reserved_parity_valid | 100.000000% | 896,884/1,378,598 (65.057689%) | 100.000000% | 0/205,798 (0.000000%) |
| validation | current_final_state_priority | 97.718892% | 29,089/35,436 (82.088836%) | 100.000000% | 61/5,365 (1.136999%) |
| validation | correction_reserved_parity_valid | 100.000000% | 22,860/35,436 (64.510667%) | 100.000000% | 0/5,365 (0.000000%) |

## Unit/invariant tests

- Execution: `direct plain test-function runner` (`python tests/test_binary_2slot_transition_audit_v0.py`).
- pytest available: `False`.
- Result: 8 test functions passed, 0 failed.
- Added coverage includes K=2 correction discard, K=5/6 compression, invalid/same-direction rejection, epsilon boundaries, same-timestamp tau=0 correction+human transition, and canonical weight normalization.

## Conclusions

**Q1. Canonical fixed class weights는?** N=0: 0.757781679128814, N=1: 2.2469842387276, N=2: 4.42805434234336. Weighted frequency sum은 1이다.

**Q2. Counterfactual mismatch에서 correction survival은?** Train ALL 97.347042% (8,774,544/9,013,673), validation ALL 97.718892% (266,840/273,069)이다.

**Q3. Correction discard 시 mismatch는 언제까지 지속되는가?** Train ALL recovery tau median/Q3/p95/p99/max는 `{'count': 239129, 'median': 0.11347517730331326, 'q3': 0.21874999999981348, 'p95': 0.45962732919268334, 'p99': 0.680320914058561, 'max': 0.9865157717312751}`이고 mismatch-duration fraction은 `{'count': 239129, 'median': 0.11532385466035092, 'q3': 0.2222222222223839, 'p95': 0.4693877551026299, 'p99': 0.68956526611534, 'max': 0.9865157717312751}`이다. Validation 값은 본문 표와 JSON에 있다.

**Q4. Current policy의 final interval-end state preservation은 100%인가?** 예. Train/validation의 PRE/MAIN/POST/ALL 모두 100%다.

**Q5. 두 policy의 transition-information trade-off는?** Current policy는 train real-transition retention 82.403500%, UP->DOWN pair retention 1.071925%인 대신 immediate correction은 97.347042%다. Reserved policy는 immediate correction 100%와 final state 100%를 보장하지만 real-transition retention이 65.057689%, UP->DOWN pair retention이 0.000000%로 낮아진다.

**Q6. 구현 전 blocking issue가 있는가?** Final-state correctness 측면의 blocker는 없다. 다만 correction token 자체가 K>=2에서 사라지고 piecewise mismatch가 지속되는 것은 의도적으로 수용해야 하는 measured behavior다. 자동 policy 변경은 하지 않았다.

**Q7. 추가 tests는 모두 통과했는가?** 예. 8/8 direct test functions가 통과했다.
