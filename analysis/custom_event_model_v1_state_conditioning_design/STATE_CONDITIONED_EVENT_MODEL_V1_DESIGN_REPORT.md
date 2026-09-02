# State-Conditioned Event Model v1 — Design Freeze Report

## Decision

**RECOMMENDED V1: B3-S — slotwise soft predicted pre-state conditioning.**

각 Main slot에서 h_i로 네 상태 posterior q_(i,k)를 직접 예측하고, 이 posterior 자체를 고정 4차원 identity state feature로 Main 5-way event head에 입력한다. GT pre-state는 auxiliary CE label로만 사용하며 conditioning input으로 사용하지 않는다. Initial, Main timing, Terminal, Tokenizer v1, Prediction Decoder v1, weighted CE는 동결한다.

**READY TO IMPLEMENT STATE-CONDITIONED EVENT MODEL V1**

B3가 auxiliary state head와 conditioning을 함께 포함한다는 점은 명시적으로 인정한다. 이것을 하나의 atomic formulation으로 freeze하며 B1/B2/B4/C, hard mask, weight 변경을 섞지 않는다.

## Frozen provenance

- Tokenizer cache ID: 3a5520155b5db1e9a1da7f8148556aa3e1da852655c9adde25d3dbbd1d966263
- Note alignment ID: 98de59ef7a41fba26a2c89fe686c273f6c8ec7f27979922e71813624ac1a4822
- Decoder v1 ID/version: 938ca7e20e1d398aa530938bbcede404fd3a04af23842fbcf351ba02df0268f8 / 1.0.0
- Custom v0 best epoch/checkpoint SHA: epoch 1 / a81c5f8f0a902523bd8abb1ca5b1097876887477f1668907f08e2280d7d096c8
- Frozen raw-prediction digest: 839c7705ac51a87d722209362756ab20677ee218de751edf5c743b029fb263e0
- Target cache: train 2,062, validation 71 performances
- Frozen v0 prediction universe: validation 71 performances, 272,927 modeled onsets

Source artifact의 수치도 재확인했다. Custom v0는 4C Accuracy 0.350434, Macro F1 0.288461, Transition F1 0.563977, 256-pattern JS 0.128919, Intersection 0.646315이다. Matched Hybrid는 0.423644, 0.324311, 0.344890, 0.015307, 0.926481이다. 이 값은 provenance와 design motivation으로만 사용했다.

## Exact current-state semantics

Canonical target state machine:

~~~text
current = Initial state

for Main interval i = 1..I:
    assert current == cached main_interval_start_state[i]
    for slot k = 1..6 in canonical tokenizer slot order:
        s_pre[i,k] = current
        y = main_event_target[i,k]
        if y == NONE:
            current is unchanged
        else:
            current = destination(y)

for Terminal slot k = 1..4:
    s_terminal_pre[k] = current
    if target is SET:
        current = destination(target)
~~~

Slot1은 첫 interval에서 Initial, 이후 interval에서 이전 Main interval의 target-final state이다. Slot2–6은 같은 interval의 앞선 target SET을 반영한다. First-NONE 이후 cached NONE slot도 state가 변하지 않으므로 pre-state가 정의된다.

B3-S는 이 semantic state를 hard rollout하지 않고 각 slot에서 posterior로 재추정한다. q의 label은 위 state machine으로 생성되지만 train/inference conditioning source는 모두 model posterior다. Decoder는 별도의 hard current state를 기존과 동일하게 관리한다.

## Target transition audit

Active Main event matrix. 행은 pre-state, 열은 destination이다.

Train:

| pre / destination | ZERO | LOW | HALF | FULL |
|---|---:|---:|---:|---:|
| ZERO | 12 | 435,249 | 61,059 | 3,009 |
| LOW | 360,738 | 392 | 613,245 | 10,133 |
| HALF | 136,262 | 535,509 | 411 | 525,930 |
| FULL | 1,996 | 13,318 | 523,501 | 129 |

Validation:

| pre / destination | ZERO | LOW | HALF | FULL |
|---|---:|---:|---:|---:|
| ZERO | 1 | 11,158 | 2,016 | 144 |
| LOW | 9,310 | 14 | 15,013 | 492 |
| HALF | 3,906 | 13,127 | 16 | 13,428 |
| FULL | 82 | 520 | 13,439 | 7 |

- Train active Main 3,220,893; NONE 50,836,401
- Validation active Main 82,673; NONE 1,554,889
- Train same-side depth changes 1,845,418; cross OFF/ON threshold destinations 1,374,531
- Validation same-side depth changes 47,335; cross threshold destinations 35,300
- Frozen same-state SET targets: Main train 944, Terminal train 3, train total 947; Main validation 38, Terminal validation 0

핵심 pair는 train ZERO→LOW 435,249, LOW→ZERO 360,738, HALF→FULL 525,930, FULL→HALF 523,501이다. Full distribution과 slot별 matrix는 state_transition_matrix.json에 있다.

## Conditional entropy

Base-2 empirical Shannon entropy:

| split | H(Destination given Event) | H(Destination given Event, CurrentState) | reduction | fraction |
|---|---:|---:|---:|---:|
| Train | 1.901859 bits | 0.961241 bits | 0.940618 bits | 49.46% |
| Validation | 1.910747 bits | 1.014852 bits | 0.895895 bits | 46.89% |

NONE 포함 5-way target은 train 0.439106→0.358094 bits(18.45%), validation 0.384920→0.311363 bits(19.11%)다. NONE가 압도적이므로 destination bottleneck에는 active-only 결과가 더 직접적이다. 이는 model metric이 아니라 target structure diagnostic이다.

## B1 exposure mismatch

Frozen v0 prefix-active Main prediction 86,793개에서 same interval/slot GT pre-state와 predicted rollout pre-state를 비교했다.

- 4-state exact match: **0.346560**
- binary OFF/ON match: **0.619290**
- exact mismatch: 65.34%

| GT / predicted | ZERO | LOW | HALF | FULL |
|---|---:|---:|---:|---:|
| ZERO | 2,251 | 6,894 | 5,545 | 3,087 |
| LOW | 1,047 | 8,430 | 5,032 | 2,136 |
| HALF | 992 | 8,092 | 8,995 | 4,220 |
| FULL | 1,571 | 6,588 | 11,510 | 10,403 |

Slot exact match는 Slot1 0.3351, Slot2 0.3426, Slot3 0.3563, Slot4 0.3120, Slot5 0.3877, Slot6 0.4519이다.

Decoder first-NONE/tau sort/same-time collapse 뒤 86,754 decision position의 exact match는 0.302487, binary match는 0.605194다. 이 보조 정의는 predicted tau보다 엄격히 이른 target event만 oracle state에 반영한다.

Consecutive mismatch run 17,111개의 p95는 10 decisions, 33 onset intervals, 2.891 s다. p99는 16 decisions, 99.9 intervals, 7.159 s이며 최대는 67 decisions, 548 intervals, 46.209 s다. 이는 v1 performance simulation이 아니라 frozen-v0 state-source exposure의 descriptive length다.

Frozen v0에는 raw tau order violation 3,250건과 same-time collapse 39건도 있었다. 실제 tau-sorted 적용 직전 hard state를 head에 넣는 B1/B2는 현재 parallel head 및 shuffled owner-window training과 순환 의존을 만든다.

## Candidate conclusion

전체 표는 candidate_comparison.csv에 있다.

- **B1:** 단순하지만 GT input과 predicted input source shift가 지나치게 크다. exact mismatch 65.34%와 p95 2.891 s persistence 때문에 배제한다.
- **B2:** source는 일치하지만 hard argmax가 비미분이고 early error propagation이 있다. v0 train loader가 35,573 owner windows를 epoch마다 shuffle하므로 exact piece carry는 sampler/forward 변경을 요구하고 predicted-tau sort와도 충돌한다.
- **B3-S:** GT state를 input으로 쓰지 않고 slot별 predicted posterior를 쓴다. Parallel owner-window training과 Decoder v1을 유지하고 discrete error chain을 제거한다.
- **B4:** schedule이라는 새 독립변수와 prior scheduled-sampling failure 때문에 배제한다.
- **C:** NONE imbalance와 destination competition에 타당한 별도 실험이다. 그러나 current state가 active destination entropy를 약 절반 줄인다는 직접 evidence가 있고 C는 state-conditioned redundancy를 직접 다루지 않으므로 먼저 시행할 근거가 없다.

## Scientific-control details

1. State feature는 learned embedding이 아니라 q 자체인 fixed 4D identity basis다.
2. GT state는 auxiliary label일 뿐 event input이 아니다.
3. Encoder→state head와 q→event head에서 stop-gradient한다. State CE가 encoder를 별도로 재형성하거나 event CE가 q를 비-semantic latent code로 바꾸지 못한다.
4. State CE coefficient는 1.0으로 구조적으로 고정하며 sweep하지 않는다.
5. C, hard mask, weights, timing, decoder, tokenizer, checkpoint criterion을 바꾸지 않는다.

관찰되는 차이는 “명시적으로 학습된 model-predicted semantic pre-state posterior를 Main event head에 추가 feature로 제공”하는 B3-S atomic formulation에 귀속한다.

## Hard mask

**사용하지 않는다.** Main same-state target이 train 944, validation 38개 있다. Train mask는 label을 invalid로 만들며 frozen tokenizer grammar를 실질적으로 바꾼다. Inference-only mask는 train/inference mismatch이고 Decoder의 same-state suppression과 중복되는 heuristic이다.

## Scope, loss, checkpoint

- Initial, encoder, window/stride, ownership, alignment: unchanged
- Main event: B3-S only
- Main timing: unchanged
- Terminal event/timing: unchanged
- Tokenizer cache: unchanged; existing main_interval_start_state와 Main targets로 pre-state label runtime derivation
- Decoder v1: unchanged
- Main inverse-sqrt slot-specific weighted CE: unchanged
- New state CE: unweighted slot-macro, fixed unit coefficient, owned Main positions
- Selection: legacy v0 validation total을 그대로 사용하고 auxiliary state CE는 selection score에서 제외, 별도 logging

Terminal primary contribution은 60 samples뿐이고 target same-state no-op도 train 3/validation 0이다. Shared head가 아니므로 Main-only가 가능하다.

## Failure coverage

직접 겨냥:

- predicted redundant same-state SET 28,586
- ZERO↔LOW under-generation
- HALF↔FULL under-generation
- four-state destination confusion

간접 겨냥:

- mismatch persistence: conditioning error 자체는 slot-local이지만 잘못된 emitted event가 Decoder trajectory에 남는 문제는 지속될 수 있다
- Transition advantage: five-way occurrence vocabulary와 frozen weighted CE를 유지해 보존을 기대하지만 결과는 후속 full training에서 확인한다

## Required research questions

**Q1. GT conditioning mismatch?** Raw-prefix exact 34.656%, binary 61.929%; chronological exact 30.249%다. B1 shift는 크다.

**Q2. Entropy reduction?** Active destination에서 train 0.940618 bits(49.46%), validation 0.895895 bits(46.89%) 감소한다.

**Q3. B1/B2/B3/B4 선택?** B3-S.

**Q4. C를 먼저?** 아니다. 별도 controlled follow-up으로 남긴다.

**Q5. Hard mask 충돌?** 그렇다. Frozen Main target train 944/validation 38개를 invalid로 만든다.

**Q6. Main-only?** 가능하다.

**Q7. Weighted CE 유지?** 가능하며 exact 유지한다.

**Q8. 새 tunable hyperparameter 없이?** 가능하다. Fixed 4D feature와 non-swept unit state-loss coefficient를 사용한다.

**Q9. Exposure bias 제한?** GT conditioning과 hard autoregressive rollout을 모두 피하고 train/inference에 동일 predicted q를 사용한다.

**Q10. 정확한 independent variable?** Detached slotwise semantic state posterior branch와 그 posterior를 추가 feature로 받는 Main 5-way heads라는 B3-S 하나다.

**Q11. 어떤 failure?** Redundant SET과 same-side depth under-generation을 직접, persistence를 간접 겨냥한다.

**Q12. 다음 task 준비?** Tensor, target, mask, loss, initialization, interface가 freeze되어 준비되었다.

## Boundary compliance

- Model/source implementation: 0
- Training/tiny-overfit/full training: 0
- New candidate inference/evaluator: 0
- Tokenizer/Decoder change: 0
- ASAP test: 0
- Repedal: 0
