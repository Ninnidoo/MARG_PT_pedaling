좋아. 오늘은 **두 번의 full training을 서로 다른 목적을 가진 실험으로 명확히 분리**해서 가는 게 좋겠어.

> **실험 1은 기존 Count-based 아이디어를 끝까지 한번 검증하는 실험**,
> **실험 2는 이번 실패를 반영해 representation 자체를 개선한 State-Anchored 모델을 검증하는 실험**.

둘 다 돌려두면 최종 발표에서도 단순히 “새 모델 하나 해봤다”가 아니라, **Count-only → failure analysis → state를 명시적으로 도입한 structural redesign**이라는 연구 흐름이 아주 깔끔하게 생겨.

---

# 오늘 전체 실험 계획

|             | 실험 1                          | 실험 2                                |
| ----------- | ----------------------------- | ----------------------------------- |
| 이름(임시)      | **Count-2Slot Base-Weighted** | **State-Anchored Transition v1**    |
| 핵심 출력       | (N\in{0,1,2}) + timing        | State + HOLD/CHANGE/RETURN + timing |
| State       | Count parity를 누적              | **각 onset에서 직접 예측**                 |
| Target      | online reconciliation 사용      | **static human target**             |
| Count/Mode  | Dynamic N                     | Static Mode                         |
| State drift | 가능                            | **직접 state anchor로 억제**             |
| Inference   | hard free-running             | **2-state constrained Viterbi**     |
| PRE/POST    | **제거**  (하지만 추후에 구현으로 변경)| **제거**                              |
| 목적          | 기존 아이디어 최선의 버전 확인             | 새로운 구조적 대안 확인                       |

오늘 두 모델 모두 **동일한 MAIN-only horizon**을 사용하게 하자. 그래야 나중에 Transition F1도 비교하기가 편해.

---

# 공통 시간축부터 새로 고정

PRE/POST는 이제 완전히 빼자.

Distinct onset:

[
t_1,t_2,\ldots,t_M
]

에 대해 모델링 interval은:

[
\boxed{
I_i=[t_i,t_{i+1}),\quad i=1,\ldots,M-1
}
]

만 사용.

따라서:

* 첫 onset 이전 구간 없음
* 마지막 onset 이후 tail 없음
* (t_i)에 정확히 발생한 pedal transition → (I_i)의 (\tau=0)
* (t_M) 이후 transition → 이번 모델링 범위 밖

이렇게 하면 두 실험 모두 boundary라는 부차적인 문제가 사라지고 **onset-to-onset pedal modeling 자체만 비교**할 수 있어.

---

# 실험 1 — 기존 N={0,1,2} 모델 Full Training

이건 우리가 최근 diagnostic에서 가장 좋았던 **Condition C를 MAIN-only로 가져가는 실험**이라고 보면 돼.

즉 기존 C와 거의 같지만 PRE/POST가 빠졌으므로 정확히는 새로운 run이야.

## 구조

각 interval에서:

[
N_i\in{0,1,2}
]

와

[
\tau_{i,1},\tau_{i,2}
]

를 예측.

State는 계속:

[
s_{i+1}
=======

s_i\oplus(\hat N_i\bmod2)
]

로 hard free-running.

Online reconciliation도 유지.

즉 human state와 model state가 다르면:

[
\text{correction@0}+\text{human transitions}
]

으로 dynamic target을 생성.

다만 Count CE의 weight는 우리가 diagnostic에서 가장 나았던 **base-count weighting**으로 간다.

[
\boxed{
L_i^{count}
===========

w[N_i^{base}]
,CE(\hat p_i,N_i^{dynamic})
}
]

기존처럼 dynamic label 자체의 class weight를 사용하는 방식은 폐기.

---

## 왜 굳이 이 모델도 full training하나?

Tiny에서는 C가 완벽히 안정적이지 않았지만 중요한 점이 있었지.

[
\text{base Count accuracy}=98.97%
]

까지 올라갔어.

그러니까 모델이 transition structure 자체를 배우는 능력은 굉장히 강해.

Full dataset에서는 tiny 한 곡에서의 moving-target oscillation이 어떻게 나타날지는 아직 몰라.

그래서:

> **현재 Count-only 계열에서 가장 좋은 formulation이 실제 validation에서는 어디까지 가는가**

를 한번 보는 건 충분히 가치가 있어.

최종적으로 실패하더라도 실험 2의 motivation이 강해져.

---

## 실험 1에서는 더 이상 tiny stability 때문에 막지 말자

우리가 이미:

* implementation sanity
* multi-window state carry
* A memorization
* B/C diagnostic

까지 충분히 했어.

그래서 오늘은 다시 tiny gate를 여러 번 반복하지 않고:

**짧은 compile/sanity → full training**

으로 바로 가도 된다고 봐.

단:

* NaN/Inf
* ownership 오류
* state lifecycle invariant 위반
* leakage

같은 **engineering failure만 hard stop**.

State agreement가 낮다고 학습 자체를 멈추지는 말자.

그게 이번 실험에서 알고 싶은 결과니까.

---

## 실험 1 학습

나는:

[
\boxed{\text{max 10 epochs}}
]

정도로 추천해.

매 epoch ASAP validation free-running 평가.

Primary metric:

[
\boxed{\text{Transition F1}}
]

그리고 반드시 같이 볼 것:

* Transition Precision
* Recall
* F1
* base Count accuracy
* dynamic Count accuracy
* predicted N distribution
* state agreement
* correction fraction
* mismatch-run length
* predicted/reference transition ratio
* timing MAE

특히 **state agreement curve가 epoch에 따라 안정화되는지**가 중요해.

Best checkpoint는 validation Transition F1.

---

# 실험 2 — State-Anchored Transition v1

이게 오늘의 메인 실험이라고 보면 돼.

다만 이건 새 tokenizer니까 바로 학습하지 말고:

> **Tokenizer audit → implementation → tiny overfit → full training**

순서로 가자.

Audit 자체는 너무 크게 벌리지 말고, full training으로 가기 위한 **구현 검증 수준**으로 제한하는 게 좋아.

---

# State-Anchored Transition Tokenizer v1

## A. State token

각 distinct onset (t_i)에서:

[
\boxed{
S_i\in{\mathrm{OFF},\mathrm{ON}}
}
]

를 직접 예측.

나는 정의를:

[
\boxed{
S_i=
\text{pedal state immediately before }t_i
}
]

로 추천해.

즉 (t_i)보다 strictly earlier한 CC64 event까지만 반영.

정확히 (t_i)에서 발생한 transition은 state에 미리 반영하지 않고:

[
\tau=0
]

의 interval event로 처리.

---

# B. Transition Mode

각 interval:

[
I_i=[t_i,t_{i+1})
]

에서:

[
\boxed{
M_i\in
{
HOLD,\ CHANGE,\ RETURN
}
}
]

를 예측.

### HOLD

transition 없음.

[
N=0
]

따라서:

[
S_i=S_{i+1}
]

Timing 없음.

### CHANGE

transition 하나.

[
N=1
]

따라서:

[
S_i\neq S_{i+1}
]

Timing:

[
\tau_1
]

### RETURN

transition 두 개.

[
N=2
]

따라서:

[
S_i=S_{i+1}
]

Timing:

[
\tau_1,\tau_2
]

ON에서 시작하면:

```text
ON → OFF → ON
```

이라 음악적으로 repedal.

OFF에서 시작하면:

```text
OFF → ON → OFF
```

이라 기술적으로 RETURN이지만 repedal이라고 부르지는 않음.

그래서 class 이름은 `RETURN`으로 하는 게 좋아.

---

# C. Raw transitions가 3개 이상이면?

여기는 기존 규칙 그대로 유지.

* K=0 → 0
* K=1 → 1
* K=2 → 2
* K≥3 odd → 마지막 1
* K≥4 even → 마지막 2

따라서:

[
0\rightarrow HOLD
]

[
1\rightarrow CHANGE
]

[
2\rightarrow RETURN
]

Mode로 변환.

Parity가 보존되기 때문에 State endpoint consistency도 유지돼야 해.

Audit에서 반드시:

[
CHANGE \iff S_i\neq S_{i+1}
]

[
HOLD/RETURN \iff S_i=S_{i+1}
]

가 **100%인지 확인**해야 해.

---

# 실험 2에서 가장 먼저 할 offline audit

여기는 하루짜리 대형 audit 필요 없어.

딱 다음 정도면 충분해.

### 1. State distribution

Train:

[
P(OFF),P(ON)
]

Validation도 distribution report 정도는 가능하지만, hyperparameter 결정은 train으로만.

### 2. Mode distribution

[
P(HOLD),P(CHANGE),P(RETURN)
]

### 3. RETURN composition

RETURN 중:

```text
ON → OFF → ON
```

비율과

```text
OFF → ON → OFF
```

비율.

### 4. Compression

* K≥3 fraction
* retained transition fraction
* endpoint State preservation

### 5. State/Mode consistency

**Violation = 반드시 0.**

### 6. Timing distribution

CHANGE:

[
\tau_1
]

RETURN:

[
\tau_1,\tau_2,\quad\tau_2-\tau_1
]

정도.

이 정도면 tokenizer가 실제 human MIDI에 자연스럽게 적용되는지 충분히 확인할 수 있어.

---

# 실험 2 모델 구조

이전보다 오히려 단순해질 수 있어.

PT encoder hidden:

[
h_i\in\mathbb R^{768}
]

에서 직접:

[
\boxed{
StateHead: Linear(768,2)
}
]

[
\boxed{
ModeHead: Linear(768,3)
}
]

[
\boxed{
TimingHead_1: Linear(768,1)
}
]

[
\boxed{
TimingHead_2: Linear(768,1)
}
]

State를 다시 head input으로 넣지 않아.

즉 기존:

[
[h_i;s_i]
]

구조를 버리고:

[
h_i
]

만 사용.

이게 중요해.

**recurrent model state 자체가 training graph에서 사라져.**

---

# 실험 2 Loss

첫 버전은 최대한 단순하게 가자.

[
\boxed{
L
=

L_{\text{state}}
+
L_{\text{mode}}
+
L_{\text{timing}}
}
]

State:

[
L_{\text{state}}
================

CE(\hat S_i,S_i)
]

Mode:

[
L_{\text{mode}}
===============

CE(\hat M_i,M_i)
]

Timing:

Smooth L1,

[
\beta=0.1
]

유지.

Mask:

* HOLD → timing 없음
* CHANGE → slot1
* RETURN → slot1+slot2

---

# State / Mode class weighting은 audit 후 결정

여기서는 아까 실패했던 weighting 경험 때문에 조심하면 돼.

하지만 이번에는 **target이 static**이라 이전의 feedback-loop 위험은 없어.

Mode distribution은 아마 HOLD가 매우 많을 가능성이 크기 때문에 transition sensitivity를 위해 weighting이 필요할 가능성이 높아.

내 추천은:

* Train static State distribution 계산
* Train static Mode distribution 계산
* inverse-sqrt fixed weight를 계산
* State imbalance가 작으면 State CE는 unweighted
* Mode는 imbalance가 크면 fixed inverse-sqrt weighted CE

로 가는 것.

단 **validation을 보고 weight를 정하면 안 돼.**

그리고 오늘은 weight sweep까지 하지 말자.

한 가지 deterministic rule로 결정해서 한 번만 학습.

---

# 실험 2 Inference — constrained Viterbi

Network가 먼저 독립적으로:

[
P_S(S_i)
]

와

[
P_M(M_i)
]

를 전 onset/interval에 대해 출력.

그 뒤 legal joint path를 찾는다.

Legal rule:

[
S_i\neq S_{i+1}
\Rightarrow M_i=CHANGE
]

[
S_i=S_{i+1}
\Rightarrow M_i\in{HOLD,RETURN}
]

첫 v1에서는:

[
\boxed{\lambda_S=\lambda_M=1}
]

로 고정하는 걸 추천해.

즉 transition을 일부러 더 세게 주는 tuning은 아직 하지 말자.

먼저 두 head 자체의 성능을 보고 나중에 조정.

Joint MAP score는 예를 들어:

[
J=
\sum_i\log P_S(S_i)
+
\sum_i\log P_M(M_i)
]

이고 legal sequence만 허용.

같은 State endpoint에서는:

[
\max(
\log P(HOLD),
\log P(RETURN)
)
]

를 사용하고, 어느 것이 선택됐는지도 backtracking하면서 저장.

다른 State endpoint에서는 무조건:

[
\log P(CHANGE)
]

사용.

이게 정확한 2-state constrained Viterbi가 돼.

---

# Viterbi 전후를 둘 다 평가하자

이건 굉장히 유용할 거야.

### Raw heads

State argmax accuracy:

[
Acc_{state}^{raw}
]

Mode argmax accuracy/F1:

[
F1_{mode}^{raw}
]

State/Mode conflict rate.

### Joint decoded

Viterbi 이후:

* State accuracy
* Mode accuracy/Macro F1
* Transition P/R/F1
* timing
* conflict = 구조적으로 0

그러면 Viterbi가 실제로 두 branch의 오류를 상호 보완해주는지 바로 알 수 있어.

---

# 실험 2 Tiny overfit

새 구조니까 이건 꼭 한 번 해야 해.

하지만 과하게 하지 말고 한 개 multi-window train performance 정도.

목표:

* State acc → 거의 1
* Mode acc → 거의 1
* Timing MAE → 낮게
* Viterbi decoded state/mode → 거의 1
* NaN 없음

여기서 중요한 건 **state recurrence 안정성 같은 걸 볼 필요가 없다는 것**이야.

애초에 recurrent state가 없으니까.

Tiny overfit이 명확하게 성공하면 바로 full train.

---

# 실험 2 Full Training

같은 canonical train:

* MAESTRO-clean 1,170
* ASAP train 892

Validation:

* ASAP val 71

Test:

* 봉인

최대:

[
\boxed{10\ epochs}
]

매 epoch validation.

Primary metric은 실험 1과 똑같이:

[
\boxed{\text{direction-aware Transition F1}}
]

그래야 최종적으로 직접 비교할 수 있어.

---

# 실험 2에서 추가로 반드시 볼 지표

새 tokenizer니까 다음도 중요해.

### State

* State Accuracy
* State Macro F1
* OFF/ON F1

### Mode

* Mode Accuracy
* Mode Macro F1
* HOLD F1
* CHANGE F1
* RETURN F1

특히:

[
\boxed{RETURN\ F1}
]

이 중요해.

Repedaling을 얼마나 민감하게 잡는지와 연결되니까.

### Consistency

Viterbi 전 State/Mode conflict rate.

Viterbi 후:

[
0
]

이어야 함.

### Transition

* Precision
* Recall
* F1
* count ratio

### Timing

* CHANGE timing MAE
* RETURN (\tau_1) MAE
* RETURN (\tau_2) MAE

---

# 오늘 두 결과를 최종적으로 이렇게 비교하면 돼

가장 중요한 표는 아마 이 형태가 될 거야.

| Model             | Transition P | Transition R | Transition F1 | State Acc | State F1 | CHANGE F1 | RETURN F1 |
| ----------------- | -----------: | -----------: | ------------: | --------: | -------: | --------: | --------: |
| Count-2Slot       |              |              |               |           |        — |         — |         — |
| State-Anchored v1 |              |              |               |           |          |           |           |

그리고 Count 모델에서는 추가로:

* correction fraction
* mismatch run
* state agreement

State-Anchored에서는:

* raw conflict rate
* Viterbi correction effect

를 보면 돼.

---

# 이 두 실험에서 우리가 알고 싶은 과학적 질문

사실 오늘 실험은 단순한 모델 두 개 비교가 아니야.

### 실험 1이 답하는 질문

> **Transition count를 직접 모델링하면서 hard free-running state를 유지해도 충분히 좋은 pedal transition prediction을 얻을 수 있는가?**

### 실험 2가 답하는 질문

> **Absolute State와 local Transition을 동시에 모델링하고 구조적으로 reconcile하면 state drift를 줄이면서 transition sensitivity까지 유지할 수 있는가?**

이 두 질문이 아주 깔끔하게 대비돼.

---

# 실행 순서도 중요해

내가 오늘 실제로 작업한다면 이렇게 할 것 같아.

**1단계 — 실험 1 준비 후 바로 GPU full training 시작**

Count C-main-only를 구현/확인하고 tmux에서 10 epoch 시작.

Codex는 학습 완료를 기다리지 않고 종료.

**2단계 — 실험 1이 GPU에서 도는 동안 실험 2 tokenizer audit/implementation**

이 작업의 상당 부분은 CPU로 할 수 있어.

* raw target 생성
* distribution audit
* State/Mode consistency
* Viterbi unit tests
* dataset/model 구현
* compile/unit test

즉 GPU가 실험 1에 묶여 있어도 상당 부분 준비 가능.

**3단계 — 실험 1 종료 후 실험 2 tiny overfit**

GPU가 비면 실제 pretrained encoder로 tiny overfit.

PASS하면 바로 full training.

**4단계 — 실험 2 full 10 epoch**

tmux background.

---

## 가능하면 자동 queue까지는 하지 말자

실험 1이 끝난 뒤 아무 검토 없이 실험 2 full training이 자동으로 시작되는 shell queue까지 만드는 건 나는 이번에는 추천하지 않아.

새 tokenizer는 첫 구현이니까:

> Audit → unit test → tiny overfit 결과

까지 한번 보고 full train으로 넘기는 게 안전해.

하지만 실험 2의 **audit과 코드 구현은 실험 1 학습 중에 미리 끝내놓으면** GPU idle time도 거의 없을 거야.

---

# 오늘 끝났을 때 이상적인 상태

오늘 하루가 끝났을 때 다음 두 산출물이 있으면 아주 좋아.

### Run A

`Count-2Slot Base-Weighted MAIN-only`

* full train complete
* best checkpoint
* epoch-wise validation Transition F1
* state/correction diagnostics

### Run B

`State-Anchored Transition v1`

* tokenizer audit
* tiny overfit PASS
* full train complete 또는 진행 중
* best checkpoint
* validation Transition F1
* State/Mode/RETURN diagnostics

그리고 그다음에는 두 모델 중 더 좋은 쪽을 골라 **ASAP test는 마지막 최종 후보에 대해서만 한 번 여는 방향**으로 갈 수 있어.

특히 연구 스토리 측면에서도 이번 계획이 좋아. **Count-only 모델이 왜 불안정했는지 실험적으로 발견했고 → absolute state를 representation에 직접 넣어서 drift 문제를 구조적으로 해결하려 했다**는 흐름이 아주 자연스럽게 이어지거든.
