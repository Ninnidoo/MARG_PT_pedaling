# 화성적 혼탁도(Harmonic Muddiness) Metric — 최종 정의

## 1. 목적

본 metric의 목적은 **연주 전체의 페달링 품질을 평가하는 것이 아니라, sustain pedal이 추가로 만들어낸 화성적 혼탁도(harmonic muddiness)만을 정량화하는 것**이다.

따라서 이 metric은 다음 질문에 답한다.

> **페달로 인해 원래는 사라졌어야 할 음들이 남으면서, 현재 연주되는 음들과 얼마나 불협한 관계를 만들고 있으며, 그러한 불협 관계가 얼마나 많이 누적되고 있는가?**

최종 metric은 두 가지 성분을 사용한다.

$$
\boxed{
M_{\mathrm{harm}}
=
H_{\mathrm{mean}}
+
0.05A_{\mathrm{acc}}
}
$$

여기서 \(H_{\mathrm{mean}}\)은 **불협 관계의 평균적인 심각도(severity)**를, \(A_{\mathrm{acc}}\)는 **불협 관계가 얼마나 많이 누적되어 있는지(accumulation)**를 나타낸다.

최종 점수는 **낮을수록 화성적으로 깨끗한 페달링**, 높을수록 pedal-induced harmonic muddiness가 큰 것으로 해석한다.

---

# 2. 시간축과 note state

Performance MIDI에 존재하는 서로 다른 note-on 시각을 오름차순으로 정렬한다.

$$
t_1<t_2<\cdots<t_N
$$

여기서 \(N\)은 연주 전체의 **distinct onset 개수**이다.

각 onset \(t_n\)에서 note instance를 다음 두 종류로 나눈다.

### Active note set \(A_n\)

$$
A_n
=
\{\text{onset }t_n\text{에서 아직 실제 건반이 눌려 있는 note instances}\}
$$

즉 note-on은 발생했지만 아직 해당 note-off가 발생하지 않은 음이다.

### Pedal-sustained note set \(P_n\)

$$
P_n
=
\{\text{건반에서는 이미 release되었지만 sustain pedal 때문에 남아 있는 note instances}\}
$$

즉 key-off 이후에도 CC64 sustain 상태 때문에 유지되는 음이다.

여기서 중요한 것은 **pitch가 아니라 note instance 단위로 계산한다는 점**이다. 같은 pitch의 음이 여러 번 연주되어 동시에 sustain되고 있다면 각각 별도의 note instance로 취급한다.

---

# 3. Sustain pedal의 정의

현재 metric에서는 pedal depth를 사용하지 않는다. CC64를 단순한 binary state로 변환한다.

$$
\boxed{
\mathrm{CC64}\ge64
\quad\Rightarrow\quad
\mathrm{Pedal\ ON}
}
$$

$$
\boxed{
\mathrm{CC64}<64
\quad\Rightarrow\quad
\mathrm{Pedal\ OFF}
}
$$

따라서 CC64=64, 80, 100, 127은 모두 동일한 Pedal ON 상태이다.

Half-pedal, pedal depth, continuous CC64 값은 현재 화성적 혼탁도 metric에 영향을 주지 않는다. 기존 Pure Hall audit에서도 동일한 binary semantics를 사용하였다. 

---

# 4. 평가하는 note pair의 정의

이 metric의 중요한 원칙은 **페달 때문에 새롭게 생긴 harmonic relation만 평가한다**는 것이다.

현재 실제로 눌려 있는 음끼리의 관계 \(A-A\)는 pedal을 밟지 않아도 존재하는 화성이므로 평가하지 않는다.

각 onset \(t_n\)에서 평가할 pair 집합은 다음과 같이 정의한다.

$$
\boxed{
Q_n
=
\{(i,j)\mid i\in P_n,\;j\in A_n\}
\;\cup\;
\{(i,j)\mid i,j\in P_n,\;i<j\}
}
$$

첫 번째 집합은 **PA pair**이다.

$$
P_n\times A_n
$$

즉 pedal로 남은 과거 음과 현재 건반으로 연주 중인 음 사이의 관계를 평가한다.

두 번째 집합은 **PP pair**이다.

$$
\binom{P_n}{2}
$$

즉 pedal로 남아 있는 residual note끼리의 관계를 평가한다. \(i<j\) 조건은 동일한 unordered pair를 한 번만 세기 위한 것이다.

반면

$$
A_n\times A_n
$$

관계는 평가하지 않는다.

따라서

$$
\boxed{
Q_n=PA_n\cup PP_n,\qquad AA_n\notin Q_n
}
$$

이다.

이 정의 덕분에 metric은 악보나 연주 자체가 원래 가지고 있던 화성적 dissonance가 아니라 **pedal이 추가한 harmonic interaction**에 집중한다.

---

# 5. Hall interval weight \(w(I)\)

각 pair \((i,j)\)에 대해 두 note의 MIDI pitch를 \(p_i,p_j\)라고 한다.

두 음 사이의 interval class는 octave equivalence를 적용하여

$$
\boxed{
I_{ij}
=
|p_i-p_j|\bmod12
}
$$

로 계산한다.

\(I=0\)은 unison/octave-equivalent class로 처리하며 현재 Hall table에서는 P8 weight를 사용한다.

본 metric은 Hall, Tamir & Rohrmeier (2025)의 **Simple Type / type-method interval weight**를 사용한다. 현재 프로젝트에서 고정한 weight는 다음과 같다. 

| \(I\) | 음정                     | \(w(I)\) | negative-only contribution |
| ----: | ---------------------- | -------: | -------------------------: |
|     0 | P8 / octave-equivalent |   +1.009 |                          0 |
|     1 | minor 2nd              |   -1.902 |                      1.902 |
|     2 | major 2nd              |   -0.299 |                      0.299 |
|     3 | minor 3rd              |   -0.029 |                      0.029 |
|     4 | major 3rd              |   +0.257 |                          0 |
|     5 | perfect 4th            |   +0.616 |                          0 |
|     6 | tritone                |   -0.582 |                      0.582 |
|     7 | perfect 5th            |   +0.676 |                          0 |
|     8 | minor 6th              |   -0.339 |                      0.339 |
|     9 | major 6th              |   -0.142 |                      0.142 |
|    10 | minor 7th              |   -0.712 |                      0.712 |
|    11 | major 7th              |   -0.899 |                      0.899 |

Hall weight가 positive인 consonant relation에는 reward를 주지 않는다.

각 pair의 penalty는 직접

$$
\boxed{
\max\left(-w(I_{ij}),\,0\right)
}
$$

으로 계산한다.

따라서 예를 들어 minor 2nd는

$$
w(1)=-1.902
$$

이므로

$$
\max(1.902,0)=1.902
$$

의 penalty를 갖는다.

반면 perfect 5th는

$$
w(7)=+0.676
$$

이므로

$$
\max(-0.676,0)=0
$$

이다.

즉 현재 metric에서는 consonance가 dissonance를 상쇄하지 않는다.

$$
\boxed{\text{positive Hall relation}=0,\qquad
\text{negative Hall relation만 penalty}}
$$

이 negative-only 설계는 기존 positive/negative decomposition audit에서 관찰된 consonant reward와 dissonant penalty의 cancellation 문제를 피하기 위한 선택이다. 

---

# 6. Severity component: \(H_{\mathrm{mean}}\)

첫 번째 성분은 **pedal-induced harmonic interaction이 실제로 존재하는 순간에, 관계 하나하나가 평균적으로 얼마나 불협한가**를 측정한다.

따라서 \(H_{\mathrm{mean}}\)은 반드시

$$
Q_n\neq\varnothing
$$

인 **valid onset만을 대상으로 평균**한다.

최종 식은 다음과 같다.

$$
\boxed{
H_{\mathrm{mean}}
=
\frac{
\displaystyle
\sum_{\substack{n=1\\Q_n\neq\varnothing}}^{N}
\left[
\frac{1}{|Q_n|}
\sum_{(i,j)\in Q_n}
\max\left(-w(I_{ij}),0\right)
\right]
}{
\displaystyle
\left|
\left\{
n\in\{1,\ldots,N\}
:
Q_n\neq\varnothing
\right\}
\right|
}
}
$$

즉 계산 순서는 각 valid onset에서

$$
\frac{1}{|Q_n|}
\sum_{(i,j)\in Q_n}
\max(-w(I_{ij}),0)
$$

를 먼저 계산하고, 그 값을 valid onset 전체에 대해 다시 평균하는 것이다.

### 왜 valid onset만 사용하는가?

만약 \(Q_n=\varnothing\)인 onset에 0을 넣어 모든 onset을 평균하면, 이 값은 더 이상 순수한 severity가 아니다.

예를 들어 pedal interaction이 드문 모델은 실제로 conflict가 발생했을 때 상당히 dissonant하더라도 수많은 zero onset 때문에 점수가 낮아질 수 있다.

따라서

$$
H_{\mathrm{mean}}
$$

에는 interaction coverage를 섞지 않고,

> **“pedal-induced harmonic relation이 실제로 존재한다는 조건에서 평균적으로 얼마나 불협한가?”**

만 측정하도록 고정한다.

valid onset이 하나도 없는 경우에는

$$
H_{\mathrm{mean}}=0
$$

으로 정의한다.

---

# 7. Accumulation component: \(A_{\mathrm{acc}}\)

\(H_{\mathrm{mean}}\)의 한계는 pair-average이기 때문에 **불협 관계가 얼마나 많이 누적되어 있는지**를 표현하지 못한다는 것이다.

예를 들어 두 onset에서 평균 Hall-negative severity가 동일하게 0.3이더라도,

* negative relation이 몇 개밖에 없는 경우와
* 수백 또는 수천 개의 negative relation이 동시에 존재하는 경우

를 \(H_{\mathrm{mean}}\)은 비슷하게 평가할 수 있다.

이를 보완하기 위해 accumulation term을 추가한다.

최종 식은 다음과 같다.

$$
\boxed{
A_{\mathrm{acc}}
=
\frac{1}{N}
\sum_{n=1}^{N}
\ln
\left[
1+
\sum_{(i,j)\in Q_n}
\max\left(-w(I_{ij}),0\right)
\right]
}
$$

중요하게도 \(A_{\mathrm{acc}}\)에는 **모든 distinct onset**이 들어간다.

즉

$$
Q_n=\varnothing
$$

인 onset에서도

$$
\sum_{(i,j)\in Q_n}
\max(-w(I_{ij}),0)=0
$$

이므로 해당 onset의 contribution은

$$
\ln(1+0)=0
$$

이다.

### 왜 모든 onset을 사용하는가?

Accumulation term의 목적은 단순히

> “conflict가 발생했을 때 얼마나 많았는가?”

뿐만 아니라,

> **“연주 전체에서 pedal-induced harmonic conflict가 얼마나 지속적이고 많이 존재했는가?”**

까지 반영하는 것이다.

따라서 conflict가 없는 onset 역시 정보가 있으며, 0으로 포함한다.

---

# 8. 왜 logarithm을 사용하는가?

Accumulation term에서는 pair 수로 나누지 않는다.

따라서 pedal-sustained note가 많이 누적되면 PP pair 수가

$$
\binom{|P_n|}{2}
$$

형태로 증가할 수 있고, negative Hall mass 역시 매우 커질 수 있다.

특히 ALWAYS_ON에서는 연주가 진행될수록 residual note가 계속 누적되므로 raw conflict mass가 매우 크게 증가한다.

이를 그대로 사용하면 몇몇 큰 값이 metric scale을 지나치게 지배할 수 있으므로

$$
\boxed{\ln(1+x)}
$$

를 적용한다.

이 방식은 다음 성질을 갖는다.

$$
x=0\Rightarrow\ln(1+x)=0
$$

이므로 conflict가 전혀 없는 onset은 정확히 0이고,

작거나 중간 정도의 conflict 차이는 유지하면서, 매우 큰 accumulation에서는 증가율이 완만해진다.

또한 accumulation은

$$
\ln\left(1+\sum_{\text{piece}} \cdots\right)
$$

처럼 곡 전체의 mass를 먼저 합친 뒤 log를 취하지 않는다.

반드시 onset별로 log를 취한 다음 평균한다.

$$
\boxed{
\frac1N
\sum_n
\ln
\left[
1+
\sum_{(i,j)\in Q_n}
\max(-w(I_{ij}),0)
\right]
}
$$

이렇게 함으로써 곡 길이에 대한 직접적인 의존성을 줄이고, onset-level accumulation의 평균적인 크기를 평가한다.

---

# 9. 최종 화성적 혼탁도

최종적으로 두 성분을 다음과 같이 합산한다.

$$
\boxed{
M_{\mathrm{harm}}
=
H_{\mathrm{mean}}
+
0.05A_{\mathrm{acc}}
}
$$

즉 식을 완전히 풀어 쓰면,

$$
\boxed{
\begin{aligned}
M_{\mathrm{harm}}
={}&
\frac{
\displaystyle
\sum_{\substack{n=1\\Q_n\neq\varnothing}}^{N}
\left[
\frac{1}{|Q_n|}
\sum_{(i,j)\in Q_n}
\max\left(-w(I_{ij}),0\right)
\right]
}{
\displaystyle
\left|
\left\{
n\in\{1,\ldots,N\}
:
Q_n\neq\varnothing
\right\}
\right|
}
\\[6pt]
&+
0.05
\left[
\frac{1}{N}
\sum_{n=1}^{N}
\ln
\left(
1+
\sum_{(i,j)\in Q_n}
\max\left(-w(I_{ij}),0\right)
\right)
\right].
\end{aligned}
}
$$

단, valid onset이 하나도 존재하지 않는 경우 첫 번째 항은 0으로 정의한다.

이것이 최종 **화성적 혼탁도 metric**이다.

---

# 10. 두 성분의 역할

| Component             | 질문                                                         | Aggregation                                |
| --------------------- | ---------------------------------------------------------- | ------------------------------------------ |
| \(H_{\mathrm{mean}}\) | pedal-induced relation 하나가 평균적으로 얼마나 불협한가?                 | \(Q_n\neq\varnothing\)인 valid onset만 사용    |
| \(A_{\mathrm{acc}}\)  | pedal-induced negative conflict가 연주 전체에서 얼마나 많이 누적되어 있는가?  | 모든 distinct onset 사용                       |
| \(M_{\mathrm{harm}}\) | severity와 accumulation을 종합했을 때 pedal이 만든 화성적 혼탁도가 어느 정도인가? | \(H_{\mathrm{mean}}+0.05A_{\mathrm{acc}}\) |

이 두 항을 곱하지 않고 **additive하게 합치는 이유**도 중요하다.

불협 관계의 평균 severity와 누적량은 서로 다른 현상이다. 하나가 작다고 다른 하나의 정보를 완전히 지워버리지 않도록 각각 독립적인 contribution을 갖게 한다.

---

# 11. 계수 \(0.05\)의 의미

$$
\boxed{0.05}
$$

는 학습이나 parameter optimization으로 찾은 값이 아니라 **휴리스틱하게 고정한 scale-balancing coefficient**이다.

13-piece validation에서 관찰된 component scale은 대략 다음과 같았다.

| System          | \(H_{\mathrm{mean}}\) — valid onset | \(A_{\mathrm{acc}}\) — all onset |
| --------------- | ----------------------------------: | -------------------------------: |
| HUMAN           |                            0.218644 |                         1.044679 |
| ORIGINAL PT     |                            0.220236 |                         1.238177 |
| Stage2 models   |                   0.225839–0.240969 |                0.683421–0.841607 |
| CUSTOM EVENT V0 |                            0.253005 |                         1.227567 |
| ALWAYS ON       |                            0.320297 |                        11.109987 |
| NO PEDAL        |                                   0 |                                0 |



정상적인 Human/PT/Stage2 영역에서는

$$
0.05A_{\mathrm{acc}}
$$

가 대략 \(0.03\sim0.06\) 수준이므로 \(H_{\mathrm{mean}}\)이 여전히 주요 성분으로 남는다.

반면 ALWAYS_ON에서는

$$
0.05\times11.109987
\approx0.5555
$$

가 되어 extreme accumulation을 강하게 반영한다.

따라서 coefficient의 의도는

> **일반적인 연주에서는 Hall severity를 중심으로 평가하되, 비정상적으로 많은 conflict가 누적될 경우 accumulation term이 충분히 강하게 작동하도록 하는 것**

이다.

이 coefficient는 **validation ordering을 최적화한 결과가 아니며**, 최종 연구에서는 heuristic design choice임을 명시한다.

---

# 12. 간략한 평가 결과

## Pure Hall-negative severity

13곡 validation에서 valid onset만 사용한 \(H_{\mathrm{mean}}\)의 piece-balanced 평균은 다음과 같았다.

$$
\boxed{
\text{HUMAN }0.2186
<
\text{PT }0.2202
<
\text{Stage2 }0.2258\sim0.2410
<
\text{ALWAYS ON }0.3203
}
$$

즉 system-level 평균에서는 Hall-negative severity 자체가 상당히 자연스러운 큰 방향을 보였다. 다만 개별 piece에서 HUMAN<PT가 항상 성립하는 것은 아니며, 기존 Pure Hall audit에서 HUMAN<PT는 7/13이었다. 따라서 이 metric을 Human-likeness metric으로 해석해서는 안 된다. 

## Accumulation

\(A_{\mathrm{acc}}\)에서는 특히 ALWAYS_ON이 매우 강하게 분리되었다.

$$
A_{\mathrm{Human}}=1.0447
$$

$$
A_{\mathrm{PT}}=1.2382
$$

$$
A_{\mathrm{Stage2}}\approx0.68\sim0.84
$$

인데 비해,

$$
\boxed{
A_{\mathrm{Always}}=11.1100
}
$$

이었다.

13곡 모두에서 ALWAYS_ON의 후반부 accumulation이 초반보다 높았고, normalized position 기준 piece-balanced 평균은

$$
6.403
\rightarrow
13.047
$$

로 증가했다. 이는 pedal을 계속 밟으면서 residual note가 누적될 때 \(H_{\mathrm{mean}}\)이 놓치는 conflict amount를 \(A_{\mathrm{acc}}\)가 포착하고 있음을 보여준다. 

## 최종 고정식의 system-level 예시

위 13-piece component 평균에 최종 coefficient \(0.05\)를 적용하면 대략 다음과 같다.

| System                       | \(M_{\mathrm{harm}}\) ↓ |
| ---------------------------- | ----------------------: |
| NO_PEDAL                     |               **0.000** |
| STANDARD CE POSTERIOR MEDIAN |                   0.267 |
| HYBRID REGRESSION ONLY       |                   0.268 |
| HUMAN                        |               **0.271** |
| STANDARD CE ARGMAX           |                   0.274 |
| WEIGHTED CE ARGMAX           |                   0.275 |
| ORIGINAL PT                  |               **0.282** |
| CUSTOM EVENT V0              |                   0.314 |
| ALWAYS ON                    |               **0.876** |

이 표는 **최종 coefficient를 이용해 기존 component summary를 합성한 system-level 참고값**이며, \(0.05\)를 최적화한 별도의 weight-search 결과는 아니다. 기반 component 값은 13-piece audit에서 검증되었다. 

가장 중요한 결과는 정상적인 Human/PT/Stage2 계열이 비교적 비슷한 범위에 위치하는 반면,

$$
\boxed{\text{ALWAYS ON}}
$$

은 accumulation 때문에 크게 분리된다는 점이다.

---

# 13. NO_PEDAL의 해석

NO_PEDAL에서는 pedal-sustained note가 존재하지 않으므로

$$
P_n=\varnothing
$$

이고 따라서 모든 onset에서

$$
Q_n=\varnothing.
$$

결과적으로

$$
\boxed{
M_{\mathrm{harm}}(\mathrm{NO\ PEDAL})=0
}
$$

이다.

이것은 의도된 동작이다.

0점은

> **“pedal이 추가한 harmonic muddiness가 없다.”**

는 뜻이지,

> **“전체적으로 가장 좋은 pedaling이다.”**

라는 뜻이 아니다.

NO_PEDAL의 문제인 sustain 부족, bass/melody connection 부족 등은 별도의 **Connection / Sustain metric**에서 평가해야 한다.

따라서 본 metric은 전체 pedal quality score가 아니라 명확하게

$$
\boxed{\text{pedal-induced harmonic muddiness component}}
$$

로만 해석한다.

---

# 14. 최종적으로 사용하지 않는 요소

현재 확정된 버전에서는 의도적으로 다음 요소를 사용하지 않는다.

| 요소                           | 사용 여부       |
| ---------------------------- | ----------- |
| Hall interval weight         | 사용          |
| Negative Hall magnitude      | 사용          |
| PA pair                      | 사용          |
| PP pair                      | 사용          |
| A-A pair                     | **사용하지 않음** |
| Positive Hall reward         | **사용하지 않음** |
| Note decay                   | **사용하지 않음** |
| Velocity / dynamics          | **사용하지 않음** |
| Low-register weighting       | **사용하지 않음** |
| PA/PP differential weighting | **사용하지 않음** |
| Pedal depth                  | **사용하지 않음** |
| CC64 binary ON/OFF           | 사용          |
| Note-count 자체의 직접적인 penalty  | **사용하지 않음** |

즉 현재 metric은 가능한 한 단순하게

$$
\boxed{
\text{Hall-negative severity}
+
\text{Hall-negative accumulation}
}
$$

두 가지에만 집중한다.

---

# 15. 최종 정의 요약

최종 화성적 혼탁도 metric은 다음과 같이 고정한다.

$$
\boxed{
Q_n
=
\{(i,j)\mid i\in P_n,j\in A_n\}
\cup
\{(i,j)\mid i,j\in P_n,i<j\}
}
$$

$$
\boxed{
H_{\mathrm{mean}}
=
\frac{
\displaystyle
\sum_{\substack{n=1\\Q_n\neq\varnothing}}^{N}
\left[
\frac{1}{|Q_n|}
\sum_{(i,j)\in Q_n}
\max\left(-w(I_{ij}),0\right)
\right]
}{
\displaystyle
\left|
\left\{
n\in\{1,\ldots,N\}:Q_n\neq\varnothing
\right\}
\right|
}
}
$$

$$
\boxed{
A_{\mathrm{acc}}
=
\frac{1}{N}
\sum_{n=1}^{N}
\ln
\left[
1+
\sum_{(i,j)\in Q_n}
\max\left(-w(I_{ij}),0\right)
\right]
}
$$

그리고 최종적으로

$$
\boxed{
\displaystyle
M_{\mathrm{harm}}
=
H_{\mathrm{mean}}
+
0.05A_{\mathrm{acc}}
}
$$

를 사용한다.

**낮을수록 pedal-induced harmonic muddiness가 적고, 높을수록 화성적으로 혼탁한 페달링으로 평가한다.**