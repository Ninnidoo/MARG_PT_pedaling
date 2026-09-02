# 베이스 연결도(Bass Connectivity) Metric — 최종 정의

## 1. 목적

**베이스 연결도(Bass Connectivity)** metric의 목적은 sustain pedal의 가장 핵심적인 기능 중 하나인 **저음부의 structural bass를 다음 bass까지 자연스럽게 연결하고, bass가 바뀌는 시점에는 적절하게 pedal을 교체하는 능력**을 정량화하는 것이다.

이 metric이 답하려는 질문은 다음과 같다.

> **연주에서 구조적으로 중요한 bass를 식별했을 때, 현재 bass의 건반에서 손이 떨어진 이후 다음 structural bass가 연주될 때까지 sustain pedal이 bass를 충분히 연결하고 있으며, bass가 바뀐 뒤에는 이전 bass를 과도하게 중첩시키지 않고 적절히 pedal을 교체하고 있는가?**

따라서 이 metric은 단순히 “pedal을 많이 밟았는가” 또는 “bass를 오래 sustain했는가”를 평가하지 않는다.

좋은 bass pedaling은 크게 두 가지 조건을 만족해야 한다.

$$
\boxed{
\text{현재 Bass를 다음 Bass까지 연결}
}
$$

그리고 bass pitch가 바뀌는 경우에는

$$
\boxed{
\text{다음 Bass 부근에서 이전 Bass를 적절히 release}
}
$$

해야 한다.

최종 Bass Connectivity score는

$$
\boxed{
0\le C_{\mathrm{Bass}}\le1
}
$$

의 범위를 가지며,

$$
\boxed{\text{높을수록 좋은 Bass Connectivity}}
$$

로 해석한다.

---

# 2. Metric의 전체 구조

Bass Connectivity는 크게 두 단계로 구성된다.

첫 번째 단계는 **Structural Bass Detection**이다.

Performance MIDI의 모든 저음을 bass로 취급하지 않고,

$$
\boxed{
\text{낮은 음인가?}
+
\text{주변 음악보다 큰 structural timescale에서 등장하는가?}
}
$$

를 이용해 실제로 pedal을 통해 연결할 가능성이 높은 structural bass를 검출한다.

두 번째 단계는 **Bass-to-Bass Connectivity Evaluation**이다.

검출된 structural bass들을 시간순으로 연결하여,

$$
b_r\rightarrow b_{r+1}
$$

각 transition에서 현재 bass의 physical key-off와 다음 bass onset 사이의 pedal release timing을 평가한다.

현재 최종 구현은 이 두 단계를 모두 MIDI 정보만으로 계산한다. Audio, acoustic decay, velocity weighting은 사용하지 않는다. Validation 구현에서도 이 정의가 그대로 고정되었다. 

---

# Part A. Structural Bass Detection

## 3. 20 ms Musical Onset Grouping

실제 인간 피아노 연주에서는 하나의 화음도 모든 건반이 정확히 같은 MIDI tick에 눌리지 않는다. 예를 들어 왼손 octave가 10 ms 정도의 차이를 두고 attack될 수 있다.

따라서 raw MIDI note-on을 모두 독립적인 onset으로 취급하지 않고 **20 ms 이내의 note attack을 하나의 musical onset group으로 묶는다.**

Performance MIDI의 non-drum, positive-velocity note-on들을 시간순으로 정렬한다.

아직 group에 할당되지 않은 가장 이른 note-on 시각을 \(t_n\)이라고 하면,

$$
\boxed{
0\le o_i-t_n\le20\text{ ms}
}
$$

를 만족하는 아직 할당되지 않은 note-on들을 하나의 onset group

$$
O_n
$$

으로 묶는다.

여기서 중요한 것은 **earliest-anchor grouping**을 사용한다는 점이다.

예를 들어 attack 시각이

$$
0,\;15,\;30\text{ ms}
$$

라면

$$
\boxed{
\{0,15\},\qquad\{30\}
}
$$

의 두 onset group으로 나눈다.

15 ms가 30 ms와 20 ms 이내라고 해서 세 attack을 모두 하나로 연결하는 single-linkage 방식은 사용하지 않는다.

각 onset group의 대표 시각은 group에서 가장 이른 attack 시각 \(t_n\)이다.

Validation implementation의 corresponding synthetic test에서도

$$
0/15/30\text{ ms}\rightarrow\{0,15\},\{30\}
$$

이 정확히 확인되었다. 

---

# 4. Onset Group의 대표 저음 \(p_n\)

각 onset group \(O_n\)에서 **그 순간 새롭게 attack된 음들 가운데 가장 낮은 MIDI pitch**를

$$
\boxed{
p_n=\min_{i\in O_n}m_i
}
$$

로 정의한다.

중요하게도 Structural Bass detection에는

* 이전부터 key-held 상태인 note,
* pedal로 sustain되고 있는 과거 note

를 사용하지 않는다.

오직 **현재 onset group에서 새롭게 attack된 note**만을 사용한다.

따라서 \(p_n\)은 현재 musical onset이 새롭게 제시하는 가장 낮은 pitch를 나타낸다.

---

# 5. Lowness Score \(L_n\)

낮은 register에 위치할수록 structural bass일 가능성이 높다는 직관을 반영하기 위해 Lowness score를 정의한다.

현재 최종 고정식은

$$
\boxed{
L_n=
\begin{cases}
1,
&
p_n\le52
\\[6pt]
\dfrac{57-p_n}{5},
&
52<p_n<57
\\[10pt]
0,
&
p_n\ge57
\end{cases}
}
$$

이다.

즉 MIDI 52 이하에서는 full lowness evidence를 주고, MIDI 52–57에서는 점차 감소시키며, 57 이상에서는 lowness evidence를 주지 않는다.

예를 들면

$$
L(52)=1
$$

이고,

$$
L(53)=0.8,\qquad
L(54)=0.6,\qquad
L(55)=0.4,\qquad
L(56)=0.2,
$$

$$
L(57)=0
$$

이다.

중요한 것은 \(L_n\) 하나만으로 bass를 결정하지 않는다는 점이다.

낮은 음이라도 빠르게 반복되는 accompaniment라면 structural bass로 취급하지 않아야 하기 때문이다.

---

# 6. Structural Timescale \(R_n^B\)

Structural Bass의 두 번째 핵심 조건은 **그 저음이 주변 musical texture보다 더 큰 시간 단위에서 등장하는가**이다.

현재 onset group \(O_n\) 이후 처음 등장하는 low-register attack을

$$
\boxed{
t_n^{\mathrm{next-low}}
=
\min
\{t_k>t_n:p_k\le57\}
}
$$

로 정의한다.

그러면 현재 저음 attack에서 다음 low-register attack까지 걸린 실제 시간은

$$
\boxed{
\Delta_n^B
=
t_n^{\mathrm{next-low}}-t_n
}
$$

이다.

하지만 \(\Delta_n^B\)를 초(second) 단위 그대로 사용하면 tempo가 다른 곡들을 비교하기 어렵다.

따라서 주변 musical onset의 local timescale을 구한다.

20 ms grouping 이후 인접 onset group 사이의 positive IOI를

$$
t_{j+1}-t_j
$$

로 계산하고, 현재 onset 주변 최대 \(\pm8\) grouped-onset neighborhood에서 얻은 positive IOI들의 median을

$$
\boxed{
\widetilde{\mathrm{IOI}}_n
}
$$

으로 사용한다.

Structural-timescale ratio는

$$
\boxed{
R_n^B
=
\frac{
t_n^{\mathrm{next-low}}-t_n
}{
\widetilde{\mathrm{IOI}}_n
}
}
$$

이다.

\(R_n^B\)는

$$
\frac{\mathrm{second}}{\mathrm{second}}
$$

이므로 **단위가 없는 dimensionless ratio**이다.

즉,

> 다음 low-register attack이 주변의 일반적인 musical onset interval에 비해 몇 배 정도 떨어져 있는가?

를 나타낸다.

예를 들어 local median IOI가 100 ms이고 다음 low attack이 300 ms 후라면

$$
R_n^B=3.
$$

---

# 7. Structural Timescale Score \(S_n\)

Structural-timescale ratio를 \([0,1]\) 범위의 soft score로 변환한다.

최종 heuristic parameter는

$$
\boxed{
R_{\mathrm{low}}=1.5,
\qquad
R_{\mathrm{high}}=3.0
}
$$

으로 고정한다.

따라서

$$
\boxed{
S_n
=
\operatorname{clip}_{[0,1]}
\left(
\frac{R_n^B-1.5}{3.0-1.5}
\right)
}
$$

이다.

즉,

$$
R_n^B\le1.5
\quad\Rightarrow\quad
S_n=0,
$$

$$
R_n^B\ge3.0
\quad\Rightarrow\quad
S_n=1
$$

이고, 그 사이에서는 선형적으로 증가한다.

따라서 빠르게 반복되는 저음은 \(S_n\)이 작아지고, 주변의 작은-scale motion과 비교해 오랫동안 다음 low attack이 등장하지 않는 저음은 높은 \(S_n\)을 얻는다.

Terminal onset처럼 이후의 next-low를 정의할 수 없거나 local timescale이 충분히 계산되지 않는 경우에는 임의의 sentinel 값을 만들지 않고 Structural Bass detection에서 제외한다. 이 처리 역시 validation implementation에서 고정되어 있다. 

---

# 8. Octave Bonus

Piano texture에서 structural bass가 왼손 octave로 연주되는 경우가 많다는 음악적 직관을 약한 보조 evidence로 반영한다.

현재 onset group의 최저음이 \(p_n\)일 때 같은 20 ms group 안에 정확히

$$
p_n+12
$$

가 존재하면

$$
\boxed{
O_n^{\mathrm{oct}}=1
}
$$

그렇지 않으면

$$
\boxed{
O_n^{\mathrm{oct}}=0
}
$$

으로 둔다.

Octave는 Structural Bass의 필수조건이 아니다.

고정된 작은 bonus

$$
\boxed{
\beta_{\mathrm{oct}}=0.2
}
$$

만을 사용한다.

따라서 octave가 존재하면 기본 bass salience가 최대 20% 증가한다.

---

# 9. Structural Bass Salience \(B_n\)

Lowness, structural timescale, octave bonus를 결합하여

$$
\boxed{
B_n
=
\operatorname{clip}_{[0,1]}
\left[
L_nS_n
\left(
1+0.2O_n^{\mathrm{oct}}
\right)
\right]
}
$$

으로 정의한다.

즉 핵심 구조는

$$
\boxed{
\text{Bass Salience}
\approx
\text{Lowness}
\times
\text{Structural Timescale}
}
$$

이고 octave는 약한 보조 evidence이다.

곱셈 구조를 사용하므로 단순히 아주 낮다는 이유만으로 높은 \(B_n\)을 얻을 수 없으며, 반대로 시간적으로 structural하더라도 register가 충분히 낮지 않으면 높은 값을 얻기 어렵다.

---

# 10. 최종 Structural Bass 판정

\(L_n\)과 \(S_n\) 각각에 별도의 hard cutoff를 두지 않는다.

먼저 continuous Bass Salience \(B_n\)을 모두 계산한 뒤,

$$
\boxed{
B_n\ge0.5
}
$$

인 onset group만 **Structural Bass**로 정의한다.

즉 Structural Bass 집합은

$$
\boxed{
\mathcal B
=
\{n:B_n\ge0.5\}
}
$$

이다.

이를 시간순으로 정렬하여

$$
\boxed{
b_1,b_2,\ldots,b_M
}
$$

이라고 한다.

현재 validation 구현에서도 정확히 이 \(B_n\ge0.5\) 기준이 사용되었으며, validation 결과를 본 뒤 threshold나 coefficient를 조정하지 않았다. 

---

# Part B. Bass-to-Bass Connectivity

## 11. 평가 단위

검출된 Structural Bass 중 **연속된 두 Bass**

$$
\boxed{
b_r\rightarrow b_{r+1}
}
$$

를 하나의 transition으로 평가한다.

현재 bass \(b_r\)에 대해

$$
p_r=\text{lowest attacked MIDI pitch},
$$

$$
k_r=\text{해당 bass note의 physical key-off 시각}
$$

으로 둔다.

다음 bass \(b_{r+1}\)에 대해서는

$$
p_{r+1}=\text{lowest attacked MIDI pitch},
$$

$$
o_{r+1}=\text{다음 Structural Bass onset}
$$

을 사용한다.

여기서 \(k_r\)은 **pedal에 의해 실제 sound가 끝나는 시각이 아니라 physical note-off, 즉 손이 건반에서 떨어지는 시각**이다.

이 distinction이 metric의 핵심이다.

Bass Connectivity는

> 손가락이 더 이상 bass를 유지하지 않는 순간 이후에 pedal이 어떤 역할을 했는가?

를 평가하기 때문이다.

---

# 12. Pedal Connectivity Opportunity

현재 bass의 key-off가 다음 bass onset 이후라면

$$
k_r\ge o_{r+1}
$$

손가락 자체가 이미 두 bass 사이를 연결하고 있다.

이 경우에는 pedal이 connection을 만들어야 할 opportunity가 없다고 보고 Bass Connectivity 계산에서 제외한다.

즉 유효한 transition은

$$
\boxed{
k_r<o_{r+1}
}
$$

을 만족하는 경우뿐이다.

마지막 Structural Bass 역시 다음 bass가 없으므로 제외한다.

유효 transition 집합을

$$
\boxed{
\mathcal T_B
}
$$

라고 한다.

Validation에서는 canonical variants에서 총 1,927개의 Structural Bass가 검출되었고, 그중 1,810개의 transition이 valid opportunity였으며, 98개는 key가 다음 bass까지 유지되어 제외되었다. 

---

# 13. Bass Gap \(d_r\)

현재 bass의 손이 떨어진 시점부터 다음 bass attack까지의 물리적 gap을

$$
\boxed{
d_r=o_{r+1}-k_r
}
$$

로 정의한다.

\(d_r>0\)인 transition만 평가 대상이다.

이 \(d_r\)는 이후 Gaussian scoring function의 scale을 자동으로 결정한다.

따라서 빠른 passage와 느린 passage가 서로 다른 고정 ms threshold를 필요로 하지 않는다.

---

# 14. Pedal State

Sustain pedal은 CC64를 binary state로 변환하여 사용한다.

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

Half-pedal이나 CC64 depth 자체는 Bass Connectivity v0에서 사용하지 않는다.

---

# 15. Pedal을 전혀 연결하지 않은 경우

현재 bass의 physical key-off \(k_r\) 시점에서 이미 pedal이 OFF라면, pedal이 해당 bass의 connection에 전혀 기여하지 않은 것으로 본다.

따라서

$$
\boxed{
\mathrm{CC64}(k_r)<64
\quad\Rightarrow\quad
C_r=0
}
$$

으로 직접 정의한다.

Gaussian 값은 계산하지 않는다.

이 예외가 필요한 이유는 raw Gaussian을 그대로 사용할 경우 key-off 위치에서도 작은 positive value가 생길 수 있기 때문이다.

하지만 음악적 의미상 pedal이 key-off 이후 bass를 한 순간도 sustain하지 않았다면 connection score는 정확히 0이어야 한다.

---

# 16. Pedal Release 시각 \(u_r\)

key-off 순간 pedal이 ON이라면 그 이후 처음으로 CC64가 64 아래로 내려가는 시각을

$$
\boxed{
u_r
=
\min
\{t\ge k_r:\mathrm{CC64}(t)<64\}
}
$$

로 정의한다.

즉 \(u_r\)은 **현재 bass를 sustain하던 pedal이 최초로 release되는 시점**이다.

한 번 pedal이 OFF되어 damper가 내려오면 기존 bass가 종료된 것으로 보기 때문에, 그 뒤 pedal이 다시 ON으로 바뀌더라도 기존 bass가 다시 살아났다고 취급하지 않는다.

따라서 첫 번째 pedal OFF만 추적한다.

Validation implementation에서는 same-timestamp MIDI message를 기존 Harmonic Muddiness parser의 merged sequential ordering으로 처리했고, same-pitch note instance는 FIFO 방식으로 physical note-off와 pairing하였다. 

---

# Part C. Bass Connection Scoring Function

## 17. 핵심 아이디어

Bass Connectivity에서는 **pedal release timing**이 가장 중요하다.

현재 bass를 너무 일찍 release하면 다음 bass까지 연결되지 않는다.

반대로 bass pitch가 바뀌었는데 새 bass가 연주된 뒤에도 이전 pedal을 지나치게 오래 유지하면 두 bass가 overlap된다.

따라서 different-bass transition에서는 다음 bass onset

$$
o_{r+1}
$$

을 최적 release timing으로 둔다.

$$
\boxed{
\mu_r=o_{r+1}
}
$$

이 시점에서 score는 최대값

$$
\boxed{C_r=1}
$$

을 갖는다.

다음 bass 이전의 early release에는 비교적 완만하게 penalty를 주고, 다음 bass 이후의 overlap에는 더 빠르게 penalty를 주기 위해 **비대칭 Gaussian(split Gaussian)**을 사용한다.

---

# 18. Gaussian Width

현재 bass의 key-off와 다음 bass onset 사이의 gap이

$$
d_r=o_{r+1}-k_r
$$

일 때 왼쪽 Gaussian의 표준편차를

$$
\boxed{
\sigma_{L,r}=\frac{d_r}{2}
}
$$

로 둔다.

오른쪽 Gaussian은 그 절반으로

$$
\boxed{
\sigma_{R,r}
=
\frac{\sigma_{L,r}}{2}
=
\frac{d_r}{4}
}
$$

로 둔다.

Gaussian에서 peak에서 변곡점까지의 거리가 \(\sigma\)이므로, 이 설정은 다음 의미를 갖는다.

다음 bass **이전**에는

$$
\frac{d_r}{2}
$$

정도 일찍 release되었을 때 score가

$$
e^{-1/2}\approx0.6065
$$

가 된다.

반대로 다음 bass **이후**에는 고작

$$
\frac{d_r}{4}
$$

늦게 release되어도 똑같이

$$
e^{-1/2}\approx0.6065
$$

까지 떨어진다.

즉 동일한 score 감소가 오른쪽에서는 절반의 시간만에 발생한다.

따라서

$$
\boxed{
\text{Early release에는 상대적으로 관대}
}
$$

하면서

$$
\boxed{
\text{Bass overlap에는 더 엄격}
}
$$

한 scoring function이 된다.

---

# 19. Different-Bass Transition

현재 bass와 다음 bass의 MIDI pitch가 다르면

$$
\boxed{
p_r\neq p_{r+1}
}
$$

실제 bass가 교체되는 transition으로 본다.

key-off에서 pedal이 ON이고 이후 pedal release 시각 \(u_r\)이 존재할 때,

$$
\boxed{
C_r=
\begin{cases}
\displaystyle
\exp
\left[
-\frac{(u_r-o_{r+1})^2}
{2(d_r/2)^2}
\right],
&
u_r\le o_{r+1}
\\[16pt]
\displaystyle
\exp
\left[
-\frac{(u_r-o_{r+1})^2}
{2(d_r/4)^2}
\right],
&
u_r>o_{r+1}
\end{cases}
}
$$

로 정의한다.

따라서

$$
u_r=o_{r+1}
$$

이면

$$
\boxed{C_r=1}.
$$

다음 bass 이전에 너무 일찍 pedal을 release할수록 score가 낮아지고, 다음 bass 이후에도 pedal을 계속 유지할수록 오른쪽 Gaussian에 의해 더 빠르게 score가 낮아진다.

---

# 20. Different-Bass에서 Pedal이 끝까지 OFF되지 않는 경우

key-off 이후 MIDI가 끝날 때까지 pedal이 한 번도 OFF되지 않는다면 previous bass가 next bass 이후에도 계속 연결되고 있는 것이다.

Different-bass transition에서는 이를 **과도한 bass overlap**으로 취급하여

$$
\boxed{
C_r=0
}
$$

으로 둔다.

따라서 ALWAYS_ON은 different-bass transition에서 모두 0점을 받는다.

---

# 21. Same-Bass Transition

현재 bass와 다음 bass가 **정확히 같은 MIDI pitch**이면

$$
\boxed{
p_r=p_{r+1}
}
$$

same-bass transition으로 정의한다.

Octave equivalence는 사용하지 않는다.

즉 C2와 C3는 different bass이고, 정확히 C2 → C2일 때만 same-bass exception을 적용한다.

같은 bass가 반복되는 경우에는 다음 bass 이후까지 pedal을 연결하더라도 서로 다른 bass harmony가 겹치는 현상으로 보지 않는다.

따라서 다음 bass 이전에 pedal을 release한 경우에는 동일한 왼쪽 Gaussian을 사용한다.

$$
\boxed{
C_r=
\exp
\left[
-\frac{(u_r-o_{r+1})^2}
{2(d_r/2)^2}
\right],
\qquad
u_r<o_{r+1}.
}
$$

하지만 pedal이 다음 bass onset까지 유지되었다면

$$
\boxed{
u_r\ge o_{r+1}
\quad\Rightarrow\quad
C_r=1.
}
$$

즉 same-bass transition의 오른쪽 함수는 Gaussian이 아니라

$$
\boxed{C_r(t)=1}
$$

인 상수함수이다.

---

# 22. Same-Bass에서 Pedal이 끝까지 유지되는 경우

같은 bass가 다시 연주되었고, 현재 bass가 다음 bass까지 성공적으로 연결된 뒤 MIDI 끝까지 pedal이 OFF되지 않았다고 하더라도 Bass Connectivity 관점에서는 실패가 아니다.

따라서 same-bass transition에서는 이후 pedal OFF가 존재하지 않는 경우에도

$$
\boxed{C_r=1}
$$

로 둔다.

Validation control에서도 이 규칙이 정확히 작동하여 ALWAYS_ON의

$$
\boxed{
\text{same-bass mean}=1
}
$$

이 확인되었다. 

---

# 23. Local Score의 전체 정의

따라서 transition \(r\)의 Bass Connectivity \(C_r\)는 다음 논리로 계산된다.

현재 bass의 key-off 순간 pedal이 OFF라면

$$
\boxed{C_r=0}.
$$

그렇지 않고 same bass라면,

$$
\boxed{
C_r=
\begin{cases}
\displaystyle
\exp
\left[
-\frac{(u_r-o_{r+1})^2}
{2(d_r/2)^2}
\right],
&
u_r<o_{r+1}
\\[16pt]
1,
&
u_r\ge o_{r+1}
\end{cases}
}
$$

이며 이후 pedal OFF가 존재하지 않는 경우도 두 번째 branch에 포함하여 \(C_r=1\)로 본다.

Different bass라면,

$$
\boxed{
C_r=
\begin{cases}
\displaystyle
\exp
\left[
-\frac{(u_r-o_{r+1})^2}
{2(d_r/2)^2}
\right],
&
u_r\le o_{r+1}
\\[16pt]
\displaystyle
\exp
\left[
-\frac{(u_r-o_{r+1})^2}
{2(d_r/4)^2}
\right],
&
u_r>o_{r+1}
\end{cases}
}
$$

이며 pedal OFF가 끝까지 존재하지 않으면

$$
\boxed{C_r=0}.
$$

모든 경우

$$
\boxed{
0\le C_r\le1
}
$$

이다.

---

# 24. Decay, Velocity, Audio를 사용하지 않는 이유

Bass Connectivity v0에서는 note decay를 사용하지 않는다.

즉

$$
e^{-\lambda t}
$$

와 같은 acoustic decay weighting을 사용하지 않는다.

Velocity도 사용하지 않으며, audio rendering 역시 필요하지 않다.

이것은 의도적인 단순화이다.

본 metric의 목적은

> bass가 얼마나 크게 들리는가?

를 평가하는 것이 아니라

> **손이 떨어진 뒤 pedal이 bass를 다음 structural bass까지 얼마나 잘 연결하고, bass가 바뀌는 경우 얼마나 적절한 시점에 이전 bass를 release하는가?**

를 평가하는 것이기 때문이다.

따라서 최종 Bass Connectivity는 **MIDI-only functional metric**이다.

---

# 25. Performance-Level Score

한 performance 안에서 valid Bass transition 집합을 \(\mathcal T_B\)라 하면

$$
\boxed{
C_{\mathrm{Bass}}^{\mathrm{perf}}
=
\frac{1}{|\mathcal T_B|}
\sum_{r\in\mathcal T_B}C_r
}
$$

로 정의한다.

Valid transition이 하나도 없는 performance는

$$
0
$$

으로 두지 않고

$$
\boxed{\mathrm{NA}}
$$

로 처리한다.

0점은 “연결할 기회가 있었지만 모두 실패했다”라는 의미이므로, “평가할 기회가 없었다”와 구분해야 한다.

---

# 26. Piece-Level 및 System-Level Aggregation

같은 piece에 여러 Human performance가 존재하는 경우에는 먼저 performance score를 piece 내부에서 평균한다.

Piece \(q\)의 Human performance 집합을 \(\mathcal H_q\)라 하면

$$
\boxed{
C_{\mathrm{Bass}}^{(q)}
=
\frac{1}{|\mathcal H_q|}
\sum_{h\in\mathcal H_q}
C_{\mathrm{Bass},h}^{\mathrm{perf}}
}
$$

로 둔다.

그리고 system-level primary score는 **piece-balanced macro mean**이다.

$$
\boxed{
C_{\mathrm{Bass}}^{\mathrm{system}}
=
\frac1{|\mathcal P|}
\sum_{q\in\mathcal P}
C_{\mathrm{Bass}}^{(q)}
}
$$

즉 transition이 많은 곡이 system score를 지배하지 않도록 모든 piece에 동일한 weight를 준다.

Transition 전체를 pooling한 micro mean은 diagnostic으로만 사용한다. Validation report도 piece-balanced macro mean을 primary로 사용하였다. 

---

# 27. Metric이 의도적으로 평가하지 않는 것

Bass Connectivity는 **전체적인 pedal quality metric이 아니다.**

특히 이 metric은

> “이 곡에서 pedal을 사용해야 하는가?”

라는 stylistic question에는 답하지 않는다.

이미 Structural Bass로 검출된 bass를 기준으로

> **pedal을 사용했다면 bass connection을 얼마나 잘 수행했는가?**

를 평가한다.

따라서 Bach나 Haydn처럼 stylistically pedal 사용이 적은 repertoire에서는 인간 연주가 낮은 Bass Connectivity를 받을 수 있다.

이것은 현재 metric의 오류로 보지 않는다.

예를 들어 Haydn Keyboard Sonata 46-1에서 실제 validation score는

$$
C_{\mathrm{Bass}}^{\mathrm{Human}}
=
0.092752
$$

였지만,

$$
C_{\mathrm{Bass}}^{\mathrm{AlwaysOn}}
=
0.496894
$$

였다. 

Human pianist가 stylistically pedal을 거의 사용하지 않기 때문에 Bass Connectivity 자체에서는 낮은 값을 받을 수 있고, ALWAYS_ON은 same-bass connection을 기계적으로 충족하면서 더 높은 값을 얻을 수 있다.

이 현상을 제거하기 위해 repertoire-dependent pedal-necessity gate 등을 추가하지 않는다.

그러한 보정을 넣으면 Romantic repertoire에서 실제로 필요한 bass connection을 놓치거나 metric의 목적 자체가 불명확해질 위험이 있다.

따라서 Bass Connectivity는 명확하게

$$
\boxed{
\text{“Bass가 pedal로 연결되는가?”}
}
$$

라는 한 축만 담당한다.

반대로 과도한 sustain이 만들어내는 harmonic contamination은 별도의 **화성적 혼탁도(Harmonic Muddiness)** metric이 담당한다. 현재 화성적 혼탁도는 NO_PEDAL에서 0이지만 ALWAYS_ON에서 매우 큰 accumulation penalty가 발생하도록 설계되어 있으므로 두 metric은 상호보완적인 역할을 한다. 

---

# 28. 극단적인 Pedaling의 해석

이 두 metric의 관계를 이용하면 극단적인 pedaling을 명확하게 해석할 수 있다.

| Pedaling  |                     Bass Connectivity | Harmonic Muddiness | 해석                                            |
| --------- | ------------------------------------: | -----------------: | --------------------------------------------- |
| NO PEDAL  |                                    낮음 |                 낮음 | 깨끗하지만 필요한 bass connection 부족                  |
| 적절한 pedal |                                    높음 |                 낮음 | 필요한 bass를 연결하면서 불필요한 잔향은 제어                   |
| ALWAYS ON | same bass에서는 높음, different bass에서는 낮음 |              매우 높음 | 일부 bass는 연결하지만 bass 교체 및 harmonic clearing 실패 |

따라서 최종적으로 좋은 pedal은

$$
\boxed{
\text{High Bass Connectivity}
+
\text{Low Harmonic Muddiness}
}
$$

를 동시에 만족하는 방향을 지향한다.

---

# 29. ASAP Validation 평가

확정된 Bass Connectivity v0를 ASAP validation set의 **이미 존재하는 MIDI만 사용하여** 평가하였다.

새 inference, training, checkpoint execution, MIDI generation, audio rendering은 수행하지 않았다.

사용된 MIDI는 다음과 같다.

| System                       | Performances | Pieces |
| ---------------------------- | -----------: | -----: |
| HUMAN                        |           71 |     19 |
| ORIGINAL PT                  |           19 |     19 |
| STANDARD CE ARGMAX           |           19 |     19 |
| STANDARD CE POSTERIOR MEDIAN |           19 |     19 |
| WEIGHTED CE ARGMAX           |           19 |     19 |
| NO PEDAL                     |           13 |     13 |
| ALWAYS ON                    |           13 |     13 |

총 173개의 기존 MIDI가 사용되었다. NO_PEDAL과 ALWAYS_ON은 동일한 6곡의 기존 MIDI가 없었으며, 이를 새로 생성하지 않고 missing으로 유지했다. Test MIDI 접근, inference, training, GPU/CUDA 사용은 모두 0이었다. 

---

# 30. Main Validation Result

Primary metric인 piece-balanced macro mean 결과는 다음과 같다.

| System                           | Pieces | \(C_{\mathrm{Bass}}\) ↑ |
| -------------------------------- | -----: | ----------------------: |
| **HUMAN**                        |     19 |            **0.326718** |
| **STANDARD CE ARGMAX**           |     19 |            **0.220004** |
| **STANDARD CE POSTERIOR MEDIAN** |     19 |            **0.214062** |
| **WEIGHTED CE ARGMAX**           |     19 |            **0.208099** |
| **ORIGINAL PT**                  |     19 |            **0.187593** |
| ALWAYS ON                        |     13 |                0.251024 |
| NO PEDAL                         |     13 |            **0.000000** |

19곡의 공통 main system universe에서는

$$
\boxed{
\text{HUMAN}
>
\text{Standard CE Argmax}
>
\text{Standard CE Posterior Median}
>
\text{Weighted CE Argmax}
>
\text{Original PT}
}
$$

의 ordering이 나타났다. 

NO_PEDAL과 ALWAYS_ON은 13곡만 존재하므로 19곡 main ordering에 직접 포함시키지 않는다.

---

# 31. Human과 Model의 차이

Human은 piece-balanced macro 기준

$$
\boxed{0.326718}
$$

로 모든 canonical model보다 높았다.

Original PT는

$$
0.187593
$$

이었으며, 세 4-class system은 모두 Original PT보다 높은 값을 얻었다.

또한 per-piece 결과를 보면 Human은 Original PT보다 **19곡 중 18곡에서 높은 Bass Connectivity**를 보였다. 

이는 단순히 몇몇 곡의 큰 값 때문에 system-level average가 올라간 결과라기보다는, 대부분의 validation piece에서 human performance가 PT보다 structural bass를 더 잘 연결하는 방향을 보였음을 시사한다.

4-class Stage 2 계열이 Original PT보다 높은 값을 보였다는 점도 긍정적인 결과이다.

즉 Stage 2의 pedal 수정이 기존 PT보다 structural-bass connection을 어느 정도 개선했을 가능성을 보여주는 sanity evidence로 해석할 수 있다.

단, 이 validation은 human perceptual study가 아니므로 이를 곧바로 “지각적으로 더 좋은 pedal”의 증명으로 해석하지 않는다.

---

# 32. Transition-Level 분포

Transition-level diagnostic은 다음과 같았다.

| System                       |     Mean |       Median |
| ---------------------------- | -------: | -----------: |
| HUMAN                        | 0.385267 | **0.222336** |
| ORIGINAL PT                  | 0.225754 | **0.000000** |
| Standard CE Argmax           | 0.266195 | **0.000000** |
| Standard CE Posterior Median | 0.261888 | **0.000000** |
| Weighted CE Argmax           | 0.255273 | **0.000000** |

Human에서는 transition median이 positive였지만 모든 canonical model의 median은 0이었다. 

이 차이는 physical key-off 순간 pedal이 이미 OFF인 transition 수에서도 나타났다.

Human에서는

$$
3931/13182
$$

개의 valid transition에서 key-off 시 pedal이 OFF였던 반면,

Original PT는

$$
966/1810,
$$

Standard CE Argmax는

$$
956/1810,
$$

Posterior Median은

$$
982/1810,
$$

Weighted CE는

$$
1012/1810
$$

이었다. 

즉 Human은 structural bass의 손이 떨어지는 순간 pedal로 bass를 이어주고 있는 비율이 모델보다 훨씬 높았다.

이는 Bass Connectivity metric의 원래 motivation과 잘 맞는 결과이다.

---

# 33. Same-Bass / Different-Bass 결과

Human에서는

$$
\boxed{
C_{\mathrm{same}}=0.542639
}
$$

$$
\boxed{
C_{\mathrm{different}}=0.332735
}
$$

였다.

Original PT는

$$
C_{\mathrm{same}}=0.331333,
$$

$$
C_{\mathrm{different}}=0.190509
$$

였다.

세 4-class system의 different-bass mean은 약

$$
0.244\sim0.252
$$

수준으로 Original PT보다 높았다. 

특히 synthetic extreme인 ALWAYS_ON은 정의상

$$
\boxed{
C_{\mathrm{same}}=1.0
}
$$

$$
\boxed{
C_{\mathrm{different}}=0.0
}
$$

을 정확히 기록했다.

NO_PEDAL은 모든 valid opportunity에서 key-off 순간 pedal이 OFF였으므로

$$
\boxed{
C_{\mathrm{Bass}}=0
}
$$

이었다. 

따라서 same-bass exception, different-bass overlap penalty, no-pedal zero rule이 모두 의도대로 작동하였다.

---

# 34. Implementation 검증

Metric의 기본 논리를 검증하기 위해 8개의 synthetic logic test를 수행하였다.

Key-off 시 pedal OFF → 0, different-bass release가 정확히 next onset → 1, 왼쪽 \(d/2\)와 오른쪽 \(d/4\)에서 각각 \(e^{-1/2}\), same-bass가 next onset까지 연결되면 1, no subsequent pedal OFF의 same/different 처리, key-held transition exclusion, 20 ms earliest-anchor grouping을 포함한 **8/8 test가 모두 PASS**하였다. 

또한 Original PT, 4-class variants, NO_PEDAL, ALWAYS_ON처럼 동일한 non-pedal performance를 공유하는 canonical MIDI 사이에서

* non-CC64 MIDI event,
* Structural Bass detection,
* valid transition set

의 identity를 검증하였다.

총

$$
\boxed{102/102}
$$

canonical-variant comparison이 모두 통과했고 failure는 0이었다. 

따라서 system 간 score 차이는 note performance 차이가 아니라 CC64 pedal behavior의 차이에서 발생하도록 통제되었다.

---

# 35. 현재 평가 결과의 해석

현재 validation 결과는 **Bass Connectivity v0의 첫 sanity evaluation으로서는 상당히 긍정적**이다.

특히 다음 현상이 동시에 확인되었다.

$$
\boxed{
\text{NO PEDAL}=0
}
$$

이고,

$$
\boxed{
\text{ALWAYS ON: same}=1,\quad different=0
}
$$

이며,

$$
\boxed{
\text{Human}>\text{all canonical models}
}
$$

이고,

$$
\boxed{
\text{all three 4-class systems}>\text{Original PT}
}
$$

라는 자연스러운 system-level pattern이 나타났다. 

또한 이 결과는 validation 결과를 본 뒤 parameter를 변경해 만든 ordering이 아니다.

Structural Bass threshold

$$
B_n\ge0.5,
$$

Structural-timescale parameters

$$
R_{\mathrm{low}}=1.5,\qquad
R_{\mathrm{high}}=3.0,
$$

octave bonus

$$
\beta_{\mathrm{oct}}=0.2,
$$

Gaussian width

$$
\sigma_L=d/2,\qquad
\sigma_R=d/4
$$

를 모두 먼저 고정한 후 measurement-only evaluation을 수행했고, 결과 확인 후 coefficient, threshold, event semantics, post-processing은 수정하지 않았다. 

---

# 36. 최종 해석 및 범위

Bass Connectivity는 **Human-likeness metric이 아니다.**

Human이 반드시 모든 piece에서 가장 높은 점수를 받아야 하는 것도 아니다.

특히 Bach나 Haydn처럼 pedal-sparse한 repertoire에서는 stylistically 적절한 Human performance가 pedal을 거의 사용하지 않아 낮은 Bass Connectivity를 받을 수 있다.

반대로 ALWAYS_ON은 같은 bass를 계속 연결하면서 일부 높은 점수를 얻을 수 있다.

이것은 metric의 목적상 허용한다.

Bass Connectivity가 측정하는 것은 오직

$$
\boxed{
\text{“Structural Bass가 pedal을 통해 얼마나 잘 연결되고 교체되는가?”}
}
$$

이기 때문이다.

“그 pedal connection 자체가 해당 repertoire에서 필요한가?”, “연결하면서 다른 음을 지나치게 남기고 있는가?”, “화성이 지저분해졌는가?”는 별도의 평가 축에서 담당한다.

따라서 현재 연구에서 Bass Connectivity는 화성적 혼탁도와 함께 다음과 같은 상호보완적 역할을 갖는다.

$$
\boxed{
\text{Bass Connectivity: 필요한 저음을 제대로 남겼는가?}
}
$$

$$
\boxed{
\text{Harmonic Muddiness: 남기지 말아야 할 음을 너무 많이 남겼는가?}
}
$$

이 두 축을 함께 사용하는 것이 pedal 사용을 단순히 억제하거나 단순히 증가시키는 방향으로 metric이 퇴화하는 것을 막는 핵심 아이디어이다.

---

# 37. 최종 정의 요약

Structural Bass detection은

$$
\boxed{
p_n=\min_{i\in O_n}m_i
}
$$

$$
\boxed{
L_n=
\begin{cases}
1,&p_n\le52\\
(57-p_n)/5,&52<p_n<57\\
0,&p_n\ge57
\end{cases}
}
$$

$$
\boxed{
R_n^B=
\frac{
t_n^{\mathrm{next-low}}-t_n
}{
\widetilde{\mathrm{IOI}}_n
}
}
$$

$$
\boxed{
S_n=
\operatorname{clip}_{[0,1]}
\left(
\frac{R_n^B-1.5}{1.5}
\right)
}
$$

$$
\boxed{
O_n^{\mathrm{oct}}
=
\mathbf1[p_n+12\in O_n]
}
$$

$$
\boxed{
B_n=
\operatorname{clip}_{[0,1]}
\left[
L_nS_n(1+0.2O_n^{\mathrm{oct}})
\right]
}
$$

이며,

$$
\boxed{
B_n\ge0.5
\Rightarrow
\text{Structural Bass}
}
$$

로 고정한다.

연속된 Structural Bass pair에 대해

$$
\boxed{
d_r=o_{r+1}-k_r
}
$$

를 정의하고,

$$
k_r\ge o_{r+1}
$$

이면 pedal opportunity에서 제외한다.

Key-off 시 pedal이 OFF라면

$$
\boxed{C_r=0}.
$$

Different-bass transition에서는

$$
\boxed{
C_r=
\begin{cases}
\displaystyle
\exp
\left[
-\frac{(u_r-o_{r+1})^2}
{2(d_r/2)^2}
\right],
&
u_r\le o_{r+1}
\\[16pt]
\displaystyle
\exp
\left[
-\frac{(u_r-o_{r+1})^2}
{2(d_r/4)^2}
\right],
&
u_r>o_{r+1},
\end{cases}
}
$$

를 사용하며 이후 pedal OFF가 존재하지 않으면 0으로 둔다.

Same-bass transition에서는

$$
\boxed{
C_r=
\begin{cases}
\displaystyle
\exp
\left[
-\frac{(u_r-o_{r+1})^2}
{2(d_r/2)^2}
\right],
&
u_r<o_{r+1}
\\[16pt]
1,
&
u_r\ge o_{r+1},
\end{cases}
}
$$

이고 이후 pedal OFF가 없더라도 1로 둔다.

최종 performance/piece/system score는 valid Bass transition의 평균을 기반으로 하며 primary system comparison에서는 piece-balanced macro mean을 사용한다.

$$
\boxed{
\displaystyle
C_{\mathrm{Bass}}
=
\frac1{|\mathcal T_B|}
\sum_{r\in\mathcal T_B}C_r
}
$$

이것을 현재 연구의 **최종 ‘베이스 연결도(Bass Connectivity) metric v0’**로 고정한다.
