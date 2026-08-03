- 7/29 진행 상황 보고
    
    ## 주제 1 - Pianist Transformer (PT) 의 페달링을 개선해보자
    
    ### Intro
    
    기존의 Pianist Transformer는 score MIDI를 입력받아 한 note마다
    
    $[Pitch,IOI,Velocity,Duration,Pedal1,Pedal2,Pedal3,Pedal4]$
    
    8개의 token을 autoregressive하게 생성한다. Pedal1–4는 다음 note까지의 구간($IOI$)에서 네 지점의 sustain-pedal state를 표현한다. 
    
    → 하지만 전반적으로 페달링의 퀄리티가 아쉽다 (지저분하게 밟히는 곳이 많고, 중요한 음들이 지속되지 않는 경우도 많다) 
    
    → CC64 event 는 0~127의 연속적인 값으로 표현될 수 있지만, PT는 0 or 127 로만 예측해서 디테일이 떨어진다. 
    
    ***페달링이 아쉬운 이유가 무엇일까?***
    
    1. (논문에서도 밝히는 이유) 약 10B MIDI 로 SPT를 한 후에, ASAP 이라는 (좋은 퀄리티의) 데이터로 SFT를 한다. 하지만 SPT에 활용하는 MIDI들은 페달 정보가 없는 것도 많고, 있는 것도 디테일하게 기록되지 않고 on/ off 만 기록되어 있다. 그래서 페달링을 0 or 127 로 예측하는 것으로 너무 굳어져버린다. 
    
    → 좋은 퀄리티의 데이터로 페달링을 더 충분하게 SFT 할 수는 없을까? 
    
    (SFT를 하기 위해서는 score MIDI → performance MIDI pair가 필요하다)
    
    2. PT의 토크나이저가 페달링을 충분하게 표현하지 못한다. 
    - PT의 토크나이저가 페달링 정보를 얼만큼 보존하는가?
        
        
        대상: 바흐 푸가 C major 의 ASAP performance MIDI 
        
        결과: 
        
        - ASAP performance MIDI에서 intermediate CC64의 비율이 94.6%에 달한다.
        - tokenizer가 intermediate value를 거의 그대로 보존했다. (시간점유율: 53.59% → 52.85%)
        - JS divergence = 0.0119
        - 근데 Repedaling은 32개 중에서 14개를 잃고 18개만 보존했다.
        
        (Repadling 은 휴리스틱하게 임의로 정의하였다 =  ≥96 → ≤31 → ≥96 & 300ms 이내)
        
        !image.png
        
        - 그리고 IOI의 네 지점에서, 바로 직전의 CC64 값을 가져오기 때문에 모델링이 전반적으로 오른쪽으로 밀린다.
        
        - PT 논문에서 제공하는 performance MIDI 165개
            
            !image.png
            
            !image.png
            
        
    
    → 페달 토큰 모델링을 조금 더 잘 해볼 수 없을까?
    
    ***어떻게 해결해볼까?***
    
    두 개의 step으로 분리해보는 것은 어떨까?
    
    $$
    \boxed{\text{Score}\xrightarrow{\text{Stage 1}}\text{Pedal-free expressive performance}\xrightarrow{\text{Stage 2}}\text{Pedaled performance}}
    $$
    
    **이유 1) Stage 2에서 페달링을 학습할 때 기존의 score MIDI → performance MIDI pair가 필요하지 않다! performance MIDI 만 있으면 페달 event만 mask 해서 학습 데이터로 활용할 수 있다. 
    
    *이유 2) 페달 모델링에 조금 더 집중할 수 있다. 그리고 expressive performance context를 보면서 페달링을 예측하면 더 좋은 추론을 할 수 있지 않을까? 
    
    ### Stage 1 - Performance Renderer
    
    **옵션 A) Original PT + Pedal Removal**
    
    ```
    기존 Pianist Transformer checkpoint
    ↓
    기존처럼 8 tokens/note 생성
    ↓
    Pedal1–4만 제거
    ↓
    Pitch / IOI / Velocity / Duration 사용
    ```
    
    → 8개는 원래대로 전부 생성해야 다음 note의 4개 non-pedal token도 원래 모델과 똑같은 context에서 생성된다. 
    
    - 장점
        - PT의 기존 performance-rendering 능력을 최대한 보존한다
        - Stage 2를 실험할 때 결과 변화의 원인을 특정하기 더 쉬워진다
    - 단점
        - 계산이 효율적이지 않다  (but 통제성 > 계산 효율)
    
    출력: Pitch / IOI / Velocity / Duration ~~+ Pedal 1~4~~
    
    - 옵션 B) 4-token Renderer
        
        📌Stage 2의 효과가 확인된 이후에, 모델의 효율을 개선하고자 할 때 시도해본다. 
        
        PT checkpoint를 가지고 4-token grammar로 추가 SFT 시켜보자
        
    
    ### Stage 2 - Pedal Renderer
    
    - Encoder-only Bidirectional Transformer을 small 하게 만들자
    
    → 이미 곡 전체의 performance가 완성되어 있기 때문에 pedal을 autoregressive하게 생성할 필요가 없다. 
    
    (하지만 autoregressive하게 생성하는 것 더 좋은 출력을 만들 수 있을까?)
    
    - Embedding - Pianist Transformer에서 학습된 embedding/projection을 transfer 하자.
    
    **옵션 A) transferred embedding freeze 하기**
    
    옵션 B) 초기 epoch에는 freeze 해두었다가 후반 epoch에 낮은 학습률로 unfreeze해서 pedal task에 맞춰 adaptation 살짝 해보기
    
    - 입력 - Stage 1에서 생성된 각 note의 Pitch / IOI / Velocity / Duration
    
    → Bidirectional Transformer가 전체 performance sequence를 처리한 뒤 각 IOI에 대해 $h_i$ 라는 contextual representation을 만든다. 
    
    - 디테일
        
        새로운 sequence를 생성할 필요가 없고, 생성된 sequence에 페달 label을 붙이는 task로 이해할 수 있으므로 encoder-only 구조로 가자. 
        
        **Step 1. Feature embedding**
        
        → 각 note의 네 feature를 vector로 바꾸고, 합쳐서 하나의 note embedding 을 만든다. 
        
        **Step 2. Positional embedding 추가**
        
        **Step 3. Bidirectional Transformer encoder 통과**
        
        → 해당 note 자체의 feature뿐 아니라 곡의 전후 문맥까지 반영한 representation $h_i$ 를 얻는다.
        
        **Step 4. Onset-group aggregation**
        
        → 같은 onset에 속한 note들의 $h_j$ 를 하나로 합친다.
        
        → 마지막 note의 representation을 사용한다. (baseline과의 통일성)
        
        **Step 5. Prediction heads**
        
        → 하나의 onset-interval representation 에서 Event, Timing, Depth의 확률을 예측한다
        
        Sequence를 어떻게 구성할 것인가? 
        
        → PT가 어떻게 구성했는지 살펴보고 baseline을 최대한 따르자
        
    
    ***페달링을 어떻게 모델링할 것인가?***
    
    옵션 A) 기존 pedal token과 동일하게 가기
    
    **옵션 B) event-based modeling으로 바꾸기**
    
    (*idea - 사실 페달링은 단순하다. 언제 눌렀다가 언제 뗄지를 잘 모델링하면 그게 끝이다*)
    
    각 IOI i에 대해서 $Y_i=(E_{i,1},E_{i,2})$ 을 예측한다. 
    
    $E_{i,k}=(c_{i,k},τ_{i,k},d_{i,k})$ 
    
    → $c_{i,k}∈(None,Up,Down)$ 
    
    → $τ_{i,k}=\frac{t_{event}−t_i}{t_{i+1}−t_i} ∈[0,1]$    ( $t_{event}=t_i+τ(t_{i+1}−t_i)$ )
    
    → $D_i∈(Full(127), Half(90))$
    
    (half 기준값이 뭐가 좋을지는 고민 필요하다)
    
    <Rule>
    
    Up / down 의 기준 (CC64 = 64 를 기준)
    
    $<64→\;≥64$ : DOWN
    $≥64→\;<64$ : UP
    
    - 좀 더 고민해보자
        
        (down, down) 인데 depth 가 달라지는 상황도 학습할 수 있으려면, Down의 기준이 조금 달라져야 한다. 
        
        예시 ) $v_{old}≥64,v_{new}≥64,∣v_{new}−v_{old}∣≥δ_D\;⇒\;DOWN$
        
    
    ---
    
    $c_{i,1}, \; c_{i,2}$ 는 시간 순서대로 배정한다
    
    ---
    
    event가 하나 뿐이면 (slot 1 = event / slot 2 = none ) 으로 배정한다. 
    
    IOI 내에서 UP 이 한번 나온다면 → $c_{i,1} = Up, \; c_{i,2}=None$
    IOI 내에서 pedal event가 등장하지 않는다면 → $c_{i,1} = None, \; c_{i,2}=None$
    
    ---
    
    event 가 반복되는 것도 허용한다. 
    즉 (Up, Up) , (Down, Down) 도 가능하다.
    
    ---
    
    none slot에서는 timing target 자체가 의미가 없으므로 timing loss를 계산하지 않는다
    
    ---
    
    만약 IOI내에서 pedal event가 3번 이상이라면
    
    → 홀수 개일 때는 마지막 transition만 slot 1로 기록 (최종상태 보존)
    
    → 짝수 개일 때는 마지막 2개의 transition만 slot 1, 2로 기록 (repedaling 보존)
    
    ---
    
    $D_i$ 는 $c_{i,k} = Down$  일 때만 활성화된다
    
    ---
    
    ***LOSS 를 어떻게 설계할 것인가?***
    
    $$
    L=\lambda_E L_{\mathrm{event}}+\lambda_T L_{\mathrm{time}}+\lambda_D L_{\mathrm{depth}}
    $$
    
    ---
    
    - batch 마다 event, time, depth의 수가 다를 수 있다.
    
    → 각 loss를 따로 평균내서 표현하자
    
    Event Loss
    
    $$
    \mathcal{L}_{\mathrm{event}}=\frac{1}{2|\mathcal{I}|}\sum_{i \in \mathcal{I}}\sum_{k=1}^{2}\mathrm{CE}_{w(k)}\left(\hat{\mathbf{p}}_{i,k},c_{i,k}\right)
    $$
    
    - Slot 1,2 의 None 비율
        
        !image.png
        
    
    Slot 1,2에는 None event 가 압도적으로 많이 등장한다. 
    
    → Weighted Cross-Entropy 사용  
    ($CE{w^{(k)}}=−w_{c_i,k}^{(k)}log\;p_{i,k,ci,k}$)
    
    Timing Loss
    
    $$
    \mathcal{L}_{\mathrm{time}}=\frac{\displaystyle\sum_{i \in \mathcal{I}}\sum_{k=1}^{2}\mathbf{1}\left[c_{i,k} \neq \mathrm{NONE}\right]\operatorname{Huber}_{\beta_T}\left(\hat{\tau}_{i,k}-\tau_{i,k}\right)}{\displaystyle\sum_{i \in \mathcal{I}}\sum_{k=1}^{2}\mathbf{1}\left[c_{i,k} \neq \mathrm{NONE}\right]+\epsilon}
    $$
    
    $\hat{\tau}_{i,k} ∈[0,1]$ / $τ_{i,k}∈[0,1]$
    
    작은 오차에서는 MSE처럼 제곱으로, 큰 오차에서는 MAE 처럼 선형으로 동작하는 Huber Loss를 사용하자
    
    $β_T$ : Huber loss가 quadratic 영역에서 linear 영역으로 바뀌는 기준
    
    평균을 냄으로서 유효한 pedal event 하나당 평균 timing error를 계산하도록 한다
    
    - Huber Loss
        
        $$
        \operatorname{Huber}_{\delta}(e)=\begin{cases}\dfrac{1}{2}e^2,& |e|\leq\delta \\[2mm]\delta\left(|e|-\dfrac{1}{2}\delta\right),& |e|>\delta\end{cases}
        $$
        
        → 제곱 형태는 오차가 0에 가까워질수록 gradient도 작아진다. 따라서 모델이 정답 근처에서 timing을 세밀하게 조정할 수 있다.
        
        → 큰 오차에서 loss가 선형으로 증가하기 때문에, outlier timing error 하나가 전체 학습을 지나치게 지배하는 것을 막을 수 있다.
        
        $$
        \frac{\partial\operatorname{Huber}_{\delta}(e)}{\partial e}=\begin{cases}e,& |e|\leq\delta \\[1mm]\delta\,\operatorname{sign}(e),& |e|>\delta\end{cases}
        $$
        
        → 오차가 작아질수록 gradient도 작아진다. 따라서 정답 근처에서 parameter를 너무 크게 바꾸지 않고 세밀하게 조정한다. 
        
        → gradient의 크기는 $\delta$ 로 제한된다. 
        
        ⇒ Smooth L1 Loss도 비슷한 맥락. 사용하기 편한걸로 고르자. 
        
    
    Depth Loss
    
    $$
    \mathcal{L}_{\mathrm{depth}}=\frac{\displaystyle\sum_{i\in\mathcal{I}}\sum_{k=1}^{2}\mathbf{1}\!\left[c_{i,k}=\mathrm{DOWN}\right]\mathrm{CE}_{\mathbf{w}^{D}}\left(\tilde{\mathbf{q}}_{i,k},d_{i,k}\right)}{\displaystyle\sum_{i\in\mathcal{I}}\sum_{k=1}^{2}\mathbf{1}\!\left[c_{i,k}=\mathrm{DOWN}\right]+\epsilon}
    $$
    
    - Full / half 페달의 비율
        
        !image.png (Full 이 압도적으로 많다)
        
    
    ## 주제 2 - 모델의 페달링을 정량적으로 평가할 metric을 제안해보자
    
    ### Motivation
    
    ***기존에는 페달링 평가를 어떻게 했는가?***
    
    | 평가 방식 | 대표 사례 | 무엇을 측정하는가 | 핵심 한계 |
    | --- | --- | --- | --- |
    | 정답 연주와의 오차 | VirtuosoNet | 예측 페달값과 특정 인간 연주의 페달값 사이 MSE | 페달링 자체가 좋은지는 알기 어려움 |
    | 인간 데이터와의 분포 유사도 | Pianist Transformer | 페달 패턴 분포의 JS divergence와 intersection area | 전체적인 특징만 보고, 국소적인 특징은 보지 못함 |
    | 청취 평가 | VirtuosoNet, Pianist Transformer | 자연스러움, articulation, human-likeness, 선호도 | 비용이 크고, 원인을 정량적으로 해석하기 어려움 |
    
    ‘좋은 페달링’에 대한 거시적인 원칙들은 존재한다. 
    
    → 생성된 연주 자체만을 보고, 페달링이 연주를 얼마나 개선했는지 혹은 훼손했는지를 정량적으로 평가해볼 수 있지 않을까?
    
    ### Brainstorming
    
    - 악보를 기반으로 평가할 것인가, 연주를 기반으로 평가할 것인가?
    
    → 같은 악보여도, 연주 스타일이 다르면 ‘좋은 페달링’에 대한 평가도 달라져야 할 수 있다. 따라서 일단 **연주 기반 평가**로 가자 (future work로 연주와 악보를 동시에 참조하는 것도 생각해보자)
    
    - ‘연주되는 소리’과 ‘페달에 의해 추가된 소리’를 구분하자.
    
    → 페달에 의해 추가된 성분에 대해 정량적으로 평가하자
    
    ---
    
    - metric의 구성 요소?
    
    <페달을 제어하는 방향>
    
    - 피드백 !!
        
        경수님: encoder only 만 사용하게 되면 각 결과들이 조건부 독립. 결과가 잘 안 나올 수 있다 (그 이유에 대해서 이해를 못했는데, 한번 찾아보자)
        
        그래서 오히려 encoder- decoder 을 같이 쓰거나, decoder만(+prefix) 쓰는 방식으로 하는게 학습이 더 안정될 수 있다. 
        
        현우님: 사람마다 페달링 특징이 다를텐데, 일관성이 없으면 학습이 잘 안 될수도 있지 않을까. 
        
        ⇒ encoder only / decoder only / encoder-decoder transformer에 대한 이론적 이해가 필요하다. 
        
        ⇒ 학습 데이터 마련할 때, 페달링이 지나치게 특이한 (?) 데이터가 있는지 확인해볼 필요는 있겠다.