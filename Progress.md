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



8/12 진행 상황 보고
    
    ## 주제 1 - Pianist Transformer (PT) 의 페달링을 개선해보자
    
    > Stage 분리된 아키텍쳐가 페달링을 실제로 개선하는지 확인해보자.
    > 
    
    → 변인을 오직 Stage 분리에만 두어야 한다!
    
    - Stage 1 에서는 PT로 inference해서 결과 midi를 얻고, 이 midi에서 pedal 1~4 token만 mask 해서 Stage 2 에 전달한다.
    - Non-pedal token 은 Stage 1의 출력과 Stage 2의 출력이 같도록 고정한다.
    - Onset-group 으로 묶지 않고, note 별로 inference 한다. (IOI = 0 허용)
    - Loss도 가장 단순한 standard cross-entropy 사용한다.
    - 학습 데이터도 PT와 동일하게 ASAP 만 사용한다.
    
    ### 실험 1 - 기존 토크나이저 + Stage 2 분리 (encoder-only)
    
    - ASAP 데이터 셋 준비
        
        ASAP performance MIDI 에 전부 토크나이저 적용. 
        
        - ASAP v1.1 데이터 audit
        
        | 항목 | 결과 |
        | --- | --- |
        | Total performances | 1,067 |
        | Unique pieces `(composer, title)` | 222 |
        | Composers | 16 |
        | Total tokens | 28,832,944 |
        | Pedal tokens | 14,416,472 |
        - pedal 분포
        
        | CC64 class | 개수 | 비율 |
        | --- | --- | --- |
        | Zero, 0 | 3,708,796 | 25.73% |
        | Full, 127 | 5,281,862 | 36.64% |
        | Intermediate, 1–126 | 5,425,814 | 37.64% |
        
        | Slot | Intermediate 비율 |
        | --- | --- |
        | Pedal1 | 37.90% |
        | Pedal2 | 37.81% |
        | Pedal3 | 37.47% |
        | Pedal4 | 37.37% |
        - CC64가 없는 13개 performance
        
        → 전체의 1.22% / 직접 확인한 결과 모두 Bach repertoire / 실제 무페달 또는 매우 제한된 pedal 사용이 음악적으로 타당하다고 판단
        
        - Data split
        
        | Split | Performances | Pieces | No-CC64 performances |
        | --- | --- | --- | --- |
        | Train | 892 | 180 | 12 |
        | Validation | 71 | 19 | 0 |
        | Test | 104 | 23 | 1 |
        | **Total** | **1,067** | **222** | **13** |
        
        - Stage 2 Pedal Dataset 제작
        
        원래 performance midi token: $\mathbf{x}_i=[p_i,\Delta_i,v_i,d_i,q_{i,1},q_{i,2},q_{i,3},q_{i,4}]$
        
        입력 : $\mathbf{x}_i=[p_i,\Delta_i,v_i,d_i,\mathrm{MASK},\mathrm{MASK},\mathrm{MASK},\mathrm{MASK}]$
        
        정답: $\mathbf{y}_i=[q_{i,1},q_{i,2},q_{i,3},q_{i,4}]$
        
        $$
        y_{i,k}^{\mathrm{class}}=y_{i,k}^{\mathrm{token}}-5261\\y_{i,k}^{\mathrm{class}}\in\{0,1,\ldots,127\}
        $$
        
        (5261 : Pedal vocabulary offset)
        
    - Windowing 설정
        
        PT에서 사용하는 maximum sequence length : 4096 tokens.
        
        → ASAP performance의 98.88%가 4,096 token보다 길기 때문에 windowing이 필요하다. 
        
        한 음당 8 tokens : 4096 tokens → 512 notes
        
        > window_notes = 512  /  stride_notes = 256
        > 
        
        같은 note에 대한 예측이 2개씩 생기는데, 이때 두 개의 prediction logits을 평균내서 사용. 
        
    - 여러가지 디테일링
        - 각 note에서 Pedal 1~4 token 4개를 각각 120-class로 예측
        
        $$
        \mathbf{z}_{i,k}=\mathbf{W}_k\mathbf{h}_i+\mathbf{b}_k,\qquad k\in\{1,2,3,4\}\\\mathbf{Z}
        \in
        \mathbb{R}^{B\times N\times4\times128}
        $$
        
        - Loss function 정의 - standard unweighted CE
        
        $$
        \mathcal{L}_{\mathrm{CE}}
        =
        -\frac{1}{|\mathcal{V}|}
        \sum_{(b,i,k)\in\mathcal{V}}
        \log
        \frac{
        \exp z_{b,i,k,y_{b,i,k}}
        }{
        \sum_{c=0}^{127}\exp z_{b,i,k,c}
        }\\\mathcal{V}
        =
        \{(b,i,k)\mid y_{b,i,k}\neq-100\}
        $$
        
        (-100 : ignore index)
        
        → encoder-only 이기 때문에, Pedal 1~4가 모두 같은 $h_i$ 는 보지만 서로의 예측값은 보지 않는다. 
        
    - PT 의 기본 성능 (좋은 metric은 아니다)
        
        이미 inference 해둔 23개의 original PT midi를 사용. ASAP test에는 같은 23개의 midi에 대해 총 104개의 human performance가 존재. 같은 곡에 대해 PT midi와 performance midi를 note 별로 align 해준 뒤에, 모든 PT vs human midi를 비교해서 결과를 도출 (총 104개의 비교)
        
        (Stage 2 의 평가와 metric은 갖지만, 동일한 조건은 아니다!)
        
        - 결과 (128 class)
        
        | Metric | Original PT | 해석 |
        | --- | --- | --- |
        | **Pedal-token accuracy** | **0.5031** | Pedal1–4 각각의 CC64 값을 정확히 맞힌 비율 |
        | **Exact-note accuracy** | **0.4563** | 한 note의 Pedal1–4를 모두 정확히 맞힌 비율 |
        | Pedal1 accuracy | 0.5019 | Pedal1 exact accuracy |
        | Pedal2 accuracy | 0.5011 | Pedal2 exact accuracy |
        | Pedal3 accuracy | 0.5042 | Pedal3 exact accuracy |
        | Pedal4 accuracy | 0.5054 | Pedal4 exact accuracy |
        | **Raw CC64 MAE ↓** | **40.9381** | 예측 CC64와 human CC64의 평균 절대 차이 |
        | **Transition accuracy** | **0.1006** | human pedal 값이 바뀐 위치에서 새 CC64 값을 정확히 맞힌 비율 |
        | **Steady-position accuracy** | **0.5741** | human pedal 값이 유지된 위치에서 exact value를 맞힌 비율 |
        | Transition precision | 0.4799 | PT가 transition을 낸 위치 중 human도 transition인 비율 |
        | Transition recall | 0.0652 | human transition 중 PT가 실제로 transition을 낸 비율 |
        | **Transition F1** | **0.1148** | transition precision/recall의 조화평균 |
        | Intermediate exact accuracy | **0.0000** | human CC64가 1–126일 때 exact match 비율 |
        | Intermediate MAE ↓ | **58.1772** | intermediate target에서의 평균 CC64 오차 |
        | Intermediate endpoint collapse | **1.0000** | intermediate target을 0/127로 보낸 비율 |
        | Predicted ZERO ratio | 0.5533 | PT 출력 중 CC64=0 비율 |
        | Predicted INTERMEDIATE ratio | **0.0000** | PT 출력 중 CC64 1–126 비율 |
        | Predicted FULL ratio | 0.4467 | PT 출력 중 CC64=127 비율 |
        - 결과 (5 class)
        
        | Metric | Original PT |
        | --- | --- |
        | **5-class token accuracy** | **0.5031** |
        | **5-class exact-note accuracy** | **0.4563** |
        | **Macro F1** | **0.2397** |
        | Micro F1 | 0.5031 |
        | Weighted F1 | 0.4064 |
        | **OFF/ON state accuracy** | **0.6921** |
        | Binary transition precision | 0.1739 |
        | Binary transition recall | 0.1030 |
        | **Binary transition F1** | **0.1294** |
        | 5-class decoded MAE ↓ | 40.9381 |
        | Decoded tolerance ±5 | 0.5108 |
        | Decoded tolerance ±10 | 0.5148 |
        | Decoded tolerance ±20 | 0.5310 |
        
        | Class | Precision | Recall | F1 | Predicted ratio |
        | --- | --- | --- | --- | --- |
        | ZERO | 0.4887 | 0.7922 | 0.6045 | 0.5533 |
        | LOW | 0 | 0 | 0 | 0 |
        | MID | 0 | 0 | 0 | 0 |
        | HIGH | 0 | 0 | 0 | 0 |
        | FULL | 0.5211 | 0.6914 | 0.5943 | 0.4467 |
    
    - 첫 번째 학습
        
        최대 20 epoch까지 설정했지만 validation loss가 더 이상 충분히 개선되지 않아 epoch 6에서 early stopping. 가장 좋은 모델은 epoch 2.
        
        - 마지막 epoch 6 결과
        
        | Metric | Train | Validation |
        | --- | --- | --- |
        | Loss | 2.3575 | 2.2855 |
        | Pedal-token accuracy | 55.70% | 57.24% |
        | Exact-note accuracy | 50.32% | 52.05% |
        | Pedal1 accuracy | 55.49% | 57.08% |
        | Pedal2 accuracy | 55.51% | 57.08% |
        | Pedal3 accuracy | 55.90% | 57.38% |
        | Pedal4 accuracy | 55.91% | 57.40% |
        | Pedal-value MAE | 26.55 | 30.77 |
        
        → epoch 2부터 유의미한 개선이 없었다 : *이유가 뭘까?*
        
        - 여러 가지 평가
        
        Stage 2 예측 분포 :
        
        | 0 예측 | 52.20% |
        | --- | --- |
        | 127 예측 | 47.80% |
        | 1–126 예 | 0.00% |
        
        Human target의 32.22%가 intermediate인데도 **단 하나의 intermediate class도 argmax로 선택되지 않았다.** (기존 PT 와 같은 문제 발생)
        
        | 구간 | Accuracy | MAE |
        | --- | --- | --- |
        | Steady positions | 65.32% | 27.44 |
        | Transition positions | 11.00% | 55.49 |
        
        Performance 별 편차도 굉장히 크다 (token accuracy 기)
        
        | 하위 10% | 22.04% |
        | --- | --- |
        | 중앙값 | 56.28% |
        | 상위 10% | 87.12% |
        
        모든 piece 에서 보편적인 pedaling rule을 학습하지 못한 것으로 보인다. 
        
    
    ### 왜 0 또는 127만 예측하게 되는 걸까?
    
    정답이 intermediate일 때 posterior mass의 평균: 
    
    P(1~126) = 0.3867 / P(0) = 0.2807 / P(127) = 0.3326
    
    intermediate 영역 전체의 확률은 평균적으로 0과 127 각각보다 높았지만, 그 확률이 수많은 class에 분산되어 각 class는 대략 0.0116 밖에 못 받아 argmax의 선택을 받을 수가 없다. 
    
    > 현재 encoder-only 모델은 intermediate 여부에 관한 정보는 학습했다. 다만 그 확률이 126개 intermediate class에 넓게 흩어져 있어서, 단일 class를 고르는 argmax에서는 0이나 127만 승리하게 된다.
    > 
    - 시도해본 해결책들
        1. Posterior median
        
        |  | Argmax | Posterior median |
        | --- | --- | --- |
        | Exact accuracy | 57.10% | 44.93% |
        | Overall MAE | 30.87 | **28.20** |
        | Intermediate MAE | 51.52 | **36.25** |
        | Predicted intermediate | 0% | **33.45%** |
        | Transition F1 | 11.61% | **30.51%** |
        
        argmax보다 문제의 연속적 성격에 더 적합한 decoding이 존재할 수 있다. 
        
        1. two-level prediction
        
        앞서 validation posterior 분석에서, intermediate 영역 전체에 의미 있는 확률 질량이 존재하는 것으로 보인다. 따라서 재학습은 시키지 않고, 두 단계로 나눠서 예측해보자. 
        
        → 0 / intermediate / 127 영역을 먼저 결정
        →  intermediate가 선택됐을 때에만 세부 깊이(depth)를 추정해보자
        
        ⇒ 잘 안됨. 생각보다 intermediate 영역 내에 의미있는 확률 분포가 있나 싶었는데 아닌 것 같다. 즉 intermediate 영역의 존재는 분명 인식하지만, 그 안에서 정확한 depth 순서를 학습하지 못한다. 
        
    
    - 두 번째 학습
        
        > 페달값을 굳이 128개의 class로 예측할 필요가 없다!
        > 
        - 6개 class
        
        | Class | CC64 | 전체 | Pedal1 | Pedal2 | Pedal3 | Pedal4 |
        | --- | --- | --- | --- | --- | --- | --- |
        | ZERO | 0 | 24.8916% | 24.9787% | 25.0539% | 24.7735% | 24.7604% |
        | LOW | 1~31 | 3.0237% | 3.0273% | 3.0646% | 3.0066% | 2.9962% |
        | MID_LOW | 32~63 | 11.6892% | 11.7539% | 11.9845% | 11.5713% | 11.4470% |
        | MID_HIGH | 64~95 | 16.8505% | 17.0264% | 16.8413% | 16.7755% | 16.7587% |
        | HIGH | 96~126 | 7.2762% | 7.2990% | 7.1390% | 7.3034% | 7.3633% |
        | FULL | 127 | 36.2689% | 35.9147% | 35.9168% | 36.5697% | 36.6743% |
        
        대표값 : 0, 22, 53, 77, 108, 127 (median)
        
        → 1~63 구간은 CC64에서 off로 처리되기 때문에 사실상 구분이 큰 의미가 없다. LOW랑 MID_LOW는 합치자. 
        
        → zero (0) 은 ‘페달을 아예 밟고 있지 않는 상태’라는 점에서 의미가 있다. 
        
        - 5개 class (최종 결정)
        
        | Class ID | 이름 | CC64 범위 | 대표값(구간 평균값) |
        | --- | --- | --- | --- |
        | 0 | ZERO | 0 | 0 |
        | 1 | LOW | 1–63 | 32 |
        | 2 | MID | 64–95 | 80 |
        | 3 | HIGH | 96–126 | 111 |
        | 4 | FULL | 127 | 127 |
        
        - 결과
        
        Best epoch: 3
        Early stopping: epoch 7
        Best validation CE: 1.11561
        
        | Metric | 기존 posterior median | 새 5-class |
        | --- | --- | --- |
        | 5-class token accuracy | 0.5156 | **0.5800** |
        | Exact-note accuracy | 0.4285 | **0.5054** |
        | Weighted F1 | 0.5160 | **0.5201** |
        | Macro F1 | **0.3714** | 0.3416 |
        
        | Class | Target | Prediction |
        | --- | --- | --- |
        | ZERO | 25.04% | **32.10%** |
        | LOW | 12.14% | **3.55%** |
        | MID | 12.17% | **7.15%** |
        | HIGH | 6.98% | **0.028%** |
        | FULL | 43.66% | **57.16%** |
        
        Low는 대부분 ZERO로 흡수됐다. 
        
        LOW → ZERO: 58.38%
        LOW → MID:   9.94%
        LOW → FULL: 23.20%
        LOW → LOW:   8.47%
        
        HIGH는 대부분 FULL에 흡수됐다. 
        
        HIGH → FULL: 70.3%
        HIGH → ZERO: 15.7%
        HIGH → MID: 11.1%
        HIGH → HIGH: 0.067%
        
        | Metric | 결과 |
        | --- | --- |
        | OFF/ON accuracy | 0.7908 |
        | Balanced accuracy | 0.7719 |
        | OFF F1 | 0.7128 |
        | ON F1 | 0.8355 |
        | ON recall | 0.8457 |
        
        OFF = ZERO + LOW
        ON  = MID + HIGH + FULL
        
        로 보고 평가하면 Off, On 상태 자체는 꽤 잘 예측했다. 
        
        | 허용 범위 | Transition F1 | Recall |
        | --- | --- | --- |
        | 정확히 같은 sample | 0.1007–0.1271 | 약 0.128–0.162 |
        | ±1 sample | 0.2137 | 0.2716 |
        | ±2 samples | 0.2740 | 0.3483 |
        | ±4 samples | 0.3396 | 0.4317 |
        
        정확한 transition 위치는 아직 정확하지는 않다. 
        
    - 세 번째 학습
        
        > 두번째 학습 컨셉으로 가되, weight 을 줘서 학습을 시켜보자.
        > 
        
        Train split에서 각 class c의 비율을 $f_
        c$ 라고 할 때, 초기 class weight를 다음과 같이 정의한다.
        
        → $\tilde{w}_c = \frac{1}{\sqrt{f_c}}$
        
        평균 weight가 1이 되도록 정규화한다.
        
        → $w_c = \frac{\tilde{w}_c}{\sum_j f_j \tilde{w}_j}$
        
        따라서 최종 class weight는 다음 조건을 만족한다.
        
        → $\sum_c f_c w_c = 1$
        
        ⇒ CE만 weighted로 바뀌고, 나머지 요소들은 두번째 실험과 완전히 동일. 
        
        - 결과
        
        사용한 weight = [0.926, 1.204, 1.125, 1.712, 0.767]
        
        | 지표 | Unweighted | Corrected weighted | 변화 |
        | --- | --- | --- | --- |
        | Token accuracy | 0.5800 | 0.5708 | ↓ 0.0092 |
        | Exact-note acc. | 0.5054 | 0.4859 | ↓ 0.0195 |
        | **Macro F1** | **0.3416** | **0.3634** | **↑ 0.0218** |
        | ZERO recall | 0.6727 | 0.6427 | ↓ |
        | **LOW recall** | **0.0847** | **0.1450** | **↑** |
        | **MID recall** | **0.1870** | **0.2215** | **↑** |
        | **HIGH recall** | **0.00067** | **0.0233** | **크게 ↑** |
        | FULL recall | 0.8668 | 0.8330 | ↓ |
        | Overall MAE | 28.7647 | **28.6056** | 소폭 개선 |
        | OFF/ON accuracy | 0.7908 | 0.7893 | 거의 동일 |
        | Binary transition F1 | 0.1271 | **0.1294** | 거의 동일 |
        | Short repedal F1 | 0.0434 | **0.0446** | 거의 동일 |
        
        | Class | Target | Unweighted pred. | Weighted pred. |
        | --- | --- | --- | --- |
        | ZERO | 25.04% | 32.10% | 30.27% |
        | LOW | 12.14% | 3.55% | **6.46%** |
        | MID | 12.17% | 7.15% | **8.99%** |
        | HIGH | 6.98% | 0.028% | **0.986%** |
        | FULL | 43.66% | 57.16% | **53.29%** |
        
        → trade-off 가 나타난다. 빈도가 낮은 class를 더 적극적으로 선택하면서 macro F1은 좋아졌지만, majority class에서 얻던 쉬운 accuracy를 일부 포기하게 된다. 
        
        → 하지만 HIGH는 여전히 거의 선택되지 않았다. 
        
        → weighting 이 trasition 문제에는 별 영향을 주지 못하는 것 같다. 
        
    
    ### 실험 2 - 기존 토크나이저 + Stage 2 분리 (encoder-decoder)
    
    - 기본 아키텍쳐
        
        PT 에서 사용한 10-layer encoder + lightweight 2-layer decoder 구조 그대로 활용하자  (hidden size 768, FFN 3072, attention head dimension 128)
        
        |  | Original Pianist Transformer | 우리 Stage 2 Encoder-Decoder |
        | --- | --- | --- |
        | 목적 | **Score → 전체 expressive performance** | **Pedal-free performance → pedal만 생성** |
        | Encoder | 10-layer PT encoder | **같은 PT encoder** |
        | Decoder | 2-layer causal decoder | **같은 2-layer causal decoder** |
        | Encoder 입력 | Score MIDI | Pitch/IOI/Velocity/Duration이 이미 결정된 performance |
        | Decoder 출력 | Pitch, IOI, Velocity, Duration, Pedal1–4 | **Pedal1–4만** |
        | 출력 vocabulary | PT unified vocab 5389 | **5 pedal classes + BOS/PAD** |
        | 출력 순서 | 8-token note sequence | **P1₁,P2₁,P3₁,P4₁,P1₂,...** |
        | Decoder weight | PT pretrained/SFT decoder | **새로 초기화한 decoder** |
        | Output head | 5389-class LM head | **shared Linear(768,5)** |
        | Pedal 간 dependency | 전체 performance autoregression 안에 포함 | **pedal끼리 직접 autoregressive dependency** |
        
        (BOS : Beginning Of Sequence / PAD : Padding token - 길이가 다른 sequence들을 batch 안에서 같은 길이로 맞추기 위한 빈칸 → Output head에는 포함 x  Decoder 입력에만 사용된다)
        
    - 첫 번째 학습
        
        이전 방식과 같이 class weighting, window/stride 등을 유지한다. 
        
        Best epoch : **4**, validation weighted CE : **0.27913**, Early stopping : epoch 8
        
        | Metric | Encoder-only weighted | Enc-Dec Teacher Forced | Enc-Dec Free Running |
        | --- | --- | --- | --- |
        | Token accuracy | 57.08% | **93.66%** | **49.34%** |
        | Macro F1 | 0.363 | **0.905** | **0.261** |
        | Overall MAE | 28.61 | **5.87** | **40.50** |
        | OFF/ON accuracy | 78.93% | **97.29%** | **69.30%** |
        | Transition F1 | 0.129 | **0.175** | **0.0039** |
        | Repedal F1 | 0.0446 | 0.0136 | **0.0000** |
        | Nonendpoint→endpoint collapse | 71.18% | **6.62%** | **89.36%** |
        
        Teacher forcing 상황 (이전 정답 pedal token을 decoder가 계속 공급받는 조건)에서는 압도적으로 높은 성능을 보이지만, 실제 free running으로 전환하면 성능이 크게 안 좋아진다.
        
        → 한번 잘못된 상태로 들어가면 잘못된 prediction이 다음 입력으로 들어가 잘못된 trajectory가 고착된다. 
        
        → 특히 free running 에서는 Zero / Full 모델로 돌아가 버렸다. (class에 weight을 줬음에도 불구하고)
        
        |  | Target distribution | PT prediction |
        | --- | --- | --- |
        | ZERO | 25.04% | 32.32% |
        | LOW | 12.14% |  4.45% |
        | MID | 12.17% | 2.35% |
        | HIGH |  6.98% | 0.00% |
        | FULL | 43.66% | 60.89% |
    - 두 번째 학습
        
        한번 early stopping 없이, epoch 20까지 full로 돌려보았다. 
        
        → epoch 4 부터 overfitting 시작, epoch 20 까지 지속적으로 overfitting. 
        
        → 혹시 validation loss가 잠깐 올라갔다가 다시 내려오지는 않을까 싶었는데, 역시나 그러지 않았다. 
        
        → no early stopping 으로 학습된 모델 연주를 들어보니 페달을 계속 아예 안 밟거나 계속 밟는 식의 극단적인 표현이 나타났다. 
        
    - 세 번째 학습
        
        > 첫 번째 학습에서, teacher forcing일 때와 실제 free-running inference일 때 성능 차이가 너무 심하므로 학습 할 때 항상 정답 이전 토큰만 보여주지 말고, 모델이 스스로 예측한 이전 토큰도 보여주면서 inference 상황에 적응시켜 보자. (Scheduled Sampling)
        > 
        
        Schedule :
        
        | Epoch | 정답 history 사용 확률 |
        | --- | --- |
        | 1 | 100% |
        | 2 | 100% |
        | 3 | 90% |
        | 4 | 80% |
        | 5 | 70% |
        | 6 이후 | 60% |
        
        (Epoch 3 기준)
        
        | Metric | 기존 Enc-Dec | Scheduled Sampling |
        | --- | --- | --- |
        | Free-running accuracy | **0.4934** | 0.4804 |
        | Free-running macro F1 | **0.2614** | 0.2437 |
        | Overall MAE | **40.50** | 43.97 |
        | ON-region MAE | **31.71** | 44.85 |
        | OFF/ON accuracy | **0.6930** | 0.6643 |
        | Transition F1 | **0.00387** | 0.00244 |
        | Repedal F1 | 0 | 0 |
        | TF ↔ free-run accuracy gap | **0.4432** | 0.4558 |
        
        | ZERO    | 49.15% |
        | --- | --- |
        | LOW      | 0.00% |
        | MID      | 2.47% |
        | HIGH     | 0.20% |
        | FULL    | 48.18% |
        
        → HIGH  비율은 조금 늘었는데, 그 대신 LOW가 완전히 사라졌다. 
        
        → 학습 중 자기 prediction을 history로 볼 때, 초기 prediction 자체가 endpoint-biased라면 그 잘못된 history가 다시 decoder input으로 들어가면서 ZERO 나 FULL을 지속적으로 생성하는 안전한 trajectory를 강화하는 것으로 보인다. 
        
    
    ### 실험 3 - 기존 토크나이저 + Stage 2 분리 (2 class 예측)
    
    > 길을 좀 잃었다. 지금 통일된 metric이 없어서 내가 성능을 개선시키고 있는건지 감이 안 온다. 그리고 너무 많은 것을 한번에 이루려고 하지 말고 일단 ‘성능 개선’부터 확실하게 해보자.
    > 
    
    → Metric은 (불완전 하긴 하지만) PT 논문에서 사용한 평가 metric을 사용해서 확실하게 성능을 개선시킨 모델을 찾아내보자. 
    
    → 페달 depth을 예측하도록 하는 것은 잠시 뒤로 미뤄두자. 
    
    ⇒ 페달 토큰 예측은 0 / 127 즉 2 class로 아예 확실하게 정해버리자
    
    ⇒ 데이터셋은 MAESTRO + ASAP 로 확장하자
    (MAESTRO: 1,174 performance + ASAP: 892 performance / validation, test set 은 앞선 실험과 동일)
    
    - PT 논문에서 사용하는 metric
        
        > 인간 performance와 생성 performance의 전역적인 token distribution을 비교한다.
        > 
        
        dimension [Pedal 1, Pedal 2, Pedal 3, Pedal 4] 에 대해 JS divergence 와 Intersection Area 을 계산한다. 
        
        (각 token 은 0 또는 1 의 값을 가지므로 총 16개의 class가 생긴다)
        
        → 특정 note가 인간의 특정 note와 얼마나 유사한지 계산하는 방식이 아니다 (국소적인 평가는 불가능하다)
        
        → 모델이 test set 전체에서 만들어내는 expressive behavior의 통계적 분포가 인간 연주의 통계적 분포와 얼마나 비슷한가를 평가하는 지표이다. 
        
    - Baseline PT Inference
        
        공식 PT 설정에 최대한 맞춰서 test set와 validation set에 대해 PT inference 파일을 만들어두고, 앞으로 metric 검사할 때는 이 파일들을 stage 1 의 출력으로 고정한다. 
        
        ```
        CPU
        bfloat16
        temperature = 1.0
        top-p = 0.95
        max context = 4096
        overlap = 0.5
        
        seed = 42 (내가 정한 값)
        
        ```
        
        Stage 2의 결과에서도 non-pedal token은 Stage 1 의 출력으로 확실하게 고정함으로서 오직 페달에 의한 변화만을 평가할 수 있도록 하자. 
        
    - 첫 번째 실험
        
        > Stage 2 는 encoder-only로 가보자.
        > 
        
        그리고 2 class로 4개의 token을 예측하는 모델(Independent 4×2)과, 아예 16개의 class 중에 하나를 예측하는 모델(Joint 16)로 나누어서 둘 다 학습시켜보자. 
        
        - Validation set
        
        | Model | Strict JS ↓ | Intersection ↑ |
        | --- | --- | --- |
        | Original PT | 0.161499 | 0.821026 |
        | **Independent 4×2 (epoch 2)** | **0.150083** | **0.829325** |
        | Joint 16 (epoch 2) | 0.202442 | 0.780598 |
        - Test set
        
        | Model | Validation JS ↓ | Test JS ↓ |
        | --- | --- | --- |
        | Original PT | 0.161499 | **0.14643** |
        | Independent 4×2 | **0.150083** | 0.18560 |
        | Joint 16 | 0.202442 | 0.25012 |
        
        → Joint 16 모델은 폐기
        
    - 두 번째 실험
        
        > Stage 2 를 encoder-decoder 구조로 만들고, Independent 4×2 모델에 대해서 학습 시켜 보자.
        > 
        
        | Model | JS ↓ | Intersection ↑ |
        | --- | --- | --- |
        | Original PT | 0.16150 | 0.82103 |
        | **Encoder-only Independent** | **0.15008** | **0.82933** |
        | Encoder-Decoder | 0.24249 | 0.74022 |
    
    Q. 항상 Best Epoch가 5 미만인데, 데이터셋을 확장해도 마찬가지이다. 정상적인 걸까? 
    
    → 꼭 집착 안 해도 괜찮다. 
    
    - 피드백
        
        decoder-only 모델도 한번 사용해보자. 어차피 stage 1 에서 만들어진 representation 이 prefix 로 들어간다면 전체적인 문맥을 충분하게 참조할 수 있다. 
        
        → decoder  만 사용했을 때의 장점 (기억이 잘 안 나는데) 이 뭐가 있을지 생각해보자
        
        classification이 아니라 regression 으로 예측해봐도 좋겠다.
        
        encoder-only 모델에서 학습을 시킬 때 masking prediction 방향으로 학습을 시켜보자. (bert 처럼)
        
        Loss도 CE 사용하는게 아니라, 주변 값들을 부드럽게 고려하는 (??) Loss를 한번 찾아서 사용해보자 
        → 논문 : Regress, Don’t Guess A Regression-like Loss on Number Tokens for Language Models
        
        Metric 도 사실 PT 가 0 / 127로 예측한다고 굳이 그거에 집착하지 말고, 더 높은 해상도의 출력을 고려하는 metric을 만들어서 PT 가 점수가 낮아도 그만큼 개선이 된거다 !! 라고 주장하는게 더 나을 수 있다. 
        → 인간 연주와 더 높은 pedal 해상도에 대해 평가하는 metric 간단하게 만들어서 그걸로 평가해도 괜찮겠다. 
        
        페달 토큰 뽑을 때 argmax 말고 top-k sampling 이나 temperature 잘 조절하면 생각보다 그냥 더 괜찮아 질수도 있다. (논문에서 사용하고 있는 방식도 참고해볼 필요 있다. top-k 방식은 사용하는 것 같던데)