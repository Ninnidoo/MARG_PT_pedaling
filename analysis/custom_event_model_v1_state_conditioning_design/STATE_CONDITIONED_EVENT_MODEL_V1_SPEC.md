# State-Conditioned Event Model v1 — Frozen Implementation Specification

## Frozen formulation

B3-S: slotwise_soft_predicted_prestate_conditioning

~~~text
masked PT tokens
      |
Pretrained PT encoder (unchanged)
      |
owned onset hidden H [B,O,768]
      |--------------------------------------> Main timing heads (unchanged)
      |
      +-> stopgrad(H)
             |
        six slot state heads Linear(768,4)
             |
        A [B,O,6,4] -> softmax -> q [B,O,6,4]
                                      |
                                 stopgrad(q)
                                      |
H expanded [B,O,6,768] ---------------+
                                      |
                            concat X [B,O,6,772]
                                      |
                       six Main Linear(772,5)
                                      |
                      event logits [B,O,6,5]
                                      |
                         Decoder v1 unchanged
~~~

Initial과 Terminal branch는 v0 그대로다.

## Tensor shapes

- B: physical micro-batch
- O: padded batch에서 maximum uniquely owned onsets
- K=6, event classes C=5, state classes S=4
- H: dropout-applied owned hidden, float[B,O,768]
- M: owned-onset validity, bool[B,O]
- Y: Main event target, int64[B,O,6]
- S_pre: derived semantic pre-state, int64[B,O,6]
- A_k=G_k(stopgrad(H)): state logits, stacked float[B,O,6,4]
- q=softmax(A, -1): predicted state posterior, float[B,O,6,4]
- State embedding: fixed E=I_4, e_state=q, dimension 4, zero learned parameters
- X=concat(H expanded over K, stopgrad(q)): float[B,O,6,772]
- Z_k=D_k(X[:,:,k,:]): stacked Main logits float[B,O,6,5]
- Main timing: float[B,O,6], unchanged

G_k와 D_k는 v0 Main head와 마찬가지로 slot별 독립 head다.

## Semantic target construction

Cache rebuild는 하지 않는다. Existing initial_state, main_interval_start_state, main_event_target만 사용한다.

~~~python
s_pre[i, 0] = main_interval_start_state[i]
for k in range(1, 6):
    previous = main_event_target[i, k - 1]
    s_pre[i, k] = (
        s_pre[i, k - 1] if previous == NONE
        else previous - 1
    )
~~~

Assertions:

1. 첫 interval start state는 initial_state와 같다.
2. 여섯 target slot rollout final은 다음 interval start state와 같다.
3. Frozen first-NONE grammar가 성립한다.
4. 모든 valid S_pre는 0..3이다.

Slot1은 첫 interval의 Initial 또는 이전 interval final state를 label로 한다. Slot2–6은 earlier target SET을 반영한다. NONE와 post-first-NONE slot은 state를 유지한다. Terminal semantic rollout은 final Main state에서 이어지지만 Terminal model/loss는 변경하지 않는다.

## Training forward

~~~python
hidden = encoder(batch.input_ids, ...)                # unchanged
H = dropout(gather_owned(hidden))                     # [B,O,768]

initial_logits = initial_head(...)
main_timing = stack(timing_k(H) for k in range(6))
terminal_outputs = terminal_heads(...)

A = stack(state_head_k(H.detach()) for k in range(6), dim=2)
q = softmax(A.float(), dim=-1).to(A.dtype)

H6 = H.unsqueeze(2).expand(-1, -1, 6, -1)
X = concat([H6, q.detach()], dim=-1)
Z = stack(event_head_k(X[:, :, k]) for k in range(6), dim=2)
~~~

S_pre는 dataset/collate에서 frozen target으로 derive하며 forward input으로 전달하지 않는다.

## Inference forward

Inference도 동일한 A, q, X, Z 식을 model.eval()에서 사용한다. GT state 접근, sampling, schedule, hard argmax state는 없다. q는 slot별로 재추정한다.

Owner windows는 v0와 동일하게 performance-level Main logits[I,6,5]와 timing[I,6]을 unique ownership으로 assemble한다. State logits는 diagnostic으로 저장할 수 있지만 Decoder에 전달하지 않는다.

Deterministic inference:

- categorical sampling 없음
- scheduled mixing 없음
- model 내부 hard-state rollout 없음
- 기존 seed/deterministic setting 유지
- Decoder argmax, first-NONE, tau clipping/sorting, same-time, same-state suppression, Terminal cumulative gaps unchanged

## State update versus Decoder state

두 state를 분리한다.

1. Semantic s_pre/q: exact target label과 그 model posterior. Main slot마다 독립 재추정.
2. Decoder current: frozen Decoder가 chronological sorting 및 same-time collapse 후 관리하는 hard predicted state.

Decoder current를 model head로 feedback하지 않는다. 이는 shuffled owner-window training을 보존하고 measured B1 source shift, B2 autoregressive dependency, predicted-tau circularity를 피하기 위한 frozen decision이다.

Decoder interface는 변하지 않는다.

~~~text
initial_logits [4]
main_event_logits [I,6,5]
main_timing_predictions [I,6]
terminal_event_logits [4,5]
terminal_timing_predictions [4]
~~~

## Loss

V는 valid owned onset positions, w_k는 frozen slot-specific inverse-sqrt weights다.

~~~text
L_event,k =
    sum_(b,o in V) w_k[Y_bok] CE(Z_bok, Y_bok)
    / sum_(b,o in V) w_k[Y_bok]

L_state,k =
    sum_(b,o in V) CE(A_bok, S_pre,bok)
    / |V|

L_main_event = (1/6) sum_k L_event,k
L_main_state = (1/6) sum_k L_state,k

L_opt =
    L_initial
  + L_main_event
  + L_main_state
  + L_main_timing
  + L_terminal_event
  + L_terminal_timing
~~~

기존 lambda는 모두 1.0을 유지한다. L_main_state coefficient도 구조적으로 1.0으로 고정하고 sweep 가능한 config로 노출하지 않는다.

Mask:

- Event CE: existing owned_onset_mask, all six positions
- State CE: same owned_onset_mask, all six semantic state positions
- Main timing: existing active-event mask
- Initial/Terminal: unchanged masks
- Padding/ignore: event/state loss 모두 제외
- Same-state target: mask하지 않음

Checkpoint selection은 legacy v0 validation objective를 exact 유지한다.

~~~text
L_select =
    L_initial
  + L_main_event
  + L_main_timing
  + L_terminal_event
  + L_terminal_timing
~~~

Auxiliary state CE는 epoch/slot별 log하지만 selection에서 제외한다. Initial-dominance 문제는 별도 follow-up이다.

## Gradient flow

- H→direct Main event path: enabled
- H→state head: stop-gradient
- state CE→state heads: enabled
- q→event head: value는 사용, gradient는 stop
- event CE→state heads: disabled
- state CE→encoder: disabled
- Initial/Timing/Terminal gradient: unchanged

GT state는 loss target으로만 나타나므로 teacher forcing은 없다. Stop-gradient는 auxiliary CE가 encoder를 별도로 바꾸거나 event CE가 q를 non-semantic latent code로 변형하는 것을 막는다.

## Parameters

Per slot:

- v0 event Linear(768,5): 3,845
- v1 event Linear(772,5): 3,865, delta +20
- v1 state Linear(768,4): 3,076

Six slots:

- event expansion +120
- state heads +18,456
- fixed identity embedding +0
- total added **18,576**
- expected total **103,339,216**

## Initialization

1. 모든 legacy module을 v0와 같은 seed-42 순서로 먼저 구성한다.
2. Expanded event head의 768-column block와 bias는 corresponding seed-42 v0 init을 copy한다.
3. 새 네 state-feature columns는 exact zero로 초기화한다.
4. State heads는 legacy modules 뒤에 deterministic seed-42 stream으로 fresh initialize한다.
5. Controlled full run은 trained v0 warm-start를 사용하지 않는다.

Step-zero Main event logits는 v0와 같고 state feature weight는 첫 update부터 학습 가능하다.

## Batch/window/ownership

- Window/stride 512/256 unchanged
- Each onset supervised exactly once
- Training shuffle, micro/effective batch unchanged
- main_interval_start_state와 S_pre는 owned_global_onset_indices로 select
- Window/batch 사이 state carry 없음
- Performance-sequential sampler 없음
- Initial loss는 first owner, Terminal은 last owner에서만 valid
- q branch는 valid owned Main positions에만 존재

## Explicit exclusions

- Hard current-destination logit mask 없음
- Same-state target 삭제 없음
- EVENT/NONE factorization 없음
- Class-weight change 없음
- Sampling/smoothing/threshold/calibration 없음
- Timing formulation change 없음

## Checkpoint compatibility

v0 checkpoint는 Main event weights [5,768]→[5,772] 및 새 state heads 때문에 strict-load incompatible다. Encoder, Initial, Main timing, Terminal event/timing tensor는 shape-compatible하다.

Controlled v1은 same pretrained PT encoder와 seed-42 fresh heads에서 시작한다. Smoke diagnostic에서 selective load는 가능하지만 full comparison은 trained v0 warm-start를 금지한다.

## Implementation-task acceptance

- Shapes/interfaces assertions 통과
- train/eval 모두 predicted q 사용
- GT state가 event forward input에 없음
- gradient-stop tests 통과
- frozen weight SHA unchanged
- Decoder/Tokenizer IDs unchanged
- Decoder-facing output shapes unchanged
- 다음 explicit task 전 training/ASAP/Repedal 0
