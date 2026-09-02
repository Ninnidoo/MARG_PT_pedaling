# Stage 2 Binary Validation Diagnosis v0

## Outcome

고정 ASAP validation 19 pieces / 71 human performances와 기존 Stage 1 cache만 사용해 Human, Original PT, independent 4×2 best epoch 2, joint 16 best epoch 3을 비교했다. 두 Stage 2 checkpoint는 deterministic argmax로 decode했으며 sampling, calibration, threshold 변경, Stage 1 regeneration은 수행하지 않았다. ASAP test MIDI access는 **0**이다.

Global primary metric은 training report와 허용오차 `1e-12` 안에서 재현됐다. 따라서 **independent 4×2 best epoch 2를 final primary candidate**, **joint 16 best epoch 3을 fixed comparison model**로 lock한다.

## Metric provenance note

Human reference는 existing validation int16 cache를 사용하며 기존 histogram과 exact 일치한다. Original PT baseline은 기존 evaluation과 동일하게 cached `original_pt.mid` 19개를 official tokenizer로 재-tokenization한다. Pre-render `generated_ids_int64.npy`를 직접 집계하면 note 수는 같지만 MIDI mapping/round-trip의 pedal sampling 때문에 histogram이 달라지므로 baseline source로 사용하지 않았다. Stage 2 후보는 training checkpoint selection과 동일하게 cached Stage 1 IDs의 Pedal1–4만 교체한 결과를 직접 집계한다. 모든 Stage 2 piece에서 non-pedal token exact equality를 assert했다.

## Global metric reproduction

| Candidate | notes | JS distance ↓ | Intersection ↑ | expected JS | expected Intersection | result |
|---|---:|---:|---:|---:|---:|---|
| Original PT | 48,894 | 0.144081839387 | 0.907531643144 | 0.144081839387 | 0.907531643144 | PASS |
| independent 4×2 | 48,894 | 0.069918887331 | 0.965298332072 | 0.069918887331 | 0.965298332072 | PASS |
| joint 16 | 48,894 | 0.168750230719 | 0.919373586922 | 0.168750230719 | 0.919373586922 | PASS |

Independent와 joint inference는 각각 182 windows와 182 windows를 처리했다. Stage 1 neural inference 재실행은 0이다.

## Global 16-pattern comparison

Pattern order는 `0000, 0001, ..., 1111`이다. Probability는 official PT metric의 epsilon `1e-10` 추가 후 renormalization semantics를 따른다.

| pattern | Human | Original PT | independent 4×2 | joint 16 |
|---:|---:|---:|---:|---:|
| `0000` | 0.336807218 | 0.429275575 | 0.362682537 | 0.417433631 |
| `0001` | 0.007991463 | 0.001595288 | 0.006667485 | 0.000347691 |
| `0010` | 0.000200755 | 0.000122715 | 0.002311122 | 0.000000000 |
| `0011` | 0.011918515 | 0.003190576 | 0.009367203 | 0.001779360 |
| `0100` | 0.000376856 | 0.000122715 | 0.002290670 | 0.000122715 |
| `0101` | 0.000024654 | 0.000020453 | 0.000409048 | 0.000000000 |
| `0110` | 0.000824153 | 0.000020453 | 0.002024788 | 0.000061357 |
| `0111` | 0.014781917 | 0.002433837 | 0.009796703 | 0.004090482 |
| `1000` | 0.010009580 | 0.001493026 | 0.008508202 | 0.000593120 |
| `1001` | 0.003194472 | 0.000552215 | 0.002229312 | 0.000081810 |
| `1010` | 0.000049308 | 0.000000000 | 0.001513478 | 0.000000000 |
| `1011` | 0.005589445 | 0.000449953 | 0.007342414 | 0.001370311 |
| `1100` | 0.009960272 | 0.002761075 | 0.003865505 | 0.000081810 |
| `1101` | 0.001535601 | 0.001288502 | 0.001411216 | 0.000000000 |
| `1110` | 0.010566059 | 0.002290670 | 0.005767579 | 0.000102262 |
| `1111` | 0.586169732 | 0.554382950 | 0.573812737 | 0.573935451 |

Largest pattern errors versus Human:

- Original PT: `0000` +0.092468, `1111` -0.031787, `0111` -0.012348, `0011` -0.008728
- independent 4×2: `0000` +0.025875, `1111` -0.012357, `1100` -0.006095, `0111` -0.004985
- joint 16: `0000` +0.080626, `1111` -0.012234, `0111` -0.010691, `1110` -0.010464

Signed/absolute error 전체 값은 `global_joint16_comparison.csv`에 있다.

## Steady and transition-containing mass

Steady mass는 `P(0000)+P(1111)`, transition-containing mass는 나머지 14개 binary pattern의 합이다.

| Candidate | steady mass | transition-containing mass | transition error vs Human |
|---|---:|---:|---:|
| Human ASAP validation | 0.922976951 | 0.077023049 | +0.000000000 |
| Original PT | 0.983658525 | 0.016341475 | -0.060681575 |
| independent 4×2 | 0.936495274 | 0.063504726 | -0.013518324 |
| joint 16 | 0.991369083 | 0.008630917 | -0.068392132 |

Original PT는 Human보다 transition-containing binary patterns를 0.060681575 적게 생성했다. Independent 4×2는 이 부족분을 0.013518324까지 줄여 Human 쪽으로 이동한 반면, joint 16의 부족분은 0.068392132로 Original PT보다 더 커졌다. 이는 음악적 pedal quality 자체가 아니라 **PT의 네 within-IOI sampling point 사이에서 binary state가 달라지는 pattern mass**에 대한 기술 통계다.

## Slot-wise OFF/ON marginals

| slot | Human ON | Original PT ON | independent ON | joint ON |
|---|---:|---:|---:|---:|
| Pedal1 | 0.627074469 | 0.563218391 | 0.604450444 | 0.576164764 |
| Pedal2 | 0.624239244 | 0.563320653 | 0.599378247 | 0.578394077 |
| Pedal3 | 0.630099884 | 0.562891152 | 0.611936025 | 0.581339223 |
| Pedal4 | 0.631205798 | 0.563913773 | 0.611036119 | 0.581605105 |

OFF/ON signed 및 absolute error는 `slotwise_binary_comparison.csv`에 있다. 이 marginal audit은 16-pattern 결과를 설명하는 supplemental diagnostic이며 selection criterion이 아니다.

## Piece-level sanity analysis

각 piece의 모든 Human validation performances를 aggregate하고, 동일 piece의 한 Original PT/Stage 2 candidate와 비교했다. Global selection은 변경하지 않는다.

| Candidate | improved vs Original PT | mean piece JS | median piece JS |
|---|---:|---:|---:|
| independent 4×2 | 8/19 | 0.245127624 | 0.223999068 |
| joint 16 | 4/19 | 0.280744741 | 0.291340467 |
| Original PT reference | — | 0.233316500 | 0.186246750 |

Best/worst JS improvement examples (`Original PT JS - model JS`):

- independent 4×2 best: Haydn Keyboard_Sonatas_32-1 +0.224858; Beethoven Piano_Sonatas_27-1 +0.218066; Schumann Kreisleriana_6 +0.063551
- independent 4×2 worst: Bach Prelude_bwv_856 -0.162145; Scriabin Etudes_op_8_11 -0.167871; Ravel Pavane -0.204962
- joint 16 best: Haydn Keyboard_Sonatas_32-1 +0.135933; Beethoven Piano_Sonatas_27-1 +0.082634; Haydn Keyboard_Sonatas_39-2 +0.066994
- joint 16 worst: Scriabin Etudes_op_8_11 -0.194315; Schubert Piano_Sonatas_664-1 -0.202121; Ravel Pavane -0.225850

전체 19-piece JS/Intersection과 improvement는 `piece_level_metrics.csv`에 있다.

## Final selection and lock

- Primary: **independent 4×2 best epoch 2**, checkpoint SHA-256 `4262081a15aa20aac607737e14e4e6b1c8e2727852ac370166f28fbba51d0833`
- Fixed comparison: **joint 16 best epoch 3**, checkpoint SHA-256 `4c3b70f09456cab3c6c8889273982837382229d18c79465465b8b42ba6c24e13`
- Primary rule: global validation Pedal JS distance lower; numerical tie이면 Intersection higher
- Binary threshold: 64; decoding: deterministic argmax
- No test-time calibration, sampling, temperature change, or post-lock hyperparameter change
- Machine-readable lock: `analysis/stage2_binary_v0/final_lock_v0/final_experiment_lock.json`

## Integrity

- Global metrics reproduce training/baseline reports: PASS
- Human and Original PT saved global histograms exact: PASS
- Stage 2 non-pedal token exact equality for all 19 pieces: PASS
- Best checkpoint epochs and checkpoint metadata: PASS
- Deterministic argmax, threshold 64, fixed 512/256 inference: PASS
- Stage 1 inference regeneration: 0
- ASAP test MIDI access: **0 / PASS**

No training, checkpoint update, calibration, sampling experiment, threshold change, Stage 1 regeneration, test evaluation, or audio rendering was performed.
