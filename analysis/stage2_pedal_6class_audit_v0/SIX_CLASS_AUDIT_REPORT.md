# Stage 2 Endpoint-aware 6-class Pedal Audit v0

## 1. 목적과 가설

flat 128-class Pedal1–4 target을 고정된 endpoint-aware 6-class로 바꿀 때의 class imbalance, 값 정보 손실, transition 보존성을 모델 학습 전에 측정했다. 가설은 ZERO/FULL을 독립 endpoint로 유지하고 네 intermediate 구간을 쓰면 binary sustain/repedal 구조를 보존하면서 128-class 분류보다 통제 가능한 target을 제공한다는 것이다. 이 문서의 oracle은 **모델 성능이 아니라 표현 자체의 정보 손실**이다.

## 2. 데이터 및 split

- ASAP v1.1 root: `/workspace/public/ASAP/asap-dataset-v1.1`
- CE baseline split CSV: `/workspace/project/analysis/stage2_encoder_only_v0/asap_split.csv` (SHA-256 `d1fe379eb123ff7abca93773296f735f40a215dcd6e6363b55756383708c965b`)
- 사용 범위: train 행만, 892 performances / 180 pieces
- normalized notes: 2,988,614; Pedal1–4 targets: 11,954,456
- tokenizer/cache: 기존 `Stage2PedalDataset(cache_mode='preload')`와 pinned `midi_to_ids`; validation/test MIDI 접근 없음
- run_id: `20260804T060246Z-11d3c2f6c93d`; PID: 552; 모델 학습 없음

## 3. Raw CC64 분포

non-drum instrument의 원본 CC64 event 3,848,103개를 셌다. event 기준 0=142,897 (3.713440%), 127=141,495 (3.677007%), 1–126=3,563,711 (92.609553%)이다. 6-class event count/ratio와 time-occupancy(초) 분포는 `class_distribution_overall.csv`, exact 0–127 histogram은 `exact_value_histogram.csv`에 있다. Occupancy는 time 0에서 value 0으로 시작하며 마지막 non-drum note end 또는 마지막 CC64 중 늦은 시점까지, 동시 event에서는 source order상 마지막 event가 이후 구간을 지배하도록 계산했다.

| Class | Raw event ratio | Raw occupancy ratio |
| --- | --- | --- |
| ZERO | 0.037134 | 0.223673 |
| LOW | 0.048222 | 0.030579 |
| MID_LOW | 0.279063 | 0.118775 |
| MID_HIGH | 0.457092 | 0.178074 |
| HIGH | 0.141718 | 0.086810 |
| FULL | 0.036770 | 0.362090 |

## 4. Tokenized Pedal1–4 분포

| Class | Overall | Pedal1 | Pedal2 | Pedal3 | Pedal4 |
| --- | --- | --- | --- | --- | --- |
| ZERO | 0.248916 | 0.249787 | 0.250539 | 0.247735 | 0.247604 |
| LOW | 0.030237 | 0.030273 | 0.030646 | 0.030066 | 0.029962 |
| MID_LOW | 0.116892 | 0.117539 | 0.119845 | 0.115713 | 0.114470 |
| MID_HIGH | 0.168505 | 0.170264 | 0.168413 | 0.167755 | 0.167587 |
| HIGH | 0.072762 | 0.072990 | 0.071390 | 0.073034 | 0.073633 |
| FULL | 0.362689 | 0.359147 | 0.359168 | 0.365697 | 0.366743 |

Performance별 비율 분포(각 performance를 동일 가중):

| Class | mean | median | std | q05 | q25 | q75 | q95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ZERO | 0.296653 | 0.194438 | 0.284499 | 0.010622 | 0.068441 | 0.449060 | 0.942060 |
| LOW | 0.037246 | 0.012874 | 0.078500 | 0.000000 | 0.003894 | 0.037226 | 0.147376 |
| MID_LOW | 0.128221 | 0.090390 | 0.137714 | 0.007769 | 0.047468 | 0.156811 | 0.372408 |
| MID_HIGH | 0.160054 | 0.124025 | 0.130618 | 0.005831 | 0.069707 | 0.217165 | 0.434727 |
| HIGH | 0.069248 | 0.037581 | 0.082461 | 0.000000 | 0.014925 | 0.092750 | 0.237545 |
| FULL | 0.308578 | 0.267430 | 0.257683 | 0.000000 | 0.065282 | 0.521864 | 0.763212 |

## 5. 6-class class imbalance

가장 드문 class는 LOW (361,465, 3.023684%), 가장 많은 class는 FULL (4,335,746, 36.268869%)이다. max/min count ratio는 11.995다. 모든 class는 train에 존재한다: True.

## 6. Train-derived representative values

ZERO=0, FULL=127로 고정했고, interior representative는 모든 train Pedal1–4 target을 합친 bin별 global median이다. 모든 slot이 동일한 global representative를 사용한다. Slot median은 비교용이며 선택에 사용하지 않았다.

| Class | Global representative | Pedal1 median | Pedal2 median | Pedal3 median | Pedal4 median |
| --- | --- | --- | --- | --- | --- |
| ZERO | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| LOW | 22.000000 | 22.000000 | 22.000000 | 22.000000 | 22.000000 |
| MID_LOW | 53.000000 | 53.000000 | 53.000000 | 53.000000 | 52.000000 |
| MID_HIGH | 77.000000 | 77.000000 | 77.000000 | 77.000000 | 78.000000 |
| HIGH | 108.000000 | 108.000000 | 108.000000 | 108.000000 | 108.000000 |
| FULL | 127.000000 | 127.000000 | 127.000000 | 127.000000 | 127.000000 |

Bin 내부 mean/median/std/min/max/unique count는 `bin_statistics.csv`에 있다.

## 7. Quantization oracle error

- overall micro MAE: 3.020231
- intermediate-only (원래 값 1–126) MAE: 7.776182
- performance-macro MAE: 3.087626 (median 2.884687)
- tolerance accuracy: ±5 75.489165%, ±10 89.141145%, ±20 99.790078%
- maximum absolute error: 21.000000
- per-slot MAE: Pedal1=3.037227, Pedal2=3.032851, Pedal3=3.007105, Pedal4=3.003741

Class별 within-bin MAE와 error quantile 전체는 `quantization_oracle_metrics.json`에 있다.

## 8. Transition preservation

Sequence는 performance별로 `(note 1 Pedal1→4, note 2 Pedal1→4, ...)` 순서로 flatten했고 performance 경계를 연결하지 않았다. Original transition은 인접 exact value가 다른 경우, class transition은 인접 6-class ID가 다른 경우다.

- original nonzero value transitions: 1,979,033
- quantized/preserved class transitions: 968,033
- preservation ratio: 48.914445%
- removed: 1,011,000; newly introduced: 0
- binary `<64`/`>=64` transitions: 401,769; preservation: 100.000000%
- preserved transition direction agreement: 100.000000%
- performance-macro preservation: mean 51.909388%, median 51.303404%
- primary short repedal-like (`>=64→<64→>=64`, low run ≤4 samples): original 74,248, quantized 74,248

Class run-length와 1/2/4/8-sample repedal 민감도는 `transition_preservation_metrics.json`에 있다.

## 9. Slot별 차이

Slot별 class ratio 차이는 위 표와 `class_distribution_per_slot.csv`에, slot별 median은 대표값 표에, slot별 oracle MAE는 Section 7과 JSON에 기록했다. 대표값은 slot별 차이에 맞춰 조정하지 않았다.

## 10. Raw와 tokenized 비교 및 위험 요소

| Class | Raw event ratio | Token ratio | Token - raw (pp as ratio) | Token/raw |
| --- | --- | --- | --- | --- |
| ZERO | 0.037134 | 0.248916 | 0.211782 | 6.703117 |
| LOW | 0.048222 | 0.030237 | -0.017985 | 0.627035 |
| MID_LOW | 0.279063 | 0.116892 | -0.162172 | 0.418872 |
| MID_HIGH | 0.457092 | 0.168505 | -0.288588 | 0.368645 |
| HIGH | 0.141718 | 0.072762 | -0.068956 | 0.513427 |
| FULL | 0.036770 | 0.362689 | 0.325919 | 9.863694 |

- Token sampling은 raw **event frequency**가 아니라 note-IOI 내부 네 지점의 held state를 표본화하므로 event ratio와 직접 같은 모집단이 아니다. 위 차이는 sampling/시간 점유 효과를 함께 반영한다.
- 6-class 내부의 모든 exact 변화는 제거되므로 exact-value transition 보존율은 48.914445%에 그친다. 반면 `<64`/`>=64` 경계는 class 경계와 일치해 100% 보존된다.
- 가장 드문 LOW class의 imbalance는 controlled training에서 class별 recall, macro-F1, confusion을 반드시 별도로 보게 만든다.
- 마지막 note의 Pedal2–4는 원 tokenizer의 고정 4,990-tick tail sampling을 그대로 포함한다. 이것은 새 audit이 만든 현상이 아니라 기존 target grammar의 특성이다.
- 짧은 repedal 통계의 “short”는 실제 millisecond가 아니라 flattened sample 개수다. IOI가 가변이므로 timing 품질의 대용치로 해석하면 안 된다.

## 11. Controlled training 진행 결론

**진행 가능** (representation audit 기준). 판정 조건은 모든 train class 존재, newly introduced transition=0, binary transition 100% 보존, overall oracle MAE≤16이었다. 이 결론은 모델이 해당 class를 잘 학습한다는 뜻이 아니다. 다음 실험은 이 고정 경계/대표값을 변경하지 않고 CE baseline과 동일 split·초기화·학습 budget을 사용하는 controlled comparison이어야 하며, validation은 모델 선택/평가에만 쓰고 대표값 재추정에는 쓰지 않아야 한다.
