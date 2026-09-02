# MARG Research

Pianist Transformer(PT)를 encoder-only Stage 2 pedal refinement 모델로 사용해,
ASAP 연주 데이터에서 sustain pedal을 5개 고정 class로 예측하는 실험 프로젝트입니다.

## 현재 상태

동일한 모델·데이터 split·학습 설정으로 다음 두 Stage 2 실험을 완료했습니다.

- **Unweighted 5-class baseline**: 일반 cross-entropy
- **Corrected weighted 5-class**: train split의 class probability 기반 inverse-sqrt class-weighted cross-entropy

두 실험은 train/validation pipeline과 validation report를 생성했습니다. `run_status.json`의
`stage`가 `complete`이며, 사전 정의한 `final_pass` gate는 두 run 모두 별도로 기록되어 있습니다.
이는 학습 프로세스의 완료 여부와 연구 가설의 통과 여부를 구분하기 위한 필드입니다.

## Stage 2 5-class 설정

| Class | Raw pedal target | Representative |
| --- | ---: | ---: |
| ZERO | 0 | 0 |
| LOW | 1–63 | 32 |
| MID | 64–95 | 80 |
| HIGH | 96–126 | 111 |
| FULL | 127 | 127 |

모델은 official pretrained PT encoder와 독립적인 `Linear(768, 5)` pedal head 4개
(Pedal1–4)로 구성됩니다. Encoder는 학습하며, window overlap에서는 raw logits를 평균한 뒤
argmax와 고정 representative decoding을 적용합니다. Class 경계와 representative는
train/validation 분포로 다시 추정하지 않습니다.

## 실험 산출물

| 실험 | Loss | 결과 |
| --- | --- | --- |
| Baseline | Unweighted 5-class CE | [`analysis/stage2_encoder_only_5class_v0/`](analysis/stage2_encoder_only_5class_v0/) · [`FIVE_CLASS_TRAINING_REPORT.md`](analysis/stage2_encoder_only_5class_v0/FIVE_CLASS_TRAINING_REPORT.md) |
| Corrected weighted | Global train Pedal1–4 weighted CE | [`analysis/stage2_encoder_only_5class_weighted_sqrt_probability_v0/`](analysis/stage2_encoder_only_5class_weighted_sqrt_probability_v0/) · [`WEIGHTED_5CLASS_REPORT.md`](analysis/stage2_encoder_only_5class_weighted_sqrt_probability_v0/WEIGHTED_5CLASS_REPORT.md) |

Weighted run의 실제 class count, probability, weight는
[`class_weights.json`](analysis/stage2_encoder_only_5class_weighted_sqrt_probability_v0/class_weights.json)에
저장되어 있습니다.

### Weighted loss 정의

ASAP **train split의 Pedal1–4 target만** 합쳐 class count `f_c`를 계산하고,
`p_c = f_c / Σf_c`, `raw_c = 1 / √p_c`로 둔 뒤 다음 조건으로 정규화합니다.

```text
w_c = raw_c / Σ_j(p_j · raw_j)
Σ_c p_c w_c = 1
```

최종 weight는 class 순서 `[ZERO, LOW, MID, HIGH, FULL]`에서

```text
[0.9258126368700947,
 1.204207482868018,
 1.1252359220805253,
 1.7123712206707022,
 0.7669776275315598]
```

이며 네 pedal head에 동일하게 적용합니다. Validation/test split은 weight 계산에 사용하지
않고, weighted sampler·focal/ordinal loss·label smoothing도 사용하지 않습니다.

## 두 실험에서 고정한 조건

- ASAP piece-wise split: train 892 performances/180 pieces, validation 71/19
- ASAP test split은 이 pipeline에 전달하지 않음
- seed `20260710`, window `512`, stride `256`, batch size `16`
- AdamW, encoder LR `1e-5`, head LR `1e-4`, weight decay `0.01`
- 최대 20 epochs, early stopping patience `4`, min delta `1e-4`
- AMP 사용, 기존 official PT pretrained checkpoint와 동일한 5-class architecture
- Host GPU 1을 기존 컨테이너에서 노출한 CUDA device(`cuda:0`) 사용

실행 entry point는 [`src/stage2_encoder_only/run_five_class.py`](src/stage2_encoder_only/run_five_class.py)이며,
학습·validation·metric 비교·report 생성 설정은 각 output directory의 `config.json`에 보존합니다.

## 검증 범위

각 report에는 validation 기준 per-class precision/recall/F1, macro F1, confusion matrix,
예측 class 분포, decoded MAE, OFF/ON accuracy, binary transition F1과 baseline 비교가
포함됩니다. Weighted 구현에 대해서는 finite loss/gradient, 네 head gradient, pedal-rich
smoke/overfit 및 관련 unit test도 확인했습니다.

```bash
python -m unittest \
  tests/test_stage2_weighted_five_class.py \
  tests/test_stage2_five_class.py \
  tests/test_stage2_five_class_runtime.py
```

현재 README의 Stage 2 결과는 학습 및 validation 연구용입니다. ASAP test 평가, 전체 23개
score inference, encoder-decoder 구조 변경은 이 실험 범위에 포함하지 않습니다.

## 주요 디렉터리

- `src/stage2_encoder_only/`: Stage 2 dataset, model, training, evaluation
- `analysis/`: split snapshot, config, checkpoint, validation metrics, reports
- `tests/`: Stage 2 모델·loss·runtime 및 weighted loss unit test
- `scripts/`: 실험 및 Stage 1 → Stage 2 listening/inference 실행 스크립트
- `docs/`: setup, workflow, research notes, troubleshooting, decisions
- `third_party/`: 외부 저장소(예: Pianist Transformer)
- `checkpoints/`: pretrained 및 학습 checkpoint (Git 미추적)
- `outputs/`: MIDI/audio 및 후처리 산출물
- `data/`: raw/interim/processed data
