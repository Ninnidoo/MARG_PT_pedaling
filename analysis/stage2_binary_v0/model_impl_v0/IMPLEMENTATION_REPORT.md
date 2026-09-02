# Stage 2 Binary Encoder-only Implementation Report

## 범위와 결론

동일한 official pretrained Pianist Transformer (PT) encoder를 사용하는 두 encoder-only 모델을 구현했다. Model A는 note별 independent 4×2-class head, Model B는 note별 joint 16-class head이다. 두 모델 모두 encoder를 freeze하지 않으며 unweighted cross-entropy와 기존 Stage 2의 `-100` padding/ignore 관례를 사용한다.

새 binary unit test 10개, 기존 Stage 2 dataset/model 회귀 test 20개, 두 official-encoder 모델의 작은 overfit smoke가 통과했다. Full MAESTRO+ASAP training, validation evaluation, ASAP test metric 계산은 시작하지 않았다.

## 재사용 source와 provenance

- Dataset audit: `analysis/stage2_binary_v0/data_prep_v1/BINARY_DATASET_AUDIT.md`
- Training manifest: `analysis/stage2_binary_v0/data_prep_v1/train_manifest.csv`
- 기존 ASAP split artifact: `analysis/stage2_encoder_only_v0/asap_split.csv`
- 기존 encoder-only 관례: `src/stage2_encoder_only/{model,dataset,training}.py`
- Official checkpoint: `checkpoints/pianist_transformer`
- Official tokenizer: `third_party/PianistTransformer/src/utils/midi.py::{normalize_midi,midi_to_ids}`
- Tokenizer revision: PT commit `747df2d12291e37f6638b39f1b71517e579ad48c`

`BinaryManifestSubsetDataset`은 새 pedal sampling을 구현하지 않는다. Official `midi_to_ids`와 기존 `_PinnedTokenizerConfig`를 직접 재사용하므로 raw Pedal1–4는 note onset 및 다음 IOI의 1/4, 2/4, 3/4 지점에서 official 방식으로 얻는다. 이번 smoke는 전체 manifest를 preprocess/cache하지 않고 ASAP-train performance 1개만 memory에서 tokenization했다. 영구 token/window cache는 만들지 않았다.

## 구현 파일

- `src/stage2_binary/model.py`: 공통 encoder base와 두 모델
- `src/stage2_binary/dataset.py`: binary/joint 변환, 기존 collator 재사용, tiny subset loader
- `src/stage2_binary/training.py`: 공통 optimizer grouping
- `configs/stage2_binary_impl_smoke_v0.json`: seed와 제한된 smoke 범위
- `scripts/run_stage2_binary_impl_smoke.py`: official encoder smoke
- `tests/test_stage2_binary.py`: binary unit tests
- `analysis/stage2_binary_v0/model_impl_v0/overfit_results.json`: machine-readable 결과

## 입력과 target

Logical input은 기존과 동일한 note당 8 tokens이다.

```text
[Pitch, IOI, Velocity, Duration, MASK, MASK, MASK, MASK]
```

모델 input은 `[B,N*8]`, token mask는 `[B,N*8]`, note mask는 `[B,N]`이다. Dataset은 non-pedal 네 token을 equality assertion으로 보존하고 pedal 위치만 `MASK_ID=1`로 바꾼다. 모델 output은 pedal target axis만 가진다.

```text
raw < 64  -> 0 (OFF)
raw >= 64 -> 1 (ON)
binary target = [P1,P2,P3,P4]
joint_id = 8*P1 + 4*P2 + 2*P3 + P4
```

Padding target은 `-100`이다. 63→0, 64→1 boundary와 joint ID 0–15 전수 round-trip이 통과했다.

## Architecture와 parameter 수

두 모델 모두 `PianoT5Gemma.from_pretrained(...).get_encoder()`로 checkpoint encoder를 로드한다. Hidden size는 768이고 모든 encoder parameter가 `requires_grad=True`이다.

| Model | Head | Logits | Target | Encoder | Head | Total / trainable |
|---|---|---|---|---:|---:|---:|
| A: independent 4×2 | 서로 다른 `Linear(768,2)` 4개 | `[B,N,4,2]` | `[B,N,4]` | 103,271,424 | 6,152 | 103,277,576 / 103,277,576 |
| B: joint 16 | `Linear(768,16)` 1개 | `[B,N,16]` | `[B,N]` | 103,271,424 | 12,304 | 103,283,728 / 103,283,728 |

두 encoder의 초기 전체 parameter SHA-256은 동일했다.

```text
3d6af38359042962e850fd4afeaedd9ad248266d99f1d2d31e3ab7bbec53aaed
```

두 모델에 공급한 tokenized input도 동일했다(SHA-256 `8ac30112761f5e2b368cb80848e4ac976571230140e78a2b6c901b1f3396ff01`).

## Loss

- Model A: `CE(logits.reshape(-1,2), binary_targets.reshape(-1), ignore_index=-100)`. 모든 유효 note×4 target을 평균한다.
- Model B: `CE(logits.reshape(-1,16), joint_targets.reshape(-1), ignore_index=-100)`. 모든 유효 note target을 평균한다.

Class weighting, ordinal loss, dataset weighting/balancing은 없다.

## Seed와 unit test

- Seed: **42**
- `python -m unittest tests/test_stage2_binary.py -v`: **10/10 PASS**

검증 항목은 (1) 63/64 boundary, (2) independent 값/shape/서로 다른 네 head, (3) 16-pattern exact round-trip, (4) joint 값/shape, (5) padding 제외, (6) 두 모델 finite loss/gradient, (7) encoder trainability와 backward gradient, (8) non-pedal 입력 보존 및 pedal-only output, (9) 두 target representation의 collator padding 정합성, (10) optimizer group 중복/누락 없음이다.

기존 `tests/test_stage2_dataset.py`와 `tests/test_stage2_model.py`도 총 **20/20 PASS**했다.

## 작은 overfit/smoke

- Fixture: `Schubert/Piano_Sonatas/664-3/Lin07.mid` (`ASAP-train`)
- 기존 split artifact에서 유일한 `train` row임을 assertion으로 확인
- 1 performance, 128-note window 4개(start 0/256/512/768), 총 512 notes
- Batch 2, model별 40 steps
- Encoder LR `1e-4`, head LR `1e-3`, weight decay 0, max grad norm 1.0, FP16 AMP
- Device: NVIDIA GeForce RTX 2080 Ti

| Model | Loss initial → final | Binary accuracy | Exact-pattern accuracy | Encoder |
|---|---:|---:|---:|---|
| independent 4×2 | 0.965420 → 0.000165 | 0.502930 → 1.000000 | 0.017578 → 1.000000 | finite gradient, params changed |
| joint 16 | 3.921995 → 0.000032 | 0.518555 → 1.000000 | 0.029297 → 1.000000 | finite gradient, params changed |

두 모델 모두 loss가 99% 이상 감소했고 tiny subset을 완전히 overfit했다. 이는 implementation sanity evidence이며 generalization 결과가 아니다.

## 범위와 contamination

- Binary smoke는 v1 train manifest의 ASAP-train row 1개만 tokenization했다.
- Split artifact는 fixture가 train인지 확인하는 데만 썼다.
- Validation MIDI는 binary smoke에서 tokenization/evaluation하지 않았다.
- Full dataset preprocessing/training, Stage 1 inference, PT JS/Intersection, validation/test metric은 수행하지 않았다.
- 원본 MIDI, metadata, split artifact, v1 manifest는 수정하지 않았다.

### ASAP test 접근 scope deviation

새 binary unit test와 smoke script는 ASAP test MIDI를 열지 않았다. 다만 최종 회귀 확인에서 기존 baseline suite 전체를 실행했고, `tests/test_stage2_dataset.py::RealDatasetTests::test_small_smoke_sample_from_each_split`가 train/validation/test의 첫 sample을 각각 tokenization하므로 ASAP test MIDI 1개가 **read-only로 의도치 않게 한 번 tokenization**되었다. Test target을 새 모델에 전달하거나 metric을 계산·저장하지 않았고 smoke 결과/학습에는 사용하지 않았다. 원본 변경도 없었다. 그러나 “ASAP test 접근 금지”를 엄격히 적용하면 이 검증 명령은 scope deviation이므로 명시한다.

## 중단 지점

두 모델 implementation, unit tests, 작은 official-encoder overfit smoke까지만 완료했다. **Full training은 시작하지 않았다.**
