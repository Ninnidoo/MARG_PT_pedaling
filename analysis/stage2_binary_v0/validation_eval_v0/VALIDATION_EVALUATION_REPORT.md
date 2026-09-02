# Stage 2 Binary Validation Evaluation v0

## 결론

고정 ASAP validation split의 **19 pieces / 71 human performances**만 사용해 Original Pianist Transformer(PT) Stage 1 baseline을 생성했다. Piece마다 Stage 1 inference를 정확히 한 번 수행해 총 19개의 generated performance와 동일한 generated-token cache를 저장했다. Binary Stage 2 full training이나 checkpoint evaluation은 수행하지 않았다.

Original PT pedal baseline은 다음과 같다.

- Official PT-style base-2 Jensen–Shannon **distance**: **0.144081839387** (lower is better)
- Base-2 Jensen–Shannon divergence(distance²): **0.020759576441**
- Histogram Intersection Area: **0.907531643144** (higher is better)

모든 필수 assertion이 통과했고 ASAP test MIDI access counter는 **0**이다.

## Validation scope와 source

- Split source of truth: `analysis/stage2_encoder_only_v0/asap_split.csv`
- Split SHA-256: `d1fe379eb123ff7abca93773296f735f40a215dcd6e6363b55756383708c965b`
- ASAP metadata: `/workspace/public/ASAP/asap-dataset-v1.1/metadata.csv`
- ASAP root: `/workspace/public/ASAP/asap-dataset-v1.1`
- Validation composition: **19 pieces**
- Validation human performances: **71**
- Human reference notes: **283,928**
- Original PT generated outputs: **19**
- Original PT generated notes: **48,894**

ASAP test performance path는 split CSV에서 문자열 membership guard를 만드는 데만 읽었다. MIDI loader는 validation performance, 선택된 score, 생성 cache만 category별 allowlist로 열며 test performance path와 일치하면 즉시 실패한다. 최종 access assertion은 `asap_test_midi=0`이다. Test를 읽는 기존 regression suite는 실행하지 않았다.

## Validation score selection

ASAP metadata의 performance별 direct `midi_score` linkage를 사용했다. Piece별 선택 규칙은 다음과 같다.

1. 해당 validation piece의 71-row subset에서 각 direct `midi_score` path의 support count를 센다.
2. 가장 많은 validation human row가 직접 연결된 score를 선택한다.
3. 동률일 때 normalized relative path의 lexical order로 결정한다.

17 pieces는 score candidate가 하나였다. 두 composition에는 repeat variant가 있었으며 다음처럼 결정됐다.

- Haydn `Keyboard_Sonatas_32-1`: `32-1_no_repeat/midi_score.mid` 3/5 vs `32-1/midi_score.mid` 2/5
- Liszt `Gran_Etudes_de_Paganini_6_Theme_and_Variations`: canonical score 3/5 vs repeat score 2/5

전체 선택 path, score hash, support와 alternative candidate는 `validation_score_manifest.csv`에 있다. 한 piece의 여러 human performance에 대해 Stage 1을 반복하지 않았다.

## Official PT inference provenance

- Checkpoint: `/workspace/project/checkpoints/pianist_transformer`
- `model.safetensors` SHA-256: `8fb9111efc147adcc61a4885bd42a6e6b625626d9fb8576811916927aa8281d0`
- Model loader: `PianoT5Gemma.from_pretrained`
- Official inference: `third_party/PianistTransformer/src/model/generate.py::batch_performance_render`
- Official MIDI mapping: `third_party/PianistTransformer/src/model/generate.py::map_midi`
- Official generation source SHA-256: `f4c0de409938cb0576a9076be61ed9bc9a27af140ff1dbb673157d36b4720472`
- Seed: **42**, piece마다 reset하여 ordering과 독립적으로 재현 가능
- Device: container `cuda:0`, host GPU 1에 bind된 NVIDIA GeForce RTX 2080 Ti (`GPU-6982dbee-fbaf-f359-d7ef-a22d0e83400b`)
- Inference dtype: float32

실제 decoding 설정:

| Setting | Value |
|---|---:|
| decoding | multinomial sampling (`do_sample=True`), not greedy |
| temperature | 1.0 |
| top-p | 0.95 |
| top-k | 50 (generation config default) |
| num beams | 1 |
| repetition penalty | 1.0 |
| maximum context | 4,096 tokens |
| overlap ratio | 0.5 |

Official `BatchSparseForcedTokenProcessor`가 각 note의 pitch token을 score pitch로 hard-force했다. Output token 수와 score token 수 및 모든 pitch token의 exact equality를 piece마다 assertion했다. 4,096-token을 넘는 score는 official overlapping block generation 및 previous-block decoder prefix stitching을 변경 없이 사용했다.

19-piece recorded neural generation runtime의 합은 **2,029.439초**였다. 최초 inference 후 cache manifest writer의 provenance column 두 개가 field list에서 빠져 aggregate가 중단됐으나, 19개 token/MIDI/metadata cache는 모두 완성되어 있었다. Column 목록만 수정한 뒤 score/MIDI/token hash를 검증하여 19개 cache를 전부 재사용했고 inference는 반복하지 않았다.

## Stage 1 cache

Cache root:

```text
analysis/stage2_binary_v0/validation_eval_v0/stage1_cache/
```

각 `piece_id` directory에는 다음을 저장했다.

- `generated_ids_int64.npy`: Original PT의 exact 8-token/note generated sequence
- `original_pt.mid`: official `ids_to_midi` output을 `map_midi`한 generated performance
- `inference_metadata.json`: source score identity/hash, seed, decoding 설정, token/MIDI hash, note 수, runtime

`stage1_cache_manifest.csv`가 19개 cache를 한 행씩 추적한다. Generated IDs가 Original PT baseline과 향후 모든 Stage 2 candidate의 shared source of truth다.

## Official pedal metric

- Official tokenizer: `third_party/PianistTransformer/src/utils/midi.py::{normalize_midi,midi_to_ids}`
- Tokenizer SHA-256: `2fb37eaca3d6e4f4a775eb8a59379e7e09651fda36c8146cf0067cde8ad78633`
- Official evaluator: `third_party/PianistTransformer/src/evaluate/evaluate.py::plot_pedal_pattern_distribution`
- Evaluator SHA-256: `42a6066496c570b974c3288eb3e8ac3b83e91c00add94da2a19b59c575a5e02a`
- Reused metric helper: `src/stage2_encoder_only/evaluate_oracle.py::distribution_similarity`
- Pedal threshold: raw `<64` = OFF, raw `>=64` = ON
- `joint_id = 8*P1 + 4*P2 + 2*P3 + P4`
- Pattern order: `0000, 0001, ..., 1111`
- Epsilon: `1e-10`, 이후 probability renormalization

Official evaluator의 변수/출력 label은 “JS divergence”지만 SciPy `jensenshannon(..., base=2)` 반환값은 divergence의 square root인 **JS distance**다. Primary number는 upstream-comparable distance이며, 혼동 방지를 위해 distance² divergence도 병기했다. Intersection은 renormalized probability에 대한 `sum_i min(p_i,q_i)`다.

현재 container에는 `scipy` package가 없어 official plotting evaluator를 runtime import하면 `ModuleNotFoundError`가 발생한다. 새 dependency를 설치하거나 metric을 다시 작성하지 않고, 기존 Stage 2 audit에서 official epsilon/base-2 semantics를 공유하도록 추출·검증한 `evaluate_oracle.distribution_similarity`를 재사용했다. 이 helper는 normalized probability에서 base-2 divergence를 계산하고 그 square root를 official SciPy-compatible distance로 반환한다.

Human reference는 validation human MIDI 71개를 official tokenizer로 다시 tokenization해 global histogram으로 합쳤으며, data-prep v1 audit count와 exact 일치했다. Candidate는 cached Original PT MIDI 19개를 official tokenizer로 다시 tokenization해 global histogram으로 합쳤다.

## Global 16-pattern distributions

| ID | Pattern | Human count | Human probability | Original PT count | Original PT probability |
|---:|---:|---:|---:|---:|---:|
| 0 | 0000 | 95,629 | 0.336807218293 | 20,989 | 0.429275575148 |
| 1 | 0001 | 2,269 | 0.007991462712 | 78 | 0.001595287863 |
| 2 | 0010 | 57 | 0.000200755221 | 6 | 0.000122714543 |
| 3 | 0011 | 3,384 | 0.011918514634 | 156 | 0.003190575626 |
| 4 | 0100 | 107 | 0.000376856204 | 6 | 0.000122714543 |
| 5 | 0101 | 7 | 0.000024654238 | 1 | 0.000020452507 |
| 6 | 0110 | 234 | 0.000824152701 | 1 | 0.000020452507 |
| 7 | 0111 | 4,197 | 0.014781916619 | 119 | 0.002433836559 |
| 8 | 1000 | 2,842 | 0.010009579977 | 73 | 0.001493025827 |
| 9 | 1001 | 907 | 0.003194471933 | 27 | 0.000552215095 |
| 10 | 1010 | 14 | 0.000049308375 | 0 | 0.000000000100 |
| 11 | 1011 | 1,587 | 0.005589445303 | 22 | 0.000449953059 |
| 12 | 1100 | 2,828 | 0.009960271702 | 135 | 0.002761075074 |
| 13 | 1101 | 436 | 0.001535600672 | 63 | 0.001288501755 |
| 14 | 1110 | 3,000 | 0.010566059084 | 112 | 0.002290669708 |
| 15 | 1111 | 166,430 | 0.586169732334 | 27,106 | 0.554382950086 |

- Human probability sum: `0.9999999999999998`
- Original PT probability sum: `0.9999999999999999`
- Original PT distribution은 15 nonzero raw-count bins를 가져 finite/non-degenerate assertion을 통과했다.

## Reusable Stage 2 evaluation interface

`src/stage2_binary/validation_evaluator.py`는 다음 shared interface를 제공한다.

- `load_binary_stage2_checkpoint`: independent 4×2 또는 joint 16 checkpoint를 동일 official encoder에 strict-load
- `infer_cached_binary_pedals`: 512-note/256-stride overlap inference와 logits averaging
- `binary_logits_to_raw_pedals`: Model A의 4×2 argmax 또는 Model B의 joint argmax를 0/127 Pedal1–4로 decode
- `replace_pedal_tokens`: cached Stage 1 token의 Pedal1–4만 교체하고 Pitch/IOI/Velocity/Duration exact equality assertion
- `official_pt_pedal_similarity`: 동일 PT-style metric helper

Future CLI는 `scripts/evaluate_stage2_binary_validation.py`다. `--architecture independent_4x2` 또는 `--architecture joint_16`과 checkpoint를 받으며, 이 cache의 19 sequences만 사용해 candidate MIDI와 같은 metric을 생성한다.

Synthetic replacement test에서 cached Stage 1 token을 복사한 뒤 Pedal1–4만 바꿨을 때 모든 Pitch/IOI/Velocity/Duration token의 exact equality가 통과했다. 실제 future evaluator도 모든 piece에 같은 assertion을 적용한다.

## Required validation 결과

| Check | Result |
|---|---|
| validation piece count = 19 | PASS |
| human validation performance count = 71 | PASS |
| Stage 1 generated output count = 19 | PASS |
| ASAP test MIDI access count = 0 | PASS |
| 63→OFF, 64→ON | PASS |
| exhaustive 16-pattern joint encoding | PASS |
| Human/Candidate distribution sums = 1 | PASS |
| identical distribution: JS=0, Intersection=1 | PASS |
| synthetic pedal-only replacement preserves non-pedal tokens | PASS |
| Original PT validation metric finite and non-degenerate | PASS |

관련 synthetic unit test 명령은 `python -m unittest tests/test_stage2_binary_validation_evaluator.py -v`이며 **5/5 PASS**했다. ASAP MIDI를 읽는 기존 regression test는 실행하지 않았다.

## 향후 checkpoint selection

- Primary: **Pedal JS distance**, lower is better
- Tie-break/secondary: **Pedal Intersection**, higher is better
- Model A와 Model B의 raw validation CE는 loss space가 다르므로 architecture 간 최종 비교나 selection에 사용하지 않는다.

## 중단 지점

Original PT validation baseline, reusable Stage 2 validation evaluator, 19-piece Stage 1 cache까지만 완료했다. Model A/B full training, checkpoint evaluation, hyperparameter tuning은 시작하지 않았다.
