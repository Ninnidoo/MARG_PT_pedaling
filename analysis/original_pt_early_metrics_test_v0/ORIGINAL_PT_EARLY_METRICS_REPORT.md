# Original Pianist Transformer Early-Style Test Metrics v0

## Outcome

저장된 Original PT MIDI 23개만을 읽어 ASAP piece-wise test split의 23 pieces / 104 unique human performances에 대해 early raw-128-style 및 endpoint-aware 5-class-style metric을 CPU-only로 계산했다. PT inference, Stage 2 inference, checkpoint load, training, GPU/CUDA, official JS/Intersection 계산은 모두 0회다.

## Data and correspondence audit

- Original PT MIDI: 23 files, canonical nums 0..22 exactly once; manifest hash 검증 PASS.
- ASAP test: 104 unique performance paths, 23 piece IDs; duplicate evaluation 0.
- Human raw tokenizer note total은 331,576이지만 canonical score grid로 반복 집계한 note samples는 336,620이다. Note 수가 다르므로 direct index, truncate, pad, nearest-neighbour matching은 사용하지 않았다.
- Correspondence: PT repository의 `align_score_and_performance`와 동일한 Nakamura MIDI-to-MIDI correspondence, 동일 `normalize_midi`, `read_corresp`, `interpolate`, `midi_to_ids` 경로를 사용했다. Metric에는 PT SFT segmentation 전 full canonical score grid를 사용하여 silent skip은 0이다.
- Matched score notes: 311,826; PT interpolation이 적용된 unmatched score notes: 24,794; match ratio: 92.634425%; 최대 consecutive interpolation: 1375 notes.
- 3 human performances는 ASAP metadata상 alternate repeat score variant를 사용한다. Fixed Original PT가 생성된 canonical score에 같은 공식 aligner로 직접 대응시켰으며 per-performance CSV에서 명시적으로 flag했다.
- Saved PT 자체도 canonical score와 공식 aligner로 대응시킨 뒤, actual saved MIDI load와 official `midi_to_ids` semantics로 Pedal1–4를 추출했다.
- PT repository는 alignment tool download page만 지정하고 version/hash를 pin하지 않는다. 이번 실행은 공식 `AlignmentTool_v240109` ZIP (`cf75af54435c6ad83a7b578b691724866f6df5701e8529d86ae2a3db1e85bddd`)을 사용했으며 `config.json`에 static program hashes까지 기록했다.

이 결과는 alignment-dependent baseline이다. 특히 24,794 score notes와 최대 1,375-note 연속 구간이 PT의 interpolation에 의존하므로, raw note-wise 숫자를 manual ground-truth note alignment 또는 Oracle Stage 2와 동일 조건인 것처럼 해석하면 안 된다.

이 baseline은 score-generated Original PT를 human performance pedal target에 score-note correspondence로 비교한다. Human non-pedal token을 그대로 입력으로 쓴 Oracle Stage 2와 입력 조건이 다르며, 동일 조건 controlled comparison이 아니다.

## Raw 128-class-style metrics

| Metric | Original PT baseline |
| --- | --- |
| Pedal-token / exact-value accuracy | 0.503142 |
| Exact-note accuracy | 0.456345 |
| Pedal1 accuracy | 0.501925 |
| Pedal2 accuracy | 0.501061 |
| Pedal3 accuracy | 0.504168 |
| Pedal4 accuracy | 0.505413 |
| Raw CC64 MAE | 40.938122 |
| Transition accuracy | 0.100571 |
| Steady-position accuracy | 0.574098 |
| Transition precision | 0.479940 |
| Transition recall | 0.065172 |
| Transition F1 | 0.114760 |
| Intermediate exact accuracy | 0.000000 |
| Intermediate MAE | 58.177207 |
| Intermediate endpoint collapse | 1.000000 |
| Predicted ZERO ratio | 0.553271 |
| Predicted INTERMEDIATE ratio | 0.000000 |
| Predicted FULL ratio | 0.446729 |

RMSE 및 raw ±5/±10/±20 accuracy는 초기 `evaluate_oracle.py`에 정의되어 있지 않아 새 정의를 추가하지 않았다. Transition/steady는 performance별 note-major `P1→P2→P3→P4` flatten 후 집계하며 performance boundary를 넘지 않는다.

## Five-class-style metrics

Classes: ZERO=0, LOW=1–63, MID=64–95, HIGH=96–126, FULL=127. Representatives: `[0, 32, 80, 111, 127]`.

| Metric | Original PT baseline |
| --- | --- |
| 5-class token accuracy | 0.503142 |
| 5-class exact-note accuracy | 0.456345 |
| Macro F1 | 0.239746 |
| Micro F1 | 0.503142 |
| Weighted F1 | 0.406376 |
| OFF/ON state accuracy | 0.692070 |
| Binary transition precision | 0.173913 |
| Binary transition recall | 0.103025 |
| Binary transition F1 | 0.129397 |
| 5-class canonical decoded MAE | 40.938122 |
| Decoded tolerance ±5 | 0.510816 |
| Decoded tolerance ±10 | 0.514826 |
| Decoded tolerance ±20 | 0.531021 |

Raw Original PT CC64 MAE는 **40.938122**, Original PT raw value를 5-class로 quantize한 뒤 representative로 decode한 MAE는 **40.938122**다. 정의는 서로 다르지만 이번 saved PT는 ZERO/FULL endpoint만 포함하므로 수치가 같아졌다.

### Per-class results

| Class | Precision | Recall | F1 | Predicted ratio |
| --- | --- | --- | --- | --- |
| ZERO | 0.488673 | 0.792153 | 0.604459 | 0.553271 |
| LOW | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| MID | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| HIGH | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| FULL | 0.521061 | 0.691412 | 0.594270 | 0.446729 |

Confusion matrix는 `five_class_confusion_matrix.csv`, performance별 결과와 alignment audit는 `per_performance_metrics.csv`에 저장했다.

## Earlier Stage 2 references (reference only)

| Metric | Original PT baseline | Oracle Stage 2 test reference |
| --- | --- | --- |
| Token accuracy | 0.503142 | 0.568735 |
| Exact-note accuracy | 0.456345 | 0.506394 |
| MAE | 40.938122 | 31.802348 |
| Transition accuracy | 0.100571 | 0.110027 |
| Steady accuracy | 0.574098 | 0.653181 |
| Transition F1 | 0.114760 | 0.124416 |

위 Oracle Stage 2 값은 같은 104-performance test split이지만 각 human performance의 non-pedal input을 사용했다. Original PT baseline은 score에서 생성된 saved output이므로 직접적인 architecture-only 또는 controlled comparison으로 해석하면 안 된다.

5-class 기존 reference는 **validation 71 performances**의 endpoint-aware encoder-only model 결과다: token accuracy 0.580023, exact-note 0.505396, macro-F1 0.341640, weighted-F1 0.520112, decoded MAE 28.764687. Split이 다르므로 Original PT test baseline과 같은-split model comparison으로 사용하지 않는다.

## Integrity and scope

- All 104 split rows evaluated exactly once; duplicate 0.
- Silent skip / truncation / padding: 0 / 0 / 0.
- No Stage 2 result was recomputed and no model selection was performed.
- No PT or Stage 2 neural inference, checkpoint loading, training, audio, JS, or Intersection evaluation was performed.
- Full provenance, source hashes, alignment tool hash, and CPU-only controls are in `config.json`.
