# Stage 2 Endpoint-aware 5-class Candidate Audit v0

## Scope

기존 6-class audit와 동일한 CE baseline split, pinned tokenizer, immutable preload cache, performance별 note-major Pedal1→4 flatten 순서를 사용했다. ASAP train 892 performances / 11,954,456 Pedal targets만 사용했으며 모델 학습과 validation/test MIDI 접근은 없었다. 6-class output은 읽기 전용 비교 baseline으로만 사용했고 수정하지 않았다.

## 5-class distribution

| Class | Overall | Pedal1 | Pedal2 | Pedal3 | Pedal4 |
| --- | --- | --- | --- | --- | --- |
| ZERO | 0.248916 | 0.249787 | 0.250539 | 0.247735 | 0.247604 |
| LOW | 0.147128 | 0.147811 | 0.150490 | 0.145780 | 0.144433 |
| MID | 0.168505 | 0.170264 | 0.168413 | 0.167755 | 0.167587 |
| HIGH | 0.072762 | 0.072990 | 0.071390 | 0.073034 | 0.073633 |
| FULL | 0.362689 | 0.359147 | 0.359168 | 0.365697 | 0.366743 |

Max/min imbalance ratio는 4.984602이며, 모든 class가 train에 존재한다: True.

## Representatives

Global train-derived shared representatives: `[0.0, 48.0, 77.0, 108.0, 127.0]`. ZERO/FULL은 0/127 고정이고 interior는 해당 train bin의 global median이다. Slot median은 `representative_values.json`에 비교용으로만 기록했다.

## Side-by-side: 5-class vs 6-class

| Metric | 5-class | 6-class | 5-class criterion |
| --- | --- | --- | --- |
| Max/min imbalance | 4.984602 | 11.994926 | descriptive |
| Representatives | [0.0, 48.0, 77.0, 108.0, 127.0] | [0.0, 22.0, 53.0, 77.0, 108.0, 127.0] | train medians |
| Overall oracle MAE | 3.727680 | 3.020231 | <=4.0 |
| Intermediate-only MAE | 9.597650 | 7.776182 | <=10.0 |
| Performance-macro MAE | 3.918835 | 3.087626 | descriptive |
| ±5 accuracy | 0.732271 | 0.754892 | descriptive |
| ±10 accuracy | 0.846027 | 0.891411 | >=0.85 |
| ±20 accuracy | 0.976996 | 0.997901 | descriptive |
| Maximum error | 40.000000 | 21.000000 | descriptive |
| Exact transition preservation | 0.462627 | 0.489144 | descriptive |
| Binary transition preservation | 1.000000 | 1.000000 | ==1.0 |
| Direction agreement | 1.000000 | 1.000000 | descriptive |
| Short repedal preservation | 1.000000 | 1.000000 | ==1.0 |
| New transitions | 0 | 0 | ==0 |

5-class overall error quantiles: `{"q01": 0.0, "q05": 0.0, "q25": 0.0, "median": 0.0, "q75": 6.0, "q95": 16.0, "q99": 30.0}`.

Exact transition은 같은 performance 안에서 인접 exact value가 달라지는 boundary 중 class도 달라지는 비율이다. Binary state는 `<64`/`>=64`, short repedal-like는 `>=64→<64→>=64`에서 low run ≤4 flattened samples로 6-class audit와 동일하다. 5-class short pattern은 74,248/74,248로 보존됐다.

## Adoption criteria

| Criterion | Pass |
| --- | --- |
| overall_oracle_mae_le_4 | True |
| intermediate_only_oracle_mae_le_10 | True |
| plus_minus_10_accuracy_ge_85_percent | False |
| binary_transition_preservation_100_percent | True |
| short_repedal_preservation_100_percent | True |
| all_five_classes_present | True |
| no_new_transitions | True |

## Conclusion

**5-class가 채택 기준을 통과하지 못했으므로 기존 6-class를 유지한다.** 모든 기준 통과: False. 이 결론은 표현 audit 결과이며 모델 성능을 의미하지 않는다.
