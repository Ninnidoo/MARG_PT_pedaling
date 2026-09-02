# Stage 2 binary v1: final dataset and target-distribution audit

## 범위

- 이 단계는 final manifest 확정과 target 통계 audit만 수행했다. 모델/head/loss/weighting/balancing/window/cache/training/Stage 1 inference는 수행하지 않았다.
- tokenization 대상: final MAESTRO-clean, ASAP train, ASAP validation.
- **ASAP test MIDI는 열거나 tokenization하지 않았고 distribution/metric 계산에도 사용하지 않았다.** 저장 split은 contamination membership 확인에만 읽었다.

## Manifest 확정

- v0 source: `/workspace/project/analysis/stage2_binary_v0/data_prep_v0`
- v0 MAESTRO-clean: 1,174 performances.
- v0 ambiguous audit 6 relationships의 distinct MAESTRO MIDI 4개를 conservative leakage prevention으로 모두 추가 제외했다:
  - `2004/MIDI-Unprocessed_SMF_05_R1_2004_01_ORIG_MID--AUDIO_05_R1_2004_03_Track03_wav.midi`
  - `2006/MIDI-Unprocessed_16_R1_2006_01-04_ORIG_MID--AUDIO_16_R1_2006_01_Track01_wav.midi`
  - `2006/MIDI-Unprocessed_19_R1_2006_01-07_ORIG_MID--AUDIO_19_R1_2006_02_Track02_wav.midi`
  - `2006/MIDI-Unprocessed_23_R1_2006_01-05_ORIG_MID--AUDIO_23_R1_2006_02_Track02_wav.midi`
- confirmed overlap 102개와 위 ambiguous 4개는 final MAESTRO manifest에 0개 남았다.
- final MAESTRO-clean: **1,170 performances**
- ASAP train: **892 performances**
- final training pool (naive concatenation): **2,062 performances**
- ASAP-train ↔ MAESTRO direct-source duplicates는 정책대로 제거하지 않았다: v0 audit 기준 **452 ASAP rows / 372 distinct MAESTRO source filenames**, source excerpt 211 rows.

## Official tokenizer provenance

- Pianist Transformer commit: `747df2d12291e37f6638b39f1b71517e579ad48c`
- implementation: `third_party/PianistTransformer/src/utils/midi.py::normalize_midi` 및 `midi_to_ids`
- config: `third_party/PianistTransformer/src/model/pianoformer.py::PianoT5GemmaConfig`
- tokenizer SHA-256: `2fb37eaca3d6e4f4a775eb8a59379e7e09651fda36c8146cf0067cde8ad78633`
- 새 pedal sampling logic은 구현하지 않았다. official `midi_to_ids`가 생성한 8-token note `[Pitch, IOI, Velocity, Duration, Pedal1, Pedal2, Pedal3, Pedal4]`에서 Pedal token offset 5,261만 제거했다.
- official sampling: Pedal1=note onset, Pedal2–4=다음-note IOI의 1/4, 2/4, 3/4 위치에서 직전 CC64 value를 샘플한다.

## Target 정의와 unit assertions

- binary threshold: **64** (official PT evaluator와 동일)
- `raw < 64 -> OFF=0`; `raw >= 64 -> ON=1`
- independent target: `[P1,P2,P3,P4]`, 각 bit는 `{0,1}`
- joint target: `joint_id = 8*P1 + 4*P2 + 2*P3 + P4`, 범위 `[0,15]`
- 모든 note에서 binary→joint→4-bit decode exact equality를 assertion했다.
- exhaustive 16-pattern round-trip과 boundary `63→OFF`, `64→ON` unit assertion을 통과했다.

## Tokenization accounting

| dataset | manifest performances | success | failed/skipped | total notes | Pedal1–4 targets |
| --- | ---: | ---: | ---: | ---: | ---: |
| MAESTRO-clean | 1,170 | 1,170 | 0 | 6,380,481 | 25,521,924 |
| ASAP train | 892 | 892 | 0 | 2,988,614 | 11,954,456 |
| ASAP validation | 71 | 71 | 0 | 283,928 | 1,135,712 |

Failure detail:
- 0건. 모든 selected performance가 성공적으로 tokenization되었다.

lightweight token cache는 생성하지 않았다. 전체 note/token array가 필요 없는 descriptive audit이므로 performance 하나씩 official tokenizer에 통과시킨 뒤 integer histogram만 누적하는 streaming 방식이 더 작고 명확하다.

## Raw pedal-value distribution

| dataset | raw=0 | intermediate 1–126 | raw=127 |
| --- | ---: | ---: | ---: |
| MAESTRO-clean | 6,876,861 (26.944916%) | 9,218,181 (36.118676%) | 9,426,882 (36.936408%) |
| ASAP train | 2,975,658 (24.891622%) | 4,643,052 (38.839509%) | 4,335,746 (36.268869%) |
| ASAP validation | 284,410 (25.042440%) | 355,418 (31.294730%) | 495,884 (43.662830%) |

## Binary OFF/ON distribution

| dataset | OFF | ON |
| --- | ---: | ---: |
| MAESTRO-clean | 10,146,799 (39.757187%) | 15,375,125 (60.242813%) |
| ASAP train | 4,734,499 (39.604471%) | 7,219,957 (60.395529%) |
| ASAP validation | 422,309 (37.184515%) | 713,403 (62.815485%) |

### Slot별 OFF/ON

| dataset | slot | OFF | ON |
| --- | --- | ---: | ---: |
| MAESTRO-clean | Pedal1 | 2,552,530 (40.005291%) | 3,827,951 (59.994709%) |
| MAESTRO-clean | Pedal2 | 2,572,548 (40.319029%) | 3,807,933 (59.680971%) |
| MAESTRO-clean | Pedal3 | 2,516,302 (39.437497%) | 3,864,179 (60.562503%) |
| MAESTRO-clean | Pedal4 | 2,505,419 (39.266930%) | 3,875,062 (60.733070%) |
| ASAP train | Pedal1 | 1,188,269 (39.759869%) | 1,800,345 (60.240131%) |
| ASAP train | Pedal2 | 1,198,521 (40.102904%) | 1,790,093 (59.897096%) |
| ASAP train | Pedal3 | 1,176,062 (39.351418%) | 1,812,552 (60.648582%) |
| ASAP train | Pedal4 | 1,171,647 (39.203691%) | 1,816,967 (60.796309%) |
| ASAP validation | Pedal1 | 105,884 (37.292553%) | 178,044 (62.707447%) |
| ASAP validation | Pedal2 | 106,689 (37.576076%) | 177,239 (62.423924%) |
| ASAP validation | Pedal3 | 105,025 (36.990012%) | 178,903 (63.009988%) |
| ASAP validation | Pedal4 | 104,711 (36.879420%) | 179,217 (63.120580%) |

## Joint 16-pattern distribution

Pattern 순서는 `0000, 0001, 0010, ..., 1111`이며 P1이 most-significant bit이다.

| joint_id | pattern | MAESTRO-clean | ASAP train | ASAP validation |
| ---: | --- | ---: | ---: | ---: |
| 0 | `0000` | 2,267,238 (35.533967%) | 1,060,023 (35.468716%) | 95,629 (33.680722%) |
| 1 | `0001` | 56,933 (0.892299%) | 28,437 (0.951511%) | 2,269 (0.799146%) |
| 2 | `0010` | 1,216 (0.019058%) | 518 (0.017332%) | 57 (0.020076%) |
| 3 | `0011` | 93,549 (1.466175%) | 42,872 (1.434511%) | 3,384 (1.191851%) |
| 4 | `0100` | 2,794 (0.043790%) | 1,300 (0.043498%) | 107 (0.037686%) |
| 5 | `0101` | 168 (0.002633%) | 59 (0.001974%) | 7 (0.002465%) |
| 6 | `0110` | 4,801 (0.075245%) | 2,113 (0.070702%) | 234 (0.082415%) |
| 7 | `0111` | 125,831 (1.972124%) | 52,947 (1.771624%) | 4,197 (1.478192%) |
| 8 | `1000` | 78,083 (1.223779%) | 35,952 (1.202966%) | 2,842 (1.000958%) |
| 9 | `1001` | 24,885 (0.390018%) | 10,393 (0.347753%) | 907 (0.319447%) |
| 10 | `1010` | 302 (0.004733%) | 142 (0.004751%) | 14 (0.004931%) |
| 11 | `1011` | 50,342 (0.789000%) | 20,184 (0.675363%) | 1,587 (0.558945%) |
| 12 | `1100` | 77,126 (1.208780%) | 36,396 (1.217822%) | 2,828 (0.996027%) |
| 13 | `1101` | 9,075 (0.142231%) | 3,502 (0.117178%) | 436 (0.153560%) |
| 14 | `1110` | 73,859 (1.157577%) | 35,203 (1.177904%) | 3,000 (1.056606%) |
| 15 | `1111` | 3,514,279 (55.078590%) | 1,658,573 (55.496394%) | 166,430 (58.616973%) |

## MAESTRO-clean vs ASAP train descriptive similarity

- official-PT-style base-2 JS metric (SciPy `jensenshannon`과 같은 **distance**): **0.010228999069**
- base-2 JS divergence (distance², 명칭 모호성 방지용 병기): **0.000104632422**
- Intersection Area `sum(min(p_i,q_i))`: **0.994935982392**
- descriptive audit 전용이다. 현재 naive concatenation 결정, dataset/class weighting, balancing, oversampling에는 사용하지 않았다.

## Pedal missing/sparse audit

`missing = raw non-drum CC64 event 0`, `sparse = 1–4 events`로 사전 정의해 audit했다. 임의 제외에는 사용하지 않았다.

| dataset | missing (0) | sparse (1–4) | total flagged |
| --- | ---: | ---: | ---: |
| MAESTRO-clean | 2 | 4 | 6 |
| ASAP train | 12 | 2 | 14 |
| ASAP validation | 0 | 0 | 0 |

상세 목록과 tokenized OFF/ON 수는 `pedal_missing_or_sparse.csv`에 있다.

## 필수 무결성 검증

| 조건 | 결과 | 근거 |
| --- | --- | --- |
| A. final train에 ASAP validation/test performance 0 | PASS | ASAP source는 저장 split train 892행과 exact equality; train/held-out path 교집합 0 |
| B. confirmed overlap + ambiguous 4가 final MAESTRO에 0 | PASS | confirmed 102 + ambiguous 4 filename 교집합 0 |
| C. 모든 binary label이 `{0,1}` | PASS | tokenized note마다 assertion |
| D. 모든 joint label이 `[0,15]` | PASS | tokenized note마다 assertion |
| E. binary→joint→binary round-trip | PASS | 전 note 및 exhaustive 16-pattern assertion |
| F. boundary 63 OFF / 64 ON | PASS | explicit unit assertion, `[63,64,63,64] -> 0101 -> 5` |
| G. 원본 MIDI/metadata 미수정 | PASS | input hash와 dataset `(relative path,size,mtime_ns)` inventory 전/후 동일 |

추가 contamination 확인: ASAP train/validation/test piece_id는 split 간 교집합 0; ASAP held-out performance path와 final ASAP-train path 교집합 0; ASAP test tokenization count 0.

Dataset inventory after audit: ASAP 2879 files / `9fb2915d6872e3623e478151a8ce72dc3aa52acfbbfb0cb131f65876fe78443f`; MAESTRO 1280 files / `d9fcd41b311e1f28d004f8f2c819f938fda6bb12602905126c235395d2ab029b`.

## Input hashes

- v0 maestro_clean_manifest: `4daadfa0fec6674cd70f780097ba608bab599e2c47feae8cd8f313547d803618`
- v0 asap_train_manifest: `f2fd1ed09294182861bb165411771bb1055ecbc28e4da8028f66c1806a3dab0c`
- v0 train_manifest: `a7e6058e122ab3d8918d73b9dbbdaf9c1ef25767cb9a50a1ecc12a15a52e26ef`
- v0 ambiguous_matches: `298776da01446f5afb3c780bae7282dcbdcbc97362809542514530d2ba909845`
- ASAP split: `d1fe379eb123ff7abca93773296f735f40a215dcd6e6363b55756383708c965b`
- ASAP metadata: `a338da028063546c17b5f4bbec24ada984a3c34d4b4d1a3d2a225de09c81070a`
- MAESTRO metadata: `cb88a64a12971a995dd41852aa3a33879501cf70d01b9a6feb637f4ad8d8ddff`
