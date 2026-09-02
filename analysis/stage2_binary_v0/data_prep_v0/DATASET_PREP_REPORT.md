# Stage 2 binary v0 dataset preparation report

## 범위와 입력

- 수행 범위: CSV manifest와 leakage/duplicate audit만 생성. tokenization, CC64 threshold, histogram, balancing, 모델 구현, 학습, 평가는 수행하지 않았다.
- 실제 bind 확인: host `/public/intern_2026_summer_public_dataset/ilkyun_data` → 기존 container `ilkyun-marg-pedaling-dev:/workspace/public`.
- ASAP root: `/workspace/public/ASAP/asap-dataset-v1.1`
- ASAP metadata: `/workspace/public/ASAP/asap-dataset-v1.1/metadata.csv` (SHA-256 `a338da028063546c17b5f4bbec24ada984a3c34d4b4d1a3d2a225de09c81070a`)
- MAESTRO root: `/workspace/public/MAESTRO/maestro-v3.0.0`
- MAESTRO metadata: `/workspace/public/MAESTRO/maestro-v3.0.0/maestro-v3.0.0.csv` (SHA-256 `cb88a64a12971a995dd41852aa3a33879501cf70d01b9a6feb637f4ad8d8ddff`)
- 발견했지만 사용하지 않은 보조 metadata: `/workspace/public/MAESTRO/maestro-v3.0.0/maestro-v3.0.0.json`
- 기존 ASAP split source of truth: `/workspace/project/analysis/stage2_encoder_only_v0/asap_split.csv` (SHA-256 `d1fe379eb123ff7abca93773296f735f40a215dcd6e6363b55756383708c965b`; Git commit `4c1d5daa527d76e815fcbda0239bcdec6af9458a`).
- full split 생성 스크립트는 현재 repository에서 발견되지 않았다. `tests/test_stage2_dataset.py`는 original PT seed-42 test-score membership만 재구성해 검증하며, 이번 작업은 seed를 실행하거나 split을 재생성하지 않고 저장 CSV를 그대로 사용했다.

## 기존 split 검증

| split | piece 수 | performance 수 |
| --- | ---: | ---: |
| train | 180 | 892 |
| validation | 19 | 71 |
| test | 23 | 104 |

- split CSV 1,067행은 ASAP metadata 1,067행과 `metadata_index`, composer, title, performance path 기준으로 전 행 exact alignment를 확인했다.
- 한 `piece_id`가 둘 이상의 split에 속하는 경우는 0건이다.
- 기대값 180/19/23 piece와 정확히 일치했다.

## Composition overlap 방법

1. ASAP validation/test의 `maestro_midi_performance`를 `{maestro}/` 뒤 exact filename으로 해석했다. 직접 링크 67개는 모두 MAESTRO CSV row와 일치했다.
2. 직접 row의 exact canonical `(composer, title)` pair가 같은 MAESTRO row를 동일 metadata composition으로 확장했다.
3. 추가 표기 변형은 BWV/Op./Hob./K./S. 번호와 movement 또는 Complete/명시적 collection 범위가 확인되는 exact allowlist만 사용했다. fuzzy score/threshold는 사용하지 않았다.
4. 확신할 수 없는 후보는 clean 제외 집합에 넣지 않고 `ambiguous_matches.csv`에 기록했다.

## 결과 수량

| 항목 | 수 |
| --- | ---: |
| ASAP held-out composition (validation 19 + test 23) | 42 |
| confirmed MAESTRO overlap이 있는 held-out ASAP composition | 35 |
| overlap audit의 distinct MAESTRO canonical pair | 76 |
| 제거되는 distinct MAESTRO performance | 102 |
| 그중 held-out direct filename | 67 |
| overlap audit row (ASAP composition × MAESTRO performance) | 110 |
| 최종 MAESTRO-clean performance | 1174 |
| 최종 ASAP-train performance | 892 |
| 최종 train_manifest performance | 2066 |
| ambiguous match row | 6 |

MAESTRO 원본은 1276 performances이며, `MAESTRO-clean = 1276 - 102 = 1174`이다. `train_manifest = 1174 + 892 = 2066`이다.

## Ambiguous cases

아래 후보는 임의 제외하지 않았으며 현재 `maestro_clean_manifest.csv`에 남아 있다. 각 행의 후속 수동 확인 전에는 clean 확정으로 과해석하지 않아야 한다.

| split | ASAP composer | ASAP composition | candidate MAESTRO title | MIDI filename | reason |
| --- | --- | --- | --- | --- | --- |
| test | Chopin | Etudes_op_25_8 | 12 Etudes, Op. 25 | 2004/MIDI-Unprocessed_SMF_05_R1_2004_01_ORIG_MID--AUDIO_05_R1_2004_03_Track03_wav.midi | canonical title says 12 Etudes, Op. 25 but 327 s duration is inconsistent with a complete set; included etude numbers are unspecified |
| test | Chopin | Etudes_op_25_4 | 12 Etudes, Op. 25 | 2004/MIDI-Unprocessed_SMF_05_R1_2004_01_ORIG_MID--AUDIO_05_R1_2004_03_Track03_wav.midi | canonical title says 12 Etudes, Op. 25 but 327 s duration is inconsistent with a complete set; included etude numbers are unspecified |
| test | Mozart | Piano_Sonatas_12-2 | Sonata in F | 2006/MIDI-Unprocessed_16_R1_2006_01-04_ORIG_MID--AUDIO_16_R1_2006_01_Track01_wav.midi | key-only MAESTRO title omits K. 332 and movement identity |
| validation | Chopin | Etudes_op_25_12 | 12 Etudes, Op. 25 | 2004/MIDI-Unprocessed_SMF_05_R1_2004_01_ORIG_MID--AUDIO_05_R1_2004_03_Track03_wav.midi | canonical title says 12 Etudes, Op. 25 but 327 s duration is inconsistent with a complete set; included etude numbers are unspecified |
| validation | Haydn | Keyboard_Sonatas_39-2 | Son. Hob. XVI:39 | 2006/MIDI-Unprocessed_23_R1_2006_01-05_ORIG_MID--AUDIO_23_R1_2006_02_Track02_wav.midi | work catalogue is exact but the MAESTRO title does not identify whether this file contains the held-out second movement |
| validation | Haydn | Keyboard_Sonatas_32-1 | Son. Hob. XVI:32 | 2006/MIDI-Unprocessed_19_R1_2006_01-07_ORIG_MID--AUDIO_19_R1_2006_02_Track02_wav.midi | work catalogue is exact but the MAESTRO title does not identify whether this file contains the held-out first movement |

## ASAP-train ↔ MAESTRO direct source duplicate audit

- direct/exact ASAP-train performance row: **452**
- distinct MAESTRO source MIDI filename: **372**
- source `start` 또는 `end`가 있어 excerpt로 표시된 ASAP row: **211**
- 목록: `asap_train_maestro_duplicates.csv`
- 이것은 ASAP train leakage가 아니라 concatenate 시 source performance 이중 집계 가능성 audit이다. 이번 단계에서는 제거 정책을 적용하지 않았다.
- held-out composition 제거 때문에 duplicate source 중 일부가 MAESTRO-clean에서 빠질 수 있으며, 각 행의 `maestro_in_clean_manifest`에 상태를 기록했다.

## 무결성 검증

| 조건 | 결과 | 근거 |
| --- | --- | --- |
| A. validation/test 직접 연결 MAESTRO performance가 clean에 0개 | PASS | direct 67 filename; clean 교집합 0 |
| B. confirmed held-out composition MAESTRO performance가 clean에 0개 | PASS | confirmed exclusion 102 filename; clean 교집합 0 |
| C. ASAP validation/test performance가 train manifest에 0개 | PASS | ASAP source 행은 저장 split의 train 892행만 사용 |
| D. 기존 ASAP split 미수정 | PASS | 전/후 SHA-256 `d1fe379eb123ff7abca93773296f735f40a215dcd6e6363b55756383708c965b` 동일 |
| E. 원본 dataset 미수정 | PASS | metadata hash와 전체 file `(relative path,size,mtime_ns)` inventory digest 전/후 동일 |

Dataset inventory: ASAP 2879 files / `9fb2915d6872e3623e478151a8ce72dc3aa52acfbbfb0cb131f65876fe78443f`; MAESTRO 1280 files / `d9fcd41b311e1f28d004f8f2c819f938fda6bb12602905126c235395d2ab029b`.

## 산출물

- `heldout_asap_pieces.csv`: 42개 held-out ASAP piece composition
- `maestro_overlap_audit.csv`: confirmed match와 제외 filename (67 direct-link audit rows 포함)
- `ambiguous_matches.csv`: 미확정 후보; clean 제외에 미반영
- `maestro_clean_manifest.csv`: confirmed held-out overlap 제거 후 MAESTRO
- `asap_train_manifest.csv`: 저장 split artifact의 ASAP train performance
- `train_manifest.csv`: `source` 포함 MAESTRO-clean + ASAP-train
- `asap_train_maestro_duplicates.csv`: train direct-source 중복 audit

## 주의

- ambiguous 후보는 요청에 따라 추측으로 제거하지 않았다. 따라서 이 manifest의 “clean”은 confirmed identity 기준이며, ambiguous 후보의 수동 판정 전까지 provisional이다.
- 원본 MIDI/audio/metadata는 삭제, 이동, 수정, 복사하지 않았다. 모든 manifest는 원본 경로를 참조한다.
