# Stage 2 Binary Locked ASAP Final-Test Report

## Final headline result

| Model | Pedal JS ↓ | Intersection ↑ |
|---|---:|---:|
| Original PT | 0.181901863968 | 0.878312399553 |
| independent 4×2 | 0.133316954870 | 0.891138750058 |
| joint 16 | 0.203767205373 | 0.850856482122 |

Locked independent 4×2 Stage 2는 동일 Stage 1 non-pedal performance 위에서 Original PT 대비 official strict Pedal JS를 **개선했다**.

Test 결과를 사용한 parameter/checkpoint/inference 변경은 수행하지 않았다.

## Changes versus Original PT

| Model | absolute JS change | relative JS change | Intersection change |
|---|---:|---:|---:|
| independent 4×2 | -0.048584909098 | -26.709407% | +0.012826350505 |
| joint 16 | +0.021865341405 | +12.020405% | -0.027455917431 |

## Piece-level supplemental analysis

| Candidate | improved pieces | mean JS | median JS |
|---|---:|---:|---:|
| Original PT | — | 0.309821937298 | 0.235401988070 |
| independent 4×2 | 14/23 | 0.291712867109 | 0.243647214431 |
| joint 16 | 11/23 | 0.321992310481 | 0.283515394002 |

Piece-level 결과는 supplemental이며 model/checkpoint 선택에 사용하지 않았다.

## Steady / transition-containing mass

| Distribution | steady mass | transition-containing mass |
|---|---:|---:|
| Human | 0.904962964750 | 0.095037035250 |
| Original PT | 0.987502044823 | 0.012497955177 |
| independent 4×2 | 0.965107148699 | 0.034892851301 |
| joint 16 | 0.990741043677 | 0.009258956323 |

## Locked provenance

- final_lock_v1: `analysis/stage2_binary_v0/final_lock_v1/final_experiment_lock.json`
- final_lock_v1 SHA-256: `2562cac4046935f94c63c694c7270b45093b3960ed1e0ed5e38397867b907d9e`
- independent checkpoint: `analysis/stage2_binary_v0/train_strict_v1/independent_4x2/best.pt`
- independent checkpoint SHA-256: `c5b4e557cdc3b142be7a297990d7260f36cacb4a318da854c3bc540e3dc8eee0`
- joint checkpoint: `analysis/stage2_binary_v0/train_strict_v1/joint_16/best.pt`
- joint checkpoint SHA-256: `f27fd9e0fe02f742b45c6158d228d0879d8cf375ec86e8dcc92b1362ecef0958`
- Original PT checkpoint: `checkpoints/pianist_transformer`
- Original PT model SHA-256: `8fb9111efc147adcc61a4885bd42a6e6b625626d9fb8576811916927aa8281d0`
- ASAP split: `analysis/stage2_encoder_only_v0/asap_split.csv`
- ASAP split SHA-256: `d1fe379eb123ff7abca93773296f735f40a215dcd6e6363b55756383708c965b`
- test score manifest SHA-256: `6519ba63df5082097e68dc2de9f68352dc2428187aa28c2c78ca52bbedd67c9f`
- Stage 1: seed 42 reset per piece; multinomial; temperature 1.0; top-p 0.95; top-k 50; beams 1; repetition penalty 1.0; context 4096; overlap 0.5
- Stage 2: deterministic argmax; threshold 64; 512-note windows; stride 256; overlap-logit averaging; binary 0→0 and 1→127
- strict evaluator: `third_party/PianistTransformer/src/evaluate/evaluate.py::plot_pedal_pattern_distribution`; pinned commit `747df2d12291e37f6638b39f1b71517e579ad48c`

## Integrity

- ASAP test scope: 23 pieces / 104 human performances
- selected scores: 23
- Stage 1 generated outputs: 23
- independent candidate MIDI: 23
- joint candidate MIDI: 23
- MIDI non-pedal equality: 46/46 PASS
- Human notes after official tokenization: 331576
- Original PT notes after strict roundtrip: 61130
- independent notes after strict roundtrip: 61130
- joint notes after strict roundtrip: 61130
- MIDI access counts: `{"human_test": 104, "original_pt_cache": 69, "stage2_candidate": 92, "test_score": 92}`

이 결과는 final_lock_v1 확정 이후 수행한 locked ASAP test final evaluation이며, test 결과를 사용한 parameter/checkpoint/inference 변경은 수행하지 않았다.

## Stop point

Final test evaluation과 locked descriptive audit만 수행했다. 추가 tuning, calibration, rerun, checkpoint reselection은 수행하지 않았다.
