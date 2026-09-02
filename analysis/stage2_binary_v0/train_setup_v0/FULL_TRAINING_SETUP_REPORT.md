# Stage 2 Binary Full-Training Setup Report

## Outcome

Model A (`independent_4x2`)와 Model B (`joint_16`)가 동일한 persistent int16 cache, deterministic window order, official pretrained Pianist Transformer encoder, optimizer 조건, validation 경로를 사용하는 full-training pipeline을 준비했다. 실제 RTX 2080 Ti에서 architecture별 **1 optimizer step** end-to-end smoke가 통과했다.

**Full training은 시작하지 않았다.** `analysis/stage2_binary_v0/train_v0/` 아래 생성된 run file은 0개이며, full runner는 명시적인 `--confirm-full-training` 없이는 실행을 거부한다. ASAP test MIDI access는 **0**이다.

## Inputs and reused implementation

- Training manifest: `analysis/stage2_binary_v0/data_prep_v1/train_manifest.csv`
- Existing ASAP split artifact: `analysis/stage2_encoder_only_v0/asap_split.csv`
- Official PT checkpoint: `checkpoints/pianist_transformer`
- Existing 19-piece Stage 1 cache: `analysis/stage2_binary_v0/validation_eval_v0/stage1_cache/`
- Existing evaluator: `src/stage2_binary/validation_evaluator.py`
- Reused encoder-only conventions:
  - `src/stage2_encoder_only/dataset.py::generate_window_starts`
  - `src/stage2_encoder_only/train.py::deterministic_train_order`
  - `src/stage2_encoder_only/train.py::atomic_torch_save`
  - `src/stage2_encoder_only/training.py::{set_deterministic_seed,move_batch_to_device}`
- New shared pipeline: `src/stage2_binary/full_training.py`
- Cache builder: `scripts/build_stage2_binary_training_cache.py`
- Guarded full runner: `scripts/run_stage2_binary_full_training.py`
- Setup smoke runner: `scripts/run_stage2_binary_training_smoke.py`

No dataset weighting, balancing, oversampling, class weighting, or architecture-specific tokenization was introduced.

## Final training and validation inventory

| group | performances | notes | 512/256 windows |
|---|---:|---:|---:|
| MAESTRO-clean | 1,170 | 6,380,481 | 24,343 |
| ASAP train | 892 | 2,988,614 | 11,230 |
| **training total** | **2,062** | **9,369,095** | **35,573** |
| ASAP validation | 71 (19 pieces) | 283,928 | 1,078 |

Training과 validation note 수는 `BINARY_DATASET_AUDIT.md`의 6,380,481 + 2,988,614 및 283,928과 **exact match**한다. Cached performance failure는 train 0, validation 0이다.

ASAP validation/test path와 cached ASAP-train path의 교집합은 0으로 재검증했다. Split CSV는 membership 검증에만 사용했으며 test MIDI를 열거나 tokenization/evaluation하지 않았다.

## Shared cache

- Cache ID: `85a79e9d10e955b10000f72f6bbc4a29dbd2cc5a4c8cfb1277054a8d45dcb877`
- Format: split별 single concatenated NumPy `.npy` token array + performance index CSV + window index `.npy`
- Token dtype: `int16`; window index dtype: `int32`
- Access: `numpy.load(..., mmap_mode="r")`, read-only
- Window size / stride: 512 / 256 notes; 기존 Stage 2 tail-window rule 그대로 사용
- Train token payload: 149,905,520 bytes
- Validation token payload: 4,542,848 bytes
- 전체 cache files: 155,469,244 bytes (약 148.27 MiB)
- 생성 시간: train 178.303 s, validation 4.781 s, total 183.092 s
- Cache bounds/dtype/shape 전수 검증: PASS
- Original MIDI/metadata modification: 0

두 architecture는 같은 `train_tokens_int16.npy`와 `train_windows_int32.npy`를 직접 mmap한다. Collator가 같은 raw Pedal1–4 target에서 independent bits와 joint ID를 동시에 만들므로 architecture별 재-tokenization이나 별도 example 생성이 없다.

## Architectures

| architecture | output | loss | encoder params | head params | total/trainable params |
|---|---|---|---:|---:|---:|
| Model A: independent 4×2 | `[B,N,4,2]` | four binary CE values over all valid targets, unweighted mean | 103,271,424 | 6,152 | 103,277,576 |
| Model B: joint 16 | `[B,N,16]` | joint-pattern CE over valid notes, unweighted mean | 103,271,424 | 12,304 | 103,283,728 |

Both models receive `[Pitch, IOI, Velocity, Duration, MASK, MASK, MASK, MASK]`. Raw pedal `<64` is OFF and `>=64` is ON. Joint target is `8*P1 + 4*P2 + 2*P3 + P4`. Encoder and head parameters are all trainable.

## Common full-training configuration

- Seed: 42
- Optimizer: AdamW
- Encoder LR / head LR: `1e-5` / `1e-4`
- Weight decay: 0.01
- Max gradient norm: 1.0
- AMP: enabled, initial scale 1024
- Batch size: 16
- Gradient accumulation: 1
- Effective batch size: 16 windows
- Max epochs: 10
- Early-stopping patience: 3 epochs
- DataLoader workers: 0; pin memory enabled
- Dataset concatenation: MAESTRO-clean followed by ASAP-train as stored in the final manifest; no reweighting

Batch 16은 기존 encoder-only full experiment의 stable setting과 동일하다. 기존 recorded peak는 9,368,943,104 bytes (약 8.72 GiB)였고, 이번 실제 smoke peak도 11 GB device capacity 안에 들었다. 따라서 architecture 간 effective batch 차이를 만들지 않고 batch 16을 유지했다.

Final machine-readable common config는 `common_full_training_config.json`, launch source config는 `configs/stage2_binary_full_training_v0.json`이다.

## Reproducibility

- Model A/B initial encoder parameter SHA-256: `3d6af38359042962e850fd4afeaedd9ad248266d99f1d2d31e3ab7bbec53aaed` (identical)
- Epoch 1 window order SHA-256: `6dac2e18773c3d1fc0b907eabcfcc2bf1f168f7505034d86541ec839259439a7`
- Smoke batch input SHA-256: `603db239a6d14188ab80a258ef05cf4562139f8dd9bc3760e927203e5b5e6d5a`
- Cache ID, ordering function, `seed + epoch` RNG rule, batch size가 architecture 간 동일하다.
- `common_full_training_config.json`과 future architecture별 `config.json`에 cache/order/hash provenance가 기록된다.

Head architecture가 다르므로 head initialization hash는 비교 대상으로 삼지 않는다.

## Validation integration

각 epoch의 validation은 두 부분으로 구성된다.

1. Cached ASAP validation 71 performances / 1,078 windows에서 CE, binary accuracy, exact-pattern accuracy 계산.
2. 기존 19-piece Stage 1 ID cache에 현재 Stage 2 model을 적용하여 Pedal1–4만 교체한 뒤 PT-style Pedal distribution metric 계산.

Stage 1 inference는 반복하지 않는다. Distribution evaluator는 cached `generated_ids_int64.npy`만 읽으며 Stage 2 replacement 후 non-pedal token exact equality를 assert한다. Model B의 binary accuracy는 joint argmax를 4 bits로 decode한 뒤 Model A와 동일하게 계산한다.

Checkpoint selection / early stopping:

1. 가장 낮은 validation Pedal JS distance
2. `abs(JS difference) <= 1e-12`이면 더 높은 Intersection
3. 개선이 없는 epoch 3회에서 early stop

Validation CE는 기록하지만 architecture selection이나 early stopping 기준이 아니다.

Original PT validation baseline:

- JS distance: **0.144081839387**
- Intersection: **0.907531643144**

Future architecture별 output은 `analysis/stage2_binary_v0/train_v0/{independent_4x2,joint_16}/`로 분리되며 `config.json`, `train.log`, `metrics.csv`, `best.pt`, `last.pt`, `run_status.json`을 생성한다. `metrics.csv`에는 epoch, train/validation loss, validation binary/exact accuracy, JS, Intersection, epoch time, optimizer step, encoder/head LR, peak GPU memory를 기록한다.

## End-to-end smoke

두 architecture 모두 final shared cache의 동일 epoch-1 order 중 첫 16 windows를 사용하여 AMP AdamW **1 optimizer step**을 실행했다. Validation CE는 실제 validation cache 16 windows, evaluator wiring은 실제 Stage 1 cache 첫 1 piece/4 windows로 확인했다. 아래 JS/Intersection은 한-piece wiring smoke이므로 model quality 또는 architecture selection 결과로 해석하지 않는다.

| item | Model A independent 4×2 | Model B joint 16 |
|---|---:|---:|
| train loss | 0.750766 | 4.030030 |
| validation loss (smoke subset) | 0.974531 | 4.030965 |
| validation binary accuracy | 0.466583 | 0.504333 |
| validation exact-pattern accuracy | 0.070923 | 0.008423 |
| one-piece Pedal JS distance | 0.827287 | 0.929446 |
| one-piece Pedal Intersection | 0.167364 | 0.061424 |
| peak GPU memory | 8,548,026,880 B (7.96 GiB) | 9,781,973,504 B (9.11 GiB) |
| elapsed smoke section | 4.666 s | 4.456 s |

Smoke checks:

- Batch forward and finite loss: PASS (A/B)
- Finite backward gradients: PASS (A/B)
- Encoder gradient and parameter update: PASS (A/B; full encoder pre/post hashes differ)
- Head gradient and parameter update: PASS (A/B)
- AMP execution: PASS (A/B)
- Peak memory below visible 11 GB GPU capacity: PASS (A/B)
- Metrics JSON/CSV row write: PASS
- Full-state checkpoint save/load, strict model state and optimizer-state restore: PASS (temporary checkpoint removed after test)
- Independent/joint output decode: PASS
- Stage 2 replacement non-pedal exact equality: PASS
- Reusable validation metric path call: PASS
- Same encoder initialization/cache/window order: PASS
- ASAP test access: **0 / PASS**

Machine-readable results: `smoke_results.json`, `smoke_results.csv`.

## Focused verification

`tests/test_stage2_binary_full_training.py` adds four MIDI-free focused tests:

1. mmap cache read and pedal-only masking/non-pedal equality
2. deterministic architecture-independent window ordering
3. equivalent independent/joint metric decoding
4. JS-primary / Intersection-tiebreak early stopping

The container does not include pytest, so these four test functions were directly invoked with container Python: **4/4 PASS**. Source `py_compile`, `git diff --check`, cache count/dtype/bounds exhaustive checks also passed. Unrelated regression suites were not run.

## Scope stop

No full epoch, tmux job, hyperparameter/LR sweep, weighting/balancing, calibration, Stage 1 inference, ASAP test access, audio rendering, or new architecture was performed. The task stops at shared cache, guarded runner, validation integration, and short smoke verification.
