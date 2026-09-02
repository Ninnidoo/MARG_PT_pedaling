# Stage 2 Binary Full-Training Report

## Outcome

고정된 동일 config/cache로 Model A와 Model B를 host tmux sequential pipeline에서 순서대로 full training했다. Model B epoch 5 도중 기존 runner가 transient AMP overflow를 GradScaler skip/backoff 전에 예외로 처리해 중단됐으며, 같은 tmux session 이름과 epoch-4 `last.pt`의 model/optimizer/scaler/RNG/early-stopping state에서 재개했다. Hyperparameter나 data order는 변경하지 않았다. 두 run은 최종 정상 완료했으며 best/last checkpoint와 metrics의 일관성 및 strict load를 검증했다. ASAP test access는 **0**이다.

## Validation summary

| Model | epochs | best epoch | best JS ↓ | Intersection ↑ | binary acc | exact-pattern acc |
|---|---:|---:|---:|---:|---:|---:|
| independent 4×2 | 5 | 2 | 0.069918887331 | 0.965298332072 | 0.801151764 | 0.736072661 |
| joint 16 | 6 | 3 | 0.168750230719 | 0.919373586922 | 0.812497735 | 0.770699864 |

## Original PT comparison

| Model | Pedal JS ↓ | Intersection ↑ |
|---|---:|---:|
| Original PT | 0.144081839387 | 0.907531643144 |
| independent 4×2 best | 0.069918887331 | 0.965298332072 |
| joint 16 best | 0.168750230719 | 0.919373586922 |

Original PT 대비 변화:

- independent 4×2: JS -0.074162952056, Intersection +0.057766688928
- joint 16: JS +0.024668391332, Intersection +0.011841943778

Primary validation metric 기준 우수 architecture: **independent 4×2** (lower JS; numerical tie이면 higher Intersection).

Validation CE는 architecture selection에 사용하지 않았으며 두 architecture 사이에서 직접 비교하지 않는다.

## Fixed common training configuration

- Cache ID: `85a79e9d10e955b10000f72f6bbc4a29dbd2cc5a4c8cfb1277054a8d45dcb877`
- Training: MAESTRO-clean 1,170 + ASAP train 892 = 2,062 performances
- Training cache: 9,369,095 notes / 35,573 windows
- Validation: ASAP validation 71 performances / 19 pieces / 1,078 windows
- Seed: 42
- Window / stride: 512 / 256 notes
- AdamW; encoder LR `1e-5`; head LR `1e-4`; weight decay 0.01
- Batch / gradient accumulation / effective batch: 16 / 1 / 16
- Max grad norm: 1.0; AMP enabled
- Max epochs: 10; early-stopping patience: 3
- Encoder trainable; unweighted CE
- Dataset/class weighting, balancing, oversampling: none
- Stage 1 validation cache reused; Stage 1 inference regeneration: none

Model A와 Model B의 initial encoder parameter SHA-256는 모두 `3d6af38359042962e850fd4afeaedd9ad248266d99f1d2d31e3ab7bbec53aaed`로 setup provenance와 일치한다.

## Run details

### independent 4×2

- Actual epochs: 5
- Total summed epoch runtime: 01:04:09.255 (3849.255 s)
- Stop: validation Pedal JS did not improve for 3 consecutive epochs
- Best epoch: 2; JS 0.069918887331; Intersection 0.965298332072
- Best validation loss: 0.437378874
- Last epoch 5: train loss 0.276554567; validation loss 0.478120165; binary acc 0.803671531; exact-pattern acc 0.743357926; JS 0.089013962745; Intersection 0.913854589892
- Last optimizer step: 11120
- Peak GPU memory: 9,773,489,664 bytes (9.102 GiB)
- Epoch-boundary resume used: False; AMP-skipped batches after resume: 0
- `best.pt`: 413,170,157 bytes; selection metadata and strict load PASS
- `last.pt`: 1,206,409,785 bytes; epoch/optimizer metadata and strict load PASS

### joint 16

- Actual epochs: 6
- Total summed epoch runtime: 01:16:32.102 (4592.102 s)
- Stop: validation Pedal JS did not improve for 3 consecutive epochs
- Best epoch: 3; JS 0.168750230719; Intersection 0.919373586922
- Best validation loss: 0.648616876
- Last epoch 6: train loss 0.496279742; validation loss 0.705114728; binary acc 0.808892698; exact-pattern acc 0.766695776; JS 0.211797524186; Intersection 0.806312679834
- Last optimizer step: 13342
- Peak GPU memory: 9,783,042,560 bytes (9.111 GiB)
- Epoch-boundary resume used: True; AMP-skipped batches after resume: 2
- `best.pt`: 413,192,697 bytes; selection metadata and strict load PASS
- `last.pt`: 1,206,476,585 bytes; epoch/optimizer metadata and strict load PASS

## Integrity verification

- Both run statuses completed: PASS
- Fixed config equality across architectures: PASS
- Same cache ID and deterministic ordering provenance: PASS
- Same official pretrained encoder provenance: PASS
- Lowest-JS / Intersection-tiebreak best row equals `best.pt`: PASS
- `best.pt` and `last.pt` load with strict architecture state dict: PASS
- Metrics finite and epoch/optimizer sequence consistent: PASS
- Required artifact set present for both architectures: PASS
- Model B recovery restored the epoch-4 checkpoint without changing fixed config; transient AMP overflow used standard dynamic-scale skip/backoff: PASS
- ASAP test access: **0 / PASS**

No test evaluation, Stage 1 regeneration, audio rendering, calibration, weighting, balancing, architecture change, or additional full run was performed.
