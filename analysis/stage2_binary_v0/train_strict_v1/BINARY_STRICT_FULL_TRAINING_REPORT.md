# Stage 2 Binary Strict Full-Training Report

## Outcome

Model A와 Model B를 기존 fixed cache/config/seed에서 official pretrained encoder로부터 fresh-start하여 순차 재학습했다. 각 epoch의 checkpoint selection과 early stopping은 official PT MIDI-roundtrip strict JS 및 no-post-epsilon-renormalization Intersection만 사용했다. Direct-token metric은 diagnostic으로만 기록했다.

| Model | epochs | strict best epoch | strict JS ↓ | Intersection ↑ | binary acc | exact-pattern acc |
|---|---:|---:|---:|---:|---:|---:|
| independent 4×2 | 5 | 2 | 0.072306914395 | 0.962942038782 | 0.801151764 | 0.736072661 |
| joint 16 | 6 | 3 | 0.150083204504 | 0.921791174179 | 0.812497735 | 0.770699864 |

| Model | strict JS ↓ | Intersection ↑ |
|---|---:|---:|
| Original PT | 0.144081839387 | 0.907531644596 |
| independent 4×2 strict best | 0.072306914395 | 0.962942038782 |
| joint 16 strict best | 0.150083204504 | 0.921791174179 |

Primary architecture under strict global validation JS: **independent 4×2**.

## Fixed training configuration

- Training: MAESTRO-clean 1,170 + ASAP train 892 = 2,062 performances
- Cache: 9,369,095 notes / 35,573 windows; ID `85a79e9d10e955b10000f72f6bbc4a29dbd2cc5a4c8cfb1277054a8d45dcb877`
- Validation: ASAP validation 71 performances / 19 pieces; existing Stage 1 cache only
- Seed 42; 512-note window; stride 256
- AdamW; encoder LR 1e-5; head LR 1e-4; weight decay 0.01
- Batch/effective batch 16/16; gradient accumulation 1; AMP enabled; max grad norm 1.0
- Max epochs 10; strict-JS early-stopping patience 3
- Encoder trainable; unweighted CE; no class/dataset weighting, balancing, or oversampling

All fixed keys were programmatically compared with `configs/stage2_binary_full_training_v0.json` before either run.

## Reproducibility

Both architectures reproduced:

- Initial encoder SHA-256: `3d6af38359042962e850fd4afeaedd9ad248266d99f1d2d31e3ab7bbec53aaed`
- Epoch-1 window order SHA-256: `6dac2e18773c3d1fc0b907eabcfcc2bf1f168f7505034d86541ec839259439a7`
- First training batch input SHA-256: `603db239a6d14188ab80a258ef05cf4562139f8dd9bc3760e927203e5b5e6d5a`
- Same seed/cache/order across architectures

- independent 4×2 historical overlap: 5 epochs; max |Δ train loss| `0.000e+00`, validation loss `0.000e+00`, direct JS `0.000e+00`, direct Intersection `0.000e+00`; 1e-12 consistency: **PASS**
- joint 16 historical overlap: 6 epochs; max |Δ train loss| `0.000e+00`, validation loss `0.000e+00`, direct JS `0.000e+00`, direct Intersection `0.000e+00`; 1e-12 consistency: **PASS**

## Strict per-epoch trajectories

### independent 4×2

| Epoch | train loss | val loss | strict JS | strict Intersection | direct JS diagnostic | direct Intersection | AMP skips |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.417203455 | 0.420215057 | 0.086614964102 | 0.923563566214 | 0.105807774492 | 0.917536722351 | 0 |
| 2 | 0.354823133 | 0.437378874 | 0.072306914395 | 0.962942038782 | 0.069918887331 | 0.965298332072 | 0 |
| 3 | 0.325545862 | 0.423863721 | 0.103827409653 | 0.893015258400 | 0.101904281350 | 0.895897885780 | 0 |
| 4 | 0.300488840 | 0.465834169 | 0.085150601530 | 0.934075965811 | 0.076350187146 | 0.938479288180 | 0 |
| 5 | 0.276554567 | 0.478120165 | 0.097811914057 | 0.910105744516 | 0.089013962745 | 0.913854589892 | 0 |

- Strict best epoch: 2 (old provisional best epoch: 2; same)
- Early stopping/completion epoch: 5
- AMP skipped optimizer steps: 0; maximum consecutive: 0
- Saved/post-hoc verified epoch checkpoints: 5/5
- `best.pt` SHA-256: `c5b4e557cdc3b142be7a297990d7260f36cacb4a318da854c3bc540e3dc8eee0`
- `last.pt` SHA-256: `742db729fb9663a3c98a2689c2c5996cda4f4490731504d14e2967c5469fa5bb`

### joint 16

| Epoch | train loss | val loss | strict JS | strict Intersection | direct JS diagnostic | direct Intersection | AMP skips |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.739575902 | 0.678049305 | 0.151918475533 | 0.934777992486 | 0.171412893895 | 0.930033032516 | 0 |
| 2 | 0.624691006 | 0.665368636 | 0.156137835363 | 0.909331456434 | 0.173858494783 | 0.907122595000 | 0 |
| 3 | 0.586426087 | 0.648616876 | 0.150083204504 | 0.921791174179 | 0.168750230719 | 0.919373586922 | 0 |
| 4 | 0.555098545 | 0.665722816 | 0.185407089600 | 0.844698080578 | 0.198915883767 | 0.843474703745 | 0 |
| 5 | 0.525870705 | 0.696162648 | 0.166864043843 | 0.881655580476 | 0.181057723094 | 0.880522725986 | 1 |
| 6 | 0.496279742 | 0.705114728 | 0.199298206865 | 0.808599581785 | 0.211797524186 | 0.806312679834 | 1 |

- Strict best epoch: 3 (old provisional best epoch: 3; same)
- Early stopping/completion epoch: 6
- AMP skipped optimizer steps: 2; maximum consecutive: 1
- Saved/post-hoc verified epoch checkpoints: 6/6
- `best.pt` SHA-256: `f27fd9e0fe02f742b45c6158d228d0879d8cf375ec86e8dcc92b1362ecef0958`
- `last.pt` SHA-256: `11c1985a8932667403af4a8d9745b63d488df8c7b7efcf02310ab53a451f1af8`

## Previous direct-selected run comparison

- Old independent epoch 2 strict reference: JS 0.072306914395 / Intersection 0.962942038782
- Old joint epoch 3 strict reference: JS 0.150083204504 / Intersection 0.921791174179
- The old values were comparison-only and did not alter this run's config or stopping decisions.

## AMP behavior

A focused CUDA test verified finite forward loss, optimizer-step skip, parameter non-update, and GradScaler backoff on an injected overflow. The same path logs and continues transient skips for both architectures. Eight consecutive skips are treated as persistent numerical failure and abort the run.

## Strict validation integrity

- Every training-time strict metric was reproduced post-hoc from its saved epoch checkpoint within 1e-12.
- Strict path: cached Stage 1 IDs → Stage 2 Pedal1–4 replacement → official ids_to_midi/ref → map_midi → MIDI dump/reload → midi_to_ids → threshold 64 → global 16-pattern metric.
- All 19 piece evaluations required exact non-pedal MIDI equality.
- Original PT strict reference reproduced: JS 0.144081839387 / Intersection 0.907531644596.
- Stage 1 inference regeneration: 0
- ASAP test MIDI access: **0 / PASS**

## Final lock

A new `analysis/stage2_binary_v0/final_lock_v1/final_experiment_lock.json` was created. `final_lock_v0` was not overwritten.

## Stop point

Definitive strict training, per-epoch post-hoc validation, report, and lock only. No ASAP test evaluation, calibration, sampling, threshold change, or architecture/hyperparameter experiment was performed.
