# Raw Huber + Auxiliary CE Clean Rerun Report

- Primary target/loss: unquantized cached raw CC64 / 127, Smooth L1 delta 14/127
- Auxiliary target/loss: canonical bins 0–25/26–63/64–103/104–127, unweighted Standard CE
- Total objective: Huber + 0.1 × CE; checkpoint selection: minimum validation total
- Architecture: pretrained PT encoder + four Linear(768,1) regression heads + four fresh Linear(768,4) auxiliary heads
- Inference: regression output only; auxiliary candidate-MIDI usage count 0
- Fresh seed 42 initialization; no Raw/Representative/CE/Weighted checkpoint warm-start
- Data: MAESTRO-clean 1,170 + ASAP train 892; ASAP validation 71
- AdamW: encoder LR 1e-5, both head types LR 1e-4, weight decay 0.01; batch 16; accumulation 1; FP16 AMP
- Hybrid v0: interrupted during epoch 4 because a non-finite unscaled gradient was fatal before scaler.step/update
- v0 audit: objectives/model/checkpoints finite; optimizer/GradScaler/RNG absent; exact original event cause not provable
- Hybrid v1: clean seed-42 rerun from official PT pretrained initialization; old checkpoint warm-start false
- Modeling hyperparameters changed from v0: none; AMP recovery/checkpoint infrastructure corrected
- Best epoch: 5; best validation total: 0.2552403667198705
- AMP overflow events / max consecutive: 4 / 1
- Forward/model-parameter non-finite counts: 0 / 0
- AMP event details: `[{'affected_groups': ['regression_heads'], 'ce': 1.0197780132293701, 'consecutive_overflows': 1, 'epoch': 4, 'global_optimizer_attempt': 8006, 'grad_norm': inf, 'huber': 0.16520091891288757, 'offending_parameter_count': 2, 'offending_parameter_names': ['regression_heads.0.weight', 'regression_heads.3.weight'], 'optimizer_step_skipped': True, 'overflow_events_total': 1, 'scale_after': 8192.0, 'scale_before': 16384.0, 'scaler_step_update_executed': True, 'step': 1334, 'total': 0.2671787142753601}, {'affected_groups': ['regression_heads'], 'ce': 0.7707827687263489, 'consecutive_overflows': 1, 'epoch': 5, 'global_optimizer_attempt': 10047, 'grad_norm': inf, 'huber': 0.12408579885959625, 'offending_parameter_count': 2, 'offending_parameter_names': ['regression_heads.2.weight', 'regression_heads.3.weight'], 'optimizer_step_skipped': True, 'overflow_events_total': 2, 'scale_after': 8192.0, 'scale_before': 16384.0, 'scaler_step_update_executed': True, 'step': 1151, 'total': 0.20116406679153442}, {'affected_groups': ['regression_heads'], 'ce': 0.7649747133255005, 'consecutive_overflows': 1, 'epoch': 6, 'global_optimizer_attempt': 12469, 'grad_norm': inf, 'huber': 0.11739889532327652, 'offending_parameter_count': 2, 'offending_parameter_names': ['regression_heads.0.weight', 'regression_heads.1.weight'], 'optimizer_step_skipped': True, 'overflow_events_total': 3, 'scale_after': 8192.0, 'scale_before': 16384.0, 'scaler_step_update_executed': True, 'step': 1349, 'total': 0.1938963681459427}, {'affected_groups': ['regression_heads'], 'ce': 0.8772043585777283, 'consecutive_overflows': 1, 'epoch': 8, 'global_optimizer_attempt': 16486, 'grad_norm': inf, 'huber': 0.12448064982891083, 'offending_parameter_count': 2, 'offending_parameter_names': ['regression_heads.2.weight', 'regression_heads.3.weight'], 'optimizer_step_skipped': True, 'overflow_events_total': 4, 'scale_after': 16384.0, 'scale_before': 32768.0, 'scaler_step_update_executed': True, 'step': 918, 'total': 0.2122010886669159}]`
- Checkpoint continuation state: model, optimizer, GradScaler, counters, Python/NumPy/Torch CPU/CUDA RNG
- Checkpoints: `/workspace/project/analysis/stage2_encoder_only_raw_huber_aux_ce_v1/best.pt`, `/workspace/project/analysis/stage2_encoder_only_raw_huber_aux_ce_v1/last.pt`
- ASAP test access: 0; Repedal execution: 0

## Epochs

| Epoch | Train Huber | Train CE | Train Total | Val Huber | Val CE | Val Total | Val Raw MAE | AMP overflows | Scaler | Seconds |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.194840184 | 1.021242746 | 0.296964461 | 0.185797645 | 0.942129030 | 0.280010550 | 29.005453 | 0 | 2048.0 | 760.3 |
| 2 | 0.157398359 | 0.915692592 | 0.248967620 | 0.172990534 | 0.908583049 | 0.263848841 | 26.816714 | 0 | 4096.0 | 760.0 |
| 3 | 0.144271604 | 0.874937335 | 0.231765339 | 0.172528652 | 0.915073986 | 0.264036052 | 27.339880 | 0 | 8192.0 | 760.5 |
| 4 | 0.134651093 | 0.843005059 | 0.218951601 | 0.181454611 | 0.949513446 | 0.276405959 | 26.805477 | 1 | 8192.0 | 760.2 |
| 5 | 0.125923337 | 0.813111424 | 0.207234481 | 0.164449677 | 0.907906877 | 0.255240367 | 25.638127 | 1 | 8192.0 | 760.3 |
| 6 | 0.117476835 | 0.780773258 | 0.195554163 | 0.163342290 | 0.920158109 | 0.255358102 | 25.839022 | 1 | 8192.0 | 760.5 |
| 7 | 0.109302415 | 0.745948330 | 0.183897249 | 0.168118628 | 0.953195689 | 0.263438199 | 25.742546 | 0 | 16384.0 | 760.8 |
| 8 | 0.101952900 | 0.711694614 | 0.173122363 | 0.170361451 | 0.995402547 | 0.269901708 | 26.168471 | 1 | 16384.0 | 760.2 |
