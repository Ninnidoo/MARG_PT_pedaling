# Raw CC64 Huber Training Report

- Objective: Smooth L1/Huber on cached raw Pedal1–4 CC64 values normalized by 127
- Target quantization: none; no canonical-class or representative mapping in training targets
- Architecture: pretrained PT 10-layer encoder + four independent Linear(768,1) heads
- Training output: unconstrained scalar; no sigmoid and no loss-time clipping
- Huber delta: 14 CC64; normalized delta 14/127
- Data: MAESTRO-clean 1,170 + ASAP train 892; ASAP validation 71
- Seed: 42; fresh pretrained initialization; Representative Huber checkpoint initialization: false
- Optimizer: AdamW; encoder LR 1e-5; head LR 1e-4; weight decay 0.01
- Window/stride: 512/256; micro/effective batch: 16/16; accumulation: 1; FP16 AMP
- Best epoch: 6
- Best validation Huber: 0.16372716573372842
- Checkpoints: `/workspace/project/analysis/stage2_encoder_only_raw_cc64_huber_v0/best.pt`, `/workspace/project/analysis/stage2_encoder_only_raw_cc64_huber_v0/last.pt`
- ASAP test access count: 0

## Epochs

| Epoch | Train Huber | Validation Huber | Validation Raw MAE | Validation Raw RMSE | Encoder LR | Head LR | Seconds |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.195464659 | 0.185245558 | 28.876788 | 42.635861 | 1e-05 | 0.0001 | 756.7 |
| 2 | 0.158305071 | 0.175521161 | 28.209968 | 40.802917 | 1e-05 | 0.0001 | 756.9 |
| 3 | 0.145856750 | 0.176316114 | 28.103459 | 42.177083 | 1e-05 | 0.0001 | 757.2 |
| 4 | 0.136320044 | 0.177192947 | 26.614152 | 43.038197 | 1e-05 | 0.0001 | 757.1 |
| 5 | 0.127967804 | 0.168383666 | 25.583528 | 41.713494 | 1e-05 | 0.0001 | 757.0 |
| 6 | 0.120151061 | 0.163727166 | 25.603756 | 40.857175 | 1e-05 | 0.0001 | 757.4 |
| 7 | 0.112915369 | 0.170656762 | 25.881713 | 41.873793 | 1e-05 | 0.0001 | 757.2 |
| 8 | 0.105929296 | 0.168489954 | 25.597604 | 41.880453 | 1e-05 | 0.0001 | 757.4 |
| 9 | 0.099364970 | 0.170330252 | 25.784870 | 42.249873 | 1e-05 | 0.0001 | 757.7 |
