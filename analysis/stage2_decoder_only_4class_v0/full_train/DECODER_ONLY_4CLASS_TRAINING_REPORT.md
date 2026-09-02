# Decoder-only 4-Class Stage 2 Training Report

## A. Experiment setup

- Architecture: pretrained PT token/projection/note compression + fresh seed-42 2-layer self-attention Prefix-LM + shared Linear(768,4)
- Parameters: 23,621,380 total/trainable
- Pretrained: PT token embedding, eight feature projections, note compression
- Random initialization: two-layer Prefix-LM, pedal/BOS/PAD embedding, Pedal1–4 slot embedding, Linear(768,4)
- Classes: `['ZERO', 'LOW', 'HALF', 'FULL']`; boundaries: `[[0, 25], [26, 63], [64, 103], [104, 127]]`; representatives: `[0, 51, 79, 127]`
- Training: MAESTRO-clean 1,170 + ASAP train 892 = 2,062 performances / 9,369,095 notes / 35,573 windows
- Validation: ASAP validation 71 performances / 283,928 notes / 1,078 windows
- Shared cache ID: `85a79e9d10e955b10000f72f6bbc4a29dbd2cc5a4c8cfb1277054a8d45dcb877`
- Seed: 42; window/stride: 512/256
- Loss: standard unweighted cross entropy; teacher-forced training and validation
- Optimizer: AdamW; pretrained representation LR 1e-05; fresh Prefix-LM LR 0.0001; weight decay 0.01; scheduler none
- Batch: micro 4, accumulation 4, effective 16
- Precision: FP16 AMP; gradient clip 1.0
- Early stopping: minimum validation CE, patience 3, minimum delta 0.0001
- ASAP test access: 0

## B. Smoke-test provenance

- Pedal-rich tiny overfit: PASS
- Smoke teacher-forced accuracy: 99.8901%
- Smoke free-running greedy accuracy: 98.4863%
- Smoke checkpoint was not loaded or reused.

## C. Resume provenance

- Initial run: interrupted during epoch 9 by a non-finite unscaled aggregate gradient norm.
- Recovery checkpoint: `/workspace/project/analysis/stage2_decoder_only_4class_v0/full_train/best.pt`
- Model/optimizer/GradScaler restored: yes
- Resume epoch/global optimizer step: 8/17792
- RNG state restored: no (the epoch-8 checkpoint contains no RNG state)
- Bit-exact resume: no; this is a near-exact resume
- Modeling hyperparameters changed: none

## D. AMP diagnostics

- Recorded AMP gradient-overflow events after recovery instrumentation: 2
- Maximum consecutive recorded events: 1
- Forward non-finite count: 0
- Model-parameter non-finite count: 0
- The pre-resume failed epoch contained at least one overflow event, but its exact count and scale transition were not recorded by the old runner.
- Event: epoch=9 step=2214 global_attempt=20006 grad_norm=inf scale=1048576.0->524288.0 groups=['fresh_prefix_lm']
- Event: epoch=10 step=1996 global_attempt=22012 grad_norm=inf scale=1048576.0->524288.0 groups=['fresh_prefix_lm']

## E. Epoch table

| Epoch | Train CE | Validation CE | Validation TF accuracy | Best so far |
|---:|---:|---:|---:|---|
| 1 | 0.245682068 | 0.206245380 | 0.944184199 | yes |
| 2 | 0.231279488 | 0.203213525 | 0.944527536 | yes |
| 3 | 0.226950221 | 0.198713865 | 0.945038012 | yes |
| 4 | 0.219271505 | 0.191699459 | 0.945889560 | yes |
| 5 | 0.209649305 | 0.184522169 | 0.946743825 | yes |
| 6 | 0.199278399 | 0.178740538 | 0.947601262 | yes |
| 7 | 0.190934433 | 0.176789838 | 0.947655163 | yes |
| 8 | 0.183437686 | 0.175932018 | 0.947817772 | yes |
| 9 | 0.176151366 | 0.177280643 | 0.947551437 | no |
| 10 | 0.168891382 | 0.177707741 | 0.947490742 | no |

## F. Final training result

- Status: completed
- Best epoch: 8
- Best validation CE: 0.17593201818653073
- Early-stop/final epoch: 10
- Best checkpoint: `/workspace/project/analysis/stage2_decoder_only_4class_v0/full_train/best.pt`
- Last checkpoint: `/workspace/project/analysis/stage2_decoder_only_4class_v0/full_train/last.pt`

## G. Training-curve interpretation

- Train CE moved from 0.245682 to 0.168891; validation CE moved from 0.206245 to 0.177708.
- Interpretation is limited to observed convergence and numerical behavior; no architecture ranking is made.

## H. Pending evaluation

The next stage must evaluate `best.pt` with true free-running greedy inference and the frozen canonical evaluator. No full-validation free-running inference or ASAP test evaluation was performed during this run.
