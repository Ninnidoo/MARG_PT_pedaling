# Binary Encoder-Decoder Canonical-v1 Training Report

## A. Setup

- Training data: MAESTRO-clean 1,170 + ASAP train 892 = 2,062 performances / 9,369,095 notes / 35,573 windows (natural concatenation).
- Shared cache ID: `85a79e9d10e955b10000f72f6bbc4a29dbd2cc5a4c8cfb1277054a8d45dcb877`
- Seed 42; micro-batch 4; gradient accumulation 4; effective batch 16; window/stride 512/256.
- AdamW: encoder LR 1e-5, decoder/head LR 1e-4, weight decay 0.01; gradient clip 1.0; AMP enabled.
- Architecture: official trainable 10-layer PT encoder (hidden 768), native two-layer causal T5GemmaDecoder with cross-attention, shared Linear(768,2).
- Decoder vocabulary: OFF=0, ON=1, BOS=2, PAD=3; note-major P1,P2,P3,P4 target order.
- Training uses standard teacher forcing and unweighted binary CE. Teacher-forced validation is diagnostic only.
- Checkpoint selection and patience-3 early stopping use minimum canonical free-running strict JS only; within 1e-12, higher Intersection then earliest epoch is the deterministic tie-break.
- Frozen canonical Stage 1 manifest: `/workspace/project/analysis/stage2_binary_canonical_v1/canonical_validation_stage1_manifest.csv`; Stage 1 neural inference regeneration: 0.
- Overlapping windows have independently reset autoregressive histories; logit averaging merges outputs, not decoder histories.

## B. Epoch trajectory

| Epoch | Train loss | TF val loss | TF acc | TF exact | Free-running JS ↓ | Intersection ↑ | Beats PT? | Beats encoder-only? |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 0.136347801 | 0.114326168 | 0.971187783 | 0.897625812 | 0.357291749303 | 0.607639763234 | no | no |
| 2 | 0.128406994 | 0.113128189 | 0.971192312 | 0.897671107 | 0.263769460180 | 0.719756142321 | no | no |
| 3 | 0.125905163 | 0.112420803 | 0.971246666 | 0.897806992 | 0.325463822163 | 0.634793266686 | no | no |
| 4 | 0.124228429 | 0.111546593 | 0.971214960 | 0.897698284 | 0.264025919144 | 0.754499453364 | no | no |
| 5 | 0.122545764 | 0.111872126 | 0.971115763 | 0.897642118 | 0.242492497946 | 0.740221874435 | no | no |
| 6 | 0.120832174 | 0.112197226 | 0.971087680 | 0.897613129 | 0.268950052700 | 0.727970631926 | no | no |
| 7 | 0.118903523 | 0.113087265 | 0.971060503 | 0.897542469 | 0.258148714364 | 0.733022479838 | no | no |
| 8 | 0.116743525 | 0.114482989 | 0.970776050 | 0.896892031 | 0.420564556228 | 0.530330777248 | no | no |

## C. Best validation comparison

| Model | Architecture | Best epoch | Validation JS ↓ | Intersection ↑ |
|---|---|---:|---:|---:|
| Original PT | original PT | — | 0.161498978464 | 0.821025730937 |
| Encoder-only Independent | encoder-only 4×2 | 2 | 0.150082839377 | 0.829325144704 |
| Binary Encoder-Decoder | autoregressive encoder-decoder | 5 | 0.242492497946 | 0.740221874435 |

## D. Transition diagnostics

| Source | Steady mass | Transition-containing mass |
|---|---:|---:|
| Human validation | 0.922976951903 | 0.077023048097 |
| Original PT | 0.952140388195 | 0.047859611805 |
| Encoder-only Independent epoch 2 | 0.925081299982 | 0.074918700018 |
| Binary Encoder-Decoder best | 0.965312007854 | 0.034687992146 |

## E. Locked best checkpoint

- Best epoch: 5
- `best.pt` SHA-256: `cb8a697a5220a6ca833e71ed4e235ac3824260f853fe43f764c10950cdfa46b9`
- Every executed epoch checkpoint plus `best.pt` and `last.pt` is retained and hashed in `checkpoint_manifest.csv`.

## F. Test isolation

ASAP test was not used for training, model selection, checkpoint selection, calibration, or evaluation in this stage.

- ASAP test Human MIDI access: 0
- Canonical test Stage 2 inference: 0/23
- Test JS / Intersection: not computed
- Test audio: not generated
- Scheduled sampling, class weighting, calibration, and post-processing: not used
