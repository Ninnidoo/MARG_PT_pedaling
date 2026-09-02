# Scheduled-sampling checkpoint epoch free-running validation report

## Scope and checkpoint availability

- Source run: `/workspace/project/analysis/stage2_encoder_decoder_5class_weighted_scheduled_sampling_v0`
- Reused baseline: `/workspace/project/analysis/stage2_encoder_decoder_5class_weighted_v0`
- Retained checkpoints: `best.pt = epoch 3`, `last.pt = epoch 7`
- Epoch 3 reuses the completed run's already-verified identical free-running validation result;
  epoch 7 is newly evaluated from `last.pt` in this analysis.
- Epoch 4, 5, and 6 checkpoints were not retained and were not reconstructed.
- ASAP validation only: 71 performances / 19 pieces / 283928 notes / 1078 windows.
- The unchanged 512-note, stride-256, Pedal1–4 flatten, five-class representatives
  [0, 32, 80, 111, 127], greedy autoregressive, overlap-logit averaging path was used.
- ASAP test rows used: 0. No training, sampling, beam search, smoothing, calibration, or post-processing.

## Main comparison

| model | epoch | p_tf | TF val CE | TF val acc | FR acc | FR macro F1 | LOW R | MID R | HIGH R | endpoint collapse | transition F1 | repedal F1 | MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline encoder-decoder | — | — | — | — | 0.493434 | 0.261422 | 0.080312 | 0.039482 | 0.000000 | 0.893621 | 0.003870 | 0.000000 | 40.500158 |
| scheduled | 3 | 0.90 | 0.282139 | 0.936252 | 0.480363 | 0.243676 | 0.000000 | 0.043143 | 0.011345 | 0.963755 | 0.002443 | 0.000000 | 43.970533 |
| scheduled | 4 | 0.80 | 0.282652 | 0.936307 | checkpoint unavailable | — | — | — | — | — | — | — | — |
| scheduled | 5 | 0.70 | 0.285945 | 0.935769 | checkpoint unavailable | — | — | — | — | — | — | — | — |
| scheduled | 6 | 0.60 | 0.290628 | 0.935614 | checkpoint unavailable | — | — | — | — | — | — | — | — |
| scheduled | 7 | 0.60 | 0.290398 | 0.935225 | 0.477572 | 0.248737 | 0.185121 | 0.044106 | 0.000000 | 0.802199 | 0.002598 | 0.000000 | 38.528520 |

## Per-class free-running results for retained scheduled checkpoints

| epoch | class | precision | recall | F1 | predicted count | predicted ratio |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 3 | ZERO | 0.360126 | 0.706779 | 0.477136 | 558180 | 0.491480 |
| 3 | LOW | 0.000000 | 0.000000 | 0.000000 | 0 | 0.000000 |
| 3 | MID | 0.212519 | 0.043143 | 0.071725 | 28054 | 0.024702 |
| 3 | HIGH | 0.394046 | 0.011345 | 0.022056 | 2284 | 0.002011 |
| 3 | FULL | 0.617107 | 0.680960 | 0.647463 | 547194 | 0.481807 |
| 7 | ZERO | 0.468895 | 0.239172 | 0.316768 | 145071 | 0.127736 |
| 7 | LOW | 0.169088 | 0.185121 | 0.176741 | 150975 | 0.132934 |
| 7 | MID | 0.288248 | 0.044106 | 0.076505 | 21145 | 0.018618 |
| 7 | HIGH | 0.000000 | 0.000000 | 0.000000 | 0 | 0.000000 |
| 7 | FULL | 0.540900 | 0.892826 | 0.673671 | 818521 | 0.720712 |

## Answers to the requested questions

1. Highest free-running token accuracy: epoch 3 (0.480363).
2. Highest free-running macro F1: epoch 7 (0.248737).
3. Teacher-forced CE best epoch 3 is not the sole free-running selection winner.
4. Among evaluable later checkpoints, epoch 7 had worse teacher-forced CE than epoch 3 and free-running token accuracy did not improve; macro F1 improved. Epochs 4–6 cannot be assessed because their checkpoints are unavailable.
5. From epoch 3 to 7, LOW recall/predicted share changed 0→0.185121 / 0→0.132934;
   MID changed 0.043143→0.044106 / 0.024702→0.018618; HIGH changed
   0.011345→0 / 0.002011→0. The later model recovered LOW, barely changed MID
   recall, and eliminated HIGH predictions while shifting heavily toward FULL.
6. Nonendpoint→endpoint collapse decreased from epoch 3 to 7.
7. HIGH→FULL collapse did not decrease from epoch 3 to 7.
8. Binary transition F1 improved (0.002443→0.002598), but the gain is tiny; tolerance
   F1 at ±1 decreased while ±2 and ±4 increased slightly. Short-repedal F1 did not
   improve (0 at both epochs).
9. CE-based selection missed a checkpoint that is better on macro F1, decoded MAE,
   endpoint collapse, and marginally transition F1, but epoch 7 is not a uniformly
   better model: token accuracy and weighted F1 declined, HIGH disappeared, and
   HIGH→FULL collapse worsened. Evidence that CE selection missed some useful
   free-running behavior is real but mixed, not enough to name epoch 7 an unequivocal
   overall best. Epochs 4–6 remain unknowable without forbidden retraining.

## Provenance

Full metrics, exact deltas, checkpoint inventory, validation accounting, and GPU/runtime
provenance are in `epoch_free_running_metrics.json`; the flattened comparison is in
`epoch_free_running_metrics.csv`.
