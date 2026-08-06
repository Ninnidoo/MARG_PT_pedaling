# Encoder-only CDF Ordinal Loss Sweep

## 1. Scope and hypothesis

Validation-only controlled comparison of CE against CE plus a distance-aware ordinal objective; the test split was not accessed.

## 2. Loss definition

The auxiliary objective is the mean squared difference between predicted and target CDFs over thresholds 0–126 (squared-CDF/Cramer-style ordinal loss), added to unweighted CE with lambda 0.1, 0.5, or 1.0.

## 3. Implementation and unit tests

Float32 softmax/CDF math, ignore-index masking, separate CE/ordinal/total logging, exact epoch-boundary resume, and initialization hashing were implemented. All 83 prelaunch tests across the eight required modules passed.

## 4. Pedal-rich overfit validation
`{"active_encoder_changed": true, "active_encoder_parameter": "embeddings.word_embeddings.weight", "all_metrics_finite": true, "encoder_gradient_finite_nonzero": true, "every_head_parameter_group_changed": true, "every_head_received_finite_nonzero_gradient": true, "final": {"ce_loss": 0.027558820322155952, "exact_note_accuracy": 1.0, "intermediate_argmax_prediction_ratio": 1.0, "intermediate_target_mae": 0.0, "intermediate_top10_recall": 1.0, "mae": 0.0, "mean_intermediate_true_class_rank": 1.0, "ordinal_loss": 0.00013702677097171545, "total_loss": 0.027695847675204277}, "initial": {"ce_loss": 5.392951488494873, "exact_note_accuracy": 0.0, "intermediate_argmax_prediction_ratio": 0.9945235848426819, "intermediate_target_mae": 40.612815856933594, "intermediate_top10_recall": 0.09104600219058051, "mae": 43.4549560546875, "mean_intermediate_true_class_rank": 62.184419496166484, "ordinal_loss": 0.16568128764629364, "total_loss": 5.558632850646973}, "oom": false, "passed": true, "performance_path": "Schubert/Piano_Sonatas/664-3/Lin07.mid", "steps": 30, "window_starts": [0, 256, 512, 768]}`

## 5. Training controls and initialization verification

All ordinal runs used the same official PT checkpoint, seed 20260710, four Linear(768,128) heads, unfrozen encoder, batch 16, encoder/head learning rates 1e-5/1e-4, weight decay 0.01, AMP scale 1024, and identical recorded initial parameter hashes.

## 6. Per-lambda training summaries

```json
{
  "lambda_0p1": {
    "best_epoch": 2,
    "completed_epochs": 6,
    "peak_memory_bytes": 9375656960,
    "runtime_seconds": 1495.0618566274643,
    "stop_reason": "early_stopping",
    "validation_ce_loss": 2.2458755153893097,
    "validation_ordinal_loss": 0.1517353158769006,
    "validation_total_loss": 2.26104905388572
  },
  "lambda_0p5": {
    "best_epoch": 2,
    "completed_epochs": 6,
    "peak_memory_bytes": 9375132672,
    "runtime_seconds": 1494.9718978777528,
    "stop_reason": "early_stopping",
    "validation_ce_loss": 2.2458088477600042,
    "validation_ordinal_loss": 0.1517302873155624,
    "validation_total_loss": 2.3216739857572795
  },
  "lambda_1p0": {
    "best_epoch": 2,
    "completed_epochs": 6,
    "peak_memory_bytes": 9389157376,
    "runtime_seconds": 1495.3910751864314,
    "stop_reason": "early_stopping",
    "validation_ce_loss": 2.245311951814202,
    "validation_ordinal_loss": 0.15158698553001726,
    "validation_total_loss": 2.396898931165352
  }
}
```

## 7. Full-performance validation comparison

| Model | Decoder | MAE | Int. MAE | Int. ratio | Collapse | Top-10 | Transition F1 | Pass |
|---|---|---:|---:|---:|---:|---:|---:|:---:|
| ce_baseline | argmax | 30.8654 | 51.5212 | 0.0000 | 1.0000 | 0.1498 | 0.1161 | no |
| ce_baseline | posterior_median | 28.1973 | 36.2451 | 0.3345 | 0.5101 | 0.1498 | 0.3051 | no |
| ordinal_0p1 | argmax | 30.8631 | 51.5217 | 0.0000 | 1.0000 | 0.1497 | 0.1165 | no |
| ordinal_0p1 | posterior_median | 28.1890 | 36.2266 | 0.3347 | 0.5098 | 0.1497 | 0.3052 | no |
| ordinal_0p5 | argmax | 30.8544 | 51.5010 | 0.0000 | 1.0000 | 0.1501 | 0.1185 | no |
| ordinal_0p5 | posterior_median | 28.1721 | 36.1775 | 0.3344 | 0.5098 | 0.1501 | 0.3061 | no |
| ordinal_1p0 | argmax | 30.8311 | 51.4917 | 0.0000 | 1.0000 | 0.1502 | 0.1208 | no |
| ordinal_1p0 | posterior_median | 28.1382 | 36.0855 | 0.3357 | 0.5085 | 0.1502 | 0.3065 | no |

Micro and performance-macro metrics, per-slot accuracy, tolerances, coarse confusion, posterior ranks, and loss components are in `validation_comparison.csv`.

## 8. Intermediate-depth results

The table and CSV quantify true-class ordering, endpoint collapse, posterior concentration, and tolerance accuracy for both decoders.

## 9. Transition results

Transition/steady accuracy and MAE, detection precision/recall/F1, direction accuracy, and delta MAE are recorded for every pair.

## 10. Success-criteria assessment

The predeclared A/B thresholds and all safeguards were applied without comparing total losses across lambdas.

## 11. Selected model and decoder

CE posterior-median baseline.

## 12. Interpretation

- Intermediate ordering improved: **yes**, based jointly on mean true-class rank and top-10 recall versus CE posterior median.
- Intermediate classes began winning argmax more often: **no**.
- Posterior median remains necessary because no ordinal pair passed selection.
- The predefined safeguards did not establish improved localization without endpoint mixing.
- Best ordinal trade-off by validation MAE: **ordinal_1p0 / posterior_median**; the predeclared rule, not total-loss scale, controlled final selection.
- Test evaluation justified: **no**.

## 13. Recommended next step

Do not tune more lambdas immediately; pursue MAESTRO continuous-pedal pretraining or a redesigned coarse-to-fine/ordinal output head.

## 14. Limitations

ASAP validation has 71 performances from 19 pieces; overlap reconstruction averages raw logits before one decode per original note. No test estimate, new calibration, audio, MIDI, or generation was produced.
