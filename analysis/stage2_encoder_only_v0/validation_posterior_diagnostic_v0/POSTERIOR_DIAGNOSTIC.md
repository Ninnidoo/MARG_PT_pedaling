# Experiment 1 v0 Validation Posterior Diagnostic

## 1. Scope

This is a validation-only posterior diagnostic. It uses Oracle human non-pedal performance inputs, does not read test targets, does not retrain or modify the model, and is not final model evaluation or end-to-end Stage 1 inference.

## 2. Checkpoint and environment

- Best checkpoint: `/workspace/project/analysis/stage2_encoder_only_v0/train_v0/best.pt`
- Best epoch: 2
- Recorded best validation loss: 2.2459586270
- Exact state load: missing keys=[], unexpected keys=[]
- Architecture: official PT encoder + four independent Linear(768, 128) heads
- Model state: `eval()`, FP16 autocast forward, `inference_mode()`, no gradients, no optimizer, and no parameter-version changes.
- Project commit: `1d64469760e963492897723c92e61ddad7c76a54`
- PT commit: `747df2d12291e37f6638b39f1b71517e579ad48c`
- GPU: host GPU 1 UUID `GPU-6982dbee-fbaf-f359-d7ef-a22d0e83400b` -> container GPU 0 -> PyTorch `cuda:0` (NVIDIA GeForce RTX 2080 Ti)
- Batch size: 16
- Preparation: 57.083 s; model load: 1.235 s; inference/statistics: 12.713 s; total: 71.409 s
- Peak allocated GPU memory: 1020216832 bytes (0.950 GiB)

Checkpoint configuration:

| Key | Value |
|---|---|
| allow_existing_log_dir | True |
| amp_enabled | True |
| amp_init_scale | 1024.0 |
| asap_root | /workspace/public/ASAP/asap-dataset-v1.1 |
| batch_size | 16 |
| checkpoint_boundary_note | Checkpoints are exact at completed epoch boundaries. An interruption within an epoch restarts that epoch. |
| checkpoint_path | /workspace/project/checkpoints/pianist_transformer |
| early_stopping_min_delta | 0.0001 |
| early_stopping_patience | 4 |
| encoder_lr | 1e-05 |
| expected_gpu_uuid | GPU-6982dbee-fbaf-f359-d7ef-a22d0e83400b |
| freeze_encoder | False |
| head_lr | 0.0001 |
| max_epochs | 20 |
| max_grad_norm | 1.0 |
| num_workers | 0 |
| output_dir | /workspace/project/analysis/stage2_encoder_only_v0/train_v0 |
| pianist_transformer_commit | 747df2d12291e37f6638b39f1b71517e579ad48c |
| pin_memory | True |
| project_git_commit | 1d64469760e963492897723c92e61ddad7c76a54 |
| resume | None |
| seed | 20260710 |
| split_csv | /workspace/project/analysis/stage2_encoder_only_v0/asap_split.csv |
| validation_overlap_note | Validation metrics count overlapping windows independently. Overlap-aware test reconstruction is deferred to a later task. |
| weight_decay | 0.01 |

## 3. Validation-set verification

- Validation performances: 71 (expected 71)
- Unique pieces: 19 (expected 19)
- Complete notes: 283928
- Complete pedal targets: 1135712
- Deterministic windows: 1078 (expected 1,078)
- Unique paths: 71; all MIDI files existed and tokenized successfully.
- Only validation rows were selected. Train tokens were read only to compute per-class train support; no test MIDI or test target was read.
- Contribution range: 1–3; first and last notes were covered, and reversing window order produced identical mean logits.
- Final statistics count each of the 283928 original notes exactly once.

## 4. Argmax reproduction

| Loss | Token accuracy | Exact-note accuracy | P1 | P2 | P3 | P4 | MAE |
|---|---|---|---|---|---|---|---|
| 2.255790 | 0.570974 | 0.517885 | 0.569489 | 0.568870 | 0.572423 | 0.573114 | 30.865419 |

These finite full-performance metrics may differ slightly from the training-time validation metrics because training counted overlapping windows independently, whereas this diagnostic averages logits and counts each complete note once.

## 5. Endpoint versus intermediate probability mass

Softmax is applied in float32 only after mean-logit reconstruction. Entropy is natural-log entropy in nats.

| Target | N | Mean p0 | Median p0 | Mean p-int | Median p-int | p-int P10 | P50 | P90 | Mean p127 | Best-int prob. | Endpoint prob. | Endpoint margin | p-int>0.25 | Best int beats endpoints | Entropy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ZERO | 284410 | 0.461679 | 0.467619 | 0.289664 | 0.261259 | 0.137099 | 0.261259 | 0.487915 | 0.248658 | 0.008072 | 0.710336 | 0.573663 | 0.535607 | 0.000000 | 2.143446 |
| INTERMEDIATE | 355418 | 0.280651 | 0.203227 | 0.386724 | 0.358429 | 0.161898 | 0.358429 | 0.665434 | 0.332625 | 0.011577 | 0.613276 | 0.500284 | 0.724389 | 0.000000 | 2.534947 |
| FULL | 495884 | 0.095447 | 0.032529 | 0.248006 | 0.202899 | 0.073556 | 0.202899 | 0.496598 | 0.656548 | 0.008592 | 0.751994 | 0.670866 | 0.396437 | 0.000000 | 1.755121 |

Intermediate-mass thresholds by target group:

| Target | p-int > 0.10 | p-int > 0.25 | p-int > 0.50 | Mean int-vs-endpoint logit margin |
|---|---|---|---|---|
| ZERO | 0.965258 | 0.535607 | 0.091150 | -4.349570 |
| INTERMEDIATE | 0.975038 | 0.724389 | 0.271354 | -3.844938 |
| FULL | 0.805509 | 0.396437 | 0.097995 | -4.490241 |

On intermediate targets, mean p-intermediate is 0.386724; under the predeclared descriptive 0.10 threshold used in Section 13, this is meaningful/nontrivial. The best intermediate class beats both endpoint classes for 0.000000 of intermediate targets; the mass and logit-margin tables show whether remaining support sits below endpoint modes.

## 6. True-class rank and coarse three-way classification

| Target | Mean rank | Median rank | Top1 | Top2 | Top5 | Top10 | Top20 | Mean true p | Mean NLL |
|---|---|---|---|---|---|---|---|---|---|
| OVERALL | 13.494690 | 1.000000 | 0.570974 | 0.687846 | 0.706904 | 0.733238 | 0.781644 | 0.403864 | 2.255790 |
| ZERO | 1.400000 | 1.000000 | 0.679153 | 0.994775 | 0.996765 | 0.997556 | 0.998245 | 0.461679 | 1.027845 |
| INTERMEDIATE | 40.485265 | 35.000000 | 0.000000 | 0.007467 | 0.066401 | 0.149762 | 0.303769 | 0.005053 | 5.640107 |
| FULL | 1.086373 | 1.000000 | 0.918166 | 0.999462 | 0.999728 | 0.999839 | 0.999925 | 0.656548 | 0.534406 |

Intermediate-only ordering:

| Correct is best intermediate | Best-intermediate MAE | MAE if true top5 | MAE if true top10 | Mean rank among intermediates | Intermediate top5 | Intermediate top10 |
|---|---|---|---|---|---|---|
| 0.023572 | 23.338430 | 6.582839 | 9.629763 | 38.555062 | 0.098160 | 0.179755 |

Posterior-mass group prediction confusion (rows=true; columns=ZERO, INTERMEDIATE, FULL):

| True group | Pred ZERO | Pred INTERMEDIATE | Pred FULL |
|---|---|---|---|
| ZERO | 163388 | 46419 | 74603 |
| INTERMEDIATE | 104851 | 126667 | 123900 |
| FULL | 28050 | 65956 | 401878 |

Collapsed 128-class argmax confusion (rows=true; columns=ZERO, INTERMEDIATE, FULL):

| True group | Pred ZERO | Pred INTERMEDIATE | Pred FULL |
|---|---|---|---|
| ZERO | 193158 | 0 | 91252 |
| INTERMEDIATE | 153389 | 0 | 202029 |
| FULL | 40580 | 0 | 455304 |

| Grouping method | Accuracy | Macro F1 | Balanced accuracy | ZERO F1 | INTERMEDIATE F1 | FULL F1 |
|---|---|---|---|---|---|---|
| Posterior summed mass | 0.609250 | 0.574021 | 0.580432 | 0.562729 | 0.426158 | 0.733177 |
| Collapsed 128-class argmax | 0.570974 | 0.653498 | 0.532440 | 0.575271 | NA | 0.731724 |

| Grouping method | Group | Precision | Recall | F1 |
|---|---|---|---|---|
| Posterior summed mass | ZERO | 0.551448 | 0.574481 | 0.562729 |
| Posterior summed mass | INTERMEDIATE | 0.529894 | 0.356389 | 0.426158 |
| Posterior summed mass | FULL | 0.669372 | 0.810427 | 0.733177 |
| Collapsed 128-class argmax | ZERO | 0.498953 | 0.679153 | 0.575271 |
| Collapsed 128-class argmax | INTERMEDIATE | NA | 0.000000 | NA |
| Collapsed 128-class argmax | FULL | 0.608220 | 0.918166 | 0.731724 |

The three-way result is a broad-region diagnostic, not a replacement for the 128-class prediction task.

## 7. Alternative decoding comparison

Posterior mean uses float32 expectation followed by NumPy `rint` (round-to-nearest, ties-to-even) and clipping to [0,127]. Posterior median is the smallest class whose cumulative mass is at least 0.5. A posterior mean near 64 can reflect a 0/127 mixture rather than learned half-pedal support.

| Decoder | Exact acc. | Exact note | MAE | RMSE | Tol±5 | Tol±10 | Tol±20 | Int. exact | Int. MAE | Int. tol±10 | Pred int. | Int. endpoint collapse | Transition acc. | Transition recall | Transition F1 | Delta MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Argmax | 0.570974 | 0.517885 | 30.865419 | 54.070801 | 0.585199 | 0.591417 | 0.611740 | 0.000000 | 51.521203 | 0.065323 | 0.000000 | 1.000000 | 0.106995 | 0.076667 | 0.116055 | 7.839006 |
| Posterior mean | 0.003725 | 0.000049 | 30.741051 | 41.168492 | 0.129547 | 0.272172 | 0.480094 | 0.011904 | 25.850846 | 0.259784 | 1.000000 | 0.000000 | 0.009571 | 0.907934 | 0.236224 | 6.167725 |
| Posterior median | 0.449330 | 0.396766 | 28.197258 | 47.035233 | 0.488864 | 0.516988 | 0.579221 | 0.006952 | 36.245111 | 0.182548 | 0.334542 | 0.510098 | 0.059504 | 0.582246 | 0.305097 | 7.033324 |

Additional decoder details:

| Decoder | P1 | P2 | P3 | P4 | Int tol±5 | Int tol±20 | Pred zero | Pred full | Steady acc. | Steady MAE | Transition MAE | Detection precision |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Argmax | 0.569489 | 0.568870 | 0.572423 | 0.573114 | 0.045454 | 0.130264 | 0.340867 | 0.659133 | 0.638899 | 27.483208 | 53.969438 | 0.238671 |
| Posterior mean | 0.003624 | 0.003846 | 0.003659 | 0.003772 | 0.135196 | 0.482727 | 0.000000 | 0.000000 | 0.002870 | 30.692540 | 31.072020 | 0.135774 |
| Posterior median | 0.448924 | 0.445997 | 0.451290 | 0.451111 | 0.107144 | 0.333492 | 0.199988 | 0.465470 | 0.506402 | 26.610721 | 39.033948 | 0.206705 |

Best-intermediate constrained diagnostic (not deployable because every prediction is forced into 1–126):

| Intermediate exact | Intermediate MAE | Tol±5 | Tol±10 | Tol±20 | Mean true intermediate rank | Top5 | Top10 |
|---|---|---|---|---|---|---|---|
| 0.023572 | 23.338430 | 0.179257 | 0.305668 | 0.530556 | 38.555062 | 0.098160 | 0.179755 |

## 8. Genuine intermediate support versus endpoint mixture

Fixed categories are: locally supported if local mass ±10 is at least 0.25; endpoint-mixture dominated if endpoint mass is at least 0.75 and local mass ±10 is below 0.25; diffuse/other otherwise.

| Scope | Mean-int predictions | Ratio | Locally supported N | Locally supported ratio | Local MAE | Endpoint-mixture N | Endpoint-mixture ratio | Mixture MAE | Diffuse N | Diffuse ratio | Diffuse MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| All targets | 1135712 | 1.000000 | 301675 | 0.265626 | 14.901258 | 292779 | 0.257793 | 39.109830 | 541258 | 0.476580 | 35.042634 |
| True intermediate targets | 355418 | 1.000000 | 63545 | 0.178789 | 26.074671 | 68156 | 0.191763 | 31.991079 | 223717 | 0.629448 | 23.916631 |

These fixed categories diagnose posterior shape; they are not tuned decoder hyperparameters.

## 9. Transition-conditioned analysis

Sequences are flattened within each performance only; boundaries are never joined.

| Subset | N | Ratio | p0 | p-int | p127 | True p | True rank | Entropy | Endpoint margin | Argmax acc. | Mean MAE | Median MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Transition | 145016 | 0.127695 | 0.264351 | 0.415475 | 0.320174 | 0.067908 | 34.099803 | 2.680212 | 0.468479 | 0.106995 | 31.072020 | 39.033948 |
| Steady | 990625 | 0.872305 | 0.242291 | 0.285213 | 0.472496 | 0.453050 | 10.477893 | 2.010932 | 0.611392 | 0.638899 | 30.692540 | 26.610721 |

Stale-state preference at true transitions:

| Transition group | N | Mean log p(prev)-log p(cur) | Median | p(prev)>p(cur) | Argmax=previous | Argmax=current |
|---|---|---|---|---|---|---|
| All | 145016 | -0.018057 | 0.004028 | 0.502200 | 0.100782 | 0.106995 |
| Upward | 77534 | -0.173986 | -0.039693 | 0.476385 | 0.072923 | 0.126512 |
| Downward | 67482 | 0.161099 | 0.060265 | 0.531860 | 0.132791 | 0.084571 |
| Endpoint To Endpoint | 2432 | 0.036813 | 0.006836 | 0.502467 | 0.502467 | 0.497533 |
| Involving Intermediate | 142584 | -0.018993 | 0.003998 | 0.502195 | 0.093931 | 0.100334 |

## 10. Slot-conditioned analysis

Slot transitions compare each Pedal slot across consecutive notes.

| Slot | Target 0 | Target int. | Target 127 | Argmax 0 | Argmax int. | Argmax 127 | Mean p-int | Int true rank | Argmax acc. | Mean MAE | Transition ratio | Stale p(prev)>p(cur) | Stale advantage | Argmax=previous | Argmax=current |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Pedal1 | 0.250490 | 0.314911 | 0.434600 | 0.343045 | 0.000000 | 0.656955 | 0.300647 | 40.436731 | 0.569489 | 30.716217 | 0.231669 | 0.486489 | -0.156968 | 0.157920 | 0.192485 |
| Pedal2 | 0.251687 | 0.314224 | 0.434089 | 0.344457 | 0.000000 | 0.655543 | 0.305847 | 40.482464 | 0.568870 | 30.810480 | 0.269833 | 0.488981 | -0.163620 | 0.155534 | 0.191307 |
| Pedal3 | 0.249782 | 0.311752 | 0.438467 | 0.340696 | 0.000000 | 0.659304 | 0.300405 | 40.469276 | 0.572423 | 30.645699 | 0.267747 | 0.496737 | -0.103759 | 0.158206 | 0.183429 |
| Pedal4 | 0.249739 | 0.310903 | 0.439358 | 0.335272 | 0.000000 | 0.664728 | 0.300498 | 40.553289 | 0.573114 | 30.791806 | 0.260610 | 0.500162 | -0.059272 | 0.162201 | 0.174219 |

## 11. Class-frequency analysis

| Summary | Classes (class:support or class:recall) |
|---|---|
| Highest validation support | 127:495884, 0:284410, 64:6243, 65:5940, 66:5912, 63:5646, 126:5642, 124:5437, 62:5396, 61:5048 |
| Highest train support among intermediate classes | 64:97136, 63:90315, 65:89720, 66:83358, 62:80467, 67:79035, 61:75291, 68:73054, 69:70360, 60:70101 |
| Highest intermediate argmax recall | 8:0.0000, 9:0.0000, 10:0.0000, 11:0.0000, 12:0.0000, 13:0.0000, 14:0.0000, 15:0.0000, 16:0.0000, 17:0.0000 |
| Intermediate classes never selected by argmax | 126 classes: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126 |

Natural-log train support versus mean true-class probability correlation: 0.714916. Natural-log train support versus argmax recall correlation: 0.698615. These correlations are descriptive, use classes supported in both train and validation, and do not establish causality or significance.

`nan` appears in the per-class CSV only when validation support is zero or best-intermediate recall is inapplicable to endpoint classes.

## 12. Performance-level variation

| Metric | 10th percentile | Median | 90th percentile |
|---|---|---|---|
| Argmax accuracy | 0.299725 | 0.575054 | 0.885556 |
| Argmax MAE | 7.508371 | 30.575389 | 47.110740 |
| Posterior-mean MAE | 15.450811 | 30.175384 | 39.713023 |
| Posterior-median MAE | 7.436269 | 27.440924 | 39.126471 |
| Intermediate p-mass on intermediate targets | 0.271027 | 0.379014 | 0.559955 |
| Intermediate true-class rank | 31.420690 | 39.964155 | 48.704587 |
| Transition stale-state fraction | 0.455259 | 0.498947 | 0.524062 |

These are descriptive validation-performance differences; no statistical significance is inferred.

## 13. Interpretation and decision

Primary diagnosis: **A. DECODING/CALIBRATION BOTTLENECK**.

mean intermediate mass=0.386724, mean true-class rank=40.485, best deterministic intermediate-MAE improvement=25.670, posterior-mean overall-MAE change=-0.124, posterior-median overall-MAE change=-2.668, stale-state fraction=0.502200, and best decoder transition recall=0.907934. For this diagnostic, a substantial decoder recovery is at least 5 pedal-value MAE units with no more than 2 units of overall-MAE degradation; these are practical descriptive thresholds, not significance tests.

## 14. Recommended next experiment

Run exactly one next primary experiment: **decoding/calibration study without retraining**. Define its architecture/loss and success criteria using train and validation only; do not tune it against the fixed test result. This task does not launch that experiment.

## 15. Limitations

- This is a validation-only model-selection diagnostic.
- Posterior mean can reflect endpoint uncertainty rather than learned intermediate depth.
- Targets use the four-point PT pedal tokenizer.
- No raw-CC64 timing reconstruction was performed.
- No end-to-end Stage 1 input was evaluated.
- No listening test was performed.
- No model retraining or calibration was performed.
