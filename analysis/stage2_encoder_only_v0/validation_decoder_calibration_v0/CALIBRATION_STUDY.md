# Experiment 1 v0 Validation Hierarchical Decoder Calibration

## 1. Scope

This is a validation-only, piece-level leave-one-piece-out calibration study using the fixed neural checkpoint and Oracle human non-pedal performance input. It does not read test targets, retrain the model, modify neural parameters, or perform end-to-end Stage 1 inference.

## 2. Motivation

The preceding posterior diagnostic found meaningful aggregate intermediate mass (mean 0.386724 on intermediate targets), weak exact intermediate ordering (mean rank 40.485), and endpoint-only ordinary argmax predictions. This study separates region choice from conditional intermediate depth.

## 3. Data and reconstruction verification

- Validation performances/pieces: 71/19
- Complete notes/targets/windows: 283928/1135712/1078
- Contribution range: 1–3; first/last coverage and reversed-window equality passed.
- Every validation MIDI existed and tokenized. Each note was counted once. Train data was used only for smoothed class counts. No test row, MIDI, target, or prior statistic was used.
- Checkpoint epoch/loss: 2/2.2459586270; exact load missing=[], unexpected=[].
- GPU: UUID GPU-6982dbee-fbaf-f359-d7ef-a22d0e83400b as cuda:0; batch size 16; FP16 model forward and float32 calibration.
- Timing: preparation 57.738s, model load 1.241s, reconstruction 13.131s, calibration/CV 6.235s, total 78.616s.
- Peak allocated GPU memory: 1020216832 bytes (0.950 GiB).

## 4. Candidate decoders

- Argmax, posterior mean, and posterior median reproduce the frozen 128-class posterior baselines.
- Raw hierarchy chooses argmax[p(0), sum p(1..126), p(127)], then uses conditional MAP, mean, or median for intermediate depth.
- Calibrated hierarchy constructs g=[z0, logsumexp(z1..z126), z127] and applies g/T + [0,b_intermediate,b_full] before region argmax.
- Conditional logits use z'_c=z_c-tau*log(pi_c) with add-one-smoothed train priors and tau in {0,.25,.50,.75,1}. Conditional mean uses NumPy rint; median is the smallest cumulative class at 0.5.

## 5. Cross-validation protocol

19 folds hold out one complete piece at a time. Region parameters and conditional tau/decoder are fit or selected using only the other pieces. Conditional selection minimizes intermediate MAE, treats values within 0.05 as tied and then minimizes overall MAE, then maximizes intermediate ±10 accuracy, then chooses smaller tau and MAP, median, mean in that order. Every performance appears in exactly one held-out fold.

## 6. Region calibration results

| Region system | OOF region NLL | Group accuracy | Macro F1 | Intermediate recall |
|---|---|---|---|---|
| uncalibrated | 0.829643 | 0.609250 | 0.574021 | 0.356389 |
| temperature | 0.825677 | 0.609250 | 0.574021 | 0.356389 |
| bias_temperature | 0.844227 | 0.599973 | 0.560859 | 0.353291 |

Original 128-class validation NLL: 2.255790. Region NLLs are three-class likelihoods and are not presented as 128-class likelihoods.

Fold parameter variation (P10/median/P90):

| Calibrator | T P10 | T median | T P90 | b-int P10 | b-int median | b-int P90 | b-full P10 | b-full median | b-full P90 |
|---|---|---|---|---|---|---|---|---|---|
| temperature | 1.205511 | 1.222978 | 1.233329 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| bias_temperature | 1.195137 | 1.220739 | 1.236528 | -0.051375 | -0.020461 | -0.005499 | -0.081120 | -0.029364 | 0.005885 |

raw_hierarchy_map confusion (rows true; columns ZERO/INTERMEDIATE/FULL):

| True | ZERO | INTERMEDIATE | FULL |
|---|---|---|---|
| ZERO | 163388 | 46419 | 74603 |
| INTERMEDIATE | 104851 | 126667 | 123900 |
| FULL | 28050 | 65956 | 401878 |

temperature_hierarchy confusion (rows true; columns ZERO/INTERMEDIATE/FULL):

| True | ZERO | INTERMEDIATE | FULL |
|---|---|---|---|
| ZERO | 163388 | 46419 | 74603 |
| INTERMEDIATE | 104851 | 126667 | 123900 |
| FULL | 28050 | 65956 | 401878 |

bias_temperature_hierarchy confusion (rows true; columns ZERO/INTERMEDIATE/FULL):

| True | ZERO | INTERMEDIATE | FULL |
|---|---|---|---|
| ZERO | 147266 | 51076 | 86068 |
| INTERMEDIATE | 96721 | 125566 | 133131 |
| FULL | 22560 | 64760 | 408564 |

## 7. Conditional intermediate-depth results

- temperature: tau selections 0:0, 0.25:0, 0.5:19, 0.75:0, 1:0; decoder selections map:0, median:0, mean:19.
- bias_temperature: tau selections 0:0, 0.25:0, 0.5:19, 0.75:0, 1:0; decoder selections map:0, median:0, mean:19.

| Candidate | Conditional rule | Mean tau | Int. MAE | Int. RMSE | Int. tol±10 | Endpoint collapse |
|---|---|---|---|---|---|---|
| temperature_hierarchy | fold-selected | 0.500000 | 38.537516 | 47.695246 | 0.184676 | 0.643611 |
| raw_hierarchy_mean | mean | 0.000000 | 38.597744 | 47.718780 | 0.182627 | 0.643611 |
| raw_hierarchy_median | median | 0.000000 | 38.601219 | 47.725539 | 0.183274 | 0.643611 |
| raw_hierarchy_map | map | 0.000000 | 39.094348 | 48.086972 | 0.177056 | 0.643611 |
| bias_temperature_hierarchy | fold-selected | 0.500000 | 38.360969 | 47.440425 | 0.185798 | 0.646709 |

## 8. Full candidate comparison

| Rank | Candidate | Exact | MAE | RMSE | Int. MAE | Int. tol±10 | Pred. int. | Group macro F1 | Transition F1 | Delta MAE | Success |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | posterior_median | 0.449330 | 28.197258 | 47.035233 | 36.245111 | 0.182548 | 0.334542 | 0.574162 | 0.305097 | 7.033324 | False |
| 2 | temperature_hierarchy | 0.499768 | 29.069928 | 49.974860 | 38.537516 | 0.184676 | 0.210478 | 0.574021 | 0.316339 | 8.074479 | False |
| 3 | raw_hierarchy_mean | 0.499659 | 29.157787 | 50.068384 | 38.597744 | 0.182627 | 0.210478 | 0.574021 | 0.310837 | 8.089453 | False |
| 4 | raw_hierarchy_median | 0.499711 | 29.157025 | 50.069553 | 38.601219 | 0.183274 | 0.210478 | 0.574021 | 0.312536 | 8.107063 | False |
| 5 | raw_hierarchy_map | 0.499944 | 29.235739 | 50.215577 | 39.094348 | 0.177056 | 0.210478 | 0.574021 | 0.318782 | 8.795781 | False |
| 6 | bias_temperature_hierarchy | 0.491456 | 29.895187 | 50.873966 | 38.360969 | 0.185798 | 0.212556 | 0.560859 | 0.309961 | 7.724855 | False |
| 7 | posterior_mean | 0.003725 | 30.741051 | 41.168492 | 25.850846 | 0.259784 | 1.000000 | 0.476710 | 0.236224 | 6.167725 | False |
| 8 | argmax | 0.570974 | 30.865419 | 54.070801 | 51.521203 | 0.065323 | 0.000000 | 0.653498 | 0.116055 | 7.839006 | False |

The candidate CSV contains every requested micro and macro metric, including slot, tolerance, coarse, and temporal measures.

`nan` is retained only for genuine zero-denominator group precision/F1 when a collapsed baseline never predicts a region; such a fold is explicitly marked non-finite and degenerate.

## 9. Performance and fold variation

| Selected-candidate fold metric | P10 | Median | P90 |
|---|---|---|---|
| Overall MAE | 11.903135 | 29.135931 | 38.305195 |
| Intermediate MAE | 20.993901 | 30.985363 | 46.418542 |
| Exact accuracy | 0.064373 | 0.300422 | 0.560042 |
| Predicted intermediate ratio | 0.117918 | 0.453281 | 0.783434 |
| Three-way macro F1 | 0.321627 | 0.482054 | 0.583942 |
| Transition F1 | 0.185433 | 0.324372 | 0.421034 |

Worst selected-candidate folds by overall MAE:

| Piece | Performances | MAE | Int. MAE | Macro F1 | Transition F1 |
|---|---|---|---|---|---|
| piece_ab42b317dcb38fc5 | 1 | 46.511698 | 29.410176 | 0.383180 | 0.263493 |
| piece_f10784ae79f0f210 | 1 | 46.073119 | 46.030690 | 0.069127 | 0.168498 |
| piece_e7ad71b998a8d477 | 5 | 36.363214 | 47.969953 | 0.370540 | 0.363080 |
| piece_69862af5096ee3fa | 7 | 35.261105 | 30.985363 | 0.509414 | 0.388847 |
| piece_e38184a2e9753bdc | 1 | 33.938729 | 30.628123 | 0.385363 | 0.490645 |

Fold differences are descriptive; no significance is claimed.

## 10. Success-criteria assessment

| Candidate | A | B | Ratio | Collapse | Macro F1 | Nondegenerate | Finite | Pass |
|---|---|---|---|---|---|---|---|---|
| posterior_median | False | False | True | False | True | True | True | False |
| temperature_hierarchy | False | False | True | False | True | True | True | False |
| raw_hierarchy_mean | False | False | True | False | True | True | True | False |
| raw_hierarchy_median | False | False | True | False | True | True | True | False |
| raw_hierarchy_map | False | False | True | False | True | True | True | False |
| bias_temperature_hierarchy | False | False | True | False | False | True | True | False |
| posterior_mean | False | False | False | True | False | False | True | False |
| argmax | False | False | False | False | True | False | False | False |

Criteria are the predeclared practical thresholds and are not statistical significance claims.

## 11. Final selected decoder

Selected: **posterior_median**.
No candidate met the success criteria; it was retained as the lowest-overall-MAE fallback. The within-0.10 intermediate-MAE, group-macro-F1, and simplicity tie rules were then applied.
- Full-validation region calibrator: none
- Temperature: 1.0000000000
- Biases [zero, intermediate, full]: [0, 0.0000000000, 0.0000000000]
- Conditional tau/decoder: 0.0/not_applicable
- Frozen configuration: /workspace/project/analysis/stage2_encoder_only_v0/validation_decoder_calibration_v0/calibration_config.json

## 12. Interpretation

- The best hierarchical candidate was temperature_hierarchy: MAE 29.069928 versus 28.197258 for posterior median.
- Temperature scaling improved region NLL from 0.829643 to 0.825677 but cannot change region argmax; bias+temperature reduced held-out macro F1 to 0.560859.
- Both calibrated families selected tau 0.5 and conditional mean in all folds. Temperature-hierarchy intermediate MAE 38.537516 only slightly improved raw-hierarchy mean 38.597744.
- Temperature-hierarchy transition F1 0.316339 exceeded posterior median 0.305097, but no hierarchy passed the predefined overall/depth and endpoint-collapse criteria.
- The remaining failure is not resolved by hierarchical calibration; the predefined decision therefore justifies retraining.

## 13. Recommended next step

Do not repeatedly tune decoding; the next task should retrain the encoder-only model with CE plus a distance-aware or ordinal auxiliary objective. Do not launch that training here.

## 14. Limitations

- Oracle human-performance input.
- Calibration and selection use validation pieces.
- Four-point PT pedal tokenizer and no raw-CC64 timing.
- No end-to-end Stage 1 input or listening test.
- No neural-model retraining.
- Conditional depth classes remain weakly ordered.
