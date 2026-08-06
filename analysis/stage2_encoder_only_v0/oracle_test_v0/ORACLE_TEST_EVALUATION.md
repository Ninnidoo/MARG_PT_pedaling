# Experiment 1 v0 Oracle Test Evaluation

## 1. Evaluation scope

This is an Oracle Stage 2 evaluation: each human ASAP performance's Pitch/IOI/Velocity/Duration tokens are retained, Pedal1–4 are masked, and the trained Stage 2 encoder predicts pedal tokens that are compared with that same human performance's original pedal targets. It is not an end-to-end score-to-performance evaluation, and original Pianist Transformer generation was not run.

## 2. Checkpoint and environment

- Best checkpoint: `/workspace/project/analysis/stage2_encoder_only_v0/train_v0/best.pt`
- Best epoch: 2
- Recorded best validation loss: 2.2459586270
- Exact state-dict load: yes; missing keys=[], unexpected keys=[]
- Architecture: official PT encoder + four independent Linear(768, 128) heads
- Encoder inference state: `freeze_encoder=False`; encoder parameters retained `requires_grad=True`, no gradients were created, and no parameter version changed during inference.
- Project commit: `1d64469760e963492897723c92e61ddad7c76a54`
- Pinned PT commit: `747df2d12291e37f6638b39f1b71517e579ad48c`
- GPU mapping: host GPU 1 UUID `GPU-6982dbee-fbaf-f359-d7ef-a22d0e83400b` -> container GPU 0 -> PyTorch `cuda:0` (NVIDIA GeForce RTX 2080 Ti)
- Inference batch size: 16; FP16 autocast, `model.eval()`, and `torch.inference_mode()`
- Dataset preload/majority time: 57.747 s; checkpoint/model load: 1.224 s; test inference and metrics: 10.042 s; total: 69.293 s
- Peak allocated GPU memory: 1009075712 bytes (0.940 GiB)

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

## 3. Test-set verification

- Performances: 104 (expected 104)
- Unique pieces: 23 (expected 23)
- Complete notes: 331576
- Complete pedal targets: 1326304
- Deterministic windows: 1237
- Unique performance paths: 104; all MIDI files existed and tokenized successfully.
- Only `split=test` rows were evaluated; metadata order from the split CSV was preserved.
- Every original note, including the first and last, had at least one window contribution. Averaged logits were invariant to reversed window input order.
- Contribution counts ranged from 1 to 3 per note. Final aggregation counted each of the 331576 original notes once, not once per overlapping window.

## 4. Overall Stage 2 results

| Aggregation | Loss | Token acc. | Exact-note acc. | P1 | P2 | P3 | P4 | MAE | Targets | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| Micro | 2.288124 | 0.568735 | 0.506394 | 0.566932 | 0.568618 | 0.569713 | 0.569679 | 31.802348 | 1326304.0 | 331576.0 |
| Macro mean | 2.418103 | 0.546397 | 0.484920 | 0.544394 | 0.550594 | 0.545619 | 0.544983 | 33.373873 | 12752.9 | 3188.2 |

Micro metrics pool correct/error counts across all complete test performances. Macro metrics first compute a complete-performance metric and then average across 104 performances.

## 5. Baseline comparison

| Method | Loss | Token acc. | Exact note | P1 | P2 | P3 | P4 | MAE | Intermediate acc. | Transition acc. | Transition F1 | JS distance | Intersection |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Stage 2 | 2.288124 | 0.568735 | 0.506394 | 0.566932 | 0.568618 | 0.569713 | 0.569679 | 31.802348 | 0.000000 | 0.110027 | 0.124416 | 0.105738 | 0.921439 |
| All zero | NA | 0.338330 | 0.306720 | 0.339657 | 0.340169 | 0.336589 | 0.336903 | 65.642811 | 0.000000 | 0.085668 | NA | 0.613991 | 0.421149 |
| All full | NA | 0.339464 | 0.304187 | 0.335422 | 0.337051 | 0.342284 | 0.343098 | 61.357189 | 0.000000 | 0.083223 | NA | 0.569415 | 0.483814 |
| Global majority | NA | 0.339464 | 0.304187 | 0.335422 | 0.337051 | 0.342284 | 0.343098 | 61.357189 | 0.000000 | 0.083223 | NA | 0.569415 | 0.483814 |
| Slot majority | NA | 0.339464 | 0.304187 | 0.335422 | 0.337051 | 0.342284 | 0.343098 | 61.357189 | 0.000000 | 0.083223 | NA | 0.569415 | 0.483814 |
| Previous-note persistence* | NA | 0.692183 | 0.605445 | 0.722531 | 0.677643 | 0.679956 | 0.688602 | 12.770875 | 0.318490 | 0.015807 | 0.292799 | 0.000368 | 0.999735 |

`*` Previous-note persistence is teacher-forced and not deployable: the first note uses train slot-majority classes, and later notes copy the previous note's complete ground-truth pedal vector. Deterministic baselines have no cross-entropy entry. Global and slot-majority classes were computed from train only: global=127, slots=[127, 127, 127, 127].

## 6. Intermediate-pedal results

Human target distribution: zero=0.338330, full-127=0.339464, intermediate=0.322207.

| Method | Intermediate N | Intermediate acc. | Intermediate MAE | Int. -> 0 | Int. -> 127 | Endpoint collapse | Pred. zero | Pred. 127 | Pred. intermediate |
|---|---|---|---|---|---|---|---|---|---|
| Stage 2 | 427344 | 0.000000 | 55.715756 | 0.551268 | 0.448732 | 1.000000 | 0.522008 | 0.477992 | 0.000000 |
| All zero | 427344 | 0.000000 | 69.926942 | 1.000000 | 0.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 |
| All full | 427344 | 0.000000 | 57.073058 | 0.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 | 0.000000 |
| Global majority | 427344 | 0.000000 | 57.073058 | 0.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 | 0.000000 |
| Slot majority | 427344 | 0.000000 | 57.073058 | 0.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 | 0.000000 |
| Previous-note persistence* | 427344 | 0.318490 | 19.419016 | 0.097879 | 0.108475 | 0.206354 | 0.338180 | 0.339681 | 0.322139 |

## 7. Transition results

This analysis flattens the original four-sample PT pedal-token sequence within each performance. It is token-grid analysis, not continuous raw-CC64 event-timing evaluation, and performance boundaries are never joined.

Per-performance subset metrics are `nan` when their denominator is zero (for example, no intermediate targets, true transitions, or predicted transitions). Macro means exclude only those undefined performance-level values.

| Aggregation | True transitions | True ratio | Transition acc. | Transition MAE | Steady acc. | Steady MAE | Detection precision | Detection recall | Detection F1 | Direction acc. |
|---|---|---|---|---|---|---|---|---|---|---|
| Micro | 206168.0 | 0.155458 | 0.110027 | 55.493355 | 0.653181 | 27.440168 | 0.270208 | 0.080813 | 0.124416 | 0.057846 |
| Macro mean | 1982.4 | 0.168056 | 0.116485 | 55.750263 | 0.615877 | 30.247285 | 0.264186 | 0.072199 | 0.115165 | 0.051510 |

## 8. PT-style 16-configuration distribution

The exact repository implementation was found at `third_party/PianistTransformer/src/evaluate/evaluate.py` in `plot_pedal_pattern_distribution` (lines 243–338 at the pinned commit). It binarizes each slot as 1 for value >= 64, maps Pedal1–4 with weights 8/4/2/1, adds epsilon 1e-10 to histogram probabilities, calls SciPy `jensenshannon(..., base=2)`, and defines histogram intersection as `sum(min(p_i, q_i))`. SciPy returns the square-root Jensen–Shannon distance even though the upstream variable/report calls it divergence. The table reports both the true base-2 divergence and the upstream-comparable base-2 distance; probabilities are renormalized after epsilon so identical intersections equal one.

| Method | Base-2 JS divergence | PT-code JS distance | Histogram intersection |
|---|---|---|---|
| Stage 2 | 0.011181 | 0.105738 | 0.921439 |
| All zero | 0.376985 | 0.613991 | 0.421149 |
| All full | 0.324233 | 0.569415 | 0.483814 |
| Global majority | 0.324233 | 0.569415 | 0.483814 |
| Slot majority | 0.324233 | 0.569415 | 0.483814 |
| Previous-note persistence* | 0.000000 | 0.000368 | 0.999735 |

These values are measured on this leakage-safe ASAP test split and Oracle pipeline; no paper number is reused as if directly comparable.

## 9. Performance-level variation

| Metric | 10th percentile | Median | 90th percentile |
|---|---|---|---|
| Token accuracy | 0.220407 | 0.562810 | 0.871196 |
| Exact-note accuracy | 0.146408 | 0.484010 | 0.836928 |
| Pedal-value MAE | 9.623415 | 33.188650 | 50.233180 |

### 10 worst performances by Stage 2 token accuracy

| Performance | Composer | Notes | Token acc. | Exact note | MAE |
|---|---|---|---|---|---|
| Bach/Prelude/bwv_885/JeonH01M.mid | Bach | 516 | 0.063953 | 0.027132 | 102.071705 |
| Bach/Prelude/bwv_885/SINKEV06.mid | Bach | 525 | 0.070000 | 0.005714 | 101.236190 |
| Chopin/Etudes_op_25/8/SOLOM03.mid | Chopin | 1471 | 0.081407 | 0.070020 | 52.270904 |
| Chopin/Etudes_op_25/8/DeTurck02.mid | Chopin | 1516 | 0.085257 | 0.072559 | 50.563325 |
| Bach/Italian_concerto/TanakaM03.mid | Bach | 1171 | 0.108027 | 0.053800 | 56.207942 |
| Bach/Prelude/bwv_885/SINKEV01.mid | Bach | 523 | 0.115679 | 0.005736 | 86.217017 |
| Chopin/Etudes_op_25/8/Toscano02.mid | Chopin | 1455 | 0.158935 | 0.136770 | 49.361512 |
| Beethoven/Piano_Sonatas/7-1/Larionova04M.mid | Beethoven | 3482 | 0.197300 | 0.180356 | 41.464173 |
| Mozart/Piano_Sonatas/12-2/WuuE03.mid | Mozart | 1377 | 0.197349 | 0.116195 | 53.581155 |
| Beethoven/Piano_Sonatas/5-1/Colafelice02M.mid | Beethoven | 2111 | 0.201800 | 0.134533 | 50.593558 |

### 10 best performances by Stage 2 token accuracy

| Performance | Composer | Notes | Token acc. | Exact note | MAE |
|---|---|---|---|---|---|
| Bach/Fugue/bwv_854/LuA01M.mid | Bach | 739 | 0.995940 | 0.994587 | 0.468539 |
| Bach/Fugue/bwv_854/WangA01M.mid | Bach | 736 | 0.992527 | 0.990489 | 0.715353 |
| Bach/Fugue/bwv_854/Richardson01M.mid | Bach | 738 | 0.967141 | 0.963415 | 2.709350 |
| Bach/Fugue/bwv_854/Ozaki01M.mid | Bach | 734 | 0.950954 | 0.944142 | 4.196866 |
| Ravel/Miroirs/3_Une_Barque/DupreeF20.mid | Ravel | 5116 | 0.932907 | 0.916927 | 4.990129 |
| Haydn/Keyboard_Sonatas/48-2/PrjevalskayaM03M.mid | Haydn | 2822 | 0.930103 | 0.910702 | 7.038448 |
| Ravel/Miroirs/3_Une_Barque/KimHJ08.mid | Ravel | 4831 | 0.929362 | 0.906645 | 4.678224 |
| Bach/Prelude/bwv_885/Guo01M.mid | Bach | 524 | 0.927004 | 0.870229 | 9.270515 |
| Bach/Fugue/bwv_854/MiyashitaM01M.mid | Bach | 738 | 0.923442 | 0.909214 | 1.482046 |
| Mozart/Piano_Sonatas/8-1/LEE_J03.mid | Mozart | 3295 | 0.879363 | 0.856146 | 10.107208 |

### Composer-level aggregate results

| Composer | Performances | Notes | Token acc. | Exact note | MAE |
|---|---|---|---|---|---|
| Bach | 15 | 11576 | 0.515139 | 0.446441 | 37.163204 |
| Beethoven | 34 | 128926 | 0.535922 | 0.476110 | 35.208682 |
| Chopin | 18 | 65782 | 0.526223 | 0.454532 | 33.039414 |
| Haydn | 5 | 13575 | 0.774715 | 0.736354 | 17.624346 |
| Liszt | 18 | 68779 | 0.592906 | 0.527865 | 29.591125 |
| Mozart | 8 | 23261 | 0.496776 | 0.438717 | 36.805587 |
| Ravel | 3 | 14591 | 0.912138 | 0.886780 | 5.772240 |
| Schumann | 3 | 5086 | 0.539668 | 0.395399 | 36.792175 |

These performance, composer, and piece differences are descriptive; no statistical significance is claimed.

## 10. Interpretation

- Trivial/majority baselines: Stage 2 token accuracy (0.568735) exceeds the strongest trivial/majority baseline, All full (0.339464).
- Persistence: Stage 2 does not exceed the teacher-forced persistence baseline (0.692183).
- Intermediate behavior: 0.000000 of all predictions are intermediate; endpoint collapse among intermediate targets is 1.000000.
- Temporal difficulty: transition exact accuracy is 0.110027 versus 0.653181 at steady positions.
- Concentration: piece-level token accuracy spans 0.155970 (piece_04919a4df9fdb995) to 0.966011 (piece_800604c74ef8b5c9); composer-level accuracy spans 0.496776 (Mozart) to 0.912138 (Ravel). The descriptive Pearson correlation between sequence length and performance token accuracy is 0.2115.

## 11. Limitations

- Inputs are Oracle human-performance non-pedal tokens.
- Stage 1 score-to-performance distribution shift is not evaluated.
- Pedal targets remain the original four-point PT tokenizer representation.
- Transition metrics operate on the token grid, not continuous time.
- No listening test was performed.
- No raw-CC64 event-timing reconstruction was performed.
- Original PT predictions were not compared in this task.

## 12. Recommended next task

Run another targeted diagnostic before end-to-end comparison: quantify how much the current objective learns beyond teacher-forced state persistence, then evaluate temporal/delta-aware loss or conditioning changes. The current model should not be retrained from this test result without first defining the change on train/validation data.
