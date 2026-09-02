# Run B Binary Validation Evaluation

**Verdict: PASS**

## Provenance and integrity

- Run B candidate: persisted epoch-9 MIDI artifacts reused; best.pt inference rerun = 0.
- best.pt: epoch 9; Frozen Transition F1 `0.624975600967`.
- Frozen Stage1: `/workspace/project/analysis/stage2_binary_canonical_v1/canonical_validation_stage1_manifest.csv`; SHA `f32e91da5d2196edee825236daa577f18709cf7d49e289d8f18ae8ea637b0037`; seed 42.
- Universe: 19 pieces / 71 structural references / 70 canonical aligned references; metadata 856 excluded.
- Non-pedal identity: 19/19 PASS; pedal leakage: 0; Stage1 regeneration: 0; ASAP test access: 0.

## Headline comparison

| System | Binary Acc ↑ | Binary Macro F1 ↑ | Transition F1 ↑ | Repedal F1 ↑ | 16-pattern JS ↓ | Intersection ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Original PT | 0.753439 | 0.748219 | 0.592327 | 0.524350 | 0.013056 | 0.887635 |
| Run B | 0.730575 | 0.721247 | 0.624976 | 0.536699 | 0.011253 | 0.909069 |

## Transition and Repedal

| System | Transition P | Transition R | Transition F1 | Pred/Ref | Repedal P | Repedal R | Repedal F1 | Repedal Pred/Ref |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original PT | 0.773325 | 0.479986 | 0.592327 | 0.620679 | 0.709222 | 0.415930 | 0.524350 | 0.586460 |
| Run B | 0.647341 | 0.604104 | 0.624976 | 0.933209 | 0.551779 | 0.522421 | 0.536699 | 0.946794 |

Transition denominator: 70 candidate-human pairs; Original PT candidate/reference events 21383 / 34451, Run B 32150 / 34451. Repedal denominator: 70 pairs; Original PT pooled candidate/reference episodes 7363 / 12555, Run B 11887 / 12555. Repedal pooled candidate counts repeat each piece candidate for each paired human reference, matching the FINAL evaluator; unique 19-piece candidate episodes are 1442 / 2298.

Run B materialized-MIDI Transition F1 reproduces training time: **YES** (`0.624975600967`). UP and DOWN diagnostics are in `transition_metrics.json`.

## Binary state and 16-pattern semantics

Pedal1–4 are sampled with the existing canonical PT score-alignment logic and thresholded as OFF `<64`, ON `>=64`. Samples from the final distinct onset onward are excluded, so Accuracy/F1 use `1087028` pooled Pedal slots on `[t1,tM)`. The 16-pattern histogram uses `271757` complete aligned notes, pools counts globally before normalization, and retains all 16 bins. JS is base-2 divergence (not distance); epsilon `1e-10` and zero handling are unchanged from the prior 256-pattern implementation.

Repedal intentionally retains the FINAL evaluator's full-saved-MIDI event universe; its denominator is therefore reported separately rather than forced onto the sampled-state or Transition horizon.

Run B OFF/ON F1: `0.670255` / `0.772238`. Target OFF/ON ratio: `0.378648` / `0.621352`. Predicted OFF/ON ratio: `0.438425` / `0.561575`.

## Answers

1. Epoch-9 MIDI: existing persistent artifacts reused; best.pt generation rerun = 0.
2. Non-pedal identity: 19/19 PASS.
3. Run B Binary Pedal1–4 Accuracy: `0.730574557`.
4. Run B Binary Macro F1: `0.721246817`.
5. Run B Transition P/R/F1: `0.647340591` / `0.604104380` / `0.624975601`; training-time value reproduced: YES.
6. Run B Repedal P/R/F1: `0.551779255` / `0.522421346` / `0.536699124`.
7. Run B 16-pattern JS / Intersection: `0.011253240` / `0.909069500`.
8. Run B vs Original PT: `{"binary_accuracy": "degraded", "binary_macro_f1": "degraded", "pattern_intersection": "improved", "pattern_js_divergence_base2": "improved", "repedal_f1": "improved", "transition_f1": "improved"}`.
9. Trade-off: Run B improves Transition F1 while lowering sampled-state Accuracy/Macro-F1; Repedal F1 and both distribution metrics improve, so the observed sacrifice is state fidelity rather than distribution or Repedaling.
10. ASAP test access: 0.

Bass Connectivity and Harmonic Muddiness were not evaluated.
