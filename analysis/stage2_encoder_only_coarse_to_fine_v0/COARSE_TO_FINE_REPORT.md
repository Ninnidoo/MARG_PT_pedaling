# Encoder-only Coarse-to-Fine Pedal Head v0

## 1. Hypothesis

Separating endpoint-region selection from conditional intermediate depth can avoid the flat 128-class argmax endpoint collapse without changing encoder, data, or training controls.

## 2. Exact architecture

Official pretrained bidirectional PT encoder (unfrozen, hidden size 768), with four independent Linear(768,3) region heads and four independent Linear(768,1) depth heads; no MLP, conditioning, smoothing, calibration, or post-processing.

## 3. Target construction

0→ZERO, 1–126→INTERMEDIATE, 127→FULL; intermediate z=(y−1)/125.

## 4. Loss and decoding formulas

Total loss is unweighted region CE plus SmoothL1(beta=0.05) on true-intermediate positions. Raw region and depth logits are averaged over windows before argmax/sigmoid decoding.

## 5. Implementation/test results

See `tests/test_results.json`.

## 6. Overfit result

```json
{
  "active_encoder_parameter": "embeddings.word_embeddings.weight",
  "encoder_and_all_head_parameters_changed": true,
  "final": {
    "end_to_end_mae": 0.0228271484375,
    "intermediate_oracle_region_mae": 0.02560241147875786,
    "region_accuracy": 1.0
  },
  "finite_nonzero_gradients_all_groups": true,
  "initial": {
    "end_to_end_mae": 49.2005615234375,
    "intermediate_oracle_region_mae": 50.37431716918945,
    "region_accuracy": 0.5164794921875
  },
  "oom": false,
  "passed": true,
  "performance_path": "Schubert/Piano_Sonatas/664-3/Lin07.mid",
  "steps": 250,
  "window_starts": [
    0,
    256,
    512,
    768
  ]
}
```

## 7. Training summary

Completed 7 epochs; best epoch 3; best validation total loss 0.959460. Best-epoch region CE/depth/total: 0.8223158854279314/0.13714374727629997/0.9594596367820074.

## 8. Full validation comparison

Overall MAE 27.982664; intermediate MAE 37.477367; exact token accuracy 0.502301; exact-note accuracy 0.446522.

## 9. Region classification analysis

Region accuracy 0.619105, macro F1 0.585478; detailed confusion and per-region P/R/F1 are in `validation/`.

## 10. Oracle-region depth analysis

Oracle-region intermediate MAE 19.531960; detailed tolerance and distribution diagnostics are in `validation/oracle_region_metrics.json`.

## 11. Intermediate concentration/collapse analysis

Predicted intermediate ratio 0.228989; endpoint collapse 0.619898; unique rounded intermediate values 92.

## 12. Transition analysis

Transition F1 0.326510; detailed transition metrics are in the validation comparison.

## 13. Success-criteria decision

Criterion A=False; Criterion B=False; safeguards=False; final pass=False.

## 14. Interpretation

Region-selection failure is indicated by low region accuracy/F1; conditional-depth failure by poor oracle-region MAE or narrow intermediate support; temporal-transition failure by transition F1. This decision uses validation only, never training loss alone.

## 15. Recommended next step

Do not automatically continue to MAESTRO or another architecture; inspect the measured region/depth/transition failure mode first.
