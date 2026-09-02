# Full Binary 2-Slot Condition D PRE/MAIN/POST run

| item | result |
|---|---|
| status | COMPLETE |
| formulation | restored Condition D, immutable static targets, hard free-running |
| lambda_state / boundary weight | 0.25 / 0.25 |
| online reconciliation / dynamic target | 0 / 0 |
| frozen Stage 1 | `/workspace/project/analysis/stage2_binary_canonical_v1/canonical_validation_stage1_manifest.csv` (19 pieces, seed 42, SHA `f32e91da5d2196edee825236daa577f18709cf7d49e289d8f18ae8ea637b0037`) |
| Stage 1 regeneration | 0 |
| previous failed run | `/workspace/project/analysis/stage2_binary_2slot_D_pre_main_post_full_v0` |
| previous failure / mapping fix | metadata mapping KeyError / `/workspace/project/analysis/stage2_binary_2slot_D_pre_main_post_full_v0/FROZEN_PT_MAPPING_FIX_REPORT.md` |
| structural / canonical aligned population | 19 pieces, 71 references / 70 of 71 (metadata 856 excluded) |
| fault-tolerant train-complete checkpoint | enabled before validation |
| epoch 1 continuation | saved post-train state; validation rerun; epoch-1 training not reexecuted |
| negative PRE serialization | origin-state projection; tick-0 anchor only when projected state is ON |
| Frozen-PT primary metric source | serialized candidate MIDI CC64 trajectory |
| saved-model renderer audit | 19/19 PASS; non-pedal identity 19/19; pedal leakage 0 |
| epochs | 10 / 10 |
| best epoch / frozen-PT Transition F1 | 8 / 0.5835638492499086 |
| best / last | `/workspace/project/analysis/stage2_binary_2slot_D_pre_main_post_full_v1/best.pt` / `/workspace/project/analysis/stage2_binary_2slot_D_pre_main_post_full_v1/last.pt` |
| ASAP test access | 0 |

Checkpoint selection uses only frozen-PT-input canonical direction-aware Transition F1, with earlier-epoch tie break. Human-input validation is auxiliary. Frozen PT pedal tokens are masked before encoding; output MIDI preserves source notes exactly and replaces CC64 only (EOT extension, if required for POST, is explicitly audited).

| epoch | train loss | human F1 | frozen P | frozen R | frozen F1 | frozen state | first-MAIN state |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.39446 | 0.5613 | 0.6981 | 0.3585 | 0.4737 | 0.4925 | 0.5714 |
| 2 | 1.21727 | 0.5791 | 0.7058 | 0.3733 | 0.4883 | 0.4936 | 0.5000 |
| 3 | 1.17656 | 0.5961 | 0.5716 | 0.5624 | 0.5670 | 0.4999 | 0.5714 |
| 4 | 1.13668 | 0.6148 | 0.6951 | 0.4276 | 0.5295 | 0.5107 | 0.6571 |
| 5 | 1.09995 | 0.6209 | 0.6841 | 0.4425 | 0.5374 | 0.5358 | 0.5714 |
| 6 | 1.06430 | 0.6240 | 0.5725 | 0.5855 | 0.5789 | 0.4977 | 0.5429 |
| 7 | 1.03518 | 0.6320 | 0.6051 | 0.5471 | 0.5747 | 0.5159 | 0.5286 |
| 8 | 1.01152 | 0.6380 | 0.5901 | 0.5772 | 0.5836 | 0.4888 | 0.5286 |
| 9 | 0.99471 | 0.6363 | 0.6223 | 0.5357 | 0.5758 | 0.5198 | 0.6429 |
| 10 | 0.98559 | 0.6436 | 0.6393 | 0.5239 | 0.5759 | 0.4906 | 0.6571 |

No validation result changes LR, loss weights, architecture, or schedule. Full training terminates after epoch 10 unless a frozen hard-stop invariant fails.
