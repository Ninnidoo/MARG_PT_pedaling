# Full Binary 2-Slot Condition D PRE/MAIN/POST v0

| item | result |
|---|---|
| status | RUNNING |
| formulation | restored Condition D, immutable static targets, hard free-running |
| lambda_state / boundary weight | 0.25 / 0.25 |
| online reconciliation / dynamic target | 0 / 0 |
| frozen Stage 1 | `/workspace/project/analysis/stage2_binary_canonical_v1/canonical_validation_stage1_manifest.csv` (19 pieces, seed 42, SHA `f32e91da5d2196edee825236daa577f18709cf7d49e289d8f18ae8ea637b0037`) |
| Stage 1 regeneration | 0 |
| epochs | 0 / 10 |
| best epoch / frozen-PT Transition F1 | pending / pending |
| best / last | `/workspace/project/analysis/stage2_binary_2slot_D_pre_main_post_full_v0/best.pt` / `/workspace/project/analysis/stage2_binary_2slot_D_pre_main_post_full_v0/last.pt` |
| ASAP test access | 0 |

Checkpoint selection uses only frozen-PT-input canonical direction-aware Transition F1, with earlier-epoch tie break. Human-input validation is auxiliary. Frozen PT pedal tokens are masked before encoding; output MIDI preserves source notes exactly and replaces CC64 only (EOT extension, if required for POST, is explicitly audited).

| epoch | train loss | human F1 | frozen P | frozen R | frozen F1 | frozen state | first-MAIN state |
|---:|---:|---:|---:|---:|---:|---:|---:|
| pending | | | | | | | |

No validation result changes LR, loss weights, architecture, or schedule. Full training terminates after epoch 10 unless a frozen hard-stop invariant fails.
