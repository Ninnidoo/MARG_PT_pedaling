# Full Run B State-Anchored Transition v1

| item | result |
|---|---|
| status | COMPLETE |
| formulation | frozen State-Anchored Transition v1 MAIN-only `[t1,tM)` |
| train / human validation | 2,062 / 71 performances |
| Frozen Stage 1 | `/workspace/project/analysis/stage2_binary_canonical_v1/canonical_validation_stage1_manifest.csv`; 19 pieces; seed 42; SHA `f32e91da5d2196edee825236daa577f18709cf7d49e289d8f18ae8ea637b0037` |
| canonical Frozen references | 71 structural / 70 aligned; metadata 856 excluded |
| architecture | official PT 10x768 + four direct linear heads; 5,383 head parameters |
| Mode weights HOLD/CHANGE/RETURN | 0.305652382570267 / 0.907267124114601 / 1.787080493315132 |
| optimizer | AdamW; encoder 1e-5; heads 1e-4; wd 0.01; clip 1.0; FP16 AMP |
| epochs | 10 / 10, no early stopping |
| best criterion | Frozen-PT global-Viterbi Transition F1 on `[t1,tM)`; earlier tie |
| best epoch / F1 | 9 / 0.6249756009669524 |
| PRE / POST / recurrence / Stage1 regeneration / ASAP test | 0 / 0 / 0 / 0 / 0 |

## Epoch table

| epoch | train loss | Human State | Human Mode Macro F1 | Human Transition F1 | Frozen raw State | Frozen Viterbi State | Frozen P | Frozen R | Frozen F1 | best |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 1.30102 | 0.7592 | 0.5805 | 0.5571 | 0.6261 | 0.6301 | 0.7234 | 0.4361 | 0.5442 |  |
| 2 | 1.12340 | 0.7932 | 0.6070 | 0.5885 | 0.6069 | 0.6081 | 0.6909 | 0.4634 | 0.5547 |  |
| 3 | 1.08093 | 0.7969 | 0.6078 | 0.6017 | 0.6203 | 0.6200 | 0.6626 | 0.5328 | 0.5907 |  |
| 4 | 1.04693 | 0.8006 | 0.6143 | 0.6138 | 0.6257 | 0.6269 | 0.6997 | 0.5157 | 0.5938 |  |
| 5 | 1.02522 | 0.7881 | 0.6204 | 0.6194 | 0.6272 | 0.6270 | 0.6826 | 0.5615 | 0.6161 |  |
| 6 | 1.00926 | 0.8055 | 0.6217 | 0.6219 | 0.6204 | 0.6212 | 0.6814 | 0.5406 | 0.6029 |  |
| 7 | 0.99146 | 0.8095 | 0.6216 | 0.6304 | 0.6208 | 0.6207 | 0.6874 | 0.5449 | 0.6079 |  |
| 8 | 0.97884 | 0.8127 | 0.6193 | 0.6296 | 0.6147 | 0.6149 | 0.6456 | 0.5844 | 0.6135 |  |
| 9 | 0.96379 | 0.8126 | 0.6218 | 0.6294 | 0.6226 | 0.6217 | 0.6473 | 0.6041 | 0.6250 | yes |
| 10 | 0.95039 | 0.8144 | 0.6273 | 0.6353 | 0.6119 | 0.6128 | 0.6678 | 0.5380 | 0.5959 |  |

## Best epoch diagnostics

Best Frozen-PT P/R/F1=0.647341/0.604104/0.624976; raw/Viterbi State=0.622606/0.621733; pred/ref=0.933209; Viterbi conflicts=0.

Frozen multi-human timing is N/A because there is no canonical one-to-one timing aggregation. Human-input CHANGE/RETURN timing MAE is recorded per epoch. Non-pedal identity, pedal masking, Viterbi legality, and S1 initialization-anchor exclusion are hard invariants.

## Direct Run A common-horizon comparison

Run A epoch 8 common `[t1,tM)`: P/R/F1=0.589162/0.576268/0.582644; State agreement=0.488841.

1. State anchoring substantial absolute-state improvement: **yes** (Run B 0.621733, Run A 0.488841).
2. Transition F1 vs Run A: **improved** (Run B 0.624976, Run A 0.582644).
3. Viterbi legality conflicts eliminated: **yes**.
4. HOLD/CHANGE/RETURN collapse avoided: **yes**.
5. Late-epoch over-specialization evidence: **yes**.
