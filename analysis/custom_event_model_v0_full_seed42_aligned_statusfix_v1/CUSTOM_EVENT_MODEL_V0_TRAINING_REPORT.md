# Custom Event Model v0 Canonical Training Report

- Status: `completed`
- Cache ID: `3a5520155b5db1e9a1da7f8148556aa3e1da852655c9adde25d3dbbd1d966263`
- Note-alignment ID: `98de59ef7a41fba26a2c89fe686c273f6c8ec7f27979922e71813624ac1a4822`
- Alignment config SHA: `77df6faf08025794e40c58d0db31cf721f582b3168d9fb201557c0500807502b`
- Seed: 42
- Best epoch: 1
- Best validation total: 4.087803782
- Micro/accumulation/effective batch: 4/4/16
- AMP overflow events: 4
- ASAP test access: 0
- Candidate MIDI/frozen evaluator/Repedal: not executed

## Epoch objectives

| Epoch | Train total | Validation total | Init | Main event | Main time | Terminal event | Terminal time |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3.609489 | 4.087804 | 1.265462 | 0.550905 | 0.147853 | 1.779560 | 0.344024 |
| 2 | 3.112792 | 4.092334 | 1.535098 | 0.530463 | 0.137533 | 1.656925 | 0.232314 |
| 3 | 2.575716 | 4.255226 | 1.653812 | 0.521758 | 0.137098 | 1.677680 | 0.264877 |
| 4 | 1.898697 | 4.921556 | 2.151187 | 0.516196 | 0.132971 | 1.816390 | 0.304812 |

## Best-epoch diagnostics

- Main event accuracy/non-NONE F1: 0.932027 / 0.483658
- Main Slot5 predicted/target non-NONE: 6057/2264 (F1 0.385050)
- Main Slot6 predicted/target non-NONE: 4073/1028 (F1 0.277593)
- Terminal event accuracy/non-NONE F1: 0.679577 / 0.707819
- Terminal Slot4 predicted/target non-NONE: 0/3 (F1 0.000000)
- Main raw tau MAE/RMSE: 0.208247 / 0.260484
- Terminal log-gap MAE/RMSE: 0.326647 / 0.439897
- Terminal decoded seconds-gap MAE: 0.469613

## Interpretation

Best checkpoint selection used only minimum ASAP-validation target total objective. No lambda, beta, weight, architecture, or tokenizer changes were made. Component histories and sparse-slot non-NONE ratios above determine whether any branch collapsed; no automatic corrective experiment was launched.

Readiness for canonical validation inference must be judged from the completed loss/diagnostic table. This runner does not generate MIDI or execute the frozen evaluator.

Recommendation: **READY FOR CANONICAL VALIDATION INFERENCE** if the run status is completed and the best checkpoint loads successfully; otherwise **NEEDS TARGETED TRAINING FIX**. No inference was started automatically.
