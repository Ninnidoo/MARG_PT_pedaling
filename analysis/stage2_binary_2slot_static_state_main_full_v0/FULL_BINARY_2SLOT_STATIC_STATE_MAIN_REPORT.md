# Full Binary 2-Slot Static-State MAIN-only Run A

| item | result |
|---|---|
| status | **STOPPED — guarded smoke FAIL** |
| MAIN-only integration | PASS: exactly M-1 intervals; PRE/POST/final tail absent |
| static target / reconciliation | static max-2 / no reconciliation or correction |
| lambda_state | 0.25 |
| smoke subset / intervals | canonical TRAIN 7 / 775 |
| human state at t1 / model initialization | ON / OFF |
| final Count accuracy / Macro F1 | 1.0000 / 1.0000 |
| final state agreement / mismatch fraction | 0.0000 / 1.0000 |
| mismatch run / recovery | 775 intervals / not recovered |
| loss reduction / timing MAE | 84.52% / 0.016005 |
| final predicted N0/N1/N2 | 722 / 26 / 27 |
| full optimizer updates / validation inference | 0 / 0 |
| checkpoints / tmux launch | 0 / not launched |
| ASAP test access | 0 |

## Hard-stop finding

The integration is internally consistent, finite, leakage-free, owner-complete, and uses only `[t1,tM)`. However, subset 7 starts with human pedal ON while the frozen model initialization is OFF. Once the model memorized every static Count target, the opposite parity offset persisted for all 775 intervals: Count accuracy was 1.0 while state agreement was 0.0. Recovery within 1/2/4/8 intervals was 0/0/0/0.

This is not an N=1 collapse (final N1 fraction 3.35%) and not an optimization divergence (loss fell 84.52%). It is the explicit MAIN-only cold-start state failure prohibited by the launch gate. With no target mutation, all base Count parities preserve the initial offset; the local state loss asks for opposite parity on every mismatched interval but at frozen `lambda_state=0.25` did not select a single recovery error.

Therefore full training, validation inference, checkpoint creation, and tmux launch were not authorized. No scientific setting was changed automatically.

Initial random-head state agreement was 0.0168; the final persistent zero is the trained smoke result, not random-head evidence.
