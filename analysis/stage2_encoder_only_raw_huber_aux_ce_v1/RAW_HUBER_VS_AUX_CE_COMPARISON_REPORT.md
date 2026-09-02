# Raw Huber vs Auxiliary CE Clean Rerun v1

ASAP validation only; ASAP test access 0; Repedal execution 0. Auxiliary logits were excluded from candidate inference.

| Objective | 4C Acc ↑ | Macro F1 ↑ | Trans P ↑ | Trans R ↑ | Trans F1 ↑ | JS ↓ | Intersection ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Standard CE | 0.569449 | 0.396136 | 0.354366 | 0.571714 | 0.437535 | 0.068187 | 0.815823 |
| Weighted CE | 0.533587 | 0.408260 | 0.340975 | 0.602783 | 0.435565 | 0.039055 | 0.858292 |
| Representative Huber | 0.477443 | 0.396158 | 0.471488 | 0.521667 | 0.495310 | 0.042943 | 0.810684 |
| Raw CC64 Huber | 0.492320 | 0.403413 | 0.497066 | 0.512063 | 0.504453 | 0.038859 | 0.825369 |
| **Raw Huber + Aux CE λ=0.1** | 0.507749 | 0.409029 | 0.502348 | 0.513770 | 0.507995 | 0.038362 | 0.834859 |

| Objective | LOW Recall | HALF Recall | Candidate Transitions |
|---|---:|---:|---:|
| Raw CC64 Huber | 0.211279 | 0.271617 | 35611 |
| Raw Huber + Aux CE | 0.186309 | 0.267974 | 35354 |

## Q1. 4-class Accuracy / Macro F1

Confirmed: changes versus Raw Huber are +0.015430 Accuracy and +0.005616 Macro F1.

## Q2. LOW/HALF discrimination

Confirmed: LOW recall -0.024970; HALF recall -0.003643. Confusion counts are reported in the validation report.

## Q3. Transition advantage

Confirmed: Transition P/R/F1 changes are +0.005282/+0.001707/+0.003542; candidate count change -257.

## Q4. Global pattern distribution

Confirmed: JS change -0.000497; Intersection change +0.009491.

## Q5. AMP recovery

Confirmed: 4 AMP gradient-overflow event(s); maximum consecutive 1. Event details: `[{'affected_groups': ['regression_heads'], 'ce': 1.0197780132293701, 'consecutive_overflows': 1, 'epoch': 4, 'global_optimizer_attempt': 8006, 'grad_norm': inf, 'huber': 0.16520091891288757, 'offending_parameter_count': 2, 'offending_parameter_names': ['regression_heads.0.weight', 'regression_heads.3.weight'], 'optimizer_step_skipped': True, 'overflow_events_total': 1, 'scale_after': 8192.0, 'scale_before': 16384.0, 'scaler_step_update_executed': True, 'step': 1334, 'total': 0.2671787142753601}, {'affected_groups': ['regression_heads'], 'ce': 0.7707827687263489, 'consecutive_overflows': 1, 'epoch': 5, 'global_optimizer_attempt': 10047, 'grad_norm': inf, 'huber': 0.12408579885959625, 'offending_parameter_count': 2, 'offending_parameter_names': ['regression_heads.2.weight', 'regression_heads.3.weight'], 'optimizer_step_skipped': True, 'overflow_events_total': 2, 'scale_after': 8192.0, 'scale_before': 16384.0, 'scaler_step_update_executed': True, 'step': 1151, 'total': 0.20116406679153442}, {'affected_groups': ['regression_heads'], 'ce': 0.7649747133255005, 'consecutive_overflows': 1, 'epoch': 6, 'global_optimizer_attempt': 12469, 'grad_norm': inf, 'huber': 0.11739889532327652, 'offending_parameter_count': 2, 'offending_parameter_names': ['regression_heads.0.weight', 'regression_heads.1.weight'], 'optimizer_step_skipped': True, 'overflow_events_total': 3, 'scale_after': 8192.0, 'scale_before': 16384.0, 'scaler_step_update_executed': True, 'step': 1349, 'total': 0.1938963681459427}, {'affected_groups': ['regression_heads'], 'ce': 0.8772043585777283, 'consecutive_overflows': 1, 'epoch': 8, 'global_optimizer_attempt': 16486, 'grad_norm': inf, 'huber': 0.12448064982891083, 'offending_parameter_count': 2, 'offending_parameter_names': ['regression_heads.2.weight', 'regression_heads.3.weight'], 'optimizer_step_skipped': True, 'overflow_events_total': 4, 'scale_after': 16384.0, 'scale_before': 32768.0, 'scaler_step_update_executed': True, 'step': 918, 'total': 0.2122010886669159}]`.

## Q6. Net value of auxiliary classification

Confirmed metrics show the observed discrimination/transition/distribution trade-off above. Inference: practical benefit is supported only if categorical gains are not offset by unacceptable transition or distribution degradation; this single λ=0.1 validation run does not establish sensitivity or causality.

No λ/delta sweep, hybrid follow-up, temporal loss, architecture change, ASAP-test evaluation, or Repedal evaluation was started.
