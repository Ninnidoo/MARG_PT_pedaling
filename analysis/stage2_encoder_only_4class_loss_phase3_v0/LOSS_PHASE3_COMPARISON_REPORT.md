# Encoder-only 4-Class Loss Phase 3 Comparison

ASAP validation only; test access count 0; frozen canonical evaluator; Repedal excluded.

| Objective | 4C Acc ↑ | Macro F1 ↑ | Transition P ↑ | Transition R ↑ | Transition F1 ↑ | JS ↓ | Intersection ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Standard CE | 0.569449 | 0.396136 | 0.354366 | 0.571714 | 0.437535 | 0.068187 | 0.815823 |
| Weighted CE | 0.533587 | 0.408260 | 0.340975 | 0.602783 | 0.435565 | 0.039055 | 0.858292 |
| CE + NTL-WAS λ=0.3 | 0.570835 | 0.395119 | 0.351994 | 0.574057 | 0.436400 | 0.071404 | 0.811103 |
| Representative Huber δ=14 | 0.477443 | 0.396158 | 0.471488 | 0.521667 | 0.495310 | 0.042943 | 0.810684 |

| Objective | LOW Recall | HALF Recall | Candidate Transitions | Best Epoch | Status |
|---|---:|---:|---:|---:|---|
| Standard CE | 0.057167 | 0.158789 | 55770 | 2 | completed |
| Weighted CE | 0.169470 | 0.197027 | 61110 | 2 | completed |
| CE + NTL-WAS λ=0.3 | 0.054766 | 0.153766 | 56376 | 2 | completed |
| Representative Huber δ=14 | 0.216820 | 0.281513 | 38247 | 6 | completed |

## Interpretation scope

This report is limited to intermediate-depth performance, class-imbalance effects, numerical-distance-aware loss effects, regression-formulation effects, spurious-transition changes, and global pattern-distribution changes. It starts no follow-up experiment.
