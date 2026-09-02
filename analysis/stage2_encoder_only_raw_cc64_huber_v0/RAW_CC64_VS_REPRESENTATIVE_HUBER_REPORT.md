# Raw CC64 vs Representative Huber

ASAP validation only; identical architecture/training schedule/inference conversion; target representation is the intended independent variable. ASAP test access count: 0.

| Objective | 4C Acc ↑ | Macro F1 ↑ | Trans P ↑ | Trans R ↑ | Trans F1 ↑ | JS ↓ | Intersection ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Representative Huber | 0.477443 | 0.396158 | 0.471488 | 0.521667 | 0.495310 | 0.042943 | 0.810684 |
| Raw CC64 Huber | 0.492320 | 0.403413 | 0.497066 | 0.512063 | 0.504453 | 0.038859 | 0.825369 |

| Objective | LOW Recall | HALF Recall | Candidate Transitions | Best Epoch |
|---|---:|---:|---:|---:|
| Representative Huber | 0.216820 | 0.281513 | 38247 | 6 |
| Raw CC64 Huber | 0.211279 | 0.271617 | 35611 | 6 |

## Q1. Intermediate depth

Confirmed: LOW recall changed by -0.005541; HALF recall by -0.009896. Both improved: False.

## Q2. Transition behavior

Confirmed: Transition F1 changed by +0.009143, precision by +0.025578, and candidate count by -2636. Improved F1: True.

## Q3. Global 256-pattern distribution

Confirmed: JS changed by -0.004084; Intersection by +0.014685. Both improved: True.

## Q4. Confusion pattern

Confirmed raw-model counts: adjacent `{'HALF_FULL': 143461, 'LOW_HALF': 51265, 'ZERO_LOW': 117273}`, same-side `{'HALF_FULL': 143461, 'ZERO_LOW': 117273, 'total': 260734}`, 64-crossing `{'directional_pair_counts': {'FULL_to_LOW': 65390, 'FULL_to_ZERO': 56066, 'HALF_to_LOW': 33161, 'HALF_to_ZERO': 42274, 'LOW_to_FULL': 12673, 'LOW_to_HALF': 18104, 'ZERO_to_FULL': 31087, 'ZERO_to_HALF': 32975}, 'high_side_to_low_side': 196891, 'low_side_to_high_side': 94839, 'total': 291730}`. These counts identify which errors accompanied the accuracy change; causal attribution remains an inference.

## Q5. Evidence for continuous raw targets

Confirmed: the run changed training targets from representatives to unquantized cached raw CC64 while holding the model and schedule fixed. Validation evidence favoring raw targets on at least one depth/main metric: True. Inference: any gain is consistent with reduced target quantization, but one validation run does not establish a general causal advantage.

No hybrid loss, delta sweep, smoothing, temporal loss, architecture change, ASAP-test evaluation, or Repedal experiment was started.
