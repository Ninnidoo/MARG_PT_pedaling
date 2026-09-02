# Custom vs Hybrid Matched-Input ASAP Validation

- Hybrid checkpoint: `/workspace/project/analysis/stage2_encoder_only_raw_huber_aux_ce_v1/best.pt`; epoch 5; SHA `2a4550de0c6737ac0c9d688d0615d9e57a8a0c877eebc1dc670bec817aaa7c5e`.
- Architecture/objective: pretrained PT Encoder-only; raw CC64/127 Smooth-L1 Huber delta=14/127; auxiliary canonical 4C CE lambda=0.1; inference regression-only.
- Inference: Pitch/IOI/Velocity/Duration input with pedal masked; 512/256 windows; regression scalar overlap mean; clip [0,1], x127, half-up integer.
- Primary input identity: 71/71 exact Custom source SHA/non-pedal projection/note count. ASAP test=0; Repedal=0.
- Frozen evaluator: 70 successful pairs / 272,053 notes; Tysman07M excluded exactly as before.

## Three-way metrics

| Model/input | 4C Acc | Macro F1 | Trans P | Trans R | Trans F1 | Candidates | JS | Intersection |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Custom Event / matched human | 0.350434 | 0.288461 | 0.554082 | 0.574231 | 0.563977 | 35,825 | 0.128919 | 0.646315 |
| Hybrid regression-only / matched human | 0.423644 | 0.324311 | 0.389638 | 0.309361 | 0.344890 | 27,446 | 0.015307 | 0.926481 |
| Hybrid regression-only / historical Original-PT | 0.507749 | 0.409029 | 0.502348 | 0.513770 | 0.507995 | 35,354 | 0.038362 | 0.834859 |

A vs B is primary; C is read-only provenance. Historical C used 19 Original-PT candidates and is not the primary matched comparison.

## Exact deltas

- Custom - matched Hybrid: `{'accuracy': -0.07320999952215196, 'macro_f1': -0.03585002423118322, 'transition_precision': 0.16444451043853975, 'transition_recall': 0.2648692432307337, 'transition_f1': 0.21908666817822736, 'js': 0.11361200359832004, 'intersection': -0.2801659898944986}`
- Matched - historical Hybrid: `{'accuracy': -0.08410493543537467, 'macro_f1': -0.08471762130049487, 'transition_precision': -0.1127098491376532, 'transition_recall': -0.2044087016894237, 'transition_f1': -0.16310475900025384, 'js': -0.023055336448071173, 'intersection': 0.09162185074928186}`
- Transition density: Custom 35825/34568=1.036363; matched Hybrid 27446/34568=0.793971; historical Hybrid 35354/34568=1.022738.

## Interpretation

- Under exact matched inputs, Custom Transition F1 advantage is +0.219087; precision delta +0.164445, recall delta +0.264869.
- Custom 4-state fidelity deltas remain: accuracy -0.073210, Macro F1 -0.035850, JS +0.113612 (lower is better), Intersection -0.280166.
- Matched-input construction changes Hybrid Transition F1 by -0.163105 from historical; this is an observed association, not a causal estimate.
- Decision: **EVENT REPRESENTATION RETAINS A TRANSITION ADVANTAGE UNDER MATCHED INPUTS**
- Recommended next task: Custom Event 4-state/depth-state distribution failure audit; do not tune decoder first.
- No training, checkpoint selection, calibration, smoothing, threshold change, or post-hoc tuning was performed.
