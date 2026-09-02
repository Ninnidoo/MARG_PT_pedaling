# State-Conditioned Event Model v1 — Full-Training Comparison Plan

이 design task에서는 실행하지 않는다.

## Controlled comparison

~~~text
Custom Event Model v0
vs
Custom Event Model v1 B3-S
~~~

동일하게 고정:

- train 2,062 / validation 71 performances
- seed 42
- pretrained PT encoder initialization
- legacy head initialization policy
- optimizer, encoder/head LR, weight decay, AMP, gradient clipping
- effective batch, window 512, stride 256, shuffled owner-window order
- unique ownership, note alignment
- Tokenizer v1 cache, absolute SET vocabulary
- slot-specific inverse-sqrt event weights
- timing heads/formulations
- Prediction Decoder v1, evaluator
- epoch cap, cadence, patience, min-delta
- legacy validation-total checkpoint selection

유일한 atomic formulation change는 B3-S Main branch다: six detached semantic pre-state posterior heads, fixed 4D posterior feature, expanded Main event heads, required detached auxiliary state CE. Candidate C, hard mask, new weights, schedules, thresholds, smoothing, calibration, decoder change를 추가하지 않는다.

## Primary question

Transition advantage를 유지하면서 four-state destination fidelity가 개선되는가?

임의 threshold 없이 report:

- 4C Accuracy
- Macro F1
- Transition F1
- 256-pattern JS
- Intersection

Diagnostics:

- ZERO/HALF occupancy relative to human
- ZERO↔LOW, HALF↔FULL change precision/recall/count
- OFF↔ON counts and binary accuracy
- raw prefix-active redundant same-state SET count/rate
- state-head exact/binary pre-state accuracy overall/by slot
- predicted-vs-GT Decoder pre-state confusion
- mismatch duration/onset/event-count distributions
- Main slot funnel, especially Slot5/6
- Terminal contribution
- auxiliary state CE by slot

## Checkpoint issue

Legacy v0 validation objective를 selection score로 유지하고 auxiliary state CE는 제외한다. 두 score를 모두 log한다. Initial-loss dominance와 Oracle-Initial negligible effect는 별도 checkpoint-selection experiment로 남긴다.

## Reproducibility and boundaries

Cache/alignment/decoder ID, v0/v1 initialization digest, class-weight SHA, training order seed, checkpoint epoch, artifact hashes를 기록한다. Candidate generation/evaluation은 authorized full training 뒤에만 수행한다. ASAP test access 0, Repedal execution 0을 유지한다.
