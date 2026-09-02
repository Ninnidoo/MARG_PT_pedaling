# Run B State-Anchored Transition v1 Implementation / Smoke

**Verdict: PASS**

## Frozen formulation

Run B models only `[t1,tM)`: M onset State targets and M-1 HOLD/CHANGE/RETURN edges. Targets are immutable human-only labels. PRE, POST, final note-off tail, recurrent state input, online reconciliation, dynamic labels, and consistency auxiliary loss are absent.

## Target audit

- Train: 2,062 performances; 9,009,549 States; 9,007,487 intervals.
- Mode counts HOLD/CHANGE/RETURN: 7,882,290/894,618/230,579.
- Legality conflicts: 0; parity/final-state preservation: 100%/100%; duplicate/missing ownership: 0/0.
- Exact-onset raw tau=0 transitions: 33,348.

## Architecture and loss

- Pretrained PT encoder: hidden 768, 10 layers, 103,271,424 parameters.
- Heads: State 1,538; Mode 2,307; Timing 769+769; total **5,383**.
- Loss: unweighted State CE + train-only mean-one inverse-sqrt weighted Mode CE + active-slot Smooth-L1 beta 0.1.
- Frozen Mode weights HOLD/CHANGE/RETURN: 0.305652382570267/0.907267124114601/1.787080493315132; arithmetic mean=1.

## Global decoding and tests

All maximum-margin owner logits are assembled exactly once across the whole performance, then one O(M*4) constrained Viterbi is run. Different states force CHANGE; same states select HOLD/RETURN. Focused direct-function tests: 46 PASS. Viterbi legality conflicts: 0.

## TRAIN-only tiny overfit

- Canonical train index 7: 776 onsets, 775 intervals, 3 windows; fresh seed-42 heads; 1,000 AdamW steps.
- Final raw State accuracy: 1.000000.
- Final raw Mode accuracy / Macro F1: 1.000000 / 1.000000.
- Final Viterbi State accuracy / Mode accuracy: 1.000000 / 1.000000.
- Final Transition P/R/F1: 1.000000/1.000000/1.000000 on train `[t1,tM)` with the canonical direction-aware ±1-onset matcher.
- Timing active MAE: 0.013503; CHANGE=0.014685; RETURN slots=0.014570/0.011297.
- Predicted HOLD/CHANGE/RETURN: 722/26/27; no persistent class collapse or 50/50 recurrence attractor.
- Target mutation/recurrent-state calls: 0/0. Validation/full training/checkpoints: 0/0/0. ASAP test access: 0.

## Stop point

Focused target/model/decoder tests and TRAIN-only tiny memorization passed. No Run B full training was started.
