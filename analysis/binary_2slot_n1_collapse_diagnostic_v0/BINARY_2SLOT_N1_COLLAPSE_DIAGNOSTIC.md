# Binary 2-Slot N=1 Collapse Diagnostic v0

## Summary

| condition | status | final loss | dynamic Count acc | base Count acc | state agreement | correction frac | dominant prediction |
|---|---:|---:|---:|---:|---:|---:|---:|
| A static/oracle + fixed weighted CE | PASS | 0.002071 | 1.0000 | 1.0000 | 0.9987* | 0.0013* | 93.06% |
| B hard free-running + unweighted CE | FAIL | 0.672876 | 0.7609 | 0.8805 | 0.7609 | 0.2391 | 82.52% |
| C hard free-running + base-count weighted CE | WARN | 0.490048 | 0.8728 | 0.9897 | 0.8715 | 0.1285 | 93.06% |

\*A uses oracle state and target-parity updates; its state/correction values are control diagnostics, not free-running performance.

All three conditions independently loaded the same pretrained PT checkpoint and reproduced the same seed-42 head fingerprint. They used canonical train performance index 7, 500 AdamW steps, encoder/head LR 1e-5/1e-4, weight decay 0.01, dropout unchanged, FP32, gradient clip 1.0, and boundary weight 0.25. No condition inherited another condition's weights.

Condition C computes each region's Count loss exactly as `sum_i w[N_base_i] * CE(logits_i, N_dynamic_i) / sum_i w[N_base_i]`. A uses the same expression with `N_dynamic=N_base`; B uses unit weights. Timing and MAIN/BOUNDARY aggregation are otherwise identical.

## Fixed-point and weighted-objective findings

At B step 500, 23.20% of base-N0 intervals were mismatched, and 23.20% of all base-N0 intervals (100.00% conditional on mismatch) became dynamic N1. At C step 500 the corresponding values were 12.15%, 12.15%, and 100.00%.

In the prior failed dynamic-class-weighted run, N1 was 50.26% of dynamic labels but received 69.94% of normalized weighted objective mass. This directly quantifies the feedback amplification. Full per-class and matched/mismatched tables are in the CSV artifacts.

Train Transition F1 was not computed: this tiny item is MAESTRO and no frozen train score-alignment Transition metric covers it. Inventing a new time tolerance would violate the controlled comparison; Count/state/timing diagnostics remain exact.

## Decision

- Interpretation: **No exact enumerated case: A PASS / B FAIL / C WARN; directionally CASE 3, but C is not stable enough to count as PASS**.
- Most likely root cause: **combination: dynamic-class weighting amplified N1, while hard moving-target reconciliation still caused late instability**.
- Recommended next formulation: **base-count weighted CE is the leading diagnostic candidate, but it needs a repeat/stability gate before any full training.**
- Full training blocker: the recommendation requires explicit human approval and a new guarded run.

## Required questions

**Q1. Did A memorize?** PASS; final base Count accuracy 1.0000, timing MAE 0.009873, loss reduction 99.95%.

**Q2. Evidence against architecture/optimizer?** A passed, so the encoder/heads/loss/optimizer can memorize this subset without moving-target feedback.

**Q3. Did B remove N=1 collapse?** The anomalous N1 concentration disappeared (N1 14.14%), but the condition is FAIL because state/Count convergence remained insufficient and unstable; final distribution {'0': 642, '1': 110, '2': 26}.

**Q4. Did C remove N=1 collapse?** Yes, N1 fell to 2.96%; overall status is WARN because the strong 0.90/0.90/0.10 gate was missed and late state agreement ranged 0.1620–0.8715. Final distribution {'0': 724, '1': 23, '2': 31}.

**Q5. Was the 50/50 cycle reproduced/explained?** Yes. B at step 275 reproduced 0.4987 agreement / 0.5013 correction, and C at step 150 reproduced 0.4936 / 0.5064. At those checkpoints, respectively 50.14% and 50.69% of base-N0 intervals converted to dynamic N1, with 100% conversion conditional on base-N0 mismatch. Final B/C values improved to 0.7609/0.2391 and 0.8715/0.1285 but remained unstable.

**Q6. Base N0 -> dynamic N1 due to mismatch?** B: 23.20% of all base-N0 (100.00% given mismatch). C: 12.15% (100.00% given mismatch).

**Q7. N1 objective mass in the prior weighting?** Dynamic frequency 50.26%, normalized weighted mass 69.94%.

**Q8. B versus C stability?** C had the better final state/correction/base accuracy (0.8715/0.1285/0.9897) than B (0.7609/0.2391/0.8805), but neither was stable: late state ranges were 0.5668–0.8470 for B and 0.1620–0.8715 for C. C is only the leading candidate, not a cleared full-training objective.

**Q9. Root cause?** combination: dynamic-class weighting amplified N1, while hard moving-target reconciliation still caused late instability.

**Q10. Next full-training candidate?** base-count weighted CE is the leading diagnostic candidate, but it needs a repeat/stability gate before any full training.

**Q11. Remaining blocker?** Full training was not run. Any changed Count objective/state policy needs explicit approval and a new guarded gate before full training.

## Provenance and prohibited activity

The fixed weights are exactly the train/base-oracle/ALL inverse-sqrt weights from `binary_2slot_transition_followup_v0/weight_normalization.json` (`sum f_c w_c=1`); they were not recomputed. Optimizer updates: 500 per condition, 1,500 total. Checkpoints: 0. Full epochs: 0. Validation inference/metrics: 0. ASAP test metadata/MIDI access: 0/0.
