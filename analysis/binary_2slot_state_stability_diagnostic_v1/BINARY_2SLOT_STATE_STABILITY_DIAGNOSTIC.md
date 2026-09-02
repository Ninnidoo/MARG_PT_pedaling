# Binary 2-Slot State-Stability Diagnostic v1

## Summary

| formulation | aggregate status | individual statuses | late state median by subset |
|---|---:|---|---|
| C: online reconciliation + base-count weights | WARN | WARN, WARN, WARN | 0.8805, 0.9951, 0.9166 |
| D: static targets + local state consistency | PASS | PASS, PASS, PASS | 0.9987, 1.0000, 1.0000 |

Selected `lambda_state=0.25` by the frozen initial-gradient rule. The same train-only subsets [7, 6, 13], pretrained checkpoint, seed-42 heads, optimizer, LR, FP32 mode, 1,000 steps, and boundary weight 0.25 were used independently for C and D. No weights were transferred.

| formulation | subset | status | late base acc median | late macro F1 median | late state median | late state p10 | late state min | late timing MAE median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C | 7 | WARN | 0.9897 | 0.9081 | 0.8805 | 0.6260 | 0.5514 | 0.0441 |
| C | 6 | WARN | 0.9958 | 0.9819 | 0.9951 | 0.1873 | 0.1725 | 0.0312 |
| C | 13 | WARN | 0.9962 | 0.9772 | 0.9166 | 0.6117 | 0.5781 | 0.0491 |
| D | 7 | PASS | 1.0000 | 1.0000 | 0.9987 | 0.9987 | 0.9987 | 0.0106 |
| D | 6 | PASS | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0183 |
| D | 13 | PASS | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0123 |

## Scientific interpretation

Condition D keeps the actual hard model state as head input and updates it only with argmax Count parity, exactly as inference. Count/Timing labels are static base targets, so model errors never mutate later labels. Its auxiliary loss does use the human interval-end state during training to provide a local differentiable parity signal; therefore D does not eliminate every train/inference information difference and does not create a soft recurrent state.

Interpretation: **CASE D: D passes while C remains unstable**. Recommended next full-training formulation: **Condition D**. No full training was started.

## Required questions

**Q1. Was C instability reproduced?** Aggregate WARN; individual statuses WARN, WARN, WARN, with late state medians 0.8805, 0.9951, 0.9166. See per-run p10/min/std in `late_stability_summary.csv`.

**Q2. Selected lambda_state?** 0.25, chosen only from the initial gradient ratio; no performance sweep was run.

**Q3. Can D memorize static Count/Timing under hard free-running?** Aggregate PASS; late median Count accuracies 1.0000, 1.0000, 1.0000, Macro F1 1.0000, 1.0000, 1.0000. Timing medians are tabulated above.

**Q4. Is D hard state stable late?** Late state medians were 0.9987, 1.0000, 1.0000; full p10/min/std determine the aggregate gate.

**Q5. Does D show a 50/50 attractor?** No; all three runs cleared the stability gate.

**Q6. Count/state conflict frequency?** Across all 1,001 optimizer states per run, interval-weighted D fractions by subset were 3.90%, 6.28%, 7.11%; the combined optimizer-state×interval fraction was 5.89%. At the final state all three subsets had zero conflict intervals.

**Q7. Conflict recovery within 1/2/4 intervals?** Conflict-observation-weighted rates across the complete training trajectories were 3.76% / 6.67% / 11.83%. These include the deliberately retained unstable initialization period; by the final state no conflict remained. Per-step/subset values are in `condition_D_conflict_intervals.csv`.

**Q8. Does State loss recreate synthetic N1?** No. D never mutates Count labels, all three late Count accuracies/Macro F1 reached 1.0, and final predicted N1 fractions were only 3.47%, 8.62%, and 5.66% (the corresponding static base distributions), not an N1 collapse. Static Count CE prevented the parity objective from becoming a synthetic N1 label loop.

**Q9. Better late stability?** Median across subset medians: C=0.9166, D=1.0000. Recommendation also respects Macro F1, mismatch runs, timing, and static-objective simplicity.

**Q10. Recommended formulation?** Condition D.

**Q11. Remaining blocker?** No scientific diagnostic blocker. Condition D still needs explicit opt-in integration into the full trainer plus a guarded full-run config and authorization; this diagnostic did not perform that integration or start full training.

## Provenance and prohibited activity

Six independent train-only runs completed: 6,000 optimizer steps total. Full-training steps/epochs: 0. Validation inference/metrics: 0. ASAP test metadata/MIDI: 0/0. Checkpoints and checkpoint selection: 0. State-loss lambda candidates were evaluated only by the mandated initial gradient-scale calculation, never by training performance.

The five focused parity-loss unit tests and 15 existing Binary 2-Slot model regression tests passed. Pytest is not installed in the existing project container, so the same test functions were executed directly; `unit_test_results.json` records the exact suites and counts.
