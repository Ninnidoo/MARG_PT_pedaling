# Custom Event Prediction Decoder v1 Verification Report

## Frozen provenance

- Decoder version: `1.0.0`
- Decoder ID: `938ca7e20e1d398aa530938bbcede404fd3a04af23842fbcf351ba02df0268f8`
- Tokenizer cache ID: `3a5520155b5db1e9a1da7f8148556aa3e1da852655c9adde25d3dbbd1d966263`
- Note-alignment ID: `98de59ef7a41fba26a2c89fe686c273f6c8ec7f27979922e71813624ac1a4822`
- Canonical checkpoint: `analysis/custom_event_model_v0_full_seed42_aligned_statusfix_v1/best.pt`
- Checkpoint epoch / best epoch: `1 / 1`
- ASAP validation prediction/metric access: `0 / 0`
- ASAP test access: `0`
- Repedal execution: `0`

The tokenizer/cache/checkpoint were used read-only. Decoder v1 was frozen and verified before canonical ASAP validation inference or metric access.

## Frozen grammar

Initial uses finite deterministic argmax with lowest-index tie resolution. Main uses first-NONE termination, finite `tau` clipping to `[0,1]`, stable `(tau, original slot)` pair sorting, right-open non-final boundaries and inclusive final-main boundary. Same-time Main SETs collapse to the last destination and same-state absolute SETs are suppressed under one piece-level Initial→Main→Terminal state machine.

Terminal uses first-NONE termination, `z=max(0,z)` followed by `expm1`, unsorted slot-order cumulative gaps, and a discrete strictly-post-noteoff/increasing one-tick grammar guard. A terminal no-op advances both the continuous and discrete active-slot clock before suppression.

No confidence threshold, calibration, sampling, beam search, smoothing, minimum duration, or validation-metric-tuned heuristic exists.

## Targeted verification

The Decoder v1 suite passed `13/13`; the frozen Tokenizer v1 regression suite passed `8/8`, for `21/21` targeted tests. Covered cases include Initial priority, prefix termination, event/timing pair preservation, reversed/equal timing order, clamps, right-open/final-inclusive tick bounds, continuous state, same-time last-SET, redundant SET suppression, terminal cumulative gaps, terminal no-op clock advance, strict terminal ticks, determinism, and fail-fast NaN/Inf handling.

## Full frozen-cache structural oracle

Ground-truth target classes were converted to deterministic logits and passed with exact cached timing targets through Decoder v1. This was a structural grammar regression, not model inference or pedal-metric evaluation.

| Split | Performances | Main intervals | Main active SET slots | Terminal active SET slots | Effective emitted changes | Same-state suppressed | Same-time collapsed | Final-state mismatches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 2,062 | 9,009,549 | 3,220,893 | 3,287 | 3,223,233 | 947 | 0 | 0 |
| Validation | 71 | 272,927 | 82,673 | 136 | 82,771 | 38 | 0 | 0 |

All Main and Terminal targets were prefix-valid. All active Main taus were chronological, all Terminal log-gap targets were nonnegative, and decoded final state matched the frozen compressed-target final state for every performance. Main and Terminal final-state preservation are therefore exact `100%` over 2,133 performances.

The difference between active SET slots and emitted state changes is fully accounted for by 947 train and 38 validation same-state no-ops. No ground-truth same-time SET group survived tokenizer canonicalization/compression.

## Canonical best-checkpoint train-only smoke

Selection was deterministic: the shortest two performances among the stable first 16 canonical train entries. The checkpoint was loaded read-only with PyTorch's restricted weights-only loader and strict model-state loading. All performance-level logits/timings were finite and two identical decoder calls were structure-exact.

| Performance index | Windows | Modeled onsets | Decoded events incl. Initial | Final state |
|---:|---:|---:|---:|---:|
| 6 | 2 | 717 | 259 | ZERO |
| 7 | 3 | 776 | 286 | LOW |

Aggregate prediction grammar diagnostics:

- Main active slots: `807`
- Main later non-NONE ignored after first NONE: `2`
- Raw Main timing order violations restored by pair sorting: `76`
- Main tau clamp low/high: `1 / 0`
- Main same-time collapse: `0`
- Main redundant SET suppression: `267`
- Terminal active slots: `5`
- Terminal later non-NONE ignored after first NONE: `0`
- Terminal negative-z clamps: `0`
- Terminal redundant SET suppression: `2`

These values are implementation diagnostics only and did not change the frozen rules.

## Verification questions

1. **Main first-NONE grammar:** exact; later active predictions are ignored.
2. **Main pair sorting:** exact; event, timing, and original slot remain one pair.
3. **Interval boundaries:** non-final right-open and final-main inclusive helpers pass exact tests.
4. **Same-time/same-state:** deterministic last destination and no-op suppression pass continuous and quantized tests.
5. **State continuity:** one state machine is maintained through Initial, every Main interval, and Terminal.
6. **Terminal first-NONE:** exact.
7. **Terminal inverse/cumulative clock:** `expm1(max(0,z))`, slot-order accumulation exact.
8. **Terminal no-op timing:** clock advances before suppression, verified in continuous and tick domains.
9. **Strict terminal boundary:** every active slot receives a tick greater than the prior active terminal tick.
10. **Ground-truth final state:** exact for all train and validation cache performances.
11. **Checkpoint smoke:** finite and deterministic on the two train-only performances.
12. **Metric-tuned heuristics:** none.

## Readiness

**READY FOR CANONICAL VALIDATION INFERENCE**

Decoder grammar verification is complete. No ASAP validation candidate MIDI, canonical pedal metric, ASAP test item, or Repedal computation was run.
