# Custom Event Model v0 Canonical Validation Report

## Frozen provenance

- Checkpoint: `/workspace/project/analysis/custom_event_model_v0_full_seed42_aligned_statusfix_v1/best.pt`; epoch/best epoch = 1; SHA256 `a81c5f8f0a902523bd8abb1ca5b1097876887477f1668907f08e2280d7d096c8`
- Tokenizer cache ID: `3a5520155b5db1e9a1da7f8148556aa3e1da852655c9adde25d3dbbd1d966263`
- Note-alignment ID: `98de59ef7a41fba26a2c89fe686c273f6c8ec7f27979922e71813624ac1a4822`
- Alignment config SHA: `77df6faf08025794e40c58d0db31cf721f582b3168d9fb201557c0500807502b`
- Prediction Decoder v1 `1.0.0` ID: `938ca7e20e1d398aa530938bbcede404fd3a04af23842fbcf351ba02df0268f8`
- ASAP validation only: 71 performances / 19 pieces. ASAP test access = 0. Repedal execution = 0.

## Input and evaluation convention

The frozen Custom Event cache defines 71 performance-level validation inputs, so inference uses each canonical ASAP-validation human performance's non-pedal stream and produces 71 candidates. This is required by the frozen 272,927-onset ownership universe. Every pitch, onset, note-off, velocity, and every non-CC64 musical message were preserved exactly. For frozen terminal events extending beyond a source EOT, only the EOT meta tick was minimally extended and explicitly counted. The conventional evaluator's metric definitions and its frozen successful 70-pair/272,053-aligned-note universe are reused exactly. The historical conventional candidate construction used 19 saved Original-PT piece candidates; that input-construction difference is retained as an explicit provenance caveat when comparing rows.

## Integrity

- Candidates: 71/71; musical non-pedal identity: 71/71 PASS. EOT-only extension was required for 13 candidates (maximum 1,441 ticks); no note or other non-CC64 event changed.
- Unique-owner predictions: 272,927/272,927; zero-owner 0; duplicate-owner 0.
- Initial/terminal sets: 71/71 and 71/71.
- Finite outputs and semantic determinism: PASS; two-candidate byte determinism: PASS.
- Frozen alignment exclusion: `Liszt/Mephisto_Waltz/Tysman07M.mid` (same unavailable shared-human alignment as canonical evaluator); common successful pairs = 70.

## Main metrics

| Model | 4C Accuracy | Macro F1 | Transition P | Transition R | Transition F1 | Candidates | JS divergence | Intersection |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Custom Event v0 / Decoder v1 | 0.350434 | 0.288461 | 0.554082 | 0.574231 | 0.563977 | 35,825 | 0.128919 | 0.646315 |
| Hybrid regression-only (read-only reference) | 0.507749 | 0.409029 | 0.502348 | 0.513770 | 0.507995 | 35,354 | 0.038362 | 0.834859 |

The evaluator uses 4^4=256 joint Pedal1-4 patterns, base-2 Jensen-Shannon divergence, and intersection Σmin(P,Q). Transitions use raw CC64 threshold 64, direction-aware one-to-one matching, and ±1 distinct-onset tolerance.

## Decoder diagnostics

- Initial-state distribution (class IDs 0..3): `{'0': 37, '3': 31, '1': 3}`
- Raw Main non-NONE slots: 87,712; active after first-NONE: 86,793; ignored later non-NONE: 919.
- Main raw tau order violations: 3,250; clamp low/high: 250/279.
- Main same-time collapse (continuous/quantized): 39/211; Main same-state suppression: 28,586.
- Raw Terminal non-NONE: 107; active: 107; ignored later: 0.
- Terminal negative-z clamps: 53; strict-tick guards: 53; same-state suppression: 8.
- Effective emitted Main/Terminal transitions before evaluator alignment: 58,061/99.
- Candidate/reference effective state changes in modeled source scope: 58,160/83,163.

The trained model previously showed Slot5/6 non-NONE overprediction. At MIDI/evaluator level the candidate has 35,825 threshold transitions versus 34,568 reference transitions (+3.64%) and 35,354 for Hybrid (+1.33%); this is a modest association, not proof of slot-level causality.

## Interpretation

- The primary transition bottleneck is **precision** because P=0.554082 and R=0.574231.
- Compared with the read-only Hybrid reference, Custom Event deltas are: accuracy -0.157315, Macro F1 -0.120568, Transition F1 +0.055982, JS +0.090557, Intersection -0.188544.
- The first failure mode to inspect in a future task is the relationship between prefix-active sparse slots, effective transition density, and precision; no corrective experiment was run here.
- No decoder, tokenizer, checkpoint, threshold, calibration, or post-processing was changed after observing metrics.

## Decision

**CUSTOM EVENT REPRESENTATION NOT YET BETTER THAN CONVENTIONAL FINALIST**
