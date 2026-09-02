# Custom Event Model v0 Note Alignment Report

## Scope and frozen provenance

- Frozen tokenizer cache ID: `3a5520155b5db1e9a1da7f8148556aa3e1da852655c9adde25d3dbbd1d966263`
- Note-alignment ID: `98de59ef7a41fba26a2c89fe686c273f6c8ec7f27979922e71813624ac1a4822`
- Alignment config SHA256: `77df6faf08025794e40c58d0db31cf721f582b3168d9fb201557c0500807502b`
- ASAP test access: 0
- Training steps / optimizer steps: 0 / 0
- The failed `analysis/custom_event_model_v0_full_seed42/` artifact was preserved and was not used for resume.
- The official PT note sequence and frozen cache/targets were not modified or rebuilt.

## Exact first failure

| Field | Value |
|---|---|
| Performance | `Chopin/Sonata_2/1st_no_repeat/Jia03.mid` |
| Piece | `piece_418b959c7292dd8d` |
| Train performance index | 1793 |
| Dataset window index | 31168 |
| Performance window index | 12 |
| Global note window | `[3072,3584)` |
| Cache / PT note count | 4176 / 4176 |
| First differing global note position | 70 |
| Pitch-order differing positions | 14 |

At cache positions 70 and 71, raw onsets 22591 and 22592 contain pitches 61 and 49. Official PT tempo normalization rounds both onsets to PT tick 19309, after which its stable `(normalized onset, pitch)` sort places pitch 49 before pitch 61. The note multiset, note count, normalized duration, and velocity remain consistent. The failure was therefore an ordering permutation caused by tempo conversion/rounding, not cache corruption or a different musical note universe. Full ±10-note context is in `first_failure_diagnostic.json`.

## Alignment rule

1. Reparse the original allowed MIDI and verify the frozen cache projection exactly against canonical raw identity `(onset, pitch, noteoff, velocity)`. Cache track/channel metadata remains part of that parser-side provenance.
2. Reject duplicate complete common identities as ambiguous; none occur in the canonical train/validation universe.
3. Carry each cache index through the official PT conversion: tempo-map seconds, 1000 target ticks/sec rounding, minimum one-tick duration, same-pitch overlap cleanup, and stable `(normalized onset, pitch)` sort.
4. Verify every resulting `(normalized onset, pitch, normalized noteoff, velocity)` against the official `normalize_midi` result.
5. Save the exact `cache_to_pt` permutation and inverse, and assert a total bijection.
6. Map the frozen cache representative with `pt_rep = cache_to_pt[cache_rep]`. Map every cache onset group to its PT min/max bounds before applying the unchanged full-containment / maximum-margin / smaller-start ownership rule.

This is an index adapter only. `S_init`, main/terminal events, tau, and log-gap targets remain frozen.

## Full universe audit

| Split | Performances | Notes | Identity-order performances | Permuted performances | Moved notes | Moved fraction | Max displacement |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 2,062 | 9,369,095 | 2,061 | 1 | 14 | 0.000149427% | 1 |
| Validation | 71 | 283,928 | 71 | 0 | 0 | 0% | 0 |

Across both splits, unmapped cache notes, unmapped PT notes, ambiguous mappings, note-count mismatches, and duplicate complete common raw identities are all zero. All 14 moved notes cross distinct raw onset ticks, but all 14 remain within the same PT-normalized onset. Cross-normalized-onset movement is zero. Thus the discrepancy is confined to a PT-onset-local permutation after normalization. Non-contiguous mapped cache onset groups are also zero.

## Ownership regression

| Split | Windows | Mapped/owned onsets | 0-owner | Duplicate-owner | Initial supervision | Terminal supervision |
|---|---:|---:|---:|---:|---:|---:|
| Train | 35,573 | 9,009,549 / 9,009,549 | 0 | 0 | 2,062 | 2,062 |
| Validation | 1,078 | 272,927 / 272,927 | 0 | 0 | 71 | 71 |

The frozen ownership totals are exactly preserved. There are 1,411 train and 46 validation groups rejected from at least one intersecting window because the full group was not contained; each still has exactly one final owner.

## Regression tests and dry-runs

- 22 alignment, dataset, ownership, loss, forward/backward, and full-training-helper tests: PASS.
- Tested identity order, PT-onset permutation, repeated pitch with full identity, explicit ambiguous/missing identity failure, inverse bijection, representative remap, mapped containment/ownership, serialization/reload, and frozen-cache immutability.
- Frozen tiny subset selection SHA `0f03efd352df6f2c8ea43950d0a9dda63a6dd2e60471aebad2da61d5d5859144`: all 4 windows PASS with unchanged targets and mapped representatives.
- Original failing window: no order assertion; 243 owned representatives match; forward, total loss, and backward gradients finite. Total loss was 2.801022053 (main event 2.064345837, main timing 0.736676097); no optimizer was created.
- Full train loader dry-run: 35,573 / 35,573 windows, failures 0.
- Full validation loader dry-run: 1,078 / 1,078 windows, failures 0.
- All 2,133 serialized mapping files pass file-SHA and alignment-config identity checks.

## Research questions

**Q1. Exact root cause?** Tempo-aware normalization rounded two adjacent raw onset ticks to the same PT tick, and PT's normalized-onset/pitch sort permuted them. The old dataset incorrectly assumed raw cache order and PT order were identical.

**Q2. How common?** One of 2,062 train performances, 14 of 9,369,095 train notes; no validation performance.

**Q3. Same onset only?** They cross raw onset ticks, but every movement is within one normalized PT onset; cross-normalized-onset movement is zero.

**Q4. Is the modeled universe bijective?** Yes, exactly, for all 9,653,023 train+validation notes.

**Q5. Ambiguous/unmatched mappings?** Zero. The implementation fails explicitly if either appears.

**Q6. Representative identity exact?** Yes. Every one of 9,282,476 cached onset representatives maps to the exact audited PT musical note identity.

**Q7. Ownership preserved?** Yes, exact canonical counts with zero missing/duplicate owners.

**Q8. Full loader dry-run?** PASS, 36,651 windows total, failures zero.

**Q9. Pure alignment fix?** Yes. Tokenizer/cache, official PT sequence, targets, representative semantics, architecture, loss, and event weights are unchanged.

**Q10. Readiness?** **READY TO RESTART FRESH FULL TRAINING.** The prior failed run must remain provenance-only; a future run must start fresh seed 42 and assert both cache ID and alignment ID.
