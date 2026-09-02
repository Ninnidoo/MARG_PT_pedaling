# Custom Event Prediction Decoder Specification v1 (Frozen)

- Decoder version: `1.0.0`
- Decoder ID: `938ca7e20e1d398aa530938bbcede404fd3a04af23842fbcf351ba02df0268f8`
- Tokenizer cache ID: `3a5520155b5db1e9a1da7f8148556aa3e1da852655c9adde25d3dbbd1d966263`
- Model checkpoint: canonical completed full-run `best.pt`, epoch 1 (read-only)

## Scope

This decoder restores the frozen Custom Event Tokenizer v1 grammar. It contains no confidence threshold, calibration, sampling, beam search, smoothing, minimum-duration rule, or validation-metric-tuned post-processing. Window overlap aggregation and MIDI writing are separate later-stage responsibilities.

## IDs and representatives

Initial state IDs are `ZERO=0`, `LOW=1`, `HALF=2`, `FULL=3`. Event IDs are `NONE=0`, `SET_ZERO=1`, `SET_LOW=2`, `SET_HALF=3`, `SET_FULL=4`. Representatives are `0, 51, 79, 127`.

All logits must be finite. Decoding uses deterministic argmax; ties select the lowest class index.

## Initial state

The initial argmax establishes one continuous piece-level state machine immediately before the first distinct onset. If Initial and Main `tau=0` share the first-onset timestamp, ordering is Initial establishment, Main SET events, then note-on messages.

## Main K=6

1. Decode every event head by argmax.
2. The first `NONE` terminates the left-packed slot sequence. Later slots are ignored, not revived.
3. All timing predictions must be finite. Clip active raw `tau` to `[0,1]`.
4. Keep each `(event, tau, original_slot_index)` pair intact.
5. Stable-sort active pairs by `(tau, original_slot_index)`.
6. Convert time using `left + tau * (right-left)` in the frozen continuous time domain.
7. Non-final intervals are `[t_i,t_(i+1))`; after time-to-tick conversion enforce `left_tick <= tick < right_tick`. The final main interval is `[t_M,T_end]` and allows `tick == latest_noteoff_tick`.
8. At one effective timestamp/tick, the last ordered destination SET wins. Zero-duration intermediate SETs are not emitted.
9. A SET whose destination equals current state is a no-op and is not emitted.

The state is never reset between intervals.

## Terminal K_T=4

1. The first `NONE` terminates the left-packed terminal sequence.
2. Decode active event heads by deterministic argmax.
3. All `z` predictions must be finite. Decode `z'=max(0,z)` and `gap=expm1(z')`.
4. Preserve Slot1→Slot4 order; terminal slots are never timing-sorted.
5. Use a cumulative clock: `T1=T_end+g1`, `Tj=T_(j-1)+gj`.
6. Decode/advance the clock before checking whether an absolute SET is a no-op.
7. Continuous zero gaps are preserved. At MIDI quantization, each active terminal slot is assigned a tick strictly greater than the prior active terminal tick, beginning at `latest_noteoff_tick+1`.

## Same-time and state continuity

Ordering is deterministic from continuous time, source priority, main interval, and original slot. Initial has priority before Main at the first onset. Main same-time/tick groups retain only the final destination; redundant destinations are suppressed against the single continuous Initial→Main→Terminal current state. Terminal active slots remain structurally ordered even when a no-op produces no CC64 message.

## Non-finite policy

NaN or infinity in any logit, timing prediction, or timeline boundary is a descriptive fatal decoder error. Only finite out-of-domain main `tau` and negative terminal `z` receive the frozen safety clamps.

## Determinism and provenance

There is no RNG or seed dependency. The Decoder ID is SHA256 over the canonical JSON semantic configuration in `prediction_decoder_v1.py`. Canonical validation inference must record this Decoder ID. This specification was frozen before any ASAP validation prediction, candidate MIDI, or evaluator metric was accessed.
