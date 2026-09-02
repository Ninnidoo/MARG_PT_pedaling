# Binary 2-Slot PT Architecture / Boundary-Query Feasibility v0

## 1. Scope and frozen design

This is a read-only architecture audit immediately before implementation. Binary thresholding, max-2 capacity, final-state-priority reconciliation, one-second PRE/POST horizons, hard free-running update, and fixed class weights were not changed. No model/head/dataset/trainer implementation, optimizer, backward pass, training, validation model inference, or ASAP-test access occurred.

The prior transition audit and follow-up were read first. Their relevant conclusions remain frozen: exactly-one monotonic owner ordering, 100% represented-horizon final-state preservation, the current reconciliation policy, and Count weights `(0.7577816791288143, 2.246984238727598, 4.428054342343362)`.

## 2. Provenance

- PT source: pinned commit `747df2d12291e37f6638b39f1b71517e579ad48c`, clean detached HEAD.
- Pretrained checkpoint read forward-only: `/workspace/project/checkpoints/pianist_transformer`.
- `model.safetensors` SHA-256: `8fb9111efc147adcc61a4885bd42a6e6b625626d9fb8576811916927aa8281d0`.
- Dry-run input: one canonical MAESTRO-clean train performance, performance index 0. No validation MIDI/data or ASAP-test metadata/MIDI was accessed; only the requested prior audit reports (including their saved validation summaries) were read.
- Stage2 checkpoint access, checkpoint creation, optimizer, backward, training, and validation inference: zero.

## 3. Actual PT input to encoder path

| Step | Code | Responsibility | Shape |
| --- | --- | --- | --- |
| MIDI features | `third_party/PianistTransformer/src/utils/midi.py:136-182`, `midi_to_ids` | Normalize, then Pitch/IOI/Velocity/Duration/Pedal1-4 | MIDI -> `[8N]` |
| Stage2 window | `src/stage2_event_model/dataset.py:182-239` | Tokenize, verify alignment, 512/256 slice, mask pedal slots | `[N,8]` -> `[W,8]` |
| Batch | `dataset.py:303-364` | PAD ids; token/note/owner masks | samples -> `[B,8Wmax]` |
| Note embedding | `pianoformer.py:100-126` | vocab `[5389,768]`; eight `Linear(768,96)`; concatenate | `[B,8W]` -> `[B,W,768]` |
| Mask | `pianoformer.py:174-201` | `view(B,W,8).any(-1)`; bidirectional masks | `[B,8W]` -> `[B,W]` |
| Position | `pianoformer.py:169-205` | note positions `0..W-1`, RoPE | `[B,W,768]` |
| Encoder | `pianoformer.py:204-223` | ten alternating sliding/full bidirectional layers, RMSNorm | `[B,W,768]` -> `[B,W,768]` |
| Stage2 wrapper | `src/stage2_binary/model.py:105-135` | official encoder call and shape assertion | `[B,8W]` -> `[B,W,768]` |
| Onset gather | `src/stage2_event_model/model.py:100-158` | gather unique owner-local representative | `[B,W,768]` -> `[B,O,768]` |

There is no learned absolute position embedding; note-index RoPE is used. Checkpoint config has `max_position_embeddings=8192` and `sliding_window=4096`, but the trained and operational block is 4,096 raw feature tokens = 512 notes (`pretrain_config.json`, `sft_config.json`, `model/generate.py:42-69`). Stage2 therefore correctly treats 512 notes as the supported window.

`PianoT5GemmaEncoder.embed_tokens` and top-level `PianoT5Gemma.embeddings` parameters exist, but active encoder `input_ids` calls `model.encoder.embeddings` at `pianoformer.py:165-167`. Keep that exact path. Full shape/provenance details are in `tensor_shape_trace.csv`.

## 4. Main onset representation

The cache defines each distinct-onset representative as its last note (`tokenizer_v1.py:120-131`; cache builder lines 60-62,177-182). Raw cache indices cannot be used directly because PT tempo normalization can reorder notes. `NoteAlignment.cache_to_pt` remains required and maps group bounds and representative (`note_alignment.py:52-86`).

`performance_from_entry` maps `onset_first_note_index`, `onset_last_note_index`, and `onset_representative_note_index`, then applies unchanged ownership (`dataset.py:70-103`). `assign_unique_owners` requires full chord containment, maximizes representative margin, and ties toward smaller start (`ownership.py:56-80`). Dataset/model use `owned_local_representative_indices` and gather those positions. The previous audit found zero missing, duplicate, or non-monotonic owners.

**Frozen Main definition:** `h_i` is `encoder.last_hidden_state` at `cache_to_pt[onset_representative_note_index[i]]`, localized inside onset `i`'s unique maximum-margin owner window; the cache representative is the last note of its distinct-onset group.

State does not alter encoder input, so head conditioning does not require re-running a computed owner-window encoder output.

## 5. Special-token audit

The checkpoint contains nonzero MASK/BOS/EOS/PLAY embedding rows and a zero PAD row, plus all eight active `[96,768]` projection layers.

- MASK 1: pretrained encoder corruption token. Pretraining independently masks 30% of attended feature tokens (`train/pretrain.py:28-98`); Stage2 masks Pedal1-4. It remains attended. Eight MASK ids make a complete pseudo-note without fake feature values.
- PAD 0: padding only. Canonical mask is zero and the 8-token `any` reduction masks the note. Reject.
- BOS 2: decoder start/shift-right only, not an encoder note. Reject.
- EOS 3: decoder configuration; no encoder insertion path. Reject.
- PLAY 4: symbol exists; no active encoder preprocessing use found. Reject.
- CLS-like token: absent.

MASK is mechanically usable, checkpoint-backed, attended, compression-compatible, and yields a normal 768-vector. It is not a dedicated CLS/boundary token. All-eight-MASK may occur under independent corruption, but is not explicitly constructed as a boundary objective. This is a semantic WARN, not a shape/attention blocker.

The real-note count/alignment assertions must remain intact. Insert the pseudo-note only after verified real-token loading in a new boundary batch builder, never as fake MIDI or part of the alignment permutation.

## 6. Same-pass effect

A deterministic pretrained forward compared 32 Stage2-masked real notes with candidate virtual notes.

| Layout | Max abs diff on real hidden | Mean abs diff |
| --- | ---: | ---: |
| real + POST MASK | 1.3305588961 | 0.0467328876 |
| PRE MASK + real + POST MASK | 1.6527584791 | 0.1052513272 |

Outputs were finite. POST-only changes earlier states, proving bidirectional influence independently of PRE position shift. PRE+POST also shifts real RoPE positions. Same-pass tokens therefore do not preserve Main representations. At the 512-note operational limit they also leave only 510 real notes.

**Decision:** do not inject boundary tokens into Main owner-window passes.

## 7. Separate boundary passes

The exact proposed layouts were tested on the real pretrained encoder:

- PRE: `[MASK]*8 + first 511 Stage2-masked real notes`; take output 0.
- POST: `last 511 Stage2-masked real notes + [MASK]*8`; take output 511.

Both inputs were `[1,4096]`, outputs finite `[1,512,768]`, query outputs `[1,768]`; peak allocation was 649,048,064 CUDA bytes. This reuses the encoder with zero new parameters, stays at 512 notes, and produces the Main hidden dimension. PRE/POST each use up to 511 real notes. Cropped context resets to window-local RoPE positions, as existing Stage2 windows already do.

Separate passes preserve Main tensors within their Main forwards. Cost: two extra encoder forwards per performance and a new post-verification boundary batch path.

## 8. Minimum alternatives

1. **Recommended:** separate all-MASK pseudo-note plus first/last 511 context. 0 parameters; Main unchanged; shared heads. Risk: no dedicated boundary objective.
2. Reuse first/last real-onset hidden. 0 parameters, but not boundary-specific and duplicates adjacent Main content. Fallback only.
3. Two learned 768-vectors. 1,536 parameters; Main unchanged; shared heads. Context-free/random and violates zero-query preference. Learned fallback only.

No fake MIDI note is required.

## 9. State-conditioned head

With scalar hard state, `z_i=concat(h_i,s_i)` is 769-dimensional. Minimum affine shapes:

- Count `[3,769]` plus `[3]`: 2,310 parameters.
- Timing1 `[1,769]` plus `[1]`: 770.
- Timing2 `[1,769]` plus `[1]`: 770.
- Total: 3,850.

All intervals produce 768-dimensional `h`, so heads can be shared. Encoder outputs are computed first; only heads roll chronologically. Hard `argmax`/XOR carries no gradient from later loss through an earlier Count decision. This is the frozen design, not a bug. Each interval loss still gradients to its own `h`, shared heads, and encoder. Sum losses before one backward per accumulation group.

## 10. Performance-major training

`CustomEventWindowDataset.windows` is constructed performance-major/window-major (`dataset.py:173-177`), but actual training sets `shuffle=True` over individual windows (`run_custom_event_model_v0_full_seed42.py:400-413`). That loader cannot carry chronological state.

Required engineering changes:

1. New Binary 2-Slot adapter and boundary builder; preserve v0.
2. Performance-aware sampler: shuffle performances, keep ascending windows within each performance.
3. Keep performance ID/window index in collate and one state per active performance.
4. Process PRE once, owned onsets globally in order, POST once.
5. Carry state across micro-batches, accumulation groups, and optimizer steps; reset only after performance.
6. Different performances may share a batch if states are independent and neither skips an earlier window; a frontier or concatenated performance-major sampler works.
7. Micro-batch 4 / accumulation 4 / effective 16 is compatible, but sampler and denominator bookkeeping change. State must not reset at optimizer steps.
8. Owner masks/positions remain exact. No overlap averaging or duplicate supervision.

## 11. PRE/POST loss aggregation

Custom v0 independently normalizes Initial, Main slot families, and Terminal slot families and sums unit-weight components (`losses.py:159-245`; `full_training.py:117-232`). Sparse branches therefore have full objective-scale weight, related to the prior Initial-loss domination concern.

- A, complete interval micro-average: no parameters, but 4,124 train boundaries are submerged by about 9.01M Main intervals.
- B, average MAIN and BOUNDARY(PRE+POST) separately, then combine: no parameters, protects boundary supervision with limited bookkeeping. Freeze/log combination scale before training.
- C, per-performance PRE/Main/POST aggregation then average: no parameters and balances piece lengths, but needs performance buffering/ragged denominators and changes long/short weighting.

**Recommendation: B.** It is the smallest safe protection and avoids two separately full-scale Initial/Terminal branches. No lambda is selected here.

## 12. Validation/checkpoint selection

Reusable pieces:

- unique-owner performance aggregation: `run_custom_event_model_v0_canonical_val_inference_v1.py:289-366`;
- timeline/tick conversion: lines 157-172;
- CC64-only MIDI reconstruction: lines 204-282;
- direction-aware one-to-one Transition P/R/F1 at +/-1 distinct onset: `stage2_four_class/validation_evaluator.py:286-330`;
- candidate/reference counts for their ratio.

New work: PRE->Main->POST hard rollout, Count/timing-to-toggle decoding, exact 1-second boundary reference, interval-start state agreement, mismatch-run recovery, and trainer integration selecting by full-performance pooled Transition F1. Current trainer selects validation loss; current v0 inference uses parallel absolute-SET heads. Neither is directly correct.

No validation inference ran in this audit.

## 13. Priority review files

See `files_to_review.csv`; it contains ten prioritized PT input/encoder, Stage2 dataset/owner/alignment/model/trainer/loss/inference/metric files.

## 14. Explicit answers

**Q1. Main `h_i`?** The 768-D PT encoder output at the mapped cache last-note representative, gathered only from its unique maximum-margin owner window.

**Q2. Head-only state conditioning possible?** Yes. Encoder stays state-independent; concat scalar state gives 769.

**Q3. Encoder once, sequential finite-state head?** Yes per window. Compute owner-window `H` once and reuse gathered `h_i` chronologically.

**Q4. Dedicated pretrained safe boundary token?** No dedicated CLS/boundary token. Best zero-parameter mechanism is a separate all-MASK pseudo-note pass with semantic WARN.

**Q5. MASK usable?** Mechanically yes: checkpoint-backed, pretrained in encoder input, attended, compression-compatible, finite. PAD is unusable. Insert after real-token/alignment checks.

**Q6. Same-pass changes Main?** Yes: PRE+POST max/mean absolute differences were 1.65275848/0.10525133; POST-only also changed earlier states.

**Q7. Separate pass?** Yes. Use `[MASK]*8 + first 511` and `last 511 + [MASK]*8`, taking first/last 768-D output.

**Q8. Pipeline changes?** New binary/boundary adapter, performance-major sampler/collate metadata, stateful rollout and accumulation, boundary-aware loss, full-performance validator. Owner/alignment stay unchanged.

**Q9. Safest simple boundary aggregation?** B: separately normalize MAIN and BOUNDARY, then combine with frozen/logged scale.

**Q10. Blocking issue?** No hard technical blocker. Explicitly accept the whole-note MASK semantic WARN and freeze loss combination scale before implementation.

**Q11. Review files?** The ten files in `files_to_review.csv`, led by `pianoformer.py`, `midi.py`, `stage2_event_model/dataset.py`, `note_alignment.py`, and `ownership.py`.

## 15. Conclusion

**WARN / FEASIBLE.** Main representation and owner carry are reusable. Head-only hard state does not require encoder repetition. Same-pass boundary tokens are rejected because they materially alter Main states. Separate all-MASK boundary passes are finite, shape-correct, zero-parameter, and 512-note compatible, but MASK is not a dedicated boundary token. Remaining work is a new dataset/sampler/trainer/evaluator path, not a blocking PT architecture change.
