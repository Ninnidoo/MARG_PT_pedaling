# Binary 2-Slot Model Implementation v0

## 1. Status and scope

**PASS.** The dataset/representation adapter, stateless pretrained-encoder model, shared state-conditioned heads, online reconciliation, hard free-running rollout state, performance-major samplers, separated loss, diagnostics, unit tests, and canonical-train smoke are implemented. The three authoritative audits were treated as frozen specification; no threshold, interval, max-2 compression, reconciliation, class weight, or boundary-query rule changed.

No tiny overfit, epoch training, optimizer update, checkpoint creation/selection, validation inference/F1, or ASAP test access occurred.

## 2. Files and responsibilities

| File | Role |
|---|---|
| `src/stage2_binary_2slot/dataset.py` | Raw-MIDI binary primitives, canonical cache/PT alignment/owner reuse, leakage masking, boundary inputs, samplers/collate. |
| `src/stage2_binary_2slot/model.py` | Pretrained encoder, owner/boundary gather, `[dropout(h);s]`, shared Count/Timing heads. |
| `src/stage2_binary_2slot/rollout.py` | Online reconciliation, XOR state update, inference decoder, external state and diagnostics. |
| `src/stage2_binary_2slot/losses.py` | Fixed weighted CE, target-masked Smooth L1, separate MAIN/BOUNDARY losses. |
| `src/stage2_binary_2slot/trainer.py` | Sequential hard-state rollout core with state preserved across chunks. |
| `tests/test_binary_2slot_model_v0.py` | New unit/invariant suite and direct runner. |
| `scripts/smoke_binary_2slot_model_v0.py` | Actual checkpoint/train-data forward plus exactly one backward; no optimizer/checkpoint write. |

No pre-existing source file was modified. Existing Stage2 binary, Custom Event v0, PT source, caches, alignment files, and artifact IDs remain untouched.

## 3. Reused components and data flow

The implementation reuses `BinaryStage2EncoderBase`, official `midi_to_ids`, Custom Event cache provenance and PT-token checks, `NoteAlignment.cache_to_pt`, `assign_unique_owners`, `parse_raw_midi`, canonical tick/second conversion, and frozen `reconcile_and_compress/state_after`.

The Main representation remains the PT hidden at `cache_to_pt[onset_representative_note_index[i]]` in the unique maximum-margin owner window. The adapter verifies source SHA, raw/cache onset and note-end identity, PT pitch/IOI/velocity/duration identity, exactly-one ownership, monotonic owner indices, chronological owned onsets, and representative containment. It never reads the old 6-slot, Initial, Terminal, 4-state, or 5-class targets in `__getitem__`.

The dataset returns human interval start state and chronological, uncompressed binary crossings. At rollout time the trainer compares current model/human state, optionally prepends the frozen tau-zero correction, invokes frozen max-2 compression, builds target N/timing masks, predicts, then updates state with Count argmax parity. Human information and current state are not model members.

## 4. Input and boundary representations

Every active real note is `Pitch, IOI, Velocity, Duration, MASK, MASK, MASK, MASK`. Leakage is checked during masking, collate, and at the model encoder entry.

- PRE: attended `[MASK]*8` plus the first at most 511 masked real notes; query position 0.
- POST: last at most 511 masked real notes plus attended `[MASK]*8`; query position equals actual context length.
- No fake MIDI note, learned boundary parameter, or same-pass boundary token exists.
- PRE, MAIN, and POST use the same Count/Timing head objects.

The canonical smoke used 366 real notes: Main `[1,2928] -> [1,366,768]`; PRE/POST `[1,2936]`, query positions 0/366, each yielding `[768]`.

## 5. Model and parameters

Loaded checkpoint: `/workspace/project/checkpoints/pianist_transformer`; `model.safetensors` SHA-256 `8fb9111efc147adcc61a4885bd42a6e6b625626d9fb8576811916927aa8281d0`.

| Component | Shape/config | Parameters |
|---|---:|---:|
| PT encoder | hidden 768, 10 layers | 103,271,424 |
| Count | `Linear(769,3)` | 2,310 |
| Timing 1 | `Linear(769,1)` | 770 |
| Timing 2 | `Linear(769,1)` | 770 |
| New heads | seed 42 | **3,850** |
| Full model | | **103,275,274** |

Dropout applies only to `h`, before concatenating the untouched scalar state. There are no Initial/Terminal heads, learned queries, target members, or current-state buffers.

## 6. Performance-major rollout and loss

The sampler shuffles performances only, keeps their windows chronological, and the batch sampler never crosses a performance boundary. The trainer lifecycle enforces PRE exactly once, strictly consecutive MAIN onset indices, and POST exactly once. Trainer-owned state persists across window, micro-batch, accumulation, and optimizer-step boundaries and resets only at performance lifetime boundaries; Dataset/DataLoader workers own no state.

Count CE uses fixed weights `[0.7577816791288143, 2.246984238727598, 4.428054342343362]`. Timing uses Smooth L1 beta 0.1 only on target-N-active slots. MAIN and BOUNDARY Count/Timing losses are returned and logged separately. `boundary_loss_weight` has no default and must be explicit; smoke's explicit 1.0 is provisional only.

Diagnostics aggregate target/predicted N counts, corrections, state agreement/mismatch, dynamic real-transition retention, region counts, and invalid/non-finite tensors. Inference decoding is isolated from training reconciliation: it clamps timing, sorts N=2, derives alternating directions from state, and updates by parity.

## 7. Unit and regression verification

The container has no `pytest` (`No module named pytest`), so every test function ran through direct runners as permitted.

- New Binary 2-Slot suite: 15/15.
- Frozen Binary 2-Slot primitives: 8/8.
- Existing Stage2 binary regression: 10/10.
- Existing Custom Event v0 regression: 13/13.
- Compile check: PASS.

Total: **46 passed, 0 failed**. Coverage includes masking and leakage rejection, PRE/POST construction and short POST position, active attention/max 512, shapes/3,850 head parameters/shared stateless heads, hard state/direction, reconciliation including frozen K=2 correction discard, final-state invariant, target timing masks, loss/backward, ownership/order, performance-major batching, and cross-chunk state persistence.

## 8. Actual checkpoint smoke

Canonical train performance index 72 (`2018/MIDI-Unprocessed_Recital16_MID--AUDIO_16_R1_2018_wav--3.midi`) has 366 notes, one owner window, and 363 MAIN intervals. Its complete 365-interval PRE -> MAIN -> POST rollout passed.

- Total loss 7.1277308464, MAIN 3.4851799011, BOUNDARY 3.6425509453; all finite.
- Exactly one backward call.
- 128/129 encoder parameter tensors participated and every produced encoder gradient was finite.
- Every Count/Timing head parameter had a finite gradient.
- CUDA peak allocated memory: 1,633,068,032 bytes.
- Optimizer objects/steps, training steps, epochs, checkpoint writes, validation inference/F1, ASAP test metadata/MIDI access: all zero.

The random new head predicted N=2 throughout this smoke; those rollout statistics only validate plumbing and are not quality evidence.

## 9. Issues and next decisions

No blocking architecture or data issue was found. The prior semantic warning remains: an all-MASK note is mechanically valid/attended but was not pretrained as a dedicated CLS/boundary query. Separate passes ensure it cannot perturb Main representations.

Before tiny overfit, explicitly decide:

1. scientific `boundary_loss_weight` (do not inherit smoke-only 1.0);
2. tiny-run micro-batch, accumulation, dropout, and encoder optimization settings;
3. the scoped epoch-driver wrapper around the implemented sampler/rollout core, preserving exact region logging.

Work stops here; no training was started.
