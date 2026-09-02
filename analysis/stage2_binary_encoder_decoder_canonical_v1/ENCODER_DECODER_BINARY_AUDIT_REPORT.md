# Binary Encoder-Decoder Canonical-v1 Audit Report

## 1. Verdict

**READY FOR AN EXPLICITLY AUTHORIZED FULL TRAINING RUN.** The binary adaptation, cache/config scaffold, free-running inference, and canonical CC64-only structural path are implemented and tested. Full training was not started.

The inherited overlap policy is scientifically imperfect but explicit: every 512-note window starts its own decoder from BOS, stride is 256, raw free-running step logits are averaged in overlaps, and argmax is applied once. This matches the actual existing 5-class path; it does not merge autoregressive histories.

## 2. Existing 5-class encoder-decoder source and architecture

- Model: `src/stage2_encoder_decoder/model.py::EncoderDecoderFiveClassModel`
- Training: `src/stage2_encoder_decoder/train.py::train`, `optimizer_step`, `validation_epoch`
- Free-running evaluation: `src/stage2_encoder_decoder/evaluate.py::infer_performance`
- Checkpoint load: `src/stage2_encoder_decoder/evaluate.py::_load_model`
- Encoder initialization: `from_pretrained_encoder` loads official `PianoT5Gemma`, retains `get_encoder()`, and leaves it trainable.
- Encoder: 10 layers, hidden 768.
- Decoder: native `T5GemmaDecoder`, 2 causal layers, hidden 768, FFN 3072, 8 attention heads, 4 KV heads, head dimension 128, cross-attention hidden 768, RoPE maximum 8192, dropout 0.
- Decoder input: native token embedding plus learned Pedal1--4 slot embedding.
- Five-class output: one shared `Linear(768,5)`.

## 3. Decoder sequence semantics

`flatten_pedal_targets` uses a contiguous view of `[B,N,4]`, so the actual order is:

`P1_1, P2_1, P3_1, P4_1, P1_2, P2_2, P3_2, P4_2, ...`

This exactly matches the requested note-major causal order. Native decoder causal self-attention makes prediction `y_t` conditional on encoder states and decoder history `y_<t`, never future tokens.

## 4. BOS, PAD, shifting, and teacher forcing

The existing five-class IDs are classes 0--4, BOS 5, PAD 6, vocabulary 7. `shift_right_pedal_targets` constructs `BOS,y_1,...,y_(L-1)`. `IGNORE_INDEX=-100` positions become PAD and are masked from both decoder attention and loss. EOS is configured as PAD, but generation never uses early stopping; every valid window generates exactly `4N` tokens.

The binary scaffold uses OFF 0, ON 1, BOS 2, PAD 3, vocabulary 4 and the same shift/mask semantics. Loss is unweighted two-class cross entropy. Scheduled sampling code exists in the historical 5-class model, but the new scaffold neither imports nor enables it.

## 5. Free-running inference

`BinaryPedalEncoderDecoderModel.greedy_from_encoder` starts with BOS, performs deterministic argmax, feeds each predicted class as the next decoder input, and uses KV cache until exactly `4N` predictions exist. The public greedy path rejects any supplied ground-truth target.

## 6. Window and overlap policy

The actual 5-class `infer_performance` policy is retained:

- 512-note windows, stride 256;
- every overlapping window independently starts from BOS;
- no probability averaging and no center/first-window selection;
- raw generated-step logits are averaged per note/slot across windows;
- final class is one argmax after averaging.

Ambiguity: two windows can condition an overlap note on different BOS-reset histories. Logit averaging combines outputs but cannot make those causal histories identical. The policy is frozen for minimal-change comparability, and this limitation must accompany future results.

## 7. Exact binary adaptation

- New model: `src/stage2_binary_encoder_decoder/model.py::BinaryPedalEncoderDecoderModel`
- New inference: `src/stage2_binary_encoder_decoder/inference.py::infer_binary_encoder_decoder_pedals`
- Training scaffold: `src/stage2_binary_encoder_decoder/scaffold.py`
- Classes/head/vocabulary only: 5→2, `Linear(768,5)`→`Linear(768,2)`, vocab 7→4.
- Existing `binarize_raw_pedals` supplies threshold 64 targets.
- Existing `make_masked_binary_sample` supplies `[Pitch, IOI, Velocity, Duration, MASK×4]` without pedal leakage.
- Actual initialized model: encoder 10 layers, decoder 2 layers, hidden 768, output head [768, 2], vocab 4.

## 8. Fair comparison scaffold

- Shared cache ID: `85a79e9d10e955b10000f72f6bbc4a29dbd2cc5a4c8cfb1277054a8d45dcb877`.
- Training: 2062 performances / 9369095 notes / 35573 windows.
- Human validation diagnostics: 71 performances / 283928 notes / 1078 windows.
- Same seed 42, official pretrained encoder, AdamW encoder LR 1e-5 and decoder/head LR 1e-4, weight decay 0.01, gradient clip 1.0, AMP, effective batch 16, 512/256 windows.
- The scaffold conservatively fixes micro-batch 4 with accumulation 4 from the measured 5-class architecture; this audit did not run a binary batch-size search. Encoder-only physically used batch 16; effective batch remains 16.
- Future checkpoint selection is minimum free-running canonical strict validation JS. Teacher-forced loss/accuracy is diagnostic only.
- The historical 5-class run trained on ASAP train only and used weighted 5-class CE plus teacher-loss selection. Those choices are not reused; the binary canonical cache and unweighted binary objective are the controlled source of truth.

## 9. Checkpoint and MIDI compatibility

Future checkpoints contain `model_state` plus configuration and reload through strict `load_state_dict`. The existing five-class validation did not render MIDI. The new path produces binary 0/127 pedal tokens, uses the validated `ids_to_midi`/`map_midi` donor conversion, transplants CC64 only, and gates on raw non-CC64 equality before any official MIDI-roundtrip metric.

## 10. Focused tests

- Synthetic tests: **7/7 PASS**.
- Verified threshold/flatten order, one-step teacher shift, future-token causal masking, free-running self-history, `[B,4N]→[B,N,4]` reconstruction, encoder pedal masking, unweighted gradients, and disjoint optimizer groups.

## 11. Canonical validation structural smoke

- Scope: one frozen validation piece `piece_b06ab236f236fd76`; no test MIDI.
- Initialized/untrained model; 688 notes, 2 windows.
- Predicted Pedal1--4: generated successfully.
- Donor → CC64-only transplant → candidate reload: PASS.
- Strict non-CC64 equality: **PASS**.
- Candidate: `/workspace/project/analysis/stage2_binary_encoder_decoder_canonical_v1/structural_smoke/candidate_cc64_only.mid`.
- JS/Intersection: deliberately not computed; initialized-model metric quality is scientifically meaningless.

## 12. Isolation and stop point

- ASAP test Human MIDI accesses: **0**.
- Canonical test MIDI Stage 2 inference: **0/23**.
- Stage 1 PT neural inference: **0**.
- Full training/optimizer steps on research data: **0**.
- Scheduled sampling, class weighting, calibration, audio, and test-based decisions: **0**.

The scaffold is ready, but starting full training requires a separate explicit instruction.
