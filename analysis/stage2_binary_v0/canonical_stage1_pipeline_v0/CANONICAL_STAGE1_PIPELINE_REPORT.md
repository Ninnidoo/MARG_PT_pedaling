# Canonical ASAP Test Stage-1 Pipeline v0

## Permanent invariant

Root `AGENTS.md` now defines `outputs/midi/{num}_original_pt.mid` as the immutable CPU/seed-42 canonical ASAP test Stage 1 bank. Future Stage 2 candidates may transplant CC64 only; full token-to-MIDI reconstruction is prohibited as the final canonical candidate path, and strict non-CC64 equality is a hard pre-metric gate.

## Canonical bank

- Exact mapping: **23/23 PASS**, nums 0--22 exactly once
- Mapping provenance: every direct score candidate recorded in locked `test_score_manifest.csv` was hashed and joined to the exact SHA-256 of `third_party/PianistTransformer/data/midis/testset/score/{num}.mid`; each piece had exactly one numbered-score hash match
- Canonical MIDI: `outputs/midi/{num}_original_pt.mid`, read-only source of truth
- Total parsed notes: 61670
- Total MIDI duration: 4905.917494 seconds
- Original canonical hashes unchanged after all tests: **PASS**

## Pedal-only implementation

`src/stage2_binary/canonical_stage1.py::transplant_cc64_only` loads canonical and donor MIDI with Mido, removes only canonical CC64, inserts only donor CC64 at absolute ticks, and copies every canonical non-CC64 raw message. `assert_strict_non_cc64_equality` compares MIDI type, ticks-per-beat, ordered tracks, all note/channel/meta events, non-CC64 controls, pitch bends, tempo/time/key signatures, markers/lyrics, and EOT timing while ignoring CC64 only.

The donor MIDI still uses the established official `ids_to_midi` + `map_midi` path, but donor non-pedal content is discarded. Final candidate non-pedal events always come from the canonical MIDI.

Donor CC64 strictly after canonical end-of-track is discarded rather than shifted, because canonical EOT is immutable; every discard is recorded by the full evaluation audit.

## Reused implementation

- Official tokenizer/conversion: `third_party/PianistTransformer/src/utils/midi.py::midi_to_ids,ids_to_midi`
- Official mapping: `third_party/PianistTransformer/src/model/generate.py::map_midi`
- Binary masking/window/decode: `src/stage2_binary/validation_evaluator.py::infer_cached_binary_pedals`
- Locked model loading: `load_binary_stage2_checkpoint`
- Mapping rule: existing locked test-score manifest and listening audit's exact score-hash join
- Equality scope: strengthened from the existing metric-fidelity/listening equality audits to raw ordered MIDI events including EOT

## Verification

- Synthetic preservation: **PASS**
- Intentional velocity mutation hard-failure: **PASS**
- No-pedal canonical edge case: **PASS**
- Real canonical structural smoke: **PASS**, num=1, piece=`piece_8ced86fd9478a1c7`, notes=1207, windows=4
- Canonical tokenization and all four original pedal input slots masked: **PASS**
- Predicted pedal donor -> CC64-only transplant: **PASS**
- Strict ordered non-CC64 equality after reload: **PASS**
- Candidate reload: **PASS**

## Scope guard

This stage performed **no Stage 1 PT neural inference, no training/retraining, no checkpoint change, no Human ASAP test MIDI access, no JS/Intersection or other test metric computation, no full 23-piece Stage 2 inference, and no audio rendering**. Existing final lock, final-test, and listening-render artifacts were not modified.
