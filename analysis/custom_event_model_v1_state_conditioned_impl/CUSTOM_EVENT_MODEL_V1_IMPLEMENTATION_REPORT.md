# Custom Event Model v1 B3-S — Implementation Report

## Frozen implementation

- Model: 1.0.0-b3s; six Linear(768,4) state heads and six Linear(772,5) Main event heads.
- Conditioning: softmax(state_logits) as a fixed 4D posterior feature.
- Detach boundaries: encoder→state head and q→event head.
- State target: runtime derivation from the frozen train cache.
- State objective: unweighted Slot1–6 macro CE, fixed coefficient 1.0.
- Legacy checkpoint-selection objective excludes state CE.

## Correctness gates

- Targeted tests: PASS (tests.test_custom_event_model_v1_state_conditioned).
- Parameter accounting: PASS; v0→v1 delta 18,576.
- Gradient isolation A/B/C/D: PASS.
- Decoder ID/version unchanged: 938ca7e20e1d398aa530938bbcede404fd3a04af23842fbcf351ba02df0268f8 / 1.0.0.
- Tokenizer cache ID unchanged: 3a5520155b5db1e9a1da7f8148556aa3e1da852655c9adde25d3dbbd1d966263.
- Alignment ID unchanged: 98de59ef7a41fba26a2c89fe686c273f6c8ec7f27979922e71813624ac1a4822.

No validation inference, ASAP test, Repedal, full training, tokenizer rebuild,
decoder change, or candidate MIDI generation was performed.
