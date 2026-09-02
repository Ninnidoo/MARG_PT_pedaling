# Custom Event Model v0 Implementation Report

## Scope and provenance

- Frozen tokenizer cache: `3a5520155b5db1e9a1da7f8148556aa3e1da852655c9adde25d3dbbd1d966263`
- Cache universe: train 2,062; validation 71; ASAP test access 0.
- Input: official PT Pitch/IOI/Velocity/Duration tokens; all pedal input tokens are MASK.
- Windowing: 512 notes / stride 256 using the stable Stage 2 tail-window generator.
- Pretrained loading: `BinaryStage2EncoderBase.from_pretrained -> PianoT5Gemma.from_pretrained -> get_encoder`
- Checkpoint: `/workspace/project/checkpoints/pianist_transformer`
- No model checkpoint warm-start, optimizer, training, candidate MIDI, or validation inference.

## Ownership

| Split | Windows | Onsets | Owned | Zero owner | Duplicate owner | Split-chord groups |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 35573 | 9009549 | 9009549 | 0 | 0 | 1411 |
| Validation | 1078 | 272927 | 272927 | 0 | 0 | 46 |

- Train initial/terminal supervision: 2062 / 2062.
- Validation initial/terminal supervision: 71 / 71.
- Owner choice: full chord required; maximum representative margin; exact ties choose smaller window start.
- Owned-onsets/window train mean/p50/p95: 253.269 / 249.0 / 365.0.
- Owner margin train mean/p50/p95: 184.912 / 189.0 / 249.0.

## Architecture

- PT Encoder-only hidden size: 768.
- Initial: one `Linear(768,4)`.
- Main: six independent `Linear(768,5)` event heads and six independent `Linear(768,1)` timing heads.
- Terminal: four independent `Linear(768,5)` event heads and four independent `Linear(768,1)` timing heads.
- Terminal uses the cached final-onset last-note hidden state.
- Slots are parallel and non-autoregressive.

| Parameters | Count |
| --- | ---: |
| Encoder | 103271424 |
| Initial head | 3076 |
| Main event heads | 23070 |
| Main timing heads | 4614 |
| Terminal event heads | 15380 |
| Terminal timing heads | 3076 |
| All heads | 49216 |
| Total | 103320640 |

## Objective

- Initial: unweighted CE, first-onset owner only.
- Main event: frozen train-frequency slot-specific weighted CE, macro-average over six slots.
- Main timing: Smooth L1 beta=0.1, only `event != NONE AND timing_valid`, macro-average over six slots.
- Terminal event: frozen train-frequency slot-specific weighted CE, macro-average over four slots, terminal owner only.
- Terminal timing: Smooth L1 beta=0.1, only `event != NONE AND timing_valid`, macro-average over four slots.
- Total is the positive weighted sum of all five components; all lambdas are 1.0.
- Empty timing slots return a differentiable finite zero.
- Weight SHA256: `26da9ede3373bd3df5c3f28c259e29f173a0920e080eb3a27cdbc5c790ed5b55`.
- Null zero-frequency terminal weights are stored as zero; canonical train has no corresponding targets.

## Real pretrained forward/backward smoke

- Performance: `2018/MIDI-Unprocessed_Recital16_MID--AUDIO_16_R1_2018_wav--3.midi`
- Notes / owned onsets: 366 / 363
- Losses: init 1.476695, main event 2.038455, main time 0.623324, terminal event 1.218550, terminal time 0.071635, total 5.428658.
- All logits, scalar predictions, and losses finite: yes.
- Encoder and every head gradient group finite and non-zero: yes.
- GPU: NVIDIA GeForce RTX 2080 Ti; peak allocated 880852480 bytes.
- Optimizer created: no. Optimizer steps: 0. Training steps: 0.

## Frozen boundary

Tokenizer/cache files were read-only. Window ownership does not alter global interval targets. No tiny overfit, full training, model validation inference, ASAP test, or Repedal evaluation was run.
