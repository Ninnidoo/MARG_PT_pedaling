# Stage 2 Binary Locked Test Listening Render Report

## Scope

This run generated listening artifacts only. It performed no model inference, training, checkpoint selection, threshold/decoding change, or metric recomputation.

## Selected pieces

Selection was deterministic from locked final-test piece metrics: `Original_PT_piece_JS - independent_4x2_piece_JS`; ties use lexical `piece_id` order.

| num | Piece | category | Original PT JS | independent JS | JS improvement |
|---:|---|---|---:|---:|---:|
| 19 | Mozart — Piano_Sonatas_12-2 (`piece_ff805dac200f8928`) | top4_improved | 0.596461304846 | 0.253704290899 | +0.342757013947 |
| 22 | Schumann — Kreisleriana_1 (`piece_307fd9e4ab88aeef`) | top4_improved | 0.595583314797 | 0.254346424363 | +0.341236890434 |
| 17 | Liszt — Transcendental_Etudes_10 (`piece_6d7bceed83883039`) | top4_improved | 0.357939217361 | 0.184667702846 | +0.173271514514 |
| 6 | Beethoven — Piano_Sonatas_32-1 (`piece_cde7e5eaa9e2232c`) | top4_improved | 0.228272131925 | 0.072118131050 | +0.156154000876 |
| 5 | Beethoven — Piano_Sonatas_2-1 (`piece_51d2b7fd3b9c6059`) | bottom2_degraded | 0.188745338652 | 0.576536501550 | -0.387791162898 |
| 9 | Beethoven — Piano_Sonatas_7-1 (`piece_736e242fbcbeaa28`) | bottom2_degraded | 0.131352043214 | 0.335022721590 | -0.203670678377 |

## Old Original PT reuse audit

- BYTE_IDENTICAL: 0
- SEMANTICALLY_IDENTICAL: 0
- DIFFERENT: 6
- Existing Original PT WAV reused: 0/6

The semantic comparison covered ordered tracks and every MIDI/meta event at its absolute tick, including notes, velocity, CC64 and other controls, programs/channels, tempo, time/key signatures, markers/lyrics, pitch bends, and end-of-track timing.

## Renderer provenance

- Existing renderer source: `/workspace/project/src/listening_comparison_5class_v0.py` (`7e45e15c1f4e598c37cfda68d6a4a5b8a7278247370e0ef015893524f61310f3`)
- Executable: `/workspace/private/midi_rendering/sfizz/install/bin/sfizz_render` (`7336673f524b487a513b728b5814cf36c706ce73cf6781c779194517d0b9b312`)
- Instrument: `/workspace/private/midi_rendering/instruments/SalamanderGrandPianoV3/SalamanderGrandPianoV3.sfz` (`1326f680002fe112c3cd74b298f66fc78089e325bd9937d89e5efff58f3ce0d2`)
- Parameters: `48000` Hz, `2` channels, default sfizz gain, `--use-eot`, no post-processing
- Historical command logs verified: 6/6

## Listening pairs

| num | Piece | category | Original PT audio | Developed PT audio |
|---:|---|---|---|---|
| 19 | Mozart — Piano_Sonatas_12-2 | top4_improved | `/workspace/project/outputs/audio/19_original_pt_locked_test.wav` | `/workspace/project/outputs/audio/19_developed_PT.wav` |
| 22 | Schumann — Kreisleriana_1 | top4_improved | `/workspace/project/outputs/audio/22_original_pt_locked_test.wav` | `/workspace/project/outputs/audio/22_developed_PT.wav` |
| 17 | Liszt — Transcendental_Etudes_10 | top4_improved | `/workspace/project/outputs/audio/17_original_pt_locked_test.wav` | `/workspace/project/outputs/audio/17_developed_PT.wav` |
| 6 | Beethoven — Piano_Sonatas_32-1 | top4_improved | `/workspace/project/outputs/audio/6_original_pt_locked_test.wav` | `/workspace/project/outputs/audio/6_developed_PT.wav` |
| 5 | Beethoven — Piano_Sonatas_2-1 | bottom2_degraded | `/workspace/project/outputs/audio/5_original_pt_locked_test.wav` | `/workspace/project/outputs/audio/5_developed_PT.wav` |
| 9 | Beethoven — Piano_Sonatas_7-1 | bottom2_degraded | `/workspace/project/outputs/audio/9_original_pt_locked_test.wav` | `/workspace/project/outputs/audio/9_developed_PT.wav` |

## Integrity

- Selected pieces: 6 unique (top four improved + bottom two degraded)
- Test-score number mapping: exact locked score SHA-256 → canonical numbered score SHA-256
- Developed renders: 6/6
- WAV checks: readable PCM, 48 kHz stereo, finite non-zero duration/audio, and no obvious truncation versus baseline
- Joint-16 MIDI was not rendered
- No existing Original PT audio was overwritten
