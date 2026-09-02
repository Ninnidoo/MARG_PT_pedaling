# Custom Event Tokenizer Oracle Audit v0

## Scope and provenance

- Train structural universe: 2,062 original performance MIDIs (MAESTRO-clean 1,170 + ASAP train 892).
- Oracle universe: ASAP validation 71 performances / 19 pieces, same-performance self-roundtrip.
- Raw provenance: original performance MIDI control_change 64 messages via mido; no Pedal1-4 target/cache reconstruction.
- Same-tick rule: per-track file order preserved; equal-tick global deterministic key=(tick,track_index,message_index); last effective CC64 state wins.
- ASAP test access: **0**; Repedal execution: **0**; training/model/optimizer/checkpoint activity: **0**.

## Structural compression

| Dataset | Intervals | 0 events | 1 | 2 | 3 | 4 | 5+ | ≥3 fraction | Event retention | Crossing retention |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MAESTRO-clean | 6127543 | 4949054 | 610044 | 296321 | 144028 | 65749 | 62347 | 0.044410 | 0.696826 | 0.681326 |
| ASAP-train | 2882007 | 2339666 | 295217 | 136529 | 61575 | 25447 | 23573 | 0.038374 | 0.723240 | 0.710380 |
| combined | 9009550 | 7288720 | 905261 | 432850 | 205603 | 91196 | 85920 | 0.042479 | 0.704890 | 0.690245 |

## Initial state

| Dataset | ZERO | LOW | HALF | FULL |
| --- | ---: | ---: | ---: | ---: |
| MAESTRO-clean | 383 | 134 | 141 | 512 |
| ASAP-train | 371 | 108 | 124 | 289 |
| combined | 754 | 242 | 265 | 801 |

## Combined slot distributions

| Slot | NONE | SET_ZERO | SET_LOW | SET_HALF | SET_FULL |
| --- | ---: | ---: | ---: | ---: | ---: |
| Slot1 | 7288720 | 215821 | 485608 | 694914 | 324487 |
| Slot2 | 8451220 | 129206 | 91922 | 164407 | 172795 |

## Worst overflow performances

### Highest >=3-event interval fraction

| Dataset | Performance | Intervals | Overflow intervals | Fraction |
| --- | --- | ---: | ---: | ---: |
| MAESTRO-clean | `2014/MIDI-UNPROCESSED_11-13_R1_2014_MID--AUDIO_13_R1_2014_wav--1.midi` | 1671 | 404 | 0.241771 |
| MAESTRO-clean | `2013/ORIG-MIDI_02_7_7_13_Group__MID--AUDIO_18_R1_2013_wav--1.midi` | 1488 | 350 | 0.235215 |
| MAESTRO-clean | `2014/MIDI-UNPROCESSED_01-03_R1_2014_MID--AUDIO_01_R1_2014_wav--2.midi` | 504 | 117 | 0.232143 |
| ASAP-train | `Beethoven/Piano_Sonatas/21-2/YOO05M.mid` | 486 | 108 | 0.222222 |
| MAESTRO-clean | `2013/ORIG-MIDI_01_7_6_13_Group__MID--AUDIO_02_R1_2013_wav--1.midi` | 1479 | 321 | 0.217039 |
| ASAP-train | `Schumann/Kreisleriana/4/ParkJH07.mid` | 661 | 135 | 0.204236 |
| ASAP-train | `Beethoven/Piano_Sonatas/26-2/Huang02.mid` | 864 | 173 | 0.200231 |
| MAESTRO-clean | `2013/ORIG-MIDI_02_7_7_13_Group__MID--AUDIO_15_R1_2013_wav--1.midi` | 1426 | 270 | 0.189341 |
| ASAP-train | `Bach/Prelude/bwv_867/HuNY01M.mid` | 708 | 133 | 0.187853 |
| MAESTRO-clean | `2018/MIDI-Unprocessed_Recital16_MID--AUDIO_16_R1_2018_wav--3.midi` | 363 | 68 | 0.187328 |

### Highest dropped-crossing fraction

| Dataset | Performance | Source crossings | Dropped | Fraction |
| --- | --- | ---: | ---: | ---: |
| ASAP-train | `Bach/Prelude/bwv_848/Lin04M.mid` | 1 | 1 | 1.000000 |
| ASAP-train | `Bach/Prelude/bwv_848/Lou01M.mid` | 1 | 1 | 1.000000 |
| ASAP-train | `Bach/Prelude/bwv_848/Mizumoto03M.mid` | 1 | 1 | 1.000000 |
| ASAP-train | `Bach/Prelude/bwv_860/YoungS01M.mid` | 1 | 1 | 1.000000 |
| ASAP-train | `Bach/Fugue/bwv_883/KaiRuiR03.mid` | 4 | 3 | 0.750000 |
| MAESTRO-clean | `2018/MIDI-Unprocessed_Recital9-11_MID--AUDIO_10_R1_2018_wav--3.midi` | 209 | 152 | 0.727273 |
| MAESTRO-clean | `2013/ORIG-MIDI_02_7_7_13_Group__MID--AUDIO_19_R1_2013_wav--1.midi` | 86 | 61 | 0.709302 |
| MAESTRO-clean | `2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_06_Track06_wav.midi` | 164 | 115 | 0.701220 |
| MAESTRO-clean | `2018/MIDI-Unprocessed_Recital1-3_MID--AUDIO_03_R1_2018_wav--4.midi` | 309 | 211 | 0.682848 |
| MAESTRO-clean | `2013/ORIG-MIDI_02_7_7_13_Group__MID--AUDIO_16_R1_2013_wav--1.midi` | 46 | 31 | 0.673913 |

## Oracle ceiling

| Oracle | 4C Acc ↑ | Macro F1 ↑ | Trans P ↑ | Trans R ↑ | Trans F1 ↑ | JS ↓ | Intersection ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Unlimited Event | 0.999915 | 0.999916 | 1.000000 | 0.997970 | 0.998984 | 0.000001 | 0.999863 |
| Actual Two-Slot | 0.986777 | 0.982207 | 1.000000 | 0.741599 | 0.851630 | 0.010113 | 0.972447 |

## Slot-cap penalty

| Metric | Unlimited | Two-Slot | Δ (Two−Unlimited) |
| --- | ---: | ---: | ---: |
| 4C Accuracy | 0.999915 | 0.986777 | -0.013139 |
| Transition F1 | 0.998984 | 0.851630 | -0.147354 |
| JS | 0.000001 | 0.010113 | +0.010112 |
| Intersection | 0.999863 | 0.972447 | -0.027415 |

## Top source patterns and oracle probabilities

| Pattern ID | Source | Unlimited | Two-Slot |
| ---: | ---: | ---: | ---: |
| 255 | 0.448402 | 0.448476 | 0.456524 |
| 0 | 0.241804 | 0.241801 | 0.245016 |
| 170 | 0.106175 | 0.106189 | 0.109947 |
| 85 | 0.078249 | 0.078249 | 0.079760 |
| 191 | 0.008048 | 0.008065 | 0.007840 |
| 254 | 0.007622 | 0.007625 | 0.008175 |
| 175 | 0.005621 | 0.005625 | 0.006357 |
| 106 | 0.004557 | 0.004565 | 0.004448 |
| 90 | 0.004117 | 0.004117 | 0.004244 |
| 64 | 0.003730 | 0.003726 | 0.003659 |

## Coverage and invariants

- Interval final-state preservation: **100.000000%** across 9,009,550 interval checks (hard invariant PASS).
- Modeled effective events: 3,233,356; retained 2,279,160; dropped 954,196.
- Source crossings retained as slots: 950,351/1,376,831.
- Post-latest-note-off effective events occur in 1,453 source performances across train+validation diagnostics; these events are outside v0 support.
- Exact-on-onset raw CC events: 215,122/12,539,285 (1.715584%); effective events: 78,466/3,322,436 (2.361701%).
- First-onset tau=0 effective events: 23; cross-track/interleaved same-tick ambiguity count: 193.
- Unlimited full-source exact transition diagnostic: `{'candidate_crossings': 35400, 'dropped_source_crossings': 96, 'exact_retained_source_crossings': 35376, 'new_artificial_crossings': 24, 'source_crossings': 35472}`.
- Two-slot modeled-support exact transition diagnostic: `{'candidate_crossings': 26282, 'dropped_source_crossings': 13566, 'exact_retained_source_crossings': 21818, 'new_artificial_crossings': 4464, 'source_crossings': 35384}`.

## Answers

1. Absolute four-state SET representation ceiling is quantified by the Unlimited row; its remaining gap is quantization/tie/terminal behavior, not model error.
2. The two-slot cap retains 70.488990% of modeled events and 69.024521% of source crossing events as slots.
3. Overflow frequency is reported source-wise above and per performance in `overflow_diagnostics.csv`.
4. The odd/even rule preserved every modeled interval final state exactly (100%).
5. Actual Two-Slot transition F1 ceiling is 0.851630; exact diagnostics separate slot loss from unsupported terminal events.
6. Initial-state imbalance and diagnostic inverse-sqrt weights are in the JSON/CSV; no loss was selected or applied.
7. Slot1/Slot2 imbalance and candidate weights are diagnostic only.
8. Exact tau summaries by slot/state, including boundary concentration, are in `train_structural_stats.json`.
9. Post-note-off coverage is explicit above and in `terminal_coverage_diagnostics.csv`.
10. Recommendation: **needs targeted design revision**. This is based on observed oracle/coverage facts; the tokenizer was not automatically redesigned.

## Interpretation boundary

Confirmed metrics/statistics are separated from the readiness recommendation. No arbitrary accuracy pass threshold, model implementation, loss, training, inference, ASAP test access, Repedal metric, or automatic tokenizer redesign was performed.
