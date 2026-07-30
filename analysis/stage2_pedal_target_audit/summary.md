# Stage 2 Pedal Target Audit

## Scope and definitions

- Source: raw repository-provided human performance MIDI.
- No Pianist Transformer pedal tokens, model predictions, training, or GPU code were used.
- Note onsets at the same tick are grouped; IOIs are `[t_i, t_(i+1))` in tempo-aware seconds.
- Baseline OFF is CC64 `<64`; ON is CC64 `>=64`; DOWN is OFF-to-ON and UP is ON-to-OFF.
- CC64 is interpreted as a zero-order-held signal with initial value 0.
- Near-threshold rapid reversal means every raw CC64 value from the first crossing event through the opposite crossing event is in `[56,72]`.
- A long IOI warning uses a transparent audit threshold of `5` seconds.

## Inventory

- MIDI files discovered: 166
- CC64-containing files analyzed: 165
- Files without CC64: 3-1.mid
- Parsing failures: 0
- Raw CC64 channel distribution: {0: 659048}
- Multi-channel conflict files: 0

## Baseline transition and two-slot coverage

- Total IOIs: 562,605
- IOI counts for 0/1/2/3/4+ transitions: 494,743 / 56,019 / 11,146 / 501 / 196
- Two-slot lossless coverage `P(count<=2)`: 0.998761
- Overflow IOIs: 697 (0.123888%)
- Baseline UP / DOWN inside valid IOIs: 40,289 / 40,309.
- Baseline UP / DOWN on the full MIDI timeline: 40,445 / 40,457; 304 transitions are outside main IOIs.

## Slot class balance

The first table includes the first two events from overflow IOIs; the second excludes overflow IOIs entirely. No training treatment is selected here.

### First two with overflow included

- Combined NONE / UP / DOWN: 1,045,505 / 39,722 / 39,983
- Candidate normalized inverse-frequency weights: {'NONE': 0.35874529533574684, 'UP': 9.44237450279442, 'DOWN': 9.380736813145587}

### Overflow IOIs excluded

- Combined NONE / UP / DOWN: 1,045,505 / 39,025 / 39,286
- Candidate normalized inverse-frequency weights: {'NONE': 0.3583008530168037, 'UP': 9.599111680546658, 'DOWN': 9.535339136927488}

## Relative transition time

- All events: median tau 0.3914, p10-p90 0.0530-0.8880.
- UP median tau: 0.4178; DOWN median tau: 0.3750.
- Near onset: tau <0.05 9.561%; tau <0.10 16.570%.
- Near next onset: tau >0.90 8.861%; tau >0.95 4.321%.

## Pedal ON-depth

- IOIs ever ON: 339,680 (60.376%).
- Continuously OFF / continuously ON / partially ON: 39.624% / 48.588% / 11.788%.
- Whole-ON-time weighted mean CC64: 112.772; median: 127.0.
- Time-weighted tertile boundary values: 122, 127.
- CC64=127 accounts for 64.691% of ON time. Because q2 equals 127, strict value-threshold tertiles collapse to two occupied classes; splitting the tied 127 mass would require an additional arbitrary rule.

| Method | Class | ON-IOI count | ON-IOI proportion | ON-time proportion | Representative CC64 |
| --- | --- | ---: | ---: | ---: | ---: |
| equal_width | LIGHT | 71,067 | 20.922% | 18.670% | 73.0 |
| equal_width | MEDIUM | 35,553 | 10.467% | 10.686% | 93.0 |
| equal_width | DEEP | 233,060 | 68.612% | 70.644% | 127.0 |
| time_weighted_tertile | LIGHT | 119,605 | 35.211% | 33.373% | 82.0 |
| time_weighted_tertile | MEDIUM | 220,075 | 64.789% | 66.627% | 127.0 |
| time_weighted_tertile | DEEP | 0 | 0.000% | 0.000% | n/a |

Equal-width boundaries are directly interpretable fixed ranges. Quantile boundaries target ON-time balance only when ties can be separated; here the dominant CC64=127 tie prevents three occupied value-threshold classes. IOI counts also need not follow ON-time proportions because one IOI target uses its time-weighted median depth.

## Crossing sensitivity

| State machine | Full-timeline UP | Full-timeline DOWN | Main-IOI total | Two-slot coverage | Rapid <=200 ms | Near-threshold rapid <=200 ms | Net main-IOI removed vs baseline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_64 | 40,445 | 40,457 | 80,598 | 0.998761 | 28,000 | 3,115 | 0 |
| narrow_68_60 | 38,432 | 38,445 | 76,583 | 0.998861 | 25,467 | 1,132 | 4,015 |
| wide_72_56 | 36,411 | 36,430 | 72,571 | 0.998989 | 23,404 | 37 | 8,027 |

Rapid reversal is an audit category, not a noise label: short UP-to-DOWN pairs can be genuine repedaling. Each hysteresis event is timing-matched to the latest same-direction baseline crossing since the previous hysteresis event; unmatched baseline events and net count reduction are both retained in the JSON.

## Data-quality observations

- CC64 before first onset: 1,147 events across 128 files.
- CC64 at or after last onset: 5,800 events across 155 files.
- Opposite transitions at the same timestamp: 1.
- IOIs longer than 5 s: 5.
- High near-threshold crossing-rate files (>= empirical p95): 2-2.mid, 6-10.mid, 6-11.mid, 6-12.mid, 6-13.mid, 6-14.mid, 6-15.mid, 11-0.mid, 17-5.mid, 17-14.mid.
- High transition-rate files (>= empirical p95): 2-0.mid, 2-1.mid, 3-0.mid, 3-2.mid, 3-3.mid, 3-4.mid, 3-5.mid, 18-0.mid, 19-1.mid.

## Objective architecture/loss observations

- The measured two-slot coverage quantifies representational fit; overflow examples remain fully enumerated in `overflow_intervals.csv`.
- NONE, UP, and DOWN frequencies plus candidate inverse-frequency weights are reported, but no class weighting is selected.
- Tau boundary concentration is reported separately from class labels, so event-time regression choices can be evaluated without changing transition definitions.
- Equal-width and time-weighted-tertile depth labels expose different balance/interpretability tradeoffs; neither is selected as final.
- Baseline and hysteresis outputs are sensitivity analyses. They do not establish which rapid reversals are performance intent versus controller jitter.

## Reproduction

```powershell
& 'C:\Users\eagle\AppData\Local\Programs\Python\Python311\python.exe' 'scripts\analyze_stage2_pedal_targets.py'
```
