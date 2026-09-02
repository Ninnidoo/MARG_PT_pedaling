# Custom Event Tokenizer Terminal Micro-Audit v0

## Scope and provenance

- Original performance MIDI raw CC64 events; no Pedal1–4 reconstruction.
- Terminal anchor: latest non-pedal note-off; only effective events strictly after it.
- Main representation fixed at K=6 chronological last-6; no main-K sweep.
- Existing v0/v1 artifacts were read-only; v1 terminal count/affected/median regression passed.
- Training/model/optimizer/checkpoint/full-oracle activity: 0. ASAP test: 0. Repedal: 0.

## Terminal event counts

| Dataset | Performances | Affected | All mean/p50/p75/p90/p95/p97.5/p99/p99.5/max | Affected mean/p50/p75/p90/p95/p97.5/p99/p99.5/max |
| --- | ---: | ---: | --- | --- |
| train | 2062 | 1400 | 1.611/2.0/3.0/3.0/3.0/4/4.0/5.0/9 | 2.372/2.0/3.0/3.0/3.0/4/5.0/5.0/9 |
| validation | 71 | 53 | 1.986/2.0/3.0/3.0/3.0/4/5.5/7.3/9 | 2.660/3.0/3.0/3.0/3.4/4/6.4/7.7/9 |

## Terminal capacity curve

| Dataset | K_T | Overflow all / affected | Event retention | Crossing retention | Final state | Last slot active all / affected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 1 | 56.3046% / 82.9286% | 42.1560% | 32.2853% | 100.0000% | 67.8952% / 100.0000% |
| train | 2 | 32.3472% / 47.6429% | 77.1153% | 95.3221% | 100.0000% | 56.3046% / 82.9286% |
| train | 3 | 2.8613% / 4.2143% | 97.1996% | 97.8528% | 100.0000% | 32.3472% / 47.6429% |
| train | 4 | 0.9699% / 1.4286% | 98.9762% | 99.3098% | 100.0000% | 2.8613% / 4.2143% |
| train | 6 | 0.1455% / 0.2143% | 99.7290% | 99.7699% | 100.0000% | 0.2425% / 0.3571% |
| train | 8 | 0.1455% / 0.2143% | 99.9097% | 100.0000% | 100.0000% | 0.1455% / 0.2143% |
| train | Unlimited | 0.0000% / 0.0000% | 100.0000% | 100.0000% | 100.0000% | N/A |
| validation | 1 | 66.1972% / 88.6792% | 37.5887% | 19.6429% | 100.0000% | 74.6479% / 100.0000% |
| validation | 2 | 46.4789% / 62.2642% | 70.9220% | 91.0714% | 100.0000% | 66.1972% / 88.6792% |
| validation | 3 | 4.2254% / 5.6604% | 94.3262% | 92.8571% | 100.0000% | 46.4789% / 62.2642% |
| validation | 4 | 1.4085% / 1.8868% | 96.4539% | 96.4286% | 100.0000% | 4.2254% / 5.6604% |
| validation | 6 | 1.4085% / 1.8868% | 97.8723% | 98.2143% | 100.0000% | 1.4085% / 1.8868% |
| validation | 8 | 1.4085% / 1.8868% | 99.2908% | 100.0000% | 100.0000% | 1.4085% / 1.8868% |
| validation | Unlimited | 0.0000% / 0.0000% | 100.0000% | 100.0000% | 100.0000% | N/A |

## Slot occupancy elbow

| Dataset | Capacity | Slot active fractions (all performances) |
| --- | ---: | --- |
| train | 3 | S1=67.8952%, S2=56.3046%, S3=32.3472% |
| train | 4 | S1=67.8952%, S2=56.3046%, S3=32.3472%, S4=2.8613% |
| train | 6 | S1=67.8952%, S2=56.3046%, S3=32.3472%, S4=2.8613%, S5=0.9699%, S6=0.2425% |
| validation | 3 | S1=74.6479%, S2=66.1972%, S3=46.4789% |
| validation | 4 | S1=74.6479%, S2=66.1972%, S3=46.4789%, S4=4.2254% |
| validation | 6 | S1=74.6479%, S2=66.1972%, S3=46.4789%, S4=4.2254%, S5=1.4085%, S6=1.4085% |

## Timing representation

| Dataset | Coordinate | Transform | Median | p95 | p99 | Max | Dynamic range |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| train | absolute | raw | 0.559896 | 3.902766 | 7.701510 | 26.704085 | 26.703044 |
| train | absolute | log1p | 0.444619 | 1.589800 | 2.163488 | 3.321580 | 3.320539 |
| train | absolute | bounded | 0.358932 | 0.796034 | 0.885076 | 0.963904 | 0.962864 |
| train | gap | raw | 0.177083 | 2.412760 | 5.408755 | 23.146390 | 23.145348 |
| train | gap | log1p | 0.163040 | 1.227521 | 1.857653 | 3.184135 | 3.183094 |
| train | gap | bounded | 0.150442 | 0.706982 | 0.843960 | 0.958586 | 0.957545 |
| validation | absolute | raw | 0.613281 | 3.904948 | 5.101908 | 7.136759 | 7.134622 |
| validation | absolute | log1p | 0.478270 | 1.590244 | 1.808337 | 2.096392 | 2.094257 |
| validation | absolute | bounded | 0.380145 | 0.796124 | 0.836030 | 0.877101 | 0.874969 |
| validation | gap | raw | 0.198985 | 3.086458 | 4.505205 | 5.402783 | 5.400646 |
| validation | gap | log1p | 0.181475 | 1.407679 | 1.701077 | 1.856733 | 1.854598 |
| validation | gap | bounded | 0.165961 | 0.755289 | 0.816646 | 0.843818 | 0.841686 |

## Transform conditioning (diagnostic only)

These numbers do not predict model accuracy; they only invert fixed transformed-space perturbations.

| Source delay | epsilon | Raw seconds error | Log1p seconds error | Bounded seconds error |
| ---: | ---: | ---: | ---: | ---: |
| 0.10 | +0.01 | 0.010000 | 0.011055 | 0.012235 |
| 0.10 | +0.05 | 0.050000 | 0.056398 | 0.064021 |
| 1.00 | +0.01 | 0.010000 | 0.020100 | 0.040816 |
| 1.00 | +0.05 | 0.050000 | 0.102542 | 0.222222 |
| 5.00 | +0.01 | 0.010000 | 0.060301 | 0.382979 |
| 5.00 | +0.05 | 0.050000 | 0.307627 | 2.571429 |
| 10.00 | +0.01 | 0.010000 | 0.110552 | 1.359551 |
| 10.00 | +0.05 | 0.050000 | 0.563982 | 13.444444 |

## Research questions

### Q1 — Count tail
Affected train performances have mean/median/p95/p99/max 2.372/2/3/5/9. The tail reaches nine events but is sparse.
### Q2 — Retention curve
K_T=3/4/6 train event retention is 97.1996%/98.9762%/99.7290%; crossing retention is 97.8528%/99.3098%/99.7699%.
### Q3 — Minimal K_T
Recommend K_T=4. It retains 98.9762% events and 99.3098% crossings; only 1.4286% of affected train performances overflow. K_T=6 gains 0.753pp event and 0.460pp crossing retention, but Slots5/6 are active in only 0.970%/0.242% of all train performances. Validation has one nine-event outlier, so its K_T=4 event retention is 96.454%.
### Q4 — Coordinate
Recommend inter-event gaps. Train median/p95/p99 are 0.177/2.413/5.409s versus absolute 0.560/3.903/7.702s, and cumulative decoding guarantees chronological order. Its trade-off is cumulative timing error across at most four terminal slots.
### Q5 — Transform
Recommend log1p. For train gaps it maps max/p99 23.146/5.409s to 3.184/1.858, while retaining a near-linear short-delay region.
### Q6 — Bounded saturation
Frequency saturation is not widespread: train z>=0.90 is 0.5420% absolute and 0.2108% gap; z>=0.98 is zero. Nevertheless inverse sensitivity near one is poor.
### Q7 — Log1p short delays
At source 0.1s, a +0.05 transformed perturbation becomes 0.056s error, close to raw's 0.050s, while long tails are compressed.
### Q8 — Inverse sensitivity
At 5s/10s with +0.05 perturbation, log1p gives 0.308/0.564s error; bounded gives 2.571/13.444s. Raw is best-conditioned in seconds but leaves the heavy tail uncompressed.
### Q9 — Recommendation
K_T=4, chronological last-4, inter-event-gap timing, log1p transform.
### Q10 — Readiness
READY. This is a proposed spec for user approval; it has not been implemented.

## Recommended terminal representation

```text
Terminal anchor:
latest note-off

Terminal capacity:
K_T = 4

Capacity compression:
chronological last-4

Timing coordinate:
inter-event-gap

Timing transform:
log1p

Event vocabulary:
NONE / SET_ZERO / SET_LOW / SET_HALF / SET_FULL

Inference ordering implication:
decode positive gaps, invert log1p, and cumulatively sum from latest note-off; ordering is structural

Reason:
K_T=4 is the capacity elbow; gaps compact targets and guarantee order; log1p balances tail compression and inverse conditioning.
```

## Custom Event Tokenizer Proposed Final Spec

```text
Initial state: state immediately before first distinct onset
Main interval: distinct-onset [t_i,t_{i+1})
Main slots: K = 6
Main compression: chronological last-6
Main timing: exact tau in [0,1)
Event vocabulary: NONE / SET_ZERO / SET_LOW / SET_HALF / SET_FULL
Terminal anchor: latest note-off
Terminal slots: K_T = 4
Terminal compression: chronological last-4
Terminal timing coordinate: inter-event-gap
Terminal timing transform: log1p
Same-timestamp semantics: keep v0 deterministic semantics; document ~2.3617% onset events and 193 ambiguous cases
Model implementation readiness: YES, after user freezes this proposed spec
```

Read-only main K=6 reference: `{'crossing_retention': 0.99663139, 'event_retention': 0.99614549, 'intersection': 0.9997288, 'js_divergence': 3.691e-05, 'macro_f1': 0.99984563, 'token_accuracy': 0.99986352, 'transition_f1': 0.99790949, 'transition_recall': 0.9958277}`.

No tokenizer final-spec implementation, model, loss, training, ASAP test, Repedal, or full oracle rerun was performed.
