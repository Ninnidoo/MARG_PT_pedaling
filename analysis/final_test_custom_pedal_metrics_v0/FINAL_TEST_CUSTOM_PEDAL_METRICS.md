# Final Test Custom Pedal Metrics

## Overall final test

| System | Harmonic Muddiness ↓ | Bass Connectivity ↑ |
|---|---:|---:|
| Human — Set A (104) | 0.302510 | 0.278505 |
| Human — Set B (166) | 0.301763 | 0.284554 |
| Original PT | 0.272566 | 0.170338 |
| RUN A | 0.345248 | 0.188973 |
| RUN B | 0.236427 | 0.180841 |
| Weighted CE | 0.229612 | 0.143450 |
| CE + NTL-WAS | 0.233811 | 0.154362 |
| Huber + Auxiliary CE | 0.249446 | 0.136929 |

## Accounting

- Human: performance score → within-piece arithmetic mean → 23-piece macro; Set A and Set B were not pooled.
- Generated: each saved MIDI measured once; 6 × 23 = 138. No Human alignment was used.
- RUN A note provenance: 13 strict + 10 frozen EOT-extension-only; pitch/onset/offset/velocity PASS 23/23.
- Undefined Bass performance/piece counts are recorded in CSV.

## Original PT improvement counts

| System | M_harm improved | Bass improved |
|---|---:|---:|
| RUN A | 5/23 | 13/23 |
| RUN B | 18/23 | 15/23 |
| Weighted CE | 18/23 | 7/23 |
| CE + NTL-WAS | 17/23 | 9/23 |
| Huber + Auxiliary CE | 18/23 | 4/23 |

## Metric provenance

Frozen definitions are recorded in metric_provenance.json. No weighted sum was created.
