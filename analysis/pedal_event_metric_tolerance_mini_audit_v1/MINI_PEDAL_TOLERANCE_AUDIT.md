# Mini Pedal Event Tolerance Audit v1

## Scope and selection

Only the canonical ASAP train manifest and train MIDI were used. ASAP validation/test and MAESTRO were not opened. No model, inference, tokenizer, or production evaluator was used.

Selected piece: **Chopin — Berceuse_op_57** (`piece_de97ef37254a11f8`). Deterministic rule: composers restricted to Chopin/Schumann; 5–8 train performances; at least 5 with raw CC64; median CC64 ≥100; median normalized notes 500–2500 and maximum ≤3500. Sort by distance from 6 performances, then descending median CC64, then piece ID. This selected 5 performances without looking at alignment or transition outcomes.

Successful alignments: 5/5; failures: 0. Every success was atomically cached immediately; reruns skip valid caches. Progress and external-tool logs are retained.

Distinct onsets are unique final-performance note-on timestamps. Local IOI uses positive differences only and its median. CC64 UP/DOWN uses the fixed threshold 64.

## A. Transition findings

Primary ±8 matching produced 2453 pairs: UP 1226, DOWN 1227.

| fixed K | all coverage | UP | DOWN |
| ---: | ---: | ---: | ---: |
| 0 | 0.790 | 0.869 | 0.711 |
| 1 | 0.963 | 0.965 | 0.962 |
| 2 | 0.975 | 0.977 | 0.973 |
| 3 | 0.982 | 0.984 | 0.980 |
| 4 | 0.988 | 0.989 | 0.986 |

Search sensitivity: +/-6: 2443 pairs, +/-8: 2453 pairs, +/-12: 2469 pairs.

| W | valid pairs | IOI p25/median/p75 ms | median relative difference vs W=8 | Spearman(IOI, delta onset) |
| ---: | ---: | --- | ---: | ---: |
| 4 | 2453 | 68.490/115.690/218.815 | 0.172 | -0.047 |
| 8 | 2453 | 63.477/116.211/249.284 | 0.000 | -0.025 |
| 12 | 2453 | 64.518/117.839/229.948 | 0.082 | -0.024 |
| 16 | 2453 | 52.734/108.398/201.497 | 0.122 | 0.006 |

Adaptive sanity-check shortlist (not final parameters):

| W | T ms | coverage | mean/median K | K=0/1/2/3/4 | UP/DOWN | trade-off |
| ---: | ---: | ---: | --- | --- | --- | --- |
| 8 | 200 | 0.939 | 1.598/1.000 | 0.29/0.29/0.16/0.05/0.21 | 0.968/0.910 | more local |
| 8 | 300 | 0.971 | 2.212/2.000 | 0.13/0.24/0.21/0.13/0.29 | 0.981/0.961 | more local |
| 12 | 300 | 0.970 | 2.232/2.000 | 0.10/0.28/0.20/0.13/0.29 | 0.980/0.959 | smoother local IOI |

W=8 and W=12 give nearly the same median IOI and transition/IOI relationship. W=4 is the most event-local and therefore more exposed to individual short IOIs; W=16 shifts the distribution most and is the most likely to smooth local tempo change. W choice does not materially change the transition conclusion in this pilot.

At W=8, the fast-region hypothesis is **weak or absent** (negative correlation means shorter IOI tends to larger onset error). Fixed K=1 reaches 0.963 coverage with exactly one onset allowed, whereas W=8/T=200 reaches only 0.939 at mean K=1.598 and W=8/T=300 reaches 0.971 at mean K=2.212. Adaptive tolerance therefore does not improve the transition coverage/width trade-off here.

## B. Repedal findings

Analyzed performances: 5; strong candidates: 689.

| span | median | p75 | p90 | p95 | <=1/2/3/4 |
| --- | ---: | ---: | ---: | ---: | --- |
| release | 9.000 | 15.000 | 22.000 | 27.000 | 0.02/0.04/0.10/0.15 |
| repress | 1.000 | 2.000 | 6.000 | 11.600 | 0.61/0.78/0.84/0.88 |
| total | 11.000 | 18.000 | 27.000 | 36.000 | 0.01/0.03/0.04/0.09 |

IOI versus total repedal span Spearman correlations: W=4: -0.303, W=8: -0.343, W=12: -0.352, W=16: -0.286. Negative values support a larger onset budget in faster passages.

Extrema rule: collapse consecutive equal CC64 values; identify strict local peaks/troughs; for each trough choose the nearest prior and following local peaks satisfying both ≥64 excursions; deduplicate exact `(pre-peak,trough,post-peak)` triples. This minimizes gratuitously long nested gestures.

## C. Research conclusions

1. This pilot favors fixed +/-1 as the primary Transition F1 tolerance (coverage 0.963), with fixed +/-2 (coverage 0.975) as a conservative sensitivity check. The tested adaptive candidates are less efficient.
2. Maximum 4-onset tolerance covers 0.988, only 0.013 above +/-2, so +/-3 to 4 appears unnecessarily broad for transition matching in this piece.
3. Local-IOI evidence for transition adaptivity is weak or absent; this pilot does not justify retaining adaptive Transition F1 tolerance.
4. Repedal scale depends on the boundary: trough-to-repress is typically 1-2 onsets, but peak-to-trough release and total peak-to-peak spans are much longer (medians 9 and 11). Thus a fixed 1-4 window describes the re-press action but not the full peak-to-peak candidate under this definition. The modest negative IOI/span correlations support further adaptive-window investigation, not a cutoff.
5. One lyrical Chopin piece cannot establish a general heuristic. Before production implementation, check one contrasting, rhythmically dense ASAP-train piece in a separate task.

No final W, T, transition tolerance, or repedal cutoff is selected by this report.

## Artifacts

- `transition_pairs.csv`
- `adaptive_tolerance_scan.csv`
- `repedal_candidates.csv`
- `repedal_span_summary.csv`
- `progress.log`, `alignment_cache/`, `alignment_logs/`
- two compact W=8 diagnostic plots
