# Custom Event Tokenizer Capacity & Terminal Audit v1

- Core v0 representation: unchanged; original v0 artifacts: read-only.
- Train: 1,170 MAESTRO-clean + 892 ASAP train; validation: 71 ASAP performances.
- Training/model/optimizer/checkpoint activity: 0. ASAP test access: 0. Repedal: 0.
- Fixed-K rule: retain all if n<=K, otherwise retain chronological final K.

## Capacity structural curve

| Capacity | Overflow % | Event retention % | Crossing retention % | Last-slot active % |
| --- | ---: | ---: | ---: | ---: |
| K=1 last-1 | 9.052272 | 53.221173 | 48.463319 | 19.100066 |
| K=2 odd/even | 4.247926 | 70.488990 | 69.024521 | 6.197091 |
| K=2 last-2 | 4.247926 | 78.444780 | 83.303688 | 9.052272 |
| K=3 last-3 | 1.965870 | 90.281367 | 90.838745 | 4.247926 |
| K=4 last-4 | 0.953655 | 95.759143 | 95.355131 | 1.965870 |
| K=6 last-6 | 0.071757 | 99.614549 | 99.663139 | 0.429977 |
| K=8 last-8 | 0.016860 | 99.908578 | 99.931001 | 0.033764 |
| Unlimited | 0.000000 | 100.000000 | 100.000000 | N/A |

## Validation oracle curve

| Capacity | 4C Acc | Macro F1 | Trans P | Trans R | Trans F1 | JS | Intersection |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| K=1 last-1 | 0.973102 | 0.960117 | 1.000000 | 0.643550 | 0.783122 | 0.026342 | 0.944014 |
| K=2 odd/even | 0.986777 | 0.982207 | 1.000000 | 0.741599 | 0.851630 | 0.010113 | 0.972447 |
| K=2 last-2 | 0.990491 | 0.987211 | 1.000000 | 0.799617 | 0.888652 | 0.008631 | 0.977160 |
| K=3 last-3 | 0.996894 | 0.996240 | 0.999942 | 0.976658 | 0.988163 | 0.002516 | 0.991614 |
| K=4 last-4 | 0.999232 | 0.999022 | 1.000000 | 0.989344 | 0.994643 | 0.000438 | 0.997802 |
| K=6 last-6 | 0.999864 | 0.999846 | 1.000000 | 0.995828 | 0.997909 | 0.000037 | 0.999729 |
| K=8 last-8 | 0.999911 | 0.999911 | 1.000000 | 0.997519 | 0.998758 | 0.000005 | 0.999845 |
| Unlimited | 0.999915 | 0.999916 | 1.000000 | 0.997970 | 0.998984 | 0.000001 | 0.999863 |

## Unlimited penalty

| Capacity | 4C Acc Δ | Macro F1 Δ | Trans F1 Δ | Trans Recall Δ | JS Δ | Intersection Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| K=1 last-1 | -0.026813 | -0.039799 | -0.215862 | -0.354420 | +0.026341 | -0.055849 |
| K=2 odd/even | -0.013139 | -0.017708 | -0.147354 | -0.256371 | +0.010112 | -0.027415 |
| K=2 last-2 | -0.009424 | -0.012704 | -0.110332 | -0.198354 | +0.008629 | -0.022703 |
| K=3 last-3 | -0.003021 | -0.003676 | -0.010821 | -0.021313 | +0.002515 | -0.008249 |
| K=4 last-4 | -0.000683 | -0.000894 | -0.004341 | -0.008627 | +0.000437 | -0.002060 |
| K=6 last-6 | -0.000052 | -0.000070 | -0.001075 | -0.002143 | +0.000036 | -0.000134 |
| K=8 last-8 | -0.000004 | -0.000005 | -0.000226 | -0.000451 | +0.000004 | -0.000018 |
| Unlimited | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |

## Pareto-like capacity view

| K | Trans F1 | Trans Recall | 4C Acc | JS | Intersection | Event retention | Crossing retention | Overflow % | Last-slot active % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.783122 | 0.643550 | 0.973102 | 0.026342 | 0.944014 | 53.2212% | 48.4633% | 9.0523% | 19.1001% |
| 2 | 0.888652 | 0.799617 | 0.990491 | 0.008631 | 0.977160 | 78.4448% | 83.3037% | 4.2479% | 9.0523% |
| 3 | 0.988163 | 0.976658 | 0.996894 | 0.002516 | 0.991614 | 90.2814% | 90.8387% | 1.9659% | 4.2479% |
| 4 | 0.994643 | 0.989344 | 0.999232 | 0.000438 | 0.997802 | 95.7591% | 95.3551% | 0.9537% | 1.9659% |
| 6 | 0.997909 | 0.995828 | 0.999864 | 0.000037 | 0.999729 | 99.6145% | 99.6631% | 0.0718% | 0.4300% |
| 8 | 0.998758 | 0.997519 | 0.999911 | 0.000005 | 0.999845 | 99.9086% | 99.9310% | 0.0169% | 0.0338% |

## Event/crossing tail

- Event-count quantiles (nearest-rank): `{'max': 53, 'p90': 1, 'p95': 2, 'p97p5': 3, 'p99': 4, 'p99p5': 5, 'p99p9': 6}`.
- Crossing-count quantiles: `{'max': 20, 'p90': 1, 'p95': 1, 'p97p5': 2, 'p99': 2, 'p99p5': 2, 'p99p9': 3}`.
- Fractions crossing-count >2/>3/>4: 0.108529% / 0.018480% / 0.000988%.

## K2 odd/even versus generic last-2

- Event retention: 70.488990% → 78.444780%.
- Structural crossing retention: 69.024521% → 83.303688%.
- 4C accuracy: 0.986777 → 0.990491.
- Transition P/R/F1: 1.000000/0.741599/0.851630 → 1.000000/0.799617/0.888652.
- JS/Intersection: 0.010113/0.972447 → 0.008631/0.977160.
- Final-state preservation: 100% for both.

## Terminal delay

- Train: `{'affected_performances': 1400, 'count': 3321, 'max': 26.70408519791667, 'mean': 1.1177089102585813, 'min': 0.0010416666666515084, 'p25': 0.18880208333331439, 'p50': 0.5598958333332575, 'p75': 1.382474554166663, 'p90': 2.72265625, 'p95': 3.9027662645833345, 'p97p5': 5.484770156250022, 'p99': 7.701509583333397, 'p99p5': 9.26525780875002, 'std': 1.6741462632794684, 'total_performances': 2062}`.
- Validation: `{'affected_performances': 53, 'count': 141, 'max': 7.136758916666679, 'mean': 1.2313847468232928, 'min': 0.0021367458333259037, 'p25': 0.3151706250000075, 'p50': 0.61328125, 'p75': 1.7240901562499857, 'p90': 3.477560624999967, 'p95': 3.9049479166666856, 'p97p5': 4.402124557291643, 'p99': 5.101907749999996, 'p99p5': 5.792329758333346, 'std': 1.35596382893568, 'total_performances': 71}`.

## Fixed-tail coverage (affected performances denominator)

| Tail | Train event | Train full perf | Validation event | Validation full perf |
| ---: | ---: | ---: | ---: | ---: |
| 0.05s | 8.0699% | 6.7143% | 4.2553% | 3.7736% |
| 0.10s | 15.8988% | 13.7143% | 9.2199% | 5.6604% |
| 0.25s | 30.1716% | 26.2857% | 21.2766% | 16.9811% |
| 0.50s | 46.8232% | 40.5714% | 39.7163% | 30.1887% |
| 0.75s | 59.6507% | 53.0000% | 58.1560% | 50.9434% |
| 1.00s | 67.5700% | 61.9286% | 63.1206% | 56.6038% |
| 1.50s | 76.8744% | 71.8571% | 71.6312% | 66.0377% |
| 2.00s | 82.5354% | 78.0000% | 78.7234% | 73.5849% |
| 3.00s | 91.3279% | 87.9286% | 85.8156% | 83.0189% |
| 5.00s | 96.8985% | 95.5000% | 98.5816% | 96.2264% |

## Data-based recommendations

- Minimal sufficient capacity recommendation: **K=6 with generic last-6**.
- Terminal recommendation: **Dedicated terminal event slot(s) anchored to latest note-off; preserve event class and predicted delay. A fixed 5.0 s tail is only a fallback candidate, not sufficient for full train coverage.**.
- Same-timestamp semantics: KEEP for now, with the existing targeted limitation (2.3617% effective events on onset; 193 ambiguous cross-track/interleaved cases).

## Final proposed configuration

```text
Core representation: KEEP
Fixed capacity: K = 6
Compression: generic chronological last-6
Initial state: KEEP
Terminal: Dedicated terminal event slot(s) anchored to latest note-off; preserve event class and predicted delay. A fixed 5.0 s tail is only a fallback candidate, not sufficient for full train coverage.
Same-timestamp semantics: KEEP / targeted limitation documented
Ready for model implementation: YES, after user approves this proposed spec
```

No tokenizer spec implementation, model, loss, tiny overfit, training, ASAP test, or Repedal execution was started.
