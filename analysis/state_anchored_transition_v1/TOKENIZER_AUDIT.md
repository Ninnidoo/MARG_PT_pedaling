# State-Anchored Transition v1 Tokenizer Audit

**Verdict: PASS**

## Target definition and implementation

The modeled horizon is MAIN-only: for distinct onsets `t_1 ... t_M`, only `[t_i, t_(i+1))` for `i=1 ... M-1` is represented. There is no PRE, POST, or final-note-off tail. State `S_i` is the threshold-64 pedal state immediately before `t_i`; an event exactly at `t_i` belongs to interval `i` with `tau=0`.

The implementation reuses the frozen canonical cache manifest/onset metadata, the existing raw MIDI parser, same-tick-last CC64 projection, `<64/OFF` and `>=64/ON` crossing extraction, and right-open Binary 2-Slot timeline builder. Raw counts 0/1/2 map to HOLD/CHANGE/RETURN. Odd overflow retains the final one transition and even overflow retains the final two; no correction or synthetic event is inserted.

## Canonical population

- Train: 2,062 performances (MAESTRO-clean 1,170 + ASAP train 892)
- Validation: 71 performances (ASAP validation 71)
- ASAP test access: 0

## State distribution

| Split | OFF | ON | Total onset states |
| --- | ---: | ---: | ---: |
| Train | 3,568,800 (39.611306%) | 5,440,749 (60.388694%) | 9,009,549 |
| Validation | 101,263 (37.102595%) | 171,664 (62.897405%) | 272,927 |

## Mode distribution

| Split | HOLD | CHANGE | RETURN | Total intervals |
| --- | ---: | ---: | ---: | ---: |
| Train | 7,882,290 (87.508203%) | 894,618 (9.931938%) | 230,579 (2.559859%) | 9,007,487 |
| Validation | 244,055 (89.444615%) | 22,792 (8.353124%) | 6,009 (2.202261%) | 272,856 |

## RETURN composition

| Split | ON -> OFF -> ON | OFF -> ON -> OFF |
| --- | ---: | ---: |
| Train | 195,595 (84.827760%) | 34,984 (15.172240%) |
| Validation | 5,087 (84.656349%) | 922 (15.343651%) |

## Compression

| Split | Raw transitions | Retained | Retained ratio | K>=3 intervals | Odd overflow | Even overflow | Endpoint preservation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 1,375,468 | 1,355,776 | 98.568342% | 9,729 (0.108010%) | 8,125 | 1,604 | 100.000000% |
| Validation | 35,346 | 34,810 | 98.483562% | 266 (0.097487%) | 209 | 57 | 100.000000% |

## Consistency invariants

| Split | CHANGE iff flip violations | HOLD/RETURN iff same violations | Timing range violations | RETURN order violations |
| --- | ---: | ---: | ---: | ---: |
| Train | 0 | 0 | 0 | 0 |
| Validation | 0 | 0 | 0 | 0 |

## Timing distribution

### Train

- CHANGE tau1: n=894,618, mean=0.459214, median=0.429487, p05=0.025641, p95=0.945652
- RETURN tau1: n=230,579, mean=0.164471, median=0.118644, p05=0.009311, p95=0.479747
- RETURN tau2: n=230,579, mean=0.564299, median=0.544757, p05=0.189474, p95=0.965614
- RETURN tau2 - tau1: n=230,579, mean=0.399829, median=0.356932, p05=0.108696, p95=0.812500

### Validation

- CHANGE tau1: n=22,792, mean=0.471808, median=0.457537, p05=0.024119, p95=0.947368
- RETURN tau1: n=6,009, mean=0.189356, median=0.141304, p05=0.010390, p95=0.553709
- RETURN tau2: n=6,009, mean=0.584179, median=0.586462, p05=0.193323, p95=0.965170
- RETURN tau2 - tau1: n=6,009, mean=0.394822, median=0.355212, p05=0.105311, p95=0.798407

## Edge-case tests

Focused command: `/opt/conda/bin/python tests/test_state_anchored_transition_v1.py`

Result: PASS (9 synthetic cases). Covered HOLD, both CHANGE directions, both RETURN compositions, 3/4-transition compression, left/right boundary ownership, and strict-before first-onset state.

## PASS / FAIL

PASS: focused tests passed; both split invariants and timing checks have zero violations; compression preserves every raw endpoint state; canonical provenance reports zero ASAP test access. No model, decoder, overfit, or training was run.
