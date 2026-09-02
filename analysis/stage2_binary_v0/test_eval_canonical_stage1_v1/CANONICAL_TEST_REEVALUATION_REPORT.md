# Canonical Fixed-Stage-1 Binary Stage 2 Re-evaluation

## Result

| Model | canonical Stage-1 strict JS ↓ | Intersection ↑ |
|---|---:|---:|
| Original PT canonical CPU MIDI | 0.146432600425 | 0.847466876476 |
| independent 4×2 locked | 0.185597746584 | 0.788894290429 |
| joint 16 locked | 0.260384376633 | 0.714874043656 |

| Model | JS change vs canonical Original PT | relative JS change | Intersection change |
|---|---:|---:|---:|
| independent 4×2 | +0.039165146159 | +26.746193% | -0.058572586047 |
| joint 16 | +0.113951776208 | +77.818584% | -0.132592832820 |

Under this revised canonical Stage-1 protocol, neither locked Stage 2 checkpoint improves the canonical Original PT global pedal metric. No model/checkpoint was reselected in response.

## Corrected protocol

- Stage 1 is the immutable 23-piece CPU/seed-42 bank `outputs/midi/{num}_original_pt.mid`; PT neural inference was not rerun.
- Each canonical MIDI was tokenized with official `midi_to_ids`; Pedal1--4 were masked by the existing Stage 2 helper.
- The unchanged final-lock-v1 independent epoch 2 and joint epoch 3 checkpoints were used with 512-note windows, stride 256, overlap-logit averaging, and deterministic argmax.
- `ids_to_midi` + `map_midi` produced temporary pedal donors only. Final candidates copied every canonical raw event and transplanted donor CC64 only.
- Donor CC64 strictly after canonical end-of-track was discarded (never shifted); canonical EOT remained exact and discard counts are recorded per piece.
- Strict raw MIDI non-CC64 equality: **46/46 PASS**. Canonical source hashes remained unchanged.
- Human MIDI was not reopened; the prior official 104-performance Human histogram was reused byte-for-byte from `analysis/stage2_binary_v0/test_eval_v0/final_test_metrics.json` (`36db39160dee3924139b97f0198587e15965c0f50184b6a5d925b6f1c970e8f3`).

## Prior GPU Stage-1 result (not directly comparable as the same realization)

| Model | prior GPU Stage-1 JS ↓ | Intersection ↑ |
|---|---:|---:|
| Original PT | 0.181901863968 | 0.878312399553 |
| independent 4×2 | 0.133316954870 | 0.891138750058 |
| joint 16 | 0.203767205373 | 0.850856482122 |

## Scientific status

No training, checkpoint selection, threshold/decoding change, calibration, Stage 1 inference, or Human-MIDI re-evaluation occurred. This is a revised canonical Stage-1 protocol adopted after inspecting the earlier test realization, so it is reported separately from the original locked final-test artifact and does not overwrite it.
