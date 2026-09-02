# Custom Event Tokenizer v1 Cache Report

- Cache ID: `3a5520155b5db1e9a1da7f8148556aa3e1da852655c9adde25d3dbbd1d966263`
- Train: 2062 performances; validation: 71 performances.
- Cache level: full performance/piece; window ownership not implemented.
- Reused valid cache files during this invocation: 0.
- ASAP test access: 0. Model/head/loss/training: 0.
- Complete-note onset diagnostic: train excluded 1 unmatched-note-only onset; validation excluded 0.

## Structural regression

| Split | Main event retention | Main crossing retention | Terminal event retention | Terminal crossing retention |
| --- | ---: | ---: | ---: | ---: |
| Train | 0.996145491 | 0.996631395 | 0.989762120 | 0.993098160 |
| Validation | 0.995796295 | 0.996183853 | 0.964539007 | 0.964285714 |

- Regression checks: `{'main_train_events_exact': True, 'main_train_crossings_exact': True, 'terminal_train_exact': True, 'terminal_validation_exact': True}`.
- Main interval final-state preservation: train/validation 100%.
- Terminal final-state preservation: train/validation 100%.
- Timing masks are explicit; NONE timing placeholders are finite zero and invalid by mask.
- Candidate inverse-sqrt frequency weights are diagnostic only; no loss was implemented.
- Full strict verification: `cache_verification.json`; real-MIDI regression: `small_real_data_regression.json`.
