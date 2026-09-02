# Frozen-PT Mapping Fix Report

| item | result |
|---|---|
| Root cause | `metadata_index` exists in the canonical split row, not in `human_dataset.performances[*].entry` |
| Canonical join | exact `performance_path` join, then verify `piece_id`; split row supplies `metadata_index` |
| Mapping | PASS: 19 frozen pieces / 71 human performances |
| Missing / ambiguous / duplicate / wrong-piece | 0 / 0 / 0 / 0 |
| Representative preparation smoke | PASS (`piece_08aa5ff2ada461e0`) |
| Pedal leakage / non-pedal identity | 0 / PASS |
| Epoch-1 weights reusable | NO |
| Exact optimizer/scaler/RNG resume | NO |
| Epoch-1 validation-only rerun | NO |
| Epoch-1 training rerun required | YES |
| ASAP test access | 0 |

## Failure provenance

Epoch 1 completed all 516 optimizer groups and Human-input validation reached 71/71. The exception occurred while constructing `human_by_metadata`, before the first Frozen-PT piece rollout. It was a schema `KeyError`, not NaN/Inf, CUDA, model-forward, or metric failure.

## Q1–Q3: cause and mapping

The dataset cache schema contains `cache_file, cache_file_bytes, main_intervals, notes, onsets, performance_index, performance_path, piece_id, reused, source, source_midi, source_sha256, split` and has no `metadata_index` in all 71 entries. The canonical split contains `metadata_index`, `performance_path`, and `piece_id`. Existing 4-class evaluation groups split rows by `piece_id`, loads each human by split `metadata_index`, and identifies the performance by split `performance_path`. The fix performs the same path-keyed join and verifies the piece on both sides. All 71 paths map exactly once to the same 19-piece universe.

Existing frozen human aligned caches are available for 70/71; metadata `856` is the same known canonical alignment exclusion. This does not affect the 71/71 structural source-reference mapping.

## Q4–Q5: focused smoke and leakage

The representative piece attached 1 correct human references, constructed PT inputs and PRE/POST queries, and rendered a CC64-replaced MIDI without model inference. All PT pedal columns and both boundary query pseudo-notes were MASK, attention remained active, and pitch/onset/note-off/velocity identity passed.

## Q6–Q7: epoch 1 and next start

No `last.pt`, `best.pt`, epoch checkpoint, model state, optimizer state, scaler state, or RNG state exists. The driver previously saved only after both validations, so the in-memory epoch-1 result was lost when mapping failed. Epoch-1 weights and validation are not reusable; exact resume is impossible and epoch 1 must be retrained after explicit approval.

The driver now atomically saves `train_complete_epoch_XX.pt` plus `resume_last_train_state.pt` immediately after training and before validation, including model/optimizer/scaler/global-step and Python/NumPy/Torch CPU/CUDA RNG state. Best selection remains exclusively frozen-PT Transition F1 after complete validation. A focused save/load round-trip passed. No training was launched in this task.
