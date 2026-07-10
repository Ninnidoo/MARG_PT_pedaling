# Troubleshooting

## Issues

## Fixes Tested

## Open Problems
## Colab GPU Inference Troubleshooting

### GPU Not Assigned

Symptoms:

- `torch.cuda.is_available()` is `False`.
- `nvidia-smi` is missing or reports no GPU.

Response:

1. In Colab, choose `Runtime > Change runtime type > GPU`.
2. Reconnect the runtime.
3. Rerun the runtime check cell.
4. If free Colab has no GPU available, try later or use a paid runtime. Do not continue this GPU-first notebook on CPU.

### CUDA Error

Symptoms:

- CUDA initialization error.
- Device mismatch error.
- CUDA out-of-memory error.

Response:

1. Restart the Colab runtime.
2. Run only this notebook to reduce memory pressure.
3. Confirm the model parameter device is `cuda` before inference.
4. Use one small example MIDI only.
5. If OOM persists, retry with the smallest bundled score MIDI and `float16`.

### dtype Error

Symptoms:

- Errors mentioning `bfloat16`, unsupported dtype, or unsupported CUDA kernel.

Response:

1. The notebook selects `torch.bfloat16` only when CUDA reports bf16 support; otherwise it uses `torch.float16`.
2. If dtype errors persist, document the GPU name, PyTorch version, CUDA version, selected dtype, and traceback.
3. TODO: test official PyTorch 2.7.1 for the active Colab CUDA runtime if the preinstalled PyTorch is incompatible.

### Checkpoint Missing After Runtime Restart

Symptoms:

- `models/sft/config.json`, `models/sft/generation_config.json`, or `models/sft/model.safetensors` is missing.

Response:

1. Rerun the checkpoint download cell.
2. If using Google Drive persistence, copy the saved `models/sft/` directory back into `/content/PianistTransformer/models/sft/`.
3. Verify all required checkpoint files exist and have nonzero size before loading the model.

### Output MIDI Missing or Empty

Symptoms:

- `/content/output_midi/0.mid` does not exist.
- Output file size is zero.
- MIDI contains no notes.

Response:

1. Check the full traceback from the inference cell.
2. Confirm the selected input MIDI exists and has nonzero size.
3. Confirm the model is on CUDA and the checkpoint files are present.
4. Rerun the inference cell after clearing stale output files if needed.
