# Troubleshooting

## Colab GPU Inference Troubleshooting

### Confirmed Not a Problem in First Smoke Test

The following items did not block the 2026-07-10 official example smoke test:

- Google Colab GPU runtime was available.
- CUDA was usable via `device=cuda`.
- `torch.bfloat16` worked for the assigned Colab GPU run.
- The official checkpoint `yhj137/pianist-transformer-rendering` loaded successfully.
- Inference completed and produced `/content/output_midi/0.mid`.
- Output MIDI was non-empty: 6661 bytes, 1 instrument, 524 notes.
- MuseScore played the generated MIDI normally.

### Still Open or Watch Items

- `/content` is temporary. Outputs and checkpoints must be downloaded or saved to Google Drive before the Colab runtime is deleted.
- Free Colab GPU availability can vary. If no GPU is assigned, the notebook should not be treated as a GPU validation run.
- Larger or denser MIDI files may hit CUDA memory limits or take longer than the first smoke test.
- The first smoke-test input `score/3.mid` produced only 6 CC64 sustain-pedal events, so it is not suitable for evaluating repedaling behavior.
- The seed was not recorded in the provided first smoke-test result. Future experiments should record seed explicitly.
- Local Windows CPU inference remains unverified.

### GPU Not Assigned

Symptoms:

- `torch.cuda.is_available()` is `False`.
- `nvidia-smi` is missing or reports no GPU.

Response:

1. In Colab, choose `Runtime > Change runtime type > GPU`.
2. Reconnect the runtime.
3. Rerun the runtime check cell.
4. If free Colab has no GPU available, try later or use a paid runtime. Do not record the run as GPU inference.

### CUDA Error

Symptoms:

- CUDA initialization error.
- Device mismatch error.
- CUDA out-of-memory error.

Response:

1. Restart the Colab runtime.
2. Run only the inference notebook to reduce memory pressure.
3. Confirm the model parameter device is `cuda` before inference.
4. Use one small example MIDI first.
5. If OOM persists, retry with a shorter MIDI or a lower-memory dtype supported by the runtime.

### dtype Error

Symptoms:

- Errors mentioning `bfloat16`, unsupported dtype, or unsupported CUDA kernel.

Response:

1. Record the GPU name, PyTorch version, CUDA version, selected dtype, and traceback.
2. If `torch.bfloat16` fails on a future runtime, retry with `torch.float16` if the model path supports it.
3. If dtype errors persist, test the official PyTorch 2.7.1 install command for the active Colab CUDA runtime and document whether a runtime restart is required.

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