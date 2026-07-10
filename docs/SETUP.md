# Setup

## Verified Project Workflow

Date: 2026-07-10

### Local Repository

- Local project root: `C:\Users\eagle\MARG_research`
- VS Code with Codex is used for code writing, documentation, notebook authoring, and project structure management.
- ChatGPT helps with research direction, methodology, step-by-step planning, and drafting prompts for Codex.
- Codex updates local files and notebooks based on the agreed prompts.
- The local VS Code repository is the source of truth for code and documentation.
- Git commits are performed manually by the user from the local Windows terminal.
- `third_party/PianistTransformer/` is ignored by the parent Git repository.

### Compute Environment

- GPU inference and future model training use Google Colab.
- Local Windows is CPU-only for this project and is not the primary inference environment.
- Google Drive persistent root: `/content/drive/MyDrive/MARG_research`
- Colab `/content` is temporary and must not be treated as persistent storage.
- Important outputs must be downloaded, saved to Google Drive, or copied into the local `outputs/` tree.

## Official Repository

- Official GitHub URL: `https://github.com/yhj137/PianistTransformer.git`
- Local third-party path: `third_party/PianistTransformer`
- Verified commit: `747df2d12291e37f6638b39f1b71517e579ad48c`
- Official checkpoint: `yhj137/pianist-transformer-rendering`

## Verified Colab Setup Sequence

These steps have been verified through the smoke-test and repeated-experiment notebooks.

1. Start a Google Colab GPU runtime.
2. Confirm Python, PyTorch, CUDA, and GPU availability.
3. Mount Google Drive when persistent storage is needed.
4. Use `/content/drive/MyDrive/MARG_research` as persistent project storage.
5. Clone or reuse `https://github.com/yhj137/PianistTransformer.git` in Colab.
6. Verify or checkout commit `747df2d12291e37f6638b39f1b71517e579ad48c`.
7. Install required dependencies when missing.
8. Prepare checkpoint `yhj137/pianist-transformer-rendering`.
9. Import official classes/functions without modifying third-party source:
   - `PianoT5Gemma`
   - `batch_performance_render`
   - `map_midi`
10. Move the model explicitly to CUDA.
11. Run inference with recorded generation parameters.
12. Save MIDI output, metadata JSON, and pedal-event CSV when using the experiment workflow.
13. Persist important files to Google Drive or download them before ending the runtime.

## Verified Smoke Test

- Runtime: Google Colab GPU
- Input MIDI: `/content/PianistTransformer/data/midis/testset/score/3.mid`
- Output MIDI: `/content/output_midi/0.mid`
- Device: `cuda`
- dtype: `torch.bfloat16`
- Temperature: `1.0`
- Top-p: `0.95`
- Inference time: about 66.26 seconds
- Generated token count: 4192
- Output MIDI created successfully
- Note count: 524
- CC64 event count: 6

## Verified Experiment Notebook Setup

The repeated baseline experiment notebook has been verified for:

- Google Drive mount.
- Persistent saving under `/content/drive/MyDrive/MARG_research`.
- Input MIDI upload.
- Seed setting.
- Repeated inference workflow.
- Pedal analysis table generation.
- Metadata JSON saving.
- Pedal-event CSV saving.

## New Colab Session Checklist

Run these again after a Colab runtime reset or new session:

1. Select GPU runtime.
2. Run GPU/Python/PyTorch/CUDA checks.
3. Mount Google Drive if persistent storage is needed.
4. Recreate or verify storage folders:
   - `inputs`
   - `outputs/midi`
   - `experiment_metadata`
   - `checkpoints`
5. Clone or reuse the Pianist Transformer repository.
6. Verify the repository commit.
7. Install missing dependencies.
8. Restore or download the checkpoint.
9. Load the model to CUDA.
10. Run smoke-test or experiment cells.

Within the same live Colab session, the model does not need to be reloaded for each experiment. After setup, repeat only the input/parameter/inference/analysis/save cells.

## Local Notes

The local Windows machine has no detected NVIDIA GPU/CUDA path and is not the primary inference runtime. Local work focuses on source control, documentation, notebook authoring, and research organization.