# Usage

## Roles and Workflow

- Local project root: `C:\Users\eagle\MARG_research`
- VS Code with Codex is used for code writing, documentation, notebook authoring, and project structure management.
- ChatGPT helps with research direction, methodology, step-by-step planning, and Codex prompt drafting.
- Codex creates and updates local project files and notebooks from those prompts.
- The local VS Code repository is the source of truth for code and documentation.
- GPU inference and future training run in Google Colab.
- Google Drive path `/content/drive/MyDrive/MARG_research` is the persistent Colab storage root.
- Colab `/content` is temporary.
- Important outputs should be saved to Google Drive or copied into local `outputs/`.
- Git commits are done manually by the user from the local Windows terminal.

## Smoke-Test Notebook

Notebook: `notebooks/pianist_transformer_colab_inference.ipynb`

Purpose: preserve the verified reference workflow for one official example MIDI. Use this notebook when checking that the official checkpoint, repository commit, CUDA model loading, and basic inference path still work.

Verified smoke-test result:

- Input MIDI: `/content/PianistTransformer/data/midis/testset/score/3.mid`
- Output MIDI: `/content/output_midi/0.mid`
- Repository commit: `747df2d12291e37f6638b39f1b71517e579ad48c`
- Checkpoint: `yhj137/pianist-transformer-rendering`
- Device: `cuda`
- dtype: `torch.bfloat16`
- Temperature: `1.0`
- Top-p: `0.95`
- Inference time: about 66.26 seconds
- Generated token count: 4192
- Note count: 524
- CC64 event count: 6

Do not casually change this notebook. It is the verified reference workflow.

## Repeated Experiment Notebook

Notebook: `notebooks/02_pianist_transformer_experiments.ipynb`

Purpose: run repeated baseline inference on arbitrary score MIDI inputs and save output MIDI, JSON metadata, and CSV pedal-event tables.

Use this notebook for baseline pedal-behavior experiments after the smoke-test path is known to work.

### Same Colab Session

Within the same live Colab session:

1. Run setup cells once.
2. Keep the model loaded on CUDA.
3. Repeat only the experiment cells:
   - choose/upload input MIDI
   - set seed, temperature, top-p, and sample count
   - run inference
   - analyze pedal events
   - save MIDI/JSON/CSV

### After Runtime Reset

After a Colab runtime reset, disconnection, or new session:

1. Start from setup again.
2. Re-check GPU availability.
3. Mount Google Drive again.
4. Reuse or clone the repository again.
5. Reinstall missing dependencies if necessary.
6. Restore or download the checkpoint.
7. Reload the model to CUDA.
8. Then run experiment cells.

## Storage

Default persistent storage:

`/content/drive/MyDrive/MARG_research`

Expected folders:

- `inputs`
- `outputs/midi`
- `experiment_metadata`
- `checkpoints`

If Drive is disabled, outputs under `/content` are temporary and must be downloaded before the runtime ends.

## Pedal Analysis Caution

The repeated experiment notebook computes exploratory pedal metrics. These metrics are not final research conclusions. In particular, fast off-on transition candidates must not be treated as repedaling labels until the annotation protocol is finalized.

The second generation result with `CC64 event count = 156` is recorded as a future baseline-analysis target only. Its cause is not interpreted yet.