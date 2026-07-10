# Project Overview

This repository is for a two-month research project using Pianist Transformer
as a baseline for sustain-pedal and repedaling refinement.

The immediate goal was to reproduce pretrained Pianist Transformer inference.
That baseline inference reproduction has now been completed once on Google
Colab GPU using the official pretrained checkpoint. The next goal is to build a
baseline experiment workflow and analyze generated pedal events, especially for
repedaling refinement.

# Current Stage

Baseline inference reproduction completed. The project is now in the baseline
experiment workflow and pedal analysis stage.

Priorities:

1. Preserve the successful Colab GPU inference workflow as the baseline path.
2. Run additional baseline inference examples with better pedal coverage.
3. Generate and inspect output MIDI files, with special attention to CC64 sustain-pedal events.
4. Compare generated pedal behavior against score/performance references.
5. Document every successful command, experiment parameter, output path, and unresolved issue.
6. Do not begin model training until the baseline experiment workflow and pedal analysis questions are clear.

# Repository Structure

- `third_party/`: external repositories, including the official Pianist Transformer code
- `src/`: original research code
- `scripts/`: executable utilities and experiment entry points
- `data/raw/`: immutable original data
- `data/interim/`: intermediate processed data
- `data/processed/`: final datasets used by experiments
- `checkpoints/`: pretrained and trained model weights
- `outputs/`: generated MIDI, audio, figures, and analysis outputs
- `experiments/`: experiment configurations, commands, metrics, and notes
- `docs/`: setup, usage, research, and troubleshooting documentation
- `notebooks/`: exploratory analysis only
- `tests/`: automated tests
- `logs/`: execution logs

# Working Rules

- Use a project-specific Python environment.
- Do not modify the system Python installation.
- Do not install packages globally.
- Do not overwrite files under `data/raw/`.
- Do not commit checkpoints, large datasets, generated outputs, or secrets.
- Do not hard-code machine-specific absolute paths in source code.
- Prefer relative paths or configuration files.
- Keep the official Pianist Transformer repository under `third_party/`.
- Avoid modifying third-party source code unless necessary.
- If third-party code must be changed, document exactly why and what changed.
- Put original research code under `src/` or `scripts/`.
- Do not train the full Pianist Transformer unless explicitly requested.
- Initially use the official pretrained checkpoint for inference only.
- Prefer small, reversible changes over large rewrites.
- Do not silently upgrade or replace dependencies.
- Explain the reason before changing Python, PyTorch, CUDA, or package versions.
- GPU inference and model training use Google Colab.
- Local VS Code repository is the source of truth for code and documentation.
- Important Colab outputs must be downloaded or saved to Google Drive.
- Files under `/content` must be treated as temporary.
- Every experiment should record seed, input, model commit, checkpoint, generation parameters, runtime, and output path.

# Documentation Rules

Whenever setup or execution succeeds:

- Record the exact command in `docs/SETUP.md` or `docs/USAGE.md`.
- Record important errors and fixes in `docs/TROUBLESHOOTING.md`.
- Record research decisions in `docs/RESEARCH_PLAN.md`.
- Record experiment commands, inputs, outputs, and results in the relevant
  experiment folder or `docs/EXPERIMENT_LOG.md`.

Documentation should reflect commands that were actually tested.
Do not present an untested command as confirmed working.

# Validation Rules

Before declaring a task complete:

1. Run the relevant command when feasible.
2. Check the command exit status.
3. Confirm that the expected files were created.
4. Report warnings and unresolved issues.
5. Summarize which files were added or modified.

For inference tasks, completion requires confirming that an output MIDI file
exists and reporting its path.

# Safety

- Ask before deleting files or replacing existing environments.
- Ask before downloading unusually large files.
- Never expose credentials, tokens, or private paths in committed files.
- Store secrets only in ignored environment files when needed.
- Do not make unrelated changes outside the requested task.