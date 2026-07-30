# Project Overview

This repository is for a two-month research project using Pianist Transformer
as a baseline for sustain-pedal and repedaling refinement.

The initial environment setup and baseline inference reproduction have been
completed. The official Pianist Transformer checkpoint was verified on Google
Colab GPU, and the next goal is to refine the research question, analyze
baseline pedal behavior, and design repedaling-focused data.

# Current Stage

Environment setup and baseline inference reproduction completed.
Next stage is research question refinement, baseline pedal behavior analysis,
and repedaling data design.

Priorities:

1. Preserve the successful Colab GPU smoke-test notebook as the verified reference workflow.
2. Use the repeated experiment notebook for baseline pedal-behavior probes.
3. Select pedal-heavy score MIDI examples before drawing repedaling conclusions.
4. Generate and inspect output MIDI files, with special attention to CC64 sustain-pedal events.
5. Record seed, input, model commit, checkpoint, generation parameters, runtime, and output path for every experiment.
6. Do not begin model training until the research question, annotation protocol, and dataset scope are clearer.

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
- `third_party/PianistTransformer/` is ignored by the parent Git repository.
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
- ChatGPT helps with research direction, methodology, step planning, and Codex prompt drafting.
- Codex creates and updates local project files and notebooks from the agreed prompts.
- Important Colab outputs must be downloaded or saved to Google Drive.
- Files under `/content` must be treated as temporary.
- Google Drive path `/content/drive/MyDrive/MARG_research` is the default persistent Colab storage root.
- Git commits are performed manually by the user from the local Windows terminal.
- Do not interpret exploratory pedal metrics as final research conclusions.
- Do not define repedaling labels before the annotation protocol is finalized.
- Preserve the smoke-test notebook as a verified reference workflow.
- Experimental notebook changes must not break the verified baseline workflow.

# Session Close Workflow

When the user says `오케이 오늘 연구 진행상황 총정리해서 노션에 기록해줘`, follow `docs/SESSION_CLOSE_WORKFLOW.md`.

For that workflow:

1. Check current Git status and changed files.
2. Read `docs/EXPERIMENT_LOG.md`, `docs/DECISIONS.md`, `docs/NEXT_SESSION.md`, and recent small experiment metadata.
3. Build a grounded local Markdown session log first.
4. Publish the same content to Notion only after the local log exists and the user has requested the Notion record.
5. Never print Notion tokens or the full parent page ID.
6. Do not read secrets from anywhere except the project-root `.env` file.
7. Do not scan large files under `third_party/`, `checkpoints/`, or `outputs/`.
8. Do not infer work that is not supported by project documents or metadata; write `미확인` instead.
9. Prevent duplicate Notion uploads using the Markdown file SHA-256 publish log.
10. Do not make Git commits from Codex.

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

