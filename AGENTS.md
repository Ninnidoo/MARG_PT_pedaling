# AGENTS.md

## Project

This repository is for research on improving sustain-pedal generation in
Pianist Transformer, with a focus on repedaling, pedal-event timing,
and pedal depth prediction.

The primary working environment is a shared laboratory GPU server.

## Important paths

- Project repository:
  `/home/intern_2026_summer/yimilkyun/MARG_PT_pedaling`

- Public datasets and public pretrained models:
  `/public/intern_2026_summer_public_dataset/ilkyun_data`

- Private checkpoints, outputs, logs, and experiment artifacts:
  `/private/intern_2026_summer_private_dataset/ilkyun_data`

Inside the Docker container, these are mounted as:

- Project: `/workspace/project`
- Public storage: `/workspace/public`
- Private storage: `/workspace/private`

Do not hard-code host paths in research code when container paths can be used.

## Host and container responsibilities

On the server host, only perform operations such as:

- Git operations
- Docker lifecycle management
- File inspection and organization
- Editing Docker and configuration files

Run the following only inside the Docker container:

- Python programs
- pip installation
- PyTorch code
- Data preprocessing
- Model inference
- Training and evaluation
- Tests that require project dependencies

Do not install Python or system packages on the host.

## Shared-server safety

This is a shared server.

Never run:

- `sudo`
- host-level `pip install`
- host-level `conda install`
- `apt install`
- `docker system prune`
- `docker image prune`
- `docker container prune`
- `docker volume prune`
- broad or destructive `rm -rf` commands

Never stop, remove, rename, or modify Docker resources belonging to other users.

Before using a GPU, inspect availability with `nvidia-smi`.

Expose only one GPU to this project container unless explicitly instructed
otherwise.

The current development container is:

`ilkyun-marg-pedaling-dev`

Do not recreate, remove, or replace it without explicit approval.

## Docker workflow

The development container uses `sleep infinity` as its main process so it can
remain running while shell sessions are opened and closed.

Enter it with:

```bash
docker exec -it ilkyun-marg-pedaling-dev bash
```

For VS Code, connect to the host with Remote SSH, then use
`Dev Containers: Attach to Running Container...`, select
`ilkyun-marg-pedaling-dev`, and open `/workspace/project`.

Do not use `Dev Containers: Reopen in Container` to replace the current
container unless the user has approved a build or recreation. The checked-in
`.devcontainer/devcontainer.json` describes a reproducible future environment;
it is not proof of the current running container's image or GPU assignment.

Read-only inspection on 2026-07-30 verified that the current container:

- is running as `ilkyun-marg-pedaling-dev`;
- uses image `pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime`;
- uses `/workspace/project` as its working directory;
- runs `sleep infinity`;
- has the project, public, and private bind mounts listed above; and
- exposes exactly one RTX 2080 Ti, which appears inside the container as GPU 0.

Before future work, re-check the live state instead of assuming it is
unchanged. The current runtime and the checked-in Docker/Dev Container
configuration differ; reconcile them only after explicit approval.

## Storage and bind-mount rules

| Purpose | Host path | Container path | Access |
| --- | --- | --- | --- |
| Project code and ignored third-party source | `/home/intern_2026_summer/yimilkyun/MARG_PT_pedaling` | `/workspace/project` | read/write |
| Public datasets and public pretrained models | `/public/intern_2026_summer_public_dataset/ilkyun_data` | `/workspace/public` | read-only |
| Private checkpoints, outputs, logs, and caches | `/private/intern_2026_summer_private_dataset/ilkyun_data` | `/workspace/private` | read/write |

Datasets, checkpoints, outputs, logs, and caches must remain on bind-mounted
host storage so they survive container termination, removal, and recreation.
Do not copy them into a Docker image. Do not put generated artifacts in Git.
Infrastructure configuration may contain the fixed server mount paths, but
research code should prefer container paths, command-line arguments, or
environment variables.

Do not mount the Docker socket, host home directory, SSH keys, Git
configuration, credentials, `.env`, or other secrets into the container.
Never bake passwords, API keys, tokens, datasets, checkpoints, or outputs into
an image. Do not use privileged mode, host networking, host PID, or host IPC.

## GPU rules

- Expose exactly one assigned GPU to the container; never request `all`.
- Confirm the assignment and availability with `nvidia-smi` before use.
- Confirm inside the container that `torch.cuda.device_count()` is exactly 1.
- The selected physical GPU appears inside the container as `cuda:0`.
- The current GPU is an NVIDIA GeForce RTX 2080 Ti. Prefer FP16 where the
  verified code path supports it; do not assume BF16 support from the Colab
  workflow.
- Do not start multi-GPU work, a large training run, or a long GPU job without
  explicit approval.

## Git and third-party rules

- The Remote SSH host repository is the source of truth for code and
  documentation.
- Perform Git inspection, diff, add, commit, pull, and push only on the host.
- Codex must not commit or push unless the user explicitly changes this rule.
- Do not create GitHub authentication inside the container.
- Do not copy or mount host SSH keys, Git configuration, or Git credentials
  into the container.
- Do not change global or system Git configuration.
- Never commit datasets, checkpoints, generated outputs, logs, caches,
  credentials, `.env`, or secrets.
- Keep the official external repository at
  `third_party/PianistTransformer/`. The parent repository ignores this path,
  and the external source is managed separately on the host.
- Preserve the verified Pianist Transformer commit
  `747df2d12291e37f6638b39f1b71517e579ad48c` unless a change is explicitly
  approved.
- Avoid modifying third-party source. If it is unavoidable, document the exact
  commit, reason, patch, and affected experiment.

## Repository structure

- `third_party/`: external repositories, including Pianist Transformer
- `src/`: original research code
- `scripts/`: executable utilities and experiment entry points
- `data/raw/`: immutable original data; do not overwrite it
- `data/interim/`: intermediate processed data
- `data/processed/`: final experiment datasets
- `checkpoints/`: local path only when needed; weights must remain untracked
- `outputs/`: generated MIDI, audio, figures, and analysis outputs; untracked
- `experiments/`: experiment configurations, commands, metrics, and notes
- `docs/`: setup, usage, research, and troubleshooting documentation
- `notebooks/`: exploratory analysis only
- `tests/`: automated tests
- `logs/`: execution logs; untracked

Put original research code under `src/` or `scripts/`. Preserve the successful
Colab smoke-test notebook as the verified reference workflow, and do not break
it with exploratory notebook changes. Colab remains a verified reference and
fallback environment; new shared-server work should use the project Docker
environment.

When using the verified Colab fallback, treat `/content` as temporary and save
important outputs under `/content/drive/MyDrive/MARG_research`. ChatGPT may help
with research direction, methodology, and planning; Codex may create or update
local project files only within the agreed task scope.

## Required Codex workflow

For every task, use this order:

1. Analyze `AGENTS.md`, relevant documentation, code, and current state.
2. Inspect Git status and runtime state with read-only commands when relevant.
3. Present or internally establish a scoped plan, risks, and affected files.
4. Make the smallest approved, reversible change.
5. Run proportionate tests or static validation.
6. Confirm expected output files for experiments and inference.
7. Report changed files, commands actually run, warnings, and unresolved items.

A request to analyze, diagnose, review, or report does not authorize an
implementation. Do not make unrelated changes. Do not silently upgrade or
replace Python, PyTorch, CUDA, or other dependencies; explain the reason and
obtain approval first.

Ask for explicit approval before:

- building, recreating, removing, or replacing a container or environment;
- installing or upgrading packages;
- downloading unusually large files or models;
- running full Pianist Transformer training;
- running a long or large GPU experiment;
- changing Python, PyTorch, CUDA, base-image, or checkpoint versions; or
- executing a command with broad, destructive, or cross-user impact.

## Current research direction

### Active priorities

1. Preserve the successful Colab GPU smoke-test notebook as the verified
   reference workflow.
2. Use the repeated-experiment notebook for baseline pedal-behavior probes.
3. Select pedal-heavy score MIDI examples before drawing repedaling
   conclusions.
4. Generate and inspect output MIDI, especially CC64 sustain-pedal events.
5. Record seed, input, model commit, checkpoint, generation parameters,
   runtime, output path, and pedal metrics for every experiment.
6. Do not begin model training until the research question, annotation
   protocol, and dataset scope are clearer.

The verified baseline is the official Pianist Transformer checkpoint
`yhj137/pianist-transformer-rendering` at commit
`747df2d12291e37f6638b39f1b71517e579ad48c`. The successful Colab GPU smoke
test remains the reference inference workflow.

### Stage 1: pedal-tokenizer information-loss analysis

- Treat raw human CC64 as the source of truth and measure what the official
  tokenizer preserves or loses in depth, transition timing, and repedaling.
- The verified batch analyzed 165 of 166 human MIDI files.
- It found 15,549 raw repedal candidates, 13,256 preserved candidates, and
  2,293 lost candidates: micro recall 0.8525 and macro recall 0.8288.
- These are exploratory operational candidates, not final repedaling labels.

### Stage 2: event-based pedal-target audit

- Audit up to two UP/DOWN transition slots per unique-note-onset IOI, normalized
  event time `tau`, and a separate pedal-ON depth target.
- The verified audit found two-slot lossless coverage of 0.998761 and 697
  overflow IOIs across 165 analyzed files.
- Depth boundaries, hysteresis, rapid-reversal treatment, and the final
  annotation protocol remain open decisions.
- Stage 2 is target analysis and prototype validation, not model prediction or
  training.

The next work should stay small and auditable: refine the research question,
validate the target prototype, define the annotation protocol, and select
pedal-heavy examples. Do not interpret exploratory pedal metrics as final
research conclusions or start model training before the target and dataset
scope are clear.

## Canonical Stage 1 and Pedal-Only Test Invariant

- The canonical ASAP test Stage 1 performances are the existing CPU/seed-42
  Original Pianist Transformer MIDI files at
  `outputs/midi/{num}_original_pt.mid`. Never regenerate them for an ASAP test
  experiment; every current and future Stage 2 pedal model must share their
  exact non-pedal performance.
- A Stage 2 test candidate may change sustain-pedal CC64 events only. Relative
  to its canonical MIDI, note count/order, pitch, onset, offset/duration,
  velocity, ticks per beat, tempo, program/instrument/channel layout, time and
  key signatures, pitch bends, markers/lyrics, every non-CC64 controller, and
  every other renderer-relevant MIDI/meta event (including end-of-track
  timing) must remain exactly identical.
- Do not create canonical test candidates by reconstructing a full MIDI from
  Stage 2 tokens. Preserve the canonical MIDI as the source of truth and
  transplant only the predicted CC64 schedule from the established PT MIDI
  conversion path.
- Before any full evaluation, hard-assert strict non-CC64 equality. Stop before
  metrics or reports if it fails. This invariant applies to every Stage 2 pedal
  experiment, not only binary two-class models.

For every experiment, record seed, input, model commit, checkpoint, generation
parameters, runtime, output path, and relevant pedal metrics.

## Session close workflow

When the user says
`오케이 오늘 연구 진행상황 총정리해서 노션에 기록해줘`, follow
`docs/SESSION_CLOSE_WORKFLOW.md`.

For that workflow:

1. Check current Git status and changed files.
2. Read `docs/EXPERIMENT_LOG.md`, `docs/DECISIONS.md`,
   `docs/NEXT_SESSION.md`, and recent small experiment metadata.
3. Build a grounded local Markdown session log first.
4. Publish the same content to Notion only after the local log exists and the
   user has requested the Notion record.
5. Never print Notion tokens or the full parent page ID.
6. Do not read secrets from anywhere except the project-root `.env` file.
7. Do not scan large files under `third_party/`, `checkpoints/`, or `outputs/`.
8. Do not infer unsupported work; write `미확인` instead.
9. Prevent duplicate Notion uploads using the Markdown file SHA-256 publish log.
10. Do not make Git commits from Codex.

## Documentation and validation

Whenever setup or execution succeeds:

- Record the exact tested command in `docs/SETUP.md` or `docs/USAGE.md`.
- Record important errors and fixes in `docs/TROUBLESHOOTING.md`.
- Record research decisions in `docs/RESEARCH_PLAN.md`.
- Record experiment commands, inputs, outputs, and results in the relevant
  experiment folder or `docs/EXPERIMENT_LOG.md`.

Do not present an untested command as confirmed working. Before declaring a
task complete, run feasible relevant validation, check exit status, confirm
expected files, report warnings and unresolved issues, and summarize every
added or modified file. Inference is complete only after confirming a
non-empty output MIDI file and reporting its persistent path.

See `docs/SERVER_WORKFLOW.md` for the human-readable shared-server workflow and
`docs/SERVER_DOCKER_SETUP.md` for Docker build and run references.
