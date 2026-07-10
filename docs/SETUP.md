# Setup

## Environment

## Commands Tested

## Notes
## Repository Analysis

Date: 2026-07-10

### Official Repository

- Official GitHub URL: https://github.com/yhj137/PianistTransformer.git
- Local path: `third_party/PianistTransformer`
- Cloned commit: `747df2d12291e37f6638b39f1b71517e579ad48c`
- Confirmation sources checked: official project page, GitHub README, cloned repository files.

### Key Files and Roles

- `README.md`: official quick start, environment requirements, model links, checkpoint download command, inference command, and expected output path.
- `requirements.txt`: Python dependencies. Core packages include `transformers==4.54.0`, `datasets==4.0.0`, `accelerate==1.10.1`, `miditoolkit==1.0.1`, `modelscope==1.32.0`, and `huggingface_hub==0.35.3`. GUI-only packages include `PyQt5` and `pygame`.
- `script/inference.sh`: sets `PYTHONPATH=.` and runs `python src/inference/inference.py`.
- `src/inference/inference.py`: loads `models/sft/` with `torch_dtype=torch.bfloat16`, reads `data/midis/testset/score/0.mid`, runs CPU inference via `batch_performance_render(..., device="cpu")`, and writes `data/midis/testset/inference/0.mid`.
- `src/utils/download_model.py`: downloads the fine-tuned rendering model `yhj137/pianist-transformer-rendering` from Hugging Face or ModelScope into `models/sft`.
- `src/model/generate.py`: generation and MIDI tempo-map post-processing logic used by inference.
- `configs/pretrain_config.json`, `configs/sft_config.json`, `configs/ds_config.json`: training and DeepSpeed configs; not required for baseline inference, but they show bf16 and training defaults.
- `data/midis/testset/score/`: example score MIDI files used by the official inference script.
- `data/midis/testset/inference/`: official expected output directory. The cloned repository already contains an example `0.mid`; this was not generated in this project session.

### Official Inference Procedure

These steps are derived from the official README and local source inspection. Only the clone step has been executed in this project so far.

1. Clone the repository under `third_party/PianistTransformer`.
2. Enter the cloned repository directory.
3. Create an isolated Python 3.11 environment with conda or venv.
4. Install PyTorch 2.7.1 for the target compute backend:
   - NVIDIA CUDA 12.8 wheel if CUDA 12.8 is available.
   - NVIDIA CUDA 11.8 wheel if CUDA 11.8 is available.
   - CPU-only/default PyTorch install if no NVIDIA GPU is available.
5. Install remaining dependencies with `pip install -r requirements.txt`.
6. Download the fine-tuned model checkpoint:
   - `python -m src.utils.download_model --source huggingface`
   - Alternative: `python -m src.utils.download_model --source modelscope`
7. Confirm `generation_config.json`, `config.json`, and `model.safetensors` exist under `models/sft/`.
8. Run official inference from the repository root:
   - `sh script/inference.sh`
9. Expected output path after successful inference:
   - `data/midis/testset/inference/0.mid`

### Windows Compatibility Notes

- The official commands use `.sh` scripts and `export PYTHONPATH=.`. On Windows PowerShell, the equivalent for inference is likely `$env:PYTHONPATH='.'` followed by `python src/inference/inference.py`, or use Git Bash/WSL to run the `.sh` script. This PowerShell equivalent has not been tested yet.
- The current machine previously showed no installed Python, no conda, and no NVIDIA/CUDA availability. CPU inference is expected to be possible according to the official README, but it has not been tested here.
- `src/inference/inference.py` hardcodes `torch_dtype=torch.bfloat16` while running with `device="cpu"`. CPU bfloat16 support can be slower or problematic depending on PyTorch/CPU kernels. If it fails, a local wrapper or a minimal documented third-party patch may be needed, but no third-party code has been modified.
- Training/data-processing scripts include Linux-style shell behavior. `src/utils/midi.py` also references `timeout 120s ./MIDIToMIDIAlign.sh ...`, which is likely not Windows-native. This is not on the baseline inference path.
- Training configs use `bf16` and `wandb`; these are not needed for first pretrained inference.
- Paths in official scripts are relative and assume commands are run from the `third_party/PianistTransformer` repository root.

### Status at Initial Repository Analysis

At the initial repository-analysis step, before the Colab smoke test, Python environment creation, package installation, checkpoint download, and model inference had not yet been performed. Third-party source code was not modified.

## Local CPU vs Colab GPU Execution Plan

Date: 2026-07-10

### Scope

This section compares two planned paths for reproducing pretrained Pianist Transformer inference:

- Path A: Windows local CPU inference on the current computer.
- Path B: Google Colab GPU inference.

No environment creation, package installation, checkpoint download, source-code modification, model inference, Git configuration change, or Git commit was performed while preparing this plan.

### Inputs Checked

- Project instructions: `AGENTS.md`.
- Official repository: `third_party/PianistTransformer`.
- Official files checked locally: `README.md`, `requirements.txt`, `script/inference.sh`, `src/inference/inference.py`, `src/utils/download_model.py`, `src/model/generate.py`, and `configs/*.json`.
- Official Colab FAQ checked for runtime, GPU availability, Drive persistence, and resource-limit behavior: https://research.google.com/colaboratory/faq.html
- Current local computer status checked with read-only commands.

### Required Versions and Dependencies

- Recommended Python: `3.11` according to the official README.
- Required PyTorch: `2.7.1` according to the official README.
- Required Transformers: `4.54.0` according to the official README and `requirements.txt`.
- Other notable dependencies: `datasets==4.0.0`, `accelerate==1.10.1`, `miditoolkit==1.0.1`, `partitura==1.7.0`, `huggingface_hub==0.35.3`, and optionally `modelscope==1.32.0`.
- GUI-only dependencies listed in `requirements.txt`: `PyQt5==5.15.11`, `pygame==2.6.1`. These are not needed for minimal command-line inference.

### Current Local Computer Summary

- OS: Windows 10 Home, display version `25H2`, build `26200.8655`.
- CPU: 13th Gen Intel(R) Core(TM) i5-1340P.
- RAM: 15.57 GB total; approximately 3.48 GB available at check time.
- GPU: Intel(R) Iris(R) Xe Graphics, driver `32.0.101.6874`.
- NVIDIA/CUDA: `nvidia-smi` was not available; no NVIDIA CUDA runtime was detected.
- Python: `python` was not found; `py -0p` reported no installed Pythons.
- Conda: `conda` was not found.
- Disk: C: has approximately 48.13 GB free.
- VS Code extensions: Microsoft Jupyter extensions are installed; a separate Google Colab-specific extension was not observed in the local extension folder.

### CUDA and bfloat16 Findings

- Minimal command-line inference is not CUDA-only. In `src/inference/inference.py`, `.cuda()` is commented out and `batch_performance_render(..., device="cpu")` is hardcoded.
- The same file still loads the model with `torch_dtype=torch.bfloat16`. This is the main local CPU risk: CPU bfloat16 support may be slow or may fail depending on PyTorch CPU kernels and the processor path.
- Training configs use `bf16: true`, and training scripts are more GPU-oriented. Training is out of scope for baseline inference.
- Some non-inference paths contain Linux/shell assumptions, e.g. `.sh` scripts and alignment-related shell commands. These are not required for first pretrained inference.

### Path A: Windows Local CPU Execution Plan

Goal: run the official pretrained inference locally on CPU after installing a project-specific environment.

Planned steps, not yet executed:

1. Install or make available Python 3.11 without modifying system Python unexpectedly.
2. Create an isolated project environment, preferably `.venv` under the project root or a named conda environment if conda is installed later.
3. From `third_party/PianistTransformer`, install CPU-compatible PyTorch 2.7.1 and the remaining requirements.
4. Download the official fine-tuned rendering checkpoint with:
   - `python -m src.utils.download_model --source huggingface`
5. Confirm the following files exist under `third_party/PianistTransformer/models/sft/`:
   - `generation_config.json`
   - `config.json`
   - `model.safetensors`
6. Convert the official shell script to PowerShell or run via Git Bash:
   - Official shell form: `sh script/inference.sh`
   - PowerShell equivalent to test later: set `PYTHONPATH=.` for the current process, then run `python src/inference/inference.py` from the repository root.
7. Verify output:
   - `third_party/PianistTransformer/data/midis/testset/inference/0.mid`

Expected local bottlenecks:

- No NVIDIA GPU, so inference is CPU-only.
- Autoregressive generation can be slow on CPU, especially for longer MIDI files.
- `torch_dtype=torch.bfloat16` on CPU is a compatibility risk.
- Available RAM at check time was only about 3.48 GB; closing other applications before inference may help.

Expected local download/storage footprint:

- Repository already cloned: about 13.79 MB in `third_party/PianistTransformer`.
- Official checkpoint: README says approximately 270 MB, stored under `third_party/PianistTransformer/models/sft/`.
- PyTorch CPU wheel and Python dependencies: likely several hundred MB in the environment directory, depending on pip cache and optional GUI packages.
- Example score MIDI files: 23 files, about 396 KB total under `data/midis/testset/score/`.

### Path B: Google Colab GPU Execution Plan

Goal: run pretrained inference in a Colab notebook with a GPU runtime.

Colab feasibility:

- Colab provides hosted Jupyter notebooks with optional GPU/TPU access, but GPU type and availability vary over time and are not guaranteed on the free tier.
- A Colab GPU runtime should be suitable for this 135M-parameter inference workload if a GPU is assigned.
- The official shell script can run in Colab with `!sh script/inference.sh`, but this will still use CPU because the official `src/inference/inference.py` hardcodes `device="cpu"` and does not move the model to CUDA.
- To actually use Colab GPU without modifying third-party source files, the notebook should import the official functions and run a small notebook cell that loads the model with `.to("cuda")` and calls `batch_performance_render(..., device="cuda")`.

Shell script conversion:

- The official `script/inference.sh` contains only:
  - `export PYTHONPATH=.`
  - `python src/inference/inference.py`
- In Colab notebook cells this becomes either:
  - `%env PYTHONPATH=.` followed by `!python src/inference/inference.py`, for faithful CPU behavior, or
  - a Python cell using the official modules directly, for GPU behavior without editing source files.

Checkpoint and example MIDI preparation in Colab:

1. Clone the official repository into `/content/PianistTransformer`, or upload/use this project's cloned copy through GitHub/Drive if later desired.
2. `%cd /content/PianistTransformer`.
3. Install the required PyTorch and dependencies for the active Colab runtime.
4. Download the checkpoint with `python -m src.utils.download_model --source huggingface`.
5. Use the bundled example score MIDI at `/content/PianistTransformer/data/midis/testset/score/0.mid`.
6. Save generated output to `/content/PianistTransformer/data/midis/testset/inference/0.mid`, then download it or copy it to Drive.

Google Drive use:

- Google Drive is not required for a minimal one-shot test.
- Drive is useful for persistence because Colab VM files and installed packages disappear after runtime deletion or restart.
- Best simple practice: run repo, dependencies, and checkpoint in `/content` for speed; copy only important outputs and optionally the checkpoint cache to Drive after success.
- Avoid heavy repeated reads/writes directly inside mounted Drive because the Colab FAQ notes Drive mount and I/O can be slow or quota-sensitive.

Runtime restart implications:

- After a Colab runtime restart or VM deletion, the notebook remains, but installed packages, cloned repository, downloaded checkpoint, and generated files in `/content` must be recreated unless copied to Drive.
- Steps normally repeated after restart: select GPU runtime, verify `nvidia-smi`, clone repository, install packages, download or restore checkpoint, run inference.
- If checkpoint is stored in Drive, it can be copied back to `/content/PianistTransformer/models/sft/` instead of downloaded again.

VS Code / Colab extension use:

- Local VS Code has Jupyter support installed, so it can edit `.ipynb` files locally.
- A dedicated Google Colab extension was not observed locally.
- The simplest reliable Colab workflow is still browser-based Colab execution, with the notebook saved in Drive or downloaded into `notebooks/` afterward.
- VS Code can be used for drafting and reviewing the notebook, but it should not be treated as the primary way to execute on Colab GPU unless a known, working Colab connection extension is installed and verified later.

Expected Colab download/storage footprint:

- Repository: about 13.79 MB cloned to `/content/PianistTransformer`.
- Checkpoint: approximately 270 MB under `/content/PianistTransformer/models/sft/`.
- PyTorch GPU wheels and dependencies: commonly much larger than CPU-only setup, potentially multiple GB depending on the selected CUDA wheel and what Colab already has preinstalled.
- Output MIDI: very small, stored initially under `/content/PianistTransformer/data/midis/testset/inference/0.mid`; optionally copied to Drive.

### Pros and Cons

| Path | Pros | Cons |
| --- | --- | --- |
| Windows local CPU | Best for reproducible local project state; output is already inside the working project tree; no Colab session limits; no Drive sync needed. | Requires installing Python 3.11 and dependencies first; no CUDA acceleration; CPU inference may be slow; bfloat16-on-CPU may be fragile; local RAM headroom is modest. |
| Colab GPU | Faster path to GPU-backed inference; avoids local Python setup at first; easy to validate CUDA behavior; notebook can document every step. | Runtime resources are not guaranteed; setup must be repeated after runtime reset unless artifacts are persisted; official script uses CPU unless notebook wrapper uses CUDA; output must be copied back to local project. |

### Recommended Path for This Project

Recommended first path: Google Colab GPU for the first successful pretrained inference, then local CPU later as a reproducibility fallback.

Reasoning:

- The current local machine has no Python, no conda, and no NVIDIA CUDA path, so local setup requires more installation work before any model test.
- The official README says CPU inference is possible, but this machine's CPU-only path has the extra risk of `torch_dtype=torch.bfloat16`.
- Colab can validate the checkpoint and inference workflow faster, especially if a GPU runtime is available.
- Once the minimal output MIDI is produced, copy it back into this project under `outputs/midi/` and document the exact notebook cells and output path.

### Proposed Colab Notebook Structure

1. Title and scope
   - State that the notebook reproduces pretrained Pianist Transformer inference only.
2. Runtime check
   - `!nvidia-smi`
   - Python and PyTorch version checks.
3. Clone official repository
   - Clone `https://github.com/yhj137/PianistTransformer.git` into `/content/PianistTransformer`.
   - Optionally checkout commit `747df2d12291e37f6638b39f1b71517e579ad48c` for consistency with this local project.
4. Install dependencies
   - Install PyTorch 2.7.1 for the active Colab CUDA runtime if needed.
   - Install non-GUI dependencies from `requirements.txt`; consider skipping `PyQt5` and `pygame` for command-line inference.
5. Download checkpoint
   - Run `python -m src.utils.download_model --source huggingface`.
   - Verify `models/sft/generation_config.json`, `models/sft/config.json`, and `models/sft/model.safetensors`.
6. Verify example MIDI
   - Confirm `data/midis/testset/score/0.mid` exists.
7. Minimal inference
   - First option: faithful official CPU script with `%env PYTHONPATH=.` and `!python src/inference/inference.py`.
   - Preferred GPU option: notebook Python cell that imports official modules, loads `models/sft/`, moves the model to CUDA, calls `batch_performance_render(..., device="cuda")`, maps the MIDI, and writes `data/midis/testset/inference/0.mid`.
8. Output verification
   - Check that `data/midis/testset/inference/0.mid` exists and has nonzero size.
9. Persist result
   - Download the MIDI from Colab or copy it to Google Drive.
   - Later copy it into this local project under `outputs/midi/` and document that transfer.

### Minimal Verification Procedure

Local CPU path, to run later:

1. Confirm Python 3.11 environment is active.
2. Confirm `python -c "import torch; print(torch.__version__)"` reports PyTorch 2.7.1.
3. Confirm checkpoint files exist under `third_party/PianistTransformer/models/sft/`.
4. From `third_party/PianistTransformer`, run the PowerShell equivalent of the official inference script.
5. Confirm `data/midis/testset/inference/0.mid` exists and has nonzero size.
6. Copy or record the output path in project docs.

Colab GPU path, to run later:

1. Confirm `!nvidia-smi` shows an assigned GPU.
2. Confirm `torch.cuda.is_available()` is `True`.
3. Confirm PyTorch version and CUDA runtime are compatible with the selected install command.
4. Confirm checkpoint files exist under `/content/PianistTransformer/models/sft/`.
5. Run one-file inference on `data/midis/testset/score/0.mid`.
6. Confirm `/content/PianistTransformer/data/midis/testset/inference/0.mid` exists and has nonzero size.
7. Save the output MIDI and notebook back to the local project or Google Drive.

### Open Risks

- Colab GPU availability and GPU type can vary by account, current load, and plan.
- Official script does not use GPU as written; GPU use requires a notebook wrapper or a documented source change later.
- Local CPU bfloat16 behavior is untested.
- Dependency installation has not been attempted on either path.
- Checkpoint download and inference have now succeeded once in Google Colab GPU using `yhj137/pianist-transformer-rendering`.
- The verified Colab output was `/content/output_midi/0.mid`; `/content` remains temporary, so important outputs must be downloaded or saved to Google Drive.

## Execution Strategy Decision

Date: 2026-07-10

Decision: first pretrained Pianist Transformer inference reproduction will be attempted on Google Colab GPU. Local Windows CPU inference remains a secondary reproducibility path.

Rationale:

- The current local PC has no detected NVIDIA GPU or CUDA runtime.
- The local PC currently has no Python or Conda installation available on PATH.
- The official inference script uses CPU by default, but the Colab notebook will call official classes and functions directly and move the model to CUDA without modifying third-party source code.
- CPU inference remains useful later for reproducibility, but it has higher setup cost and possible `bfloat16` compatibility risk.

Artifact prepared:

- `notebooks/pianist_transformer_colab_inference.ipynb`

Updated status after first Colab smoke test:

- The Colab GPU notebook workflow has been executed successfully once.
- The official checkpoint `yhj137/pianist-transformer-rendering` was used in Colab.
- A rendered output MIDI was generated at `/content/output_midi/0.mid`.

## Verified Colab GPU Baseline Inference

Date: 2026-07-10

The first official pretrained Pianist Transformer inference reproduction has been verified on Google Colab GPU.

### Verified Run

- Runtime: Google Colab GPU
- Input MIDI: `/content/PianistTransformer/data/midis/testset/score/3.mid`
- Output MIDI: `/content/output_midi/0.mid`
- Repository commit: `747df2d12291e37f6638b39f1b71517e579ad48c`
- Checkpoint: `yhj137/pianist-transformer-rendering`
- Device: `cuda`
- dtype: `torch.bfloat16`
- Temperature: `1.0`
- Top-p: `0.95`
- Inference time: 66.26 seconds
- Generated token count: 4192
- Output size: 6661 bytes
- Instrument count: 1
- Note count: 524
- CC64 sustain pedal event count: 6

### Status Update

- Colab GPU is now verified as the primary path for pretrained inference reproduction.
- Local Windows CPU remains a secondary fallback and has not been verified for inference.
- Files under `/content` are temporary; important outputs must be downloaded or saved to Google Drive.
- The first smoke-test example was musically playable in MuseScore, but it is not a strong repedaling-analysis candidate because it has only 6 CC64 sustain-pedal events.
