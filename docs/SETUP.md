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

### Not Yet Done

- Python environment creation has not been performed.
- Packages have not been installed.
- Checkpoints have not been downloaded.
- Model inference has not been executed.
- Third-party source code has not been modified.
