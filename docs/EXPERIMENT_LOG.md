# Experiment Log

## Entries

| Date | Experiment | Command / Artifact | Result | Notes |
| --- | --- | --- | --- | --- |
| 2026-07-10 | Initialize local project structure | Local VS Code / Codex project setup | Success | Created project folders, README, docs skeletons, `.gitignore`, and `.gitkeep` files. |
| 2026-07-10 | Clone official Pianist Transformer repository | `git clone https://github.com/yhj137/PianistTransformer.git C:\Users\eagle\MARG_research\third_party\PianistTransformer` | Success | Cloned official repository. Parent Git repository ignores `third_party/PianistTransformer/`. |
| 2026-07-10 | Analyze official repository | `README.md`, `requirements.txt`, `script/inference.sh`, `src/inference/inference.py`, `src/utils/download_model.py` | Success | Identified Python 3.11, PyTorch 2.7.1, checkpoint download path, CPU default in official script, and CUDA notebook workaround. |
| 2026-07-10 | Create smoke-test Colab notebook | `notebooks/pianist_transformer_colab_inference.ipynb` | Prepared | Notebook imports official classes/functions and moves the model to CUDA without modifying third-party source. |
| 2026-07-10 | Official Example Smoke Test | Colab GPU notebook using official Pianist Transformer classes/functions | Success | First official pretrained checkpoint inference succeeded on Google Colab GPU. Details below. |
| 2026-07-10 | Create repeated baseline experiment notebook | `notebooks/02_pianist_transformer_experiments.ipynb` | Prepared | Notebook supports Drive storage, upload/Drive MIDI input, seeds, repeated samples, pedal metrics, JSON metadata, and CSV pedal-event tables. |
| 2026-07-10 | Repeated experiment workflow verification | `notebooks/02_pianist_transformer_experiments.ipynb` in Google Colab | Success | Google Drive mount, persistent saving, input upload, seed setting, repeated inference, pedal analysis, metadata JSON, and CSV save workflow were verified. |

## Official Example Smoke Test

Date: 2026-07-10

### Purpose

Verify that the official pretrained Pianist Transformer checkpoint can render one bundled example score MIDI in Google Colab GPU without modifying third-party source code.

### Environment

- Runtime: Google Colab GPU
- Device: `cuda`
- dtype: `torch.bfloat16`
- Repository commit: `747df2d12291e37f6638b39f1b71517e579ad48c`
- Checkpoint: `yhj137/pianist-transformer-rendering`

### Input and Output

- Input MIDI: `/content/PianistTransformer/data/midis/testset/score/3.mid`
- Output MIDI: `/content/output_midi/0.mid`
- Local saved copy: `outputs/midi/baseline/example_03/rendered_run01.mid`
- Local metadata copy: `outputs/midi/baseline/example_03/metadata.json`
- Output size: 6661 bytes

### Generation Parameters

- Temperature: `1.0`
- Top-p: `0.95`
- Seed: not recorded in the first smoke-test metadata

### Runtime and Output Metrics

- Inference time: 66.26 seconds
- Generated token count: 4192
- Instrument count: 1
- Note count: 524
- CC64 sustain pedal event count: 6

### Observation

MuseScore에서 정상적으로 재생되었으며 표현력이 느껴졌다. 다만 이 예제는 페달을 많이 사용하는 곡이 아니어서 repedaling 문제를 평가하기에는 적합하지 않았다.

## Repeated Experiment Workflow Verification

Date: 2026-07-10

### Verified Workflow

The repeated experiment notebook was executed in Google Colab and verified for workflow behavior:

- Google Drive mount worked.
- Persistent root `/content/drive/MyDrive/MARG_research` worked.
- Input MIDI upload worked.
- Seed configuration worked.
- Repeated inference workflow executed.
- Pedal analysis table generation worked.
- Metadata JSON saving worked.
- Pedal-event CSV saving worked.
- Google Drive persistent saving worked.

### Second Generation Note

A later generation of the same score produced a different sampling result and showed `CC64 event count = 156`.

This is recorded only as a future baseline-analysis target. The cause of the difference between 6 and 156 CC64 events is not interpreted here, and no research conclusion is drawn from it at this stage.

## End-of-Day Summary

- Local project root: `C:\Users\eagle\MARG_research`
- Official repository commit: `747df2d12291e37f6638b39f1b71517e579ad48c`
- Official checkpoint: `yhj137/pianist-transformer-rendering`
- Primary compute environment: Google Colab GPU
- Persistent Colab storage: `/content/drive/MyDrive/MARG_research`
- Local source of truth: VS Code repository at `C:\Users\eagle\MARG_research`
- Git commits: manual from local Windows terminal by the user
- Next stage: research question refinement, baseline pedal behavior analysis, and repedaling data design