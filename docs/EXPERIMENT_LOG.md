# Experiment Log

## Entries

| Date | Experiment | Command | Result | Notes |
| --- | --- | --- | --- | --- |
| 2026-07-10 | Clone official Pianist Transformer repository | `git clone https://github.com/yhj137/PianistTransformer.git C:\Users\eagle\MARG_research\third_party\PianistTransformer` | Success | Cloned to `third_party/PianistTransformer` at commit `747df2d12291e37f6638b39f1b71517e579ad48c`. No environment setup, checkpoint download, inference, or commit performed. |
| 2026-07-10 | Create Colab GPU inference notebook skeleton | Created `notebooks/pianist_transformer_colab_inference.ipynb` and updated docs only | Prepared | Notebook not executed at creation time; no install, checkpoint download, inference, source modification, commit, or Git config change performed locally. |
| 2026-07-10 | Official Example Smoke Test | Colab GPU notebook using official Pianist Transformer classes/functions | Success | First official pretrained checkpoint inference succeeded on Google Colab GPU. Details recorded below. |

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
- Output size: 6661 bytes

### Generation Parameters

- Temperature: `1.0`
- Top-p: `0.95`
- Seed: not recorded in the provided smoke-test result

### Runtime and Output Metrics

- Inference time: 66.26 seconds
- Generated token count: 4192
- Instrument count: 1
- Note count: 524
- CC64 sustain pedal event count: 6

### Observation

MuseScore에서 정상적으로 재생되었으며 표현력이 느껴졌다. 다만 이 예제는 페달을 많이 사용하는 곡이 아니어서 repedaling 문제를 평가하기에는 적합하지 않았다.

### Follow-up

Use a different example or custom score with richer sustain-pedal behavior for repedaling analysis. Future experiments should record seed, input, model commit, checkpoint, generation parameters, runtime, and output path.