# Usage

## Baseline Inference

The first official Pianist Transformer pretrained checkpoint inference has been reproduced successfully on Google Colab GPU. Local Windows CPU inference remains a secondary fallback and has not been validated yet.

## Colab GPU Inference Notebook

Notebook: `notebooks/pianist_transformer_colab_inference.ipynb`

Use this notebook for the primary pretrained inference workflow. The notebook avoids `sh script/inference.sh` because the official script path uses CPU by default. Instead, it imports official Pianist Transformer classes and functions, moves the model to CUDA, and writes output MIDI under `/content/output_midi/`.

### Successful Smoke-Test Flow

1. Open the notebook in Google Colab.
2. Select `Runtime > Change runtime type > GPU`.
3. Confirm CUDA is available.
4. Clone or reuse `https://github.com/yhj137/PianistTransformer.git`.
5. Use repository commit `747df2d12291e37f6638b39f1b71517e579ad48c` for the recorded smoke test.
6. Install required non-GUI dependencies if missing.
7. Download or reuse checkpoint `yhj137/pianist-transformer-rendering`.
8. Load `PianoT5Gemma` with dtype `torch.bfloat16` when supported by the assigned GPU.
9. Move the model to `cuda` explicitly.
10. Run `batch_performance_render` with `temperature=1.0`, `top_p=0.95`, and `device="cuda"`.
11. Convert the generated performance with `map_midi`.
12. Save the result to `/content/output_midi/0.mid`.
13. Download the output MIDI or save it to Google Drive before ending the runtime.

### Verified Smoke-Test Result

- Input MIDI: `/content/PianistTransformer/data/midis/testset/score/3.mid`
- Output MIDI: `/content/output_midi/0.mid`
- Device: `cuda`
- dtype: `torch.bfloat16`
- Inference time: 66.26 seconds
- Generated token count: 4192
- Output size: 6661 bytes
- Instrument count: 1
- Note count: 524
- CC64 sustain pedal event count: 6

## Inputs

Use official bundled score MIDI files under `/content/PianistTransformer/data/midis/testset/score/` for baseline smoke tests. For pedal analysis, choose examples with richer CC64 sustain-pedal behavior than `score/3.mid`.

## Outputs

Colab outputs under `/content` are temporary. Download important MIDI files or save them to Google Drive, then copy selected results into the local project under `outputs/midi/` when they should become part of the research record.

## Notes

The smoke-test output played normally in MuseScore and sounded expressive, but the selected example is not a strong repedaling test case because it produced only 6 CC64 sustain-pedal events.