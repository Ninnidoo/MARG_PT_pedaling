# Usage

## Baseline Inference

## Inputs

## Outputs

## Notes
## Colab GPU Inference Notebook

Notebook: `notebooks/pianist_transformer_colab_inference.ipynb`

Purpose: run the official pretrained Pianist Transformer rendering checkpoint on one bundled example MIDI in Google Colab GPU. This notebook is for first baseline inference reproduction and has not been executed yet.

Usage order:

1. Open the notebook in Google Colab.
2. Select `Runtime > Change runtime type > GPU`.
3. Run the GPU/runtime check cell. Stop if CUDA is unavailable.
4. Run the path definition cell.
5. Clone the official repository into `/content/PianistTransformer` if it is not already present.
6. Install missing non-GUI dependencies. The notebook does not reinstall Colab's existing PyTorch automatically.
7. Import official Pianist Transformer modules.
8. Download the official fine-tuned rendering checkpoint from Hugging Face if missing.
9. Select the smallest bundled score MIDI for a smoke test.
10. Load `PianoT5Gemma`, move it to CUDA explicitly, run `batch_performance_render`, map the MIDI with `map_midi`, and save output to `/content/output_midi/0.mid`.
11. Verify the output MIDI exists, is non-empty, and contains notes; record CC64 pedal event count when available.
12. Download the generated MIDI from Colab.

The notebook intentionally avoids `sh script/inference.sh` because the official script path currently runs `src/inference/inference.py`, which hardcodes CPU inference.
