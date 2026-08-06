# Stage 2 Encoder-Only v0: Repository Audit

## 1. Runtime setup

- Official source: `https://github.com/yhj137/PianistTransformer.git`, commit `747df2d12291e37f6638b39f1b71517e579ad48c`, clean detached HEAD at `/workspace/project/third_party/PianistTransformer`. The container has no `git` executable, so the authorized clone was performed with host Git into the same bind-mounted path; all Python work remained inside the container.
- Direct inference dependencies installed in the existing container: `transformers==4.54.0`, `datasets==4.0.0`, `accelerate==1.10.1`, `miditoolkit==1.0.1`, and `huggingface_hub==0.35.3`. Relevant resolved packages include `safetensors==0.8.0`, `mido==1.3.3`, and `pandas==3.0.5`. Existing PyTorch and NumPy were retained. `fsspec` changed from 2025.7.0 to 2025.3.0 to satisfy the pinned `datasets` requirement. Training, GUI, ModelScope, and system packages were not installed.
- Checkpoint: official Hugging Face `yhj137/pianist-transformer-rendering`, downloaded snapshot `8f1568201d1ec035c73939d5a67c5168c0b80221`, at `/workspace/project/checkpoints/pianist_transformer`. Total local size is 271,514,274 bytes; `model.safetensors` is 271,506,656 bytes. `config.json`, `generation_config.json`, and `model.safetensors` are present and `PianoT5Gemma.from_pretrained` loaded all 135,742,464 parameters successfully.
- Example MIDI: `/workspace/project/third_party/PianistTransformer/data/midis/testset/score/3.mid` (3,550 bytes, 534 raw MIDI notes; normalization yields 524 notes / 4,192 tokens). This is the small official example used by the project's prior verified smoke test; it was parsed in place and not copied.

## 2. Execution environment

- Host: `user`; host repository: `/home/intern_2026_summer/yimilkyun/MARG_PT_pedaling`.
- Container: `ilkyun-marg-pedaling-dev` (`d3affcb5b8ad`), image `pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime`, working/project path `/workspace/project`. The project, public, and private roots are bind-mounted, and the result files are visible on the host.
- Python: `/opt/conda/bin/python`, 3.11.13; no activated `CONDA_DEFAULT_ENV` or `VIRTUAL_ENV`. PyTorch `2.8.0+cu126`, PyTorch CUDA 12.6, Transformers 4.54.0, NumPy 2.3.2; CUDA available with exactly one container-visible device.
- Main repository: branch `main`, commit `1d64469760e963492897723c92e61ddad7c76a54`. Pre-existing changes (`AGENTS.md`, Docker/server setup files) were preserved. This task added only the ignored/isolated official clone, the checkpoint runtime resource, installed container packages, and this untracked analysis directory; it did not modify original research or third-party source.

## 3. GPU selection

Immediate pre-inference host state:

| Host index | UUID | Model | Used / free | Util. | Meaningful active compute | Decision |
|---:|---|---|---:|---:|---|---|
| 0 | `GPU-c6fcf288-c0d2-af93-4b99-579fb2817af0` | RTX 2080 Ti | 59 / 10,754 MiB | 0% | None; Xorg/desktop graphics only | Idle, not exposed to container |
| 1 | `GPU-6982dbee-fbaf-f359-d7ef-a22d0e83400b` | RTX 2080 Ti | 6 / 10,816 MiB | 0% | None; Xorg used 4 MiB only | Selected |

- Mapping: host physical GPU 1 → UUID `GPU-6982dbee-fbaf-f359-d7ef-a22d0e83400b` → container-visible GPU 0 → PyTorch logical `cuda:0` after `CUDA_VISIBLE_DEVICES=0`.
- Immediately before inference, PyTorch reported exactly one GPU, `NVIDIA GeForce RTX 2080 Ti`, with 0 allocated and 0 reserved bytes. No other user's process was changed or interrupted.

## 4. Relevant implementation

All source statements below were verified at the pinned commit; execution-specific statements are marked explicitly.

| Component | Exact implementation | Verified role |
|---|---|---|
| MIDI normalization | `src/utils/midi.py:10`, `normalize_midi` | Converts all non-drum tracks through real time to 500 ticks/beat at 120 BPM, repairs zero-length notes, clips same-pitch overlaps, merges tracks, and retains merged CC64 events. |
| MIDI → tokens | `src/utils/midi.py:136`, `midi_to_ids` | Normalizes by default, computes note IOI/duration and four CC64 samples, clamps each feature, and appends exactly `[Pitch, IOI, Velocity, Duration, Pedal1, Pedal2, Pedal3, Pedal4]` per note. |
| Tokens → MIDI | `src/utils/midi.py:184`, `ids_to_midi` | Iterates in steps of eight, reconstructs one note and four pedal samples per group, deduplicates consecutive equal CC64 values, and emits 500-TPB/120-BPM MIDI. An incomplete group is not tolerated: indexed access through `i+7` would fail rather than silently discard it. |
| Expressive remapping | `src/model/generate.py:130`, `map_midi` | Applies the generated performance to the score's musical grid/tempo mapping; used unchanged for the final MIDI. |
| Vocabulary/config | `src/model/pianoformer.py:33`, `PianoT5GemmaConfig`; checkpoint `config.json` | Vocabulary size 5,389. Specials: PAD=0, MASK=1, BOS=2, EOS=3, PLAY=4. Musical ID ranges are half-open: Pitch `[5,133)` (MIDI 0–127), IOI `[261,5252)` (ticks 0–4990), Velocity `[133,261)` (0–127), Duration `[261,5261)` (ticks 0–4999), and each pedal position `[5261,5389)` (CC64 0–127). |
| Eight-token grouping | `src/model/pianoformer.py:100`, `PianoEncoderEmbeddings`; `src/model/generate.py:55`, `slide_window` | The encoder reshapes `[B,L,768]` to `[B,L/8,8,768]`. Inference windows and strides are rounded to multiples of eight. Official MIDI tokenization always emits eight tokens per retained normalized note. |
| Feature-position information | `src/model/pianoformer.py:103-124` | A shared 5,389×768 token embedding is followed by eight distinct `768→96` linear projections; their concatenation produces one 768-vector per note. Pedal1–4 therefore have distinct encoder projection layers (positions 4–7). Decoder feature identity is also imposed by sequence position modulo eight and `valid_id_range`; all four pedals share the same 128 value IDs. There is no separate token-type embedding. |
| Encoder | `src/model/pianoformer.py:129`, `PianoT5GemmaEncoder` | Checkpoint has 10 layers, hidden size 768, 8 query heads, 4 KV heads, alternating sliding/full attention. Lines 193–200 construct full/sliding masks with Transformers' bidirectional mask functions, so encoder self-attention is bidirectional (sliding layers remain window-limited). |
| Decoder and LM head | `src/model/pianoformer.py:225`, `PianoT5GemmaModel`; `src/model/pianoformer.py:308`, `PianoT5Gemma` | Checkpoint has a 2-layer autoregressive `T5GemmaDecoder`; `T5GemmaLMHead` projects to 5,389 logits. Decoder input embedding and LM-head output weight are declared tied and were pointer-equal after loading. Encoder token embeddings are separate. |
| Note compression shapes | `PianoEncoderEmbeddings.forward` | Executed first window: token IDs `[1,4096]` → shared embeddings `[1,4096,768]` → grouped `[1,512,8,768]` → each projection `[1,512,96]` → compressed encoder input `[1,512,768]`. Second window: `[1,2144]` → `[1,2144,768]` → `[1,268,8,768]` → `[1,268,768]`. |
| Checkpoint loading | `src/utils/download_model.py:5`, `download_hf`; `src/inference/inference.py:11`; Transformers `from_pretrained` | Official snapshot download and `PianoT5Gemma.from_pretrained`. Safetensors contains 182 keys: prefixes `embeddings.*` (17), `model.encoder.*` (129), and `model.decoder.*` (36); there is no stored `lm_head.*` tensor because the head is tied. |
| Encoder transfer | `PianoT5Gemma.get_encoder` at `src/model/pianoformer.py:335` | Execution verified a loaded `PianoT5GemmaEncoder` is directly available from the full pretrained model. The repository provides no dedicated standalone encoder-checkpoint loader; a future transfer should load the full checkpoint and extract `get_encoder()` (or explicitly remap the `model.encoder.*` keys), not assume prefix-free loading. |
| Original inference | `script/inference.sh` → `src/inference/inference.py`; core `src/model/generate.py:43`, `batch_performance_render` | Official entry loads `models/sft`, uses temperature 1.0/top-p 0.95, and calls the core renderer. Its hard-coded CPU/output path would overwrite a bundled file, so the authorized inline wrapper imported and called the same functions with the selected GPU and protected output path; generation logic was not modified. |
| Pitch/grammar constraint | `src/model/generate.py:11`, `BatchSparseForcedTokenProcessor` | Every eighth output token is forced to the corresponding input pitch; other positions are masked to the configured feature range. Execution confirmed all 524 generated pitch IDs equal the score-token pitch IDs. |
| Generation and long sequences | `src/model/generate.py:43-127` | `do_sample=True`, max context 4,096 tokens (512 notes), overlap ratio 0.5, temperature 1.0, top-p 0.95, and output length equal to input length. Long inputs use overlapping eight-aligned windows and carry a 20% portion of the prior overlap into the next decoder call. The 4,192-token example executed windows `[0,4096)` and `[2048,4192)`. Checkpoint position capacity is 8,192, but this inference function rejects context values above 4,096. |
| Specials during generation | `src/model/generate.py:91-126` | Transformers generation prepends BOS; the renderer removes it with `output[:,1:]` and returns only the musical tokens. Feature-range masking prevents PAD/MASK/BOS/EOS/PLAY as musical outputs, and fixed-length generation produced no EOS suffix. The NPZ therefore records prefix `[2]`, empty suffix, and the unmodified 4,192 returned musical IDs. |
| Seed behavior | `src/inference/inference.py` and executed wrapper | The official script sets no seed. Generation samples, so Python/NumPy/PyTorch/CUDA seeds and temperature/top-p affect reproducibility. With all seeds reset to 20260710, two 18–19 second runs produced bit-identical token arrays and identical MIDI hashes in this environment. |
| ASAP loading/split | `src/data_process/06_generate_sft_data.py:14-58`; `src/train/sft.py:163-188` | Reads ASAP `metadata.csv`, groups by score, sorts/shuffles score paths with Python seed 42, takes the first 10% of unique scores as test, writes `split`, then SFT loads JSON and filters train/test. The bundled repository is only a one-row ASAP subset, yielding one `train` record and no test record; no separate canonical ASAP split file is present. |

## 5. Baseline inference

- Effective command: `docker exec ilkyun-marg-pedaling-dev bash -lc 'cd /workspace/project/third_party/PianistTransformer && CUDA_VISIBLE_DEVICES=0 python -'`, with an inline, non-persistent wrapper importing official `PianoT5Gemma`, `batch_performance_render`, and `map_midi`.
- Model load: `PianoT5Gemma.from_pretrained('/workspace/project/checkpoints/pianist_transformer', torch_dtype=torch.float16)`, then `model.to('cuda:0').eval()`. FP16 was used for the RTX 2080 Ti; architecture, weights, tokenizer, constraints, sampling, and reconstruction code were unchanged.
- Core call: `batch_performance_render(model, [score], temperature=1.0, top_p=0.95, device='cuda:0')`; defaults preserved: `max_context_length=4096`, `overlap_ratio=0.5`, `pedal_offset=0.0`, `enabled_pedal_points=None`, `do_sample=True`.
- Seed 20260710 was applied to Python, NumPy, PyTorch, and CUDA; cuDNN benchmark was disabled. First inference: 18.901 s. Optional in-memory reproducibility inference: 18.578 s; no duplicate artifact remained.
- Input: `/workspace/project/third_party/PianistTransformer/data/midis/testset/score/3.mid`.
- Outputs: `/workspace/project/analysis/stage2_encoder_only_v0/baseline_generated.mid` and `/workspace/project/analysis/stage2_encoder_only_v0/baseline_tokens.npz`.

Generated structures (all NPZ arrays are `int64`):

| Array | Shape |
|---|---:|
| `full_generated_tokens` | `(4193,)` |
| `special_token_prefix` / `special_token_suffix` | `(1,)` / `(0,)` |
| `musical_tokens` | `(4192,)` |
| `musical_tokens_8col` | `(524, 8)` |
| `non_pedal_tokens` | `(524, 4)` |
| `pedal_tokens` | `(524, 4)` |

## 6. Validation results

| Check | Result |
|---|---|
| `musical_tokens_8col.reshape(-1) == musical_tokens` | PASS (exact integer equality) |
| `non_pedal_tokens == musical_tokens_8col[:, :4]` | PASS |
| `pedal_tokens == musical_tokens_8col[:, 4:8]` | PASS |
| Concatenated non-pedal/pedal arrays reconstruct `(524,8)` | PASS |
| No unmatched musical token / special-token separation | PASS; BOS prefix `[2]`, empty suffix, musical length divisible by eight |
| MIDI exists, nonzero, and parses with project parser | PASS; 6,620 bytes, one instrument, 524 notes, two deduplicated CC64 events |
| Fixed-seed reproducibility | PASS; second-run tokens and MIDI SHA-256 both matched |

Required future preservation contract: `final_two_stage_tokens[:, :4] == original_stage1_tokens[:, :4]` by exact integer equality. Only columns 4–7 (Pedal1–4) may be replaced.

## 7. Artifact hashes

- `baseline_generated.mid`: `136a546f63ff8006c7013e929a5c527dadef444f347d7b72688d25cf54357e3b`
- `baseline_tokens.npz`: `1cdfab92a27bf079f797e0aa791c37555f2f5d99f160559bb13dd47519cd1988`

## 8. Remaining ambiguities or blockers

- No blocker remains for Experiment 1. The official checkpoint was downloaded by model identifier without an explicit requested revision; the resolved Hugging Face snapshot hash above should be pinned in later reproducibility work.
- The checkpoint config declares BF16 and the official CLI is CPU-hard-coded, whereas the authorized server run required the selected GPU. FP16 was the repository-guided RTX 2080 Ti runtime adaptation; sampling/token grammar and reconstruction behavior were unchanged.
- The bundled ASAP tree is a one-performance miniature, not the complete ASAP dataset. A real Stage 2 dataset build must point to a verified complete performance-MIDI source and define a stable split without silently treating this miniature as the full corpus.

## 9. Next step

In the next task only, add a small dataset module (for example `src/stage2_encoder_only/dataset.py`) plus focused tests. It should load verified ASAP performance MIDI, call the original `normalize_midi`/`midi_to_ids`, require an exact `[N,8]` reshape, copy columns 0–3 unchanged, replace columns 4–7 with MASK ID 1 for the encoder input, and retain the original four IDs (each in `[5261,5389)`) as independent targets. Keep note granularity, the original tokenizer/eight-token grammar, standard unweighted cross-entropy, and a deterministic score-level split. No Stage 2 model or training was implemented here.
