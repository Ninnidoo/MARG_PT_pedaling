# Usage

## Roles and Workflow

- Local project root: `C:\Users\eagle\MARG_research`
- VS Code with Codex is used for code writing, documentation, notebook authoring, and project structure management.
- ChatGPT helps with research direction, methodology, step-by-step planning, and Codex prompt drafting.
- Codex creates and updates local project files and notebooks from those prompts.
- The local VS Code repository is the source of truth for code and documentation.
- GPU inference and future training run in Google Colab.
- Google Drive path `/content/drive/MyDrive/MARG_research` is the persistent Colab storage root.
- Colab `/content` is temporary.
- Important outputs should be saved to Google Drive or copied into local `outputs/`.
- Git commits are done manually by the user from the local Windows terminal.

## Smoke-Test Notebook

Notebook: `notebooks/pianist_transformer_colab_inference.ipynb`

Purpose: preserve the verified reference workflow for one official example MIDI. Use this notebook when checking that the official checkpoint, repository commit, CUDA model loading, and basic inference path still work.

Verified smoke-test result:

- Input MIDI: `/content/PianistTransformer/data/midis/testset/score/3.mid`
- Output MIDI: `/content/output_midi/0.mid`
- Repository commit: `747df2d12291e37f6638b39f1b71517e579ad48c`
- Checkpoint: `yhj137/pianist-transformer-rendering`
- Device: `cuda`
- dtype: `torch.bfloat16`
- Temperature: `1.0`
- Top-p: `0.95`
- Inference time: about 66.26 seconds
- Generated token count: 4192
- Note count: 524
- CC64 event count: 6

Do not casually change this notebook. It is the verified reference workflow.

## Repeated Experiment Notebook

Notebook: `notebooks/02_pianist_transformer_experiments.ipynb`

Purpose: run repeated baseline inference on arbitrary score MIDI inputs and save output MIDI, JSON metadata, and CSV pedal-event tables.

Use this notebook for baseline pedal-behavior experiments after the smoke-test path is known to work.

### Same Colab Session

Within the same live Colab session:

1. Run setup cells once.
2. Keep the model loaded on CUDA.
3. Repeat only the experiment cells:
   - choose/upload input MIDI
   - set seed, temperature, top-p, and sample count
   - run inference
   - analyze pedal events
   - save MIDI/JSON/CSV

### After Runtime Reset

After a Colab runtime reset, disconnection, or new session:

1. Start from setup again.
2. Re-check GPU availability.
3. Mount Google Drive again.
4. Reuse or clone the repository again.
5. Reinstall missing dependencies if necessary.
6. Restore or download the checkpoint.
7. Reload the model to CUDA.
8. Then run experiment cells.

## Storage

Default persistent storage:

`/content/drive/MyDrive/MARG_research`

Expected folders:

- `inputs`
- `outputs/midi`
- `experiment_metadata`
- `checkpoints`

If Drive is disabled, outputs under `/content` are temporary and must be downloaded before the runtime ends.

## Pedal Analysis Caution

The repeated experiment notebook computes exploratory pedal metrics. These metrics are not final research conclusions. In particular, fast off-on transition candidates must not be treated as repedaling labels until the annotation protocol is finalized.

The second generation result with `CC64 event count = 156` is recorded as a future baseline-analysis target only. Its cause is not interpreted yet.
## Research Session Close And Notion Logging

Trigger phrase:

`오케이 오늘 연구 진행상황 총정리해서 노션에 기록해줘`

When closing a research session, use the local Markdown log as the source of truth before publishing to Notion.

Dry-run the summary:

```powershell
python scripts/build_session_summary.py --title "세션 제목" --dry-run
```

Write the local session log:

```powershell
python scripts/build_session_summary.py --title "세션 제목"
```

Dry-run Notion conversion:

```powershell
python scripts/publish_session_to_notion.py docs/session_logs/YYYY-MM-DD_<slug>.md --dry-run
```

Publish to Notion only after the local summary has been reviewed:

```powershell
python scripts/publish_session_to_notion.py docs/session_logs/YYYY-MM-DD_<slug>.md
```

Required local setup for real upload:

- Install `requirements-notion.txt` in the active Python environment.
- Copy `.env.example` to `.env`.
- Fill `NOTION_TOKEN` and `NOTION_PARENT_PAGE_ID` in `.env` only.
- Share the target Notion parent page with the Notion integration.

Do not commit from Codex during this workflow.

## Local Human Pedal-Tokenizer Batch

Purpose: measure information loss from the official Pianist Transformer MIDI
tokenizer on the repository-provided human ground-truth/reference performance
MIDI. This is CPU analysis only and does not run model inference, prediction,
training, or CUDA code.

Verified command:

```powershell
& 'C:\Users\eagle\AppData\Local\Programs\Python\Python311\python.exe' 'analysis\pedal_tokenizer_analysis\analyze_human_batch.py'
```

Verified on 2026-07-27:

- input: `third_party/PianistTransformer/data/midis/testset/human/`
- inventoried: 166 MIDI files
- analyzed: 165
- excluded: `3-1.mid` (`CC64 event count = 0`)
- failed: 0
- CSV output: `outputs/csv/human_*`
- figure output: `outputs/figures/pedal_tokenizer_analysis/human_*`
- full report: `analysis/pedal_tokenizer_analysis/HUMAN_BATCH_REPORT.md`

## Stage 2 Event-Based Pedal Target Audit

Purpose: audit raw repository-provided human performance MIDI for a target with
up to two UP/DOWN transition slots per unique-note-onset IOI and a separate
LIGHT/MEDIUM/DEEP depth label. This CPU analysis does not use tokenizer pedal
tokens, model prediction, training, or CUDA.

Verified test command:

```powershell
& 'C:\Users\eagle\AppData\Local\Programs\Python\Python311\python.exe' -m unittest discover -s tests -p 'test_stage2_pedal_targets.py' -v
```

Verified analysis command:

```powershell
& 'C:\Users\eagle\AppData\Local\Programs\Python\Python311\python.exe' 'scripts\analyze_stage2_pedal_targets.py'
```

Optional CLI paths:

```powershell
& 'C:\Users\eagle\AppData\Local\Programs\Python\Python311\python.exe' 'scripts\analyze_stage2_pedal_targets.py' --input-dir 'path\to\human' --output-dir 'path\to\audit'
```

Verified on 2026-07-28:

- discovered: 166 MIDI files
- analyzed: 165
- excluded: `3-1.mid` (no CC64)
- failed: 0
- synthetic tests: 11 passed
- output: `analysis/stage2_pedal_target_audit/`
- full report: `analysis/stage2_pedal_target_audit/summary.md`

## Structural Bass Blind Annotation v0

Serve the existing 100-candidate, MIDI-only blind annotation page from the
development container (CPU-only), then forward port 8765 in VS Code and open
`http://localhost:8765`:

```bash
docker exec -it ilkyun-marg-pedaling-dev python /workspace/project/scripts/serve_structural_bass_annotation.py --host 0.0.0.0 --port 8765
```

Verified smoke test on 2026-08-25:

```bash
docker exec -e CUDA_VISIBLE_DEVICES='' ilkyun-marg-pedaling-dev python -m unittest tests.test_structural_bass_annotation_v0 -v
```

The persistent result file is
`analysis/structural_bass_annotation_v0/annotation_results.csv`. See
`analysis/structural_bass_annotation_v0/README.md` for shortcuts, export, and
blind/hidden-data separation.
