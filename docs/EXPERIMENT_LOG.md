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
| 2026-07-27 | Human reference pedal-tokenizer information-loss batch | `C:\Users\eagle\AppData\Local\Programs\Python\Python311\python.exe analysis\pedal_tokenizer_analysis\analyze_human_batch.py` | Success | Inventoried 166 repository-provided human reference MIDI files, excluded only `3-1.mid` for no CC64, and analyzed 165/165 with no per-file failures. No model prediction or training. |
| 2026-07-28 | Stage 2 event-based pedal target audit | `C:\Users\eagle\AppData\Local\Programs\Python\Python311\python.exe scripts\analyze_stage2_pedal_targets.py` | Success | Analyzed raw CC64 in 165/166 human MIDI files with tempo-aware IOIs, two-slot coverage, tau, time-weighted depth, crossing-noise, and hysteresis sensitivity. No tokenizer, prediction, training, GPU, or new package installation. |

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

## Human Reference Pedal-Tokenizer Batch

Date: 2026-07-27

### Purpose

Measure pedal depth, repedal gesture, and transition timing information retained
by the official Pianist Transformer tokenizer on repository-provided human
ground-truth/reference performance MIDI.

### Verified Execution

```powershell
& 'C:\Users\eagle\AppData\Local\Programs\Python\Python311\python.exe' 'analysis\pedal_tokenizer_analysis\analyze_human_batch.py'
```

- Input files inventoried: 166
- Files analyzed: 165
- Excluded: `3-1.mid`, no CC64 events
- Per-file failures: 0
- Model inference, prediction, and training: not run
- Random seed: 20260727
- Bootstrap iterations: 2000
- Comparison grid: 10 ms

### Key Results

- Raw CC64 events: 659,048
- Raw repedal candidates: 15,549
- Preserved repedals: 13,256
- Lost repedals: 2,293
- Micro repedal recall: 0.8525
- Macro repedal recall: 0.8288
- Mean/median CC64 NMAE: 0.0511 / 0.0419
- Matched/unmatched raw transitions: 76,630 / 4,272
- Transition median/p90/p95 absolute error: 16 / 92 / 139 ms
- IOI-bin repedal recall decreased from 0.9653 below 100 ms to 0.0731
  at or above 1000 ms.

### Artifacts

- Full report: `analysis/pedal_tokenizer_analysis/HUMAN_BATCH_REPORT.md`
- CSV: `outputs/csv/human_*`
- Figures: `outputs/figures/pedal_tokenizer_analysis/human_*`
- Validation: `outputs/csv/human_validation_report.json`

## Stage 2 Event-Based Pedal Target Audit

Date: 2026-07-28

### Purpose

Measure whether at most two UP/DOWN transition slots per unique-note-onset IOI
can represent raw human sustain-pedal behavior, and audit event-time, ON-depth,
crossing, and hysteresis label choices before model implementation.

### Verified Execution

```powershell
& 'C:\Users\eagle\AppData\Local\Programs\Python\Python311\python.exe' -m unittest discover -s tests -p 'test_stage2_pedal_targets.py' -v
& 'C:\Users\eagle\AppData\Local\Programs\Python\Python311\python.exe' 'scripts\analyze_stage2_pedal_targets.py'
```

- Python: 3.11.2
- mido / numpy / matplotlib: 1.3.3 / 1.26.4 / 3.8.4
- New packages installed: none
- Synthetic tests: 11 passed
- Input files: 166
- Analyzed: 165
- Excluded: `3-1.mid`, no CC64
- Parsing/analysis failures: 0
- Raw CC64: 659,048, all on channel 0
- Tokenizer, model prediction, training, GPU/CUDA: not used

### Key Results

- Unique-onset IOIs: 562,605
- IOIs with 0 / 1 / 2 / 3 / 4+ transitions:
  494,743 / 56,019 / 11,146 / 501 / 196
- Two-slot lossless coverage: 0.998761
- Overflow IOIs: 697
- Main-IOI UP / DOWN transitions: 40,289 / 40,309
- Combined slot NONE / UP / DOWN:
  1,045,505 / 39,722 / 39,983
- All-event median tau: 0.3914
- IOIs ever pedal-ON: 60.376%
- Time-weighted ON-depth mean / median: 112.772 / 127
- Equal-width depth class IOI proportions:
  LIGHT 20.922%, MEDIUM 10.467%, DEEP 68.612%
- Time-weighted quantile values: 122 and 127
- CC64=127 ON-time share: 64.691%; the upper quantile tie leaves the
  strict value-threshold DEEP quantile class empty
- Baseline rapid reversals within 25 / 50 / 100 / 200 ms:
  633 / 2,268 / 12,617 / 28,000
- Baseline / narrow / wide full-timeline transition totals:
  80,902 / 76,877 / 72,841
- Narrow / wide main-IOI net transition reductions:
  4,015 / 8,027
- Tau outside `[0,1)`, nonpositive IOIs, invalid CC64 values:
  0 / 0 / 0

### Audit Warnings

- One same-timestamp opposite transition occurs in `17-12.mid` at tick
  127518: CC64 127 to 39 (UP), then 39 to 79 (DOWN).
- Five IOIs exceed the transparent 5-second long-IOI warning threshold.
- CC64 events before the first onset and at/after the last onset are recorded
  separately and excluded from main IOI targets.
- Rapid reversals remain candidates, not noise labels; they can include genuine
  repedaling.

### Artifacts

- Script: `scripts/analyze_stage2_pedal_targets.py`
- Tests: `tests/test_stage2_pedal_targets.py`
- Report: `analysis/stage2_pedal_target_audit/summary.md`
- Machine-readable aggregate:
  `analysis/stage2_pedal_target_audit/aggregate_summary.json`
- CSV tables: `analysis/stage2_pedal_target_audit/*.csv`
- Figures: `analysis/stage2_pedal_target_audit/figures/*.png`
