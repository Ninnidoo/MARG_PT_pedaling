# Pedal Tokenizer Analysis

This folder contains isolated analysis code for measuring how much raw
performance-MIDI sustain pedal information is retained by the existing Pianist
Transformer tokenizer. It does not modify files under
`third_party/PianistTransformer/`, does not train a model, and does not run
model inference.

## Verified tokenizer source

Tokenizer implementation:
- file: `third_party/PianistTransformer/src/utils/midi.py`
- function: `midi_to_ids(config, midi_obj, normalize=True)`, lines 136-182
- token layout per note: `Pitch, IOI, Velocity, Duration, Pedal1, Pedal2, Pedal3, Pedal4`

Pedal extraction:
- file: `third_party/PianistTransformer/src/utils/midi.py`
- function: `normalize_midi()`, lines 10-89
- behavior: non-drum instruments are merged; only controller 64 is retained as
  pedal data; event times are converted through the original MIDI tempo map into
  a normalized 120 BPM, 500 ticks/beat coordinate.

Pedal sampling:
- file: `third_party/PianistTransformer/src/utils/midi.py`
- function: `midi_to_ids()`, lines 136-182
- exact rule: `get_pedal()` uses `bisect_right` over normalized CC64 event
  ticks and returns the latest prior value, or 0 if no prior event exists.
- Pedal1 is sampled at the current note onset.
- Pedal2, Pedal3, and Pedal4 are sampled at 1/4, 2/4, and 3/4 of the IOI to
  the next note, respectively.
- If the next IOI is 0, the four pedal samples for that note are all read from
  the same timestamp.

Pedal reconstruction:
- file: `third_party/PianistTransformer/src/utils/midi.py`
- function: `ids_to_midi(config, ids, ...)`, lines 184-256
- available: yes
- method: Pedal1-4 token values are converted back to CC64 events at the same
  per-note positions, then sorted by time. Consecutive same-valued CC64 events
  are removed.

Important implementation details:
- `normalize_midi()` performs tempo normalization before tokenization:
  120 BPM, 500 ticks/beat, therefore 1000 ticks/second.
- CC64 values 1-126 are representable because pedal token IDs are
  `config.pedal_start + value`, with valid range 5261-5388.
- The comparison in this analysis is done in the official normalized coordinate
  so raw and tokenized curves share the same time axis.
- The script calls the official `midi_to_ids`, `ids_to_midi`, and
  `normalize_midi` functions. If importing the full model config is blocked by
  missing model dependencies, it uses a tokenizer-only config object mirroring
  `PianoT5GemmaConfig` constants from `src/model/pianoformer.py` lines 46-66.

## Local MIDI candidates

A local scan found one ASAP performance entry:

- `third_party/PianistTransformer/data/midis/asap-dataset-master/Bach/Fugue/bwv_846/Shi05M.mid`
- metadata source: `asap-dataset-master/metadata.csv`, column `midi_performance`
- scanner result: 754 note-on events, 2428 CC64 events, 2297 intermediate CC64
  events, ticks/beat 384

The bundled ASAP score MIDI is not a performance candidate:

- `third_party/PianistTransformer/data/midis/asap-dataset-master/Bach/Fugue/bwv_846/midi_score.mid`
- scanner result: 0 CC64 events

The repository also contains
`third_party/PianistTransformer/data/midis/testset/performance/*.mid`. These are
usable fallback candidates, but the local scanner showed their CC64 streams are
mostly binary 0/127 and often exactly four CC64 events per note. Treat them as
secondary examples, not as evidence of raw half-pedal preservation.

No internet download is performed. To analyze a fuller ASAP dataset, place the
dataset under:

`third_party/PianistTransformer/data/midis/asap-dataset-master/`

with the standard `metadata.csv` and relative `midi_performance` paths.

## Outputs

The script writes:

- `outputs/csv/midi_candidates.csv`
- `outputs/csv/summary_per_file.csv`
- `outputs/csv/repedal_events.csv`
- `outputs/csv/tokenized_notes.csv`
- `outputs/csv/cc64_events.csv`
- `outputs/csv/ioi_bins.csv`
- `outputs/csv/note_density_bins.csv`
- `outputs/csv/transition_matches.csv`
- `outputs/csv/analysis_run_metadata.json`
- `outputs/figures/*.png`

## Operational definitions

The repedaling detector is an analysis convenience, not a musicological label.
Defaults:

- high threshold: 96
- low threshold: 31
- maximum release-to-redepress duration: 300 ms
- raw/token repedal match tolerance: 150 ms

The partial-pedal occupancy thresholds are also operational:

- released: 0-15
- partial: 16-111
- deep/full: 112-127

## Intended usage

The local analysis workflow is verified with:

`C:\Users\eagle\AppData\Local\Programs\Python\Python311\python.exe`

Use that executable explicitly for the commands below.

Pilot, one selected file:

```powershell
python analysis/pedal_tokenizer_analysis/analyze_pedal_tokenizer.py --pilot
```

Batch, up to 10 selected files:

```powershell
python analysis/pedal_tokenizer_analysis/analyze_pedal_tokenizer.py --max-files 10
```

ASAP only:

```powershell
python analysis/pedal_tokenizer_analysis/analyze_pedal_tokenizer.py --asap-only
```

Explicit MIDI file:

```powershell
python analysis/pedal_tokenizer_analysis/analyze_pedal_tokenizer.py --midi path/to/performance.mid
```

If figures fail because `matplotlib` is unavailable, rerun with
`--skip-figures` to generate CSV outputs only, then add plotting dependencies to
the isolated project environment after approval.

## Verified human-reference batch

The repository-provided human ground-truth/reference performance set was
analyzed with:

```powershell
& 'C:\Users\eagle\AppData\Local\Programs\Python\Python311\python.exe' 'analysis\pedal_tokenizer_analysis\analyze_human_batch.py'
```

Verified result on 2026-07-27:

- inventory: 166 MIDI files
- analyzed: 165
- excluded: `3-1.mid` because it has no CC64 events
- per-file failures: 0
- model inference, prediction, and training: not run

The human script uses valley/minimum-time local IOI attribution, ordered
one-to-one repedal and transition matching, file-level bootstrap intervals, and
per-file exception isolation. Detailed methods and results are in
`HUMAN_BATCH_REPORT.md`.
