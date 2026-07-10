# Research Plan

## Objective

Develop a lightweight repedaling refinement system using Pianist Transformer as the baseline expressive-performance renderer.

## Current Stage

Baseline inference reproduction has succeeded once on Google Colab GPU. The next stage is baseline experiment workflow development and pedal behavior analysis.

## Baseline

- Use the official Pianist Transformer repository at commit `747df2d12291e37f6638b39f1b71517e579ad48c`.
- Use the official checkpoint `yhj137/pianist-transformer-rendering`.
- Use Google Colab GPU as the primary execution environment for inference-heavy work.
- Keep the local VS Code repository as the source of truth for code and documentation.
- Record seed, input, model commit, checkpoint, generation parameters, runtime, output path, and pedal metrics for every experiment.

## Next Step: Pedal-Heavy Score Selection

The first smoke-test example produced only 6 CC64 sustain-pedal events, so it is not a strong repedaling-analysis candidate. The next task is to select or prepare score MIDI inputs with richer sustain-pedal behavior or musical contexts where pedal timing matters.

Candidate selection criteria:

- Sufficient note density and phrase length for expressive timing analysis.
- Expected sustain-pedal usage or harmonic texture where pedal changes are musically meaningful.
- Clean piano-focused MIDI structure with interpretable notes and timing.
- A manageable duration for repeated baseline inference samples.

## Baseline Pedal Behavior Analysis

For each generated MIDI, compute and inspect:

- Note count.
- CC64 sustain-pedal event count.
- CC64 event times and values.
- Pedal-on and pedal-off event counts.
- Pedal-on interval count and durations.
- Mean, median, and maximum pedal-on duration.
- Fast off-on transition candidates.

Fast off-on transition candidates are exploratory only. They are not repedaling labels until a formal repedal definition is established.

## Repedaling Refinement

After collecting baseline pedal behavior over several pedal-heavy examples:

1. Define candidate repedaling events more formally.
2. Compare generated CC64 timing against musical context and listening observations.
3. Design a small post-processing or refinement rule set.
4. Evaluate whether the refinement improves editability or musical plausibility.

## Open Questions

- Which score MIDI examples expose the clearest pedal timing issues?
- What threshold should define a meaningful fast off-on repedaling candidate?
- Should analysis use MIDI ticks, milliseconds after tempo mapping, or both?
- How should subjective listening notes be connected to CC64 event metrics?
- What seed and sampling settings provide stable enough baseline comparisons?