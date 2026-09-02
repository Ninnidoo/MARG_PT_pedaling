# Stage 2 Binary Metric Fidelity Audit v0

## Outcome

Pinned official Pianist Transformer pedal evaluator의 MIDI-roundtrip semantics로 ASAP validation 19 pieces만 재평가했다. Existing final lock은 수정하지 않았으며 이 audit 동안 **PROVISIONAL_PENDING_OFFICIAL_MIDI_ROUNDTRIP_AUDIT**로 취급했다. Training/retraining, checkpoint selection 변경, Stage 1 regeneration, calibration, ASAP test MIDI access는 수행하지 않았다.

## Official evaluator provenance

- Pinned PT revision: `747df2d12291e37f6638b39f1b71517e579ad48c`
- Evaluator: `third_party/PianistTransformer/src/evaluate/evaluate.py` SHA-256 `42a6066496c570b974c3288eb3e8ac3b83e91c00add94da2a19b59c575a5e02a`
- Tokenizer: `third_party/PianistTransformer/src/utils/midi.py` SHA-256 `2fb37eaca3d6e4f4a775eb8a59379e7e09651fda36c8146cf0067cde8ad78633`
- Renderer/mapping: `third_party/PianistTransformer/src/model/generate.py` SHA-256 `f4c0de409938cb0576a9076be61ed9bc9a27af140ff1dbb673157d36b4720472`
- `plot_pedal_pattern_distribution`: lines 243–338
- GT MIDI load/tokenize: lines 257–258; prediction MIDI load/tokenize: lines 259–260
- Pedal1–4 extraction and threshold `>=64`: lines 262–284
- 16-bin histogram, epsilon, SciPy base-2 JS distance, Intersection: lines 298–307
- `midi_to_ids` official pedal sample locations: `midi.py` lines 136–182, especially 167–170
- `ids_to_midi`: `midi.py` lines 184–256; `map_midi`: `generate.py` lines 130–237

The pinned function unequivocally sends **both ground truth and prediction through `MIDI file → MidiFile → midi_to_ids`** before extracting Pedal1–4. Therefore pre-render Stage 2 IDs are diagnostic inputs only, not official headline metric inputs.

## Arithmetic fidelity

Official source adds `1e-10` after histogram normalization, calls `scipy.spatial.distance.jensenshannon(..., base=2)`, and computes `sum(min(gt_prob, pred_prob))` without a second normalization. SciPy normalizes its JS inputs internally; the existing vetted helper explicitly normalizes them and is JS-equivalent. The existing helper also normalizes before Intersection, which is not bit-for-bit official.

- Synthetic/actual equivalence cases: 5
- Maximum JS absolute difference: `0.000e+00`
- Maximum official-Intersection reference difference: `0.000e+00`
- SciPy package in existing Docker: unavailable (`ModuleNotFoundError`); no dependency was installed. The pinned call semantics were reproduced with the source-faithful base-2 calculation and cross-checked against the existing vetted helper.

## Existing path versus strict path

- Existing training/diagnosis Stage 2 headline path: cached Stage 1 IDs → Stage 2 pedal replacement → direct token histogram → legacy helper.
- Strict path: cached Stage 1 IDs → Stage 2 pedal replacement → official `ids_to_midi(ref=score_ids)` → official `map_midi(score, performance)` → candidate MIDI dump/reload → official `midi_to_ids` → threshold 64 → histogram → official arithmetic.
- Original PT strict path continues to use the existing cached `original_pt.mid` files; Stage 1 inference was not rerun.

## Available checkpoints

| Candidate | Epoch | Bytes | SHA-256 |
|---|---:|---:|---|
| independent_best | 2 | 413170157 | `4262081a15aa20aac607737e14e4e6b1c8e2727852ac370166f28fbba51d0833` |
| independent_last | 5 | 1206409785 | `a1ad0768ba220baf6b4b69d90bdc2d1755cd89e23acc8c44e253909178664629` |
| joint_best | 3 | 413192697 | `4c3b70f09456cab3c6c8889273982837382229d18c79465465b8b42ba6c24e13` |
| joint_last | 6 | 1206476585 | `72d1676135c8c20aedbbc1f4de7006d87dde6a3b51087eee6172276430b6d462` |

Only `best.pt` and `last.pt` existed in each run directory. No per-epoch checkpoint was reconstructed.

## Direct-token versus strict MIDI-roundtrip metrics

Intersection values in the strict columns use the pinned official no-post-epsilon-renormalization arithmetic. `strict − direct` below compares strict official values against the historical legacy direct metric; the CSV additionally supplies same-arithmetic differences.

| Candidate | Direct notes | Strict notes | Direct JS | Strict JS | ΔJS | Direct Intersection | Strict Intersection | ΔIntersection |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| original_pt | 48894 | 48894 | 0.136886271543 | 0.144081839387 | +0.007195567845 | 0.908513358691 | 0.907531644596 | -0.000981714094 |
| independent_best | 48894 | 48894 | 0.069918887331 | 0.072306914395 | +0.002388027064 | 0.965298332072 | 0.962942038782 | -0.002356293291 |
| independent_last | 48894 | 48894 | 0.089013962745 | 0.097811914057 | +0.008797951311 | 0.913854589892 | 0.910105744516 | -0.003748845376 |
| joint_best | 48894 | 48894 | 0.168750230719 | 0.150083204504 | -0.018667026216 | 0.919373586922 | 0.921791174179 | +0.002417587257 |
| joint_last | 48894 | 48894 | 0.211797524186 | 0.199298206865 | -0.012499317321 | 0.806312679834 | 0.808599581785 | +0.002286901950 |

## Original PT baseline reproduction

- Strict Original PT JS: `0.144081839387225`; existing report: `0.144081839387225` — reproduced within `1e-12`.
- Strict official Intersection: `0.907531644596245`.
- Existing reported Intersection: `0.907531643144195`. It differs only because the legacy helper post-normalized epsilon-smoothed probabilities; pinned official source does not. This is an arithmetic correction, not a histogram mismatch.
- Official-minus-legacy baseline Intersection: `+1.452050701900021e-09`.
- Human strict histogram and Original PT strict histogram exactly match the previously saved MIDI-tokenized histograms.

## Empirical steady / transition-containing mass

These descriptive masses use raw count-normalized empirical distributions (no epsilon).

| Candidate | Direct steady | Direct transition | Strict steady | Strict transition |
|---|---:|---:|---:|---:|
| original_pt | 0.981695096 | 0.018304904 | 0.983658527 | 0.016341473 |
| independent_best | 0.936495275 | 0.063504725 | 0.951834581 | 0.048165419 |
| independent_last | 0.943224117 | 0.056775883 | 0.954227513 | 0.045772487 |
| joint_best | 0.991369084 | 0.008630916 | 0.985949196 | 0.014050804 |
| joint_last | 0.990182845 | 0.009817155 | 0.984701599 | 0.015298401 |

## Strict global 16-pattern distributions

All displayed probabilities are raw empirical `count / total` distributions in order `0000, 0001, ..., 1111`.

| Pattern | Human | Original PT | Independent best | Independent last | Joint best | Joint last |
|---:|---:|---:|---:|---:|---:|---:|
| `0000` | 0.336807219 | 0.429275576 | 0.371047572 | 0.424980570 | 0.414999795 | 0.528203870 |
| `0001` | 0.007991463 | 0.001595288 | 0.006340246 | 0.006810652 | 0.001063525 | 0.001390764 |
| `0010` | 0.000200755 | 0.000122714 | 0.001493026 | 0.001349859 | 0.000061357 | 0.000204524 |
| `0011` | 0.011918515 | 0.003190576 | 0.009735346 | 0.009898965 | 0.003701886 | 0.004070029 |
| `0100` | 0.000376856 | 0.000122714 | 0.001124882 | 0.000797644 | 0.000081810 | 0.000184072 |
| `0101` | 0.000024654 | 0.000020452 | 0.000184072 | 0.000081810 | 0.000040905 | 0.000000000 |
| `0110` | 0.000824153 | 0.000020452 | 0.000859001 | 0.000818096 | 0.000040905 | 0.000061357 |
| `0111` | 0.014781917 | 0.002433836 | 0.005665317 | 0.006442508 | 0.003926862 | 0.003988219 |
| `1000` | 0.010009580 | 0.001493026 | 0.005828936 | 0.005501698 | 0.002883789 | 0.003272385 |
| `1001` | 0.003194472 | 0.000552215 | 0.002106598 | 0.001431669 | 0.000265881 | 0.000204524 |
| `1010` | 0.000049308 | 0.000000000 | 0.000245429 | 0.000143167 | 0.000020452 | 0.000000000 |
| `1011` | 0.005589445 | 0.000449953 | 0.002638361 | 0.001533931 | 0.000429501 | 0.000245429 |
| `1100` | 0.009960272 | 0.002761075 | 0.005828936 | 0.005603960 | 0.000818096 | 0.000981716 |
| `1101` | 0.001535601 | 0.001288502 | 0.001922526 | 0.001104430 | 0.000634025 | 0.000429501 |
| `1110` | 0.010566059 | 0.002290670 | 0.004192743 | 0.004254101 | 0.000081810 | 0.000265881 |
| `1111` | 0.586169733 | 0.554382951 | 0.580787009 | 0.529246942 | 0.570949401 | 0.456497730 |

## MIDI-level pedal-only preservation

- Piece-checkpoint comparisons: **76/76 PASS**
- Exact assertions: note count/order, pitch, onset, offset/duration, velocity, ticks-per-beat, instrument layout, tempo changes, time/key signatures, markers, lyrics, and all non-CC64 controls.
- CC64 events were the only allowed MIDI-level difference.
- Any non-pedal mismatch would have aborted before metric aggregation.

## Selection impact assessment (no selection performed)

Historical direct-token order among available checkpoints: `independent_best < independent_last < joint_best < joint_last` (lower JS first).
Strict MIDI-roundtrip order among available checkpoints: `independent_best < independent_last < joint_best < joint_last` (lower JS first).
The observed ranking is unchanged, so the available checkpoint set shows no selection flip under strict semantics. Nevertheless, the headline values move materially and the prior lock was based on direct-token metrics. This audit does **not** select a checkpoint or modify the existing lock; any lock update must be handled in a separate explicitly authorized step.

## Integrity and scope

- Existing final lock SHA-256 before/after audit: `f0f6de66e4e6d15b63395ac1b3c5ee68d4b08fe522fc05ca949051be14f5fb28` / `f0f6de66e4e6d15b63395ac1b3c5ee68d4b08fe522fc05ca949051be14f5fb28` — unchanged
- Human validation MIDI loads: 71 (71 expected)
- Validation score MIDI loads: 95
- Cached Original PT MIDI loads: 95
- Stage 1 neural inference regeneration: 0
- Training/retraining: 0
- Checkpoint reconstruction/update: 0
- ASAP test MIDI access: **0 / PASS**
- Existing final lock status for this report: **PROVISIONAL_PENDING_OFFICIAL_MIDI_ROUNDTRIP_AUDIT**

## Artifacts

- `checkpoint_metric_comparison.csv`
- `strict_global_joint16_comparison.csv`
- `midi_nonpedal_equality.csv`
- `arithmetic_equivalence.json`
- `audit_summary.json`
- `candidate_midis/<checkpoint_label>/<piece_id>.mid`

## Stop point

Metric-fidelity audit only. No retraining, calibration, checkpoint selection change, Stage 1 regeneration, or ASAP test evaluation was performed.
