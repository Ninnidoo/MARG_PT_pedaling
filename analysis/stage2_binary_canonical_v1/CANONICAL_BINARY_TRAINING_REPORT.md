# Canonical Binary Stage 2 Training Report

## A. Experiment setup

This controlled run retrains both existing binary hypotheses from the same official pretrained Pianist Transformer encoder. The training pool, natural concatenation, unweighted losses, optimizer, seed, and budget are unchanged from the validated binary configuration.

- Training data: MAESTRO-clean 1,170 + ASAP train 892 = 2,062 performances / 9,369,095 notes / 35,573 windows
- Shared cache ID: `85a79e9d10e955b10000f72f6bbc4a29dbd2cc5a4c8cfb1277054a8d45dcb877`
- Seed: 42; window/stride: 512/256; batch/effective batch: 16/16
- AdamW: encoder LR 1e-5, head LR 1e-4, weight decay 0.01; gradient clip 1.0; AMP enabled
- Maximum 10 epochs; canonical strict-JS early stopping patience 3
- Frozen canonical Stage 1 manifest: `/workspace/project/analysis/stage2_binary_canonical_v1/canonical_validation_stage1_manifest.csv`
- Original PT baseline: JS `0.161498978463802`, Intersection `0.821025730937020`
- Stage 2 input: `[Pitch, IOI, Velocity, Duration, MASK, MASK, MASK, MASK]`
- Final validation candidates are canonical MIDI plus transplanted CC64 only; all non-CC64 raw MIDI/meta events pass exact equality before metrics.

## B. Independent 4×2 trajectory

| Epoch | Train loss | Human-val loss | Human binary acc | Human exact-pattern acc | Canonical JS ↓ | Intersection ↑ | ΔJS | Relative ΔJS | ΔIntersection | Beats PT? |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.417203455 | 0.420215057 | 0.802530547 | 0.732834966 | 0.211183944768 | 0.758456104146 | +0.049684966304 | +30.764880% | -0.062569626791 | no |
| 2 | 0.354823133 | 0.437378874 | 0.801151764 | 0.736072661 | 0.150082839377 | 0.829325144704 | -0.011416139086 | -7.068861% | +0.008299413767 | yes |
| 3 | 0.325545862 | 0.423863721 | 0.806503834 | 0.741094982 | 0.184957465921 | 0.787206737300 | +0.023458487457 | +14.525471% | -0.033818993637 | no |
| 4 | 0.300488840 | 0.465834169 | 0.802157768 | 0.740814152 | 0.153075822363 | 0.825739860651 | -0.008423156101 | -5.215610% | +0.004714129714 | yes |
| 5 | 0.276554567 | 0.478120165 | 0.803671531 | 0.743357926 | 0.174399799616 | 0.799601149588 | +0.012900821153 | +7.988175% | -0.021424581349 | no |

## C. Joint 16 trajectory

| Epoch | Train loss | Human-val loss | Human binary acc | Human exact-pattern acc | Canonical JS ↓ | Intersection ↑ | ΔJS | Relative ΔJS | ΔIntersection | Beats PT? |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.739575902 | 0.678049305 | 0.797834350 | 0.756562355 | 0.211309269897 | 0.769697113233 | +0.049810291433 | +30.842481% | -0.051328617704 | no |
| 2 | 0.624691006 | 0.665368636 | 0.806821805 | 0.765340547 | 0.202442336778 | 0.780598469256 | +0.040943358314 | +25.352085% | -0.040427261681 | no |
| 3 | 0.586426087 | 0.648616876 | 0.812497735 | 0.770699864 | 0.207687753564 | 0.770801565814 | +0.046188775101 | +28.600042% | -0.050224165123 | no |
| 4 | 0.555098545 | 0.665722816 | 0.814102088 | 0.772647553 | 0.217424845886 | 0.759139682111 | +0.055925867422 | +34.629239% | -0.061886048826 | no |
| 5 | 0.525870705 | 0.696162648 | 0.811085887 | 0.769348258 | 0.207831892984 | 0.772127226340 | +0.046332914520 | +28.689293% | -0.048898504597 | no |

## D. Best validation result

| Model | Best epoch | Strict JS ↓ | Intersection ↑ | Beats Original PT? |
|---|---:|---:|---:|---|
| Original PT | — | 0.161498978464 | 0.821025730937 | — |
| Independent 4×2 | 2 | 0.150082839377 | 0.829325144704 | yes |
| Joint 16 | 2 | 0.202442336778 | 0.780598469256 | no |

## E. Locked checkpoint hashes

- Independent 4×2 `best.pt`: `9fc60f9b3d91942afaf56f0553fea3f2c39509f32c9965d6236de178077d65d4`
- Independent 4×2 `last.pt`: `b193b11d3321ff9f74242967c047de3913d5e5c513c6ff6f08d8eaa767baf270`
- Per-epoch and alias hashes: `/workspace/project/analysis/stage2_binary_canonical_v1/train_v0/independent_4x2/checkpoint_manifest.csv`
- Joint 16 `best.pt`: `4d4d009b0ce9dae44c88dbb1c6e80883295146bbcd6daca3421954ff5ebdacb6`
- Joint 16 `last.pt`: `558573941f136b0394ac099d2e3f444969122bbf4e58765e29793bb0d845cc0d`
- Per-epoch and alias hashes: `/workspace/project/analysis/stage2_binary_canonical_v1/train_v0/joint_16/checkpoint_manifest.csv`

## F. Test isolation

ASAP test was not used for training, model selection, checkpoint selection, calibration, or evaluation in this stage.

- ASAP test human MIDI access: 0
- Canonical test-bank evaluation: not performed
- Test JS / Intersection: not computed
- Test audio: not generated
- Stage 1 validation neural inference regeneration: 0
