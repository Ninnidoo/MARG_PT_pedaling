# Final Test Dual-Reference Canonical Pedal Evaluation

## Table 1 — Set A: ASAP canonical 104

| System | 4C Acc ↑ | Macro F1 ↑ | JS ↓ | Intersection ↑ | Trans P ↑ | Trans R ↑ | Trans F1 ↑ | Repedal P ↑ | Repedal R ↑ | Repedal F1 ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Human–Human | 0.560947 | 0.437879 | 0.015998 | 0.875330 | 0.651841 | 0.685348 | 0.668175 | 0.574745 | 0.583380 | 0.579030 |
| Original PT | 0.538575 | 0.311073 | 0.203045 | 0.667818 | 0.681166 | 0.395986 | 0.500825 | 0.542927 | 0.320759 | 0.403269 |
| RUN A | 0.369325 | 0.212466 | 0.223627 | 0.667869 | 0.589131 | 0.539938 | 0.563463 | 0.459586 | 0.313157 | 0.372498 |
| RUN B | 0.547677 | 0.317215 | 0.233886 | 0.637493 | 0.651443 | 0.552095 | 0.597668 | 0.556289 | 0.450235 | 0.497675 |
| Weighted CE | 0.489807 | 0.397504 | 0.067969 | 0.751206 | 0.399968 | 0.570899 | 0.470386 | 0.325184 | 0.467007 | 0.383400 |
| CE + NTL-WAS | 0.528024 | 0.381797 | 0.083262 | 0.718341 | 0.407905 | 0.562739 | 0.472973 | 0.329226 | 0.466717 | 0.386096 |
| Huber + Auxiliary CE | 0.467535 | 0.377608 | 0.066954 | 0.749750 | 0.516484 | 0.457784 | 0.485365 | 0.432182 | 0.338402 | 0.379585 |

## Table 2 — Set B: PT Human 166

| System | 4C Acc ↑ | Macro F1 ↑ | JS ↓ | Intersection ↑ | Trans P ↑ | Trans R ↑ | Trans F1 ↑ | Repedal P ↑ | Repedal R ↑ | Repedal F1 ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Human–Human | 0.603001 | 0.487994 | 0.014726 | 0.879117 | 0.657436 | 0.670260 | 0.663786 | 0.593217 | 0.564455 | 0.578479 |
| Original PT | 0.543145 | 0.312976 | 0.197778 | 0.678848 | 0.655591 | 0.357544 | 0.462727 | 0.503447 | 0.271489 | 0.352753 |
| RUN A | 0.378017 | 0.215594 | 0.213363 | 0.679133 | 0.555512 | 0.520197 | 0.537275 | 0.414182 | 0.295012 | 0.344584 |
| RUN B | 0.544528 | 0.314761 | 0.226358 | 0.637541 | 0.610806 | 0.506971 | 0.554066 | 0.498289 | 0.391276 | 0.438345 |
| Weighted CE | 0.480503 | 0.386276 | 0.065869 | 0.749040 | 0.358075 | 0.529579 | 0.427259 | 0.271079 | 0.420504 | 0.329649 |
| CE + NTL-WAS | 0.522184 | 0.374448 | 0.083539 | 0.710680 | 0.363739 | 0.524437 | 0.429551 | 0.273566 | 0.419105 | 0.331046 |
| Huber + Auxiliary CE | 0.467599 | 0.376411 | 0.065584 | 0.752487 | 0.462843 | 0.429421 | 0.445506 | 0.357285 | 0.303847 | 0.328406 |

## Reference inventory / pair accounting

- Set A: 23 pieces / 104 Human performances from `asap_split.csv` test rows; never pooled with Set B.
- Set B: 23 pieces / 166 bundled PT Human MIDI; numeric prefix mapping matched the candidate universe 23/23; never pooled with Set A.
- setA: six-system common pairs 104/104; Human–Human pairs 81/81; alignment/feature failures 0.
- setB: six-system common pairs 166/166; Human–Human pairs 143/143; alignment/feature failures 0.

## Human–Human representative rule

Within each reference universe and piece, `(performance identifier, path)` was sorted lexicographically before any alignment or metric computation. The first entry was frozen as H0 and compared with every other Human. H0↔H0 self-comparisons are zero. The two mappings are stored separately.

## Alignment failures

Failures are recorded separately for Set A and Set B. Any unavailable Human comparison unit or candidate piece was excluded from all six model rows of that reference set. Human–Human retains its separate fixed-H0 pair inventory.

## RUN A EOT identity provenance

RUN A is the final `stage2_binary_2slot_D_pre_main_post_full_v1` PRE/MAIN/POST candidate. Prior inference verification recorded 13/23 strict identities and 10/23 frozen terminal/EOT-extension-only identities. This evaluation independently confirmed exact note count, pitch, onset, offset, and velocity identity for 23/23 before parsing pedal metrics.

## Canonical artifacts

- `summary_setA_asap104.csv`, `summary_setB_pt166.csv`
- `per_piece_setA_asap104.csv`, `per_piece_setB_pt166.csv` (23 × 7 rows each)
- `reference_set_sensitivity.csv` uses Set B minus Set A.
- Confusion, Transition, Repedal, and 256-pattern diagnostics are under their respective subdirectories.
