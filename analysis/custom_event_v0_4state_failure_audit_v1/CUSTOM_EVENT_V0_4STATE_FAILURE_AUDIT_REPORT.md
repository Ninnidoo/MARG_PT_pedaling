# Custom Event v0 — 4-State / Depth-State Failure Audit v1

## 1. Frozen provenance / universe

All requested IDs and checkpoint epochs passed exact assertions. ASAP validation is 71 performances / 19 pieces; the frozen successful universe is 70 pairs / 272,053 aligned notes / 1,088,212 Pedal1–4 samples. The sole exclusion is `Liszt/Mephisto_Waltz/Tysman07M.mid`.

## 2. Executive finding

Custom's strong threshold-transition score coexists with weak 4-state fidelity because a large share of its sample errors remains on the correct OFF/ON side but selects the wrong depth state. Binary collapse recovers substantially more accuracy for Custom than for Hybrid. The absolute-SET decoder also suppresses 28,586 repeated same-state destinations, while mismatch persistence and Oracle-Initial quantify how an incorrect state can survive across later samples. Slot5/6 overprediction is visible in raw heads but contributes only 10.20% of effective Main changes.

## 3. State occupancy + confusion

Human ratios: {'ZERO': 0.27179354758080226, 'LOW': 0.10679996177215469, 'HALF': 0.14049468302132306, 'FULL': 0.48091180762572}

Custom ratios: {'ZERO': 0.06970792455881758, 'LOW': 0.2332541820895193, 'HALF': 0.328490220655534, 'FULL': 0.36854767269612904}

Hybrid ratios: {'ZERO': 0.22636306160931877, 'LOW': 0.11674655306135201, 'HALF': 0.17002293670718574, 'FULL': 0.4868674486221435}

Full counts, row-normalized confusion, precision/recall/F1, and explicit ZERO↔LOW / HALF↔FULL / OFF↔ON confusions are in the JSON/CSV artifacts.

## 4. Same-side vs cross-threshold sample errors

Custom: 311,028 same-side (44.00%) vs 395,837 cross-threshold (56.00%). Hybrid: 36.04% vs 63.96%.

## 5. Binary collapse

Custom binary accuracy 0.636250, recovery +0.285816; Hybrid binary accuracy 0.631362, recovery +0.207718.

## 6. Effective state-change taxonomy

Human: {'TYPE_A': {'count': 20374, 'ratio': 0.25040866241411947}, 'TYPE_B': {'count': 26421, 'ratio': 0.32472991408871354}, 'TYPE_C': {'count': 34568, 'ratio': 0.424861423497167}}

Custom: {'TYPE_A': {'count': 4739, 'ratio': 0.08324404082277925}, 'TYPE_B': {'count': 16365, 'ratio': 0.2874633315182069}, 'TYPE_C': {'count': 35825, 'ratio': 0.6292926276590138}}

Hybrid: {'TYPE_A': {'count': 18635, 'ratio': 0.24607481942188594}, 'TYPE_B': {'count': 29648, 'ratio': 0.3915012742806587}, 'TYPE_C': {'count': 27446, 'ratio': 0.3624239062974554}}

The secondary type-specific matcher is diagnostic only and does not replace the canonical Transition metric.

## 7. Initial-state audit + Oracle-Initial

Initial accuracy 0.549296, Macro F1 0.347235. Oracle-Initial deltas: {'four_class_accuracy': 0.001239648156793005, 'macro_f1': 0.0007026398686458268, 'transition_precision': 0.00015167821866290954, 'transition_recall': 2.8928488775670225e-05, 'transition_f1': 9.251704559276597e-05, 'js_divergence_base2': 0.0009666926869897619, 'intersection': -0.0004557935285609549}. Ground-truth Initial is diagnostic only and is not deployable performance.

## 8. Main Slot1–6 audit

Raw distributions, 5×5 confusions, active rates, class metrics, frozen weights, and validation distortion ratios are saved. Slot5/6 raw non-NONE=10,130; quantized effective changes=5,921/58,061; threshold crossings=2,017/36,419. Secondary one-to-one +/-1 onset matching attributes 994/18,623 (5.34%) unmatched Main threshold crossings to Slot5/6.

## 9. Redundant SET audit

There are exactly 28,586 continuous-decoder same-state suppressions (32.94% of prefix-active events). State/slot concentrations are in `redundant_set_analysis.json`.

## 10. Timing conditional analysis

Correct-class active targets: {'count': 17871, 'tau_mae': 0.19354501366615295, 'tau_rmse': 0.2537086606025696}; incorrect-class active targets: {'count': 64802, 'tau_mae': 0.2123008817434311, 'tau_rmse': 0.26232215762138367}. This separates destination classification from timing regression without changing a tolerance.

## 11. Residence / error persistence

Custom mismatch p95=2.067125s; Hybrid p95=1.640438s; ratio=1.260. Quantile summaries, longest-episode concentration, and observable start causes are saved. Seconds use the canonical normalized aligned timing grid; no new sampling universe was introduced.

## 12. 256-pattern distortion

Custom JS/intersection={'js_divergence_base2': 0.12891866867254204, 'intersection': 0.6463152490994679}; Hybrid={'js_divergence_base2': 0.015306665074222, 'intersection': 0.9264812389939665}. Top absolute-error patterns: ['HALF/HALF/HALF/HALF', 'ZERO/ZERO/ZERO/ZERO', 'LOW/LOW/LOW/LOW', 'FULL/FULL/FULL/FULL', 'HALF/FULL/FULL/FULL', 'LOW/ZERO/ZERO/ZERO', 'LOW/HALF/HALF/HALF', 'FULL/FULL/FULL/HALF', 'ZERO/ZERO/ZERO/LOW', 'FULL/FULL/LOW/LOW'].

## 13. 16-pattern binary collapse

Custom JS/intersection={'js_divergence_base2': 0.010206632131425055, 'intersection': 0.9059117158824024}; Hybrid={'js_divergence_base2': 0.004282929823911581, 'intersection': 0.9479880759360859}.

## 14. Performance/piece consistency

Custom 4C accuracy is higher on 22/70 performances and lower/equal on 48/70; Custom binary accuracy is higher on 32/70. Piece aggregates are in `piece_failure_summary.csv`.

## 15. Terminal contribution

Predicted active terminal slots=107, negative-z clamps=53, strict-tick guards=53, same-state suppressions=8. Exact main-only counterfactual alignment attributes 60 primary samples to Terminal; this is too small to explain the main failure.

## 16. H1–H5 verdicts

{
  "H1_same_side_depth_modeling_failure": {
    "verdict": "SUPPORTED",
    "statistics": {
      "custom_same_side_error_percent": 44.00104687599471,
      "custom_binary_recovery": 0.28581563151297723,
      "hybrid_binary_recovery": 0.20771779763501963
    }
  },
  "H2_absolute_set_error_persistence": {
    "verdict": "SUPPORTED",
    "statistics": {
      "custom_p95_seconds": 2.0671249999999923,
      "hybrid_p95_seconds": 1.6404375000000264,
      "ratio": 1.2601059168666646
    }
  },
  "H3_slot5_6_not_primary": {
    "verdict": "SUPPORTED",
    "statistics": {
      "raw_non_none": 10130,
      "effective_changes": 5921,
      "effective_share": 0.10197895316994196,
      "threshold_crossings": 2017,
      "threshold_share": 0.05538317910980532,
      "unmatched_threshold_crossings": 994,
      "all_unmatched_threshold_crossings": 18623,
      "unmatched_threshold_share": 0.05337485904526661
    }
  },
  "H4_redundant_same_state_set": {
    "verdict": "SUPPORTED",
    "statistics": {
      "redundant_sets": 28586,
      "prefix_active": 86793,
      "ratio": 0.32935835839295796
    }
  },
  "H5_initial_state_error_persistence": {
    "verdict": "PARTIALLY SUPPORTED",
    "statistics": {
      "initial_accuracy": 0.5492957746478874,
      "oracle_4c_accuracy_delta": 0.001239648156793005,
      "oracle_transition_f1_delta": 9.251704559276597e-05,
      "oracle_js_delta": 0.0009666926869897619
    }
  }
}

## 17. Single recommended next experiment

**Candidate B: State-conditioned event destination head**. See `next_experiment_recommendation.md` for the evidence summary.

## 18. Execution boundary

No training / no tuning / no checkpoint reselection / no ASAP test / no Repedal. Epochs 2–4 were not evaluated. Decoder v1 and the frozen evaluator were not modified.
