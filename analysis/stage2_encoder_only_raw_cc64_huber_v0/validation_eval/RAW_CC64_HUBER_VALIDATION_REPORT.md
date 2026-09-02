# Raw CC64 Huber Canonical Validation Report

- ASAP validation only; ASAP test access count: 0
- Frozen 4-class boundaries: ZERO 0–25, LOW 26–63, HALF 64–103, FULL 104–127
- Transition: CC64 64 crossing, direction-aware one-to-one, tolerance ±1 distinct onset
- Repedal metric execution count: 0
- Common Representative-Huber universe: 70 pairs / 272053 aligned notes

## Main metrics

| 4C Acc | Macro F1 | Transition P | Transition R | Transition F1 | JS | Intersection |
|---:|---:|---:|---:|---:|---:|---:|
| 0.492320 | 0.403413 | 0.497066 | 0.512063 | 0.504453 | 0.038859 | 0.825369 |

## Regression diagnostics

- Raw CC64 MAE/RMSE: 25.603756 / 40.857175
- Prediction mean/std: 82.597224 / 47.004309
- Target mean/std: 78.501645 / 53.099105
- Prediction quantiles: `{'p01': 0.1451498126983643, 'p05': 4.06951904296875, 'p25': 35.439697265625, 'p50': 106.97021484375, 'p75': 124.0234375, 'p95': 127.0, 'p99': 127.0}`
- Pre-clip min/max: -20.324341 / 157.881836
- Clip-at-0 / clip-at-127: 0.009257 / 0.111327
- Absolute error by target range: `{'FULL': {'count': 1075802, 'mae': 14.541138092553117, 'range': [104, 127], 'rmse': 30.08252821833538}, 'HALF': {'count': 318161, 'mae': 31.861950313202367, 'range': [64, 103], 'rmse': 37.93018114730209}, 'LOW': {'count': 232769, 'mae': 34.0362697673467, 'range': [26, 63], 'rmse': 40.10778021802586}, 'ZERO': {'count': 581012, 'mae': 39.28203525248183, 'range': [0, 25], 'rmse': 56.87854969790546}}`

## Classification diagnostics

- Prediction distribution: `[0.3074327428846585, 0.1649402873704756, 0.18391728817546582, 0.34370968156940007]`
- Class P/R/F1: `[{'class_id': 0, 'f1': 0.5562975055566924, 'precision': 0.5240530620053085, 'recall': 0.5927700333706373, 'support': 295769}, {'class_id': 1, 'f1': 0.16607430903821638, 'precision': 0.13680427878990473, 'recall': 0.21127851248913707, 'support': 116221}, {'class_id': 2, 'f1': 0.23526112585651604, 'precision': 0.20748872045208128, 'recall': 0.27161713149495054, 'support': 152888}, {'class_id': 3, 'f1': 0.6560176873795777, 'precision': 0.7869523486146797, 'recall': 0.5624381370214815, 'support': 523334}]`
- Confusion matrix (rows target, columns prediction): `[[175323, 56384, 32975, 31087], [60889, 24555, 18104, 12673], [42274, 33161, 41527, 35926], [56066, 65390, 107535, 294343]]`
- Adjacent directional errors: `{'FULL_to_HALF': 107535, 'HALF_to_FULL': 35926, 'HALF_to_LOW': 33161, 'LOW_to_HALF': 18104, 'LOW_to_ZERO': 60889, 'ZERO_to_LOW': 56384}`
- Same-side errors: `{'HALF_FULL': 143461, 'ZERO_LOW': 117273, 'total': 260734}`
- 64-boundary-crossing errors: `{'directional_pair_counts': {'FULL_to_LOW': 65390, 'FULL_to_ZERO': 56066, 'HALF_to_LOW': 33161, 'HALF_to_ZERO': 42274, 'LOW_to_FULL': 12673, 'LOW_to_HALF': 18104, 'ZERO_to_FULL': 31087, 'ZERO_to_HALF': 32975}, 'high_side_to_low_side': 196891, 'low_side_to_high_side': 94839, 'total': 291730}`

## Transition diagnostics

- Pooled: `{'candidate': 35611, 'f1': 0.504452899015375, 'fn': 16867, 'fp': 17910, 'precision': 0.4970655134649406, 'recall': 0.5120631798194862, 'reference': 34568, 'tolerance_distinct_onsets': 1, 'tp': 17701}`
- UP: `{'candidate': 17778, 'f1': 0.4874639892752218, 'fn': 8736, 'fp': 9233, 'precision': 0.4806502418719766, 'recall': 0.4944736994386899, 'reference': 17281, 'tp': 8545}`
- DOWN: `{'candidate': 17833, 'f1': 0.5214123006833713, 'fn': 8131, 'fp': 8677, 'precision': 0.5134301575730388, 'recall': 0.5296465552149013, 'reference': 17287, 'tp': 9156}`
- Representative Huber baseline candidate transitions: 38247

## Pattern diagnostics

- Top predicted patterns: `[{'count': 86456, 'pattern': 'FULL FULL FULL FULL', 'pattern_id': 255, 'probability': 0.31779101866180487}, {'count': 78185, 'pattern': 'ZERO ZERO ZERO ZERO', 'pattern_id': 0, 'probability': 0.28738885437763967}, {'count': 37774, 'pattern': 'HALF HALF HALF HALF', 'pattern_id': 170, 'probability': 0.13884794506952688}, {'count': 33485, 'pattern': 'LOW LOW LOW LOW', 'pattern_id': 85, 'probability': 0.1230826346336927}, {'count': 2396, 'pattern': 'HALF HALF FULL FULL', 'pattern_id': 175, 'probability': 0.008807107438624092}, {'count': 2208, 'pattern': 'LOW LOW HALF HALF', 'pattern_id': 90, 'probability': 0.008116065619566775}, {'count': 1868, 'pattern': 'HALF HALF LOW LOW', 'pattern_id': 165, 'probability': 0.006866309138292906}, {'count': 1846, 'pattern': 'FULL FULL HALF HALF', 'pattern_id': 250, 'probability': 0.006785442542445773}, {'count': 1743, 'pattern': 'FULL FULL FULL HALF', 'pattern_id': 254, 'probability': 0.006406839843706925}, {'count': 1659, 'pattern': 'LOW LOW LOW HALF', 'pattern_id': 86, 'probability': 0.0060980764777451455}, {'count': 1644, 'pattern': 'LOW LOW ZERO ZERO', 'pattern_id': 80, 'probability': 0.006042940162394828}, {'count': 1498, 'pattern': 'ZERO ZERO LOW LOW', 'pattern_id': 5, 'probability': 0.005506280026318401}, {'count': 1345, 'pattern': 'HALF HALF HALF LOW', 'pattern_id': 169, 'probability': 0.00494388960974516}, {'count': 1340, 'pattern': 'HALF HALF HALF FULL', 'pattern_id': 171, 'probability': 0.004925510837961721}, {'count': 1336, 'pattern': 'HALF FULL FULL FULL', 'pattern_id': 191, 'probability': 0.0049108078205349695}, {'count': 1321, 'pattern': 'HALF LOW LOW LOW', 'pattern_id': 149, 'probability': 0.004855671505184652}, {'count': 1227, 'pattern': 'LOW LOW LOW ZERO', 'pattern_id': 84, 'probability': 0.004510150595655993}, {'count': 1124, 'pattern': 'LOW HALF HALF HALF', 'pattern_id': 106, 'probability': 0.004131547896917145}, {'count': 1108, 'pattern': 'FULL HALF HALF HALF', 'pattern_id': 234, 'probability': 0.004072735827210139}, {'count': 1031, 'pattern': 'ZERO LOW LOW LOW', 'pattern_id': 21, 'probability': 0.0037897027417451747}]`
- Top target patterns: `[{'count': 121628, 'pattern': 'FULL FULL FULL FULL', 'pattern_id': 255, 'probability': 0.44707465089522996}, {'count': 67486, 'pattern': 'ZERO ZERO ZERO ZERO', 'pattern_id': 0, 'probability': 0.24806195851543633}, {'count': 27852, 'pattern': 'HALF HALF HALF HALF', 'pattern_id': 170, 'probability': 0.10237711034247003}, {'count': 21782, 'pattern': 'LOW LOW LOW LOW', 'pattern_id': 85, 'probability': 0.08006528139737477}, {'count': 2132, 'pattern': 'HALF FULL FULL FULL', 'pattern_id': 191, 'probability': 0.0078367082884585}, {'count': 2028, 'pattern': 'FULL FULL FULL HALF', 'pattern_id': 254, 'probability': 0.007454429835362962}, {'count': 1433, 'pattern': 'HALF HALF FULL FULL', 'pattern_id': 175, 'probability': 0.005267355993133691}, {'count': 1181, 'pattern': 'LOW HALF HALF HALF', 'pattern_id': 106, 'probability': 0.004341065895248353}, {'count': 1082, 'pattern': 'LOW LOW HALF HALF', 'pattern_id': 90, 'probability': 0.003977166213936255}, {'count': 972, 'pattern': 'LOW ZERO ZERO ZERO', 'pattern_id': 64, 'probability': 0.0035728332347005915}, {'count': 903, 'pattern': 'FULL FULL HALF HALF', 'pattern_id': 250, 'probability': 0.0033192061840891297}, {'count': 888, 'pattern': 'ZERO ZERO ZERO LOW', 'pattern_id': 1, 'probability': 0.003264069868738812}, {'count': 864, 'pattern': 'HALF HALF HALF LOW', 'pattern_id': 169, 'probability': 0.0031758517641783035}, {'count': 860, 'pattern': 'HALF HALF LOW LOW', 'pattern_id': 165, 'probability': 0.003161148746751552}, {'count': 813, 'pattern': 'HALF LOW LOW LOW', 'pattern_id': 149, 'probability': 0.0029883882919872233}, {'count': 761, 'pattern': 'LOW LOW LOW HALF', 'pattern_id': 86, 'probability': 0.002797249065439455}, {'count': 754, 'pattern': 'HALF HALF HALF FULL', 'pattern_id': 171, 'probability': 0.0027715187849426397}, {'count': 646, 'pattern': 'LOW LOW ZERO ZERO', 'pattern_id': 80, 'probability': 0.002374537314420352}, {'count': 643, 'pattern': 'HALF ZERO ZERO ZERO', 'pattern_id': 128, 'probability': 0.0023635100513502883}, {'count': 629, 'pattern': 'ZERO ZERO LOW LOW', 'pattern_id': 5, 'probability': 0.0023120494903566585}]`
