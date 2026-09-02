# weighted_ce Training Report

- Final status: completed
- Objective: Weighted cross entropy with train-cache-only mean-one inverse-sqrt frequency weights.
- Architecture: PT pretrained 10-layer encoder + four independent Linear(768,4) heads
- Dataset/cache: canonical MAESTRO-clean 1,170 + ASAP train 892; ASAP validation 71
- Seed: 42
- Optimizer/LR: AdamW; encoder 1e-05; head 0.0001
- Best epoch: 2
- Best validation objective: 0.9956573645076358
- Checkpoints: `/workspace/project/analysis/stage2_encoder_only_4class_loss_phase3_v0/weighted_ce/best.pt`, `/workspace/project/analysis/stage2_encoder_only_4class_loss_phase3_v0/weighted_ce/last.pt`
- ASAP test access count: 0

## Epoch table

| epoch | train_objective | validation_objective | validation_ce |
|---:|---:|---:|---:|
| 1 | 1.0493568556359258 | 1.0107748231536606 | 0.9207947380246391 |
| 2 | 0.9551460250779081 | 0.9956573645076358 | 0.9171660897461956 |
| 3 | 0.9112674659991369 | 1.0020076545121417 | 0.8962229269112638 |
| 4 | 0.8670364583345637 | 1.0473844334407854 | 0.9298151094528652 |
| 5 | 0.8160768730223235 | 1.058277104911768 | 0.9483692263406813 |

## Canonical ASAP-validation evaluation

- 4C accuracy: 0.533587; Macro F1: 0.408260
- Transition P/R/F1: 0.340975 / 0.602783 / 0.435565
- JS divergence / Intersection: 0.039055 / 0.858292
- Prediction distribution: `[0.3572493227422598, 0.11553631093941254, 0.09431158634530772, 0.43290277997301996]`
- Class P/R/F1: `[{'class_id': 0, 'f1': 0.5353263251389271, 'precision': 0.47129999511270365, 'recall': 0.6194834482315591, 'support': 295769}, {'class_id': 1, 'f1': 0.16281117094924966, 'precision': 0.1566556375668109, 'recall': 0.16947023343457723, 'support': 116221}, {'class_id': 2, 'f1': 0.2357789440315593, 'precision': 0.29350780953123323, 'recall': 0.19702658154989272, 'support': 152888}, {'class_id': 3, 'f1': 0.6991243171926663, 'precision': 0.7378908488823792, 'recall': 0.6642278162702977, 'support': 523334}]`
- Confusion matrix: `[[183224, 43497, 11381, 57667], [65528, 19696, 11218, 19779], [52494, 24240, 30123, 46031], [87517, 38295, 49909, 347613]]`
- Candidate/reference transitions: 61110 / 34568; TP/FP/FN: 20837/40273/13731
- Top 256-pattern probabilities: `[{'count': 107051, 'pattern': 'FULL FULL FULL FULL', 'pattern_id': 255, 'probability': 0.3934931796377912}, {'count': 85673, 'pattern': 'ZERO ZERO ZERO ZERO', 'pattern_id': 0, 'probability': 0.3149129030005183}, {'count': 20368, 'pattern': 'LOW LOW LOW LOW', 'pattern_id': 85, 'probability': 0.07486776473701816}, {'count': 16424, 'pattern': 'HALF HALF HALF HALF', 'pattern_id': 170, 'probability': 0.060370589554241266}, {'count': 1704, 'pattern': 'LOW LOW ZERO ZERO', 'pattern_id': 80, 'probability': 0.006263485423796099}, {'count': 1627, 'pattern': 'FULL FULL ZERO ZERO', 'pattern_id': 240, 'probability': 0.005980452338331134}, {'count': 1485, 'pattern': 'ZERO ZERO FULL FULL', 'pattern_id': 15, 'probability': 0.0054584952196814595}, {'count': 1439, 'pattern': 'HALF HALF FULL FULL', 'pattern_id': 175, 'probability': 0.005289410519273818}, {'count': 1396, 'pattern': 'ZERO ZERO LOW LOW', 'pattern_id': 5, 'probability': 0.005131353081936241}, {'count': 1363, 'pattern': 'HALF HALF HALF FULL', 'pattern_id': 171, 'probability': 0.005010053188165541}, {'count': 1321, 'pattern': 'LOW LOW HALF HALF', 'pattern_id': 90, 'probability': 0.004855671505184652}, {'count': 1309, 'pattern': 'FULL FULL FULL ZERO', 'pattern_id': 252, 'probability': 0.004811562452904397}, {'count': 1263, 'pattern': 'ZERO ZERO ZERO LOW', 'pattern_id': 1, 'probability': 0.004642477752496756}, {'count': 1164, 'pattern': 'HALF HALF LOW LOW', 'pattern_id': 165, 'probability': 0.004278578071184659}, {'count': 1087, 'pattern': 'LOW LOW LOW HALF', 'pattern_id': 86, 'probability': 0.003995544985719694}, {'count': 1044, 'pattern': 'LOW ZERO ZERO ZERO', 'pattern_id': 64, 'probability': 0.0038374875483821167}, {'count': 1012, 'pattern': 'HALF FULL FULL FULL', 'pattern_id': 191, 'probability': 0.0037198634089681054}, {'count': 1012, 'pattern': 'FULL FULL FULL HALF', 'pattern_id': 254, 'probability': 0.0037198634089681054}, {'count': 971, 'pattern': 'LOW LOW LOW ZERO', 'pattern_id': 84, 'probability': 0.0035691574803439034}, {'count': 939, 'pattern': 'FULL FULL HALF HALF', 'pattern_id': 250, 'probability': 0.0034515333409298925}]`
