# ce_ntl_was Training Report

- Final status: completed
- Objective: CE - 0.3 × Σ_c p_c |V_c-V_y|/127, V=[0,51,79,127].
- Architecture: PT pretrained 10-layer encoder + four independent Linear(768,4) heads
- Dataset/cache: canonical MAESTRO-clean 1,170 + ASAP train 892; ASAP validation 71
- Seed: 42
- Optimizer/LR: AdamW; encoder 1e-05; head 0.0001
- Best epoch: 2
- Best validation objective: 0.8084597620760576
- Checkpoints: `/workspace/project/analysis/stage2_encoder_only_4class_loss_phase3_v0/ce_ntl_was_lambda_0p3/best.pt`, `/workspace/project/analysis/stage2_encoder_only_4class_loss_phase3_v0/ce_ntl_was_lambda_0p3/last.pt`
- ASAP test access count: 0

## Epoch table

| epoch | train_objective | validation_objective | validation_ce | validation_ntl_was |
|---:|---:|---:|---:|---:|
| 1 | 0.8800801433157749 | 0.8177164857153105 | 0.899321992897147 | 0.2720183445925173 |
| 2 | 0.7975951324233534 | 0.8084597620760576 | 0.8865216979290426 | 0.26020643203510646 |
| 3 | 0.7597749211022055 | 0.8171285662889923 | 0.8933586527996028 | 0.2541002725183632 |
| 4 | 0.7224059043191731 | 0.8572195370254799 | 0.9313103232622589 | 0.24696927588149656 |
| 5 | 0.6803727844968448 | 0.8660226363640326 | 0.9393543204040385 | 0.24443892125517186 |

## Canonical ASAP-validation evaluation

- 4C accuracy: 0.570835; Macro F1: 0.395119
- Transition P/R/F1: 0.351994 / 0.574057 / 0.436400
- JS divergence / Intersection: 0.071404 / 0.811103
- Prediction distribution: `[0.4193907069578354, 0.030760550333942285, 0.06521799061212337, 0.4846307520960989]`
- Class P/R/F1: `[{'class_id': 0, 'f1': 0.562578191995001, 'precision': 0.4635834578624235, 'recall': 0.7153318975281385, 'support': 295769}, {'class_id': 1, 'f1': 0.08503958048030996, 'precision': 0.19014757722411424, 'recall': 0.05476635031534748, 'support': 116221}, {'class_id': 2, 'f1': 0.2100339946126803, 'precision': 0.3312479745248059, 'recall': 0.15376615561718382, 'support': 152888}, {'class_id': 3, 'f1': 0.7228258852305336, 'precision': 0.7200524857740419, 'recall': 0.7256207316933354, 'support': 523334}]`
- Confusion matrix: `[[211573, 10189, 7975, 66032], [76050, 6365, 9765, 24041], [63945, 7868, 23509, 57566], [104818, 9052, 29722, 379742]]`
- Candidate/reference transitions: 56376 / 34568; TP/FP/FN: 19844/36532/14724
- Top 256-pattern probabilities: `[{'count': 120951, 'pattern': 'FULL FULL FULL FULL', 'pattern_id': 255, 'probability': 0.4445861651957523}, {'count': 103581, 'pattern': 'ZERO ZERO ZERO ZERO', 'pattern_id': 0, 'probability': 0.38073831202008435}, {'count': 11332, 'pattern': 'HALF HALF HALF HALF', 'pattern_id': 170, 'probability': 0.04165364836998673}, {'count': 3656, 'pattern': 'LOW LOW LOW LOW', 'pattern_id': 85, 'probability': 0.013438557928050784}, {'count': 2640, 'pattern': 'ZERO ZERO FULL FULL', 'pattern_id': 15, 'probability': 0.009703991501655927}, {'count': 2515, 'pattern': 'FULL FULL ZERO ZERO', 'pattern_id': 240, 'probability': 0.009244522207069946}, {'count': 1716, 'pattern': 'FULL FULL FULL ZERO', 'pattern_id': 252, 'probability': 0.0063075944760763525}, {'count': 1351, 'pattern': 'HALF HALF FULL FULL', 'pattern_id': 175, 'probability': 0.004965944135885287}, {'count': 1339, 'pattern': 'ZERO FULL FULL FULL', 'pattern_id': 63, 'probability': 0.004921835083605033}, {'count': 1271, 'pattern': 'FULL ZERO ZERO ZERO', 'pattern_id': 192, 'probability': 0.004671883787350259}, {'count': 1095, 'pattern': 'ZERO ZERO ZERO FULL', 'pattern_id': 3, 'probability': 0.004024951020573197}, {'count': 893, 'pattern': 'HALF HALF HALF FULL', 'pattern_id': 171, 'probability': 0.003282448640522251}, {'count': 850, 'pattern': 'LOW LOW ZERO ZERO', 'pattern_id': 80, 'probability': 0.0031243912031846737}, {'count': 789, 'pattern': 'HALF FULL FULL FULL', 'pattern_id': 191, 'probability': 0.0029001701874267144}, {'count': 784, 'pattern': 'FULL FULL FULL HALF', 'pattern_id': 254, 'probability': 0.0028817914156432753}, {'count': 675, 'pattern': 'FULL FULL HALF HALF', 'pattern_id': 250, 'probability': 0.0024811341907642996}, {'count': 626, 'pattern': 'LOW LOW HALF HALF', 'pattern_id': 90, 'probability': 0.0023010222272865947}, {'count': 619, 'pattern': 'ZERO ZERO LOW LOW', 'pattern_id': 5, 'probability': 0.00227529194678978}, {'count': 544, 'pattern': 'LOW LOW LOW HALF', 'pattern_id': 86, 'probability': 0.0019996103700381913}, {'count': 523, 'pattern': 'FULL FULL ZERO FULL', 'pattern_id': 243, 'probability': 0.0019224195285477462}]`
