# representative_huber Training Report

- Final status: completed
- Objective: Unconstrained four-scalar representative regression with Huber δ=14/127.
- Architecture: PT pretrained 10-layer encoder + four independent Linear(768,1) heads
- Dataset/cache: canonical MAESTRO-clean 1,170 + ASAP train 892; ASAP validation 71
- Seed: 42
- Optimizer/LR: AdamW; encoder 1e-05; head 0.0001
- Best epoch: 6
- Best validation objective: 0.16955313484611229
- Checkpoints: `/workspace/project/analysis/stage2_encoder_only_4class_loss_phase3_v0/representative_huber_delta14/best.pt`, `/workspace/project/analysis/stage2_encoder_only_4class_loss_phase3_v0/representative_huber_delta14/last.pt`
- ASAP test access count: 0

## Epoch table

| epoch | train_objective | validation_objective | validation_huber | validation_representative_mae_cc64 |
|---:|---:|---:|---:|---:|
| 1 | 0.2014884454724524 | 0.1911863513802332 | 0.1911863513802332 | 29.54340346270899 |
| 2 | 0.1648409333254431 | 0.1842794553247144 | 0.1842794553247144 | 29.231005107759323 |
| 3 | 0.1524243017214137 | 0.18359018721253179 | 0.18359018721253179 | 28.7434359534552 |
| 4 | 0.14298060185033426 | 0.17539452784910714 | 0.17539452784910714 | 26.457176877306654 |
| 5 | 0.13418919372233853 | 0.17083718416199833 | 0.17083718416199833 | 26.08644915777146 |
| 6 | 0.1259195472751289 | 0.16955313484611229 | 0.16955313484611229 | 26.32525294344589 |
| 7 | 0.11804584343901044 | 0.17215657145724889 | 0.17215657145724889 | 26.171928738397657 |
| 8 | 0.11084202212745436 | 0.17492896202322725 | 0.17492896202322725 | 26.843414823287937 |
| 9 | 0.10412590804695951 | 0.17623427468792627 | 0.17623427468792627 | 26.686155251978942 |

## Canonical ASAP-validation evaluation

- 4C accuracy: 0.477443; Macro F1: 0.396158
- Transition P/R/F1: 0.471488 / 0.521667 / 0.495310
- JS divergence / Intersection: 0.042943 / 0.810684
- Prediction distribution: `[0.30188143486747065, 0.17587565658162196, 0.19583040804549112, 0.3264125005054162]`
- Class P/R/F1: `[{'class_id': 0, 'f1': 0.5465143845710259, 'precision': 0.519279415301162, 'recall': 0.5767642991659031, 'support': 295769}, {'class_id': 1, 'f1': 0.16383679387278088, 'precision': 0.13166309629552223, 'recall': 0.21681967974806618, 'support': 116221}, {'class_id': 2, 'f1': 0.2351957551100704, 'precision': 0.2019661669130241, 'recall': 0.28151326461200354, 'support': 152888}, {'class_id': 3, 'f1': 0.6390853006123797, 'precision': 0.7903329335653114, 'recall': 0.5364279790726382, 'support': 523334}]`
- Confusion matrix: `[[170589, 61426, 34806, 28948], [60894, 25199, 18499, 11629], [41957, 33993, 43040, 33898], [55071, 70772, 116760, 280731]]`
- Candidate/reference transitions: 38247 / 34568; TP/FP/FN: 18033/20214/16535
- Top 256-pattern probabilities: `[{'count': 82796, 'pattern': 'FULL FULL FULL FULL', 'pattern_id': 255, 'probability': 0.3043377577163273}, {'count': 77022, 'pattern': 'ZERO ZERO ZERO ZERO', 'pattern_id': 0, 'probability': 0.2831139520608117}, {'count': 41433, 'pattern': 'HALF HALF HALF HALF', 'pattern_id': 170, 'probability': 0.15229753026064774}, {'count': 36209, 'pattern': 'LOW LOW LOW LOW', 'pattern_id': 85, 'probability': 0.13309538950131042}, {'count': 2379, 'pattern': 'LOW LOW HALF HALF', 'pattern_id': 90, 'probability': 0.008744619614560398}, {'count': 2121, 'pattern': 'HALF HALF FULL FULL', 'pattern_id': 175, 'probability': 0.0077962749905349325}, {'count': 1990, 'pattern': 'HALF HALF LOW LOW', 'pattern_id': 165, 'probability': 0.007314751169808824}, {'count': 1948, 'pattern': 'LOW LOW LOW HALF', 'pattern_id': 86, 'probability': 0.0071603694868279345}, {'count': 1615, 'pattern': 'FULL FULL FULL HALF', 'pattern_id': 254, 'probability': 0.00593634328605088}, {'count': 1603, 'pattern': 'HALF HALF HALF LOW', 'pattern_id': 169, 'probability': 0.005892234233770626}, {'count': 1440, 'pattern': 'ZERO ZERO LOW LOW', 'pattern_id': 5, 'probability': 0.005293086273630506}, {'count': 1399, 'pattern': 'LOW LOW ZERO ZERO', 'pattern_id': 80, 'probability': 0.005142380345006304}, {'count': 1320, 'pattern': 'LOW HALF HALF HALF', 'pattern_id': 106, 'probability': 0.004851995750827964}, {'count': 1251, 'pattern': 'FULL FULL HALF HALF', 'pattern_id': 250, 'probability': 0.004598368700216502}, {'count': 1229, 'pattern': 'HALF LOW LOW LOW', 'pattern_id': 149, 'probability': 0.0045175021043693695}, {'count': 1186, 'pattern': 'HALF FULL FULL FULL', 'pattern_id': 191, 'probability': 0.004359444667031791}, {'count': 1140, 'pattern': 'LOW LOW LOW ZERO', 'pattern_id': 84, 'probability': 0.004190359966624151}, {'count': 1083, 'pattern': 'HALF HALF HALF FULL', 'pattern_id': 171, 'probability': 0.003980841968292943}, {'count': 1061, 'pattern': 'ZERO LOW LOW LOW', 'pattern_id': 21, 'probability': 0.0038999753724458103}, {'count': 920, 'pattern': 'LOW ZERO ZERO ZERO', 'pattern_id': 64, 'probability': 0.0033816940081528232}]`
