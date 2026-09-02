# Custom Event Model v1 B3-S Canonical Training Report

## Frozen provenance

- Model/version: B3-S / `1.0.0-b3s`
- Parameters: 103,339,216
- Tokenizer cache ID: `3a5520155b5db1e9a1da7f8148556aa3e1da852655c9adde25d3dbbd1d966263`
- Alignment ID: `98de59ef7a41fba26a2c89fe686c273f6c8ec7f27979922e71813624ac1a4822`
- Alignment config SHA: `77df6faf08025794e40c58d0db31cf721f582b3168d9fb201557c0500807502b`
- Decoder: `1.0.0` / `938ca7e20e1d398aa530938bbcede404fd3a04af23842fbcf351ba02df0268f8`
- Event-weight SHA: `26da9ede3373bd3df5c3f28c259e29f173a0920e080eb3a27cdbc5c790ed5b55`
- Seed/head initialization SHA: 42 / `abdaafa1b58bb8216ffd91ae51f1d129a213ff64b2589ff6119e9051100fc771`

## Data and training configuration

- Train: 2,062 performances / 35,573 windows / 9,009,549 owned onsets
- Validation: 71 performances / 1,078 windows / 272,927 owned onsets (19 pieces)
- AdamW; encoder/head LR 1e-5/1e-4; weight decay 0.01; grad clip 1.0
- FP16 AMP; micro/accumulation/effective batch 4/4/16; no scheduler
- Maximum epochs 10; patience 3; min delta 1e-4
- Optimization adds unit-weight unweighted slot-macro state CE; selection is the legacy v0 total and excludes state CE.

## Epoch objectives

| epoch | train optimization | train legacy | val legacy selection | val state CE | val main event | val main timing | val initial | val terminal event | val terminal timing |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4.766672 | 3.601900 | 4.108575 | 1.072536 | 0.551362 | 0.147765 | 1.265945 | 1.778477 | 0.365026 |
| 2 | 4.177412 | 3.102011 | 4.070437 | 1.004152 | 0.530394 | 0.137311 | 1.534921 | 1.645148 | 0.222664 |
| 3 | 3.586757 | 2.547111 | 4.297156 | 0.979888 | 0.522703 | 0.137242 | 1.670728 | 1.684461 | 0.282022 |
| 4 | 2.887442 | 1.869605 | 4.962112 | 0.984884 | 0.516689 | 0.133198 | 2.188331 | 1.841840 | 0.282055 |
| 5 | 2.309988 | 1.307206 | 6.832405 | 0.963223 | 0.511507 | 0.130739 | 3.190515 | 2.660733 | 0.338911 |

## Selection and numerical summary

- Best epoch: 2; legacy validation selection total: 4.070436936
- Stop reason / early-stopping epoch: early_stopping / 5
- AMP overflow events / maximum consecutive: 5 / 1

## Best-checkpoint state-head diagnostics

- Overall accuracy / Macro F1: 0.598484 / 0.384822
- Target distribution: `{"FULL": {"count": 772822, "rate": 0.4719344977472609}, "HALF": {"count": 249776, "rate": 0.15252918668117604}, "LOW": {"count": 185554, "rate": 0.11331112959387186}, "ZERO": {"count": 429410, "rate": 0.26222518597769123}}`
- Prediction distribution: `{"FULL": {"count": 1122667, "rate": 0.685572210395698}, "HALF": {"count": 85977, "rate": 0.05250305026618839}, "LOW": {"count": 10766, "rate": 0.006574407564415882}, "ZERO": {"count": 418152, "rate": 0.2553503317736977}}`
- Slot1 accuracy / Macro F1: 0.605851 / 0.372879; recalls `{"FULL": 0.9156978736039102, "HALF": 0.11176981272479226, "LOW": 0.0022316830323422373, "ZERO": 0.5615148952687248}`
- Slot2 accuracy / Macro F1: 0.586794 / 0.415110; recalls `{"FULL": 0.9054948945517747, "HALF": 0.15065128722424456, "LOW": 0.09196770821233592, "ZERO": 0.5477481416703105}`
- Slot3 accuracy / Macro F1: 0.593085 / 0.385313; recalls `{"FULL": 0.9099160087218684, "HALF": 0.1301477522791575, "LOW": 0.026097381040206567, "ZERO": 0.5561419345878759}`
- Slot4 accuracy / Macro F1: 0.600179 / 0.374327; recalls `{"FULL": 0.9145143829260749, "HALF": 0.12045879688884513, "LOW": 0.0036767141437562107, "ZERO": 0.5568835950487024}`
- Slot5 accuracy / Macro F1: 0.602571 / 0.376869; recalls `{"FULL": 0.9147470398277718, "HALF": 0.1289637495422922, "LOW": 0.0016996034258672976, "ZERO": 0.5584811182975172}`
- Slot6 accuracy / Macro F1: 0.602421 / 0.371451; recalls `{"FULL": 0.9178754673987016, "HALF": 0.11531786962579131, "LOW": 0.000785975463896388, "ZERO": 0.5507405451968767}`

## Best-checkpoint Main-event target-space diagnostics

- Overall accuracy / non-NONE precision/recall/F1: 0.926719 / 0.446649 / 0.578883 / 0.504241
- Predicted active / target active: 107149 / 82673
- Destination target/predicted frequency: `{"SET_FULL": {"predicted": 30455, "target": 14071}, "SET_HALF": {"predicted": 43047, "target": 30484}, "SET_LOW": {"predicted": 23987, "target": 24819}, "SET_ZERO": {"predicted": 9660, "target": 13299}}`
- Slot1 predicted active / target active / F1: 43048 / 44170 / 0.526841
- Slot2 predicted active / target active / F1: 26045 / 20893 / 0.536239
- Slot3 predicted active / target active / F1: 17310 / 9778 / 0.487374
- Slot4 predicted active / target active / F1: 9388 / 4540 / 0.466686
- Slot5 predicted active / target active / F1: 6865 / 2264 / 0.371563
- Slot6 predicted active / target active / F1: 4493 / 1028 / 0.272052

## Checkpoints

- `/workspace/project/analysis/custom_event_model_v1_b3s_full_seed42/best.pt` — `929ef546f352cf549f6d2eceb9eefeae2fb8cb02bcc2fbc08a046ed9e55382bb` (1207208496 bytes)
- `/workspace/project/analysis/custom_event_model_v1_b3s_full_seed42/last.pt` — `75d5cb4af739bf101ca7e78065f47539860dbb591ee3b7df93921c048da66000` (1207208496 bytes)
- `/workspace/project/analysis/custom_event_model_v1_b3s_full_seed42/checkpoints/epoch_01.pt` — `aedc272be08d6535106004c97b55140493f154e606da4e9ac4b2a844665b9e53` (413431600 bytes)
- `/workspace/project/analysis/custom_event_model_v1_b3s_full_seed42/checkpoints/epoch_02.pt` — `219dea9c78a5f1f52c36e8135f739c064b513b4617ab44e267de4f2eb37f9a8a` (413431600 bytes)
- `/workspace/project/analysis/custom_event_model_v1_b3s_full_seed42/checkpoints/epoch_03.pt` — `de47d127655b6730ce9a3383e974294ea40af05304659e238442b21e4546e433` (413431600 bytes)
- `/workspace/project/analysis/custom_event_model_v1_b3s_full_seed42/checkpoints/epoch_04.pt` — `3dc7c59135eaa16a15a88fc38db8687540e844305ff49ba10235f6c0de49f568` (413431600 bytes)
- `/workspace/project/analysis/custom_event_model_v1_b3s_full_seed42/checkpoints/epoch_05.pt` — `31e7c8d34f7a20f96239b1970375482d5f2ff9d3b181dbdc24b26ce09950e840` (413431600 bytes)

## Scientific boundary

This report makes no v0-versus-v1 final-performance superiority claim.
- No canonical MIDI validation inference
- No matched Hybrid comparison
- No ASAP test
- No Repedal
- No candidate MIDI or 4C/Transition/JS/Intersection evaluation
