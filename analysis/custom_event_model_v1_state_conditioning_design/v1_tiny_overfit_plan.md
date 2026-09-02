# State-Conditioned Event Model v1 — Tiny-Overfit Plan

이 design task에서는 실행하지 않는다.

## Reuse

기존 v0 tiny-overfit infrastructure, frozen tokenizer cache, owner windows, seed 42, optimizer family, timing target, five legacy loss branches를 재사용한다. B3-S state output/target/loss check만 추가한다.

## Coverage subset

main_interval_start_state와 target slot rollout으로 exact pre-state를 derive한다. 다음 active transition을 모두 포함하는 최소 deterministic train owner-window 집합을 greedy select한다.

- ZERO→LOW
- LOW→ZERO
- HALF→FULL
- FULL→HALF
- OFF→ON: ZERO/LOW에서 HALF/FULL
- ON→OFF: HALF/FULL에서 ZERO/LOW

한 window가 여러 category를 덮으면 우선한다. Unchanged Initial/Terminal branch도 exercise하도록 first-owner와 last-owner sample을 최소 하나씩 포함한다. Manifest에 performance/window/global onset/slot/pre-state/destination/category를 기록한다.

## Checks

1. State target가 direct global target rollout과 exact 일치.
2. State-head pre-state memorization을 전체 valid slot 및 slot별 report.
3. Required transition을 포함한 Main destination memorization.
4. Active recall을 확인하여 all-NONE collapse 배제.
5. Decoder suppression 전 predicted same-state SET를 report하고 legitimate target과 redundant prediction 구분.
6. Unchanged SmoothL1 path로 active Main timing memorization.
7. Initial/Terminal loss와 shapes exercise.
8. Gradient freeze: state CE→encoder 0, event CE→state heads 0, direct Main event→encoder/head nonzero.
9. Zero new columns 때문에 step-zero Main logits가 corresponding v0 init과 동일.
10. Decoder-v1 interface regression 통과.

Full-validation pass/fail numerical threshold는 만들지 않는다. Tiny-overfit은 coverage accounting과 명확한 memorization evidence로 판정한다.

## Stop point

Implementation smoke와 tiny-overfit evidence 뒤 중단한다. Full training, validation candidate inference, ASAP test, Repedal은 다음 explicit task 전 수행하지 않는다.
