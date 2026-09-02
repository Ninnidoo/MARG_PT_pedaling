# Pre-overfit sanity

- Stage A: PASS
- Stage B multi-window: PASS (4 windows, 1041 owned onsets)
- Missing/duplicate supervision: 0/0
- State continuity at every window boundary: PASS
- Stage C gradient audit: PASS
- No-grad encoder parameter: `encoder.embed_tokens.weight`; expected inactive inherited embedding path.
- Boundary/Main scalar loss ratio: 0.974211
- Encoder gradient ratio: 1.227165
- Head gradient ratio: 1.090235
- Selected boundary loss weight: **0.25**
- ASAP test access: 0
