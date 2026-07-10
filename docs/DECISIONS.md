# Decisions

## Decision Log

| Date | Decision | Rationale | Status |
| --- | --- | --- | --- |
| 2026-07-10 | Use Colab GPU first for pretrained Pianist Transformer inference | Local machine has no NVIDIA/CUDA path and no Python/Conda available; Colab GPU is the fastest path to verify the official checkpoint while local CPU remains a fallback. | Planned |