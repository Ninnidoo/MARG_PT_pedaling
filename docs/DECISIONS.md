# Decisions

## Decision Log

| Date | Decision | Rationale | Status |
| --- | --- | --- | --- |
| 2026-07-10 | Use Colab GPU first for pretrained Pianist Transformer inference | Local machine has no NVIDIA/CUDA path and no Python/Conda available; Colab GPU was the fastest path to verify the official checkpoint while local CPU remains a fallback. | Completed for smoke test |
| 2026-07-10 | Use Colab GPU as the main runtime for compute-heavy research work | Official example inference succeeded on Colab GPU using `cuda` and `torch.bfloat16`; local VS Code remains the source of truth for code and documentation, while `/content` files are temporary. | Active |