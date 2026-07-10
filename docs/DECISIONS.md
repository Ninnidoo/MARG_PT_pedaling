# Decisions

## Decision Log

| Date | Decision | Rationale | Status |
| --- | --- | --- | --- |
| 2026-07-10 | Use Colab GPU first for pretrained Pianist Transformer inference | Local machine has no NVIDIA/CUDA path and no Python/Conda available; Colab GPU was the fastest path to verify the official checkpoint while local CPU remains a fallback. | Completed for smoke test |
| 2026-07-10 | Use Colab GPU as the main runtime for compute-heavy research work | Official example inference succeeded on Colab GPU using `cuda` and `torch.bfloat16`; local VS Code remains the source of truth for code and documentation, while `/content` files are temporary. | Active |
| 2026-07-10 | Use local VS Code repository as source of truth | Local files are easier to version, review, and document consistently; Colab runtimes are temporary. | Active |
| 2026-07-10 | Use Google Drive for persistent Colab experiment storage | `/content` is temporary; `/content/drive/MyDrive/MARG_research` allows checkpoint, input, output, and metadata persistence across Colab sessions. | Active |
| 2026-07-10 | Split ChatGPT and Codex roles | ChatGPT supports research direction, methodology, step planning, and prompt drafting; Codex updates local files, notebooks, and documentation. | Active |
| 2026-07-10 | Keep Git commits manual from local Windows terminal | Codex sandbox/user differences caused Git ownership friction; user-controlled local commits are clearer and safer. | Active |
| 2026-07-10 | Preserve smoke-test notebook as verified reference workflow | The smoke-test notebook verified CUDA model loading and official checkpoint inference; experiment notebook changes should not break this baseline. | Active |
| 2026-07-10 | Do not interpret exploratory pedal metrics as conclusions | CC64 counts and fast off-on candidates need research definitions and annotation protocol before becoming claims or labels. | Active |