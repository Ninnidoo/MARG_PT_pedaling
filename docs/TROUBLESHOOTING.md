# Troubleshooting

## Resolved or Worked Around

### Windows Codex Writable Root

Issue:

- The active VS Code project moved to `C:\Users\eagle\MARG_research`, while the initial writable root was an older OneDrive path.

Resolution:

- Confirm the current VS Code workspace from VS Code workspaceStorage.
- Use the actual project root `C:\Users\eagle\MARG_research` for project files.
- When Codex needs to write there, use explicit user-approved file operations.

### Git Dubious Ownership

Issue:

- Git reported dubious ownership because the repository owner and Codex sandbox user differed.

Resolution:

- Do not change global or system Git config from Codex.
- Use temporary `git -c safe.directory="C:/Users/eagle/MARG_research" ...` for read-only status checks when needed.
- User performs final commits from the local Windows terminal.

### Codex Sandbox User vs Local Git User

Issue:

- Codex ran as a sandbox user, while the local repository belongs to the Windows user.

Resolution:

- Avoid relying on Codex for final commit workflow.
- Keep file authoring/documentation in Codex and commit manually from the local terminal.

### CPU-Only Local Environment

Issue:

- Local Windows environment has no detected NVIDIA GPU/CUDA path and initially no Python/Conda available on PATH.

Resolution:

- Use Google Colab GPU as the primary inference and future training environment.
- Keep local machine focused on documentation, project organization, and Git.

### Colab PyTorch Version Difference

Issue:

- Official README recommends PyTorch 2.7.1, but Colab runtimes may ship a different CUDA-enabled PyTorch version.

Resolution:

- Do not automatically replace Colab PyTorch unless a compatibility failure occurs.
- Record PyTorch/CUDA versions in metadata.
- If a future run fails, test the official PyTorch 2.7.1 install command for the active CUDA runtime and restart Colab if required.

### GPU Explicit Use Required

Issue:

- The official `script/inference.sh` calls `src/inference/inference.py`, which uses CPU by default.

Resolution:

- Use notebooks that import official classes/functions directly.
- Explicitly move `PianoT5Gemma` to CUDA with `model.to("cuda")`.
- Call `batch_performance_render(..., device="cuda")`.
- Do not modify third-party source for this workaround.

### Google Drive Persistence

Issue:

- Colab `/content` files are temporary.

Resolution:

- Mount Google Drive when persistence is needed.
- Use `/content/drive/MyDrive/MARG_research` as the default persistent root.
- Save or download important MIDI, JSON, and CSV outputs before ending the runtime.

## Confirmed Working

- Official repository clone.
- Official checkpoint `yhj137/pianist-transformer-rendering` in Colab.
- CUDA runtime in Colab.
- Loading the model on CUDA.
- `torch.bfloat16` for the verified smoke-test runtime.
- Official example `score/3.mid` inference.
- Output MIDI generation and MuseScore playback.
- Google Drive mount.
- Persistent output saving.
- Input upload in Colab.
- Seed configuration in the repeated experiment workflow.
- Repeated inference workflow.
- Pedal analysis table generation.
- Metadata JSON and pedal-event CSV saving.

## Still Open or Watch Items

- Free Colab GPU availability can vary.
- Larger or denser MIDI files may hit CUDA OOM or take longer.
- Local Windows CPU inference remains unverified.
- The first smoke-test seed was not recorded.
- The difference between `CC64 event count = 6` and a later `CC64 event count = 156` is not explained yet.
- Repedaling labels are not defined yet.
- Fast off-on transition candidates are exploratory only.
- Human performance MIDI data strategy is unresolved.

## If GPU Is Not Assigned

1. Select `Runtime > Change runtime type > GPU`.
2. Reconnect the runtime.
3. Rerun GPU checks.
4. If no free GPU is available, try later or use another runtime option.
5. Do not record the run as GPU inference unless CUDA is actually available.

## If CUDA or dtype Errors Occur

1. Record GPU name, PyTorch version, CUDA version, selected dtype, and traceback.
2. Restart the runtime and rerun setup.
3. Try a smaller input MIDI.
4. If `torch.bfloat16` fails, test `torch.float16` if compatible.
5. If version mismatch appears likely, test official PyTorch 2.7.1 for the active CUDA runtime.

## If Checkpoint or Output Files Disappear

1. Treat `/content` as temporary.
2. Remount Google Drive.
3. Restore checkpoint from Drive or re-download it.
4. Verify required checkpoint files are nonzero before loading.
5. Save output MIDI, JSON metadata, and CSV tables to Drive or download them immediately.
## Session Close And Notion Publishing

### `.env` Missing

Issue:

- `publish_session_to_notion.py` requires project-root `.env` for real upload.

Resolution:

1. Copy `.env.example` to `.env`.
2. Fill `NOTION_TOKEN` and `NOTION_PARENT_PAGE_ID` locally.
3. Do not commit `.env`.

### Notion Dependency Missing

Issue:

- Real upload requires the `requests` package.

Resolution:

```powershell
python -m pip install -r requirements-notion.txt
```

### Notion 401 or 403

Issue:

- The token may be invalid, or the integration may not have access to the parent page.

Resolution:

1. Regenerate or verify the Notion integration token.
2. Share the target parent page with the integration in Notion.
3. Confirm `NOTION_PARENT_PAGE_ID` points to the intended parent page.
4. Do not print the token or full parent page ID while debugging.

### Notion Rate Limit or Service Overload

Issue:

- The API may return 429 or 529.

Resolution:

- The publish script retries limited transient failures and respects `Retry-After` when present.
- If retries fail, wait and run the same command again.
- Duplicate upload prevention uses SHA-256 to avoid creating a second page for the same Markdown file after a successful publish.

### Duplicate Upload Prevented

Issue:

- The Markdown SHA-256 already exists in `docs/session_logs/.notion_publish_log.json`.

Resolution:

- Use the existing Notion page URL from the publish log output.
- If the content intentionally changed, regenerate or edit the Markdown so the SHA-256 changes, then dry-run again before uploading.

## Human Pedal-Tokenizer Batch

### Matplotlib 3.8 `boxplot()` Keyword Error

Issue:

- The first 165-file analysis pass completed its tokenizer and CSV work, then
  stopped while creating the score-index boxplot.
- Matplotlib 3.8.4 raised:
  `TypeError: Axes.boxplot() got an unexpected keyword argument 'tick_labels'`.

Cause:

- The `tick_labels` keyword is not supported by the installed Matplotlib 3.8.4
  API.

Resolution:

- Changed the compatible keyword to `labels`.
- Re-ran the exact batch command.
- The second run completed with exit code 0: inventory 166, eligible 165,
  successful 165, failed 0.
- All requested CSV files and PNG figures were then verified.

## Stage 2 Pedal Target Audit

### Dynamic Test Import Failed With `dataclass` Error

Issue:

- The first synthetic-test run failed while dynamically importing
  `scripts/analyze_stage2_pedal_targets.py`.
- Python 3.11 `dataclass` processing could not find the temporary module in
  `sys.modules`.

Resolution:

- Registered the module under `SPEC.name` in `sys.modules` before
  `exec_module()`.
- Re-ran the suite; all 11 synthetic tests passed.

### Empty Depth Class Broke Markdown Formatting

Issue:

- The first complete 166-file calculation wrote CSV, JSON, and figures, then
  failed while formatting `summary.md`.
- The time-weighted second quantile was CC64=127, leaving the strict
  quantile-based DEEP class empty and its representative median equal to
  `None`.

Resolution:

- Render optional aggregate values as `n/a` in Markdown.
- Added an explicit quantile-tie warning instead of hiding the empty class.
- Re-ran the full command to exit code 0 and verified all requested artifacts.
