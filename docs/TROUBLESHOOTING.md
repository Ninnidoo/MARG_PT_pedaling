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