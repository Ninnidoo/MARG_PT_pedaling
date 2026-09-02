# Storage and transfer policy

This project is split in place so that paths used by scripts and reports do not
change. Git and `.gitignore` define the GitHub layer; large artifacts stay at
their existing paths and form the local-only layer.

## GitHub layer

Keep these in the GitHub repository:

- source and entry points: `src/`, `scripts/`, `tests/`;
- reproducibility inputs: `configs/`, `Dockerfile`,
  `requirements-server.txt`, `.devcontainer/`;
- project documentation: root Markdown files, `docs/`, and notebooks;
- curated analysis records: Markdown reports and PNG figures under
  `analysis/`; Python and small HTML/CSS/JavaScript tools colocated with analyses;
- existing analysis files that were already tracked before this policy.

Do not add checkpoints, datasets, generated MIDI/audio, raw arrays, caches,
logs, or environment secrets. Git LFS is intentionally not required by this
repository policy.

## Local-only layer

Keep these outside GitHub and copy them separately only when needed:

| Path or pattern | Current purpose |
| --- | --- |
| `analysis/**/*.pt` | Selected 4-class and RUN-A/B training checkpoints |
| `analysis/**/*.npy`, `analysis/**/*.npz` | Re-creatable arrays and caches |
| Untracked `analysis/**/*.csv`, `analysis/**/*.json` | Generated metrics, manifests, and provenance |
| `outputs/` | Generated MIDI, WAV, and figures |
| `checkpoints/` | Downloaded pretrained Pianist Transformer weights |
| `logs/` | Runtime logs |
| `data/raw/`, `data/interim/`, `data/processed/` | Local datasets and derived data |
| `third_party/PianistTransformer/` | Separately managed upstream checkout |
| `.env` and `.env.*` except `.env.example` | Local credentials and configuration |

As of 2026-09-03, the largest local-only groups are approximately:

- remaining selected checkpoints: 19.755 GiB across 35 `.pt` files;
- generated audio: 2.807 GiB across 75 WAV files;
- NPY/NPZ arrays and caches: 0.836 GiB;
- downloaded pretrained weights: 0.253 GiB.

The final RUN A checkpoint pair is:

- `analysis/stage2_binary_2slot_D_pre_main_post_full_v1/best.pt`
- `analysis/stage2_binary_2slot_D_pre_main_post_full_v1/last.pt`

The final RUN B inference checkpoint is:

- `analysis/stage2_state_anchored_transition_v1_full_v0/best.pt`

## Moving to another server

1. Clone the GitHub repository.
2. Clone Pianist Transformer at commit
   `747df2d12291e37f6638b39f1b71517e579ad48c` into
   `third_party/PianistTransformer/`.
3. Build or attach the environment described by `Dockerfile` and
   `requirements-server.txt`.
4. Restore only the local artifacts required for the intended experiment,
   preserving their repository-relative paths.
5. Mount public and private datasets at `/workspace/public` and
   `/workspace/private`, or update the corresponding runtime arguments.

Example third-party checkout:

```bash
git clone https://github.com/yhj137/PianistTransformer.git \
  third_party/PianistTransformer
git -C third_party/PianistTransformer checkout \
  747df2d12291e37f6638b39f1b71517e579ad48c
```

Before copying local artifacts, compare available space with:

```bash
du -sh analysis outputs checkpoints logs
```

For inference, copy `best.pt` only. Copy `last.pt` or a resume checkpoint only
when exact training continuation is required.
