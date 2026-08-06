# Shared GPU Server Docker Setup

This document describes the Docker environment for the Pianist Transformer
repedaling project. The existing research code and verified Colab notebooks are
not changed by this setup.

## Safety Boundaries

- Install Python and operating-system packages only while building the image.
- Do not install project packages on the host with pip, Conda, apt, or sudo.
- Do not mount the Docker socket, host home directory, SSH keys, Git
  configuration, or Git credentials into the container.
- Do not put .env files, passwords, API keys, datasets, checkpoints, or outputs
  in the image.
- Do not use privileged mode, host networking, host PID, or host IPC.
- Never use docker system prune, docker image prune, or docker container prune
  on the shared server.
- Only operate on containers and images whose names or labels clearly belong to
  this project and user.

## Bind Mounts

| Purpose | Host path | Container path | Access |
| --- | --- | --- | --- |
| Project and ignored third-party source | /home/intern_2026_summer/yimilkyun/MARG_PT_pedaling | /workspace/project | read/write |
| Public datasets and public pretrained models | /public/intern_2026_summer_public_dataset/ilkyun_data | /workspace/public | read-only |
| Private checkpoints, outputs, logs, and caches | /private/intern_2026_summer_private_dataset/ilkyun_data | /workspace/private | read/write |

These are host bind mounts. Project and private files remain on the server
after the container is stopped, rebuilt, or deleted. The public mount is
read-only to protect shared data.

Recommended private directory layout:

    cache/
      huggingface/
      matplotlib/
      wandb/
    checkpoints/
    logs/
    outputs/
    runs/

Hugging Face, Matplotlib, and Weights & Biases caches are redirected to the
private mount. Existing scripts are unchanged. When a script supports an
output-path argument, pass a path under /workspace/private/outputs to keep the
result outside the project tree.

## Third-Party Source And Git

Git operations remain on the Remote SSH host. The image neither installs Git
nor clones the official repository. Before inference, the host project should
contain:

    third_party/PianistTransformer/

The parent repository already ignores this directory. Verify its commit on the
host:

    git -C third_party/PianistTransformer rev-parse HEAD

Expected commit:

    747df2d12291e37f6638b39f1b71517e579ad48c

Do not create GitHub authentication inside the container. VS Code Git
integration is disabled in the Dev Container; commit, pull, and push from a
Remote SSH host terminal.

## GPU Selection

The Dev Container currently exposes exactly this physical GPU:

    GPU-c6fcf288-c0d2-af93-4b99-579fb2817af0

Confirm that this GPU is assigned and available under the laboratory scheduling
rules before opening the Dev Container. If another GPU is assigned, replace
only the device value in .devcontainer/devcontainer.json with that GPU UUID.
Never change it to all.

Inside the container, the selected physical GPU appears as cuda:0, and
torch.cuda.device_count() must return 1. RTX 2080 Ti should use float16; do not
assume that the Colab bfloat16 path is supported.

## VS Code Remote SSH And Dev Containers

1. Connect to the server with VS Code Remote SSH.
2. Open /home/intern_2026_summer/yimilkyun/MARG_PT_pedaling.
3. Confirm the assigned GPU UUID in .devcontainer/devcontainer.json.
4. Run Dev Containers: Reopen in Container.
5. Select /opt/conda/bin/python if VS Code asks for an interpreter.

The container uses a non-root research user. Dev Containers updates its UID/GID
to match the Remote SSH user on Linux so that bind-mounted files retain the
correct host ownership.

## Manual Build Reference

The following command is documentation only. Run it from the host after review:

    cd /home/intern_2026_summer/yimilkyun/MARG_PT_pedaling

    docker build \
      --pull \
      --build-arg USER_UID="$(id -u)" \
      --build-arg USER_GID="$(id -g)" \
      --label owner=yimilkyun \
      --label project=MARG_PT_pedaling \
      --tag yimilkyun/marg-pt-pedaling:pt2.7.1-cu118 \
      .

The .dockerignore allowlist sends only Dockerfile, requirements-server.txt, and
.dockerignore to the builder.

## Manual One-Off Run Reference

Replace ASSIGNED_GPU_UUID with exactly one assigned GPU UUID:

    docker run --rm -it \
      --name yimilkyun-marg-pt-smoke \
      --label owner=yimilkyun \
      --label project=MARG_PT_pedaling \
      --gpus device=ASSIGNED_GPU_UUID \
      --user "$(id -u):$(id -g)" \
      --init \
      --security-opt=no-new-privileges:true \
      --shm-size=4g \
      --mount type=bind,src=/home/intern_2026_summer/yimilkyun/MARG_PT_pedaling,dst=/workspace/project \
      --mount type=bind,src=/public/intern_2026_summer_public_dataset/ilkyun_data,dst=/workspace/public,readonly \
      --mount type=bind,src=/private/intern_2026_summer_private_dataset/ilkyun_data,dst=/workspace/private \
      --workdir /workspace/project \
      yimilkyun/marg-pt-pedaling:pt2.7.1-cu118 \
      bash

The --rm option removes only this named one-off container when it exits. It
does not remove images, Docker volumes, bind-mounted files, or another user's
resources.

## Validation After A Future Build

Run these commands inside the project container:

    nvidia-smi -L

    python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"

    python -m unittest discover \
      -s tests \
      -p 'test_stage2_pedal_targets.py' \
      -v

Expected GPU results:

- CUDA is available.
- Device count is exactly 1.
- Device name is NVIDIA GeForce RTX 2080 Ti.
- BF16 support is false, so model inference uses FP16.

The setup is not verified until the image builds, these checks pass, and a CUDA
smoke inference produces a non-empty MIDI file in a bind-mounted persistent
path.
