# syntax=docker/dockerfile:1.7

# PyTorch 2.7.1, CUDA 11.8, and cuDNN 9 match the verified
# Pianist Transformer software baseline and the RTX 2080 Ti server.
FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime@sha256:8d409f72f99e5968b5c4c9396a21f4b723982cfdf2c1a5b9cc045c5d0a7345a1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG USERNAME=research
ARG USER_UID=1000
ARG USER_GID=1000

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PROJECT_ROOT=/workspace/project \
    PUBLIC_DATA_ROOT=/workspace/public \
    PRIVATE_DATA_ROOT=/workspace/private \
    PIANIST_TRANSFORMER_ROOT=/workspace/project/third_party/PianistTransformer

USER root

# Install only small runtime/VS Code prerequisites. All apt operations happen
# inside the image; the host operating system is never modified.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        curl \
        libgl1 \
        libglib2.0-0 \
        libsndfile1 \
        procps \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-server.txt /tmp/requirements-server.txt

RUN python -m pip install \
        --no-cache-dir \
        --disable-pip-version-check \
        --requirement /tmp/requirements-server.txt \
    && python - <<'PY'
import importlib.metadata as metadata

import torch

assert torch.__version__.split("+", 1)[0] == "2.7.1", torch.__version__
assert torch.version.cuda == "11.8", torch.version.cuda
assert metadata.version("transformers") == "4.54.0"
assert metadata.version("miditoolkit") == "1.0.1"
assert metadata.version("mido") == "1.3.3"
PY

# Run VS Code terminals and research processes as a non-root user. Dev
# Containers updates this UID/GID to the Remote SSH user on Linux.
RUN groupadd --gid "$USER_GID" "$USERNAME" \
    && useradd \
        --uid "$USER_UID" \
        --gid "$USER_GID" \
        --create-home \
        --shell /bin/bash \
        "$USERNAME" \
    && mkdir -p /workspace/project /workspace/public /workspace/private \
    && chown -R "$USER_UID:$USER_GID" \
        "/home/$USERNAME" \
        /workspace

ENV HOME=/home/$USERNAME \
    PATH=/opt/conda/bin:$PATH

WORKDIR /workspace/project
USER $USERNAME

CMD ["sleep", "infinity"]
