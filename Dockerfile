# =============================================================================
#  LTX-2.3 — Vast.ai Docker Image
#  Base: NVIDIA CUDA 12.8 + cuDNN 9 + Ubuntu 22.04
#  PyTorch 2.7 installed via pip (official torch index)
#
#  Build:
#    docker build -t ltx23:latest .
#
#  Push to Docker Hub (required for Vast.ai custom template):
#    docker tag ltx23:latest <your-dockerhub-user>/ltx23:latest
#    docker push <your-dockerhub-user>/ltx23:latest
#
#  On Vast.ai, set the "On-start script" to:
#    HF_TOKEN=hf_xxx bash /root/setup.sh
#  This handles model downloads separately (models are too large for the image).
# =============================================================================

FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

# ── Labels ────────────────────────────────────────────────────────────────────
LABEL maintainer="thesonicprint" \
      description="LTX-2.3 22B inference — CUDA 12.8, PyTorch 2.7, Python 3.12" \
      cuda="12.8" \
      pytorch="2.7" \
      python="3.12"

# ── System ────────────────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/root/.local/bin:$PATH"

RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends software-properties-common gpg-agent && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
        python3.12 python3.12-dev python3.12-venv \
        git git-lfs curl wget ffmpeg openssh-server \
        libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
        build-essential ca-certificates aria2 \
    && rm -rf /var/lib/apt/lists/* \
    && python3.12 -m ensurepip --upgrade \
    && git lfs install --skip-smudge \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/pip     pip     /usr/local/bin/pip3.12 1 \
    && ln -s libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so

# ── SSH Configuration for Vast.ai ──────────────────────────────────────────────
RUN mkdir /var/run/sshd && \
    echo 'root:root' | chpasswd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config && \
    sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

# ── uv (fast Python package manager) ─────────────────────────────────────────
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# ── PyTorch is installed via uv during the Python dependency sync step ──────

# ── Clone the LTX-2 repo (code only — no weights) ─────────────────────────────
WORKDIR /workspace
RUN git clone https://github.com/Sonicprint/LTX-2.git /workspace/LTX-2

# ── Install Python dependencies via uv ───────────────────────────────────────
WORKDIR /workspace/LTX-2
# --frozen: respect lockfile exactly, as documented in README (uv sync --frozen)
# --extra xformers: installs attention optimizations (recommended by README)
RUN uv sync --frozen --extra xformers --no-progress || \
    pip install -e packages/ltx-core -e packages/ltx-pipelines

# ── Checkpoint directory placeholders (models downloaded at runtime) ──────────
RUN mkdir -p /workspace/LTX-2/checkpoints/loras

# ── Copy setup script for model downloads at runtime ─────────────────────────
COPY setup.sh /root/setup.sh
RUN chmod +x /root/setup.sh

# ── Environment for inference runtime ────────────────────────────────────────
ENV LTX_CHECKPOINTS_DIR=/workspace/LTX-2/checkpoints \
    LTX_PACKAGES_DIR=/workspace/LTX-2/packages \
    CUDA_VISIBLE_DEVICES=0

WORKDIR /workspace/LTX-2

# ── SSH Port Expose ──────────────────────────────────────────────────────────
EXPOSE 22

# ── Startup Script ───────────────────────────────────────────────────────────
RUN echo '#!/bin/bash\n/usr/sbin/sshd\nexec /bin/bash "$@"' > /start.sh && \
    chmod +x /start.sh

# ── Default: start script (Vast.ai runs its own on-start script after boot) ──
CMD ["/start.sh"]
