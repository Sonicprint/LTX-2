# LTX-2.3 — Vast.ai Deployment Guide

## Architecture: Why Hybrid (Dockerfile + setup.sh)?

Vast.ai **requires** a Docker image as the base for every instance. But the models are 70–80 GB total — far too large to bake into a Docker image and push to Docker Hub. The correct split is:

```
Dockerfile  →  bakes in: OS, CUDA 12.8, PyTorch 2.7, system tools,
                          repo clone, Python deps
                          (~8–10 GB image, pushable to Docker Hub)

setup.sh    →  runs at first boot via Vast.ai "On-start script":
               downloads all model weights into /workspace/LTX-2/checkpoints/
               (~70–80 GB, stays on the instance disk / Volume)
```

This means you **build and push the Docker image once**, then Vast.ai re-uses it across launches. Model downloads happen on the instance (fast datacenter bandwidth), and persist if you use a Volume.

---

## 1. Choosing the Right Instance

| Requirement | Minimum | Recommended |
|---|---|---|
| **VRAM** | 48 GB | 48–80 GB (L40S, A6000, H100) |
| **RAM** | 64 GB | 128 GB |
| **Disk** | 200 GB | 300 GB (with LoRAs) |
| **CUDA** | 12.8 | 12.8+ |
| **Internet** | 500 Mbps | 1+ Gbps |

> **RTX 4090 (24 GB) — ❌ not enough VRAM.** L40S 48 GB is the minimum that fits comfortably.

---

## 2. Build & Push Your Docker Image

```bash
# From inside LTX-2/  (where Dockerfile lives)
cd /teamspace/studios/this_studio/LTX-2

# Build
docker build -t spltx23:latest .

# Tag and push to Docker Hub
docker tag spltx23:latest thesonicprint/spltx23:latest
docker push thesonicprint/spltx23:latest
```

What the image contains:
- `nvidia/cuda:12.8.0-cudnn9-runtime-ubuntu22.04` base
- PyTorch 2.7.0 (CUDA 12.8 wheel)
- `uv`, `git`, `ffmpeg`, system libs
- Repo cloned at `/workspace/LTX-2` with Python deps installed
- `setup.sh` copied to `/root/setup.sh`
- `LTX_CHECKPOINTS_DIR` and `LTX_PACKAGES_DIR` env vars pre-set

---

## 3. Create the Vast.ai Template

On [cloud.vast.ai](https://cloud.vast.ai) → **Templates** → **Create New**:

| Field | Value |
|---|---|
| **Docker image** | `thesonicprint/spltx23:latest` |
| **On-start script** | `HF_TOKEN=hf_your_token bash /root/setup.sh` |
| **Environment variables** | `HF_TOKEN=hf_your_token` (add as secret) |
| **Disk** | 250 GB |
| **Ports** | 22/tcp |

> **Tip:** Use Vast.ai's secret env vars for `HF_TOKEN` rather than hard-coding in the on-start script — this keeps your token out of logs.

---

## 4. On-Start Script Options

The on-start script runs **every time the instance boots**. `setup.sh` skips files that already exist, so repeat runs are safe and fast.

```bash
# Full setup (all models + LoRAs, ~80 GB download on first boot)
HF_TOKEN=$HF_TOKEN bash /root/setup.sh

# Minimal (distilled model + Gemma only, ~52 GB — faster first boot)
HF_TOKEN=$HF_TOKEN bash /root/setup.sh minimal

# Use the smaller dev-fp8 model instead of distilled (~27 GB saving)
DISTILLED_MODEL=dev-fp8 HF_TOKEN=$HF_TOKEN bash /root/setup.sh minimal
```

---

## 5. Running Inference

```bash
cd /workspace/LTX-2

# Text-to-video
uv run python inference_new.py \
  --prompt "A cinematic aerial shot of a misty mountain range at dawn" \
  --resolution 16:9 --duration 6 --output /workspace/out.mp4

# Image-to-video
uv run python inference_new.py \
  --prompt "The woman slowly turns to face the camera" \
  --first-frame /workspace/portrait.png \
  --resolution 9:16 --duration 8 --seed 42 \
  --output /workspace/out.mp4

# Audio-synced generation
uv run python inference_new.py \
  --prompt "A musician playing piano, fingers moving expressively" \
  --first-frame /workspace/musician.png --audio /workspace/piano.wav \
  --duration 10 --output /workspace/out.mp4
```

### Copy output back locally
```bash
scp -P <port> root@<host>:/workspace/out.mp4 ./out.mp4
```

---

## 6. Persistent Storage

Vast.ai instance disks wipe on destroy. To avoid re-downloading ~80 GB:

| Strategy | Cost | Effort |
|---|---|---|
| **Vast.ai Volume** (best) | ~$0.02/GB/month | Create once, attach at launch |
| **Snapshot template** | Storage cost for disk | Create after first setup |
| **Keep instance running** | Compute cost while idle | Simplest, most expensive |

**Volume workflow:**
1. Create a 250 GB Volume in your Vast.ai dashboard (same region as instance)
2. Mount at `/workspace` when launching the instance
3. Run `setup.sh` once — all models persist across reboots and new instances

---

## 7. VRAM Budget (L40S 48 GB)

| Component | VRAM |
|---|---|
| Transformer 22B @ FP8 | ~22 GB |
| VAE encoder/decoder | ~4 GB |
| Gemma text encoder (Q4) | ~8 GB |
| Activations / buffers | ~8 GB |
| **Total** | **~42 GB** (6 GB headroom) |

`--no-quantize` (BF16) raises transformer to ~44 GB — still fits, minimal headroom.

---

## 8. Download Size Reference

| File | Size | @ 1 Gbps |
|---|---|---|
| `ltx-2.3-22b-distilled.safetensors` | ~43 GB | ~6 min |
| `ltx-2.3-22b-dev-fp8.safetensors` | ~27 GB | ~4 min |
| `gemma-3-12b-it-qat-q4_0-unquantized/` | ~8 GB | ~1 min |
| `ltx-2.3-spatial-upscaler-x2-*` | ~950 MB | ~8 sec |
| All LoRAs | ~10 GB | ~1.5 min |
| **Full setup total** | **~72 GB** | **~10 min** |

---

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| `CUDA out of memory` | Stay on FP8 (default), shorten `--duration` |
| Gemma 401 error | Accept licence at [huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized) first |
| `Package not found` | Env var `LTX_PACKAGES_DIR=/workspace/LTX-2/packages` is pre-set in image |
| `torch.cuda not available` | Confirm instance has CUDA 12.8+ driver; check with `nvidia-smi` |
| `uv: command not found` | Run `export PATH="$HOME/.local/bin:$PATH"` |
| setup.sh re-downloads on restart | Use a Volume mounted at `/workspace` |
