# LTX-2.3 Local Inference

Local inference script for [LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) — a 22B DiT audio-video model. Designed for Lightning.ai L40S (48 GB VRAM) and containerised Vast.ai deployment.

---

## Directory Layout

Place `inference_new.py` in the repo root. The script resolves all paths relative to its own location.

```
LTX-2/
├── inference_new.py
├── inference_new_distilled.py
├── checkpoints/
│   ├── gemma-3-12b-it-qat-q4_0-unquantized/
│   ├── ltx-2.3-22b-dev.safetensors
│   ├── ltx-2.3-22b-distilled.safetensors
│   ├── ltx-2.3-spatial-upscaler-x2-1.0.safetensors
│   ├── ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors   (optional)
│   ├── ltx-2.3-temporal-upscaler-x2-2.0.safetensors    (optional)
│   └── loras/
└── packages/
    ├── ltx-core/src/
    └── ltx-pipelines/src/
```

All weights are loaded from local disk — no HuggingFace downloads at runtime.

---

## Modes

| Mode | Inputs |
|---|---|
| Text-to-Video | `--prompt` only |
| Image-to-Video | `--prompt` + `--first-frame` |
| Interpolation | `--prompt` + `--first-frame` + `--end-frame` |

---

## Usage

Run all commands from inside `LTX-2/`.

### Text-to-Video

```bash
uv run python inference_new.py \
  --prompt "A cinematic aerial shot of a misty mountain range at dawn" \
  --resolution 16:9 \
  --duration 6 \
  --output out.mp4
```

### Image-to-Video

```bash
uv run python inference_new.py \
  --prompt "The woman slowly turns to face the camera" \
  --first-frame ./assets/portrait.png \
  --resolution 9:16 \
  --duration 8 \
  --seed 42 \
  --output out.mp4
```

### Frame Interpolation

```bash
uv run python inference_new.py \
  --prompt "Smooth transition between two scenes" \
  --first-frame ./assets/start.png \
  --end-frame ./assets/end.png \
  --duration 6 \
  --output out.mp4
```

---

## Distilled Inference (Ultra Fast)

For the fastest possible inference (8 steps Stage 1, 4 steps Stage 2), use `inference_new_distilled.py`. This script is optimized for the distilled model and uses a predefined noise schedule.

### Text-to-Video (Distilled)

```bash
uv run python inference_new_distilled.py \
  --prompt "A cinematic aerial shot of a misty mountain range at dawn" \
  --resolution 16:9 \
  --duration 6 \
  --output out_distilled.mp4
```

### Image-to-Video (Distilled)

```bash
python inference_new_distilled.py \
  --prompt "The woman slowly turns to face the camera" \
  --first-frame ./assets/portrait.png \
  --resolution 9:16 \
  --duration 8 \
  --output out_distilled.mp4
```

---

## All Arguments

| Argument | Default | Description |
|---|---|---|
| `--prompt` | *(required)* | Text description of the video |
| `--model` | `distilled` | Which checkpoint to use: `dev` or `distilled` |
| `--output` | `output.mp4` | Output file path |
| `--first-frame` | None | First (or only) conditioning image |
| `--end-frame` | None | Last frame image — enables Interpolate mode |
| `--resolution` | `16:9` | Aspect ratio preset: `16:9`, `1:1`, `9:16` |
| `--width` | None | Override width in pixels (ignores `--resolution`) |
| `--height` | None | Override height in pixels (ignores `--resolution`) |
| `--duration` | `6.0` | Clip length in seconds |
| `--fps` | `24.0` | Output frame rate |
| `--seed` | random | Fixed seed for reproducibility |
| `--randomize-seed` | off | Force random seed even when `--seed` is set |
| `--enhance-prompt` | on | Rewrite prompt with Gemma before encoding |
| `--no-enhance-prompt` | — | Skip Gemma prompt enhancement (faster) |
| `--no-quantize` | — | Use BF16 instead of FP8 (~44 GB VRAM) |
| `--device` | `cuda` | Torch device string |
| `--checkpoints-dir` | `./checkpoints` | Override path to checkpoints directory |

### Resolution Presets

| Preset | Resolution | Aspect | Notes |
| :--- | :--- | :--- | :--- |
| `16:9` | 768 × 512 | Wide | Standard cinematic (Default) |
| `1:1` | 512 × 512 | Square | Good for social media |
| `9:16` | 512 × 768 | Tall | Portrait / TikTok |

---

## VRAM Budget (L40S, 48 GB)

| Component | Approx. VRAM |
|---|---|
| Transformer 22B @ FP8 | ~22 GB |
| VAE encoder/decoder | ~4-12 GB* |
| Gemma text encoder (Q4) | ~8 GB |
| Activations / buffers | ~8-12 GB* |
| **Total** | **~42-48 GB** |

*\*Varies significantly with resolution and duration. Longer clips or higher resolutions push the L40S to its limit.*

This leaves ~6 GB headroom. Using `--no-quantize` (BF16) raises transformer usage to ~44 GB — still fits the L40S but with very little headroom for longer clips.

---

## Setup Downloads

By default, `setup.sh` downloads **both** the `dev-fp8` and `distilled` model checkpoints.
You can control this behavior by setting the `DISTILLED_MODEL` environment variable before running `setup.sh`:
- `export DISTILLED_MODEL=both` (default): downloads both models.
- `export DISTILLED_MODEL=dev`: downloads only the dev model.
- `export DISTILLED_MODEL=distilled`: downloads only the distilled model.

---

## Environment Variable Overrides

For Vast.ai containers or non-standard layouts, override paths without touching the script:

```bash
export LTX_CHECKPOINTS_DIR=/workspace/checkpoints
export LTX_PACKAGES_DIR=/workspace/packages
python inference.py --prompt "..."
```

Or use the CLI flag:

```bash
python inference.py \
  --prompt "..." \
  --checkpoints-dir /workspace/checkpoints
```

---

## Tips

- **Faster runs:** use `--no-enhance-prompt` to skip Gemma rewriting. Useful when your prompt is already detailed.
- **Reproducibility:** set `--seed 42` (or any fixed integer) to get the same output across runs.
- **OOM errors:** stay on FP8 (default), shorten `--duration`, or lower `--resolution`.
- **Interpolation strength:** the end frame is conditioned at weight `0.5` by default. This is hardcoded in the pipeline; adjust in `inference_new.py` if needed.

---

### Temporal Upscaling (Smoother Motion)

The temporal upscaler (`ltx-2.3-temporal-upscaler-x2-1.0.safetensors`) doubles the frame count of your video, turning 121 frames into 241 frames. This results in much smoother motion and improved temporal consistency.

**Usage with Distilled Script:**

```bash
uv run python inference_new_distilled.py \
  --prompt "A waterfall in a lush jungle, slow motion" \
  --duration 2 \
  --temporal-upscaler ltx-2.3-temporal-upscaler-x2-1.0.safetensors \
  --output smooth_waterfall.mp4
```

> [!TIP]
> Use the temporal upscaler for scenes with complex motion or when you want a "slow-motion" high-frame-rate feel. Combining this with the **Detailer LoRA** provides the absolute highest fidelity output within VRAM limits.

**Ultimate Quality Example (Distilled):**
```bash
uv run python inference_new_distilled.py \
  --prompt "A cinematic slow-motion waterfall, 8k, highly detailed" \
  --lora ltx-2-19b-ic-lora-detailer.safetensors 1.0 \
  --temporal-upscaler ltx-2.3-temporal-upscaler-x2-1.0.safetensors \
  --resolution 16:9 \
  --output ultimate_quality.mp4
```

---

## Improving Realism

To get the most photorealistic results from LTX-2.3:

### 1. Use the Detailer LoRA
The `ltx-2-19b-ic-lora-detailer.safetensors` LoRA significantly improves fine details and textures. You can use it with both the standard and distilled scripts.

**Standard:**
```bash
uv run python inference_new.py --prompt "..." --lora ltx-2-19b-ic-lora-detailer.safetensors 1.0
```

**Distilled (Ultra Fast):**
```bash
uv run python inference_new_distilled.py --prompt "..." --lora ltx-2-19b-ic-lora-detailer.safetensors 1.0
```

### 2. Prompt Engineering
The model responds well to descriptive quality keywords. Try adding these to your prompt:
- `cinematic, photorealistic, 8k, highly detailed, sharp focus, natural lighting`
- `masterpiece, professional photography, realistic textures`

### 3. Model Trade-offs
- **`dev` model (`inference_new.py`):** Highest quality, supports CFG (guidance scale), but slower (30+ steps).
- **`distilled` model (`inference_new_distilled.py`):** Extremely fast (8+4 steps), but might have slightly less fine detail than the base model. Using the Detailer LoRA helps bridge this gap.

### 4. Resolution
Higher resolutions generally look more "real" as the model has more pixels to describe textures. Stick to the `16:9` preset (768x512) or higher if VRAM allows.