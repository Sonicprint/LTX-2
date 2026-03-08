# LTX-2.3 Local Inference

Local inference script for [LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) — a 22B DiT audio-video model. Designed for Lightning.ai L40S (48 GB VRAM) and containerised Vast.ai deployment.

---

## Directory Layout

Place `inference_new.py` in the repo root. The script resolves all paths relative to its own location.

```
LTX-2/
├── inference_new.py
├── checkpoints/
│   ├── gemma-3-12b-it-qat-q4_0-unquantized/
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
python inference_new.py \
  --prompt "A cinematic aerial shot of a misty mountain range at dawn" \
  --resolution 16:9 \
  --duration 6 \
  --output out.mp4
```

### Image-to-Video

```bash
python inference_new.py \
  --prompt "The woman slowly turns to face the camera" \
  --first-frame ./assets/portrait.png \
  --resolution 9:16 \
  --duration 8 \
  --seed 42 \
  --output out.mp4
```

### Frame Interpolation

```bash
python inference_new.py \
  --prompt "Smooth transition between two scenes" \
  --first-frame ./assets/start.png \
  --end-frame ./assets/end.png \
  --duration 6 \
  --output out.mp4
```

---

## All Arguments

| Argument | Default | Description |
|---|---|---|
| `--prompt` | *(required)* | Text description of the video |
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

| Preset | Width × Height |
|---|---|
| `16:9` | 768 × 512 |
| `1:1` | 512 × 512 |
| `9:16` | 512 × 768 |

---

## VRAM Budget (L40S, 48 GB)

| Component | Approx. VRAM |
|---|---|
| Transformer 22B @ FP8 | ~22 GB |
| VAE encoder/decoder | ~4 GB |
| Gemma text encoder (Q4) | ~8 GB |
| Activations / buffers | ~8 GB |
| **Total** | **~42 GB** |

This leaves ~6 GB headroom. Using `--no-quantize` (BF16) raises transformer usage to ~44 GB — still fits the L40S but with very little headroom for longer clips.

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