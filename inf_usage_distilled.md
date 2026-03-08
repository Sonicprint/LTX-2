# LTX-2.3 Distilled Inference Guide

This guide covers the usage of `inference_new_distilled.py`, the optimized script for the 22B distilled model.

## 🚀 Quick Start (T2V)

```bash
uv run python inference_new_distilled.py \
  --prompt "A highly detailed, cinematic aerial shot of a waterfall in a tropical mountain range at dawn, 8k, stable camera, sharp details" \
  --lora ltx-2-19b-ic-lora-detailer.safetensors 0.5 \
  --output restored_quality.mp4
```

## 📸 Image-to-Video (I2V)

To use an image as the starting frame:

```bash
uv run python inference_new_distilled.py \
  --prompt "A sailboat gently rocking on calm blue water, sunset" \
  --first-frame path/to/your_image.png \
  --output i2v_result.mp4
```

## 🎞️ Frame Interpolation

To interpolate between two images:

```bash
uv run python inference_new_distilled.py \
  --prompt "A flower slowly blooming" \
  --first-frame start.png \
  --end-frame end.png \
  --output interpolation.mp4
```

## 🎵 Audio Conditioning (Optional)

The pipeline supports generating video with synchronized audio. While the distilled model focuses on video, it can accept an audio prompt or reference:

```bash
uv run python inference_new_distilled.py \
  --prompt "A thunderstorm over a dark forest" \
  --audio path/to/thunder_reference.wav \
  --output video_with_audio.mp4
```

## 🛠️ Flags Guide

| Flag | Description | Default |
| :--- | :--- | :--- |
| `--prompt` | **Required**. Text description of the video. | - |
| `--output` | Filename for the result. | `output.mp4` |
| `--audio` | Optional path to an audio file for conditioning. | - |
| `--resolution` | Aspect ratio: `16:9`, `1:1`, or `9:16`. | `16:9` |
| `--width` / `--height` | Override resolution with specific pixels. | - |
| `--duration` | Total video length in seconds. | `6.0` |
| `--fps` | Frames per second. | `24.0` |
| `--seed` | Fix the random seed for reproducibility. | - |
| `--randomize-seed` | Generate a new seed for every run. | `False` |
| `--lora` | Path to LoRA + Strength (0.0 - 1.0). Can be used multiple times. | `Detailer @ 0.5` |
| `--no-enhance-prompt` | Skip the Gemma-based prompt rewrite (faster). | `False` |
| `--no-quantize` | Run in **BF16** (unquantized) for highest quality. Fits L40S. | `True` (FP8) |

> [!TIP]
> **BF16 Mode**: If you have 48GB VRAM (like an L40S), use `--no-quantize` to get the best possible quality from the distilled model.
