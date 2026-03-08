"""
LTX-2.3 Local Inference Script
================================
Designed for Lightning.ai L40S (48GB VRAM) and containerised Vast.ai deployment.

Expected directory layout (matches your Lightning.ai studio):

  LTX-2/                                             ← repo root, run from here
  ├── inference_new.py                               ← THIS FILE
  ├── checkpoints/
  │   ├── gemma-3-12b-it-qat-q4_0-unquantized/      ← local Gemma weights
  │   ├── ltx-2.3-22b-dev-fp8.safetensors           ← Non-distilled FP8 base model
  │   └── loras/
  └── packages/
      ├── ltx-core/src/
      └── ltx-pipelines/src/

This script wraps the TI2VidOneStagePipeline to properly generate video
using the base non-distilled model (which requires CFG and a full 30-step schedule).
It fixes the "pure noise" issue caused by using a distilled noise schedule
on a non-distilled model.

All weights are resolved locally — zero HuggingFace downloads at runtime.

Supports:
  - Text-to-Video
  - Image-to-Video  (first frame conditioning)
  - Interpolation   (first + last frame)

────────────────────────────────────────────────────────────────────────
Usage (run from inside LTX-2/):

  # Text-to-video
  python inference_new.py \\
    --prompt "A cinematic aerial shot of misty mountains at dawn" \\
    --resolution 16:9 --duration 6 --output out.mp4

  # Image-to-video, portrait, 8 s, fixed seed
  python inference_new.py \\
    --prompt "The woman slowly turns to face the camera" \\
    --first-frame ./assets/portrait.png \\
    --resolution 9:16 --duration 8 --seed 42 --output out.mp4

  # Frame interpolation
  python inference_new.py \\
    --prompt "Smooth scene transition" \\
    --first-frame ./assets/start.png --end-frame ./assets/end.png \\
    --duration 6 --output out.mp4

  # Skip prompt enhancement (faster — no Gemma rewrite)
  python inference_new.py --prompt "..." --no-enhance-prompt --output out.mp4

  # BF16 instead of FP8 (needs ~44 GB VRAM — still fits L40S)
  python inference_new.py --prompt "..." --no-quantize --output out.mp4
────────────────────────────────────────────────────────────────────────
"""

import argparse
import logging
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# ─────────────────────────────────────────────────────────────────────────────
# Resolve repo root and package/checkpoint paths
# ─────────────────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).parent.resolve()

_CHECKPOINTS_DIR = Path(
    os.environ.get("LTX_CHECKPOINTS_DIR", str(_REPO_ROOT / "checkpoints"))
)
_PACKAGES_DIR = Path(
    os.environ.get("LTX_PACKAGES_DIR", str(_REPO_ROOT / "packages"))
)

_LTX_CORE      = _PACKAGES_DIR / "ltx-core"      / "src"
_LTX_PIPELINES = _PACKAGES_DIR / "ltx-pipelines" / "src"

for _pkg in (_LTX_CORE, _LTX_PIPELINES):
    if not _pkg.exists():
        print(
            f"[ERROR] Package not found: {_pkg}\n"
            f"        Expected: {_REPO_ROOT}/packages/ltx-{{core,pipelines}}/src\n"
            f"        Override with env var LTX_PACKAGES_DIR.",
            file=sys.stderr,
        )
    sys.path.insert(0, str(_pkg))

# ─────────────────────────────────────────────────────────────────────────────
from ltx_core.quantization import QuantizationPolicy                          # noqa: E402
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline           # noqa: E402
from ltx_core.loader import LoraPathStrengthAndSDOps, LTXV_LORA_COMFY_RENAMING_MAP  # noqa: E402
from ltx_pipelines.utils.args import ImageConditioningInput                   # noqa: E402
from ltx_pipelines.utils.media_io import encode_video                         # noqa: E402
from ltx_pipelines.utils.constants import LTX_2_3_PARAMS, DISTILLED_SIGMA_VALUES, DEFAULT_NEGATIVE_PROMPT

# ─────────────────────────────────────────────────────────────────────────────
# Local checkpoint paths — all resolved from checkpoints/ directory
# ─────────────────────────────────────────────────────────────────────────────
GEMMA_ROOT = _CHECKPOINTS_DIR / "gemma-3-12b-it-qat-q4_0-unquantized"
CKPT_BASE  = _CHECKPOINTS_DIR / "ltx-2.3-22b-dev-fp8.safetensors"
UPSCALER   = _CHECKPOINTS_DIR / "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
DIST_LORA  = _CHECKPOINTS_DIR / "ltx-2.3-22b-distilled-lora-384.safetensors"

def _check_path(p: Path, label: str) -> None:
    """Warn at import time if a required path is missing."""
    if not p.exists():
        print(
            f"[ERROR] {label} not found: {p}\n"
            f"        Set LTX_CHECKPOINTS_DIR if your checkpoints are elsewhere.",
            file=sys.stderr,
        )

_check_path(GEMMA_ROOT, "Gemma directory")
_check_path(UPSCALER,   "Spatial Upscaler")
_check_path(DIST_LORA,  "Distilled LoRA")
# Check transformer checkpoint after parsing args

# ─────────────────────────────────────────────────────────────────────────────
# Logging & Constants
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ltx-inference")

MAX_SEED     = np.iinfo(np.int32).max
DEFAULT_FPS  = 24.0

RESOLUTION_MAP = {
    "16:9": (768, 512),
    "1:1":  (512, 512),
    "9:16": (512, 768),
}

def calc_frames(duration: float, fps: float = DEFAULT_FPS) -> int:
    """Return num_frames = 8k+1 with frames >= 9  (LTX requirement)."""
    raw = int(duration * fps) + 1
    raw = max(raw, 9)
    k   = (raw - 1 + 7) // 8
    return k * 8 + 1

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline wrapper
# ─────────────────────────────────────────────────────────────────────────────

class LTXInferencePipeline:
    def __init__(self, device: str = "cuda", quantize: bool = True):
        self.device = torch.device(device)
        quantization = (
            QuantizationPolicy.fp8_cast()
            if quantize
            else QuantizationPolicy.no_quantization()
        )

        t0 = time.time()
        log.info("=" * 60)
        log.info("Loading LTX-2.3  (all weights from local disk)")
        log.info(f"  Transformer : {CKPT_BASE.name}")
        log.info(f"  Upscaler    : {UPSCALER.name}")
        log.info(f"  Dist Lora   : {DIST_LORA.name}")
        log.info(f"  Gemma       : {GEMMA_ROOT.name}")
        log.info("=" * 60)

        distilled_lora_obj = [LoraPathStrengthAndSDOps(str(DIST_LORA), 1.0, LTXV_LORA_COMFY_RENAMING_MAP)]

        self._pipeline = TI2VidTwoStagesPipeline(
            checkpoint_path        = str(CKPT_BASE),
            distilled_lora         = distilled_lora_obj,
            spatial_upsampler_path = str(UPSCALER),
            gemma_root             = str(GEMMA_ROOT),
            loras                  = [],
            device                 = self.device,
            quantization           = quantization,
        )

        log.info(f"All models ready in {time.time() - t0:.1f}s ✓")
        log.info("=" * 60)

    @torch.inference_mode()
    def generate(
        self,
        prompt:         str,
        output_path:    str,
        first_frame:    Optional[str]  = None,
        end_frame:      Optional[str]  = None,
        duration:       float          = 6.0,
        fps:            float          = DEFAULT_FPS,
        width:          int            = 768,
        height:         int            = 512,
        seed:           Optional[int]  = None,
        randomize_seed: bool           = True,
        enhance_prompt: bool           = True,
    ) -> dict:

        if not prompt or not prompt.strip():
            raise ValueError("prompt must not be empty.")

        current_seed = random.randint(0, MAX_SEED) if randomize_seed else int(seed or 42)
        log.info(f"Seed: {current_seed}")

        num_frames = calc_frames(duration, fps)
        log.info(f"Config: {width}×{height}  {duration}s @ {fps} fps  → {num_frames} frames")

        images: list[ImageConditioningInput] = []

        if first_frame is not None:
            images.append(ImageConditioningInput(
                path      = str(first_frame),
                frame_idx = 0,
                strength  = 1.0,
            ))
            log.info(f"First frame : {first_frame}")

        if end_frame is not None:
            end_idx = max(0, num_frames - 1)
            images.append(ImageConditioningInput(
                path      = str(end_frame),
                frame_idx = end_idx,
                strength  = 0.5,
            ))
            log.info(f"End frame   : {end_frame}  (index {end_idx})")

        t_start = time.time()
        log.info("Running TI2VidTwoStagesPipeline (Stage 1 CFG + Stage 2 Distilled Upscale)...")

        video_iterator, audio = self._pipeline(
            prompt              = prompt,
            negative_prompt     = DEFAULT_NEGATIVE_PROMPT,
            seed                = current_seed,
            height              = height,
            width               = width,
            num_frames          = num_frames,
            frame_rate          = fps,
            num_inference_steps = LTX_2_3_PARAMS.num_inference_steps,
            video_guider_params = LTX_2_3_PARAMS.video_guider_params,
            audio_guider_params = LTX_2_3_PARAMS.audio_guider_params,
            images              = images,
            enhance_prompt      = enhance_prompt,
        )

        log.info(f"Encoding video → {output_path}")
        from ltx_core.model.video_vae import get_video_chunks_number, TilingConfig
        chunks = get_video_chunks_number(num_frames, TilingConfig.default())

        encode_video(
            video               = video_iterator,
            fps                 = fps,
            audio               = audio,
            output_path         = output_path,
            video_chunks_number = chunks,
        )

        elapsed = time.time() - t_start
        log.info(f"Done in {elapsed:.1f}s → {output_path}")

        return {
            "output_path":  output_path,
            "seed":         current_seed,
            "elapsed":      elapsed,
            "num_frames":   num_frames,
            "width":        width,
            "height":       height,
            "duration":     duration,
        }

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="LTX-2.3 local inference — Base Model One-Stage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--prompt",       required=True)
    p.add_argument("--model",        default="dev-fp8", choices=["dev-fp8", "distilled"], help="Which model checkpoint to use")
    p.add_argument("--output",       default="output.mp4")
    p.add_argument("--first-frame",  default=None, metavar="PATH")
    p.add_argument("--end-frame",    default=None, metavar="PATH")
    p.add_argument("--resolution",   default="16:9", choices=list(RESOLUTION_MAP.keys()))
    p.add_argument("--width",        type=int, default=None)
    p.add_argument("--height",       type=int, default=None)
    p.add_argument("--duration",     type=float, default=6.0)
    p.add_argument("--fps",          type=float, default=DEFAULT_FPS)
    p.add_argument("--seed",         type=int, default=None)
    p.add_argument("--randomize-seed", action="store_true", default=False)
    p.add_argument("--enhance-prompt",    dest="enhance_prompt", action="store_true",  default=True)
    p.add_argument("--no-enhance-prompt", dest="enhance_prompt", action="store_false")
    p.add_argument("--no-quantize",       dest="quantize",       action="store_false", default=True)
    p.add_argument("--device",       default="cuda")
    p.add_argument("--checkpoints-dir", default=None, metavar="DIR")
    # For backwards compatibility with old commands in readme, ignore audio arg
    p.add_argument("--audio",        default=None, metavar="PATH", help=argparse.SUPPRESS)
    return p

def main():
    parser = build_parser()
    args   = parser.parse_args()

    global GEMMA_ROOT, CKPT_BASE, UPSCALER, DIST_LORA
    _ckpt = Path(args.checkpoints_dir) if args.checkpoints_dir else _CHECKPOINTS_DIR
    GEMMA_ROOT = _ckpt / "gemma-3-12b-it-qat-q4_0-unquantized"
    CKPT_BASE  = _ckpt / f"ltx-2.3-22b-{args.model}.safetensors"
    UPSCALER   = _ckpt / "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
    DIST_LORA  = _ckpt / "ltx-2.3-22b-distilled-lora-384.safetensors"

    if not CKPT_BASE.exists():
        log.error(f"Transformer checkpoint not found: {CKPT_BASE}\n        Set LTX_CHECKPOINTS_DIR if your checkpoints are elsewhere.")
        sys.exit(1)

    w, h = RESOLUTION_MAP[args.resolution]
    if args.width  is not None: w = args.width
    if args.height is not None: h = args.height

    randomize = args.randomize_seed or (args.seed is None)
    seed      = args.seed if args.seed is not None else 42

    mode = "Text-to-Video"
    if args.first_frame and args.end_frame:
        mode = "Interpolate"
    elif args.first_frame:
        mode = "Image-to-Video"

    log.info("=" * 60)
    log.info(f"Mode        : {mode}")
    log.info(f"Prompt      : {args.prompt[:80]}{'...' if len(args.prompt) > 80 else ''}")
    log.info(f"Resolution  : {w}×{h}  ({args.resolution})")
    log.info(f"Duration    : {args.duration}s @ {args.fps} fps")
    log.info(f"Output      : {args.output}")
    log.info(f"Quantize    : {'FP8' if args.quantize else 'BF16'}")
    log.info(f"Enhance     : {args.enhance_prompt}")
    log.info("=" * 60)

    pipe = LTXInferencePipeline(device=args.device, quantize=args.quantize)

    try:
        result = pipe.generate(
            prompt          = args.prompt,
            output_path     = args.output,
            first_frame     = args.first_frame,
            end_frame       = args.end_frame,
            duration        = args.duration,
            fps             = args.fps,
            width           = w,
            height          = h,
            seed            = seed,
            randomize_seed  = randomize,
            enhance_prompt  = args.enhance_prompt,
        )
    except torch.cuda.OutOfMemoryError:
        log.error("CUDA out-of-memory.")
        sys.exit(1)
    except Exception:
        log.error("Generation failed:\n" + traceback.format_exc())
        sys.exit(1)

    print()
    print("=" * 60)
    print("  Generation complete!")
    print(f"  Output   : {result['output_path']}")
    print(f"  Seed     : {result['seed']}")
    print(f"  Elapsed  : {result['elapsed']:.1f}s")
    print("=" * 60)

if __name__ == "__main__":
    main()