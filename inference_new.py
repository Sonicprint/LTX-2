"""
LTX-2.3 Local Inference Script
================================
Designed for Lightning.ai L40S (48GB VRAM) and containerised Vast.ai deployment.

Expected directory layout (matches your Lightning.ai studio):

  LTX-2/                                             ← repo root, run from here
  ├── inference.py                                   ← THIS FILE
  ├── checkpoints/
  │   ├── gemma-3-12b-it-qat-q4_0-unquantized/      ← local Gemma weights
  │   ├── ltx-2.3-22b-distilled.safetensors
  │   ├── ltx-2.3-spatial-upscaler-x2-1.0.safetensors
  │   ├── ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors  (optional)
  │   ├── ltx-2.3-temporal-upscaler-x2-2.0.safetensors   (optional)
  │   └── loras/
  └── packages/
      ├── ltx-core/src/
      └── ltx-pipelines/src/

All weights are resolved locally — zero HuggingFace downloads at runtime.

Supports:
  - Text-to-Video
  - Image-to-Video  (first frame conditioning)
  - Interpolation   (first + last frame)
  - Audio sync      (user-provided .wav / .mp3 / .flac)

────────────────────────────────────────────────────────────────────────
Usage (run from inside LTX-2/):

  # Text-to-video
  python inference.py \\
    --prompt "A cinematic aerial shot of misty mountains at dawn" \\
    --resolution 16:9 --duration 6 --output out.mp4

  # Image-to-video, portrait, 8 s, fixed seed
  python inference.py \\
    --prompt "The woman slowly turns to face the camera" \\
    --first-frame ./assets/portrait.png \\
    --resolution 9:16 --duration 8 --seed 42 --output out.mp4

  # Frame interpolation
  python inference.py \\
    --prompt "Smooth scene transition" \\
    --first-frame ./assets/start.png --end-frame ./assets/end.png \\
    --duration 6 --output out.mp4

  # Image-to-video + audio sync
  python inference.py \\
    --prompt "A musician playing piano, fingers moving expressively" \\
    --first-frame ./assets/musician.png --audio ./assets/piano.wav \\
    --duration 10 --output out.mp4

  # Skip prompt enhancement (faster — no Gemma rewrite)
  python inference.py --prompt "..." --no-enhance-prompt --output out.mp4

  # BF16 instead of FP8 (needs ~44 GB VRAM — still fits L40S)
  python inference.py --prompt "..." --no-quantize --output out.mp4
────────────────────────────────────────────────────────────────────────
"""

import argparse
import logging
import os
import random
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

# ─────────────────────────────────────────────────────────────────────────────
# Resolve repo root and package/checkpoint paths
#
# This file lives at:  <repo_root>/inference.py
# Packages at:         <repo_root>/packages/ltx-{core,pipelines}/src
# Checkpoints at:      <repo_root>/checkpoints/
# ─────────────────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).parent.resolve()

# Allow env-var overrides for containerised deployments on Vast.ai
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
# Local checkpoint paths — all resolved from checkpoints/ directory
# ─────────────────────────────────────────────────────────────────────────────
GEMMA_ROOT     = _CHECKPOINTS_DIR / "gemma-3-12b-it-qat-q4_0-unquantized"
CKPT_DISTILLED = _CHECKPOINTS_DIR / "ltx-2.3-22b-dev-fp8.safetensors" # use fp8 or "ltx-2.3-22b-distilled.safetensors"
CKPT_UPSCALER  = _CHECKPOINTS_DIR / "ltx-2.3-spatial-upscaler-x2-1.0.safetensors" # use x2 or "ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors"   


def _check_path(p: Path, label: str) -> None:
    """Warn at import time if a required path is missing."""
    if not p.exists():
        print(
            f"[ERROR] {label} not found: {p}\n"
            f"        Set LTX_CHECKPOINTS_DIR if your checkpoints are elsewhere.",
            file=sys.stderr,
        )


_check_path(GEMMA_ROOT,     "Gemma directory")
_check_path(CKPT_DISTILLED, "Distilled transformer checkpoint")
_check_path(CKPT_UPSCALER,  "Spatial upscaler checkpoint")

# ─────────────────────────────────────────────────────────────────────────────
# LTX imports  (after sys.path is configured)
# ─────────────────────────────────────────────────────────────────────────────
from ltx_core.model.video_vae import TilingConfig          # noqa: E402
from ltx_core.quantization import QuantizationPolicy       # noqa: E402
from ltx_pipelines.distilled import DistilledPipeline      # noqa: E402
from ltx_pipelines.utils import ModelLedger                # noqa: E402
from ltx_pipelines.utils.helpers import generate_enhanced_prompt  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ltx-inference")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
MAX_SEED         = np.iinfo(np.int32).max
DEFAULT_FPS      = 24.0
DEFAULT_AUDIO_SR = 48_000

RESOLUTION_MAP = {
    "16:9": (768, 512),   # (width, height)
    "1:1":  (512, 512),
    "9:16": (512, 768),
}


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def calc_frames(duration: float, fps: float = DEFAULT_FPS) -> int:
    """Return num_frames = 8k+1 with frames >= 9  (LTX requirement)."""
    raw = int(duration * fps) + 1
    raw = max(raw, 9)
    k   = (raw - 1 + 7) // 8
    return k * 8 + 1


def load_image_as_path(src) -> str:
    """Accept a file path (str/Path) or PIL Image; return an on-disk path string."""
    if isinstance(src, (str, Path)) and Path(src).exists():
        return str(src)
    try:
        from PIL import Image as PILImage
        if isinstance(src, PILImage.Image):
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            src.save(tmp.name)
            return tmp.name
    except ImportError:
        pass
    raise ValueError(f"Cannot resolve image source: {src!r}")


def match_audio_to_duration(
    audio_path:     str,
    target_seconds: float,
    target_sr:      int  = DEFAULT_AUDIO_SR,
    to_mono:        bool = True,
    pad_mode:       str  = "silence",
    device:         str  = "cuda",
) -> tuple[torch.Tensor, int]:
    """
    Load audio, resample, mono-mix, then trim or silence-pad to target_seconds.
    Returns (waveform [1, T] float32, sample_rate).
    """
    wav, sr = torchaudio.load(audio_path)          # [C, T] float32 CPU

    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr  = target_sr

    if to_mono and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    target_len = int(round(target_seconds * sr))
    cur_len    = wav.shape[-1]

    if cur_len > target_len:
        wav = wav[..., :target_len]
    elif cur_len < target_len:
        pad_len = target_len - cur_len
        if pad_mode == "repeat" and cur_len > 0:
            reps = (target_len + cur_len - 1) // cur_len
            wav  = wav.repeat(1, reps)[..., :target_len]
        else:
            wav = F.pad(wav, (0, pad_len))

    return wav.to(device, non_blocking=True), sr


def encode_text_simple(
    text_encoder,
    model_ledger,
    prompt: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Encode a text prompt → (video_context, audio_context) tensors.
    The embeddings_processor is created and immediately freed each call.
    """
    hidden_states, attention_mask = text_encoder.encode(prompt)
    embeddings_processor = model_ledger.gemma_embeddings_processor()
    result = embeddings_processor.process_hidden_states(hidden_states, attention_mask)
    del embeddings_processor
    return result.video_encoding, result.audio_encoding


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class LTXInferencePipeline:
    """
    LTX-2.3 single-GPU inference wrapper — fully local, no network calls.

    VRAM budget on L40S (48 GB):
      Transformer  22B @ FP8   ≈ 22 GB
      VAE                      ≈  4 GB
      Gemma text enc (Q4)      ≈  8 GB
      Activations / buffers    ≈  8 GB
      ─────────────────────────────────
      Total                    ≈ 42 GB  ✓  (6 GB headroom)

    With --no-quantize the transformer is in BF16 (~44 GB) — still fits the
    L40S but leaves very little headroom for longer clips.
    """

    def __init__(self, device: str = "cuda", quantize: bool = True):
        self.device        = torch.device(device)
        self._quantization = (
            QuantizationPolicy.fp8_cast()
            if quantize
            else QuantizationPolicy.no_quantization()
        )
        self._text_encoder = None
        self._model_ledger = None
        self._pipeline     = None
        self._load_models()

    # ── private ───────────────────────────────────────────────────────────────

    def _load_models(self):
        t0 = time.time()
        log.info("=" * 60)
        log.info("Loading LTX-2.3  (all weights from local disk)")
        log.info(f"  Transformer : {CKPT_DISTILLED.name}")
        log.info(f"  Upscaler    : {CKPT_UPSCALER.name}")
        log.info(f"  Gemma       : {GEMMA_ROOT.name}")
        log.info("=" * 60)

        log.info("Building ModelLedger...")
        self._model_ledger = ModelLedger(
            dtype                  = torch.bfloat16,
            device                 = str(self.device),
            checkpoint_path        = str(CKPT_DISTILLED),
            gemma_root_path        = str(GEMMA_ROOT),
            spatial_upsampler_path = str(CKPT_UPSCALER),
            loras                  = (),
            quantization           = self._quantization,
        )

        log.info("Loading Gemma text encoder...")
        self._text_encoder = self._model_ledger.text_encoder()
        log.info("  Text encoder ready.")

        log.info("Building DistilledPipeline (text encoder managed separately)...")
        self._pipeline = DistilledPipeline(
            device                 = self.device,
            checkpoint_path        = str(CKPT_DISTILLED),
            spatial_upsampler_path = str(CKPT_UPSCALER),
            gemma_root             = None,   # encoding done separately
            loras                  = [],
            quantization           = self._quantization,
        )

        log.info("Pre-loading video encoder and transformer...")
        self._pipeline._video_encoder = self._model_ledger.video_encoder()
        self._pipeline._transformer   = self._model_ledger.transformer()

        log.info(f"All models ready in {time.time() - t0:.1f}s ✓")
        log.info("=" * 60)

    def _encode_prompt(
        self,
        prompt:     str,
        enhance:    bool,
        image_path: Optional[str],
        seed:       int,
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        """Optionally enhance, then encode → (video_ctx, audio_ctx, final_prompt)."""
        final_prompt = prompt

        if enhance:
            log.info("Enhancing prompt with Gemma...")
            final_prompt = generate_enhanced_prompt(
                text_encoder = self._text_encoder,
                prompt       = prompt,
                image_path   = image_path,
                seed         = seed,
            )
            log.info(f"  Enhanced: {final_prompt[:160]}...")

        log.info("Encoding prompt...")
        with torch.inference_mode():
            video_ctx, audio_ctx = encode_text_simple(
                self._text_encoder, self._model_ledger, final_prompt
            )

        return video_ctx.to(self.device), audio_ctx.to(self.device), final_prompt

    # ── public API ────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def generate(
        self,
        prompt:         str,
        output_path:    str,
        first_frame:    Optional[str]  = None,
        end_frame:      Optional[str]  = None,
        audio_path:     Optional[str]  = None,
        duration:       float          = 6.0,
        fps:            float          = DEFAULT_FPS,
        width:          int            = 768,
        height:         int            = 512,
        seed:           Optional[int]  = None,
        randomize_seed: bool           = True,
        enhance_prompt: bool           = True,
    ) -> dict:
        """
        Full LTX-2.3 inference pass. Writes video to output_path.
        Returns summary dict: output_path, seed, elapsed, final_prompt,
        num_frames, width, height, duration.
        """
        if not prompt or not prompt.strip():
            raise ValueError("prompt must not be empty.")

        # Seed
        current_seed = random.randint(0, MAX_SEED) if randomize_seed else int(seed or 42)
        log.info(f"Seed: {current_seed}")

        num_frames = calc_frames(duration, fps)
        log.info(f"Config: {width}×{height}  {duration}s @ {fps} fps  → {num_frames} frames")

        # Conditioning images
        images: list[tuple[str, int, float]] = []
        image_path_for_enhance: Optional[str] = None

        if first_frame is not None:
            img_path = load_image_as_path(first_frame)
            images.append((img_path, 0, 1.0))
            image_path_for_enhance = img_path
            log.info(f"First frame : {img_path}")

        if end_frame is not None:
            end_path = load_image_as_path(end_frame)
            end_idx  = max(0, num_frames - 1)
            images.append((end_path, end_idx, 0.5))
            log.info(f"End frame   : {end_path}  (index {end_idx})")

        t_start = time.time()

        # Phase 1 — prompt encoding
        video_ctx, audio_ctx, final_prompt = self._encode_prompt(
            prompt     = prompt,
            enhance    = enhance_prompt,
            image_path = image_path_for_enhance,
            seed       = current_seed,
        )
        torch.cuda.empty_cache()

        # Phase 2 — audio conditioning
        input_waveform             = None
        input_waveform_sample_rate = None

        if audio_path is not None:
            log.info(f"Loading audio: {audio_path}")
            # Replace text-derived audio context with a neutral (empty-prompt)
            # version so the model attends to the supplied waveform instead.
            with torch.inference_mode():
                _, neutral_audio_ctx = encode_text_simple(
                    self._text_encoder, self._model_ledger, ""
                )
            del audio_ctx
            audio_ctx = neutral_audio_ctx.to(self.device)

            video_seconds = (num_frames - 1) / fps
            input_waveform, input_waveform_sample_rate = match_audio_to_duration(
                audio_path     = audio_path,
                target_seconds = video_seconds,
                target_sr      = DEFAULT_AUDIO_SR,
                to_mono        = True,
                pad_mode       = "silence",
                device         = str(self.device),
            )
            log.info(
                f"Audio ready : {video_seconds:.2f}s  "
                f"({input_waveform.shape[-1]} samples @ {input_waveform_sample_rate} Hz)"
            )

        torch.cuda.empty_cache()

        # Phase 3 — diffusion
        log.info("Running DistilledPipeline...")
        self._pipeline(
            prompt                     = prompt,
            output_path                = output_path,
            seed                       = current_seed,
            height                     = height,
            width                      = width,
            num_frames                 = num_frames,
            frame_rate                 = fps,
            images                     = images if images else None,
            tiling_config              = TilingConfig.default(),
            video_context              = video_ctx,
            audio_context              = audio_ctx,
            input_waveform             = input_waveform,
            input_waveform_sample_rate = input_waveform_sample_rate,
        )

        del video_ctx, audio_ctx
        if input_waveform is not None:
            del input_waveform
        torch.cuda.empty_cache()

        elapsed = time.time() - t_start
        log.info(f"Done in {elapsed:.1f}s → {output_path}")

        return {
            "output_path":  output_path,
            "seed":         current_seed,
            "elapsed":      elapsed,
            "final_prompt": final_prompt,
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
        description="LTX-2.3 local inference — Image-to-Video / Interpolate / Audio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument("--prompt",       required=True,
                   help="Text description of the video.")
    p.add_argument("--output",       default="output.mp4",
                   help="Output .mp4 path. (default: output.mp4)")

    # Conditioning
    p.add_argument("--first-frame",  default=None, metavar="PATH",
                   help="First (or only) conditioning image.")
    p.add_argument("--end-frame",    default=None, metavar="PATH",
                   help="Last frame image — enables Interpolate mode.")

    # Audio
    p.add_argument("--audio",        default=None, metavar="PATH",
                   help="Audio file (.wav/.mp3/.flac) for audio-synced generation.")

    # Resolution
    p.add_argument("--resolution",   default="16:9",
                   choices=list(RESOLUTION_MAP.keys()),
                   help="Aspect ratio preset. (default: 16:9 → 768×512)")
    p.add_argument("--width",        type=int, default=None,
                   help="Override width in pixels.")
    p.add_argument("--height",       type=int, default=None,
                   help="Override height in pixels.")

    # Timing
    p.add_argument("--duration",     type=float, default=6.0,
                   help="Clip length in seconds. (default: 6)")
    p.add_argument("--fps",          type=float, default=DEFAULT_FPS,
                   help=f"Output frame rate. (default: {DEFAULT_FPS})")

    # Seed
    p.add_argument("--seed",         type=int, default=None,
                   help="Fixed seed for reproducibility. Omit for random.")
    p.add_argument("--randomize-seed", action="store_true", default=False,
                   help="Force random seed even when --seed is given.")

    # Quality
    p.add_argument("--enhance-prompt",    dest="enhance_prompt",
                   action="store_true",  default=True,
                   help="Rewrite prompt with Gemma before encoding. (default: on)")
    p.add_argument("--no-enhance-prompt", dest="enhance_prompt",
                   action="store_false",
                   help="Skip Gemma prompt enhancement.")
    p.add_argument("--no-quantize",       dest="quantize",
                   action="store_false",  default=True,
                   help="Use BF16 instead of FP8 (~44 GB VRAM).")
    p.add_argument("--device",       default="cuda",
                   help="Torch device. (default: cuda)")

    # Checkpoint override (useful when running from a different working dir)
    p.add_argument("--checkpoints-dir", default=None, metavar="DIR",
                   help="Path to checkpoints/ directory. "
                        "Overrides LTX_CHECKPOINTS_DIR env var.")

    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()

    # Optional checkpoint dir override from CLI flag
    if args.checkpoints_dir:
        global GEMMA_ROOT, CKPT_DISTILLED, CKPT_UPSCALER
        _ckpt          = Path(args.checkpoints_dir)
        GEMMA_ROOT     = _ckpt / "gemma-3-12b-it-qat-q4_0-unquantized"
        CKPT_DISTILLED = _ckpt / "ltx-2.3-22b-distilled.safetensors"
        CKPT_UPSCALER  = _ckpt / "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"

    # Resolution
    w, h = RESOLUTION_MAP[args.resolution]
    if args.width  is not None: w = args.width
    if args.height is not None: h = args.height

    # Seed
    randomize = args.randomize_seed or (args.seed is None)
    seed      = args.seed if args.seed is not None else 42

    # Mode label
    mode = "Text-to-Video"
    if args.first_frame and args.end_frame:
        mode = "Interpolate"
    elif args.first_frame:
        mode = "Image-to-Video"
    if args.audio:
        mode += " + Audio"

    log.info("=" * 60)
    log.info(f"Mode        : {mode}")
    log.info(f"Prompt      : {args.prompt[:80]}{'...' if len(args.prompt) > 80 else ''}")
    log.info(f"Resolution  : {w}×{h}  ({args.resolution})")
    log.info(f"Duration    : {args.duration}s @ {args.fps} fps")
    log.info(f"Output      : {args.output}")
    log.info(f"Quantize    : {'FP8' if args.quantize else 'BF16'}")
    log.info(f"Enhance     : {args.enhance_prompt}")
    log.info(f"Checkpoints : {_CHECKPOINTS_DIR}")
    log.info("=" * 60)

    pipe = LTXInferencePipeline(device=args.device, quantize=args.quantize)

    try:
        result = pipe.generate(
            prompt          = args.prompt,
            output_path     = args.output,
            first_frame     = args.first_frame,
            end_frame       = args.end_frame,
            audio_path      = args.audio,
            duration        = args.duration,
            fps             = args.fps,
            width           = w,
            height          = h,
            seed            = seed,
            randomize_seed  = randomize,
            enhance_prompt  = args.enhance_prompt,
        )
    except torch.cuda.OutOfMemoryError:
        log.error(
            "CUDA out-of-memory.\n"
            "  → Keep FP8 quantization enabled (default)\n"
            "  → Try a shorter --duration or lower --resolution"
        )
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
    print(f"  Frames   : {result['num_frames']}  ({result['duration']}s @ {args.fps} fps)")
    if result["final_prompt"] != args.prompt:
        print(f"  Enhanced :\n    {result['final_prompt'][:300]}")
    print("=" * 60)


if __name__ == "__main__":
    main()