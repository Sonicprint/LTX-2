import os
import torch
import logging
from ltx_core.loader import LoraPathStrengthAndSDOps, LTXV_LORA_COMFY_RENAMING_MAP
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
from ltx_core.components.guiders import MultiModalGuiderParams

def run_inference():
    logging.basicConfig(level=logging.INFO)
    
    # Paths to models
    checkpoint_path = "checkpoints/ltx-2-19b-dev-fp8.safetensors"
    distilled_lora_path = "checkpoints/ltx-2-19b-distilled-lora-384.safetensors"
    upsampler_path = "checkpoints/ltx-2-spatial-upscaler-x2-1.0.safetensors"
    gemma_root = "checkpoints/gemma-3-12b-it-qat-q4_0-unquantized"
    
    # Configure Distilled LoRA for Stage 2
    distilled_lora = [
        LoraPathStrengthAndSDOps(
            distilled_lora_path,
            0.8,
            LTXV_LORA_COMFY_RENAMING_MAP
        ),
    ]
    
    print("Loading pipeline...")
    pipeline = TI2VidTwoStagesPipeline(
        checkpoint_path=checkpoint_path,
        distilled_lora=distilled_lora,
        spatial_upsampler_path=upsampler_path,
        gemma_root=gemma_root,
        loras=[],
        fp8transformer=True, # Enable FP8 for memory efficiency
    )
    
    # Default guidance parameters
    video_guider_params = MultiModalGuiderParams(
        cfg_scale=3.0,
        stg_scale=1.0,
        rescale_scale=0.7,
        modality_scale=3.0,
        stg_blocks=[29],
    )
    
    audio_guider_params = MultiModalGuiderParams(
        cfg_scale=7.0,
        stg_scale=1.0,
        rescale_scale=0.7,
        modality_scale=3.0,
        stg_blocks=[29],
    )
    
    prompt = "A majestic dragon flying over a snowy mountain range, cinematic lighting, high detail"
    output_path = "output_dragon.mp4"
    
    print(f"Generating video for prompt: {prompt}")
    # Using environment variable for memory optimization as recommended in README
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    pipeline(
        prompt=prompt,
        negative_prompt="low quality, blurry, distorted, static",
        seed=42,
        height=512,
        width=768,
        num_frames=121,
        frame_rate=25.0,
        num_inference_steps=40,
        video_guider_params=video_guider_params,
        audio_guider_params=audio_guider_params,
        images=[], # Text-to-video if no images provided
        output_path=output_path,
    )
    
    print(f"Inference complete! Output saved to {output_path}")

if __name__ == "__main__":
    run_inference()
