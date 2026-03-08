import torch

BLOCK_SIZE = 1024


def calculate_weight_float8(target_weights: torch.Tensor, original_weights: torch.Tensor) -> torch.Tensor:
    result = _fused_add_round_launch(target_weights, original_weights, seed=0).to(target_weights.dtype)
    target_weights.copy_(result, non_blocking=True)
    return target_weights


def _fused_add_round_launch(target_weight: torch.Tensor, original_weight: torch.Tensor, seed: int) -> torch.Tensor:
    # Lazy import triton - only available on CUDA platforms
    import triton  # noqa: PLC0415

    from ltx_core.loader.kernels import fused_add_round_kernel  # noqa: PLC0415

    if original_weight.dtype == torch.float8_e4m3fn:
        exponent_bits, mantissa_bits, exponent_bias = 4, 3, 7
    elif original_weight.dtype == torch.float8_e5m2:
        exponent_bits, mantissa_bits, exponent_bias = 5, 2, 15  # noqa: F841
    else:
        raise ValueError("Unsupported dtype")

    if target_weight.dtype != torch.bfloat16:
        raise ValueError("target_weight dtype must be bfloat16")

    # Calculate grid and block sizes
    n_elements = original_weight.numel()
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    try:
        # Launch kernel
        fused_add_round_kernel[grid](
            original_weight,
            target_weight,
            seed,
            n_elements,
            exponent_bias,
            mantissa_bits,
            BLOCK_SIZE,
        )
    except Exception as e:
        # Fallback for Triton compilation errors on architectures (like L40S)
        # where the fp8e4nv triton type alias is not supported or missing.
        if "fp8" in str(e).lower() or "triton" in str(type(e)).lower():
            import logging
            logging.getLogger(__name__).debug(f"Triton FP8 kernel failed: {e}. Falling back to PyTorch native.")
            target_weight.add_(original_weight.to(target_weight.dtype))
        else:
            raise

    return target_weight
