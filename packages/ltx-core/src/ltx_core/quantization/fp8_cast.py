import torch

from ltx_core.loader.module_ops import ModuleOps
from ltx_core.loader.sd_ops import KeyValueOperationResult, SDOps
from ltx_core.model.transformer.model import LTXModel

from ltx_core.quantization.fp8_utils import _fused_add_round_launch


def _naive_weight_or_bias_downcast(key: str, value: torch.Tensor) -> list[KeyValueOperationResult]:
    """
    Downcast the weight or bias to the float8_e4m3fn dtype.
    """
    return [KeyValueOperationResult(key, value.to(dtype=torch.float8_e4m3fn))]


def _upcast_and_round(
    weight: torch.Tensor, dtype: torch.dtype, with_stochastic_rounding: bool = False, seed: int = 0
) -> torch.Tensor:
    """
    Upcast the weight to the given dtype and optionally apply stochastic rounding.
    Input weight needs to have float8_e4m3fn or float8_e5m2 dtype.
    """
    if not with_stochastic_rounding:
        return weight.to(dtype)
    return _fused_add_round_launch(torch.zeros_like(weight, dtype=dtype), weight, seed)


def _replace_fwd_with_upcast(layer: torch.nn.Linear, with_stochastic_rounding: bool = False, seed: int = 0) -> None:
    """
    Replace linear.forward and rms_norm.forward with a version that:
      - upcasts weight and bias to input's dtype
      - returns F.linear or F.rms_norm calculated in that dtype
    """

    layer.original_forward = layer.forward

    def new_linear_forward(*args, **_kwargs) -> torch.Tensor:
        # assume first arg is the input tensor
        x = args[0]
        w_up = _upcast_and_round(layer.weight, x.dtype, with_stochastic_rounding, seed)
        b_up = None

        if layer.bias is not None:
            b_up = _upcast_and_round(layer.bias, x.dtype, with_stochastic_rounding, seed)

        return torch.nn.functional.linear(x, w_up, b_up)

    layer.forward = new_linear_forward


def _amend_forward_with_upcast(
    model: torch.nn.Module, with_stochastic_rounding: bool = False, seed: int = 0
) -> torch.nn.Module:
    """
    Replace the forward method of the model's Linear and RMSNorm layers to forward
    with upcast and optional stochastic rounding.
    """
    for m in model.modules():
        if isinstance(m, (torch.nn.Linear)):
            _replace_fwd_with_upcast(m, with_stochastic_rounding, seed)
    return model


TRANSFORMER_LINEAR_DOWNCAST_MAP = (
    SDOps("TRANSFORMER_LINEAR_DOWNCAST_MAP")
    .with_kv_operation(
        key_prefix="transformer_blocks.", key_suffix=".to_q.weight", operation=_naive_weight_or_bias_downcast
    )
    .with_kv_operation(
        key_prefix="transformer_blocks.", key_suffix=".to_q.bias", operation=_naive_weight_or_bias_downcast
    )
    .with_kv_operation(
        key_prefix="transformer_blocks.", key_suffix=".to_k.weight", operation=_naive_weight_or_bias_downcast
    )
    .with_kv_operation(
        key_prefix="transformer_blocks.", key_suffix=".to_k.bias", operation=_naive_weight_or_bias_downcast
    )
    .with_kv_operation(
        key_prefix="transformer_blocks.", key_suffix=".to_v.weight", operation=_naive_weight_or_bias_downcast
    )
    .with_kv_operation(
        key_prefix="transformer_blocks.", key_suffix=".to_v.bias", operation=_naive_weight_or_bias_downcast
    )
    .with_kv_operation(
        key_prefix="transformer_blocks.", key_suffix=".to_out.0.weight", operation=_naive_weight_or_bias_downcast
    )
    .with_kv_operation(
        key_prefix="transformer_blocks.", key_suffix=".to_out.0.bias", operation=_naive_weight_or_bias_downcast
    )
    .with_kv_operation(
        key_prefix="transformer_blocks.", key_suffix="ff.net.0.proj.weight", operation=_naive_weight_or_bias_downcast
    )
    .with_kv_operation(
        key_prefix="transformer_blocks.", key_suffix="ff.net.0.proj.bias", operation=_naive_weight_or_bias_downcast
    )
    .with_kv_operation(
        key_prefix="transformer_blocks.", key_suffix="ff.net.2.weight", operation=_naive_weight_or_bias_downcast
    )
    .with_kv_operation(
        key_prefix="transformer_blocks.", key_suffix="ff.net.2.bias", operation=_naive_weight_or_bias_downcast
    )
)

UPCAST_DURING_INFERENCE = ModuleOps(
    name="upcast_fp8_during_linear_forward",
    matcher=lambda model: isinstance(model, LTXModel),
    mutator=lambda model: _amend_forward_with_upcast(model, False),
)


class UpcastWithStochasticRounding(ModuleOps):
    """
    ModuleOps for upcasting the model's float8_e4m3fn weights and biases to the bfloat16 dtype
    and applying stochastic rounding during linear forward.
    """

    def __new__(cls, seed: int = 0):
        return super().__new__(
            cls,
            name="upcast_fp8_during_linear_forward_with_stochastic_rounding",
            matcher=lambda model: isinstance(model, LTXModel),
            mutator=lambda model: _amend_forward_with_upcast(model, True, seed),
        )
