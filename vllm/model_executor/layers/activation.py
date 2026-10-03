# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom activation functions."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.distributed import (
    divide,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.utils import set_weight_attrs
from vllm.triton_utils import tl, triton
from vllm.utils.collection_utils import LazyDict

logger = init_logger(__name__)


@triton.jit
def _swiglustep_and_mul_kernel(
    o_ptr,
    o_stride,
    x_ptr,
    x_stride,
    limit: tl.constexpr,
    d: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    i = tl.program_id(axis=0).to(tl.int64)
    j = tl.program_id(axis=1)
    o_row_ptr = o_ptr + o_stride * i
    x_row_ptr = x_ptr + x_stride * i
    offsets = j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < d

    gate = tl.load(x_row_ptr + offsets, mask=mask).to(tl.float32)
    up = tl.load(x_row_ptr + offsets + d, mask=mask).to(tl.float32)

    gate_silu = tl.sigmoid(gate) * gate
    gate_clamped = tl.minimum(gate_silu, limit)
    up_clamped = tl.minimum(tl.maximum(up, -limit), limit)

    result = gate_clamped * up_clamped
    result = result.to(x_ptr.dtype.element_ty)
    tl.store(o_row_ptr + offsets, result, mask=mask)


def swiglustep_and_mul_triton(
    output: torch.Tensor, input: torch.Tensor, limit: float = 7.0
):
    b, n = input.shape
    assert input.ndim == 2
    assert n % 2 == 0
    d = n // 2

    def grid(meta):
        return (b, triton.cdiv(d, meta["BLOCK_SIZE"]))

    _swiglustep_and_mul_kernel[grid](
        output,
        output.stride(0),
        input,
        input.stride(0),
        limit=limit,
        d=d,
        BLOCK_SIZE=1024,
    )


# --8<-- [start:fatrelu_and_mul]
@CustomOp.register("fatrelu_and_mul")
class FatreluAndMul(CustomOp):
    """An activation function for FATReLU.

    The function computes x -> FATReLU(x[:d]) * x[d:] where
    d = x.shape[-1] // 2.
    This is used in openbmb/MiniCPM-S-1B-sft.

    Shapes:
        x: (num_tokens, 2 * d) or (batch_size, seq_len, 2 * d)
        return: (num_tokens, d) or (batch_size, seq_len, d)
    """

    # --8<-- [end:fatrelu_and_mul]

    def __init__(self, threshold: float = 0.0):
        super().__init__()
        self.threshold = threshold
        pass  # Unsupported P4 branch removed.

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        x1 = x[..., :d]
        x2 = x[..., d:]
        x1 = F.threshold(x1, self.threshold, 0.0)
        return x1 * x2


# --8<-- [start:silu_and_mul]
@CustomOp.register("silu_and_mul")
class SiluAndMul(CustomOp):
    """An activation function for SwiGLU.

    The function computes x -> silu(x[:d]) * x[d:] where d = x.shape[-1] // 2.

    Shapes:
        x: (num_tokens, 2 * d) or (batch_size, seq_len, 2 * d)
        return: (num_tokens, d) or (batch_size, seq_len, d)
    """

    # --8<-- [end:silu_and_mul]

    def __init__(self, *, compile_native: bool = True):
        super().__init__(compile_native=compile_native)
        pass  # Unsupported P4 branch removed.

    @staticmethod
    def forward_native(x: torch.Tensor) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward()."""
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]


# --8<-- [start:mul_and_silu]
@CustomOp.register("mul_and_silu")
class MulAndSilu(CustomOp):
    """An activation function for SwiGLU.

    The function computes x -> x[:d] * silu(x[d:]) where d = x.shape[-1] // 2.

    Shapes:
        x: (num_tokens, 2 * d) or (batch_size, seq_len, 2 * d)
        return: (num_tokens, d) or (batch_size, seq_len, d)
    """

    # --8<-- [end:mul_and_silu]

    def __init__(self):
        super().__init__()
        pass  # Unsupported P4 branch removed.

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward()."""
        d = x.shape[-1] // 2
        return x[..., :d] * F.silu(x[..., d:])


# --8<-- [start:gelu_and_mul_sparse]
@CustomOp.register("gelu_and_mul_sparse")
class GeluAndMulSparse(CustomOp):
    """An activation function for GeluAndMulSparse.
    This activation function is used in Gemma3n. It computes:
        up_proj = self.up_proj(x)
        gate_proj = self.gate_proj(x)
        gate_proj = self._gaussian_topk(gate_proj) # sparsity
        activations = self.act_fn(gate_proj) # gelu
        down_proj = self.down_proj(activations * up_proj)
    Shapes:
        x: (num_tokens, 2 * d) or (batch_size, seq_len, 2 * d)
        return: (num_tokens, d) or (batch_size, seq_len, d)
    """

    # --8<-- [end:gelu_and_mul_sparse]

    def __init__(self, activation_sparsity: float, approximate: str = "none"):
        super().__init__()
        # Gelu.
        self.approximate = approximate
        if approximate not in ("none", "tanh"):
            raise ValueError(f"Unknown approximate mode: {approximate}")
        pass  # Unsupported P4 branch removed.

        # Sparsity.
        if activation_sparsity == 0.0:
            raise ValueError("activation_sparsity is 0.0. Please use GeluAndMul.")
        target_sparsity_tensor = torch.tensor(activation_sparsity, dtype=torch.float32)
        normal_dist = torch.distributions.normal.Normal(0, 1)
        self.std_multiplier = normal_dist.icdf(target_sparsity_tensor)

    def _gaussian_topk(self, x: torch.Tensor) -> torch.Tensor:
        """Get % sparse percentile of the Gaussian distribution."""
        # NOTE(rob): for TP>1, we could all-gather to get the means/std.
        # But we do not do this because in expectation they are the same
        # and in practice the eval scores are good without gathering.
        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True, unbiased=False)
        cutoff_x = mean + std * self.std_multiplier
        return nn.functional.relu(x - cutoff_x)

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward()."""
        d = x.shape[-1] // 2
        out = self._gaussian_topk(x[..., :d])
        out = F.gelu(out, approximate=self.approximate)
        return out * x[..., d:]


# --8<-- [start:gelu_and_mul]
@CustomOp.register("gelu_and_mul")
class GeluAndMul(CustomOp):
    """An activation function for GeGLU.

    The function computes x -> GELU(x[:d]) * x[d:] where d = x.shape[-1] // 2.

    Shapes:
        x: (batch_size, seq_len, 2 * d) or (num_tokens, 2 * d)
        return: (batch_size, seq_len, d) or (num_tokens, d)
    """

    # --8<-- [end:gelu_and_mul]

    def __init__(self, approximate: str = "none"):
        super().__init__()
        self.approximate = approximate
        if approximate not in ("none", "tanh"):
            raise ValueError(f"Unknown approximate mode: {approximate}")
        pass  # Unsupported P4 branch removed.
        pass  # Unsupported P4 branch removed.

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward()."""
        # TODO: [ROCm] PyTorch's native GELU with tanh is unstable with torch.compile
        approximate = self.approximate
        pass  # Unsupported P4 branch removed.
        d = x.shape[-1] // 2
        return F.gelu(x[..., :d], approximate=approximate) * x[..., d:]

    def extra_repr(self) -> str:
        return f"approximate={repr(self.approximate)}"


# --8<-- [start:swigluoai_and_mul]
@CustomOp.register("swigluoai_and_mul")
class SwigluOAIAndMul(CustomOp):
    # https://github.com/huggingface/transformers/blob/v4.55.0/src/transformers/models/gpt_oss/modeling_gpt_oss.py#L106-L110
    # --8<-- [end:swigluoai_and_mul]

    def __init__(self, alpha: float = 1.702, limit: float = 7.0):
        super().__init__()
        self.alpha = alpha
        self.limit = limit

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward()."""

        gate, up = x[..., ::2], x[..., 1::2]
        gate = gate.clamp(min=None, max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        glu = gate * torch.sigmoid(gate * self.alpha)
        gated_output = (up + 1) * glu
        return gated_output

    def extra_repr(self) -> str:
        return f"alpha={repr(self.alpha)}, limit={repr(self.limit)}"


# --8<-- [start:swiglustep_and_mul]
@CustomOp.register("swiglustep_and_mul")
class SwigluStepAndMul(CustomOp):
    """An activation function for SwiGLU with clamping.

    Computes x -> silu(x[:d]).clamp(max=limit) * x[d:].clamp(-limit, limit)
    where d = x.shape[-1] // 2.

    Shapes:
        x: (num_tokens, 2 * d) or (batch_size, seq_len, 2 * d)
        return: (num_tokens, d) or (batch_size, seq_len, d)
    """

    def __init__(self, limit: float = 7.0):
        super().__init__()
        if limit is None:
            raise ValueError("SwigluStepAndMul requires limit to be set.")
        self.limit = limit

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward()."""
        gate, up = x.chunk(2, dim=-1)
        gate = F.silu(gate)
        gate = gate.clamp(max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        return gate * up

    def extra_repr(self) -> str:
        return f"limit={repr(self.limit)}"


# --8<-- [start:gelu_new]
@CustomOp.register("gelu_new")
class NewGELU(CustomOp):
    # --8<-- [end:gelu_new]

    def __init__(self):
        super().__init__()
        pass  # Unsupported P4 branch removed.

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward()."""
        c = math.sqrt(2.0 / math.pi)
        return 0.5 * x * (1.0 + torch.tanh(c * (x + 0.044715 * torch.pow(x, 3.0))))


# --8<-- [start:gelu_fast]
@CustomOp.register("gelu_fast")
class FastGELU(CustomOp):
    # --8<-- [end:gelu_fast]

    def __init__(self):
        super().__init__()
        pass  # Unsupported P4 branch removed.

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward()."""
        return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))


# --8<-- [start:quick_gelu]
@CustomOp.register("quick_gelu")
class QuickGELU(CustomOp):
    # https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py#L90
    # --8<-- [end:quick_gelu]

    def __init__(self):
        super().__init__()
        pass  # Unsupported P4 branch removed.

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward()."""
        return x * torch.sigmoid(1.702 * x)


# --8<-- [start:relu2]
@CustomOp.register("relu2")
class ReLUSquaredActivation(CustomOp):
    """
    Applies the relu^2 activation introduced in https://arxiv.org/abs/2109.08668v2
    """

    # --8<-- [end:relu2]

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward()."""
        return torch.square(F.relu(x))


# --8<-- [start:xielu]


class ScaledActivation(nn.Module):
    """An activation function with post-scale parameters.

    This is used for some quantization methods like AWQ.
    """

    def __init__(
        self,
        act_module: nn.Module,
        intermediate_size: int,
        input_is_parallel: bool = True,
        params_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.act = act_module
        self.input_is_parallel = input_is_parallel
        if input_is_parallel:
            tp_size = get_tensor_model_parallel_world_size()
            intermediate_size_per_partition = divide(intermediate_size, tp_size)
        else:
            intermediate_size_per_partition = intermediate_size
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.scales = nn.Parameter(
            torch.empty(intermediate_size_per_partition, dtype=params_dtype)
        )
        set_weight_attrs(self.scales, {"weight_loader": self.weight_loader})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x) / self.scales

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        if self.input_is_parallel:
            tp_rank = get_tensor_model_parallel_rank()
            shard_size = param_data.shape[0]
            start_idx = tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)


_ACTIVATION_REGISTRY = LazyDict(
    {
        "gelu": lambda: nn.GELU(),
        "gelu_fast": lambda: FastGELU(),
        "gelu_new": lambda: NewGELU(),
        "gelu_pytorch_tanh": lambda: (
            (
                # TODO:[ROCm] PyTorch native GELU with tanh is unstable with torch.compile
                logger.warning_once(
                    "[ROCm] PyTorch's native GELU with tanh approximation is unstable. "
                    "Falling back to GELU(approximate='none')."
                ),
                nn.GELU(approximate="none"),
            )[1]
            if False
            else nn.GELU(approximate="tanh")
        ),
        "relu": lambda: nn.ReLU(),
        "relu2": lambda: ReLUSquaredActivation(),
        "silu": lambda: nn.SiLU(),
        "quick_gelu": lambda: QuickGELU(),
        "tanh": lambda: nn.Tanh(),
        "sigmoid": lambda: nn.Sigmoid(),
    }
)


def get_act_fn(act_fn_name: str) -> nn.Module:
    """Get an activation function by name."""
    act_fn_name = act_fn_name.lower()

    if act_fn_name.startswith("torch.nn.modules."):
        activation_name = act_fn_name.split(".")[-1]
        if activation_name == "identity":
            return nn.Identity()
        act_fn_name = activation_name

    if act_fn_name not in _ACTIVATION_REGISTRY:
        raise ValueError(f"Activation function {act_fn_name!r} is not supported.")

    return _ACTIVATION_REGISTRY[act_fn_name]


_ACTIVATION_AND_MUL_REGISTRY = LazyDict(
    {
        "gelu": lambda: GeluAndMul(),
        "silu": lambda: SiluAndMul(),
        "geglu": lambda: GeluAndMul(),
        "swigluoai": lambda *args, **kwargs: SwigluOAIAndMul(*args, **kwargs),
    }
)


def get_act_and_mul_fn(act_fn_name: str) -> nn.Module:
    """Get an activation-and-mul (i.e. SiluAndMul) function by name."""
    act_fn_name = act_fn_name.lower()
    if act_fn_name not in _ACTIVATION_AND_MUL_REGISTRY:
        raise ValueError(f"Activation function {act_fn_name!r} is not supported.")

    return _ACTIVATION_AND_MUL_REGISTRY[act_fn_name]
