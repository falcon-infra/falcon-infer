# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import importlib.metadata
import os
import random
import threading
from collections.abc import Callable, Collection
from functools import lru_cache
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
import numpy.typing as npt
import torch
from packaging import version
from packaging.version import Version
from torch.library import Library, infer_schema

import vllm.envs as envs
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import ModelConfig
    from vllm.sequence import IntermediateTensors
else:
    ModelConfig = object
    IntermediateTensors = object

logger = init_logger(__name__)


STR_DTYPE_TO_TORCH_DTYPE = {
    "float32": torch.float32,
    "half": torch.half,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float": torch.float,
    "fp8": torch.uint8,
    "fp8_e4m3": torch.uint8,
    "fp8_e5m2": torch.uint8,
    "int8": torch.int8,
    "fp8_inc": torch.float8_e4m3fn,
    "fp8_ds_mla": torch.uint8,
}

TORCH_DTYPE_TO_NUMPY_DTYPE = {
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.uint8: np.uint8,
    torch.int32: np.int32,
    torch.int64: np.int64,
}


MODELOPT_TO_VLLM_KV_CACHE_DTYPE_MAP = {
    # TODO: Add more modelopt kv cache dtype
    # mappings here when it supported by some attention backend
    # (for example supports nvfp4).
    "fp8": "fp8_e4m3",
}

T = TypeVar("T")


def is_strictly_contiguous(t: torch.Tensor) -> bool:
    """
    Check if tensor is contiguous AND has no degenerate strides.

    A degenerate stride occurs when a dimension has size 1 but the stride
    doesn't match the canonical contiguous layout. This can cause issues
    in some CUDA kernels that rely on stride values for memory access.

    For a C-contiguous tensor of shape (d0, d1, ..., dn), the expected
    strides are: stride[i] = product(shape[i+1:]) for all i, with stride[-1]=1.

    Example with torch.Size([16, 1, 8, 32]):
        - Canonical strides: (256, 256, 32, 1)
        - Degenerate strides: (256, 1, 32, 1)  # dim=1 has size=1, allowing
                                                  # non-canonical stride in dim=0
    """
    if not t.is_contiguous():
        return False

    # Check that strides match canonical contiguous layout
    shape = t.shape
    strides = t.stride()
    expected_stride = 1
    for i in range(len(shape) - 1, -1, -1):
        if strides[i] != expected_stride:
            return False
        expected_stride *= shape[i]
    return True


@contextlib.contextmanager
def set_default_torch_dtype(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(old_dtype)


@contextlib.contextmanager
def set_default_torch_num_threads(num_threads: int | None = None):
    """
    Sets the default number of threads for PyTorch to the given value.

    `None` means using the value of the environment variable `OMP_NUM_THREADS`
    (or `1` if that is not available).
    """
    if num_threads is None:
        num_threads = 1

        try:
            num_threads = int(os.environ["OMP_NUM_THREADS"])
        except KeyError:
            logger.debug_once(
                "OMP_NUM_THREADS is not set; defaulting Torch threads to %d.",
                num_threads,
            )
        except ValueError:
            logger.warning_once(
                "OMP_NUM_THREADS is invalid; defaulting Torch threads to %d.",
                num_threads,
            )

    old_num_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads)

    try:
        yield
    finally:
        torch.set_num_threads(old_num_threads)


@contextlib.contextmanager
def guard_cuda_initialization():
    """Avoid unexpected CUDA initialization."""

    yield
    return


def get_dtype_size(dtype: torch.dtype) -> int:
    """Get the size of the data type in bytes."""
    return torch.tensor([], dtype=dtype).element_size()


# bool = 0, int = 1, float = 2, complex = 3
def _get_precision_level(dtype: torch.dtype) -> int:
    # NOTE: Complex dtypes return `is_floating_point=False`
    return (dtype != torch.bool) + dtype.is_floating_point + dtype.is_complex * 2


def is_lossless_cast(src_dtype: torch.dtype, tgt_dtype: torch.dtype):
    """
    Test whether it is lossless to cast a tensor from
    `src_dtype` to `tgt_dtype`.
    """
    if src_dtype == tgt_dtype:
        return True

    src_level = _get_precision_level(src_dtype)
    tgt_level = _get_precision_level(tgt_dtype)

    if src_level < tgt_level:
        return True
    if src_level > tgt_level:
        return False

    # Compare integral types
    if not src_dtype.is_floating_point and not src_dtype.is_complex:
        src_info = torch.iinfo(src_dtype)
        tgt_info = torch.iinfo(tgt_dtype)
        return src_info.min >= tgt_info.min and src_info.max <= tgt_info.max

    # Compare floating-point types
    src_info = torch.finfo(src_dtype)
    tgt_info = torch.finfo(tgt_dtype)
    return (
        src_info.min >= tgt_info.min
        and src_info.max <= tgt_info.max
        and src_info.resolution >= tgt_info.resolution
    )


def common_broadcastable_dtype(dtypes: Collection[torch.dtype]):
    """
    Get the common `dtype` where all of the other `dtypes` can be
    cast to it without losing any information.
    """
    return max(
        dtypes,
        key=lambda dtype: sum(is_lossless_cast(dt, dtype) for dt in dtypes),
    )


def get_kv_cache_torch_dtype(
    cache_dtype: str | torch.dtype | None,
    model_dtype: str | torch.dtype | None = None,
) -> torch.dtype:
    if isinstance(cache_dtype, str):
        if cache_dtype == "auto":
            if isinstance(model_dtype, str) and model_dtype in STR_DTYPE_TO_TORCH_DTYPE:
                torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[model_dtype]
            elif isinstance(model_dtype, torch.dtype):
                torch_dtype = model_dtype
            else:
                raise ValueError(f"Invalid model dtype: {model_dtype}")
        elif cache_dtype in STR_DTYPE_TO_TORCH_DTYPE:
            torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_dtype]
        else:
            raise ValueError(f"Invalid kv cache dtype: {cache_dtype}")
    elif isinstance(cache_dtype, torch.dtype):
        torch_dtype = cache_dtype
    else:
        raise ValueError(f"Invalid kv cache dtype: {cache_dtype}")
    return torch_dtype


def get_kv_cache_quant_algo_string(quant_cfg: dict[str, Any]) -> str | None:
    """Get the KV cache quantization algorithm string from the quantization config.

    Maps various FP8 format names to vLLM's standard cache dtype strings.
    Returns None if no kv_cache_quant_algo is specified.
    Returns "auto" if the value is not recognized/supported.
    """
    # Mapping from model config values to vLLM cache_dtype strings

    quant_method = quant_cfg.get("quant_method", "")
    if quant_method.startswith("modelopt"):
        quantization_inner = quant_cfg.get("quantization", quant_cfg)
        # Check if quant config is specified and use kv cache quant algo
        kv_algo = (
            quantization_inner.get("kv_cache_scheme")
            or quant_cfg.get("kv_cache_scheme")
            or quantization_inner.get("kv_cache_quant_algo")
            or quant_cfg.get("kv_cache_quant_algo")
        )
        if isinstance(kv_algo, dict):
            if (
                kv_algo.get("dynamic") is False
                and kv_algo.get("num_bits") == 8
                and kv_algo.get("type") == "float"
            ):
                kv_algo = "fp8"
            else:
                # Unknown/unsupported format - return "auto" as safe fallback
                logger.warning(
                    "WARNING: Unknown kv_cache_quant_algo '%s' in model "
                    "config. Supported values: %s. Falling back to 'auto'.",
                    f"{kv_algo}",
                    list(MODELOPT_TO_VLLM_KV_CACHE_DTYPE_MAP.keys()),
                )
                return "auto"
        if isinstance(kv_algo, str):
            kv_algo_lower = kv_algo.lower()

            # Try to map to vLLM's standard format
            if kv_algo_lower in MODELOPT_TO_VLLM_KV_CACHE_DTYPE_MAP:
                return MODELOPT_TO_VLLM_KV_CACHE_DTYPE_MAP[kv_algo_lower]
            else:
                # Unknown/unsupported format - return "auto" as safe fallback
                logger.warning(
                    "WARNING: Unknown kv_cache_quant_algo '%s' in model "
                    "config. Supported values: %s. Falling back to 'auto'.",
                    kv_algo,
                    list(MODELOPT_TO_VLLM_KV_CACHE_DTYPE_MAP.keys()),
                )
                return "auto"
    return None


def get_kv_cache_quant_algo_dtype(quant_cfg: dict[str, Any]) -> torch.dtype | None:
    """Get the KV cache quantization algorithm dtype from the quantization config."""
    kv_algo_str = get_kv_cache_quant_algo_string(quant_cfg)
    if kv_algo_str is not None and kv_algo_str != "auto":
        # Only convert if we have a valid dtype string (not "auto" fallback)
        return STR_DTYPE_TO_TORCH_DTYPE[kv_algo_str]
    return None


def resolve_kv_cache_dtype_string(
    kv_cache_dtype: str, model_config: ModelConfig
) -> str:
    """Resolve 'auto' kv_cache_dtype to the actual string value from model config.
    Returns the resolved cache_dtype string.
    """
    if kv_cache_dtype != "auto":
        return kv_cache_dtype

    hf_cfg = getattr(model_config, "hf_config", None)
    if hf_cfg is not None:
        quant_cfg = getattr(hf_cfg, "quantization_config", None)
        if quant_cfg is not None:
            kv_algo_str = get_kv_cache_quant_algo_string(quant_cfg)
            if kv_algo_str is not None:
                return kv_algo_str

    # Default to auto (will be handled by downstream code)
    return "auto"


def kv_cache_dtype_str_to_dtype(
    kv_cache_dtype: str, model_config: ModelConfig
) -> torch.dtype:
    if kv_cache_dtype == "auto":
        # Model config may not be specified for unit tests, default to float16
        return model_config.dtype if model_config else torch.half
    return STR_DTYPE_TO_TORCH_DTYPE[kv_cache_dtype]


def set_random_seed(seed: int | None) -> None:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.manual_seed_all(seed)


def async_tensor_h2d(
    data: list,
    dtype: torch.dtype,
    target_device: str | torch.device,
    pin_memory: bool,
) -> torch.Tensor:
    """Asynchronously create a tensor and copy it from host to device."""
    t = torch.tensor(data, dtype=dtype, pin_memory=pin_memory, device="cpu")
    return t.to(device=target_device, non_blocking=True)


def make_ndarray_with_pad(
    x: list[list[T]],
    pad: T,
    dtype: npt.DTypeLike,
    *,
    max_len: int | None = None,
) -> npt.NDArray:
    """
    Make a padded array from 2D inputs.

    The padding is applied to the end of each inner list until it reaches
    `max_len`.
    """
    if max_len is None:
        # Unlike for most functions, map is faster than a genexpr over `len`
        max_len = max(map(len, x), default=0)

    padded_x = np.full((len(x), max_len), pad, dtype=dtype)
    for ind, blocktb in enumerate(x):
        assert len(blocktb) <= max_len
        padded_x[ind, : len(blocktb)] = blocktb

    return padded_x


def make_tensor_with_pad(
    x: list[list[T]],
    pad: T,
    dtype: torch.dtype,
    *,
    max_len: int | None = None,
    device: str | torch.device | None = None,
    pin_memory: bool = False,
) -> torch.Tensor:
    """
    Make a padded tensor from 2D inputs.

    The padding is applied to the end of each inner list until it reaches
    `max_len`.
    """
    np_dtype = TORCH_DTYPE_TO_NUMPY_DTYPE[dtype]
    padded_x = make_ndarray_with_pad(x, pad, np_dtype, max_len=max_len)

    tensor = torch.from_numpy(padded_x).to(device)
    if pin_memory:
        tensor = tensor.pin_memory()

    return tensor


_current_stream_tls = threading.local()


class _StreamPlaceholder:
    def __init__(self):
        self.synchronize = lambda: None


def current_stream():
    """Return the active platform stream without replacing framework functions."""

    return torch.npu.current_stream()


# Global auxiliary stream for running operations in background streams.
# We have single global auxiliary stream to avoid an explosion of streams
# for every layer (and make profiling look sane).
#
# aux_stream() is currently used for:
#   - MoE shared_expert overlap with router
_aux_stream: Any = None


def aux_stream() -> Any:
    """
    Ensures aux_stream is initialized only once
    """
    global _aux_stream

    pass  # Unsupported P4 branch removed.

    return _aux_stream


def cuda_device_count_stateless() -> int:
    """Retired device probe retained only as a negative compatibility query."""
    return 0


def weak_ref_tensor(tensor: Any) -> Any:
    """Use the native NPU weak reference operation (no CUDA extension)."""
    if isinstance(tensor, torch.Tensor) and tensor.numel() > 0:
        import torch_npu

        return torch_npu._C._weak_ref_tensor(tensor)
    return tensor


def weak_ref_tensors(
    tensors: torch.Tensor
    | list[torch.Tensor]
    | tuple[torch.Tensor]
    | IntermediateTensors,
) -> torch.Tensor | list[Any] | tuple[Any] | Any:
    """
    Convenience function to create weak references to tensors,
    for single tensor, list of tensors or tuple of tensors.
    """
    if isinstance(tensors, torch.Tensor):
        return weak_ref_tensor(tensors)
    if isinstance(tensors, list):
        return [weak_ref_tensor(t) for t in tensors]
    if isinstance(tensors, tuple):
        return tuple(weak_ref_tensor(t) for t in tensors)

    # For IntermediateTensors used in pipeline parallelism
    from vllm.sequence import IntermediateTensors

    if isinstance(tensors, IntermediateTensors):
        ret = IntermediateTensors(
            {key: weak_ref_tensor(val) for key, val in tensors.tensors.items()}
        )
        return ret
    raise ValueError("Invalid type for tensors")


def get_accelerator_view_from_cpu_tensor(cpu_tensor: torch.Tensor) -> torch.Tensor:
    """
    Get an accelerator view of a CPU tensor using Unified Virtual Addressing (UVA).
    """
    from vllm.platforms import current_platform

    raise ValueError(
        f"`get_accelerator_view_from_cpu_tensor` is currently "
        f"not supported in: {current_platform.device_name}"
    )


# Helper function used in testing.
def _is_torch_equal_or_newer(torch_version: str, target: str) -> bool:
    return version.parse(torch_version) >= version.parse(target)


def is_torch_equal_or_newer(target: str) -> bool:
    """Check if the installed torch version is >= the target version.

    Args:
        target: a version string, like "2.6.0".

    Returns:
        Whether the condition meets.
    """
    try:
        return _is_torch_equal_or_newer(str(torch.__version__), target)
    except Exception:
        # Fallback to PKG-INFO to load the package info, needed by the doc gen.
        return Version(importlib.metadata.version("torch")) >= Version(target)


def _is_torch_equal(target: str) -> bool:
    assert target.count(".") == 2
    torch_version = str(torch.__version__)
    torch_version = version.parse(torch_version)
    # torch version is like "2.6.0.dev20240101" or "2.6.0.dev20240101+cpu"
    # or "2.6.0+cu128" but never "2.6.0.1"
    return (
        torch_version >= version.parse(target)
        and version.parse(target + ".1") > torch_version
    )


def is_torch_equal(target: str) -> bool:
    """Check if the installed torch version is == the target version.

    Args:
        target: a version string, like "2.6.0".

    Returns:
        Whether the condition meets.
    """
    try:
        return _is_torch_equal(target)
    except Exception:
        return Version(importlib.metadata.version("torch")) == Version(target)


HAS_OPAQUE_TYPE = is_torch_equal_or_newer("2.11.0.dev")

if HAS_OPAQUE_TYPE:
    from torch._opaque_base import OpaqueBase
else:
    OpaqueBase = object  # type: ignore[misc, assignment]


class ModuleName(OpaqueBase):  # type: ignore[misc]
    """Wraps a module name string for use as a torch opaque type.

    When torch >= 2.11, this is registered as a hoisted value-type opaque
    object so that torch.compile lifts it as a graph input instead of baking
    it as a constant.  This avoids per-layer recompilation for MOE ops.
    """

    def __init__(self, value: str):
        self.value = value

    def __eq__(self, other):
        return isinstance(other, ModuleName) and self.value == other.value

    def __hash__(self):
        return hash(self.value)

    def __fx_repr__(self):
        return (f"ModuleName({self.value!r})", {ModuleName})


if HAS_OPAQUE_TYPE:
    from torch._library.opaque_object import register_opaque_type

    register_opaque_type(ModuleName, typ="value", hoist=True)


# Supports xccl with PyTorch versions >= 2.8.0.dev for XPU platform
def supports_xccl() -> bool:
    return torch.distributed.is_xccl_available()


# Supports XPU Graph with PyTorch versions >= 2.11.0.dev for XPU platform
def supports_xpu_graph() -> bool:
    return is_torch_equal_or_newer("2.11.0.dev")


# create a library to hold the custom op
vllm_lib = Library("vllm", "FRAGMENT")  # noqa


def direct_register_custom_op(
    op_name: str,
    op_func: Callable,
    mutates_args: list[str] | None = None,
    fake_impl: Callable | None = None,
    target_lib: Library | None = None,
    dispatch_key: str | None = None,
    tags: tuple[torch.Tag, ...] = (),
):
    """
    `torch.library.custom_op` can have significant overhead because it
    needs to consider complicated dispatching logic. This function
    directly registers a custom op and dispatches it to the CUDA backend.
    See https://gist.github.com/youkaichao/ecbea9ec9fc79a45d2adce1784d7a9a5
    for more details.

    By default, the custom op is registered to the vLLM library. If you
    want to register it to a different library, you can pass the library
    object to the `target_lib` argument.

    IMPORTANT: the lifetime of the operator is tied to the lifetime of the
    library object. If you want to bind the operator to a different library,
    make sure the library object is alive when the operator is used.
    """
    if mutates_args is None:
        mutates_args = []

    if dispatch_key is None:
        from vllm.platforms import current_platform

        dispatch_key = current_platform.dispatch_key

    schema_str = infer_schema(op_func, mutates_args=mutates_args)

    my_lib = target_lib or vllm_lib
    my_lib.define(op_name + schema_str, tags=tags)
    my_lib.impl(op_name, op_func, dispatch_key=dispatch_key)
    if fake_impl is not None:
        my_lib._register_fake(op_name, fake_impl)
