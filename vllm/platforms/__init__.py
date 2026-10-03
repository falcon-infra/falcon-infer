# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging
import traceback
from importlib.util import find_spec
from typing import TYPE_CHECKING

from vllm.utils.import_utils import resolve_obj_by_qualname

from .interface import CpuArchEnum, Platform, PlatformEnum

logger = logging.getLogger(__name__)


def npu_platform_plugin() -> str | None:
    """Detect the NPU runtime without importing it or initializing a device."""
    if find_spec("torch_npu") is not None:
        return "vllm.platforms.npu.NPUPlatform"
    return None


# Only native Ascend detection is available; there is no fallback platform.
builtin_platform_plugins = {"npu": npu_platform_plugin}


def resolve_current_platform_cls_qualname() -> str:
    """Resolve the built-in platform for this Ascend-only distribution."""
    platform_cls_qualname = npu_platform_plugin()
    if platform_cls_qualname is None:
        raise RuntimeError(
            "This Ascend build of vLLM requires torch_npu. "
            "Use the prepared Ascend environment; CPU/CUDA fallback is disabled."
        )
    return platform_cls_qualname


_current_platform = None
_init_trace: str = ""

if TYPE_CHECKING:
    current_platform: Platform


def __getattr__(name: str):
    if name == "current_platform":
        # Keep construction lazy: configuration imports and spawn startup must
        # not initialize device-specific components before they are needed.
        global _current_platform
        if _current_platform is None:
            platform_cls_qualname = resolve_current_platform_cls_qualname()
            _current_platform = resolve_obj_by_qualname(platform_cls_qualname)()
            global _init_trace
            _init_trace = "".join(traceback.format_stack())
        return _current_platform
    elif name in globals():
        return globals()[name]
    else:
        raise AttributeError(f"No attribute named '{name}' exists in {__name__}.")


def __setattr__(name: str, value):
    if name == "current_platform":
        global _current_platform
        _current_platform = value
    elif name in globals():
        globals()[name] = value
    else:
        raise AttributeError(f"No attribute named '{name}' exists in {__name__}.")


__all__ = [
    "Platform",
    "PlatformEnum",
    "current_platform",
    "CpuArchEnum",
    "_init_trace",
    "_is_amd_zen_cpu",
]
