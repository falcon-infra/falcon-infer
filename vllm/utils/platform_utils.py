# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import cache


def cuda_is_initialized() -> bool:
    """Negative compatibility query; no CUDA backend exists in P4."""
    return False


def xpu_is_initialized() -> bool:
    """Negative compatibility query; no XPU backend exists in P4."""
    return False


@cache
def is_pin_memory_available() -> bool:
    from vllm.platforms import current_platform

    return current_platform.is_pin_memory_available()


@cache
def is_uva_available() -> bool:
    """Check if Unified Virtual Addressing (UVA) is available."""
    # UVA requires pinned memory.
    # TODO: Add more requirements for UVA if needed.
    return is_pin_memory_available()


@cache
def num_compute_units(device_id: int = 0) -> int:
    """Get the number of compute units of the current device."""
    from vllm.platforms import current_platform

    return current_platform.num_compute_units(device_id)
