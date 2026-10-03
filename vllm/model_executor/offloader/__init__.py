# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model parameter offloading infrastructure."""

from vllm.model_executor.offloader.base import (
    BaseOffloader,
    NoopOffloader,
    create_offloader,
    get_offloader,
    set_offloader,
)

__all__ = [
    "BaseOffloader",
    "NoopOffloader",
    "create_offloader",
    "get_offloader",
    "set_offloader",
]
