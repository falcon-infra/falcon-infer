# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Literal

from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

QuantizationMethods = Literal["ascend", "compressed-tensors"]
QUANTIZATION_METHODS = ["ascend", "compressed-tensors"]
DEPRECATED_QUANTIZATION_METHODS: list[str] = []


def register_quantization_config(quantization: str):
    """Reject external quantizers; the native Ascend registry is closed."""
    raise ValueError(f"Custom quantization is not supported: {quantization}")


def get_quantization_config(quantization: str) -> type[QuantizationConfig]:
    """Load a native NPU quantizer without importing vendor kernels."""
    if quantization == "ascend":
        from vllm.model_executor.layers.quantization.ascend.modelslim_config import (
            AscendModelSlimConfig,
        )

        return AscendModelSlimConfig
    if quantization == "compressed-tensors":
        from vllm.model_executor.layers.quantization.ascend.compressed_tensors_config import (
            AscendCompressedTensorsConfig,
        )

        return AscendCompressedTensorsConfig
    raise ValueError(f"Unsupported Ascend quantization: {quantization}")
