# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bootstrap constants with no config, model, utility or device dependencies.

The platform class must be constructible before vllm.config is imported.
Keep these values shared with vllm.utils.ascend without importing that module.
"""

ASCEND_QUANTIZATION_METHOD = "ascend"
COMPRESSED_TENSORS_METHOD = "compressed-tensors"
COMPILATION_PASS_KEY = "graph_fusion_manager"
