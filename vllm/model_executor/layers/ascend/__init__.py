# SPDX-License-Identifier: Apache-2.0
"""Native Ascend operators, initialized explicitly by the NPU worker."""

from importlib import import_module


def initialize_native_ops() -> None:
    from vllm.triton_utils import HAS_TRITON

    modules = [
        "fused_moe.fused_moe",
        "layernorm",
        "register_custom_ops",
    ]
    if HAS_TRITON:
        modules.extend(
            [
                "triton.linearnorm.split_qkv_rmsnorm_rope",
                "triton.linearnorm.split_qkv_rmsnorm_mrope",
                "triton.linearnorm.split_qkv_tp_rmsnorm_rope",
            ]
        )
    modules.extend(["vocab_parallel_embedding", "activation", "rotary_embedding"])
    for module in modules:
        import_module(f"{__name__}.{module}")
