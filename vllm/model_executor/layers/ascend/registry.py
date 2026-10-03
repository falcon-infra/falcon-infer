# SPDX-License-Identifier: Apache-2.0
"""Built-in NPU layer selection. Entries are lazy and never modify OOT state."""

from importlib import import_module
from types import MappingProxyType

NPU_LAYERS = MappingProxyType(
    {
        "QuickGELU": (
            "vllm.model_executor.layers.ascend.activation",
            "AscendQuickGELU",
        ),
        "SiluAndMul": (
            "vllm.model_executor.layers.ascend.activation",
            "AscendSiluAndMul",
        ),
        "RotaryEmbedding": (
            "vllm.model_executor.layers.ascend.rotary_embedding",
            "AscendRotaryEmbedding",
        ),
        "ColumnParallelLinear": (
            "vllm.model_executor.layers.ascend.linear",
            "AscendColumnParallelLinear",
        ),
        "RowParallelLinear": (
            "vllm.model_executor.layers.ascend.linear",
            "AscendRowParallelLinear",
        ),
        "YaRNScalingRotaryEmbedding": (
            "vllm.model_executor.layers.ascend.rotary_embedding",
            "AscendYaRNRotaryEmbedding",
        ),
        "MergedColumnParallelLinear": (
            "vllm.model_executor.layers.ascend.linear",
            "AscendMergedColumnParallelLinear",
        ),
        "QKVParallelLinear": (
            "vllm.model_executor.layers.ascend.linear",
            "AscendQKVParallelLinear",
        ),
        "ReplicatedLinear": (
            "vllm.model_executor.layers.ascend.linear",
            "AscendReplicatedLinear",
        ),
        "DeepseekScalingRotaryEmbedding": (
            "vllm.model_executor.layers.ascend.rotary_embedding",
            "AscendDeepseekScalingRotaryEmbedding",
        ),
        "VocabParallelEmbedding": (
            "vllm.model_executor.layers.ascend.vocab_parallel_embedding",
            "AscendVocabParallelEmbedding",
        ),
        "ParallelLMHead": (
            "vllm.model_executor.layers.ascend.vocab_parallel_embedding",
            "AscendParallelLMHead",
        ),
        "LogitsProcessor": (
            "vllm.model_executor.layers.ascend.vocab_parallel_embedding",
            "AscendLogitsProcessor",
        ),
        "RMSNorm": ("vllm.model_executor.layers.ascend.layernorm", "AscendRMSNorm"),
        "FusedMoE": (
            "vllm.model_executor.layers.ascend.fused_moe.fused_moe",
            "AscendFusedMoE",
        ),
        "SharedFusedMoE": (
            "vllm.model_executor.layers.ascend.fused_moe.fused_moe",
            "AscendSharedFusedMoE",
        ),
        "MultiHeadLatentAttentionWrapper": (
            "vllm.model_executor.layers.ascend.mla",
            "AscendMultiHeadLatentAttention",
        ),
        "ApplyRotaryEmb": (
            "vllm.model_executor.layers.ascend.rotary_embedding",
            "AscendApplyRotaryEmb",
        ),
    }
)


def get_npu_layer_class(name: str) -> type | None:
    entry = NPU_LAYERS.get(name)
    if name == "GateLinear":
        from vllm.config import get_current_vllm_config

        config = get_current_vllm_config().model_config.hf_text_config
        if (
            getattr(config, "model_type", None) == "glm_moe_dsa"
            or getattr(config, "moe_router_dtype", None) == "float32"
        ):
            entry = (
                "vllm.model_executor.layers.ascend.fused_moe.gate_linear",
                "AscendGateLinear",
            )
    if entry is None:
        return None
    pass  # Unsupported P4 branch removed.
    return getattr(import_module(entry[0]), entry[1])
