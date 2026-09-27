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
        "MRotaryEmbedding": (
            "vllm.model_executor.layers.ascend.rotary_embedding",
            "AscendMRotaryEmbedding",
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
        "GemmaRMSNorm": (
            "vllm.model_executor.layers.ascend.layernorm",
            "AscendGemmaRMSNorm",
        ),
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
        "MMEncoderAttention": (
            "vllm.model_executor.layers.ascend.mm_encoder_attention",
            "AscendMMEncoderAttention",
        ),
        "ApplyRotaryEmb": (
            "vllm.model_executor.layers.ascend.rotary_embedding",
            "AscendApplyRotaryEmb",
        ),
        "RMSNormGated": (
            "vllm.model_executor.layers.ascend.layernorm",
            "AscendRMSNormGated",
        ),
        "Conv3dLayer": ("vllm.model_executor.layers.ascend.conv", "AscendConv3dLayer"),
        "RelPosAttention": (
            "vllm.model_executor.layers.ascend.rel_pos_attention",
            "AscendRelPosAttention",
        ),
    }
)
NPU_310P_LAYERS = MappingProxyType(
    {
        "SiluAndMul": (
            "vllm.platforms.ascend_310p.ops.activation",
            "AscendSiluAndMul310",
        ),
        "RotaryEmbedding": (
            "vllm.platforms.ascend_310p.ops.rotary_embedding",
            "AscendRotaryEmbedding310",
        ),
        "RMSNorm": ("vllm.platforms.ascend_310p.ops.layernorm", "AscendRMSNorm310"),
        "GemmaRMSNorm": (
            "vllm.platforms.ascend_310p.ops.layernorm",
            "AscendGemmaRMSNorm310",
        ),
        "RMSNormGated": (
            "vllm.platforms.ascend_310p.ops.layernorm",
            "AscendRMSNormGated310",
        ),
        "FusedMoE": (
            "vllm.platforms.ascend_310p.fused_moe.fused_moe",
            "AscendFusedMoE310",
        ),
        "SharedFusedMoE": (
            "vllm.platforms.ascend_310p.fused_moe.fused_moe",
            "AscendSharedFusedMoE310",
        ),
        "ParallelLMHead": (
            "vllm.platforms.ascend_310p.ops.vocab_parallel_embedding",
            "AscendParallelLMHead310",
        ),
        "VocabParallelEmbedding": (
            "vllm.platforms.ascend_310p.ops.vocab_parallel_embedding",
            "AscendVocabParallelEmbedding310",
        ),
        "MMEncoderAttention": (
            "vllm.platforms.ascend_310p.ops.mm_encoder_attention",
            "AscendMMEncoderAttention310",
        ),
    }
)


def get_npu_layer_class(name: str) -> type | None:
    from vllm.utils.ascend import is_310p

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
    if is_310p():
        if name == "MRotaryEmbedding":
            return None
        entry = NPU_310P_LAYERS.get(name, entry)
    return getattr(import_module(entry[0]), entry[1])
