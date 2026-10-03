# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dependency-free, fail-closed boundary for the Ascend GLM-5.2 fork.

Shared DeepSeek layer names describe implementation reuse, not additional
supported checkpoints. MTP is derived from the same GLM checkpoint internally.
"""

from collections.abc import Mapping, Sequence
from typing import Any

MODEL_ARCHITECTURE = "GlmMoeDsaForCausalLM"
MODEL_TYPE = "glm_moe_dsa"
MTP_ARCHITECTURE = "DeepSeekMTPModel"
MTP_MODEL_TYPE = "deepseek_mtp"
CONNECTOR = "LMCacheConnectorV1"
CONNECTOR_MODULE = "vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector"
NATIVE_CONNECTORS = {
    CONNECTOR: (CONNECTOR_MODULE, CONNECTOR),
    "MultiConnector": (
        "vllm.distributed.kv_transfer.ascend.ascend_multi_connector",
        "AscendMultiConnector",
    ),
    "MooncakeConnectorV1": (
        "vllm.distributed.kv_transfer.ascend.kv_p2p.mooncake_connector",
        "MooncakeConnector",
    ),
    "MooncakeDSAIndexConnectorV1": (
        "vllm.distributed.kv_transfer.ascend.kv_p2p.mooncake_dsa_index_connector",
        "MooncakeDSAIndexConnector",
    ),
    "MooncakeLayerwiseConnector": (
        "vllm.distributed.kv_transfer.ascend.kv_p2p.mooncake_layerwise_connector",
        "MooncakeLayerwiseConnector",
    ),
    "CPUOffloadingConnector": (
        "vllm.distributed.kv_transfer.ascend.kv_pool.cpu_offload.cpu_offload_connector",
        "CPUOffloadingConnector",
    ),
    "AscendStoreConnector": (
        "vllm.distributed.kv_transfer.ascend.kv_pool.ascend_store.ascend_store_connector",
        "AscendStoreConnector",
    ),
}


def validate_device_name(name: str) -> None:
    """Accept only the validated SoC, not the wider A2 family."""
    if name.lower().replace(" ", "").replace("_", "") != "ascend910b3":
        raise RuntimeError(f"Ascend P4 requires Ascend910B3; detected {name!r}")


def validate_runtime_features(config: Any) -> None:
    """Reject removed capabilities before workers or engine resources start."""
    for name in ("lora_config", "weight_transfer_config", "ec_transfer_config"):
        if getattr(config, name, None) is not None:
            raise ValueError(f"Ascend P4 does not support {name}")
    cache = getattr(config, "cache_config", None)
    if cache is not None and cache.cache_dtype not in ("auto", "bfloat16", "float16"):
        raise ValueError("Ascend P4 requires unquantized KV cache (C8/FP8 disabled)")
    if cache is not None and getattr(cache, "mamba_cache_mode", "none") != "none":
        raise ValueError("Ascend P4 does not support Mamba/linear-attention state")


def validate_model_metadata(config: Any, *, draft: bool = False) -> None:
    """Reject unsupported model metadata, including nested multimodal configs."""
    get = (
        config.get
        if isinstance(config, Mapping)
        else lambda k, d=None: getattr(config, k, d)
    )
    expected = (
        (MTP_MODEL_TYPE, MTP_ARCHITECTURE)
        if draft
        else (MODEL_TYPE, MODEL_ARCHITECTURE)
    )
    actual = (get("model_type"), get("architectures"))
    if actual != (expected[0], [expected[1]]):
        raise ValueError(
            f"Ascend P4 supports only GLM-5.2 native text generation "
            f"(internal MTP when runner=draft); expected {expected}, got {actual}. "
            "Other GLM versions, distilled and multimodal models are not supported."
        )
    for key in ("vision_config", "audio_config", "text_config", "encoder_config"):
        if get(key) is not None:
            raise ValueError(f"Ascend P4 does not support nested/multimodal {key}")


def validate_model_options(
    *, runner: str, convert: str, model_impl: str, quantization: str | None
) -> None:
    """Validate public engine options before model inspection or weight loading."""
    if runner not in ("auto", "generate", "draft") or convert not in ("auto", "none"):
        raise ValueError(
            "Ascend P4 supports text generation and internal MTP only; no pooling/conversion"
        )
    if model_impl not in ("auto", "vllm"):
        raise ValueError(
            "Ascend P4 requires the native vLLM model; no Transformers/custom fallback"
        )
    if quantization not in (None, "ascend", "compressed-tensors"):
        raise ValueError(f"Unsupported Ascend P4 quantization: {quantization}")


def validate_architectures(
    architectures: str | Sequence[str], config: Any
) -> list[str]:
    """Enforce registry lookup without suffix aliases or fallback implementations."""
    names = [architectures] if isinstance(architectures, str) else list(architectures)
    draft = getattr(config, "runner", "auto") == "draft"
    validate_model_metadata(config.hf_config, draft=draft)
    expected = MTP_ARCHITECTURE if draft else MODEL_ARCHITECTURE
    if names != [expected]:
        raise ValueError(
            f"Unsupported architecture lookup {names}; expected {[expected]}"
        )
    validate_model_options(
        runner=getattr(config, "runner", "auto"),
        convert=getattr(config, "convert", "auto"),
        model_impl=config.model_impl,
        quantization=getattr(config, "quantization", None),
    )
    return names


def validate_speculation(method: str | None, model: str | None, target: Any) -> None:
    """Allow only GLM's own MTP head, never a separate draft checkpoint."""
    if method not in (None, "mtp", "deepseek_mtp"):
        raise ValueError("Ascend P4 speculative decoding supports GLM-5.2 MTP only")
    if target is None:
        raise ValueError("MTP requires target_model_config")
    validate_model_metadata(target.hf_config)
    if model is not None and model != target.model:
        raise ValueError("MTP must use the same GLM-5.2 checkpoint as the target")


def validate_connector(name: str | None, module: str | None = None) -> None:
    """Reject dynamic plugin and other-device KV connectors before import."""
    if name not in NATIVE_CONNECTORS or module not in (
        None,
        NATIVE_CONNECTORS[name][0],
    ):
        raise ValueError(
            "Ascend P4 requires a built-in native Ascend KV connector; remove legacy/external module paths"
        )


def validate_text_prompt(prompt: Any) -> None:
    """Reject multimodal/embed payloads at the shared offline input boundary."""
    if isinstance(prompt, bytes):
        raise ValueError("Ascend P4 does not support serialized prompt embeddings")
    if isinstance(prompt, Mapping):
        if "type" in prompt and prompt["type"] != "token":
            raise ValueError("Ascend P4 processed inputs must be token inputs")
        for key in (
            "multi_modal_data",
            "multi_modal_uuids",
            "mm_processor_kwargs",
            "prompt_embeds",
        ):
            if key in prompt and prompt[key] is not None:
                raise ValueError(
                    f"Ascend P4 accepts text/token prompts only, not {key}"
                )
        for key in (
            "encoder_prompt",
            "decoder_prompt",
            "encoder_inputs",
            "decoder_inputs",
        ):
            if key in prompt:
                raise ValueError("Ascend P4 does not support encoder-decoder inputs")
        if prompt.get("mm_features"):
            raise ValueError("Ascend P4 does not support media features")
