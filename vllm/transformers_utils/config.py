# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from collections.abc import Callable
from dataclasses import asdict
from functools import partial
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal, TypeAlias

import huggingface_hub
from huggingface_hub import get_safetensors_metadata
from packaging.version import Version
from transformers import GenerationConfig, PretrainedConfig
from transformers.models.auto.tokenization_auto import get_tokenizer_config

from vllm import envs
from vllm.inference_profile import validate_model_metadata
from vllm.logger import init_logger
from vllm.transformers_utils.utils import (
    parse_safetensors_file_metadata,
    without_trust_remote_code,
)

from .config_parser_base import ConfigParserBase
from .repo_utils import (
    with_retry,
)

try:
    # Transformers v5
    from transformers.configuration_utils import ALLOWED_ATTENTION_LAYER_TYPES
except ImportError:
    # Transformers v4
    from transformers.configuration_utils import (
        ALLOWED_LAYER_TYPES as ALLOWED_ATTENTION_LAYER_TYPES,
    )

if envs.VLLM_USE_MODELSCOPE:
    from modelscope import AutoConfig
else:
    from transformers import AutoConfig

MISTRAL_CONFIG_NAME = "params.json"

logger = init_logger(__name__)


_CONFIG_REGISTRY: dict[str, type[PretrainedConfig]] = {}

_CONFIG_ATTRS_MAPPING: dict[str, str] = {}

_AUTO_CONFIG_KWARGS_OVERRIDES: dict[str, dict[str, Any]] = {}


def is_rope_parameters_nested(rope_parameters: dict[str, Any]) -> bool:
    """Check if rope_parameters is nested by layer types."""
    # Cannot be nested if rope_parameters is empty
    if not rope_parameters:
        return False
    return set(rope_parameters.keys()).issubset(ALLOWED_ATTENTION_LAYER_TYPES)


class HFConfigParser(ConfigParserBase):
    def parse(
        self,
        model: str | Path,
        trust_remote_code: bool,
        revision: str | None = None,
        code_revision: str | None = None,
        **kwargs,
    ) -> tuple[dict, PretrainedConfig]:
        kwargs["local_files_only"] = huggingface_hub.constants.HF_HUB_OFFLINE
        trust_remote_code |= kwargs.get("trust_remote_code", False)
        kwargs = without_trust_remote_code(kwargs)
        config_dict, _ = PretrainedConfig.get_config_dict(
            model,
            revision=revision,
            code_revision=code_revision,
            **kwargs,
        )
        # Validate the original checkpoint before any override can disguise it.
        validate_model_metadata(config_dict)
        model_type = config_dict["model_type"]
        kwargs.pop("hf_overrides", None)

        if model_type in _CONFIG_REGISTRY:
            config_class = _CONFIG_REGISTRY[model_type]
            config = config_class.from_pretrained(
                model,
                revision=revision,
                code_revision=code_revision,
                trust_remote_code=trust_remote_code,
                **kwargs,
            )
        else:
            try:
                kwargs = _maybe_update_auto_config_kwargs(kwargs, model_type=model_type)
                config = AutoConfig.from_pretrained(
                    model,
                    trust_remote_code=trust_remote_code,
                    revision=revision,
                    code_revision=code_revision,
                    **kwargs,
                )
            except ValueError as e:
                if (
                    not trust_remote_code
                    and "requires you to execute the configuration file" in str(e)
                ):
                    err_msg = (
                        "Failed to load the model config. If the model "
                        "is a custom model not yet available in the "
                        "HuggingFace transformers library, consider setting "
                        "`trust_remote_code=True` in LLM or using the "
                        "`--trust-remote-code` flag in the CLI."
                    )
                    raise RuntimeError(err_msg) from e
                else:
                    raise e
        config = _maybe_remap_hf_config_attrs(config)
        return config_dict, config


_CONFIG_FORMAT_TO_CONFIG_PARSER: dict[str, type[ConfigParserBase]] = {
    "hf": HFConfigParser,
}

ConfigFormat = Literal[
    "auto",
    "hf",
]


def get_config_parser(config_format: str) -> ConfigParserBase:
    """Get the config parser for a given config format."""
    if config_format not in _CONFIG_FORMAT_TO_CONFIG_PARSER:
        raise ValueError(f"Unknown config format `{config_format}`.")
    return _CONFIG_FORMAT_TO_CONFIG_PARSER[config_format]()


def register_config_parser(config_format: str):
    raise ValueError("Ascend P4 supports only the built-in HF config parser")


def set_default_rope_theta(config: PretrainedConfig, default_theta: float) -> None:
    """Some models may have no rope_theta in their config but still use RoPE.
    This function sets a default rope_theta if it's missing."""
    if getattr(config, "rope_parameters", None) is None:
        config.rope_parameters = {"rope_type": "default"}
    if "rope_theta" not in config.rope_parameters:
        config.rope_parameters["rope_theta"] = default_theta


def patch_rope_parameters(config: PretrainedConfig) -> None:
    """Provide backwards compatibility for RoPE."""
    from vllm.config.utils import getattr_iter

    # Older custom models may use non-standard field names
    # which need patching for both Transformers v4 and v5.
    names = ["rope_theta", "rotary_emb_base"]
    rope_theta = getattr_iter(config, names, None, warn=True)
    names = ["partial_rotary_factor", "rotary_pct", "rotary_emb_fraction"]
    partial_rotary_factor = getattr_iter(config, names, None, warn=True)
    ompe = getattr(config, "original_max_position_embeddings", None)

    if Version(version("transformers")) < Version("5.0.0"):
        # Transformers v4 installed, legacy config fields may be present
        if (rope_scaling := getattr(config, "rope_scaling", None)) is not None:
            config.rope_parameters = rope_scaling
        if (
            rope_theta is not None
            or partial_rotary_factor is not None
            or ompe is not None
        ) and not getattr(config, "rope_parameters", None):
            config.rope_parameters = {"rope_type": "default"}
        # Patch legacy fields into rope_parameters
        if rope_theta is not None:
            config.rope_parameters["rope_theta"] = rope_theta
        if partial_rotary_factor is not None:
            config.rope_parameters["partial_rotary_factor"] = partial_rotary_factor
        if ompe is not None:
            config.rope_parameters["original_max_position_embeddings"] = ompe
    elif rope_theta is not None or getattr(config, "rope_parameters", None):
        # Transformers v5 installed
        # Patch these fields in case they used non-standard names
        if rope_theta is not None:
            config.rope_theta = rope_theta
        if partial_rotary_factor is not None:
            config.partial_rotary_factor = partial_rotary_factor
        # Standardize and validate RoPE parameters
        config.standardize_rope_params()
        config.validate_rope()

    # No RoPE parameters to patch
    if getattr(config, "rope_parameters", None) is None:
        return

    # Handle nested rope_parameters in interleaved sliding attention models
    if is_rope_parameters_nested(config.rope_parameters):
        for rope_parameters_layer_type in config.rope_parameters.values():
            patch_rope_parameters_dict(rope_parameters_layer_type)
    else:
        patch_rope_parameters_dict(config.rope_parameters)


def patch_rope_parameters_dict(rope_parameters: dict[str, Any]) -> None:
    if "rope_type" in rope_parameters and "type" in rope_parameters:
        rope_type = rope_parameters["rope_type"]
        rope_type_legacy = rope_parameters["type"]
        if (rope_type_legacy == "su" and rope_type == "longrope") or (
            rope_type_legacy == "mrope" and rope_type == "default"
        ):
            pass  # No action needed
        elif rope_type != rope_type_legacy:
            raise ValueError(
                f"Found conflicts between 'rope_type={rope_type}' (modern "
                f"field) and 'type={rope_type_legacy}' (legacy field). "
                "You should only specify one of them."
            )

    if "rope_type" not in rope_parameters and "type" in rope_parameters:
        rope_parameters["rope_type"] = rope_parameters["type"]
        logger.info("Replacing legacy 'type' key with 'rope_type'")

    if "rope_type" not in rope_parameters:
        raise ValueError("rope_parameters should have a 'rope_type' key")

    if rope_parameters["rope_type"] == "su":
        rope_parameters["rope_type"] = "longrope"
        logger.warning("Replacing legacy rope_type 'su' with 'longrope'")
    elif rope_parameters["rope_type"] == "mrope":
        if "mrope_section" not in rope_parameters:
            raise ValueError(
                "Legacy rope_type 'mrope' requires 'mrope_section' in rope_parameters"
            )
        rope_parameters["rope_type"] = "default"
        logger.warning("Replacing legacy rope_type 'mrope' with 'default'")


def _uses_mrope(config: PretrainedConfig) -> bool:
    rope_parameters = getattr(config, "rope_parameters", None)
    if rope_parameters is None:
        return False

    return "mrope_section" in rope_parameters


def uses_mrope(config: PretrainedConfig) -> bool:
    """Detect if the model with this config uses M-ROPE."""
    return (
        _uses_mrope(config)
        or _uses_mrope(config.get_text_config())
        or thinker_uses_mrope(config)
    )


def thinker_uses_mrope(config: PretrainedConfig) -> bool:
    """Detect if the model contains a thinker config and it uses M-ROPE."""
    thinker_config = getattr(config, "thinker_config", None)
    if thinker_config is None:
        return False

    thinker_text_config = getattr(thinker_config, "text_config", None)
    if thinker_text_config is None:
        return False

    return uses_mrope(thinker_text_config)


def uses_xdrope_dim(config: PretrainedConfig) -> int:
    """Detect if the model with this config uses XD-ROPE."""
    xdrope_section = getattr(config, "xdrope_section", None)
    if xdrope_section is not None and isinstance(xdrope_section, list):
        return len(xdrope_section)
    rope_scaling = getattr(config, "rope_scaling", None)
    if rope_scaling is None:
        return 0

    if isinstance(rope_scaling, dict) and "xdrope_section" in rope_scaling:
        xdrope_section = rope_scaling["xdrope_section"]
        if xdrope_section is not None and isinstance(xdrope_section, list):
            return len(xdrope_section)

    return 0


def is_encoder_decoder(config: PretrainedConfig) -> bool:
    """Detect if the model with this config is used as an encoder/decoder."""

    def _is_encoder_decoder(config: PretrainedConfig) -> bool:
        return getattr(config, "is_encoder_decoder", False)

    return _is_encoder_decoder(config) or _is_encoder_decoder(config.get_text_config())


def is_interleaved(config: PretrainedConfig) -> bool:
    """
    Detect if the model with this config is used with interleaved attention.
    """
    text_config = config.get_text_config()
    if layer_types := getattr(text_config, "layer_types", None):
        return len(set(layer_types)) > 1
    return False


def _maybe_update_auto_config_kwargs(kwargs: dict[str, Any], model_type: str):
    """
    Update kwargs for AutoConfig initialization based on model_type
    """
    if model_type in _AUTO_CONFIG_KWARGS_OVERRIDES:
        kwargs.update(_AUTO_CONFIG_KWARGS_OVERRIDES[model_type])
    return kwargs


def _maybe_remap_hf_config_attrs(config: PretrainedConfig) -> PretrainedConfig:
    """Remap config attributes to match the expected names."""
    for old_attr, new_attr in _CONFIG_ATTRS_MAPPING.items():
        if hasattr(config, old_attr):
            if not hasattr(config, new_attr):
                config.update({new_attr: getattr(config, old_attr)})
            logger.debug("Remapped config attribute '%s' to '%s'", old_attr, new_attr)
    return config


def maybe_override_with_speculators(
    model: str,
    tokenizer: str | None,
    trust_remote_code: bool,
    revision: str | None = None,
    vllm_speculative_config: dict[str, Any] | None = None,
    **kwargs,
) -> tuple[str, str | None, dict[str, Any] | None]:
    """Keep GLM's same-checkpoint MTP configuration; never select an external draft."""
    return model, tokenizer, vllm_speculative_config


def get_config(
    model: str | Path,
    trust_remote_code: bool,
    revision: str | None = None,
    code_revision: str | None = None,
    config_format: str | ConfigFormat = "auto",
    hf_overrides_kw: dict[str, Any] | None = None,
    hf_overrides_fn: Callable[[PretrainedConfig], PretrainedConfig] | None = None,
    **kwargs,
) -> PretrainedConfig:
    if config_format not in ("auto", "hf") or str(model).lower().endswith(".gguf"):
        raise ValueError("Ascend P4 requires a GLM-5.2 Hugging Face checkpoint")
    config_format = "hf"

    config_parser = get_config_parser(config_format)
    config_dict, config = config_parser.parse(
        model,
        trust_remote_code=trust_remote_code,
        revision=revision,
        code_revision=code_revision,
        hf_overrides=hf_overrides_kw or hf_overrides_fn,
        **kwargs,
    )

    quantization_config = config_dict.get("quantization_config")
    if quantization_config is not None:
        config.quantization_config = quantization_config

    if hf_overrides_kw:
        logger.debug("Overriding HF config with %s", hf_overrides_kw)
        config.update(hf_overrides_kw)
    if hf_overrides_fn:
        logger.debug("Overriding HF config with %s", hf_overrides_fn)
        config = hf_overrides_fn(config)

    # ModelConfig validates overrides with the target/draft runner context.
    # Exhaustively patch RoPE parameters everywhere they might be
    patch_rope_parameters(config)
    patch_rope_parameters(config.get_text_config())
    SubConfigs: TypeAlias = dict[str, PretrainedConfig]
    sub_configs: SubConfigs | None = getattr(config, "sub_configs", None)
    if sub_configs:
        for sub_config in sub_configs:
            patch_rope_parameters(getattr(config, sub_config))

    if trust_remote_code:
        maybe_register_config_serialize_by_value()

    return config


def maybe_register_config_serialize_by_value() -> None:
    """Try to register HF model configuration class to serialize by value

    If trust_remote_code is set, and the model's config file specifies an
    `AutoConfig` class, then the config class is typically an instance of
    a custom class imported from the HF modules cache.

    Examples:

    >>> from transformers import AutoConfig
    >>> klass = AutoConfig.from_pretrained(
    ...     "meta-llama/Meta-Llama-3-8B", trust_remote_code=True
    ... )
    >>> klass.__class__  # transformers.models.llama.configuration_llama.LlamaConfig
    >>> import transformers_modules  # error, not initialized
    >>> klass = AutoConfig.from_pretrained(
    ...     "deepseek-ai/DeepSeek-V2.5", trust_remote_code=True
    ... )
    >>> import transformers_modules  # success, initialized
    >>> klass.__class__  # transformers_modules.deepseek-ai.DeepSeek-V2.5.98b11844770b2c3ffc18b175c758a803640f4e77.configuration_deepseek.DeepseekV2Config

    In the DeepSeek example, the config class is an instance of a custom
    class that is not serializable by default. This class will not be
    importable in spawned workers, and won't exist at all on
    other nodes, which breaks serialization of the config.

    In this function we tell the cloudpickle serialization library to pass
    instances of these generated classes by value instead of by reference,
    i.e. the class definition is serialized along with its data so that the
    class module does not need to be importable on the receiving end.

    See: https://github.com/cloudpipe/cloudpickle?tab=readme-ov-file#overriding-pickles-serialization-mechanism-for-importable-constructs
    """  # noqa
    try:
        import transformers_modules

        transformers_modules_available = True
    except ImportError:
        transformers_modules_available = False

    try:
        import multiprocessing
        import pickle

        import cloudpickle

        from vllm.config import VllmConfig

        # Register multiprocessing reducers to handle cross-process
        # serialization of VllmConfig objects that may contain custom configs
        # from transformers_modules
        def _reduce_config(config: VllmConfig):
            return (pickle.loads, (cloudpickle.dumps(config),))

        multiprocessing.reducer.register(VllmConfig, _reduce_config)

        # Register transformers_modules with cloudpickle if available
        if transformers_modules_available:
            cloudpickle.register_pickle_by_value(transformers_modules)

            # ray vendors its own version of cloudpickle
            from vllm.v1.executor.ray_utils import ray

            if ray:
                ray.cloudpickle.register_pickle_by_value(transformers_modules)

    except Exception as e:
        logger.warning(
            "Unable to register remote classes used by"
            " trust_remote_code with by-value serialization. This may"
            " lead to a later error. If remote code is not needed"
            " remove `--trust-remote-code`",
            exc_info=e,
        )


def get_hf_text_config(config: PretrainedConfig):
    """Get the "sub" config relevant to llm for multi modal models.
    No op for pure text models.
    """
    text_config = config.get_text_config()

    if text_config is not config and not hasattr(text_config, "num_attention_heads"):
        raise ValueError(
            "The text_config extracted from the model config does not have "
            "`num_attention_heads` attribute. This indicates a mismatch "
            "between the model config and vLLM's expectations. Please "
            "ensure that the model config is compatible with vLLM."
        )

    return text_config


def try_get_generation_config(
    model: str,
    trust_remote_code: bool,
    revision: str | None = None,
    config_format: str | ConfigFormat = "auto",
) -> GenerationConfig | None:
    # GGUF files don't have generation_config.json - their config is embedded
    # in the file header. Skip all filesystem lookups to avoid re-reading the
    # memory-mapped file, which can hang in multi-process scenarios when the
    # EngineCore process already has the file mapped.

    try:
        return GenerationConfig.from_pretrained(
            model,
            revision=revision,
        )
    except OSError:  # Not found
        try:
            config = get_config(
                model,
                trust_remote_code=trust_remote_code,
                revision=revision,
                config_format=config_format,
            )
            return GenerationConfig.from_model_config(config)
        except OSError:  # Not found
            return None


def try_get_safetensors_metadata(
    model: str,
    *,
    revision: str | None = None,
):
    get_safetensors_metadata_partial = partial(
        get_safetensors_metadata, model, revision=revision
    )

    try:
        return with_retry(
            get_safetensors_metadata_partial, "Error retrieving safetensors"
        )
    except Exception:
        return None


def try_get_tokenizer_config(
    pretrained_model_name_or_path: str | os.PathLike,
    trust_remote_code: bool,
    revision: str | None = None,
) -> dict[str, Any] | None:
    try:
        return get_tokenizer_config(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            revision=revision,
        )
    except Exception:
        return None


def get_safetensors_params_metadata(
    model: str,
    *,
    revision: str | None = None,
) -> dict[str, Any]:
    """
    Get the safetensors parameters metadata for remote/local model repository.
    """
    full_metadata = {}
    if (model_path := Path(model)).exists():
        safetensors_to_check = model_path.glob("*.safetensors")
        full_metadata = {
            param_name: info
            for file_path in safetensors_to_check
            if file_path.is_file()
            for param_name, info in parse_safetensors_file_metadata(file_path).items()
        }
    else:
        repo_mt = try_get_safetensors_metadata(model, revision=revision)
        if repo_mt and (files_mt := repo_mt.files_metadata):
            full_metadata = {
                param_name: asdict(info)
                for file_mt in files_mt.values()
                for param_name, info in file_mt.tensors.items()
            }
    return full_metadata
