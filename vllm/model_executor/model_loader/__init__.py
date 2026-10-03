# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Literal

from torch import nn

from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader
from vllm.model_executor.model_loader.sharded_state_loader import ShardedStateLoader
from vllm.model_executor.model_loader.utils import (
    get_architecture_class_name,
    get_model_architecture,
    get_model_cls,
)

logger = init_logger(__name__)

# Reminder: Please update docstring in `LoadConfig`
# if a new load format is added here
LoadFormats = Literal[
    "auto",
    "hf",
    "safetensors",
    "pt",
    "npcache",
    "dummy",
    "sharded_state",
    "netloader",
    "rfork",
]
_LOAD_FORMAT_TO_MODEL_LOADER: dict[str, type[BaseModelLoader]] = {
    "auto": DefaultModelLoader,
    "hf": DefaultModelLoader,
    "safetensors": DefaultModelLoader,
    "pt": DefaultModelLoader,
    "npcache": DefaultModelLoader,
    "dummy": DummyModelLoader,
    "sharded_state": ShardedStateLoader,
}


def register_model_loader(load_format: str):
    raise ValueError("Ascend P4 does not support external model loaders")


def get_model_loader(load_config: LoadConfig) -> BaseModelLoader:
    """Get a model loader based on the load format."""

    native_loaders = {
        "netloader": (
            "vllm.model_executor.model_loader.ascend.netloader.netloader",
            "ModelNetLoaderElastic",
        ),
        "rfork": (
            "vllm.model_executor.model_loader.ascend.rfork.rfork_loader",
            "RForkModelLoader",
        ),
    }
    if load_config.load_format in native_loaders:
        from importlib import import_module

        module, name = native_loaders[load_config.load_format]
        return getattr(import_module(module), name)(load_config)
    load_format = load_config.load_format
    if load_format not in _LOAD_FORMAT_TO_MODEL_LOADER:
        raise ValueError(f"Load format `{load_format}` is not supported")
    return _LOAD_FORMAT_TO_MODEL_LOADER[load_format](load_config)


def get_model(
    *,
    vllm_config: VllmConfig,
    model_config: ModelConfig | None = None,
    prefix: str = "",
    load_config: LoadConfig | None = None,
) -> nn.Module:
    loader = get_model_loader(load_config or vllm_config.load_config)
    if model_config is None:
        model_config = vllm_config.model_config
    return loader.load_model(
        vllm_config=vllm_config, model_config=model_config, prefix=prefix
    )


__all__ = [
    "get_model",
    "get_model_loader",
    "get_architecture_class_name",
    "get_model_architecture",
    "get_model_cls",
    "register_model_loader",
    "BaseModelLoader",
    "BitsAndBytesModelLoader",
    "GGUFModelLoader",
    "DefaultModelLoader",
    "DummyModelLoader",
    "RunaiModelStreamerLoader",
    "ShardedStateLoader",
    "TensorizerLoader",
]
