# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING

from vllm.distributed.kv_transfer.kv_connector.base import (
    KVConnectorBase,
    KVConnectorBaseType,
)
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorRole,
    supports_hma,
)
from vllm.inference_profile import (
    CONNECTOR,
    CONNECTOR_MODULE,
    NATIVE_CONNECTORS,
    validate_connector,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.config.kv_transfer import KVTransferConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


class KVConnectorFactory:
    _registry: dict[str, Callable[[], type[KVConnectorBase]]] = {}

    @classmethod
    def register_connector(cls, name: str, module_path: str, class_name: str) -> None:
        """Register a connector with a lazy-loading module and class name."""
        validate_connector(name, module_path)
        if class_name != NATIVE_CONNECTORS[name][1]:
            raise ValueError("Unsupported connector class")
        if name in cls._registry:
            raise ValueError(f"Connector '{name}' is already registered.")

        def loader() -> type[KVConnectorBase]:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)

        cls._registry[name] = loader

    @classmethod
    def create_connector(
        cls,
        config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig | None" = None,
    ) -> KVConnectorBase:
        kv_transfer_config = config.kv_transfer_config
        if kv_transfer_config is None:
            raise ValueError("kv_transfer_config must be set to create a connector")
        connector_cls, compat_sig = cls._get_connector_class_with_compat(
            kv_transfer_config
        )

        # check if the connector supports HMA
        hma_enabled = not config.scheduler_config.disable_hybrid_kv_cache_manager
        if hma_enabled and not supports_hma(connector_cls):
            raise ValueError(
                f"Connector {connector_cls.__name__} does not support HMA but "
                f"HMA is enabled. Please set `--disable-hybrid-kv-cache-manager`."
            )

        logger.info(
            "Creating v1 connector with name: %s and engine_id: %s",
            connector_cls.__name__,
            kv_transfer_config.engine_id,
        )
        # NOTE(Kuntai): v1 connector is explicitly separated into two roles.
        # Scheduler connector:
        # - Co-locate with scheduler process
        # - Should only be used inside the Scheduler class
        # Worker connector:
        # - Co-locate with worker process
        # - Should only be used inside the forward context & attention layer
        # We build separately to enforce strict separation
        if compat_sig:
            # Old signature: __init__(self, vllm_config, role)
            return connector_cls(config, role)
        else:
            # New signature: __init__(self, vllm_config, role, kv_cache_config)
            return connector_cls(config, role, kv_cache_config)

    @classmethod
    def get_connector_class_by_name(
        cls, connector_name: str
    ) -> type[KVConnectorBaseType]:
        """Get a registered connector class by name.

        Raises ValueError if the connector is not registered.

        Args:
            connector_name: Name of the registered connector.

        Returns:
            The connector class.
        """
        if connector_name not in cls._registry:
            raise ValueError(f"Connector '{connector_name}' is not registered.")
        return cls._registry[connector_name]()

    @classmethod
    def _get_connector_class_with_compat(
        cls, kv_transfer_config: "KVTransferConfig"
    ) -> tuple[type[KVConnectorBaseType], bool]:
        connector_name = kv_transfer_config.kv_connector
        validate_connector(connector_name, kv_transfer_config.kv_connector_module_path)
        return cls.get_connector_class_by_name(connector_name), False

    @classmethod
    def get_connector_class(
        cls, kv_transfer_config: "KVTransferConfig"
    ) -> type[KVConnectorBaseType]:
        """Get the connector class by name."""
        connector_cls, _ = cls._get_connector_class_with_compat(kv_transfer_config)
        return connector_cls


# The dependency remains lazy: no LMCache import for no-KV inference.
KVConnectorFactory.register_connector(CONNECTOR, CONNECTOR_MODULE, CONNECTOR)
for name, (module, class_name) in NATIVE_CONNECTORS.items():
    if name != CONNECTOR:
        KVConnectorFactory.register_connector(name, module, class_name)
