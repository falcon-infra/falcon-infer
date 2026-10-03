# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import uuid
from dataclasses import field
from typing import Any, Literal, get_args

from vllm.config.utils import config
from vllm.logger import init_logger
from vllm.utils.hashing import safe_hash

logger = init_logger(__name__)

KVProducer = Literal["kv_producer", "kv_both"]
KVConsumer = Literal["kv_consumer", "kv_both"]
KVRole = Literal[KVProducer, KVConsumer]


def kv_buffer_device_default_factory() -> str:
    from vllm.platforms import current_platform

    return current_platform.device_type


@config
class KVTransferConfig:
    """Configuration for distributed KV cache transfer."""

    kv_connector: str | None = None
    """The KV connector for vLLM to transmit KV caches between vLLM instances.
    """

    engine_id: str | None = None
    """The engine id for KV transfers."""

    kv_buffer_device: str = field(default_factory=kv_buffer_device_default_factory)
    """The device used by kv connector to buffer the KV cache. Choices are
    'cuda', 'cpu' and 'xpu'."""

    kv_buffer_size: float = 1e9
    """The buffer size for TorchDistributedConnector. Measured in number of
    bytes. Recommended value: 1e9 (about 1GB)."""

    kv_role: KVRole | None = None
    """Whether this vLLM instance produces, consumes KV cache, or both. Choices
    are 'kv_producer', 'kv_consumer', and 'kv_both'."""

    kv_rank: int | None = None
    """The rank of this vLLM instance in the KV cache transfer. Typical value:
    0 for prefill instance, 1 for decode instance.
    Currently only 1P1D is supported."""

    kv_parallel_size: int = 1
    """The number of parallel instances for KV cache transfer. For
    P2pNcclConnector, this should be 2."""

    kv_ip: str = "127.0.0.1"
    """The KV connector ip, used to build distributed connection."""

    kv_port: int = 14579
    """The KV connector port, used to build distributed connection."""

    kv_connector_extra_config: dict[str, Any] = field(default_factory=dict)
    """any extra config that the connector may need."""

    kv_connector_module_path: str | None = None
    """The Python module path to dynamically load the KV connector from.
    Only supported in V1."""

    enable_permute_local_kv: bool = False
    """Experiment feature flag to enable HND to NHD KV Transfer"""

    kv_load_failure_policy: Literal["recompute", "fail"] = "fail"
    """Policy for handling KV cache load failures.
    'recompute': reschedule the request to recompute failed blocks
    'fail': immediately fail the request with an error finish reason (default)"""

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        # no factors to consider.
        # this config will not affect the computation graph.
        factors: list[Any] = []
        hash_str = safe_hash(str(factors).encode(), usedforsecurity=False).hexdigest()
        return hash_str

    def __post_init__(self) -> None:
        if self.engine_id is None:
            self.engine_id = str(uuid.uuid4())

        if self.kv_role is not None and self.kv_role not in get_args(KVRole):
            raise ValueError(
                f"Unsupported kv_role: {self.kv_role}. "
                f"Supported roles are {get_args(KVRole)}"
            )

        if self.kv_connector is not None and self.kv_role is None:
            raise ValueError(
                "Please specify kv_role when kv_connector "
                f"is set, supported roles are {get_args(KVRole)}"
            )

        self._normalize_lmcache_connector()

    def _normalize_lmcache_connector(self) -> None:
        """Migrate the known P1/P2 launch fields without importing retired code."""
        if self.kv_connector is None:
            return
        module_path = self.kv_connector_module_path
        legacy_module = "lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1"
        if self.kv_connector == "LMCacheAscendConnectorV1Dynamic" and module_path in (
            None,
            legacy_module,
        ):
            # P3's formal connector owns the same Ascend lifecycle. Normalize
            # before factory lookup so scheduler and workers share one identity.
            # An explicit custom module is not redirected by class name alone.
            self.kv_connector = "LMCacheConnectorV1"
            self.kv_connector_module_path = None
            logger.warning(
                "P3 migrated retired LMCache connector configuration to "
                "kv_connector='LMCacheConnectorV1', kv_connector_module_path=None. "
                "Update the launch configuration; do not install lmcache-ascend."
            )
        elif module_path and (
            module_path == "lmcache_ascend" or module_path.startswith("lmcache_ascend.")
        ):
            raise ValueError(
                "P3 removed the lmcache_ascend package. Set "
                "kv_connector='LMCacheConnectorV1' and remove "
                "kv_connector_module_path, or use LMCacheConnectorV1Dynamic from "
                "lmcache.integration.vllm.lmcache_connector_v1. "
                "Preserve the other KV transfer fields."
            )

        native_lmcache = self.kv_connector in (
            "LMCacheConnectorV1",
            "LMCacheAscendConnector",
        ) or (
            self.kv_connector == "LMCacheConnectorV1Dynamic"
            and self.kv_connector_module_path
            == "lmcache.integration.vllm.lmcache_connector_v1"
        )
        if native_lmcache and self.get_from_extra_config("use_native", False):
            raise ValueError(
                "P3 requires use_native=false and the paired lmcache package"
            )

    @property
    def is_kv_transfer_instance(self) -> bool:
        return self.kv_connector is not None and self.kv_role in get_args(KVRole)

    @property
    def is_kv_producer(self) -> bool:
        return self.kv_connector is not None and self.kv_role in get_args(KVProducer)

    @property
    def is_kv_consumer(self) -> bool:
        return self.kv_connector is not None and self.kv_role in get_args(KVConsumer)

    def get_from_extra_config(self, key, default) -> Any:
        return self.kv_connector_extra_config.get(key, default)
