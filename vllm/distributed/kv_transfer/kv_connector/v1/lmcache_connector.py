# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Optional

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_events import (
    BlockStored,
    KVCacheEvent,
    KVConnectorKVEvents,
    KVEventAggregator,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.logger import init_logger
from vllm.utils.func_utils import supports_kw
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import KVConnectorOutput

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


class LMCacheKVEvents(KVConnectorKVEvents):
    """
    Concrete implementation of KVConnectorKVEvents using KVEventAggregator.
    """

    def __init__(self, num_workers: int) -> None:
        self._aggregator = KVEventAggregator(num_workers)

    def add_events(self, events: list[KVCacheEvent]) -> None:
        self._aggregator.add_events(events)

    def aggregate(self) -> "LMCacheKVEvents":
        """
        Aggregate KV events and retain only common events.
        """
        common_events = self._aggregator.get_common_events()
        self._aggregator.clear_events()
        self._aggregator.add_events(common_events)
        self._aggregator.reset_workers()
        return self

    def increment_workers(self, count: int = 1) -> None:
        self._aggregator.increment_workers(count)

    def get_all_events(self) -> list[KVCacheEvent]:
        return self._aggregator.get_all_events()

    def get_number_of_workers(self) -> int:
        return self._aggregator.get_number_of_workers()

    def clear_events(self) -> None:
        self._aggregator.clear_events()
        self._aggregator.reset_workers()

    def __repr__(self) -> str:
        return f"<LMCacheKVEvents events={self.get_all_events()}>"


class LMCacheConnectorV1(KVConnectorBase_V1, SupportsHMA):
    @classmethod
    def requires_piecewise_for_cudagraph(cls, extra_config: dict[str, Any]) -> bool:
        """
        LMCache requires PIECEWISE CUDA graph mode when layerwise
        operations are enabled. The wait_for_layer_load and save_kv_layer
        methods perform actual async synchronization that cannot be
        captured in CUDA graphs.
        """
        return extra_config.get("use_layerwise", False)

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(
            vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config
        )
        assert vllm_config.kv_transfer_config is not None
        # P3 has one paired LMCache implementation. The old "use_native"
        # choice selected a vendored, CUDA-era adapter and is no longer valid.
        if vllm_config.kv_transfer_config.get_from_extra_config("use_native", False):
            raise ValueError(
                "P3 requires use_native=false and the paired lmcache package"
            )
        from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl

        cls = LMCacheConnectorV1Impl
        transfer = vllm_config.kv_transfer_config
        parallel = vllm_config.parallel_config
        extra = dict(transfer.kv_connector_extra_config or {})
        dp_rank = getattr(parallel, "data_parallel_index", None)
        if dp_rank is None:
            dp_rank = getattr(parallel, "data_parallel_rank_local", 0)
        extra["lmcache_remote_fill_destination_dp_rank"] = int(dp_rank or 0)
        extra["lmcache_remote_fill_destination_dp_size"] = int(
            getattr(parallel, "data_parallel_size", 1) or 1
        )
        transfer.kv_connector_extra_config = extra

        if supports_kw(cls, "kv_cache_config"):
            self._lmcache_engine = cls(
                vllm_config,
                role,
                self,
                kv_cache_config=kv_cache_config,
            )
        else:
            if len(kv_cache_config.kv_cache_groups) > 1:
                raise ValueError(
                    f"{cls.__name__} does not accept runtime KV cache topology "
                    "and cannot serve a multi-group model. Upgrade the LMCache "
                    "implementation or disable the native adapter."
                )
            self._lmcache_engine = cls(vllm_config, role, self)

        self._kv_cache_events: LMCacheKVEvents | None = None

    # ==============================
    # Worker-side methods
    # ==============================
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """
        Initialize with the KV caches. Useful for pre-registering the
        KV Caches in the KVConnector (e.g. for NIXL).

        Args: kv_caches:
            dictionary of layer names, kv cache
        """
        self._lmcache_engine.register_kv_caches(kv_caches)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        """
        Start loading the KV cache from the connector to vLLM's paged
        KV buffer. This is called from the forward context before the
        forward pass to enable async loading during model execution.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation

        Note:
            The number of elements in kv_caches and layer_names should be
            the same.

        """
        self._lmcache_engine.start_load_kv(forward_context, **kwargs)

    def wait_for_layer_load(
        self,
        layer_name: str,
        selected_tokens: list = None,
        token_start_index: list = None,
        request_ids=None,
        target_slot_mapping=None,
        payload_event=None,
        selected_token_counts=None,
    ) -> None:
        """
        Block until the KV for a specific layer is loaded into vLLM's
        paged buffer. This is called from within attention layer to ensure
        async copying from start_load_kv is complete.

        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
            selected_tokens: batched per-request DSA selected token indices.
            token_start_index: per-request start offset into slot_mapping.
            request_ids: req_id for each selected_tokens row (input_batch order),
                used to pair rows to requests by identity instead of by position.
            target_slot_mapping: optional batched physical destination slots
                matching selected_tokens. Used by sparse decode scratch loads.
            payload_event: optional producer event for device selected-token
                payloads. LMCache waits on it before row selection.
        """
        self._lmcache_engine.wait_for_layer_load(
            layer_name,
            selected_tokens,
            token_start_index,
            request_ids,
            target_slot_mapping=target_slot_mapping,
            payload_event=payload_event,
            selected_token_counts=selected_token_counts,
        )

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        """
        Start saving the a layer of KV cache from vLLM's paged buffer
        to the connector. This is called from within attention layer to
        enable async copying during execution.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """
        self._lmcache_engine.save_kv_layer(
            layer_name, kv_layer, attn_metadata, **kwargs
        )

    def wait_for_save(self):
        """
        Block until all the save operations is done. This is called
        as the forward context exits to ensure that the async saving
        from save_kv_layer is complete before finishing the forward.

        This prevents overwrites of paged KV buffer before saving done.
        """
        self._lmcache_engine.wait_for_save()

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        """
        Notifies worker-side connector ids of requests that have
        finished generating tokens.

        Returns:
            ids of requests that have finished asynchronous transfer
            (requests that previously returned True from request_finished()),
            tuple of (sending/saving ids, recving/loading ids).
            The finished saves/sends req ids must belong to a set provided in a
            call to this method (this call or a prior one).
        """
        return self._lmcache_engine.get_finished(finished_req_ids)

    def get_block_ids_with_load_errors(self) -> set[int]:
        """Return block IDs that failed to load during the last interval."""
        return self._lmcache_engine.get_block_ids_with_load_errors()

    def get_kv_connector_kv_cache_events(self) -> LMCacheKVEvents | None:
        """
        Get the KV connector kv cache events collected during the last interval.
        """

        events = self._lmcache_engine.get_kv_events()  # type: ignore [attr-defined]
        if not events:
            return None

        blocks: list[BlockStored] = [
            BlockStored(
                block_hashes=e.block_hashes,
                parent_block_hash=e.parent_block_hash,
                token_ids=e.token_ids,
                lora_id=e.lora_id,
                block_size=e.block_size,
                medium=e.medium,
                lora_name=getattr(e, "lora_name", None),
            )
            for e in events
        ]

        lmcache_kv_events = LMCacheKVEvents(num_workers=1)
        lmcache_kv_events.add_events(blocks)
        return lmcache_kv_events

    # ==============================
    # Scheduler-side methods
    # ==============================
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[Optional[int], bool]:
        """
        Get number of new tokens that can be loaded from the
        external KV cache beyond the num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            the number of tokens that can be loaded from the
            external KV cache beyond what is already computed.
        """
        matched_tokens = self._lmcache_engine.get_num_new_matched_tokens(
            request, num_computed_tokens
        )
        return matched_tokens, self._lmcache_engine.should_load_kv_async(
            request.request_id
        )

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        """
        Update KVConnector state after block allocation.
        """
        self._lmcache_engine.update_state_after_alloc(
            request, num_external_tokens, blocks
        )

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        """
        Build the connector metadata for this step.

        This function should NOT modify fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """
        return self._lmcache_engine.build_connector_meta(scheduler_output)

    def update_connector_output(self, connector_output: KVConnectorOutput):
        """
        Update KVConnector state from worker-side connectors output.

        Args:
            connector_output (KVConnectorOutput): the worker-side
                connectors output.
        """
        # Get the KV events
        kv_cache_events = connector_output.kv_cache_events
        if not kv_cache_events or not isinstance(kv_cache_events, LMCacheKVEvents):
            return

        if self._kv_cache_events is None:
            self._kv_cache_events = kv_cache_events
        else:
            self._kv_cache_events.add_events(kv_cache_events.get_all_events())
            self._kv_cache_events.increment_workers(
                kv_cache_events.get_number_of_workers()
            )
        return

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        """
        Called when a request has finished, before its blocks are freed.

        Returns:
            True if the request is being saved/sent asynchronously and blocks
            should not be freed until the request_id is returned from
            get_finished().
            Optional KVTransferParams to be included in the request outputs
            returned by the engine.
        """
        return self._lmcache_engine.request_finished(request, block_ids)

    def take_events(self) -> Iterable["KVCacheEvent"]:
        """
        Take the KV cache events from the connector.

        Yields:
            New KV cache events since the last call.
        """
        if self._kv_cache_events is not None:
            self._kv_cache_events.aggregate()
            kv_cache_events = self._kv_cache_events.get_all_events()
            yield from kv_cache_events
            self._kv_cache_events.clear_events()
            self._kv_cache_events = None

    @property
    def supports_dsa_compact_external_load(self) -> bool:
        return self._lmcache_engine.supports_dsa_cold_compact_load()

    @property
    def supports_preemption_checkpoint(self) -> bool:
        """Whether this connector requests pre-overwrite decoder snapshots."""
        return bool(
            getattr(self._lmcache_engine.config, "decode_preemption_checkpoint", False)
        )

    def handle_preemptions(self, preempted_req_ids: set[str]) -> None:
        """Drain source owners before the runner reuses preempted blocks."""
        handle = getattr(self._lmcache_engine, "handle_preemptions", None)
        if callable(handle):
            handle(preempted_req_ids)

    def prepare_preemption_checkpoint(self, snapshot: tuple) -> None:
        """Forward a scheduler victim snapshot to the actual implementation."""
        self._lmcache_engine.prepare_preemption_checkpoint(snapshot)

    def handle_preemptions_with_metadata(
        self, preempted_req_ids: set[str], metadata: KVConnectorMetadata
    ) -> None:
        """Bind only this LMCache child's checkpoint controls before overwrite."""
        try:
            self.bind_connector_metadata(metadata)
            self.handle_preemptions(preempted_req_ids)
        finally:
            self.clear_connector_metadata()

    @property
    def supports_dsa_live_split_source(self) -> bool:
        return self._lmcache_engine.supports_dsa_live_split()

    @property
    def supports_dsa_live_latent_split_source(self) -> bool:
        return self._lmcache_engine.supports_dsa_live_latent_source()

    @property
    def supports_dsa_live_latent_split_destination(self) -> bool:
        return self._lmcache_engine.supports_dsa_live_latent_destination()

    def configure_live_latent_source(self, enabled: bool) -> None:
        """Enable latent live-source capture after transport negotiation.

        The implementation deliberately defaults this feature to disabled so
        that upgrading LMCache without a hybrid-capable transport cannot make
        an older consumer reject the otherwise valid group-1 descriptor.
        """
        configure = getattr(self._lmcache_engine, "configure_live_latent_source", None)
        if callable(configure):
            configure(enabled)

    def _take_live_split_destination_plans(
        self, handled_groups: tuple[int, ...]
    ) -> dict[str, dict[str, Any]]:
        """Internal worker hook consumed by AscendMultiConnector."""
        return self._lmcache_engine.take_live_split_destination_plans(handled_groups)

    def _accept_live_split_results(self, results: dict[str, str]) -> None:
        """Internal worker hook for negotiated live-transfer acknowledgements."""
        self._lmcache_engine.accept_live_split_results(results)

    def synchronize_staged_sfa_capture_unsafe_loads(self) -> None:
        """Finish background NPU loads before serving-time graph capture."""
        self._lmcache_engine.synchronize_staged_sfa_capture_unsafe_loads()

    def get_completed_decode_window_saves(self) -> dict[str, int]:
        return self._lmcache_engine.get_completed_decode_window_saves()

    def build_connector_worker_meta(self):
        build = getattr(self._lmcache_engine, "build_connector_worker_meta", None)
        return build() if callable(build) else None

    def shutdown(self):
        """
        Shutdown the connector. This is called when the worker process
        is shutting down to ensure that all the async operations are
        completed and the connector is cleaned up properly.
        """
        return self._lmcache_engine.shutdown()

    def has_pending_control(self) -> bool:
        """Report an acknowledgement waiting for an otherwise idle worker step."""
        return self._lmcache_engine.has_pending_control()

    def update_connector_worker_metadata(
        self, worker_metadata: Any, active_req_ids: set[str]
    ) -> None:
        update = getattr(self._lmcache_engine, "update_connector_worker_metadata", None)
        if callable(update):
            update(worker_metadata, active_req_ids)

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        # LMCache's layerwise DSA path owns both groups, while its scheduler
        # completion bookkeeping is keyed by the primary group's block table.
        return self.request_finished(request, block_ids[0])

    supports_dsa_index_lmcache = True

    @property
    def uses_layerwise_model_callbacks(self) -> bool:
        """Whether model-layer Python callbacks are part of this execution."""
        return bool(getattr(self._lmcache_engine, "use_layerwise", False))

    @property
    def supports_staged_sfa_sparse_load(self) -> bool:
        """Advertise the exact staged-SFA selective-load contract."""
        engine = self._lmcache_engine
        config = getattr(engine, "config", None)
        return bool(
            getattr(engine, "use_layerwise", False)
            and getattr(engine, "kv_role", None) in ("kv_both", "kv_consumer")
            and getattr(config, "dsa_two_groups", False)
            and getattr(config, "enable_sparse_attention", False)
        )

    def capture_live_source_event_handoff(self, forward_context: Any) -> bool:
        """Forward an armed post-forward producer event to the implementation."""

        return bool(
            self._lmcache_engine.capture_live_source_event_handoff(forward_context)
        )

    def seal_sparse_destination_layout(self) -> None:
        """Forward the final staged-capture storage contract when supported."""
        seal = getattr(self._lmcache_engine, "seal_sparse_destination_layout", None)
        if callable(seal):
            seal()

    def get_remote_fill_placement_info(
        self,
    ) -> dict[str, int | str | bool] | None:
        """Return the decoder's pointer-free remote-fill placement."""

        engine = getattr(self._lmcache_engine, "lmcache_engine", None)
        discover = getattr(engine, "get_remote_fill_placement_info", None)
        return discover() if callable(discover) else None

    def get_remote_fill_metrics(self) -> dict[str, int] | None:
        """Return fixed-cardinality decoder protocol metrics when active."""

        engine = getattr(self._lmcache_engine, "lmcache_engine", None)
        snapshot = getattr(engine, "get_remote_fill_metrics", None)
        return snapshot() if callable(snapshot) else None

    def remote_fill_requires_paired_restart(self) -> bool:
        """Expose an armed-transfer fatal latch to the worker supervisor."""

        engine = getattr(self._lmcache_engine, "lmcache_engine", None)
        check = getattr(engine, "remote_fill_requires_paired_restart", None)
        return bool(check()) if callable(check) else False
