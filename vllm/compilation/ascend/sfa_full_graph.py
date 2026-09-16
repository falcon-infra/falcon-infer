# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One target-model ACL graph, including device-selected LMCache transfers."""

from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import Any

import torch
from vllm.compilation.counter import compilation_counter
from vllm.compilation.monitor import validate_cudagraph_capturing_enabled
from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context
from vllm.platforms import current_platform

from vllm.compilation.ascend.sfa_source_lifetime import SFASourceLease

MAX_PENDING_SOURCE_RETIREMENTS = 64


def tensor_signature(value: Any) -> Any:
    """Describe inputs, inspecting each shared tensor once within this call."""
    tensors: dict[int, tuple[Any, Any]] = {}

    def visit(item: Any) -> Any:
        if isinstance(item, torch.Tensor):
            previous = tensors.get(id(item))
            if previous is not None:
                return previous[1]
            signature = (item.data_ptr(), tuple(item.shape), item.stride(), item.dtype, item.device)
            tensors[id(item)] = (item, signature)
            return signature
        if isinstance(item, dict):
            return tuple((k, visit(v)) for k, v in sorted(item.items()))
        if isinstance(item, (tuple, list)):
            return tuple(visit(v) for v in item)
        if item is None or isinstance(item, (bool, int, float, str)):
            return item
        raise TypeError(f"Unsupported full SFA graph input: {type(item)}")

    return visit(value)


@dataclass
class SFAFullGraphEntry:
    graph: Any
    output: Any
    signature: Any


@dataclass
class SFAValidatedCall:
    owner: Any
    generation: int
    context: Any
    key: Hashable
    entry: SFAFullGraphEntry | None
    signature: Any
    kwargs: dict[str, Any]
    consumed: bool = False


@dataclass(frozen=True)
class SFASourceBinding:
    request_ids: tuple[str, ...]
    sources: tuple[Any, ...]
    lease: Any = None
    completion: Any = None

    def matches(self, request_ids: tuple[str, ...], sources: tuple[Any, ...]) -> bool:
        # PreparedSparseSource is an immutable, request-owned snapshot. A store,
        # restore or pointer-table replacement publishes a NEW snapshot. Never
        # compare its dataclass values: tensor equality would run device work.
        # Retaining the objects also prevents Python id/address reuse (ABA).
        return (
            self.request_ids == request_ids
            and len(self.sources) == len(sources)
            and all(old is new for old, new in zip(self.sources, sources))
        )


class SFAFullGraph:
    """Startup-only capture; exactly one replay per authorized target forward.

    FX partitions remain available for compilation, but their ACL wrappers are
    bypassed while the root graph is captured and are not replayed at runtime.
    """

    def __init__(self) -> None:
        self.entries: dict[Hashable, SFAFullGraphEntry] = {}
        self.sealed = False
        self.replay_count = 0
        self.graph_pool = None
        self.source_bindings: dict[int, SFASourceBinding] = {}
        self.source_binding_count = 0
        self._transfer_bundles: dict[int, tuple] = {}
        self._table_history: dict[int, tuple] = {}
        self._generation = 0
        self.retired_sources: list[SFASourceBinding] = []
        self._submission_failed = False
        self._stream = None

    def clear(self) -> None:
        """Discard graphs before profiling's temporary KV storage is released."""
        if (
            self.entries
            or self.source_bindings
            or self.retired_sources
            or self._transfer_bundles
        ):
            torch.npu.synchronize()
        for binding in (*self.source_bindings.values(), *self.retired_sources):
            if binding.lease is not None:
                binding.lease.close()
        self.entries.clear()
        self.sealed = False
        self.replay_count = 0
        self.source_bindings.clear()
        self._transfer_bundles.clear()
        self._table_history.clear()
        self.source_binding_count = 0
        self._generation += 1
        self.retired_sources.clear()
        self._submission_failed = False
        self._stream = None

    def collect_retired_sources(self) -> None:
        if self._submission_failed:
            raise RuntimeError("Full SFA submission failed; source leases retained until shutdown")
        if not self.retired_sources:
            return
        pending = []
        for binding in self.retired_sources:
            if binding.completion.query():
                binding.lease.close()
            else:
                pending.append(binding)
        self.retired_sources = pending

    def release_requests(self, request_ids: set[str]) -> None:
        if request_ids:
            for capacity, binding in tuple(self.source_bindings.items()):
                if request_ids.intersection(binding.request_ids):
                    self.retired_sources.append(self.source_bindings.pop(capacity))
        if self.retired_sources:
            self.collect_retired_sources()

    def register_transfers(
        self, capacity: int, transfers: tuple[Any, ...]
    ) -> None:
        """Register startup-owned transfer tables for one request capacity."""
        context = get_forward_context()
        if (
            self.sealed
            or not context.staged_sfa_graph_dummy_run
            or not transfers
        ):
            raise RuntimeError("Full SFA transfers must be registered during startup")
        previous = self._transfer_bundles.get(capacity)
        if previous is not None:
            if len(previous[0]) != len(transfers) or any(
                old is not new for old, new in zip(previous[0], transfers)
            ):
                raise RuntimeError(
                    "Full SFA transfer bundle changed without clear/recapture"
                )
            return
        first = transfers[0]
        layout = (
            type(first),
            getattr(first, "request_capacity", None),
            getattr(first, "capacity", None),
            getattr(first, "chunk_size", None),
        )
        planner = getattr(first, "plan_bind_update", None)
        if (
            not callable(planner)
            or layout[1] != capacity
            or any(
                (
                    type(transfer),
                    getattr(transfer, "request_capacity", None),
                    getattr(transfer, "capacity", None),
                    getattr(transfer, "chunk_size", None),
                )
                != layout
                for transfer in transfers
            )
        ):
            planner = None
        self._transfer_bundles[capacity] = transfers, planner

    def get_transfers(self, capacity: int) -> tuple[Any, ...]:
        """Return the transfer tables registered during graph startup."""
        if capacity not in self._transfer_bundles:
            raise RuntimeError("Full SFA transfer bundle was not registered at startup")
        return self._transfer_bundles[capacity][0]

    def bind_sources(
        self,
        sources: tuple[Any, ...],
        request_ids: tuple[str, ...],
        transfers: Callable[[], tuple[Any, ...]] | None = None,
    ) -> bool:
        """Rebind layers only when the request-owned source batch changes."""
        capacity = get_forward_context().staged_sfa_graph_key.request_capacity
        if len(sources) > capacity or len(request_ids) != len(sources):
            raise ValueError("Full SFA source lanes do not match request IDs/capacity")
        self.collect_retired_sources()
        previous = self.source_bindings.get(capacity)
        if previous is not None and previous.matches(request_ids, sources):
            return False
        stream = torch.npu.current_stream()
        if self._stream is not None and self._stream != stream:
            raise RuntimeError("Full SFA metadata, pointer uploads and replay must use the same stream")
        self._stream = stream
        if len(self.retired_sources) >= MAX_PENDING_SOURCE_RETIREMENTS:
            raise RuntimeError("Full SFA source retirements are not completing; refusing unbounded retention")
        bundle = self._transfer_bundles.get(capacity)
        if transfers is None:
            self.get_transfers(capacity)
        elif bundle is not None:
            raise RuntimeError("Cannot replace registered full SFA transfers with a callback")
        tokens = None
        lanes = None
        if bundle is not None and bundle[1] is not None:
            tokens = tuple(
                None if source is None else getattr(source, "binding_token", None)
                for source in sources
            )
            if any(
                source is not None and token is None
                for source, token in zip(sources, tokens)
            ):
                tokens = None
            old = self._table_history.get(capacity)
            if tokens is not None and old is not None:
                changed = tuple(
                    index
                    for index in range(max(len(old), len(tokens)))
                    if (
                        old[index] if index < len(old) else None
                    ) is not (
                        tokens[index] if index < len(tokens) else None
                    )
                )
                lanes = bundle[1](sources, changed)
        lease = SFASourceLease(sources)
        try:
            completion = torch.npu.Event()
        except BaseException:
            lease.close()
            raise
        binding = SFASourceBinding(request_ids, sources, lease, completion)
        # Invalidate partial writes before trying to bind a replacement batch.
        self._table_history.pop(capacity, None)
        self.source_bindings.pop(capacity, None)
        if previous is not None:
            self.retired_sources.append(previous)
        try:
            if lanes != ():
                for layer_id, transfer in enumerate(
                    bundle[0] if bundle is not None else transfers()
                ):
                    if lanes is None:
                        transfer.bind_batch(sources, layer_id)
                    else:
                        transfer.bind_batch(sources, layer_id, lanes=lanes)
        except BaseException:
            self.retired_sources.append(binding)
            try:
                completion.record(stream)
            except BaseException:
                self._submission_failed = True
            raise
        self.source_bindings[capacity] = binding
        try:
            completion.record(stream)
        except BaseException:
            self._submission_failed = True
            raise
        if tokens is not None:
            self._table_history[capacity] = tokens
        self.source_binding_count += 1
        return True

    def seal(self, keys: tuple[Hashable, ...]) -> int:
        """Require precisely one startup graph per authorized shape."""
        if set(self.entries) != set(keys):
            raise RuntimeError(
                f"Incomplete full SFA capture: expected={keys}, "
                f"actual={tuple(self.entries)}"
            )
        if self._transfer_bundles and any(
            getattr(key, "request_capacity", None) not in self._transfer_bundles
            for key in keys
        ):
            raise RuntimeError("Full SFA capture is missing a transfer bundle")
        if self._stream is not None:
            torch.npu.synchronize()
            self._stream = None
        self.sealed = True
        return len(self.entries)

    def validate_inputs(self, *, graph_inputs: Any = None, **kwargs: Any) -> Any:
        """Validate replay addresses before entering captured model collectives."""
        context = get_forward_context()
        key = context.staged_sfa_graph_key
        signature = tensor_signature((kwargs, graph_inputs))
        entry = self.entries.get(key)
        if entry is None:
            if self.sealed or not context.staged_sfa_graph_dummy_run:
                raise RuntimeError(
                    f"Full SFA graph missing at runtime: {key}; live capture is prohibited"
                )
        elif entry.signature != signature:
            raise RuntimeError(f"Full SFA graph inputs changed address or layout: {key}")
        return signature

    def prepare_run(self, *, graph_inputs: Any = None, **kwargs: Any) -> SFAValidatedCall:
        """Select the captured entry; inspect layouts only during startup."""
        if self._submission_failed:
            raise RuntimeError("Full SFA submission failed; cannot replay again")
        context = get_forward_context()
        key = context.staged_sfa_graph_key
        entry = self.entries.get(key)
        if context.staged_sfa_graph_dummy_run or entry is None:
            signature = self.validate_inputs(graph_inputs=graph_inputs, **kwargs)
        else:
            signature = entry.signature
        return SFAValidatedCall(self, self._generation, context, key, entry, signature, kwargs)

    def run(
        self,
        runnable: Callable[..., Any],
        *,
        prepared: SFAValidatedCall | None = None,
        graph_inputs: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Capture or replay a whole target forward using stable runner inputs."""
        context = get_forward_context()
        key = context.staged_sfa_graph_key
        if context.cudagraph_runtime_mode == CUDAGraphMode.NONE:
            if prepared is not None:
                raise RuntimeError("Cannot use validated graph inputs for eager execution")
            return runnable(**kwargs)
        if prepared is None:
            prepared = self.prepare_run(graph_inputs=graph_inputs, **kwargs)
        elif kwargs or graph_inputs is not None:
            raise RuntimeError("Cannot replace already validated full SFA graph inputs")
        if (
            self._submission_failed
            or not isinstance(prepared, SFAValidatedCall)
            or prepared.owner is not self
            or prepared.generation != self._generation
            or prepared.context is not context
            or prepared.key != key
            or prepared.consumed
            or self.entries.get(key) is not prepared.entry
        ):
            raise RuntimeError("Stale, foreign or consumed full SFA graph validation")
        prepared.consumed = True
        signature = prepared.signature
        kwargs = prepared.kwargs
        entry = self.entries.get(key)
        if entry is None:
            if self.sealed or not context.staged_sfa_graph_dummy_run:
                raise RuntimeError("Full SFA capture authorization changed after validation")
            if self.sealed or not context.staged_sfa_graph_dummy_run:
                raise RuntimeError(
                    f"Full SFA graph missing at runtime: {key}; live capture is prohibited"
                )
            validate_cudagraph_capturing_enabled()
            graph = torch.npu.NPUGraph()
            previous_capturing = context.capturing
            context.sfa_full_graph_active = True
            context.capturing = True
            try:
                pool = (
                    self.graph_pool
                    if self.graph_pool is not None
                    else current_platform.get_global_graph_pool()
                )
                with torch.npu.graph(graph, pool=pool):
                    output = runnable(**kwargs)
            finally:
                context.sfa_full_graph_active = False
                context.capturing = previous_capturing
            self.entries[key] = SFAFullGraphEntry(graph, output, signature)
            compilation_counter.num_cudagraph_captured += 1
            return output
        binding = self.source_bindings.get(getattr(key, "request_capacity", None))
        if hasattr(key, "request_capacity") and binding is None:
            raise RuntimeError("Full SFA replay requires a bound source batch, including empty lanes")
        stream = torch.npu.current_stream()
        if self._stream is not None and self._stream != stream:
            raise RuntimeError("Full SFA replay stream differs from pointer upload stream")
        self._stream = stream
        with torch.profiler.record_function("sfa_full_graph::target_replay"):
            try:
                entry.graph.replay()
                if binding is not None:
                    binding.completion.record(stream)
            except BaseException:
                self._submission_failed = True
                raise
        self.replay_count += 1
        return entry.output
