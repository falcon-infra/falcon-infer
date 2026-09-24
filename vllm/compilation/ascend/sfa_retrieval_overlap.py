# SPDX-License-Identifier: Apache-2.0
"""Startup-owned one-layer-ahead retrieval for native SFA full graphs."""

from dataclasses import dataclass
from typing import Any

import torch


def validate_destinations(caches: dict[str, tuple], destinations: set[str]) -> None:
    """Validate byte ranges while allowing distinct non-overlapping views."""
    ranges = []
    for name, planes in caches.items():
        for plane in planes:
            if not isinstance(plane, torch.Tensor) or not plane.numel():
                continue
            if not plane.is_contiguous():
                raise ValueError(f"overlap requires contiguous cache planes: {name}")
            begin = plane.data_ptr()
            ranges.append(
                (
                    name,
                    plane.device,
                    begin,
                    begin + plane.numel() * plane.element_size(),
                )
            )
    for index, (name, device, begin, end) in enumerate(ranges):
        for other, other_device, first, last in ranges[index + 1 :]:
            if (name in destinations or other in destinations) and device == other_device:
                if begin < last and first < end:
                    raise ValueError(
                        f"prefetch cache ranges overlap: {name}, {other}"
                    )


@dataclass
class RetrievalEdge:
    stream: Any
    transfer: Any
    destinations: tuple
    max_aiv_cores: int = 0

    def __post_init__(self) -> None:
        self.payload = None
        self.pending_capture_launch = False
        self.start = torch.npu.Event(enable_timing=False)
        self.done = torch.npu.Event(enable_timing=False)
        main = torch.npu.current_stream()
        self.start.record(main)
        self.stream.wait_event(self.start)
        self.done.record(self.stream)
        main.wait_event(self.done)

    def launch(self, selected, counts, slots, destinations) -> None:
        if self.pending_capture_launch:
            raise RuntimeError("prefetch launched twice without a consumer join")
        if len(destinations) != len(self.destinations) or any(
            actual.data_ptr() != expected.data_ptr()
            or actual.shape != expected.shape
            or actual.stride() != expected.stride()
            or actual.dtype != expected.dtype
            or actual.device != expected.device
            for actual, expected in zip(destinations, self.destinations)
        ):
            raise RuntimeError("prefetch destination changed after startup")
        payload = (selected, counts, slots)
        if self.payload is not None and any(
            (actual is None) != (previous is None)
            or (
                actual is not None
                and (
                    actual.data_ptr() != previous.data_ptr()
                    or actual.shape != previous.shape
                    or actual.stride() != previous.stride()
                )
            )
            for actual, previous in zip(payload, self.payload)
        ):
            raise RuntimeError("prefetch payload storage changed after capture")
        if self.payload is None:
            self.payload = payload
        self.start.record(torch.npu.current_stream())
        with torch.npu.stream(self.stream):
            self.stream.wait_event(self.start)
            if self.max_aiv_cores:
                self.transfer.load(
                    selected, counts, slots, max_aiv_cores=self.max_aiv_cores
                )
            else:
                self.transfer.load(selected, counts, slots)
            self.done.record(self.stream)
        self.pending_capture_launch = True

    def join(self) -> None:
        if not self.pending_capture_launch:
            raise RuntimeError(
                "consumer prefetch wait has no producer launch in this capture"
            )
        torch.npu.current_stream().wait_event(self.done)
        self.pending_capture_launch = False


class ProbeCopy:
    def __init__(self, destination: torch.Tensor) -> None:
        self.destination = destination

    def load(self, selected, counts, slots) -> None:
        self.destination.copy_(selected)


class CaptureProbe:
    """Fail startup if the device cannot replay the required fork/join."""

    def __init__(self, stream, device) -> None:
        self.source = torch.zeros(64, device=device)
        self.dest = torch.empty_like(self.source)
        self.observed = torch.empty_like(self.source)
        self.matrix = torch.ones((16, 16), device=device, dtype=torch.float16)
        self.product = torch.empty_like(self.matrix)
        self.edge = RetrievalEdge(stream, ProbeCopy(self.dest), (self.dest,))
        self.graph = torch.npu.NPUGraph()

    def verify(self) -> None:
        torch.mm(self.matrix, self.matrix, out=self.product)
        torch.npu.synchronize()
        with torch.npu.graph(self.graph):
            self.edge.launch(self.source, None, None, (self.dest,))
            torch.mm(self.matrix, self.matrix, out=self.product)
            self.edge.join()
            self.observed.copy_(self.dest)
        for value in (1, 7, 2):
            self.source.fill_(value)
            self.graph.replay()
            if not torch.equal(
                self.observed.cpu(), torch.full((64,), float(value))
            ):
                raise RuntimeError("shared retrieval multi-stream capture probe failed")
        torch.npu.synchronize()


class SharedRetrievalOverlap:
    """Immutable one-stream-per-worker retrieval edges after startup setup."""

    def __init__(self) -> None:
        self.stream = None
        self.probe = None
        self.pairs: dict[str, tuple[str, tuple]] = {}
        self.incoming: dict[Any, dict[str, RetrievalEdge]] = {}
        self.outgoing: dict[Any, dict[str, RetrievalEdge]] = {}
        self.failed = False

    def configure(self, impls, groups, kv_caches) -> None:
        if self.stream is not None or self.incoming:
            raise RuntimeError("cannot replace overlap layout after capture setup")
        by_name = dict(impls)
        pairs = {}
        for group in groups:
            for current, following in zip(group.members, group.members[1:]):
                producer, consumer = by_name[current], by_name[following]
                if (
                    not consumer.skip_topk
                    or producer.shared_resident_plan
                    is not consumer.shared_resident_plan
                ):
                    raise ValueError("invalid shared retrieval topology")
                if kv_caches[current][0].dtype not in (
                    torch.float16,
                    torch.bfloat16,
                ):
                    continue
                if not hasattr(torch.ops._C_ascend, "batch_matmul_transpose"):
                    continue
                pairs[current] = (following, tuple(kv_caches[following][:2]))
        if pairs:
            validate_destinations(kv_caches, {name for name, _ in pairs.values()})
        self.pairs = pairs
        for name, impl in impls:
            impl._retrieval_overlap = self

    def prepare(self, key, impls, groups, max_cube_tokens: int) -> None:
        if self.failed:
            raise RuntimeError("shared retrieval setup failed; restart the worker")
        if key in self.incoming:
            return
        by_name = dict(impls)
        if any(not group.active for group in groups):
            raise RuntimeError("shared retrieval requires active resident plans")
        incoming, outgoing = {}, {}
        self.incoming[key], self.outgoing[key] = incoming, outgoing
        if key.token_capacity <= max_cube_tokens and self.pairs:
            try:
                for current, (following, destinations) in self.pairs.items():
                    before = by_name[current]._full_graph_transfers.get(
                        key.request_capacity
                    )
                    transfer = by_name[following]._full_graph_transfers.get(
                        key.request_capacity
                    )
                    if before is None or transfer is None:
                        raise RuntimeError(
                            "shared retrieval transfer was not prepared at startup"
                        )
                    query = by_name[current]._staged_sfa_bridge_buffers[0]
                    if query.dtype not in (torch.float16, torch.bfloat16):
                        continue
                    actual = by_name[following]._staged_sfa_capture_state.runtime[1][:2]
                    if len(actual) != len(destinations) or any(
                        a.data_ptr() != b.data_ptr()
                        for a, b in zip(actual, destinations)
                    ):
                        raise RuntimeError(
                            "prefetch destination does not match prepared transfer"
                        )
                    if self.stream is None:
                        device = destinations[0].device
                        self.stream = torch.npu.Stream(device=device)
                        self.probe = CaptureProbe(self.stream, device)
                        self.probe.verify()
                        self.probe = None
                    edge = RetrievalEdge(
                        self.stream, transfer, destinations, max_aiv_cores=12
                    )
                    incoming[following] = outgoing[current] = edge
            except BaseException:
                self.failed = True
                raise

    def edges(self, key):
        if self.failed or key not in self.incoming:
            raise RuntimeError("shared retrieval key is not prepared")
        return self.incoming[key], self.outgoing[key]

    def clear(self) -> None:
        if self.stream is not None or self.incoming or self.outgoing:
            torch.npu.synchronize()
        self.incoming.clear()
        self.outgoing.clear()
        self.stream = None
        self.probe = None
        self.failed = False

    def release_layout(self, impls) -> None:
        if self.stream is not None or self.incoming or self.outgoing:
            raise RuntimeError("clear overlap graphs before releasing cache layout")
        for _, impl in impls:
            if getattr(impl, "_retrieval_overlap", None) is self:
                impl._retrieval_overlap = None
        self.pairs.clear()
