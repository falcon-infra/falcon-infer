# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU semantics and deferred-DMA contracts; no NPU performance claims."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def modules(monkeypatch):
    def load(name, relative):
        spec = importlib.util.spec_from_file_location(name, ROOT / relative)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    diagnostics = load("vllm.v1.sample.ascend.rejection_diagnostics", "vllm/v1/sample/ascend/rejection_diagnostics.py")
    masks = load("tested_native_min_tokens", "vllm/v1/sample/min_tokens.py")
    return masks, diagnostics


def processor(min_toks):
    return SimpleNamespace(min_toks=min_toks, neg_inf_tensor=torch.tensor(-float("inf")))


def expected_mask(logits, min_toks, drafts):
    result = logits.clone()
    offset = 0
    for req, count in enumerate(drafts):
        if req in min_toks:
            minimum, output, stops = min_toks[req]
            for draft in range(count):
                if len(output) + draft < minimum:
                    for token in stops:
                        result[offset + draft, token] = -float("inf")
        offset += count
    return result


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "drafts,min_toks",
    [
        ([1], {}),
        ([1], {0: (5, [], set())}),
        ([1], {0: (0, [], {2})}),
        ([2], {0: (2, [5, 6], {1})}),
        ([0], {0: (3, [], {1})}),
        ([3], {0: (3, [5], {1, 4})}),
        ([2, 3, 1], {0: (5, [6], {0, 4}), 1: (2, [3], {1}), 2: (1, [], {2})}),
    ],
)
def test_mask_semantics_and_inplace_result(modules, dtype, drafts, min_toks):
    masks, _ = modules
    logits = torch.arange(sum(drafts) * 7, dtype=dtype).reshape(sum(drafts), 7).to(torch.float32)
    expected = expected_mask(logits, min_toks, drafts)
    actual = masks.apply_with_spec_decode(processor(min_toks), logits, drafts)
    assert actual is logits
    assert torch.equal(actual, expected)


def test_unchanged_mask_reuses_both_buffers_as_history_grows(modules, monkeypatch):
    masks, _ = modules
    history = []
    owner = processor({0: (512, history, {1, 3})})
    masks.apply_with_spec_decode(owner, torch.zeros(1, 7), [1])
    cached = owner._ascend_spec_mask_indices
    monkeypatch.setattr(masks, "_allocate_indices", Mock(side_effect=AssertionError("unexpected reallocation")))
    monkeypatch.setattr(masks, "_fill_indices", Mock(side_effect=AssertionError("unexpected host write")))
    for _ in range(20):
        history.append(0)
        result = masks.apply_with_spec_decode(owner, torch.zeros(1, 7), [1])
        assert owner._ascend_spec_mask_indices is cached
        assert torch.isneginf(result[0, [1, 3]]).all()
    assert cached.host.shape == cached.indices.shape == (2, 2)


def test_boundary_stop_ids_draft_lengths_and_request_moves_invalidate_cache(modules):
    masks, _ = modules
    owner = processor({0: (4, [], {1})})
    steps = [
        ([3, 2], {0: (4, [], {1})}),
        ([3, 2], {0: (4, [8, 8], {1})}),
        ([3, 2], {0: (4, [8, 8], {2})}),
        ([3, 2], {1: (4, [8, 8], {2})}),
        ([1, 2], {1: (4, [8, 8], {2})}),
        ([1, 2], {1: (4, [8, 8, 8], {2})}),
        ([1, 2], {}),
        ([1, 2], {0: (6, [], {5})}),
    ]
    previous = None
    for drafts, state in steps:
        owner.min_toks = state
        logits = torch.zeros(sum(drafts), 7)
        assert torch.equal(masks.apply_with_spec_decode(owner, logits, drafts), expected_mask(logits, state, drafts))
        assert owner._ascend_spec_mask_indices is not previous
        previous = owner._ascend_spec_mask_indices
    owner.min_toks = {0: (1, [0], {5})}
    masks.apply_with_spec_decode(owner, torch.zeros(1, 7), [1])
    assert owner._ascend_spec_mask_indices is None


class DeferredNPU:
    """Emulate async H2D and masks: no device work runs until drain()."""

    def __init__(self):
        self.device = SimpleNamespace(type="npu", index=0)
        self.stream = SimpleNamespace(npu_stream=1)
        self.pending = []
        self.host_allocations = []
        self.uploads = 0
        self.torch = SimpleNamespace(
            int64=torch.int64,
            empty=self.empty,
            npu=SimpleNamespace(current_stream=lambda device: self.stream),
        )

    def empty(self, shape, *, dtype, device, pin_memory=None):
        if device == "cpu":
            assert pin_memory is True
            host = torch.empty(shape, dtype=dtype)
            self.host_allocations.append(host)
            return host
        return self.Indices(self, shape, dtype)

    class Indices:
        def __init__(self, backend, shape, dtype):
            self.backend = backend
            self.data = torch.empty(shape, dtype=dtype)

        def copy_(self, host, *, non_blocking):
            assert non_blocking is True
            assert any(host is allocation for allocation in self.backend.host_allocations)
            self.backend.uploads += 1
            self.backend.pending.append(lambda: self.data.copy_(host))
            return self

        def __getitem__(self, row):
            return self, row

    class Logits:
        def __init__(self, backend, rows):
            self.backend = backend
            self.device = backend.device
            self.data = torch.zeros(rows, 7)

        def index_put_(self, indices, value):
            def execute():
                actual = tuple(owner.data[row] for owner, row in indices)
                self.data.index_put_(actual, value)

            self.backend.pending.append(execute)
            return self

    def drain(self):
        for operation in self.pending:
            operation()
        self.pending.clear()


def test_pending_upload_is_never_overwritten_or_waited_on(modules, monkeypatch):
    masks, _ = modules
    backend = DeferredNPU()
    monkeypatch.setattr(masks, "torch", backend.torch)
    owner = processor({0: (4, [], {1})})
    first = backend.Logits(backend, 2)
    masks.apply_with_spec_decode(owner, first, [2])
    old_cache = owner._ascend_spec_mask_indices
    old_host = old_cache.host.clone()
    owner.min_toks = {0: (4, [0, 0, 0], {2})}
    second, third = backend.Logits(backend, 2), backend.Logits(backend, 2)
    masks.apply_with_spec_decode(owner, second, [2])
    masks.apply_with_spec_decode(owner, third, [2])
    assert owner._ascend_spec_mask_indices is not old_cache
    assert torch.equal(old_cache.host, old_host)
    assert backend.uploads == 2
    assert torch.count_nonzero(first.data) == 0
    backend.drain()
    assert torch.isneginf(first.data[:, 1]).all()
    assert torch.isneginf(second.data[0, 2]) and second.data[1, 2] == 0
    assert torch.equal(second.data, third.data)


@pytest.mark.parametrize("change", ["stream", "device"])
def test_cache_is_not_consumed_on_another_stream_or_device(modules, monkeypatch, change):
    masks, _ = modules
    backend = DeferredNPU()
    monkeypatch.setattr(masks, "torch", backend.torch)
    owner = processor({0: (4, [], {1})})
    masks.apply_with_spec_decode(owner, backend.Logits(backend, 1), [1])
    previous = owner._ascend_spec_mask_indices
    if change == "stream":
        backend.stream = SimpleNamespace(npu_stream=2)
    else:
        backend.device = SimpleNamespace(type="npu", index=1)
    masks.apply_with_spec_decode(owner, backend.Logits(backend, 1), [1])
    assert owner._ascend_spec_mask_indices is not previous
    assert backend.uploads == 2
    backend.drain()


def test_stages_split_upload_and_mask_and_leave_normal_path_untimed(modules):
    masks, diagnostics = modules
    stages = []

    def record(name, operation, args, kwargs):
        stages.append(name)
        return operation(*args, **kwargs)

    owner = processor({0: (4, [], {1})})
    token = diagnostics.set_stage_recorder(record)
    try:
        masks.apply_with_spec_decode(owner, torch.zeros(1, 7), [1])
        assert stages == [
            "min_tokens.layout",
            "min_tokens.allocate",
            "min_tokens.fill",
            "min_tokens.h2d",
            "min_tokens.mask",
        ]
        stages.clear()
        masks.apply_with_spec_decode(owner, torch.zeros(1, 7), [1])
        assert stages == ["min_tokens.layout", "min_tokens.mask"]
    finally:
        diagnostics.reset_stage_recorder(token)
    stages.clear()
    masks.apply_with_spec_decode(owner, torch.zeros(1, 7), [1])
    assert not stages


def test_failed_upload_is_propagated_without_publishing_bad_cache(modules, monkeypatch):
    masks, _ = modules
    backend = DeferredNPU()
    monkeypatch.setattr(masks, "torch", backend.torch)
    owner = processor({0: (4, [], {1})})
    masks.apply_with_spec_decode(owner, backend.Logits(backend, 1), [1])
    previous = owner._ascend_spec_mask_indices
    owner.min_toks = {0: (4, [], {2})}
    monkeypatch.setattr(backend.Indices, "copy_", Mock(side_effect=RuntimeError("copy failed")))
    with pytest.raises(RuntimeError, match="copy failed"):
        masks.apply_with_spec_decode(owner, backend.Logits(backend, 1), [1])
    assert owner._ascend_spec_mask_indices is previous


def test_no_new_completion_waits_or_device_readbacks():
    source = (ROOT / "vllm/v1/sample/min_tokens.py").read_text(encoding="utf-8")
    tree = __import__("ast").parse(source)
    forbidden = {"item", "tolist", "cpu", "synchronize", "query", "Event", "wait_event", "wait_stream"}
    assert not [
        node.func.attr
        for node in __import__("ast").walk(tree)
        if isinstance(node, __import__("ast").Call)
        and isinstance(node.func, __import__("ast").Attribute)
        and node.func.attr in forbidden
    ]
