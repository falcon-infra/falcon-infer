# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host contract tests for the native SFA full-graph owner."""

import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture
def graph_module(monkeypatch):
    context = SimpleNamespace(
        staged_sfa_graph_key="q2",
        cudagraph_runtime_mode="piecewise",
        staged_sfa_graph_dummy_run=True,
        capturing=False,
    )
    counter = SimpleNamespace(num_cudagraph_captured=0)
    validator = Mock()
    modules = {
        "vllm.compilation.counter": {"compilation_counter": counter},
        "vllm.compilation.monitor": {
            "validate_cudagraph_capturing_enabled": validator
        },
        "vllm.config": {"CUDAGraphMode": SimpleNamespace(NONE="none")},
        "vllm.forward_context": {"get_forward_context": lambda: context},
        "vllm.platforms": {
            "current_platform": SimpleNamespace(
                get_global_graph_pool=lambda: "pool"
            )
        },
    }
    for name, attrs in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    lifetime = ModuleType("vllm.compilation.ascend.sfa_source_lifetime")
    lifetime.SFASourceLease = Mock(side_effect=lambda sources: SimpleNamespace(close=Mock()))
    monkeypatch.setitem(sys.modules, lifetime.__name__, lifetime)
    captures = []
    stream = SimpleNamespace(synchronize=Mock())

    @contextmanager
    def capture(graph, pool):
        assert context.sfa_full_graph_active and context.capturing
        captures.append((graph, pool))
        yield

    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            NPUGraph=lambda: SimpleNamespace(replay=Mock()),
            graph=capture,
            current_stream=lambda: stream,
            synchronize=Mock(),
            Event=lambda: SimpleNamespace(record=Mock(), query=Mock(return_value=True)),
        ),
        raising=False,
    )
    path = (
        Path(__file__).resolve().parents[3]
        / "vllm/compilation/ascend/sfa_full_graph.py"
    )
    spec = importlib.util.spec_from_file_location("tested_sfa_full_graph", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module, context, captures, stream


def test_one_replay_without_reentering_target_python(graph_module):
    module, context, captures, stream = graph_module
    wrapper = module.SFAFullGraph()
    x = torch.ones(2)
    target = Mock(side_effect=lambda **kw: kw["input_ids"] + 2)
    out = wrapper.run(target, input_ids=x, positions={"nested": x})
    assert wrapper.seal(("q2",)) == 1
    context.staged_sfa_graph_dummy_run = False
    x.add_(3)
    for _ in range(3):
        assert wrapper.run(target, input_ids=x, positions={"nested": x}) is out
    assert target.call_count == 1
    assert len(captures) == 1
    assert captures[0][0].replay.call_count == 3
    assert stream.synchronize.call_count == 3


def test_no_live_capture_or_changed_keyword_addresses(graph_module):
    module, context, _, _ = graph_module
    wrapper = module.SFAFullGraph()
    x = torch.ones(2)
    wrapper.run(lambda **kw: x, positions=x)
    wrapper.seal(("q2",))
    context.staged_sfa_graph_dummy_run = False
    with pytest.raises(RuntimeError, match="address or layout"):
        wrapper.validate_inputs(positions=x.clone())
    context.staged_sfa_graph_key = "q1"
    with pytest.raises(RuntimeError, match="live capture is prohibited"):
        wrapper.run(Mock(), positions=x)


def test_capture_failure_restores_context_and_profile_reset(graph_module):
    module, context, _, _ = graph_module
    wrapper = module.SFAFullGraph()
    with pytest.raises(ValueError, match="capture failed"):
        wrapper.run(Mock(side_effect=ValueError("capture failed")))
    assert not context.capturing and not context.sfa_full_graph_active
    assert not wrapper.entries
    wrapper.run(lambda: torch.ones(1))
    with pytest.raises(RuntimeError, match="Incomplete"):
        wrapper.seal(("q1", "q2"))
    wrapper.clear()
    assert not wrapper.entries and not wrapper.sealed


@pytest.mark.parametrize("layers", [1, 8, 80])
def test_unchanged_sources_do_not_even_enumerate_layers(graph_module, layers):
    module, context, _, _ = graph_module
    context.staged_sfa_graph_key = SimpleNamespace(request_capacity=2)
    wrapper = module.SFAFullGraph()

    class Source:
        def __eq__(self, other):
            raise AssertionError("Must compare snapshot identity, not tensor values")

    sources = (Source(), Source())
    transfers = tuple(SimpleNamespace(bind_batch=Mock()) for _ in range(layers))
    enumerate_layers = Mock(return_value=transfers)
    assert wrapper.bind_sources(sources, ("a", "b"), enumerate_layers)
    for _ in range(300):
        assert not wrapper.bind_sources(tuple(list(sources)), ("a", "b"), enumerate_layers)
    enumerate_layers.assert_called_once()
    for index, transfer in enumerate(transfers):
        transfer.bind_batch.assert_called_once_with(sources, index)
    assert wrapper.source_binding_count == 1


@pytest.mark.parametrize("change", ["new_source", "window", "restore", "reorder", "request", "empty", "remove"])
def test_source_changes_always_rebind_all_layers(graph_module, change):
    module, context, _, _ = graph_module
    context.staged_sfa_graph_key = SimpleNamespace(request_capacity=2)
    wrapper = module.SFAFullGraph()
    source = SimpleNamespace(total_tokens=4096)
    other = SimpleNamespace(total_tokens=4096)
    transfers = tuple(SimpleNamespace(bind_batch=Mock()) for _ in range(8))
    sources, requests = (source, other), ("a", "b")
    wrapper.bind_sources(sources, requests, lambda: transfers)
    if change in ("new_source", "restore"):
        sources = (SimpleNamespace(total_tokens=4096), other)
    elif change == "window":
        sources = (SimpleNamespace(total_tokens=4352), other)
    elif change == "reorder":
        sources, requests = (other, source), ("b", "a")
    elif change == "request":
        requests = ("new-a", "b")
    elif change == "empty":
        sources, requests = (), ()
    else:
        sources, requests = (source, None), ("a", "b")
    assert wrapper.bind_sources(sources, requests, lambda: transfers)
    for index, transfer in enumerate(transfers):
        assert transfer.bind_batch.call_count == 2
        transfer.bind_batch.assert_called_with(sources, index)
    assert not wrapper.bind_sources(sources, requests, Mock(side_effect=AssertionError("layer loop")))


def test_bindings_follow_shared_capacity_not_graph_key(graph_module):
    module, context, _, _ = graph_module
    wrapper = module.SFAFullGraph()
    source_a, source_b = object(), object()
    transfer = SimpleNamespace(bind_batch=Mock())
    lazy = Mock(return_value=(transfer,))
    context.staged_sfa_graph_key = SimpleNamespace(request_capacity=1, name="q1")
    assert wrapper.bind_sources((source_a,), ("a",), lazy)
    context.staged_sfa_graph_key = SimpleNamespace(request_capacity=1, name="q2")
    assert not wrapper.bind_sources((source_a,), ("a",), lazy)
    assert wrapper.bind_sources((source_b,), ("b",), lazy)
    context.staged_sfa_graph_key = SimpleNamespace(request_capacity=1, name="q1")
    assert wrapper.bind_sources((source_a,), ("a",), lazy)
    context.staged_sfa_graph_key = SimpleNamespace(request_capacity=2, name="r2")
    assert wrapper.bind_sources((source_a,), ("a",), lazy)
    context.staged_sfa_graph_key = SimpleNamespace(request_capacity=1, name="q1")
    assert not wrapper.bind_sources((source_a,), ("a",), lazy)
    assert lazy.call_count == 4


def test_failed_partial_binding_invalidates_previous_binding(graph_module):
    module, context, _, _ = graph_module
    context.staged_sfa_graph_key = SimpleNamespace(request_capacity=1)
    wrapper = module.SFAFullGraph()
    transfers = tuple(SimpleNamespace(bind_batch=Mock()) for _ in range(3))
    source_a, source_b = object(), object()
    wrapper.bind_sources((source_a,), ("a",), lambda: transfers)
    transfers[1].bind_batch.side_effect = RuntimeError("invalid source layer")
    with pytest.raises(RuntimeError, match="invalid source layer"):
        wrapper.bind_sources((source_b,), ("b",), lambda: transfers)
    assert not wrapper.source_bindings
    transfers[1].bind_batch.side_effect = None
    assert wrapper.bind_sources((source_a,), ("a",), lambda: transfers)
    assert transfers[0].bind_batch.call_count == 3
    assert transfers[2].bind_batch.call_count == 2


def test_capture_reset_forces_rebinding_even_for_identical_sources(graph_module):
    module, context, _, _ = graph_module
    context.staged_sfa_graph_key = SimpleNamespace(request_capacity=1)
    wrapper = module.SFAFullGraph()
    source = object()
    lazy = Mock(return_value=(SimpleNamespace(bind_batch=Mock()),))
    wrapper.bind_sources((source,), ("a",), lazy)
    wrapper.clear()
    assert not wrapper.source_bindings and wrapper.source_binding_count == 0
    assert wrapper.bind_sources((source,), ("a",), lazy)
    assert lazy.call_count == 2


@pytest.mark.parametrize("sources,requests", [((None, None), ("a", "b")), ((None,), ()), ((), ("a",))])
def test_invalid_source_lanes_fail_before_binding(graph_module, sources, requests):
    module, context, _, _ = graph_module
    context.staged_sfa_graph_key = SimpleNamespace(request_capacity=1)
    lazy = Mock()
    with pytest.raises(ValueError, match="source lanes"):
        module.SFAFullGraph().bind_sources(sources, requests, lazy)
    lazy.assert_not_called()


def test_runner_handoff_does_not_walk_inputs_or_fence_on_live_replay(graph_module, monkeypatch):
    module, context, captures, stream = graph_module
    graph = module.SFAFullGraph()
    x = torch.ones(2)
    output = graph.run(lambda **kw: x, input_ids=x)
    graph.seal(("q2",))
    context.staged_sfa_graph_dummy_run = False
    validate = Mock(wraps=graph.validate_inputs)
    monkeypatch.setattr(graph, "validate_inputs", validate)
    for _ in range(20):
        prepared = graph.prepare_run(input_ids=x)
        assert graph.run(Mock(side_effect=AssertionError("target Python")), prepared=prepared) is output
    validate.assert_not_called()
    assert captures[0][0].replay.call_count == 20
    assert stream.synchronize.call_count == 0
    with pytest.raises(RuntimeError, match="address or layout"):
        graph.validate_inputs(input_ids=x.clone())


@pytest.mark.parametrize("change", ["consume", "owner", "clear", "key", "context", "entry", "kwargs", "eager"])
def test_prepared_call_cannot_skip_checks_on_other_calls(graph_module, monkeypatch, change):
    module, context, captures, _ = graph_module
    graph = module.SFAFullGraph()
    x = torch.ones(2)
    graph.run(lambda **kw: x, input_ids=x)
    context.staged_sfa_graph_dummy_run = False
    prepared = graph.prepare_run(input_ids=x)
    kwargs = {"prepared": prepared}
    if change == "consume":
        graph.run(Mock(), **kwargs)
    elif change == "owner":
        graph = module.SFAFullGraph()
    elif change == "clear":
        graph.clear()
    elif change == "key":
        context.staged_sfa_graph_key = "other"
    elif change == "context":
        monkeypatch.setattr(module, "get_forward_context", lambda: SimpleNamespace(**vars(context)))
    elif change == "entry":
        graph.entries["q2"] = module.SFAFullGraphEntry(None, None, prepared.signature)
    elif change == "kwargs":
        kwargs["input_ids"] = x.clone()
    else:
        context.cudagraph_runtime_mode = "none"
    before = captures[0][0].replay.call_count
    with pytest.raises(RuntimeError):
        graph.run(Mock(), **kwargs)
    assert captures[0][0].replay.call_count == before


def test_capture_uses_prevalidated_kwargs_and_cannot_reuse_after_failure(graph_module):
    module, context, _, _ = graph_module
    graph = module.SFAFullGraph()
    x = torch.ones(1)
    prepared = graph.prepare_run(input_ids=x)
    target = Mock(side_effect=ValueError("capture failed"))
    with pytest.raises(ValueError, match="capture failed"):
        graph.run(target, prepared=prepared)
    target.assert_called_once_with(input_ids=x)
    with pytest.raises(RuntimeError, match="consumed"):
        graph.run(target, prepared=prepared)
