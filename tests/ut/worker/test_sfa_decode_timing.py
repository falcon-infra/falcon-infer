# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for opt-in timing; these tests do not claim NPU performance."""

import ast
import importlib.util
import sys
import threading
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def timing_module(monkeypatch):
    path = ROOT / "vllm/v1/worker/sfa_decode_timing.py"
    spec = importlib.util.spec_from_file_location("tested_sfa_decode_timing", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def schedule(computed, outputs, query=2, *, new=False):
    return SimpleNamespace(
        num_scheduled_tokens={"request": query},
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[] if new else ["request"], num_computed_tokens=[computed], num_output_tokens=[outputs]
        ),
    )


def test_moments(timing_module):
    values = timing_module.Moments()
    for value in (0, 1, 2, 3):
        values.add(value)
    assert values.report() == {"count": 4, "total_ms": 6, "mean_ms": 1.5, "std_ms": 1.25**0.5, "max_ms": 3}
    for value in (float("nan"), float("inf"), -1):
        with pytest.raises(ValueError):
            values.add(value)


def test_route_histogram_counts_selected_profile_without_tensor_reads(timing_module):
    timing = timing_module.DecodeTiming()
    routes = [
        SimpleNamespace(
            graph_key=SimpleNamespace(
                query_profile=SimpleNamespace(value=profile)
            )
        )
        for profile in ("spec_fixed", "decode_bounded", "spec_fixed")
    ]
    original = Mock(side_effect=[SimpleNamespace(graph_key=None), *routes])
    runner = SimpleNamespace(_staged_sfa_live_route=original)
    timing_module.install_route_timing(runner, timing)
    runner._staged_sfa_live_route()
    timing.active = True
    for expected in routes:
        assert runner._staged_sfa_live_route() is expected
    assert timing.report()["graph_profiles_histogram"] == {
        "spec_fixed": 2,
        "decode_bounded": 1,
    }
    timing.close()
    assert runner._staged_sfa_live_route is original


def test_exclusive_wall_and_cpu_do_not_double_count(timing_module, monkeypatch):
    ticks = iter([0, 1_000_000, 4_000_000, 10_000_000])
    cpu = iter([0, 1_000_000, 2_000_000, 3_000_000])
    monkeypatch.setattr(timing_module.time, "perf_counter_ns", lambda: next(ticks))
    monkeypatch.setattr(timing_module.time, "thread_time_ns", lambda: next(cpu))
    timing = timing_module.DecodeTiming()
    timing.active = True
    with timing.scope("parent"), timing.scope("child"):
        pass
    stages = timing.report()["stages"]
    assert stages["parent"]["wall"]["total_ms"] == 10
    assert stages["parent"]["self_wall"]["total_ms"] == 7
    assert stages["child"]["self_wall"]["total_ms"] == 3
    assert stages["parent"]["self_cpu"]["total_ms"] == 2


def test_disabled_and_other_threads_never_time_or_record_events(timing_module, monkeypatch):
    clock, factory = Mock(side_effect=AssertionError("No clocks")), Mock(side_effect=AssertionError("No events"))
    timing = timing_module.DecodeTiming(factory)
    monkeypatch.setattr(timing_module.time, "perf_counter_ns", clock)
    with timing.scope("prefill", device=True):
        pass
    timing.active = True

    thread = threading.Thread(target=lambda: timing.scope("background", device=True).__enter__())
    thread.start()
    thread.join(timeout=5)
    clock.assert_not_called()
    factory.assert_not_called()
    assert not timing.scopes


def test_event_intervals_are_bounded_and_read_after_request(timing_module, monkeypatch):
    monkeypatch.setattr(timing_module, "MAX_DEVICE_INTERVALS", 2)
    events, ready = [], False

    class Event:
        def __init__(self):
            self.recorded = False
            events.append(self)

        def record(self):
            self.recorded = True

        def elapsed_time(self, end):
            assert ready and self.recorded and end.recorded
            return 3

    timing = timing_module.DecodeTiming(Event)
    timing.active = True
    for _ in range(4):
        with timing.scope("target", device=True):
            pass
    assert len(events) == 4 and "stream_span" not in timing.scopes["target"]
    ready = True
    report = timing.report()
    assert report["device_intervals_dropped"] == 2
    assert report["stages"]["target"]["stream_span"]["total_ms"] == 6
    assert timing.report() == report


@pytest.mark.parametrize(
    "computed,outputs,query,new,active",
    [(0, 0, 512, True, False), (4999, 0, 1, False, False), (5000, 0, 1, False, False), (5000, 1, 1, False, True)],
)
def test_decode_gate_uses_scheduler_history(timing_module, computed, outputs, query, new, active):
    timing = timing_module.DecodeTiming()
    timing.begin_step(schedule(computed, outputs, query, new=new), 5000)
    assert timing.active is active
    assert timing.decode_steps == int(active)
    assert timing.prefill_steps == int(not active)


def test_multi_request_gate_rejected(timing_module):
    with pytest.raises(RuntimeError, match="exactly one"):
        timing_module.DecodeTiming().begin_step(
            SimpleNamespace(num_scheduled_tokens={"a": 1, "b": 2}), 5000
        )


def test_scope_failure_keeps_original_exception_and_unwinds_stack(timing_module):
    timing = timing_module.DecodeTiming()
    timing.active = True
    with pytest.raises(ValueError, match="original"), timing.scope("parent"), timing.scope("child"):
        raise ValueError("original")
    assert not timing.stack
    assert set(timing.report()["stages"]) == {"parent", "child"}


def test_worker_rpc_paths_install_and_restore_native_timing(monkeypatch):
    install = Mock()
    timing = SimpleNamespace(close=Mock(), report=Mock(return_value={"decode_steps": 3}))
    install.return_value = timing
    for name, fields in {
        "vllm.distributed.kv_transfer": {"get_kv_transfer_group": lambda: "connector"},
        "vllm.v1.worker.sfa_decode_timing": {"install_decode_timing": install},
    }.items():
        stub = ModuleType(name)
        stub.__dict__.update(fields)
        monkeypatch.setitem(sys.modules, name, stub)
    path = ROOT / "vllm/v1/worker/sfa_benchmark_worker.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SFABenchmarkWorker")
    methods = [
        node
        for node in cls.body
        if getattr(node, "name", "") in ("benchmark_start_decode_timing", "benchmark_stop_decode_timing")
    ]
    module = ast.parse("from __future__ import annotations")
    module.body.extend(methods)
    namespace = {"torch": SimpleNamespace(npu=SimpleNamespace(synchronize=Mock(), Event=Mock()))}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    graph = SimpleNamespace(replay_count=100, source_binding_count=5)
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(_sfa_full_graph=graph, use_async_scheduling=False),
        benchmark_process_info=lambda: {"rank": 1, "pid": 123},
    )
    assert namespace["benchmark_start_decode_timing"](worker, 5000) == {"rank": 1, "pid": 123}
    graph.replay_count += 3
    graph.source_binding_count += 1
    result = namespace["benchmark_stop_decode_timing"](worker)
    assert result["root_replays"] == 3 and result["source_binding_updates"] == 1
    assert worker._decode_timing is None


def test_target_boundary_events_and_pre_replay_span_are_retained(timing_module):
    events = []

    def factory():
        event = SimpleNamespace(record=Mock(), elapsed_time=lambda end: 1)
        events.append(event)
        return event

    timing = timing_module.DecodeTiming(factory)
    timing.active = True
    for _ in range(2):
        with timing.scope("target.forward", device=True), timing.scope("root.replay_submit", device=True):
            pass
    assert timing.last_target_events == (events[4], events[5])
    assert timing.pending[-3] == ("target.before_replay", events[4], events[6])
    assert timing.report()["stages"]["target.before_replay"]["stream_span"]["count"] == 2


@pytest.mark.parametrize("step,expected", [(1, True), (4, True), (5, False), (31, False), (32, True), (33, False)])
def test_sampling_device_event_stride_does_not_change_host_coverage(timing_module, step, expected):
    timing = timing_module.DecodeTiming()
    timing.decode_steps = step
    assert timing.detail_device_sample is expected


@pytest.mark.parametrize("failure", [False, True])
def test_sampling_hooks_use_real_scoped_recorder_and_restore_on_failure(timing_module, monkeypatch, failure):
    path = ROOT / "vllm/v1/sample/ascend/rejection_diagnostics.py"
    name = "vllm.v1.sample.ascend.rejection_diagnostics"
    spec = importlib.util.spec_from_file_location(name, path)
    recorder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recorder)
    monkeypatch.setitem(sys.modules, name, recorder)
    sampler = SimpleNamespace(
        **{
            method: Mock(side_effect=lambda value: value + 1)
            for method in ("apply_logits_processors", "sample", "gather_logprobs")
        }
    )
    original_method = sampler.sample

    def sample(value):
        output = recorder.record_stage("bonus_sampler", sampler.sample, value)
        if failure:
            raise ValueError("original sample error")
        return output

    runner = SimpleNamespace(_sample=sample, sampler=sampler)
    timing = timing_module.DecodeTiming(event_factory=Mock(side_effect=AssertionError("Unsampled step")))
    timing.decode_steps = 5
    timing_module.install_sampling_timing(runner, timing)
    try:
        for active in (False, True):
            timing.active = active
            if failure:
                with pytest.raises(ValueError, match="original sample error"):
                    runner._sample(8)
            else:
                assert runner._sample(8) == 9
            assert not recorder.stage_recorder_active()
            if not active:
                assert not timing.scopes
        assert timing.report()["stages"]["sampling.bonus_sampler"]["wall"]["count"] == 1
        assert timing.report()["stages"]["sampling.sampler.sample"]["wall"]["count"] == 1
        assert original_method.call_count == 2
    finally:
        timing.close()
    assert runner._sample is sample and sampler.sample is original_method


@pytest.mark.parametrize("supported", [False, True])
def test_worker_reads_graph_events_only_after_request_fence(supported):
    events = []
    bounds = object()
    timing = SimpleNamespace(
        close=lambda: events.append("restore"),
        report=lambda: {},
        last_target_events=bounds,
    )
    path = ROOT / "vllm/v1/worker/sfa_benchmark_worker.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SFABenchmarkWorker")
    method = next(node for node in cls.body if getattr(node, "name", "") == "benchmark_stop_decode_timing")
    module = ast.parse("from __future__ import annotations")
    module.body.append(method)

    def graph_report(actual_bounds, *, full):
        assert events == ["restore", "sync"]
        assert actual_bounds is bounds and full
        return {"status": "complete"}

    namespace = {
        "torch": SimpleNamespace(npu=SimpleNamespace(synchronize=lambda: events.append("sync"))),
        "envs": SimpleNamespace(VLLM_ASCEND_SFA_FULL_GRAPH=True),
    }
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    support = {"status": "supported"}
    worker = SimpleNamespace(
        _decode_timing=timing,
        _graph_phase_timing=SimpleNamespace(report=graph_report) if supported else None,
        _graph_phase_support=support if supported else {"status": "unavailable", "reason": "event recorder null"},
        _decode_timing_root_start=0,
        _decode_timing_source_start=0,
        model_runner=SimpleNamespace(_sfa_full_graph=SimpleNamespace(replay_count=3, source_binding_count=1)),
        benchmark_process_info=lambda: {"rank": 0},
    )
    if not supported:
        del timing.last_target_events
    result = namespace["benchmark_stop_decode_timing"](worker)
    assert result["graph_phases"] == ({"status": "complete"} if supported else worker._graph_phase_support)


@pytest.mark.parametrize("processed", [False, True])
@pytest.mark.parametrize("logprobs", [None, 1])
def test_instrumented_rejection_forward_matches_sibling_baseline_on_cpu(processed, logprobs):
    """Shared CPU shims catch diagnostic-only extra sampling or input changes."""
    try:
        import torch
    except ModuleNotFoundError:
        pytest.skip("Requires torch for CPU sampling equivalence")

    baseline_path = ROOT.parent / "vllm/vllm/v1/sample/rejection_sampler.py"
    if not baseline_path.is_file():
        pytest.skip("Requires matching sibling vLLM checkout")

    def forward(path, cls, namespace):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == cls)
        body = next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
        body.decorator_list = []
        module = ast.parse("from __future__ import annotations")
        module.body.append(body)
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
        return namespace["forward"]

    @dataclass
    class Sampling:
        max_num_logprobs: int | None

    calls, tensors, stages = [], [], []

    def capture(name, value):
        calls.append(name)
        tensors.append(value.detach().clone())

    def bonus(*, logits, sampling_metadata, **kwargs):
        assert sampling_metadata.max_num_logprobs == -1
        capture("bonus", logits)
        return SimpleNamespace(sampled_token_ids=logits.argmax(-1), logprobs_tensors=SimpleNamespace(logprobs=logits))

    def processors(logits, *args):
        capture("processors", logits)
        return logits.add_(0.25)

    def constraints(logits, *args):
        capture("constraints", logits)
        return logits

    def rejection(*args):
        capture("rejection", args[5])
        return args[5].argmax(-1)

    def get_logprobs(*args):
        capture("logprobs", args[3])
        return args[3].clone()

    def record(name, operation, *args, **kwargs):
        stages.append(name)
        return operation(*args, **kwargs)

    namespace = dict(
        torch=torch,
        MAX_SPEC_LEN=8,
        replace=replace,
        SamplerOutput=SimpleNamespace,
        stage_recorder_active=lambda: True,
        record_stage=record,
        apply_sampling_constraints=constraints,
        rejection_sample=rejection,
    )
    baseline = forward(baseline_path, "RejectionSampler", dict(namespace))
    observed = forward(ROOT / "vllm/v1/sample/ascend/rejection_sampler.py", "AscendRejectionSampler", dict(namespace))
    sampler = SimpleNamespace(
        sampler=bonus,
        is_processed_logprobs_mode=processed,
        apply_logits_processors=processors,
        _get_logprobs_tensors=get_logprobs,
    )
    metadata = SimpleNamespace(
        max_spec_len=1,
        bonus_logits_indices=torch.tensor([1]),
        target_logits_indices=torch.tensor([0]),
        cu_num_draft_tokens=torch.tensor([1]),
        draft_token_ids=torch.tensor([2]),
        num_draft_tokens=[1],
    )
    logits = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
    expected = baseline(sampler, metadata, None, logits.clone(), Sampling(logprobs))
    expected_calls, expected_tensors = list(calls), list(tensors)
    calls.clear()
    tensors.clear()
    actual = observed(sampler, metadata, None, logits.clone(), Sampling(logprobs))
    assert calls == expected_calls
    assert all(actual_value.equal(expected_value) for actual_value, expected_value in zip(tensors, expected_tensors))
    assert torch.equal(actual.sampled_token_ids, expected.sampled_token_ids)
    if logprobs is not None:
        assert torch.equal(actual.logprobs_tensors, expected.logprobs_tensors)
    else:
        assert actual.logprobs_tensors is expected.logprobs_tensors is None
    assert stages[:4] == ["bonus_index", "bonus_sampler", "target_index_cast", "logits_processors"]
