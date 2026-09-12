# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for opt-in timing; these tests do not claim NPU performance."""

import ast
import importlib.util
import sys
import threading
from contextlib import nullcontext
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
    methods = [node for node in cls.body if getattr(node, "name", "") in ("benchmark_start_decode_timing", "benchmark_stop_decode_timing")]
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
