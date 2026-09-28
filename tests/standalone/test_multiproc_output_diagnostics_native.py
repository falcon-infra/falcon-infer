# SPDX-License-Identifier: Apache-2.0
"""Exercise native multiprocess output diagnostics on sync and async outputs."""

import ast
import queue
import time
from enum import Enum, auto
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "vllm/v1/executor/multiproc_executor.py"


class AsyncModelRunnerOutput:
    """Host stand-in for the production async output protocol."""

    def get_output(self):  # pragma: no cover - overridden by the test object
        raise NotImplementedError


def _load_worker(monkeypatch, *, enabled=True):
    source = ast.parse(SOURCE.read_text(encoding="utf-8"))
    worker = next(
        node
        for node in source.body
        if isinstance(node, ast.ClassDef) and node.name == "WorkerProc"
    )
    selected_names = {
        "ResponseStatus",
        "enqueue_output",
        "handle_output",
        "_handle_output_response",
        "_enqueue_output_response",
    }
    worker.body = [
        node
        for node in worker.body
        if getattr(node, "name", None) in selected_names
    ]
    worker.bases = []
    worker.decorator_list = []

    globals_ns = {
        "AsyncModelRunnerOutput": AsyncModelRunnerOutput,
        "Any": object,
        "Enum": Enum,
        "auto": auto,
        "time": time,
        "cold_perf_enabled": lambda: enabled,
        "log_cold_perf_event": lambda *_args, **fields: events.append(
            ("event", fields)
        ),
        "log_cold_perf_process_event": lambda *_args, **fields: events.append(
            ("process", fields)
        ),
    }
    events = []
    for node in source.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name)
            and target.id.startswith(("_COLD_PERF_", "_SLOW_", "_worker_"))
            for target in node.targets
        ):
            module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
            exec(compile(module, str(SOURCE), "exec"), globals_ns)
    module = ast.fix_missing_locations(ast.Module(body=[worker], type_ignores=[]))
    exec(compile(module, str(SOURCE), "exec"), globals_ns)
    instance = object.__new__(globals_ns["WorkerProc"])
    instance.rank = 0
    instance.use_async_scheduling = False
    instance.async_output_queue = queue.Queue()
    responses = []
    instance.worker_response_mq = SimpleNamespace(enqueue=responses.append)
    return instance, responses, events, globals_ns


class AsyncResult(AsyncModelRunnerOutput):
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def get_output(self):
        self.calls += 1
        assert self.calls == 1
        return self.payload


def test_native_diagnostics_preserve_sync_results_and_unwrap_async_results(monkeypatch):
    worker, responses, events, ns = _load_worker(monkeypatch)
    sync_result = SimpleNamespace()
    setattr(sync_result, ns["_COLD_PERF_REQUEST_IDS"], ("sync",))

    worker.handle_output(sync_result)
    assert responses == [(worker.ResponseStatus.SUCCESS, sync_result)]
    assert not hasattr(sync_result, ns["_COLD_PERF_QUEUED_NS"])
    assert not events

    responses.clear()
    worker.use_async_scheduling = True
    async_result = AsyncResult(SimpleNamespace())
    setattr(async_result, ns["_COLD_PERF_REQUEST_IDS"], ("async",))
    worker.handle_output(async_result)
    assert responses == []
    worker.enqueue_output(worker.async_output_queue.get_nowait())
    assert responses == [(worker.ResponseStatus.SUCCESS, async_result.payload)]
    assert async_result.calls == 1
    assert events


def test_native_diagnostics_disabled_path_keeps_original_output(monkeypatch):
    worker, responses, events, _ = _load_worker(monkeypatch, enabled=False)
    result = SimpleNamespace()
    worker.handle_output(result)
    assert responses == [(worker.ResponseStatus.SUCCESS, result)]
    assert not events
