# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for the opt-in serving stall observer."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def stall_module(monkeypatch):
    path = Path(__file__).resolve().parents[3] / "vllm/v1/worker/sfa_serving_stall.py"
    spec = importlib.util.spec_from_file_location(
        "tested_sfa_serving_stall", path
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def worker():
    return SimpleNamespace(
        rank=5,
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(data_parallel_rank=1, tensor_parallel_size=4)
        ),
        model_runner=SimpleNamespace(
            _sfa_full_graph=SimpleNamespace(replay_count=10)
        ),
        execute_model=Mock(return_value="hidden"),
        sample_tokens=Mock(return_value=None),
        execute_dummy_batch=Mock(return_value=None),
    )


def test_observer_wraps_active_and_idle_rpc_paths_and_restores(stall_module):
    target = worker()
    timing = SimpleNamespace(decode_steps=2)
    original = {
        name: getattr(target, name)
        for name in ("execute_model", "sample_tokens", "execute_dummy_batch")
    }
    observer = stall_module.ServingStallDiagnostic(target, timing).start()
    target.execute_model()
    target.sample_tokens()
    target.execute_dummy_batch()
    assert observer.progress[0] == "execute_dummy_batch.return"
    observer.close()
    for name, method in original.items():
        assert getattr(target, name) is method
    assert observer.thread is not None
    assert not observer.thread.is_alive()


def test_snapshot_is_host_only_and_emitted_once(stall_module, monkeypatch):
    target = worker()
    timing = SimpleNamespace(decode_steps=7)
    observer = stall_module.ServingStallDiagnostic(target, timing)
    observer.progress = ("execute_dummy_batch.enter", 0.0)
    monkeypatch.setattr(stall_module.time, "monotonic", lambda: 31.0)
    snapshot = observer.snapshot_if_stalled()
    assert snapshot["event"] == "no_worker_rpc_progress"
    assert snapshot["dp_rank"] == 1 and snapshot["tp_rank"] == 1
    assert snapshot["decode_steps"] == 7
    assert snapshot["root_replays_submitted"] == 0
    assert observer.snapshot_if_stalled() is None
    observer.close()


def test_progress_change_invalidates_stale_snapshot(stall_module, monkeypatch):
    target = worker()
    observer = stall_module.ServingStallDiagnostic(target, SimpleNamespace(decode_steps=0))
    progress = ("execute_model.enter", 0.0)
    observer.progress = progress
    monkeypatch.setattr(stall_module.time, "monotonic", lambda: 31.0)
    observer.progress = ("execute_model.return", 31.0)
    assert observer.snapshot_if_stalled() is None
    observer.close()


def test_start_failure_restores_partial_hooks(stall_module):
    target = worker()
    delattr(target, "execute_dummy_batch")
    observer = stall_module.ServingStallDiagnostic(target, SimpleNamespace(decode_steps=0))
    with pytest.raises(AttributeError):
        observer.start()
    assert observer.thread is None
