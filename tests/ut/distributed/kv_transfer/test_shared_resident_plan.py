# SPDX-License-Identifier: Apache-2.0
"""Dependency-light contracts for the native shared resident planner."""

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[4]


class _FakeTensor:
    def __init__(self, shape, name="tensor"):
        self.shape = tuple(shape)
        self.name = name

    def __getitem__(self, key):
        if isinstance(key, tuple):
            shape = list(self.shape)
            for index in key:
                if isinstance(index, int) and shape:
                    shape.pop(0)
                elif isinstance(index, slice) and shape:
                    shape[0] = self.shape[0] if index.stop is None else min(
                        self.shape[0], index.stop
                    )
            return _FakeTensor(shape, self.name + "[view]")
        if isinstance(key, slice):
            size = self.shape[0] if key.stop is None else min(self.shape[0], key.stop)
            return _FakeTensor((size, *self.shape[1:]), self.name + "[prefix]")
        return _FakeTensor(self.shape[1:], self.name + "[item]")


def _load_resident_module():
    module_name = "native_shared_resident_plan_contract"
    previous_torch = sys.modules.get("torch")
    torch = ModuleType("torch")
    torch.Tensor = _FakeTensor
    torch.device = object
    sys.modules["torch"] = torch
    spec = importlib.util.spec_from_file_location(
        module_name,
        ROOT
        / "vllm/distributed/kv_transfer/ascend/sparse_offload/resident_sorted_cache.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
        if previous_torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = previous_torch
    return module


@pytest.fixture(scope="module")
def resident_api():
    return _load_resident_module()


def _workspace(api):
    return api.SortedResidentWorkspace(
        *[
            _FakeTensor((2, 4, 8), name)
            for name in (
                "shard_packed",
                "shard_mapping",
                "shard_counts",
                "prior_slots",
                "shard_miss_tokens",
                "shard_miss_positions",
                "shard_evictable_slots",
            )
        ],
        _FakeTensor((2, 8), "miss_tokens"),
        _FakeTensor((2, 16), "miss_counts"),
        _FakeTensor((2, 8), "target_slots"),
    )


def test_shared_plan_keeps_common_storage_and_capacity_views(resident_api):
    state = resident_api.SortedResidentState(
        _FakeTensor((3, 4, 8), "tokens"),
        _FakeTensor((3, 4, 8), "slots"),
        _FakeTensor((3, 4, 16), "counts"),
        _FakeTensor((3, 8), "generations"),
        2,
    )
    workspace = _workspace(resident_api)
    topk = _FakeTensor((2, 1, 8), "topk")
    plan = resident_api.SharedResidentPlan(
        ("layers.0.attn", "layers.1.attn"),
        state,
        workspace,
        topk,
        1,
        128,
    )

    assert set(plan.workspace_views) == {1, 2}
    assert plan.plans[1][0].name.startswith("topk")
    assert plan.plans[2][0].name.startswith("topk")
    assert plan.writes[0] is topk
    assert len(plan.writes) == 15  # top-k + 4 state + 10 workspace tensors
    bound_state, bound_workspace = plan.planner_storage(plan.writes, 1)
    assert bound_state.tokens is state.tokens
    assert bound_workspace.miss_tokens.shape[0] == 1


def _tree(path: str) -> ast.Module:
    return ast.parse((ROOT / path).read_text(encoding="utf-8"))


def _method(path: str, name: str) -> ast.FunctionDef:
    node = next(
        node
        for node in ast.walk(_tree(path))
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    return node


def test_shared_operator_schema_and_clone_rewrite_are_explicit():
    source = (ROOT / "vllm/model_executor/layers/ascend/mla.py").read_text(
        encoding="utf-8"
    )
    assert "def sfa_forward_pre_shared(" in source
    assert "resident_reads: list[torch.Tensor]" in source
    assert "resident_writes: list[torch.Tensor]" in source
    assert 'mutates_args=["output", "resident_writes"]' in source

    compiler = (ROOT / "vllm/compilation/ascend/compiler_interface.py").read_text(
        encoding="utf-8"
    )
    assert "def _reuse_shared_resident_buffers" in compiler
    assert "_only_clone_these_tensors" in compiler
    assert "_only_clone_these_bases" in compiler


def test_preemption_and_forward_failure_are_fail_closed():
    runner = _method("vllm/v1/worker/npu_model_runner.py", "_update_states")
    update_source = ast.unparse(runner)
    assert "preempted_req_ids" in update_source
    assert "invalidate" in update_source

    model_forward = _method("vllm/v1/worker/npu_model_runner.py", "_model_forward")
    forward_source = ast.unparse(model_forward)
    assert "_prepare_shared_resident_plans" in forward_source
    assert "_shared_resident_failed" in forward_source


def test_shared_plan_is_opt_in_and_sfa_candidate_defers_allocation():
    envs = (ROOT / "vllm/envs_ascend.py").read_text(encoding="utf-8")
    assert '"VLLM_ASCEND_SFA_SHARED_RESIDENT_PLAN"' in envs
    assert 'os.getenv("VLLM_ASCEND_SFA_SHARED_RESIDENT_PLAN", "0")' in envs

    sfa = (ROOT / "vllm/v1/attention/backends/ascend/sfa_v1.py").read_text(
        encoding="utf-8"
    )
    assert "shared_resident_candidate" in sfa
    assert "initialize_sorted_resident_cache" in sfa
    assert "shared resident route disagrees" in sfa


def test_runner_qualifies_same_slot_layout_before_execution():
    runner = (ROOT / "vllm/v1/worker/npu_model_runner.py").read_text(
        encoding="utf-8"
    )
    assert "_validate_shared_resident_layout" in runner
    assert "cache.stride() != reference_cache.stride()" in runner
    assert "cache.dtype != reference_cache.dtype" in runner
    assert "non-CP " in runner and "SFA layout" in runner
