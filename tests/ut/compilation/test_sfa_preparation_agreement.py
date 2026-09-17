# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Static contracts for EP-wide SFA preparation agreement."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _function(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    return ast.unparse(node)


def test_supported_ep_topology_validates_membership_and_selects_ep_group():
    path = ROOT / "vllm/compilation/ascend/sfa_fail_stop.py"
    source = path.read_text(encoding="utf-8")
    method = _function(path, "preparation_error_groups")
    assert "enable_expert_parallel" in method
    assert "distributed_executor_backend" in method
    assert "tp.world_size" in method and "dp.world_size" in method
    assert "ep.world_size" in method
    assert "EP membership" in source
    assert "prepare_agreement_ep" in source


def test_runner_caches_startup_groups_and_keeps_dynamic_fallback():
    runner = ROOT / "vllm/v1/worker/npu_model_runner.py"
    source = runner.read_text(encoding="utf-8")
    coordinate = _function(runner, "_coordinate_sfa_full_graph_preparation")
    assert "_sfa_preparation_groups" in source
    assert "preparation_error_groups" in source
    assert "get_ep_group" in source
    assert "prepare_agreement_" in coordinate
    assert "record_function_or_nullcontext" in coordinate
    assert "get_tp_group" in coordinate and "get_dp_group" in coordinate


def test_dp_layout_agreement_is_traced_without_changing_the_wire_protocol():
    runner = ROOT / "vllm/v1/worker/npu_model_runner.py"
    method = _function(runner, "_sync_metadata_across_dp")
    assert "ascend::dp_layout_agreement" in method
    assert "npu_all_reduce" in method
