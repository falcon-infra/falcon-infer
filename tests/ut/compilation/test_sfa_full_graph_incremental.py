# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Static contracts for startup-owned full-graph transfer reuse."""

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


def test_full_graph_registers_transfer_tables_and_tracks_source_tokens():
    path = ROOT / "vllm/compilation/ascend/sfa_full_graph.py"
    source = path.read_text(encoding="utf-8")
    assert "_transfer_bundles" in source
    assert "_table_history" in source
    assert "def register_transfers" in source
    assert "def get_transfers" in source
    bind = _function(path, "bind_sources")
    register = _function(path, "register_transfers")
    assert "plan_bind_update" in register
    assert "binding_token" in bind
    assert "lanes=lanes" in bind
    assert "_table_history" in bind


def test_live_runner_reuses_metadata_once_and_requires_registered_transfers():
    runner = ROOT / "vllm/v1/worker/npu_model_runner.py"
    source = _function(runner, "_model_forward")
    assert "register_transfers" in source
    assert "get_transfers" in source
    assert "prepare_full_graph_metadata" in source
    assert "self._sfa_full_graph.sealed" in source
    assert "changed after startup" in source
    assert "seen = set()" in source
    assert "bind_sources(source, request_ids)" in source


def test_sfa_layer_separates_metadata_refresh_from_transfer_selection():
    path = ROOT / "vllm/v1/attention/backends/ascend/sfa_v1.py"
    source = path.read_text(encoding="utf-8")
    metadata = _function(path, "prepare_full_graph_metadata")
    layer = _function(path, "prepare_full_graph_layer")
    assert "_prepare_sfa_remap_boundary" in metadata
    assert "reshape_cache_event = None" in metadata
    assert "prepare_full_graph_metadata" in layer
    assert "bind_source" in layer


def test_idle_replay_reuses_prepared_boundary_staging_without_live_validation():
    remap = ROOT / "vllm/v1/attention/backends/ascend/sfa_remap_boundary.py"
    sfa = ROOT / "vllm/v1/attention/backends/ascend/sfa_v1.py"
    buffer_source = remap.read_text(encoding="utf-8")
    prepare_source = sfa.read_text(encoding="utf-8")
    assert "def update_prepared" in buffer_source
    assert "def _commit" in buffer_source
    assert "buffer.update_prepared" in prepare_source
    assert "reuse_dummy" in prepare_source


def test_sealed_idle_validation_is_layer_grouped_and_signature_checked():
    graph = ROOT / "vllm/compilation/ascend/sfa_full_graph.py"
    runner = ROOT / "vllm/v1/worker/npu_model_runner.py"
    validate = _function(graph, "validate_idle_metadata")
    run = _function(graph, "prepare_run")
    forward = _function(runner, "_model_forward")
    assert "expected_by_layer" in validate
    assert "tensor_signature(inputs)" in validate
    assert "self.sealed and entry is not None" in run
    assert "metadata_groups" in forward
    assert "validate_idle_metadata" in forward
    assert "reuse_dummy=True" in forward
