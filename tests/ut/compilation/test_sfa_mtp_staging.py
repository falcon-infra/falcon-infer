# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Static contracts for lazy native async-MTP host staging fences."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PATH = ROOT / "vllm/v1/worker/sfa_async_mtp.py"


def methods():
    tree = ast.parse(PATH.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AsyncSFAModelRunner"
    )
    return {
        node.name: ast.unparse(node)
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
    }


def test_fast_path_defers_fence_until_host_writer():
    source = methods()
    assert "_async_host_write = False" in source["synchronize_input_prep"]
    assert "prepare_inputs_event.record" in source["synchronize_input_prep"]
    assert "prepare_inputs_event.synchronize" in source["_ensure_host_staging_ready"]
    assert "_ensure_host_staging_ready" in source["_prepare_inputs"]
    assert "_async_query_layout = None" in source["_prepare_inputs"]


def test_transition_and_dummy_paths_clear_async_state():
    source = methods()
    assert "_ensure_host_staging_ready" in source["_update_states"]
    assert "_ensure_host_staging_ready" in source["_apply_staged_sfa_route"]
    assert "_async_snapshot = self._async_query_layout = None" in source["_dummy_run"]
    assert "self._async_snapshot = self._async_pending" in source["execute_model"]
