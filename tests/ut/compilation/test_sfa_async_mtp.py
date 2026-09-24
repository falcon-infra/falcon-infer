# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Static contracts for the opt-in native async MTP runner."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "vllm/v1/worker/sfa_async_mtp.py"


def test_async_runner_has_explicit_eligibility_and_reconciliation_boundaries():
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AsyncSFAModelRunner"
    )
    methods = {
        node.name: ast.unparse(node)
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
    }
    assert {
        "_eligible",
        "_update_states",
        "_prepare_inputs",
        "_reconcile",
        "_build_attention_metadata",
        "_model_forward",
        "_remember",
    } <= methods.keys()
    assert "super()._update_states(scheduled)" in methods["_reconcile"]
    assert "prepare_async_mtp_kernel" in methods["_build_attention_metadata"]
    assert "indexer_c8_slot_mapping" in methods["_build_attention_metadata"]
    assert "self._reconcile()" in methods["_model_forward"]
    assert "self._remember" in methods["_model_forward"]
    assert "_async_counts_epoch" in methods["_eligible"]
    assert "num_common_prefix_blocks" in methods["_eligible"]
    assert "synchronize_input_prep" in methods
    assert "ExitStack" in methods["_dummy_run"]
    assert "_input_prep" in methods["_dummy_run"]
    assert "_ensure_host_staging_ready" in methods
    assert "_pad_query_start_loc_for_fia" in methods
    assert "self._staged_sfa_layer_names" in methods["_remember"]


def test_continuing_mtp_clears_previous_cold_resume_proof():
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AsyncSFAModelRunner"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_build_attention_metadata"
    )
    assignments = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and target.attr == "cold_compact_resumes"
            for target in node.targets
        )
    ]
    assert len(assignments) == 1
    assignment = assignments[0]
    assert isinstance(assignment.value, ast.Tuple)
    assert not assignment.value.elts


def test_runner_opt_in_is_v1_only_and_default_off():
    env = (ROOT / "vllm/envs_ascend.py").read_text(encoding="utf-8")
    worker = (ROOT / "vllm/v1/worker/npu_worker.py").read_text(encoding="utf-8")
    assert "VLLM_ASCEND_SFA_ASYNC_MTP_PREP" in env
    assert 'os.getenv("VLLM_ASCEND_SFA_ASYNC_MTP_PREP", "0")' in env
    assert "requires the V1 model runner" in worker
    assert "AsyncSFAModelRunner" in worker
