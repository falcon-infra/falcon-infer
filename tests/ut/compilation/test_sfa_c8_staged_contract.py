# SPDX-License-Identifier: Apache-2.0
"""Static native contracts for staged C8 key/scale publication."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SFA = ROOT / "vllm/v1/attention/backends/ascend/sfa_v1.py"
RUNNER = ROOT / "vllm/v1/worker/npu_model_runner.py"


def _class_methods(path: Path, class_name: str) -> dict[str, str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return {
        node.name: ast.unparse(node)
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
    }


def test_staged_sfa_retains_c8_scale_plane_and_unbundled_binding():
    methods = _class_methods(SFA, "AscendSFAImpl")
    pre = methods["_cross_layer_pre_compute"]
    binding = methods["_cross_layer_kv_cache"]
    assert "indexer_scale_cache" in pre
    assert "scale_updates" in pre
    assert "npu_scatter_nd_update_" in pre
    assert "_dsa_idx_scale_cache_t" in binding
    assert "Unbundled C8 indexer requires key and scale caches" in binding


def test_staged_sfa_accepts_int8_keys_and_fp16_scales():
    method = _class_methods(SFA, "AscendSFAImpl")[
        "_cross_layer_ineligible_reason"
    ]
    assert "the sparse C8 indexer is enabled" not in method
    assert "torch.int8" in method
    assert "torch.float16" in method
    assert "share block ownership" in method


def test_runner_allocates_scale_sidecar_and_uses_shared_geometry():
    source = RUNNER.read_text(encoding="utf-8")
    assert "dsa_shared_block_layout" in source
    assert "C8 shared slab and scale sidecar disagree" in source
    assert "raw_tensor[shared_bytes:]" in source
    assert "indexer_scale=" in source
    assert "raws[1]" in source
