# SPDX-License-Identifier: Apache-2.0
"""Dependency-light contracts for opt-in native retrieval overlap."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_retrieval_overlap_is_default_off_and_full_graph_gated():
    env = (ROOT / "vllm/envs_ascend.py").read_text(encoding="utf-8")
    runner = (ROOT / "vllm/v1/worker/npu_model_runner.py").read_text(
        encoding="utf-8"
    )
    module = (
        ROOT / "vllm/compilation/ascend/sfa_retrieval_overlap.py"
    ).read_text(encoding="utf-8")
    assert 'os.getenv("VLLM_ASCEND_SFA_SHARED_RETRIEVAL_OVERLAP", "0")' in env
    assert "requires full graphs and" in runner
    assert "CaptureProbe" in module
    assert "pending_capture_launch" in module


def test_retrieval_overlap_uses_fixed_destinations_and_consumer_join():
    sfa = (ROOT / "vllm/v1/attention/backends/ascend/sfa_v1.py").read_text(
        encoding="utf-8"
    )
    mla = (ROOT / "vllm/model_executor/layers/ascend/mla.py").read_text(
        encoding="utf-8"
    )
    compiler = (
        ROOT / "vllm/compilation/ascend/compiler_interface.py"
    ).read_text(encoding="utf-8")
    overlap = (
        ROOT / "vllm/compilation/ascend/sfa_retrieval_overlap.py"
    ).read_text(encoding="utf-8")
    full_graph = (
        ROOT / "vllm/compilation/ascend/sfa_full_graph.py"
    ).read_text(encoding="utf-8")
    assert "_shared_retrieval_edges" in sfa
    assert "prefetch=prefetch" in sfa
    assert "sfa_forward_post_prefetch" in mla
    assert "destinations" in overlap
    assert "resident_writes" in compiler
    assert "retrieval_overlap.clear()" in full_graph
