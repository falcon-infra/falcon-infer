# SPDX-License-Identifier: Apache-2.0
"""Dependency-light contracts for mixed-C8 full-graph resident ownership."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_mixed_c8_metadata_is_sized_after_full_graph_padding():
    sfa = (ROOT / "vllm/v1/attention/backends/ascend/sfa_v1.py").read_text(
        encoding="utf-8"
    )
    assert "max_num_seqs + 1" in sfa
    assert "Full-graph attention adds an attention-only padding request" in sfa
    assert sfa.index("sfa_full_graph") < sfa.index(
        "self._mixed_indexer_metadata.update"
    )
    assert "indexer_c8_block_table" in sfa
    assert "indexer_c8_slot_mapping" in sfa


def test_shared_resident_c8_requires_paired_physical_storage():
    runner = (ROOT / "vllm/v1/worker/npu_model_runner.py").read_text(
        encoding="utf-8"
    )
    assert "shared resident C8 producer requires paired" in runner
    assert "indexer[0].dtype != torch.int8" in runner
    assert "indexer[1].dtype != torch.float16" in runner
    assert "or impl.use_sparse_c8_indexer" not in runner
