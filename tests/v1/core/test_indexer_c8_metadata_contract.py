# SPDX-License-Identifier: Apache-2.0
"""Dependency-light contracts for shared-LCM physical C8 metadata."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_metadata_builder_uses_paired_lcm_capacity_and_mapping():
    builder = (ROOT / "vllm/v1/attention/backends/ascend/sfa_v1.py").read_text(
        encoding="utf-8"
    )
    worker = (ROOT / "vllm/v1/worker/dsa_shared_pool.py").read_text(
        encoding="utf-8"
    )
    runner = (ROOT / "vllm/v1/worker/npu_model_runner.py").read_text(
        encoding="utf-8"
    )
    assert "indexer_hbm_block_map" in builder
    assert "blocks_per_bundle = 18 if block_map is not None else 9" in builder
    assert "block_map=block_map" in builder
    assert "self.block_map" in worker
    assert "self.block_map[table.clamp(min=0).long()]" in worker
    assert "paired_bank_indexer_block_map(paired_bank_slots)" in runner


def test_mixed_c8_spec_preserves_pairing_and_alignment_policy():
    spec = (ROOT / "vllm/v1/kv_cache_interface.py").read_text(encoding="utf-8")
    assert "indexer_paired_banks: bool = False" in spec
    assert "shared_pool_alignment_bytes: int = 0" in spec
    assert "not self.indexer_paired_banks" in spec
    assert "specs[0].indexer_paired_banks" in spec


def test_async_mtp_uses_physical_table_when_available():
    kernel = (
        ROOT / "vllm/model_executor/layers/ascend/triton/spec_decode/async_mtp.py"
    ).read_text(encoding="utf-8")
    runner = (ROOT / "vllm/v1/worker/sfa_async_mtp.py").read_text(
        encoding="utf-8"
    )
    assert "if table_c8 is None" in kernel
    assert "table_c8 + rows * stride_c8" in kernel
    assert '"table_c8": item.indexer_c8_block_table' in runner
