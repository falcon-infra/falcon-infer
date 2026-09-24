# SPDX-License-Identifier: Apache-2.0
"""Contracts keeping P-node bank behavior isolated from ordinary decode."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_shared_pool_free_path_is_explicitly_split_by_residency_mode():
    source = (ROOT / "vllm/v1/core/block_pool.py").read_text(encoding="utf-8")
    assert "if isinstance(self.allocator, PrefillLayerBundlePool)" in source
    assert "def _free_prefill_blocks" in source
    assert "bundle_id_for_block(self.owner, block.block_id)" in source


def test_coordinator_rolls_back_only_layerwise_child_allocations():
    source = (ROOT / "vllm/v1/core/kv_cache_coordinator.py").read_text(
        encoding="utf-8"
    )
    assert "if not self.layerwise_prefill_p_node:" in source
    assert "else:\n            try:" in source
    assert "Layerwise-prefill allocation rollback lost" in source


def test_ordinary_concurrency_and_runner_block_state_keep_baseline_owners():
    utils = (ROOT / "vllm/v1/core/kv_cache_utils.py").read_text(encoding="utf-8")
    runner = (ROOT / "vllm/v1/worker/npu_runner_state.py").read_text(
        encoding="utf-8"
    )
    assert 'if mode != "prefill_child":' in utils
    assert "Only P-node banked requests use the layerwise metadata helper" in runner
    assert "req_state.block_ids = new_block_ids" in runner
