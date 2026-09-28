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


def test_npu_runner_and_sfa_gate_layerwise_paths_to_p_nodes():
    runner = (ROOT / "vllm/v1/worker/npu_model_runner.py").read_text(
        encoding="utf-8"
    )
    sfa = (ROOT / "vllm/v1/attention/backends/ascend/sfa_v1.py").read_text(
        encoding="utf-8"
    )
    assert "if self.layerwise_prefill_p_node:" in runner
    assert "not self.layerwise_prefill_p_node" in runner
    assert "self._sfa_preprocess_with_mlapo" in sfa
    assert "self._sfa_preprocess_with_mlapo_after_layerwise_wait" in sfa
    assert "and layerwise_prefill_transfer_window_supported()" in sfa


def test_dense_pd_decode_rows_are_not_reclassified_as_padding():
    sfa = (ROOT / "vllm/v1/attention/backends/ascend/sfa_v1.py").read_text(
        encoding="utf-8"
    )
    assert "if fixed_decode_width" in sfa
    assert "Dense PD prefix loads can leave a real final prompt" in sfa


def test_mtp_prefill_saves_flush_without_an_next_layer_callback():
    sfa = (ROOT / "vllm/v1/attention/backends/ascend/sfa_v1.py").read_text(
        encoding="utf-8"
    )
    assert "self._layerwise_prefill_mtp" in sfa
    assert 'getattr(get_forward_context(), "is_draft_model", False)' in sfa
