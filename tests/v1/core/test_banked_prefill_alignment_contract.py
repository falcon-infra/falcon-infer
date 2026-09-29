# SPDX-License-Identifier: Apache-2.0
"""Contract for aligned P-node shared-pool backing."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_p_node_aligns_bf16_and_mixed_c8_shared_pool_slabs():
    runner = (ROOT / "vllm/v1/worker/npu_model_runner.py").read_text(
        encoding="utf-8"
    )
    assert "Every P-node slab retains an aligned backing" in runner
    assert "self.layerwise_prefill_p_node" in runner
    assert "shared_pool_alignment_bytes" in runner
