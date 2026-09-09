# SPDX-License-Identifier: Apache-2.0
"""Native SFA full-graph lane-layout contracts."""

import pytest

torch = pytest.importorskip("torch")

from vllm.v1.attention.backends.ascend.sfa_graph_layout import (
    FullGraphAttentionBuffers,
    pack_decode_lanes,
    unpack_decode_lanes,
)


def test_pack_and_unpack_preserve_mixed_request_lanes():
    topk = torch.tensor([[10], [11], [20]], dtype=torch.int32)
    boundaries = torch.tensor([4, 4, 8], dtype=torch.int32)
    row_requests = torch.tensor([0, 0, 1], dtype=torch.int32)
    query_ends = torch.tensor([2, 3], dtype=torch.int32)

    packed, packed_boundaries, packed_requests, inverse = pack_decode_lanes(
        topk, boundaries, row_requests, query_ends, width=2
    )

    assert packed.shape[0] == 4
    assert packed_boundaries.shape == (4,)
    assert packed_requests.tolist() == [0, 0, 1, -1]
    assert unpack_decode_lanes(packed, inverse, row_requests).tolist() == topk.tolist()


def test_attention_buffers_add_only_a_zero_kv_padding_lane():
    block = torch.ones((2, 3), dtype=torch.int32)
    indexer = torch.ones((2, 3), dtype=torch.int32)
    query_ends = torch.tensor([1, 2], dtype=torch.int32)
    seq_lens = torch.tensor([8, 16], dtype=torch.int32)
    buffers = FullGraphAttentionBuffers(3, block, indexer, query_ends, seq_lens)

    _, _, new_query_ends, new_seq_lens, new_cpu = buffers.update(
        block, indexer, query_ends, seq_lens, seq_lens, tokens=3
    )

    assert new_query_ends.tolist() == [1, 2, 3]
    assert new_seq_lens.tolist() == [8, 16, 0]
    assert new_cpu.tolist() == [8, 16, 0]

def test_full_graph_runner_requires_cudagraph_runtime_before_returning_output():
    """The full-graph branch must fall through when graph replay is disabled."""
    path = Path(__file__).resolve().parents[3] / "vllm/v1/worker/npu_model_runner.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_model_forward"
    )
    graph_branch = next(
        node
        for node in method.body
        if isinstance(node, ast.If)
        and any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "sfa_full_graph_enabled"
            for call in ast.walk(node.test)
        )
    )
    assert "context.cudagraph_runtime_mode != CUDAGraphMode.NONE" in ast.unparse(
        graph_branch.test
    )
