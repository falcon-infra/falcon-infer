# SPDX-License-Identifier: Apache-2.0
"""Native SFA full-graph lane-layout contracts."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

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


@pytest.mark.parametrize("requests", [1, 2, 8, 16])
@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("width", [1, 2])
def test_uniform_planner_bypasses_pack_and_matches_bounded(requests, padded, width):
    """Execute the native pre-compute planner branch, not a copied model."""
    path = Path(__file__).resolve().parents[3] / "vllm/v1/attention/backends/ascend/sfa_v1.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_cross_layer_pre_compute"
    )
    start = next(
        index
        for index, node in enumerate(method.body)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "graph_key"
            for target in node.targets
        )
    )
    code = compile(
        ast.Module(body=method.body[start:-1], type_ignores=[]), str(path), "exec"
    )
    capacity = requests + int(padded)
    tokens = capacity * width
    owners = torch.arange(capacity, dtype=torch.int32).repeat_interleave(width)
    owners[requests * width :] = -1
    topk = torch.arange(tokens * 7, dtype=torch.int32).reshape(tokens, 1, 7)
    topk[owners < 0] = -1
    boundary = torch.full((tokens,), 4096, dtype=torch.int32)
    boundary[owners < 0] = 0
    outputs = []
    for profile in ("fixed", "bounded"):
        pack = Mock(wraps=pack_decode_lanes)
        unpack = Mock(wraps=unpack_decode_lanes)
        table = torch.zeros(
            capacity + int(profile == "bounded"), 3, dtype=torch.int32
        )

        def plan(
            indices,
            boundaries,
            request_rows,
            blocks,
            selected,
            counts,
            slots,
            *args,
            **kwargs,
        ):
            assert kwargs["staged_mtp"] == width
            assert blocks.shape[0] == capacity
            assert torch.equal(request_rows, owners)
            assert torch.equal(boundaries, boundary)
            result = torch.where(
                request_rows[:, None, None] >= 0,
                indices + request_rows[:, None, None] * 1000,
                -1,
            )
            return result, selected, counts, slots

        key = SimpleNamespace(
            query_profile=(
                SimpleNamespace(value="decode_bounded")
                if profile == "bounded"
                else SimpleNamespace(value="spec_fixed")
            ),
            request_capacity=capacity,
            max_query_len=width,
        )
        namespace = dict(
            get_forward_context=lambda key=key: SimpleNamespace(
                staged_sfa_graph_key=key
            ),
            StagedSFAQueryProfile=SimpleNamespace(DECODE_BOUNDED=key.query_profile)
            if profile == "bounded"
            else SimpleNamespace(DECODE_BOUNDED=SimpleNamespace(value="other")),
            pack_decode_lanes=pack,
            unpack_decode_lanes=unpack,
            self=SimpleNamespace(_prepare_decode_sparse_indices=plan),
            topk_indices=topk,
            remap_boundary=boundary,
            row_req_indices=owners,
            request_block_table=table,
            actual_seq_lengths_query=torch.arange(1, capacity + 1).clamp(
                max=requests
            )
            * width,
            **{
                name: torch.zeros(1)
                for name in (
                    "selected_packed",
                    "selected_counts",
                    "target_slot_mapping",
                    "request_state_indices",
                    "request_state_generations",
                    "local_to_union_workspace",
                    "shard_packed_workspace",
                    "shard_mapping_workspace",
                    "shard_counts_workspace",
                )
            },
        )
        exec(code, namespace)
        assert pack.call_count == unpack.call_count == int(profile == "bounded")
        outputs.append(namespace["topk_indices"])
    assert torch.equal(outputs[0], outputs[1])

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
