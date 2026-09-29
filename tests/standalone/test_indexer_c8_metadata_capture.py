# SPDX-License-Identifier: Apache-2.0
"""Exercise the native paired-indexer mapper at long graph-capture widths."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")
pytest.importorskip("triton")

from vllm.model_executor.layers.ascend.triton.spec_decode.indexer_c8_metadata import (  # noqa: E402
    map_indexer_metadata,
)


def _block_map(slots: int):
    mapping = torch.empty(18 * slots, dtype=torch.int32)
    for bundle in range(slots):
        logical = [
            *range(8 * bundle, 8 * bundle + 8),
            *range(8 * (slots + bundle), 8 * (slots + bundle) + 8),
            16 * slots + bundle,
            17 * slots + bundle,
        ]
        physical = [
            *range(16 * bundle, 16 * bundle + 16),
            16 * slots + 2 * bundle,
            16 * slots + 2 * bundle + 1,
        ]
        mapping[logical] = torch.tensor(physical, dtype=torch.int32)
    return mapping


@pytest.mark.parametrize("width", [1404, 8208])
@pytest.mark.parametrize("slot_dtype", [torch.int32, torch.int64])
def test_long_native_mapper_eager_and_captured(width, slot_dtype):
    if not torch.npu.is_available():
        pytest.skip("requires NPU")
    mapping = _block_map(4)
    table = (torch.arange(17 * width, dtype=torch.int32).view(17, width) % len(mapping))
    table[-1].zero_()
    slots = torch.arange(32, dtype=slot_dtype) * 128 + 7
    slots[-1] = -1
    expected_table = mapping[table.long()]
    expected_slots = torch.where(
        slots >= 0,
        mapping[(slots.clamp(min=0) // 128).long()].long() * 128 + slots % 128,
        -1,
    )
    device_table, device_slots, device_map = table.npu(), slots.npu(), mapping.npu()
    out_table = torch.empty_like(device_table)
    out_slots = torch.empty(32, dtype=torch.int64, device="npu")

    def launch():
        map_indexer_metadata(
            device_table, device_slots, device_map, out_table, out_slots
        )

    launch()
    torch.npu.synchronize()
    assert torch.equal(out_table.cpu(), expected_table)
    assert torch.equal(out_slots.cpu(), expected_slots)

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        launch()
    graph.replay()
    torch.npu.synchronize()
    assert torch.equal(out_table.cpu(), expected_table)
    assert torch.equal(out_slots.cpu(), expected_slots)
