# SPDX-License-Identifier: Apache-2.0
"""Device mapping from shared logical indexer IDs to physical C8 banks."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _map_indexer_metadata(
    table,
    slots,
    mapping,
    out_table,
    out_slots,
    TABLE_SIZE: tl.constexpr,
    SLOT_SIZE: tl.constexpr,
    TABLE_WIDTH: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    indices = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    logical = tl.load(
        table + (indices // TABLE_WIDTH) * TABLE_STRIDE + indices % TABLE_WIDTH,
        indices < TABLE_SIZE,
        other=-1,
    )
    physical = tl.load(
        mapping + logical,
        (indices < TABLE_SIZE) & (logical >= 0),
        other=-1,
    )
    tl.store(out_table + indices, physical, indices < TABLE_SIZE)

    slots_logical = tl.load(
        slots + indices,
        indices < SLOT_SIZE,
        other=-1,
    )
    slot_block = slots_logical // 128
    slot_physical = tl.load(
        mapping + slot_block,
        (indices < SLOT_SIZE) & (slots_logical >= 0),
        other=-1,
    )
    tl.store(
        out_slots + indices,
        tl.where(
            slots_logical >= 0,
            slot_physical * 128 + slots_logical % 128,
            -1,
        ),
        indices < SLOT_SIZE,
    )


def map_indexer_metadata(
    table: torch.Tensor,
    slots: torch.Tensor,
    mapping: torch.Tensor,
    out_table: torch.Tensor,
    out_slots: torch.Tensor,
) -> None:
    """Map padded block tables and token slots into fixed physical C8 views."""
    _map_indexer_metadata[
        (triton.cdiv(max(table.numel(), slots.numel()), 256),)
    ](
        table,
        slots,
        mapping,
        out_table,
        out_slots,
        table.numel(),
        slots.numel(),
        table.shape[1],
        table.stride(0),
        BLOCK=256,
    )
