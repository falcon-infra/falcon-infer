# SPDX-License-Identifier: Apache-2.0
"""Host coverage for native prefill bank physical metadata isolation."""

import pytest

torch = pytest.importorskip("torch")

from vllm.v1.worker.dsa_shared_pool import MixedIndexerMetadata  # noqa: E402


def _block_map(slots: int) -> tuple[int, ...]:
    mapping = [0] * (18 * slots)
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
        for index, value in zip(logical, physical):
            mapping[index] = value
    return tuple(mapping)


def test_flashcomm1_prefill_bank_views_are_independent():
    mapping = _block_map(4)
    held = []
    for bank in (18, 72, 18):
        builder = MixedIndexerMetadata(2, 8, 8, torch.device("cpu"), block_map=mapping)
        table = torch.full((2, 8), bank, dtype=torch.int32)
        slots = torch.tensor([bank * 128 + i for i in range(5)] + [-1] * 3)
        physical_table, physical_slots = builder.update(table, slots)
        assert physical_table.tolist() == [mapping[bank]] * 16
        assert physical_slots.tolist() == [mapping[bank // 128] * 128 + i for i in range(5)] + [-1] * 3
        held.append((builder, physical_table, physical_slots))
    assert [int(table[0, 0]) for _, table, _ in held] == [36, 144, 36]
    assert len({table.data_ptr() for _, table, _ in held}) == 3
