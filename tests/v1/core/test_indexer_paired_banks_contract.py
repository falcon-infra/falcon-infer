# SPDX-License-Identifier: Apache-2.0
"""Host contracts for paired mixed-C8 shared-pool geometry."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _api():
    path = ROOT / "vllm/v1/core/dsa_shared_pool.py"
    spec = importlib.util.spec_from_file_location("paired_dsa_layout", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _layout(api, capacity=7):
    latent = NS(
        page_size_bytes=147456,
        sparse_head_dim=(512, 64),
        dtype=NS(itemsize=2),
    )
    indexer = NS(
        page_size_bytes=33024,
        sparse_head_dim=(128,),
        dtype=NS(itemsize=2),
        shared_indexer_key_dtype=NS(itemsize=2),
        c8_k_cache_dtype=NS(itemsize=1),
        indexer_scale_page_size_bytes=256,
        indexer_scale_layer_count=16,
        indexer_paired_banks=True,
    )
    return api.dsa_shared_block_layout(latent, indexer, capacity)


def _merge(ranges):
    merged = []
    for start, end in sorted(ranges):
        if merged and merged[-1][1] == start:
            merged[-1] = merged[-1][0], end
        else:
            merged.append((start, end))
    return merged


@pytest.mark.parametrize("capacity", [1, 2, 7, 16])
def test_paired_layout_maps_two_indexer_banks_without_overlap(capacity):
    api = _api()
    layout = _layout(api, capacity)
    slots = layout.slot_count
    assert layout.latent_blocks_per_bundle == 2
    assert layout.indexer_blocks_per_bundle == 18
    mapping = api.paired_bank_indexer_block_map(slots)
    assert sorted(mapping) == list(range(18 * slots))
    assert mapping[0] == 0

    indexer = api.DSASharedBlockOwner.INDEXER
    all_ids = []
    for bundle in range(1, slots):
        blocks = layout.block_ids_for_bundle(indexer, bundle)
        all_ids.extend(blocks)
        assert len(blocks) == 18
        assert all(layout.bundle_id_for_block(indexer, block) == bundle for block in blocks)
        for quantized, page_size in ((True, 16384), (False, 32768)):
            physical = [mapping[block] if quantized else block for block in blocks]
            indexer_ranges = _merge(
                [(block * page_size, (block + 1) * page_size) for block in physical]
            )
            for other in range(1, slots):
                latent_ranges = [
                    (2 * other * 131072, 2 * (other + 1) * 131072),
                    (
                        2 * slots * 131072 * (1 if quantized else 2)
                        + 2 * other * 16384,
                        2 * slots * 131072 * (1 if quantized else 2)
                        + 2 * (other + 1) * 16384,
                    ),
                ]
                overlaps = any(
                    max(first, second) < min(last, end)
                    for first, last in indexer_ranges
                    for second, end in latent_ranges
                )
                assert overlaps == (bundle == other)
    assert len(set(all_ids)) == 18 * capacity
    assert layout.allocation_bytes_per_bundle(79, 22) == 25141248


def test_paired_layout_maps_two_indexer_banks_and_charges_bf16_owners():
    api = _api()
    latent = NS(
        page_size_bytes=147456,
        sparse_head_dim=(512, 64),
        dtype=NS(itemsize=2),
    )
    indexer = NS(
        page_size_bytes=33024,
        sparse_head_dim=(128,),
        dtype=NS(itemsize=2),
        shared_indexer_key_dtype=NS(itemsize=2),
        c8_k_cache_dtype=NS(itemsize=1),
        indexer_scale_page_size_bytes=256,
        indexer_scale_layer_count=1,
        indexer_paired_banks=True,
    )
    layout = api.dsa_shared_block_layout(
        latent, indexer, capacity_bundles=3
    )

    assert layout.indexer_bank_count == 2
    assert layout.latent_blocks_per_bundle == 2
    assert len(api.paired_bank_indexer_block_map(3)) == 54
    assert layout.allocation_bytes_per_bundle(4, 2) > 0
    assert len(layout.block_ids_for_bundle(api.DSASharedBlockOwner.INDEXER, 1)) == 18


@pytest.mark.parametrize("slots", [0, -1, True, 1.5])
def test_invalid_paired_bank_capacity_is_rejected(slots):
    api = _api()
    with pytest.raises(ValueError):
        api.paired_bank_indexer_block_map(slots)
