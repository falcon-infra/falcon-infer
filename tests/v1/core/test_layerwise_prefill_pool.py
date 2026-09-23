# SPDX-License-Identifier: Apache-2.0
"""Dependency-light contracts for source 91's native child allocator."""

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def api():
    spec = importlib.util.spec_from_file_location(
        "layerwise_prefill_dsa_shared_pool",
        ROOT / "vllm/v1/core/dsa_shared_pool.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def layout(api, capacity=4):
    return api.DSASharedBlockLayout(
        latent_page_size_bytes=147456,
        indexer_page_size_bytes=32768,
        capacity_bundles=capacity,
    )


def test_child_pool_reserves_parents_and_releases_them_atomically(api):
    parent = api.DSASharedBundleAllocator(layout(api))
    child = api.PrefillLayerBundlePool(parent, num_physical_slots=3)
    owner = api.DSASharedBlockOwner.LATENT

    banks = child.allocate_banks(owner, 1)
    assert len(banks) == 2
    assert all(len(bank) == 1 for bank in banks)
    assert child.reserved_parent_count == 1
    assert parent.owner_bundle_counts()[api.DSASharedBlockOwner.PREFILL_RESERVED] == 1
    assert child.free_bundle_count == 10

    child.free(owner, (*banks[0], *banks[1]))
    assert child.reserved_parent_count == 0
    assert parent.free_bundle_count == 4
    assert child.free_bundle_count == 12


def test_child_mapping_preserves_parent_and_layer_slot_identity(api):
    parent = api.DSASharedBundleAllocator(layout(api))
    child = api.PrefillLayerBundlePool(parent, num_physical_slots=3)
    assert child.child_bundle_id(0, 2) == 2
    assert child.child_bundle_id(1, 2) == 6
    assert child.parent_bundle_id(6) == 2
    assert child.physical_slot(6) == 1
    with pytest.raises(ValueError):
        child.parent_bundle_id(0)


def test_prefill_layer_refs_use_independent_group_ordinals(api):
    latent = [f"model.layers.{i}.self_attn.attn" for i in range(4)]
    indexer = [
        name.rsplit(".", 1)[0] + ".indexer.k_cache" for name in latent[:2]
    ]
    refs = api.build_prefill_layer_refs(
        latent,
        indexer,
        execution_order=latent,
    )
    assert refs[latent[3]].group_ordinal == 3
    assert refs[indexer[1]].group_ordinal == 1
    assert refs[latent[3]].bank == 1
    assert refs[indexer[1]].bank == 1
    assert refs[indexer[1]].execution_index == 1


def test_native_owners_propagate_explicit_bank_contracts():
    manager = (ROOT / "vllm/v1/core/kv_cache_manager.py").read_text()
    scheduler = (ROOT / "vllm/v1/core/sched/scheduler.py").read_text()
    runner = (ROOT / "vllm/v1/worker/npu_runner_state.py").read_text()
    assert "get_block_ids_by_bank" in manager
    assert "new_block_ids_by_bank" in scheduler
    assert "_update_request_kv_block_state" in runner
    assert "PREFILL_CHILD" in runner
