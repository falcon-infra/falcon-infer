# SPDX-License-Identifier: Apache-2.0
"""CPU ownership contracts for asynchronous SFA source retirement."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def lifetime(monkeypatch):
    class Owner:
        def __init__(self, valid=True):
            self.valid = valid
            self.ref_count = 1

        def is_valid(self):
            return self.valid

        def ref_count_up(self):
            self.ref_count += 1

        def ref_count_down(self):
            self.ref_count -= 1

    memory = ModuleType("lmcache.v1.memory_management")
    memory.TensorMemoryObj = Owner
    monkeypatch.setitem(sys.modules, memory.__name__, memory)
    from vllm.compilation.ascend.sfa_source_lifetime import SFASourceLease

    def source(owner):
        return SimpleNamespace(
            layers=(
                SimpleNamespace(
                    memory_objs=(owner,),
                    chunk_ptrs_npu=SimpleNamespace(numel=lambda: 1),
                ),
            )
        )

    return Owner, source, SFASourceLease


def test_lease_retains_and_releases_actual_allocator_owner(lifetime):
    owner_type, source, lease_type = lifetime
    owner = owner_type()
    lease = lease_type((source(owner),))
    assert owner.ref_count == 2
    lease.close()
    lease.close()
    assert owner.ref_count == 1


def test_lease_rejects_tensor_only_or_proxy_sources(lifetime):
    owner_type, source, lease_type = lifetime
    proxy = SimpleNamespace(ref_count_up=Mock(), ref_count_down=Mock(), is_valid=lambda: True)
    with pytest.raises(RuntimeError, match="ref-counted"):
        lease_type((SimpleNamespace(layers=(SimpleNamespace(memory_objs=(proxy,), chunk_ptrs_npu=SimpleNamespace(numel=lambda: 1)),)),))
    with pytest.raises(RuntimeError, match="allocator owner"):
        lease_type((SimpleNamespace(layers=(SimpleNamespace(memory_objs=(), chunk_ptrs_npu=SimpleNamespace(numel=lambda: 1)),)),))


def test_lease_partial_retain_failure_rolls_back(lifetime):
    owner_type, source, lease_type = lifetime
    first, second = owner_type(), owner_type()
    second.ref_count_up = Mock(side_effect=RuntimeError("retain failed"))
    value = SimpleNamespace(
        layers=(
            SimpleNamespace(memory_objs=(first, second), chunk_ptrs_npu=SimpleNamespace(numel=lambda: 2)),
        )
    )
    with pytest.raises(RuntimeError, match="retain failed"):
        lease_type((value,))
    assert first.ref_count == 1 and second.ref_count == 1


def test_invalid_owner_is_rejected_before_reference_increment(lifetime):
    owner_type, source, lease_type = lifetime
    owner = owner_type(valid=False)
    with pytest.raises(RuntimeError, match="invalid"):
        lease_type((source(owner),))
    assert owner.ref_count == 1


def test_lease_reuses_prevalidated_graph_owner_set(lifetime):
    owner_type, _, lease_type = lifetime
    owner = owner_type()
    source = SimpleNamespace(graph_owners=(owner,))
    lease = lease_type((source,))
    assert owner.ref_count == 2
    lease.close()
    assert owner.ref_count == 1
