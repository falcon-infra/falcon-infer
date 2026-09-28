# SPDX-License-Identifier: Apache-2.0
"""Formal native LMCache lifecycle contracts without importing either runtime."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py"


def connector():
    tree = ast.parse(SOURCE.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "LMCacheConnectorV1"
    )
    namespace = {
        "KVConnectorBase_V1": type("Base", (), {}),
        "SupportsHMA": type("HMA", (), {}),
    }
    prefix = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[prefix, cls], type_ignores=[])),
            str(SOURCE),
            "exec",
        ),
        namespace,
    )
    return object.__new__(namespace[cls.name])


@pytest.mark.parametrize(
    "layerwise,role,dsa,sparse,expected",
    [
        (True, "kv_both", True, True, True),
        (True, "kv_consumer", True, True, True),
        (True, "kv_producer", True, True, False),
        (False, "kv_both", True, True, False),
        (True, "kv_both", False, True, False),
        (True, "kv_both", True, False, False),
    ],
)
def test_staged_sfa_capability_is_explicit(layerwise, role, dsa, sparse, expected):
    wrapper = connector()
    wrapper._lmcache_engine = NS(
        use_layerwise=layerwise,
        kv_role=role,
        config=NS(dsa_two_groups=dsa, enable_sparse_attention=sparse),
    )
    assert wrapper.supports_dsa_index_lmcache
    assert wrapper.uses_layerwise_model_callbacks == layerwise
    assert wrapper.supports_staged_sfa_sparse_load == expected


def test_preemption_failure_still_clears_metadata():
    wrapper = connector()
    calls = []

    def fail(ids):
        calls.append(ids)
        raise RuntimeError("fixture")

    wrapper._lmcache_engine = NS(handle_preemptions=fail)
    wrapper.bind_connector_metadata = lambda value: calls.append(value)
    wrapper.clear_connector_metadata = lambda: calls.append("clear")
    with pytest.raises(RuntimeError, match="fixture"):
        wrapper.handle_preemptions_with_metadata({"r"}, "snapshot")
    assert calls == ["snapshot", {"r"}, "clear"]


def test_seal_handoff_and_restart_are_forwarded_without_patching():
    wrapper = connector()
    calls = []
    wrapper._lmcache_engine = NS(
        capture_live_source_event_handoff=lambda ctx: (calls.append(ctx), True)[1],
        seal_sparse_destination_layout=lambda: calls.append("seal"),
        lmcache_engine=NS(
            remote_fill_requires_paired_restart=lambda: True,
            get_remote_fill_placement_info=lambda: {"tp": 8},
            get_remote_fill_metrics=lambda: {"pending": 1},
        ),
    )
    assert wrapper.capture_live_source_event_handoff("producer_event")
    wrapper.seal_sparse_destination_layout()
    assert wrapper.remote_fill_requires_paired_restart()
    assert wrapper.get_remote_fill_placement_info() == {"tp": 8}
    assert wrapper.get_remote_fill_metrics() == {"pending": 1}
    assert calls == ["producer_event", "seal"]


def test_optional_lmcache_is_not_imported_at_module_scope():
    tree = ast.parse(SOURCE.read_text())
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("lmcache")
        elif isinstance(node, ast.Import):
            assert not any(a.name.startswith("lmcache") for a in node.names)
    factory = (
        ROOT / "vllm/distributed/kv_transfer/kv_connector/factory.py"
    ).read_text()
    assert (
        '"LMCacheAscendConnector",\n    "vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector"'
        in factory
    )
    assert "lmcache_ascend_connector" not in factory
