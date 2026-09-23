# SPDX-License-Identifier: Apache-2.0
"""Host-only contracts for native indexer-C8 startup policy."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "vllm/config/ascend.py"


@pytest.fixture
def validate(monkeypatch):
    tree = ast.parse(CONFIG.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AscendConfig"
    )
    mask_method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "indexer_c8_layer_mask"
    )
    namespace = {}
    exec(
        compile(
            ast.Module(body=[mask_method], type_ignores=[]),
            str(CONFIG),
            "exec",
        ),
        namespace,
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "vllm.model_executor.models.utils",
        SimpleNamespace(
            extract_layer_index=lambda name: int(
                name.split("layers.", 1)[1].split(".", 1)[0]
            )
        ),
    )

    def run(description, enabled=True, names=None, resolve=False):
        obj = SimpleNamespace(
            enable_sparse_li_c8=enabled,
            vllm_config=SimpleNamespace(
                quant_config=SimpleNamespace(quant_description=description)
            ),
        )
        obj.indexer_c8_layer_mask = lambda values: namespace[mask_method.name](
            obj, values
        )
        names = names or [
            "model.layers.0.self_attn.indexer.k_cache",
            "model.layers.3.self_attn.indexer.k_cache",
        ]
        return namespace[mask_method.name](obj, names)

    return run


@pytest.mark.parametrize("description", [None, {}, {"some.weight": "INT8_DYNAMIC"}])
def test_absent_layer_filter_keeps_global_c8_selection(validate, description):
    validate(description)


def test_partial_physical_layer_policy_is_preserved(validate):
    assert validate(
        {"model.layers.0.self_attn.indexer.quant_type": "INT8_DYNAMIC"}
    ) == (True, False)


def test_all_bf16_layer_policy_is_preserved(validate):
    assert validate({"model.layers.0.self_attn.indexer.quant_type": "FLOAT"}) == (
        False,
        False,
    )


def test_layer_aliases_match_by_layer_index(validate):
    validate(
        {
            "other.layers.0.self_attn.indexer.quant_type": "INT8_DYNAMIC",
            "other.layers.3.self_attn.indexer.wq_b_weight": "W8A8_MXFP8",
        }
    )


def test_mixed_policy_preserves_selected_physical_owners(validate):
    selected = [6, 10, 14, 18]
    owners = [0, 1, *selected, 22]
    description = {
        f"model.layers.{index}.self_attn.indexer.quant_type": "INT8_DYNAMIC"
        for index in selected
    }
    names = [f"model.layers.{index}.self_attn.indexer.k_cache" for index in owners]
    assert validate(description, names=names, resolve=True) == (
        False,
        False,
        True,
        True,
        True,
        True,
        False,
    )


def test_config_exposes_indexer_alias_and_rejects_sfa_c8():
    source = CONFIG.read_text(encoding="utf-8")
    assert "enable_sparse_li_c8" in source
    assert "enable_sparse_sfa_c8" in source
    assert "Indexer INT8/FP16-scale caching" in source
