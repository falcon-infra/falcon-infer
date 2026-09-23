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
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "validate_indexer_c8_layers"
    )
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(CONFIG), "exec"), namespace)
    monkeypatch.setitem(
        __import__("sys").modules,
        "vllm.model_executor.models.utils",
        SimpleNamespace(
            extract_layer_index=lambda name: int(
                name.split("layers.", 1)[1].split(".", 1)[0]
            )
        ),
    )

    def run(description, enabled=True):
        obj = SimpleNamespace(
            enable_sparse_li_c8=enabled,
            vllm_config=SimpleNamespace(
                quant_config=SimpleNamespace(quant_description=description)
            ),
        )
        namespace[method.name](
            obj,
            [
                "model.layers.0.self_attn.indexer.k_cache",
                "model.layers.3.self_attn.indexer.k_cache",
            ],
        )

    return run


@pytest.mark.parametrize("description", [None, {}, {"some.weight": "INT8_DYNAMIC"}])
def test_absent_layer_filter_keeps_global_c8_selection(validate, description):
    validate(description)


def test_partial_physical_layer_policy_is_rejected(validate):
    with pytest.raises(ValueError, match="uniform quantization"):
        validate({"model.layers.0.self_attn.indexer.quant_type": "INT8_DYNAMIC"})


def test_layer_aliases_match_by_layer_index(validate):
    validate(
        {
            "other.layers.0.self_attn.indexer.quant_type": "INT8_DYNAMIC",
            "other.layers.3.self_attn.indexer.wq_b_weight": "W8A8_MXFP8",
        }
    )


def test_config_exposes_indexer_alias_and_rejects_sfa_c8():
    source = CONFIG.read_text(encoding="utf-8")
    assert "enable_sparse_li_c8" in source
    assert "enable_sparse_sfa_c8" in source
    assert "Indexer INT8/FP16-scale caching" in source
