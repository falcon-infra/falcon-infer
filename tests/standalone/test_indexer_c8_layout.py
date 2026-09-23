# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for native shared key and C8 scale views."""

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def reshape():
    path = ROOT / "vllm/v1/worker/dsa_shared_pool.py"
    spec = importlib.util.spec_from_file_location("native_dsa_shared_pool", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.reshape_dsa_shared_pool_raw


@pytest.mark.parametrize("c8", [False, True])
def test_shared_views_preserve_key_layout_and_scale_sidecar(reshape, c8):
    slots, latent_page = 4, 147456
    bundle = latent_page if c8 else latent_page * 2
    raw = torch.zeros(slots * bundle, dtype=torch.int8)
    scale = torch.zeros(slots * 9 * 128 * 2, dtype=torch.int8) if c8 else None
    kwargs = {"indexer_dtype": torch.int8} if c8 else {}
    latent = reshape(
        raw,
        torch.bfloat16,
        128,
        1,
        512,
        64,
        128,
        is_indexer=False,
        **kwargs,
    )
    indexer = reshape(
        raw,
        torch.bfloat16,
        128,
        1,
        512,
        64,
        128,
        is_indexer=True,
        indexer_scale=scale,
        **kwargs,
    )
    assert indexer[0].shape == (slots * 9, 128, 1, 128)
    assert indexer[0].dtype == (torch.int8 if c8 else torch.bfloat16)
    assert sum(t.numel() * t.element_size() for t in latent) == raw.numel()
    assert indexer[0].data_ptr() == raw.data_ptr()
    if c8:
        assert len(indexer) == 2
        assert indexer[1].dtype == torch.float16
        assert indexer[1].shape == (slots * 9, 128, 1, 1)
        assert indexer[1].data_ptr() == scale.data_ptr()
    else:
        assert len(indexer) == 1


@pytest.mark.parametrize("scale_bytes", [None, 0, 256])
def test_c8_shared_view_rejects_missing_or_short_sidecar(reshape, scale_bytes):
    raw = torch.zeros(2 * 147456, dtype=torch.int8)
    scale = None if scale_bytes is None else torch.zeros(scale_bytes, dtype=torch.int8)
    with pytest.raises(ValueError, match="sidecar"):
        reshape(
            raw,
            torch.bfloat16,
            128,
            1,
            512,
            64,
            128,
            is_indexer=True,
            indexer_dtype=torch.int8,
            indexer_scale=scale,
        )
