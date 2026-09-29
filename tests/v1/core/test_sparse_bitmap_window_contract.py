# SPDX-License-Identifier: Apache-2.0
"""Host contract for bounded sparse bitmap preparation."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_sparse_bitmap_kernel_uses_bounded_windows_and_native_guard():
    kernel = (ROOT / "csrc/kernels/prepare_sparse_indices.cpp").read_text(
        encoding="utf-8"
    )
    binding = (ROOT / "csrc/torch_binding.cpp").read_text(encoding="utf-8")
    limits = (ROOT / "csrc/kernels/prepare_sparse_indices_limits.h").read_text(
        encoding="utf-8"
    )
    assert "DSA_BITMAP_WINDOW_WORDS" in kernel
    assert "ProcessWindow<true>" in kernel
    assert "wordBase" in kernel
    assert "DSA_BITMAP_WINDOW_TOKENS" in binding
    assert "windowed sparse preparation scratch exceeds token-window capacity" in binding
    assert "16384" in limits
