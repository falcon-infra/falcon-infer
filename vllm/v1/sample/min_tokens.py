# SPDX-License-Identifier: Apache-2.0
"""Spec-decode stop-token masking without pageable H2D copies."""

from dataclasses import dataclass

import numpy as np
import torch

from vllm.v1.sample.ascend.rejection_diagnostics import record_stage


@dataclass
class _MaskIndices:
    layout: tuple
    device: torch.device
    stream_id: int | None
    host: torch.Tensor
    indices: torch.Tensor


def _mask_layout(min_toks: dict, num_draft_tokens: list[int]) -> tuple:
    """Key the cache by the actual mask, not the growing output history."""
    offsets = [0]
    for count in num_draft_tokens:
        offsets.append(offsets[-1] + count)
    layout = []
    for req_idx, (minimum, output_ids, stop_ids) in min_toks.items():
        if not stop_ids:
            continue
        count = min(max(minimum - len(output_ids), 0), num_draft_tokens[req_idx])
        if count:
            layout.append((offsets[req_idx], count, tuple(sorted(stop_ids))))
    return tuple(layout)


def _allocate_indices(layout: tuple, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    count = sum(num_rows * len(stop_ids) for _, num_rows, stop_ids in layout)
    host = torch.empty((2, count), dtype=torch.int64, device="cpu", pin_memory=device.type != "cpu")
    indices = torch.empty((2, count), dtype=torch.int64, device=device)
    return host, indices


def _fill_indices(host: torch.Tensor, layout: tuple) -> None:
    array = host.numpy()
    position = 0
    for offset, num_rows, stop_ids in layout:
        count = num_rows * len(stop_ids)
        array[0, position : position + count] = np.repeat(
            np.arange(offset, offset + num_rows, dtype=np.int64), len(stop_ids)
        )
        array[1, position : position + count] = np.tile(stop_ids, num_rows)
        position += count


def apply_with_spec_decode(self, logits: torch.Tensor, num_draft_tokens: list[int]) -> torch.Tensor:
    """Apply unchanged min-token/EOS semantics with cached index buffers."""
    if not self.min_toks:
        self._ascend_spec_mask_indices = None
        return logits

    layout = record_stage("min_tokens.layout", _mask_layout, self.min_toks, num_draft_tokens)
    if not layout:
        self._ascend_spec_mask_indices = None
        return logits

    device = logits.device
    stream_id = None if device.type == "cpu" else torch.npu.current_stream(device).npu_stream
    cached = getattr(self, "_ascend_spec_mask_indices", None)
    if cached is None or (cached.layout, cached.device, cached.stream_id) != (layout, device, stream_id):
        host, indices = record_stage("min_tokens.allocate", _allocate_indices, layout, device)
        record_stage("min_tokens.fill", _fill_indices, host, layout)
        record_stage("min_tokens.h2d", indices.copy_, host, non_blocking=True)
        cached = _MaskIndices(layout, device, stream_id, host, indices)
        self._ascend_spec_mask_indices = cached

    record_stage("min_tokens.mask", logits.index_put_, (cached.indices[0], cached.indices[1]), self.neg_inf_tensor)
    return logits
