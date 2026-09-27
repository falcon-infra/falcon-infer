# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable, Iterable, Sequence
from functools import partial

import numpy as np
import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import (
    async_tensor_h2d,
)


def async_copy_to_gpu(
    x: torch.Tensor | np.ndarray,
    out: torch.Tensor | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    assert x.is_cpu

    if out is None:
        assert device is not None
        out = torch.empty_like(x, device=device)

    # Copy directly to GPU — explicit pin_memory() causes sporadic stalls
    # under high concurrency due to CUDA driver contention. The driver
    # handles the transfer efficiently without manual pinning.
    return out.copy_(x, non_blocking=True)


class UvaBuffer:
    """Ascend NPU doesn't support UVA tensors directly. This is a wrapper class
    that provides CPU and NPU views of a UVA tensor."""

    def __init__(self, size: int | Sequence[int], dtype: torch.dtype):
        self._cpu: torch.Tensor = torch.zeros(
            size, dtype=dtype, device="cpu", pin_memory=True
        )
        self._np = self._cpu.numpy()
        self._uva: torch.Tensor = torch.zeros_like(self._cpu, device="npu")
        self._modified_indices: set[int] = set()

    def _mark_cpu_modified(self, key: int):
        self._modified_indices.add(key)

    @property
    def cpu(self):
        return MonitoredTorchTensor(self._cpu, self._mark_cpu_modified)

    @property
    def np(self):
        return MonitoredNumPyArray(self._np, self._mark_cpu_modified)

    @property
    def uva(self):
        """Get the device data of the buffer."""
        if self._modified_indices:
            # Sort for better memory access locality
            dirty_rows = sorted(self._modified_indices)
            # can't use copy_ method, because copy_ for index tensor
            #  will malloc new memory.
            self._uva[dirty_rows] = self._cpu[dirty_rows].to(
                device="npu", non_blocking=True
            )
            self._modified_indices.clear()
        return self._uva


class UvaBufferPool:
    def __init__(
        self,
        size: int | Sequence[int],
        dtype: torch.dtype,
        max_concurrency: int = 2,
    ):
        self.size = size
        self.dtype = dtype
        self.max_concurrency = max_concurrency

        # UVA buffers for concurrency
        self._uva_bufs = [UvaBuffer(size, dtype) for _ in range(max_concurrency)]
        # Current buffer index
        self._curr = 0

    def copy_to_uva(self, x: torch.Tensor | np.ndarray | list) -> torch.Tensor:
        # Round robin to the next buffer.
        self._curr = (self._curr + 1) % self.max_concurrency
        buf = self._uva_bufs[self._curr]
        # CPU-to-CPU copy
        dst = buf.cpu if isinstance(x, torch.Tensor) else buf.np
        n = len(x)
        dst[:n] = x
        return buf.uva[:n]

    def copy_to_gpu(
        self,
        x: torch.Tensor | np.ndarray,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        uva = self.copy_to_uva(x)
        # CPU-to-GPU copy
        return uva.clone() if out is None else out.copy_(uva, non_blocking=True)


class UvaBackedTensor:
    def __init__(
        self, size: int | Sequence[int], dtype: torch.dtype, max_concurrency: int = 2
    ):
        self.dtype = dtype

        # Source of truth
        self.cpu = torch.zeros(size, dtype=dtype, device="cpu", pin_memory=False)
        self.np = self.cpu.numpy()

        # Buffers for concurrency
        self.pool = UvaBufferPool(size, dtype, max_concurrency)
        self.gpu = self.pool.copy_to_uva(self.np)

    def copy_to_uva(self, n: int | None = None) -> torch.Tensor:
        # CPU-to-CPU copy
        self.gpu = self.pool.copy_to_uva(self.np[:n] if n is not None else self.np)
        return self.gpu


class StagedWriteTensor:
    def __init__(
        self,
        size: int | Sequence[int],
        dtype: torch.dtype,
        device: torch.device,
        max_concurrency: int = 2,
        uva_instead_of_gpu: bool = False,
    ):
        supported_dtypes = [torch.int32, torch.int64, torch.float32]
        if dtype not in supported_dtypes:
            raise ValueError(
                f"Unsupported dtype {dtype}: should be one of {supported_dtypes}"
            )
        self.num_rows = size if isinstance(size, int) else size[0]
        self.dtype = dtype
        self.device = device
        self.max_concurrency = max_concurrency

        if not uva_instead_of_gpu:
            # Create a GPU tensor (default)
            self.gpu = torch.zeros(size, dtype=dtype, device=device)
        else:
            # For a large but not-frequently-accessed tensor, we can use UVA instead of
            # GPU to save GPU memory
            self._uva_buf = UvaBuffer(size, dtype)
            self.gpu = self._uva_buf.uva

        self._staged_write_indices: list[int] = []
        self._staged_write_starts: list[int] = []
        self._staged_write_contents: list[int | float] = []
        self._staged_write_cu_lens: list[int] = []

        new_buffer = partial(UvaBufferPool, max_concurrency=max_concurrency)

        self.write_indices = new_buffer(self.num_rows, dtype=torch.int32)
        self.write_starts = new_buffer(self.num_rows, dtype=torch.int32)
        self.write_cu_lens = new_buffer(self.num_rows, dtype=torch.int32)

    def stage_write(
        self, index: int, start: int, x: Iterable[int] | Iterable[float]
    ) -> None:
        assert index >= 0
        assert start >= 0
        if not x:
            return
        self._staged_write_indices.append(index)
        self._staged_write_starts.append(start)
        self._staged_write_contents.extend(x)
        self._staged_write_cu_lens.append(len(self._staged_write_contents))

    def stage_write_elem(self, index: int, x: int) -> None:
        assert index >= 0
        self._staged_write_indices.append(index)
        self._staged_write_starts.append(0)
        self._staged_write_contents.append(x)
        self._staged_write_cu_lens.append(len(self._staged_write_contents))

    def apply_write(self) -> None:
        n = len(self._staged_write_indices)
        if n == 0:
            return

        indices_uva = self.write_indices.copy_to_uva(self._staged_write_indices)
        starts_uva = self.write_starts.copy_to_uva(self._staged_write_starts)
        cu_lens_uva = self.write_cu_lens.copy_to_uva(self._staged_write_cu_lens)

        # Special handling for write_contents
        write_contents = async_tensor_h2d(
            self._staged_write_contents, self.dtype, self.device, pin_memory=True
        )

        # Write diffs to the GPU buffer
        _apply_write_kernel[(n,)](
            self.gpu,
            self.gpu.stride(0),
            indices_uva,
            starts_uva,
            write_contents,
            cu_lens_uva,
            BLOCK_SIZE=1024,
        )
        # Clear the staged writes
        self.clear_staged_writes()

    def clear_staged_writes(self) -> None:
        self._staged_write_indices.clear()
        self._staged_write_starts.clear()
        self._staged_write_contents.clear()
        self._staged_write_cu_lens.clear()


@triton.jit
def _apply_write_kernel(
    output_ptr,
    output_stride,
    write_indices_ptr,
    write_starts_ptr,
    write_contents_ptr,
    write_cu_lens_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_idx = tl.load(write_indices_ptr + pid)
    start_idx = tl.load(write_starts_ptr + pid)

    cu_start = tl.load(write_cu_lens_ptr + pid - 1) if pid > 0 else 0
    cu_end = tl.load(write_cu_lens_ptr + pid)
    content_len = cu_end - cu_start

    for i in range(0, content_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < content_len
        content = tl.load(write_contents_ptr + cu_start + block, mask=mask)
        tl.store(
            output_ptr + row_idx * output_stride + start_idx + block, content, mask=mask
        )


def get_row_indices_from_key(key: int | slice | tuple, dim_size: int) -> set[int]:
    """get the set of row indices involved in the given key."""
    if isinstance(key, int):
        # parse index such as np[1]
        key = key if key >= 0 else dim_size + key
        # handle negative index
        if key < 0 or key >= dim_size:
            raise IndexError(f"row index {key} out of [0, {dim_size})")
        return {key}
    elif isinstance(key, slice):
        # parse slice such as np[1:3]
        start, stop, step = key.indices(dim_size)
        return set(range(start, stop, step))
    elif isinstance(key, tuple):
        # parse row slice such as np[1,:100]
        if len(key) == 0:
            return set(range(dim_size))
        return get_row_indices_from_key(key[0], dim_size)
    else:
        # for other types such as list/ndarray, we return all rows.
        return set(range(dim_size))


class MonitoredNumPyArray:
    """A wrapper around a NumPy array that monitors modifications."""

    def __init__(self, array: np.ndarray, callback: Callable):
        self._array = array
        self._callback = callback

    def __setitem__(self, key, value):
        self._array[key] = value
        dim_size = self._array.shape[0]
        row_indices = get_row_indices_from_key(key, dim_size)
        for row in row_indices:
            self._callback(row)

    def __getitem__(self, key):
        return self._array[key]

    def __getattr__(self, name):
        return getattr(self._array, name)


class MonitoredTorchTensor:
    """A wrapper around a torch tensor that monitors modifications."""

    def __init__(self, tensor: torch.Tensor, callback: Callable):
        self._tensor = tensor
        self._callback = callback

    def __setitem__(self, key, value):
        self._tensor[key] = value
        dim_size = self._tensor.size(0)
        row_indices = get_row_indices_from_key(key, dim_size)
        for row in row_indices:
            self._callback(row)

    def __getitem__(self, key):
        return self._tensor[key]

    def __getattr__(self, name):
        return getattr(self._tensor, name)
