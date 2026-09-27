# SPDX-License-Identifier: Apache-2.0
"""Explicit NPU collective adaptation; CPU and non-310P use torch directly."""

import torch


def _uses_310p():
    from vllm.platforms import current_platform

    if not current_platform.is_npu():
        return False
    from vllm.utils.ascend import is_310p

    return is_310p()


class NullHandle:
    def __init__(self):
        pass

    def wait(self):
        pass


def broadcast(tensor, src=0, group=None, async_op=False, group_src=None):

    if not _uses_310p():
        kwargs = {"src": src, "group": group, "async_op": async_op}
        if group_src is not None:
            kwargs.pop("src")
            kwargs["group_src"] = group_src
        return torch.distributed.broadcast(tensor, **kwargs)
    root = group_src if group_src is not None else src

    if tensor.device == torch.device("cpu"):
        return torch.distributed.broadcast(
            tensor, src=root, group=group, async_op=async_op
        )
    rank = torch.distributed.get_rank(group)
    world_size = torch.distributed.get_world_size(group)
    tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
    tensor_list[rank] = tensor
    torch.distributed.all_gather(tensor_list, tensor, group=group)
    tensor[...] = tensor_list[src]
    if async_op:
        return NullHandle()
    else:
        return None


def all_reduce(
    tensor,
    op=torch.distributed.ReduceOp.SUM,
    group=None,
    async_op=False,
):

    if not _uses_310p():
        return torch.distributed.all_reduce(tensor, op, group, async_op)
    if tensor.dtype != torch.int64:
        return torch.distributed.all_reduce(tensor, op, group, async_op)
    rank = torch.distributed.get_rank(group)
    world_size = torch.distributed.get_world_size(group)
    tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
    tensor_list[rank] = tensor
    torch.distributed.all_gather(tensor_list, tensor, group=group)
    if op == torch.distributed.ReduceOp.SUM:
        return torch.stack(tensor_list).sum(0)
    elif op == torch.distributed.ReduceOp.MAX:
        return torch.tensor(
            torch.stack(tensor_list).cpu().numpy().max(0),
            device=tensor.device,
        )
    else:
        raise RuntimeError(f"not implement op {op}")
