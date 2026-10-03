# SPDX-License-Identifier: Apache-2.0
"""Native 910B3 collectives; preserve PyTorch synchronous/asynchronous semantics."""

import torch


def broadcast(tensor, src=0, group=None, async_op=False, group_src=None):
    """Broadcast through the selected HCCL or host process group."""
    kwargs = {"src": src, "group": group, "async_op": async_op}
    if group_src is not None:
        kwargs.pop("src")
        kwargs["group_src"] = group_src
    return torch.distributed.broadcast(tensor, **kwargs)


def all_reduce(tensor, op=torch.distributed.ReduceOp.SUM, group=None, async_op=False):
    """Reduce in place and return PyTorch's native work handle when asynchronous."""
    return torch.distributed.all_reduce(tensor, op, group, async_op)
