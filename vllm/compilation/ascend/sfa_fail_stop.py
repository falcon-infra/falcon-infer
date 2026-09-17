# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Error-only fail-stop for a single supervised, local SFA worker cohort."""

import os
from collections.abc import Callable
from multiprocessing import parent_process
from threading import Timer
from typing import Any, NoReturn

SFA_FATAL_EXIT_CODE = 1
SFA_FATAL_CLEANUP_TIMEOUT_SECONDS = 5.0


def preparation_error_groups(
    config: Any,
    *,
    is_moe: bool,
    tp: Any,
    dp: Any,
    get_ep: Callable[[], Any],
) -> tuple[tuple[str, Any], ...] | None:
    """Choose one startup preparation-agreement group for supported EP jobs."""
    if not (
        is_moe
        and getattr(config, "enable_expert_parallel", False)
        and getattr(config, "distributed_executor_backend", None) == "mp"
        and getattr(config, "data_parallel_size", 0) > 1
        and getattr(config, "tensor_parallel_size", 0) > 1
        and all(
            getattr(config, field, 0) == 1
            for field in (
                "pipeline_parallel_size",
                "prefill_context_parallel_size",
                "decode_context_parallel_size",
            )
        )
        and not getattr(config, "enable_elastic_ep", True)
        and not getattr(config, "enable_dbo", True)
    ):
        return None
    ep = get_ep()
    width = config.tensor_parallel_size
    ranks = list(ep.ranks)
    index = ep.rank_in_group
    if not (
        tp.world_size == width
        and dp.world_size == config.data_parallel_size
        and ep.world_size == len(ranks) == tp.world_size * dp.world_size
        and len(set(ranks)) == len(ranks)
        and 0 <= index < len(ranks)
        and ranks[index] == tp.rank == dp.rank
        and list(tp.ranks)
        == list(ranks[index // width * width : (index // width + 1) * width])
        and list(dp.ranks) == list(ranks[index % width :: width])
    ):
        raise ValueError(
            "SFA preparation agreement EP membership does not match TP x DP"
        )
    return (("sfa_full_graph::prepare_agreement_ep", ep.cpu_group),)


def uses_local_sfa_fail_stop(parallel_config: Any) -> bool:
    """Only opt into fail-stop when one local MP supervisor owns all peers."""
    return (
        getattr(parallel_config, "distributed_executor_backend", None) == "mp"
        and getattr(parallel_config, "nnodes", None) == 1
        and getattr(parallel_config, "data_parallel_size", None) == 1
        and getattr(parallel_config, "pipeline_parallel_size", None) == 1
        and parent_process() is not None
    )


def exit_failed_sfa_worker(error: Exception) -> NoReturn:
    """Exit this failed worker, with a bounded cleanup fallback."""
    try:
        fallback = Timer(SFA_FATAL_CLEANUP_TIMEOUT_SECONDS, os._exit, args=(SFA_FATAL_EXIT_CODE,))
        fallback.daemon = True
        fallback.start()
    except Exception:
        os._exit(SFA_FATAL_EXIT_CODE)
    raise SystemExit(SFA_FATAL_EXIT_CODE) from error
