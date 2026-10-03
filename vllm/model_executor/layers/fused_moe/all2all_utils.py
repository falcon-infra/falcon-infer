# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

from vllm.distributed import (
    get_ep_group,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEPrepareAndFinalize,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    make_moe_prepare_and_finalize_naive_dp_ep,
    make_moe_prepare_and_finalize_no_dp_ep,
)

logger = init_logger(__name__)

pass  # Unsupported P4 branch removed.


def maybe_roundup_layer_hidden_size(
    hidden_size: int,
    act_dtype: torch.dtype,
    moe_parallel_config: FusedMoEParallelConfig,
) -> int:
    """
    Given layer hidden size and MoE configurations, round up hidden_size
    if necessary.

    Args:
        hidden_size: Layer hidden-size
        act_dtype: Data type of the layer activations.
        moe_parallel_config: Fused MoE parallelization strategy configuration.

    Return:
        Rounded up hidden_size if rounding up is required based on the configs
        and all2all backend.
        Original hidden size otherwise.
    """
    return hidden_size


def maybe_make_prepare_finalize(
    moe: FusedMoEConfig,
    quant_config: FusedMoEQuantConfig | None,
    routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    allow_new_interface: bool = False,
    use_monolithic: bool = False,
) -> FusedMoEPrepareAndFinalize | None:
    # NOTE(rob): we are migrating each quant_method to hold the MK
    # in all cases. The allow_new_interface=False flag allow us to fall
    # back to the old method for methods that have not yet been migrated.
    #
    # In old method:
    #   * maybe_init_modular_kernel() calls this function. If we are
    #     using no Dp/Ep or naive all2all, we return None this function
    #     returns None and no ModularKernelMethod is created. If non-naive
    #     all2all is used, this returns a PrepareAndFinalize object and
    #     a ModularKernelMethod is created.
    # In new method:
    #   * maybe_make_prepare_finalize() is called from the oracle. We
    #     always return a PrepareAndFinalize object and the quant method
    #     holds the ModularKernel.
    if not moe.moe_parallel_config.use_all2all_kernels:
        if not allow_new_interface:
            return None

        # For DP/TP case, fall back to naive P/F.
        if moe.moe_parallel_config.dp_size > 1:
            logger.info_once(
                "Detected DP deployment with no --enable-expert-parallel. "
                "Falling back to AllGather+ReduceScatter dispatch/combine."
            )
            return make_moe_prepare_and_finalize_naive_dp_ep(
                is_sequence_parallel=moe.moe_parallel_config.is_sequence_parallel,
                num_dispatchers=(
                    get_ep_group().device_communicator.all2all_manager.world_size
                ),
                use_monolithic=use_monolithic,
            )
        else:
            return make_moe_prepare_and_finalize_no_dp_ep(use_monolithic)

    # HCCL dispatch/combine is implemented by the Ascend layer.
    # Do not select a vendor GPU prepare/finalize object in the common base.
    return None
