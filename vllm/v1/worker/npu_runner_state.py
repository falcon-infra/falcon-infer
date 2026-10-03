# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import functools
import gc
import itertools
import threading
import time
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple, TypeAlias, cast

import numpy as np
import torch
import torch.distributed
from tqdm import tqdm

import vllm.envs as envs
from vllm.compilation.ascend.acl_graph import ACLGraphWrapper
from vllm.compilation.counter import compilation_counter
from vllm.compilation.graph_types import CUDAGraphStat
from vllm.compilation.monitor import set_cudagraph_capturing_enabled
from vllm.config import (
    CompilationMode,
    CUDAGraphMode,
    VllmConfig,
    get_layers_from_vllm_config,
    set_current_vllm_config,
    update_config,
)
from vllm.config.cache import CacheConfig
from vllm.distributed.ascend.parallel_state import graph_capture
from vllm.distributed.eplb.eplb_state import EplbState
from vllm.distributed.parallel_state import (
    get_dcp_group,
    get_pp_group,
    is_global_first_rank,
)
from vllm.forward_context import (
    BatchDescriptor,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention, MLAAttention
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsCapturer,
)
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.model_loader.reload import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
)
from vllm.model_executor.models.interfaces import (
    is_mixture_of_experts,
    supports_realtime,
    supports_transcription,
)
from vllm.model_executor.models.interfaces_base import (
    is_text_generation_model,
)
from vllm.model_executor.offloader import (
    create_offloader,
    set_offloader,
)
from vllm.platforms import current_platform
from vllm.sampling_params import SamplingType
from vllm.sequence import IntermediateTensors
from vllm.tasks import GenerationTask, SupportedTask
from vllm.tracing import instrument
from vllm.utils import length_from_prompt_token_ids_or_embeds
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.utils.torch_utils import (
    kv_cache_dtype_str_to_dtype,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.utils import (
    reorder_batch_to_split_decodes_and_prefills,
)
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    ChunkedLocalAttentionSpec,
    CrossAttentionSpec,
    EncoderOnlyAttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    SlidingWindowSpec,
)
from vllm.v1.outputs import (
    DraftTokenIds,
    ECConnectorOutput,
    KVConnectorOutput,
    LogprobsTensors,
)
from vllm.v1.sample.logits_processor import build_logitsprocs
from vllm.v1.sample.logits_processor.interface import LogitsProcessor
from vllm.v1.sample.rejection_sampler import RejectionSampler
from vllm.v1.sample.sampler import Sampler
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.input_batch import CachedRequestState, InputBatch
from vllm.v1.worker.kv_connector_model_runner_mixin import KVConnectorModelRunnerMixin
from vllm.v1.worker.ubatch_utils import (
    UBatchSlices,
    check_ubatch_thresholds,
)
from vllm.v1.worker.utils import is_residual_scattered_for_sp
from vllm.v1.worker.workspace import lock_workspace

from .utils import (
    AttentionGroup,
    KVBlockZeroer,
    add_kv_sharing_layers_to_kv_cache_groups,
)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)

AttnMetadataDict: TypeAlias = dict[str, AttentionMetadata]
# list when ubatching is enabled
PerLayerAttnMetadata: TypeAlias = list[AttnMetadataDict] | AttnMetadataDict


# Wrapper for ModelRunnerOutput to support overlapped execution.


class ExecuteModelState(NamedTuple):
    """Ephemeral cached state transferred between execute_model() and
    sample_tokens(), after execute_model() returns None."""

    scheduler_output: "SchedulerOutput"
    logits: torch.Tensor
    spec_decode_metadata: SpecDecodeMetadata | None
    spec_decode_common_attn_metadata: CommonAttentionMetadata | None
    hidden_states: torch.Tensor
    sample_hidden_states: torch.Tensor
    aux_hidden_states: list[torch.Tensor] | None
    ec_connector_output: ECConnectorOutput | None
    cudagraph_stats: CUDAGraphStat | None
    slot_mappings: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None


class NPUModelRunnerState(KVConnectorModelRunnerMixin):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.offload_config = vllm_config.offload_config
        self.compilation_config = vllm_config.compilation_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config
        self.observability_config = vllm_config.observability_config

        model_config = self.model_config
        cache_config = self.cache_config
        scheduler_config = self.scheduler_config
        parallel_config = self.parallel_config
        self.device = device
        self.pin_memory = is_pin_memory_available()
        self.dtype = self.model_config.dtype

        self.kv_cache_dtype = kv_cache_dtype_str_to_dtype(
            cache_config.cache_dtype, self.model_config
        )

        self.is_pooling_model = model_config.runner_type == "pooling"
        self.enable_prompt_embeds = model_config.enable_prompt_embeds
        self.is_multimodal_raw_input_only_model = (
            model_config.is_multimodal_raw_input_only_model
        )
        # This will be overridden in load_model()
        self.is_multimodal_pruning_enabled = False
        # Set to True after init_routed_experts_capturer() completes.
        # Prevents routed experts code from running during profiling/dummy run.
        self.routed_experts_initialized = False
        self.max_model_len = model_config.max_model_len

        # Always set to false after the first forward pass
        self.calculate_kv_scales = self.cache_config.calculate_kv_scales
        self.dcp_world_size = self.parallel_config.decode_context_parallel_size
        self.dcp_rank = 0 if self.dcp_world_size <= 1 else get_dcp_group().rank_in_group
        self.max_num_tokens = scheduler_config.max_num_batched_tokens
        self.max_num_reqs = scheduler_config.max_num_seqs

        # Broadcast PP output for external_launcher (torchrun)
        # to make sure we are synced across pp ranks
        # TODO: Support overlapping micro-batches
        # https://github.com/vllm-project/vllm/issues/18019
        self.broadcast_pp_output = (
            self.parallel_config.distributed_executor_backend == "external_launcher"
            and len(get_pp_group().ranks) > 1
        )

        # Model-related.
        self.num_query_heads = model_config.get_num_attention_heads(parallel_config)
        self.inputs_embeds_size = model_config.get_inputs_embeds_size()
        self.attention_chunk_size = model_config.attention_chunk_size
        # Only relevant for models using ALiBi (e.g, MPT)
        self.use_alibi = model_config.uses_alibi

        self.cascade_attn_enabled = not self.model_config.disable_cascade_attn
        self.is_mm_prefix_lm = self.model_config.is_mm_prefix_lm

        # Multi-modal data support
        self.mm_registry = None
        self.uses_mrope = model_config.uses_mrope
        self.uses_xdrope_dim = model_config.uses_xdrope_dim
        self.supports_mm_inputs = False

        self.max_encoder_len = 0

        # Async scheduling
        self.use_async_scheduling = self.scheduler_config.async_scheduling

        # Sampler
        self.sampler = Sampler(logprobs_mode=self.model_config.logprobs_mode)

        self.eplb_state: EplbState | None = None
        # NOTE(yongji): flag to temporarily disable EPLB during scaling up/down
        self.eep_eplb_suppressed = False
        """
        State of the expert parallelism load balancer.

        Will be lazily initialized when the model is loaded.
        """

        # Lazy initializations
        # self.model: nn.Module  # Set after load_model
        # Initialize in initialize_kv_cache
        self.kv_caches: list[torch.Tensor] = []
        # Initialize in initialize_kv_cache_tensors
        self.cross_layers_kv_cache: torch.Tensor | None = None
        self.cross_layers_attn_backend: type[AttentionBackend] | None = None
        # indexes: [kv_cache_group_id][attn_group]
        self.attn_groups: list[list[AttentionGroup]] = []
        # self.kv_cache_config: KVCacheConfig

        # mm_hash ->  encoder_output
        self.encoder_cache: dict[str, torch.Tensor] = {}

        self.use_aux_hidden_state_outputs = False
        # Set up speculative decoding.
        # NOTE(Jiayi): currently we put the entire draft model on
        # the last PP rank. This is not ideal if there are many
        # layers in the draft model.
        if self.speculative_config and get_pp_group().is_last_rank:
            self.drafter: EagleProposer | None
            if self.speculative_config.use_eagle():
                self.drafter = EagleProposer(self.vllm_config, self.device, self)
                pass  # Unsupported P4 branch removed.
            raise ValueError(
                f"Unknown speculative decoding method: {self.speculative_config.method}"
            )
            self.rejection_sampler = RejectionSampler(self.sampler)

        self.num_spec_tokens = 0
        if self.speculative_config:
            self.num_spec_tokens = self.speculative_config.num_speculative_tokens
            draft_config = self.speculative_config.draft_model_config
            if draft_config is not None and draft_config.max_model_len is not None:
                self.effective_drafter_max_model_len = draft_config.max_model_len
            else:
                self.effective_drafter_max_model_len = self.max_model_len

        # Request states.
        self.requests: dict[str, CachedRequestState] = {}
        # NOTE(rob): num_prompt_logprobs only includes reqs
        # that are currently in the prefill phase.
        self.num_prompt_logprobs: dict[str, int] = {}
        self.comm_stream = torch.npu.Stream()

        # Input Batch
        # NOTE(Chen): Ideally, we should initialize the input batch inside
        # `initialize_kv_cache` based on the kv cache config. However, as in
        # https://github.com/vllm-project/vllm/pull/18298, due to some unknown
        # reasons, we have to initialize the input batch before `load_model`,
        # quantization + weight offloading will fail otherwise. As a temporary
        # solution, we initialize the input batch here, and re-initialize it
        # in `initialize_kv_cache` if the block_sizes here is different from
        # the block_sizes in the kv cache config.
        logits_processors = model_config.logits_processors
        custom_logitsprocs: Sequence[str | type[LogitsProcessor]] = (
            tuple(logits_processors) if logits_processors is not None else ()
        )
        placeholder_block_size = (
            self.cache_config.block_size or CacheConfig.DEFAULT_BLOCK_SIZE
        )
        self._init_block_sizes = [placeholder_block_size]
        self._init_kernel_block_sizes = [placeholder_block_size]
        self.input_batch = InputBatch(
            max_num_reqs=self.max_num_reqs,
            # We need to use the encoder length for encoder-decoder
            # because of KV cache for cross-attention.
            max_model_len=max(self.max_model_len, self.max_encoder_len),
            max_num_batched_tokens=self.max_num_tokens,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=self.model_config.get_vocab_size(),
            block_sizes=[placeholder_block_size],
            kernel_block_sizes=[placeholder_block_size],
            is_spec_decode=bool(self.vllm_config.speculative_config),
            logitsprocs=build_logitsprocs(
                self.vllm_config,
                self.device,
                self.pin_memory,
                self.is_pooling_model,
                custom_logitsprocs,
            ),
            # We currently don't know whether a particular custom logits processor
            # uses output token ids so we set this conservatively.
            logitsprocs_need_output_token_ids=bool(custom_logitsprocs),
            is_pooling_model=self.is_pooling_model,
            cp_kv_cache_interleave_size=self.parallel_config.cp_kv_cache_interleave_size,
        )

        # Separate cuda stream for overlapping transfer of sampled token ids from
        # GPU to CPU when async scheduling is enabled.
        self.async_output_copy_stream: torch.npu.Stream | None = None
        # cuda event to synchronize use of reused CPU tensors between steps
        # when async scheduling is enabled.
        self.prepare_inputs_event: torch.npu.Event | None = None
        if self.use_async_scheduling:
            self.async_output_copy_stream = torch.npu.Stream()
            self.prepare_inputs_event = torch.npu.Event()

        # self.cudagraph_batch_sizes sorts in ascending order.
        if (
            self.compilation_config.cudagraph_capture_sizes
            and self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
        ):
            self.cudagraph_batch_sizes = sorted(
                self.compilation_config.cudagraph_capture_sizes
            )
        else:
            self.cudagraph_batch_sizes = []

        # Cache the device properties.
        self._init_device_properties()

        # Encoder timing registry for observability
        self.encoder_timing_registry: dict[str, EncoderTimingStats] = {}
        self._encoder_timing_lock = threading.Lock()

        # Persistent buffers for CUDA graphs.
        self.input_ids = self._make_buffer(self.max_num_tokens, dtype=torch.int32)
        self.positions = self._make_buffer(self.max_num_tokens, dtype=torch.int64)
        self.query_start_loc = self._make_buffer(
            self.max_num_reqs + 1, dtype=torch.int32
        )
        self.seq_lens = self._make_buffer(self.max_num_reqs, dtype=torch.int32)
        self.encoder_seq_lens = self._make_buffer(self.max_num_reqs, dtype=torch.int32)
        if self.dcp_world_size > 1:
            self.dcp_local_seq_lens = self._make_buffer(
                self.max_num_reqs, dtype=torch.int32
            )
        # Because inputs_embeds may be bfloat16 and we don't need a numpy
        # version of this tensor, avoid a RuntimeError by not creating a
        # numpy buffer.
        self.inputs_embeds = self._make_buffer(
            self.max_num_tokens, self.inputs_embeds_size, dtype=self.dtype, numpy=False
        )
        self.is_token_ids = self._make_buffer(self.max_num_tokens, dtype=torch.bool)
        self.discard_request_mask = self._make_buffer(
            self.max_num_reqs, dtype=torch.bool
        )
        self.num_decode_draft_tokens = self._make_buffer(
            self.max_num_reqs, dtype=torch.int32
        )
        self.num_accepted_tokens = self._make_buffer(
            self.max_num_reqs, dtype=torch.int64
        )

        # Only relevant for multimodal models
        pass  # Unsupported P4 branch removed.

        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        pass  # Unsupported P4 branch removed.

        # Only relevant for models using XD-RoPE (e.g, HunYuan-VL)
        pass  # Unsupported P4 branch removed.

        # None in the first PP rank. The rest are set after load_model.
        self.intermediate_tensors: IntermediateTensors | None = None

        # OPTIMIZATION: Cache the tensors rather than creating them every step.
        # Keep in int64 to avoid overflow with long context
        self.arange_np = np.arange(
            max(self.max_num_reqs + 1, self.max_model_len, self.max_num_tokens),
            dtype=np.int64,
        )

        # Layer pairings for cross-layer KV sharing.
        # If an Attention layer `layer_name` is in the keys of this dict, it
        # means this layer will perform attention using the keys and values
        # from the KV cache of `shared_kv_cache_layers[layer_name]`.
        self.shared_kv_cache_layers: dict[str, str] = {}
        self.kv_sharing_fast_prefill_eligible_layers: set[str] = set()

        self.kv_sharing_fast_prefill_logits_indices = None
        if self.cache_config.kv_sharing_fast_prefill:
            self.kv_sharing_fast_prefill_logits_indices = torch.zeros(
                self.max_num_tokens, dtype=torch.int32, device=self.device
            )

        self.uniform_decode_query_len = 1 + self.num_spec_tokens

        # Cudagraph dispatcher for runtime cudagraph dispatching.
        self.cudagraph_dispatcher = CudagraphDispatcher(self.vllm_config)

        self.mm_budget = None

        self.reorder_batch_threshold: int | None = None

        # Attention layers that are only in the KVCacheConfig of the runner
        # (e.g., KV sharing, encoder-only attention), but not in the
        # KVCacheConfig of the scheduler.
        self.runner_only_attn_layers: set[str] = set()

        # Cached outputs.
        self._draft_token_ids: list[list[int]] | torch.Tensor | None = None
        # N-gram GPU path: async D2H buffer/event for per-request valid draft counts.
        self._num_valid_draft_tokens: torch.Tensor | None = None
        self._num_valid_draft_tokens_cpu: torch.Tensor | None = None
        self._num_valid_draft_tokens_event: torch.npu.Event | None = None
        self._num_valid_draft_tokens_copy_stream: torch.npu.Stream | None = None
        pass  # Unsupported P4 branch removed.

        self._draft_token_req_ids: list[str] | None = None
        self.transfer_event = torch.npu.Event()
        self.sampled_token_ids_pinned_cpu = torch.empty(
            (self.max_num_reqs, 1),
            dtype=torch.int64,
            device="cpu",
            pin_memory=self.pin_memory,
        )

        # Pre-allocated tensor for copying valid sampled token counts to CPU,
        # with dedicated stream for overlapping and event for coordination.
        self.valid_sampled_token_count_event: torch.npu.Event | None = None
        self.valid_sampled_token_count_copy_stream: torch.npu.Stream | None = None
        # We also copy the drafted tokens to the CPU asynchronously,
        # in case we need them for structured outputs.
        self.draft_token_ids_event: torch.npu.Event | None = None
        self.draft_token_ids_copy_stream: torch.npu.Stream | None = None
        self.valid_sampled_token_count_cpu: torch.Tensor | None = None
        self.draft_token_ids_cpu: torch.Tensor | None = None
        self.num_accepted_tokens_event: torch.npu.Event | None = None
        if self.num_spec_tokens:
            self.draft_token_ids_event = torch.npu.Event()
            self.num_accepted_tokens_event = torch.npu.Event()
            self.draft_token_ids_copy_stream = torch.npu.Stream()
            self.draft_token_ids_cpu = torch.empty(
                (self.max_num_reqs, self.num_spec_tokens),
                dtype=torch.int64,
                device="cpu",
                pin_memory=self.pin_memory,
            )
            if self.use_async_scheduling:
                self.valid_sampled_token_count_event = torch.npu.Event()
                self.valid_sampled_token_count_copy_stream = torch.npu.Stream()
                self.valid_sampled_token_count_cpu = torch.empty(
                    self.max_num_reqs,
                    dtype=torch.int64,
                    device="cpu",
                    pin_memory=self.pin_memory,
                )

        # Model weight offloader
        # Make sure this is called before any get_offloader call
        set_offloader(create_offloader(self.offload_config))

        # Ephemeral state transferred between execute_model() and sample_tokens().
        self.execute_model_state: ExecuteModelState | None = None
        self.kv_connector_output: KVConnectorOutput | None = None
        self.layerwise_nvtx_hooks_registered = False

    def update_max_model_len(self, max_model_len: int) -> None:
        self.max_model_len = max_model_len
        if self.speculative_config:
            draft_config = self.speculative_config.draft_model_config
            if draft_config is None or draft_config.max_model_len is None:
                self.effective_drafter_max_model_len = self.max_model_len

    def reset_mm_cache(self) -> None:
        """
        Clear the multi-modal cache that was used during profiling,
        but no longer needed during inference.
        """
        if self.mm_budget:
            self.mm_budget.reset_cache()

    def reset_encoder_cache(self) -> None:
        """Clear the GPU-side encoder cache storing vision embeddings.

        This should be called when model weights are updated to ensure
        stale embeddings computed with old weights are not reused.
        """
        self.encoder_cache.clear()

    @torch.inference_mode()
    def init_fp8_kv_scales(self) -> None:
        """
        Re-initialize the KV cache and FP8 scales after waking from sleep.
        1. Zero out the KV cache tensors to remove garbage data from re-allocation.
        2. Reset Attention layer scaling factors (_k_scale, _v_scale) to 1.0.
          If these are left at 0.0 (default after wake_up), all KV cache values
          become effectively zero, causing gibberish output.
        """
        if not self.cache_config.cache_dtype.startswith("fp8"):
            return

        kv_caches = getattr(self, "kv_caches", [])
        for cache_tensor in kv_caches:
            if cache_tensor is not None:
                cache_tensor.zero_()

        k_attr_names = ("_k_scale", "k_scale")
        v_attr_names = ("_v_scale", "v_scale")

        attn_layers = self.compilation_config.static_forward_context
        for name, module in attn_layers.items():
            if isinstance(module, (Attention, MLAAttention)):
                # TODO: Generally, scale is 1.0 if user uses on-the-fly fp8
                # kvcache quant. However, to get better accuracy, compression
                # frameworks like llm-compressors allow users to tune the
                # scale. We may need to restore the specific calibrated scales
                # here in the future.
                k_scale_val, v_scale_val = 1.0, 1.0

                # Processing K Scale
                for attr in k_attr_names:
                    if hasattr(module, attr):
                        param = getattr(module, attr)
                        if isinstance(param, torch.Tensor):
                            param.fill_(k_scale_val)

                # Processing V Scale
                for attr in v_attr_names:
                    if hasattr(module, attr):
                        param = getattr(module, attr)
                        if isinstance(param, torch.Tensor):
                            param.fill_(v_scale_val)

    def _get_positions(self, num_tokens: Any):
        if isinstance(num_tokens, int):
            pass  # Unsupported P4 branch removed.
            pass  # Unsupported P4 branch removed.
            return self.positions.gpu[:num_tokens]
        else:
            pass  # Unsupported P4 branch removed.
            pass  # Unsupported P4 branch removed.
            return self.positions.gpu[num_tokens]

    def _make_buffer(
        self, *size: int | torch.SymInt, dtype: torch.dtype, numpy: bool = True
    ) -> CpuGpuBuffer:
        return CpuGpuBuffer(
            *size,
            dtype=dtype,
            device=self.device,
            pin_memory=self.pin_memory,
            with_numpy=numpy,
        )

    def _init_model_kwargs(self):
        model_kwargs = dict[str, Any]()

        return model_kwargs

    def _may_reorder_batch(self, scheduler_output: "SchedulerOutput") -> None:
        """
        Update the order of requests in the batch based on the attention
        backend's needs. For example, some attention backends (namely MLA) may
        want to separate requests based on if the attention computation will be
        compute-bound or memory-bound.

        Args:
            scheduler_output: The scheduler output.
        """
        # Attention free models have zero kv_cache_groups, however models
        # like Mamba are also attention free but use the kv_cache for
        # keeping its internal state. This is why we check the number
        # of kv_cache groups instead of solely checking
        # for self.model_config.is_attention_free.
        if len(self.kv_cache_config.kv_cache_groups) == 0:
            return

        if self.reorder_batch_threshold is not None:
            reorder_batch_to_split_decodes_and_prefills(
                self.input_batch,
                scheduler_output,
                decode_threshold=self.reorder_batch_threshold,
            )

    def _init_kv_zero_meta(self) -> None:
        """One-time precomputation for _zero_block_ids.

        Delegates to KVBlockZeroer.init_meta with the runner's state.
        Called from gpu_worker.py outside the CuMem pool context.
        """
        self._kv_block_zeroer = KVBlockZeroer(self.device, self.pin_memory)
        self._kv_block_zeroer.init_meta(
            attn_groups_iter=self._kv_cache_spec_attn_group_iterator(),
            kernel_block_sizes=self._kernel_block_sizes,
            cache_dtype=self.cache_config.cache_dtype,
            runner_only_attn_layers=self.runner_only_attn_layers,
            static_forward_context=(self.compilation_config.static_forward_context),
        )

    def _zero_block_ids(self, block_ids: list[int]) -> None:
        """Zero the KV cache memory for the given block IDs."""
        if hasattr(self, "_kv_block_zeroer"):
            self._kv_block_zeroer.zero_block_ids(block_ids)

    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
        """Update the cached states and the persistent batch with the scheduler
        output.

        The updated states are used by the `_prepare_inputs` function to create
        the input GPU tensors for the model.

        The SamplingMetadata is updated and copied to the GPU if there is a
        new/resumed/paused/finished request in the batch.
        """
        # Remove finished requests from the cached states.
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)
            self.num_prompt_logprobs.pop(req_id, None)
        # Remove the finished requests from the persistent batch.
        # NOTE(woosuk): There could be an edge case where finished_req_ids and
        # scheduled_req_ids overlap. This happens when a request is aborted and
        # then resubmitted with the same ID. In this case, we treat them as two
        # distinct requests - clearing the cached states for the first request
        # and handling the second as a new request.
        for req_id in scheduler_output.finished_req_ids:
            self.input_batch.remove_request(req_id)

        # Zero GPU memory for freshly allocated cache blocks to prevent
        # stale NaN/data from corrupting attention or SSM computation.
        if scheduler_output.new_block_ids_to_zero:
            self._zero_block_ids(scheduler_output.new_block_ids_to_zero)

        # Free the cached encoder outputs.
        for mm_hash in scheduler_output.free_encoder_mm_hashes:
            self.encoder_cache.pop(mm_hash, None)

        # Remove the unscheduled requests from the persistent batch.
        # NOTE(woosuk): The unscheduled requests are either preempted requests
        # or running requests that are not scheduled in this step. We remove
        # them from the persistent batch but keep their cached states since
        # they will be scheduled again sometime in the future.
        scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
        cached_req_ids = self.input_batch.req_id_to_index.keys()
        resumed_req_ids = scheduler_output.scheduled_cached_reqs.resumed_req_ids
        # NOTE(zhuohan): cached_req_ids and resumed_req_ids are usually disjoint,
        # so `(scheduled_req_ids - resumed_req_ids) == scheduled_req_ids` holds
        # apart from the forced-preemption case in reset_prefix_cache. And in
        # that case we include the resumed_req_ids in the unscheduled set so
        # that they get cleared from the persistent batch before being re-scheduled
        # in the normal resumed request path.
        unscheduled_req_ids = cached_req_ids - (scheduled_req_ids - resumed_req_ids)
        # NOTE(woosuk): The persistent batch optimization assumes that
        # consecutive batches contain mostly the same requests. If batches
        # have low request overlap (e.g., alternating between two distinct
        # sets of requests), this optimization becomes very inefficient.
        for req_id in unscheduled_req_ids:
            self.input_batch.remove_request(req_id)

        is_ngram_gpu = self.speculative_config is not None and False
        pass  # Unsupported P4 branch removed.

        reqs_to_add: list[CachedRequestState] = []
        # Add new requests to the cached states.
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            if req_id in self.requests:
                # For streaming case only.
                req_state = self._update_streaming_request(req_id, new_req_data)
                reqs_to_add.append(req_state)
                continue

            sampling_params = new_req_data.sampling_params
            pooling_params = new_req_data.pooling_params

            if (
                sampling_params
                and sampling_params.sampling_type == SamplingType.RANDOM_SEED
            ):
                generator = torch.Generator(device=self.device)
                generator.manual_seed(sampling_params.seed)
            else:
                generator = None

            pass  # Unsupported P4 branch removed.

            req_state = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                prompt_embeds=new_req_data.prompt_embeds,
                mm_features=new_req_data.mm_features,
                sampling_params=sampling_params,
                pooling_params=pooling_params,
                generator=generator,
                block_ids=new_req_data.block_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=[],
                lora_request=new_req_data.lora_request,
            )
            self.requests[req_id] = req_state

            if sampling_params and sampling_params.prompt_logprobs is not None:
                self.num_prompt_logprobs[req_id] = (
                    self.input_batch.vocab_size
                    if sampling_params.prompt_logprobs == -1
                    else sampling_params.prompt_logprobs
                )

            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            pass  # Unsupported P4 branch removed.

            # Only relevant for models using XD-RoPE (e.g, HunYuan-VL)
            pass  # Unsupported P4 branch removed.

            reqs_to_add.append(req_state)
            # Track new requests for ngram_gpu full tensor copy
            pass  # Unsupported P4 branch removed.

        # Update the states of the running/resumed requests.
        is_last_rank = get_pp_group().is_last_rank
        req_data = scheduler_output.scheduled_cached_reqs
        scheduled_spec_tokens = scheduler_output.scheduled_spec_decode_tokens

        # Save scheduler-allocated spec lengths before trimming so
        # prev_num_draft_len keeps the optimistic count for rejection correction.
        original_num_spec_per_req: dict[str, int] = {}
        pass  # Unsupported P4 branch removed.

        # Wait until valid_sampled_tokens_count is copied to cpu,
        # then use it to update actual num_computed_tokens of each request.
        valid_sampled_token_count = self._get_valid_sampled_token_count()

        for i, req_id in enumerate(req_data.req_ids):
            req_state = self.requests[req_id]
            num_computed_tokens = req_data.num_computed_tokens[i]
            new_block_ids = req_data.new_block_ids[i]
            resumed_from_preemption = req_id in req_data.resumed_req_ids
            num_output_tokens = req_data.num_output_tokens[i]
            req_index = self.input_batch.req_id_to_index.get(req_id)

            if req_state.prev_num_draft_len and self.use_async_scheduling:
                # prev_num_draft_len is used in async scheduling mode with
                # spec decode. it indicates if need to update num_computed_tokens
                # of the request. for example:
                # first step: num_computed_tokens = 0, spec_tokens = [],
                # prev_num_draft_len = 0.
                # second step: num_computed_tokens = 100(prompt length),
                # spec_tokens = [a,b], prev_num_draft_len = 0.
                # third step: num_computed_tokens = 100 + 2, spec_tokens = [c,d],
                # prev_num_draft_len = 2.
                # num_computed_tokens in first step and second step doesn't contain
                # the spec tokens length, but in third step it contains the
                # spec tokens length. we only need to update num_computed_tokens
                # when prev_num_draft_len > 0.
                if req_index is None:
                    req_state.prev_num_draft_len = 0
                else:
                    assert self.input_batch.prev_req_id_to_index is not None
                    prev_req_index = self.input_batch.prev_req_id_to_index[req_id]
                    num_accepted = valid_sampled_token_count[prev_req_index] - 1
                    num_rejected = req_state.prev_num_draft_len - num_accepted
                    num_computed_tokens -= num_rejected
                    req_state.output_token_ids.extend([-1] * num_accepted)

                    pass  # Unsupported P4 branch removed.

            # Update the cached states.
            req_state.num_computed_tokens = num_computed_tokens

            if not is_last_rank:
                if not req_data.new_token_ids:
                    # Async scheduled PP: Sampled tokens propagated via GPU broadcast.
                    new_token_ids: list[int] = []
                else:
                    # Non-async scheduling with PP: The scheduler sends
                    # sampled token ids back because there's no direct communication
                    # between the first-stage worker and the last-stage worker.
                    new_token_ids = req_data.new_token_ids[i]
                    # Add the sampled token(s) from the previous step (if any).
                    # This doesn't include "unverified" tokens like spec tokens.
                    num_new_tokens = (
                        num_computed_tokens + len(new_token_ids) - req_state.num_tokens
                    )
                    if num_new_tokens == 1:
                        # Avoid slicing list in most common case.
                        req_state.output_token_ids.append(new_token_ids[-1])
                    elif num_new_tokens > 0:
                        req_state.output_token_ids.extend(
                            new_token_ids[-num_new_tokens:]
                        )
            elif num_output_tokens < len(req_state.output_token_ids):
                # Some output tokens were discarded due to a sync-KV-load
                # failure. Align the cached state.
                del req_state.output_token_ids[num_output_tokens:]
                if req_index is not None:
                    end_idx = (
                        self.input_batch.num_prompt_tokens[req_index]
                        + num_output_tokens
                    )
                    self.input_batch.num_tokens_no_spec[req_index] = end_idx

            # Update the block IDs.
            if not resumed_from_preemption:
                if new_block_ids is not None:
                    # Append the new blocks to the existing block IDs.
                    for block_ids, new_ids in zip(req_state.block_ids, new_block_ids):
                        block_ids.extend(new_ids)
            else:
                assert req_index is None
                assert new_block_ids is not None
                # The request is resumed from preemption.
                # Replace the existing block IDs with the new ones.
                req_state.block_ids = new_block_ids

            if req_index is None:
                # The request is not in the persistent batch.
                # The request was either preempted and resumed later, or was not
                # scheduled in the previous step and needs to be added again.

                if self.use_async_scheduling and num_output_tokens > 0:
                    # We must recover the output token ids for resumed requests in the
                    # async scheduling case, so that correct input_ids are obtained.
                    resumed_token_ids = req_data.all_token_ids[req_id]
                    req_state.output_token_ids = resumed_token_ids[-num_output_tokens:]

                reqs_to_add.append(req_state)
                # Track resumed requests for ngram_gpu full tensor copy
                pass  # Unsupported P4 branch removed.
                continue

            # Update the persistent batch.
            self.input_batch.num_computed_tokens_cpu[req_index] = num_computed_tokens
            if new_block_ids is not None:
                self.input_batch.block_table.append_row(new_block_ids, req_index)

            # For the last rank, we don't need to update the token_ids_cpu
            # because the sampled tokens are already cached.
            if not is_last_rank:
                # Add new_token_ids to token_ids_cpu.
                start_token_index = num_computed_tokens
                end_token_index = num_computed_tokens + len(new_token_ids)
                self.input_batch.token_ids_cpu[
                    req_index, start_token_index:end_token_index
                ] = new_token_ids
                self.input_batch.num_tokens_no_spec[req_index] = end_token_index

            # Add spec_token_ids to token_ids_cpu.
            self.input_batch.update_req_spec_token_ids(req_state, scheduled_spec_tokens)
            # Restore scheduler-side draft count after ngram trimming.
            if original_num_spec_per_req:
                orig = original_num_spec_per_req.get(req_id, 0)
                if orig != req_state.prev_num_draft_len:
                    req_state.prev_num_draft_len = orig

        # Add the new or resumed requests to the persistent batch.
        # The smaller empty indices are filled first.
        for request in reqs_to_add:
            self.input_batch.add_request(request)
            self.input_batch.update_req_spec_token_ids(request, scheduled_spec_tokens)

        # Condense the batched states if there are gaps left by removed requests
        self.input_batch.condense()
        # Allow attention backend to reorder the batch, potentially
        self._may_reorder_batch(scheduler_output)
        # Refresh batch metadata with any pending updates.
        self.input_batch.refresh_metadata()

        # Incrementally update ngram_gpu tensors after batch is stable
        pass  # Unsupported P4 branch removed.

    def _update_states_after_model_execute(
        self, output_token_ids: torch.Tensor, scheduler_output: "SchedulerOutput"
    ) -> None:
        """GLM DSA/MTP has no recurrent model state to roll back after acceptance."""
        return

    def _update_streaming_request(
        self, req_id: str, new_req_data: NewRequestData
    ) -> CachedRequestState:
        """Updates streaming session request from `scheduled_new_reqs`.

        Removes the request from InputBatch (if present), updates the cached
        state, and prepares it for re-addition to the batch.

        NOTE: prompt_token_ids includes intermediate output tokens - tokens
        previously generated but now are input context (part of the prompt).
        """
        self.input_batch.remove_request(req_id)
        req_state = self.requests[req_id]

        req_state.prompt_token_ids = new_req_data.prompt_token_ids
        req_state.mm_features = new_req_data.mm_features
        req_state.prompt_embeds = new_req_data.prompt_embeds
        req_state.sampling_params = new_req_data.sampling_params
        req_state.pooling_params = new_req_data.pooling_params
        req_state.block_ids = new_req_data.block_ids
        req_state.num_computed_tokens = new_req_data.num_computed_tokens
        req_state.num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
            req_state.prompt_token_ids, req_state.prompt_embeds
        )

        # Clear `output_token_ids` as previous output tokens are now part of
        # `prompt_token_ids`.
        req_state.output_token_ids.clear()

        pass  # Unsupported P4 branch removed.

        return req_state

    def _get_cumsum_and_arange(
        self,
        num_tokens: np.ndarray,
        cumsum_dtype: np.dtype | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get the cumulative sum and batched arange of the given array.
        # E.g., [2, 5, 3] -> ([2, 7, 10], [0, 1, 0, 1, 2, 3, 4, 0, 1, 2])
        # Equivalent to but faster than:
        # np.concatenate([np.arange(n) for n in num_tokens])
        """
        # Step 1. [2, 5, 3] -> [2, 7, 10]
        cu_num_tokens = np.cumsum(num_tokens, dtype=cumsum_dtype)
        total_num_tokens = cu_num_tokens[-1]
        # Step 2. [2, 7, 10] -> [0, 0, 2, 2, 2, 2, 2, 7, 7, 7]
        cumsums_offsets = np.repeat(cu_num_tokens - num_tokens, num_tokens)
        # Step 3. [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        arange = self.arange_np[:total_num_tokens] - cumsums_offsets

        return cu_num_tokens, arange

    def _prepare_input_ids(
        self,
        scheduler_output: "SchedulerOutput",
        total_num_scheduled_tokens: int,
        cu_num_tokens: np.ndarray,
    ) -> None:
        """Prepare the input IDs for the current batch.

        Carefully handles the `prev_sampled_token_ids` which can be cached
        from the previous engine iteration, in which case those tokens on the
        GPU need to be copied into the corresponding slots into input_ids."""

        if self.input_batch.prev_sampled_token_ids is None:
            # Normal scheduling case
            self.input_ids.copy_to_gpu(total_num_scheduled_tokens)
            pass  # Unsupported P4 branch removed.
            return

        # Async scheduling case, where some decode requests from the previous
        # iteration won't have entries in input_ids_cpu and need to be copied
        # on the GPU from prev_sampled_token_ids.
        prev_req_id_to_index = self.input_batch.prev_req_id_to_index
        assert prev_req_id_to_index is not None
        sample_flattened_indices: list[int] = []
        spec_flattened_indices: list[int] = []
        prev_common_req_indices: list[int] = []
        prev_draft_token_indices: list[int] = []
        indices_match = True
        max_flattened_index = -1
        total_num_spec_tokens = 0
        scheduled_spec_tokens = scheduler_output.scheduled_spec_decode_tokens

        for req_id, cur_index in self.input_batch.req_id_to_index.items():
            if (prev_index := prev_req_id_to_index.get(req_id)) is not None:
                prev_common_req_indices.append(prev_index)
                # We need to compute the flattened input_ids index of the
                # last token in each common request.
                draft_len = len(scheduled_spec_tokens.get(req_id, ()))
                total_num_spec_tokens += draft_len
                flattened_index = cu_num_tokens[cur_index].item() - 1
                # example: cu_num_tokens = [2, 5, 8], draft_tokens = [1, 2, 2]
                # sample_flattened_indices = [0, 2, 5]
                # spec_flattened_indices = [1,   3, 4,    6, 7]
                sample_flattened_indices.append(flattened_index - draft_len)
                spec_flattened_indices.extend(
                    range(flattened_index - draft_len + 1, flattened_index + 1)
                )
                start = prev_index * self.num_spec_tokens
                # prev_draft_token_indices is used to find which draft_tokens_id
                # should be copied to input_ids
                # example: prev draft_tokens_id [[1,2], [3,4], [5, 6]]
                # flatten draft_tokens_id [1,2,3,4,5,6]
                # draft_len of each request [1, 2, 1]
                # then prev_draft_token_indices is [0,   2, 3,   4]
                prev_draft_token_indices.extend(range(start, start + draft_len))
                indices_match &= prev_index == flattened_index
                max_flattened_index = max(max_flattened_index, flattened_index)
        num_common_tokens = len(sample_flattened_indices)
        total_without_spec = total_num_scheduled_tokens - total_num_spec_tokens
        if num_common_tokens < total_without_spec:
            # If not all requests are decodes from the last iteration,
            # We need to copy the input_ids_cpu to the GPU first.
            self.input_ids.copy_to_gpu(total_num_scheduled_tokens)
            pass  # Unsupported P4 branch removed.
        if num_common_tokens == 0:
            # No requests in common with the previous iteration
            # So input_ids.cpu will have all the input ids.
            return
        if indices_match and max_flattened_index == (num_common_tokens - 1):
            # Common-case optimization: the batch is unchanged
            # and no reordering happened.
            # The indices are both the same permutation of 0..N-1 so
            # we can copy directly using a single slice.
            self.input_ids.gpu[:num_common_tokens].copy_(
                self.input_batch.prev_sampled_token_ids[:num_common_tokens, 0],
                non_blocking=True,
            )
            pass  # Unsupported P4 branch removed.
            return
        # Upload the index tensors asynchronously so the scatter can be non-blocking.
        sampled_tokens_index_tensor = torch.tensor(
            sample_flattened_indices, dtype=torch.int64, pin_memory=self.pin_memory
        ).to(self.device, non_blocking=True)
        prev_common_req_indices_tensor = torch.tensor(
            prev_common_req_indices, dtype=torch.int64, pin_memory=self.pin_memory
        ).to(self.device, non_blocking=True)
        self.input_ids.gpu.scatter_(
            dim=0,
            index=sampled_tokens_index_tensor,
            src=self.input_batch.prev_sampled_token_ids[
                prev_common_req_indices_tensor, 0
            ],
        )

        # Scatter the draft tokens after the sampled tokens are scattered.
        if self._draft_token_ids is None or not spec_flattened_indices:
            return

        assert isinstance(self._draft_token_ids, torch.Tensor)
        draft_tokens_index_tensor = torch.tensor(
            spec_flattened_indices, dtype=torch.int64, pin_memory=self.pin_memory
        ).to(self.device, non_blocking=True)
        prev_draft_token_indices_tensor = torch.tensor(
            prev_draft_token_indices, dtype=torch.int64, pin_memory=self.pin_memory
        ).to(self.device, non_blocking=True)

        # because input_ids dtype is torch.int32,
        # so convert draft_token_ids to torch.int32 here.
        draft_token_ids = self._draft_token_ids.to(dtype=torch.int32)

        self.input_ids.gpu.scatter_(
            dim=0,
            index=draft_tokens_index_tensor,
            src=draft_token_ids.flatten()[prev_draft_token_indices_tensor],
        )

    def _get_encoder_seq_lens(
        self,
        num_scheduled_tokens: dict[str, int],
        kv_cache_spec: KVCacheSpec,
        num_reqs: int,
        for_cudagraph_capture: bool = False,
    ) -> tuple[torch.Tensor | None, np.ndarray | None]:
        if not isinstance(kv_cache_spec, CrossAttentionSpec):
            return None, None

        # Zero out buffer for padding requests that are not actually scheduled (CGs)
        self.encoder_seq_lens.np[:num_reqs] = 0

        # Build encoder_seq_lens array mapping request indices to
        # encoder lengths for inputs scheduled in this batch
        for req_id in num_scheduled_tokens:
            req_index = self.input_batch.req_id_to_index[req_id]
            req_state = self.requests[req_id]
            if req_state.mm_features is None:
                self.encoder_seq_lens.np[req_index] = 0
                continue

            # Get the total number of encoder input tokens for running encoder requests
            # whether encoding is finished or not so that cross-attention knows how
            # many encoder tokens to attend to.
            encoder_input_tokens = sum(
                feature.mm_position.length for feature in req_state.mm_features
            )
            self.encoder_seq_lens.np[req_index] = encoder_input_tokens
        if for_cudagraph_capture:
            # During CUDA graph capture, we need to use realistic encoder lengths
            # so that max_seqlen_k is captured with the correct value.
            max_encoder_len = getattr(
                self.model_config.hf_config,
                "max_source_positions",
                self.max_encoder_len,
            )
            self.encoder_seq_lens.np[:num_reqs] = max_encoder_len

        self.encoder_seq_lens.copy_to_gpu(num_reqs)
        encoder_seq_lens = self.encoder_seq_lens.gpu[:num_reqs]
        encoder_seq_lens_cpu = self.encoder_seq_lens.np[:num_reqs]

        return encoder_seq_lens, encoder_seq_lens_cpu

    def _compute_cascade_attn_prefix_lens(
        self,
        num_scheduled_tokens: np.ndarray,
        num_computed_tokens: np.ndarray,
        num_common_prefix_blocks: list[int],
    ) -> list[list[int]] | None:
        """
        :return: Optional[cascade_attn_prefix_lens]
            cascade_attn_prefix_lens is 2D: ``[kv_cache_group_id][attn_group_idx]``,
            None if we should not use cascade attention
        """

        use_cascade_attn = False
        num_kv_cache_groups = len(self.kv_cache_config.kv_cache_groups)
        cascade_attn_prefix_lens: list[list[int]] = [
            [] for _ in range(num_kv_cache_groups)
        ]

        for kv_cache_gid in range(num_kv_cache_groups):
            for attn_group in self.attn_groups[kv_cache_gid]:
                if isinstance(attn_group.kv_cache_spec, EncoderOnlyAttentionSpec):
                    cascade_attn_prefix_len = 0
                else:
                    # 0 if cascade attention should not be used
                    cascade_attn_prefix_len = self._compute_cascade_attn_prefix_len(
                        num_scheduled_tokens,
                        num_computed_tokens,
                        num_common_prefix_blocks[kv_cache_gid],
                        attn_group.kv_cache_spec,
                        attn_group.get_metadata_builder(),
                    )
                cascade_attn_prefix_lens[kv_cache_gid].append(cascade_attn_prefix_len)
                use_cascade_attn |= cascade_attn_prefix_len > 0

        return cascade_attn_prefix_lens if use_cascade_attn else None

    def _compute_cascade_attn_prefix_len(
        self,
        num_scheduled_tokens: np.ndarray,
        num_computed_tokens: np.ndarray,
        num_common_prefix_blocks: int,
        kv_cache_spec: KVCacheSpec,
        attn_metadata_builder: AttentionMetadataBuilder,
    ) -> int:
        """Compute the length of the common prefix for cascade attention.

        NOTE(woosuk): The common prefix length returned by this function
        represents the length used specifically for cascade attention, not the
        actual number of tokens shared between requests. When cascade attention
        is disabled (use_cascade=False), this function returns 0 even if
        requests share common tokens. Additionally, the common prefix length is
        truncated to a multiple of the block size and may be further truncated
        due to implementation details explained below.

        Args:
            num_scheduled_tokens: Number of tokens scheduled per request.
            num_common_prefix_blocks: Number of shared KV cache blocks.

        Returns:
            int: Length of common prefix in tokens.
        """

        common_prefix_len = num_common_prefix_blocks * kv_cache_spec.block_size
        if common_prefix_len == 0:
            # Common case.
            return 0

        # NOTE(woosuk): Cascade attention uses two attention kernels: one
        # for the common prefix and the other for the rest. For the first
        # kernel, we concatenate all the query tokens (possibly from
        # different requests) and treat them as if they are from the same
        # request. Then, we use bi-directional attention to process the
        # common prefix in the KV cache. Importantly, this means that the
        # first kernel does not do any masking.

        # Consider the following example:
        # Request 1's input query: [D, E, X]
        # Request 1's kv cache: [A, B, C, D, E, X]
        # Request 1's num_computed_tokens: 3 (i.e., [A, B, C])
        # Request 2's input query: [E, Y]
        # Request 2's kv cache: [A, B, C, D, E, Y]
        # Request 2's num_computed_tokens: 4 (i.e., [A, B, C, D])

        # If we use [A, B, C, D, E] as the common prefix, then the
        # first kernel will compute the bi-directional attention between
        # input query [D, E, X, E, Y] and common prefix [A, B, C, D, E].
        # However, this is wrong because D in Request 1 should not attend to
        # E in the common prefix (i.e., we need masking).
        # To avoid this, [A, B, C, D] should be the common prefix.
        # That is, the common prefix should be capped by the minimum
        # num_computed_tokens among the requests, and plus one to include
        # the first token of the query.

        # In practice, we use [A, B, C] as the common prefix, instead of
        # [A, B, C, D] (i.e., the common prefix is capped by the minimum
        # num_computed_tokens, without plus one).
        # This is because of an implementation detail: We want to always
        # use two kernels for cascade attention. Let's imagine:
        # Request 3's input query: [D]
        # Request 3's kv cache: [A, B, C, D]
        # Request 3's num_computed_tokens: 3 (i.e., [A, B, C])
        # If we use [A, B, C, D] as the common prefix for Request 1-3,
        # then Request 3 will be processed only by the first kernel,
        # and the second kernel will get an empty input. While this is not
        # a fundamental problem, our current implementation does not support
        # this case.
        common_prefix_len = min(common_prefix_len, num_computed_tokens.min())
        # common_prefix_len should be a multiple of the block size.
        common_prefix_len = (
            common_prefix_len // kv_cache_spec.block_size * kv_cache_spec.block_size
        )
        use_sliding_window = isinstance(kv_cache_spec, SlidingWindowSpec) or (
            isinstance(kv_cache_spec, FullAttentionSpec)
            and kv_cache_spec.sliding_window is not None
        )
        use_local_attention = isinstance(kv_cache_spec, ChunkedLocalAttentionSpec) or (
            isinstance(kv_cache_spec, FullAttentionSpec)
            and kv_cache_spec.attention_chunk_size is not None
        )
        assert isinstance(kv_cache_spec, AttentionSpec)
        use_cascade = attn_metadata_builder.use_cascade_attention(
            common_prefix_len=common_prefix_len,
            query_lens=num_scheduled_tokens,
            num_query_heads=self.num_query_heads,
            num_kv_heads=kv_cache_spec.num_kv_heads,
            use_alibi=self.use_alibi,
            use_sliding_window=use_sliding_window,
            use_local_attention=use_local_attention,
            num_sms=self.num_sms,
            dcp_world_size=self.dcp_world_size,
        )
        return common_prefix_len if use_cascade else 0

    def _prepare_kv_sharing_fast_prefill(
        self,
        logits_indices: torch.Tensor,
    ) -> torch.Tensor:
        assert self.kv_sharing_fast_prefill_logits_indices is not None
        num_logits = logits_indices.shape[0]
        assert num_logits > 0
        self.kv_sharing_fast_prefill_logits_indices[:num_logits].copy_(logits_indices)
        # There might have leftover indices in logits_indices[num_logits:]
        # from previous iterations, whose values may be greater than the
        # batch size in the current iteration. To ensure indices are always
        # valid, we fill the padded indices with the last index.
        self.kv_sharing_fast_prefill_logits_indices[num_logits:].fill_(
            logits_indices[-1].item()
        )
        # Dispatch for the decoder portion of the model.
        _, batch_desc = self.cudagraph_dispatcher.dispatch(
            num_logits, invalid_modes={CUDAGraphMode.FULL}
        )
        num_logits_padded = batch_desc.num_tokens
        logits_indices_padded = self.kv_sharing_fast_prefill_logits_indices[
            :num_logits_padded
        ]
        return logits_indices_padded

    def get_supported_generation_tasks(self) -> list[GenerationTask]:
        model = self.get_model()
        supported_tasks = list[GenerationTask]()

        if is_text_generation_model(model):
            supported_tasks.append("generate")

        if supports_transcription(model):
            if model.supports_transcription_only:
                return ["transcription"]

            supported_tasks.append("transcription")

        if supports_realtime(model):
            supported_tasks.append("realtime")

        return supported_tasks

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        tasks = list[SupportedTask]()

        if self.model_config.runner_type == "generate":
            tasks.extend(self.get_supported_generation_tasks())
        pass  # Unsupported P4 branch removed.

        return tuple(tasks)

    def sync_and_slice_intermediate_tensors(
        self,
        num_tokens: int,
        intermediate_tensors: IntermediateTensors | None,
        sync_self: bool,
    ) -> IntermediateTensors:
        assert self.intermediate_tensors is not None

        tp = self.vllm_config.parallel_config.tensor_parallel_size
        is_rs = is_residual_scattered_for_sp(self.vllm_config, num_tokens)

        # When sequence parallelism is enabled, the "residual" tensor is sharded
        # across tensor parallel ranks, so each rank only needs its own slice.
        if sync_self:
            assert intermediate_tensors is not None
            for k, v in intermediate_tensors.items():
                is_scattered = k == "residual" and is_rs
                copy_len = num_tokens // tp if is_scattered else num_tokens
                self.intermediate_tensors[k][:copy_len].copy_(
                    v[:copy_len], non_blocking=True
                )

        return IntermediateTensors(
            {
                k: v[: num_tokens // tp]
                if k == "residual" and is_rs
                else v[:num_tokens]
                for k, v in self.intermediate_tensors.items()
            }
        )

    def eplb_step(self, is_dummy: bool = False, is_profile: bool = False) -> None:
        """
        Step for the EPLB (Expert Parallelism Load Balancing) state.
        """
        if not self.parallel_config.enable_eplb or self.eep_eplb_suppressed:
            return

        assert self.eplb_state is not None
        model = self.get_model()
        assert is_mixture_of_experts(model)
        self.eplb_state.step(
            is_dummy,
            is_profile,
            log_stats=self.parallel_config.eplb_config.log_balancedness,
        )

    def setup_eplb_from_mapping(
        self,
        expanded_physical_to_logical: torch.Tensor,
        old_num_physical_experts: int,
    ) -> None:
        model = self.get_model()
        assert is_mixture_of_experts(model)

        self.eplb_state = EplbState.from_mapping(
            model=model,
            model_config=self.model_config,
            device=self.device,
            parallel_config=self.parallel_config,
            expanded_physical_to_logical=expanded_physical_to_logical,
            num_valid_physical_experts=old_num_physical_experts,
        )

    def _preprocess(
        self,
        scheduler_output: "SchedulerOutput",
        num_input_tokens: int,  # Padded
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor,
        IntermediateTensors | None,
        dict[str, Any],
        ECConnectorOutput | None,
    ]:
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        is_first_rank = get_pp_group().is_first_rank
        is_encoder_decoder = self.model_config.is_encoder_decoder

        # _prepare_inputs may reorder the batch, so we must gather multi
        # modal outputs after that to ensure the correct order
        ec_connector_output = None

        input_ids = self.input_ids.gpu[:num_input_tokens]
        inputs_embeds = None
        model_kwargs = self._init_model_kwargs()

        positions = self.positions.gpu[:num_input_tokens]

        if is_first_rank:
            intermediate_tensors = None
        else:
            assert intermediate_tensors is not None
            intermediate_tensors = self.sync_and_slice_intermediate_tensors(
                num_input_tokens, intermediate_tensors, True
            )

        pass  # Unsupported P4 branch removed.

        return (
            input_ids,
            inputs_embeds,
            positions,
            intermediate_tensors,
            model_kwargs,
            ec_connector_output,
        )

    @contextmanager
    def synchronize_input_prep(self):
        if self.prepare_inputs_event is None:
            yield
            return

        # Ensure prior step has finished with reused CPU tensors.
        # This is required in the async scheduling case because
        # the CPU->GPU transfer happens async.
        self.prepare_inputs_event.synchronize()
        try:
            yield
        finally:
            self.prepare_inputs_event.record()

    @staticmethod
    def _is_uniform_decode(
        max_num_scheduled_tokens: int,
        uniform_decode_query_len: int,
        num_tokens: int,
        num_reqs: int,
        force_uniform_decode: bool | None = None,
    ) -> bool:
        """
        Checks if it's a decode batch with same amount scheduled tokens
        across all requests.
        """
        return (
            (
                (max_num_scheduled_tokens == uniform_decode_query_len)
                and (num_tokens == max_num_scheduled_tokens * num_reqs)
            )
            if force_uniform_decode is None
            else force_uniform_decode
        )

    def _register_layerwise_nvtx_hooks(self) -> None:
        if self.vllm_config.observability_config.enable_layerwise_nvtx_tracing:
            raise ValueError("NVTX is not supported on Ascend; use the NPU profiler")

    def _get_slot_mappings(
        self,
        num_tokens_padded: int,
        num_reqs_padded: int,
        num_tokens_unpadded: int,
        ubatch_slices: "UBatchSlices | None" = None,
    ) -> tuple[
        dict[int, torch.Tensor] | None,
        dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None,
    ]:
        """
        Build slot mappings in both formats needed by the system.

        Args:
            num_tokens_padded: Total number of tokens (padded)
            num_reqs_padded: Total number of requests (padded)
            num_tokens_unpadded: Actual number of tokens (unpadded)
            ubatch_slices: Optional ubatch slicing info for DBO

        Returns:
            A tuple of:
            - slot_mappings_by_gid: dict[int, torch.Tensor] for attention metadata
            - slot_mappings_by_layer: dict[str, torch.Tensor] or list for ForwardContext
        """
        if not (
            hasattr(self, "kv_cache_config")
            and self.kv_cache_config is not None
            and len(self.kv_cache_config.kv_cache_groups) > 0
        ):
            return None, None

        def _get_slot_mapping(kv_cache_gid: int):
            assert num_reqs_padded is not None and num_tokens_padded is not None
            kv_cache_spec = self.kv_cache_config.kv_cache_groups[
                kv_cache_gid
            ].kv_cache_spec
            if isinstance(kv_cache_spec, EncoderOnlyAttentionSpec):
                slot_mapping = torch.zeros(
                    (num_tokens_padded,),
                    dtype=torch.int64,
                    device=self.device,
                )
            else:
                blk_table = self.input_batch.block_table[kv_cache_gid]
                slot_mapping = blk_table.slot_mapping.gpu[:num_tokens_padded]

            # Fill unused with -1. Needed for reshape_and_cache in full cuda
            # graph mode. `blk_table_tensor` -1 to match mamba PAD_SLOT_ID
            slot_mapping[num_tokens_unpadded:num_tokens_padded].fill_(-1)

            return slot_mapping

        slot_mappings_by_gid = {
            gid: _get_slot_mapping(gid)
            for gid, _ in enumerate(self.kv_cache_config.kv_cache_groups)
        }

        slot_mappings_by_layer: dict[str, torch.Tensor] = {}
        for gid, kv_cache_group in enumerate(self.kv_cache_config.kv_cache_groups):
            slot_mapping = slot_mappings_by_gid[gid]
            for layer_name in kv_cache_group.layer_names:
                slot_mappings_by_layer[layer_name] = slot_mapping

        if ubatch_slices is not None:
            result: list[dict[str, torch.Tensor]] = []
            for ubatch in ubatch_slices:
                sliced_mappings: dict[str, torch.Tensor] = {}
                for layer_name, slot_mapping in slot_mappings_by_layer.items():
                    sliced_mappings[layer_name] = slot_mapping[ubatch.token_slice]
                result.append(sliced_mappings)
            return slot_mappings_by_gid, result

        return slot_mappings_by_gid, slot_mappings_by_layer

    def _pp_broadcast_prev_sampled_token_ids(
        self, sampled_token_ids: torch.Tensor
    ) -> None:
        """Broadcast sampled token ids (GPU) from last PP stage"""
        pp = get_pp_group()
        assert pp.is_last_rank
        # `prev_sampled_token_ids` is expected to have shape [num_reqs, 1].
        assert sampled_token_ids.dim() == 2 and sampled_token_ids.shape[-1] == 1, (
            "PP+async expects sampled_token_ids to have shape [num_reqs, 1]"
        )
        torch.distributed.broadcast(
            sampled_token_ids, src=pp.rank, group=pp.device_group
        )

    def _pp_receive_prev_sampled_token_ids_to_input_batch(self) -> None:
        """Receive sampled token ids broadcast from last PP stage"""
        pp = get_pp_group()
        assert not pp.is_last_rank
        num_reqs = self.input_batch.num_reqs
        # `prev_sampled_token_ids` is expected to have shape [num_reqs, 1].
        recv = torch.empty((num_reqs, 1), dtype=torch.int32, device=self.device)
        torch.distributed.broadcast(recv, src=pp.last_rank, group=pp.device_group)
        self.input_batch.prev_sampled_token_ids = recv

        # construct `prev_req_id_to_index` here so `_prepare_input_ids`
        # can map req_id -> previous batch row
        discard_req_indices = np.nonzero(self.discard_request_mask.np[:num_reqs])[0]
        discard_req_indices_set = set(discard_req_indices)
        prev_req_id_to_index: dict[str, int] = {}
        for i, req_id in enumerate(self.input_batch.req_ids):
            if i in discard_req_indices_set:
                continue
            prev_req_id_to_index[req_id] = i
            # PP+async scheduling: advance per-request local cached output length by
            # appending a placeholder (-1) token id.
            if (req_state := self.requests.get(req_id)) is not None:
                req_state.output_token_ids.append(-1)
        self.input_batch.prev_req_id_to_index = prev_req_id_to_index

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        if not self.num_spec_tokens or not self._draft_token_req_ids:
            return None
        draft_token_ids, req_ids = self._get_draft_token_ids_cpu()
        return DraftTokenIds(req_ids, draft_token_ids)

    def _copy_draft_token_ids_to_cpu(
        self, scheduler_output: "SchedulerOutput", zeros_only: bool = False
    ) -> None:
        # Check if we need to copy draft tokens to CPU. In async scheduling,
        # we only copy when needed for structured output, penalties or bad_words.
        if self.use_async_scheduling and not (
            scheduler_output.has_structured_output_requests
            or self.input_batch.sampling_metadata.output_token_ids
        ):
            return
        # We must also set the corresponding request ids.
        self._draft_token_req_ids = self.input_batch.req_ids.copy()

        draft_token_ids: torch.Tensor = self._draft_token_ids
        if not torch.is_tensor(draft_token_ids):
            return
        assert self.draft_token_ids_event is not None
        assert self.draft_token_ids_copy_stream is not None
        assert self.draft_token_ids_cpu is not None
        default_stream = torch.npu.current_stream()
        num_reqs = draft_token_ids.shape[0]
        with torch.npu.stream(self.draft_token_ids_copy_stream):
            if not zeros_only:
                # Trigger async copy of draft token ids to cpu.
                self.draft_token_ids_copy_stream.wait_stream(default_stream)
                self.draft_token_ids_cpu[:num_reqs].copy_(
                    draft_token_ids, non_blocking=True
                )
            else:
                # No copy needed, just zero-out cpu tensor.
                self.draft_token_ids_cpu[:num_reqs] = 0
            self.draft_token_ids_event.record()

    def _get_draft_token_ids_cpu(self) -> tuple[list[list[int]], list[str]]:
        if isinstance(self._draft_token_ids, list):
            return self._draft_token_ids, self.input_batch.req_ids
        req_ids = self._draft_token_req_ids
        if req_ids is None:
            return [], []
        assert self.draft_token_ids_event is not None
        assert self.draft_token_ids_cpu is not None
        self.draft_token_ids_event.synchronize()
        return self.draft_token_ids_cpu[: len(req_ids)].tolist(), req_ids

    def _copy_valid_sampled_token_count(
        self, next_token_ids: torch.Tensor, valid_sampled_tokens_count: torch.Tensor
    ) -> None:
        if self.valid_sampled_token_count_event is None:
            return

        default_stream = torch.npu.current_stream()
        # Initialize a new stream to overlap the copy operation with
        # prepare_input of draft model.
        with torch.npu.stream(self.valid_sampled_token_count_copy_stream):
            self.valid_sampled_token_count_copy_stream.wait_stream(default_stream)  # type: ignore
            counts = valid_sampled_tokens_count
            counts_cpu = self.valid_sampled_token_count_cpu
            assert counts_cpu is not None
            counts_cpu[: counts.shape[0]].copy_(counts, non_blocking=True)
            self.valid_sampled_token_count_event.record()

        self.input_batch.prev_sampled_token_ids = next_token_ids.unsqueeze(1)

    def _get_valid_sampled_token_count(self) -> list[int]:
        # Wait until valid_sampled_tokens_count is copied to cpu,
        prev_sampled_token_ids = self.input_batch.prev_sampled_token_ids
        sampled_count_event = self.valid_sampled_token_count_event
        if sampled_count_event is None or prev_sampled_token_ids is None:
            return []

        counts_cpu = self.valid_sampled_token_count_cpu
        assert counts_cpu is not None
        sampled_count_event.synchronize()
        return counts_cpu[: prev_sampled_token_ids.shape[0]].tolist()

    def update_config(self, overrides: dict[str, Any]) -> None:
        allowed_config_names = {"load_config", "model_config"}
        for config_name, config_overrides in overrides.items():
            assert config_name in allowed_config_names, (
                f"Config `{config_name}` not supported. "
                f"Allowed configs: {allowed_config_names}"
            )
            config = getattr(self, config_name)
            new_config = update_config(config, config_overrides)
            setattr(self, config_name, new_config)

    def _get_eagle3_aux_layers_from_config(self) -> tuple[int, ...] | None:
        """Extract Eagle3 auxiliary layer indices from speculative config.

        These indices specify which hidden states from the base model should
        be used as auxiliary inputs for the Eagle3 drafter model during
        speculative decoding.

        Returns:
            Tuple of layer indices if found in draft model config,
            None otherwise.
        """
        if not (self.speculative_config and self.speculative_config.draft_model_config):
            return None

        hf_config = self.speculative_config.draft_model_config.hf_config
        if not hasattr(hf_config, "eagle_aux_hidden_state_layer_ids"):
            return None

        layer_ids = hf_config.eagle_aux_hidden_state_layer_ids
        if layer_ids and isinstance(layer_ids, (list, tuple)):
            return tuple(layer_ids)

        return None

    def reload_weights(
        self,
        weights_iterator: Iterable[tuple[str, torch.Tensor]] | None = None,
        weights_path: str | None = None,
        is_checkpoint_format: bool = True,
    ) -> None:
        """
        Reload weights from a weights iterator or from disk

        :param weights_iterator: weights to load into model
        :param weights_path: path to load weights from if weights_iterator is not
            provided. Use path of original model if neither is provided.
        :param is_checkpoint_format: set to False if weights have already been processed
            into kernel format (repacking, renaming, etc.)
        """
        # TODO(@kylesayrs): generalize to all runners and loaders
        # argument validation
        if weights_iterator is None and not is_checkpoint_format:
            logger.warning(
                "Reloading from disk means that weights will be in checkpoint format. "
                "Please use `is_checkpoint_format=True` "
                "to avoid weight reloading errors"
            )

        model = self.get_model()
        weights_to_load = {name for name, _ in model.named_parameters()}
        counter_before_reloading = time.perf_counter()

        # load weights from disk if none are provided
        if weights_iterator is None:
            model_loader = get_model_loader(self.load_config)
            if not hasattr(model_loader, "get_all_weights"):
                raise NotImplementedError(
                    f"Model reloading with `{self.load_config.load_format}` format"
                )

            if weights_path is not None:
                self.model_config.model = weights_path
            weights_iterator = model_loader.get_all_weights(self.model_config, model)
            weights_iterator = cast(
                Iterable[tuple[str, torch.Tensor]], weights_iterator
            )

        # begin loading weights
        logger.info_once("Reloading weights inplace...", scope="local")
        load_device = (
            self.vllm_config.load_config.device or self.vllm_config.device_config.device
        )
        with torch.device(load_device):
            if is_checkpoint_format:
                # load weights from checkpoint/ original model format
                initialize_layerwise_reload(model)
                loaded_weights = model.load_weights(weights_iterator)
                finalize_layerwise_reload(model, self.model_config)

            else:
                # load weights from kernel format
                logger.warning_once(
                    "Reloading with `is_checkpoint_format=True` requires that "
                    "weights be in kernel format and already sharded",
                    scope="local",
                )
                loaded_weights = set()
                for name, loaded_weight in weights_iterator:
                    param = model.get_parameter(name)  # TODO: buffers?
                    param.copy_(loaded_weight)
                    loaded_weights.add(name)

        # logging and validation
        counter_after_reloading = time.perf_counter()
        diff_seconds = counter_after_reloading - counter_before_reloading
        logger.info_once(
            "Reloading and processing weights took %.2f seconds",
            diff_seconds,
            scope="local",
        )
        if self.model_config.quantization is None and loaded_weights is not None:
            weights_not_loaded = weights_to_load - loaded_weights
            if weights_not_loaded:
                logger.warning(
                    "Following weights were not loaded from checkpoint: %s",
                    weights_not_loaded,
                )

    def _get_prompt_logprobs_dict(
        self,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: dict[str, int],
    ) -> dict[str, LogprobsTensors | None]:
        num_prompt_logprobs_dict = self.num_prompt_logprobs
        if not num_prompt_logprobs_dict:
            return {}

        in_progress_dict = self.input_batch.in_progress_prompt_logprobs_cpu
        prompt_logprobs_dict: dict[str, LogprobsTensors | None] = {}

        # Since prompt logprobs are a rare feature, prioritize simple,
        # maintainable loop over optimal performance.
        completed_prefill_reqs = []
        for req_id, num_prompt_logprobs in num_prompt_logprobs_dict.items():
            num_tokens = num_scheduled_tokens.get(req_id)
            if num_tokens is None:
                # This can happen if the request was preempted in prefill stage.
                continue

            # Get metadata for this request.
            request = self.requests[req_id]
            if request.prompt_token_ids is None:
                # Prompt logprobs is incompatible with prompt embeddings
                continue

            num_prompt_tokens = len(request.prompt_token_ids)
            prompt_token_ids = torch.tensor(request.prompt_token_ids).to(
                self.device, non_blocking=True
            )

            # Set up target LogprobsTensors object.
            logprobs_tensors = in_progress_dict.get(req_id)
            if not logprobs_tensors:
                # Create empty logprobs CPU tensors for the entire prompt.
                # If chunked, we'll copy in slice by slice.
                logprobs_tensors = LogprobsTensors.empty_cpu(
                    num_prompt_tokens - 1, num_prompt_logprobs + 1
                )
                in_progress_dict[req_id] = logprobs_tensors

            # Determine number of logits to retrieve.
            start_idx = request.num_computed_tokens
            start_tok = start_idx + 1
            num_remaining_tokens = num_prompt_tokens - start_tok
            if num_tokens <= num_remaining_tokens:
                # This is a chunk, more tokens remain.
                # In the == case, there are no more prompt logprobs to produce
                # but we want to defer returning them to the next step where we
                # have new generated tokens to return.
                num_logits = num_tokens
            else:
                # This is the last chunk of prompt tokens to return.
                num_logits = num_remaining_tokens
                completed_prefill_reqs.append(req_id)
                prompt_logprobs_dict[req_id] = logprobs_tensors

            if num_logits <= 0:
                # This can happen for the final chunk if we prefilled exactly
                # (num_prompt_tokens - 1) tokens for this request in the prior
                # step. There are no more prompt logprobs to produce.
                continue

            # Get the logits corresponding to this req's prompt tokens.
            # If this is a partial request (i.e. chunked prefill),
            # then there is prompt logprob generated for each index.
            req_idx = self.input_batch.req_id_to_index[req_id]
            offset = self.query_start_loc.np[req_idx].item()
            prompt_hidden_states = hidden_states[offset : offset + num_logits]
            logits = self.model.compute_logits(prompt_hidden_states)

            # Get the "target" tokens for each index. For prompt at index i,
            # the token at prompt index i+1 is the "sampled" token we want
            # to gather the logprob for.
            tgt_token_ids = prompt_token_ids[start_tok : start_tok + num_logits]

            # Compute prompt logprobs.
            logprobs = self.sampler.compute_logprobs(logits)
            token_ids, logprobs, ranks, _ = self.sampler.gather_logprobs(
                logprobs, num_prompt_logprobs, tgt_token_ids
            )

            # Transfer GPU->CPU async.
            chunk_slice = slice(start_idx, start_idx + num_logits)
            logprobs_tensors.logprob_token_ids[chunk_slice].copy_(
                token_ids, non_blocking=True
            )
            logprobs_tensors.logprobs[chunk_slice].copy_(logprobs, non_blocking=True)
            logprobs_tensors.selected_token_ranks[chunk_slice].copy_(
                ranks, non_blocking=True
            )

        # Remove requests that have completed prefill from the batch
        # num_prompt_logprobs_dict.
        for req_id in completed_prefill_reqs:
            del num_prompt_logprobs_dict[req_id]
            del in_progress_dict[req_id]

        # Must synchronize the non-blocking GPU->CPU transfers.
        if prompt_logprobs_dict:
            self._sync_device()

        return prompt_logprobs_dict

    def _get_nans_in_logits(
        self,
        logits: torch.Tensor | None,
    ) -> dict[str, int]:
        try:
            if logits is None:
                return {req_id: 0 for req_id in self.input_batch.req_ids}

            num_nans_in_logits = {}
            num_nans_for_index = logits.isnan().sum(dim=-1).cpu().numpy()
            for req_id in self.input_batch.req_ids:
                req_index = self.input_batch.req_id_to_index[req_id]
                num_nans_in_logits[req_id] = (
                    int(num_nans_for_index[req_index])
                    if num_nans_for_index is not None and req_index < logits.shape[0]
                    else 0
                )
            return num_nans_in_logits
        except IndexError:
            return {}

    @contextmanager
    def maybe_randomize_inputs(
        self, input_ids: torch.Tensor | None, inputs_embeds: torch.Tensor | None
    ):
        """
        Randomize input_ids if VLLM_RANDOMIZE_DP_DUMMY_INPUTS is set.
        This is to help balance expert-selection
         - during profile_run
         - during DP rank dummy run
        """

        dp_size = self.vllm_config.parallel_config.data_parallel_size
        randomize_inputs = envs.VLLM_RANDOMIZE_DP_DUMMY_INPUTS and dp_size > 1
        if not randomize_inputs:
            yield
        elif input_ids is not None:

            @functools.cache
            def rand_input_ids() -> torch.Tensor:
                return torch.randint_like(
                    self.input_ids.gpu,
                    low=0,
                    high=self.model_config.get_vocab_size(),
                )

            logger.debug_once("Randomizing dummy input_ids for DP Rank")
            input_ids.copy_(rand_input_ids()[: input_ids.size(0)], non_blocking=True)
            yield
            input_ids.fill_(0)
        else:

            @functools.cache
            def rand_inputs_embeds() -> torch.Tensor:
                return torch.randn_like(
                    self.inputs_embeds.gpu,
                )

            assert inputs_embeds is not None
            logger.debug_once("Randomizing dummy inputs_embeds for DP Rank")
            inputs_embeds.copy_(
                rand_inputs_embeds()[: inputs_embeds.size(0)], non_blocking=True
            )
            yield
            inputs_embeds.fill_(0)

    def profile_run(self) -> None:
        # Profile with multimodal encoder & encoder cache.
        pass  # Unsupported P4 branch removed.

        # Add `is_profile` here to pre-allocate communication buffers
        hidden_states, last_hidden_states = self._dummy_run(
            self.max_num_tokens, is_profile=True
        )
        if get_pp_group().is_last_rank:
            output = self._dummy_sampler_run(last_hidden_states)
        else:
            output = None
        self._sync_device()
        del hidden_states, output
        self.encoder_cache.clear()
        gc.collect()

    def _init_minimal_kv_cache_for_profiling(self) -> None:
        from vllm.v1.core.kv_cache_utils import (
            get_kv_cache_config_from_groups,
            get_kv_cache_groups,
        )

        kv_cache_spec = self.get_kv_cache_spec()
        kv_cache_groups = get_kv_cache_groups(self.vllm_config, kv_cache_spec)
        min_blocks = self.compilation_config.max_cudagraph_capture_size or 1

        # Temporarily change num_gpu_blocks_override to allocate a minimal KV cache
        saved_override = self.cache_config.num_gpu_blocks_override
        self.cache_config.num_gpu_blocks_override = min_blocks
        minimal_config = get_kv_cache_config_from_groups(
            self.vllm_config, kv_cache_groups, available_memory=0
        )
        self.cache_config.num_gpu_blocks_override = saved_override

        self.initialize_kv_cache(minimal_config)
        self.cache_config.num_gpu_blocks = minimal_config.num_blocks

        logger.debug("Initialized minimal KV cache for CUDA graph profiling")

    @staticmethod
    @contextmanager
    def _freeze_gc():
        gc.collect()
        should_freeze = not envs.VLLM_ENABLE_CUDAGRAPH_GC
        if should_freeze:
            gc.freeze()
        try:
            yield
        finally:
            if should_freeze:
                gc.unfreeze()
                gc.collect()

    def _cleanup_profiling_kv_cache(self) -> None:
        torch.npu.synchronize()
        if hasattr(self, "kv_caches") and self.kv_caches:
            for i in range(len(self.kv_caches)):
                self.kv_caches[i] = None  # type: ignore
            self.kv_caches.clear()
        if hasattr(self, "cross_layers_kv_cache"):
            self.cross_layers_kv_cache = None
            self.cross_layers_attn_backend = None
        if hasattr(self, "attn_groups"):
            self.attn_groups.clear()
        if hasattr(self, "kv_cache_config"):
            delattr(self, "kv_cache_config")
        self.cache_config.num_gpu_blocks = None

        for layer in self.compilation_config.static_forward_context.values():
            if hasattr(layer, "kv_cache"):
                layer.kv_cache = []

        gc.collect()
        torch.npu.empty_cache()

        logger.debug("Cleaned up profiling KV cache and CUDA graphs")

    @torch.inference_mode()
    def profile_cudagraph_memory(self) -> int:
        with set_current_vllm_config(self.vllm_config):
            self._init_minimal_kv_cache_for_profiling()

        saved_num_cudagraph_captured = compilation_counter.num_cudagraph_captured

        capture_descs = self.cudagraph_dispatcher.get_capture_descs()

        total_graphs = sum(len(descs) for _, descs in capture_descs)
        if total_graphs == 0:
            logger.debug("No CUDA graphs will be captured, skipping profiling")
            self._cleanup_profiling_kv_cache()
            return 0

        logger.info(
            "Profiling CUDA graph memory: %s",
            ", ".join(
                f"{mode.name}={len(descs)} (largest={descs[0].num_tokens})"
                for mode, descs in capture_descs
                if descs
            ),
        )

        # Use a temporary pool for profiling to avoid fragmentation in the main pool.
        profiling_pool = current_platform.graph_pool_handle()
        original_pools: dict[int, Any] = {}
        for instance in list(ACLGraphWrapper._all_instances):
            original_pools[id(instance)] = instance.graph_pool
            instance.graph_pool = profiling_pool

        set_cudagraph_capturing_enabled(True)
        with self._freeze_gc(), graph_capture(device=self.device):
            shared_memory_estimate = {}
            per_graph_estimate = {}
            torch.npu.synchronize()
            torch.npu.empty_cache()

            for mode, descs in capture_descs:
                profile_descs = descs[:2]
                mem_samples: list[int] = []

                for i, desc in enumerate(profile_descs):
                    mem_before = torch.npu.mem_get_info()[0]
                    self._warmup_and_capture(
                        desc,
                        cudagraph_runtime_mode=mode,
                        profile_seq_lens=(
                            min(
                                self.max_model_len,
                                self.max_num_tokens // desc.num_tokens,
                            )
                            if mode == CUDAGraphMode.FULL and i == 0
                            else None
                        ),
                    )
                    torch.npu.synchronize()
                    free_after = torch.npu.mem_get_info()[0]
                    mem_samples.append(mem_before - free_after)

                first_capture = mem_samples[0]
                # Use at least 1 MiB per graph for driver overhead
                per_graph = max(mem_samples[1] if len(mem_samples) > 1 else 0, 1 << 20)

                shared_memory_estimate[mode] = first_capture
                per_graph_estimate[mode] = per_graph * (len(descs) - 1)

                logger.debug(
                    "Estimated %s CUDA graph memory: "
                    "%.2f MiB first-capture + (%d-1) × %.2f MiB per-graph",
                    mode.name,
                    first_capture / (1 << 20),
                    len(descs),
                    per_graph / (1 << 20),
                )

        set_cudagraph_capturing_enabled(False)
        ACLGraphWrapper.clear_all_graphs()
        for instance in list(ACLGraphWrapper._all_instances):
            if id(instance) in original_pools:
                instance.graph_pool = original_pools[id(instance)]
        for key_set in self.cudagraph_dispatcher.cudagraph_keys.values():
            key_set.clear()
        self.cudagraph_dispatcher.keys_initialized = False
        self._cleanup_profiling_kv_cache()
        compilation_counter.num_cudagraph_captured = saved_num_cudagraph_captured

        # FULL and PIECEWISE graphs share the global pool at runtime and are
        # never replayed concurrently, so the pool overlays their memory.
        # Take the max to avoid double-counting the overlap.
        total_estimate = max(shared_memory_estimate.values()) + sum(
            per_graph_estimate.values()
        )
        logger.info(
            "Estimated CUDA graph memory: %.2f GiB total",
            total_estimate / (1 << 30),
        )

        return int(total_estimate)

    @instrument(span_name="Capture model")
    def capture_model(self) -> int:
        if self.compilation_config.cudagraph_mode == CUDAGraphMode.NONE:
            logger.warning(
                "Skipping CUDA graph capture. To turn on CUDA graph capture, "
                "ensure `cudagraph_mode` was not manually set to `NONE`"
            )
            return 0

        compilation_counter.num_gpu_runner_capture_triggers += 1

        start_time = time.perf_counter()

        # Trigger CUDA graph capture for specific shapes.
        # Capture the large shapes first so that the smaller shapes
        # can reuse the memory pool allocated for the large shapes.
        set_cudagraph_capturing_enabled(True)
        with self._freeze_gc(), graph_capture(device=self.device):
            torch.npu.synchronize()
            torch.npu.empty_cache()
            start_free_gpu_memory = torch.npu.mem_get_info()[0]

            for (
                runtime_mode,
                batch_descs,
            ) in self.cudagraph_dispatcher.get_capture_descs():
                self._capture_cudagraphs(
                    batch_descriptors=batch_descs,
                    cudagraph_runtime_mode=runtime_mode,
                )
                torch.npu.synchronize()

            torch.npu.synchronize()
            end_free_gpu_memory = torch.npu.mem_get_info()[0]

        # Disable cudagraph capturing globally, so any unexpected cudagraph
        # capturing will be detected and raise an error after here.
        # Note: We don't put it into graph_capture context manager because
        # we may do lazy capturing in future that still allows capturing
        # after here.
        set_cudagraph_capturing_enabled(False)

        torch.npu.synchronize()
        torch.npu.empty_cache()

        # Lock workspace to prevent resizing during execution.
        # Max workspace sizes should have been captured during warmup/profiling.
        lock_workspace()

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        cuda_graph_size = start_free_gpu_memory - end_free_gpu_memory
        # This usually takes 5~20 seconds.
        logger.info_once(
            "Graph capturing finished in %.0f secs, took %.2f GiB",
            elapsed_time,
            cuda_graph_size / (1 << 30),
            scope="local",
        )
        return cuda_graph_size

    def _warmup_and_capture(
        self,
        desc: BatchDescriptor,
        cudagraph_runtime_mode: CUDAGraphMode,
        profile_seq_lens: int | None = None,
        allow_microbatching: bool = False,
        num_warmups: int | None = None,
    ):
        if num_warmups is None:
            num_warmups = self.compilation_config.cudagraph_num_of_warmups
        force_attention = cudagraph_runtime_mode == CUDAGraphMode.FULL
        for _ in range(num_warmups):
            self._dummy_run(
                desc.num_tokens,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                force_attention=force_attention,
                uniform_decode=desc.uniform,
                allow_microbatching=allow_microbatching,
                skip_eplb=True,
                remove_lora=False,
                num_active_loras=desc.num_active_loras,
            )
        self._dummy_run(
            desc.num_tokens,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            uniform_decode=desc.uniform,
            allow_microbatching=allow_microbatching,
            skip_eplb=True,
            remove_lora=False,
            num_active_loras=desc.num_active_loras,
            is_graph_capturing=True,
            profile_seq_lens=profile_seq_lens,
        )

    def _capture_cudagraphs(
        self,
        batch_descriptors: list[BatchDescriptor],
        cudagraph_runtime_mode: CUDAGraphMode,
    ):
        assert (
            cudagraph_runtime_mode != CUDAGraphMode.NONE
            and cudagraph_runtime_mode.is_valid_runtime_mode()
        ), f"Invalid cudagraph runtime mode: {cudagraph_runtime_mode}"

        if not batch_descriptors:
            return

        uniform_decode = batch_descriptors[0].uniform

        # Only rank 0 should print progress bar during capture
        if is_global_first_rank():
            batch_descriptors = tqdm(
                batch_descriptors,
                disable=not self.load_config.use_tqdm_on_load,
                desc="Capturing CUDA graphs ({}, {})".format(
                    "decode" if uniform_decode else "mixed prefill-decode",
                    cudagraph_runtime_mode.name,
                ),
            )

        # We skip EPLB here since we don't want to record dummy metrics
        for batch_desc in batch_descriptors:
            # We currently only capture ubatched graphs when its a FULL
            # cudagraph, a uniform decode batch, and the number of tokens
            # is above the threshold. Otherwise we just capture a non-ubatched
            # version of the graph
            allow_microbatching = (
                self.parallel_config.use_ubatching
                and cudagraph_runtime_mode == CUDAGraphMode.FULL
                and uniform_decode
                and check_ubatch_thresholds(
                    config=self.vllm_config.parallel_config,
                    num_tokens=batch_desc.num_tokens,
                    uniform_decode=uniform_decode,
                )
            )
            self._warmup_and_capture(
                batch_desc,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                allow_microbatching=allow_microbatching,
            )
            torch.npu.synchronize()

    def initialize_metadata_builders(
        self, kv_cache_config: KVCacheConfig, kernel_block_sizes: list[int]
    ) -> None:
        """
        Create the metadata builders for all KV cache groups and attn groups.
        """
        for kv_cache_group_id in range(len(kv_cache_config.kv_cache_groups)):
            for attn_group in self.attn_groups[kv_cache_group_id]:
                attn_group.create_metadata_builders(
                    self.vllm_config,
                    self.device,
                    kernel_block_sizes[kv_cache_group_id]
                    if kv_cache_group_id < len(kernel_block_sizes)
                    else None,
                    num_metadata_builders=1
                    if not self.parallel_config.use_ubatching
                    else self.parallel_config.num_ubatches,
                )
        # Calculate reorder batch threshold (if needed)
        # Note (tdoublep): do this *after* constructing builders,
        # because some of them change the threshold at init time.
        self.calculate_reorder_batch_threshold()

        # Initialize drafter attention backend
        if self.speculative_config and (self.speculative_config.use_eagle() or False):
            assert isinstance(self.drafter, EagleProposer)
            self.drafter.initialize_attn_backend(kv_cache_config, kernel_block_sizes)

    def _check_and_update_cudagraph_mode(
        self,
        attention_backends: list[set[type[AttentionBackend]]],
        kv_cache_groups: list[KVCacheGroupSpec],
    ) -> None:
        """
        Resolve the cudagraph_mode when there are multiple attention
        groups with potential conflicting CUDA graph support.
        Then initialize the cudagraph_dispatcher based on the resolved
        cudagraph_mode.
        """
        min_cg_support = AttentionCGSupport.ALWAYS
        min_cg_backend_name = None

        for attn_backend_set, kv_cache_group in zip(
            attention_backends, kv_cache_groups
        ):
            for attn_backend in attn_backend_set:
                builder_cls = attn_backend.get_builder_cls()

                cg_support = builder_cls.get_cudagraph_support(
                    self.vllm_config, kv_cache_group.kv_cache_spec
                )
                if cg_support.value < min_cg_support.value:
                    min_cg_support = cg_support
                    min_cg_backend_name = attn_backend.__name__
        # Flexible resolve the cudagraph mode
        cudagraph_mode = self.compilation_config.cudagraph_mode
        assert cudagraph_mode is not None
        # check cudagraph for mixed batch is supported
        if (
            cudagraph_mode.mixed_mode() == CUDAGraphMode.FULL
            and min_cg_support != AttentionCGSupport.ALWAYS
        ):
            msg = (
                f"CUDAGraphMode.{cudagraph_mode.name} is not supported "
                f"with {min_cg_backend_name} backend (support: "
                f"{min_cg_support})"
            )
            if min_cg_support == AttentionCGSupport.NEVER:
                # if not supported any full cudagraphs, just raise it.
                msg += (
                    "; please try cudagraph_mode=PIECEWISE, and "
                    "make sure compilation mode is VLLM_COMPILE"
                )
                raise ValueError(msg)

            # attempt to resolve the full cudagraph related mode
            if self.compilation_config.splitting_ops_contain_attention():
                msg += "; setting cudagraph_mode=FULL_AND_PIECEWISE"
                cudagraph_mode = self.compilation_config.cudagraph_mode = (
                    CUDAGraphMode.FULL_AND_PIECEWISE
                )
            else:
                msg += "; setting cudagraph_mode=FULL_DECODE_ONLY"
                cudagraph_mode = self.compilation_config.cudagraph_mode = (
                    CUDAGraphMode.FULL_DECODE_ONLY
                )
            logger.warning(msg)

        # check that if we are doing decode full-cudagraphs it is supported
        if (
            cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
            and min_cg_support == AttentionCGSupport.NEVER
        ):
            msg = (
                f"CUDAGraphMode.{cudagraph_mode.name} is not supported "
                f"with {min_cg_backend_name} backend (support: "
                f"{min_cg_support})"
            )
            if self.compilation_config.mode == CompilationMode.VLLM_COMPILE and (
                self.compilation_config.splitting_ops_contain_attention()
                or self.compilation_config.use_inductor_graph_partition
            ):
                msg += (
                    "; setting cudagraph_mode=PIECEWISE because "
                    "attention is compiled piecewise"
                )
                cudagraph_mode = self.compilation_config.cudagraph_mode = (
                    CUDAGraphMode.PIECEWISE
                )
            else:
                msg += (
                    "; setting cudagraph_mode=NONE because "
                    "attention is not compiled piecewise"
                )
                cudagraph_mode = self.compilation_config.cudagraph_mode = (
                    CUDAGraphMode.NONE
                )
            logger.warning(msg)

        # check that if we are doing spec-decode + decode full-cudagraphs it is
        # supported
        if (
            cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
            and self.uniform_decode_query_len > 1
            and min_cg_support.value < AttentionCGSupport.UNIFORM_BATCH.value
        ):
            msg = (
                f"CUDAGraphMode.{cudagraph_mode.name} is not supported"
                f" with spec-decode for attention backend "
                f"{min_cg_backend_name} (support: {min_cg_support})"
            )
            if self.compilation_config.splitting_ops_contain_attention():
                msg += "; setting cudagraph_mode=PIECEWISE"
                cudagraph_mode = self.compilation_config.cudagraph_mode = (
                    CUDAGraphMode.PIECEWISE
                )
            else:
                msg += "; setting cudagraph_mode=NONE"
                cudagraph_mode = self.compilation_config.cudagraph_mode = (
                    CUDAGraphMode.NONE
                )
            logger.warning(msg)

        # double check that we can support full cudagraph if they are requested
        # even after automatic downgrades
        if (
            cudagraph_mode.has_full_cudagraphs()
            and min_cg_support == AttentionCGSupport.NEVER
        ):
            raise ValueError(
                f"CUDAGraphMode.{cudagraph_mode.name} is not "
                f"supported with {min_cg_backend_name} backend ("
                f"support:{min_cg_support}) "
                "; please try cudagraph_mode=PIECEWISE, "
                "and make sure compilation mode is VLLM_COMPILE"
            )

        # if we have dedicated decode cudagraphs, and spec-decode is enabled,
        # we need to adjust the cudagraph sizes to be a multiple of the uniform
        # decode query length to avoid: https://github.com/vllm-project/vllm/issues/28207
        # temp-fix: https://github.com/vllm-project/vllm/issues/28207#issuecomment-3504004536
        # Will be removed in the near future when we have separate cudagraph capture
        # sizes for decode and mixed prefill-decode.
        if (
            cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
            and cudagraph_mode.separate_routine()
            and self.uniform_decode_query_len > 1
        ):
            self.compilation_config.adjust_cudagraph_sizes_for_spec_decode(
                self.uniform_decode_query_len, self.parallel_config.tensor_parallel_size
            )

        # Trigger cudagraph dispatching keys initialization after
        # resolved cudagraph mode.
        self.compilation_config.cudagraph_mode = cudagraph_mode
        self.cudagraph_dispatcher.initialize_cudagraph_keys(
            cudagraph_mode, self.uniform_decode_query_len
        )

        # Initialize drafter's cudagraph dispatcher if using spec decode.
        if self.speculative_config and (
            self.speculative_config.use_eagle()
            or self.speculative_config.uses_extract_hidden_states()
        ):
            assert isinstance(self.drafter, EagleProposer)
            self.drafter.initialize_cudagraph_keys(cudagraph_mode)

    def _attn_group_iterator(self) -> Iterator[AttentionGroup]:
        return itertools.chain.from_iterable(self.attn_groups)

    def _kv_cache_spec_attn_group_iterator(self) -> Iterator[AttentionGroup]:
        if not self.kv_cache_config.kv_cache_groups:
            return
        for attn_groups in self.attn_groups:
            yield from attn_groups

    def maybe_add_kv_sharing_layers_to_kv_cache_groups(
        self, kv_cache_config: KVCacheConfig
    ) -> None:
        """
        Add layers that re-use KV cache to KV cache group of its target layer.
        Mapping of KV cache tensors happens in `initialize_kv_cache_tensors()`
        """
        if not self.shared_kv_cache_layers:
            # No cross-layer KV sharing, return
            return

        add_kv_sharing_layers_to_kv_cache_groups(
            self.shared_kv_cache_layers,
            kv_cache_config.kv_cache_groups,
            self.runner_only_attn_layers,
        )

        if self.cache_config.kv_sharing_fast_prefill:
            # In You Only Cache Once (https://arxiv.org/abs/2405.05254) or other
            # similar KV sharing setups, only the layers that generate KV caches
            # are involved in the prefill phase, enabling prefill to early exit.
            attn_layers = get_layers_from_vllm_config(self.vllm_config, Attention)
            for layer_name in reversed(attn_layers):
                if layer_name in self.shared_kv_cache_layers:
                    self.kv_sharing_fast_prefill_eligible_layers.add(layer_name)
                else:
                    break

    def _get_attention_kv_cache_gid(self) -> int:
        """Find the KV cache group index for attention layers."""
        for gid, group in enumerate(self.kv_cache_config.kv_cache_groups):
            if isinstance(group.kv_cache_spec, AttentionSpec):
                return gid
        return 0

    def init_routed_experts_capturer(self):
        logger.info(
            "Initializing routed experts capturer, enable_return_routed_experts: %s",
            self.model_config.enable_return_routed_experts,
        )
        routed_experts_capturer = RoutedExpertsCapturer.create()
        self.routed_experts_attn_gid = self._get_attention_kv_cache_gid()
        min_block_size = min(
            [
                group.kv_cache_spec.block_size
                for group in self.kv_cache_config.kv_cache_groups
            ]
        )
        num_groups = len(self.kv_cache_config.kv_cache_groups)
        self.max_num_kv_tokens = (
            self.kv_cache_config.num_blocks // num_groups
        ) * min_block_size
        dcp_size = self.vllm_config.parallel_config.decode_context_parallel_size
        pcp_size = self.vllm_config.parallel_config.prefill_context_parallel_size
        if pcp_size * dcp_size > 1:
            self.max_num_kv_tokens *= pcp_size * dcp_size

        routed_experts_capturer.init_buffer(
            max_num_batched_tokens=self.scheduler_config.max_num_batched_tokens,
            max_num_kv_tokens=self.max_num_kv_tokens,
            vllm_config=self.vllm_config,
        )
        self._bind_routed_experts_capturer(routed_experts_capturer)
        self.routed_experts_initialized = True

    def _bind_routed_experts_capturer(self, capturer: RoutedExpertsCapturer) -> None:
        from vllm.model_executor.layers.fused_moe.layer import FusedMoE
        from vllm.model_executor.layers.fused_moe.router.base_router import (
            BaseRouter,
        )

        for module in self.compilation_config.static_forward_context.values():
            if isinstance(module, FusedMoE) and isinstance(module.router, BaseRouter):
                layer_id = module.layer_id

                def _capture_fn(topk_ids, _layer_id=layer_id, _capturer=capturer):
                    _capturer.capture(_layer_id, topk_ids)

                module.router.set_capture_fn(_capture_fn)

    def may_add_encoder_only_layers_to_kv_cache_config(self) -> None:
        """
        Add encoder-only layers to the KV cache config.
        """
        block_size = self.vllm_config.cache_config.block_size
        encoder_only_attn_specs: dict[AttentionSpec, list[str]] = defaultdict(list)
        attn_layers = get_layers_from_vllm_config(self.vllm_config, Attention)
        for layer_name, attn_module in attn_layers.items():
            if attn_module.attn_type == AttentionType.ENCODER_ONLY:
                attn_spec: AttentionSpec = EncoderOnlyAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=attn_module.num_kv_heads,
                    head_size=attn_module.head_size,
                    dtype=self.kv_cache_dtype,
                )
                encoder_only_attn_specs[attn_spec].append(layer_name)
                self.runner_only_attn_layers.add(layer_name)
        if len(encoder_only_attn_specs) > 0:
            assert len(encoder_only_attn_specs) == 1, (
                "Only support one encoder-only attention spec now"
            )
            spec, layer_names = encoder_only_attn_specs.popitem()
            self.kv_cache_config.kv_cache_groups.append(
                KVCacheGroupSpec(layer_names=layer_names, kv_cache_spec=spec)
            )

    def _to_list(self, sampled_token_ids: torch.Tensor) -> list[list[int]]:
        # This is a short term mitigation for issue mentioned in
        # https://github.com/vllm-project/vllm/issues/22754.
        # `tolist` would trigger a cuda wise stream sync, which
        # would block other copy ops from other cuda streams.
        # A cuda event sync would avoid such a situation. Since
        # this is in the critical path of every single model
        # forward loop, this has caused perf issue for a disagg
        # setup.
        pinned = self.sampled_token_ids_pinned_cpu[: sampled_token_ids.shape[0]]
        pinned.copy_(sampled_token_ids, non_blocking=True)
        self.transfer_event.record()
        self.transfer_event.synchronize()
        return pinned.tolist()

    def get_encoder_timing_stats(self) -> dict[str, dict[str, float | int]]:
        """
        Get encoder timing stats for all requests and clear the registry.

        Returns:
            Dictionary mapping request_id to stats dict.
        """
        with self._encoder_timing_lock:
            stats = {
                req_id: stats_obj.to_dict()
                for req_id, stats_obj in self.encoder_timing_registry.items()
            }
            self.encoder_timing_registry.clear()
            return stats

    @contextmanager
    def timed_encoder_operation(
        self,
        should_time: bool,
        group_lora_refs: list[tuple[str, Any]],
        current_item_idx: int,
        num_items: int,
    ):
        """
        Context manager to time encoder forward operations.

        Args:
            should_time: Whether timing is enabled
            group_lora_refs: Full list of (request_id, pos_info) tuples
            current_item_idx: Starting index for this group
            num_items: Number of items in this group
        """
        if not should_time:
            yield
            return

        group_refs = group_lora_refs[current_item_idx : current_item_idx + num_items]
        group_request_ids = {req_id for req_id, _ in group_refs}

        torch.npu.synchronize()
        start_time = time.perf_counter()

        try:
            yield
        finally:
            torch.npu.synchronize()
            elapsed = time.perf_counter() - start_time

            per_request_time = elapsed / max(len(group_request_ids), 1)

            with self._encoder_timing_lock:
                for req_id in group_request_ids:
                    if req_id not in self.encoder_timing_registry:
                        self.encoder_timing_registry[req_id] = EncoderTimingStats()

                    stats = self.encoder_timing_registry[req_id]
                    stats.encoder_forward_secs += per_request_time
                    stats.num_encoder_calls += 1


@dataclass
class EncoderTimingStats:
    """Per-request timing statistics for encoder forward pass."""

    encoder_forward_secs: float = 0.0
    """Time spent in vision encoder forward pass (seconds)."""

    num_encoder_calls: int = 0
    """Number of times encoder was called for this request."""

    def to_dict(self) -> dict[str, float | int]:
        return {
            "encoder_forward_secs": self.encoder_forward_secs,
            "num_encoder_calls": self.num_encoder_calls,
        }
