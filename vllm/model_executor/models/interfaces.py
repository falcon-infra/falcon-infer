# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import (
    Iterable,
    MutableSequence,
    Sequence,
)
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Literal,
    Protocol,
    TypeAlias,
    overload,
    runtime_checkable,
)

import torch
import torch.nn as nn
from torch import Tensor
from typing_extensions import Self, TypeIs

from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateCopyFunc
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.tasks import ScoreType
from vllm.utils.func_utils import supports_kw

from .interfaces_base import VllmModel

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.models.utils import WeightsMapper
    from vllm.inputs.legacy_wire import MultiModalFeatureSpec
    from typing import Any as _ProcessorFactories
    from vllm.sequence import IntermediateTensors
else:
    VllmConfig = object
    WeightsMapper = object
    MultiModalFeatureSpec = object
    _ProcessorFactories = object
    IntermediateTensors = object

logger = init_logger(__name__)

MultiModalEmbeddings: TypeAlias = list[Tensor] | Tensor | tuple[Tensor, ...]
"""
The output embeddings must be one of the following formats:

- A list or tuple of 2D tensors, where each tensor corresponds to
    each input multimodal data item (e.g, image).
- A single 3D tensor, with the batch dimension grouping the 2D tensors.
"""


# Cache results of `SupportsMultiModal.get_language_model`
_language_model_by_module = dict[nn.Module, VllmModel]()


@runtime_checkable
class SupportsMultiModal(Protocol):
    """Retired feature marker; no execution implementation in P4."""

    supports_multimodal: ClassVar[bool] = False


@runtime_checkable
class SupportsMultiModalPruning(Protocol):
    """Retired feature marker; no execution implementation in P4."""

    supports_multimodal_pruning: ClassVar[bool] = False


def supports_multimodal(model: type[object] | object) -> bool:
    return False


def supports_multimodal_raw_input_only(model: type[object] | object) -> bool:
    return False


def requires_raw_input_tokens(model: type[object] | object) -> bool:
    return False


def supports_multimodal_encoder_tp_data(model: type[object] | object) -> bool:
    return False


def supports_multimodal_pruning(model: type[object] | object) -> bool:
    return False


@runtime_checkable
class SupportsScoreTemplate(Protocol):
    """Retired feature marker; no execution implementation in P4."""

    supports_score_template: ClassVar[bool] = False


def supports_score_template(model: type[object] | object) -> bool:
    return False


@runtime_checkable
class SupportsLoRA(Protocol):
    """Retired feature marker; no execution implementation in P4."""

    supports_lora: ClassVar[bool] = False


# We can't use runtime_checkable with ClassVar for issubclass checks
# so we need to treat the class as an instance and use isinstance instead
@runtime_checkable
class _SupportsLoRAType(Protocol):
    supports_lora: Literal[True]

    packed_modules_mapping: dict[str, list[str]]
    embedding_modules: dict[str, str]


def supports_lora(model: type[object] | object) -> bool:
    return False


def _supports_lora(model: type[object] | object) -> bool:
    return False


@runtime_checkable
class SupportsPP(Protocol):
    """The interface required for all models that support pipeline parallel."""

    supports_pp: ClassVar[Literal[True]] = True
    """
    A flag that indicates this model supports pipeline parallel.

    Note:
        There is no need to redefine this flag if this class is in the
        MRO of your model class.
    """

    def make_empty_intermediate_tensors(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        """Called when PP rank > 0 for profiling purposes."""
        ...

    def forward(
        self,
        input_ids: Tensor | None,
        positions: Tensor,
        *,
        intermediate_tensors: IntermediateTensors | None,
    ) -> IntermediateTensors | None:
        """
        Accept [`IntermediateTensors`][vllm.sequence.IntermediateTensors] when
        PP rank > 0.

        Return [`IntermediateTensors`][vllm.sequence.IntermediateTensors] only
        for the last PP rank.
        """
        ...


# We can't use runtime_checkable with ClassVar for issubclass checks
# so we need to treat the class as an instance and use isinstance instead
@runtime_checkable
class _SupportsPPType(Protocol):
    supports_pp: Literal[True]

    def make_empty_intermediate_tensors(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors: ...

    def forward(
        self,
        input_ids: Tensor | None,
        positions: Tensor,
        *,
        intermediate_tensors: IntermediateTensors | None,
    ) -> Tensor | IntermediateTensors: ...


@overload
def supports_pp(model: type[object]) -> TypeIs[type[SupportsPP]]: ...


@overload
def supports_pp(model: object) -> TypeIs[SupportsPP]: ...


def supports_pp(
    model: type[object] | object,
) -> bool | TypeIs[type[SupportsPP]] | TypeIs[SupportsPP]:
    supports_attributes = _supports_pp_attributes(model)
    supports_inspect = _supports_pp_inspect(model)

    if supports_attributes and not supports_inspect:
        logger.warning(
            "The model (%s) sets `supports_pp=True`, but does not accept "
            "`intermediate_tensors` in its `forward` method",
            model,
        )

    if not supports_attributes:
        pp_attrs = ("make_empty_intermediate_tensors",)
        missing_attrs = tuple(attr for attr in pp_attrs if not hasattr(model, attr))

        if getattr(model, "supports_pp", False):
            if missing_attrs:
                logger.warning(
                    "The model (%s) sets `supports_pp=True`, "
                    "but is missing PP-specific attributes: %s",
                    model,
                    missing_attrs,
                )
        else:
            if not missing_attrs:
                logger.warning(
                    "The model (%s) contains all PP-specific attributes, "
                    "but does not set `supports_pp=True`.",
                    model,
                )

    return supports_attributes and supports_inspect


def _supports_pp_attributes(model: type[object] | object) -> bool:
    if isinstance(model, type):
        return isinstance(model, _SupportsPPType)

    return isinstance(model, SupportsPP)


def _supports_pp_inspect(model: type[object] | object) -> bool:
    model_forward = getattr(model, "forward", None)
    if not callable(model_forward):
        return False

    return supports_kw(model_forward, "intermediate_tensors")


@runtime_checkable
class HasInnerState(Protocol):
    """The interface required for all models that has inner state."""

    has_inner_state: ClassVar[Literal[True]] = True
    """
        A flag that indicates this model has inner state.
        Models that has inner state usually need access to the scheduler_config
        for max_num_seqs, etc. True for e.g. both Mamba and Jamba.
    """


@overload
def has_inner_state(model: object) -> TypeIs[HasInnerState]: ...


@overload
def has_inner_state(model: type[object]) -> TypeIs[type[HasInnerState]]: ...


def has_inner_state(
    model: type[object] | object,
) -> TypeIs[type[HasInnerState]] | TypeIs[HasInnerState]:
    return getattr(model, "has_inner_state", False)


@runtime_checkable
class IsAttentionFree(Protocol):
    """The interface required for all models like Mamba that lack attention,
    but do have state whose size is constant wrt the number of tokens."""

    is_attention_free: ClassVar[Literal[True]] = True
    """
        A flag that indicates this model has no attention.
        Used for block manager and attention backend selection.
        True for Mamba but not Jamba.
    """


@overload
def is_attention_free(model: object) -> TypeIs[IsAttentionFree]: ...


@overload
def is_attention_free(model: type[object]) -> TypeIs[type[IsAttentionFree]]: ...


def is_attention_free(
    model: type[object] | object,
) -> TypeIs[type[IsAttentionFree]] | TypeIs[IsAttentionFree]:
    return getattr(model, "is_attention_free", False)


@runtime_checkable
class IsHybrid(Protocol):
    """The interface required for all models like Jamba that have both
    attention and mamba blocks, indicates that
    hf_config has 'layers_block_type'"""

    is_hybrid: ClassVar[Literal[True]] = True
    """
        A flag that indicates this model has both mamba and attention blocks
        , also indicates that the model's hf_config has 
        'layers_block_type' """

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[tuple[int, int], tuple[int, int, int]]:
        """Calculate shapes for Mamba's convolutional and state caches.

        Args:
            vllm_config: vLLM config

        Returns:
            Tuple containing:
            - conv_state_shape: Shape for convolutional state cache
            - temporal_state_shape: Shape for state space model cache
        """
        ...

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc, ...]:
        """Calculate copy-function callables for each Mamba state.

        Returns:
            A tuple of MambaStateCopyFunc callables that correspond, in order,
            to the Mamba states produced by the model. Each callable accepts
            (state, block_ids, cur_block_idx, num_accepted_tokens) and returns
            a MambaCopySpec describing the memory-copy parameters for prefix
            caching in align mode.
        """
        ...


@overload
def is_hybrid(model: object) -> TypeIs[IsHybrid]: ...


@overload
def is_hybrid(model: type[object]) -> TypeIs[type[IsHybrid]]: ...


def is_hybrid(
    model: type[object] | object,
) -> TypeIs[type[IsHybrid]] | TypeIs[IsHybrid]:
    return getattr(model, "is_hybrid", False)


@runtime_checkable
class MixtureOfExperts(Protocol):
    """
    Check if the model is a mixture of experts (MoE) model.
    """

    expert_weights: MutableSequence[Sequence[Tensor]]
    """
    Expert weights saved in this rank.

    The first dimension is the layer, and the second dimension is different
    parameters in the layer, e.g. up/down projection weights.
    """

    num_moe_layers: int
    """Number of MoE layers in this model."""

    num_expert_groups: int
    """Number of expert groups in this model."""

    num_logical_experts: int
    """Number of logical experts in this model."""

    num_physical_experts: int
    """Number of physical experts in this model."""

    num_local_physical_experts: int
    """Number of local physical experts in this model."""

    num_routed_experts: int
    """Number of routed experts in this model."""

    num_shared_experts: int
    """Number of shared experts in this model."""

    num_redundant_experts: int
    """Number of redundant experts in this model."""

    moe_layers: Iterable[nn.Module]
    """List of MoE layers in this model."""

    def set_eplb_state(
        self,
        expert_load_view: Tensor,
        logical_to_physical_map: Tensor,
        logical_replica_count: Tensor,
    ) -> None:
        """
        Register the EPLB state in the MoE model.

        Since these are views of the actual EPLB state, any changes made by
        the EPLB algorithm are automatically reflected in the model's behavior
        without requiring additional method calls to set new states.

        You should also collect model's `expert_weights` here instead of in
        the weight loader, since after initial weight loading, further
        processing like quantization may be applied to the weights.

        Args:
            expert_load_view: A view of the expert load metrics tensor.
            logical_to_physical_map: Mapping from logical to physical experts.
            logical_replica_count: Count of replicas for each logical expert.
        """
        for layer_idx, layer in enumerate(self.moe_layers):
            # Register the expert weights.
            self.expert_weights.append(layer.get_expert_weights())
            layer.set_eplb_state(
                moe_layer_idx=layer_idx,
                expert_load_view=expert_load_view,
                logical_to_physical_map=logical_to_physical_map,
                logical_replica_count=logical_replica_count,
            )

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None: ...


def is_mixture_of_experts(model: object) -> TypeIs[MixtureOfExperts]:
    return (
        isinstance(model, MixtureOfExperts) and getattr(model, "num_moe_layers", 0) > 0
    )


@runtime_checkable
class HasNoOps(Protocol):
    has_noops: ClassVar[Literal[True]] = True


@overload
def has_noops(model: object) -> TypeIs[HasNoOps]: ...


@overload
def has_noops(model: type[object]) -> TypeIs[type[HasNoOps]]: ...


def has_noops(
    model: type[object] | object,
) -> TypeIs[type[HasNoOps]] | TypeIs[HasNoOps]:
    return getattr(model, "has_noops", False)


@runtime_checkable
class SupportsMambaPrefixCaching(Protocol):
    """The interface for models whose mamba layers support prefix caching.

    This is currently experimental.
    """

    supports_mamba_prefix_caching: ClassVar[Literal[True]] = True


@overload
def supports_mamba_prefix_caching(
    model: object,
) -> TypeIs[SupportsMambaPrefixCaching]: ...


@overload
def supports_mamba_prefix_caching(
    model: type[object],
) -> TypeIs[type[SupportsMambaPrefixCaching]]: ...


def supports_mamba_prefix_caching(
    model: type[object] | object,
) -> TypeIs[type[SupportsMambaPrefixCaching]] | TypeIs[SupportsMambaPrefixCaching]:
    return getattr(model, "supports_mamba_prefix_caching", False)


@runtime_checkable
class SupportsCrossEncoding(Protocol):
    """The interface required for all models that support cross encoding."""

    score_type: ClassVar[ScoreType] = "cross-encoder"


@runtime_checkable
class SupportsLateInteraction(Protocol):
    """The interface required for all models that support late interaction.

    Late interaction models (like ColBERT) encode queries and documents
    separately into per-token embeddings, then compute similarity via
    MaxSim (max over document tokens, sum over query tokens).
    """

    score_type: ClassVar[ScoreType] = "late-interaction"


class SupportsQuant:
    """The interface required for all models that support quantization."""

    hf_to_vllm_mapper: ClassVar[WeightsMapper | None] = None
    packed_modules_mapping: ClassVar[dict[str, list[str]] | None] = None
    quant_config: QuantizationConfig | None = None

    def __new__(cls, *args, **kwargs) -> Self:
        instance = super().__new__(cls)

        # find config passed in arguments and attach it to model for general use
        instance.quant_config = cls._find_quant_config(*args, **kwargs)

        cls._maybe_apply_model_mapping(instance)

        return instance

    @staticmethod
    def _find_quant_config(*args, **kwargs) -> QuantizationConfig | None:
        """Find quant config passed through model constructor args"""
        from vllm.config import VllmConfig  # avoid circular import

        args_values = list(args) + list(kwargs.values())
        for arg in args_values:
            if isinstance(arg, VllmConfig):
                return arg.quant_config

            if isinstance(arg, QuantizationConfig):
                return arg

        return None

    def _maybe_apply_model_mapping(self):
        """Apply model mappings to config for proper config-model matching"""
        if self.quant_config is None:
            return
        if (hf_to_vllm_mapper := self.hf_to_vllm_mapper) is not None:
            self.quant_config.apply_vllm_mapper(hf_to_vllm_mapper)
        if self.packed_modules_mapping is not None:
            self.quant_config.packed_modules_mapping.update(self.packed_modules_mapping)


@runtime_checkable
class SupportsRealtime(Protocol):
    """Retired feature marker; no execution implementation in P4."""

    supports_realtime: ClassVar[bool] = False


def supports_realtime(model: type[object] | object) -> bool:
    return False


@runtime_checkable
class SupportsTranscription(Protocol):
    """Retired feature marker; no execution implementation in P4."""

    supports_transcription: ClassVar[bool] = False


def supports_transcription(model: type[object] | object) -> bool:
    return False


@runtime_checkable
class SupportsEagleBase(Protocol):
    """Base interface for models that support EAGLE-based speculative decoding."""

    has_own_lm_head: bool = False
    """
    A flag that indicates this model has trained its own lm_head.
    """

    has_own_embed_tokens: bool = False
    """
    A flag that indicates this model has trained its own input embeddings.
    """


@overload
def supports_any_eagle(model: type[object]) -> TypeIs[type[SupportsEagleBase]]: ...


@overload
def supports_any_eagle(model: object) -> TypeIs[SupportsEagleBase]: ...


def supports_any_eagle(
    model: type[object] | object,
) -> TypeIs[type[SupportsEagleBase]] | TypeIs[SupportsEagleBase]:
    """Check if model supports any EAGLE variant (1, 2, or 3)."""
    return supports_eagle(model) or supports_eagle3(model)


class EagleModelMixin:
    aux_hidden_state_layers: tuple[int, ...] = ()

    def _set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.aux_hidden_state_layers = layers

    def _maybe_add_hidden_state(
        self,
        aux_hidden_states: list[torch.Tensor],
        layer_idx: int,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
    ) -> list[torch.Tensor]:
        if layer_idx in self.aux_hidden_state_layers:
            value = hidden_states + residual if residual is not None else hidden_states
            aux_hidden_states.append(value)
        return aux_hidden_states


@runtime_checkable
class SupportsEagle(SupportsEagleBase, Protocol):
    """The interface required for models that support
    EAGLE-1 and EAGLE-2 speculative decoding."""

    supports_eagle: ClassVar[Literal[True]] = True
    """
    A flag that indicates this model supports EAGLE-1 and EAGLE-2 
    speculative decoding.

    Note:
        There is no need to redefine this flag if this class is in the
        MRO of your model class.
    """


@overload
def supports_eagle(model: type[object]) -> TypeIs[type[SupportsEagle]]: ...


@overload
def supports_eagle(model: object) -> TypeIs[SupportsEagle]: ...


def supports_eagle(
    model: type[object] | object,
) -> TypeIs[type[SupportsEagle]] | TypeIs[SupportsEagle]:
    return isinstance(model, SupportsEagle)


@runtime_checkable
class SupportsEagle3(SupportsEagleBase, Protocol):
    """The interface required for models that support
    EAGLE-3 speculative decoding."""

    supports_eagle3: ClassVar[Literal[True]] = True
    """
    A flag that indicates this model supports EAGLE-3 
    speculative decoding.

    Note:
        There is no need to redefine this flag if this class is in the
        MRO of your model class.
    """

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        """
        Set which layers should output auxiliary hidden states for EAGLE-3.

        Args:
            layers: Tuple of layer indices that should output auxiliary
                hidden states.
        """
        parent_ref = self
        if hasattr(self, "get_language_model"):
            parent_ref = self.get_language_model()
        elif hasattr(self, "language_model"):
            parent_ref = self.language_model
        assert hasattr(parent_ref, "model"), (
            "Model instance must have 'model' attribute to set number of layers"
        )
        assert isinstance(parent_ref.model, EagleModelMixin), (
            "Model instance must inherit from EagleModelMixin to set auxiliary layers"
        )
        parent_ref.model._set_aux_hidden_state_layers(layers)

    def get_eagle3_default_aux_hidden_state_layers(self) -> tuple[int, ...]:
        """
        Get the default layer indices that should output auxiliary hidden states
        for EAGLE-3 for this model. Models can override this method to provide
        different default layers based on their architecture, but it is encouraged
        to instead include the layer specification in the model's config if possible.

        Returns:
            Tuple of layer indices for auxiliary hidden state outputs.
        """
        parent_ref = self
        if hasattr(self, "get_language_model"):
            parent_ref = self.get_language_model()
        elif hasattr(self, "language_model"):
            parent_ref = self.language_model
        assert hasattr(parent_ref, "model"), (
            "Model instance must have 'model' attribute to get number of layers"
        )
        assert hasattr(parent_ref.model, "layers"), (
            "Model instance must have 'layers' attribute to get number of layers"
        )
        num_layers = len(parent_ref.model.layers)
        return (2, num_layers // 2, num_layers - 3)


@overload
def supports_eagle3(model: type[object]) -> TypeIs[type[SupportsEagle3]]: ...


@overload
def supports_eagle3(model: object) -> TypeIs[SupportsEagle3]: ...


def supports_eagle3(
    model: type[object] | object,
) -> TypeIs[type[SupportsEagle3]] | TypeIs[SupportsEagle3]:
    return isinstance(model, SupportsEagle3)


@runtime_checkable
class SupportsMRoPE(Protocol):
    """Retired feature marker; no execution implementation in P4."""

    supports_mrope: ClassVar[bool] = False


def supports_mrope(model: type[object] | object) -> bool:
    return False


@runtime_checkable
class SupportsXDRoPE(Protocol):
    """Retired feature marker; no execution implementation in P4."""

    supports_xdrope: ClassVar[bool] = False


def supports_xdrope(model: type[object] | object) -> bool:
    return False
