# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
import torch
import torch_npu


class BaseDeviceAdaptor:
    @classmethod
    def reshape_and_cache(cls, key, value, key_cache, value_cache, slot_mapping):
        torch_npu._npu_reshape_and_cache(
            key=key,
            value=value,
            key_cache=key_cache,
            value_cache=value_cache,
            slot_indices=slot_mapping,
        )

    @staticmethod
    def npu_moe_init_routing(
        hidden_states,
        topk_ids,
        *,
        scale=None,
        active_num: int,
        expert_num: int,
        expert_tokens_num_type: int = 1,
        expert_tokens_num_flag: bool = True,
        active_expert_range=None,
        quant_mode: int = -1,
    ):
        return torch.ops._C_ascend.npu_moe_init_routing_custom(
            hidden_states,
            topk_ids,
            scale=scale,
            active_num=active_num,
            expert_num=expert_num,
            expert_tokens_num_type=expert_tokens_num_type,
            expert_tokens_num_flag=expert_tokens_num_flag,
            active_expert_range=active_expert_range,
            quant_mode=quant_mode,
        )

    @staticmethod
    def maybe_normalize_mxfp_scale_layout(
        scale: torch.Tensor | None,
    ) -> torch.Tensor | None:
        return scale

    @staticmethod
    def moe_gating_top_k(
        x: torch.Tensor,
        *,
        k: int,
        k_group: int,
        group_count: int,
        group_select_mode: int,
        renorm: int,
        norm_type: int,
        out_flag: bool,
        routed_scaling_factor: float = 1.0,
        eps: float = 1e-20,
        bias_opt: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        topk_weights, topk_ids, out = torch.ops._C_ascend.moe_gating_top_k(
            x,
            k=k,
            k_group=k_group,
            group_count=group_count,
            group_select_mode=group_select_mode,
            renorm=renorm,
            norm_type=norm_type,
            out_flag=out_flag,
            routed_scaling_factor=routed_scaling_factor,
            eps=eps,
            bias_opt=bias_opt,
        )
        return topk_weights, topk_ids.to(torch.int32), out

    @staticmethod
    def npu_dynamic_quant(
        hidden_states: torch.Tensor,
        dynamic_scale: torch.Tensor | None = None,
        *,
        act_quant_type=torch.float8_e4m3fn,
        use_mxfp_quant: bool = False,
    ):
        if use_mxfp_quant:
            raise RuntimeError("MXFP8 MoE quantization is only supported on Ascend A5.")

        if dynamic_scale is None:
            return torch_npu.npu_dynamic_quant(hidden_states)

        return hidden_states, dynamic_scale

    @staticmethod
    def npu_grouped_matmul_swiglu_quant(
        *,
        x: torch.Tensor,
        weight: torch.Tensor,
        group_list: torch.Tensor,
        weight_scale: torch.Tensor,
        x_scale: torch.Tensor,
        bias=None,
        use_mxfp_quant: bool = False,
    ):
        if use_mxfp_quant:
            raise RuntimeError("MXFP8 MoE quantization is only supported on Ascend A5.")

        return torch_npu.npu_grouped_matmul_swiglu_quant(
            x=x,
            weight=weight,
            bias=bias,
            group_list=group_list,
            weight_scale=weight_scale,
            x_scale=x_scale,
        )

    @staticmethod
    def get_quant_gmm2_kwargs(
        *,
        input_dtype: torch.dtype,
        act_quant_type,
        weight_quant_type,
        scale_type,
        per_token_scale_type,
        use_bf16: bool = True,
        use_mxfp_quant: bool = False,
    ) -> dict:
        if use_mxfp_quant:
            raise RuntimeError("MXFP8 MoE quantization is only supported on Ascend A5.")

        return {
            "output_dtype": input_dtype
            if input_dtype in [torch.bfloat16, torch.float16]
            else torch.bfloat16,
        }

    @classmethod
    def npu_grouped_matmul_gmm2(
        cls,
        *,
        hidden_states: torch.Tensor,
        weight: list[torch.Tensor] | torch.Tensor,
        weight_scale: list[torch.Tensor] | torch.Tensor,
        per_token_scale: torch.Tensor,
        group_list: torch.Tensor,
        group_list_type: int,
        input_dtype: torch.dtype,
        act_quant_type,
        weight_quant_type,
        scale_type,
        per_token_scale_type,
        use_bf16: bool = True,
        use_mxfp_quant: bool = False,
        bias=None,
        fallback_output_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if use_mxfp_quant:
            raise RuntimeError("MXFP8 MoE quantization is only supported on Ascend A5.")

        if fallback_output_dtype is None:
            fallback_output_dtype = (
                weight_scale[0].dtype
                if isinstance(weight_scale, list)
                else weight_scale.dtype
            )
        return torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=weight,
            scale=weight_scale,
            bias=bias,
            per_token_scale=[per_token_scale],
            split_item=2,
            group_list_type=group_list_type,
            group_type=0,
            group_list=group_list,
            output_dtype=fallback_output_dtype,
        )[0]

    @staticmethod
    def mla_cache_load(
        cache_kv_c, cache_k_pe, block_table, context_seq_len_npu, seq_starts, key, value
    ):
        torch_npu.atb.npu_paged_cache_load(
            cache_kv_c,
            cache_k_pe,
            block_table,
            context_seq_len_npu,
            seq_starts=seq_starts,
            key=key,
            value=value,
        )


def get_device_adaptor() -> type["BaseDeviceAdaptor"]:
    return BaseDeviceAdaptor


DeviceOperator: type["BaseDeviceAdaptor"] = get_device_adaptor()
