import torch
import torch_npu

from vllm.model_executor.layers.ascend.layernorm import (
    AscendGemmaRMSNorm,
    AscendRMSNorm,
)
from vllm.model_executor.layers.layernorm import RMSNormGated


class AscendRMSNorm310(AscendRMSNorm):
    def forward_npu(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            x, _, residual = torch_npu.npu_add_rms_norm(
                x, residual, self.weight, self.variance_epsilon
            )
            if self.bias is not None:
                x.add_(self.bias)
            return x, residual

        x, _ = torch_npu.npu_rms_norm(x, self.weight, self.variance_epsilon)
        if self.bias is not None:
            x.add_(self.bias)
        return x

    name = "RMSNorm"


class AscendGemmaRMSNorm310(AscendGemmaRMSNorm):
    def forward_npu(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            orig_dtype = residual.dtype
            x = x + residual.to(x.dtype)
            residual = x.to(orig_dtype)
            x, _ = torch_npu.npu_rms_norm(x, 1.0 + self.weight, self.variance_epsilon)
            return x, residual

        x, _ = torch_npu.npu_rms_norm(x, 1.0 + self.weight, self.variance_epsilon)
        return x

    name = "GemmaRMSNorm"


class AscendRMSNormGated310(RMSNormGated):
    def forward_npu(
        self,
        x: torch.Tensor,
        z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 310P should not depend on the Triton-gated layernorm path.
        # Reuse the upstream native implementation directly.
        return super().forward_native(x, z)

    name = "RMSNormGated"
