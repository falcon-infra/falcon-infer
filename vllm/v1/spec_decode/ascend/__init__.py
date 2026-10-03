# SPDX-License-Identifier: Apache-2.0
"""GLM same-checkpoint MTP, sharing the native Eagle proposer machinery."""

from vllm.v1.spec_decode.ascend.eagle_proposer import AscendEagleProposer


def get_spec_decode_method(method, vllm_config, device, runner):
    if method != "mtp":
        raise ValueError("Ascend P4 supports only GLM-5.2 MTP")
    return AscendEagleProposer(vllm_config, device, runner)
