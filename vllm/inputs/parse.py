# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from .data import ProcessorInputs, SingletonInputs


def split_enc_dec_inputs(
    inputs: ProcessorInputs,
) -> tuple[SingletonInputs | None, SingletonInputs]:
    pass  # Unsupported P4 branch removed.

    return None, inputs
