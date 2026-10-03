# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read-only Ascend environment summary, without upstream usage reporting."""

from collections import namedtuple
from importlib.metadata import PackageNotFoundError, version
import json
import os
import platform
import sys

EnvironmentInfo = namedtuple(
    "EnvironmentInfo", "python platform machine torch torch_npu transformers cann_home"
)


def get_env_info():
    """Read package metadata only; this does not initialize NPU or prove ABI."""

    def installed(name):
        try:
            return version(name)
        except PackageNotFoundError:
            return "not installed"

    return EnvironmentInfo(
        sys.version,
        platform.platform(),
        platform.machine(),
        installed("torch"),
        installed("torch-npu"),
        installed("transformers"),
        os.getenv("ASCEND_HOME_PATH"),
    )


def main():
    print(json.dumps(get_env_info()._asdict(), indent=2))


if __name__ == "__main__":
    main()
