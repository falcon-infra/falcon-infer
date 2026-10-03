# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Retired wire-slot annotations, not media processors or model support.

The paired scheduler protocol keeps its empty legacy slots during P4. Ingress
and Request reject nonempty media, adapter, embedding and pooling payloads.
No encoder, processor, registry, downloader or media serializer is installed.
"""

from typing import Any, TypeAlias

MultiModalFeatureSpec: TypeAlias = Any
MultiModalDataDict: TypeAlias = dict[str, Any]
MultiModalUUIDDict: TypeAlias = dict[str, Any]
MultiModalInputs: TypeAlias = dict[str, Any]
MultiModalEncDecInputs: TypeAlias = dict[str, Any]
NestedTensors: TypeAlias = Any
BatchedTensorInputs: TypeAlias = dict[str, Any]
