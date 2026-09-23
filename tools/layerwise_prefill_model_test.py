# SPDX-License-Identifier: Apache-2.0
"""NPU-only smoke test for the native layerwise-prefill owner path.

This script intentionally uses the formal native ``LMCacheConnectorV1`` name and
does not import a plugin module. It is a smoke test, not serving acceptance.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("p", "d"), default="p")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--devices", default="0")
    parser.add_argument("--prompt-tokens", type=int, default=10_000)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    return parser.parse_args()


def install_environment(args: argparse.Namespace) -> None:
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = args.devices
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE"] = (
        "true" if args.mode == "p" else "false"
    )
    os.environ["VLLM_ASCEND_DSA_UNBUNDLE"] = "1"
    os.environ["VLLM_ASCEND_DSA_TWO_GROUPS"] = "1"
    os.environ["VLLM_ASCEND_DSA_SHARED_POOL"] = "1"
    os.environ["VLLM_ASCEND_DSA_DISABLE_INDEX_LMCACHE"] = "0"
    os.environ["VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE"] = "0"
    os.environ["LMCACHE_USE_LAYERWISE"] = "true"
    os.environ["LMCACHE_ENABLE_SPARSE_ATTENTION"] = "true"
    os.environ["LMCACHE_DSA_TWO_GROUPS"] = "true"


def main() -> int:
    args = parse_args()
    if args.prompt_tokens <= 0 or args.max_num_batched_tokens <= 0:
        raise ValueError("prompt and batch token limits must be positive")
    if not args.prompt_file.is_file():
        raise FileNotFoundError(args.prompt_file)
    install_environment(args)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompt = args.prompt_file.read_text(encoding="utf-8")
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    token_ids = token_ids[: args.prompt_tokens]
    if not token_ids:
        raise ValueError("prompt file produced no token IDs")

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=len(args.devices.split(",")),
        pipeline_parallel_size=1,
        max_model_len=max(16_384, len(token_ids) + args.max_num_batched_tokens),
        max_num_seqs=1,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        enforce_eager=True,
        kv_transfer_config={
            "kv_connector": "LMCacheConnectorV1",
            "kv_role": "kv_producer",
        },
    )
    try:
        result = llm.generate(
            {"prompt_token_ids": token_ids},
            SamplingParams(temperature=0, max_tokens=1),
            use_tqdm=False,
        )[0]
        output_ids = list(result.outputs[0].token_ids)
        if not output_ids:
            raise RuntimeError("native layerwise smoke test returned no token")
        print(
            json.dumps(
                {
                    "mode": args.mode,
                    "prompt_tokens": len(token_ids),
                    "output_token_ids": output_ids,
                    "qualified": False,
                },
                sort_keys=True,
            )
        )
    finally:
        llm.llm_engine.engine_core.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
