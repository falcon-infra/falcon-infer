# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contract for the fixed one-draft MTP input-ID fast path."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[3]
PATH = ROOT / "vllm/v1/worker/npu_model_runner.py"


def load_subject():
    tree = ast.parse(PATH.read_text(encoding="utf-8"))
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_prepare_input_ids"
    )
    module = ast.parse("from __future__ import annotations")
    module.body.extend(
        [
            ast.ClassDef(
                name="Base",
                bases=[],
                keywords=[],
                body=[
                    ast.parse(
                        "def _prepare_input_ids(self, *args):\n"
                        "    self.fallback = True"
                    ).body[0],
                ],
                decorator_list=[],
            ),
            ast.ClassDef(
                name="Subject",
                bases=[ast.Name(id="Base")],
                keywords=[],
                body=[method],
                decorator_list=[],
            ),
        ]
    )
    namespace = {"torch": torch, "np": np}
    exec(compile(ast.fix_missing_locations(module), str(PATH), "exec"), namespace)
    return namespace["Subject"]


def make_runner(subject_type, requests=4):
    subject = subject_type()
    previous = {f"r{i}": i for i in range(requests)}
    subject.input_batch = SimpleNamespace(
        num_reqs=requests,
        req_id_to_index=previous.copy(),
        prev_req_id_to_index=previous.copy(),
        prev_sampled_token_ids=torch.arange(requests, dtype=torch.int32).view(
            requests, 1
        )
        + 100,
    )
    subject._draft_token_ids = torch.arange(requests, dtype=torch.int64).view(
        requests, 1
    ) + 200
    subject.input_ids = SimpleNamespace(gpu=torch.full((2 * requests + 2,), -77))
    subject._fixed_mtp_metadata = (object(),)
    subject._fixed_decode_cu_num_tokens = np.arange(
        2, 2 * requests + 1, 2, dtype=np.int64
    )
    subject.use_async_scheduling = True
    subject.enable_prompt_embeds = False
    return subject


def scheduler(requests):
    return SimpleNamespace(
        scheduled_new_reqs=[],
        finished_req_ids=set(),
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
        scheduled_spec_decode_tokens={f"r{i}": [-1] for i in range(requests)},
    )


def test_fixed_mtp_interleave_writes_sample_then_draft_without_reallocation():
    subject = make_runner(load_subject())
    address = subject.input_ids.gpu.data_ptr()
    subject._prepare_input_ids(scheduler(4), 8, np.arange(2, 9, 2))
    assert subject.input_ids.gpu[:8].tolist() == [100, 200, 101, 201, 102, 202, 103, 203]
    assert subject.input_ids.gpu.data_ptr() == address
    assert not hasattr(subject, "fallback")


@pytest.mark.parametrize(
    "change",
    ["sync", "prompt_embeds", "new_request", "reorder", "wrong_width"],
)
def test_transition_delegates_before_fast_path_writes(change):
    subject = make_runner(load_subject())
    output = scheduler(4)
    if change == "sync":
        subject.use_async_scheduling = False
    elif change == "prompt_embeds":
        subject.enable_prompt_embeds = True
    elif change == "new_request":
        output.scheduled_new_reqs = [object()]
    elif change == "reorder":
        subject.input_batch.prev_req_id_to_index = {f"r{i}": 3 - i for i in range(4)}
    else:
        output.scheduled_spec_decode_tokens["r0"] = [-1, -1]
    before = subject.input_ids.gpu.clone()
    subject._prepare_input_ids(output, 8, np.arange(2, 9, 2))
    assert subject.fallback
    torch.testing.assert_close(subject.input_ids.gpu, before, rtol=0, atol=0)
