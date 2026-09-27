# SPDX-License-Identifier: Apache-2.0
"""Metadata-before-output ordering is also exercised by the host native contract suite."""
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

def test_worker_metadata_precedes_stock_scheduler_update():
    source = Path(__file__).resolve().parents[4] / "vllm/v1/core/sched/scheduler.py"
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Scheduler")
    fn = next(node for node in cls.body if getattr(node, "name", None) == "update_from_output")
    calls = []
    connector = MagicMock()
    connector.update_connector_worker_metadata.side_effect = lambda *_: calls.append("metadata")
    self = SimpleNamespace(connector=connector, requests={"active": SimpleNamespace(is_finished=lambda: False), "finished": SimpleNamespace(is_finished=lambda: True)})
    metadata = object()
    model_runner_output = SimpleNamespace(kv_connector_output=SimpleNamespace(kv_connector_worker_meta=metadata))
    # Execute the production prefix and inspect its position before stock output processing.
    prefix = ast.Module(body=fn.body[:2], type_ignores=[])
    exec(compile(prefix, str(source), "exec"), {"self": self, "model_runner_output": model_runner_output})
    assert calls == ["metadata"]
    connector.update_connector_worker_metadata.assert_called_once_with(metadata, {"active"})
