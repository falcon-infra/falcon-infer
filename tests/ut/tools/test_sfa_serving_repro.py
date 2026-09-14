# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for the native online TP4 x DP2 SFA reproducer."""

import importlib.util
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.error import HTTPError

import pytest


@pytest.fixture
def repro(monkeypatch):
    pytest.importorskip("psutil")
    tools = Path(__file__).resolve().parents[3] / "tools"
    monkeypatch.syspath_prepend(str(tools))
    spec = importlib.util.spec_from_file_location(
        "tested_sfa_serving_repro", tools / "sfa_serving_repro.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def args(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    return SimpleNamespace(
        model=str(tmp_path),
        devices="0,1,2,3,4,5,6,7",
        prompt_tokens=30000,
        output_tokens=512,
        port=9000,
        diagnose=False,
    )


def argument(command, name):
    return command[command.index(name) + 1]


def test_server_reproduces_native_serving_topology(repro, args):
    staged = repro.server_command(args, "staged")
    full = repro.server_command(args, "full")
    assert staged == full
    assert argument(full, "--tensor-parallel-size") == "4"
    assert argument(full, "--data-parallel-size") == "2"
    assert argument(full, "--data-parallel-size-local") == "2"
    assert "--enable-expert-parallel" in full
    assert argument(full, "--worker-cls") == (
        "vllm.v1.worker.sfa_benchmark_worker.SFABenchmarkWorker"
    )
    assert "--no-async-scheduling" not in full
    assert "--async-scheduling" not in full
    assert argument(full, "--hf-overrides") == json.dumps({"num_hidden_layers": 8})
    assert argument(full, "--max-num-seqs") == "16"
    assert argument(full, "--max-num-batched-tokens") == "4096"
    assert argument(full, "--max-model-len") == "140000"
    assert json.loads(argument(full, "--additional-config"))["recompute_scheduler_enable"]
    assert "enforce_eager" not in json.loads(argument(full, "--speculative-config"))
    transfer = json.loads(argument(full, "--kv-transfer-config"))
    assert transfer["kv_connector_module_path"] == "lmcache.integration.vllm.lmcache_connector_v1"

    staged_env = repro.serving_environment("staged", args.devices, diagnose=False)
    full_env = repro.serving_environment("full", args.devices, diagnose=False)
    assert {
        name for name in full_env if full_env[name] != staged_env[name]
    } == {"VLLM_ASCEND_SFA_FULL_GRAPH"}
    assert full_env["VLLM_ASCEND_SFA_STAGED_GRAPH_CAPTURE_SIZES"] == "4,8,12,16"
    assert full_env["HCCL_OP_EXPANSION_MODE"] == "AIV"
    assert full_env["PD_SERVING_PERF"] == "1"


@pytest.mark.parametrize("mode", ["staged", "full"])
@pytest.mark.parametrize("diagnose", [False, True])
def test_serving_cold_compact_policy_is_shared_and_strict(repro, args, mode, diagnose):
    environment = repro.serving_environment(mode, args.devices, diagnose=diagnose)
    assert environment["LMCACHE_ENABLE_DSA_COLD_COMPACT_LOAD"] == "true"
    assert environment["LMCACHE_ENABLE_SHARED_CPU_CACHE"] == "true"
    assert environment["LMCACHE_SHARED_CPU_CACHE_STRICT"] == "true"
    assert json.loads(environment["LMCACHE_EXTRA_CONFIG"]) == {
        "save_only_first_rank": True
    }


def test_serving_cache_policy_does_not_change_offline_benchmark(repro, args):
    before = repro.benchmark_environment("full", args.devices)
    repro.serving_environment("full", args.devices, diagnose=True)
    after = repro.benchmark_environment("full", args.devices)
    assert after == before
    assert after["LMCACHE_ENABLE_SHARED_CPU_CACHE"] == "false"
    assert json.loads(after["LMCACHE_EXTRA_CONFIG"])["save_only_first_rank"] is False


def test_client_is_one_request_without_hidden_probe_or_warmup(repro, args, tmp_path):
    command = repro.client_command(args, tmp_path)
    assert argument(command, "--num-prompts") == "1"
    assert argument(command, "--max-concurrency") == "1"
    assert argument(command, "--header") == "X-data-parallel-rank=0"
    assert argument(command, "--ready-check-timeout-sec") == "0"
    assert argument(command, "--random-input-len") == "30000"
    assert argument(command, "--random-output-len") == "512"
    assert argument(command, "--percentile-metrics") == "ttft,tpot,itl,e2el"
    assert "--ignore-eos" in command and "--save-detailed" in command
    assert "--num-warmups" not in command


def test_comparison_requires_paired_token_lengths_and_reports_regression(repro):
    staged = {
        "input_lens": [30000],
        "output_lens": [512],
        "mean_ttft_ms": 100,
        "mean_tpot_ms": 80,
        "mean_itl_ms": 80,
        "mean_e2el_ms": 41000,
        "generated_texts": ["same dummy output"],
    }
    full = dict(staged, mean_tpot_ms=120, mean_itl_ms=120, mean_e2el_ms=61500)
    result = repro.compare_results(staged, full)
    assert result["tpot_reduction_percent"] == -50
    assert result["decode_speedup"] == pytest.approx(2 / 3)
    assert result["outputs_equal"]
    with pytest.raises(ValueError, match="identical token lengths"):
        repro.compare_results(staged, dict(full, output_lens=[511]))


@pytest.mark.parametrize(
    "change,error",
    [
        ({"devices": "0,1,2,3"}, "eight"),
        ({"prompt_tokens": 4096}, "prompt_tokens"),
        ({"output_tokens": 1}, "output_tokens"),
        ({"port": 0}, "Port"),
        ({"model": "missing"}, "Missing model"),
    ],
)
def test_invalid_reproduction_fails_before_launch(repro, args, change, error):
    for name, value in change.items():
        setattr(args, name, value)
    with pytest.raises((ValueError, FileNotFoundError), match=error):
        repro.validate_args(args)


def test_diagnostics_disable_other_recorder_and_serialize_prompt(repro, args, monkeypatch):
    assert repro.serving_environment("full", args.devices, diagnose=True)["PD_SERVING_PERF"] == "0"
    captured = []
    monkeypatch.setattr(
        repro,
        "request_json",
        lambda url, body, timeout: captured.append((url, body, timeout)) or {},
    )
    repro.timing_rpc(args, "benchmark_start_decode_timing")
    assert captured[0][1]["args"] == ["30000"]


@pytest.mark.parametrize("with_server_log", [False, True])
def test_rpc_error_preserves_http_body_and_worker_trace(
    repro, args, monkeypatch, tmp_path, with_server_log
):
    error = HTTPError(
        "http://127.0.0.1:9000/collective_rpc",
        500,
        "Internal Server Error",
        {},
        io.BytesIO(b'{"detail":"Worker failed during diagnostic setup"}'),
    )
    opener = SimpleNamespace(open=Mock(side_effect=error))
    monkeypatch.setattr(
        repro.urllib.request,
        "build_opener",
        Mock(return_value=opener),
    )
    log_path = tmp_path / "server.log"
    log_path.write_text(
        "old line\n" * 10000 + "RuntimeError: actual worker exception\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError) as caught:
        repro.timing_rpc(
            args,
            "benchmark_start_decode_timing",
            server_log=log_path if with_server_log else None,
        )
    assert caught.value.__cause__ is error
    message = str(caught.value)
    assert "benchmark_start_decode_timing failed: HTTP 500" in message
    assert "Worker failed during diagnostic setup" in message
    if with_server_log:
        assert str(log_path) in message
        assert "RuntimeError: actual worker exception" in message
        assert len(message) < 8192
    request = opener.open.call_args.args[0]
    assert json.loads(request.data)["args"] == ["30000"]


def test_unreadable_server_log_does_not_hide_rpc_error(repro, args, monkeypatch, tmp_path):
    error = HTTPError(
        "http://127.0.0.1:9000/collective_rpc",
        500,
        "Internal Server Error",
        {},
        io.BytesIO(b"failed"),
    )
    monkeypatch.setattr(repro, "request_json", Mock(side_effect=error))
    with pytest.raises(RuntimeError, match="HTTP 500") as caught:
        repro.timing_rpc(
            args,
            "benchmark_stop_decode_timing",
            server_log=tmp_path / "missing.log",
        )
    assert "Unable to read" in str(caught.value)


def test_timing_summary_exposes_actual_async_policy(repro, capsys):
    repro.print_timing(
        "full",
        {"results": [{"async_scheduling": True}, {"async_scheduling": True}]},
    )
    assert "full async_workers=2/2" in capsys.readouterr().out


def test_diagnostic_summary_separates_dp_and_failure_agreement(repro, capsys):
    metric = {"count": 2, "total_ms": 6}
    response = {
        "results": [
            {
                "decode_steps": 2,
                "stages": {
                    "dp.batch_sync": {"wall": metric},
                    "full.prepare_agreement": {"wall": metric},
                },
            }
        ]
    }
    repro.print_timing("full", response)
    output = capsys.readouterr().out
    assert "full dp.batch_sync ms/forward mean=3.000" in output
    assert "full full.prepare_agreement ms/forward mean=3.000" in output
