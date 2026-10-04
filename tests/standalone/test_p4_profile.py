# SPDX-License-Identifier: Apache-2.0
"""Exercise the public P4 boundary without torch/CANN imports."""

import ast
from pathlib import Path
import re
import runpy
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[2]
api = NS(**runpy.run_path(str(ROOT / "vllm/inference_profile.py")))


def target():
    return NS(
        model="/models/approved-glm52",
        hf_config=NS(model_type="glm_moe_dsa", architectures=["GlmMoeDsaForCausalLM"]),
    )


@pytest.mark.parametrize("name", ["Ascend910B3", "ascend_910b3", "Ascend 910B3"])
def test_only_910b3_positive(name):
    api.validate_device_name(name)


@pytest.mark.parametrize(
    "name",
    ["Ascend910B", "Ascend910B2", "Ascend910_9391", "Ascend310P3", "cuda", "cpu", ""],
)
def test_other_devices_rejected(name):
    with pytest.raises(RuntimeError):
        api.validate_device_name(name)


def test_target_and_internal_mtp_metadata():
    api.validate_model_metadata(target().hf_config)
    draft = {"model_type": "deepseek_mtp", "architectures": ["DeepSeekMTPModel"]}
    api.validate_model_metadata(draft, draft=True)
    with pytest.raises(ValueError):
        api.validate_model_metadata(draft)
    api.validate_speculation("mtp", target().model, target())


@pytest.mark.parametrize(
    "arch",
    [
        "LlamaForCausalLM",
        "Qwen2ForCausalLM",
        "DeepseekV3ForCausalLM",
        "GLM4VForConditionalGeneration",
    ],
)
def test_other_architectures_rejected(arch):
    with pytest.raises(ValueError):
        api.validate_model_metadata(
            {"model_type": "glm_moe_dsa", "architectures": [arch]}
        )


@pytest.mark.parametrize(
    "key", ["vision_config", "audio_config", "text_config", "encoder_config"]
)
def test_nested_model_rejected(key):
    model = target().hf_config
    setattr(model, key, {})
    with pytest.raises(ValueError):
        api.validate_model_metadata(model)


@pytest.mark.parametrize(
    "feature", ["lora_config", "weight_transfer_config", "ec_transfer_config"]
)
def test_removed_runtime_config_rejected(feature):
    with pytest.raises(ValueError):
        api.validate_runtime_features(NS(**{feature: object()}))


def test_c8_pooling_and_other_draft_rejected():
    with pytest.raises(ValueError):
        api.validate_runtime_features(NS(cache_config=NS(cache_dtype="fp8")))
    with pytest.raises(ValueError):
        api.validate_model_options(
            runner="pooling", convert="auto", model_impl="auto", quantization=None
        )
    with pytest.raises(ValueError):
        api.validate_speculation("mtp", "/models/other", target())
    with pytest.raises(ValueError):
        api.validate_speculation("ngram", None, target())
    with pytest.raises(ValueError, match="Mamba"):
        api.validate_runtime_features(
            NS(cache_config=NS(cache_dtype="auto", mamba_cache_mode="align"))
        )
    api.validate_runtime_features(
        NS(cache_config=NS(cache_dtype="auto", mamba_cache_mode="none"))
    )


def test_native_connectors_and_no_legacy_import():
    for name, (module, _) in api.NATIVE_CONNECTORS.items():
        api.validate_connector(name, module)
        assert (ROOT / (module.replace(".", "/") + ".py")).is_file()
    with pytest.raises(ValueError):
        api.validate_connector(
            "LMCacheConnectorV1",
            "lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1",
        )


@pytest.mark.parametrize(
    "prompt",
    [
        b"embedding",
        {"multi_modal_data": {}},
        {"prompt_embeds": []},
        {"type": "multimodal"},
        {"encoder_prompt": "x"},
    ],
)
def test_media_and_embedding_prompt_rejected(prompt):
    with pytest.raises(ValueError):
        api.validate_text_prompt(prompt)


def test_text_and_tokens_retained():
    for prompt in (
        "hello",
        {"prompt": "hello"},
        {"prompt_token_ids": [1, 2]},
        {"type": "token", "prompt_token_ids": [1]},
    ):
        api.validate_text_prompt(prompt)


@pytest.mark.parametrize(
    "file, name",
    [
        ("lora", "LoRAConfig"),
        ("pooler", "PoolerConfig"),
        ("multimodal", "MultiModalConfig"),
    ],
)
def test_retired_config_initialization_fails_without_feature_implementation(file, name):
    """Execute the production guard without importing torch or the package."""
    tree = ast.parse((ROOT / f"vllm/config/{file}.py").read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == name
    )
    guard = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "__post_init__"
    )
    namespace = {}
    exec(
        compile(
            ast.Module(body=[guard], type_ignores=[]), "<retired config guard>", "exec"
        ),
        namespace,
    )
    with pytest.raises(ValueError, match="Ascend P4 does not support"):
        namespace["__post_init__"](NS())


def test_only_internal_mtp_converter_registered():
    tree = ast.parse(
        (ROOT / "vllm/transformers_utils/model_arch_config_convertor.py").read_text()
    )
    classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assert classes == {
        "ModelArchConfigConvertorBase",
        "DeepSeekMTPModelArchConfigConvertor",
    }
    mapping = next(
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "MODEL_ARCH_CONFIG_CONVERTORS"
            for target in node.targets
        )
    )
    assert [ast.literal_eval(key) for key in mapping.keys] == ["deepseek_mtp"]


def test_removed_cli_groups_and_native_bindings_are_absent():
    cli = (ROOT / "vllm/engine/arg_utils.py").read_text()
    frontend = (ROOT / "vllm/entrypoints/openai/cli_args.py").read_text()
    for flag in (
        "--enable-lora",
        "--pooler-config",
        "--limit-mm-per-prompt",
        "--enable-mm-embeds",
    ):
        assert flag not in cli
    assert "LoRAParserAction" not in frontend
    assert "lora_modules:" not in frontend
    for relative in (
        "csrc/torch_binding.cpp",
        "csrc/torch_binding_meta.cpp",
    ):
        content = (ROOT / relative).read_text()
        for symbol in (
            "npu_gemma_rms_norm",
            "bgmv_expand",
            "bgmv_shrink",
            "sgmv_expand",
            "sgmv_shrink",
            "npu_causal_conv1d_custom",
        ):
            assert symbol not in content


def test_native_operator_schemas_implementations_and_meta_stay_consistent():
    """Detect dangling bindings before CANN build or torch extension import."""
    bindings = (ROOT / "csrc/torch_binding.cpp").read_text()
    meta = (ROOT / "csrc/torch_binding_meta.cpp").read_text()
    schemas = re.findall(r'ops\.def\(\s*"(\w+)\(', bindings)
    implementations = re.findall(r'ops\.impl\(\s*"(\w+)"', bindings)
    meta_implementations = re.findall(r'ops\.impl\(\s*"(\w+)"', meta)
    assert schemas and len(schemas) == len(set(schemas))
    assert len(implementations) == len(set(implementations))
    assert len(meta_implementations) == len(set(meta_implementations))
    assert set(schemas) == set(implementations)
    assert set(meta_implementations) <= set(schemas)
    # DSA and GLM MTP production operators must not disappear with old models.
    assert {
        "npu_dsa_prepare_sparse_indices_staged_",
        "npu_sparse_flash_attention",
        "npu_copy_and_expand_eagle_inputs",
    } <= set(schemas)


def test_no_other_model_config_rewriters_or_pooling_execution():
    cfg = runpy.run_path(str(ROOT / "vllm/model_executor/models/config.py"))
    assert cfg["MODELS_CONFIG_MAP"] == {}
    assert not (ROOT / "vllm/v1/pool/late_interaction.py").exists()
    assert not (ROOT / "vllm/v1/worker/mamba_utils.py").exists()
    assert not any(p.is_file() for p in (ROOT / "csrc/causal_conv1d").rglob("*"))
    code = ast.parse((ROOT / "vllm/v1/worker/input_batch.py").read_text())
    assert not {"get_pooling_metadata", "make_lora_inputs"} & {
        n.name for n in ast.walk(code) if isinstance(n, ast.FunctionDef)
    }
