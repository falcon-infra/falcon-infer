# P4：Ascend910B3 / GLM-5.2 原生文本推理

> 本页为历史 P4 记录，不是当前安装入口。P6 请执行
> [基线一致的安装与验证指南](baseline-validation.md)，不要混用下文的 P4 版本与旧目录。

P4 是源码整改候选版，版本 `0.18.0+ascend.p4`，须配对
`lmcache==0.4.3+ascend.p4`。未在源码工作站验证 CANN 编译、ABI 或模型运行。
冻结输入为 `p3-frozen-20261003`，此前阶段分支不变。

## 支持边界

仅保留 910B3、GLM-5.2 原生文本生成、同 checkpoint 的 MTP、DSA 双组、
CPU KV 卸载/共享、跨实例缓存、2P2D、RemoteFill、checkpoint 和恢复链路。
离线 `vllm.LLM.generate` 以及在线 chat/completions（含流式、reasoning/tool-call 文本协议）保留。
CPU 是主机执行/存储资源，不是推理设备后端。

删除其他厂商后端、310P/A3/A5 专用实现、其他模型、多模态、LoRA、pooling、训练、
其他推测算法、GPU 权重卸载及 Responses/Harmony 服务。保留的 DeepSeek/Eagle 命名
用于 GLM 的共享实现，不代表开放对应模型。少量历史 DTO/枚举仅用于兼容序列化或明确拒绝旧请求。
用户配置不能重新启用被删除的能力；不支持的选项应在入口失败。
其他模型的配置改写、pooling 路由、Mamba 状态执行与 causal-conv1d 算子也已删除。
共用 CANN/CATLASS 材料维持审核过的固定版本；材料中的通用定义不扩大 P4 的硬件范围。

模型入口校验 `glm_moe_dsa` / `GlmMoeDsaForCausalLM`；内部 MTP 使用
`deepseek_mtp` / `DeepSeekMTPModel`。架构名不是 GLM-5.2 版本认证：
内网仍必须审核 checkpoint 标识、config/量化描述和权重索引哈希，不自动接受 GLM-5.3。
硬件检查要求具体设备名 `Ascend910B3`，不把不明确的 `Ascend910B` 当作已核验型号。

## 构建与开发态安装

复用 Python 3.11/aarch64、CANN 8.5.1、torch 2.9.0、torch-npu 2.9.0.post2、
transformers 5.2.0、triton-ascend 3.2.0.dev20260322。不得由 pip 隐式升级。
`p1_dev.py` 保留原文件名，已更新为 P4 的版本、材料和制品检查。

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.1
source "$ASCEND_HOME_PATH/set_env.sh"
export SOC_VERSION=ascend910b3
export VLLM_TARGET_DEVICE=ascend
python -B p1_dev.py doctor
python -B p1_dev.py editable --isolated-env --output /tmp/p4-vllm-editable-new
python -B p1_dev.py verify --mode editable
```

输出目录必须尚不存在。仅在独立测试容器执行安装；不要覆盖正在提供 P3 服务的环境。
P4 删除了 native LoRA 绑定并改变文件布局，必须重建 native 扩展并重新执行 strict editable 安装，
不能只切分支或复用 P3 的 `.so`/链接树。四个节点的每个服务容器都要安装两仓。

LMCache 必须通过正式 `LMCacheConnectorV1` 配置加载，不使用 `lmcache_ascend.*` module path；
未配置缓存连接器时不应导入 LMCache。C8 保持关闭：`enable_sparse_c8=false`，KV dtype 为 auto/bfloat16/float16。
P/D 路由、MTP 数量、DSA 双组、恢复策略和请求长度采用冻结 P3 的已验证配置，不另行猜测。

## 检查与验收

源码检查：`python -B tools/check_p4_profile.py`、`python -B tools/check_npu_native.py`。
内网安装后，从仓库外执行 `tools/check_npu_bootstrap.py --inspect-glm --check-lmcache --output <新目录>`。
另执行 `tools/validate_npu_native.py --spawn --output <新文件>` 检查无缓存冷导入；
可在空闲测试卡显式添加 `--device-smoke`。这些检查不加载模型，不能代替推理验收。

必须补齐：普通 wheel、sdist 重建、离线生成、无缓存在线服务、配对缓存 2P2D
TP8/DP2 和 TP4/DP4、DSA/MTP、并发/接受率、RemoteFill/共享 CPU cache、
checkpoint 抢占恢复、故障与长稳回归，以及旧设备/模型/多模态/LoRA 的负向测试。
逐节点归档实际 Git SHA、安装来源、依赖、制品哈希、启动配置、服务日志和请求结果。
