# 原生目录整改与安装

本分支从 `p4-frozen-20261004` 派生，只调整仓库目录、构建入口、测试与文档。
运行时 Python 和原生实现的冻结指纹由 `tools/check_native_layout.py` 校验。
冻结 P4 是回退点，不代表完整验收矩阵已经通过。

## 目录职责

| 路径 | 职责 |
| --- | --- |
| `vllm/` | 唯一 Python 产品包，模型、NPU 平台、DSA/MTP 和缓存桥接 |
| `csrc/` | NPU 内核、torch 绑定、CANN 自定义算子与固定 CATLASS 材料 |
| `cmake/` | 主扩展构建模块，由根 `CMakeLists.txt` 引入 |
| `tests/` | standalone 主机契约、UT 与 E2E；不是全部都能无设备执行 |
| `tools/` | 校验、诊断、运维辅助工具 |
| `examples/disaggregated_prefill_v1/` | 原 2P2D proxy 及配套诊断示例 |

`csrc/CMakeLists.txt` 仍是 CANN 自定义算子工程，不能代替根 CMake 入口。
`vllm._ascend_C`、`libvllm_ascend_kernels.so`、`_cann_ops_custom/vendors/vllm-ascend/`
以及 `VLLM_ASCEND_*` 环境变量名称不变。不要把包内的 `ascend` 功能目录继续扁平化。

## 内网安装

使用独立验证容器及新 checkout，保留冻结 P4 环境。配对安装
`vllm==0.18.0+ascend.layout1` 与 `lmcache==0.4.3+ascend.layout1`。
沿用 Python 3.11/aarch64、CANN 8.5.1、torch 2.9.0、torch-npu 2.9.0.post2、
transformers 5.2.0 和 triton-ascend 3.2.0.dev20260322，不隐式升级依赖。

在本仓根目录执行以下命令。Git 子模块获取可通过已配置的内网 proxy；构建不自动下载。

```bash
set -euo pipefail
git submodule update --init -- csrc/third_party/catlass
python -B p1_dev.py materials --from-submodule csrc/third_party/catlass
python -B tools/check_native_layout.py
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.1
source "$ASCEND_HOME_PATH/set_env.sh"
export SOC_VERSION=ascend910b3
python -B p1_dev.py doctor
LAYOUT_REPORT=$(mktemp -d /tmp/vllm-layout.XXXXXXXX)
python -B p1_dev.py editable --isolated-env --output "$LAYOUT_REPORT/editable"
python -B p1_dev.py verify --mode editable
```

CATLASS 固定提交为 `716fd7baa7fb7f6cac0488bb628fd1dd0e875641`。
也可用 `materials --from-submodule /已审核的本地子模块路径`，禁止覆盖已有漂移材料。
材料记录现在位于根 `submodule-materials.json`，sdist 必须携带该记录及实际材料文件。
`p1_dev.py build --output <新目录>` 保留普通 wheel 构建能力。

`--isolated-env` 表示当前是专用调测环境，并非绕过包版本冲突。
若环境仍装有 P4/旧插件，先准备干净验证容器。不要在运行着基线服务的环境卸载或切换源码。
原位切换旧 checkout 会留下未跟踪的 `ascend/` 材料/构建文件，源检查将拒绝该目录；
优先使用新 checkout，不要用 `git clean` 或递归删除处理它。

## 验证与回退

执行 `tools/run_layout_host_checks.py --list` 查看主机范围与明确延后的张量测试。
主机配对测试要求 `vllm/` 与 `LMCache/` 为同级目录。该入口不安装依赖，不探测 NPU。
测试中读取另一仓的契约仍需配对源码，GitHub 工作流也 checkout 配对分支；
若另一仓为私有仓，需由仓库管理员配置跨仓只读访问后重跑，不要把凭据写入源码。

内网需重新验证普通 wheel、sdist 解包重建、strict editable 原生资源路径及 2P2D。
从 checkout 外运行 `tools/check_npu_bootstrap.py` 与 `tools/validate_npu_native.py --help`，
再按 P4 已验证参数启动，DSA 双组/MTP 开启、C8 关闭，保持 TP8/DP2、TP4/DP4 验收项。
新的 proxy 路径是 `examples/disaggregated_prefill_v1/`，其余服务参数不应因目录调整改变。

回退须两仓一起使用 `p4-frozen-20261004` 的独立环境，不混用版本。
此处只说明目录整改，不承诺尚未执行的 NPU/性能验收结果。
