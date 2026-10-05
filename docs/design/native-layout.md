# 原生目录结构与基线继承

目录整改已冻结于 `native-layout-frozen-20261005`。当前 P5/P6 从此点派生，
只调整发布候选元数据、检查与交付工具，不改模型、缓存或原生实现。
运行时 Python 和原生实现的冻结指纹由 `tools/check_native_layout.py` 校验。
冻结 native-layout 是本轮直接对照与回退点，冻结 P4 保留为更早的回退点；
冻结源码不代表完整验收矩阵已经通过。
实际安装和部署执行[基线一致的安装与验证指南](baseline-validation.md)。

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

使用独立验证容器及新 checkout，保留冻结 native-layout 环境。配对安装
`vllm==0.18.0+ascend.p5p6rc1` 与 `lmcache==0.4.3+ascend.p5p6rc1`。
沿用 Python 3.11/aarch64、CANN 8.5.1、torch 2.9.0、torch-npu 2.9.0.post2、
transformers 5.2.0 和 triton-ascend 3.2.0.dev20260322，不隐式升级依赖。

沿用标准 `pip install -e .` 与根 `setup.py`，不再使用阶段安装包装脚本。
完整成对命令、CANN 环境初始化、旧安装处理及四容器检查统一见安装指南，
不要仅装本仓而遗漏 LMCache。安装路径检查和冷导入从 `/tmp` 等源码外目录执行。
Git 子模块获取可通过已配置的内网 proxy；构建不自动下载。

CATLASS 固定提交为 `716fd7baa7fb7f6cac0488bb628fd1dd0e875641`。
通过标准 Git 子模块流程准备本地固定材料，禁止覆盖已有漂移材料。
材料记录现在位于根 `submodule-materials.json`，sdist 必须携带该记录及实际材料文件。
`setup.py` 在构建或 sdist 时自动核验并首次登记材料，不自动下载。
`python setup.py bdist_wheel`、`python setup.py sdist` 保留标准制品构建能力。

内网标准 pip 安装须保留 `--no-build-isolation --no-deps --no-index`。
若环境仍装有 P4/旧插件，先准备干净验证容器。不要在运行着基线服务的环境卸载或切换源码。
原位切换旧 checkout 会留下未跟踪的 `ascend/` 材料/构建文件，源检查将拒绝该目录；
优先使用新 checkout，不要用 `git clean` 或递归删除处理它。

## 验证与回退

执行 `tools/run_layout_host_checks.py --list` 查看主机范围与明确延后的张量测试。
主机配对测试要求 `vllm/` 与 `LMCache/` 为同级目录。该入口不安装依赖，不探测 NPU。
测试中读取另一仓的契约仍需配对源码，GitHub 工作流也 checkout 配对分支；
若另一仓为私有仓，需由仓库管理员配置跨仓只读访问后重跑，不要把凭据写入源码。

本轮可继续在 strict editable 下复测 2P2D，与冻结 native-layout 保持安装方式一致。
正式发布另需普通 wheel、sdist 解包重建与镜像检查，不能用 editable 结果替代。
从 checkout 外运行 `tools/check_npu_bootstrap.py` 与 `tools/validate_npu_native.py --help`，
再按 native-layout 已验证参数启动，DSA 双组/MTP 开启、C8 关闭，保持 TP8/DP2、TP4/DP4 验收项。
proxy 路径已在 native-layout 基线中改为 `examples/disaggregated_prefill_v1/`，
P6 不再改变此路径或其他服务参数。

回退须两仓一起使用 `native-layout-frozen-20261005` 的独立环境，不混用版本。
P5/P6 使用最终 P6 配对执行一次完整内网验收，安装命令不变但两仓必须重装。
此处只说明目录整改，不承诺尚未执行的 NPU/性能验收结果。
