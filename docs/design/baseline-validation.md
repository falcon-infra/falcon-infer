# 基线一致的安装部署与验证

本页用于在内网使用基线的标准 pip/setup.py 入口安装当前 P6 配对，
然后沿用原有启动自动化做功能、性能验证。不是重新执行 P1 合仓，也不是重新配置模型。
两仓 README 均以本页为安装入口；以下步骤在四个实际运行容器中分别执行一次。

## 版本和继承关系

| 项目 | 冻结基线 | 当前验证对象 |
| --- | --- | --- |
| Git 入口 | `native-layout-frozen-20261005` | 两仓 `p6`，完整包含 `p5` |
| vLLM | `0.18.0+ascend.layout1` | `0.18.0+ascend.p5p6rc1` |
| LMCache | `0.4.3+ascend.layout1` | `0.4.3+ascend.p5p6rc1` |
| 安装入口 | `pip install -e .`、根 `setup.py` | 相同标准入口，不再需要阶段包装脚本 |
| 部署入口 | `vllm serve`、原 proxy、原客户端 | 原入口及原参数不变 |

冻结基线提交：vLLM `18fde02dc54a432f81ce5350c3358429a40f4bd3`，
LMCache `e8236d8e1c41cd10752204d8cb9ef3b6074f999e`。
当前两个精确提交取本批审核的 `design/p6/baseline/pair.json`，不要只按分支名、
包版本或另一节点的报告判断身份。发布者须同时交付匹配的两仓源码与配对清单。

已核对：根 CMake、`cmake/`、产品 Python、native 实现、固定材料和 ABI
继承冻结基线；本次只调整打包入口、验证工具及说明，不改变推理算法或部署参数。
构建逻辑现直接位于根 `setup.py`，通过 `setuptools.build_meta` 接入 pip；
已删除 `p1_build.py`、`p1_dev.py`，无需新安装器。
vLLM 的隔离构建依赖补齐同版本的 `triton-ascend==3.2.0.dev20260322`，
没有改变运行依赖版本。源码一致不等于本轮编译或性能已经验收。

两仓在准备好下文环境和固定材料后，均可在各自仓根直接执行：

```bash
python -m pip install -e . --no-build-isolation --no-deps --no-index
python setup.py bdist_wheel
python setup.py sdist
```

`pip install -e .` 本身受支持，默认自动选择 strict editable，不必额外指定
`editable_mode=strict`。不加上述 pip 参数时，pip 按标准行为解析运行依赖、
创建隔离构建环境并获取构建依赖，需要可用的索引/完整制品源；
它不会自动复用已安装的全部构建依赖。内网基线验证请保留上述三个参数，
不升级 torch/torch_npu 等现有基础环境。普通 wheel 和 sdist 默认输出到 `dist/`。
`setup.py` 命令兼容本次固定的 setuptools；也支持标准 PEP 517 前端，
无需把既有打包自动化改成新的脚本。

可以继续用 strict editable 完成本轮基线对比，不强制先改用 wheel 或新镜像。
正式发布仍需普通 wheel、sdist 重建、镜像和恢复证据；不能以 editable 结果替代它们。
P5 不单独重复整轮测试，使用最终 P6 配对统一验证。

## 环境和操作边界

沿用 910B3、4 节点各 8 卡、Python 3.11.14/aarch64、CANN 8.5.1、
torch 2.9.0+cpu、torch-npu 2.9.0.post2、transformers 5.2.0、
triton-ascend 3.2.0.dev20260322。torch 的 `+cpu` 后缀不表示 CPU 推理，
NPU 能力由 torch_npu 提供。不升级依赖，不运行旧插件 patch/install 脚本。

构建工具也须预先满足 `requirements/build.txt`，沿用内网的 setuptools 79.0.1、
packaging 26.0 及已有 wheel/CMake/Ninja/pybind11；`setup.py` 不会替你安装依赖。
setuptools 68 等旧版本无法解析当前许可证元数据，不属于本候选构建环境。

使用独立验证容器，保留冻结基线服务及其源码。若复用已停止测试的专用容器，
先归档旧安装，再按下文卸载旧框架；不能改动仍承载基线服务的环境。
不要在已运行的 editable checkout 切分支、移动目录或清理 `build/`。
相同容器镜像不能代替四容器的实际安装核对。

核心构建入口都在两仓内。下面沿用基线的两个审核辅助脚本，并读取新的配对清单，
请把以下三个文件同步到 `/workspace/zzj/design/` 的同名相对路径：
`p1/tools/check_pip_dependencies.py`、`p4/tools/check_pair.py`、
`p6/baseline/pair.json`。它们是验收输入，不是第三个运行时包。
没有这些文件时先补齐，不用旧 P4 清单或跳过依赖/身份门槛。

以下命令在同一 Bash 会话按顺序执行；任一步失败立即停止并保留报告。
默认源码路径与基线相同。若它仍被旧服务使用，把 `P56_REPOS` 改为新的独立目录，
并同步调整启动自动化中的源码路径。

```bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
P56_PY=/usr/local/python3.11.14/bin/python
P56_REPOS=/workspace/zzj/p1-repos
P56_DESIGN=/workspace/zzj/design
P56_PAIR="$P56_DESIGN/p6/baseline/pair.json"
test -x "$P56_PY"
test -f "$P56_PAIR"
test -f "$P56_DESIGN/p1/tools/check_pip_dependencies.py"
test -f "$P56_DESIGN/p4/tools/check_pair.py"
mkdir -p -- /workspace/zzj/p56-check
P56_REPORT=$(mktemp -d /workspace/zzj/p56-check/baseline-compatible.XXXXXXXX)
printf '本容器报告目录：%s\n' "$P56_REPORT"
cd /tmp
"$P56_PY" -B -m pip list --format=json > "$P56_REPORT/packages-before.json"
"$P56_PY" -B -m pip freeze --all > "$P56_REPORT/freeze-before.txt"
```

若已装相同 P6 配对且安装来源就是本次源码，后续直接重装即可。
标准 pip 会替换同名旧产品包，但不会替你清除独立的旧插件或检查成对版本。
不能保留独立 `vllm-ascend/lmcache-ascend`；安装后的配对/路径检查会拒绝混装。
仅在确认本容器不承载保留服务且测试进程已停止后，才执行下面的可选卸载块：

```bash
"$P56_PY" -B -m pip --disable-pip-version-check --no-color uninstall -y \
  vllm lmcache vllm-ascend lmcache-ascend \
  2>&1 | tee "$P56_REPORT/framework-uninstall.log"
```

不卸载 torch/CANN 等基础依赖，不删除模型、缓存、源码或构建目录。
标准安装没有 `--isolated-env` 参数；专用容器由操作者提前准备，pip 不负责创建。
保留 CANN 所需的 `PYTHONPATH`，只核查自己添加的旧框架路径；
不要通过添加源码路径来绕过缺模块/旧链接树问题。

## 检出配对源码和固定材料

首次克隆须等两仓 P6 提交发布完成。下面只在目标不存在时克隆，不切换已有 checkout。
若已有目录不符合本批 SHA，停止并准备新目录，不覆盖修改。
构建前核对跟踪文件干净、冻结标签为祖先，以及不存在旧根 `ascend/`。

```bash
P56_VLLM_REF=$("$P56_PY" -B -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["repositories"]["vllm"]["commit"])' "$P56_PAIR")
P56_LMC_REF=$("$P56_PY" -B -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["repositories"]["LMCache"]["commit"])' "$P56_PAIR")
mkdir -p -- "$P56_REPOS"
if [ ! -e "$P56_REPOS/vllm" ] && [ ! -L "$P56_REPOS/vllm" ]; then
  git clone --branch p6 --single-branch \
    git@github.com:zhangzhaoju/vllm-dual.git "$P56_REPOS/vllm"
  git -C "$P56_REPOS/vllm" switch --detach "$P56_VLLM_REF"
fi
if [ ! -e "$P56_REPOS/LMCache" ] && [ ! -L "$P56_REPOS/LMCache" ]; then
  git clone --branch p6 --single-branch \
    git@github.com:zhangzhaoju/LMCache-dual.git "$P56_REPOS/LMCache"
  git -C "$P56_REPOS/LMCache" switch --detach "$P56_LMC_REF"
fi
for P56_NAME in vllm LMCache; do
  git -C "$P56_REPOS/$P56_NAME" fetch origin tag native-layout-frozen-20261005
  git -C "$P56_REPOS/$P56_NAME" merge-base --is-ancestor \
    native-layout-frozen-20261005 HEAD
  git -C "$P56_REPOS/$P56_NAME" diff --quiet
  git -C "$P56_REPOS/$P56_NAME" diff --cached --quiet
  git -C "$P56_REPOS/$P56_NAME" status --short --branch \
    | tee "$P56_REPORT/source-$P56_NAME.txt"
  test ! -e "$P56_REPOS/$P56_NAME/ascend"
  test ! -L "$P56_REPOS/$P56_NAME/ascend"
done
test "$(git -C "$P56_REPOS/vllm" rev-parse HEAD)" = "$P56_VLLM_REF"
test "$(git -C "$P56_REPOS/LMCache" rev-parse HEAD)" = "$P56_LMC_REF"

git -C "$P56_REPOS/vllm" submodule update --init -- csrc/third_party/catlass
git -C "$P56_REPOS/LMCache" submodule update --init -- third_party/kvcache-ops
for P56_NAME in vllm LMCache; do
  (cd "$P56_REPOS/$P56_NAME" && \
    "$P56_PY" -B setup.py sdist --dist-dir "$P56_REPORT/sdist-$P56_NAME") \
    2>&1 | tee "$P56_REPORT/sdist-$P56_NAME.log"
  "$P56_PY" -B "$P56_REPOS/$P56_NAME/tools/check_native_layout.py" \
    | tee "$P56_REPORT/layout-$P56_NAME.json"
done
```

Git 获取可使用已批准的代理；构建/安装不访问包索引。
CATLASS 固定 `716fd7baa7fb7f6cac0488bb628fd1dd0e875641`，
kvcache-ops 固定 `9f18d2339bc58a43429f7d5bdaef1628c820eff5`。
不再运行独立 materials 子命令。`setup.py` 在 sdist/编译时自动核验本地材料并首次生成
根 `submodule-materials.json`，不会自动下载、覆盖材料或重写已有清单。
未初始化、非固定提交、不干净或内容漂移均阻断；sdist 自带已核验材料和清单，
解包构建不需要 Git。上面的 sdist 命令也可提前发现材料问题且不编译 native。
使用已审核本地 Git 镜像时，按原 Git 子模块流程初始化到相同路径、相同固定提交。
不能用 `--source-only` 放过内网构建材料缺失。

## 沿用基线的 editable 安装

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.1
set +u
source "$ASCEND_HOME_PATH/set_env.sh"
set -u
export SOC_VERSION=ascend910b3
export VLLM_TARGET_DEVICE=ascend
export USE_MINDSPORE=0
export BUILD_WITH_HIP=0
export VLLM_USE_PRECOMPILED=0
export COMPILE_CUSTOM_KERNELS=1
export VLLM_PLUGINS=""
export VLLM_NO_USAGE_STATS=1
cd /tmp

"$P56_PY" -B "$P56_DESIGN/p1/tools/check_pip_dependencies.py" \
  --output "$P56_REPORT/pip-before"
for P56_NAME in vllm LMCache; do
  (cd "$P56_REPOS/$P56_NAME" && \
    "$P56_PY" -B -m pip install -e . \
      --no-index --no-deps --no-build-isolation --force-reinstall) \
    2>&1 | tee "$P56_REPORT/editable-$P56_NAME.log"
done
```

两个包版本已变，必须成对重新编译安装，不能复用 layout1 的安装链接树或手工复制旧
`.so`。脚本自动使用新 native 构建目录，自动安装自定义算子，不用手工 mkdir/patch。
成功后保留 `build/p1-native/run-*` 和 `build/__editable__.*`。
安装日志在本次报告目录的 `editable-vllm.log`、`editable-LMCache.log`；
`set -euo pipefail` 保留失败退出语义，失败时停止，不继续启动服务。
重试时新建报告目录，不删除失败证据来复用路径。

strict editable 中已有 Python 文件的修改可被新进程读到；新增/删除文件、切分支、
改版本或 native 源码后须重新执行安装。原生库始终重新编译，不复用旧 `.so`。
生成数据文件 `p1_build_info.json` 和内部构建目录名保持历史兼容，
不是安装脚本，也不代表运行在 P1 版本。
原来显式传入 `--config-settings editable_mode=strict` 的标准 pip 自动化仍可沿用；
不再调用 materials/doctor/editable/install 等阶段脚本子命令。

pip 审核仅允许 op-compile-tool 0.1.0 错报缺失的
`getopt`、`inspect`、`multiprocessing` 三个标准库，且标准库必须实际可用；
其余错误继续阻断，不用 `pip check || true` 跳过。

## 四容器安装核对和 NPU 检查

从源码目录外执行。以下导入检查不加载模型，但原生库可能访问 CANN 运行时。

```bash
cd /tmp
"$P56_PY" -B "$P56_REPOS/vllm/tools/check_native_layout.py" --installed editable \
  | tee "$P56_REPORT/vllm-installed.json"
"$P56_PY" -B "$P56_REPOS/LMCache/tools/check_native_layout.py" --installed editable \
  | tee "$P56_REPORT/lmcache-installed.json"
"$P56_PY" -B "$P56_DESIGN/p4/tools/check_pair.py" \
  --workspace "$P56_REPOS" --pair "$P56_PAIR" \
  --output "$P56_REPORT/pair-installed.json"
"$P56_PY" -B "$P56_DESIGN/p1/tools/check_pip_dependencies.py" \
  --output "$P56_REPORT/pip-after"
"$P56_PY" -B "$P56_REPOS/vllm/tools/check_npu_bootstrap.py" \
  --inspect-glm --check-lmcache --output "$P56_REPORT/bootstrap"
"$P56_PY" -B "$P56_REPOS/vllm/tools/validate_npu_native.py" \
  --spawn --output "$P56_REPORT/native-import.json"
"$P56_PY" -B "$P56_REPOS/LMCache/tools/p4_runtime_smoke.py" \
  --output "$P56_REPORT/lmcache-import"
"$P56_PY" -B -m vllm.entrypoints.cli.main serve --help \
  2>&1 | tee "$P56_REPORT/serve-help.log"
```

两包均为 `.p5p6rc1`，导入位置为本次 checkout 的 strict editable 链接树；
每条命令退出码为 0，检查报告 `passed=true`，仅 pip 的已批豁免允许
`passed_with_waivers`。没有独立 `vllm_ascend/lmcache_ascend` 安装。
这些结果尚不能证明 NPU 算子和模型可用。

在已分配的空闲测试卡上，沿用原设备可见性设置再执行：

```bash
"$P56_PY" -B "$P56_REPOS/vllm/tools/validate_npu_native.py" \
  --spawn --device-smoke --output "$P56_REPORT/native-device.json"
"$P56_PY" -B "$P56_REPOS/LMCache/tools/p4_runtime_smoke.py" \
  --npu --output "$P56_REPORT/lmcache-device"
```

要求算子 smoke 及 LMCache pinned-host/NPU 小数据复制通过。
不能拿无设备参数的导入结果代替；没有空闲测试卡时记录未执行。
不要指向旧安装的 OPP 目录绕过检查。

## 沿用基线部署和负载

四容器通过安装及设备检查后，执行 native-layout 成功时的同一套节点、proxy 和客户端
启动命令。现场 launcher/Ansible、网卡/地址/端口、LMCache YAML、模型文件和负载
不在本仓内；继续使用已归档版本，不根据通用示例重新拼装它们。

启动自动化只更新本批源码 SHA、两个包版本和报告路径。
若源码位置没改，服务命令和路径也不必改。
停用自动化中检出旧 P1/P2/P4 SHA、安装四包、patch 插件或清理 editable build 的旧步骤。
如需新目录，只替换该目录引用，不调整推理配置。

确保原来的 `vllm` 命令属于安装用的 Python：

```bash
export PATH="$(dirname "$P56_PY"):$PATH"
hash -r
command -v vllm | tee "$P56_REPORT/vllm-cli-path.txt"
head -n 1 "$(command -v vllm)" | tee "$P56_REPORT/vllm-cli-shebang.txt"
"$P56_PY" -B -m vllm.entrypoints.cli.main --version \
  2>&1 | tee "$P56_REPORT/vllm-cli-version.txt"
```

随后照原流程运行 `vllm serve`。也可将命令入口固定为
`"$P56_PY" -B -m vllm.entrypoints.cli.main serve`，其余参数原样保留。
不要在本页未指定节点地址的情况下复制一个简化的 serve 示例启动完整 2P2D。

已上传成功日志的 TP8/DP2 对照配置如下，最终以现场归档的有效参数和 YAML 为准：

| 项目 | 必须继承的基线配置 |
| --- | --- |
| 模型 | 同一 GLM-5.2 checkpoint/量化文件/权重索引，W4A8，保留原 RoPE 配置 |
| DSA 和 MTP | DSA 双组开启；`method=deepseek_mtp`、`num_speculative_tokens=1`；C8 关闭，KV 实际为 BF16 |
| 连接器 | `LMCacheConnectorV1`、`kv_connector_module_path=None`，P/D 均为 `kv_both`，失败策略 `fail` |
| TP8 拓扑 | 2P2D，TP8/DP2，每节点一个实例；TP4/DP4 用另行归档的原拓扑参数 |
| 调度预算 | `max_model_len=140000`、`max_num_seqs=16`、`max_num_batched_tokens=4096`、prefix caching 关闭 |
| P 与 D 差异 | P eager、recompute scheduler 关闭；D PIECEWISE 图、recompute scheduler 开启，保留原图大小 |
| Proxy | `examples/disaggregated_prefill_v1/load_balance_proxy_server_enhanced.py`，沿用原全部参数 |
| 缓存与网络 | 原 YAML、CPU KV/RemoteFill、Mooncake、设备可见性、NUMA/网卡、控制/数据端口配置 |

`max_model_len` 等旧 CLI 名称未改，不表示恢复 GPU 支持。
`deepseek_mtp` 是 GLM 的共享实现名称，不扩大模型支持范围。
不要改用历史 MooncakeConnectorV1/DeepSeek 示例，或恢复 `lmcache_ascend.*` module path。
不要在本轮对比时改调度预算、MTP 数量、图开关或 profiler 采样方式来优化结果。

使用相同的有效请求集、采样参数、客户端并发、预热及冷/热缓存阶段。
不改客户端原启动命令；若需要补墙钟/逐 token 时间，则对基线和候选同时应用相同采集方式。
不以 Waiting Req 为 0 判断 Prefill 变慢，不平均节点 MTP 百分比；
接受率用各节点接受/草稿总数计算。HTTP400、零输出、中断/未完成请求单独计数，
不将请求延迟之和当整轮墙钟，不只比较成功请求交集。

每容器归档本次报告、配对清单、材料清单、实际命令/YAML 与哈希、安装来源，
并归档完整 P/D/proxy/客户端日志和原始 benchmark。TP8/DP2、TP4/DP4 分开记录；
2P2D 成功不能替代离线、无缓存在线、流式/取消、CPU KV、checkpoint/恢复和长稳。
旧日志只证明主链运行，P99 和尾段等待仍需本轮同负载确认，不能预先宣称性能等价。

## 普通 wheel 和回退

若基线使用 editable，本轮可继续 editable。普通 wheel 同样由根 `setup.py` 打包，
无需额外构建脚本。先完成前面的环境、材料和依赖检查，再按以下方式生成 wheel/sdist。
这是另一种安装模式，不在仍运行的 editable 服务上执行：

```bash
for P56_NAME in vllm LMCache; do
  (cd "$P56_REPOS/$P56_NAME" && \
    "$P56_PY" -B setup.py bdist_wheel --dist-dir "$P56_REPORT/wheel-$P56_NAME") \
    2>&1 | tee "$P56_REPORT/wheel-$P56_NAME.log"
done
P56_VLLM_WHEEL="$P56_REPORT/wheel-vllm/vllm-0.18.0+ascend.p5p6rc1-cp311-cp311-linux_aarch64.whl"
P56_LMC_WHEEL="$P56_REPORT/wheel-LMCache/lmcache-0.4.3+ascend.p5p6rc1-cp311-cp311-linux_aarch64.whl"
"$P56_PY" -B "$P56_REPOS/vllm/tools/check_native_layout.py" --wheel "$P56_VLLM_WHEEL" \
  | tee "$P56_REPORT/wheel-contents-vllm.json"
"$P56_PY" -B "$P56_REPOS/LMCache/tools/check_native_layout.py" --wheel "$P56_LMC_WHEEL" \
  | tee "$P56_REPORT/wheel-contents-lmcache.json"
"$P56_PY" -B -m pip install --no-index --no-deps --force-reinstall \
  "$P56_VLLM_WHEEL" "$P56_LMC_WHEEL" \
  2>&1 | tee "$P56_REPORT/install-wheels.log"
cd /tmp
for P56_NAME in vllm LMCache; do
  "$P56_PY" -B "$P56_REPOS/$P56_NAME/tools/check_native_layout.py" --installed wheel \
    | tee "$P56_REPORT/wheel-installed-$P56_NAME.json"
done
```

前面的 `setup.py sdist` 已生成两份源码包。标准 `sdist` 不编译、不探测 NPU，
但必须包含固定第三方材料和清单；它不是“下载依赖后才能用”的空壳源码包。
正式发布须再从两份 sdist 的独立解包目录执行相同 `python setup.py bdist_wheel`，
不得借用原 checkout、其 `build/` 或另一个包的源码。
工作区 `design/p5/tools/qualify.py build` 已改为调用上述标准命令并检查两轮 wheel；
该工具用于批量归档，不是安装或打包的必要入口。

wheel 可以在一台内网容器构建，审核并复制同一对制品到四容器。
保留并核对 SHA256，再成对安装；重复身份/pip/导入/NPU 检查时使用新的报告路径。
普通 wheel、sdist、镜像及正式切换要求见本仓 release operations 文档及工作区 P5/P6 指导。

失败时保留日志。在独立环境中恢复两仓冻结标签、两包 layout1 安装、原启动/YAML、
模型和缓存 namespace，不做单仓回退。不删除原四仓和仍在使用的构建/缓存目录。
本页文档与主机检查不代表已完成内网构建、功能、性能或生产切换验收。
