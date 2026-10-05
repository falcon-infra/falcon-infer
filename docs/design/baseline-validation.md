# 基线一致的安装部署与验证

本页用于在内网使用与 native-layout 基线相同的指令安装当前 P6 配对，
然后沿用原有启动自动化做功能、性能验证。不是重新执行 P1 合仓，也不是重新配置模型。
两仓 README 均以本页为安装入口；以下步骤在四个实际运行容器中分别执行一次。

## 版本和继承关系

| 项目 | 冻结基线 | 当前验证对象 |
| --- | --- | --- |
| Git 入口 | `native-layout-frozen-20261005` | 两仓 `p6`，完整包含 `p5` |
| vLLM | `0.18.0+ascend.layout1` | `0.18.0+ascend.p5p6rc1` |
| LMCache | `0.4.3+ascend.layout1` | `0.4.3+ascend.p5p6rc1` |
| 安装入口 | 根 `p1_dev.py` | 同名脚本、相同子命令和参数 |
| 部署入口 | `vllm serve`、原 proxy、原客户端 | 原入口及原参数不变 |

冻结基线提交：vLLM `18fde02dc54a432f81ce5350c3358429a40f4bd3`，
LMCache `e8236d8e1c41cd10752204d8cb9ef3b6074f999e`。
当前两个精确提交取本批审核的 `design/p6/baseline/pair.json`，不要只按分支名、
包版本或另一节点的报告判断身份。发布者须同时交付匹配的两仓源码与配对清单。

已核对：相对冻结基线，两仓 `p1_build.py`、根 CMake、`cmake/`、
`requirements/`、产品 Python 和 native 实现不变；`p1_dev.py` 只更新
`VERSIONS`，`pyproject.toml` 只更新产品版本。构建方式、固定材料、
native ABI、CLI 和缓存协议继承基线。源码一致不等于本轮编译或性能已经验收。

可以继续用 strict editable 完成本轮基线对比，不强制先改用 wheel 或新镜像。
正式发布仍需普通 wheel、sdist 重建、镜像和恢复证据；不能以 editable 结果替代它们。
P5 不单独重复整轮测试，使用最终 P6 配对统一验证。

## 环境和操作边界

沿用 910B3、4 节点各 8 卡、Python 3.11.14/aarch64、CANN 8.5.1、
torch 2.9.0+cpu、torch-npu 2.9.0.post2、transformers 5.2.0、
triton-ascend 3.2.0.dev20260322。torch 的 `+cpu` 后缀不表示 CPU 推理，
NPU 能力由 torch_npu 提供。不升级依赖，不运行旧插件 patch/install 脚本。

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
若存在 layout1/P4 等旧版本或独立 `vllm-ascend/lmcache-ascend`，安装脚本会拒绝。
仅在确认本容器不承载保留服务且测试进程已停止后，才执行下面的可选卸载块：

```bash
"$P56_PY" -B -m pip --disable-pip-version-check --no-color uninstall -y \
  vllm lmcache vllm-ascend lmcache-ascend \
  2>&1 | tee "$P56_REPORT/framework-uninstall.log"
```

不卸载 torch/CANN 等基础依赖，不删除模型、缓存、源码或构建目录。
`--isolated-env` 是专用环境声明，不会创建环境、卸载旧包或忽略冲突。
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
"$P56_PY" -B "$P56_REPOS/vllm/p1_dev.py" materials \
  --from-submodule "$P56_REPOS/vllm/csrc/third_party/catlass" \
  | tee "$P56_REPORT/materials-vllm.json"
"$P56_PY" -B "$P56_REPOS/LMCache/p1_dev.py" materials \
  --from-submodule "$P56_REPOS/LMCache/third_party/kvcache-ops" \
  | tee "$P56_REPORT/materials-lmcache.json"
for P56_NAME in vllm LMCache; do
  "$P56_PY" -B "$P56_REPOS/$P56_NAME/tools/check_native_layout.py" \
    | tee "$P56_REPORT/layout-$P56_NAME.json"
done
```

Git 获取可使用已批准的代理；构建/安装不访问包索引。
CATLASS 固定 `716fd7baa7fb7f6cac0488bb628fd1dd0e875641`，
kvcache-ops 固定 `9f18d2339bc58a43429f7d5bdaef1628c820eff5`。
也可跳过相应 submodule update，给 `--from-submodule` 传已审核的本地 Git 材料路径。
不要覆盖漂移材料。清单位于根 `submodule-materials.json`，不能用
`--source-only` 放过内网构建材料缺失。

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
"$P56_PY" -B "$P56_REPOS/vllm/p1_dev.py" doctor \
  --output "$P56_REPORT/vllm-doctor.json"
"$P56_PY" -B "$P56_REPOS/LMCache/p1_dev.py" doctor \
  --output "$P56_REPORT/lmcache-doctor.json"
"$P56_PY" -B "$P56_REPOS/vllm/p1_dev.py" editable --isolated-env \
  --output "$P56_REPORT/vllm-editable"
"$P56_PY" -B "$P56_REPOS/LMCache/p1_dev.py" editable --isolated-env \
  --output "$P56_REPORT/lmcache-editable"
```

两个包版本已变，必须成对重新编译安装，不能复用 layout1 的安装链接树或手工复制旧
`.so`。脚本自动使用新 native 构建目录，自动安装自定义算子，不用手工 mkdir/patch。
成功后保留 `build/p1-native/run-*` 和 `build/__editable__.*`。
安装日志在各输出目录的 `command.log`、`command-result.json`。
重试时新建报告目录，不删除失败证据来复用路径。

实际 pip 参数仍为 `--no-index --no-deps --no-build-isolation --force-reinstall`
和 `--config-settings editable_mode=strict --editable <仓根>`。
可给上述 editable 命令追加 `--dry-run` 查看完整命令；它不安装也不执行 doctor。
若既有自动化直接使用这些 pip 参数，可以保留，但仍需先执行材料/doctor/旧包门槛，
再执行下节安装核对；不能省略 `--no-deps` 或改成 CUDA 预编译安装。
pip 审核仅允许 op-compile-tool 0.1.0 错报缺失的
`getopt`、`inspect`、`multiprocessing` 三个标准库，且标准库必须实际可用；
其余错误继续阻断，不用 `pip check || true` 跳过。

## 四容器安装核对和 NPU 检查

从源码目录外执行。以下导入检查不加载模型，但原生库可能访问 CANN 运行时。

```bash
cd /tmp
"$P56_PY" -B "$P56_REPOS/vllm/p1_dev.py" verify --mode editable \
  --output "$P56_REPORT/vllm-installed.json"
"$P56_PY" -B "$P56_REPOS/LMCache/p1_dev.py" verify --mode editable \
  --output "$P56_REPORT/lmcache-installed.json"
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

若基线使用 editable，本轮继续 editable；若使用普通 wheel，沿用同名 build/install，
将 wheel 文件名换成本批版本。以下是另一种安装模式，不在已运行的 editable 服务上执行：

```bash
"$P56_PY" -B "$P56_REPOS/vllm/p1_dev.py" build \
  --output "$P56_REPORT/wheel-vllm"
"$P56_PY" -B "$P56_REPOS/LMCache/p1_dev.py" build \
  --output "$P56_REPORT/wheel-lmcache"
"$P56_PY" -B "$P56_REPOS/vllm/p1_dev.py" install --isolated-env \
  --wheel "$P56_REPORT/wheel-vllm/wheels/vllm-0.18.0+ascend.p5p6rc1-cp311-cp311-linux_aarch64.whl" \
  --output "$P56_REPORT/install-wheel-vllm"
"$P56_PY" -B "$P56_REPOS/LMCache/p1_dev.py" install --isolated-env \
  --wheel "$P56_REPORT/wheel-lmcache/wheels/lmcache-0.4.3+ascend.p5p6rc1-cp311-cp311-linux_aarch64.whl" \
  --output "$P56_REPORT/install-wheel-lmcache"
"$P56_PY" -B "$P56_REPOS/vllm/p1_dev.py" verify --mode wheel \
  --output "$P56_REPORT/vllm-wheel-installed.json"
"$P56_PY" -B "$P56_REPOS/LMCache/p1_dev.py" verify --mode wheel \
  --output "$P56_REPORT/lmcache-wheel-installed.json"
```

wheel 可以在一台内网容器构建，审核并复制同一对制品到四容器。
保留并核对 SHA256，再成对安装；重复身份/pip/导入/NPU 检查时使用新的报告路径。
普通 wheel、sdist、镜像及正式切换要求见本仓 release operations 文档及工作区 P5/P6 指导。

失败时保留日志。在独立环境中恢复两仓冻结标签、两包 layout1 安装、原启动/YAML、
模型和缓存 namespace，不做单仓回退。不删除原四仓和仍在使用的构建/缓存目录。
本页文档与主机检查不代表已完成内网构建、功能、性能或生产切换验收。
