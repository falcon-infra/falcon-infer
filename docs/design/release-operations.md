# P5 P6 联合验收与成对切换

当前 `p6` 包含完整 `p5`，输入是不可变的 `native-layout-frozen-20261005`。
使用 `release-profile.json` 中的两个候选包，在最终配对提交上只执行一次完整的内网
功能和性能验收；无需在 P5 上重复跑同一套 2P2D。本批不改推理、缓存及原生实现。

## 安装与证据

可直接执行[基线一致的安装与验证指南](baseline-validation.md)，在四容器成对重装后，
沿用 native-layout 的 strict editable 和原启动命令进行本轮功能/性能对照。
不要求先改成 wheel 或新镜像；不要为了本轮比较改变部署方式或推理参数。
沿用 `p1_dev.py` 的 materials、doctor、build、editable、install 和 verify 子命令。
不能在运行中的 editable checkout 切分支，不能只更新 vLLM。
正式发布资格仍要求普通 wheel、sdist 解包重建和干净镜像验证；editable 报告须标明安装模式，
不冒充 wheel/镜像结果。已完成的功能用例是否可复用由验收负责人按来源和模式差异审核。

基础环境仍为 Python 3.11/aarch64、CANN 8.5.1、torch 2.9.0、torch-npu 2.9.0.post2、
Transformers 5.2.0、triton-ascend 3.2.0.dev20260322。禁止隐式重装 torch、安装旧插件或
联网模型下载。材料准备可使用获准代理，运行时不连接外部大模型服务。

归档配对 SHA、wheel 哈希、镜像 digest、依赖、材料、安装来源与四容器报告。GLM-5.2、
DSA 双组/MTP 开启、C8 关闭，TP8/DP2 和 TP4/DP4 分别覆盖离线/在线、缓存与恢复。
冷/热性能分别至少三轮，使用实际墙钟；HTTP 错误、零输出和中断不能记作成功。
native-layout 旧日志仅证明主链运行，P99 TTFT 和 Decode 尾段等待仍需同负载复测。

## 切换门槛

只有联合功能、性能、长稳、制品和恢复证据审核通过后才安排正式切换：

1. 在独立环境恢复冻结配对与基线配置，确认模型权重、材料和旧缓存可访问。
2. 停止新流量，按现场流程排空/取消在途请求，确认异步 store 与传输完成。
3. 原子化部署同一配对的两个 wheel/镜像和配置，再检查各节点身份与就绪状态。
4. 按已有 RemoteFill 协议执行 required paired restart，不单独重启一侧。
5. 新旧持久化 cache namespace 默认隔离；只有格式/权重/布局兼容性实测通过才复用。

回退使用完整冻结配对、配置、镜像与旧 namespace。不要把单仓回退或取消 DSA/MTP 当作
恢复等价。源码备份不包括模型、LFS payload、运行环境、未提交文件或服务配置。
原四仓只有在这些外部材料另行归档并完成恢复演练后才能退出活动工作区；不直接永久删除。

源码/host 工具不会实施生产切换。未执行的内网用例保持 pending，不以本页或候选版本号
代替阶段验收。
