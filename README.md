# vllm — Ascend910B3 / GLM-5.2

P4 原生推理源码候选，版本 `0.18.0+ascend.p4`。仅支持 GLM-5.2 原生文本生成及
DSA/MTP、CPU KV 卸载与共享、跨实例缓存、P/D、RemoteFill、checkpoint 和恢复。
不提供其他厂商/Ascend 型号、多模态、LoRA、pooling 或训练能力。

配对安装 `vllm==0.18.0+ascend.p4` 与 `lmcache==0.4.3+ascend.p4`。
冻结 P3 输入为两仓同名标签 `p3-frozen-20261003`；删除的源码可从该标签恢复。
P4 源码检查不等于 CANN 编译、ABI、910B3 或 2P2D 验收通过。

请先阅读 [P4 安装与验证指南](docs/design/ascend_p4.md)。历史文件名 `p1_dev.py` 继续使用，
但已按 P4 更新。切换阶段后必须重建 native 扩展并重新安装两仓 strict editable，
四个节点的每个容器均需执行。不要复用 P3 `.so`，也不要覆盖正在运行的基线环境。
构建不隐式联网或升级依赖，内网材料使用审核过的固定版本。

```bash
python -B tools/check_p4_profile.py
python -B tools/check_npu_native.py
python -B p1_dev.py --help
```

上游项目及 Ascend donor 的许可证保留在 `LICENSE`、`ascend/LICENSE`。
源码中的共享 DeepSeek/Eagle 等命名不代表对其他 checkpoint 的支持。
