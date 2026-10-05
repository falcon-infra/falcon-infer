# vllm 原生 Ascend 推理框架

当前为 P5/P6 联合验收候选，输入为不可变标签
`native-layout-frozen-20261005`。仅支持 Ascend910B3 和 GLM-5.2 原生文本推理，
保留 DSA 双组、MTP、CPU KV 缓存、P/D、RemoteFill、checkpoint 和恢复。
本次不修改推理算法、参数、C8 开关或二进制接口。

配对版本为 `vllm==0.18.0+ascend.p5p6rc1` 和
`lmcache==0.4.3+ascend.p5p6rc1`。根目录不再有 `ascend/`：
Python 实现在 `vllm/`，原生实现和构建配置在
`csrc/`、`cmake/`，测试、工具和部署示例归入各自的根目录。
包内按功能划分的 Ascend 后端及已有 ABI 名称保持不变。

阅读[目录与安装指南](docs/design/native-layout.md)。
`p1_dev.py` 名称保留，但须使用当前分支的脚本，重新编译并安装两仓 strict editable。
四节点每个容器都要核对配对版本；不要复用 P4/layout1 的安装链接树或原生制品。
P5 的发布准备由 P6 完整继承，最终只验证一次 P6 配对；
`release-profile.json` 标明候选范围，尚不代表构建、功能、性能或切换通过。

```bash
python -B tools/check_native_layout.py --source-only
python -B tools/run_layout_host_checks.py
python -B p1_dev.py --help
```

以上主机检查不等于 CANN 编译、ABI 或 2P2D 验收通过。
迁移明细和冻结源码指纹见 `docs/design/layout-migration.json`。
同文许可证已归并为根 `LICENSE`；各文件的版权声明不变，历史路径可从冻结标签恢复。
