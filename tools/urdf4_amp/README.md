# URDF4 V2 AMP + CTS

当前任务为 `urdf4_amp_cts_v1`，实验目录为 `urdf4_amp_cts_style`。本阶段先学习 V2 覆盖的全部12类运动和站立，验收后再扩速度；不自动外推到0.6 m/s。

在已配置 Isaac Gym、PyTorch、legged_gym/rsl_rl 的仓库环境中，从头启动：

```bash
bash tools/urdf4_amp/train_cts.sh
```

默认4096环境、8000轮、GPU0。不要直接续训旧双帧AMP检查点，新检查点契约会拒绝不兼容的恢复。

本仓库收录训练必需的 V2 数据和名义验证回执：

- `artifacts/urdf4_amp_design_20261113_v2/amp_reference_v2/`
- `artifacts/urdf4_amp_isaac_validation_20261113/validation_receipt.json`

数据原 manifest 保持不变；加载时核对 URDF、全部片段哈希及验证回执。回执只证明此前名义验证通过，不代表新策略或实机验收。

有限检查：

```bash
python tools/urdf4_amp/smoke_cts_isaac.py \
  --output /tmp/urdf4_cts_smoke --updates 3 --num_envs 128 \
  --headless --sim_device cuda:0 --rl_device cuda:0
python tools/urdf4_amp/check_style_learning.py \
  --output /tmp/urdf4_cts_short --updates 150 --num_envs 1024 \
  --headless --sim_device cuda:0 --rl_device cuda:0
```

`play.py --task urdf4_amp_cts_v1 --sweep` 使用原生 Gym 播放学生策略，切换12个V2指令及站立，保留鼠标相机操作；零速度加 `--march` 可查看原地踏步。

奖励设计、128/4096环境冒烟及150轮短测结果见 [验证记录](../../artifacts/urdf4_amp_cts_20261113/reward_revision/REPORT.md)。短测确认风格学习信号改善，速度和完整步态尚未验收。

检查点、完整rollout、运行日志和修改前备份保留在本机，不随本次代码提交上传。`design*`、`physical*` 等为参考数据设计/验证工具，使用单独的 MuJoCo 环境，并需要相应模型输入。
