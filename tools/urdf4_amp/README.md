# URDF4 V2 AMP + CTS

当前任务为 `urdf4_amp_cts_v1`，版本 `v3_rsi_pair_amp`，实验目录 `urdf4_amp_cts_rsi`。先学习 V2 覆盖的12类运动，0–0.6 m/s 连续全向和鲁棒性尚未验收。

用户后续明确选择“先学步态、后做速度追踪”，对应独立入口：

```bash
cd /home/nszb/gym
source unitree-rl/bin/activate
python tools/urdf4_amp/train_cts_gait_first.py \
  --max_iterations 30000 --num_envs 4096 --seed 1 \
  --headless --sim_device cuda:0 --rl_device cuda:0
```

默认从 `rsi_bootstrap/trust300/run/model_100.pt` 续训到总计30000轮（新增29900轮），实验目录 `urdf4_amp_cts_gait_first`。该入口把显式指令追踪奖励设为0，保留AMP、CTS、原有动作偏离约束和支撑/执行器约束；速度仍统计，不作为该阶段长训启动条件。这是用户明确要求的新训练阶段，不生成或修改动作验收通过回执。优化器状态重新初始化，模型与AMP判别器继续加载；RS01参数不变。可以用 `--load_run /absolute/run/path --checkpoint N` 续训本阶段检查点，`--max_iterations` 仍表示目标总轮数。

流程为实际 RS01 控制指令预热神经网络，再用相邻状态 AMP + CTS/PPO 优化。不能把参考关节位置直接当作 actor 标签。导出的学生网络只读取公共历史，不调用参考控制器。

RSI 同时恢复关节、浮动基座、目标限速器、延迟队列、电机响应、相位和观测历史。RS01 模型、PD、限速和延迟不变。足端参考几何仅统计，风格由 AMP 学习。PPO 使用一项相对预热网络的动作偏离约束，防止短训破坏已有步态。

长训入口：

```bash
bash tools/urdf4_amp/train_cts.sh
```

默认4096环境、8000轮、GPU0，只续训已验收模型。缺少通过回执，或模型/源代码/数据契约不匹配时拒绝启动。不要直接续训旧版本检查点。

本仓库收录训练必需的 V2 数据和名义验证回执：

- `artifacts/urdf4_amp_design_20261113_v2/amp_reference_v2/`
- `artifacts/urdf4_amp_isaac_validation_20261113/validation_receipt.json`

数据原 manifest 保持不变；加载时核对 URDF、全部片段哈希及验证回执。回执只证明此前名义验证通过，不代表新策略或实机验收。

在仓库根目录、已配置的 Isaac Gym 环境中执行有限检查：

```bash
source unitree-rl/bin/activate
python tools/urdf4_amp/smoke_cts_isaac.py \
  --output /tmp/urdf4_cts_smoke --updates 3 --num_envs 128 \
  --headless --sim_device cuda:0 --rl_device cuda:0
python tools/urdf4_amp/evaluate_cts_behavior.py \
  --checkpoint /absolute/path/model_N.pt --output /tmp/urdf4_behavior \
  --seconds 30 --seeds 1,2,3 --modes cold,rsi \
  --headless --sim_device cuda:0 --rl_device cuda:0
```

`play.py --task urdf4_amp_cts_v1 --sweep` 使用原生 Gym 播放学生策略，切换12个V2指令及站立，保留鼠标相机操作；零速度加 `--march` 可查看原地踏步。

`--updates N` 在评估前执行有限 PPO 优化，单次上限500轮；`--approve` 只有通过完整验收才写入长训回执。验收覆盖12类动作、每类4个环境、3个种子、冷启动与 RSI 启动。检查速度/转速、抬脚、真实迈步、对角交替与落地时差、腾空、非法支撑、重置和原始扭矩，不能只凭 AMP 分数放行。

本机预热数据在 `artifacts/urdf4_amp_cts_20261113/rsi_bootstrap/`。重建入口 `prepare_cts_bootstrap.py` 依赖 `artifacts/urdf4_amp_isaac_validation_20261113/commands60/` 的真实控制记录；随后用 `refit_cts_seed.py` 拟合周期控制目标。重建会改变状态库哈希，应重新生成模型与验收记录，保留旧库和旧模型。

本轮结果和限制见 [验证记录](../../artifacts/urdf4_amp_cts_20261113/rsi_bootstrap/REPORT.md)。旧版150轮风格检查不构成当前行为验收。

检查点、完整rollout、运行日志和修改前备份保留在本机，不随本次代码提交上传。`design*`、`physical*` 等为参考数据设计/验证工具，使用单独的 MuJoCo 环境，并需要相应模型输入。

新机器仅克隆代码后，还需要从本机复制配套的 `rsi_bootstrap/reference_bank.pt` 和选定检查点（默认 `rsi_bootstrap/trust300/run/model_100.pt`），或根据真实控制记录重新准备状态库与模型。状态库和模型必须使用同一契约，不能混配。
