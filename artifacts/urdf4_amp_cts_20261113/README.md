> 历史记录：当前奖励与启动方式见 [reward_revision/REPORT.md](reward_revision/REPORT.md)。下文为早期全范围双帧AMP版本。

# URDF4 V2 全向 AMP + CTS：代码与冒烟验证

此次交付是训练实现和有限冒烟验证；未启动长训练，没有已验收的全向行走策略。

1. 仓库状态

工作目录 `/home/nszb/gym`，分支 `fix/rs01-heading-odom-soft-inhibit`，起点提交 `999d9527313831ce4f70ded7e15e95240cec0124`。保留已有的 `11.md` 修改及其他用户文件。本次没有提交或推送。原任务 `urdf4_omni_v1`、旧策略与 V1/V2 数据保留。

2. URDF 与物理模型

使用 `unitree_rl_gym/resources/robots/urdf4/urdf/urdf4_rs01.urdf`，SHA256 为 `528c088ff240916b04dfc98611ae9437b92a31cfafc9a297399632e8d0774b91`。读取到总质量 **11.5698648317 kg**、12 个活动关节。FL 大腿关节到膝关节的偏移为 `[-0.16579,0,-0.070098] m`，长度约 0.1800 m；膝到足偏移 `[0.186,0,0.004] m`，长度约 0.1860 m。沿用已验证的关节镜像、足序和碰撞体，没有改 URDF。

3. RS01 控制链

控制周期 20 ms，物理步长 2.5 ms，decimation=8；执行器响应更新 5 ms，40 ms 延迟，31.9 ms 时间常数，响应增益 0.928，摩擦 0.156 Nm。Kp=40、Kd=1，软件电机上限 14 Nm，连续参考值 6 Nm。hip/thigh/calf 目标速率分别为 2/2.6/3.2 rad/s，目标加速度为 60/78/96 rad/s²；动作尺度为 .15/.25/.25 rad，动作截断 ±2。均保留原实现。14 Nm 是现有控制器工作上限，17 Nm 是硬件峰值参考，6 Nm 不是每个瞬间的硬截断。

4. 已有基线证据

V2 数据与此前 Isaac 验证见 `../urdf4_amp_isaac_validation_20261113/REPORT.md` 和 `validation_receipt.json`。此前 12 类参考、每类 60 s × 4 环境的名义跟踪验证，不等于本次学生策略已学会行走。此前测得前进参考约 .291 m/s，关节 RMS 跟踪误差 .0035–.0063 rad，原始 PD 峰值约 10.71 Nm。这次没有加载旧 5400 策略，也没有把参考回放当作训练结果。

5. 问题、证据和改动

| 问题 | 证据 | 改动 | 检查指标 |
|---|---|---|---|
| 需要立即全向 | V2 有前后、横移、转向、斜走、弧线 12 类片段 | 从第 1 轮混合全部 V2 指令和扩大后的连续指令，不做速度课程 | 实际采样 min/max；学生各方向跟踪日志 |
| 高速超出专家范围 | V2 前进约 .28–.29 m/s | 对真假使用同一个最近专家指令类别；D 不使用根部线/角速度特征；专家覆盖外逐渐减弱风格奖励 | .6 与 .28 的条件类别一致、奖励覆盖权重下降 |
| 高速摆腿时间不足 | V2 前进 duty .68；RS01 目标速率有硬上限 | V2 范围内保持每片段 duty；超范围逐渐缩短至 .54，仍大于 .5，保持对角交接窗口 | 所有 V2 指令的 duty 原样保留，高速 duty∈(.5,.56) |
| 直走偏航、横移串方向 | 单纯每步速度奖励不能表达累计偏航 | 朝向目标积分、朝向误差观测与低转速时保持奖励、运动方向垂直速度惩罚、原地转向平移惩罚 | heading/cross velocity/linear/yaw rate errors |
| 静止与原地踏步冲突 | V2 零速度片段为 march，没有真实站立片段 | 独立 gait-enable 观测；站立用任务奖励，禁用该状态的 AMP；支持 `--march` | 站立/踏步采样、重置模式保持 |
| 学生可能依赖特权信息 | 训练时使用打包张量 | 学生编码器仅读取 public history；推理与 JIT 导出隔离教师状态 | 更改全部特权字段后学生动作完全相同 |
| CTS 分支和 PPO 概率混淆 | 两种上下文共同使用一个 actor | 存储 role bit，按原分支重新计算 likelihood，分别归一化并平衡优势 | 更新前 old/new logprob 最大误差约 1.9e-6 |
| 重置污染 AMP | 自动重置发生在 step 内 | 在 compute_reward、自动重置前捕获终态；屏蔽重置、指令变化和模式变化 | mask 与终态路径检查 |

6. 代码位置

- `unitree_rl_gym/legged_gym/envs/urdf4/amp_cts_config.py`：独立任务配置，名义物理模型、全向范围。
- `unitree_rl_gym/legged_gym/envs/urdf4/amp_cts_env.py`：朝向、站立/踏步、V2/高速相位语义、终态 AMP 采集。
- `unitree_rl_gym/legged_gym/envs/urdf4/amp_reference.py`：URDF、数据格式、数据哈希与已有名义验证回执核对。V2 原 manifest 保持不变。
- `unitree_rl_gym/legged_gym/algorithms/urdf4_amp_cts.py`：真实并发教师/学生 PPO、学生估计器监督、AMP、保存/恢复和学生 JIT 导出。
- `unitree_rl_gym/legged_gym/envs/__init__.py`、`utils/task_registry.py`：仅为新任务注册和选择新 runner。
- `unitree_rl_gym/legged_gym/scripts/play.py`：新任务原生回放，默认学生，支持鼠标操作相机和零速度时的 `--march`。
- `tools/urdf4_amp/smoke_cts_isaac.py`：有限集成冒烟、梯度/概率/导出/恢复/录制验证。
- `tools/urdf4_amp/train_cts.sh`：用户手动执行的前台长训练入口。

7. 配置与训练语义

| 项目 | 取值 |
|---|---|
| 新任务 | `urdf4_amp_cts_v1` |
| 机身坐标指令 vx | [-.4,.6] m/s，即前进 0–.6、后退到 -.4 |
| vy / wz | [-.4,.4] m/s / [-1.2,1.2] rad/s |
| 采样 | 40% V2 指令、10%站立、10%踏步、40%连续全范围；连续部分一半为单轴 |
| public frame / history | 51 × 20 = 1020；最新帧在前，重置时填充首帧 |
| public 新增字段 | heading error sin、cos−1、gait-enable |
| 教师状态 / 上下文 | 99 / 真实速度3+教师latent16 |
| 学生上下文 | 历史1020 → 估计速度3+latent16 |
| 共享 actor / critic | [512,256,128]；动作12维 |
| CTS 环境比例 | 教师75%、学生25%；PPO优势按分支归一化并平衡 |
| 梯度 | PPO更新共享策略和教师编码器；学生编码器从学生访问状态拟合 stop-gradient 教师上下文，独立优化器 |
| PPO | 24步/轮、4 epochs、4 minibatches、初始LR 3e-4、adaptive KL |
| AMP | 46维相邻状态对；conditional LS-GAN+gradient penalty，2步/轮；最大系数.6，按dt计，100轮warmup |
| 名义阶段 | 无摩擦/质量/电机模型随机化、无传感器噪声、无外力推扰；保留重置关节位置±.02 rad |
| 相位 | 2 Hz；FL+RR 和 FR+RL 错开半周期；非法双脚、单脚和腾空受罚 |

模型检查点保存策略、教师/学生编码器、PPO/学生/判别器优化器及 AMP 正规化和数据哈希。导出只含学生编码器和共享 actor：`RUN/exported/student_N.pt`，附 JSON 输入/动作/控制器契约。恢复不是逐随机数复现：环境现场状态和判别器 replay buffer 会重新建立。

8. 验证命令与证据

```bash
cd /home/nszb/gym
source unitree-rl/bin/activate
python tools/urdf4_amp/smoke_cts_isaac.py \
  --output artifacts/urdf4_amp_cts_20261113/final128 --updates 3 \
  --num_envs 128 --headless --sim_device cuda:0 --rl_device cuda:0
python tools/urdf4_amp/smoke_cts_isaac.py \
  --output artifacts/urdf4_amp_cts_20261113/final4096 --updates 1 \
  --num_envs 4096 --headless --sim_device cuda:0 --rl_device cuda:0
```

结果为 `final128/result.json`、`final4096/result.json`；固定 [.28,0,0] 指令的学生录制各 1 s，seed=1，见各目录 `student_recorder.npz`。录制同时包含状态、关节、动作、接触、raw PD/motor/applied 三种扭矩。它只验证记录器和数值链路，不能作为行走验收。

另外通过标准 `train.py` 完成 4096 环境×1轮入口冒烟；从 model_3 恢复后用标准入口追加 1 轮，正常保存 model_4；标准 `play.py` 加载学生完成 headless 入口检查。长训练没有启动。旧任务 `tests.test_urdf4_omni` 的 13 项兼容性测试全部通过（`regression_tests.log`）；最终 `train_cts.sh` 也用覆盖参数运行了128环境×1轮入口检查（`launcher_smoke.log`）。

9. 长训练与回放命令

手动启动，4096 环境、8000 轮、GPU0、seed1：

```bash
cd /home/nszb/gym
bash tools/urdf4_amp/train_cts.sh
```

等价直接命令：

```bash
cd /home/nszb/gym/unitree_rl_gym
source ../unitree-rl/bin/activate
python legged_gym/scripts/train.py \
  --task urdf4_amp_cts_v1 --num_envs 4096 --max_iterations 8000 \
  --run_name v2_omni_nominal_cts_seed1 --seed 1 \
  --headless --sim_device cuda:0 --rl_device cuda:0
```

日志为 `unitree_rl_gym/logs/urdf4_amp_cts_v1/时间_v2_omni_nominal_cts_seed1/`，每100轮保存检查点和学生导出。`tracking/student_forward`、`student_backward`、`student_left/right`、`student_turn_left/right`、`student_diagonal`、`student_arc`、`student_stand/march` 均统计学生分支，含 linear_error_m_s / yaw_rate_error_rad_s / cross_velocity_m_s / heading_error_rad 和样本数。样本数为0时不输出该组均值。

训练后用实际运行目录名替换 `实际运行目录`，查看原生 Gym 全向效果：

```bash
python legged_gym/scripts/play.py \
  --task urdf4_amp_cts_v1 --load_run 实际运行目录 --checkpoint 8000 \
  --num_envs 1 --sweep --duration_s 100 \
  --sim_device cuda:0 --rl_device cuda:0
```

相机不自动跟随，保留鼠标拖动。固定动作改为 `--vx .6 --vy 0 --wz 0 --duration_s 30`，去掉 `--sweep`。站立使用三项均0；原地踏步另外加 `--march`。

恢复训练（`max_iterations` 为追加轮数）：

```bash
bash /home/nszb/gym/tools/urdf4_amp/train_cts.sh \
  --resume --load_run 实际运行目录 --checkpoint 4000 --max_iterations 4000
```

10. 冒烟评估表

| 检查 | 128环境×3轮 | 4096环境×1轮 |
|---|---|---|
| PPO/CTS/AMP 有限损失 | 通过 | 通过 |
| 教师/共享actor更新，PPO不更新学生编码器 | 通过 | 通过 |
| 学生监督更新、教师target停止梯度 | 通过 | 通过 |
| 学生无特权信息泄漏 | 通过 | 通过 |
| 学生导出动作最大误差 | 0 | 0 |
| 检查点恢复动作最大误差 | 0 | 0 |
| 站立/踏步、指令/重置mask、正负朝向积分 | 通过 | 通过 |
| 不匹配验证回执拒绝 | 通过 | 通过 |
| V2 duty保留/高速增加摆腿时间 | 通过 | 通过 |
| 1 s固定指令学生记录器重置数 | 0 | 0 |
| 学会目标速度和步态 | 未验证 | 未验证 |

完整数值和检查点路径见 JSON。4096 冒烟的 PyTorch 峰值张量分配约 1772 MiB；不包含全部 PhysX/驱动开销，不当作总显存测量。

11. 剩余限制和后续验收

.6 m/s、横移.4 m/s、转向1.2 rad/s 是训练指令范围，尚无训练结果证明达到；组合范围角点也不保证可执行。V2专家没有扩大到这些速度，不能靠这批数据直接声称高速物理验证通过。高速 duty 自适应只是训练调度设计，真实对角落地和双支撑还要测量。下一步先对学生策略逐项测试：全部12类V2动作及高速前进、后退、横移、转向；建议每项30–60 s，检查重置/跌倒、速度与侧偏、朝向、步序/腾空、raw和电机扭矩、连续6 Nm占比和峰值截断。确认名义运动后再增加随机化、推扰和传感器误差。此次未作实机或鲁棒性验收。

12. 回退

继续使用原任务 `urdf4_omni_v1` 及原检查点即可。新任务、数据读取和日志目录独立；无需覆盖旧策略，不要用全仓库 reset/clean 删除用户已有内容。
