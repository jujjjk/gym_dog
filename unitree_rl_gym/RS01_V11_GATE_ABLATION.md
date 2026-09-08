# RS01 V11：速度反馈接触门控单变量试验

## 范围

保留 V10、V9 任务和检查点。此次只做奖励去重和门控对照；不新增对称奖励，
不改 RS01 URDF、电机模型、PD、动作限制、命令采样、观测或姿态/路径奖励。
条件化 gait/symmetry 不是此次已经完成的功能，留待这个对照结果明确后再设计。

| 问题 | 代码证据 | 本次处理 | 验证指标 |
|---|---|---|---|
| 接触奖励重复 | V10 的 phase_support_tracking 经合法门控后，与 phase_two_contact_quality 相同 | 1.0 + 0.75 合并为 1.75，旧项归零 | 16 接触掩码 × 3 目标掩码 × 2 gait 状态 × 2 接触时长，收益严格相等 |
| 失配接触时速度收益归零 | V10 tracking = accuracy × legal_contact_gate | A 保留；B 仅去掉 tracking 的门控 | 分方向速度 RMSE、门控归零比例、实际每步奖励贡献 |
| 去门控可能牺牲步态 | 速度准确也可能来自三足支撑或四足停驻 | 所有独立步态/接触惩罚保持一致 | 对角匹配、三足/四足、腾空、原地踏步表现、复位 |

## 两组定义

- A `rs01_omni_v11_hard_gate`：V10 的数学等价去重对照。
- B `rs01_omni_v11_continuous`：仅移除核心速度奖励的接触门控。
- 两组均保留 `pose_error=-0.30`、`tracking_command_velocity=14`。
- `gait_enable=0` 表示站立，`gait_enable=1` 且速度为零表示原地踏步。
- 57 维观测、12 维直接动作，学习率固定 `1e-4`，恢复时不加载旧优化器。
- 连续速度反馈不保证一定改善行为，也不表示已具备实机可靠的长期位置纠偏。

## 短试验协议

共同起点：
`logs/rs01_omni_v9_speed14/Sep05_10-53-43_omni_b_long_from350/model_3350.pt`

先分别用 512 环境做 1 次更新，再从上述共同起点分别用 4096 环境做 200 次更新。
训练种子 `20260909`。一轮冒烟的模型不作为后续短试验起点。
评估采用相同种子 `20260906`、每种命令 8 环境、30 秒、前 2 秒不计稳态指标，
但复位计数包含全部时间；关闭随机化和外力。另测 36 秒连续动作切换。

评估脚本已增加 V11 的奖励时刻诊断，记录发生自动复位之前的：

- `legal_contact_gate_zero_ratio`：V10 合法门控为零的比例；B 中仅用于诊断。
- `mean_ungated_velocity_accuracy`：不乘门控的速度准确度。
- `mean_tracking_reward_per_step`：实际 tracking 输出 × 权重 × dt。
- `mean_contact_quality_reward_per_step`：合并后接触奖励的实际每步贡献。

当前期望接触中，约 70% 时间是对角两足、30% 是四足交接。
因此 exact_diagonal_contact_ratio 的目标上限约 70%，不能把所有非两足时刻都当作非法。
raw torque 是 PD 未限幅请求，不是实际电机扭矩；17 Nm 饱和需单独观察。

## 检查命令

```bash
cd /home/nszb/gym/unitree_rl_gym
source /home/nszb/gym/unitree-rl/bin/activate
python tests/test_rs01_omni_v11.py
python tests/test_rs01_omni_v10.py
```

长训练由用户自行启动。`--max_iterations` 在恢复训练时表示追加更新数，
不是目标检查点编号。两组改了奖励定义，总 reward 不能横向直接比较。

## 长训练命令（实验用途，不表示质量验收通过）

两组短试验已出现持续后退退化，因此目前不建议直接消耗两张卡长训。
以下命令仅供用户决定继续这个对照时使用；没有自动执行。
在两个终端分别运行，均从各自 200 次更新后的检查点再追加 3000 次更新。

GPU 0，硬门控对照：

```bash
cd /home/nszb/gym/unitree_rl_gym
source /home/nszb/gym/unitree-rl/bin/activate
python legged_gym/scripts/train.py \
  --task=rs01_omni_v11_hard_gate \
  --num_envs=4096 --max_iterations=3000 \
  --run_name=v11_hard_long_from3550 --seed=20260909 \
  --resume \
  --load_run=/home/nszb/gym/unitree_rl_gym/logs/rs01_omni_v11_hard_gate/Sep05_18-10-15_gate_ab_pilot200_from_b3350 \
  --checkpoint=3550 --headless \
  --sim_device=cuda:0 --rl_device=cuda:0
```

GPU 1，连续速度反馈试验：

```bash
cd /home/nszb/gym/unitree_rl_gym
source /home/nszb/gym/unitree-rl/bin/activate
python legged_gym/scripts/train.py \
  --task=rs01_omni_v11_continuous \
  --num_envs=4096 --max_iterations=3000 \
  --run_name=v11_continuous_long_from3550 --seed=20260909 \
  --resume \
  --load_run=/home/nszb/gym/unitree_rl_gym/logs/rs01_omni_v11_continuous/Sep05_18-10-16_gate_ab_pilot200_from_b3350 \
  --checkpoint=3550 --headless \
  --sim_device=cuda:1 --rl_device=cuda:1
```
