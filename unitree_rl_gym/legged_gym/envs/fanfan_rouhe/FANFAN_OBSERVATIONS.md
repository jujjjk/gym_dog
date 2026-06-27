# Fanfan Rouhe 50维观测说明

本文档对应：

- `legged_gym/envs/fanfan_rouhe/fanfan_config.py`
- `legged_gym/envs/fanfan_rouhe/fanfan_env.py`
- `FanfanRouheRoughCfg.env.num_observations = 50`

## 总体结构

策略观测由以下数据依次拼接：

```python
obs = [
    base_lin_vel * 2.0,                 # 3
    base_ang_vel * 0.25,                # 3
    projected_gravity,                  # 3
    commands[:3] * [2.0, 2.0, 0.25],   # 3
    (dof_pos - default_dof_pos) * 1.0,  # 12
    dof_vel * 0.05,                     # 12
    actions,                            # 12
    sin(2*pi*gait_phase),               # 1
    cos(2*pi*gait_phase),               # 1
]
```

总维度：

```text
3 + 3 + 3 + 3 + 12 + 12 + 12 + 2 = 50
```

## 索引表

索引采用 Python 的 0-based 规则，区间右端不包含。

| 索引 | 维数 | 内容 | 坐标系/单位 | 缩放 |
|---|---:|---|---|---:|
| `[0:3]` | 3 | 机身线速度 `vx, vy, vz` | 机身坐标系，m/s | `2.0` |
| `[3:6]` | 3 | 机身角速度 `wx, wy, wz` | 机身坐标系，rad/s | `0.25` |
| `[6:9]` | 3 | 重力方向投影 `gx, gy, gz` | 机身坐标系，无量纲 | `1.0` |
| `[9:12]` | 3 | 速度命令 `vx_cmd, vy_cmd, yaw_cmd` | m/s、m/s、rad/s | `2.0, 2.0, 0.25` |
| `[12:24]` | 12 | 关节位置相对默认站姿的偏差 | rad | `1.0` |
| `[24:36]` | 12 | 关节速度 | rad/s | `0.05` |
| `[36:48]` | 12 | 上一个控制步的策略动作 | 无量纲 | 已裁剪到 `[-1, 1]` |
| `[48]` | 1 | `sin(2*pi*gait_phase)` | 无量纲 | `1.0` |
| `[49]` | 1 | `cos(2*pi*gait_phase)` | 无量纲 | `1.0` |

## 12个关节的排列

Isaac Gym 当前从 URDF 读取到的关节顺序为：

| 关节槽位 | 关节名称 | 位置观测索引 | 速度观测索引 | 动作观测索引 |
|---:|---|---:|---:|---:|
| 0 | `FL_hip_joint` | 12 | 24 | 36 |
| 1 | `FL_thigh_joint` | 13 | 25 | 37 |
| 2 | `FL_calf_joint` | 14 | 26 | 38 |
| 3 | `FR_hip_joint` | 15 | 27 | 39 |
| 4 | `FR_thigh_joint` | 16 | 28 | 40 |
| 5 | `FR_calf_joint` | 17 | 29 | 41 |
| 6 | `RL_hip_joint` | 18 | 30 | 42 |
| 7 | `RL_thigh_joint` | 19 | 31 | 43 |
| 8 | `RL_calf_joint` | 20 | 32 | 44 |
| 9 | `RR_hip_joint` | 21 | 33 | 45 |
| 10 | `RR_thigh_joint` | 22 | 34 | 46 |
| 11 | `RR_calf_joint` | 23 | 35 | 47 |

关节位置不是绝对角度，而是：

```python
dof_position_observation = dof_pos - default_dof_pos
```

当前默认站姿为：

```text
hip   =  0.000 rad
thigh =  0.563 rad
calf  = -0.950 rad
```

## 步态相位

步态周期为：

```text
gait_period = 0.54 s
```

基础相位：

```python
gait_phase = (episode_time % gait_period) / gait_period
```

相位范围为 `[0, 1)`。最后两维使用正弦和余弦编码，避免相位从
`0.999` 跳回 `0.0` 时产生不连续输入。

对角腿分组：

```text
A组：FL + RR，相位偏移 0.0
B组：FR + RL，相位偏移 0.5
```

因此两组腿相差半个周期，用于生成对角小跑步态。

## 动作与关节目标

策略输出先裁剪：

```python
actions = clip(policy_output, -1.0, 1.0)
```

RL 残差目标幅度：

```python
hip_action_target = hip_actions * 0.05
front_thigh_calf_target = front_thigh_calf_actions * 0.12
rear_thigh_calf_target = rear_thigh_calf_actions * 0.14
```

最终 PD 目标还包含默认站姿和对角步态参考偏移：

```python
target_dof_pos = (
    default_dof_pos
    + gait_reference_offset
    + scaled_actions
)
```

其中 `scaled_actions` 对 hip 使用 `0.05`，前腿 thigh/calf 使用 `0.12`，
后腿 thigh/calf 使用 `0.14`。后腿拥有稍大的前后摆动修正范围；hip
仍保持较小幅度，用于抑制左右高速打髋和机身 yaw 甩尾。

策略的原始高斯输出在进入环境前通过 `tanh` 平滑映射到 `[-1, 1]`。
这避免硬裁剪导致大量动作长期卡在 `-1` 或 `1`。

参考轨迹包含完整的支撑和摆动循环。大腿在支撑期从前向后扫，在摆动期
平滑向前收；小腿只在摆动期屈曲抬脚。参考偏移峰值为：

```text
thigh offset = 0
calf offset  = -0.22 rad
```

参考相位只负责交替屈小腿抬脚，不强制 thigh 的前后摆动方向。
前后推进由策略根据速度命令学习，避免 URDF 实际接触方向与简化运动学不一致。
摆动曲线使用三次平滑相位，抬脚和落脚端点的速度均为零，减少快速换腿时
对机身的冲击。

支撑相占周期的 `0.62`，FL+RR 与 FR+RL 保持相差半周期的对角关系。
不对策略动作增加跨控制帧低通，避免滤波延迟破坏对角腿换相。

`gait_reference_offset` 没有作为独立观测输入。策略通过观测
`[48:50]` 的相位编码判断当前参考轨迹处于哪个阶段。

## 训练噪声

训练开启观测噪声时，各部分最大噪声幅度为：

| 观测 | 最大均匀噪声 |
|---|---:|
| 缩放后的机身线速度 | `+-0.20` |
| 缩放后的机身角速度 | `+-0.05` |
| 重力投影 | `+-0.05` |
| 速度命令 | `0` |
| 关节位置偏差 | `+-0.01` |
| 缩放后的关节速度 | `+-0.075` |
| 上一步动作 | `0` |
| 步态相位 sin/cos | `0` |

所有观测最终会裁剪到：

```text
[-100, 100]
```

## 未包含的数据

当前 50 维策略观测不包含：

- 机身绝对位置或绝对高度
- roll、pitch、yaw 欧拉角
- 足端位置和足端速度
- 足端接触力或接触状态
- 地形高度扫描
- 电机力矩

其中姿态信息主要由重力投影表示；足端状态仅用于奖励计算，不直接输入策略。
