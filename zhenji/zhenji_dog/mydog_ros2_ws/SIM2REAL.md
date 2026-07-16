# Fanfan ONNX Sim2Real（Jetson / ROS 2）

目标模型：`FanfanStraightPD8Cfg / fanfan_straight_pd8_best.onnx`（`model_1100.pt`）

```text
SHA256 e12d25d718e6afe12dc84368341f42dbb9eb156e1a23aea83bdc9982410450e3
```

当前 `policy.onnx` 自带 `fanfan_deployment_config`。ROS 2 节点会从模型读取并校验：

- 52 维观测与 12 维动作；
- `FL, FR, RL, RR` 关节顺序和默认关节角；
- 观测缩放、动作缩放及 `tanh` 输出变换；
- 50 Hz 控制频率；
- 0.54 s 步态周期、对角腿相位和小腿参考轨迹；
- 航向误差的正余弦观测。

不要同时启动其他会占用 `/dev/myimu` 的节点。`sim2real.launch.py` 已经同时启动状态估计器和策略节点，由状态估计器独占 IMU。

## 1. 同步和构建

把本机 `mydog_ros2_ws/src/mydog_policy` 同步到 Jetson 的同一目录，然后执行：

```bash
cd /home/jetson/mydog_ros2_ws
python3 -c "import numpy, requests, onnxruntime; print(onnxruntime.__version__)"
colcon build --packages-select mydog_policy --symlink-install
source install/setup.bash
```

## 2. 架空腿 dry-run（不发电机命令）

机器人必须架空，并准备硬件急停：

```bash
ros2 launch mydog_policy sim2real.launch.py \
  enable_send:=false \
  startup_stand_first:=false \
  debug_print_arrays:=false \
  debug_csv_async:=true
```

另开终端检查：

```bash
source /home/jetson/mydog_ros2_ws/install/setup.bash
ros2 topic echo /mydog/policy_obs --once
ros2 topic echo /mydog/policy_target --once
```

必须确认日志中出现 `observations [batch, 52]`、模型契约加载成功、所有电机在线且反馈不超过 100 ms。静止水平时重力观测应接近 `[0, 0, -1]`；默认姿态附近的 12 个关节位置误差应接近 0。

随后在 `enable_send:=false` 状态发布一次真实训练域命令，确认日志中的命令最终为
`[0.1, 0.0, 0.0]`，横移和转向没有混入：

```bash
ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.15, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
```

## 3. 架空腿低风险发令

先让电机服务进入可控状态，再启动：

```bash
ros2 launch mydog_policy sim2real.launch.py \
  enable_send:=true \
  startup_stand_first:=true \
  use_model_pd_gains:=true \
  model_kp_scale:=1.0 \
  model_kd_scale:=1.0 \
  motor_torque_limit_nm:=6.0 \
  calf_target_rate_mul:=3.0 \
  calf_target_accel_mul:=3.0 \
  debug_csv_period_sec:=0.1
```

启动后不要发布 `/cmd_vel`。节点会先从实时反馈保持 0.5 秒，再用低刚度
`Kp=12, Kd=2` 和 `0.12/0.15 rad/s` 的关节限速，在约 12 秒内过渡到 ONNX
元数据中的默认站姿。只有日志出现下面一行才允许继续：

进入策略阶段后，`use_model_pd_gains:=true` 会忽略标量回退值 `send_kp/send_kd`，
读取 ONNX 中的12维分关节增益。PD8 模型已经按真机需要完成阻尼训练，因此必须保持
`model_kp_scale=1.0`、`model_kd_scale=1.0`；有效增益为髋关节 `60/1.2`、
大腿和小腿 `70/1.6`，不能再额外把 Kd 乘 2。只有显式设置
`use_model_pd_gains:=false` 时才使用标量回退值。

模型内嵌训练位置误差上限：髋/大腿/小腿分别为 `0.133/0.137/0.183 rad`。
部署端同时保留 6 Nm 真机扭矩预算，最终使用“模型误差上限”和“真机预算推导上限”
中更严格的一项，不会因为换用 PD8 模型而放宽真机保护。

部署关节限位与更新后的仿真 URDF 对齐：髋关节 `[-0.8, 0.8]`、大腿
`[-0.5, 4.0]`、小腿 `[-2.7, -0.85]` rad，均为训练/策略坐标。

```text
[STARTUP_STAND][READY] ONNX default pose is stable.
```

如果出现 `[STARTUP_STAND][FAULT]`，不要启动策略命令；保持吊带支撑并检查日志中的
跟踪误差、力矩或超时原因。

策略只训练了前进速度 `0.10-0.20 m/s`，没有训练横移和转向。第一次测试从
`0.10 m/s` 开始；部署节点会把横移、转向硬限制为零，并拒绝错误模型哈希。

看到 `READY` 后再持续发布命令（停止发布超过 0.5 秒后策略不再发送电机目标）：

```bash
ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.15, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
```

注意：停止发布或发送全零命令只会禁止新的策略目标，不等同于主动阻尼或断使能；
下位机可能保持最后目标。结束测试应先保持吊带支撑，再用硬件急停/电机服务的停止流程，
不要把 `/cmd_vel=0` 当作急停。

观察关节方向、对角腿关系 `FL+RR / FR+RL`、电机温度、实际力矩和目标误差。任一关节方向相反、连续触发目标限幅、反馈超时或 IMU 轴向错误时立即急停，不要落地。

## 4. 落地测试

新参数必须先通过架空测试，已有 `air_split_pd_20260630_172820.csv` **不能直接作为落地依据**。策略阶段使用分关节 Kp 和真机阻尼缩放后的 Kd；启动站姿仍单独使用低增益 `12/2`。至少确认 `[LOOP_RATE]` 稳定在 49-51 Hz、小腿速度峰值明显低于上一轮的 `4.10 rad/s`、没有持续高频抖动，并且对角相位肉眼正确，再使用吊带卸载大部分重量做 1-2 秒触地测试。Sim2real 启动文件的架空验证上限保持 6 Nm；ONNX 内的 17 Nm 是训练模型峰值参数，不会直接覆盖真机安全上限。

本模型的 `/cmd_vel.angular.z` 必须保持为零。52维观测末尾仍包含相对启动航向的
`sin/cos` 误差，供策略自行纠正直行偏航，但训练契约没有开放转向命令。

## 5. CSV 录制与 sim2real 分析

训练语义与真机语义由 `JointSemanticMapper` 桥接（`joint_sign`、电机顺序等），**不要**直接把
`policy_joint_name` 下的角度和 `joint_name`（真机）角度数值对比。

`sim2real.launch.py` 默认写 CSV：

```text
/home/jetson/mydog_ros2_ws/log/sim2real_debug.csv
```

控制循环仍为 50 Hz；CSV 默认按 0.1 秒抽样（约 10 Hz）以避免磁盘写入拖慢控制，含 `startup_stand` 与策略行走全程。实际控制频率以终端中的 `[LOOP_RATE]` 为准，不能用 CSV 相邻时间戳判断。自定义路径：

```bash
ros2 launch mydog_policy sim2real.launch.py \
  enable_send:=true \
  startup_stand_first:=true \
  use_model_pd_gains:=true \
  model_kp_scale:=1.0 \
  model_kd_scale:=1.0 \
  motor_torque_limit_nm:=6.0 \
  calf_target_rate_mul:=3.0 \
  calf_target_accel_mul:=3.0 \
  debug_csv_path:=/home/jetson/mydog_ros2_ws/log/sim2real_$(date +%Y%m%d_%H%M%S).csv
```

流程：启动 → 等 `[STARTUP_STAND][READY]` → 持续发 `/cmd_vel` 至少 10 s → 停节点。

分析（Jetson 或开发机均可）：

```bash
python3 src/mydog_policy/tools/analyze_sim2real_csv.py \
  log/sim2real_debug.csv \
  --skip-sec 2.0 --min-cmd-x 0.15 --include-startup
```

或通用关节表：

```bash
python3 src/mydog_policy/tools/analyze_policy_debug_csv.py \
  log/sim2real_debug.csv --skip-sec 2.0 --min-cmd-x 0.15
```

### CSV 关键列（判断小腿为何不动）

| 列名 | 空间 | 含义 |
|------|------|------|
| `policy_joint_name` | 训练契约 | ONNX 关节名（FL,FR,RL,RR 顺序） |
| `joint_name` / `q_*_real` | 真机电机 | FR,FL,RL,RR 电机反馈与目标 |
| `gait_ref_policy` | 训练契约 | sim2sim 同款步态参考（小腿摆动主来源） |
| `rl_action_contrib_policy` | 训练契约 | 神经网络动作 × `action_scale` |
| `q_final_target_minus_default_policy` | 训练契约 | 下发前最终目标相对默认站姿 |
| `q_current_minus_default_policy` | 训练契约 | 反馈在训练空间的偏差 |
| `rate_limited` / `final_limited_joint_mask` | 真机 | 部署安全限速、力矩限幅 |
| `mode` | — | `startup_stand` / `pure_rl` 等 |

**判读口诀：**

1. `gait_ref_policy` 摆幅小 → 步态相位或 `/cmd_vel` 未持续发布  
2. `gait_ref_policy` 正常、`q_current_real`（小腿）不动 → 电机/力矩/跟踪问题  
3. `final_limited` 高 → 先确认控制频率和 Kd；不要直接提高扭矩或放宽限幅  

单关节探针（确认 0x13/0x23/0x33/0x43 小腿电机独立可动）：

```bash
ros2 launch mydog_policy policy_walk.launch.py \
  joint_probe_enable:=true joint_probe_name:=FR_calf_joint \
  joint_probe_delta_rad:=0.08 enable_send:=true \
  debug_csv_path:=/home/jetson/mydog_ros2_ws/log/calf_probe_FR.csv
```

## 6. 同步到 Jetson

```bash
rsync -avz --delete \
  ./mydog_ros2_ws/src/mydog_policy/ \
  jetson:/home/jetson/mydog_ros2_ws/src/mydog_policy/

ssh jetson 'cd /home/jetson/mydog_ros2_ws && colcon build --packages-select mydog_policy --symlink-install'
```
