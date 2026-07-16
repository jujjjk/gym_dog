# symmetric-transition 5530 真机部署

当前 5530 parity 路径保留模型的 raw position target。13 Nm 总力矩限幅通过 RS04
motion-control 的 torque feed-forward 字段实现；只有 ±17 Nm 前馈字段仍不足时，才
对位置目标做最小残差修正。CSV 中的 `q_target_safe` 应与 `q_target_raw` 基本一致，
新增 `*_command_torque_ff` 用于核对真实发送的力矩前馈。

这条链路按 `gym_dog/mujoko/sim2sim.py` 对齐：50 Hz 策略、0.45 s 步态、ONNX 动作滤波、动作缩放、10/10/13 Nm 力矩限幅和完整 52 维观测。

真机默认不修改仿真策略行为：

- `/cmd_vel` 暂时没有新消息时，继续使用当前命令，和仿真保持当前 command 的语义一致。
- 倾角保护默认关闭，不用真机姿态阈值接管策略。
- 如需额外保护，可显式设置 `enable_command_timeout_stand_hold:=true` 或 `enable_tilt_protection:=true`；这两项不是 sim2real 默认路径。

## 模型和部署范围

模型命令范围为：`vx=-0.25..0.60 m/s`、`vy=-0.26..0.26 m/s`、`yaw=-1.30..1.30 rad/s`。launch 默认使用完整范围，不再把命令压到低速测试值。

真机全局软件上限为 13 Nm，再与 ONNX 每关节 `10 / 10 / 13 Nm` 限幅取交集，因此最终仍是每关节 `10 / 10 / 13 Nm`。

## 启动

```bash
cd /home/jetson/mydog_ros2_ws
colcon build --packages-select mydog_policy --symlink-install
source install/setup.bash

ros2 run mydog_policy mydog_validate_symmetric_transition_model \
  src/mydog_policy/resource/fanfan_symmetric_transition_5530.onnx

ros2 launch mydog_policy sim2real_symmetric_transition_5530.launch.py \
  enable_send:=true \
  startup_stand_first:=true \
  motor_torque_limit_nm:=13.0 \
  enable_tilt_protection:=false \
  enable_command_timeout_stand_hold:=false \
  debug_csv_path:=/home/jetson/mydog_ros2_ws/log/symmetric_transition_5530.csv
```

## 完整动作矩阵

动作发布器默认使用 `fast` profile：

- 前进：`0.45, 0, 0`
- 后退：`-0.18, 0, 0`
- 左右横移：`0, ±0.15, 0`
- 左右转向：`0, 0, ±0.90`
- 左右斜行：`0.35, ±0.15, 0`
- 左右弧线：`0.35, 0, ±0.90`

连续执行全部动作：

```bash
ros2 run mydog_policy mydog_symmetric_transition_command \
  --profile fast --segment-sec 3 --repeat 1 --rate 20
```

单独执行一个动作：

```bash
ros2 run mydog_policy mydog_symmetric_transition_command \
  --profile fast --action left_yaw --rate 20
```

单独动作不指定 `--segment-sec` 时会持续运行，按 Ctrl+C 停止；发布器退出前会自动连续发送零指令。

可选动作：`forward`、`backward`、`left_lateral`、`right_lateral`、`left_yaw`、`right_yaw`、`left_diagonal`、`right_diagonal`、`left_arc`、`right_arc`。

`--profile low` 只用于低架排查；正常 sim2real 验证使用 `--profile fast`。动作切换期间发布器持续发送 `/cmd_vel`，不会插入零速度空档；序列结束时会显式发送零指令。

## 日志

CSV 会记录 ONNX 原始/滤波动作、步态相位、目标关节角、当前关节角、PD 力矩和发送耗时。若启用姿态保护，还会记录 `projected_gravity_*` 与 `tilt_rad`。
