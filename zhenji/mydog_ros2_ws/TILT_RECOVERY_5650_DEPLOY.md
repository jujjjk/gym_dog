# 5650 侧倾恢复模型真机验证

此版本从原始 5530 继续训练，部署使用新跑次 checkpoint 5650。它通过模型哈希、任务名、
52 维观测、左右对称、50 Hz、PD 参数和 10/10/13 Nm 力矩契约的启动校验。

MuJoCo 严格筛选仍有三项未通过：动作饱和周期 9.67%（要求 <5%）、反向过冲
4.29°（要求 <3°）、左右扰动峰值差 3.14°（要求 <1.5°）。因此此版本只批准
架空/低架保护测试，尚未批准无保护落地运行；
生产入口 `sim2real_realdata.launch.py` 仍保留原始 5530。

## 构建和模型校验

```bash
cd /home/jetson/mydog_ros2_ws
source /opt/ros/humble/setup.bash
python3 -c "import numpy, requests, onnxruntime, YbImuLib"
colcon build --symlink-install --packages-select mydog_policy
source install/setup.bash

ros2 run mydog_policy mydog_validate_tilt_recovery_model \
  "$(ros2 pkg prefix mydog_policy)/share/mydog_policy/models/fanfan_tilt_recovery_5530_5650.onnx"
```

预期 SHA256：

```text
8f370f9fd1165774426d80eef72391152137d28dc6b429af57af96284e68893f
```

## 不下发电机的预演

机器人架空并准备硬件急停：

```bash
mkdir -p /home/jetson/mydog_ros2_ws/log
ros2 launch mydog_policy sim2real_tilt_recovery_5530.launch.py \
  enable_send:=false \
  print_only:=true \
  startup_stand_first:=false
```

确认模型契约通过、IMU 重力方向接近 `[0, 0, -1]`、12 个电机在线、控制频率
49–51 Hz、状态延迟不超过 100 ms。

## 架空下发与低架测试

```bash
ros2 launch mydog_policy sim2real_tilt_recovery_5530.launch.py \
  enable_send:=true \
  print_only:=false \
  startup_stand_first:=true \
  debug_csv_path:=/home/jetson/mydog_ros2_ws/log/tilt5650_$(date +%Y%m%d_%H%M%S).csv
```

等待启动站立完成后，从 `vx=0.12` 开始：

```bash
ros2 topic pub -r 50 /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.12, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
```

启动文件默认把命令限制为 `vx=-0.15..0.35`、`vy=-0.12..0.12`、
`yaw=-0.75..0.75`。策略检测到超过 3° 的倾斜后进入恢复；回到 2° 内并持续
10 个控制周期（0.2 秒）时，清空异常 previous-action 和动作滤波状态，同时保留步态
相位与航向目标，避免水平后仍停留在异常饱和步态。

任一关节方向错误、持续限扭、IMU/电机状态超时、机身超过保护角或恢复后继续异常摆腿，
立即硬件急停，不要继续扩大速度范围。
