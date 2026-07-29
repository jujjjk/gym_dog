# 最终 hardware-balance 模型真机部署

本入口只部署已通过 Gym/MuJoCo 验收的最终模型：

```text
fanfan_hardware_balance_5530_best.onnx
SHA256 2f834c76f0297b13230c04ce53ed707d1f614416e0e957862303187b3d59ed2f
```

运行时严格使用 52 维观测、12 维动作、50 Hz、0.45 s 步态周期，PD 为
`hip 60/1.2, thigh 70/1.6, calf 70/1.6`，逐关节力矩上限为每条腿
`10/10/13 Nm`。ONNX 负责速度/航向反馈、左右对称和动作输出；ROS 节点
负责动作滤波、参考步态、连续动作切换，以及与 Gym/MuJoCo 完全一致的后退
保护：`vx < -0.03` 时后腿 calf 目标不得小于 `-1.38 rad`。

## 1. 真机服务前置检查

本部署要求当前仓库中的 RS01 参数读回固件和 `text/app.py`。两块 STM32 的
`board_capabilities` 都必须为 `3`，否则节点会拒绝使能电机：

```bash
curl -s http://127.0.0.1:8000/api/state | \
  python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["0x11"]["board_capabilities"], d["0x31"]["board_capabilities"])'
```

确认 IMU 串口未被旧程序占用：

```bash
readlink -f /dev/myimu
sudo lsof "$(readlink -f /dev/myimu)"
```

只启动一个电机服务进程：

```bash
sudo pkill -f 'uvicorn app:app' || true
cd /home/jetson/text
sudo -E env LINGZU_STATE_REFRESH_HZ=50 \
  python3 -m uvicorn app:app --host 0.0.0.0 --port 8000
```

## 2. 构建和模型强校验

```bash
cd /home/jetson/mydog_ros2_ws
source /opt/ros/humble/setup.bash
rm -rf build/mydog_policy install/mydog_policy
colcon build --packages-select mydog_policy --symlink-install
source install/setup.bash

ros2 run mydog_policy mydog_validate_hardware_balance_model \
  src/mydog_policy/resource/fanfan_hardware_balance_5530_best.onnx
```

必须看到 `hardware-balance ONNX validation PASSED` 和本文开头的 SHA256。

## 3. 不发电机命令的预演

```bash
mkdir -p /home/jetson/mydog_ros2_ws/log
ros2 launch mydog_policy sim2real_hardware_balance.launch.py \
  enable_send:=false \
  debug_print_arrays:=true
```

确认 12 个电机全部在线、IMU 直立重力接近 `[0,0,-1]`，日志中模型 task 为
`FanfanOmniHardwareBalance5530V2Cfg`，且无哈希、关节映射、状态时序错误。

## 4. 悬空低速验收

机器人可靠悬空、急停可随时触发后启动真机输出：

```bash
ros2 launch mydog_policy sim2real_hardware_balance.launch.py \
  enable_send:=true \
  debug_csv_path:=/home/jetson/mydog_ros2_ws/log/hardware_balance_rack.csv
```

节点会先 STOP，再写入并读回全部 12 个电机的 `10/10/13 Nm` 和
`12/12/16 Apk` 限制，随后一次性使能。必须依次看到：

```text
RS01 volatile safety limits accepted
readback_verified=True
[STARTUP_STAND] one-shot live-target prime and all-motor ENABLE acknowledged
```

硬件标零后的悬空数据表明前髋保持中位约需 `1.2 Nm`，因此本入口的启动站姿
使用低增益 `18/2`，并保持 12 秒平滑过渡、`0.08 rad` 接管误差门限以及全部
原有力矩和跟踪故障保护。策略正式接管后仍严格使用模型导出的分关节 PD。

首次卸载触地数据还表明，从零命令直接打开完整的 `0.30 rad` 小腿参考会产生
`5.64–6.52 rad/s` 速度峰值。真机入口因此启用已有的命令幅度平滑门控；
`low forward (vx=0.12)` 的动作和参考步态约以 `13%` 幅度起步，而训练范围上限
仍可达到完整幅度。模型、分关节 PD 和 `10/10/13 Nm` 限幅均保持不变。

另开终端先逐个执行低速动作，每项 3 秒：

```bash
source /opt/ros/humble/setup.bash
source /home/jetson/mydog_ros2_ws/install/setup.bash

ros2 run mydog_policy mydog_hardware_balance_command \
  --profile low --action forward --segment-sec 3
ros2 run mydog_policy mydog_hardware_balance_command \
  --profile low --action backward --segment-sec 3
ros2 run mydog_policy mydog_hardware_balance_command \
  --profile low --action left_lateral --segment-sec 3
ros2 run mydog_policy mydog_hardware_balance_command \
  --profile low --action right_lateral --segment-sec 3
ros2 run mydog_policy mydog_hardware_balance_command \
  --profile low --action left_yaw --segment-sec 3
ros2 run mydog_policy mydog_hardware_balance_command \
  --profile low --action right_yaw --segment-sec 3
```

启动参数固定启用 `0.45 rad` 倾角保护、0.5 s 命令超时站立保持、100 ms 电机
状态时效检查和状态连续丢失后的 `/api/stop` 锁存。不要关闭这些保护。

## 5. 落地后的推荐命令

先使用低速动作确认足端落地与关节方向，再使用 nominal。横移和后退仍保守：

```bash
ros2 run mydog_policy mydog_hardware_balance_command \
  --profile nominal --action forward
ros2 run mydog_policy mydog_hardware_balance_command \
  --profile nominal --action left_lateral
ros2 run mydog_policy mydog_hardware_balance_command \
  --profile nominal --action backward
```

单动作省略 `--segment-sec` 时持续运行，按 Ctrl+C 后命令节点会连续发送零速度
0.5 秒。完整动作切换验收使用：

```bash
ros2 run mydog_policy mydog_hardware_balance_command \
  --profile nominal --action all --segment-sec 5
```

`full` 是仿真验收速度上限，只能在 low 和 nominal 已全部通过、周围留有足够空间
后使用。不要发布超出模型范围的命令：`vx [-0.12,0.45]`、
`vy [-0.08,0.08]`、`yaw [-0.8,0.8]`；节点仍会在入口处强制裁剪。
