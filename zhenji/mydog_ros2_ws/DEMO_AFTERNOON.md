# 下午演示：回到 5530（symmetric-transition）

生产入口已切回：

```text
sim2real_realdata.launch.py
  -> sim2real_symmetric_transition_5530.launch.py
  -> fanfan_symmetric_transition_5530.onnx
SHA256 45af53978ba7c83c5b3419fe172ceb1a91b7f647861d77e9fc5b460e431a0343
```

不要用 5650 / saturation_recovery 做今天下午演示。

---

## A. 环境

```bash
cd /home/jetson/mydog_ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
mkdir -p /home/jetson/mydog_ros2_ws/log
```

校验模型：

```bash
ros2 run mydog_policy mydog_validate_symmetric_transition_model \
  "$(ros2 pkg prefix mydog_policy)/share/mydog_policy/models/fanfan_symmetric_transition_5530.onnx"
```

## B. 终端 1：启动 5530

先预演：

```bash
ros2 launch mydog_policy sim2real_realdata.launch.py \
  enable_send:=false \
  print_only:=true \
  startup_stand_first:=false
```

架空下发：

```bash
ros2 launch mydog_policy sim2real_realdata.launch.py \
  enable_send:=true \
  print_only:=false \
  startup_stand_first:=true \
  debug_csv_path:=/home/jetson/mydog_ros2_ws/log/demo_5530_$(date +%Y%m%d_%H%M%S).csv
```

看到 `[STARTUP_STAND][READY]` 再发速度。

## C. 终端 2：低速命令（推荐）

优先用官方低速 profile：

```bash
source /opt/ros/humble/setup.bash
source /home/jetson/mydog_ros2_ws/install/setup.bash

ros2 run mydog_policy mydog_symmetric_transition_command \
  --profile low --action forward --segment-sec 5
```

或阶梯脚本（默认 `0.00 → 0.08 → 0.12`，勿跳 0.32）：

```bash
bash /home/jetson/mydog_ros2_ws/demo_ladder_cmd.sh
```

## D. 急停

硬件急停优先；软件发零速或停掉 `/cmd_vel`。
