# symmetric-transition 5530 最终真机部署

当前生产入口已固定到 `fanfan_symmetric_transition_5530.onnx`：

- SHA256：`45af53978ba7c83c5b3419fe172ceb1a91b7f647861d77e9fc5b460e431a0343`
- task：`FanfanOmniSymmetricTransitionCfg`
- 观测/动作：`52 -> 12`
- 控制频率：50 Hz
- 步态周期：0.45 s
- PD：hip `60/1.2`，thigh `70/1.6`，calf `70/1.6`
- 电机限制：每条腿 `10/10/13 Nm`，`12/12/16 A`

ONNX 内部保留速度反馈、航向纠偏、动作滤波和左右严格对称。ROS2
只负责观测、连续命令、关节映射和真机安全握手，不重复叠加反馈或镜像。

## 构建和模型校验

```bash
cd /home/jetson/mydog_ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select mydog_policy --symlink-install
source install/setup.bash

ros2 run mydog_policy mydog_validate_symmetric_transition_model \
  install/mydog_policy/share/mydog_policy/models/fanfan_symmetric_transition_5530.onnx
```

必须看到 `symmetric-transition 5530 ONNX validation PASSED` 和上面的 SHA256。

## 不发送电机命令的预演

```bash
ros2 launch mydog_policy sim2real_realdata.launch.py \
  enable_send:=false \
  print_only:=true \
  debug_print_arrays:=true
```

`sim2real_realdata.launch.py` 是保留的生产别名，现在固定包含 5530 启动文件。

## 悬空启动

机器人可靠悬空并准备物理急停后：

```bash
mkdir -p /home/jetson/mydog_ros2_ws/log

ros2 launch mydog_policy sim2real_realdata.launch.py \
  enable_send:=true \
  print_only:=false \
  debug_csv_path:=/home/jetson/mydog_ros2_ws/log/symmetric_transition_5530.csv
```

必须依次看到模型 SHA/task 校验通过、12 电机限制读回成功和
`[STARTUP_STAND][READY]`。默认启用 0.45 rad 倾角保护、0.5 秒命令超时
站立保持、100 ms 电机状态时效检查以及连续状态丢失后的全电机 STOP。
后腿 17 Nm 临时增力已关闭。

## 动作命令

首次只使用低架速度：

```bash
ros2 run mydog_policy mydog_symmetric_transition_command \
  --profile low --action forward --segment-sec 3
```

可选动作：`forward`、`backward`、`left_lateral`、`right_lateral`、
`left_yaw`、`right_yaw`、`left_diagonal`、`right_diagonal`、`left_arc`、
`right_arc`。

低速全部通过后，执行模型验收速度的连续转换：

```bash
ros2 run mydog_policy mydog_symmetric_transition_command \
  --profile fast --action all --segment-sec 3 --repeat 1 --rate 20
```

模型完整范围为 `vx [-0.25,0.60]`、`vy [-0.26,0.26]`、
`yaw [-1.30,1.30]`。真机首次落地不要直接使用完整范围。

## 停止

先停止动作发布器，使其发送零命令，再执行真正的电机 STOP：

```bash
curl -sS -X POST http://127.0.0.1:8000/api/stop
```

然后停止 ROS2 launch 和电机服务。零速度命令是带电站立，不是急停。
