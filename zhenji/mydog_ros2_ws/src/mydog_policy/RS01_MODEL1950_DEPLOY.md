# RS01 新机 estimator-parity model_1950 真机部署

本入口只接受 `rs01_go2_estimator_parity/model_1950`。它保持训练中的
54维观测、12路直接关节输出、50 Hz、RS01电机语义和腿里程计路径观测。
ONNX任务名、SHA-256、观测来源、关节顺序、电机ID、反馈下标和六个反向
电机都会在打开硬件输出前严格校验。

## 当前放行级别

这是系留短程候选，不是无保护自由行走版本。30秒MuJoCo测试无跌倒、无腾空、
无非法接触，但实际速度为 `0.208 m/s`，横向路径RMS为 `0.182 m`，
Raw torque P95为 `17.92 N.m`，17 N.m饱和率为 `6.68%`。真机节点继续
使用14 N.m硬件上限和6 N.m起始热降额，不能用增加限幅的方式追求仿真速度。

## Jetson构建与离线验证

```bash
cd /home/jetson/mydog_ros2_ws
source /opt/ros/humble/setup.bash

colcon build --packages-select mydog_policy --symlink-install
source install/setup.bash

colcon test --packages-select mydog_policy
colcon test-result --verbose
ros2 run mydog_policy mydog_validate_rs01_model1950
```

验证器必须输出 `54 observations -> 12 actions`、两个腿里程计来源和12个
电机到关节的映射。任何SHA、任务、维度、符号或观测来源不一致都会拒绝启动。

## 三阶段系留验收

第一阶段只读反馈，不发送目标。机器人保持静止完成5秒IMU偏置标定：

```bash
ros2 launch mydog_policy rs01_model1950.launch.py \
  enable_send:=false stand_only:=true \
  debug_csv_path:=/tmp/rs01_model1950_dryrun.csv
```

第二阶段在可靠吊架或悬空条件下只执行柔和站立：

```bash
ros2 launch mydog_policy rs01_model1950.launch.py \
  enable_send:=true stand_only:=true \
  debug_csv_path:=/tmp/rs01_model1950_stand.csv
```

只有电机符号、站立姿态、急停、14 N.m硬件限幅回读、温度和腿里程计全部
通过后，才允许在吊架、双侧牵引绳和物理急停条件下短程行走：

```bash
ros2 launch mydog_policy rs01_model1950.launch.py \
  enable_send:=true stand_only:=false \
  debug_csv_path:=/tmp/rs01_model1950_tether.csv
```

另一个终端先发送允许范围下限 `0.21 m/s`：

```bash
ros2 topic pub --rate 20 /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.21, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
```

首次行走限制在3--5秒。立即核对CSV中的姿态、横向路径、逐足里程计、
实测扭矩、PD限幅、2秒热RMS、温度和低置信度事件。出现支撑不足、持续纠偏、
单侧拖脚、温度快速上升或电机反馈陈旧时立即物理急停，不得反复尝试。
