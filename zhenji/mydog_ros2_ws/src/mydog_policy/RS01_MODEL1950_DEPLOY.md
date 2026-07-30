# RS01 新机 estimator-parity model_1950 真机部署

本入口只接受 `rs01_go2_estimator_parity/model_1950`：

- ONNX：`model_1950_rs01_estimator_parity.onnx`
- SHA-256：`f78242f6ac60354421d7354b8a5f4b61284864c18be409aa0f854520f8d1202c`
- 输入/输出：54维观测、12路直接关节位置目标
- 控制频率：50 Hz
- 速度观测：`rs01_leg_odometry`
- 横向路径观测：`rs01_leg_odometry_integral`

ONNX任务名、SHA-256、观测来源、关节顺序、电机ID、反馈下标和六个反向
电机都会在打开硬件输出前严格校验。`model_1950` 与旧的 `model_1850`
不是同一个模型；验证和启动时必须使用带 `1950` 的入口。

## 当前放行级别

这是系留短程候选，不是无保护自由行走版本。30秒MuJoCo测试无跌倒、无腾空、
无非法接触，但实际速度为 `0.208 m/s`，横向路径RMS为 `0.182 m`，
Raw torque P95为 `17.92 N·m`，17 N·m饱和率为 `6.68%`。真机节点继续
使用14 N·m硬件上限和6 N·m起始热降额，不能用增加限幅的方式追求仿真速度。

## Jetson构建与离线验证

以下操作只构建、运行pytest和执行ONNX CPU推理，不启动电机或IMU设备：

```bash
cd /home/jetson/mydog_ros2_ws
source /opt/ros/humble/setup.bash

colcon build --packages-select mydog_policy --symlink-install
source install/setup.bash

colcon test --packages-select mydog_policy
colcon test-result --verbose

ros2 run mydog_policy mydog_validate_rs01_model1950
```

当前Jetson验收结果应为：

```text
Summary: 30 tests, 0 errors, 0 failures, 1 skipped
PASS sha256=f78242f6ac60354421d7354b8a5f4b61284864c18be409aa0f854520f8d1202c
graph=54 observations -> 12 actions, policy_hz=50
velocity_source=rs01_leg_odometry
path_source=rs01_leg_odometry_integral
```

若再次出现 `collection failure`，先查看完整异常：

```bash
cat /home/jetson/mydog_ros2_ws/log/latest_test/mydog_policy/stdout_stderr.log
```

不要用下面这个错误命令：

```bash
ros2 run mydog_policy model_1950_rs01_estimator_parity.onnx
```

ONNX是数据文件，不是ROS可执行程序。

## IMU标定、柔和站立和行走时序

一次启动内的真实顺序是：

```text
启动IMU
→ 5秒静止陀螺仪零偏标定（不发送电机命令）
→ 开启电机输出
→ startup_hold
→ stand_ramp柔和站立
→ ready
→ 行走稳定门控
→ 清零行走会话状态并锁定航向
→ walk
```

柔和站立发生在5秒IMU标定完成之后，因此不会写入标定样本。标定期间必须由
吊架或刚性支撑保持机身完全静止，不能扶正、抬升、卸载吊架或旋转机身。
看到 `Gyro bias calibrated` 后才允许机器人执行柔和站立。

不要先站立再重启节点。重启后的5秒标定期没有电机目标输出，机器人可能下沉，
反而无法保持静止。应在同一次启动中完成标定、站立和行走。

## 行走前稳定门控

进入 `ready` 前，全部关节必须在默认站立目标的 `0.12 rad` 内连续保持2秒。
这个门限来自model_1950真机CSV：负载下后腿calf最大稳态误差约
`0.108 rad`。共享默认值 `0.08 rad` 无法连续满足1秒，会导致速度命令已经
收到但节点仍永久停在 `stand_ramp`。

节点到达 `ready` 后，即使已经收到速度命令，也必须连续1秒满足：

- `|roll| <= 0.10 rad`
- `|pitch| <= 0.10 rad`
- 校正后三轴角速度范数不超过 `0.08 rad/s`
- 腿里程计平面速度不超过 `0.05 m/s`
- 合法对角支撑（FL+RR或FR+RL）里程计置信度不低于 `0.5`
- vendor yaw角增量与校正后z轴陀螺仪的低频差不超过 `0.06 rad/s`

真正进入 `walk` 时，节点会同时清零腿里程计滤波、支撑腿历史、策略相位、
动作历史、目标限速器和横向路径积分，并将此刻IMU yaw锁定为本次直行方向。
这可避免人工扶机留下的假速度或随机航向进入第一帧策略观测。

## 三阶段系留验收

### 第一阶段：只读干跑

不发送任何电机目标：

```bash
ros2 launch mydog_policy rs01_model1950.launch.py \
  enable_send:=false stand_only:=true \
  debug_csv_path:=/tmp/rs01_model1950_dryrun.csv
```

确认ONNX校验通过、12台电机在线、IMU有效、反馈持续新鲜且观测无NaN/Inf。

### 第二阶段：只允许站立

在可靠吊架或刚性支撑条件下执行：

```bash
ros2 launch mydog_policy rs01_model1950.launch.py \
  enable_send:=true stand_only:=true \
  debug_csv_path:=/tmp/rs01_model1950_stand.csv
```

前5秒完全静止。标定完成、柔和站立开始后，再缓慢卸载吊架重量，让四脚逐渐
承重。不要施加侧向力或旋转机身。只有电机符号、默认站姿、物理急停、
14 N·m硬件限幅回读、温度和腿里程计全部通过，才进入第三阶段。

### 第三阶段：短程系留行走

机器人必须位于吊架、双侧牵引绳和物理急停保护范围内：

```bash
ros2 launch mydog_policy rs01_model1950.launch.py \
  enable_send:=true stand_only:=false \
  debug_csv_path:=/tmp/rs01_model1950_tether_01.csv
```

另一个终端观察状态：

```bash
source /opt/ros/humble/setup.bash
source /home/jetson/mydog_ros2_ws/install/setup.bash

ros2 topic echo /mydog/model1950/status \
  std_msgs/msg/String --field data
```

发送速度命令前，等待以下条件成立：

```text
mode=ready
walk_start_stable=true
```

`walk_ready_duration_s`不需要在发命令前达到`1.0`。它只会在节点已经处于
`ready`且收到非零速度命令后开始累计；稳定累计满1秒后，节点才会从
`ready`切换到`walk`。因此发出命令后的最初约1秒仍处于安全门控阶段，
这是正常现象。

若合法对角里程计低置信度或yaw/gyro不一致持续超过`0.60 s`，节点会锁存
`mode=soft_hold`和`walk_inhibit_latched=true`。这条路径不会调用
`/api/stop`，而是从当前实测关节位置按原站立速率和14 N·m保护柔和回到默认
站姿；非零速度命令持续存在时也不会重新进入`walk`。必须先停止发布速度命令，
等待站姿、航向一致性连续恢复2秒，锁存才会解除。硬件掉线、过温、反馈过期、
姿态越界和节点退出仍保留紧急停机，这是独立的硬安全链。

确认四脚稳定承重、人已完全松手且吊绳不产生侧向拉力后，再发送第一次短命令。
总发布4秒，其中最初约1秒用于稳定门控，实际进入`walk`约3秒。不要在
`stand_ramp`阶段提前持续发送：

```bash
timeout 4s ros2 topic pub --rate 20 \
  /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.23, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
```

命令结束约0.5秒后节点应退出 `walk`。首次测试不要直接使用长时间持续发布。

## CSV验收重点

每次尝试使用新的CSV文件名，重点检查：

- `walk_inhibit_latched`、`walk_inhibit_reason`、`walk_inhibit_count`
- `heading_consistency_healthy`、`yaw_gyro_mean_error_rad_s`
- `odom_guard_confidence`、`odom_selected_pair_index`
- `odom_pair_residual_m_s`、`odom_legal_diagonal_support`

- `mode`是否按 `startup_hold → stand_ramp → ready → walk`变化；
- `walk_start_stable`和 `walk_ready_duration_s`是否满足门控；
- `walk_session_reset_count`是否在每次进入行走时增加；
- `heading_error_rad`是否持续扩大；
- `path_lateral_displacement_m`和 `path_lateral_velocity_m_s`是否突跳；
- `odom_stance_*`是否出现长期单侧或错误支撑；
- `odom_velocity_by_foot_*`是否有单足明显离群；
- `pd_limited_count`、`max_thermal_rms_nm`和最小有效扭矩限值；
- 电机温度、反馈延迟、实测扭矩及姿态。

出现以下任一情况应立即物理急停，不得连续反复尝试：

- 无法稳定进入 `ready`；
- 起步瞬间路径速度或航向误差突跳；
- 持续单向纠偏或横向偏移扩大；
- 单侧拖脚、支撑不足或吊绳持续受力；
- 温度快速上升、热降额持续触发；
- 电机反馈陈旧、离线、错误码或异常响声。

每次失败后先保存CSV和ROS日志，再重新固定机器人并从5秒静止标定开始。
