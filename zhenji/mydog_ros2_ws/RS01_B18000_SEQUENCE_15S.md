# B18000：15 秒动作 + 5 秒踏步过渡（2026-09-20）

Jetson：172.19.18.165。工作区：/home/jetson/deploy_staging/b18000-sequence-20260920。
本次仅部署、构建、离线测试、--print-plan，无电机指令。

序列：踏步5秒 → 前进15秒 → 踏步5秒 → 后退15秒 → 踏步5秒 → 左移15秒 → 踏步5秒 → 右移15秒 → 踏步5秒 → 左转15秒 → 踏步5秒 → 右转15秒 → 踏步5秒 → 组合15秒 → 踏步5秒 → 反向组合15秒 → 踏步5秒 → 解除行走授权。

速度(vx,vy,wz)：前进(.20,0,0)，后退(-.20,0,0)，左移(0,.12,0)，右移(0,-.12,0)，左转(0,0,.25)，右转(0,0,-.25)，组合(.20,.10,.25)，反向组合(-.20,-.10,-.25)。单位m/s、rad/s。这是指令速度，不是已验证实际速度。

## 两个终端均执行
```bash
ssh jetson@172.19.18.165
source /opt/ros/humble/setup.bash
source /home/jetson/deploy_staging/b18000-sequence-20260920/install/setup.bash
export ROS_DOMAIN_ID=99
ros2 pkg prefix mydog_policy
```
预期prefix：/home/jetson/deploy_staging/b18000-sequence-20260920/install/mydog_policy。

## 终端A：开启站立控制（会使能电机）
先确认只有一个控制节点，机身有支撑、测试区足够、物理急停可用。
```bash
mkdir -p /home/jetson/mydog_ros2_ws/log/manual_trials
ros2 launch mydog_policy rs01_model18000.launch.py \
  enable_send:=true stand_only:=false \
  debug_csv_path:="/home/jetson/mydog_ros2_ws/log/manual_trials/b18000_sequence_$(date +%Y%m%d_%H%M%S).csv"
```
保持终端A运行。

## 终端B：观察站立，手动标定
```bash
ros2 topic echo /mydog/model18000/status std_msgs/msg/String --field data
```
等待mode=ready。让四脚落地承重、机身稳定；Ctrl+C仅结束终端B的状态显示，然后：
```bash
ros2 service call /mydog/model18000/calibrate_imu std_srvs/srv/SetBool "{data: true}"
ros2 topic echo /mydog/model18000/status std_msgs/msg/String --field data
```
保持静止至少5秒；等待calibration_state=complete、imu_calibrated=true、mode=ready、walk_start_stable=true、timing_ready=true，同时trial_lease_sec=180。未完成时查看calibration_reason。Ctrl+C结束终端B显示。

## 终端B：预览与执行
预览不会发送指令：
```bash
ros2 run mydog_policy mydog_model18000_sequence --print-plan
```
下面开始实际自动运动序列，不要同时运行其他cmd_vel发布程序：
```bash
ros2 run mydog_policy mydog_model18000_sequence
```
动作从确认walk后计时，总165秒，初始等待最多5秒。状态超过250ms、失去时序、进入保护或失去授权时中止并请求disarm，不自动重试或自动重新授权。

## 取消行走
序列终端Ctrl+C，或另一个同环境终端：
```bash
ros2 service call /mydog/model18000/arm std_srvs/srv/SetBool "{data: false}"
```
结束/取消均不等于电机断电，可能继续保持站姿。失稳使用物理急停；先支撑机身再按现场流程结束控制并停止电机输出。

旧mydog_model18000_command仍限制单次6秒，本序列必须使用新的mydog_model18000_sequence入口。

## 单独执行动作，每次15秒后柔和回站

在已完成标定、控制器ready且时序通过时，选择一条执行，不要同时执行多条：

```bash
ros2 run mydog_policy mydog_model18000_sequence --action march
ros2 run mydog_policy mydog_model18000_sequence --action forward
ros2 run mydog_policy mydog_model18000_sequence --action backward
ros2 run mydog_policy mydog_model18000_sequence --action left
ros2 run mydog_policy mydog_model18000_sequence --action right
ros2 run mydog_policy mydog_model18000_sequence --action turn-left
ros2 run mydog_policy mydog_model18000_sequence --action turn-right
ros2 run mydog_policy mydog_model18000_sequence --action combo
ros2 run mydog_policy mydog_model18000_sequence --action reverse-combo
```

每条结束后请求解除行走授权，等待ready且机身稳定持续2秒，最多等待20秒。
这只是控制器状态确认，不是外部测量的关节到位证明。等待失败或触发保护时
先检查机器人，不继续下一个动作。回站后电机仍保持使能。

部署依赖：Jetson现有电机服务须提供兼容的/tmp/lingzu_motor_rt.sock接口，
并保留已验证的扭矩/电流配置接口。本提交包含客户端及协议定义，
不替换Jetson的电机服务、驱动或厂家IMU库。