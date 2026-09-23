# B23500 / V22 真机部署

## 版本与边界

基于 `fd36394` 的最新真机代码新增独立入口，不替换 B18000。
模型：`rs01_omni_v22_sensor`，GPU1 seed16，`model_23500.pt`。
打包文件：`src/mydog_policy/resource/B23500.onnx`（及同名 JSON）。
SHA256：`4367b59d30a75bd9150866bd90d0fa304b05f7eeab874c477e8dde452362423f`。
本次离线验证没有连接/启用真机，也没有启动训练；代码发布不代表真机验收通过。

真实 RS01 硬件约束保持：61D 输入、12D 输出、50 Hz，Kp=40/Kd=1，
运行峰值保护14 Nm、连续参考6 Nm、热降额 full RMS=8 Nm/tau=2 s。
URDF/质量/杆长没有改动（当前基线11.7317 kg、腿杆180/202.158 mm）。
不向真机观测再次注入仿真噪声、延迟，不软件模拟真实电机的 FOPDT。
保持模型学习的对角步态，不更换奖励或模型权重。

## 已修正的部署差异

| 问题 | 本次处理 | 检验 |
|---|---|---|
| 最新 B18000 入口弱化了直行纠偏，不同于 V22 训练 | B23500 恢复训练的 V13 有效指令公式 | 同传感器回放61D一致性 |
| 旧入口首帧里程计输出零 | B23500 首帧从实际 q/dq/gyro 估计；不提前推进相位 | 首帧测试、全向回放 |
| 浮点到达判断放大微小差异 | B23500 与 V22 MuJoCo 限速器增加1e-12 rad到达容差 | 修正前目标差0.01839 rad，修正后见下表 |
| 新旧模型误接命令/模型文件 | 独立命名空间、固定模型SHA、合同校验 | ROS mock及模型测试 |

浮点容差只消除 float64 数值残差，不改关节速度/加速度/扭矩上限。
旧 B18000/A6850 默认行为不变。训练端与 checkpoint 未改：这不是声称跨
Torch float32/NumPy float64 逐位相同，也不是重新训练后的认证。
部署Actor、里程计和PD保护使用同控制周期传感器快照；估计速度不是外部真值。

## 离线结果

47项模型/ROS mock/校准/频率保护/采集回归测试通过；colcon构建通过。
测试所有设备接口替换为mock，缺失的IMU库用禁止构造的测试替身隔离。

MuJoCo同状态比较：每5秒切换，动作间踏步，共125秒/6250步；
seed=20260925，phase=0，B23500。两端共享相同传感器输入，各自维护历史，
不是每步复制限速器状态来掩盖漂移。

| 最大绝对差 | Normal | Robust |
|---|---:|---:|
| 61D归一化观测 | 2.21e-6 | 2.75e-6 |
| 动作 | 2.21e-6 | 2.75e-6 |
| 限速后目标/rad | 3.41e-7 | 4.59e-7 |
| 保护后目标/rad | 3.77e-7 | 4.64e-7 |
| 动态扭矩上限/Nm | 1.72e-5 | 1.65e-5 |

修正后的Normal独立MuJoCo快档全向125秒：完成，无停止/非有限值，腾空0%；
roll RMS=1.827°，base vz RMS=0.10683 m/s，原始PD峰值16.58 Nm，
关节速度峰值18.02 rad/s。仍存在垂向弹跳，不能宣称真机步态已经修好。
快档：前进0.4、后退-0.3、横移±0.3 m/s，旋转±0.6 rad/s；
对角±0.3/±0.15，组合±(0.3,0.15,0.5)。这些不是初次真机测试速度。
原始PD请求不等于实际输出扭矩。

复现（训练电脑）：

```bash
cd /home/nszb/gym
source unitree-rl/bin/activate
python mujoko/rs01_go2/check_v22_deployment.py --seconds 125 --sensor_profile normal
python mujoko/rs01_go2/check_v22_deployment.py --seconds 125 --sensor_profile robust
python mujoko/rs01_go2/sim2sim_v22.py \
  --policy artifacts/rs01_v22_sim2sim/B23500.onnx \
  --scene artifacts/rs01_v20_sim2sim/scene.xml \
  --sequence --fast --sensor_profile normal --sensor_seed 20260925 \
  --output artifacts/rs01_v22_sim2sim/deployment_numerical_fix
```

## 安装与不发指令检查（真机电脑）

使用真机已有 ROS/ONNXRuntime/YbImuLib 环境，不激活训练电脑的 Isaac Python3.8。
进入本仓库 `zhenji/mydog_ros2_ws` 后：

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select mydog_policy
source install/setup.bash
ros2 run mydog_policy mydog_validate_model23500 \
  "$(ros2 pkg prefix mydog_policy)/share/mydog_policy/models/B23500.onnx"
ros2 launch mydog_policy rs01_model23500.launch.py
```

最后一条默认 `enable_send=false, stand_only=true`，但仍会读取传感器。
validator是纯离线校验，其计算耗时不等于真实发送频率。
先确认安装的是新模型、IMU方向、电机零位/顺序、反馈时间戳以及传输服务正确。
启动前关闭其他策略进程，不绕过硬件独占锁。

## 有支撑的人工测试（以下会发送电机目标）

必须先确认机器人有可靠防倒支撑，关节/线缆运动空间足够，人工急停可用。
停止上面的dry-run节点。以下两种入口二选一，不能同时启动。

普通入口：

```bash
ros2 launch mydog_policy rs01_model23500.launch.py enable_send:=true stand_only:=false
```

推荐首次使用带61维采集的入口，`capture_dir` 必须是新目录：

```bash
ros2 run mydog_policy mydog_capture23500_node --ros-args \
  -p enable_send:=true -p stand_only:=false \
  -p max_motor_age_ms:=80.0 -p max_imu_age_sec:=0.06 \
  -p capture_dir:="/tmp/rs01_B23500_$(date +%Y%m%d_%H%M%S)"
```

节点启动后先保持有支撑静止站立。另开已source环境的终端：

```bash
ros2 service call /mydog/model23500/calibrate_imu std_srvs/srv/SetBool '{data: true}'
ros2 topic echo /mydog/model23500/status --once
```

等待至少5秒稳定采样并检查 `imu_calibrated=true`、`timing_ready=true`、
`mode=ready`、`walk_start_stable=true`。未满足时不要跳过保护。
实际控制和成功发送需有40个合格间隔：中位18–22 ms，P95≤30 ms，最大≤40 ms。
掉频立即撤销运动授权。校准必须静止且有支撑，不能把倾斜姿态标成水平。

先3秒踏步，通过之后才单独3秒低速前进，每条命令结束均撤销授权：

```bash
ros2 run mydog_policy mydog_model23500_command --march --seconds 3
# 检查姿态、抬脚、支撑和记录，再执行下一条
ros2 run mydog_policy mydog_model23500_command --vx 0.10 --seconds 3
```

立即撤销步行授权：

```bash
ros2 service call /mydog/model23500/arm std_srvs/srv/SetBool '{data: false}'
```

**撤销授权、超时和Ctrl+C不是硬件断电急停；继承的停步链会保留使能/支撑目标。**
需要切断驱动时使用已验证的硬件急停流程，不依赖软件退出。
试验指令上限保持 |vx|≤0.30、|vy|≤0.20 m/s、|wz|≤0.30 rad/s，
单次≤6秒，350 ms命令失联保护，无自动arm。

## 风险、下一步与回退

软件接入完成不等于Sim2Real验收通过。仍缺真实发送/反馈日志、设备采样时间戳同步证据、
真机支撑/直行/横移测量。80 ms电机、60 ms IMU是初次受控测试接收新鲜度门槛，
不是设备采集延迟证明，也不代表V22训练覆盖全部真实时延。
采集文件明确保留 `acquisition_sync_verified=false`，不把接收时间冒充采样时间。
先检查实测50Hz、快照年龄/丢帧、61D范围、保护介入率与机身振荡，再逐方向扩大测试。
本轮无需长训练命令；尚未获得新真机证据前不要仅凭仿真推进无支撑行走。

回退：退出B23500节点，使用原 `rs01_model18000.launch.py` 与原模型/命令；
不能仅将旧ONNX放进B23500路径。旧入口、旧SHA与旧控制行为均保留。
