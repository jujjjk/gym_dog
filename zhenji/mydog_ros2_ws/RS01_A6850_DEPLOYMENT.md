# A6850 部署适配：离线内核与 ROS2 影子节点

## 当前状态（2026-09-10）

本轮已验证最佳候选为 A组seed16 A6850：`stand_only_6850.onnx`，不是V15 support短训7000。
ONNX SHA256：`fa5988d2c29bb91bfd5e064532261a04f7ea7eefdffa9aeedb9ab14e197ab556`。
原件在`artifacts/rs01_v15_support/`，部署副本在`src/mydog_policy/resource/stand_only_6850.onnx`，JSON一并复制。

**已完成：**61维推理内核、原腿里程计、相位/方向控制、映射、当前训练目标限速器、模型哈希检查、反馈时间检查、影子ROS节点、launch及安装配置。
**未完成：**真实反馈发布适配、真实采样时钟对齐、带保护的启动/急停/硬件发送联调、ROS实机运行与热安全验证。此版本不是可直接落地行走的完整部署版本。

旧930/1850/1950节点保留不变。不要把A6850文件直接塞进旧54维节点。

## 已验证一致性

固定A6850与修正传感器时序的MuJoCo，550帧/11段状态回放，包含站立、踏步、前后左右、双向旋转、组合与反向组合：

|对象|最大绝对差|
|---|---:|
|61维观测|0|
|12维执行动作|0|
|12维限速后目标|0|

这是相同传感器状态输入的离线一致性，不是本部署节点闭环真机验收。新增测试还覆盖过期/未来/不同步/重复反馈与错误模型哈希拒绝。

修复了继承旧部署代码会产生的两项不一致：旧节点使用54维/仅vx；旧目标限速器是制动距离算法，与当前训练的目标跨越处理不同。新内核保留JSON浮点精度，避免饱和目标附近的浮点舍入改变限速器分支。

不在真实部署内核中再模拟电机FOPDT、摩擦、增益或通信延迟，那些由真实硬件产生，重复添加会改变闭环。候选目标不是经过硬件保护验收的可发送指令。训练侧目标跨越时归零速度的语义被原样保留，不声称它是无例外的严格加速度约束。

## 离线检查（当前开发机可运行）

```bash
cd /home/nszb/gym
source unitree-rl/bin/activate
PYTHONPATH=zhenji/mydog_ros2_ws/src/mydog_policy python -m mydog_policy.validate_rs01_model6850 \
  zhenji/mydog_ros2_ws/src/mydog_policy/resource/stand_only_6850.onnx
```

## ROS2 影子验证（在已有ROS2的Jetson上）

```bash
cd /path/to/gym/zhenji/mydog_ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select mydog_policy --symlink-install
source install/setup.bash
ros2 launch mydog_policy rs01_model6850_shadow.launch.py
```

本机没有rclpy，未运行上述ROS命令。需要与设备Python版本匹配的onnxruntime。节点不打开电机HTTP控制接口或串口，只订阅已发布的反馈；`enable_send=true`直接拒绝启动。

### 输入契约：必须由真实反馈适配器实现

|话题|类型|要求|
|---|---|---|
|`/mydog/model6850/joint_feedback`|sensor_msgs/JointState|12电机实际角度/速度，name为FR/FL/RL/RR的hip/thigh/calf_joint完整名字；不是已经转换符号的策略坐标|
|`/mydog/model6850/imu_body`|sensor_msgs/Imu|frame_id=base_link；机身姿态xyzw和机身角速度rad/s；发布者先完成安装变换及静止陀螺偏置标定|
|`/mydog/model6850/cmd_vel`|geometry_msgs/Twist|x/y/z偏航速度；需持续刷新，零速度表示踏步而不是站立|

JointState的header必须表示12关节已同步重采样到的共同采样时刻，不能把12个不同时间的电机值打包后用接收时间冒充。IMU header亦必须是同一时钟域的采样时刻。节点只检查声明的时间，无法独立证明发布者没有伪造同步。

初始守护阈值：样本年龄≤40ms，声明的关节/IMU时差≤10ms，样本严格递增；策略调度间隔10–40ms、command超时500ms。这是影子诊断阈值，不是已验证的真实硬件容差。发现异常停止发布并锁存，调用`/mydog/model6850/reset_shadow`（std_srvs/Trigger）后需重新提供反馈与命令。

输出：`/mydog/model6850/shadow_target`仅为JointState候选目标，**不得直接接至电机发送器**；`observation`为61维观测，`status`为锁存故障信息。影子节点始终gait_enable=1，未实现带电站立/起身流程；命令超时停止影子输出，不伪装为安全停车。

## 为什么暂不接真实发送

现有`motor_state_interface.py`的snapshot.stamp在HTTP响应之后调用time.time()；IMU接口也在接收更新时调用time.time()。这些是接收时间，不是已同步的采样时间。电机还提供board_tick_ms/last_update_ts等字段，但两块板与IMU之间的时钟关系尚未验证。

现有1950链路还有14Nm硬件限制、热降额和PD等效目标修正，与当前17Nm仿真候选不等价。不能为了“跑起来”绕过这些保护，也不能不说明差异就继承使用。

下一步需要设备侧提供：一小段12电机q/dq、每电机采样tick/时间/seq，以及IMU原始时间/角速度/姿态的只读日志；确认实际使用的HTTP/固件接口及电流、扭矩安全设置。随后才能实现时间对齐适配器和硬件安全发送状态机，并进行悬挂/系留短测。

不需要因本次内核适配重训；高速.6m/s尚不合格，原地漂移/收腿/起伏仍存在。不得直接脱离保护上真机。

## 回退

新模块和launch均使用6850独立名称。继续使用原launch即可回到旧版本；无需覆盖原模型、修改电机参数或回滚固件。本次没有执行硬件操作、长训练或仓库推送。
