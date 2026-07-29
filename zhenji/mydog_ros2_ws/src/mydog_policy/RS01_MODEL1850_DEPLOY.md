# RS01 新机 model_1850 真机部署

本部署只用于
`rs01_go2_path54_sim2sim_transfer/model_1850`。策略输入54维、输出12维、
控制频率50 Hz，使用RS01实测执行器参数对应的控制契约：

- Kp：全部关节 `40 N·m/rad`；
- Kd：hip/thigh `1.0 N·m/(rad/s)`，calf `0.5 N·m/(rad/s)`；
- 策略输出直接生成12个关节位置目标；
- 没有额外CPG、步态补偿或关节偏置；
- 真机峰值命令上限为14 N·m，热RMS从6 N·m开始降额。

## 54维观测

前52维与model_930一致。末尾两维为：

1. 相对本次直行启动方向的横向位移，单位米、乘以2后裁剪到 `[-1, 1]`；
2. 初始航向坐标系内的横向速度，单位米每秒、乘以2后裁剪到 `[-1, 1]`。

真机没有全局定位时，节点使用新机URDF腿里程计估计机身速度，将体坐标速度
按IMU航向旋转到直线路径坐标系，并从零积分横向位移。节点不会在柔和站立
或人工扶正期间反复锁定路径原点。只有完成站立并同时满足 roll/pitch、
三轴角速度、腿里程计速度和置信度门限，且连续稳定1秒后，才会清空腿里程计
滤波状态并一次性锁定当前航向、路径原点和步态相位。命令停止后回到 ready；
再次行走也必须重新通过同一稳定门控。控制循环或里程计更新间隔超过100 ms
会触发停止，避免陈旧路径状态进入策略。

CSV和状态话题会额外记录 `walk_start_stable`、稳定持续时间、路径重置次数、
锁定航向、航向误差、四足支撑选择以及每只脚独立估计出的机身速度。这些字段
用于区分“策略纠偏”与“腿里程计漂移/单脚打滑导致的假纠偏”。

节点启动后会先采集5秒静止IMU数据。只有三轴角速度标准差、姿态变化范围和
估计偏置都在配置边界内才继续；估计偏置会同时从策略角速度观测和腿里程计
角速度中扣除。标定期间不会发送电机命令，机器人必须保持完全静止。

## 电机语义

策略顺序为FL、FR、RL、RR，每条腿hip、thigh、calf；真机反馈顺序为
FR、FL、RL、RR。用户确认的反向电机保持不变：

```text
0x11 FR_hip    0x13 FR_calf
0x21 FL_hip    0x22 FL_thigh
0x32 RL_thigh  0x43 RR_calf
```

节点在任何硬件输出前校验ONNX SHA-256、54→12图维度、任务名、动作顺序、
电机ID、反馈下标和六个符号。

## 构建和离线验证

```bash
cd /home/jetson/mydog_ros2_ws
source /opt/ros/humble/setup.bash

colcon build --packages-select mydog_policy --symlink-install
source install/setup.bash

colcon test --packages-select mydog_policy
colcon test-result --verbose

ros2 run mydog_policy mydog_validate_rs01_model1850
```

## 分阶段启动

第一步只读反馈，不发送任何电机命令：

```bash
ros2 launch mydog_policy rs01_model1850.launch.py \
  enable_send:=false stand_only:=true \
  debug_csv_path:=/tmp/rs01_model1850_dryrun.csv
```

确认54维观测持续有限、12台电机在线、IMU与腿里程计有效后，在可靠支撑或
悬空条件下仅执行柔和站立：

```bash
ros2 launch mydog_policy rs01_model1850.launch.py \
  enable_send:=true stand_only:=true \
  debug_csv_path:=/tmp/rs01_model1850_stand.csv
```

只有站立、急停、硬件限幅回读及符号验收全部通过后，才允许在吊架、牵引绳
和物理急停条件下进行短程行走：

```bash
ros2 launch mydog_policy rs01_model1850.launch.py \
  enable_send:=true stand_only:=false \
  debug_csv_path:=/tmp/rs01_model1850_tether.csv
```

另一个终端持续发送0.23 m/s直行命令：

```bash
ros2 topic pub --rate 20 /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.23, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
```

## 放行边界

model_1850已通过同一新机URDF下的30/60秒MuJoCo无跌倒、无腾空和无非法
接触测试，但60秒横向路径RMS约0.298 m，电机超过6 N·m的比例约30.5%。
因此这是吊架/系绳短程验证版本，不是无保护自由行走放行版本。必须根据新采集
CSV继续核对横移、实际反馈扭矩、2秒热RMS、温度和低里程计置信度事件。
