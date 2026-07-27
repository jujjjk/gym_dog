# RS01 新机 model_930 真机部署

本部署只用于 `rs01_go2_sim2sim_heading52/model_930`，策略输入 52 维、输出
12 维、控制频率 50 Hz，`Kp=40`，hip/thigh `Kd=1.0`，calf
`Kd=0.5`。策略输出直接生成 12 个关节位置目标，没有额外 CPG、步态补偿或
关节偏置。

## 电机和符号契约

策略顺序为 FL、FR、RL、RR，每条腿 hip、thigh、calf；真机反馈顺序为
FR、FL、RL、RR。用户确认的反向电机为：

```text
0x11 FR_hip    0x13 FR_calf
0x21 FL_hip    0x22 FL_thigh
0x32 RL_thigh  0x43 RR_calf
```

转换规则为 `q_policy = sign * q_real`，以上六个 sign=-1，其余 sign=+1。
映射和符号同时写入 ONNX 元数据；节点在任何硬件输出前校验文件 SHA、模型
维度、动作顺序、电机 ID、反馈下标和全部符号。

模型站立姿态转换到真机反馈顺序后为：

```text
FR: [ 0.0, -0.32987297, -1.31853104]
FL: [ 0.0,  0.32987297,  1.31853104]
RL: [ 0.0,  0.32987297,  1.31853104]
RR: [ 0.0, -0.32987297, -1.31853104]
```

## 构建和离线测试

```bash
cd /home/nszb/gym/zhenji/zhenji_dog/mydog_ros2_ws
source /opt/ros/humble/setup.bash
python3 -c "import numpy, requests, onnxruntime, YbImuLib"
colcon build --packages-select mydog_policy --symlink-install
source install/setup.bash

colcon test --packages-select mydog_policy
colcon test-result --verbose

ros2 run mydog_policy mydog_validate_rs01_model930
```

## 分阶段启动

先启动现有 RS01 FastAPI 驱动，然后只做不发电机命令的观测检查：

```bash
ros2 launch mydog_policy rs01_model930.launch.py \
  enable_send:=false stand_only:=true \
  debug_csv_path:=/tmp/rs01_model930_dryrun.csv
```

确认 12 个电机在线、六个符号与抬腿方向正确、IMU 方向正确后，机器人悬空或
可靠支撑，仅允许柔和进入站立：

```bash
ros2 launch mydog_policy rs01_model930.launch.py \
  enable_send:=true stand_only:=true \
  debug_csv_path:=/tmp/rs01_model930_stand.csv
```

只有站立和急停验收通过后，才可在吊架、牵引绳和物理急停条件下允许行走：

```bash
ros2 launch mydog_policy rs01_model930.launch.py \
  enable_send:=true stand_only:=false \
  debug_csv_path:=/tmp/rs01_model930_tether.csv
```

另一个终端以至少 10 Hz 持续发送直行命令；节点仅接受 0.21–0.25 m/s，
横移和转向命令会被拒绝，0.5 秒未收到命令会回到站立：

```bash
ros2 topic pub --rate 20 /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.23, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
```

节点启动时对全部电机写入并回读验证 14 N·m 峰值限制和
12/12/16 A 的 hip/thigh/calf 电流限制。14 N·m 只用于短时峰值：节点用
实际反馈扭矩和 PD 请求中较大的一个计算 2 秒热 RMS；6–8 N·m 区间会把
活动上限连续降到 6 N·m。反馈超时、掉线、电机错误、温度超过 70 °C、
IMU 超时或姿态超限都会调用停止接口。

## 放行边界

model_930 已通过 PhysX 和匹配 MuJoCo 的 30 秒不跌倒测试，但训练时后腿
电机辨识参数的左右分配曾有错误；修正映射后的 MuJoCo 仍出现约 1.287 m
横移和 0.697 rad 航向漂移。因此当前代码是悬空、站立和有保护短程联调版本，
不是无需保护的真机自由行走放行版本。
