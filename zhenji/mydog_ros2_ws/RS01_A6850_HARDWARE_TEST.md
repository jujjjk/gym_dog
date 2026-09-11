# A6850 全向真机短测（2026-09-11）

本入口为 `rs01_model6850.launch.py`，使用 A6850 的 61 维观测和原全向推理内核。
旧 `rs01_model6850_shadow.launch.py` 保持只输出候选目标、不控制电机。
模型 SHA256：`fa5988d2c29bb91bfd5e064532261a04f7ea7eefdffa9aeedb9ab14e197ab556`。

2026-09-11 已在吊架上带电短测。策略网络没有改；步态形态问题要重训才能从根上消掉。
下面的带电命令只能由操作者在吊架/系留和物理急停保护下执行。
测试通过不等于已验收自由行走。

## 真机问题（数据，不改策略）

日志 `a6850_omni_seq_20260911_160141.csv`：356 s，10889 行。

| 现象 | 数据 | 含义 |
|---|---|---|
| 步态反复被掐 | `walk` 仅 17%；8 次 `walk→soft_hold` | 不是失能。全部是 `yaw/gyro signed-rate mismatch`（0.067–0.295 rad/s）超过约 0.6 s 后航向保护回站 |
| 内八 | 走路时髋目标约 ±12.6°，髋动作均值约 −0.95/+0.93/−0.90/+0.87 | `action_scale=0.22 rad` 打满 ±1；左右髋反向。一进 gait（约 t=19 s）髋就从 0° 顶到 ±13° |
| 不稳、跟不住 | 走路 `cmd_vx` 均值 0.03、里程计 −0.004；横滚峰值 −25°；`max\|action\|=1` 占 100% | 髋打满产生偏航力矩，喂给上面的航向保护 |
| 训练/硬件不等价 | 训练 17 Nm，真机 14 Nm + 6 Nm 热降额 | 保护改变实际输出，不能当成仿真闭环 |

部署侧已做、**不替代重训**的改动：Isaac 同序连续命令；零速踏步；故障不再 `/api/stop`，继续发上一拍目标以保持使能。航向 `soft_inhibit` 仍会停步态、保持使能。物理急停仍是唯一断电手段。

## 接口和保护

- 节点自己读取 HTTP 电机反馈和 `/dev/myimu`，运行模型内部的原腿里程计。
  不需要单独启动 `mydog_state_estimator_node`，不可让它与本节点争用 IMU。
- IMU 在解析校验和正确的 RAW/QUAT/EULER 数据帧后才更新时间戳；读取旧缓存不会刷新时间戳。
- 电机有效反馈年龄计入电机报告 age、HTTP 服务缓存和本地缓存。当前接收门槛约 250 ms；
  超过后记录并保持使能、重发上一拍目标，不再 `/api/stop`。
  12 路反馈的板序号/tick 回退/重启仍视为异常，同样保持使能。
- 时间基准明确为 **主机接收时间与板报告年龄**。固件没有提供可验证的统一采样时钟，
  因此本入口不声称完成硬件采样同步或具备 MuJoCo 的传感器时序一致性。
  状态 `acquisition_sync_verified=false` 是如实记录这一限制，不是可关闭的保护选项。
- 开启发送时必须完成原有 12 电机限值设置与回读验证：14 Nm，电流上限 12/12/16 A；
  保留 6 Nm 连续扭矩热降额和 PD 目标保护。A6850 训练使用 17 Nm，因此硬件保护会改变实际输出。
- 首先静止标定 5 秒，然后 0.5 秒保持当前角度，再按髋 0.12、大腿/小腿 0.15 rad/s 到站姿。
  关节误差不超过 0.12 rad 持续 2 秒后 `ready`。
- 三轴命令 `/mydog/model6850/cmd_vel`：x 前后、y 左右、z 偏航。
  硬上限 `|vx|<=0.30 m/s`、`|vy|<=0.20 m/s`、`|wz|<=0.30 rad/s`，与 Isaac 演示矩阵一致。
- 每次运动先调用 `/mydog/model6850/arm`，必须 ready 且满足站稳门控。
  解锁后零速表示原地踏步，不是回站。停发约 350 ms 或 `/arm false` 才柔和回站。
  单次解锁覆盖 Isaac 17×5 s 序列（约 120 s）。
- Isaac 同序入口：`ros2 run mydog_policy mydog_model6850_omni_sequence`。
- 软件异常保持使能并重发上一拍目标。航向/里程计 `soft_inhibit` 仍会停步态并回站，但不切使能。
  物理急停始终必要。Ctrl+C 后电机会停在最后一拍，先吊架承重再结束进程。

## 1. 终端 A：构建与离线验证（不启动设备）

```bash
ssh jetson@172.19.61.166
cd ~/mydog_ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select mydog_policy --symlink-install
source install/setup.bash
python3 -m mydog_policy.validate_rs01_model6850 \
  src/mydog_policy/resource/stand_only_6850.onnx

python3 -m pytest \
  src/mydog_policy/test/test_rs01_model6850_guard.py \
  src/mydog_policy/test/test_rs01_model6850_node_offline.py \
  src/mydog_policy/test/test_rs01_timestamped_imu.py -q
```

以上测试把 ROS、HTTP 和串口替换成模拟对象，不实例化真实 ROS 控制节点或操作硬件。
离线验证器输出的 `motor_send_supported=false` 表示该验证器只做推理；
真实发送由新的 `rs01_model6850.launch.py` 入口负责，并非由验证器负责。
原 MuJoCo 550 帧对比测试需要训练仓库场景资产，不能当成 Jetson 部署包独立测试运行。

## 2. 只读干跑（此步由操作者执行，打开 IMU，不使能电机）

先退出占用 `/dev/myimu` 的旧估计器或策略节点，确保只有一个控制程序。
已有 `http://127.0.0.1:8000` 电机服务应保持运行，不重复启动服务。
机器人用吊架静止支撑，不要让原带电控制退出后无人支撑。

```bash
mkdir -p ~/mydog_ros2_ws/log
ros2 launch mydog_policy rs01_model6850.launch.py \
  enable_send:=false stand_only:=true \
  debug_csv_path:="$HOME/mydog_ros2_ws/log/a6850_dry_$(date +%Y%m%d_%H%M%S).csv"
```

看到 `Gyro bias calibrated`，确认没有 fault、反馈过期或传感器异常，然后 Ctrl+C。
干跑不会移动关节，不要求它从当前姿态真的到达 ready。
若帧频或反馈时延不能满足门限，保留日志查原因，不放宽门限继续带电测试。

## 3. 终端 B：监视状态

```bash
ssh jetson@172.19.61.166
source /opt/ros/humble/setup.bash
source ~/mydog_ros2_ws/install/setup.bash
ros2 topic echo /mydog/model6850/status std_msgs/msg/String --field data
```

## 4. 终端 A：柔和站立，等待显式解锁

这条命令 **会使能电机并站立**。先确认吊架支撑、腿部空间和物理急停。
前 5 秒机身完全静止；标定完成、动作方向正确后才逐渐让四脚承重。

```bash
ros2 launch mydog_policy rs01_model6850.launch.py \
  enable_send:=true stand_only:=false \
  debug_csv_path:="$HOME/mydog_ros2_ws/log/a6850_omni_$(date +%Y%m%d_%H%M%S).csv"
```

虽然 stand_only=false，节点仍默认不解锁运动。没有 `/arm` 解锁就一直保持站立。
这样站立和短测在同一进程完成，不需要重复启动标定。
只想站立、不进行全向测试时可改为 `stand_only:=true`，该模式拒绝解锁。

等待状态 `mode=ready`、`walk_start_stable=true`、`trial_armed=false`。
检查关节方向、四脚支撑、温度和反馈；站姿不稳或无法 ready 时不要发送运动命令。

## 5. 终端 C：Isaac 同序或单条短测

```bash
ssh jetson@172.19.61.166
source /opt/ros/humble/setup.bash
source ~/mydog_ros2_ws/install/setup.bash
```

Isaac 同序（5 s 踏步 + 八向各 5 s，中间踏步；vx=±0.2、vy=±0.2、wz=±0.3、组合 0.3/0.1/0.25）：
```bash
ros2 run mydog_policy mydog_model6850_omni_sequence
```
单条短测仍可用。`--seconds` 上限 6。解锁后零速是踏步。航向保护仍可能中途回站并保持使能。

前进：
```bash
ros2 run mydog_policy mydog_model6850_command --vx 0.10 --seconds 3
```
后退：
```bash
ros2 run mydog_policy mydog_model6850_command --vx -0.10 --seconds 3
```
左移（机身 +Y）：
```bash
ros2 run mydog_policy mydog_model6850_command --vy 0.08 --seconds 3
```
右移：
```bash
ros2 run mydog_policy mydog_model6850_command --vy -0.08 --seconds 3
```
左转（从上方看逆时针）：
```bash
ros2 run mydog_policy mydog_model6850_command --wz 0.15 --seconds 3
```
右转：
```bash
ros2 run mydog_policy mydog_model6850_command --wz -0.15 --seconds 3
```

六个单方向均确认正常后，才能尝试低速组合：
```bash
ros2 run mydog_policy mydog_model6850_command --vx 0.08 --vy 0.05 --wz 0.10 --seconds 3
```

## 6. 停止和日志

主动结束运动、柔和回站（终端 C）：
```bash
ros2 service call /mydog/model6850/arm std_srvs/srv/SetBool '{data: false}'
```

也可以直接运行无速度参数的辅助命令：
```bash
ros2 run mydog_policy mydog_model6850_command
```

正常结束节点：先用吊架承重，再在终端 A Ctrl+C。出现反向抽动、明显抖动、异响、
拖脚、倾倒趋势或温度异常时使用物理急停，并结束控制节点。
HTTP 停止备用命令（不能代替物理急停）：
```bash
curl --max-time 2 -sS -X POST \
  'http://127.0.0.1:8000/api/stop?clear_error=false' \
  -H 'Content-Type: application/json' \
  -d '{"motor_ids":[17,18,19,33,34,35,49,50,51,65,66,67]}'
```

CSV 记录 61 维观测、12 路动作/目标与电机反馈；同名 `.csv.status.jsonl` 记录三轴命令、
解锁状态、退出原因、反馈年龄、姿态、温度、限扭矩及里程计诊断。
故障退出后保留日志，重新固定机身查原因，不要靠提高限值反复尝试。
