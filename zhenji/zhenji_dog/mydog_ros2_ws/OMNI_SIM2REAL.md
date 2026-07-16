# Fanfan Omni ONNX 真机部署

目标模型：`FanfanOmniV4Cfg / model_500.pt`，SHA256：

```text
95bd99ad12813a0b21220af70f8a36b75f22dca2c6ecd75d0b5f7fa5a7eebbff
```

部署节点直接复用已验证的 SPI/CAN、50 Hz 状态估计、关节语义映射、启动站姿、
异步 CSV 和扭矩保护。Omni 专用差异包括：直接 yaw-rate 观测、横移镜像 ONNX、
安全命令限幅和命令斜坡。

## 1. 放置模型并构建

把仓库中的模型放到 Jetson 工作区资源目录，或者启动时传入绝对路径：

```bash
cd /home/jetson/mydog_ros2_ws
cp /你的训练仓库/unitree_rl_gym/deploy/deploy_real/models/fanfan_omni.onnx \
  src/mydog_policy/resource/policy.onnx

colcon build --packages-select mydog_policy --symlink-install
source install/setup.bash
mkdir -p log
```

发令前必须通过模型预检：

```bash
MODEL=/home/jetson/mydog_ros2_ws/src/mydog_policy/resource/policy.onnx
ros2 run mydog_policy mydog_validate_omni "$MODEL"
```

输出必须包含 `OK task=FanfanOmniV4Cfg` 和上面的完整 SHA256。

## 2. 不发电机的 dry-run

```bash
RUN=$(date +%Y%m%d_%H%M%S)

ros2 launch mydog_policy sim2real_omni.launch.py \
  onnx_path:="$MODEL" \
  enable_send:=false \
  startup_stand_first:=false \
  debug_csv_path:=/home/jetson/mydog_ros2_ws/log/omni_dry_${RUN}.csv \
  2>&1 | tee /home/jetson/mydog_ros2_ws/log/omni_dry_${RUN}.console.log
```

另一个终端测试指令输入：

```bash
source /home/jetson/mydog_ros2_ws/install/setup.bash
ros2 run mydog_policy mydog_omni_command forward --duration 3
ros2 run mydog_policy mydog_omni_command left --duration 3
ros2 run mydog_policy mydog_omni_command turn_left --duration 3
```

确认 `[LOOP_RATE]` 为 49-51 Hz、`policy_task=FanfanOmniV4Cfg`、所有反馈在线，
并确认观测命令缩放分别为 `vx*2、vy*2、yaw*0.25`。

## 3. 架空发令

机器人必须用吊带架空并准备硬件急停：

```bash
RUN=$(date +%Y%m%d_%H%M%S)

ros2 launch mydog_policy sim2real_omni.launch.py \
  onnx_path:="$MODEL" \
  enable_send:=true \
  startup_stand_first:=true \
  model_kp_scale:=1.0 \
  model_kd_scale:=2.0 \
  motor_torque_limit_nm:=6.0 \
  debug_csv_path:=/home/jetson/mydog_ros2_ws/log/omni_air_${RUN}.csv \
  2>&1 | tee /home/jetson/mydog_ros2_ws/log/omni_air_${RUN}.console.log
```

只有出现以下日志才允许发布动作：

```text
[ONNX] SHA256 verified
[STARTUP_STAND][READY] ONNX default pose is stable.
```

按下列顺序逐个测试，每项先运行 3-5 秒：

```bash
ros2 run mydog_policy mydog_omni_command stand --duration 3
ros2 run mydog_policy mydog_omni_command forward --duration 5
ros2 run mydog_policy mydog_omni_command backward --duration 3
ros2 run mydog_policy mydog_omni_command turn_left --duration 3
ros2 run mydog_policy mydog_omni_command turn_right --duration 3
ros2 run mydog_policy mydog_omni_command left --duration 3
ros2 run mydog_policy mydog_omni_command right --duration 3
ros2 run mydog_policy mydog_omni_command diagonal_left --duration 3
ros2 run mydog_policy mydog_omni_command diagonal_right --duration 3
```

每个命令结束或按 `Ctrl+C` 后，动作发布器都会持续约 0.5 秒发送零指令。

## 4. 命名动作

| 名称 | `vx vy yaw` |
|---|---:|
| `stand` | `0 0 0` |
| `forward` | `0.15 0 0` |
| `backward` | `-0.05 0 0` |
| `left` / `right` | `0 ±0.04 0` |
| `turn_left` / `turn_right` | `0 0 ±0.30` |
| `forward_left` / `forward_right` | `0.12 0 ±0.25` |
| `backward_left` / `backward_right` | `-0.04 0 ±0.15` |
| `diagonal_left` / `diagonal_right` | `0.08 ±0.04 0` |
| `diagonal_turn_left` / `diagonal_turn_right` | `0.08 ±0.03 ±0.15` |

节点还会硬限制真机命令：`vx=[-0.06,0.18]`、`vy=[-0.04,0.04]`、
`yaw=[-0.35,0.35]`，并按 `0.50/0.25/1.50` 每秒的速率平滑变化。

## 5. 日志分析

纯横移、转向和后退不能使用旧的 `--min-cmd-x` 筛选，改用：

```bash
python3 src/mydog_policy/tools/analyze_sim2real_csv.py \
  log/omni_air_时间戳.csv \
  --skip-sec 2 --all-commands --include-startup
```

架空通过后仍需用吊带卸载大部分重量，按“前进→后退→转向→横移→组合动作”顺序
逐项触地。出现高频抖动、关节方向异常、反馈超时、持续扭矩限幅或横移方向相反时，
立即停止发布并执行硬件急停，不要继续组合动作。
