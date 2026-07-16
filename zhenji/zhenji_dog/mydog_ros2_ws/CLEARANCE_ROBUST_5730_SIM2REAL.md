# clearance-robust 5730 真机部署

这套部署同时使用 `5730` ONNX 和导出的动态步态配置。ONNX 负责策略、速度/航向反馈和严格左右对称；ROS 2 节点负责复刻 Gym/MuJoCo 的 50 Hz 动作滤波、0.45 s 相位、参考步态、动作切换后 0.9 s 抬脚增强以及身体倾斜抬脚增强。

模型 SHA256：

```text
a7dd106fb5df1385cbc3c4a0be38916f8e75506a9b2026d87e7671704a9a9b39
```

启动时会强制校验这个哈希、ONNX 内嵌配置、52 维观测、12 维动作、关节顺序、PD、动作缩放和步态参数。模型或配置不匹配时节点会拒绝运行。

## 13 Nm 与仿真一致的含义

启动配置不再使用 8 Nm 悬空测试上限，全局真机预算直接设为 13 Nm。仿真导出的逐关节限幅是每条腿 `hip/thigh/calf = 10/10/13 Nm`，所以最终有效限幅仍为 `10/10/13 Nm`，不是所有关节强行 13 Nm。这样才与 `5730` 的 Gym/MuJoCo 配置一致。

模型 PD 直接使用导出值：

```text
hip:   kp=60, kd=1.2
thigh: kp=70, kd=1.6
calf:  kp=70, kd=1.6
```

旧版 17 Nm 后腿临时增强已关闭，命令平滑、目标平滑、额外动作门控、速度前馈和后腿姿态偏置也全部关闭。零速度命令仍运行训练得到的站立策略，只关闭参考步态，不切换到额外的真机站立控制器。

## 可直接覆盖的文件

将部署包按原目录覆盖到 `/home/jetson` 后，关键文件位于：

```bash
tar -xzf fanfan_clearance_robust_5730_verified_safety_deploy_bundle.tar.gz \
  -C /home/jetson
```

这次不能只覆盖工作空间：电机服务与 ROS 2 策略必须同时更新。

```text
/home/jetson/text/app.py
/home/jetson/text/lingzu_motor.py
/home/jetson/DOG_0.1A/Core/Src/main.c
/home/jetson/DOG_0.1A/RS01_PARAM_READBACK_FLASH.md
/home/jetson/mydog_ros2_ws/src/mydog_policy/setup.py
/home/jetson/mydog_ros2_ws/src/mydog_policy/launch/sim2real_clearance_robust_5730.launch.py
/home/jetson/mydog_ros2_ws/src/mydog_policy/mydog_policy/policy_contract.py
/home/jetson/mydog_ros2_ws/src/mydog_policy/mydog_policy/imu_serial_interface.py
/home/jetson/mydog_ros2_ws/src/mydog_policy/mydog_policy/state_estimator_node.py
/home/jetson/mydog_ros2_ws/src/mydog_policy/mydog_policy/state_estimator_fixed_node.py
/home/jetson/mydog_ros2_ws/src/mydog_policy/mydog_policy/state_estimator_contract.py
/home/jetson/mydog_ros2_ws/src/mydog_policy/mydog_policy/obs_builder.py
/home/jetson/mydog_ros2_ws/src/mydog_policy/mydog_policy/mydog_policy_node.py
/home/jetson/mydog_ros2_ws/src/mydog_policy/mydog_policy/sim2real_parity_fixed_node.py
/home/jetson/mydog_ros2_ws/src/mydog_policy/mydog_policy/clearance_robust_contract.py
/home/jetson/mydog_ros2_ws/src/mydog_policy/mydog_policy/sim2real_clearance_robust_node.py
/home/jetson/mydog_ros2_ws/src/mydog_policy/mydog_policy/clearance_robust_command.py
/home/jetson/mydog_ros2_ws/src/mydog_policy/mydog_policy/validate_clearance_robust_model.py
/home/jetson/mydog_ros2_ws/src/mydog_policy/resource/fanfan_clearance_robust_5730.onnx
/home/jetson/mydog_ros2_ws/src/mydog_policy/resource/fanfan_clearance_robust_5730.json
```

`text` 下的服务会依次写入并读回 RS01 的 `0x700B limit_torque` 和 `0x7018 limit_cur`。每条腿 hip/thigh/calf 为 `10/10/13 Nm` 与 `12/12/16 Apk`。参数掉电丢失，因此每次策略启动都会重新配置12个电机；任何读回不一致都会STOP全部电机并拒绝策略启动。

## 必须更新两块 STM32

本版本新增SPI参数读回帧，旧STM32固件不能使用。使用Keil工程 `/home/jetson/DOG_0.1A/MDK-ARM/DOG_0.1.uvprojx` 分别构建并刷写：

- A板编译定义：`BOARD_IS_A=1`，对应电机 `0x11–0x23`。
- B板编译定义：`BOARD_IS_A=0`，对应电机 `0x31–0x43`。

不要把A板固件同时刷入两块板。刷写完成并重启 `app.py` 后检查：

```bash
curl -s http://127.0.0.1:8000/api/state | \
  python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["0x11"]["board_capabilities"], d["0x31"]["board_capabilities"])'
```

两项都必须是 `3`：bit0为批量CAN，bit1为参数读回。任意一项为 `1` 都表示对应STM32仍是旧固件。

## IMU独占检查

启动ROS前检查串口只能由本次launch中的状态估计节点打开：

```bash
readlink -f /dev/myimu
sudo lsof "$(readlink -f /dev/myimu)"
```

停止所有旧IMU测试程序后再启动。新代码会锁定 `/tmp/mydog_imu_dev_myimu.lock`，厂家接收线程死亡或IMU数据超过100 ms未更新时，状态估计停止发布；策略连续5周期失去状态后会锁存故障并调用 `/api/stop`，排障后必须重启节点，不会自动恢复。

更新文件后，先停止旧服务，确保只有一个进程占用两路SPI，再以前台方式启动新版 FastAPI/uvicorn：

```bash
sudo pkill -f 'uvicorn app:app' || true
cd /home/jetson/text
sudo -E env LINGZU_STATE_REFRESH_HZ=50 \
  python3 -m uvicorn app:app --host 0.0.0.0 --port 8000
```

保持这个终端运行；需要后台常驻时再把同一条命令写入现有 systemd 服务，不要同时保留手动进程。另开终端检查：

```bash
curl -s http://127.0.0.1:8000/api/rs04/verified_motion_safety_limits
```

服务刚重启时应返回 `configured=false`、`verified=false`、`count=0`。如果接口为404，说明仍在运行旧版服务。

## 构建与模型校验

```bash
cd /home/jetson/mydog_ros2_ws
source /opt/ros/humble/setup.bash

# 新增 Python 模块后必须清理这个包的旧增量构建缓存。
rm -rf build/mydog_policy install/mydog_policy
colcon build --packages-select mydog_policy --symlink-install
source install/setup.bash

# 先确认新模块确实来自当前工作空间。
python3 -c "import mydog_policy.clearance_robust_command as m; print(m.__file__)"

ros2 run mydog_policy mydog_validate_clearance_robust_model \
  src/mydog_policy/resource/fanfan_clearance_robust_5730.onnx
```

校验成功时会输出 `clearance-robust 5730 ONNX validation PASSED` 和上述 SHA256。

## 先做不发电机命令的预演

先启动电机状态服务和 IMU/状态估计依赖，再运行：

```bash
ros2 launch mydog_policy sim2real_clearance_robust_5730.launch.py \
  enable_send:=false \
  debug_print_arrays:=true
```

确认 12 个电机在线、关节语义正确、IMU 重力方向直立时接近 `[0, 0, -1]`，且日志显示：

```text
task=FanfanOmniClearanceRobustCfg
strict_symmetry=true
active torque limits ... 10/10/13
```

## 真机启动

机器人可靠悬空、急停可用后启动：

```bash
ros2 launch mydog_policy sim2real_clearance_robust_5730.launch.py \
  enable_send:=true \
  enable_tilt_protection:=false \
  debug_csv_path:=/home/jetson/mydog_ros2_ws/log/clearance_robust_5730.csv
```

安全参数配置会先STOP全部电机。启动站立首帧随后在失能状态预写实时姿态目标，设置运控模式，一次性ENABLE全部12个电机并立即重发相同目标；后续50 Hz周期不重复使能。启动日志必须依次出现：

```text
RS01 volatile safety limits accepted
readback_verified=True
[STARTUP_STAND] one-shot live-target prime and all-motor ENABLE acknowledged
```

如果第三行没有出现，节点不会推进启动站立；不要手动调用 `/api/enable`。随后再次执行：

```bash
curl -s http://127.0.0.1:8000/api/rs04/verified_motion_safety_limits
```

应返回 `configured=true`、`verified=true`、`count=12`，并显示每条腿力矩为 `10/10/13 Nm`、电流为 `12/12/16 Apk`。这是电机通过CAN返回的参数值，不只是服务端写入记录。启动站立阶段仍使用13 Nm停止阈值，不使用8 Nm。

新限幅路径直接发送策略原始位置目标、模型 Kp/Kd 和零力矩前馈，由 RS01 在内部控制周期连续限制输出。它替代旧版每 20 ms 计算负 `t_ff` 抵消 PD 的方式；后者会在下一次 ROS 更新前持续减小电机力矩，是本次悬空数据中 calf 约 100 ms 跟踪滞后和大量限幅周期的主要软件原因。

## 本次修复后的悬空验收

RS01 说明书给出的运动模式控制律为：

```text
t_ref = Kd * (v_set - v_actual) + Kp * (p_set - p_actual) + t_ff
```

说明书同时定义 `0x700B limit_torque`（0–17 Nm）、`0x7018 limit_cur`（0–23 Apk）和默认10 ms主动状态上报。新STM32固件会转发type-17参数读回。状态估计器把用于腿部里程计的同一份12关节快照附在ROS消息中，策略直接使用它，不再独立HTTP轮询，因此base速度、q和dq属于同一板端时刻。

保持悬空并依次运行 zero、low forward、左右横移和左右转向，每个动作 3 秒。新 CSV 应满足：

- 所有 `*_command_torque_ff` 接近 0；不再出现旧版持续负前馈。
- `q_target_raw` 与 `q_target_safe` 一致（机械位置安全夹紧触发时除外）。
- 同步原因应持续为 `ok`，不应再出现周期性 `snapshot_tick_lag_*`。
- 日志不得出现 `task_serial_receive`、`SerialException`、`IMU backend invalid` 或 `FAIL_SAFE_STOP`。
- calf 相位滞后由旧数据约 100 ms 降到不高于 40 ms 为理想目标。
- 下地前建议 calf 位置误差 P95 不高于 0.15 rad、任一关节触及限幅的周期占比低于 10%、连续限幅不超过 40 ms。

只要上述跟踪指标没有明显改善，就先不要下地。下一步应检查 `0x700B` 是否被电机实际应用、电源压降/电流能力、机械摩擦和传动间隙，而不是继续提高 13 Nm。首次落地仍需吊带承重和随时急停，先 zero，再 low forward 1 秒点动，最后才逐步增加到仿真命令。

## 对标 MuJoCo 的 45 秒连续切换

另开终端：

```bash
cd /home/jetson/mydog_ros2_ws
source install/setup.bash

ros2 run mydog_policy mydog_clearance_robust_command \
  --profile parity \
  --segment-sec 5 \
  --repeat 1 \
  --rate 20
```

该序列与 `mujoko/sim2sim.py --demo-matrix --segment-duration 5` 一致，动作之间不插入零速度空档。左右完整扩展测试使用：

```bash
ros2 run mydog_policy mydog_clearance_robust_command \
  --profile full --segment-sec 5 --repeat 1 --rate 20
```

单动作持续运行示例：

```bash
ros2 run mydog_policy mydog_clearance_robust_command \
  --profile parity --action left_lateral --rate 20
```

按 `Ctrl+C` 后发布器会连续发送零速度命令，策略回到学习到的站立状态。

## `ModuleNotFoundError` 处理

如果 `ros2 run` 报错：

```text
ModuleNotFoundError: No module named 'mydog_policy.clearance_robust_command'
```

说明 console script 已更新，但 `--symlink-install` 仍引用旧的 Python 包缓存。执行：

```bash
cd /home/jetson/mydog_ros2_ws

test -f src/mydog_policy/mydog_policy/clearance_robust_command.py
grep -n mydog_clearance_robust_command src/mydog_policy/setup.py

source /opt/ros/humble/setup.bash
rm -rf build/mydog_policy install/mydog_policy
colcon build --packages-select mydog_policy --symlink-install \
  --event-handlers console_direct+
source install/setup.bash

python3 -c "import mydog_policy; print(mydog_policy.__file__)"
python3 -c "import mydog_policy.clearance_robust_command as m; print(m.__file__)"
```

第二条导入命令应输出：

```text
/home/jetson/mydog_ros2_ws/build/mydog_policy/mydog_policy/clearance_robust_command.py
```

不同 ROS 2/colcon 版本也可能直接指向 `src/mydog_policy/mydog_policy/clearance_robust_command.py`，两者都正常。随后重新运行 `ros2 run` 即可。

## 真机与仿真核对项

- 策略频率稳定在 50 Hz，控制周期约 20 ms。
- 观测顺序为 52 维：机身线速度、角速度、投影重力、命令、关节位置误差、关节速度、上一帧滤波动作、相位正余弦、航向误差正余弦。
- 直行稳态 calf 参考幅度为 `-0.30 rad`。
- 横移/斜行 calf 参考幅度为 `-0.42 rad`，纯转向为 `-0.35 rad`。
- 动作切换后的前 0.9 s 幅度乘以 1.10；横移、转向或切换期间再按投影重力倾斜量增强，最大 1.28 倍。
- CSV 中 `*_gait_offset` 是实际叠加的动态步态偏置，`*_action_filtered` 是送入动作缩放前的滤波策略动作。
- 命令切换不会重置步态相位、策略动作滤波器或累计航向目标，与 MuJoCo 连续动作矩阵一致；只重置 0.9 s 动态抬脚计时。

真实地面的摩擦、总线延迟和电机温升不会与仿真完全相同。首次落地应先运行 `low` profile，并保留急停人员；确认关节方向和 IMU 正确后再运行 parity/full 序列。











ros2 run mydog_policy mydog_clearance_robust_command \
  --profile low \
  --action forward \
  --segment-sec 3 \
  --repeat 1 \
  --rate 20


ros2 run mydog_policy mydog_clearance_robust_command \
  --profile low \
  --action left_lateral \
  --segment-sec 3 \
  --repeat 1 \
  --rate 20



ros2 run mydog_policy mydog_clearance_robust_command \
  --profile low \
  --action right_lateral \
  --segment-sec 3 \
  --repeat 1 \
  --rate 20


ros2 run mydog_policy mydog_clearance_robust_command \
  --profile low \
  --action left_yaw \
  --segment-sec 3 \
  --repeat 1 \
  --rate 20



ros2 run mydog_policy mydog_clearance_robust_command \
  --profile low \
  --action right_yaw \
  --segment-sec 3 \
  --repeat 1 \
  --rate 20



ros2 launch mydog_policy sim2real_clearance_robust_5730.launch.py \
  enable_send:=true \
  enable_tilt_protection:=false \
  debug_csv_path:=/home/jetson/mydog_ros2_ws/log/clearance_robust_5730.csv
