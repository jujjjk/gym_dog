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
tar -xzf fanfan_clearance_robust_5730_deploy_bundle.tar.gz -C /home/jetson
```

如果使用文件名带 `workspace_overlay` 的包，则直接覆盖工作空间：

```bash
tar -xzf fanfan_clearance_robust_5730_workspace_overlay.tar.gz \
  -C /home/jetson/mydog_ros2_ws
```

```text
/home/jetson/mydog_ros2_ws/src/mydog_policy/setup.py
/home/jetson/mydog_ros2_ws/src/mydog_policy/launch/sim2real_clearance_robust_5730.launch.py
/home/jetson/mydog_ros2_ws/src/mydog_policy/mydog_policy/clearance_robust_contract.py
/home/jetson/mydog_ros2_ws/src/mydog_policy/mydog_policy/sim2real_clearance_robust_node.py
/home/jetson/mydog_ros2_ws/src/mydog_policy/mydog_policy/clearance_robust_command.py
/home/jetson/mydog_ros2_ws/src/mydog_policy/mydog_policy/validate_clearance_robust_model.py
/home/jetson/mydog_ros2_ws/src/mydog_policy/resource/fanfan_clearance_robust_5730.onnx
/home/jetson/mydog_ros2_ws/src/mydog_policy/resource/fanfan_clearance_robust_5730.json
```

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

机器人落地、急停可用后启动：

```bash
ros2 launch mydog_policy sim2real_clearance_robust_5730.launch.py \
  enable_send:=true \
  enable_tilt_protection:=false \
  debug_csv_path:=/home/jetson/mydog_ros2_ws/log/clearance_robust_5730.csv
```

启动站立阶段也使用 13 Nm 停止阈值，不再使用 8 Nm。策略阶段固定使用模型的 `10/10/13 Nm` 限幅。

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
