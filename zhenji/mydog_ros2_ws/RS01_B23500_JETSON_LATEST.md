# B23500 Jetson 当前部署源码（2026-09-29同步）

来源：`jetson@172.19.54.119`，策略工作空间 `/home/jetson/deploy_staging/b23500-ready-20260927`；
电机后端 `/home/jetson/text` 同步到仓库 `zhenji/jetson_motor_service`。
生产 Python 代码逻辑与 Jetson 一致（规范化文本换行并清理 app.py 文件末尾空白），本地未提交的其他修改不包含在此次同步中。
B23500 ONNX 与主分支一致，没有替换模型权重。校验清单见 JETSON_SYNC_20260929.json。

当前代码包含 common-time 对齐、周期发送、重力方向支撑判定、航向观测恢复、
回站后确认新 ready 再切换动作、事务式 IMU 标定和站姿未到位诊断。
短时缺失对角支撑允许0.6秒恢复；此数值不是动作时长限制。
实际关节未到位、反馈过期和机械/姿态/扭矩保护仍然独立生效。
历史 Observation 文档描述当时版本，本文件描述本次同步来源和当前启动入口。

## 构建与启动

仓库启动脚本读取脚本所在工作空间的 install/setup.bash，避免误加载旧部署。
可显式设置 B23500_WORKSPACE 指定另一个已构建工作空间。旧入口 start_b23500_observation.sh 转发到同一脚本。

```bash
cd /path/to/gym_dog/zhenji/mydog_ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select mydog_policy --symlink-install
# dry 默认不使能电机；live 仅由现场操作者启动。
bash start_capture.sh dry
```

切换 live 前停止其他硬件控制节点；现场使用 `bash start_capture.sh live`。
B终端 source 同一工作空间的 install/setup.bash，设置 `ROS_DOMAIN_ID=99`。
首次启动稳定 ready 后校准一次，等待 calibration_state=complete；动作之间不用重新标定。

```bash
source /opt/ros/humble/setup.bash
source /path/to/gym_dog/zhenji/mydog_ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=99
ros2 service call /mydog/model23500/calibrate_imu std_srvs/srv/SetBool '{data: true}'
ros2 run mydog_policy mydog_model23500_command --fast --interactive
```

键名+回车：w前进0.4，s后退-0.3，a/d横移±0.3，m踏步，q/e转向±0.6，z回站，x/Ctrl+C退出并请求回站。
每次切换先回站、确认新ready，再启动待请求动作。出现保护不会擅自恢复旧动作。

## 验证及边界

2026-09-29 在 Jetson 最新源码上重新执行部署回归集：158 passed（13.19秒）。
仓库可运行 `bash tools/validate_b23500.sh` 重现该回归集；该脚本不触发硬件动作。
此次同步没有进行实机运动验证，也没有改变正在部署的文件、服务状态或电机使能状态。
运行日志、CSV、缓存和历史备份未入库。
