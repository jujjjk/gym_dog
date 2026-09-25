# B23500 Observation Phase 1 — 2026-09-23

依据《B18000_V21_部署端Observation链路修改建议》，应用到当前 B23500 V22。
保留 B23500 ONNX/SHA、61D 布局、V13 heading 公式、PD/扭矩/温度保护、
快档连续指令和 Ctrl+C 回站；包含已修复的 uint16 序号回绕判断。
未改 B18000 入口的默认行为。没有远程启动电机。

## 已实现与边界

- IMU 保存最多32个快照。B23500姿态角和重力统一由同一四元数计算，
  控制快照只依赖RAW/QUAT接收时间；单独EULER报文时间保留诊断，不作为使用中数据的新鲜度。
- 每周期保留最新电机反馈和最新有效IMU。默认 `observation_timing_mode=reception`，
  不为接收时间估计匹配旧IMU。显式 `strict_host_alignment` 模式仍可选择
  历史快照；历史读取使用内部只读视图，不每周期深拷贝32帧。
- 电机时间使用 `host snapshot - cache age - per-motor age` 估计。
  匹配诊断包含12个电机及使用中的IMU RAW/QUAT报文接收时间跨度。
- 接收时间模式的门控：电机/IMU age <=40ms、时间不倒退/不在未来，
  IMU内部接收跨度<=60ms（沿用串口有效性边界）。10ms跨传感器接收跨度
  保留为 `observation_alignment_target_ok` 诊断目标，不冒充采样同步门槛。
  `strict_host_alignment` 显式模式仍要求跨度<=10ms。两者都不认证硬件同步。
- 真实 acquisition timestamp 尚未由现有 IMU 协议提供，也没有经过验证的
  板端时钟到主机时钟映射。字段明确为 null/CSV空白，
  `acquisition_sync_verified=false`；绝不把接收时间标作采样时间。
- 外参参数 `imu_mount_roll_deg/pitch_deg/yaw_deg` 定义
  `R_base_imu=Rz(yaw) Ry(pitch) Rx(roll)`；同时旋转 gyro、gravity，
  通过 `R_world_base=R_world_imu R_base_imu.T` 计算控制姿态角。
  未提供测量值时保持单位阵，并不声称已完成机械标定。
- 5ms EMA 仅并行预览 dq/gyro，不送入 Policy 或 PD。记录 raw、preview、
  最终观测以比较噪声/时延；Phase 2 是否启用需先验证实际同步和回放。
- 记录 actor 和 guard 各自的原始/滤波速度、置信度、选中的支撑对、合法性、
  逐腿速度及两个候选对角支撑对的残差；无有效原始速度时记 NaN。
- 时间质量/置信度>=0.5/已就绪的 heading consistency 参与质量门控。
  短暂不佳只标为 transient；同一异常项连续>=200ms才撤销授权并柔和回站；时间、里程计、航向分别计时。
  原有严重反馈故障、姿态、掉频等保护继续生效。该200ms是异常容忍窗口，
  **不是动作时长限制**；`--continuous` 没有固定结束时间。
- 不修改低置信度衰减或 heading deadband。文档对旧 B18000 的 2°/0.5
  描述不等同于当前 B23500 的训练 V13 公式，不能直接套用。

## Jetson 启动

新目录：`/home/jetson/deploy_staging/b23500-observation-opt-20260923`。
原 continuous 目录保留。先停止其他控制器，保持支撑。

终端A只读检查（默认不发送电机目标）：

```bash
ssh jetson@172.19.54.119
bash /home/jetson/deploy_staging/b23500-observation-opt-20260923/start_capture.sh dry
```

确认传感器/质量状态后退出 dry 节点，操作者启动有支撑站立：

```bash
bash /home/jetson/deploy_staging/b23500-observation-opt-20260923/start_capture.sh live
# 若已有经机械确认的外参文件：
# bash /home/jetson/deploy_staging/b23500-observation-opt-20260923/start_capture.sh live /home/jetson/imu_mount.yaml
```

终端B：

```bash
ssh jetson@172.19.54.119
source /opt/ros/humble/setup.bash
source /home/jetson/deploy_staging/b23500-observation-opt-20260923/install/setup.bash
export ROS_DOMAIN_ID=99
ros2 topic echo /mydog/model23500/status std_msgs/msg/String --field data
```

站稳、机身水平静止，退出状态显示，再标定 gyro bias：

```bash
ros2 service call /mydog/model23500/calibrate_imu std_srvs/srv/SetBool '{data: true}'
ros2 topic echo /mydog/model23500/status std_msgs/msg/String --field data
```

至少5秒后确认 `imu_calibrated=true`、`timing_ready=true`、`mode=ready`、
`walk_start_stable=true`、`observation_temporal_ok=true`、`obs_quality_ok=true`，
以及 capture 无丢行/错误。新鲜度不满足时先检查日志，不扩大门槛绕过。

连续快档动作仍用原命令，终端B Ctrl+C 结束动作并回站：

```bash
ros2 run mydog_policy mydog_model23500_command --fast --continuous --vx 0.4
```

取消授权/退出程序不是电机断电；硬件故障不能保证回站，失稳使用物理急停。

## 离线外参建议与 Ground Truth

在确认机械水平的平台，有支撑、四脚承重静止采集至少10秒。
使用新 capture 的 `cycles.csv`，选择全程 ready 的连续10–20秒区间。
以下 START/END 为 CSV 的 `timestamp_policy` 单调时钟值，必须替换为实际区间。

```bash
python3 -m mydog_policy.observation_audit /path/to/cycles.csv \
  --start START --end END mount --level-confirmed --output /home/jetson/imu_mount.yaml
```

工具检查静止、关节位移、gyro、gravity 稳定性，生成建议YAML，不自动应用。
重力只能约束倾斜，不能识别安装yaw；输出采用最小旋转假设。
应用外参后必须重新校准 gyro bias，不能把机身实际倾斜当成安装误差。

外部速度验证：在视频中标定同一匀速时间窗和实际前后向位移，
用带符号的距离（前正后负），不要填目标速度反算的距离：

```bash
python3 -m mydog_policy.observation_audit /path/to/cycles.csv \
  --start START --end END ground-truth --distance-m MEASURED_DISTANCE
```

输出测得平均速度、里程计时间加权平均速度和差值。
只有获得真实标定、接收/采样时序证据和外部速度测量后，才决定启用滤波、
调整 estimator fallback 或开展 heading A/B；本次不宣称这些实测目标已达成。

## 2026-09-23 性能修正

只读实测定位到旧版每周期历史深拷贝成本。新版接收模式不读取历史，
严格模式用内部只读历史视图；Compute >12ms 警告最多每秒一次，
完整逐周期时间指标继续记录，50Hz与故障保护未放宽。
新旧版本保留在不同目录，切换需退出旧控制器、加载新环境并重新标定。

B23500的Euler角改为从用于gravity的同一份四元数计算，避免不同报文时刻的姿态与重力混用。
命令客户端在发送arm前等待新鲜度、标定和timing门控，不在仅仅mode=ready时抢先申请。

## 2026-09-23 走后误停修正

从193703_6713会话确认：旧实现把不同原因的瞬态异常累计到同一200ms计时器。
改为 timing/odometry/heading 独立计时和恢复清零，阈值不变，不关闭故障保护。
状态新增 obs_quality_timing_bad_sec、obs_quality_odometry_bad_sec、
obs_quality_heading_bad_sec、obs_quality_stop_reason；命令退出区分状态超时与具体回站原因。
更新部署在原 observation-opt 目录，原文件备份为 .before-quality-fix。
回放只验证记录下来的首次停步窗口，不等于已完成持续真机步行验证。

## 2026-09-23 步行时状态发布负载修正

B23500 ROS状态和附属controller CSV/JSONL以10Hz发布，模式/授权/故障变化立即发布。
cycles.csv仍逐控制周期50Hz采集，控制和电机发送频率、时序保护边界不变。
降低大段状态序列化与发送线程的Python GIL竞争。
掉频原因现包含control/send窗口的样本数、min/median/p95/max、控制间隔与成功发送年龄。
备份后更新原observation-opt目录，启动路径不变；需操作者重启验证实际步行发送时序。
