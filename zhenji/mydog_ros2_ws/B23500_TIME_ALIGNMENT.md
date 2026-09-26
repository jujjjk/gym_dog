# B23500 时间链路修复（2026-09-26）

## 已确认原因
IMU RAW/QUAT 实测约25Hz，而策略50Hz。厂商协议只有传感器值，没有采样tick；
MCU tick对应CAN反馈接收，并非电机内部采样时间。旧实现混合最新值，默认reception允许大skew。

## 修改
- B23500启动使用厂商set_report_rate(100)，并在checksum通过的parser处独立记录RAW/QUAT的monotonic时间与值，保留128帧历史。
- RT服务端缓存epoch改用monotonic，响应中的timestamp与cache_age在同一临界区取值；客户端不再用整个RTT回推对齐epoch。
- 两块MCU分别unwrap uint32 tick并拟合host=a*board+b，处理回绕、重复、复位、时钟跳变。低延迟边界估计offset，拟合残差门限5ms；残差不代表绝对采样精度。
- 默认common_time：t_obs=monotonic_now-50ms，12路电机q/dq、gyro线性插值，quaternion最短弧SLERP。同一时刻的姿态、重力和gyro进入actor；actor里程计重新使用该时刻q/dq/gyro。
- 必须有真实前后样本且间隔<=60ms，不做外推；缺失则拒绝arm/回站。所有PD、关节、姿态、电机故障保护继续使用最新反馈。
- 控制命令未增加总运行时间限制，Ctrl+C仍disarm并由A控制器软回站。
- CSV新增aligned_q/dq/gyro/gravity/yaw、t_obs、原始接收skew、插值间隔和MCU拟合诊断，保持warmup/成功/缺失状态列集合一致。

## 指标解释
observation_alignment_verified=True代表接收时间估计轴上的公共时刻重采样成功。
observation_resampled_skew_ms=0是共同插值目标，不是硬件采样误差为0。
acquisition_sync_verified仍False，真实采样时钟不存在时不得改成True。
对真正硬件采样同步，需要IMU设备采样计数/触发及电机采样时刻信息，不能仅靠Jetson软件伪造。

## 验证记录
IMU设置后RAW/QUAT中位帧间隔约10.1ms，12个电机在线。
只读采样1000周期，预热100周期后900/900成功；对齐耗时中位1.73ms、p95 3.89ms。
插值前后源样本间隔中位30ms、p95 31ms、最大46ms；这是源数据间隔，不是重采样后skew。
没有执行真机运动，持续步行效果尚待操作者验证。

## 启动
原路径start_capture.sh已切换common_time；A重启后重新标定，B命令沿用原来版本。
备份文件后缀.before-time-sync-20260926；服务为用户级lingzu-motor.service。
可用acceptance.py进行读取传感器的独立验证；只调用RT GET_STATE，设置IMU100Hz，不发送电机命令。


## 完整节点验证补充
完整节点下最老一路电机反馈可达45.7ms，40ms回看不能保证已有右侧样本。
最终固定回看50ms，age/源样本间隔门限没有提高，仍不允许外推。
B23500独立100Hz RT GET_STATE后台读取在启动、dry、stand和walk均运行，
使用独立连接且只读缓存；命令响应不再覆盖读线程的新状态。A/B其他型号保持原读接口。

最终已安装控制节点35秒dry验证：采集1609行，预热100周期后1509/1509对齐成功，
observation_temporal_ok全部True，capture dropped=0，error为空。
日志：/home/jetson/mydog_ros2_ws/log/B23500_observation_20260926_110542_6852

最终119项回归通过；common_time下aligned_sensor_skew_ms表示插值后的公共时刻差，原始接收跨度保留在raw_reception_skew_ms。
