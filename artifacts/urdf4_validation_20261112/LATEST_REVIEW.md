# Claude 最新 gyroobs400 复核

结论：偏航持续偏置有所改善，但速度、瞬时偏航误差和部分扰动场景出现退步，暂不建议整体替换旧 validate400 基线。

## 对象与仓库
分支 fix/rs01-heading-odom-soft-inhibit，HEAD 285e1dc19e636c25a66ddd86a72b6570485ad7cb。新增任务代码仍未提交，未修改 Claude 的实现。新权重 `Nov12_19-53-48_gyroobs400/model_400.pt`；旧权重 `Nov12_18-33-33_validate400/model_400.pt`，均位于 unitree_rl_gym/logs/urdf4_omni_v1。

## 改动评价
- 周期平均线速度/偏航误差奖励各权重 0.5；增加 0.5 s（25 步）平均的实测 yaw-rate 误差输入，缩放 4。这比只依赖瞬时 yaw-rate 更容易识别持续偏航，actor 输入使用有噪陀螺仪，没有直接使用仿真真实速度作为新增 actor 特征。
- 单轴命令采样概率从 0.30 改成 0.50，有助于学习手柄常用动作。
- gyro noise sigma 从 0.05 降到 0.01 rad/s，初始 bias 从 ±0.03 降到 ±0.005 rad/s，bias 上限从 0.06 降到 0.01。与仓库 V21 文档的“已校准”假设更接近；文档同时声明无法从该批数据反推未校准 bias，不能把这组参数当作新机器的完整测量结果。
- 观测 235→240，critic 95→96；仍用相同 task/experiment 名称，新旧权重无法直接互换。当前新版播放脚本不能直接加载之前提供的 validate400 播放命令中的旧权重。推荐另设版本名称或自动校验并选用匹配配置。
- RS01 执行器主要参数保持：50 Hz 控制、5 ms 响应更新、40 ms 延迟、31.9 ms 时间常数、响应增益 0.928、Kp40/Kd1、14 N·m motor 限幅。

## 独立验证
13 项单元测试通过。使用 Isaac Gym GPU PhysX、cuda:0、seed=1、11 指令×4 环境×20 s。统计排除前 2 s，跌倒统计覆盖整个运行。旧 actor 在同一新版环境中通过 eval_47.py 去掉每帧新增的第 48 项输入，以保证原 actor 输入语义；该方式不改变旧网络权重。新旧无扰动模式下均 0/44 轨迹跌倒。

|无扰动指标|旧 validate400|新 gyroobs400|评价|
|---|---:|---:|---|
|前进0.3 平均偏航漂移 deg/s|2.4|0.5|改善|
|前进0.6 平均偏航漂移 deg/s|4.6|2.7|改善|
|后退0.4 平均偏航漂移 deg/s|4.8|2.2|改善|
|左移0.3 平均偏航漂移 deg/s|4.2|0.3|改善|
|前进0.6 实际 vx m/s|0.57|0.52|跟踪变差|
|右移0.3 实际 vy m/s|-0.28|-0.23|跟踪变差|
|后退 yaw 逐时刻平均绝对误差 rad/s|0.116|0.361|约3.1倍，偏置变小但波动明显变大|
|后退 motor>6 N·m 采样占比|13.4%|20.2%|增加|
|混合命令平均偏航漂移 deg/s|1.1|5.7|变差|
|混合命令横向漂移 cm/s|0.2|2.9|变差|

平均偏航漂移是每个机器人时间平均误差取绝对值后再跨机器人平均，不会因不同机器人相反漂移互相抵消。但机器人自身随时间正负交替的偏航仍会抵消，因此还必须看逐时刻绝对误差和时序。

新版带扰动测试：前进0.6 m/s有1/4轨迹跌倒，右移-0.3 m/s有1/4轨迹跌倒，其余9组无跌倒。旧策略在相同新版扰动分布下，混合指令有1/4轨迹跌倒，其余10组没有。两者合计分别为新版2/44、旧版1/44。每组仅4个环境、单seed，且网络构造的随机数消耗不同，不能保证逐时刻推力完全相同；不能据此推断总体跌倒概率或认定显著稳定性差异。带扰动速度统计包含自动reset后的片段，不能视为无跌倒的连续轨迹指标。

完整原始表：latest_nominal.log、latest_disturbed.log、latest_baseline_same_config.log、latest_baseline_disturbed_same_config.log；测试日志 latest_tests.log。

## 尚未解决的问题
1. self_collisions=1，训练仍关闭自碰撞。
2. 实际接触仍只统计逐足计划匹配，没有 flight/非法支撑/对角落足时差的实际物理验收。单元测试 no_flight 仍仅检查时钟。
3. 力矩峰值仍只采样20 ms结束的末子步，且剔除启动2秒，不是全部物理子步的 raw/motor/applied 峰值。
4. 占空比仍0.60、抬脚30 mm，仍未向低抬腿高占空比目标调整。
5. 周期平均误差奖励和滤波输入不等于航向保持闭环；无绝对航向/外部参考时，陀螺仪bias仍会导致残余航向漂移。

## 建议
保留两个候选。下一步先检查后退的相位相关 yaw/contact/torque 时序，并把持续偏置和周期内波动分别作为验收项，避免只凭平均漂移或总reward选型。增加多seed、较长时域、指令切换、真实接触和所有物理子步力矩统计后再决定替换。无需本次启动训练。

## 复现
在 unitree_rl_gym 中激活 ../unitree-rl/bin/activate：
```bash
python -m unittest tests.test_urdf4_omni
python legged_gym/scripts/play_urdf4_omni.py --sweep --headless --load_run Nov12_19-53-48_gyroobs400 --checkpoint 400 --duration_s 20 --eval_envs 4 --sim_device cuda:0 --rl_device cuda:0
```
加 `--disturbed` 得到带扰动模式。旧策略比较使用 artifacts/urdf4_omni_v1/eval_47.py，参数见对应日志文件名和脚本。

原模型、配置和权重未改；本次只新增复核报告/日志，无需回滚。
