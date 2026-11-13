# Claude URDF4 版本复核

复核对象：任务 urdf4_omni_v1，checkpoint `unitree_rl_gym/logs/urdf4_omni_v1/Nov12_18-33-33_validate400/model_400.pt`。仅复核，没有修改 Claude 的实现或训练权重。

结论：可用作继续开发的全向行走仿真基线。模型来源和关节符号正确，短时固定指令测试通过；尚不构成严格对角步态验收或实机部署验收。

## 独立验证
- 原始 URDF 与训练 URDF 去掉 visual、robot 名称后，全部结构一致，质量/惯性/关节/碰撞未变化。原文件哈希与上次验证相同。
- 执行 `python -m unittest tests.test_urdf4_omni`：12 tests，全部通过。
- Isaac Gym / GPU PhysX，cuda:0，seed 默认 1，固定 checkpoint 400。
- 两种模式各 11 指令×4 环境×20 秒；共 88 条机器人轨迹、1760 机器人秒，均无跌倒。统计去掉前 2 秒，但跌倒统计包含启动阶段。
- 无扰动：名义参数、理想传感器、无推力。
- 带扰动：保留训练随机化/传感器噪声，push.ramp_s=0，启用完整推力范围。4 个环境/指令和单 seed 仅是有限抽样。

|指令|无扰动实际速度|带扰动实际速度|
|---|---:|---:|
|前进 0.30 m/s|0.32|0.30|
|前进 0.60 m/s|0.57|0.56|
|后退 -0.40 m/s|-0.31|-0.35|
|左移 0.30 m/s|0.26|0.25|
|右移 -0.30 m/s|-0.28|-0.29|
|左转 1.0 rad/s|0.90|0.93|
|右转 -1.0 rad/s|-0.94|-0.97|

完整表见 claude_model400_nominal.log / claude_model400_disturbed.log。

## 主要发现（按优先级）
1. **严格对角步态尚未证明。** gait_contact 为逐足布尔匹配率，不是合法支撑模式比例；没有腾空、pace、crawl、单足支撑、对角落足时差的专项统计。`tests/test_urdf4_omni.py` 的 no_flight 检查测试的是时钟计划，不是物理接触。无扰动运动指令下匹配率约 0.81–0.89，不能据此断言始终为对角步态。对应 `urdf4_omni_env.py:606`。
2. **训练关闭自碰撞。** `urdf4_omni_config.py:154` self_collisions=1，尚未证明运行中腿与机身、腿与腿没有相互穿过。
3. **当前步态目标比低抬腿、高占空比目标更激进。** `urdf4_omni_config.py:237` 占空比 0.60、抬脚目标 30 mm，每次交接计划四脚重叠 50 ms；此前技能建议的初始范围为 0.74–0.78 / 15–22 mm，属于设计目标差异而非运行错误。测试中摆动中段平均足底高度约 28–39 mm。`urdf4_omni_env.py:610` 只惩罚抬脚不足，没有抬脚过高项。
4. **力矩表有采样边界。** `play_urdf4_omni.py:183` 附近读取每个 20 ms 步结束的 motor_torques/raw_pd_torques，没有记录全部 2.5 ms 子步峰值；前 2 秒也不计入 peak。所列 14 N·m 是截断后的 motor 扭矩，训练随机 motor_strength 后的 applied 扭矩可能更高；不能据表宣称所有子步 applied 都小于 14。`>cap` 门限是 14 N·m，并非硬件 17 N·m。
5. **导出协议不足以独立复现仿真执行器。** contract.json 包含 PD、动作缩放、目标限速，但没有完整记录约 40 ms 延迟、0.0319 s 时间常数、0.928 响应增益、摩擦项及 5 ms 更新规则。实机是否应复现这些环节需依部署架构决定；不能仅凭该文件认定跨仿真执行器一致。

带扰动下右移 raw 超 14 N·m 的末子步采样比例约 1.0%，后退约 0.4%；运动模式中 >6 N·m 采样比例约 8.7%–16.9%。这些不是热验证。

## 复现
在 `/home/nszb/gym/unitree_rl_gym` 激活 `../unitree-rl/bin/activate` 后执行：
```bash
python -m unittest tests.test_urdf4_omni
python legged_gym/scripts/play_urdf4_omni.py --sweep --headless --load_run Nov12_18-33-33_validate400 --checkpoint 400 --duration_s 20 --eval_envs 4 --sim_device cuda:0 --rl_device cuda:0
python legged_gym/scripts/play_urdf4_omni.py --sweep --disturbed --headless --load_run Nov12_18-33-33_validate400 --checkpoint 400 --duration_s 20 --eval_envs 4 --sim_device cuda:0 --rl_device cuda:0
```

不需要回滚：原实现及权重未修改；仅新增本目录的复核日志及此报告。未启动新训练。
