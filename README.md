# gym_dog

RS01 四足机器人强化学习训练与 Sim2Sim 验证项目。

## Sim2Sim 效果视频

▶ **[点击观看 Sim2Sim 演示视频](./fe204b258b949d205d11b0cb348f7e79.mp4)**

[下载原始 MP4（约 6.4 MB）](https://raw.githubusercontent.com/jujjjk/gym_dog/fix/rs01-heading-odom-soft-inhibit/fe204b258b949d205d11b0cb348f7e79.mp4)

用户提供的仿真效果录像。视频用于展示运动效果，不作为真实机器人部署安全性证明。

## 当前验证进展

- 基线模型：A 组 seed16，`model_6850.pt`；保留站立相位冻结。
- 修复 MuJoCo 桥接中关节状态与机身角速度的采样时序不一致；保留真实 RS01 执行器模型、原策略和原腿里程计，无需为该修复重训。
- 修复后 MuJoCo：12 个指令 × 4 个初始相位，每例 30 秒，48/48 完成。
- 每 5 秒切换动作的 55 秒序列：PhysX 与 MuJoCo 均 4/4 完成。该已测序列包含站立，不等同于后来提供的全程踏步衔接序列。
- 尚未通过整个速度域与 Sim2Real 验收：PhysX 的 0.6 m/s 前进测试仍有失稳，方向精度、机身起伏和足端内收仍需改善。

详见 [传感器时序修复、量化结果与复现命令](unitree_rl_gym/RS01_SENSOR_SYNC_FINDINGS.md)。

## 代码入口

- [训练项目](unitree_rl_gym/README_zh.md)
- [MuJoCo 修正版播放器](mujoko/rs01_go2/sim2sim_sensor_sync.py)
- [MuJoCo 固定动作与切换测试](mujoko/rs01_go2/evaluate_v15.py)（修正版使用 `--sensor-sync`）
- [PhysX 动作切换测试](unitree_rl_gym/legged_gym/scripts/check_rs01_sensor_sync_transitions.py)
- [弹跳、足端间距与里程计诊断](mujoko/rs01_go2/diagnose_support_motion.py)
