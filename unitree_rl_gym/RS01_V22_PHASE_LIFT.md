# RS01 V22 phase-lift：当前训练与播放入口

当前任务：`rs01_v22_phase_lift`，非 AMP，从零训练，沿用 V22 的 61 维观测、
12 维动作与真实 RS01 执行器模型。没有强制修改物理关节位置或绕过电机限制。

在原任务上修复：真实离地事件计时、长期不换脚时降低速度收益、原姿态项加入
俯仰范围、从零学习的探索幅度。原奖励名称和权重不变，旧任务仍保留用于对照。
`rs01_stable_phase*` 与 AMP 分支是历史实验，不是当前推荐训练入口。

## 从零训练

```bash
cd /home/nszb/gym/unitree_rl_gym
source /home/nszb/gym/unitree-rl/bin/activate
python legged_gym/scripts/train_rs01_v22_phase.py \
  --task=rs01_v22_phase_lift --num_envs=4096 --max_iterations=30000 \
  --run_name=v22_phase_lift_seed16 --seed=16 \
  --sim_device=cuda:0 --rl_device=cuda:0 --headless
```

这个专用入口拒绝 `--resume`，防止误加载旧模型。训练输出包含
`experiment_contract.json`。长训练由用户自行启动。

## 当前常速演示候选

2026-10-09 筛选 `24250/26450/27700/29750/30000` 后，推荐 **29750 作为常速
演示候选**，不是完整的真机部署验收模型。

- NORMAL 传感器，13 类指令 ×2 环境 ×12 秒：29750 零重置、零腾空。
- 13 个动作每3秒直接切换，4环境共39秒：29750 常速零重置。
- 1.5倍高速切换、1.25倍切换均出现重置；1.5倍60秒定速右横移也出现重置。
- 30000 在常速定速筛选中重置1次，不能直接把最后检查点当最佳。
- 内收、支撑间距、斜后退跟踪与高速稳定性仍需改进，尚无本轮 sim2real 验收。

本次提交只上传代码，**不上传模型权重、训练日志、运动参考数据或临时测试产物**。
以下本地路径需要已有对应模型；其他机器需自行提供模型并修改 `--load_run`：

```bash
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
python /home/nszb/gym/artifacts/rs01_v22_phase_scratch/select_play.py \
  --task=rs01_v22_phase_lift \
  --load_run=/home/nszb/gym/unitree_rl_gym/logs/rs01_v22_phase_lift/Oct08_19-28-46_v22_phase_lift_seed16 \
  --checkpoint=29750 --sequence --duration_s=3 --num_envs=1 \
  --speed_scale=1 --sensor_profile=normal --seed=20261010 \
  --sim_device=cuda:0 --rl_device=cuda:0
```

`--speed_scale` 改的是速度指令，不是播放速度。倍率1对应前后/横移0.2m/s、
转向0.3rad/s；倍率2对应0.4m/s、0.6rad/s，但更高速稳定性没有通过。
移除 `--sequence`，使用 `--duration_s=12 --eval_envs=2 --headless` 可做定速筛选。

## 测试

```bash
python -m unittest discover -s tests -p 'test_rs01_v22_phase*.py' -v
python -m unittest discover -s tests -p test_rs01_omni_v22.py -v
```

这些测试覆盖配置继承、逐脚奖励、悬脚不能重复得分、接触计时、传感器时序等；
不能代替长时间物理评估。历史任务的参考动作数据和模型需另行准备。
