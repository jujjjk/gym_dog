# 当前选择模型：symmetric-transition 5530

训练、Gym、MuJoCo 和真机部署统一使用：

- task：`fanfan_omni_symmetric_transition`
- run：`Jul14_11-59-33_symmetric_transition_from_force_coord_5280`
- checkpoint：`5530`
- PT：`mujoko/models/fanfan_symmetric_transition_5530.pt`
- ONNX：`mujoko/models/fanfan_symmetric_transition_5530.onnx`

## 从 5530 继续训练

任务配置已经固定 `resume=True`、上述 run 和 checkpoint。执行：

```bash
cd /home/nszb/gym/unitree_rl_gym
PATH=/home/nszb/gym/unitree-rl/bin:$PATH \
CUDA_VISIBLE_DEVICES=0 \
/home/nszb/gym/unitree-rl/bin/python legged_gym/scripts/train.py \
  --task fanfan_omni_symmetric_transition \
  --sim_device cuda:0 \
  --rl_device cuda:0
```

如只使用最终 5530，不需要继续训练。

## Gym

```bash
cd /home/nszb/gym/unitree_rl_gym
FANFAN_PLAY_TRANSITIONS=1 \
/home/nszb/gym/unitree-rl/bin/python legged_gym/scripts/play_omni.py \
  --task fanfan_omni_symmetric_transition \
  --load_run Jul14_11-59-33_symmetric_transition_from_force_coord_5280 \
  --checkpoint 5530 \
  --num_envs 18
```

## MuJoCo

```bash
cd /home/nszb/gym
mujoko/.venv/bin/python mujoko/sim2sim.py \
  --policy mujoko/models/fanfan_symmetric_transition_5530.onnx \
  --viewer --demo-matrix --duration 120 --segment-duration 8
```

## 真机

生产入口 `sim2real_realdata.launch.py` 已回退并固定到 5530。完整流程见
`zhenji/zhenji_dog/mydog_ros2_ws/SYMMETRIC_TRANSITION_5530_SIM2REAL.md`。
