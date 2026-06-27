# Fanfan MuJoCo Sim2Sim

独立环境，不修改 Isaac Gym。使用最新 `model_800.pt`，50 Hz policy、200 Hz MuJoCo控制、RS01 ±17 Nm。

部署端不加入航向补偿或额外控制器，仅用于检验 policy 本身的跨引擎迁移。
复制的 URDF 显式增加了 `world -> Trunk` floating joint，MuJoCo 因而直接保留
原始四腿关节原点和惯量，不再把第一条腿当作转换参考根。

```bash
cd /home/nszb/gym/mujoko
source .venv/bin/activate
python sim2sim.py --duration 20
python sim2sim.py --duration 60 --viewer
```

部署参数不在 `sim2sim.py` 中维护。`export_onnx.py` 从 `FanfanRoughCfg` 和
原始URDF提取关节顺序、初始状态、默认角、观测缩放、command、PD、动作缩放、
力矩限制及参考步态，写入ONNX metadata并生成同名JSON。`prepare_model.py` 也从
该metadata读取仿真步长和初始位置：

```bash
python prepare_model.py assets/fanfan.urdf models/fanfan_scene.xml \
  --policy models/fanfan_best.onnx
```

训练 episode 长度是20秒，因此 viewer 每20秒自动复位。单段20秒验证的横向
位移由修复前约 `-2.16 m` 降至约 `-0.28 m`，最低机身高度约 `0.266 m`。

当前默认 `fanfan_best.onnx` 是 domain-randomization 续训实验中实测迁移最好的
1000轮checkpoint。MuJoCo端没有航向补偿；20秒结果约为前进5.25 m、横漂
-1.79 m。漂移仍存在，说明下一步需要在Isaac中进一步做接触/执行器域随机化，
而不是继续修改MuJoCo控制器。
