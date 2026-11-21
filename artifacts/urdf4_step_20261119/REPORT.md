# URDF4 的 14 cm 上台阶可行性验证

1. 验证基于当前 URDF4 和 RS01 名义执行器。步行模型 `model_2500.pt` 保持原状。
2. 当前质量 12.37 kg，已包含 0.80 kg 电池。大腿和小腿的关节间距约为 0.180、0.186 m。足端是半径 0.020 m 的球，台阶竖直边缘位于 x=0。电池位置和惯量仍沿用原模型的假设。
3. MuJoCo 为 400 Hz，关节目标为 50 Hz。使用 Kp 40、Kd 1、0.040 s 延迟、0.0319 s 响应时间常数、0.928 响应增益和关节类型对应的原目标速率及加速度限制。关节电机扭矩限制 14 N·m，原始 PD 需求验收线 17 N·m。6 N·m 用于统计负荷持续时间。
4. 第一版路径在 9.98 s 两只前脚上台，后腿第一次大步前移时跌倒。该探索采用较早的 IK 命令中心，最终结果已用 V4 的实际关节目标中心重跑。
5. 几何与静态负载可行，动态交接余量小。短步方案的计划最小关节余量为 4.10°，计划点无非足端穿透。对 350 个采样姿态求单向接触力和摩擦约束下的最小单关节峰值扭矩，最吃力的一帧为 11.58 N·m。
6. `tools/urdf4_amp/verify_step_climb.py` 生成足端、重心和关节参考，也记录自由机身动力学。步态不是原全向策略的输出。控制器使用仿真姿态和速度反馈分配支撑力，只施加关节扭矩，台阶接触均由物理求解器处理。起始状态在台阶前的自然站姿。
7. 最佳复跑采用较短后腿步幅、三足支撑下轮流换脚、静态法向支撑力非负约束及适度姿态反馈。仿真为固定前方台阶、名义摩擦和零推力。
8. 复跑在 8.48 s 两只前脚有台面支撑，56.16 s 四足都有台面支撑，最后连续稳定 2.20 s。记录到的最小足底越沿余量 9.21 mm。物理子步未检测到超过 2 N 的非足端碰撞。
9. 本轮为可行性验证，未启动台阶训练。原全向训练和选中的检查点保持独立。
10. 最佳复跑中原始 PD 峰值为 17.23 N·m，发生在约 14.18 s 的右后脚平地前移阶段。电机峰值受限为 14.00 N·m。最高单关节 RMS 为 5.30 N·m；左后小腿累计约 13.62 s 超过 6 N·m，最长连续约 3.45 s。整个序列完成，但原始扭矩超线，未按原验收门槛通过。
11. 尚未做 Isaac Gym 交叉验证、摩擦和起始位置随机化、连续多次攀爬或实物带载转速与温升测试。温度无法由当前名义扭矩仿真得出。本轮未验证倒立及 40 cm 高台。
12. 稳定行走采用原第 2500 轮粗糙面模型。验证脚本和输出放在独立路径。

复跑命令

```bash
cd /home/nszb/gym
source unitree-rl/bin/activate
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/urdf4_amp/verify_step_climb.py \
  --existing-reference --simulate --height .14 \
  --output artifacts/urdf4_step_20261119/h14_current \
  --physical-output artifacts/urdf4_step_20261119/h14_repeat \
  --feed .78 --swing-world-weight 0 --duration-scale .86 \
  --unilateral-support --planar-feedback 1.1 --attitude-feedback .8
```

[运动学](h14_current/kinematics.json) · [静态负载](h14_current/static_load.json) · [物理复跑](h14_best_replay/physical.json)
