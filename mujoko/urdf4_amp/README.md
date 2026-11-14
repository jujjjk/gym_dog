# URDF4 AMP/CTS MuJoCo 闭环回放

当前导出 `model_17800.pt` 的学生网络，1020维公共历史输入、12维关节动作。使用float32 NumPy执行同一ELU/Linear网络，不依赖训练时的教师或参考轨迹。TorchScript→NumPy最大误差4.53e-6。

```bash
cd /home/nszb/gym
bash mujoko/urdf4_amp/play_selected.sh --sweep --seconds 10
```

单独看一个动作：

```bash
bash mujoko/urdf4_amp/play_selected.sh --motion forward_slow
```

动作名：`march forward_slow forward backward left right turn_left turn_right forward_left forward_right arc_left arc_right stand`。

N下一动作、B上一动作、空格暂停、R重新初始化。鼠标旋转/缩放/平移；默认跟随平移，保留鼠标视角。每次换动作重新初始化，保持默认关节目标落地2秒，再执行学生网络；动作过程中不写入基座/关节状态。跌倒会暂停，按R重启。

12类动态动作已经在MuJoCo完成3种子×30秒验证，全部通过步态、安全与相对Gym足高检查。额外的`stand`能保持平衡，但前脚仍有抬起，静止站立检查未通过。速度误差继续记录，没有作为当前步态阶段的放行门槛。

资源位于 `artifacts/urdf4_mujoco_20261114/`，包含权重、URDF绑定契约、MuJoCo模型、数值对齐检查和逐动作验证。RS01完整保留：2.5ms物理步长、20ms策略、目标限速/加速度限制、5ms响应更新、40ms延迟、31.9ms时间常数、0.928增益、Kp40/Kd1、14N·m电机限幅和0.156N·m库仑摩擦。URDF惯量/质量/几何和关闭自碰撞均保持；接触求解使用MuJoCo的显式参数，与PhysX无法逐位等同。

默认Python为本机 `/tmp/urdf4_validation_env/bin/python`（NumPy2.2.6、MuJoCo3.17.0）。可以设置 `URDF4_MJ_PYTHON=/path/to/python`。新环境仅需安装 `numpy==2.2.6 mujoco==3.17.0`；导出阶段使用原Isaac Gym的`unitree-rl`环境。模型与状态库仍留本机，单独克隆代码需要准备配套资源。

重新导出和验证：

```bash
source unitree-rl/bin/activate
OPENBLAS_NUM_THREADS=1 python mujoko/urdf4_amp/export_student.py --output artifacts/urdf4_mujoco_20261114
/tmp/urdf4_validation_env/bin/python mujoko/urdf4_amp/sim2sim.py --prepare
OPENBLAS_NUM_THREADS=1 /tmp/urdf4_validation_env/bin/python mujoko/urdf4_amp/validate.py \
  --output artifacts/urdf4_mujoco_20261114/recheck --seconds 30 --seeds 1,2,3
```

`check_parity.py`读取仓库附带的确定性`control_fixture.npz`，对照原Torch执行器的1000子步测试，并检查600个独立参考姿态的FK、重力与机体速度坐标系。其状态写入仅限独立数值检查；运行前仍需准备本机学生权重和MuJoCo场景。

不改原训练模型或参数，不自动训练、部署或推送。完整结果见 [验证报告](../../artifacts/urdf4_mujoco_20261114/REPORT.md)。
