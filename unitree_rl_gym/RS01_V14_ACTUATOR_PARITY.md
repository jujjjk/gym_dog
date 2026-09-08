# V14：统一反馈时序，保留已辨识 RS01 模型

2026-09-08。基于提交1f9bd76，新增任务`rs01_omni_v14_actuator_parity`，不覆盖V13。保留用户原有11.md等修改。本次没有训练、推送或真机操作。

## 执行结果

两端统一了**执行器计算契约**，并非宣称已完成整机动力学或Sim2Real验证：

|环节|PhysX / MuJoCo|
|---|---|
|策略与目标速度/加速度限制|20ms，50Hz，原参数不变|
|逐电机延迟队列、一阶响应状态|5ms，200Hz，保留原延迟量化与随机化tick含义|
|PD及速度相关库仑摩擦反馈|2.5ms，400Hz，每物理步重算|
|电磁扭矩|原17Nm峰值限幅；减摩擦后不作第二次隐藏裁剪|
|速度语义|32.9867rad/s作为超速有效性边界，检测到就终止/记录，不投影qvel|

保留原逐电机响应增益、约39.5–56.2ms观测延迟、约19.3–38.0ms一阶时间常数、摩擦系数、Kp40/Kd1、每关节0.22rad动作幅度、目标速率2/2.6/3.2rad/s及加速度60/78/96rad/s²。6Nm仍是持续负荷参考，不是硬限幅。没有添加理想位置执行器、额外支撑力或轨迹补偿。奖励、观测、动作语义、采样范围不变。

物理模型仍为`dog_urdf/urdf/dog_rs01.urdf`，质量11.7317368kg，大腿约0.180m、小腿至足约0.202158m，16mm足球；FL+RR / FR+RL步态结构不变。URDF哈希仍为48b177e9977cc3644dd7a84433d82c3b6d9293e9fdc40c33daa35b4194dd11f4。

## “真实RS01”的准确边界

这是**保留真实机器辨识参数的非理想近似模型**，不是已经证明完全等价的数字孪生。

- `dog_urdf/config/rs01_motor_limits.yaml`记录手册的315rpm空载速度；它不等于任意载荷、任意电压下都能硬性保持的速度上限，也不等于协议±44rad/s编码范围。
- `rs01shujv/rs01_readonly_parameters_20260720.json`记录实机17Nm配置及约36V母线等信息；其中CSP速度限制44rad/s不应混为当前运控模式的瞬时物理刹车。
- 为避免PhysX独有的32.99rad/s理想速度墙，V14仅将其内部数值上限移出有效域到1000rad/s；**这不是允许机器人跑到1000rad/s**。两端都逐反馈步检查32.9867，超出使该轨迹失效/终止，URDF审计速度及奖励参考仍保留原值。
- 本次没有发明未经辨识的扭矩-转速曲线、反电动势或热降额曲线。当前17Nm峰值模型不能代表高转速处仍一定有17Nm可用；高速负载能力、电压/温度影响和固件内部反馈带宽仍需要真实资料或台架辨识。
- 400Hz是数值求解选择，不是对真实电机固件频率的断言。不得仅因sim2sim更稳定就直接下地部署。

## 问题 → 改动 → 验证

|问题|改动|验证|
|---|---|---|
|5ms保持力矩激起轻惯量关节数值振荡|两端反馈/积分统一2.5ms；响应辨识保留5ms|256个逐反馈步，实际PhysX类与MuJoCo类状态/力矩对照|
|缩小dt可能悄悄缩短延迟或加快响应|独立响应时钟，延迟ticks仍以5ms计|delay ticks一致，分频计数回归通过|
|引擎独有硬速度墙|共同超速失效语义，不修改关节速度|注入34rad/s两端均检测，速度仍34|
|担心以理想执行器替代RS01|逐字段锁定旧辨识、PD、限制参数|参数及奖励一致性断言通过|

最大误差：响应状态见`actuator_parity.json`；raw torque 9.070e-6Nm、motor torque 9.070e-6Nm、applied torque 9.097e-6Nm。ONNX/Torch随机输入最大动作差3.815e-6。58项新旧回归测试通过；缺pytest环境下直接调用测试函数，所需tmp_path用独立临时目录提供。未声称运行了整个仓库测试套件。

## A6850原权重复测

checkpoint：`logs/rs01_omni_v13_direction/Sep07_14-56-33_v13_direction_seed16_from3850/model_6850.pt`，训练seed16。没有微调权重。

- PhysX：24指令×4环境×30秒×评估种子20260919/20260920；名义参数，速度统计跳过2秒，异常重置14/11次，其中0.60前进23次、0.20前进1次、右移-0.20共1次。评估仍会自动重置，因此次数不是失败环境比例。
- 第二轮完整记录速度保护：超速0次，最大关节速度24.60rad/s。第一轮在增加该统计前已启动，不补造第一轮速度峰值。
- MuJoCo：12指令×4确定性相位0/.25/.5/.75×目标30秒；无噪声/推扰/随机化，不自动重置，第一次物理跌倒或超速即停止。40/48完成，最大关节速度25.01rad/s，超速0次，所有记录有限。
- 旧V13 MuJoCo同12指令×4相位协议为0/48完成。这里比较的是运行链路变化，不是重新训练产生的策略提升。

|MuJoCo指令|30秒完成|
|---|---:|
|站立0/0/0，gait=0|0/4|
|踏步0/0/0，gait=1|4/4|
|前进.20/.40/.60，各组|各4/4|
|后退-.20|1/4|
|左移.20 / 右移-.20|各4/4|
|左转.30|4/4|
|右转-.30|3/4|
|组合.30/.10/.25 / 反向组合-.30/-.10/-.25|各4/4|

通过不等于速度合格：MuJoCo四相位平均，.20前进实际vx=.184、vy=-.012m/s；.40前进实际vx=.327、vy=-.035m/s；.60前进实际vx约.350m/s。零速踏步仍vx约.017、vy约.041m/s。站立与后退失败不能用其他动作平均速度掩盖。

结论：基础执行器契约修正有效，但剩余站立、后退、方向偏移和高速欠速仍需进一步区分动态接触差异与旧策略适应性。**先不要堆奖励或直接启动长训**；此次没有改奖励，也不把40/48当作全向验收。

## 文件与回退

新增V14 config/env、export_v14.py、sim2sim_v14.py、check_v14_parity.py及V14回归测试。公共RS01环境只增加可选响应分频，旧任务默认每原物理步更新，55项旧回归仍通过；V13及更早任务的配置不变。公共play/evaluate允许选择V14并记录超速指标，旧任务不启用新保护。

回退选`rs01_omni_v13_direction`及旧V13 ONNX/场景即可；不需要覆盖或恢复checkpoint。完整结果在`../artifacts/rs01_v14_actuator_parity/`，两份PhysX日志、48组MuJoCo CSV/JSON、接口验证和回归日志均已保存。

## 可视化（不是长训）

PhysX：
```bash
cd /home/nszb/gym/unitree_rl_gym
source /home/nszb/gym/unitree-rl/bin/activate
python legged_gym/scripts/play_rs01_go2_omni.py \
  --task=rs01_omni_v14_actuator_parity \
  --load_run=/home/nszb/gym/unitree_rl_gym/logs/rs01_omni_v13_direction/Sep07_14-56-33_v13_direction_seed16_from3850 \
  --checkpoint=6850 --vx=0.20 --vy=0 --wz=0 \
  --duration_s=30 --seed=20260919 --sim_device=cuda:0 --rl_device=cuda:0
```

MuJoCo：
```bash
cd /home/nszb/gym
source /home/nszb/gym/unitree-rl/bin/activate
python mujoko/rs01_go2/sim2sim_v14.py \
  --scene artifacts/rs01_v14_actuator_parity/scene.xml \
  --policy artifacts/rs01_v14_actuator_parity/model_6850.onnx \
  --command 0.20 0 0 --phase 0 --duration 30 --viewer \
  --output artifacts/rs01_v14_actuator_parity/view_forward02
```

重新导出必须使用`export_v14.py --task rs01_omni_v14_actuator_parity CHECKPOINT OUTPUT.onnx`，再用原`prepare_model.py POLICY.onnx SCENE.xml`生成同契约场景。旧V13导出不能当作V14使用。上述本地绝对路径按实际机器调整。
