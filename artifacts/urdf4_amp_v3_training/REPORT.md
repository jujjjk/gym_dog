# 新质量 URDF / AMP V3 全向训练交付

## 1. Repository state

2026-11-16，`/home/nszb/gym`，分支 `fix/rs01-heading-odom-soft-inhibit`，HEAD `8a1e9ba7c76ea8b366911a94575cd04f61150606`。工作区原本有较多未提交修改；初始状态记录在 `/tmp/urdf4_v3_training_initial_status.txt`。保留 `11.md`、旧数据和旧检查点。本次未提交或推送，**没有启动长训练**。

采用 [RS01 步态训练工作流](/home/nszb/.codex/skills/rs01-quadruped-gait-training/SKILL.md)。本报告的训练就绪判断与实机验收分开。

## 2. URDF and physical-model findings

使用 `unitree_rl_gym/resources/robots/urdf4/urdf/urdf4_rs01.urdf`，SHA256 `6e66765a5f2f633480ff5902fff5891a586147867436c6a41433659b12f7ee7f`。总质量 **12.3698648317 kg**，trunk **5.9784426439 kg**，含增加的0.8 kg电池。大腿/小腿约0.180/0.186 m，足碰撞球半径0.020 m；12关节顺序FL/FR/RL/RR，每腿hip/thigh/calf。

电池仍是原trunk重心处的集中质量近似；没有实测位置、尺寸和附加惯量，不能称为完整实机惯量标定。质量导入证据见 `artifacts/urdf4_battery_mass_20261116/`；运动数据的几何/物理审计见 `artifacts/urdf4_amp_design_20261116_v3/amp_reference_v3/REPORT.md`。本次没有修改URDF、PD或实体碰撞参数。

## 3. Control/actuator findings

策略50 Hz、物理400 Hz、执行器响应200 Hz；Kp=40、Kd=1，延迟40 ms、时间常数31.9 ms、增益0.928、库仑摩擦0.156 N·m。关节目标速率上限2.0/2.6/3.2 rad/s，加速度60/78/96 rad/s²；电机工作限幅14 N·m。17 N·m用作raw PD峰值验收能力，6 N·m是连续工况参考，二者并非同一硬限幅。

零速度下0.05/0.10/0.20 rad位置误差产生2/4/8 N·m原始PD需求；额外1 rad/s关节速度对应1 N·m阻尼项。未采用分相位PD，也未为扩大步幅提高电机能力。`clip_actions`从2调到4只扩大位置目标表示；速率、加速度、延迟、响应和力矩链保持原值。新数据的MuJoCo平衡坐标转换为Gym平衡坐标后再归一化。

动作链：公开指令/连续相位 → 数据关节目标先验 + 12维有界网络反馈修正 → action clip → 关节目标速率/加速度限制 → 延迟/响应 → raw PD → motor/active限幅 → applied力矩。移动修正上限每维1个归一化动作单位，站立0.1；导出包含同一先验、插值和修正运算。

## 4. Baseline evidence

旧1020维方向策略只迁移actor/context编码器作初始化参考，旧critic、AMP判别器、数据归一化不沿用。直接离线拟合虽有较小离线误差，闭环验收仅1/16；补站立/相位后4/16，第一轮纠偏2/16，均未用于长训练种子。相关失败记录保存在 `bc_nominal20/`、`stand_nominal12/`、`corrective1_eval20/`，没有把辅助教师回放当成学生策略验收。

最终选择从V3真实Gym执行目标构造关节先验，网络学习12维反馈修正。先验原凸混合时初始化11/16通过、40轮有限PPO后12/16；局部线性插值修正后，同一权重16/16通过，并延长验证至60秒。该先验是显式结构化初始化，不把它宣称为从零PPO自主学出的步态。

## 5. Problem-evidence-change matrix

| 问题 | 证据 | 修改 | 验证 |
|---|---|---|---|
| 旧质量/固定周期契约不兼容V3 | V3 cadence有1.923/2/2.083 Hz | 独立新任务、hash校验、连续相位 | `contract_checks.json` |
| 小动作输出及侧移裁剪限制扩速 | 旧hip±0.3 rad、侧移修正硬上限0.1 m/s | clip4、可配置侧移输出0.3 m/s | 0.2侧移未被旧裁剪 |
| 离线动作回归闭环走样 | 上述失败验收 | 已执行关节目标先验+可学习修正 | 纯学生16/16名义通过 |
| 右转纠偏被混合轨迹削弱 | 原插值右转航向误差大、圆弧平移串扰 | 8近邻局部仿射拟合，斜率正则1e-4 | 同权重16/16，圆弧速度/方向恢复 |
| AMP可用性与方向/速度目标冲突 | V3变命令、站立、速度超数据边界 | real/fake同条件；速度残差归一化；覆盖外降低风格权重 | 125441有效对，站立和过渡均参与 |
| 需要随机化且保留清晰步态 | 随机化倒退后脚抬脚不足、部分方向偏差 | 25%名义环境+75%随机化，外力渐进 | 未通过项保留，长训继续学习 |

## 6. Files changed and rationale

新核心文件：

- `unitree_rl_gym/legged_gym/envs/urdf4/amp_v3_config.py`：独立任务的速度课程、DR、奖励和数据路径。
- `unitree_rl_gym/legged_gym/envs/urdf4/amp_v3_env.py`：连续相位、站立、平滑变命令、方向伺服、验收计数。
- `unitree_rl_gym/legged_gym/algorithms/urdf4_amp_v3.py`：条件AMP、CTS、先验修正actor、严格保存/加载和JIT。
- `unitree_rl_gym/legged_gym/algorithms/urdf4_v3_prior.py`：仅公开指令/相位的可导出局部线性插值。
- `tools/urdf4_amp/{prepare_v3_reference_policy,train_cts_v3,evaluate_v3_policy,check_v3_contract,write_v3_preflight}.py`：初始化、手动训练入口、纯学生验收及可追溯就绪收据。

兼容改动：`envs/__init__.py`和`utils/task_registry.py`注册新任务；`amp_cts_env.py`仅对显式V3放开固定cadence限制；`path_tracking.py`新参数默认保持旧行为；`scripts/play.py`增加V3原生viewer分支，保留鼠标控制及相同学生输入。`teacher_v3.py`仅新增离线纠偏诊断的可选参数，最终策略运行和验收不调用该教师。失败的有限BC/纠偏工具与日志作为诊断保留，不是推荐训练入口。

## 7. Configuration values and why

| 阶段 | x m/s | y m/s | yaw rad/s |
|---|---|---|---|
| 初始全向方向/步态 | -0.20～0.30 | ±0.10 | ±0.40 |
| 扩展 | -0.35～0.50 | ±0.20 | ±0.60 |
| 最终目标 | -0.35～0.60 | ±0.20 | ±0.80 |

65%采样有Gym验证的参考命令，10%站立、25%连续/边界组合；零命令站立，不安排原地踏步。组合命令受归一化速度椭球约束，各轴最大值不保证同时执行。正反直行、左右侧移、左右转、四象限斜行、正反圆弧和侧移转向均采样。

每阶段至少200更新，连续两个50更新统计窗口达标才升级。名义学生的各方向均需满足周期均速误差≤0.04 m/s、角速误差≤0.10 rad/s、平均横偏≤0.03 m、平均航向误差≤0.035 rad、非法支撑/摔倒比例≤0.001、调度错相≤0.25，各脚抬脚和两组对角支撑均有足够占比；站立也要达标。未通过会停留当前阶段，不按轮数强制加速。

随机化：摩擦0.6–1.1；附加质量±0.6 kg；COM每轴±8 mm；腿link质量0.97–1.03；armature 0–0.001 kg·m²；Kp±5%、Kd±10%；motor strength±5%；响应增益±5%、时间常数±10%、延迟±1响应tick；轻量IMU/编码器延迟和噪声。每4个环境1个名义环境。外力基准8–18 N、持续0.1–0.2 s、间隔6–10 s，强度由25%在120模拟秒内渐增至100%；这些为力施加，不是直接修改速度。

奖励仅6项：command_tracking +1、gait_quality -1、raw_torque_over_peak -0.5、collision -1、action_rate -0.001、termination -10，另加AMP风格系数3。不另叠加航向奖励：方向伺服生成一致的纠偏命令，命令跟踪优化该命令。raw_torque_over_peak沿用已有实现，实际阈值是14 N·m工作上限，不能按名字理解成17 N·m。AMP在推扰恢复/覆盖边界减少约束，避免逼迫恢复动作贴静态模板。PPO lr=3e-5，entropy=0.0005，每更新2次D；teacher encoder前200更新冻结以稳定学生蒸馏目标。

数据使用 `gym_verified` 的43稳态+16过渡，共125441有效相邻对。全数据69段仍保留，未把仅MuJoCo通过的片段冒充Gym通过。先验43段、100相位bin，实际关节目标来源和hash见 `reference_prior.pt`。0.60 m/s和±0.80 rad/s是训练目标，不是已通过能力。

## 8. Validation commands and results

所有行为测试使用公开1020维历史的学生；无全状态教师、无自动reset掩盖失败、无逐帧写根位姿/关节位姿。种子**1**，每动作**3个并行环境**；这不是3个随机种子。运动60秒，再给零命令6秒，统计运动6–60秒及停步末2秒。raw/motor峰值在400 Hz采集；RMS为400 Hz各关节motor RMS，不能替代热模型/实机温升验证。

- `python -m unittest discover -s unitree_rl_gym/tests -p test_urdf4_path_tracking.py -v`：18/18通过，日志 `/tmp/urdf4_v3_path_tests_latest.log`。
- `check_v3_contract.py --sim_device cuda:0 --rl_device cuda:0 --headless`：190步通过；站立/过渡valid pair均119，JIT误差<1e-5、公开输入隔离、PPO执行均值一致、0.2侧移不被旧限幅裁剪。
- `train_cts_v3.py --smoke --seed-checkpoint artifacts/urdf4_amp_v3_training/reference_init/run/model_0.pt --output artifacts/urdf4_amp_v3_training/prior_ppo40 --max_iterations 40 --num_envs 256 --sim_device cuda:0 --rl_device cuda:0 --headless`：40轮有限PPO。
- 更新插值后 `train_cts_v3.py --smoke --seed-checkpoint artifacts/urdf4_amp_v3_training/prior_ppo40/model_40.pt --output artifacts/urdf4_amp_v3_training/affine_ppo60 --max_iterations 60 --num_envs 256 --sim_device cuda:0 --rl_device cuda:0 --headless`：再20轮有限PPO，actor真实更新、AMP valid=1.0、D有限、reset比例0，累计193次外力事件。本次选用经过完整行为验证的40轮权重，不默认选最后模型。
- `evaluate_v3_policy.py --model artifacts/urdf4_amp_v3_training/prior_ppo40/model_40.pt --output artifacts/urdf4_amp_v3_training/affine_nominal60 --seconds 60 --copies 3 --sim_device cuda:0 --rl_device cuda:0 --headless`：16/16名义通过。
- 同命令换output为`affine_robust60`，增加`--robust --frontier`，device改cuda:1：常规13/16、高速2/6通过。402次外力事件是全22动作66环境合计，不是常规动作单独计数。
- 原生`play.py --task urdf4_amp_cts_v3 --load_run /home/nszb/gym/artifacts/urdf4_amp_v3_training/selected --checkpoint 40 --num_envs 1 --duration_s 4 --vx .25 --vy 0 --wz 0 --headless --sim_device cuda:0 --rl_device cuda:0`：正常加载和运行；此次headless检查不宣称做了人工视觉验收。

名义raw峰值 **11.363 N·m**，常规随机化raw峰值 **12.491 N·m**。名义所有动作0摔倒/0腾空/0非法支撑，全部实际停步通过。收据 `preflight.json` 绑定当前源码、URDF、参考数据与选定权重；修改契约后应重新验证。

## 9. Training command or staged plan

长训练只由用户启动。当前种子是 `selected/model_40.pt`，与已验证40轮actor全部参数逐tensor相同，重新保存了当前源码/JIT元数据。`preflight.json` 是训练初始化就绪，不是实机许可；`deployment_accepted=false`。

```bash
cd /home/nszb/gym
source unitree-rl/bin/activate
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/urdf4_amp/train_cts_v3.py \
  --max_iterations 30000 --num_envs 4096 --seed 1 \
  --sim_device cuda:0 --rl_device cuda:0 --headless
```

目标是总迭代30000，从选定40轮种子继续；每100轮保存。日志在 `unitree_rl_gym/logs/urdf4_amp_cts_v3/`，包含metrics.jsonl、契约及导出JIT。训练入口在长训练前严格核对preflight的checkpoint/source/contract。`--check-only`已按4096环境/30000目标完整检查启动链和一步物理，实际PPO更新0；收据`launch_check.json`，PyTorch峰值已分配约1.15 GB（不含所有PhysX/Vulkan和驱动内存）。后续检查点应按同协议评估选择，不以最后轮或总奖励最高替代方向/步态验收。

原生查看当前全向初始化效果（鼠标控制相机）：

```bash
cd /home/nszb/gym/unitree_rl_gym
source ../unitree-rl/bin/activate
python legged_gym/scripts/play.py --task urdf4_amp_cts_v3 \
  --load_run /home/nszb/gym/artifacts/urdf4_amp_v3_training/selected --checkpoint 40 \
  --sweep --num_envs 1 --duration_s 150 --sim_device cuda:0 --rl_device cuda:0
```

## 10. Evaluation table

名义/随机化均使用同一权重、同一命令，60秒，seed=1，各3环境。以下方向量为轨迹相对误差，转弯会相对目标圆弧计算。

| 动作/命令 (m/s,m/s,rad/s) | 名义实际均速 | 最大横偏 cm | 平均航向误差 ° | 随机化验收 |
|---|---|---:|---:|---|
| stand [0, 0, 0] | -0.000, 0.000, -0.000 | 0.00 | 0.02 | 通过 |
| forward_slow [0.18, 0, 0] | 0.207, -0.000, -0.000 | 1.82 | 0.41 | 通过 |
| forward [0.25, 0, 0] | 0.287, -0.000, 0.000 | 1.50 | 0.23 | 通过 |
| backward [-0.15, 0, 0] | -0.155, 0.000, -0.001 | 0.46 | 0.12 | alternation, diagonal_timing |
| left [0, 0.1, 0] | -0.000, 0.102, -0.001 | 0.78 | 0.20 | 通过 |
| right [0, -0.1, 0] | -0.000, -0.101, 0.000 | 0.78 | 0.26 | 通过 |
| turn_left [0, 0, 0.4] | 0.001, 0.002, 0.403 | 1.36 | 0.30 | 通过 |
| turn_right [0, 0, -0.4] | -0.000, 0.000, -0.402 | 1.15 | 0.15 | direction |
| forward_left [0.15, 0.08, 0] | 0.154, 0.082, -0.001 | 1.58 | 0.33 | 通过 |
| forward_right [0.15, -0.08, 0] | 0.154, -0.082, 0.001 | 1.31 | 0.21 | 通过 |
| backward_left [-0.15, 0.08, 0] | -0.168, 0.090, 0.000 | 0.53 | 0.30 | 通过 |
| backward_right [-0.15, -0.08, 0] | -0.154, -0.082, 0.000 | 0.94 | 0.12 | 通过 |
| arc_left [0.2, 0, 0.3] | 0.200, 0.000, 0.301 | 0.96 | 0.21 | 通过 |
| arc_right [0.2, 0, -0.3] | 0.196, 0.001, -0.300 | 3.56 | 0.28 | direction |
| back_arc_left [-0.15, 0, 0.3] | -0.155, 0.002, 0.301 | 3.37 | 0.45 | 通过 |
| back_arc_right [-0.15, 0, -0.3] | -0.153, -0.001, -0.301 | 2.52 | 0.14 | 通过 |

随机化高速边界探索（不作为当前已实现能力）：

| 高速命令 | 通过 | 未通过原因 |
|---|---|---|
| fast_forward [0.5, 0, 0] | False | fall, raw_torque, support, alternation, diagonal_timing, direction, tracking, stop |
| fast_backward [-0.3, 0, 0] | False | alternation, diagonal_timing, tracking |
| fast_left [0, 0.2, 0] | True | — |
| fast_right [0, -0.2, 0] | False | direction |
| fast_turn_left [0, 0, 0.6] | True | — |
| fast_turn_right [0, 0, -0.6] | False | direction |

## 11. Remaining risks

- 常规随机化的倒退存在后脚抬脚不足/对角着地事件超差，右转和右前圆弧最大横偏分别约5.9/7.1 cm；应继续学习，未宣称鲁棒性全部通过。
- 高速前进0.5 m/s在随机化探索中有摔倒/原始扭矩超标，课程初始不开到该范围；0.6 m/s和0.8 rad/s没有合格策略验证。严禁把最终课程上限描述为已达速度。
- 一次seed、3个环境不代表跨种子泛化。平地验证也不代表复杂地形、极端摩擦或无定位实机。
- 方向伺服当前用仿真位姿模拟外部校正定位；公开策略只接收指令/航向信息，不把真值坐标送给student。实机需要有相同延迟/误差契约的外部校正里程计；仅IMU积分无法保证长距离绝对直线无漂移。
- 电池实体重心/惯量、实际电机温升仍待校准。关节RMS/无17 N·m超限不等于通过长期热验收。

## 12. Rollback instructions

停用新任务即可，不覆盖旧训练和旧数据。旧资产精确快照 `urdf4_rs01_no_battery_legacy.urdf` 与旧数据/旧契约配套使用，不能让旧检查点默默配新质量。V2 hash保留；若要调整本V3方案，创建新run并保留本目录的选择收据。工作区有用户原有修改，避免`git reset --hard`或整文件回退。
