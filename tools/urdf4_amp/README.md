# URDF4 V2 AMP + CTS

## 全向方向追踪续训（2026-11-15）

新任务 `urdf4_amp_cts_direction_v1` 将固定参考系路径控制接入训练，AMP+CTS学生仍为1020输入、12动作。
只保留一项修正指令追踪奖励，`heading_gain=0`，避免奖励再次添加转向纠偏；原步态、RS01、PD与随机扰动参数保留。
用户指令范围仍为后退−0.18、侧移±0.10m/s、转向±0.40rad/s；前进先0.25，方向与步态门槛共同通过后可晋级0.40/0.60m/s。
内部转向上限0.55rad/s仅用于纠偏余量；没有将用户转向范围扩大到0.55。
零指令为站立，不采样原地踏步。默认假设有可长期校正的外部定位，模拟定位不代表实机腿里程计精度。

训练前完成100轮真实PPO及3种子×30秒全向测试，报告区分“可开始训练”和“最终全向验收”；未通过最终门槛的工况不会被标记为合格策略。
入口核对检查点、源码与技术/行为回执后，续训到本阶段总计30000轮（含预检100轮）。

```bash
bash /home/nszb/gym/artifacts/urdf4_direction_20261115/start_long.sh
```

该命令会启动新进程；已经运行时不要重复启动。实际PID、日志与输出目录见 `artifacts/urdf4_direction_20261115/training_process.json`。
原生 `play.py --task urdf4_amp_cts_direction_v1 --load_run /absolute/run --checkpoint N --vx .25 --vy 0 --wz 0` 支持新任务，保留鼠标相机控制。

## 方向闭环（冻结原步态，2026-11-15）

新增 `path_tracking.py` / `path_runtime.py`，在固定的起始参考系内纠正航向与横向位置。
默认只对纯前进生效，已验收指令为0.18和0.25m/s；其他动作透传原策略。
大于0.25m/s的前进闭环需显式使用实验选项。全向实验仍有少量步态同步不合格，未默认启用。
本阶段没有训练网络，也没有改PD、RS01执行器或参考数据。

```bash
cd /home/nszb/gym
bash artifacts/urdf4_path_20261115/play_path.sh
# 相同模型关闭闭环作对照：
bash artifacts/urdf4_path_20261115/play_path.sh --controller off
```

演示默认 `--pose_source oracle`，明确使用仿真真值定位；绿色为参考线、黄色为±5cm走廊、红色为实际轨迹，保留鼠标拖动视角。
真实部署需提供外部里程计，不会自动回退为仿真位置。`student`源仅使用公共历史与学生速度预测积分，已实测存在漂移，不作为合格定位来源。
外部数据先用 `FixedOdometryFrame` 一次对齐至路线起点，再调用 `PlanarPathController.update()`；航向参考仅由用户转速（经命令整形）更新，不积分纠偏转速。
接入原策略时只替换当前帧指令，不额外推进观测历史或IMU平均窗口。

入口 `evaluate_path_tracking.py` 支持off/heading/path对照、明确的oracle/student/localization_sim源、60秒固定指令、启动/停止、推力与接触/扭矩记录。
报告、图表及范围见 `artifacts/urdf4_path_20261115/REPORT.md`。以下扩速长训入口属于前一阶段，不能当作本阶段方向闭环训练命令。

扩速与轻扰动阶段使用新任务 `urdf4_amp_cts_expand_v1`，保留原任务和第17800轮基线。新入口为 `train_cts_expand.py`，默认课程上限为前进0.4 m/s、后退−0.25 m/s、侧移±0.15 m/s、转速±0.6 rad/s。实际采样从连续旧范围开始，按学生各方向的速度、接触相位及双对角支撑表现晋级。

本机已验证的续训起点是新阶段第100轮：

```bash
cd /home/nszb/gym
source unitree-rl/bin/activate
python tools/urdf4_amp/train_cts_expand.py \
  --load_run /home/nszb/gym/artifacts/urdf4_expand_20261114/bounded_short100/run \
  --checkpoint 100 --max_iterations 30000 --target_stage 1 \
  --num_envs 4096 --seed 1 --headless --sim_device cuda:0 --rl_device cuda:0
```

这是新阶段总计30000轮，含已完成的100轮；代码没有代用户启动长训。省略`--load_run/--checkpoint`时从原17800权重开始，新阶段计数归零。`--target_stage 2`是后续0.6 m/s的实验范围，尚未验收。

此阶段恢复速度/方向追踪，AMP按指令覆盖权重衰减，明显越界与外力恢复样本不进入D的主要负样本池。25%环境保持名义参数；其余使用窄范围摩擦/质量/质心/电机模型/传感噪声与5–10 N平滑推力。旧范围的名义动作有较弱参考约束，扩速区不被旧动作锚定。直接指令分支采用已验证范围初始化，完整目标仍进入历史编码器、教师上下文和任务奖励。

新阶段仍导出1020维公共历史、12动作；打包训练输入1121维，其中额外1维仅供PPO使用。学生航向来自模拟IMU，真值留给教师。RSI只用于名义环境、完全匹配的参考指令，并统一提交GPU重置。

入口和动作检查：

```bash
python tools/urdf4_amp/smoke_cts_expand.py --output /tmp/expand_smoke \
  --num_envs 128 --updates 3 --headless --sim_device cuda:0 --rl_device cuda:0
python tools/urdf4_amp/evaluate_cts_expand.py \
  --checkpoint /absolute/path/model_N.pt --output /tmp/expand_behavior \
  --seconds 30 --seeds 1,2,3 --headless --sim_device cuda:0 --rl_device cuda:0
```

原生`play.py --task urdf4_amp_cts_expand_v1`已接入；`--sweep`按检查点当前课程档位展示。MuJoCo NumPy桥支持新学生网络的指令分支边界。验证结果与未解决项见[扩速阶段报告](../../artifacts/urdf4_expand_20261114/REPORT.md)。

当前任务为 `urdf4_amp_cts_v1`，版本 `v3_rsi_pair_amp`，实验目录 `urdf4_amp_cts_rsi`。先学习 V2 覆盖的12类运动，0–0.6 m/s 连续全向和鲁棒性尚未验收。

用户后续明确选择“先学步态、后做速度追踪”，对应独立入口：

```bash
cd /home/nszb/gym
source unitree-rl/bin/activate
python tools/urdf4_amp/train_cts_gait_first.py \
  --max_iterations 30000 --num_envs 4096 --seed 1 \
  --headless --sim_device cuda:0 --rl_device cuda:0
```

默认从 `rsi_bootstrap/trust300/run/model_100.pt` 续训到总计30000轮（新增29900轮），实验目录 `urdf4_amp_cts_gait_first`。该入口把显式指令追踪奖励设为0，保留AMP、CTS、原有动作偏离约束和支撑/执行器约束；速度仍统计，不作为该阶段长训启动条件。这是用户明确要求的新训练阶段，不生成或修改动作验收通过回执。优化器状态重新初始化，模型与AMP判别器继续加载；RS01参数不变。可以用 `--load_run /absolute/run/path --checkpoint N` 续训本阶段检查点，`--max_iterations` 仍表示目标总轮数。

流程为实际 RS01 控制指令预热神经网络，再用相邻状态 AMP + CTS/PPO 优化。不能把参考关节位置直接当作 actor 标签。导出的学生网络只读取公共历史，不调用参考控制器。

RSI 同时恢复关节、浮动基座、目标限速器、延迟队列、电机响应、相位和观测历史。RS01 模型、PD、限速和延迟不变。足端参考几何仅统计，风格由 AMP 学习。PPO 使用一项相对预热网络的动作偏离约束，防止短训破坏已有步态。

长训入口：

```bash
bash tools/urdf4_amp/train_cts.sh
```

默认4096环境、8000轮、GPU0，只续训已验收模型。缺少通过回执，或模型/源代码/数据契约不匹配时拒绝启动。不要直接续训旧版本检查点。

本仓库收录训练必需的 V2 数据和名义验证回执：

- `artifacts/urdf4_amp_design_20261113_v2/amp_reference_v2/`
- `artifacts/urdf4_amp_isaac_validation_20261113/validation_receipt.json`

数据原 manifest 保持不变；加载时核对 URDF、全部片段哈希及验证回执。回执只证明此前名义验证通过，不代表新策略或实机验收。

在仓库根目录、已配置的 Isaac Gym 环境中执行有限检查：

```bash
source unitree-rl/bin/activate
python tools/urdf4_amp/smoke_cts_isaac.py \
  --output /tmp/urdf4_cts_smoke --updates 3 --num_envs 128 \
  --headless --sim_device cuda:0 --rl_device cuda:0
python tools/urdf4_amp/evaluate_cts_behavior.py \
  --checkpoint /absolute/path/model_N.pt --output /tmp/urdf4_behavior \
  --seconds 30 --seeds 1,2,3 --modes cold,rsi \
  --headless --sim_device cuda:0 --rl_device cuda:0
```

`play.py --task urdf4_amp_cts_v1 --sweep` 使用原生 Gym 播放学生策略，切换12个V2指令及站立，保留鼠标相机操作；零速度加 `--march` 可查看原地踏步。

`--updates N` 在评估前执行有限 PPO 优化，单次上限500轮；`--approve` 只有通过完整验收才写入长训回执。验收覆盖12类动作、每类4个环境、3个种子、冷启动与 RSI 启动。检查速度/转速、抬脚、真实迈步、对角交替与落地时差、腾空、非法支撑、重置和原始扭矩，不能只凭 AMP 分数放行。

本机预热数据在 `artifacts/urdf4_amp_cts_20261113/rsi_bootstrap/`。重建入口 `prepare_cts_bootstrap.py` 依赖 `artifacts/urdf4_amp_isaac_validation_20261113/commands60/` 的真实控制记录；随后用 `refit_cts_seed.py` 拟合周期控制目标。重建会改变状态库哈希，应重新生成模型与验收记录，保留旧库和旧模型。

本轮结果和限制见 [验证记录](../../artifacts/urdf4_amp_cts_20261113/rsi_bootstrap/REPORT.md)。旧版150轮风格检查不构成当前行为验收。

检查点、完整rollout、运行日志和修改前备份保留在本机，不随本次代码提交上传。`design*`、`physical*` 等为参考数据设计/验证工具，使用单独的 MuJoCo 环境，并需要相应模型输入。

新机器仅克隆代码后，还需要从本机复制配套的 `rsi_bootstrap/reference_bank.pt` 和选定检查点（默认 `rsi_bootstrap/trust300/run/model_100.pt`），或根据真实控制记录重新准备状态库与模型。状态库和模型必须使用同一契约，不能混配。
### 仅扩大直行速度，保留其他动作和轻扰动

任务 `urdf4_amp_cts_expand_x_v1`，入口在 `train_cts_expand.py` 加 `--x_only`。
前进课程上限为 0.40→0.60 m/s；后退始终 −0.18 m/s，横移始终 ±0.10 m/s，
转向始终 ±0.40 rad/s。超过 0.28 m/s 的目标只采直行；斜走/弧线保持旧范围。
域随机化、外力、IMU/关节噪声、执行器、奖励项与原扩速任务相同。
60%保留原12类指令、30%训练新增高速直行、8%站立、2%原地踏步。
旧范围名义环境的原策略动作约束系数1000、容许偏差0.05；扩速及扰动环境不受该约束。
使用独立契约、日志目录和新阶段计数，不覆盖原训练。

```bash
cd /home/nszb/gym
source unitree-rl/bin/activate
python tools/urdf4_amp/train_cts_expand.py --x_only \
  --load_run /home/nszb/gym/artifacts/urdf4_expand_20261114/bounded_short100/run \
  --checkpoint 100 --target_stage 2 --max_iterations 30000 \
  --num_envs 4096 --seed 1 --headless --sim_device cuda:0 --rl_device cuda:0
```

从原扩速任务迁入时使用已验证权重、独立的新优化器和计数；从本任务检查点续训时，
保持优化器和课程状态。0.60是课程目标，必须通过晋级门槛才会开放，并不代表已经达到。

## 新质量 AMP V3 数据（2026-11-16）

使用加上0.8kg电池后的12.3698648317kg URDF重新执行真实物理轨迹。交付51段稳态、18段过渡，148500帧，50Hz；全部通过MuJoCo和逐帧FK/源记录校验。`gym_verified/`另含43段稳态、16段过渡的Gym闭环通过子集。前进最高合格命令0.50m/s，0.60未通过且未加入；V2逐文件hash保留。

数据、速度表、验证收据及原生查看命令见 [V3报告](../../artifacts/urdf4_amp_design_20261116_v3/amp_reference_v3/REPORT.md)。`reference_v3.py`提供变命令、多周期、站立phase mask和严格URDF/hash校验的数据适配器。本轮没有修改训练奖励或启动长训练；新数据需要相应V3训练任务，不能直接替换旧固定周期任务的数据路径。

## 新质量 V3 全向 AMP+CTS 训练入口

现已新增独立任务 `urdf4_amp_cts_v3`，使用上述 Gym 通过子集、连续相位和公开观测学生。以实际关节目标先验初始化12维反馈修正；保留方向伺服、随机化和渐进外力。先练低速全向，按方向/步态验收升级至 x[-0.35,0.60]、y±0.20 m/s、yaw±0.80 rad/s；这些是课程目标。

```bash
cd /home/nszb/gym
source unitree-rl/bin/activate
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/urdf4_amp/train_cts_v3.py \
  --max_iterations 30000 --num_envs 4096 --seed 1 \
  --sim_device cuda:0 --rl_device cuda:0 --headless
```

长训练由用户启动。入口检查 `artifacts/urdf4_amp_v3_training/preflight.json`，默认加载其中选定的40轮初始化权重。60秒×3环境名义全向16/16通过；随机化常规动作13/16通过，高速仍需训练。只运行 `--check-only` 可以分配同样环境并检查一步物理，不更新策略。完整证据、未通过项及原生查看命令见 [V3训练报告](../../artifacts/urdf4_amp_v3_training/REPORT.md)。

## V4 高速反馈修复候选（2026-11-17）

新增独立任务 `urdf4_amp_cts_v4`，保留旧V3。使用原生周期、公开定位速度、相位平衡反馈和执行动作锚定，保持原RS01限制及六项奖励。选定50轮权重，名义全向16/16通过；0.50 m/s在60秒、3种子×3环境下通过。0.60及全强度干扰高速仍未通过，`preflight.json`明确不放行长训。完整证据、限制和原生鼠标查看命令见 [高速修复报告](../../artifacts/urdf4_speed_fix_20261117/REPORT.md)。

入口：`evaluate_v3_policy.py --v4`、`check_v4_contract.py`、`train_cts_v4.py --smoke`。V4公开输入1029，不能与V3的1020混用；部署需要与方向闭环相同的外部校正定位/速度流。新机器还须复制报告中的原始数据、`native_prior.pt`、`balance_temporal2.pt`、`feedback_data/samples.pt`、冻结的原700学生和选定权重；仅克隆代码不能恢复这些本机资产。
