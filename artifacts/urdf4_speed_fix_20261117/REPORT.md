# URDF4 高速失败定位与修复验证（2026-11-17）

本记录区分真实参考控制器诊断、辅助采样、纯公开输入策略验收。辅助控制器的成功不作为策略成功；0.60 m/s 没有合格参考数据，不应标成已完成。

## 1. Repository state

工作目录 `/home/nszb/gym`，分支 `fix/rs01-heading-odom-soft-inhibit`，基线提交 `8a1e9ba7c76ea8b366911a94575cd04f61150606`。修改前工作区已有大量未提交文件，状态保存在 `/tmp/urdf4_speed_fix_initial_status.txt`。本次未覆盖 V3 700 权重、旧数据或用户已有修改，未提交/推送。

## 2. URDF and physical-model findings

使用已加 0.8 kg 电池的 `unitree_rl_gym/resources/robots/urdf4/urdf/urdf4_rs01.urdf`，SHA256 `6e66765a5f2f633480ff5902fff5891a586147867436c6a41433659b12f7ee7f`。总质量 12.3698648317 kg，trunk 5.9784426439 kg，大腿/小腿约 0.180/0.186 m，足碰撞半径 0.020 m。电池仍暂按原 trunk 重心处集中质量处理，安装位置和体积惯量未实测。本轮不修改质量、惯量或碰撞几何。

## 3. Control/actuator findings

50 Hz 策略，400 Hz 物理，200 Hz 执行器响应；Kp=40、Kd=1；延迟 40 ms、时间常数 31.9 ms、响应增益 0.928、库仑摩擦 0.156 N·m。hip/thigh/calf 目标速度上限 2.0/2.6/3.2 rad/s，加速度 60/78/96 rad/s²。动作缩放 0.15/0.25/0.25 rad，执行动作截断 ±4。

链路：策略归一化目标 → 限幅 → 关节目标 → 速度/加速度约束 → 延迟/一阶响应 → PD raw torque → 电机 14 N·m 限幅 → 摩擦后的 applied torque。raw、motor、applied 分别统计；applied 可因摩擦略超过 14。17 N·m 是原始请求验收峰值；6 N·m 是连续能力参考，不能用全关节均值代替逐关节 RMS/热验证。仅位置误差 0.05/0.10/0.20 rad 即产生 2/4/8 N·m 的 P 项，还需叠加速度 D 项。诊断放宽速率/加速度、降低 PD 未使 0.60 合格，正式代码保留原执行器限制。

## 4. Baseline evidence

旧模型 `artifacts/urdf4_v3_selection_20261117/best/model_700.pt`：低速全向名义验收 16/16；0.50 m/s 下实际约 0.35 m/s，raw 约 22 N·m。旧模型 30000 轮末期仅粗筛 2/16，不能以最后轮数选模型。

当前质量与同一执行器下，完整参考闭环控制器在 0.50 命令下实际约 0.509 m/s、raw 约 14.4–14.6 N·m（20 s 诊断），原始参考 60 s 记录也通过。说明 0.50 并非简单的电机或质量不可行。参考闭环使用完整状态，仅用于因果诊断和训练标签，未用于纯策略验收。

具体数值、时长和对照模式见 `diagnosis/result.json`、`focused/result.json`、`rate_ablation/result.json`，不要将不同试验的峰值拼成同一次测试。

## 5. Problem-evidence-change matrix

| 问题 | 实验证据 | 机制与修正 | 验证目标 |
|---|---|---|---|
| 高速参考好、策略慢且扭矩高 | 完整闭环约 0.509 m/s；均值周期/旧策略约 0.35 | 周期平均丢失高速平衡修正；用实际执行标签拟合公开输入相位反馈 | 纯策略达到速度且 raw≤17 |
| 方向修正时关节目标失真 | 固定模板查询虽降低扭矩却仅约 0.216 m/s，回放完整执行命令可达约 0.504 | 不同高度、占空比、校准的模板不宜直接充当高速转向导数；高速模板按用户目标选择，方向/速度偏差交给状态反馈 | 保留方向追踪与真实迈步 |
| 相位与数据采样不一致 | 24/25/26 个原生控制拍被重采样成 100 格 | 原生采样与 manifest 相位统一；此项单独修正仍不足 | 训练/播放/导出同相位 |
| PPO 后期丢失动作 | 700 优于 30000，合格 Gym 参考的旧 D 评分仍高 | 冻结教师上下文、限制神经修正、在实际执行动作空间加专家锚定 | 不以奖励增加代替行为验收 |
| 高速位移反馈不足 | 仅运动学估计速度存在约 −0.1 m/s 偏差 | 使用原方向闭环已要求的外部校正定位流提供平面速度，并显式暴露为公开输入 | 不向学生注入隐藏仿真真值 |
| 0.60 缺乏可执行专家 | 原闭环与有限模板搜索均失败 | 保留为未通过边界，不能将 0.50 重标成 0.60 | 先验证轨迹再扩范围 |

`diagnosis/domain_audit.json` 中参考 MuJoCo/Gym 的旧判别器得分均较高，未支持“判别器主要在区分模拟器”的解释。

## 6. Files changed and rationale

- `unitree_rl_gym/legged_gym/envs/urdf4/amp_v4_config.py`、`amp_v4_env.py`：独立 V4 任务、公开定位速度、有限速度补偿，保持原方向和执行器链。
- `unitree_rl_gym/legged_gym/algorithms/urdf4_amp_v4.py`、`urdf4_native_prior.py`、`urdf4_phase_feedback.py`：原生周期先验、公开状态反馈、有限神经修正、实际动作锚定、可携带的学生导出。
- `conservative_ppo.py`：可选辅助 actor loss 回调，默认关闭，旧任务语义不变。
- `envs/__init__.py`、`utils/task_registry.py`：新任务/runner 注册。
- `scripts/play.py`：原生鼠标相机播放 V4，按检查点加载反馈模型与速度环设置。
- `tools/urdf4_amp/{prepare_v4_feedback,fit_phase_feedback,train_cts_v4,check_v4_contract}.py`：有限初始化、拟合、短训/手动长训入口及集成检查；`evaluate_v3_policy.py --v4` 使用同样的纯学生验收。
- 其余 `diagnose*`、`probe*`、`identify*`、`correct_v4_feedback.py` 为诊断/训练标签工具，辅助输出不冒充 AMP 物理轨迹或策略验收。

## 7. Configuration values and why

公开输入 1029：旧 51×20 历史 1020，加用户目标 3、渐变用户命令 3、同一外部定位流的机体平面速度 2、动作幅度 1。私有状态 99 仅供 CTS 训练，打包维度 1130；推理只依赖公开 1029。

0.35–0.45 m/s 平滑切换高速控制。低速以冻结的原700学生为基准，神经修正归一化上限 0.05；高速相位反馈上限 1.0，细修正上限 0.10。动作锚定系数 20，教师 encoder 固定，学习率 2e-5；环境仍只有原来的六项正交奖励。

最终选定 `balance_temporal2.pt` 的variant2（ridge=0.01）、43维公开相位反馈特征；使用 `native_probe/native_prior.pt` 的43段原生周期。45维航向扩展和局部导数版本是保留的未通过实验，不用于最终候选。基础类继承的旧 `feedback_residual_limit=0.35` 字段不参与V4计算；V4实际界限以 `balance_feedback_limit=1.0`、`neural_residual_limit=0.10`、`low_speed_neural_limit=0.05` 及本节描述为准。

速度 PI 仅用于纯前进超过 0.35：P=0.5、I=0.5、补偿 ±0.20 m/s、补偿变化率 0.30 m/s²、滤波 0.30 s，停止清积分并做抗饱和。补偿从周期速度奖励的任务目标中移除，避免把补偿量误当成用户要求的额外速度。高速横向恢复同时耦合到已有航向伺服：增益 2 (rad/s)/m、上限 0.12 rad/s，使用原 yaw 加速度限制；低速全向不启用此耦合。

反馈拟合的反事实局部导数只用于辨识控制器，不进入 AMP 真实状态库；足端保持一致的姿态扰动 FK 最大误差约 0.018 mm，仍不构成动态可行性证明。

## 8. Validation commands and results

见下文最终结果及相邻 JSON/rollout。所有策略行为评估关闭自动重置，禁用教师控制器，不在运行中写入根状态/关节状态；含 6 s 停止测试。扭矩峰值含启动和停止，速度/方向/步态窗口从第6秒开始。名义与随机化分别报告，未通过项保持可见。

CUDA 导出曾因整数 tensor remainder 被本机 TorchScript fuser 错误编译而失败，已改为等价索引回绕并通过 GPU 推理对照。集成检查覆盖公开/私有输入隔离、PPO 均值与执行动作一致、JIT、编码器梯度、速度环抗饱和与停止。

## 9. Training command or staged plan

本轮只运行有限初始化、反馈辨识、有限 PPO 冒烟和物理评估，没有启动长训练。保留低速全向 → 合格高速 → 随机化/干扰 → 新速度参考的阶段门槛。`train_cts_v4.py` 的长训入口要求匹配检查点、数据、源代码的验收回执；未通过时不生成放行记录。训练准备通过不等于部署验收。

## 10. Evaluation table

最终候选为 `selected/model_50.pt`，由 `ppo_fast50/model_50.pt` 原样保存网络权重，重新绑定当前运行代码和导出；`selected/provenance.json` 记录两者哈希与逐 tensor 相等检查。0.5 通过后的150轮候选仍发生扭矩超标，未选。原生 Python 入口和保存的学生导出使用相同相位、执行器和方向/速度闭环。

以下均为纯公开输入学生、冷启动、60 s 运动+6 s停止；运动统计窗口6–60 s，峰值统计覆盖全程，3个环境/命令。表中速度是机体 x，横向偏移是固定路径的最大误差，航向是平均绝对误差。

| 候选/命令 | seed | 实际速度 m/s | raw峰值 N·m | 最大偏移 m | 航向 ° | 结果 |
|---|---:|---:|---:|---:|---:|---|
| 原700 / x=0.50 | 1 | 0.323 | 25.34 | 0.117 | 21.28 | 不通过，另有腾空/支撑/步态失稳 |
| V4 50 / x=0.50，全向批次 | 1 | 0.5001 | 16.757 | 0.04782 | 0.988 | 通过 |
| V4 50 / x=0.50 | 2 | 0.5028 | 16.287 | 0.04461 | 0.934 | 通过 |
| V4 50 / x=0.50 | 3 | 0.5030 | 16.287 | 0.04362 | 0.936 | 通过 |
| V4 50 / x=0.60，全向批次 | 1 | 0.4661 | 23.47 | 1.2647 | 5.077 | 不通过 |
| V4 50 / x=0.50，全强度外力+随机化 | 1 | 0.4239 | 15.996 | 4.1259 | 21.56 | 不通过，步态/方向/跟速 |

原700对照：`../urdf4_v3_selection_20261117/final_700_nominal_s1/result.json`，同60秒×3环境、seed1；V4：`selected_nominal60/result.json`、`selected_fast60_seed{2,3}/result.json`。后两项为高速子集诊断，`core_passed=false` 仅表示没有重新测试完整16类动作，不能把它误解成完整全向通过。

全向名义 `selected_nominal60/result.json`：16/16通过；新增边界中0.50前进、−0.30后退、±0.20横移、±0.60转向均通过。右转−0.80 rad/s未通过；左转+0.80通过。这里只证明实际测试的离散命令，不把它外推为整个连续指令盒均通过。

seed1 的0.50：motor峰值14.0 N·m，applied峰值14.156 N·m，全关节平均绝对motor扭矩2.585 N·m，逐关节RMS为 `[3.414,2.594,3.734,3.017,3.486,3.852,3.666,1.869,4.852,3.760,1.923,4.859]` N·m。raw>17占比0，motor>6占比11.88%；没有完成热稳态验证。

同一0.50测试：0次摔倒、0腾空，对角落地/抬离时差P95为60/40 ms，非法支撑占比0.309%，四脚P95净离地高度约22.5/22.3/25.4/24.4 mm；停止段四足静立比例100%。详细逐帧记录均在相应 `rollout.npz`。

全强度随机化/外力 `selected_robust60_seed1/result.json`：467次外力事件，全向12/16通过，后退、前左斜、后左斜仍有相位/迈步未达标，右前弧线方向未达标；高速0.50不通过。保持未通过，不生成长训放行记录。

`selected_contract_checks.json`：350步、1130/1029维输入、12维动作，GPU JIT误差1.192e-7；改写所有私有输入不影响学生动作，PPO执行均值一致，辅助动作锚定不向两个encoder传梯度，速度积分抗饱和和停止清零通过。`final_smoke52/smoke.json` 为当前代码从50到52轮的2次更新冒烟，不替换选定的50轮权重；另有前序50轮和100轮有限PPO实测。原生 `play.py` 的8秒无界面启动检查通过。

可在原生Gym查看已验证的0.50直行，保留鼠标相机操作：

```bash
cd /home/nszb/gym
source unitree-rl/bin/activate
python unitree_rl_gym/legged_gym/scripts/play.py \
  --task urdf4_amp_cts_v4 \
  --load_run /home/nszb/gym/artifacts/urdf4_speed_fix_20261117/selected \
  --checkpoint 50 --num_envs 1 \
  --vx 0.5 --vy 0 --wz 0 --duration_s 120 \
  --sim_device cuda:0 --rl_device cuda:0
```

有限检查复现：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/urdf4_amp/check_v4_contract.py \
  --model artifacts/urdf4_speed_fix_20261117/selected/model_50.pt \
  --output /tmp/urdf4_v4_contract.json --sim_device cuda:0 --rl_device cuda:0
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/urdf4_amp/evaluate_v3_policy.py --v4 \
  --model artifacts/urdf4_speed_fix_20261117/selected/model_50.pt \
  --output /tmp/urdf4_v4_eval --seconds 60 --copies 3 --extended-frontier \
  --seed 1 --sim_device cuda:0 --rl_device cuda:0
```

评估输出目录必须尚不存在。`--robust --full-push` 复现干扰测试；`--motions fast_forward` 只测对应子集，不构成完整全向验收。

## 11. Remaining risks

0.60 m/s 和更大转向尚无完整通过证据；随机化/外力不能从名义表现外推。当前定位流必须在真机上以一致的延迟、坐标系和噪声提供。电池质心/惯量、逐关节热负荷、MuJoCo 策略迁移和实机尚需单独验证。所有“通过”限定为对应 JSON 的命令、时长、种子和环境设置。

## 12. Rollback instructions

继续使用旧任务 `urdf4_amp_cts_v3` 和原 `artifacts/urdf4_v3_selection_20261117/best/model_700.pt` 即可回到原低速全向策略。新任务、拟合数据及权重独立保存。不要对整个脏工作区执行 `git reset --hard` 或 `git clean`。
