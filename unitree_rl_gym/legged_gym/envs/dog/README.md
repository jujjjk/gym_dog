# dog_rs01_trot — 12 维策略全控 + 真机 RS01 执行器（无 CPG 关节补偿）

## 控制合同

```
target_q = default_stand + action * 0.22
→ 真机辨识执行器链（延迟 / FOPDT / 摩擦 / 限幅）→ 关节力矩
```

- **没有** 足端 CPG、**没有** 开环抬腿关节偏置、**没有** 对角动作投影
- 观测里仍有 `sin/cos(phase)`，只给对角走路**奖励塑形**用，不写进关节指令
- 资源：`dog_urdf/urdf/dog_rs01.urdf`；电机：`rs01shujv`；限幅：RS01 说明书 17/6 N·m

## 默认站立

```
z=0.316；hip=0, thigh=-0.32987297, calf=1.31853104
```

真机上电先跑柔和 0→站立（不训练）：

```bash
cd /home/nszb/gym/unitree_rl_gym
source /home/nszb/gym/unitree-rl/bin/activate
python legged_gym/envs/dog/stand_transition.py --duration 4.0
```

## 训练命令（请自行启动，勿 resume 旧 CPG 残差模型）

```bash
cd /home/nszb/gym/unitree_rl_gym
source /home/nszb/gym/unitree-rl/bin/activate
export PATH=/home/nszb/gym/unitree-rl/bin:$PATH

python3 legged_gym/scripts/train.py --task=dog_rs01_trot \
  --run_name=rs01_direct12_forward_diagonal_v3 \
  --max_iterations=5000 \
  --headless
```

播放：

```bash
python3 legged_gym/scripts/play.py --task=dog_rs01_trot --checkpoint=-1
```

零动作站立抽检（带真实延迟）：

```bash
python3 legged_gym/scripts/play_cpg.py --task=dog_rs01_trot
```

## 对角顺序硬约束（本次修复）

合法走动接触只能是：

1. 四足支撑（对角交接），或
2. **仅** `FL+RR` 腾空，或
3. **仅** `FR+RL` 腾空

禁止：

- **后脚同时腾空**（`RL+RR`）→ 重罚，并按训练课程逐渐收紧到一帧终止
- **两对角交叉抬脚**（一对还没落地，另一对已离地）→ 重罚，并逐渐收紧到一帧终止
- 前双脚同时腾空、侧步等非对角模式

奖励时钟：`period=0.50 s`，`stance=0.66`。两组每 0.25 s
交换一次，目标为约 80 ms 四足承重交接；相位只进入观测与奖励，不生成关节目标。

前进奖励只在“计划中的两只对角脚共同向前摆、另一对角实际承重”时成立；
四脚一直着地不再获得正的对角支撑奖励。训练前 500/1400/2800 次迭代逐步
收紧错误步态的连续容忍帧数，测试和部署始终按一帧严格检查。

仍是 12 维全控 + 真机延迟，无 CPG 关节补偿。

| 项 | 旧 CPG 残差 | 当前 |
|----|-------------|------|
| 12 维动作 | 只学残差 | **完全决定目标角** |
| 开环抬腿 | CPG/IK 写入 thigh/calf | **关闭（幅度=0）** |
| 执行器 | 真机模型 | 真机模型（保留） |
| 相位 | 驱动 CPG | 仅奖励/观测 |

生效配置：`envs/dog/dog_cpg_fixed_config.py`。

## 从 model_2900 微调对称和平衡

独立任务 `dog_rs01_balance` 固定加载
`Jul23_13-59-23_rs01_direct12_forward_diagonal_v3/model_2900.pt`。
它保持 12 维直接输出和 RS01 实测执行器链，不启用 CPG、动作投影或增益补偿。

```bash
cd /home/nszb/gym/unitree_rl_gym
source /home/nszb/gym/unitree-rl/bin/activate
export PATH=/home/nszb/gym/unitree-rl/bin:$PATH

python3 legged_gym/scripts/train.py \
  --task=dog_rs01_balance \
  --headless
```

该阶段运行 1500 次微调迭代，checkpoint 编号从 2900 继续到 4400。
查看最新微调模型：

```bash
python3 legged_gym/scripts/play.py \
  --task=dog_rs01_balance \
  --load_run=-1 \
  --checkpoint=-1
```

## 从 model_4300 压制机身扭动

独立任务 `dog_rs01_body_stable` 固定从平衡微调的最佳
`model_4300.pt` 加载。它直接惩罚机身瞬时角速度和角加速度，并加强滚转、
偏航力矩、动作变化率与关节加速度约束。

```bash
cd /home/nszb/gym/unitree_rl_gym
source /home/nszb/gym/unitree-rl/bin/activate
export PATH=/home/nszb/gym/unitree-rl/bin:$PATH

python3 legged_gym/scripts/train.py \
  --task=dog_rs01_body_stable \
  --headless
```

该阶段运行 1000 次保守微调，checkpoint 从 4300 继续到 5300。

## 从 model_5300 同时降低扭动和扭矩饱和

任务 `dog_rs01_low_twist` 固定加载 `model_5300.pt`。它针对相位0/0.5的
承重交接扭动以及大腿/小腿电机饱和进行低学习率微调。

```bash
cd /home/nszb/gym/unitree_rl_gym
source /home/nszb/gym/unitree-rl/bin/activate
export PATH=/home/nszb/gym/unitree-rl/bin:$PATH

python3 legged_gym/scripts/train.py \
  --task=dog_rs01_low_twist \
  --headless
```

该阶段运行 1000 次，checkpoint 从 5300 继续到 6300。

## 从 model_6100 压缩髋关节摆幅并降低扭矩

任务 `dog_rs01_hip_torque` 固定加载上一阶段严格评估选出的
`model_6100.pt`。它仍是 12 个策略输出直接控制 12 个电机目标，不启用
CPG、动作投影或增益补偿。髋关节目标权限从 0.22 rad 温和缩到 0.18 rad，
并新增髋实际摆幅、髋目标幅度、髋动作变化率和全电机归一化扭矩奖励。
RS01 的 14/16/17 N·m 硬峰值限幅和 6 N·m 持续额定奖励保持不变。

```bash
cd /home/nszb/gym/unitree_rl_gym
source /home/nszb/gym/unitree-rl/bin/activate
export PATH=/home/nszb/gym/unitree-rl/bin:$PATH

python3 legged_gym/scripts/train.py \
  --task=dog_rs01_hip_torque \
  --headless
```

该阶段运行 1000 次低学习率微调，checkpoint 从 6100 继续到 7100。

## 从 model_7000 纠偏直线并保持身体平衡

任务 `dog_rs01_straight_balance` 固定加载髋关节/扭矩阶段严格筛选出的
`model_7000.pt`。它在世界路径坐标中惩罚横向位移、横向速度和横向加速度，
同时加强航向、横滚、偏航与左右接触载荷约束。动作合同保持不变：策略的
12 个输出仍直接生成 12 个电机目标，不增加 CPG、动作投影或关节补偿。

```bash
cd /home/nszb/gym/unitree_rl_gym
source /home/nszb/gym/unitree-rl/bin/activate
export PATH=/home/nszb/gym/unitree-rl/bin:$PATH

python3 legged_gym/scripts/train.py \
  --task=dog_rs01_straight_balance \
  --headless
```

该阶段运行 1000 次低学习率微调，checkpoint 从 7000 继续到 8000。

## 从 model_7700 进一步限制扭胯并保持协调稳定

任务 `dog_rs01_compact_hip` 固定加载直线/平衡阶段长时严格评估选出的
`model_7700.pt`。髋关节直接目标权限从 0.18 rad 收紧到 0.16 rad，并对
单个髋峰值、对角髋实际角度/速度镜像误差，以及髋高速摆动和机身扭动的
耦合进行约束。直线、身体平衡、扭矩和合法对角接触奖励全部保留。

```bash
cd /home/nszb/gym/unitree_rl_gym
source /home/nszb/gym/unitree-rl/bin/activate
export PATH=/home/nszb/gym/unitree-rl/bin:$PATH

python3 legged_gym/scripts/train.py \
  --task=dog_rs01_compact_hip \
  --headless
```

该阶段运行 1000 次极低学习率微调，checkpoint 从 7700 继续到 8700。

## 从 model_7750 按6 N·m连续额定降扭矩并继续纠偏

任务 `dog_rs01_safe6nm` 固定加载紧髋阶段长时评估选出的
`model_7750.pt`。电机热等效状态以2秒时间常数累计：冷态允许说明书峰值，
持续RMS负载接近6 N·m后逐步把可用扭矩降额到连续额定区间。训练前400次
迭代渐进启用降额，测试和播放始终执行完整降额。

```bash
cd /home/nszb/gym/unitree_rl_gym
source /home/nszb/gym/unitree-rl/bin/activate
export PATH=/home/nszb/gym/unitree-rl/bin:$PATH

python3 legged_gym/scripts/train.py \
  --task=dog_rs01_safe6nm \
  --headless
```

该阶段运行1000次低学习率微调，checkpoint从7750继续到8750。

## 6 N·m连续额定重训v2（重新从model_7750开始）

`dog_rs01_safe6nm_v2` 不加载失败的v1模型。它重新加载
`Jul23_21-29-50_rs01_direct12_compact_hip_coord_v1/model_7750.pt`，
把热时间常数延长到8秒，并在1000次迭代内渐进启用降额。连续扭矩和热
惩罚从25%开始、用700次迭代平滑增至100%，避免再次压掉前进奖励。训练
接触终止在500/1100次迭代后由连续3帧收紧到2帧、1帧；播放和严格评估
始终使用1帧规则。

```bash
cd /home/nszb/gym/unitree_rl_gym
source /home/nszb/gym/unitree-rl/bin/activate
export PATH=/home/nszb/gym/unitree-rl/bin:$PATH

python3 legged_gym/scripts/train.py \
  --task=dog_rs01_safe6nm_v2 \
  --headless
```

该阶段训练1500次，checkpoint从7750继续到9250。

## 从 v2 model_8700 优化速度、平衡、对称和直线

`dog_rs01_smooth_straight` 加载严格评测综合最好的
`Jul24_10-48-23_rs01_direct12_safe6nm_path_v2/model_8700.pt`。训练速度集中在
0.13–0.18 m/s，播放命令为0.16 m/s；不使用额外的单向超速惩罚，只通过
“速度跟踪、路径直、身体安静、对角接触正确”联合奖励塑形。扭矩降为次要
目标，但继续使用真实 RS01 电机链、说明书峰值限幅和热降额。

```bash
cd /home/nszb/gym/unitree_rl_gym
source /home/nszb/gym/unitree-rl/bin/activate
export PYTHONPATH=/home/nszb/gym/isaacgym/python:/home/nszb/gym/unitree_rl_gym
export LD_LIBRARY_PATH=/home/nszb/gym/isaacgym/python/isaacgym/_bindings/linux-x86_64:${LD_LIBRARY_PATH}

python3 legged_gym/scripts/train.py \
  --task=dog_rs01_smooth_straight \
  --headless
```

该阶段训练1000次，checkpoint从8700继续到9700。

## 从 smooth-straight model_9200 重做换撑和平稳性

`dog_rs01_smooth_straight_v2` 从上一阶段严格评测综合较好的
`Jul24_13-19-42_rs01_direct12_smooth_straight_v1/model_9200.pt` 继续。
它保持12个独立直接电机输出，不启用目标投影或负载补偿，也不修改 URDF
惯量。该版本针对 model_9200 暴露出的结构问题做了以下重建：

- 原52维观测顺序保持不变，追加RS01反馈帧直接提供的12维电机扭矩和
  12维电机温度（温度除以100归一化），总计76维；热 RMS 和动态限幅只
  保留在底层安全控制器；
- 启动时自动把旧模型 Actor/Critic 的52维首层扩展到76维，新增权重以0
  初始化，不加载不兼容的旧优化器状态；
- 第一阶段固定为实测电机参数中值和固定42.3 ms延迟，不做质量、摩擦、
  电机强度、零位或传感器噪声随机化；
- 用22项有正有负的核心奖励替代大量相关奖励，并关闭负奖励截零和强制
  镜像动作损失；
- 对角错误在训练中连续3帧才终止，在播放/选模中连续2帧终止，避免短
  rollout 学不到长期换撑和直线纠偏；
- CPG 支撑占空比固定为0.70，继续使用6 Nm连续额定热模型以及髋14 Nm、
  大腿16 Nm、小腿17 Nm峰值安全限幅。

```bash
cd /home/nszb/gym/unitree_rl_gym
source /home/nszb/gym/unitree-rl/bin/activate
export PYTHONPATH=/home/nszb/gym/isaacgym/python:/home/nszb/gym/unitree_rl_gym
export LD_LIBRARY_PATH=/home/nszb/gym/isaacgym/python/isaacgym/_bindings/linux-x86_64:${LD_LIBRARY_PATH}

python3 legged_gym/scripts/train.py \
  --task=dog_rs01_smooth_straight_v2 \
  --headless
```

该阶段训练1500次，checkpoint从9200继续到10700。

## 从 structural model_9200 保守纠正直线与偏航

`dog_rs01_straight_guarded` 固定从
`Jul24_15-09-38_rs01_direct12_structural_rebuild_v1/model_9200.pt`
继续，不覆盖原模型。V4冻结完整的9200步态主网络，只训练一个64单元的
对角同步纠偏头。纠偏头读取横向/偏航速度、命令、12个关节位置、12个关节
速度、相位、航向误差和12个RS01反馈扭矩，用真实可部署的扭矩信号判断足端
承重，不使用足底力传感器观测。输出仍合并为策略内部的12路直接电机动作，
每路最大只能改变0.03。训练直接惩罚FL-RR和FR-RL的接触时间差、足高差和
步幅差，同时用落后脚最小离地高度、对角支撑和腾空项保护已有动作。9200
主网络权重不可更新。学习率为 `5e-5`，关闭探索熵，PPO裁剪为0.10；每25
轮保存一次，共训练400轮到9600，最终仍需和原9200做严格回归比较。

```bash
cd /home/nszb/gym/unitree_rl_gym
source /home/nszb/gym/unitree-rl/bin/activate
export PYTHONPATH=/home/nszb/gym/isaacgym/python:/home/nszb/gym/unitree_rl_gym
export LD_LIBRARY_PATH=/home/nszb/gym/isaacgym/python/isaacgym/_bindings/linux-x86_64:${LD_LIBRARY_PATH}

python3 legged_gym/scripts/train.py \
  --task=dog_rs01_straight_guarded \
  --headless
```

## 从 smooth_straight_v1 model_9200 加力矩约束并减少 reset（推荐）

`dog_rs01_torque_from9200_v7` **不要**接着 v5/v6 或 76 维 structural 链训。
它固定加载效果更好的 52 维基线：

`Jul24_13-19-42_rs01_direct12_smooth_straight_v1/model_9200.pt`

相对当前更差的续训链，本阶段做了三件事：

1. **去掉** `DogRs01Robot` 里后脚同抬 / 对角重叠的**单帧立即 reset**，只保留父类 debounce（训练约 80–100 ms，评测约 60 ms）；
2. 在保留峰值 14/16/17 N·m 的同时启用 **6 N·m 连续额定热降额**，并用 `torque_clip` / `motor_*` / `sagittal_motor_saturation` 压 raw PD 饱和；
3. 速度范围收紧到 `0.12–0.16 m/s`，加强速度跟踪、压超速与 2 Hz 摆头；全 actor 微调，并用冻结 9200 作 conservative 参考，避免步态被扭矩惩罚冲垮。

```bash
cd /home/nszb/gym/unitree_rl_gym
source /home/nszb/gym/unitree-rl/bin/activate
export PYTHONPATH=/home/nszb/gym/isaacgym/python:/home/nszb/gym/unitree_rl_gym
export LD_LIBRARY_PATH=/home/nszb/gym/isaacgym/python/isaacgym/_bindings/linux-x86_64:${LD_LIBRARY_PATH}

python3 legged_gym/scripts/train.py \
  --task=dog_rs01_torque_from9200_v7 \
  --headless
```

该阶段训练 1200 次，checkpoint 从 9200 继续到 10400。播放与录 CSV：

```bash
python3 legged_gym/scripts/play.py \
  --task=dog_rs01_torque_from9200_v7 \
  --load_run=-1 \
  --checkpoint=-1

python3 legged_gym/scripts/record_dog.py \
  --task=dog_rs01_torque_from9200_v7 \
  --load_run=-1 \
  --checkpoint=-1 \
  --duration 30
```
