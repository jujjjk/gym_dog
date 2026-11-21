"""Merge measured finalists and publish an immutable preview choice."""
import hashlib,json,shutil
from pathlib import Path
from select_cts_direction import rank

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/urdf4_direction_selection_20261116'
records=[]
for location in [OUT/'native_early/candidate_100.json',OUT/'native_early/candidate_200.json',OUT/'final.json']:
    data=json.loads(location.read_text());records.extend(data if isinstance(data,list) else [data])
assert len(records)==6
for x in records:
    seeds=[r['seed'] for r in x['reports']]
    assert seeds==[1,2,3] or (x.get('early_rejected') and x['summary']['unsafe_groups']>0 and seeds==[1,2])
best=min(records,key=rank);source=Path(best['checkpoint']);destination=OUT/'best';destination.mkdir(exist_ok=True)
target=destination/source.name;shutil.copy2(str(source),str(target))
export=source.parent/'exported'/('student_%d.pt'%best['iteration'])
shutil.copy2(str(export),str(destination/'student.pt'));shutil.copy2(str(export)+'.json',str(destination/'student.pt.json'))
coarse=json.loads((OUT/'coarse.json').read_text())
summary=best['summary']
selected=dict(selected=best,checkpoint=str(target),sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
    scanned_checkpoints=len(coarse),snapshot_last_iteration=max(x['iteration'] for x in coarse),final_candidates=len(records),
    qualified=summary['full_passed']==summary['groups'],fully_static_stopping_qualified=summary['nonstatic_stops']==0,
    status='best measured preview candidate; qualification is reported separately',
    selection_rule='Safety including stops, gait legality, path gates, static stopping, tracking score',
    evaluation_note='100: native inference, 60s x 3 seeds. 200: native 60s x seeds 1/2, early rejected after nominal support failure. Later policies: vectorized 30s x 3 seeds. All use 5s stop and full 5–10N pushes. Shorter late tests are easier, and still fail stop gates.')
if (OUT/'selected.json').exists():shutil.copy2(str(OUT/'selected.json'),str(OUT/'walking_pool_selected.json'))
(OUT/'selected.json').write_text(json.dumps(selected,indent=2)+'\n')
(OUT/'all_final.json').write_text(json.dumps(records,indent=2)+'\n')
rows=[]
for x in sorted(records,key=rank):
    s=x['summary'];seconds=x['reports'][0]['seconds']
    rows.append('| %d | %d | %d/%d | %d | %d | %.2f | %.2f | %.2f |'%(x['iteration'],seconds,
        s['full_passed'],s['groups'],s['gait_failures'],s['stop_safety_failures'],100*s['mean_cross_rms_m'],s['mean_heading_deg'],s['max_raw_Nm']))
report='''# URDF4 全向模型选择（2026-11-16）

## 1. 仓库状态
沿用fix/rs01-heading-odom-soft-inhibit，训练继续运行。仅新增选择器和演示产物，未修改训练网络、奖励、URDF、执行器，未提交推送。

## 2. URDF与物理模型
与[训练报告](../urdf4_direction_20261115/REPORT.md)一致，质量11.569864831733414kg、12DOF、大腿/小腿约0.180/0.186m；足端碰撞球半径0.02m。

## 3. 控制与执行器
原AMP+CTS、公共1020输入、12动作、50Hz策略、400Hz物理、RS01延迟/响应、Kp40/Kd1、14Nm电机限幅保留。原始PD峰值17Nm为拒绝门槛，6Nm为持续参考，不混淆原始请求和电机输出。

## 4. 基线
运行目录Nov15_22-33-55_omni_direction_ampfix_30000中的model_100为本阶段预检起点；与所有后续检查点同协议筛查。不依据训练奖励选择最后一轮。

## 5. 问题—证据—选择
一些后期模型行走方向改善，但零指令后的速度、踏步或扭矩退化。选择先排除原始扭矩、跌倒、飞行及停止安全失败，再比较对角步态、路径门槛和跟踪误差。停止和站立不是一个指标：安全低速停止通过也不代表四足完全静止。

## 6. 文件
select_cts_direction.py批量初筛与复测，finalize_direction_selection.py合并证据，play_best.sh调用原生play.py。完整指标coarse.json、all_final.json、selected.json；NPZ保留真实世界位置、参考、接触、命令、终止与推力。

## 7. 配置
外部指令：直行0.18/0.25、后退−0.18、侧移±0.10m/s、转向±0.40rad/s、斜走[0.20,±0.08,0]、弧线[0.22,0,±0.30]。未加入原地踏步指令。前进速度课程仍为0.25档，未宣称0.6合格。

## 8. 验证
初筛10秒运动＋3秒停止，1种子，每模型11动作×4个名义副本；torch.vmap和原网络逐模型输出最大差异小于1e-5。
原生推理复测100：60秒运动＋5秒停止、种子1/2/3、每动作1名义＋3随机化环境。200在60秒的种子2中无扰动左前斜走非法支撑比例0.002105，超过原门槛0.001，明确淘汰后不再浪费第三种子的评估。后期候选30秒×3种子并行复测，完整5–10N推力，保留摩擦/质量/COM/RS01/传感扰动。
名义定位为诊断真值，扰动定位是3mm、0.2度、0.005m/s噪声和40ms延迟的模拟全局校正定位，不是实机里程计证据。步态/方向统计从第3秒开始，重置及原始扭矩检查包含启动与停止。

## 9. 演示
play_best.sh默认0.25m/s直接前进；--omni按原生sweep展示全向。相机保留鼠标拖动；不自动启动额外长训练。新阶段原训练继续运行。

## 10. 结果
| 轮数 | 单种子运动秒数 | 综合通过组 | 步态失败组 | 停止安全失败组 | 平均偏线RMS cm | 平均周期航向绝对误差 度 | 原始扭矩峰值 Nm |
|---:|---:|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(rows)+f'''

选中第{best['iteration']}轮，扫描{len(coarse)}个检查点。qualified={selected['qualified']}，fully_static_stopping_qualified={selected['fully_static_stopping_qualified']}。这是当前证据下的演示候选，并非对全部全向指标的无条件认证。

## 11. 限制
完整全向门槛尚未全部通过，具体失败见selected.json逐动作/逐种子记录。早期/后期复测时长不同，后期短测门槛较容易但仍存在停步退化；没有用这些短测证明长期稳定。仿真没有证明真实里程计、MuJoCo、硬件落地或热能力。全部模型仍有一定零指令踏步/静止不足，默认演示从直行开始，不隐藏这些限制。

## 12. 回滚
所有源模型保持不变。选中检查点仅复制到本目录，sha256={selected['sha256']}。可使用原训练run/旧play恢复其他检查点；本轮没有终止训练进程。
'''
(OUT/'REPORT.md').write_text(report)
script=f'''#!/usr/bin/env bash
set -euo pipefail
cd /home/nszb/gym/unitree_rl_gym
source /home/nszb/gym/unitree-rl/bin/activate
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
motion=(--vx 0.25 --vy 0 --wz 0)
if [[ "${{1:-}}" == "--omni" ]]; then motion=(--sweep); shift; fi
exec python -u legged_gym/scripts/play.py --task urdf4_amp_cts_direction_v1 \\
  --load_run {destination} --checkpoint {best['iteration']} \\
  --num_envs 1 --duration_s 120 "${{motion[@]}}" \\
  --sim_device cuda:1 --rl_device cuda:1 "$@"
'''
(OUT/'play_best.sh').write_text(script);(OUT/'play_best.sh').chmod(0o755)
walking=next(x for x in records if x['iteration']==3600)
walking_dir=OUT/'walking_best';walking_dir.mkdir(exist_ok=True)
walk_source=Path(walking['checkpoint']);shutil.copy2(str(walk_source),str(walking_dir/walk_source.name))
walk_export=walk_source.parent/'exported/student_3600.pt'
shutil.copy2(str(walk_export),str(walking_dir/'student.pt'))
shutil.copy2(str(walk_export)+'.json',str(walking_dir/'student.pt.json'))
(walking_dir/'status.json').write_text(json.dumps(dict(record=walking,
    scope='walking-only preview; failed zero-command stopping, not a qualified complete policy'),indent=2)+'\n')
walk_script=script.replace(str(destination),str(walking_dir)).replace('--checkpoint %d'%best['iteration'],'--checkpoint 3600').replace('--duration_s 120','--duration_s 110')
(OUT/'play_walking_best.sh').write_text(walk_script);(OUT/'play_walking_best.sh').chmod(0o755)
print('FINALIZED',best['iteration'],summary)
