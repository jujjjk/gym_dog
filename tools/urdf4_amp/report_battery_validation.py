"""Summarize paired physical-load tests without relabeling legacy AMP data."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/urdf4_battery_validation_20261116'
old=json.loads((OUT/'old_mass/result.json').read_text());new=json.loads((OUT/'new_mass/result.json').read_text());ref=json.loads((OUT/'reference_replay/result.json').read_text())
assert old['policy_sha256']==new['policy_sha256']
for result in [old,new]:
    assert result['frozen_weights_unchanged'] and result['reference_manifest_unchanged']
    assert [x['seed'] for x in result['reports']]==[1,2,3]
    assert all(x['seconds']==60 and x['stop_s']==5 for x in result['reports'])
delta=np.array(new['compiled_actor_masses_kg'])-old['compiled_actor_masses_kg']
assert np.max(abs(delta-.8))<2.e-6
manifest_path=ROOT/'artifacts/urdf4_amp_design_20261113_v2/amp_reference_v2/manifest.json'
manifest=json.loads(manifest_path.read_text())
assert all(hashlib.sha256((manifest_path.parent/(c['clip']+'.npz')).read_bytes()).hexdigest()==c['sha256'] for c in manifest['clips'])
legacy_receipt=json.loads((ROOT/'artifacts/urdf4_amp_isaac_validation_20261113/validation_receipt.json').read_text())
assert legacy_receipt['dataset_manifest_sha256']==hashlib.sha256(manifest_path.read_bytes()).hexdigest()

def metrics(r):
    rows=[v for rep in r['reports'] for v in rep['results']]
    return dict(nominal_gait_passed=sum(v['gait_passed'] for v in rows if v['kind']=='nominal'),
        robust_gait_passed=sum(v['gait_passed'] for v in rows if v['kind']=='robust'),
        gait_passed=sum(v['gait_passed'] for v in rows),groups=len(rows),full_passed=r['summary']['full_passed'],
        resets=sum(v['resets'] for v in rows),flight_fraction_max=max(v['flight_fraction'] for v in rows),
        raw_peak_Nm=max(v['raw_peak_Nm'] for v in rows),motor_peak_Nm=max(v['motor_peak_Nm'] for v in rows),
        applied_peak_Nm=max(v['applied_peak_Nm'] for v in rows),
        mean_group_motor_rms_Nm=float(np.mean([v['motor_rms_Nm'] for v in rows])),
        max_reported_joint_group_rms_Nm=max(max(v['per_joint_motor_rms_Nm']) for v in rows),
        mean_cross_rms_m=r['summary']['mean_cross_rms_m'],mean_heading_deg=r['summary']['mean_heading_deg'])
a,b=metrics(old),metrics(new)
failures=[dict(seed=rep['seed'],**r) for rep in new['reports'] for r in rep['results'] if not(r['gait_passed'] and r['path_passed'])]
reference_gait=all(v['fallen_count']==0 and v['flight_fraction']==0 and v['single_fraction']+v['illegal_two_fraction']<=.001
    and v['diagonal_touchdown_p95_ms'] is not None and v['diagonal_touchdown_p95_ms']<=40.001 and v['raw_peak_Nm']<=17 for v in ref['cases'])
summary=dict(old_mass=a,new_mass=b,policy_transfer_fully_passed=b['full_passed']==b['groups'],
    reference_replay_passed=ref['passed'],reference_extra_gait_gates_passed=reference_gait,
    reference_cases=len(ref['cases']),reference_raw_peak_Nm=max(v['raw_peak_Nm'] for v in ref['cases']),
    compiled_mass_delta_minmax_kg=[float(delta.min()),float(delta.max())],
    old_dataset_files_unchanged=True,policy_weights_unchanged=True,amp_training_approved=False,
    reference_state_bank_rebuilt=False,new_mass_reference_dataset_collected=False,long_training_started=False,
    verdict='Original reference commands remain nominally executable. Frozen model_100 is not fully adapted: reverse gait degrades under the extra 0.8 kg. New-load policy adaptation and freshly captured reference states are required.',failures=failures)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
rows=[]
for tag,r in [('11.57kg',old),('12.37kg',new)]:
    for rep in r['reports']:
        for x in rep['results']:
            rows.append(dict(mass=tag,seed=rep['seed'],motion=x['name'],kind=x['kind'],gait_passed=x['gait_passed'],path_passed=x['path_passed'],
                cross_max_m=x['cross_max_m'],cross_rms_m=x['cross_rms_m'],heading_mean_deg=x['heading_cycle_mean_abs_deg'],
                vx_m_s=x['actual_body_velocity'][0],vy_m_s=x['actual_body_velocity'][1],wz_rad_s=x['actual_body_velocity'][2],
                raw_peak_Nm=x['raw_peak_Nm'],motor_rms_Nm=x['motor_rms_Nm'],touchdown_p95_s=x['diagonal_touchdown_p95_s'],
                reset_count=x['resets'],rejection_reasons=';'.join(x['rejection_reasons'])))
with (OUT/'comparison.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

table=[]
for title,key,fmt in [('名义步态通过组 /33','nominal_gait_passed','d'),('扰动步态通过组 /33','robust_gait_passed','d'),
    ('方向＋步态＋停止安全综合通过组 /66','full_passed','d'),('重置/跌倒次数','resets','d'),
    ('原始PD峰值 Nm','raw_peak_Nm','.2f'),('电机RMS群组均值 Nm','mean_group_motor_rms_Nm','.2f'),
    ('最坏报告关节群组RMS Nm','max_reported_joint_group_rms_Nm','.2f'),
    ('平均偏线RMS cm','mean_cross_rms_m','.2f'),('平均周期航向绝对误差 度','mean_heading_deg','.2f')]:
    x=a[key]*(100 if key=='mean_cross_rms_m' else 1);y=b[key]*(100 if key=='mean_cross_rms_m' else 1)
    table.append('| %s | %s | %s |'%(title,format(x,fmt),format(y,fmt)))
report='''# URDF4 电池加重后的适配验证（2026-11-16）

结论：新模型质量正确，参考动作设计可以保留；旧第100轮策略没有完整适配新增0.8kg，后退出现后腿低抬脚/对角交替失败。不能把此次检查当作新重量的完整策略或真机验收。

## 1. 仓库状态
分支fix/rs01-heading-odom-soft-inhibit、HEAD 8a1e9ba，保留现有修改，未提交推送。原30000轮训练已结束，本次没有启动新长训或进行PPO更新。

## 2. URDF物理模型
trunk由5.1784426438803变为5.9784426438803kg，总质量11.569864831733414→12.369864831733413kg。旧URDF有独立字节备份，几何/碰撞/关节参数相同，12DOF、大腿/小腿约0.180/0.186m。
本次0.8kg按集中质量放在原trunk重心处；trunk惯量暂不变，电池实际位置/尺寸尚未测量。Gym和MuJoCo导入质量检查见 ../urdf4_battery_mass_20261116/。

## 3. 控制/执行器
50Hz策略、400Hz物理、5ms响应更新，RS01名义40ms延迟、31.9ms时间常数、0.928增益，Kp40/Kd1、动作缩放及关节目标速率/加速度限制不变。电机限幅14Nm；原始PD17Nm为峰值检查，6Nm仅作持续参考。

## 4. 对照基线
同一已选model_100学生网络，1020公共输入、12动作。新旧重量各11动作、每动作1名义＋3随机化环境、种子1/2/3；每次60秒运动＋5秒停止，先零动作落地2秒。群组统计为33名义＋33扰动群组，扰动群组每组包含3副本。
新旧逐环境加载质量差为0.8kg（浮点误差小于2e-6kg），其余随机化范围相同。旧重量也未全部合格，不能把新测试失败都归因于电池。

## 5. 问题—证据—判断
| 发现 | 证据 | 判断 |
|---|---|---|
| 后退发生新增退化 | 新重量三个种子的名义与扰动后退均未通过步态；部分后腿P95抬脚约0–4mm，对角交替不足 | 旧策略需要新负载适配 |
| 扭矩未超峰值，但负载增加 | 峰值仍低于17Nm，电机RMS增加 | 不能仅凭不摔或扭矩未超限判通过 |
| 参考动作仍可执行 | 原始命令在新质量下12类×4副本、60秒自由基座回放通过 | 可保留动作设计，重新采集新重量物理状态 |

## 6. 文件与边界
新增 validate_battery_transfer.py、report_battery_validation.py。validate_isaac.py新增显式--diagnostic-urdf，仅允许参考命令物理回放，输出实际URDF哈希和旧数据哈希，不修改数据、不放行训练。
冻结策略测试不创建AMP学习器，完全禁用旧参考状态库。旧manifest仅用于命令集合、步态标签和几何诊断，参考状态不会覆盖策略轨迹。原有AMP训练/导出哈希检查保持严格。

## 7. 配置
前进0.18/0.25、后退−0.18、侧移±0.10m/s、转向±0.40rad/s，斜走[0.20,±0.08,0]，弧线[0.22,0,±0.30]。名义环境无物理/传感扰动；其余保持原摩擦、质量、COM、执行器、传感参数与完整5–10N推力。
名义外环用仿真真值定位，扰动外环使用模拟全局校正定位（3mm位置、0.2度yaw、0.005m/s速度噪声、40ms延迟）。没有证明真实里程计的精度。

## 8. 验证命令
```bash
source unitree-rl/bin/activate
python tools/urdf4_amp/validate_battery_transfer.py --urdf unitree_rl_gym/resources/robots/urdf4/urdf/urdf4_rs01.urdf --output /tmp/new_mass_transfer --seconds 60 --seeds 1,2,3 --sim_device cuda:1 --rl_device cuda:1
python tools/urdf4_amp/validate_isaac.py --mode replay --dataset artifacts/urdf4_amp_design_20261113_v2/amp_reference_v2 --commands-dir artifacts/urdf4_amp_isaac_validation_20261113/commands60 --diagnostic-urdf unitree_rl_gym/resources/robots/urdf4/urdf/urdf4_rs01.urdf --output /tmp/new_mass_reference --duration_s 60 --eval_envs 4 --seed 1 --sim_device cuda:0 --rl_device cuda:0
```
策略和参考数据文件哈希均未改变。参考回放只在初始时写入姿态，动作过程中自由基座动力学推进，没有强行写入轨迹，也不自动重置。参考回放的判别器只做一次数值更新，不是PPO或新训练验证。

## 9. 下一阶段
以修正后的12.37kg为统一物理模型，旧100轮仅作为参数起点；需要补新质量下的策略适配，后退/站稳先纳入回归，再执行已讨论的更大步幅、更高速度训练。V2原文件保留，新的参考物理状态需重采集；本报告不创建训练批准回执或参考状态库。

## 10. 结果
接触/方向从运动第3秒开始统计，原始PD峰值400Hz覆盖运动和停止；附加电机/施加峰值及raw-over17计数也包含前置2秒落地。raw为PD请求，motor为限幅后电机力矩，applied另计摩擦，三者不混用。

| 指标 | 11.57kg | 12.37kg |
|---|---:|---:|
'''+ '\n'.join(table)+f'''

原AMP参考回放：{len(ref['cases'])}/12类通过原回放门槛，附加对角落地≤40ms、无非法支撑门槛通过={reference_gait}；原始峰值{summary['reference_raw_peak_Nm']:.2f}Nm。参考测试为名义、单种子、每类4副本，不代表全扰动参考验证。
完整逐动作/逐种子数据见comparison.csv、summary.json和各目录NPZ。新质量综合通过{b['full_passed']}/66，policy_transfer_fully_passed=False。两种重量均未达到完全静止站立标准，低速度停止安全和四足完全静止分别报告。

## 11. 限制
此次是零样本策略负载迁移和参考命令可执行性验证，不是完成新质量AMP数据集/参考状态库。原数据中的物理状态、力矩仍来自旧重量，不允许只替换URDF哈希冒充新数据。未完成新质量下完整MuJoCo策略Sim2Sim、长期热验证或真机测试；MuJoCo本轮仅验证质量/场景导入。

## 12. 保留/回滚
旧URDF、原V2全部NPZ/manifest、原验证回执、检查点均未覆盖。新报告另目录保存；模型权重冻结，未部署/推送，也未启动新长训。
'''
(OUT/'REPORT.md').write_text(report)
(OUT/'source_sha256.json').write_text(json.dumps({p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['tools/urdf4_amp/validate_battery_transfer.py','tools/urdf4_amp/report_battery_validation.py','tools/urdf4_amp/validate_isaac.py','unitree_rl_gym/resources/robots/urdf4/urdf/urdf4_rs01.urdf']},indent=2)+'\n')
print('BATTERY VALIDATION',a,b,'reference',ref['passed'],reference_gait)
