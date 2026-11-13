from pathlib import Path
import json,csv,math
p=Path(__file__).resolve().parent
labels={'forward 0.1':'前进0.1','stand':'站立','forward 0.3':'前进0.3','forward 0.6':'前进0.6','backward 0.4':'后退0.4','left 0.3':'左移0.3','right 0.3':'右移0.3','turn left 1.0':'左转1.0rad/s','turn right 1.0':'右转1.0rad/s','diagonal':'斜走(0.4,0.2,0)','arc left':'弧线(0.3,0,0.6)','mixed':'混合(0.3,-0.2,-0.5)'}
report=['# model_5400 全向动作扭矩统计','', '使用当前RS01执行器，50Hz策略、400Hz物理步；每动作独立60s，稳定统计窗口2–60s。名义模式1环境/动作、seed1；扰动模式4环境/动作、seed1。未包含每10s方向切换的连续播放实验。', '', 'raw_pd=未经限幅的PD请求；motor=±14N·m限幅后的电磁力矩；applied=电机强度随机化和库仑摩擦后的最终施加力矩。6N·m为连续参考，不是本实验的硬限幅。']
summary=[];jointrows=[]
for mode in ['nominal','disturbed']:
 r=json.loads((p/(mode+'.json')).read_text());report+=['','## '+('无扰动' if mode=='nominal' else '随机化、传感器误差及推力扰动'),'', '|动作|raw峰值|motor峰值|applied峰值|motor RMS|motor P95|motor>6比例|最重关节RMS|重置|','|---|---:|---:|---:|---:|---:|---:|---|---:|']
 for c in r['cases']:
  ch=c['channels'];raw=ch['raw_pd']['steady'];motor=ch['motor']['steady'];applied=ch['applied']['steady'];row={'mode':mode,'action':c['label'],'vx':c['command_vx_vy_wz'][0],'vy':c['command_vx_vy_wz'][1],'wz':c['command_vx_vy_wz'][2],'resets':c['resets'],'worst_motor_rms_joint':c['highest_motor_rms_joint'],'worst_motor_rms_Nm':c['highest_motor_rms_Nm']}
  for chan,val in ch.items():
   row.update({chan+'_'+k:v for k,v in val['steady'].items()});row[chan+'_whole_peak_Nm']=val['whole_run_peak_Nm']
  summary.append(row)
  report.append(f"|{labels[c['label']]}|{raw['peak_abs_Nm']:.2f}|{motor['peak_abs_Nm']:.2f}|{applied['peak_abs_Nm']:.2f}|{motor['rms_Nm']:.2f}|{motor['p95_abs_Nm']:.2f}|{motor['over_6_pct']:.2f}%|{c['highest_motor_rms_joint']} {c['highest_motor_rms_Nm']:.2f}|{c['resets']}|")
  for j in c['joints']:
   for name,values in j['channels'].items():jointrows.append({'mode':mode,'action':c['label'],'joint':j['name'],'channel':name,**values,'motor_longest_over6_s':j['motor_longest_over6_s'],'motor_longest_over12_s':j['motor_longest_over12_s']})
report+=['','所有力矩单位为N·m。峰值=稳定窗口全部物理步、全部电机、全部该动作环境的最大绝对值；RMS与P95由同样样本合并计算。>6比例分母为时间×环境×12个电机，不是每台电机各自的占比；请结合最重关节RMS与逐关节CSV判断。', '', 'whole_run_peak字段覆盖0–60s并包含启动；逐关节CSV提供P95/P99、>6/12/14/17比例和单电机连续超过6/12N·m的最长时长。扰动测试不同环境的motor_strength不同，applied峰值可能超过14N·m。', '', '本次无完整温度、热降额或速度力矩包络模型，RMS用于负载比较，不构成真实持续能力/温升验证。自碰撞仍关闭。', '', '原始JSON包含checkpoint路径及SHA256、增益、关节顺序、随机种子和采样设置；NPZ保存逐子步力矩，可复算。训练代码、权重和URDF未修改。']
for name,rows in [('summary.csv',summary),('per_joint.csv',jointrows)]:
 with (p/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(p/'REPORT.md').write_text('\n'.join(report)+'\n')
print('\n'.join(report))
