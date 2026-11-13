"""Screen a higher-speed/higher-clearance V2 without editing V1 or motor limits."""
from pathlib import Path
import json,numpy as np
from physical_v2 import run
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'artifacts/urdf4_amp_design_20261113_v2';out=BASE/'batch_screen'
MOVES={'march':(0,0,0),'forward_slow':(.18,0,0),'forward':(.28,0,0),'backward':(-.18,0,0),'left':(0,.10,0),'right':(0,-.10,0),'turn_left':(0,0,.4),'turn_right':(0,0,-.4),'forward_left':(.20,.08,0),'forward_right':(.20,-.08,0),'arc_left':(.22,0,.3),'arc_right':(.22,0,-.3)}
bases=json.loads((BASE/'refine_fast/ranking.json').read_text())[:8]
def evaluate(folder,cmd,b,seconds=12):
 r=run(ROOT/'artifacts/urdf4_amp_design_20261113/kinematic_v1',cmd,seconds,b['lead'],b['feed'],b['height'],save=folder,overrides=b['design']);c=np.array(r['clearance_p95_mm']);v=np.array(r['mean_velocity']);co=r['coordination'];reasons=[]
 if r['stop']!='completed':reasons.append('fall')
 if r['flight_fraction']>0:reasons.append('flight')
 if r['single_fraction']+r['illegal_two_fraction']>.005:reasons.append('illegal_support')
 if r['raw_peak_Nm']>17:reasons.append('raw_over17')
 if min(c)<25 or max(c)>45:reasons.append('clearance')
 if max(abs(c[[0,1]]-c[[3,2]]))>10:reasons.append('diagonal_height_imbalance')
 if max(r['max_roll_pitch_deg'])>4.5:reasons.append('body_tilt')
 if co['diagonal_touchdown_p95_ms']>40.001 or co['diagonal_liftoff_p95_ms']>40.001:reasons.append('diagonal_timing')
 if np.linalg.norm(v[:2]-np.array(cmd[:2]))>.035 or abs(v[2]-cmd[2])>.08:reasons.append('tracking')
 if abs(cmd[2])<1e-8 and abs(v[2])>.03:reasons.append('straight_yaw_drift')
 if min(co['lift_events_per_foot'])<seconds/b['design']['period_s']*.65:reasons.append('repeated_lift')
 if abs(cmd[0]-.28)<1e-6 and cmd[1]==0 and cmd[2]==0 and not .25<=v[0]<=.30:reasons.append('forward_speed_band')
 r.update(design=b['design'],source_index=b.get('index'),rejection_reasons=reasons,physical_screen_passed=not reasons,source_folder=str(folder))
 r['score']=len(reasons)*100+np.mean((c-32)**2)*.02+np.linalg.norm(v[:2]-np.array(cmd[:2]))*10+abs(v[2]-cmd[2])*3+co['body_z_std_mm']**2*.04+(co['diagonal_touchdown_p95_ms']+co['diagonal_liftoff_p95_ms'])*.01
 (folder/'report.json').write_text(json.dumps(r,indent=2)+'\n');return r
if __name__=='__main__':
 out.mkdir(exist_ok=False)
 selected=[]
 for name,cmd in MOVES.items():
  results=[]
  for b in bases:
   folder=out/(name+'_'+str(b['index']));r=evaluate(folder,cmd,b);r['motion']=name;results.append(r)
  results.sort(key=lambda x:x['score']);best=results[0];selected.append(best);(out/(name+'_ranking.json')).write_text(json.dumps(results,indent=2)+'\n');(out/'selected.json').write_text(json.dumps(selected,indent=2)+'\n');print(name,best['source_index'],best['rejection_reasons'],np.round(best['mean_velocity'],3),np.round(best['clearance_p95_mm'],1),flush=True)
