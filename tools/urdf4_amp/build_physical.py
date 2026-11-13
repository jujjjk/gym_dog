"""Export physically recorded candidates with per-clip, explicit quality gates."""
from pathlib import Path
import json,hashlib,numpy as np
from physical import run
from design import MOVES,NAMES,LEGS
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'artifacts/urdf4_amp_design_20261113';out=BASE/'physical_candidates_v1';out.mkdir(exist_ok=False)
candidates=json.loads((BASE/'search_v2/ranking.json').read_text())[:6]
def event_count(mask,n=2):
 padded=np.r_[False,mask,False];starts=np.flatnonzero(~padded[:-1]&padded[1:]);ends=np.flatnonzero(padded[:-1]&~padded[1:]);return starts[ends-starts>=n]
records=[]
for name,cmd in MOVES.items():
 attempts=[]
 for c in candidates:
  folder=out/(name+'_candidate'+str(c['index']));r=run(BASE/'kinematic_v1',cmd,16,c['lead'],c['feed'],c['height'],folder,c['design'])
  a=np.load(folder/'physical.npz');ct=a['contact'];height=a['foot_clearance'];lift=(height>.008)&~ct;events=[event_count(lift[:,i],2) for i in range(4)];pairs=[event_count((ct==np.array(x)).all(1),2) for x in [[1,0,0,1],[0,1,1,0]]];seq=sorted([(t,0) for t in pairs[0]]+[(t,1) for t in pairs[1]]);alt=sum(seq[i][1]!=seq[i-1][1] for i in range(1,len(seq)));r['lift_events_per_foot']=[len(x) for x in events];r['diagonal_alternations']=int(alt);r['actual_stance_fraction']=ct.mean(0).tolist();r['candidate_index']=c['index'];r['design']=c['design'];r['motion']=name
  reasons=[];v=np.array(r['mean_velocity']);command=np.array(cmd)
  if r['stop']!='completed':reasons.append('fall_or_incomplete')
  if r['flight_fraction']>0:reasons.append('flight')
  if r['single_fraction']+r['illegal_two_fraction']>.01:reasons.append('illegal_support')
  if r['raw_peak_Nm']>17:reasons.append('raw_torque_over_17')
  if max(r['max_roll_pitch_deg'])>5:reasons.append('body_tilt_over_5deg')
  if np.linalg.norm(v[:2]-command[:2])>.03:reasons.append('linear_tracking_error')
  if abs(v[2]-command[2])>.10:reasons.append('yaw_tracking_error')
  if name!='stand':
   if min(r['clearance_p95_mm'])<10:reasons.append('insufficient_clearance')
   if max(r['clearance_p95_mm'])>35:reasons.append('excessive_clearance')
   if min(r['lift_events_per_foot'])<10:reasons.append('insufficient_repeated_lift')
   if alt<15:reasons.append('insufficient_diagonal_alternation')
  r['rejection_reasons']=reasons;r['screen_passed']=not reasons;r['source_folder']=str(folder);r['amp_training_approved']=False
  r['selection_cost']=len(reasons)*100+np.linalg.norm(v[:2]-command[:2])*10+abs(v[2]-command[2])*2+float(np.mean((np.array(r['clearance_p95_mm'])-18)**2))*.001
  (folder/'report.json').write_text(json.dumps(r,indent=2)+'\n');attempts.append(r)
 best=min(attempts,key=lambda r:r['selection_cost']);records.append(best);print(name,best['candidate_index'],best['rejection_reasons'],np.round(best['clearance_p95_mm'],1),np.round(best['mean_velocity'],3),best['lift_events_per_foot'],best['diagonal_alternations'],flush=True)
 (out/'selected_candidates.json').write_text(json.dumps(records,indent=2)+'\n')
