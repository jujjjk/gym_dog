from pathlib import Path
import json,numpy as np
from batch_v2 import evaluate,MOVES
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'artifacts/urdf4_amp_design_20261113_v2';out=BASE/'action_refinement';out.mkdir(exist_ok=False);initial=json.loads((BASE/'batch_screen/selected.json').read_text());rng=np.random.default_rng(20261118);final=[]
for motion in initial:
 name=motion['motion']
 if not motion['rejection_reasons']:final.append(motion);continue
 prior=json.loads((BASE/'batch_screen'/(name+'_ranking.json')).read_text());records=[]
 for i in range(72):
  base=prior[i%3];b=dict(base);s=dict(base['design']);actual=np.array(base['clearance_p95_mm']);old=np.array(s['leg_clearance_m']);s['leg_clearance_m']=np.clip(old+(32-actual)*.0006+rng.normal(0,.0018,4),.018,.060).tolist();s['leg_lead_offsets_s']=(np.array(s.get('leg_lead_offsets_s',[0]*4))+rng.choice([-.005,0.,.005],4)).tolist();s['velocity_gain']=float(rng.choice([2,3.5,5.]));s['yaw_velocity_gain']=float(rng.choice([1.5,2.,3.]));s['height_kp']=float(rng.choice([80,100,120]));s['height_kd']=float(rng.choice([15,20,25]));b['design']=s;b['lead']+=float(rng.choice([-.005,0.,.005]));b['feed']+=float(rng.choice([-.05,0.,.05]));b['height']+=float(rng.choice([-.003,0.,.003]));b['index']=i;r=evaluate(out/(name+'_'+str(i)),MOVES[name],b,12);r['index']=i;r['motion']=name;records.append(r)
  (out/(name+'_ranking.json')).write_text(json.dumps(sorted(records,key=lambda x:x['score']),indent=2)+'\n')
 best=min(records,key=lambda x:x['score']);final.append(best);(out/'selected.json').write_text(json.dumps(final,indent=2)+'\n');print(name,best['rejection_reasons'],np.round(best['mean_velocity'],3),np.round(best['clearance_p95_mm'],1),flush=True)
(out/'selected.json').write_text(json.dumps(final,indent=2)+'\n')
