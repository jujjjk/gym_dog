from pathlib import Path
import json,numpy as np
from batch_v2 import evaluate,MOVES
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'artifacts/urdf4_amp_design_20261113_v2';out=BASE/'yaw_placement';out.mkdir(exist_ok=False);allselected=[]
for name in ['left','forward_left']:
 prior=json.loads((BASE/'batch_screen'/(name+'_ranking.json')).read_text());records=[];i=0
 for base in prior[:2]:
  for gain in [.5,1.,1.5,2.]:
   for ki in [0.,.3,.6]:
    b=dict(base);b['design']=dict(base['design']);b['design'].update(yaw_step_feedback=gain,yaw_step_integral=ki);r=evaluate(out/(name+'_'+str(i)),MOVES[name],b,14);r['motion']=name;r['index']=i;records.append(r);i+=1
    (out/(name+'_ranking.json')).write_text(json.dumps(sorted(records,key=lambda r:r['score']),indent=2)+'\n')
 best=min(records,key=lambda r:r['score']);allselected.append(best);print(name,best['rejection_reasons'],np.round(best['mean_velocity'],3),np.round(best['clearance_p95_mm'],1),flush=True)
(out/'selected.json').write_text(json.dumps(allselected,indent=2)+'\n')
