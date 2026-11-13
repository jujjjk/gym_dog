from pathlib import Path
import json,numpy as np
from batch_v2 import evaluate
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'artifacts/urdf4_amp_design_20261113_v2';out=BASE/'forward_yaw_trim';out.mkdir(exist_ok=False);prior=json.loads((BASE/'refine_fast/ranking.json').read_text());records=[];i=0
for ix in [50,153,221]:
 b=next(x for x in prior if x['index']==ix)
 for yg in [2.,3.,4.]:
  for vg in [2.,3.5]:
   c=dict(b);c['design']=dict(b['design']);c['design'].update(yaw_velocity_gain=yg,velocity_gain=vg);r=evaluate(out/str(i),(.28,0,0),c,14);r['index']=i;r['base_index']=ix;r['motion']='forward';records.append(r);i+=1;print(i,r['rejection_reasons'],np.round(r['mean_velocity'],3),np.round(r['clearance_p95_mm'],1),flush=True);(out/'ranking.json').write_text(json.dumps(sorted(records,key=lambda r:r['score']),indent=2)+'\n')
