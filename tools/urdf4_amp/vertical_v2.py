from pathlib import Path
import json,numpy as np
from physical_v2 import run
from batch_v2 import evaluate
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'artifacts/urdf4_amp_design_20261113_v2';out=BASE/'vertical_refinement';out.mkdir(exist_ok=False);prior=json.loads((BASE/'refine_fast/ranking.json').read_text());records=[];i=0
for ix in [50,221,200]:
 b=next(x for x in prior if x['index']==ix)
 for kp in [60,100,140,180]:
  for kd in [12,18,24,30]:
   c=dict(b);c['design']=dict(b['design']);c['design'].update(height_kp=kp,height_kd=kd);r=evaluate(out/str(i),(.28,0,0),c,12);r['index']=i;r['base_index']=ix;r['motion']='forward';records.append(r);i+=1;print(i,r['rejection_reasons'],'zstd',round(r['coordination']['body_z_std_mm'],2),'v',round(r['mean_velocity'][0],3),flush=True);(out/'ranking.json').write_text(json.dumps(sorted(records,key=lambda x:x['score']),indent=2)+'\n')
