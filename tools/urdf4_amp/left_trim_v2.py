from pathlib import Path
import json,numpy as np
from batch_v2 import evaluate,MOVES
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'artifacts/urdf4_amp_design_20261113_v2';out=BASE/'left_final_trim';out.mkdir(exist_ok=False);prior=json.loads((BASE/'yaw_placement/left_ranking.json').read_text());records=[];i=0
for base in prior[:3]:
 for up in [.002,.004,.006]:
  for rear in [0.,-.002]:
   for lead in [0.,.005]:
    b=dict(base);b['design']=dict(base['design']);clr=list(b['design']['leg_clearance_m']);clr[1]+=up;clr[2]+=rear;b['design']['leg_clearance_m']=clr;ld=list(b['design'].get('leg_lead_offsets_s',[0.,0.,0.,0.]));ld[1]+=lead;b['design']['leg_lead_offsets_s']=ld;r=evaluate(out/str(i),MOVES['left'],b,14);r['index']=i;r['motion']='left';records.append(r);i+=1;(out/'ranking.json').write_text(json.dumps(sorted(records,key=lambda r:r['score']),indent=2)+'\n')
b=sorted(records,key=lambda r:r['score'])[0];print(b['rejection_reasons'],b['mean_velocity'],b['clearance_p95_mm'],flush=True)
