from pathlib import Path
import json,numpy as np
from batch_v2 import evaluate
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'artifacts/urdf4_amp_design_20261113_v2';out=BASE/'forward_finalists60';out.mkdir(exist_ok=False);prior=json.loads((BASE/'refine_fast/ranking.json').read_text());vertical=json.loads((BASE/'vertical_refinement/ranking.json').read_text());choices=[(str(i),next(x for x in prior if x['index']==i)) for i in [50,221,153]]+ [('vertical'+str(x['index']),x) for x in vertical[:2]];records=[]
for name,b in choices:
 r=evaluate(out/name,(.28,0,0),b,60);r['candidate_name']=name;r['motion']='forward';records.append(r);print(name,r['rejection_reasons'],np.round(r['mean_velocity'],3),np.round(r['clearance_p95_mm'],1),r['coordination']['body_z_std_mm'],flush=True);(out/'ranking.json').write_text(json.dumps(sorted(records,key=lambda r:r['score']),indent=2)+'\n')
