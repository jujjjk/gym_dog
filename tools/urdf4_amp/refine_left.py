from pathlib import Path
import json,numpy as np
from physical import run
BASE=Path(__file__).resolve().parents[2]/'artifacts/urdf4_amp_design_20261113';prior=json.loads((BASE/'physical_candidates_v1/selected_candidates.json').read_text());base=next(x for x in prior if x['motion']=='left');rng=np.random.default_rng(20261115);records=[];out=BASE/'left_refinement';out.mkdir(exist_ok=False)
for i in range(72):
 params=dict(base['design']);params.update(rear_clearance_m=float(rng.choice([.026,.030,.034,.038])),front_clearance_m=float(rng.choice([.012,.016,.020])),foot_x_offset_m=float(rng.choice([-.01,0.,.01])))
 feed=float(rng.choice([1.25,1.4,1.55]));lead=float(rng.choice([.04,.06,.08]));h=float(rng.choice([.245,.25,.255]));r=run(BASE/'kinematic_v1',(0,.05,0),10,lead,feed,h,save=out/str(i),overrides=params);c=np.array(r['clearance_p95_mm']);v=np.array(r['mean_velocity']);r['design']=params;r['index']=i;r['source_folder']=str(out/str(i));r['score']=np.mean(np.maximum(12-c,0)**2)+np.mean(np.maximum(c-28,0)**2)*.2+1000*(r['flight_fraction']+r['single_fraction']+r['illegal_two_fraction'])+100*np.linalg.norm(v-np.array([0,.05,0]))**2+max(r['raw_peak_Nm']-17,0)**2
 if r['stop']!='completed':r['score']+=1000
 records.append(r);print(i,round(r['score'],2),np.round(c,1),flush=True);(out/'ranking.json').write_text(json.dumps(sorted(records,key=lambda r:r['score']),indent=2)+'\n')
