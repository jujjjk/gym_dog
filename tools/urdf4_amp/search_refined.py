from pathlib import Path
import json,numpy as np
from physical import run
root=Path(__file__).resolve().parents[2];folder=root/'artifacts/urdf4_amp_design_20261113/kinematic_v1';out=root/'artifacts/urdf4_amp_design_20261113/search_v2';out.mkdir(exist_ok=True)
rng=np.random.default_rng(20261114);records=[]
for i in range(160):
 T=float(rng.choice([.5,.6,.7]));duty=float(rng.choice([.72,.74,.76]));front=float(rng.choice([.012,.016,.020]));rear=float(rng.choice([.024,.030,.036]));lead=float(rng.choice([.04,.06,.08]));feed=float(rng.choice([1.,1.2,1.4]));h=float(rng.choice([.25,.26,.27]));offset=float(rng.choice([-.03,-.02,-.01,0.]));params=dict(period_s=T,stance_ratio=duty,clearance_m=.018,front_clearance_m=front,rear_clearance_m=rear,foot_x_offset_m=offset)
 r=run(folder,(.10,0,0),8,lead,feed,h,overrides=params);c=np.array(r['clearance_p95_mm']);vel=np.array(r['mean_velocity']);rp=np.array(r['max_roll_pitch_deg']);r['design']=params;r['index']=i
 r['score']=(1000 if r['stop']!='completed' else 0)+np.mean(np.maximum(12-c,0)**2)*.3+np.mean(np.maximum(c-24,0)**2)*.1+1000*(r['flight_fraction']+r['single_fraction']+r['illegal_two_fraction'])+100*(vel[0]-.1)**2+100*vel[1]**2+10*vel[2]**2+float(np.sum(rp**2))*.05+max(r['raw_peak_Nm']-17,0)**2
 records.append(r);print(i,round(r['score'],3),np.round(c,1),round(vel[0],3),flush=True);(out/'ranking.json').write_text(json.dumps(sorted(records,key=lambda x:x['score']),indent=2)+'\n')
