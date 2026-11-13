"""Bounded parameter search, no RL, no policy, and no automatic AMP approval."""
from pathlib import Path
import json,numpy as np
from physical import run
root=Path(__file__).resolve().parents[2];folder=root/'artifacts/urdf4_amp_design_20261113/kinematic_v1';out=root/'artifacts/urdf4_amp_design_20261113/search_v1';out.mkdir(exist_ok=True)
rng=np.random.default_rng(20261113);records=[]
for i in range(80):
 T=float(rng.choice([.50,.60,.70,.80]));duty=float(rng.choice([.70,.74,.76]));clear=float(rng.choice([.018,.024,.030]));lead=float(rng.choice([.05,.08,.11,.14]));feed=float(rng.choice([.7,1.,1.2]));h=float(rng.choice([.25,.26,.27]));params=dict(period_s=T,stance_ratio=duty,clearance_m=clear)
 r=run(folder,(.10,0,0),6,lead,feed,h,overrides=params)
 c=np.array(r['clearance_p95_mm']);vel=np.array(r['mean_velocity']);rp=np.array(r['max_roll_pitch_deg'])
 r['design']=params;r['index']=i
 r['score']=(1000 if r['stop']!='completed' else 0)+np.mean(np.maximum(12-c,0)**2)*.2+1000*(r['flight_fraction']+r['single_fraction']+r['illegal_two_fraction'])+100*(vel[0]-.1)**2+100*vel[1]**2+10*vel[2]**2+float(np.sum(rp**2))*.05+max(r['raw_peak_Nm']-17,0)**2
 records.append(r);print(i,round(r['score'],3),np.round(c,1),round(vel[0],3),flush=True)
 (out/'ranking.json').write_text(json.dumps(sorted(records,key=lambda x:x['score']),indent=2)+'\n')
