"""Balance individual feet while retaining physically measured diagonal timing."""
import argparse,json
from pathlib import Path
import numpy as np
from physical_v2 import run
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'artifacts/urdf4_amp_design_20261113_v2'
p=argparse.ArgumentParser();p.add_argument('--count',type=int,default=240);a=p.parse_args();out=BASE/'refine_fast';out.mkdir(exist_ok=False)
prior=json.loads((BASE/'search_fast/ranking.json').read_text());pool=[x for x in prior if x['stop']=='completed' and .24<x['mean_velocity'][0]<.315 and x['flight_fraction']==0 and x['single_fraction']+x['illegal_two_fraction']<.001 and x['raw_peak_Nm']<17][:12]
rng=np.random.default_rng(20261117);records=[]
for i in range(a.count):
 b=pool[i%len(pool)];s=dict(b['design']);nominal=np.array([s['front_clearance_m']]*2+[s['rear_clearance_m']]*2);corrected=nominal+np.clip((32-np.array(b['clearance_p95_mm']))*.0007,-.012,.012);s['leg_clearance_m']=np.clip(corrected+rng.normal(0,.003,4),.018,.060).tolist();s['leg_lead_offsets_s']=rng.choice([-.01,0.,.01],4).tolist();s['foot_x_offset_m']+=float(rng.choice([-.005,0.,.005]));lead=b['lead']+float(rng.choice([-.01,0.,.01]));feed=b['feed']+float(rng.choice([-.1,0.,.1]));h=b['height']+float(rng.choice([-.005,0.,.005]));cmdspeed=float(rng.choice([.26,.28,.30]));r=run(ROOT/'artifacts/urdf4_amp_design_20261113/kinematic_v1',(cmdspeed,0.,0.),10,lead,feed,h,overrides=s);c=np.array(r['clearance_p95_mm']);v=np.array(r['mean_velocity']);rp=np.array(r['max_roll_pitch_deg']);co=r['coordination'];r['design']=s;r['index']=i;r['source_index']=b['index'];
 r['score']=(1000 if r['stop']!='completed' else 0)+np.mean(np.maximum(27-c,0)**2)*.6+np.mean(np.maximum(c-40,0)**2)*.3+1000*(r['flight_fraction']+r['single_fraction']+r['illegal_two_fraction'])+1e4*(max(.25-v[0],0)**2+max(v[0]-.30,0)**2)+300*(v[0]-.275)**2+300*v[1]**2+30*v[2]**2+float(np.sum(rp**2))*.06+max(r['raw_peak_Nm']-17,0)**2+((co['diagonal_touchdown_p95_ms']/40)**2+(co['diagonal_liftoff_p95_ms']/40)**2)*.5+co['body_z_std_mm']**2*.03+np.mean((c[[0,1]]-c[[3,2]])**2)*.01
 records.append(r);print(i,round(r['score'],3),np.round(c,1),np.round(v,3),'diag',co['diagonal_touchdown_p95_ms'],co['diagonal_liftoff_p95_ms'],flush=True);(out/'ranking.json').write_text(json.dumps(sorted(records,key=lambda r:r['score']),indent=2)+'\n')
