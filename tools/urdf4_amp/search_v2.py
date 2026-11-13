"""V2 bounded search: higher actual speed/clearance and measured diagonal coordination."""
import argparse,json
from pathlib import Path
import numpy as np
from physical_v2 import run
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--speed',type=float,default=.18);p.add_argument('--count',type=int,default=200);p.add_argument('--fast',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
rng=np.random.default_rng(20261116);records=[]
for i in range(a.count):
 T=float(rng.choice(([.4,.5,.6] if a.fast else [.5,.6,.7])));duty=float(rng.choice(([.60,.64,.68,.70] if a.fast else [.68,.70,.72,.74])));front=float(rng.choice(([.030,.038,.046] if a.fast else [.026,.032,.038])));rear=float(rng.choice(([.034,.044,.054] if a.fast else [.034,.042,.050])));lead=float(rng.choice([.04,.06,.08]));feed=float(rng.choice([1.2,1.4,1.55]));height=float(rng.choice(([.25,.27,.29] if a.fast else [.25,.26,.27])));offset=float(rng.choice([-.02,-.01,0,.01]));params=dict(period_s=T,stance_ratio=duty,clearance_m=.03,front_clearance_m=front,rear_clearance_m=rear,foot_x_offset_m=offset,front_lead_offset_s=float(rng.choice([-.02,0.,.02])),rear_lead_offset_s=float(rng.choice([-.02,0.,.02])),feedforward_alpha=float(rng.choice([.35,.5,.7])))
 r=run(ROOT/'artifacts/urdf4_amp_design_20261113/kinematic_v1',(a.speed,0.,0.),8,lead,feed,height,overrides=params);c=np.array(r['clearance_p95_mm']);v=np.array(r['mean_velocity']);rp=np.array(r['max_roll_pitch_deg']);co=r['coordination'];r['design']=params;r['index']=i
 r['score']=(1000 if r['stop']!='completed' else 0)+np.mean(np.maximum(26-c,0)**2)*.4+np.mean(np.maximum(c-42,0)**2)*.25+1000*(r['flight_fraction']+r['single_fraction']+r['illegal_two_fraction'])+300*(v[0]-a.speed)**2+200*v[1]**2+20*v[2]**2+float(np.sum(rp**2))*.08+max(r['raw_peak_Nm']-17,0)**2+(co['diagonal_touchdown_p95_ms']/40)**2+(co['diagonal_liftoff_p95_ms']/40)**2+co['body_z_std_mm']**2*.05
 records.append(r);print(i,round(r['score'],3),np.round(c,1),np.round(v,3),'diag_ms',co['diagonal_touchdown_p95_ms'],co['diagonal_liftoff_p95_ms'],flush=True)
 (a.output/'ranking.json').write_text(json.dumps(sorted(records,key=lambda r:r['score']),indent=2)+'\n')
