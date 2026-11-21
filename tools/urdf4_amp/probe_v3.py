"""Bounded V3 speed/step feasibility on the corrected 12.37 kg plant."""
import argparse,itertools,json,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from physical_v2 import run
ROOT=Path(__file__).resolve().parents[2]


def probe(job):
    i,speed,design,lead,feed,scene,output,version=job
    controller=run
    if version==3:
        from physical_v3 import run as controller
    started=time.monotonic()
    r=controller(Path(scene),(speed,0.,0.),8,lead,feed,design['body_height_m'],overrides=design)
    r.update(index=i,design=design,wall_seconds=time.monotonic()-started)
    c=np.array(r['clearance_p95_mm']);v=np.array(r['mean_velocity']);co=r['coordination'];fail=[]
    if r['stop']!='completed':fail.append('fall')
    if r['flight_fraction']>0:fail.append('flight')
    if r['single_fraction']+r['illegal_two_fraction']>.001:fail.append('support')
    if r['raw_peak_Nm']>17:fail.append('raw_torque')
    if min(c)<15 or max(c)>55:fail.append('clearance')
    if co['diagonal_touchdown_p95_ms']>40.001 or co['diagonal_liftoff_p95_ms']>40.001:fail.append('diagonal_timing')
    if abs(v[0]-speed)>max(.03,.05*speed) or abs(v[1])>.03 or abs(v[2])>.04:fail.append('tracking')
    if min(co['lift_events_per_foot'])<r.get('steady_seconds',8)/design['period_s']*.7:fail.append('alternation')
    if r.get('cross_max_m',0)>.05 or r.get('heading_mean_abs_deg',0)>2:fail.append('direction')
    r['rejection_reasons']=fail;r['passed']=not fail
    r['score']=len(fail)*100+abs(v[0]-speed)*50+abs(v[1])*20+abs(v[2])*10+co['diagonal_touchdown_p95_ms']*.05+co['diagonal_liftoff_p95_ms']*.05+max(r['raw_peak_Nm']-14,0)
    return r


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--speeds',default='.4,.6');p.add_argument('--search_count',type=int,default=0)
    p.add_argument('--controller_version',type=int,choices=[2,3],default=2)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    scene=ROOT/'artifacts/urdf4_amp_design_20261116_v3/plant'
    config=[(.5,.60,.29,.03),(.5,.56,.29,.03),(.4,.60,.29,.03),(.4,.56,.29,.03),(.44,.56,.30,.025),(.44,.52,.30,.025)]
    jobs=[]
    for speed,(period,duty,height,clearance),feed in itertools.product(map(float,a.speeds.split(',')),config,[1.25,1.5]):
        d=dict(period_s=period,stance_ratio=duty,body_height_m=height,clearance_m=clearance,
            front_clearance_m=clearance,rear_clearance_m=clearance+.004,half_width_m=.1475,
            foot_x_offset_m=0.,front_lead_offset_s=0.,rear_lead_offset_s=0.,feedforward_alpha=.5,
            yaw_step_feedback=.5,velocity_gain=3.)
        jobs.append((len(jobs),speed,d,.07,feed,str(scene),str(a.output),a.controller_version))
    if a.search_count:
        rng=np.random.default_rng(20261116)
        manifest=json.loads((ROOT/'artifacts/urdf4_amp_design_20261113_v2/amp_reference_v2/manifest.json').read_text())
        base=next(c for c in manifest['clips'] if c['clip']=='forward')
        speeds=list(map(float,a.speeds.split(',')));jobs=[]
        for i in range(a.search_count):
            speed=speeds[i%len(speeds)];d=dict(base['design'])
            d.update(period_s=float(rng.choice([.40,.44,.48,.50,.56,.60,.70])),
                stance_ratio=float(rng.choice([.56,.60,.64,.68,.72])),body_height_m=float(rng.choice([.27,.285,.30,.315])),
                half_width_m=.1475,foot_x_offset_m=float(rng.choice([-.035,-.025,-.015,0.])),
                leg_clearance_m=rng.uniform(.03,.055,4).tolist(),leg_lead_offsets_s=rng.choice([-.03,-.02,-.01,0.,.01,.02,.03],4).tolist(),
                yaw_step_feedback=float(rng.choice([0.,.1,.3,.5])),velocity_gain=float(rng.choice([2.,3.,4.])))
            jobs.append((i,speed,d,float(rng.choice([.05,.07,.09,.11])),float(rng.choice([1.2,1.35,1.45,1.55])),str(scene),str(a.output),a.controller_version))
    results=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for future in as_completed([pool.submit(probe,j) for j in jobs]):
            r=future.result();results.append(r)
            (a.output/'ranking.json').write_text(json.dumps(sorted(results,key=lambda x:x['score']),indent=2)+'\n')
            print('V3 PROBE',r['index'],r['command'][0],r['rejection_reasons'],np.round(r['mean_velocity'],3),round(r['raw_peak_Nm'],2),round(r['wall_seconds'],1),flush=True)


if __name__=='__main__':main()
