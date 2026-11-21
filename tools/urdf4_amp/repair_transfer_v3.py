"""Bounded transfer calibration using measured Gym foot events, then fresh physics.

Only reference trajectory parameters change. All hardware/actuator limits stay
fixed. Every candidate is regenerated on the new-mass MuJoCo plant first.
"""
import copy
from concurrent.futures import ProcessPoolExecutor
import json
import numpy as np
from complete_v3 import BASE, evaluate
from review_gym_v3 import starts


def main():
    rng=np.random.default_rng(6621);out=BASE/'transfer_candidates';out.mkdir(exist_ok=True)
    passed=set();poolrows={}
    for run,dataset in [('gym_closed01','gym_batch01'),('gym_closed02','gym_batch02')]:
        receipt=json.loads((BASE/run/'v3_windowed_result.json').read_text())
        manifest=json.loads((BASE/dataset/'manifest.json').read_text())
        z=np.load(BASE/run/'live_rollout.npz');mask=(z['time_s']>=5)&(z['time_s']<55)
        for i,(row,spec) in enumerate(zip(receipt['cases'],manifest['clips'])):
            base=row['clip'].split('__')[0]
            if row['passed']:passed.add(base)
            poolrows.setdefault(base,[]).append((row,spec,z['time_s'][mask],z['contact'][mask,i]))
    jobs=[]
    for base,rows in poolrows.items():
        if base in passed:continue
        def score(item):
            r=item[0]
            return len(r['rejection_reasons'])*100+max(0,20-min(r['foot_clearance_p95_mm']))+max(0,r['heading_mean_abs_deg']-2)*10+max(0,r['cross_max_m']-.05)*100
        rows.sort(key=score)
        for i in range(12):
            measured,spec,t,ct=rows[i%min(2,len(rows))];b=copy.deepcopy(spec);d=b['design'];T=d['period_s']
            lead=np.asarray(d.get('leg_lead_offsets_s',[0.]*4));duties=np.asarray(d.get('leg_stance_ratio',[d['stance_ratio']]*4));clear=np.asarray(d['leg_clearance_m'])
            for aa,bb in [(0,3),(1,2)]:
                deltas=[]
                for active in [ct,~ct]:
                    phases=[]
                    for leg in [aa,bb]:
                        ix=starts(active[:,leg]);phase=np.exp(2j*np.pi*(t[ix]/T-np.array([0,.5,.5,0])[leg]));phases.append(np.angle(phase.mean())/(2*np.pi) if len(ix) else None)
                    if None not in phases:deltas.append((phases[1]-phases[0]+.5)%1-.5)
                shift=np.clip(np.mean(deltas)*T*.3,-.008,.008) if deltas else 0.
                lead[aa]-=shift;lead[bb]+=shift
                actual=ct[:,[aa,bb]].mean(0);duties[[aa,bb]]-=np.clip((actual-actual.mean())*.3,-.015,.015)
            clear+=np.clip((26-np.asarray(measured['foot_clearance_p95_mm']))*.00055,0,.007)
            d['leg_clearance_m']=np.clip(clear+rng.normal(0,.001,4),.018,.08).tolist()
            d['leg_lead_offsets_s']=np.clip(lead+rng.normal(0,.0025,4),-.065,.065).tolist()
            d['leg_stance_ratio']=np.clip(duties+rng.normal(0,.004,4),.51,.77).tolist()
            d['body_height_m']+=float(rng.normal(0,.001));b['feed']+=float(rng.normal(0,.015))
            if 'direction' in measured['rejection_reasons'] or abs(b['command'][2])>.5:
                d.update(heading_integral_gain=float(rng.choice([.1,.2,.4,.6])),path_gain=max(1.5,d.get('path_gain',1.)))
            if 'tracking' in measured['rejection_reasons'] or 'velocity_tracking' in measured['rejection_reasons']:
                d.update(velocity_step_gain=max(.3,d.get('velocity_step_gain',0.)),velocity_step_integral=.1)
            b['clip']=base+'__g%02d'%i;jobs.append((b,60,str(out/b['clip'])))
    with ProcessPoolExecutor(8) as pool:
        for r in pool.map(evaluate,jobs):
            (out/(r['clip']+'.json')).write_text(json.dumps(r,indent=2)+'\n');print(r['clip'],r['passed'],r['rejection_reasons'],flush=True)


if __name__=='__main__':main()
