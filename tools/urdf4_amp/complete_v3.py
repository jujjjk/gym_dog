"""Generate and screen new-mass AMP teachers; no PPO/long training.

All objective values come from free-base MuJoCo rollouts through the frozen
RS01 actuator. Search results are diagnostics until independently validated.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from physical_v3 import run

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'artifacts/urdf4_amp_design_20261116_v3'
V2 = ROOT/'artifacts/urdf4_amp_design_20261113_v2/amp_reference_v2'


def assess(r):
    """Steady-state gates, also used after the longer confirmation rollout."""
    cmd = np.asarray(r['command']); v = np.asarray(r['mean_velocity'])
    c = np.asarray(r['clearance_p95_mm']); co = r['coordination']
    moving = np.linalg.norm(cmd) > 1e-8
    residual = dict(
        fall=100. if r['stop']!='completed' else 0.,
        flight=100*r['flight_fraction'],
        support=max(0., r['single_fraction']+r['illegal_two_fraction']-.001)/.005,
        raw_torque=max(0.,r['raw_peak_Nm']-17)/2.,
        ik=max(0.,r['max_ik_error_m']-.001)/.005,
        tracking_xy=max(0.,np.linalg.norm(v[:2]-cmd[:2])-.03)/.02,
        tracking_yaw=max(0.,abs(v[2]-cmd[2])-(.04 if abs(cmd[2])<1e-8 else .06))/.04,
        direction=max(0.,r['cross_max_m']-.05)/.04,
        heading=max(0.,r['heading_mean_abs_deg']-2.)/2.,
        body_tilt=max(0.,max(r['max_roll_pitch_deg'])-5.)/3.)
    if moving:
        residual.update(
            clearance_low=max(0.,20.-min(c))/15.,
            clearance_high=max(0.,max(c)-55.)/20.,
            touchdown=max(0.,co['diagonal_touchdown_p95_ms']-40.001)/40.,
            liftoff=max(0.,co['diagonal_liftoff_p95_ms']-40.001)/40.,
            alternation=max(0.,.7*r['steady_seconds']/r['design']['period_s']-min(co['lift_events_per_foot']))/3.)
    else:
        residual['standing_feet']=max(0.,max(c)-5.)/10.
    r['rejection_reasons']=[k for k,x in residual.items() if x>1e-7]
    r['passed']=not r['rejection_reasons']
    violation=sum(x*x for x in residual.values())
    quality=.0001*r['raw_peak_Nm']+.001*np.linalg.norm(v[:2]-cmd[:2])+.0001*co['yaw_rate_rms_rad_s']
    # Never rank a narrowly failed gate above an accepted sample just because
    # the latter has a slightly higher (but executable) torque.
    r['score']=float(quality if r['passed'] else 100.+violation)
    return r


def evaluate(job):
    candidate, seconds, output = job
    d=candidate['design']
    r=run(BASE/'plant',candidate['command'],seconds,candidate['lead'],candidate['feed'],
        d['body_height_m'],save=Path(output) if output else None,overrides=d)
    r.update(design=d,clip=candidate['clip'],family=candidate['family'])
    return assess(r)


def cases():
    result=[]
    def add(family,cmd):
        label='_'.join(('%+.3f'%x).replace('+','p').replace('-','m').replace('.','') for x in cmd)
        result.append(dict(clip=family+'_'+label,family=family,command=list(cmd)))
    add('stand',(0.,0.,0.))
    for x in [.08,.15,.20,.25,.30,.35,.40,.50,.60]:add('forward',(x,0.,0.))
    for x in [-.08,-.15,-.20,-.25,-.30,-.35]:add('backward',(x,0.,0.))
    for y in [.05,.10,.15,.20,-.05,-.10,-.15,-.20]:add('left' if y>0 else 'right',(0.,y,0.))
    for w in [.2,.4,.6,.8,-.2,-.4,-.6,-.8]:add('turn_left' if w>0 else 'turn_right',(0.,0.,w))
    for x in [.15,.30,-.15,-.25]:
        for y in [.08,-.08]:add(('forward' if x>0 else 'backward')+('_left' if y>0 else '_right'),(x,y,0.))
    for x in [.2,.25,.30,.35,-.15,-.25]:
        for w in [.3,-.3]:add(('arc' if x>0 else 'back_arc')+('_left' if w>0 else '_right'),(x,0.,w))
    for y in [.1,-.1]:
        for w in [.3,-.3]:add('lateral_yaw',(0.,y,w))
    return result


def seeds(case):
    old=json.loads((V2/'manifest.json').read_text())['clips']
    preferred=case['family']
    if preferred=='forward' and case['command'][0]<=.20:preferred='forward_slow'
    if preferred.startswith('back_arc'):preferred='backward'
    if preferred.startswith('backward_'):preferred='backward'
    if preferred=='lateral_yaw':preferred='left' if case['command'][1]>0 else 'right'
    if preferred=='stand':preferred='forward_slow'
    bases=[x for x in old if x['clip']==preferred]
    if not bases:bases=[x for x in old if x['clip']=='forward']
    choices=[]
    for b in bases:
        for ramp in [2.,0.]:
            for heading,velocity in [(.5,0.),(.5,.3),(1.,.15)]:
                d=copy.deepcopy(b['design']);d.update(body_height_m=b['height'],half_width_m=.1475,
                    ramp_s=ramp,path_gain=1.,heading_gain=1.,heading_step_gain=heading,
                    velocity_step_gain=velocity,yaw_step_feedback=.2,yaw_step_integral=.1,
                    stride_scale=[1.,1.,1.],leg_stance_ratio=[d['stance_ratio']]*4)
                choices.append(dict(**case,design=d,lead=b['lead'],feed=b['feed']))
    # Warm starts from physically measured higher-speed searches.
    if preferred=='forward' and case['command'][0]>=.30:
        a=json.loads((BASE/'forward_local/ranking.json').read_text())
        for speed in [.4,.6]:
            for b in [x for x in a if x['command'][0]==speed][:2]:
                d=copy.deepcopy(b['design']);d.update(stride_scale=[1.,1.,1.],leg_stance_ratio=[d['stance_ratio']]*4)
                choices.append(dict(**case,design=d,lead=b['lead'],feed=b['feed']))
                if speed==.6 and case['command'][0]>=.4:
                    for factor in [.95,1.,1.05,1.10]:
                        trial=copy.deepcopy(d);trial['stride_scale']=[speed/case['command'][0]*factor,1.,1.]
                        trial['ramp_s']=3.
                        choices.append(dict(**case,design=trial,lead=b['lead'],feed=b['feed']))
    confirmed=BASE/'confirmation60'
    if confirmed.exists():
        for file in confirmed.glob('*.json'):
            b=json.loads(file.read_text())
            if isinstance(b,dict) and b.get('passed') and b['family']==case['family']:
                choices.append(dict(**case,design=b['design'],lead=b['lead'],feed=b['feed']))
    return choices


def mutate(base,rng,scale=.5,calibrate=False):
    b=copy.deepcopy(base);d=b['design'];co=base['coordination'];T=d['period_s']
    clear=np.asarray(d.get('leg_clearance_m',[d.get('clearance_m',.035)]*4))
    lead=np.asarray(d.get('leg_lead_offsets_s',[0.]*4))
    duties=np.asarray(d.get('leg_stance_ratio',[d['stance_ratio']]*4))
    if calibrate:
        clear+=np.clip((32.-np.asarray(base['clearance_p95_mm']))*.0004,-.006,.006)
        for a,c in [(0,3),(1,2)]:
            deltas=[]
            for event in ['touchdown','liftoff']:
                phases=co[event+'_mean_phase_per_foot']
                if phases[a] is not None and phases[c] is not None:
                    delta=(phases[c]-phases[a]+.5)%1-.5;deltas.append(delta)
            if deltas:
                shift=np.clip(np.mean(deltas)*T*.3,-.01,.01)
                lead[a]-=shift;lead[c]+=shift
    clear=np.clip(clear+rng.normal(0,.003*scale,4),.018,.080)
    lead=np.clip(lead+rng.normal(0,.007*scale,4),-.065,.065)
    duties=np.clip(duties+rng.normal(0,.012*scale,4),.51,.77)
    d['leg_clearance_m']=clear.tolist();d['leg_lead_offsets_s']=lead.tolist();d['leg_stance_ratio']=duties.tolist()
    if rng.random()<.25:d['period_s']=float(np.clip(round((T+rng.choice([-.02,.02]))/.02)*.02,.4,.8))
    d['body_height_m']=float(np.clip(d['body_height_m']+rng.normal(0,.004*scale),.265,.325))
    d['foot_x_offset_m']=float(np.clip(d.get('foot_x_offset_m',-.02)+rng.normal(0,.003*scale),-.045,.015))
    b['feed']=float(np.clip(b['feed']+rng.normal(0,.04*scale),1.1,1.6))
    b['lead']=float(np.clip(b['lead']+rng.normal(0,.005*scale),.03,.14))
    for key,default,noise,low,high in [('velocity_gain',2.,.4,1.,6.),('path_gain',1.,.2,.2,3.),
            ('heading_gain',1.,.3,.2,4.),('heading_step_gain',.5,.25,0.,3.),
            ('velocity_step_gain',0.,.12,0.,1.)]:
        d[key]=float(np.clip(d.get(key,default)+rng.normal(0,noise*scale),low,high))
    scale0=np.asarray(d.get('stride_scale',[1.,1.,1.]));cmd=np.asarray(base['command']);v=np.asarray(base['mean_velocity'])
    correction=np.clip((cmd-v)*.5,-.04,.04) if calibrate else np.zeros(3)
    scale0=np.clip(scale0+rng.normal(0,.03*scale,3)+np.divide(correction,cmd,out=np.zeros(3),where=abs(cmd)>.01),.65,1.65)
    d['stride_scale']=scale0.tolist()
    return {k:b[k] for k in ['clip','family','command','design','lead','feed']}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=BASE/'completion_search')
    p.add_argument('--families',default='');p.add_argument('--clips',default='')
    p.add_argument('--workers',type=int,default=8);p.add_argument('--generations',type=int,default=12)
    p.add_argument('--population',type=int,default=32);p.add_argument('--seconds',type=float,default=12.)
    a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True)
    selected=[c for c in cases() if (not a.families or c['family'] in a.families.split(',')) and (not a.clips or c['clip'] in a.clips.split(','))]
    rng=np.random.default_rng(20261116)
    with ProcessPoolExecutor(a.workers) as pool:
        for case in selected:
            dest=a.output/(case['clip']+'.json')
            if dest.exists() and json.loads(dest.read_text())[0]['passed']:continue
            prior=json.loads(dest.read_text())[:8] if dest.exists() else []
            candidates=seeds(case);best=prior;history=[]
            # Adjacent bins provide a better starting point than a global random search.
            nearest=[]
            for file in a.output.glob('*.json'):
                if file.stem==case['clip']:continue
                rows=json.loads(file.read_text())
                if rows and rows[0]['family']==case['family']:
                    nearest.append((np.linalg.norm(np.asarray(rows[0]['command'])-case['command']),rows[:3]))
            if nearest:
                for b in min(nearest,key=lambda x:x[0])[1]:
                    candidates.append(dict(**case,design=b['design'],lead=b['lead'],feed=b['feed']))
            for generation in range(a.generations+1):
                if generation:
                    noise=max(.18,.9*.9**generation)
                    candidates=[mutate(best[k%min(6,len(best))],rng,noise,k%3==0) for k in range(a.population)]
                batch=list(pool.map(evaluate,[(c,a.seconds,None) for c in candidates]))
                history.extend(batch);best=sorted([*best,*batch],key=lambda r:r['score'])[:12]
                dest.write_text(json.dumps(best,indent=2)+'\n')
                b=best[0]
                print(case['clip'],generation,round(b['score'],4),b['passed'],np.round(b['mean_velocity'],3),b['rejection_reasons'],flush=True)
                if b['passed']:break
            (a.output/(case['clip']+'.history.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in history))


if __name__=='__main__':main()
