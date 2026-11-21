"""Continuous physical start/stop, speed changes and direction changes for V3.

These variable-command records are kept separate from legacy constant-command
AMP files. A stop holds the actual stopping point rather than commanding a
return along the travelled path. Heading remains referenced to the commanded yaw.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from physical_v3 import run
from complete_v3 import BASE


def programs():
    out=[]
    for family,cmd in [('forward',(.2,0,0)),('backward',(-.15,0,0)),('left',(0,.1,0)),
            ('right',(0,-.1,0)),('turn_left',(0,0,.4)),('turn_right',(0,0,-.4)),
            ('forward_left',(.15,.08,0)),('forward_right',(.15,-.08,0)),
            ('backward_left',(-.15,.08,0)),('backward_right',(-.15,-.08,0))]:
        points=[(0,(0,0,0)),(2,(0,0,0)),(5,cmd),(9,cmd),(12,(0,0,0)),(18,(0,0,0))]
        out.append(dict(clip='start_stop_'+family,family=family,command=list(cmd),points=points))
    for name,first,second in [
        ('forward_speed',(.15,0,0),(.30,0,0)),
        ('backward_speed',(-.08,0,0),(-.25,0,0)),
        ('lateral_speed',(0,.05,0),(0,.15,0)),
        ('turn_speed',(0,0,.2),(0,0,.6)),
        ('forward_to_left',(.15,0,0),(0,.1,0)),
        ('forward_to_right',(.15,0,0),(0,-.1,0)),
        ('forward_to_turn',(.15,0,0),(0,0,.3)),
        ('turn_to_forward',(0,0,.3),(.15,0,0))]:
        points=[(0,(0,0,0)),(2,(0,0,0)),(5,first),(8,first),(11,second),
            (15,second),(18,first),(21,first),(24,(0,0,0)),(30,(0,0,0))]
        family='backward' if name.startswith('backward') else 'left' if name.startswith('lateral') else 'turn_left' if name in ['turn_speed','turn_to_forward'] else 'forward'
        out.append(dict(clip=name,family=family,command=list(first),points=points))
    return out


def assess(r,a):
    contact=a['contact'];count=contact.sum(1);diag=(contact[:,0]&contact[:,3])|(contact[:,1]&contact[:,2])
    # Initial and final quiet support are explicitly tested, not called marching.
    final=a['time']>=a['time'][-1]-2.;velocity=a['features'][final][:,[36,37,41]].mean(0)
    zeros=np.linalg.norm(a['command'],axis=1)<1e-6
    heading=np.arctan2(np.sin(a['rpy'][:,2]-a['reference_yaw']),np.cos(a['rpy'][:,2]-a['reference_yaw']))
    cmd=a['command'];normal=np.c_[-np.sin(a['reference_yaw']+np.arctan2(cmd[:,1],cmd[:,0])),np.cos(a['reference_yaw']+np.arctan2(cmd[:,1],cmd[:,0]))]
    delta=a['qpos'][:,:2]-a['reference_xy'];cross=np.where(np.linalg.norm(cmd[:,:2],axis=1)>.01,abs((delta*normal).sum(1)),np.linalg.norm(delta,axis=1))
    illegal=(count==1)|((count==2)&~diag);pad=np.r_[False,illegal,False];durations=(np.flatnonzero(pad[:-1]&~pad[1:])-np.flatnonzero(~pad[:-1]&pad[1:]))*.02
    r.update(final_mean_velocity=velocity.tolist(),final_four_foot_support_fraction=float(contact[final].all(1).mean()),longest_illegal_support_s=float(durations.max()) if len(durations) else 0.,
        full_flight_fraction=float((count==0).mean()),full_illegal_support_fraction=float(((count==1)|((count==2)&~diag)).mean()),
        full_heading_mean_abs_deg=float(abs(heading).mean()*180/np.pi),full_cross_max_m=float(cross.max()),
        first_stand_clearance_max_mm=float(a['foot_clearance'][a['time']<1.5].max()*1000))
    errors=[]
    if r['stop']!='completed':errors.append('fall')
    if r['raw_peak_Nm']>17:errors.append('raw_torque')
    if r['full_flight_fraction']>0:errors.append('flight')
    # Short clips need an event-duration gate: one isolated 20ms force-threshold
    # crossing must not fail only because the clip is shorter than the 60s screen.
    if r['full_illegal_support_fraction']>.004 or r['longest_illegal_support_s']>.020001:errors.append('support')
    if np.linalg.norm(velocity[:2])>.015 or abs(velocity[2])>.03 or r['final_four_foot_support_fraction']<.98:errors.append('stop')
    if r['full_cross_max_m']>.06 or r['full_heading_mean_abs_deg']>2.5:errors.append('direction')
    if r['first_stand_clearance_max_mm']>5:errors.append('zero_command_marching')
    windows=[];schedule=r['command_schedule'];T=r['design']['period_s']
    for first,last in zip(schedule[:-1],schedule[1:]):
        c=np.asarray(first['command'])
        if np.linalg.norm(c)<1e-8 or not np.allclose(c,last['command']):continue
        mask=(a['time']>=first['time_s']+.5)&(a['time']<last['time_s']-.1)
        features=a['features'][mask];clear=a['foot_clearance'][mask];ct=contact[mask]
        mean=features[:,[36,37,41]].mean(0);lifts=[]
        for j in range(4):
            active=(clear[:,j]>.012)&~ct[:,j];pad=np.r_[False,active,False];st=np.flatnonzero(~pad[:-1]&pad[1:]);en=np.flatnonzero(pad[:-1]&~pad[1:]);lifts.append(int((en-st>=2).sum()))
        passed=np.linalg.norm(mean[:2]-c[:2])<=.04 and abs(mean[2]-c[2])<=.10 and min(np.quantile(clear,.95,axis=0))>=.015 and min(lifts)>=max(1,int(mask.sum()*.02/T*.5))
        windows.append(dict(time_s=[first['time_s']+.5,last['time_s']-.1],command=c.tolist(),mean_velocity=mean.tolist(),lift_events=lifts,clearance_p95_mm=(np.quantile(clear,.95,axis=0)*1000).tolist(),passed=bool(passed)))
    r['steady_windows']=windows
    if not windows or not all(w['passed'] for w in windows):errors.append('steady_motion')
    r['passed']=not errors;r['rejection_reasons']=errors
    r['score']=100*len(errors)+100*r['full_illegal_support_fraction']+r['full_cross_max_m']+r['full_heading_mean_abs_deg']*.01
    return r


def evaluate(job):
    program,b,output=job;d=copy.deepcopy(b['design']);d.update(command_schedule=[dict(time_s=t,command=list(c)) for t,c in program['points']],ramp_s=0.,stop_anchor=True,route_progress_projection=True)
    folder=Path(output);r=run(BASE/'plant',program['command'],program['points'][-1][0],b['lead'],b['feed'],d['body_height_m'],save=folder,overrides=d)
    a=dict(np.load(folder/'physical.npz'));r.update(design=d,clip=program['clip'],family=program['family'],candidate=b['clip'],command_schedule=d['command_schedule'],source_folder=str(folder))
    r=assess(r,a);(folder/'report.json').write_text(json.dumps(r,indent=2)+'\n');return r


def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=6);a=p.parse_args();out=BASE/'transition_search';out.mkdir(exist_ok=True)
    bases=[]
    for file in (BASE/'confirmation60').glob('*.json'):
        row=json.loads(file.read_text())
        if isinstance(row,dict) and row.get('passed'):bases.append(row)
    jobs=[]
    for program in programs():
        candidates=sorted([b for b in bases if b['family']==program['family']],key=lambda b:np.linalg.norm(np.asarray(b['command'])-program['command']))[:3]
        for i,b in enumerate(candidates):
            for ki in [0.,.3]:
                for activation in [0.,.35]:
                    b=copy.deepcopy(b);b['design'].update(heading_integral_gain=ki,gait_activation_threshold=activation)
                    jobs.append((program,b,str(out/(program['clip']+'_'+str(i)+'_'+str(ki)+'_a'+str(activation)))))
    rows=[]
    with ProcessPoolExecutor(a.workers) as pool:
        for r in pool.map(evaluate,jobs):
            rows.append(r);print(r['clip'],r['passed'],r['rejection_reasons'],flush=True)
    (out/'ranking.json').write_text(json.dumps(sorted(rows,key=lambda r:r['score']),indent=2)+'\n')


if __name__=='__main__':main()
