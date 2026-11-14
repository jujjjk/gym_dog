"""Finite closed-loop second-simulator checks, no pose playback or auto reset."""
import argparse,json,hashlib,time
from pathlib import Path
import numpy as np
from sim2sim import ROOT,StudentSim,prepare_scene

def starts(mask):
    pad=np.r_[False,mask,False];ss=np.flatnonzero(~pad[:-1]&pad[1:]);ee=np.flatnonzero(pad[:-1]&~pad[1:]);return ss[ee-ss>=2]

def run(sim,motion,seconds,seed):
    sim.reset(seed=seed)
    for _ in range(100):sim.step(np.zeros(12,np.float32))
    sim.start(motion['command'],motion['gait_enabled'])
    records={k:[] for k in ['state','contact','clearance','qpos','action','raw','motor','applied','raw_substep_peak']}
    stopped='completed'
    for k in range(round(seconds/float(sim.policy_dt))):
        torque=sim.step()
        values=[sim.state(),sim.contacts.copy(),sim.d.xpos[sim.feet,2]-.02,sim.d.qpos.copy(),sim.action.copy(),
            torque[:,0],torque[:,1],torque[:,2],np.max(abs(torque[:,0]))]
        for key,value in zip(records,values):records[key].append(value)
        if sim.fallen():stopped='fallen';break
    arrays={key:np.array(v) for key,v in records.items()};cut=min(150,len(arrays['state'])//3)
    state=arrays['state'][cut:];contact=arrays['contact'][cut:];count=contact.sum(1)
    diag=(contact[:,0]&contact[:,3])|(contact[:,1]&contact[:,2]);bad=(count==0)|(count==1)|((count==2)&~diag)
    v=state[:,[36,37,41]];cmd=np.array(motion['command']);cycle=25
    mean=np.stack([np.convolve(v[:,j],np.ones(cycle)/cycle,'valid') for j in range(3)],1) if len(v)>=cycle else v
    gaps=[];lifts=[len(starts(~contact[:,i])) for i in range(4)]
    for aa,bb in [(0,3),(1,2)]:
        a,b=starts(contact[:,aa]),starts(contact[:,bb])
        if len(a) and len(b):gaps.extend(float(np.min(abs(b-t)))*float(sim.policy_dt) for t in a)
    touchdown=float(np.percentile(gaps,95)) if gaps else None
    groups=np.full(len(contact),-1);groups[contact[:,0]&contact[:,3]&~contact[:,1]&~contact[:,2]]=0
    groups[contact[:,1]&contact[:,2]&~contact[:,0]&~contact[:,3]]=1
    seq=groups[groups>=0];alternations=int(np.sum(seq[1:]!=seq[:-1]));elapsed=len(state)*float(sim.policy_dt)
    foot=np.percentile(arrays['clearance'][cut:],95,axis=0)*1000
    torque={}
    for key in ['raw','motor','applied']:
        x=arrays[key][cut:];torque[key]=dict(mean_abs_Nm=float(abs(x).mean()),rms_Nm=float(np.sqrt((x*x).mean())),
            steady_peak_Nm=float(abs(x).max()),whole_rollout_peak_Nm=float(abs(arrays[key]).max()),
            fraction_over_14=float((abs(x)>14).mean()),fraction_over_17=float((abs(x)>17).mean()))
    reasons=[]
    if stopped!='completed':reasons.append('fallen')
    if np.mean(count==0)>0:reasons.append('flight')
    if np.mean(bad)>.001:reasons.append('illegal_support')
    if torque['raw']['whole_rollout_peak_Nm']>17 or torque['raw']['fraction_over_14']>.01:reasons.append('torque')
    if motion['gait_enabled']:
        if min(lifts)<.7*elapsed*2:reasons.append('not_stepping')
        if alternations<.7*elapsed*4:reasons.append('not_alternating_diagonals')
        if touchdown is None or touchdown>.040001:reasons.append('diagonal_timing')
    elif float(np.mean(count==4))<.95 or float(foot.max())>5.:
        reasons.append('standing_not_static')
    gym_path=ROOT/'artifacts/urdf4_selection_20261114/verify17800/result.json'
    gym_foot_error=None
    if gym_path.exists() and motion['gait_enabled']:
        source=json.loads(gym_path.read_text())
        if source['checkpoint_sha256']==sim.c['checkpoint_sha256']:
            baseline=next(r for r in source['reports'][0]['results'] if r['clip']==motion['name'])
            gym_foot_error=float(abs(foot-np.array(baseline['clearance_p95_mm'])).max())
            if gym_foot_error>10:reasons.append('foot_trajectory_vs_gym')
    row=dict(motion=motion['name'],command=motion['command'],gait_enabled=motion['gait_enabled'],seed=seed,duration_s=seconds,stop=stopped,
        completed_steps=len(arrays['state']),warmup_excluded_s=cut*float(sim.policy_dt),mean_velocity=v.mean(0).tolist(),
        cycle_linear_error_m_s=float(np.linalg.norm(mean[:,:2]-cmd[:2],axis=1).mean()),
        cycle_yaw_error_rad_s=float(abs(mean[:,2]-cmd[2]).mean()),
        velocity_cycle_p05=np.percentile(mean,5,axis=0).tolist(),velocity_cycle_p95=np.percentile(mean,95,axis=0).tolist(),
        foot_clearance_p95_mm=foot.tolist(),max_clearance_difference_vs_gym_mm=gym_foot_error,
        diagonal_touchdown_p95_s=touchdown,lifts_per_foot=lifts,diagonal_alternations=alternations,
        flight_fraction=float(np.mean(count==0)),all_feet_support_fraction=float(np.mean(count==4)),illegal_support_fraction=float(np.mean(bad)),torque=torque,
        max_tilt_deg=float(np.degrees(np.arccos(np.clip(-state[:,44],-1,1))).max()),
        rejection_reasons=reasons,gait_passed=not reasons)
    return row,arrays

def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,default=ROOT/'artifacts/urdf4_mujoco_20261114')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--seconds',type=float,default=30.)
    p.add_argument('--seeds',default='1,2,3');p.add_argument('--motion',default='all');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);sim=StudentSim(a.bundle);rows=[]
    motions=[m for m in sim.c['motions'] if a.motion in ('all',m['name'])]
    if not motions:p.error('Unknown motion')
    for seed in map(int,a.seeds.split(',')):
        for move in motions:
            r,arrays=run(sim,move,a.seconds,seed);rows.append(r)
            np.savez_compressed(str(a.output/(move['name']+'_seed%d.npz'%seed)),**arrays)
            result=dict(checkpoint=sim.c['checkpoint'],checkpoint_sha256=sim.c['checkpoint_sha256'],
                numpy_export_max_error=sim.c['numpy_export_max_error'],scene_sha256=hashlib.sha256((a.bundle/'scene.xml').read_bytes()).hexdigest(),
                physics_dt_s=sim.dt,policy_dt_s=float(sim.policy_dt),root_or_joint_overwrite_during_rollout=False,
                automatic_resets=False,tracking_used_as_gait_gate=False,results=rows,gait_passed=all(r['gait_passed'] for r in rows),
                trained_dynamic_gaits_passed=all(r['gait_passed'] for r in rows if r['gait_enabled']))
            (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
            print(move['name'],seed,'gait',r['gait_passed'],'v',np.round(r['mean_velocity'],3),'foot',np.round(r['foot_clearance_p95_mm'],1),
                'raw',round(r['torque']['raw']['whole_rollout_peak_Nm'],2),r['rejection_reasons'],flush=True)
    print('COMPLETE',a.output,flush=True)

if __name__=='__main__':main()
