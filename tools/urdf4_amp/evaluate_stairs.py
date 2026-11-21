"""Measure a learned stair policy on continuous, matched single-platform courses."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
from xml.etree import ElementTree as ET

import isaacgym
import numpy as np
import torch

from legged_gym.envs.urdf4.amp_cts_env import yaw_from_quat
from legged_gym.envs.urdf4.amp_stairs import stair_scan

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'mujoko/urdf4_amp'))
from sim2sim_transition import TransitionSim, prepare_bundle, sha


def course_metrics(samples,height,length,exit_m,dt):
    x,y,vx,roll,pitch,raw,done=samples[:,:7].T
    foot=samples[:,7:11].astype(bool)
    foot_x,foot_z=samples[:,11:15],samples[:,15:19]
    on_top=(foot_x>0.) & (foot_x<length) & (foot_z>height+.008)
    top=np.flatnonzero((on_top & foot).all(1) & ~done.astype(bool))
    rear_clear=(foot_x>length+.025).all(1)
    ground = rear_clear & (foot_z<.045).all(1) & (foot.sum(1)>=2) & ~done.astype(bool)
    settled=np.flatnonzero(ground)
    finish=np.flatnonzero((x>length+exit_m) & ground)
    completion=bool(len(top) and len(finish) and finish[0]>top[0] and not done[:finish[0]+1].any())
    limit=finish[0]+1 if completion else len(samples)
    # Stop scoring after the required flat exit.
    moving=(np.arange(len(x))*dt>1.6) & (np.arange(len(x))<limit)
    stall=moving & (vx<.04)
    switches=np.flatnonzero(np.diff(np.r_[False,stall,False]))
    longest=float(np.diff(switches)[::2].max()*dt) if len(switches) else 0.
    metrics=dict(height_m=height,length_m=length,completed=completion,
                 four_feet_on_top_s=float((top[0]+1)*dt) if len(top) else None,
                 recovered_flat_support_s=float((settled[0]+1)*dt) if len(settled) else None,
                 complete_s=float((finish[0]+1)*dt) if completion else None,
                 max_x_m=float(x[:limit].max()),
                 peak_cross_m=float(abs(y[:limit]).max()), peak_roll_deg=float(abs(roll[:limit]).max()),
                 peak_pitch_deg=float(abs(pitch[:limit]).max()),
                 peak_raw_Nm=float(raw[:limit].max()),longest_stall_s=longest)
    metrics['passed']=bool(completion and metrics['peak_cross_m']<=.08
                           and metrics['peak_raw_Nm']<=17. and longest<=.60)
    return metrics


def gym_evaluate(env,policy,model,out):
    steps=round(19.5/env.dt)
    rows=[]
    raw=torch.zeros(env.num_envs,device=env.device)
    source=env._compute_torques
    def measure(action):
        torque=source(action)
        raw.copy_(torch.maximum(raw,env.raw_pd_torques.abs().amax(1)))
        return torque
    env._compute_torques=measure
    with torch.inference_mode():
        obs,_=env.reset()
        for _ in range(steps):
            raw.zero_()
            obs,_,_,done,_=env.step(policy(obs))
            gx,gy,gz=env.projected_gravity.unbind(1)
            rows.append(torch.cat((env.root_states[:,:1],(env.root_states[:,1]-env.env_origins[:,1])[:,None],
                env.root_states[:,7:8],torch.rad2deg(torch.atan2(-gy,-gz))[:,None],
                torch.rad2deg(torch.asin(gx.clamp(-1,1)))[:,None],raw[:,None],done[:,None],
                env.get_foot_contact_mask(),env.feet_pos[:,:,0],env.feet_pos[:,:,2]),1))
    trace=torch.stack(rows).cpu().numpy()
    reports=[]
    for i,(height,length) in enumerate(env.stair_lanes[env.stair_lane].cpu().tolist()):
        run=trace[:,i]
        stop=np.flatnonzero(run[:,6])
        reports.append(course_metrics(run[:stop[0]+1] if len(stop) else run,height,length,env.cfg.stairs.exit_m,env.dt))
    np.savez_compressed(out/'trace.npz',trace=trace,dt_s=env.dt)
    write_report(out,model,'gym',reports)


def platform_height(xy,height,length,half_width):
    return height*((xy[...,0]>=0.) & (xy[...,0]<=length) & (abs(xy[...,1])<half_width))


class StairSim(TransitionSim):
    def __init__(self,bundle,policy,height,length,half_width):
        self.height,self.length,self.half_width=height,length,half_width
        super().__init__(bundle,policy)
        self.support_geoms=(self.ground,self.m.geom('step').id)

    def ground_height(self,xy):
        return platform_height(xy,self.height,self.length,self.half_width)

    def observe(self):
        super().observe()
        if self.controller is not None:
            c=self.controller
            xy=c.root_states[:,:2]
            scan=stair_scan(xy,yaw_from_quat(c.root_states[:,3:7]),self.ground_height)
            self.public=torch.cat((self.public,scan),1)


def add_platform(source,dest,height,length,half_width):
    scene=ET.parse(source)
    ET.SubElement(scene.find('worldbody'),'geom',name='step',type='box',
        pos=f'{length/2} 0 {height/2}',size=f'{length/2} {half_width} {height/2}',rgba='.36 .48 .57 1',
        contype='1',conaffinity='2',friction='.8 0 0',solref='.0065 1',solimp='.99 .999 .0001 .5 2',condim='3')
    scene.write(dest)


def mujoco_trial(job):
    out,bundle,height,length,cfg,offset=job
    path=out/f'h{height:.3f}_l{length:.2f}'
    path.mkdir()
    (path/'contract.json').write_bytes((bundle/'contract.json').read_bytes())
    add_platform(bundle/'scene.xml',path/'scene.xml',height,length,cfg['half_width_m'])
    checkpoint=json.loads((path/'contract.json').read_text())
    policy=torch.jit.load(checkpoint['torchscript'],map_location='cpu').eval()
    sim=StairSim(path,policy,height,length,cfg['half_width_m'])
    dt=float(sim.policy_dt)
    rows,positions,velocities=[],[],[]
    with torch.inference_mode():
        sim.reset(offset=(-cfg['approach_m']+offset,0.))
        sim.start((cfg['speed_m_s'],0.,0.))
        for _ in range(round(19.5/dt)):
            torque=sim.step()
            c=sim.controller
            gx,gy,gz=c.projected_gravity[0].tolist()
            roll,pitch=np.degrees([np.arctan2(-gy,-gz),np.arcsin(np.clip(gx,-1,1))])
            xy=sim.d.qpos[:2]
            feet=sim.d.xpos[sim.feet]
            fall=sim.fallen() or sim.d.qpos[2]-sim.ground_height(xy)<sim.c['min_base_height_m']
            rows.append(np.r_[xy,c.root_states[0,7],roll,pitch,abs(torque[:,0]).max(),fall,sim.contacts,feet[:,0],feet[:,2]])
            positions.append(sim.d.qpos.copy())
            velocities.append(sim.d.qvel.copy())
            if fall:
                break
    data=np.stack(rows)
    np.savez_compressed(path/'trace.npz',trace=data,dt_s=dt,qpos=positions,qvel=velocities,
                        time_s=(np.arange(len(data))+1)*dt)
    metrics=course_metrics(data,height,length,cfg['exit_m'],dt)
    metrics['offset_m']=offset
    return metrics


def write_report(out,model,engine,reports):
    report=dict(engine=engine,model=str(model.resolve()),checkpoint_sha256=sha(model),cases=reports,
                completed=sum(r['completed'] for r in reports),passed=sum(r['passed'] for r in reports))
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--spawn-shift',type=float,default=0.)
    args=parser.parse_args()
    torch.set_num_threads(1)
    torch._C._jit_set_profiling_mode(False)
    torch._C._jit_set_profiling_executor(False)
    args.output.mkdir(parents=True,exist_ok=False)
    bundle=args.output/'base_bundle'
    prepare_bundle(bundle,args.model,False)
    data=json.loads((bundle/'contract.json').read_text())
    cfg=data['stairs']
    jobs=[(args.output,bundle,h,l,cfg,args.spawn_shift) for h in cfg['heights_m'] for l in cfg['lengths_m']]
    with ProcessPoolExecutor(max_workers=min(6,len(jobs))) as pool:
        reports=list(pool.map(mujoco_trial,jobs))
    write_report(args.output,args.model,'mujoco',reports)


if __name__=='__main__':
    main()
