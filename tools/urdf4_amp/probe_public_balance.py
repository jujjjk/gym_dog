"""Compare privileged and sensor-only balance feedback through the real plant.

Privileged rows are calibration references, never presented as deployable policy.
Public rows reconstruct foot kinematics from joints and IMU only.
"""
import argparse,copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'unitree_rl_gym/legged_gym/scripts'))
import isaacgym
import numpy as np,torch
import mujoco as mj
from legged_gym.envs import *
from legged_gym.utils import get_args,task_registry
from teacher_v3 import Teacher,rot
from legged_gym.envs.urdf4.amp_reference import amp_features


def public_state(teacher,frame,default):
    q=frame[9:21]+default;dq=frame[21:33]/.05;omega=frame[:3]/.25;gravity=frame[3:6];gravity/=np.linalg.norm(gravity)
    roll=np.arctan2(-gravity[1],-gravity[2]);pitch=np.arcsin(np.clip(gravity[0],-1,1))
    cr,sr=np.cos(roll/2),np.sin(roll/2);cp,sp=np.cos(pitch/2),np.sin(pitch/2);quat=np.array([cr*cp,sr*cp,cr*sp,-sr*sp])
    m=teacher.m;d=teacher.d;d.qpos[:]=0;d.qvel[:]=0;d.qpos[3]=1;d.qpos[teacher.qa]=q;mj.mj_forward(m,d)
    feet=d.xpos[teacher.fb].copy();vel=[];angular=[];jacr=np.zeros_like(teacher.J)
    for foot in teacher.fb:
        mj.mj_jacBody(m,d,teacher.J,jacr,foot);vel.append(teacher.J[:,teacher.va]@dq);angular.append(jacr[:,teacher.va]@dq+omega)
    heights=.02+feet@gravity
    weight=np.exp((heights-heights.max())/.003);weight/=weight.sum()
    velocity=-(np.asarray(vel)+np.cross(np.tile(omega,(4,1)),feet)+np.cross(np.asarray(angular),np.tile(.02*gravity,(4,1))));bodyv=(weight[:,None]*velocity).sum(0)
    h=float(heights@weight);qpos=np.zeros(m.nq);qvel=np.zeros(m.nv);qpos[:3]=[0,0,h];qpos[3:7]=quat;qpos[teacher.qa]=q
    qvel[:3]=rot(quat)@bodyv;qvel[3:6]=omega;qvel[teacher.va]=dq
    return qpos,qvel,bodyv,h


def main():
    p=argparse.ArgumentParser(add_help=False,allow_abbrev=False);p.add_argument('--output',type=Path,required=True);p.add_argument('--seconds',type=float,default=20)
    p.add_argument('--rate-ablation',action='store_true')
    p.add_argument('--intermediate',action='store_true',help='Diagnose true closed-loop 0.40/0.45 references; not policy acceptance')
    p.add_argument('--fast-template',action='store_true',help='Keep the validated 0.50 geometry/timing and redesign Cartesian stride for intermediate speed')
    p.add_argument('--held-geometry',action='store_true',help='Hold full .50 foot geometry; test exact public height/vertical-velocity feedback and the existing public speed/path servo')
    own,rest=p.parse_known_args();sys.argv=sys.argv[:1]+rest;own.output.mkdir(parents=True,exist_ok=False)
    args=get_args();args.task='urdf4_amp_cts_v3';args.headless=True
    commands=[(.35,0,0),(.5,0,0),(.6,0,0)];modes=['truth_teacher','public_teacher','public_pose_velocity']
    if own.rate_ablation:commands=[(.5,0,0),(.6,0,0)];modes=['truth_base','truth_rate125','truth_rate150','truth_both150','truth_pd90']
    if own.intermediate:commands=[(.4,0,0),(.45,0,0)];modes=['truth_teacher']
    if own.held_geometry:
        args.task='urdf4_amp_cts_v4';commands=[(.4,0,0),(.45,0,0),(.5,0,0),(.6,0,0)];modes=['truth_held','public_held']
    args.num_envs=len(commands)*len(modes)
    cfg,_=task_registry.get_cfgs(args.task);cfg.env.test=True;cfg.env.episode_length_s=own.seconds+10;cfg.expand.cold_settle_s=2.
    if own.held_geometry:cfg.direction.speed_feedback_enabled=True
    env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg);target=torch.tensor(commands,device=env.device).repeat(len(modes),1)
    if own.fast_template or own.held_geometry:
        fast_id=(env.reference_commands-env.commands.new_tensor([.5,0,0])).square().sum(1).argmin()
        env._select_reference=lambda command:torch.full((len(command),),int(fast_id),device=env.device,dtype=torch.long)
    manifest=json.load(open(Path(cfg.amp_cts.reference)/'manifest.json'));teachers=[];fix=[]
    for i,cmd in enumerate(target.cpu().numpy()):
        spec=copy.deepcopy(next(c for c in manifest['clips'] if c['command']==[.5,0,0]) if own.fast_template or own.held_geometry else min(manifest['clips'],key=lambda c:np.linalg.norm(np.array(c['command'])-cmd)))
        if not own.held_geometry:spec['command']=cmd.tolist()
        spec['design'].update(stop_anchor=True,route_progress_projection=True)
        if modes[i//len(commands)]=='truth_pd90':spec['feed']/=.9
        teacher=Teacher(ROOT/'artifacts/urdf4_amp_design_20261116_v3/plant',spec);teachers.append(teacher)
        fix.append((teacher.q0-env.default_dof_pos[0].cpu().numpy())*(1/.928-1))
    fix=np.stack(fix);env.set_command_override(target);env.reset();fallen=torch.zeros(env.num_envs,device=env.device,dtype=torch.bool)
    if own.rate_ablation:
        env.target_rate_limit=env.target_rate_limit.expand(env.num_envs,-1).clone();env.target_acceleration_limit=env.target_acceleration_limit.expand(env.num_envs,-1).clone()
        for i in range(env.num_envs):
            mode=modes[i//len(commands)]
            if mode=='truth_rate125':env.target_rate_limit[i]*=1.25
            if mode in ['truth_rate150','truth_both150']:env.target_rate_limit[i]*=1.5
            if mode=='truth_both150':env.target_acceleration_limit[i]*=1.5
            if mode=='truth_pd90':env.kp[i]*=.9;env.kd[i]*=.9
    def no_reset(ids):fallen[ids]=True
    env.reset_idx=no_reset;original=env._compute_torques;peak=torch.zeros(env.num_envs,device=env.device)
    def torques(actions):
        value=original(actions);peak.copy_(torch.maximum(peak,env.raw_pd_torques.abs().amax(1)*(~fallen)));return value
    env._compute_torques=torques;records=[];default=env.default_dof_pos[0].cpu().numpy();scale=env.action_scale.cpu().numpy()
    for k in range(round(own.seconds/env.dt)):
        roots=env.root_states.cpu().numpy().copy();roots[:,:3]-=env.env_origins.cpu().numpy();q=env.dof_pos.cpu().numpy();dq=env.dof_vel.cpu().numpy()
        frame=env.obs_buf[:,:51].cpu().numpy();phase=env.motion_phase.cpu().numpy();external=env.external_commands.cpu().numpy();pose=env.pose_queue[-1].cpu().numpy();qcommand=[];est=[]
        for i,teacher in enumerate(teachers):
            privileged=modes[i//len(commands)].startswith('truth')
            qp,qv,vkin,hkin=public_state(teacher,frame[i].copy(),default)
            if own.held_geometry:
                if privileged:qp[2]=roots[i,2];qv[2]=roots[i,9]
                # The localization stream is yaw-aligned horizontal velocity,
                # not a full roll/pitch body vector. Keep FK-estimated world
                # vertical velocity; rotating it by world yaw again is wrong.
                qv[:2]=env.obs_buf[i,1026:1028].cpu().numpy()
                qp[:2]=0.;teacher.reference_xy[:]=0.;teacher.reference_yaw=0.;teacher.heading_integral=0.
            elif privileged:
                qp[:3]=roots[i,:3];qp[3:7]=roots[i,[6,3,4,5]];qp[teacher.qa]=q[i];qv[:3]=roots[i,7:10];qv[3:6]=rot(qp[3:7]).T@roots[i,10:13];qv[teacher.va]=dq[i]
            else:
                teacher.reference_xy[:]=0.;teacher.reference_yaw=0.;teacher.heading_integral=0.
                if i>=2*len(commands):
                    yaw=pose[i,2];co,si=np.cos(yaw),np.sin(yaw);vx,vy=pose[i,3:5]
                    qv[:3]=[co*vx+si*vy,-si*vx+co*vy,0.]
            if k<2 and not own.held_geometry:
                teacher.reference_xy=qp[:2].copy();teacher.reference_yaw=0.
            amplitude=float(np.linalg.norm(external[i])/max(np.linalg.norm(target[i].cpu().numpy() if own.held_geometry else teacher.cmd),1e-9))
            theta=(phase[i]-float(env.reference_phase_offsets[env.reference_id[i]]))%1
            planned=external[i] if privileged and not own.held_geometry else frame[i,6:9]/np.array([2.,2.,.25])
            qcommand.append(teacher.command(k*env.dt-2.,qp,qv,dt=env.dt,phase_override=theta,planned_override=planned,amplitude_override=amplitude))
            est.append([*vkin,hkin])
        action=torch.tensor((np.asarray(qcommand)+fix-default)/scale,device=env.device,dtype=torch.float);action[fallen]=0.
        with torch.no_grad():env.step(action)
        records.append(dict(features=amp_features(env).cpu().numpy(),contact=env.get_foot_contact_mask().cpu().numpy(),height=(env.feet_pos[:,:,2]-env.env_origins[:,None,2]-.02).cpu().numpy(),
            estimates=np.asarray(est),cross=env.path.cross_error.cpu().numpy().copy(),heading=env.path.heading_error.cpu().numpy().copy(),valid=(~fallen).cpu().numpy()))
        if (k+1)%250==0:print('PUBLIC BALANCE',k*env.dt,'fallen',int(fallen.sum()),'peak',peak.tolist(),flush=True)
    a={key:np.stack([r[key] for r in records]) for key in records[0]};t=np.arange(1,len(records)+1)*env.dt;rows=[]
    for i in range(env.num_envs):
        mask=(t>=6)&a['valid'][:,i];f=a['features'][mask,i];ct=a['contact'][mask,i];n=ct.sum(1);di=(ct[:,0]&ct[:,3])|(ct[:,1]&ct[:,2])
        row=dict(mode=modes[i//len(commands)],command=target[i].tolist(),fallen=bool(fallen[i]),raw_peak_Nm=float(peak[i]),
            mean_velocity=f[:,[36,37,41]].mean(0).tolist() if len(f) else None,mean_height=float(f[:,45].mean()) if len(f) else None,
            estimator_velocity_bias=(a['estimates'][mask,i,:3]-f[:,36:39]).mean(0).tolist() if len(f) else None,
            illegal_support_fraction=float(((n==0)|(n==1)|((n==2)&~di)).mean()) if len(n) else 1.,
            cross_max_m=float(abs(a['cross'][mask,i]).max()) if mask.any() else None,heading_mean_deg=float(abs(a['heading'][mask,i]).mean()*180/np.pi) if mask.any() else None)
        rows.append(row);print('PUBLIC RESULT',row,flush=True)
    np.savez_compressed(own.output/'rollout.npz',**a,time_s=t)
    (own.output/'result.json').write_text(json.dumps(dict(cases=rows,public_sensor_contract='joints, IMU, commands; public_pose_velocity also uses the existing corrected external planar localization stream',training=False),indent=2)+'\n')


if __name__=='__main__':main()
