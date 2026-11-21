"""Native Gym viewer for the frozen gait and explicit path-pose backend."""
import argparse,copy,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'unitree_rl_gym/legged_gym/scripts'))
import isaacgym
from isaacgym import gymapi
import torch,numpy as np
from legged_gym.envs import *
from legged_gym.utils import get_args,task_registry
from legged_gym.envs.urdf4.path_runtime import PathTrackingSession
from legged_gym.envs.urdf4.path_tracking import PathGains,rotate_xy


def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--pose_source',choices=['oracle','student'],default='oracle')
    p.add_argument('--controller',choices=['off','heading','path'],default='path')
    p.add_argument('--experimental_all',action='store_true')
    p.add_argument('--gains',type=Path,default=ROOT/'artifacts/urdf4_path_20261115/qualified_gains.json')
    own,rest=p.parse_known_args();sys.argv=sys.argv[:1]+rest
    args=get_args();args.task='urdf4_amp_cts_expand_x_v1';args.num_envs=1
    cfg,_=task_registry.get_cfgs(args.task);cfg=copy.deepcopy(cfg)
    cfg.env.test=True;cfg.noise.add_noise=False;cfg.push.enabled=False
    cfg.env.episode_length_s=args.duration_s+10;cfg.amp_cts.reference_reset_probability=0.
    cfg.rewards.heading_gain=0.
    env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
    policy_path=ROOT/'artifacts/urdf4_selection_x_20261115/best/student_0.pt'
    profile=json.loads((ROOT/'artifacts/urdf4_path_20261115/controller_profile.json').read_text())
    if hashlib.sha256(policy_path.read_bytes()).hexdigest()!=profile['policy_sha256']:
        raise ValueError('Frozen policy changed; revalidate this controller profile first')
    policy=torch.jit.load(str(policy_path),map_location=env.device).eval()
    for parameter in policy.parameters():parameter.requires_grad_(False)
    env.set_command_override(torch.zeros(1,3,device=env.device),gait_enabled=False);obs,_=env.reset()
    with torch.no_grad():
        for _ in range(round(2/env.dt)):obs,_,_,_,_=env.step(torch.zeros(1,12,device=env.device))
        mode=['off','heading','path'].index(own.controller)
        gains=PathGains(**json.loads(own.gains.read_text()))
        session=PathTrackingSession(env,policy,mode=mode,pose_source=own.pose_source,gains=gains,
            control_scope='experimental_all' if own.experimental_all else 'forward')
        env.episode_length_buf.zero_();env.clock_offset.zero_();env.command_resample_steps=10**9
        env.compute_observations();obs=env.obs_buf
        command=torch.tensor([[args.vx,args.vy,args.wz]],device=env.device)
        tangent=command[0,:2]/command[0,:2].norm().clamp(min=1.e-6)
        normal=torch.stack((-tangent[1],tangent[0]));last=env.root_states[0,:3].cpu().numpy().copy()
        if env.viewer is not None:
            origin=env.root_states[0,:2].clone()-env.env_origins[0,:2]
            t=rotate_xy(session.truth_heading,tangent[None])[0]
            n=rotate_xy(session.truth_heading,normal[None])[0]
            for offset,color in [(0.,[0.,1.,0.]),(-.05,[1.,1.,0.]),(.05,[1.,1.,0.])]:
                first=origin+n*offset-t;second=origin+n*offset+t*25
                vertices=np.array([[float(first[0]),float(first[1]),.025],[float(second[0]),float(second[1]),.025]],dtype=np.float32)
                env.gym.add_lines(env.viewer,env.envs[0],1,vertices,np.asarray([color],dtype=np.float32))
        print('PATH VIEWER:',own.controller,'pose_source='+own.pose_source,'external_command='+str(command[0].tolist()),flush=True)
        if own.pose_source=='oracle':print('DIAGNOSTIC: simulator pose is used as external localization; real hardware requires its own odometry.',flush=True)
        else:print('DIAGNOSTIC: student velocity dead reckoning may drift; this is not a globally corrected pose source.',flush=True)
        try:
            for k in range(round(args.duration_s/env.dt)):
                (obs,_,_,done,info),telemetry=session.step(command,obs)
                if k%10==0 and env.viewer is not None:
                    here=env.root_states[0,:3].cpu().numpy().copy();a=last-env.env_origins[0].cpu().numpy();b=here-env.env_origins[0].cpu().numpy();a[2]=b[2]=.03
                    env.gym.add_lines(env.viewer,env.envs[0],1,np.array([a,b],np.float32),np.array([[1.,.2,.2]],np.float32));last=here
                if k%250==0:
                    xy,yaw,_=session.truth_pose();cross=float((xy[0]*normal).sum())
                    print('t=%.1fs true_cross=%+.3fm heading=%+.2fdeg corrected_cmd=%s'%(k*env.dt,cross,float(yaw[0])*180/np.pi,telemetry['command'][0].tolist()),flush=True)
                if bool(done.any()):
                    print('Rollout reset; demonstration stopped rather than hiding a failed segment.',flush=True);break
        finally:session.close()


if __name__=='__main__':main()
