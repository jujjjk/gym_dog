"""Read-only rollout: body-frame foot placement and leg coordination."""
import sys
import os
import json
import types
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import isaacgym  # before torch
import torch
from isaacgym.torch_utils import quat_rotate_inverse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'unitree_rl_gym/legged_gym/scripts'))
from evaluate_rs01_go2_omni import _set_nominal_eval_cfg
from legged_gym.envs.base import legged_robot
from legged_gym.utils import get_args, task_registry

LEGS = ('FL','FR','RL','RR')


def fk(tree, angles):
    transforms = {'Trunk': np.eye(4)}
    pending = list(tree.findall('joint'))
    def rotation(axis, angle):
        axis=np.asarray(axis,dtype=float); axis/=np.linalg.norm(axis)
        x,y,z=axis; K=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
        return np.eye(3)+np.sin(angle)*K+(1-np.cos(angle))*(K@K)
    while pending:
        progress=False
        for j in pending[:]:
            parent=j.find('parent').get('link')
            if parent not in transforms: continue
            origin=j.find('origin'); t=np.eye(4)
            t[:3,3]=np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
            r,p,y=np.fromstring(origin.get('rpy','0 0 0'),sep=' ')
            t[:3,:3]=rotation([0,0,1],y)@rotation([0,1,0],p)@rotation([1,0,0],r)
            if j.get('type') in ('revolute','continuous'):
                t[:3,:3] = t[:3,:3] @ rotation(np.fromstring(j.find('axis').get('xyz'),sep=' '),angles[j.get('name')])
            transforms[j.find('child').get('link')]=transforms[parent]@t
            pending.remove(j); progress=True
        if not progress: raise ValueError('Disconnected URDF')
    return {k:v[:3,3] for k,v in transforms.items()}


def main():
    args=get_args()
    if not args.headless: raise ValueError('Headless diagnostic only')
    legged_robot.time=types.SimpleNamespace(sleep=lambda seconds:None)
    cfg, train=task_registry.get_cfgs(args.task)
    commands=((0.,0.,0.),(.2,0.,0.),(-.2,0.,0.))
    _set_nominal_eval_cfg(cfg,args.duration_s,3*args.eval_envs)
    env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
    train.runner.resume=True
    runner,_=task_registry.make_alg_runner(env=env,name=args.task,args=args,train_cfg=train,log_root=None)
    policy=runner.get_inference_policy(device=env.device)
    slots=[env.foot_slot_by_leg[x] for x in LEGS]
    joints=torch.tensor([[env.dof_names.index(x+'_'+j+'_joint') for j in ('hip','thigh','calf')] for x in LEGS],device=env.device)
    names=env.gym.get_actor_rigid_body_names(env.envs[0],env.actor_handles[0])
    knee_ids=[names.index(x+'_calf_joint') for x in LEGS]
    urdf=ROOT/'dog_urdf/urdf/dog_rs01.urdf'; tree=ET.parse(urdf).getroot()
    default=fk(tree,dict(zip(env.dof_names,env.default_dof_pos[0].cpu().tolist())))
    hips=np.array([default[x+'_hip_joint'] for x in LEGS])
    cmd=torch.tensor(commands,device=env.device).repeat_interleave(args.eval_envs,0)
    def body_frame(p):
        n=p.shape[1]; q=env.base_quat[:,None,:].expand(-1,n,-1).reshape(-1,4)
        return quat_rotate_inverse(q,(p-env.root_states[:,None,:3]).reshape(-1,3)).reshape(-1,n,3)
    records=[]; resets=np.zeros(env.num_envs,dtype=int); finite=True; fk_errors=[]
    with torch.no_grad():
        for step in range(round(args.duration_s/env.dt)):
            env.set_evaluation_command(cmd,1.);env.compute_observations()
            action=policy(env.get_observations())
            obs,_,reward,done,_=env.step(action)
            finite &= bool(torch.isfinite(obs).all() & torch.isfinite(reward).all())
            resets+=done.cpu().numpy().astype(int)
            if not finite: raise RuntimeError('Nonfinite rollout')
            if step<round(6./env.dt):continue
            feet=body_frame(env.feet_pos[:,slots]); knees=body_frame(env.rigid_body_states_view[:,knee_ids,:3])
            if not fk_errors:
                actual=fk(tree,dict(zip(env.dof_names,env.dof_pos[0].cpu().tolist())))
                expected=np.array([actual[names[int(env.feet_indices[s])]] for s in slots])
                fk_errors.append(float(np.max(np.linalg.norm(expected-feet[0].cpu().numpy(),axis=1))))
                if fk_errors[0]>.003: raise RuntimeError('FK/state frame mismatch')
            records.append([x.cpu().numpy().copy() for x in (
                feet,knees,env.dof_pos[:,joints],env.get_foot_contact_mask()[:,slots],
                env.contact_forces[:,env.feet_indices[slots],2],env._gait_phase(),
                action[:,joints],env.rs01_limited_position_target_rad[:,joints],
                env.rs01_response_target_rad[:,joints],env.rs01_target_rate_rad_s[:,joints],
                env.raw_pd_torques[:,joints],env.base_lin_vel,env.rpy,
                getattr(env,'v19_placement_cost',torch.zeros_like(reward))*(env._placement_reward_weight() if hasattr(env,'_placement_reward_weight') else 0.)*env._walking_command_gate()*env.reward_scales['phase_two_contact_quality'],
                env.v11_tracking_reward*env.reward_scales['tracking_command_velocity'])])
    keys=('feet','knees','q','contact','force','phase','action','target','response','target_rate','raw_torque','velocity','rpy','placement_deduction','tracking_reward')
    data={k:np.stack([row[i] for row in records]) for i,k in enumerate(keys)}
    result=dict(task=args.task,run=args.load_run,checkpoint=args.checkpoint,seed=args.seed,
        duration_s=args.duration_s,skip_s=6,envs_per_case=args.eval_envs,dt_s=env.dt,
        leg_order=LEGS,dof_names=env.dof_names,finite=finite,fk_max_error_m=fk_errors[0],
        urdf_sha256=hashlib.sha256(urdf.read_bytes()).hexdigest(),
        default_foot_xyz_m=[default[names[int(env.feet_indices[s])]].tolist() for s in slots],cases=[])
    side=np.array([1.,-1.,1.,-1.])
    for j,name in enumerate(('march','forward','backward')):
        sl=slice(j*args.eval_envs,(j+1)*args.eval_envs)
        d={k:v[:,sl] for k,v in data.items()}; f=d['feet'];k=d['knees'];q=d['q'];ct=d['contact']
        cycles=[]
        for e in range(args.eval_envs):
            edges=np.where(np.diff(d['phase'][:,e])<-.5)[0]+1
            for a,b in zip(edges[:-1],edges[1:]):
                cycles.append(dict(env=e,front_dx_mm=float((f[a:b,e,0,0]-f[a:b,e,1,0]).mean()*1000),
                    foot_x_amplitude_mm=((f[a:b,e,:,0].max(0)-f[a:b,e,:,0].min(0))*1000).tolist()))
        inward=np.degrees(np.arctan2(side*(k[:,:,:,1]-f[:,:,:,1]),k[:,:,:,2]-f[:,:,:,2]))
        width=np.stack((f[:,:,0,1]-f[:,:,1,1],f[:,:,2,1]-f[:,:,3,1]),axis=-1)
        lower=env.guard_joint_lower[joints].cpu().numpy();upper=env.guard_joint_upper[joints].cpu().numpy()
        near=(q<lower+.02*(upper-lower)) | (q>upper-.02*(upper-lower))
        row=dict(case=name,command=commands[j],resets=int(resets[sl].sum()),
            mean_placement_deduction_per_step=float(d['placement_deduction'].mean()),
            mean_tracking_reward_per_step=float(d['tracking_reward'].mean()),
            foot_relative_hip_mean_xyz_mm=((f-hips).mean((0,1))*1000).tolist(),
            front_dx_per_env_mm=((f[:,:,0,0]-f[:,:,1,0]).mean(0)*1000).tolist(),
            front_dx_p05_p50_p95_mm=(np.quantile(f[:,:,0,0]-f[:,:,1,0],[.05,.5,.95])*1000).tolist(),
            front_left_ahead_fraction=float(np.mean(f[:,:,0,0]>f[:,:,1,0])),
            completed_cycles=cycles, contact_fraction=ct.mean((0,1)).tolist(),
            mean_vertical_force_n=d['force'].clip(0).mean((0,1)).tolist(),
            mean_q_deg=np.degrees(q.mean((0,1))).tolist(),
            mean_inward_hip_deg=(-side*np.degrees(q[:,:,:,0].mean((0,1)))).tolist(),
            lower_leg_inward_angle_deg=inward.mean((0,1)).tolist(),
            width_mean_front_rear_mm=(width.mean((0,1))*1000).tolist(),
            width_below_200mm_fraction=(width<.2).mean((0,1)).tolist(),
            raw_action_mean=d['action'].mean((0,1)).tolist(),
            action_clip_fraction=(np.abs(d['action'])>=1.).mean((0,1)).tolist(),
            near_joint_limit_fraction=near.mean((0,1)).tolist(),
            target_rate_limit_fraction=(np.abs(d['target_rate'])>=.99*env.rs01_target_rate_limit_rad_s[joints].cpu().numpy()).mean((0,1)).tolist(),
            mean_target_minus_q_deg=np.degrees((d['target']-q).mean((0,1))).tolist(),
            mean_response_minus_q_deg=np.degrees((d['response']-q).mean((0,1))).tolist(),
            velocity_mean=d['velocity'].mean((0,1)).tolist())
        result['cases'].append(row)
    output_dir=Path(os.environ.get('RS01_AUDIT_OUTPUT_DIR',str(Path(__file__).parent)))
    output_dir.mkdir(parents=True,exist_ok=True)
    stem=output_dir/(args.task+'_'+str(args.checkpoint)+'_'+str(args.seed)+'_posture')
    np.savez_compressed(str(stem)+'.npz',**data)
    Path(str(stem)+'.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Saved',str(stem)+'.json','finite',finite,'resets',int(resets.sum()),flush=True)


if __name__=='__main__':main()
