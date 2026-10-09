"""Fixed-command physical audit; does not train or infer gait quality from reward."""
import json
from pathlib import Path
import time
import isaacgym
import numpy as np
import torch
from isaacgym.torch_utils import quat_rotate_inverse
from evaluate_rs01_go2_omni import _set_nominal_eval_cfg
from legged_gym.utils import get_args,task_registry


def sustained_events(mask,minimum=2):
    edges=np.diff(np.r_[False,mask,False].astype(int))
    starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)
    return starts[(ends-starts)>=minimum].tolist()


def main():
    args=get_args();cfg,train=task_registry.get_cfgs(args.task)
    count=len(cfg.amp.commands);n=count*args.eval_envs
    if args.num_envs is not None and args.num_envs!=n:raise ValueError('num_envs must equal command count * eval_envs')
    _set_nominal_eval_cfg(cfg,args.duration_s,n)
    if not cfg.env.test:raise RuntimeError('Audit must disable reference-state initialization')
    env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
    train.runner.resume=True
    runner,_=task_registry.make_alg_runner(env=env,name=args.task,args=args,train_cfg=train,log_root=None)
    policy=runner.get_inference_policy(device=env.device)
    cmd=torch.tensor(cfg.amp.commands,device=env.device).repeat_interleave(args.eval_envs,0)
    ids=[env.foot_slot_by_leg[l] for l in ['FL','FR','RL','RR']]
    samples=[];resets=torch.zeros(n,device=env.device);finite=True
    with torch.no_grad():
        for step in range(round(args.duration_s/env.dt)):
            env.set_evaluation_command(cmd,1.);env.compute_observations()
            raw_action=policy(env.get_observations())
            obs,_,reward,done,_=env.step(raw_action);resets+=done
            finite &= bool(torch.isfinite(obs).all() and torch.isfinite(reward).all() and torch.isfinite(raw_action).all())
            if step<round(2/env.dt):continue
            c=env.get_foot_contact_mask()[:,ids]
            h=env.feet_pos[:,ids,2]-env.env_origins[:,None,2]-cfg.rewards.foot_collision_radius_m
            relative=env.feet_pos[:,ids]-env.root_states[:,None,:3]
            feet=quat_rotate_inverse(env.base_quat[:,None,:].expand(-1,4,-1).reshape(-1,4),relative.reshape(-1,3)).reshape(-1,4,3)
            samples.append(tuple(x.cpu().numpy().copy() for x in (env.base_lin_vel,env.base_ang_vel[:,2],
                env.rpy[:,:2],h,c,env.raw_pd_torques,feet[:,:,1]*feet.new_tensor([1,-1,1,-1]),env.torques,raw_action,
                env.dof_pos,env.dof_vel,env._gait_phase())))
    stacked=[np.stack([r[i] for r in samples]) for i in range(12)];cases=[]
    names=['march','forward','backward','left','right','turn_left','turn_right','forward_left','forward_right','combined']
    for j,name in enumerate(names):
        sl=slice(j*args.eval_envs,(j+1)*args.eval_envs)
        v,w,r,h,c,raw,y,applied,action=[x[:,sl] for x in stacked[:9]];counts=c.sum(-1)
        diag=(c==[1,0,0,1]).all(-1)|(c==[0,1,1,0]).all(-1)
        alternating=[];liftoffs=[]
        for robot in range(args.eval_envs):
            swing=(h[:,robot]>.008)&~c[:,robot]
            liftoffs.append([len(sustained_events(swing[:,leg])) for leg in range(4)])
            a=sustained_events(swing[:,0]&swing[:,3]);b=sustained_events(swing[:,1]&swing[:,2])
            events=sorted([(t,0) for t in a]+[(t,1) for t in b])
            alternating.append(sum(events[i][1]!=events[i-1][1] for i in range(1,len(events))))
        cases.append(dict(case=name,command=cfg.amp.commands[j],resets=int(resets[sl].sum()),
            mean_velocity=[float(v[:,:,0].mean()),float(v[:,:,1].mean()),float(w.mean())],
            roll_pitch_rms_deg=(np.sqrt((r*r).mean((0,1)))*180/np.pi).tolist(),
            foot_height_p95_mm=(np.quantile(h.reshape(-1,4),.95,axis=0)*1000).tolist(),
            liftoffs_per_robot=liftoffs,diagonal_alternations_per_robot=alternating,
            four_contact=float((counts==4).mean()),illegal_support=float((~(diag|(counts==4))).mean()),
            three_contact=float((counts==3).mean()),
            unsafe_support=float((~(diag|(counts>=3))).mean()),
            flight=float((counts==0).mean()),inner_support=float(((y<.10)&c).sum()/max(1,c.sum())),
            raw_torque_peak=float(abs(raw).max()),applied_torque_peak=float(abs(applied).max()),
            joint_excursion_p95_p05_rad=(np.quantile(stacked[9][:,sl],.95,axis=0)-np.quantile(stacked[9][:,sl],.05,axis=0)).mean(0).tolist(),
            raw_action_absmax=float(abs(action).max()),raw_action_clipped_fraction=float((abs(action)>cfg.normalization.clip_actions).mean())))
    f=cases[1]
    # Initial-learning gate, NOT deployment/omnidirectional acceptance.
    passed=bool(finite and f['resets']==0 and f['flight']==0 and f['illegal_support']<.10
        and f['mean_velocity'][0]>=.5*f['command'][0]
        and min(f['foot_height_p95_mm'])>=10 and min(f['diagonal_alternations_per_robot'])>=4
        and min(min(x) for x in f['liftoffs_per_robot'])>=3 and f['raw_torque_peak']<17)
    out=Path('/home/nszb/gym/artifacts/rs01_amp/core_audit');out.mkdir(parents=True,exist_ok=True)
    tag=args.task+'_'+str(args.checkpoint)+'_'+str(time.time_ns())
    result=dict(task=args.task,run=args.load_run,checkpoint=args.checkpoint,seed=args.seed,
        duration_s=args.duration_s,skip_s=2,envs_per_command=args.eval_envs,finite=finite,
        reference_initialization_enabled=False,
        forward_initial_learning_gate=passed,not_deployment_approval=True,cases=cases)
    (out/(tag+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(out/(tag+'.npz'),velocity=stacked[0],yaw_rate=stacked[1],rpy=stacked[2],
        height=stacked[3],contact=stacked[4],raw_torque=stacked[5],signed_foot_y=stacked[6],applied_torque=stacked[7],raw_action=stacked[8],
        joint_pos=stacked[9],joint_vel=stacked[10],phase=stacked[11])
    print('RESULT',out/(tag+'.json'),json.dumps(result),flush=True)


if __name__=='__main__':main()
