"""Read-only policy ablations: same weights, real unchanged RS01 actuator."""
import argparse
import json
import sys
import types
from pathlib import Path
import isaacgym
import torch
from legged_gym.envs.base import legged_robot
from legged_gym.utils import get_args,task_registry
from legged_gym.envs.rs01_omni_v2.rs01_omni_v20_env import map_hip_targets
from legged_gym.envs.rs01_omni_v2.rs01_omni_v26_env import landing_bound_weight
from evaluate_rs01_go2_omni import _set_nominal_eval_cfg


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    own,rest=p.parse_known_args();sys.argv=[sys.argv[0]]+rest;args=get_args()
    variants=['compress','old','restore_FL','restore_FR','restore_RL','restore_RR','cap','stance_free_compress','stance_free_cap']
    commands=[(-.3,.15,0),(-.3,-.15,0)]
    cases=[(v,c) for v in variants for c in commands]
    cfg,train=task_registry.get_cfgs(args.task)
    _set_nominal_eval_cfg(cfg,args.duration_s,len(cases)*args.eval_envs)
    cfg.sensor.clean_fraction=0.;cfg.sensor.robust_fraction=0.
    if args.headless:legged_robot.time=types.SimpleNamespace(sleep=lambda s:None)
    env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
    train.runner.resume=True
    runner,_=task_registry.make_alg_runner(env=env,name=args.task,args=args,train_cfg=train,log_root=None)
    policy=runner.get_inference_policy(device=env.device)
    hips=[env.dof_names.index(x+'_hip_joint') for x in ('FL','FR','RL','RR')]
    feet=[env.foot_slot_by_leg[x] for x in ('FL','FR','RL','RR')]
    sides=torch.tensor([1.,-1.,1.,-1.],device=env.device)
    ranges=env.rs01_action_scale_rad[hips]*cfg.normalization.clip_actions
    original=env._project_rs01_policy_target
    def project(self,target):
        result=original(target)
        old=map_hip_targets(target,self.commands,hips,sides,ranges,cfg.control.straight_inward_target_rad)
        weight=landing_bound_weight(self._gait_phase(),self.diagonal_a_contact_mask,cfg.rewards.gait_stance_ratio)[:,feet]
        limit=cfg.control.straight_inward_target_rad+(cfg.control.omni_inward_target_rad-cfg.control.straight_inward_target_rad)*weight
        capped=target.clone();capped[:,hips]=torch.maximum(target[:,hips]*sides,-limit)*sides
        free_limit=ranges[None,:]+(cfg.control.omni_inward_target_rad-ranges[None,:])*weight
        free_cap=target.clone();free_cap[:,hips]=torch.maximum(target[:,hips]*sides,-free_limit)*sides
        free_compress=target.clone();free_compress[:,hips]=torch.where(target[:,hips]*sides<0.,target[:,hips]*free_limit/ranges,target[:,hips])
        for i,(variant,_) in enumerate(cases):
            sl=slice(i*args.eval_envs,(i+1)*args.eval_envs)
            if variant=='old':result[sl]=old[sl]
            elif variant=='cap':result[sl]=capped[sl]
            elif variant=='stance_free_cap':result[sl]=free_cap[sl]
            elif variant=='stance_free_compress':result[sl]=free_compress[sl]
            elif variant.startswith('restore_'):
                h=hips[('FL','FR','RL','RR').index(variant[-2:])];result[sl,h]=old[sl,h]
        self.audit_projection=(result[:,hips]-target[:,hips]).abs().detach()
        return result
    env._project_rs01_policy_target=types.MethodType(project,env)
    command=torch.tensor([c for _,c in cases],device=env.device).repeat_interleave(args.eval_envs,0)
    resets=torch.zeros(env.num_envs,device=env.device);rows=[]
    with torch.no_grad():
        for step in range(round(args.duration_s/env.dt)):
            env.set_evaluation_command(command,1.);env.compute_observations()
            obs,_,rew,done,_=env.step(policy(env.get_observations()))
            if not bool(torch.isfinite(obs).all() & torch.isfinite(rew).all()):raise RuntimeError('Nonfinite')
            resets+=done
            if step>=round(2/env.dt):
                contact=env.v24_support_contact
                rows.append(tuple(x.clone() for x in (env.rpy[:,0],env.base_lin_vel[:,:2],env.base_ang_vel[:,2],
                    env.v24_signed_y,contact,env.v24_support_force,env.audit_projection,
                    env.raw_pd_torques[:,hips],env.torques[:,hips],env.dof_pos[:,hips],env.v24_touchdown)))
    values=[torch.stack([r[i] for r in rows]) for i in range(11)];results=[]
    for i,(variant,cmd) in enumerate(cases):
        sl=slice(i*args.eval_envs,(i+1)*args.eval_envs)
        roll,v,w,y,contact,force,projection,raw,applied,q,touch=[x[:,sl] for x in values]
        results.append(dict(variant=variant,command=cmd,resets=int(resets[sl].sum()),
            roll_rms_deg=float(roll.square().mean().sqrt()*180/torch.pi),
            mean_vx_vy_wz=[*v.mean((0,1)).tolist(),float(w.mean())],
            support_inside_10cm_ratio=float(((y<.10)&contact).sum()/contact.sum().clamp(min=1)),
            foot_force_mean_n=force.mean((0,1)).tolist(),hip_projection_mean_deg=(projection.mean((0,1))*180/torch.pi).tolist(),
            hip_actual_mean_deg=(q.mean((0,1))*180/torch.pi).tolist(),
            hip_raw_peak_nm=raw.abs().amax((0,1)).tolist(),hip_applied_peak_nm=applied.abs().amax((0,1)).tolist(),
            touchdown_hz=(touch.sum(0).float().mean(0)/(len(rows)*env.dt)).tolist()))
    result=dict(task=args.task,load_run=args.load_run,checkpoint=args.checkpoint,seed=args.seed,
        duration_s=args.duration_s,envs_per_case=args.eval_envs,order=['FL','FR','RL','RR'],finite=True,
        note='Independent noise/initial states per environment; two-seed group comparison, not identical paired trajectories.',cases=results)
    out=Path(own.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
    print('Saved',out)


if __name__=='__main__':main()
