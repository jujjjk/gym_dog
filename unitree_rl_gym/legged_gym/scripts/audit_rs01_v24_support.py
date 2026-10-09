"""Controlled, separate force/mass/COM probes; not a hardware safety certificate."""
import argparse
import json
import sys
import types
from pathlib import Path
import isaacgym
import torch
from legged_gym.envs.base import legged_robot
from legged_gym.utils import get_args, task_registry
from evaluate_rs01_go2_omni import _set_nominal_eval_cfg


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    own,rest=p.parse_known_args();sys.argv=[sys.argv[0]]+rest
    args=get_args()
    if args.task not in ('rs01_omni_v24_support10','rs01_omni_v24_support11','rs01_omni_v25_cadence20','rs01_omni_v25_cadence22'):
        raise ValueError('V24/V25 task required')
    profiles=[('nominal',{},None),('push_x',{},(10,0,0)),('push_minus_x',{},(-10,0,0)),
              ('push_y',{},(0,10,0)),('push_minus_y',{},(0,-10,0)),
              ('mass_plus_0.5',{'mass_delta_kg':.5},None),
              ('com_y_plus_0.01',{'com_offset_m':(0,.01,0)},None),
              ('com_y_minus_0.01',{'com_offset_m':(0,-.01,0)},None)]
    commands=((.2,0,0),(0,.2,0),(-.2,-.08,-.25))
    cases=[(name,props,force,cmd) for name,props,force in profiles for cmd in commands]
    cfg,train=task_registry.get_cfgs(args.task)
    _set_nominal_eval_cfg(cfg,args.duration_s,len(cases)*args.eval_envs)
    cfg.sensor.clean_fraction=0.;cfg.sensor.robust_fraction=0.
    cfg.disturbance.audit_profiles=[c[1] for c in cases for _ in range(args.eval_envs)]
    if args.headless:legged_robot.time=types.SimpleNamespace(sleep=lambda s:None)
    env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
    train.runner.resume=True
    runner,_=task_registry.make_alg_runner(env=env,name=args.task,args=args,train_cfg=train,log_root=None)
    policy=runner.get_inference_policy(device=env.device)
    command=torch.tensor([c[3] for c in cases],device=env.device).repeat_interleave(args.eval_envs,0)
    resets=torch.zeros(env.num_envs,device=env.device);rows=[]
    if args.duration_s<4:raise ValueError('Probe needs at least 4 seconds')
    with torch.no_grad():
        for step in range(round(args.duration_s/env.dt)):
            if step==round(3/env.dt):
                for j,c in enumerate(cases):
                    if c[2] is not None:
                        env.queue_push(range(j*args.eval_envs,(j+1)*args.eval_envs),c[2],.2)
            env.set_evaluation_command(command,1.);env.compute_observations()
            obs=env.get_observations().clone();target=env.sensor_effective_command.clone()
            torch.testing.assert_close(obs[:,3:6],env.sensor.gyro*env.obs_scales.ang_vel)
            torch.testing.assert_close(obs[:,6:9],env.sensor.gravity)
            torch.testing.assert_close(obs[:,9:12],target*env.commands_scale)
            torch.testing.assert_close(obs[:,12:24],(env.sensor.q-env.default_dof_pos)*env.obs_scales.dof_pos)
            torch.testing.assert_close(obs[:,24:36],env.sensor.dq*env.obs_scales.dof_vel)
            new,_,reward,done,_=env.step(policy(obs))
            torch.testing.assert_close(env.direction_reward_target,target,rtol=0,atol=0)
            if not bool(torch.isfinite(new).all() & torch.isfinite(reward).all()):raise RuntimeError('Nonfinite')
            resets+=done
            if step>=round(2/env.dt):
                rows.append(tuple(x.clone() for x in (env.rpy[:,:2],env.v24_signed_y,
                    env.v24_support_contact,env.v24_touchdown,env.base_lin_vel[:,:2]-target[:,:2],
                    env.raw_pd_torques.abs().amax(1),env.v24_region_cost)))
    values=[torch.stack([r[i] for r in rows]) for i in range(7)];results=[]
    for j,c in enumerate(cases):
        sl=slice(j*args.eval_envs,(j+1)*args.eval_envs)
        rp,y,contact,landing,error,raw,cost=[x[:,sl] for x in values]
        inside=y<cfg.rewards.support_inner_m
        results.append(dict(profile=c[0],command=c[3],resets=int(resets[sl].sum()),
            roll_pitch_rms_deg=(rp.square().mean((0,1)).sqrt()*180/torch.pi).tolist(),
            roll_pitch_max_deg=(rp.abs().amax((0,1))*180/torch.pi).tolist(),
            support_inside_ratio=float((inside&contact).sum()/contact.sum().clamp(min=1)),
            touchdown_inside_ratio=float((inside&landing).sum()/landing.sum().clamp(min=1)),
            support_crossed_ratio=float(((y<=0)&contact).sum()/contact.sum().clamp(min=1)),
            support_signed_y_p05_m=float(torch.quantile(y[contact],.05)) if contact.any() else None,
            flight_ratio=float((contact.sum(2)==0).float().mean()),
            planar_rmse=float(error.square().mean().sqrt()),raw_peak_nm=float(raw.max()),
            region_cost_mean=float(cost.mean()),applied_impulse_ns=env.v24_push_impulse[sl].tolist()))
    result=dict(task=args.task,checkpoint=args.checkpoint,load_run=args.load_run,seed=args.seed,
        duration_s=args.duration_s,envs_per_case=args.eval_envs,finite=True,actor_reward_aligned=True,
        force_start_s=3.,force_duration_s=.2,force_frame='ENV_SPACE',
        note='Separate probes, not paired identical sensor noise; no recovery-time or hardware certification.',
        verified_body_properties=env.v24_verified_properties,cases=results)
    out=Path(own.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+'\n')
    print('AUDIT',out,'resets',int(resets.sum()),'finite=True')


if __name__=='__main__':main()
